from collections import deque
import copy
import json
import logging

import cv2
import numpy as np
from openpi_client.base_policy import BasePolicy
from openpi_client.image_tools import resize_with_pad
import torch

logger = logging.getLogger("policy")
logger.setLevel(20)  # info

RESIZE_SIZE = 224
DESPTH_RESIZE_SIZE = 720

SKILL_PROMPT = """
You are a robot that is trying to complete the global task: {task_prompt}

The skills are:
{skill_prompts}

What's the next skill to perform? Only respond with a single skill name.
"""


class B1KPolicyWrapper:
    # Camera keys that need frame history for video memory (K>1).
    _HISTORY_CAMERA_KEYS = (
        "observation/egocentric_camera",
        "observation/wrist_image_left",
        "observation/wrist_image_right",
    )
    _B1K_FPS = 30.0

    def __init__(
        self,
        policy: BasePolicy,
        task_name: str = "turning_on_radio",
        control_mode: str = "temporal_ensemble",
        max_len: int = 32,  # receeding horizon | receeding temporal mode
        action_horizon: int = 5,  # temporal ensemble mode | receeding temporal mode
        temporal_ensemble_max: int = 3,  # receeding temporal mode
        fine_grained_level: int = 0,
        video_memory_frames: int = 1,
        video_memory_stride_s: float = 1.0,
    ) -> None:
        self.policy = policy
        self.task_name = task_name
        self.video_memory_frames = video_memory_frames
        self.video_memory_stride_s = video_memory_stride_s
        self.history_stride = 1
        # Frame history buffer: stores up to K-1 past frames per camera.
        self._frame_history: dict[str, deque] = {
            k: deque(maxlen=max(video_memory_frames - 1, 0))
            for k in self._HISTORY_CAMERA_KEYS
        }

        # load the task name from the metadata
        metadata = json.load(open("scripts/task_mapping.json"))
        self.task_prompt = metadata[task_name].get("task")
        self.subtask_prompts = metadata[task_name].get("subtask")
        self.skill_prompts = metadata[task_name].get("skill")

        self.control_mode = control_mode
        self.action_queue = deque(maxlen=action_horizon)
        self.last_action = {"actions": np.zeros((action_horizon, 23), dtype=np.float64)}
        self.action_horizon = action_horizon

        self.replan_interval = action_horizon  # K: replan every 10 steps
        self.max_len = max_len  # how long the policy sequences are
        self.temporal_ensemble_max = temporal_ensemble_max  # max number of sequences to ensemble
        self.step_counter = 0

        self.fine_grained_level = fine_grained_level
        if self.fine_grained_level > 0:
            from openpi.shared.client import Client

            self.reasoner = Client(model="/workspace/model")
        else:
            self.reasoner = None

        self.log_config()

    def log_config(self):
        logger.info(f"{self.task_name=}")
        logger.info(f"{self.control_mode=}")
        logger.info(f"{self.max_len=}")
        logger.info(f"{self.action_horizon=}")
        logger.info(f"{self.temporal_ensemble_max=}")
        logger.info(f"{self.replan_interval=}")
        logger.info(f"{self.fine_grained_level=}")
        logger.info(f"{self.video_memory_frames=}")
        logger.info(f"{self.video_memory_stride_s=}")
        logger.info(f"{self.history_stride=}")
        logger.info(f"{self.step_counter=}")
        logger.info(f"{self.action_queue=}")
        logger.info(f"{self.task_prompt=}")
        logger.info(f"{self.subtask_prompts=}")
        logger.info(f"{self.skill_prompts=}")

    def reset(self):
        self.action_queue = deque(maxlen=self.action_horizon)
        self.last_action = {"actions": np.zeros((self.action_horizon, 23), dtype=np.float64)}
        self.step_counter = 0
        for k in self._frame_history:
            self._frame_history[k].clear()
        if self.reasoner:
            self.reasoner.reset()

    def _resolve_prompt(self, obs: dict) -> str:
        """Prefer the evaluator-provided prompt; fall back to the task prompt."""
        prompt = obs.get("prompt")
        if prompt is None:
            return self.task_prompt
        prompt = str(prompt).strip()
        return prompt or self.task_prompt

    def _maybe_reason_about_subtask(self, batch: dict) -> None:
        """Best-effort subtask reasoning. Fall back to the current prompt if the reasoner is unavailable."""
        if self.fine_grained_level <= 0 or self.reasoner is None:
            return
        try:
            reasoner_response = self.reasoner.generate_subtask(
                high_level_task=self.task_prompt,
                multi_modals=[batch["observation/egocentric_camera"]],
            )
            logger.info(f"* {reasoner_response}")
            batch["prompt"] = reasoner_response
        except Exception as exc:
            logger.warning("Reasoner unavailable, falling back to prompt: %s", exc)

    def _attach_history(self, batch: dict) -> dict:
        """Attach frame history to batch and update buffers for K>1 video memory."""
        if self.video_memory_frames <= 1:
            return batch
        for key in self._HISTORY_CAMERA_KEYS:
            if key in batch:
                current_frame = batch[key]
                history = list(self._frame_history[key])
                sampled = []
                valid = []
                missing = max(self.video_memory_frames - 1 - len(history), 0)
                if missing > 0:
                    pad_frame = history[0] if history else current_frame
                    sampled.extend([pad_frame] * missing)
                    valid.extend([False] * missing)
                sampled.extend(history[-(self.video_memory_frames - 1) :])
                valid.extend([True] * min(len(history), self.video_memory_frames - 1))
                batch[f"{key}_history"] = sampled
                batch[f"{key}_history_valid"] = valid
                self._frame_history[key].append(current_frame.copy())
        return batch

    def process_obs(self, obs: dict) -> dict:
        """
        Process the observation dictionary to match the expected input format for the model.
        """
        prop_state = obs["robot_r1::proprio"][None]
        img_obs = np.stack(
            [
                resize_with_pad(
                    obs["robot_r1::robot_r1:zed_link:Camera:0::rgb"][None, ..., :3],
                    RESIZE_SIZE,
                    RESIZE_SIZE,
                ),
                resize_with_pad(
                    obs["robot_r1::robot_r1:left_realsense_link:Camera:0::rgb"][None, ..., :3],
                    RESIZE_SIZE,
                    RESIZE_SIZE,
                ),
                resize_with_pad(
                    obs["robot_r1::robot_r1:right_realsense_link:Camera:0::rgb"][None, ..., :3],
                    RESIZE_SIZE,
                    RESIZE_SIZE,
                ),
            ],
            axis=1,
        )

        if "robot_r1::robot_r1:right_realsense_link:Camera:0::instance_seg" in obs:
            pass  # TODO: add instance segmentation

        processed_obs = {
            "observation": img_obs,  # Shape: (1, 3, H, W, C)
            "proprio": prop_state,
        }

        if "prompt" in obs:
            processed_obs["prompt"] = obs["prompt"]

        if "robot_r1::robot_r1:zed_link:Camera:0::depth_linear" in obs:
            depth_obs = obs["robot_r1::robot_r1:zed_link:Camera:0::depth_linear"]
            depth_obs = cv2.resize(depth_obs, (DESPTH_RESIZE_SIZE, DESPTH_RESIZE_SIZE), interpolation=cv2.INTER_LINEAR)
            processed_obs["observation/egocentric_depth"] = depth_obs[None]

        # if "robot_r1::robot_r1:left_realsense_link:Camera:0::depth_linear" in obs:
        #     depth_obs = obs["robot_r1::robot_r1:left_realsense_link:Camera:0::depth_linear"][None]
        #     processed_obs["observation/wrist_depth_left"] = depth_obs

        # if "robot_r1::robot_r1:right_realsense_link:Camera:0::depth_linear" in obs:
        #     depth_obs = obs["robot_r1::robot_r1:right_realsense_link:Camera:0::depth_linear"][None]
        #     processed_obs["observation/wrist_depth_right"] = depth_obs

        return processed_obs

    def act_receeding_temporal(self, input_obs):
        # Step 1: check if we should re-run policy
        if self.step_counter % self.replan_interval == 0:
            nbatch = copy.deepcopy(input_obs)
            if nbatch["observation"].shape[-1] != 3:
                # make B, num_cameras, H, W, C  from B, num_cameras, C, H, W
                # permute if pytorch
                nbatch["observation"] = np.transpose(nbatch["observation"], (0, 1, 3, 4, 2))

            # nbatch["proprio"] is B, 16, where B=1
            joint_positions = nbatch["proprio"][0]
            prompt = self._resolve_prompt(nbatch)
            batch = {
                "observation/egocentric_camera": nbatch["observation"][0, 0],
                "observation/wrist_image_left": nbatch["observation"][0, 1],
                "observation/wrist_image_right": nbatch["observation"][0, 2],
                "observation/state": joint_positions,
                "prompt": prompt,
            }

            self._maybe_reason_about_subtask(batch)

            if "observation/egocentric_depth" in nbatch:
                batch["observation/egocentric_depth"] = nbatch["observation/egocentric_depth"][0]

            batch = self._attach_history(batch)

            try:
                action = self.policy.infer(batch)
                self.last_action = action
            except Exception as e:
                action = self.last_action
                logger.info(
                    f"Error in action prediction at step {self.step_counter}, {joint_positions.shape=}, using last action: {e}"
                )

            target_joint_positions = action["actions"].copy()

            # Add this sequence to action queue
            new_seq = deque([a for a in target_joint_positions[: self.max_len]])
            self.action_queue.append(new_seq)

            # Optional: limit memory
            while len(self.action_queue) > self.temporal_ensemble_max:
                self.action_queue.popleft()

        # Step 2: Smooth across current step from all stored sequences
        if len(self.action_queue) == 0:
            raise ValueError("Action queue empty in receeding_temporal mode.")

        actions_current_timestep = np.empty((len(self.action_queue), self.action_queue[0][0].shape[0]))

        for i in range(len(self.action_queue)):
            actions_current_timestep[i] = self.action_queue[i].popleft()

        # Drop exhausted sequences
        self.action_queue = deque([q for q in self.action_queue if len(q) > 0])

        # Apply temporal ensemble
        k = 0.005
        exp_weights = np.exp(k * np.arange(actions_current_timestep.shape[0]))
        exp_weights = exp_weights / exp_weights.sum()

        final_action = (actions_current_timestep * exp_weights[:, None]).sum(axis=0)

        # Preserve grippers from most recent rollout
        final_action[-9] = actions_current_timestep[0, -9]
        final_action[-1] = actions_current_timestep[0, -1]
        final_action = final_action[None]

        self.step_counter += 1

        return torch.from_numpy(final_action)

    def act(self, input_obs):
        # TODO reformat data into the correct format for the model
        # TODO: communicate with justin that we are using numpy to pass the data. Also we are passing in uint8 for images
        """
        Model input expected:
            📌 Key: observation/exterior_image_1_left
            Type: ndarray
            Dtype: uint8
            Shape: (224, 224, 3)

            📌 Key: observation/exterior_image_2_left
            Type: ndarray
            Dtype: uint8
            Shape: (224, 224, 3)

            📌 Key: observation/joint_position
            Type: ndarray
            Dtype: float64
            Shape: (16,)

            📌 Key: prompt
            Type: str
            Value: do something

        Model will output:
            📌 Key: actions
            Type: ndarray
            Dtype: float64
            Shape: (10, 16)
        """
        input_obs = self.process_obs(input_obs)
        if self.control_mode == "receeding_temporal":
            return self.act_receeding_temporal(input_obs)

        if self.control_mode == "receeding_horizon":
            if len(self.action_queue) > 0:
                # pop the first action in the queue
                final_action = self.action_queue.popleft()[None]
                return torch.from_numpy(final_action)

        nbatch = copy.deepcopy(input_obs)
        if nbatch["observation"].shape[-1] != 3:
            # make B, num_cameras, H, W, C  from B, num_cameras, C, H, W
            # permute if pytorch
            nbatch["observation"] = np.transpose(nbatch["observation"], (0, 1, 3, 4, 2))

        # nbatch["proprio"] is B, 16, where B=1
        joint_positions = nbatch["proprio"][0]
        prompt = self._resolve_prompt(nbatch)
        batch = {
            "observation/egocentric_camera": nbatch["observation"][0, 0],
            "observation/wrist_image_left": nbatch["observation"][0, 1],
            "observation/wrist_image_right": nbatch["observation"][0, 2],
            "observation/state": joint_positions,
            "prompt": prompt,
        }

        if "observation/egocentric_depth" in nbatch:
            batch["observation/egocentric_depth"] = nbatch["observation/egocentric_depth"][0]

        batch = self._attach_history(batch)

        self._maybe_reason_about_subtask(batch)

        try:
            action = self.policy.infer(batch)
            self.last_action = action
        except Exception as e:
            action = self.last_action
            raise e
        # convert to absolute action and append gripper command
        # action shape: (10, 23), joint_positions shape: (23,)
        # Need to broadcast joint_positions to match action sequence length
        target_joint_positions = action["actions"].copy()
        if self.control_mode == "receeding_horizon":
            self.action_queue = deque([a for a in target_joint_positions[: self.max_len]])
            final_action = self.action_queue.popleft()[None]

        # # temporal emsemble start
        elif self.control_mode == "temporal_ensemble":
            new_actions = deque(target_joint_positions)
            self.action_queue.append(new_actions)
            actions_current_timestep = np.empty((len(self.action_queue), target_joint_positions.shape[1]))

            # k = 0.01
            k = 0.005
            for i, q in enumerate(self.action_queue):
                actions_current_timestep[i] = q.popleft()

            exp_weights = np.exp(k * np.arange(actions_current_timestep.shape[0]))
            exp_weights = exp_weights / exp_weights.sum()

            final_action = (actions_current_timestep * exp_weights[:, None]).sum(axis=0)
            final_action[-9] = target_joint_positions[0, -9]
            final_action[-1] = target_joint_positions[0, -1]
            final_action = final_action[None]
        else:
            final_action = target_joint_positions
        return torch.from_numpy(final_action)
