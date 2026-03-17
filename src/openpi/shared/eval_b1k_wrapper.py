from collections import deque
import copy
import json
import logging
import os

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
    def __init__(
        self,
        policy: BasePolicy,
        task_name: str | None = "turning_on_radio",
        # If provided, overrides any task_mapping-derived prompt.
        task_prompt_override: str | None = None,
        # If True, and incoming obs contains a prompt, use it (unless overridden).
        prompt_from_obs: bool = False,
        # Which key to read the prompt from in the incoming obs.
        prompt_key: str = "prompt",
        # Final fallback if no mapping / override / obs prompt is available.
        fallback_prompt: str | None = None,
        # If True, append `working_memory` into the final prompt as a [WM]...[/WM] block.
        # If False, ignore WM for prompt injection (ablation: no-WM-in-prompt).
        wm_in_prompt: bool = True,
        control_mode: str = "temporal_ensemble",
        max_len: int = 32,  # receeding horizon | receeding temporal mode
        action_horizon: int = 5,  # temporal ensemble mode | receeding temporal mode
        temporal_ensemble_max: int = 3,  # receeding temporal mode
        fine_grained_level: int = 0,
        # MEM: video memory parameters
        video_memory_frames: int = 1,
        video_memory_stride: int = 1,
    ) -> None:
        self.policy = policy
        self.task_name = task_name
        self.task_prompt_override = task_prompt_override
        self.prompt_from_obs = prompt_from_obs
        self.prompt_key = prompt_key
        self.fallback_prompt = fallback_prompt
        self.wm_in_prompt = wm_in_prompt

        # load the task name from the metadata
        self._task_mapping = None
        self._task_prompt_from_mapping = None
        self.subtask_prompts = None
        self.skill_prompts = None
        mapping_path = "scripts/task_mapping.json"
        if os.path.exists(mapping_path):
            try:
                self._task_mapping = json.load(open(mapping_path))
            except Exception:
                logger.exception("Failed to load %s", mapping_path)
                self._task_mapping = None

        if self._task_mapping and self.task_name and self.task_name in self._task_mapping:
            entry = self._task_mapping[self.task_name]
            self._task_prompt_from_mapping = entry.get("task")
            self.subtask_prompts = entry.get("subtask")
            self.skill_prompts = entry.get("skill")
        elif self.task_name is not None:
            logger.warning(
                "task_name '%s' not found in %s; will rely on override / obs prompt / fallback",
                self.task_name,
                mapping_path,
            )

        self.control_mode = control_mode
        self.action_queue = deque(maxlen=action_horizon)
        self.last_action = {"actions": np.zeros((action_horizon, 23), dtype=np.float64)}
        self.action_horizon = action_horizon

        self.replan_interval = action_horizon  # K: replan every 10 steps
        self.max_len = max_len  # how long the policy sequences are
        self.temporal_ensemble_max = temporal_ensemble_max  # max number of sequences to ensemble
        self.step_counter = 0

        # MEM: video memory frame buffers
        self.video_memory_frames = video_memory_frames
        self.video_memory_stride = max(1, video_memory_stride)
        self._frame_buffers: dict[str, deque] = {}
        self._frame_buffer_maxlen = (video_memory_frames - 1) * self.video_memory_stride + 1

        self.fine_grained_level = fine_grained_level
        if self.fine_grained_level > 0:
            from openpi.shared.client import Client

            self.reasoner = Client(model="/workspace/model")
        else:
            self.reasoner = None

        self.log_config()

    def _effective_task_prompt(self, input_obs: dict) -> str:
        """
        Decide which prompt to send to the model.

        Priority:
          1) task_prompt_override (CLI)
          2) prompt from incoming obs (if enabled)
          3) task_mapping.json-derived prompt (50-task mapping)
          4) fallback_prompt
          5) empty string
        """
        if self.task_prompt_override:
            return self.task_prompt_override

        if self.prompt_from_obs and isinstance(input_obs, dict):
            obs_prompt = input_obs.get(self.prompt_key)
            if isinstance(obs_prompt, str) and obs_prompt.strip():
                return obs_prompt

        if isinstance(self._task_prompt_from_mapping, str) and self._task_prompt_from_mapping.strip():
            return self._task_prompt_from_mapping

        if isinstance(self.fallback_prompt, str) and self.fallback_prompt.strip():
            return self.fallback_prompt

        return ""

    def log_config(self):
        logger.info(f"{self.task_name=}")
        logger.info(f"{self.control_mode=}")
        logger.info(f"{self.max_len=}")
        logger.info(f"{self.action_horizon=}")
        logger.info(f"{self.temporal_ensemble_max=}")
        logger.info(f"{self.replan_interval=}")
        logger.info(f"{self.fine_grained_level=}")
        logger.info(f"{self.step_counter=}")
        logger.info(f"{self.action_queue=}")
        logger.info(f"{self.task_prompt_override=}")
        logger.info(f"{self.prompt_from_obs=}")
        logger.info(f"{self.prompt_key=}")
        logger.info(f"{self.fallback_prompt=}")
        logger.info(f"{self.wm_in_prompt=}")
        logger.info(f"{self._task_prompt_from_mapping=}")
        logger.info(f"{self.subtask_prompts=}")
        logger.info(f"{self.skill_prompts=}")
        logger.info(f"{self.video_memory_frames=}")
        logger.info(f"{self.video_memory_stride=}")

    def reset(self):
        self.action_queue = deque(maxlen=self.action_horizon)
        self.last_action = {"actions": np.zeros((self.action_horizon, 23), dtype=np.float64)}
        self.step_counter = 0
        self._frame_buffers = {}
        if self.reasoner:
            self.reasoner.reset()

    def _format_working_memory_for_prompt(self, wm) -> str:
        """
        Convert `working_memory` (usually a small dict) into a compact, stable prompt block.

        This is intentionally minimal for ablations: it only includes fields that are reliably
        available from the eval-side WM extractor.
        """
        if wm is None:
            return ""

        # If wm is already a string, wrap it for delimiting.
        if not isinstance(wm, dict):
            text = str(wm).strip()
            if not text:
                return ""
            return "\n".join(["[WM]", text, "[/WM]"])

        def _fmt_subtask_done_map(m) -> str:
            """
            Format a subtask-done mapping into: key1=true, key2=false
            Keeps a stable key order (sorted).
            """
            if m is None:
                return ""
            # Allow passing a single "k=v" string through.
            if isinstance(m, str):
                return m.strip()
            if not isinstance(m, dict):
                return str(m).strip()
            items = []
            for k in sorted(m.keys(), key=lambda x: str(x)):
                v = m.get(k)
                if hasattr(v, "item"):  # numpy scalar
                    v = v.item()
                if isinstance(v, bool):
                    v_str = "true" if v else "false"
                else:
                    v_str = str(v)
                items.append(f"{k}={v_str}")
            return ", ".join(items).strip()

        def _fmt_val(v) -> str:
            # 处理 numpy 数组（如 array(False), array(True)）
            if hasattr(v, 'item'):  # numpy scalar
                v = v.item()
            elif hasattr(v, 'tolist'):  # numpy array
                v = v.tolist()
                if isinstance(v, list) and len(v) == 1:
                    v = v[0]
            
            # 统一布尔值为小写
            if isinstance(v, bool):
                return "true" if v else "false"
            
            # 字符串直接返回（如 stage, target, holding）
            if isinstance(v, str):
                return v
            
            return str(v)

        lines = ["[WM]"]
        # Keep a stable field order.
        for k in ("stage", "target", "grasped", "holding", "dropped"):
            if k in wm and wm[k] is not None:
                if k == "subtask_done":
                    s = _fmt_subtask_done_map(wm[k])
                    if s:
                        lines.append(f"{k.upper()}: {s}")
                else:
                    lines.append(f"{k.upper()}: {_fmt_val(wm[k])}")
        lines.append("[/WM]")

        # If nothing beyond header/footer, skip.
        if len(lines) <= 2:
            return ""
        return "\n".join(lines)

    def _append_working_memory_to_prompt(self, prompt: str, wm) -> str:
        wm_block = self._format_working_memory_for_prompt(wm)
        if not wm_block:
            return prompt
        prompt = prompt or ""
        # Separate cleanly from the main task prompt.
        return (prompt.rstrip() + "\n\n" + wm_block).strip()

    def _buffer_frame(self, cam_name: str, frame: np.ndarray) -> tuple[list[np.ndarray], list[bool]]:
        """Buffer a frame and return (K-1 history frames, K-1 valid flags) for MEM.

        Returns ([], []) if video_memory_frames <= 1.
        """
        K = self.video_memory_frames
        if K <= 1:
            return [], []

        if cam_name not in self._frame_buffers:
            self._frame_buffers[cam_name] = deque(maxlen=self._frame_buffer_maxlen)
        buf = self._frame_buffers[cam_name]
        buf.append(frame.copy())

        available = list(buf)[:-1]
        sampled = []
        valid_flags = []
        for i in range(K - 1, 0, -1):
            idx = len(available) - i * self.video_memory_stride
            if idx < 0:
                sampled.append(available[0] if available else frame)
                valid_flags.append(False)
            else:
                sampled.append(available[idx])
                valid_flags.append(True)
        return sampled, valid_flags

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

        # MEM: buffer each camera frame for video memory
        if self.video_memory_frames > 1:
            head_hist, head_valid = self._buffer_frame("head", img_obs[0, 0])
            left_hist, left_valid = self._buffer_frame("left_wrist", img_obs[0, 1])
            right_hist, right_valid = self._buffer_frame("right_wrist", img_obs[0, 2])
            processed_obs["_frame_history"] = {
                "head": head_hist,
                "left_wrist": left_hist,
                "right_wrist": right_hist,
            }
            processed_obs["_frame_history_valid"] = {
                "head": head_valid,
                "left_wrist": left_valid,
                "right_wrist": right_valid,
            }

        if isinstance(obs, dict) and self.prompt_key in obs:
            processed_obs[self.prompt_key] = obs[self.prompt_key]

        if "robot_r1::robot_r1:zed_link:Camera:0::depth_linear" in obs:
            depth_obs = obs["robot_r1::robot_r1:zed_link:Camera:0::depth_linear"]
            depth_obs = cv2.resize(depth_obs, (DESPTH_RESIZE_SIZE, DESPTH_RESIZE_SIZE), interpolation=cv2.INTER_LINEAR)
            processed_obs["observation/egocentric_depth"] = depth_obs[None]

        if "working_memory" in obs:
            processed_obs["working_memory"] = obs["working_memory"]

        return processed_obs

    def _build_policy_batch(self, nbatch: dict) -> dict:
        """Build the dict expected by policy.infer from processed obs."""
        if nbatch["observation"].shape[-1] != 3:
            nbatch["observation"] = np.transpose(nbatch["observation"], (0, 1, 3, 4, 2))

        joint_positions = nbatch["proprio"][0]
        prompt = self._effective_task_prompt(nbatch)

        batch = {
            "observation/egocentric_camera": nbatch["observation"][0, 0],
            "observation/wrist_image_left": nbatch["observation"][0, 1],
            "observation/wrist_image_right": nbatch["observation"][0, 2],
            "observation/state": joint_positions,
            "prompt": prompt,
        }

        # MEM: attach frame history + valid flags from process_obs
        if self.video_memory_frames > 1 and "_frame_history" in nbatch:
            fh = nbatch["_frame_history"]
            batch["observation/egocentric_camera_history"] = fh["head"]
            batch["observation/wrist_image_left_history"] = fh["left_wrist"]
            batch["observation/wrist_image_right_history"] = fh["right_wrist"]
            if "_frame_history_valid" in nbatch:
                fv = nbatch["_frame_history_valid"]
                batch["observation/egocentric_camera_history_valid"] = fv["head"]
                batch["observation/wrist_image_left_history_valid"] = fv["left_wrist"]
                batch["observation/wrist_image_right_history_valid"] = fv["right_wrist"]

        if self.wm_in_prompt and "working_memory" in nbatch:
            batch["working_memory"] = nbatch["working_memory"]
            if self.step_counter % 100 == 0:
                logger.info(f"[WM] {batch['working_memory']}")

        if "observation/egocentric_depth" in nbatch:
            batch["observation/egocentric_depth"] = nbatch["observation/egocentric_depth"][0]

        return batch

    def act_receeding_temporal(self, input_obs):
        # Step 1: check if we should re-run policy
        if self.step_counter % self.replan_interval == 0:
            nbatch = copy.deepcopy(input_obs)
            batch = self._build_policy_batch(nbatch)

            if self.fine_grained_level > 0:
                reasoner_response = self.reasoner.generate_subtask(
                    high_level_task=batch["prompt"],
                    multi_modals=[batch["observation/egocentric_camera"]],
                )
                logger.info(f"* {reasoner_response}")
                batch["prompt"] = reasoner_response

            if self.wm_in_prompt and "working_memory" in batch:
                batch["prompt"] = self._append_working_memory_to_prompt(batch["prompt"], batch["working_memory"])
                if self.step_counter % 10 == 0:
                    logger.info(f"[PROMPT_TAIL] ...{batch['prompt'][-400:]}")

            try:
                action = self.policy.infer(batch)
                self.last_action = action
            except Exception as e:
                action = self.last_action
                logger.info(
                    f"Error in action prediction at step {self.step_counter}, using last action: {e}"
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
        batch = self._build_policy_batch(nbatch)

        if self.fine_grained_level > 0:
            reasoner_response = self.reasoner.generate_subtask(
                high_level_task=batch["prompt"],
                multi_modals=[batch["observation/egocentric_camera"]],
            )
            logger.info(f"* {reasoner_response}")
            batch["prompt"] = reasoner_response

        if self.wm_in_prompt and "working_memory" in batch:
            batch["prompt"] = self._append_working_memory_to_prompt(batch["prompt"], batch["working_memory"])
            logger.info(f"[PROMPT_TAIL] ...{batch['prompt'][-400:]}")

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
