import dataclasses
from collections import OrderedDict

import einops
import numpy as np

try:
    from omnigibson.learning.utils.eval_utils import PROPRIOCEPTION_INDICES
except ModuleNotFoundError:
    # Minimal fallback to avoid hard OmniGibson dependency during training setup.
    # Keep indices aligned with BEHAVIOR-1K OmniGibson eval_utils.
    PROPRIOCEPTION_INDICES = {
        "R1Pro": OrderedDict(
            {
                "joint_qpos": np.s_[0:28],
                "joint_qpos_sin": np.s_[28:56],
                "joint_qpos_cos": np.s_[56:84],
                "joint_qvel": np.s_[84:112],
                "joint_qeffort": np.s_[112:140],
                "robot_pos": np.s_[140:143],
                "robot_ori_cos": np.s_[143:146],
                "robot_ori_sin": np.s_[146:149],
                "robot_2d_ori": np.s_[149:150],
                "robot_2d_ori_cos": np.s_[150:151],
                "robot_2d_ori_sin": np.s_[151:152],
                "robot_lin_vel": np.s_[152:155],
                "robot_ang_vel": np.s_[155:158],
                "arm_left_qpos": np.s_[158:165],
                "arm_left_qpos_sin": np.s_[165:172],
                "arm_left_qpos_cos": np.s_[172:179],
                "arm_left_qvel": np.s_[179:186],
                "eef_left_pos": np.s_[186:189],
                "eef_left_quat": np.s_[189:193],
                "gripper_left_qpos": np.s_[193:195],
                "gripper_left_qvel": np.s_[195:197],
                "arm_right_qpos": np.s_[197:204],
                "arm_right_qpos_sin": np.s_[204:211],
                "arm_right_qpos_cos": np.s_[211:218],
                "arm_right_qvel": np.s_[218:225],
                "eef_right_pos": np.s_[225:228],
                "eef_right_quat": np.s_[228:232],
                "gripper_right_qpos": np.s_[232:234],
                "gripper_right_qvel": np.s_[234:236],
                "trunk_qpos": np.s_[236:240],
                "trunk_qvel": np.s_[240:244],
                "base_qpos": np.s_[244:247],
                "base_qpos_sin": np.s_[247:250],
                "base_qpos_cos": np.s_[250:253],
                "base_qvel": np.s_[253:256],
            }
        )
    }

from openpi import transforms
from openpi.models import model as _model

MAX_DEPTH = 10.0

CAMERA_INTRINSICS = {
    "head": np.array([[306.0, 0.0, 360.0], [0.0, 306.0, 360.0], [0.0, 0.0, 1.0]], dtype=np.float32),  # 720x720
    "left_wrist": np.array(
        [[388.6639, 0.0, 240.0], [0.0, 388.6639, 240.0], [0.0, 0.0, 1.0]], dtype=np.float32
    ),  # 480x480
    "right_wrist": np.array(
        [[388.6639, 0.0, 240.0], [0.0, 388.6639, 240.0], [0.0, 0.0, 1.0]], dtype=np.float32
    ),  # 480x480
}


def make_b1k_example() -> dict:
    """Creates a random input example for the Droid policy."""
    return {
        "observation/egocentric_camera": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "observation/wrist_image_left": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "observation/wrist_image_right": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "observation/joint_position": np.random.rand(23),
        "prompt": "do something",
    }


def extract_state_from_proprio(proprio_data):
    """
    We assume perfect correlation for the two gripper fingers.
    """
    # extract joint position
    base_qvel = proprio_data[..., PROPRIOCEPTION_INDICES["R1Pro"]["base_qvel"]]  # 3
    trunk_qpos = proprio_data[..., PROPRIOCEPTION_INDICES["R1Pro"]["trunk_qpos"]]  # 4
    arm_left_qpos = proprio_data[..., PROPRIOCEPTION_INDICES["R1Pro"]["arm_left_qpos"]]  #  7
    arm_right_qpos = proprio_data[..., PROPRIOCEPTION_INDICES["R1Pro"]["arm_right_qpos"]]  #  7
    left_gripper_width = proprio_data[..., PROPRIOCEPTION_INDICES["R1Pro"]["gripper_left_qpos"]].sum(
        axis=-1, keepdims=True
    )  # 1
    right_gripper_width = proprio_data[..., PROPRIOCEPTION_INDICES["R1Pro"]["gripper_right_qpos"]].sum(
        axis=-1, keepdims=True
    )  # 1
    return np.concatenate(
        [
            base_qvel,
            trunk_qpos,
            arm_left_qpos,
            # left_gripper_width,
            arm_right_qpos,
            left_gripper_width,  # NOTE: we rearrange the gripper from 21 to 14 to match the action space
            right_gripper_width,
        ],
        axis=-1,
    )


def _parse_image(image) -> np.ndarray:
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).astype(np.uint8)
    if image.shape[0] == 3:
        image = einops.rearrange(image, "c h w -> h w c")
    return image


def _parse_seg_image(image) -> np.ndarray:
    MAX_SEG = 8
    image = np.asarray(image)
    image = image / MAX_SEG * 255
    image = np.repeat(image[..., np.newaxis], 3, axis=-1)
    return image.astype(np.uint8)


def depth_to_pcd(depth_image: np.ndarray, camera_intrinsics: np.ndarray, downsample: int = 6) -> np.ndarray:
    """
    Convert depth image to point cloud.
    """
    depth_image = np.asarray(depth_image)
    h, w = depth_image.shape[:2]
    u, v = np.meshgrid(np.arange(w), np.arange(h))
    x = (u - camera_intrinsics[0, 2]) * depth_image / camera_intrinsics[0, 0]
    y = (v - camera_intrinsics[1, 2]) * depth_image / camera_intrinsics[1, 1]
    z = depth_image
    pcd_xyz = np.stack([x, y, z], axis=-1)  # (h, w, 3)

    pcd_xyz = pcd_xyz[::downsample, ::downsample].reshape(16, -1, 3)
    return pcd_xyz  # (16, 2025, 3)


@dataclasses.dataclass(frozen=True)
class B1kInputs(transforms.DataTransformFn):
    # The action dimension of the model. Will be used to pad state and actions.
    action_dim: int

    # Determines which model will be used.
    model_type: _model.ModelType = _model.ModelType.PI0

    meta_image_keys: list[str] = dataclasses.field(default_factory=list)

    depth_as_pcd: bool = False

    pcd_downsample: int = 6

    # MEM: number of video memory frames. 1 = single-frame (original behavior).
    video_memory_frames: int = 1

    def __call__(self, data: dict) -> dict:
        proprio_data = data["observation/state"]
        # extract joint position
        state = extract_state_from_proprio(proprio_data)
        if "actions" in data:
            action = data["actions"]

        K = self.video_memory_frames

        # Possibly need to parse images to uint8 (H,W,C) since LeRobot automatically
        # stores as float32 (C,H,W), gets skipped for policy inference
        base_image = _parse_image(data["observation/egocentric_camera"])
        wrist_image_left = _parse_image(data["observation/wrist_image_left"])
        wrist_image_right = _parse_image(data["observation/wrist_image_right"])

        # MEM: stack history frames if available
        temporal_mask = None
        if K > 1:
            base_image, temporal_mask = _stack_history_frames(data, "observation/egocentric_camera", base_image, K)
            wrist_image_left, _ = _stack_history_frames(data, "observation/wrist_image_left", wrist_image_left, K)
            wrist_image_right, _ = _stack_history_frames(data, "observation/wrist_image_right", wrist_image_right, K)

        meta_images, meta_image_names = [], []

        if self.depth_as_pcd:
            depth_image = data["observation/egocentric_depth"]
            pcd_xyz = depth_to_pcd(depth_image, CAMERA_INTRINSICS["head"], self.pcd_downsample)

        if "observation/egocentric_seg" in self.meta_image_keys:
            seg_image = _parse_seg_image(data["observation/egocentric_seg"])
            meta_images.append(seg_image)
            meta_image_names.append("base_0_seg")

        match self.model_type:
            case _model.ModelType.PI0 | _model.ModelType.PI05:
                names = ("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb")
                images = (base_image, wrist_image_left, wrist_image_right)
                image_masks = (np.True_, np.True_, np.True_)
            case _model.ModelType.PI0_FAST:
                names = ("base_0_rgb", "base_1_rgb", "wrist_0_rgb")
                # We don't mask out padding images for FAST models.
                images = (base_image, wrist_image_left, wrist_image_right)
                image_masks = (np.True_, np.True_, np.True_)
            case _:
                raise ValueError(f"Unsupported model type: {self.model_type}")

        names += tuple(meta_image_names)
        images += tuple(meta_images)
        image_masks += tuple(np.True_ for _ in meta_image_names)

        inputs = {
            "state": state,
            "image": dict(zip(names, images, strict=True)),
            "image_mask": dict(zip(names, image_masks, strict=True)),
        }

        if "actions" in data:
            inputs["actions"] = action

        if "prompt" in data:
            inputs["prompt"] = data["prompt"]

        if self.depth_as_pcd:
            inputs["pcd_xyz"] = pcd_xyz
        if temporal_mask is not None:
            inputs["temporal_mask"] = temporal_mask
        return inputs


# Mapping from repacked key to raw B1K dataset key (for history lookup)
_REPACK_TO_RAW = {
    "observation/egocentric_camera": "observation.images.rgb.head",
    "observation/wrist_image_left": "observation.images.rgb.left_wrist",
    "observation/wrist_image_right": "observation.images.rgb.right_wrist",
}


def _stack_history_frames(data: dict, key: str, current_frame: np.ndarray, K: int) -> tuple[np.ndarray, np.ndarray]:
    """Stack K frames: K-1 history + 1 current. Returns ([K, H, W, C], [K] bool mask).

    Looks for history frames in data using both repacked and raw key conventions.
    If not available, repeats the current frame K times.
    The mask indicates which frames are valid (True) vs padded (False).
    """
    history = None
    valid_flags = None
    for candidate in (f"{key}_history", f"{_REPACK_TO_RAW.get(key, key)}_history"):
        if candidate in data and len(data[candidate]) >= K - 1:
            history = [_parse_image(f) for f in data[candidate][-(K - 1):]]
            valid_key = f"{candidate}_valid"
            if valid_key in data:
                valid_flags = list(data[valid_key][-(K - 1):])
            break

    if history is not None:
        frames = np.stack(history + [current_frame], axis=0)
        if valid_flags is not None:
            mask = np.array(valid_flags + [True], dtype=np.bool_)
        else:
            mask = np.ones(K, dtype=np.bool_)
        return frames, mask
    frames = np.stack([current_frame] * K, axis=0)
    mask = np.array([False] * (K - 1) + [True], dtype=np.bool_)
    return frames, mask


@dataclasses.dataclass(frozen=True)
class B1kOutputs(transforms.DataTransformFn):
    action_dim: int = 23

    def __call__(self, data: dict) -> dict:
        # Only return the first 23 dims.
        return {"actions": np.asarray(data["actions"][:, : self.action_dim])}
