from __future__ import annotations

from pathlib import Path
from typing import Any

from omegaconf import OmegaConf


class DummyPolicy:
    """
    占位符 policy，用于 GR00T 等不使用 Evaluator 内置 policy 的 VLA。
    """

    def reset(self) -> None:
        pass

    def act(self, obs):
        raise NotImplementedError(
            "DummyPolicy.act() should not be called. "
            "The actual policy should be set before running episodes."
        )


def build_behavior_b1k_config(
    configs_root: Path,
    task_name: str,
    log_path: Path,
    vla_type: str = "openpi-comet",
) -> Any:
    base_cfg = OmegaConf.load(configs_root / "base_config.yaml")
    robot_cfg = OmegaConf.create({"robot": OmegaConf.load(configs_root / "robot" / "r1pro.yaml")})
    task_cfg = OmegaConf.create({"task": OmegaConf.load(configs_root / "task" / "behavior.yaml")})

    if vla_type == "groot":
        policy_cfg = OmegaConf.create(
            {
                "policy_name": "placeholder",
                "model": {
                    "_target_": "benchmark.eval.config.DummyPolicy",
                },
            }
        )
    else:
        policy_cfg = OmegaConf.load(configs_root / "policy" / "websocket.yaml")

    cfg = OmegaConf.merge(base_cfg, robot_cfg, task_cfg, policy_cfg)
    cfg.task.name = task_name
    cfg.log_path = str(log_path)
    return cfg
