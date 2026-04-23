"""Helpers for richer W&B training logs without heavy per-step host sync.

PI0 ``compute_loss_and_metrics`` 会在 ``info`` 里带上 ``loss_flow``、``loss_token``、
``loss_fast``、``loss_vqa`` 等键；本模块在 W&B 里额外写入 ``loss_components/...``
别名（带 ``_std`` / ``_min`` / ``_max`` 后缀时一并映射），便于在面板里按子损失分组查看。
"""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
import jax.tree_util as jtu

# 与 ``openpi.models.pi0.Pi0.compute_loss_and_metrics`` 返回的 metrics 键对齐。
_LOSS_COMPONENT_PANEL: dict[str, str] = {
    "loss_flow": "loss_components/flow_matching_unweighted",
    "loss_flow_weighted": "loss_components/flow_matching_weighted",
    "loss_token": "loss_components/language_token_ce_mean",
    "loss_fast": "loss_components/language_fast_ce_unweighted",
    "loss_fast_weighted": "loss_components/language_fast_ce_weighted",
    "loss_vqa": "loss_components/language_vqa_ce_unweighted",
    "loss_vqa_weighted": "loss_components/language_vqa_ce_weighted",
    "noise_time_mean": "loss_components/flow_noise_time_mean",
    "noise_time_std": "loss_components/flow_noise_time_std",
}

_LOSS_AGG_SUFFIXES: tuple[str, ...] = ("_std", "_min", "_max")


def _loss_metric_base_and_suffix(key: str) -> tuple[str, str]:
    for suf in _LOSS_AGG_SUFFIXES:
        if key.endswith(suf):
            return key[: -len(suf)], suf
    return key, ""


def add_loss_component_groups_for_wandb(metrics: dict[str, float]) -> None:
    """为 PI0 多任务子损失增加 ``loss_components/*`` 键，便于 W&B 按目录分组；不删除原有键。"""
    additions: dict[str, float] = {}
    for key, val in metrics.items():
        base, suf = _loss_metric_base_and_suffix(key)
        panel = _LOSS_COMPONENT_PANEL.get(base)
        if panel is None:
            continue
        additions[panel + suf] = val
    metrics.update(additions)


def jax_stacked_infos_to_wandb(
    stacked_infos: dict[str, Any],
    *,
    extra: dict[str, float] | None = None,
) -> dict[str, float]:
    """Map each (log_interval,) series to mean/std/min/max on device, then one small host copy.

    Primary scalars keep the original key (mean). Dispersion uses ``{key}_std`` / ``_min`` / ``_max``.
    """
    out: dict[str, float] = {}

    def _summarize_series(x: Any) -> jnp.ndarray:
        v = jnp.asarray(x)
        if v.ndim == 0:
            m = v
            return jnp.stack([m, jnp.array(0.0, dtype=m.dtype), m, m])
        m = jnp.mean(v)
        return jnp.stack([m, jnp.std(v), jnp.min(v), jnp.max(v)])

    summarized = jtu.tree_map(_summarize_series, stacked_infos)
    cpu = jax.device_get(summarized)
    for k, arr in cpu.items():
        row = jnp.asarray(arr)
        out[str(k)] = float(row[0])
        out[f"{k}_std"] = float(row[1])
        out[f"{k}_min"] = float(row[2])
        out[f"{k}_max"] = float(row[3])
    if extra:
        out.update(extra)
    add_loss_component_groups_for_wandb(out)
    return out
