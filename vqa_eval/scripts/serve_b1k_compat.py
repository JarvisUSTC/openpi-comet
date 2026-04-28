"""
Compatibility launcher for scripts/serve_b1k.py.

The reason this exists:
  scripts/serve_b1k.py imports
      from omnigibson.learning.utils.network_utils import WebsocketPolicyServer
  and the venv we use for inference (/b1k/hyn/openpi-codebase/.venv) does not
  have the full OmniGibson package installed -- installing it pulls in NVIDIA
  Isaac, physx, pxr, etc. and is overkill for a pure VLA inference server.

What this launcher does:
  1. Synthesise minimal stub modules for `omnigibson` and `omnigibson.macros`
     (only `gm.DEBUG` is read by network_utils.py).
  2. Load the REAL `network_utils.py` shipped with the BEHAVIOR-1K source tree
     via importlib, so the wire protocol stays byte-identical to what the
     simulation client (also using BEHAVIOR-1K's WebsocketClientPolicy) speaks.
  3. Hand off to scripts/serve_b1k.py via runpy, preserving all CLI args.

If the BEHAVIOR-1K source tree moves, edit OMNIGIBSON_SRC below.
"""

from __future__ import annotations

import importlib.util
import os
import runpy
import sys
import types

OMNIGIBSON_SRC = "/b1k/Benchmark/BEHAVIOR-1K/OmniGibson/omnigibson"

# Submodules that openpi (this repo) actually imports from omnigibson. We load
# each one via importlib from the BEHAVIOR-1K source tree so the wire / numeric
# definitions stay byte-identical, without dragging in the rest of the package.
_REAL_SUBMODULES = {
    "omnigibson.learning.utils.network_utils": "learning/utils/network_utils.py",
    "omnigibson.learning.utils.eval_utils":    "learning/utils/eval_utils.py",
}


class _Gm:
    DEBUG = False


def _install_omnigibson_stub() -> None:
    root = types.ModuleType("omnigibson"); root.__path__ = []
    macros = types.ModuleType("omnigibson.macros"); macros.gm = _Gm()
    learning = types.ModuleType("omnigibson.learning"); learning.__path__ = []
    utils = types.ModuleType("omnigibson.learning.utils"); utils.__path__ = []
    sys.modules.update({
        "omnigibson": root,
        "omnigibson.macros": macros,
        "omnigibson.learning": learning,
        "omnigibson.learning.utils": utils,
    })

    for fqname, rel in _REAL_SUBMODULES.items():
        path = os.path.join(OMNIGIBSON_SRC, rel)
        if not os.path.isfile(path):
            raise FileNotFoundError(
                f"Expected omnigibson submodule at {path}, but it is missing. "
                "Edit OMNIGIBSON_SRC / _REAL_SUBMODULES at the top of this file, "
                "or install the full omnigibson package."
            )
        spec = importlib.util.spec_from_file_location(fqname, path)
        mod = importlib.util.module_from_spec(spec)
        sys.modules[fqname] = mod
        spec.loader.exec_module(mod)


_install_omnigibson_stub()

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.normpath(os.path.join(THIS_DIR, "..", ".."))
SERVE_B1K = os.path.join(REPO_ROOT, "scripts", "serve_b1k.py")

# The hyn venv has an OLDER editable-installed `openpi` (different repo) that
# is missing both `pi05_b1k-knowledge_insulation-vqa-joint-skill-pick-up-from-
# no-task-planning` in training/config.py and `openpi.shared.eval_b1k_wrapper`.
# Prepend THIS repo's src to sys.path so we import the right code, exactly the
# same pattern eval_vqa.py uses.
for _p in (
    os.path.join(REPO_ROOT, "src"),
    os.path.join(REPO_ROOT, "packages", "openpi-client", "src"),
):
    if _p not in sys.path:
        sys.path.insert(0, _p)

if not os.path.isfile(SERVE_B1K):
    raise FileNotFoundError(f"Cannot locate serve_b1k.py at {SERVE_B1K}")


def _patch_get_config_for_multi_data_inference() -> None:
    """KI+VQA-joint TrainConfigs declare `data` as a list of DataConfigFactories
    (behavior dataset + RoboInter-VQA + RobustVLGuard, mixed via sample_weights).

    `policy_config.create_trained_policy` calls `train_config.data.create(...)`
    and assumes a single DataConfigFactory, so it crashes with
        AttributeError: 'list' object has no attribute 'create'

    For inference-time serving we only need the robot-side DataConfig (the first
    one), since the VQA factories are only relevant during training.
    """
    import dataclasses
    import openpi.training.config as _train_config_mod

    if getattr(_train_config_mod.get_config, "_compat_patched", False):
        return
    _orig = _train_config_mod.get_config

    def _patched(name):
        cfg = _orig(name)
        data = cfg.data
        if isinstance(data, (list, tuple)) and len(data) > 0:
            cfg = dataclasses.replace(cfg, data=data[0], sample_weights=None)
        return cfg

    _patched._compat_patched = True
    _train_config_mod.get_config = _patched


def _patch_b1k_wrapper_to_read_obs_prompt() -> None:
    """Make B1KPolicyWrapper read prompt from the client-supplied observation.

    The shipped wrapper (src/openpi/shared/eval_b1k_wrapper.py) hard-codes
        batch["prompt"] = self.task_prompt
    where ``self.task_prompt`` is locked at server startup from
    ``--task_name`` via ``scripts/task_mapping.json``. That means every frame
    the model sees the same fixed sentence, regardless of what the simulation
    client attached as ``obs["prompt"]`` (e.g. via
    ``Evaluator.run_vla_episode(prompt=...)`` in BEHAVIOR-1K eval.py).

    To make the client authoritative WITHOUT touching eval_b1k_wrapper.py we
    monkey-patch ``B1KPolicyWrapper.act`` here. ``act`` is the single entry
    point invoked by ``WebsocketPolicyServer._handler`` (it later forwards to
    ``act_receeding_temporal`` internally), so swapping ``self.task_prompt``
    in/out around ``_orig_act`` covers every control_mode without touching the
    inner methods.

    Behaviour:
      * if ``obs["prompt"]`` is present and non-empty  -> use it for THIS call
      * otherwise                                       -> fall back to the
                                                          server's ``--task_name``
                                                          value (unchanged)

    The override is purely thread-local-equivalent (single-threaded asyncio
    handler), guarded by try/finally so the original task_prompt is always
    restored, even on exceptions.
    """
    from openpi.shared import eval_b1k_wrapper as _w

    if getattr(_w.B1KPolicyWrapper, "_compat_obs_prompt_patched", False):
        return

    _orig_act = _w.B1KPolicyWrapper.act

    def _patched_act(self, input_obs):
        client_prompt = None
        if isinstance(input_obs, dict):
            cp = input_obs.get("prompt", None)
            if cp is not None:
                if isinstance(cp, (bytes, bytearray)):
                    cp = cp.decode("utf-8", errors="replace")
                cp = str(cp).strip()
                if cp:
                    client_prompt = cp

        if client_prompt is None:
            return _orig_act(self, input_obs)

        saved = self.task_prompt
        self.task_prompt = client_prompt
        try:
            return _orig_act(self, input_obs)
        finally:
            self.task_prompt = saved

    _w.B1KPolicyWrapper.act = _patched_act
    _w.B1KPolicyWrapper._compat_obs_prompt_patched = True


_patch_get_config_for_multi_data_inference()
_patch_b1k_wrapper_to_read_obs_prompt()

sys.argv[0] = SERVE_B1K
runpy.run_path(SERVE_B1K, run_name="__main__")
