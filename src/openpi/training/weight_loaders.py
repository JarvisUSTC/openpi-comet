import dataclasses
import logging
import re
from typing import Protocol, runtime_checkable

import flax.traverse_util
import numpy as np

import openpi.models.model as _model
import openpi.shared.array_typing as at
import openpi.shared.download as download

logger = logging.getLogger(__name__)


@runtime_checkable
class WeightLoader(Protocol):
    def load(self, params: at.Params) -> at.Params:
        """Loads the model weights.

        Args:
            params: Parameters of the model. This is a nested structure of array-like objects that
                represent the model's parameters.

        Returns:
            Loaded parameters. The structure must be identical to `params`. If returning a subset of
            the parameters the loader must merge the loaded parameters with `params`.
        """


@dataclasses.dataclass(frozen=True)
class NoOpWeightLoader(WeightLoader):
    def load(self, params: at.Params) -> at.Params:
        return params


@dataclasses.dataclass(frozen=True)
class CheckpointWeightLoader(WeightLoader):
    """Loads an entire set of weights from a checkpoint.

    Compatible with:
      trained checkpoints:
        example: "./checkpoints/<config>/<exp>/<step>/params"
      released checkpoints:
        example: "gs://openpi-assets/checkpoints/<model>/params"
    """

    params_path: str

    def load(self, params: at.Params) -> at.Params:
        # We are loading np.ndarray and relying on the training code to properly convert and shard the params.
        loaded_params = _model.restore_params(download.maybe_download(self.params_path), restore_type=np.ndarray)
        # Add all missing LoRA weights.
        return _merge_params(loaded_params, params, missing_regex=".*lora.*|pointnet.*")


@dataclasses.dataclass(frozen=True)
class PaliGemmaWeightLoader(WeightLoader):
    """Loads weights from the official PaliGemma checkpoint.

    This will overwrite existing weights with similar names while keeping all extra weights intact.
    This allows us to support the action expert which is used by the Pi0 model.
    """

    def load(self, params: at.Params) -> at.Params:
        path = download.maybe_download(
            "gs://vertex-model-garden-paligemma-us/paligemma/pt_224.npz", gs={"token": "anon"}
        )
        with path.open("rb") as f:
            flat_params = dict(np.load(f, allow_pickle=False))
        loaded_params = {"PaliGemma": flax.traverse_util.unflatten_dict(flat_params, sep="/")["params"]}
        # Add all missing weights.
        return _merge_params(loaded_params, params, missing_regex=".*")


def _expand_siglip_scan_to_unroll(flat_loaded: dict, flat_ref: dict) -> dict:
    """Convert SigLIP scan-format encoder block params to unrolled format.

    The base checkpoint stores SigLIP encoder blocks in scan format:
      Transformer/encoderblock/<param>  shape=[num_layers, ...]
    When K>1, the model unrolls these into separate entries:
      Transformer/encoderblock_0/<param>, encoderblock_1/<param>, ...
    This function expands the scan params to match the unrolled model.
    """
    scan_prefix = "/img/Transformer/encoderblock/"
    unroll_prefix_re = re.compile(r".*/img/Transformer/encoderblock_(\d+)/")

    has_scan = any(scan_prefix in k for k in flat_loaded)
    has_unroll_ref = any(unroll_prefix_re.search(k) for k in flat_ref)

    if not (has_scan and has_unroll_ref):
        return flat_loaded

    # Determine num_layers from ref keys
    layer_indices = set()
    for k in flat_ref:
        m = unroll_prefix_re.search(k)
        if m:
            layer_indices.add(int(m.group(1)))
    if not layer_indices:
        return flat_loaded
    num_layers = max(layer_indices) + 1

    result = {}
    for k, v in flat_loaded.items():
        if scan_prefix in k:
            # Split stacked axis 0 into per-layer keys
            idx = k.index(scan_prefix)
            prefix = k[:idx + len("/img/Transformer/")]
            suffix = k[idx + len(scan_prefix):]
            for i in range(num_layers):
                new_key = f"{prefix}encoderblock_{i}/{suffix}"
                result[new_key] = v[i]
        else:
            result[k] = v

    logger.info("Expanded SigLIP scan params: 1 encoderblock → %d encoderblock_N entries", num_layers)
    return result


def _merge_params(loaded_params: at.Params, params: at.Params, *, missing_regex: str) -> at.Params:
    """Merges the loaded parameters with the reference parameters.

    Args:
        loaded_params: The parameters to merge.
        params: The reference parameters.
        missing_regex: A regex pattern for all missing keys that should be merged from the reference parameters.

    Returns:
        A new dictionary with the merged parameters.
    """
    flat_ref = flax.traverse_util.flatten_dict(params, sep="/")
    flat_loaded = flax.traverse_util.flatten_dict(loaded_params, sep="/")

    # Expand SigLIP scan-format params to unrolled format if needed (K>1 training).
    flat_loaded = _expand_siglip_scan_to_unroll(flat_loaded, flat_ref)

    # First, take all weights that are a subset of the reference weights.
    result = {}
    for k, v in flat_loaded.items():
        if k in flat_ref:
            if v.dtype == flat_ref[k].dtype:
                result[k] = v
            else:
                print(f"Warning: {k} has dtype {v.dtype} but reference has dtype {flat_ref[k].dtype}")
                result[k] = v.astype(flat_ref[k].dtype)
    flat_loaded.clear()

    # Then, merge any missing weights as defined by the missing regex.
    pattern = re.compile(missing_regex)
    for k in {k for k in flat_ref if pattern.fullmatch(k)}:
        if k not in result:
            result[k] = flat_ref[k]

    return flax.traverse_util.unflatten_dict(result, sep="/")
