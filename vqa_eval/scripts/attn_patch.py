"""Monkey-patch ``openpi.models.gemma.Attention.__call__`` to ship the per-layer
attention probabilities (last query row only) out to a Python-side buffer via
``jax.debug.callback``.

Why this approach (vs. ``flax.linen``'s ``self.sow`` mechanism):

- ``Block`` is wrapped by ``nn.scan`` with ``variable_axes={"params": 0}``;
  adding a new ``intermediates`` collection requires changing that scan config,
  which forces touching ``Module.setup`` and risks param-structure drift across
  the lazy-init/restore boundary.
- ``jax.debug.callback`` is purely side-effecting and is preserved through JIT
  and ``nn.scan``; it ships concrete numpy arrays to host code in iteration
  order (``ordered=True``), so the LAST callback invocation per forward pass
  corresponds to the deepest transformer layer.
- Training is completely unaffected: this module is only imported and patched
  by the inference / attention-extraction script. ``src/openpi/`` is untouched.

Public API::

    apply_attention_patch()            # call once before any model forward
    reset_attn_buffer()                # clear the buffer before each capture
    get_attn_layers() -> list[np.ndarray]
                                       # list of per-layer (B, K, G, T_k)
                                       # arrays in scan/depth order
    restore_attention_patch()          # restore the original __call__ (cleanup)

The captured slice is ``probs[:, :, :, -1, :]`` -- i.e. the attention from the
LAST query position. During the prefix-fill forward, that query position is the
last prompt token, whose hidden state directly produces the next-token logits
(the would-be first answer token). For an autoregressive single-token step,
``T_q == 1`` so the slice is just the new token's full attention row.
"""

from __future__ import annotations

import threading

import einops
import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
from flax.linen.module import wrap_method_once  # private but stable across recent flax

from openpi.models import gemma as _gemma
from openpi.models import lora as _lora


_lock = threading.Lock()
_attn_buffer: list[np.ndarray] = []
_orig_attn_call = None

# Number of trailing query rows to export per layer. 32 comfortably covers any
# realistic prompt suffix (image_len=768, text_len typically 12-20). The
# captured slice has shape (B, K, G, n_q, T_k) where n_q = min(T_q, N_QUERY_ROWS).
N_QUERY_ROWS = 32


def reset_attn_buffer() -> None:
    """Clear the captured per-layer attention slices."""
    with _lock:
        _attn_buffer.clear()


def get_attn_layers() -> list[np.ndarray]:
    """Return a copy of the captured per-layer attention slices.

    Each element has shape ``(B, K, G, T_k)``. Order matches scan iteration
    order, so the last element is the deepest layer.
    """
    with _lock:
        return list(_attn_buffer)


def _record(probs_arr) -> None:
    arr = np.asarray(probs_arr)
    with _lock:
        _attn_buffer.append(arr)


def _name(prefix: str, i: int) -> str:
    """Mirror the private ``gemma._name`` (kept here to avoid touching src/)."""
    return prefix if i == 0 else f"{prefix}_{i}"


def _make_patched_attn_call():
    """Build the patched ``__call__``. Decorated with ``@nn.compact`` (which is
    just ``fn.compact = True``) so flax treats it the same as the original.
    """

    def _patched_attn_call(self, xs, positions, attn_mask, kv_cache):
        # ---- begin verbatim copy of gemma.Attention.__call__ ----
        assert all(config.head_dim == self.configs[0].head_dim for config in self.configs)
        assert all(config.num_heads == self.configs[0].num_heads for config in self.configs)
        assert all(config.num_kv_heads == self.configs[0].num_kv_heads for config in self.configs)

        dtype = next(x.dtype for x in xs if x is not None)

        qkvs = []
        for i, (x, config) in enumerate(zip(xs, self.configs, strict=True)):
            if x is None:
                continue
            if config.num_kv_heads == config.num_heads:
                qkv_einsum = _lora.Einsum(
                    shape=(3, config.num_heads, config.width, config.head_dim),
                    name=_name("qkv_einsum", i),
                    init_fn=nn.initializers.lecun_normal(in_axis=-2, out_axis=-1, batch_axis=(0, 1)),
                    lora_config=config.lora_configs.get("attn"),
                )
                qkvs.append(qkv_einsum("BSD,3KDH->3BSKH", x))
            else:
                q_einsum = _lora.Einsum(
                    shape=(config.num_heads, config.width, config.head_dim),
                    name=_name("q_einsum", i),
                    init_fn=nn.initializers.lecun_normal(in_axis=-2, out_axis=-1, batch_axis=(0,)),
                    lora_config=config.lora_configs.get("attn"),
                )
                q = q_einsum("BTD,NDH->BTNH", x)
                kv_einsum = _lora.Einsum(
                    shape=(2, config.num_kv_heads, config.width, config.head_dim),
                    name=_name("kv_einsum", i),
                    init_fn=nn.initializers.lecun_normal(in_axis=-2, out_axis=-1, batch_axis=(0, 1)),
                    lora_config=config.lora_configs.get("attn"),
                )
                k, v = kv_einsum("BSD,2KDH->2BSKH", x)
                qkvs.append((q, k, v))

        q, k, v = (jnp.concatenate(y, axis=1) for y in zip(*qkvs, strict=True))

        q = _gemma._apply_rope(q, positions=positions)
        q *= self.configs[0].head_dim ** -0.5
        k = _gemma._apply_rope(k, positions=positions)

        assert q.dtype == k.dtype == v.dtype == dtype

        if kv_cache is not None:
            cache_k, cache_v = kv_cache
            k = jnp.concatenate([cache_k, k], axis=1)
            v = jnp.concatenate([cache_v, v], axis=1)

        q = einops.rearrange(q, "B T (K G) H -> B T K G H", K=self.configs[0].num_kv_heads)
        logits = jnp.einsum("BTKGH,BSKH->BKGTS", q, k, preferred_element_type=jnp.float32)

        if attn_mask.shape != (q.shape[0], 1, q.shape[1], k.shape[1]):
            raise ValueError(
                f"Attention mask with shape {attn_mask.shape} but shapes for q and k are: {q.shape} and {k.shape}"
            )

        big_neg = -2.3819763e38
        masked_logits = jnp.where(attn_mask[:, :, None, :, :], logits, big_neg)

        probs = jax.nn.softmax(masked_logits, axis=-1).astype(dtype)

        # ---- attention capture (the only addition vs. the original) ----
        # probs shape: (B, K, G, T_q, T_k). We export the LAST `N_QUERY_ROWS`
        # rows so callers can pick attention from arbitrary text-token query
        # positions (e.g. the object-name token, not just the trailing `\n`).
        # For T_q < N_QUERY_ROWS (e.g. autoregressive single-token steps), the
        # slice is naturally smaller and still meaningful.
        last_query_probs = probs[:, :, :, -N_QUERY_ROWS:, :].astype(jnp.float32)
        jax.debug.callback(_record, last_query_probs, ordered=True)
        # ---- end addition ----

        encoded = jnp.einsum("BKGTS,BSKH->BTKGH", probs, v)
        encoded = einops.rearrange(encoded, "B T K G H -> B T (K G) H")

        out = []
        start = 0
        for i, (x, config) in enumerate(zip(xs, self.configs, strict=True)):
            if x is not None:
                end = start + x.shape[1]
                out_einsum = _lora.Einsum(
                    shape=(config.num_heads, config.head_dim, config.width),
                    name=_name("attn_vec_einsum", i),
                    init_fn=nn.initializers.lecun_normal(in_axis=(-3, -2), out_axis=-1),
                    lora_config=config.lora_configs.get("attn"),
                )
                out.append(out_einsum("BTNH,NHD->BTD", encoded[:, start:end]))
                start = end
            else:
                out.append(None)

        return out, (k, v)
        # ---- end verbatim copy ----

    _patched_attn_call.compact = True  # equivalent to @nn.compact
    return _patched_attn_call


def apply_attention_patch() -> None:
    """Install the patched ``Attention.__call__``. Idempotent.

    We re-use ``flax.linen.module.wrap_method_once`` so the replacement
    function gets the same scope-binding wrapper that the original received
    during class creation. Without this, ``self.compact`` calls inside the
    replacement raise ``CallCompactUnboundModuleError`` -- or worse, the
    submodules end up double-registered and break checkpoint loading.
    """
    global _orig_attn_call
    if _orig_attn_call is not None:
        return
    _orig_attn_call = _gemma.Attention.__call__
    _gemma.Attention.__call__ = wrap_method_once(_make_patched_attn_call())


def restore_attention_patch() -> None:
    """Uninstall the patch. Idempotent."""
    global _orig_attn_call
    if _orig_attn_call is None:
        return
    _gemma.Attention.__call__ = _orig_attn_call
    _orig_attn_call = None
