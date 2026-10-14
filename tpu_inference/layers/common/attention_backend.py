# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""The TPU attention backends, selected with
--additional-config '{"attention_backend": ..., "linear_attn_backend": ...}'.

vLLM's --attention-backend picks the framework adapter (PallasAttentionBackend
registered as FLASH_ATTN), not the TPU kernel behind it, and only accepts the
names of its closed AttentionBackendEnum, so the kernel is selected here.

AttentionBackend itself is the default "rpa" backend: each hook returns
nothing to add, and the RPA code inline in the platform, the runner, the
compilation manager and attention_interface runs as it is. Another backend
overrides the hooks it needs, the way a vLLM backend implements
validate_configuration / get_kv_cache_shape / AttentionMetadataBuilder.build /
build_for_cudagraph_capture / AttentionImpl.forward.

The linear attention layers of a hybrid model select their kernels apart from
the full attention ones, like SGLang's --linear-attn-backend, since the two
families have different kernels. LinearAttentionBackend itself is the
"default" backend, the GDN v3 kernel.
"""

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    import jax
    from jax.sharding import Mesh, NamedSharding
    from vllm.config import VllmConfig
    from vllm.v1.kv_cache_interface import KVCacheGroupSpec

    from tpu_inference.layers.common.attention_metadata import (
        AttentionMetadata, SharedAttentionMetadata)
    from tpu_inference.utils import DeviceBuffer

ATTENTION_BACKEND_KEY = "attention_backend"
LINEAR_ATTN_BACKEND_KEY = "linear_attn_backend"

# (unpacked device buffer, KV cache group id) -> extra AttentionMetadata kwargs
# of that group.
AttentionFields = Callable[[Mapping[str, "jax.Array"], int], dict[str, Any]]


def no_attention_fields(metadata: Mapping[str, "jax.Array"],
                        gid: int) -> dict[str, Any]:
    del metadata, gid
    return {}


@dataclass(frozen=True)
class HostStepInputs:
    """The host arrays TPUModelRunner._prepare_inputs has filled for one step.

    They follow its per-DP-rank layout: rank r's requests start at
    r * max_num_reqs_per_dp_rank in seq_lens and the block tables, and at
    r * (max_num_reqs_per_dp_rank + 1) in query_start_loc. request_distribution
    is the flat (dp_size * 3,) [num_decode, num_decode, num_reqs] per rank.
    """
    kv_cache_groups: "Sequence[KVCacheGroupSpec]"
    # KV cache group id -> (max_num_seqs, max_num_blocks_per_req) view.
    block_tables: Mapping[int, np.ndarray]
    query_start_loc: np.ndarray
    seq_lens: np.ndarray
    request_distribution: np.ndarray
    scheduled_tokens_per_dp_rank: Mapping[int, Sequence[int]]
    dp_size: int
    max_num_reqs_per_dp_rank: int
    padded_num_scheduled_tokens_per_dp_rank: int


class AttentionBackend:
    """The default "rpa" backend; see the module docstring."""

    name = "rpa"

    # Whether the warmup primer runs on all-zero seq_lens / query_start_loc,
    # a step with no active request, instead of the default dummy lengths.
    zero_warmup_lengths = False

    def check_config(self, vllm_config: "VllmConfig") -> None:
        """Raise for an engine configuration this backend cannot serve; may
        adjust it (e.g. the block size). Runs in
        TpuPlatform.check_and_update_config, before any worker starts."""

    def get_kv_cache_shape(self, total_num_pages: int, page_size: int,
                           num_kv_heads: int, head_dim: int,
                           kv_dtype: Any) -> tuple[int, ...] | None:
        """One shard's KV cache shape for num_kv_heads heads, or None for the
        RPA shape."""
        return None

    def attention(
        self,
        kv_cache: "jax.Array",
        q: "jax.Array",
        k: "jax.Array",
        v: "jax.Array",
        md: "AttentionMetadata",
        shared_md: "AttentionMetadata | SharedAttentionMetadata",
        mesh: "Mesh",
        **options: Any,
    ) -> tuple["jax.Array", "jax.Array"] | None:
        """(kv_cache, output) of attention_interface.attention, or None for
        its RPA path. options are attention()'s keyword arguments."""
        return None

    def prepare_step(self, step: HostStepInputs,
                     device_buffer: "DeviceBuffer") -> AttentionFields:
        """Host-side metadata of one step. Arrays the device needs go into
        device_buffer, which reaches the device in the step's one transfer;
        the returned fields read them back from the unpacked buffer."""
        return no_attention_fields

    def warmup_variants(self, num_tokens: int,
                        num_tokens_paddings: Sequence[int], max_num_reqs: int,
                        dp_size: int) -> list[Any]:
        """One entry per backbone executable a step at this token bucket can
        ask for; the warmup compiles each one."""
        return [None]

    def prepare_warmup(
        self, variant: Any, kv_cache_groups: "Sequence[KVCacheGroupSpec]", *,
        num_tokens: int, max_num_reqs: int, dp_size: int, mesh: "Mesh",
        sharding: "NamedSharding"
    ) -> tuple[AttentionFields, Mapping[str, "jax.Array"]]:
        """The fields of a warmup_variants entry and the dummy arrays they
        read, shaped like prepare_step's so serving reuses the executable."""
        return no_attention_fields, {}


def _create_rpa_backend() -> AttentionBackend:
    return AttentionBackend()


def _create_flywheel_backend() -> AttentionBackend:
    # Imported here so that the other backends do not load the flywheel_tpu
    # kernels.
    from tpu_inference.layers.common.flywheel_attention import \
        FlywheelAttentionBackend
    return FlywheelAttentionBackend()


ATTENTION_BACKENDS: dict[str, Callable[[], AttentionBackend]] = {
    "rpa": _create_rpa_backend,
    "flywheel": _create_flywheel_backend,
}


def _create_backend(additional_config: Mapping[str, Any], key: str,
                    default: str, backends: Mapping[str, Callable[[], Any]]):
    name = additional_config.get(key, default)
    if name not in backends:
        raise ValueError(f"Unknown additional_config[{key!r}] {name!r}; "
                         f"valid values are {tuple(backends)}.")
    return backends[name]()


def create_attention_backend(
        additional_config: Mapping[str, Any]) -> AttentionBackend:
    """The backend vllm_config.additional_config selects; raises on an
    unknown name."""
    return _create_backend(additional_config, ATTENTION_BACKEND_KEY, "rpa",
                           ATTENTION_BACKENDS)


# Process-wide, because the attention call sites and the KV cache allocator
# have no vllm_config of their own. TPUModelRunner sets it from vllm_config
# before it builds the model and the KV cache; until then it is the default
# backend, the upstream behavior for code that runs without a runner.
_backend: AttentionBackend = AttentionBackend()


def set_attention_backend(backend: AttentionBackend) -> None:
    global _backend
    _backend = backend


def get_attention_backend() -> AttentionBackend:
    return _backend


class LinearAttentionBackend:
    """The default linear attention backend; see the module docstring."""

    name = "default"

    def check_config(self, vllm_config: "VllmConfig") -> None:
        """Same contract as AttentionBackend.check_config."""

    def gdn_kernel(self) -> Callable[..., Any] | None:
        """The fused conv1d + GDN kernel run_jax_gdn_attention shard-maps,
        with the signature of kernels.gdn.v3.wrapper.fused_conv1d_gdn, or None
        for that kernel."""
        return None


def _create_default_linear_backend() -> LinearAttentionBackend:
    return LinearAttentionBackend()


def _create_flywheel_linear_backend() -> LinearAttentionBackend:
    from tpu_inference.layers.common.flywheel_linear_attention import \
        FlywheelLinearAttentionBackend
    return FlywheelLinearAttentionBackend()


LINEAR_ATTN_BACKENDS: dict[str, Callable[[], LinearAttentionBackend]] = {
    "default": _create_default_linear_backend,
    "flywheel": _create_flywheel_linear_backend,
}


def create_linear_attention_backend(
        additional_config: Mapping[str, Any]) -> LinearAttentionBackend:
    """The linear attention backend vllm_config.additional_config selects;
    raises on an unknown name."""
    return _create_backend(additional_config, LINEAR_ATTN_BACKEND_KEY,
                           "default", LINEAR_ATTN_BACKENDS)


# Process-wide for the same reason as the attention backend, and set at the
# same point.
_linear_backend: LinearAttentionBackend = LinearAttentionBackend()


def set_linear_attention_backend(backend: LinearAttentionBackend) -> None:
    global _linear_backend
    _linear_backend = backend


def get_linear_attention_backend() -> LinearAttentionBackend:
    return _linear_backend
