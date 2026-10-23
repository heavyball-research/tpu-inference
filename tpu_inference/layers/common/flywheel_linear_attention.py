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
"""The flywheel_tpu fused conv1d + GDN kernel as a linear attention
backend."""

from collections.abc import Callable

import jax
from flywheel_tpu.linear_attention.gdn import wrapper as gdn_wrapper

from tpu_inference.layers.common.attention_backend import (
    Conv1dGDNKernel, LinearAttentionBackend)

# flywheel_tpu's fused_conv1d_gdn_replay. It returns ((new_conv_state,
# new_recurrent_state, new_replay_state, new_replay_gates), output), the
# recurrent state as its int32 hi and lo planes.
Conv1dGDNReplayKernel = Callable[..., tuple[tuple[jax.Array, jax.Array,
                                                  jax.Array, jax.Array],
                                            jax.Array]]


class FlywheelLinearAttentionBackend(LinearAttentionBackend):
    """--attention-backend flywheel. The kernel takes bf16 activations, which
    FlywheelAttentionBackend.check_config requires of the model."""

    def conv1d_gdn_kernel(self) -> Conv1dGDNKernel:
        return gdn_wrapper.fused_conv1d_gdn

    def conv1d_gdn_replay_kernel(self) -> Conv1dGDNReplayKernel:
        """The kernel run_jax_gdn_replay_attention shard-maps when
        additional_config.flywheel_gdn_replay_window turns ReplaySSM decode
        on."""
        return gdn_wrapper.fused_conv1d_gdn_replay

    def replay_layout(self, n_kq: int, n_v: int, d_k: int,
                      d_v: int) -> gdn_wrapper.ReplayLayout:
        """The per-slot HBM layout of the ReplaySSM state of n_kq key/query
        heads and n_v value heads: the int32 checkpoint's shape, the hi and
        lo planes of the f32 recurrent state, (2, n_v, d_k // 2, d_v), and
        the last axes of the replay records [u | k] and of their f32 log
        decays (the replay gates)."""
        return gdn_wrapper.replay_layout(n_kq, n_v, d_k, d_v)
