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

from typing import TYPE_CHECKING

import torch
from flywheel_tpu.linear_attention.gdn import wrapper as gdn_wrapper

from tpu_inference.layers.common.attention_backend import (
    GDNKernel, LinearAttentionBackend)

if TYPE_CHECKING:
    from vllm.config import VllmConfig


class FlywheelLinearAttentionBackend(LinearAttentionBackend):
    """additional_config["linear_attn_backend"] = "flywheel"."""

    name = "flywheel"

    def check_config(self, vllm_config: "VllmConfig") -> None:
        # Note (david): the kernel takes bf16 activations and raises on any
        # other dtype at trace time.
        model_dtype = vllm_config.model_config.dtype
        if model_dtype != torch.bfloat16:
            raise ValueError(
                "The flywheel linear attention backend cannot serve this "
                f"configuration: model dtype {model_dtype} is not bfloat16.")

    def gdn_kernel(self) -> GDNKernel:
        return gdn_wrapper.fused_conv1d_gdn
