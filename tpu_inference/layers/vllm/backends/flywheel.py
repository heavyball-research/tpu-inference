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
"""The vLLM adapter that TpuPlatform.get_attn_backend_cls returns for
--attention-backend flywheel: PallasAttentionBackend running the flywheel
attention backend."""

from tpu_inference.layers.vllm.backends.flash_attn import (
    PallasAttentionBackend, PallasAttentionBackendImpl)


class FlywheelPallasAttentionBackendImpl(PallasAttentionBackendImpl):

    attention_backend = "flywheel"


class FlywheelPallasAttentionBackend(PallasAttentionBackend):

    @staticmethod
    def get_name() -> str:
        return "FLYWHEEL"

    @staticmethod
    def get_impl_cls() -> type[FlywheelPallasAttentionBackendImpl]:
        return FlywheelPallasAttentionBackendImpl
