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
"""--attention-backend selecting the TPU attention backends."""

from types import SimpleNamespace

import pytest
from vllm.config.attention import AttentionConfig
from vllm.utils.import_utils import resolve_obj_by_qualname
from vllm.v1.attention.backends.registry import AttentionBackendEnum

from tpu_inference.layers.common import attention_backend
from tpu_inference.layers.common import flywheel_attention as fa
from tpu_inference.layers.common import \
    flywheel_linear_attention as flywheel_linear
from tpu_inference.layers.vllm.backends import flash_attn, flywheel
from tpu_inference.platforms.tpu_platform import TpuPlatform

pytestmark = pytest.mark.skipif(
    not hasattr(AttentionBackendEnum, "register"),
    reason="vLLM before vllm#41091 cannot add a backend name")


@pytest.fixture(autouse=True)
def registered() -> None:
    attention_backend.register_vllm_attention_backends()


def make_vllm_config(backend: str | None):
    return SimpleNamespace(attention_config=AttentionConfig(backend=backend))


def test_register_lets_vllm_parse_the_name():
    # Idempotent: vLLM calls the plugin hook in every process, and more than
    # once in one.
    attention_backend.register_vllm_attention_backends()

    assert (AttentionConfig.validate_backend_before("flywheel")
            is AttentionBackendEnum.FLYWHEEL)
    assert (AttentionBackendEnum.FLYWHEEL.get_path() ==
            attention_backend.VLLM_ATTENTION_BACKENDS["flywheel"])


@pytest.mark.parametrize(("backend", "name"), [
    ("flywheel", "flywheel"),
    (None, "rpa"),
    ("FLASH_ATTN", "rpa"),
    ("TRITON_ATTN", "rpa"),
])
def test_attention_backend_name(backend, name):
    assert attention_backend.attention_backend_name(
        make_vllm_config(backend)) == name


def test_one_name_selects_both_backends():
    assert isinstance(attention_backend.create_attention_backend("flywheel"),
                      fa.FlywheelAttentionBackend)
    assert isinstance(
        attention_backend.create_linear_attention_backend("flywheel"),
        flywheel_linear.FlywheelLinearAttentionBackend)
    assert type(attention_backend.create_linear_attention_backend(
        "rpa")) is attention_backend.LinearAttentionBackend


@pytest.mark.parametrize(("backend", "use_mla", "expected"), [
    ("FLYWHEEL", False, flywheel.FlywheelPallasAttentionBackend),
    ("FLASH_ATTN", False, flash_attn.PallasAttentionBackend),
    ("TRITON_ATTN", False, flash_attn.PallasAttentionBackend),
    ("FLYWHEEL", True, None),
])
def test_platform_selects_the_adapter(backend, use_mla, expected):
    path = TpuPlatform.get_attn_backend_cls(AttentionBackendEnum[backend],
                                            SimpleNamespace(use_mla=use_mla))

    if expected is None:
        assert path == AttentionBackendEnum.FLASH_ATTN_MLA.get_path()
    else:
        assert resolve_obj_by_qualname(path) is expected


def test_adapter_impls_name_their_backend():
    assert flash_attn.PallasAttentionBackendImpl.attention_backend == "rpa"
    adapter = flywheel.FlywheelPallasAttentionBackend
    assert adapter.get_name() == "FLYWHEEL"
    assert adapter.get_impl_cls().attention_backend == "flywheel"


@pytest.mark.parametrize(("env", "value"), [
    ("TPU_MULTIHOST_BACKEND", "ray"),
    ("VLLM_WORKER_MULTIPROC_METHOD", "spawn"),
])
def test_rejects_a_config_that_crosses_a_process_boundary(
        monkeypatch, env, value):
    monkeypatch.setenv(env, value)

    with pytest.raises(ValueError, match=f"{env}={value}"):
        attention_backend.check_attention_backend_config(
            make_vllm_config("flywheel"))


def test_default_backend_skips_the_process_check(monkeypatch):
    monkeypatch.setenv("TPU_MULTIHOST_BACKEND", "ray")

    attention_backend.check_attention_backend_config(make_vllm_config(None))
