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
"""attention() dispatch onto the flywheel backend, with the kernel stubbed."""

from unittest.mock import MagicMock

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.sharding import Mesh

from tpu_inference.layers.common import attention_backend, attention_interface
from tpu_inference.layers.common import flywheel_attention as fa
from tpu_inference.layers.common.attention_metadata import (
    AttentionMetadata, SharedAttentionMetadata)
from tpu_inference.runner import kv_cache as kv_cache_mod

TOTAL_TOKENS, NUM_SEQS, MAX_NUM_SEQS = 10, 2, 4
NUM_HEADS, NUM_KV_HEADS, HEAD_DIM = 24, 4, 256
NUM_BLOCKS, BLOCK_SIZE, MAX_BLOCKS_PER_SEQ = 32, 128, 8
KV_WRITE_BOUND = fa.kv_write_slice_bound(TOTAL_TOKENS, MAX_NUM_SEQS,
                                         BLOCK_SIZE)


@pytest.fixture
def mesh() -> Mesh:
    """One CPU device over the axes the dispatch path reads."""
    devices = np.array([jax.devices("cpu")[0]])
    return Mesh(devices.reshape((-1, 1, 1)), ("data", "attn_dp", "model"))


@pytest.fixture
def flywheel_flag(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(attention_backend, "selected_attention_backend",
                        fa.FlywheelAttentionBackend())


def build_metadata() -> tuple[AttentionMetadata, SharedAttentionMetadata]:
    """Metadata for a two-request prefill step of TOTAL_TOKENS tokens."""
    seq_lens = jnp.array([5, 5, 0, 0], jnp.int32)
    query_start_loc = jnp.array([0, 5, 10, 10, 10], jnp.int32)
    distribution = jnp.array([0, 0, NUM_SEQS], jnp.int32)
    md = AttentionMetadata(input_positions=jnp.arange(TOTAL_TOKENS,
                                                      dtype=jnp.int32),
                           block_tables=jnp.zeros(
                               (MAX_NUM_SEQS * MAX_BLOCKS_PER_SEQ, ),
                               jnp.int32),
                           seq_lens=seq_lens,
                           query_start_loc=query_start_loc,
                           request_distribution=distribution,
                           varlen_block_span=TOTAL_TOKENS,
                           attention_phase=fa.PREFILL_PHASE,
                           kv_write_slices=jnp.zeros((3 * KV_WRITE_BOUND, ),
                                                     jnp.int32),
                           kv_write_num_slices=jnp.zeros((1, ), jnp.int32))
    shared = SharedAttentionMetadata(input_positions=jnp.arange(
        TOTAL_TOKENS, dtype=jnp.int32),
                                     seq_lens=seq_lens,
                                     query_start_loc=query_start_loc,
                                     request_distribution=distribution)
    return md, shared


class TestDispatch:

    def run_attention(
            self, mesh: Mesh, monkeypatch: pytest.MonkeyPatch,
            **overrides: object
    ) -> tuple[MagicMock, tuple[jax.Array, jax.Array]]:
        """attention() over a stubbed sharded wrapper; returns the stub and
        what attention() returned."""
        q = jnp.ones((TOTAL_TOKENS, NUM_HEADS, HEAD_DIM), jnp.bfloat16)
        k = jnp.ones((TOTAL_TOKENS, NUM_KV_HEADS, HEAD_DIM), jnp.bfloat16)
        v = jnp.ones((TOTAL_TOKENS, NUM_KV_HEADS, HEAD_DIM), jnp.bfloat16)
        kv_cache = jnp.zeros(
            kv_cache_mod.get_kv_cache_shape_with_mesh(mesh, NUM_BLOCKS,
                                                      BLOCK_SIZE, NUM_KV_HEADS,
                                                      HEAD_DIM, jnp.bfloat16),
            jnp.bfloat16)
        wrapper = MagicMock(return_value=(jnp.ones(q.shape, q.dtype),
                                          kv_cache))
        monkeypatch.setattr(fa, "sharded_flywheel_tpu_attention", wrapper)
        md, shared = build_metadata()
        result = attention_interface.attention(
            kv_cache=kv_cache,
            q=q,
            k=k,
            v=v,
            attention_metadata=md,
            mesh=mesh,
            shared_attention_metadata=shared,
            **overrides)
        return wrapper, result

    def test_routes_to_flywheel_wrapper(self, mesh, flywheel_flag,
                                        monkeypatch):
        wrapper, (kv_cache, output) = self.run_attention(mesh, monkeypatch)
        wrapper.assert_called_once()
        args, kwargs = wrapper.call_args
        *_, kv_write_slices, kv_write_num_slices, distribution = args
        assert args[0] is mesh
        assert len(args) == 11
        assert kv_write_slices.shape == (3 * KV_WRITE_BOUND, )
        assert kv_write_num_slices.shape == (1, )
        np.testing.assert_array_equal(distribution, [0, 0, NUM_SEQS])
        assert kwargs == {
            "phase": fa.PREFILL_PHASE,
            "sm_scale": HEAD_DIM**-0.5,
            "varlen_block_span": TOTAL_TOKENS,
            "dense_batch_size": -1,
            "dense_seq_len": -1,
        }
        assert output.shape == (TOTAL_TOKENS, NUM_HEADS, HEAD_DIM)
        assert kv_cache.shape == (NUM_BLOCKS, BLOCK_SIZE, 2 * NUM_KV_HEADS,
                                  HEAD_DIM)

    @pytest.mark.parametrize(("option", "value", "expected"), [
        ("sinks", jnp.ones((NUM_HEADS, ), jnp.float32), "sinks"),
        ("attention_chunk_size", 512, "attention_chunk_size"),
        ("q_scale", 0.5, "q_scale"),
    ])
    def test_rejects_unsupported_features(self, mesh, flywheel_flag,
                                          monkeypatch, option, value,
                                          expected):
        with pytest.raises(NotImplementedError) as excinfo:
            self.run_attention(mesh, monkeypatch, **{option: value})
        assert expected in str(excinfo.value)
