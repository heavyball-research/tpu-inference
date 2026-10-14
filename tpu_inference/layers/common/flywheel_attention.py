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
"""FlyWheel TPU attention backend
(additional_config["attention_backend"] = "flywheel")."""

import math
import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import jax
import jax.numpy as jnp
import numpy as np
from flywheel_tpu import flash_attn_func
from flywheel_tpu import flash_attn_varlen_func as _flash_attn_varlen_func
from flywheel_tpu import flash_attn_with_kvcache
from flywheel_tpu.pallas.kv_cache_write import write_kv_cache_pages
from jax import lax
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P
from vllm.v1.kv_cache_interface import (AttentionSpec, KVCacheGroupSpec,
                                        UniformTypeKVCacheSpecs)

from tpu_inference.layers.common.attention_backend import (AttentionBackend,
                                                           AttentionFields,
                                                           HostStepInputs)
from tpu_inference.layers.common.attention_metadata import (
    AttentionMetadata, SharedAttentionMetadata)
from tpu_inference.layers.common.sharding import ShardingAxisName
from tpu_inference.logger import init_logger
from tpu_inference.utils import (DeviceBuffer, align_to, device_array,
                                 get_mesh_shape_product)

if TYPE_CHECKING:
    from vllm.config import VllmConfig

logger = init_logger(__name__)

# Note (david): flash_attn_with_kvcache and the KV writer carry their own
# jax.jit, so every layer of a backbone calls one shared traced kernel.
# flash_attn_varlen_func did not, and each layer traced and lowered its own
# inlined copy: a 36-layer Qwen3-4B attention stack at the 4096-token bucket
# on v6e-1 lowered to 64 MB of StableHLO in 22 s for a span-2048 prefill,
# Python time a warm persistent compile cache cannot skip. Shared, it lowers
# to 2.3 MB in 1.0 s, and its cold compile drops from 71 s to 34 s, at an
# unchanged step time (30.3 ms ragged prefill).
flash_attn_varlen_func = jax.jit(
    _flash_attn_varlen_func,
    static_argnames=("max_seqlen_q", "max_seqlen_k", "softmax_scale", "causal",
                     "interpret", "token_major"),
)

# Note (david): one KV cache layout, the merged (num_pages, page_size,
# 2 * kv_heads, head_dim) whose token rows interleave each KV head's K row and
# V row, [k0, v0, k1, v1, ...] (RPA v3's head order). It has no singleton axis
# at one KV head per chip (a stacked (2, ..., 1, head_dim) one did, which XLA
# laid out so that every written layer paid two whole-cache copies), and its
# leading axis is the page axis. One head's (K, V) row pair is a 2-row window
# of the head axis, so the paged prefill kernel stages only the KV head each
# q head reads.


def _padded_head_dim(head_dim: int) -> int:
    if not 0 < head_dim <= 256:
        raise NotImplementedError(
            "The flywheel attention backend needs 0 < head_dim <= 256; "
            f"got {head_dim}.")
    return align_to(head_dim, 128)


# Note (david): the decode kernel runs only at the head_dims it needs no
# padding for. At a padded head_dim with 2 KV heads per chip its rows come
# out near zero (test_padded_head_dim_numerics, decode-2-{64,96,129,192,255}).
DECODE_HEAD_DIMS = (128, 256)


def check_decode_head_dim(head_dim: int) -> None:
    if head_dim not in DECODE_HEAD_DIMS:
        raise NotImplementedError(
            f"The flywheel attention backend decodes only head_dim in "
            f"{DECODE_HEAD_DIMS}; got {head_dim}.")


def get_kv_cache_shape(
    total_num_pages: int,
    page_size: int,
    actual_num_kv_heads: int,
    actual_head_dim: int,
    kv_dtype: jnp.dtype,
) -> tuple[int, ...]:
    """Shape of one shard's merged K/V page pool for actual_num_kv_heads KV
    heads. bf16 only, so kv_dtype does not size it."""
    head_dim = _padded_head_dim(actual_head_dim)
    # Note (david): the kernels read a KV head's (K, V) row pair as one u32
    # word per lane, so any KV head count per chip works.
    if actual_num_kv_heads < 1:
        raise ValueError(
            "The flywheel attention backend needs at least one KV head per chip; "
            f"got {actual_num_kv_heads}.")
    return (total_num_pages, page_size, 2 * actual_num_kv_heads, head_dim)


def kv_write_slice_bound(num_tokens: int, max_num_seqs: int,
                         page_size: int) -> int:
    """Upper bound on the page pieces one step can write, which sizes the
    host slice table.

    A request writing q_len tokens touches at most q_len // page_size + 2
    pages, so the bound depends only on the token bucket and the request
    slots, never on the batch.
    """
    return num_tokens // page_size + 2 * max_num_seqs


def build_kv_write_slices(
        cu_q_lens: np.ndarray,  # i32[num_reqs + 1]
        kv_lens: np.ndarray,  # i32[num_reqs], includes this step's tokens
        block_table: np.ndarray,  # i32[num_reqs, pages_per_seq]
        *,
        page_size: int,
        out: np.ndarray,  # i32[3, bound], overwritten
) -> int:
    """Fill out with the token slices that copy this step's new K/V rows into
    the paged cache and return their count.

    Column j < count is (cache_row_start, new_row_start, length) in rows of
    the flat (num_pages * page_size, ...) cache view, each piece inside one
    page; the remaining columns are zero.
    """
    # Note (david): host-side numpy on purpose. Deriving the same table on the
    # device scanned every (request slot, page) pair with three scatters,
    # ~1 ms per layer at 128 request slots, more than the attention kernel
    # itself at short sequence lengths.
    num_reqs = kv_lens.shape[0]
    out[:] = 0
    if num_reqs == 0:
        return 0
    q_lens = cu_q_lens[1:num_reqs + 1] - cu_q_lens[:num_reqs]
    kv_start = kv_lens - q_lens
    active = q_lens > 0
    if np.any(kv_start[active] < 0):
        raise ValueError(
            "kv_lens must include this step's tokens: got kv_len < q_len for "
            f"requests {np.nonzero(active & (kv_start < 0))[0].tolist()}.")
    first_page = kv_start // page_size
    last_page = (kv_lens - 1) // page_size
    counts = np.where(active, last_page - first_page + 1, 0)
    total = int(counts.sum())
    if total > out.shape[1]:
        raise ValueError(
            f"{total} page pieces exceed the slice table's {out.shape[1]} "
            "columns; size it with kv_write_slice_bound.")
    if total == 0:
        return 0
    req_idx = np.repeat(np.arange(num_reqs), counts)
    piece_start = np.repeat(np.cumsum(counts) - counts, counts)
    page_slot = first_page[req_idx] + (np.arange(total) - piece_start)
    page_lo = page_slot * page_size
    lo = np.maximum(kv_start[req_idx], page_lo)
    hi = np.minimum(kv_lens[req_idx], page_lo + page_size)
    out[0, :total] = (block_table[req_idx, page_slot] * page_size +
                      (lo - page_lo))
    out[1, :total] = cu_q_lens[:num_reqs][req_idx] + (lo - kv_start[req_idx])
    out[2, :total] = hi - lo
    return total


# Note (david): these kernels tile the KV token axis in 128-token blocks, so a
# page that is not a whole number of blocks cannot be indexed.
PAGE_SIZE_MULTIPLE = 128

# Note (david): these are the values AttentionMetadata.attention_phase takes
# on this backend, and each one is a separate static executable.
PREFILL_PHASE = "prefill"
DECODE_PHASE = "decode"
EXTEND_PHASE = "extend"
ATTENTION_PHASES = (PREFILL_PHASE, DECODE_PHASE, EXTEND_PHASE)

DENSE_MIN_SEQ_LEN = 128

# Note (david): the dense kernel's largest physical block. A longer axis that
# is not a whole number of these falls to the largest candidate dividing it
# (128 at a 131200-token bucket), a schedule the cost model below does not
# describe, so such a bucket stays ragged.
DENSE_MAX_BLOCK = 2048

# Note (david): flash_attn_with_kvcache reads the cache in the largest
# KV block up to its VMEM target that divides the row capacity
# pages_per_seq * page_size, so each row is padded until every power-of-two
# block up to this many tokens divides it. A prime page count (257 pages for
# max_model_len 32801) left only one-page blocks, which made a decode step
# about twice RPA v3's at 64K KV tokens on v6e.
DECODE_CAPACITY_ALIGN_TOKENS = 4096


def decode_block_table_pages(pages_per_seq: int, page_size: int) -> int:
    """Width, in pages, of the block table handed to the decode kernel: each
    row padded so blocks up to DECODE_CAPACITY_ALIGN_TOKENS divide it."""
    # Note (david): the padding entries are only ever read for the
    # zero-length tail copies of a row's last block (the kernel clips each
    # page copy to what is left of the row), so page 0 is a safe filler.
    align_pages = math.lcm(DECODE_CAPACITY_ALIGN_TOKENS,
                           page_size) // page_size
    return align_to(pages_per_seq, align_pages)


def phase_from_request_counts(num_decode: int, num_reqs: int) -> str:
    """Classify request counts alone: all decode, no decode, or mixed.

    A mixed batch is EXTEND. A batch without decode rows is reported as
    PREFILL, which select_attention_phase still turns into EXTEND when a
    request has a cached prefix, something the counts cannot tell.
    """
    if not 0 <= num_decode <= num_reqs:
        raise ValueError(
            f"Invalid num_decode={num_decode}, num_reqs={num_reqs}.")
    if 0 < num_decode < num_reqs:
        return EXTEND_PHASE
    elif num_decode == num_reqs:
        return DECODE_PHASE
    else:
        return PREFILL_PHASE


def select_attention_phase(num_decode: int, cu_q_lens: np.ndarray,
                           kv_lens: np.ndarray) -> str:
    """A rank's static kernel choice, from this step's own host lengths.

    PREFILL is a step of fresh prefill requests alone (kv_len == q_len and no
    decode row), which the prefill kernels serve from this step's K/V. A
    mixed step, or a request whose keys are partly in the pages, is EXTEND.
    """
    if cu_q_lens.shape != (kv_lens.size + 1, ) or cu_q_lens[0] != 0:
        raise ValueError("cu_q_lens must start at zero and match kv_lens.")
    q_lens = np.diff(cu_q_lens)
    if np.any(q_lens <= 0) or np.any(kv_lens < q_lens):
        raise ValueError("Active requests require 0 < q_len <= kv_len.")
    phase = phase_from_request_counts(num_decode, kv_lens.size)
    if kv_lens.size == 0:
        return PREFILL_PHASE
    elif phase == DECODE_PHASE and np.any(q_lens != 1):
        raise ValueError("Decode requires exactly one query per request.")
    elif phase == DECODE_PHASE:
        return DECODE_PHASE
    elif np.any(q_lens[:num_decode] != 1):
        # Note (david): a mixed step hands the leading num_decode requests to
        # the one-token-per-row decode kernel, so a regression in the runner's
        # reordering has to fail here instead of silently attending wrong
        # rows.
        raise ValueError(
            "A mixed batch must put its decode requests first, one query "
            f"each; got q_lens {q_lens[:num_decode].tolist()} for the "
            f"leading {num_decode}.")
    elif phase == EXTEND_PHASE or np.any(kv_lens != q_lens):
        # Note (david): a mixed step is EXTEND even when its prefill requests
        # are fresh. The ragged kernel served it by merging the decode rows
        # into its own [T, N, D] output, a read and a write of one buffer that
        # XLA cannot do in place: the second copy per layer made its
        # memory-space assignment drop the MLP weight prefetch into VMEM on 56
        # of 64 layers, +176.8 ms of a 3311.7 ms step with Qwen3.8-27B at TP 8
        # on v6e-8. The price is the paged cache kernel on every serving
        # prefill step (with the extend kernel it replaced, a 320-prompt
        # benchmark on v6e-1 with Qwen3-4B went from 113.80 s to 146.74 s).
        return EXTEND_PHASE
    else:
        return PREFILL_PHASE


def combine_attention_phases(phases: set[str]) -> str:
    """Use one static executable across DP ranks: ranks that disagree share
    the mixed one."""
    if not phases.issubset(ATTENTION_PHASES):
        raise ValueError(f"Unknown attention phases: {phases}.")
    if not phases:
        return PREFILL_PHASE
    elif len(phases) == 1:
        return next(iter(phases))
    else:
        return EXTEND_PHASE


def dense_prefill_shape(
    scheduled_tokens_per_dp_rank: Sequence[Sequence[int]],
    padded_num_tokens_per_dp_rank: int,
) -> tuple[int, int]:
    """(batch, seq_len) for the dense prefill kernel, or (-1, -1) for ragged.

    Dense needs every DP rank to schedule the same power-of-two count of
    equal, power-of-two length requests filling the rank's token bucket, with
    seq_len at least DENSE_MIN_SEQ_LEN. One request per rank is dense at any
    length, as the (1, bucket) row single_request_dense_seq_len describes,
    whenever the cost model puts that row under the ragged kernel.
    """
    # Note (david): the pair is static metadata and each one compiles its own
    # backbone executable, so the set has to stay bounded by the bucket ladder
    # (at most log2(bucket / DENSE_MIN_SEQ_LEN) + 1 shapes) instead of
    # following the traffic.
    if len(scheduled_tokens_per_dp_rank) and all(
            len(lengths) == 1 and lengths[0] > 0
            for lengths in scheduled_tokens_per_dp_rank):
        seq_len = single_request_dense_seq_len(padded_num_tokens_per_dp_rank)
        if seq_len < 0:
            return -1, -1
        # Note (david): both paths pay for the bucket's pad tail, the dense
        # row inside its own causal mask and the ragged kernel as a trailing
        # sequence, so the crossover sits near the top of the bucket instead
        # of at it: measured on one v6e chip with 3 query heads of head_dim
        # 256, one request is faster ragged up to 0.90 of the 98304 bucket
        # (19.07 ms against 19.81 ms dense) and faster dense from 0.95 on
        # (20.60 ms against 19.81 ms). The model puts it at 0.94 of either
        # bucket and picked the faster path on every one of the 42 measured
        # one-request steps.
        ragged_cost = _ragged_cost(scheduled_tokens_per_dp_rank,
                                   padded_num_tokens_per_dp_rank)
        if _dense_cost(seq_len) > ragged_cost:
            return -1, -1
        return 1, seq_len
    shapes = set()
    for lengths in scheduled_tokens_per_dp_rank:
        if (not len(lengths) or any(length <= 0 for length in lengths)
                or len(set(lengths)) != 1):
            return -1, -1
        shapes.add((len(lengths), int(lengths[0])))
    if len(shapes) != 1:
        return -1, -1
    batch, seq_len = shapes.pop()
    is_power_of_two = ((batch & (batch - 1)) == 0
                       and (seq_len & (seq_len - 1)) == 0)
    if (batch * seq_len != padded_num_tokens_per_dp_rank
            or seq_len < DENSE_MIN_SEQ_LEN or not is_power_of_two):
        return -1, -1
    return batch, seq_len


# Note (david): the ragged kernel's own physical block sizes minus 128, which
# measured slower than 256 on every sampled batch (v6e). A span is passed as
# max_seqlen_q/k, which the interface turns into the q/kv block cap.
VARLEN_BLOCK_SPANS = (256, 512, 1024, 2048)

# Note (david): fixed cost of one ragged kernel step, in units of the compute
# of a VARLEN_STEP_OVERHEAD_TOKENS^2 tile. Calibrated on one v6e chip by
# timing the kernel alone over 19 request mixes x 4 spans x 3 token buckets
# per head config, bf16: the per-step cost fits a * (span^2 + c^2) with
# c = 762 for 3 query heads of head_dim 256 (Qwen3.8-27B at TP 8, per chip)
# and c = 698 for 32 query heads of head_dim 128 (Qwen3-4B at TP 1), even
# though a itself differs 10.4x between the two. So the overhead is a
# property of the schedule, not of the head shape, and the host functions
# need no head counts. Scored by regret over both grids the optimum is 704,
# on a plateau from 672 to 832.
VARLEN_STEP_OVERHEAD_TOKENS = 704


def varlen_block_spans(padded_num_tokens_per_dp_rank: int) -> tuple[int, ...]:
    """Every block span varlen_block_span can return at this token bucket,
    which is the set of ragged prefill backbones the warmup compiles."""
    # Note (david): one unwarmed backbone variant costs 20-106 s to compile on
    # v6e-1 with Qwen3-4B, a stall of that length mid-serving; warming every
    # span instead of one per bucket took engine startup up by 0.8%.
    spans = tuple(s for s in VARLEN_BLOCK_SPANS
                  if s <= padded_num_tokens_per_dp_rank)
    return spans or (padded_num_tokens_per_dp_rank, )


def varlen_block_span(
    scheduled_tokens_per_dp_rank: Sequence[Sequence[int]],
    padded_num_tokens_per_dp_rank: int,
) -> int:
    """The varlen_block_spans entry minimising the modelled ragged kernel cost
    of the step (one executable serves every DP rank); ties go wider."""
    if not any(length > 0 for rank_lengths in scheduled_tokens_per_dp_rank
               for length in rank_lengths):
        raise ValueError("a prefill step schedules at least one token; got "
                         f"{scheduled_tokens_per_dp_rank}.")
    spans = varlen_block_spans(padded_num_tokens_per_dp_rank)
    return min(spans,
               key=lambda span:
               (_ragged_step_cost(scheduled_tokens_per_dp_rank,
                                  padded_num_tokens_per_dp_rank, span), -span))


def _ragged_step_cost(
    scheduled_tokens_per_dp_rank: Sequence[Sequence[int]],
    padded_num_tokens_per_dp_rank: int,
    span: int,
) -> int:
    """Modelled cost of one ragged prefill step at this block span."""
    # Note (david): the DP ranks run their own kernel call concurrently and
    # the step ends with the slowest, so the step costs the maximum and not
    # the sum. With one rank, which is every case measured here, the two are
    # the same number.
    return max(
        _varlen_cost(lengths, padded_num_tokens_per_dp_rank, span)
        for lengths in scheduled_tokens_per_dp_rank)


def _ragged_cost(
    scheduled_tokens_per_dp_rank: Sequence[Sequence[int]],
    padded_num_tokens_per_dp_rank: int,
) -> int:
    """Modelled cost of one ragged prefill step at the span it would take."""
    return min(
        _ragged_step_cost(scheduled_tokens_per_dp_rank,
                          padded_num_tokens_per_dp_rank, span)
        for span in varlen_block_spans(padded_num_tokens_per_dp_rank))


def _varlen_cost(lengths: Sequence[int], padded_num_tokens: int,
                 span: int) -> int:
    """Modelled cost of one DP rank's ragged kernel call: its own request
    lengths plus the pad tail of its token bucket."""
    # Note (david): a sequence of t blocks visits the t (t + 1) / 2
    # (q block, kv block) pairs of its causal block-diagonal, so many
    # short sequences push the span down and one long one pushes it up.
    blocks = [-(-length // span) for length in lengths if length > 0]
    steps = sum(t * (t + 1) // 2 for t in blocks)
    # Note (david): flash_attn_varlen_func hands the unfilled tail
    # [sum(lengths), bucket) to a trailing sequence of its own with a square
    # window (flywheel_tpu/pallas/seqlen_info.py), so the tail walks
    # a block-diagonal of its own and an under-filled bucket is at most 2x
    # cheaper than a full one, never unboundedly cheaper. Measured on one v6e
    # chip at the 131072 bucket, span 2048, 3 query heads of head_dim 256: one
    # 65536-token request takes 19.88 ms for 528 real plus 528 tail steps,
    # against 38.94 ms for a request filling the bucket and 37.60 ms for a
    # 2048-token one (1 real plus 2016 tail steps). Leaving the tail out cost
    # up to 1.40x and 2.19x the best span on the two head configs.
    tail = padded_num_tokens - sum(lengths)
    if tail < 0:
        raise ValueError(
            f"a rank's {sum(lengths)} scheduled tokens overflow its "
            f"{padded_num_tokens}-token bucket.")
    if tail:
        tail_blocks = -(-tail // span)
        steps += tail_blocks * (tail_blocks + 1) // 2
    return steps * (span * span + VARLEN_STEP_OVERHEAD_TOKENS**2)


def _dense_cost(seq_len: int) -> int:
    """_varlen_cost's model for one dense causal row of seq_len tokens."""
    # Note (david): the dense schedule is built on the host, so a step costs
    # its tile alone. Measured, the dense kernel's per-step cost is 0.0168 ms
    # (3 query heads, head_dim 256) and 0.1743 ms (32 heads, head_dim 128)
    # against a ragged 2048-block step's 0.0185 ms and 0.1892 ms, so its own
    # overhead would be about 364 tokens either way -- a term small enough
    # that adding it changed no decision in the measured grid, and one that
    # only made the crossover worse on 3 heads, so this stays bare.
    block = min(seq_len, DENSE_MAX_BLOCK)
    blocks = seq_len // block
    return blocks * (blocks + 1) // 2 * block * block


def single_request_dense_seq_len(padded_num_tokens_per_dp_rank: int) -> int:
    """seq_len of the (1, seq_len) dense row a one-request prefill step takes
    at this token bucket, or -1 when the dense kernel cannot tile the bucket.
    """
    # Note (david): the runner packs requests end to end and pads the bucket's
    # tail, so one request is already a dense row: the causal mask keeps every
    # real query off the padded keys, and the KV writer copies real rows only.
    # seq_len is the bucket itself, which adds one shape per bucket and needs
    # no power of two.
    seq_len = padded_num_tokens_per_dp_rank
    if seq_len < DENSE_MIN_SEQ_LEN or seq_len % PAGE_SIZE_MULTIPLE:
        return -1
    if seq_len > DENSE_MAX_BLOCK and seq_len % DENSE_MAX_BLOCK:
        return -1
    return seq_len


def write_kv_cache(
    k: jax.Array,
    v: jax.Array,
    kv_cache: jax.Array,
    kv_write_slices: jax.Array,
    kv_write_num_slices: jax.Array,
) -> jax.Array:
    """Copy this step's K/V rows into their pages as interleaved
    [k0, v0, k1, v1, ...] cache rows; return the same buffer.

    k and v are (T, H, D), kv_cache (P, S, 2H, D); kv_write_slices is
    i32[3, bound] and kv_write_num_slices i32[1].
    """
    # Note (david): the fused writer interleaves K and V in VMEM because an
    # XLA stack before a one-run write costs two extra HBM passes over them:
    # +16 to +94% of the write on v6e (h32 k32 d256, 1024 tokens: 0.173
    # against 0.090 ms per layer), while the fused writer measured 6 to 57%
    # below the blocked layout's two runs.
    return write_kv_cache_pages(k, v, kv_cache, kv_write_slices,
                                kv_write_num_slices)


def fold_heads_outer(x: jax.Array, batch_size: int, seq_len: int) -> jax.Array:
    """[T, H, D] -> (H, batch_size, seq_len, D), flash_attn_func's heads-outer
    layout."""
    # Note (david): the projections are written as [T, N, D] einsums, but XLA
    # gives their outputs, and the per-head norms and K's RoPE after them, the
    # physical layout {2,0,1}, heads outermost, so in effect they produce the
    # row-major [N, T, D] the kernel reads. The transpose(1, 0, 2) here then
    # compiles to a bitcast and the reshape only splits the contiguous token
    # axis. The batch-outer layout splits the token axis with heads in between
    # and cost three copies per layer, ~0.3 ms on v6e with Qwen3-4B at batch
    # 8 x 1024 tokens, for a bitwise identical result.
    heads, head_dim = x.shape[1:]
    return x.transpose(1, 0, 2).reshape(heads, batch_size, seq_len, head_dim)


def unfold_heads_outer(out: jax.Array) -> jax.Array:
    """(H, batch_size, seq_len, D) -> [T, H, D], the inverse of
    fold_heads_outer."""
    # Note (david): left as a trailing 3-D transpose that XLA folds into the
    # consumer (o_proj) instead of a copy.
    heads, batch_size, seq_len, head_dim = out.shape
    return out.reshape(heads, batch_size * seq_len,
                       head_dim).transpose(1, 0, 2)


def flywheel_tpu_dense_prefill(
    q: jax.Array,  # [T, N, D]
    k: jax.Array,  # [T, H, D]
    v: jax.Array,  # [T, H, D]
    kv_cache: jax.Array,
    kv_write_slices: jax.Array,
    kv_write_num_slices: jax.Array,
    *,
    batch_size: int,
    seq_len: int,
    sm_scale: float,
) -> tuple[jax.Array, jax.Array]:
    """Dense batched prefill on a static (batch_size, seq_len) with the
    head-major FlyWheel kernel."""
    total_tokens = q.shape[0]
    if (total_tokens != batch_size * seq_len or k.shape[0] != total_tokens
            or v.shape != k.shape):
        raise ValueError("Dense prefill requires equal Q/K/V batch lengths.")
    output = flash_attn_func(
        fold_heads_outer(q, batch_size, seq_len),
        fold_heads_outer(k, batch_size, seq_len),
        fold_heads_outer(v, batch_size, seq_len),
        causal=True,
        softmax_scale=sm_scale,
        token_major=False,
        heads_outer_fold=True,
    )
    cache = write_kv_cache(k, v, kv_cache, kv_write_slices,
                           kv_write_num_slices)
    return unfold_heads_outer(output), cache


def flywheel_tpu_varlen_prefill(
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    kv_cache: jax.Array,
    cu_q_lens: jax.Array,
    kv_write_slices: jax.Array,
    kv_write_num_slices: jax.Array,
    *,
    sm_scale: float,
    varlen_block_span: int = -1,
    dense_batch_size: int = -1,
    dense_seq_len: int = -1,
) -> tuple[jax.Array, jax.Array]:
    """Prefill with the dense kernel on a static (batch, seq_len), otherwise
    with the ragged one at the host-chosen block span.

    Every request is a fresh prefill one: select_attention_phase sends a step
    with a decode row or a cached prefix to extend, so both kernels attend
    this step's own K/V and the cache is only written.
    """
    if dense_batch_size > 0 and dense_seq_len > 0:
        return flywheel_tpu_dense_prefill(
            q,
            k,
            v,
            kv_cache,
            kv_write_slices,
            kv_write_num_slices,
            batch_size=dense_batch_size,
            seq_len=dense_seq_len,
            sm_scale=sm_scale,
        )
    if not 0 < varlen_block_span <= q.shape[0]:
        raise ValueError(
            "ragged prefill needs the host-chosen block span "
            "(AttentionMetadata.varlen_block_span, see the varlen_block_span "
            f"helper) in [1, {q.shape[0]}]; got {varlen_block_span}.")
    cache = write_kv_cache(k, v, kv_cache, kv_write_slices,
                           kv_write_num_slices)
    # Note (david): the kernel's output is returned as is. Writing rows back
    # into it (a decode-row merge used to) reads and writes one buffer, which
    # XLA answers with a second [T, N, D] copy per layer; see
    # select_attention_phase for what that copy cost. The head-major API takes
    # (N, T, D).
    output = flash_attn_varlen_func(
        q.transpose(1, 0, 2),
        k.transpose(1, 0, 2),
        v.transpose(1, 0, 2),
        cu_q_lens,
        cu_q_lens,
        max_seqlen_q=varlen_block_span,
        max_seqlen_k=varlen_block_span,
        causal=True,
        softmax_scale=sm_scale,
        token_major=False,
    )
    return output.transpose(1, 0, 2), cache


def paged_cache_decode(
    q: jax.Array,  # [T, N, D]
    k: jax.Array | None,  # [T, H, D], or None when the cache already holds
    v: jax.Array | None,  # this step's tokens
    kv_cache: jax.Array,
    kv_lens: jax.Array,  # i32[max_num_seqs], includes this step's token
    page_indices: jax.Array,  # i32[max_num_seqs * pages_per_seq]
    num_active: jax.Array,  # i32 scalar: leading rows that are requests
    *,
    sm_scale: float,
) -> tuple[jax.Array, jax.Array]:
    """One query token per row against the paged cache, for the leading
    min(max_num_seqs, T) rows of the token bucket; returns those rows, NOT
    padded back to T, and the cache the kernel aliased through.

    Passing k/v lets the kernel append them to the pages itself; None means
    the DMA writer has already copied this step's tokens there.
    """
    page_size = kv_cache.shape[1]
    max_num_seqs = kv_lens.shape[0]
    pages_per_seq = page_indices.shape[0] // max_num_seqs
    batch = min(max_num_seqs, q.shape[0])
    block_table = page_indices[:batch * pages_per_seq].reshape(
        batch, pages_per_seq)
    padded_pages = decode_block_table_pages(pages_per_seq, page_size)
    if padded_pages != pages_per_seq:
        block_table = jnp.pad(block_table,
                              ((0, 0), (0, padded_pages - pages_per_seq)))
    has_new_kv = k is not None
    # Note (david): kv_lens count this step's token, so an in-kernel append
    # takes the length before it (padding rows go to -1, which the kernel
    # never reads); without the append that token is in its page already.
    cache_seqlens = kv_lens[:batch] - 1 if has_new_kv else kv_lens[:batch]
    # Note (david): num_active bounds the rows that are real requests; the
    # kernel reads nothing for the rest and leaves their output unspecified.
    # Unbounded, every padding row cost a KV block iteration plus a tile
    # write, ~0.14 ms per row per step on v6e with Qwen3-4B.
    output, cache = flash_attn_with_kvcache(
        q[:batch, None],
        kv_cache,
        None,
        k[:batch, None] if has_new_kv else None,
        v[:batch, None] if has_new_kv else None,
        cache_seqlens=cache_seqlens,
        block_table=block_table,
        num_active=num_active,
        softmax_scale=sm_scale,
    )
    return output[:, 0], cache


def flywheel_tpu_decode(
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    kv_cache: jax.Array,
    kv_lens: jax.Array,
    page_indices: jax.Array,
    num_active: jax.Array,  # i32 scalar: leading rows that are requests
    *,
    sm_scale: float,
) -> tuple[jax.Array, jax.Array]:
    """Paged decode attention for a decode-only batch, zero-filling the rows
    of the token bucket that no request slot covers.

    The kernel appends each row's new K/V itself, so the DMA writer and its
    slice table are prefill-only.
    """
    output, cache = paged_cache_decode(q,
                                       k,
                                       v,
                                       kv_cache,
                                       kv_lens,
                                       page_indices,
                                       num_active,
                                       sm_scale=sm_scale)
    padding_rows = q.shape[0] - output.shape[0]
    if padding_rows == 0:
        return output, cache
    zeros = jnp.zeros((padding_rows, q.shape[1], q.shape[2]), q.dtype)
    return jnp.concatenate([output, zeros]), cache


def flywheel_tpu_extend(
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    kv_cache: jax.Array,
    kv_lens: jax.Array,
    page_indices: jax.Array,
    cu_q_lens: jax.Array,
    kv_write_slices: jax.Array,
    kv_write_num_slices: jax.Array,
    num_decode: jax.Array,  # i32 scalar: leading one-token requests
    num_active: jax.Array,  # i32 scalar: leading rows that are requests
    *,
    sm_scale: float,
) -> tuple[jax.Array, jax.Array]:
    """Append only the scheduled tokens, then attend to their entire cached
    prefix; the phase for a prefill request whose keys are partly in the pages.

    A mixed step splits over the same written cache: its leading num_decode
    one-token rows go to the decode kernel, the prefill requests to the paged
    varlen kernel, over a request window slid past them.
    """
    cache = write_kv_cache(k, v, kv_cache, kv_write_slices,
                           kv_write_num_slices)
    # Note (david): the decode rows are batched in the decode kernel; a
    # kernel that walks requests serially costs the same per request whatever
    # its query length (21 decode rows at kv_len 7800: 1.378 ms per Qwen3-4B
    # layer on v6e against 0.448 ms batched). The cache is threaded writer ->
    # decode kernel so the one donated buffer aliases the chain; the varlen
    # kernel only reads it.
    # Note (david): KNOWN ISSUE, TP 4, seen on the blocked [K heads, V heads]
    # cache layout this interleaved one replaced. With 2 KV heads per chip
    # (Qwen3-4B at TP 4 on v6e-8) the decode rows of a mixed step came out
    # wrong: six concurrent requests under 2048-token chunked prefill left TP
    # 1's greedy tokens at the first token decoded in a mixed step. TP 1, 2
    # and 8 matched. The rows were right when this kernel read a materialized
    # copy of the written cache (cache * a runtime one behind an optimization
    # barrier); a barrier alone or a one-run writer did not help. Root cause
    # open; recheck before serving several requests at 2 KV heads per chip.
    decode_out, cache = paged_cache_decode(q,
                                           None,
                                           None,
                                           cache,
                                           kv_lens,
                                           page_indices,
                                           num_decode,
                                           sm_scale=sm_scale)
    page_size = cache.shape[1]
    max_num_seqs = kv_lens.shape[0]
    block_table = page_indices.reshape(max_num_seqs, -1)
    pages_per_seq = block_table.shape[1]
    # Note (david): the window has to MOVE past the decode requests. Leaving
    # them in place with their query ranges clamped to zero tokens still cost
    # 0.042 ms per request against 0.003 ms for the slid window. dynamic_slice
    # needs its start in range, so each array is padded by its own length; the
    # padding is never read, because the kernel only touches the first
    # num_active - num_decode slots plus cu_q_lens[num_reqs].
    cu_q_window = lax.dynamic_slice(jnp.pad(cu_q_lens, (0, max_num_seqs)),
                                    (num_decode, ), (max_num_seqs + 1, ))
    kv_lens_window = lax.dynamic_slice(jnp.pad(kv_lens, (0, max_num_seqs)),
                                       (num_decode, ), (max_num_seqs, ))
    block_table_window = lax.dynamic_slice(
        jnp.pad(block_table, ((0, max_num_seqs), (0, 0))), (num_decode, 0),
        (max_num_seqs, pages_per_seq))
    # Note (david): max_seqlen_q/k are static block-size bounds, not lengths
    # (the token bucket and the table's capacity), so one executable serves
    # every kv_len of a bucket. The kernel clamps page indices into each
    # sequence's own pages, so its table needs none of the decode kernel's
    # capacity padding.
    output = flash_attn_varlen_func(
        q.transpose(1, 0, 2),
        cache,
        None,
        cu_q_window,
        None,
        max_seqlen_q=q.shape[0],
        max_seqlen_k=pages_per_seq * page_size,
        softmax_scale=sm_scale,
        causal=True,
        token_major=False,
        block_table=block_table_window,
        seqused_k=kv_lens_window,
        num_active=num_active - num_decode,
    ).transpose(1, 0, 2)
    # Note (david): the varlen kernel does not own the rows below its first
    # request (row num_decode): its first q block may start anywhere below it,
    # and a decode-only DP rank leaves it no request. num_decode never exceeds
    # num_leading_rows, so rebuilding the leading rows covers all of them.
    num_leading_rows = min(align_to(decode_out.shape[0], 8), q.shape[0])
    if decode_out.shape[0] < num_leading_rows:
        # Note (david): shape padding only; num_decode cannot reach these rows.
        padded_decode_out = jnp.pad(
            decode_out,
            ((0, num_leading_rows - decode_out.shape[0]), (0, 0), (0, 0)))
    else:
        padded_decode_out = decode_out
    row_index = jnp.arange(num_leading_rows)[:, None, None]
    num_scheduled_tokens = cu_q_lens[num_active]
    leading_output = jnp.where(
        row_index < num_decode, padded_decode_out,
        jnp.where(row_index < num_scheduled_tokens, output[:num_leading_rows],
                  0))
    return output.at[:num_leading_rows].set(leading_output), cache


def flywheel_tpu_attention(
    q: jax.Array,  # [T, N, D]
    k: jax.Array,  # [T, H, D]
    v: jax.Array,  # [T, H, D]
    kv_cache: jax.Array,  # [num_pages, page_size, 2 * H, D]
    kv_lens: jax.Array,  # i32[max_num_seqs]
    page_indices: jax.Array,  # i32[max_num_seqs * pages_per_seq]
    cu_q_lens: jax.Array,  # i32[max_num_seqs + 1]
    kv_write_slices: jax.Array,  # i32[3 * bound]
    kv_write_num_slices: jax.Array,  # i32[1]
    distribution: jax.Array,  # i32[3]: num_decode, num_decode, num_reqs
    *,
    phase: str,  # one of ATTENTION_PHASES, static
    sm_scale: float,
    varlen_block_span: int = -1,
    dense_batch_size: int = -1,
    dense_seq_len: int = -1,
) -> tuple[jax.Array, jax.Array]:
    """One shard's attention for a fresh prefill, cached extend, or decode
    step, dispatched on the host-decided phase and prefill kernel choice.

    distribution is this rank's request_distribution: its last entry bounds
    the decode kernel's active rows and its first splits a mixed batch
    between the decode kernel and the paged varlen kernel.
    """
    # Note (david): the phase is static, so each step traces only its selected
    # kernel and the in-place cache update sits in the entry computation,
    # where it can alias the donated cache. With lax.cond instead, XLA copied
    # the whole cache once per layer (~0.7 ms on v6e, any batch).
    if phase not in ATTENTION_PHASES:
        raise ValueError(
            f"phase must be one of {ATTENTION_PHASES}; got {phase!r}.")
    _, _, head_dim = q.shape
    # Note (david): an EXTEND step runs the decode kernel too, on its leading
    # decode rows, and whether it has any is only known on device.
    if phase != PREFILL_PHASE:
        check_decode_head_dim(head_dim)
    padded_head_dim = _padded_head_dim(head_dim)
    if k.shape[-1] != head_dim or v.shape != k.shape:
        raise ValueError(
            "Q/K/V must share head_dim and K/V shapes must match.")
    num_kv_heads = k.shape[1]
    if kv_cache.ndim != 4:
        raise ValueError(
            "flywheel_tpu expects a (pages, page_size, 2 * kv_heads, "
            f"head_dim) cache; got {kv_cache.shape}.")
    page_size, cache_head_shape = kv_cache.shape[1], kv_cache.shape[2:]
    expected_head_shape = (2 * num_kv_heads, padded_head_dim)
    if cache_head_shape != expected_head_shape:
        raise ValueError(
            f"cache heads/head_dim {cache_head_shape} do not match k/v "
            f"{expected_head_shape}.")
    if page_size % PAGE_SIZE_MULTIPLE:
        raise ValueError(f"page_size must be a multiple of "
                         f"{PAGE_SIZE_MULTIPLE}; got {page_size}.")
    if padded_head_dim == head_dim:
        padded_q, padded_k, padded_v = q, k, v
    else:
        # Note (david): get_kv_cache_shape allocates the cache at the padded
        # head_dim once, so only this step's activations need padding.
        padding = ((0, 0), (0, 0), (0, padded_head_dim - head_dim))
        padded_q, padded_k, padded_v = (jnp.pad(activation, padding)
                                        for activation in (q, k, v))
    # Note (david): per-rank metadata is sharded as flat arrays, so the
    # writer's (3, bound) table arrives flattened.
    kv_write_slices = kv_write_slices.reshape(3, -1)

    if phase == DECODE_PHASE:
        output, kv_cache = flywheel_tpu_decode(padded_q,
                                               padded_k,
                                               padded_v,
                                               kv_cache,
                                               kv_lens,
                                               page_indices,
                                               distribution[2],
                                               sm_scale=sm_scale)
    elif phase == EXTEND_PHASE:
        output, kv_cache = flywheel_tpu_extend(padded_q,
                                               padded_k,
                                               padded_v,
                                               kv_cache,
                                               kv_lens,
                                               page_indices,
                                               cu_q_lens,
                                               kv_write_slices,
                                               kv_write_num_slices,
                                               distribution[0],
                                               distribution[2],
                                               sm_scale=sm_scale)
    else:
        output, kv_cache = flywheel_tpu_varlen_prefill(
            padded_q,
            padded_k,
            padded_v,
            kv_cache,
            cu_q_lens,
            kv_write_slices,
            kv_write_num_slices,
            sm_scale=sm_scale,
            varlen_block_span=varlen_block_span,
            dense_batch_size=dense_batch_size,
            dense_seq_len=dense_seq_len)
    return output[..., :head_dim], kv_cache


def sharded_flywheel_tpu_attention(
    mesh: Mesh,
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    kv_cache: jax.Array,
    kv_lens: jax.Array,
    page_indices: jax.Array,
    cu_q_lens: jax.Array,
    kv_write_slices: jax.Array,  # i32[dp * 3 * bound]
    kv_write_num_slices: jax.Array,  # i32[dp]
    distribution: jax.Array,  # i32[dp * 3]
    *,
    phase: str,
    sm_scale: float,
    varlen_block_span: int = -1,
    dense_batch_size: int = -1,
    dense_seq_len: int = -1,
) -> tuple[jax.Array, jax.Array]:
    """Shards along KV heads, mirroring sharded_ragged_paged_attention."""
    tp_size = get_mesh_shape_product(mesh, ShardingAxisName.ATTN_HEAD)
    num_kv_heads = k.shape[1]
    if (tp_size > 1 and num_kv_heads < tp_size
            and tp_size % num_kv_heads != 0):
        raise ValueError(
            f"For GQA/MQA, tp_size {tp_size} must be divisible by "
            f"num_kv_heads {num_kv_heads}")
    if tp_size > 1 and num_kv_heads < tp_size:
        kv_head_repeats = tp_size // num_kv_heads
        sharded_k = jnp.repeat(k, kv_head_repeats, axis=1)
        sharded_v = jnp.repeat(v, kv_head_repeats, axis=1)
    else:
        sharded_k, sharded_v = k, v

    qkv_spec = P(ShardingAxisName.ATTN_DATA, ShardingAxisName.ATTN_HEAD, None)
    # Note (david): the cache's head axis splits into each shard's
    # contiguous run of interleaved (K, V) row pairs.
    kv_cache_spec = P(ShardingAxisName.ATTN_DATA, None,
                      ShardingAxisName.ATTN_HEAD, None)
    in_specs = ((qkv_spec, qkv_spec, qkv_spec, kv_cache_spec) +
                (P(ShardingAxisName.ATTN_DATA), ) * 6)
    out_specs = (qkv_spec, kv_cache_spec)

    def _attention(*args: jax.Array) -> tuple[jax.Array, jax.Array]:
        return flywheel_tpu_attention(*args,
                                      phase=phase,
                                      sm_scale=sm_scale,
                                      varlen_block_span=varlen_block_span,
                                      dense_batch_size=dense_batch_size,
                                      dense_seq_len=dense_seq_len)

    return jax.shard_map(
        _attention,
        mesh=mesh,
        in_specs=in_specs,
        out_specs=out_specs,
        check_vma=False,
    )(q, sharded_k, sharded_v, kv_cache, kv_lens, page_indices, cu_q_lens,
      kv_write_slices, kv_write_num_slices, distribution)


def check_supported(
    mesh: Mesh,
    q: jax.Array,
    *,
    sinks: jax.Array | None,
    attention_chunk_size: int | None,
    attn_logits_soft_cap: float | None,
    q_scale: float | None,
    k_scale: float | None,
    v_scale: float | None,
    update_kv_cache: bool,
    decode_query_size: int,
    use_causal_mask: bool = True,
) -> None:
    """Raises NotImplementedError for anything the backend does not do."""
    for cp_axis in ("dcp", "pcp"):
        if cp_axis in mesh.shape and mesh.shape[cp_axis] > 1:
            raise NotImplementedError(
                "The flywheel attention backend does not support context "
                f"parallel meshes (axis {cp_axis!r} has size "
                f"{mesh.shape[cp_axis]}).")
    unsupported = {
        "sinks":
        sinks is not None,
        "sliding window (attention_chunk_size)":
        attention_chunk_size is not None,
        "logits soft cap":
        attn_logits_soft_cap is not None,
        "q_scale/k_scale/v_scale (quantized KV)":
        any(s is not None for s in (q_scale, k_scale, v_scale)),
        "update_kv_cache=False (KV-share layers)":
        not update_kv_cache,
        "decode_query_size != 1 (speculative decoding)":
        decode_query_size != 1,
        "use_causal_mask=False (non-causal attention)":
        not use_causal_mask,
    }
    for name, is_unsupported in unsupported.items():
        if is_unsupported:
            raise NotImplementedError(
                f"The flywheel attention backend does not support {name}.")
    _padded_head_dim(q.shape[-1])
    if q.dtype != jnp.bfloat16:
        raise NotImplementedError(
            f"The flywheel attention backend needs bfloat16 activations; got "
            f"{q.dtype}.")


def flywheel_attention(
    kv_cache: jax.Array,
    q: jax.Array,  # [T, N, D]
    k: jax.Array,  # [T, H, D]
    v: jax.Array,  # [T, H, D]
    md: AttentionMetadata,
    shared_md: AttentionMetadata | SharedAttentionMetadata,
    mesh: Mesh,
    *,
    sm_scale: float,
    attention_chunk_size: int | None,
    q_scale: float | None,
    k_scale: float | None,
    v_scale: float | None,
    sinks: jax.Array | None,
    update_kv_cache: bool,
    use_causal_mask: bool,
    attn_logits_soft_cap: float | None,
    decode_query_size: int,
) -> tuple[jax.Array, jax.Array]:
    """attention_interface.attention on the flywheel backend; returns
    (kv_cache, output) like it."""
    check_supported(
        mesh,
        q,
        sinks=sinks,
        attention_chunk_size=attention_chunk_size,
        attn_logits_soft_cap=attn_logits_soft_cap,
        q_scale=q_scale,
        k_scale=k_scale,
        v_scale=v_scale,
        update_kv_cache=update_kv_cache,
        decode_query_size=decode_query_size,
        use_causal_mask=use_causal_mask,
    )
    if md.kv_write_slices is None or md.kv_write_num_slices is None:
        raise ValueError(
            "The flywheel attention backend needs the host-built KV write "
            "slice table (AttentionMetadata.kv_write_slices / "
            "kv_write_num_slices); this metadata was built without it.")
    if md.attention_phase not in ATTENTION_PHASES:
        raise ValueError(
            "The flywheel attention backend needs the host-decided attention "
            f"phase (AttentionMetadata.attention_phase in {ATTENTION_PHASES}); "
            f"got {md.attention_phase!r}.")
    output, kv_cache = sharded_flywheel_tpu_attention(
        mesh,
        q,
        k,
        v,
        kv_cache,
        shared_md.seq_lens,
        md.block_tables,
        shared_md.query_start_loc,
        md.kv_write_slices,
        md.kv_write_num_slices,
        shared_md.request_distribution,
        phase=md.attention_phase,
        sm_scale=sm_scale,
        varlen_block_span=md.varlen_block_span,
        dense_batch_size=md.dense_batch_size,
        dense_seq_len=md.dense_seq_len,
    )
    return kv_cache, output


def attention_group_page_size_tokens(
        kv_cache_group: KVCacheGroupSpec) -> int | None:
    """Tokens per page of an attention KV cache group, None for a mamba or
    other non-attention group."""
    spec = kv_cache_group.kv_cache_spec
    if isinstance(spec, UniformTypeKVCacheSpecs):
        layer_spec = next(iter(spec.kv_cache_specs.values()))
    else:
        layer_spec = spec
    if isinstance(layer_spec, AttentionSpec):
        return spec.block_size
    return None


@dataclass(frozen=True)
class FlywheelStepFields:
    """The flywheel AttentionMetadata fields of one step."""
    attention_phase: str
    dense_batch_size: int
    dense_seq_len: int
    varlen_block_span: int
    # KV cache group id -> (slice table key, slice count key) in the device
    # buffer layout; attention groups only.
    kv_write_keys: Mapping[int, tuple[str, str]]

    def __call__(self, metadata: Mapping[str, jax.Array],
                 gid: int) -> dict[str, Any]:
        if gid in self.kv_write_keys:
            slices_key, count_key = self.kv_write_keys[gid]
            kv_write_slices = metadata[slices_key]
            kv_write_num_slices = metadata[count_key]
        else:
            kv_write_slices = kv_write_num_slices = None
        return dict(
            attention_phase=self.attention_phase,
            dense_batch_size=self.dense_batch_size,
            dense_seq_len=self.dense_seq_len,
            varlen_block_span=self.varlen_block_span,
            kv_write_slices=kv_write_slices,
            kv_write_num_slices=kv_write_num_slices,
        )


@dataclass(frozen=True)
class FlywheelWarmupVariant:
    """The static metadata of one backbone executable: (attention phase,
    ragged block span, dense seq_len)."""
    attention_phase: str
    varlen_block_span: int
    dense_seq_len: int


class FlywheelAttentionBackend(AttentionBackend):
    """additional_config["attention_backend"] = "flywheel"."""

    name = "flywheel"

    # Note (david): the primer stands for a step with no active request, and
    # the extend kernel only clears the whole output bucket, first tile
    # included, when query_start_loc[0] is zero.
    zero_warmup_lengths = True

    def check_config(self, vllm_config: "VllmConfig") -> None:
        """Round the block size the platform chose up to what the kernels can
        index, then reject every configuration this backend cannot serve,
        listing each unsupported feature in one error."""
        cache_config = vllm_config.cache_config
        model_config = vllm_config.model_config
        parallel_config = vllm_config.parallel_config
        problems: list[str] = []

        if not cache_config.user_specified_block_size:
            block_size = align_to(cache_config.block_size, PAGE_SIZE_MULTIPLE)
            if block_size != cache_config.block_size:
                logger.info(
                    "The flywheel attention backend indexes whole %d-token "
                    "pages; increasing the block size from %d to %d.",
                    PAGE_SIZE_MULTIPLE, cache_config.block_size, block_size)
                cache_config.block_size = block_size
        elif cache_config.block_size % PAGE_SIZE_MULTIPLE:
            problems.append(
                f"block size {cache_config.block_size} is not a multiple of "
                f"{PAGE_SIZE_MULTIPLE}; pass a --block-size that is, or leave "
                "it unset")

        if "USE_BATCHED_RPA_KERNEL" in os.environ:
            problems.append("USE_BATCHED_RPA_KERNEL selects an RPA kernel; "
                            "unset it")
        if vllm_config.speculative_config is not None:
            problems.append("speculative decoding is not supported")
        if vllm_config.additional_config.get("enable_continue_decode", False):
            # Note (david): the fused on-device decode loop rebuilds
            # AttentionMetadata without the KV write slice table or the static
            # attention phase this backend reads.
            problems.append("continue_decode is not supported; drop "
                            "additional_config.enable_continue_decode")
        if vllm_config.kv_transfer_config is not None:
            problems.append(
                "KV transfer (disaggregated serving) is not supported")
        if parallel_config.decode_context_parallel_size > 1:
            problems.append("decode context parallelism is not supported")
        if parallel_config.prefill_context_parallel_size > 1:
            problems.append("prefill context parallelism is not supported")
        if model_config.use_mla:
            problems.append("MLA models are not supported")
        # Note (david): every served model decodes, so a head_dim the decode
        # kernel rejects fails here rather than at the first decode step.
        if model_config.get_head_size() not in DECODE_HEAD_DIMS:
            problems.append(
                f"head_dim {model_config.get_head_size()} is not one of the "
                f"decode kernel's {DECODE_HEAD_DIMS}")
        if cache_config.cache_dtype not in ("auto", "bfloat16"):
            problems.append(
                f"kv cache dtype {cache_config.cache_dtype!r} is not bfloat16")
        if str(model_config.dtype).removeprefix("torch.") != "bfloat16":
            problems.append(
                f"model dtype {model_config.dtype} is not bfloat16")

        if problems:
            raise ValueError(
                "The flywheel attention backend cannot serve this "
                "configuration: " + "; ".join(problems) + ".")

    def get_kv_cache_shape(self, total_num_pages: int, page_size: int,
                           num_kv_heads: int, head_dim: int,
                           kv_dtype: Any) -> tuple[int, ...]:
        return get_kv_cache_shape(total_num_pages, page_size, num_kv_heads,
                                  head_dim, kv_dtype)

    def attention(
        self,
        kv_cache: jax.Array,
        q: jax.Array,
        k: jax.Array,
        v: jax.Array,
        md: AttentionMetadata,
        shared_md: AttentionMetadata | SharedAttentionMetadata,
        mesh: Mesh,
        **options: Any,
    ) -> tuple[jax.Array, jax.Array]:
        return flywheel_attention(kv_cache, q, k, v, md, shared_md, mesh,
                                  **options)

    def prepare_step(self, step: HostStepInputs,
                     device_buffer: DeviceBuffer) -> AttentionFields:
        """Decide the step's static phase and prefill kernel, and append each
        attention group's KV write slice table to device_buffer."""

        def rank_slices(dp_rank: int) -> tuple[int, slice, slice]:
            num_reqs = int(step.request_distribution[3 * dp_rank + 2])
            req_offset = dp_rank * step.max_num_reqs_per_dp_rank
            query_offset = dp_rank * (step.max_num_reqs_per_dp_rank + 1)
            return (num_reqs, slice(req_offset, req_offset + num_reqs),
                    slice(query_offset, query_offset + num_reqs + 1))

        attention_phases: set[str] = set()
        for dp_rank in range(step.dp_size):
            num_reqs, req_range, query_range = rank_slices(dp_rank)
            if num_reqs == 0:
                continue
            attention_phases.add(
                select_attention_phase(
                    int(step.request_distribution[3 * dp_rank]),
                    step.query_start_loc[query_range],
                    step.seq_lens[req_range]))
        # Note (david): the step's phase is static metadata, one backbone
        # executable per phase, so DP ranks that disagree have to settle on
        # extend, the only phase that serves cached and fresh requests alike.
        attention_phase = combine_attention_phases(attention_phases)

        # Note (david): the dense shape and the ragged block span are static
        # metadata, so both are decided here on the host from this step's
        # request lengths. Only a prefill step has a choice: dense when the
        # lengths land on one of the bucketed shapes, ragged otherwise.
        bucket = step.padded_num_scheduled_tokens_per_dp_rank
        if attention_phase == PREFILL_PHASE:
            scheduled_tokens_by_rank = [
                step.scheduled_tokens_per_dp_rank[dp_rank]
                for dp_rank in range(step.dp_size)
            ]
            dense_batch_size, dense_seq_len = dense_prefill_shape(
                scheduled_tokens_by_rank, bucket)
            if dense_batch_size < 0:
                block_span = varlen_block_span(scheduled_tokens_by_rank,
                                               bucket)
            else:
                block_span = -1
        else:
            dense_batch_size = dense_seq_len = block_span = -1

        # Note (david): the KV write slice table of every attention group is
        # built on the host from the same arrays as the block tables, so the
        # device never scans the page table. Its bound follows the token bucket
        # and the request slots alone, keeping the compiled shapes fixed across
        # steps.
        kv_write_keys: dict[int, tuple[str, str]] = {}
        for gid, kv_cache_group in enumerate(step.kv_cache_groups):
            page_size = attention_group_page_size_tokens(kv_cache_group)
            if page_size is None:
                continue
            max_num_slices = kv_write_slice_bound(
                bucket, step.max_num_reqs_per_dp_rank, page_size)
            slices_key = f"kv_write_slices_gid_{gid}"
            count_key = f"kv_write_num_slices_gid_{gid}"
            slices_view = device_buffer.get_view(
                (step.dp_size * 3, max_num_slices), key=slices_key)
            num_slices_view = device_buffer.get_view((step.dp_size, ),
                                                     key=count_key)
            for dp_rank in range(step.dp_size):
                _, req_range, query_range = rank_slices(dp_rank)
                num_slices_view[dp_rank] = build_kv_write_slices(
                    step.query_start_loc[query_range],
                    step.seq_lens[req_range],
                    step.block_tables[gid][req_range],
                    page_size=page_size,
                    out=slices_view[dp_rank * 3:(dp_rank + 1) * 3])
            kv_write_keys[gid] = (slices_key, count_key)

        return FlywheelStepFields(
            attention_phase=attention_phase,
            dense_batch_size=dense_batch_size,
            dense_seq_len=dense_seq_len,
            varlen_block_span=block_span,
            kv_write_keys=kv_write_keys,
        )

    def warmup_variants(self, num_tokens: int,
                        num_tokens_paddings: Sequence[int], max_num_reqs: int,
                        dp_size: int) -> list[FlywheelWarmupVariant]:
        # Note (david): the phase and the span are both static metadata, so an
        # unwarmed pair traces a fresh Pallas backbone mid-serving, which costs
        # 20-106 s of engine time on v6e-1 with Qwen3-4B even against a warm
        # persistent compile cache. Warming every reachable span instead took
        # that model from 27 to 42 backbone variants and its startup from
        # 1561.33 s to 1574.39 s.
        variants = [
            FlywheelWarmupVariant(PREFILL_PHASE, span, -1)
            for span in varlen_block_spans(num_tokens // dp_size)
        ]
        # Note (david): a one-request prefill step near the top of its bucket
        # runs dense at (1, bucket), which a lone long prompt reaches on any
        # serving day. The batched dense shapes need equal power-of-two
        # lengths filling the bucket and stay unwarmed.
        dense_seq_len = single_request_dense_seq_len(num_tokens // dp_size)
        if dense_seq_len > 0:
            variants.append(
                FlywheelWarmupVariant(PREFILL_PHASE, -1, dense_seq_len))
        variants.append(FlywheelWarmupVariant(EXTEND_PHASE, -1, -1))
        # Note (david): a decode step carries one token per request, so the
        # bucket a full request batch rounds up to is the largest one decode
        # can reach.
        max_decode_bucket = next(
            (p for p in num_tokens_paddings if p >= max_num_reqs),
            num_tokens_paddings[-1])
        if num_tokens <= max_decode_bucket:
            variants.append(FlywheelWarmupVariant(DECODE_PHASE, -1, -1))
        return variants

    def prepare_warmup(
        self, variant: FlywheelWarmupVariant,
        kv_cache_groups: Sequence[KVCacheGroupSpec], *, num_tokens: int,
        max_num_reqs: int, dp_size: int, mesh: Mesh, sharding: NamedSharding
    ) -> tuple[AttentionFields, Mapping[str, jax.Array]]:
        # Note (david): at runtime every attention group carries a KV write
        # slice table, so the primer must carry one of the same shape or the
        # cached HLO is not reused.
        arrays: dict[str, jax.Array] = {}
        kv_write_keys: dict[int, tuple[str, str]] = {}
        for gid, kv_cache_group in enumerate(kv_cache_groups):
            page_size = attention_group_page_size_tokens(kv_cache_group)
            if page_size is None:
                continue
            max_num_slices = kv_write_slice_bound(num_tokens // dp_size,
                                                  max_num_reqs // dp_size,
                                                  page_size)
            slices_key = f"kv_write_slices_gid_{gid}"
            count_key = f"kv_write_num_slices_gid_{gid}"
            arrays[slices_key] = device_array(mesh,
                                              np.zeros(dp_size * 3 *
                                                       max_num_slices,
                                                       dtype=np.int32),
                                              sharding=sharding)
            arrays[count_key] = device_array(mesh,
                                             np.zeros(dp_size, dtype=np.int32),
                                             sharding=sharding)
            kv_write_keys[gid] = (slices_key, count_key)
        fields = FlywheelStepFields(
            attention_phase=variant.attention_phase,
            dense_batch_size=1 if variant.dense_seq_len > 0 else -1,
            dense_seq_len=variant.dense_seq_len,
            varlen_block_span=variant.varlen_block_span,
            kv_write_keys=kv_write_keys,
        )
        return fields, arrays
