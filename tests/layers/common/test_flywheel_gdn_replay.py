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
"""The runner's host update of the ReplaySSM records against flywheel_tpu's
next_replay_lens."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flywheel_tpu.linear_attention.gdn import wrapper as flywheel_gdn

from tpu_inference.layers.common.flywheel_gdn_replay import \
    next_gdn_replay_lens

REPLAY_WINDOW = 8
NUM_STEPS = 64


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_next_gdn_replay_lens_matches_flywheel(seed: int):
    """Requests arrive, prefill in chunks, decode across several windows and
    finish, and some sit out a step; every step's host update matches
    flywheel's next_replay_lens over the scheduled requests."""
    rng = np.random.default_rng(seed)
    requests: list[dict[str, int]] = []
    max_replay_len = 0
    for _ in range(NUM_STEPS):
        for _ in range(rng.integers(0, 3)):
            prompt_len = int(rng.integers(1, 40))
            requests.append({
                "prompt_len": prompt_len,
                "max_len": prompt_len + int(rng.integers(1, 30)),
                "num_computed": 0,
                "replay_len": 0,
            })
        scheduled = [r for r in requests if rng.random() < 0.8]
        if not scheduled:
            continue
        # Note: a prompt chunk has at least 2 tokens unless it is the
        # prompt's last, so only finished prompts run as decodes.
        query_lens = [
            min(r["prompt_len"] - r["num_computed"], int(rng.integers(2, 24)))
            if r["num_computed"] < r["prompt_len"] else 1 for r in scheduled
        ]
        # The runner orders the decodes (1 token) first.
        order = sorted(range(len(scheduled)), key=lambda i: query_lens[i] > 1)
        scheduled = [scheduled[i] for i in order]
        query_lens = np.array([query_lens[i] for i in order], dtype=np.int32)
        replay_lens = np.array([r["replay_len"] for r in scheduled],
                               dtype=np.int32)
        seq_lens = np.array([r["num_computed"] for r in scheduled],
                            dtype=np.int32) + query_lens
        num_decode = int(np.sum(query_lens == 1))

        actual = next_gdn_replay_lens(replay_lens,
                                      query_lens,
                                      seq_lens,
                                      num_decode=num_decode,
                                      replay_window=REPLAY_WINDOW)
        query_start_loc = np.cumsum(query_lens, dtype=np.int32)
        query_start_loc = np.concatenate([[0], query_start_loc])
        # One slot per request, read and written.
        slots = np.arange(1, len(scheduled) + 1, dtype=np.int32)
        distribution = np.array(
            [num_decode, num_decode, len(scheduled)], dtype=np.int32)
        with jax.default_device(jax.devices("cpu")[0]):
            expected = flywheel_gdn.next_replay_lens(
                *map(jnp.asarray, (replay_lens, query_start_loc, seq_lens,
                                   slots, slots, distribution, query_lens)),
                REPLAY_WINDOW)
        np.testing.assert_array_equal(actual, np.asarray(expected))

        max_replay_len = max(max_replay_len, int(actual.max()))
        for request, query_len, replay_len in zip(scheduled, query_lens,
                                                  actual):
            request["num_computed"] += int(query_len)
            request["replay_len"] = int(replay_len)
        requests = [r for r in requests if r["num_computed"] < r["max_len"]]
    # Note: full windows, so the steps after them flushed.
    assert max_replay_len == REPLAY_WINDOW
