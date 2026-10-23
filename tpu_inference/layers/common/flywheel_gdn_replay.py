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
"""ReplaySSM decode of the GDN layers under --attention-backend flywheel: its
switch, and the host side of each request's committed record count.

A replay request keeps its recurrent state as an f32 checkpoint plus the
records of its recent decode tokens, and its kernel computes the outputs from
them without materializing the state (fused_conv1d_gdn_replay). The runner
tracks how many records each request has committed (InputBatch's
mamba_replay_lens_cpu) and ships them in AttentionMetadata.mamba_replay_lens.
"""

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from vllm.config import VllmConfig

# The additional_config key that turns replay on: its int value is the window,
# the records a request keeps before it folds them into its checkpoint.
GDN_REPLAY_WINDOW_KEY = "flywheel_gdn_replay_window"


def gdn_replay_window(vllm_config: "VllmConfig") -> int | None:
    """The replay window, or None when replay is off.
    FlywheelAttentionBackend.check_config validates it, and
    TpuPlatform.check_and_update_config rejects it under any other backend."""
    return vllm_config.additional_config.get(GDN_REPLAY_WINDOW_KEY)


def next_gdn_replay_lens(replay_lens: np.ndarray, query_lens: np.ndarray,
                         seq_lens: np.ndarray, *, num_decode: int,
                         replay_window: int) -> np.ndarray:
    """flywheel_tpu's next_replay_lens on the host, for one DP rank's requests
    in the step's order, with every scheduled token accepted (no speculative
    decoding) and one state slot read and written (mamba_cache_mode "none").
      Input:
        replay_lens: int32 [num_reqs], the committed records the step reads.
        query_lens: [num_reqs], the scheduled tokens of each request.
        seq_lens: [num_reqs], its computed plus scheduled tokens, as the step's
          seq_lens hold them.
        num_decode: the leading requests the kernel runs as decodes, the
          rank's request_distribution[0].
        replay_window: gdn_replay_window.
      Output:
        int32 [num_reqs], the committed records each request's next step
        reads: 0 after a prefill or mixed step."""
    has_initial_state = seq_lens - query_lens > 0
    replay_lens = np.where(has_initial_state, replay_lens, 0)
    is_decode = np.arange(replay_lens.size) < num_decode
    # Note: the kernel resumes a prefill or mixed request from its
    # checkpoint alone. Only a decode commits records, and InputBatch.is_decode
    # runs a step as one only when it reaches the last known token, so the
    # request's next step is a decode as well.
    has_records = ~is_decode & (replay_lens > 0)
    assert not has_records.any(), (
        "A prefill or mixed request would resume from committed ReplaySSM "
        f"records {replay_lens[has_records].tolist()}, which the flywheel GDN "
        "kernel does not read.")
    # Note: a decode flushes, folding its records into a new checkpoint
    # and appending from record 0, when its tokens overflow the window or it
    # has no initial state.
    is_flush = (replay_lens + query_lens > replay_window) | ~has_initial_state
    return np.where(is_decode,
                    np.where(is_flush, 0, replay_lens) + query_lens,
                    0).astype(np.int32)
