# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.

"""Offline tests for patch_scheduler_spec_decode_pad.

The patch forks upstream ``Scheduler.schedule`` (~740 lines), so these tests
drive the forked method through a fake ``self`` that mirrors only the
attributes/methods the fork touches, and assert on the ``SchedulerOutput`` it
builds. No NPU and no real vLLM engine is required; only the vllm/omni_npu
packages must be importable (same style as test_patch_scheduler.py).

Run with pytest or directly (``python test_patch_scheduler_spec_decode_pad.py``).
"""

import inspect
import sys
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import pytest

from vllm.v1.core.sched.interface import PauseState
from vllm.v1.core.sched.request_queue import SchedulingPolicy
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.request import RequestStatus

from omni_npu import envs
from omni_npu.vllm_patches.patches.models.pangu_v2_base import (
    patch_scheduler_spec_decode_pad as sched_mod,
)

GUARD = "OMNI_DISABLE_SPEC_DECODE_PAD"

# The fork deviates from upstream with a 3-line comment plus the guard clause;
# the drift test strips them before diffing against upstream.
FORK_MARKERS = (
    "omni-npu:",
    "padded to the full spec-decode",
    "dummy draft slots",
)

# True when the file is executed as a script (patch_test_utils harness) instead
# of collected by pytest; pytest.skip() aborts that harness.
_STANDALONE = __name__ == "__main__"

_TESTS_ROOT = next(
    p for p in Path(__file__).resolve().parents if (p / "unit" / "vllm_patch").is_dir()
)
if str(_TESTS_ROOT) not in sys.path:
    sys.path.insert(0, str(_TESTS_ROOT))
from unit.vllm_patch.patches.patch_test_utils import run_standalone_tests  # noqa: E402


# --------------------------------------------------------------------------
# env helper
# --------------------------------------------------------------------------


@contextmanager
def _spec_pad_env(enabled: bool):
    """Pin envs.OMNI_DISABLE_SPEC_DECODE_PAD and restore it afterwards."""
    sentinel = object()
    original = getattr(envs, GUARD, sentinel)
    setattr(envs, GUARD, enabled)
    try:
        yield
    finally:
        if original is sentinel:
            try:
                delattr(envs, GUARD)
            except AttributeError:
                pass
        else:
            setattr(envs, GUARD, original)


# --------------------------------------------------------------------------
# fakes
# --------------------------------------------------------------------------


class _FakeQueue:
    """Minimal stand-in for vllm.v1.core.sched.request_queue.RequestQueue."""

    def __init__(self, requests=()):
        self._items = list(requests)

    def __bool__(self):
        return bool(self._items)

    def __len__(self):
        return len(self._items)

    def __iter__(self):
        return iter(self._items)

    def peek_request(self):
        return self._items[0]

    def pop_request(self):
        return self._items.pop(0)

    def prepend_request(self, request):
        self._items.insert(0, request)

    def prepend_requests(self, other):
        self._items[:0] = list(other)


class _BlockIds:
    def __init__(self, ids=(11, 12)):
        self._ids = list(ids)

    def get_block_ids(self):
        return self._ids


class _Request(SimpleNamespace):
    """Hashable request stub: the real Scheduler keeps requests in sets."""

    __hash__ = object.__hash__


class _FakeKVCacheManager:
    """``fail_after``: fail every allocate_slots call after the Nth one."""

    def __init__(self):
        self.allocate_calls = []
        self.fail_after = None
        self.empty_kv_cache_blocks = _BlockIds([])
        self.new_block_ids = [5]
        self.coordinator = None
        self.log_stats = False
        self.prefix_cache_stats = None

    def new_step_starts(self):
        pass

    def create_kv_cache_blocks(self, computed):
        return _BlockIds([])

    def allocate_slots(self, request, num_new_tokens, *args, **kwargs):
        self.allocate_calls.append((request.request_id, num_new_tokens, kwargs))
        if self.fail_after is not None and len(self.allocate_calls) > self.fail_after:
            return None
        return _BlockIds()

    def get_computed_blocks(self, request):
        return _BlockIds([]), 0

    def get_blocks(self, request_id):
        return _BlockIds()

    def get_num_common_prefix_blocks(self, request_id):
        return [0]

    def take_new_block_ids(self):
        return self.new_block_ids


class _FakeEncoderCacheManager:
    def __init__(self):
        self.allocated = []
        self.freed = []

    def allocate(self, request, index):
        self.allocated.append((request.request_id, index))

    def free(self, request):
        self.freed.append(request.request_id)

    def get_freed_mm_hashes(self):
        return []


def _request(
    request_id,
    *,
    num_tokens,
    num_computed_tokens=0,
    is_prefill_chunk=False,
    has_encoder_inputs=False,
    status=RequestStatus.WAITING,
    spec_token_ids=(),
):
    return _Request(
        request_id=request_id,
        num_tokens=num_tokens,
        num_tokens_with_spec=num_tokens,
        num_prompt_tokens=num_tokens,
        num_computed_tokens=num_computed_tokens,
        num_output_placeholders=0,
        next_decode_eligible_step=0,
        is_prefill_chunk=is_prefill_chunk,
        has_encoder_inputs=has_encoder_inputs,
        spec_token_ids=list(spec_token_ids),
        status=status,
        prefill_stats=None,
        num_preemptions=0,
        lora_request=None,
        mm_features=None,
        block_hashes=None,
        prompt_token_ids=[],
        sampling_params=None,
        pooling_params=None,
        prompt_embeds=None,
        prompt_is_token_ids=None,
        priority=0,
        arrival_time=0.0,
        record_event=lambda *args, **kwargs: None,
        get_num_encoder_embeds=lambda index: 1,
    )


class _FakeScheduler:
    """Mirrors the Scheduler surface the forked schedule() touches."""

    def __init__(self, *, running=(), waiting=(), **overrides):
        self.current_step = 0
        self._pause_state = PauseState.UNPAUSED
        self.max_num_scheduled_tokens = 512
        self.max_num_encoder_input_tokens = 1024
        self.max_num_running_reqs = 64
        self.max_model_len = 4096
        self.num_sampled_tokens_per_step = 1
        self.num_spec_tokens = 0
        self.num_lookahead_tokens = 0
        self.dynamic_sd_lookup = None
        self.scheduler_config = SimpleNamespace(
            long_prefill_token_threshold=0,
            enable_chunked_prefill=False,
        )
        self.scheduler_reserve_full_isl = False
        self.use_eagle = False
        self.use_v2_model_runner = False
        self.need_mamba_block_aligned_split = False
        self.has_mamba_layers = False
        self.is_encoder_decoder = False
        self.needs_kv_cache_zeroing = False
        self.defer_block_free = False
        self.prefill_capacity_bound = False
        self.log_stats = False
        self.lora_config = None
        self.connector = None
        self.ec_connector = None
        self.connector_prefix_cache_stats = None
        self.policy = SchedulingPolicy.FCFS
        self.running = list(running)
        self.waiting = _FakeQueue(waiting)
        self.skipped_waiting = _FakeQueue()
        self.num_waiting_for_streaming_input = 0
        self._inflight_prefills = set()
        self.prev_step_scheduled_req_ids = set()
        self.finished_req_ids = set()
        self.reset_preempted_req_ids = set()
        self.sched_step_seq = 0
        self.kv_cache_config = SimpleNamespace(kv_cache_groups=["group-0"])
        self.kv_cache_manager = _FakeKVCacheManager()
        self.encoder_cache_manager = _FakeEncoderCacheManager()
        self.updated_after_schedule = []
        self.preempted = []
        self.encoder_schedule_calls = []
        for name, value in overrides.items():
            setattr(self, name, value)

    def _is_blocked_waiting_status(self, status):
        return False

    def _try_promote_blocked_waiting_request(self, request):
        return True

    def _select_waiting_queue_for_scheduling(self):
        return self.waiting

    def _make_cached_request_data(self, *args):
        self.cached_request_args = args
        return SimpleNamespace(cached=True)

    def _update_after_schedule(self, scheduler_output):
        self.updated_after_schedule.append(scheduler_output)

    def _preempt_request(self, request, scheduled_timestamp):
        self.preempted.append(request.request_id)

    def _try_schedule_encoder_inputs(
        self, request, num_computed_tokens, num_new_tokens, budget, *,
        shift_computed_tokens,
    ):
        self.encoder_schedule_calls.append(request.request_id)
        return [0], num_new_tokens, budget, []

    def _mamba_block_aligned_split(self, request, num_new_tokens, *_args):
        return num_new_tokens

    def _build_kv_connector_meta(self, connector, scheduler_output):
        return None

    def _inflight_prefill_reserved_blocks(self):
        return 0


def _schedule(fake, throttle_prefills=False):
    return sched_mod.PanguV2SpecDecodePadPatch.schedule(fake, throttle_prefills)


# --------------------------------------------------------------------------
# structural checks
# --------------------------------------------------------------------------


def test_patch_targets_scheduler_schedule():
    assert sched_mod.PanguV2SpecDecodePadPatch._target is Scheduler
    assert sched_mod.PanguV2SpecDecodePadPatch._attr_names_to_apply == ["schedule"]


def test_fork_disables_spec_decode_padding():
    source = inspect.getsource(sched_mod.PanguV2SpecDecodePadPatch.schedule)
    assert f"and not envs.{GUARD}" in source


def test_fork_matches_upstream_except_the_guard():
    if not hasattr(Scheduler, "schedule"):
        if _STANDALONE:
            return
        pytest.skip("Scheduler stub without upstream schedule()")

    def normalized(func):
        lines = [line.rstrip() for line in inspect.getsource(func).splitlines()]
        while lines and not lines[-1].strip():
            lines.pop()
        # Drop the guard clause and its comment so the fork can be diffed
        # against upstream even when the local vLLM checkout still carries the
        # raw os.environ variant.
        return [
            line for line in lines
            if GUARD not in line and not any(m in line for m in FORK_MARKERS)
        ]

    assert normalized(Scheduler.schedule) == normalized(
        sched_mod.PanguV2SpecDecodePadPatch.schedule
    )


# --------------------------------------------------------------------------
# schedule() fork behaviour
# --------------------------------------------------------------------------


def test_idle_step_returns_empty_output():
    fake = _FakeScheduler()

    output = _schedule(fake)

    assert output.num_scheduled_tokens == {}
    assert output.total_num_scheduled_tokens == 0
    assert output.scheduled_spec_decode_tokens == {}
    assert output.scheduled_encoder_inputs == {}
    assert output.new_block_ids_to_zero is None
    assert output.num_spec_tokens_to_schedule == 0
    assert output.scheduled_new_reqs == []
    assert fake.updated_after_schedule == [output]
    assert fake.current_step == 1


def test_paused_all_schedules_nothing():
    running = _request("r0", num_tokens=5, num_computed_tokens=4)
    waiting = _request("w0", num_tokens=10)
    fake = _FakeScheduler(
        running=[running], waiting=[waiting], _pause_state=PauseState.PAUSED_ALL
    )

    output = _schedule(fake)

    assert output.num_scheduled_tokens == {}
    assert fake.kv_cache_manager.allocate_calls == []
    assert len(fake.waiting) == 1


def test_running_decode_request_is_scheduled():
    running = _request("r0", num_tokens=5, num_computed_tokens=4)
    fake = _FakeScheduler(
        running=[running], need_mamba_block_aligned_split=True
    )

    output = _schedule(fake)

    assert output.num_scheduled_tokens == {"r0": 1}
    assert output.scheduled_spec_decode_tokens == {}
    assert fake.running == [running]
    assert "r0" in fake.prev_step_scheduled_req_ids
    assert [call[:2] for call in fake.kv_cache_manager.allocate_calls] == [("r0", 1)]


def test_running_prefill_with_encoder_inputs():
    ec_calls = []
    running = _request(
        "r0",
        num_tokens=10,
        num_computed_tokens=0,
        is_prefill_chunk=True,
        has_encoder_inputs=True,
    )
    fake = _FakeScheduler(running=[running])
    fake.ec_connector = SimpleNamespace(
        update_state_after_alloc=lambda request, index: ec_calls.append(
            request.request_id
        ),
        build_connector_meta=lambda output: None,
    )
    # Also reports an external encoder input to load remotely.
    def _schedule_encoder_inputs(
        request, num_computed_tokens, num_new_tokens, budget, *,
        shift_computed_tokens,
    ):
        fake.encoder_schedule_calls.append(request.request_id)
        return [0], num_new_tokens, budget, [1]

    fake._try_schedule_encoder_inputs = _schedule_encoder_inputs

    output = _schedule(fake)

    assert output.num_scheduled_tokens == {"r0": 10}
    assert output.scheduled_encoder_inputs == {"r0": [0]}
    assert fake.encoder_schedule_calls == ["r0"]
    assert fake.encoder_cache_manager.allocated == [("r0", 0), ("r0", 1)]
    assert ec_calls == ["r0", "r0"]


def test_waiting_prefill_is_scheduled_and_tracked_inflight():
    events = []
    stats_calls = []
    waiting = _request("w0", num_tokens=10)
    waiting.record_event = lambda *args: events.append(args)
    waiting.prefill_stats = SimpleNamespace(
        set=lambda **kwargs: stats_calls.append(kwargs)
    )
    fake = _FakeScheduler(
        waiting=[waiting],
        max_num_scheduled_tokens=5,
        scheduler_config=SimpleNamespace(
            long_prefill_token_threshold=0,
            enable_chunked_prefill=True,
        ),
        log_stats=True,
    )

    output = _schedule(fake)

    # Chunked prefill: only the first 5 tokens fit in this step's budget.
    assert output.num_scheduled_tokens == {"w0": 5}
    assert any(req.request_id == "w0" for req in fake._inflight_prefills)
    assert fake.running == [waiting]
    assert waiting.status == RequestStatus.RUNNING
    assert waiting.num_computed_tokens == 0
    assert fake.prefill_capacity_bound is False
    assert len(events) == 1
    assert stats_calls == [
        {"num_prompt_tokens": 10, "num_local_cached_tokens": 0,
         "num_external_cached_tokens": 0}
    ]


def test_resumed_preempted_request_is_reported_as_cached():
    waiting = _request("w0", num_tokens=10, status=RequestStatus.PREEMPTED)
    fake = _FakeScheduler(waiting=[waiting])

    output = _schedule(fake)

    assert output.num_scheduled_tokens == {"w0": 10}
    assert output.scheduled_new_reqs == []
    resumed = fake.cached_request_args[1]
    assert [req.request_id for req in resumed] == ["w0"]


def test_spec_padding_disabled_by_default():
    running = _request("r0", num_tokens=5, num_computed_tokens=4)
    # Fresh decode request: exactly one token left to compute.
    waiting = _request("w0", num_tokens=6, num_computed_tokens=5)
    fake = _FakeScheduler(running=[running], waiting=[waiting], num_spec_tokens=2)

    with _spec_pad_env(True):
        output = _schedule(fake)

    # The omni guard keeps the fresh decode at 1 token (upstream pads to 3).
    assert output.num_scheduled_tokens == {"r0": 1, "w0": 1}
    assert output.scheduled_spec_decode_tokens == {}


def test_spec_padding_applied_when_opted_out():
    running = _request("r0", num_tokens=5, num_computed_tokens=4)
    waiting = _request("w0", num_tokens=6, num_computed_tokens=5)
    fake = _FakeScheduler(running=[running], waiting=[waiting], num_spec_tokens=2)

    with _spec_pad_env(False):
        output = _schedule(fake)

    assert output.num_scheduled_tokens["w0"] == 3
    assert output.scheduled_spec_decode_tokens["w0"] == [-1, -1]
    assert fake.kv_cache_manager.allocate_calls[-1][1] == 3


def test_spec_padding_breaks_when_budget_insufficient():
    running = _request("r0", num_tokens=5, num_computed_tokens=4)
    waiting = _request("w0", num_tokens=6, num_computed_tokens=5)
    fake = _FakeScheduler(
        running=[running],
        waiting=[waiting],
        num_spec_tokens=2,
        max_num_scheduled_tokens=2,
    )

    with _spec_pad_env(False):
        output = _schedule(fake)

    # Upstream prefers not scheduling at all over scheduling un-padded.
    assert output.num_scheduled_tokens == {"r0": 1}
    assert len(fake.waiting) == 1


def test_spec_padding_skipped_when_dynamic_sd_lookup_active():
    running = _request("r0", num_tokens=5, num_computed_tokens=4)
    waiting = _request("w0", num_tokens=6, num_computed_tokens=5)
    fake = _FakeScheduler(
        running=[running],
        waiting=[waiting],
        num_spec_tokens=2,
        dynamic_sd_lookup=[3, 3, 3],
    )

    with _spec_pad_env(False):
        output = _schedule(fake)

    assert output.num_scheduled_tokens == {"r0": 1, "w0": 1}
    # Dynamic SD selects K by batch size instead of the padded width.
    assert output.num_spec_tokens_to_schedule == 3


def test_spec_padding_skipped_when_prefill_scheduled():
    running = _request(
        "r0",
        num_tokens=10,
        num_computed_tokens=0,
        is_prefill_chunk=True,
        has_encoder_inputs=True,
    )
    waiting = _request("w0", num_tokens=6, num_computed_tokens=5)
    fake = _FakeScheduler(running=[running], waiting=[waiting], num_spec_tokens=2)

    with _spec_pad_env(False):
        output = _schedule(fake)

    # prefill_scheduled blocks the padding branch even when opted out.
    assert output.num_scheduled_tokens == {"r0": 10, "w0": 1}
    assert output.scheduled_spec_decode_tokens == {}


def test_preemption_when_kv_blocks_unavailable():
    running = _request("r0", num_tokens=5, num_computed_tokens=4)
    waiting = _request("w0", num_tokens=10)
    fake = _FakeScheduler(running=[running], waiting=[waiting])
    fake.kv_cache_manager.fail_after = 0

    output = _schedule(fake)

    # FCFS pops the request itself; the waiting loop is skipped entirely.
    assert fake.preempted == ["r0"]
    assert output.num_scheduled_tokens == {}
    assert output.scheduled_new_reqs == []
    assert len(fake.waiting) == 1


def test_unchunked_prefill_breaks_when_over_budget():
    waiting = _request("w0", num_tokens=100)
    fake = _FakeScheduler(waiting=[waiting], max_num_scheduled_tokens=10)

    output = _schedule(fake)

    assert output.num_scheduled_tokens == {}
    assert len(fake.waiting) == 1


def test_v2_model_runner_builds_new_request_data():
    waiting = _request("w0", num_tokens=10)
    waiting._all_token_ids = list(range(10))
    fake = _FakeScheduler(waiting=[waiting], use_v2_model_runner=True)

    output = _schedule(fake)

    assert output.num_scheduled_tokens == {"w0": 10}
    assert len(output.scheduled_new_reqs) == 1
    new_req = output.scheduled_new_reqs[0]
    assert new_req.req_id == "w0"
    assert new_req.block_ids == [11, 12]
    assert new_req.prefill_token_ids == list(range(10))


def test_throttled_step_defers_prefill_chunks():
    prefill = _request(
        "r0", num_tokens=10, num_computed_tokens=0, is_prefill_chunk=True
    )
    decode = _request("r1", num_tokens=5, num_computed_tokens=4)
    fake = _FakeScheduler(running=[prefill, decode])

    output = _schedule(fake, throttle_prefills=True)

    # DP prefill balancing: decodes still run, the prefill chunk is deferred.
    assert output.num_scheduled_tokens == {"r1": 1}
    assert fake.prefill_capacity_bound is False


def test_running_request_with_spec_tokens():
    running = _request(
        "r0",
        num_tokens=5,
        num_computed_tokens=4,
        spec_token_ids=[7, 8, 9],
    )
    running.num_tokens_with_spec = 7
    fake = _FakeScheduler(running=[running])

    output = _schedule(fake)

    # 3 scheduled tokens cover 1 real + 2 spec tokens.
    assert output.num_scheduled_tokens == {"r0": 3}
    assert output.scheduled_spec_decode_tokens == {"r0": [7, 8]}
    assert running.spec_token_ids == []


def test_running_request_skipped_when_no_new_tokens():
    running = _request("r0", num_tokens=5, num_computed_tokens=5)
    fake = _FakeScheduler(running=[running])

    output = _schedule(fake)

    assert output.num_scheduled_tokens == {}


def test_running_request_deferred_by_decode_eligible_step():
    running = _request("r0", num_tokens=5, num_computed_tokens=4)
    running.next_decode_eligible_step = 10
    fake = _FakeScheduler(running=[running])

    output = _schedule(fake)

    assert output.num_scheduled_tokens == {}


def test_running_request_skipped_when_output_placeholders_done():
    running = _request("r0", num_tokens=5, num_computed_tokens=10)
    running.num_output_placeholders = 2
    running.num_prompt_tokens = 5
    running.max_tokens = 5
    fake = _FakeScheduler(running=[running])

    output = _schedule(fake)

    assert output.num_scheduled_tokens == {}


def test_long_prefill_threshold_caps_running_request():
    running = _request("r0", num_tokens=20, num_computed_tokens=0)
    fake = _FakeScheduler(
        running=[running],
        scheduler_config=SimpleNamespace(
            long_prefill_token_threshold=3,
            enable_chunked_prefill=False,
        ),
    )

    output = _schedule(fake)

    assert output.num_scheduled_tokens == {"r0": 3}


def test_running_request_capped_by_max_model_len():
    running = _request("r0", num_tokens=20, num_computed_tokens=10)
    fake = _FakeScheduler(running=[running], max_model_len=12)

    output = _schedule(fake)

    # num_new_tokens is capped to max_model_len - computed - sampled(1).
    assert output.num_scheduled_tokens == {"r0": 1}


def test_mamba_block_aligned_split_zero_breaks_waiting_loop():
    waiting = _request("w0", num_tokens=10)
    fake = _FakeScheduler(
        waiting=[waiting], need_mamba_block_aligned_split=True
    )
    fake._mamba_block_aligned_split = lambda *args: 0

    output = _schedule(fake)

    assert output.num_scheduled_tokens == {}
    assert len(fake.waiting) == 1


def test_priority_policy_preempts_scheduled_request():
    high = _request(
        "r_high",
        num_tokens=10,
        num_computed_tokens=0,
        is_prefill_chunk=True,
        has_encoder_inputs=True,
    )
    high.priority = 5
    low = _request("r_low", num_tokens=5, num_computed_tokens=4)
    fake = _FakeScheduler(
        running=[high, low],
        policy=SchedulingPolicy.PRIORITY,
        waiting=[_request("w0", num_tokens=10)],
    )
    fake.kv_cache_manager.fail_after = 1

    output = _schedule(fake)

    # r_high is scheduled first, then preempted to make room for r_low,
    # which is preempted in turn when allocation keeps failing.
    assert fake.preempted == ["r_high", "r_low"]
    assert output.num_scheduled_tokens == {}
    assert output.scheduled_spec_decode_tokens == {}
    assert len(fake.waiting) == 1


def test_blocked_waiting_request_is_requeued():
    waiting = _request(
        "w0", num_tokens=10, status=RequestStatus.WAITING_FOR_REMOTE_KVS
    )
    fake = _FakeScheduler(waiting=[waiting])
    fake._is_blocked_waiting_status = lambda status: True
    fake._try_promote_blocked_waiting_request = lambda request: False

    output = _schedule(fake)

    assert output.num_scheduled_tokens == {}
    assert len(fake.waiting) == 0
    assert any(req.request_id == "w0" for req in fake.skipped_waiting)


def test_waiting_request_with_pending_mm_prefetch_is_skipped():
    ec_calls = []
    waiting = _request("w0", num_tokens=10)
    waiting.mm_features = [object()]
    fake = _FakeScheduler(waiting=[waiting])
    fake.ec_connector = SimpleNamespace(
        ensure_cache_available=lambda request, num_computed: False,
        update_state_after_alloc=lambda request, index: ec_calls.append(
            request.request_id
        ),
        build_connector_meta=lambda output: None,
    )

    output = _schedule(fake)

    assert output.num_scheduled_tokens == {}
    assert any(req.request_id == "w0" for req in fake.skipped_waiting)


def test_waiting_request_with_encoder_inputs():
    ec_calls = []
    waiting = _request("w0", num_tokens=10, has_encoder_inputs=True)
    fake = _FakeScheduler(waiting=[waiting], is_encoder_decoder=True)
    fake.ec_connector = SimpleNamespace(
        update_state_after_alloc=lambda request, index: ec_calls.append(
            request.request_id
        ),
        build_connector_meta=lambda output: None,
    )
    # Also reports an external encoder input to load remotely.
    def _schedule_encoder_inputs(
        request, num_computed_tokens, num_new_tokens, budget, *,
        shift_computed_tokens,
    ):
        fake.encoder_schedule_calls.append(request.request_id)
        return [0], num_new_tokens, budget, [1]

    fake._try_schedule_encoder_inputs = _schedule_encoder_inputs

    output = _schedule(fake)

    assert output.num_scheduled_tokens == {"w0": 10}
    assert output.scheduled_encoder_inputs == {"w0": [0]}
    assert fake.encoder_schedule_calls == ["w0"]
    assert fake.encoder_cache_manager.allocated == [("w0", 0), ("w0", 1)]
    assert ec_calls == ["w0", "w0"]


def test_waiting_loop_breaks_at_max_num_running_reqs():
    running = _request("r0", num_tokens=5, num_computed_tokens=4)
    waiting = _request("w0", num_tokens=10)
    fake = _FakeScheduler(
        running=[running], waiting=[waiting], max_num_running_reqs=1
    )

    output = _schedule(fake)

    assert output.num_scheduled_tokens == {"r0": 1}
    assert len(fake.waiting) == 1


def test_connector_sync_prefix_cache_match():
    stats_calls = []
    connector_calls = []
    waiting = _request("w0", num_tokens=10)
    fake = _FakeScheduler(waiting=[waiting])
    fake.connector = SimpleNamespace(
        get_num_new_matched_tokens=lambda request, num_computed: (2, False),
        update_state_after_alloc=lambda request, blocks, num_external:
            connector_calls.append((request.request_id, num_external)),
    )
    fake.connector_prefix_cache_stats = SimpleNamespace(
        record=lambda **kwargs: stats_calls.append(kwargs)
    )

    output = _schedule(fake)

    # 2 tokens come from the external cache, only 8 are computed locally.
    assert output.num_scheduled_tokens == {"w0": 8}
    assert waiting.num_computed_tokens == 2
    assert connector_calls == [("w0", 2)]
    assert stats_calls == [
        {"num_tokens": 10, "num_hits": 2, "preempted": False}
    ]
    assert fake.kv_cache_manager.allocate_calls[0][2][
        "num_external_computed_tokens"
    ] == 2


def test_connector_async_kv_load_defers_scheduling():
    waiting = _request("w0", num_tokens=10)
    fake = _FakeScheduler(waiting=[waiting])
    fake.connector = SimpleNamespace(
        get_num_new_matched_tokens=lambda request, num_computed: (3, True),
        update_state_after_alloc=lambda request, blocks, num_external: None,
    )

    output = _schedule(fake)

    # Async KV load: no local work this step, request parked for the transfer.
    assert output.num_scheduled_tokens == {}
    assert waiting.status == RequestStatus.WAITING_FOR_REMOTE_KVS
    assert waiting.num_computed_tokens == 3
    assert fake.running == []
    assert any(req.request_id == "w0" for req in fake._inflight_prefills)
    assert any(req.request_id == "w0" for req in fake.skipped_waiting)
    assert fake.kv_cache_manager.allocate_calls[0][2]["delay_cache_blocks"] is True


def test_lora_requests_are_collected():
    running = _request("r0", num_tokens=5, num_computed_tokens=4)
    running.lora_request = SimpleNamespace(lora_int_id=3)
    waiting = _request("w0", num_tokens=10)
    waiting.lora_request = SimpleNamespace(lora_int_id=3)
    fake = _FakeScheduler(
        running=[running],
        waiting=[waiting],
        lora_config=SimpleNamespace(max_loras=1),
    )

    output = _schedule(fake)

    assert output.num_scheduled_tokens == {"r0": 1, "w0": 10}


def test_lora_budget_exceeded_skips_waiting_request():
    running = _request("r0", num_tokens=5, num_computed_tokens=4)
    running.lora_request = SimpleNamespace(lora_int_id=1)
    waiting = _request("w0", num_tokens=10)
    waiting.lora_request = SimpleNamespace(lora_int_id=2)
    fake = _FakeScheduler(
        running=[running],
        waiting=[waiting],
        lora_config=SimpleNamespace(max_loras=1),
    )

    output = _schedule(fake)

    assert output.num_scheduled_tokens == {"r0": 1}
    assert any(req.request_id == "w0" for req in fake.skipped_waiting)


def test_kv_cache_zeroing_and_defer_block_free():
    waiting = _request("w0", num_tokens=10)
    fake = _FakeScheduler(
        waiting=[waiting],
        needs_kv_cache_zeroing=True,
        defer_block_free=True,
    )
    fake.kv_cache_manager.new_block_ids = [5]

    output = _schedule(fake)

    assert output.new_block_ids_to_zero == [5]
    assert fake.sched_step_seq == 1


def test_invalid_waiting_status_raises():
    waiting = _request("w0", num_tokens=10, status=RequestStatus.RUNNING)
    fake = _FakeScheduler(waiting=[waiting])

    with pytest.raises(RuntimeError, match="Invalid request status"):
        _schedule(fake)


def test_mamba_hybrid_prefix_cache_hit():
    from vllm.v1.core.kv_cache_coordinator import HybridKVCacheCoordinator

    stats_calls = []
    connector_calls = []
    waiting = _request("w0", num_tokens=10)
    fake = _FakeScheduler(waiting=[waiting], has_mamba_layers=True)
    # Real coordinator instance (bypassing __init__) so the isinstance check
    # in the fork takes the hybrid branch.
    coordinator = HybridKVCacheCoordinator.__new__(HybridKVCacheCoordinator)
    coordinator.find_longest_cache_hit_per_group = (
        lambda block_hashes, num_tokens: ("hit-blocks", [3])
    )
    fake.kv_cache_manager.coordinator = coordinator
    fake.kv_cache_manager.log_stats = True
    fake.kv_cache_manager.prefix_cache_stats = SimpleNamespace(
        record=lambda **kwargs: stats_calls.append(kwargs)
    )
    fake.connector = SimpleNamespace(
        get_num_new_matched_tokens=lambda request, num_computed: (0, False),
        update_state_after_alloc=lambda request, blocks, num_external:
            connector_calls.append(request.request_id),
    )

    output = _schedule(fake)

    # 3 tokens come from the FA prefix hit, only 7 are computed locally.
    assert output.num_scheduled_tokens == {"w0": 7}
    assert waiting.num_computed_tokens == 3
    assert stats_calls == [
        {"num_tokens": 10, "num_hits": 3, "preempted": False}
    ]
    assert connector_calls == ["w0"]


def test_connector_unmatched_request_is_requeued():
    waiting = _request("w0", num_tokens=10)
    fake = _FakeScheduler(waiting=[waiting])
    fake.connector = SimpleNamespace(
        get_num_new_matched_tokens=lambda request, num_computed: (None, False),
        update_state_after_alloc=lambda request, blocks, num_external: None,
    )

    output = _schedule(fake)

    assert output.num_scheduled_tokens == {}
    assert any(req.request_id == "w0" for req in fake.skipped_waiting)


def test_waiting_long_prefill_threshold():
    waiting = _request("w0", num_tokens=20)
    fake = _FakeScheduler(
        waiting=[waiting],
        scheduler_config=SimpleNamespace(
            long_prefill_token_threshold=3,
            enable_chunked_prefill=True,
        ),
    )

    output = _schedule(fake)

    assert output.num_scheduled_tokens == {"w0": 3}


def test_waiting_encoder_request_frees_cache_on_alloc_failure():
    waiting = _request("w0", num_tokens=10, has_encoder_inputs=True)
    fake = _FakeScheduler(waiting=[waiting])
    fake.kv_cache_manager.fail_after = 0

    output = _schedule(fake)

    # The encoder cache is untouched again when the request cannot be served.
    assert output.num_scheduled_tokens == {}
    assert fake.encoder_schedule_calls == ["w0"]
    assert fake.encoder_cache_manager.freed == ["w0"]
    assert len(fake.waiting) == 1


def test_throttled_step_defers_waiting_prefill():
    decode = _request("r0", num_tokens=5, num_computed_tokens=4)
    waiting = _request("w0", num_tokens=10)
    fake = _FakeScheduler(running=[decode], waiting=[waiting])

    output = _schedule(fake, throttle_prefills=True)

    # DP prefill balancing: the waiting prefill is deferred to an
    # aligned step, only the decode runs.
    assert output.num_scheduled_tokens == {"r0": 1}
    assert len(fake.waiting) == 1


def test_waiting_encoder_budget_exhausted_breaks():
    waiting = _request("w0", num_tokens=10, has_encoder_inputs=True)
    fake = _FakeScheduler(waiting=[waiting])
    fake._try_schedule_encoder_inputs = (
        lambda *args, **kwargs: (None, 0, args[3], [])
    )

    output = _schedule(fake)

    assert output.num_scheduled_tokens == {}


if __name__ == "__main__":
    run_standalone_tests(globals())
