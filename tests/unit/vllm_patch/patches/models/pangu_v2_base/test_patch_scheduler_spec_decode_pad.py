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
import re
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from vllm.v1.core.sched.interface import PauseState
from vllm.v1.core.sched.request_queue import SchedulingPolicy
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.request import RequestStatus

from omni_npu.diagnostics import kv_conservation
from omni_npu.vllm_patches.patches.models.pangu_v2_base import (
    patch_scheduler_spec_decode_pad as sched_mod,
)

GUARD = "OMNI_DISABLE_SPEC_DECODE_PAD"

# The fork deviates from upstream with omni-npu comment lines plus the
# offloading-connector guard clause; the drift test strips them before
# diffing against upstream.
FORK_MARKERS = (
    "omni-npu:",
    "padded to the full spec-decode",
    "dummy draft slots",
)

# The fork additionally carries the KV-conservation periodic check at the very
# tail of schedule() (it owns the schedule attribute, so the old
# KVConservationSchedulerPatch wrapper could no longer compose). The drift
# test cuts that tail from the fork -- the "# patch start/end" block, whose
# lines carry no omni-npu marker -- before the marker filter and upstream
# diffing; its behaviour is guarded by the test_kv_conservation_* tests
# below instead.
KV_CHECK_START = "# patch start: for kv conservation check"
KV_CHECK_END = "# patch end: for kv conservation check"

# The fork additionally absorbs a later upstream fix (reworked pad branch:
# `padded_num_tokens` + reserving `num_sampled_tokens_per_step` room near
# max_model_len) that the pinned vendored vLLM revision (752a3a50) predates.
# The drift test therefore cuts the whole pad branch out of both sides;
# its behaviour is guarded by the test_spec_padding_* tests below instead.
PAD_BRANCH_START = "# Pad new decode requests"
PAD_BRANCH_END = "pad_spec_decode = True"

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


# Recording stub for kv_conservation.check. The fork reads the attribute
# through the module object it imported, so rebinding it there redirects both
# the pytest path (via the kv_check_calls fixture) and the standalone
# `python this_file.py` path (no pytest fixtures available).
_kv_check_calls: list = []


def _stub_kv_check(scheduler, *, trigger):
    _kv_check_calls.append((scheduler, trigger))


def _standalone_kv_stub():
    """Install the module-level recording stub; returns the shared list.

    Only used when the file runs as a script (no pytest fixtures): the
    stub replaces kv_conservation.check process-wide for the remaining
    standalone tests, which is fine because nothing else reads it here.
    """
    kv_conservation.check = _stub_kv_check
    _kv_check_calls.clear()
    return _kv_check_calls


@pytest.fixture
def kv_check_calls(monkeypatch):
    """Point the fork's kv_conservation.check at a per-test recorder.

    due() stays real so the interval alignment logic itself is under test.
    """
    calls = []
    monkeypatch.setattr(
        sched_mod.kv_conservation, "check",
        lambda scheduler, *, trigger: calls.append((scheduler, trigger)),
    )
    return calls


def _env_override(monkeypatch, initial):
    """setenv/restore pair working with or without pytest's monkeypatch."""
    import os

    name = "OMNI_KV_CONSERVATION_CHECK_INTERVAL"

    if monkeypatch is not None:
        monkeypatch.setenv(name, initial)
        # monkeypatch undoes every setenv at teardown itself.
        return monkeypatch.setenv, lambda: None

    old = os.environ.get(name)

    def _set(value):
        os.environ[name] = value

    def _restore():
        if old is None:
            os.environ.pop(name, None)
        else:
            os.environ[name] = old

    _set(initial)
    return _set, _restore


# --------------------------------------------------------------------------
# structural checks
# --------------------------------------------------------------------------


def test_patch_targets_scheduler_schedule():
    assert sched_mod.PanguV2SpecDecodePadPatch._target is Scheduler
    assert sched_mod.PanguV2SpecDecodePadPatch._attr_names_to_apply == ["schedule"]


def test_fork_pads_only_when_batch_or_history_exists():
    source = inspect.getsource(sched_mod.PanguV2SpecDecodePadPatch.schedule)
    # Padding must not fire for the very first fresh request of the engine
    # (nothing running, nothing computed): the extra dummy draft slots break
    # the NPU graph step.
    assert "scheduled_running_reqs or num_computed_tokens > 0" in source
    # Offloading connectors assemble prefix hits through the cache manager,
    # so the locally-cached-tokens hybrid branch must be skipped for them.
    assert "_omni_is_offloading_connector" in source


def test_fork_matches_upstream_except_the_guard():
    if not hasattr(Scheduler, "schedule"):
        if _STANDALONE:
            return
        pytest.skip("Scheduler stub without upstream schedule()")

    def normalized(func):
        text = inspect.getsource(func)
        # Cut the fork-only KV-conservation tail first: its omni-npu comment
        # block has marker-less continuation lines, so it must go before the
        # FORK_MARKERS filter. The cut spans the preceding blank line down to
        # the last check call, and stops before the final
        # `return scheduler_output`, which both sides share.
        try:
            start = text.index(KV_CHECK_START)
            end = text.index(KV_CHECK_END, start)
        except ValueError:
            pass
        else:
            block_start = text.rindex("\n\n", 0, start) + 1
            text = text[:block_start] + text[end + len(KV_CHECK_END):].lstrip("\n")
        # Canonicalize the intentionally-deviating pad condition: upstream
        # keeps `scheduled_running_reqs` on a single line, the fork splits it
        # to add the `num_computed_tokens > 0` arm.
        text = re.sub(
            r"\n\s*and not prefill_scheduled"
            r"\n\s*and \(scheduled_running_reqs or num_computed_tokens > 0\)",
            "\nPAD_CONDITION",
            text,
        )
        text = re.sub(
            r"\n\s*and \(scheduled_running_reqs and not prefill_scheduled\)",
            "\nPAD_CONDITION",
            text,
        )
        lines = [line.rstrip() for line in text.splitlines()]
        while lines and not lines[-1].strip():
            lines.pop()
        # Drop the offloading-connector guard clause and the omni comment
        # lines so the fork can be diffed against upstream even when the
        # local vLLM checkout still carries the raw os.environ variant.
        lines = [
            line for line in lines
            if GUARD not in line
            and "_omni_is_offloading_connector" not in line
            and not any(m in line for m in FORK_MARKERS)
        ]
        # Cut the whole pad branch from both sides: the fork carries the
        # newer upstream rewrite (padded_num_tokens / max_model_len headroom)
        # that the pinned vendored vLLM revision does not have yet.
        try:
            start = next(
                i for i, line in enumerate(lines) if PAD_BRANCH_START in line
            )
            end = next(
                i for i, line in enumerate(lines[start:], start)
                if PAD_BRANCH_END in line
            )
        except StopIteration:
            return lines
        return lines[:start] + lines[end + 1:]

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


def test_spec_padding_applied_for_waiting_request_with_running_batch():
    running = _request("r0", num_tokens=5, num_computed_tokens=4)
    # First decode after a chunked prefill: exactly one token left.
    waiting = _request("w0", num_tokens=6, num_computed_tokens=5)
    fake = _FakeScheduler(running=[running], waiting=[waiting], num_spec_tokens=2)

    output = _schedule(fake)

    # Padding only happens in the waiting loop: the running decode stays at
    # 1 token, the waiting request is padded via the `scheduled_running_reqs`
    # arm of the condition.
    assert output.num_scheduled_tokens == {"r0": 1, "w0": 3}
    assert output.scheduled_spec_decode_tokens == {"w0": [-1, -1]}
    assert fake.kv_cache_manager.allocate_calls[-1][1] == 3


def test_spec_padding_applied_for_computed_request_without_running_reqs():
    # The `num_computed_tokens > 0` arm: a request whose KV history was
    # already computed/loaded (e.g. resumed after a remote KV transfer) is
    # padded even when nothing else is running.
    waiting = _request("w0", num_tokens=6, num_computed_tokens=5)
    fake = _FakeScheduler(waiting=[waiting], num_spec_tokens=2)

    output = _schedule(fake)

    assert output.num_scheduled_tokens == {"w0": 3}
    assert output.scheduled_spec_decode_tokens == {"w0": [-1, -1]}
    assert fake.kv_cache_manager.allocate_calls[-1][1] == 3


def test_spec_padding_skipped_for_fresh_first_request():
    # Nothing running and nothing computed yet: the very first request of
    # the engine stays un-padded so no dummy draft slots are scheduled.
    waiting = _request("w0", num_tokens=1)
    fake = _FakeScheduler(waiting=[waiting], num_spec_tokens=2)

    output = _schedule(fake)

    assert output.num_scheduled_tokens == {"w0": 1}
    assert output.scheduled_spec_decode_tokens == {}


def test_spec_padding_breaks_when_budget_insufficient():
    running = _request("r0", num_tokens=5, num_computed_tokens=4)
    waiting = _request("w0", num_tokens=6, num_computed_tokens=5)
    fake = _FakeScheduler(
        running=[running],
        waiting=[waiting],
        num_spec_tokens=2,
        max_num_scheduled_tokens=2,
    )

    output = _schedule(fake)

    # r0 schedules un-padded, then the padded w0 no longer fits the budget:
    # upstream prefers not scheduling at all over scheduling un-padded.
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

    output = _schedule(fake)

    # prefill_scheduled blocks the padding branch.
    assert output.num_scheduled_tokens == {"r0": 10, "w0": 1}
    assert output.scheduled_spec_decode_tokens == {}


def test_spec_padding_near_max_model_len_keeps_unpadded():
    running = _request("r0", num_tokens=5, num_computed_tokens=4)
    # w0 is 1 token short of done: padding to 1 + 2 spec tokens plus the
    # 1 sampled token per step would run past max_model_len.
    waiting = _request("w0", num_tokens=6, num_computed_tokens=5)
    fake = _FakeScheduler(
        running=[running],
        waiting=[waiting],
        num_spec_tokens=2,
        max_model_len=7,
    )

    output = _schedule(fake)

    # The newer upstream rewrite keeps the unpadded 1-token decode instead
    # of refusing to schedule: 5 + 3 + 1 > 7, so the pad branch is skipped.
    assert output.num_scheduled_tokens == {"r0": 1, "w0": 1}
    assert output.scheduled_spec_decode_tokens == {}
    assert fake.kv_cache_manager.allocate_calls[-1][1] == 1


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



@pytest.mark.parametrize("mome_hit", [0, 1280])
def test_offload_hybrid_missing_mome_waits_for_load(mome_hit):
    from vllm.v1.core.kv_cache_coordinator import HybridKVCacheCoordinator

    waiting = _request("missing-mome", num_tokens=1536)
    fake = _FakeScheduler(waiting=[waiting], has_mamba_layers=True)
    manager = fake.kv_cache_manager
    manager._omni_is_offloading_connector = True
    coordinator = HybridKVCacheCoordinator.__new__(HybridKVCacheCoordinator)
    coordinator.find_longest_cache_hit_per_group = (
        lambda *_: ("uneven-hits", [1408, mome_hit, 1408])
    )
    manager.coordinator = coordinator
    lookups = []
    loads = []

    def get_computed(request):
        lookups.append(request.request_id)
        request.local_computed_tokens_per_group = (1408, mome_hit, 1408)
        return _BlockIds([]), mome_hit

    manager.get_computed_blocks = get_computed
    fake.connector = SimpleNamespace(
        get_num_new_matched_tokens=lambda request, local: (
            1408 - local, local < 1408
        ),
        update_state_after_alloc=lambda request, blocks, external: loads.append(external),
    )

    output = _schedule(fake)

    # FA hits do not make a missing MOME history safe to read. The request
    # must wait for the missing group's load before any local forward.
    assert output.num_scheduled_tokens == {}
    assert lookups == ["missing-mome"]
    assert loads == [1408 - mome_hit]
    assert waiting.status == RequestStatus.WAITING_FOR_REMOTE_KVS
    assert waiting.num_computed_tokens == 1408
    assert waiting in fake._inflight_prefills
    allocation = manager.allocate_calls[0][2]
    assert allocation["delay_cache_blocks"] is True
    assert allocation["num_new_computed_tokens"] == mome_hit
    assert allocation["num_external_computed_tokens"] == 1408 - mome_hit


# --------------------------------------------------------------------------
# KV-conservation periodic check (merged from the KVConservationSchedulerPatch
# wrapper that used to wrap the same schedule attribute)
# --------------------------------------------------------------------------


def test_kv_conservation_check_fires_on_interval_aligned_step(kv_check_calls=None):
    calls = kv_check_calls if kv_check_calls is not None else _standalone_kv_stub()
    interval = kv_conservation.interval()
    fake = _FakeScheduler(current_step=interval - 1)

    output = _schedule(fake)

    # After the increment, current_step is aligned with the interval, so the
    # audit fires once, after _update_after_schedule and before returning.
    assert calls == [(fake, "step")]
    assert fake.updated_after_schedule == [output]


def test_kv_conservation_check_skipped_on_unaligned_step(kv_check_calls=None):
    calls = kv_check_calls if kv_check_calls is not None else _standalone_kv_stub()
    interval = kv_conservation.interval()
    fake = _FakeScheduler(current_step=2 * interval)

    _schedule(fake)

    # Non-aligned steps carry no audit overhead past one due() probe.
    assert calls == []


@pytest.mark.parametrize("interval", [0, -1])
def test_kv_conservation_check_disabled(kv_check_calls, monkeypatch, interval):
    _set, _restore = _env_override(monkeypatch, str(interval))
    fake = _FakeScheduler(current_step=0)

    try:
        _schedule(fake)
    finally:
        _restore()

    # interval <= 0 disables the trigger entirely (due() is never consulted).
    assert kv_check_calls == []


def test_kv_conservation_env_read_lazily_each_call(kv_check_calls=None, monkeypatch=None):
    """_kv_conservation_interval() reads envs on every step, so runtime
    changes to OMNI_KV_CONSERVATION_CHECK_INTERVAL apply without restart."""
    calls = kv_check_calls if kv_check_calls is not None else _standalone_kv_stub()
    setenv, restore_env = _env_override(monkeypatch, "2")
    fake = _FakeScheduler(current_step=1)

    try:
        _schedule(fake)  # lands on step 2: aligned -> fires
        assert len(calls) == 1

        setenv("0")
        _schedule(fake)  # lands on step 4 but disabled mid-run -> silent
        assert len(calls) == 1
    finally:
        restore_env()


if __name__ == "__main__":
    run_standalone_tests(globals())
