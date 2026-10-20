# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.

"""Unit tests for patch_realtime_kv, with external interfaces isolated."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest


def _load_patch(patch_env):
    targets = {
        "vllm.v1.core.kv_cache_coordinator": "HybridKVCacheCoordinator",
        "vllm.v1.core.kv_cache_manager": "KVCacheManager",
        "vllm.v1.core.sched.scheduler": "Scheduler",
        "vllm.v1.engine.core": "EngineCore",
        "vllm.v1.outputs": "ModelRunnerOutput",
        "omni_npu.connector.llmdatadist_connector_v1": "LLMDataDistConnector",
    }
    for module_name, class_name in targets.items():
        patch_env.stub(module_name, **{class_name: type(class_name, (), {})})
    session = patch_env.stub(
        "omni_models.models.openpangu_vl.realtime_video.kv_session",
        close_session=Mock(), finish_request=Mock(), get_store=Mock(),
        realtime_swa_reserve_blocks=Mock(return_value=5),
    )
    patch_env.stub(
        "omni_models.models.openpangu_vl.realtime_video.metadata",
        MROPE_DELTA_KEY="mrope_delta", POSITION_OFFSET_KEY="position_offset",
        is_realtime_kv_request=lambda request: getattr(
            request, "is_realtime", False
        ),
        kv_reuse_plan=Mock(), realtime_kv_metadata=Mock(),
        request_extra_args=Mock(),
    )
    module = patch_env.load(
        "patch_realtime_kv", group="openpangu_vl_realtime_video"
    )
    return module, session


@pytest.mark.parametrize("raise_inside", [False, True])
def test_boundary_reservation_is_restored(patch_env, raise_inside):
    module, session = _load_patch(patch_env)
    mome = SimpleNamespace(
        kv_cache_spec=type("MomeSpec", (), {})(), num_extra_reserved_blocks=0,
    )
    swa = SimpleNamespace(
        kv_cache_spec=type("ShareKVSlidingWindowSpec", (), {})(),
        num_extra_reserved_blocks=2, block_size=128,
    )
    manager = SimpleNamespace(
        coordinator=SimpleNamespace(single_type_managers=[mome, swa])
    )

    def reserve():
        with module._retain_realtime_kv_boundaries(manager, 107976, 107843):
            assert mome.num_extra_reserved_blocks == 1
            assert swa.num_extra_reserved_blocks == 7
            if raise_inside:
                raise RuntimeError("operation failed")

    if raise_inside:
        with pytest.raises(RuntimeError, match="operation failed"):
            reserve()
    else:
        reserve()
    assert (mome.num_extra_reserved_blocks, swa.num_extra_reserved_blocks) == (0, 2)
    session.realtime_swa_reserve_blocks.assert_called_once_with(
        107976, 107843, 128
    )


@pytest.mark.parametrize("is_realtime", [False, True])
def test_allocation_reserves_boundary_only_for_realtime(patch_env, is_realtime):
    module, _ = _load_patch(patch_env)
    mome = SimpleNamespace(
        kv_cache_spec=type("MomeSpec", (), {})(), num_extra_reserved_blocks=0,
    )
    manager = SimpleNamespace(
        coordinator=SimpleNamespace(single_type_managers=[mome])
    )
    request = SimpleNamespace(is_realtime=is_realtime)
    observed = []
    result = object()

    def allocate(owner, given_request, count):
        assert owner is manager and given_request is request and count == 3
        observed.append(mome.num_extra_reserved_blocks)
        return result

    patch = module.RealtimeKVComputedBlocksPatch
    patch._upstream_allocate_slots = allocate
    assert patch.allocate_slots(manager, request, 3) is result
    assert observed == [int(is_realtime)]
    assert mome.num_extra_reserved_blocks == 0


def test_finish_patch_wraps_connector_and_keeps_free_request(patch_env):
    module, _ = _load_patch(patch_env)
    target = module.Scheduler
    original_free = Mock()
    original_connector = Mock()
    target._free_request = original_free
    target._connector_finished = original_connector
    target.add_request = Mock()
    target.update_from_output = Mock()

    patch = module.RealtimeKVFinishRequestPatch
    patch.apply()

    assert target._free_request is original_free
    assert target._connector_finished is patch._connector_finished
    assert patch._upstream_connector_finished is original_connector


@pytest.mark.parametrize("failure", [None, "snapshot", "connector"])
def test_finish_preserves_snapshot_order_and_cleans_lookup(patch_env, failure):
    module, session = _load_patch(patch_env)
    request = SimpleNamespace(block_hashes=[])
    retained = {id(request.block_hashes): request, 99: "another request"}
    scheduler = SimpleNamespace(kv_cache_manager=SimpleNamespace(
        coordinator=SimpleNamespace(_realtime_requests_by_hashes=retained)
    ))
    events = []
    transfer_result = (True, {"kv": 1})

    def snapshot(owner, finished):
        assert owner is scheduler and finished is request
        events.append("snapshot")
        if failure == "snapshot":
            raise RuntimeError("snapshot failed")

    def connector(owner, finished):
        assert owner is scheduler and finished is request
        events.append("connector")
        if failure == "connector":
            raise RuntimeError("connector failed")
        return transfer_result

    session.finish_request.side_effect = snapshot
    patch = module.RealtimeKVFinishRequestPatch
    patch._upstream_connector_finished = connector
    if failure:
        with pytest.raises(RuntimeError, match=f"{failure} failed"):
            patch._connector_finished(scheduler, request)
    else:
        assert patch._connector_finished(scheduler, request) is transfer_result
    expected = ["snapshot"] if failure == "snapshot" else ["snapshot", "connector"]
    assert events == expected
    assert retained == {99: "another request"}


def test_finish_works_without_realtime_lookup(patch_env):
    module, session = _load_patch(patch_env)
    scheduler = SimpleNamespace(
        kv_cache_manager=SimpleNamespace(coordinator=SimpleNamespace())
    )
    request = SimpleNamespace(block_hashes=[])
    patch = module.RealtimeKVFinishRequestPatch
    patch._upstream_connector_finished = Mock(return_value=(False, None))

    assert patch._connector_finished(scheduler, request) == (False, None)
    session.finish_request.assert_called_once_with(scheduler, request)


def test_mrope_delta_is_updated_before_upstream_scheduler(patch_env):
    module, _ = _load_patch(patch_env)
    realtime = SimpleNamespace(
        is_realtime=True, sampling_params=SimpleNamespace(extra_args={})
    )
    ordinary = SimpleNamespace(
        is_realtime=False, sampling_params=SimpleNamespace(extra_args={})
    )
    scheduler = SimpleNamespace(requests={"rt": realtime, "normal": ordinary})
    output = SimpleNamespace(
        _realtime_pd_mrope_deltas={"rt": 43, "normal": 7, "missing": 9}
    )

    def upstream(owner, scheduled, received):
        assert owner is scheduler and scheduled == "schedule"
        assert received is output
        assert realtime.sampling_params.extra_args == {"mrope_delta": 43}
        assert ordinary.sampling_params.extra_args == {}
        return "updated"

    patch = module.RealtimeKVFinishRequestPatch
    patch._upstream_update_from_output = upstream
    assert patch.update_from_output(scheduler, "schedule", output) == "updated"
