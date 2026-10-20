# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.

import hashlib
import json
import os
from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch

from omni_npu.connector.kv_dump import (
    append_items,
    DecodeReq,
    Dump,
    gen_hash,
    load_layers,
    maybe_dump_kv,
    parse_remote_reqs,
    PrefillReq,
    _diff_stat,
    _id_set,
    _log_mismatch,
)

KV_DUMP_MODULE = "omni_npu.connector.kv_dump"

pytestmark = pytest.mark.unit


@pytest.fixture(autouse=True)
def _reset_maybe_dump_kv_singleton():
    maybe_dump_kv.__dict__.pop("dump_prefill", None)
    yield
    maybe_dump_kv.__dict__.pop("dump_prefill", None)

cfg_i32 = {"device": "cpu", "dtype": torch.int32}
cfg_uint8 = {"device": "cpu", "dtype": torch.uint8}
cfg_f32 = {"device": "cpu", "dtype": torch.float32}

# vllm extract_layer_index 要求层名中仅含一个整数
DEFAULT_LAYER_NAME = "model.layers.0.self_attn"


# =========================
# 1. 无效果或简单 mock
# =========================

def _layer_index_from_name(name: str) -> int:
    digits = "".join(ch for ch in name if ch.isdigit())
    return int(digits) if digits else 0


@contextmanager
def _patch_kv_dump_misc():
    with patch(f"{KV_DUMP_MODULE}.extract_layer_index", side_effect=_layer_index_from_name):
        yield


@contextmanager
def _patch_dump_thread(*, run_id="1704067200"):
    mock_thread = MagicMock()
    with (
        patch(f"{KV_DUMP_MODULE}.threading.Thread", mock_thread),
        patch(f"{KV_DUMP_MODULE}.time.sleep", MagicMock()),
        patch(f"{KV_DUMP_MODULE}._make_run_id", return_value=run_id),
    ):
        yield mock_thread


@contextmanager
def _mock_world_group(rank=0):
    with patch(
        f"{KV_DUMP_MODULE}.get_world_group",
        return_value=SimpleNamespace(rank=rank),
    ):
        yield


# =========================
# 2. 共享 fixture 构造（供后续类测试复用）
# =========================

def _make_uint8_cache(
    num_blocks: int = 4,
    block_elems: int = 32,
) -> torch.Tensor:
    return torch.arange(
        num_blocks * block_elems, **cfg_i32
    ).reshape(num_blocks, block_elems)


def _make_shapes(num_blocks: int = 4, block_elems: int = 32) -> list[list[torch.Tensor]]:
    cache = _make_uint8_cache(num_blocks, block_elems)
    return load_layers([cache])


def _make_new_request(
    req_id: str = "req0",
    tokens: int = 128,
    block_ids=None,
    computed: int = 0,
) -> SimpleNamespace:
    if block_ids is None:
        block_ids = [[1, 2]]
    return SimpleNamespace(
        req_id=req_id,
        prompt_token_ids=list(range(tokens)),
        block_ids=block_ids,
        num_computed_tokens=computed,
    )


def _make_scheduler_output_prefill(
    req_ids=None,
    new_block_ids=None,
    new_reqs=None,
) -> MagicMock:
    if req_ids is None:
        req_ids = []
    if new_block_ids is None:
        new_block_ids = [None] * len(req_ids)
    cached = SimpleNamespace(
        req_ids=req_ids,
        new_block_ids=new_block_ids,
        num_computed_tokens=[],
    )
    return MagicMock(
        scheduled_cached_reqs=cached,
        scheduled_new_reqs=new_reqs or [],
        kv_connector_metadata=SimpleNamespace(req_params={}),
    )


def _make_scheduler_output_decode(
    req_ids=None,
    computed=None,
    new_block_ids=None,
    new_reqs=None,
    req_params=None,
    allocated_blocks=None,
    resumed_req_ids=None,
) -> MagicMock:
    if req_ids is None:
        req_ids = []
    if computed is None:
        computed = [0] * len(req_ids)
    if new_block_ids is None:
        new_block_ids = [None] * len(req_ids)
    meta = SimpleNamespace(req_params=req_params if req_params is not None else {})
    cached = SimpleNamespace(
        req_ids=req_ids,
        new_block_ids=new_block_ids,
        num_computed_tokens=computed,
        resumed_req_ids=resumed_req_ids or set(),
    )
    return MagicMock(
        scheduled_cached_reqs=cached,
        scheduled_new_reqs=new_reqs or [],
        kv_connector_metadata=meta,
        new_block_ids_to_zero=allocated_blocks,
    )


def _make_model_runner(*, kv_role: str = "kv_producer", spec_tokens: int = 0) -> MagicMock:
    layer = SimpleNamespace(kv_cache=[(_make_uint8_cache(),)])
    runner = MagicMock()
    runner.compilation_config.static_forward_context = {
        DEFAULT_LAYER_NAME: layer,
    }
    runner.kv_cache_config.kv_cache_groups = [
        SimpleNamespace(
            layer_names=[DEFAULT_LAYER_NAME],
            kv_cache_spec=SimpleNamespace(block_size=128),
        ),
    ]
    runner.vllm_config.kv_transfer_config.kv_role = kv_role
    runner.num_spec_tokens = spec_tokens
    return runner


def _make_prefill_req(
    req_id: str = "req0",
    tokens: int = 128,
    block_ids=None,
) -> PrefillReq:
    return PrefillReq(_make_new_request(req_id, tokens, block_ids))


def _make_decode_req(
    req_id: str = "req0",
    tokens: int = 128,
    block_ids=None,
) -> DecodeReq:
    return DecodeReq(_make_new_request(req_id, tokens, block_ids))


def _make_decode_groups(
    num_blocks: int = 4,
    pg: int = 128,
    block_elems: int = 8,
) -> list[dict[str, tuple[torch.Tensor, ...]]]:
    cache = torch.arange(
        num_blocks * pg * block_elems, **cfg_i32
    ).reshape(num_blocks, pg, block_elems)
    assert cache.device.type == "cpu"
    return [{"layer0": (cache,)}]


def _make_dump(path: str = "/tmp/kv_dump") -> Dump:
    with _patch_dump_thread():
        return Dump(path)


# =========================
# 3. 全局函数
# =========================

class TestGenHash:

    def test_known_vector(self):
        buf = torch.tensor([1, 2, 3, 4], **cfg_uint8)
        kind, digest = gen_hash(buf)
        expected = hashlib.md5(buf.cpu().numpy().tobytes()).hexdigest()
        assert digest == expected
        assert kind == "uint8(4).blk"
        assert buf.device.type == "cpu"

    def test_invalid_dtype(self):
        with pytest.raises(ValueError):
            gen_hash(torch.tensor([1.0, 2.0], **cfg_f32))

    def test_invalid_dim(self):
        with pytest.raises(ValueError):
            gen_hash(torch.zeros(2, 2, **cfg_uint8))


class TestAppendItems:

    def test_empty_items_no_op(self, tmp_path):
        path = str(tmp_path / "out.json")
        append_items(path, {})
        assert not os.path.exists(path)

    def test_first_write(self, tmp_path):
        path = str(tmp_path / "out.json")
        append_items(path, {"req0": '{"a": 1}'})
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        assert data == {"req0": {"a": 1}}

    def test_append_second_batch(self, tmp_path):
        path = str(tmp_path / "out.json")
        append_items(path, {"req0": '{"a": 1}'})
        append_items(path, {"req1": '{"b": 2}'})
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        assert data == {"req0": {"a": 1}, "req1": {"b": 2}}

    def test_invalid_items_type(self, tmp_path):
        path = str(tmp_path / "out.json")
        with pytest.raises(TypeError):
            append_items(path, [])  # type: ignore[arg-type]


class TestLoadLayers:

    def test_list_format(self):
        cache = _make_uint8_cache(num_blocks=4, block_elems=16)
        shapes = load_layers([cache])
        assert len(shapes) == 1
        assert len(shapes[0]) == 1
        assert shapes[0][0].dtype == torch.uint8
        assert shapes[0][0].shape[0] == 4
        assert shapes[0][0].device.type == "cpu"
        assert cache.device.type == "cpu"

    def test_dict_format(self):
        cache = _make_uint8_cache(num_blocks=4, block_elems=16)
        with _patch_kv_dump_misc():
            shapes = load_layers({"model.layers.3.self_attn": cache})
        assert len(shapes) == 1
        assert shapes[0][0].shape[0] == 4

    def test_multi_tensor_per_layer(self):
        t1 = _make_uint8_cache(num_blocks=4, block_elems=8)
        t2 = _make_uint8_cache(num_blocks=4, block_elems=8) + 1
        shapes = load_layers([[t1, t2]])
        assert len(shapes) == 1
        assert len(shapes[0]) == 2

    def test_mismatched_block_num(self):
        t1 = _make_uint8_cache(num_blocks=4, block_elems=8)
        t2 = _make_uint8_cache(num_blocks=5, block_elems=8)
        with pytest.raises(ValueError):
            load_layers([t1, t2])

    def test_invalid_format(self):
        with pytest.raises(TypeError):
            load_layers(42)  # type: ignore[arg-type]


class TestParseRemoteReqs:

    def test_extract_prefill_ids(self):
        meta = SimpleNamespace(req_params={
            "decode_a": {"prefill_req_id": "prefill_a"},
            "decode_b": {"prefill_req_id": "prefill_b"},
        })
        scheduler_output = MagicMock(kv_connector_metadata=meta)
        assert parse_remote_reqs(scheduler_output) == {
            "decode_a": "prefill_a",
            "decode_b": "prefill_b",
        }

    def test_empty_params(self):
        scheduler_output = MagicMock(
            kv_connector_metadata=SimpleNamespace(req_params={}))
        assert parse_remote_reqs(scheduler_output) == {}


class TestDiffStat:

    def test_equal(self):
        a = torch.zeros(8, dtype=torch.uint8)
        n, total, first = _diff_stat(a, a.clone())
        assert (n, total, first) == (0, 8, -1)

    def test_differs(self):
        a = torch.zeros(8, dtype=torch.uint8)
        b = torch.zeros(8, dtype=torch.uint8)
        b[3] = 1
        b[5] = 1
        n, total, first = _diff_stat(a, b)
        assert n == 2
        assert total == 8
        assert first == 3


class TestLogMismatch:

    def test_formats_block(self):
        with patch(f"{KV_DUMP_MODULE}.logger.error") as mock_err:
            _log_mismatch("DECODE BLOCK", "req0", ["full    = 2"], 3)
        msg = mock_err.call_args[0][0]
        assert "KV HASH MISMATCH :: DECODE BLOCK" in msg
        assert "mismatch #3" in msg
        assert "req_id  = req0" in msg
        assert msg.count("!") > 100


class TestMaybeDumpKv:

    def test_no_path_bypasses_dump(self):
        def execute_model(self, scheduler_output, intermediate_tensors=None):
            return "ok"

        with patch(f"{KV_DUMP_MODULE}._kv_dump_path", ""):
            wrapped = maybe_dump_kv("update")(execute_model)
            runner = MagicMock()
            sched = MagicMock()
            assert wrapped(runner, sched) == "ok"
            assert "dump_prefill" not in maybe_dump_kv.__dict__

    def test_with_path_wraps_execute_model(self):
        mock_dump = MagicMock()
        cm = MagicMock()
        cm.__enter__ = MagicMock(return_value=None)
        cm.__exit__ = MagicMock(return_value=False)
        mock_dump.update.return_value = cm

        calls = []

        def execute_model(self, scheduler_output, intermediate_tensors=None):
            calls.append((self, scheduler_output, intermediate_tensors))
            return "result"

        with patch(f"{KV_DUMP_MODULE}._kv_dump_path", "/tmp/kv_dump"):
            with patch(f"{KV_DUMP_MODULE}.Dump", return_value=mock_dump):
                wrapped = maybe_dump_kv("update")(execute_model)

                runner = MagicMock()
                sched = MagicMock()
                tensors = MagicMock()
                assert wrapped(runner, sched, tensors) == "result"
                assert calls == [(runner, sched, tensors)]
                mock_dump.update.assert_called_once_with(runner, sched)
                cm.__enter__.assert_called_once()
                cm.__exit__.assert_called_once()

    def test_singleton_reuse(self):
        mock_dump = MagicMock()
        cm = MagicMock()
        cm.__enter__ = MagicMock(return_value=None)
        cm.__exit__ = MagicMock(return_value=False)
        mock_dump.update.return_value = cm

        def execute_model(self, scheduler_output, intermediate_tensors=None):
            return 1

        with patch(f"{KV_DUMP_MODULE}._kv_dump_path", "/tmp/kv_dump"):
            with patch(f"{KV_DUMP_MODULE}.Dump", return_value=mock_dump) as dump_cls:
                wrapped = maybe_dump_kv("update")(execute_model)
                runner = MagicMock()
                sched = MagicMock()
                wrapped(runner, sched)
                wrapped(runner, sched)
                dump_cls.assert_called_once_with("/tmp/kv_dump")
                assert mock_dump.update.call_count == 2

    def test_invalid_stage_raises(self):
        with pytest.raises(ValueError):
            maybe_dump_kv("bogus")

    def test_finish_stage_without_instance_noop(self):
        def sample_tokens(self, *args, **kwargs):
            return "sampled"

        with patch(f"{KV_DUMP_MODULE}._kv_dump_path", "/tmp/kv_dump"):
            wrapped = maybe_dump_kv("finish")(sample_tokens)
            assert wrapped(MagicMock()) == "sampled"
        assert "dump_prefill" not in maybe_dump_kv.__dict__

    def test_finish_stage_calls_finish(self):
        mock_dump = MagicMock()

        def sample_tokens(self, *args, **kwargs):
            return "sampled"

        with patch(f"{KV_DUMP_MODULE}._kv_dump_path", "/tmp/kv_dump"):
            maybe_dump_kv.__dict__["dump_prefill"] = mock_dump
            wrapped = maybe_dump_kv("finish")(sample_tokens)
            assert wrapped(MagicMock()) == "sampled"
            mock_dump.finish.assert_called_once()

    def test_finish_stage_no_path_bypasses(self):
        def sample_tokens(self, *args, **kwargs):
            return "sampled"

        with patch(f"{KV_DUMP_MODULE}._kv_dump_path", ""):
            maybe_dump_kv.__dict__["dump_prefill"] = MagicMock()
            wrapped = maybe_dump_kv("finish")(sample_tokens)
            assert wrapped(MagicMock()) == "sampled"
            getattr(maybe_dump_kv, "dump_prefill").finish.assert_not_called()


# =========================
# 4. PrefillReq
# =========================

class TestPrefillReq:

    def test_init_copies_blocks(self):
        new = _make_new_request(block_ids=[[1, 2], [3]])
        req = PrefillReq(new)
        new.block_ids[0].append(99)
        assert req.blocks == [[1, 2], [3]]
        assert req.block_set == {1, 2, 3}
        assert req.req_id == "req0"
        assert req.tok_num == 128

    def test_append_blocks_success(self):
        req = _make_prefill_req(block_ids=[[1, 2]])
        req.append_blocks(([3],))
        assert req.recycled is False
        assert req.blocks == [[1, 2, 3]]
        assert req.block_set == {1, 2, 3}

    def test_append_blocks_duplicate_recycles(self):
        # 滑窗/Mome 回收后复用 block id: 旧位置置 null(0), 新块追加末尾, 不再抛错
        req = _make_prefill_req(block_ids=[[1, 2]])
        req.append_blocks(([2],))
        assert req.recycled is True
        assert req.blocks == [[1, 0, 2]]
        assert req.block_set == {0, 1, 2}

    def test_evict_found_and_missing(self):
        req = _make_prefill_req(block_ids=[[1, 2], [3]])
        req._evict(3)
        assert req.blocks == [[1, 2], [0]]
        req._evict(99)  # 不存在: 无副作用
        assert req.blocks == [[1, 2], [0]]

    def test_append_blocks_matches_signed_physical_id(self):
        req = _make_prefill_req(block_ids=[[-2]])
        req.append_blocks(([2],))
        assert req.blocks == [[0, 2]]
        assert req.block_set == {0, 2}

    def test_reuse_evicts_every_old_occurrence(self):
        req = _make_prefill_req(block_ids=[[2, -2], [2, 3]])
        req.append_blocks(([2], []))
        assert req.blocks == [[0, 0, 2], [0, 3]]
        assert req.block_set == {0, 2, 3}

    def test_append_blocks_none(self):
        req = _make_prefill_req(block_ids=[[1, 2]])
        before = [group.copy() for group in req.blocks]
        assert req.append_blocks(None) is req
        assert req.blocks == before
        assert req.block_set == {1, 2}

    def test_dump_block_zero(self):
        req = _make_prefill_req(block_ids=[[0, 1]])
        shapes = _make_shapes(num_blocks=4, block_elems=16)
        group0 = req.dump(shapes)["group0"]
        assert group0[0] == {"block_id": 0}
        assert len(group0[1]) == 2  # block_id + one shape hash layer

    def test_dump_negative_block(self):
        req = _make_prefill_req(block_ids=[[-3]])
        shapes = _make_shapes(num_blocks=4, block_elems=16)
        blk = req.dump(shapes)["group0"][0]
        assert blk["block_id"] == 3
        assert blk["hashed"] is True

    def test_dump_with_hashes(self):
        req = _make_prefill_req(block_ids=[[1]])
        shapes = _make_shapes(num_blocks=4, block_elems=16)
        blk = req.dump(shapes)["group0"][0]
        assert blk["block_id"] == 1
        hash_keys = [k for k in blk if k.endswith(".blk")]
        assert len(hash_keys) == len(shapes)
        for key in hash_keys:
            assert isinstance(blk[key], list)
            assert all(":" in item for item in blk[key])

    def test_dump_remote_req_id(self):
        req = _make_prefill_req(block_ids=[[0]])
        shapes = _make_shapes(num_blocks=4, block_elems=16)
        data = req.dump(shapes, remote_req="prefill_req_1")
        assert data["remote_req_id"] == "prefill_req_1"


# =========================
# 5. DecodeReq
# =========================

class TestDecodeReq:

    def test_update_rest_zero(self):
        req = _make_decode_req(block_ids=[[0, 1, 2]])
        shapes = _make_shapes(num_blocks=4, block_elems=16)
        groups = _make_decode_groups(num_blocks=4)
        assert req.update(128, None, groups, shapes) is req
        assert req.tok_num == 128
        assert req.hashed_blks == []

    def test_update_with_new_blocks(self):
        req = _make_decode_req(block_ids=[[0, 1]])
        shapes = _make_shapes(num_blocks=4, block_elems=16)
        groups = _make_decode_groups(num_blocks=4)
        req.update(128, ([2],), groups, shapes)
        assert req.blocks == [[0, 1, 2]]
        assert req.block_set == {0, 1, 2}
        assert req.hashed_batch is not None

    def test_check_toks_incremental(self):
        req = _make_decode_req(block_ids=[[0, 1, 2]])
        shapes = _make_shapes(num_blocks=4, block_elems=16)
        groups = _make_decode_groups(num_blocks=4, pg=128, block_elems=8)
        req.update(129, None, groups, shapes, pg=128)
        assert "layer0.0" in req.hist_hash
        req.update(130, None, groups, shapes, pg=128)

    def test_check_toks_mismatch_logs_and_refreshes_baseline(self):
        req = _make_decode_req(block_ids=[[0, 1, 2]])
        shapes = _make_shapes(num_blocks=4, block_elems=16)
        groups = _make_decode_groups(num_blocks=4, pg=128, block_elems=8)
        req.update(129, None, groups, shapes, pg=128)
        name = "layer0.0"
        full, rest, sect = req.hist_hash[name]
        # 篡改历史切片，使第二步 update 时 sect[:_rest] 与 hist 不一致
        req.hist_hash[name] = (full, rest, sect + 1)
        # 不再抛错: 报错 + 刷新基线 (n_mismatch 累计, hist_hash 更新为本次值)
        req.update(130, None, groups, shapes, pg=128)
        assert req.n_mismatch == 1
        assert req.hist_hash[name][0] == 130 // 128
        assert req.hist_hash[name][1] == 130 % 128

    def test_check_blks_mismatch_reports_and_updates(self):
        req = _make_decode_req(block_ids=[[1, 2]])
        shapes = _make_shapes(num_blocks=4, block_elems=16)
        groups = _make_decode_groups(num_blocks=4)
        req.update(128, None, groups, shapes)
        assert req.n_mismatch == 0
        assert req.hashed_blks == [1]
        # 篡改上次的 block 哈希基线, 使下一步与当前块不一致
        req.hashed_batch[0][0] = req.hashed_batch[0][0] + 1
        req.update(129, None, groups, shapes)
        assert req.n_mismatch == 1

    def test_update_recycled_resets_baseline(self):
        req = _make_decode_req(block_ids=[[0, 1]])
        shapes = _make_shapes(num_blocks=4, block_elems=16)
        groups = _make_decode_groups(num_blocks=4)
        req.update(128, None, groups, shapes)
        assert req.hashed_blks == []
        # 复用已在表中的 block id(1) => recycled, 旧位置置 0 后新块追加
        req.update(129, ([1],), groups, shapes)
        assert req.recycled is True
        assert req.blocks == [[0, 0, 1]]
        assert req.block_set == {0, 1}

    def test_finalize_clears_batch(self):
        req = _make_decode_req(block_ids=[[0, 1, 2]])
        shapes = _make_shapes(num_blocks=4, block_elems=16)
        groups = _make_decode_groups(num_blocks=4)
        req.update(128, None, groups, shapes)
        assert req.hashed_batch is not None
        req.finalize()
        assert req.hashed_batch is None

    def test_recycle_preserves_detection_on_other_blocks(self):
        req = _make_decode_req(block_ids=[[1, 2, 3]])
        shapes = _make_shapes(num_blocks=4)
        groups = _make_decode_groups(num_blocks=4)
        req.update(256, None, groups, shapes)
        shapes[0][0][1].add_(1)  # still-owned block is corrupted
        shapes[0][0][2].add_(1)  # recycled block may legitimately change

        req.update(256, ([2],), groups, shapes)

        assert req.blocks == [[1, 0, 3, 2]]
        assert req.hashed_blks == [1]
        assert req.n_mismatch == 1

    def test_null_blocks_are_not_compared(self):
        req = _make_decode_req(block_ids=[[0, 0]])
        shapes = _make_shapes()
        groups = _make_decode_groups()
        req.update(129, None, groups, shapes)
        shapes[0][0][0].add_(1)
        groups[0]["layer0"][0][0].add_(1)
        req.update(130, None, groups, shapes)
        assert req.hashed_blks == []
        assert not req.hist_hash
        assert req.n_mismatch == 0

    def test_signed_tables_use_physical_slots(self):
        req = _make_decode_req(block_ids=[[-1, -2]])
        shapes = _make_shapes()
        groups = _make_decode_groups()
        req.update(129, None, groups, shapes)
        assert req.hashed_blks == [1]
        assert req.hist_blocks == {"layer0.0": 2}
        # Negative indexing would incorrectly compare block 3 for -1.
        shapes[0][0][3].add_(1)
        groups[0]["layer0"][0][2, 0].add_(1)
        req.update(130, None, groups, shapes)
        assert req.hashed_blks == [1]
        assert req.n_mismatch == 1
        assert req.hist_blocks["layer0.0"] == 2

    def test_invalidate_keeps_other_group_token_history(self):
        req = _make_decode_req(block_ids=[[1, 2], [3, 4]])
        shapes = _make_shapes(num_blocks=5)
        cache0 = _make_decode_groups(num_blocks=5)[0]["layer0"]
        cache1 = _make_decode_groups(num_blocks=5)[0]["layer0"]
        groups = [{"layer0": cache0}, {"layer1": cache1}]
        req.update(129, None, groups, shapes)

        req.invalidate_blocks([2])
        assert "layer0.0" not in req.hist_hash
        assert "layer1.0" in req.hist_hash
        cache1[0][4, 0].add_(1)
        req.update(130, None, groups, shapes)

        assert req.n_mismatch == 1
        assert req.hist_blocks["layer1.0"] == 4


# =========================
# 6. Dump
# =========================

class TestDump:

    def test_init_path_format(self):
        dump = _make_dump("/data/kv_dump")
        assert dump.path == "/data/kv_dump/1704067200/"
        assert dump.reqs == {}
        assert dump.rank is None

    def test_register_once(self):
        dump = _make_dump("/tmp/kv_dump")
        runner = _make_model_runner()
        dump._register(runner)
        shapes_first = dump.shapes
        groups_first = dump.groups
        dump._register(runner)
        assert dump.shapes is shapes_first
        assert dump.groups is groups_first
        assert dump.shapes[0][0].device.type == "cpu"

    def test_prefill_lifecycle(self):
        with _mock_world_group(rank=2):
            dump = _make_dump("/tmp/kv_dump")
            runner = _make_model_runner(kv_role="kv_producer")
            new = _make_new_request(req_id="p0", block_ids=[[0, 1]])

            with dump.update(runner, _make_scheduler_output_prefill(new_reqs=[new])):
                pass
            assert not hasattr(dump.reqs["p0"], "data")
            dump.finish()
            assert "p0" in dump.reqs
            assert "group0" in dump.reqs["p0"].data
            assert dump.rank == 2

            with dump.update(runner, _make_scheduler_output_prefill()):
                pass

            assert not dump.change.empty()
            req_id, data = dump.change.get()
            assert req_id == "p0"
            assert isinstance(data, dict)
            assert "group0" in data
            assert "p0" not in dump.reqs

    def test_decode_new_req_immediate_dump(self):
        with _mock_world_group():
            dump = _make_dump("/tmp/kv_dump")
            runner = _make_model_runner(kv_role="kv_consumer")
            new = _make_new_request(req_id="d0", block_ids=[[0, 1]])
            sched = _make_scheduler_output_decode(
                new_reqs=[new],
                req_params={"d0": {"prefill_req_id": "p0"}},
            )
            with dump.update(runner, sched):
                pass
            assert not dump.change.empty()
            req_id, data = dump.change.get()
            assert req_id == "d0"
            assert data["remote_req_id"] == "p0"
            assert "group0" in data

    def test_decode_cached_req_update(self):
        with _mock_world_group():
            dump = _make_dump("/tmp/kv_dump")
            runner = _make_model_runner(kv_role="kv_consumer")
            req = _make_decode_req(req_id="d0", block_ids=[[0, 1]])
            dump.reqs = {"d0": req}
            dump.shapes = _make_shapes(num_blocks=4, block_elems=16)
            dump.groups = _make_decode_groups(num_blocks=4)

            sched = _make_scheduler_output_decode(
                req_ids=["d0"],
                computed=[128],
                new_block_ids=[None],
            )
            with dump.update(runner, sched):
                pass

            assert req.tok_num == 128
            assert dump.reqs["d0"] is req

    def test_decode_new_request_reuse_invalidates_old_baseline(self):
        with _mock_world_group():
            dump = _make_dump("/tmp/kv_dump")
            dump.shapes = _make_shapes(num_blocks=4, block_elems=16)
            dump.groups = _make_decode_groups(num_blocks=4)

            old = _make_decode_req(req_id="old", block_ids=[[1, 2, 3]])
            old.update(256, None, dump.groups, dump.shapes)
            # Make the reused physical slot differ after the old baseline was
            # captured. Without lifecycle invalidation this is a block alarm.
            dump.shapes[0][0][2].add_(1)
            dump.reqs = {"old": old}

            new = _make_new_request(
                req_id="new", tokens=0, block_ids=[[2]]
            )
            dump._decode(
                ["old"], [256], [None], [new], 0, {"new": "prefill_new"},
            )

            assert old.n_mismatch == 0
            assert old.blocks == [[1, 0, 3]]
            assert 2 not in old.hashed_blks
            assert old.hashed_batch is not None
            assert "new" in dump.reqs

    def test_decode_same_request_overwrite_still_logs(self):
        with _mock_world_group():
            dump = _make_dump("/tmp/kv_dump")
            dump.shapes = _make_shapes(num_blocks=4, block_elems=16)
            dump.groups = _make_decode_groups(num_blocks=4)
            req = _make_decode_req(req_id="old", block_ids=[[1, 2, 3]])
            req.update(256, None, dump.groups, dump.shapes)
            dump.reqs = {"old": req}

            dump.shapes[0][0][1].add_(1)
            dump._decode(["old"], [256], [None], [], 0, {})

            assert req.n_mismatch == 1
            assert 1 in req.hashed_blks

    def test_async_pd_allocation_invalidates_before_new_req_arrives(self):
        with _mock_world_group():
            dump = _make_dump()
            runner = _make_model_runner(kv_role="kv_consumer")
            dump.shapes = _make_shapes(num_blocks=5)
            dump.groups = _make_decode_groups(num_blocks=5)
            old = _make_decode_req(req_id="old", block_ids=[[1, 2, 3]])
            old.update(256, None, dump.groups, dump.shapes)
            dump.reqs = {"old": old}
            dump.shapes[0][0][1].add_(1)  # real corruption must still report
            dump.shapes[0][0][2].add_(1)  # new PD destination

            sched = _make_scheduler_output_decode(
                req_ids=["old"], computed=[256],
                req_params={"new": {
                    "prefill_req_id": "prefill_new",
                    "decode_blocks": [[2]],
                }},
            )
            with dump.update(runner, sched):
                pass
            assert old.blocks == [[1, 0, 3]]
            assert old.n_mismatch == 1
            assert 1 in old.hashed_blks
            assert 2 not in old.hashed_blks

            # The loaded tokens are computed tokens when the new request is
            # finally scheduled. This must not revive the retired baseline.
            new = _make_new_request(
                req_id="new", block_ids=[[2, 4]], computed=127,
            )
            with dump.update(runner, _make_scheduler_output_decode(
                req_ids=["old"], computed=[256], new_reqs=[new],
            )):
                pass
            assert dump.reqs["new"].blocks == [[2, 4]]
            assert dump.reqs["new"].n_mismatch == 0
            assert old.n_mismatch == 1

    def test_prefix_hit_marker_preserves_block_one_check(self):
        with _mock_world_group():
            dump = _make_dump()
            runner = _make_model_runner(kv_role="kv_consumer")
            dump.shapes = _make_shapes(num_blocks=5)
            dump.groups = _make_decode_groups(num_blocks=5)
            old = _make_decode_req(req_id="old", block_ids=[[1, 2, 3]])
            old.update(256, None, dump.groups, dump.shapes)
            dump.reqs = {"old": old}
            dump.shapes[0][0][1].add_(1)

            with dump.update(runner, _make_scheduler_output_decode(
                req_ids=["old"], computed=[256],
                new_reqs=[_make_new_request(
                    req_id="new", block_ids=[[1, 4]], computed=128,
                )],
                req_params={"new": {
                    "prefill_req_id": "prefill_new",
                    "decode_blocks": [[-1, 4]],
                }},
            )):
                pass

            assert old.blocks == [[1, 2, 3]]
            assert old.n_mismatch == 1
            assert 1 in old.hashed_blks

    @pytest.mark.parametrize("corrupt_owned_block", [False, True])
    def test_async_pd_first_schedule_retires_lookahead_blocks(
        self, corrupt_owned_block,
    ):
        dump = _make_dump()
        runner = _make_model_runner(kv_role="kv_consumer", spec_tokens=7)
        runner.kv_cache_config.kv_cache_groups *= 6
        dump.shapes = _make_shapes(num_blocks=104)
        dump.groups = _make_decode_groups(num_blocks=104) * 6
        old = _make_decode_req(req_id="old", block_ids=[
            [1, 2, 82, 3], [4, 5, 45, 6], [7, 8, 51, 9],
            [10, 11, 12, 13], [14, 15, 16, 17], [18, 19, 20, 21],
        ])
        old.update(433, None, dump.groups, dump.shapes, mtp=7)
        dump.reqs = {"old": old}
        # PD metadata precedes the first schedule and only contains the
        # imported prompt blocks. It cannot retire the later lookahead slots.
        with _mock_world_group(), dump.update(runner, _make_scheduler_output_decode(
            req_ids=["old"], computed=[433],
            req_params={"new": {
                "prefill_req_id": "prefill_new",
                "decode_blocks": [[90], [89], [88], [57], [56], [76]],
            }},
        )):
            pass

        new = _make_new_request(
            req_id="new", tokens=127, computed=126,
            block_ids=[
                [90, 82], [89, 45], [88, 51],
                [57, 100], [56, 101], [76, 102],
            ],
        )
        with _mock_world_group(), dump.update(runner, _make_scheduler_output_decode(
            req_ids=["old"], computed=[433], new_reqs=[new],
        )):
            pass
        assert old.blocks[:3] == [[1, 2, 0, 3], [4, 5, 0, 6], [7, 8, 0, 9]]

        # The new request's legitimate forward changes the reassigned slots.
        for block_id in (51, 45, 82):
            dump.shapes[0][0][block_id].add_(1)
        if corrupt_owned_block:
            dump.shapes[0][0][1].add_(1)
        with _mock_world_group(), dump.update(runner, _make_scheduler_output_decode(
            req_ids=["old", "new"], computed=[441, 134],
        )):
            pass
        for block_id in (51, 45, 82):
            assert block_id not in old.hashed_blks
        assert old.n_mismatch == (1 if corrupt_owned_block else 0)
        if corrupt_owned_block:
            assert 1 in old.hashed_blks

    @pytest.mark.parametrize("computed, block_size, cp_world_size", [
        (128, 128, 1), (63, 64, 1), (255, 128, 2),
    ])
    def test_first_schedule_keeps_computed_prefix_and_retires_suffix(
        self, computed, block_size, cp_world_size,
    ):
        dump = _make_dump()
        runner = _make_model_runner(kv_role="kv_consumer")
        runner.kv_cache_config.kv_cache_groups[0].kv_cache_spec.block_size = block_size
        dump.shapes = _make_shapes(num_blocks=5)
        dump.groups = _make_decode_groups(num_blocks=5)
        old = _make_decode_req(req_id="old", block_ids=[[1, 2, 3]])
        old.update(256, None, dump.groups, dump.shapes)
        dump.reqs = {"old": old}
        dump.shapes[0][0][1].add_(1)
        dump.shapes[0][0][2].add_(1)

        with (
            _mock_world_group(),
            patch(f"{KV_DUMP_MODULE}.get_total_cp_world_size", return_value=cp_world_size),
            dump.update(runner, _make_scheduler_output_decode(
                req_ids=["old"], computed=[256],
                new_reqs=[_make_new_request(
                    req_id="new", tokens=computed + 1, computed=computed,
                    block_ids=[[1, 2]],
                )],
                req_params={"new": {"prefill_req_id": "prefill_new"}},
            )),
        ):
            pass
        assert old.blocks == [[1, 0, 3]]
        assert old.n_mismatch == 1
        assert 1 in old.hashed_blks
        assert 2 not in old.hashed_blks

    def test_first_schedule_uses_each_groups_logical_block_size(self):
        dump = _make_dump()
        runner = _make_model_runner(kv_role="kv_consumer")
        runner.kv_cache_config.kv_cache_groups.append(SimpleNamespace(
            kv_cache_spec=SimpleNamespace(block_size=64),
        ))
        dump.shapes = _make_shapes(num_blocks=8)
        dump.groups = _make_decode_groups(num_blocks=8) * 2
        old = _make_decode_req(req_id="old", block_ids=[[1, 2, 6], [3, 4, 5, 7]])
        old.update(384, None, dump.groups, dump.shapes)
        dump.reqs = {"old": old}
        for block_id in (1, 2, 3, 4, 5):
            dump.shapes[0][0][block_id].add_(1)

        with _mock_world_group(), dump.update(runner, _make_scheduler_output_decode(
            req_ids=["old"], computed=[384],
            new_reqs=[_make_new_request(
                req_id="new", tokens=129, computed=128,
                block_ids=[[1, 2], [3, 4, 5]],
            )],
            req_params={"new": {"prefill_req_id": "prefill_new"}},
        )):
            pass
        assert old.blocks == [[1, 0, 6], [3, 4, 0, 7]]
        assert old.n_mismatch == 1
        for block_id in (1, 3, 4):
            assert block_id in old.hashed_blks
        for block_id in (2, 5):
            assert block_id not in old.hashed_blks

    def test_cached_allocation_evicts_other_request_stale_block(self):
        dump = _make_dump()
        dump.shapes = _make_shapes(num_blocks=5)
        dump.groups = _make_decode_groups(num_blocks=5)
        old = _make_decode_req(req_id="old", block_ids=[[1, 2, 3]])
        other = _make_decode_req(req_id="other", block_ids=[[4]])
        old.update(256, None, dump.groups, dump.shapes)
        dump.reqs = {"old": old, "other": other}
        dump.shapes[0][0][2].add_(1)

        dump._decode(
            ["old", "other"], [256, 128], [None, ([2],)], [], 0, {},
        )

        assert old.blocks == [[1, 0, 3]]
        assert other.blocks == [[4, 2]]
        assert old.n_mismatch == 0
        assert 2 not in old.hashed_blks

    def test_resumed_full_table_preserves_shared_prefix_baseline(self):
        dump = _make_dump()
        dump.shapes = _make_shapes(num_blocks=5)
        dump.groups = _make_decode_groups(num_blocks=5)
        old = _make_decode_req(req_id="old", block_ids=[[1, 2, 3]])
        resumed = _make_decode_req(req_id="resumed", block_ids=[[1, 4]])
        old.update(256, None, dump.groups, dump.shapes)
        dump.reqs = {"old": old, "resumed": resumed}
        dump.shapes[0][0][1].add_(1)

        dump._decode(
            ["old", "resumed"], [256, 128], [None, ([1, 4],)], [], 0, {},
            resumed_req_ids={"resumed"},
        )

        # A resumed full table is not a fresh allocation of shared block 1.
        assert old.blocks == [[1, 2, 3]]
        assert old.n_mismatch == 1
        assert 1 in old.hashed_blks

    def test_unscheduled_decode_keeps_request_for_later_step(self):
        dump = _make_dump()
        dump.shapes = _make_shapes(num_blocks=4)
        dump.groups = _make_decode_groups(num_blocks=4)
        paused = _make_decode_req(req_id="paused", block_ids=[[1, 2, 3]])
        live = _make_decode_req(req_id="live", block_ids=[[4]])
        paused.update(256, None, dump.groups, dump.shapes)
        live.update(128, None, dump.groups, dump.shapes)
        dump.reqs = {"paused": paused, "live": live}

        # A running request can miss a step without being finished.
        dump._decode(["live"], [128], [None], [], 0, {})
        assert "paused" in dump.reqs

        dump.shapes[0][0][1].add_(1)
        dump._decode(["paused"], [256], [None], [], 0, {})
        assert "paused" in dump.reqs
        assert paused.n_mismatch == 1
        assert 1 in paused.hashed_blks

    def test_finished_decode_req_is_removed_without_dropping_others(self):
        dump = _make_dump()
        dump.shapes = _make_shapes(num_blocks=4)
        dump.groups = _make_decode_groups(num_blocks=4)
        done = _make_decode_req(req_id="done", block_ids=[[1]])
        live = _make_decode_req(req_id="live", block_ids=[[2]])
        done.update(128, None, dump.groups, dump.shapes)
        live.update(128, None, dump.groups, dump.shapes)
        dump.reqs = {"done": done, "live": live}
        dump.remote_reqs = {"done": "prefill-done", "live": "prefill-live"}

        dump._decode(
            ["live"], [128], [None], [], 0, {},
            finished_req_ids={"done"},
        )

        assert "done" not in dump.reqs
        assert "done" not in dump.remote_reqs
        assert "live" in dump.reqs
        assert dump.remote_reqs["live"] == "prefill-live"

    def test_unknown_cached_req_does_not_raise(self):
        dump = _make_dump()
        dump.shapes = _make_shapes(num_blocks=4)
        dump.groups = _make_decode_groups(num_blocks=4)

        dump._decode(
            ["chatcmpl-missing"], [26], [None], [], 0, {},
        )

        assert "chatcmpl-missing" not in dump.reqs
        assert dump.missing_cached == {"chatcmpl-missing"}

    def test_resumed_request_replaces_block_table(self):
        dump = _make_dump()
        dump.shapes = _make_shapes(num_blocks=8)
        dump.groups = _make_decode_groups(num_blocks=8)
        resumed = _make_decode_req(req_id="resumed", block_ids=[[1, 4]])
        resumed.update(128, None, dump.groups, dump.shapes)
        dump.reqs = {"resumed": resumed}
        dump.shapes[0][0][1].add_(1)

        dump._decode(
            ["resumed"], [128], [([5, 6],)], [], 0, {},
            resumed_req_ids={"resumed"},
        )

        assert resumed.blocks == [[5, 6]]
        assert resumed.block_set == {5, 6}
        assert resumed.n_mismatch == 0
        assert resumed.hashed_blks == [5]

    def test_resumed_request_missing_state_is_rebuilt(self):
        dump = _make_dump()
        dump.shapes = _make_shapes(num_blocks=8)
        dump.groups = _make_decode_groups(num_blocks=8)

        dump._decode(
            ["back"], [128], [([5, 6],)], [], 0, {},
            resumed_req_ids={"back"},
        )

        assert dump.reqs["back"].blocks == [[5, 6]]

    def test_scheduler_zeroing_signal_invalidates_reused_block(self):
        with _mock_world_group():
            dump = _make_dump()
            runner = _make_model_runner(kv_role="kv_consumer")
            dump.shapes = _make_shapes()
            dump.groups = _make_decode_groups()
            old = _make_decode_req(req_id="old", block_ids=[[1, 2, 3]])
            old.update(256, None, dump.groups, dump.shapes)
            dump.reqs = {"old": old}
            dump.shapes[0][0][2].add_(1)

            with dump.update(runner, _make_scheduler_output_decode(
                req_ids=["old"], computed=[256], allocated_blocks=[2],
            )):
                pass

            assert old.blocks == [[1, 0, 3]]
            assert old.n_mismatch == 0
            assert 2 not in old.hashed_blks

    def test_task_save_json_writes_file(self):
        dump = _make_dump("kv_dump")
        dump.rank = 3
        dump.path = "kv_dump/1704067200/"
        dump.change.put(("req0", {"group0": [{"block_id": 1}]}))

        calls = 0

        def sleep_side_effect(_sec):
            nonlocal calls
            calls += 1
            if calls >= 2:
                raise StopIteration

        with (
            patch(f"{KV_DUMP_MODULE}.append_items") as mock_append,
            patch(f"{KV_DUMP_MODULE}.os.makedirs"),
            patch(f"{KV_DUMP_MODULE}.time.sleep", side_effect=sleep_side_effect),
        ):
            with pytest.raises(StopIteration):
                dump._task_save_json()

        mock_append.assert_called_once()
        path, items = mock_append.call_args[0]
        assert path == dump.path + "rank3.json"
        assert "req0" in items

    def test_finish_dumps_prefill_and_clears_pending(self):
        with _mock_world_group():
            dump = _make_dump("/tmp/kv_dump")
            runner = _make_model_runner(kv_role="kv_producer")
            new = _make_new_request(req_id="p0", block_ids=[[0, 1]])
            with dump.update(runner, _make_scheduler_output_prefill(new_reqs=[new])):
                pass
            assert dump.pending is True
            dump.finish()
            assert dump.pending is False
            assert "group0" in dump.reqs["p0"].data
            # 二次 finish 幂等, 不再重复 dump
            dump.finish()
            assert dump.pending is False

    def test_finish_noop_when_not_pending(self):
        dump = _make_dump("/tmp/kv_dump")
        dump.finish()  # 未 register / 非 prefill: 直接返回
        assert dump.pending is False

    def test_missing_cached_req_is_logged_once(self):
        dump = _make_dump()
        dump.shapes = _make_shapes(num_blocks=4)
        dump.groups = _make_decode_groups(num_blocks=4)
        with patch(f"{KV_DUMP_MODULE}.logger.error") as err:
            dump._decode(["gone"], [26], [None], [], 0, {})
            dump._decode(["gone"], [27], [None], [], 0, {})
        assert err.call_count == 1
        assert dump.missing_cached == {"gone"}

    def test_replace_blocks_without_table_keeps_ids(self):
        req = _make_decode_req(req_id="r", block_ids=[[1, 4]])
        req.hashed_blks = [1]
        req.replace_blocks(None)
        assert req.blocks == [[1, 4]]
        assert req.hashed_blks == []


def test_id_set_ignores_non_containers():
    assert _id_set(MagicMock()) == set()
    assert _id_set(["a", 2]) == {"a", "2"}
    assert _id_set({"b"}) == {"b"}

