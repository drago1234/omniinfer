# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""Run the actual Triton kernel arithmetic through its CPU interpreter.

This checks indexing, masks and history against independent CPU references;
NPU compilation, mapped-memory access and collectives remain integration tests.
"""

import importlib.util
import os
from pathlib import Path
import sys
from types import FunctionType, ModuleType, SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
import torch

from omni_npu.layers import engram
from tests.integration.layers.engram_test_utils import run_isolated_pytest
from tests.integration.layers.test_engram_device import Validator, batch
from tests.unit.layers.test_engram import _config

pytestmark = pytest.mark.unit


@pytest.fixture
def cpu_kernels(monkeypatch):
    # CPU interpretation only needs core tl operations. Match the block-kernel
    # tests by bypassing eager CANN discovery when the native Ascend module is
    # unavailable; keep explicit backend imports possible through the real path.
    with monkeypatch.context() as imports:
        triton_root = Path(importlib.util.find_spec("triton").origin).parent
        extra = ModuleType("triton.language.extra")
        extra.__path__ = [str(triton_root / "language/extra")]
        imports.setitem(sys.modules, extra.__name__, extra)
        from triton.runtime.interpreter import InterpretedFunction
        import triton
        import triton.language as tl

        monkeypatch.setattr(engram, "triton", triton)
        monkeypatch.setattr(engram, "tl", tl)
        # Ascend's interpreter patches module globals. Give it only kernel
        # globals so it cannot replace torch.ops or other host-side APIs.
        language_objects = [tl, tl.core, tl.math, tl.tensor, tl.dtype]
        snapshots = [(obj, dict(vars(obj))) for obj in language_objects]
        try:
            for name in ("_engram_lookup_kernel", "_write_hash_cache_kernel", "_hash_ids_kernel"):
                kernel = getattr(engram, name)
                # vLLM's Triton placeholder leaves functions undecorated.
                original = getattr(kernel, "fn", kernel)
                function = FunctionType(original.__code__, {"tl": tl, "triton": triton}, original.__name__)
                function.__annotations__ = original.__annotations__
                monkeypatch.setattr(engram, name, InterpretedFunction(function))
            monkeypatch.setattr(torch.npu, "current_stream", lambda: None)
            monkeypatch.setattr(torch.npu, "get_stream_limit", lambda stream: {"vector_core_num": 16})
            monkeypatch.setattr(torch.npu, "current_device", lambda: 0)
            monkeypatch.setattr(torch.npu, "get_device_properties", lambda device: SimpleNamespace(vector_core_num=16))
            monkeypatch.setattr(torch.distributed, "get_rank", lambda: 0)

            def allocate(self, shape, dtype):
                return torch.zeros(shape, dtype=dtype)

            def gather(errors, error, group):
                errors[:] = [error] + [None] * (len(errors) - 1)

            # Keep the real loader and its validation; only replace shared-memory
            # allocation, device registration and collectives for CPU execution.
            monkeypatch.setattr(engram.DPSharedEngramStorage, "_allocate", allocate)
            monkeypatch.setattr(engram.DPSharedEngramStorage, "device_view", lambda self: self.tensor)
            monkeypatch.setattr(torch.distributed, "all_gather_object", gather)
            yield
        finally:
            for obj, snapshot in reversed(snapshots):
                for key in set(vars(obj)) - snapshot.keys():
                    delattr(obj, key)
                for key, value in snapshot.items():
                    if vars(obj).get(key) is not value:
                        setattr(obj, key, value)


class TestEngramCpuKernels:
    __test__ = os.environ.get("OMNI_ENGRAM_TEST_CHILD") == "1"

    @staticmethod
    def test_hash_kernels_match_independent_reference_with_strides_and_slot_history(cpu_kernels):
        validator = Validator(torch.device("cpu"))
        validator.random_cases()
        validator.cache_cases()
        assert len(validator.passed) >= 20
        for rank in range(4):
            validator.cache_cases(sp_rank=rank, sp_size=4)
        with pytest.raises(ValueError, match="slot_mapping and block_table"):
            validator.run(validator.upload(batch(42, 3, 2)), cache=torch.zeros(32, dtype=torch.int32))

    @staticmethod
    @pytest.mark.parametrize("requests", [1, 3])
    def test_context_lengths_override_positions_with_arbitrary_coordinates(cpu_kernels, requests):
        """num_computed_tokens supplies logical context for 2D mRoPE coordinates.

        The kernel derives each token's logical position from the request's
        context length, so 2D positions may hold arbitrary per-axis coordinates.
        The reference sees the equivalent one-dimensional logical positions.
        1D positions already carry logical order, so context lengths are ignored.
        """
        validator = Validator(torch.device("cpu"))
        validator.set_config(validator.config)
        for tokens in (1, 7, 257):
            inputs = batch(42 + tokens, tokens, 2, num_reqs=requests)
            ids, positions, starts, dead, lookbacks, lookback_dead = inputs
            starts_np = starts.numpy()
            rng = np.random.default_rng(1000 + tokens)
            # Two requests may share an entry: chunked prefill keeps the context
            # of earlier chunks; fresh requests restart from zero.
            context = rng.integers(0, 9, requests)
            request_of = np.searchsorted(starts_np[1:], np.arange(tokens), side="right")
            offset_within = np.arange(tokens) - np.repeat(starts_np[:-1], np.diff(starts_np))
            logical = context[request_of] + offset_within
            # 2D mRoPE coordinates (t/h/w axes) that do not encode logical positions.
            coordinates = torch.from_numpy(rng.integers(0, 5, (tokens, 3)))
            context_t = torch.from_numpy(context).to(torch.int64)
            out = validator.state(
                ids, coordinates, starts, dead, lookbacks, lookback_dead,
                slot_mapping=None, block_table=None,
                num_computed_tokens=context_t,
            )
            reference_inputs = (ids, torch.from_numpy(logical), starts, dead, lookbacks, lookback_dead)
            expected = validator.expected(reference_inputs)
            torch.testing.assert_close(out, expected, rtol=0, atol=0)
            # 1D positions are already logical: mismatching lengths must not apply.
            ignored = validator.state(
                ids, positions, starts, dead, lookbacks, lookback_dead,
                slot_mapping=None, block_table=None,
                num_computed_tokens=context_t + 1000,
            )
            torch.testing.assert_close(ignored, validator.expected(inputs), rtol=0, atol=0)

    @staticmethod
    @pytest.mark.parametrize("requests", [1, 3])
    def test_sp_hash_windows_match_reference(cpu_kernels, requests):
        validator = Validator(torch.device("cpu"))
        for tokens in (1, 7, 257):
            inputs = batch(42 + tokens, tokens, 2, num_reqs=requests)
            out = torch.empty_like(validator.expected(inputs))
            for rank in range(4):
                validator.state.sp_rank = rank
                validator.state.sp_size = 4
                out.fill_(-123)
                validator.check(f"sp_requests{requests}_tokens{tokens}_rank{rank}", inputs, out=out)

    @staticmethod
    @pytest.mark.parametrize("model_type", ["low_latency", "high_throughout"])
    def test_shared_model_preparation_compiles_and_reads_new_inputs(cpu_kernels, monkeypatch, model_type):
        from omni_npu.v1.models.pangu import pangu_ultra_moe, pangu_v2_moe

        group = SimpleNamespace(world_size=1, rank_in_group=0, ranks=[0], cpu_group=None)
        monkeypatch.setattr(engram, "get_tp_group", lambda: group)

        def linear(size, width, **kwargs):
            module = torch.nn.Linear(size, width, bias=False)
            module.tp_size = 1
            return module

        monkeypatch.setattr(engram, "ColumnParallelLinear", linear)
        layout = engram.EngramLayout(_config(shard_heads=False))
        fusion = engram.PanGuEngramFusion(layout, 2, group, False)
        table = torch.arange(fusion.embed_tokens.weight.numel(), dtype=torch.float32).reshape_as(
            fusion.embed_tokens.weight
        )
        fusion.embed_tokens.weight.data.copy_(table)
        model_cls = pangu_v2_moe.OpenPanguV2Model if model_type == "low_latency" else pangu_ultra_moe.OpenPanguModel
        model = model_cls.__new__(model_cls)
        torch.nn.Module.__init__(model)
        model.engram_layout = layout
        model.engram_hash = SimpleNamespace(has_bound_cache=lambda: True)
        model._get_engram_metadata = lambda: True
        model._compute_engram_hashes = lambda ids, positions, lookbacks: (
            layout.offsets[None] + ids[:, None, None] % 3
        ).int()
        decoder = torch.nn.Module()
        decoder.engram = fusion
        model.layers = torch.nn.ModuleList([decoder])
        model.start_layer, model.end_layer = 0, 1
        model._prepare_key = id(model)
        engram._engram_model_owners[model._prepare_key] = model

        def prepare(ids):
            hashes = model.prepare_model_embeddings(ids, ids, None)
            return hashes, fusion.staged_rows

        torch._dynamo.reset()
        try:
            compiled = torch.compile(prepare, backend="eager", fullgraph=True)
            for ids in (torch.tensor([0, 1, 2]), torch.tensor([2, 0, 1])):
                hashes, rows = compiled(ids)
                expected_hashes = (layout.offsets[None] + ids[:, None, None] % 3).int()
                torch.testing.assert_close(hashes, expected_hashes)
                torch.testing.assert_close(rows, table[expected_hashes[:, 0].long()])
        finally:
            torch._dynamo.reset()

    @staticmethod
    @pytest.mark.parametrize("shard_heads", [False, True])
    def test_embedding_load_lookup_gather_and_device_moves(cpu_kernels, monkeypatch, shard_heads):
        layout = engram.EngramLayout(_config(
            num_heads_per_ngram=7, embedding_dim_per_ngram=14, shard_heads=shard_heads
        ))
        group = SimpleNamespace(world_size=4, rank_in_group=3, ranks=[0, 1, 2, 3], cpu_group=None)
        monkeypatch.setattr(engram, "get_tp_group", lambda: group)
        embedding = engram.ParallelEngramEmbedding(layout, 2, group)
        full = torch.arange(embedding.num_embeddings * 2, dtype=torch.float32).reshape(-1, 2)
        embedding.weight.copy_(full[embedding.vocab_start_idx:embedding.vocab_end_idx])
        pointer = embedding.weight.data_ptr()
        assert embedding.to(dtype=torch.float16) is embedding
        assert embedding.weight.dtype == torch.float32 and embedding.weight.data_ptr() == pointer
        ids = layout.offsets[0].int()[None].repeat(3, 1) + torch.arange(3)[:, None]
        ids[-1, 0] = -1
        expected = full[ids.long().clamp_min(0)].masked_fill(ids[:, :, None] < 0, 0)
        gather = Mock(return_value=expected)
        monkeypatch.setattr(engram, "tensor_model_parallel_all_gather", gather)
        torch.testing.assert_close(embedding(ids), expected, rtol=0, atol=0)
        if shard_heads:
            local = torch.nn.functional.pad(expected[:, 12:], (0, 0, 0, 2))
            torch.testing.assert_close(gather.call_args.args[0], local, rtol=0, atol=0)
            assert gather.call_args.kwargs == {"dim": 1}
        else:
            gather.assert_not_called()
        fusion = engram.PanGuEngramFusion.__new__(engram.PanGuEngramFusion)
        torch.nn.Module.__init__(fusion)
        fusion.embed_tokens, fusion.layout, fusion.use_sequence_parallel = embedding, layout, False
        fusion.staged_rows = local if shard_heads else expected
        torch.testing.assert_close(fusion.embed(ids), expected, rtol=0, atol=0)
        if shard_heads:
            assert gather.call_args.kwargs == {"dim": 1}
        if not shard_heads:
            assert embedding(ids[:0]).shape == (0, layout.n_hash_cols, 2)
        embedding.weight = torch.nn.Parameter(embedding.weight.clone(), requires_grad=False)
        embedding.weight.engram_shared_ptr = pointer
        with pytest.raises(RuntimeError, match="must not be replaced"):
            embedding.weight_loader(embedding.weight, full)
        monkeypatch.setattr(engram, "get_tp_group", lambda: SimpleNamespace(world_size=16, rank_in_group=0))
        with pytest.raises(ValueError, match="without hash heads"):
            engram.ParallelEngramEmbedding(engram.EngramLayout(_config(shard_heads=True)), 2, group)

    @staticmethod
    @pytest.mark.parametrize("sequence_parallel", [False, True])
    def test_staging_projection_mask_and_layout(cpu_kernels, monkeypatch, sequence_parallel):
        from omni_npu.model_config.config_loader.loader import model_extra_config

        group = SimpleNamespace(world_size=4 if sequence_parallel else 1, rank_in_group=1 if sequence_parallel else 0,
                                ranks=[0, 1, 2, 3], cpu_group=None)
        monkeypatch.setattr(engram, "get_tp_group", lambda: group)

        def linear(size, width, **kwargs):
            module = torch.nn.Linear(size, width, bias=False)
            module.tp_size = 1
            return module

        monkeypatch.setattr(engram, "ColumnParallelLinear", linear)
        monkeypatch.setattr(model_extra_config.operator_opt_config, "use_engram_proj_gate_fusion_op", False)
        norm = Mock(side_effect=lambda x, weight, epsilon: (
            x * torch.rsqrt(x.square().mean(-1, keepdim=True) + epsilon) * weight, None
        ))
        monkeypatch.setattr(engram.torch_npu, "npu_rms_norm", norm)
        layer = engram.PanGuEngramFusion(engram.EngramLayout(_config(shard_heads=False)), 2, group, sequence_parallel)
        ids = layer.layout.offsets[0].int()[None].repeat(5, 1)
        ids[2] = -1
        with pytest.raises(RuntimeError, match="not initialized"):
            layer.prepare_embeddings(ids)
        layer.staged_rows = torch.empty(0, 4, 2)
        with pytest.raises(ValueError, match="too small"):
            layer.prepare_embeddings(ids)
        layer.staged_rows = torch.ops.omni_npu.engram_lookup_rows(ids, layer._lookup_key)
        assert engram._engram_lookup_rows_fake(ids, layer._lookup_key).shape == layer.staged_rows.shape
        layer.embed_tokens.weight.data.fill_(0.25)
        layer.prepare_embeddings(ids)
        tokens = layer._lookup_num_tokens(5)
        hidden = torch.arange(tokens * 12, dtype=torch.float32).reshape(tokens, 2, 6) / 32
        memory = layer.embed(ids).flatten(1)
        projected = torch.nn.functional.linear(memory, layer.wkv.weight).reshape(tokens, 3, 6)
        keys = torch.nn.functional.rms_norm(projected[:, :2], (6,), eps=layer.eps) * layer.k_weight
        query = torch.nn.functional.rms_norm(hidden, (6,), eps=layer.eps) * layer.q_weight
        scores = (keys * query).sum(-1) / 6 ** 0.5
        expected = (
            torch.sigmoid(scores.sign() * scores.abs().clamp_min(1e-6).sqrt())[:, :, None] * projected[:, 2, None]
        )
        mask = ids[:, 0] >= 0
        if sequence_parallel:
            mask = torch.nn.functional.pad(mask, (0, tokens * 4 - 5))[tokens:2 * tokens]
        expected = expected.masked_fill(~mask[:, None, None], 0)
        for value in (hidden, hidden.flatten(1)):
            torch.testing.assert_close(layer(value, ids), expected.reshape_as(value))
        assert torch.count_nonzero(layer(hidden, ids, torch.zeros(5, dtype=torch.bool))) == 0
        with pytest.raises(ValueError, match="token_mask"):
            layer(hidden, ids, torch.ones(4, dtype=torch.bool))
        with pytest.raises(ValueError, match="not aligned"):
            layer(hidden[:0], ids)
        layer.use_engram_proj_gate_fusion_op = True
        layer.proj_weights_key = layer.proj_weights_value = None
        fused = Mock(return_value=torch.ones_like(hidden))
        monkeypatch.setattr(torch.ops.omni_npu, "engram_proj_gate", fused)
        torch.testing.assert_close(layer(hidden, ids), torch.ones_like(hidden).masked_fill(~mask[:, None, None], 0))
        fused.assert_called_once()
        post_norm = engram.PanGuEngramPostNorm(6, eps=1e-6)
        torch.testing.assert_close(post_norm(hidden), torch.nn.functional.rms_norm(hidden, (6,), eps=1e-6))


def test_cpu_kernels_in_subprocess(tmp_path):
    run_isolated_pytest(str(Path(__file__).resolve()) + "::TestEngramCpuKernels", tmp_path,
                        timeout=120, interpreter=True)
