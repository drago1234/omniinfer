# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""Hardware regressions for PanGu hash arithmetic, slot history and graph replay."""

from dataclasses import replace
import os
from pathlib import Path
import numpy as np
import pytest
import torch
import torch_npu
from omni_npu.layers import engram
from tests.integration.layers.engram_test_utils import run_isolated_pytest
from tests.integration.layers.engram_reference import PanGuNGramContextProvider, PanGuNGramHashReference

pytestmark = pytest.mark.skipif(not torch.npu.is_available(), reason="requires an Ascend NPU")


def device_copy(value, device, use_strides=False):
    if use_strides:
        storage = torch.empty((*value.shape[:-1], value.shape[-1] * 2), dtype=value.dtype, device=device)
        result = storage[..., ::2]
        result.copy_(value)
        return result
    return value.to(device)


def batch(seed, tokens, depth, dtype=torch.int64, *, num_reqs=None):
    rng = np.random.default_rng(seed)
    if num_reqs is None:
        num_reqs = min(tokens + 2, 9)
    lengths = rng.multinomial(tokens, np.ones(num_reqs) / num_reqs)
    starts = np.cumsum([0, *lengths])
    offsets = rng.integers(0, 8, num_reqs)
    positions = np.concatenate([np.arange(length) + offset for length, offset in zip(lengths, offsets)])
    ids = rng.integers(0, 200001, tokens)
    dead = rng.random(tokens) < 0.1
    lookbacks = rng.integers(0, 200001, (num_reqs, depth))
    lookbacks[rng.random(lookbacks.shape) < 0.25] = -1
    lookback_dead = rng.random(lookbacks.shape) < 0.15
    return (
        torch.tensor(ids, dtype=dtype),
        torch.tensor(positions, dtype=torch.int64),
        torch.tensor(starts, dtype=torch.int32),
        torch.tensor(dead),
        torch.tensor(lookbacks, dtype=dtype),
        torch.tensor(lookback_dead),
    )


class Validator:
    def __init__(self, device):
        self.device = device
        self.config = engram.PanGuEngramConfig(
            layer_ids=(3, 18),
            vocab_sizes=(1714000, 1714000),
            max_ngram_size=3,
            num_heads_per_ngram=8,
            embedding_dim_per_ngram=512,
            pad_token_id=0,
            seed=42,
            hidden_size=1536,
            num_streams=4,
            post_norm=True,
            rms_norm_eps=1e-5,
            table_dtype=torch.bfloat16,
        )
        self.passed = []
        self.set_config(self.config)

    def set_config(self, config):
        self.layout = engram.EngramLayout(config)
        self.hashers = [PanGuNGramHashReference(self.layout, layer) for layer in self.layout.layer_ids]
        self.cache_module = torch.nn.Module()
        self.state = engram.NgramHashState(self.layout, self.device, self.cache_module)

    def expected(self, inputs, *, sp_rank=0, sp_size=1):
        ids, positions, starts, dead, lookbacks, lookback_dead = inputs
        windows = PanGuNGramContextProvider(self.layout.max_ngram_size, self.layout.pad_token_id).build_from_inputs(
            ids, positions, starts, dead, lookbacks, lookback_dead
        )
        relative = torch.stack([hasher(windows) for hasher in self.hashers], dim=1)
        result = (relative + self.layout.offsets).int()
        if sp_size > 1 and ids.numel():
            chunk = (ids.numel() + sp_size - 1) // sp_size
            owners = torch.arange(ids.numel()) // chunk
            result[owners != sp_rank] = -1
        return result

    def upload(self, inputs, use_strides=False):
        return tuple(device_copy(value, self.device, use_strides) for value in inputs)

    def run(self, inputs, *, cache=None, block_size=0, slot_mapping=None, block_table=None, out=None):
        self.state._cache = cache
        self.state.block_size = block_size
        return self.state(*inputs, slot_mapping=slot_mapping, block_table=block_table, out=out)

    def check(self, name, inputs, **kwargs):
        actual = self.run(self.upload(inputs, use_strides="strided" in name), **kwargs)
        try:
            expected = self.expected(inputs, sp_rank=self.state.sp_rank, sp_size=self.state.sp_size)
            torch.testing.assert_close(actual.cpu(), expected, rtol=0, atol=0)
        except AssertionError as exc:
            raise AssertionError(f"Failed hash case: {name}") from exc
        self.passed.append(name)

    def random_cases(self):
        configs = [
            self.config,
            replace(self.config, vocab_sizes=(101, 101), pad_token_id=7),
            replace(
                self.config,
                layer_ids=(0,),
                max_ngram_size=2,
                vocab_sizes=(113,),
                num_heads_per_ngram=1,
                embedding_dim_per_ngram=64,
            ),
            replace(
                self.config,
                layer_ids=(1, 6),
                max_ngram_size=4,
                vocab_sizes=(101,) * 3,
                num_heads_per_ngram=3,
                embedding_dim_per_ngram=192,
            ),
        ]
        for config_idx, config in enumerate(configs):
            self.set_config(config)
            # Exercise both tile sizes around 256 once; other configs vary
            # n-gram order, head padding and signed hash arithmetic.
            cases = (
                [(0, torch.int64), (1, torch.int64), (256, torch.int64), (257, torch.int64), (7, torch.int32)]
                if config_idx == 0
                else [(37, torch.int64)]
            )
            for tokens, dtype in cases:
                inputs = batch(42 + tokens, tokens, config.max_ngram_size - 1, dtype)
                self.check(f"random_config{config_idx}_tokens{tokens}_{dtype}", inputs)
            if config_idx == 0:
                self.check("strided_config0", batch(912, 37, config.max_ngram_size - 1))
            print(f"random config {config_idx}: exact", flush=True)

    def cache_cases(self, sp_rank=0, sp_size=1):
        self.set_config(self.config)
        self.state.sp_rank = sp_rank
        self.state.sp_size = sp_size
        block_size = 4
        # Request A's blocks 2,5,7; B's blocks 1,4,6, in a shared physical pool.
        blocks_cpu = torch.tensor([[2, 5, 7], [1, 4, 6]], dtype=torch.int32)
        cache = torch.full((32,), 777, dtype=torch.int32, device=self.device)

        def step(name, sequences, offsets, order, explicit=False, padded=False):
            lengths = [len(sequence) for sequence in sequences]
            starts = torch.tensor(np.cumsum([0, *lengths]), dtype=torch.int32)
            ids = torch.tensor([v for sequence in sequences for v in sequence], dtype=torch.int64)
            positions = torch.tensor(
                [offset + i for sequence, offset in zip(sequences, offsets) for i in range(len(sequence))],
                dtype=torch.int64,
            )
            dead = ids == 999
            ids[dead] = 88
            window = torch.full((len(order), 2), -1, dtype=torch.int64)
            window_dead = torch.zeros_like(window, dtype=torch.bool)
            reference_window = window.clone()
            reference_dead = window_dead.clone()
            snapshot = cache.cpu()
            current_blocks = blocks_cpu[order]
            slots = []
            for req, offset in enumerate(offsets):
                for distance in (1, 2):
                    pos = offset - distance
                    if pos >= 0:
                        slot = int(current_blocks[req, pos // block_size]) * block_size + pos % block_size
                        value = int(snapshot[slot])
                        reference_window[req, distance - 1] = 88 if value == -1 else value
                        reference_dead[req, distance - 1] = value == -1
                slots.extend(
                    int(current_blocks[req, pos // block_size]) * block_size + pos % block_size
                    for pos in range(offset, offset + lengths[req])
                )
            if explicit:
                # P/D or offload loaded KV does not populate our slot history.
                # Known prompt lookbacks must override even stale nonzero slots.
                window[:] = torch.tensor([51, 52])
                reference_window.copy_(window)
                reference_dead.zero_()
            slots = torch.tensor(slots, dtype=torch.int64)
            if padded:
                slots[-1] = -1
            inputs = (ids, positions, starts, dead, window, window_dead)
            reference_inputs = (*inputs[:4], reference_window, reference_dead)
            out = self.run(
                self.upload(inputs),
                cache=cache,
                slot_mapping=slots.to(self.device),
                block_table=device_copy(current_blocks, self.device, True),
                block_size=block_size,
            )
            expected = self.expected(reference_inputs, sp_rank=sp_rank, sp_size=sp_size)
            torch.testing.assert_close(out.cpu(), expected, rtol=0, atol=0)
            expected_cache = snapshot.clone()
            valid = slots >= 0
            expected_cache[slots[valid]] = torch.where(dead, -1, ids).int()[valid]
            torch.testing.assert_close(cache.cpu(), expected_cache, rtol=0, atol=0)
            self.passed.append(name)

        step("cache_prefill", [[10, 11, 12, 13], [20, 21, 22]], [0, 0], [0, 1])
        step("cache_chunk_boundary", [[14, 15], [23, 24]], [4, 3], [0, 1])
        step("cache_request_reorder", [[25], [16]], [5, 6], [1, 0])
        step("cache_speculative_drafts", [[17, 18, 19]], [7], [0])
        step("cache_reject_overwrite", [[117, 118]], [7], [0])
        step("cache_accepted_history", [[119]], [9], [0])
        step("cache_explicit_pd_lookback", [[60]], [5], [1], explicit=True)
        step("cache_dead_token", [[999, 62]], [6], [1])
        step("cache_dead_lookback", [[63]], [8], [1])
        step("cache_slot_reuse_new_request", [[101, 102]], [0], [0])
        step("cache_padding_no_write", [[103, 104]], [2], [0], padded=True)
        print("slot cache: exact", flush=True)

    def graph_cases(self, sp_rank=0, sp_size=1, single_query=False):
        self.set_config(self.config)
        self.state.sp_rank = sp_rank
        self.state.sp_size = sp_size

        source = batch(731, 33, 2, num_reqs=1 if single_query else None)
        inputs = self.upload(source)
        out = torch.empty((33, 2, 16), dtype=torch.int32, device=self.device)
        # Synthetic identity block table; unique requests own disjoint slots.
        blocks = torch.arange(9 * 4, dtype=torch.int32).reshape(9, 4).to(self.device)
        cache = torch.zeros(9 * 4 * 16, dtype=torch.int32, device=self.device)
        slots = torch.arange(33, device=self.device, dtype=torch.int64)
        kwargs = dict(cache=cache, slot_mapping=slots, block_table=blocks, block_size=16, out=out)
        # Supply complete explicit history so this test can vary inputs without
        # relying on the synthetic slots; cache write/read semantics are tested above.
        inputs[4].clamp_min_(0)
        stream = torch.npu.Stream()
        stream.wait_stream(torch.npu.current_stream())
        with torch.npu.stream(stream):
            for _ in range(3):
                self.run(inputs, **kwargs)
        torch.npu.current_stream().wait_stream(stream)
        torch.npu.synchronize()
        graph = torch.npu.NPUGraph()
        with torch.npu.graph(graph, stream=stream):
            self.run(inputs, **kwargs)
        for seed in (732, 733, 734):
            changed = batch(seed, 33, 2, num_reqs=1 if single_query else None)
            changed[4].clamp_min_(0)
            for target, value in zip(inputs, changed):
                target.copy_(value)
            graph.replay()
            expected = self.expected(changed, sp_rank=sp_rank, sp_size=sp_size)
            torch.testing.assert_close(out.cpu(), expected, rtol=0, atol=0)
            self.passed.append(f"graph_replay_seed{seed}")
        print("graph replay: exact", flush=True)


@pytest.fixture
def validator():
    torch.npu.set_device(0)
    torch.set_num_threads(1)
    return Validator(torch.device("npu", 0))


# Storage and sharding combinations are covered by test_engram_parallel.py.
@pytest.fixture
def fusion(tmp_path):
    from vllm.config import VllmConfig, set_current_vllm_config
    from vllm.distributed import (
        init_distributed_environment,
        initialize_model_parallel,
        get_tp_group,
        destroy_model_parallel,
        destroy_distributed_environment,
    )

    torch.npu.set_device(0)
    torch.set_num_threads(1)
    config = VllmConfig()
    config.scheduler_config.max_num_batched_tokens = 64
    old_dtype = torch.get_default_dtype()
    torch.set_default_dtype(torch.bfloat16)
    with set_current_vllm_config(config):
        init_distributed_environment(1, 0, f'file://{tmp_path}/dist', 0, 'hccl')
        initialize_model_parallel(1, 1)
        try:
            layout = engram.EngramLayout(
                engram.PanGuEngramConfig(
                    layer_ids=(0,),
                    vocab_sizes=(101, 101),
                    max_ngram_size=3,
                    num_heads_per_ngram=8,
                    embedding_dim_per_ngram=512,
                    pad_token_id=0,
                    seed=42,
                    hidden_size=32,
                    num_streams=2,
                    post_norm=True,
                    rms_norm_eps=1e-5,
                    table_dtype=torch.bfloat16,
                )
            )
            with torch.device('npu:0'):
                module = engram.PanGuEngramFusion(layout, 0, get_tp_group(), False)
            with torch.no_grad():
                module.wkv.weight.normal_(std=0.01)
                module.k_weight.uniform_(0.5, 1.5)
                module.q_weight.uniform_(0.5, 1.5)
                module.embed_tokens.weight.copy_(torch.randn_like(module.embed_tokens.weight) * 0.01)
            module.process_weights_after_loading()
            yield module
            torch.npu.synchronize()
        finally:
            destroy_model_parallel()
            destroy_distributed_environment()
            torch.set_default_dtype(old_dtype)


class TestEngramDevice:
    # Collect these cases only inside the bounded, clean child process.
    __test__ = os.environ.get("OMNI_ENGRAM_TEST_CHILD") == "1"

    @staticmethod
    @pytest.mark.parametrize("case", ["random_cases", "cache_cases", "graph_cases"])
    def test_device_hash(validator, case):
        getattr(validator, case)()
        assert validator.passed

    @staticmethod
    @pytest.mark.parametrize("single_query", [False, True])
    def test_sp_hash_graph_replays_new_inputs(validator, single_query):
        validator.graph_cases(sp_rank=1, sp_size=4, single_query=single_query)

    @staticmethod
    @pytest.mark.parametrize("num_reqs", [8, 9, 129])
    def test_device_hash_repeated_request_boundaries(validator, num_reqs):
        # Empty rows at both ends and between requests exercise upper-bound search.
        lengths = np.zeros(num_reqs, dtype=np.int64)
        lengths[num_reqs // 2] = 3
        starts = np.cumsum([0, *lengths])
        inputs = (
            torch.tensor([17, 23, 31]),
            torch.arange(3),
            torch.tensor(starts, dtype=torch.int32),
            torch.zeros(3, dtype=torch.bool),
            torch.full((num_reqs, 2), -1),
            torch.zeros((num_reqs, 2), dtype=torch.bool),
        )
        validator.check(f"repeated_boundaries_{num_reqs}", inputs)

    @staticmethod
    @pytest.mark.parametrize("model_type,static_compile,sp_tail", [
        pytest.param("low_latency", False, False, id="low_latency_eager"),
        pytest.param("low_latency", True, True, id="low_latency_compiled_sp"),
        pytest.param("high_throughout", False, True, id="high_throughout_eager_sp"),
        pytest.param("high_throughout", True, False, id="high_throughout_compiled"),
    ])
    @torch.inference_mode()
    def test_model_hash_lookup_pipeline_replays_new_inputs(fusion, model_type, static_compile, sp_tail, monkeypatch):
        """Replay must consume the updated hashes and completed lookup rows."""
        from types import SimpleNamespace
        from omni_npu.v1.models.pangu import pangu_v2_moe, pangu_ultra_moe

        if sp_tail:
            # Exercise rank 7's local SP slice without a distributed launch: with
            # unsharded heads this path has no collective, only token selection.
            fusion.config = replace(fusion.config, shard_heads=False)
            fusion.use_sequence_parallel = True
            fusion.embed_tokens.model_tp_size = 8
            group = SimpleNamespace(rank_in_group=7)
            monkeypatch.setattr(engram, "get_tp_group", lambda: group)

        model_cls = (
            pangu_v2_moe.OpenPanguV2Model if model_type == "low_latency" else pangu_ultra_moe.OpenPanguModel
        )
        model = model_cls.__new__(model_cls)
        torch.nn.Module.__init__(model)
        model.start_layer = 0
        model.engram_layout = fusion.layout
        model.engram_hash = SimpleNamespace(has_bound_cache=lambda: True)
        model._get_engram_metadata = lambda: True
        offsets = fusion.layout.offsets.to('npu:0').int()

        def compute(ids, positions, lookbacks):
            # Arithmetic correctness has separate kernel tests; here the producer
            # must read replay inputs and the consumer must observe its new output.
            hashes = offsets[None, :, :] + (ids % 17)[:, None, None]
            return hashes.masked_fill((ids < 0)[:, None, None], -1).int()

        model._compute_engram_hashes = compute
        decoder = torch.nn.Module()
        decoder.engram = fusion
        model.layers = torch.nn.ModuleList([decoder])
        model.end_layer = 1
        model._prepare_key = id(model)
        engram._engram_model_owners[model._prepare_key] = model

        class Pipeline(torch.nn.Module):
            def forward(self, ids, hidden):
                hashes = model.prepare_model_embeddings(ids, ids, None)
                output = fusion(hidden, hashes[:, 0])
                return output, hashes.clone(), fusion.staged_rows.clone()

        # Rank 7 owns two valid tokens plus one padding token in the SP case.
        ids = torch.arange(23, device='npu:0', dtype=torch.int32)
        hidden = torch.randn(3 if sp_tail else 23, 2, 32, device='npu:0')
        torch._dynamo.reset()
        backend = 'eager'
        if static_compile:
            import npugraph_ex

            config = npugraph_ex.CompilerConfig()
            config.mode = 'npugraph_ex'
            config.force_eager = True
            config.static_kernel_compile = True
            backend = npugraph_ex.get_npu_backend(compiler_config=config)
        compiled = torch.compile(Pipeline(), backend=backend, fullgraph=True)
        for _ in range(3):
            compiled(ids, hidden)
        torch.npu.synchronize()
        graph = torch.npu.NPUGraph()
        with torch.npu.graph(graph):
            output, hashes, rows = compiled(ids, hidden)
        for step in range(5):
            ids.add_(1)
            if step == 4:
                ids[-2:] = -1
            graph.replay()
            actual_output, actual_hashes, actual_rows = output.cpu(), hashes.cpu(), rows.cpu()
            expected_hashes = compute(ids, ids, None)
            lookup_ids = expected_hashes[:, 0]
            if sp_tail:
                lookup_ids = torch.nn.functional.pad(lookup_ids[21:], (0, 0, 0, 1), value=-1)
            expected_rows = torch.empty_like(rows)
            fusion.embed_tokens.lookup(lookup_ids, expected_rows)
            torch.testing.assert_close(actual_rows, expected_rows.cpu(), rtol=0, atol=0)
            fusion.prepare_embeddings(expected_hashes[:, 0])
            torch.testing.assert_close(actual_hashes, expected_hashes.cpu(), rtol=0, atol=0)
            # Check graph replay against eager execution of the same fused path.
            expected_output = fusion(hidden, expected_hashes[:, 0]).cpu()
            torch.testing.assert_close(actual_output, expected_output, rtol=0, atol=0)

    @staticmethod
    @pytest.mark.parametrize("graph_mode", [False, True], ids=["eager", "captured_graph"])
    @torch.inference_mode()
    def test_online_post_refreshes_projections_without_refilling_table(fusion, graph_mode, monkeypatch):
        """An existing graph must consume B projections at the original addresses."""
        from omni_npu.v1.layers.attention.weight_utils import run_post_weight_load

        fusion.use_engram_proj_gate_fusion_op = True
        fusion.process_weights_after_loading()
        table = fusion.embed_tokens.weight
        table_before = table.clone()
        table_pointer = table.data_ptr()
        mapped_pointer = fusion.embed_tokens._mapped_weight.data_ptr()
        key, value = fusion.proj_weights_key, fusion.proj_weights_value
        pointers = (key.data_ptr(), value.data_ptr())
        model = torch.nn.Module()
        model.add_module("engram", fusion)
        # A repeated table initialization must fail even when dummy fill is disabled.
        monkeypatch.setattr(fusion.embed_tokens, "process_weights_after_loading",
                            lambda: pytest.fail("online post must not initialize the table"))
        ids = fusion.layout.offsets[0].int().to("npu:0").expand(7, -1).contiguous()
        fusion.staged_rows = torch.empty((7, fusion.layout.n_hash_cols, fusion.layout.head_dim),
                                         device="npu:0", dtype=torch.bfloat16)
        fusion.prepare_embeddings(ids)
        hidden = torch.randn(7, fusion.num_streams, fusion.hidden_size, device="npu:0")
        for _ in range(3):
            fusion(hidden, ids)
        torch.npu.synchronize()
        graph = None
        if graph_mode:
            graph = torch.npu.NPUGraph()
            with torch.npu.graph(graph):
                output = fusion(hidden, ids)
            graph.replay()
        else:
            output = fusion(hidden, ids)
        original_output = output.cpu()

        for _ in range(3):
            fusion.wkv.weight.mul_(0.5)
            fusion.k_weight.mul_(1.125)
            fusion.q_weight.add_(0.125)
            run_post_weight_load(model)
            assert fusion.proj_weights_key is key and fusion.proj_weights_value is value
            assert (key.data_ptr(), value.data_ptr()) == pointers
            packed = fusion.wkv.weight.reshape(fusion.num_streams + 1, fusion.hidden_size, fusion.memory_dim)
            torch.testing.assert_close(key, packed[:-1].transpose(1, 2), rtol=0, atol=0)
            torch.testing.assert_close(value, packed[-1].t(), rtol=0, atol=0)
            if graph is not None:
                graph.replay()
            else:
                output = fusion(hidden, ids)
            actual = output.cpu()
            assert not torch.equal(actual, original_output)
            assert table.data_ptr() == table_pointer
            assert fusion.embed_tokens._mapped_weight.data_ptr() == mapped_pointer
            torch.testing.assert_close(table, table_before, rtol=0, atol=0)

    @staticmethod
    @pytest.mark.parametrize("tokens", [1, 7, 129])
    @torch.inference_mode()
    def test_projection_gate_inference_replay_overwrites_poisoned_output(tokens):
        """No-cache inference must fully write its output, including the tile tail."""
        torch.npu.set_device(0)
        generator = torch.Generator().manual_seed(1047 + tokens)

        def random_tensor(*shape):
            return torch.randn(shape, generator=generator).bfloat16().to("npu:0")

        hidden = random_tensor(tokens, 2, 32)
        memory = random_tensor(tokens, 128) * 0.1
        key = random_tensor(2, 128, 32) * 0.1
        value = random_tensor(128, 32) * 0.1
        gamma_key, gamma_query = random_tensor(2, 32), random_tensor(2, 32)

        def run():
            return torch.ops.omni_npu.engram_proj_gate(
                hidden, memory, key, value, gamma_key, gamma_query, 1e-6, 1e-5
            )

        def reference():
            # Independent FP32 CPU math, matching the fused kernel's accumulation.
            queries = hidden.cpu().float()
            embeddings = memory.cpu().float()
            keys = torch.stack([embeddings @ weight for weight in key.cpu().float()], dim=1)
            keys *= torch.rsqrt(keys.square().mean(-1, keepdim=True) + 1e-5)
            queries *= torch.rsqrt(queries.square().mean(-1, keepdim=True) + 1e-5)
            scores = ((keys * gamma_key.cpu().float()) * (queries * gamma_query.cpu().float())).sum(-1)
            scores /= 32 ** 0.5
            gates = torch.sigmoid(scores.sign() * scores.abs().clamp_min(1e-6).sqrt())
            values = embeddings @ value.cpu().float()
            return (gates[:, :, None] * values[:, None, :]).bfloat16()

        for _ in range(3):
            run()
        torch.npu.synchronize()
        graph = torch.npu.NPUGraph()
        with torch.npu.graph(graph):
            output = run()
        for _ in range(3):
            hidden.add_(0.25)
            memory.mul_(0.75)
            output.fill_(float("nan"))
            graph.replay()
            torch.npu.synchronize()
            actual = output.cpu()
            assert actual.isfinite().all()
            torch.testing.assert_close(actual, reference(), rtol=2e-2, atol=1e-4)


def test_device_cases_in_subprocess(tmp_path):
    from vllm.triton_utils import HAS_TRITON

    if not HAS_TRITON:
        pytest.skip("requires an active Ascend Triton backend; vLLM is using placeholder kernels")
    run_isolated_pytest(str(Path(__file__).resolve()) + "::TestEngramDevice", tmp_path)
