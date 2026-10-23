# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""Real TP2 x DP2 collectives for shared host tables with optional TP head sharding.

Run with four visible NPUs. Uses tiny, nonconstant tables and checks uneven DP
batches, padding, sequence parallelism, and graph replay with changing hashes.
"""

from datetime import timedelta
from pathlib import Path
import os
import torch
import torch_npu
import pytest
from tests.integration.layers.engram_test_utils import run_isolated_pytest

MODES = {'cpu_shared/npu_npu/none': False, 'cpu_shared/npu_npu/tp': True}


@torch.inference_mode()
def _worker(rank, rendezvous, output_dir, static_compile):
    # Match the FULL-graph Decode service communication backend.
    os.environ["HCCL_OP_EXPANSION_MODE"] = "AI_CPU"
    os.environ["HCCL_DETERMINISTIC"] = "true"
    # Direct module tests bypass EngineArgs, which normally loads NPU patches.
    from vllm.plugins import load_general_plugins

    load_general_plugins()
    from vllm.config import VllmConfig, set_current_vllm_config
    from vllm.distributed import parallel_state
    from vllm.distributed.parallel_state import (
        init_distributed_environment,
        get_tp_group,
        get_dp_group,
        destroy_distributed_environment,
    )
    from vllm.forward_context import set_forward_context
    from omni_npu.layers import engram
    from omni_npu.v1.distributed import parallel_state_ext as group_state

    # npugraph_ex requires torchrun-style local topology for static compilation.
    os.environ["LOCAL_WORLD_SIZE"] = "4"
    os.environ["WORLD_SIZE"] = "4"
    os.environ["RANK"] = str(rank)
    os.environ["LOCAL_RANK"] = str(rank)
    torch.set_num_threads(1)
    torch.npu.set_device(rank)
    torch.set_default_dtype(torch.bfloat16)
    config = VllmConfig()
    config.parallel_config.tensor_parallel_size = 2
    config.parallel_config.data_parallel_size = 2
    config.parallel_config.data_parallel_rank = rank // 2
    config.parallel_config.distributed_executor_backend = 'external_launcher'
    config.scheduler_config.max_num_batched_tokens = 32
    print(f"Engram rank {rank}: initializing distributed groups", flush=True)
    with set_current_vllm_config(config):
        init_distributed_environment(4, rank, rendezvous, rank, 'hccl', timedelta(seconds=180))
        parallel_state.initialize_model_parallel(2, 1, backend='hccl')
        try:
            dp_rank = get_dp_group().rank_in_group
            tp_rank = get_tp_group().rank_in_group
            # Match service import order: model helpers before npugraph_ex.
            from omni_npu.v1.models.pangu.pangu_v2_moe import OpenPanguV2Model

            completed = []
            for name, shard_heads in MODES.items():
                print(f"Engram rank {rank}: {name}, static_compile={static_compile}", flush=True)
                # Each mode represents a fresh deployment with one shared group.
                group_state.destroy_parallel_state_ext_groups()
                # Shared TP shards are reused by the two DP replicas.
                layout = engram.EngramLayout(
                    engram.PanGuEngramConfig(
                        layer_ids=(0,),
                        vocab_sizes=(31, 37),
                        max_ngram_size=3,
                        num_heads_per_ngram=7,
                        embedding_dim_per_ngram=224,
                        pad_token_id=0,
                        seed=42,
                        hidden_size=32,
                        num_streams=2,
                        post_norm=True,
                        rms_norm_eps=1e-5,
                        table_dtype=torch.bfloat16,
                        shard_heads=shard_heads,
                    )
                )
                group_state.initialize_engram_shared_group(layout.config.shard_heads)
                storage_group = group_state.get_engram_shared_group()
                with torch.device(f'npu:{rank}'):
                    fusion = engram.PanGuEngramFusion(layout, 0, storage_group, False)
                assert fusion.staged_rows is None
                emb = fusion.embed_tokens
                full = (
                    torch.arange(emb.num_embeddings * 32, dtype=torch.float32).reshape(-1, 32).remainder(251) / 128
                ).bfloat16()
                emb.weight_loader(emb.weight, full)
                assert emb.weight.device.type == 'cpu'
                assert emb.storage is not None
                assert emb.weight.numel() == emb.part_num_embeddings * 32
                cache_attention = torch.nn.Module()
                cache_attention.kv_cache = torch.empty(4, 16, 1, device=f'npu:{rank}')
                state = engram.NgramHashState(layout, torch.device(f'npu:{rank}'), cache_attention)
                state.block_size = 16
                assert state.ensure_cache()

                def make_ids(n, step):
                    device = f'npu:{rank}'
                    raw = torch.arange(n, device=device, dtype=torch.int64) + step * 3 + dp_rank * 100
                    ids = state(
                        raw,
                        torch.arange(n, device=device),
                        torch.tensor([0, n], dtype=torch.int32, device=device),
                        torch.zeros(n, dtype=torch.bool, device=device),
                        torch.full((1, 2), -1, dtype=torch.int64, device=device),
                        torch.zeros((1, 2), dtype=torch.bool, device=device),
                        slot_mapping=torch.arange(n, device=device),
                        block_table=torch.arange(4, dtype=torch.int32, device=device).view(1, -1),
                    )[:, 0]
                    if n:
                        ids[-1] = -1
                    return ids

                def stage(ids):
                    fusion.staged_rows = torch.ops.omni_npu.engram_lookup_rows(ids, fusion._lookup_key)

                def expected(ids, sequence):
                    values = full[ids.cpu().long().clamp_min(0)]
                    values[ids.cpu() < 0] = 0
                    if sequence:
                        chunk = (ids.shape[0] + 1) // 2
                        values = torch.nn.functional.pad(values, (0, 0, 0, 0, 0, chunk * 2 - ids.shape[0]))
                        values = values[tp_rank * chunk : (tp_rank + 1) * chunk]
                    return values

                # Different lengths on each DP, including a rank with no tokens.
                for sizes in ([3, 7], [0, 5], [9, 2]):
                    counts = torch.tensor(sizes, dtype=torch.int32, device='cpu')
                    ids = make_ids(sizes[dp_rank], 1)
                    with set_forward_context(None, config, num_tokens=ids.shape[0], num_tokens_across_dp=counts):
                        direct = emb(ids)
                        torch.testing.assert_close(direct.cpu(), expected(ids, False), rtol=0, atol=0)
                        for sequence in ((False,) if shard_heads else (False, True)):
                            fusion.use_sequence_parallel = sequence
                            stage(ids)
                            actual = fusion.embed(ids)
                            torch.testing.assert_close(actual.cpu(), expected(ids, sequence), rtol=0, atol=0)

                counts = torch.tensor([8, 8], dtype=torch.int32, device='cpu')
                backend = 'eager'
                if static_compile:
                    import npugraph_ex

                    compiler_config = npugraph_ex.CompilerConfig()
                    compiler_config.mode = 'npugraph_ex'
                    compiler_config.force_eager = True
                    compiler_config.static_kernel_compile = True
                    backend = npugraph_ex.get_npu_backend(compiler_config=compiler_config)
                print(f"Engram rank {rank}: compiling and replaying WKV", flush=True)
                # Check full checkpoint loading and WKV output order for TP/SP.
                for sequence in ((False,) if shard_heads else (False, True)):
                    with torch.device(f'npu:{rank}'):
                        projection = engram.PanGuEngramFusion(layout, 0, storage_group, sequence)
                    width = (layout.config.num_streams + 1) * layout.config.hidden_size
                    checkpoint = (
                        torch.arange(width * layout.config.memory_dim, dtype=torch.float32)
                        .reshape(width, -1).remainder(19).sub(9).div(256).bfloat16()
                    )
                    weight = projection.wkv.weight
                    weight.weight_loader(weight, checkpoint)
                    projection_tp = 1 if sequence else get_tp_group().world_size
                    assert projection.wkv.tp_size == projection_tp
                    assert tuple(weight.shape) == (width // projection_tp, layout.config.memory_dim)
                    begin = 0 if sequence else tp_rank * (width // projection_tp)
                    torch.testing.assert_close(
                        weight.cpu(), checkpoint[begin : begin + width // projection_tp], rtol=0, atol=0
                    )
                    memory = (
                        torch.arange(6 * layout.config.memory_dim, device=f'npu:{rank}', dtype=torch.float32)
                        .reshape(6, -1).remainder(37).div(256).bfloat16()
                    )
                    reference_weight = checkpoint.to(memory.device)
                    torch._dynamo.reset()
                    project = torch.compile(projection.wkv, backend=backend, fullgraph=True)
                    with set_forward_context(None, config, num_tokens=6, num_tokens_across_dp=counts.new_full((2,), 6)):
                        for _ in range(3):
                            project(memory)
                        torch.npu.synchronize()
                        projection_graph = torch.npu.NPUGraph()
                        with torch.npu.graph(projection_graph):
                            projected = project(memory)
                        for scale in (1, 2):
                            memory.mul_(scale)
                            projection_graph.replay()
                            torch.npu.synchronize()
                            reference = torch.nn.functional.linear(memory, reference_weight)
                            torch.testing.assert_close(projected, reference, rtol=1e-2, atol=1e-4)
                    del project, projection_graph, projection

                # Check the model-owned producer and graph-local staging.
                with torch.device(f'npu:{rank}'):
                    fusion = engram.PanGuEngramFusion(layout, 0, storage_group, False)
                fusion.embed_tokens.weight_loader(fusion.embed_tokens.weight, full)
                assert fusion.staged_rows is None
                from types import SimpleNamespace

                attention = torch.nn.Module()
                attention.layer_name = 'engram.test'
                attention.kv_cache = torch.zeros(4, 16, 1, device=f'npu:{rank}')
                model_state = engram.NgramHashState(layout, torch.device(f'npu:{rank}'), attention)
                model_state.block_size = 16
                model = OpenPanguV2Model.__new__(OpenPanguV2Model)
                torch.nn.Module.__init__(model)
                model.start_layer = 0
                model.engram_hash = model_state
                model.engram_layout = layout
                decoder = torch.nn.Module()
                decoder.engram = fusion
                model.layers = torch.nn.ModuleList([decoder])
                model.end_layer = 1
                model._prepare_key = id(model)
                engram._engram_model_owners[model._prepare_key] = model
                raw = torch.arange(8, device=f'npu:{rank}', dtype=torch.int64) + dp_rank * 100
                positions = torch.arange(8, device=raw.device)
                slots = positions.clone()
                slots[-1] = -1
                lookbacks = torch.full((1, 2), -1, dtype=torch.int64, device=raw.device)
                metadata = SimpleNamespace(
                    num_prefills=0,
                    query_start_loc=torch.tensor([0, 8], dtype=torch.int32, device=raw.device),
                    slot_mapping=slots,
                    decode=SimpleNamespace(
                        block_table=torch.arange(4, dtype=torch.int32, device=raw.device).view(1, -1)
                    ),
                    prefill=None,
                )
                # Profiling must neither consume stale staging nor update history.
                for profile_metadata in (None, {'engram.test': metadata}):
                    bound_kv = attention.kv_cache
                    attention.kv_cache = torch.empty(0, device=raw.device)
                    with set_forward_context(profile_metadata, config, num_tokens=8, num_tokens_across_dp=counts):
                        dummy = model.prepare_model_embeddings(raw, positions, lookbacks)
                        assert dummy is None
                        assert model_state._cache is None
                    attention.kv_cache = bound_kv

                def model_forward(raw, positions, lookbacks):
                    hashes = model.prepare_model_embeddings(raw, positions, lookbacks)
                    if hashes is None:
                        return None, None
                    # Real PanGu records mHC events after Engram preparation.
                    # A persistent graph-input mutation here is illegal.
                    rows = fusion.embed(hashes[:, 0])
                    # Attention performs a slice assignment before mHC.
                    # This also makes Dynamo check graph-input versions.
                    hidden = raw.float()
                    hidden[:] = hidden + 1
                    event = torch.npu.Event()
                    event.record()
                    return hashes, rows

                if name == 'cpu_shared/npu_npu/tp' and not static_compile:
                    # Use vLLM's actual trace-once wrapper, which drops guards.
                    # Normal torch.compile retracing would hide a stale skip.
                    from vllm.compilation.decorators import support_torch_compile
                    from vllm.config.compilation import CompilationMode
                    from vllm.config import CUDAGraphMode

                    @support_torch_compile(dynamic_arg_dims={"raw": 0, "positions": 0})
                    class ProfileHarness(torch.nn.Module):
                        def __init__(self, *, vllm_config, prefix=""):
                            super().__init__()

                        def forward(self, raw, positions, lookbacks):
                            return model_forward(raw, positions, lookbacks)

                    profile_config = VllmConfig()
                    profile_config.compilation_config.mode = CompilationMode.DYNAMO_TRACE_ONCE
                    profile_config.compilation_config.backend = 'eager'
                    with set_current_vllm_config(profile_config):
                        harness = ProfileHarness(vllm_config=profile_config)
                        for iteration in range(2):
                            bound_kv = attention.kv_cache
                            attention.kv_cache = torch.empty(0, device=raw.device)
                            # The actual Omni wrapper bypasses both metadata-free
                            # profiling and metadata-bearing non-FULL warmup.
                            for profile_metadata in (None, {'engram.test': metadata}):
                                with set_forward_context(profile_metadata, profile_config):
                                    assert harness(raw, positions, lookbacks) == (None, None)
                                    assert harness.compiled == bool(iteration)
                            attention.kv_cache = bound_kv
                            with set_forward_context(
                                {'engram.test': metadata}, profile_config, cudagraph_runtime_mode=CUDAGraphMode.FULL
                            ):
                                live_hashes, live_rows = harness(raw, positions, lookbacks)
                                assert live_hashes is not None and harness.compiled
                                torch.testing.assert_close(
                                    live_rows.cpu(), expected(live_hashes[:, 0], False), rtol=0, atol=0
                                )
                            raw.add_(3)
                    del harness
                print(f"Engram rank {rank}: compiling and replaying model pipeline", flush=True)
                fusion.use_sequence_parallel = not shard_heads
                torch._dynamo.reset()
                compiled_model = torch.compile(model_forward, backend=backend, fullgraph=True)
                with set_forward_context({'engram.test': metadata}, config, num_tokens=8, num_tokens_across_dp=counts):
                    for _ in range(3):
                        compiled_model(raw, positions, lookbacks)
                    torch.npu.synchronize()
                    model_graph = torch.npu.NPUGraph()
                    with torch.npu.graph(model_graph):
                        replay_hashes, replay_rows = compiled_model(raw, positions, lookbacks)
                    for step in (2, 5, 9):
                        raw.copy_(torch.arange(8, device=raw.device) + step * 3 + dp_rank * 100)
                        model_graph.replay()
                        torch.npu.synchronize()
                        reference = make_ids(8, step)
                        torch.testing.assert_close(replay_hashes[:, 0].cpu(), reference.cpu(), rtol=0, atol=0)
                        torch.testing.assert_close(
                            replay_rows.cpu(), expected(reference, fusion.use_sequence_parallel), rtol=0, atol=0
                        )
                del model_graph, replay_hashes, replay_rows, compiled_model, model_state
                del fusion, emb
                torch.npu.synchronize()
                torch.distributed.barrier()
                completed.append(name)
                if rank == 0:
                    print(f"Engram mode passed: {name}", flush=True)
                Path(output_dir, f'rank{rank}.txt').write_text('\n'.join(completed))
        finally:
            parallel_state.destroy_model_parallel()
            assert group_state._ENGRAM_SHARED is None
            destroy_distributed_environment()


class TestEngramParallel:
    __test__ = os.environ.get("OMNI_ENGRAM_TEST_CHILD") == "1"

    @staticmethod
    @pytest.mark.skipif(
        not torch.npu.is_available() or torch.npu.device_count() < 4, reason='requires four visible Ascend NPUs'
    )
    @pytest.mark.parametrize("static_compile", [False, True])
    def test_storage_modes_tp2_dp2(tmp_path, static_compile):
        torch.multiprocessing.spawn(
            _worker, args=(f'file://{tmp_path}/dist', str(tmp_path), static_compile), nprocs=4, join=True
        )
        for rank in range(4):
            assert (tmp_path / f'rank{rank}.txt').read_text().splitlines() == list(MODES)


def test_storage_modes_in_subprocess(tmp_path):
    from vllm.triton_utils import HAS_TRITON

    if not torch.npu.is_available() or torch.npu.device_count() < 4:
        pytest.skip("requires four visible Ascend NPUs")
    if not HAS_TRITON:
        pytest.skip("requires an active Ascend Triton backend; vLLM is using placeholder kernels")
    run_isolated_pytest(str(Path(__file__).resolve()) + "::TestEngramParallel", tmp_path)
