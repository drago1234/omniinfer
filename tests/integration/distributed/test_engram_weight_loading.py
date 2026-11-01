# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""Real CPU mmap/Gloo loading with Tensor on the writer and None on its peer."""

from datetime import timedelta
import os
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.multiprocessing as mp


def _worker(rank, rendezvous, shard_heads):
    from omni_npu.layers.engram import DPSharedEngramStorage, ParallelEngramEmbedding

    torch.distributed.init_process_group(
        "gloo", init_method=rendezvous, rank=rank, world_size=2,
        timeout=timedelta(seconds=45),
    )
    try:
        group = SimpleNamespace(ranks=[0, 1], rank_in_group=rank, world_size=2,
                                cpu_group=torch.distributed.group.WORLD)

        def broadcast(value):
            values = [value]
            torch.distributed.broadcast_object_list(values, src=0)
            return values[0]

        group.broadcast_object = broadcast
        start, rows = (2, 3) if shard_heads else (0, 7)
        embedding = ParallelEngramEmbedding.__new__(ParallelEngramEmbedding)
        torch.nn.Module.__init__(embedding)
        embedding.num_embeddings, embedding.dim = 7, 2
        embedding.storage = DPSharedEngramStorage((rows, 2), torch.bfloat16, group)
        embedding.weight = torch.nn.Parameter(embedding.storage.tensor, requires_grad=False)
        embedding.weight.engram_vocab_start = start
        embedding.weight.engram_shared_ptr = embedding.weight.data_ptr()
        embedding.weight.dummy_weight_value = 1.0
        embedding.dummy_weights = True
        embedding.process_weights_after_loading()
        embedding.dummy_weights = False
        pointer = embedding.weight.data_ptr()

        for offset in (0, 10):
            # Simulate RL's inputs; the non-writer never creates a source table.
            source = torch.arange(16, dtype=torch.float32).reshape(8, 2) + offset if rank == 0 else None
            embedding.weight_loader(embedding.weight, source)
            expected = (torch.arange(start * 2, (start + rows) * 2).reshape(rows, 2) + offset).bfloat16()
            torch.testing.assert_close(embedding.weight, expected, rtol=0, atol=0)
            assert embedding.weight.data_ptr() == pointer

            # Both ranks must report validation failure, then accept another round.
            invalid = source[:2] if rank == 0 else None
            with pytest.raises(RuntimeError, match="load failed.*requires at least"):
                embedding.weight_loader(embedding.weight, invalid)
            torch.testing.assert_close(embedding.weight, expected, rtol=0, atol=0)
            # Finish reading this version before the writer starts the next update.
            embedding.storage.synchronize()
    finally:
        torch.distributed.destroy_process_group()


@pytest.mark.parametrize("shard_heads", [False, True])
def test_writer_tensor_and_peer_none_update_original_shared_table(tmp_path, monkeypatch, shard_heads):
    # Avoid container hostname resolution and keep this CPU test self-contained.
    monkeypatch.setenv("GLOO_SOCKET_IFNAME", "lo")
    mp.spawn(_worker, args=(f"file://{tmp_path}/rendezvous", shard_heads), nprocs=2, join=True)


@torch.inference_mode()
def _npu_worker(rank, rendezvous):
    from vllm.plugins import load_general_plugins

    load_general_plugins()
    from vllm.config import VllmConfig, set_current_vllm_config
    from vllm.distributed import parallel_state
    from omni_npu.layers import engram
    from omni_npu.v1.distributed import parallel_state_ext
    from omni_npu.v1.models.pangu.pangu_ultra_moe import PanguUltraMoEForCausalLM
    from omni_npu.worker.npu_worker import NPUWorker

    torch.set_num_threads(1)
    torch.npu.set_device(rank)
    config = VllmConfig()
    config.parallel_config.tensor_parallel_size = 2
    config.parallel_config.data_parallel_size = 2
    config.parallel_config.data_parallel_rank = rank // 2
    config.parallel_config.distributed_executor_backend = "external_launcher"
    with set_current_vllm_config(config):
        parallel_state.init_distributed_environment(4, rank, rendezvous, rank, "hccl", timedelta(seconds=120))
        parallel_state.initialize_model_parallel(2, 1, backend="hccl")
        try:
            for shard_heads in (False, True):
                parallel_state_ext.destroy_parallel_state_ext_groups()
                group = parallel_state_ext.initialize_engram_shared_group(shard_heads)
                layout = engram.EngramLayout(engram.PanGuEngramConfig(
                    layer_ids=(0,), vocab_sizes=(31, 37), max_ngram_size=3,
                    num_heads_per_ngram=2, embedding_dim_per_ngram=64, pad_token_id=0,
                    seed=42, hidden_size=32, num_streams=2, post_norm=True,
                    rms_norm_eps=1e-5, table_dtype=torch.bfloat16, shard_heads=shard_heads,
                ))
                embedding = engram.ParallelEngramEmbedding(layout, 0, group)
                # Exercise the real Ultra loader without allocating a full backbone.
                model = PanguUltraMoEForCausalLM.__new__(PanguUltraMoEForCausalLM)
                torch.nn.Module.__init__(model)
                model.config = SimpleNamespace(tie_word_embeddings=False, n_routed_experts=0,
                                               num_nextn_predict_layers=0)
                model.num_redundant_experts = 0
                model.model = torch.nn.Module()
                decoder = torch.nn.Module()
                decoder.engram = torch.nn.Module()
                decoder.engram.embed_tokens = embedding
                model.model.layers = torch.nn.ModuleList([decoder])
                model.model.norm = torch.nn.Linear(4, 4, bias=False, device=f"npu:{rank}")
                worker = SimpleNamespace(model_runner=SimpleNamespace(get_model=lambda: model))
                role = NPUWorker.get_engram_weight_layout(worker)
                expected_group = [rank % 2, rank % 2 + 2] if shard_heads else [0, 1, 2, 3]
                name = "model.layers.0.engram.embed_tokens.weight"
                assert role == dict(global_rank=rank, shared_ranks=expected_group,
                                    writer_global_rank=expected_group[0], is_writer=rank == expected_group[0],
                                    table_names=[name])
                pointers = embedding.weight.data_ptr(), embedding._mapped_weight.data_ptr()
                ids = torch.tensor([[int(begin) + token % capacity for begin, capacity in
                                     zip(layout.offsets[0], layout.head_sizes)] for token in range(5)],
                                   dtype=torch.int32)
                ids[-1] = -1
                device_ids = ids.to(f"npu:{rank}")
                output = torch.empty((5, embedding.part_n_hash_cols, embedding.dim),
                                     device=f"npu:{rank}", dtype=torch.bfloat16)

                def full_table(offset):
                    return (torch.arange(embedding.num_embeddings * embedding.dim, dtype=torch.float32)
                            .reshape(-1, embedding.dim).remainder(127).div(128).add(offset).bfloat16())

                def expected(offset):
                    selected = ids[:, embedding.head_start:embedding.head_start + embedding.part_n_hash_cols]
                    values = (selected.clamp_min(0).long()[:, :, None] * embedding.dim
                              + torch.arange(embedding.dim)).remainder(127).float().div(128).add(offset)
                    return values.bfloat16().masked_fill(selected[:, :, None] < 0, 0)

                def update(offset):
                    source = full_table(offset) if role["is_writer"] else None
                    assert model.load_weights([(name.replace("embed_tokens", "embed"), source)]) == {name}

                update(0)
                embedding.lookup(device_ids, output)
                torch.npu.synchronize()
                graph = torch.npu.NPUGraph()
                with torch.npu.graph(graph):
                    embedding.lookup(device_ids, output)
                for offset in (1, 2):
                    torch.npu.synchronize()
                    update(offset)
                    graph.replay()
                    torch.npu.synchronize()
                    torch.testing.assert_close(output.cpu(), expected(offset), rtol=0, atol=0)
                    embedding.lookup(device_ids, output)
                    torch.npu.synchronize()
                    torch.testing.assert_close(output.cpu(), expected(offset), rtol=0, atol=0)
                    assert pointers == (embedding.weight.data_ptr(), embedding._mapped_weight.data_ptr())
                    # Frozen table: load only a regular parameter, preserving table contents.
                    model.load_weights([("model.norm.weight", torch.ones(4, 4))])
                    graph.replay()
                    torch.npu.synchronize()
                    torch.testing.assert_close(output.cpu(), expected(offset), rtol=0, atol=0)
                    with pytest.raises(ValueError, match="only valid for an online Engram"):
                        model.load_weights([("model.norm.weight", None)])
                    with pytest.raises(RuntimeError, match="load failed.*writer requires"):
                        model.load_weights([(name, None)])
                    invalid = full_table(offset)[:1] if role["is_writer"] else None
                    with pytest.raises(RuntimeError, match="load failed.*requires at least"):
                        model.load_weights([(name, invalid)])
                    start = embedding.weight.engram_vocab_start
                    if role["is_writer"]:
                        embedding.weight.engram_vocab_start = embedding.num_embeddings + 1
                    try:
                        with pytest.raises(RuntimeError, match="load failed"):
                            update(offset)
                    finally:
                        embedding.weight.engram_vocab_start = start
                print(f"NPU_PASS rank={rank} sharding={'tp' if shard_heads else 'none'} "
                      f"shared_ranks={group.ranks} writer={role['is_writer']} "
                      "updates=2 eager=exact graph=exact frozen=exact errors=propagated pointers=stable",
                      flush=True)
                del graph, output, embedding, model
                torch.npu.synchronize()
        finally:
            parallel_state_ext.destroy_parallel_state_ext_groups()
            parallel_state.destroy_model_parallel()
            parallel_state.destroy_distributed_environment()


@pytest.mark.skipif(os.environ.get("OMNI_ENGRAM_NPU_TEST") != "1", reason="opt-in four-NPU validation")
def test_npu_writer_notifications_tp2_dp2(tmp_path, monkeypatch):
    from tests.integration.layers.engram_test_utils import run_isolated_pytest

    if os.environ.get("OMNI_ENGRAM_TEST_CHILD") != "1":
        run_isolated_pytest(str(Path(__file__).resolve()) + "::test_npu_writer_notifications_tp2_dp2",
                            tmp_path, timeout=300)
        print((tmp_path / "engram.log").read_text())
        return
    monkeypatch.setenv("GLOO_SOCKET_IFNAME", "lo")
    monkeypatch.setenv("HCCL_OP_EXPANSION_MODE", "AI_CPU")
    monkeypatch.setenv("HCCL_DETERMINISTIC", "true")
    mp.spawn(_npu_worker, args=(f"file://{tmp_path}/npu_rendezvous",), nprocs=4, join=True)
