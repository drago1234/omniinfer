# SPDX-License-Identifier: MIT
# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""NPU DFlash compatibility and selector proposals."""

from types import SimpleNamespace

import torch

from vllm.v1.worker.gpu.spec_decode.dflash.speculator import DFlashSpeculator
from vllm.v1.worker.gpu.spec_decode.dspark.speculator import DSparkSpeculator

from omni_npu.worker.npu.sampler import gumbel_sample


def _ensure_acl_update_stream(manager):
    if hasattr(manager, "_omni_acl_update_stream"):
        return

    manager._omni_acl_update_stream = torch.npu.Stream()


def _refresh_draft_graph_context(speculator, metadata):
    manager = speculator.query_cudagraph_manager
    if manager is None:
        return

    _ensure_acl_update_stream(manager)
    speculator._omni_runtime_draft_attn_metadata = metadata

    def context_provider(desc):
        return SimpleNamespace(
            attn_metadata=speculator._omni_runtime_draft_attn_metadata,
            batch_descriptor=SimpleNamespace(
                num_tokens=desc.num_tokens, num_reqs=desc.num_reqs
            ),
        )

    manager._omni_acl_context_provider = context_provider


class NPUDFlashSpeculator(DFlashSpeculator):
    def load_draft_model(self, target_model, target_attn_layer_names):
        # vLLM 0.25.1 reads lm_head from the outer target model.
        if hasattr(target_model, "get_language_model"):
            target_model = target_model.get_language_model()
        return super().load_draft_model(target_model, target_attn_layer_names)

    def _build_draft_attn_metadata(self, *args, **kwargs):
        metadata = super()._build_draft_attn_metadata(*args, **kwargs)
        _refresh_draft_graph_context(self, metadata)
        return metadata


class NPUDFlash2Speculator(NPUDFlashSpeculator):
    """DFlash backbone followed by conditional candidate sampling."""

    _speculator_name = "DFlash2"

    def __init__(self, vllm_config, device: torch.device):
        super().__init__(vllm_config, device)
        hf_config = vllm_config.speculative_config.draft_model_config.hf_config
        knobs = getattr(hf_config, "dflash_config", None) or {}
        trained_block_size = int(knobs.get("block_size", 0))
        runtime_block_size = self.num_query_per_req
        if trained_block_size and trained_block_size != runtime_block_size:
            raise ValueError(
                "DFlash2 runtime layout must match training: "
                f"dflash_config.block_size={trained_block_size}, but "
                f"runtime query block size={runtime_block_size}"
            )

        self._candidate_step_cols = torch.arange(
            self.num_speculative_steps, dtype=torch.int32, device=device
        )
        self._candidate_logits: torch.Tensor | None = None
        if self.draft_logits is not None:
            self._candidate_logits = torch.full(
                (self.max_num_reqs, self.vocab_size),
                float("-inf"),
                dtype=self.draft_logits.dtype,
                device=device,
            )

    def _generate_draft(
        self,
        num_reqs,
        num_tokens_padded,
        attn_metadata,
        slot_mappings,
        num_tokens_across_dp,
        cudagraph_runtime_mode=None,
    ) -> None:
        if cudagraph_runtime_mode is None:
            from vllm.config.compilation import CUDAGraphMode

            cudagraph_runtime_mode = CUDAGraphMode.NONE
        last_hidden_states = self._run_model(
            num_tokens_padded,
            attn_metadata,
            slot_mappings,
            num_tokens_across_dp,
            cudagraph_runtime_mode,
        )
        num_steps = self.num_speculative_steps
        num_sample = num_reqs * num_steps
        sample_hidden = last_hidden_states[
            self.sample_indices[:num_sample]
        ].reshape(num_reqs, num_steps, -1)
        query = self.input_buffers.input_ids[: num_reqs * self.num_query_per_req]
        anchor_ids = query.reshape(num_reqs, self.num_query_per_req)[:, 0]
        candidate_ids, edge_scores = self.model.compute_candidate_lattice(
            sample_hidden, anchor_ids
        )
        idx_map = self.sample_idx_mapping[:num_sample].reshape(num_reqs, num_steps)
        sample_pos = self.sample_pos[:num_sample].reshape(num_reqs, num_steps)
        batch = torch.arange(num_reqs, device=candidate_ids.device)
        predecessor_index = torch.zeros(
            num_reqs, dtype=torch.long, device=candidate_ids.device
        )
        for step in range(num_steps):
            scores = edge_scores[batch, step, predecessor_index]
            candidates = candidate_ids[:, step]
            if self.draft_logits is not None:
                if self._candidate_logits is None:
                    raise RuntimeError(
                        "DFlash2 probabilistic sampling requires an initialized candidate logits buffer"
                    )
                logits = self._candidate_logits[:num_reqs]
                logits.fill_(float("-inf"))
                logits.scatter_(1, candidates, scores.to(logits.dtype))
                sampled = gumbel_sample(
                    logits,
                    idx_map[:, step],
                    self.temperature,
                    self.seeds,
                    sample_pos[:, step] - 1,
                    apply_temperature=True,
                    output_processed_logits=self.draft_logits,
                    output_processed_logits_col=self._candidate_step_cols[step],
                    use_fp64=self.use_fp64_gumbel,
                )
                selected_index = (
                    candidates == sampled.unsqueeze(-1)
                ).long().argmax(dim=-1)
            else:
                selected_index = scores.argmax(dim=-1)
                sampled = candidates[batch, selected_index]
            self.draft_tokens[:num_reqs, step] = sampled
            predecessor_index = selected_index


class NPUDSparkSpeculator(DSparkSpeculator, NPUDFlashSpeculator):
    """Upstream Markov sampling with shared NPU DFlash support."""

    def load_draft_model(self, target_model, target_attn_layer_names):
        if hasattr(target_model, "get_language_model"):
            target_model = target_model.get_language_model()
        return super().load_draft_model(target_model, target_attn_layer_names)
