# Copyright (c) 2026, Huawei Technologies Co., Ltd. All rights reserved.
# SPDX-License-Identifier: MIT
#
# This file is based on vLLM implementation:
# Copyright 2023 The vLLM team.
# https://github.com/vllm-project/vllm/blob/v0.14.0/vllm/model_executor/models/openpangu_mtp.py
#
# The upstream vLLM implementation retains attribution to vllm-ascend and is
# adapted from:
# https://github.com/vllm-project/vllm/blob/v0.7.3/vllm/model_executor/models/deepseek_mtp.py

from collections.abc import Iterable

import torch
import torch.nn as nn
from transformers import PretrainedConfig

from vllm.compilation.decorators import support_torch_compile
from vllm.config import CUDAGraphMode, VllmConfig
from vllm.distributed import get_dp_group, get_tp_group
from vllm.forward_context import get_forward_context
from vllm.model_executor.layers.fused_moe import fused_moe_make_expert_params_mapping
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import ReplicatedLinear
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.model_executor.models.interfaces import SupportsPP
from vllm.model_executor.models.utils import maybe_prefix
from vllm.sequence import IntermediateTensors
from vllm.utils.math_utils import cdiv

from omni_npu.attention.backends.utils import SPManager
from omni_npu.v1.layers.attention.weight_utils import run_post_weight_load
from omni_npu.model_config.config_loader.loader import model_extra_config
from omni_npu.v1.distributed.parallel_state_ext import get_local_world_group
from omni_npu.v1.layers.logits_processor import NPULogitsProcessor
from omni_npu.v1.layers.vocab_parallel_embedding import NPUParallelLMHead, NPUVocabParallelEmbedding

from .pangu_v2_moe import (
    OpenPanguV2DecoderLayer,
    _maybe_gather_and_unpadding,
    _maybe_padding_and_slice,
    rewrite_mome_conv_name,
    high_throughout,
)


class SharedHead(nn.Module):
    def __init__(
        self,
        config: PretrainedConfig,
        prefix: str,
        quant_config: QuantizationConfig | None = None,
    ) -> None:
        super().__init__()
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        parall_cfg = model_extra_config.parall_config
        if parall_cfg.ena_local_lmhead_parallel:
            local_lmhead = get_local_world_group().world_size > 1
            dp_lmhead = False
        elif parall_cfg.ena_dp_lmhead_parallel:
            local_lmhead = False
            dp_lmhead = get_dp_group().world_size > 1
        else:
            local_lmhead = False
            dp_lmhead = False
        self.head = NPUParallelLMHead(
            config.vocab_size,
            config.hidden_size,
            quant_config=quant_config,
            prefix=maybe_prefix(prefix, "head"),
            local_lmhead_parallel=local_lmhead,
            dp_parallel=dp_lmhead,
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.norm(hidden_states)


@support_torch_compile(
    dynamic_arg_dims={
        "input_ids": 0,
        # MRoPE: (3, seq_len); language RoPE: (seq_len,).
        "positions": -1,
        "previous_hidden_states": 0,
        "inputs_embeds": 0,
    }
)
class OpenPanguV2MultiTokenPredictorLayer(nn.Module):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str) -> None:
        super().__init__()

        config = vllm_config.speculative_config.draft_model_config.hf_config
        self.config = config
        self.quant_config = vllm_config.quant_config

        self.enorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.hnorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.eh_proj = ReplicatedLinear(
            config.hidden_size * 2,
            config.hidden_size,
            bias=False,
            quant_config=self.quant_config,
            prefix=maybe_prefix(prefix, "eh_proj"),
        )
        self.shared_head = SharedHead(
            config=config,
            quant_config=self.quant_config,
            prefix=maybe_prefix(prefix, "shared_head"),
        )
        self.mtp_block = OpenPanguV2DecoderLayer(config, prefix, vllm_config)
        self.mtp_block._tail_refs = (
            None,            # tail_mhc_pre — unused when tail_use_mhc is False
            nn.Identity(),   # tail_layernorm — no-op
            True,            # is_model_tail
        )

        self.need_tp_padding = model_extra_config.parall_config.ena_seq_parallel

    def set_side_stream(self, side_stream: torch.npu.Stream, fetch_stream: torch.npu.Stream = None) -> None:
        """Set the shared side/fetch streams for MTP block."""
        self.mtp_block.set_side_stream(side_stream, fetch_stream)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        previous_hidden_states: torch.Tensor,
        inputs_embeds: torch.Tensor | None = None,
        spec_step_index: int = 0,
    ) -> torch.Tensor:
        assert inputs_embeds is not None
        # The SP shard's length specializes to the first capture (width / tp),
        # while positions carries the dynamic full length. Re-slice so the shard
        # and the all-gathers sized from it track each capture. Only prefill
        # widths vary, so decode-only graphs keep what they always had.
        if (
            self.need_tp_padding
            and model_extra_config.operator_opt_config.enable_prefill_cudagraph
            and torch.compiler.is_compiling()
        ):
            # cdiv, like SPManager.slice_tokens: floor would drop the last row
            # of a width that does not divide, and the all-gather would then
            # rebuild a sequence shorter than the main model's.
            local = cdiv(positions.shape[-1], get_tp_group().world_size)
            assert inputs_embeds.shape[0] >= local, (inputs_embeds.shape, local)
            inputs_embeds = inputs_embeds[:local]
            previous_hidden_states = previous_hidden_states[:local]
        inputs_embeds = self.enorm(inputs_embeds)
        previous_hidden_states = self.hnorm(previous_hidden_states)

        hidden_states, _ = self.eh_proj(
            torch.cat([inputs_embeds, previous_hidden_states], dim=-1)
        )

        cos, sin = self.mtp_block.self_attn.rotary_emb.get_cos_sin(positions)
        hidden_states, residual, _, _, sk_event = self.mtp_block.mhc_head(hidden_states)
        hidden_states, _, _, _, _ = self.mtp_block(
            hidden_states, residual, None, None, cos, sin, sk_event,
        )

        return hidden_states


class OpenPanguV2MultiTokenPredictor(nn.Module):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        config = vllm_config.model_config.hf_config
        self.mtp_start_layer_idx = config.num_hidden_layers
        self.num_mtp_layers = config.num_nextn_predict_layers
        # to map the exact layer index from weights
        self.layers = torch.nn.ModuleDict(
            {
                str(idx): OpenPanguV2MultiTokenPredictorLayer(
                    vllm_config=vllm_config, 
                    prefix=f"{prefix}.layers.{idx}",
                )
                for idx in range(self.mtp_start_layer_idx, self.mtp_start_layer_idx + self.num_mtp_layers)
            }
        )
        self.wrapped_layers = None
        if (
            model_extra_config.parall_config.ena_seq_parallel
            and model_extra_config.operator_opt_config.enable_prefill_cudagraph
            and vllm_config.compilation_config.cudagraph_mode != CUDAGraphMode.NONE
        ):
            # A replay reads these shards at their capture-time addresses.
            # Sized for the widest batch the scheduler can issue, so one address
            # serves every capture. Decode shards need no pinning: upstream
            # already hands those to the graph at a stable address. Allocating
            # them does not commit a forward to using them; _graph_serves_batch
            # decides that per step, so an eager step keeps the fresh shard.
            self.compilation_config = vllm_config.compilation_config
            # Only a graph-served batch is pinned, so the widest capture bounds
            # the shard. Sizing by max_num_batched_tokens instead would hold 16x
            # the rows a replay can reach. _pin_sp_shard asserts the fit.
            graph_tokens = (
                self.compilation_config.max_cudagraph_capture_size
                or vllm_config.scheduler_config.max_num_batched_tokens
            )
            shard_shape = (
                cdiv(graph_tokens,
                     vllm_config.parallel_config.tensor_parallel_size),
                config.hidden_size,
            )
            dtype = vllm_config.model_config.dtype
            self.register_buffer(
                "inputs_embeds_shard",
                torch.empty(shard_shape, dtype=dtype),
                persistent=False,
            )
            self.register_buffer(
                "previous_hidden_states_shard",
                torch.empty(shard_shape, dtype=dtype),
                persistent=False,
            )
        else:
            self.inputs_embeds_shard = None
            self.previous_hidden_states_shard = None
        self.embed_tokens = NPUVocabParallelEmbedding(
            config.vocab_size,
            config.hidden_size,
        )
        self.logits_processor = NPULogitsProcessor(config.vocab_size)

        use_multi_stream = model_extra_config.operator_opt_config.enable_multi_stream
        if use_multi_stream:
            self.side_stream = torch.npu.Stream()
            self.fetch_stream = torch.npu.Stream()
            for idx in range(self.mtp_start_layer_idx, self.mtp_start_layer_idx + self.num_mtp_layers):
                self.layers[str(idx)].set_side_stream(self.side_stream, self.fetch_stream)

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        enable_scatter = model_extra_config.parall_config.ena_seq_parallel
        return self.embed_tokens(input_ids, enable_scatter=enable_scatter)

    def _graph_serves_batch(self) -> bool:
        """Whether a captured graph will serve this forward.

        Pinning only matters then: outside a graph the fresh shard is read at
        the address it was just written to. Reads the width the builder stored
        on the metadata, so the shards and the attention layout see one answer.
        """
        attn_metadata = get_forward_context().attn_metadata
        if not attn_metadata:
            return False
        metadata = attn_metadata[next(iter(attn_metadata))]
        return getattr(metadata, "prefill_graph_width", None) is not None

    @staticmethod
    def _pin_sp_shard(buf: torch.Tensor, shard: torch.Tensor) -> torch.Tensor:
        """Copy an SP shard into its persistent buffer before it enters the graph.

        A per-forward shard has a new address every step, and the one seen at
        capture is unmapped by the empty_cache that follows. The copy also
        drops the storage offset slice_tokens leaves on its view.
        """
        assert (
            shard.dtype == buf.dtype
            and shard.shape[1:] == buf.shape[1:]
            and shard.shape[0] <= buf.shape[0]
        ), f"SP shard {tuple(shard.shape)} {shard.dtype} does not fit " \
           f"{tuple(buf.shape)} {buf.dtype}"
        pinned = buf[: shard.shape[0]]
        pinned.copy_(shard)
        return pinned

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        previous_hidden_states: torch.Tensor,
        inputs_embeds: torch.Tensor | None = None,
        spec_step_idx: int = 0,
    ) -> torch.Tensor:
        enable_scatter = model_extra_config.parall_config.ena_seq_parallel
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)
        if enable_scatter:
            sp_manager = SPManager.init_sp(previous_hidden_states.size(0), get_tp_group())
            if inputs_embeds.size(0) == previous_hidden_states.size(0):
                inputs_embeds = sp_manager.slice_tokens(inputs_embeds)
            previous_hidden_states = sp_manager.slice_tokens(previous_hidden_states)
            if self.inputs_embeds_shard is not None and self._graph_serves_batch():
                inputs_embeds = self._pin_sp_shard(self.inputs_embeds_shard, inputs_embeds)
                previous_hidden_states = self._pin_sp_shard(
                    self.previous_hidden_states_shard, previous_hidden_states
                )
                # A ReduceScatter-ed inputs_embeds has to split the same
                # width; a graph would hide the mismatch behind the pad rows.
                assert inputs_embeds.size(0) == previous_hidden_states.size(0), (
                    inputs_embeds.shape, previous_hidden_states.shape)
        current_step_idx = spec_step_idx % self.num_mtp_layers

        if self.wrapped_layers is not None:
            layer = self.wrapped_layers[str(self.mtp_start_layer_idx + current_step_idx)]
        else:
            layer = self.layers[str(self.mtp_start_layer_idx + current_step_idx)]
    
        hidden_states = layer(
            input_ids,
            positions,
            previous_hidden_states,
            inputs_embeds,
            current_step_idx,
        )

        if enable_scatter:
            hidden_states = get_tp_group().all_gather(hidden_states, dim=0)

        return hidden_states

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
        spec_step_idx: int = 0,
    ) -> torch.Tensor:
        current_step_idx = spec_step_idx % self.num_mtp_layers
        mtp_layer = self.layers[str(self.mtp_start_layer_idx + current_step_idx)]
        hidden_states = mtp_layer.shared_head(hidden_states)
        logits = self.logits_processor(
            mtp_layer.shared_head.head, hidden_states
        )
        return logits


class OpenPanguV2MTP(nn.Module, SupportsPP):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        self.config = vllm_config.model_config.hf_config
        self.model = OpenPanguV2MultiTokenPredictor(
            vllm_config=vllm_config, prefix=maybe_prefix(prefix, "model")
        )

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model.embed_input_ids(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        spec_step_idx: int = 0,
    ) -> torch.Tensor:
        hidden_states = self.model(
            input_ids,
            positions,
            hidden_states,
            inputs_embeds,
            spec_step_idx,
        )
        return hidden_states

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
        spec_step_idx: int = 0,
    ) -> torch.Tensor | None:
        return self.model.compute_logits(hidden_states, spec_step_idx)

    def get_spec_layer(self, name):
        if (
            "layers" in name
            and hasattr(self.config, "num_nextn_predict_layers")
            and self.config.num_nextn_predict_layers > 0
        ):
            layer_idx = int(name.split("layers.")[-1].split(".")[0])
            mtp_idx = layer_idx - self.config.num_hidden_layers
            if mtp_idx >= 0 and mtp_idx < self.config.num_nextn_predict_layers:
                return layer_idx
        return None

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        stacked_params_mapping = [
            ("gate_up_proj", "gate_proj", 0),
            ("gate_up_proj", "up_proj", 1),
        ]

        expert_params_mapping = fused_moe_make_expert_params_mapping(
            self.model,
            ckpt_gate_proj_name="gate_proj",
            ckpt_down_proj_name="down_proj",
            ckpt_up_proj_name="up_proj",
            num_experts=self.config.n_routed_experts,
            routed_experts_prefix="",
        )

        params_dict = dict(self.named_parameters())
        loaded_params: set[str] = set()
        for name, loaded_weight in weights:
            if "rotary_emb.inv_freq" in name:
                continue
            spec_layer = self.get_spec_layer(name)
            if spec_layer is None:
                continue

            name = self._rewrite_spec_layer_name(spec_layer, name)
            if high_throughout():
                name = rewrite_mome_conv_name(name)
            for param_name, weight_name, shard_id in stacked_params_mapping:
                if f".{weight_name}." not in name:
                    continue
                if ("mlp.experts." in name) and name not in params_dict:
                    continue
                name_mapped = name.replace(weight_name, param_name)
                if (
                    param_name == "fused_qkv_a_proj"
                    and name_mapped not in params_dict
                ):
                    continue
                name = name_mapped
                if name.endswith(".bias") and name not in params_dict:
                    continue
                param = params_dict[name]
                param.weight_loader(param, loaded_weight, shard_id)
                break
            else:
                for mapping in expert_params_mapping:
                    param_name, weight_name, expert_id, shard_id = mapping
                    if weight_name not in name:
                        continue
                    name = name.replace(weight_name, param_name)
                    param = params_dict[name]
                    param.weight_loader(
                        param,
                        loaded_weight,
                        name,
                        shard_id=shard_id,
                        expert_id=expert_id,
                    )
                    break
                else:
                    # Skip loading extra bias for GPTQ models.
                    if name.endswith(".bias") and name not in params_dict:
                        continue

                    if (
                        spec_layer != self.model.mtp_start_layer_idx
                        and ".layers" not in name
                    ):
                        continue

                    if name not in params_dict:
                        continue

                    param = params_dict[name]
                    weight_loader = getattr(
                        param, "weight_loader", default_weight_loader
                    )
                    weight_loader(param, loaded_weight)

            # MLA multi-stream split path: q_b_proj's weight_loader (installed
            # by NPUPanguSparseAttention._install_q_b_split_loaders) also
            # populates the q_b_nope_proj / q_b_pe_proj children's matching
            # params in-place. vLLM's "unloaded weights" check looks at
            # state_dict keys, so mark those synthetic names as loaded too
            # whenever the parent q_b_proj key was just consumed.
            if ".q_b_proj." in name:
                for synthetic in (
                    name.replace(".q_b_proj.", ".q_b_nope_proj."),
                    name.replace(".q_b_proj.", ".q_b_pe_proj."),
                ):
                    if synthetic in params_dict:
                        loaded_params.add(synthetic)

            loaded_params.add(name)

        if high_throughout():
            self.post_weight_load()
        return loaded_params

    def post_weight_load(self) -> None:
        run_post_weight_load(self)

    def _rewrite_spec_layer_name(self, spec_layer: int, name: str) -> str:
        """
        Rewrite the weight name to match the format of the original model.
        Add .mtp_block for modules in transformer layer block for spec layer
        and rename shared layer weights to be top level.
        """
        spec_layer_weight_names = [
            "embed_tokens",
            "enorm",
            "hnorm",
            "eh_proj",
            "shared_head",
        ]
        shared_weight_names = ["embed_tokens"]
        spec_layer_weight = False
        shared_weight = False
        for weight_name in spec_layer_weight_names:
            if weight_name in name:
                spec_layer_weight = True
                if weight_name in shared_weight_names:
                    shared_weight = True
                break
        if not spec_layer_weight:
            # treat rest weights as weights for transformer layer block
            name = name.replace(
                f"model.layers.{spec_layer}.", f"model.layers.{spec_layer}.mtp_block."
            )
        elif shared_weight:
            # treat shared weights as top level weights
            name = name.replace(f"model.layers.{spec_layer}.", "model.")
        return name
    
    def set_shared_weight(self, target_model: nn.Module) -> None:
        if hasattr(target_model, "embed_tokens"):
            self.model.embed_tokens = target_model.embed_tokens
        if hasattr(target_model, "lm_head"):
            for layer in self.model.layers.values():
                layer.shared_head.head = target_model.lm_head
