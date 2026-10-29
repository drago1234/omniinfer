# SPDX-License-Identifier: MIT
# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd. All Rights Reserved.

from __future__ import annotations

import inspect

from collections.abc import Iterable

import torch
import torch_npu
from torch import nn
from transformers import PretrainedConfig

from vllm.config import VllmConfig, get_current_vllm_config
from vllm.distributed import (
    get_pp_group,
)
from vllm.logger import init_logger
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.vocab_parallel_embedding import VocabParallelEmbedding
from vllm.model_executor.layers.vocab_parallel_embedding import ParallelLMHead
from vllm.model_executor.models.interfaces import (
    MixtureOfExperts,
    SupportsLoRA,
    SupportsPP,
)
from vllm.model_executor.models.openpangu import (
    OpenPanguSinkAttention as VllmOpenPanguSinkAttention,
    OpenPanguMoE as VllmOpenPanguMoE,
    OpenPanguModel as VllmOpenPanguModel,
    OpenPanguDecoderLayer as VllmOpenPanguDecoderLayer,
)
from vllm.model_executor.models.utils import (
    AutoWeightsLoader,
    PPMissingLayer,
    make_empty_intermediate_tensors_factory,
    make_layers,
    maybe_prefix,
)
from vllm.sequence import IntermediateTensors
from vllm.v1 import kv_cache_interface
from vllm.compilation.decorators import support_torch_compile
from vllm.forward_context import get_forward_context

from omni_npu.model_config.config_loader.loader import model_extra_config

logger = init_logger(__name__)
AttentionSpec = kv_cache_interface.AttentionSpec
KVCacheSpec = kv_cache_interface.KVCacheSpec
SinkFullAttentionSpec = getattr(
    kv_cache_interface,
    "SinkFullAttentionSpec",
    kv_cache_interface.FullAttentionSpec,
)


class NPUOpenPanguSinkAttention(VllmOpenPanguSinkAttention):
    def __init__(
        self,
        config: PretrainedConfig,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        **kwargs,
    ) -> None:
        # Whether to use the fused operator: k RMSNorm + RoPE + KV cache update.
        # Must be set before super().__init__(): the parent init ends with a
        # post_weight_load() call that dispatches to our override below,
        # which reads this flag.
        object.__setattr__(
            self,
            "enable_kv_rmsnorm_rope_cache",
            model_extra_config.operator_opt_config.enable_kv_rmsnorm_rope_cache,
        )
        super().__init__(config, hidden_size, num_heads, num_kv_heads, **kwargs)

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        forward_context: ForwardContext = get_forward_context()
        attn_metadata = forward_context.attn_metadata

        # QKV projection.
        qkv, _ = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.k_size, self.v_size], dim=-1)

        use_fused_kv_cache_update = (
            self.enable_kv_rmsnorm_rope_cache
            and attn_metadata is not None
            and (
                not isinstance(attn_metadata, dict)
                or self.attn.layer_name in attn_metadata
            )
        )

        if use_fused_kv_cache_update:
            # ------------------------------------------------------------------
            # Fused path:
            # 1. Apply RoPE to Q and export cos/sin for the fused KV-cache op.
            # 2. Use npu_kv_rmsnorm_rope_cache to:
            #      - RMSNorm K
            #      - apply RoPE to K
            #      - update K/V cache
            #
            # Notes:
            # - vLLM uses packed token layout. Here we reshape to the layout
            #   expected by the fused NPU operator.
            # - During DP dummy execution, batch data may be padded, so
            #   num_actual_tokens must be used instead of the raw tensor length.
            # ------------------------------------------------------------------
            q, _, cos, sin = self.rotary_emb(
                positions,
                q.contiguous(),
                output_cos_sin=True,
            )

            q = q.view(-1, self.q_size)
            k = k.view(-1, self.k_size)

            if isinstance(attn_metadata, dict):
                attn_metadata = attn_metadata[self.attn.layer_name]
            assert hasattr(attn_metadata, "slot_mapping") and hasattr(
                attn_metadata, "num_actual_tokens"
            ), "attn_metadata must provide slot_mapping and num_actual_tokens"

            self_kv_cache = self.attn.kv_cache[forward_context.virtual_engine]

            # Static sink KV is populated lazily on first use.
            if (
                not self.attn.sink_populated
                and self_kv_cache is not None
                and len(self_kv_cache) > 0
            ):
                torch.ops.vllm.maybe_populate_sink(
                    self_kv_cache[0],
                    self_kv_cache[1],
                    self.attn.layer_name,
                )

            key_cache, value_cache = self_kv_cache[0], self_kv_cache[1]
            # Slot mapping is built for the packed batch.
            # In multi-DP mode, idle DP ranks may still enter execute_dummy_batch() with
            # mock/padded tokens for shape alignment. In that case attn_metadata exists,
            # but only num_actual_tokens correspond to real requests.
            slots = attn_metadata.slot_mapping
            num_tokens = attn_metadata.num_actual_tokens

            # The fused operator expects BNSD-style input.
            # Current packed-token layout is treated as:
            #   B = 1, S = num_tokens
            batch_size = 1
            # Kernel contract: the fused op applies RoPE on the first
            # `qk_rope_dim` dims of each K head (64 for PanguProMoE), i.e.
            # exactly the dims rotary_emb produced cos/sin for.
            rope_rotary_dim = self.qk_rope_dim

            k_input = (
                k[:num_tokens]
                .view(batch_size, -1, self.num_kv_heads, self.attn.impl.head_size)
                .transpose(1, 2)
            )
            v_input = (
                v[:num_tokens]
                .view(batch_size, -1, self.num_kv_heads, self.v_channels)
                .transpose(1, 2)
            )

            cos_input = (
                cos[:num_tokens]
                .view(batch_size, -1, rope_rotary_dim)
                .unsqueeze(1)
                .repeat(1, self.num_kv_heads, 1, 1)
            )
            sin_input = (
                sin[:num_tokens]
                .view(batch_size, -1, rope_rotary_dim)
                .unsqueeze(1)
                .repeat(1, self.num_kv_heads, 1, 1)
            )
            assert cos.shape[-1] == rope_rotary_dim and sin.shape[-1] == rope_rotary_dim, (
                "fused npu_kv_rmsnorm_rope_cache expects cos/sin to cover the "
                f"rope dims (qk_rope_dim={rope_rotary_dim}, "
                f"got cos={cos.shape[-1]}, sin={sin.shape[-1]})"
            )

            torch_npu.npu_kv_rmsnorm_rope_cache(
                k_input,
                self.k_layernorm.weight,
                cos_input,
                sin_input,
                slots[:num_tokens].to(torch.int64),
                key_cache,
                value_cache,
                v=v_input,
                epsilon=self.k_layernorm.variance_epsilon,
                cache_mode="PA",
            )

        else:
            # ------------------------------------------------------------------
            # Default path:
            # Used when fused KV-cache update is disabled, or when attention
            # metadata is unavailable (for example during certain dummy runs).
            # In this case, follow the original OpenPangu logic:
            #   1. RMSNorm on K
            #   2. RoPE on Q/K
            # ------------------------------------------------------------------
            k = self.k_layernorm(k.view(-1, self.num_kv_heads, self.head_dim))
            q, k = self.rotary_emb(positions, q, k)

            q = q.view(-1, self.q_size)
            k = k.view(-1, self.k_size)

        # Attention output shape is based on V head dim instead of K/Q head dim.
        attn_output = self.attn(
            q,
            k,
            v,
            output_shape=torch.Size(
                [q.shape[0], q.shape[1] // self.head_dim * self.v_channels]
            ),
        )
        output, _ = self.o_proj(attn_output)
        return output

    def sink_key_interleave(self, weight: torch.Tensor) -> torch.Tensor:
        # Must interleave exactly the rope dims so that the statically cached
        # sink K matches the interleaved layout of runtime K (whose rope dims
        # are defined by config qk_rope_dim).
        rope_dim = self.qk_rope_dim
        key_rot = weight[..., :rope_dim]
        key_pass = weight[..., rope_dim:]
        o1 = key_rot[..., ::2]
        o2 = key_rot[..., 1::2]

        key_rot = torch.cat((o1, o2), dim=-1)
        return torch.cat((key_rot, key_pass), dim=-1)

    def post_weight_load(self) -> None:
        if hasattr(self, "k_layernorm") and self.k_layernorm is not None:
            param_sink_key = self.k_layernorm(self.param_sink_key)
        else:
            param_sink_key = self.param_sink_key
        if self.enable_kv_rmsnorm_rope_cache:
            # K & Q 保持一致，都做interleave(奇偶交叉处理)
            param_sink_key = self.sink_key_interleave(param_sink_key)
        self.attn.update_sink_kv(param_sink_key, self.param_sink_value)


class OpenPanguDecoderLayer(VllmOpenPanguDecoderLayer):
    """NPU variant of the vLLM OpenPangu decoder layer.

    Reuses the parent initialization wholesale; the only difference is that
    the sink-attention branch builds NPUOpenPanguSinkAttention (fused K
    RMSNorm + RoPE + KV-cache update) instead of OpenPanguSinkAttention.
    Everything else (MLA/embedded attention branches, MLP/MoE, layernorms,
    sandwich norm) stays identical to the parent.

    The swap is done by temporarily substituting the parent module's
    OpenPanguSinkAttention symbol around the super().__init__() call, so the
    NPU attention is built directly with the parent's own argument wiring and
    layer-name registration (a post-hoc rebuild would collide with vLLM's
    duplicate-layer-name check). Model construction is single-threaded, so
    the temporary module-global substitution is safe here.
    """

    def __init__(
        self,
        config: PretrainedConfig,
        prefix: str,
        vllm_config: VllmConfig,
    ) -> None:
        if config is None:
            config = vllm_config.model_config.hf_config
        parent_mod = inspect.getmodule(VllmOpenPanguDecoderLayer)
        vllm_sink_cls = parent_mod.OpenPanguSinkAttention
        parent_mod.OpenPanguSinkAttention = NPUOpenPanguSinkAttention
        try:
            super().__init__(config, prefix, vllm_config)
        finally:
            parent_mod.OpenPanguSinkAttention = vllm_sink_cls


@support_torch_compile
class OpenPanguModel(VllmOpenPanguModel):
    fall_back_to_pt_during_load = False

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super(VllmOpenPanguModel, self).__init__()

        config = vllm_config.model_config.hf_config
        quant_config = vllm_config.quant_config
        eplb_config = vllm_config.parallel_config.eplb_config
        self.config = config
        self.num_redundant_experts = eplb_config.num_redundant_experts

        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        if get_pp_group().is_first_rank or (
            config.tie_word_embeddings and get_pp_group().is_last_rank
        ):
            self.embed_tokens = VocabParallelEmbedding(
                config.vocab_size,
                config.hidden_size,
                quant_config=quant_config,
                prefix=f"{prefix}.embed_tokens",
            )
        else:
            self.embed_tokens = PPMissingLayer()

        self.start_layer, self.end_layer, self.layers = make_layers(
            config.num_hidden_layers,
            lambda prefix: OpenPanguDecoderLayer(config, prefix, vllm_config),
            prefix=f"{prefix}.layers",
        )

        if get_pp_group().is_last_rank:
            self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        else:
            self.norm = PPMissingLayer()
        self.make_empty_intermediate_tensors = make_empty_intermediate_tensors_factory(
            ["hidden_states", "residual"], config.hidden_size
        )


class OpenPanguModelBase(nn.Module, SupportsPP, SupportsLoRA):
    packed_modules_mapping = {
        "qkv_proj": ["q_proj", "k_proj", "v_proj"],
        "gate_up_proj": ["gate_proj", "up_proj"],
    }

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        config = vllm_config.model_config.hf_config
        quant_config = vllm_config.quant_config
        self.config = config
        self.quant_config = quant_config

        self.fuse_qkv_a_proj = (
            hasattr(config, "q_lora_rank") and config.q_lora_rank is not None
        )
        if self.fuse_qkv_a_proj:
            self.packed_modules_mapping["fused_qkv_a_proj"] = [
                "q_a_proj",
                "kv_a_proj_with_mqa",
            ]

        self.model = OpenPanguModel(
            vllm_config=vllm_config, prefix=maybe_prefix(prefix, "model")
        )
        if get_pp_group().is_last_rank:
            self.lm_head = ParallelLMHead(
                config.vocab_size,
                config.hidden_size,
                quant_config=quant_config,
                prefix=maybe_prefix(prefix, "lm_head"),
            )
            if config.tie_word_embeddings:
                self.lm_head.weight = self.model.embed_tokens.weight
        else:
            self.lm_head = PPMissingLayer()
        self.logits_processor = LogitsProcessor(config.vocab_size)
        self.make_empty_intermediate_tensors = (
            self.model.make_empty_intermediate_tensors
        )

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model.embed_input_ids(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor | IntermediateTensors:
        hidden_states = self.model(
            input_ids, positions, intermediate_tensors, inputs_embeds
        )
        return hidden_states

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor | None:
        logits = self.logits_processor(self.lm_head, hidden_states)
        return logits

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        loader = AutoWeightsLoader(
            self,
            skip_prefixes=(["lm_head."] if self.config.tie_word_embeddings else None),
        )
        return loader.load_weights(weights)


class OpenPanguMoEModel(OpenPanguModelBase, MixtureOfExperts):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__(vllm_config=vllm_config, prefix=prefix)
        config = vllm_config.model_config.hf_config

        self.expert_weights = []
        self.num_moe_layers = config.num_hidden_layers - config.first_k_dense_replace
        self.num_expert_groups = 1

        self.moe_layers = []
        example_moe = None
        for layer in self.model.layers:
            if isinstance(layer, PPMissingLayer):
                continue

            assert isinstance(layer, OpenPanguDecoderLayer)
            if isinstance(layer.mlp, VllmOpenPanguMoE):
                example_moe = layer.mlp
                self.moe_layers.append(layer.mlp.experts)

        if example_moe is None:
            raise RuntimeError("No MOE layer found in model.layers.")

        self.num_logical_experts = example_moe.n_logical_experts
        self.num_physical_experts = example_moe.n_physical_experts
        self.num_local_physical_experts = example_moe.n_local_physical_experts
        self.n_routed_experts = example_moe.n_routed_experts
        self.n_shared_experts = example_moe.n_shared_experts
        self.num_redundant_experts = example_moe.n_redundant_experts

    def update_physical_experts_metadata(
        self,
        num_physical_experts: int,
        num_local_physical_experts: int,
    ) -> None:
        assert self.num_local_physical_experts == num_local_physical_experts
        self.num_physical_experts = num_physical_experts
        self.num_local_physical_experts = num_local_physical_experts
        self.num_redundant_experts = num_physical_experts - self.num_logical_experts
        for layer in self.model.layers:
            if isinstance(layer.mlp, VllmOpenPanguMoE):
                moe = layer.mlp
                moe.n_local_physical_experts = num_local_physical_experts
                moe.n_physical_experts = num_physical_experts
                moe.n_redundant_experts = self.num_redundant_experts
                moe.experts.update_expert_map()

    def set_eplb_state(
        self,
        expert_load_view: torch.Tensor,
        logical_to_physical_map: torch.Tensor,
        logical_replica_count: torch.Tensor,
    ) -> None:
        for layer_idx, layer in enumerate(self.moe_layers):
            self.expert_weights.append(layer.get_expert_weights())
            layer.set_eplb_state(
                moe_layer_idx=layer_idx,
                expert_load_view=expert_load_view,
                logical_to_physical_map=logical_to_physical_map,
                logical_replica_count=logical_replica_count,
            )


class PanguProMoEV2ForCausalLM(OpenPanguMoEModel):
    pass
