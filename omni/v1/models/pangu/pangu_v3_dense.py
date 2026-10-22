# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.

"""High-throughput dense OpenPangu-V3 inference."""

from __future__ import annotations

from collections.abc import Iterable

import torch
import torch_npu
from torch import nn
from transformers import PretrainedConfig

from vllm.compilation.decorators import support_torch_compile
from vllm.config import CacheConfig, VllmConfig
from vllm.distributed import get_pp_group, get_tp_group
from vllm.distributed.parallel_state import GroupCoordinator
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import QKVParallelLinear, RowParallelLinear
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.rotary_embedding import get_rope
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.model_executor.layers.attention import Attention
from vllm.model_executor.models.utils import (
    extract_layer_index,
    is_pp_missing_parameter,
    make_empty_intermediate_tensors_factory,
    make_layers,
    maybe_prefix,
    PPMissingLayer,
    sequence_parallel_chunk,
)
from vllm.model_executor.utils import set_weight_attrs

from omni_npu import envs
from omni_npu.layers.engram import (
    EngramLayout,
    PanGuEngramConfig,
    PanGuEngramFusion,
    PanGuEngramPostNorm,
)
from omni_npu.layers.mhc.mhc_rl import NPUmHCRL
from omni_npu.model_config.config_loader.loader import model_extra_config
from omni_npu.v1.distributed.parallel_state_ext import get_engram_shared_group
from omni_npu.v1.layers.attention.gpt_oss_sink import GptOssSinkMixin
from omni_npu.v1.layers.attention.weight_utils import run_post_weight_load
from omni_npu.v1.layers.utils import get_cla_mapping
from omni_npu.v1.layers.vocab_parallel_embedding import (
    NPUParallelLMHead,
    NPUVocabParallelEmbedding,
)

from .pangu_ultra_moe import (
    get_spec_layer_idx_from_weight_name,
    OpenPanguDecoderLayer,
    OpenPanguMLP,
    OpenPanguModel,
    OpenPanguModelBase,
)


class NPUPanguV3GQAAttention(nn.Module, GptOssSinkMixin):
    """GQA with global TP projections and the shared Pangu-V3 decoder interface."""

    def __init__(
        self,
        config: PretrainedConfig, *,
        cache_config: CacheConfig | None = None,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        layer_idx = extract_layer_index(prefix)
        self.rope_dim = config.qk_rope_dim
        self.head_dim = config.head_dim
        rope_parameters = {
            "rope_type": "default",
            "rope_theta": config.rope_theta,
            "partial_rotary_factor": self.rope_dim / self.head_dim,
        }
        self.rotary_emb = get_rope(
            self.head_dim, max_position=config.max_position_embeddings,
            rope_parameters=rope_parameters, is_neox_style=True,
        )
        self.qkv_proj = QKVParallelLinear(
            config.hidden_size, self.head_dim, config.num_attention_heads,
            config.num_key_value_heads, bias=False,
            quant_config=quant_config, prefix=f"{prefix}.qkv_proj",
        )
        self.num_heads = self.qkv_proj.num_heads
        self.num_kv_heads = self.qkv_proj.num_kv_heads
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.o_proj = RowParallelLinear(
            config.num_attention_heads * self.head_dim, config.hidden_size,
            bias=False, quant_config=quant_config, prefix=f"{prefix}.o_proj",
        )
        self.sliding_window = self._get_sliding_window(config, layer_idx)
        source_idx = get_cla_mapping(config).get(layer_idx)
        source_name = (
            f"{prefix.rsplit('.layers.', 1)[0]}.layers.{source_idx}.self_attn.attn"
            if source_idx is not None else None
        )
        # Main-model layers share metadata for each window; MTP refreshes it
        # on every forward because its query and KV lengths can differ.
        should_recompute_attn_meta = (
            layer_idx >= config.num_hidden_layers
            or not any(
                self._get_sliding_window(config, i) == self.sliding_window
                for i in range(layer_idx)
            )
        )
        self.attn = Attention(
            self.num_heads, self.head_dim, self.head_dim ** -0.5,
            num_kv_heads=self.num_kv_heads, cache_config=cache_config,
            quant_config=quant_config, per_layer_sliding_window=self.sliding_window or 0,
            prefix=f"{prefix}.attn", kv_sharing_target_layer_name=source_name,
            should_recompute_attn_meta=should_recompute_attn_meta,
        )
        self.ena_sp = model_extra_config.parall_config.ena_seq_parallel
        self.num_local_heads = self.num_heads
        self.q_layernorm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_layernorm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.attention_gate = nn.Linear(config.hidden_size, self.q_size, bias=False)
        set_weight_attrs(self.attention_gate.weight, {"weight_loader": self._load_attention_gate_weight})
        self._init_gpt_oss_sink(config)
        self.attn.has_sink = self.gpt_oss_sink is not None
        self.attn.impl.attn_sinks = self.gpt_oss_sink
        # Attention.sliding_window defines the KV cache spec and eviction window.
        # Full-attention layers pass 0 above to bypass the global default, then
        # restore None here to select a FullAttentionSpec.
        self.attn.sliding_window = self.sliding_window
        if (
            self.sliding_window is not None
            and envs.OMNI_PD_ROLE in ("prefill", "decode")
            and not envs.OMNI_REUSE_PREFILLED_TOKENS
        ):
            # Keep one extra KV token for Decode to recompute the last prompt
            # token. The kernel's actual sliding window remains unchanged.
            self.attn.sliding_window += 1
        self.attn.impl.sliding_window = self.sliding_window

    @staticmethod
    def _get_sliding_window(config, layer_idx: int) -> int | None:
        return config.sliding_window_list[layer_idx] if layer_idx in config.swa_layers else None

    def _load_gpt_oss_sink_weight(self, param, loaded_weight: torch.Tensor) -> None:
        start = get_tp_group().rank_in_group * self.num_heads
        param.data.copy_(loaded_weight.narrow(0, start, self.num_heads).to(param))

    def _load_attention_gate_weight(self, param, loaded_weight: torch.Tensor) -> None:
        start = get_tp_group().rank_in_group * self.q_size
        param.data.copy_(loaded_weight.narrow(0, start, self.q_size).to(param))

    def _apply_rope(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        rope, nope = x.split((self.rope_dim, self.head_dim - self.rope_dim), dim=-1)
        shape = rope.shape
        rope = rope.contiguous().view(shape[0], -1, 1, self.rope_dim)
        cos = cos.reshape(-1, 1, 1, self.rope_dim)[:shape[0]]
        sin = sin.reshape(-1, 1, 1, self.rope_dim)[:shape[0]]
        rope = torch_npu.npu_rotary_mul(rope, cos, sin)
        return torch.cat((rope.view(shape), nope), dim=-1).contiguous()

    def forward(
        self, hidden_states: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor,
        topk_indices_buffer: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        if self.ena_sp:
            hidden_states = get_tp_group().all_gather(hidden_states, dim=0)
            # Embedding pads SP shards; attention uses the supplied positions.
            hidden_states = hidden_states[:cos.numel() // self.rope_dim]
        qkv, _ = self.qkv_proj(hidden_states)
        q, k, v = qkv.split((self.q_size, self.kv_size, self.kv_size), dim=-1)
        q = self.q_layernorm(q.reshape(-1, self.num_heads, self.head_dim))
        q = self._apply_rope(q, cos, sin)
        if self.attn.kv_sharing_target_layer_name is None:
            k = self.k_layernorm(k.reshape(-1, self.num_kv_heads, self.head_dim))
            k = self._apply_rope(k, cos, sin)
        else:
            k = v = None
        output = self.attn(q, k, v)
        output = output * torch.sigmoid(self.attention_gate(hidden_states))
        output, _ = self.o_proj(output)
        if self.ena_sp:
            output = sequence_parallel_chunk(output)
        return output, topk_indices_buffer


class PanguV3DenseDecoderLayer(OpenPanguDecoderLayer):
    """Build GQA/dense submodules; reuse the shared decoder forward methods."""

    def __init__(
        self, config: PretrainedConfig | None, prefix: str, vllm_config: VllmConfig,
        engram_layout: EngramLayout | None = None,
        engram_group: GroupCoordinator | None = None,
    ) -> None:
        nn.Module.__init__(self)
        config = config if config is not None else vllm_config.model_config.hf_config
        self.hidden_size = config.hidden_size
        self.layer_idx = extract_layer_index(prefix)
        self.engram = (
            PanGuEngramFusion(
                engram_layout, self.layer_idx, engram_group,
                use_sequence_parallel=model_extra_config.parall_config.ena_seq_parallel,
            ) if engram_layout is not None and self.layer_idx in engram_layout.layer_ids else None
        )
        self.post_self_attn_engram_layernorm = (
            PanGuEngramPostNorm(config.hidden_size, config.rms_norm_eps)
            if self.engram is not None and engram_layout.config.post_norm else None
        )
        self.self_attn = NPUPanguV3GQAAttention(
            config=config,
            cache_config=vllm_config.cache_config, quant_config=vllm_config.quant_config,
            prefix=f"{prefix}.self_attn",
        )
        self.mlp = OpenPanguMLP(
            config.hidden_size, config.intermediate_size, config.hidden_act,
            quant_config=vllm_config.quant_config, prefix=f"{prefix}.mlp",
        )
        self.sandwich_norm = getattr(config, "sandwich_norm", False)
        norm_names = ["input_layernorm", "post_attention_layernorm"]
        if self.sandwich_norm:
            norm_names += ["pre_mlp_layernorm", "post_mlp_layernorm"]
        for name in norm_names:
            setattr(self, name, RMSNorm(config.hidden_size, eps=config.rms_norm_eps))

        self.use_mhc = getattr(config, "use_mhc", False) and self.layer_idx < config.num_hidden_layers
        if self.use_mhc:
            for name in ("attn_mhc_module", "mlp_mhc_module"):
                setattr(self, name, NPUmHCRL(config=config, pre_only=False, prefix=f"{prefix}.{name}"))
            self.attn_mhc_task_key = f"{prefix}.self_attn.o_proj"
            self.mlp_mhc_task_key = f"{prefix}.mlp.down_proj"
        self.has_block_post_layernorm = self.layer_idx in (getattr(config, "block_post_layernorm_idx", None) or ())
        if self.has_block_post_layernorm:
            norm_size = config.hidden_size * (config.mhc_num_stream if self.use_mhc else 1)
            self.block_post_layernorm = RMSNorm(norm_size, eps=config.rms_norm_eps)
        self.use_mhc_fusion_op = bool(
            self.use_mhc
            and self.sandwich_norm
            and model_extra_config.operator_opt_config.use_mhc_fusion_op
        )


@support_torch_compile(
    dynamic_arg_dims={
        "input_ids": 0,
        "positions": -1,
        "intermediate_tensors": 0,
        "inputs_embeds": 0,
    }
)
class PanguV3DenseModel(OpenPanguModel):
    """Construct dense decoder layers and inherit the shared model forward."""

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        nn.Module.__init__(self)

        config = vllm_config.model_config.hf_config
        quant_config = vllm_config.quant_config
        self.config = config

        self.vocab_size = config.vocab_size
        engram_config = PanGuEngramConfig.from_vllm_config(vllm_config)
        self.engram_layout = EngramLayout(engram_config) if engram_config is not None else None
        engram_group = get_engram_shared_group() if self.engram_layout is not None else None

        if get_pp_group().is_first_rank or (
            config.tie_word_embeddings and get_pp_group().is_last_rank
        ):
            self.embed_tokens = NPUVocabParallelEmbedding(
                config.vocab_size,
                config.hidden_size,
                quant_config=quant_config,
                prefix=f"{prefix}.embed_tokens",
            )
        else:
            self.embed_tokens = PPMissingLayer()

        self.start_layer, self.end_layer, self.layers = make_layers(
            config.num_hidden_layers,
            lambda prefix: PanguV3DenseDecoderLayer(config, prefix, vllm_config, self.engram_layout, engram_group),
            prefix=f"{prefix}.layers",
        )

        self.norm = (
            RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
            if get_pp_group().is_last_rank else PPMissingLayer()
        )
        self.make_empty_intermediate_tensors = make_empty_intermediate_tensors_factory(
            ["hidden_states", "residual"], config.hidden_size
        )
        self.use_mhc = getattr(config, "use_mhc", False)
        self._init_engram_hash()
        if self.use_mhc:
            self.num_stream = getattr(config, "mhc_num_stream", 4)
            self.merge_mhc_module = NPUmHCRL(
                config=config, pre_only=True, prefix=f"{prefix}.merge_mhc_module",
            )

        self.use_mhc_fusion_op = bool(
            self.use_mhc
            and self.start_layer < self.end_layer
            and all(
                self.layers[i].use_mhc_fusion_op
                for i in range(self.start_layer, self.end_layer)
            )
        )


class PanguV3DenseForCausalLM(OpenPanguModelBase):
    """Dense Pangu-V3 with shared decoder/mHC/Engram support and GQA weight loading."""

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        config = vllm_config.model_config.hf_config
        nn.Module.__init__(self)
        quant_config = vllm_config.quant_config
        self.config = config
        self.quant_config = quant_config

        self.model = PanguV3DenseModel(
            vllm_config=vllm_config, prefix=maybe_prefix(prefix, "model")
        )
        if get_pp_group().is_last_rank:
            self.lm_head = NPUParallelLMHead(
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

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        stacked_params_mapping = [
            (".self_attn.qkv_proj.", ".self_attn.q_proj.", "q"),
            (".self_attn.qkv_proj.", ".self_attn.k_proj.", "k"),
            (".self_attn.qkv_proj.", ".self_attn.v_proj.", "v"),
            (".mlp.gate_up_proj.", ".mlp.gate_proj.", 0),
            (".mlp.gate_up_proj.", ".mlp.up_proj.", 1),
        ]
        params_dict = dict(self.named_parameters())
        loaded_params: set[str] = set()
        for name, loaded_weight in weights:
            if "rotary_emb.inv_freq" in name:
                continue
            if get_spec_layer_idx_from_weight_name(self.config, name) is not None:
                continue
            if self.config.tie_word_embeddings and name == "lm_head.weight":
                continue
            name = name.replace(".engram.embed.weight", ".engram.embed_tokens.weight")
            shard_args = ()
            for param_name, weight_name, mapped_shard_id in stacked_params_mapping:
                if weight_name in name:
                    name = name.replace(weight_name, param_name)
                    shard_args = (mapped_shard_id,)
                    break
            if is_pp_missing_parameter(name, self):
                continue
            param = params_dict[name]
            weight_loader = getattr(param, "weight_loader", default_weight_loader)
            weight_loader(param, loaded_weight, *shard_args)
            loaded_params.add(name)
        self.post_weight_load()
        return loaded_params

    def post_weight_load(self) -> None:
        run_post_weight_load(self)
