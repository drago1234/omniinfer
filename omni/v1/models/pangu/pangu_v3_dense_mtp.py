# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.

"""Dense Pangu-V3 MTP using the shared Pangu predictor forward methods."""

from collections.abc import Iterable

import torch
from torch import nn

from vllm.compilation.decorators import support_torch_compile
from vllm.config import VllmConfig
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.model_executor.models.utils import maybe_prefix

from omni_npu.v1.layers.vocab_parallel_embedding import NPUVocabParallelEmbedding

from .pangu_ultra_moe_mtp import (
    OpenPanguMTP,
    OpenPanguMultiTokenPredictor,
    OpenPanguMultiTokenPredictorLayer,
    SharedHead,
)
from .pangu_v3_dense import PanguV3DenseDecoderLayer


@support_torch_compile
class PanguV3DenseMultiTokenPredictorLayer(OpenPanguMultiTokenPredictorLayer):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str) -> None:
        nn.Module.__init__(self)
        config = vllm_config.speculative_config.draft_model_config.hf_config
        self.config = config
        self.enorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.hnorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.eh_proj = nn.Linear(config.hidden_size * 2, config.hidden_size, bias=False)
        self.shared_head = SharedHead(
            config=config, quant_config=vllm_config.quant_config,
            prefix=maybe_prefix(prefix, "shared_head"),
        )
        self.mtp_block = PanguV3DenseDecoderLayer(config, prefix, vllm_config)


class PanguV3DenseMultiTokenPredictor(OpenPanguMultiTokenPredictor):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        nn.Module.__init__(self)
        config = vllm_config.model_config.hf_config
        self.mtp_start_layer_idx = config.num_hidden_layers
        self.num_mtp_layers = config.num_nextn_predict_layers
        self.layers = nn.ModuleDict({
            str(idx): PanguV3DenseMultiTokenPredictorLayer(
                vllm_config=vllm_config, prefix=f"{prefix}.layers.{idx}",
            ) for idx in range(self.mtp_start_layer_idx, self.mtp_start_layer_idx + self.num_mtp_layers)
        })
        self.wrapped_layers = None
        self.embed_tokens = NPUVocabParallelEmbedding(config.vocab_size, config.hidden_size)
        self.logits_processor = LogitsProcessor(config.vocab_size)


class PanguV3DenseMTP(OpenPanguMTP):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        nn.Module.__init__(self)
        self.config = vllm_config.model_config.hf_config
        self.model = PanguV3DenseMultiTokenPredictor(
            vllm_config=vllm_config, prefix=maybe_prefix(prefix, "model"),
        )

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        stacked = [
            (".self_attn.qkv_proj.", ".self_attn.q_proj.", "q"),
            (".self_attn.qkv_proj.", ".self_attn.k_proj.", "k"),
            (".self_attn.qkv_proj.", ".self_attn.v_proj.", "v"),
            (".mlp.gate_up_proj.", ".mlp.gate_proj.", 0),
            (".mlp.gate_up_proj.", ".mlp.up_proj.", 1),
        ]
        params = dict(self.named_parameters())
        loaded = set()
        for name, weight in weights:
            layer = self.get_spec_layer(name)
            if layer is None:
                continue
            name = self._rewrite_spec_layer_name(layer, name)
            shard_args = ()
            for target, source, shard_id in stacked:
                if source in name:
                    name = name.replace(source, target)
                    shard_args = (shard_id,)
                    break
            param = params[name]
            getattr(param, "weight_loader", default_weight_loader)(param, weight, *shard_args)
            loaded.add(name)
        self.post_weight_load()
        return loaded
