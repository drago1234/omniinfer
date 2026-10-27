# SPDX-License-Identifier: MIT
# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd. All Rights Reserved.


import contextlib
import math
from collections.abc import Callable

import torch
import torch_npu
from torch import nn
from transformers import DeepseekV2Config, DeepseekV3Config
from vllm.model_executor.layers.attention.kv_transfer_utils import (
    maybe_transfer_kv_layer,
)
from vllm.compilation.breakable_cudagraph import eager_break_during_capture
from vllm.config import CacheConfig, VllmConfig, get_current_vllm_config
from vllm.distributed import (
    get_dcp_group,
    get_tp_group,
)
from vllm.forward_context import get_forward_context
from vllm.logger import init_logger
from vllm.model_executor.layers.layernorm import LayerNorm, RMSNorm
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.rotary_embedding import get_rope
from vllm.model_executor.models.utils import extract_layer_index
from vllm.model_executor.utils import set_weight_attrs
from vllm.platforms import current_platform
from vllm.utils.torch_utils import direct_register_custom_op

logger = init_logger(__name__)

try:
    from vllm.model_executor.layers.mome import AggregateConv
except ImportError:
    logger.warning("AggregateConv has not being defined, skipping...")

from omni_npu.layers.attention.npu_sparse_attentions import DSAAttention
from omni_npu.attention import ops
from omni_npu.attention.backends.dsa import NPUDSAMetadata
from omni_npu.v1.layers.attention.gpt_oss_sink import GptOssSinkMixin
from omni_npu.attention.backends.utils import (
    CrossLayerSharedOp,
    DummyKVSPMaganer,
    DummySPManager,
    KVSPMaganer,
    SPManager,
    cache_fit_shape,
    conv_sp,
    enable_fia_metadata_sharing,
    get_batch_desc,
    get_cross_layer_shared_op,
    lazy_zero_like,
    sp_disabled,
)
from omni_npu.layers.mhc.mhc_deferred import (
    attention_mhc_deferred_fake,
    call_attention_mhc_deferred,
)
from omni_npu.layers.utils import SIDE_STREAM_NAME, named_stream
from omni_npu.model_config.config_loader.loader import model_extra_config
from omni_npu.plugin_decorators import attn_decorator
from omni_npu.v1.layers.attention.npu_mla import (
    MomeAttentionMixin,
    _a5_sink_attention_kwargs,
    _prefill_o_proj_sp_slice,
    NPUMLAAttention,
)
from omni_npu.v1.layers.attention.dsa_local_query import ZigzagQueryLayout
from omni_npu.v1.layers.attention.weight_utils import (
    install_q_b_split_loaders,
    release_q_b_proj_storage,
)
from omni_npu.v1.layers.linear import (
    ColumnParallelFlashCommLinear,
    ReplicatedFlashCommLinear,
    RowParallelFlashCommLinear,
    ShardedLinear,
)
from omni_npu.v1.layers.utils import yarn_get_mscale
from omni_npu.v1.utils import on_ascend950


# Enable via --kv-cache-dtype {fp8,hif8}_ds_mla.
SFA_C8_DTYPES = ("fp8_ds_mla", "hif8_ds_mla")
FP8_TILE_SIZE = 128

quant_lightning_indexer_v2_metadata: CrossLayerSharedOp | None = None


def _skip_topk_for_layer(config, layer_id: int) -> bool:
    index_topk_pattern = getattr(config, "index_topk_pattern", None)
    indexer_types = getattr(config, "indexer_types", None)
    if index_topk_pattern is not None:
        return (
            0 <= layer_id < len(index_topk_pattern)
            and index_topk_pattern[layer_id] == "S"
        )
    if indexer_types is not None:
        return (
            0 <= layer_id < len(indexer_types)
            and indexer_types[layer_id] == "shared"
        )
    index_topk_freq = getattr(config, "index_topk_freq", 1)
    index_skip_topk_offset = getattr(config, "index_skip_topk_offset", 2)
    return max(layer_id - index_skip_topk_offset + 1, 0) % index_topk_freq != 0


def is_sfa_c8_dtype(cache_dtype: str | None) -> bool:
    return cache_dtype in SFA_C8_DTYPES


def get_li_quant_mode(cache_dtype: str | None) -> str | None:
    if cache_dtype == "fp8_ds_mla":
        return "fp8"
    if cache_dtype == "hif8_ds_mla":
        return "hif8"
    if cache_dtype == "int8_ds_mla":
        raise ValueError("INT8 C8 is not supported by this DSA path; use fp8_ds_mla or hif8_ds_mla.")
    if cache_dtype == "li_int8_ds_mla":
        return "int8"
    return None


class NPUDSAAttention(NPUMLAAttention, DSAAttention):
    """A DSA cache page plus the NPU-derived absorb weights.

    DSAAttention sizes the page for the indexer key, which MLAAttention does not,
    and NPUMLAAttention reads W_UK_T/W_UV off a FlashComm kv_b_proj, which the
    vLLM base cannot. A DSA layer here needs both.
    """


class Indexer(torch.nn.Module):
    def __init__(
        self,
        vllm_config: VllmConfig,
        config: DeepseekV2Config | DeepseekV3Config,
        hidden_size: int,
        q_lora_rank: int,
        quant_config: QuantizationConfig | None,
        cache_config: CacheConfig | None,
        sink_len: int = 0,
        prefix: str = "",
        is_metadata_producer: bool = False,
    ):
        super().__init__()
        self.vllm_config = vllm_config
        self.config = config
        self.on_ascend950 = on_ascend950()
        self.topk_tokens = config.index_topk
        self.n_head = config.index_n_heads  # 64
        self.head_dim = config.index_head_dim  # 128
        self.rope_dim = config.qk_rope_head_dim  # 64
        self.q_lora_rank = q_lora_rank  # 1536
        self.is_metadata_producer = is_metadata_producer
        if model_extra_config.operator_opt_config.enable_precision_strong_consistency:
            self.softmax_scale = self.head_dim ** -0.5
            self.weights_scale = self.n_head ** -0.5
        # no tensor parallel, just replicated
        self.wq_b = ReplicatedFlashCommLinear(
            self.q_lora_rank,
            self.head_dim * self.n_head,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.wq_b",
        )
        self.wk = ReplicatedFlashCommLinear(
            hidden_size,
            self.head_dim,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.wk",
        )
        # Pangu uses RMSNorm and no bias; DeepSeek uses LayerNorm and a bias.
        if "pangu" in str(getattr(config, "model_type", "")).lower():
            self.k_norm = RMSNorm(self.head_dim, config.rms_norm_eps)
            self.weights_proj = ReplicatedFlashCommLinear(
                hidden_size, self.n_head, bias=False, quant_config=None, prefix=f"{prefix}.weights_proj"
            )
        else:
            self.k_norm = LayerNorm(self.head_dim, eps=1e-6)
            self.weights_proj = ReplicatedFlashCommLinear(
                hidden_size, self.n_head, quant_config=None, prefix=f"{prefix}.weights_proj"
            )
        self.sink_len = sink_len
        self.wi_stream = None
        self.ki_stream = None
        self.use_rope_fusion_op = (
            model_extra_config.operator_opt_config.use_rope_fusion_op
            and not model_extra_config.operator_opt_config.enable_precision_strong_consistency
        )
        self.cache_dtype = getattr(cache_config, "cache_dtype", None) if cache_config else None
        self.li_quant_mode = get_li_quant_mode(self.cache_dtype)
        if self.li_quant_mode and self.head_dim != 128:
            raise ValueError("Quantized DSA indexer caches require index_head_dim=128.")
        if self.li_quant_mode and not self.on_ascend950:
            raise ValueError(f"{self.cache_dtype} LI quant is only enabled for Ascend950/A5.")
        if self.li_quant_mode and not model_extra_config.operator_opt_config.use_noncontiguous_kv:
            raise ValueError(f"{self.cache_dtype} requires use_noncontiguous_kv=True.")
        self.use_qli_v2 = bool(
            self.cache_dtype == "hif8_ds_mla"
            and self.on_ascend950
            and getattr(
                model_extra_config.operator_opt_config,
                "use_quant_lightning_indexer_v2",
                False,
            )
        )
        if self.use_qli_v2:
            self._init_qli_v2_ops()

    def _init_qli_v2_ops(self) -> None:
        global quant_lightning_indexer_v2_metadata
        import cann_ops_transformer.ops.attention.indexer_quant_cache.indexer_quant_cache  # noqa: F401
        import cann_ops_transformer.ops.attention.quant_lightning_indexer_v2.quant_lightning_indexer  # noqa: F401

        if quant_lightning_indexer_v2_metadata is None:
            callers = tuple(
                (caller, 3)
                for caller in ("decode", "prefill")
            )
            quant_lightning_indexer_v2_metadata = CrossLayerSharedOp(
                op=torch.ops.cann_ops_transformer.quant_lightning_indexer_metadata,
                shape=(1024,),
                dtype=torch.int32,
                callers=callers,
            )
        self.qli_v2_metadata_op = quant_lightning_indexer_v2_metadata

    def _apply_quant_lightning_indexer_v2(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        weights: torch.Tensor,
        query_scale: torch.Tensor,
        key_scale: torch.Tensor,
        q_cumlens: torch.Tensor,
        kv_lens: torch.Tensor,
        block_table: torch.Tensor,
        metadata_caller: str,
    ) -> torch.Tensor:
        quant_mode = 3
        cu_seqlens_q = torch.cat((q_cumlens.new_zeros(1), q_cumlens))
        common_kwargs = {
            "cu_seqlens_q": cu_seqlens_q,
            "seqused_k": kv_lens,
            "layout_q": "TND",
            "layout_k": "PA_BBND",
            "mask_mode": 3,
            "cmp_ratio": 1,
        }
        metadata = self.qli_v2_metadata_op(
            {
                "num_heads_q": self.n_head,
                "num_heads_k": 1,
                "head_dim": self.head_dim,
                "topk": self.topk_tokens,
                "quant_mode": quant_mode,
                **common_kwargs,
            },
            self.is_metadata_producer,
            (metadata_caller, quant_mode),
        )
        out = torch.ops.cann_ops_transformer.quant_lightning_indexer(
            query,
            key,
            weights,
            query_scale,
            key_scale,
            self.topk_tokens,
            quant_mode,
            block_table=block_table,
            metadata=metadata,
            **common_kwargs,
        )
        return out[0] if isinstance(out, (tuple, list)) else out

    def _apply_rope(
        self,
        x: torch.Tensor,  # TND
        cos: torch.Tensor,  # BNSD
        sin: torch.Tensor,  # BNSD
    ) -> torch.Tensor:  # TND
        assert x.dim() == 3  # TND
        R, D, N = self.rope_dim, self.head_dim, x.size(1)
        if self.use_rope_fusion_op:
            shape = x.shape
            x, _ = torch_npu.npu_apply_rotary_pos_emb(
                x.view(-1, 1, N, D),
                x.view(-1, 1, N, D),
                cos.view(-1, 1, 1, R),
                sin.view(-1, 1, 1, R),
                layout="BSND",
                rotary_mode="half",
            )
            return x.view(shape)

        pe, nope = torch.split(x, [R, D - R], dim=-1)
        pe = pe.view(-1, N, 1, R)  # BNSD
        if getattr(self.config, "indexer_rope_interleave", False):
            pe = torch_npu.npu_interleave_rope(pe, cos, sin)
        else:
            pe = torch_npu.npu_rotary_mul(pe, cos, sin)
        pe = pe.view(-1, N, R)  # TND
        return torch.cat([pe, nope], dim=-1)  # TND

    def _li_prolog_ext(
        self,
        wx: torch.Tensor, # TD
        qr: torch.Tensor | dict[str, torch.Tensor], # TD
        kx: torch.Tensor, # TD
        q_cos_sin: tuple[torch.Tensor, torch.Tensor], # BNSD, BNSD
        k_cos_sin: tuple[torch.Tensor, torch.Tensor], # BNSD, BNSD
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        N, D = self.n_head, self.head_dim

        if self.ki_stream is not None:
            wi_ctx = torch.npu.stream(self.wi_stream)
            ki_ctx = torch.npu.stream(self.ki_stream)
            cur_stream = torch.npu.current_stream()
        else:
            wi_ctx = contextlib.nullcontext()
            ki_ctx = contextlib.nullcontext()

        qi = self.wq_b(qr)[0].view(-1, N, D)  # TND
        qi = self._apply_rope(qi, *q_cos_sin) # TND
        with wi_ctx:
            wi = self.weights_proj(wx)[0]         # TN
            if model_extra_config.operator_opt_config.enable_precision_strong_consistency:
                wi = wi * self.weights_scale * self.softmax_scale
        with ki_ctx:
            ki = self.wk(kx)[0]                   # TD
            ki = self.k_norm(ki).view(-1, 1, D)   # T1D
            ki = self._apply_rope(ki, *k_cos_sin) # T1D
        if self.ki_stream is not None:
            cur_stream.wait_stream(self.wi_stream)
            cur_stream.wait_stream(self.ki_stream)

        return wi, qi, ki # TN, TND, T1D

    def _li_prolog(
        self,
        x: torch.Tensor,
        qr: torch.Tensor | dict[str, torch.Tensor],
        cos: torch.Tensor,
        sin: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]: # -> wi, qi, ki
        return self._li_prolog_ext(x, qr, x, (cos, sin), (cos, sin))

    def _update_cache(
        self,
        ki: torch.Tensor,  # TD or T1D
        slots_2d: torch.Tensor | NPUDSAMetadata | None,
        ki_cache: torch.Tensor | None,  # [*, pg, D] or [*, pg, 1, D]
        ki_scale_cache: torch.Tensor | None = None,
    ):
        metadata = slots_2d
        slots_2d = getattr(metadata, "slot_mapping_2d", metadata)
        # An empty update writes nothing, so it must not depend on the cache
        # being wired yet: dummy_run, ranks that own no local tokens under KVSP
        # and the DummyKVSPMaganer fallback (local_slots_2d is None there) all
        # land here with ki.size(0) == 0.
        if ki.size(0) == 0:
            return
        # dummy_run leaves both sides absent; a non-empty update needs both.
        if slots_2d is None or ki_cache is None:
            assert slots_2d is None and ki_cache is None, (
                "slots_2d and ki_cache must both be None"
            )
            return
        assert slots_2d.dim() == 2 and ki.dim() in [2, 3] and ki_cache.dim() in [3, 4]
        assert slots_2d.size(0) == ki.size(0)

        # Mirror npu_pangu Indexer quant cache update for quantLI.
        li_quant_mode = getattr(self, "li_quant_mode", None)
        if li_quant_mode:
            assert ki_scale_cache is not None
            ki_flat = ki.view(-1, ki.size(-1))
            if getattr(self, "use_qli_v2", False):
                slots = getattr(metadata, "slot_mapping", None)
                assert slots is not None and slots.size(0) == ki_flat.size(0)
                key_cache = ki_cache if ki_cache.dim() == 4 else ki_cache.unsqueeze(2)
                scale_cache = (
                    ki_scale_cache
                    if ki_scale_cache.dim() == 4
                    else ki_scale_cache.reshape(
                        ki_scale_cache.shape[0], ki_scale_cache.shape[1], 1, -1
                    )
                )
                torch.ops.cann_ops_transformer.indexer_quant_cache(
                    key_cache,
                    scale_cache,
                    ki_flat,
                    slots.to(torch.int32),
                    quant_mode="mxfp8",
                    round_scale=True,
                )
                return
            elif li_quant_mode == "int8":
                ki_quant, ki_scale = torch_npu.npu_dynamic_quant(ki_flat)
                ki_scale = ki_scale.to(torch.float16).view(-1, 1)
                torch_npu.npu_scatter_nd_update_(ki_cache, slots_2d, ki_quant)
            elif li_quant_mode == "fp8":
                ki_quant, ki_scale = torch_npu.npu_dynamic_quant(
                    ki_flat, dst_type=torch.float8_e4m3fn
                )
                ki_scale = ki_scale.to(torch.float32).view(-1, 1)
                torch_npu.npu_scatter_nd_update_(ki_cache, slots_2d, ki_quant)
            elif li_quant_mode == "hif8":
                ki_quant, ki_scale = torch_npu.npu_dynamic_quant(
                    ki_flat,
                    dst_type=torch_npu.hifloat8,
                    dst_type_max=15.0,
                )
                ki_scale = ki_scale.to(torch.float32).view(-1, 1)
                torch_npu.npu_scatter_nd_update_(
                    ki_cache.view(torch.int8),
                    slots_2d,
                    ki_quant.view(torch.int8),
                )
            else:
                raise ValueError(f"Unsupported LI quant mode: {li_quant_mode}")
            torch_npu.npu_scatter_nd_update_(ki_scale_cache, slots_2d, ki_scale)
            return

        if self.on_ascend950:
            torch_npu.npu_scatter_nd_update_(
                ki_cache,
                slots_2d,
                ki.view(-1, ki.size(-1)),
            )
        else:
            torch.ops.custom.npu_ai_infra_scatter_block_update_(
                cache_fit_shape(ki_cache, "3D"),
                slots_2d,
                ki.view(-1, ki.size(-1)),
            )

    def _apply_lightning_indexer(
        self,
        wi: torch.Tensor,  # [T, N]
        qi: torch.Tensor,  # [T, N, D]
        ki_cache: torch.Tensor,  # [*, pg, 1, D]
        q_cumlens: torch.Tensor = None,  # int32 [B]
        kv_lens: torch.Tensor = None,  # int32 [B]
        block_table: torch.Tensor = None,  # int32 [B, *]
        ki_scale_cache: torch.Tensor | None = None,
        metadata_caller: str = "prefill",
    ) -> torch.Tensor:  # int32 [T, 1, K]
        if None in [q_cumlens, kv_lens, block_table]:
            return None  # dummy_run

        if model_extra_config.operator_opt_config.use_noncontiguous_kv:
            key = ki_cache.unsqueeze(2) if ki_cache.dim() == 3 else ki_cache
            li_quant_mode = getattr(self, "li_quant_mode", None)
            if li_quant_mode:
                # Mirror npu_pangu quantLI path.
                assert ki_scale_cache is not None
                query_dtype = None
                key_dtype = None
                if getattr(self, "use_qli_v2", False):
                    qi_shape = qi.shape
                    qi_quant, qi_scale = torch_npu.npu_dynamic_mx_quant(
                        qi.reshape(-1, qi_shape[-1]),
                        dst_type=torch.float8_e4m3fn,
                        scale_alg=1,
                    )
                    qi_quant = qi_quant.reshape(qi_shape)
                    qi_scale = qi_scale.reshape(
                        *qi_shape[:-1], qi_shape[-1] // 64, 2
                    )
                    return self._apply_quant_lightning_indexer_v2(
                        qi_quant,
                        key,
                        wi.to(torch.float32),
                        qi_scale.view(torch.float8_e8m0fnu),
                        ki_scale_cache.view(torch.float8_e8m0fnu).reshape(
                            *ki_scale_cache.shape[:-1], self.head_dim // 64, 2
                        ),
                        q_cumlens,
                        kv_lens,
                        block_table,
                        metadata_caller,
                    )
                if li_quant_mode == "int8":
                    qi_quant, qi_scale = torch_npu.npu_dynamic_quant(qi)
                    weights = wi.to(torch.float16)
                    qi_scale = qi_scale.to(torch.float16)
                elif li_quant_mode == "fp8":
                    qi_quant, qi_scale = torch_npu.npu_dynamic_quant(
                        qi, dst_type=torch.float8_e4m3fn
                    )
                    weights = wi.to(torch.bfloat16)
                    qi_scale = qi_scale.to(torch.float32)
                elif li_quant_mode == "hif8":
                    qi_quant, qi_scale = torch_npu.npu_dynamic_quant(
                        qi,
                        dst_type=torch_npu.hifloat8,
                        dst_type_max=15.0,
                    )
                    weights = wi.to(torch.bfloat16)
                    qi_scale = qi_scale.to(torch.float32)
                    query_dtype = torch_npu.hifloat8
                    key_dtype = torch_npu.hifloat8
                else:
                    raise ValueError(f"Unsupported LI quant mode: {li_quant_mode}")

                kwargs = dict(
                    query=qi_quant,
                    key=key,
                    weights=weights,
                    query_dequant_scale=qi_scale,
                    key_dequant_scale=ki_scale_cache,
                    actual_seq_lengths_query=q_cumlens,
                    actual_seq_lengths_key=kv_lens,
                    block_table=block_table,
                    query_quant_mode=0,
                    key_quant_mode=0,
                    layout_query="TND",
                    layout_key="PA_BSND",
                    sparse_count=self.topk_tokens,
                    sparse_mode=3,
                )
                if query_dtype is not None:
                    kwargs["query_dtype"] = query_dtype
                    kwargs["key_dtype"] = key_dtype
                # Ascend950 built-in QuantLightningIndexer covers int8/fp8/hif8.
                # AiInfraQuantLightningIndexer has no binary in this CANN build.
                if not hasattr(torch_npu, "npu_quant_lightning_indexer"):
                    raise RuntimeError(
                        "torch_npu.npu_quant_lightning_indexer is required "
                        f"for {self.cache_dtype}."
                    )
                out = torch_npu.npu_quant_lightning_indexer(**kwargs)
                return out[0] if isinstance(out, (tuple, list)) else out

            kwargs = dict(
                query=qi,
                key=key,
                weights=wi,
                actual_seq_lengths_query=q_cumlens,
                actual_seq_lengths_key=kv_lens,
                block_table=block_table,
                layout_key="PA_BSND",
                layout_query="TND",
                sparse_count=self.topk_tokens,
                sparse_mode=3,
            )
            if self.on_ascend950:
                return torch_npu.npu_lightning_indexer(**kwargs)[0]

            return torch.ops.custom.npu_lightning_indexer_enhance(
                **kwargs,
                sparse_block_size=1,
                sparse_block_mode=False,
            )[0]

        if self.sink_len > 0:
            num_sink_blocks = self.sink_len // self.vllm_config.cache_config.block_size
            block_table = block_table[:, num_sink_blocks:]

        return torch_npu.npu_lightning_indexer(
            query=qi,
            key=ki_cache,
            weights=wi,
            actual_seq_lengths_query=q_cumlens,
            actual_seq_lengths_key=kv_lens - self.sink_len,
            block_table=block_table,
            layout_key="PA_BSND",
            layout_query="TND",
            sparse_count=self.topk_tokens,
            sparse_mode=3,
        )[0]

    def forward(
        self,
        x: torch.Tensor,  # TD
        qr: torch.Tensor,  # TD
        cos: torch.Tensor,  # BNSD
        sin: torch.Tensor,  # BNSD
        attn_metadata: NPUDSAMetadata,
        ki_cache: torch.Tensor,  # [*, pg, 1, D]
        ki_scale_cache: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        wi, qi, ki = self._li_prolog(x, qr, cos, sin)
        if None in [attn_metadata, ki_cache]:
            return None, ki  # dummy_run

        self._update_cache(ki, attn_metadata, ki_cache, ki_scale_cache)
        tok_idx = self._apply_lightning_indexer(
            wi,
            qi,
            ki_cache,
            q_cumlens=attn_metadata.query_cumlens.to(torch.int32),
            kv_lens=attn_metadata.seq_lens.to(torch.int32),
            block_table=attn_metadata.block_table,
            ki_scale_cache=ki_scale_cache,
        )
        return tok_idx, ki


# Core budgets for the ``limit_core_num(Cube, Vector)`` scopes in
# ``_forward_decode_multistream``: the first value is the cube-core budget and
# the second the vector-core budget granted to the scope. Tuning these trades
# core utilization between the two streams; keep each pair aligned with the
# workload of its scope.
_MULTISTREAM_PROLOG_CORE_NUM = (16, 24)
# Scope covers the latent-KV projection as well as the indexer key/query
# projection and its cache update.
_MULTISTREAM_KV_INDEXER_CORE_NUM = (8, 24)
_MULTISTREAM_INDEXER_CORE_NUM = (20, 40)
_MULTISTREAM_Q_ABSORB_CORE_NUM = (12, 24)


class NPUDeepseekSparseAttention(MomeAttentionMixin, GptOssSinkMixin, torch.nn.Module):
    def __init__(
        self,
        vllm_config: VllmConfig,
        config: DeepseekV2Config | DeepseekV3Config,
        hidden_size: int,
        num_heads: int,
        qk_nope_head_dim: int,
        qk_rope_head_dim: int,
        v_head_dim: int,
        q_lora_rank: int | None,
        kv_lora_rank: int,
        max_position_embeddings: int = 8192,
        cache_config: CacheConfig | None = None,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.qk_nope_head_dim = qk_nope_head_dim
        self.qk_rope_head_dim = qk_rope_head_dim
        self.qk_head_dim = qk_nope_head_dim + qk_rope_head_dim
        self.v_head_dim = v_head_dim
        self.q_lora_rank = q_lora_rank
        self.kv_lora_rank = kv_lora_rank
        self.dtype = model_extra_config.dtype
        self.default_cfg = {"device": current_platform.device_type, "dtype": self.dtype}

        self.num_heads = num_heads
        self.tp_size = get_tp_group().world_size

        self.scaling = 1 / math.sqrt(self.qk_head_dim)
        self.max_position_embeddings = max_position_embeddings
        self.prefix = prefix
        self.layer_idx = extract_layer_index(prefix)
        self.quant_symbol = quant_config is not None
        self.is_pd_disagg = vllm_config.kv_transfer_config is not None

        self.kv_nz = model_extra_config.operator_opt_config.kv_nz
        self.noncontiguous_kv = model_extra_config.operator_opt_config.use_noncontiguous_kv
        self.enable_decode_multi_stream = model_extra_config.operator_opt_config.enable_multi_stream
        self.split_q_up_in_multistream = (
            self.enable_decode_multi_stream
            and model_extra_config.operator_opt_config.split_q_up_in_multistream
            and self.q_lora_rank is not None
        )
        self.use_mome_inplace_update = model_extra_config.operator_opt_config.use_mome_inplace_update
        self.on_ascend950 = on_ascend950()
        # C8 enable: --kv-cache-dtype {fp8,hif8}_ds_mla.
        self.cache_dtype = getattr(cache_config, "cache_dtype", None) if cache_config else None
        self.is_sfa_c8 = is_sfa_c8_dtype(self.cache_dtype)
        if self.is_sfa_c8 and self.dtype != torch.bfloat16:
            raise ValueError("C8 DSA attention requires BF16 activations and rotary cache data.")
        if self.is_sfa_c8 and (self.kv_lora_rank != 512 or self.qk_rope_head_dim != 64):
            raise ValueError("C8 DSA caches require kv_lora_rank=512 and qk_rope_head_dim=64.")
        # packed: nope_quant + rope_bf16 + fp32 scale per tile
        self.fp8_packed_kv_dim = (
            self.kv_lora_rank
            + self.qk_rope_head_dim * 2
            + self.kv_lora_rank // FP8_TILE_SIZE * 4
        )
        self.li_quant_mode = get_li_quant_mode(self.cache_dtype)
        if (self.is_sfa_c8 or self.li_quant_mode) and not self.noncontiguous_kv:
            raise ValueError(f"{self.cache_dtype} KV cache requires use_noncontiguous_kv=True.")
        if (self.is_sfa_c8 or self.li_quant_mode) and not self.on_ascend950:
            raise ValueError(f"{self.cache_dtype} KV cache is only enabled for Ascend950/A5.")
        if (
            (self.is_sfa_c8 or self.li_quant_mode)
            and model_extra_config.operator_opt_config.enable_precision_strong_consistency
        ):
            raise ValueError(
                f"{self.cache_dtype} does not support precision strong consistency."
            )
        self.use_mlaprolog = model_extra_config.operator_opt_config.enable_mlaprolog
        self.ena_sp = model_extra_config.parall_config.ena_seq_parallel
        self.ena_cp = model_extra_config.parall_config.ena_context_parallel
        self.enable_prefill_graph_token_parallel = (
            model_extra_config.operator_opt_config.enable_prefill_graph_token_parallel
        )
        if self.enable_prefill_graph_token_parallel:
            supported_layout = self.on_ascend950 and self.ena_sp and self.ena_cp
            supported_projection = self.tp_size > 1 and self.q_lora_rank is not None
            if not (supported_layout and supported_projection):
                raise ValueError("DSA local query requires A5 TP>1, q_lora_rank and external SPCP enabled")
        self.ena_kvsp = bool(get_dcp_group().world_size > 1)
        self.sharded_o_proj = model_extra_config.parall_config.sharded_o_proj
        self.num_speculative_tokens = (
            0 if not vllm_config.speculative_config else vllm_config.speculative_config.num_speculative_tokens
        )

        if self.q_lora_rank is not None:
            self.q_a_proj = ReplicatedFlashCommLinear(
                self.hidden_size,
                self.q_lora_rank,
                bias=False,
                quant_config=quant_config,
                prefix=f"{prefix}.q_a_proj",
            )
            self.q_a_layernorm = RMSNorm(self.q_lora_rank, eps=config.rms_norm_eps)
            self.q_b_proj = ColumnParallelFlashCommLinear(
                self.q_lora_rank,
                self.num_heads * self.qk_head_dim,
                bias=False,
                quant_config=quant_config,
                prefix=f"{prefix}.q_b_proj",
                disable_tp=self.ena_cp,
            )
            if self.split_q_up_in_multistream:
                self.q_b_nope_proj = ColumnParallelFlashCommLinear(
                    self.q_lora_rank,
                    self.num_heads * self.qk_nope_head_dim,
                    bias=False,
                    quant_config=quant_config,
                    prefix=f"{prefix}.q_b_nope_proj",
                    disable_tp=self.ena_cp,
                )
                self.q_b_pe_proj = ColumnParallelFlashCommLinear(
                    self.q_lora_rank,
                    self.num_heads * self.qk_rope_head_dim,
                    bias=False,
                    quant_config=quant_config,
                    prefix=f"{prefix}.q_b_pe_proj",
                    disable_tp=self.ena_cp,
                )
                install_q_b_split_loaders(
                    self.q_b_proj,
                    self.q_b_nope_proj,
                    self.q_b_pe_proj,
                    self.qk_head_dim,
                    self.qk_nope_head_dim,
                )
        else:
            self.q_proj = ColumnParallelFlashCommLinear(
                self.hidden_size,
                self.num_heads * self.qk_head_dim,
                bias=False,
                quant_config=quant_config,
                prefix=f"{prefix}.q_proj",
                disable_tp=self.ena_cp,
            )

        self.kv_a_proj_with_mqa = ReplicatedFlashCommLinear(
            self.hidden_size,
            self.kv_lora_rank + self.qk_rope_head_dim,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.kv_a_proj_with_mqa",
        )
        self.kv_a_layernorm = RMSNorm(self.kv_lora_rank, eps=config.rms_norm_eps, dtype=self.dtype)
        self.kv_b_proj = ColumnParallelFlashCommLinear(
            self.kv_lora_rank,
            self.num_heads * (self.qk_nope_head_dim + self.v_head_dim),
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.kv_b_proj",
            disable_tp=self.ena_cp,
        )

        if self.sharded_o_proj:
            self.o_proj = ShardedLinear(
                self.num_heads * self.v_head_dim,
                self.hidden_size,
                bias=False,
                shard_group=get_tp_group(),
                quant_config=quant_config,
                prefix=f"{prefix}.o_proj",
            )
        else:
            self.o_proj = RowParallelFlashCommLinear(
                self.num_heads * self.v_head_dim,
                self.hidden_size,
                bias=False,
                quant_config=quant_config,
                prefix=f"{prefix}.o_proj",
            )

        assert self.q_b_proj.tp_size == self.kv_b_proj.tp_size
        self.num_local_heads = self.num_heads // self.q_b_proj.tp_size

        self.rope_interleaved = getattr(
            config,
            "rope_interleaved",
            getattr(config, "rope_interleave", True),
        )
        if config.rope_parameters["rope_type"] != "default":
            config.rope_parameters["rope_type"] = (
                "deepseek_yarn" if config.rope_parameters.get("apply_yarn_scaling", True) else "deepseek_llama_scaling"
            )
            is_neox_style = False  # Deepseek V3.2
        else:
            is_neox_style = True  # GLM 5 (for generating neox style sin and cos caches.
            # gptj style will be applied by the npu_interleave_rope operator)

        rope_scaling = getattr(config, "rope_scaling", None)
        is_mrope = rope_scaling is not None and rope_scaling.get("mrope_section") is not None
        if is_mrope:
            cache_layer = config.num_hidden_layers
            is_mtp_layer = getattr(config, "is_mtp_layer", False)
            if is_mtp_layer:
                cache_layer = config.num_nextn_predict_layers
            from vllm.model_executor.layers.rotary_embedding import get_rope_wrapper

            self.rotary_emb = get_rope_wrapper(
                qk_rope_head_dim,
                max_position=max_position_embeddings,
                rotary_dim=qk_rope_head_dim,
                base=config.rope_parameters["rope_theta"],
                rope_scaling=rope_scaling,
                num_hidden_layers_cache=cache_layer,
            )
        else:
            self.rotary_emb = get_rope(
                qk_rope_head_dim,
                max_position=max_position_embeddings,
                rope_parameters=config.rope_parameters,
                is_neox_style=is_neox_style,
            )

        if config.rope_parameters["rope_type"] != "default" and config.rope_parameters["rope_type"] == "deepseek_yarn":
            mscale_all_dim = config.rope_parameters.get("mscale_all_dim", False)
            scaling_factor = config.rope_parameters["factor"]
            mscale = yarn_get_mscale(scaling_factor, float(mscale_all_dim))
            self.scaling = self.scaling * mscale * mscale

        self.use_mome = getattr(config, "use_mome", False)
        self.enable_mome_sp = getattr(
            model_extra_config.operator_opt_config,
            "enable_mome_sp",
            False,
        )
        self.merge_q_kv_conv = model_extra_config.operator_opt_config.merge_q_kv_conv
        self.param_sink_number = getattr(config, "param_sink_number", 0)
        self.index_topk = config.index_topk

        # IndexCache config
        layer_id = extract_layer_index(prefix)
        self.skip_topk = _skip_topk_for_layer(config, layer_id)
        if not self.skip_topk:
            dsa_layers = tuple(
                getattr(config, "dsa_layers", ())
                or range(config.num_hidden_layers)
            )
            indexer_layers = tuple(
                idx for idx in dsa_layers if not _skip_topk_for_layer(config, idx)
            )
            first_indexer_layer = min(indexer_layers, default=layer_id)
            is_metadata_producer = (
                layer_id >= config.num_hidden_layers
                or layer_id == first_indexer_layer
            )
            self.indexer = Indexer(
                vllm_config,
                config,
                hidden_size,
                q_lora_rank,
                quant_config,
                cache_config,
                self.param_sink_number,
                f"{prefix}.indexer",
                is_metadata_producer=is_metadata_producer,
            )
        else:
            self.indexer = None

        if self.use_mome:
            if self.noncontiguous_kv:
                num_extra_token = 1 if self.is_pd_disagg else 0
                fake_num_spec_tokens = max(self.num_speculative_tokens, num_extra_token)
                self._build_mome_conv(
                    fake_num_spec_tokens, config, vllm_config, prefix
                )
            else:
                self.qa_conv = AggregateConv(
                    self.q_lora_rank, config, vllm_config, output_parallel=False, attn_prefix=f"{prefix}.attn"
                )
                self.compresskv_conv = AggregateConv(
                    self.kv_lora_rank, config, vllm_config, output_parallel=False, attn_prefix=f"{prefix}.attn"
                )
                self.o_conv = AggregateConv(
                    self.num_local_heads * self.v_head_dim,
                    config,
                    vllm_config,
                    output_parallel=True,
                    attn_prefix=f"{prefix}.attn",
                )

        if self.param_sink_number == 0:
            self.attn = NPUDSAAttention(
                **self._inner_mla_attn_kwargs(cache_config, quant_config, prefix),
                indexer_head_dim=config.index_head_dim,
                cache_dtype_str=self.cache_dtype,
                block_size=cache_config.block_size * (2 if self.is_sfa_c8 else 1) if cache_config else None,
                page_size_padded=getattr(cache_config, "mamba_page_size_padded", None),
                indexer=self.indexer,
            )
        else:
            from vllm.model_executor.layers.attention.static_sink_attention import (
                StaticSinkMLAAttention,
            )

            self.attn = StaticSinkMLAAttention(
                **self._static_sink_attn_base_kwargs(
                    cache_config, quant_config, prefix
                ),
                use_sparse=True,
                indexer=self.indexer,
            )
            self._register_sink_params(config)

        if self.noncontiguous_kv:
            self.dummy_value_cache = torch.zeros((1, cache_config.block_size, 1, self.kv_lora_rank), **self.default_cfg)
            if self.is_sfa_c8:
                dsa_c8_block_size = cache_config.block_size * 2
                self.dummy_value_cache_quant = torch.zeros(
                    (1, dsa_c8_block_size, 1, self.fp8_packed_kv_dim),
                    device=current_platform.device_type,
                    dtype=torch.uint8,
                )

        self._init_gpt_oss_sink(config)
        if self.is_sfa_c8 and self.use_gpt_oss_sink:
            raise ValueError("C8 DSA attention does not support GPT-OSS sink logits.")

        self.post_weight_load()  # To enable dummy run with out weight

        compilation_config = get_current_vllm_config().compilation_config
        if prefix in compilation_config.static_forward_context:
            raise ValueError(f"Duplicate layer name: {prefix}")
        compilation_config.static_forward_context[prefix] = self
        self._init_fia_metadata_sharing(config)
        self.side_stream = (
            named_stream(SIDE_STREAM_NAME)
            if self.enable_decode_multi_stream
            else None
        )
        self.pre_epilog_callback = None

        self.enable_a5_multistream = (
            self.enable_decode_multi_stream
            and self.on_ascend950
            and self.dtype == torch.bfloat16
            and self.q_lora_rank is not None
            and not self.ena_kvsp
        )
        self.li_stream = (
            named_stream("dsa_li_stream")
            if self.enable_a5_multistream and not self.skip_topk
            else None
        )

        self.shared_indexer_kv_stream = (
            named_stream("dsa_shared_kv_stream")
            if self.enable_a5_multistream and self.skip_topk
            else None
        )

        if not self.skip_topk and model_extra_config.operator_opt_config.li_prolog_multi_stream:
            self.wi_stream = named_stream("wi_stream")
            self.ki_stream = named_stream("ki_stream")
            self.indexer.wi_stream = self.wi_stream
            self.indexer.ki_stream = self.ki_stream

        if model_extra_config.operator_opt_config.enable_precision_strong_consistency and self.on_ascend950:
            self.atten_mask = torch.triu(
                torch.ones(2048, 2048, dtype=torch.bool), diagonal=1
            ).to(current_platform.device_type)

    def post_weight_load(self):
        if self.param_sink_number > 0:
            param_sink_compressed_kv = self.kv_a_layernorm(self.param_sink_compressed_kv)
            self.attn.update_sink_kv(self.param_sink_k_pe, param_sink_compressed_kv)

        # With the split q-up projections populated, decode takes
        # _forward_decode_multistream, so the full q_b_proj storage is
        # redundant; release it (reload-safe, see release_q_b_proj_storage).
        # Skip when the fused mla_prolog path can run: it consumes
        # q_b_proj.weight directly.
        use_fused_mla_prolog = (
            self.use_mlaprolog and not self.is_sfa_c8 and not self.use_mome and self.param_sink_number == 0
        )
        if self.split_q_up_in_multistream and not use_fused_mla_prolog:
            release_q_b_proj_storage(self.q_b_proj)

    def _apply_rope(
        self,
        x: torch.Tensor,  # BNSD or TND or TD
        cos: torch.Tensor,  # BNSD
        sin: torch.Tensor,  # BNSD
    ) -> torch.Tensor:
        assert x.dim() in [2, 3, 4]
        assert cos.dim() == 4 and sin.dim() == 4
        T, D, shape = x.size(0), x.size(-1), x.shape
        x = x.view(T, -1, 1, D)  # BNSD
        if self.rope_interleaved:
            x = torch_npu.npu_interleave_rope(x, cos, sin)
        else:
            x = torch_npu.npu_rotary_mul(x, cos, sin)
        return x.view(shape)

    def _q_project(
        self,
        q_lora: torch.Tensor,  # TD
        cos: torch.Tensor,  # BNSD
        sin: torch.Tensor,  # BNSD
    ) -> tuple[torch.Tensor, torch.Tensor]:  # TND
        Q = self.qk_nope_head_dim
        R = self.qk_rope_head_dim
        N = self.num_heads // self.q_b_proj.tp_size

        if self.split_q_up_in_multistream:
            # The full q_b_proj storage is released after loading in this
            # mode; the split projections are numerically identical per row.
            q_nope = self.q_b_nope_proj(q_lora)[0].view(-1, N, Q)  # TND
            q_pe = self.q_b_pe_proj(q_lora)[0].view(-1, N, R)  # TND
        else:
            q = self.q_b_proj(q_lora)[0].view(-1, N, Q + R)  # TND
            q_nope, q_pe = torch.split(q, [Q, R], dim=-1)  # TND
        q_pe = self._apply_rope(q_pe, cos, sin)  # TND
        return q_nope, q_pe  # TND, q_nope not yet absorbed

    def _q_absorb(
        self,
        q_lora: torch.Tensor,  # TD
        cos: torch.Tensor, sin: torch.Tensor,  # BNSD
    ) -> tuple[torch.Tensor, torch.Tensor]:  # TND
        q_nope, q_pe = self._q_project(q_lora, cos, sin)
        return self._q_nope_absorb(q_nope), q_pe

    def _q_nope_absorb(self, q_nope: torch.Tensor, w_uk_t: torch.Tensor | None = None) -> torch.Tensor:
        # A CLA layer's local branch absorbs the same q with its own W_UK_T.
        absorb = self.attn.W_UK_T if w_uk_t is None else w_uk_t
        if absorb.size(-1) % 128 != 0 or absorb.size(-2) % 128 != 0:
            return (q_nope.transpose(0, 1) @ absorb).transpose(1, 0)

        if q_nope.size(1) * q_nope.size(2) < 65536:
            args = {"input": q_nope, "perm_x1": (1, 0, 2)}
        else:  # perm_x1 only support batch*k < 65536
            args = {"input": q_nope.transpose(0, 1)}
        return torch_npu.npu_transpose_batchmatmul(
            **args,  # TND -> NTD
            weight=absorb,  # [N, Q, L]
            perm_y=(1, 0, 2),  # NTD -> TND
        )

    def _pack_pertile128_kv(
        self,
        k_nope: torch.Tensor,
        k_pe: torch.Tensor,
    ) -> torch.Tensor:
        """Pack DSA C8 KV. Mirrors npu_pangu: hif8 unit-scale; fp8 dynamic tile128."""
        if k_nope.size(-1) != self.kv_lora_rank or k_pe.size(-1) != self.qk_rope_head_dim:
            raise ValueError(
                f"Unexpected dims for C8 pack: k_nope={k_nope.shape}, k_pe={k_pe.shape}"
            )
        tiled_shape = (*k_nope.shape[:-1], self.kv_lora_rank // FP8_TILE_SIZE, FP8_TILE_SIZE)
        if self.cache_dtype == "hif8_ds_mla":
            # npu_pangu default: direct cast + unit scale
            k_nope_quant = torch_npu.npu_dtype_cast(k_nope, torch_npu.hifloat8)
            scale_shape = (*k_nope.shape[:-1], self.kv_lora_rank // FP8_TILE_SIZE)
            k_nope_scale = torch.ones(scale_shape, dtype=torch.float32, device=k_nope.device)
        elif self.cache_dtype == "fp8_ds_mla":
            k_nope_quant, k_nope_scale = torch_npu.npu_dynamic_quant(
                k_nope.contiguous().view(tiled_shape),
                dst_type=torch.float8_e4m3fn,
            )
            k_nope_quant = k_nope_quant.view(k_nope.shape)
            k_nope_scale = k_nope_scale.view(*k_nope.shape[:-1], self.kv_lora_rank // FP8_TILE_SIZE)
        else:
            raise ValueError(f"Unsupported C8 dtype for pack: {self.cache_dtype}")

        packed = torch.cat(
            (
                k_nope_quant.view(torch.uint8),
                k_pe.contiguous().view(torch.uint8),
                k_nope_scale.contiguous().view(torch.uint8),
            ),
            dim=-1,
        )
        if packed.size(-1) != self.fp8_packed_kv_dim:
            raise RuntimeError(f"packed KV dim must be {self.fp8_packed_kv_dim}, got {packed.size(-1)}")
        return packed

    def _scatter_c8_kv_cache(
        self,
        k_nope: torch.Tensor,
        k_pe: torch.Tensor,
        kv_cache: tuple[torch.Tensor, ...],
        slots: torch.Tensor,
        slots_2d: torch.Tensor | None,
    ) -> None:
        packed = self._pack_pertile128_kv(k_nope, k_pe)
        cache = kv_cache[0].view(torch.int8)
        values = packed.view(torch.int8)
        if slots_2d is not None:
            torch_npu.npu_scatter_nd_update_(cache, slots_2d, values)
            return
        block_size = kv_cache[0].size(1)
        flat_slots = slots.reshape(-1)
        block_slots = torch.stack(
            (
                torch.div(flat_slots, block_size, rounding_mode="floor"),
                torch.remainder(flat_slots, block_size),
            ),
            dim=-1,
        )
        torch_npu.npu_scatter_nd_update_(
            cache,
            block_slots,
            values.view(-1, self.fp8_packed_kv_dim),
        )

    def _get_li_caches(
        self,
        kv_cache: tuple[torch.Tensor, ...] | None,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        if kv_cache is None:
            return None, None
        if self.noncontiguous_kv:
            # C8 layout from develop dsa.py: [0]=packed MLA, [1]=LI, [2]=LI scale
            ki_cache = kv_cache[1]
            ki_scale = kv_cache[2] if self.li_quant_mode else None
        else:
            ki_cache = kv_cache[2]
            ki_scale = None
        return ki_cache, ki_scale

    def _kv_norm_rope_cache(
        self,
        latent_kv: torch.Tensor,  # TD
        cos: torch.Tensor,  # BNSD
        sin: torch.Tensor,  # BNSD
        slots: torch.Tensor | NPUDSAMetadata | None,  # None for dummy_run
        kv_cache: tuple[torch.Tensor] | None,  # None for dummy_run
        kv_a_layernorm: torch.nn.Module | None = None,
        packed_kv: bool | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:  # k_nope: T1D, k_pe: T1D
        kv_a_layernorm = self.kv_a_layernorm if kv_a_layernorm is None else kv_a_layernorm
        packed_kv = self.noncontiguous_kv if packed_kv is None else packed_kv
        is_c8_cache = packed_kv and getattr(self, "is_sfa_c8", False)
        R, L = self.qk_rope_head_dim, self.kv_lora_rank
        slots_2d = getattr(slots, "slot_mapping_2d", None)
        slots = getattr(slots, "slot_mapping", slots)
        assert latent_kv.dim() == 2 and latent_kv.size(1) == R + L
        # TODO: CP/SP does not support DP > 1
        assert slots is None or slots.size(0) == latent_kv.size(0)
        valid_cache = None not in [kv_cache, slots]
        if latent_kv.size(0) == 0:  # no local_kv
            return cos.new_zeros(0, 1, L), cos.new_zeros(0, 1, R)
        # Quantized packed (C8) caches always take the explicit per-tile packing
        # path below, so the fused-op switch does not apply to them.
        use_fused_op = (
            model_extra_config.operator_opt_config.enable_kv_rmsnorm_rope_cache
            and not is_c8_cache
        )

        if valid_cache and use_fused_op and packed_kv:
            # Packed layout: one cache tensor holds latent (L) + RoPE (R).
            assert kv_cache[0].size(-1) == L + R
            k_pe, k_nope = torch.ops.custom.npu_ai_infra_kv_rmsnorm_rope_cache_v2(
                latent_kv.view(-1, 1, 1, L + R),  # BNSD
                kv_a_layernorm.weight,
                cos,
                sin,  # BNSD
                slots,
                k_cache=None,
                ckv_cache=cache_fit_shape(kv_cache[0], "4D"),
                k_rope_scale=None,
                k_rope_offset=None,
                epsilon=kv_a_layernorm.variance_epsilon,
                cache_mode="PA_NZ" if (self.kv_nz and self.dtype != torch.float16) else "PA",
                rotary_mode="half" if not self.rope_interleaved else "interleave",
                quant_mode="none",
                is_output_kv=True,
            )
        elif valid_cache and use_fused_op and self.rope_interleaved:
            # Dense layout: kv_cache[0] = nope (L), kv_cache[1] = rope (R).
            _, _, k_pe, k_nope = torch_npu.npu_kv_rmsnorm_rope_cache(
                latent_kv.view(-1, 1, 1, L + R),  # BNSD
                kv_a_layernorm.weight,
                cos,
                sin,  # BNSD
                slots,
                cache_fit_shape(kv_cache[1], "4D"),
                cache_fit_shape(kv_cache[0], "4D"),
                epsilon=kv_a_layernorm.variance_epsilon,
                cache_mode="PA_NZ" if (self.kv_nz and self.dtype != torch.float16) else "PA",
                is_output_kv=True,
            )  # -> rope cache [*, pg, 1, R], nope cache [*, pg, 1, L], k_pe BNSD, k_nope BNSD
        else:
            k_nope, k_pe = torch.split(latent_kv, [L, R], dim=-1)  # TD
            k_nope = kv_a_layernorm(k_nope)  # TD
            k_pe = self._apply_rope(k_pe, cos, sin)  # TD
            if valid_cache:
                if is_c8_cache:
                    # Quantized packed cache: per-tile scatter of nope and rope.
                    assert slots_2d is not None or slots is not None
                    self._scatter_c8_kv_cache(k_nope, k_pe, kv_cache, slots, slots_2d)
                elif self.on_ascend950 and self.noncontiguous_kv:
                    assert slots_2d is not None
                    if packed_kv:
                        kv = torch.cat([k_nope, k_pe], dim=-1)
                        torch_npu.npu_scatter_nd_update_(kv_cache[0], slots_2d, kv)
                    else:
                        torch_npu.npu_scatter_nd_update_(kv_cache[0], slots_2d, k_nope)
                        torch_npu.npu_scatter_nd_update_(kv_cache[1], slots_2d, k_pe)
                elif packed_kv:
                    # Packed layout: nope+rope written together into kv_cache[0].
                    assert slots_2d is not None
                    slot_indices = slots_2d
                    kv = torch.cat([k_nope, k_pe], dim=-1)
                    torch.ops.custom.npu_ai_infra_scatter_block_update_(kv_cache[0], slot_indices, kv)
                elif self.noncontiguous_kv:
                    # Dense layout, block scatter: kv_cache[0] nope, kv_cache[1] rope.
                    assert slots_2d is not None
                    torch.ops.custom.npu_ai_infra_scatter_block_update_(kv_cache[0], slots_2d, k_nope)
                    torch.ops.custom.npu_ai_infra_scatter_block_update_(kv_cache[1], slots_2d, k_pe)
                else:
                    # Dense layout, dense scatter: kv_cache[0] nope, kv_cache[1] rope.

                    def cache_kv(x: torch.Tensor, cache: torch.Tensor):
                        torch_npu.npu_scatter_nd_update_(
                            cache.view(-1, 1, cache.size(-1)),  # T1D
                            slots.view(-1, 1),  # T1
                            x.view(-1, 1, x.size(-1)),  # T1D
                        )

                    cache_kv(k_nope, kv_cache[0])
                    cache_kv(k_pe, kv_cache[1])
        return k_nope.view(-1, 1, L), k_pe.view(-1, 1, R)

    def _post_attn_absorb(self, out: torch.Tensor, w_uv: torch.Tensor | None = None) -> torch.Tensor:
        # A CLA layer's local branch projects with its own W_UV.
        w_uv = self.attn.W_UV if w_uv is None else w_uv
        assert out.dim() == 3
        if out.size(1) * out.size(2) < 65536:
            args = {"input": out, "perm_x1": (1, 0, 2)}
        else:  # perm_x1 only support batch*k < 65536
            args = {"input": out.transpose(0, 1)}

        return torch_npu.npu_transpose_batchmatmul(
            **args,  # TND -> NTD
            weight=w_uv,  # [N, L, V]
            perm_y=(1, 0, 2),  # NTD -> TND
        ).reshape(out.size(0), -1)  # [T, NV]

    def _mla_prolog(
        self,
        x: torch.Tensor,  # TD
        cos: torch.Tensor,  # BNSD
        sin: torch.Tensor,  # BNSD
        kv_cache: tuple[torch.Tensor],
        attn_metadata: NPUDSAMetadata,
    ):
        bs, _ = x.view(-1, x.shape[-1]).shape
        q_nope, q_pe, dequant_scale_q_nope, q_norm, dequant_scale_q_norm = torch_npu.npu_mla_prolog_v3(
            token_x=x.view(bs, 1, -1),
            weight_dq=self.q_a_proj.weight,  # BF16, NZ
            weight_uq_qr=self.q_b_proj.weight,  # BF16, NZ
            weight_uk=self.attn.W_UK_T,  # BF16, ND
            weight_dkv_kr=self.kv_a_proj_with_mqa.weight,  # BF16, NZ
            rmsnorm_gamma_cq=self.q_a_layernorm.weight,
            rmsnorm_gamma_ckv=self.kv_a_layernorm.weight,
            rope_sin=sin.squeeze(1),
            rope_cos=cos.squeeze(1),
            kv_cache=kv_cache[0],
            kr_cache=kv_cache[1],
            cache_index=attn_metadata.slot_mapping.view(bs, -1),
            dequant_scale_x=None,
            dequant_scale_w_dq=None,
            dequant_scale_w_uq_qr=self.q_b_proj.weight_scale.view(1, -1) if self.quant_symbol else None,
            dequant_scale_w_dkv_kr=None,
            rmsnorm_epsilon_cq=self.q_a_layernorm.variance_epsilon,
            rmsnorm_epsilon_ckv=self.kv_a_layernorm.variance_epsilon,
            cache_mode="PA_BSND",
            query_norm_flag=True,
            weight_quant_mode=1 if self.quant_symbol else 0,
        )
        k_nope = kv_cache[0]
        k_pe = kv_cache[1]
        q_nope = q_nope.view(bs, self.num_local_heads, self.kv_lora_rank)
        q_pe = q_pe.view(bs, self.num_local_heads, -1)
        if self.quant_symbol:
            q_norm = q_norm.view(-1, q_norm.shape[-1])
            dequant_scale_q_norm = dequant_scale_q_norm.view(-1)
            q_norm = {"x_int8": q_norm, "pertoken_scale": dequant_scale_q_norm}
        return q_nope, q_pe, q_norm, k_nope, k_pe, dequant_scale_q_nope, dequant_scale_q_norm

    def _apply_mome_out_prefill_sp(
        self,
        cp_out: torch.Tensor,
        sp_manager: SPManager,
    ) -> torch.Tensor | None:
        mome_sp_args = self._get_mome_prefill_sp_args(2)
        if mome_sp_args is None:
            return None

        sp_out = sp_manager.cp_to_sp(cp_out)
        mome_sp_out = conv_sp(sp_out, *mome_sp_args, inplace=False)
        if self.o_proj.requires_input_partition():
            mome_sp_out = sp_manager.sp_to_tp(mome_sp_out)
            mome_sp_out = self._apply_o_proj(mome_sp_out)
            return sp_manager.slice_tokens(mome_sp_out)
        return self._apply_o_proj(mome_sp_out)

    # ======================= attention =======================

    def _apply_sink_offset(self, topk_idx: torch.Tensor | None):
        if self.param_sink_number == 0 or topk_idx is None:
            return topk_idx  # dummy_run or sink_disabled

        if not self.noncontiguous_kv:  # only contiguous kv
            sink_indices = torch.arange(
                self.param_sink_number,
                device=topk_idx.device,
                dtype=topk_idx.dtype,
            ).expand(topk_idx.size(0), 1, self.param_sink_number)
            mask = (topk_idx != -1).to(topk_idx.dtype)
            with_offset = topk_idx + mask * self.param_sink_number
            topk_idx = torch.concat([sink_indices, with_offset], dim=2)
        return topk_idx

    def _init_fia_metadata_sharing(
        self,
        config: DeepseekV2Config | DeepseekV3Config,
    ) -> None:
        dsa_layers = enable_fia_metadata_sharing(
            self, config, model_extra_config.operator_opt_config
        )
        if dsa_layers is None:
            return
        first_dsa_sink_layer = min(dsa_layers)
        first_decode_sink_layer = first_dsa_sink_layer
        mla_num_heads = config.num_attention_heads // self.tp_size
        if mla_num_heads == self.num_local_heads:
            mla_layers = set(range(config.num_hidden_layers)) - dsa_layers
            if mla_layers:
                first_decode_sink_layer = min(first_decode_sink_layer, min(mla_layers))
        self._produce_sink_fia_metadata = {
            "decode": self.layer_idx == first_decode_sink_layer,
            "prefill_absorb": self.layer_idx == first_dsa_sink_layer,
            "prefill": self.layer_idx == first_dsa_sink_layer,
        }

        self._sink_fia_metadata_key_suffix = (
            "sink",
            self.num_local_heads,
            self.kv_lora_rank,
            self.qk_rope_head_dim,
            self.param_sink_number,
        )
        callers = tuple(
            (caller, *self._sink_fia_metadata_key_suffix)
            for caller in ("decode", "prefill_absorb", "prefill")
        )
        self._fia_metadata_shared_op = get_cross_layer_shared_op(
            name="pangu_v2_fia_sink_metadata",
            op=torch.ops.custom._npu_fused_infer_attention_sink_metadata,
            shape=(1024,),
            dtype=torch.int32,
            callers=callers,
        )

    def _get_sink_fia_metadata(
        self,
        op_args: dict,
        metadata_caller: str,
    ) -> torch.Tensor:
        if not getattr(self, "_share_fia_metadata", False):
            return torch.ops.custom._npu_fused_infer_attention_sink_metadata(
                **op_args
            )
        return self._fia_metadata_shared_op(
            op_args,
            self._produce_sink_fia_metadata.get(metadata_caller, True),
            (metadata_caller, *self._sink_fia_metadata_key_suffix),
        )

    def _apply_attention_rescale_pioneer(
        self,
        q_nope: torch.Tensor,
        q_pe: torch.Tensor,
        q_cumlens: torch.Tensor,
        kv_lens: torch.Tensor,
        topk_idx: torch.Tensor,
        block_table: torch.Tensor,
        kv_cache: tuple[torch.Tensor, ...],
        metadata_caller: str = "decode",
    ) -> torch.Tensor:

        output_normal, smax_normal, ssum_normal = (
            torch.ops.custom.npu_ai_infra_sparse_flash_attention_pioneer(
                query=q_nope,
                key=kv_cache[0].unsqueeze(2),
                value=self.dummy_value_cache,
                query_rope=q_pe,
                sparse_indices=topk_idx,
                scale_value=self.scaling,
                sparse_block_size=1,
                block_table=block_table,
                actual_seq_lengths_query=q_cumlens.to(torch.int32),
                actual_seq_lengths_kv=kv_lens.to(torch.int32),
                attention_mode=2,
                layout_query="TND",
                layout_kv="PA_BSND",
                sparse_mode=3,
                batch_invariant=True,
                return_softmax_lse=True,
            )
        )


        act_seq_qlen = q_cumlens.to(torch.int64)
        if self.on_ascend950:
            batch_size = q_cumlens.numel()
            key_cache = self.attn.impl.sink_k_nope.unsqueeze(0).expand(batch_size, -1, -1, -1).squeeze(2).contiguous()
            key_rope_cache = (
                self.attn.impl.sink_kv[..., self.kv_lora_rank:]
                .unsqueeze(0).expand(batch_size, -1, -1, -1).squeeze(2).contiguous()
            )

            block_table = torch.zeros(q_cumlens.numel(), dtype=torch.int32, device=key_cache.device).unsqueeze(1)
            kv_sequence_lengths = torch.full_like(act_seq_qlen, self.param_sink_number)

            meta_data = torch.ops.custom.npu_ai_infra_attention_pioneer_metadata(
                                        self.num_local_heads,
                                        1,
                                        q_nope.shape[-1],
                                        key_cache.shape[-1],
                                        actual_seq_lengths=act_seq_qlen,
                                        actual_seq_lengths_kv=kv_sequence_lengths,
                                        sparse_mode=4,
                                        pre_tokens=2147483647,
                                        next_tokens=2147483647,
                                        input_layout="TND",
                                        rope_head_dim=q_pe.shape[-1],
                                        sink_number=0,
                                        block_size=self.param_sink_number,
                                        soc_version="ascend950",
                                    )

            output_sink, _, smax_sink, ssum_sink = torch.ops.custom.npu_ai_infra_attention_pioneer_v2(
                q_nope,
                key_cache,
                key_cache,
                meta_data,
                atten_mask=self.atten_mask,
                actual_seq_lengths=act_seq_qlen,
                actual_seq_lengths_kv=kv_sequence_lengths,
                block_table=block_table,
                query_rope=q_pe,
                key_rope=key_rope_cache,
                **_a5_sink_attention_kwargs(self.num_local_heads, self.scaling, self.param_sink_number),
            )
        else:
            sink_k_nope = self.attn.impl.sink_k_nope.repeat(q_cumlens.numel(), 1, 1)
            sink_value = sink_k_nope
            sink_k_pe = self.attn.impl.sink_kv[..., self.kv_lora_rank:].repeat(q_cumlens.numel(), 1, 1)
            actual_seq_sink_len = torch.arange(
                            self.param_sink_number,
                            (q_cumlens.numel() + 1) * self.param_sink_number,
                            self.param_sink_number,
                            dtype=q_cumlens.dtype,
                            device=q_cumlens.device,
                        ).to(torch.int64)
            meta_data = self._get_sink_fia_metadata(
                {
                    "num_heads_q": self.num_local_heads,
                    "num_heads_kv": 1,
                    "head_dim_qk": q_nope.shape[-1],
                    "head_dim_v": sink_k_nope.shape[-1],
                    "actual_seq_lengths": act_seq_qlen,
                    "actual_seq_lengths_kv": actual_seq_sink_len,
                    "sparse_mode": 0,
                    "batch_invariant": True,
                    "input_layout": "TND",
                    "input_layout_kv": "TND",
                    "rope_head_dim": q_pe.shape[-1],
                },
                metadata_caller,
            )
            output_sink, _, smax_sink, ssum_sink = (
                torch.ops.custom.npu_fused_infer_attention_sink_v2(
                    q_nope,
                    sink_k_nope,
                    sink_value,
                    query_rope=q_pe,
                    key_rope=sink_k_pe,
                    actual_seq_qlen=act_seq_qlen,
                    actual_seq_kvlen=actual_seq_sink_len,
                    num_query_heads=self.num_local_heads,
                    num_key_value_heads=1,
                    softmax_scale=self.scaling,
                    input_layout="TND",
                    sparse_mode=0,
                    batch_invariant=True,
                    return_softmax_lse=False,
                    return_softmax_max_sum=True,
                    meta_data=meta_data,
                )
            )

        attn_output = torch.empty_like(output_normal)
        smax_normal = smax_normal.squeeze(0).unsqueeze(2)  # 1TN -> TN1
        ssum_normal = ssum_normal.squeeze(0).unsqueeze(2)  # 1TN -> TN1

        attn_output, _, _ = ops.apply_FA_rescale_forward(
            output_normal, smax_normal, ssum_normal, output_sink, smax_sink, ssum_sink)

        return attn_output

    @attn_decorator(type="dsa")
    def _apply_attn_absorb(
        self,
        q_nope: torch.Tensor,  # [T, N, D]
        q_pe: torch.Tensor,  # [T, N, R]
        q_cumlens: torch.Tensor = None,  # int32 [B]
        kv_lens: torch.Tensor = None,  # int32 [B]
        topk_idx: torch.Tensor = None,  # int32 [T, 1, K]
        block_table: torch.Tensor = None,  # int32 [B, *]
        kv_cache: torch.Tensor = None,
        attn_metadata: NPUDSAMetadata = None,  # required by omni-cache
        metadata_caller: str = "decode",
    ) -> torch.Tensor:  # [T, N, L]
        if None in [q_cumlens, kv_lens, block_table, topk_idx]:
            return torch.zeros_like(q_nope)  # dummy
        if getattr(self, "is_sfa_c8", False):
            # Packed FP8/HiF8 latent cache plus BF16 rotary data and independent sinks.
            if self.cache_dtype == "hif8_ds_mla" and topk_idx.size(-1) != 2048:
                raise ValueError(
                    "HiFloat8 SFA requires sparse_indices.size(-1) == 2048, "
                    f"got {topk_idx.size(-1)}."
                )
            query = torch.cat([q_nope, q_pe], dim=-1).contiguous()
            key = kv_cache[0].unsqueeze(2) if kv_cache[0].dim() == 3 else kv_cache[0]
            key_dtype = None
            value_dtype = None
            if self.cache_dtype == "fp8_ds_mla":
                key = key.view(torch.float8_e4m3fn)
                value = self.dummy_value_cache_quant.view(torch.float8_e4m3fn)
            elif self.cache_dtype == "hif8_ds_mla":
                value = self.dummy_value_cache_quant
                key_dtype = torch_npu.hifloat8
                value_dtype = torch_npu.hifloat8
            kwargs = dict(
                query=query,
                key=key,
                value=value,
                sparse_indices=topk_idx,
                scale_value=self.scaling,
                key_quant_mode=2,
                value_quant_mode=2,
                block_table=block_table,
                actual_seq_lengths_query=q_cumlens.clone(),
                actual_seq_lengths_kv=kv_lens.clone(),
                sparse_block_size=1,
                layout_query="TND",
                layout_kv="PA_BSND",
                sparse_mode=3,
                attention_mode=2,
                quant_scale_repo_mode=1,
                tile_size=128,
                rope_head_dim=64,
            )
            if self.param_sink_number > 0:
                kwargs["key_sink"] = self.attn.impl.sink_kv
                kwargs["value_sink"] = self.attn.impl.sink_k_nope
            if key_dtype is not None:
                kwargs["key_dtype"] = key_dtype
                kwargs["value_dtype"] = value_dtype
            return torch.ops.custom.npu_ai_infra_kv_quant_sparse_flash_attention(**kwargs)

        if self.noncontiguous_kv and self.param_sink_number == 0:
            kwargs = dict(
                query=q_nope,
                query_rope=q_pe,
                key=kv_cache[0].unsqueeze(2),
                value=kv_cache[0][..., : self.kv_lora_rank].unsqueeze(2),
                sparse_indices=topk_idx,
                scale_value=self.scaling,
                sparse_block_size=1,
                block_table=block_table,
                actual_seq_lengths_query=q_cumlens,
                actual_seq_lengths_kv=kv_lens,
                pre_tokens=(1 << 63) - 1, next_tokens=(1 << 63) - 1,
                attention_mode=2, layout_query="TND",
                layout_kv="PA_BSND", sparse_mode=3,
            )
            if not self.use_gpt_oss_sink:
                return torch.ops.custom.npu_ai_infra_sparse_flash_attention(**kwargs)[0]
            kwargs["batch_invariant"] = model_extra_config.operator_opt_config.use_batch_invariant_op
            if not self.use_gpt_oss_sink_rescale:
                kwargs["sinks"] = self.gpt_oss_sink
                return torch.ops.custom.npu_ai_infra_sparse_flash_attention(**kwargs)[0]
            kwargs["return_softmax_lse"] = True
            attn_out, softmax_max, softmax_sum = (
                torch.ops.custom.npu_ai_infra_sparse_flash_attention_pioneer(**kwargs)
            )
            softmax_lse = (softmax_max + torch.log(softmax_sum)).squeeze(0)
            return self._rescale_gpt_oss_sink_output(attn_out, softmax_lse, self.gpt_oss_sink, "TND")

        if self.noncontiguous_kv:
            assert self.param_sink_number > 0
            if model_extra_config.operator_opt_config.enable_precision_strong_consistency:
                return self._apply_attention_rescale_pioneer(
                    q_nope, q_pe, q_cumlens, kv_lens,
                    topk_idx, block_table, kv_cache, metadata_caller,
                )
            kwargs = dict(
                query=q_nope,
                query_rope=q_pe,
                key=kv_cache[0].unsqueeze(2),
                value=self.dummy_value_cache,
                sparse_indices=topk_idx,
                scale_value=self.scaling,
                sparse_block_size=1,
                block_table=block_table,
                actual_seq_lengths_query=q_cumlens,
                actual_seq_lengths_kv=kv_lens,
                pre_tokens=(1 << 63) - 1,
                next_tokens=(1 << 63) - 1,
                attention_mode=2,
                layout_query="TND",
                layout_kv="PA_BSND",
                sparse_mode=3,
                key_sink=self.attn.impl.sink_kv,
                value_sink=self.attn.impl.sink_k_nope,
            )
            if model_extra_config.operator_opt_config.use_batch_invariant_op:
                kwargs["batch_invariant"] = True
            return torch.ops.custom.npu_ai_infra_sparse_flash_attention_pioneer(**kwargs)[0]

        if self.use_gpt_oss_sink:
            raise NotImplementedError(
                "npu_sparse_flash_attention_enhance takes no sink logit, so a "
                "GPT-OSS sink layer needs use_noncontiguous_kv."
            )
        return torch.ops.custom.npu_sparse_flash_attention_enhance(
            query=q_nope,  # [T, N, L]
            key=kv_cache[0],  # [*, pg, 1, L]
            value=kv_cache[0],  # [*, pg, 1, L]
            query_rope=q_pe,  # [T, N, R]
            key_rope=kv_cache[1],  # [*, pg, 1, R]
            sparse_indices=topk_idx,  # [T, 1, 2048]
            sparse_block_size=1,
            layout_query="TND",
            layout_kv="PA_BSND",
            block_table=block_table,
            actual_seq_lengths_query=q_cumlens,
            actual_seq_lengths_kv=kv_lens,
            scale_value=self.scaling,
            attention_mode=2,
            sparse_mode=3,
        )[0]  # -> [T, N, L]

    # ======================= forward =======================

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        topk_indices_buffer: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return torch.ops.vllm.npu_dsa_forward(
            x, cos, sin, self.prefix, topk_indices_buffer, self.attn.layer_name
        )

    def forward_mhc_deferred(
        self, x, cos, sin, residual, mhc_layer_name, task_key, topk_indices_buffer=None
    ):
        hidden_states, h_post, h_res = torch.ops.vllm.npu_dsa_forward_mhc_deferred(
            x, cos, sin, residual, self.prefix, mhc_layer_name, task_key,
            self.attn.layer_name,
        )
        return (hidden_states, h_post, h_res), topk_indices_buffer

    def _forward_prefill(
        self,
        x: torch.Tensor,  # TD
        cos: torch.Tensor,  # BNSD
        sin: torch.Tensor,  # BNSD
        attn_metadata: NPUDSAMetadata = None,
        kv_cache: tuple[torch.Tensor] = None,
        topk_indices_buffer: torch.Tensor | None = None,
        pd_mixed_flag: bool = False,
        use_local_query: bool = False,
        output_sp_slice: slice | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        assert not self.sharded_o_proj  # TODO: sharded_o_proj only support cp
        assert not attn_metadata or not self.ena_kvsp  # TODO: kvsp only support cp

        def get_mome_args():
            if self.noncontiguous_kv:
                return {"is_prefill": True}
            else:
                return {"only_prefill": bool(pd_mixed_flag)}

        if attn_metadata:
            kv_lens = attn_metadata.seq_lens.to(torch.int32)
            q_cumlens = attn_metadata.query_cumlens.to(torch.int32)
            block_table = attn_metadata.block_table
            ki_cache, ki_scale_cache = self._get_li_caches(kv_cache)
        else:
            q_cumlens, kv_lens, block_table = None, None, None
            ki_cache, ki_scale_cache = None, None
        is_dummy = None in [q_cumlens, kv_lens, block_table]

        sp_manager = getattr(attn_metadata, "sp_manager", DummySPManager())

        if self.ena_sp:
            sp_cos = sp_manager.slice_tokens(cos, cached="cos")
            sp_sin = sp_manager.slice_tokens(sin, cached="sin")

        q = self.q_a_proj(x)[0]  # TD
        q = sp_manager.ag_tokens(q) if self.ena_sp and self.use_mome else q
        q = self._maybe_mome_q(q, get_mome_args)  # TD
        q = self.q_a_layernorm(q)  # TD
        q_lora = sp_manager.ag_tokens(q) if self.ena_sp and not self.use_mome else q

        if use_local_query:
            # The PR graph wrapper has gathered full tokens and disabled SP.
            assert not self.ena_sp and self.q_b_proj.tp_size == 1 and not is_dummy
            assert not pd_mixed_flag and q_lora.size(0) == x.size(0)
            group = get_tp_group()
            query_layout = ZigzagQueryLayout(x.size(0), group.world_size, group.rank_in_group)
            local_q, local_cos, local_sin = map(query_layout.select, (q_lora, cos, sin))
            q_nope, q_pe = self._q_absorb(local_q, local_cos, local_sin)
            local_q_cumlens, local_kv_lens, local_blocks = query_layout.metadata(q_cumlens, kv_lens, block_table)
        else:
            q_nope, q_pe = self._q_absorb(q_lora, cos, sin)  # TND

        kv = self.kv_a_proj_with_mqa(x)[0]  # TD
        kv = self._maybe_mome_kv(kv, get_mome_args)  # TD
        if self.ena_sp:
            kv = sp_manager.ag_tokens(kv)
        self._kv_norm_rope_cache(kv, cos, sin, attn_metadata, kv_cache)

        cur_stream = torch.npu.current_stream()

        if not self.skip_topk:
            if use_local_query:
                local_x = query_layout.select(x)
            if model_extra_config.operator_opt_config.li_prolog_multi_stream:
                self.ki_stream.wait_stream(cur_stream)
                self.wi_stream.wait_stream(cur_stream)

            if use_local_query:
                wi, qi, ki = self.indexer._li_prolog_ext(
                    local_x, local_q, x,
                    (local_cos, local_sin), (cos, sin))
            elif self.ena_sp:
                wi, qi, ki = self.indexer._li_prolog(x, q, sp_cos, sp_sin)
                ki = sp_manager.ag_tokens(ki)
                wi = sp_manager.ag_tokens(wi)
                qi = sp_manager.ag_tokens(qi)
            else:
                wi, qi, ki = self.indexer._li_prolog(x, q, cos, sin)
            self.indexer._update_cache(ki, attn_metadata, ki_cache, ki_scale_cache)
            if use_local_query:
                q_cumlens, kv_lens, block_table = local_q_cumlens, local_kv_lens, local_blocks
            topk_idx = self.indexer._apply_lightning_indexer(
                wi, qi, ki_cache, q_cumlens, kv_lens, block_table,
                ki_scale_cache, metadata_caller="prefill"
            )
        elif not is_dummy:
            if use_local_query:
                q_cumlens, kv_lens, block_table = local_q_cumlens, local_kv_lens, local_blocks
                topk_idx = topk_indices_buffer[:local_q.size(0)]
            else:
                topk_idx = topk_indices_buffer
        else:
            topk_idx = None

        next_topk_indices_buffer = (
            topk_idx if topk_idx is not None else topk_indices_buffer
        )
        if use_local_query:
            # The SP caller already owns this rank's zigzag query count.
            # Reuse layers keep these indices local, without AG or re-slicing.
            assert next_topk_indices_buffer.shape == topk_indices_buffer.shape
        topk_idx = self._apply_sink_offset(topk_idx)

        out = self._apply_attn_absorb(
            q_nope,
            q_pe,  # [T, N, D]
            q_cumlens,
            kv_lens,
            topk_idx,  # int32 [T, 1, K]
            block_table,  # int32 [B, *]
            kv_cache,
            attn_metadata=attn_metadata,
            metadata_caller="prefill",
        )  # -> TND

        out = self._post_attn_absorb(out)  # T,ND
        if use_local_query:
            out = query_layout.gather_full_tokens(out)
        out = self._maybe_mome_out(out, get_mome_args)  # T,ND
        if output_sp_slice is not None:
            assert not self.ena_sp  # The graph wrapper owns the final SP output.
            out = out[output_sp_slice].contiguous()
        if self.ena_sp:
            out = sp_manager.align_tokens(out)  # T,ND
        out = self._apply_o_proj(out)
        if self.ena_sp and out.size(0) != x.size(0):
            out = sp_manager.slice_tokens(out)
        return out, next_topk_indices_buffer

    def _forward_prefill_cp(
        self,
        sp_x: torch.Tensor,  # TD
        cos: torch.Tensor,  # BNSD
        sin: torch.Tensor,  # BNSD
        attn_metadata: NPUDSAMetadata = None,
        kv_cache: tuple[torch.Tensor] = None,
        topk_indices_buffer: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        assert self.q_b_proj.tp_size == 1  # full head required
        assert self.kv_b_proj.tp_size == 1  # full head required
        assert self.ena_sp  # dependency
        """
        SP here refers to standard SP splitting, i.e., splitting tokens with ceil division
        based on sp_size.

        CP refers to zigzag-form SP splitting, aimed at adjusting query distribution to
        balance attention computation across ranks.
        """

        if attn_metadata:
            ki_cache, ki_scale_cache = self._get_li_caches(kv_cache)
        else:
            ki_cache, ki_scale_cache = None, None

        def get_mome_args():
            if self.noncontiguous_kv:
                return {"is_prefill": True}
            else:
                return {"only_prefill": False}

        sp_manager = getattr(attn_metadata, "sp_manager", DummySPManager())
        q_cumlens, kv_lens, blk_table = self._cp_attention_metadata(sp_manager)

        if self.ena_kvsp:
            assert not self.use_mome and not self.noncontiguous_kv
            kvsp_manager = getattr(attn_metadata, "kvsp_manager", DummyKVSPMaganer())
            local_cos = kvsp_manager.select_local(cos, cached="cos")
            local_sin = kvsp_manager.select_local(sin, cached="sin")
            local_slots, local_slots_2d = kvsp_manager.local_slots()

        cur_stream = torch.npu.current_stream()
        sub_stream = named_stream("npu_dsa_sub_stream")

        sp_cos_sin = (sp_manager.slice_tokens(cos, cached="cos"), sp_manager.slice_tokens(sin, cached="sin"))
        cp_cos_sin = self._cp_rotary_rows(sp_manager, cos, sin)

        q_lora = self._cp_q_lora(sp_x, sp_manager, get_mome_args)  # TD, cp
        q_nope, q_pe = self._q_absorb(q_lora, *cp_cos_sin)  # TND, full head, cp

        kv = self.kv_a_proj_with_mqa(sp_x)[0]  # TD, sp
        if self.ena_kvsp:
            kv = kvsp_manager.sp_to_local(kv)  # TD, sp -> local
            self._kv_norm_rope_cache(kv, local_cos, local_sin, local_slots, kv_cache)
            nope_cache, rope_cache = self._cp_gather_cache_pages(
                kvsp_manager, kv_cache, cur_stream, sub_stream,
            )
        else:
            mome_sp_kv_applied = False
            if self.use_mome:
                mome_sp_kv_applied = self._apply_mome_prefill_sp(
                    kv[:, :self.kv_lora_rank],
                    1,
                    inplace=True,
                )
            kv = sp_manager.ag_tokens(kv)  # TD
            if self.use_mome and not mome_sp_kv_applied:
                kv = self._maybe_mome_kv(kv, get_mome_args)  # TD
            self._kv_norm_rope_cache(kv, cos, sin, attn_metadata, kv_cache)

        if not self.skip_topk:
            if model_extra_config.operator_opt_config.li_prolog_multi_stream:
                self.ki_stream.wait_stream(cur_stream)
                self.wi_stream.wait_stream(cur_stream)

            wi, qi, ki = self.indexer._li_prolog_ext(sp_x, q_lora, sp_x, cp_cos_sin, sp_cos_sin)  # TND
            wi = sp_manager.sp_to_cp(wi)  # T, cp

            if self.ena_kvsp:
                ki = kvsp_manager.sp_to_local(ki)
                self.indexer._update_cache(ki, local_slots_2d, ki_cache)
                ki_cache, _, blk_table = kvsp_manager.ag_pages(ki_cache)
                kv_cache = (nope_cache, rope_cache, ki_cache)
                cur_stream.wait_stream(sub_stream)  # wait for ag_pages
            else:
                ki = sp_manager.ag_tokens(ki)
                if hasattr(attn_metadata, "cache_fn") and not self.noncontiguous_kv:
                    attn_metadata.cache_fn(ki.view(-1, ki.size(-1)), ki_cache)
                else:
                    self.indexer._update_cache(ki, attn_metadata, ki_cache, ki_scale_cache)

        if self.sharded_o_proj:
            sub_stream.wait_stream(cur_stream)
            with torch.npu.stream(sub_stream):
                self.o_proj.prefetch(sub_stream)

        if not self.skip_topk:
            topk_idx = self.indexer._apply_lightning_indexer(
                wi,
                qi,  # TND, cp
                ki_cache,
                q_cumlens,
                kv_lens,  # int32 [2B]
                blk_table,  # int32 [2B, *]
                ki_scale_cache,
                metadata_caller="prefill",
            )  # int32 [T_cp, 1, K] or None for dummy_run
        else:
            # A full DSA layer returns CP-layout top-k; shared consumes it directly.
            topk_idx = topk_indices_buffer

        next_topk_indices_buffer = (
            topk_idx if topk_idx is not None else topk_indices_buffer
        )
        topk_idx = self._apply_sink_offset(topk_idx)

        cp_out = self._apply_attn_absorb(
            q_nope,
            q_pe,  # TND, full head, cp
            q_cumlens,
            kv_lens,  # int32 [2B]
            topk_idx,  # int32 [T, 1, K]
            blk_table,  # int32 [2B, *]
            kv_cache,
            attn_metadata=attn_metadata,
            metadata_caller="prefill",
        )
        cp_out = self._post_attn_absorb(cp_out)

        if self.use_mome:
            mome_sp_out = self._apply_mome_out_prefill_sp(cp_out, sp_manager)
            if mome_sp_out is not None:
                return mome_sp_out, next_topk_indices_buffer

            cp_out = sp_manager.cp_to_sp(cp_out)  # sp
            cp_out = sp_manager.ag_tokens(cp_out)  # TD
            cp_out = self._maybe_mome_out(cp_out, get_mome_args)  # TD
            # Pad to a multiple of sp_size before o_proj, exactly as the
            # non-CP prefill does: this branch hands o_proj the full token
            # set, and a ReduceScatter y_transform (the correct SP setting)
            # requires the row count to divide by the group size. Without
            # the pad, any request whose token count is not a multiple of
            # sp_size aborts in the communicator.
            cp_out = sp_manager.align_tokens(cp_out)  # TD, padded
            cp_out = self._apply_o_proj(cp_out)
            if cp_out.size(0) != sp_x.size(0):
                cp_out = sp_manager.slice_tokens(cp_out)
            return cp_out, next_topk_indices_buffer

        return self._cp_o_proj(cp_out, sp_manager), next_topk_indices_buffer

    @staticmethod
    def _cp_attention_metadata(sp_manager: SPManager) -> tuple:
        """Lengths and cache pages for this rank's zigzag query layout."""
        q_cumlens, kv_lens, _, block_table = sp_manager.cp_attn_meta()
        return q_cumlens, kv_lens, block_table

    @staticmethod
    def _cp_rotary_rows(sp_manager: SPManager, cos: torch.Tensor, sin: torch.Tensor) -> tuple:
        """Prepare CP rotary rows separately so DSA retains its SP-before-CP order."""
        return (sp_manager.cp_slice(cos, cached="cos"), sp_manager.cp_slice(sin, cached="sin"))

    @staticmethod
    def _cp_gather_cache_pages(kvsp_manager, kv_cache, cur_stream, sub_stream) -> tuple:
        """Launch both latent and rotary page gathers on the CP cache stream."""
        nope_cache, rope_cache, _ = kv_cache or (None, None, None)
        ag_nope_cache, *_ = kvsp_manager.ag_pages(nope_cache, seperate=True)
        ag_rope_cache, *_ = kvsp_manager.ag_pages(rope_cache, seperate=True)
        sub_stream.wait_stream(cur_stream)
        with torch.npu.stream(sub_stream):
            return ag_nope_cache(), ag_rope_cache()

    def _cp_q_lora(
        self,
        sp_x: torch.Tensor,  # TD, sp
        sp_manager: SPManager,
        get_mome_args: Callable | None,
    ) -> torch.Tensor:  # TD, cp
        """Down-project this rank's SP tokens and move them to the zigzag CP layout."""
        q_lora = self.q_a_proj(sp_x)[0]  # TD, sp

        if self.use_mome:
            if self._apply_mome_prefill_sp(q_lora, 0, inplace=True):
                q_lora = sp_manager.sp_to_cp(q_lora)  # TD, sp -> cp
            else:
                q_lora = sp_manager.ag_tokens(q_lora)
                q_lora = self._maybe_mome_q(q_lora, get_mome_args)  # TD
                q_lora = sp_manager.cp_slice(q_lora)  # TD, cp
        else:
            q_lora = sp_manager.sp_to_cp(q_lora)  # TD, cp

        return self.q_a_layernorm(q_lora)  # TD, cp

    def _cp_o_proj(self, cp_out: torch.Tensor, sp_manager: SPManager) -> torch.Tensor:  # TD, sp
        """Project a full-head CP-layout output and hand back this rank's SP tokens."""
        if self.o_proj.requires_input_partition():
            cp_out = sp_manager.sp_to_tp(cp_out)
        cp_out = self._apply_o_proj(cp_out)
        return sp_manager.cp_to_sp(cp_out)

    def _forward_decode(
        self,
        x: torch.Tensor,   # TD
        cos: torch.Tensor, # BNSD
        sin: torch.Tensor, # BNSD
        attn_metadata: NPUDSAMetadata,
        kv_cache: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        topk_indices_buffer: torch.Tensor | None = None,
        pd_mixed_flag: int = 0,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        assert not self.ena_sp  # TODO: support decode sp in the future
        assert not self.sharded_o_proj  # sharded_o_proj is for prefill only
        use_fused_mla_prolog = (
            self.use_mlaprolog
            and not self.is_sfa_c8
            and not self.use_mome
            and self.param_sink_number == 0
        )

        def get_mome_args():
            if self.noncontiguous_kv:
                return {}
            else:
                return {
                    "force_decode": True if pd_mixed_flag == 1 else False,
                    "short_prefill": True if pd_mixed_flag == 2 else False,
                }

        q_cumlens = attn_metadata.query_cumlens.to(torch.int32)
        kv_lens = attn_metadata.seq_lens.to(torch.int32)
        blk_table = attn_metadata.block_table

        can_decode_shared_indexer_multistream = (
            self.skip_topk
            and self.shared_indexer_kv_stream is not None
        )
        if can_decode_shared_indexer_multistream and not use_fused_mla_prolog and not self.ena_kvsp:
            q_nope, q_pe = self._decode_prolog_multistream_a5(
                x, cos, sin, attn_metadata, kv_cache, get_mome_args,
                kv_stream=self.shared_indexer_kv_stream,
            )
            return self._finish_multistream_decode(
                q_nope, q_pe, topk_indices_buffer[:x.shape[0]],
                q_cumlens, kv_lens, blk_table,
                kv_cache, attn_metadata, get_mome_args,
            )

        can_decode_multistream = self.side_stream is not None and not self.skip_topk
        if can_decode_multistream and not use_fused_mla_prolog and not self.ena_kvsp:
            if self.li_stream is not None:
                return self._forward_decode_multistream_a5(
                    x, cos, sin, attn_metadata, kv_cache, get_mome_args
                )
            return self._forward_decode_multistream(
                x, cos, sin, attn_metadata, kv_cache,
                topk_indices_buffer, pd_mixed_flag,
            )
        ki_cache, ki_scale_cache = self._get_li_caches(kv_cache)

        cur_stream = torch.npu.current_stream()

        if use_fused_mla_prolog:
            q_nope, q_pe, q_lora, *_ = self._mla_prolog(x, cos, sin, kv_cache, attn_metadata)
        else:
            q_lora = self.q_a_proj(x)[0]  # TD
            q_lora = self._maybe_mome_q(q_lora, get_mome_args)  # TD
            q_lora = self.q_a_layernorm(q_lora)  # TD
            q_nope, q_pe = self._q_absorb(q_lora, cos, sin)  # TND

            kv = self.kv_a_proj_with_mqa(x)[0]  # TD
            kv = self._maybe_mome_kv(kv, get_mome_args)  # TD
            self._kv_norm_rope_cache(kv, cos, sin, attn_metadata, kv_cache)

        if not self.skip_topk:
            if model_extra_config.operator_opt_config.li_prolog_multi_stream:
                self.ki_stream.wait_stream(cur_stream)
                self.wi_stream.wait_stream(cur_stream)

            wi, qi, ki = self.indexer._li_prolog(x, q_lora, cos, sin)
            self.indexer._update_cache(ki, attn_metadata, ki_cache, ki_scale_cache)

            if self.ena_kvsp:
                assert not self.use_mome and not self.noncontiguous_kv
                assert not self.li_quant_mode
                kvsp_manager: KVSPMaganer = attn_metadata.kvsp_manager
                nope_cache, rope_cache, ki_cache = kv_cache
                nope_cache, *_ = kvsp_manager.ag_pages(nope_cache)
                rope_cache, *_ = kvsp_manager.ag_pages(rope_cache)
                ki_cache, blk_table, _ = kvsp_manager.ag_pages(ki_cache)
                # replace local-cache with full-cache
                kv_cache = (nope_cache, rope_cache, ki_cache)
                ki_scale_cache = None

            topk_idx = self.indexer._apply_lightning_indexer(
                wi,
                qi,  # TND
                ki_cache,  # [*, pg, 1, D]
                q_cumlens,
                kv_lens,  # [B]
                blk_table,  # [B, *]
                ki_scale_cache,
                metadata_caller="decode",
            )  # -> int32 [T, 1, K]
        else:
            topk_idx = topk_indices_buffer[:x.shape[0]]

        next_topk_indices_buffer = (
            topk_idx if topk_idx is not None else topk_indices_buffer
        )
        topk_idx = self._apply_sink_offset(topk_idx)

        out = self._apply_attn_absorb(
            q_nope=q_nope,
            q_pe=q_pe,  # TND
            q_cumlens=q_cumlens,
            kv_lens=kv_lens,  # [B]
            topk_idx=topk_idx,  # [T, 1, K]
            block_table=blk_table,  # [B, *]
            kv_cache=kv_cache,
            attn_metadata=attn_metadata,
        )  # -> TND

        return (
            self._decode_attn_epilog(out, get_mome_args),
            next_topk_indices_buffer,
        )

    def _forward_decode_multistream(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        attn_metadata: NPUDSAMetadata,
        kv_cache: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        topk_indices_buffer: torch.Tensor | None = None,
        pd_mixed_flag: int = 0,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Overlap attention Q/KV-cache work with the complete indexer path.

        Event DAG, one event per cross-stream dependency:

        - ``input_ready`` (main -> side): x / cos / sin ready for side-stream ops.
        - ``q_lora_ready`` (main -> side): q_lora ready for the indexer query path.
        - ``indexer_kw_ready`` (main -> side): indexer_k / indexer_weights ready.
        - ``kv_ready`` (side -> main): kv ready for ``_kv_norm_rope_cache``.
        - ``side_done`` (side -> main): topk_idx ready for the decode epilog.
        """

        q_cumlens = attn_metadata.query_cumlens.to(torch.int32)
        kv_lens = attn_metadata.seq_lens.to(torch.int32)
        block_table = attn_metadata.block_table
        ki_cache, ki_scale_cache = self._get_li_caches(kv_cache)

        def get_mome_args():
            if self.noncontiguous_kv:
                return {}
            return {
                "force_decode": pd_mixed_flag == 1,
                "short_prefill": pd_mixed_flag == 2,
            }

        main_stream = torch.npu.current_stream()
        side_stream = self.side_stream
        input_ready = torch.npu.Event()
        input_ready.record()  # main -> side
        x.record_stream(side_stream)
        cos.record_stream(side_stream)
        sin.record_stream(side_stream)

        with torch.npu.npugraph_ex.scope.limit_core_num(*_MULTISTREAM_PROLOG_CORE_NUM):
            q_lora = self.q_a_proj(x)[0]
            q_lora = self._maybe_mome_q(q_lora, get_mome_args)
            q_lora = self.q_a_layernorm(q_lora)
            q_lora_ready = torch.npu.Event()
            q_lora_ready.record()  # main -> side
            q_lora.record_stream(side_stream)

            indexer_k = self.indexer.wk(x)[0]
            indexer_k = self.indexer.k_norm(indexer_k).view(-1, 1, self.indexer.head_dim)
            indexer_weights = self.indexer.weights_proj(x)[0]
            if model_extra_config.operator_opt_config.enable_precision_strong_consistency:
                indexer_weights = (
                    indexer_weights * self.indexer.weights_scale * self.indexer.softmax_scale
                )
            indexer_kw_ready = torch.npu.Event()
            indexer_kw_ready.record()  # main -> side
            indexer_k.record_stream(side_stream)
            indexer_weights.record_stream(side_stream)

        kv_ready = torch.npu.Event()

        with torch.npu.stream(side_stream):
            input_ready.wait(side_stream)  # side waits for main's x / cos / sin
            with torch.npu.npugraph_ex.scope.limit_core_num(*_MULTISTREAM_KV_INDEXER_CORE_NUM):
                kv = self.kv_a_proj_with_mqa(x)[0]
                kv = self._maybe_mome_kv(kv, get_mome_args)
                kv.record_stream(main_stream)
                kv_ready.record()  # side -> main

                q_lora_ready.wait(side_stream)  # side waits for main's q_lora
                indexer_q = self.indexer.wq_b(q_lora)[0].view(
                    -1, self.indexer.n_head, self.indexer.head_dim
                )
                indexer_q = self.indexer._apply_rope(indexer_q, cos, sin)

                indexer_kw_ready.wait(side_stream)
                indexer_k = self.indexer._apply_rope(indexer_k, cos, sin)
                self.indexer._update_cache(indexer_k, attn_metadata, ki_cache, ki_scale_cache)
            with torch.npu.npugraph_ex.scope.limit_core_num(*_MULTISTREAM_INDEXER_CORE_NUM):
                topk_idx = self.indexer._apply_lightning_indexer(
                    indexer_weights, indexer_q, ki_cache,
                    q_cumlens, kv_lens, block_table, ki_scale_cache,
                    metadata_caller="decode",
                )
            side_done = torch.npu.Event()
            side_done.record()  # side -> main

        with torch.npu.npugraph_ex.scope.limit_core_num(*_MULTISTREAM_Q_ABSORB_CORE_NUM):
            if self.split_q_up_in_multistream:
                q_nope = self.q_b_nope_proj(q_lora)[0].view(
                    -1, self.num_local_heads, self.qk_nope_head_dim
                )
                q_pe = self.q_b_pe_proj(q_lora)[0].view(
                    -1, self.num_local_heads, self.qk_rope_head_dim
                )
                q_nope = self._q_nope_absorb(q_nope)
                q_pe = self._apply_rope(q_pe, cos, sin)
            else:
                q_nope, q_pe = self._q_absorb(q_lora, cos, sin)
            kv_ready.wait(main_stream)  # main waits for side's kv
            self._kv_norm_rope_cache(kv, cos, sin, attn_metadata, kv_cache)
        side_done.wait(main_stream)  # main waits for side's topk_idx
        topk_idx.record_stream(main_stream)

        return self._finish_multistream_decode(
            q_nope, q_pe, topk_idx, q_cumlens, kv_lens, block_table,
            kv_cache, attn_metadata, get_mome_args,
        )

    def _finish_multistream_decode(
        self,
        q_nope: torch.Tensor,
        q_pe: torch.Tensor,
        topk_idx: torch.Tensor,
        q_cumlens: torch.Tensor,
        kv_lens: torch.Tensor,
        block_table: torch.Tensor,
        kv_cache: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        attn_metadata: NPUDSAMetadata,
        get_mome_args: Callable[[], dict[str, bool]],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply the common top-k bookkeeping and attention decode epilog."""
        next_topk_indices_buffer = topk_idx
        topk_idx = self._apply_sink_offset(topk_idx)

        out = self._apply_attn_absorb(
            q_nope=q_nope,
            q_pe=q_pe,
            q_cumlens=q_cumlens,
            kv_lens=kv_lens,
            topk_idx=topk_idx,
            block_table=block_table,
            kv_cache=kv_cache,
            attn_metadata=attn_metadata,
        )
        return (
            self._decode_attn_epilog(out, get_mome_args),
            next_topk_indices_buffer,
        )

    def _forward_decode_multistream_a5(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        attn_metadata: NPUDSAMetadata,
        kv_cache: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        get_mome_args: Callable[[], dict[str, bool]],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Pack the A5 attention prolog across three streams before LI.

        时间从上到下，三条流上的执行及大致掩盖关系如下。括号内为
        ``limit_core_num(Cube, Vector)``；横向相邻的算子可并行执行。

        .. code-block:: text

            main stream                 LI side stream              KV/weight side stream
            ------------------------    -------------------------   -------------------------
            q_a_proj (24, 48)       |   wk (8, 32)                  wait wk, weights_proj (8, 32)
            q_mome + q_norm (8, 32) |   k_norm/rope/cache (8, 32)   wait q_a_proj, kv_a_proj (32, 32)
            wq_b (24, 48)           |   q_b_pe if split (8, 16)     kv_mome
            q_b_nope/q_b (32, 32)   |   indexer_q rope (16, 32)     kv_norm/rope/cache
            q_nope absorb           |   q_pe rope (16, 32)          (16, 32)
                    \\_________________________|___________________________/
                                              |
                                  lightning indexer (32, 64)
                                              |
                                      attention + epilog

        ``q_a_proj`` 与 ``wk`` 首先并行；随后两条侧流主要掩盖主流上的
        Q/MoME、Q up 和 absorb 路径。LI 是这段 prolog 的关键路径汇合点，
        因此 main stream 在 LI 前等待两个侧流的最终 event，使 LI 独占完整
        Cube/Vector 资源，避免与仍在执行的 vector/cache 算子争核。
        """
        assert self.on_ascend950
        assert self.li_stream is not None

        q_cumlens = attn_metadata.query_cumlens.to(torch.int32)
        kv_lens = attn_metadata.seq_lens.to(torch.int32)
        block_table = attn_metadata.block_table
        ki_cache, ki_scale_cache = self._get_li_caches(kv_cache)

        main_stream = torch.npu.current_stream()
        side_stream = self.side_stream
        li_stream = self.li_stream

        input_ready = torch.npu.Event()
        wk_done = torch.npu.Event()
        q_a_cube_done = torch.npu.Event()
        q_side_ready = torch.npu.Event()
        wq_b_ready = torch.npu.Event()
        q_pe_ready = torch.npu.Event()
        attn_kv_ready = torch.npu.Event()

        input_ready.record()
        for stream in (side_stream, li_stream):
            x.record_stream(stream)
            cos.record_stream(stream)
            sin.record_stream(stream)

        # q_a -> q_b_nope/q_b -> absorb is the longest attention-Q path and
        # stays on main. wk starts concurrently on the LI stream.
        with torch.npu.npugraph_ex.scope.limit_core_num(24, 48):
            q_lora = self.q_a_proj(x)[0]
        q_a_cube_done.record()
        with torch.npu.npugraph_ex.scope.limit_core_num(8, 32):
            q_lora = self._maybe_mome_q(q_lora, get_mome_args)
            q_lora = self.q_a_layernorm(q_lora)

        if self.split_q_up_in_multistream:
            q_side_ready.record()
            q_lora.record_stream(li_stream)
        with torch.npu.npugraph_ex.scope.limit_core_num(24, 48):
            indexer_q = self.indexer.wq_b(q_lora)[0].view(
                -1, self.indexer.n_head, self.indexer.head_dim
            )
        indexer_q.record_stream(li_stream)
        wq_b_ready.record()
        with torch.npu.npugraph_ex.scope.limit_core_num(32, 32):
            if self.split_q_up_in_multistream:
                q_nope = self.q_b_nope_proj(q_lora)[0].view(
                    -1, self.num_local_heads, self.qk_nope_head_dim
                )
            else:
                q_nope, q_pe = self._project_and_split_q(q_lora)
                q_pe.record_stream(li_stream)
                q_side_ready.record()
            q_nope = self._q_nope_absorb(q_nope)

        with torch.npu.stream(li_stream):
            input_ready.wait(li_stream)
            with torch.npu.npugraph_ex.scope.limit_core_num(8, 32):
                indexer_k = self.indexer.wk(x)[0]
                wk_done.record()
                indexer_k = self.indexer.k_norm(indexer_k).view(
                    -1, 1, self.indexer.head_dim
                )
                indexer_k = self.indexer._apply_rope(indexer_k, cos, sin)
                self.indexer._update_cache(indexer_k, attn_metadata, ki_cache, ki_scale_cache)
            q_side_ready.wait(li_stream)
            if self.split_q_up_in_multistream:
                with torch.npu.npugraph_ex.scope.limit_core_num(8, 16):
                    q_pe = self.q_b_pe_proj(q_lora)[0].view(
                        -1, self.num_local_heads, self.qk_rope_head_dim
                    )
            wq_b_ready.wait(li_stream)
            with torch.npu.npugraph_ex.scope.limit_core_num(16, 32):
                indexer_q = self.indexer._apply_rope(indexer_q, cos, sin)
                q_pe = self._apply_rope(q_pe, cos, sin)
            indexer_q.record_stream(main_stream)
            q_pe.record_stream(main_stream)
            q_pe_ready.record()

        with torch.npu.stream(side_stream):
            wk_done.wait(side_stream)
            with torch.npu.npugraph_ex.scope.limit_core_num(32, 32):
                indexer_weights = self.indexer.weights_proj(x)[0]
                if (
                    model_extra_config.operator_opt_config.enable_precision_strong_consistency
                ):
                    indexer_weights = (
                        indexer_weights
                        * self.indexer.weights_scale
                        * self.indexer.softmax_scale
                    )
                q_a_cube_done.wait(side_stream)
                kv = self.kv_a_proj_with_mqa(x)[0]
            indexer_weights.record_stream(main_stream)
            with torch.npu.npugraph_ex.scope.limit_core_num(16, 32):
                kv = self._maybe_mome_kv(kv, get_mome_args)
                self._kv_norm_rope_cache(
                    kv, cos, sin, attn_metadata, kv_cache
                )
            attn_kv_ready.record()

        q_pe_ready.wait(main_stream)
        attn_kv_ready.wait(main_stream)
        topk_idx = self.indexer._apply_lightning_indexer(
            indexer_weights, indexer_q, ki_cache,
            q_cumlens, kv_lens, block_table, ki_scale_cache,
            metadata_caller="decode",
        )

        return self._finish_multistream_decode(
            q_nope, q_pe, topk_idx, q_cumlens, kv_lens, block_table,
            kv_cache, attn_metadata, get_mome_args,
        )


@eager_break_during_capture
@maybe_transfer_kv_layer
def npu_dsa_forward(
    hs: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    module_name: str,
    topk_indices_buffer: torch.Tensor,
    layer_name: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    forward_context = get_forward_context()
    self: NPUDeepseekSparseAttention = forward_context.no_compile_layers[module_name]
    attn_metadata = forward_context.attn_metadata
    if isinstance(attn_metadata, dict):
        attn_metadata = attn_metadata[layer_name]
    kv_cache = self.attn.kv_cache if attn_metadata else None
    p_slice, d_slice, has_prefill, has_decode = get_batch_desc(
        attn_metadata, self.layer_idx
    )
    prefill = getattr(attn_metadata, "prefill", None)  # None for dummy_run
    decode = getattr(attn_metadata, "decode", None)  # None for dummy_run

    if self.param_sink_number > 0 and not self.noncontiguous_kv:
        assert self.attn.sink_k_pe is not None and self.attn.sink_compressed_kv is not None, (
            "sink_k_pe and sink_compressed_kv have not been prepared"
        )
        if not self.attn.sink_populated and kv_cache is not None:
            self.attn.populate_sink_kv(kv_cache[0], kv_cache[1])

    if has_decode:
        sp_enabled = self.ena_sp
        if sp_enabled:
            topk_indices_buffer = get_tp_group().all_gather(topk_indices_buffer, dim=0)
        with sp_disabled(self, hs) as (x, y, out):
            pd_mixed_flag = 0
            if has_prefill:
                y[p_slice], topk_indices_buffer[p_slice] = self._forward_prefill(
                    x[p_slice],
                    cos[p_slice],
                    sin[p_slice],
                    prefill,
                    kv_cache,
                    topk_indices_buffer[p_slice],
                    pd_mixed_flag=True,
                )
                pd_mixed_flag = (
                    2 if attn_metadata.num_decode_tokens > attn_metadata.num_decodes
                    else 1
                )
            y[d_slice], topk_indices_buffer[d_slice] = self._forward_decode(
                x[d_slice],
                cos[d_slice],
                sin[d_slice],
                decode,
                kv_cache,
                topk_indices_buffer[d_slice],
                pd_mixed_flag,
            )
        if sp_enabled:
            topk_indices_buffer = topk_indices_buffer.split(hs.size(0))[
                get_tp_group().rank_in_group
            ]
    else:
        out = lazy_zero_like(hs)
        if (
            self.ena_cp
            and prefill is not None
            and getattr(attn_metadata, "prefill_graph_width", None) is not None
        ):
            # The builder laid this batch out for a FULL graph and left out the
            # CP metadata, whose layout a graph cannot follow. That needs the
            # whole sequence: run prefill with SP disabled. Query-parallel
            # prefill selects zigzag fragments and keeps TopK indices local;
            # otherwise gather/split indices as in the decode branch.
            use_local_query = (
                self.enable_prefill_graph_token_parallel
                and attn_metadata.num_actual_tokens % (2 * self.tp_size) == 0
            )
            sp_enabled = self.ena_sp
            output_sp_slice = _prefill_o_proj_sp_slice(self, attn_metadata.num_actual_tokens)
            if sp_enabled and not use_local_query:
                topk_indices_buffer = get_tp_group().all_gather(
                    topk_indices_buffer, dim=0
                )
            if output_sp_slice is not None:
                # TP1 projection already returns this rank's SP rows. Keep
                # the ordinary full-output sp_disabled contract unchanged.
                self.ena_sp = False
                try:
                    x = get_tp_group().all_gather(hs, dim=0)
                    out[:], next_topk_indices = self._forward_prefill(
                        x[p_slice],
                        cos[p_slice],
                        sin[p_slice],
                        prefill,
                        kv_cache,
                        topk_indices_buffer if use_local_query else topk_indices_buffer[p_slice],
                        use_local_query=use_local_query,
                        output_sp_slice=output_sp_slice,
                    )
                finally:
                    self.ena_sp = sp_enabled
            else:
                with sp_disabled(self, hs) as (x, y, out):
                    y[p_slice], next_topk_indices = self._forward_prefill(
                        x[p_slice],
                        cos[p_slice],
                        sin[p_slice],
                        prefill,
                        kv_cache,
                        topk_indices_buffer if use_local_query else topk_indices_buffer[p_slice],
                        use_local_query=use_local_query,
                    )
            if use_local_query:
                topk_indices_buffer = next_topk_indices
            else:
                topk_indices_buffer[p_slice] = next_topk_indices
            if sp_enabled and not use_local_query:
                topk_indices_buffer = topk_indices_buffer.split(hs.size(0))[
                    get_tp_group().rank_in_group
                ]
        elif self.ena_cp:
            out[:], topk_indices_buffer = self._forward_prefill_cp(
                hs,
                cos,
                sin,
                prefill,
                kv_cache,
                topk_indices_buffer
            )
        elif self.ena_sp:
            out[:], topk_indices_buffer = self._forward_prefill(
                hs,
                cos,
                sin,
                prefill,
                kv_cache,
                topk_indices_buffer
            )
        else:
            out[p_slice], topk_indices_buffer = self._forward_prefill(
                hs[p_slice],
                cos[p_slice],
                sin[p_slice],
                prefill,
                kv_cache,
                topk_indices_buffer[p_slice],
            )
    return out.tensor(), topk_indices_buffer


def npu_dsa_forward_fake(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    module_name: str,
    topk_indices_buffer: torch.Tensor,
    layer_name: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    return torch.empty_like(x), torch.empty_like(topk_indices_buffer)


def npu_dsa_forward_mhc_deferred(
    hidden_states: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    residual: torch.Tensor,
    module_name: str,
    mhc_layer_name: str,
    task_key: str,
    layer_name: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Launch next-layer MHC work between attention and the DSA epilog."""
    layer = get_forward_context().no_compile_layers[module_name]
    topk_indices_buffer = torch.zeros(
        (hidden_states.shape[0], 1, layer.index_topk),
        dtype=torch.int32, device=hidden_states.device,
    )
    return call_attention_mhc_deferred(
        lambda x, cos, sin, module, kv_layer: npu_dsa_forward(
            x, cos, sin, module, topk_indices_buffer, kv_layer
        )[0],
        hidden_states,
        cos,
        sin,
        residual,
        module_name,
        mhc_layer_name,
        task_key,
        layer_name,
    )


direct_register_custom_op(
    op_name="npu_dsa_forward_mhc_deferred",
    op_func=npu_dsa_forward_mhc_deferred,
    mutates_args=[],
    fake_impl=attention_mhc_deferred_fake,
    dispatch_key="PrivateUse1",
)

direct_register_custom_op(
    op_name="npu_dsa_forward",
    op_func=npu_dsa_forward,
    mutates_args=[],
    fake_impl=npu_dsa_forward_fake,
    dispatch_key="PrivateUse1",
)
