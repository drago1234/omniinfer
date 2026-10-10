# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.

from types import SimpleNamespace

import torch
from transformers import PretrainedConfig

from vllm.config import CacheConfig, VllmConfig
from vllm.forward_context import get_forward_context
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.utils.torch_utils import direct_register_custom_op, get_dtype_size
from vllm.distributed import get_tp_group

from omni_npu.layers.attention.npu_sparse_attentions import MLASWAAttention
from omni_npu.attention.backends.mla import NPUMLAImpl
from omni_npu.attention.backends.utils import (
    CrossLayerSharedOp,
    SPManager,
    cache_fit_shape,
    get_batch_desc,
    sp_disabled,
)
from omni_npu.model_config.config_loader.loader import model_extra_config
from omni_npu.v1.layers.attention.npu_dsa import NPUDeepseekSparseAttention
from omni_npu.v1.layers.attention.npu_mla import NPUDeepseekMLAAttention, NPUMLAAttention
from omni_npu.v1.layers.linear import ColumnParallelFlashCommLinear, ReplicatedFlashCommLinear
from omni_npu.v1.layers.utils import get_cla_mapping

# Tiling metadata for the local window branch, shared across a step's CLA layers:
# the first of them fills the buffer for its key, the rest read it.
npu_fused_infer_attention_sink_metadata: CrossLayerSharedOp | None = None


class NPUMLASWAAttention(NPUMLAAttention, MLASWAAttention):
    """A sliding-window cache page plus the NPU-derived absorb weights.

    MLASWAAttention sizes the page for a window, and NPUMLAAttention reads
    W_UK_T/W_UV off a FlashComm kv_b_proj, which the vLLM base cannot. The local
    branch's kv_b_proj_swa is a FlashComm linear, so it needs both.
    """


class CLAOutputGate(torch.nn.Module):
    """Apply a per-query-head sigmoid gate to one CLA attention branch."""

    def __init__(
        self,
        hidden_size: int,
        num_heads: int, num_local_heads: int,
        quant_config: QuantizationConfig | None,
        prefix: str,
        disable_tp: bool,
    ) -> None:
        super().__init__()
        self.num_local_heads = num_local_heads
        self.w_gate = ColumnParallelFlashCommLinear(
            hidden_size, num_heads,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.w_gate",
            return_bias=False,
            disable_tp=disable_tp,
        )

    def forward(self, hidden_states: torch.Tensor, attention_output: torch.Tensor) -> torch.Tensor:
        gate = torch.sigmoid(self.w_gate(hidden_states))
        return self._apply_gate(gate, attention_output)

    def forward_cp(
        self,
        hidden_states: torch.Tensor,  # TD, sp
        attention_output: torch.Tensor,  # [T, N*V], cp
        sp_manager: SPManager,
    ) -> torch.Tensor:
        gate = torch.sigmoid(sp_manager.sp_to_cp(self.w_gate(hidden_states)))
        return self._apply_gate(gate, attention_output)

    def _apply_gate(self, gate: torch.Tensor, attention_output: torch.Tensor) -> torch.Tensor:
        if gate.shape[-1] != self.num_local_heads:
            head_start = get_tp_group().rank_in_group * self.num_local_heads
            gate = gate.narrow(-1, head_start, self.num_local_heads)
        value_dim = attention_output.shape[-1] // self.num_local_heads
        output = attention_output.view(-1, self.num_local_heads, value_dim)
        return (output * gate.unsqueeze(-1)).flatten(1)


class CLAReuseLayerMixin:
    """Turn a plain attention layer into a CLA reuse layer.

    The base class keeps owning the global branch; this mixin redirects that
    branch at the source layer's cache, adds the local sliding-window branch
    and combines the two through the learned gates.
    """

    def __init__(
        self,
        vllm_config: VllmConfig,
        config: PretrainedConfig,
        *,
        cache_config: CacheConfig | None = None,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
        **kwargs,
    ) -> None:
        super().__init__(
            vllm_config,
            config,
            cache_config=cache_config,
            quant_config=quant_config,
            prefix=prefix,
            **kwargs,
        )
        self._init_cla_reuse(config, prefix, cache_config, quant_config)

    def _init_cla_reuse(
        self,
        config: PretrainedConfig,
        prefix: str,
        cache_config: CacheConfig | None,
        quant_config: QuantizationConfig | None,
    ) -> None:
        mapping = get_cla_mapping(config)
        if self.layer_idx not in mapping:
            raise ValueError(
                f"Layer {self.layer_idx} is not a CLA reuse layer; "
                f"cla_explicit_mapping covers {sorted(mapping)}."
            )
        if self.ena_cp:
            # Each rank attends its zigzag fragments with every head, as npu_dsa's CP prefill does.
            if self.q_b_proj.tp_size != 1:
                raise ValueError("CLA context parallelism requires q_b_proj.tp_size == 1 (all query heads).")
            if self.kv_b_proj.tp_size != 1:
                raise ValueError("CLA context parallelism requires kv_b_proj.tp_size == 1 (all KV heads).")
            if self.ena_kvsp:
                raise NotImplementedError("CLA does not gather the cache pages that KV sequence parallelism splits.")
        self.cla_source_layer_idx = mapping[self.layer_idx]
        # The first reuse layer refreshes the metadata its peers then read.
        self.is_cla_fa_metadata_producer = self.layer_idx == min(mapping)

        window = getattr(config, "cla_swa_gate_window", None)
        if window is None:
            raise ValueError("cla_swa_gate_window is required to size a CLA layer's local sliding-window branch.")
        self.cla_swa_gate_window = window

        # The checkpoint stores an empty kv_a_proj_with_mqa for a reuse layer,
        # so drop the modules the base class built for the global KV it never
        # projects. The model's loader skips weights with no parameter.
        del self.kv_a_proj_with_mqa
        del self.kv_a_layernorm

        layers_prefix = prefix.split(".layers.", 1)[0]
        # The global branch reads the source layer's latent/RoPE cache. vLLM's
        # cache binder picks this up and allocates no cache for this layer.
        self.attn.kv_sharing_target_layer_name = f"{layers_prefix}.layers.{self.cla_source_layer_idx}.self_attn.attn"

        self.kv_a_proj_with_mqa_swa = ReplicatedFlashCommLinear(
            self.hidden_size,
            self.kv_lora_rank + self.qk_rope_head_dim,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.kv_a_proj_with_mqa_swa",
        )
        self.kv_a_layernorm_swa = RMSNorm(self.kv_lora_rank, eps=config.rms_norm_eps, dtype=self.dtype)
        disable_tp = self.kv_b_proj.tp_size == 1
        self.kv_b_proj_swa = ColumnParallelFlashCommLinear(
            self.kv_lora_rank,
            self.num_heads * (self.qk_nope_head_dim + self.v_head_dim),
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.kv_b_proj_swa",
            return_bias=False,
            disable_tp=disable_tp,
        )
        self.swa_gate_global = CLAOutputGate(
            self.hidden_size, self.num_heads, self.num_local_heads,
            quant_config, f"{prefix}.swa_gate_global", disable_tp,
        )
        self.swa_gate_local = CLAOutputGate(
            self.hidden_size, self.num_heads, self.num_local_heads,
            quant_config, f"{prefix}.swa_gate_local", disable_tp,
        )

        # The cache binder keys one KV cache per layer index, but this decoder
        # layer owns two attentions, so the local one is registered past the real
        # layers and the MTP ones. Its weights are unaffected: they stay under
        # self_attn.*_swa.
        virtual_layer_idx = config.num_hidden_layers + getattr(config, "num_nextn_predict_layers", 0) + self.layer_idx
        self.cla_swa_attn_name = f"{layers_prefix}.cla_swa_layers.{virtual_layer_idx}.attn"
        swa_attn_kwargs = self._inner_mla_attn_kwargs(cache_config, quant_config, prefix)
        swa_attn_kwargs.update(
            prefix=self.cla_swa_attn_name,
            kv_b_proj=self.kv_b_proj_swa,
            sliding_window=window,
            page_size_padded=self._cla_swa_page_size_padded(config, cache_config),
            num_extra_reserved_blocks=(
                model_extra_config.operator_opt_config.num_extra_reserved_blocks
            ),
        )
        self.attn_cla_swa = NPUMLASWAAttention(**swa_attn_kwargs)

        # Both branches carry their own GPT-OSS sink.
        self._init_gpt_oss_swa_sink()

        global npu_fused_infer_attention_sink_metadata
        callers = tuple(
            self._cla_metadata_key(caller)
            for caller in ("decode", "prefill_absorb", "prefill", "prefill_absorb_cp")
        )
        if npu_fused_infer_attention_sink_metadata is None:
            npu_fused_infer_attention_sink_metadata = CrossLayerSharedOp(
                op=torch.ops.custom._npu_fused_infer_attention_sink_metadata,
                shape=(1024,),
                dtype=torch.int32,
                callers=callers,
            )
        if self.use_gpt_oss_sink and not self.use_gpt_oss_sink_rescale:
            self._reserve_gpt_oss_fia_metadata(callers)

    @staticmethod
    def _cla_swa_page_size_padded(config, cache_config) -> int | None:
        """Pad the local cache's page to the largest page in its group.

        A group's caches are carved out of one allocation sized by the group's
        page. Where DSA layers exist theirs is the largest -- it also holds the
        indexer key -- so the local sliding-window cache has to declare that
        same page, or its reshape is handed a tensor sized for the other one.
        """
        if (getattr(config, "index_topk", 0) or 0) <= 0:
            return None
        cache_dtype_str = cache_config.cache_dtype
        if cache_dtype_str not in (None, "auto"):
            raise NotImplementedError(
                f"CLA does not size its local cache page for a quantized KV "
                f"cache yet (cache_dtype={cache_dtype_str!r})."
            )
        head_size = config.kv_lora_rank + config.qk_rope_head_dim + config.index_head_dim
        return cache_config.block_size * head_size * get_dtype_size(model_extra_config.dtype)

    def _write_local_kv(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        local_slots,
        sp_manager: SPManager | None = None,
    ) -> None:
        """Project this layer's own KV down and write the local window cache."""
        local_kv = self.kv_a_proj_with_mqa_swa(x)[0]  # TD
        if sp_manager is not None:
            # The slots and cos/sin cover the whole step in token order, so the
            # rows have to as well; a rank's own slice would land in its peers' slots.
            local_kv = sp_manager.ag_tokens(local_kv)  # TD, sp -> all tokens
        self._kv_norm_rope_cache(
            local_kv,
            cos,
            sin,
            local_slots,
            self.attn_cla_swa.kv_cache,
            kv_a_layernorm=self.kv_a_layernorm_swa,
            **self._local_kv_layout_kwargs(),
        )

    def _local_kv_layout_kwargs(self) -> dict:
        """Where the local cache's layout differs from what the base assumes.

        The local window cache is a plain pair of latent and RoPE tensors, which
        is already what a dense MLA layer's cache looks like. A DSA layer's is
        not, so the sparse CLA class says so.
        """
        return {}

    def _cla_metadata_key(self, caller: str) -> tuple[str, int]:
        """Key this branch's share of the tiling buffer.

        The plain layers' window is 512 as well, so without the prefix these
        keys would be theirs and the buffer would hold tiling built for the
        full-context cache.
        """
        return f"cla_{caller}", self.cla_swa_gate_window - 1

    def _apply_local_swa(
        self,
        q_nope_raw: torch.Tensor,  # TND, shared with the global branch
        q_pe: torch.Tensor,  # TND, already roped
        attn_metadata_cla_swa,
        metadata_caller: str,
        num_actual_tokens: int,
    ) -> torch.Tensor:  # [T, N*V]
        """Attend over the local window cache and project with its own W_UV.

        The same banded attention the plain layers run, against this layer's own
        cache, window and sink. Kept as its own copy: the two differ in the
        metadata buffer, the producer flag and which sink they apply.
        """
        q_nope = self._q_nope_absorb(q_nope_raw, self.attn_cla_swa.W_UK_T)  # TND
        window_size = self.cla_swa_gate_window - 1
        nope_cache, rope_cache = self.attn_cla_swa.kv_cache
        block_size = nope_cache.size(1)
        nope_cache = cache_fit_shape(nope_cache, "3D")
        rope_cache = cache_fit_shape(rope_cache, "3D")
        NPUMLAImpl.ensure_decode_attn_mask()
        q_head = q_nope.size(1)
        num_tokens = q_nope.size(0)
        # A captured decode batch is padded; slicing by q_cumlens[-1] would read a
        # device tensor and break capture, so use the host-side count.
        valid_tok = num_actual_tokens
        q_cumlens = NPUDeepseekMLAAttention._lengths_to_i64(attn_metadata_cla_swa.query_cumlens, q_nope.device)
        kv_lens = NPUDeepseekMLAAttention._lengths_to_i64(attn_metadata_cla_swa.seq_lens, q_nope.device)
        caller = self._cla_metadata_key(metadata_caller)
        kwargs = {
            "query": q_nope[:valid_tok],
            "key": nope_cache,
            "value": nope_cache,
            "query_rope": q_pe[:valid_tok],
            "key_rope": rope_cache,
            "num_query_heads": q_head, "num_key_value_heads": 1,
            "input_layout": "TND_NTD",
            "atten_mask": NPUMLAImpl.SHARE_MASK_TRIL_SPARSE,
            "sparse_mode": 4,
            "pre_tokens": window_size,
            "next_tokens": 0,
            "softmax_scale": self.scaling,
            "block_table": attn_metadata_cla_swa.block_table,
            "block_size": block_size,
            "actual_seq_qlen": q_cumlens,
            "actual_seq_kvlen": kv_lens,
        }

        if self.use_gpt_oss_sink:
            kwargs["batch_invariant"] = model_extra_config.operator_opt_config.use_batch_invariant_op
        if self.use_gpt_oss_sink and not self.use_gpt_oss_sink_rescale:
            # The kernel applies the sink itself and builds its own metadata.
            local_out = self._apply_gpt_oss_fia_native(
                kwargs, self.gpt_oss_sink_swa, caller, self.is_cla_fa_metadata_producer,
            )
        else:
            stream_limit = torch.npu.get_stream_limit(torch.npu.current_stream())
            kwargs["meta_data"] = npu_fused_infer_attention_sink_metadata(
                {
                    "batch_invariant": kwargs.get("batch_invariant", False),
                    "num_heads_q": q_head,
                    "num_heads_kv": 1,
                    "head_dim_qk": q_nope.shape[-1],
                    "head_dim_v": nope_cache.shape[-1],
                    "actual_seq_lengths": q_cumlens, "actual_seq_lengths_kv": kv_lens,
                    "sparse_mode": 4,
                    "pre_tokens": window_size, "next_tokens": 0,
                    "input_layout": "TND", "input_layout_kv": "BnBsH",
                    "rope_head_dim": q_pe.shape[-1],
                    "k_sink_num": 0,
                    "block_size": block_size,
                    "aic_core_num": stream_limit["cube_core_num"],
                    "aiv_core_num": stream_limit["vector_core_num"],
                },
                self.is_cla_fa_metadata_producer,
                caller,
            )
            kwargs["return_softmax_lse"] = self.use_gpt_oss_sink
            local_out, lse = torch.ops.custom.npu_fused_infer_attention_sink(**kwargs)
            if self.use_gpt_oss_sink:
                local_out = self._rescale_gpt_oss_sink_output(
                    local_out, lse, self.gpt_oss_sink_swa, "NTD"
                )

        if local_out.size(1) != num_tokens:  # pad the captured tail back on
            padded = q_nope.new_zeros(q_head, num_tokens, self.kv_lora_rank)  # NTD
            padded[:, :valid_tok, :] = local_out
            local_out = padded
        # NTD from the op, TND for the absorb epilog.
        return self._post_attn_absorb(local_out.transpose(0, 1), self.attn_cla_swa.W_UV)

    def _cla_global_attention(
        self,
        q_nope_raw: torch.Tensor,  # TND, shared with the local branch
        q_pe: torch.Tensor,  # TND, already roped
        attn_metadata,
        metadata_caller: str,
        topk_indices: torch.Tensor,
    ) -> torch.Tensor:
        """Attend over the source layer's cache and project with W_UV.

        Implemented per base class: sparsely for a DSA layer, densely for any
        other. Returns [T, N*V].
        """
        raise NotImplementedError

    def _combine_cla_branches(
        self,
        x: torch.Tensor,
        global_output: torch.Tensor,  # [T, N*V]
        local_output: torch.Tensor,  # [T, N*V]
    ) -> torch.Tensor:
        return self.swa_gate_global(x, global_output) + self.swa_gate_local(x, local_output)

    def forward(  # type: ignore[override]
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        topk_indices_buffer: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        out = torch.ops.vllm.npu_cla_forward(x, cos, sin, self.prefix, topk_indices_buffer)
        return out, topk_indices_buffer

    def forward_cla(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        attn_metadata,
        attn_metadata_cla_swa,
        metadata_caller: str,
        local_slots,
        topk_indices: torch.Tensor,
    ) -> torch.Tensor:
        """Global branch over the shared cache, local window branch, then gate.

        Under SP, x is this rank's slice of the step while cos/sin cover all of
        it: both branches attend over every token, so each projection is gathered.
        """
        if metadata_caller == "decode" and self.ena_sp:
            raise RuntimeError("CLA decode requires sequence parallelism to be disabled via sp_disabled.")
        sp_manager = SPManager.init_sp(tok=cos.size(0)) if self.ena_sp else None
        q_lora = self.q_a_proj(x)[0]  # TD
        if sp_manager:
            q_lora = sp_manager.ag_tokens(q_lora)  # TD
        q_lora = self.q_a_layernorm(q_lora)  # TD
        q_nope_raw, q_pe = self._q_project(q_lora, cos, sin)  # TND

        # Only the local branch has KV of its own to project and cache.
        self._write_local_kv(x, cos, sin, local_slots, sp_manager)

        global_output = self._cla_global_attention(q_nope_raw, q_pe, attn_metadata, metadata_caller, topk_indices)
        # Both branches cover the same tokens, and a captured decode batch is
        # padded, so take the unpadded count the metadata carries. Fall back to
        # the query rows, not x's: under SP x holds only this rank's slice.
        num_actual_tokens = getattr(attn_metadata, "num_tokens", None) or q_nope_raw.size(0)
        local_output = self._apply_local_swa(
            q_nope_raw, q_pe, attn_metadata_cla_swa, metadata_caller, num_actual_tokens
        )
        # The gates weigh every token the branches produced. Gather x, not the
        # logits: a rank's gate only scores its own heads.
        gate_x = sp_manager.ag_tokens(x) if sp_manager else x
        combined = self._combine_cla_branches(gate_x, global_output, local_output)
        # Both branches produce this rank's heads, and _maybe_mome_out is what
        # gathers them for an o_proj that wants them all. A CLA layer has no
        # MoME, so it reaches that gather through the same call.
        out = self._maybe_mome_out(combined, None)
        if sp_manager and self.o_proj.tp_size > 1:
            out = sp_manager.align_tokens(out)
        out = self._apply_o_proj(out)
        if sp_manager and out.size(0) != x.size(0):
            out = sp_manager.slice_tokens(out)
        return out


class NPUDeepseekSparseCLAAttention(CLAReuseLayerMixin, NPUDeepseekSparseAttention):
    """CLA reuse layer whose global branch selects sparsely (a DSA layer)."""

    def _local_kv_layout_kwargs(self) -> dict:
        # A DSA cache packs latent and RoPE into one tensor; this one does not.
        return {"packed_kv": False}

    def _cla_global_attention(
        self,
        q_nope_raw: torch.Tensor,
        q_pe: torch.Tensor,
        attn_metadata,
        metadata_caller: str,
        topk_indices: torch.Tensor,
    ) -> torch.Tensor:
        return self._cla_sparse_attention(
            q_nope_raw,
            q_pe,
            attn_metadata.query_cumlens.to(torch.int32),
            attn_metadata.seq_lens.to(torch.int32),
            attn_metadata.block_table,
            topk_indices,
            attn_metadata,
        )

    def _cla_sparse_attention(
        self,
        q_nope_raw: torch.Tensor,  # TND
        q_pe: torch.Tensor,  # TND, already roped
        q_cumlens: torch.Tensor,  # int32 [B], or [2B] per zigzag fragment
        kv_lens: torch.Tensor,  # int32, as q_cumlens
        block_table: torch.Tensor,
        topk_indices: torch.Tensor,
        attn_metadata,
    ) -> torch.Tensor:  # [T, N*V]
        q_nope = self._q_nope_absorb(q_nope_raw)  # TND, own W_UK_T
        topk_idx = self._apply_sink_offset(topk_indices)
        out = self._apply_attn_absorb(
            q_nope,
            q_pe,
            q_cumlens,
            kv_lens,
            topk_idx,
            block_table,
            self.attn.kv_cache,  # the source layer's cache
            attn_metadata=attn_metadata,
        )  # TND
        return self._post_attn_absorb(out)  # [T, N*V]

    def _forward_cla_prefill_cp(
        self,
        sp_x: torch.Tensor,  # TD, sp
        cos: torch.Tensor,  # BNSD, every token of the step
        sin: torch.Tensor,  # BNSD, every token of the step
        attn_metadata,
        attn_metadata_cla_swa,
        local_slots,
        topk_indices: torch.Tensor,  # CP layout, from the last full DSA layer
    ) -> torch.Tensor:  # TD, sp
        """forward_cla over this rank's zigzag CP fragments.

        Both branches attend with the CP query layout and lengths. The global
        one runs no indexer: as a shared DSA layer it takes the CP-layout top-k
        the last full layer left. The local one writes every token's KV, then
        attends its own cache with each request's pages repeated per fragment.
        """
        sp_manager: SPManager = attn_metadata.sp_manager
        q_cumlens, kv_lens, blk_table = self._cp_attention_metadata(sp_manager)
        cp_cos_sin = self._cp_rotary_rows(sp_manager, cos, sin)
        q_lora = self._cp_q_lora(sp_x, sp_manager, None)  # TD, cp
        q_nope_raw, q_pe = self._q_project(q_lora, *cp_cos_sin)  # TND, full head, cp

        self._write_local_kv(sp_x, cos, sin, local_slots, sp_manager)

        global_output = self._cla_sparse_attention(
            q_nope_raw, q_pe, q_cumlens, kv_lens, blk_table, topk_indices, attn_metadata
        )
        local_cp_meta = SimpleNamespace(
            query_cumlens=q_cumlens,
            seq_lens=kv_lens,
            block_table=SPManager.cp_block_table(attn_metadata_cla_swa.block_table),
        )
        local_output = self._apply_local_swa(
            q_nope_raw, q_pe, local_cp_meta, "prefill_absorb_cp", q_nope_raw.size(0)
        )
        combined = (
            self.swa_gate_global.forward_cp(sp_x, global_output, sp_manager)
            + self.swa_gate_local.forward_cp(sp_x, local_output, sp_manager)
        )
        return self._cp_o_proj(combined, sp_manager)


class NPUDeepseekMLACLAAttention(CLAReuseLayerMixin, NPUDeepseekMLAAttention):
    """CLA reuse layer whose global branch attends densely (a non-DSA layer)."""

    # Context parallelism splits DSA layers only; a dense host runs SP, as plain MLA layers do.
    ena_cp = False

    def _init_cla_reuse(self, config, prefix, cache_config, quant_config) -> None:
        super()._init_cla_reuse(config, prefix, cache_config, quant_config)
        # Non-absorbed MLA writes TND metadata, not the paged absorb tiling
        # this branch needs. Generate it here instead of reading an unwritten buffer.
        if not self.mla_absorb:
            self.is_fa_metadata_producer = True

    def _post_attn_absorb(self, out: torch.Tensor, w_uv: torch.Tensor | None = None) -> torch.Tensor:
        # CLA uses the sparse host's TND epilog contract; MLA's epilog takes NTD.
        return super()._post_attn_absorb(out.transpose(0, 1), w_uv)

    def _cla_global_attention(
        self,
        q_nope_raw: torch.Tensor,
        q_pe: torch.Tensor,
        attn_metadata,
        metadata_caller: str,
        topk_indices: torch.Tensor,  # unused: a dense branch selects nothing
    ) -> torch.Tensor:
        q_nope = self._q_nope_absorb(q_nope_raw)  # TND, own W_UK_T
        out = self._apply_attention(
            q_nope,
            q_pe,
            keys=self.attn.kv_cache,  # the source layer's cache
            q_cumlens=attn_metadata.query_cumlens,
            kv_lens=attn_metadata.seq_lens,
            block_table=attn_metadata.block_table,
            num_tokens=q_nope.size(0),
            layer_name=f"{self.prefix}.attn",
            attn_metadata=attn_metadata,
            # This branch always uses paged absorb attention. An absorb source
            # supplies its tiling; with a standard source this layer generates it.
            metadata_caller="prefill_absorb" if metadata_caller == "prefill" else metadata_caller,
        )  # NTD
        return super()._post_attn_absorb(out)  # NTD -> [T, N*V]


def npu_cla_forward(
    hs: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    layer_name: str,
    topk_indices_buffer: torch.Tensor | None,
) -> torch.Tensor:
    """Run one CLA layer over the step's prefill tokens and its decode tokens.

    Mirrors npu_dsa_forward's batch handling: get_batch_desc splits the step
    into one contiguous slice of prefill tokens and one of decode tokens, each
    with its own metadata, so they run separately and write into their own
    slice of the output. Either slice may be empty.
    """
    forward_context = get_forward_context()
    self: CLAReuseLayerMixin = forward_context.no_compile_layers[layer_name]
    attn_metadata = forward_context.attn_metadata
    if isinstance(attn_metadata, dict):
        attn_metadata_cla_swa = attn_metadata.get(self.cla_swa_attn_name)
        attn_metadata = attn_metadata.get(f"{self.prefix}.attn")
    else:  # dummy run
        attn_metadata = attn_metadata_cla_swa = None
    if attn_metadata is None or attn_metadata_cla_swa is None:
        return torch.zeros_like(hs)

    p_slice, d_slice, has_prefill, has_decode = get_batch_desc(attn_metadata, self.layer_idx)
    # One cache slot per token of the whole step, in the flat form and in the
    # (block, offset) form the block-update op wants. Both live on this
    # top-level metadata only -- the builder leaves them unset on the prefill
    # and decode copies -- so slice each of them per token slice below.
    slots = attn_metadata_cla_swa.slot_mapping
    slots_2d = attn_metadata_cla_swa.get_slot_mapping_2d()

    def local_slots(token_slice):
        # _kv_norm_rope_cache reads both forms off whatever it is handed.
        return SimpleNamespace(slot_mapping=slots[token_slice], slot_mapping_2d=slots_2d[token_slice])

    # SP or CP on a prefill-only step: hs is this rank's slice while cos/sin and
    # the slots cover the whole step, so nothing is sliced here.
    if self.ena_cp and not has_decode:
        return self._forward_cla_prefill_cp(
            hs,
            cos,
            sin,
            attn_metadata.prefill,
            attn_metadata_cla_swa.prefill,
            local_slots(p_slice),
            topk_indices_buffer,
        )
    if self.ena_sp and not has_decode:
        return self.forward_cla(
            hs,
            cos,
            sin,
            attn_metadata.prefill,
            attn_metadata_cla_swa.prefill,
            "prefill",
            local_slots(p_slice),
            topk_indices_buffer,
        )

    # A prefill node still schedules decode-shaped requests (a prompt cached up
    # to its last token). As npu_dsa_forward does, run such a step without SP:
    # sp_disabled gathers the rows and hands this rank its slice back.
    if self.ena_sp and topk_indices_buffer is not None:
        topk_indices_buffer = get_tp_group().all_gather(topk_indices_buffer, dim=0)
    with sp_disabled(self, hs) as (x, y, out):
        if has_prefill:
            y[p_slice] = self.forward_cla(
                x[p_slice],
                cos[p_slice],
                sin[p_slice],
                attn_metadata.prefill,
                attn_metadata_cla_swa.prefill,
                "prefill",
                local_slots(p_slice),
                topk_indices_buffer[p_slice] if topk_indices_buffer is not None else None,
            )
        if has_decode:
            y[d_slice] = self.forward_cla(
                x[d_slice],
                cos[d_slice],
                sin[d_slice],
                attn_metadata.decode,
                attn_metadata_cla_swa.decode,
                "decode",
                local_slots(d_slice),
                topk_indices_buffer[d_slice] if topk_indices_buffer is not None else None,
            )
    return out.tensor()


def npu_cla_forward_fake(
    hs: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    layer_name: str,
    topk_indices_buffer: torch.Tensor | None,
) -> torch.Tensor:
    return torch.empty_like(hs)


direct_register_custom_op(
    op_name="npu_cla_forward",
    op_func=npu_cla_forward,
    mutates_args=[],
    fake_impl=npu_cla_forward_fake,
    dispatch_key="PrivateUse1",
)
