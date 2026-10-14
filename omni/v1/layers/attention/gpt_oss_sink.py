# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.

import torch

from vllm.distributed import get_tp_group
from vllm.utils.torch_utils import direct_register_custom_op

from omni_npu.attention.backends.utils import CrossLayerSharedOp
from omni_npu.model_config.config_loader.loader import model_extra_config

# Metadata buffer for the FIA that takes the sink logit itself, shared across the
# layers that drive it exactly as the FIA-sink metadata is.
npu_gpt_oss_fia_metadata: CrossLayerSharedOp | None = None


class GptOssSinkMixin:
    """Adds ``gpt_oss_sink`` parameters and the rescale that consumes them.

    The host class must already define ``num_heads`` (global query heads) and
    ``num_local_heads`` (this rank's shard) before calling
    :meth:`_init_gpt_oss_sink`.
    """

    def _init_gpt_oss_sink(self, config) -> None:
        """Create the per-query-head sink parameter declared by ``config``."""
        self.use_gpt_oss_sink = getattr(config, "use_gpt_oss_sink", False)
        # Either hand the sink logit to the kernel or rebuild its effect from the
        # log-sum-exp, as npu_pangu does for the other stack.
        self.use_gpt_oss_sink_rescale = (
            self.use_gpt_oss_sink
            and model_extra_config.operator_opt_config.use_gpt_oss_sink_rescale
        )
        self.gpt_oss_sink = self._create_gpt_oss_sink() if self.use_gpt_oss_sink else None
        self.gpt_oss_sink_swa = None

    def _init_gpt_oss_swa_sink(self) -> None:
        """Add the second sink a CLA layer's local branch needs.

        The base class has already created the global branch's sink, so only
        the extra parameter is added here.
        """
        if self.use_gpt_oss_sink:
            self.gpt_oss_sink_swa = self._create_gpt_oss_sink()

    def _create_gpt_oss_sink(self) -> torch.nn.Parameter:
        # Sink logits are FP32 in training and stay FP32 for inference.
        sink = torch.nn.Parameter(
            torch.empty(self.num_local_heads, device="npu", dtype=torch.float32),
            requires_grad=False,
        )
        sink.weight_loader = self._load_gpt_oss_sink_weight
        return sink

    def _load_gpt_oss_sink_weight(self, param: torch.nn.Parameter, loaded_weight: torch.Tensor) -> None:
        """Load a per-query-head GPT-OSS sink, sharding it like Q heads."""
        if loaded_weight.ndim != 1 or loaded_weight.shape[0] != self.num_heads:
            raise ValueError(
                f"gpt_oss_sink must have shape ({self.num_heads},), "
                f"but got {tuple(loaded_weight.shape)}."
            )
        if param.ndim != 1 or param.shape[0] != self.num_local_heads:
            raise ValueError(
                f"The local gpt_oss_sink parameter must have shape "
                f"({self.num_local_heads},), but got {tuple(param.shape)}."
            )

        if self.num_local_heads == self.num_heads:
            local_weight = loaded_weight
        else:
            head_start = get_tp_group().rank_in_group * self.num_local_heads
            local_weight = loaded_weight.narrow(0, head_start, self.num_local_heads)
        param.data.copy_(local_weight.to(device=param.device, dtype=param.dtype))

    def _rescale_gpt_oss_sink_output(
        self,
        output: torch.Tensor,
        softmax_lse: torch.Tensor,
        learnable_sink: torch.Tensor,
        output_layout: str,
    ) -> torch.Tensor:
        """Fold the sink logit into an attention output via its softmax_lse."""
        scale = torch.sigmoid(
            softmax_lse.reshape(-1, self.num_local_heads)
            - learnable_sink.view(1, self.num_local_heads)
        )
        if output_layout == "NTD":
            scale = scale.transpose(0, 1)
        elif output_layout != "TND":
            raise ValueError(f"Unsupported GPT-OSS sink output layout: {output_layout}.")
        return (output.to(scale.dtype) * scale.unsqueeze(-1)).to(output.dtype)

    @staticmethod
    def _reserve_gpt_oss_fia_metadata(callers: tuple) -> None:
        """Reserve a native-sink metadata buffer per caller key."""
        global npu_gpt_oss_fia_metadata
        if npu_gpt_oss_fia_metadata is None:
            npu_gpt_oss_fia_metadata = CrossLayerSharedOp(
                op=torch.ops.custom.npu_ai_infra_flash_attention_metadata,
                shape=(1024,),
                dtype=torch.int32,
                callers=callers,
            )
        else:
            npu_gpt_oss_fia_metadata.reserve(callers)

    @staticmethod
    def _apply_gpt_oss_fia_native(
        kwargs: dict,
        learnable_sink: torch.Tensor,
        metadata_caller,
        is_metadata_producer: bool,
    ) -> torch.Tensor:
        """Run the FIA that takes the sink logit itself, given FIA-sink kwargs.

        The two ops differ in how they name the sequence lengths and the head
        count, and this one needs its own tiling metadata, so translate rather
        than ask every call site to build both forms.
        """
        kwargs["batch_invariant"] = model_extra_config.operator_opt_config.use_batch_invariant_op
        query_cumlens = kwargs.pop("actual_seq_qlen").to(torch.int64)
        kv_lens = kwargs.pop("actual_seq_kvlen").to(torch.int64)
        num_heads = kwargs.pop("num_query_heads")
        num_kv_heads = kwargs.pop("num_key_value_heads")
        block_table = kwargs.get("block_table")
        paged = block_table is not None
        stream_limit = torch.npu.get_stream_limit(torch.npu.current_stream())
        meta_data = npu_gpt_oss_fia_metadata(
            {
                "batch_invariant": kwargs["batch_invariant"],
                "num_heads_q": num_heads,
                "num_heads_kv": num_kv_heads,
                "head_dim_qk": kwargs["query"].shape[-1],
                "head_dim_v": kwargs["value"].shape[-1],
                "actual_seq_lengths": query_cumlens,
                "actual_seq_lengths_kv": kv_lens,
                "batch_size": block_table.shape[0] if paged else query_cumlens.numel(),
                "sparse_mode": kwargs["sparse_mode"],
                "pre_tokens": kwargs["pre_tokens"],
                "next_tokens": kwargs["next_tokens"],
                "input_layout": "TND",
                "input_layout_kv": "BnBsH" if paged else "TND",
                "rope_head_dim": kwargs["query_rope"].shape[-1],
                "block_size": kwargs.get("block_size", 0) if paged else 0,
                "aic_core_num": stream_limit["cube_core_num"],
                "aiv_core_num": stream_limit["vector_core_num"],
            },
            is_metadata_producer,
            metadata_caller,
        )
        return torch.ops.vllm.npu_gpt_oss_fia(
            kwargs.pop("query"),
            kwargs.pop("key"),
            kwargs.pop("value"),
            meta_data,
            query_cumlens,
            kv_lens,
            learnable_sink.view(-1),
            num_heads,
            num_kv_heads,
            **kwargs,
        )


# The operator package ships no FakeTensor-safe meta kernel for
# npu_ai_infra_flash_attention, so FULL graph tracing would enter the device
# kernel with unallocated fake tensors. This boundary keeps that out; metadata
# generation stays outside it so its cross-layer sharing is unaffected.
def npu_gpt_oss_fia(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    meta_data: torch.Tensor,
    actual_seq_lengths: torch.Tensor,
    actual_seq_lengths_kv: torch.Tensor,
    sinks: torch.Tensor,
    num_heads: int, num_key_value_heads: int,
    atten_mask: torch.Tensor | None = None,
    block_table: torch.Tensor | None = None,
    query_rope: torch.Tensor | None = None,
    key_rope: torch.Tensor | None = None,
    softmax_scale: float = 1.0,
    pre_tokens: int = (1 << 31) - 1, next_tokens: int = (1 << 31) - 1,
    input_layout: str = "TND",
    sparse_mode: int = 0,
    block_size: int = 0,
    batch_invariant: bool = False,
) -> torch.Tensor:
    return torch.ops.custom.npu_ai_infra_flash_attention(
        query, key, value, meta_data,
        atten_mask=atten_mask,
        actual_seq_lengths=actual_seq_lengths,
        actual_seq_lengths_kv=actual_seq_lengths_kv,
        block_table=block_table,
        query_rope=query_rope,
        key_rope=key_rope,
        sinks=sinks,
        num_heads=num_heads,
        softmax_scale=softmax_scale,
        pre_tokens=pre_tokens, next_tokens=next_tokens,
        input_layout=input_layout,
        num_key_value_heads=num_key_value_heads,
        sparse_mode=sparse_mode,
        block_size=block_size,
        batch_invariant=batch_invariant,
    )[0]


def npu_gpt_oss_fia_fake(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    meta_data: torch.Tensor,
    actual_seq_lengths: torch.Tensor, actual_seq_lengths_kv: torch.Tensor,
    sinks: torch.Tensor,
    num_heads: int,
    num_key_value_heads: int,
    atten_mask: torch.Tensor | None = None,
    block_table: torch.Tensor | None = None,
    query_rope: torch.Tensor | None = None,
    key_rope: torch.Tensor | None = None,
    softmax_scale: float = 1.0,
    pre_tokens: int = (1 << 31) - 1,
    next_tokens: int = (1 << 31) - 1,
    input_layout: str = "TND",
    sparse_mode: int = 0, block_size: int = 0,
    batch_invariant: bool = False,
) -> torch.Tensor:
    if input_layout == "TND_NTD":
        shape = (num_heads, query.size(0), value.size(-1))
    else:
        shape = (*query.shape[:-1], value.size(-1))
    return torch.empty(shape, device=query.device, dtype=query.dtype)


direct_register_custom_op(
    op_name="npu_gpt_oss_fia",
    op_func=npu_gpt_oss_fia,
    mutates_args=[],
    fake_impl=npu_gpt_oss_fia_fake,
    dispatch_key="PrivateUse1",
)
