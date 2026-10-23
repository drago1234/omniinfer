# SPDX-License-Identifier: MIT
# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd. All Rights Reserved.

from typing import Any

import torch
import torch_npu
from vllm.model_executor.layers.activation import SiluAndMul, SiluAndMulWithClamp

from omni_npu.model_config.config_loader.loader import model_extra_config


@SiluAndMulWithClamp.register_oot
class NPUSiluAndMulWithClamp(SiluAndMulWithClamp):
    """NPU clamped SwiGLU with optional routing weights before down projection."""

    @property
    def use_swiglu_group_fusion(self) -> bool:
        return (
            model_extra_config.operator_opt_config.use_swiglu_group_fusion_op
            and self.swiglu_limit > 0
            # The fusion operator implements standard SwiGLU, not the OAI variant.
            and self.alpha == 1.0
            and self.beta == 0.0
        )

    def forward(self, x: torch.Tensor, weight: torch.Tensor | None = None) -> torch.Tensor:
        if weight is not None:
            weight = weight.reshape(*x.shape[:-1], 1).to(torch.float32)
        return super().forward(x, weight=weight)

    def forward_native(self, x: torch.Tensor, weight: torch.Tensor | None = None) -> torch.Tensor:
        output = super().forward_native(x)
        return output if weight is None else (output.float() * weight).to(output.dtype)

    def forward_oot(self, x: torch.Tensor, weight: torch.Tensor | None = None) -> torch.Tensor:
        # The fusion operator does not support empty tensors.
        if not self.use_swiglu_group_fusion or x.numel() == 0:
            return self.forward_native(x, weight=weight)
        import omni_training_custom_ops  # noqa: F401

        return torch.ops.custom.npu_ai_infra_swiglu_group(
            x, weight=weight, group_index=None, clamp_limit=self.swiglu_limit,
        )


@SiluAndMul.register_oot
class NPUSiluAndMul(SiluAndMul):
    def forward_oot(
        self,
        x: torch.Tensor | dict[str, Any],
        quant_symbol: bool = False
    ) -> torch.Tensor | dict[str, Any]:
        if quant_symbol and isinstance(x, dict):
            kwargs = {
                "x": x.get("x_int8"),
                "weight_scale": x.get("out_scale").to(torch.float32),
                "quant_scale": x.get("in_scale", None),
                "activation_scale": x.get("pertoken_scale", None),
                "bias": None,
                "quant_offset": None,
                "group_index": None,
                "activate_left": True,
                "quant_mode": 1,
            }
            h, pertoken_scale = torch_npu.npu_dequant_swiglu_quant(**kwargs)
            return {"x_int8": h, "pertoken_scale": pertoken_scale}

        return torch_npu.npu_swiglu(x)
