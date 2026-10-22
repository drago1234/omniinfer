# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.

"""FRACTAL_NZ weight cast for vLLM 0.25.1 UnquantizedLinearMethod.

Upstream ``process_weights_after_loading`` is a no-op on NPU, so weights
stay ND and aclnnMatmul pays an online NZ conversion on every forward.
The vLLM loader instantiates ``UnquantizedLinearMethod`` directly with no
plugin hook for post-load weight formatting, so a VLLMPatch is the only
seam short of forking upstream.

Remove once upstream (> 0.25.1) handles NPU weight format inside
``process_weights_after_loading``.
"""

import torch
import torch_npu

from vllm.logger import init_logger
from vllm.model_executor.layers.linear import UnquantizedLinearMethod
from vllm.model_executor.utils import set_weight_attrs

from omni_npu.model_config.config_loader.loader import model_extra_config
from omni_npu.vllm_patches.core import VLLMPatch, register_patch

logger = init_logger(__name__)

_orig_process_weights_after_loading = (
    UnquantizedLinearMethod.process_weights_after_loading
)


def _patched_process_weights_after_loading(self, layer: torch.nn.Module) -> None:
    _orig_process_weights_after_loading(self, layer)

    # unquant_bmm_nz: opt-out switch; is_weight_nz: idempotency across
    # reloads / RL weight sync; fp32 (MHC phi) feeds aclnnAiInfra* ops
    # that require ND format.
    if (
        model_extra_config.operator_opt_config.unquant_bmm_nz
        and not getattr(layer.weight, "is_weight_nz", False)
        and layer.weight.dtype in (torch.float16, torch.bfloat16)
    ):
        layer.weight.data = torch_npu.npu_format_cast(
            layer.weight.data, torch_npu.Format.FRACTAL_NZ
        )
        set_weight_attrs(layer.weight, {"is_weight_nz": True})


@register_patch("NPUUnquantizedLinearMethod", UnquantizedLinearMethod)
class NPUUnquantizedLinearMethodPatch(VLLMPatch):
    _attr_names_to_apply = ["process_weights_after_loading"]
    process_weights_after_loading = _patched_process_weights_after_loading
