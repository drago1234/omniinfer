# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""NaN/Inf feature actions registered with the worker collective RPC registry."""
from omni_npu import envs
from omni_npu.worker.collective_features import register_collective_feature


def _detector(worker):
    detector = getattr(worker.model_runner, "nan_inf_detector", None)
    if detector is None:
        raise RuntimeError(
            f"{type(worker.model_runner).__name__} does not support NaN/Inf detection")
    return detector


@register_collective_feature("nan_inf_detector", "configure")
def configure_detector(worker, **options):
    return _detector(worker).configure(
        worker.model_runner.get_model(),
        default_dump_dir=envs.OMNI_DUMP_DIR,
        **options,
    )


@register_collective_feature("nan_inf_detector", "status")
def detector_status(worker, **options):
    if options:
        raise ValueError("'status' does not accept feature options")
    return _detector(worker).status()
