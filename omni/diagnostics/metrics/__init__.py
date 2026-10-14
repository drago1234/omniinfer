# SPDX-License-Identifier: MIT
# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd. All Rights Reserved.

"""KV failure, worker memory, and request-stage telemetry."""

from __future__ import annotations


def install() -> None:
    """Install EngineCore-side request-stage and PD-pull hooks. Idempotent."""
    try:
        from omni_npu import envs

        if not envs.OMNI_ENABLE_TTFT_STAGE_METRICS:
            return
    except Exception:  # noqa: BLE001
        return
    try:
        from omni_npu.diagnostics.metrics import stage_hooks

        stage_hooks.install_scheduler_getter_hook()
        stage_hooks.install_platform_config_hook()
    except Exception:  # noqa: BLE001, S110
        pass
    try:
        from omni_npu.diagnostics.metrics import kv_transfer as kv_mod

        kv_mod.install_pd_pull_hooks()
    except Exception:  # noqa: BLE001, S110
        pass


# EngineCore imports NPUWorker before resolving SchedulerConfig. NPUWorker's
# stage_metrics import initializes this package early enough to install the
# scheduler and PD-pull instrumentation without worker-specific bootstrap code.
install()
