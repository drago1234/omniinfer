# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""msprobe precision-debugging helper (mirrors vllm-ascend).

omni-npu reuses MindStudio Probe (``msprobe``) to capture intermediate
tensors for accuracy diagnosis. This module owns the *config resolution*
and *debugger construction* logic that the model runner needs; the
per-forward ``start -> forward -> stop -> step`` lifecycle itself lives in
:class:`~omni_npu.worker.npu_model_runner.NPUModelRunner` (see
``_start_dump_data`` / ``_finalize_dump_data``), so it can wrap every
return path with a ``try/finally``.

The two entry points exposed here are deliberately free of any runtime NPU
object — they take a resolved config path and a mode flag and return a
debugger handle (or ``None`` when disabled), which keeps them unit-testable
without a device.

See ``docs/msprobe_guide.md`` for the end-user workflow.
"""
import json
import os
from typing import Any

# Directory where an inline ``msprobe_dump_config`` dict is materialized into
# a JSON file before being handed to ``PrecisionDebugger``. Mirrors
# vllm-ascend's ``.vllm_ascend/msprobe`` directory.
MSPROBE_DUMP_DIR = os.path.join(os.getcwd(), ".omni_npu", "msprobe")

# Fixed filename for the materialized inline config.
MSPROBE_DUMP_CONFIG_FILENAME = "msprobe_dump_config.json"


def materialize_dump_config_to_file(dump_config: dict[str, Any]) -> str:
    """Write an inline ``msprobe_dump_config`` dict to disk and return its path.

    ``PrecisionDebugger`` / ``AclGraphDumper`` both take a *path* to a JSON
    config file. When the user passes the config inline via
    ``additional_config.msprobe_dump_config`` we materialize it here first.
    The directory is created on demand and the file is overwritten each run
    so the latest config wins.
    """
    os.makedirs(MSPROBE_DUMP_DIR, exist_ok=True)
    dump_config_file_path = os.path.join(
        MSPROBE_DUMP_DIR, MSPROBE_DUMP_CONFIG_FILENAME
    )
    with open(dump_config_file_path, "w", encoding="utf-8") as file:
        json.dump(dump_config, file, ensure_ascii=False, indent=2)
    return dump_config_file_path


def resolve_dump_config_path(
    dump_config_path: str | None,
    dump_config: dict[str, Any] | None,
) -> str | None:
    """Resolve the two mutually-exclusive dump config inputs to one path.

    The type checks here are a defense-in-depth mirror of the
    :class:`~omni_npu.configs.additional_config.OmniAdditionalConfig`
    ``__post_init__`` validation; callers that already go through the
    dataclass will never hit the ``TypeError`` branches, but this function
    is also used directly by the runner and must stay safe on its own.

    Args:
        dump_config_path: A path to an existing msprobe config JSON file.
        dump_config: An inline config dict to be materialized to a file.

    Returns:
        The final path the debugger should consume, or ``None`` when
        neither input is set (msprobe disabled).

    Raises:
        ValueError: Both inputs are set at once.
        TypeError: ``dump_config`` is not a dict, or ``dump_config_path``
            is not a string.
    """
    if dump_config_path is not None and dump_config is not None:
        raise ValueError(
            "Only one of additional_config.msprobe_dump_config_path or "
            "additional_config.msprobe_dump_config can be set."
        )
    if dump_config is not None:
        if not isinstance(dump_config, dict):
            raise TypeError(
                f"additional_config.msprobe_dump_config must be a dict, "
                f"got {type(dump_config).__name__}."
            )
        return materialize_dump_config_to_file(dump_config)
    if dump_config_path is not None and not isinstance(dump_config_path, str):
        raise TypeError(
            f"additional_config.msprobe_dump_config_path must be a string, "
            f"got {type(dump_config_path).__name__}."
        )
    return dump_config_path


def build_debugger(
    dump_config_path: str | None,
    *,
    use_graph: bool,
) -> Any | None:
    """Construct the msprobe debugger for the current mode.

    The import of ``msprobe`` is *deferred* to this function so that
    environments without msprobe pay no import cost when dump is disabled.
    When dump *is* enabled, a missing msprobe is a hard failure: msprobe is
    an explicitly-requested debugging tool, and silently degrading would
    let a user believe they are collecting dumps when they are not.

    Args:
        dump_config_path: Resolved config path (from
            :func:`resolve_dump_config_path`). ``None`` disables dump.
        use_graph: ``True`` when the runner runs in ACL-graph (cudagraph)
            mode, which requires ``AclGraphDumper`` instead of the eager
            ``PrecisionDebugger``.

    Returns:
        ``(debugger, started)`` — ``debugger`` is ``None`` when disabled,
        ``started`` is always ``False`` (the runner flips it on the first
        ``_start_dump_data``).
    """
    if dump_config_path is None:
        return None

    if not use_graph:
        # Eager mode: the standard PrecisionDebugger.
        from msprobe.pytorch import PrecisionDebugger

        return PrecisionDebugger(dump_config_path)

    # Graph mode: AclGraphDumper requires msprobe to be built from source
    # with the aclgraph_dump module enabled. A plain pip install lacks it,
    # so an ImportError here means the environment is wrong, not that the
    # user forgot to enable dump — surface it loudly.
    try:
        from msprobe.pytorch import AclGraphDumper
    except Exception as exc:
        raise RuntimeError(
            "Failed to import AclGraphDumper from msprobe. "
            "Please install/rebuild msprobe with aclgraph_dump enabled."
        ) from exc

    return AclGraphDumper(dump_config_path)
