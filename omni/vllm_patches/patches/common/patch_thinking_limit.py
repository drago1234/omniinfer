# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Server-side ``thinking_token_budget`` and ``--reasoning-config`` env override.

This is the vLLM 0.25.1 port of the omni-npu (vLLM 0.14.0) ``patch_thinking_limit``
feature authored by thunderloves (commits !1155 / !1476 / !1780).

Why this file is small
----------------------
vLLM 0.25.1 already owns the *entire* request-side thinking-budget pipeline
natively:

* ``SamplingParams.thinking_token_budget`` (a real, validated field),
* ``ChatCompletionRequest.thinking_token_budget`` flowing into ``to_sampling_params``,
* ``VllmConfig.reasoning_config`` + the ``--reasoning-config`` CLI flag,
* the native ``vllm/v1/sample/thinking_budget_state.py`` ``ThinkingBudgetStateHolder``
  wired through ``SamplingMetadata`` / ``Sampler`` / ``RejectionSampler`` /
  ``InputBatch``.

So, unlike the 0.14.0 patch, this file does **not** re-implement any of that. It
adds only the three Omni value-adds that have no native equivalent:

1. ``OMNI_REASONING_CONFIG`` env-var override of ``--reasoning-config``
   (env > CLI; dict/JSON-string coercion),
2. a **server-side default** ``thinking_token_budget``: when the launch config
   sets ``reasoning_config.thinking_token_budget`` and a request omits its own
   budget, the default is injected so the request is capped. The request's own
   ``thinking_token_budget`` always wins,
3. lifting 0.25.1's "``thinking_token_budget`` is not yet supported by the V2
   model runner" rejection, which upstream removes in vllm#46727 together with
   its own V2 implementation. Omni's V2 implementation is
   ``omni/worker/npu/thinking_budget.py`` (see ``patch_mrv2_thinking_budget.py``).

The thinking-ban feature (``ban_tool_start/end_in_thinking``) and the MTP
structured-output fix from the same upstream commits already live in
``patch_thinking_ban.py`` / ``patch_vllm_structured_output.py`` and are not
touched here.
"""

from __future__ import annotations

import json
from typing import Any

from vllm import EngineArgs, SamplingParams
from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.v1.engine.input_processor import InputProcessor

from omni_npu import envs
from omni_npu.v1.config import ReasoningConfig
from omni_npu.vllm_patches.core import VLLMPatch, register_patch

logger = init_logger(__name__)

_THINKING_TOKEN_BUDGET_KEY = "thinking_token_budget"
_REASONING_CONFIG_ENV = "OMNI_REASONING_CONFIG"


# ────────────────────────────────────────────────────────────
# Helpers (ported verbatim from the v0.14.0 patch; version-agnostic)
# ────────────────────────────────────────────────────────────
def _normalize_thinking_token_budget(value: Any) -> int | None:
    """Return a non-negative int, or ``None`` if unset/invalid.

    ``bool`` is rejected explicitly (``True``/``False`` are ints in Python but
    are not valid budgets), matching the v0.14.0 behaviour.
    """
    if value is None:
        return None
    if isinstance(value, bool):
        return None
    if isinstance(value, int) and value >= 0:
        return value
    return None


def _coerce_reasoning_config(value: Any) -> ReasoningConfig | None:
    """Coerce a CLI/env value into a ``ReasoningConfig`` (or ``None``)."""
    if value is None or isinstance(value, ReasoningConfig):
        return value
    if isinstance(value, str):
        value = json.loads(value)
    if isinstance(value, dict):
        return ReasoningConfig(**value)
    # Unknown type (e.g. a programmatically-supplied int): treat as unset
    # rather than transparently passing it through to initialize_token_ids,
    # which would raise AttributeError deep inside create_engine_config.
    return None


def _reasoning_config_from_env() -> ReasoningConfig | None:
    """Resolve a ``ReasoningConfig`` from ``OMNI_REASONING_CONFIG``.

    The value must be the same JSON string accepted by ``--reasoning-config``.
    Returns ``None`` when the variable is unset/empty or fails to parse, so a
    malformed env value leaves the CLI value in place rather than silently
    disabling reasoning.
    """
    raw = envs.OMNI_REASONING_CONFIG
    if not raw:
        return None
    try:
        cfg = _coerce_reasoning_config(raw)
    except Exception as exc:  # noqa: BLE001
        logger.error(
            "Error parsing %s environment variable: %s", _REASONING_CONFIG_ENV, exc
        )
        return None
    if cfg is not None:
        logger.info(
            "Loaded reasoning_config from the %s environment variable",
            _REASONING_CONFIG_ENV,
        )
    return cfg


# ────────────────────────────────────────────────────────────
# Patch 1: EngineArgs.create_engine_config — env override + coercion
# ────────────────────────────────────────────────────────────
# Other Omni patches (patch_repetition_detection_config.py,
# patch_vllm_structured_output.py) also wrap EngineArgs.create_engine_config via
# apply_bypass_conflict, forming a relay. This file sorts after both, so it
# becomes the next link in that chain. Capture the *currently active* upstream
# at apply-time (after every patch module has imported), not at import time.


@register_patch("ThinkingLimitEngineArgsPatch", EngineArgs)
class ThinkingLimitEngineArgsPatch(VLLMPatch):
    """Apply ``OMNI_REASONING_CONFIG`` (env > CLI) and dict coercion.

    Native 0.25.1 already registers ``--reasoning-config``, parses its JSON into
    a ``ReasoningConfig``, and feeds it into ``VllmConfig`` (which calls
    ``initialize_token_ids``). This patch only overrides
    ``create_engine_config`` to (a) coerce a programmatically-supplied dict and
    (b) let the env var win over the CLI flag. Both resolve BEFORE the native
    build: ``self.reasoning_config`` is cleaned/replaced up front so the freshly
    built ``vllm_config`` already carries the resolved config and the native
    ``VllmConfig`` construction itself runs ``initialize_token_ids``.
    """

    _attr_names_to_apply = ["create_engine_config"]

    @classmethod
    def apply(cls):
        cls.apply_bypass_conflict("create_engine_config")

    def create_engine_config(
        self,
        usage_context=None,
        headless: bool = False,
    ) -> VllmConfig:
        # Coerce a programmatically-supplied dict/JSON-str (the CLI path already
        # produces a ReasoningConfig natively; this is a defensive no-op there).
        # A garbage value must never reach the native VllmConfig construction,
        # so resolve before building, not after.
        try:
            cfg = _coerce_reasoning_config(getattr(self, "reasoning_config", None))
        except Exception as exc:  # noqa: BLE001
            logger.error("Error parsing reasoning_config: %s", exc)
            cfg = None
        self.reasoning_config = cfg

        # Env wins over CLI.
        env_cfg = _reasoning_config_from_env()
        if env_cfg is not None:
            self.reasoning_config = env_cfg

        return ThinkingLimitEngineArgsPatch._upstream_create_engine_config(
            self, usage_context, headless
        )


# ────────────────────────────────────────────────────────────
# Patch 2: InputProcessor.process_inputs — server-side default budget
# ────────────────────────────────────────────────────────────
# ``InputProcessor.process_inputs`` is *not* ours alone: the repetition-detection
# patch (patch_repetition_detection_config.py) already wraps it. So unlike
# ``create_engine_config`` above we are the *second* link, not the first, and
# must chain onto whatever wrapper is currently active via
# ``apply_bypass_conflict`` rather than preempting with the default ``apply()``.
# Applying with the default ``apply()`` raises
# ``process_inputs already patched by InputProcessorRepetitionDetectionPatch``
# and silently skips this patch -- see issue "92BVL_A3_patch_conflict".
#
# Chaining means calling ``_upstream_process_inputs`` (the prior wrapper), never
# the raw native: calling native directly would bypass repetition-detection's
# injection and silently drop that feature. The prior wrapper is itself
# responsible for reaching native, so native runs exactly once.


def _find_sampling_params(args: tuple, kwargs: dict) -> SamplingParams | None:
    params = kwargs.get("params")
    if params is None:
        for arg in args:
            if isinstance(arg, SamplingParams):
                return arg
        return None
    return params if isinstance(params, SamplingParams) else None


@register_patch("ThinkingLimitInputProcessorPatch", InputProcessor)
class ThinkingLimitInputProcessorPatch(VLLMPatch):
    """Inject ``reasoning_config.thinking_token_budget`` when a request omits its own.

    The request's own ``thinking_token_budget`` always wins. The server default
    is applied only when it is unset. Runs before native ``_validate_params`` so
    an injected value is validated exactly like a client-supplied one.
    """

    _attr_names_to_apply = ["process_inputs"]

    @classmethod
    def apply(cls):
        # Second link on process_inputs (repetition-detection is first): chain
        # onto the active wrapper instead of preempting. _upstream_process_inputs
        # is captured here, after every patch module has been imported.
        cls.apply_bypass_conflict("process_inputs")

    def process_inputs(self, *args, **kwargs):
        params = _find_sampling_params(args, kwargs)
        if params is not None and params.thinking_token_budget is None:
            reasoning_config = getattr(self.vllm_config, "reasoning_config", None)
            server_budget = _normalize_thinking_token_budget(
                getattr(reasoning_config, _THINKING_TOKEN_BUDGET_KEY, None)
            )
            if server_budget is not None:
                params.thinking_token_budget = server_budget

        # Chain to the prior wrapper (repetition-detection, etc.), not the raw
        # native, so every link's injection still runs and native runs once.
        return ThinkingLimitInputProcessorPatch._upstream_process_inputs(
            self, *args, **kwargs
        )


# ────────────────────────────────────────────────────────────
# Patch 3: InputProcessor._validate_params — lift the MRv2 hard rejection
# ────────────────────────────────────────────────────────────
# vllm#46727 removes this gate; the NPU V2 sampler now provides the feature.


@register_patch("ThinkingLimitV2GatePatch", InputProcessor)
class ThinkingLimitV2GatePatch(VLLMPatch):
    """Let ``thinking_token_budget`` through on the V2 model runner.

    The gate keys on ``InputProcessor.use_v2_model_runner`` (set in
    ``__init__`` from ``vllm_config``), which nothing else reads in 0.25.1.
    Rather than re-implementing the native validator, the flag is cleared for
    the duration of the upstream call so ``params.verify`` and the
    ``reasoning_config`` presence check still run exactly as before.

    ``apply_bypass_conflict``: ``_validate_params`` is also wrapped by
    patch_vllm_structured_output.py. Chaining through
    ``_upstream_validate_params`` keeps both link's behaviour independent of
    the patch apply order (file names decide it today).
    """

    _attr_names_to_apply = ["_validate_params"]

    @classmethod
    def apply(cls):
        cls.apply_bypass_conflict("_validate_params")

    def _validate_params(self, params, supported_tasks):
        # Only the combination that 0.25.1 rejects is special-cased; the flag
        # is left untouched (and the call is a plain passthrough) otherwise.
        budgeted = (
            isinstance(params, SamplingParams)
            and params.thinking_token_budget is not None
        )
        v2 = getattr(self, "use_v2_model_runner", False)
        if not (budgeted and v2):
            return ThinkingLimitV2GatePatch._upstream_validate_params(
                self, params, supported_tasks
            )

        self.use_v2_model_runner = False
        try:
            return ThinkingLimitV2GatePatch._upstream_validate_params(
                self, params, supported_tasks
            )
        finally:
            self.use_v2_model_runner = v2
