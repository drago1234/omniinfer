# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""Dump-only full-module observation for pure inference runs."""

from __future__ import annotations

import logging
import re
from pathlib import Path
from threading import local
from typing import Any

import torch

from .config import AlignmentConfigError, PureRealConfig
from .dump_io import TrainRefDataError, primary_tensor
from .hooks import MethodHandle, close_handles, close_runner_session
from .train_ref import should_skip_runtime_hook

logger = logging.getLogger(__name__)

_SESSION_ATTR = "_omni_accuracy_alignment_pure_real_session"


def _rank() -> int:
    try:
        from vllm.distributed import get_world_group

        return int(get_world_group().rank_in_group)
    except (ImportError, RuntimeError, AttributeError):
        return 0


def _safe_name(name: str) -> str:
    return name.replace("/", "__").replace(":", "_")


class PureRealSession:
    """Own output-only hooks for one loaded Pangu model instance."""

    def __init__(self, config: PureRealConfig, rank: int) -> None:
        self.config = config
        self.rank = rank
        self._handles: list[Any] = []
        self._call_tls = local()
        try:
            self._include = tuple(re.compile(item) for item in config.include_patterns)
            self._exclude = tuple(re.compile(item) for item in config.exclude_patterns)
        except re.error as exc:
            raise AlignmentConfigError(
                f"invalid pure_real include/exclude regular expression: {exc}"
            ) from exc

    @property
    def installed_hook_count(self) -> int:
        return len(self._handles)

    def _selected(self, qualname: str) -> bool:
        if self._exclude and any(regex.search(qualname) for regex in self._exclude):
            return False
        return not self._include or any(
            regex.search(qualname) for regex in self._include
        )

    def install(self, model: torch.nn.Module) -> None:
        self.close()
        try:
            for qualname, module in model.named_modules():
                if not qualname or not self._selected(qualname):
                    continue
                original = module.forward
                self._handles.append(
                    MethodHandle(
                        module,
                        "forward",
                        self._make_wrapper(qualname, original),
                    )
                )
        except Exception:
            self.close()
            raise

    def close(self) -> None:
        close_handles(self._handles)
        self._call_tls.counters = {}

    def _dump(self, qualname: str, output: Any) -> None:
        if should_skip_runtime_hook():
            return
        # Match the legacy convention: P-side tensors before TP sharding
        # are redundant, so only world rank 0 writes them.
        if self.config.role == "P" and self.rank != 0:
            return
        try:
            tensor = primary_tensor(output)
        except TrainRefDataError:
            return
        if tensor.ndim == 0:
            return
        counters = getattr(self._call_tls, "counters", None)
        if counters is None:
            counters = {}
            self._call_tls.counters = counters
        call = counters.get(qualname, 0)
        counters[qualname] = call + 1
        stage_dir = self.config.infer_dump_dir / self.config.stage
        stage_dir.mkdir(parents=True, exist_ok=True)
        filename = (
            f"decorator_{_safe_name(qualname)}_run{self.config.run_id}"
            f"_call{call}_rank{self.rank}.pt"
        )
        try:
            torch.save(tensor.detach().cpu(), stage_dir / filename)
        except (OSError, RuntimeError) as exc:
            logger.warning("pure_real failed to save %s: %s", filename, exc)

    def _make_wrapper(self, qualname: str, original):
        def wrapped(_module, *args, **kwargs):
            output = original(*args, **kwargs)
            self._dump(qualname, output)
            return output

        return wrapped


def _ultra_model_classes():
    from omni_npu.v1.models.pangu.pangu_ultra_moe import (
        OpenPanguModel,
        OpenPanguModelBase,
    )

    return OpenPanguModelBase, OpenPanguModel


def install_pure_real(
    runner: Any,
    config: PureRealConfig | None = None,
) -> PureRealSession | None:
    """Install pure-real observers after the runner owns a loaded model."""

    close_runner_session(runner, _SESSION_ATTR)

    model = runner.get_model()
    base_cls, model_cls = _ultra_model_classes()
    if not isinstance(model, base_cls) or not isinstance(
        getattr(model, "model", None), model_cls
    ):
        logger.warning(
            "accuracy-alignment pure_real supports PanguUltraMoE only; got %s",
            type(model).__name__,
        )
        return None
    if not bool(getattr(runner.model_config, "enforce_eager", False)):
        raise AlignmentConfigError(
            "pure_real requires model_config.enforce_eager=True; graph replay "
            "does not execute Python module hooks reliably"
        )
    resolved = PureRealConfig.from_env() if config is None else config
    session = PureRealSession(resolved, _rank())
    session.install(model)
    setattr(runner, _SESSION_ATTR, session)
    logger.info(
        "installed pure_real hooks for rank=%d modules=%d run_id=%s",
        session.rank,
        session.installed_hook_count,
        resolved.run_id,
    )
    return session


def pure_real_dump_path(
    root: Path,
    stage: str,
    qualname: str,
    run_id: str,
    call: int,
    rank: int,
) -> Path:
    """Expose the stable filename convention to the comparator and tests."""

    return (
        root
        / stage
        / (f"decorator_{_safe_name(qualname)}_run{run_id}_call{call}_rank{rank}.pt")
    )
