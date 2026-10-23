# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""Instance-level input substitution and output dump hooks for train-ref."""

from __future__ import annotations

import contextvars
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from .config import TrainRefConfig
from .dump_io import (
    TrainRefDataError,
    dump_output,
    load_training_input,
    prepare_mock_input,
)

_IN_DUMMY_RUN = contextvars.ContextVar(
    "omni_accuracy_alignment_in_dummy_run", default=False
)


@contextmanager
def dummy_run_guard():
    token = _IN_DUMMY_RUN.set(True)
    try:
        yield
    finally:
        _IN_DUMMY_RUN.reset(token)


def _capturing() -> bool:
    try:
        from vllm.forward_context import get_forward_context

        try:
            forward_context = get_forward_context()
        except (AssertionError, RuntimeError):
            # A direct eager module call can legitimately run without vLLM's
            # forward context. Profile/dummy entry points are guarded by the
            # model-runner patch instead of disabling all such calls here.
            return False
        return bool(getattr(forward_context, "capturing", False))
    except (ImportError, AttributeError):
        return False


def should_skip_runtime_hook() -> bool:
    return _IN_DUMMY_RUN.get() or _capturing()


@dataclass(frozen=True)
class TrainRefTarget:
    module_name: str
    target_kind: str
    module: torch.nn.Module
    training_dump_path: Path


class TrainRefSession:
    """Owns hooks and cached golden inputs for one loaded model instance."""

    def __init__(self, config: TrainRefConfig, rank: int) -> None:
        self.config = config
        self.rank = rank
        self._handles: list[Any] = []
        self._extensions: list[Any] = []
        self._training_inputs: dict[str, torch.Tensor] = {}

    @property
    def installed_hook_count(self) -> int:
        return len(self._handles) + sum(
            int(getattr(extension, "installed_hook_count", 0))
            for extension in self._extensions
        )

    def install(self, targets: list[TrainRefTarget]) -> None:
        self.close()
        # Validate and cache every input before installing the first hook. A
        # missing/corrupt later file must not leave a partially active session.
        loaded_inputs = {
            target.module_name: load_training_input(target.training_dump_path)
            for target in targets
            if self.config.should_mock(target.module_name, target.target_kind)
        }
        self._training_inputs.update(loaded_inputs)
        for target in targets:
            try:
                if self.config.should_mock(target.module_name, target.target_kind):
                    self._handles.append(
                        target.module.register_forward_pre_hook(
                            self._make_pre_hook(target), with_kwargs=True
                        )
                    )
                self._handles.append(
                    target.module.register_forward_hook(
                        self._make_post_hook(target), with_kwargs=True
                    )
                )
            except Exception:
                self.close()
                raise

    def close(self) -> None:
        for extension in reversed(self._extensions):
            extension.close()
        self._extensions.clear()
        for handle in self._handles:
            handle.remove()
        self._handles.clear()
        self._training_inputs.clear()

    def attach_extension(self, extension: Any) -> None:
        """Make an auxiliary hook set part of this session's lifetime."""

        self._extensions.append(extension)

    def _make_pre_hook(self, target: TrainRefTarget):
        def pre_hook(_module, args, kwargs):
            if should_skip_runtime_hook():
                return None
            training_input = self._training_inputs.get(target.module_name)
            if training_input is None:
                raise TrainRefDataError(
                    f"training input cache is missing {target.module_name}"
                )
            new_args = list(args)
            new_kwargs = dict(kwargs)
            for index, value in enumerate(new_args):
                if isinstance(value, torch.Tensor):
                    new_args[index] = prepare_mock_input(
                        training_input,
                        value,
                        self.config,
                        target.target_kind,
                    )
                    return tuple(new_args), new_kwargs
            for name, value in new_kwargs.items():
                if isinstance(value, torch.Tensor):
                    new_kwargs[name] = prepare_mock_input(
                        training_input,
                        value,
                        self.config,
                        target.target_kind,
                    )
                    return tuple(new_args), new_kwargs
            raise TrainRefDataError(
                f"{target.module_name} forward input does not contain a Tensor"
            )

        return pre_hook

    def _make_post_hook(self, target: TrainRefTarget):
        def post_hook(_module, _args, _kwargs, output):
            if should_skip_runtime_hook():
                return
            dump_output(output, target.module_name, self.config, self.rank)

        return post_hook
