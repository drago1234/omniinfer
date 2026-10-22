# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""Reusable instance-local hooks for accuracy-alignment modules."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from types import MethodType
from typing import Any

import torch

from .config import TrainRefConfig
from .dump_io import (
    LayerTrainingInputs,
    TrainRefDataError,
    dump_output,
    load_layer_training_inputs,
    prepare_layer_mock_input,
    primary_tensor,
)
from .train_ref import should_skip_runtime_hook


def get_tp_rank() -> int:
    try:
        from vllm.distributed import get_tensor_model_parallel_rank

        return int(get_tensor_model_parallel_rank())
    except (ImportError, RuntimeError, AttributeError, AssertionError):
        return 0


def gather_tp_output(module: Any, tensor: torch.Tensor) -> torch.Tensor:
    """Gather ColumnParallel-like outputs while leaving replicated rows alone."""

    tp_size = int(getattr(module, "tp_size", 1) or 1)
    if tp_size <= 1 or tensor.ndim == 0:
        return tensor
    if "Replicated" in type(module).__name__:
        return tensor
    if bool(getattr(module, "reduce_results", False)):
        return tensor
    if bool(getattr(module, "gather_output", False)):
        return tensor
    try:
        from vllm.distributed import get_tp_group

        return get_tp_group().all_gather(tensor, dim=-1)
    except (ImportError, RuntimeError, AttributeError):
        return tensor


class MethodHandle:
    """Install and later restore one method on a single instance."""

    def __init__(self, instance: Any, name: str, wrapper: Callable[..., Any]):
        self._instance = instance
        self._name = name
        self._had_instance_value = name in vars(instance)
        self._instance_value = vars(instance).get(name)
        setattr(instance, name, MethodType(wrapper, instance))
        self._active = True

    def remove(self) -> None:
        if not self._active:
            return
        if self._had_instance_value:
            setattr(self._instance, self._name, self._instance_value)
        else:
            delattr(self._instance, self._name)
        self._active = False


def close_handles(handles: list[Any]) -> None:
    """Remove hook handles in reverse installation order."""

    for handle in reversed(handles):
        handle.remove()
    handles.clear()


def close_hook_groups(handles: list[Any], extensions: list[Any]) -> None:
    """Close direct handles followed by nested hook owners."""

    close_handles(handles)
    for extension in reversed(extensions):
        extension.close()
    extensions.clear()


def close_runner_session(runner: Any, attr_name: str) -> None:
    """Close and detach one previously installed runner session."""

    previous = getattr(runner, attr_name, None)
    if previous is None:
        return
    previous.close()
    delattr(runner, attr_name)


@dataclass(frozen=True)
class ModuleTrainRefTarget:
    module_name: str
    training_name: str
    module: torch.nn.Module
    inject_input: bool = True
    gather_output: bool = True
    direct_forward: bool = False
    enforce_token_count: bool = True
    output_selector: Callable[[Any], torch.Tensor] = primary_tensor


class ModuleTrainRefHooks:
    """Inject Megatron input.0 and dump one selected module output tensor."""

    def __init__(self, config: TrainRefConfig, rank: int) -> None:
        self.config = config
        self.rank = rank
        self.tp_rank = get_tp_rank()
        self._handles: list[Any] = []
        self._references: dict[str, LayerTrainingInputs] = {}

    @property
    def installed_hook_count(self) -> int:
        return len(self._handles)

    def install(self, targets: list[ModuleTrainRefTarget]) -> None:
        self.close()
        self._references = {
            target.module_name: load_layer_training_inputs(
                self.config, target.training_name
            )
            for target in targets
            if self._injects(target)
        }
        try:
            for target in targets:
                if target.direct_forward:
                    self._install_direct_forward(target)
                else:
                    self._install_module_hooks(target)
        except Exception:
            self.close()
            raise

    def close(self) -> None:
        close_handles(self._handles)
        self._references.clear()

    def _injects(self, target: ModuleTrainRefTarget) -> bool:
        return target.inject_input and self.config.should_mock(
            target.training_name,
            target.module_name,
        )

    def _replace_first_tensor(
        self,
        target: ModuleTrainRefTarget,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> tuple[tuple[Any, ...], dict[str, Any]]:
        reference = self._references.get(target.module_name)
        if reference is None:
            raise TrainRefDataError(
                f"training input cache is missing {target.module_name}"
            )
        new_args = list(args)
        new_kwargs = dict(kwargs)
        for index, value in enumerate(new_args):
            if isinstance(value, torch.Tensor):
                new_args[index] = prepare_layer_mock_input(
                    reference,
                    value,
                    self.config,
                    self.tp_rank,
                )
                return tuple(new_args), new_kwargs
            replacement = self._replace_mapping_tensor(reference, value)
            if replacement is not None:
                new_args[index] = replacement
                return tuple(new_args), new_kwargs
        for name, value in new_kwargs.items():
            if isinstance(value, torch.Tensor):
                new_kwargs[name] = prepare_layer_mock_input(
                    reference,
                    value,
                    self.config,
                    self.tp_rank,
                )
                return tuple(new_args), new_kwargs
            replacement = self._replace_mapping_tensor(reference, value)
            if replacement is not None:
                new_kwargs[name] = replacement
                return tuple(new_args), new_kwargs
        raise TrainRefDataError(f"{target.module_name} input does not contain a Tensor")

    def _replace_mapping_tensor(
        self,
        reference: LayerTrainingInputs,
        value: Any,
    ) -> dict[str, Any] | None:
        if not isinstance(value, dict):
            return None
        # Fused MHC may pass both a bf16 main input and an fp32 copy used only
        # by the router.  The module-level Megatron input.0 corresponds to the
        # bf16 activation; preserve the auxiliary fp32 value exactly.
        for key in ("hidden_states_bf16", "hidden_states"):
            tensor = value.get(key)
            if isinstance(tensor, torch.Tensor):
                replacement = dict(value)
                replacement[key] = prepare_layer_mock_input(
                    reference,
                    tensor,
                    self.config,
                    self.tp_rank,
                )
                return replacement
        return None

    def _dump(self, target: ModuleTrainRefTarget, output: Any) -> None:
        tensor = target.output_selector(output)
        if target.gather_output:
            tensor = gather_tp_output(target.module, tensor)
        dump_output(
            tensor,
            target.module_name,
            self.config,
            self.rank,
            enforce_token_count=target.enforce_token_count,
        )

    def _install_module_hooks(self, target: ModuleTrainRefTarget) -> None:
        if self._injects(target):

            def pre_hook(_module, args, kwargs):
                if should_skip_runtime_hook():
                    return None
                return self._replace_first_tensor(target, args, kwargs)

            self._handles.append(
                target.module.register_forward_pre_hook(pre_hook, with_kwargs=True)
            )

        def post_hook(_module, _args, _kwargs, output):
            if not should_skip_runtime_hook():
                self._dump(target, output)

        self._handles.append(
            target.module.register_forward_hook(post_hook, with_kwargs=True)
        )

    def _install_direct_forward(self, target: ModuleTrainRefTarget) -> None:
        original = target.module.forward

        def wrapped(_module, *args, **kwargs):
            call_args = args
            call_kwargs = kwargs
            if self._injects(target) and not should_skip_runtime_hook():
                call_args, call_kwargs = self._replace_first_tensor(
                    target, args, kwargs
                )
            output = original(*call_args, **call_kwargs)
            if not should_skip_runtime_hook():
                self._dump(target, output)
            return output

        self._handles.append(MethodHandle(target.module, "forward", wrapped))
