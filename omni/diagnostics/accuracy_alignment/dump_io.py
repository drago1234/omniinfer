# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""Tensor loading, slicing, validation, and saving for train-ref."""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from .config import TrainRefConfig


class TrainRefDataError(RuntimeError):
    """Raised when a training dump cannot safely replace a runtime input."""


_ROW_PARALLEL_INPUT_MODULES = frozenset(
    {
        "self_attention.linear_proj",
        "mlp.linear_fc2",
        "mlp.experts.shared_expert.linear_fc2",
    }
)


@dataclass(frozen=True)
class LayerTrainingInputs:
    """Training input.0 tensors saved by one or more training TP ranks."""

    module_name: str
    by_rank: dict[int, torch.Tensor]


def _load_payload(
    path: Path,
    map_location: str | torch.device | None = "cpu",
) -> Any:
    if not path.is_file():
        raise TrainRefDataError(f"training dump file does not exist: {path}")
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except Exception as exc:
        raise TrainRefDataError(f"failed to load training dump {path}: {exc}") from exc


def _tensor_payload(payload: Any) -> dict[str, Any] | None:
    if not isinstance(payload, dict):
        return None
    tensors = payload.get("tensors")
    return tensors if isinstance(tensors, dict) else None


def load_reference_tensor(
    path: Path,
    *,
    map_location: str | torch.device | None = "cpu",
) -> torch.Tensor:
    """Load either a raw tensor or a Megatron input/output dump."""

    payload = _load_payload(path, map_location=map_location)
    if isinstance(payload, torch.Tensor):
        tensor = payload
    elif (tensors := _tensor_payload(payload)) is not None:
        tensor = tensors.get("input.0")
        if not isinstance(tensor, torch.Tensor):
            tensor = tensors.get("output.0")
    else:
        tensor = None
    if not isinstance(tensor, torch.Tensor):
        raise TrainRefDataError(
            f"training dump {path} must be a Tensor or contain "
            "tensors['input.0'|'output.0']"
        )
    if tensor.ndim == 0:
        raise TrainRefDataError(f"training input in {path} must not be scalar")
    return tensor


def load_training_input(path: Path) -> torch.Tensor:
    """Load the primary input from a Megatron module dump."""

    payload = _load_payload(path)
    tensors = _tensor_payload(payload)
    if tensors is None:
        raise TrainRefDataError(
            f"training dump {path} must contain a 'tensors' dictionary"
        )
    tensor = tensors.get("input.0")
    if not isinstance(tensor, torch.Tensor):
        raise TrainRefDataError(f"training dump {path} must contain tensors['input.0']")
    if tensor.ndim == 0:
        raise TrainRefDataError(f"training input in {path} must not be scalar")
    return tensor


def load_layer_training_inputs(
    config: TrainRefConfig,
    module_name: str,
) -> LayerTrainingInputs:
    """Load rank-0 and any available TP shards using the legacy filenames."""

    by_rank = {0: load_training_input(config.layer_training_dump_path(module_name, 0))}
    for rank in range(1, config.train_tp_size):
        path = config.layer_training_dump_path(module_name, rank)
        if path.is_file():
            by_rank[rank] = load_training_input(path)
    return LayerTrainingInputs(module_name=module_name, by_rank=by_rank)


def _normalize_training_input(tensor: torch.Tensor, target_kind: str) -> torch.Tensor:
    value = tensor
    if target_kind == "lm_head" and value.ndim == 3 and value.shape[1] > 0:
        # Megatron stores [sequence, speculative-head, hidden].
        value = value[:, 0, :]
    if value.ndim >= 3 and value.shape[0] == 1:
        value = value.squeeze(0)
    if value.ndim >= 2 and value.shape[1] == 1:
        value = value.squeeze(1)
    return value


def prepare_mock_input(
    training_input: torch.Tensor,
    runtime_input: torch.Tensor,
    config: TrainRefConfig,
    target_kind: str,
) -> torch.Tensor:
    """Slice the golden sequence for P/D and exactly match the runtime input."""

    if runtime_input.ndim == 0:
        raise TrainRefDataError(f"{target_kind} runtime input must not be scalar")
    expected_runtime_tokens = config.mock_shape if config.role == "P" else 1
    # vLLM normally asks lm_head for the one token selected by logits_indices,
    # even after a multi-token prefill. That token corresponds to the final
    # prompt row in the training dump.
    selected_lm_head_token = target_kind == "lm_head" and runtime_input.shape[0] == 1
    if runtime_input.shape[0] != expected_runtime_tokens and not selected_lm_head_token:
        raise TrainRefDataError(
            f"{target_kind} runtime token count {runtime_input.shape[0]} does not "
            f"match role={config.role} expected={expected_runtime_tokens}"
        )

    value = _normalize_training_input(training_input, target_kind)
    if value.ndim == 0 or value.shape[0] < config.mock_shape:
        available = 0 if value.ndim == 0 else value.shape[0]
        raise TrainRefDataError(
            f"{target_kind} training sequence is too short: "
            f"available={available}, required={config.mock_shape}"
        )
    if selected_lm_head_token:
        mock = value.narrow(0, config.mock_shape - 1, 1)
    elif config.role == "P":
        mock = value[: config.mock_shape]
    else:
        mock = value.narrow(0, config.mock_shape - 1, 1)

    if tuple(mock.shape) != tuple(runtime_input.shape):
        raise TrainRefDataError(
            f"{target_kind} mock shape {tuple(mock.shape)} does not match "
            f"runtime shape {tuple(runtime_input.shape)}"
        )
    return mock.to(device=runtime_input.device, dtype=runtime_input.dtype)


def prepare_mhc_mock_input(
    training_input: torch.Tensor,
    runtime_input: torch.Tensor,
    config: TrainRefConfig,
    boundary_name: str,
) -> torch.Tensor:
    """Slice and strictly validate a legacy raw MHC boundary tensor."""

    return prepare_mock_input(
        training_input,
        runtime_input,
        config,
        f"mhc_{boundary_name}",
    )


def _slice_training_sequence(
    value: torch.Tensor,
    config: TrainRefConfig,
    target_kind: str,
    runtime_input: torch.Tensor | None = None,
) -> torch.Tensor:
    if _is_legacy_ncl_layout(value, runtime_input):
        # Legacy convolution dumps use NCL while the inference-side module
        # receives [tokens, channels].  Detect this before the generic
        # leading-dimension squeeze so token slicing stays on dim 0.
        value = value.squeeze(0).transpose(0, 1).contiguous()
    else:
        value = _normalize_training_input(value, target_kind)
    runtime_tokens = (
        runtime_input.shape[0]
        if runtime_input is not None and runtime_input.ndim > 0
        else 1
    )
    required_tokens = (
        config.mock_shape
        if config.role == "P"
        else config.mock_shape - 1 + runtime_tokens
    )
    if value.ndim == 0 or value.shape[0] < required_tokens:
        available = 0 if value.ndim == 0 else value.shape[0]
        raise TrainRefDataError(
            f"{target_kind} training sequence is too short: "
            f"available={available}, required={required_tokens}"
        )
    if config.role == "P":
        return value[: config.mock_shape]
    start = config.mock_shape - 1
    return value.narrow(0, start, runtime_tokens)


def _is_legacy_ncl_layout(
    value: torch.Tensor,
    runtime_input: torch.Tensor | None,
) -> bool:
    if runtime_input is None or value.ndim != 3 or runtime_input.ndim != 2:
        return False
    if value.shape[0] != 1:
        return False
    channel_size = runtime_input.shape[-1]
    return value.shape[1] == channel_size and value.shape[2] != channel_size


def prepare_layer_mock_input(
    training_inputs: LayerTrainingInputs,
    runtime_input: torch.Tensor,
    config: TrainRefConfig,
    tp_rank: int,
) -> torch.Tensor:
    """Adapt legacy TP-rank input.0 files to one inference-rank input."""

    module_name = training_inputs.module_name
    expected_tokens = config.mock_shape if config.role == "P" else None
    invalid_tokens = runtime_input.ndim == 0 or runtime_input.shape[0] < 1
    if expected_tokens is not None:
        invalid_tokens = invalid_tokens or runtime_input.shape[0] != expected_tokens
    if invalid_tokens:
        expected = str(expected_tokens) if expected_tokens is not None else "at least 1"
        raise TrainRefDataError(
            f"{module_name} runtime token count must be {expected}, got "
            f"{tuple(runtime_input.shape)}"
        )

    sliced = {
        rank: _slice_training_sequence(
            tensor,
            config,
            module_name,
            runtime_input,
        )
        for rank, tensor in training_inputs.by_rank.items()
    }
    preferred_rank = tp_rank if module_name in _ROW_PARALLEL_INPUT_MODULES else 0
    rank_zero = sliced.get(0)
    if rank_zero is None:
        raise TrainRefDataError(f"{module_name} has no training rank-0 input")
    candidate = sliced.get(preferred_rank, rank_zero)
    if tuple(candidate.shape) == tuple(runtime_input.shape):
        return candidate.to(
            device=runtime_input.device,
            dtype=runtime_input.dtype,
        )

    ordered = [
        tensor for rank in sorted(sliced) if (tensor := sliced.get(rank)) is not None
    ]
    if len(ordered) > 1 and all(
        tensor.shape[:-1] == ordered[0].shape[:-1] for tensor in ordered
    ):
        combined = torch.cat(ordered, dim=-1)
        if tuple(combined.shape) == tuple(runtime_input.shape):
            candidate = combined
        elif (
            combined.shape[:-1] == runtime_input.shape[:-1]
            and combined.shape[-1] % runtime_input.shape[-1] == 0
        ):
            parts = combined.shape[-1] // runtime_input.shape[-1]
            part = tp_rank % parts
            start = part * runtime_input.shape[-1]
            candidate = combined.narrow(-1, start, runtime_input.shape[-1])

    if tuple(candidate.shape) != tuple(runtime_input.shape):
        raise TrainRefDataError(
            f"{module_name} mock shape {tuple(candidate.shape)} does not match "
            f"runtime shape {tuple(runtime_input.shape)}; loaded train ranks="
            f"{sorted(training_inputs.by_rank)}"
        )
    return candidate.to(device=runtime_input.device, dtype=runtime_input.dtype)


def primary_tensor(value: Any) -> torch.Tensor:
    """Return output.0 while accepting tensor, tuple/list, and mapping outputs."""

    if isinstance(value, torch.Tensor):
        return value
    if isinstance(value, Mapping):
        for item in value.values():
            try:
                return primary_tensor(item)
            except TrainRefDataError:
                continue
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        for item in value:
            try:
                return primary_tensor(item)
            except TrainRefDataError:
                continue
    raise TrainRefDataError("module output does not contain a Tensor")


def dump_output(
    output: Any,
    module_name: str,
    config: TrainRefConfig,
    rank: int,
    *,
    enforce_token_count: bool = True,
) -> Path | None:
    tensor = primary_tensor(output)
    if _should_skip_output(tensor, config, enforce_token_count):
        return None
    tensor = tensor.detach().cpu()
    stage_dir = config.infer_dump_dir / config.stage
    stage_dir.mkdir(parents=True, exist_ok=True)
    safe_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", module_name)
    path = stage_dir / (f"{safe_name}_{config.stage}_{config.dump_tag}_rank{rank}.pt")
    torch.save(tensor, path)
    return path


def _should_skip_output(
    tensor: torch.Tensor,
    config: TrainRefConfig,
    enforce_token_count: bool,
) -> bool:
    if not enforce_token_count or config.role != "P":
        return False
    return tensor.ndim > 0 and tensor.shape[0] != config.mock_shape
