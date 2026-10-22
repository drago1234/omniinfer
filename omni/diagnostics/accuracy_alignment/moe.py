# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""Dense-MLP and MoE train-reference hooks for Pangu UltraMoE."""

from __future__ import annotations

import contextvars
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from .config import TrainRefConfig
from .dump_io import (
    TrainRefDataError,
    dump_output,
    load_reference_tensor,
    primary_tensor,
)
from .hooks import (
    ModuleTrainRefHooks,
    ModuleTrainRefTarget,
    close_hook_groups,
    get_tp_rank,
)
from .train_ref import should_skip_runtime_hook


@dataclass(frozen=True)
class _MoeCall:
    hooks: MoeTrainRefHooks
    original_tokens: int
    tp_size: int
    tp_rank: int


_ACTIVE_MOE_CALL: contextvars.ContextVar[_MoeCall | None] = contextvars.ContextVar(
    "omni_accuracy_alignment_active_moe_call", default=None
)


def _required_reference(
    references: dict[str, torch.Tensor],
    name: str,
) -> torch.Tensor:
    value = references.get(name)
    if value is None:
        raise TrainRefDataError(f"cached expert reference is missing {name}")
    return value


class _StaticMethodHandle:
    def __init__(self, owner: type, name: str, replacement: Any) -> None:
        self._owner = owner
        self._name = name
        self._descriptor = owner.__dict__.get(name)
        if self._descriptor is None:
            raise AttributeError(f"{owner.__name__}.{name} does not exist")
        setattr(owner, name, staticmethod(replacement))
        self._active = True

    def remove(self) -> None:
        if self._active:
            setattr(self._owner, self._name, self._descriptor)
            self._active = False


class _InstanceMethodHandle:
    def __init__(self, instance: Any, name: str, replacement: Any) -> None:
        from .hooks import MethodHandle

        self._handle = MethodHandle(instance, name, replacement)

    def remove(self) -> None:
        self._handle.remove()


class MoeTrainRefHooks:
    def __init__(self, config: TrainRefConfig, rank: int) -> None:
        self.config = config
        self.rank = rank
        self.tp_rank = get_tp_rank()
        self._handles: list[Any] = []
        self._extensions: list[Any] = []
        self._custom: dict[str, torch.Tensor] = {}
        self._expert_reference_cache: dict[str, torch.Tensor] | None = None
        self._has_expert_reference = False
        self._expert_ranks: tuple[int, ...] = ()

    @property
    def installed_hook_count(self) -> int:
        return len(self._handles) + sum(
            int(getattr(extension, "installed_hook_count", 0))
            for extension in self._extensions
        )

    def install(self, mlp: torch.nn.Module) -> None:
        self.close()
        generic_targets = self._generic_targets(mlp)
        generic = ModuleTrainRefHooks(self.config, self.rank)
        generic.install(generic_targets)
        self._extensions.append(generic)
        if not self._is_moe(mlp):
            return
        try:
            self._load_custom_router_reference()
            self._install_gate_hook(mlp)
            self._install_quant_method_hooks(mlp)
        except Exception:
            self.close()
            raise

    def close(self) -> None:
        close_hook_groups(self._handles, self._extensions)
        self._custom.clear()
        self._expert_reference_cache = None
        self._has_expert_reference = False
        self._expert_ranks = ()

    def _name(self, training_name: str) -> str:
        return f"decoder.layers.{self.config.layer_id}.{training_name}"

    @staticmethod
    def _is_moe(mlp: torch.nn.Module) -> bool:
        return hasattr(mlp, "experts") and hasattr(mlp, "gate")

    def _generic_targets(
        self,
        mlp: torch.nn.Module,
    ) -> list[ModuleTrainRefTarget]:
        targets = []
        if self._is_moe(mlp):
            candidates = (
                ("gate", "mlp.router", True, None),
                (
                    "experts",
                    "mlp.experts",
                    True,
                    self._select_experts_output,
                ),
                (
                    "shared_experts.gate_up_proj",
                    "mlp.experts.shared_expert.linear_fc1",
                    True,
                    None,
                ),
                (
                    "shared_experts.down_proj",
                    "mlp.experts.shared_expert.linear_fc2",
                    True,
                    None,
                ),
            )
        else:
            candidates = (
                ("gate_up_proj", "mlp.linear_fc1", True, None),
                ("down_proj", "mlp.linear_fc2", True, None),
            )
        for path, training_name, inject, selector in candidates:
            module = self._resolve_module(mlp, path)
            if module is None:
                continue
            targets.append(
                ModuleTrainRefTarget(
                    module_name=self._name(training_name),
                    training_name=training_name,
                    module=module,
                    inject_input=inject,
                    output_selector=selector or primary_tensor,
                    enforce_token_count=training_name
                    not in {"mlp.router", "mlp.experts"},
                )
            )
        return targets

    @staticmethod
    def _resolve_module(parent: Any, path: str) -> torch.nn.Module | None:
        value = parent
        for piece in path.split("."):
            value = getattr(value, piece, None)
            if value is None:
                return None
        return value if isinstance(value, torch.nn.Module) else None

    @staticmethod
    def _select_experts_output(output: Any) -> torch.Tensor:
        if isinstance(output, tuple) and len(output) >= 2:
            return primary_tensor(output[1])
        return primary_tensor(output)

    def _load_custom_router_reference(self) -> None:
        if self.config.mhc_custom_dump_dir is None:
            return
        lid1 = self.config.layer_id + 1
        gate_path = self.config.mhc_custom_dump_dir / f"moe_gate_output_{lid1}_rank0.pt"
        if self.config.should_mock("moe_gate_output") and gate_path.is_file():
            self._custom["gate"] = load_reference_tensor(gate_path)
        if self.config.should_mock("gmm_experts_output", "mlp.experts"):
            self._has_expert_reference = self._validate_expert_reference_paths()

    def _slice_custom_for_runtime(
        self,
        tensor: torch.Tensor,
        runtime: torch.Tensor,
        tp_size: int,
        tp_rank: int,
    ) -> torch.Tensor:
        if self.config.role == "P":
            if tensor.shape[0] < self.config.mock_shape:
                raise TrainRefDataError(
                    "custom MoE training sequence is shorter than mock shape"
                )
            value = tensor[: self.config.mock_shape]
        else:
            if tensor.shape[0] < self.config.mock_shape:
                raise TrainRefDataError(
                    "custom MoE training sequence is shorter than decode index"
                )
            value = tensor.narrow(0, self.config.mock_shape - 1, 1)
        if tuple(value.shape) == tuple(runtime.shape):
            return value.to(device=runtime.device, dtype=runtime.dtype)
        if tp_size > 1 and value.shape[1:] == runtime.shape[1:]:
            padded = -(-value.shape[0] // tp_size) * tp_size
            if padded != value.shape[0]:
                value = torch.nn.functional.pad(
                    value,
                    (0, 0) * (value.ndim - 1) + (0, padded - value.shape[0]),
                )
            local_tokens = padded // tp_size
            start = tp_rank * local_tokens
            value = value.narrow(0, start, local_tokens)
        if tuple(value.shape) != tuple(runtime.shape):
            raise TrainRefDataError(
                f"custom MoE shape {tuple(value.shape)} does not match runtime "
                f"shape {tuple(runtime.shape)}"
            )
        return value.to(device=runtime.device, dtype=runtime.dtype)

    def _install_gate_hook(self, mlp: torch.nn.Module) -> None:
        gate = self._resolve_module(mlp, "gate")
        if gate is None:
            return

        def gate_post(_module, _args, _kwargs, output):
            if should_skip_runtime_hook():
                return None
            tensor = primary_tensor(output)
            call = _ACTIVE_MOE_CALL.get()
            tp_size = (
                call.tp_size
                if call is not None
                else int(getattr(mlp, "tp_size", 1) or 1)
            )
            tp_rank = (
                call.tp_rank
                if call is not None
                else int(getattr(mlp, "tp_rank", 0) or 0)
            )
            dump_output(
                tensor,
                f"moe_gate_output_{self.config.layer_id + 1}",
                self.config,
                self.rank,
                enforce_token_count=False,
            )
            reference = self._custom.get("gate")
            if reference is None or not self.config.should_mock("moe_gate_output"):
                return None
            replacement = self._slice_custom_for_runtime(
                reference, tensor, tp_size, tp_rank
            )
            if isinstance(output, tuple):
                return (replacement, *output[1:])
            return replacement

        self._handles.append(gate.register_forward_hook(gate_post, with_kwargs=True))

    def _install_quant_method_hooks(self, mlp: torch.nn.Module) -> None:
        fused_moe = mlp.experts
        routed_experts = getattr(fused_moe, "routed_experts", None)
        quant_method = getattr(routed_experts, "quant_method", None)
        if quant_method is None:
            raise RuntimeError("cannot locate routed-experts quant method")
        from omni_npu.layers.fused_moe.layer import (
            NPUFusedMoERunner,
            NPUUnquantizedFusedMoEMethod,
        )

        supports_expert_recompute = isinstance(
            quant_method,
            NPUUnquantizedFusedMoEMethod,
        )
        if self._has_expert_reference and not supports_expert_recompute:
            raise TrainRefDataError(
                "training-side apply_experts recompute only supports the "
                "unquantized NPU MoE method; remove the expert raw files to "
                "dump the native quantized apply_experts output"
            )
        original_apply = quant_method.apply

        def apply_wrapper(_method, layer, hidden_states, *args, **kwargs):
            call = _MoeCall(
                hooks=self,
                original_tokens=int(hidden_states.shape[0]),
                tp_size=int(getattr(quant_method, "tp_size", 1) or 1),
                tp_rank=int(getattr(quant_method, "tp_rank", 0) or 0),
            )
            token = _ACTIVE_MOE_CALL.set(call)
            try:
                return original_apply(layer, hidden_states, *args, **kwargs)
            finally:
                _ACTIVE_MOE_CALL.reset(token)

        self._handles.append(
            _InstanceMethodHandle(quant_method, "apply", apply_wrapper)
        )

        original_experts = quant_method.apply_experts

        def experts_wrapper(
            _method,
            layer,
            prepare_permute_result,
            activation="silu",
        ):
            if should_skip_runtime_hook():
                return original_experts(layer, prepare_permute_result, activation)
            if not self._has_expert_reference or not self.config.should_mock(
                "gmm_experts_output", "mlp.experts"
            ):
                output = original_experts(layer, prepare_permute_result, activation)
            else:
                output = self._recompute_apply_experts(
                    layer,
                    prepare_permute_result,
                    activation,
                )
            dump_output(
                output,
                f"gmm_experts_output_{self.config.layer_id + 1}",
                self.config,
                self.rank,
                enforce_token_count=False,
            )
            return output

        self._handles.append(
            _InstanceMethodHandle(quant_method, "apply_experts", experts_wrapper)
        )

        original_select = NPUFusedMoERunner.select_experts

        def select_wrapper(*args, **kwargs):
            topk_weights, topk_ids = original_select(*args, **kwargs)
            call = _ACTIVE_MOE_CALL.get()
            if (
                call is not None
                and call.hooks is self
                and not should_skip_runtime_hook()
            ):
                dump_output(
                    topk_weights,
                    f"moe_router_scores_{self.config.layer_id + 1}",
                    self.config,
                    self.rank,
                    enforce_token_count=False,
                )
                dump_output(
                    topk_ids,
                    f"moe_router_indices_{self.config.layer_id + 1}",
                    self.config,
                    self.rank,
                    enforce_token_count=False,
                )
            return topk_weights, topk_ids

        self._handles.append(
            _StaticMethodHandle(NPUFusedMoERunner, "select_experts", select_wrapper)
        )

    def _expert_path(self, prefix: str, rank: int) -> Path:
        root = self.config.mhc_custom_dump_dir
        if root is None:
            raise TrainRefDataError("custom expert references require a dump directory")
        return root / (
            f"{prefix}_{self.config.layer_id + 1}_rank{rank}.pt"
        )

    def _validate_expert_reference_paths(self) -> bool:
        prefixes = (
            "moe_experts_up_permuted",
            "moe_experts_up_tokens_per_expert",
            "moe_experts_up_weight1",
            "moe_experts_down_weight2",
        )
        primary_prefix = prefixes[0]
        configured_ranks = tuple(range(self.config.train_ep_size))
        if all(
            self._expert_path(primary_prefix, rank).is_file()
            for rank in configured_ranks
        ):
            ranks = configured_ranks
        else:
            detected = []
            for rank in range(64):
                if self._expert_path(primary_prefix, rank).is_file():
                    detected.append(rank)
                else:
                    break
            ranks = tuple(detected)
        if not ranks:
            return False
        paths = [
            self._expert_path(prefix, rank) for rank in ranks for prefix in prefixes
        ]
        missing = [str(path) for path in paths if not path.is_file()]
        if missing:
            raise TrainRefDataError(
                "incomplete custom expert reference set; missing: " + ", ".join(missing)
            )
        self._expert_ranks = ranks
        return True

    def _load_expert_references(self) -> dict[str, torch.Tensor]:
        if self._expert_reference_cache is not None:
            return self._expert_reference_cache
        prefixes = (
            "moe_experts_up_permuted",
            "moe_experts_up_tokens_per_expert",
            "moe_experts_up_weight1",
            "moe_experts_down_weight2",
        )
        self._expert_reference_cache = {
            prefix: torch.cat(
                [
                    # Preserve saved NPU tensor formats (for example
                    # FRACTAL_NZ); forcing these weights through CPU changes
                    # grouped-matmul kernel selection and defeats alignment.
                    load_reference_tensor(
                        self._expert_path(prefix, rank),
                        map_location=None,
                    )
                    for rank in self._expert_ranks
                ],
                dim=0,
            )
            for prefix in prefixes
        }
        return self._expert_reference_cache

    def _recompute_apply_experts(
        self,
        layer: Any,
        prepare_permute_result: Any,
        activation: str,
    ) -> torch.Tensor:
        if activation != "silu":
            raise TrainRefDataError(
                f"custom expert recompute only supports silu, got {activation}"
            )
        import torch_npu

        references = self._load_expert_references()
        hidden_target = prepare_permute_result.hidden_states_sorted_by_experts
        tokens_target = prepare_permute_result.expert_tokens.to(torch.int64)
        routed = layer.routed_experts
        hidden_reference = _required_reference(
            references, "moe_experts_up_permuted"
        )
        tokens_reference = _required_reference(
            references, "moe_experts_up_tokens_per_expert"
        )
        w13_reference = _required_reference(references, "moe_experts_up_weight1")
        w2_reference = _required_reference(references, "moe_experts_down_weight2")
        hidden = hidden_reference.to(
            device=hidden_target.device, dtype=hidden_target.dtype
        )
        tokens = tokens_reference.to(
            device=tokens_target.device, dtype=tokens_target.dtype
        )
        w13 = w13_reference.to(
            device=routed.w13_weight.device, dtype=routed.w13_weight.dtype
        )
        w2 = w2_reference.to(
            device=routed.w2_weight.device, dtype=routed.w2_weight.dtype
        )
        group_list_type = int(layer.moe_config.moe_parallel_config.use_ep)
        gate_up = torch_npu.npu_grouped_matmul(
            [hidden],
            [w13],
            bias=None,
            group_list=tokens,
            split_item=3,
            group_type=0,
            group_list_type=group_list_type,
        )[0]
        intermediate = torch_npu.npu_swiglu(gate_up)
        return torch_npu.npu_grouped_matmul(
            [intermediate],
            [w2],
            bias=None,
            group_list=tokens,
            split_item=3,
            group_type=0,
            group_list_type=group_list_type,
        )[0]
