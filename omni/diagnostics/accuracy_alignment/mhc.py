# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""MHC boundary injection and observation for train-ref.

The 0.25.1 fused MHC path invokes methods directly instead of calling the
DecoderLayer module.  Instance-local wrappers keep the original methods and
only replace/dump tensors at the same semantic boundaries used by the legacy
alignment tool.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import torch

from .config import TrainRefConfig
from .dump_io import (
    dump_output,
    load_reference_tensor,
    prepare_mhc_mock_input,
)
from .hooks import MethodHandle, close_handles
from .train_ref import should_skip_runtime_hook

_MHC_BOUNDARIES = (
    "pre_in",
    "post_in",
    "post_h_post",
    "post_residual",
    "post_h_res",
)


@dataclass(frozen=True)
class MhcTrainRefTarget:
    layer_id: int
    branch: str
    module: Any
    decoder_layer: Any

    @property
    def name(self) -> str:
        return f"layer_{self.layer_id}_{self.branch}_mhc"


class MhcTrainRefHooks:
    """Own MHC method wrappers and their cached legacy reference tensors."""

    def __init__(self, config: TrainRefConfig, rank: int) -> None:
        self.config = config
        self.rank = rank
        self._handles: list[MethodHandle] = []
        self._references: dict[tuple[str, str], torch.Tensor] = {}
        self._wrapped_decoder_ids: set[int] = set()

    @property
    def installed_hook_count(self) -> int:
        return len(self._handles)

    def install(self, targets: list[MhcTrainRefTarget]) -> None:
        self.close()
        references: dict[tuple[str, str], torch.Tensor] = {}
        for target in targets:
            selected = [
                boundary
                for boundary in _MHC_BOUNDARIES
                if self._requested_mock(target, boundary)
            ]
            if not selected:
                continue
            if self.config.mhc_custom_dump_dir is None:
                if self.config.mode == "real":
                    raise RuntimeError(
                        f"{target.branch} MHC mock selected but "
                        "OMNI_NPU_MHC_CUSTOM_DUMP_DIR is unset"
                    )
                continue
            paths = {
                boundary: self.config.mhc_training_dump_path(target.branch, boundary)
                for boundary in selected
            }
            existing = {boundary: path.is_file() for boundary, path in paths.items()}
            if any(existing.values()) and not all(existing.values()):
                missing = [
                    str(paths.get(boundary))
                    for boundary, exists in existing.items()
                    if not exists
                ]
                raise RuntimeError(
                    f"incomplete {target.branch} MHC custom reference set: "
                    + ", ".join(missing)
                )
            if not any(existing.values()):
                if self.config.mode == "real":
                    raise RuntimeError(
                        f"selected {target.branch} MHC custom references are "
                        "missing: " + ", ".join(map(str, paths.values()))
                    )
                continue
            for boundary, path in paths.items():
                references[(target.name, boundary)] = load_reference_tensor(path)
        self._references = references
        try:
            for target in targets:
                self._install_module_methods(target)
                self._install_fused_decoder_output(target)
        except Exception:
            self.close()
            raise

    def close(self) -> None:
        close_handles(self._handles)
        self._references.clear()
        self._wrapped_decoder_ids.clear()

    def _reference(self, target: MhcTrainRefTarget, boundary: str) -> torch.Tensor:
        value = self._references.get((target.name, boundary))
        if value is None:
            raise RuntimeError(
                f"MHC training reference is missing: {target.name}/{boundary}"
            )
        return value

    @staticmethod
    def _tag(target: MhcTrainRefTarget, boundary: str) -> str:
        prefix = "pangu_transformer_hc"
        if target.branch == "mlp":
            prefix = "pangu_transformer_mlp_hc"
        return f"{prefix}_{boundary}"

    def _requested_mock(self, target: MhcTrainRefTarget, boundary: str) -> bool:
        return self.config.should_mock(
            self._tag(target, boundary),
            f"{target.branch}_mhc.{boundary}",
        )

    def _should_mock(self, target: MhcTrainRefTarget, boundary: str) -> bool:
        return (
            self._requested_mock(target, boundary)
            and (
                target.name,
                boundary,
            )
            in self._references
        )

    def _mock(
        self,
        target: MhcTrainRefTarget,
        boundary: str,
        runtime: torch.Tensor,
    ) -> torch.Tensor:
        return prepare_mhc_mock_input(
            self._reference(target, boundary),
            runtime,
            self.config,
            f"{target.name}_{boundary}",
        )

    def _dump(self, target: MhcTrainRefTarget, boundary: str, value: Any) -> None:
        dump_output(
            value,
            f"{target.name}_{boundary}",
            self.config,
            self.rank,
        )
        legacy_prefix = {
            "attn": "pangu_transformer_hc",
            "mlp": "pangu_transformer_mlp_hc",
        }.get(target.branch)
        if legacy_prefix is None:
            raise RuntimeError(f"unsupported MHC branch: {target.branch}")
        legacy_suffix = {
            "pre": "pre_out",
            "post": "post_out",
        }.get(boundary)
        if legacy_suffix is not None:
            dump_output(
                value,
                f"{legacy_prefix}_{legacy_suffix}_{target.layer_id + 1}",
                self.config,
                self.rank,
            )

    def _add_wrapper(
        self,
        instance: Any,
        method_name: str,
        factory: Callable[[Callable[..., Any]], Callable[..., Any]],
    ) -> None:
        original = getattr(instance, method_name)
        wrapper = factory(original)
        self._handles.append(MethodHandle(instance, method_name, wrapper))

    def _install_module_methods(self, target: MhcTrainRefTarget) -> None:
        def pre_factory(original):
            def wrapped(_module, hidden_states, *args, **kwargs):
                if not should_skip_runtime_hook() and self._should_mock(
                    target, "pre_in"
                ):
                    hidden_states = self._mock(target, "pre_in", hidden_states)
                output = original(hidden_states, *args, **kwargs)
                if not should_skip_runtime_hook():
                    self._dump(target, "pre", output)
                return output

            return wrapped

        def sinkhorn_factory(original):
            def wrapped(_module, h_res, *args, **kwargs):
                output = original(h_res, *args, **kwargs)
                if not should_skip_runtime_hook():
                    self._dump(target, "sinkhorn", output)
                return output

            return wrapped

        def fused_sinkhorn_factory(original):
            def wrapped(_module, h_post, h_res, *args, **kwargs):
                output = original(h_post, h_res, *args, **kwargs)
                if not should_skip_runtime_hook():
                    self._dump(target, "sinkhorn", output[1])
                return output

            return wrapped

        def post_factory(original):
            def wrapped(
                _module,
                hidden_states,
                h_post,
                residual,
                h_res,
                *args,
                **kwargs,
            ):
                if not should_skip_runtime_hook():
                    if self._should_mock(target, "post_in"):
                        hidden_states = self._mock(target, "post_in", hidden_states)
                    if self._should_mock(target, "post_h_post"):
                        h_post = self._mock(target, "post_h_post", h_post)
                    if self._should_mock(target, "post_residual"):
                        residual = self._mock(target, "post_residual", residual)
                    if self._should_mock(target, "post_h_res"):
                        h_res = self._mock(target, "post_h_res", h_res)
                output = original(
                    hidden_states,
                    h_post,
                    residual,
                    h_res,
                    *args,
                    **kwargs,
                )
                if not should_skip_runtime_hook():
                    self._dump(target, "post", output)
                return output

            return wrapped

        def sandwich_factory(original):
            def wrapped(
                _module,
                hidden_states,
                residual,
                h_post,
                h_res,
                *args,
                **kwargs,
            ):
                if not should_skip_runtime_hook():
                    if self._should_mock(target, "post_in"):
                        hidden_states = self._mock(target, "post_in", hidden_states)
                    if self._should_mock(target, "post_residual"):
                        residual = self._mock(target, "post_residual", residual)
                    if self._should_mock(target, "post_h_post"):
                        h_post = self._mock(target, "post_h_post", h_post)
                    if self._should_mock(target, "post_h_res"):
                        h_res = self._mock(target, "post_h_res", h_res)
                output = original(
                    hidden_states,
                    residual,
                    h_post,
                    h_res,
                    *args,
                    **kwargs,
                )
                if not should_skip_runtime_hook():
                    self._dump(target, "post_fused", output)
                return output

            return wrapped

        def launch_factory(original):
            def wrapped(_module, residual, *args, **kwargs):
                if not should_skip_runtime_hook() and self._should_mock(
                    target, "pre_in"
                ):
                    residual = self._mock(target, "pre_in", residual)
                return original(residual, *args, **kwargs)

            return wrapped

        self._add_wrapper(target.module, "mhc_pre", pre_factory)
        self._add_wrapper(target.module, "resolve_sinkhorn", sinkhorn_factory)
        self._add_wrapper(
            target.module,
            "resolve_fused_split_sinkhorn",
            fused_sinkhorn_factory,
        )
        self._add_wrapper(target.module, "mhc_post", post_factory)
        self._add_wrapper(
            target.module,
            "mhc_sandwich_norm_post_preonly",
            sandwich_factory,
        )
        self._add_wrapper(
            target.module,
            "launch_fused_split_sinkhorn",
            launch_factory,
        )

    def _install_fused_decoder_output(self, target: MhcTrainRefTarget) -> None:
        decoder_id = id(target.decoder_layer)
        if decoder_id in self._wrapped_decoder_ids:
            return
        self._wrapped_decoder_ids.add(decoder_id)

        def fused_decoder_factory(original):
            def wrapped(_layer, *args, **kwargs):
                output = original(*args, **kwargs)
                if not should_skip_runtime_hook():
                    dump_output(
                        output,
                        f"decoder_layer_{target.layer_id}",
                        self.config,
                        self.rank,
                    )
                return output

            return wrapped

        self._add_wrapper(
            target.decoder_layer,
            "forward_mhc_fused",
            fused_decoder_factory,
        )
