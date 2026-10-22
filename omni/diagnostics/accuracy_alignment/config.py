# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""Environment-backed configuration for accuracy-alignment tools."""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, cast


class AlignmentConfigError(ValueError):
    """Raised when an enabled accuracy-alignment mode is misconfigured."""


def _first_value(env: Mapping[str, str], *names: str) -> str:
    for name in names:
        value = env.get(name, "").strip()
        if value:
            return value
    return ""


def _is_enabled(value: str) -> bool:
    return value.strip().lower() in {"1", "true", "yes", "on"}


AlignmentMode = Literal["train_ref", "real", "pure_real"]


def alignment_requested(env: Mapping[str, str] | None = None) -> bool:
    """Return whether the accuracy-alignment runtime is enabled at all."""

    values = os.environ if env is None else env
    return _is_enabled(values.get("OMNI_NPU_ACCURACY_ALIGNMENT_ENABLE", ""))


def requested_mode(env: Mapping[str, str] | None = None) -> AlignmentMode:
    """Resolve explicit 0.25.1 mode and the legacy three-state interface."""

    values = os.environ if env is None else env
    if not alignment_requested(values):
        raise AlignmentConfigError(
            "accuracy alignment requires "
            "OMNI_NPU_ACCURACY_ALIGNMENT_ENABLE=True"
        )
    raw_mode = values.get("OMNI_NPU_DEBUG_MODE", "").strip().lower()
    if raw_mode:
        if raw_mode not in {"train_ref", "real", "pure_real"}:
            raise AlignmentConfigError(
                "OMNI_NPU_DEBUG_MODE must be train_ref, real, or pure_real"
            )
        return cast(AlignmentMode, raw_mode)
    if _is_enabled(values.get("OMNI_NPU_DEBUG_PURE_REAL", "")):
        return "pure_real"
    # Legacy behavior: an unset whitelist means train_ref; a present value,
    # including an empty string, means a real-cascade pass.
    if "OMNI_NPU_DEBUG_MOCK_MODULES" in values:
        return "real"
    return "train_ref"


def train_ref_requested(env: Mapping[str, str] | None = None) -> bool:
    """Return whether the process explicitly selected ``train_ref`` mode."""

    values = os.environ if env is None else env
    if not alignment_requested(values):
        return False
    try:
        return requested_mode(values) == "train_ref"
    except AlignmentConfigError:
        return False


def real_requested(env: Mapping[str, str] | None = None) -> bool:
    values = os.environ if env is None else env
    if not alignment_requested(values):
        return False
    try:
        return requested_mode(values) == "real"
    except AlignmentConfigError:
        return False


def pure_real_requested(env: Mapping[str, str] | None = None) -> bool:
    values = os.environ if env is None else env
    if not alignment_requested(values):
        return False
    try:
        return requested_mode(values) == "pure_real"
    except AlignmentConfigError:
        return False


@dataclass(frozen=True)
class TrainRefConfig:
    """Validated configuration for the first train-ref implementation."""

    train_dump_dir: Path
    infer_dump_dir: Path
    layer_id: int
    role: str
    mock_shape: int
    train_name_prefix: str = "iter0_module.module"
    layer_train_name_prefix: str = "iter0_module.module.decoder.layers"
    embedding_train_name: str = "embedding.word_embeddings"
    decoder_train_name: str = "decoder.layers.{layer_id}"
    lm_head_train_name: str = "output_layer"
    mhc_custom_dump_dir: Path | None = None
    train_tp_size: int = 1
    train_ep_size: int = 4
    mode: Literal["train_ref", "real"] = "train_ref"
    mock_modules: tuple[str, ...] | None = None
    layer_ids: tuple[int, ...] = ()
    all_layers: bool = False

    @property
    def stage(self) -> str:
        return "prefill" if self.role == "P" else "decode"

    @property
    def dump_tag(self) -> str:
        return self.mode

    def should_mock(self, *module_names: str) -> bool:
        """Return whether this pass should inject training data at a boundary."""

        if self.mode == "train_ref" and self.mock_modules is None:
            return True
        patterns = self.mock_modules or ()
        for name in module_names:
            normalized = name.strip(".")
            for pattern in patterns:
                if pattern.endswith("*"):
                    prefix = pattern[:-1]
                    if normalized.startswith(prefix):
                        return True
                elif normalized == pattern:
                    return True
        return False

    def requested_layer_ids(self, total_layers: int) -> tuple[int, ...]:
        if self.all_layers:
            return tuple(range(total_layers))
        return self.layer_ids or (self.layer_id,)

    def training_dump_path(self, target_kind: str) -> Path:
        names = {
            "embedding": self.embedding_train_name,
            "decoder_layer": self.decoder_train_name.format(layer_id=self.layer_id),
            "lm_head": self.lm_head_train_name,
        }
        try:
            module_name = names[target_kind]
        except KeyError as exc:
            raise AlignmentConfigError(
                f"unsupported train-ref target kind: {target_kind}"
            ) from exc
        filename = f"{self.train_name_prefix}.{module_name}_forward_rank0_0.pt"
        return self.train_dump_dir / filename

    def model_training_dump_path(self, module_name: str, rank: int = 0) -> Path:
        filename = f"{self.train_name_prefix}.{module_name}_forward_rank{rank}_0.pt"
        return self.train_dump_dir / filename

    def layer_training_dump_path(self, module_name: str, rank: int = 0) -> Path:
        filename = (
            f"{self.layer_train_name_prefix}.{self.layer_id}.{module_name}"
            f"_forward_rank{rank}_0.pt"
        )
        return self.train_dump_dir / filename

    def mhc_training_dump_path(
        self,
        branch: str,
        boundary: str,
    ) -> Path:
        """Return one legacy raw-tensor MHC reference path."""

        if self.mhc_custom_dump_dir is None:
            raise AlignmentConfigError(
                "MHC train_ref requires OMNI_NPU_MHC_CUSTOM_DUMP_DIR"
            )
        prefixes = {
            "attn": "pangu_transformer_hc",
            "mlp": "pangu_transformer_mlp_hc",
        }
        prefix = prefixes.get(branch)
        if prefix is None:
            raise AlignmentConfigError(f"unsupported MHC branch: {branch}")
        supported_boundaries = {
            "pre_in",
            "post_in",
            "post_h_post",
            "post_residual",
            "post_h_res",
        }
        if boundary not in supported_boundaries:
            raise AlignmentConfigError(
                f"unsupported MHC train-ref boundary: {boundary}"
            )
        filename = f"{prefix}_{boundary}_{self.layer_id + 1}_rank0.pt"
        return self.mhc_custom_dump_dir / filename

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> TrainRefConfig:
        values = os.environ if env is None else env
        mode = requested_mode(values)
        if mode not in {"train_ref", "real"}:
            raise AlignmentConfigError(
                "TrainRefConfig supports train_ref and real modes only"
            )

        train_dir = _first_value(
            values,
            "OMNI_NPU_DEBUG_TRAIN_DUMP_DIR",
            "OMNI_NPU_MHC_DEBUG_TRAIN_DUMP_DIR",
        )
        infer_dir = _first_value(
            values,
            "OMNI_NPU_DEBUG_INFER_DUMP_DIR",
            "OMNI_NPU_MHC_DEBUG_INFER_DUMP_DIR",
        )
        mock_shape_raw = _first_value(
            values,
            "OMNI_NPU_DEBUG_MOCK_SHAPE",
            "OMNI_NPU_MHC_DEBUG_MOCK_SHAPE",
        )
        role = values.get("OMNI_NPU_MHC_DEBUG_ROLE", "").strip().upper()

        missing = []
        if not train_dir:
            missing.append("OMNI_NPU_DEBUG_TRAIN_DUMP_DIR")
        if not infer_dir:
            missing.append("OMNI_NPU_DEBUG_INFER_DUMP_DIR")
        if not mock_shape_raw:
            missing.append("OMNI_NPU_DEBUG_MOCK_SHAPE")
        if role not in {"P", "D"}:
            missing.append("OMNI_NPU_MHC_DEBUG_ROLE=P|D")
        if missing:
            raise AlignmentConfigError(
                f"incomplete {mode} configuration: " + ", ".join(missing)
            )

        try:
            layer_id = int(values.get("OMNI_NPU_DEBUG_LAYER_ID", "0"))
            mock_shape = int(mock_shape_raw)
            train_tp_size = int(values.get("OMNI_NPU_TRAIN_TP_SIZE", "1"))
            train_ep_size = int(values.get("OMNI_NPU_TRAIN_EP_SIZE", "4"))
        except ValueError as exc:
            raise AlignmentConfigError(
                "layer id, mock shape, train TP size, and train EP size must "
                "be integers"
            ) from exc
        if layer_id < 0:
            raise AlignmentConfigError("OMNI_NPU_DEBUG_LAYER_ID must be >= 0")
        if mock_shape <= 0:
            raise AlignmentConfigError("OMNI_NPU_DEBUG_MOCK_SHAPE must be > 0")
        if train_tp_size <= 0:
            raise AlignmentConfigError("OMNI_NPU_TRAIN_TP_SIZE must be > 0")
        if train_ep_size <= 0:
            raise AlignmentConfigError("OMNI_NPU_TRAIN_EP_SIZE must be > 0")

        layer_ids_raw = values.get("OMNI_NPU_DEBUG_LAYER_IDS", "").strip()
        all_layers = layer_ids_raw.lower() == "all"
        layer_ids: tuple[int, ...] = ()
        if layer_ids_raw and not all_layers:
            parsed_ids: set[int] = set()
            try:
                for item in layer_ids_raw.split(","):
                    token = item.strip()
                    if not token:
                        continue
                    if "-" in token:
                        start_raw, end_raw = token.split("-", 1)
                        start, end = int(start_raw), int(end_raw)
                        if start > end:
                            raise ValueError("descending layer range")
                        parsed_ids.update(range(start, end + 1))
                    else:
                        parsed_ids.add(int(token))
            except ValueError as exc:
                raise AlignmentConfigError(
                    "OMNI_NPU_DEBUG_LAYER_IDS must be 'all' or a comma-separated "
                    "list of non-descending ids/ranges (for example 0,3-5)"
                ) from exc
            if any(value < 0 for value in parsed_ids):
                raise AlignmentConfigError(
                    "OMNI_NPU_DEBUG_LAYER_IDS values must be >= 0"
                )
            layer_ids = tuple(sorted(parsed_ids))

        train_path = Path(train_dir).expanduser()
        if not train_path.is_dir():
            raise AlignmentConfigError(
                f"training dump directory does not exist: {train_path}"
            )
        infer_path = Path(infer_dir).expanduser()
        infer_path.mkdir(parents=True, exist_ok=True)
        mhc_custom_dir_raw = values.get("OMNI_NPU_MHC_CUSTOM_DUMP_DIR", "").strip()
        mhc_custom_path = None
        if mhc_custom_dir_raw:
            mhc_custom_path = Path(mhc_custom_dir_raw).expanduser()
            if not mhc_custom_path.is_dir():
                raise AlignmentConfigError(
                    f"MHC custom dump directory does not exist: {mhc_custom_path}"
                )

        raw_mock_modules = values.get("OMNI_NPU_DEBUG_MOCK_MODULES")
        mock_modules = None
        if mode == "real" or raw_mock_modules is not None:
            mock_modules = tuple(
                item.strip()
                for item in (raw_mock_modules or "").split(",")
                if item.strip()
            )

        return cls(
            train_dump_dir=train_path,
            infer_dump_dir=infer_path,
            layer_id=layer_id,
            role=role,
            mock_shape=mock_shape,
            train_name_prefix=values.get(
                "OMNI_NPU_MODEL_LEVEL_TRAIN_PREFIX", "iter0_module.module"
            ).strip()
            or "iter0_module.module",
            layer_train_name_prefix=values.get(
                "OMNI_NPU_MHC_DEBUG_TRAIN_NAME_PREFIX",
                "iter0_module.module.decoder.layers",
            ).strip()
            or "iter0_module.module.decoder.layers",
            embedding_train_name=values.get(
                "OMNI_NPU_EMBEDDING_TRAIN_NAME", "embedding.word_embeddings"
            ).strip()
            or "embedding.word_embeddings",
            decoder_train_name=values.get(
                "OMNI_NPU_DECODER_LAYER_TRAIN_NAME",
                "decoder.layers.{layer_id}",
            ).strip()
            or "decoder.layers.{layer_id}",
            lm_head_train_name=values.get(
                "OMNI_NPU_LM_HEAD_TRAIN_NAME", "output_layer"
            ).strip()
            or "output_layer",
            mhc_custom_dump_dir=mhc_custom_path,
            train_tp_size=train_tp_size,
            train_ep_size=train_ep_size,
            mode=mode,
            mock_modules=mock_modules,
            layer_ids=layer_ids,
            all_layers=all_layers,
        )


@dataclass(frozen=True)
class PureRealConfig:
    """Configuration for dump-only, run-to-run determinism captures."""

    infer_dump_dir: Path
    role: str
    run_id: str = "0"
    include_patterns: tuple[str, ...] = ()
    exclude_patterns: tuple[str, ...] = ()

    @property
    def stage(self) -> str:
        return "prefill" if self.role == "P" else "decode"

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> PureRealConfig:
        values = os.environ if env is None else env
        if requested_mode(values) != "pure_real":
            raise AlignmentConfigError(
                "pure_real requires "
                "OMNI_NPU_ACCURACY_ALIGNMENT_ENABLE=True and "
                "OMNI_NPU_DEBUG_MODE=pure_real (or legacy "
                "OMNI_NPU_DEBUG_PURE_REAL=1)"
            )
        infer_dir = _first_value(
            values,
            "OMNI_NPU_DEBUG_INFER_DUMP_DIR",
            "OMNI_NPU_MHC_DEBUG_INFER_DUMP_DIR",
        )
        role = values.get("OMNI_NPU_MHC_DEBUG_ROLE", "").strip().upper()
        missing = []
        if not infer_dir:
            missing.append("OMNI_NPU_DEBUG_INFER_DUMP_DIR")
        if role not in {"P", "D"}:
            missing.append("OMNI_NPU_MHC_DEBUG_ROLE=P|D")
        if missing:
            raise AlignmentConfigError(
                "incomplete pure_real configuration: " + ", ".join(missing)
            )
        infer_path = Path(infer_dir).expanduser()
        infer_path.mkdir(parents=True, exist_ok=True)

        def patterns(name: str) -> tuple[str, ...]:
            return tuple(
                item.strip() for item in values.get(name, "").split(",") if item.strip()
            )

        return cls(
            infer_dump_dir=infer_path,
            role=role,
            run_id=values.get("OMNI_NPU_DEBUG_RUN_ID", "0").strip() or "0",
            include_patterns=patterns("OMNI_NPU_PURE_REAL_DUMP_INCLUDE"),
            exclude_patterns=patterns("OMNI_NPU_PURE_REAL_DUMP_EXCLUDE"),
        )
