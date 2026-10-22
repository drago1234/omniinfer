# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""Install train-ref observers after NPUModelRunner has loaded its model."""

from __future__ import annotations

import logging
from dataclasses import replace
from typing import Any

from .attention import AttentionTrainRefHooks
from .config import AlignmentConfigError, TrainRefConfig, requested_mode
from .hooks import ModuleTrainRefHooks, close_runner_session
from .layer import collect_layer_targets
from .mhc import MhcTrainRefHooks, MhcTrainRefTarget
from .model_level import ModelLevelTrainRefHooks
from .moe import MoeTrainRefHooks
from .train_ref import TrainRefSession, TrainRefTarget

logger = logging.getLogger(__name__)

_SESSION_ATTR = "_omni_accuracy_alignment_train_ref_session"


class _AttributeOverride:
    """Temporarily override one model-instance attribute for a session."""

    installed_hook_count = 0

    def __init__(self, instance: Any, name: str, value: Any) -> None:
        self._instance = instance
        self._name = name
        self._had_value = hasattr(instance, name)
        self._value = getattr(instance, name, None)
        self._active = True
        setattr(instance, name, value)

    def close(self) -> None:
        if not self._active:
            return
        if self._had_value:
            setattr(self._instance, self._name, self._value)
        else:
            delattr(self._instance, self._name)
        self._active = False


def _rank() -> int:
    try:
        from vllm.distributed import get_world_group

        return int(get_world_group().rank_in_group)
    except (ImportError, RuntimeError, AttributeError, AssertionError):
        return 0


def _ultra_model_classes():
    from omni_npu.v1.models.pangu.pangu_ultra_moe import (
        OpenPanguModel,
        OpenPanguModelBase,
    )

    return OpenPanguModelBase, OpenPanguModel


def _is_real_module(module: Any) -> bool:
    return module is not None and type(module).__name__ != "PPMissingLayer"


def collect_train_ref_targets(
    model: Any,
    config: TrainRefConfig,
) -> list[TrainRefTarget]:
    pangu_model = model.model
    targets: list[TrainRefTarget] = []

    if pangu_model.start_layer <= config.layer_id < pangu_model.end_layer:
        layer = pangu_model.layers[config.layer_id]
        if _is_real_module(layer):
            targets.append(
                TrainRefTarget(
                    module_name=f"decoder_layer_{config.layer_id}",
                    target_kind="decoder_layer",
                    module=layer,
                    training_dump_path=config.training_dump_path("decoder_layer"),
                )
            )

    return targets


def collect_mhc_train_ref_targets(
    model: Any,
    config: TrainRefConfig,
) -> list[MhcTrainRefTarget]:
    """Collect both MHC branches only for the selected locally owned layer."""

    pangu_model = model.model
    if not (pangu_model.start_layer <= config.layer_id < pangu_model.end_layer):
        return []
    layer = pangu_model.layers[config.layer_id]
    if not _is_real_module(layer) or not bool(getattr(layer, "use_mhc", False)):
        return []
    targets = []
    for branch in ("attn", "mlp"):
        targets.append(
            MhcTrainRefTarget(
                layer_id=config.layer_id,
                branch=branch,
                module=getattr(layer, f"{branch}_mhc_module"),
                decoder_layer=layer,
            )
        )
    return targets


def install_train_ref(
    runner: Any,
    config: TrainRefConfig | None = None,
) -> TrainRefSession | None:
    """Install hooks on the local PP shard, replacing an older session."""

    close_runner_session(runner, _SESSION_ATTR)

    model = runner.get_model()
    base_cls, model_cls = _ultra_model_classes()
    if not isinstance(model, base_cls) or not isinstance(
        getattr(model, "model", None), model_cls
    ):
        logger.warning(
            "accuracy-alignment train_ref supports PanguUltraMoE only; got %s",
            type(model).__name__,
        )
        return None

    config = TrainRefConfig.from_env() if config is None else config
    if not bool(getattr(runner.model_config, "enforce_eager", False)):
        raise AlignmentConfigError(
            f"{config.mode} requires model_config.enforce_eager=True; graph replay "
            "does not execute Python module hooks reliably"
        )

    total_layers = len(model.model.layers)
    requested_layers = config.requested_layer_ids(total_layers)
    invalid_layers = [
        layer_id
        for layer_id in requested_layers
        if layer_id < 0 or layer_id >= total_layers
    ]
    if invalid_layers:
        raise AlignmentConfigError(
            f"requested layers outside [0, {total_layers}): {invalid_layers}"
        )

    session = TrainRefSession(config, rank=_rank())
    installed_names: list[str] = []
    try:
        session.install([])
        if config.mode == "train_ref" and bool(
            getattr(model.model, "use_mhc_fusion_op", False)
        ):
            # The fused path combines current mhc_post, next mhc_pre, and
            # normalization into one custom op.  It therefore has no separate
            # runtime boundaries matching the imported pre_out/post_out files.
            # Use the model's own non-fused forward for train_ref and restore
            # the production setting when this debug session closes.
            session.attach_extension(
                _AttributeOverride(model.model, "use_mhc_fusion_op", False)
            )
            logger.info(
                "disabled model-level MHC fusion for train_ref legacy "
                "boundary alignment"
            )
        model_hooks = ModelLevelTrainRefHooks(config, rank=session.rank)
        model_hooks.install(model)
        session.attach_extension(model_hooks)
        for layer_id in requested_layers:
            if not (model.model.start_layer <= layer_id < model.model.end_layer):
                continue
            layer_config = replace(config, layer_id=layer_id)
            selected_layer = model.model.layers[layer_id]
            attention = selected_layer.self_attn
            if bool(getattr(attention, "use_mlaprolog", False)):
                if config.mode == "train_ref":
                    # npu_mla_prolog_v3 fuses q_a/q_b/kv_a and their norms,
                    # so none of the imported Megatron sub-module boundaries
                    # execute. Use the model's native modular decode branch for
                    # train_ref and restore the production flag on close.
                    session.attach_extension(
                        _AttributeOverride(attention, "use_mlaprolog", False)
                    )
                    logger.info(
                        "disabled fused MLA prolog for train_ref layer=%d",
                        layer_id,
                    )
                else:
                    logger.warning(
                        "real mode keeps fused MLA prolog for layer=%d; "
                        "linear_qkv/linear_qb internals are not observable "
                        "on that decode path",
                        layer_id,
                    )
            mhc_targets = collect_mhc_train_ref_targets(model, layer_config)

            targets = collect_train_ref_targets(model, layer_config)
            decoder_hooks = TrainRefSession(layer_config, rank=session.rank)
            decoder_hooks.install(targets)
            session.attach_extension(decoder_hooks)
            installed_names.extend(target.module_name for target in targets)

            layer_targets = collect_layer_targets(model, layer_config)
            if layer_targets:
                layer_hooks = ModuleTrainRefHooks(layer_config, rank=session.rank)
                layer_hooks.install(layer_targets)
                session.attach_extension(layer_hooks)
                installed_names.extend(target.module_name for target in layer_targets)

            attention_hooks = AttentionTrainRefHooks(layer_config, rank=session.rank)
            attention_hooks.install(attention)
            session.attach_extension(attention_hooks)
            moe_hooks = MoeTrainRefHooks(layer_config, rank=session.rank)
            moe_hooks.install(selected_layer.mlp)
            session.attach_extension(moe_hooks)

            if mhc_targets:
                mhc_hooks = MhcTrainRefHooks(layer_config, rank=session.rank)
                mhc_hooks.install(mhc_targets)
                session.attach_extension(mhc_hooks)
                installed_names.extend(target.name for target in mhc_targets)
    except Exception:
        session.close()
        raise
    setattr(runner, _SESSION_ATTR, session)
    logger.info(
        "installed %s hooks for rank=%d targets=%s",
        config.mode,
        session.rank,
        installed_names,
    )
    return session


def install_accuracy_alignment(runner: Any):
    """Dispatch the enabled mode after ``NPUModelRunner.load_model``."""

    for attr_name in (
        _SESSION_ATTR,
        "_omni_accuracy_alignment_pure_real_session",
    ):
        close_runner_session(runner, attr_name)
    mode = requested_mode()
    if mode == "pure_real":
        from .pure_real import install_pure_real

        return install_pure_real(runner)
    return install_train_ref(runner)
