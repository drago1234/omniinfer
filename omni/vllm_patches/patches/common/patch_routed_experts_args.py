# SPDX-License-Identifier: MIT
# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd. All Rights Reserved.

"""Routed-experts serialization CLI and configuration support."""

import argparse

from vllm import EngineArgs
from vllm.config import VllmConfig
from vllm.utils.argparse_utils import FlexibleArgumentParser

from omni_npu.vllm_patches.core import VLLMPatch, register_patch

ROUTED_EXPERTS_SERIALIZATION_MODES = ("zip_base64", "base64")
DEFAULT_ROUTED_EXPERTS_SERIALIZATION_MODE = "zip_base64"


@register_patch("RoutedExpertsArgsVllmConfigPatch", VllmConfig)
class RoutedExpertsArgsVllmConfigPatch(VLLMPatch):
    """Expose the serialization mode on every ``VllmConfig`` instance."""

    _attr_names_to_apply = ["routed_experts_serialization_mode"]

    routed_experts_serialization_mode: str = (
        DEFAULT_ROUTED_EXPERTS_SERIALIZATION_MODE
    )


@register_patch("RoutedExpertsArgsEngineArgsPatch", EngineArgs)
class RoutedExpertsArgsEngineArgsPatch(VLLMPatch):
    """Accept the flag on the CLI and bridge it onto ``VllmConfig``."""

    _attr_names_to_apply = [
        "routed_experts_serialization_mode",
        "add_cli_args",
        "from_cli_args",
        "create_engine_config",
    ]

    routed_experts_serialization_mode: str = (
        DEFAULT_ROUTED_EXPERTS_SERIALIZATION_MODE
    )

    _upstream_add_cli_args = EngineArgs.add_cli_args
    _upstream_from_cli_args = EngineArgs.from_cli_args.__func__
    _upstream_create_engine_config = EngineArgs.create_engine_config

    @classmethod
    def apply(cls):
        # Retain the active wrappers installed earlier in the common patch chain.
        cls.apply_bypass_conflict(
            "add_cli_args", "from_cli_args", "create_engine_config"
        )

    @staticmethod
    def add_cli_args(parser: FlexibleArgumentParser) -> FlexibleArgumentParser:
        parser = RoutedExpertsArgsEngineArgsPatch._upstream_add_cli_args(parser)
        parser.add_argument(
            "--routed-experts-serialization-mode",
            choices=ROUTED_EXPERTS_SERIALIZATION_MODES,
            default=DEFAULT_ROUTED_EXPERTS_SERIALIZATION_MODE,
            help=(
                "Serialization mode for routed-experts payloads. "
                "'zip_base64' zlib-compresses before base64 to reduce PD relay "
                "payload size; 'base64' skips compression and requires larger "
                "proxy response buffers for long prompts."
            ),
        )
        return parser

    @classmethod
    def from_cli_args(cls, args: argparse.Namespace):
        instance = RoutedExpertsArgsEngineArgsPatch._upstream_from_cli_args(cls, args)
        instance.routed_experts_serialization_mode = getattr(
            args,
            "routed_experts_serialization_mode",
            DEFAULT_ROUTED_EXPERTS_SERIALIZATION_MODE,
        )
        return instance

    def create_engine_config(
        self,
        usage_context=None,
        headless: bool = False,
    ) -> VllmConfig:
        vllm_config = RoutedExpertsArgsEngineArgsPatch._upstream_create_engine_config(
            self, usage_context, headless
        )
        # A class attribute remains visible without polluting dataclass fields.
        type(vllm_config).routed_experts_serialization_mode = getattr(
            self,
            "routed_experts_serialization_mode",
            DEFAULT_ROUTED_EXPERTS_SERIALIZATION_MODE,
        )
        return vllm_config
