# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""Model-level embedding, rotary, and LM-head train-reference hooks."""

from __future__ import annotations

from typing import Any

import torch

from .config import TrainRefConfig
from .dump_io import dump_output, load_training_input, prepare_mock_input
from .hooks import MethodHandle
from .train_ref import should_skip_runtime_hook


class ModelLevelTrainRefHooks:
    def __init__(self, config: TrainRefConfig, rank: int) -> None:
        self.config = config
        self.rank = rank
        self._handles: list[Any] = []
        self._inputs: dict[str, torch.Tensor] = {}

    @property
    def installed_hook_count(self) -> int:
        return len(self._handles)

    def install(self, model: Any) -> None:
        self.close()
        pangu_model = model.model
        embed = getattr(pangu_model, "embed_tokens", None)
        lm_head = getattr(model, "lm_head", None)
        has_embed = embed is not None and type(embed).__name__ != "PPMissingLayer"
        has_lm_head = lm_head is not None and type(lm_head).__name__ != "PPMissingLayer"

        required: dict[str, Any] = {}
        if has_embed and self.config.should_mock(
            "embedding", "embedding.word_embeddings"
        ):
            required["embedding"] = self.config.training_dump_path("embedding")
        if self.config.should_mock("rotary_pos_emb"):
            required["rotary_pos_emb"] = self.config.model_training_dump_path(
                "rotary_pos_emb"
            )
        if has_lm_head and self.config.should_mock("lm_head"):
            required["lm_head"] = self.config.training_dump_path("lm_head")
        self._inputs = {
            name: load_training_input(path) for name, path in required.items()
        }

        try:
            if has_embed:
                self._install_embedding(embed, pangu_model)
            self._install_rotary(pangu_model)
            if has_lm_head:
                self._install_compute_logits(model)
        except Exception:
            self.close()
            raise

    def close(self) -> None:
        for handle in reversed(self._handles):
            handle.remove()
        self._handles.clear()
        self._inputs.clear()

    def _input(self, name: str) -> torch.Tensor:
        value = self._inputs.get(name)
        if value is None:
            raise RuntimeError(f"model-level training input is missing: {name}")
        return value

    def _install_embedding(self, embed: torch.nn.Module, pangu_model: Any) -> None:
        def pre_hook(_module, args, kwargs):
            if should_skip_runtime_hook() or not self.config.should_mock(
                "embedding", "embedding.word_embeddings"
            ):
                return None
            new_args = list(args)
            new_kwargs = dict(kwargs)
            for index, value in enumerate(new_args):
                if isinstance(value, torch.Tensor):
                    new_args[index] = prepare_mock_input(
                        self._input("embedding"),
                        value,
                        self.config,
                        "embedding",
                    )
                    return tuple(new_args), new_kwargs
            for name, value in new_kwargs.items():
                if isinstance(value, torch.Tensor):
                    new_kwargs[name] = prepare_mock_input(
                        self._input("embedding"),
                        value,
                        self.config,
                        "embedding",
                    )
                    return tuple(new_args), new_kwargs
            raise RuntimeError("embedding input does not contain a Tensor")

        self._handles.append(
            embed.register_forward_pre_hook(pre_hook, with_kwargs=True)
        )

        start_layer = int(getattr(pangu_model, "start_layer", 0))
        layers = getattr(pangu_model, "layers", None)
        if layers is None or start_layer >= len(layers):
            raise RuntimeError("cannot locate first local layer for embedding dump")
        layer = layers[start_layer]
        norm = getattr(layer, "input_layernorm", None)
        if norm is None or type(norm).__name__ == "PPMissingLayer":
            raise RuntimeError("cannot locate input_layernorm for embedding dump")

        def dump_embedding(_module, args, _kwargs):
            if should_skip_runtime_hook() or not args:
                return
            dump_output(
                args[0],
                "embedding.word_embeddings",
                self.config,
                self.rank,
            )

        self._handles.append(
            norm.register_forward_pre_hook(dump_embedding, with_kwargs=True)
        )

    def _install_rotary(self, pangu_model: Any) -> None:
        start_layer = int(getattr(pangu_model, "start_layer", 0))
        layer = pangu_model.layers[start_layer]
        rotary = layer.self_attn.rotary_emb
        original = rotary.get_cos_sin

        def wrapped(_rotary, positions, *args, **kwargs):
            if not should_skip_runtime_hook() and self.config.should_mock(
                "rotary_pos_emb"
            ):
                positions = prepare_mock_input(
                    self._input("rotary_pos_emb"),
                    positions,
                    self.config,
                    "rotary_pos_emb",
                )
            output = original(positions, *args, **kwargs)
            if not should_skip_runtime_hook():
                if not isinstance(output, tuple) or len(output) < 2:
                    raise RuntimeError("rotary get_cos_sin did not return cos/sin")
                dump_output(
                    output[0],
                    "rotary_pos_emb.cos",
                    self.config,
                    self.rank,
                )
                dump_output(
                    output[1],
                    "rotary_pos_emb.sin",
                    self.config,
                    self.rank,
                )
            return output

        self._handles.append(MethodHandle(rotary, "get_cos_sin", wrapped))

    def _install_compute_logits(self, model: Any) -> None:
        original = model.compute_logits

        def wrapped(_model, hidden_states, *args, **kwargs):
            if not should_skip_runtime_hook() and self.config.should_mock("lm_head"):
                hidden_states = prepare_mock_input(
                    self._input("lm_head"),
                    hidden_states,
                    self.config,
                    "lm_head",
                )
            logits = original(hidden_states, *args, **kwargs)
            if isinstance(logits, torch.Tensor) and not should_skip_runtime_hook():
                output = logits
                if (
                    self.config.train_tp_size > 1
                    and logits.shape[-1] % self.config.train_tp_size == 0
                ):
                    output = logits[
                        ..., : logits.shape[-1] // self.config.train_tp_size
                    ].contiguous()
                dump_output(
                    output,
                    "lm_head",
                    self.config,
                    self.rank,
                    enforce_token_count=False,
                )
            return logits

        self._handles.append(MethodHandle(model, "compute_logits", wrapped))
