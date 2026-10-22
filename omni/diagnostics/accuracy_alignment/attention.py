# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""MLA/DSA attention train-reference hooks for Pangu UltraMoE."""

from __future__ import annotations

from typing import Any

import torch

from .config import TrainRefConfig
from .dump_io import (
    dump_output,
    load_layer_training_inputs,
    load_reference_tensor,
    prepare_layer_mock_input,
    prepare_mhc_mock_input,
    primary_tensor,
)
from .hooks import (
    MethodHandle,
    ModuleTrainRefHooks,
    ModuleTrainRefTarget,
    close_hook_groups,
    gather_tp_output,
    get_tp_rank,
)
from .train_ref import should_skip_runtime_hook


def _module(parent: Any, path: str) -> torch.nn.Module | None:
    value = parent
    for piece in path.split("."):
        value = getattr(value, piece, None)
        if value is None:
            return None
    return value if isinstance(value, torch.nn.Module) else None


class AttentionTrainRefHooks:
    def __init__(self, config: TrainRefConfig, rank: int) -> None:
        self.config = config
        self.rank = rank
        self.tp_rank = get_tp_rank()
        self._handles: list[Any] = []
        self._extensions: list[Any] = []
        self._qkv_reference = None
        self._qkv_parts: dict[str, torch.Tensor] = {}
        self._qb_reference = None
        self._qb_parts: dict[str, torch.Tensor] = {}
        self._mome_depth = 0
        self._mome_sp_pending: set[int] = set()
        self._mome_sp_out_depth = 0
        self._dsa_custom: dict[str, torch.Tensor] = {}

    @property
    def installed_hook_count(self) -> int:
        return len(self._handles) + sum(
            int(getattr(extension, "installed_hook_count", 0))
            for extension in self._extensions
        )

    def install(self, attention: torch.nn.Module) -> None:
        self.close()
        generic_targets = self._generic_targets(attention)
        generic = ModuleTrainRefHooks(self.config, self.rank)
        generic.install(generic_targets)
        self._extensions.append(generic)
        try:
            self._install_qkv(attention)
            self._install_qb(attention)
            self._install_mome(attention)
            self._install_dsa_indexer(attention)
        except Exception:
            self.close()
            raise

    def close(self) -> None:
        close_hook_groups(self._handles, self._extensions)
        self._qkv_reference = None
        self._qkv_parts.clear()
        self._qb_reference = None
        self._qb_parts.clear()
        self._mome_depth = 0
        self._mome_sp_pending.clear()
        self._mome_sp_out_depth = 0
        self._dsa_custom.clear()

    def _name(self, training_name: str) -> str:
        return f"decoder.layers.{self.config.layer_id}.{training_name}"

    def _generic_targets(
        self,
        attention: torch.nn.Module,
    ) -> list[ModuleTrainRefTarget]:
        candidates = [
            ("q_a_layernorm", "self_attention.q_layernorm", True, False),
            ("o_proj", "self_attention.linear_proj", True, True),
            ("qa_conv", "self_attention.qa_conv", False, False),
            (
                "compresskv_conv",
                "self_attention.compresskv_conv",
                False,
                False,
            ),
            (
                "indexer.wq_b",
                "self_attention.index_linear_qb",
                True,
                False,
            ),
            (
                "indexer.wk",
                "self_attention.index_linear_k",
                True,
                False,
            ),
            (
                "indexer.k_norm",
                "self_attention.index_k_layernorm",
                True,
                False,
            ),
            (
                "indexer.weights_proj",
                "self_attention.linear_merge_weight",
                True,
                False,
            ),
        ]
        if not bool(getattr(attention, "split_q_up_in_multistream", False)):
            candidates.insert(
                1,
                ("q_b_proj", "self_attention.linear_qb", True, False),
            )
        if self.config.role == "P":
            candidates.append(("kv_b_proj", "self_attention.linear_kvb", True, False))
        targets = []
        for path, training_name, inject, direct in candidates:
            module = _module(attention, path)
            if module is None:
                continue
            targets.append(
                ModuleTrainRefTarget(
                    module_name=self._name(training_name),
                    training_name=training_name,
                    module=module,
                    inject_input=inject,
                    direct_forward=direct,
                )
            )
        return targets

    def _install_qkv(self, attention: torch.nn.Module) -> None:
        q_proj = _module(attention, "q_a_proj")
        kv_proj = _module(attention, "kv_a_proj_with_mqa")
        if q_proj is None or kv_proj is None:
            return
        training_name = "self_attention.linear_qkv"
        should_mock = self.config.should_mock(training_name)
        if should_mock:
            self._qkv_reference = load_layer_training_inputs(self.config, training_name)

        def make_pre_hook():
            def pre_hook(_module, args, kwargs):
                if should_skip_runtime_hook() or not args or not should_mock:
                    return None
                value = args[0]
                if not isinstance(value, torch.Tensor):
                    return None
                mock = prepare_layer_mock_input(
                    self._qkv_reference,
                    value,
                    self.config,
                    self.tp_rank,
                )
                return ((mock, *args[1:]), kwargs)

            return pre_hook

        def make_post_hook(part: str):
            def post_hook(module, _args, _kwargs, output):
                if should_skip_runtime_hook():
                    return
                # q_a and kv_a may run on different NPU streams in 0.25.1.
                # Materialize each half before storing it so the later cat
                # cannot read a producer stream without an explicit event.
                tensor = gather_tp_output(module, primary_tensor(output)).detach().cpu()
                self._qkv_parts[part] = tensor
                if "q" in self._qkv_parts and "kv" in self._qkv_parts:
                    combined = torch.cat(
                        [self._qkv_parts.pop("q"), self._qkv_parts.pop("kv")],
                        dim=-1,
                    )
                    dump_output(
                        combined,
                        self._name(training_name),
                        self.config,
                        self.rank,
                    )

            return post_hook

        for module, part in ((q_proj, "q"), (kv_proj, "kv")):
            self._handles.append(
                module.register_forward_pre_hook(make_pre_hook(), with_kwargs=True)
            )
            self._handles.append(
                module.register_forward_hook(make_post_hook(part), with_kwargs=True)
            )

    def _install_qb(self, attention: torch.nn.Module) -> None:
        """Reconstruct the legacy linear_qb point from split q-up projections.

        vLLM 0.25.1 can release ``q_b_proj`` after loading and execute
        ``q_b_nope_proj`` and ``q_b_pe_proj`` on different streams.  The
        imported training dump still stores the unsplit per-head q_b output,
        so concatenate the two pieces inside each head instead of concatenating
        their flattened outputs.
        """

        if not bool(getattr(attention, "split_q_up_in_multistream", False)):
            return
        nope_proj = _module(attention, "q_b_nope_proj")
        pe_proj = _module(attention, "q_b_pe_proj")
        if nope_proj is None or pe_proj is None:
            raise RuntimeError(
                "split_q_up_in_multistream requires q_b_nope_proj and "
                "q_b_pe_proj for accuracy alignment"
            )

        training_name = "self_attention.linear_qb"
        should_mock = self.config.should_mock(training_name)
        if should_mock:
            self._qb_reference = load_layer_training_inputs(
                self.config,
                training_name,
            )

        def pre_hook(_module, args, kwargs):
            if should_skip_runtime_hook() or not args or not should_mock:
                return None
            value = args[0]
            if not isinstance(value, torch.Tensor):
                return None
            mock = prepare_layer_mock_input(
                self._qb_reference,
                value,
                self.config,
                self.tp_rank,
            )
            return ((mock, *args[1:]), kwargs)

        def post_hook(part: str):
            def hook(module, _args, _kwargs, output):
                if should_skip_runtime_hook():
                    return
                # The split projections can execute concurrently on main and
                # LI streams.  Debug capture intentionally synchronizes each
                # half before reconstructing the legacy q_b tensor.
                tensor = gather_tp_output(module, primary_tensor(output)).detach().cpu()
                self._qb_parts[part] = tensor
                nope = self._qb_parts.get("nope")
                pe = self._qb_parts.get("pe")
                if nope is None or pe is None:
                    return
                self._qb_parts.clear()

                nope_dim = int(getattr(attention, "qk_nope_head_dim", 0) or 0)
                rope_dim = int(getattr(attention, "qk_rope_head_dim", 0) or 0)
                if nope_dim <= 0 or rope_dim <= 0:
                    raise RuntimeError(
                        "split q-up accuracy alignment requires positive "
                        "qk_nope_head_dim and qk_rope_head_dim"
                    )
                if nope.shape[:-1] != pe.shape[:-1]:
                    raise RuntimeError(
                        "split q-up projection shapes are incompatible: "
                        f"nope={tuple(nope.shape)} pe={tuple(pe.shape)}"
                    )
                if nope.shape[-1] % nope_dim or pe.shape[-1] % rope_dim:
                    raise RuntimeError(
                        "split q-up output widths are not divisible by their "
                        "per-head dimensions"
                    )
                nope_heads = nope.shape[-1] // nope_dim
                pe_heads = pe.shape[-1] // rope_dim
                if nope_heads != pe_heads:
                    raise RuntimeError(
                        "split q-up projections produced different head counts: "
                        f"nope={nope_heads} pe={pe_heads}"
                    )
                combined = torch.cat(
                    [
                        nope.reshape(*nope.shape[:-1], nope_heads, nope_dim),
                        pe.reshape(*pe.shape[:-1], pe_heads, rope_dim),
                    ],
                    dim=-1,
                ).flatten(-2)
                dump_output(
                    combined,
                    self._name(training_name),
                    self.config,
                    self.rank,
                )

            return hook

        for module, part in ((nope_proj, "nope"), (pe_proj, "pe")):
            self._handles.append(
                module.register_forward_pre_hook(pre_hook, with_kwargs=True)
            )
            self._handles.append(
                module.register_forward_hook(post_hook(part), with_kwargs=True)
            )

    def _install_mome(self, attention: torch.nn.Module) -> None:
        dispatcher = _module(attention, "conv")
        if dispatcher is None:
            self._install_contiguous_o_conv(attention)
            return
        self._install_mome_sp_boundaries(attention)
        state_names = {
            0: self._name("self_attention.qa_conv"),
            1: self._name("self_attention.compresskv_conv"),
            2: f"pangu_transformer_o_conv_{self.config.layer_id + 1}",
        }
        current_state: list[int | None] = [None]

        def dispatcher_pre(_module, args, kwargs):
            self._mome_depth += 1
            state = kwargs.get("state_indice")
            if state is None and len(args) >= 2:
                state = args[1]
            current_state[0] = state if isinstance(state, int) else None

        def dispatcher_post(_module, _args, _kwargs, output):
            try:
                state = current_state[0]
                if not should_skip_runtime_hook() and state in state_names:
                    output_name = state_names.get(state)
                    if output_name is not None:
                        dump_output(
                            output,
                            output_name,
                            self.config,
                            self.rank,
                        )
            finally:
                current_state[0] = None
                self._mome_depth = max(self._mome_depth - 1, 0)

        self._handles.append(
            dispatcher.register_forward_pre_hook(dispatcher_pre, with_kwargs=True)
        )
        self._handles.append(
            dispatcher.register_forward_hook(dispatcher_post, with_kwargs=True)
        )

        child_names = {
            "qa_conv": self._name("self_attention.qa_conv"),
            "compresskv_conv": self._name("self_attention.compresskv_conv"),
            "o_conv": f"pangu_transformer_o_conv_{self.config.layer_id + 1}",
        }
        for child_attr, output_name in child_names.items():
            child = _module(dispatcher, child_attr)
            if child is None:
                continue
            for method_name in ("forward", "forward_prefill", "forward_decode"):
                if not hasattr(child, method_name):
                    continue
                original = getattr(child, method_name)

                def make_wrapper(original_method, name):
                    def wrapped(_child, *args, **kwargs):
                        output = original_method(*args, **kwargs)
                        if self._mome_depth == 0 and not should_skip_runtime_hook():
                            tensor = gather_tp_output(
                                _child,
                                primary_tensor(output),
                            )
                            dump_output(
                                tensor,
                                name,
                                self.config,
                                self.rank,
                            )
                        return output

                    return wrapped

                self._handles.append(
                    MethodHandle(
                        child,
                        method_name,
                        make_wrapper(original, output_name),
                    )
                )

    def _install_mome_sp_boundaries(self, attention: torch.nn.Module) -> None:
        """Restore legacy MoME dump points for the functional SP fast path.

        The 0.25.1 Ascend-950 SP path calls ``conv_sp`` directly, bypassing
        both ``MomeAttention.forward`` and its child modules. Observe the
        tensors at their next stable semantic consumers instead: q-layernorm,
        KV norm/cache, and o-projection. The wrappers are active only after a
        functional SP convolution reports that it actually ran.
        """

        apply_prefill_sp = getattr(attention, "_apply_mome_prefill_sp", None)
        if callable(apply_prefill_sp):

            def wrapped_prefill_sp(
                _attention,
                value,
                state_indice,
                *args,
                **kwargs,
            ):
                applied = apply_prefill_sp(value, state_indice, *args, **kwargs)
                if (
                    applied
                    and state_indice in (0, 1)
                    and not should_skip_runtime_hook()
                ):
                    self._mome_sp_pending.add(int(state_indice))
                return applied

            self._handles.append(
                MethodHandle(
                    attention,
                    "_apply_mome_prefill_sp",
                    wrapped_prefill_sp,
                )
            )

        q_layernorm = _module(attention, "q_a_layernorm")
        if q_layernorm is not None:

            def q_pre(_module, args, _kwargs):
                if (
                    0 not in self._mome_sp_pending
                    or should_skip_runtime_hook()
                    or not args
                ):
                    return
                self._mome_sp_pending.discard(0)
                dump_output(
                    args[0],
                    self._name("self_attention.qa_conv"),
                    self.config,
                    self.rank,
                )

            self._handles.append(
                q_layernorm.register_forward_pre_hook(q_pre, with_kwargs=True)
            )

        kv_norm_rope = getattr(attention, "_kv_norm_rope_cache", None)
        if callable(kv_norm_rope):

            def wrapped_kv_norm(_attention, latent_kv, *args, **kwargs):
                if 1 in self._mome_sp_pending and not should_skip_runtime_hook():
                    self._mome_sp_pending.discard(1)
                    kv_width = int(getattr(attention, "kv_lora_rank", 0) or 0)
                    if kv_width <= 0 or latent_kv.shape[-1] < kv_width:
                        raise RuntimeError(
                            "MoME SP accuracy alignment requires a valid kv_lora_rank"
                        )
                    dump_output(
                        latent_kv[..., :kv_width],
                        self._name("self_attention.compresskv_conv"),
                        self.config,
                        self.rank,
                    )
                return kv_norm_rope(latent_kv, *args, **kwargs)

            self._handles.append(
                MethodHandle(
                    attention,
                    "_kv_norm_rope_cache",
                    wrapped_kv_norm,
                )
            )

        apply_out_sp = getattr(attention, "_apply_mome_out_prefill_sp", None)
        apply_o_proj = getattr(attention, "_apply_o_proj", None)
        if callable(apply_out_sp) and callable(apply_o_proj):

            def wrapped_out_sp(_attention, *args, **kwargs):
                self._mome_sp_out_depth += 1
                try:
                    return apply_out_sp(*args, **kwargs)
                finally:
                    self._mome_sp_out_depth = max(
                        self._mome_sp_out_depth - 1,
                        0,
                    )

            def wrapped_o_proj(_attention, value, *args, **kwargs):
                if self._mome_sp_out_depth and not should_skip_runtime_hook():
                    dump_output(
                        value,
                        f"pangu_transformer_o_conv_{self.config.layer_id + 1}",
                        self.config,
                        self.rank,
                    )
                return apply_o_proj(value, *args, **kwargs)

            self._handles.append(
                MethodHandle(
                    attention,
                    "_apply_mome_out_prefill_sp",
                    wrapped_out_sp,
                )
            )
            self._handles.append(
                MethodHandle(attention, "_apply_o_proj", wrapped_o_proj)
            )

    def _install_contiguous_o_conv(self, attention: torch.nn.Module) -> None:
        o_conv = _module(attention, "o_conv")
        if o_conv is None:
            return

        def post_hook(module, _args, _kwargs, output):
            if should_skip_runtime_hook():
                return
            tensor = gather_tp_output(module, primary_tensor(output))
            dump_output(
                tensor,
                f"pangu_transformer_o_conv_{self.config.layer_id + 1}",
                self.config,
                self.rank,
            )

        self._handles.append(o_conv.register_forward_hook(post_hook, with_kwargs=True))

    def _install_dsa_indexer(self, attention: torch.nn.Module) -> None:
        indexer = getattr(attention, "indexer", None)
        if indexer is None or not hasattr(indexer, "_apply_lightning_indexer"):
            return
        lid1 = self.config.layer_id + 1
        raw_names = {
            "qi": f"dsa_attention_index_query_all_heads_{lid1}_rank0.pt",
            "wi": f"dsa_attention_merge_weight_all_heads_{lid1}_rank0.pt",
        }
        mock_qi = self.config.should_mock("dsa_attention_index_query_all_heads")
        mock_wi = self.config.should_mock("dsa_attention_merge_weight_all_heads")
        if self.config.mhc_custom_dump_dir is None:
            if self.config.mode == "real" and (mock_qi or mock_wi):
                raise RuntimeError(
                    "DSA indexer mock was selected but "
                    "OMNI_NPU_MHC_CUSTOM_DUMP_DIR is unset"
                )
            mock_qi = False
            mock_wi = False
        elif mock_qi or mock_wi:
            paths = {
                name: self.config.mhc_custom_dump_dir / filename
                for name, filename in raw_names.items()
            }
            existing = {name: path.is_file() for name, path in paths.items()}
            required_names = {
                name for name, enabled in (("qi", mock_qi), ("wi", mock_wi)) if enabled
            }
            any_required = any(existing.get(name, False) for name in required_names)
            all_required = all(existing.get(name, False) for name in required_names)
            if any_required and not all_required:
                missing = [
                    str(paths.get(name))
                    for name in required_names
                    if not existing.get(name, False)
                ]
                raise RuntimeError(
                    "incomplete DSA custom reference set; missing: "
                    + ", ".join(missing)
                )
            if not any_required:
                if self.config.mode == "real":
                    missing = [str(paths.get(name)) for name in required_names]
                    raise RuntimeError(
                        "selected DSA custom references are missing: "
                        + ", ".join(missing)
                    )
                mock_qi = False
                mock_wi = False
            self._dsa_custom = {
                name: load_reference_tensor(path)
                for name, path in paths.items()
                if name in required_names and existing.get(name, False)
            }
        original = indexer._apply_lightning_indexer

        def wrapped(_indexer, wi, qi, ki_cache, *args, **kwargs):
            wi_value = wi
            qi_value = qi
            if not should_skip_runtime_hook():
                wi_reference = self._dsa_custom.get("wi")
                qi_reference = self._dsa_custom.get("qi")
                if mock_wi:
                    if wi_reference is None:
                        raise RuntimeError("cached DSA WI reference is missing")
                    wi_value = prepare_mhc_mock_input(
                        wi_reference,
                        wi,
                        self.config,
                        "dsa_indexer_wi",
                    )
                if mock_qi:
                    if qi_reference is None:
                        raise RuntimeError("cached DSA QI reference is missing")
                    qi_value = prepare_mhc_mock_input(
                        qi_reference,
                        qi,
                        self.config,
                        "dsa_indexer_qi",
                    )
            output = original(wi_value, qi_value, ki_cache, *args, **kwargs)
            if not should_skip_runtime_hook():
                dump_output(
                    qi,
                    f"dsa_attention_index_query_all_heads_{lid1}",
                    self.config,
                    self.rank,
                )
                dump_output(
                    wi,
                    f"dsa_attention_merge_weight_all_heads_{lid1}",
                    self.config,
                    self.rank,
                )
                if output is not None:
                    dump_output(
                        output,
                        f"dsa_attention_topk_indices_{lid1}",
                        self.config,
                        self.rank,
                    )
            return output

        self._handles.append(MethodHandle(indexer, "_apply_lightning_indexer", wrapped))
