# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.

"""UTs for the OmniInfer-side ``maybe_transfer_kv_layer`` op-level wiring.

Pangu / OpenPangu custom attention entry points are wired into the vLLM 0.25.1
layer-wise KV-transfer lifecycle by decorating the **custom ops themselves**
(``npu_mla_forward`` / ``npu_dsa_forward`` / ``npu_pangu_forward``), mirroring
upstream ``unified_mla_attention_with_output``. The decorators are therefore
registered with the decorated function, and ``forward`` reaches its op directly
(no intermediate ``_run_*`` seam).

What these tests pin down:

* Each op carries ``@eager_break_during_capture`` + ``@maybe_transfer_kv_layer``
  and performs ``wait_for_layer_load -> kernel -> save_kv_layer``.
* The decorator anchor is the **inner** KV attention layer name
  (``...self_attn.attn``), while the outer module instance is resolved through a
  second, distinct ``module_name`` parameter (doc 6.2). The KV name must stay
  the *last* positional parameter: ``maybe_transfer_kv_layer`` reads it by slot.
* The MHC entry points are deliberately **not** decorated -- they forward to the
  decorated inner op, which fires the lifecycle exactly once.
* The old per-class shim (``_forward_with_kv_transfer``) is gone; the classes
  must not reintroduce it, and ``forward`` must not gate on the transfer group
  itself (that gate belongs to the decorator).
* ``npu_pangu_forward`` resolves the per-layer metadata dict with the inner
  names (``<kv_layer_name>`` and ``<kv_layer_name>.mome``).

All tests run on the host side only (CPU / mocked kernels): the decorator
wrapper, the routing logic and the metadata key resolution are pure Python.
"""

import inspect
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch

import vllm.model_executor.layers.attention.attention as vllm_attn
import vllm.model_executor.layers.attention.kv_transfer_utils as kv_transfer_utils

MODULE_NAME = "model.layers.2.self_attn"
KV_LAYER_NAME = "model.layers.2.self_attn.attn"
OTHER_KV_LAYER_NAME = "model.cla_swa_layers.40.attn"

_SHIM_NAMES = (
    "_forward_with_kv_transfer",
    "_forward_mhc_deferred_with_kv_transfer",
)
_MHC_OPS = (
    ("omni_npu.v1.layers.attention.npu_mla", "npu_mla_forward_mhc_deferred"),
    ("omni_npu.v1.layers.attention.npu_dsa", "npu_dsa_forward_mhc_deferred"),
)


# =========================
# Shared helpers
# =========================

def _enable_v1_group(monkeypatch, connector):
    """Turn on the v1 KV transfer group for the vLLM decorator wrapper."""
    monkeypatch.setattr(kv_transfer_utils, "has_kv_transfer_group", lambda: True)
    monkeypatch.setattr(kv_transfer_utils, "is_v1_kv_transfer_group", lambda: True)
    monkeypatch.setattr(kv_transfer_utils, "get_kv_transfer_group", lambda: connector)


def _recording_connector(events):
    """Connector stub that records the wait/run/save ordering."""
    connector = MagicMock()
    connector.has_connector_metadata.return_value = True
    connector.wait_for_layer_load.side_effect = (
        lambda name: events.append(("wait", name))
    )
    connector.save_kv_layer.side_effect = (
        lambda name, kv, meta: events.append(("save", name, kv, meta))
    )
    return connector


def _assert_lifecycle(events, kv_cache, attn_metadata):
    """The wrapper must bracket the kernel: wait -> run -> save."""
    assert [e[0] for e in events] == ["wait", "run", "save"], events
    assert events[0][1] == KV_LAYER_NAME
    assert events[2][1] == KV_LAYER_NAME
    assert events[2][2] is kv_cache
    assert events[2][3] is attn_metadata


def _forward_context(module, kv_cache, attn_metadata):
    """Both names must resolve: the outer module and the inner KV layer."""
    return SimpleNamespace(
        no_compile_layers={
            MODULE_NAME: module,
            KV_LAYER_NAME: SimpleNamespace(kv_cache=kv_cache),
        },
        attn_metadata=attn_metadata,
        slot_mapping={KV_LAYER_NAME: object()},
    )


def _raise_if_called(*args, **kwargs):
    raise AssertionError("get_attention_context ran without a v1 KV group")


# =========================
# 1. NPUDeepseekMLAAttention / npu_mla_forward
# =========================

@pytest.mark.unit
class TestMLAKVTransferLifecycle:

    def test_op_runs_wait_kernel_save_on_the_kv_layer_name(self, monkeypatch):
        import omni_npu.v1.layers.attention.npu_mla as mla_mod
        from omni_npu.v1.layers.attention.npu_mla import npu_mla_forward

        events = []
        _enable_v1_group(monkeypatch, _recording_connector(events))

        attn_metadata = SimpleNamespace(prefill=None, decode=SimpleNamespace())
        kv_cache = object()

        def fake_decode(x, cos, sin, decode):
            events.append(("run",))
            return torch.zeros_like(x)

        module = SimpleNamespace(
            param_sink_number=0,
            noncontiguous_kv=False,
            ena_sp=False,
            o_proj=SimpleNamespace(tp_size=1),
            _forward_decode=fake_decode,
        )
        ctx = _forward_context(module, kv_cache, attn_metadata)
        monkeypatch.setattr(mla_mod, "get_forward_context", lambda: ctx)
        monkeypatch.setattr(vllm_attn, "get_forward_context", lambda: ctx)
        monkeypatch.setattr(
            mla_mod,
            "get_batch_desc",
            lambda meta: (slice(0, 1), slice(0, 1), False, True),
        )

        hs = torch.zeros(1, 8)
        cos = torch.zeros(1, 1, 1, 4)
        out = npu_mla_forward(hs, cos, torch.zeros_like(cos), MODULE_NAME, KV_LAYER_NAME)

        _assert_lifecycle(events, kv_cache, attn_metadata)
        assert out.shape == hs.shape

    def test_forward_calls_the_op_with_module_and_kv_layer_names(self, monkeypatch):
        """``forward`` reaches the op directly, with the KV anchor last."""
        from omni_npu.v1.layers.attention.npu_mla import NPUDeepseekMLAAttention

        op = MagicMock(return_value=torch.zeros(1, 8))
        monkeypatch.setattr(torch.ops.vllm, "npu_mla_forward", op, raising=False)

        fake = NPUDeepseekMLAAttention.__new__(NPUDeepseekMLAAttention)
        fake.prefix = MODULE_NAME
        fake.attn = SimpleNamespace(layer_name=KV_LAYER_NAME)

        out = fake.forward("x", "cos", "sin", "topk")

        op.assert_called_once_with("x", "cos", "sin", MODULE_NAME, KV_LAYER_NAME)
        assert out == (op.return_value, "topk")

    def test_forward_mhc_deferred_calls_the_op_with_both_names(self, monkeypatch):
        from omni_npu.v1.layers.attention.npu_mla import NPUDeepseekMLAAttention

        op = MagicMock(return_value=("h", "post", "res"))
        monkeypatch.setattr(
            torch.ops.vllm, "npu_mla_forward_mhc_deferred", op, raising=False
        )

        fake = NPUDeepseekMLAAttention.__new__(NPUDeepseekMLAAttention)
        fake.prefix = MODULE_NAME
        fake.attn = SimpleNamespace(layer_name=KV_LAYER_NAME)

        out = fake.forward_mhc_deferred("x", "cos", "sin", "res", "mhc", "task", "topk")

        op.assert_called_once_with(
            "x", "cos", "sin", "res", MODULE_NAME, "mhc", "task", KV_LAYER_NAME
        )
        assert out == (("h", "post", "res"), "topk")


# =========================
# 2. NPUDeepseekSparseAttention / npu_dsa_forward
# =========================

@pytest.mark.unit
class TestDSAKVTransferLifecycle:

    def test_op_runs_wait_kernel_save_on_the_kv_layer_name(self, monkeypatch):
        import omni_npu.v1.layers.attention.npu_dsa as dsa_mod
        from omni_npu.v1.layers.attention.npu_dsa import npu_dsa_forward

        events = []
        _enable_v1_group(monkeypatch, _recording_connector(events))

        attn_metadata = SimpleNamespace(prefill=None, decode=SimpleNamespace())
        kv_cache = object()

        def fake_decode(x, cos, sin, decode, cache, topk, pd_mixed_flag):
            events.append(("run",))
            return torch.zeros_like(x), topk

        module = SimpleNamespace(
            param_sink_number=0,
            noncontiguous_kv=False,
            ena_sp=False,
            layer_idx=0,
            o_proj=SimpleNamespace(tp_size=1),
            _forward_decode=fake_decode,
            attn=SimpleNamespace(kv_cache=kv_cache),
        )
        ctx = _forward_context(module, kv_cache, attn_metadata)
        monkeypatch.setattr(dsa_mod, "get_forward_context", lambda: ctx)
        monkeypatch.setattr(vllm_attn, "get_forward_context", lambda: ctx)
        monkeypatch.setattr(
            dsa_mod,
            "get_batch_desc",
            lambda meta, idx: (slice(0, 1), slice(0, 1), False, True),
        )

        hs = torch.zeros(1, 8)
        cos = torch.zeros(1, 1, 1, 4)
        topk = torch.zeros(1, 1, 2, dtype=torch.int32)
        out, out_topk = npu_dsa_forward(
            hs, cos, torch.zeros_like(cos), MODULE_NAME, topk, KV_LAYER_NAME
        )

        _assert_lifecycle(events, kv_cache, attn_metadata)
        assert out.shape == hs.shape
        assert out_topk.shape == topk.shape

    def test_forward_calls_the_op_with_module_and_kv_layer_names(self, monkeypatch):
        """``forward`` reaches the op directly, with the KV anchor last."""
        from omni_npu.v1.layers.attention.npu_dsa import NPUDeepseekSparseAttention

        op = MagicMock(return_value=("out", "topk"))
        monkeypatch.setattr(torch.ops.vllm, "npu_dsa_forward", op, raising=False)

        fake = NPUDeepseekSparseAttention.__new__(NPUDeepseekSparseAttention)
        fake.prefix = MODULE_NAME
        fake.attn = SimpleNamespace(layer_name=KV_LAYER_NAME)

        out = fake.forward("x", "cos", "sin", "topk")

        op.assert_called_once_with(
            "x", "cos", "sin", MODULE_NAME, "topk", KV_LAYER_NAME
        )
        assert out == ("out", "topk")

    def test_forward_mhc_deferred_calls_the_op_with_both_names(self, monkeypatch):
        from omni_npu.v1.layers.attention.npu_dsa import NPUDeepseekSparseAttention

        op = MagicMock(return_value=("h", "post", "res"))
        monkeypatch.setattr(
            torch.ops.vllm, "npu_dsa_forward_mhc_deferred", op, raising=False
        )

        fake = NPUDeepseekSparseAttention.__new__(NPUDeepseekSparseAttention)
        fake.prefix = MODULE_NAME
        fake.attn = SimpleNamespace(layer_name=KV_LAYER_NAME)

        out = fake.forward_mhc_deferred("x", "cos", "sin", "res", "mhc", "task", "topk")

        op.assert_called_once_with(
            "x", "cos", "sin", "res", MODULE_NAME, "mhc", "task", KV_LAYER_NAME
        )
        assert out == (("h", "post", "res"), "topk")


# =========================
# 3. NPUPanguSparseAttention / npu_pangu_forward
# =========================

@pytest.mark.unit
class TestPanguKVTransferLifecycle:

    def test_op_runs_wait_kernel_save_on_the_kv_layer_name(self, monkeypatch):
        import omni_npu.v1.layers.attention.npu_pangu as pangu_mod
        from omni_npu.v1.layers.attention.npu_pangu import npu_pangu_forward

        events = []
        _enable_v1_group(monkeypatch, _recording_connector(events))

        attn_metadata = SimpleNamespace(
            causal=True,
            num_actual_tokens=1,
            num_decode_tokens=1,
            num_decodes=1,
            num_prefills=0,
            prefill=None,
            decode=SimpleNamespace(),
        )
        kv_cache = object()

        def fake_decode(hidden, cos, sin, meta, cla_swa, mome):
            events.append(("run",))
            return torch.zeros_like(hidden)

        module = SimpleNamespace(
            is_dsa_layer=False,
            is_cp_layer=False,
            is_attn_sp_layer=False,
            enable_flashcomm2=False,
            is_flashcomm2_layer=False,
            is_cla_reuse_layer=False,
            tp_size=1,
            _forward_decode=fake_decode,
        )
        ctx = _forward_context(module, kv_cache, attn_metadata)
        monkeypatch.setattr(pangu_mod, "get_forward_context", lambda: ctx)
        monkeypatch.setattr(vllm_attn, "get_forward_context", lambda: ctx)

        hidden = torch.zeros(1, 8)
        cos = torch.zeros(1, 2)
        out = npu_pangu_forward(
            hidden, cos, torch.zeros_like(cos), MODULE_NAME, KV_LAYER_NAME
        )

        _assert_lifecycle(events, kv_cache, attn_metadata)
        assert out.shape == hidden.shape

    @pytest.mark.parametrize("on_ascend950", [True, False])
    def test_forward_passes_the_two_names_on_both_paths(
        self, monkeypatch, on_ascend950
    ):
        """A5 (op) and non-950 (direct call) must pass the same two names."""
        import omni_npu.v1.layers.attention.npu_pangu as pangu_mod
        from omni_npu.v1.layers.attention.npu_pangu import NPUPanguSparseAttention

        op = MagicMock(return_value="a5")
        monkeypatch.setattr(torch.ops.vllm, "npu_pangu_forward", op, raising=False)
        direct = MagicMock(return_value="direct")
        monkeypatch.setattr(pangu_mod, "npu_pangu_forward", direct)

        fake = NPUPanguSparseAttention.__new__(NPUPanguSparseAttention)
        fake.on_ascend950 = on_ascend950
        fake.layer_name = MODULE_NAME
        fake.attn = SimpleNamespace(layer_name=KV_LAYER_NAME)

        out = fake.forward("hs", "cos", "sin")

        (op if on_ascend950 else direct).assert_called_once_with(
            "hs", "cos", "sin", MODULE_NAME, KV_LAYER_NAME
        )
        assert out == ("a5" if on_ascend950 else "direct")


# =========================
# 4. Deferred-MHC entry points
# =========================

@pytest.mark.unit
class TestMHCDeferredForwarding:
    """The MHC ops must forward both names and stay undecorated.

    ``run_attention_mhc_deferred`` calls the *inner* op directly, so the inner
    op fires the Connector lifecycle. Decorating the MHC op as well would fire
    it twice per layer.
    """

    @pytest.mark.parametrize("module_path,op_name", _MHC_OPS)
    def test_mhc_op_is_not_decorated(self, module_path, op_name):
        import importlib

        op = getattr(importlib.import_module(module_path), op_name)
        assert not hasattr(op, "__wrapped__"), op_name

    def test_mla_mhc_op_forwards_module_and_kv_layer_names(self, monkeypatch):
        import omni_npu.v1.layers.attention.npu_mla as mla_mod
        from omni_npu.v1.layers.attention.npu_mla import (
            npu_mla_forward,
            npu_mla_forward_mhc_deferred,
        )

        calls = {}

        def fake_call(attn_forward, hidden, cos, sin, residual, module, mhc, task, kv):
            calls.update(
                attn_forward=attn_forward,
                module=module,
                mhc=mhc,
                task=task,
                kv=kv,
            )
            return ("out", "post", "res")

        monkeypatch.setattr(mla_mod, "call_attention_mhc_deferred", fake_call)

        out = npu_mla_forward_mhc_deferred(
            "h", "cos", "sin", "res", MODULE_NAME, "mhc", "task", KV_LAYER_NAME
        )

        assert out == ("out", "post", "res")
        # The decorated inner op carries the lifecycle, not the MHC op.
        assert calls["attn_forward"] is npu_mla_forward
        assert calls["module"] == MODULE_NAME
        assert calls["kv"] == KV_LAYER_NAME
        assert (calls["mhc"], calls["task"]) == ("mhc", "task")

    def test_dsa_mhc_op_passes_both_names_to_the_inner_op(self, monkeypatch):
        import omni_npu.v1.layers.attention.npu_dsa as dsa_mod
        from omni_npu.v1.layers.attention.npu_dsa import (
            npu_dsa_forward,
            npu_dsa_forward_mhc_deferred,
        )

        seen = {}

        def fake_npu_dsa_forward(x, cos, sin, module, topk, kv):
            seen.update(module=module, kv=kv)
            return ("out", topk)

        monkeypatch.setattr(dsa_mod, "npu_dsa_forward", fake_npu_dsa_forward)
        monkeypatch.setattr(
            dsa_mod,
            "get_forward_context",
            lambda: SimpleNamespace(
                no_compile_layers={MODULE_NAME: SimpleNamespace(index_topk=2)}
            ),
        )

        def fake_call(attn_forward, hidden, cos, sin, residual, module, mhc, task, kv):
            attn_forward(hidden, cos, sin, module, kv)
            return ("out", "post", "res")

        monkeypatch.setattr(dsa_mod, "call_attention_mhc_deferred", fake_call)

        npu_dsa_forward_mhc_deferred(
            torch.zeros(3, 8), "cos", "sin", "res",
            MODULE_NAME, "mhc", "task", KV_LAYER_NAME,
        )

        assert seen == {"module": MODULE_NAME, "kv": KV_LAYER_NAME}

    @pytest.mark.parametrize("module_path,op_name", _MHC_OPS)
    def test_shared_fake_matches_the_mhc_op_signature(self, module_path, op_name):
        import importlib

        from omni_npu.layers.mhc.mhc_deferred import attention_mhc_deferred_fake

        op = getattr(importlib.import_module(module_path), op_name)
        assert inspect.signature(op) == inspect.signature(attention_mhc_deferred_fake)


# =========================
# 5. Op-level contract (anti-regression)
# =========================

@pytest.mark.unit
class TestOpDecoratorContract:
    """The decorator surface must stay identical across the attention layers."""

    _OPS = (
        ("omni_npu.v1.layers.attention.npu_mla", "npu_mla_forward"),
        ("omni_npu.v1.layers.attention.npu_dsa", "npu_dsa_forward"),
        ("omni_npu.v1.layers.attention.npu_pangu", "npu_pangu_forward"),
    )

    @staticmethod
    def _ops():
        import importlib

        return [
            getattr(importlib.import_module(module_path), op_name)
            for module_path, op_name in TestOpDecoratorContract._OPS
        ]

    def test_every_op_is_decorated(self):
        for op in self._ops():
            assert hasattr(op, "__wrapped__"), op.__name__

    def test_kv_layer_name_is_the_last_positional_parameter(self):
        """``maybe_transfer_kv_layer`` reads the anchor by slot, so it must be last."""
        for op in self._ops():
            params = list(inspect.signature(op).parameters)
            assert params[-1] == "layer_name", (op.__name__, params)
            assert "module_name" in params, (op.__name__, params)

    def test_module_and_kv_layer_names_are_distinct_parameters(self):
        for op in self._ops():
            params = list(inspect.signature(op).parameters)
            assert params.index("module_name") != params.index("layer_name")

    def test_op_does_not_gate_on_the_transfer_group(self):
        """The gate belongs to ``maybe_transfer_kv_layer``, not to the op body."""
        for op in self._ops():
            # ``inspect.unwrap`` skips every wrapper layer (breakable-cudagraph
            # included) down to the kernel body itself.
            source = inspect.getsource(inspect.unwrap(op))
            assert "has_kv_transfer_group" not in source, op.__name__
            assert "is_v1_kv_transfer_group" not in source, op.__name__

    @staticmethod
    def _classes():
        from omni_npu.v1.layers.attention.npu_dsa import NPUDeepseekSparseAttention
        from omni_npu.v1.layers.attention.npu_mla import NPUDeepseekMLAAttention
        from omni_npu.v1.layers.attention.npu_pangu import NPUPanguSparseAttention

        return (
            NPUDeepseekMLAAttention,
            NPUDeepseekSparseAttention,
            NPUPanguSparseAttention,
        )

    @staticmethod
    def _routing_names():
        return ("forward", "forward_mhc_deferred")

    def test_shim_surface_is_gone(self):
        """The per-class decorated shim must not come back."""
        for cls in self._classes():
            for name in _SHIM_NAMES:
                assert name not in cls.__dict__, (cls.__name__, name)

    def test_routing_methods_do_not_apply_the_transfer_decorator(self):
        """The decorator lives on the op; ``forward`` must not re-apply it."""
        for cls in self._classes():
            for name in self._routing_names():
                if name not in cls.__dict__:
                    continue
                source = inspect.getsource(getattr(cls, name))
                assert "maybe_transfer_kv_layer" not in source, (cls.__name__, name)


# =========================
# 6. No-connector pass-through (the decorator's own short-circuit)
# =========================

@pytest.mark.unit
class TestOpsShortCircuitWithoutGroup:
    """Without a v1 group the decorator must not resolve the attention context."""

    @staticmethod
    def _force_group_off(monkeypatch):
        monkeypatch.setattr(kv_transfer_utils, "has_kv_transfer_group", lambda: False)
        # ``get_attention_context`` lives in the vLLM attention module and reads
        # its own module-global ``get_forward_context``.
        monkeypatch.setattr(vllm_attn, "get_forward_context", _raise_if_called)

    def test_mla_op_short_circuits(self, monkeypatch):
        import omni_npu.v1.layers.attention.npu_mla as mla_mod
        from omni_npu.v1.layers.attention.npu_mla import npu_mla_forward

        self._force_group_off(monkeypatch)
        attn_metadata = SimpleNamespace(prefill=None, decode=SimpleNamespace())
        module = SimpleNamespace(
            param_sink_number=0,
            noncontiguous_kv=False,
            ena_sp=False,
            o_proj=SimpleNamespace(tp_size=1),
            _forward_decode=lambda x, *a: torch.zeros_like(x),
        )
        monkeypatch.setattr(
            mla_mod,
            "get_forward_context",
            lambda: SimpleNamespace(
                no_compile_layers={MODULE_NAME: module},
                attn_metadata=attn_metadata,
                slot_mapping={},
            ),
        )
        monkeypatch.setattr(
            mla_mod,
            "get_batch_desc",
            lambda meta: (slice(0, 1), slice(0, 1), False, True),
        )

        hs = torch.zeros(1, 8)
        cos = torch.zeros(1, 1, 1, 4)
        assert npu_mla_forward(
            hs, cos, torch.zeros_like(cos), MODULE_NAME, KV_LAYER_NAME
        ).shape == hs.shape

    def test_pangu_op_short_circuits(self, monkeypatch):
        import omni_npu.v1.layers.attention.npu_pangu as pangu_mod
        from omni_npu.v1.layers.attention.npu_pangu import npu_pangu_forward

        self._force_group_off(monkeypatch)
        module = SimpleNamespace(_forward_dummy=lambda hidden: hidden)
        monkeypatch.setattr(
            pangu_mod,
            "get_forward_context",
            lambda: SimpleNamespace(
                no_compile_layers={MODULE_NAME: module},
                attn_metadata=None,
            ),
        )

        hidden = torch.zeros(1, 8)
        out = npu_pangu_forward(
            hidden, torch.zeros(1, 2), torch.zeros(1, 2),
            MODULE_NAME, KV_LAYER_NAME,
        )
        assert torch.equal(out, hidden)


# =========================
# 7. Pangu metadata resolution by inner name
# =========================

@pytest.mark.unit
class TestPanguForwardMetadataResolution:

    @pytest.fixture(autouse=True)
    def _imports(self):
        import omni_npu.v1.layers.attention.npu_pangu as pangu_mod

        self.mod = pangu_mod

    def _run(self, monkeypatch, *, attn_metadata, is_cla_reuse_layer=False):
        fake = SimpleNamespace()
        fake.attn = SimpleNamespace(layer_name=KV_LAYER_NAME)
        fake.layer_name = MODULE_NAME
        fake.is_cla_reuse_layer = is_cla_reuse_layer
        fake.cla_swa_attn_name = OTHER_KV_LAYER_NAME
        fake.tp_size = 1
        fake.moe_comm_strategy = "allreduce"
        fake.is_cp_layer = False
        fake.is_attn_sp_layer = False
        fake.enable_flashcomm2 = False
        fake.is_flashcomm2_layer = False
        fake.is_dsa_layer = False
        fake._forward_decode = MagicMock(
            side_effect=lambda *args: (
                args[0].clone() if hasattr(args[0], "clone") else args[0]
            )
        )
        fake._forward_dummy = MagicMock(side_effect=lambda hidden: hidden.clone())

        ctx = SimpleNamespace(
            no_compile_layers={MODULE_NAME: fake},
            attn_metadata=attn_metadata,
            slot_mapping={},
        )
        monkeypatch.setattr(self.mod, "get_forward_context", lambda: ctx)

        hidden = torch.zeros(2, 8)
        cos = torch.zeros(2, 1, 1, 4)
        sin = torch.zeros_like(cos)
        out = self.mod.npu_pangu_forward(
            hidden, cos, sin, MODULE_NAME, KV_LAYER_NAME
        )
        return out, fake, hidden

    def _decode_arguments(self, fake):
        """Bind the last ``_forward_decode`` call onto the real parameter names."""
        method = self.mod.NPUPanguSparseAttention._forward_decode
        bound = inspect.signature(method).bind(
            fake, *fake._forward_decode.call_args.args
        )
        return bound.arguments

    def test_dict_metadata_resolves_attn_and_mome_by_inner_names(self, monkeypatch):
        attn_meta = SimpleNamespace(
            num_actual_tokens=2,
            num_decode_tokens=2,
            num_decodes=1,
            num_prefills=0,
            prefill=None,
            decode=SimpleNamespace(),
        )
        mome_meta = SimpleNamespace(decode=object(), prefill=None)
        out, fake, _ = self._run(
            monkeypatch,
            attn_metadata={
                KV_LAYER_NAME: attn_meta,
                f"{KV_LAYER_NAME}.mome": mome_meta,
            },
        )

        assert fake._forward_decode.call_count == 1
        kwargs = self._decode_arguments(fake)
        assert kwargs["attn_metadata"] is attn_meta
        assert kwargs["mome_metadata"] is mome_meta
        assert out.shape == (2, 8)

    def test_dict_metadata_missing_attn_key_falls_back_to_dummy(self, monkeypatch):
        out, fake, hidden = self._run(
            monkeypatch,
            attn_metadata={f"{KV_LAYER_NAME}.mome": object()},
        )

        fake._forward_dummy.assert_called_once_with(hidden)
        fake._forward_decode.assert_not_called()
        assert torch.equal(out, hidden)

    def test_non_dict_metadata_keeps_mome_none(self, monkeypatch):
        attn_meta = SimpleNamespace(
            num_actual_tokens=2,
            num_decode_tokens=2,
            num_decodes=1,
            num_prefills=0,
            prefill=None,
            decode=SimpleNamespace(),
        )
        out, fake, _ = self._run(monkeypatch, attn_metadata=attn_meta)

        assert fake._forward_decode.call_count == 1
        assert self._decode_arguments(fake)["mome_metadata"] is None
        assert out.shape == (2, 8)

    def test_none_metadata_falls_back_to_dummy(self, monkeypatch):
        out, fake, hidden = self._run(monkeypatch, attn_metadata=None)

        fake._forward_dummy.assert_called_once_with(hidden)
        assert torch.equal(out, hidden)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
