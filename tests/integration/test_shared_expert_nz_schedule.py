# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""NPU numerical tests for shared-expert NZ and decode R2 -> S2 scheduling.

Real Linear loaders, NZ conversion, MatMul/GMM and NPU streams/events are used.
UT-03 fixes top-k=2 and replaces only routing/EP communication with local tensor
operations. It does not validate HCCL, graph replay or device execution overlap.
"""

from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
torch_npu = pytest.importorskip("torch_npu")
pytestmark = [pytest.mark.integration, pytest.mark.npu]

HIDDEN = 128
INTERMEDIATE = 256
# Explicit BF16 numerical tolerance for these bounded synthetic weights.
RTOL, ATOL = 1.6e-2, 1e-2


@pytest.fixture
def runtime(monkeypatch, default_vllm_config):
    if not torch.npu.is_available():
        pytest.skip("A real Ascend NPU is required")

    from vllm.model_executor import parameter
    from omni_npu.model_config.config_loader.loader import model_extra_config
    from omni_npu.v1.distributed import parallel_state_ext
    from omni_npu.v1.models.pangu import pangu_v2_moe as model

    # Shared experts have disable_tp=True. Supply only topology metadata; no
    # kernels or stream/event APIs are mocked and no HCCL group is necessary.
    monkeypatch.setattr(parameter, "get_tensor_model_parallel_rank", lambda: 0)
    monkeypatch.setattr(parameter, "get_tensor_model_parallel_world_size", lambda: 1)
    monkeypatch.setattr(model, "get_tp_group", lambda: SimpleNamespace(world_size=1))
    # FlashComm selects its quantization method through the custom-model path.
    # No configured transforms: shared experts remain local, without HCCL.
    monkeypatch.setenv("VLLM_PLUGINS", "omni_custom_models")
    monkeypatch.setattr(parallel_state_ext, "_LAYER_COMM_DICT", None)
    monkeypatch.setattr(model_extra_config.operator_opt_config, "unquant_bmm_nz", True)
    previous_dtype = torch.get_default_dtype()
    # Some torch_npu releases expose a write-only config property. Read the
    # underlying option so teardown restores the actual process-wide setting.
    previous_format = torch_npu._C._npu_getOption("ALLOW_INTERNAL_FORMAT")
    torch.set_default_dtype(torch.bfloat16)
    torch.npu.config.allow_internal_format = True
    try:
        with torch.inference_mode():
            yield model
    finally:
        try:
            torch.npu.synchronize()
        finally:
            torch.set_default_dtype(previous_dtype)
            torch_npu._C._npu_setOption(
                {"ALLOW_INTERNAL_FORMAT": previous_format.decode()}
            )


def _make_shared(model, mode):
    from compressed_tensors.quantization import QuantizationArgs, QuantizationStrategy
    from omni_npu.layers.quantization.compressed_tensors.compressed_tensors import (
        NPUCompressedTensorsConfig,
    )
    from omni_npu.v1.layers.linear import UnquantizedFlashCommLinearMethod
    from omni_npu.v1.layers.quantization.compressed_tensors.npu_compressed_tensors_linear import (
        W8A8Int8FCLinearMethod,
    )

    prefix = "model.layers.3.mlp.shared_experts"
    quant_config = None
    if mode == "w8a8":
        # Exercise real get_quant_method -> get_fc_method selection, as well
        # as the FlashComm weight loaders, NZ conversion and quantized MatMul.
        scheme = {
            "weights": QuantizationArgs(
                num_bits=8, type="int", strategy=QuantizationStrategy.CHANNEL,
                symmetric=True, dynamic=False,
            ),
            "input_activations": QuantizationArgs(
                num_bits=8, type="int", strategy=QuantizationStrategy.TOKEN,
                symmetric=True, dynamic=True,
            ),
        }
        quant_config = NPUCompressedTensorsConfig(
            {f"{prefix}.{name}": scheme for name in ("gate_up_proj", "down_proj")},
            [], "int-quantized",
        )
    mlp = model.OpenPanguV2MLP(
        HIDDEN, INTERMEDIATE, "silu", quant_config=quant_config,
        disable_tp=True, prefix=prefix,
    ).to("npu")
    generator = torch.Generator().manual_seed(123)
    original = []
    for index, layer in enumerate((mlp.gate_up_proj, mlp.down_proj)):
        shape = tuple(layer.weight.shape)
        if mode == "bf16":
            assert isinstance(layer.quant_method, UnquantizedFlashCommLinearMethod)
            weight = (torch.randn(shape, generator=generator, dtype=torch.float32)
                      / shape[-1] ** 0.5).to(torch.bfloat16).to("npu")
        else:
            assert isinstance(layer.quant_method, W8A8Int8FCLinearMethod)
            weight = torch.randint(-7, 8, shape, generator=generator,
                                   dtype=torch.int8).to("npu")
            layer.weight_scale.data.fill_(0.02)
            layer.weight_offset.data.zero_()
        # Exercise the actual checkpoint loaders, including separate gate/up
        # shards rather than copying an already-merged/transposed tensor.
        loader = layer.weight.weight_loader
        if index == 0:
            for shard_id, shard in enumerate(weight.chunk(2, dim=0)):
                loader(layer.weight, shard, shard_id)
        else:
            loader(layer.weight, weight)
        original.append(weight.clone())
        layer.quant_method.process_weights_after_loading(layer)
    return mlp, original


def _nd_projection(layer, weight, x, mode):
    if mode == "bf16":
        return torch.matmul(x, weight.t())
    x_int8, token_scale = torch_npu.npu_dynamic_quant(x)
    return torch_npu.npu_quant_matmul(
        x_int8, weight.t().contiguous(), layer.weight_scale,
        pertoken_scale=token_scale, output_dtype=torch.bfloat16,
    )


def _input(tokens):
    return torch.randn(
        (tokens, HIDDEN), generator=torch.Generator().manual_seed(456),
        dtype=torch.float32,
    ).to(dtype=torch.bfloat16, device="npu")


@pytest.mark.parametrize("mode", ["bf16", "w8a8"])
def test_ut01_loaded_shared_weights_are_nz(runtime, mode):
    """Both projections retain exact values/dtype and the required [K,N] shape."""
    mlp, original = _make_shared(runtime, mode)
    for layer, source in zip((mlp.gate_up_proj, mlp.down_proj), original):
        assert torch_npu.get_npu_format(layer.weight) == 29
        assert layer.weight.shape == source.t().shape
        assert layer.weight.dtype == source.dtype
        restored = torch_npu.npu_format_cast(layer.weight, torch_npu.Format.ND)
        torch.testing.assert_close(restored.cpu(), source.t().cpu(), rtol=0, atol=0)


@pytest.mark.parametrize("mode", ["bf16", "w8a8"])
@pytest.mark.parametrize("tokens", [1, 2, 256])
def test_ut02_nd_nz_projections_and_shared_mlp_agree(runtime, mode, tokens):
    """Compare S1, S2 on the same activation, and the complete production MLP."""
    mlp, (w13, w2) = _make_shared(runtime, mode)
    x = _input(tokens)
    reference_s1 = _nd_projection(mlp.gate_up_proj, w13, x, mode)
    actual_s1 = mlp.gate_up_proj(x)
    torch.testing.assert_close(actual_s1, reference_s1, rtol=RTOL, atol=ATOL)
    activation = mlp.act_fn(reference_s1)
    reference_s2 = _nd_projection(mlp.down_proj, w2, activation, mode)
    torch.testing.assert_close(mlp.down_proj(activation), reference_s2, rtol=RTOL, atol=ATOL)
    actual = mlp(x)
    assert actual.shape == (tokens, HIDDEN)
    assert actual.dtype == torch.bfloat16
    torch.testing.assert_close(actual, reference_s2, rtol=RTOL, atol=ATOL)


def _make_routed(mode):
    """Small non-square expert weights; reuse production quantized packing."""
    from omni_npu.layers.quantization.compressed_tensors.compressed_tensors_moe import (
        NPUCompressedTensorsW4A8Int4MoEMethod,
        NPUCompressedTensorsW8A8Int8MoEMethod,
    )

    experts = torch.nn.Module()
    experts.top_k, experts.topk_group, experts.num_expert_group = 2, 1, 1
    generator = torch.Generator().manual_seed(789)
    for name, n, k in (("w13", 2 * INTERMEDIATE, HIDDEN), ("w2", HIDDEN, INTERMEDIATE)):
        if mode == "bf16":
            weight = (torch.randn((2, n, k), generator=generator, dtype=torch.float32)
                      / k ** 0.5).to(torch.bfloat16)
        else:
            weight = torch.randint(-7, 8, (2, n, k), generator=generator, dtype=torch.int8)
        if mode == "w4a8":
            # Checkpoint INT4 pairs are packed along N, before the production
            # post-load transpose/NZ/reinterpretation as int32.
            weight = ((weight[:, 0::2, :] & 15)
                      | ((weight[:, 1::2, :] & 15) << 4))
            scale = torch.full((2, 1, n), 0.02, dtype=torch.float32)
            packed_scale = scale.view(torch.int32).to(torch.int64) & 0xFFFFFFFF
            setattr(experts, f"{name}_weight_int4_scale",
                    torch.nn.Parameter(packed_scale.to("npu"), requires_grad=False))
            setattr(experts, f"{name}_weight_bias",
                    torch.nn.Parameter(torch.zeros((2, n), device="npu", dtype=torch.float32),
                                       requires_grad=False))
            setattr(experts, f"{name}_weight_offset", None)
        elif mode == "w8a8":
            setattr(experts, f"{name}_weight_scale",
                    torch.nn.Parameter(torch.full((2, n, 1), 0.02, device="npu",
                                                  dtype=torch.bfloat16), requires_grad=False))
        else:
            weight = torch_npu.npu_format_cast(weight.transpose(1, 2).contiguous().to("npu"), 29)
        setattr(experts, f"{name}_weight",
                torch.nn.Parameter(weight.to("npu"), requires_grad=False))
    if mode == "w8a8":
        NPUCompressedTensorsW8A8Int8MoEMethod.process_weights_after_loading(None, experts)
    elif mode == "w4a8":
        NPUCompressedTensorsW4A8Int4MoEMethod.process_weights_after_loading(
            SimpleNamespace(asymmetric_quant=False), experts
        )
    return experts


def _local_dispatch(*, x, quant_mode, **kwargs):
    # Fixed top-k: every token visits both experts. Expert-major layout.
    expanded = torch.cat((x, x), dim=0)
    scale = None
    if quant_mode == 2:
        expanded, scale = torch_npu.npu_dynamic_quant(expanded)
    counts = torch.tensor([x.shape[0], x.shape[0]], device=x.device, dtype=torch.int64)
    return expanded, scale, None, counts, None, None


def _local_combine(*, expand_x, expert_scales, **kwargs):
    first, second = expand_x.chunk(2, dim=0)
    return (first.float() * expert_scales[:, :1]
            + second.float() * expert_scales[:, 1:]).to(torch.bfloat16)


class _BeforeR2Wait(torch.nn.Module):
    """Baseline: original shared forward, ignoring only the newly added wait."""

    def __init__(self, mlp):
        super().__init__()
        self.mlp = mlp

    def forward(self, x, down_proj_ready_event=None):
        return self.mlp(x)


@pytest.mark.parametrize("shared_mode", ["bf16", "w8a8"])
@pytest.mark.parametrize("routed_mode", ["bf16", "w8a8", "w4a8"])
@pytest.mark.parametrize("tokens", [1, 2, 256, 257])
def test_ut03_decode_two_stream_outputs_agree(
    runtime, monkeypatch, shared_mode, routed_mode, tokens,
):
    """Real two-stream output regression; 257 tokens exercise the final sub-batch.

    Compare the pre-change no-R2-wait behavior with the production event path.
    This asserts numerical preservation, not that the kernels are exclusive.
    """
    from omni_npu.model_config.config_loader.loader import model_extra_config

    # This suite covers the standard W4A8 GMM API, not the optional EnK kernel.
    monkeypatch.setattr(model_extra_config.operator_opt_config, "use_w4a8_enk_gmm", False)
    shared, _ = _make_shared(runtime, shared_mode)
    moe = runtime.OpenPanguV2MOE.__new__(runtime.OpenPanguV2MOE)
    torch.nn.Module.__init__(moe)
    moe.experts = _make_routed(routed_mode)
    moe.act_fn = runtime.SiluAndMul()
    moe.routed_expert_down_proj = None
    moe.routed_expert_up_proj = None
    moe.routed_expert_norm = None
    moe.shared_experts = _BeforeR2Wait(shared)
    moe._is_quant = routed_mode != "bf16"
    moe._is_w4a8 = routed_mode == "w4a8"
    moe.ep_comm_name = "local-test-only"
    moe.ep_size, moe.ep_rank = 1, 0
    moe.n_physical_experts = 2
    moe.routed_scaling_factor = 1.0
    moe.e_score_correction_bias = None
    moe.enable_eplb = moe.use_moe_force_load_balance = False
    moe.moe_dispatch_combine_max_batch_size = 256
    moe.side_stream = torch.npu.Stream()
    moe.gate = lambda x: (torch.zeros((x.shape[0], 2), device=x.device, dtype=torch.float32), None)
    moe._get_mc2_mask = lambda _: None
    x = _input(tokens)
    ids = torch.tensor([[0, 1]], device="npu", dtype=torch.int32).repeat(tokens, 1)
    scales = torch.tensor([[0.25, 0.75]], device="npu", dtype=torch.float32).repeat(tokens, 1)
    monkeypatch.setattr(runtime.torch_npu, "npu_moe_gating_top_k", lambda *a, **kw: (scales, ids, None))
    monkeypatch.setattr(runtime.torch_npu, "npu_moe_distribute_dispatch_v2", _local_dispatch)
    monkeypatch.setattr(runtime.torch_npu, "npu_moe_distribute_combine_v2", _local_combine)

    baseline = moe._forward_dispatch_combine(x)
    torch.npu.synchronize()
    moe.shared_experts = shared
    for _ in range(2):
        actual = moe._forward_dispatch_combine(x)
        torch.npu.synchronize()
        assert actual.shape == baseline.shape == (tokens, HIDDEN)
        assert actual.dtype == baseline.dtype == torch.bfloat16
        torch.testing.assert_close(actual, baseline, rtol=RTOL, atol=ATOL)
