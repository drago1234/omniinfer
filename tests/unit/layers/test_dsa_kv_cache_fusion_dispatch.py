# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""Dependency-free dispatch tests; run directly with Python on a CPU host.

Extract the real method to test its dispatch contract without importing the
NPU/distributed model stack. These mocks do not validate kernel numerics.
"""
import ast
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import Mock


def tensor(shape):
    value = Mock()
    value.dim.return_value = len(shape)
    value.size.side_effect = lambda dim: shape[dim]
    value.view.return_value = value
    return value


class TestKVCacheFusionDispatch(unittest.TestCase):
    def setUp(self):
        path = Path(__file__).resolve().parents[3] / 'omni/v1/layers/attention/npu_dsa.py'
        cls = next(n for n in ast.parse(path.read_text()).body
                   if isinstance(n, ast.ClassDef) and n.name == 'NPUDeepseekSparseAttention')
        method = next(n for n in cls.body if isinstance(n, ast.FunctionDef)
                      and n.name == '_kv_norm_rope_cache')
        module = ast.Module(body=[
            ast.ImportFrom(module='__future__', names=[ast.alias(name='annotations')], level=0),
            method,
        ], type_ignores=[])
        self.nope, self.pe = tensor((3, 512)), tensor((3, 64))
        self.fused = Mock(return_value=(self.pe, self.nope))
        self.torch = SimpleNamespace(
            float16='float16', float8_e4m3fn='fp8',
            ops=SimpleNamespace(custom=SimpleNamespace(
                npu_ai_infra_kv_rmsnorm_rope_cache_v2=self.fused)),
            split=Mock(return_value=(self.nope, self.pe)),
            cat=Mock(return_value=tensor((3, 576))),
        )
        self.npu = SimpleNamespace(
            hifloat8='hif8', npu_scatter_nd_update_=Mock(),
            npu_kv_rmsnorm_rope_cache=Mock(return_value=(None, None, self.pe, self.nope)),
        )
        self.config = SimpleNamespace(enable_kv_rmsnorm_rope_cache=True)
        scope = dict(torch=self.torch, torch_npu=self.npu,
                     model_extra_config=SimpleNamespace(operator_opt_config=self.config),
                     cache_fit_shape=lambda cache, shape: cache)
        exec(compile(ast.fix_missing_locations(module), str(path), 'exec'), scope)
        self.method = scope['_kv_norm_rope_cache']
        self.norm = Mock(side_effect=lambda x: x)
        self.norm.weight, self.norm.variance_epsilon = object(), 1e-6
        self.host = SimpleNamespace(
            kv_a_layernorm=self.norm, noncontiguous_kv=True, is_sfa_c8=True,
            qk_rope_head_dim=64, kv_lora_rank=512, cache_dtype='fp8_ds_mla',
            on_ascend950=True, kv_nz=False, dtype='bf16', rope_interleaved=False,
            _apply_rope=Mock(side_effect=lambda x, *_: x), _scatter_c8_kv_cache=Mock(),
        )
        self.cache = tensor((4, 128, 1, 656))
        self.slots = SimpleNamespace(slot_mapping=tensor((3,)), slot_mapping_2d=tensor((3, 2)))

    def run_method(self, **kwargs):
        return self.method(
            self.host, tensor((3, self.host.kv_lora_rank + self.host.qk_rope_head_dim)),
            tensor((3, 1, 1, 64)), tensor((3, 1, 1, 64)),
            self.slots, (self.cache, tensor((4, 128, 1, 64))), **kwargs,
        )

    def test_fp8_uses_typed_view_and_pertile128(self):
        self.assertEqual(self.run_method(), (self.nope, self.pe))
        self.fused.assert_called_once()
        kw = self.fused.call_args.kwargs
        self.assertEqual(kw['ckv_cache_dtype'], 'fp8')
        self.cache.view.assert_called_once_with('fp8')
        self.assertIs(kw['ckv_cache'], self.cache)
        self.assertEqual(kw['quant_mode'], 'pertile128')
        self.assertEqual(kw['cache_mode'], 'PA')
        self.assertEqual(kw['rotary_mode'], 'half')
        self.host._scatter_c8_kv_cache.assert_not_called()
        self.norm.assert_not_called()

    def test_hif8_preserves_storage_and_sets_logical_dtype(self):
        self.host.cache_dtype = 'hif8_ds_mla'
        self.host.rope_interleaved = True
        self.run_method()
        kw = self.fused.call_args.kwargs
        self.assertEqual(kw['ckv_cache_dtype'], 'hif8')
        self.assertEqual(kw['rotary_mode'], 'interleave-half')
        self.cache.view.assert_not_called()

    def test_float8_kv_nz_uses_fused_pa_nz(self):
        self.host.kv_nz = True
        for cache_dtype, logical_dtype in [('fp8_ds_mla', 'fp8'), ('hif8_ds_mla', 'hif8')]:
            with self.subTest(cache_dtype=cache_dtype):
                self.host.cache_dtype = cache_dtype
                self.run_method()
                self.fused.assert_called_once()
                kw = self.fused.call_args.kwargs
                self.assertEqual(kw['cache_mode'], 'PA_NZ')
                self.assertEqual(kw['quant_mode'], 'pertile128')
                self.assertEqual(kw['ckv_cache_dtype'], logical_dtype)
                self.host._scatter_c8_kv_cache.assert_not_called()
                self.norm.assert_not_called()
                self.fused.reset_mock()

    def test_non_ascend950_float8_falls_back(self):
        # L/R are validated during initialization, not by this dispatch method.
        self.host.on_ascend950 = False
        self.run_method()
        self.fused.assert_not_called()
        self.host._scatter_c8_kv_cache.assert_called_once()

    def test_config_controls_fusion(self):
        self.config.enable_kv_rmsnorm_rope_cache = False
        self.run_method()
        self.fused.assert_not_called()
        self.host._scatter_c8_kv_cache.assert_called_once()
        self.host._scatter_c8_kv_cache.reset_mock()
        self.config.enable_kv_rmsnorm_rope_cache = True
        self.run_method()
        self.fused.assert_called_once()
        self.host._scatter_c8_kv_cache.assert_not_called()

    def test_cla_norm_override_reaches_fused_operator(self):
        norm = SimpleNamespace(weight=object(), variance_epsilon=1e-5)
        self.run_method(kv_a_layernorm=norm)
        self.assertIs(self.fused.call_args.args[1], norm.weight)
        self.assertEqual(self.fused.call_args.kwargs['epsilon'], 1e-5)

    def test_cla_separate_cache_is_not_treated_as_c8(self):
        self.run_method(packed_kv=False)
        self.fused.assert_not_called()
        self.host._scatter_c8_kv_cache.assert_not_called()
        self.assertEqual(self.npu.npu_scatter_nd_update_.call_count, 2)

    def test_nonquantized_packed_cache_retains_original_contract(self):
        self.host.is_sfa_c8 = False
        self.host.cache_dtype = 'auto'
        self.host.rope_interleaved = True
        self.cache = tensor((4, 128, 1, 576))
        self.run_method()
        kw = self.fused.call_args.kwargs
        self.assertEqual(kw['quant_mode'], 'none')
        self.assertEqual(kw['rotary_mode'], 'interleave')
        self.assertNotIn('ckv_cache_dtype', kw)

    def test_missing_cache_and_empty_tokens_skip_fused_operator(self):
        self.method(self.host, tensor((3, 576)), tensor((3, 1, 1, 64)),
                    tensor((3, 1, 1, 64)), None, None)
        cos = tensor((0, 1, 1, 64))
        self.method(self.host, tensor((0, 576)), cos, cos, None, None)
        self.fused.assert_not_called()
        self.assertEqual(cos.new_zeros.call_count, 2)


if __name__ == '__main__':
    unittest.main()
