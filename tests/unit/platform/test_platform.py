# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
import os
import subprocess
import sys
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch
import pytest

from omni_npu.platform import NPUPlatform


class TestNPUPlatform:
    def test_platform_attributes(self):
        """Test platform class attributes.
        
        Verifies that NPUPlatform has the correct static attributes
        for device name, type, dispatch key, and backend configuration.
        """
        assert NPUPlatform.device_name == "npu"
        assert NPUPlatform.device_type == "npu"
        assert NPUPlatform.dispatch_key == "PrivateUse1"
        assert NPUPlatform.ray_device_key == "NPU"
        assert NPUPlatform.dist_backend == "hccl"
        assert NPUPlatform.device_control_env_var == "ASCEND_RT_VISIBLE_DEVICES"

    def test_platform_init(self):
        """Test platform initialization.
        
        Verifies that NPUPlatform can be instantiated correctly.
        """
        platform = NPUPlatform()
        assert isinstance(platform, NPUPlatform)

    def test_is_cuda_alike_is_false_for_normal_npu_callers(self, monkeypatch):
        monkeypatch.setattr(
            "omni_npu.platform.traceback.format_stack",
            lambda: ["root", "caller", "is_cuda_alike"],
        )
        monkeypatch.setattr(
            "omni_npu.platform.traceback.extract_stack",
            lambda limit: [SimpleNamespace(filename="/tmp/normal.py", name="caller")],
        )

        assert NPUPlatform().is_cuda_alike() is False

    def test_is_cuda_alike_is_true_for_parallel_config_callsite(self, monkeypatch):
        monkeypatch.setattr(
            "omni_npu.platform.traceback.format_stack",
            lambda: [
                "root",
                "/opt/vllm/vllm/config/parallel.py current_platform.is_cuda_alike()",
                "is_cuda_alike",
            ],
        )
        monkeypatch.setattr(
            "omni_npu.platform.traceback.extract_stack",
            lambda limit: [
                SimpleNamespace(
                    filename="/tmp/moved/parallel.py",
                    name="__post_init__",
                )
            ],
        )

        assert NPUPlatform().is_cuda_alike() is True

    def test_is_cuda_alike_is_true_for_bind_kv_cache_callsite(self, monkeypatch):
        monkeypatch.setattr(
            "omni_npu.platform.traceback.format_stack",
            lambda: ["root", "caller", "is_cuda_alike"],
        )
        monkeypatch.setattr(
            "omni_npu.platform.traceback.extract_stack",
            lambda limit: [
                SimpleNamespace(
                    filename="/opt/vllm/vllm/v1/worker/utils.py",
                    name="bind_kv_cache",
                )
            ],
        )

        assert NPUPlatform().is_cuda_alike() is True

    def test_torch_npu_proxy_methods(self, monkeypatch):
        """Test all methods that directly proxy to torch.npu (consolidated test).
        
        Verifies that all proxy methods correctly delegate to torch.npu
        and return the expected values.
        """
        # Mock torch.npu methods
        mock_set_device = MagicMock()
        mock_get_device_name = MagicMock(return_value="Ascend910")
        mock_device_count = MagicMock(return_value=8)
        mock_mem_get_info = MagicMock(return_value=(1000, 2000))
        mock_reset_peak = MagicMock()
        mock_max_memory = MagicMock(return_value=500.0)

        monkeypatch.setattr("torch.npu.set_device", mock_set_device)
        monkeypatch.setattr("torch.npu.get_device_name", mock_get_device_name)
        monkeypatch.setattr("torch.npu.device_count", mock_device_count)
        monkeypatch.setattr("torch.npu.mem_get_info", mock_mem_get_info)
        monkeypatch.setattr("torch.npu.reset_peak_memory_stats", mock_reset_peak)
        monkeypatch.setattr("torch.npu.max_memory_allocated", mock_max_memory)

        # Test set_device
        test_device = torch.device("npu:0")
        NPUPlatform.set_device(test_device)
        mock_set_device.assert_called_once_with(test_device)

        # Test get_device_name
        result = NPUPlatform.get_device_name(0)
        assert result == "Ascend910"
        mock_get_device_name.assert_called_once_with(0)

        # Test device_count
        result = NPUPlatform.device_count()
        assert result == 8
        mock_device_count.assert_called_once()

        # Test mem_get_info
        result = NPUPlatform.mem_get_info()
        assert result == (1000, 2000)
        mock_mem_get_info.assert_called_once()

        # Test get_current_memory_usage
        result = NPUPlatform.get_current_memory_usage(test_device)
        assert result == 500.0
        mock_reset_peak.assert_called_once_with(test_device)
        mock_max_memory.assert_called_once_with(test_device)

    def test_inference_mode(self):
        """Test inference_mode method.
        
        Verifies that inference_mode returns a context manager
        similar to torch.no_grad().
        """
        result = NPUPlatform.inference_mode()
        assert isinstance(result, torch.no_grad().__class__)

    def test_import_kernels(self, monkeypatch):
        """Test import_kernels method.
        
        Verifies that import_kernels calls patch_compile_decorators
        and register_connectors.
        """
        patch_decorators_called = {"called": False}
        register_connectors_called = {"called": False}

        def mock_patch_decorators():
            patch_decorators_called["called"] = True

        def mock_register_connectors():
            register_connectors_called["called"] = True

        monkeypatch.setattr(
            "omni_npu.compilation.decorators.patch_compile_decorators",
            mock_patch_decorators,
        )
        monkeypatch.setattr(
            "omni_npu.connector.register_connectors",
            mock_register_connectors,
        )

        NPUPlatform.import_kernels()
        assert patch_decorators_called["called"] is True
        assert register_connectors_called["called"] is True

    def test_import_kernels_connector_load_failure(self, monkeypatch):
        """Test import_kernels handles connector load failure (covers lines 89-93).

        Verifies that when a connector entry point fails to load,
        a warning is logged and the method continues without raising.
        """
        patch_decorators_called = {"called": False}
        register_connectors_called = {"called": False}

        def mock_patch_decorators():
            patch_decorators_called["called"] = True

        def mock_register_connectors():
            register_connectors_called["called"] = True

        monkeypatch.setattr(
            "omni_npu.compilation.decorators.patch_compile_decorators",
            mock_patch_decorators,
        )
        monkeypatch.setattr(
            "omni_npu.connector.register_connectors",
            mock_register_connectors,
        )

        # Create a mock entry point that fails to load
        mock_ep = MagicMock()
        mock_ep.name = "failing_connector"
        mock_ep.load = MagicMock(side_effect=ImportError("Connector not found"))

        # Mock entry_points to return our failing entry point
        # entry_points is imported inside import_kernels, so we patch the builtin
        from importlib.metadata import entry_points

        def mock_entry_points():
            class MockEntryPointsResult:
                def select(self, group):
                    if group == "omni_npu.kv_connectors":
                        return [mock_ep]
                    return []
            return MockEntryPointsResult()

        monkeypatch.setattr(
            "omni_npu.platform.entry_points",
            mock_entry_points,
        )

        # Mock logger.warning to verify it's called
        warning_called = {"called": False, "msg": ""}
        original_warning = warning_called

        def mock_warning(msg):
            original_warning["called"] = True
            original_warning["msg"] = msg

        monkeypatch.setattr("omni_npu.platform.logger.warning", mock_warning)

        # Should not raise, just log a warning
        NPUPlatform.import_kernels()

        assert patch_decorators_called["called"] is True
        assert register_connectors_called["called"] is True
        assert warning_called["called"] is True
        assert "Failed to load connector" in warning_called["msg"]
        assert "failing_connector" in warning_called["msg"]

    def test_pre_register_and_update(self, monkeypatch):
        """Test pre_register_and_update method (covers lines 111-112).
        
        This method mainly imports modules, we just need to ensure
        it doesn't raise exceptions and imports are successful.
        """
        # Verify imports don't raise exceptions
        NPUPlatform.pre_register_and_update()
        # The method should complete without errors

    def test_get_punica_wrapper(self):
        """Test get_punica_wrapper method.
        
        Verifies that the correct Punica wrapper class name is returned.
        """
        result = NPUPlatform.get_punica_wrapper()
        assert result == "vllm.lora.punica_wrapper.punica_cpu.PunicaWrapperCPU"

    def test_get_device_communicator_cls(self):
        """Test get_device_communicator_cls method.
        
        Verifies that the correct device communicator class name is returned.
        """
        result = NPUPlatform.get_device_communicator_cls()
        assert result == "omni_npu.distributed.communicator.NPUCommunicator"

    def test_get_attn_backend_cls(self, monkeypatch):
        """Test get_attn_backend_cls method.
        
        Verifies that the correct attention backend class is returned
        based on use_mla and use_sparse flags, with and without VLLM_PLUGINS.
        """
        # Test use_mla=True, use_sparse=True (without VLLM_PLUGINS, covers line 161)
        monkeypatch.delenv("VLLM_PLUGINS", raising=False)
        from vllm.v1.attention.selector import AttentionSelectorConfig

        result = NPUPlatform.get_attn_backend_cls(
            "test",
            AttentionSelectorConfig(
                head_size=64,
                dtype=torch.float16,
                kv_cache_dtype="float16",
                block_size=16,
                use_mla=True,
                has_sink=False,
                use_sparse=True,
            ))
        assert result == "omni_npu.attention.backends.dsa.NPUDSABackend"

        # Test use_mla=True, use_sparse=False
        result = NPUPlatform.get_attn_backend_cls(
            "test",
            AttentionSelectorConfig(
                head_size=64,
                dtype=torch.float16,
                kv_cache_dtype="float16",
                block_size=16,
                use_mla=True,
                has_sink=False,
                use_sparse=False,
            ))
        assert result == "omni_npu.attention.backends.mla.NPUMLABackend"

        # Test use_mla=False
        result = NPUPlatform.get_attn_backend_cls(
            "test",
            AttentionSelectorConfig(
                head_size=64,
                dtype=torch.float16,
                kv_cache_dtype="float16",
                block_size=16,
                use_mla=False,
                has_sink=False,
                use_sparse=False,
            ))
        assert result == "omni_npu.attention.backends.attention.NPUAttentionBackend"

        # Test with VLLM_PLUGINS containing "omni_custom_models" (covers lines 150-157)
        monkeypatch.setenv("VLLM_PLUGINS", "omni_custom_models")
        result = NPUPlatform.get_attn_backend_cls(
            "test",
            AttentionSelectorConfig(
                head_size=64,
                dtype=torch.float16,
                kv_cache_dtype="float16",
                block_size=16,
                use_mla=True,
                has_sink=False,
                use_sparse=True,
            ))
        assert result == "omni_npu.attention.backends.dsa.NPUDSABackend"

        result = NPUPlatform.get_attn_backend_cls(
            "test",
            AttentionSelectorConfig(
                head_size=64,
                dtype=torch.float16,
                kv_cache_dtype="float16",
                block_size=16,
                use_mla=True,
                has_sink=False,
                use_sparse=False,
            ))
        assert result == "omni_npu.attention.backends.mla.NPUMLABackend"

        result = NPUPlatform.get_attn_backend_cls(
            "test",
            AttentionSelectorConfig(
                head_size=64,
                dtype=torch.float16,
                kv_cache_dtype="float16",
                block_size=16,
                use_mla=False,
                has_sink=False,
                use_sparse=False,
            ))
        assert result == "omni_npu.attention.backends.attention.NPUAttentionBackend"

    def test_simple_compile_backend(self):
        """Test simple_compile_backend property.
        
        Verifies that simple_compile_backend returns "eager".
        """
        platform = NPUPlatform()
        assert platform.simple_compile_backend == "eager"

    def test_support_static_graph_mode(self):
        """Test support_static_graph_mode method.
        
        Verifies that static graph mode is supported.
        """
        result = NPUPlatform.support_static_graph_mode()
        assert result is True

    def test_get_static_graph_wrapper_cls(self):
        """Test get_static_graph_wrapper_cls method.
        
        Verifies that the correct static graph wrapper class name is returned.
        """
        result = NPUPlatform.get_static_graph_wrapper_cls()
        assert result == "omni_npu.compilation.acl_graph.ACLGraphWrapper"

    def test_is_sleep_mode_available(self, monkeypatch):
        """Test the is_sleep_mode_available method of NPUPlatform."""
        platform = NPUPlatform()

        assert platform.is_sleep_mode_available() is True

@pytest.fixture
def mome_config(monkeypatch):
    monkeypatch.setattr(
        "omni_npu.model_config.config_loader.loader.model_extra_config.operator_opt_config.use_noncontiguous_kv",
        True,
    )
    return SimpleNamespace(
        model_config=SimpleNamespace(
            hf_config=SimpleNamespace(use_mome=True), max_model_len=131072,
        ),
        cache_config=SimpleNamespace(
            block_size=128, mamba_block_size=128, mamba_page_size_padded=None,
            enable_prefix_caching=True,
        ),
    )


@pytest.mark.parametrize("enable_prefix_caching,mamba_block_multiplier", [
    pytest.param(True, 1, id="apc"),
    pytest.param(True, 2, id="apc-different-blocks"),
    pytest.param(False, None, id="no-apc"),
])
@pytest.mark.parametrize("num_spec_tokens,cache_dtype,block_size,padding,expected", [
    # MTP3: each quantized DSA layout determines the largest page.
    pytest.param(3, "fp8_ds_mla", 128, None, 2 * 128 * (656 + 128 + 4), id="mtp3-fp8"),
    pytest.param(3, "hif8_ds_mla", 128, None, 2 * 128 * (656 + 128 + 4), id="mtp3-hif8"),
    pytest.param(3, "int8_ds_mla", 128, None, 2 * 128 * (656 + 128 + 2), id="mtp3-int8"),
    pytest.param(3, "li_int8_ds_mla", 128, None, 128 * (576 * 2 + 128 + 2), id="mtp3-li-int8"),
    # MTP7: nine BF16 states make MoME largest at block size 128.
    pytest.param(7, "auto", 128, None, 9 * 2 * (1536 + 512 + 64 * 128), id="mtp7-mome-largest"),
    pytest.param(7, "auto", 256, None, 256 * (576 + 128) * 2, id="mtp7-dsa-largest"),
    pytest.param(7, "auto", 128, 164096, 9 * 2 * (1536 + 512 + 64 * 128), id="mtp7-raise-small-padding"),
    pytest.param(7, "auto", 128, 294912, 294912, id="mtp7-keep-large-padding"),
])
def test_pangu_page_uses_registered_specs(
    monkeypatch, mome_config, num_spec_tokens, cache_dtype, block_size, padding, expected,
    enable_prefix_caching, mamba_block_multiplier,
):
    """Unify physical pages without enlarging logical blocks or shrinking padding."""
    from omni_npu.vllm_patches.patches.models.pangu_v2_base.patch_kv_cache_interface import (
        DSAAttentionSpec, MomeSpec, ShareKVSlidingWindowSpec,
    )

    # Pangu doubles the DSA storage block for FP8/HIF8/INT8 layouts.
    dsa_block_size = 2 * block_size if cache_dtype in (
        "fp8_ds_mla", "hif8_ds_mla", "int8_ds_mla",
    ) else block_size
    # Without APC, vLLM uses one logical MoME block per request.
    mamba_block_size = (
        block_size * mamba_block_multiplier
        if enable_prefix_caching else mome_config.model_config.max_model_len
    )
    specs = [
        DSAAttentionSpec(block_size=dsa_block_size, num_kv_heads=1, head_size=704,
                         dtype=torch.bfloat16, cache_dtype_str=cache_dtype),
        MomeSpec(block_size=mamba_block_size,
                 mamba_cache_mode="all" if enable_prefix_caching else "none", shapes=((1536,), (512,), (64 * 128,)),
                 dtypes=(torch.bfloat16,) * 3, kernel_size=3, num_spec_tokens=num_spec_tokens),
        ShareKVSlidingWindowSpec(block_size=block_size, num_kv_heads=1, head_size=576,
                                 dtype=torch.bfloat16, sliding_window=512),
    ]
    layers = {
        str(i): SimpleNamespace(get_kv_cache_spec=lambda cfg, spec=spec: spec)
        for i, spec in enumerate(specs)
    }
    monkeypatch.setattr("vllm.config.get_layers_from_vllm_config", lambda *args: layers)
    cache = mome_config.cache_config
    cache.enable_prefix_caching = enable_prefix_caching
    cache.block_size = block_size
    cache.mamba_block_size = mamba_block_size
    cache.mamba_page_size_padded = padding
    with patch("vllm.platforms.interface.Platform._align_hybrid_block_size") as upstream:
        NPUPlatform._align_hybrid_block_size(mome_config, object)
    upstream.assert_not_called()
    assert cache.block_size == block_size
    assert cache.mamba_block_size == mamba_block_size
    assert cache.mamba_page_size_padded == expected
    assert all(layer.page_size_padded == expected for layer in layers.values())


def test_pangu_page_rejects_contiguous_layout(monkeypatch, mome_config):
    """Reject layouts that cannot share padded physical pages safely."""
    monkeypatch.setattr(
        "omni_npu.model_config.config_loader.loader.model_extra_config.operator_opt_config.use_noncontiguous_kv",
        False,
    )
    with pytest.raises(ValueError, match="noncontiguous"):
        NPUPlatform._align_hybrid_block_size(mome_config, object)
