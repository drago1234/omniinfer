# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""CPU-only tests for static-kernel compiler concurrency control."""

import os
import unittest
from types import ModuleType
from unittest.mock import patch

from omni_npu import envs
from omni_npu.compilation import static_kernel_compile as mod


class TestStaticKernelCompileJobs(unittest.TestCase):
    def test_environment_variable_default_and_override(self):
        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(envs.OMNI_STATIC_KERNEL_COMPILE_JOBS, 0)
        with patch.dict(
            os.environ, {"OMNI_STATIC_KERNEL_COMPILE_JOBS": "32"}, clear=True
        ):
            self.assertEqual(envs.OMNI_STATIC_KERNEL_COMPILE_JOBS, 32)

    def test_default_does_not_import_or_patch_backend(self):
        with patch.object(mod.importlib, "import_module") as load:
            mod.configure_static_kernel_compile_jobs(0)
        load.assert_not_called()

    def test_cap_and_repeat_configuration(self):
        backend = ModuleType("static_kernel")
        original = lambda: "64"
        backend._compute_opc_compile_jobs = original
        with patch.object(mod.importlib, "import_module", return_value=backend):
            mod.configure_static_kernel_compile_jobs(4)
            self.assertEqual(backend._compute_opc_compile_jobs(), "4")
            # Reconfiguration must not wrap the previous cap of 4.
            mod.configure_static_kernel_compile_jobs(8)
            self.assertEqual(backend._compute_opc_compile_jobs(), "8")
            mod.configure_static_kernel_compile_jobs(128)
            self.assertEqual(backend._compute_opc_compile_jobs(), "64")
            self.assertIs(getattr(backend, mod._ORIGINAL), original)

    def test_bundled_layout_fallback(self):
        backend = ModuleType("static_kernel")
        backend._compute_opc_compile_jobs = lambda: "16"
        missing = ModuleNotFoundError(name="npugraph_ex._acl_concrete_graph")
        with (
            patch.object(mod.importlib, "import_module", side_effect=[missing, backend]),
            patch.object(mod.logger, "info") as info,
        ):
            mod.configure_static_kernel_compile_jobs(2)
            self.assertEqual(backend._compute_opc_compile_jobs(), "2")
        info.assert_called_once_with(
            "Static kernel op_compiler jobs: backend=%s, "
            "default=%d, limit=%d, effective=%d",
            "static_kernel",
            16,
            2,
            2,
        )

    def test_unsupported_backend_and_negative_limit(self):
        with self.assertRaises(ValueError):
            mod.configure_static_kernel_compile_jobs(-1)
        with patch.object(mod.importlib, "import_module", return_value=ModuleType("old")):
            with self.assertRaisesRegex(RuntimeError, "cannot enforce"):
                mod.configure_static_kernel_compile_jobs(4)

    def test_internal_import_error_is_not_silenced(self):
        missing = ModuleNotFoundError(name="required_compiler_dependency")
        with patch.object(mod.importlib, "import_module", side_effect=missing):
            with self.assertRaises(ModuleNotFoundError):
                mod.configure_static_kernel_compile_jobs(4)

    def test_backend_rejects_non_positive_default_job_count(self):
        backend = ModuleType("static_kernel")
        backend._compute_opc_compile_jobs = lambda: "0"
        with patch.object(mod.importlib, "import_module", return_value=backend):
            mod.configure_static_kernel_compile_jobs(4)
        with self.assertRaisesRegex(RuntimeError, "non-positive"):
            backend._compute_opc_compile_jobs()


if __name__ == "__main__":
    unittest.main()
