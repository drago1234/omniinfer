# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""Limit op_compiler jobs without serializing model execution or collectives."""

import importlib
import logging

logger = logging.getLogger(__name__)

_MODULES = (
    "npugraph_ex._acl_concrete_graph.static_kernel",
    "torch_npu.dynamo.npugraph_ex._acl_concrete_graph.static_kernel",
)
_ORIGINAL = "_omni_original_compute_opc_compile_jobs"


def configure_static_kernel_compile_jobs(max_jobs: int) -> None:
    """Cap each op_compiler invocation; zero leaves the backend untouched.

    The wrapper must persist after backend creation: static kernels are compiled
    lazily during graph execution. Only the job-count helper is replaced, never
    os.cpu_count, model forward, or the multi-rank synchronization path.
    """
    if max_jobs < 0:
        raise ValueError("OMNI_STATIC_KERNEL_COMPILE_JOBS must be >= 0")
    if max_jobs == 0:
        return

    modules = []
    for name in _MODULES:
        try:
            module = importlib.import_module(name)
        except ModuleNotFoundError as exc:
            # A missing optional package/layout is allowed; a missing dependency
            # inside an installed backend is a real error and must not be hidden.
            if exc.name and (name == exc.name or name.startswith(exc.name + ".")):
                continue
            raise
        if all(module is not other for other in modules):
            modules.append(module)

    if not modules or any(
        not callable(getattr(module, "_compute_opc_compile_jobs", None))
        for module in modules
    ):
        raise RuntimeError(
            "OMNI_STATIC_KERNEL_COMPILE_JOBS was set, but this npugraph_ex "
            "version does not expose _compute_opc_compile_jobs; "
            "cannot enforce the static kernel compiler job limit."
        )

    for module in modules:
        original = getattr(module, _ORIGINAL, module._compute_opc_compile_jobs)

        def capped_jobs(
            original=original,
            limit=max_jobs,
            backend_name=module.__name__,
        ):
            default_jobs = int(original())
            if default_jobs <= 0:
                raise RuntimeError("npugraph_ex returned a non-positive compiler job count")
            jobs = min(default_jobs, limit)
            logger.info(
                "Static kernel op_compiler jobs: backend=%s, "
                "default=%d, limit=%d, effective=%d",
                backend_name,
                default_jobs,
                limit,
                jobs,
            )
            return str(jobs)

        # Keep one original per module, even across multiple FX graph compiles.
        setattr(module, _ORIGINAL, original)
        module._compute_opc_compile_jobs = capped_jobs
