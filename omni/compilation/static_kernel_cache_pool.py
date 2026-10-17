# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""Engine bridge for the static-kernel compile-cache pool.

Layering: the pool mechanics — preflighted, time-bounded store
construction, the staged atomic restore, run-package consolidation and
the first-wins put — live in the static_kernel_cache package
(static_kernel_cache_store), which also owns the ambient pool-config
resolution and the pull-timeout default. This module keeps only engine
lifecycle and policy:

- the ``static_kernel_cache_enable`` config gate, checked BEFORE any static_kernel_cache
  import so a disabled feature never requires the package;
- rank0/barrier coordination (torch.distributed engine semantics);
- the boot reuse marker ``.static_kernel_cache_pool_restored`` — written ONLY by this
  bridge after a gate-passed pool HIT, read by
  ``static_kernel_cache_patch._cache_present`` (single definition: imported from
  static_kernel_cache_patch, which must stay loadable in file-path fallback mode).
  The pool is thereby the only cache source: a CLI ``static-kernel-cache pull``
  restores products but writes no marker, so CLI-restored products never
  satisfy the boot decision — the CLI is an inspection/debug tool, not a
  prewarm channel;
- the stable log lines the verification tooling greps.

npu_model_runner.py call sites are unchanged.
"""

from __future__ import annotations

import logging
import os
import threading
from pathlib import Path
from typing import Any

from .static_kernel_cache_patch import _POOL_MARKER

logger = logging.getLogger(__name__)


def _sk_config() -> dict[str, Any] | None:
    try:
        from omni_npu.compilation.npugraph_ex_config import get_aclgraph_config

        return get_aclgraph_config().npugraph_ex_config
    except Exception:
        return None


def _rank() -> int:
    import torch.distributed as dist

    if dist.is_available() and dist.is_initialized():
        return dist.get_rank()
    return 0


def _kill_pool_marker(cfgdict: dict[str, Any]) -> None:
    """Remove the reuse marker so leftover products can't satisfy the
    boot-static decision — the pool is the sole cache source.
    """
    try:
        root = Path(cfgdict.get("static_kernel_cache_dir") or os.getcwd())
        (root / "static_kernel_compile_outputs" / _POOL_MARKER).unlink(missing_ok=True)
    except OSError:
        logger.exception("static_kernel_cache: failed to clear %s", _POOL_MARKER)


def _fingerprint(vllm_config: Any):
    from static_kernel_cache.fingerprint import compute_static_kernel_cache_fingerprint

    additional = getattr(vllm_config, "additional_config", None) or {}
    model_path = vllm_config.model_config.model
    return compute_static_kernel_cache_fingerprint(vllm_config, additional, model_path)


def _barrier_once() -> None:
    """The single rendezvous every rank passes after the static_kernel_cache gate. Every
    path through maybe_pull_static_kernel_cache must reach exactly one barrier —
    rank0's early returns (package missing, hook failure) included — or the
    non-zero ranks block on theirs for the whole HCCL timeout."""
    import torch.distributed as dist

    if dist.is_available() and dist.is_initialized():
        dist.barrier()


def maybe_pull_static_kernel_cache(vllm_config: Any) -> None:
    """Pull hook — see module docstring. Never raises.

    rank0-of-pod runs the (preflighted, time-bounded) pull and rewrites the
    reuse marker only on a gate-passed HIT; every past-the-gate path —
    success, MISS, missing package, failure — reaches exactly one barrier.
    """
    try:
        cfgdict = _sk_config()
        if not cfgdict or not cfgdict.get("static_kernel_cache_enable", False):
            return  # gate off: every rank exits here, no rendezvous needed

        try:
            # Only rank0 talks to the pool: MooncakeDistributedStore.setup
            # installs the Ascend transfer engine, and concurrent init from
            # the 4 DP ranks on one host races on the engine listen port.
            if _rank() != 0:
                return  # non-zero ranks rendezvous in the finally below

            # Kill the reuse marker BEFORE anything can fail; only a
            # gate-passed HIT below may rewrite it.
            _kill_pool_marker(cfgdict)

            try:
                from static_kernel_cache.static_kernel_cache_store import (
                    BundleLayout,
                    PullResult,
                    bounded_pull,
                )
            except ImportError as exc:
                logger.info(
                    "static_kernel_cache: static_kernel_cache package not importable (%s); "
                    "continuing with full compile "
                    "(ungated local reuse disabled)",
                    exc,
                )
                return

            root = cfgdict.get("static_kernel_cache_dir") or os.getcwd()
            fingerprint, _factors = _fingerprint(vllm_config)
            result = bounded_pull(BundleLayout(root), fingerprint)
            logger.info(
                "static_kernel_cache pool pull: %s (fingerprint=%s)",
                result.value if result else "failed",
                fingerprint,
            )
            if result is PullResult.HIT:
                # Revive reuse only now — the gate-passed bundle has
                # landed. Written before the barrier so every rank's
                # boot-static decision (post-barrier) observes it.
                (Path(root) / "static_kernel_compile_outputs" / _POOL_MARKER).write_text(
                    f"{fingerprint}\n"
                )
        finally:
            # Single barrier for ALL past-the-gate paths: success, MISS,
            # missing package, and exceptions alike. rank0 wrote (or did not
            # write) the marker before this point either way.
            _barrier_once()
    except Exception:
        logger.exception(
            "static_kernel_cache pool pull failed; continuing with full compile "
            "(ungated local reuse disabled)"
        )


def schedule_static_kernel_cache_upload(vllm_config: Any) -> None:
    """Upload hook — never raises; work happens on a daemon thread so
    capture completion is not delayed.
    """
    try:
        cfgdict = _sk_config()
        if not cfgdict or not cfgdict.get("static_kernel_cache_enable", False):
            return
        if _rank() != 0:
            return

        def _upload() -> None:
            try:
                import time

                time.sleep(5)  # let the last index-json appends settle (flock'd anyway)

                try:
                    from static_kernel_cache.static_kernel_cache_store import (
                        BundleLayout,
                        bounded_upload,
                    )
                except ImportError as exc:
                    logger.info(
                        "static_kernel_cache: static_kernel_cache package not importable (%s); "
                        "upload skipped",
                        exc,
                    )
                    return

                root = cfgdict.get("static_kernel_cache_dir") or os.getcwd()
                fingerprint, factors = _fingerprint(vllm_config)
                ok = bounded_upload(BundleLayout(root), fingerprint, factors)
                if ok is None:
                    return  # pool unreachable — already logged by the preflight
                logger.info(
                    "static_kernel_cache pool upload: %s (fingerprint=%s)",
                    "committed or already present" if ok else "skipped (bundle unusable)",
                    fingerprint,
                )
            except Exception:
                logger.exception("static_kernel_cache pool upload failed (cache stays local)")

        threading.Thread(target=_upload, daemon=True, name="static-kernel-cache-upload").start()
    except Exception:
        logger.exception("static_kernel_cache pool upload scheduling failed")
