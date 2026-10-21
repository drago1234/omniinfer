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

- rank coordination over the engine's own process groups. The external
  data-parallel launcher places every API-server worker of a pod in one
  shared distributed world (world_size equals the total server count,
  one rendezvous), so plain rank checks and dist barriers DO serialize
  them — and serialization is mandatory because the servers share one
  product root: an uncoordinated landing rmtree+rename would race
  sibling compilations, and concurrent pulls would clobber the shared
  staging dir. The decisions are split by scope:
  - pull: the pod's rank 0 — the LOCAL world group's
    ``rank_in_group == 0`` — pulls, and every past-the-gate process
    rendezvouses on that same local group. The local group spans one
    pod's processes, so multi-node deployments get one puller and one
    barrier PER POD: each pod lands its own copy, and a slow pod can no
    longer stall another pod's boot (a world-group barrier would couple
    them);
  - upload: the WORLD rank 0 alone. Sibling pods compile identical
    products, so a single canonical commit suffices — the pool's
    first-wins put dedups the rest;
  - upload timing: a capture-end rendezvous on the local group replaces
    the old settle timer. Compilation happens inside the compiling
    process's capture, so once the rendezvous returns (every capture
    returned) the shared compile index is provably final — a partial-set
    commit is constructively impossible. The store-side gate
    (run_files == graph_keys) stays as the backstop for decidable
    anomalies such as a torn index json;

  - coordination failure: a pod group that cannot be established logs
    an error and abandons pooling for the boot. No silent fallback onto
    the world group — it would quietly change who pulls and recouple
    pods' boots. The failure is deterministic, so every rank exits
    before any collective and no rendezvous is owed;

- the boot reuse marker ``.static_kernel_cache_pool_restored`` — written ONLY by this
  bridge after a gate-passed pool HIT, read by
  ``static_kernel_cache_patch._cache_present`` (single definition: imported from
  static_kernel_cache_patch, which must stay loadable in file-path fallback mode).
  The pool is thereby the only cache source: a CLI ``static-kernel-cache pull``
  restores products but writes no marker, so CLI-restored products never
  satisfy the boot decision — the CLI is an inspection/debug tool, not a
  prewarm channel;

- the stable log lines the verification tooling greps.

Deployment contract: the local-size inference (NNODES, else
NUM_SERVERS, else visible devices) must match the real pod layout.
Multi-node decode MUST export NNODES — without it a leftover
NUM_SERVERS can stretch the local group to the whole world and couple
pods again. Multi-server pods MUST export NUM_SERVERS — without it each
externally pinned worker sees a single device, every process becomes
its own pod rank 0, and the pod-wide pull serialization is lost.

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


def _world_rank() -> int:
    """The global rank — the upload dedup decision only (exactly one
    process per cluster commits). Pull serialization is per-pod instead;
    see ``_pod_group``.
    """
    import torch.distributed as dist

    if dist.is_available() and dist.is_initialized():
        return dist.get_rank()
    return 0


class _PodCoordinationUnavailable(Exception):
    """The pod's process group could not be established; pooling is
    abandoned for the boot."""


def _pod_group() -> Any:
    """The process group spanning this pod's engine processes, or None.

    ``initialize_local_world_group`` is idempotent but collective: every
    process must run it BEFORE any rank branching, or the group creation
    wedges. None means no distributed runtime — a single process is its
    own pod. A build failure is logged and raised instead of falling
    back to the world group: the failure is deterministic, so every rank
    takes the same exit and owes no barrier.
    """
    import torch.distributed as dist

    if not (dist.is_available() and dist.is_initialized()):
        return None
    try:
        from omni_npu.v1.distributed.parallel_state_ext import (
            get_local_world_group,
            initialize_local_world_group,
        )

        initialize_local_world_group()
        return get_local_world_group()
    except Exception:
        logger.error(
            "static_kernel_cache: local world group unavailable; pool reuse "
            "abandoned for this boot (multi-node deployments must export "
            "NNODES, multi-server pods NUM_SERVERS)",
            exc_info=True,
        )
        raise _PodCoordinationUnavailable() from None


def _is_pod_rank0(pod_group: Any) -> bool:
    """The pull decision: rank 0 within the pod's group."""
    if pod_group is not None:
        return pod_group.rank_in_group == 0
    # No group = no distributed runtime; this single process is its own
    # rank 0.
    return _world_rank() == 0


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


def _barrier_once(group: Any = None) -> None:
    """The one rendezvous per past-the-gate process, on the pod group's
    own communicator; None (no distributed runtime) is a no-op. Every
    hook path that resolved the pod group must reach exactly one
    barrier — early returns included — or the other members block for
    the whole HCCL timeout."""
    import torch.distributed as dist

    if group is None:
        return
    dist.barrier(group=group.device_group)


def maybe_pull_static_kernel_cache(vllm_config: Any) -> None:
    """Pull hook — see module docstring. Never raises.

    The pod's rank 0 clears any stale reuse marker right before its
    bounded pull and rewrites the marker only on a gate-passed HIT; the
    other members never touch it. Every path that resolves the pod group
    reaches exactly one pod-local barrier, so every process's
    boot-static decision observes the marker outcome.
    """
    try:
        cfgdict = _sk_config()
        if not cfgdict or not cfgdict.get("static_kernel_cache_enable", False):
            return  # gate off: every rank exits here, no rendezvous needed

        # Collective setup precedes any rank branching (see _pod_group).
        pod_group = _pod_group()

        try:
            if not _is_pod_rank0(pod_group):
                return  # the other members rendezvous in the finally below

            # The marker dies only here — in the pod's rank 0, before its
            # pull. The old shape (kill in every member's entry) raced the
            # rank-0 post-pull rewrite: a slowly spawned sibling deleted the
            # freshly written marker after a pull HIT and the whole pod fell
            # back to full compile (s10 hot boot 2026-10-20). Stale markers
            # cannot arise on a fresh per-boot cache dir anyway, and the HIT
            # landing rebuilds the entire product directory.
            _kill_pool_marker(cfgdict)

            # Only the pod's rank 0 talks to the pool: concurrent store
            # setups race on the Ascend transfer engine's listen port.
            try:
                from static_kernel_cache.static_kernel_cache_store import (
                    BundleLayout,
                    PullResult,
                    bounded_pull,
                )
            except ImportError as exc:
                logger.info(
                    "static_kernel_cache: static_kernel_cache package not importable (%s); "
                    "continuing with full compile (ungated local reuse disabled)",
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
                # landed. Written before the barrier so every process's
                # boot-static decision (post-barrier) observes it.
                (Path(root) / "static_kernel_compile_outputs" / _POOL_MARKER).write_text(
                    f"{fingerprint}\n"
                )
        finally:
            # All group-resolved paths barrier once, exceptions included:
            # the marker outcome is already on disk either way.
            _barrier_once(pod_group)
    except _PodCoordinationUnavailable:
        return  # already logged; every rank exits the same way
    except Exception:
        logger.exception(
            "static_kernel_cache pool pull failed; continuing with full compile "
            "(ungated local reuse disabled)"
        )


# Per-process latch: the capture-end rendezvous runs once per engine
# life. Recapture (/resume on one API server) re-runs capture_model on a
# single process, which must not wait on a rendezvous the siblings will
# never join.
_upload_rendezvous_done = False


def schedule_static_kernel_cache_upload(vllm_config: Any) -> None:
    """Upload hook — never raises.

    Every engine process reaches the capture-end rendezvous on the pod's
    group in its boot thread; once it returns, the pod's compile index is
    final (see module docstring) and the WORLD rank 0 alone spawns the
    consolidation+upload on a daemon thread so ready is not delayed.
    """
    global _upload_rendezvous_done
    try:
        cfgdict = _sk_config()
        if not cfgdict or not cfgdict.get("static_kernel_cache_enable", False):
            return
        if _upload_rendezvous_done:
            logger.debug(
                "static_kernel_cache: upload rendezvous already passed; "
                "skipping recapture-time upload"
            )
            return
        _upload_rendezvous_done = True

        # Same collective-before-branching rule as the pull hook; in
        # practice the pull hook has already built the group.
        pod_group = _pod_group()
        _barrier_once(pod_group)

        if _world_rank() != 0:
            return

        def _upload() -> None:
            try:
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
    except _PodCoordinationUnavailable:
        return  # already logged; pooling abandoned this boot
    except Exception:
        logger.exception("static_kernel_cache pool upload scheduling failed")
