# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""Worker-side NaN/Inf detector driven by module forward hooks.

Lifecycle (all ranks of an engine see the same sequence because the toggle is
delivered through ``/collective_rpc`` on the same broadcast queue as
``execute_model``):

1. ``configure(enable=True, ...)`` registers a forward hook on every
   ``nn.Module`` of the main model. Hooks are dormant until a checked forward.
2. ``begin_step`` / ``end_step`` bracket ``NPUModelRunner._model_forward``.
   Only inside that bracket do hooks inspect tensors, so dummy runs (DP idle
   ranks) and graph captures are never touched.
3. The first module whose floating-point input or output holds NaN/Inf is
   dumped as a self-contained ``.pt`` report; later modules of the same step
   are skipped, since the value propagates.
4. ``end_step`` ORs the verdict across the TP (and PCP) group on the gloo CPU
   group -- raising mid-forward would desynchronise collectives -- so the
   output rank always knows when any peer tripped.
5. ``attach`` stamps the report on the step's ``ModelRunnerOutput``; the
   scheduler patch then fails the batch with ``ModelDegradation``.

Hooks only fire in eager execution. With ``force_eager`` (default) a checked
step sets ``cudagraph_runtime_mode=NONE`` / ``skip_compiled`` so ACL-graph
decode steps run eagerly while the check is on.
"""
import copy as _copy
import dataclasses
import json
import logging
import os
import re
import socket
import time
from collections.abc import Mapping

import torch

logger = logging.getLogger(__name__)

MARKER = "ModelDegradation"
REPORT_ATTR = "omni_nan_inf_report"
DUMP_SUBDIR = "nan_inf"

_TRUE = {"1", "true", "yes", "on", "y", "t"}
_FALSE = {"0", "false", "no", "off", "n", "f", ""}


def parse_bool(value, default=None):
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    text = str(value).strip().lower()
    if text in _TRUE:
        return True
    if text in _FALSE:
        return False
    raise ValueError(f"cannot parse boolean from {value!r}")


def parse_int(value, default=None):
    if value is None or (isinstance(value, str) and not value.strip()):
        return default
    return int(value)


def _group_rank(getter_name):
    try:
        from vllm.distributed import parallel_state as ps
        return getattr(ps, getter_name)().rank_in_group
    except Exception:  # noqa: BLE001
        return None


def collect_ranks(dp_rank=None):
    """Rank coordinates of this process; uninitialised groups degrade to None."""
    global_rank = None
    try:
        if torch.distributed.is_initialized():
            global_rank = torch.distributed.get_rank()
    except Exception:  # noqa: BLE001
        pass
    if global_rank is None:
        try:
            global_rank = parse_int(os.environ.get("RANK"))
        except ValueError:
            global_rank = None
    if dp_rank is None:
        dp_rank = _group_rank("get_dp_group")
    return {
        "global_rank": global_rank,
        "dp_rank": dp_rank,
        "tp_rank": _group_rank("get_tp_group"),
        "ep_rank": _group_rank("get_ep_group"),
        "pp_rank": _group_rank("get_pp_group"),
        "cp_rank": _group_rank("get_pcp_group"),
        "dcp_rank": _group_rank("get_dcp_group"),
    }


def iter_tensors(prefix, obj, depth=0):
    """Yield (name, tensor) for every tensor nested in args/kwargs/outputs."""
    if depth > 4:
        return
    if isinstance(obj, torch.Tensor):
        yield prefix, obj
    elif isinstance(obj, (list, tuple)):
        for i, item in enumerate(obj):
            yield from iter_tensors(f"{prefix}.{i}", item, depth + 1)
    elif isinstance(obj, dict):
        for key, item in obj.items():
            yield from iter_tensors(f"{prefix}.{key}", item, depth + 1)
    elif isinstance(getattr(obj, "tensors", None), dict):  # IntermediateTensors
        yield from iter_tensors(prefix, obj.tensors, depth + 1)


def _checkable(tensor):
    return (tensor.is_floating_point() and tensor.numel() > 0
            and not tensor.is_meta)


def tensor_stats(tensor, max_rows=64):
    """Summary of one tensor; finite_* stats ignore NaN/Inf entries."""
    stats = {}
    if not _checkable(tensor):
        return stats
    t = tensor.detach()
    nan_mask = torch.isnan(t)
    inf_mask = torch.isinf(t)
    finite_mask = ~(nan_mask | inf_mask)
    stats["nan_count"] = int(nan_mask.sum().item())
    stats["posinf_count"] = int((inf_mask & (t > 0)).sum().item())
    stats["neginf_count"] = int((inf_mask & (t < 0)).sum().item())
    stats["has_nan"] = stats["nan_count"] > 0
    stats["has_inf"] = (stats["posinf_count"] + stats["neginf_count"]) > 0
    finite = t[finite_mask].float()
    if finite.numel() > 0:
        stats["finite_min"] = float(finite.min().item())
        stats["finite_max"] = float(finite.max().item())
        stats["finite_mean"] = float(finite.mean().item())
        stats["finite_absmax"] = float(finite.abs().max().item())
    if t.dim() >= 2 and (stats["has_nan"] or stats["has_inf"]):
        bad_rows = (~finite_mask).flatten(1).any(dim=1).nonzero().flatten()
        stats["bad_row_count"] = int(bad_rows.numel())
        stats["bad_rows_head"] = bad_rows[:max_rows].tolist()
    return stats


def _type_name(value):
    value_type = type(value)
    return f"{value_type.__module__}.{value_type.__qualname__}"


def _metadata_fields(value):
    if dataclasses.is_dataclass(value):
        return [field.name for field in dataclasses.fields(value)]
    fields = getattr(value, "__struct_fields__", None)
    if fields is not None:
        return list(fields)
    if hasattr(value, "__dict__"):
        return [name for name in value.__dict__ if not name.startswith("_")]
    return []


def snapshot_metadata(value, depth=0, seen=None):
    """Copy structured forward metadata and tensor values into a PT-safe form."""
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if depth >= 6:
        return {"__type__": _type_name(value)}
    if seen is None:
        seen = set()
    value_id = id(value)
    if value_id in seen:
        return {"__type__": _type_name(value), "__cycle__": True}
    seen.add(value_id)
    try:
        if isinstance(value, Mapping):
            return {
                str(key): snapshot_metadata(item, depth + 1, seen)
                for key, item in value.items()
            }
        if isinstance(value, list):
            return [snapshot_metadata(item, depth + 1, seen) for item in value]
        if isinstance(value, tuple):
            return tuple(snapshot_metadata(item, depth + 1, seen) for item in value)
        fields = _metadata_fields(value)
        if not fields:
            return {"__type__": _type_name(value)}
        result = {"__type__": _type_name(value)}
        for field in fields:
            try:
                result[field] = snapshot_metadata(
                    getattr(value, field), depth + 1, seen)
            except Exception as e:  # noqa: BLE001
                result[field] = {"__error__": type(e).__name__}
        return result
    finally:
        seen.remove(value_id)


def _matching_attention_metadata(module_name, attn_metadata):
    def matches(key):
        key = str(key)
        key_without_suffix = key.removesuffix(".attn")
        return (
            module_name == key
            or module_name == key_without_suffix
            or module_name.startswith(f"{key_without_suffix}.")
            or key_without_suffix.startswith(f"{module_name}.")
        )

    if isinstance(attn_metadata, Mapping):
        return {
            str(key): snapshot_metadata(value)
            for key, value in attn_metadata.items()
            if matches(key)
        }
    if isinstance(attn_metadata, list):
        return [
            _matching_attention_metadata(module_name, item)
            for item in attn_metadata
        ]
    return {}


def module_metadata(module_name, module, forward_context):
    """Return PT-safe metadata associated with the detected module."""
    metadata = {
        "class": _type_name(module),
        "training": module.training,
    }
    if forward_context is not None:
        metadata["attention"] = _matching_attention_metadata(
            module_name, getattr(forward_context, "attn_metadata", None))
    return metadata


_SUMMARY_KEYS = ("step", "global_rank", "dp_rank", "tp_rank", "ep_rank",
                 "pp_rank", "cp_rank", "module", "dump_file")


def summarize(report):
    return {k: report.get(k) for k in _SUMMARY_KEYS}


class NanInfDetector:
    def __init__(self, dp_rank=None):
        self.dp_rank = dp_rank
        self.enabled = False
        self.force_eager = True
        self.max_dumps = 1
        self.dump_dir = None
        self.include = None
        self.exclude = None
        self._handles = []
        self._hooked_modules = 0
        self.contain_exceptional_values = True
        self._active = False
        self._step = None
        self._step_ctx = {}
        self._forward_context = None
        self._hit = None
        self._pending = None
        self._saved_ctx = None
        self._hook_error_logged = False
        self.num_checked_steps = 0
        self.num_skipped_graph_steps = 0
        self.num_detections = 0
        self.num_dumps = 0
        self.last_report = None

    # ------------------------------------------------------------------
    # control plane (/collective_rpc)
    # ------------------------------------------------------------------
    def configure(self, model, enable=None, dump_dir=None, max_dumps=None,
                  force_eager=None, contain_exceptional_values=None,
                  include=None, exclude=None, default_dump_dir=None):
        if not parse_bool(enable, True):
            self.disable()
            return self.status()
        if model is None:
            raise RuntimeError("model is not loaded; cannot install NaN/Inf hooks")
        # parse everything before mutating state so a bad option is a no-op
        force_eager = parse_bool(force_eager, self.force_eager)
        max_dumps = parse_int(max_dumps, self.max_dumps)
        contain_exceptional_values = parse_bool(
            contain_exceptional_values, self.contain_exceptional_values)
        include_re = self.include if include is None else (
            re.compile(include) if include else None)
        exclude_re = self.exclude if exclude is None else (
            re.compile(exclude) if exclude else None)

        self.force_eager = force_eager
        self.max_dumps = max_dumps
        self.contain_exceptional_values = contain_exceptional_values
        self.include, self.exclude = include_re, exclude_re
        if dump_dir:
            self.dump_dir = str(dump_dir)
        elif self.dump_dir is None:
            self.dump_dir = str(default_dump_dir or ".")
        self._remove_hooks()
        self._install_hooks(model)
        self.enabled = True
        self.num_dumps = 0
        logger.warning(
            "omni-nan-inf: enabled on %d modules (force_eager=%s, max_dumps=%s, "
            "contain_exceptional_values=%s, dump_dir=%s)", self._hooked_modules,
            self.force_eager, self.max_dumps, self.contain_exceptional_values,
            self.dump_dir)
        return self.status()

    def disable(self):
        was_enabled = self.enabled
        self.enabled = False
        self._active = False
        self._pending = None
        self._remove_hooks()
        if was_enabled:
            logger.warning("omni-nan-inf: disabled")

    def status(self):
        status = {
            "enabled": self.enabled,
            "hooked_modules": self._hooked_modules,
            "force_eager": self.force_eager,
            "contain_exceptional_values": self.contain_exceptional_values,
            "max_dumps": self.max_dumps,
            "dump_dir": self.dump_dir,
            "include": self.include.pattern if self.include else None,
            "exclude": self.exclude.pattern if self.exclude else None,
            "checked_steps": self.num_checked_steps,
            "skipped_graph_steps": self.num_skipped_graph_steps,
            "detections": self.num_detections,
            "dumps": self.num_dumps,
            "last_report": self.last_report,
        }
        status.update(collect_ranks(self.dp_rank))
        return status

    def _selected(self, name):
        if self.include is not None and not self.include.search(name):
            return False
        return self.exclude is None or not self.exclude.search(name)

    def _install_hooks(self, model):
        root_name = type(model).__name__
        for name, module in model.named_modules():
            full_name = name or root_name
            if self._selected(full_name):
                self._handles.append(module.register_forward_pre_hook(
                    self._make_pre_hook(full_name), with_kwargs=True))
                self._handles.append(module.register_forward_hook(
                    self._make_hook(full_name), with_kwargs=True))
                self._hooked_modules += 1

    def _remove_hooks(self):
        for handle in self._handles:
            try:
                handle.remove()
            except Exception:  # noqa: BLE001
                pass
        self._handles = []
        self._hooked_modules = 0

    # ------------------------------------------------------------------
    # data plane (per forward step)
    # ------------------------------------------------------------------
    def should_check(self, forward_context=None):
        """Rank-consistent decision whether this forward is inspected."""
        if not self.enabled:
            return False
        if self.force_eager or forward_context is None:
            return True
        mode = getattr(forward_context, "cudagraph_runtime_mode", None)
        if mode is not None and getattr(mode, "name", str(mode)) != "NONE":
            # graph replay bypasses Python hooks, and a capture must never
            # see the host syncs they issue.
            self.num_skipped_graph_steps += 1
            return False
        return True

    def begin_step(self, step, forward_context=None, step_ctx=None):
        self._step = step
        self._step_ctx = step_ctx or {}
        self._forward_context = forward_context
        self._hit = None
        self._saved_ctx = None
        if self.force_eager and forward_context is not None:
            self._saved_ctx = (
                getattr(forward_context, "cudagraph_runtime_mode", None),
                getattr(forward_context, "skip_compiled", False),
            )
            try:
                from vllm.config import CUDAGraphMode
                forward_context.cudagraph_runtime_mode = CUDAGraphMode.NONE
            except Exception:  # noqa: BLE001
                pass
            forward_context.skip_compiled = True
        self._active = True

    def end_step(self, forward_context=None):
        """Must be reached by every TP/PCP peer that ran begin_step."""
        self._active = False
        if self._saved_ctx is not None and forward_context is not None:
            (forward_context.cudagraph_runtime_mode,
             forward_context.skip_compiled) = self._saved_ctx
        self._saved_ctx = None
        self.num_checked_steps += 1
        local = summarize(self._hit) if self._hit is not None else None
        self._hit = None
        detections = self._gather(local)
        if detections:
            self.num_detections += 1
            self._pending = {"step": self._step, "detections": detections}
            logger.error("omni-nan-inf: %s at step %s, first hit: %s",
                         MARKER, self._step, detections[0])
        self._forward_context = None
        return self._pending

    @staticmethod
    def _peer_groups():
        groups = []
        try:
            from vllm.distributed import parallel_state as ps
        except Exception:  # noqa: BLE001
            return groups
        for getter in ("get_tp_group", "get_pcp_group"):
            try:
                group = getattr(ps, getter)()
            except Exception:  # noqa: BLE001
                continue
            if group is not None and group.world_size > 1:
                groups.append(group.cpu_group)
        return groups

    def _gather(self, local):
        """OR the verdict over TP/PCP (gloo) and collect per-rank summaries."""
        groups = self._peer_groups()
        if not groups:
            return [local] if local is not None else []
        dist = torch.distributed
        flag = torch.tensor([0 if local is None else 1], dtype=torch.int32)
        for group in groups:
            dist.all_reduce(flag, op=dist.ReduceOp.MAX, group=group)
        if int(flag.item()) == 0:
            return []
        collected = [local] if local is not None else []
        for group in groups:
            gathered = [None] * dist.get_world_size(group=group)
            dist.all_gather_object(gathered, collected, group=group)
            collected = [d for part in gathered for d in (part or [])]
        seen, unique = set(), []
        for d in collected:
            key = json.dumps(d, sort_keys=True, default=str)
            if key not in seen:
                seen.add(key)
                unique.append(d)
        return unique

    def _make_pre_hook(self, name):
        def hook(module, args, kwargs):
            if not self._active:
                return None
            try:
                detected = self._check(name, module, args, kwargs, None)
                if detected and self.contain_exceptional_values:
                    clean_args, args_changed = self._sanitize(args)
                    clean_kwargs, kwargs_changed = self._sanitize(kwargs)
                    if args_changed or kwargs_changed:
                        return clean_args, clean_kwargs
            except Exception as e:  # noqa: BLE001
                self._log_hook_error(name, e)
            return None
        return hook

    def _make_hook(self, name):
        def hook(module, args, kwargs, output):
            if not self._active:
                return None
            try:
                detected = self._check(name, module, args, kwargs, output)
                if detected and self.contain_exceptional_values:
                    clean_output, changed = self._sanitize(output)
                    if changed:
                        return clean_output
            except Exception as e:  # noqa: BLE001
                self._log_hook_error(name, e)
            return None
        return hook

    def _log_hook_error(self, name, error):
        if not self._hook_error_logged:
            self._hook_error_logged = True
            logger.warning("omni-nan-inf: check failed on %s: %r", name, error)

    def _check(self, name, module, args, kwargs, output):
        named = list(iter_tensors("input", args))
        named += list(iter_tensors("input", kwargs))
        if output is not None and not isinstance(output, (tuple, list)):
            output = (output,)
        if output is not None:
            named += list(iter_tensors("output", output))
        named = [(n, t) for n, t in named if _checkable(t)]
        if not named:
            return False
        # One host sync per module call. Continue sanitizing after the first
        # report so rank-local corruption cannot reach a later TP collective.
        flags = torch.stack([~torch.isfinite(t).all() for _, t in named])
        if bool(flags.any().item()):
            if self._hit is None:
                self._hit = self._report(name, module, named)
            return True
        return False

    @classmethod
    def _sanitize(cls, value):
        """Return a non-mutating finite replacement and whether it changed."""
        if isinstance(value, torch.Tensor):
            if not _checkable(value) or bool(torch.isfinite(value).all().item()):
                return value, False
            sanitized = value.clone()
            sanitized.masked_fill_(~torch.isfinite(sanitized), 0)
            return sanitized, True
        if isinstance(value, tuple):
            pairs = [cls._sanitize(item) for item in value]
            return tuple(item for item, _ in pairs), any(changed for _, changed in pairs)
        if isinstance(value, list):
            pairs = [cls._sanitize(item) for item in value]
            return [item for item, _ in pairs], any(changed for _, changed in pairs)
        if isinstance(value, dict):
            pairs = {key: cls._sanitize(item) for key, item in value.items()}
            return ({key: item for key, (item, _) in pairs.items()},
                    any(changed for _, changed in pairs.values()))
        return value, False

    def _report(self, name, module, named_tensors):
        report = {"step": self._step}
        report.update(collect_ranks(self.dp_rank))
        report["module"] = name
        report["module_type"] = type(module).__name__
        report["time"] = time.time()
        report["hostname"] = socket.gethostname()
        report["pid"] = os.getpid()
        report.update(self._step_ctx)
        report["tensors"] = {n: tensor_stats(t) for n, t in named_tensors}
        report["dump_file"] = None
        if self.max_dumps is None or self.max_dumps < 0 or self.num_dumps < self.max_dumps:
            self._write_dump(report, named_tensors, name, module)
        self.last_report = summarize(report)
        return report

    def _write_dump(self, report, named_tensors, module_name, module):
        try:
            out_dir = os.path.join(self.dump_dir or ".", DUMP_SUBDIR)
            os.makedirs(out_dir, exist_ok=True)
            stem = "step{}_grank{}_pid{}_{}".format(
                report["step"], report["global_rank"], report["pid"],
                time.strftime("%Y%m%d-%H%M%S"))
            pt_path = os.path.join(out_dir, stem + ".pt")
            # Keep report tensors out of the in-memory report that crosses
            # worker/scheduler IPC. The persisted report is self-contained.
            dump_report = dict(report)
            dump_report["module_metadata"] = module_metadata(
                module_name, module, self._forward_context)
            dump_report["tensors"] = {
                name: {
                    **report["tensors"][name],
                    "value": tensor.detach().cpu().clone(),
                }
                for name, tensor in named_tensors
            }
            report["dump_file"] = pt_path
            dump_report["dump_file"] = pt_path
            tmp = pt_path + ".tmp"
            torch.save(dump_report, tmp)
            os.replace(tmp, pt_path)
            self.num_dumps += 1
            logger.error("omni-nan-inf: dumped %s (module=%s)", pt_path, report["module"])
        except Exception as e:  # noqa: BLE001
            report["dump_file"] = None
            logger.warning("omni-nan-inf: dump failed: %r", e)

    # ------------------------------------------------------------------
    # propagation to the scheduler
    # ------------------------------------------------------------------
    def attach(self, output):
        """Stamp the pending report on the step's runner output.

        ``None`` (async scheduling defers the output to ``sample_tokens``)
        keeps it pending; ``IntermediateTensors`` (non-last PP stage) has no
        scheduler channel -- the NaN reaches the last stage and trips there.
        """
        report = self._pending
        if report is None or output is None:
            return output
        self._pending = None
        from vllm.v1.outputs import AsyncModelRunnerOutput, ModelRunnerOutput

        def stamp(result):
            if isinstance(result, ModelRunnerOutput):
                # never mutate shared singletons such as EMPTY_MODEL_RUNNER_OUTPUT
                result = _copy.copy(result)
                setattr(result, REPORT_ATTR, report)
            return result

        if isinstance(output, ModelRunnerOutput):
            return stamp(output)
        if isinstance(output, AsyncModelRunnerOutput):
            orig_get_output = output.get_output

            def get_stamped_output():
                return stamp(orig_get_output())

            output.get_output = get_stamped_output
        return output
