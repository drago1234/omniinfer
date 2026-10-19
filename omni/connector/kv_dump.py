# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.

import os
import json
import time
import torch
import queue
import hashlib
import threading
from numbers import Integral
from types import SimpleNamespace
from collections import defaultdict
from contextlib import contextmanager

from vllm.model_executor.models.utils import extract_layer_index
from vllm.v1.core.sched.output import SchedulerOutput, NewRequestData
from vllm.v1.worker.gpu_model_runner import GPUModelRunner
from vllm.v1.worker.cp_utils import get_total_cp_world_size
from vllm.distributed.parallel_state import get_world_group
from vllm.logger import init_logger

from omni_npu import envs

logger = init_logger(__name__)
_kv_dump_path: str = envs.OMNI_KV_DUMP_PATH


def _make_run_id() -> str:
    return str(int(time.time()))


def gen_hash(buf: torch.Tensor):
    if buf.dtype != torch.uint8:
        raise ValueError(f"buf dtype must be uint8, got {buf.dtype}")
    if buf.dim() != 1:
        raise ValueError(f"buf dim must be 1, got {buf.dim()}")

    def tensor_desc(x: torch.Tensor):
        dtype = f"{x.dtype}".replace("torch.", "")
        shape = ",".join(str(i) for i in x.shape)
        return f"{dtype}({shape})"

    kind = f"{tensor_desc(buf)}.blk"
    dat = buf.cpu().numpy().tobytes()
    digest = hashlib.md5(dat).hexdigest()
    return kind, digest


def _diff_stat(a: torch.Tensor, b: torch.Tensor) -> tuple[int, int, int]:
    """返回 (不同字节数, 总字节数, 首个不同位置); 完全相同返回 (0, n, -1)。"""
    af = a.reshape(-1)
    bf = b.reshape(-1)
    if torch.equal(af, bf):
        return 0, af.numel(), -1
    neq = af != bf
    n = int(neq.sum().item())
    first = int(neq.nonzero()[0].item())
    return n, af.numel(), first


def _iter_block_ids(value):
    """Yield block IDs from flat or per-KV-group scheduler data."""
    if value is None:
        return
    if isinstance(value, torch.Tensor):
        value = value.detach().reshape(-1).to(device="cpu").tolist()
    if isinstance(value, (list, tuple, set)):
        for item in value:
            yield from _iter_block_ids(item)
        return
    if isinstance(value, Integral):
        yield int(value)


def _id_set(value) -> set[str]:
    """Scheduler id containers are sets or lists. Ignore mocks and other types."""
    if isinstance(value, (set, list, tuple, frozenset)):
        return {str(item) for item in value}
    return set()


def _physical_block_ids(value, *, positive_only=False) -> set[int]:
    # Connector decode_blocks uses -1 for local prefix hits, whereas dump
    # tables use -block_id for hashed blocks. Do not turn the -1 sentinel
    # into a fresh allocation of physical block 1.
    return {abs(idx) for idx in _iter_block_ids(value)
            if idx > 0 or (idx < 0 and not positive_only)}


def _log_mismatch(title: str, req_id: str, detail_lines: list[str], n_mismatch: int = 0):
    """把哈希不匹配打印成醒目、可定位的多行块(仍然只报错, 不中断)。"""
    bar = "!" * 72
    head = f"!!! KV HASH MISMATCH :: {title} !!!"
    if n_mismatch:
        head += f"  (mismatch #{n_mismatch} for this req)"
    lines = [bar, head, f"  req_id  = {req_id}"]
    lines.extend("  " + line for line in detail_lines)
    lines.append(bar)
    logger.error("\n".join(lines))


def _cpu_ints(value, count: int | None = None) -> list[int]:
    if value is None:
        return []
    if isinstance(value, torch.Tensor):
        value = value.detach().reshape(-1).to(device="cpu").tolist()
    elif hasattr(value, "tolist"):
        value = value.tolist()
    if not isinstance(value, (tuple, list)):
        value = [value]
    values = [int(v) for v in value]
    return values if count is None else values[:count]


def _mome_block_id(cache_indices: torch.Tensor, row: int, block_idx: int) -> int | None:
    if cache_indices.ndim == 1:
        return int(cache_indices[row].item()) if block_idx == 0 else None
    if cache_indices.ndim != 2 or block_idx < 0 or block_idx >= cache_indices.shape[1]:
        return None
    return int(cache_indices[row, block_idx].item())


def _update_mome_block_positions(
    positions: list[int], block_idx: int, first_idx: int, last_idx: int,
    end: int, block_size: int, tail: int, state_len: int,
    position_shift: int = 0,
) -> list[int]:
    if block_idx == last_idx:
        start = end - state_len + 1
        return list(range(start, end + 1))
    if first_idx <= block_idx < last_idx:
        # Block indices use unshifted scheduling positions. The head shift
        # labels the stored activations; it does not change block ownership.
        # A crossed block saves the full convolution tail at its boundary,
        # including history preceding this query's final running-state window.
        # Scheduler positions are zero-based; tracker token labels are
        # one-based. A completed block 0 saves body=[127, 128] and
        # MTP head 0=[128, 129]. This is the boundary write, not the P seed.
        boundary_end = (block_idx + 1) * block_size + position_shift
        tail_positions = list(range(boundary_end - tail + 1, boundary_end + 1))
        return positions[:-tail] + tail_positions
    return positions


def _get_mome_position_state(model_runner) -> dict:
    """Keep position history on the runner, independently of KV tensor dumps."""
    positions = getattr(model_runner, "_mome_cache_positions", None)
    if positions is None:
        positions = {}
        model_runner._mome_cache_positions = positions
    # The runner keeps preempted requests outside input_batch. Preserve their
    # history until the request leaves the worker's live request registry.
    requests = getattr(model_runner, "requests", None)
    live_req_ids = requests if requests is not None else model_runner.input_batch.req_ids
    live_req_ids = {str(req_id) for req_id in live_req_ids if req_id is not None}
    for req_id in list(positions):
        if req_id not in live_req_ids:
            del positions[req_id]
    return positions


def maybe_note_mome_cache(
    model_runner, *, draft: bool = False, draft_head: int = 0, draft_step: int = 0
) -> None:
    """Track and validate the token positions read from MoME cache rows.

    OMNI_KV_DUMP_PATH enables this check independently of the KV dump
    decorators. It runs before a forward while its metadata is available.
    draft_head identifies the cache owner; draft_step identifies the forward
    within the current proposal. Both indices are zero-based.
    """
    if not _kv_dump_path:
        return
    try:
        from vllm.forward_context import get_forward_context

        if model_runner is None:
            return
        mome_positions = _get_mome_position_state(model_runner)
        context = get_forward_context()
        metadata_by_layer = context.attn_metadata
        if isinstance(metadata_by_layer, dict):
            metadata_items = [
                (layer_name, metadata, 0)
                for layer_name, metadata in metadata_by_layer.items()
            ]
        elif isinstance(metadata_by_layer, list):
            # DBO exposes one metadata dictionary per microbatch. Request rows
            # follow the same order, so advance by each microbatch's decode rows.
            metadata_items = []
            row_offset = 0
            for microbatch_metadata in metadata_by_layer:
                if not isinstance(microbatch_metadata, dict):
                    continue
                metadata_items.extend(
                    (layer_name, metadata, row_offset)
                    for layer_name, metadata in microbatch_metadata.items()
                )
                row_offset += max(
                    (int(getattr(metadata, "num_decodes", 0))
                     for metadata in microbatch_metadata.values()),
                    default=0,
                )
        else:
            logger.info(
                "MOME cache position check skipped: unsupported attn_metadata type %s",
                type(metadata_by_layer).__name__,
            )
            return
        req_ids = getattr(model_runner.input_batch, "req_ids", [])
        # Dummy/profile forwards have attention metadata but no scheduled rows.
        # They are expected and should not emit one warning per model layer.
        if not req_ids:
            return
        prompt_tokens = getattr(model_runner.input_batch, "num_prompt_tokens", None)
        prompt_lens = _cpu_ints(prompt_tokens)
        num_spec = int(getattr(model_runner, "num_spec_tokens", 0))
        cache_owner = f"mtp_head_{draft_head}" if draft else "non_mtp"
        cache_source = f"{cache_owner}_step_{draft_step}" if draft else cache_owner
        position_shift = draft_head + 1 if draft else 0
        skip_counts = defaultdict(int)
        logged_rows = 0
        logged_position_reqs = set()
        logged_mismatch_reqs = set()
        logged_untracked_reqs = set()

        for layer_name, metadata, row_offset in metadata_items:
            accepted_tensor = getattr(metadata, "num_accepted_tokens", None)
            cache_indices = getattr(metadata, "cache_indices", None)
            if accepted_tensor is None or cache_indices is None:
                skip_counts["not_mome_metadata"] += 1
                continue
            num_decodes = int(getattr(metadata, "num_decodes", 0))
            n = min(num_decodes, len(req_ids) - row_offset, cache_indices.shape[0])
            if n <= 0:
                skip_counts["no_decode_rows"] += 1
                continue
            block_size = int(getattr(metadata, "B_size", 0))
            query_start = _cpu_ints(getattr(metadata, "query_start_loc", None), n + 1)
            computed = _cpu_ints(getattr(metadata, "num_computed_tokens", None), n)
            accepted = _cpu_ints(accepted_tensor, n)
            metadata_last_computed = _cpu_ints(
                getattr(metadata, "block_idx_last_computed_token", None), n
            )
            if cache_indices.ndim == 1:
                # Without prefix caching, each request has a single state row.
                last_computed = first_scheduled = last_scheduled = [0] * n
            else:
                last_computed = metadata_last_computed
                first_scheduled = _cpu_ints(
                    getattr(metadata, "block_idx_first_scheduled_token", None), n
                )
                last_scheduled = _cpu_ints(
                    getattr(metadata, "block_idx_last_scheduled_token", None), n
                )
            if not (block_size > 0 and len(query_start) >= n + 1
                    and len(computed) >= n and len(accepted) >= n
                    and len(last_computed) >= n and len(first_scheduled) >= n
                    and len(last_scheduled) >= n):
                skip_counts["metadata_fields_incomplete"] += 1
                continue

            # Position tracking depends only on MoME metadata. Each state row
            # holds the convolution history plus speculative slots; derive its
            # logical length from the runner configuration rather than a layer
            # cache tensor. MOME's default router sliding window is 3.
            vllm_config = getattr(model_runner, "vllm_config", None)
            model_config = getattr(vllm_config, "model_config", None)
            hf_config = getattr(model_config, "hf_config", None)
            kernel_width = int(getattr(hf_config, "router_sliding_window", 3) or 3)
            tail = max(kernel_width - 1, 1)
            fake_num_spec = max(num_spec, 1) if getattr(
                vllm_config, "kv_transfer_config", None
            ) is not None else num_spec
            state_len = tail + fake_num_spec

            for row in range(n):
                batch_row = row + row_offset
                req_id = req_ids[batch_row]
                if req_id is None:
                    continue
                req_id = str(req_id)
                qlen = query_start[row + 1] - query_start[row]
                if qlen <= 0:
                    # Padded decode rows do not execute a query or write state.
                    skip_counts["no_query_tokens"] += 1
                    continue
                block_idx = last_computed[row]
                metadata_block_idx = (
                    metadata_last_computed[row] if row < len(metadata_last_computed) else None
                )
                block_id = _mome_block_id(cache_indices, row, block_idx)
                if block_id is None or block_id <= 0 or block_id == getattr(metadata, "pad_slot_id", -1):
                    skip_counts["invalid_cache_block"] += 1
                    continue
                by_layer = mome_positions.setdefault(req_id, {})
                # Target and MTP can use the same logical layer name while
                # owning different cache tensors and position spaces. Steps
                # using the same head share its cache history, while their
                # metadata and diagnostic labels remain distinct.
                position_key = f"{cache_owner}:{layer_name}"
                blocks = by_layer.setdefault(position_key, [])
                initial_cache_state = not blocks
                tracked = next((b for b in blocks if b["block_idx"] == block_idx), None)
                previous_block_id = tracked["block_id"] if tracked is not None else None
                block_mapping_changed = tracked is not None and previous_block_id != block_id
                if block_mapping_changed:
                    blocks.remove(tracked)
                    tracked = None

                prompt_len = prompt_lens[batch_row] if batch_row < len(prompt_lens) else 0
                prompt_block_idx = (
                    (prompt_len - 1) // block_size if cache_indices.ndim != 1 else 0
                )
                # D starts from P's running state in the final prompt block,
                # which need not be block 0. Seed only the initial history;
                # a later missing/remapped block must remain untracked.
                untracked_initial = (
                    tracked is None and initial_cache_state and prompt_len > 0
                )
                in_prompt_block = (
                    computed[row] <= prompt_len and block_idx == prompt_block_idx
                )
                if untracked_initial and in_prompt_block:
                    seed_start = prompt_len - state_len + 1 + position_shift
                    seed = list(range(seed_start, seed_start + state_len))
                    seed = [-1] * max(0, state_len - len(seed)) + seed[-state_len:]
                    tracked = {"block_idx": block_idx, "block_id": block_id, "positions": seed}
                    blocks.append(tracked)
                # Scheduler query positions are [computed, computed + qlen).
                # Convert to the same one-based token labels as the P seed
                # before predicting the state written by this forward.
                end = computed[row] + qlen + position_shift
                # On the first decode recompute, MoME deliberately reads
                # state[-3:-1] (the third- and second-last cache slots).
                # This path is independent of num_accepted_tokens.
                first_decode_recompute = prompt_len > 0 and computed[row] < prompt_len
                draft_step0_at_prompt = (
                    draft and draft_step == 0 and computed[row] == prompt_len
                )
                full_spec_window = (
                    num_spec > 1 and qlen == num_spec + 1 and accepted[row] == tail
                )
                if first_decode_recompute:
                    offset = state_len - 3
                    read_mode = "first_decode_recompute"
                    expected = [
                        prompt_len - 2 + position_shift,
                        prompt_len - 1 + position_shift,
                    ]
                elif draft_step0_at_prompt and full_spec_window:
                    # MTP7's single-head draft step0 is a full-window forward
                    # (query_len=1+num_spec). build_for_drafting synthesizes
                    # accepted=kernel_width-1, so the kernel reads state[1:3]
                    # rather than the ordinary decode tail. This identifies
                    # the read path only: preceding draft forwards may have
                    # changed the state, so the original P-side seed is not an
                    # independent correctness expectation for this read.
                    offset = accepted[row] - 1
                    read_mode = "mtp_draft_step0"
                else:
                    offset = accepted[row] - 1
                    read_mode = "normal_decode"
                if not first_decode_recompute:
                    # Compare with the required history independently of the
                    # selected slots. query_len affects the write window only.
                    expected = [
                        computed[row] - 1 + position_shift,
                        computed[row] + position_shift,
                    ]
                read_positions = None
                read_state_positions = (
                    list(tracked["positions"]) if tracked is not None else None
                )
                read_status = "tracked"
                mismatch_key = (req_id, cache_source)
                if tracked is None:
                    read_status = "block_mapping_changed" if block_mapping_changed else "untracked"
                    if mismatch_key not in logged_untracked_reqs:
                        logger.warning(
                            "MOME READ UNTRACKED source=%s req_id=%s layer=%s "
                            "block_idx_last_computed_token=%s read_block_idx=%s "
                            "previous_block_id=%s read_block_id=%s reason=%s "
                            "read_mode=%s read_offset=%s expected_positions=%s comparison=skipped",
                            cache_source, req_id, layer_name, metadata_block_idx, block_idx,
                            previous_block_id, block_id, read_status, read_mode, offset, expected,
                        )
                        logged_untracked_reqs.add(mismatch_key)
                elif offset < 0 or offset + 1 >= len(tracked["positions"]):
                    read_status = "offset_out_of_range"
                    if mismatch_key not in logged_mismatch_reqs:
                        _log_mismatch("MOME READ", req_id, [
                            f"cache_source = {cache_source}",
                            f"layer   = {layer_name}",
                            f"block_idx_last_computed_token = {metadata_block_idx}",
                            f"block_idx = {block_idx}",
                            f"block_id  = {block_id}",
                            f"read_mode = {read_mode}",
                            f"offset  = {offset}",
                            "actual  = offset out of range",
                            f"expected = {expected}",
                        ])
                        logged_mismatch_reqs.add(mismatch_key)
                else:
                    actual = tracked["positions"][offset:offset + 2]
                    read_positions = actual
                    if actual != expected:
                        read_status = "position_mismatch"
                        if mismatch_key not in logged_mismatch_reqs:
                            _log_mismatch("MOME READ", req_id, [
                                f"cache_source = {cache_source}",
                                f"layer   = {layer_name}",
                                f"block_idx_last_computed_token = {metadata_block_idx}",
                                f"block_idx = {block_idx}",
                                f"block_id  = {block_id}",
                                f"read_mode = {read_mode}",
                                f"offset  = {offset}",
                                f"actual  = {actual}",
                                f"expected = {expected}",
                            ])
                            logged_mismatch_reqs.add(mismatch_key)

                first_idx, last_idx = first_scheduled[row], last_scheduled[row]
                for target_idx in range(max(first_idx, last_idx - 1), last_idx + 1):
                    target_id = _mome_block_id(cache_indices, row, target_idx)
                    if target_id is None or target_id <= 0:
                        continue
                    existing = next((b for b in blocks if b["block_idx"] == target_idx), None)
                    if target_idx == last_idx:
                        new_positions = _update_mome_block_positions(
                            [], target_idx, first_idx, last_idx, end,
                            block_size, tail, state_len,
                            position_shift,
                        )
                        if existing is not None:
                            blocks.remove(existing)
                        blocks.append({"block_idx": target_idx, "block_id": target_id,
                                       "positions": new_positions})
                    elif existing is not None:
                        if existing["block_id"] != target_id:
                            blocks.remove(existing)
                            existing = None
                        if existing is not None:
                            existing["positions"] = _update_mome_block_positions(
                                existing["positions"], target_idx, first_idx,
                                last_idx, end, block_size, tail, state_len,
                                position_shift,
                            )
                blocks.sort(key=lambda b: b["block_idx"])
                if len(blocks) > 2:
                    del blocks[:-2]
                block_positions = [
                    {
                        "block_idx": block["block_idx"],
                        "block_id": block["block_id"],
                        "slots": list(enumerate(block["positions"])),
                    }
                    for block in blocks
                ]
                log_key = (req_id, cache_source)
                if log_key not in logged_position_reqs:
                    logger.info(
                        "MOME CACHE POSITIONS source=%s req_id=%s layer=%s "
                        "block_idx_last_computed_token=%s read_block_idx=%s "
                        "read_block_id=%s computed=%s prompt=%s query_len=%s accepted=%s "
                        "position_shift=%s read_mode=%s read_status=%s read_offset=%s read_positions=%s "
                        "expected_positions=%s read_state_positions=%s "
                        "first_scheduled_block_idx=%s last_scheduled_block_idx=%s "
                        "position_labels=one_based tracker_version=2 "
                        "blocks_phase=predicted_after_forward blocks=%s",
                        cache_source,
                        req_id,
                        layer_name,
                        metadata_block_idx,
                        block_idx,
                        block_id,
                        computed[row],
                        prompt_len,
                        qlen,
                        accepted[row],
                        position_shift,
                        read_mode,
                        read_status,
                        offset,
                        read_positions,
                        expected,
                        read_state_positions,
                        first_idx,
                        last_idx,
                        block_positions,
                    )
                    logged_position_reqs.add(log_key)
                logged_rows += 1
        if logged_rows == 0:
            logger.warning(
                "MOME cache position check produced no rows: attn_metadata=%s "
                "groups=%d requests=%d tracked_requests=%d skips=%s",
                type(metadata_by_layer).__name__,
                len(metadata_by_layer) if hasattr(metadata_by_layer, "__len__") else -1,
                len(req_ids),
                len(mome_positions),
                dict(skip_counts),
            )
    except Exception:
        logger.exception("MoME cache position check failed; continuing forward")


def append_items(path: str, items: dict):
    if not isinstance(items, dict):
        raise TypeError(f"items must be dict, got {type(items)}")
    if not items:
        return
    if not os.path.exists(path):
        open(path, "w").close()
    with open(path, "r+") as f:
        f.seek(0, 2)
        file_size = f.tell()
        if file_size == 0:
            f.write("{\n")
        else:
            f.seek(file_size - 2, 0)
            f.write(",\n")

        prefix = ""
        for k, v in items.items():
            f.write(f'{prefix}"{k}": {v}')
            prefix = ",\n"

        f.write("\n}")
        f.truncate()


def load_layers(vllm_caches: dict | list):
    def parse_layers(layers):
        if isinstance(layers, dict):
            for name, layer in layers.items():
                yield name, extract_layer_index(name), layer
        elif isinstance(layers, (tuple, list)):
            for i, layer in enumerate(layers):
                yield str(i), i, layer
        else:
            raise TypeError("invalid format")

    def parse_layer(name, layer_id, layer):
        if not isinstance(layer, (tuple, list)):
            layer = [layer]
        for i, x in enumerate(layer):
            if not isinstance(x, torch.Tensor):
                raise TypeError(f"{name}.{i} must be torch.Tensor")
            yield f"{name}.{i}", layer_id, x

    block_num = set()

    def to_buffer(x: torch.Tensor):
        if not isinstance(x, torch.Tensor):
            raise TypeError("layer cache must be torch.Tensor")
        if x.dim() not in [2, 3, 4]:
            raise ValueError(f"tensor dim must be 2/3/4, got {x.dim()}")
        block_num.add(x.size(0))
        buf = x.untyped_storage()
        y = x.new_empty(0, dtype=torch.uint8)
        size = x.size(0) * x.stride(0) * x.element_size()
        if x.is_contiguous():
            offset = x.data_ptr() - buf.data_ptr()
            buf = buf[offset:offset + size]
        else:
            if not x[0].is_contiguous():
                raise ValueError("first block must be contiguous")
            if buf.size() != size:
                raise ValueError(f"untrimed storage required: {buf.size()} != {size}")
        y.set_(buf)
        return y.view(x.size(0), -1)

    def sort_by_key(s: dict):
        return [s[k] for k in sorted(s.keys())]

    buffers = sort_by_key({
        name: (i, to_buffer(x))
        for layer in parse_layers(vllm_caches)
        for name, i, x in parse_layer(*layer)
    })
    if len(block_num) != 1:
        raise ValueError(f"inconsistent block_num: {block_num}")

    layers = defaultdict(list)
    for i, buf in buffers:
        layers[i].append(buf)

    shapes = defaultdict(lambda: (list(), list()))
    for i in sorted(layers.keys()):
        for buf in dict(layers)[i]:
            ptrs, bufs = shapes[str(buf.shape)]
            if buf.data_ptr() not in ptrs:
                ptrs.append(buf.data_ptr())
                bufs.append(buf)
    shapes = [shapes[k] for k in sorted(shapes.keys())]
    shapes = [bufs for ptrs, bufs in shapes]
    shapes: list[list[torch.Tensor]]
    return shapes


def parse_remote_reqs(scheduler_output: SchedulerOutput) -> dict:
    meta = getattr(scheduler_output, "kv_connector_metadata", None)
    req_params = getattr(meta, "req_params", None) or {}
    return {
        req_id: params["prefill_req_id"]
        for req_id, params in req_params.items()
        if isinstance(params, dict) and "prefill_req_id" in params
    }


class PrefillReq:

    def __init__(self, new: NewRequestData):
        self.req_id = new.req_id
        self.tok_num = len(new.prompt_token_ids)
        self.blocks = [[idx for idx in group] for group in new.block_ids]
        # A negative ID only marks a hashed/prefix block; ownership is based
        # on the physical slot shared by both signed representations.
        self.block_set = {abs(idx) for blk in self.blocks for idx in blk}
        self.recycled = False
        logger.warning(f"new blocks: {new.block_ids}")

    def _evict(self, idx: int):
        # 该 block id 被滑窗/Mome 回收后复用: 旧占用已作废, 在表中置为 null block(0)
        self.invalidate_blocks([idx])

    def invalidate_blocks(self, block_ids) -> set[int]:
        """Drop stale baselines for physical blocks allocated to another request."""
        invalid_ids = _physical_block_ids(block_ids)
        if not invalid_ids:
            return set()
        invalidated = set()
        for group in self.blocks:
            for pos, blk in enumerate(group):
                physical_id = abs(blk)
                if physical_id in invalid_ids:
                    invalidated.add(physical_id)
                    group[pos] = 0
        if invalidated:
            self.block_set = {abs(blk) for group in self.blocks for blk in group}
        return invalidated

    def dump(self, shapes: list[list[torch.Tensor]], remote_req=None):
        def gen(idx_: int) -> dict:
            idx = abs(idx_)
            attrs = {"block_id": idx}
            if idx == 0:
                return attrs
            if idx_ < 0:
                attrs["hashed"] = True
            for shape in shapes:
                layer = []
                for i, cache in enumerate(shape):
                    kind, digest = gen_hash(cache[idx])
                    layer.append(f"{i}:{digest}")
                attrs[kind] = layer
            return attrs

        t0 = time.monotonic()
        data = {f"group{i}": [gen(idx) for idx in group]
            for i, group in enumerate(self.blocks)}
        dt = int((time.monotonic() - t0) * 1000)
        logger.info("[%dms] hash blocks", dt)
        if remote_req is not None:
            data["remote_req_id"] = remote_req
        return data

    def append_blocks(self, new_blks: tuple[list[int]] | None):
        self.recycled = False
        if new_blks is not None:
            logger.warning(f"new blocks: {new_blks}")
            for group, blocks in zip(self.blocks, new_blks):
                for idx in blocks:
                    physical_id = abs(idx)
                    if physical_id and physical_id in self.block_set:
                        # block id 复用 => 旧块已被丢弃(滑窗/Mome 回收), 不是错误
                        self.recycled = True
                        self._evict(physical_id)
                    self.block_set.add(physical_id)
                    group.append(idx)
        return self


class DecodeReq(PrefillReq):

    def __init__(self, new: NewRequestData):
        super().__init__(new)
        self.hist_hash: dict[str, tuple] = defaultdict(lambda: (None, None, None))
        self.hist_blocks: dict[str, int] = {}
        self.hashed_blocks: dict[int, list[list[str]]] = {}
        self.hashed_blks: list[int] = []
        self.hashed_batch: torch.Tensor = None
        self.n_mismatch: int = 0

    @classmethod
    def from_block_table(cls, req_id: str, block_ids, tok_num: int):
        """Rebuild a resumed request from the scheduler's full block table."""
        new = SimpleNamespace(
            req_id=req_id,
            prompt_token_ids=[0] * max(int(tok_num), 0),
            block_ids=block_ids or (),
        )
        return cls(new)

    def replace_blocks(self, block_ids):
        """Preemption hands back a full table, and the old KV baseline is stale."""
        if block_ids is not None:
            self.blocks = [[idx for idx in group] for group in block_ids]
            self.block_set = {abs(idx) for group in self.blocks for idx in group}
        self.recycled = False
        self.hashed_blks = []
        self.hashed_batch = None
        self.hist_hash = defaultdict(lambda: (None, None, None))
        self.hist_blocks = {}

    def invalidate_blocks(self, block_ids) -> set[int]:
        invalidated = super().invalidate_blocks(block_ids)
        if invalidated:
            self._retain_block_baselines(set(self.hashed_blks) - invalidated)
            for name, block_id in list(self.hist_blocks.items()):
                if block_id in invalidated:
                    self.hist_blocks.pop(name)
                    self.hist_hash.pop(name, None)
        return invalidated

    def _retain_block_baselines(self, valid_ids: set[int]):
        # Snapshots are already on CPU. Filter only retired rows so a real
        # overwrite of a still-owned block remains detectable in this step.
        keep = [i for i, idx in enumerate(self.hashed_blks) if idx in valid_ids]
        if len(keep) == len(self.hashed_blks):
            return
        self.hashed_blks = [self.hashed_blks[i] for i in keep]
        if not keep:
            self.hashed_batch = None
        elif self.hashed_batch is not None:
            self.hashed_batch = [
                [cmp[keep] if cmp is not None else None for cmp in shape]
                for shape in self.hashed_batch
            ]

    def _check_blks(self, full: int, shapes: list[list[torch.Tensor]]):
        idxs = []
        for group in self.blocks:
            idxs.extend(abs(idx) for idx in group[:min(full, len(group) - 1)]
                        if idx != 0)

        self._retain_block_baselines(set(idxs))
        hashed_num = len(self.hashed_blks)
        reorg = self.hashed_blks.copy()
        seen = set(reorg)
        for idx in idxs:
            if idx not in seen:
                seen.add(idx)
                reorg.append(idx)

        idx_ = shapes[0][0].new_tensor(reorg, dtype=torch.int)

        if self.hashed_batch is None:
            self.hashed_batch = [[None for cache in shape] for shape in shapes]

        batch = []
        for si, (shape, prev) in enumerate(zip(shapes, self.hashed_batch)):
            layer_out = []
            for ci, (cache, cmp) in enumerate(zip(shape, prev)):
                blks = cache[idx_].contiguous()
                if cmp is not None:
                    a = blks[:hashed_num]
                    b = cmp.to(device=a.device)
                    if not torch.equal(a, b):
                        bad = []
                        for j in range(hashed_num):
                            n, total, first = _diff_stat(a[j], b[j])
                            if n:
                                bad.append((reorg[j], n, total, first))
                        self.n_mismatch += 1
                        detail = [
                            f"kind    = block",
                            f"full    = {full}",
                            f"cache   = shapes[{si}].caches[{ci}]",
                            f"bad     = {len(bad)}/{hashed_num} blocks",
                        ]
                        for blk_id, n, total, first in bad[:16]:
                            detail.append(
                                f"  block {blk_id}: {n}/{total} bytes differ, "
                                f"first_off={first}")
                        if len(bad) > 16:
                            detail.append(f"  ... {len(bad) - 16} more blocks")
                        detail.append("action  = update hash baseline")
                        _log_mismatch("DECODE BLOCK", self.req_id, detail,
                                      self.n_mismatch)
                layer_out.append(blks.cpu())
            batch.append(layer_out)

        self.hashed_batch = batch
        self.hashed_blks = reorg

    def _check_toks(self, full, rest, groups: list[dict[str, tuple[torch.Tensor]]], pg):
        if rest <= 0:
            return

        def active_blocks():
            for group_cache, group_blk in zip(groups, self.blocks):
                if not group_blk:
                    continue
                active_id = abs(group_blk[min(full, len(group_blk) - 1)])
                for name, layer in group_cache.items():
                    for i, cache in enumerate(layer):
                        if cache.size(1) == pg:
                            key = f"{name}.{i}"
                            if active_id == 0:
                                self.hist_hash.pop(key, None)
                                self.hist_blocks.pop(key, None)
                                continue
                            yield key, active_id, cache[active_id]

        for name, active_id, blk in active_blocks():
            sect = blk[:rest].contiguous()
            _full, _rest, _sect = self.hist_hash[name]
            if _full == full and self.hist_blocks.get(name) == active_id:
                if _rest >= rest:
                    # 结构性异常(不满足增长), 不再中断: 报错并重置该层基线
                    self.n_mismatch += 1
                    _log_mismatch("DECODE TOKEN (rest not grow)", self.req_id, [
                        f"kind    = token",
                        f"layer   = {name}",
                        f"full    = {full}",
                        f"rest    = {rest} (prev {_rest}, not growing)",
                        "action  = reset hash baseline",
                    ], self.n_mismatch)
                else:
                    a = sect[:_rest]
                    b = _sect.to(device=a.device)
                    if not torch.equal(a, b):
                        # 哈希不匹配不再中断: 报错并写入新哈希
                        n, total, first = _diff_stat(a, b)
                        self.n_mismatch += 1
                        _log_mismatch("DECODE TOKEN", self.req_id, [
                            f"kind    = token",
                            f"layer   = {name}",
                            f"full    = {full}",
                            f"rest    = {rest} (prev {_rest})",
                            f"diff    = {n}/{total} bytes differ, first_off={first}",
                            "action  = update hash baseline",
                        ], self.n_mismatch)
            # A snapshot must own its storage, including when the cache is
            # already on CPU. On NPU this is the same device-to-host copy.
            self.hist_hash[name] = (full, rest, sect.to(device="cpu", copy=True))
            self.hist_blocks[name] = active_id

    def update(self,
        tok_num: int,
        new_blks: tuple[list[int]] | None,
        groups: list[dict[str, tuple[torch.Tensor]]],
        shapes: list[list[torch.Tensor]],
        mtp: int = 0,
        pg: int = 128,
    ):
        self.append_blocks(new_blks)
        self.tok_num = tok_num
        stable_tok = tok_num - mtp
        full = stable_tok // pg
        rest = stable_tok % pg

        t0 = time.monotonic()
        self._check_blks(full, shapes)
        dt = int((time.monotonic() - t0) * 1000)
        logger.info("[%dms] check_blks ok: blocks=%d", dt, full)

        t0 = time.monotonic()
        self._check_toks(full, rest, groups, pg)
        dt = int((time.monotonic() - t0) * 1000)
        logger.info("[%dms] check_toks ok: blocks=%d", dt, full)
        return self

    def finalize(self):
        self.hashed_batch = None


class Dump:

    def __init__(self, path: str):
        self.shapes: list[list[torch.Tensor]] = None  # [types, layers]
        self.groups: list[dict[str, tuple[torch.Tensor]]] = None  # [group, layers, caches]
        self.reqs: dict[str, DecodeReq | PrefillReq] = {}
        self.remote_reqs: dict = {}
        self.missing_cached: set[str] = set()
        self.pending = False

        run_id = _make_run_id()
        self.path = f"{path}/{run_id}/"
        self.rank = None
        self.change = queue.Queue()
        threading.Thread(target=self._task_save_json, daemon=True).start()

    def _task_save_json(self):
        while self.rank is None:
            time.sleep(1.0)

        outfile = f"rank{self.rank}.json"
        os.makedirs(self.path, exist_ok=True)

        while True:
            time.sleep(1.0)
            if self.change.empty():
                continue
            items = {}
            while not self.change.empty():
                req_id, data = self.change.get()
                t0 = time.monotonic()
                items[req_id] = json.dumps(data, indent=2)
                dt = int((time.monotonic() - t0) * 1000)
                logger.info("[%dms] save json: %s", dt, self.path + outfile)
            append_items(self.path + outfile, items)

    def _register(self, model_runner: GPUModelRunner):
        if self.shapes is not None:
            return
        layers = model_runner.compilation_config.static_forward_context
        groups = model_runner.kv_cache_config.kv_cache_groups

        def validate(x):
            if isinstance(x, tuple):
                return x
            if not isinstance(x, list):
                raise TypeError(f"kv_cache must be list, got {type(x)}")
            if not isinstance(x[0], tuple):
                raise TypeError(f"kv_cache[0] must be tuple, got {type(x[0])}")
            return x[0]

        self.groups = [{
            name: validate(layers[name].kv_cache)
            for name in group.layer_names}
            for group in groups]

        flatten = {k: v for group in self.groups for k, v in group.items()}
        self.shapes = load_layers(flatten)

    @contextmanager
    def update(self,
        model_runner: GPUModelRunner,
        scheduler_output: SchedulerOutput,
    ):
        self.rank = get_world_group().rank
        self._register(model_runner)

        role = {"kv_producer": True, "kv_consumer": False}
        cfg = model_runner.vllm_config.kv_transfer_config
        kv_role = cfg.kv_role
        if kv_role not in role:
            raise ValueError(f"unsupported kv_role: {kv_role}")
        is_prefill = role[kv_role]
        # prefill 的 dump 要等 sample_tokens(含 MTP/draft forward) 之后再取, 见 Dump.finish()
        self.pending = is_prefill

        news = scheduler_output.scheduled_new_reqs
        reqs = scheduler_output.scheduled_cached_reqs
        req_ids = reqs.req_ids
        blocks = reqs.new_block_ids or [None] * len(req_ids)

        if is_prefill:
            self._prefill(req_ids, blocks, news)
            yield
        else:
            computed = reqs.num_computed_tokens
            mtp = model_runner.num_spec_tokens
            remote_reqs = parse_remote_reqs(scheduler_output)

            new_block_ids_to_zero = getattr(
                scheduler_output, "new_block_ids_to_zero", None
            )
            allocated_blocks = set()
            # PD allocation metadata arrives before scheduled_new_reqs when
            # KV is loaded asynchronously. Positive decode_blocks are fresh
            # destinations; -1 marks a local prefix-cache hit in the connector.
            meta = getattr(scheduler_output, "kv_connector_metadata", None)
            req_params = getattr(meta, "req_params", None) or {}
            for params in req_params.values():
                allocated_blocks.update(_physical_block_ids(
                    params.get("decode_blocks"), positive_only=True,
                ))
            cp_world_size = get_total_cp_world_size()
            block_sizes = [
                group.kv_cache_spec.block_size * cp_world_size
                for group in model_runner.kv_cache_config.kv_cache_groups
            ]
            self._decode(
                req_ids, computed, blocks, news, mtp, remote_reqs,
                new_block_ids_to_zero, allocated_blocks,
                getattr(reqs, "resumed_req_ids", ()),
                block_sizes=block_sizes,
                finished_req_ids=getattr(
                    scheduler_output, "finished_req_ids", ()
                ),
            )
            yield

    def finish(self):
        # 在 sample_tokens(含 MTP/draft forward, 会写入 lookahead 的 KV) 之后调用,
        # 此时 KV cache 才是完整的, 才能安全地做 req.dump
        if not self.pending or self.shapes is None:
            return
        self.pending = False
        for req in self.reqs.values():
            req.data = req.dump(self.shapes)

    def _decode(
        self, req_ids, computed, blocks, news, mtp, remote_reqs,
        new_block_ids_to_zero=None, allocated_blocks=None, resumed_req_ids=(),
        block_sizes=None, finished_req_ids=(),
    ):
        self.remote_reqs.update(remote_reqs)
        resumed_ids = _id_set(resumed_req_ids)
        finished_ids = _id_set(finished_req_ids)

        # The checker runs before the worker registers scheduled_new_reqs.
        # In a sliding-window/pool-reuse step, a fresh request can therefore
        # own a physical block that is still present in an old request's
        # table. Invalidate that old baseline before comparing its cache.
        reused_ids = _physical_block_ids(new_block_ids_to_zero)
        reused_ids.update(_physical_block_ids(allocated_blocks))
        for req_id, new_blks in zip(req_ids, blocks):
            # Running requests carry allocation deltas. Resumed requests carry
            # full tables that can include shared prefix blocks.
            if req_id not in resumed_ids:
                reused_ids.update(_physical_block_ids(new_blks, positive_only=True))
        for new in news:
            num_computed = getattr(new, "num_computed_tokens", 0)
            if num_computed == 0:
                reused_ids.update(_physical_block_ids(new.block_ids, positive_only=True))
            else:
                # Async PD metadata only covers the initial load allocation.
                # The first schedule can add lookahead blocks even though PD
                # tokens already count as computed. Those slots are private;
                # retain the computed prefix (including a partial imported
                # block) and retire only the newly allocated suffix.
                for group, block_size in zip(new.block_ids, block_sizes or ()):
                    prefix_blocks = (num_computed + block_size - 1) // block_size
                    reused_ids.update(_physical_block_ids(
                        group[prefix_blocks:], positive_only=True,
                    ))
        if reused_ids:
            for req in self.reqs.values():
                invalidated = req.invalidate_blocks(reused_ids)
                if invalidated:
                    logger.info(
                        "invalidate stale KV dump baseline req_id=%s blocks=%s",
                        req.req_id, sorted(invalidated),
                    )

        # vLLM keeps a running request that misses a step, and sends it back as
        # a cached request. Dropping it here makes the next step KeyError.
        # Only finished requests leave the table.
        for req_id in finished_ids:
            req = self.reqs.pop(req_id, None)
            if req is not None:
                req.finalize()
            self.remote_reqs.pop(req_id, None)

        scheduled = {}
        for req_id, tok_num, new_blks in zip(req_ids, computed, blocks):
            if req_id in resumed_ids:
                scheduled[req_id] = self._resume_decode_req(
                    req_id, tok_num, new_blks, mtp,
                )
                continue
            req = self.reqs.get(req_id)
            if req is None:
                if req_id not in self.missing_cached:
                    self.missing_cached.add(req_id)
                    logger.error(
                        "kv dump missing cached request req_id=%s; skip check",
                        req_id,
                    )
                continue
            self.missing_cached.discard(req_id)
            scheduled[req_id] = req.update(
                tok_num, new_blks, self.groups, self.shapes, mtp,
            )
        for req_id, req in self.reqs.items():
            if req_id not in scheduled:
                scheduled[req_id] = req
        self.reqs = scheduled

        for new in news:
            req_id = new.req_id
            self.missing_cached.discard(req_id)
            req = DecodeReq(new)
            self.reqs[req_id] = req
            data = req.dump(self.shapes, self.remote_reqs.get(req_id))
            self.change.put((req_id, data))

    def _resume_decode_req(self, req_id, tok_num, new_blks, mtp):
        self.missing_cached.discard(req_id)
        req = self.reqs.get(req_id)
        if req is None:
            req = DecodeReq.from_block_table(req_id, new_blks, tok_num)
        else:
            req.replace_blocks(new_blks)
        return req.update(tok_num, None, self.groups, self.shapes, mtp)

    def _prefill(self, req_ids, blocks, news):
        scheduled_ids = set(req_ids)
        for req_id, req in self.reqs.items():
            if req_id not in scheduled_ids and hasattr(req, "data"):
                self.change.put((req_id, req.data))

        scheduled = {}
        for req_id, new_blks in zip(req_ids, blocks):
            req = self.reqs.get(req_id)
            if req is None:
                continue
            scheduled[req_id] = req.append_blocks(new_blks)
        self.reqs = scheduled

        for new in news:
            self.reqs[new.req_id] = PrefillReq(new)


def _get_instance(name: str = "dump_prefill") -> Dump:
    if not hasattr(maybe_dump_kv, name):
        setattr(maybe_dump_kv, name, Dump(_kv_dump_path))
    return getattr(maybe_dump_kv, name)


def maybe_dump_kv(stage: str):
    if stage not in ("update", "finish"):
        raise ValueError(f"unknown stage: {stage!r}, expected 'update' or 'finish'")

    def decorator(fn):
        if stage == "update":
            # 挂在 execute_model 上: 进入前 _prefill/_decode 记账, 退出后交还
            def wrapper(
                self: GPUModelRunner,
                scheduler_output: SchedulerOutput,
                intermediate_tensors=None,
            ):
                if not _kv_dump_path:
                    return fn(self, scheduler_output, intermediate_tensors)
                with _get_instance().update(self, scheduler_output):
                    return fn(self, scheduler_output, intermediate_tensors)
            return wrapper

        # stage == "finish": 挂在 sample_tokens (MTP/draft forward) 之后
        def wrapper(self, *args, **kwargs):
            ret = fn(self, *args, **kwargs)
            if _kv_dump_path and hasattr(maybe_dump_kv, "dump_prefill"):
                getattr(maybe_dump_kv, "dump_prefill").finish()
            return ret
        return wrapper

    return decorator
