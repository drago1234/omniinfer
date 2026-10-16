# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.

import os
import json
import time
import torch
import queue
import hashlib
import threading
from collections import defaultdict
from contextlib import contextmanager

from vllm.model_executor.models.utils import extract_layer_index
from vllm.v1.core.sched.output import SchedulerOutput, NewRequestData
from vllm.v1.worker.gpu_model_runner import GPUModelRunner
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
    from omni_npu.connector.llmdatadist_connector_v1 import Metadata
    meta: Metadata = scheduler_output.kv_connector_metadata
    return {req_id: params["prefill_req_id"]
        for req_id, params in meta.req_params.items()}


class PrefillReq:

    def __init__(self, new: NewRequestData):
        self.req_id = new.req_id
        self.tok_num = len(new.prompt_token_ids)
        self.blocks = [[idx for idx in group] for group in new.block_ids]
        self.block_set = {idx for blk in self.blocks for idx in blk}
        self.recycled = False
        logger.warning(f"new blocks: {new.block_ids}")

    def _evict(self, idx: int):
        # 该 block id 被滑窗/Mome 回收后复用: 旧占用已作废, 在表中置为 null block(0)
        for blocks in self.blocks:
            for pos, blk in enumerate(blocks):
                if blk == idx:
                    blocks[pos] = 0
                    return

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
                    if idx in self.block_set:
                        # block id 复用 => 旧块已被丢弃(滑窗/Mome 回收), 不是错误
                        self.recycled = True
                        self._evict(idx)
                    self.block_set.add(idx)
                    group.append(idx)
        return self


class DecodeReq(PrefillReq):

    def __init__(self, new: NewRequestData):
        super().__init__(new)
        self.hist_hash: dict[str, tuple] = defaultdict(lambda: (None, None, None))
        self.hashed_blocks: dict[int, list[list[str]]] = {}
        self.hashed_blks: list[int] = []
        self.hashed_batch: torch.Tensor = None
        self.n_mismatch: int = 0

    def _check_blks(self, full: int, shapes: list[list[torch.Tensor]]):
        idxs = []
        for group in self.blocks:
            idxs.extend(group[:min(full, len(group) - 1)])

        hashed_num = len(self.hashed_blks)
        reorg = self.hashed_blks.copy()
        idxs_set = set(idxs)
        if any(idx not in idxs_set for idx in reorg):
            # 有已哈希块被丢弃(滑窗回收): 之前的基线作废, 重新建立
            self.hashed_blks = []
            self.hashed_batch = None
            hashed_num = 0
            reorg = []
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
                active_id = group_blk[min(full, len(group_blk) - 1)]
                for name, layer in group_cache.items():
                    for i, cache in enumerate(layer):
                        if cache.size(1) == pg:
                            yield f"{name}.{i}", cache[active_id]

        for name, blk in active_blocks():
            sect = blk[:rest].contiguous()
            _full, _rest, _sect = self.hist_hash[name]
            if _full == full:
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
            self.hist_hash[name] = (full, rest, sect.cpu())

    def update(self,
        tok_num: int,
        new_blks: tuple[list[int]] | None,
        groups: list[dict[str, tuple[torch.Tensor]]],
        shapes: list[list[torch.Tensor]],
        mtp: int = 0,
        pg: int = 128,
    ):
        self.append_blocks(new_blks)
        if self.recycled:
            # 有块被回收复用: 旧的哈希基线作废, 重新建立
            self.hashed_blks = []
            self.hashed_batch = None
            self.hist_hash = defaultdict(lambda: (None, None, None))
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

            self._decode(req_ids, computed, blocks, news, mtp, remote_reqs)
            yield

    def finish(self):
        # 在 sample_tokens(含 MTP/draft forward, 会写入 lookahead 的 KV) 之后调用,
        # 此时 KV cache 才是完整的, 才能安全地做 req.dump
        if not self.pending or self.shapes is None:
            return
        self.pending = False
        for req in self.reqs.values():
            req.data = req.dump(self.shapes)

    def _decode(self, req_ids, computed, blocks, news, mtp, remote_reqs):
        self.remote_reqs.update(remote_reqs)

        for req_id, req in self.reqs.items():
            if req_id not in req_ids:
                req.finalize()
                self.remote_reqs.pop(req_id, None)

        self.reqs = {req_id: self.reqs[req_id].update(
            tok_num, new_blks, self.groups, self.shapes, mtp,
        ) for req_id, tok_num, new_blks in zip(req_ids, computed, blocks)}

        for new in news:
            req_id = new.req_id
            req = DecodeReq(new)
            self.reqs[req_id] = req
            data = req.dump(self.shapes, self.remote_reqs[req_id])
            self.change.put((req_id, data))

    def _prefill(self, req_ids, blocks, news):
        for req_id, req in self.reqs.items():
            if req_id not in req_ids:
                self.change.put((req_id, req.data))

        self.reqs = {req_id: self.reqs[req_id].append_blocks(new_blks)
            for req_id, new_blks in zip(req_ids, blocks)}

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
