# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""Compare Megatron training dumps with vLLM accuracy-alignment captures."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import torch

MHC_LAYER_MODULES = (
    "input_layernorm",
    "self_attention",
    "post_self_attn_layernorm",
    "pre_mlp_layernorm",
    "mlp",
    "post_mlp_layernorm",
    "block_post_layernorm",
)
ATTN_COMMON_MODULES = (
    "self_attention.linear_qkv",
    "self_attention.q_layernorm",
    "self_attention.linear_qb",
    "self_attention.k_layernorm",
    "self_attention.core_attention",
    "self_attention.linear_proj",
)
DSA_MODULES = (
    "self_attention.qa_conv",
    "self_attention.compresskv_conv",
    "self_attention.index_linear_qb",
    "self_attention.index_linear_k",
    "self_attention.index_k_layernorm",
    "self_attention.linear_merge_weight",
)
MLA_MODULES = ("self_attention.linear_kvb",)
DENSE_MLP_MODULES = ("mlp.linear_fc1", "mlp.linear_fc2")
MOE_MODULES = (
    "mlp.router",
    "mlp.experts",
    "mlp.experts.shared_expert",
    "mlp.experts.shared_expert.linear_fc1",
    "mlp.experts.shared_expert.linear_fc2",
)
MODEL_MODULES = (
    "embedding.word_embeddings",
    "lm_head",
    "rotary_pos_emb.cos",
    "rotary_pos_emb.sin",
)
MHC_CUSTOM_MODULES = (
    "pangu_transformer_hc_pre_out",
    "pangu_transformer_hc_post_out",
    "pangu_transformer_mlp_hc_pre_out",
    "pangu_transformer_mlp",
    "pangu_transformer_mlp_hc_post_out",
    "pangu_transformer_o_conv",
)
DSA_CUSTOM_MODULES = (
    "dsa_attention_index_query_all_heads",
    "dsa_attention_merge_weight_all_heads",
    "dsa_attention_topk_indices",
)
MOE_GATE_MODULES = (
    "moe_gate_output",
    "moe_router_scores",
    "moe_router_indices",
)
APPLY_EXPERTS_MODULES = ("gmm_experts_output",)
DEFAULT_DSA_LAYERS = tuple(range(0, 46, 3))
EXPECTED_SKIPS = {
    "self_attention.k_layernorm": (
        "kv_a_layernorm.weight is fused into the NPU cache operator"
    ),
    "self_attention.core_attention": "function-style attention op has no hook",
    "self_attention.linear_kvb": "absorbing attention may fuse kv_b_proj",
}
MODULE_SETS = {
    "mhc": MHC_LAYER_MODULES,
    "attn": tuple(dict.fromkeys(ATTN_COMMON_MODULES + DSA_MODULES + MLA_MODULES)),
    "dsa": tuple(
        dict.fromkeys(
            MHC_LAYER_MODULES + ATTN_COMMON_MODULES + DSA_MODULES + DENSE_MLP_MODULES
        )
    ),
    "mla": tuple(
        dict.fromkeys(
            MHC_LAYER_MODULES + ATTN_COMMON_MODULES + MLA_MODULES + DENSE_MLP_MODULES
        )
    ),
    "model": MODEL_MODULES,
    "moe": tuple(dict.fromkeys(DENSE_MLP_MODULES + MOE_MODULES)),
    "mhc_custom": MHC_CUSTOM_MODULES,
    "dsa_custom": DSA_CUSTOM_MODULES,
    "moe_gate": MOE_GATE_MODULES,
    "apply_experts": APPLY_EXPERTS_MODULES,
}
MODULE_SETS["all"] = tuple(
    dict.fromkeys(module for values in MODULE_SETS.values() for module in values)
)


@dataclass
class CompareResult:
    layer_idx: int | None
    module: str
    status: str
    message: str = ""
    # Keep the original tool's report schema so existing report consumers do
    # not need to be rewritten.  left/right are retained as explicit aliases
    # for pure-real comparisons, where neither side is a training dump.
    train_path: str = ""
    infer_path: str = ""
    seq_len: int = 0
    shapes_match: bool = False
    left_path: str = ""
    right_path: str = ""
    left_shape: list[int] = field(default_factory=list)
    right_shape: list[int] = field(default_factory=list)
    all_close: bool = False
    max_abs_diff: float = 0.0
    mean_abs_diff: float = 0.0
    max_rel_diff_percent: float = 0.0
    avg_rel_diff_percent: float = 0.0
    mse: float = 0.0
    corr_coef: float = float("nan")
    error_distribution: dict[str, int] = field(default_factory=dict)


def _primary_tensor(value: Any, key: str = "output.0") -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        return value
    if isinstance(value, dict):
        tensors = value.get("tensors")
        if isinstance(tensors, dict):
            selected = tensors.get(key)
            if isinstance(selected, torch.Tensor):
                return selected
        for item in value.values():
            try:
                return _primary_tensor(item, key)
            except (TypeError, KeyError):
                continue
    if isinstance(value, (tuple, list)):
        output_index = 0
        match = re.fullmatch(r"output\.(\d+)", key)
        if match:
            output_index = int(match.group(1))
        if output_index < len(value) and isinstance(value[output_index], torch.Tensor):
            return value[output_index]
    raise TypeError(f"payload does not contain tensor key {key!r}")


def load_tensor(path: Path, key: str = "output.0") -> torch.Tensor:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    return _primary_tensor(payload, key).detach().cpu()


def _squeeze_middle(value: torch.Tensor) -> torch.Tensor:
    tensor = value
    while tensor.ndim >= 3:
        for dim in range(1, tensor.ndim - 1):
            if tensor.shape[dim] == 1:
                tensor = tensor.squeeze(dim)
                break
        else:
            break
    return tensor


def _load_training_shards(
    path: Path,
    key: str,
    infer_last_dim: int | None,
) -> torch.Tensor:
    tensor = load_tensor(path, key)
    if infer_last_dim is None or tensor.ndim == 0:
        return tensor
    if _is_ncl_training_tensor(tensor, infer_last_dim):
        return tensor.squeeze(0).transpose(0, 1).contiguous()
    if tensor.shape[-1] == infer_last_dim:
        return tensor
    match = re.match(r"(.+_)rank0_(\d+)$", path.stem)
    if match is None:
        return tensor
    prefix, call = match.groups()
    shards = [tensor]
    for rank in range(1, 64):
        shard_path = path.with_name(f"{prefix}rank{rank}_{call}{path.suffix}")
        if not shard_path.is_file():
            break
        shards.append(load_tensor(shard_path, key))
        if sum(item.shape[-1] for item in shards) >= infer_last_dim:
            break
    return torch.cat(shards, dim=-1) if len(shards) > 1 else tensor


def _is_ncl_training_tensor(tensor: torch.Tensor, infer_last_dim: int) -> bool:
    if tensor.ndim != 3 or tensor.shape[0] != 1:
        return False
    return tensor.shape[1] == infer_last_dim and tensor.shape[2] != infer_last_dim


def normalize_training_tensor(
    path: Path,
    *,
    key: str,
    role: str,
    mock_seq_len: int,
    infer: torch.Tensor,
) -> torch.Tensor:
    infer_last_dim = infer.shape[-1] if infer.ndim else None
    tensor = _load_training_shards(path, key, infer_last_dim)
    tensor = _squeeze_middle(tensor)
    if role == "D":
        if tensor.ndim == 0 or tensor.shape[0] < mock_seq_len:
            raise ValueError(
                f"training sequence shorter than decode index {mock_seq_len - 1}"
            )
        tensor = tensor.narrow(0, mock_seq_len - 1, 1)
    elif tensor.ndim:
        if tensor.shape[0] < mock_seq_len:
            raise ValueError(
                f"training sequence shorter than prefill length {mock_seq_len}"
            )
        tensor = tensor[:mock_seq_len]
    return tensor


def compare_tensors(
    left: torch.Tensor,
    right: torch.Tensor,
    *,
    rtol: float,
    atol: float,
) -> tuple[bool, dict[str, Any]]:
    lhs = left.to(torch.float32)
    rhs = right.to(torch.float32)
    metrics: dict[str, Any] = {
        "left_shape": list(lhs.shape),
        "right_shape": list(rhs.shape),
        "shapes_match": lhs.shape == rhs.shape,
        "all_close": False,
        "max_abs_diff": 0.0,
        "mean_abs_diff": 0.0,
        "max_rel_diff_percent": 0.0,
        "avg_rel_diff_percent": 0.0,
        "mse": 0.0,
        "corr_coef": float("nan"),
        "error_distribution": {},
    }
    if not metrics["shapes_match"]:
        return False, metrics
    if lhs.numel() == 0:
        metrics["all_close"] = True
        return True, metrics
    diff = (lhs - rhs).abs()
    metrics["max_abs_diff"] = float(diff.max())
    metrics["mean_abs_diff"] = float(diff.mean())
    metrics["mse"] = float((diff * diff).mean())
    # Match the imported tool: values whose training/reference magnitude is at
    # or below atol do not contribute to relative-error statistics.  Clamping
    # the denominator would report large, misleading percentages around zero.
    rel = torch.zeros_like(diff)
    rel_mask = lhs.abs() > atol
    if rel_mask.any():
        rel[rel_mask] = diff[rel_mask] / lhs.abs()[rel_mask] * 100
    metrics["max_rel_diff_percent"] = float(rel.max())
    metrics["avg_rel_diff_percent"] = float(rel.mean())
    buckets = torch.bucketize(rel.flatten(), torch.tensor([0.1, 1.0, 5.0, 10.0]))
    counts = torch.bincount(buckets, minlength=5).tolist()
    labels = ("0-0.1%", "0.1-1%", "1-5%", "5-10%", ">10%")
    metrics["error_distribution"] = dict(zip(labels, map(int, counts)))
    if lhs.numel() > 1 and float(lhs.std()) > 0 and float(rhs.std()) > 0:
        metrics["corr_coef"] = float(
            torch.corrcoef(torch.stack((lhs.flatten(), rhs.flatten())))[0, 1]
        )
    passed = metrics["max_abs_diff"] <= atol + rtol * float(rhs.abs().max())
    metrics["all_close"] = passed
    return passed, metrics


def _stage_dir(root: Path, role: str) -> Path:
    stage = "prefill" if role == "P" else "decode"
    candidate = root / stage
    return candidate if candidate.is_dir() else root


def _is_custom(module: str) -> bool:
    return module in MHC_CUSTOM_MODULES + DSA_CUSTOM_MODULES + MOE_GATE_MODULES


def _infer_stem(layer: int | None, module: str) -> str:
    if layer is None:
        return module
    if _is_custom(module) or module == "gmm_experts_output":
        return f"{module}_{layer + 1}"
    return f"decoder.layers.{layer}.{module}"


def infer_candidates(
    infer_root: Path,
    *,
    layer: int | None,
    module: str,
    mode: str,
    role: str,
    rank: int,
) -> list[Path]:
    root = _stage_dir(infer_root, role)
    stage = "prefill" if role == "P" else "decode"
    stem = _infer_stem(layer, module)
    current = root / f"{stem}_{stage}_{mode}_rank{rank}.pt"
    legacy = root / (
        f"{stem}_train_ref_rank{rank}.pt"
        if mode == "train_ref"
        else f"{stem}_rank{rank}.pt"
    )
    candidates = [current, legacy]
    if layer is not None and _is_custom(module):
        lid = layer + 1
        candidates.extend(
            [
                root / f"{module}_{mode}_{lid}_rank{rank}.pt",
                root / f"{module}_{lid}_{mode}_rank{rank}.pt",
            ]
        )
    return candidates


def _first_file(paths: list[Path]) -> Path:
    for path in paths:
        if path.is_file():
            return path
    return paths[0]


def training_path(
    train_root: Path,
    custom_root: Path | None,
    *,
    layer: int | None,
    module: str,
    layer_prefix: str,
    model_prefix: str,
    lm_head_name: str,
) -> tuple[Path, str]:
    if layer is not None and module == "pangu_transformer_o_conv":
        return (
            train_root
            / (f"{layer_prefix}.{layer}.self_attention.linear_proj_forward_rank0_0.pt"),
            "input.0",
        )
    if layer is not None and module == "gmm_experts_output":
        root = custom_root or train_root
        return root / f"moe_experts_up_fc2_output_{layer + 1}_rank0.pt", "output.0"
    if layer is not None and _is_custom(module):
        root = custom_root or train_root
        return root / f"{module}_{layer + 1}_rank0.pt", "output.0"
    if module == "embedding.word_embeddings":
        return (
            train_root / f"{layer_prefix}.0.input_layernorm_forward_rank0_0.pt",
            "input.0",
        )
    if module.startswith("rotary_pos_emb."):
        output = "output.1" if module.endswith("sin") else "output.0"
        return train_root / f"{model_prefix}.rotary_pos_emb_forward_rank0_0.pt", output
    if module == "lm_head":
        path = train_root / f"{model_prefix}.{lm_head_name}_forward_rank0_0.pt"
        return path, "output.0"
    if layer is None:
        return train_root / f"{model_prefix}.{module}_forward_rank0_0.pt", "output.0"
    # Conv training output is captured as the next layernorm input in the
    # original Megatron dump flow.
    next_inputs = {
        "self_attention.qa_conv": "self_attention.q_layernorm",
        "self_attention.compresskv_conv": "self_attention.k_layernorm",
    }
    training_module = next_inputs.get(module, module)
    key = "input.0" if module in next_inputs else "output.0"
    return (
        train_root / f"{layer_prefix}.{layer}.{training_module}_forward_rank0_0.pt",
        key,
    )


def _apply_metrics(
    result: CompareResult,
    left: torch.Tensor,
    right: torch.Tensor,
    rtol: float,
    atol: float,
) -> CompareResult:
    passed, metrics = compare_tensors(left, right, rtol=rtol, atol=atol)
    for name, value in metrics.items():
        setattr(result, name, value)
    result.seq_len = int(right.shape[0]) if right.ndim else 0
    if not result.train_path:
        result.train_path = result.left_path
    if not result.infer_path:
        result.infer_path = result.right_path
    if not result.left_path:
        result.left_path = result.train_path
    if not result.right_path:
        result.right_path = result.infer_path
    if left.shape != right.shape:
        result.status = "FAIL"
        result.message = f"shape mismatch {tuple(left.shape)} vs {tuple(right.shape)}"
    elif passed:
        result.status = "PASS"
        result.message = "allclose within tolerance"
    else:
        result.status = "FAIL"
        result.message = f"max_abs_diff={result.max_abs_diff:.4e}"
    return result


def compare_one(
    *,
    layer: int | None,
    module: str,
    train_root: Path,
    infer_root: Path,
    custom_root: Path | None,
    mode: str,
    role: str,
    mock_seq_len: int,
    rank: int,
    rtol: float,
    atol: float,
    layer_prefix: str,
    model_prefix: str,
    lm_head_name: str,
) -> CompareResult:
    if module in EXPECTED_SKIPS:
        reason = EXPECTED_SKIPS.get(module, "structurally unavailable")
        return CompareResult(layer, module, "SKIP", reason)
    infer_path = _first_file(
        infer_candidates(
            infer_root,
            layer=layer,
            module=module,
            mode=mode,
            role=role,
            rank=rank,
        )
    )
    train_path, key = training_path(
        train_root,
        custom_root,
        layer=layer,
        module=module,
        layer_prefix=layer_prefix,
        model_prefix=model_prefix,
        lm_head_name=lm_head_name,
    )
    result = CompareResult(
        layer,
        module,
        "ERROR",
        train_path=str(train_path),
        infer_path=str(infer_path),
        left_path=str(train_path),
        right_path=str(infer_path),
    )
    if not infer_path.is_file():
        result.status = "SKIP"
        result.message = "inference dump missing"
        return result

    # Real router/apply-experts values use the train_ref inference pass as
    # their closest like-for-like golden because raw training files represent
    # a different EP dispatch layout.
    router_outputs = ("moe_router_scores", "moe_router_indices")
    if mode == "real" and module in router_outputs:
        train_path = _first_file(
            infer_candidates(
                infer_root,
                layer=layer,
                module=module,
                mode="train_ref",
                role=role,
                rank=rank,
            )
        )
        result.train_path = str(train_path)
        result.left_path = str(train_path)
        key = "output.0"
    if not train_path.is_file():
        result.status = "SKIP"
        result.message = "training/reference dump missing"
        return result
    try:
        infer = _squeeze_middle(load_tensor(infer_path))
        if mode == "real" and module in router_outputs:
            train = _squeeze_middle(load_tensor(train_path))
            if train.ndim and infer.ndim:
                tokens = min(train.shape[0], infer.shape[0], mock_seq_len)
                train, infer = train[:tokens], infer[:tokens]
        elif module == "lm_head":
            train = _squeeze_middle(load_tensor(train_path, key))
            if train.ndim == 3 and train.shape[1] > 0:
                train = train[:, 0, :]
            if train.ndim == 0 or train.shape[0] < mock_seq_len:
                raise ValueError("LM-head training output is shorter than mock_seq_len")
            train = train.narrow(0, mock_seq_len - 1, 1)
        elif module in MOE_GATE_MODULES:
            full = _squeeze_middle(load_tensor(train_path, key))
            local = infer.shape[0] if infer.ndim else 0
            start = rank * local
            train = full.narrow(0, start, local)
        elif module == "gmm_experts_output":
            train = _squeeze_middle(load_tensor(train_path, key))
            if train.ndim and infer.ndim:
                tokens = min(train.shape[0], infer.shape[0], mock_seq_len)
                train, infer = train[:tokens], infer[:tokens]
        else:
            train = normalize_training_tensor(
                train_path,
                key=key,
                role=role,
                mock_seq_len=mock_seq_len,
                infer=infer,
            )
        if module == "pangu_transformer_o_conv" and train.ndim and infer.ndim:
            width = min(train.shape[-1], infer.shape[-1])
            train, infer = train[..., :width], infer[..., :width]
        return _apply_metrics(result, train, infer, rtol, atol)
    except Exception as exc:  # noqa: BLE001 - one bad case must remain reportable
        result.status = "ERROR"
        result.message = str(exc)
        return result


def _decorator_pairs(
    root: Path,
    role: str,
    rank: int,
    run_a: str,
    run_b: str,
) -> list[tuple[str, int, Path, Path]]:
    directory = _stage_dir(root, role)
    pattern = re.compile(
        rf"^decorator_(?P<name>.+)_run(?P<run>{re.escape(run_a)}|"
        rf"{re.escape(run_b)})_call(?P<call>\d+)_rank{rank}\.pt$"
    )
    if not directory.is_dir():
        return []
    keys: set[tuple[str, int]] = set()
    for path in directory.iterdir():
        match = pattern.match(path.name)
        if match is None:
            continue
        keys.add((match.group("name"), int(match.group("call"))))
    pairs = []
    for name, call in keys:
        base = f"decorator_{name}_run{{run}}_call{call}_rank{rank}.pt"
        pairs.append(
            (
                name,
                call,
                directory / base.format(run=run_a),
                directory / base.format(run=run_b),
            )
        )
    return sorted(pairs, key=lambda item: (item[0], item[1]))


def compare_pure_real(
    infer_root: Path,
    *,
    role: str,
    rank: int,
    run_a: str,
    run_b: str,
    mock_seq_len: int,
    rtol: float,
    atol: float,
) -> list[CompareResult]:
    results = []
    layer_pattern = re.compile(r"^(?:model\.)?layers\.(\d+)\.(.+)$")
    for qualname, call, left_path, right_path in _decorator_pairs(
        infer_root, role, rank, run_a, run_b
    ):
        match = layer_pattern.match(qualname)
        layer = int(match.group(1)) if match else None
        module = match.group(2) if match else qualname
        if call:
            module = f"{module}#call{call}"
        result = CompareResult(
            layer,
            module,
            "ERROR",
            train_path=str(left_path),
            infer_path=str(right_path),
            left_path=str(left_path),
            right_path=str(right_path),
        )
        if not left_path.is_file() or not right_path.is_file():
            result.status = "SKIP"
            missing = left_path if not left_path.is_file() else right_path
            result.message = f"pure_real run dump missing: {missing}"
            results.append(result)
            continue
        try:
            left = _squeeze_middle(load_tensor(left_path))
            right = _squeeze_middle(load_tensor(right_path))
            if left.ndim and right.ndim:
                tokens = min(left.shape[0], right.shape[0], mock_seq_len)
                left, right = left[:tokens], right[:tokens]
            results.append(_apply_metrics(result, left, right, rtol, atol))
        except Exception as exc:  # noqa: BLE001 - continue reporting other pairs
            result.status = "ERROR"
            result.message = str(exc)
            results.append(result)
    return results


def discover_layers(infer_root: Path, role: str) -> list[int]:
    root = _stage_dir(infer_root, role)
    layers = set()
    if root.is_dir():
        for path in root.iterdir():
            match = re.match(r"decoder\.layers\.(\d+)\.", path.name)
            if match:
                layers.add(int(match.group(1)))
    return sorted(layers)


def write_reports(
    results: list[CompareResult],
    report_dir: Path,
    tag: str,
    notable_threshold: float = 1e-4,
) -> None:
    report_dir.mkdir(parents=True, exist_ok=True)
    rows = [asdict(item) for item in results]
    (report_dir / f"summary_{tag}.json").write_text(
        json.dumps(rows, indent=2, ensure_ascii=False, allow_nan=True),
        encoding="utf-8",
    )
    with (report_dir / f"summary_{tag}.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        if rows:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            for row in rows:
                row = dict(row)
                row["error_distribution"] = json.dumps(
                    row["error_distribution"], ensure_ascii=False
                )
                writer.writerow(row)
    counts = {
        status: sum(item.status == status for item in results)
        for status in ("PASS", "FAIL", "SKIP", "ERROR")
    }
    lines = [
        "# Accuracy-alignment comparison",
        "",
        f"Generated: {datetime.now(tz=UTC).isoformat(timespec='seconds')}",
        "",
        "| Status | Count |",
        "|---|---:|",
        *(f"| {name} | {value} |" for name, value in counts.items()),
        "",
        (
            "| Status | Layer | Module | Seq len | Shapes | All close | "
            "Max abs | Mean abs | Max rel % | Avg rel % | Corr | Message |"
        ),
        "|---|---:|---|---:|:---:|:---:|---:|---:|---:|---:|---:|---|",
    ]
    for item in results:
        if item.status == "PASS" and item.max_abs_diff <= notable_threshold:
            continue
        layer = "-" if item.layer_idx is None else str(item.layer_idx)
        corr = "" if math.isnan(item.corr_coef) else f"{item.corr_coef:.6f}"
        lines.append(
            f"| {item.status} | {layer} | `{item.module}` | "
            f"{item.seq_len} | {item.shapes_match} | {item.all_close} | "
            f"{item.max_abs_diff:.4e} | {item.mean_abs_diff:.4e} | "
            f"{item.max_rel_diff_percent:.4f} | "
            f"{item.avg_rel_diff_percent:.4f} | {corr} | {item.message} |"
        )
    (report_dir / f"summary_{tag}.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    by_module: dict[str, list[CompareResult]] = defaultdict(list)
    for item in results:
        if item.status != "SKIP":
            by_module[item.module].append(item)
    for module, module_results in by_module.items():
        module_results.sort(
            key=lambda item: (item.layer_idx is not None, item.layer_idx or 0)
        )
        module_dir = report_dir / module.replace("/", "_")
        module_dir.mkdir(parents=True, exist_ok=True)
        (module_dir / f"all_layers_{tag}.json").write_text(
            json.dumps(
                [asdict(item) for item in module_results],
                indent=2,
                ensure_ascii=False,
                allow_nan=True,
            ),
            encoding="utf-8",
        )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-root", type=Path)
    parser.add_argument("--infer-root", type=Path, required=True)
    parser.add_argument("--custom-train-root", type=Path)
    parser.add_argument(
        "--role",
        "--mock-role",
        dest="role",
        choices=("P", "D"),
        default="P",
    )
    parser.add_argument(
        "--compare-mode",
        choices=("train_ref", "real", "pure_real", "pure_real_decorator"),
        default="train_ref",
    )
    parser.add_argument("--run-ids", nargs=2, default=("0", "1"))
    parser.add_argument("--mock-seq-len", type=int, required=True)
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument("--layer-start", type=int)
    parser.add_argument("--layer-end", type=int)
    parser.add_argument("--modules", nargs="+")
    parser.add_argument("--module-set", choices=tuple(MODULE_SETS), default="all")
    parser.add_argument(
        "--dsa-layers",
        nargs="*",
        type=int,
        default=list(DEFAULT_DSA_LAYERS),
    )
    parser.add_argument("--rtol", type=float, default=1e-5)
    parser.add_argument("--atol", type=float, default=1e-5)
    parser.add_argument("--workers", type=int)
    parser.add_argument("--md-diff-threshold", type=float, default=1e-4)
    parser.add_argument("--report-dir", type=Path, default=Path("reports"))
    parser.add_argument(
        "--report-prefix",
        help="Legacy-compatible report directory prefix; defaults to --module-set.",
    )
    parser.add_argument(
        "--layer-train-prefix",
        "--train-dump-name-prefix",
        dest="layer_train_prefix",
        default="iter0_module.module.decoder.layers",
    )
    parser.add_argument("--model-train-prefix", default="iter0_module.module")
    parser.add_argument("--lm-head-train-name", default="output_layer")
    return parser.parse_args(argv)


def _run_parallel(tasks: list[Any], function, workers: int | None):
    if not tasks:
        return []
    count = workers or min(16, os.cpu_count() or 8)
    count = max(1, min(count, len(tasks)))
    if count == 1:
        return [function(task) for task in tasks]
    with ThreadPoolExecutor(max_workers=count) as pool:
        return list(pool.map(function, tasks))


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    mode = (
        "pure_real" if args.compare_mode == "pure_real_decorator" else args.compare_mode
    )
    if mode == "pure_real":
        results = compare_pure_real(
            args.infer_root,
            role=args.role,
            rank=args.rank,
            run_a=args.run_ids[0],
            run_b=args.run_ids[1],
            mock_seq_len=args.mock_seq_len,
            rtol=args.rtol,
            atol=args.atol,
        )
    else:
        if args.train_root is None:
            raise SystemExit("--train-root is required outside pure_real mode")
        if args.layer_start is not None or args.layer_end is not None:
            if args.layer_start is None or args.layer_end is None:
                raise SystemExit("--layer-start and --layer-end must be used together")
            layers = list(range(args.layer_start, args.layer_end))
        else:
            layers = discover_layers(args.infer_root, args.role)
        selected_set = MODULE_SETS.get(args.module_set)
        if selected_set is None:
            raise SystemExit(f"unknown module set: {args.module_set}")
        modules = tuple(args.modules or selected_set)
        if args.module_set in {"dsa", "mla"} and args.dsa_layers:
            dsa = set(args.dsa_layers)
            layers = [
                layer
                for layer in layers
                if (layer in dsa) == (args.module_set == "dsa")
            ]
        tasks = [
            (None, module) if module in MODEL_MODULES else (layer, module)
            for module in modules
            for layer in ([None] if module in MODEL_MODULES else layers)
        ]

        def compare_task(task):
            layer, module = task
            return compare_one(
                layer=layer,
                module=module,
                train_root=args.train_root,
                infer_root=args.infer_root,
                custom_root=args.custom_train_root,
                mode=mode,
                role=args.role,
                mock_seq_len=args.mock_seq_len,
                rank=args.rank,
                rtol=args.rtol,
                atol=args.atol,
                layer_prefix=args.layer_train_prefix,
                model_prefix=args.model_train_prefix,
                lm_head_name=args.lm_head_train_name,
            )

        results = _run_parallel(tasks, compare_task, args.workers)
    timestamp = datetime.now(tz=UTC).strftime("%Y%m%d_%H%M%S")
    report_prefix = args.report_prefix or args.module_set
    tag = f"{report_prefix}_{args.role.lower()}_{mode}_{timestamp}"
    report_dir = args.report_dir / tag
    write_reports(
        results,
        report_dir,
        mode,
        notable_threshold=args.md_diff_threshold,
    )
    counts = {
        status: sum(item.status == status for item in results)
        for status in ("PASS", "FAIL", "SKIP", "ERROR")
    }
    print(" ".join(f"{name}={value}" for name, value in counts.items()))
    print(f"Report: {report_dir}")
    return 1 if counts.get("FAIL", 0) or counts.get("ERROR", 0) else 0


if __name__ == "__main__":
    raise SystemExit(main())
