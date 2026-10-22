# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""Portable one-shot driver for an accuracy-alignment capture and compare."""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path


def _env_value(*names: str) -> str:
    for name in names:
        value = os.environ.get(name, "").strip()
        if value:
            return value
    return ""


def _env_path(*names: str) -> Path | None:
    value = _env_value(*names)
    return Path(value) if value else None


def _http_status(url: str, timeout: float = 5.0) -> int:
    request = urllib.request.Request(url, method="GET")
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return int(response.status)
    except urllib.error.HTTPError as exc:
        return int(exc.code)
    except (OSError, urllib.error.URLError):
        return 0


def wait_for_health(urls: list[str], timeout: float, interval: float) -> None:
    deadline = time.monotonic() + timeout
    pending = set(urls)
    while pending and time.monotonic() < deadline:
        pending = {url for url in pending if _http_status(url) != 200}
        if pending:
            time.sleep(interval)
    if pending:
        raise RuntimeError("health check timed out: " + ", ".join(sorted(pending)))


def send_request(
    url: str,
    *,
    model: str,
    prompt_token: int,
    prompt_length: int,
    max_tokens: int,
    timeout: float,
    content: str = "",
) -> None:
    if content:
        payload = {
            "messages": [{"role": "user", "content": content}],
            "max_tokens": max_tokens,
            "temperature": 0,
            "top_k": 1,
            "top_p": 1,
        }
    else:
        payload = {
            "prompt": [prompt_token] * prompt_length,
            "max_tokens": max_tokens,
            "temperature": 0,
        }
    if model:
        payload["model"] = model
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            if not 200 <= response.status < 300:
                raise RuntimeError(f"inference request returned HTTP {response.status}")
            response.read()
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"inference request failed HTTP {exc.code}: {body}") from exc


def _run(command: list[str], *, env: dict[str, str] | None = None) -> None:
    subprocess.run(command, check=True, env=env)


def _run_ansible(args: argparse.Namespace) -> None:
    if args.ansible_playbook is None:
        return
    command = ["ansible-playbook"]
    if args.inventory is not None:
        command.extend(("-i", str(args.inventory)))
    command.append(str(args.ansible_playbook))
    if args.ansible_tags:
        command.extend(("--tags", args.ansible_tags))
    for value in args.ansible_arg:
        command.extend(shlex.split(value))
    _run(command, env=os.environ.copy())


def _sync_decode(args: argparse.Namespace) -> None:
    if not args.decode_source:
        return
    destination = args.infer_root / "decode"
    destination.mkdir(parents=True, exist_ok=True)
    command = ["rsync", "-a"]
    if args.ssh_key is not None:
        command.extend(("-e", f"ssh -i {args.ssh_key}"))
    command.extend((args.decode_source.rstrip("/") + "/", str(destination) + "/"))
    _run(command)


def _validate_dump(root: Path, stage: str) -> None:
    directory = root / stage
    if not directory.is_dir() or not any(directory.glob("*.pt")):
        raise RuntimeError(f"no {stage} dump files found under {directory}")


def _reject_stale_dumps(root: Path) -> None:
    stale = [
        path for stage in ("prefill", "decode") for path in (root / stage).glob("*.pt")
    ]
    if stale:
        raise RuntimeError(
            f"found {len(stale)} existing dump files under {root}; move them "
            "to a run-specific directory or pass --keep-old intentionally"
        )


def _compare(args: argparse.Namespace, role: str) -> int:
    command = [
        sys.executable,
        "-m",
        "omni_npu.diagnostics.accuracy_alignment.compare",
        "--infer-root",
        str(args.infer_root),
        "--role",
        role,
        "--compare-mode",
        args.compare_mode,
        "--mock-seq-len",
        str(args.prompt_length),
        "--module-set",
        args.module_set,
        "--report-dir",
        str(args.report_dir),
        "--rank",
        str(args.rank),
        "--rtol",
        str(args.rtol),
        "--atol",
        str(args.atol),
    ]
    if args.compare_mode not in {"pure_real", "pure_real_decorator"}:
        if args.train_root is None:
            raise RuntimeError("--train-root is required for train_ref/real compare")
        command.extend(("--train-root", str(args.train_root)))
    if args.custom_train_root is not None:
        command.extend(("--custom-train-root", str(args.custom_train_root)))
    if args.compare_mode in {"pure_real", "pure_real_decorator"}:
        command.extend(("--run-ids", *args.run_ids))
    return subprocess.run(command, check=False).returncode


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ansible-playbook", type=Path)
    parser.add_argument("--inventory", type=Path)
    parser.add_argument("--ansible-tags")
    parser.add_argument("--ansible-arg", action="append", default=[])
    parser.add_argument("--health-url", action="append", default=[])
    parser.add_argument("--health-timeout", type=float, default=600)
    parser.add_argument("--health-interval", type=float, default=10)
    parser.add_argument("--request-url")
    parser.add_argument("--model", default=os.environ.get("MODEL_NAME", ""))
    parser.add_argument("--content", default=os.environ.get("CONTENT", ""))
    parser.add_argument("--prompt-token", type=int, default=1)
    parser.add_argument(
        "--prompt-length",
        type=int,
        default=int(
            _env_value(
                "OMNI_NPU_DEBUG_MOCK_SHAPE",
                "OMNI_NPU_MHC_DEBUG_MOCK_SHAPE",
            )
            or "0"
        ),
    )
    parser.add_argument("--max-tokens", type=int, default=1)
    parser.add_argument("--request-timeout", type=float, default=600)
    parser.add_argument("--decode-source")
    parser.add_argument("--ssh-key", type=Path)
    parser.add_argument(
        "--train-root",
        type=Path,
        default=_env_path(
            "OMNI_NPU_DEBUG_TRAIN_DUMP_DIR",
            "OMNI_NPU_MHC_DEBUG_TRAIN_DUMP_DIR",
        ),
    )
    parser.add_argument(
        "--custom-train-root",
        type=Path,
        default=_env_path("OMNI_NPU_MHC_CUSTOM_DUMP_DIR"),
    )
    parser.add_argument(
        "--infer-root",
        type=Path,
        default=_env_path(
            "OMNI_NPU_DEBUG_INFER_DUMP_DIR",
            "OMNI_NPU_MHC_DEBUG_INFER_DUMP_DIR",
        ),
    )
    parser.add_argument("--report-dir", type=Path, default=Path("reports"))
    parser.add_argument(
        "--compare-mode",
        choices=("train_ref", "real", "pure_real", "pure_real_decorator"),
        default="train_ref",
    )
    parser.add_argument("--run-ids", nargs=2, default=("0", "1"))
    parser.add_argument("--module-set", default="all")
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument("--rtol", type=float, default=1e-5)
    parser.add_argument("--atol", type=float, default=1e-5)
    parser.add_argument("--skip-request", action="store_true")
    parser.add_argument("--skip-compare", action="store_true")
    parser.add_argument(
        "--keep-old",
        action="store_true",
        help="Allow existing prefill/decode .pt files (disabled by default)",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.infer_root is None:
        raise SystemExit("--infer-root or OMNI_NPU_DEBUG_INFER_DUMP_DIR is required")
    if args.prompt_length <= 0:
        raise SystemExit("--prompt-length or OMNI_NPU_DEBUG_MOCK_SHAPE must be > 0")
    _run_ansible(args)
    if args.health_url:
        wait_for_health(args.health_url, args.health_timeout, args.health_interval)
    if not args.keep_old:
        _reject_stale_dumps(args.infer_root)
    if not args.skip_request:
        if not args.request_url:
            raise SystemExit("--request-url is required unless --skip-request is set")
        send_request(
            args.request_url,
            model=args.model,
            prompt_token=args.prompt_token,
            prompt_length=args.prompt_length,
            max_tokens=args.max_tokens,
            timeout=args.request_timeout,
            content=args.content,
        )
    _sync_decode(args)
    if args.skip_compare:
        return 0
    _validate_dump(args.infer_root, "prefill")
    rc = _compare(args, "P")
    decode_dir = args.infer_root / "decode"
    if decode_dir.is_dir() and any(decode_dir.glob("*.pt")):
        rc = max(rc, _compare(args, "D"))
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
