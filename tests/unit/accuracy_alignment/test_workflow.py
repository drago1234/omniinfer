# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.

import io
import json
import subprocess
import urllib.error
from types import SimpleNamespace

import pytest
from omni_npu.diagnostics.accuracy_alignment import workflow


class _Response:
    def __init__(self, status=200, body=b"ok"):
        self.status = status
        self.body = body

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def read(self):
        return self.body


def _workflow_args(tmp_path, **overrides):
    values = {
        "ansible_playbook": None,
        "inventory": None,
        "ansible_tags": None,
        "ansible_arg": [],
        "decode_source": None,
        "ssh_key": None,
        "infer_root": tmp_path / "infer",
        "compare_mode": "train_ref",
        "train_root": tmp_path / "train",
        "custom_train_root": None,
        "prompt_length": 3,
        "module_set": "all",
        "report_dir": tmp_path / "reports",
        "rank": 0,
        "rtol": 1e-5,
        "atol": 1e-5,
        "run_ids": ("0", "1"),
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_environment_helpers_prefer_first_nonempty_value(monkeypatch, tmp_path):
    monkeypatch.setenv("AA_EMPTY", "  ")
    monkeypatch.setenv("AA_VALUE", str(tmp_path))

    assert workflow._env_value("AA_EMPTY", "AA_VALUE") == str(tmp_path)
    assert workflow._env_path("AA_EMPTY", "AA_VALUE") == tmp_path
    assert workflow._env_path("AA_MISSING") is None


def test_http_status_handles_success_http_error_and_transport_error(monkeypatch):
    monkeypatch.setattr(
        workflow.urllib.request,
        "urlopen",
        lambda *_args, **_kwargs: _Response(204),
    )
    assert workflow._http_status("http://service/health") == 204

    def http_error(*_args, **_kwargs):
        raise urllib.error.HTTPError(
            "http://service/health", 503, "busy", {}, io.BytesIO(b"busy")
        )

    monkeypatch.setattr(workflow.urllib.request, "urlopen", http_error)
    assert workflow._http_status("http://service/health") == 503

    def transport_error(*_args, **_kwargs):
        raise urllib.error.URLError("offline")

    monkeypatch.setattr(workflow.urllib.request, "urlopen", transport_error)
    assert workflow._http_status("http://service/health") == 0


def test_wait_for_health_retries_and_reports_pending_urls(monkeypatch):
    statuses = iter((503, 200))
    monkeypatch.setattr(workflow, "_http_status", lambda _url: next(statuses))
    monkeypatch.setattr(workflow.time, "monotonic", lambda: 0.0)
    sleeps = []
    monkeypatch.setattr(workflow.time, "sleep", sleeps.append)

    workflow.wait_for_health(["http://service/health"], 10, 0.25)
    assert sleeps == [0.25]

    with pytest.raises(RuntimeError, match="health check timed out"):
        workflow.wait_for_health(["http://bad/health"], 0, 1)


@pytest.mark.parametrize(
    ("content", "expected_key"),
    (("hello", "messages"), ("", "prompt")),
)
def test_send_request_builds_chat_and_token_payloads(
    monkeypatch, content, expected_key
):
    captured = {}

    def urlopen(request, timeout):
        captured["request"] = request
        captured["timeout"] = timeout
        return _Response(200)

    monkeypatch.setattr(workflow.urllib.request, "urlopen", urlopen)
    workflow.send_request(
        "http://service/v1/chat/completions",
        model="pangu",
        prompt_token=7,
        prompt_length=3,
        max_tokens=2,
        timeout=9,
        content=content,
    )

    payload = json.loads(captured["request"].data)
    assert expected_key in payload
    assert payload["model"] == "pangu"
    assert payload["max_tokens"] == 2
    assert captured["timeout"] == 9


def test_send_request_reports_http_error_body(monkeypatch):
    def urlopen(*_args, **_kwargs):
        raise urllib.error.HTTPError(
            "http://service", 400, "bad", {}, io.BytesIO(b"invalid request")
        )

    monkeypatch.setattr(workflow.urllib.request, "urlopen", urlopen)
    with pytest.raises(RuntimeError, match="HTTP 400: invalid request"):
        workflow.send_request(
            "http://service",
            model="",
            prompt_token=1,
            prompt_length=1,
            max_tokens=1,
            timeout=1,
        )


def test_ansible_and_decode_sync_build_expected_commands(monkeypatch, tmp_path):
    commands = []
    monkeypatch.setattr(
        workflow,
        "_run",
        lambda command, **kwargs: commands.append((command, kwargs)),
    )
    args = _workflow_args(
        tmp_path,
        ansible_playbook=tmp_path / "server.yml",
        inventory=tmp_path / "hosts.yml",
        ansible_tags="server",
        ansible_arg=["--extra-vars 'role=P'"],
        decode_source="user@host:/dump/decode/",
        ssh_key=tmp_path / "id_rsa",
    )

    workflow._run_ansible(args)
    workflow._sync_decode(args)

    assert commands[0][0][:3] == [
        "ansible-playbook",
        "-i",
        str(args.inventory),
    ]
    assert commands[0][0][-2:] == ["--extra-vars", "role=P"]
    assert commands[1][0][:3] == [
        "rsync",
        "-a",
        "-e",
    ]
    assert commands[1][0][-2:] == [
        "user@host:/dump/decode/",
        str(args.infer_root / "decode") + "/",
    ]


def test_optional_ansible_and_decode_sync_are_noops(monkeypatch, tmp_path):
    monkeypatch.setattr(
        workflow,
        "_run",
        lambda *_args, **_kwargs: pytest.fail("unexpected command"),
    )
    args = _workflow_args(tmp_path)
    workflow._run_ansible(args)
    workflow._sync_decode(args)


def test_dump_validation_and_stale_rejection(tmp_path):
    root = tmp_path / "infer"
    with pytest.raises(RuntimeError, match="no prefill dump"):
        workflow._validate_dump(root, "prefill")

    prefill = root / "prefill"
    prefill.mkdir(parents=True)
    (prefill / "output.pt").touch()
    workflow._validate_dump(root, "prefill")

    with pytest.raises(RuntimeError, match="existing dump files"):
        workflow._reject_stale_dumps(root)


def test_compare_builds_train_ref_and_pure_real_commands(monkeypatch, tmp_path):
    commands = []

    def run(command, check):
        commands.append((command, check))
        return SimpleNamespace(returncode=4)

    monkeypatch.setattr(workflow.subprocess, "run", run)
    args = _workflow_args(tmp_path, custom_train_root=tmp_path / "custom")
    assert workflow._compare(args, "P") == 4
    assert "--train-root" in commands[-1][0]
    assert "--custom-train-root" in commands[-1][0]

    args.compare_mode = "pure_real"
    assert workflow._compare(args, "D") == 4
    assert "--train-root" not in commands[-1][0]
    assert commands[-1][0][-3:] == ["--run-ids", "0", "1"]

    args.compare_mode = "real"
    args.train_root = None
    with pytest.raises(RuntimeError, match="--train-root is required"):
        workflow._compare(args, "P")


def test_parse_args_uses_environment_defaults(monkeypatch, tmp_path):
    monkeypatch.setenv("OMNI_NPU_DEBUG_MOCK_SHAPE", "5")
    monkeypatch.setenv("OMNI_NPU_DEBUG_TRAIN_DUMP_DIR", str(tmp_path / "train"))
    monkeypatch.setenv("OMNI_NPU_DEBUG_INFER_DUMP_DIR", str(tmp_path / "infer"))

    args = workflow.parse_args([])

    assert args.prompt_length == 5
    assert args.train_root == tmp_path / "train"
    assert args.infer_root == tmp_path / "infer"


def test_main_runs_prefill_and_decode_comparisons(monkeypatch, tmp_path):
    infer_root = tmp_path / "infer"
    for stage in ("prefill", "decode"):
        directory = infer_root / stage
        directory.mkdir(parents=True)
        (directory / "output.pt").touch()
    calls = []
    monkeypatch.setattr(workflow, "_run_ansible", lambda _args: calls.append("run"))
    monkeypatch.setattr(
        workflow,
        "wait_for_health",
        lambda urls, timeout, interval: calls.append((urls, timeout, interval)),
    )
    monkeypatch.setattr(workflow, "_sync_decode", lambda _args: calls.append("sync"))
    monkeypatch.setattr(
        workflow,
        "_compare",
        lambda _args, role: calls.append(role) or (1 if role == "D" else 0),
    )

    result = workflow.main(
        [
            "--infer-root",
            str(infer_root),
            "--train-root",
            str(tmp_path / "train"),
            "--prompt-length",
            "3",
            "--health-url",
            "http://service/health",
            "--skip-request",
            "--keep-old",
        ]
    )

    assert result == 1
    assert "P" in calls and "D" in calls


def test_main_validates_required_inputs_and_request_url(tmp_path):
    with pytest.raises(SystemExit, match="--infer-root"):
        workflow.main(["--prompt-length", "1"])
    with pytest.raises(SystemExit, match="--prompt-length"):
        workflow.main(["--infer-root", str(tmp_path)])
    with pytest.raises(SystemExit, match="--request-url"):
        workflow.main(
            [
                "--infer-root",
                str(tmp_path),
                "--prompt-length",
                "1",
                "--keep-old",
                "--skip-compare",
            ]
        )


def test_run_propagates_subprocess_failure(monkeypatch):
    def fail(*_args, **_kwargs):
        raise subprocess.CalledProcessError(2, ["bad"])

    monkeypatch.setattr(workflow.subprocess, "run", fail)
    with pytest.raises(subprocess.CalledProcessError):
        workflow._run(["bad"])
