# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# SPDX-License-Identifier: MIT
"""Unit tests for omni_npu.diagnostics.msprobe.

These cover the pure config-resolution / debugger-construction logic only;
the per-forward start/stop/step lifecycle is exercised in
``tests/unit/platform/test_npu_model_runner.py``.

``build_debugger`` imports ``msprobe.pytorch`` lazily, so each test that
exercises a non-None path injects a fake ``msprobe.pytorch`` module into
``sys.modules`` — no real msprobe install is required.
"""
import json
import os
import sys
import types

import pytest

from omni_npu.diagnostics import msprobe
from omni_npu.diagnostics.msprobe import (
    build_debugger,
    materialize_dump_config_to_file,
    resolve_dump_config_path,
)


# ---------------------------------------------------------------------------
# resolve_dump_config_path
# ---------------------------------------------------------------------------

def test_resolve_dump_config_path_none_returns_none():
    assert resolve_dump_config_path(None, None) is None


def test_resolve_dump_config_path_passthrough(monkeypatch, tmp_path):
    # Isolate cwd so we never touch the real working directory even though
    # the passthrough path is absolute and arbitrary.
    monkeypatch.chdir(tmp_path)
    assert resolve_dump_config_path("/data/cfg.json", None) == "/data/cfg.json"


def test_resolve_dump_config_path_mutually_exclusive():
    with pytest.raises(ValueError, match="Only one of"):
        resolve_dump_config_path(
            "/data/cfg.json", {"task": "statistics"}
        )


def test_resolve_dump_config_path_dump_config_must_be_dict():
    with pytest.raises(TypeError, match="msprobe_dump_config must be a dict"):
        resolve_dump_config_path(None, "not-a-dict")


def test_resolve_dump_config_path_must_be_str():
    with pytest.raises(
        TypeError, match="msprobe_dump_config_path must be a string"
    ):
        resolve_dump_config_path(0, None)


def test_resolve_dump_config_path_materializes_inline_dict(
    monkeypatch, tmp_path
):
    # MSPROBE_DUMP_DIR is computed at import time from os.getcwd(); chdir to
    # a tmp dir and rebind the module constant so the file lands there.
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        msprobe, "MSPROBE_DUMP_DIR",
        os.path.join(os.getcwd(), ".omni_npu", "msprobe"),
    )

    inline = {"task": "statistics", "level": "L1", "dump_path": "/dump"}
    path = resolve_dump_config_path(None, inline)

    assert path is not None
    assert os.path.isfile(path)
    assert path.endswith("msprobe_dump_config.json")
    with open(path, encoding="utf-8") as fh:
        assert json.load(fh) == inline


# ---------------------------------------------------------------------------
# materialize_dump_config_to_file (direct)
# ---------------------------------------------------------------------------

def test_materialize_dump_config_to_file_writes_json(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        msprobe, "MSPROBE_DUMP_DIR",
        os.path.join(os.getcwd(), ".omni_npu", "msprobe"),
    )

    cfg = {"task": "tensor", "level": "L0"}
    path = materialize_dump_config_to_file(cfg)

    assert os.path.isfile(path)
    with open(path, encoding="utf-8") as fh:
        assert json.load(fh) == cfg


def test_materialize_dump_config_to_file_overwrites(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        msprobe, "MSPROBE_DUMP_DIR",
        os.path.join(os.getcwd(), ".omni_npu", "msprobe"),
    )

    materialize_dump_config_to_file({"task": "statistics"})
    path = materialize_dump_config_to_file({"task": "tensor"})

    with open(path, encoding="utf-8") as fh:
        assert json.load(fh) == {"task": "tensor"}


# ---------------------------------------------------------------------------
# build_debugger
# ---------------------------------------------------------------------------

def _install_fake_msprobe(monkeypatch, *, has_aclgraph_dumper=True):
    """Inject a fake ``msprobe.pytorch`` module with stub debugger classes.

    Returns the two stub classes so a test can assert on which one was used.
    """
    fake_module = types.ModuleType("msprobe.pytorch")

    class FakePrecisionDebugger:
        def __init__(self, config_path):
            self.config_path = config_path
            self.started = False
            self.stopped = False
            self.step_count = 0

        def start(self, model):
            self.started = True

        def stop(self):
            self.stopped = True

        def step(self, **kwargs):
            self.step_count += 1

    class FakeAclGraphDumper:
        def __init__(self, config_path):
            self.config_path = config_path
            self.started = False
            self.step_count = 0

        def start(self, model):
            self.started = True

        # NOTE: no stop() — mirrors the real AclGraphDumper API.

        def step(self, **kwargs):
            self.step_count += 1

    fake_module.PrecisionDebugger = FakePrecisionDebugger
    if has_aclgraph_dumper:
        fake_module.AclGraphDumper = FakeAclGraphDumper

    # Register both ``msprobe`` and ``msprobe.pytorch`` so the lazy
    # ``from msprobe.pytorch import ...`` inside build_debugger resolves.
    msprobe_pkg = types.ModuleType("msprobe")
    msprobe_pkg.pytorch = fake_module
    monkeypatch.setitem(sys.modules, "msprobe", msprobe_pkg)
    monkeypatch.setitem(sys.modules, "msprobe.pytorch", fake_module)

    return FakePrecisionDebugger, FakeAclGraphDumper


def test_build_debugger_none_when_no_config():
    debugger = build_debugger(None, use_graph=False)
    assert debugger is None

    debugger = build_debugger(None, use_graph=True)
    assert debugger is None


def test_build_debugger_eager_uses_precision_debugger(monkeypatch):
    fake_pd, _ = _install_fake_msprobe(monkeypatch)

    debugger = build_debugger("/data/cfg.json", use_graph=False)

    assert isinstance(debugger, fake_pd)
    assert debugger.config_path == "/data/cfg.json"


def test_build_debugger_graph_uses_acl_graph_dumper(monkeypatch):
    _, fake_agd = _install_fake_msprobe(monkeypatch)

    debugger = build_debugger("/data/cfg.json", use_graph=True)

    assert isinstance(debugger, fake_agd)
    assert debugger.config_path == "/data/cfg.json"


def test_build_debugger_graph_import_error_raises_runtimeerror(monkeypatch):
    # AclGraphDumper absent from the (pip-installed) msprobe build.
    _install_fake_msprobe(monkeypatch, has_aclgraph_dumper=False)

    with pytest.raises(RuntimeError, match="AclGraphDumper"):
        build_debugger("/data/cfg.json", use_graph=True)


def test_build_debugger_eager_import_error_propagates(monkeypatch):
    # No msprobe installed at all — eager mode must fail loudly rather than
    # silently skip dump. Remove any cached fake module first.
    monkeypatch.delitem(sys.modules, "msprobe", raising=False)
    monkeypatch.delitem(sys.modules, "msprobe.pytorch", raising=False)

    with pytest.raises(ImportError):
        build_debugger("/data/cfg.json", use_graph=False)
