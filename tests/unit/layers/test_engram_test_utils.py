# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""Deadlines must kill every rank, and failures must retain child diagnostics."""

from pathlib import Path
import subprocess
import sys
import time
from types import SimpleNamespace

import pytest

from tests.integration.layers import engram_test_utils as runner


@pytest.mark.parametrize("mode,interpreter", [
    ("success", False), ("failure", False), ("timeout", False), ("success", True),
])
def test_isolated_runner_reports_results_and_kills_descendants(tmp_path, monkeypatch, mode, interpreter):
    real_popen = subprocess.Popen
    # Exercise real process-group cleanup without loading NPU code or coverage.
    monkeypatch.setattr(runner, "sys", SimpleNamespace(executable=sys.executable, modules={}))
    script = (
        "import subprocess, sys, time\nfrom pathlib import Path\n"
        "child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)'])\n"
        "Path(sys.argv[1]).write_text(str(child.pid))\n"
        "print('rank 0: reached graph replay', flush=True)\n"
        + ("time.sleep(60)\n" if mode == "timeout" else f"sys.exit({7 if mode == 'failure' else 0})\n")
    )

    def launch(command, **kwargs):
        assert kwargs["start_new_session"]
        if interpreter:
            assert str(tmp_path / "interpreter_tests.py") in command
            assert "InterpretedFunction" in (tmp_path / "interpreter_tests.py").read_text()
        else:
            assert "pytest" in command
        assert kwargs["env"]["OMNI_ENGRAM_TEST_CHILD"] == "1"
        return real_popen([sys.executable, "-c", script, str(tmp_path / "rank.pid")], **kwargs)

    monkeypatch.setattr(runner.subprocess, "Popen", launch)
    if mode == "success":
        runner.run_isolated_pytest("engram_case", tmp_path, timeout=3, interpreter=interpreter)
    else:
        expected = "timed out after 1s" if mode == "timeout" else "exited with code 7"
        with pytest.raises(pytest.fail.Exception, match=expected) as error:
            runner.run_isolated_pytest("engram_case", tmp_path, timeout=1, interpreter=interpreter)
        assert "rank 0: reached graph replay" in str(error.value)
    pid = int((tmp_path / "rank.pid").read_text())
    status = Path(f"/proc/{pid}/stat")

    def alive():
        try:
            return status.read_text().split()[2] not in {"Z", "X"}
        except FileNotFoundError:
            return False

    # Signal delivery is asynchronous; PID 1 may also retain a dead zombie.
    deadline = time.monotonic() + 2
    while alive() and time.monotonic() < deadline:
        time.sleep(0.01)
    assert not alive()
