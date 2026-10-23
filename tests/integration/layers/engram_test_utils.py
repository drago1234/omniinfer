# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""Run Engram regressions with a deadline and isolated compiler/device state."""

import os
from pathlib import Path
import signal
import subprocess
import sys

import pytest


def run_isolated_pytest(target, tmp_path, timeout=300, *, interpreter=False):
    """Bound compilation and collectives, including all spawned rank processes."""
    env = dict(os.environ, OMNI_ENGRAM_TEST_CHILD="1", PYTEST_DISABLE_PLUGIN_AUTOLOAD="1",
               OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
    entrypoint = ["-m", "pytest"]
    if interpreter:
        # Load core Triton before pytest/conftest imports vLLM. Optional CANN
        # discovery can otherwise fail and leave a partially initialized module.
        bootstrap = tmp_path / "interpreter_tests.py"
        bootstrap.write_text(
            "if __name__ == '__main__':\n"
            "    import importlib.util, sys\n"
            "    from pathlib import Path\n"
            "    from types import ModuleType\n"
            "    root = Path(importlib.util.find_spec('triton').origin).parent\n"
            "    extra = ModuleType('triton.language.extra')\n"
            "    extra.__path__ = [str(root / 'language/extra')]\n"
            "    sys.modules[extra.__name__] = extra\n"
            "    from triton.runtime.interpreter import InterpretedFunction\n"
            "    import pytest\n"
            "    raise SystemExit(pytest.main(sys.argv[1:]))\n"
        )
        entrypoint = [str(bootstrap)]
    command = [sys.executable] + entrypoint
    coverage_module = sys.modules.get("coverage")
    parent_coverage = coverage_module.Coverage.current() if coverage_module else None
    if parent_coverage is not None:
        # Keep spawned-rank coverage as well as the child pytest process.
        source = Path(__file__).resolve().parents[3] / "omni"
        config = tmp_path / "engram.coveragerc"
        config.write_text(
            "[run]\nbranch = " + str(parent_coverage.config.branch)
            + "\nparallel = True\nconcurrency = multiprocessing,thread\ninclude =\n    " + str(source / "*") + "\n"
        )
        child_data = tmp_path / ".coverage.engram"
        env["COVERAGE_FILE"] = str(child_data)
        command = [sys.executable, "-m", "coverage", "run", "--rcfile=" + str(config),
                   "--data-file=" + str(child_data)] + entrypoint

    command += [target, "-vv", "-s", "-x", "--tb=short", "--junitxml=" + str(tmp_path / "engram.xml")]
    log_path = tmp_path / "engram.log"
    timed_out = False
    with log_path.open("w") as log:
        process = subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        try:
            returncode = process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            timed_out = True
        finally:
            # Killing only pytest would leave torch.multiprocessing ranks alive.
            # Also clean up surviving descendants after an assertion failure.
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.wait(timeout=5)

    if timed_out or returncode != 0:
        status = f"timed out after {timeout}s" if timed_out else f"exited with code {returncode}"
        pytest.fail(f"{target} {status}. Log: {log_path}\n{log_path.read_text()}", pytrace=False)
    if parent_coverage is not None:
        child_coverage = coverage_module.Coverage(data_file=str(child_data))
        child_coverage.combine(data_paths=[str(tmp_path)], strict=True)
        parent_coverage.get_data().update(child_coverage.get_data())
