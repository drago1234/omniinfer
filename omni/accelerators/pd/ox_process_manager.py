import subprocess
import threading
import queue
import time
import os
import re
import signal
from pathlib import Path
from typing import Callable, List, Optional

from vllm.logger import init_logger
logger = init_logger(__name__)
_NATIVE_LOG_LEVEL = re.compile(r"^\S+\s+(DEBUG|INFO|WARN|ERROR)\s")
_NATIVE_STACK_FRAME = re.compile(r"^#\d+\s")
_NATIVE_UNSTRUCTURED_ERROR = re.compile(r"\b(error|exception|failed|fatal|unknown argument)\b", re.IGNORECASE)
_PROCESS_CREATION_LOCK = threading.RLock()


def start_connector_process(process):
    """Do not let a fork-only helper inherit Popen's exec-error pipe.

    OX and helpers are restarted from different threads in the same parent.
    A helper that forks during Popen can retain its error-pipe writer, leaving
    Popen waiting for EOF even after the native executable started or died.
    """
    with _PROCESS_CREATION_LOCK:
        process.start()


def isolate_connector_child_signals():
    """Detach a forked helper from its parent's asyncio signal pipe.

    A child SIGTERM otherwise writes to the inherited wakeup fd and can make
    the API server run its own shutdown callback. Keep the inherited Python
    handlers: multiprocessing catches SystemExit and runs its queue cleanup.
    Only helpers call this, from their main thread, before starting any work.
    """
    signal.set_wakeup_fd(-1)


def close_retired_queue(q, timeout=3):
    """Dispose Linux multiprocessing IPC only after ALL peers are reaped.

    A killed reader may poison Queue.get's lock or leave a partial frame.
    Discard raw bytes rather than decoding frames or acquiring that lock.
    This also lets the parent's feeder finish instead of leaking a thread
    when another inherited fd keeps the retired pipe open.
    """
    q.cancel_join_thread()
    q.close()
    feeder = getattr(q, '_thread', None)
    if feeder is not None:
        deadline = time.monotonic() + timeout
        while feeder.is_alive():
            if time.monotonic() >= deadline:
                raise RuntimeError('retired OX queue feeder did not finish')
            try:
                if q._reader.poll(0.02):
                    os.read(q._reader.fileno(), 65536)
            except (OSError, ValueError):
                feeder.join(timeout=0.05)
                if feeder.is_alive():
                    raise
        feeder.join()
    q._reader.close()
    q._writer.close()


class OxProcessManager:
    def __init__(self, log_prefix: str = "ox"):
        self.process: Optional[subprocess.Popen] = None
        self.monitor_thread: Optional[threading.Thread] = None
        self._log_reader_thread: Optional[threading.Thread] = None
        self._log_printer_thread: Optional[threading.Thread] = None
        self.log_prefix = log_prefix
        self._startup_wait_seconds = 0.5
        self._stop = threading.Event()
        self._cmd = None
        self._log_file_path = None
        self._ready_check: Optional[Callable[[], bool]] = None
        self._before_restart: Optional[Callable[[], None]] = None

    def start(self, cmd: List[str], log_file_path: Optional[str] = None,
              ready_check: Optional[Callable[[], bool]] = None,
              before_restart: Optional[Callable[[], None]] = None) -> subprocess.Popen:
        self._check_executable(cmd[0])
        if self.monitor_thread and self.monitor_thread.is_alive():
            raise RuntimeError("OX process manager is already running")
        self._cmd = list(cmd)
        self._log_file_path = log_file_path
        self._ready_check = ready_check
        self._before_restart = before_restart
        self._stop.clear()
        proc = self._launch_and_check()
        self.process = proc
        self._start_monitor(proc)
        return proc

    def _launch_and_check(self):
        proc = self._lauch_process(self._cmd)
        # Begin draining before readiness checks, so a chatty OX cannot block
        # while its stdout pipe fills up.
        self._start_log_reader(proc, self._log_file_path)
        time.sleep(self._startup_wait_seconds)
        if proc.poll() is not None:
            self._join_log_threads()
            raise RuntimeError(f"OX process exited during startup: {proc.returncode}")
        if self._ready_check and not self._ready_check():
            self._terminate(proc)
            self._join_log_threads()
            raise RuntimeError("OX process did not become ready")
        if proc.poll() is not None:
            self._join_log_threads()
            raise RuntimeError(f"OX process exited during startup: {proc.returncode}")
        return proc

    def _check_executable(self, executable: str):
        import shutil
        exe_path = Path(executable)
        if not exe_path.is_absolute():
            exe_path = shutil.which(executable)
            if exe_path is None:
                raise FileNotFoundError(f"Executable files are not in the PATH: {executable}")
            exe_path = Path(exe_path)
        if not exe_path.exists():
            raise FileNotFoundError(f"Executable file does not exist: {exe_path}")
        if not os.access(exe_path, os.X_OK):
            raise PermissionError(f"Executable file does not have the execute permission: {exe_path}")

    def _lauch_process(self, cmd: List[str]) -> subprocess.Popen:
        try:
            with _PROCESS_CREATION_LOCK:
                return subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    errors="replace",
                    bufsize=1
                )
        except Exception as e:
            logger.error(f"[{self.log_prefix}] startup failed: {e}")
            raise RuntimeError(f"OX process startup failed") from e

    def _check_immediate_exit(self, proc: subprocess.Popen):
        time.sleep(self._startup_wait_seconds)
        if proc.poll() is not None:
            stdout, _ = proc.communicate()
            logger.error(f"[{self.log_prefix}] process exit immediately after being started. Exit code: {proc.returncode}")
            if stdout:
                for line in stdout.splitlines():
                    self._forward_ox_line(line)
            raise RuntimeError(f"OX process exit immediately after being started.")

    def _start_log_reader(self, proc: subprocess.Popen, log_file_path: Optional[str] = None):
        q = queue.Queue()
        t_read = threading.Thread(target=self._stdout_reader, args=(proc.stdout, q), daemon=True)
        if log_file_path:
            t_print = threading.Thread(target=self._stdout_printer_with_file, args=(q, log_file_path), daemon=True)
        else:
            t_print = threading.Thread(target=self._stdout_printer, args=(q,), daemon=True)
        t_read.start()
        t_print.start()
        self._log_reader_thread = t_read
        self._log_printer_thread = t_print
    
    def _start_monitor(self, proc: subprocess.Popen):
        def monitor():
            nonlocal proc
            failures = 0
            while not self._stop.is_set():
                # A stopped or zombie process is unusable even if poll() has
                # not yet reported an exit.
                state = self._linux_process_state(proc.pid)
                if state in ("T", "t", "Z"):
                    logger.error("[%s] OX process is in state %s", self.log_prefix, state)
                    self._terminate(proc)
                code = proc.poll()
                if code is None:
                    if self._stop.wait(1):
                        break
                    continue
                self._join_log_threads()
                if self._stop.is_set():
                    break
                logger.error("[%s] OX exited unexpectedly: %s; restarting", self.log_prefix, code)
                while not self._stop.is_set():
                    if self._stop.wait(min(failures + 1, 5)):
                        break
                    try:
                        if self._before_restart:
                            self._before_restart()
                        proc = self._launch_and_check()
                    except Exception:
                        failures += 1
                        logger.exception("[%s] OX restart attempt %s failed", self.log_prefix, failures)
                        continue
                    if self._stop.is_set():
                        self._terminate(proc)
                        self._join_log_threads()
                        break
                    self.process = proc
                    failures = 0
                    logger.warning("[%s] OX restarted, pid=%s", self.log_prefix, proc.pid)
                    break

        self.monitor_thread = threading.Thread(target=monitor, daemon=True)
        self.monitor_thread.start()

    @staticmethod
    def _linux_process_state(pid):
        try:
            # comm may contain spaces or parentheses.
            return Path(f"/proc/{pid}/stat").read_text().rsplit(") ", 1)[1][0]
        except (OSError, IndexError):
            return None

    def _join_log_threads(self):
        if self._log_reader_thread:
            self._log_reader_thread.join(timeout=3)
        if self._log_printer_thread:
            self._log_printer_thread.join(timeout=3)

    @staticmethod
    def _terminate(proc):
        if proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()
    
    @staticmethod
    def _stdout_reader(pipe, q):
        try:
            for line in iter(pipe.readline, ''):
                if line:
                    q.put(line)
        except Exception as exc:
            logger.error("OX stdout reader failed: %s", exc)
        finally:
            pipe.close()
            q.put(None)
    
    def _forward_ox_line(self, line):
        message = line.rstrip("\r\n")
        match = _NATIVE_LOG_LEVEL.match(message)
        if match:
            if match.group(1) == "ERROR":
                logger.error("[%s] %s", self.log_prefix, message)
        elif ("Assertion failed:" in message or _NATIVE_STACK_FRAME.match(message)
              or _NATIVE_UNSTRUCTURED_ERROR.search(message)):
            logger.error("[%s] %s", self.log_prefix, message)

    def _stdout_printer(self, q):
        while True:
            line = q.get()
            if line is None:
                break
            self._forward_ox_line(line)
    
    def _stdout_printer_with_file(self, q, log_file_path):
        try:
            os.makedirs(os.path.dirname(log_file_path), exist_ok=True)
            log_file = open(log_file_path, 'a')
        except OSError as exc:
            logger.error("[%s] Cannot write OX log file %s: %s", self.log_prefix, log_file_path, exc)
            log_file = None
        try:
            while True:
                line = q.get()
                if line is None:
                    break
                if log_file:
                    try:
                        log_file.write(line)
                        log_file.flush()
                    except OSError as exc:
                        logger.error("[%s] OX log file write failed: %s", self.log_prefix, exc)
                        log_file.close()
                        log_file = None
                self._forward_ox_line(line)
        finally:
            if log_file:
                log_file.close()
    
    def stop(self):
        self._stop.set()
        if self.process:
            self._terminate(self.process)
        if self.monitor_thread and self.monitor_thread is not threading.current_thread():
            self.monitor_thread.join(timeout=6)
