"""Keep OCR subprocesses tied to the desktop application's lifetime."""

from __future__ import annotations

import os
import signal
import subprocess
from threading import Lock


_lock = Lock()
_processes: set[subprocess.Popen] = set()


def register_process(process: subprocess.Popen) -> None:
    with _lock:
        _processes.add(process)


def unregister_process(process: subprocess.Popen) -> None:
    with _lock:
        _processes.discard(process)


def stop_active_processes() -> None:
    """Stop each active process group, including model subprocess children."""
    with _lock:
        processes = tuple(_processes)
    for process in processes:
        if process.poll() is not None:
            continue
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        try:
            process.wait(timeout=3)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.wait()
