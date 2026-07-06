"""
Cleanup Registry and Graceful Shutdown
=======================================
Production-grade process lifecycle management:
  - Global cleanup function registry with unregister support
  - Signal handlers (SIGINT, SIGTERM) with graceful shutdown
  - Failsafe timeout to prevent hang on cleanup
  - Resource tracking for daemon processes and temp files
"""

from __future__ import annotations

import atexit
import logging
import os
import signal
import sys
import threading
import time
from typing import Callable, Dict, Optional, Set

logger = logging.getLogger(__name__)


# 鈹€鈹€ Cleanup Registry 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€

class CleanupRegistry:
    """
    Global registry for cleanup functions that run on shutdown.

    Thread-safe. Functions are called in reverse registration order
    (LIFO 鈥?most recently registered runs first).

    Migrated from: upstream/src/utils/cleanupRegistry.ts

    Usage:
        from zhuangzhou.utils.cleanup import cleanup_registry

        # Register a cleanup function
        unregister = cleanup_registry.register(my_cleanup_fn)

        # Later, if you want to cancel:
        unregister()

        # On shutdown, all registered functions run automatically.
    """

    def __init__(self):
        self._functions: Dict[int, Callable] = {}
        self._lock = threading.Lock()
        self._next_id = 0
        self._shutting_down = False

    def register(self, fn: Callable[[], None]) -> Callable[[], None]:
        """
        Register a cleanup function.

        Returns an unregister function that removes this cleanup.
        """
        with self._lock:
            fn_id = self._next_id
            self._next_id += 1
            self._functions[fn_id] = fn

        def unregister():
            with self._lock:
                self._functions.pop(fn_id, None)

        return unregister

    def run_all(self, timeout: float = 10.0) -> list[str]:
        """
        Run all registered cleanup functions.

        Returns list of error messages from failed cleanups.
        Runs with a total timeout to prevent hanging.
        """
        with self._lock:
            if self._shutting_down:
                return []
            self._shutting_down = True
            # LIFO order
            fns = list(reversed(list(self._functions.values())))
            self._functions.clear()

        errors = []
        deadline = time.time() + timeout

        for fn in fns:
            if time.time() > deadline:
                errors.append(
                    f"Cleanup timeout ({timeout}s) 鈥?"
                    f"{len(fns) - len(errors)} functions skipped")
                break
            try:
                fn()
            except Exception as e:
                name = getattr(fn, '__name__', repr(fn))
                msg = f"Cleanup {name} failed: {e}"
                logger.warning(msg)
                errors.append(msg)

        self._shutting_down = False
        return errors

    @property
    def count(self) -> int:
        with self._lock:
            return len(self._functions)


# Global singleton
cleanup_registry = CleanupRegistry()


# 鈹€鈹€ Signal Handlers 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€

_original_handlers: Dict[int, Optional[signal.Handlers]] = {}
_shutdown_initiated = False
_shutdown_lock = threading.Lock()


def _graceful_shutdown(signum: int, frame) -> None:
    """
    Handle SIGINT/SIGTERM with graceful cleanup.

    Migrated from: upstream/src/utils/gracefulShutdown.ts
    """
    global _shutdown_initiated
    with _shutdown_lock:
        if _shutdown_initiated:
            # Second signal = force exit
            logger.warning("Second signal received, forcing exit")
            os._exit(128 + signum)
        _shutdown_initiated = True

    sig_name = signal.Signals(signum).name if signum in signal.Signals.__members__.values() else str(signum)
    logger.info("Received %s, running cleanup...", sig_name)

    # Run cleanup with failsafe timeout
    errors = cleanup_registry.run_all(timeout=10.0)
    if errors:
        for err in errors:
            logger.warning("Cleanup error: %s", err)

    # Call original handler if it was a callable
    original = _original_handlers.get(signum)
    if callable(original):
        original(signum, frame)
    else:
        sys.exit(128 + signum)


def install_signal_handlers() -> None:
    """
    Install graceful shutdown handlers for SIGINT and SIGTERM.

    Safe to call multiple times (idempotent).
    On Windows, only SIGINT is available (SIGTERM is not).

    Migrated from: upstream/src/utils/gracefulShutdown.ts
    """
    signals_to_handle = [signal.SIGINT]

    # SIGTERM not available on Windows
    if hasattr(signal, 'SIGTERM'):
        signals_to_handle.append(signal.SIGTERM)

    for sig in signals_to_handle:
        if sig not in _original_handlers:
            try:
                _original_handlers[sig] = signal.getsignal(sig)
                signal.signal(sig, _graceful_shutdown)
            except (OSError, ValueError):
                # Can't set signal handler (e.g., not main thread)
                pass


# 鈹€鈹€ atexit Integration 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€

def _atexit_cleanup():
    """Run cleanup on normal interpreter exit."""
    cleanup_registry.run_all(timeout=5.0)


atexit.register(_atexit_cleanup)


# 鈹€鈹€ Convenience Helpers 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€

def register_temp_file(filepath: str) -> Callable[[], None]:
    """
    Register a temp file for cleanup on shutdown.

    Returns unregister function.
    """
    def _cleanup():
        try:
            if os.path.exists(filepath):
                os.unlink(filepath)
        except OSError:
            pass

    return cleanup_registry.register(_cleanup)


def register_pid_file(filepath: str) -> Callable[[], None]:
    """
    Register a PID file: writes current PID, deletes on cleanup.

    Migrated from: upstream/src/utils/concurrentSessions.ts

    Returns unregister function.
    """
    import json as _json
    try:
        os.makedirs(os.path.dirname(filepath) or '.', exist_ok=True)
        with open(filepath, 'w', encoding='utf-8') as f:
            _json.dump({
                'pid': os.getpid(),
                'started_at': time.time(),
            }, f)
    except OSError as e:
        logger.warning("Cannot write PID file %s: %s", filepath, e)

    def _cleanup():
        try:
            if os.path.exists(filepath):
                os.unlink(filepath)
        except OSError:
            pass

    return cleanup_registry.register(_cleanup)


def is_pid_alive(pid: int) -> bool:
    """Check if a process with given PID is still running."""
    if sys.platform == 'win32':
        import ctypes
        kernel32 = ctypes.windll.kernel32
        handle = kernel32.OpenProcess(0x1000, False, pid)  # PROCESS_QUERY_LIMITED_INFORMATION
        if handle:
            kernel32.CloseHandle(handle)
            return True
        return False
    else:
        try:
            os.kill(pid, 0)
            return True
        except (OSError, ProcessLookupError):
            return False

