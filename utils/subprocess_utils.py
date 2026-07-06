"""
Robust Subprocess Execution Utilities
=======================================
Production-grade subprocess management:
  - Non-throwing subprocess wrapper with structured results
  - Error message extraction priority chain
  - Cross-platform encoding handling
  - Timeout and signal management
"""

from __future__ import annotations

import logging
import os
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

logger = logging.getLogger(__name__)

# Default timeout: 10 minutes (matches LLM agent runtime)
DEFAULT_TIMEOUT_MS = 10 * 60 * 1000
DEFAULT_TIMEOUT_S = DEFAULT_TIMEOUT_MS // 1000

# Env var override for default timeout
_ENV_TIMEOUT = os.environ.get('ZZ_SUBPROCESS_TIMEOUT_S', '')


def _get_default_timeout() -> int:
    """Get default subprocess timeout, respecting env var override."""
    if _ENV_TIMEOUT:
        try:
            val = int(_ENV_TIMEOUT)
            if val > 0:
                return val
        except ValueError:
            pass
    return DEFAULT_TIMEOUT_S


@dataclass(frozen=True)
class SubprocessResult:
    """
    Structured result from subprocess execution.

    NEVER raises on non-zero exit 鈥?always returns a result object.
    Preserves stdout/stderr even on failure.

    Migrated from: upstream/src/utils/execFileNoThrow.ts
    """
    stdout: str = ""
    stderr: str = ""
    returncode: int = 0
    success: bool = True
    error: str = ""
    timed_out: bool = False
    signal_name: str = ""

    @property
    def output(self) -> str:
        """Combined stdout + stderr for convenience."""
        parts = []
        if self.stdout:
            parts.append(self.stdout)
        if self.stderr:
            parts.append(self.stderr)
        return '\n'.join(parts)

    def to_dict(self) -> dict:
        """Convert to dict for JSON bridge compatibility."""
        return {
            'success': self.success,
            'stdout': self.stdout,
            'stderr': self.stderr,
            'returncode': self.returncode,
            'error': self.error,
            'timed_out': self.timed_out,
        }


def _extract_error_message(
    result: subprocess.CompletedProcess,
    stderr: str,
    returncode: int,
) -> str:
    """
    Extract the most informative error message from a failed subprocess.

    Priority chain (most informative first):
    1. Signal name (if killed by signal)
    2. Last non-empty stderr line (most likely the actual error)
    3. Exit code

    Migrated from: upstream/src/utils/execFileNoThrow.ts :: getErrorMessage
    """
    # Check for signal
    if returncode < 0:
        try:
            import signal
            sig = signal.Signals(-returncode)
            return f"Killed by {sig.name}"
        except (ValueError, AttributeError):
            return f"Killed by signal {-returncode}"

    # Last meaningful stderr line
    if stderr:
        for line in reversed(stderr.strip().split('\n')):
            line = line.strip()
            if line and not line.startswith('Warning:'):
                return line[:500]

    return f"Process exited with code {returncode}"


def run_subprocess(
    cmd: Union[str, Sequence[str]],
    *,
    timeout: Optional[int] = None,
    cwd: Optional[Union[str, Path]] = None,
    env: Optional[Dict[str, str]] = None,
    input_data: Optional[Union[str, bytes]] = None,
    encoding: str = 'utf-8',
    shell: bool = False,
) -> SubprocessResult:
    """
    Execute a subprocess and return a structured result. NEVER raises.

    Preserves stdout/stderr even on failure. Handles encoding, timeouts,
    and signal-based termination gracefully.

    Migrated from: upstream/src/utils/execFileNoThrow.ts

    Args:
        cmd: Command to execute (string for shell=True, list otherwise)
        timeout: Timeout in seconds (default from env or 600)
        cwd: Working directory
        env: Environment variables (None = inherit current)
        input_data: Data to send to stdin
        encoding: Output encoding (default utf-8)
        shell: Run via shell (use with caution)

    Returns:
        SubprocessResult with stdout, stderr, returncode, success, error

    Usage:
        result = run_subprocess(['python', 'script.py', '--arg', 'value'])
        if result.success:
            print(result.stdout)
        else:
            print(f"Failed: {result.error}")
    """
    effective_timeout = timeout or _get_default_timeout()

    # Prepare input
    stdin_data = None
    if input_data is not None:
        if isinstance(input_data, str):
            stdin_data = input_data.encode(encoding)
        else:
            stdin_data = input_data

    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            timeout=effective_timeout,
            cwd=cwd,
            env=env,
            input=stdin_data,
            shell=shell,
        )

        stdout = proc.stdout.decode(encoding, errors='replace')
        stderr = proc.stderr.decode(encoding, errors='replace')

        if proc.returncode == 0:
            return SubprocessResult(
                stdout=stdout,
                stderr=stderr,
                returncode=0,
                success=True,
            )
        else:
            error_msg = _extract_error_message(proc, stderr, proc.returncode)
            signal_name = ''
            if proc.returncode < 0:
                try:
                    import signal as _sig
                    signal_name = _sig.Signals(-proc.returncode).name
                except (ValueError, AttributeError):
                    signal_name = f"SIG{-proc.returncode}"

            return SubprocessResult(
                stdout=stdout,
                stderr=stderr,
                returncode=proc.returncode,
                success=False,
                error=error_msg,
                signal_name=signal_name,
            )

    except subprocess.TimeoutExpired as e:
        stdout = (e.stdout or b'').decode(encoding, errors='replace')
        stderr = (e.stderr or b'').decode(encoding, errors='replace')
        return SubprocessResult(
            stdout=stdout,
            stderr=stderr,
            returncode=-1,
            success=False,
            error=f"Timeout after {effective_timeout}s",
            timed_out=True,
        )

    except FileNotFoundError:
        cmd_str = cmd if isinstance(cmd, str) else cmd[0]
        return SubprocessResult(
            success=False,
            returncode=-1,
            error=f"Command not found: {cmd_str}",
        )

    except PermissionError:
        cmd_str = cmd if isinstance(cmd, str) else cmd[0]
        return SubprocessResult(
            success=False,
            returncode=-1,
            error=f"Permission denied: {cmd_str}",
        )

    except OSError as e:
        return SubprocessResult(
            success=False,
            returncode=-1,
            error=f"OS error: {e}",
        )

    except Exception as e:
        return SubprocessResult(
            success=False,
            returncode=-1,
            error=f"Unexpected error: {e}",
        )


def run_python_subprocess(
    script: str,
    args: Optional[List[str]] = None,
    *,
    python: Optional[str] = None,
    timeout: Optional[int] = None,
    cwd: Optional[Union[str, Path]] = None,
    env: Optional[Dict[str, str]] = None,
    input_data: Optional[str] = None,
) -> SubprocessResult:
    """
    Run a Python script via subprocess with proper interpreter selection.

    Args:
        script: Path to Python script or inline code (with -c flag)
        args: Additional arguments to pass to the script
        python: Python interpreter path (default: sys.executable)
        timeout: Timeout in seconds
        cwd: Working directory
        env: Environment variables
        input_data: Stdin data (JSON payload for bridge scripts)
    """
    interpreter = python or sys.executable
    cmd = [interpreter, script] + (args or [])

    return run_subprocess(
        cmd,
        timeout=timeout,
        cwd=cwd,
        env=env,
        input_data=input_data,
    )


def run_inline_python(
    code: str,
    *,
    python: Optional[str] = None,
    timeout: Optional[int] = None,
    cwd: Optional[Union[str, Path]] = None,
    env: Optional[Dict[str, str]] = None,
) -> SubprocessResult:
    """
    Run inline Python code via subprocess (-c flag).

    Args:
        code: Python code string to execute
        python: Python interpreter path
        timeout: Timeout in seconds
        cwd: Working directory
        env: Environment variables
    """
    interpreter = python or sys.executable
    cmd = [interpreter, '-c', code]

    return run_subprocess(
        cmd,
        timeout=timeout,
        cwd=cwd,
        env=env,
    )

