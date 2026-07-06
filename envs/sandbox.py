"""
Environment Sandbox
====================
Subprocess-based isolation for different conda envs and WSL.

Design principle: NEVER import from foreign environments directly.
Instead, serialize inputs as JSON, call via subprocess, deserialize outputs.
This prevents DLL conflicts, version clashes, and CUDA context collisions.
"""

import json
import logging
import os
import shlex
import shutil
import subprocess
import sys
import tempfile
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# ── Environment Registry ─────────────────────────────────────────────────────

CONDA_ENVS = {
    'graphrag': {
        'path': os.environ.get('ZHUANGZHOU_GRAPHRAG_ENV', ''),
        'python': os.environ.get('ZHUANGZHOU_GRAPHRAG_PYTHON', 'python'),
        'purpose': 'AutoGraphRAG parquet queries, pandas, networkx',
    },
    'dplm2': {
        'path': os.environ.get('ZHUANGZHOU_DPLM2_ENV', ''),
        'python': os.environ.get('ZHUANGZHOU_DPLM2_PYTHON', 'python'),
        'purpose': 'UniMol, ESM-2, torch GPU, deep learning',
    },
    'ezs_env': {
        'path': os.environ.get('ZHUANGZHOU_EZS_ENV', ''),
        'python': os.environ.get('ZHUANGZHOU_EZS_PYTHON', 'python'),
        'purpose': 'Enzyme specificity GNN, torch_geometric, ESM-2, Grover',
    },
}


@dataclass
class SandboxResult:
    """Result from a sandboxed execution."""
    success: bool
    data: Any = None
    stdout: str = ""
    stderr: str = ""
    returncode: int = 0
    error: str = ""


class CondaSandbox:
    """
    Execute Python code in a specific conda environment via subprocess.

    Usage:
        sandbox = CondaSandbox('graphrag')
        result = sandbox.run_script('my_script.py', {'input': 'data'})
        result = sandbox.run_function('module.func', arg1=val1)
    """

    def __init__(self, env_name: str, timeout: int = 300):
        self.env_name = env_name
        self.timeout = timeout
        env_info = CONDA_ENVS.get(env_name, {})
        self.python_path = env_info.get('python', 'python')
        self._available = None

    @property
    def available(self) -> bool:
        if self._available is None:
            self._available = os.path.exists(self.python_path)
        return self._available

    def run_script(self, script_path: str, args: Dict[str, Any] = None,
                   timeout: int = None) -> SandboxResult:
        """Run a Python script in the conda environment."""
        if not self.available:
            return SandboxResult(
                success=False,
                error=f"Conda env '{self.env_name}' not found at {self.python_path}")

        cmd = [self.python_path, script_path]
        env = os.environ.copy()
        input_data = json.dumps(args or {}).encode() if args else None

        try:
            result = subprocess.run(
                cmd, capture_output=True, timeout=timeout or self.timeout,
                env=env, input=input_data)
            stdout = result.stdout.decode('utf-8', errors='replace')
            stderr = result.stderr.decode('utf-8', errors='replace')

            from zhuangzhou.utils.robustness import safe_json_parse

            if result.returncode == 0:
                data = safe_json_parse(stdout, default=stdout, log_error=False)
                return SandboxResult(success=True, data=data,
                                     stdout=stdout, stderr=stderr)
            else:
                return SandboxResult(
                    success=False, stdout=stdout, stderr=stderr,
                    returncode=result.returncode,
                    error=f"Script exited with code {result.returncode}")
        except subprocess.TimeoutExpired:
            return SandboxResult(success=False, error=f"Timeout after {timeout or self.timeout}s")
        except Exception as e:
            return SandboxResult(success=False, error=str(e))

    def run_inline(self, code: str, timeout: int = None) -> SandboxResult:
        """Run inline Python code in the conda environment."""
        if not self.available:
            return SandboxResult(
                success=False,
                error=f"Conda env '{self.env_name}' not found")

        cmd = [self.python_path, '-c', code]
        try:
            result = subprocess.run(
                cmd, capture_output=True, timeout=timeout or self.timeout)
            stdout = result.stdout.decode('utf-8', errors='replace')
            stderr = result.stderr.decode('utf-8', errors='replace')

            from zhuangzhou.utils.robustness import safe_json_parse, truncate_error

            if result.returncode == 0:
                data = safe_json_parse(stdout, default=stdout.strip(),
                                       log_error=False)
                return SandboxResult(success=True, data=data,
                                     stdout=stdout, stderr=stderr)
            else:
                return SandboxResult(
                    success=False, stdout=stdout, stderr=stderr,
                    returncode=result.returncode,
                    error=truncate_error(stderr, max_len=500))
        except subprocess.TimeoutExpired:
            return SandboxResult(
                success=False,
                error=f"Timeout after {timeout or self.timeout}s")
        except Exception as e:
            return SandboxResult(success=False, error=str(e))

    def run_json_bridge(self, bridge_script: str, payload: dict,
                        timeout: int = None) -> SandboxResult:
        """
        Run a bridge script that reads JSON from stdin, writes JSON to stdout.
        This is the primary IPC pattern for cross-env communication.
        """
        if not self.available:
            return SandboxResult(
                success=False,
                error=f"Conda env '{self.env_name}' not found")

        cmd = [self.python_path, bridge_script]
        input_bytes = json.dumps(payload).encode('utf-8')

        try:
            result = subprocess.run(
                cmd, input=input_bytes, capture_output=True,
                timeout=timeout or self.timeout)
            stdout = result.stdout.decode('utf-8', errors='replace')
            stderr = result.stderr.decode('utf-8', errors='replace')

            from zhuangzhou.utils.robustness import (
                safe_json_parse_last_line, truncate_error)

            if result.returncode == 0:
                data = safe_json_parse_last_line(
                    stdout, default={'raw_output': stdout})
                return SandboxResult(success=True, data=data,
                                     stdout=stdout, stderr=stderr)
            else:
                return SandboxResult(
                    success=False, stdout=stdout, stderr=stderr,
                    returncode=result.returncode,
                    error=truncate_error(stderr, max_len=500))
        except subprocess.TimeoutExpired:
            return SandboxResult(
                success=False,
                error=f"Timeout after {timeout or self.timeout}s")
        except Exception as e:
            return SandboxResult(success=False, error=str(e))

    def check_packages(self, packages: List[str]) -> Dict[str, bool]:
        """Check which packages are available in this env."""
        # Use importlib.util.find_spec per package — no exec() required
        lines = ['import importlib.util as _iu']
        for p in packages:
            # Validate package name: only allow alphanumeric, underscore, hyphen, dot
            import re as _re
            if not _re.match(r'^[A-Za-z0-9_.\-]+$', p):
                continue
            safe = p.replace('-', '_')  # importlib uses underscore form
            lines.append(
                f"print('{p}:OK' if _iu.find_spec('{safe}') is not None else '{p}:MISSING')"
            )
        code = '\n'.join(lines)
        result = self.run_inline(code, timeout=15)
        status = {}
        if result.success:
            for line in result.stdout.strip().split('\n'):
                if ':' in line:
                    pkg, st = line.strip().rsplit(':', 1)
                    status[pkg] = (st == 'OK')
        return status


class WSLSandbox:
    """Execute commands in WSL (Windows Subsystem for Linux)."""

    def __init__(self, timeout: int = 600):
        self.timeout = timeout
        self._available = None

    @property
    def available(self) -> bool:
        if self._available is None:
            self._available = shutil.which('wsl') is not None
        return self._available

    def run(self, command: str, timeout: int = None) -> SandboxResult:
        """Run a shell command in WSL."""
        if not self.available:
            return SandboxResult(success=False, error="WSL not available")

        cmd = ['wsl', 'bash', '-c', command]
        try:
            result = subprocess.run(
                cmd, capture_output=True, timeout=timeout or self.timeout)
            stdout = result.stdout.decode('utf-8', errors='replace')
            stderr = result.stderr.decode('utf-8', errors='replace')
            return SandboxResult(
                success=result.returncode == 0,
                data=stdout.strip(), stdout=stdout, stderr=stderr,
                returncode=result.returncode,
                error=stderr[:500] if result.returncode != 0 else "")
        except subprocess.TimeoutExpired:
            return SandboxResult(success=False, error="Timeout")
        except Exception as e:
            return SandboxResult(success=False, error=str(e))

    def run_python(self, script: str, args: dict = None,
                   timeout: int = None) -> SandboxResult:
        """Run a Python script in WSL.

        Args are passed via a temp JSON file (not shell embedding) to avoid
        injection risks from bash special characters in JSON values.
        Pattern borrowed from Snakemake's WSL bridge.
        """
        wsl_script = script.replace('\\', '/').replace('E:', '/mnt/e')
        if args:
            # Write JSON to temp file, pass path via env var (safe, no escaping needed)
            args_file = script + '.args.json'
            with open(args_file, 'w', encoding='utf-8') as f:
                json.dump(args, f, ensure_ascii=False)
            wsl_args_file = args_file.replace('\\', '/').replace('E:', '/mnt/e')
            cmd = f'ZZ_ARGS_FILE="{wsl_args_file}" python3 {wsl_script}'
        else:
            cmd = f"python3 {wsl_script}"
        return self.run(cmd, timeout)


class NativeSandbox:
    """Execute commands natively (current Python or system tools).

    Security: commands are validated against an allowlist of known-safe
    executables before execution (defense-in-depth against LLM injection).
    Inspired by Docker's seccomp + AppArmor command filtering pattern.
    """

    # Allowlisted command prefixes — only these executables may be invoked.
    # Modeled after Kubernetes pod security policies: explicit allow > implicit deny.
    ALLOWED_COMMANDS = {
        'gmx', 'gromacs', 'vina', 'obabel', 'acpype',          # molecular tools
        'python', 'python3', 'pip',                              # python
        'cat', 'echo', 'ls', 'dir', 'cp', 'copy', 'mkdir',     # filesystem (read/create)
        'head', 'tail', 'wc', 'grep', 'find', 'sort',          # text tools
        'sobtop', 'editconf', 'pdb2gmx', 'genion', 'mdrun',    # gromacs sub-tools
        'trjconv', 'grompp', 'make_ndx', 'solvate', 'genrestr', # gromacs sub-tools
        'rsync', 'scp', 'ssh',                                   # remote
    }

    def __init__(self, timeout: int = 300):
        self.timeout = timeout

    @property
    def available(self) -> bool:
        return True

    @classmethod
    def _validate_command(cls, command: str) -> bool:
        """Check command against allowlist. Returns True if safe."""
        # Extract the first token (executable name)
        import shlex
        try:
            # Handle quoted paths like '"C:/path/to/gmx" mdrun ...'
            stripped = command.strip().strip('"').strip("'")
            # Get basename of first token
            first_token = stripped.split()[0].strip('"').strip("'")
            exe_name = os.path.basename(first_token).lower()
            # Strip .exe suffix on Windows
            if exe_name.endswith('.exe'):
                exe_name = exe_name[:-4]
        except (IndexError, ValueError):
            return False
        return exe_name in cls.ALLOWED_COMMANDS

    def run(self, command: str, timeout: int = None) -> SandboxResult:
        """Run a system command (validated against allowlist)."""
        if not self._validate_command(command):
            exe = command.split()[0] if command.strip() else '(empty)'
            logger.warning("NativeSandbox blocked disallowed command: %s", exe)
            return SandboxResult(
                success=False,
                error=f"Command not in allowlist: {exe}. "
                      f"Allowed: {sorted(self.ALLOWED_COMMANDS)}")
        try:
            result = subprocess.run(
                shlex.split(command), capture_output=True,
                timeout=timeout or self.timeout)
            stdout = result.stdout.decode('utf-8', errors='replace')
            stderr = result.stderr.decode('utf-8', errors='replace')
            return SandboxResult(
                success=result.returncode == 0,
                data=stdout.strip(), stdout=stdout, stderr=stderr,
                returncode=result.returncode)
        except subprocess.TimeoutExpired:
            return SandboxResult(success=False, error="Timeout")
        except Exception as e:
            return SandboxResult(success=False, error=str(e))

    def run_python_func(self, module_path: str, func_name: str,
                        kwargs: dict = None) -> SandboxResult:
        """Run a Python function in the current interpreter."""
        try:
            import importlib
            mod = importlib.import_module(module_path)
            fn = getattr(mod, func_name)
            result = fn(**(kwargs or {}))
            return SandboxResult(success=True, data=result)
        except Exception as e:
            return SandboxResult(success=False, error=str(e))


class SSHSandbox:
    """
    Execute commands on a remote server via SSH.

    Architecture: local orchestration, remote computation.
    Uses system ssh/scp binaries (no paramiko dependency).
    Long-running jobs wrapped in tmux for disconnect resilience.

    Usage:
        sandbox = SSHSandbox('gpu_server')
        result = sandbox.run('gmx mdrun -deffnm md')
        result = sandbox.upload('/local/input.pdb', '/remote/work/')
        result = sandbox.download('/remote/work/md.xtc', '/local/output/')
    """

    # Cache SSH availability for 5 minutes (inspired by Consul health check TTL)
    _AVAILABILITY_TTL = 300  # seconds

    def __init__(self, host_name: str = 'default', timeout: int = None):
        from .ssh_config import get_host
        self.host_profile = get_host(host_name)
        self.host_name = host_name
        self.timeout = timeout
        self._available = None
        self._available_checked_at = 0.0

    @property
    def available(self) -> bool:
        # Re-check if cache expired (transient failures shouldn't be permanent)
        if (self._available is not None and
                (time.time() - self._available_checked_at) < self._AVAILABILITY_TTL):
            return self._available
        if self.host_profile is None:
            self._available = False
            self._available_checked_at = time.time()
            return False
        # Quick connectivity test (5s timeout)
        try:
            result = subprocess.run(
                self._ssh_base_cmd() + ['echo', 'zz_ok'],
                capture_output=True, timeout=10)
            self._available = (result.returncode == 0 and
                               b'zz_ok' in result.stdout)
        except Exception:
            self._available = False
        self._available_checked_at = time.time()
        return self._available

    def _ssh_base_cmd(self) -> List[str]:
        """Build base ssh command with connection options.

        ControlMaster multiplexing is only enabled on Linux/macOS where
        Unix domain sockets work reliably. Windows OpenSSH has persistent
        issues with stale ControlSocket files causing 'read from master failed'.
        """
        hp = self.host_profile
        cmd = [
            'ssh',
            '-o', 'StrictHostKeyChecking=accept-new',
            '-o', 'ConnectTimeout=10',
            '-o', 'ServerAliveInterval=30',
            '-o', 'ServerAliveCountMax=3',
            '-p', str(hp.port),
        ]
        # ControlMaster only on Unix (not Windows — stale sockets break all connections)
        if sys.platform != 'win32':
            import hashlib
            cp_hash = hashlib.md5(f'{hp.host}_{hp.port}_{hp.user}'.encode()).hexdigest()[:12]
            cmd.extend([
                '-o', 'ControlMaster=auto',
                '-o', f'ControlPath=~/.ssh/zz_{cp_hash}',
                '-o', 'ControlPersist=10m',
            ])
        key = hp.key_path_expanded
        if key:
            cmd.extend(['-i', key])
        cmd.append(f'{hp.user}@{hp.host}')
        return cmd

    def _scp_base_cmd(self) -> List[str]:
        """Build base scp command."""
        hp = self.host_profile
        cmd = [
            'scp',
            '-o', 'StrictHostKeyChecking=accept-new',
            '-o', 'ConnectTimeout=10',
            '-P', str(hp.port),
        ]
        if sys.platform != 'win32':
            import hashlib
            cp_hash = hashlib.md5(f'{hp.host}_{hp.port}_{hp.user}'.encode()).hexdigest()[:12]
            cmd.extend([
                '-o', 'ControlMaster=auto',
                '-o', f'ControlPath=~/.ssh/zz_{cp_hash}',
                '-o', 'ControlPersist=10m',
            ])
        key = hp.key_path_expanded
        if key:
            cmd.extend(['-i', key])
        return cmd

    def _remote_prefix(self) -> str:
        """SCP remote prefix: user@host:"""
        hp = self.host_profile
        return f'{hp.user}@{hp.host}:'

    def _wrap_command(self, command: str, work_dir: str = None,
                      use_conda: bool = True) -> str:
        """Wrap command with cd, conda activation, etc."""
        hp = self.host_profile
        parts = []

        # cd to work directory
        wd = work_dir or hp.remote_work_dir
        parts.append(f'mkdir -p {wd} && cd {wd}')

        # conda activation
        if use_conda and hp.conda_activate:
            parts.append(hp.conda_activate)

        parts.append(command)
        return ' && '.join(parts)

    def run(self, command: str, timeout: int = None,
            work_dir: str = None, use_conda: bool = True) -> SandboxResult:
        """Run a command on the remote server."""
        if not self.host_profile:
            return SandboxResult(
                success=False,
                error=f"SSH host '{self.host_name}' not configured. "
                      f"Create ~/.zhuangzhou/ssh_compute.ini")

        hp = self.host_profile
        full_cmd = self._wrap_command(command, work_dir, use_conda)
        ssh_cmd = self._ssh_base_cmd() + [full_cmd]
        effective_timeout = timeout or self.timeout or hp.timeout

        try:
            result = subprocess.run(
                ssh_cmd, capture_output=True, timeout=effective_timeout)
            stdout = result.stdout.decode('utf-8', errors='replace')
            stderr = result.stderr.decode('utf-8', errors='replace')
            return SandboxResult(
                success=result.returncode == 0,
                data=stdout.strip(), stdout=stdout, stderr=stderr,
                returncode=result.returncode,
                error=stderr[:500] if result.returncode != 0 else "")
        except subprocess.TimeoutExpired:
            return SandboxResult(
                success=False,
                error=f"SSH command timed out after {effective_timeout}s")
        except Exception as e:
            return SandboxResult(success=False, error=str(e))

    def run_detached(self, command: str, session_name: str = None,
                     work_dir: str = None, use_conda: bool = True) -> SandboxResult:
        """
        Run a long-running command in a tmux session (disconnect-resilient).

        Returns immediately with the session name. Use check_session() to poll.
        """
        if not self.host_profile:
            return SandboxResult(success=False,
                                 error=f"SSH host '{self.host_name}' not configured")

        session = session_name or f'zz_{uuid.uuid4().hex[:8]}'
        full_cmd = self._wrap_command(command, work_dir, use_conda)

        # tmux new-session -d -s <name> '<command>'
        # Escape single quotes in command: replace ' with '"'"'
        # This is the POSIX-safe way to embed single quotes inside single-quoted strings
        escaped_cmd = full_cmd.replace("'", "'\"'\"'")
        tmux_cmd = f"tmux new-session -d -s {session} '{escaped_cmd}'"
        result = self.run(tmux_cmd, timeout=30, use_conda=False)

        if result.success:
            result.data = {'session': session, 'host': self.host_name,
                           'status': 'running'}
        return result

    def check_session(self, session_name: str) -> SandboxResult:
        """Check if a tmux session is still running."""
        result = self.run(
            f"tmux has-session -t {session_name} 2>/dev/null && echo RUNNING || echo FINISHED",
            timeout=15, use_conda=False)
        if result.success:
            status = 'running' if 'RUNNING' in result.stdout else 'finished'
            result.data = {'session': session_name, 'status': status}
        return result

    def get_session_output(self, session_name: str,
                           lines: int = 100) -> SandboxResult:
        """Capture recent output from a tmux session."""
        cmd = f"tmux capture-pane -t {session_name} -p -S -{lines} 2>/dev/null"
        return self.run(cmd, timeout=15, use_conda=False)

    def kill_session(self, session_name: str) -> SandboxResult:
        """Kill a tmux session."""
        return self.run(f"tmux kill-session -t {session_name} 2>/dev/null",
                        timeout=15, use_conda=False)

    def upload(self, local_path: str, remote_path: str = None,
               recursive: bool = False) -> SandboxResult:
        """Upload file(s) to the remote server via scp."""
        if not self.host_profile:
            return SandboxResult(success=False,
                                 error=f"SSH host '{self.host_name}' not configured")

        hp = self.host_profile
        remote = remote_path or hp.remote_work_dir

        # Ensure remote directory exists
        self.run(f'mkdir -p {remote}', timeout=15, use_conda=False)

        cmd = self._scp_base_cmd()
        if recursive:
            cmd.append('-r')
        cmd.extend([local_path, f'{self._remote_prefix()}{remote}/'])

        try:
            result = subprocess.run(cmd, capture_output=True, timeout=600)
            stdout = result.stdout.decode('utf-8', errors='replace')
            stderr = result.stderr.decode('utf-8', errors='replace')
            return SandboxResult(
                success=result.returncode == 0,
                data={'uploaded': local_path, 'to': remote},
                stdout=stdout, stderr=stderr,
                returncode=result.returncode,
                error=stderr[:500] if result.returncode != 0 else "")
        except subprocess.TimeoutExpired:
            return SandboxResult(success=False, error="Upload timed out")
        except Exception as e:
            return SandboxResult(success=False, error=str(e))

    def download(self, remote_path: str, local_path: str,
                 recursive: bool = False) -> SandboxResult:
        """Download file(s) from the remote server via scp."""
        if not self.host_profile:
            return SandboxResult(success=False,
                                 error=f"SSH host '{self.host_name}' not configured")

        # Ensure local directory exists
        os.makedirs(local_path, exist_ok=True)

        cmd = self._scp_base_cmd()
        if recursive:
            cmd.append('-r')
        cmd.extend([f'{self._remote_prefix()}{remote_path}', local_path])

        try:
            result = subprocess.run(cmd, capture_output=True, timeout=600)
            stdout = result.stdout.decode('utf-8', errors='replace')
            stderr = result.stderr.decode('utf-8', errors='replace')
            return SandboxResult(
                success=result.returncode == 0,
                data={'downloaded': remote_path, 'to': local_path},
                stdout=stdout, stderr=stderr,
                returncode=result.returncode,
                error=stderr[:500] if result.returncode != 0 else "")
        except subprocess.TimeoutExpired:
            return SandboxResult(success=False, error="Download timed out")
        except Exception as e:
            return SandboxResult(success=False, error=str(e))

    def upload_files(self, local_files: List[str],
                     remote_dir: str = None) -> SandboxResult:
        """Upload multiple files to a remote directory."""
        remote = remote_dir or self.host_profile.remote_work_dir
        errors = []
        uploaded = []
        for f in local_files:
            r = self.upload(f, remote)
            if r.success:
                uploaded.append(f)
            else:
                errors.append(f"{f}: {r.error}")
        if errors:
            return SandboxResult(
                success=False,
                data={'uploaded': uploaded, 'failed': errors},
                error=f"Failed to upload {len(errors)} file(s)")
        return SandboxResult(success=True,
                             data={'uploaded': uploaded, 'remote_dir': remote})

    def download_files(self, remote_files: List[str],
                       local_dir: str) -> SandboxResult:
        """Download multiple files from remote."""
        os.makedirs(local_dir, exist_ok=True)
        errors = []
        downloaded = []
        for f in remote_files:
            r = self.download(f, local_dir)
            if r.success:
                downloaded.append(f)
            else:
                errors.append(f"{f}: {r.error}")
        if errors:
            return SandboxResult(
                success=False,
                data={'downloaded': downloaded, 'failed': errors},
                error=f"Failed to download {len(errors)} file(s)")
        return SandboxResult(success=True,
                             data={'downloaded': downloaded, 'local_dir': local_dir})

    def sync_to_remote(self, local_dir: str, remote_dir: str = None) -> SandboxResult:
        """Sync a local directory to remote using rsync over SSH (faster than scp for many files)."""
        if not self.host_profile:
            return SandboxResult(success=False, error="SSH host not configured")

        hp = self.host_profile
        remote = remote_dir or hp.remote_work_dir

        # Prefer rsync if available, fallback to scp -r
        if shutil.which('rsync'):
            rsh = f"ssh -p {hp.port} -o StrictHostKeyChecking=accept-new"
            key = hp.key_path_expanded
            if key:
                rsh += f" -i {key}"
            cmd = [
                'rsync', '-avz', '--progress',
                '-e', rsh,
                local_dir.rstrip('/\\') + '/',
                f'{hp.user}@{hp.host}:{remote}/'
            ]
        else:
            # No rsync on Windows: tar + single scp + remote extract (fast, one connection)
            import tempfile
            self.run(f'mkdir -p {remote}', timeout=15, use_conda=False)
            tar_path = os.path.join(tempfile.gettempdir(), '_zz_sync.tar.gz')
            try:
                import tarfile
                with tarfile.open(tar_path, 'w:gz') as tf:
                    for item in os.listdir(local_dir):
                        tf.add(os.path.join(local_dir, item), arcname=item)
                # Upload single tar
                r = self.upload(tar_path, remote)
                if not r.success:
                    return SandboxResult(success=False, error=f"Upload tar failed: {r.error}")
                # Extract on remote and clean up
                r = self.run(f'cd {remote} && tar xzf _zz_sync.tar.gz && rm _zz_sync.tar.gz',
                             timeout=120, use_conda=False)
                if not r.success:
                    return SandboxResult(success=False, error=f"Remote extract failed: {r.error}")
                return SandboxResult(success=True, data={'synced': local_dir, 'to': remote})
            finally:
                if os.path.exists(tar_path):
                    os.remove(tar_path)

        try:
            result = subprocess.run(cmd, capture_output=True, timeout=600)
            stdout = result.stdout.decode('utf-8', errors='replace')
            stderr = result.stderr.decode('utf-8', errors='replace')
            return SandboxResult(
                success=result.returncode == 0,
                data={'synced': local_dir, 'to': remote},
                stdout=stdout, stderr=stderr,
                returncode=result.returncode,
                error=stderr[:500] if result.returncode != 0 else "")
        except subprocess.TimeoutExpired:
            return SandboxResult(success=False, error="Sync timed out")
        except Exception as e:
            return SandboxResult(success=False, error=str(e))

    def sync_from_remote(self, remote_dir: str, local_dir: str) -> SandboxResult:
        """Sync a remote directory to local using rsync."""
        if not self.host_profile:
            return SandboxResult(success=False, error="SSH host not configured")

        hp = self.host_profile
        os.makedirs(local_dir, exist_ok=True)

        if shutil.which('rsync'):
            rsh = f"ssh -p {hp.port} -o StrictHostKeyChecking=accept-new"
            key = hp.key_path_expanded
            if key:
                rsh += f" -i {key}"
            cmd = [
                'rsync', '-avz', '--progress',
                '-e', rsh,
                f'{hp.user}@{hp.host}:{remote_dir.rstrip("/")}/',
                local_dir.rstrip('/\\') + '/'
            ]
        else:
            # No rsync on Windows: remote tar + single scp download + local extract
            import tempfile, tarfile
            os.makedirs(local_dir, exist_ok=True)
            tar_name = '_zz_sync_dl.tar.gz'
            r = self.run(f'cd {remote_dir} && tar czf /tmp/{tar_name} .',
                         timeout=300, use_conda=False)
            if not r.success:
                return SandboxResult(success=False, error=f"Remote tar failed: {r.error}")
            # Download tar to temp dir (download() creates local_path as dir)
            tmp_dir = tempfile.mkdtemp(prefix='zz_dl_')
            r = self.download(f'/tmp/{tar_name}', tmp_dir)
            local_tar = os.path.join(tmp_dir, tar_name)
            if not r.success or not os.path.isfile(local_tar):
                shutil.rmtree(tmp_dir, ignore_errors=True)
                return SandboxResult(success=False, error=f"Download tar failed: {r.error}")
            try:
                with tarfile.open(local_tar, 'r:gz') as tf:
                    tf.extractall(local_dir)
                self.run(f'rm /tmp/{tar_name}', timeout=10, use_conda=False)
                return SandboxResult(success=True, data={'synced': remote_dir, 'to': local_dir})
            finally:
                shutil.rmtree(tmp_dir, ignore_errors=True)

        try:
            result = subprocess.run(cmd, capture_output=True, timeout=600)
            stdout = result.stdout.decode('utf-8', errors='replace')
            stderr = result.stderr.decode('utf-8', errors='replace')
            return SandboxResult(
                success=result.returncode == 0,
                data={'synced': remote_dir, 'to': local_dir},
                stdout=stdout, stderr=stderr,
                returncode=result.returncode,
                error=stderr[:500] if result.returncode != 0 else "")
        except subprocess.TimeoutExpired:
            return SandboxResult(success=False, error="Sync timed out")
        except Exception as e:
            return SandboxResult(success=False, error=str(e))


# ── Singleton Factories ──────────────────────────────────────────────────────

_sandboxes: Dict[str, Any] = {}


def get_sandbox(env_type: str) -> Any:
    """
    Get a sandbox by type.

    Args:
        env_type: 'graphrag', 'dplm2', 'ezs_env', 'wsl', 'native',
                  or 'ssh:<host_name>' (e.g. 'ssh:gpu_server', 'ssh:default')
    """
    if env_type not in _sandboxes:
        if env_type in CONDA_ENVS:
            _sandboxes[env_type] = CondaSandbox(env_type)
        elif env_type == 'wsl':
            _sandboxes[env_type] = WSLSandbox()
        elif env_type == 'native':
            _sandboxes[env_type] = NativeSandbox()
        elif env_type.startswith('ssh:'):
            host_name = env_type.split(':', 1)[1]
            _sandboxes[env_type] = SSHSandbox(host_name)
        elif env_type == 'ssh':
            _sandboxes[env_type] = SSHSandbox('default')
        elif env_type == 'colab':
            from .colab_sandbox import ColabSandbox
            _sandboxes[env_type] = ColabSandbox()
        else:
            raise ValueError(f"Unknown sandbox type: {env_type}")
    return _sandboxes[env_type]


def get_ssh_sandbox(host_name: str = 'default') -> SSHSandbox:
    """Convenience: get an SSH sandbox by host name."""
    return get_sandbox(f'ssh:{host_name}')


def check_all_environments() -> Dict[str, Dict[str, Any]]:
    """Check availability of all sandboxes (including SSH hosts)."""
    status = {}
    for name, info in CONDA_ENVS.items():
        sb = get_sandbox(name)
        status[name] = {
            'available': sb.available,
            'python': info['python'],
            'purpose': info['purpose'],
        }
    wsl = get_sandbox('wsl')
    status['wsl'] = {
        'available': wsl.available,
        'purpose': 'PLIP, Linux bioinformatics tools',
    }
    status['native'] = {
        'available': True,
        'purpose': 'Vina, GROMACS, basic Python',
    }

    # Colab (v0.9.0: cloudflared HTTP tunnel)
    try:
        from .colab_sandbox import ColabSandbox
        colab = ColabSandbox()
        status['colab'] = {
            'available': colab.available,
            'tunnel_url': colab.tunnel_url or '(not configured)',
            'purpose': 'Google Colab GPU (T4/L4) via cloudflared tunnel — ESM-2, ESM-1v',
        }
    except Exception:
        status['colab'] = {'available': False, 'purpose': 'Google Colab GPU'}

    # SSH hosts
    from .ssh_config import list_hosts
    for host in list_hosts():
        key = f'ssh:{host.name}'
        sb = get_sandbox(key)
        status[key] = {
            'available': sb.available,
            'host': host.host,
            'user': host.user,
            'capabilities': host.capabilities,
            'purpose': f'Remote compute ({", ".join(host.capabilities)})',
        }

    return status
