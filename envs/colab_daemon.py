"""
Colab MCP Daemon — DEPRECATED (v0.9.0)
========================================
** This module is deprecated. Use the cloudflared HTTP tunnel approach instead. **
** See scripts/colab_gpu_server.py and envs/colab_sandbox.py (v0.9.0) **

Previous approach: colab-mcp MCP server → TCP daemon → ColabSandbox
Problem: Each colab-mcp startup opens a new browser tab, frequent reconnections
         trigger Google's abuse detection and may result in Colab access restrictions.

New approach (v0.9.0):
    1. User opens ONE Colab notebook, runs colab_gpu_server.py
    2. Server exposes FastAPI + cloudflared tunnel (single session, zero abuse risk)
    3. ColabSandbox connects via HTTP — no browser tabs, no daemon needed

This file is kept for backward compatibility only.
"""

import json
import logging
import os
import signal
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path

logger = logging.getLogger(__name__)

_USER_DIR = Path.home() / '.zhuangzhou'
_PID_FILE = _USER_DIR / 'colab_daemon.pid'
_PORT_FILE = _USER_DIR / 'colab_daemon.port'
DEFAULT_PORT = 19527


# ── Daemon Server ────────────────────────────────────────────────────────

class ColabDaemon:
    """TCP bridge between ColabSandbox clients and colab-mcp subprocess."""

    def __init__(self, port: int = DEFAULT_PORT):
        self.port = port
        self.mcp_process: subprocess.Popen = None
        self.server_socket: socket.socket = None
        self._lock = threading.Lock()
        self._running = False
        self._stderr_lines = []
        # Single active client — MCP stdout can only be consumed by one reader
        self._active_client: socket.socket = None
        self._active_client_lock = threading.Lock()

    def start_mcp(self) -> bool:
        """Start the colab-mcp subprocess."""
        from .colab_config import get_config
        config = get_config()

        cmd = [config.mcp_command] + config.mcp_args
        logger.info("Starting colab-mcp: %s", ' '.join(cmd))

        try:
            self.mcp_process = subprocess.Popen(
                cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                bufsize=0,
            )
            # Drain stderr in background
            t = threading.Thread(target=self._drain_stderr, daemon=True)
            t.start()
            logger.info("colab-mcp started (pid=%d)", self.mcp_process.pid)
            return True
        except Exception as e:
            logger.error("Failed to start colab-mcp: %s", e)
            return False

    def _wait_mcp_ready(self, timeout: int = 60) -> bool:
        """Wait for colab-mcp to finish starting and respond to initialize.

        This consumes the MCP initialize handshake at the daemon level,
        so clients don't need to worry about MCP startup latency.
        """
        logger.info("Waiting for colab-mcp to be ready...")
        import select

        # Send initialize request
        req = json.dumps({
            'jsonrpc': '2.0',
            'id': 0,
            'method': 'initialize',
            'params': {
                'protocolVersion': '2024-11-05',
                'capabilities': {},
                'clientInfo': {'name': 'zhuangzhou-daemon', 'version': '0.8.6'},
            },
        }) + '\n'

        deadline = time.time() + timeout
        # Wait until colab-mcp is ready to accept input
        while time.time() < deadline:
            if self.mcp_process.poll() is not None:
                logger.error("colab-mcp exited with code %d",
                             self.mcp_process.returncode)
                return False
            try:
                self.mcp_process.stdin.write(req.encode('utf-8'))
                self.mcp_process.stdin.flush()
                break
            except Exception:
                time.sleep(0.5)

        # Read response
        while time.time() < deadline:
            if self.mcp_process.poll() is not None:
                return False
            line = self.mcp_process.stdout.readline()
            if not line:
                time.sleep(0.1)
                continue
            line = line.decode('utf-8', errors='replace').strip()
            if not line:
                continue
            try:
                resp = json.loads(line)
                if resp.get('id') == 0 and 'result' in resp:
                    server_info = resp['result'].get('serverInfo', {})
                    logger.info("colab-mcp ready: %s v%s",
                                server_info.get('name', '?'),
                                server_info.get('version', '?'))
                    # Send initialized notification
                    notif = json.dumps({
                        'jsonrpc': '2.0',
                        'method': 'notifications/initialized',
                        'params': {},
                    }) + '\n'
                    self.mcp_process.stdin.write(notif.encode('utf-8'))
                    self.mcp_process.stdin.flush()
                    time.sleep(0.3)
                    return True
            except json.JSONDecodeError:
                pass

        logger.error("colab-mcp did not respond within %ds", timeout)
        return False

    def _drain_stderr(self):
        try:
            for line in self.mcp_process.stderr:
                text = line.decode('utf-8', errors='replace').strip()
                if text:
                    self._stderr_lines.append(text)
                    if len(self._stderr_lines) > 100:
                        self._stderr_lines = self._stderr_lines[-100:]
        except Exception:
            pass

    def _start_mcp_reader(self):
        """Single background thread: read MCP stdout → forward to active client."""
        def _reader():
            logger.info("MCP stdout reader started")
            try:
                while self._running and self.mcp_process.poll() is None:
                    line = self.mcp_process.stdout.readline()
                    if not line:
                        logger.error("colab-mcp stdout closed")
                        break
                    with self._active_client_lock:
                        client = self._active_client
                    if client:
                        try:
                            client.sendall(line)
                        except (BrokenPipeError, ConnectionResetError, OSError):
                            with self._active_client_lock:
                                self._active_client = None
            except Exception as e:
                logger.error("MCP reader error: %s", e)

        t = threading.Thread(target=_reader, daemon=True)
        t.start()

    def handle_client(self, client_sock: socket.socket, addr):
        """Handle a TCP client — single active client at a time.

        Only the client→MCP direction needs a per-client thread.
        MCP→client is handled by the global _start_mcp_reader thread.
        """
        logger.info("Client connected: %s", addr)
        client_sock.settimeout(None)

        # Set as active client (replaces previous if any)
        with self._active_client_lock:
            self._active_client = client_sock

        recv_buf = b''
        try:
            while self._running and self.mcp_process.poll() is None:
                chunk = client_sock.recv(65536)
                if not chunk:
                    logger.info("Client disconnected: %s", addr)
                    break
                recv_buf += chunk

                while b'\n' in recv_buf:
                    line, recv_buf = recv_buf.split(b'\n', 1)
                    line = line.strip()
                    if not line:
                        continue
                    with self._lock:
                        self.mcp_process.stdin.write(line + b'\n')
                        self.mcp_process.stdin.flush()

        except (ConnectionResetError, BrokenPipeError, OSError):
            pass
        except Exception as e:
            logger.error("Client handler error: %s", e)
        finally:
            with self._active_client_lock:
                if self._active_client is client_sock:
                    self._active_client = None
            try:
                client_sock.close()
            except Exception:
                pass
            logger.info("Client session ended: %s", addr)

    def run(self):
        """Start the daemon: launch colab-mcp + TCP server."""
        _USER_DIR.mkdir(parents=True, exist_ok=True)

        if not self.start_mcp():
            sys.exit(1)

        # Wait for colab-mcp to be ready by sending initialize handshake
        if not self._wait_mcp_ready():
            logger.error("colab-mcp failed to become ready")
            self.mcp_process.kill()
            sys.exit(1)

        self._running = True  # Must be set BEFORE starting reader thread

        # Start single MCP stdout reader thread
        self._start_mcp_reader()

        # Start TCP server
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            self.server_socket.bind(('127.0.0.1', self.port))
        except OSError as e:
            logger.error("Port %d already in use (daemon already running?): %s",
                         self.port, e)
            self._running = False
            self.mcp_process.kill()
            sys.exit(1)

        self.server_socket.listen(2)
        self.server_socket.settimeout(1.0)  # For clean shutdown

        # Write PID and port files
        _PID_FILE.write_text(str(os.getpid()), encoding='utf-8')
        _PORT_FILE.write_text(str(self.port), encoding='utf-8')

        logger.info("Colab daemon listening on 127.0.0.1:%d (pid=%d)",
                     self.port, os.getpid())
        print(f"Colab daemon running on 127.0.0.1:{self.port} (pid={os.getpid()})",
              flush=True)

        try:
            while self._running:
                if self.mcp_process.poll() is not None:
                    logger.error("colab-mcp process died (rc=%d)",
                                 self.mcp_process.returncode)
                    break
                try:
                    client, addr = self.server_socket.accept()
                    t = threading.Thread(target=self.handle_client,
                                         args=(client, addr), daemon=True)
                    t.start()
                except socket.timeout:
                    continue
        except KeyboardInterrupt:
            logger.info("Daemon shutting down (Ctrl+C)")
        finally:
            self.shutdown()

    def shutdown(self):
        """Clean shutdown."""
        self._running = False
        if self.server_socket:
            try:
                self.server_socket.close()
            except Exception:
                pass
        if self.mcp_process and self.mcp_process.poll() is None:
            try:
                self.mcp_process.stdin.close()
                self.mcp_process.wait(timeout=5)
            except Exception:
                self.mcp_process.kill()
        # Cleanup files
        for f in (_PID_FILE, _PORT_FILE):
            try:
                f.unlink(missing_ok=True)
            except Exception:
                pass
        logger.info("Daemon stopped")


# ── Client-side utilities (used by ColabSandbox) ──────────────────────────

def is_daemon_running() -> bool:
    """Check if the daemon process is alive (no TCP connect — avoids
    triggering handle_client which would steal the MCP stdout pipe)."""
    if not _PID_FILE.exists() or not _PORT_FILE.exists():
        return False
    try:
        pid = int(_PID_FILE.read_text().strip())
        from zhuangzhou.utils.cleanup import is_pid_alive
        if not is_pid_alive(pid):
            _cleanup_stale_files()
            return False
        return True
    except (ValueError, OSError):
        _cleanup_stale_files()
        return False


def get_daemon_port() -> int:
    """Get the port of the running daemon."""
    if _PORT_FILE.exists():
        try:
            return int(_PORT_FILE.read_text().strip())
        except ValueError:
            pass
    return DEFAULT_PORT


def start_daemon_background() -> bool:
    """Start the daemon as a background process."""
    if is_daemon_running():
        logger.info("Colab daemon already running on port %d", get_daemon_port())
        return True

    _USER_DIR.mkdir(parents=True, exist_ok=True)
    python = sys.executable

    # Start as background subprocess.
    # On Windows: CREATE_NEW_PROCESS_GROUP keeps child alive when parent exits,
    # but does NOT fully detach console (DETACHED_PROCESS breaks child pipe I/O).
    # Redirect daemon's own stdio to log file, NOT DEVNULL — the daemon's CHILD
    # (colab-mcp) uses subprocess.PIPE which is unaffected by parent stdio.
    _USER_DIR.mkdir(parents=True, exist_ok=True)
    log_path = str(_USER_DIR / 'colab_daemon.log')

    if sys.platform == 'win32':
        CREATE_NEW_PROCESS_GROUP = 0x00000200
        log_fh = open(log_path, 'w', encoding='utf-8')
        proc = subprocess.Popen(
            [python, '-m', 'zhuangzhou.envs.colab_daemon'],
            creationflags=CREATE_NEW_PROCESS_GROUP,
            stdout=log_fh,
            stderr=log_fh,
            stdin=subprocess.DEVNULL,
        )
    else:
        log_fh = open(log_path, 'w', encoding='utf-8')
        proc = subprocess.Popen(
            [python, '-m', 'zhuangzhou.envs.colab_daemon'],
            stdout=log_fh,
            stderr=log_fh,
            stdin=subprocess.DEVNULL,
            start_new_session=True,
        )

    # Wait for daemon to be ready
    for _ in range(30):
        time.sleep(0.5)
        if is_daemon_running():
            logger.info("Colab daemon started (pid=%d, port=%d)",
                         proc.pid, get_daemon_port())
            return True

    logger.error("Colab daemon failed to start within 15s")
    return False


def stop_daemon():
    """Stop the running daemon."""
    if not _PID_FILE.exists():
        return
    try:
        pid = int(_PID_FILE.read_text().strip())
        if sys.platform == 'win32':
            subprocess.run(['taskkill', '/F', '/PID', str(pid)],
                           capture_output=True)
        else:
            os.kill(pid, signal.SIGTERM)
        time.sleep(1)
    except (ValueError, ProcessLookupError, PermissionError):
        pass
    _cleanup_stale_files()


def _cleanup_stale_files():
    for f in (_PID_FILE, _PORT_FILE):
        try:
            f.unlink(missing_ok=True)
        except Exception:
            pass


# ── Main ──────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s [%(name)s] %(message)s')

    if '--background' in sys.argv:
        start_daemon_background()
    elif '--stop' in sys.argv:
        stop_daemon()
        print("Daemon stopped")
    elif '--status' in sys.argv:
        if is_daemon_running():
            print(f"Daemon running on port {get_daemon_port()}")
        else:
            print("Daemon not running")
    else:
        # Foreground mode
        daemon = ColabDaemon()
        daemon.run()
