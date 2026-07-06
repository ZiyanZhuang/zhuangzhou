"""
Google Colab Sandbox (v0.9.0 — HTTP Tunnel Mode)
=================================================
Connects to a Colab GPU server via cloudflared HTTP tunnel.

Architecture:
    ColabSandbox ──HTTP POST──> cloudflared tunnel ──> FastAPI (Colab GPU)
                                                        ├── /esm_scan
                                                        ├── /esm_consensus_scan
                                                        ├── /gpu_info
                                                        ├── /execute
                                                        └── /health

Setup (choose one):
    A. Automatic (recommended):
        sandbox = ColabSandbox()
        sandbox.auto_setup()   # Uses colab-mcp to bootstrap, opens ONE browser tab

    B. Manual:
        1. Open Colab notebook, run colab_gpu_server.py / .ipynb
        2. Copy tunnel URL to config or env var
        3. sandbox.connect()

Usage:
    sandbox = ColabSandbox()
    sandbox.auto_setup()                 # One-click: bootstrap + connect
    result = sandbox.gpu_info()          # Check GPU
    result = sandbox.esm_scan(...)       # Run ESM-2 mutation scan
    result = sandbox.execute_code(code)  # Arbitrary code execution
"""

import json
import logging
import os
import time
from typing import Any, Dict, List, Optional
from urllib import request as urllib_request
from urllib.error import URLError, HTTPError

from .sandbox import SandboxResult

logger = logging.getLogger(__name__)


class ColabSandbox:
    """
    Execute Python code on Google Colab GPU via HTTP tunnel.

    Connection goes through a cloudflared tunnel to a FastAPI server
    running in a single Colab notebook. No new browser tabs are opened.

    Usage:
        sandbox = ColabSandbox()
        result = sandbox.gpu_info()
        result = sandbox.esm_scan(sequence='MKTL...', top_k=20)
    """

    def __init__(self, timeout: int = None):
        from .colab_config import get_config
        self.config = get_config()
        self.timeout = timeout or self.config.timeout
        self._connected = False
        self._tunnel_url = self.config.tunnel_url

    # ── Lifecycle ─────────────────────────────────────────────────────────

    @property
    def available(self) -> bool:
        """Check if tunnel URL is configured."""
        return bool(self._tunnel_url)

    @property
    def connected(self) -> bool:
        return self._connected

    @property
    def tunnel_url(self) -> str:
        return self._tunnel_url

    @tunnel_url.setter
    def tunnel_url(self, url: str):
        self._tunnel_url = url.rstrip('/') if url else ''
        self._connected = False

    def connect(self, timeout: int = 10) -> SandboxResult:
        """
        Verify the Colab GPU server is reachable.

        Does NOT open any browser tabs — just an HTTP health check.
        """
        if not self._tunnel_url:
            return SandboxResult(
                success=False,
                error="No tunnel URL configured. "
                      "Set ZZ_COLAB_TUNNEL env var or add tunnel_url to "
                      "~/.zhuangzhou/colab_config.json. "
                      "See scripts/colab_gpu_server.py for setup instructions.")

        result = self._http_get('/health', timeout=timeout)
        if result.success:
            self._connected = True
            data = result.data if isinstance(result.data, dict) else {}
            logger.info("Connected to Colab GPU server: gpu=%s, uptime=%.0fs",
                        data.get('gpu_name', '?'),
                        data.get('uptime_seconds', 0))
        return result

    def auto_setup(self, timeout: int = 180) -> SandboxResult:
        """
        Fully automatic Colab setup: bootstrap + connect.

        Uses colab-mcp to open ONE browser tab, inject server code,
        start GPU server + cloudflared tunnel, and retrieve the URL.

        After the first successful setup, the tunnel URL is cached in
        ~/.zhuangzhou/colab_config.json. Subsequent calls only need
        connect() unless the Colab runtime has been disconnected.

        Args:
            timeout: Total timeout for bootstrap process (seconds)

        Returns:
            SandboxResult with success status and tunnel URL in data
        """
        # Check if already connected
        if self._connected:
            return SandboxResult(
                success=True,
                data={'tunnel_url': self._tunnel_url, 'status': 'already connected'})

        # Check if existing URL still works
        if self._tunnel_url:
            result = self.connect(timeout=5)
            if result.success:
                return SandboxResult(
                    success=True,
                    data={'tunnel_url': self._tunnel_url, 'status': 'reconnected'})
            logger.info("Existing tunnel URL no longer reachable, re-bootstrapping...")

        # Run bootstrap
        from .colab_bootstrap import bootstrap_colab
        tunnel_url = bootstrap_colab(timeout=timeout)

        if not tunnel_url:
            return SandboxResult(
                success=False,
                error="Colab bootstrap failed. Check ~/.zhuangzhou/colab_bootstrap.log "
                      "for details. Ensure 'uv' is installed (pip install uv).")

        self._tunnel_url = tunnel_url
        self._connected = False  # Will be set by connect()

        # Verify server is reachable — retry with increasing delays because
        # cloudflared tunnel needs 10-60s to fully establish SSL after URL appears
        delays = [3, 5, 5, 10, 10, 15, 15, 20]  # total ~83s max wait
        for attempt, delay in enumerate(delays):
            time.sleep(delay)
            result = self.connect(timeout=10)
            if result.success:
                return SandboxResult(
                    success=True,
                    data={'tunnel_url': tunnel_url, 'status': 'bootstrapped'})
            logger.info("Connect attempt %d/%d failed: %s",
                        attempt + 1, len(delays), result.error)

        return SandboxResult(
            success=False,
            error=f"Bootstrap completed but server not reachable after {len(delays)} retries: {result.error}")

    def shutdown(self):
        """Mark as disconnected (server keeps running on Colab)."""
        self._connected = False

    # ── HTTP Transport ────────────────────────────────────────────────────

    def _http_get(self, path: str, timeout: int = None) -> SandboxResult:
        """Send GET request to Colab server."""
        if not self._tunnel_url:
            return SandboxResult(success=False,
                                 error='No tunnel URL configured')
        url = f'{self._tunnel_url}{path}'
        effective_timeout = timeout or self.config.connect_timeout
        try:
            req = urllib_request.Request(url, method='GET')
            req.add_header('User-Agent', 'Zhuangzhou/0.9.0')
            with urllib_request.urlopen(req, timeout=effective_timeout) as resp:
                body = resp.read().decode('utf-8', errors='replace')
                data = body
                try:
                    data = json.loads(body)
                except (json.JSONDecodeError, ValueError):
                    pass
                return SandboxResult(success=True, data=data, stdout=body)
        except HTTPError as e:
            body = ''
            try:
                body = e.read().decode('utf-8', errors='replace')
            except Exception:
                pass
            return SandboxResult(
                success=False,
                error=f'HTTP {e.code}: {body[:500]}',
                returncode=e.code)
        except URLError as e:
            return SandboxResult(
                success=False,
                error=f'Cannot reach Colab server: {e.reason}. '
                      f'Is the Colab notebook still running?')
        except Exception as e:
            return SandboxResult(success=False, error=str(e))

    def _http_post(self, path: str, payload: dict,
                   timeout: int = None) -> SandboxResult:
        """Send POST request to Colab server."""
        if not self._tunnel_url:
            return SandboxResult(success=False,
                                 error='No tunnel URL configured')
        url = f'{self._tunnel_url}{path}'
        effective_timeout = timeout or self.timeout
        body_bytes = json.dumps(payload, ensure_ascii=False).encode('utf-8')
        try:
            req = urllib_request.Request(
                url, data=body_bytes, method='POST')
            req.add_header('Content-Type', 'application/json')
            req.add_header('User-Agent', 'Zhuangzhou/0.9.0')
            with urllib_request.urlopen(req, timeout=effective_timeout) as resp:
                resp_body = resp.read().decode('utf-8', errors='replace')
                data = resp_body
                try:
                    data = json.loads(resp_body)
                except (json.JSONDecodeError, ValueError):
                    pass
                # Check if the response indicates an error
                is_error = False
                error_msg = ''
                if isinstance(data, dict):
                    if data.get('success') is False:
                        is_error = True
                        error_msg = data.get('error', '')
                    elif 'error' in data and 'mutations' not in data:
                        is_error = True
                        error_msg = data['error']
                return SandboxResult(
                    success=not is_error,
                    data=data,
                    stdout=resp_body,
                    error=error_msg)
        except HTTPError as e:
            body = ''
            try:
                body = e.read().decode('utf-8', errors='replace')
            except Exception:
                pass
            # Try to parse error detail from FastAPI
            error_detail = f'HTTP {e.code}'
            try:
                err_json = json.loads(body)
                if 'detail' in err_json:
                    error_detail = f'HTTP {e.code}: {err_json["detail"]}'
            except Exception:
                if body:
                    error_detail = f'HTTP {e.code}: {body[:500]}'
            return SandboxResult(
                success=False, error=error_detail, returncode=e.code)
        except URLError as e:
            self._connected = False
            return SandboxResult(
                success=False,
                error=f'Cannot reach Colab server: {e.reason}. '
                      f'The Colab notebook may have disconnected. '
                      f'Please check and restart if needed.')
        except Exception as e:
            return SandboxResult(success=False, error=str(e))

    # ── Public API (backward compatible) ──────────────────────────────────

    def execute_code(self, code: str, timeout: int = None) -> SandboxResult:
        """
        Execute arbitrary Python code on Colab runtime.

        Uses the /execute endpoint of the GPU server.
        """
        if not self._connected:
            conn = self.connect()
            if not conn.success:
                return conn

        result = self._http_post('/execute', {
            'code': code,
            'timeout': timeout or self.timeout,
        }, timeout=timeout)

        # Normalize execute response to SandboxResult
        if result.success and isinstance(result.data, dict):
            inner = result.data
            return SandboxResult(
                success=inner.get('success', True),
                data=inner.get('data', inner.get('stdout', '')),
                stdout=inner.get('stdout', ''),
                stderr=inner.get('stderr', ''),
                error=inner.get('error', ''),
                returncode=0 if inner.get('success', True) else 1,
            )
        return result

    def run_json_bridge(self, code: str, payload: dict,
                        timeout: int = None) -> SandboxResult:
        """Execute JSON bridge pattern on Colab GPU (backward compatible).

        For new code, prefer using the dedicated endpoints:
          esm_scan(), esm_consensus_scan(), gpu_info()
        """
        wrapped_code = (
            f"import json as _json\n"
            f"_payload = _json.loads({json.dumps(payload, ensure_ascii=False)!r})\n"
            f"{code}\n"
        )
        return self.execute_code(wrapped_code, timeout=timeout)

    def gpu_info(self, timeout: int = 30) -> SandboxResult:
        """Get GPU information from Colab runtime."""
        if not self._connected:
            conn = self.connect()
            if not conn.success:
                return conn
        return self._http_get('/gpu_info', timeout=timeout)

    def esm_scan(self, sequence: str, positions: list = None,
                 model: str = 'esm2_t33_650M_UR50D',
                 top_k: int = 20, timeout: int = None) -> SandboxResult:
        """Run ESM-2/ESM-1v mutation scan via dedicated endpoint (model stays loaded)."""
        if not self._connected:
            conn = self.connect()
            if not conn.success:
                return conn
        payload = {
            'sequence': sequence,
            'model': model,
            'top_k': top_k,
        }
        if positions:
            payload['positions'] = positions
        return self._http_post('/esm_scan', payload,
                               timeout=timeout or self.timeout)

    def esm_consensus_scan(self, sequence: str, positions: list = None,
                           models: list = None, top_k: int = 20,
                           timeout: int = None) -> SandboxResult:
        """Run multi-model consensus mutation scan.

        Averages log-likelihood ratio scores across specified models.
        Default: ESM-2 + ESM-1v (model 1).
        For full ESM-1v ensemble: pass all 5 esm1v models.
        """
        if not self._connected:
            conn = self.connect()
            if not conn.success:
                return conn
        payload = {
            'sequence': sequence,
            'models': models or ['esm2_t33_650M_UR50D', 'esm1v_t33_650M_UR90S_1'],
            'top_k': top_k,
        }
        if positions:
            payload['positions'] = positions
        return self._http_post('/esm_consensus_scan', payload,
                               timeout=timeout or self.timeout)

    def esm_union_scan(self, sequence: str, positions: list = None,
                       models: list = None, top_k: int = 96,
                       threshold: float = 0.0, weights: list = None,
                       timeout: int = None) -> SandboxResult:
        """Union-filtered dual-model scan: keep mutations positive in ANY model.

        Implements the Zero_Shot.ipynb strategy:
          - Run ESM-2 + ESM-1v independently
          - Keep union of LL > threshold mutations
          - Classify: Tier 1 (consensus) vs Tier 2 (single positive)
          - Sort by tier then max_potential
          - Return top_k candidates (default 96 for plate ordering)

        This is the recommended method for selecting wet-lab candidates.
        """
        if not self._connected:
            conn = self.connect()
            if not conn.success:
                return conn
        payload = {
            'sequence': sequence,
            'models': models or ['esm2_t33_650M_UR50D', 'esm1v_t33_650M_UR90S_1'],
            'top_k': top_k,
            'threshold': threshold,
        }
        if positions:
            payload['positions'] = positions
        if weights:
            payload['weights'] = weights
        return self._http_post('/esm_union_scan', payload,
                               timeout=timeout or self.timeout)

    def install_packages(self, packages: List[str],
                         timeout: int = None) -> SandboxResult:
        """Install Python packages on Colab runtime."""
        preinstalled = set(self.config.colab_preinstalled)
        to_install = [p for p in packages
                      if p.split('==')[0].split('[')[0] not in preinstalled]
        if not to_install:
            return SandboxResult(
                success=True,
                data={'message': 'All packages already available on Colab',
                      'packages': packages})

        pkg_args = ', '.join(repr(p) for p in to_install)
        code = (
            f"import subprocess, sys\n"
            f"r = subprocess.run(\n"
            f"    [sys.executable, '-m', 'pip', 'install', '-q', {pkg_args}],\n"
            f"    capture_output=True, text=True)\n"
            f"print(r.stdout)\n"
            f"if r.returncode != 0:\n"
            f"    print('INSTALL_ERROR:', r.stderr)\n"
        )
        return self.execute_code(
            code, timeout=timeout or self.config.install_timeout)

    def upload_data(self, data: str, remote_path: str,
                    timeout: int = 30) -> SandboxResult:
        """Write text data to a file on Colab runtime."""
        escaped = json.dumps(data, ensure_ascii=False)
        code = (
            f"import os\n"
            f"os.makedirs(os.path.dirname({remote_path!r}) or '.', exist_ok=True)\n"
            f"with open({remote_path!r}, 'w', encoding='utf-8') as f:\n"
            f"    f.write({escaped})\n"
            f"print('OK:', {remote_path!r})\n"
        )
        return self.execute_code(code, timeout=timeout)

    def download_data(self, remote_path: str,
                      timeout: int = 30) -> SandboxResult:
        """Read text data from a file on Colab runtime."""
        code = (
            f"with open({remote_path!r}, 'r', encoding='utf-8') as f:\n"
            f"    print(f.read())\n"
        )
        return self.execute_code(code, timeout=timeout)

    def check_packages(self, packages: List[str],
                       timeout: int = 30) -> Dict[str, bool]:
        """Check which packages are available on Colab."""
        checks = ', '.join(f"'{p}'" for p in packages)
        code = (
            f"import importlib.util\n"
            f"for p in [{checks}]:\n"
            f"    safe = p.replace('-', '_')\n"
            f"    ok = importlib.util.find_spec(safe) is not None\n"
            f"    print(f'{{p}}:{{\"OK\" if ok else \"MISSING\"}}')\n"
        )
        result = self.execute_code(code, timeout=timeout)
        status = {}
        if result.success and isinstance(result.data, str):
            for line in result.data.strip().split('\n'):
                if ':' in line:
                    pkg, st = line.strip().rsplit(':', 1)
                    status[pkg] = (st == 'OK')
        return status

    def preload_esm(self, model: str = 'esm2_t33_650M_UR50D',
                    timeout: int = 120) -> SandboxResult:
        """Pre-load ESM model into GPU memory on Colab."""
        if not self._connected:
            conn = self.connect()
            if not conn.success:
                return conn
        return self._http_post(f'/preload_esm?model={model}', {},
                               timeout=timeout)
