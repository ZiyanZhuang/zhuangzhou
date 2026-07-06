"""
Colab Bootstrap — One-shot automated Colab setup via colab-mcp
================================================================
Uses Google's official colab-mcp MCP server to:
  1. Open ONE Colab notebook (one browser tab, OAuth cached after first run)
  2. Inject the GPU server code into a cell
  3. Execute the cell (starts FastAPI + cloudflared in background threads)
  4. Parse the tunnel URL from cell output
  5. Save URL to ~/.zhuangzhou/colab_config.json

After bootstrap, all GPU calls go through HTTP — colab-mcp is no longer needed.

Usage:
    from zhuangzhou.envs.colab_bootstrap import bootstrap_colab
    url = bootstrap_colab()  # Returns tunnel URL or None

Prerequisites:
    pip install uv    (for uvx command)
"""

import json
import logging
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

logger = logging.getLogger(__name__)

_USER_DIR = Path.home() / '.zhuangzhou'
_BOOTSTRAP_LOG = _USER_DIR / 'colab_bootstrap.log'


def _find_uvx() -> str:
    """Find uvx executable."""
    uvx = shutil.which('uvx')
    if uvx:
        return uvx
    # Try common locations on Windows
    for candidate in [
        os.environ.get('UVX_EXE', ''),
        os.path.expanduser('~/.local/bin/uvx'),
        os.path.expanduser('~/.cargo/bin/uvx'),
    ]:
        if os.path.isfile(candidate):
            return candidate
    return ''


def _read_bootstrap_code() -> str:
    """Read the bootstrap cell code."""
    script_path = os.path.join(
        os.path.dirname(__file__), '..', 'scripts', 'colab_bootstrap_cell.py')
    with open(script_path, 'r', encoding='utf-8') as f:
        return f.read()


class _MCPClient:
    """Lightweight MCP JSON-RPC client for colab-mcp bootstrap.

    Only implements what's needed for bootstrap:
      - initialize handshake
      - tools/call (create_cell, execute_cell, add_code_cell, run_code_cell)
    """

    def __init__(self, process: subprocess.Popen):
        self.proc = process
        self._request_id = 0
        self._recv_buf = b''

    def _next_id(self) -> int:
        self._request_id += 1
        return self._request_id

    def _send(self, method: str, params: dict, timeout: int = 120) -> dict:
        """Send JSON-RPC request and wait for matching response."""
        req_id = self._next_id()
        request = {
            'jsonrpc': '2.0',
            'id': req_id,
            'method': method,
            'params': params,
        }
        msg = json.dumps(request, ensure_ascii=False) + '\n'
        logger.debug("MCP send: %s", msg.strip())

        try:
            self.proc.stdin.write(msg.encode('utf-8'))
            self.proc.stdin.flush()
        except Exception as e:
            logger.error("Failed to send MCP request: %s", e)
            return {}

        deadline = time.time() + timeout
        while time.time() < deadline:
            if self.proc.poll() is not None:
                logger.error("colab-mcp process exited (rc=%d)", self.proc.returncode)
                return {}

            # Check buffer for complete lines
            if b'\n' in self._recv_buf:
                line, self._recv_buf = self._recv_buf.split(b'\n', 1)
                line_str = line.decode('utf-8', errors='replace').strip()
                if not line_str:
                    continue
                try:
                    resp = json.loads(line_str)
                    if resp.get('id') == req_id:
                        logger.debug("MCP recv (id=%d): %s",
                                     req_id, line_str[:200])
                        return resp
                    # Skip notifications
                    if 'method' in resp and 'id' not in resp:
                        continue
                except json.JSONDecodeError:
                    continue
                continue

            # Read more data from stdout
            import select
            try:
                # On Windows, select doesn't work on pipes, use a short read with timeout
                if sys.platform == 'win32':
                    # Non-blocking read attempt
                    import msvcrt
                    if msvcrt.kbhit():
                        pass
                    # Fallback: blocking read with a line
                    self.proc.stdout.flush() if hasattr(self.proc.stdout, 'flush') else None
                    line = self.proc.stdout.readline()
                    if line:
                        self._recv_buf += line
                    else:
                        time.sleep(0.1)
                else:
                    readable, _, _ = select.select([self.proc.stdout], [], [], 1.0)
                    if readable:
                        chunk = self.proc.stdout.read1(65536) if hasattr(self.proc.stdout, 'read1') else self.proc.stdout.readline()
                        if chunk:
                            self._recv_buf += chunk
                        else:
                            time.sleep(0.1)
                    else:
                        time.sleep(0.1)
            except Exception:
                time.sleep(0.1)

        logger.warning("MCP timeout for method=%s id=%d after %ds",
                        method, req_id, timeout)
        return {}

    def _send_notification(self, method: str, params: dict = None):
        """Send JSON-RPC notification (no response expected)."""
        notification = {
            'jsonrpc': '2.0',
            'method': method,
            'params': params or {},
        }
        msg = json.dumps(notification, ensure_ascii=False) + '\n'
        try:
            self.proc.stdin.write(msg.encode('utf-8'))
            self.proc.stdin.flush()
        except Exception:
            pass

    def initialize(self, timeout: int = 60) -> bool:
        """MCP initialize handshake."""
        resp = self._send('initialize', {
            'protocolVersion': '2024-11-05',
            'capabilities': {},
            'clientInfo': {'name': 'zhuangzhou-bootstrap', 'version': '0.9.0'},
        }, timeout=timeout)

        if not resp or 'result' not in resp:
            return False

        server_info = resp.get('result', {}).get('serverInfo', {})
        logger.info("MCP server: %s v%s",
                     server_info.get('name', '?'),
                     server_info.get('version', '?'))

        # Send initialized notification
        self._send_notification('notifications/initialized')
        time.sleep(0.5)
        return True

    def list_tools(self, timeout: int = 30) -> list:
        """List available MCP tools."""
        resp = self._send('tools/list', {}, timeout=timeout)
        if not resp or 'result' not in resp:
            return []
        tools = resp.get('result', {}).get('tools', [])
        return [t.get('name', '') for t in tools if isinstance(t, dict)]

    def call_tool(self, tool_name: str, arguments: dict,
                  timeout: int = 120) -> dict:
        """Call an MCP tool and return result."""
        resp = self._send('tools/call', {
            'name': tool_name,
            'arguments': arguments,
        }, timeout=timeout)

        if not resp:
            return {'error': 'No response'}

        if 'error' in resp:
            return {'error': f"{resp['error'].get('code', '?')}: "
                            f"{resp['error'].get('message', str(resp['error']))}"}

        result = resp.get('result', {})
        # Extract text content
        content = result.get('content', [])
        text_parts = []
        for item in content:
            if isinstance(item, dict) and item.get('type') == 'text':
                text_parts.append(item.get('text', ''))

        combined = '\n'.join(text_parts)
        is_error = result.get('isError', False)

        return {
            'success': not is_error,
            'text': combined,
            'error': combined if is_error else '',
        }


def bootstrap_colab(timeout: int = 180, mcp_startup_timeout: int = 60) -> str:
    """
    Automatically set up Colab GPU server and return tunnel URL.

    Flow:
      1. Start colab-mcp subprocess
      2. MCP initialize handshake
      3. Open Colab browser connection (ONE tab)
      4. Inject server code into a cell
      5. Execute cell (server starts in background)
      6. Parse tunnel URL from output
      7. Save to config
      8. Return URL

    Args:
        timeout: Total timeout for the entire bootstrap process (seconds)
        mcp_startup_timeout: Timeout for colab-mcp to start and connect

    Returns:
        Tunnel URL string, or empty string on failure
    """
    _USER_DIR.mkdir(parents=True, exist_ok=True)

    uvx = _find_uvx()
    if not uvx:
        logger.error("uvx not found. Install: pip install uv")
        return ''

    # Read bootstrap code
    try:
        bootstrap_code = _read_bootstrap_code()
    except FileNotFoundError:
        logger.error("Bootstrap script not found: scripts/colab_bootstrap_cell.py")
        return ''

    logger.info("Starting colab-mcp for bootstrap...")

    # Start colab-mcp subprocess
    cmd = [uvx, 'git+https://github.com/googlecolab/colab-mcp']
    log_fh = open(_BOOTSTRAP_LOG, 'w', encoding='utf-8')

    try:
        proc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=log_fh,
            bufsize=0,
        )
    except Exception as e:
        logger.error("Failed to start colab-mcp: %s", e)
        log_fh.close()
        return ''

    client = _MCPClient(proc)
    tunnel_url = ''

    try:
        # Step 1: Initialize MCP
        logger.info("[1/6] MCP initialize handshake...")
        if not client.initialize(timeout=mcp_startup_timeout):
            logger.error("MCP initialize failed. Check %s for details.", _BOOTSTRAP_LOG)
            return ''

        # Step 2: Discover available tools
        logger.info("[2/6] Discovering available MCP tools...")
        tools = client.list_tools(timeout=15)
        logger.info("  Available tools: %s", tools)

        # Resolve tool names — colab-mcp may use proxied names with prefixes
        def _find_tool(base_name: str) -> str:
            """Find a tool by base name, handling possible prefixes."""
            for t in tools:
                if t == base_name or t.endswith(f'_{base_name}') or t.endswith(base_name):
                    return t
            logger.warning("Tool '%s' not found in available tools: %s",
                           base_name, tools)
            return base_name  # fallback to base name (may fail)

        tool_open = _find_tool('open_colab_browser_connection')

        # Step 3: Connect to Colab — reuse existing browser tab if possible
        # colab-mcp only exposes 'open_colab_browser_connection' initially;
        # notebook tools (add_code_cell, run_code_cell, etc.) appear AFTER
        # the browser connects. Switching GPU runtime restarts the runtime
        # and drops the WebSocket, causing tools to disappear temporarily.
        # We NEVER open a second browser tab — only wait for reconnection.
        logger.info("[3/6] Checking for existing Colab connection...")

        def _has_notebook_tools(tool_list):
            return any(t for t in tool_list
                       if 'add_code_cell' in t or 'run_code_cell' in t)

        def _wait_for_notebook_tools(max_wait: int, reason: str) -> bool:
            """Poll tools/list until notebook tools appear, or timeout."""
            nonlocal tools
            logger.info("  Waiting for notebook tools (%s)...", reason)
            poll_interval = 10
            for attempt in range(max_wait // poll_interval):
                time.sleep(poll_interval)
                tools = client.list_tools(timeout=15)
                if _has_notebook_tools(tools):
                    logger.info("  Notebook tools available (attempt %d): %s",
                                attempt + 1, tools)
                    return True
                logger.info("  Polling attempt %d — tools: %s",
                            attempt + 1, tools)
            return False

        if _has_notebook_tools(tools):
            # Browser already connected and runtime is up
            logger.info("  Notebook tools available — reusing existing connection")
            print("Reusing existing Colab connection (no new browser tab).",
                  flush=True)
        else:
            # Need to open browser — do this ONCE only
            logger.info("  No notebook tools, opening Colab in browser...")
            print("Opening Colab in browser...", flush=True)
            print("If you need to switch to T4 GPU runtime, do it now.",
                  flush=True)
            print("Waiting for Colab connection (up to 2 minutes)...",
                  flush=True)
            result = client.call_tool(tool_open, {}, timeout=timeout)
            if not result.get('success', False) and result.get('error'):
                logger.warning("open_colab_browser_connection: %s",
                               result.get('error', ''))

            # Wait for notebook tools to appear
            # Use a long timeout (120s) to account for:
            #   - Browser loading Colab (~5-10s)
            #   - User selecting T4 GPU runtime (~10-30s)
            #   - Runtime restart after GPU switch (~15-30s)
            #   - WebSocket reconnection (~5s)
            if not _wait_for_notebook_tools(120, "browser connection + GPU setup"):
                logger.error("Notebook tools not available after 120s. "
                             "Is the Colab notebook open in the browser?")
                return ''

        # Resolve notebook tool names (may have proxy prefixes)
        tool_add_cell = _find_tool('add_code_cell')
        tool_run_cell = _find_tool('run_code_cell')
        tool_get_cells = _find_tool('get_cells')

        # Step 4: Create code cell with bootstrap code
        logger.info("[4/6] Injecting server code into Colab cell...")
        result = client.call_tool(tool_add_cell, {
            'cellIndex': 0,
            'language': 'python',
            'code': bootstrap_code,
        }, timeout=30)

        cell_id = None
        text = result.get('text', '')
        # Try to extract cell ID from response
        try:
            data = json.loads(text)
            cell_id = data.get('newCellId') or data.get('cellId') or data.get('id')
        except (json.JSONDecodeError, ValueError):
            cell_id = text.strip().strip('"') if text.strip() else None

        if not cell_id:
            logger.error("Failed to create code cell: %s", result)
            return ''

        logger.info("  Cell created: %s", cell_id)

        # Step 5: Execute cell (this starts the server + tunnel)
        logger.info("[5/6] Executing cell (starting GPU server + tunnel)...")
        print("Starting GPU server on Colab... This may take 30-60 seconds.",
              flush=True)
        result = client.call_tool(tool_run_cell, {
            'cellId': cell_id,
        }, timeout=120)

        output_text = result.get('text', '')
        logger.info("  Cell output: %s", output_text[:500])

        # Step 6: Parse tunnel URL from output
        # cloudflared takes 5-15s to start and print the URL, so
        # run_code_cell may return before the URL is in stdout.
        # We poll get_cells repeatedly to capture the full output.
        logger.info("[6/6] Parsing tunnel URL...")
        tunnel_url = _parse_tunnel_url(output_text)

        if not tunnel_url:
            logger.info("  URL not in run_code_cell output, polling cell output...")
            for poll in range(12):  # Up to 60s
                time.sleep(5)
                result = client.call_tool(tool_get_cells, {
                    'cellIndexStart': 0,
                    'includeOutputs': True,
                }, timeout=15)
                output_text = result.get('text', '')
                tunnel_url = _parse_tunnel_url(output_text)
                if tunnel_url:
                    logger.info("  Found URL on poll attempt %d", poll + 1)
                    break
                logger.info("  Poll %d — no URL yet", poll + 1)

        if tunnel_url:
            logger.info("Tunnel URL: %s", tunnel_url)
            print(f"\nColab GPU server ready: {tunnel_url}", flush=True)

            # Save to config
            from .colab_config import get_config, save_config
            config = get_config()
            config.tunnel_url = tunnel_url
            save_config(config)
            logger.info("Saved tunnel URL to config")
        else:
            logger.error("Failed to parse tunnel URL from output")
            logger.error("Output was: %s", output_text[:1000])

    except Exception as e:
        logger.error("Bootstrap failed: %s", e)
        import traceback
        traceback.print_exc()
    finally:
        # Kill colab-mcp process — we don't need it anymore
        # (The server keeps running in the Colab notebook)
        try:
            proc.stdin.close()
            proc.wait(timeout=5)
        except Exception:
            try:
                proc.kill()
            except Exception:
                pass
        log_fh.close()

    return tunnel_url


def _parse_tunnel_url(text: str) -> str:
    """Parse tunnel URL from cell output text.

    Handles multiple formats:
      - Plain text: ZZ_TUNNEL_URL=https://xxx.trycloudflare.com
      - Nested JSON from get_cells (outputs in arrays of strings)
      - JSON-escaped strings (\\n, nested quotes)
    """
    if not text:
        return ''

    # Strategy 1: Direct regex on raw text (works for plain output)
    match = re.search(r'ZZ_TUNNEL_URL=(https://[a-zA-Z0-9-]+\.trycloudflare\.com)', text)
    if match:
        return match.group(1)

    match = re.search(r'(https://[a-zA-Z0-9-]+\.trycloudflare\.com)', text)
    if match:
        return match.group(1)

    # Strategy 2: Parse JSON and extract all strings recursively
    # get_cells returns: {"cells": [{"outputs": [{"text": ["ZZ_TUNNEL_URL=..."]}]}]}
    try:
        data = json.loads(text)
        flat_text = _extract_all_strings(data)
        match = re.search(r'ZZ_TUNNEL_URL=(https://[a-zA-Z0-9-]+\.trycloudflare\.com)', flat_text)
        if match:
            return match.group(1)
        match = re.search(r'(https://[a-zA-Z0-9-]+\.trycloudflare\.com)', flat_text)
        if match:
            return match.group(1)
    except (json.JSONDecodeError, ValueError):
        pass

    # Strategy 3: Un-escape JSON string literals (e.g. \" and \\n in raw text)
    unescaped = text.replace('\\n', '\n').replace('\\"', '"').replace('\\/', '/')
    match = re.search(r'(https://[a-zA-Z0-9-]+\.trycloudflare\.com)', unescaped)
    if match:
        return match.group(1)

    return ''


def _extract_all_strings(obj, depth: int = 0) -> str:
    """Recursively extract all string values from a JSON structure."""
    if depth > 20:
        return ''
    if isinstance(obj, str):
        return obj + '\n'
    if isinstance(obj, list):
        return ''.join(_extract_all_strings(item, depth + 1) for item in obj)
    if isinstance(obj, dict):
        return ''.join(_extract_all_strings(v, depth + 1) for v in obj.values())
    return ''


def is_bootstrap_needed() -> bool:
    """Check if bootstrap is needed (no valid tunnel URL configured)."""
    from .colab_config import get_config
    config = get_config()
    if not config.tunnel_url:
        return True
    # Quick health check
    try:
        from urllib import request as urllib_request
        req = urllib_request.Request(f'{config.tunnel_url}/health', method='GET')
        req.add_header('User-Agent', 'Zhuangzhou/0.9.0')
        with urllib_request.urlopen(req, timeout=5) as resp:
            return resp.status != 200
    except Exception:
        return True


# ── CLI entry point ──────────────────────────────────────────────────────

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s [%(name)s] %(message)s')
    sys.stdout.reconfigure(encoding='utf-8')

    url = bootstrap_colab()
    if url:
        print(f'\nBootstrap successful!')
        print(f'Tunnel URL: {url}')
        print(f'Saved to ~/.zhuangzhou/colab_config.json')
        print(f'\nTest: python scripts/test_colab.py')
    else:
        print('\nBootstrap failed. Check ~/.zhuangzhou/colab_bootstrap.log')
        sys.exit(1)
