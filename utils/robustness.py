"""
Core Robustness Utilities
=========================
Production-grade patterns for defensive I/O and data handling:
  - Safe JSON parsing with bounded caching
  - BOM stripping (Windows PowerShell compatibility)
  - Safe file read with encoding fallback
  - Path traversal security check
  - Path normalization for cross-platform
  - Error message truncation for telemetry safety
  - JSONL parsing with graceful malformed-line skipping
"""

from __future__ import annotations

import json
import copy
import logging
import os
import tempfile
from functools import lru_cache
from pathlib import Path
from typing import Any, Generator, Optional, Union

logger = logging.getLogger(__name__)

# ── Constants ──────────────────────────────────────────────────────────────────

_JSON_CACHE_MAX_BYTES = 8 * 1024  # Cache JSON parses for inputs <= 8KB
_ERROR_MSG_MAX_LEN = 2000  # Max error message length for logging
UTF8_BOM = '\ufeff'


# ── Safe JSON Parsing ─────────────────────────────────────────────────────────

@lru_cache(maxsize=64)
def _parse_json_cached(text: str) -> Any:
    """LRU-cached JSON parse for small, repeated strings."""
    return json.loads(text)


def safe_json_parse(
    text: Optional[str],
    default: Any = None,
    log_error: bool = True,
) -> Any:
    """
    Parse JSON safely, returning default on failure.

    Uses LRU cache for small inputs (<=8KB) to speed up repeated parses.
    Never raises — returns default on None/empty/malformed input.

"""
    if not text:
        return default
    try:
        if len(text) <= _JSON_CACHE_MAX_BYTES:
            return copy.deepcopy(_parse_json_cached(text))
        return json.loads(text)
    except (json.JSONDecodeError, ValueError, TypeError) as e:
        if log_error:
            preview = text[:100] + '...' if len(text) > 100 else text
            logger.debug("JSON parse failed: %s | input: %s", e, preview)
        return default


def safe_json_parse_last_line(
    text: str,
    default: Any = None,
) -> Any:
    """
    Try to parse JSON from text; if full parse fails, try last non-empty line.

    Useful for subprocess output where model downloads or progress bars
    pollute stdout before the final JSON result line.
    """
    if not text:
        return default

    # Try full text first
    result = safe_json_parse(text, default=None, log_error=False)
    if result is not None:
        return result

    # Try last JSON-like line (reverse scan)
    for line in reversed(text.split('\n')):
        line = line.strip()
        if line and (line.startswith('{') or line.startswith('[')):
            result = safe_json_parse(line, default=None, log_error=False)
            if result is not None:
                return result

    return default


# ── JSONL Parsing ──────────────────────────────────────────────────────────────

def parse_jsonl(
    text: str,
    skip_malformed: bool = True,
) -> list:
    """
    Parse JSONL text into a list of objects.

    Skips malformed lines instead of crashing (graceful degradation).

"""
    results = []
    for i, line in enumerate(text.split('\n')):
        line = line.strip()
        if not line:
            continue
        try:
            results.append(json.loads(line))
        except (json.JSONDecodeError, ValueError) as e:
            if skip_malformed:
                logger.debug("JSONL line %d malformed, skipping: %s", i, e)
            else:
                raise
    return results


def read_jsonl_tail(
    filepath: str,
    max_bytes: int = 100 * 1024 * 1024,
) -> list:
    """
    Read JSONL file, using tail-buffer for large files.

    For files > max_bytes, reads only the last max_bytes and skips the
    first (likely partial) line.

"""
    path = Path(filepath)
    if not path.exists():
        return []

    size = path.stat().st_size
    if size <= max_bytes:
        return parse_jsonl(
            path.read_text(encoding='utf-8', errors='replace'))

    # Read tail of large file
    with open(filepath, 'rb') as f:
        f.seek(max(0, size - max_bytes))
        buf = f.read(max_bytes)

    # Skip first partial line
    newline_idx = buf.find(b'\n')
    if newline_idx >= 0:
        buf = buf[newline_idx + 1:]

    text = buf.decode('utf-8', errors='replace')
    return parse_jsonl(text)


# ── BOM & Encoding ────────────────────────────────────────────────────────────

def strip_bom(text: str) -> str:
    """
    Strip UTF-8 BOM from text.

    Windows PowerShell 5.x adds BOM by default. This corrupts JSON.parse()
    and other strict parsers.

"""
    return text.lstrip(UTF8_BOM)


def safe_read_file(
    filepath: Union[str, Path],
    encoding: str = 'utf-8',
    default: Optional[str] = None,
) -> Optional[str]:
    """
    Read a file safely with encoding fallback and BOM stripping.

    Returns default on any filesystem error (not found, permission, etc.).
    Always strips BOM for Windows compatibility.

"""
    try:
        text = Path(filepath).read_text(encoding=encoding, errors='replace')
        return strip_bom(text)
    except FileNotFoundError:
        logger.debug("File not found: %s", filepath)
        return default
    except IsADirectoryError:
        logger.debug("Path is a directory: %s", filepath)
        return default
    except PermissionError:
        logger.warning("Permission denied: %s", filepath)
        return default
    except OSError as e:
        logger.warning("Cannot read %s: %s", filepath, e)
        return default


def safe_write_file(
    filepath: Union[str, Path],
    content: str,
    encoding: str = 'utf-8',
    make_parents: bool = True,
) -> bool:
    """
    Write file safely with parent directory creation.

    Returns True on success, False on failure.
    """
    try:
        p = Path(filepath)
        if make_parents:
            p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content, encoding=encoding)
        return True
    except OSError as e:
        logger.error("Cannot write %s: %s", filepath, e)
        return False


# ── Path Security & Normalization ──────────────────────────────────────────────


def atomic_write_json(filepath: Union[str, Path], data: Any) -> None:
    """Replace JSON only after a complete write; failed writes preserve the old file."""
    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', dir=path.parent,
                                         prefix=path.name + '.', suffix='.tmp', delete=False) as handle:
            temporary = Path(handle.name)
            json.dump(data, handle, ensure_ascii=False, indent=2, default=str, allow_nan=False)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def redact_secrets(value: Any) -> Any:
    """Copy structured metadata while removing credentials identified by field name."""
    if isinstance(value, dict):
        sensitive = {'api_key', 'apikey', 'authorization', 'password', 'passwd',
                     'secret', 'secret_key', 'token', 'access_token', 'refresh_token',
                     'client_secret', 'private_key', 'credentials'}
        return {key: '[REDACTED]' if str(key).lower().replace('-', '_') in sensitive
                else redact_secrets(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [redact_secrets(item) for item in value]
    return value

def check_path_traversal(path: str) -> Optional[str]:
    """
    Check if a path contains traversal sequences.

    Returns error message if traversal detected, None if safe.

"""
    if '..' in path:
        return f'Path contains ".." which could be a traversal attack: {path}'
    return None


def normalize_path(path: Union[str, Path]) -> str:
    """
    Normalize a path for cross-platform consistency.

    - Resolves to absolute path
    - Uses forward slashes (works on both Windows and Unix)
    - Strips trailing slashes
    """
    return str(Path(path).resolve()).replace('\\', '/')


def normalize_windows_path(path: str) -> str:
    """
    Normalize a Windows path: resolve, forward slashes, consistent drive letter.
    """
    p = Path(path).resolve()
    result = str(p).replace('\\', '/')
    # Uppercase drive letter for consistency
    if len(result) >= 2 and result[1] == ':':
        result = result[0].upper() + result[1:]
    return result


# ── Error Message Safety ───────────────────────────────────────────────────────

def truncate_error(
    message: str,
    max_len: int = _ERROR_MSG_MAX_LEN,
) -> str:
    """
    Truncate error message for safe logging/telemetry.

"""
    if len(message) <= max_len:
        return message
    return message[:max_len] + f'... (truncated, {len(message)} chars total)'


def sanitize_error_html(message: str) -> str:
    """
    Strip HTML from error messages (e.g., CloudFlare error pages in API responses).

"""
    if '<!DOCTYPE html' in message or '<html' in message:
        import re
        title_match = re.search(r'<title>([^<]+)</title>', message)
        return title_match.group(1).strip() if title_match else 'HTML error page'
    return message


def walk_error_chain(
    error: BaseException,
    max_depth: int = 5,
) -> list[BaseException]:
    """
    Walk the exception __cause__ chain to find root causes.

    Returns list of exceptions from outermost to innermost cause.
    """
    chain = [error]
    current = error
    for _ in range(max_depth):
        cause = getattr(current, '__cause__', None)
        if cause is None:
            cause = getattr(current, '__context__', None)
        if cause is None or cause in chain:
            break
        chain.append(cause)
        current = cause
    return chain
