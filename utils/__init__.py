"""
Zhuangzhou Utils — Production-grade robustness utilities.

  - robustness: Safe JSON, file I/O, path security, error truncation
  - retry: Exponential backoff, error classification, retry decorator
  - cleanup: Global cleanup registry, signal handlers, PID files
  - cache: TTL memoization (stale-while-revalidate), LRU, in-flight dedup
  - subprocess_utils: Non-throwing subprocess, structured results
"""

from .robustness import (
    safe_json_parse,
    safe_json_parse_last_line,
    parse_jsonl,
    read_jsonl_tail,
    strip_bom,
    safe_read_file,
    safe_write_file,
    check_path_traversal,
    normalize_path,
    normalize_windows_path,
    truncate_error,
    sanitize_error_html,
    walk_error_chain,
)

from .retry import (
    ErrorKind,
    classify_error,
    is_retryable,
    exponential_backoff,
    retry,
    retry_call,
)

from .cleanup import (
    cleanup_registry,
    install_signal_handlers,
    register_temp_file,
    register_pid_file,
    is_pid_alive,
)

from .cache import (
    memoize_with_ttl,
    memoize_with_lru,
    LRUCache,
    InflightDedup,
)

from .subprocess_utils import (
    SubprocessResult,
    run_subprocess,
    run_python_subprocess,
    run_inline_python,
)

__all__ = [
    # robustness
    'safe_json_parse', 'safe_json_parse_last_line',
    'parse_jsonl', 'read_jsonl_tail',
    'strip_bom', 'safe_read_file', 'safe_write_file',
    'check_path_traversal', 'normalize_path', 'normalize_windows_path',
    'truncate_error', 'sanitize_error_html', 'walk_error_chain',
    # retry
    'ErrorKind', 'classify_error', 'is_retryable',
    'exponential_backoff', 'retry', 'retry_call',
    # cleanup
    'cleanup_registry', 'install_signal_handlers',
    'register_temp_file', 'register_pid_file', 'is_pid_alive',
    # cache
    'memoize_with_ttl', 'memoize_with_lru', 'LRUCache', 'InflightDedup',
    # subprocess
    'SubprocessResult', 'run_subprocess',
    'run_python_subprocess', 'run_inline_python',
]
