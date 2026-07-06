"""
Zhuangzhou UI Engine - Legacy Redirect
====================================
This module is now deprecated. Use zhuangzhou.ui.core instead.
"""

from zhuangzhou.ui.core import (
    ZZInterface, ZZTheme, ZZStatus, console,
    ToolSpinner, StreamRenderer,
    format_tokens, format_elapsed, format_speed,
)

__all__ = [
    "ZZInterface", "ZZTheme", "ZZStatus", "console",
    "ToolSpinner", "StreamRenderer",
    "format_tokens", "format_elapsed", "format_speed",
]
