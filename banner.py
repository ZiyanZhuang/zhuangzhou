"""
Zhuangzhou Banner - Legacy Redirect
==================================
This module is now deprecated. Use zhuangzhou.ui.core.ZZInterface instead.
"""

from zhuangzhou.ui.core import ZZInterface, console, Text, Style

VERSION = "0.8.5"
WELCOME_TEXT = """[dim]
  Type [bold]/help[/bold] for commands  │  [bold]/skills[/bold] tools  │  [bold]Ctrl+C[/bold] exit
[/dim]"""

def print_banner(c=None):
    if c is None: c = console
    ZZInterface.render_logo()

def get_banner():
    return f"Zhuangzhou Framework v{VERSION}"

def get_banner_rich():
    return []
