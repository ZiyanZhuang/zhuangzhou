"""
Zhuangzhou UI Core - Unified Aesthetic Engine v2
=================================================
Single source of truth for the "Cyan-Gold" scientific interface.

v2 improvements:
- Token counting in K units with smooth animation
- Animated spinners for tool execution
- Streaming text display support
- Phase indicators for ReAct loop
- Smoother Live updates with delta rendering
"""

import os
import re
import sys
import time
import math
import threading
from datetime import datetime
from typing import Any, List, Optional, Tuple, Callable

from rich.console import Console, Group
from rich.theme import Theme
from rich.text import Text
from rich.style import Style
from rich.markdown import Markdown
from rich.live import Live
from rich.panel import Panel
from rich.box import ROUNDED, SIMPLE, HEAVY
from rich.table import Table
from rich.columns import Columns
from rich.progress import (
    Progress,
    BarColumn,
    TextColumn,
    TimeElapsedColumn,
    SpinnerColumn,
)
from rich.color import Color
from rich.rule import Rule
from rich.spinner import Spinner

# =============================================================================
# 1. Color Palette & Theme
# =============================================================================


class ZZTheme:
    """Core color constants and rich theme definition."""

    CYAN = "#00E5FF"  # Electric Cyan
    GOLD = "#FFC107"  # Amber Gold
    MID = "#00D4E0"  # Gradient Midpoint
    DARK = "#1A1A1A"  # Dark Background (for panels/rows)
    BG = "#12151C"  # Deep Blue-Black Background
    LAVA = "#FF8F00"  # Lava Orange (Warning/Error)
    DIM = "#888888"  # Dim Gray
    TEXT = "#F5F5F5"  # Pearl White
    GREEN = "#66BB6A"  # Success Green
    RED = "#FF5252"  # Error Red

    THEME = Theme(
        {
            "zz.primary": CYAN,
            "zz.accent": GOLD,
            "zz.mid": MID,
            "zz.dim": f"dim {DIM}",
            "zz.thought": f"italic #00ACC1 on {DARK}",
            "zz.error": f"bold {LAVA}",
            "zz.value": f"bold {GOLD}",
            "zz.header": f"bold black on {MID}",
            "zz.prompt": f"bold {CYAN}",
            "zz.chevron": f"bold {GOLD}",
            "zz.success": f"bold {GREEN}",
            "zz.phase": f"bold {MID}",
        }
    )


# Windows UTF-8 fix — must set before Console init
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

# Global console instance (force_terminal enables VT100 on Windows)
console = Console(theme=ZZTheme.THEME, highlight=False, force_terminal=True)

EMOJI_PATTERN = re.compile(
    "["
    "\U0001f300-\U0001f5ff"
    "\U0001f600-\U0001f64f"
    "\U0001f680-\U0001f6ff"
    "\U0001f700-\U0001f77f"
    "\U0001f780-\U0001f7ff"
    "\U0001f800-\U0001f8ff"
    "\U0001f900-\U0001f9ff"
    "\U0001fa00-\U0001faff"
    "\U00002600-\U000026ff"
    "\U00002700-\U000027bf"
    "]+",
    flags=re.UNICODE,
)

# =============================================================================
# 2. Token Formatting Helpers
# =============================================================================


def format_tokens(count: int) -> str:
    """Format token count in human-readable K units.

    0       -> "0"
    500     -> "500"
    1000    -> "1.0k"
    1500    -> "1.5k"
    23456   -> "23.5k"
    150000  -> "150k"
    1200000 -> "1.2M"
    """
    if count < 1000:
        return str(count)
    elif count < 9_950:
        return f"{count / 1000:.1f}k"   # 1.0k – 9.9k
    elif count < 999_500:
        return f"{round(count / 1000)}k" # 10k – 999k (integer)
    else:
        return f"{count / 1_000_000:.1f}M"


def format_elapsed(seconds: float) -> str:
    """Format elapsed time compactly.

    < 60s   -> "12.3s"
    < 3600s -> "2m34s"
    >= 3600 -> "1h05m"
    """
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        m, s = divmod(int(seconds), 60)
        return f"{m}m{s:02d}s"
    else:
        h, remainder = divmod(int(seconds), 3600)
        m = remainder // 60
        return f"{h}h{m:02d}m"


def format_speed(tokens: int, elapsed: float) -> str:
    """Format token generation speed."""
    if elapsed <= 0:
        return ""
    tps = tokens / elapsed
    if tps >= 1000:
        return f"{tps / 1000:.1f}k/s"
    elif tps >= 1:
        return f"{tps:.0f}/s"
    else:
        return f"{tps:.1f}/s"


def sanitize_display_text(text: Any) -> str:
    """Strip emoji and presentation selectors from rendered dialogue text."""
    if text is None:
        return ""
    if not isinstance(text, str):
        text = str(text)
    text = text.replace("\ufe0f", "").replace("\u200d", "")
    text = EMOJI_PATTERN.sub("", text)
    return text


def build_context_bar(status: dict) -> Text:
    """Build a compact context usage bar shown before each prompt."""
    ratio = float(status.get("usage_ratio", 0.0) or 0.0)
    estimated = int(status.get("estimated_tokens", 0) or 0)
    limit = int(status.get("context_limit", 0) or 0)
    last_input = int(status.get("last_input_tokens", 0) or 0)
    last_output = int(status.get("last_output_tokens", 0) or 0)
    last_total = int(status.get("last_total_tokens", 0) or 0)
    backend = str(status.get("backend", "") or "").strip()
    model = str(status.get("model", "") or "").strip() or backend or "unknown"
    message_count = int(status.get("message_count", 0) or 0)
    layers = status.get("layers_active", []) or ["none"]
    reset_recommended = bool(status.get("reset_recommended"))

    if ratio >= 0.90:
        accent = ZZTheme.RED
    elif ratio >= 0.80:
        accent = ZZTheme.LAVA
    elif ratio >= 0.60:
        accent = ZZTheme.GOLD
    else:
        accent = ZZTheme.CYAN

    line = Text()
    line.append("  ctx~ ", style=f"bold {accent}")
    line.append(format_tokens(estimated), style=f"bold {accent}")
    line.append("/", style=ZZTheme.DIM)
    line.append(format_tokens(limit), style=f"bold {accent}")
    line.append(f" ({int(ratio * 100)}%)", style=ZZTheme.DIM)
    if last_total > 0:
        line.append("  │  token ", style=ZZTheme.DIM)
        line.append(format_tokens(last_total), style=f"bold {ZZTheme.CYAN}")
        line.append(" (", style=ZZTheme.DIM)
        line.append(f"in:{format_tokens(last_input)}", style=ZZTheme.DIM)
        line.append(" ", style=ZZTheme.DIM)
        line.append(f"out:{format_tokens(last_output)}", style=ZZTheme.DIM)
        line.append(")", style=ZZTheme.DIM)
    line.append("  │  mem ", style=ZZTheme.DIM)
    line.append("+".join(layers), style=f"bold {ZZTheme.MID}")
    line.append("  │  msgs ", style=ZZTheme.DIM)
    line.append(str(message_count), style=f"bold {ZZTheme.GOLD}")
    if reset_recommended:
        line.append("  │  ", style=ZZTheme.DIM)
        line.append("consider /reset", style=f"bold {accent}")
    line.append("  │  model ", style=ZZTheme.DIM)
    line.append(model, style="white")
    return line


def build_context_hint(status: dict) -> Optional[Text]:
    """Build a secondary hint line for session memory and reset guidance."""
    session_summary = (status.get("session_summary") or "").strip()
    reset_reason = (status.get("reset_reason") or "").strip()
    parts: List[Text] = []

    if session_summary and session_summary != "No active context":
        summary = session_summary
        if len(summary) > 120:
            summary = summary[:117] + "..."
        item = Text()
        item.append("  active ", style=ZZTheme.DIM)
        item.append(summary, style="white")
        parts.append(item)

    if reset_reason:
        item = Text()
        item.append("  note ", style=ZZTheme.DIM)
        item.append(reset_reason, style=f"italic {ZZTheme.GOLD}")
        parts.append(item)

    if not parts:
        return None

    line = Text()
    for idx, part in enumerate(parts):
        if idx:
            line.append("  │  ", style=ZZTheme.DIM)
        line.append_text(part)
    return line


def build_response_header(
    status: dict, elapsed: float = 0.0, input_tokens: int = 0, output_tokens: int = 0,
    usage_estimated: bool = False,
) -> Text:
    """Build the assistant panel header with time, token usage, context, and model."""
    ratio = float(status.get("usage_ratio", 0.0) or 0.0)
    estimated = int(status.get("estimated_tokens", 0) or 0)
    limit = int(status.get("context_limit", 0) or 0)
    backend = str(status.get("backend", "") or "").strip()
    model = str(status.get("model", "") or "").strip() or backend or "unknown"
    total_tokens = int((input_tokens or 0) + (output_tokens or 0))

    if ratio >= 0.90:
        accent = ZZTheme.RED
    elif ratio >= 0.80:
        accent = ZZTheme.LAVA
    elif ratio >= 0.60:
        accent = ZZTheme.GOLD
    else:
        accent = ZZTheme.CYAN

    line = Text()
    line.append("time ", style=ZZTheme.DIM)
    line.append(format_elapsed(elapsed or 0.0), style=f"bold {ZZTheme.GOLD}")
    if total_tokens > 0:
        line.append("  │  token ", style=ZZTheme.DIM)
        line.append(('~' if usage_estimated else '') + format_tokens(total_tokens), style=f"bold {ZZTheme.CYAN}")
        line.append(" (", style=ZZTheme.DIM)
        line.append(f"in:{format_tokens(int(input_tokens or 0))}", style=ZZTheme.DIM)
        line.append(" ", style=ZZTheme.DIM)
        line.append(f"out:{format_tokens(int(output_tokens or 0))}", style=ZZTheme.DIM)
        line.append(")", style=ZZTheme.DIM)
    line.append("\nctx~ ", style=ZZTheme.DIM)
    line.append(format_tokens(estimated), style=f"bold {accent}")
    line.append("/", style=ZZTheme.DIM)
    line.append(format_tokens(limit), style=f"bold {accent}")
    line.append(f" ({int(ratio * 100)}%)", style=ZZTheme.DIM)
    line.append("  │  model ", style=ZZTheme.DIM)
    line.append(model, style="white")
    return line


def build_response_footer(status: dict) -> Optional[Group]:
    """Build footer lines for assistant panels with memory and context hints."""
    layers = status.get("layers_active", []) or ["none"]
    message_count = int(status.get("message_count", 0) or 0)
    reset_recommended = bool(status.get("reset_recommended"))

    footer = Text()
    footer.append("mem ", style=ZZTheme.DIM)
    footer.append("+".join(layers), style=f"bold {ZZTheme.MID}")
    footer.append("  │  msgs ", style=ZZTheme.DIM)
    footer.append(str(message_count), style=f"bold {ZZTheme.GOLD}")
    if reset_recommended:
        footer.append("  │  ", style=ZZTheme.DIM)
        footer.append("consider /reset", style=f"bold {ZZTheme.LAVA}")

    hint = build_context_hint(status)
    if hint is not None:
        return Group(footer, hint)
    return Group(footer)


def build_panel_separator() -> Rule:
    """Build a gold divider between the panel status line and markdown body."""
    return Rule(style=ZZTheme.GOLD, characters="─")


# =============================================================================
# 3. Gradient Helpers
# =============================================================================


def lerp_rgb(
    c1: Tuple[int, int, int], c2: Tuple[int, int, int], t: float
) -> Tuple[int, int, int]:
    """Linear interpolation for RGB colors."""
    t = max(0.0, min(1.0, t))
    return (
        int(c1[0] + (c2[0] - c1[0]) * t),
        int(c1[1] + (c2[1] - c1[1]) * t),
        int(c1[2] + (c2[2] - c1[2]) * t),
    )


def get_gradient_text(text: str, total_width: int, bold: bool = True) -> Text:
    """Applies a Cyan -> Gold horizontal gradient to text."""
    c_left = (0, 229, 255)  # CYAN
    c_right = (255, 193, 7)  # GOLD

    result = Text()
    for i, ch in enumerate(text):
        if ch == " ":
            result.append(ch)
        else:
            t = i / max(total_width - 1, 1)
            r, g, b = lerp_rgb(c_left, c_right, t)
            style = Style(color=Color.from_rgb(r, g, b), bold=bold)
            result.append(ch, style)
    return result


# =============================================================================
# 4. Spinner Frames (Braille-based smooth animation)
# =============================================================================

# Braille dot spinner — ultra-smooth, scientific aesthetic
SPINNER_FRAMES = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]
SPINNER_INTERVAL = 0.08  # seconds per frame

# Phase indicators for ReAct loop
PHASE_ICONS = {
    "think": "◆",  # Thinking
    "act": "▶",  # Executing tool
    "observe": "◉",  # Observing result
    "done": "✓",  # Completed
    "error": "✗",  # Failed
    "wait": "◇",  # Waiting for confirmation
}

# =============================================================================
# 5. Live Monitoring (v2 — smooth token counter)
# =============================================================================


class ZZStatus:
    """Real-time Engineering Monitor with smooth token counting and spinners.

    Implements __rich_console__ so that Rich's Live auto-refresh calls _render()
    on every tick — this makes the elapsed timer and spinner animate smoothly
    without requiring explicit update() calls.

    Usage:
        with ZZStatus(action="Reasoning") as monitor:
            monitor.update(tokens=150, action="search_pubmed")
            ...
    """

    def __init__(self, start_time: float = None, action: str = "PROCESSING"):
        self.start_time = start_time if start_time is not None else time.perf_counter()
        self.total_tokens = 0
        self.input_tokens = 0
        self.output_tokens = 0
        self.usage_estimated = True
        self.active_action = action
        self.phase = "think"
        self._live = None
        self._frame_idx = 0
        self._last_frame_time = 0.0
        self._tool_count = 0
        self._tool_done = 0
        self._lock = threading.Lock()
        # Smooth token interpolation
        self._display_tokens = 0.0
        self._target_tokens = 0

    # -- Rich renderable protocol --
    # Live calls __rich_console__ on every refresh tick, so _render() runs
    # automatically at refresh_per_second rate — no manual push needed.

    def __rich_console__(self, console, options):
        yield self._render()

    def _get_spinner_char(self) -> str:
        """Get current spinner frame, advancing based on elapsed time."""
        now = time.perf_counter()
        if now - self._last_frame_time >= SPINNER_INTERVAL:
            self._frame_idx = (self._frame_idx + 1) % len(SPINNER_FRAMES)
            self._last_frame_time = now
        return SPINNER_FRAMES[self._frame_idx]

    def _smooth_tokens(self) -> int:
        """Smoothly interpolate displayed token count toward target."""
        if self._display_tokens < self._target_tokens:
            diff = self._target_tokens - self._display_tokens
            # Ease-out: faster when far, slower when close
            step = max(1, diff * 0.3)
            self._display_tokens = min(self._display_tokens + step, self._target_tokens)
        return int(self._display_tokens)

    def _render(self) -> Text:
        elapsed = time.perf_counter() - self.start_time
        spinner = self._get_spinner_char()
        display_tok = self._smooth_tokens()

        footer = Text()

        # Spinner + phase icon
        phase_icon = PHASE_ICONS.get(self.phase, "◆")
        footer.append(f"  {spinner} ", style=ZZTheme.CYAN)
        footer.append(f"{phase_icon} ", style=ZZTheme.MID)

        # Elapsed time — always shown
        footer.append(format_elapsed(elapsed), style=ZZTheme.GOLD)

        # Token count in K units
        if display_tok > 0:
            footer.append("  │  ", style=ZZTheme.DIM)
            approximate = self.usage_estimated or display_tok != self.total_tokens
            footer.append(('~' if approximate else '') + format_tokens(display_tok), style=ZZTheme.CYAN)
            footer.append(" tokens", style="dim")
            # Speed
            if elapsed > 0.5 and self.output_tokens > 0:
                speed = format_speed(self.output_tokens, elapsed)
                if speed:
                    footer.append(f"  out {'~' if self.usage_estimated else ''}{speed}", style="dim")

        # Active action
        if self.active_action:
            footer.append("  │  ", style=ZZTheme.DIM)
            footer.append(self.active_action, style=f"italic {ZZTheme.CYAN}")

        # Tool progress (if parallel execution)
        if self._tool_count > 1:
            footer.append(f"  [{self._tool_done}/{self._tool_count}]", style="dim")

        return footer

    def update(
        self,
        tokens: int = 0,
        action: str = None,
        phase: str = None,
        input_tokens: int = 0,
        output_tokens: int = 0,
    ):
        with self._lock:
            if tokens or input_tokens or output_tokens:
                self.usage_estimated = True
            if tokens:
                self.total_tokens += tokens
                self._target_tokens = self.total_tokens
            if input_tokens:
                self.input_tokens += input_tokens
            if output_tokens:
                self.output_tokens += output_tokens
            if action is not None:
                self.active_action = action
            if phase is not None:
                self.phase = phase
            # No need to manually push to Live — __rich_console__ handles it

    def set_usage(self, input_tokens: int = 0, output_tokens: int = 0, *, estimated: bool = False):
        """Replace counters with exact usage reported by the backend."""
        with self._lock:
            self.usage_estimated = estimated
            self.input_tokens = int(input_tokens or 0)
            self.output_tokens = int(output_tokens or 0)
            self.total_tokens = self.input_tokens + self.output_tokens
            self._target_tokens = self.total_tokens
            if self._display_tokens > self._target_tokens:
                self._display_tokens = float(self._target_tokens)

    def set_tool_progress(self, total: int, done: int):
        with self._lock:
            self._tool_count = total
            self._tool_done = done

    def __enter__(self):
        self._live = Live(
            self,  # pass self as renderable — Live calls __rich_console__ on each tick
            console=console,
            refresh_per_second=12,
            transient=True,
        )
        self._live.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is not None:
            interrupted = issubclass(exc_type, KeyboardInterrupt)
            self.phase = 'wait' if interrupted else 'error'
            self.active_action = 'Interrupted' if interrupted else 'Failed'
        if self._live:
            self._live.stop()
        self._print_summary()

    def _print_summary(self):
        """Print a clean, final status line after the Live display ends."""
        elapsed = time.perf_counter() - self.start_time

        summary = Text()
        summary.append("  └─ ", style=ZZTheme.DIM)

        phase = self.phase if self.phase in ('error', 'wait') else 'done'
        icon = PHASE_ICONS[phase]
        icon_style = {'done': ZZTheme.GREEN, 'wait': ZZTheme.GOLD, 'error': ZZTheme.RED}[phase]
        summary.append(f"{icon} ", style=icon_style)
        if phase in ('error', 'wait'):
            summary.append(f'{self.active_action}  │  ', style=icon_style)

        # Time
        summary.append(format_elapsed(elapsed), style=ZZTheme.GOLD)

        # Tokens breakdown — only show in/out detail when both have real values
        if self.total_tokens > 0:
            summary.append("  │  ", style=ZZTheme.DIM)
            summary.append(('~' if self.usage_estimated else '') + format_tokens(self.total_tokens), style=ZZTheme.CYAN)
            summary.append(" tokens", style="dim")
            if self.input_tokens > 0 or self.output_tokens > 0:
                summary.append(
                    f" (in:{format_tokens(self.input_tokens)}"
                    f" out:{format_tokens(self.output_tokens)})",
                    style="dim",
                )
            if elapsed > 0.5 and self.output_tokens > 0:
                speed = format_speed(self.output_tokens, elapsed)
                if speed:
                    summary.append(f"  out {'~' if self.usage_estimated else ''}{speed}", style="dim")

        console.print(summary)


# =============================================================================
# 6. Tool Execution Spinner
# =============================================================================


class ToolSpinner:
    """Animated inline spinner for individual tool execution.

    Usage:
        with ToolSpinner("search_pubmed") as ts:
            result = do_work()
            ts.done(success=True, elapsed=1.2)
    """

    def __init__(self, name: str, monitor: "ZZStatus" = None):
        self.name = name
        self.monitor = monitor
        self._start = time.perf_counter()
        self._live = None

    def _render(self, phase: str = "act") -> Text:
        elapsed = time.perf_counter() - self._start
        idx = int(elapsed / SPINNER_INTERVAL) % len(SPINNER_FRAMES)
        spinner = SPINNER_FRAMES[idx]

        line = Text()
        line.append("  ┃ ", style=ZZTheme.MID)
        line.append(f"{spinner} ", style=ZZTheme.CYAN)
        line.append("ACT ", style=Style(color=ZZTheme.CYAN, bold=True))
        line.append(self.name, style="white")
        line.append(f"  {format_elapsed(elapsed)}", style="dim")
        return line

    def done(self, success: bool = True, elapsed: float = None):
        """Print final status line (replaces spinner)."""
        if elapsed is None:
            elapsed = time.perf_counter() - self._start

        line = Text()
        line.append("  ┃ ", style=ZZTheme.MID)
        if success:
            line.append(f"{PHASE_ICONS['done']} ", style=ZZTheme.GREEN)
            line.append("DONE ", style=Style(color=ZZTheme.GOLD, bold=True))
        else:
            line.append(f"{PHASE_ICONS['error']} ", style=ZZTheme.RED)
            line.append("FAIL ", style=Style(color=ZZTheme.RED, bold=True))
        line.append(self.name, style="white")
        line.append(f"  {format_elapsed(elapsed)}", style="dim")
        console.print(line)

    def __rich_console__(self, console, options):
        """Rich renderable protocol — allows Live to auto-refresh this spinner."""
        yield self._render()

    def __enter__(self):
        if self.monitor and self.monitor._live:
            # Don't create nested Live — just update monitor
            self.monitor.update(action=self.name, phase="act")
            return self
        # Standalone spinner: pass self as renderable so Live auto-refreshes
        self._live = Live(
            self,  # renderable = self (uses __rich_console__)
            console=console,
            refresh_per_second=12,
            transient=True,
        )
        self._live.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._live:
            self._live.stop()


# =============================================================================
# 7. Streaming Text Renderer
# =============================================================================


class StreamRenderer:
    """Renders LLM output token-by-token with smooth display.

    Usage:
        renderer = StreamRenderer()
        renderer.start()
        for chunk in stream:
            renderer.feed(chunk)
        final_text = renderer.finish()
    """

    def __init__(
        self,
        show_panel: bool = True,
        status_provider: Callable[[], dict] = None,
        monitor: "ZZStatus" = None,
    ):
        self.buffer = []
        self.show_panel = show_panel
        self.status_provider = status_provider
        self.monitor = monitor
        self._live = None
        self._last_render = 0.0
        self._render_interval = 0.05  # 50ms between renders for smoothness

    def _render_content(self) -> Panel:
        text = sanitize_display_text("".join(self.buffer))
        if not text.strip():
            text = "..."
        md = Markdown(text, code_theme="bw")
        status = self.status_provider() if self.status_provider else {}
        elapsed = 0.0
        input_tokens = 0
        output_tokens = 0
        if self.monitor is not None:
            elapsed = time.perf_counter() - self.monitor.start_time
            input_tokens = int(getattr(self.monitor, "input_tokens", 0) or 0)
            output_tokens = int(getattr(self.monitor, "output_tokens", 0) or 0)
        header = build_response_header(status, elapsed, input_tokens, output_tokens,
                                       usage_estimated=bool(getattr(self.monitor, 'usage_estimated', False)))
        footer = build_response_footer(status)
        body = Group(header, build_panel_separator(), md, Text(""), footer)
        if self.show_panel:
            return Panel(
                body,
                border_style=ZZTheme.CYAN,
                box=ROUNDED,
                title="[bold zz.primary] zhuangzhou [/]",
                title_align="left",
                subtitle="[dim] Rational Engine [/]",
                subtitle_align="right",
                padding=(1, 2),
                style=f"on {ZZTheme.BG}",
            )
        return body

    def start(self):
        self._live = Live(
            self._render_content(),
            console=console,
            refresh_per_second=15,
            transient=False,  # keep final render
        )
        self._live.start()

    def feed(self, chunk: str):
        """Feed a text chunk from the stream."""
        if not chunk:
            return
        self.buffer.append(sanitize_display_text(chunk))
        now = time.perf_counter()
        if self._live and (now - self._last_render >= self._render_interval):
            try:
                self._live.update(self._render_content())
                self._last_render = now
            except Exception:
                pass

    def finish(self) -> str:
        """Finalize the display and return the complete text."""
        if self._live:
            # Final render
            try:
                self._live.update(self._render_content())
            except Exception:
                pass
            self._live.stop()
            self._live = None
        return "".join(self.buffer)


# =============================================================================
# 8. Core Interface Components
# =============================================================================


class ZZInterface:
    """Unified rendering library for Zhuangzhou."""

    @classmethod
    def render_logo(cls):
        """Renders the high-fidelity gradient banner."""
        zhuang = [
            "  ███████╗██╗  ██╗██╗   ██╗ █████╗ ███╗   ██╗ ██████╗ ",
            "  ╚══███╔╝██║  ██║██║   ██║██╔══██╗████╗  ██║██╔════╝ ",
            "    ███╔╝ ███████║██║   ██║███████║██╔██╗ ██║██║  ███╗",
            "   ███╔╝  ██╔══██║██║   ██║██╔══██║██║╚██╗██║██║   ██║",
            "  ███████╗██║  ██║╚██████╔╝██║  ██║██║ ╚████║╚██████╔╝",
            "  ╚══════╝╚═╝  ╚═╝ ╚═════╝ ╚═╝  ╚═╝╚═╝  ╚═══╝ ╚═════╝ ",
        ]
        zhou = [
            "  ███████╗██╗  ██╗ ██████╗ ██╗   ██╗",
            "  ╚══███╔╝██║  ██║██╔═══██╗██║   ██║",
            "    ███╔╝ ███████║██║   ██║██║   ██║",
            "   ███╔╝  ██╔══██║██║   ██║██║   ██║",
            "  ███████╗██║  ██║╚██████╔╝╚██████╔╝",
            "  ╚══════╝╚═╝  ╚═╝ ╚═════╝  ╚═════╝ ",
        ]

        logo_width = max(len(line) for line in zhuang)
        from zhuangzhou import __version__

        started = datetime.now().strftime("%Y-%m-%d %H:%M")
        cwd = os.getcwd()
        if len(cwd) > 40:
            cwd = "..." + cwd[-37:]

        metadata = Table.grid(padding=(0, 1), expand=True)
        metadata.add_column(style=ZZTheme.DIM, width=10, no_wrap=True)
        metadata.add_column(style=f'bold {ZZTheme.GOLD}', ratio=1, no_wrap=True, overflow='ellipsis')
        for label, value in [('Version:', f'v{__version__}'), ('Interface:', 'Commands + Agent'),
                             ('Started:', started), ('Path:', cwd)]:
            metadata.add_row(label, Text(value))

        console.print('')
        if console.width >= 100:
            banner = Table.grid(padding=(0, 3), expand=True)
            banner.add_column(width=logo_width, no_wrap=True)
            banner.add_column(ratio=1)
            banner.add_row(Group(*(get_gradient_text(row, logo_width) for row in zhuang + zhou)), metadata)
            console.print(banner)
        else:
            console.print(get_gradient_text('  庄周 · Zhuangzhou', 20))
            console.print(metadata)

        # Tagline with gradient midpoint color
        mid_r, mid_g, mid_b = lerp_rgb((0, 229, 255), (255, 193, 7), 0.5)
        tag_style = Style(color=f"rgb({mid_r},{mid_g},{mid_b})", italic=True, dim=True)
        console.print(
            Text(
                "\n  Rational Enzyme Engineering\n  Chemistry over consensus · LLM as engine, not oracle.",
                style=tag_style,
            )
        )

        # Author attribution
        author_style = Style(color=ZZTheme.DIM, italic=True)
        console.print(Text("  Ziyan Zhuang · Tianjin University", style=author_style))
        console.print(Text("  Qianyu Zhao · Xinjiang University", style=author_style))
        console.print(Text("  Zifan Zhao · Lanzhou University", style=author_style))
        console.print(Text(
            "  Shengfei Wang\n"
            "    University of Chinese Academy of Sciences\n"
            "    Wuhan Institute of Virology, Chinese Academy of Sciences",
            style=author_style,
        ))

        console.print(Text("\n  Academic supervisors", style=ZZTheme.DIM))
        for name, university, school in (
            ("Delvin Ce Zhang", "University of Sheffield", "School of Computer Science"),
            ("Guang-Rong Zhao", "Tianjin University", "School of Synthetic Biology and Biomanufacturing"),
        ):
            console.print(Text(f"  {name} · {university}\n    {school}", style=author_style))

        # Thin separator
        console.print(Text("  " + "─" * min(60, max(0, console.width - 2)), style="dim"))
        console.print("")

    @classmethod
    def get_prompt(cls, protein: str = None) -> Text:
        prompt = Text()
        prompt.append("\n")
        prompt.append("zhuangzhou", style="zz.prompt")
        if protein:
            prompt.append(f" ({protein})", style="zz.dim")
        prompt.append(" ❯ ", style="zz.chevron")
        return prompt

    @classmethod
    def display_message(
        cls, role: str, content: str, status: dict = None, monitor: "ZZStatus" = None
    ):
        if not content:
            return

        content = sanitize_display_text(content)

        if role == "thought":
            msg = Text()
            for line in content.strip().split("\n"):
                msg.append("  ┃ ", style=ZZTheme.GOLD)
                msg.append(f"{line}\n", style="zz.thought")
            console.print(msg)

        elif role == "assistant":
            md = Markdown(content, code_theme="bw")
            status = status or {}
            elapsed = 0.0
            input_tokens = 0
            output_tokens = 0
            if monitor is not None:
                elapsed = time.perf_counter() - monitor.start_time
                input_tokens = int(getattr(monitor, "input_tokens", 0) or 0)
                output_tokens = int(getattr(monitor, "output_tokens", 0) or 0)
            header = build_response_header(status, elapsed, input_tokens, output_tokens,
                                           usage_estimated=bool(getattr(monitor, 'usage_estimated', False)))
            footer = build_response_footer(status)
            assistant_panel = Panel(
                Group(header, build_panel_separator(), md, Text(""), footer),
                border_style=ZZTheme.CYAN,
                box=ROUNDED,
                title="[bold zz.primary] zhuangzhou [/]",
                title_align="left",
                subtitle="[dim] Rational Engine [/]",
                subtitle_align="right",
                padding=(1, 2),
                style=f"on {ZZTheme.BG}",
            )
            console.print(Text(""))
            console.print(assistant_panel)
            console.print("")

        elif role == 'confirmation':
            console.print(Panel(Text(content), title='Confirmation required',
                                border_style=ZZTheme.GOLD, box=ROUNDED, padding=(0, 2)))

        elif role == "error":
            error_panel = Panel(
                Text(content, style="white"),
                title="[bold #FF5252] ERROR [/]",
                border_style="#FF5252",
                box=ROUNDED,
                padding=(0, 2),
                style="on #2A1212",
            )
            console.print(error_panel)

    @classmethod
    def print_action(cls, name: str, status: str = "RUNNING", elapsed: float = None):
        """Render action status with Cyan/Gold accents and phase icons."""
        msg = Text()
        msg.append("  ┃ ", style=ZZTheme.MID)
        if status == "RUNNING":
            msg.append(f"{PHASE_ICONS['act']} ", style=ZZTheme.CYAN)
            msg.append("ACT  ", style=Style(color=ZZTheme.CYAN, bold=True))
            msg.append(name, style="white")
        elif status == "DONE":
            msg.append(f"{PHASE_ICONS['done']} ", style=ZZTheme.GREEN)
            msg.append("DONE ", style=Style(color=ZZTheme.GOLD, bold=True))
            msg.append(name, style="white")
            if elapsed:
                msg.append(f"  {format_elapsed(elapsed)}", style="dim")
        elif status == "FAILED":
            msg.append(f"{PHASE_ICONS['error']} ", style=ZZTheme.RED)
            msg.append("FAIL ", style="bold #FF5252")
            msg.append(name, style="white")

        console.print(msg)

    @classmethod
    def display_status(
        cls, action: str, status: str = "RUNNING", elapsed: float = None
    ):
        """Alias for print_action (legacy compatibility)."""
        cls.print_action(action, status, elapsed)

    @classmethod
    def print_phase(cls, phase: str, detail: str = ""):
        """Print a ReAct phase transition line."""
        icon = PHASE_ICONS.get(phase, "◆")
        phase_names = {
            "think": "THINK",
            "act": "ACT",
            "observe": "OBSERVE",
            "done": "COMPLETE",
            "error": "ERROR",
            "wait": "AWAITING",
        }
        label = phase_names.get(phase, phase.upper())

        line = Text()
        line.append("  ┃ ", style=ZZTheme.MID)
        line.append(f"{icon} ", style=ZZTheme.CYAN if phase != "error" else ZZTheme.RED)
        line.append(f"{label}", style="zz.phase")
        if detail:
            line.append(f"  {detail}", style="dim")
        console.print(line)

    @classmethod
    def print_token_summary(cls, input_tok: int, output_tok: int, elapsed: float):
        """Print a compact token usage summary line."""
        total = input_tok + output_tok
        line = Text()
        line.append("  └─ ", style=ZZTheme.DIM)
        line.append(f"{PHASE_ICONS['done']} ", style=ZZTheme.GREEN)
        line.append(format_tokens(total), style=ZZTheme.CYAN)
        line.append(" tokens", style="dim")
        line.append(
            f" (in:{format_tokens(input_tok)} out:{format_tokens(output_tok)})",
            style="dim",
        )
        line.append(f"  {format_elapsed(elapsed)}", style=ZZTheme.GOLD)
        if elapsed > 0:
            line.append(f"  {format_speed(output_tok, elapsed)}", style="dim")
        console.print(line)

    @classmethod
    def create_table(cls, title: str, columns: list) -> Table:
        """Create high-contrast scientific table."""
        table = Table(
            title=Text(f"┃ {title}", style=f"bold {ZZTheme.CYAN}"),
            title_justify="left",
            header_style="zz.header",
            box=SIMPLE,
            show_edge=False,
            row_styles=["none", f"on {ZZTheme.DARK}"],
        )
        for col in columns:
            table.add_column(col)
        return table

    @classmethod
    def get_progress(cls) -> Progress:
        """Cyan-Gold gradient progress bar with spinner."""
        return Progress(
            SpinnerColumn("dots", style=ZZTheme.CYAN),
            TextColumn("  ┃ ", style=ZZTheme.MID),
            TextColumn("[zz.primary]{task.description}"),
            BarColumn(
                bar_width=40,
                style="dim",
                complete_style=ZZTheme.CYAN,
                finished_style=ZZTheme.GOLD,
            ),
            TextColumn("[zz.accent]{task.percentage:>3.0f}%"),
            TimeElapsedColumn(),
            console=console,
            transient=True,
        )

    @classmethod
    def select_one(
        cls, title: str, options: list, default: str = None
    ) -> Optional[str]:
        """
        Interactive vertical selection menu for CLI.
        'options' can be a list of strings or (value, label) tuples.
        Returns the selected value or None if cancelled.
        """
        from prompt_toolkit.application import Application
        from prompt_toolkit.key_binding import KeyBindings
        from prompt_toolkit.layout.containers import Window, HSplit, VSplit
        from prompt_toolkit.layout.controls import FormattedTextControl
        from prompt_toolkit.layout.layout import Layout
        from prompt_toolkit.formatted_text import HTML

        if not options:
            return None

        # Standardize options to (value, label)
        choices = []
        for o in options:
            if isinstance(o, tuple):
                choices.append(o)
            else:
                choices.append((o, o))

        # Initial state
        state = {"index": 0, "cancelled": False, "value": None}
        if default:
            for i, (val, _) in enumerate(choices):
                if val == default:
                    state["index"] = i
                    break

        def get_text():
            text = []
            text.append(("", f"\n  \u2503 {title}\n"))
            for i, (val, label) in enumerate(choices):
                if i == state["index"]:
                    text.append((f"bold {ZZTheme.CYAN}", f"   \u27a4 {label}\n"))
                else:
                    text.append((ZZTheme.DIM, f"     {label}\n"))
            text.append(("", "  [Up/Down to select, Enter to confirm, Esc to cancel]"))
            return text

        kb = KeyBindings()

        @kb.add("up")
        def _(event):
            state["index"] = (state["index"] - 1) % len(choices)

        @kb.add("down")
        def _(event):
            state["index"] = (state["index"] + 1) % len(choices)

        @kb.add("enter")
        def _(event):
            state["value"] = choices[state["index"]][0]
            event.app.exit()

        @kb.add("escape")
        @kb.add("c-c")
        def _(event):
            state["cancelled"] = True
            event.app.exit()

        control = FormattedTextControl(get_text)
        win = Window(content=control, height=len(choices) + 4)

        # In-place application (not full-screen)
        app = Application(
            layout=Layout(HSplit([win])),
            key_bindings=kb,
            full_screen=False,
        )

        app.run()

        if state["cancelled"]:
            return None
        return state["value"]

    @classmethod
    def print_warning(cls, text: str):
        console.print(f"\n[zz.error] ┃ WARNING: {text}[/]\n")

    @classmethod
    def print_separator(cls, label: str = ""):
        """Print a thin separator line."""
        if label:
            rule = Rule(label, style="dim", characters="─")
        else:
            rule = Text("  " + "─" * 60, style="dim")
        console.print(rule)

    @classmethod
    def clear(cls):
        console.clear()
        cls.render_logo()

    @classmethod
    def print_skill_card(
        cls,
        name: str,
        category: str,
        description: str,
        risk: str = "low",
        time_est: str = None,
    ):
        """Print a compact skill info card."""
        risk_style = {
            "low": "dim",
            "medium": ZZTheme.GOLD,
            "high": ZZTheme.LAVA,
        }.get(risk, "dim")

        line = Text()
        line.append("  ", style="")
        line.append(f"[{category}]", style="dim")
        line.append(f" {name}", style=f"bold {ZZTheme.CYAN}")
        if risk != "low":
            line.append(f"  ⚠{risk}", style=risk_style)
        if time_est:
            line.append(f"  ~{time_est}", style="dim")
        console.print(line)
        if description:
            # Truncate long descriptions
            desc = description[:80] + "..." if len(description) > 80 else description
            console.print(f"    {desc}", style="dim")
