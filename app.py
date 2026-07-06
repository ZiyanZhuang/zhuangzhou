#!/usr/bin/env python3
"""
Zhuangzhou Interactive CLI
==========================
Gemini CLI-style interactive terminal for rational enzyme engineering.
"""

import json
import os
import sys
import time
import shlex
import traceback

# Ensure zhuangzhou is importable
BETATHINK_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BETATHINK_ROOT not in sys.path:
    sys.path.insert(0, BETATHINK_ROOT)

from prompt_toolkit import PromptSession, HTML
from prompt_toolkit.history import FileHistory
from prompt_toolkit.auto_suggest import AutoSuggestFromHistory
from zhuangzhou.ui.core import ZZInterface, console, ZZTheme
from zhuangzhou.commands import COMMANDS

# Config file
CONFIG_PATH = os.path.expanduser("~/.zhuangzhou/config.json")

DEFAULT_CONFIG = {
    "unimol_root": "E:/Code/Uni-Mol-main",
    "vina_path": "E:/computing bio/MDS-gmx/Docking/vina/vina.exe",
    "gmx_path": "E:/gmx2025.1_GPU/bin/gmx.exe",
    "pymol_path": "D:/Soft/Pymol3.03/PyMOLWin.exe",
    "betathink_api": "http://localhost:8000",
    "default_output_dir": "",  # empty = use cwd; set via /workdir command
}


def _render_help_table():
    from rich.table import Table

    table = Table(
        border_style=ZZTheme.MID,
        show_header=True,
        header_style=f"bold {ZZTheme.CYAN}",
        title="Available Commands",
        title_style=f"bold {ZZTheme.GOLD}",
    )
    table.add_column("Command", style=ZZTheme.CYAN, min_width=12)
    table.add_column("Description", style="white")

    for name, cmd in COMMANDS.items():
        table.add_row(f"/{name}", cmd.description)

    table.add_row("/skills", "Show all computational skills")
    table.add_row("/envs", "Check compute environments")
    table.add_row("/clear", "Clear terminal")
    table.add_row("/quit", "Exit")

    console.print(table)
    console.print(
        f"\n[dim]  Tip: You can use natural language directly without /[/]\n"
    )


class ZhuangzhouApp:
    """Main interactive CLI application. Re-designed for Gemini-CLI aesthetics."""

    def __init__(self, config: dict = None):
        self.config = self._load_config()
        if config:
            self.config.update(config)
        self.running = True
        self.history = []
        self.session_tokens = 0
        self.session_queries = 0

        # Restore console for command compatibility
        self.console = console

        # Initialize PromptSession with history
        history_file = os.path.expanduser("~/.zhuangzhou/app_history.txt")
        os.makedirs(os.path.dirname(history_file), exist_ok=True)
        self._history_file = history_file
        self.session = None

        # Memory system
        from zhuangzhou.memory import Journal, SessionContext, KnowledgeStore

        self.journal = Journal()
        self.context = SessionContext()
        self.knowledge = KnowledgeStore()

        # Brain
        from zhuangzhou.brain import BrainConnector

        self.brain = BrainConnector(self.config)

        # Skills
        from zhuangzhou.skills import get_registry

        self.skills = get_registry(self.config)

        # Workflow
        from zhuangzhou.workflow import WorkflowEngine

        self.workflow_engine = WorkflowEngine(self.config)

        # Build alias map
        self.alias_map = {}
        for cmd_name, cmd_cls in COMMANDS.items():
            self.alias_map[cmd_name] = cmd_name
            for alias in getattr(cmd_cls, "aliases", []):
                self.alias_map[alias] = cmd_name

        self.commands = {name: cls(self) for name, cls in COMMANDS.items()}

    def _get_nlp_agent(self):
        if not hasattr(self, "nlp_agent"):
            from zhuangzhou.nlp.agent import ZhuangzhouAgent

            model_override = os.environ.get("ZHUANGZHOU_DEFAULT_MODEL")
            backend_name = (
                model_override.split(":", 1)[0]
                if (model_override and ":" in model_override)
                else None
            )
            self.nlp_agent = ZhuangzhouAgent(
                backend=backend_name,
                model=model_override,
                api_key=os.environ.get("ZHUANGZHOU_API_KEY"),
            )
        return self.nlp_agent

    def _get_plan_mode(self):
        """Lazy-initialize PlanMode. Returns None if unavailable."""
        if not hasattr(self, "_plan_mode"):
            self._plan_mode = None
            try:
                from zhuangzhou.nlp.plan_mode import PlanMode
                agent = self._get_nlp_agent()
                self._plan_mode = PlanMode(
                    backend=agent.backend, registry=agent.registry, enabled=True
                )
            except Exception as e:
                console.print(f"[dim yellow]⚠ Plan Mode init failed: {e}[/]")
        return self._plan_mode

    def _get_prompt_session(self):
        if self.session is None:
            from zhuangzhou.nlp.completer import ZhuangzhouCompleter

            self.session = PromptSession(
                history=FileHistory(self._history_file),
                completer=ZhuangzhouCompleter(),
                auto_suggest=AutoSuggestFromHistory(),
                complete_while_typing=True,
            )
        return self.session

    def _get_slash_context(self):
        """Return (agent, plan_mode) tuple for slash command dispatch."""
        return (self._get_nlp_agent(), self._get_plan_mode())

    def _load_config(self) -> dict:
        config = dict(DEFAULT_CONFIG)
        if os.path.exists(CONFIG_PATH):
            try:
                with open(CONFIG_PATH, encoding="utf-8", errors="replace") as f:
                    config.update(json.load(f))
            except Exception:
                pass
        return config

    def show_banner(self):
        ZZInterface.render_logo()

    def show_help(self):
        _render_help_table()

    def dispatch(self, line: str):
        """Parse and dispatch. Falls back to NLP Agent for natural language."""
        line = line.strip()
        if not line:
            return

        self.history.append(line)
        self.journal.log_command(line)

        # 1. Handle Commands
        is_explicit_cmd = line.startswith("/")
        cmd_line = line[1:] if is_explicit_cmd else line
        parts = shlex.split(cmd_line) if '"' in cmd_line else cmd_line.split()
        if not parts:
            return
        cmd_name = parts[0].lower()
        args = parts[1:]

        is_builtin = cmd_name in (
            "help",
            "h",
            "?",
            "quit",
            "exit",
            "q",
            "clear",
            "cls",
            "envs",
            "skills",
            "brain",
        )
        is_registered = cmd_name in self.alias_map

        if is_explicit_cmd or is_builtin or is_registered:
            if cmd_name in ("quit", "exit", "q"):
                self.running = False
                return
            if cmd_name in ("clear", "cls"):
                ZZInterface.clear()
                return
            if cmd_name in ("help", "h", "?"):
                self.show_help()
                return

            # 1. Handle Slash Commands via Centralized Manager
            if is_explicit_cmd:
                from zhuangzhou.nlp.slash import manager as slash_manager

                if slash_manager.dispatch(line, self._get_slash_context()):
                    return

            # 2. Handle Traditional Registered CLI Commands
            resolved = self.alias_map.get(cmd_name)
            if resolved and resolved in self.commands:
                try:
                    self.commands[resolved].execute(args)
                except Exception as e:
                    ZZInterface.display_message("error", str(e))
                return

        # 2. Treat as Natural Language for the Agent
        self._chat(line)

    def _chat(self, message: str):
        """Invoke the ReAct Agent with live timing and token monitoring."""
        import logging
        import zhuangzhou.nlp.biodesign_mode as _bdm_mod
        from zhuangzhou.ui.core import ZZStatus

        logger = logging.getLogger(__name__)
        agent = self._get_nlp_agent()
        plan_mode = self._get_plan_mode()
        effective_input = message

        # Bio-design Mode intercept
        _biodesign_keywords = [
            "代谢通路", "代谢工程", "合成生物", "设计生产", "生产通路",
            "代谢流", "通量分析", "酶工程", "启动子", "底盘菌",
            "metabolic engineering", "biosynthesis", "生物合成",
            "通路设计", "bio-design", "pathway design",
            "metakg", "synbio_", "底盘细胞", "chassis",
        ]
        _is_biodesign = any(kw in message.lower() for kw in _biodesign_keywords)
        goto_agent = True
        if _is_biodesign:
            try:
                bdm = _bdm_mod.BioDesignMode(backend=agent.backend, registry=agent.registry)
                bd_result = bdm.run(message, console=console)
                if not bd_result.skipped and bd_result.design:
                    effective_input = bd_result.agent_message
                    goto_agent = True
            except KeyboardInterrupt:
                console.print("\n[dim][Bio-design interrupted][/dim]")
                goto_agent = False
            except Exception as e:
                logger.debug("BioDesignMode error: %s", e)

        # Plan Mode preprocessing
        if goto_agent and plan_mode is not None:
            try:
                pm_result = plan_mode.process(message, console=console)
                if not pm_result.skipped:
                    effective_input = pm_result.agent_message
            except KeyboardInterrupt:
                console.print("\n[dim][Plan Mode interrupted][/dim]")
            except Exception as e:
                logger.debug("PlanMode.process error: %s", e)

        if not goto_agent:
            return

        # 命令发出即刻启动实时底栏
        with ZZStatus(action="Reasoning") as status:
            try:
                self.session_queries += 1
                agent.chat(effective_input, status_monitor=status)
                self.session_tokens += status.total_tokens
            except KeyboardInterrupt:
                console.print("\n[yellow]Interrupted.[/]")
            except Exception as e:
                ZZInterface.display_message("error", f"Agent Error: {e}")

    def run_interactive(self):
        """Main REPL loop using Gemini-style prompting."""
        self.show_banner()
        self._get_nlp_agent()
        session = self._get_prompt_session()

        # Matches the look of ZZInterface.get_prompt visually
        # Using fixed colors from ZZTheme to ensure stability
        prompt_html = HTML(
            f'<style color="{ZZTheme.CYAN}">zhuangzhou</style><style color="{ZZTheme.GOLD}"> ❯ </style>'
        )

        while self.running:
            try:
                # Spacing for readability
                console.print("")
                line = session.prompt(prompt_html).strip()
                self.dispatch(line)
            except KeyboardInterrupt:
                continue
            except EOFError:
                self.running = False

        if self.session_queries:
            from zhuangzhou.ui.core import format_tokens

            console.print(
                f"\n[dim]Session: {self.session_queries} queries, {format_tokens(self.session_tokens)} tokens[/dim]"
            )
        console.print(
            f"\n[dim]~{{[/] [bold {ZZTheme.GOLD}]Goodbye[/] [dim]}}~[/]  [italic {ZZTheme.MID}]庄周梦蝶[/]\n"
        )


def main():
    """Entry point for `python -m zhuangzhou`."""
    if len(sys.argv) > 1:
        cmd_line = " ".join(sys.argv[1:])
        if cmd_line.strip() in ("--help", "-h"):
            ZZInterface.render_logo()
            _render_help_table()
            return
        app = ZhuangzhouApp()
        app.dispatch(cmd_line)
    else:
        app = ZhuangzhouApp()
        app.run_interactive()


if __name__ == "__main__":
    main()
