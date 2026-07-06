"""Configuration management."""

import json
import os
import re
import shutil
from rich.panel import Panel
from rich.table import Table
from rich.prompt import Prompt

from .base import BaseCommand

CONFIG_PATH = os.path.expanduser("~/.zhuangzhou/config.json")
SENSITIVE_KEY_RE = re.compile(r"(api[_-]?key|password|passwd|token|secret|credential)", re.I)


def _mask_value(key, value):
    text = str(value)
    if not SENSITIVE_KEY_RE.search(str(key)):
        return text
    if len(text) <= 8:
        return "********"
    return f"{text[:2]}...{text[-2:]}"


class ConfigCommand(BaseCommand):
    name = "config"
    description = "View or set configuration"
    usage = "/config [key] [value]  or  /config --list"
    aliases = ["set", "settings"]

    def execute(self, args: list):
        if not args or (len(args) == 1 and args[0] == "--list"):
            self._show_config()
            return

        if args[0] == "--reset":
            self._reset_config()
            return

        if args[0] == "--edit":
            self._interactive_edit()
            return

        if len(args) >= 2:
            key = args[0]
            value = " ".join(args[1:])
            self._set_config(key, value)
        else:
            # Show single key
            key = args[0]
            val = self.config.get(key)
            if val is not None:
                self.console.print(f"[bright_cyan]{key}[/] = [bright_white]{_mask_value(key, val)}[/]")
            else:
                self.console.print(f"[dim]Key '{key}' not set.[/]")

    def _show_config(self):
        table = Table(
            title="Configuration",
            border_style="bright_cyan",
            title_style="bold bright_cyan",
        )
        table.add_column("Key", style="bright_white")
        table.add_column("Value", style="bright_yellow")
        table.add_column("Source", style="dim")

        defaults = self._defaults()
        all_keys = set(list(self.config.keys()) + list(defaults.keys()))

        for key in sorted(all_keys):
            val = self.config.get(key, defaults.get(key, ""))
            source = "user" if key in self.config else "default"
            table.add_row(key, _mask_value(key, val), source)

        self.console.print(table)
        self.console.print(f"[dim]Config file: {CONFIG_PATH}[/]")
        self.console.print()

    def _set_config(self, key, value):
        if SENSITIVE_KEY_RE.search(str(key)):
            self.console.print(
                "[red]Refusing to save sensitive values in config.[/] "
                "Use an environment variable or OS credential manager instead."
            )
            return
        self.config[key] = value
        self._save()
        self.console.print(
            f"[green]Set[/] [bright_cyan]{key}[/] = [bright_white]{_mask_value(key, value)}[/]"
        )

    def _reset_config(self):
        self.config.clear()
        self.config.update(self._defaults())
        self._save()
        self.console.print("[green]Configuration reset to defaults.[/]")

    def _interactive_edit(self):
        self.console.print(
            Panel(
                "Edit configuration values. Press Enter to keep current.",
                border_style="bright_cyan",
            )
        )
        defaults = self._defaults()
        for key, default in defaults.items():
            current = self.config.get(key, default)
            if SENSITIVE_KEY_RE.search(str(key)):
                continue
            new_val = Prompt.ask(f"[bright_cyan]{key}[/]", default=str(current))
            if new_val != str(current):
                self.config[key] = new_val
        self._save()
        self.console.print("[green]Configuration saved.[/]")

    def _save(self):
        os.makedirs(os.path.dirname(CONFIG_PATH), exist_ok=True)
        with open(CONFIG_PATH, "w", encoding="utf-8") as f:
            json.dump(self.config, f, indent=2)

    def _defaults(self):
        return {
            "unimol_root": os.environ.get("UNIMOL_ROOT", ""),
            "vina_path": os.environ.get("VINA_EXE", shutil.which("vina") or ""),
            "gmx_path": os.environ.get("GMX_BIN", shutil.which("gmx") or ""),
            "pymol_path": os.environ.get("PYMOL_EXE", shutil.which("pymol") or ""),
            "betathink_api": "http://localhost:8000",
            "default_enzyme_class": "",
            "default_output": "results",
        }
