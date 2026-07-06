"""System status and environment check."""

import os
import sys
import platform
from rich.panel import Panel
from rich.table import Table
from rich.columns import Columns

from .base import BaseCommand


class StatusCommand(BaseCommand):
    name = "status"
    description = "Show system status, environment, and version info"
    usage = "/status"
    aliases = ["info", "env"]

    def execute(self, args: list):
        from zhuangzhou import __version__

        # System info
        sys_table = Table(
            title="System",
            border_style="bright_cyan",
            title_style="bold bright_cyan",
            show_header=False,
        )
        sys_table.add_column("Key", style="bright_white", min_width=18)
        sys_table.add_column("Value", style="bright_yellow")

        sys_table.add_row("Zhuangzhou", f"v{__version__}")
        sys_table.add_row("Python", sys.version.split()[0])
        sys_table.add_row("Platform", platform.platform())
        sys_table.add_row("Working Dir", os.getcwd())

        # Check key packages
        packages = {
            "torch": "PyTorch",
            "numpy": "NumPy",
            "scipy": "SciPy",
            "Bio": "BioPython",
            "rich": "Rich",
            "rdkit": "RDKit",
        }
        pkg_table = Table(
            title="Packages",
            border_style="cyan",
            title_style="bold cyan",
            show_header=False,
        )
        pkg_table.add_column("Package", style="bright_white", min_width=14)
        pkg_table.add_column("Version", style="bright_yellow")

        for mod, label in packages.items():
            try:
                m = __import__(mod)
                ver = getattr(m, "__version__", "installed")
                pkg_table.add_row(label, f"[green]{ver}[/]")
            except ImportError:
                pkg_table.add_row(label, "[red]not found[/]")

        self.console.print(Columns([sys_table, pkg_table], padding=2))

        # BeTaThink connection
        self.console.print()
        try:
            import urllib.request
            import json

            base = self.config.get("betathink_api", "http://localhost:8000")
            req = urllib.request.Request(
                f"{base}/api/health", headers={"Accept": "application/json"}
            )
            with urllib.request.urlopen(req, timeout=3) as resp:
                data = json.loads(resp.read())
            self.console.print(f"[green]BeTaThink API:[/] connected ({base})")
            if "services" in data:
                for svc, status in data["services"].items():
                    icon = "[green]OK[/]" if status else "[red]DOWN[/]"
                    self.console.print(f"  {svc}: {icon}")
        except Exception:
            self.console.print(f"[yellow]BeTaThink API:[/] offline (standalone mode)")

        self.console.print()
