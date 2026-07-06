"""Check and display available tools."""

from rich.table import Table
from rich.panel import Panel

from .base import BaseCommand


class ToolsCommand(BaseCommand):
    name = "tools"
    description = "Show available computational tools and their status"
    usage = "/tools"
    aliases = ["toolbox", "check"]

    def execute(self, args: list):
        from zhuangzhou.tools import check_tool_availability, TOOL_REGISTRY

        avail = check_tool_availability(self.config)

        table = Table(
            title="Zhuangzhou Toolbox",
            border_style="bright_cyan",
            title_style="bold bright_cyan",
        )
        table.add_column("Tool", style="bright_white", min_width=22)
        table.add_column("Status", justify="center", min_width=8)
        table.add_column("GPU", justify="center", width=5)
        table.add_column("Description", style="dim")

        for name, info in TOOL_REGISTRY.items():
            ok = avail.get(name, False)
            status = "[green]OK[/]" if ok else "[red]MISSING[/]"
            gpu = "[yellow]GPU[/]" if info.get("requires_gpu") else ""
            table.add_row(name, status, gpu, info.get("description", ""))

        self.console.print(table)

        # External binaries
        ext_table = Table(title="External Binaries", border_style="cyan")
        ext_table.add_column("Binary", style="bright_white", min_width=20)
        ext_table.add_column("Status", justify="center")
        ext_table.add_column("Path", style="dim")

        binaries = {
            "vina_binary": self.config.get(
                "vina_path", "E:/computing bio/MDS-gmx/Docking/vina/vina.exe"
            ),
            "gmx_binary": self.config.get("gmx_path", "E:/gmx2025.1_GPU/bin/gmx.exe"),
            "wsl": "wsl (PATH)",
        }
        for key, path in binaries.items():
            ok = avail.get(key, False)
            status = "[green]OK[/]" if ok else "[red]MISSING[/]"
            ext_table.add_row(key, status, path)

        self.console.print(ext_table)
        self.console.print()
