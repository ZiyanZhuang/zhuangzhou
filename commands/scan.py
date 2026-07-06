"""ESM-2 scan result loading and analysis."""

import os
from rich.table import Table
from rich.panel import Panel

from .base import BaseCommand


class ScanCommand(BaseCommand):
    name = "scan"
    description = "Load and display ESM-2 mutation scan results"
    usage = "/scan <esm2_dir1> [esm2_dir2 ...] [--top 20]"
    aliases = ["esm2", "plm"]

    def execute(self, args: list):
        if not args:
            self.console.print("[red]Usage:[/] /scan <dir1> [dir2 ...] [--top N]")
            return

        top_n = 20
        dirs = []
        i = 0
        while i < len(args):
            if args[i] == "--top" and i + 1 < len(args):
                top_n = int(args[i + 1])
                i += 2
            elif not args[i].startswith("--"):
                dirs.append(args[i])
                i += 1
            else:
                i += 1

        # Validate dirs
        valid_dirs = []
        for d in dirs:
            if os.path.isdir(d):
                valid_dirs.append(d)
            else:
                self.console.print(f"[yellow]Warning:[/] directory not found: {d}")

        if not valid_dirs:
            self.console.print("[red]No valid ESM-2 result directories found.[/]")
            return

        from zhuangzhou.tools import load_esm2_results

        mutations = load_esm2_results(valid_dirs)

        self.console.print(
            Panel(
                f"Loaded [bright_cyan]{len(mutations)}[/] mutations from "
                f"[bright_yellow]{len(valid_dirs)}[/] model directories",
                title="ESM-2 Scan Results",
                border_style="bright_cyan",
            )
        )

        if not mutations:
            self.console.print("[dim]No mutations found.[/]")
            return

        table = Table(border_style="cyan")
        table.add_column("#", style="dim", width=4)
        table.add_column("Mutation", style="bright_white")
        table.add_column("Consensus", style="bright_yellow", justify="right")
        table.add_column("Avg dLL", justify="right")
        table.add_column("Max dLL", justify="right")
        table.add_column("Models", justify="center")

        for i, m in enumerate(mutations[:top_n], 1):
            table.add_row(
                str(i),
                m["mutation"],
                f"{m['consensus_score']:.3f}",
                f"{m['avg_delta_ll']:.3f}",
                f"{m['delta_ll']:.3f}",
                str(m["n_models"]),
            )

        self.console.print(table)
        if len(mutations) > top_n:
            self.console.print(
                f"[dim]  ... and {len(mutations) - top_n} more. Use --top N to see more.[/]"
            )
        self.console.print()
