"""Docking commands."""

import os
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, TimeElapsedColumn
from rich.table import Table
from rich.prompt import Prompt

from .base import BaseCommand


class DockCommand(BaseCommand):
    name = "dock"
    description = "Run molecular docking (Vina/SmartDocking)"
    usage = "/dock <protein.pdb> <ligand.sdf> [--center x,y,z] [--exhaustiveness 32]"
    aliases = ["vina", "docking"]

    def execute(self, args: list):
        if len(args) < 2:
            self.console.print(
                "[red]Usage:[/] /dock <protein.pdb> <ligand.sdf> "
                "[--center x,y,z] [--exhaustiveness 32]"
            )
            return

        protein = args[0]
        ligand = args[1]
        params = self._parse_flags(args[2:])

        if not os.path.exists(protein):
            self.console.print(f"[red]Protein file not found:[/] {protein}")
            return
        if not os.path.exists(ligand):
            self.console.print(f"[red]Ligand file not found:[/] {ligand}")
            return

        center = None
        if "center" in params:
            parts = params["center"].split(",")
            center = tuple(float(x) for x in parts)
        else:
            center_str = Prompt.ask("[bright_cyan]Grid center[/] (x,y,z)")
            parts = center_str.split(",")
            center = tuple(float(x) for x in parts)

        exhaustiveness = int(params.get("exhaustiveness", params.get("ex", "32")))
        box_str = params.get("box", "20,20,20")
        box_size = tuple(float(x) for x in box_str.split(","))

        self.console.print(
            Panel(
                f"Protein: [bright_white]{protein}[/]\n"
                f"Ligand:  [bright_white]{ligand}[/]\n"
                f"Center:  [bright_cyan]{center}[/]  Box: [bright_yellow]{box_size}[/]\n"
                f"Exhaustiveness: [bright_yellow]{exhaustiveness}[/]",
                title="Vina Docking",
                border_style="bright_cyan",
            )
        )

        from zhuangzhou.tools import run_vina_docking

        with Progress(
            SpinnerColumn(style="bright_cyan"),
            TextColumn("[progress.description]{task.description}"),
            TimeElapsedColumn(),
            console=self.console,
        ) as progress:
            task = progress.add_task("Docking in progress...", total=None)
            result = run_vina_docking(
                protein, ligand, center, box_size, exhaustiveness, self.config
            )
            progress.update(task, description="[green]Complete[/]")

        if result.get("error"):
            self.console.print(f"[red]Docking failed:[/] {result['error']}")
            return

        # Display results
        table = Table(title="Docking Results", border_style="bright_cyan")
        table.add_column("Metric", style="bright_white")
        table.add_column("Value", style="bright_yellow")

        for key, val in result.items():
            if key not in ("error",):
                table.add_row(str(key), str(val))

        self.console.print(table)
        self.console.print()

    def _parse_flags(self, args):
        params = {}
        i = 0
        while i < len(args):
            if args[i].startswith("--"):
                key = args[i][2:]
                if i + 1 < len(args) and not args[i + 1].startswith("--"):
                    params[key] = args[i + 1]
                    i += 2
                else:
                    params[key] = True
                    i += 1
            else:
                i += 1
        return params
