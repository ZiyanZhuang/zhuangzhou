"""Run the full Zhuangzhou pipeline."""

import os
import time
from rich.panel import Panel
from rich.progress import (
    Progress,
    SpinnerColumn,
    TextColumn,
    BarColumn,
    TimeElapsedColumn,
)
from rich.table import Table
from rich.prompt import Prompt

from .base import BaseCommand


class RunCommand(BaseCommand):
    name = "run"
    description = "Run the full 5-phase enzyme engineering pipeline"
    usage = (
        "/run <pdb_path> --site 15,21,23 [--esm2 dir1,dir2] [--enzyme oxidoreductase]"
    )
    aliases = ["pipeline", "start"]

    def execute(self, args: list):
        if not args:
            return self._interactive_run()

        pdb_path = args[0]
        params = self._parse_args(args[1:])

        if not os.path.exists(pdb_path):
            self.console.print(f"[red]PDB file not found:[/] {pdb_path}")
            return

        binding_site = set()
        if "site" in params:
            binding_site = set(int(x) for x in params["site"].split(","))
        elif "binding-site" in params:
            binding_site = set(int(x) for x in params["binding-site"].split(","))

        if not binding_site:
            site_str = Prompt.ask(
                "[bright_cyan]Binding site residues[/] (comma-separated)"
            )
            binding_site = set(int(x) for x in site_str.split(","))

        esm2_dirs = []
        if "esm2" in params:
            esm2_dirs = params["esm2"].split(",")
        elif "esm2-dirs" in params:
            esm2_dirs = params["esm2-dirs"].split(",")

        conserved = set()
        if "conserved" in params:
            conserved = set(int(x) for x in params["conserved"].split(","))

        enzyme_class = params.get("enzyme", params.get("enzyme-class", ""))
        ligand = params.get("ligand", "")
        name = params.get("name", os.path.splitext(os.path.basename(pdb_path))[0])
        output = params.get("output", params.get("out", f"results/{name}_zhuangzhou"))

        self._run_pipeline(
            pdb_path,
            binding_site,
            esm2_dirs,
            conserved,
            enzyme_class,
            ligand,
            name,
            output,
        )

    def _interactive_run(self):
        self.console.print(
            Panel(
                "[bright_cyan]Interactive Pipeline Setup[/]\n"
                "Fill in the parameters below to run the 5-phase pipeline.",
                title="Pipeline Configuration",
                border_style="bright_cyan",
            )
        )

        pdb_path = Prompt.ask("[bright_cyan]PDB file path[/]")
        if not os.path.exists(pdb_path):
            self.console.print(f"[red]File not found:[/] {pdb_path}")
            return

        site_str = Prompt.ask("[bright_cyan]Binding site residues[/] (comma-separated)")
        binding_site = set(int(x) for x in site_str.split(","))

        esm2_str = Prompt.ask(
            "[bright_cyan]ESM-2 result dirs[/] (comma-separated, or empty)", default=""
        )
        esm2_dirs = (
            [d.strip() for d in esm2_str.split(",") if d.strip()] if esm2_str else []
        )

        conserved_str = Prompt.ask(
            "[bright_cyan]Conserved positions[/] (or empty)", default=""
        )
        conserved = (
            set(int(x) for x in conserved_str.split(",") if x.strip())
            if conserved_str
            else set()
        )

        enzyme_class = Prompt.ask("[bright_cyan]Enzyme class[/]", default="")
        ligand = Prompt.ask("[bright_cyan]Ligand name[/]", default="")
        name = Prompt.ask(
            "[bright_cyan]Protein name[/]",
            default=os.path.splitext(os.path.basename(pdb_path))[0],
        )
        output = Prompt.ask(
            "[bright_cyan]Output directory[/]", default=f"results/{name}_zhuangzhou"
        )

        self._run_pipeline(
            pdb_path,
            binding_site,
            esm2_dirs,
            conserved,
            enzyme_class,
            ligand,
            name,
            output,
        )

    def _run_pipeline(
        self,
        pdb_path,
        binding_site,
        esm2_dirs,
        conserved,
        enzyme_class,
        ligand,
        name,
        output,
    ):
        self.console.print()
        phases = [
            ("Phase 1", "PLM Consensus Analysis", "Loading ESM-2 results..."),
            ("Phase 2", "Structural Context", "Building 3D context..."),
            ("Phase 3", "Chemical Intuition", "Analyzing mutations..."),
            ("Phase 4", "Multi-site Combination", "Generating combinations..."),
            ("Phase 5", "Report Generation", "Writing report..."),
        ]

        self.console.print(
            Panel(
                f"[bright_white]{name}[/] | [dim]{pdb_path}[/]\n"
                f"Sites: [bright_cyan]{sorted(binding_site)}[/] | "
                f"ESM-2 dirs: [bright_yellow]{len(esm2_dirs)}[/] | "
                f"Enzyme: [bright_cyan]{enzyme_class or 'auto'}[/]",
                title="Starting Pipeline",
                border_style="bright_cyan",
            )
        )

        from zhuangzhou.core import ZhuangzhouFramework

        fw = ZhuangzhouFramework(self.config)

        t0 = time.time()
        with Progress(
            SpinnerColumn(style="bright_cyan"),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(bar_width=30, style="cyan", complete_style="bright_yellow"),
            TimeElapsedColumn(),
            console=self.console,
        ) as progress:
            task = progress.add_task("Running pipeline...", total=5)

            def phase_callback(phase_num, phase_name, detail=""):
                progress.update(
                    task,
                    completed=phase_num,
                    description=f"[bright_white]{phase_name}[/] {detail}",
                )

            results = fw.run_enzyme_engineering(
                pdb_path=pdb_path,
                binding_site=binding_site,
                esm2_dirs=esm2_dirs,
                enzyme_class=enzyme_class,
                ligand=ligand,
                protein_name=name,
                conserved_positions=conserved,
                output_dir=output,
            )
            progress.update(task, completed=5, description="[green]Complete[/]")

        elapsed = time.time() - t0

        # Results summary
        table = Table(
            title="Pipeline Results",
            border_style="bright_cyan",
            title_style="bold bright_cyan",
        )
        table.add_column("Metric", style="bright_white")
        table.add_column("Value", style="bright_yellow")

        n_mutations = len(results.get("mutations", []))
        n_combos = len(results.get("combinations", []))
        table.add_row("Mutations analyzed", str(n_mutations))
        table.add_row("Combinations generated", str(n_combos))
        table.add_row("Time elapsed", f"{elapsed:.1f}s")
        table.add_row("Report", results.get("report_path", "N/A"))
        table.add_row("JSON", results.get("json_path", "N/A"))

        self.console.print(table)
        self.console.print()

        # Top mutations
        mutations = results.get("mutations", [])
        if mutations:
            top = sorted(mutations, key=lambda m: -m.get("final_score", 0))[:10]
            top_table = Table(title="Top 10 Mutations", border_style="cyan")
            top_table.add_column("Rank", style="dim", width=4)
            top_table.add_column("Mutation", style="bright_white")
            top_table.add_column("Score", style="bright_yellow", justify="right")
            top_table.add_column("Chemical", style="cyan", justify="right")
            top_table.add_column("Role", style="dim")

            for i, m in enumerate(top, 1):
                chem = m.get("chemical_score", 0)
                chem_str = (
                    f"[green]{chem:+.2f}[/]" if chem > 0 else f"[red]{chem:+.2f}[/]"
                )
                top_table.add_row(
                    str(i),
                    m.get("mutation", "?"),
                    f"{m.get('final_score', 0):.3f}",
                    chem_str,
                    m.get("role", ""),
                )

            self.console.print(top_table)

    def _parse_args(self, args):
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
