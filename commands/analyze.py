"""Analyze a single mutation with chemical intuition."""

from rich.panel import Panel
from rich.table import Table

from .base import BaseCommand


class AnalyzeCommand(BaseCommand):
    name = "analyze"
    description = "Analyze a mutation with chemical intuition engine"
    usage = "/analyze V21L [--role active_site] [--conserved] [--ss helix]"
    aliases = ["chem", "chemistry"]

    def execute(self, args: list):
        if not args:
            self.console.print(
                "[red]Usage:[/] /analyze <mutation> [--role active_site|buried_core|surface] [--conserved]"
            )
            return

        mut_str = args[0].upper()
        if len(mut_str) < 3 or not mut_str[0].isalpha() or not mut_str[-1].isalpha():
            self.console.print(
                f"[red]Invalid mutation format:[/] {mut_str}  (expected e.g. V21L)"
            )
            return

        params = self._parse_flags(args[1:])
        wt_aa = mut_str[0]
        mut_aa = mut_str[-1]
        position = int(mut_str[1:-1])

        role = params.get("role", "unknown")
        is_conserved = "conserved" in params
        ss = params.get("ss", params.get("secondary", "loop"))
        distance = float(params["distance"]) if "distance" in params else None

        context = {
            "role": role,
            "secondary_structure": ss,
            "interactions": [],
            "neighbors": [],
            "distance_to_site": distance,
            "sasa": None,
            "in_conserved_motif": is_conserved,
        }

        from zhuangzhou.chemical_intuition import analyze_mutation_chemistry

        result = analyze_mutation_chemistry(wt_aa, mut_aa, context)

        score = result["chemical_score"]
        if score > 0.5:
            score_color = "green"
            verdict = "FAVORABLE"
        elif score > -0.5:
            score_color = "yellow"
            verdict = "NEUTRAL"
        else:
            score_color = "red"
            verdict = "UNFAVORABLE"

        # Header
        self.console.print()
        self.console.print(
            Panel(
                f"[bold bright_white]{wt_aa}{position}{mut_aa}[/]  "
                f"[dim]role={role}  ss={ss}  conserved={'yes' if is_conserved else 'no'}[/]\n"
                f"\n"
                f"Score: [{score_color}]{score:+.2f}[/]  |  "
                f"Verdict: [{score_color}]{verdict}[/]  |  "
                f"Confidence: [bright_yellow]{result.get('confidence', 'medium')}[/]  |  "
                f"Recommendation: [bright_white]{result.get('recommendation', '')}[/]",
                title="Chemical Analysis",
                border_style="bright_cyan",
            )
        )

        # Score breakdown
        if result.get("score_breakdown"):
            bd_table = Table(title="Score Breakdown", border_style="cyan")
            bd_table.add_column("Component", style="bright_white")
            bd_table.add_column("Score", justify="right")
            for comp, val in result["score_breakdown"].items():
                color = "green" if val > 0 else ("red" if val < 0 else "dim")
                bd_table.add_row(comp, f"[{color}]{val:+.2f}[/]")
            self.console.print(bd_table)

        # Reasoning
        if result.get("reasoning"):
            self.console.print("\n[bold bright_cyan]Reasoning:[/]")
            for r in result["reasoning"]:
                self.console.print(f"  [bright_white]-[/] {r}")

        # Warnings
        if result.get("warnings"):
            self.console.print("\n[bold yellow]Warnings:[/]")
            for w in result["warnings"]:
                self.console.print(f"  [yellow]![/] {w}")

        # Opportunities
        if result.get("opportunities"):
            self.console.print("\n[bold green]Opportunities:[/]")
            for o in result["opportunities"]:
                self.console.print(f"  [green]+[/] {o}")

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
