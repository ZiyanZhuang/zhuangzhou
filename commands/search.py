"""Knowledge graph search via BeTaThink."""

import os
import sys
from rich.panel import Panel
from rich.table import Table

from .base import BaseCommand


class SearchCommand(BaseCommand):
    name = "search"
    description = "Search BeTaThink knowledge graph (enzyme, mutation, pathway)"
    usage = "/search <query> [--type enzyme|mutation|pathway] [--limit 10]"
    aliases = ["kg", "knowledge", "find"]

    def execute(self, args: list):
        if not args:
            self.console.print(
                "[red]Usage:[/] /search <query> [--type enzyme|mutation|pathway]"
            )
            return

        query_parts = []
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
                query_parts.append(args[i])
                i += 1

        query = " ".join(query_parts)
        search_type = params.get("type", "all")
        limit = int(params.get("limit", "10"))

        self.console.print(
            Panel(
                f"Query: [bright_white]{query}[/]\n"
                f"Type: [bright_cyan]{search_type}[/]  Limit: [bright_yellow]{limit}[/]",
                title="Knowledge Graph Search",
                border_style="bright_cyan",
            )
        )

        # Try BeTaThink API
        try:
            self._search_api(query, search_type, limit)
        except Exception as e:
            self.console.print(f"[yellow]BeTaThink API unavailable:[/] {e}")
            self.console.print("[dim]Falling back to local literature RAG...[/]")
            self._search_local(query, limit)

    def _search_api(self, query: str, search_type: str, limit: int):
        import urllib.request
        import json

        base = self.config.get("betathink_api", "http://localhost:8000")
        url = f"{base}/api/entity/search?q={query}&limit={limit}"

        req = urllib.request.Request(url, headers={"Accept": "application/json"})
        with urllib.request.urlopen(req, timeout=5) as resp:
            data = json.loads(resp.read())

        entities = data.get("entities", data.get("results", []))
        if not entities:
            self.console.print("[dim]No results found.[/]")
            return

        table = Table(title=f"Results for '{query}'", border_style="bright_cyan")
        table.add_column("#", style="dim", width=4)
        table.add_column("Name", style="bright_white")
        table.add_column("Type", style="bright_cyan")
        table.add_column("Description", style="dim", max_width=50)

        for i, ent in enumerate(entities[:limit], 1):
            table.add_row(
                str(i),
                ent.get("name", ent.get("label", "?")),
                ent.get("type", ent.get("entity_type", "?")),
                str(ent.get("description", ent.get("properties", "")))[:50],
            )

        self.console.print(table)

    def _search_local(self, query: str, limit: int):
        try:
            betathink_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            if betathink_root not in sys.path:
                sys.path.insert(0, betathink_root)

            from zhuangzhou.literature_rag import LiteratureRAG

            rag = LiteratureRAG()
            results = rag.search(query, top_k=limit)

            if not results:
                self.console.print("[dim]No local literature results found.[/]")
                return

            for i, r in enumerate(results[:limit], 1):
                self.console.print(
                    f"  [bright_cyan]{i}.[/] [bright_white]{r.get('title', '?')}[/]"
                )
                if r.get("snippet"):
                    self.console.print(f"     [dim]{r['snippet'][:100]}[/]")

        except Exception as e:
            self.console.print(f"[dim]Local search also unavailable: {e}[/]")
