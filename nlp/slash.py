"""
Slash Command System -- Centralized command registry for Zhuangzhou.
Inspired by industrial-grade CLIs like LLM agent runtime.
"""

import os
import sys
import json
import logging
import re
from typing import List, Dict, Any, Optional, Callable

logger = logging.getLogger(__name__)


class SlashCommand:
    """Base class for all slash commands."""

    trigger: str = ""
    description: str = ""
    aliases: List[str] = []

    def execute(self, context: Any, args: List[str]):
        """Execute the command logic.
        'context' is usually the ZhuangzhouAgent or App instance.
        """
        raise NotImplementedError


class ModelCommand(SlashCommand):
    trigger = "/model"
    description = "List or switch LLM models (local/cloud)"

    def execute(self, context, args: List[str]):
        agent = SlashCommandManager._unpack_agent(context)
        from zhuangzhou.nlp.backends import list_available_models, get_backend
        from zhuangzhou.ui.core import ZZInterface, console, ZZTheme

        if not args:
            # Interactive selection menu
            models = list_available_models()
            if not models:
                console.print(
                    "[bold red]\u2718 Error:[/] No models available. Is Ollama running?"
                )
                return

            current_full = (
                f"{agent.backend_name}:{agent.backend.model}"
                if hasattr(agent.backend, "model")
                else agent.backend_name
            )

            # Format options for the menu: (value, label)
            options = []
            for m in models:
                is_curr = m["name"] == current_full or m["name"] == agent.backend_name
                label = f"{m['name']} ({m['type']})"
                if is_curr:
                    label += " [CURRENT]"
                options.append((m["name"], label))

            selected = ZZInterface.select_one(
                "Select Model Engine", options, default=current_full
            )

            if not selected:
                return  # User cancelled

            target = selected
        else:
            # Switch via direct argument
            target = args[0].strip()

        # Perform the actual switch
        try:
            new_backend = get_backend(target)
            if not new_backend.is_available():
                console.print(
                    f"[bold red]\u2718 Error:[/] Model {target} is offline or not found."
                )
                return

            # Hot-swap components
            agent.backend = new_backend
            agent.executor.backend = new_backend
            # The setter we just added in agent.py will handle this silently
            agent.backend_name = new_backend.name

            console.print(
                f"[bold {ZZTheme.GREEN}]\u2714[/] Hot-swapped to [bold]{target}[/]. Context preserved."
            )
        except Exception as e:
            console.print(f"[bold red]\u2718 Error switching model:[/] {e}")


class SkillsCommand(SlashCommand):
    trigger = "/skills"
    description = "List all available computational skills"

    def execute(self, context, args: List[str]):
        agent = SlashCommandManager._unpack_agent(context)
        from zhuangzhou.ui.core import ZZInterface, console, ZZTheme

        # Use the existing table formatter from repl.py logic
        # For Phase 1, we re-use the registry's list
        registry = agent.registry
        categories = {}
        for s in registry.list_all():
            categories.setdefault(s.category, []).append(s)

        table = ZZInterface.create_table(
            f"Zhuangzhou Computational Skills ({sum(len(v) for v in categories.values())})",
            ["Skill", "Category", "Risk", "Description"],
        )

        for cat in sorted(categories.keys()):
            for s in sorted(categories[cat], key=lambda x: x.name):
                risk = getattr(s, "risk_level", "low") or "low"
                risk_display = {
                    "low": "[dim]low[/dim]",
                    "medium": f"[{ZZTheme.GOLD}]med[/]",
                    "high": f"[{ZZTheme.LAVA}]HIGH[/]",
                }.get(risk, "[dim]low[/dim]")
                table.add_row(
                    f"[bold {ZZTheme.CYAN}]{s.name}[/]",
                    f"[dim]{s.category}[/]",
                    risk_display,
                    (s.description or "")[:60],
                )

        console.print(table)


class MemCommand(SlashCommand):
    trigger = "/mem"
    description = "Inspect agent episodic and procedural memory"

    def execute(self, context, args: List[str]):
        agent = SlashCommandManager._unpack_agent(context)
        from zhuangzhou.ui.core import console, ZZTheme, format_tokens
        from rich.panel import Panel
        from rich.table import Table
        from zhuangzhou.memory.kg_engine import LocalKGEngine

        # Stats from the real executor
        exec = agent.executor
        ui_status = agent.get_ui_status()

        if args:
            sub = args[0].lower()
            if sub == "context":
                full = len(args) > 1 and args[1].lower() == "full"
                if full:
                    console.print(
                        Panel(
                            ui_status["system_prompt"] or "(empty system prompt)",
                            title=f"[bold {ZZTheme.CYAN}]CURRENT SYSTEM PROMPT[/]",
                            border_style=ZZTheme.MID,
                        )
                    )
                    for block in ui_status.get("prompt_blocks", []):
                        header = f"{block['title']}  [{format_tokens(block['estimated_tokens'])}]"
                        console.print(
                            Panel(
                                block["content"] or "(empty)",
                                title=f"[bold {ZZTheme.GOLD}]{header}[/]",
                                border_style=ZZTheme.MID,
                            )
                        )
                    return

                table = Table(title="Prompt Blocks", border_style=ZZTheme.MID)
                table.add_column("Block", style=ZZTheme.CYAN)
                table.add_column("Tokens", style=ZZTheme.GOLD)
                table.add_column("Preview", style="white")
                for block in ui_status.get("prompt_blocks", []):
                    preview = re.sub(r"\s+", " ", block.get("content", "")).strip()
                    if len(preview) > 120:
                        preview = preview[:117] + "..."
                    table.add_row(
                        block.get("title", block.get("name", "block")),
                        format_tokens(block.get("estimated_tokens", 0)),
                        preview or "(empty)",
                    )
                console.print(table)
                console.print(
                    "[dim]Use /mem context full to expand the complete system prompt.[/]"
                )
                return
            if sub == "kg":
                query = " ".join(args[1:]).strip()
                stats = LocalKGEngine.stats()
                stats_table = Table(box=None, show_header=False)
                stats_table.add_column("Key", style=ZZTheme.CYAN)
                stats_table.add_column("Value", style="white")
                stats_table.add_row("Path", stats["path"])
                stats_table.add_row("Records", str(stats["records"]))
                stats_table.add_row("Triples", str(stats["triples"]))
                stats_table.add_row("Entities", str(stats["entities"]))
                stats_table.add_row("Relations", str(stats["relations"]))
                console.print(
                    Panel(
                        stats_table,
                        title=f"[bold {ZZTheme.CYAN}]LOCAL GRAPH OVERLAY[/]",
                        border_style=ZZTheme.MID,
                    )
                )
                triples = (
                    LocalKGEngine.query(query, limit=12)
                    if query
                    else LocalKGEngine.recent_triples(limit=12)
                )
                triple_table = Table(
                    title="Overlay Hits" if query else "Recent Overlay Triples",
                    border_style=ZZTheme.MID,
                )
                triple_table.add_column("Head", style=ZZTheme.CYAN)
                triple_table.add_column("Relation", style=ZZTheme.GOLD)
                triple_table.add_column("Tail", style="white")
                triple_table.add_column("Hits", style=ZZTheme.GOLD)
                triple_table.add_column("Time", style="dim")
                for triple in triples:
                    triple_table.add_row(
                        triple.get("head", ""),
                        triple.get("relation", ""),
                        triple.get("tail", ""),
                        str(triple.get("access_count", 0)),
                        triple.get("timestamp", "")[:19],
                    )
                if triples:
                    console.print(triple_table)
                else:
                    console.print("[dim]No local overlay triples found.[/]")
                return

        stats_table = Table(box=None, show_header=False)
        stats_table.add_column("Key", style=ZZTheme.CYAN)
        stats_table.add_column("Value", style="white")

        stats_table.add_row(
            "Backend",
            f"{agent.backend_name} ({'Native Tools' if getattr(agent.backend, 'supports_tools', True) else 'Reasoning Mode'})",
        )
        stats_table.add_row("Messages", str(len(exec.messages)))
        stats_table.add_row("Tools Loaded", f"{len(exec.tools)} schemas")
        stats_table.add_row("Categories", ", ".join(exec._loaded_categories))
        stats_table.add_row("Session ID", exec._session_id)
        stats_table.add_row(
            "Context",
            f"{format_tokens(ui_status['estimated_tokens'])}/{format_tokens(ui_status['context_limit'])} ({int(ui_status['usage_ratio'] * 100)}%)",
        )
        stats_table.add_row(
            "Memory Layers", ", ".join(ui_status["layers_active"]) or "none"
        )
        stats_table.add_row(
            "Session Context", ui_status.get("session_summary") or "No active context"
        )
        stats_table.add_row(
            "Reset Hint", ui_status.get("reset_reason") or "No reset needed"
        )
        stats_table.add_row("Compactions", str(ui_status.get("compaction_count", 0)))
        stats_table.add_row(
            "Last Compact",
            ui_status.get("last_compaction_reason") or "No automatic compaction yet",
        )

        console.print(
            Panel(
                stats_table,
                title=f"[bold {ZZTheme.CYAN}]AGENT RUNTIME MEMORY[/]",
                border_style=ZZTheme.MID,
            )
        )

        # Show pattern memory if available
        try:
            from zhuangzhou.memory.knowledge import KnowledgeStore

            ks = KnowledgeStore()
            recent_errors = len(exec.execution_log)
            console.print(
                f"\n[dim]Pattern Memory:[/] Active ({recent_errors} events this session)"
            )
        except Exception:
            pass


class GraphRAGCommand(SlashCommand):
    trigger = "/graphrag"
    description = "GraphRAG overlay tools (stats, rebuild-overlay export)"

    def execute(self, context, args: List[str]):
        from zhuangzhou.ui.core import console, ZZTheme
        from rich.panel import Panel
        from rich.table import Table
        from zhuangzhou.memory.kg_engine import LocalKGEngine

        action = args[0].lower() if args else "stats"
        if action == "rebuild-overlay":
            export_dir = os.path.join(
                os.getcwd(), "storage", "graphrag", "_overlay_exports"
            )
            result = LocalKGEngine.export_overlay_for_graphrag(export_dir)
            table = Table(box=None, show_header=False)
            table.add_column("Key", style=ZZTheme.CYAN)
            table.add_column("Value", style="white")
            table.add_row("Status", result.get("status", "unknown"))
            table.add_row("Export", result.get("export_jsonl", ""))
            table.add_row("Manifest", result.get("manifest", ""))
            table.add_row("Triples", str(result.get("triples", 0)))
            table.add_row("Entities", str(result.get("entities", 0)))
            console.print(
                Panel(
                    table,
                    title=f"[bold {ZZTheme.CYAN}]GRAPHRAG OVERLAY EXPORT[/]",
                    border_style=ZZTheme.MID,
                )
            )
            return

        if action == "stats":
            stats = LocalKGEngine.stats()
            table = Table(box=None, show_header=False)
            table.add_column("Key", style=ZZTheme.CYAN)
            table.add_column("Value", style="white")
            for key in ("path", "records", "triples", "entities", "relations"):
                table.add_row(key.title(), str(stats.get(key, "")))
            console.print(
                Panel(
                    table,
                    title=f"[bold {ZZTheme.CYAN}]GRAPHRAG OVERLAY STATUS[/]",
                    border_style=ZZTheme.MID,
                )
            )
            console.print(
                "[dim]Use /graphrag rebuild-overlay to export local_graph.jsonl for monthly parquet rebuilds.[/]"
            )
            return

        console.print("[dim]Usage: /graphrag stats | /graphrag rebuild-overlay[/]")


class HelpCommand(SlashCommand):
    trigger = "/help"
    description = "Show this help information"

    def execute(self, context, args: List[str]):
        from zhuangzhou.ui.core import console, ZZTheme
        from rich.table import Table

        table = Table(
            title="Zhuangzhou Slash Commands", border_style=ZZTheme.MID, box=None
        )
        table.add_column("Command", style=ZZTheme.GOLD)
        table.add_column("Description", style="white")

        # Global manager instance is accessible 鈥?deduplicate aliased commands by identity
        seen_ids = set()
        for cmd in sorted(manager.commands.values(), key=lambda x: x.trigger):
            if id(cmd) not in seen_ids:
                seen_ids.add(id(cmd))
                table.add_row(cmd.trigger, cmd.description)

        console.print(table)
        console.print(
            f"\n[dim]Tip: Commands can be used at any time. Natural language is sent to the Agent.[/]\n"
        )


class ResetCommand(SlashCommand):
    trigger = "/reset"
    description = "Clear conversation history"

    def execute(self, context, args: List[str]):
        agent = SlashCommandManager._unpack_agent(context)
        from zhuangzhou.ui.core import console

        agent.reset()
        console.print("[dim][Conversation history cleared][/dim]")


class ResumeCommand(SlashCommand):
    trigger = "/resume"
    description = "Resume saved SessionContext state explicitly"

    def execute(self, context, args: List[str]):
        agent = SlashCommandManager._unpack_agent(context)
        from zhuangzhou.ui.core import console, ZZTheme
        from rich.table import Table

        if args and args[0].lower() == "list":
            sessions = agent.list_resumable_sessions(limit=10)
            if not sessions:
                console.print("[dim]No resumable sessions found.[/]")
                return
            table = Table(title="Resumable Sessions", border_style=ZZTheme.MID)
            table.add_column("Session ID", style=ZZTheme.CYAN)
            table.add_column("Protein", style="white")
            table.add_column("PDB", style="white")
            table.add_column("Commands", style=ZZTheme.GOLD)
            for item in sessions:
                table.add_row(
                    item.get("session_id", ""),
                    item.get("protein", "") or "-",
                    item.get("active_pdb", "") or "-",
                    str(item.get("commands", 0)),
                )
            console.print(table)
            console.print(
                "[dim]Use /resume <session_id> to restore one of these session states.[/]"
            )
            return

        if args and args[0].lower() == "current":
            ok = agent.resume_current_context()
        else:
            session_id = args[0] if args else ""
            ok = agent.resume_session_context(session_id=session_id)
        if ok:
            console.print(f"[bold {ZZTheme.GREEN}]OK[/] Session context restored.")
        else:
            console.print("[dim]No matching saved session context found.[/]")


class VerboseCommand(SlashCommand):
    trigger = "/verbose"
    description = "Toggle verbose mode (show tool calls)"

    def execute(self, context, args: List[str]):
        agent = SlashCommandManager._unpack_agent(context)
        from zhuangzhou.ui.core import console, ZZTheme

        agent.verbose = not agent.verbose
        state = f"[bold {ZZTheme.GREEN}]ON[/]" if agent.verbose else "[dim]OFF[/dim]"
        console.print(f"  Verbose mode: {state}")


class InitCommand(SlashCommand):
    trigger = "/init"
    description = "Initialize ZHUANGZHOU.md and scan local environment"

    def execute(self, context, args: List[str]):
        from zhuangzhou.nlp.repl import handle_init_command

        handle_init_command()


class BioDesignCommand(SlashCommand):
    trigger = "/biodesign"
    description = "Launch Bio-design Mode for automated metabolic engineering design workflows"

    def execute(self, context, args: List[str]):
        agent = SlashCommandManager._unpack_agent(context)
        from zhuangzhou.ui.core import console, ZZTheme

        sub = args[0].lower() if args else "run"

        if sub == "help":
            from rich.table import Table
            from rich.panel import Panel
            table = Table(box=None, show_header=False)
            table.add_column("", style=ZZTheme.CYAN)
            table.add_column("", style="white")
            table.add_row("/biodesign", "Run the interactive design workflow from a natural-language goal")
            table.add_row("/biodesign help", "Show this help")
            table.add_row("/biodesign skills", "List metabolism-engineering related skills")
            console.print(Panel(
                table,
                title=f"[bold {ZZTheme.CYAN}]Bio-design Mode Commands[/]",
                border_style=ZZTheme.MID,
            ))
            console.print(
                "[dim]Usage: enter a design goal directly, e.g. 'Design an E. coli pathway for L-alanine production'[/]"
            )
            return

        if sub == "skills":
            registry = agent.registry
            from rich.table import Table
            table = Table(title="MetaKG + SynBio Skills", border_style=ZZTheme.MID)
            table.add_column("Skill", style=f"bold {ZZTheme.CYAN}")
            table.add_column("Category", style="dim")
            table.add_column("Description", style="white")
            for s in registry.list_all():
                if s.category in ("metakg", "synbio"):
                    table.add_row(s.name, s.category, (s.description or "")[:65])
            console.print(table)
            return

        # Run Bio-design Mode on provided query or prompt for one
        query = " ".join(args) if args else ""
        if not query or sub == "run":
            console.print("[dim]Enter a design goal (e.g. 'produce L-alanine'): [/dim]", end="")
            try:
                query = input().strip()
            except (EOFError, KeyboardInterrupt):
                return
            if not query:
                return

        try:
            from zhuangzhou.nlp.biodesign_mode import BioDesignMode
            bdm = BioDesignMode(backend=agent.backend, registry=agent.registry)
            result = bdm.run(query, console=console)
            if not result.skipped and result.design:
                # Optionally continue with agent
                console.print(
                    f"\n  [dim]Bio-design completed. Send the result to the agent for follow-up analysis? [Y/n]: [/dim]",
                    end="",
                )
                try:
                    choice = input().strip().lower()
                except (EOFError, KeyboardInterrupt):
                    choice = "n"
                if choice not in ("n", "no"):
                    agent.chat(result.agent_message)
        except Exception as e:
            console.print(f"[bold red]Bio-design error:[/] {e}")
            import traceback
            if getattr(agent, "verbose", False):
                traceback.print_exc()


class PlanCommand(SlashCommand):
    trigger = "/plan"
    description = "Toggle Plan Mode (intent alignment tree, default: ON)"

    def execute(self, context, args: List[str]):
        plan_mode = SlashCommandManager._unpack_plan_mode(context)
        from zhuangzhou.ui.core import console, ZZTheme

        if plan_mode is None:
            console.print("[dim]Plan Mode is not available in this session.[/]")
            return

        if args and args[0].lower() in ("on", "off"):
            plan_mode.enabled = args[0].lower() == "on"
        else:
            plan_mode.toggle()

        state = f"[bold {ZZTheme.GREEN}]ON[/]" if plan_mode.enabled else "[dim]OFF[/dim]"
        hint = (
            "[dim]  (ambiguous intent will trigger interactive parameter alignment)[/dim]"
            if plan_mode.enabled
            else "[dim]  (queries are sent directly to the agent)[/dim]"
        )
        console.print(f"  Plan Mode: {state}{hint}")


class AgentsCommand(SlashCommand):
    trigger = "/agents"
    description = "Multi-agent system status, routing, and specialist info"

    def execute(self, context, args: List[str]):
        from zhuangzhou.ui.core import console, ZZTheme
        from rich.table import Table
        from rich.panel import Panel

        sub = args[0].lower() if args else "status"

        if sub == "list":
            from zhuangzhou.agents.specialists import SPECIALIST_REGISTRY
            table = Table(title="Domain Specialists", border_style=ZZTheme.MID)
            table.add_column("Name", style=f"bold {ZZTheme.CYAN}")
            table.add_column("Description", style="white")
            table.add_column("Skill Categories", style="dim")
            for name, cls in SPECIALIST_REGISTRY.items():
                table.add_row(
                    name,
                    cls.description,
                    ", ".join(cls.skill_categories),
                )
            console.print(table)
            console.print(
                "[dim]Specialists are lazy-initialized on first use. "
                "Use /agents status to see which are loaded.[/]"
            )
            return

        if sub == "routing":
            from zhuangzhou.agents.orchestrator import _ROUTING_RULES
            table = Table(title="Keyword Routing Rules", border_style=ZZTheme.MID)
            table.add_column("Domain", style=f"bold {ZZTheme.CYAN}")
            table.add_column("Sample Keywords", style="dim")
            for keywords, domain in _ROUTING_RULES:
                table.add_row(domain, ", ".join(keywords[:6]) + " ...")
            console.print(table)
            return

        # Default: status
        try:
            from zhuangzhou.agents import get_orchestrator
            orc = get_orchestrator()
            status = orc.status()
            stats_table = Table(box=None, show_header=False)
            stats_table.add_column("Key", style=ZZTheme.CYAN)
            stats_table.add_column("Value", style="white")
            stats_table.add_row("Session", status["session_id"])
            stats_table.add_row(
                "Specialists Loaded",
                ", ".join(status["specialists_loaded"]) or "none",
            )
            stats_table.add_row(
                "Main Agent", "loaded" if status["main_agent_loaded"] else "not loaded"
            )
            stats_table.add_row(
                "Routing", "specialists" if status["use_specialists"] else "main agent only"
            )
            if status["shared_state"]:
                stats_table.add_row("Shared State", status["shared_state"])
            console.print(
                Panel(
                    stats_table,
                    title=f"[bold {ZZTheme.CYAN}]MULTI-AGENT STATUS[/]",
                    border_style=ZZTheme.MID,
                )
            )
            console.print(
                "[dim]Subcommands: /agents list  /agents routing[/]"
            )
        except Exception as e:
            console.print(f"[dim]Multi-agent system: {e}[/]")
            console.print("[dim]Subcommands: /agents list  /agents routing[/]")


def get_output_dir(config: dict, stem: str = "") -> str:
    """Return the effective output directory for a computation.

    Priority:
    1. ``config['default_output_dir']`` if set and non-empty
    2. Current working directory (``os.getcwd()``)

    If *stem* is provided and the directory does not yet exist it is created.
    """
    base = (config or {}).get("default_output_dir", "").strip()
    if not base:
        base = os.getcwd()
    if stem:
        out = os.path.join(base, stem)
        os.makedirs(out, exist_ok=True)
        return out
    return base


class CompactCommand(SlashCommand):
    trigger = "/compact"
    description = "Compress conversation history: summarize older messages, keep last N (default 5)"

    def execute(self, context, args: List[str]):
        agent = SlashCommandManager._unpack_agent(context)
        from zhuangzhou.ui.core import console, ZZTheme
        from rich.panel import Panel

        # Parse --keep N flag
        keep = 5
        remaining_args = list(args)
        if "--keep" in remaining_args:
            idx = remaining_args.index("--keep")
            try:
                keep = int(remaining_args[idx + 1])
                remaining_args.pop(idx + 1)
                remaining_args.pop(idx)
            except (IndexError, ValueError):
                console.print("[bold red]Usage:[/] /compact [--keep N]")
                return
        keep = max(1, keep)

        exec_ = agent.executor
        messages = exec_.messages
        total_before = len(messages)

        # Need at least keep+1 messages (system prompt + at least one to summarize)
        # messages[0] is typically the system-context injection or first user turn.
        if total_before <= keep + 1:
            console.print(
                f"[dim]Nothing to compact: only {total_before} messages, keeping last {keep}.[/]"
            )
            return

        # Estimate token reduction (rough: 4 chars per token)
        older = messages[:-keep] if keep > 0 else messages[:]
        chars_before = sum(
            len(str(m.get("content", ""))) for m in older
        )
        tokens_before = chars_before // 4

        # Build summary using the executor's battle-tested helper, falling back inline
        summary_text = ""
        try:
            summary_text = exec_._summarize_for_budget(older)
        except Exception:
            pass

        if not summary_text:
            # Inline fallback using backend directly
            serialized = []
            for m in older:
                content = m.get("content", "")
                if not isinstance(content, str):
                    content = json.dumps(content, ensure_ascii=False, default=str)
                serialized.append(f"{m.get('role', 'unknown').upper()}: {content}")
            payload = "\n\n".join(serialized)
            prompt = (
                "Summarize the following conversation history for a bioinformatics assistant. "
                "Preserve all file paths, PDB IDs, mutation names, numeric results, and "
                "decisions. Be concise but complete.\n\n" + payload
            )
            try:
                resp = agent.backend.chat(
                    "You are a concise scientific summarizer.",
                    [{"role": "user", "content": prompt}],
                    [],
                )
                summary_text = resp.text or ""
            except Exception as e:
                console.print(f"[bold red]Compact failed:[/] {e}")
                return

        if not summary_text:
            console.print("[bold red]Compact failed:[/] LLM returned empty summary.")
            return

        # Replace history: keep first message (system injection) + summary block + last N
        first_msg = messages[0] if messages else None
        tail = messages[-keep:]

        new_messages = []
        if first_msg is not None:
            new_messages.append(first_msg)
        new_messages.append(
            {"role": "user", "content": f"[Conversation summary 鈥?/compact]: {summary_text}"}
        )
        new_messages.append(
            {"role": "assistant", "content": "Understood. I have the summary and will continue."}
        )
        new_messages.extend(tail)

        exec_.messages = new_messages

        # Count savings
        chars_after = len(str(summary_text))
        tokens_after = chars_after // 4
        tokens_saved = max(0, tokens_before - tokens_after)
        total_after = len(exec_.messages)

        console.print(
            Panel(
                f"[bold {ZZTheme.GREEN}]OK[/] Compacted {total_before} -> {total_after} messages\n"
                f"  Kept last [bold]{keep}[/] turns + 1 summary block\n"
                f"  Estimated tokens freed: [bold {ZZTheme.GOLD}]~{tokens_saved:,}[/]",
                title=f"[bold {ZZTheme.CYAN}]/compact[/]",
                border_style=ZZTheme.MID,
            )
        )


class WorkdirCommand(SlashCommand):
    trigger = "/workdir"
    description = "Show or set the default output directory (persisted to config.json)"
    aliases = ["/setdir"]

    def execute(self, context, args: List[str]):
        from zhuangzhou.ui.core import console, ZZTheme
        from rich.panel import Panel
        from rich.table import Table

        # Resolve config 鈥?may come from agent context or app context
        raw_ctx = context
        agent = SlashCommandManager._unpack_agent(raw_ctx)

        # Try to get config from app-level or fall back to loading it
        config = None
        if isinstance(raw_ctx, tuple) and len(raw_ctx) >= 3:
            config = raw_ctx[2]
        if config is None:
            # Load from disk
            cfg_path = os.path.expanduser("~/.zhuangzhou/config.json")
            if os.path.exists(cfg_path):
                try:
                    with open(cfg_path, encoding="utf-8") as f:
                        config = json.load(f)
                except Exception:
                    config = {}
            else:
                config = {}

        current_output_dir = config.get("default_output_dir", "").strip()
        current_cwd = os.getcwd()

        if not args:
            # Show current state
            table = Table(box=None, show_header=False)
            table.add_column("Key", style=ZZTheme.CYAN)
            table.add_column("Value", style="white")
            table.add_row("CWD (process)", current_cwd)
            table.add_row(
                "default_output_dir",
                current_output_dir if current_output_dir else "[dim](not set 鈥?uses CWD)[/dim]",
            )
            effective = current_output_dir if current_output_dir else current_cwd
            table.add_row("Effective output dir", effective)
            console.print(
                Panel(
                    table,
                    title=f"[bold {ZZTheme.CYAN}]WORKING DIRECTORY[/]",
                    border_style=ZZTheme.MID,
                )
            )
            console.print(
                "[dim]Usage: /workdir <path>  鈥?change cwd and default_output_dir[/]"
            )
            return

        # Set new directory
        new_path = " ".join(args).strip().strip('"').strip("'")
        new_path = os.path.expandvars(os.path.expanduser(new_path))
        new_path = os.path.abspath(new_path)

        # Validate / create
        if not os.path.exists(new_path):
            try:
                os.makedirs(new_path, exist_ok=True)
                console.print(f"[dim]Created directory: {new_path}[/]")
            except Exception as e:
                console.print(f"[bold red]Cannot create directory:[/] {e}")
                return

        if not os.path.isdir(new_path):
            console.print(f"[bold red]Not a directory:[/] {new_path}")
            return

        # Update process cwd
        try:
            os.chdir(new_path)
        except Exception as e:
            console.print(f"[bold red]Cannot chdir:[/] {e}")
            return

        # Persist to config.json
        cfg_path = os.path.expanduser("~/.zhuangzhou/config.json")
        try:
            existing = {}
            if os.path.exists(cfg_path):
                with open(cfg_path, encoding="utf-8") as f:
                    existing = json.load(f)
            existing["default_output_dir"] = new_path
            os.makedirs(os.path.dirname(cfg_path), exist_ok=True)
            with open(cfg_path, "w", encoding="utf-8") as f:
                json.dump(existing, f, indent=2, ensure_ascii=False)
        except Exception as e:
            console.print(f"[dim yellow]Warning: could not persist config: {e}[/]")

        console.print(
            f"[bold {ZZTheme.GREEN}]OK[/] Working directory set to [bold]{new_path}[/]\n"
            f"  [dim]All subsequent outputs will default here.[/]"
        )


class KGSummaryCommand(SlashCommand):
    trigger = "/kgsummary"
    description = "Summarize session entities and enrich with KG data. Usage: /kgsummary [--save] [--last N]"

    # Regex patterns to extract biomedical entities from text
    _PROTEIN_RE = re.compile(
        r'\b(?:CYP\w+|P450|ESM[\w-]*|[A-Z]{2,6}\d[\w-]*|'
        r'[A-Z][a-z]+(?:ase|ase\d|in|ase\s))\b'
    )
    _MUTATION_RE = re.compile(r'\b[A-Z]\d{1,4}[A-Z]\b')
    _COMPOUND_RE = re.compile(
        r'\b(?:camphor|heme|NADPH|FAD|FMN|ATP|ADP|[A-Z]{2,}(?:-\d+)?'
        r'|aspirin|warfarin|testosterone|caffeine)\b',
        re.IGNORECASE,
    )

    def execute(self, context, args: List[str]):
        agent = SlashCommandManager._unpack_agent(context)
        from zhuangzhou.ui.core import console, ZZTheme
        from rich.panel import Panel
        from rich.table import Table

        # --- Parse flags ---
        save_flag = "--save" in args
        last_n = 20
        remaining = [a for a in args if a != "--save"]
        if "--last" in remaining:
            idx = remaining.index("--last")
            try:
                last_n = int(remaining[idx + 1])
            except (IndexError, ValueError):
                console.print("[bold red]Usage:[/] /kgsummary [--save] [--last N]")
                return

        # --- Gather messages ---
        history = agent.executor.messages[-last_n:]
        if not history:
            console.print("[dim]No conversation history to summarize.[/]")
            return

        # Build corpus from user + assistant turns (skip system/tool_result)
        text_corpus = ""
        for msg in history:
            role = msg.get("role", "")
            content = msg.get("content", "")
            if isinstance(content, list):
                # tool_result or assistant content blocks
                for block in content:
                    if isinstance(block, dict) and block.get("type") == "text":
                        text_corpus += " " + block.get("text", "")
            elif isinstance(content, str) and role in ("user", "assistant"):
                text_corpus += " " + content

        # --- Extract entities ---
        proteins = sorted(set(self._PROTEIN_RE.findall(text_corpus)))
        mutations = sorted(set(self._MUTATION_RE.findall(text_corpus)))
        compounds = sorted(set(m.lower() for m in self._COMPOUND_RE.findall(text_corpus)))

        # Also do a simple word-based scan for PDB IDs (4-char alphanumeric)
        pdb_ids = sorted(set(re.findall(r'\b[1-9][A-Z0-9]{3}\b', text_corpus)))

        all_entities = list(set(proteins + mutations + compounds + pdb_ids))

        if not all_entities:
            console.print("[dim]No recognizable biomedical entities found in recent history.[/]")
            return

        # --- Query GraphRAG KG ---
        kg_hits: Dict[str, list] = {}
        kg_gaps: list = []

        try:
            from zhuangzhou.brain.graphrag import GraphRAGBridge
            bridge = GraphRAGBridge()

            for entity in all_entities[:15]:  # cap to avoid too many queries
                try:
                    result = bridge.search_entities([entity], kg_id="all", limit=3)
                    # result is a dict: {kg_id: [entity_dicts]}
                    found = []
                    for kg_id, ents in result.items():
                        for e in (ents or []):
                            desc = e.get("description", "") or e.get("summary", "")
                            if desc:
                                found.append(f"[{kg_id}] {desc[:120]}")
                    if found:
                        kg_hits[entity] = found
                    else:
                        kg_gaps.append(entity)
                except Exception:
                    kg_gaps.append(entity)
        except Exception as kg_err:
            console.print(f"[dim yellow]GraphRAG unavailable: {kg_err}[/]")
            kg_gaps = all_entities[:]

        # --- Build Markdown summary ---
        lines = ["# KG Conversation Summary\n"]
        lines.append(f"**Messages analyzed:** last {min(last_n, len(history))} turns\n")

        lines.append("## Entities Discussed")
        if proteins:
            lines.append(f"- **Proteins/Enzymes:** {', '.join(proteins[:20])}")
        if mutations:
            lines.append(f"- **Mutations:** {', '.join(mutations[:20])}")
        if compounds:
            lines.append(f"- **Compounds:** {', '.join(compounds[:15])}")
        if pdb_ids:
            lines.append(f"- **PDB IDs:** {', '.join(pdb_ids[:10])}")
        lines.append("")

        lines.append("## KG-Enriched Context")
        if kg_hits:
            for entity, descs in kg_hits.items():
                lines.append(f"### {entity}")
                for d in descs[:2]:
                    lines.append(f"  - {d}")
        else:
            lines.append("_(No KG matches found for identified entities)_")
        lines.append("")

        lines.append("## Session Knowledge Gaps")
        if kg_gaps:
            lines.append(
                "The following entities were mentioned but not found in the KG:\n"
                + ", ".join(kg_gaps)
            )
        else:
            lines.append("_(All identified entities matched KG records)_")

        summary_md = "\n".join(lines)

        # --- Display ---
        # Entities table
        ent_table = Table(title="Entities Detected", border_style=ZZTheme.MID)
        ent_table.add_column("Type", style=ZZTheme.CYAN)
        ent_table.add_column("Values", style="white")
        if proteins:
            ent_table.add_row("Proteins", ", ".join(proteins[:10]))
        if mutations:
            ent_table.add_row("Mutations", ", ".join(mutations[:10]))
        if compounds:
            ent_table.add_row("Compounds", ", ".join(compounds[:8]))
        if pdb_ids:
            ent_table.add_row("PDB IDs", ", ".join(pdb_ids[:8]))
        console.print(ent_table)

        # KG hits table
        if kg_hits:
            hits_table = Table(title="KG-Enriched Context", border_style=ZZTheme.MID)
            hits_table.add_column("Entity", style=f"bold {ZZTheme.CYAN}")
            hits_table.add_column("KG Description", style="white")
            for entity, descs in kg_hits.items():
                hits_table.add_row(entity, descs[0][:100] if descs else "")
            console.print(hits_table)

        if kg_gaps:
            console.print(
                f"\n[{ZZTheme.GOLD}]Knowledge gaps (not in KG):[/] "
                + ", ".join(kg_gaps[:15])
            )

        # --- Save ---
        if save_flag:
            save_path = os.path.expanduser("~/.zhuangzhou/session_kg_summary.md")
            try:
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                with open(save_path, "w", encoding="utf-8") as f:
                    f.write(summary_md)
                console.print(
                    f"[bold {ZZTheme.GREEN}]OK[/] KG summary saved to [bold]{save_path}[/]"
                )
            except Exception as e:
                console.print(f"[bold red]Save failed:[/] {e}")


class ColabCommand(SlashCommand):
    trigger = "/colab"
    description = "Manage Colab GPU connection: /colab [url] | /colab status | /colab setup"

    def execute(self, context, args: List[str]):
        from zhuangzhou.ui.core import console, ZZTheme
        from rich.panel import Panel
        from rich.table import Table

        sub = args[0] if args else "status"

        # /colab https://xxx.trycloudflare.com  — set URL directly
        if sub.startswith("https://"):
            url = sub.rstrip("/")
            from zhuangzhou.envs.colab_config import set_tunnel_url, get_config
            # Reset singleton so new URL takes effect
            import zhuangzhou.envs.colab_config as _cc
            _cc._config = None
            set_tunnel_url(url)
            console.print(
                f"[bold {ZZTheme.GREEN}]OK[/] Tunnel URL set: [bold]{url}[/]\n"
                f"  Saved to ~/.zhuangzhou/colab_config.json"
            )
            # Quick health check
            try:
                from zhuangzhou.envs.colab_sandbox import ColabSandbox
                sandbox = ColabSandbox()
                sandbox.tunnel_url = url
                result = sandbox.connect(timeout=10)
                if result.success:
                    data = result.data if isinstance(result.data, dict) else {}
                    console.print(
                        f"  [bold {ZZTheme.GREEN}]Connected[/] — GPU: {data.get('gpu_name', '?')}, "
                        f"uptime: {data.get('uptime_seconds', 0):.0f}s"
                    )
                else:
                    console.print(
                        f"  [{ZZTheme.GOLD}]Warning:[/] Server not reachable yet: {result.error}\n"
                        f"  The server may still be starting. Try /colab status in a few seconds."
                    )
            except Exception as e:
                console.print(f"  [dim]Health check error: {e}[/]")
            return

        # /colab status
        if sub == "status":
            from zhuangzhou.envs.colab_config import get_config
            import zhuangzhou.envs.colab_config as _cc
            _cc._config = None  # Force reload
            config = get_config()

            table = Table(box=None, show_header=False)
            table.add_column("Key", style=ZZTheme.CYAN)
            table.add_column("Value", style="white")
            table.add_row("Tunnel URL", config.tunnel_url or "[dim](not set)[/]")
            table.add_row("Runtime", config.runtime_type)
            table.add_row("Timeout", f"{config.timeout}s")

            if config.tunnel_url:
                try:
                    from zhuangzhou.envs.colab_sandbox import ColabSandbox
                    sandbox = ColabSandbox()
                    result = sandbox.connect(timeout=10)
                    if result.success:
                        data = result.data if isinstance(result.data, dict) else {}
                        table.add_row("Status", f"[bold {ZZTheme.GREEN}]Connected[/]")
                        table.add_row("GPU", data.get("gpu_name", "?"))
                        table.add_row("VRAM", f"{data.get('gpu_memory_total_gb', '?')} GB")
                        table.add_row("Uptime", f"{data.get('uptime_seconds', 0):.0f}s")
                    else:
                        table.add_row("Status", f"[bold red]Disconnected[/] — {result.error}")
                except Exception as e:
                    table.add_row("Status", f"[bold red]Error[/] — {e}")
            else:
                table.add_row("Status", "[dim]Not configured[/]")

            console.print(Panel(
                table,
                title=f"[bold {ZZTheme.CYAN}]COLAB GPU CONNECTION[/]",
                border_style=ZZTheme.MID,
            ))
            console.print(
                "[dim]Usage: /colab https://xxx.trycloudflare.com  — set URL manually\n"
                "       /colab setup  — auto-bootstrap via colab-mcp\n"
                "       /colab status  — check connection[/]"
            )
            return

        # /colab setup — run auto bootstrap
        if sub == "setup":
            console.print(f"[dim]Starting Colab auto-setup...[/]")
            try:
                from zhuangzhou.envs.colab_sandbox import ColabSandbox
                sandbox = ColabSandbox()
                result = sandbox.auto_setup(timeout=180)
                if result.success:
                    data = result.data if isinstance(result.data, dict) else {}
                    console.print(
                        f"[bold {ZZTheme.GREEN}]OK[/] Colab GPU server ready\n"
                        f"  URL: {data.get('tunnel_url', '?')}\n"
                        f"  Status: {data.get('status', '?')}"
                    )
                else:
                    console.print(
                        f"[bold red]Setup failed:[/] {result.error}\n"
                        f"  [dim]Tip: Open scripts/colab_gpu_server.ipynb in Colab manually,\n"
                        f"  copy the tunnel URL, then run: /colab https://xxx.trycloudflare.com[/]"
                    )
            except Exception as e:
                console.print(f"[bold red]Setup error:[/] {e}")
            return

        console.print(
            "[dim]Usage: /colab [url] | /colab status | /colab setup[/]"
        )


class SlashCommandManager:
    """Centralized manager to registry and dispatch commands."""

    def __init__(self):
        self.commands: Dict[str, SlashCommand] = {}
        self._register_defaults()

    def _register_defaults(self):
        self.register(HelpCommand())
        self.register(ModelCommand())
        self.register(SkillsCommand())
        self.register(GraphRAGCommand())
        self.register(ResumeCommand())
        self.register(ResetCommand())
        self.register(VerboseCommand())
        self.register(InitCommand())
        self.register(MemCommand())
        self.register(PlanCommand())
        self.register(AgentsCommand())
        self.register(BioDesignCommand())
        self.register(CompactCommand())
        self.register(WorkdirCommand())
        self.register(KGSummaryCommand())
        self.register(ColabCommand())

    def register(self, cmd: SlashCommand):
        self.commands[cmd.trigger] = cmd
        for alias in cmd.aliases:
            self.commands[alias] = cmd

    def dispatch(self, text: str, context: Any) -> bool:
        """Process a line of text. Returns True if it was a command.

        context may be either a ZhuangzhouAgent or a (agent, plan_mode) tuple.
        Commands that need plan_mode receive the raw context; others get
        the agent extracted from it automatically via _unpack_context().
        """
        if not text.startswith("/"):
            return False

        parts = text.split()
        trigger = parts[0].lower()
        args = parts[1:]

        if trigger in self.commands:
            self.commands[trigger].execute(context, args)
            return True

        return False

    @staticmethod
    def _unpack_agent(context: Any):
        """Extract agent from (agent, plan_mode) tuple or return as-is."""
        if isinstance(context, tuple) and len(context) >= 1:
            return context[0]
        return context

    @staticmethod
    def _unpack_plan_mode(context: Any):
        """Extract plan_mode from context tuple, or None."""
        if isinstance(context, tuple) and len(context) >= 2:
            return context[1]
        return None


# Global singleton
manager = SlashCommandManager()

