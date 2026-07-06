"""
Zhuangzhou Interactive Chat REPL
=================================

Natural language interface to the Zhuangzhou enzyme engineering system.
Updated with prompt_toolkit for slash-command completions.
"""

import os
import sys
import argparse
from prompt_toolkit import PromptSession, HTML
from prompt_toolkit.history import FileHistory
from prompt_toolkit.auto_suggest import AutoSuggestFromHistory

# Windows UTF-8
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")


def handle_init_command():
    """Handle the /init command: scan environment and update ZHUANGZHOU.md."""
    from zhuangzhou.memory.project_memory import ProjectMemory
    from zhuangzhou.tools import check_tool_availability, _DEFAULT_UNIMOL_ROOT
    from zhuangzhou.ui.core import ZZInterface, console

    ZZInterface.print_action("Initializing Project Environment")

    memory = ProjectMemory()
    memory.ensure_local_file()

    avail = check_tool_availability()
    vina_path = "E:/computing bio/MDS-gmx/Docking/vina/vina.exe"
    gmx_path = "E:/gmx2025.1_GPU/bin/gmx.exe"
    unimol_root = _DEFAULT_UNIMOL_ROOT

    files = os.listdir(".")
    pdbs = [f for f in files if f.endswith(".pdb")]
    sdfs = [f for f in files if f.endswith(".sdf")]

    from zhuangzhou.skills.registry import get_registry

    registry = get_registry()
    skill_categories = {}
    for s in registry.list_all():
        skill_categories[s.category] = skill_categories.get(s.category, 0) + 1

    skill_summary = ", ".join(
        [f"{cat} ({count})" for cat, count in skill_categories.items()]
    )

    import datetime
    env_snapshot = f"""
- **GROMACS:** {"[OK] " + gmx_path if avail.get("gmx_binary") else "[MISSING]"}
- **Vina:** {"[OK] " + vina_path if avail.get("vina_binary") else "[MISSING]"}
- **Uni-Mol:** {"[OK] " + unimol_root if os.path.exists(unimol_root) else "[MISSING]"}
- **Local Files:** {len(pdbs)} PDBs, {len(sdfs)} SDFs found in CWD.
- **Available Skills:** {skill_summary}
- **Timestamp:** {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""

    memory.update_section("Environment", env_snapshot)
    console.print(
        f"[bold green]✓[/bold green] ZHUANGZHOU.md updated with environmental snapshot."
    )
    console.print(f"[dim]Path: {memory.memory_file}[/dim]")


def run_repl(
    backend: str = None, api_key: str = None, model: str = None, verbose: bool = False
):
    """Run the interactive chat REPL."""
    from zhuangzhou.nlp.agent import ZhuangzhouAgent
    from zhuangzhou.nlp.slash import manager as slash_manager
    from zhuangzhou.nlp.completer import ZhuangzhouCompleter
    from zhuangzhou.ui.core import (
        ZZInterface,
        ZZStatus,
        console,
        ZZTheme,
        Text,
        format_tokens,
        format_elapsed,
        build_context_bar,
        build_context_hint,
    )

    # Create agent
    try:
        agent = ZhuangzhouAgent(
            backend=backend,
            api_key=api_key,
            model=model,
            verbose=verbose,
        )
    except Exception as e:
        console.print(f"\n[bold red]Error:[/] Failed to initialize agent: {e}")
        return

    import logging
    logger = logging.getLogger(__name__)

    # Initialize Plan Mode (default: enabled)
    plan_mode = None
    try:
        from zhuangzhou.nlp.plan_mode import PlanMode
        plan_mode = PlanMode(backend=agent.backend, registry=agent.registry, enabled=True)
    except Exception as e:
        logger.warning("PlanMode init failed (disabled): %s", e)
        console.print(f"[dim yellow]⚠ Plan Mode init failed: {e}[/]")

    # Print unified banner
    ZZInterface.render_logo()

    # Status bar
    total_skills = sum(1 for _ in agent.registry.list_all())
    status_line = Text()
    status_line.append("  ", style="")
    status_line.append(f"Backend: ", style="dim")
    status_line.append(f"{agent.backend_name}", style=f"bold {ZZTheme.CYAN}")
    status_line.append("  │  ", style=ZZTheme.DIM)
    status_line.append(f"Tools: ", style="dim")
    status_line.append(f"{agent.skill_count}", style=f"bold {ZZTheme.GOLD}")
    status_line.append(f" loaded", style="dim")
    if hasattr(agent.backend, "model"):
        status_line.append("  │  ", style=ZZTheme.DIM)
        status_line.append(f"Model: ", style="dim")
        status_line.append(f"{agent.backend.model}", style="white")
    if plan_mode is not None:
        status_line.append("  │  ", style=ZZTheme.DIM)
        status_line.append(f"Plan: ", style="dim")
        status_line.append(f"{'ON' if plan_mode.enabled else 'OFF'}", style=f"bold {ZZTheme.GREEN if plan_mode.enabled else ZZTheme.DIM}")
    console.print(status_line)

    console.print("""[dim]
  Type [bold]/help[/bold] for commands  │  [bold]Ctrl+C[/bold] interrupt  │  [bold]Ctrl+D[/bold] exit
[/dim]""")

    # Initialize PromptSession
    history_file = os.path.expanduser("~/.zhuangzhou/history.txt")
    os.makedirs(os.path.dirname(history_file), exist_ok=True)

    session = PromptSession(
        history=FileHistory(history_file),
        completer=ZhuangzhouCompleter(),
        auto_suggest=AutoSuggestFromHistory(),
        complete_while_typing=True,
    )

    # Styling for HTML prompt (matches zz.prompt and zz.chevron)
    prompt_html = HTML(
        f'<style color="{ZZTheme.CYAN}">zhuangzhou</style><style color="{ZZTheme.GOLD}"> ❯ </style>'
    )

    session_tokens = 0
    session_queries = 0

    while True:
        try:
            console.print("")
            user_input = session.prompt(prompt_html).strip()
        except KeyboardInterrupt:
            continue  # CTRL+C within prompt: clear line
        except EOFError:
            console.print(
                f"\n\n[dim]Session: {session_queries} queries, {format_tokens(session_tokens)} tokens[/dim]"
            )
            console.print("[dim]Bye![/dim]")
            break

        if not user_input:
            continue

        # 1. Handle Commands via Slash Manager
        if user_input.startswith("/"):
            # Pass plan_mode reference via context tuple so commands can toggle it
            ctx = (agent, plan_mode)
            if slash_manager.dispatch(user_input, ctx):
                continue
            if user_input.lower() in ("/exit", "/quit", "/q"):
                break
            console.print(
                f"[dim]Unknown command: {user_input}. Type /help for list.[/]"
            )
            continue

        # 2a. Bio-design Mode — intercept metabolic engineering queries
        effective_input = user_input
        _biodesign_keywords = [
            "代谢通路", "代谢工程", "合成生物", "设计生产", "生产通路",
            "代谢流", "通量分析", "酶工程", "启动子", "底盘菌",
            "metabolic engineering", "biosynthesis", "生物合成",
            "通路设计", "bio-design", "pathway design",
            "metakg", "metakg_", "synbio_", "底盘细胞", "chassis",
            "大肠杆菌生产", "酵母生产", "ecoli production", "yeast production",
        ]
        _is_biodesign = any(kw in user_input.lower() for kw in _biodesign_keywords)
        if _is_biodesign:
            try:
                from zhuangzhou.nlp.biodesign_mode import BioDesignMode
                bdm = BioDesignMode(backend=agent.backend, registry=agent.registry)
                bd_result = bdm.run(user_input, console=console)
                if not bd_result.skipped and bd_result.design:
                    effective_input = bd_result.agent_message
                    # Skip Plan Mode for bio-design queries
                    goto_agent = True
                else:
                    goto_agent = False
            except KeyboardInterrupt:
                console.print("\n[dim][Bio-design interrupted][/dim]")
                goto_agent = False
            except Exception as e:
                logger.debug("BioDesignMode error: %s", e)
                goto_agent = False
        else:
            goto_agent = True

        # 2b. Plan Mode preprocessing (default: enabled, skipped if bio-design handled)
        if goto_agent and plan_mode is not None:
            try:
                pm_result = plan_mode.process(user_input, console=console)
                if not pm_result.skipped:
                    effective_input = pm_result.agent_message
            except KeyboardInterrupt:
                console.print("\n[dim][Plan Mode interrupted, sending original query][/dim]")
            except Exception as e:
                logger.debug("PlanMode.process error: %s", e)

        # 3. Natural language -> Agent
        try:
            session_queries += 1
            with ZZStatus(action="Reasoning") as monitor:
                response = agent.chat(effective_input, status_monitor=monitor)
                session_tokens += monitor.total_tokens
        except KeyboardInterrupt:
            console.print("\n[dim][Interrupted][/dim]")
        except Exception as e:
            from zhuangzhou.ui.core import PHASE_ICONS

            console.print(f"\n[bold red]{PHASE_ICONS['error']} Error:[/] {e}")
            if verbose:
                import traceback

                traceback.print_exc()


def main():
    parser = argparse.ArgumentParser(description="Zhuangzhou NLP Chat Interface")
    parser.add_argument(
        "--backend",
        "-b",
        choices=["claude", "deepseek", "openai", "ollama"],
        help="LLM backend (default: auto-detect)",
    )
    parser.add_argument("--model", "-m", help="Model name override")
    parser.add_argument(
        "--api-key",
        "-k",
        help="Deprecated: use provider environment variables instead.",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Show tool call details"
    )
    parser.add_argument(
        "--oneshot", "-1", help="One-shot mode: process single message and exit"
    )

    args = parser.parse_args()
    if args.api_key:
        parser.error("--api-key is not accepted because command-line secrets leak via shell history and process lists. Use environment variables instead.")

    if args.oneshot:
        from zhuangzhou.nlp import chat

        result = chat(args.oneshot, backend=args.backend)
        print(result)
    else:
        run_repl(
            backend=args.backend,
            api_key=None,
            model=args.model,
            verbose=args.verbose,
        )


if __name__ == "__main__":
    main()
