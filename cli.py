#!/usr/bin/env python3
"""
Zhuangzhou CLI
==============
Command-line interface for the Zhuangzhou enzyme engineering framework.

Usage:
  python -m zhuangzhou.cli run --pdb protein.pdb --binding-site 15,21,23 --esm2-dirs dir1,dir2
  python -m zhuangzhou.cli tools --check
  python -m zhuangzhou.cli analyze --mutation V21L --pdb protein.pdb --binding-site 15,21,23
"""

import argparse
import json
import os
import sys

# Ensure parent is on path
BETATHINK_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BETATHINK_ROOT not in sys.path:
    sys.path.insert(0, BETATHINK_ROOT)


def cmd_run(args):
    """Run the full 5-phase Zhuangzhou pipeline."""
    from zhuangzhou.core import ZhuangzhouFramework
    from zhuangzhou.ui.core import ZZInterface, console

    ZZInterface.render_logo()
    ZZInterface.print_action("Initializing Pipeline")

    binding_site = set(int(x) for x in args.binding_site.split(','))
    conserved = set(int(x) for x in args.conserved.split(',')) if args.conserved else set()
    esm2_dirs = args.esm2_dirs.split(',') if args.esm2_dirs else []

    config = {
        'unimol_root': args.unimol_root,
        'vina_path': args.vina_path,
        'gmx_path': args.gmx_path,
    }

    fw = ZhuangzhouFramework(config)
    results = fw.run_enzyme_engineering(
        pdb_path=args.pdb,
        binding_site=binding_site,
        esm2_dirs=esm2_dirs,
        enzyme_class=args.enzyme_class or '',
        ligand=args.ligand or '',
        protein_name=args.name or os.path.basename(args.pdb),
        conserved_positions=conserved,
        output_dir=args.output or None,
    )

    console.print(f"\n[zz.accent]\u2503 ENGINEERING REPORT COMPLETE[/]")
    console.print(f"  [dim]Report:[/] [zz.primary]{results.get('report_path', 'N/A')}[/]")
    console.print(f"  [dim]JSON:[/]   [zz.primary]{results.get('json_path', 'N/A')}[/]\n")


def cmd_tools(args):
    """Check available tools."""
    from zhuangzhou.tools import check_tool_availability, TOOL_REGISTRY
    from zhuangzhou.ui.core import ZZInterface, console, ZZTheme

    config = {'unimol_root': args.unimol_root} if args.unimol_root else {}
    avail = check_tool_availability(config)

    ZZInterface.render_logo()
    table = ZZInterface.create_table("Zhuangzhou Tool Availability", ["Component", "Status", "Requirements"])
    
    for name, info in TOOL_REGISTRY.items():
        is_ok = avail.get(name, False)
        status = "[bold green]OK[/]" if is_ok else "[bold red]MISSING[/]"
        gpu = "[zz.accent]GPU[/]" if info.get('requires_gpu') else "-"
        table.add_row(name, status, gpu)

    console.print(table)

    console.print(f"\n[zz.accent]\u2503 External Binaries[/]")
    for key in ('vina_binary', 'gmx_binary', 'wsl'):
        is_ok = avail.get(key, False)
        status = "[bold green]OK[/]" if is_ok else "[bold red]MISSING[/]"
        console.print(f"  [dim]{key:25s}[/] {status}")
    console.print("")


def cmd_analyze(args):
    """Analyze a single mutation with chemical intuition."""
    from zhuangzhou.chemical_intuition import analyze_mutation_chemistry
    from zhuangzhou.ui.core import ZZInterface, console, ZZTheme, Panel, Text
    from rich.box import ROUNDED

    mut_str = args.mutation
    wt_aa = mut_str[0]
    mut_aa = mut_str[-1]
    position = int(mut_str[1:-1])

    context = {
        'role': args.role or 'unknown',
        'secondary_structure': 'loop',
        'interactions': [],
        'neighbors': [],
        'distance_to_site': float(args.distance) if args.distance else None,
        'sasa': None,
        'in_conserved_motif': args.conserved_flag,
    }

    result = analyze_mutation_chemistry(wt_aa, mut_aa, context)

    ZZInterface.render_logo()
    
    score = result['chemical_score']
    score_color = "green" if score > 0 else "red" if score < -1 else "yellow"
    
    content = Text()
    content.append(f"Mutation: ", style="dim")
    content.append(f"{mut_str}\n", style=ZZTheme.GOLD)
    content.append(f"Score:    ", style="dim")
    content.append(f"{score:+.2f}\n", style=f"bold {score_color}")
    content.append(f"Result:   ", style="dim")
    content.append(f"{result['recommendation']}\n", style="white")
    
    if result.get('reasoning'):
        content.append("\nRational:\n", style="italic " + ZZTheme.CYAN)
        for r in result['reasoning']:
            content.append(f" \u2022 {r}\n", style="dim")
            
    panel = Panel(
        content,
        title=f"[bold zz.primary] Chemical Analysis [/]",
        border_style=ZZTheme.CYAN,
        box=ROUNDED,
        padding=(1, 2)
    )
    console.print(panel)
    
    if result.get('warnings'):
        for w in result['warnings']:
            ZZInterface.print_warning(w)
    console.print("")


def main():
    parser = argparse.ArgumentParser(
        description='Zhuangzhou Enzyme Engineering Framework',
        prog='zhuangzhou',
    )
    # Global options
    parser.add_argument('--model', '-m', help='LLM model to use (e.g. ollama:deepseek-r1:8b, claude-sonnet)')
    
    sub = parser.add_subparsers(dest='command')

    # run
    p_run = sub.add_parser('run', help='Run the full 5-phase pipeline')
    p_run.add_argument('--pdb', required=True, help='Input PDB file')
    p_run.add_argument('--binding-site', required=True,
                       help='Comma-separated active site residue numbers')
    p_run.add_argument('--esm2-dirs', help='Comma-separated ESM-2 result directories')
    p_run.add_argument('--conserved', help='Comma-separated conserved position numbers')
    p_run.add_argument('--enzyme-class', help='Enzyme class (oxidoreductase, etc.)')
    p_run.add_argument('--ligand', help='Ligand name')
    p_run.add_argument('--name', help='Protein name')
    p_run.add_argument('--output', help='Output directory')
    p_run.add_argument('--unimol-root', default=None,
                       help='Path to Uni-Mol-main root')
    p_run.add_argument('--vina-path', default='E:/computing bio/MDS-gmx/Docking/vina/vina.exe')
    p_run.add_argument('--gmx-path', default='E:/gmx2025.1_GPU/bin/gmx.exe')

    # tools
    p_tools = sub.add_parser('tools', help='Check tool availability')
    p_tools.add_argument('--check', action='store_true', default=True)
    p_tools.add_argument('--unimol-root', default=None)

    # analyze
    p_analyze = sub.add_parser('analyze', help='Analyze a single mutation')
    p_analyze.add_argument('--mutation', required=True, help='Mutation string (e.g., V21L)')
    p_analyze.add_argument('--role', help='Structural role (active_site, buried_core, surface)')
    p_analyze.add_argument('--distance', help='Distance to active site center')
    p_analyze.add_argument('--conserved-flag', action='store_true',
                           help='Mark position as conserved')

    args = parser.parse_args()

    # Pass global model to commands if they support it or via env
    if args.model:
        os.environ['ZHUANGZHOU_DEFAULT_MODEL'] = args.model

    if args.command == 'run':
        cmd_run(args)
    elif args.command == 'tools':
        cmd_tools(args)
    elif args.command == 'analyze':
        cmd_analyze(args)
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
