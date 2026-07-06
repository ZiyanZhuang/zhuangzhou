"""
Shi Yigong Agent (施一公 Agent)
===============================
A structural biology expert agent that reasons about protein mutations
using chemical intuition, not just statistical patterns.

Named after the renowned structural biologist who pioneered cryo-EM
structure determination of the spliceosome and other macromolecular
machines. His approach: look at the structure, understand the chemistry,
then predict function.

This agent can be invoked:
  1. As a standalone CLI tool
  2. As a BeTaThink ADK agent (in the synthetic biology pipeline)
  3. Via the Zhuangzhou API endpoint

The agent's "personality":
  - Skeptical of purely statistical predictions at conserved sites
  - Trusts chemical logic: steric, electronic, H-bonding, hydrophobic
  - Considers the catalytic mechanism holistically
  - Distinguishes enzyme-level activity from organism-level fitness
  - Provides natural language reasoning like a seminar discussion
"""

import json
import os
import sys
from typing import Dict, List, Optional, Set

# Add parent paths for imports
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(SCRIPT_DIR))  # BeTaThink root
sys.path.insert(0, os.path.join(os.path.dirname(SCRIPT_DIR), '..', 'Code', 'Uni-Mol-main', 'scripts'))

from .fusion_model import run_zhuangzhou_analysis
from .structural_context import build_all_contexts, generate_natural_language_description
from .chemical_intuition import AA_PROPERTIES


# ═══════════════════════════════════════════════
# LLM Prompt for Shi Yigong Agent
# ═══════════════════════════════════════════════

SHIYIGONG_SYSTEM_PROMPT = """你是施一公，世界顶尖的结构生物学家。你正在审查一组通过计算预测的酶突变方案。

## 你的核心原则

1. **化学直觉优先于统计**: PLM（蛋白质语言模型）学习的是进化共识。但进化选择的是生物体适应度，不是最大酶活性。一个极大提高催化速率的突变可能因为对宿主有毒而被淘汰。

2. **保守位点需要第一性原理**: 当PLM告诉你一个保守活性位点残基"应该"突变回共识序列时，这只是说这个位置在大多数物种中是那个氨基酸。但你手中的序列已经是优化过的，它可能就是那个罕见但强力的变体。

3. **空间化学**: 你能"看到"结构。氢键网络、π堆积、静电相互作用、空间位阻——这些才是决定催化的基本物理化学。

4. **机理思维**: 每个酶都有特定的催化机理。突变必须与机理兼容。丝氨酸蛋白酶的催化三联体中，Ser→Cys可能增强亲核性（pKa 8.3 vs 13），但Ser→Ala就彻底破坏了。

5. **正交组合**: 不同功能区域的突变可以正交组合（催化+稳定性），但同一功能区域的多个突变风险很高。

## 你的任务

分析提供的突变列表和结构信息，给出你作为结构生物学家的判断。对每个关键位置：
- 这个位置在做什么？(催化、底物定位、构象调节、稳定性)
- 突变后化学性质如何变化？(大小、电荷、氢键、疏水性)
- 这种变化对催化机理的影响是什么？
- PLM分数在这里可信吗？还是被进化偏见误导了？

用中文回答，像在组会上讨论一样自然。"""


MUTATION_ANALYSIS_TEMPLATE = """## 结构信息

蛋白: {protein_name} ({n_residues} 残基)
酶分类: {enzyme_class}
配体: {ligand}
活性位点残基: {binding_site}

## PLIP相互作用分析
{plip_summary}

## 待分析突变

{mutations_table}

## 每个位置的结构环境
{structural_contexts}

## Zhuangzhou化学分析预评估
{zhuangzhou_prescoring}

---

请以结构生物学家的视角，对以上突变进行深入分析。
特别关注:
1. 活性位点突变 — PLM分数是否误导？化学上合理吗？
2. 通道/入口突变 — 对底物进出有何影响？
3. 稳定性突变 — 哪些真的有助于热稳定性？
4. 你认为最佳的2-3个突变组合是什么？为什么？
"""


def format_mutations_table(mutations: List[dict]) -> str:
    """Format mutations as a readable table."""
    lines = ['| Mutation | PLM Score | Chemical Score | Zhuangzhou | Role | Recommendation |',
             '|----------|:---------:|:-------------:|:----------:|------|---------------|']
    for m in mutations:
        lines.append('| {} | {:.3f} | {:+.2f} | {:.3f} | {} | {} |'.format(
            m.get('mutation', '?'),
            m.get('plm_score', 0),
            m.get('chemical_score', 0),
            m.get('zhuangzhou_score', 0),
            m.get('role', '?'),
            m.get('recommendation', '')[:30],
        ))
    return '\n'.join(lines)


def format_structural_contexts(contexts: List[dict]) -> str:
    """Format structural contexts as natural language."""
    lines = []
    for ctx in contexts:
        desc = generate_natural_language_description(ctx.get('context', ctx))
        lines.append('- **Position {}**: {}'.format(ctx.get('position', '?'), desc))
    return '\n'.join(lines)


def build_agent_prompt(
    protein_name: str,
    n_residues: int,
    enzyme_class: str,
    ligand: str,
    binding_site: set,
    plip_summary: str,
    scored_mutations: List[dict],
    contexts: List[dict],
) -> str:
    """Build the full analysis prompt for the LLM agent."""
    return MUTATION_ANALYSIS_TEMPLATE.format(
        protein_name=protein_name,
        n_residues=n_residues,
        enzyme_class=enzyme_class,
        ligand=ligand,
        binding_site=', '.join(str(r) for r in sorted(binding_site)),
        plip_summary=plip_summary or '(PLIP data not available)',
        mutations_table=format_mutations_table(scored_mutations[:20]),
        structural_contexts=format_structural_contexts(contexts[:20]),
        zhuangzhou_prescoring='\n'.join(
            '- {}: {}'.format(m['mutation'], m.get('reasoning', ''))
            for m in scored_mutations[:10]
        ),
    )


def run_full_analysis(
    pdb_path: str,
    mutations: List[dict],
    binding_site: Set[int],
    output_dir: str,
    enzyme_class: str = '',
    ligand: str = '',
    protein_name: str = '',
    uniprot_id: str = '',
    sequence: str = '',
    conserved_positions: Set[int] = None,
    plip_json: str = None,
    fpocket_json: str = None,
    channel_residues: Set[int] = None,
) -> dict:
    """
    Run the complete Zhuangzhou / Shi Yigong analysis.

    This is the main entry point that:
    1. Builds structural contexts for all positions
    2. Runs chemical intuition analysis
    3. Fuses with PLM scores
    4. Generates the LLM prompt for deep reasoning
    5. Saves comprehensive results

    Returns:
        dict with all analysis results and the LLM prompt
    """
    os.makedirs(output_dir, exist_ok=True)

    # Step 1: Build structural contexts
    contexts = build_all_contexts(
        pdb_path=pdb_path,
        mutations=mutations,
        binding_site=binding_site,
        plip_json=plip_json,
        fpocket_json=fpocket_json,
        channel_residues=channel_residues,
        conserved_positions=conserved_positions,
        sequence=sequence,
    )

    # Step 2: Run Zhuangzhou fusion analysis
    analysis = run_zhuangzhou_analysis(
        mutations=mutations,
        contexts=contexts,
        binding_site=binding_site,
        enzyme_class=enzyme_class,
        uniprot_id=uniprot_id,
        sequence=sequence,
        conserved_positions=conserved_positions,
    )

    # Step 3: Build LLM prompt
    n_residues = len(sequence) if sequence else 0
    plip_summary = ''
    if plip_json and os.path.exists(plip_json):
        with open(plip_json, encoding='utf-8', errors='replace') as f:
            plip_data = json.load(f)
        interactions = plip_data.get('interactions', plip_data.get('binding_sites', []))
        plip_summary = json.dumps(interactions[:3], indent=2, ensure_ascii=False)

    llm_prompt = build_agent_prompt(
        protein_name=protein_name or os.path.basename(pdb_path),
        n_residues=n_residues,
        enzyme_class=enzyme_class,
        ligand=ligand,
        binding_site=binding_site,
        plip_summary=plip_summary,
        scored_mutations=analysis['ranked_mutations'],
        contexts=contexts,
    )

    # Step 4: Save results
    # Zhuangzhou report
    report_path = os.path.join(output_dir, 'zhuangzhou_report.md')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(analysis['report_narrative'])

    # LLM prompt (for external LLM invocation)
    prompt_path = os.path.join(output_dir, 'shi_yigong_prompt.md')
    with open(prompt_path, 'w', encoding='utf-8') as f:
        f.write(SHIYIGONG_SYSTEM_PROMPT)
        f.write('\n\n---\n\n')
        f.write(llm_prompt)

    # JSON results
    json_path = os.path.join(output_dir, 'zhuangzhou_results.json')
    # Clean non-serializable items
    clean_mutations = []
    for m in analysis['ranked_mutations']:
        clean = {k: v for k, v in m.items() if k != 'chemical_analysis'}
        clean_mutations.append(clean)

    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump({
            'ranked_mutations': clean_mutations,
            'first_principles_suggestions': analysis['first_principles_suggestions'],
            'n_conserved_overrides': analysis['n_conserved_overrides'],
            'scoring_method': analysis['scoring_method'],
        }, f, indent=2, ensure_ascii=False)

    print('Zhuangzhou report: {}'.format(report_path))
    print('Shi Yigong prompt: {}'.format(prompt_path))
    print('JSON results: {}'.format(json_path))

    return {
        **analysis,
        'llm_prompt': llm_prompt,
        'output_dir': output_dir,
    }


# ═══════════════════════════════════════════════
# CLI Entry Point
# ═══════════════════════════════════════════════

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
        description='Zhuangzhou: Chemical Intuition Engine for Enzyme Engineering'
    )
    parser.add_argument('--pdb', required=True, help='Protein PDB file')
    parser.add_argument('--mutations', required=True, help='ESM-2 mutations JSON')
    parser.add_argument('--binding-site', nargs='+', type=int, required=True,
                       help='Binding site residue numbers')
    parser.add_argument('--output', required=True, help='Output directory')
    parser.add_argument('--enzyme-class', default='', help='Enzyme class (e.g. oxidoreductase)')
    parser.add_argument('--ligand', default='', help='Ligand name')
    parser.add_argument('--protein-name', default='', help='Protein name')
    parser.add_argument('--uniprot-id', default='', help='UniProt accession')
    parser.add_argument('--sequence', default='', help='Protein sequence')
    parser.add_argument('--conserved', nargs='*', type=int, default=[],
                       help='Known conserved positions')
    parser.add_argument('--plip-json', default=None, help='PLIP results JSON')
    parser.add_argument('--fpocket-json', default=None, help='fpocket results JSON')

    args = parser.parse_args()

    with open(args.mutations, encoding='utf-8', errors='replace') as f:
        mut_data = json.load(f)

    result = run_full_analysis(
        pdb_path=args.pdb,
        mutations=mut_data.get('mutations', mut_data.get('classified', [])),
        binding_site=set(args.binding_site),
        output_dir=args.output,
        enzyme_class=args.enzyme_class,
        ligand=args.ligand,
        protein_name=args.protein_name,
        uniprot_id=args.uniprot_id,
        sequence=args.sequence,
        conserved_positions=set(args.conserved) if args.conserved else None,
        plip_json=args.plip_json,
        fpocket_json=args.fpocket_json,
    )

    print('\nTop 10 by Zhuangzhou Score:')
    for i, m in enumerate(result['ranked_mutations'][:10], 1):
        print('  {}. {} ZZ={:.3f} (Chem={:+.2f}, PLM={:.3f}, C{:.0f}/P{:.0f}) {}'.format(
            i, m['mutation'], m['zhuangzhou_score'],
            m['chemical_score'], m['plm_score'],
            m['weight_chemical'] * 100, m['weight_plm'] * 100,
            m['recommendation'][:40]))

    if result['first_principles_suggestions']:
        print('\nFirst-Principles Suggestions (active site):')
        for s in result['first_principles_suggestions'][:5]:
            print('  {} score={:+.2f} — {}'.format(
                s['mutation'], s['chemical_score'],
                '; '.join(s.get('key_reasoning', []))[:60]))
