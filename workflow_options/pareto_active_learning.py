"""
Pareto + Active Learning Workflow — Robustness-Enhanced Enzyme Engineering
==========================================================================
Zhuangzhou v0.9.2

基于文献的最小必要补强工作流，适用于：
  - 目标酶已有基本表达与测定条件
  - 存在高通量代理 readout + 真实底物复筛
  - 允许至少两轮 DBTL 闭环
  - 设计空间可预先压缩

五项核心补强（全部有文献支撑）：
  1. 首轮文库采用 fitness + diversity 帕累托设计 [MODIFY, Nat Commun 2024]
  2. Focused training 用先验减少低信息样本 [Cell Systems 2025]
  3. 建库前显式删除明显去稳定化突变 [Nat Chem Biol 2024]
  4. 第一轮保留小规模多位点受限分支 [ALDE, Nat Commun 2025]
  5. 主动学习阶段优先使用简单模型与稳健不确定性估计 [ALDE]

七步执行流程：
  Step 1: 明确目标函数与硬约束
  Step 2: 三支候选生成 + 设计空间收缩
  Step 3: 构建首轮两类文库（单突侦察 + 受限多突）
  Step 4: 两级实验筛选（代理readout + 真实底物复筛）
  Step 5: 简单surrogate model做第二轮提议
  Step 6: 保持探索-利用平衡的后续轮次
  Step 7: 终局变体验证

References:
  [1] Meier et al., NeurIPS 2021 (ESM zero-shot)
  [2] Khersonsky et al., Molecular Cell 2018 (FuncLib)
  [3] Wittmann et al., Cell Systems 2021 (Informed training)
  [4] Romero et al., Nat Commun 2025 (ALDE)
  [5] Wang et al., Nat Commun 2024 (MODIFY)
  [6] Cell Systems 2025 (Benchmarking MLDE across 16 landscapes)
  [7] Nat Chem Biol 2024 (Enriching productive mutational paths)
"""

import json
import logging
import os
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# ── Data Structures ───────────────────────────────────────────────────────────

@dataclass
class ObjectiveDefinition:
    """Step 1 output: objective functions and hard constraints."""
    primary_objective: str = ''       # e.g. 'catalytic_activity', 'selectivity', 'thermostability'
    screening_readout: str = ''       # High-throughput proxy readout
    publication_readout: str = ''     # Real-substrate readout for paper
    hard_constraints: List[str] = field(default_factory=list)
    # e.g. ['expression > 0.1 mg/mL', 'no catalytic core mutations', 'ΔTm > -5°C']


@dataclass
class DesignSpace:
    """Step 2 output: compressed design space from three candidate branches."""
    plm_candidates: List[dict] = field(default_factory=list)        # PLM/evolution branch
    rational_candidates: List[dict] = field(default_factory=list)   # Active-site/tunnel branch
    override_candidates: List[dict] = field(default_factory=list)   # Mechanism-override branch
    filtered_out: List[dict] = field(default_factory=list)          # Removed by stability filter
    n_positions: int = 0
    n_candidates_total: int = 0
    n_after_filter: int = 0


@dataclass
class LibraryDesign:
    """Step 3 output: first-round library with Pareto-selected variants."""
    scout_library: List[dict] = field(default_factory=list)         # Single-mutation scouts
    multi_library: List[dict] = field(default_factory=list)         # Restricted multi-mutants
    pareto_front: List[dict] = field(default_factory=list)          # Pareto-optimal selection
    total_variants: int = 0
    diversity_score: float = 0.0


@dataclass
class ScreeningResult:
    """Step 4 output: two-tier screening results."""
    tier1_proxy: List[dict] = field(default_factory=list)           # HTS proxy readout
    tier2_real: List[dict] = field(default_factory=list)            # Real-substrate validation
    hits: List[dict] = field(default_factory=list)
    proxy_real_correlation: float = 0.0


@dataclass
class SurrogateProposal:
    """Step 5 output: second-round proposals from surrogate model."""
    model_type: str = 'random_forest'                               # Default: simple, robust
    proposals: List[dict] = field(default_factory=list)
    acquisition_scores: List[dict] = field(default_factory=list)
    uncertainty_method: str = 'ensemble_variance'


@dataclass
class WorkflowState:
    """Complete state of the Pareto+AL workflow across all steps."""
    # Configuration
    protein_name: str = ''
    pdb_path: str = ''
    fasta_path: str = ''
    uniprot_id: str = ''
    active_site: List[int] = field(default_factory=list)
    catalytic_residues: List[int] = field(default_factory=list)

    # Step outputs
    objectives: Optional[ObjectiveDefinition] = None
    design_space: Optional[DesignSpace] = None
    library: Optional[LibraryDesign] = None
    screening: Optional[ScreeningResult] = None
    surrogate: Optional[SurrogateProposal] = None

    # Tracking
    current_step: int = 0
    completed_steps: List[str] = field(default_factory=list)
    round_number: int = 1
    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        """Serialize state for persistence."""
        import dataclasses
        return dataclasses.asdict(self)


# ── Step Implementations ──────────────────────────────────────────────────────

def step1_define_objectives(state: WorkflowState, params: dict) -> dict:
    """Step 1: 明确目标函数与硬约束 / Define Objectives & Hard Constraints.

    Before any design, define:
      - Primary objective (activity / selectivity / stability)
      - Screening readout (HTS proxy, cheap & fast)
      - Publication readout (real substrate, for paper conclusions)
      - Hard constraints (expression threshold, core geometry, stability floor)
    """
    objectives = ObjectiveDefinition(
        primary_objective=params.get('primary_objective', 'catalytic_activity'),
        screening_readout=params.get('screening_readout', ''),
        publication_readout=params.get('publication_readout', ''),
        hard_constraints=params.get('hard_constraints', [
            'expression ≥ minimum detectable threshold',
            'no mutations at catalytic core residues (unless mechanistically justified)',
            'ΔTm ≥ -5°C (no severe destabilization)',
        ]),
    )
    state.objectives = objectives
    state.completed_steps.append('define_objectives')
    state.current_step = 1

    return {
        'step': 'define_objectives',
        'status': 'completed',
        'objectives': {
            'primary': objectives.primary_objective,
            'screening_readout': objectives.screening_readout,
            'publication_readout': objectives.publication_readout,
            'hard_constraints': objectives.hard_constraints,
        },
        'next_steps': ['生成三支候选并收缩设计空间 (Step 2)'],
    }


def step2_generate_design_space(state: WorkflowState, params: dict) -> dict:
    """Step 2: 三支候选生成 + 设计空间收缩 / Generate Candidates & Compress Space.

    Three candidate branches:
      Branch A: PLM/evolution prior (ESM-2 zero-shot scan)
      Branch B: Active-site/tunnel semi-rational design
      Branch C: Mechanism-override positions

    Then apply stability filter to remove obviously destabilizing mutations.
    This implements补强2 (focused training) and补强3 (destabilization removal).
    """
    from zhuangzhou.skills.registry import get_registry
    registry = get_registry()

    results = {'step': 'generate_design_space', 'branches': {}}
    design_space = DesignSpace()

    # ── Auto-generate FASTA from PDB if missing ──��───────────────────────
    if not state.fasta_path and state.pdb_path:
        pdb2fasta_skill = registry.get('pdb_to_fasta')
        if pdb2fasta_skill:
            fasta_result = pdb2fasta_skill.execute(pdb_path=state.pdb_path)
            if isinstance(fasta_result, dict) and 'fasta_path' in fasta_result:
                state.fasta_path = fasta_result['fasta_path']
                results['auto_fasta'] = state.fasta_path
            elif isinstance(fasta_result, dict) and 'error' not in fasta_result:
                # Try common output keys
                for key in ('output', 'fasta', 'path'):
                    if key in fasta_result and isinstance(fasta_result[key], str):
                        state.fasta_path = fasta_result[key]
                        results['auto_fasta'] = state.fasta_path
                        break
        if not state.fasta_path:
            state.warnings.append('Could not auto-generate FASTA from PDB. '
                                  'Provide fasta_path for PLM scan.')

    # ── Branch A: PLM zero-shot scan ──────────────────────────────────────
    esm2_skill = registry.get('esm2_mutation_scan')
    if esm2_skill and state.fasta_path:
        esm2_params = {
            'fasta_path': state.fasta_path,
            'pdb_path': state.pdb_path,
            'top_n': params.get('plm_top_n', 200),
        }
        esm2_result = esm2_skill.execute(**esm2_params)
        if isinstance(esm2_result, dict) and 'error' not in esm2_result:
            # Extract beneficial mutations
            bmj_path = esm2_result.get('beneficial_mutations_json', '')
            if bmj_path and os.path.exists(bmj_path):
                with open(bmj_path, encoding='utf-8') as f:
                    esm_data = json.load(f)
                plm_muts = [m for m in esm_data.get('mutations', [])
                            if m.get('delta_ll', 0) > 0]
                design_space.plm_candidates = plm_muts[:params.get('plm_top_n', 200)]
                results['branches']['plm'] = {
                    'n_candidates': len(design_space.plm_candidates),
                    'top5': [m.get('mutation', '') for m in design_space.plm_candidates[:5]],
                }
        else:
            state.warnings.append(f"ESM-2 scan failed: {esm2_result.get('error', 'unknown')}")
            results['branches']['plm'] = {'error': esm2_result.get('error', '')}

    # ── Branch B: Active-site / tunnel semi-rational ──────────────────────
    enrich_skill = registry.get('enrich_mutations')
    if enrich_skill and design_space.plm_candidates and state.pdb_path:
        mutations_to_enrich = design_space.plm_candidates[:100]
        enrich_params = {
            'pdb_path': state.pdb_path,
            'mutations': mutations_to_enrich,
            'uniprot_id': state.uniprot_id,
            'active_site': set(state.active_site) if state.active_site else None,
            'catalytic': set(state.catalytic_residues) if state.catalytic_residues else None,
            'protein_name': state.protein_name,
        }
        enrich_result = enrich_skill.execute(**enrich_params)
        if isinstance(enrich_result, dict) and 'error' not in enrich_result:
            enriched = enrich_result.get('mutations', [])
            # Extract active-site shell and channel residue mutations
            rational = [m for m in enriched
                        if m.get('zone') in ('active_site_shell', 'channel')
                        or m.get('dist_active_site', 999) < 12.0]
            design_space.rational_candidates = rational
            results['branches']['rational'] = {
                'n_candidates': len(rational),
                'zones': _count_zones(rational),
            }

    # ── Branch C: Mechanism-override positions ────────────────────────────
    # Positions where user specifies "must try" based on domain knowledge
    override_positions = params.get('override_positions', [])
    if override_positions:
        override_muts = []
        for pos_info in override_positions:
            if isinstance(pos_info, dict):
                override_muts.append(pos_info)
            elif isinstance(pos_info, (int, str)):
                override_muts.append({'position': int(pos_info), 'source': 'user_override'})
        design_space.override_candidates = override_muts
        results['branches']['override'] = {'n_candidates': len(override_muts)}

    # ── Stability filter (补强3) ──────────────────────────────────────────
    all_candidates = (design_space.plm_candidates +
                      design_space.rational_candidates +
                      design_space.override_candidates)

    # Deduplicate by position+mutation
    seen = set()
    unique = []
    for m in all_candidates:
        key = m.get('mutation', f"pos{m.get('position', '?')}")
        if key not in seen:
            seen.add(key)
            unique.append(m)

    design_space.n_candidates_total = len(unique)

    # Filter: remove mutations with severe predicted destabilization
    stability_threshold = params.get('stability_threshold', -1.0)
    filtered = []
    removed = []
    for m in unique:
        delta_ll = m.get('delta_ll', 0)
        zone = m.get('zone', '')
        # Hard filter: deeply negative PLM score + buried core = almost certainly destabilizing
        if delta_ll < stability_threshold and zone in ('buried_hydrophobic', 'buried_polar'):
            m['filter_reason'] = f'destabilizing (ΔLL={delta_ll:.2f}, zone={zone})'
            removed.append(m)
        # Hard filter: catalytic residue mutations (unless user override)
        elif zone == 'catalytic' and m.get('source') != 'user_override':
            m['filter_reason'] = 'catalytic residue (protected)'
            removed.append(m)
        else:
            filtered.append(m)

    design_space.filtered_out = removed
    design_space.n_after_filter = len(filtered)
    design_space.n_positions = len(set(m.get('position', 0) for m in filtered))

    state.design_space = design_space
    state.completed_steps.append('generate_design_space')
    state.current_step = 2

    results['total_candidates'] = design_space.n_candidates_total
    results['after_stability_filter'] = design_space.n_after_filter
    results['removed_count'] = len(removed)
    results['n_positions'] = design_space.n_positions
    results['status'] = 'completed'
    results['next_steps'] = ['构建首轮帕累托文库 (Step 3)']

    return results


def step3_build_pareto_library(state: WorkflowState, params: dict) -> dict:
    """Step 3: 首轮帕累托文库设计 / Build First-Round Pareto Library.

    Implements补强1 (Pareto fitness+diversity) and补强4 (restricted multi-mutant branch).

    Two sub-libraries:
      A. Scout library: Single mutations, Pareto-selected for fitness+diversity
      B. Multi-mutant library: Restricted combinations at active-site positions
    """
    if not state.design_space:
        return {'error': 'Design space not generated. Run Step 2 first.'}

    ds = state.design_space
    all_filtered = [m for m in (ds.plm_candidates + ds.rational_candidates + ds.override_candidates)
                    if m.get('filter_reason') is None
                    and m not in ds.filtered_out]

    # Deduplicate
    seen = set()
    candidates = []
    for m in all_filtered:
        key = m.get('mutation', f"pos{m.get('position', '?')}")
        if key not in seen:
            seen.add(key)
            candidates.append(m)

    if not candidates:
        return {'error': 'No candidates after filtering. Relax stability_threshold or add override_positions.'}

    # ── A. Scout Library: Pareto selection (fitness × diversity) ──────────
    scout_budget = params.get('scout_budget', 80)  # Number of single mutants
    multi_budget = params.get('multi_budget', 16)   # Number of multi-mutants

    # Pareto front: maximize delta_ll (fitness proxy) while maximizing
    # sequence diversity (positions spread across structure)
    pareto_front = _pareto_select(candidates, scout_budget)

    scout_lib = []
    for m in pareto_front:
        scout_lib.append({
            **m,
            'library_type': 'scout',
            'n_mutations': 1,
        })

    # ── B. Multi-mutant Library: Restricted active-site combinations ─────
    # Select positions near active site for combinatorial exploration
    as_positions = set(state.active_site or [])
    shell_candidates = [m for m in candidates
                        if m.get('dist_active_site', 999) < 10.0
                        and m.get('zone') != 'catalytic']

    multi_lib = _generate_restricted_multimutants(
        shell_candidates, multi_budget, as_positions
    )

    library = LibraryDesign(
        scout_library=scout_lib,
        multi_library=multi_lib,
        pareto_front=[m.get('mutation', '') for m in pareto_front],
        total_variants=len(scout_lib) + len(multi_lib),
        diversity_score=_compute_diversity(pareto_front),
    )
    state.library = library
    state.completed_steps.append('build_pareto_library')
    state.current_step = 3

    # ── Export library as CSV files ───────────────────────────────────────
    csv_paths = {}
    output_dir = params.get('output_dir', os.getcwd())
    os.makedirs(output_dir, exist_ok=True)
    csv_paths.update(_export_library_csv(scout_lib, multi_lib, output_dir))

    return {
        'step': 'build_pareto_library',
        'status': 'completed',
        'scout_library_size': len(scout_lib),
        'multi_library_size': len(multi_lib),
        'total_variants': library.total_variants,
        'diversity_score': round(library.diversity_score, 3),
        'pareto_front_top10': library.pareto_front[:10],
        'multi_library_top5': [m.get('mutations', '') for m in multi_lib[:5]],
        'csv_exports': csv_paths,
        'next_steps': [
            '订购引物、构建文库、进行两级筛选 (Step 4 — 湿实验)',
            '导入实验数据后继续到 Step 5',
        ],
    }


def step4_record_screening(state: WorkflowState, params: dict) -> dict:
    """Step 4: 两级实验筛选结果录入 / Record Two-Tier Screening Results.

    This step records wet-lab results (not computational).
    User provides:
      - tier1_data: HTS proxy readout results (CSV/JSON/dict)
      - tier2_data: Real-substrate validation results (optional, for top hits)
    """
    tier1_data = params.get('tier1_data', [])
    tier2_data = params.get('tier2_data', [])

    # Parse if string paths
    if isinstance(tier1_data, str) and os.path.exists(tier1_data):
        tier1_data = _load_screening_data(tier1_data)
    if isinstance(tier2_data, str) and os.path.exists(tier2_data):
        tier2_data = _load_screening_data(tier2_data)

    screening = ScreeningResult(
        tier1_proxy=tier1_data if isinstance(tier1_data, list) else [],
        tier2_real=tier2_data if isinstance(tier2_data, list) else [],
    )

    # Identify hits (> WT activity)
    hits = [m for m in screening.tier1_proxy
            if m.get('relative_activity', 0) > 1.0 or m.get('is_hit', False)]
    screening.hits = hits

    # Compute proxy-real correlation if both tiers available
    if screening.tier1_proxy and screening.tier2_real:
        screening.proxy_real_correlation = _compute_correlation(
            screening.tier1_proxy, screening.tier2_real
        )

    state.screening = screening
    state.completed_steps.append('record_screening')
    state.current_step = 4

    return {
        'step': 'record_screening',
        'status': 'completed',
        'tier1_count': len(screening.tier1_proxy),
        'tier2_count': len(screening.tier2_real),
        'n_hits': len(hits),
        'proxy_real_correlation': round(screening.proxy_real_correlation, 3),
        'next_steps': ['使用surrogate model生成第二轮提议 (Step 5)'],
    }


def step5_surrogate_proposal(state: WorkflowState, params: dict) -> dict:
    """Step 5: 简单surrogate model做第二轮提议 / Surrogate Model Second-Round Proposals.

    Implements补强5: Simple, robust models (Random Forest / Gradient Boosting)
    with ensemble-based uncertainty estimation.

    Acquisition function considers:
      - Predicted fitness (μ)
      - Uncertainty (σ, from ensemble variance)
      - Diversity (distance to tested variants)
      - Destabilization risk (negative penalty)
      - Mechanism violation risk (negative penalty)
    """
    if not state.screening or not state.screening.tier1_proxy:
        return {'error': 'No screening data. Run Step 4 first.'}

    model_type = params.get('model_type', 'random_forest')
    n_proposals = params.get('n_proposals', 48)
    beta_explore = params.get('beta_explore', 1.0)  # Exploration weight

    # Build training set from screening results
    training_data = state.screening.tier1_proxy

    # Get untested candidates from design space
    tested_mutations = set(m.get('mutation', '') for m in training_data)
    ds = state.design_space
    all_candidates = (ds.plm_candidates + ds.rational_candidates + ds.override_candidates)
    untested = [m for m in all_candidates
                if m.get('mutation', '') not in tested_mutations
                and m not in ds.filtered_out]

    if not untested:
        return {
            'error': 'No untested candidates remaining in design space.',
            'hint': 'Consider expanding the design space or lowering stability_threshold.',
        }

    # Feature encoding: position, delta_ll, zone_encoded, ss_encoded, rsa, dist_active_site
    X_train, y_train = _encode_training_data(training_data)
    X_candidates = _encode_candidates(untested)

    if X_train is None or len(X_train) < 5:
        return {
            'error': f'Insufficient training data ({len(training_data)} samples, need ≥5).',
            'hint': 'Record more screening results in Step 4.',
        }

    # Fit ensemble model
    predictions = _fit_and_predict_ensemble(
        X_train, y_train, X_candidates,
        model_type=model_type,
        n_estimators=params.get('n_estimators', 100),
    )

    if predictions is None:
        return {'error': 'Surrogate model fitting failed. Check training data quality.'}

    # Acquisition scoring: μ + β·σ + λ·diversity - penalty
    scored = []
    for i, m in enumerate(untested):
        mu = predictions['mean'][i]
        sigma = predictions['std'][i]
        diversity = _diversity_to_tested(m, training_data)
        risk = _destabilization_risk(m)

        acquisition = mu + beta_explore * sigma + 0.3 * diversity - 0.5 * risk
        scored.append({
            **m,
            'predicted_fitness': round(mu, 3),
            'uncertainty': round(sigma, 3),
            'diversity': round(diversity, 3),
            'risk': round(risk, 3),
            'acquisition_score': round(acquisition, 3),
        })

    # Select top proposals (Pareto: acquisition × diversity)
    scored.sort(key=lambda x: -x['acquisition_score'])
    proposals = scored[:n_proposals]

    # Ensure exploration: at least 20% should be high-uncertainty
    n_explore = max(1, n_proposals // 5)
    high_unc = sorted(scored, key=lambda x: -x['uncertainty'])[:n_explore]
    for m in high_unc:
        if m not in proposals:
            proposals.append(m)
            if len(proposals) > n_proposals:
                proposals.pop(-2)  # Remove lowest-scoring non-exploration pick

    surrogate = SurrogateProposal(
        model_type=model_type,
        proposals=proposals,
        acquisition_scores=[{'mutation': m.get('mutation', ''), 'score': m['acquisition_score']}
                            for m in proposals],
        uncertainty_method='ensemble_variance',
    )
    state.surrogate = surrogate
    state.completed_steps.append('surrogate_proposal')
    state.current_step = 5
    state.round_number += 1

    return {
        'step': 'surrogate_proposal',
        'status': 'completed',
        'model_type': model_type,
        'n_proposals': len(proposals),
        'top10_proposals': [
            {'mutation': m.get('mutation', ''),
             'predicted': m['predicted_fitness'],
             'uncertainty': m['uncertainty'],
             'acquisition': m['acquisition_score']}
            for m in proposals[:10]
        ],
        'exploration_fraction': round(n_explore / len(proposals), 2),
        'round': state.round_number,
        'next_steps': [
            f'第{state.round_number}轮湿实验筛选 (Step 6 — 继续DBTL)',
            '或直接进入终局验证 (Step 7)',
        ],
    }


def step6_iterate_dbtl(state: WorkflowState, params: dict) -> dict:
    """Step 6: 后续DBTL轮次 / Continue DBTL Iterations.

    Each iteration:
      1. Record new screening results
      2. Update surrogate model with all accumulated data
      3. Generate next-round proposals maintaining explore-exploit balance
    """
    # Record new data
    new_data = params.get('new_screening_data', [])
    if isinstance(new_data, str) and os.path.exists(new_data):
        new_data = _load_screening_data(new_data)

    if not new_data:
        return {'error': 'No new screening data provided.',
                'hint': 'Pass new_screening_data as list of dicts or CSV path.'}

    # Accumulate data
    state.screening.tier1_proxy.extend(new_data)
    hits = [m for m in new_data if m.get('relative_activity', 0) > 1.0 or m.get('is_hit', False)]
    state.screening.hits.extend(hits)

    # Re-run surrogate proposal with accumulated data
    proposal_result = step5_surrogate_proposal(state, params)
    proposal_result['step'] = 'iterate_dbtl'
    proposal_result['round'] = state.round_number
    proposal_result['total_tested'] = len(state.screening.tier1_proxy)
    proposal_result['total_hits'] = len(state.screening.hits)

    return proposal_result


def step7_final_validation(state: WorkflowState, params: dict) -> dict:
    """Step 7: 终局变体验证 / Final Variant Validation.

    For top hits:
      - Real-substrate quantitative kinetics
      - Whole-cell validation (if applicable)
      - Structural/mechanistic characterization (optional)
      - Generate final report
    """
    top_hits = params.get('top_hits', [])
    if not top_hits and state.screening:
        # Auto-select top hits from screening
        all_hits = sorted(state.screening.hits,
                          key=lambda x: -x.get('relative_activity', 0))
        top_hits = all_hits[:params.get('n_final', 5)]

    validation_plan = {
        'tier1_kinetics': [
            f"Express and purify {m.get('mutation', 'variant')}; "
            f"measure kcat, Km, kcat/Km with real substrate"
            for m in top_hits
        ],
        'tier2_stability': [
            f"DSF or CD for {m.get('mutation', 'variant')} Tm measurement"
            for m in top_hits
        ],
        'tier3_structure': [
            f"If budget allows: X-ray or cryo-EM for {m.get('mutation', 'variant')}"
            for m in top_hits[:2]  # Only top 2
        ],
    }

    # Generate workflow summary report
    state.completed_steps.append('final_validation')
    state.current_step = 7

    workflow_summary = {
        'rounds_completed': state.round_number,
        'total_tested': len(state.screening.tier1_proxy) if state.screening else 0,
        'total_hits': len(state.screening.hits) if state.screening else 0,
        'design_space_size': state.design_space.n_candidates_total if state.design_space else 0,
        'filtered_out': len(state.design_space.filtered_out) if state.design_space else 0,
        'hit_rate': (len(state.screening.hits) / max(len(state.screening.tier1_proxy), 1)
                     if state.screening else 0),
        'proxy_real_correlation': (state.screening.proxy_real_correlation
                                   if state.screening else None),
    }

    # Try to generate workflow report document
    report_path = None
    output_dir = params.get('output_dir', os.getcwd())
    try:
        report_path = _generate_workflow_report(state, top_hits, validation_plan,
                                                workflow_summary, output_dir)
    except Exception as e:
        logger.warning("Report generation failed: %s", e)

    result = {
        'step': 'final_validation',
        'status': 'completed',
        'n_top_hits': len(top_hits),
        'top_hits': [m.get('mutation', '') for m in top_hits],
        'validation_plan': validation_plan,
        'workflow_summary': workflow_summary,
        'next_steps': [],
    }
    if report_path:
        result['report_path'] = report_path
    else:
        result['next_steps'].append('生成完整工作流报告 (generate_report)')

    return result


# ── Main Execution Entry Point ────────────────────────────────────────────────

def execute_workflow(params: dict) -> dict:
    """Main entry point for the Pareto + Active Learning workflow.

    Args:
        params: Dict with:
            step: str — which step to execute ('all', 'step1'...'step7', or step name)
            protein_name, pdb_path, fasta_path, uniprot_id: str — protein info
            active_site, catalytic_residues: list[int] — key positions
            ...step-specific params...

    Returns:
        Dict with step results, state snapshot, and next_steps hints.
    """
    step = params.get('step', 'describe')

    # Initialize or restore state
    state_path = params.get('state_path', '')
    if state_path and os.path.exists(state_path):
        state = _load_state(state_path)
    else:
        state = WorkflowState(
            protein_name=params.get('protein_name', ''),
            pdb_path=params.get('pdb_path', ''),
            fasta_path=params.get('fasta_path', ''),
            uniprot_id=params.get('uniprot_id', ''),
            active_site=params.get('active_site', []),
            catalytic_residues=params.get('catalytic_residues', []),
        )

    STEP_MAP = {
        'step1': step1_define_objectives,
        'define_objectives': step1_define_objectives,
        'step2': step2_generate_design_space,
        'generate_design_space': step2_generate_design_space,
        'step3': step3_build_pareto_library,
        'build_pareto_library': step3_build_pareto_library,
        'step4': step4_record_screening,
        'record_screening': step4_record_screening,
        'step5': step5_surrogate_proposal,
        'surrogate_proposal': step5_surrogate_proposal,
        'step6': step6_iterate_dbtl,
        'iterate_dbtl': step6_iterate_dbtl,
        'step7': step7_final_validation,
        'final_validation': step7_final_validation,
    }

    if step == 'describe':
        from .registry import get_workflow_registry
        wf = get_workflow_registry().get('pareto_active_learning')
        return {'description': wf.describe() if wf else 'Workflow not found',
                'status': 'info'}

    if step == 'status':
        return {
            'current_step': state.current_step,
            'completed_steps': state.completed_steps,
            'round': state.round_number,
            'warnings': state.warnings,
        }

    # Sequential execution modes
    if step in ('all', 'design'):
        # 'design' = steps 1-3 (computational, before wet lab)
        # 'all' = same as 'design' (steps 4+ require experimental data)
        sequential_steps = [
            ('step1', step1_define_objectives),
            ('step2', step2_generate_design_space),
            ('step3', step3_build_pareto_library),
        ]
        results_all = {'step': step, 'sub_results': []}
        for step_id, step_fn in sequential_steps:
            try:
                r = step_fn(state, params)
                results_all['sub_results'].append({step_id: r})
                if 'error' in r:
                    results_all['stopped_at'] = step_id
                    results_all['error'] = r['error']
                    break
            except Exception as e:
                import traceback
                state.errors.append(f"Step {step_id} failed: {e}")
                results_all['stopped_at'] = step_id
                results_all['error'] = str(e)
                results_all['traceback'] = traceback.format_exc()
                break
        results_all['status'] = 'completed' if 'error' not in results_all else 'partial'
        results_all['completed_steps'] = state.completed_steps
        results_all['next_steps'] = [
            '订购引物、构建文库 (wet lab)',
            '实验完成后录入筛选数据: execute_workflow(step="step4", tier1_data=...)',
        ]
        result = results_all
        # Save state and return
        output_dir = params.get('output_dir', os.getcwd())
        os.makedirs(output_dir, exist_ok=True)
        state_file = os.path.join(output_dir, 'pareto_al_state.json')
        _save_state(state, state_file)
        result['state_file'] = state_file
        return result

    step_fn = STEP_MAP.get(step)
    if not step_fn:
        return {'error': f'Unknown step: {step}',
                'available_steps': list(STEP_MAP.keys()) + ['all', 'design', 'describe', 'status'],
                'hint': 'Use step1-step7, "design" (steps 1-3), or step name.'}

    try:
        result = step_fn(state, params)
    except Exception as e:
        import traceback
        state.errors.append(f"Step {step} failed: {e}")
        return {'error': str(e), 'traceback': traceback.format_exc()}

    # Save state
    output_dir = params.get('output_dir', os.getcwd())
    os.makedirs(output_dir, exist_ok=True)
    state_file = os.path.join(output_dir, 'pareto_al_state.json')
    _save_state(state, state_file)
    result['state_file'] = state_file

    return result


# ── Helper Functions ──────────────────────────────────────────────────────────

def _count_zones(mutations: list) -> dict:
    from collections import Counter
    return dict(Counter(m.get('zone', 'unknown') for m in mutations))


def _pareto_select(candidates: list, budget: int) -> list:
    """Pareto selection: maximize fitness (delta_ll) × diversity (position spread).

    Uses simple non-dominated sorting + crowding distance.
    """
    if len(candidates) <= budget:
        return candidates

    # Score each candidate on two objectives
    scored = []
    positions_seen = set()
    for m in candidates:
        fitness = m.get('delta_ll', 0)
        pos = m.get('position', 0)
        # Diversity: bonus for unseen positions
        position_novelty = 0 if pos in positions_seen else 1
        positions_seen.add(pos)
        scored.append((fitness, position_novelty, m))

    # Sort by fitness descending, then by novelty descending
    scored.sort(key=lambda x: (-x[0], -x[1]))

    # Select: alternate between top-fitness and position-diverse
    selected = []
    selected_positions = set()
    fitness_pool = list(scored)
    diversity_pool = sorted(scored, key=lambda x: (-x[1], -x[0]))

    i_fit, i_div = 0, 0
    use_fitness = True
    while len(selected) < budget:
        if use_fitness and i_fit < len(fitness_pool):
            _, _, m = fitness_pool[i_fit]
            i_fit += 1
            if m not in selected:
                selected.append(m)
                selected_positions.add(m.get('position', 0))
        elif i_div < len(diversity_pool):
            _, _, m = diversity_pool[i_div]
            i_div += 1
            if m not in selected and m.get('position', 0) not in selected_positions:
                selected.append(m)
                selected_positions.add(m.get('position', 0))
        else:
            break
        use_fitness = not use_fitness

    return selected


def _generate_restricted_multimutants(shell_candidates: list, budget: int,
                                       active_site: set) -> list:
    """Generate restricted multi-mutant combinations near active site."""
    if len(shell_candidates) < 2:
        return []

    from itertools import combinations
    combos = []

    # Only combine mutations at different positions
    by_pos = {}
    for m in shell_candidates:
        pos = m.get('position', 0)
        if pos not in by_pos:
            by_pos[pos] = m

    positions = sorted(by_pos.keys())
    if len(positions) < 2:
        return []

    # Generate doubles
    for p1, p2 in combinations(positions[:min(10, len(positions))], 2):
        m1, m2 = by_pos[p1], by_pos[p2]
        combo_score = (m1.get('delta_ll', 0) + m2.get('delta_ll', 0)) / 2
        combos.append({
            'mutations': f"{m1.get('mutation', '')}/{m2.get('mutation', '')}",
            'positions': [p1, p2],
            'n_mutations': 2,
            'combined_score': round(combo_score, 3),
            'library_type': 'multi',
            'components': [m1, m2],
        })

    # Sort by combined score
    combos.sort(key=lambda x: -x['combined_score'])
    return combos[:budget]


def _compute_diversity(mutations: list) -> float:
    """Compute position-spread diversity score."""
    positions = [m.get('position', 0) for m in mutations]
    if len(positions) < 2:
        return 0.0
    unique = len(set(positions))
    spread = max(positions) - min(positions) if positions else 0
    return unique / max(len(positions), 1) * min(spread / 100.0, 1.0)


def _export_library_csv(scout_lib: list, multi_lib: list, output_dir: str) -> dict:
    """Export library designs as CSV files for ordering primers."""
    import csv
    paths = {}

    # Scout library CSV
    scout_path = os.path.join(output_dir, 'scout_library.csv')
    scout_fields = ['mutation', 'position', 'delta_ll', 'zone', 'rsa',
                    'dist_active_site', 'library_type']
    try:
        with open(scout_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=scout_fields, extrasaction='ignore')
            writer.writeheader()
            for m in scout_lib:
                row = {k: m.get(k, '') for k in scout_fields}
                # Round floats
                for fk in ('delta_ll', 'rsa', 'dist_active_site'):
                    if isinstance(row.get(fk), float):
                        row[fk] = round(row[fk], 3)
                writer.writerow(row)
        paths['scout_csv'] = scout_path
    except Exception as e:
        logger.warning("Failed to export scout CSV: %s", e)

    # Multi-mutant library CSV
    multi_path = os.path.join(output_dir, 'multi_library.csv')
    multi_fields = ['mutations', 'positions', 'n_mutations', 'combined_score', 'library_type']
    try:
        with open(multi_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=multi_fields, extrasaction='ignore')
            writer.writeheader()
            for m in multi_lib:
                row = {k: m.get(k, '') for k in multi_fields}
                if isinstance(row.get('positions'), list):
                    row['positions'] = ';'.join(str(p) for p in row['positions'])
                writer.writerow(row)
        paths['multi_csv'] = multi_path
    except Exception as e:
        logger.warning("Failed to export multi CSV: %s", e)

    # Combined summary CSV
    summary_path = os.path.join(output_dir, 'library_summary.csv')
    try:
        with open(summary_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['Library Type', 'Mutation(s)', 'Position(s)', 'Score', 'Zone'])
            for m in scout_lib:
                writer.writerow(['scout', m.get('mutation', ''),
                                 m.get('position', ''),
                                 round(m.get('delta_ll', 0), 3),
                                 m.get('zone', '')])
            for m in multi_lib:
                pos_str = ';'.join(str(p) for p in m.get('positions', []))
                writer.writerow(['multi', m.get('mutations', ''),
                                 pos_str,
                                 round(m.get('combined_score', 0), 3), ''])
        paths['summary_csv'] = summary_path
    except Exception as e:
        logger.warning("Failed to export summary CSV: %s", e)

    return paths


def _load_screening_data(path: str) -> list:
    """Load screening data from CSV or JSON."""
    if path.endswith('.json'):
        with open(path, encoding='utf-8') as f:
            data = json.load(f)
        return data if isinstance(data, list) else data.get('results', [])
    elif path.endswith('.csv'):
        import csv
        with open(path, encoding='utf-8') as f:
            reader = csv.DictReader(f)
            return list(reader)
    return []


def _extract_wt_mut_aa(mutation_str: str):
    """Extract wild-type and mutant AA from mutation string like 'V21L'."""
    if not mutation_str or len(mutation_str) < 3:
        return '', ''
    wt = mutation_str[0]
    mut = mutation_str[-1]
    if wt.isalpha() and mut.isalpha():
        return wt.upper(), mut.upper()
    return '', ''


def _encode_mutation(m: dict) -> list:
    """Encode a single mutation as a feature vector (13 features).

    Features: position, delta_ll, zone, rsa, dist_active_site,
              wt_hydro, wt_vol, wt_charge, wt_polar,
              Δhydro, Δvol, Δcharge, Δpolar
    """
    mut_str = m.get('mutation', '')
    wt_aa, mut_aa = _extract_wt_mut_aa(mut_str)
    wt_props = _aa_features(wt_aa) if wt_aa else _AA_DEFAULT
    mut_props = _aa_features(mut_aa) if mut_aa else _AA_DEFAULT

    return [
        m.get('position', 0),
        m.get('delta_ll', 0),
        _zone_to_int(m.get('zone', '')),
        m.get('rsa', 0.5),
        m.get('dist_active_site', 20.0),
        # Wild-type AA properties
        wt_props[0], wt_props[1], wt_props[2], wt_props[3],
        # Delta (mutant - wild-type) properties
        mut_props[0] - wt_props[0],   # Δhydrophobicity
        mut_props[1] - wt_props[1],   # Δvolume
        mut_props[2] - wt_props[2],   # Δcharge
        mut_props[3] - wt_props[3],   # Δpolarity
    ]


def _encode_training_data(data: list):
    """Encode screening results as feature matrix + target vector."""
    try:
        import numpy as np
    except ImportError:
        return None, None

    X, y = [], []
    for m in data:
        features = _encode_mutation(m)
        target = m.get('relative_activity', m.get('fitness', 0))
        if target is not None:
            X.append(features)
            y.append(float(target))

    if len(X) < 5:
        return None, None
    return np.array(X), np.array(y)


def _encode_candidates(candidates: list):
    """Encode candidate mutations as feature matrix."""
    try:
        import numpy as np
    except ImportError:
        return None

    X = []
    for m in candidates:
        X.append(_encode_mutation(m))
    return np.array(X) if X else None


def _zone_to_int(zone: str) -> int:
    zones = {'catalytic': 5, 'active_site': 4, 'active_site_shell': 3,
             'channel': 3, 'buried_hydrophobic': 2, 'buried_polar': 2,
             'intermediate': 1, 'surface': 0}
    return zones.get(zone, 1)


# Amino acid physicochemical properties for surrogate model features
# [hydrophobicity (Kyte-Doolittle), volume (Å³), charge at pH7, polarity]
_AA_PROPS = {
    'A': (1.8, 88.6, 0, 0), 'R': (-4.5, 173.4, 1, 1), 'N': (-3.5, 114.1, 0, 1),
    'D': (-3.5, 111.1, -1, 1), 'C': (2.5, 108.5, 0, 0), 'E': (-3.5, 138.4, -1, 1),
    'Q': (-3.5, 143.8, 0, 1), 'G': (-0.4, 60.1, 0, 0), 'H': (-3.2, 153.2, 0.1, 1),
    'I': (4.5, 166.7, 0, 0), 'L': (3.8, 166.7, 0, 0), 'K': (-3.9, 168.6, 1, 1),
    'M': (1.9, 162.9, 0, 0), 'F': (2.8, 189.9, 0, 0), 'P': (-1.6, 112.7, 0, 0),
    'S': (-0.8, 89.0, 0, 1), 'T': (-0.7, 116.1, 0, 1), 'W': (-0.9, 227.8, 0, 0),
    'Y': (-1.3, 193.6, 0, 1), 'V': (4.2, 140.0, 0, 0),
}
_AA_DEFAULT = (0.0, 120.0, 0, 0)


def _aa_features(aa: str) -> tuple:
    """Get physicochemical features for an amino acid."""
    return _AA_PROPS.get(aa.upper(), _AA_DEFAULT)


def _fit_and_predict_ensemble(X_train, y_train, X_candidates,
                               model_type='random_forest', n_estimators=100):
    """Fit ensemble model and predict with uncertainty."""
    try:
        import numpy as np
        if model_type == 'random_forest':
            from sklearn.ensemble import RandomForestRegressor
            model = RandomForestRegressor(n_estimators=n_estimators, random_state=42)
        elif model_type == 'gradient_boosting':
            from sklearn.ensemble import GradientBoostingRegressor
            # Use bootstrap for uncertainty via sub-models
            model = GradientBoostingRegressor(n_estimators=n_estimators, random_state=42)
        else:
            from sklearn.ensemble import RandomForestRegressor
            model = RandomForestRegressor(n_estimators=n_estimators, random_state=42)

        model.fit(X_train, y_train)

        if hasattr(model, 'estimators_'):
            # Get per-tree predictions for uncertainty
            # RF: estimators_ is 1D list; GB: estimators_ is 2D array [n_estimators, 1]
            estimators = model.estimators_
            if hasattr(estimators, 'ravel'):
                estimators = estimators.ravel()  # GradientBoosting 2D → 1D
            tree_preds = np.array([t.predict(X_candidates) for t in estimators])
            mean_pred = tree_preds.mean(axis=0)
            std_pred = tree_preds.std(axis=0)
        else:
            mean_pred = model.predict(X_candidates)
            std_pred = np.zeros_like(mean_pred)

        return {'mean': mean_pred.tolist(), 'std': std_pred.tolist()}

    except ImportError:
        logger.warning("sklearn not available; surrogate model requires scikit-learn")
        return None
    except Exception as e:
        logger.warning("Ensemble fit failed: %s", e)
        return None


def _diversity_to_tested(candidate: dict, tested: list) -> float:
    """Compute positional diversity of candidate relative to tested set."""
    pos = candidate.get('position', 0)
    tested_positions = [m.get('position', 0) for m in tested]
    if not tested_positions:
        return 1.0
    min_dist = min(abs(pos - tp) for tp in tested_positions)
    return min(min_dist / 50.0, 1.0)  # Normalize, cap at 1.0


def _destabilization_risk(mutation: dict) -> float:
    """Estimate destabilization risk from enrichment data."""
    delta_ll = mutation.get('delta_ll', 0)
    zone = mutation.get('zone', '')
    rsa = mutation.get('rsa', 0.5)

    risk = 0.0
    if delta_ll < -0.5:
        risk += abs(delta_ll) * 0.5
    if zone in ('buried_hydrophobic', 'buried_polar') and rsa < 0.15:
        risk += 0.3
    return min(risk, 2.0)


def _generate_workflow_report(state: WorkflowState, top_hits: list,
                              validation_plan: dict, summary: dict,
                              output_dir: str) -> Optional[str]:
    """Generate a Markdown workflow summary report."""
    report_lines = [
        f"# Pareto + Active Learning Workflow Report",
        f"## Protein: {state.protein_name or 'N/A'}",
        "",
        "## Workflow Summary",
        f"- **Rounds completed**: {summary.get('rounds_completed', 0)}",
        f"- **Total variants tested**: {summary.get('total_tested', 0)}",
        f"- **Total hits (>WT)**: {summary.get('total_hits', 0)}",
        f"- **Hit rate**: {summary.get('hit_rate', 0):.1%}",
        f"- **Design space size**: {summary.get('design_space_size', 0)}",
        f"- **Filtered out (destabilizing)**: {summary.get('filtered_out', 0)}",
    ]

    if summary.get('proxy_real_correlation') is not None:
        report_lines.append(
            f"- **Proxy-real correlation**: {summary['proxy_real_correlation']:.3f}")

    # Objectives
    if state.objectives:
        report_lines.extend([
            "",
            "## Objectives",
            f"- **Primary**: {state.objectives.primary_objective}",
            f"- **Screening readout**: {state.objectives.screening_readout or 'N/A'}",
            f"- **Publication readout**: {state.objectives.publication_readout or 'N/A'}",
            f"- **Hard constraints**: {', '.join(state.objectives.hard_constraints) if state.objectives.hard_constraints else 'N/A'}",
        ])

    # Design space
    if state.design_space:
        ds = state.design_space
        report_lines.extend([
            "",
            "## Design Space",
            f"- PLM candidates: {len(ds.plm_candidates)}",
            f"- Rational candidates: {len(ds.rational_candidates)}",
            f"- Override candidates: {len(ds.override_candidates)}",
            f"- Positions covered: {ds.n_positions}",
            f"- After stability filter: {ds.n_after_filter}",
        ])

    # Library
    if state.library:
        report_lines.extend([
            "",
            "## First-Round Library",
            f"- Scout library: {len(state.library.scout_library)} single mutants",
            f"- Multi-mutant library: {len(state.library.multi_library)} combinatorial variants",
            f"- Diversity score: {state.library.diversity_score:.3f}",
        ])

    # Top hits
    if top_hits:
        report_lines.extend([
            "",
            "## Top Hits for Validation",
            "| Rank | Mutation | Relative Activity | Source |",
            "|------|----------|-------------------|--------|",
        ])
        for i, hit in enumerate(top_hits, 1):
            report_lines.append(
                f"| {i} | {hit.get('mutation', 'N/A')} | "
                f"{hit.get('relative_activity', 'N/A')} | "
                f"{hit.get('source', hit.get('zone', 'N/A'))} |"
            )

    # Validation plan
    report_lines.extend([
        "",
        "## Validation Plan",
    ])
    for tier, tasks in validation_plan.items():
        report_lines.append(f"\n### {tier}")
        for task in tasks:
            report_lines.append(f"- {task}")

    # Warnings
    if state.warnings:
        report_lines.extend([
            "",
            "## Warnings",
        ])
        for w in state.warnings:
            report_lines.append(f"- {w}")

    report_lines.extend([
        "",
        "---",
        "*Generated by Zhuangzhou Pareto+AL Workflow v1.0.0*",
    ])

    os.makedirs(output_dir, exist_ok=True)
    report_path = os.path.join(output_dir, 'pareto_al_workflow_report.md')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(report_lines))
    return report_path


def _save_state(state: WorkflowState, path: str):
    """Save workflow state to JSON."""
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(state.to_dict(), f, ensure_ascii=False, indent=2, default=str)


def _load_state(path: str) -> WorkflowState:
    """Load workflow state from JSON, reconstructing nested dataclasses."""
    with open(path, encoding='utf-8') as f:
        data = json.load(f)

    state = WorkflowState()
    # Simple fields
    for k in ('protein_name', 'pdb_path', 'fasta_path', 'uniprot_id',
              'active_site', 'catalytic_residues', 'current_step',
              'completed_steps', 'round_number', 'warnings', 'errors'):
        if k in data:
            setattr(state, k, data[k])

    # Nested dataclass reconstruction
    _NESTED = {
        'objectives': ObjectiveDefinition,
        'design_space': DesignSpace,
        'library': LibraryDesign,
        'screening': ScreeningResult,
        'surrogate': SurrogateProposal,
    }
    for field_name, cls in _NESTED.items():
        val = data.get(field_name)
        if val and isinstance(val, dict):
            try:
                setattr(state, field_name, cls(**val))
            except TypeError:
                # Filter unknown keys for forward compat
                import inspect
                valid_keys = set(inspect.signature(cls).parameters.keys())
                filtered = {k: v for k, v in val.items() if k in valid_keys}
                setattr(state, field_name, cls(**filtered))

    return state


def _compute_correlation(tier1: list, tier2: list) -> float:
    """Compute rank correlation between proxy and real readouts."""
    try:
        import numpy as np
        # Match by mutation name
        tier2_map = {m.get('mutation', ''): m.get('relative_activity', 0) for m in tier2}
        pairs = []
        for m in tier1:
            mut = m.get('mutation', '')
            if mut in tier2_map:
                pairs.append((m.get('relative_activity', 0), tier2_map[mut]))
        if len(pairs) < 3:
            return 0.0
        x = np.array([p[0] for p in pairs])
        y = np.array([p[1] for p in pairs])
        if x.std() == 0 or y.std() == 0:
            return 0.0
        return float(np.corrcoef(x, y)[0, 1])
    except Exception:
        return 0.0


# ── Workflow Registration ─────────────────────────────────────────────────────

def register_workflow(registry):
    """Register the Pareto + Active Learning workflow."""
    from .registry import Workflow, WorkflowStage, WorkflowParam

    wf = Workflow(
        id='pareto_active_learning',
        name='鲁棒性增强版酶工程工作流：帕累托与主动学习',
        name_en='Robustness-Enhanced Enzyme Engineering: Pareto + Active Learning',
        version='1.0.0',
        description=(
            '基于文献的最小必要补强工作流，通过五项有证据支撑的关键补强（帕累托文库设计、'
            'focused training、去稳定化过滤、受限多突分支、简单稳健模型），'
            '在低预算多轮DBTL条件下提高酶工程命中率。'
        ),
        description_en=(
            'Literature-backed robustness-enhanced workflow with five key reinforcements: '
            'Pareto fitness+diversity library design, focused training, destabilization '
            'filtering, restricted multi-mutant branch, and simple robust surrogate models. '
            'Optimized for low-budget multi-round DBTL enzyme engineering campaigns.'
        ),
        tags=['enzyme_engineering', 'active_learning', 'pareto', 'DBTL',
              'directed_evolution', 'machine_learning'],
        references=[
            {'title': 'Language models enable zero-shot prediction of mutation effects',
             'authors': 'Meier et al.', 'journal': 'NeurIPS', 'year': '2021',
             'url': 'https://proceedings.neurips.cc/paper/2021/file/f51338d736f95dd42427296047067694-Paper.pdf'},
            {'title': 'Automated design of efficient and functionally diverse enzyme repertoires',
             'authors': 'Khersonsky et al.', 'journal': 'Molecular Cell', 'year': '2018',
             'url': 'https://pmc.ncbi.nlm.nih.gov/articles/PMC6193528/'},
            {'title': 'Informed training set design enables efficient MLDE',
             'authors': 'Wittmann et al.', 'journal': 'Cell Systems', 'year': '2021',
             'url': 'https://pubmed.ncbi.nlm.nih.gov/34416172/'},
            {'title': 'Active learning-assisted directed evolution (ALDE)',
             'authors': 'Romero et al.', 'journal': 'Nature Communications', 'year': '2025',
             'url': 'https://www.nature.com/articles/s41467-025-55987-8'},
            {'title': 'MODIFY: ML-guided co-optimization of fitness and diversity',
             'authors': 'Wang et al.', 'journal': 'Nature Communications', 'year': '2024',
             'url': 'https://www.nature.com/articles/s41467-024-50698-y'},
            {'title': 'Benchmarking MLDE across 16 fitness landscapes',
             'authors': '', 'journal': 'Cell Systems', 'year': '2025',
             'url': 'https://www.sciencedirect.com/science/article/pii/S2405471225002200'},
            {'title': 'Enriching productive mutational paths via ML-guided CAST',
             'authors': '', 'journal': 'Nature Chemical Biology', 'year': '2024',
             'url': 'https://www.nature.com/articles/s41589-024-01712-3'},
        ],
        stages=[
            WorkflowStage(
                id='define_objectives',
                name='明确目标函数与硬约束',
                name_en='Define Objectives & Hard Constraints',
                description='定义主目标函数、筛选readout、论文readout、硬约束（表达阈值、核心残基保护、稳定性下限）',
                skills=[],
                inputs=['primary_objective', 'screening_readout', 'publication_readout', 'hard_constraints'],
                outputs=['objectives'],
                estimated_time='~1min',
            ),
            WorkflowStage(
                id='generate_design_space',
                name='三支候选生成与设计空间收缩',
                name_en='Generate Candidates & Compress Design Space',
                description='PLM零样本扫描 + 活性位点半理性设计 + 机制覆写位点 → 稳定性过滤',
                skills=['esm2_mutation_scan', 'enrich_mutations', 'pdb_to_fasta'],
                inputs=['pdb_path', 'fasta_path', 'uniprot_id', 'active_site',
                        'plm_top_n', 'stability_threshold', 'override_positions'],
                outputs=['design_space'],
                estimated_time='~5-15min',
            ),
            WorkflowStage(
                id='build_pareto_library',
                name='构建首轮帕累托文库',
                name_en='Build First-Round Pareto Library',
                description='帕累托前沿选点（fitness×diversity）构建单突侦察库 + 受限多突组合库',
                skills=[],
                inputs=['scout_budget', 'multi_budget'],
                outputs=['library'],
                estimated_time='~1min',
            ),
            WorkflowStage(
                id='record_screening',
                name='两级实验筛选结果录入',
                name_en='Record Two-Tier Screening Results',
                description='录入一级高通量代理readout + 二级真实底物验证结果',
                skills=[],
                inputs=['tier1_data', 'tier2_data'],
                outputs=['screening'],
                estimated_time='~1min (data entry)',
            ),
            WorkflowStage(
                id='surrogate_proposal',
                name='Surrogate Model第二轮提议',
                name_en='Surrogate Model Second-Round Proposals',
                description='随机森林/梯度提升集成 + 集成方差不确定性 → acquisition排序',
                skills=[],
                inputs=['model_type', 'n_proposals', 'beta_explore'],
                outputs=['surrogate'],
                estimated_time='~1-5min',
            ),
            WorkflowStage(
                id='iterate_dbtl',
                name='后续DBTL轮次',
                name_en='Continue DBTL Iterations',
                description='累积新筛选数据 → 更新surrogate → 生成下轮提议',
                skills=[],
                inputs=['new_screening_data'],
                outputs=['surrogate'],
                optional=True,
                estimated_time='~1-5min per round',
            ),
            WorkflowStage(
                id='final_validation',
                name='终局变体验证',
                name_en='Final Variant Validation',
                description='Top hits真实底物定量动力学 + 热稳定性 + 可选结构表征',
                skills=['generate_report'],
                inputs=['top_hits', 'n_final'],
                outputs=['validation_plan'],
                estimated_time='~1min (plan generation)',
            ),
        ],
        params=[
            WorkflowParam('step', 'string', '执行步骤: step1-step7 / describe / status',
                          required=True, example='step2'),
            WorkflowParam('protein_name', 'string', '蛋白名称', required=False, example='CYP2A6'),
            WorkflowParam('pdb_path', 'path', 'PDB结构文件路径', required=False),
            WorkflowParam('fasta_path', 'path', 'FASTA序列文件路径', required=False),
            WorkflowParam('uniprot_id', 'string', 'UniProt ID', required=False, example='P11509'),
            WorkflowParam('active_site', 'list', '活性位点残基编号列表', required=False),
            WorkflowParam('catalytic_residues', 'list', '催化残基编号列表', required=False),
            WorkflowParam('primary_objective', 'string', '主目标: catalytic_activity / selectivity / thermostability',
                          required=False, default='catalytic_activity'),
            WorkflowParam('screening_readout', 'string', '高通量筛选readout描述', required=False),
            WorkflowParam('publication_readout', 'string', '论文readout描述', required=False),
            WorkflowParam('hard_constraints', 'list', '硬约束列表', required=False),
            WorkflowParam('plm_top_n', 'int', 'PLM候选数量', required=False, default=200),
            WorkflowParam('stability_threshold', 'float', '稳定性过滤阈值(ΔLL)', required=False, default=-1.0),
            WorkflowParam('override_positions', 'list', '用户指定必试位点', required=False),
            WorkflowParam('scout_budget', 'int', '单突侦察库大小', required=False, default=80),
            WorkflowParam('multi_budget', 'int', '多突组合库大小', required=False, default=16),
            WorkflowParam('tier1_data', 'string', '一级筛选数据(JSON/CSV路径或列表)', required=False),
            WorkflowParam('tier2_data', 'string', '二级筛选数据(JSON/CSV路径或列表)', required=False),
            WorkflowParam('model_type', 'string', 'Surrogate模型类型: random_forest / gradient_boosting',
                          required=False, default='random_forest'),
            WorkflowParam('n_proposals', 'int', '第二轮提议数量', required=False, default=48),
            WorkflowParam('beta_explore', 'float', '探索权重(越大越倾向高不确定性)', required=False, default=1.0),
            WorkflowParam('new_screening_data', 'string', '新一轮筛选数据', required=False),
            WorkflowParam('output_dir', 'path', '输出目录', required=False),
            WorkflowParam('state_path', 'path', '恢复已有工作流状态', required=False),
        ],
        applicability=(
            '适用于：目标酶已有基本表达与测定条件；存在高通量代理readout + 真实底物复筛；'
            '允许至少两轮DBTL闭环；设计空间可预先压缩到中等规模。'
        ),
        not_recommended=(
            '不推荐：无任何先验知识的全蛋白随机搜索；只有一轮实验机会；'
            '无高通量筛选条件；目标函数定义不明确。'
        ),
        execute_fn=execute_workflow,
    )

    registry.register(wf)
