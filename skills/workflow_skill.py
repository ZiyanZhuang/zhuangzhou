"""Workflow Skills — Expose optional multi-step workflows as NLP-callable skills.

Bridges workflow_options/ modules to the skill registry so the NLP agent
can discover and execute workflows like any other skill.
"""

import logging
import os
from .registry import Skill, SkillParam, SkillRegistry

logger = logging.getLogger(__name__)


# ── Pareto + Active Learning Workflow ────────────────────────────────────────

def pareto_al_workflow(step: str = 'describe', protein_name: str = '',
                       pdb_path: str = '', fasta_path: str = '',
                       uniprot_id: str = '', active_site: list = None,
                       catalytic_residues: list = None,
                       primary_objective: str = 'catalytic_activity',
                       screening_readout: str = '', publication_readout: str = '',
                       hard_constraints: list = None,
                       plm_top_n: int = 200, stability_threshold: float = -1.0,
                       override_positions: list = None,
                       scout_budget: int = 80, multi_budget: int = 16,
                       tier1_data=None, tier2_data=None,
                       model_type: str = 'random_forest',
                       n_proposals: int = 48, beta_explore: float = 1.0,
                       new_screening_data=None,
                       output_dir: str = None, state_path: str = '',
                       top_hits: list = None, n_final: int = 5,
                       **kw) -> dict:
    """Execute the Pareto + Active Learning enzyme engineering workflow.

    A literature-backed robustness-enhanced workflow with 7 steps:
      Step 1: Define objectives & constraints
      Step 2: Generate candidates (PLM + rational + override) & compress space
      Step 3: Build Pareto-selected library (scout + multi-mutant)
      Step 4: Record screening results (wet lab data entry)
      Step 5: Surrogate model second-round proposals
      Step 6: Continue DBTL iterations
      Step 7: Final variant validation & report

    Use step='describe' for overview, step='design' for steps 1-3, or step1-step7.
    """
    from zhuangzhou.workflow_options.pareto_active_learning import execute_workflow

    params = {
        'step': step,
        'protein_name': protein_name,
        'pdb_path': pdb_path,
        'fasta_path': fasta_path,
        'uniprot_id': uniprot_id,
        'active_site': active_site or [],
        'catalytic_residues': catalytic_residues or [],
        'primary_objective': primary_objective,
        'screening_readout': screening_readout,
        'publication_readout': publication_readout,
        'hard_constraints': hard_constraints,
        'plm_top_n': plm_top_n,
        'stability_threshold': stability_threshold,
        'override_positions': override_positions,
        'scout_budget': scout_budget,
        'multi_budget': multi_budget,
        'tier1_data': tier1_data,
        'tier2_data': tier2_data,
        'model_type': model_type,
        'n_proposals': n_proposals,
        'beta_explore': beta_explore,
        'new_screening_data': new_screening_data,
        'output_dir': output_dir,
        'state_path': state_path,
        'top_hits': top_hits,
        'n_final': n_final,
    }
    # Remove None values so execute_workflow uses its own defaults
    params = {k: v for k, v in params.items() if v is not None}

    return execute_workflow(params)


def pareto_al_harness(step: str = 'status', output_dir: str = None,
                      state_path: str = '', reset_step: str = '',
                      **params) -> dict:
    """Harness-managed execution of Pareto+AL workflow with checkpoints and recovery.

    The harness adds:
      - Precondition validation before each step
      - Automatic state persistence (survives session restart)
      - Retry on failure (max 2 retries per step)
      - Clear recovery options on failure
      - Resume from any checkpoint

    Use step='status' to check progress, step='design' to run steps 1-3.
    """
    from zhuangzhou.workflow_options.harness import WorkflowHarness

    # Resume from existing state
    if state_path and os.path.exists(state_path):
        harness = WorkflowHarness.resume(state_path)
    else:
        harness = WorkflowHarness(
            workflow_id='pareto_active_learning',
            output_dir=output_dir or os.getcwd(),
        )

    if step == 'status':
        return harness.status()

    if reset_step:
        return harness.reset_step(reset_step)

    # Validate only (dry-run)
    if step == 'validate':
        target = params.get('validate_step', 'step1')
        return harness.validate(target, params)

    return harness.run(step, params)


def list_workflows(**kw) -> dict:
    """List all available optional workflows with descriptions."""
    from zhuangzhou.workflow_options import get_workflow_registry
    registry = get_workflow_registry()
    workflows = registry.list_all()
    return {
        'workflows': [
            {
                'id': wf.id,
                'name': wf.name,
                'name_en': wf.name_en,
                'version': wf.version,
                'n_stages': len(wf.stages),
                'tags': wf.tags,
                'description': wf.description_en[:300],
                'applicability': wf.applicability,
            }
            for wf in workflows
        ],
        'total': len(workflows),
    }


# ── Registration ─────────────────────────────────────────────────────────────

def register_skills(registry: SkillRegistry):
    """Register workflow skills."""

    registry.register(Skill(
        name='pareto_al_workflow',
        description=(
            'Robustness-enhanced enzyme engineering workflow: '
            'Pareto fitness+diversity library design + active learning DBTL. '
            '7-step pipeline from objective definition through final validation. '
            'Use step="describe" for overview, step="design" for computational steps 1-3.'
        ),
        category='workflow',
        fn=pareto_al_workflow,
        inputs=[
            SkillParam('step', 'string',
                       'Workflow step: describe / design / status / step1-step7',
                       required=True, default='describe',
                       choices=['describe', 'design', 'status', 'all',
                                'step1', 'step2', 'step3', 'step4',
                                'step5', 'step6', 'step7',
                                'define_objectives', 'generate_design_space',
                                'build_pareto_library', 'record_screening',
                                'surrogate_proposal', 'iterate_dbtl',
                                'final_validation']),
            SkillParam('protein_name', 'string', 'Protein name (e.g. CYP2A6)', required=False),
            SkillParam('pdb_path', 'path', 'PDB structure file path', required=False),
            SkillParam('fasta_path', 'path', 'FASTA sequence file path (auto-generated from PDB if missing)',
                       required=False),
            SkillParam('uniprot_id', 'string', 'UniProt accession ID', required=False),
            SkillParam('active_site', 'list', 'Active site residue numbers', required=False),
            SkillParam('catalytic_residues', 'list', 'Catalytic residue numbers (protected)', required=False),
            SkillParam('primary_objective', 'string',
                       'Primary objective: catalytic_activity / selectivity / thermostability',
                       required=False, default='catalytic_activity'),
            SkillParam('screening_readout', 'string', 'HTS proxy readout description', required=False),
            SkillParam('publication_readout', 'string', 'Real-substrate readout for paper', required=False),
            SkillParam('scout_budget', 'int', 'Single-mutant scout library size', required=False, default=80),
            SkillParam('multi_budget', 'int', 'Multi-mutant library size', required=False, default=16),
            SkillParam('stability_threshold', 'float', 'Stability filter threshold (ΔLL)',
                       required=False, default=-1.0),
            SkillParam('override_positions', 'list', 'User-specified must-try positions', required=False),
            SkillParam('tier1_data', 'string', 'Tier-1 screening data (JSON/CSV path or list)', required=False),
            SkillParam('tier2_data', 'string', 'Tier-2 real-substrate data (JSON/CSV path)', required=False),
            SkillParam('model_type', 'string', 'Surrogate model: random_forest / gradient_boosting',
                       required=False, default='random_forest'),
            SkillParam('n_proposals', 'int', 'Number of second-round proposals', required=False, default=48),
            SkillParam('beta_explore', 'float', 'Exploration weight (higher = more uncertain picks)',
                       required=False, default=1.0),
            SkillParam('new_screening_data', 'string', 'New round screening data for DBTL iteration',
                       required=False),
            SkillParam('output_dir', 'path', 'Output directory', required=False, is_output_path=True),
            SkillParam('state_path', 'path', 'Restore workflow state from JSON', required=False),
        ],
        outputs=[
            SkillParam('status', 'string', 'Step completion status'),
            SkillParam('state_file', 'path', 'Saved workflow state JSON path'),
            SkillParam('csv_exports', 'dict', 'Exported CSV file paths (scout, multi, summary)'),
        ],
        source='zhuangzhou/workflow_options/pareto_active_learning.py',
        estimated_time='~1-15min per step',
        output_type='analysis',
        when_to_use=(
            'When user wants to run a complete enzyme engineering campaign with '
            'Pareto library design, active learning, and DBTL iterations. '
            'Also when user mentions "帕累托", "主动学习", "DBTL", "工作流", '
            '"Pareto", "active learning", "workflow", or "robustness enhanced".'
        ),
        when_not_to_use=(
            'For single mutation analysis (use esm2_mutation_scan). '
            'For one-time docking or MD (use dedicated skills). '
            'When no iterative experiment is planned.'
        ),
        workflow_stage='compute',
    ))

    registry.register(Skill(
        name='pareto_al_harness',
        description=(
            'Harness-managed Pareto+AL workflow with checkpoints, precondition validation, '
            'retry on failure, and session resumption. Use for production runs. '
            'step="status" for progress, step="design" for steps 1-3, step="validate" for dry-run.'
        ),
        category='workflow',
        fn=pareto_al_harness,
        inputs=[
            SkillParam('step', 'string',
                       'Step to run: status / validate / design / step1-step7',
                       required=True, default='status'),
            SkillParam('state_path', 'path', 'Resume from saved harness state JSON', required=False),
            SkillParam('output_dir', 'path', 'Output directory', required=False, is_output_path=True),
            SkillParam('reset_step', 'string', 'Reset a failed step to pending', required=False),
            SkillParam('protein_name', 'string', 'Protein name', required=False),
            SkillParam('pdb_path', 'path', 'PDB structure file path', required=False),
            SkillParam('fasta_path', 'path', 'FASTA sequence file path', required=False),
        ],
        outputs=[
            SkillParam('progress', 'string', 'N/M steps completed'),
            SkillParam('state_path', 'path', 'Saved harness state for resumption'),
        ],
        source='zhuangzhou/workflow_options/harness.py',
        estimated_time='~1-15min',
        output_type='analysis',
        when_to_use=(
            'When user wants robust, checkpoint-protected workflow execution. '
            'Preferred over pareto_al_workflow for production enzyme engineering campaigns. '
            'Also when resuming interrupted workflows or checking workflow status.'
        ),
    ))

    registry.register(Skill(
        name='list_workflows',
        description='List all available optional multi-step workflows with descriptions',
        category='workflow',
        fn=list_workflows,
        inputs=[],
        outputs=[SkillParam('workflows', 'list', 'Available workflow metadata')],
        source='zhuangzhou/workflow_options/registry.py',
        estimated_time='<1s',
        output_type='analysis',
        when_to_use='When user asks what workflows are available or wants to browse workflow options',
    ))
