"""
Skill Registry
===============
Central registry for all computational skills.
Skills are discovered from:
  1. Built-in skill modules (plm, structure, docking, etc.)
  2. Uni-Mol-main/scripts/ (auto-wrapped)
  3. External plugins (future)

Each skill has a manifest: name, category, inputs, outputs, requirements.
"""

import importlib
import logging
import os
import sys
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Set

logger = logging.getLogger(__name__)

# Singleton
_registry_instance = None


@dataclass
class SkillParam:
    """A skill input/output parameter."""
    name: str
    type: str  # 'path', 'string', 'int', 'float', 'list', 'set', 'dict', 'bool'
    description: str = ""
    required: bool = True
    default: Any = None
    choices: Optional[List[str]] = None  # enum constraint for LLM + runtime validation
    is_output_path: bool = False  # True = this path is an output dir/file, skip exists() check


@dataclass
class Skill:
    """A registered computational skill."""
    name: str
    description: str
    category: str  # plm, structure, docking, simulation, mutagenesis, chemistry, visualization, brain
    fn: Callable
    inputs: List[SkillParam] = field(default_factory=list)
    outputs: List[SkillParam] = field(default_factory=list)
    requires_gpu: bool = False
    requires_wsl: bool = False
    requires_network: bool = False
    remote_capable: bool = False  # can be offloaded to SSH compute node
    remote_preferred: bool = False  # should prefer remote when available
    remote_capability: str = ''  # SSH host capability tag (e.g. 'gromacs', 'qm', 'gpu')
    source: str = ""  # file path or module
    available: Optional[bool] = None  # cached availability
    risk_level: str = 'low'  # 'low' | 'medium' | 'high'
    estimated_time: str = ''  # e.g. '~5s', '~30min'
    depends_on: List[str] = field(default_factory=list)  # skills that must run before this one
    output_type: str = ''  # semantic output type: 'pdb', 'sdf', 'report', 'analysis', 'trajectory'

    # Parallel execution: None = auto-detect from category (see nlp/parallel.py)
    # True = safe to run concurrently with other safe skills (read-only, no shared file I/O)
    # False = must run serially (writes files, uses shared resources, long-running)
    concurrent_safe: Optional[bool] = None

    # Planning & routing metadata
    when_to_use: str = ""          # Conditions that trigger this skill
    when_not_to_use: str = ""      # Anti-patterns / when to avoid
    fallback_skills: List[str] = field(default_factory=list)  # alternatives if this fails
    input_from_skills: List[str] = field(default_factory=list)  # typical upstream skill names
    workflow_stage: str = ""       # fetch|prepare|compute|analyze|visualize|report

    def preflight(self, **kwargs) -> Optional[dict]:
        """Pre-execution validation (dry run). Returns error dict or None if OK.

        Checks:
        1. Required params present
        2. Path params: file/dir exists
        3. Dependency outputs: referenced files accessible
        4. Numeric ranges: within reasonable bounds
        """
        # Required params
        missing = [p.name for p in self.inputs
                   if p.required and p.name not in kwargs]
        if missing:
            return {'error': f'Missing required parameters: {missing}',
                    'hint': {p.name: f'{p.type} — {p.description}'
                             for p in self.inputs if p.name in missing}}

        # Path validation
        for p in self.inputs:
            if p.type != 'path' or p.name not in kwargs:
                continue
            val = kwargs[p.name]
            if val is None:
                continue
            # Skip existence check for output paths (directories/files to be created)
            if p.is_output_path or p.name in ('output_dir', 'output_path', 'work_dir_new'):
                continue
            if isinstance(val, str) and val.strip():
                if not os.path.exists(val):
                    try:
                        avail = [f for f in os.listdir(".") if f.endswith((".pdb",".sdf",".mol2",".gro",".xtc"))][:10]
                    except OSError:
                        avail = []
                    return {'error': f'File not found: "{val}" (parameter: {p.name})',
                            'hint': f'Check the path. CWD={os.getcwd()}. Available: {avail}'}

        # Choices validation (early) — case-insensitive to tolerate LLM output variance
        for p in self.inputs:
            if p.choices and p.name in kwargs:
                val = kwargs[p.name]
                if val is None:
                    continue
                choices_lower = [str(c).lower() for c in p.choices]
                if str(val).lower() not in choices_lower:
                    return {'error': f'Parameter "{p.name}" must be one of {p.choices}, got "{val}"'}
                # Normalize to canonical case
                idx = choices_lower.index(str(val).lower())
                kwargs[p.name] = p.choices[idx]

        return None  # All checks passed

    def check_available(self, config: dict = None) -> bool:
        """Check if this skill can run in current environment."""
        import shutil
        try:
            if self.requires_gpu:
                import torch
                if not torch.cuda.is_available():
                    self.available = False
                    return False

            # Check category-specific binaries
            cat = self.category
            if cat == 'simulation' or 'gromacs' in self.name:
                if not shutil.which('gmx') and not shutil.which('gmx_mpi'):
                    self.available = False
                    return False
            if self.requires_wsl:
                if not shutil.which('wsl'):
                    self.available = False
                    return False
            if 'docking' in self.name or cat == 'docking':
                # Vina — check via config or PATH
                cfg = config or {}
                vina_path = cfg.get('vina_path', '')
                if not vina_path and not shutil.which('vina'):
                    # Not a hard fail — SmartDocking has its own path detection
                    pass
            self.available = True
            return True
        except Exception:
            self.available = False
            return False

    def execute(self, **kwargs) -> Any:
        """Execute the skill with validated parameters."""
        # 0. Inject defaults for missing optional params (before preflight)
        for p in self.inputs:
            if p.name not in kwargs and p.default is not None:
                kwargs[p.name] = p.default

        # 1. Pre-flight validation (catches missing params + bad paths before wasting compute)
        preflight_err = self.preflight(**kwargs)
        if preflight_err:
            return preflight_err

        # 2. Type coercion + validation
        for p in self.inputs:
            if p.name not in kwargs:
                continue
            val = kwargs[p.name]
            if val is None:
                continue

            # path must be string
            if p.type == 'path' and not isinstance(val, str):
                return {'error': f'Parameter "{p.name}" must be a file path string, got {type(val).__name__}'}

            # list/set: auto-split comma-separated strings
            if p.type in ('list', 'set') and isinstance(val, str):
                split_val = [x.strip() for x in val.split(',') if x.strip()]
                if not split_val and p.required:
                    return {'error': f'Parameter "{p.name}" requires a non-empty list',
                            'hint': p.description}
                kwargs[p.name] = split_val

            # int coercion
            if p.type == 'int' and not isinstance(val, int):
                try:
                    kwargs[p.name] = int(val)
                except (ValueError, TypeError):
                    return {'error': f'Parameter "{p.name}" must be integer, got "{val}"'}

            # float coercion
            if p.type == 'float' and not isinstance(val, (int, float)):
                try:
                    kwargs[p.name] = float(val)
                except (ValueError, TypeError):
                    return {'error': f'Parameter "{p.name}" must be number, got "{val}"'}

            # bool coercion
            if p.type == 'bool' and not isinstance(val, bool):
                if isinstance(val, str):
                    if val.lower() in ('true', '1', 'yes'):
                        kwargs[p.name] = True
                    elif val.lower() in ('false', '0', 'no'):
                        kwargs[p.name] = False
                    # else leave as string
                elif isinstance(val, (int, float)):
                    kwargs[p.name] = bool(val)

        return self.fn(**kwargs)


class SkillRegistry:
    """Central registry of all available skills."""

    def __init__(self, config: dict = None):
        self.config = config or {}
        self._skills: Dict[str, Skill] = {}
        self._categories: Dict[str, List[str]] = {}
        self._aliases: Dict[str, Skill] = {}  # Old skill names -> original Skill objects
        self._loaded = False

    def register(self, skill: Skill):
        """Register a skill."""
        self._skills[skill.name] = skill
        cat = skill.category
        if cat not in self._categories:
            self._categories[cat] = []
        if skill.name not in self._categories[cat]:
            self._categories[cat].append(skill.name)

    def get(self, name: str) -> Optional[Skill]:
        """Get a skill by name. Falls back to legacy aliases for backward compatibility."""
        if not self._loaded:
            self.load_all()
        skill = self._skills.get(name)
        if skill is None:
            skill = self._aliases.get(name)
        return skill

    def list_all(self) -> List[Skill]:
        """List all registered skills."""
        if not self._loaded:
            self.load_all()
        return list(self._skills.values())

    def list(self) -> List[Skill]:
        """Alias for list_all() for convenience."""
        return self.list_all()

    def list_by_category(self, category: str) -> List[Skill]:
        """List skills in a category."""
        if not self._loaded:
            self.load_all()
        names = self._categories.get(category, [])
        return [self._skills[n] for n in names if n in self._skills]

    def categories(self) -> List[str]:
        """List all categories."""
        if not self._loaded:
            self.load_all()
        return list(self._categories.keys())

    def check_all(self) -> Dict[str, bool]:
        """Check availability of all skills."""
        if not self._loaded:
            self.load_all()
        result = {}
        for name, skill in self._skills.items():
            skill.check_available(self.config)
            result[name] = skill.available
        return result

    def load_all(self):
        """Load all skill modules."""
        if self._loaded:
            return
        self._loaded = True

        # Load built-in skill modules
        from . import plm, structure, docking, simulation, mutagenesis, chemistry, visualization
        for mod in [plm, structure, docking, simulation, mutagenesis, chemistry, visualization]:
            if hasattr(mod, 'register_skills'):
                mod.register_skills(self)

        # Load UniMol deep learning skills (sandboxed in dplm2 env)
        try:
            from . import unimol
            unimol.register_skills(self)
        except ImportError as e:
            logger.debug("Skipped unimol skills: %s", e)

        # Load brain skills if available
        try:
            from . import brain_skills
            brain_skills.register_skills(self)
        except ImportError as e:
            logger.debug("Skipped brain_skills: %s", e)

        # Load LigBuilder V3 de novo design skills
        try:
            from . import ligbuilder
            ligbuilder.register_skills(self)
        except ImportError as e:
            logger.debug("Skipped ligbuilder skills: %s", e)

        # Load QM Engine skills (PySCF + geomeTRIC via WSL)
        try:
            from . import qm
            qm.register_skills(self)
        except ImportError as e:
            logger.debug("Skipped qm skills: %s", e)

        # Enzyme Specificity GNN skills DISABLED:
        # Model trained exclusively with protein 3D structures (PDB/AlphaFold pocket),
        # cannot work with sequence+SMILES only input. See benchmark: 1/14 TP.
        # Requires retraining in ligand-only mode to be usable without structures.

        # Load Non-Standard Residue parameterization skills
        try:
            from . import nonstandard
            nonstandard.register_skills(self)
        except ImportError as e:
            logger.debug("Skipped nonstandard skills: %s", e)

        # Load Codon Optimization skills (K. phaffii / Pichia pastoris)
        try:
            from . import codon_opt
            codon_opt.register_skills(self)
        except ImportError as e:
            logger.debug("Skipped codon_opt skills: %s", e)

        # Load Research skills (PubMed, UniProt, GraphRAG, mutation evidence)
        try:
            from . import research
            research.register_skills(self)
        except ImportError as e:
            logger.debug("Skipped research skills: %s", e)

        # Load BioDBAPI skills (NCBI, InterPro, PDB, AlphaFold, enhanced UniProt)
        try:
            from . import biodbapi
            biodbapi.register_skills(self)
        except ImportError as e:
            logger.debug("Skipped biodbapi skills: %s", e)

        # Load Report generation skills (Word .docx)
        try:
            from . import report
            report.register_skills(self)
        except ImportError as e:
            logger.debug("Skipped report skills: %s", e)

        # Load MetaKG metabolic knowledge graph skills
        try:
            from . import metakg
            metakg.register(self)
        except Exception as e:
            logger.debug("Skipped metakg skills: %s", e)

        # Load Synthetic Biology design skills
        try:
            from . import synbio
            synbio.register(self)
        except Exception as e:
            logger.debug("Skipped synbio skills: %s", e)

        # Load population-genomics workflow planning skills
        try:
            from . import genomics
            genomics.register(self)
        except Exception as e:
            logger.debug("Skipped genomics skills: %s", e)

        # Load local AI bioinformatics skills (Boltz-2, ProteinMPNN, LigandMPNN via dplm2; BLAST via NCBI)
        try:
            from . import openbio
            openbio.register(self)
        except Exception as e:
            logger.debug("Skipped openbio skills: %s", e)

        # ── Unified Skills (consolidated wrappers) ──
        # Replace fine-grained skills with unified entry points.
        # Old skill names are kept as aliases pointing to the unified functions.
        try:
            from . import unified
            # Track which skills the unified module will replace
            _replaced = set()
            _REPLACED_SKILLS = {
                # uniprot_query replaces these
                'search_uniprot', 'get_uniprot_entry', 'get_uniprot_sequence',
                'get_uniprot_domains', 'get_uniprot_full',
                # ncbi_query replaces these
                'search_ncbi_protein', 'get_ncbi_protein_fasta', 'search_ncbi_gene',
                # pubchem_query replaces these
                'search_pubchem_compound', 'download_pubchem_sdf', 'validate_compound_smiles',
                # generate_conformer replaces these
                'conformer_generate', 'batch_conformer_generate', 'unimol_conformer',
                # esm2_mutation_scan replaces these
                'esm2_scan', 'esm2_scan_gpu', 'esm2_load',
                # chimerax_visualize replaces these
                'chimerax_render', 'chimerax_trajectory', 'chimerax_defattr',
                'chimerax_comparison', 'chimerax_porcupine', 'chimerax_morph',
                # generate_report replaces these
                'generate_mutation_report', 'generate_docking_report',
                'generate_md_report', 'generate_general_report',
                # mutate_protein (enhanced) replaces both
                'mutate_protein', 'mutate_minimize',
            }

            # Save old skills as _legacy_<name> aliases before replacing
            for name in _REPLACED_SKILLS:
                if name in self._skills:
                    self._aliases[name] = self._skills[name]
                    _replaced.add(name)

            # Remove replaced skills from registry
            for name in _replaced:
                del self._skills[name]
                # Clean up category lists
                for cat_skills in self._categories.values():
                    if name in cat_skills:
                        cat_skills.remove(name)

            # Register unified skills (they take over the slot)
            unified.register_unified_skills(self)

            logger.info("Unified %d skills into %d consolidated entry points",
                        len(_replaced), 8)
        except Exception as e:
            logger.warning("Failed to load unified skills, keeping originals: %s", e)

        # Load Workflow skills (optional multi-step pipelines)
        try:
            from . import workflow_skill
            workflow_skill.register_skills(self)
        except Exception as e:
            logger.debug("Skipped workflow skills: %s", e)

        logger.info("Loaded %d skills in %d categories",
                     len(self._skills), len(self._categories))

    def resolve_execution_target(self, skill_name: str) -> str:
        """
        Decide where a skill should run: local or SSH.

        Returns sandbox type string: 'native', 'wsl', 'ssh:<host>', etc.
        Logic:
          1. If skill is remote_preferred and a matching SSH host exists -> ssh
          2. If skill requires_gpu and no local GPU but SSH has gpu -> ssh
          3. If skill requires_wsl -> wsl
          4. Otherwise -> native (local)
        """
        skill = self.get(skill_name)
        if not skill:
            return 'native'

        # Check if SSH compute is available for this skill
        if skill.remote_capable and (skill.remote_preferred or skill.requires_gpu):
            try:
                from ..envs.ssh_config import find_host_for_capability
                cap = skill.remote_capability or ('gpu' if skill.requires_gpu else 'all')
                host = find_host_for_capability(cap)
                if host:
                    return f'ssh:{host.name}'
            except ImportError:
                pass

        if skill.requires_wsl:
            return 'wsl'
        return 'native'

    def resolve_chain(self, target_skill: str) -> List[str]:
        """Resolve the dependency chain for a target skill (topological sort).

        Returns ordered list of skill names that must execute before `target_skill`,
        ending with `target_skill` itself. Detects circular dependencies.
        """
        if not self._loaded:
            self.load_all()
        skill = self._skills.get(target_skill)
        if not skill:
            return [target_skill]

        visited = set()
        in_stack = set()
        order = []

        def _visit(name):
            if name in visited:
                return
            if name in in_stack:
                logger.warning("Circular dependency detected: %s", name)
                return  # break cycle gracefully
            in_stack.add(name)
            s = self._skills.get(name)
            if s:
                for dep in s.depends_on:
                    _visit(dep)
            in_stack.discard(name)
            visited.add(name)
            order.append(name)

        _visit(target_skill)
        return order

    def get_downstream(self, skill_name: str) -> List[str]:
        """Find skills that depend on the given skill (what to do next)."""
        if not self._loaded:
            self.load_all()
        return [s.name for s in self._skills.values()
                if skill_name in s.depends_on]

    def build_planning_catalog(self) -> str:
        """Build a structured skill catalog string for LLM planning prompts.

        Groups skills by workflow_stage, includes routing metadata and a
        VISUALIZATION ROUTING GUIDE section at the end.
        Returns plain structured text (not JSON) for optimal LLM reasoning.
        """
        self.load_all()

        # Canonical stage order
        stage_order = ['fetch', 'prepare', 'compute', 'analyze', 'visualize', 'report']
        # Collect skills by stage; unstaged skills go into 'other'
        by_stage: Dict[str, List[Skill]] = {s: [] for s in stage_order}
        by_stage['other'] = []

        for skill in self._skills.values():
            stage = skill.workflow_stage.strip() if skill.workflow_stage else ''
            if stage in by_stage:
                by_stage[stage].append(skill)
            else:
                by_stage['other'].append(skill)

        lines = ["=== AVAILABLE SKILL CATALOG ===", ""]

        def _fmt_requires(skill: Skill) -> str:
            reqs = []
            if skill.requires_gpu:
                reqs.append('gpu')
            if skill.requires_wsl:
                reqs.append('wsl')
            if skill.requires_network:
                reqs.append('network')
            return ', '.join(reqs) if reqs else 'none'

        all_stages = stage_order + ['other']
        for stage in all_stages:
            skills_in_stage = by_stage.get(stage, [])
            if not skills_in_stage:
                continue

            stage_label = stage.upper() if stage != 'other' else 'OTHER'
            lines.append(f"── {stage_label} STAGE ──")

            for skill in sorted(skills_in_stage, key=lambda s: s.name):
                lines.append(f"  skill: {skill.name}")
                lines.append(f"    description: {skill.description}")
                if skill.estimated_time:
                    lines.append(f"    estimated_time: {skill.estimated_time}")
                if skill.when_to_use:
                    lines.append(f"    when_to_use: {skill.when_to_use}")
                if skill.when_not_to_use:
                    lines.append(f"    when_not_to_use: {skill.when_not_to_use}")
                req_str = _fmt_requires(skill)
                if req_str != 'none':
                    lines.append(f"    requires: {req_str}")
                if skill.fallback_skills:
                    lines.append(f"    fallback_skills: {', '.join(skill.fallback_skills)}")
                lines.append("")

            lines.append("")

        # Visualization routing guide
        lines += [
            "=== VISUALIZATION ROUTING GUIDE ===",
            "DISABLED: chimerax_plip, chatmol_viz, chimerax_render, chimerax_trajectory,",
            "chimerax_defattr, chimerax_comparison, chimerax_porcupine, chimerax_morph",
            "are ALL temporarily disabled. Do NOT attempt to call them.",
            "",
            "Available visualization skills:",
            "- PLIP results available: → plip_visualize (PyMOL-based)",
            "- Open structure interactively: → pymol_view",
            "- Word report / docx output: → generate_report or enriched_docking_report",
            "",
            "CRITICAL: Never use 'pymol_render', 'chimerax_visualize', 'visualize_interactions', or",
            "'plip_analyze' — these names do NOT exist.",
        ]

        return "\n".join(lines)

    def to_manifest(self) -> List[dict]:
        """Export skill manifest as JSON-serializable list."""
        if not self._loaded:
            self.load_all()
        manifest = []
        for skill in self._skills.values():
            manifest.append({
                'name': skill.name,
                'description': skill.description,
                'category': skill.category,
                'requires_gpu': skill.requires_gpu,
                'requires_wsl': skill.requires_wsl,
                'requires_network': skill.requires_network,
                'remote_capable': skill.remote_capable,
                'remote_preferred': skill.remote_preferred,
                'available': skill.available,
                'inputs': [{'name': p.name, 'type': p.type, 'required': p.required,
                            'description': p.description} for p in skill.inputs],
                'outputs': [{'name': p.name, 'type': p.type,
                             'description': p.description} for p in skill.outputs],
            })
        return manifest


_registry_lock = __import__('threading').Lock()


def get_registry(config: dict = None) -> SkillRegistry:
    """Get or create the singleton registry (thread-safe)."""
    global _registry_instance
    if _registry_instance is None:
        with _registry_lock:
            if _registry_instance is None:  # double-checked locking
                _registry_instance = SkillRegistry(config)
    return _registry_instance
