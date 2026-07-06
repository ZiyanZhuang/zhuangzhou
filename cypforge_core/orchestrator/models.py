from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


# ── status constants ────────────────────────────────────────────────────────

MODULE_STATUS = ("PENDING", "RUNNING", "PASS", "WARN", "FAIL", "SKIPPED")
WORKFLOW_STATUS = ("INITIALIZING", "RUNNING", "PAUSED", "STOPPED_ON_FAIL", "COMPLETED", "ERROR")

# ── RunConfig ───────────────────────────────────────────────────────────────

@dataclass
class RunConfig:
    run_name: str
    run_root: str
    project_root: str

    # user-provided input files (paths resolved at init time)
    raw_protein_heme_pdb: str = ""
    ligand_template_sdf: str = ""

    # heme/CYM settings
    heme_state: str = "IC6"          # IC6 | DIOXY | CPDI | CUSTOM
    heme_resname: str = "HEM"
    heme_chain: str = ""
    protein_chain: str = ""
    axial_cys_resid: int | None = None
    trim_transmembrane_ranges: str = ""
    trim_transmembrane_confirmed: bool = False

    # ligand settings
    ligand_resname: str = "NCT"
    ligand_chain: str = "L"
    formal_charge: int = 0
    spin: int = 1
    basis: str = "6-31G*"
    points_per_atom: int = 8
    fit_method: str = "multiwfn-resp"
    pre_resp_relax: str = "pbe-h-only"

    # protonation decision (path to the decision JSON)
    protonation_decision_json: str = ""

    # solvation settings
    protein_force_field: str = "ff19SB"
    water_leaprc: str = "leaprc.water.tip3p"
    water_model: str = "TIP3PBOX"
    box_type: str = "oct"
    buffer_a: float = 10.0
    neutralizing_anion: str = "Cl-"

    # WSL / Amber
    wsl_user: str = "john"
    amber_sh: str = ""
    multiwfn_bin: str = ""

    # workflow control
    auto_accept_warn: bool = False
    max_retries: int = 0

    @property
    def python_path(self) -> str:
        if not self.project_root:
            return ""
        return str(Path(self.project_root) / "src")


# ── StepDef ─────────────────────────────────────────────────────────────────

@dataclass
class StepDef:
    name: str
    kind: str                       # "python_script" | "wsl_command" | "python_inline"
    description: str = ""
    command_template: str = ""      # template string with {key} placeholders
    args: dict[str, Any] = field(default_factory=dict)
    timeout_seconds: int | None = None  # None = no explicit timeout


# ── ModuleDef ───────────────────────────────────────────────────────────────

@dataclass
class ModuleDef:
    skill_id: str
    skill_file: str
    description: str
    steps: list[StepDef] = field(default_factory=list)
    required_input_manifests: list[str] = field(default_factory=list)
    required_input_files: list[str] = field(default_factory=list)
    output_manifests: list[str] = field(default_factory=list)
    output_dir_rel: str = ""
    index: int = 0


# ── StepRecord ──────────────────────────────────────────────────────────────

@dataclass
class StepRecord:
    name: str
    status: str = "PENDING"
    command: str = ""
    working_dir: str = ""
    stdout_path: str = ""
    stderr_path: str = ""
    exit_code: int | None = None
    started_at: str = ""
    completed_at: str = ""
    sha256_inputs: dict[str, str] = field(default_factory=dict)
    error_message: str = ""


# ── ModuleRecord ────────────────────────────────────────────────────────────

@dataclass
class ModuleRecord:
    skill_id: str
    status: str = "PENDING"
    steps: list[StepRecord] = field(default_factory=list)
    gate_result: str = ""
    manifest_paths: dict[str, str] = field(default_factory=dict)
    output_dir: str = ""
    log_dir: str = ""
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    summary: str = ""
    started_at: str = ""
    completed_at: str = ""


# ── GateResult ──────────────────────────────────────────────────────────────

@dataclass
class GateResult:
    gate_id: str
    name: str
    status: str                    # PASS | WARN | FAIL
    detail: str
    evidence_file: str = ""


# ── RunManifest ─────────────────────────────────────────────────────────────

@dataclass
class RunManifest:
    schema: str = "cypforge.orchestrator.run_manifest.v1"
    run_name: str = ""
    run_root: str = ""
    project_root: str = ""
    workflow_status: str = "INITIALIZING"
    modules: dict[str, dict[str, Any]] = field(default_factory=dict)
    run_config: dict[str, Any] = field(default_factory=dict)
    absolute_stop_conditions: list[dict[str, Any]] = field(default_factory=list)
    started_at: str = ""
    updated_at: str = ""
    completed_at: str = ""


# ══════════════════════════════════════════════════════════════════════════════
#  Workflow module definitions  (10 stages)
# ══════════════════════════════════════════════════════════════════════════════

def build_module_definitions() -> list[ModuleDef]:
    """Return the ordered list of 10 workflow module definitions.

    These encode the commands, input/output manifests, and output directories
    described in skills/cypforge/*.md and execution_plan.md.
    """
    return [
        ModuleDef(
            skill_id="cypforge.environment_check",
            skill_file="environment_check.md",
            description="Verify project tree, Python imports, WSL Amber tools",
            index=0,
            output_dir_rel="00_environment_check",
            steps=[
                StepDef(
                    name="check_python_import",
                    kind="python_script",
                    description="Verify cypforge_core can be imported",
                    command_template='python -c "import cypforge_core; print(\'cypforge_core OK\')"',
                ),
                StepDef(
                    name="check_tleap",
                    kind="wsl_command",
                    description="Verify tleap is available",
                    command_template="which tleap",
                    timeout_seconds=30,
                ),
                StepDef(
                    name="check_pmemd_cuda",
                    kind="wsl_command",
                    description="Verify pmemd.cuda is available",
                    command_template="which pmemd.cuda",
                    timeout_seconds=30,
                ),
                StepDef(
                    name="check_cpptraj",
                    kind="wsl_command",
                    description="Verify cpptraj is available",
                    command_template="which cpptraj",
                    timeout_seconds=30,
                ),
                StepDef(
                    name="check_antechamber",
                    kind="wsl_command",
                    description="Verify antechamber is available",
                    command_template="which antechamber",
                    timeout_seconds=30,
                ),
                StepDef(
                    name="check_parmchk2",
                    kind="wsl_command",
                    description="Verify parmchk2 is available",
                    command_template="which parmchk2",
                    timeout_seconds=30,
                ),
            ],
            required_input_manifests=[],
            required_input_files=[],
            output_manifests=[],  # environment check validates tools; no manifest file needed
        ),

        ModuleDef(
            skill_id="cypforge.core1_prepare_heme_cym",
            skill_file="core1_prepare_heme_cym.md",
            description="Prepare CYP450 protein + HEME/CYM coordinates and LEaP mapping",
            index=1,
            output_dir_rel="01_heme_only",
            steps=[
                StepDef(
                    name="heme_only_prepare",
                    kind="python_script",
                    description="Prepare protein + heme/CYP coordinates",
                    command_template=(
                        "python scripts/heme_only.py"
                        " --heme-state {heme_state}"
                        " --output-dir {run_root}\\01_heme_only"
                        " --heme-resname {heme_resname}"
                        " --heme-chain {heme_chain}"
                        " --protein-chain {protein_chain}"
                        " --axial-cys-resid {axial_cys_resid}"
                        ' "{raw_protein_heme_pdb}"'
                    ),
                    timeout_seconds=120,
                ),
                StepDef(
                    name="heme_mapping_leapin",
                    kind="python_script",
                    description="Build heme/CYM LEaP mapping input",
                    command_template=(
                        "python scripts/heme_mapping_leapin.py"
                        ' --prepared-pdb "{run_root}\\01_heme_only\\prepared_heme_complex.pdb"'
                        ' --prepare-report-json "{run_root}\\01_heme_only\\prepare_report.json"'
                        " --output-dir {run_root}\\02_heme_mapping_leapin"
                        " --heme-resname {heme_resname}"
                    ),
                    timeout_seconds=60,
                ),
            ],
            required_input_manifests=[],
            required_input_files=["{raw_protein_heme_pdb}"],
            output_manifests=[
                "01_heme_only/prepare_report.json",
                "02_heme_mapping_leapin/heme_mapping_leapin_manifest.json",
            ],
        ),

        ModuleDef(
            skill_id="cypforge.core2_prepare_ligand_resp_gaff2",
            skill_file="core2_prepare_ligand_resp_gaff2.md",
            description="Generate ligand GAFF2/RESP parameters from SDF + complex PDB",
            index=2,
            output_dir_rel="10_ligand_gpu4pyscf_esp",
            steps=[
                StepDef(
                    name="ligand_gpu4pyscf_esp",
                    kind="python_script",
                    description="Run GPU4PySCF/Multiwfn RESP charge fitting",
                    command_template=(
                        "python scripts/ligand_gpu4pyscf_esp.py"
                        ' --complex-pdb "{raw_protein_heme_pdb}"'
                        ' --ligand-template-sdf "{ligand_template_sdf}"'
                        " --ligand-resname {ligand_resname}"
                        " --ligand-chain {ligand_chain}"
                        " --formal-charge {formal_charge}"
                        " --spin {spin}"
                        " --basis {basis}"
                        " --points-per-atom {points_per_atom}"
                        " --fit-method {fit_method}"
                        " --pre-resp-relax {pre_resp_relax}"
                        " --output-dir {run_root}\\10_ligand_gpu4pyscf_esp"
                    ),
                    timeout_seconds=600,
                ),
                StepDef(
                    name="ligand_mapping_leapin",
                    kind="python_script",
                    description="Build ligand-aware LEaP mapping input",
                    command_template=(
                        "python scripts/ligand_mapping_leapin.py"
                        ' --complex-pdb "{raw_protein_heme_pdb}"'
                        ' --prepare-report-json "{run_root}\\01_heme_only\\prepare_report.json"'
                        ' --ligand-mol2 "{run_root}\\10_ligand_gpu4pyscf_esp\\{ligand_resname}_multiwfn_resp.mol2"'
                        ' --ligand-frcmod "{run_root}\\10_ligand_gpu4pyscf_esp\\{ligand_resname}.frcmod"'
                        " --ligand-resname {ligand_resname}"
                        " --ligand-chain {ligand_chain}"
                        " --expected-ligand-charge {formal_charge}"
                        " --output-dir {run_root}\\13_ligand_mapping_leapin"
                        " --heme-resname {heme_resname}"
                    ),
                    timeout_seconds=60,
                ),
            ],
            required_input_manifests=[
                "01_heme_only/prepare_report.json",
                "02_heme_mapping_leapin/heme_mapping_leapin_manifest.json",
            ],
            required_input_files=["{ligand_template_sdf}"],
            output_manifests=["13_ligand_mapping_leapin/ligand_mapping_leapin_manifest.json"],
        ),

        ModuleDef(
            skill_id="cypforge.core3_finalize_protonation",
            skill_file="core3_finalize_protonation.md",
            description="Apply explicit residue rename decisions to the final complex",
            index=3,
            output_dir_rel="14_complex_protonation_finalize",
            steps=[
                StepDef(
                    name="protonation_finalize",
                    kind="python_script",
                    description="Finalize protonation states with explicit decisions",
                    command_template=(
                        "python scripts/complex_protonation_finalize.py"
                        ' --ligand-mapping-manifest-json "{run_root}\\13_ligand_mapping_leapin\\ligand_mapping_leapin_manifest.json"'
                        ' --original-prepared-pdb "{run_root}\\01_heme_only\\prepared_heme_complex.pdb"'
                        ' --protonation-decision-json "{protonation_decision_json}"'
                        " --output-dir {run_root}\\14_complex_protonation_finalize"
                    ),
                    timeout_seconds=120,
                ),
            ],
            required_input_manifests=["13_ligand_mapping_leapin/ligand_mapping_leapin_manifest.json"],
            required_input_files=[],
            output_manifests=["14_complex_protonation_finalize/protonation_finalize_manifest.json"],
        ),

        ModuleDef(
            skill_id="cypforge.core3_solvate_ionize",
            skill_file="core3_solvate_ionize.md",
            description="Build final solvated and neutralized Amber system",
            index=4,
            output_dir_rel="15_complex_solvation_ionization",
            steps=[
                StepDef(
                    name="solvate_ionize",
                    kind="python_script",
                    description="Solvate and neutralize the complex via tleap",
                    command_template=(
                        "python scripts/complex_solvation_ionization.py"
                        ' --protonation-manifest-json "{run_root}\\14_complex_protonation_finalize\\protonation_finalize_manifest.json"'
                        " --output-dir {run_root}\\15_complex_solvation_ionization"
                        " --protein-force-field {protein_force_field}"
                        " --water-leaprc {water_leaprc}"
                        " --water-model {water_model}"
                        " --box-type {box_type}"
                        " --buffer-a {buffer_a}"
                        " --neutralizing-anion {neutralizing_anion}"
                    ),
                    timeout_seconds=300,
                ),
            ],
            required_input_manifests=["14_complex_protonation_finalize/protonation_finalize_manifest.json"],
            required_input_files=[],
            output_manifests=["15_complex_solvation_ionization/solvation_manifest.json"],
        ),

        ModuleDef(
            skill_id="cypforge.core3_render_pre_md",
            skill_file="core3_render_pre_md.md",
            description="Generate nine-stage conservative pre-MD protocol inputs",
            index=5,
            output_dir_rel="17_complex_pre_md_equilibration",
            steps=[
                StepDef(
                    name="render_pre_md",
                    kind="python_script",
                    description="Generate mdin files and run script for 9-stage pre-MD",
                    command_template=(
                        "python scripts/complex_pre_md_equilibration.py"
                        ' --solvation-manifest-json "{run_root}\\15_complex_solvation_ionization\\solvation_manifest.json"'
                        " --output-dir {run_root}\\17_complex_pre_md_equilibration"
                    ),
                    timeout_seconds=120,
                ),
            ],
            required_input_manifests=["15_complex_solvation_ionization/solvation_manifest.json"],
            required_input_files=[],
            output_manifests=["17_complex_pre_md_equilibration/complex_pre_md_equilibration_manifest.json"],
        ),

        ModuleDef(
            skill_id="cypforge.core3_run_pre_md",
            skill_file="core3_run_pre_md.md",
            description="Run nine-stage pre-MD equilibration through WSL Amber",
            index=6,
            output_dir_rel="17_complex_pre_md_equilibration",
            steps=[
                StepDef(
                    name="run_pre_md",
                    kind="wsl_command",
                    description="Execute run_pre_md.sh via WSL (9 stages, stage 09 = 20ns free NPT)",
                    command_template=(
                        "cd {run_root_wsl}/17_complex_pre_md_equilibration && bash run_pre_md.sh"
                    ),
                    timeout_seconds=None,   # long-running: no timeout
                ),
            ],
            required_input_manifests=["17_complex_pre_md_equilibration/complex_pre_md_equilibration_manifest.json"],
            required_input_files=[
                "17_complex_pre_md_equilibration/run_pre_md.sh",
            ],
            output_manifests=["17_complex_pre_md_equilibration/complex_pre_md_equilibration_run_validation.json"],
        ),

        ModuleDef(
            skill_id="cypforge.global_audit",
            skill_file="global_audit.md",
            description="Full CYP450-specific global audit after pre-MD",
            index=7,
            output_dir_rel="18_global_cyp450_audit",
            steps=[
                StepDef(
                    name="global_audit",
                    kind="python_script",
                    description="Run all 9 audit gates on the completed pre-MD run",
                    command_template=(
                        "python scripts/complex_global_audit.py"
                        ' --ligand-mapping-manifest-json "{run_root}\\13_ligand_mapping_leapin\\ligand_mapping_leapin_manifest.json"'
                        ' --protonation-manifest-json "{run_root}\\14_complex_protonation_finalize\\protonation_finalize_manifest.json"'
                        ' --solvation-manifest-json "{run_root}\\15_complex_solvation_ionization\\solvation_manifest.json"'
                        ' --pre-md-manifest-json "{run_root}\\17_complex_pre_md_equilibration\\complex_pre_md_equilibration_manifest.json"'
                        ' --pre-md-run-validation-json "{run_root}\\17_complex_pre_md_equilibration\\complex_pre_md_equilibration_run_validation.json"'
                        " --output-dir {run_root}\\18_global_cyp450_audit"
                    ),
                    timeout_seconds=300,
                ),
            ],
            required_input_manifests=[
                "13_ligand_mapping_leapin/ligand_mapping_leapin_manifest.json",
                "14_complex_protonation_finalize/protonation_finalize_manifest.json",
                "15_complex_solvation_ionization/solvation_manifest.json",
                "17_complex_pre_md_equilibration/complex_pre_md_equilibration_manifest.json",
                "17_complex_pre_md_equilibration/complex_pre_md_equilibration_run_validation.json",
            ],
            required_input_files=[],
            output_manifests=["18_global_cyp450_audit/00_manifest.json"],
        ),

        ModuleDef(
            skill_id="cypforge.equilibration_decision",
            skill_file="equilibration_decision.md",
            description="Interpret global audit and decide next allowed action",
            index=8,
            output_dir_rel="18_global_cyp450_audit",
            steps=[
                StepDef(
                    name="equilibration_decision",
                    kind="python_inline",
                    description="Read global audit manifest and produce decision state",
                    command_template=(
                        "from cypforge_core.orchestrator.gates import make_equilibration_decision;"
                        ' make_equilibration_decision(r"{run_root}")'
                    ),
                    timeout_seconds=30,
                ),
            ],
            required_input_manifests=["18_global_cyp450_audit/00_manifest.json"],
            required_input_files=[],
            output_manifests=["18_global_cyp450_audit/equilibration_decision_state.json"],
        ),

        ModuleDef(
            skill_id="cypforge.production_readiness_check",
            skill_file="production_readiness_check.md",
            description="Prevent premature production MD claims",
            index=9,
            output_dir_rel="",
            steps=[
                StepDef(
                    name="production_readiness_check",
                    kind="python_inline",
                    description="Validate production readiness evidence",
                    command_template=(
                        "from cypforge_core.orchestrator.gates import make_production_readiness_check;"
                        ' make_production_readiness_check(r"{run_root}")'
                    ),
                    timeout_seconds=30,
                ),
            ],
            required_input_manifests=["18_global_cyp450_audit/equilibration_decision_state.json"],
            required_input_files=[],
            output_manifests=["production_readiness_state.json"],
        ),
    ]


def get_module_by_index(index: int) -> ModuleDef | None:
    modules = build_module_definitions()
    for m in modules:
        if m.index == index:
            return m
    return None


def get_module_by_skill_id(skill_id: str) -> ModuleDef | None:
    for m in build_module_definitions():
        if m.skill_id == skill_id:
            return m
    return None
