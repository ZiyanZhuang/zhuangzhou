"""QM Engine skills — DFT, TS search, frequency analysis for enzyme catalysis.

All skills route through zhuangzhou/qm_engine/bridge.py which invokes PySCF inside WSL.
requires_wsl=True reflects this dependency. The bridge validates WSL availability at runtime.
"""

import logging
import os

from .registry import Skill, SkillParam, SkillRegistry

logger = logging.getLogger(__name__)


def _resolve_output_dir(output_dir: str, input_path: str, suffix: str) -> str:
    """Return output_dir if given, else CWD/<input_stem>_<suffix>/."""
    if output_dir:
        return output_dir
    stem = os.path.splitext(os.path.basename(input_path))[0]
    return os.path.join(os.getcwd(), f'{stem}_{suffix}')


def _check_qm_engine_available() -> dict | None:
    """Return error dict if qm_engine bridge is not importable, else None."""
    try:
        import zhuangzhou.qm_engine  # noqa: F401
        return None
    except ImportError as e:
        return {
            'error': f'QM engine unavailable: {e}',
            'hint': 'Ensure PySCF is installed in WSL: pip install pyscf geometric',
        }


def _qm_cluster(
    pdb_path: str,
    output_dir: str = None,
    center_residues: list = None,
    cutoff: float = 5.0,
    config: dict = None,
    **kw,
) -> dict:
    err = _check_qm_engine_available()
    if err:
        return err
    output_dir = _resolve_output_dir(output_dir, pdb_path, 'qm_cluster')
    from zhuangzhou.qm_engine import extract_qm_cluster

    return extract_qm_cluster(
        pdb_path,
        output_dir,
        center_residues=center_residues,
        plip_result=kw.get("plip_result"),
        cutoff=cutoff,
        include_waters=kw.get("include_waters", True),
        max_waters=kw.get("max_waters", 5),
        charge=kw.get("charge"),
        spin=kw.get("spin"),
        keep_backbone=kw.get("keep_backbone"),
        keep_residues=kw.get("keep_residues"),
        keep_waters=kw.get("keep_waters"),
    )


def _dft_opt(
    xyz_path: str,
    output_dir: str = None,
    charge: int = 0,
    spin: int = 0,
    functional: str = "b3lyp",
    basis: str = "def2-svp",
    config: dict = None,
    **kw,
) -> dict:
    err = _check_qm_engine_available()
    if err:
        return err
    output_dir = _resolve_output_dir(output_dir, xyz_path, 'dft_opt')
    from zhuangzhou.qm_engine import run_dft_optimization

    return run_dft_optimization(
        xyz_path,
        output_dir,
        charge=charge,
        spin=spin,
        functional=functional,
        basis=basis,
        constraints_path=kw.get("constraints_path"),
    )


def _ts_search(
    xyz_path: str,
    output_dir: str = None,
    charge: int = 0,
    spin: int = 0,
    functional: str = "b3lyp",
    basis: str = "def2-svp",
    config: dict = None,
    **kw,
) -> dict:
    err = _check_qm_engine_available()
    if err:
        return err
    output_dir = _resolve_output_dir(output_dir, xyz_path, 'ts_search')
    from zhuangzhou.qm_engine import run_ts_search

    return run_ts_search(
        xyz_path,
        output_dir,
        charge=charge,
        spin=spin,
        functional=functional,
        basis=basis,
        constraints_path=kw.get("constraints_path"),
    )


def _frequency(
    xyz_path: str,
    output_dir: str = None,
    charge: int = 0,
    spin: int = 0,
    functional: str = "b3lyp",
    basis: str = "def2-svp",
    config: dict = None,
    **kw,
) -> dict:
    err = _check_qm_engine_available()
    if err:
        return err
    output_dir = _resolve_output_dir(output_dir, xyz_path, 'frequency')
    from zhuangzhou.qm_engine import run_frequency

    return run_frequency(
        xyz_path,
        output_dir,
        charge=charge,
        spin=spin,
        functional=functional,
        basis=basis,
    )


def _single_point(
    xyz_path: str,
    output_dir: str = None,
    charge: int = 0,
    spin: int = 0,
    functional: str = "b3lyp",
    basis: str = "def2-tzvp",
    config: dict = None,
    **kw,
) -> dict:
    err = _check_qm_engine_available()
    if err:
        return err
    output_dir = _resolve_output_dir(output_dir, xyz_path, 'single_point')
    from zhuangzhou.qm_engine import run_single_point

    return run_single_point(
        xyz_path,
        output_dir,
        charge=charge,
        spin=spin,
        functional=functional,
        basis=basis,
    )


def _ts_pipeline(
    pdb_path: str,
    output_dir: str = None,
    center_residues: list = None,
    cutoff: float = 5.0,
    functional: str = "b3lyp",
    config: dict = None,
    **kw,
) -> dict:
    err = _check_qm_engine_available()
    if err:
        return err
    output_dir = _resolve_output_dir(output_dir, pdb_path, 'ts_pipeline')
    from zhuangzhou.qm_engine import run_ts_pipeline

    return run_ts_pipeline(
        pdb_path,
        output_dir,
        center_residues=center_residues,
        plip_result=kw.get("plip_result"),
        cutoff=cutoff,
        charge=kw.get("charge"),
        spin=kw.get("spin"),
        functional=functional,
    )


def _energy_profile(
    energies: dict,
    output_path: str,
    title: str = "Reaction Energy Profile",
    level: str = "DFT",
    config: dict = None,
    **kw,
) -> dict:
    err = _check_qm_engine_available()
    if err:
        return err
    from zhuangzhou.qm_engine import plot_energy_profile

    return plot_energy_profile(energies, output_path, title=title, level=level)


def _reaction_analysis(
    stationary_points: list,
    output_dir: str,
    title: str = "Reaction Energy Profile",
    level: str = "",
    config: dict = None,
    **kw,
) -> dict:
    err = _check_qm_engine_available()
    if err:
        return err
    from zhuangzhou.qm_engine import analyze_reaction

    return analyze_reaction(stationary_points, output_dir, title=title, level=level)


def _qm_report(
    results: dict,
    output_dir: str,
    title: str = "QM Computation Report",
    config: dict = None,
    **kw,
) -> dict:
    err = _check_qm_engine_available()
    if err:
        return err
    from zhuangzhou.qm_engine import generate_report

    return generate_report(results, output_dir, title=title)


def register_skills(registry: SkillRegistry):
    registry.register(
        Skill(
            name="qm_cluster",
            description="Extract QM cluster from PDB active site (PLIP-guided or manual)",
            category="qm",
            fn=_qm_cluster,
            inputs=[
                SkillParam("pdb_path", "path", "Input PDB file"),
                SkillParam("output_dir", "path", "Output directory (default: CWD/<stem>_<type>/)",
                           required=False, is_output_path=True),
                SkillParam(
                    "center_residues", "list", "Center residue names", required=False
                ),
                SkillParam(
                    "cutoff", "float", "Radius in Angstrom", required=False, default=5.0
                ),
                SkillParam(
                    "keep_backbone",
                    "list",
                    "List of (chain, resnum) to keep full backbone",
                    required=False,
                ),
                SkillParam(
                    "keep_residues",
                    "list",
                    "List of (chain, resnum) to keep regardless of cutoff",
                    required=False,
                ),
                SkillParam(
                    "keep_waters",
                    "list",
                    "List of int (resnum) of waters to explicitly keep",
                    required=False,
                ),
                SkillParam(
                    "include_waters",
                    "bool",
                    "Whether to auto-include waters within cutoff",
                    required=False,
                    default=True,
                ),
                SkillParam(
                    "max_waters",
                    "int",
                    "Max automated waters to include",
                    required=False,
                    default=5,
                ),
            ],
            outputs=[SkillParam("result", "dict", "XYZ, constraints, charge, spin")],
            requires_wsl=False,
            source="zhuangzhou/qm_engine/cluster.py",
        )
    )

    registry.register(
        Skill(
            name="dft_optimization",
            description="DFT geometry optimization (PySCF + geomeTRIC, constrained)",
            category="qm",
            fn=_dft_opt,
            inputs=[
                SkillParam("xyz_path", "path", "Input XYZ file"),
                SkillParam("output_dir", "path", "Output directory (default: CWD/<stem>_<type>/)",
                           required=False, is_output_path=True),
                SkillParam("charge", "int", "System charge", required=False, default=0),
                SkillParam("spin", "int", "2S value", required=False, default=0),
                SkillParam(
                    "functional",
                    "string",
                    "DFT functional",
                    required=False,
                    default="b3lyp",
                ),
                SkillParam(
                    "basis", "string", "Basis set", required=False, default="def2-svp"
                ),
            ],
            outputs=[SkillParam("result", "dict", "Optimized XYZ, energy")],
            requires_wsl=True,
            source="zhuangzhou/qm_engine/bridge.py",
        )
    )

    registry.register(
        Skill(
            name="ts_search",
            description="Transition state search (PySCF + geomeTRIC RS-P-RFO)",
            category="qm",
            fn=_ts_search,
            inputs=[
                SkillParam("xyz_path", "path", "Input XYZ (TS guess geometry)"),
                SkillParam("output_dir", "path", "Output directory (default: CWD/<stem>_<type>/)",
                           required=False, is_output_path=True),
                SkillParam("charge", "int", "System charge", required=False, default=0),
                SkillParam("spin", "int", "2S value", required=False, default=0),
                SkillParam(
                    "functional",
                    "string",
                    "DFT functional",
                    required=False,
                    default="b3lyp",
                ),
                SkillParam(
                    "basis", "string", "Basis set", required=False, default="def2-svp"
                ),
            ],
            outputs=[SkillParam("result", "dict", "TS XYZ, energy, convergence")],
            requires_wsl=True,
            source="zhuangzhou/qm_engine/bridge.py",
        )
    )

    registry.register(
        Skill(
            name="frequency_analysis",
            description="Vibrational frequency analysis (verify TS, thermodynamics)",
            category="qm",
            fn=_frequency,
            inputs=[
                SkillParam("xyz_path", "path", "Input XYZ file"),
                SkillParam("output_dir", "path", "Output directory (default: CWD/<stem>_<type>/)",
                           required=False, is_output_path=True),
                SkillParam("charge", "int", "System charge", required=False, default=0),
                SkillParam("spin", "int", "2S value", required=False, default=0),
            ],
            outputs=[
                SkillParam("result", "dict", "Frequencies, imaginary count, is_ts")
            ],
            requires_wsl=True,
            source="zhuangzhou/qm_engine/bridge.py",
        )
    )

    registry.register(
        Skill(
            name="dft_single_point",
            description="High-basis DFT single-point energy (def2-TZVP)",
            category="qm",
            fn=_single_point,
            inputs=[
                SkillParam("xyz_path", "path", "Input XYZ file"),
                SkillParam("output_dir", "path", "Output directory (default: CWD/<stem>_<type>/)",
                           required=False, is_output_path=True),
                SkillParam("charge", "int", "System charge", required=False, default=0),
                SkillParam("spin", "int", "2S value", required=False, default=0),
                SkillParam(
                    "basis", "string", "Basis set", required=False, default="def2-tzvp"
                ),
            ],
            outputs=[SkillParam("result", "dict", "Energy in Hartree and kcal/mol")],
            requires_wsl=True,
            source="zhuangzhou/qm_engine/bridge.py",
        )
    )

    registry.register(
        Skill(
            name="ts_pipeline",
            description="Full TS pipeline: PDB -> QM cluster -> DFT opt -> TS -> Freq -> SP",
            category="qm",
            fn=_ts_pipeline,
            inputs=[
                SkillParam("pdb_path", "path", "Input PDB (MD snapshot or crystal)"),
                SkillParam("output_dir", "path", "Output directory (default: CWD/<stem>_<type>/)",
                           required=False, is_output_path=True),
                SkillParam(
                    "center_residues", "list", "Center residue names", required=False
                ),
                SkillParam(
                    "cutoff", "float", "Active site radius", required=False, default=5.0
                ),
                SkillParam(
                    "functional",
                    "string",
                    "DFT functional",
                    required=False,
                    default="b3lyp",
                ),
            ],
            outputs=[SkillParam("result", "dict", "Full pipeline results with report")],
            requires_wsl=True,
            source="zhuangzhou/qm_engine/pipeline.py",
        )
    )

    registry.register(
        Skill(
            name="qm_energy_profile",
            description="Plot reaction energy barrier diagram (publication-quality matplotlib)",
            category="qm",
            fn=_energy_profile,
            inputs=[
                SkillParam("energies", "dict", "Ordered dict {label: energy_Ha}"),
                SkillParam("output_path", "path", "Output image path (.png/.pdf/.svg)"),
                SkillParam(
                    "title",
                    "string",
                    "Plot title",
                    required=False,
                    default="Reaction Energy Profile",
                ),
                SkillParam(
                    "level",
                    "string",
                    "Theory level string",
                    required=False,
                    default="DFT",
                ),
            ],
            outputs=[
                SkillParam("result", "dict", "Plot path, barriers, relative energies")
            ],
            requires_wsl=False,
            source="zhuangzhou/qm_engine/analysis.py",
        )
    )

    registry.register(
        Skill(
            name="qm_reaction_analysis",
            description="Analyze reaction stationary points: energy profile + report",
            category="qm",
            fn=_reaction_analysis,
            inputs=[
                SkillParam(
                    "stationary_points", "list", "List of {label, energy, frequencies}"
                ),
                SkillParam("output_dir", "path", "Output directory (default: CWD/<stem>_<type>/)",
                           required=False, is_output_path=True),
                SkillParam("title", "string", "Report title", required=False),
                SkillParam("level", "string", "Theory level string", required=False),
            ],
            outputs=[SkillParam("result", "dict", "Plot path, report path, barriers")],
            requires_wsl=False,
            source="zhuangzhou/qm_engine/analysis.py",
        )
    )

    registry.register(
        Skill(
            name="qm_report",
            description="Generate comprehensive QM computation report from pipeline results",
            category="qm",
            fn=_qm_report,
            inputs=[
                SkillParam("results", "dict", "Pipeline results dict"),
                SkillParam("output_dir", "path", "Output directory (default: CWD/<stem>_<type>/)",
                           required=False, is_output_path=True),
                SkillParam("title", "string", "Report title", required=False),
            ],
            outputs=[SkillParam("result", "dict", "Report path, JSON path, barriers")],
            requires_wsl=False,
            source="zhuangzhou/qm_engine/analysis.py",
        )
    )
