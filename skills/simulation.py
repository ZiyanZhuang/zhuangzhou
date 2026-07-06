"""Molecular dynamics simulation skills 閳?GROMACS + AMBER."""

from .registry import Skill, SkillParam, SkillRegistry


def _get_sandbox(remote_host: str = None):
    """Helper to create a sandbox for remote or local execution."""
    if not remote_host:
        return None
    from zhuangzhou.envs.sandbox import SSHSandbox

    return SSHSandbox(remote_host)


def gromacs_md(
    pdb_path: str,
    output_dir: str,
    simulation_time_ns: float = 10.0,
    remote_host: str = None,
    config: dict = None,
    **kw,
) -> dict:
    from zhuangzhou.gromacs_md import run_protein_ligand_md

    sandbox = _get_sandbox(remote_host)
    return run_protein_ligand_md(
        pdb_path, output_dir, sim_time_ns=simulation_time_ns, sandbox=sandbox
    )


def gromacs_analysis(
    work_dir: str,
    key_residues: str = "",
    ligand_resname: str = "MOL",
    remote_host: str = None,
    config: dict = None,
    **kw,
) -> dict:
    from zhuangzhou.gromacs_md import run_full_analysis

    if remote_host:
        sandbox = _get_sandbox(remote_host)
        sandbox.sync_from_remote(work_dir, work_dir)
    residues = [r.strip() for r in key_residues.split(",") if r.strip()] or None
    return run_full_analysis(
        work_dir, key_residues=residues, ligand_resname=ligand_resname
    )


def gromacs_visualize(
    work_dir: str, viz_type: str = "all", config: dict = None, **kw
) -> dict:
    from zhuangzhou.gromacs_md import render_trajectory, render_porcupine

    results = {}
    if viz_type in ("all", "animation"):
        results["animation"] = render_trajectory(work_dir)
    if viz_type in ("all", "porcupine"):
        results["porcupine"] = render_porcupine(work_dir)
    return results


def gromacs_process_trajectory(
    work_dir: str,
    dt_ps: int = 0,
    output_format: str = "both",
    config: dict = None,
    **kw,
) -> dict:
    """Fix PBC + remove translation/rotation -> analysis-ready trajectory."""
    from zhuangzhou.gromacs_md import process_trajectory

    return process_trajectory(
        work_dir, dt_ps=dt_ps or None, output_format=output_format
    )


def gromacs_comprehensive(
    work_dir: str,
    key_residues: str = "",
    ligand_resname: str = "MOL",
    config: dict = None,
    **kw,
) -> dict:
    """One-click comprehensive MD analysis: metrics + coordination + animation + porcupine."""
    from zhuangzhou.gromacs_md import run_full_analysis

    residues = [r.strip() for r in key_residues.split(",") if r.strip()] or None
    return run_full_analysis(
        work_dir, key_residues=residues, ligand_resname=ligand_resname, visualize=True
    )


def mutate_minimize(
    pdb_path: str,
    mutations: list,
    output_path: str = None,
    mode: str = "solvated",
    config: dict = None,
    **kw,
) -> dict:
    from zhuangzhou.mutate_minimize import mutate_and_minimize

    chain = (config or {}).get("chain", kw.get("chain", "A"))
    return mutate_and_minimize(
        pdb_path, mutations, output=output_path, chain=chain, mode=mode
    )


def minimize_only(
    pdb_path: str,
    output_path: str = None,
    mode: str = "solvated",
    config: dict = None,
    output_dir: str = None,   # accepted but resolved into output_path
    **kw,
) -> dict:
    import os
    from zhuangzhou.mutate_minimize import minimize_structure

    # Resolve output_dir → output_path if caller passed output_dir
    if output_dir and not output_path:
        os.makedirs(output_dir, exist_ok=True)
        stem = os.path.splitext(os.path.basename(pdb_path))[0]
        output_path = os.path.join(output_dir, f'{stem}_em.pdb')

    return minimize_structure(pdb_path, output=output_path, mode=mode)


def cgenff_parameterize(
    ligand_path: str,
    output_dir: str,
    resname: str = "LIG",
    str_file: str = None,
    config: dict = None,
    **kw,
) -> dict:
    """Deprecated: CGenFF parameterization (CHARMM). Use SOBTOP/GAFF instead."""
    return {
        "error": "CGenFF/CHARMM is deprecated. Use SOBTOP/GAFF via gromacs_md pipeline.",
        "hint": "The metalloprotein pipelines now use AMBER14SB + GAFF exclusively.",
    }


def metalloprotein_md(
    pdb_path: str,
    output_dir: str = None,
    simulation_time_ns: float = 10.0,
    stepdown: bool = True,
    remote_host: str = None,
    config: dict = None,
    **kw,
) -> dict:
    import os
    if output_dir is None:
        stem = os.path.splitext(os.path.basename(pdb_path))[0]
        output_dir = os.path.join(os.getcwd(), f'{stem}_metalloprotein_md')
    from zhuangzhou.gromacs_md.pipeline import run_metalloprotein_md

    sandbox = _get_sandbox(remote_host)
    return run_metalloprotein_md(
        pdb_path,
        output_dir,
        sim_time_ns=simulation_time_ns,
        stepdown=stepdown,
        sandbox=sandbox,
    )


def metalloprotein_ligand_md(
    pdb_path: str,
    output_dir: str = None,
    simulation_time_ns: float = 10.0,
    stepdown: bool = True,
    ligand_str_files: dict = None,
    remote_host: str = None,
    config: dict = None,
    **kw,
) -> dict:
    import os
    if output_dir is None:
        stem = os.path.splitext(os.path.basename(pdb_path))[0]
        output_dir = os.path.join(os.getcwd(), f'{stem}_metalloprotein_ligand_md')
    from zhuangzhou.gromacs_md.pipeline import run_metalloprotein_ligand_md

    sandbox = _get_sandbox(remote_host)
    return run_metalloprotein_ligand_md(
        pdb_path,
        output_dir,
        sim_time_ns=simulation_time_ns,
        stepdown=stepdown,
        ligand_str_files=ligand_str_files,
        sandbox=sandbox,
    )


def auto_md(
    pdb_path: str,
    output_dir: str = None,
    simulation_time_ns: float = 10.0,
    remote_host: str = None,
    config: dict = None,
    **kw,
) -> dict:
    """Auto-detect system type and route to the correct MD pipeline."""
    import os

    if output_dir is None:
        stem = os.path.splitext(os.path.basename(pdb_path))[0]
        output_dir = os.path.join(os.getcwd(), f'{stem}_md')

    if not os.path.exists(pdb_path):
        return {"error": f"PDB not found: {pdb_path}"}

    sandbox = _get_sandbox(remote_host)

    has_heme = False
    has_ligand = False
    heme_names = {"HEM", "HEME", "HEC", "HEA", "HEO2"}
    metal_names = {
        "FE",
        "ZN",
        "MN",
        "MG",
        "CU",
        "CO",
        "NI",
        "CA",
        "FE2",
        "FE3",
        "ZN2",
        "MN2",
        "MG2",
        "CU2",
        "CA2",
    }
    skip_names = (
        heme_names
        | metal_names
        | {
            "HOH",
            "WAT",
            "SOL",
            "NA",
            "CL",
            "K",
            "NA+",
            "CL-",
            "K+",
            "IOD",
            "BR",
            "DOD",
            "TIP",
            "SPC",
        }
    )
    standard_aa = {
        "ALA",
        "ARG",
        "ASN",
        "ASP",
        "CYS",
        "GLN",
        "GLU",
        "GLY",
        "HIS",
        "ILE",
        "LEU",
        "LYS",
        "MET",
        "PHE",
        "PRO",
        "SER",
        "THR",
        "TRP",
        "TYR",
        "VAL",
        "HID",
        "HIE",
        "HIP",
        "HSD",
        "HSE",
        "HSP",
        "CYX",
        "CYM",
        "CYS2",
        "CYF",
        "ASH",
        "GLH",
        "LYN",
    }

    with open(pdb_path, encoding="utf-8", errors="replace") as f:
        for line in f:
            if line.startswith("HETATM") or line.startswith("ATOM  "):
                resname = line[17:20].strip()
                if resname in heme_names or resname in metal_names:
                    has_heme = True
                elif (
                    line.startswith("HETATM")
                    and resname not in skip_names
                    and resname not in standard_aa
                ):
                    has_ligand = True

    if has_heme and has_ligand:
        from zhuangzhou.gromacs_md.pipeline import run_metalloprotein_ligand_md

        return run_metalloprotein_ligand_md(
            pdb_path,
            output_dir,
            sim_time_ns=simulation_time_ns,
            ligand_str_files=kw.get("ligand_str_files"),
            sandbox=sandbox,
        )
    elif has_heme:
        from zhuangzhou.gromacs_md.pipeline import run_metalloprotein_md

        return run_metalloprotein_md(
            pdb_path, output_dir, sim_time_ns=simulation_time_ns, sandbox=sandbox
        )
    elif has_ligand:
        from zhuangzhou.gromacs_md import run_protein_ligand_md

        return run_protein_ligand_md(
            pdb_path, output_dir, sim_time_ns=simulation_time_ns, sandbox=sandbox
        )
    else:
        from zhuangzhou.gromacs_md import run_protein_md

        return run_protein_md(
            pdb_path, output_dir, sim_time_ns=simulation_time_ns, sandbox=sandbox
        )


def amber_ligand_prep(
    ligand_path: str,
    output_dir: str,
    ligand_name: str = "LIG",
    charge: int = 0,
    spin: int = 1,
    resp_method: str = "multiwfn",
    protein_pdb: str = None,
    heme_state: str = None,
    config: dict = None,
    **kw,
) -> dict:
    from zhuangzhou.amber_md.ligand_prep import run_ligand_prep

    return run_ligand_prep(
        ligand_path,
        output_dir,
        ligand_name=ligand_name,
        charge=charge,
        spin=spin,
        resp_method=resp_method,
        protein_pdb=protein_pdb,
        heme_state=heme_state,
        config=config,
        **kw,
    )


def register_skills(registry: SkillRegistry):
    registry.register(
        Skill(
            name="mutate_minimize",
            description="Mutate protein + GROMACS energy minimization -> optimized PDB",
            category="mutagenesis",
            fn=mutate_minimize,
            inputs=[
                SkillParam("pdb_path", "path", "Input PDB file"),
                SkillParam(
                    "mutations",
                    "list",
                    'Mutation strings, e.g. ["V9L"] or ["V9L","A5G"]',
                ),
                SkillParam(
                    "output_path",
                    "path",
                    "Output PDB path (auto-generated if omitted)",
                    required=False,
                ),
                SkillParam(
                    "mode",
                    "string",
                    '"vacuum" (fast) or "solvated" (physical)',
                    required=False,
                    default="solvated",
                    choices=["vacuum", "solvated"],
                ),
            ],
            outputs=[
                SkillParam(
                    "result", "dict", "Minimized PDB, EM stats, mutation details"
                )
            ],
            source="zhuangzhou/mutate_minimize/engine.py",
            estimated_time="~15s",
            output_type="pdb",
        )
    )

    registry.register(
        Skill(
            name="minimize_structure",
            description="GROMACS energy minimization on any protein PDB (no mutation)",
            category="simulation",
            fn=minimize_only,
            inputs=[
                SkillParam("pdb_path", "path", "Input PDB file"),
                SkillParam(
                    "output_path",
                    "path",
                    "Output PDB path (auto-generated if omitted)",
                    required=False,
                ),
                SkillParam(
                    "mode",
                    "string",
                    '"vacuum" or "solvated"',
                    required=False,
                    default="solvated",
                    choices=["vacuum", "solvated"],
                ),
            ],
            outputs=[
                SkillParam("result", "dict", "Minimized PDB path, convergence, energy")
            ],
            source="zhuangzhou/mutate_minimize/engine.py",
            estimated_time="~10s",
        )
    )

    registry.register(
        Skill(
            name="gromacs_md",
            description="GROMACS protein-ligand MD (AMBER+GAFF, GPU): Split -> SOBTOP -> EM -> NVT -> NPT -> MD",
            category="simulation",
            fn=gromacs_md,
            requires_gpu=False,
            inputs=[
                SkillParam("pdb_path", "path", "Input PDB (protein or complex)"),
                SkillParam("output_dir", "path", "Simulation output directory (default: CWD/<stem>_md/)",
                           required=False, is_output_path=True),
                SkillParam(
                    "simulation_time_ns",
                    "float",
                    "Production run time (ns)",
                    required=False,
                    default=10.0,
                ),
                SkillParam(
                    "remote_host",
                    "string",
                    "Remote host name from ssh_compute.ini",
                    required=False,
                    default=None,
                ),
            ],
            outputs=[
                SkillParam("result", "dict", "Stage results, trajectories, performance")
            ],
            source="zhuangzhou/gromacs_md/",
            risk_level="high",
            estimated_time="~30min per 10ns",
        )
    )

    registry.register(
        Skill(
            name="metalloprotein_md",
            description="GROMACS metalloprotein MD (AMBER14SB+GAFF, GPU): Split PDB -> CYF -> HEM itp -> Fe-S bond -> EM -> NVT -> NPT -> MD",
            category="simulation",
            fn=metalloprotein_md,
            requires_gpu=False,
            inputs=[
                SkillParam(
                    "pdb_path", "path", "Input PDB (metalloprotein with HEME/Fe)"
                ),
                SkillParam("output_dir", "path", "Simulation output directory (default: CWD/<stem>_metalloprotein_md/)",
                           required=False, is_output_path=True),
                SkillParam(
                    "simulation_time_ns",
                    "float",
                    "Production run time (ns)",
                    required=False,
                    default=10.0,
                ),
                SkillParam(
                    "remote_host",
                    "string",
                    "Remote host name from ssh_compute.ini",
                    required=False,
                    default=None,
                ),
            ],
            outputs=[
                SkillParam(
                    "result", "dict", "Stage results, coordination info, trajectories"
                )
            ],
            source="zhuangzhou/gromacs_md/pipeline.py",
            risk_level="high",
            estimated_time="~30min per 10ns",
        )
    )

    registry.register(
        Skill(
            name="metalloprotein_ligand_md",
            description="GROMACS metalloprotein+ligand MD (AMBER14SB+GAFF, GPU): "
            "protein+CYF (pdb2gmx) + HEM itp + ligand (SOBTOP/GAFF) -> Fe-S bond -> stepdown equil -> MD",
            category="simulation",
            fn=metalloprotein_ligand_md,
            requires_gpu=False,
            inputs=[
                SkillParam(
                    "pdb_path", "path", "Input PDB (metalloprotein + ligand HETATM)"
                ),
                SkillParam("output_dir", "path", "Simulation output directory (default: CWD/<stem>_metalloprotein_ligand_md/)",
                           required=False, is_output_path=True),
                SkillParam(
                    "simulation_time_ns",
                    "float",
                    "Production run time (ns)",
                    required=False,
                    default=10.0,
                ),
                SkillParam(
                    "remote_host",
                    "string",
                    "Remote host name from ssh_compute.ini",
                    required=False,
                    default=None,
                ),
                SkillParam(
                    "stepdown",
                    "bool",
                    "Gradual restraint release (recommended)",
                    required=False,
                    default=True,
                ),
            ],
            outputs=[
                SkillParam(
                    "result", "dict", "Stage results, coordination, trajectories"
                )
            ],
            source="zhuangzhou/gromacs_md/pipeline.py",
            risk_level="high",
            estimated_time="~30min per 10ns",
        )
    )

    registry.register(
        Skill(
            name="cgenff_parameterize",
            description="[DEPRECATED] CGenFF ligand parameterization. Use SOBTOP/GAFF via metalloprotein pipelines instead.",
            category="simulation",
            fn=cgenff_parameterize,
            requires_gpu=False,
            inputs=[
                SkillParam("ligand_path", "path", "Ligand PDB or SDF file"),
                SkillParam(
                    "output_dir", "path", "Output directory for parameterized files"
                ),
                SkillParam(
                    "resname",
                    "str",
                    "3-letter residue name (e.g., FLP)",
                    required=False,
                    default="LIG",
                ),
                SkillParam(
                    "str_file",
                    "path",
                    "Pre-generated CGenFF .str file (optional)",
                    required=False,
                    default=None,
                ),
            ],
            outputs=[
                SkillParam(
                    "result",
                    "dict",
                    "itp, prm, gro paths + penalty scores + quality rating",
                )
            ],
            source="zhuangzhou/gromacs_md/cgenff_param.py",
        )
    )

    registry.register(
        Skill(
            name="auto_md",
            description="Auto-detect system type (protein/ligand/metalloprotein/complex) and route to optimal MD pipeline",
            category="simulation",
            fn=auto_md,
            requires_gpu=False,
            inputs=[
                SkillParam("pdb_path", "path", "Input PDB (any type - auto-detected)"),
                SkillParam("output_dir", "path", "Simulation output directory (default: CWD/<stem>_md/)",
                           required=False, is_output_path=True),
                SkillParam(
                    "simulation_time_ns",
                    "float",
                    "Production run time (ns)",
                    required=False,
                    default=10.0,
                ),
                SkillParam(
                    "remote_host",
                    "string",
                    "Remote host name from ssh_compute.ini",
                    required=False,
                    default=None,
                ),
            ],
            outputs=[SkillParam("result", "dict", "Stage results, trajectories")],
            source="zhuangzhou/gromacs_md/",
            risk_level="high",
            estimated_time="~30min per 10ns",
        )
    )

    registry.register(
        Skill(
            name="gromacs_analysis",
            description="MD trajectory analysis: RMSD, RMSF, Rg, H-bonds, FEL, distances, interaction energy",
            category="simulation",
            fn=gromacs_analysis,
            inputs=[
                SkillParam("work_dir", "path", "MD simulation directory"),
                SkillParam(
                    "key_residues",
                    "string",
                    "Comma-separated residues (e.g. ASP10,LEU151)",
                    required=False,
                    default="",
                ),
                SkillParam(
                    "ligand_resname",
                    "string",
                    "Ligand residue name",
                    required=False,
                    default="MOL",
                ),
            ],
            outputs=[SkillParam("result", "dict", "Analysis results + figure path")],
            source="zhuangzhou/gromacs_md/analysis.py",
            depends_on=["gromacs_process_trajectory"],
            output_type="analysis",
        )
    )

    registry.register(
        Skill(
            name="gromacs_visualize",
            description="MD trajectory animation (MP4/GIF) + PCA porcupine plot (PyMOL)",
            category="simulation",
            fn=gromacs_visualize,
            inputs=[
                SkillParam("work_dir", "path", "MD simulation directory"),
                SkillParam(
                    "viz_type",
                    "string",
                    "all/animation/porcupine",
                    required=False,
                    default="all",
                    choices=["all", "animation", "porcupine"],
                ),
            ],
            outputs=[SkillParam("result", "dict", "Video paths, porcupine PNG/PSE")],
            source="zhuangzhou/gromacs_md/visualize.py",
            depends_on=["gromacs_process_trajectory"],
            output_type="visualization",
        )
    )

    registry.register(
        Skill(
            name="gromacs_process_trajectory",
            description="Fix PBC + remove translation/rotation (least-squares backbone fit) "
            "-> analysis-ready trajectory (XTC/PDB)",
            category="simulation",
            fn=gromacs_process_trajectory,
            inputs=[
                SkillParam("work_dir", "path", "MD simulation directory"),
                SkillParam(
                    "dt_ps",
                    "int",
                    "Output time step in ps (0 = keep all frames)",
                    required=False,
                    default=0,
                ),
                SkillParam(
                    "output_format",
                    "string",
                    "xtc / pdb / both",
                    required=False,
                    default="both",
                    choices=["xtc", "pdb", "both"],
                ),
            ],
            outputs=[SkillParam("result", "dict", "Fitted XTC/PDB paths")],
            source="zhuangzhou/gromacs_md/visualize.py",
            depends_on=["auto_md"],
            output_type="trajectory",
        )
    )

    registry.register(
        Skill(
            name="gromacs_comprehensive",
            description="One-click comprehensive MD analysis: RMSD/RMSF/Rg/Energy/FEL/H-bonds "
            "+ auto metal coordination + trajectory animation + PCA porcupine",
            category="simulation",
            fn=gromacs_comprehensive,
            inputs=[
                SkillParam("work_dir", "path", "MD simulation directory"),
                SkillParam(
                    "key_residues",
                    "string",
                    "Comma-separated residues (e.g. ASP10,LEU151)",
                    required=False,
                    default="",
                ),
                SkillParam(
                    "ligand_resname",
                    "string",
                    "Ligand residue name",
                    required=False,
                    default="MOL",
                ),
            ],
            outputs=[
                SkillParam(
                    "result",
                    "dict",
                    "Full analysis + coordination + animation + porcupine",
                )
            ],
            source="zhuangzhou/gromacs_md/analysis.py",
            depends_on=["gromacs_process_trajectory"],
            output_type="analysis",
        )
    )

    # ----------------------------------------------------------------
    # AMBER CYP450+Heme MD Skills
    # ----------------------------------------------------------------

    def amber_cyp_md(
        pdb_path: str,
        output_dir: str = None,
        state: str = None,
        prod_ns: float = 10.0,
        keep_chains: list = None,
        config: dict = None,
        **kw,
    ) -> dict:
        from zhuangzhou.amber_md import run_cyp_heme_md

        return run_cyp_heme_md(
            pdb_path,
            output_dir=output_dir,
            state=state,
            keep_chains=keep_chains,
            prod_ns=prod_ns,
            config=config,
        )

    registry.register(
        Skill(
            name="amber_cyp_md",
            description="AMBER CYP450+Heme MD (ff19SB+GAFF, Shahrokh protocol): "
            "PDB -> heme detect -> Kabsch superpose -> tleap -> 10-stage equil -> production",
            category="simulation",
            fn=amber_cyp_md,
            requires_gpu=False,
            inputs=[
                SkillParam("pdb_path", "path", "Input PDB (CYP450 with heme)"),
                SkillParam(
                    "output_dir",
                    "path",
                    "Output directory (auto if omitted)",
                    required=False,
                    default=None,
                    is_output_path=True,
                ),
                SkillParam(
                    "state",
                    "string",
                    "Heme state: CPDI (Compound I), DIOXY, IC6 (resting). Auto-detect if omitted.",
                    required=False,
                    default=None,
                    choices=["CPDI", "DIOXY", "IC6"],
                ),
                SkillParam(
                    "prod_ns",
                    "float",
                    "Production MD time in nanoseconds",
                    required=False,
                    default=10.0,
                ),
                SkillParam(
                    "keep_chains",
                    "list",
                    "Chain IDs to keep (auto if omitted)",
                    required=False,
                    default=None,
                ),
            ],
            outputs=[
                SkillParam(
                    "result",
                    "dict",
                    "state, axial_cys, rmsd, output_dir, run_script, next_steps",
                )
            ],
            source="zhuangzhou/amber_md/",
            risk_level="medium",
            estimated_time="~1min prep, ~2h for 10ns on RTX 5070",
            output_type="trajectory",
        )
    )

    registry.register(
        Skill(
            name="amber_ligand_prep",
            description="AMBER high-precision ligand preparation: PySCF/Multiwfn RESP charges + GAFF2 + complex assembly",
            category="simulation",
            fn=amber_ligand_prep,
            inputs=[
                SkillParam("ligand_path", "path", "Input ligand file (SDF, MOL2, PDB)"),
                SkillParam(
                    "output_dir", "path", "Output directory", is_output_path=True
                ),
                SkillParam(
                    "ligand_name",
                    "string",
                    "Residue name (default: LIG)",
                    required=False,
                    default="LIG",
                ),
                SkillParam("charge", "int", "Net charge", required=False, default=0),
                SkillParam(
                    "spin",
                    "int",
                    "Spin multiplicity (singlet=1)",
                    required=False,
                    default=1,
                ),
                SkillParam(
                    "resp_method",
                    "string",
                    "multiwfn (PySCF required) or antechamber",
                    required=False,
                    default="multiwfn",
                    choices=["multiwfn", "antechamber"],
                ),
                SkillParam(
                    "protein_pdb",
                    "path",
                    "Protein PDB for complex assembly (optional)",
                    required=False,
                ),
                SkillParam(
                    "heme_state",
                    "string",
                    "Heme state (CPDI/DIOXY/IC6) for CYP systems",
                    required=False,
                ),
            ],
            outputs=[
                SkillParam("result", "dict", "mol2, frcmod, prmtop, inpcrd paths")
            ],
            source="zhuangzhou/amber_md/ligand_prep.py",
            estimated_time="~5min (QM-based) or ~30s (Antechamber)",
        )
    )

    # ----------------------------------------------------------------
    # AMBER General Protein-Ligand MD
    # ----------------------------------------------------------------

    def amber_protein_ligand_md(
        protein_pdb: str,
        ligand_path: str,
        output_dir: str = None,
        ligand_name: str = "LIG",
        charge: int = 0,
        spin: int = 1,
        resp_method: str = "multiwfn",
        force_field: str = "ff14SB",
        water_model: str = "TIP3P",
        box_buffer: float = 12.0,
        prod_ns: float = 50.0,
        do_docking: bool = True,
        docking_exhaustiveness: int = 32,
        engine: str = "pmemd.cuda",
        config: dict = None,
        **kw,
    ) -> dict:
        from zhuangzhou.amber_md.general_pipeline import run_protein_ligand_md

        return run_protein_ligand_md(
            protein_pdb,
            ligand_path,
            output_dir=output_dir,
            ligand_name=ligand_name,
            charge=charge,
            spin=spin,
            resp_method=resp_method,
            force_field=force_field,
            water_model=water_model,
            box_buffer=box_buffer,
            prod_ns=prod_ns,
            do_docking=do_docking,
            docking_exhaustiveness=docking_exhaustiveness,
            engine=engine,
            config=config,
            **kw,
        )

    registry.register(
        Skill(
            name="amber_protein_ligand_md",
            description="General AMBER protein+ligand MD pipeline: "
            "pdb4amber → GAFF2+RESP → Vina docking → tleap(ff14SB) → continuous step-down equil → production. "
            "Robust defaults, all hyperparameters tunable.",
            category="simulation",
            fn=amber_protein_ligand_md,
            requires_gpu=False,
            inputs=[
                SkillParam("protein_pdb", "path", "Input protein PDB"),
                SkillParam("ligand_path", "path", "Input ligand file (MOL2, SDF, PDB)"),
                SkillParam(
                    "output_dir",
                    "path",
                    "Output directory (auto if omitted)",
                    required=False,
                    default=None,
                    is_output_path=True,
                ),
                SkillParam(
                    "ligand_name",
                    "string",
                    "Ligand residue name (3-letter)",
                    required=False,
                    default="LIG",
                ),
                SkillParam(
                    "charge", "int", "Ligand net charge", required=False, default=0
                ),
                SkillParam(
                    "spin", "int", "Ligand spin multiplicity", required=False, default=1
                ),
                SkillParam(
                    "resp_method",
                    "string",
                    '"multiwfn" (PySCF+Multiwfn RESP) or "antechamber" (AM1-BCC)',
                    required=False,
                    default="multiwfn",
                    choices=["multiwfn", "antechamber"],
                ),
                SkillParam(
                    "force_field",
                    "string",
                    "Protein force field",
                    required=False,
                    default="ff14SB",
                    choices=["ff14SB", "ff19SB"],
                ),
                SkillParam(
                    "water_model",
                    "string",
                    "Water model",
                    required=False,
                    default="TIP3P",
                    choices=["TIP3P", "OPC", "SPC/E"],
                ),
                SkillParam(
                    "prod_ns",
                    "float",
                    "Production MD length (ns)",
                    required=False,
                    default=50.0,
                ),
                SkillParam(
                    "do_docking",
                    "bool",
                    "Use Vina to place ligand",
                    required=False,
                    default=True,
                ),
                SkillParam(
                    "engine",
                    "string",
                    "MD engine",
                    required=False,
                    default="pmemd.cuda",
                    choices=["pmemd.cuda", "pmemd", "sander"],
                ),
            ],
            outputs=[
                SkillParam(
                    "result",
                    "dict",
                    "output_dir, prmtop, inpcrd, run_script, protocol_summary, next_steps",
                )
            ],
            source="zhuangzhou/amber_md/general_pipeline.py",
            risk_level="medium",
            estimated_time="~5min prep, ~4h for 50ns on GPU",
            output_type="trajectory",
        )
    )

    # ----------------------------------------------------------------
    # AMBER Apo-Protein MD (no ligand)
    # ----------------------------------------------------------------

    def amber_apo_md(
        protein_pdb: str,
        output_dir: str = None,
        force_field: str = "ff14SB",
        water_model: str = "TIP3P",
        box_buffer: float = 12.0,
        prod_ns: float = 50.0,
        engine: str = "pmemd.cuda",
        config: dict = None,
        **kw,
    ) -> dict:
        from zhuangzhou.amber_md.general_pipeline import run_apo_protein_md

        return run_apo_protein_md(
            protein_pdb,
            output_dir=output_dir,
            force_field=force_field,
            water_model=water_model,
            box_buffer=box_buffer,
            prod_ns=prod_ns,
            engine=engine,
            config=config,
            **kw,
        )

    registry.register(
        Skill(
            name="amber_apo_md",
            description="AMBER apo-protein MD (no ligand): "
            "pdb4amber → tleap(ff14SB) → continuous step-down equil → production. "
            "Same robust protocol as protein-ligand but skips docking and ligand parameterization.",
            category="simulation",
            fn=amber_apo_md,
            requires_gpu=False,
            inputs=[
                SkillParam("protein_pdb", "path", "Input protein PDB"),
                SkillParam(
                    "output_dir",
                    "path",
                    "Output directory (auto if omitted)",
                    required=False,
                    default=None,
                    is_output_path=True,
                ),
                SkillParam(
                    "force_field",
                    "string",
                    "Protein force field",
                    required=False,
                    default="ff14SB",
                    choices=["ff14SB", "ff19SB"],
                ),
                SkillParam(
                    "water_model",
                    "string",
                    "Water model",
                    required=False,
                    default="TIP3P",
                    choices=["TIP3P", "OPC", "SPC/E"],
                ),
                SkillParam(
                    "box_buffer",
                    "float",
                    "Solvation box buffer (Å)",
                    required=False,
                    default=12.0,
                ),
                SkillParam(
                    "prod_ns",
                    "float",
                    "Production MD length (ns)",
                    required=False,
                    default=50.0,
                ),
                SkillParam(
                    "engine",
                    "string",
                    "MD engine",
                    required=False,
                    default="pmemd.cuda",
                    choices=["pmemd.cuda", "pmemd", "sander"],
                ),
            ],
            outputs=[
                SkillParam(
                    "result",
                    "dict",
                    "output_dir, prmtop, inpcrd, run_script, protocol_summary, next_steps",
                )
            ],
            source="zhuangzhou/amber_md/general_pipeline.py",
            risk_level="medium",
            estimated_time="~1min prep, ~4h for 50ns on GPU",
            output_type="trajectory",
        )
    )

    # ----------------------------------------------------------------
    # AMBER MD Setup (protocol-only, no prep — for pre-built topologies)
    # ----------------------------------------------------------------

    def amber_md_setup(
        output_dir: str,
        system_type: str = "protein_ligand",
        ligand_name: str = "LIG",
        prod_ns: float = 50.0,
        prmtop: str = "complex.prmtop",
        inpcrd: str = "complex.inpcrd",
        engine: str = "pmemd.cuda",
        temp0: float = 300.0,
        config: dict = None,
        **kw,
    ) -> dict:
        """Generate MD input files + run script for a pre-built AMBER topology."""
        from zhuangzhou.amber_md.protocol import default_protocol, setup_md

        proto = default_protocol(
            system_type=system_type,
            ligand_name=ligand_name,
            prod_ns=prod_ns,
            temp0=temp0,
            engine=engine,
        )
        return setup_md(proto, output_dir, prmtop=prmtop, inpcrd=inpcrd)

    registry.register(
        Skill(
            name="amber_md_setup",
            description="Generate AMBER MD input files + run script for a pre-built topology. "
            "Continuous step-down protocol with robust defaults. "
            "Use when prmtop/inpcrd already exist and you only need the MD protocol.",
            category="simulation",
            fn=amber_md_setup,
            inputs=[
                SkillParam("output_dir", "path", "Directory containing prmtop/inpcrd"),
                SkillParam(
                    "system_type",
                    "string",
                    "System type for default parameters",
                    required=False,
                    default="protein_ligand",
                    choices=["protein", "protein_ligand", "metalloprotein", "membrane"],
                ),
                SkillParam(
                    "ligand_name",
                    "string",
                    "Ligand residue name (for restraint masks)",
                    required=False,
                    default="LIG",
                ),
                SkillParam(
                    "prod_ns",
                    "float",
                    "Production MD length (ns)",
                    required=False,
                    default=50.0,
                ),
                SkillParam(
                    "prmtop",
                    "string",
                    "Topology filename",
                    required=False,
                    default="complex.prmtop",
                ),
                SkillParam(
                    "inpcrd",
                    "string",
                    "Coordinate filename",
                    required=False,
                    default="complex.inpcrd",
                ),
                SkillParam(
                    "engine",
                    "string",
                    "MD engine",
                    required=False,
                    default="pmemd.cuda",
                    choices=["pmemd.cuda", "pmemd", "sander"],
                ),
                SkillParam(
                    "temp0",
                    "float",
                    "Target temperature (K)",
                    required=False,
                    default=300.0,
                ),
            ],
            outputs=[
                SkillParam(
                    "result",
                    "dict",
                    "md_input_dir, run_script, protocol_summary, total_stages",
                )
            ],
            source="zhuangzhou/amber_md/protocol.py",
            risk_level="low",
            estimated_time="<1s",
        )
    )

    # ----------------------------------------------------------------
    # AMBER Post-MD Analysis (general protein–ligand)
    # ----------------------------------------------------------------

    def amber_md_analysis(
        work_dir: str,
        topology: str = "complex.prmtop",
        trajectory: str = "md_output/08_prod.nc",
        ligand_resname: str = "LIG",
        ligand_resid: int = None,
        catalytic_residues: str = "",
        pocket_residues: str = "",
        stable_frame_start: int = 0,
        stable_frame_end: int = 0,
        n_clusters: int = 5,
        extract_qm: bool = True,
        qm_residues: str = "",
        config: dict = None,
        **kw,
    ) -> dict:
        from zhuangzhou.amber_md.analysis import run_protein_ligand_analysis

        def _split(s):
            return [x.strip() for x in s.split(",") if x.strip()] if s else None

        frame_range = None
        if stable_frame_start > 0 and stable_frame_end > 0:
            frame_range = (stable_frame_start, stable_frame_end)

        return run_protein_ligand_analysis(
            work_dir=work_dir,
            topology=topology,
            trajectory=trajectory,
            ligand_resname=ligand_resname,
            ligand_resid=ligand_resid if ligand_resid and ligand_resid > 0 else None,
            catalytic_residues=_split(catalytic_residues),
            pocket_residues=_split(pocket_residues),
            stable_frame_range=frame_range,
            n_clusters=n_clusters,
            extract_qm=extract_qm,
            qm_residues=_split(qm_residues),
        )

    registry.register(
        Skill(
            name="amber_md_analysis",
            description="Post-production MD analysis for protein–ligand systems: "
            "RMSD/RMSF/Rg, H-bond network (direct + water bridges), "
            "K-means clustering on active-site + ligand, "
            "extract centroid PDB & QM region. Fully automated cpptraj pipeline.",
            category="simulation",
            fn=amber_md_analysis,
            requires_gpu=False,
            inputs=[
                SkillParam(
                    "work_dir",
                    "path",
                    "MD run directory (contains topology + md_output/)",
                ),
                SkillParam(
                    "topology",
                    "string",
                    "Topology filename",
                    required=False,
                    default="complex.prmtop",
                ),
                SkillParam(
                    "trajectory",
                    "string",
                    "Production trajectory path (relative to work_dir)",
                    required=False,
                    default="md_output/08_prod.nc",
                ),
                SkillParam(
                    "ligand_resname",
                    "string",
                    "Ligand residue name (3-letter)",
                    required=False,
                    default="LIG",
                ),
                SkillParam(
                    "ligand_resid",
                    "int",
                    "Ligand residue number (0 = auto-detect)",
                    required=False,
                    default=0,
                ),
                SkillParam(
                    "catalytic_residues",
                    "string",
                    "Comma-separated catalytic residue names, e.g. 'CYS16,HIP54,ALA56,GLN89,PHE159'",
                    required=False,
                    default="",
                ),
                SkillParam(
                    "pocket_residues",
                    "string",
                    "Comma-separated actual pocket residues for clustering (overrides catalytic if set)",
                    required=False,
                    default="",
                ),
                SkillParam(
                    "stable_frame_start",
                    "int",
                    "Start frame of stable binding phase (0 = full traj)",
                    required=False,
                    default=0,
                ),
                SkillParam(
                    "stable_frame_end",
                    "int",
                    "End frame of stable binding phase (0 = full traj)",
                    required=False,
                    default=0,
                ),
                SkillParam(
                    "n_clusters",
                    "int",
                    "Number of K-means clusters",
                    required=False,
                    default=5,
                ),
                SkillParam(
                    "extract_qm",
                    "bool",
                    "Extract centroid PDB + QM region",
                    required=False,
                    default=True,
                ),
                SkillParam(
                    "qm_residues",
                    "string",
                    "Comma-separated residues for QM region (default = pocket_residues)",
                    required=False,
                    default="",
                ),
            ],
            outputs=[
                SkillParam(
                    "result",
                    "dict",
                    "standard, hbond, clustering, structures, report_path, json_path",
                )
            ],
            source="zhuangzhou/amber_md/analysis.py",
            risk_level="low",
            estimated_time="~2-5 min depending on trajectory size",
            output_type="analysis",
        )
    )
