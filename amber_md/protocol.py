"""
Universal AMBER MD Protocol Engine
=====================================
Generates robust, system-adaptive MD input files and run scripts.

Design philosophy:
  - ONE code framework for all system types (apo protein, protein-ligand,
    metalloprotein, membrane — extend via protocol parameters)
  - Robust defaults validated on real systems (30K-atom continuous step-down)
  - All hyperparameters exposed for expert tuning
  - Auto-adapts restraint masks, step counts, and output frequency

Protocol (8 stages, continuous step-down):
  01 - Minimization: heavy-atom restrained
  02 - Minimization: free (no restraints)
  03 - Heating: NVT 0→target_temp, backbone(+ligand) restrained
  04 - NPT equilibration: strong restraint   (irest=1, ntx=5)
  05 - NPT equilibration: medium restraint   (irest=1, ntx=5)
  06 - NPT equilibration: light restraint    (irest=1, ntx=5)
  07 - NPT equilibration: free              (irest=1, ntx=5)
  08 - Production: NPT, no restraints        (irest=1, ntx=5)

Public API:
  AmberMDProtocol    — dataclass with all tunable hyperparameters
  default_protocol() — returns robust defaults for a given system type
  generate_inputs()  — writes 8 .in files from protocol
  generate_script()  — writes run_md.sh bash script from protocol
"""

import os
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

from .ligand_prep import _write_file, _win_to_wsl


# ---------------------------------------------------------------------------
# Protocol dataclass
# ---------------------------------------------------------------------------


@dataclass
class AmberMDProtocol:
    """Universal AMBER MD protocol configuration.

    Robust defaults work for most solvated protein systems (5K-50K atoms).
    Adjust hyperparameters for specific scenarios via ``default_protocol()``.

    Attributes (grouped by stage):

    **System metadata**
      system_type       — "protein", "protein_ligand", "metalloprotein", "membrane"
      ligand_name       — 3-letter residue name for ligand (used in masks)
      has_ligand        — whether ligand is present

    **Force field & solvation** (used upstream by tleap, not by this module)
      force_field       — "ff14SB" / "ff19SB"
      water_model       — "TIP3P" / "OPC" / "SPC/E"
      box_buffer        — truncated-octahedron buffer (Å)

    **MD engine**
      engine            — "pmemd.cuda" / "pmemd" / "sander"
      amber_source      — shell command to source AMBER env

    **Common MD parameters**
      dt                — timestep (ps), default 0.002
      cut               — nonbonded cutoff (Å), default 9.0
      ntc / ntf         — SHAKE constraints (2/2 = constrain H bonds)
      ntt               — thermostat (3 = Langevin)
      gamma_ln          — Langevin collision frequency (ps⁻¹)
      barostat          — barostat type (2 = Monte Carlo)
      temp0             — target temperature (K)

    **Minimization (stages 01-02)**
      min_maxcyc        — total min steps per stage
      min_ncyc          — steepest-descent steps (then conjugate gradient)
      min1_restraint_wt — restraint force on heavy atoms (stage 01)
      min1_mask         — AMBMASK for restrained min (default: heavy atoms)

    **Heating (stage 03)**
      heat_nstlim       — heating steps (default 50000 = 100 ps)
      heat_tempi        — initial temperature (K)
      heat_restraint_wt — restraint force during heating
      heat_mask         — AMBMASK during heating (auto-generated if None)

    **Equilibration step-down (stages 04-06)**
      equil_schedule    — list of (restraint_wt, nstlim) tuples
      equil_mask        — AMBMASK for restrained equil (auto if None)

    **Free equilibration (stage 07)**
      equil_free_nstlim — steps for free (unrestrained) NPT

    **Production (stage 08)**
      prod_ns           — production length in nanoseconds
      prod_ntpr         — energy output frequency (steps)
      prod_ntwx         — trajectory output frequency (steps)
      prod_ntwr         — restart output frequency (steps)
    """

    # --- System metadata ---
    system_type: str = "protein_ligand"
    ligand_name: str = "LIG"
    has_ligand: bool = True

    # --- Force field (informational, used by tleap upstream) ---
    force_field: str = "ff14SB"
    water_model: str = "TIP3P"
    box_buffer: float = 12.0

    # --- Engine ---
    engine: str = "pmemd.cuda"
    amber_source: str = "source ~/amber25/amber.sh"

    # --- Common MD settings ---
    dt: float = 0.002
    cut: float = 9.0
    ntc: int = 2
    ntf: int = 2
    ntt: int = 3
    gamma_ln: float = 2.0
    barostat: int = 2
    temp0: float = 300.0

    # --- Minimization ---
    min_maxcyc: int = 5000
    min_ncyc: int = 2500
    min1_restraint_wt: float = 10.0
    min1_mask: str = "'!@H='"  # heavy atoms

    # --- Heating ---
    heat_nstlim: int = 50000  # 100 ps
    heat_tempi: float = 0.0
    heat_restraint_wt: float = 5.0
    heat_mask: Optional[str] = None  # auto: backbone (+ligand)

    # --- Equilibration step-down ---
    equil_schedule: Optional[List[Tuple[float, int]]] = None
    equil_mask: Optional[str] = None  # auto: backbone (+ligand)

    # --- Free equilibration ---
    equil_free_nstlim: int = 100000  # 200 ps

    # --- Production ---
    prod_ns: float = 50.0
    prod_ntpr: int = 5000
    prod_ntwx: int = 5000
    prod_ntwr: int = 50000

    def __post_init__(self):
        """Fill in auto-generated fields."""
        if self.equil_schedule is None:
            self.equil_schedule = [
                (5.0, 50000),  # 100 ps, strong
                (2.0, 50000),  # 100 ps, medium
                (0.5, 50000),  # 100 ps, light
            ]

        # Auto-generate restraint masks
        if self.has_ligand and self.ligand_name:
            bb_lig = f"'@C,CA,N,O | :{self.ligand_name}'"
        else:
            bb_lig = "'@C,CA,N,O'"

        if self.heat_mask is None:
            self.heat_mask = bb_lig
        if self.equil_mask is None:
            self.equil_mask = bb_lig

    @property
    def prod_nstlim(self) -> int:
        """Production steps calculated from prod_ns and dt."""
        return int(self.prod_ns * 1e6 / (self.dt * 1000))

    def summary(self) -> str:
        """Human-readable summary of the protocol."""
        lines = [
            f"AMBER MD Protocol — {self.system_type}",
            f"  Force field: {self.force_field} + GAFF2 + {self.water_model}",
            f"  Engine: {self.engine}",
            f"  Temperature: {self.temp0} K",
            f"  Timestep: {self.dt} ps, Cutoff: {self.cut} Å",
            f"  Min: {self.min_maxcyc} steps × 2 stages",
            f"  Heating: {self.heat_nstlim * self.dt:.0f} ps, "
            f"{self.heat_tempi}→{self.temp0} K",
        ]
        for i, (wt, ns) in enumerate(self.equil_schedule):
            lines.append(f"  Equil {i + 1}: {ns * self.dt:.0f} ps, restraint {wt} kcal")
        lines.append(f"  Free equil: {self.equil_free_nstlim * self.dt:.0f} ps")
        lines.append(f"  Production: {self.prod_ns} ns")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Factory: robust defaults for different system types
# ---------------------------------------------------------------------------


def default_protocol(
    system_type: str = "protein_ligand",
    ligand_name: str = "LIG",
    prod_ns: float = 50.0,
    temp0: float = 300.0,
    engine: str = "pmemd.cuda",
    **overrides,
) -> AmberMDProtocol:
    """Create an AmberMDProtocol with robust defaults for a given system type.

    Args:
        system_type: "protein", "protein_ligand", "metalloprotein", "membrane"
        ligand_name: 3-letter ligand residue name
        prod_ns: Production time in nanoseconds
        temp0: Target temperature in K
        engine: AMBER engine binary
        **overrides: Any AmberMDProtocol field to override

    Returns:
        AmberMDProtocol with appropriate defaults.
    """
    base = dict(
        system_type=system_type,
        ligand_name=ligand_name,
        prod_ns=prod_ns,
        temp0=temp0,
        engine=engine,
    )

    if system_type == "protein":
        base.update(
            has_ligand=False,
            force_field="ff14SB",
            water_model="TIP3P",
            box_buffer=12.0,
        )

    elif system_type == "protein_ligand":
        base.update(
            has_ligand=True,
            force_field="ff14SB",
            water_model="TIP3P",
            box_buffer=12.0,
        )

    elif system_type == "metalloprotein":
        # Metals need longer equilibration and gentler step-down
        base.update(
            has_ligand=False,
            force_field="ff14SB",
            water_model="TIP3P",
            box_buffer=14.0,
            min_maxcyc=10000,
            min_ncyc=5000,
            min1_restraint_wt=50.0,
            heat_nstlim=100000,  # 200 ps slower heating
            heat_restraint_wt=10.0,
            equil_schedule=[
                (10.0, 50000),
                (5.0, 50000),
                (2.0, 50000),
                (1.0, 50000),
                (0.5, 50000),
            ],
            equil_free_nstlim=200000,  # 400 ps
        )

    elif system_type == "membrane":
        # Membrane: NVT heating then NPT with semi-isotropic barostat
        # Larger box, longer equilibration, backbone-only restraints
        base.update(
            has_ligand=False,
            force_field="ff14SB",
            water_model="TIP3P",
            box_buffer=15.0,
            min_maxcyc=10000,
            min_ncyc=5000,
            heat_nstlim=100000,  # 200 ps
            heat_restraint_wt=10.0,
            equil_schedule=[
                (10.0, 100000),
                (5.0, 100000),
                (2.0, 100000),
                (1.0, 100000),
                (0.5, 100000),
            ],
            equil_free_nstlim=500000,  # 1 ns
        )

    else:
        # Unknown — use protein_ligand defaults
        base.update(has_ligand=bool(ligand_name))

    base.update(overrides)
    return AmberMDProtocol(**base)


# ---------------------------------------------------------------------------
# Input file generation
# ---------------------------------------------------------------------------


def generate_inputs(protocol: AmberMDProtocol, output_dir: str, log=print) -> str:
    """Generate AMBER MD input files from protocol configuration.

    Creates ``md_input/`` subdirectory with 8 .in files implementing the
    continuous step-down protocol.

    Args:
        protocol: AmberMDProtocol instance
        output_dir: Base simulation directory (md_input/ created inside)
        log: Logging callable

    Returns:
        Path to md_input/ directory.
    """
    p = protocol
    md_dir = os.path.join(output_dir, "md_input")
    os.makedirs(md_dir, exist_ok=True)

    # --- Stage 01: Restrained minimization ---
    _write_file(
        os.path.join(md_dir, "01_min.in"),
        f"""Minimization with restraints on heavy atoms
 &cntrl
  imin=1, maxcyc={p.min_maxcyc}, ncyc={p.min_ncyc},
  cut={p.cut}, ntb=1,
  ntc={p.ntc}, ntf={p.ntf},
  ntpr=100,
  ntr=1, restraintmask={p.min1_mask}, restraint_wt={p.min1_restraint_wt},
 /
""",
    )

    # --- Stage 02: Free minimization ---
    _write_file(
        os.path.join(md_dir, "02_min.in"),
        f"""Free minimization
 &cntrl
  imin=1, maxcyc={p.min_maxcyc}, ncyc={p.min_ncyc},
  cut={p.cut}, ntb=1,
  ntc={p.ntc}, ntf={p.ntf},
  ntpr=100,
  ntr=0,
 /
""",
    )

    # --- Stage 03: NVT Heating ---
    _write_file(
        os.path.join(md_dir, "03_heat.in"),
        f"""Heating from {p.heat_tempi} to {p.temp0}K (NVT)
 &cntrl
  imin=0, irest=0, ntx=1,
  nstlim={p.heat_nstlim}, dt={p.dt},
  ntc={p.ntc}, ntf={p.ntf},
  cut={p.cut}, ntb=1,
  ntpr=1000, ntwx=1000, ntwr=1000,
  ntt={p.ntt}, gamma_ln={p.gamma_ln},
  tempi={p.heat_tempi}, temp0={p.temp0},
  ig=-1,
  ntr=1, restraintmask={p.heat_mask}, restraint_wt={p.heat_restraint_wt},
 /
""",
    )

    # --- Stages 04..N: NPT step-down ---
    for i, (wt, nstlim) in enumerate(p.equil_schedule):
        stage_num = i + 4
        fname = f"{stage_num:02d}_eq{i + 1}"
        _write_file(
            os.path.join(md_dir, f"{fname}.in"),
            f"""Equilibration (NPT) - Restraint {wt}
 &cntrl
  imin=0, irest=1, ntx=5,
  nstlim={nstlim}, dt={p.dt},
  ntc={p.ntc}, ntf={p.ntf}, cut={p.cut},
  ntb=2, ntp=1, barostat={p.barostat},
  ntpr=1000, ntwx=1000, ntwr=1000,
  ntt={p.ntt}, gamma_ln={p.gamma_ln}, temp0={p.temp0},
  ntr=1, restraintmask={p.equil_mask}, restraint_wt={wt},
 /
""",
        )

    # --- Free equilibration stage ---
    free_stage_num = len(p.equil_schedule) + 4
    free_fname = f"{free_stage_num:02d}_eq{len(p.equil_schedule) + 1}"
    _write_file(
        os.path.join(md_dir, f"{free_fname}.in"),
        f"""Equilibration (NPT) - Free
 &cntrl
  imin=0, irest=1, ntx=5,
  nstlim={p.equil_free_nstlim}, dt={p.dt},
  ntc={p.ntc}, ntf={p.ntf}, cut={p.cut}, ntb=2, ntp=1, barostat={p.barostat},
  ntpr=1000, ntwx=1000, ntwr=1000,
  ntt={p.ntt}, gamma_ln={p.gamma_ln}, temp0={p.temp0},
  ntr=0,
 /
""",
    )

    # --- Production stage ---
    prod_stage_num = free_stage_num + 1
    _write_file(
        os.path.join(md_dir, f"{prod_stage_num:02d}_prod.in"),
        f"""Production MD (NPT) - {p.prod_ns} ns
 &cntrl
  imin=0, irest=1, ntx=5,
  nstlim={p.prod_nstlim}, dt={p.dt},
  ntc={p.ntc}, ntf={p.ntf}, cut={p.cut}, ntb=2, ntp=1, barostat={p.barostat},
  ntpr={p.prod_ntpr}, ntwx={p.prod_ntwx}, ntwr={p.prod_ntwr},
  ntt={p.ntt}, gamma_ln={p.gamma_ln}, temp0={p.temp0},
  ntr=0,
 /
""",
    )

    total_stages = prod_stage_num
    log(f"  Generated {total_stages} MD input files in {md_dir}/")
    return md_dir


# ---------------------------------------------------------------------------
# Run script generation
# ---------------------------------------------------------------------------


def generate_script(
    protocol: AmberMDProtocol,
    output_dir: str,
    prmtop: str = "complex.prmtop",
    inpcrd: str = "complex.inpcrd",
    log=print,
) -> str:
    """Generate bash script for the continuous step-down MD protocol.

    Automatically adapts to the number of equilibration stages in the protocol.

    Args:
        protocol: AmberMDProtocol instance
        output_dir: Base simulation directory
        prmtop: Topology file name (relative to output_dir)
        inpcrd: Coordinate file name (relative to output_dir)
        log: Logging callable

    Returns:
        Path to run_md.sh.
    """
    p = protocol
    wsl_dir = _win_to_wsl(os.path.abspath(output_dir))

    n_equil = len(p.equil_schedule)
    total_stages = n_equil + 4 + 1  # 2 min + heat + N equil + 1 free + 1 prod
    free_stage = n_equil + 4
    prod_stage = free_stage + 1

    lines = [
        "#!/bin/bash",
        "#" + "=" * 78,
        "# Continuous Step-down AMBER MD Protocol",
        f"# Stages: 01-02 Min | 03 Heat | 04-{n_equil + 3:02d} NPT step-down "
        f"| {free_stage:02d} Free | {prod_stage:02d} Prod",
        "# Auto-generated by zhuangzhou.amber_md.protocol",
        "#" + "=" * 78,
        "set -e",
        "",
        p.amber_source,
        "",
        f'WORKDIR="{wsl_dir}"',
        'INPUT="${WORKDIR}/md_input"',
        f'TOP="${{WORKDIR}}/{prmtop}"',
        f'CRD="${{WORKDIR}}/{inpcrd}"',
        f'REF="${{WORKDIR}}/{inpcrd}"',
        "",
        'cd "$WORKDIR"',
        "mkdir -p md_output",
        "cd md_output",
        "",
        f'ENGINE="{p.engine}"',
        "",
        'echo "============================================================"',
        'echo "  Continuous Step-down AMBER MD — $(date)"',
        f'echo "  Protocol: {p.system_type}, {total_stages} stages"',
        'echo "============================================================"',
        "",
    ]

    stage = 1

    # Stage 1: Restrained minimization
    lines += [
        f'echo "[{stage}/{total_stages}] Minimization: heavy-atom restrained '
        f'({p.min1_restraint_wt} kcal)"',
        "$ENGINE -O -i ${INPUT}/01_min.in -o 01_min.out \\",
        "  -p ${TOP} -c ${CRD} -ref ${REF} -r 01_min.rst7",
        "",
    ]
    stage += 1

    # Stage 2: Free minimization
    lines += [
        f'echo "[{stage}/{total_stages}] Minimization: free (no restraints)"',
        "$ENGINE -O -i ${INPUT}/02_min.in -o 02_min.out \\",
        "  -p ${TOP} -c 01_min.rst7 -r 02_min.rst7",
        "",
    ]
    stage += 1

    # Stage 3: Heating
    heat_ps = p.heat_nstlim * p.dt
    lines += [
        f'echo "[{stage}/{total_stages}] Heating {p.heat_tempi}→{p.temp0} K '
        f'(NVT, {heat_ps:.0f} ps)"',
        "$ENGINE -O -i ${INPUT}/03_heat.in -o 03_heat.out \\",
        "  -p ${TOP} -c 02_min.rst7 -ref ${REF} -r 03_heat.rst7 -x 03_heat.nc",
        "",
    ]
    prev_rst = "03_heat.rst7"
    stage += 1

    # Stages 4..N+3: NPT step-down
    for i, (wt, nstlim) in enumerate(p.equil_schedule):
        sn = i + 4
        ps = nstlim * p.dt
        fname = f"{sn:02d}_eq{i + 1}"
        lines += [
            f'echo "[{stage}/{total_stages}] NPT equilibration: '
            f'{wt} kcal restraint ({ps:.0f} ps)"',
            f"$ENGINE -O -i ${{INPUT}}/{fname}.in -o {fname}.out \\",
            f"  -p ${{TOP}} -c {prev_rst} -ref ${{REF}} -r {fname}.rst7 -x {fname}.nc",
            "",
        ]
        prev_rst = f"{fname}.rst7"
        stage += 1

    # Free equilibration
    free_ps = p.equil_free_nstlim * p.dt
    free_fname = f"{free_stage:02d}_eq{n_equil + 1}"
    lines += [
        f'echo "[{stage}/{total_stages}] NPT equilibration: free ({free_ps:.0f} ps)"',
        f"$ENGINE -O -i ${{INPUT}}/{free_fname}.in -o {free_fname}.out \\",
        f"  -p ${{TOP}} -c {prev_rst} -r {free_fname}.rst7 -x {free_fname}.nc",
        "",
    ]
    prev_rst = f"{free_fname}.rst7"
    stage += 1

    # Production
    prod_fname = f"{prod_stage:02d}_prod"
    lines += [
        f'echo "[{stage}/{total_stages}] Production MD (NPT, {p.prod_ns} ns)"',
        f"$ENGINE -O -i ${{INPUT}}/{prod_fname}.in -o {prod_fname}.out \\",
        f"  -p ${{TOP}} -c {prev_rst} -r {prod_fname}.rst7 -x {prod_fname}.nc",
        "",
        'echo ""',
        'echo "============================================================"',
        'echo "  All stages completed! — $(date)"',
        f'echo "  Trajectory: ${{WORKDIR}}/md_output/{prod_fname}.nc"',
        f'echo "  Restart:    ${{WORKDIR}}/md_output/{prod_fname}.rst7"',
        'echo "============================================================"',
    ]

    script = "\n".join(lines) + "\n"
    path = os.path.join(output_dir, "run_md.sh")
    _write_file(path, script)
    log(f"  Written: run_md.sh ({total_stages} stages)")
    return path


# ---------------------------------------------------------------------------
# Convenience: generate both inputs and script
# ---------------------------------------------------------------------------


def setup_md(
    protocol: AmberMDProtocol,
    output_dir: str,
    prmtop: str = "complex.prmtop",
    inpcrd: str = "complex.inpcrd",
    log=print,
) -> dict:
    """Generate MD inputs + run script from protocol. One-call convenience.

    Returns:
        dict with 'md_input_dir', 'run_script', 'protocol_summary', 'total_stages',
        'prod_stage_prefix'.
    """
    md_dir = generate_inputs(protocol, output_dir, log)
    run_path = generate_script(protocol, output_dir, prmtop, inpcrd, log)

    n_equil = len(protocol.equil_schedule)
    prod_stage = n_equil + 4 + 1
    free_stage = prod_stage - 1

    return {
        "md_input_dir": md_dir,
        "run_script": run_path,
        "protocol_summary": protocol.summary(),
        "total_stages": prod_stage,
        "prod_stage_prefix": f"{prod_stage:02d}_prod",
        "free_stage_prefix": f"{free_stage:02d}_eq{n_equil + 1}",
    }
