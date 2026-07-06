#!/usr/bin/env python3
"""
CYP101A1 (P450cam) + HEME + Camphor — 100ns MD on AutoDL GPU
=============================================================
Self-contained script for remote execution on AutoDL server.
Uses CHARMM27 (native HEME) + ACPYPE/GAFF2 (camphor parameterization).

NOTE: CHARMM27+GAFF combination rule conflict exists but is acceptable
for a validation test. Production should use CHARMM36+CGenFF.
"""

import os
import re
import sys
import glob
import json
import time
import shutil
import logging
import subprocess
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger('cyp450_md')

# ── Configuration ──────────────────────────────────────────────────────
WORK_DIR = '/root/autodl-tmp/zhuangzhou_jobs/cyp450_md_test'
PDB_ID = '2CPP'  # P450cam with HEME + camphor
SIM_TIME_NS = 100.0
TEMP = 300.0
FF = 'charmm27'
WATER = 'tip3p'
BOX_DIST = 1.2
BOX_TYPE = 'dodecahedron'
GPU_EM = '-nb gpu'
GPU_FULL = '-nb gpu -pme gpu -bonded gpu -update gpu'

# HEME coordination bond parameters (Fe-S covalent bond)
FE_SG_R0 = 0.232   # nm
FE_SG_K = 209200.0  # kJ/mol/nm^2


def run_cmd(cmd, work_dir=WORK_DIR, input_text=None, timeout=3600):
    """Run a shell command with logging."""
    logger.info("CMD: %s", cmd)
    try:
        result = subprocess.run(
            cmd, shell=True, cwd=work_dir,
            input=input_text, capture_output=True, text=True,
            timeout=timeout
        )
        if result.returncode != 0:
            logger.error("FAILED (rc=%d): %s", result.returncode, result.stderr[-500:])
            return result.stdout + result.stderr, False
        return result.stdout + result.stderr, True
    except subprocess.TimeoutExpired:
        logger.error("TIMEOUT after %ds: %s", timeout, cmd)
        return f"Timeout after {timeout}s", False


def run_gmx(cmd, work_dir=WORK_DIR, input_text=None, timeout=3600):
    """Run a GROMACS command."""
    return run_cmd(f'gmx {cmd}', work_dir, input_text, timeout)


def download_pdb(pdb_id, output_path):
    """Download PDB from RCSB."""
    import urllib.request
    url = f'https://files.rcsb.org/download/{pdb_id}.pdb'
    logger.info("Downloading %s from RCSB...", pdb_id)
    urllib.request.urlretrieve(url, output_path)
    logger.info("Downloaded: %s", output_path)


def prepare_pdb(pdb_path, work_dir):
    """
    Clean PDB for GROMACS pdb2gmx with CHARMM27 HEME support.

    1. Keep ATOM + HETATM (protein + HEME)
    2. Remove waters, ions, camphor (will be added back after parameterization)
    3. Rename HEM -> HEME (CHARMM27 naming)
    4. Fix atom names for CHARMM27 compatibility
    5. Extract camphor to separate PDB
    """
    heme_names = {'HEM', 'HEME', 'HEC'}
    skip_names = {'HOH', 'WAT', 'SOL', 'DOD', 'TIP',
                  'NA', 'CL', 'K', 'NA+', 'CL-', 'K+',
                  'SO4', 'PO4', 'GOL', 'EDO', 'ACE', 'NH2',
                  'IOD', 'BR', 'MG', 'CA', 'ZN', 'MN'}
    camphor_names = {'CAM'}

    atom_lines = []  # (line, is_heme) tuples
    camphor_lines = []
    heme_detected = False
    fe_coords = None
    cys_sg = {}  # resid -> (x, y, z)

    with open(pdb_path, encoding='utf-8', errors='replace') as f:
        for line in f:
            if not (line.startswith('ATOM') or line.startswith('HETATM')):
                if line.startswith('END'):
                    continue
                continue

            resname = line[17:20].strip()
            atom_name = line[12:16].strip()

            # Extract camphor separately
            if resname in camphor_names:
                camphor_lines.append(line)
                continue

            # Skip waters, ions, additives
            if resname in skip_names:
                continue

            # HEME: rename HEM -> HEME for CHARMM27
            is_heme = resname in heme_names
            if is_heme:
                heme_detected = True
                # PDB format: cols 17(altloc) 18-20(resname) 21(chain)
                # GROMACS 4-letter resname: use cols 17-20 for "HEME", col 21 = space
                line = line[:17] + 'HEME ' + line[22:]
                # Track Fe position
                if atom_name == 'FE':
                    try:
                        fe_coords = (
                            float(line[30:38]),
                            float(line[38:46]),
                            float(line[46:54])
                        )
                    except ValueError:
                        pass

            # Track CYS SG for coordination detection
            if resname == 'CYS' and atom_name == 'SG':
                try:
                    resid = int(line[22:26].strip())
                    coords = (
                        float(line[30:38]),
                        float(line[38:46]),
                        float(line[46:54])
                    )
                    cys_sg[resid] = coords
                except (ValueError, IndexError):
                    pass

            atom_lines.append((line, is_heme))

    # Detect Fe-Cys coordination
    coordination = []
    if fe_coords and cys_sg:
        for resid, sg_coords in cys_sg.items():
            dist = sum((a - b)**2 for a, b in zip(fe_coords, sg_coords)) ** 0.5
            if dist < 3.0:  # Within 3 Angstrom
                coordination.append({
                    'type': 'Fe-SG',
                    'cys_resid': resid,
                    'distance_A': round(dist, 2)
                })
                logger.info("Fe-Cys coordination: CYS%d SG at %.2f A", resid, dist)

    # Write clean protein+HEME PDB with TER between protein and HEME
    clean_pdb = os.path.join(work_dir, 'protein_heme.pdb')
    with open(clean_pdb, 'w', encoding='utf-8') as f:
        last_was_protein = False
        for line, is_heme in atom_lines:
            # Insert TER between protein ATOM and HEME HETATM
            if is_heme and last_was_protein:
                f.write('TER\n')
            f.write(line)
            last_was_protein = not is_heme
        f.write('END\n')

    protein_lines = [line for line, _ in atom_lines]

    # Write camphor PDB
    camphor_pdb = None
    if camphor_lines:
        camphor_pdb = os.path.join(work_dir, 'camphor.pdb')
        with open(camphor_pdb, 'w', encoding='utf-8') as f:
            for line in camphor_lines:
                f.write(line)
            f.write('END\n')
        logger.info("Extracted camphor: %d atoms", len(camphor_lines))

    logger.info("Clean PDB: %d lines, HEME=%s, coordination=%d",
                len(protein_lines), heme_detected, len(coordination))

    return {
        'clean_pdb': clean_pdb,
        'camphor_pdb': camphor_pdb,
        'heme_detected': heme_detected,
        'coordination': coordination,
    }


def parameterize_camphor_acpype(camphor_pdb, work_dir):
    """
    Parameterize camphor with ACPYPE (GAFF2 + AM1-BCC charges).
    Returns path to GROMACS .itp and .gro files.
    """
    logger.info("=== Parameterizing camphor with ACPYPE ===")
    acpype_dir = os.path.join(work_dir, 'acpype_camphor')
    os.makedirs(acpype_dir, exist_ok=True)

    # Convert PDB to MOL2 via OpenBabel (better for ACPYPE)
    mol2_path = os.path.join(acpype_dir, 'camphor.mol2')
    out, ok = run_cmd(f'obabel {camphor_pdb} -O {mol2_path} --gen3d', work_dir)
    if not ok:
        # Try direct PDB input
        mol2_path = camphor_pdb

    # Run ACPYPE
    out, ok = run_cmd(
        f'acpype -i {mol2_path} -b CAM -c bcc -a gaff2 -o gmx -n 0',
        acpype_dir, timeout=600
    )
    if not ok:
        # Try with net charge auto-detect
        out, ok = run_cmd(
            f'acpype -i {camphor_pdb} -b CAM -c bcc -a gaff2 -o gmx',
            acpype_dir, timeout=600
        )
        if not ok:
            return {'error': f'ACPYPE failed: {out[-500:]}'}

    # Find output files
    acpype_out = None
    for d in glob.glob(os.path.join(acpype_dir, 'CAM.acpype')):
        acpype_out = d
    if not acpype_out:
        for d in glob.glob(os.path.join(acpype_dir, '*.acpype')):
            acpype_out = d

    if not acpype_out:
        return {'error': 'ACPYPE output directory not found'}

    # Find GROMACS files
    itp_file = os.path.join(acpype_out, 'CAM_GMX.itp')
    gro_file = os.path.join(acpype_out, 'CAM_GMX.gro')

    if not os.path.exists(itp_file):
        # Try alternative naming
        for f in glob.glob(os.path.join(acpype_out, '*_GMX.itp')):
            itp_file = f
        for f in glob.glob(os.path.join(acpype_out, '*_GMX.gro')):
            gro_file = f

    if not os.path.exists(itp_file):
        return {'error': f'ITP not found in {acpype_out}'}

    # Copy to work_dir
    shutil.copy(itp_file, os.path.join(work_dir, 'CAM.itp'))
    if os.path.exists(gro_file):
        shutil.copy(gro_file, os.path.join(work_dir, 'CAM.gro'))

    # Copy atomtypes if separate
    at_file = os.path.join(acpype_out, 'CAM_GMX_atomtypes.itp')
    if os.path.exists(at_file):
        shutil.copy(at_file, os.path.join(work_dir, 'CAM_atomtypes.itp'))

    logger.info("Camphor parameterization complete: %s", itp_file)
    return {
        'itp': os.path.join(work_dir, 'CAM.itp'),
        'gro': os.path.join(work_dir, 'CAM.gro') if os.path.exists(gro_file) else None,
        'atomtypes': os.path.join(work_dir, 'CAM_atomtypes.itp') if os.path.exists(at_file) else None,
    }


def fix_heme_topology(work_dir, coordination):
    """
    Inject explicit Fe-SG covalent bond into topology.
    This is CRITICAL — specbond.dat alone is unreliable.
    """
    logger.info("=== Fixing HEME Fe-SG coordination topology ===")
    topol_path = os.path.join(work_dir, 'topol.top')

    with open(topol_path, encoding='utf-8') as f:
        content = f.read()

    fixes = []

    for coord in coordination:
        cys_resid = coord['cys_resid']

        # Find Fe atom index and SG atom index from the .gro file
        gro_path = os.path.join(work_dir, 'protein.gro')
        fe_idx = None
        sg_idx = None

        with open(gro_path, encoding='utf-8') as f:
            lines = f.readlines()

        for i, line in enumerate(lines[2:-1], start=1):  # Skip header and box
            if len(line) < 44:
                continue
            resname = line[5:10].strip()
            atom = line[10:15].strip()
            resid = int(line[0:5].strip())

            if resname == 'HEME' and atom == 'FE':
                fe_idx = i
            if resid == cys_resid and atom == 'SG':
                sg_idx = i

        if fe_idx and sg_idx:
            logger.info("Fe atom: %d, SG atom (CYS%d): %d", fe_idx, cys_resid, sg_idx)

            # Add bond to [ bonds ] section
            bond_line = f"  {sg_idx:5d}  {fe_idx:5d}    1    {FE_SG_R0:.4f}  {FE_SG_K:.1f}  ; Fe-SG CYS{cys_resid} bonded coordination\n"

            # Find the [ bonds ] section in the protein moleculetype
            bonds_match = re.search(r'(\[ bonds \]\n;.*\n)', content)
            if bonds_match:
                insert_pos = bonds_match.end()
                content = content[:insert_pos] + bond_line + content[insert_pos:]
                fixes.append(f"Added Fe({fe_idx})-SG({sg_idx}) bond: r0={FE_SG_R0}nm, k={FE_SG_K}")
            else:
                logger.warning("[ bonds ] section not found in topology!")

            # Neutralize HG1 charge on coordinating CYS (thiolate form)
            # Find the atom in [ atoms ] section
            # Pattern: atom_number atom_type resid resname atom charge mass
            hg_pattern = re.compile(
                rf'(\s+\d+\s+\S+\s+{cys_resid}\s+CYS\d*\s+HG1\s+\d+\s+)([\d.+-]+)',
            )
            hg_match = hg_pattern.search(content)
            if hg_match:
                old_charge = float(hg_match.group(2))
                content = content[:hg_match.start(2)] + '0.000000' + content[hg_match.end(2):]
                fixes.append(f"Neutralized CYS{cys_resid} HG1 charge: {old_charge} -> 0.0")

                # Transfer charge to SG
                sg_pattern = re.compile(
                    rf'(\s+\d+\s+\S+\s+{cys_resid}\s+CYS\d*\s+SG\s+\d+\s+)([\d.+-]+)',
                )
                sg_match = sg_pattern.search(content)
                if sg_match:
                    old_sg = float(sg_match.group(2))
                    new_sg = old_sg + old_charge
                    content = content[:sg_match.start(2)] + f'{new_sg:.6f}' + content[sg_match.end(2):]
                    fixes.append(f"Transferred charge to SG: {old_sg} -> {new_sg}")

    with open(topol_path, 'w', encoding='utf-8') as f:
        f.write(content)

    logger.info("HEME topology fixes applied: %d", len(fixes))
    for fix in fixes:
        logger.info("  %s", fix)

    return {'fixes': fixes}


def merge_ligand_topology(work_dir, ligand_itp, camphor_pdb, atomtypes_itp=None):
    """
    Merge ligand topology into topol.top and prepare camphor .gro for insertion.
    Does NOT merge coordinates — uses gmx insert-molecules later.
    """
    logger.info("=== Merging ligand topology ===")

    # --- Fix ITP: ensure moleculetype is CAM, replace UNL residue name ---
    itp_path = os.path.join(work_dir, os.path.basename(ligand_itp))
    with open(itp_path, encoding='utf-8') as f:
        itp_content = f.read()

    # Fix moleculetype name
    itp_content = re.sub(
        r'(\[ moleculetype \]\s*\n;\s*\S+\s+\S+\s*\n)\s*(\S+)',
        r'\1CAM',
        itp_content
    )
    # Fix residue name UNL -> CAM in atoms section
    itp_content = itp_content.replace(' UNL ', ' CAM ')

    # If atomtypes are embedded in the ITP, extract them to separate file
    # to avoid conflicts when included after forcefield
    at_match = re.search(r'(\[ atomtypes \].*?)(\[ moleculetype \])',
                         itp_content, re.DOTALL)
    if at_match:
        atomtypes_block = at_match.group(1)
        itp_content = itp_content[at_match.start(2):]  # Remove atomtypes from ITP
        # Write atomtypes to separate file
        at_path = os.path.join(work_dir, 'CAM_atomtypes.itp')
        with open(at_path, 'w', encoding='utf-8') as f:
            f.write(atomtypes_block)
        atomtypes_itp = at_path
        logger.info("Extracted atomtypes from ITP to separate file")

    with open(itp_path, 'w', encoding='utf-8') as f:
        f.write(itp_content)

    # --- Update topology ---
    topol_path = os.path.join(work_dir, 'topol.top')
    with open(topol_path, encoding='utf-8') as f:
        content = f.read()

    # 1. Include atomtypes after forcefield include
    if atomtypes_itp and os.path.exists(atomtypes_itp):
        at_basename = os.path.basename(atomtypes_itp)
        ff_include_end = content.find('.ff/forcefield.itp"')
        if ff_include_end > 0:
            line_end = content.index('\n', ff_include_end)
            insert = f'\n#include "{at_basename}"\n'
            content = content[:line_end + 1] + insert + content[line_end + 1:]

    # 2. Include ligand .itp before [ system ]
    itp_basename = os.path.basename(ligand_itp)
    sys_pos = content.find('[ system ]')
    if sys_pos >= 0:
        insert = f'\n; Ligand topology\n#include "{itp_basename}"\n\n'
        content = content[:sys_pos] + insert + content[sys_pos:]

    # NOTE: Do NOT add CAM to [ molecules ] yet — insert-molecules will do it

    with open(topol_path, 'w', encoding='utf-8') as f:
        f.write(content)

    # --- Prepare camphor .gro at crystal position for insert-molecules ---
    acpype_gro = os.path.join(work_dir, 'CAM.gro')
    crystal_com = _get_pdb_com(camphor_pdb)
    camphor_gro = os.path.join(work_dir, 'camphor_insert.gro')
    _translate_gro_to_com(acpype_gro, camphor_gro, crystal_com)

    # Fix residue name in GRO to CAM
    with open(camphor_gro, encoding='utf-8') as f:
        gro_lines = f.readlines()
    with open(camphor_gro, 'w', encoding='utf-8') as f:
        for i, line in enumerate(gro_lines):
            if i >= 2 and i < len(gro_lines) - 1:  # atom lines
                # Replace residue name (positions 5-9)
                line = line[:5] + 'CAM  ' + line[10:]
            f.write(line)

    logger.info("Ligand topology prepared, camphor .gro at crystal position")


def _get_pdb_com(pdb_path):
    """Get center of mass from PDB file (returns nm coordinates)."""
    xs, ys, zs = [], [], []
    with open(pdb_path, encoding='utf-8', errors='replace') as f:
        for line in f:
            if line.startswith('ATOM') or line.startswith('HETATM'):
                try:
                    xs.append(float(line[30:38]) / 10.0)  # A -> nm
                    ys.append(float(line[38:46]) / 10.0)
                    zs.append(float(line[46:54]) / 10.0)
                except ValueError:
                    continue
    if not xs:
        return (0.0, 0.0, 0.0)
    return (sum(xs)/len(xs), sum(ys)/len(ys), sum(zs)/len(zs))


def _translate_gro_to_com(gro_in, gro_out, target_com):
    """Read a .gro file, translate all atoms so COM matches target_com."""
    with open(gro_in, encoding='utf-8') as f:
        lines = f.readlines()

    n_atoms = int(lines[1].strip())
    atom_lines = lines[2:2 + n_atoms]

    # Calculate current COM
    xs, ys, zs = [], [], []
    for line in atom_lines:
        try:
            x = float(line[20:28])
            y = float(line[28:36])
            z = float(line[36:44])
            xs.append(x)
            ys.append(y)
            zs.append(z)
        except (ValueError, IndexError):
            continue

    if not xs:
        shutil.copy(gro_in, gro_out)
        return

    cur_com = (sum(xs)/len(xs), sum(ys)/len(ys), sum(zs)/len(zs))
    dx = target_com[0] - cur_com[0]
    dy = target_com[1] - cur_com[1]
    dz = target_com[2] - cur_com[2]

    logger.info("Translating camphor: (%.3f, %.3f, %.3f) -> (%.3f, %.3f, %.3f), shift=(%.3f, %.3f, %.3f)",
                cur_com[0], cur_com[1], cur_com[2],
                target_com[0], target_com[1], target_com[2],
                dx, dy, dz)

    new_lines = [lines[0], lines[1]]
    for line in atom_lines:
        try:
            x = float(line[20:28]) + dx
            y = float(line[28:36]) + dy
            z = float(line[36:44]) + dz
            new_line = f'{line[:20]}{x:8.3f}{y:8.3f}{z:8.3f}\n'
            new_lines.append(new_line)
        except (ValueError, IndexError):
            new_lines.append(line)

    # Keep box line if present
    if len(lines) > 2 + n_atoms:
        new_lines.append(lines[-1])
    else:
        new_lines.append('   0.00000   0.00000   0.00000\n')

    with open(gro_out, 'w', encoding='utf-8') as f:
        f.writelines(new_lines)


def _pdb_to_gro_manual(pdb_path, gro_path):
    """Fallback manual PDB to GRO conversion."""
    atoms = []
    with open(pdb_path, encoding='utf-8', errors='replace') as f:
        for line in f:
            if line.startswith('ATOM') or line.startswith('HETATM'):
                resid = int(line[22:26].strip())
                resname = line[17:20].strip()
                atom_name = line[12:16].strip()
                x = float(line[30:38]) / 10.0  # A -> nm
                y = float(line[38:46]) / 10.0
                z = float(line[46:54]) / 10.0
                atoms.append((resid, resname, atom_name, x, y, z))

    with open(gro_path, 'w', encoding='utf-8') as f:
        f.write('Camphor\n')
        f.write(f'{len(atoms):5d}\n')
        for i, (resid, resname, atom_name, x, y, z) in enumerate(atoms, 1):
            f.write(f'{resid:5d}{resname:<5s}{atom_name:>5s}{i:5d}{x:8.3f}{y:8.3f}{z:8.3f}\n')
        f.write('   0.00000   0.00000   0.00000\n')


def write_mdp_files(work_dir, sim_time_ns, temp, tc_grps='Protein Non-Protein'):
    """Write MDP files for all simulation phases."""
    dt = 0.002  # 2 fs timestep
    nsteps_md = int(sim_time_ns * 1e6 / dt)

    # Energy minimization
    em_mdp = f"""; Energy minimization
integrator  = steep
emtol       = 1000.0
emstep      = 0.01
nsteps      = 50000
nstlist     = 10
cutoff-scheme = Verlet
ns_type     = grid
coulombtype = PME
rcoulomb    = 1.2
rvdw        = 1.2
pbc         = xyz
"""

    # Ions MDP
    ions_mdp = f"""; Ions
integrator  = steep
emtol       = 1000.0
emstep      = 0.01
nsteps      = 50000
nstlist     = 10
cutoff-scheme = Verlet
ns_type     = grid
coulombtype = PME
rcoulomb    = 1.2
rvdw        = 1.2
pbc         = xyz
"""

    # NVT equilibration (100ps)
    nvt_mdp = f"""; NVT equilibration
define       = -DPOSRES
integrator   = md
nsteps       = 50000
dt           = {dt}
nstxout-compressed = 5000
nstenergy    = 5000
nstlog       = 5000
continuation = no
constraint_algorithm = lincs
constraints  = h-bonds
lincs_iter   = 1
lincs_order  = 4
cutoff-scheme = Verlet
ns_type      = grid
nstlist      = 20
rcoulomb     = 1.2
rvdw         = 1.2
coulombtype  = PME
pme_order    = 4
fourierspacing = 0.12
tcoupl       = V-rescale
tc-grps      = {tc_grps}
tau_t        = {' '.join(['0.1'] * len(tc_grps.split()))}
ref_t        = {' '.join([str(temp)] * len(tc_grps.split()))}
pcoupl       = no
pbc          = xyz
DispCorr     = EnerPres
gen_vel      = yes
gen_temp     = {temp}
gen_seed     = -1
"""

    # NPT equilibration (100ps, C-rescale barostat)
    npt_mdp = f"""; NPT equilibration (C-rescale barostat)
define       = -DPOSRES
integrator   = md
nsteps       = 50000
dt           = {dt}
nstxout-compressed = 5000
nstenergy    = 5000
nstlog       = 5000
continuation = yes
constraint_algorithm = lincs
constraints  = h-bonds
lincs_iter   = 1
lincs_order  = 4
cutoff-scheme = Verlet
ns_type      = grid
nstlist      = 20
rcoulomb     = 1.2
rvdw         = 1.2
coulombtype  = PME
pme_order    = 4
fourierspacing = 0.12
tcoupl       = V-rescale
tc-grps      = {tc_grps}
tau_t        = {' '.join(['0.1'] * len(tc_grps.split()))}
ref_t        = {' '.join([str(temp)] * len(tc_grps.split()))}
pcoupl       = C-rescale
pcoupltype   = isotropic
tau_p        = 5.0
ref_p        = 1.0
compressibility = 4.5e-5
refcoord_scaling = com
pbc          = xyz
DispCorr     = EnerPres
gen_vel      = no
"""

    # Production MD (Parrinello-Rahman)
    md_mdp = f"""; Production MD ({sim_time_ns} ns)
integrator   = md
nsteps       = {nsteps_md}
dt           = {dt}
nstxout-compressed = 5000
nstenergy    = 5000
nstlog       = 5000
nstcalcenergy = 100
continuation = yes
constraint_algorithm = lincs
constraints  = h-bonds
lincs_iter   = 1
lincs_order  = 4
cutoff-scheme = Verlet
ns_type      = grid
nstlist      = 20
rcoulomb     = 1.2
rvdw         = 1.2
coulombtype  = PME
pme_order    = 4
fourierspacing = 0.12
tcoupl       = V-rescale
tc-grps      = {tc_grps}
tau_t        = {' '.join(['0.1'] * len(tc_grps.split()))}
ref_t        = {' '.join([str(temp)] * len(tc_grps.split()))}
pcoupl       = Parrinello-Rahman
pcoupltype   = isotropic
tau_p        = 2.0
ref_p        = 1.0
compressibility = 4.5e-5
pbc          = xyz
DispCorr     = EnerPres
gen_vel      = no
"""

    for name, content in [('em', em_mdp), ('ions', ions_mdp),
                           ('nvt', nvt_mdp), ('npt', npt_mdp), ('md', md_mdp)]:
        path = os.path.join(work_dir, f'{name}.mdp')
        with open(path, 'w', encoding='utf-8') as f:
            f.write(content)
    logger.info("Written MDP files: em, ions, nvt, npt, md")


def solvate_and_ionize(work_dir, input_gro='protein.gro'):
    """Solvate system and add counter-ions."""
    logger.info("=== Solvating system ===")

    # editconf: set box
    out, ok = run_gmx(
        f'editconf -f {input_gro} -o boxed.gro '
        f'-c -d {BOX_DIST} -bt {BOX_TYPE}',
        work_dir
    )
    if not ok:
        return None, f'editconf failed: {out[-300:]}'

    # solvate
    out, ok = run_gmx('solvate -cp boxed.gro -cs spc216.gro -o solv.gro -p topol.top', work_dir)
    if not ok:
        return None, f'solvate failed: {out[-300:]}'

    # genion: add ions to neutralize
    out, ok = run_gmx('grompp -f ions.mdp -c solv.gro -p topol.top -o ions.tpr -maxwarn 5', work_dir)
    if not ok:
        return None, f'grompp ions failed: {out[-300:]}'

    out, ok = run_gmx('genion -s ions.tpr -o system.gro -p topol.top -pname NA -nname CL -neutral',
                       work_dir, input_text='SOL\n')
    if not ok:
        return None, f'genion failed: {out[-300:]}'

    logger.info("Solvation + ionization complete")
    return os.path.join(work_dir, 'system.gro'), None


def run_stage(work_dir, stage, grompp_cmd, mdrun_extra='', timeout=3600):
    """Run a GROMACS simulation stage (grompp + mdrun)."""
    t0 = time.time()
    logger.info("=== Stage: %s ===", stage.upper())

    # grompp
    out, ok = run_gmx(grompp_cmd, work_dir)
    if not ok:
        return {'error': f'{stage} grompp failed', 'detail': out[-500:]}

    # mdrun
    cmd = f'mdrun -v -deffnm {stage} {mdrun_extra}'
    out, ok = run_gmx(cmd, work_dir, timeout=timeout)
    elapsed = time.time() - t0

    if not ok:
        return {'error': f'{stage} mdrun failed', 'detail': out[-500:], 'elapsed_s': elapsed}

    # Parse performance
    perf = None
    log_path = os.path.join(work_dir, f'{stage}.log')
    if os.path.exists(log_path):
        with open(log_path, encoding='utf-8', errors='replace') as f:
            for line in f:
                if 'Performance:' in line:
                    parts = line.split()
                    try:
                        idx = parts.index('Performance:')
                        perf = float(parts[idx + 1])
                    except (ValueError, IndexError):
                        pass

    result = {'success': True, 'elapsed_s': round(elapsed, 1)}
    if perf:
        result['performance_ns_day'] = perf
        logger.info("%s complete: %.1fs, %.1f ns/day", stage, elapsed, perf)
    else:
        logger.info("%s complete: %.1fs", stage, elapsed)

    return result


def fix_pbc_after_em(work_dir, stage='em'):
    """Fix PBC-wrapped atoms after EM (prevents LINCS blowup in NVT)."""
    gro_in = os.path.join(work_dir, f'{stage}.gro')
    tpr_in = os.path.join(work_dir, f'{stage}.tpr')
    gro_out = os.path.join(work_dir, f'{stage}_pbc.gro')

    if not os.path.exists(tpr_in):
        return

    out, ok = run_gmx(
        f'trjconv -s {stage}.tpr -f {stage}.gro -o {stage}_pbc.gro -pbc whole',
        work_dir, input_text='0\n'  # System
    )
    if ok and os.path.exists(gro_out):
        shutil.copy(gro_out, gro_in)
        logger.info("PBC fix applied after %s", stage)


def main():
    """Full CYP450 MD workflow."""
    os.makedirs(WORK_DIR, exist_ok=True)
    logger.info("=" * 60)
    logger.info("CYP101A1 (P450cam) + HEME + Camphor — %s ns MD", SIM_TIME_NS)
    logger.info("Work dir: %s", WORK_DIR)
    logger.info("GPU flags: %s", GPU_FULL)
    logger.info("=" * 60)

    report = {
        'pdb_id': PDB_ID,
        'sim_time_ns': SIM_TIME_NS,
        'temp': TEMP,
        'ff': FF,
        'work_dir': WORK_DIR,
        'start_time': time.strftime('%Y-%m-%d %H:%M:%S'),
        'stages': {},
    }

    # ── 1. Download PDB ───────────────────────────────────────────────
    pdb_path = os.path.join(WORK_DIR, f'{PDB_ID}.pdb')
    if not os.path.exists(pdb_path):
        try:
            download_pdb(PDB_ID, pdb_path)
        except Exception as e:
            report['error'] = f'Download failed: {e}'
            _save_report(report)
            return report
    report['stages']['download'] = {'success': True}

    # ── 2. Prepare PDB ────────────────────────────────────────────────
    prep = prepare_pdb(pdb_path, WORK_DIR)
    report['stages']['pdb_prep'] = {
        'success': True,
        'heme_detected': prep['heme_detected'],
        'coordination': prep['coordination'],
        'has_camphor': prep['camphor_pdb'] is not None,
    }

    # ── 3. Parameterize camphor (if present) ──────────────────────────
    has_ligand = False
    if prep['camphor_pdb']:
        cam_result = parameterize_camphor_acpype(prep['camphor_pdb'], WORK_DIR)
        if cam_result.get('error'):
            logger.warning("Camphor parameterization failed: %s", cam_result['error'])
            logger.warning("Proceeding without camphor (metalloprotein-only)")
            report['stages']['camphor_param'] = cam_result
        else:
            has_ligand = True
            report['stages']['camphor_param'] = {'success': True}

    # ── 4. pdb2gmx (CHARMM27 + native HEME) ──────────────────────────
    logger.info("=== pdb2gmx: CHARMM27 with native HEME support ===")
    pdb2gmx_cmd = (f'pdb2gmx -f protein_heme.pdb -o protein.gro '
                   f'-water {WATER} -ff {FF} -ignh -merge all -missing')
    out, ok = run_gmx(pdb2gmx_cmd, WORK_DIR)
    if not ok:
        logger.warning("pdb2gmx -merge all failed, retrying without -merge...")
        pdb2gmx_cmd = (f'pdb2gmx -f protein_heme.pdb -o protein.gro '
                       f'-water {WATER} -ff {FF} -ignh -missing')
        out, ok = run_gmx(pdb2gmx_cmd, WORK_DIR)
        if not ok:
            report['error'] = f'pdb2gmx failed: {out[-500:]}'
            _save_report(report)
            return report
    report['stages']['pdb2gmx'] = {'success': True}

    # ── 5. Fix HEME topology (explicit Fe-SG bond) ────────────────────
    if prep['coordination']:
        heme_fix = fix_heme_topology(WORK_DIR, prep['coordination'])
        report['stages']['heme_fix'] = heme_fix

    # ── 6. Merge camphor into system ──────────────────────────────────
    if has_ligand:
        merge_ligand_topology(
            WORK_DIR,
            os.path.join(WORK_DIR, 'CAM.itp'),
            prep['camphor_pdb'],  # Use ORIGINAL crystal coordinates
            os.path.join(WORK_DIR, 'CAM_atomtypes.itp') if os.path.exists(
                os.path.join(WORK_DIR, 'CAM_atomtypes.itp')) else None,
        )
        report['stages']['merge_ligand'] = {'success': True}

    # ── 7. Write MDP files ────────────────────────────────────────────
    tc_grps = 'Protein Non-Protein'
    write_mdp_files(WORK_DIR, SIM_TIME_NS, TEMP, tc_grps)

    # ── 8. Solvate + ions ─────────────────────────────────────────────
    system_gro, err = solvate_and_ionize(WORK_DIR)
    if err:
        report['error'] = f'Solvation failed: {err}'
        _save_report(report)
        return report
    report['stages']['solvate'] = {'success': True}

    # ── 9. Create index ───────────────────────────────────────────────
    ndx_path = os.path.join(WORK_DIR, 'index.ndx')
    out, ok = run_gmx(f'make_ndx -f system.gro -o index.ndx', WORK_DIR, input_text='q\n')
    ndx_flag = ' -n index.ndx' if ok else ''

    # ── 10. EM ────────────────────────────────────────────────────────
    em_result = run_stage(
        WORK_DIR, 'em',
        f'grompp -f em.mdp -c system.gro -p topol.top -o em.tpr -maxwarn 5',
        GPU_EM, timeout=1800
    )
    report['stages']['em'] = em_result
    if 'error' in em_result:
        report['error'] = 'EM failed'
        _save_report(report)
        return report
    fix_pbc_after_em(WORK_DIR)

    # ── 11. NVT ───────────────────────────────────────────────────────
    nvt_result = run_stage(
        WORK_DIR, 'nvt',
        f'grompp -f nvt.mdp -c em.gro -r em.gro -p topol.top{ndx_flag} -o nvt.tpr -maxwarn 5',
        GPU_FULL, timeout=3600
    )
    report['stages']['nvt'] = nvt_result
    if 'error' in nvt_result:
        report['error'] = 'NVT failed'
        _save_report(report)
        return report

    # ── 12. NPT ───────────────────────────────────────────────────────
    npt_result = run_stage(
        WORK_DIR, 'npt',
        f'grompp -f npt.mdp -c nvt.gro -r nvt.gro -t nvt.cpt -p topol.top{ndx_flag} -o npt.tpr -maxwarn 5',
        GPU_FULL, timeout=3600
    )
    report['stages']['npt'] = npt_result
    if 'error' in npt_result:
        report['error'] = 'NPT failed'
        _save_report(report)
        return report

    # ── 13. Production MD (100 ns) ────────────────────────────────────
    md_timeout = max(86400, int(SIM_TIME_NS * 3600 * 2))  # generous timeout
    md_result = run_stage(
        WORK_DIR, 'md',
        f'grompp -f md.mdp -c npt.gro -t npt.cpt -p topol.top{ndx_flag} -o md.tpr -maxwarn 5',
        GPU_FULL, timeout=md_timeout
    )
    report['stages']['md'] = md_result
    if 'error' in md_result:
        report['error'] = 'Production MD failed'

    # ── 14. Summary ───────────────────────────────────────────────────
    report['end_time'] = time.strftime('%Y-%m-%d %H:%M:%S')
    report['success'] = 'error' not in report
    if md_result.get('performance_ns_day'):
        report['performance_ns_day'] = md_result['performance_ns_day']
        eta_hours = SIM_TIME_NS / md_result['performance_ns_day'] * 24
        report['estimated_total_hours'] = round(eta_hours, 1)

    _save_report(report)
    logger.info("=" * 60)
    logger.info("MD COMPLETE" if report['success'] else "MD FAILED")
    logger.info("Report: %s", os.path.join(WORK_DIR, 'md_report.json'))
    logger.info("=" * 60)
    return report


def _save_report(report):
    path = os.path.join(WORK_DIR, 'md_report.json')
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, default=str)


if __name__ == '__main__':
    main()
