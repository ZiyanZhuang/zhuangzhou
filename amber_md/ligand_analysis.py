"""
Comprehensive MD Trajectory Analysis for Protein-Ligand Complexes.

Four analysis modules:
  1. Structural Analysis — RMSD, RMSF, Rg, key distances
  2. Interaction Analysis — PLIP snapshots, residue distances, H-bonds
  3. MM-GBSA Binding Energy — per-frame ΔG, per-residue decomposition
  4. Water Analysis — watershell, water bridges, closest waters

Public API:
  run_ligand_analysis()  — Full 4-module analysis pipeline
  generate_structural_analysis()  — Module 1 only
  generate_interaction_analysis()  — Module 2 only
  generate_mmpbsa_analysis()  — Module 3 only
  generate_water_analysis()  — Module 4 only
"""

import os

def _write_file(path, content):
    with open(path, 'w', encoding='utf-8', newline='\n') as f:
        f.write(content)


def _win_to_wsl(path):
    """Convert Windows path to WSL path."""
    path = path.replace('\\', '/')
    if len(path) >= 2 and path[1] == ':':
        drive = path[0].lower()
        return f'/mnt/{drive}{path[2:]}'
    return path


def run_ligand_analysis(work_dir, prmtop='complex.prmtop',
                        ligand_name='LIG', receptor_range='1-512',
                        hem_resid=None, axial_cys_resid=None,
                        binding_residues=None, heme_state=None,
                        modules='all', prod_ns=30.0,
                        config=None, **kw):
    """Generate comprehensive analysis scripts for protein-ligand MD.

    Args:
        work_dir: MD simulation directory (containing complex.prmtop, md_output/)
        prmtop: Topology file name
        ligand_name: 3-letter ligand residue name
        receptor_range: Protein residue range (e.g., '1-512')
        hem_resid: Heme residue ID (CYP systems)
        axial_cys_resid: Axial cysteine ID (CYP systems)
        binding_residues: List of binding site residue strings (e.g., ['PHE118', 'ALA315'])
        heme_state: Heme state (CPDI/DIOXY/IC6)
        modules: 'all' or comma-separated list (structural,interaction,mmpbsa,water)
        prod_ns: Production trajectory length in ns

    Returns:
        dict with script paths and next_steps, or error dict.
    """
    if not os.path.isdir(work_dir):
        return {'error': f'Directory not found: {work_dir}'}

    work_dir = os.path.abspath(work_dir)
    wsl_dir = _win_to_wsl(work_dir)

    # Parse modules
    if modules == 'all':
        run_modules = {'structural', 'interaction', 'mmpbsa', 'water'}
    else:
        run_modules = {m.strip() for m in modules.split(',')}

    results = {'output_dir': work_dir, 'scripts': {}}
    binding_residues = binding_residues or []

    # Determine last protein residue
    rec_parts = receptor_range.split('-')
    last_prot_res = int(rec_parts[-1]) if len(rec_parts) == 2 else 512

    try:
        if 'structural' in run_modules:
            path = generate_structural_analysis(
                work_dir, prmtop, ligand_name, receptor_range,
                hem_resid, axial_cys_resid, heme_state, prod_ns)
            results['scripts']['structural'] = path

        if 'interaction' in run_modules:
            path = generate_interaction_analysis(
                work_dir, prmtop, ligand_name, receptor_range,
                hem_resid, binding_residues, prod_ns)
            results['scripts']['interaction'] = path

        if 'mmpbsa' in run_modules:
            path = generate_mmpbsa_analysis(
                work_dir, prmtop, ligand_name, last_prot_res,
                hem_resid, binding_residues)
            results['scripts']['mmpbsa'] = path

        if 'water' in run_modules:
            path = generate_water_analysis(
                work_dir, prmtop, ligand_name, receptor_range, hem_resid)
            results['scripts']['water'] = path

        # Generate plotting scripts
        plot_path = _generate_plot_scripts(
            work_dir, ligand_name, receptor_range, hem_resid,
            axial_cys_resid, binding_residues, run_modules, prod_ns)
        results['scripts']['plot'] = plot_path

        results['next_steps'] = []
        if 'structural' in run_modules:
            results['next_steps'].append(f'bash {results["scripts"]["structural"]}')
        if 'interaction' in run_modules:
            results['next_steps'].append(f'bash {results["scripts"]["interaction"]}')
        if 'mmpbsa' in run_modules:
            results['next_steps'].append(f'bash {results["scripts"]["mmpbsa"]}')
        if 'water' in run_modules:
            results['next_steps'].append(f'bash {results["scripts"]["water"]}')
        results['next_steps'].append(f'python3 {plot_path}')

        return results

    except Exception as e:
        return {'error': f'Analysis generation failed: {e}'}


def generate_structural_analysis(work_dir, prmtop, ligand_name, receptor_range,
                                  hem_resid, axial_cys_resid, heme_state, prod_ns):
    """Generate cpptraj structural analysis script (RMSD/RMSF/Rg/distances)."""
    wsl_dir = _win_to_wsl(work_dir)
    analysis_dir = os.path.join(work_dir, 'analysis')
    os.makedirs(analysis_dir, exist_ok=True)

    rec_parts = receptor_range.split('-')
    first_res = rec_parts[0]
    last_res = rec_parts[1] if len(rec_parts) == 2 else rec_parts[0]

    # Build distance commands
    dist_cmds = []
    if hem_resid:
        dist_cmds.append(
            f'distance d_lig_fe :{ligand_name}&!@H= :{hem_resid}@FE out ${{OUTDIR}}/lig_fe_dist.dat')
        if heme_state == 'CPDI':
            # Per-carbon distances to heme O1 for hydroxylation analysis
            for atom in ['C2', 'C3', 'C4', 'C5', 'C6']:
                dist_cmds.append(
                    f'distance d_{atom.lower()}_o1 :{ligand_name}@{atom} :{hem_resid}@O1 '
                    f'out ${{OUTDIR}}/{atom.lower()}_o1_dist.dat')
    if axial_cys_resid and hem_resid:
        dist_cmds.append(
            f'distance d_fes :{axial_cys_resid}@SG :{hem_resid}@FE out ${{OUTDIR}}/fe_s_dist.dat')

    dist_str = '\n'.join(dist_cmds)

    # Heme RMSD
    heme_rmsd = ''
    if hem_resid:
        heme_rmsd = f"""
# Heme RMSD (heavy atoms, after CA fit)
rms first :{first_res}-{last_res}@CA
rms heme_rmsd :{hem_resid}&!@H= first out ${{OUTDIR}}/heme_rmsd.dat
"""

    script = f"""#!/bin/bash
# Structural Analysis: RMSD, RMSF, Rg, key distances
# Auto-generated by zhuangzhou.amber_md.ligand_analysis
source ~/amber25/amber.sh

WORKDIR="{wsl_dir}"
TOP="${{WORKDIR}}/{prmtop}"
TRAJ="${{WORKDIR}}/md_output/10_prod.nc"
OUTDIR="${{WORKDIR}}/analysis"
mkdir -p ${{OUTDIR}}

echo "=== Structural Analysis ==="

cpptraj -p ${{TOP}} <<'EOF'
trajin ${{TRAJ}}

# Protein RMSD (CA atoms)
rms protein_rmsd :{first_res}-{last_res}@CA first out ${{OUTDIR}}/protein_rmsd.dat
{heme_rmsd}
# Ligand RMSD (heavy atoms, nofit after CA alignment)
rms first :{first_res}-{last_res}@CA
rms lig_rmsd :{ligand_name}&!@H= first nofit out ${{OUTDIR}}/lig_rmsd.dat

# Key distances
{dist_str}

# RMSF (per-residue CA fluctuation)
rms first :{first_res}-{last_res}@CA
atomicfluct rmsf_ca :{first_res}-{last_res}@CA out ${{OUTDIR}}/rmsf_ca.dat byres

# Ligand RMSF (per-atom)
atomicfluct rmsf_lig :{ligand_name}&!@H= out ${{OUTDIR}}/rmsf_lig.dat byatom

# Radius of gyration
radgyr rog :{first_res}-{last_res} out ${{OUTDIR}}/rog.dat

# Extract first and last frames
go
quit
EOF

# Extract frames for visual comparison
cpptraj -p ${{TOP}} <<'EOF2'
trajin ${{TRAJ}} 1 1
strip :WAT,Na+,Cl-
trajout ${{OUTDIR}}/frame_first.pdb pdb
go
quit
EOF2

cpptraj -p ${{TOP}} <<'EOF3'
trajin ${{TRAJ}} lastframe lastframe
strip :WAT,Na+,Cl-
trajout ${{OUTDIR}}/frame_last.pdb pdb
go
quit
EOF3

echo "Structural analysis complete."
"""
    # Fix: cpptraj needs actual paths, not shell vars in heredoc
    script = script.replace('${TRAJ}', f'{wsl_dir}/md_output/10_prod.nc')
    script = script.replace('${TOP}', f'{wsl_dir}/{prmtop}')
    script = script.replace('${OUTDIR}', f'{wsl_dir}/analysis')

    path = os.path.join(work_dir, 'analyze_structural.sh')
    _write_file(path, script)
    return path


def generate_interaction_analysis(work_dir, prmtop, ligand_name, receptor_range,
                                   hem_resid, binding_residues, prod_ns):
    """Generate PLIP + cpptraj interaction analysis script."""
    wsl_dir = _win_to_wsl(work_dir)
    analysis_dir = os.path.join(work_dir, 'analysis')
    os.makedirs(analysis_dir, exist_ok=True)

    rec_parts = receptor_range.split('-')
    first_res = rec_parts[0]
    last_res = rec_parts[1] if len(rec_parts) == 2 else rec_parts[0]

    # Build residue distance commands from binding_residues
    res_dist_cmds = []
    for res_str in binding_residues:
        # Parse residue string like 'PHE118' -> resname=PHE, resid=118
        import re
        m = re.match(r'([A-Z]+)(\d+)', res_str)
        if m:
            resname, resid = m.group(1), m.group(2)
            res_dist_cmds.append(
                f'distance d_{resname.lower()}{resid} '
                f':{resname}&:{resid}&!@H= :{ligand_name}&!@H= '
                f'out ${{OUTDIR}}/dist_{resname.lower()}{resid}.dat')
    res_dist_str = '\n'.join(res_dist_cmds)

    script = f"""#!/bin/bash
# Interaction Analysis: PLIP + residue distances + H-bonds
# Auto-generated by zhuangzhou.amber_md.ligand_analysis
source ~/amber25/amber.sh

WORKDIR="{wsl_dir}"
TOP="${{WORKDIR}}/{prmtop}"
TRAJ="${{WORKDIR}}/md_output/10_prod.nc"
OUTDIR="${{WORKDIR}}/analysis"
mkdir -p ${{OUTDIR}}

echo "=== PLIP Snapshot Analysis ==="

# Extract mid-trajectory frame for PLIP
cpptraj -p ${{TOP}} <<'EOF'
trajin ${{TRAJ}} 1500 1500
strip :WAT,Na+,Cl-
trajout ${{OUTDIR}}/plip_frame.pdb pdb
go
quit
EOF

# Run PLIP
cd ${{OUTDIR}}
plip -f plip_frame.pdb -x -o plip_output 2>/dev/null || echo "PLIP failed (check installation)"

echo ""
echo "=== Residue Distance Analysis ==="

cpptraj -p ${{TOP}} <<'EOF2'
trajin ${{TRAJ}}

# Binding pocket residue distances (COM-COM)
{res_dist_str}

# Protein-ligand H-bonds (both directions)
hbond HB1 donormask :{first_res}-{last_res} acceptormask :{ligand_name} \\
  out ${{OUTDIR}}/hbond_prot_lig.dat avgout ${{OUTDIR}}/hbond_prot_lig_avg.dat
hbond HB2 donormask :{ligand_name} acceptormask :{first_res}-{last_res} \\
  out ${{OUTDIR}}/hbond_lig_prot.dat avgout ${{OUTDIR}}/hbond_lig_prot_avg.dat

go
quit
EOF2

echo "Interaction analysis complete."
"""
    script = script.replace('${TRAJ}', f'{wsl_dir}/md_output/10_prod.nc')
    script = script.replace('${TOP}', f'{wsl_dir}/{prmtop}')
    script = script.replace('${OUTDIR}', f'{wsl_dir}/analysis')

    path = os.path.join(work_dir, 'analyze_interactions.sh')
    _write_file(path, script)
    return path


def generate_mmpbsa_analysis(work_dir, prmtop, ligand_name, last_prot_res,
                              hem_resid, binding_residues):
    """Generate MM-GBSA binding energy analysis scripts."""
    wsl_dir = _win_to_wsl(work_dir)
    analysis_dir = os.path.join(work_dir, 'analysis')
    os.makedirs(analysis_dir, exist_ok=True)

    # Build print_res string for decomp
    decomp_residues = []
    for res_str in (binding_residues or []):
        import re
        m = re.match(r'([A-Z]+)(\d+)', res_str)
        if m:
            decomp_residues.append(m.group(2))
    if hem_resid:
        decomp_residues.append(str(hem_resid))

    # Receptor includes everything except ligand and solvent
    # For CYP: receptor = protein + heme
    if hem_resid:
        rec_strip = f':WAT,Na+,Cl-,:{ligand_name}'
        lig_strip = f':WAT,Na+,Cl-,:{1}-{last_prot_res},:{hem_resid}'
    else:
        rec_strip = f':WAT,Na+,Cl-,:{ligand_name}'
        lig_strip = f':WAT,Na+,Cl-,:{1}-{last_prot_res}'

    # MMPBSA input
    decomp_section = ''
    if decomp_residues:
        print_res = ','.join(decomp_residues)
        decomp_section = f"""&decomp
  idecomp=1,
  dec_verbose=0,
  print_res="{print_res}",
  csv_format=1,
/
"""

    mmpbsa_input = f"""Input for MM-GBSA
&general
  interval=30,
  verbose=2,
  keep_files=0,
/
&gb
  igb=5,
  saltcon=0.150,
/
{decomp_section}"""

    _write_file(os.path.join(analysis_dir, 'mmpbsa_gb.in'), mmpbsa_input)

    script = f"""#!/bin/bash
# MM-GBSA Binding Energy Analysis
# Auto-generated by zhuangzhou.amber_md.ligand_analysis
source ~/amber25/amber.sh

WORKDIR="{wsl_dir}"
TOP="${{WORKDIR}}/{prmtop}"
TRAJ="${{WORKDIR}}/md_output/10_prod.nc"
OUTDIR="${{WORKDIR}}/analysis"

cd ${{OUTDIR}}

echo "=== Preparing stripped topologies ==="

# Strip solvent from complex
cpptraj -p ${{TOP}} <<'STRIP'
parm ${{TOP}} [full]
parmstrip :WAT,Na+,Cl- parmindex 0
parmwrite out ${{OUTDIR}}/com_stripped.prmtop

parm ${{TOP}} [rec]
parmstrip {rec_strip} parmindex 1
parmwrite out ${{OUTDIR}}/rec_stripped.prmtop

parm ${{TOP}} [lig]
parmstrip {lig_strip} parmindex 2
parmwrite out ${{OUTDIR}}/lig_stripped.prmtop
quit
STRIP

echo ""
echo "=== Running MMPBSA.py ==="

MMPBSA.py -O \\
  -i ${{OUTDIR}}/mmpbsa_gb.in \\
  -o ${{OUTDIR}}/mmpbsa_results.dat \\
  -eo ${{OUTDIR}}/mmpbsa_per_frame.csv \\
  -do ${{OUTDIR}}/mmpbsa_decomp.dat \\
  -sp ${{TOP}} \\
  -cp ${{OUTDIR}}/com_stripped.prmtop \\
  -rp ${{OUTDIR}}/rec_stripped.prmtop \\
  -lp ${{OUTDIR}}/lig_stripped.prmtop \\
  -y ${{TRAJ}}

echo ""
echo "=== MM-GBSA Complete ==="
echo "Results: ${{OUTDIR}}/mmpbsa_results.dat"
"""
    script = script.replace('${TRAJ}', f'{wsl_dir}/md_output/10_prod.nc')
    script = script.replace('${TOP}', f'{wsl_dir}/{prmtop}')
    script = script.replace('${OUTDIR}', f'{wsl_dir}/analysis')

    path = os.path.join(work_dir, 'analyze_mmpbsa.sh')
    _write_file(path, script)
    return path


def generate_water_analysis(work_dir, prmtop, ligand_name, receptor_range, hem_resid):
    """Generate water analysis scripts (watershell, bridges, closest)."""
    wsl_dir = _win_to_wsl(work_dir)
    water_dir = os.path.join(work_dir, 'water_analysis')
    os.makedirs(water_dir, exist_ok=True)

    rec_parts = receptor_range.split('-')
    first_res = rec_parts[0]
    last_res = rec_parts[1] if len(rec_parts) == 2 else rec_parts[0]

    # Active site mask for watershell
    if hem_resid:
        activesite_mask = f':{ligand_name}|(:{hem_resid}@O1)'
    else:
        activesite_mask = f':{ligand_name}'

    script = f"""#!/bin/bash
# Water Analysis: watershell, water bridges, closest waters
# Auto-generated by zhuangzhou.amber_md.ligand_analysis
source ~/amber25/amber.sh

WORKDIR="{wsl_dir}"
TOP="{wsl_dir}/{prmtop}"
TRAJ="{wsl_dir}/md_output/10_prod.nc"
OUTDIR="{wsl_dir}/water_analysis"

mkdir -p ${{OUTDIR}}
cd ${{OUTDIR}}

# ============================================================
# 1. Watershell: solvation shells around ligand
# ============================================================
echo "=== Step 1: Watershell analysis ==="
cpptraj -p ${{TOP}} <<'EOF'
trajin {wsl_dir}/md_output/10_prod.nc

# First shell (3.4 A) and second shell (5.0 A)
watershell :{ligand_name} out watershell_lig.dat lower 3.4 upper 5.0

# Active site watershell (ligand + heme O1)
watershell {activesite_mask} out watershell_activesite.dat lower 3.4 upper 5.0

go
quit
EOF

# ============================================================
# 2. Water-bridged H-bonds
# ============================================================
echo ""
echo "=== Step 2: Water-bridged H-bond analysis ==="
cpptraj -p ${{TOP}} <<'EOF2'
trajin {wsl_dir}/md_output/10_prod.nc

# Protein as donor, ligand as acceptor, water as bridge
hbond WatBridge1 \\
  donormask :{first_res}-{last_res} acceptormask :{ligand_name} \\
  solventdonor :WAT solventacceptor :WAT \\
  bridgeout bridge_prot_lig.dat \\
  out watbridge_prot_lig_time.dat \\
  avgout watbridge_prot_lig_avg.dat \\
  dist 3.5 angle 135

# Ligand as donor, protein as acceptor, water as bridge
hbond WatBridge2 \\
  donormask :{ligand_name} acceptormask :{first_res}-{last_res} \\
  solventdonor :WAT solventacceptor :WAT \\
  bridgeout bridge_lig_prot.dat \\
  out watbridge_lig_prot_time.dat \\
  avgout watbridge_lig_prot_avg.dat \\
  dist 3.5 angle 135

# Direct ligand-water H-bonds
hbond LW1 donormask :{ligand_name} acceptormask :WAT \\
  out hbond_lig_wat_don.dat avgout hbond_lig_wat_don_avg.dat dist 3.5 angle 135
hbond LW2 donormask :WAT acceptormask :{ligand_name} \\
  out hbond_wat_lig_acc.dat avgout hbond_wat_lig_acc_avg.dat dist 3.5 angle 135

go
quit
EOF2

# ============================================================
# 3. Closest waters to ligand
# ============================================================
echo ""
echo "=== Step 3: Closest water analysis ==="
cpptraj -p ${{TOP}} <<'EOF3'
trajin {wsl_dir}/md_output/10_prod.nc

closest 10 :{ligand_name} closestout closest_waters.dat \\
  outprefix closest_wat

go
quit
EOF3

# ============================================================
# 4. Extract hydrated frames for PLIP
# ============================================================
echo ""
echo "=== Step 4: Extract frames with active-site waters ==="

for frame in 1500 2000 2800; do
    cpptraj -p ${{TOP}} <<PLIP
trajin {wsl_dir}/md_output/10_prod.nc ${{frame}} ${{frame}}
strip :Na+,Cl-
closest 30 :{ligand_name}
trajout plip_with_water_${{frame}}.pdb pdb
go
quit
PLIP
done

echo ""
echo "=== Step 5: Run PLIP on hydrated frames ==="
for pdb in plip_with_water_*.pdb; do
    name=$(basename ${{pdb}} .pdb)
    echo "  PLIP: ${{pdb}}"
    plip -f ${{pdb}} -o ${{name}}_plip -x 2>/dev/null || echo "  (PLIP failed for ${{pdb}})"
done

echo ""
echo "=== Water analysis complete ==="
"""
    path = os.path.join(work_dir, 'analyze_water.sh')
    _write_file(path, script)
    return path


def _generate_plot_scripts(work_dir, ligand_name, receptor_range, hem_resid,
                           axial_cys_resid, binding_residues, modules, prod_ns):
    """Generate comprehensive Python plotting script."""
    analysis_dir = os.path.join(work_dir, 'analysis')
    os.makedirs(analysis_dir, exist_ok=True)

    # Time conversion factor
    dt_ns = 0.01  # frame_number * dt_ns = time in ns

    rec_parts = receptor_range.split('-')
    last_res = int(rec_parts[1]) if len(rec_parts) == 2 else 512

    # Build binding residue list for plot
    binding_res_str = repr(binding_residues) if binding_residues else '[]'

    script = f"""#!/usr/bin/env python3
\"\"\"Comprehensive MD analysis plots. Auto-generated by zhuangzhou.amber_md.\"\"\"
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os

ANALYSIS_DIR = "{_win_to_wsl(analysis_dir)}"
WATER_DIR = "{_win_to_wsl(os.path.join(work_dir, 'water_analysis'))}"
DT_NS = {dt_ns}
LIGAND = "{ligand_name}"
BINDING_RESIDUES = {binding_res_str}

# Use local paths if running on Windows
if os.name == 'nt':
    ANALYSIS_DIR = r"{analysis_dir}"
    WATER_DIR = r"{os.path.join(work_dir, 'water_analysis')}"

def load_dat(path):
    \"\"\"Load cpptraj .dat file (skip header, return time in ns + values).\"\"\"
    d = np.loadtxt(path, skiprows=1)
    return d[:, 0] * DT_NS, d[:, 1]

def savefig(fig, name, directory=None):
    d = directory or ANALYSIS_DIR
    fig.savefig(os.path.join(d, name + '.png'), dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved {{name}}.png")

# ================================================================
# MODULE 1: Structural Analysis
# ================================================================
print("\\n=== Module 1: Structural Plots ===")

# Combined RMSD
rmsd_files = [
    ('protein_rmsd.dat', 'Protein CA', 'k'),
    ('lig_rmsd.dat', f'{{LIGAND}} (heavy)', 'C0'),
    ('heme_rmsd.dat', 'Heme (heavy)', 'r'),
]

fig, ax = plt.subplots(figsize=(10, 5))
for fname, label, color in rmsd_files:
    fpath = os.path.join(ANALYSIS_DIR, fname)
    if os.path.exists(fpath):
        t, v = load_dat(fpath)
        ax.plot(t, v, color=color, alpha=0.7, lw=0.6, label=label)
ax.set(xlabel='Time (ns)', ylabel='RMSD (\\u00c5)', title='RMSD Time Series')
ax.legend()
savefig(fig, 'combined_rmsd')

# RMSF by residue
rmsf_path = os.path.join(ANALYSIS_DIR, 'rmsf_ca.dat')
if os.path.exists(rmsf_path):
    fig, ax = plt.subplots(figsize=(12, 4))
    rmsf = np.loadtxt(rmsf_path, skiprows=1)
    ax.plot(rmsf[:, 0], rmsf[:, 1], 'c-', lw=0.8)
    # Highlight binding residues
    import re
    for res_str in BINDING_RESIDUES:
        m = re.match(r'([A-Z]+)(\\d+)', res_str)
        if m:
            resid = int(m.group(2))
            ax.axvline(resid, color='r', ls='--', alpha=0.3, lw=0.8)
    ax.set(xlabel='Residue', ylabel='RMSF (\\u00c5)', title='C\\u03b1 RMSF by Residue')
    savefig(fig, 'rmsf_by_residue')

# Radius of gyration
rog_path = os.path.join(ANALYSIS_DIR, 'rog.dat')
if os.path.exists(rog_path):
    fig, ax = plt.subplots(figsize=(8, 4))
    t, v = load_dat(rog_path)
    ax.plot(t, v, 'm-', alpha=0.7, lw=0.6)
    ax.set(xlabel='Time (ns)', ylabel='Rg (\\u00c5)', title='Radius of Gyration')
    savefig(fig, 'radius_of_gyration')

# Key distances (Fe-related)
fe_dist_files = [
    ('fe_s_dist.dat', 'Fe-S(Cys)', 2.565),
    ('lig_fe_dist.dat', f'{{LIGAND}}-Fe', None),
]
for fname, label, ref in fe_dist_files:
    fpath = os.path.join(ANALYSIS_DIR, fname)
    if os.path.exists(fpath):
        fig, ax = plt.subplots(figsize=(8, 4))
        t, v = load_dat(fpath)
        ax.plot(t, v, alpha=0.7, lw=0.6)
        if ref:
            ax.axhline(ref, color='r', ls='--', lw=1.2, label=f'Ref: {{ref}} \\u00c5')
        ax.set(xlabel='Time (ns)', ylabel='Distance (\\u00c5)', title=f'{{label}} Distance')
        ax.legend()
        savefig(fig, fname.replace('.dat', ''))

# ================================================================
# MODULE 2: Interaction Analysis
# ================================================================
print("\\n=== Module 2: Interaction Plots ===")

# Binding residue distances
res_dist_data = []
for fname in sorted(os.listdir(ANALYSIS_DIR)):
    if fname.startswith('dist_') and fname.endswith('.dat'):
        fpath = os.path.join(ANALYSIS_DIR, fname)
        try:
            t, v = load_dat(fpath)
            label = fname.replace('dist_', '').replace('.dat', '')
            res_dist_data.append((label, t, v))
        except Exception:
            pass

if res_dist_data:
    fig, ax = plt.subplots(figsize=(10, 5))
    for label, t, v in res_dist_data:
        ax.plot(t, v, alpha=0.5, lw=0.5, label=label)
    ax.set(xlabel='Time (ns)', ylabel='Distance (\\u00c5)',
           title=f'Binding Pocket Residue Distances to {{LIGAND}}')
    ax.legend(fontsize=8, ncol=2)
    savefig(fig, 'key_residue_distances')

    # Bar chart of mean distances
    fig, ax = plt.subplots(figsize=(8, 5))
    labels = [d[0] for d in res_dist_data]
    means = [np.mean(d[2]) for d in res_dist_data]
    stds = [np.std(d[2]) for d in res_dist_data]
    idx = np.argsort(means)
    ax.barh([labels[i] for i in idx], [means[i] for i in idx],
            xerr=[stds[i] for i in idx], color='steelblue', alpha=0.8)
    ax.set(xlabel='Mean Distance (\\u00c5)', title='Binding Pocket Summary')
    ax.axvline(4.0, color='r', ls='--', alpha=0.5, label='Contact cutoff (4 \\u00c5)')
    ax.legend()
    savefig(fig, 'binding_pocket_summary')

# H-bond time series
for hb_file, label in [('hbond_prot_lig.dat', 'Protein\\u2192Ligand'),
                        ('hbond_lig_prot.dat', 'Ligand\\u2192Protein')]:
    fpath = os.path.join(ANALYSIS_DIR, hb_file)
    if os.path.exists(fpath):
        try:
            data = np.loadtxt(fpath, skiprows=1)
            if data.ndim == 2 and data.shape[1] >= 2:
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4),
                                                gridspec_kw={{'width_ratios': [3, 1]}})
                t = data[:, 0] * DT_NS
                # Total H-bonds = sum of all H-bond columns
                hb_count = np.sum(data[:, 1:], axis=1) if data.shape[1] > 2 else data[:, 1]
                ax1.plot(t, hb_count, 'b-', alpha=0.5, lw=0.5)
                ax1.set(xlabel='Time (ns)', ylabel='H-bond Count', title=f'H-bonds: {{label}}')
                ax2.hist(hb_count, bins=range(int(hb_count.max()) + 2),
                        orientation='horizontal', color='steelblue', alpha=0.8)
                ax2.set(xlabel='Frequency', ylabel='H-bond Count')
                plt.tight_layout()
                savefig(fig, hb_file.replace('.dat', ''))
        except Exception:
            pass

# ================================================================
# MODULE 3: MM-GBSA Plots
# ================================================================
print("\\n=== Module 3: MM-GBSA Plots ===")

mmpbsa_results = os.path.join(ANALYSIS_DIR, 'mmpbsa_results.dat')
if os.path.exists(mmpbsa_results):
    # Parse energy components from results file
    components = {{}}
    with open(mmpbsa_results, 'r') as f:
        for line in f:
            line = line.strip()
            for key in ['VDWAALS', 'EEL', 'EGB', 'ESURF', 'DELTA G gas',
                        'DELTA G solv', 'DELTA TOTAL']:
                if line.startswith(key):
                    parts = line.split()
                    try:
                        components[key] = (float(parts[-3]), float(parts[-1]))
                    except (ValueError, IndexError):
                        pass
    if components:
        fig, ax = plt.subplots(figsize=(10, 6))
        names = list(components.keys())
        means = [components[k][0] for k in names]
        stds = [components[k][1] for k in names]
        colors = ['#2196F3' if m < 0 else '#F44336' for m in means]
        ax.bar(range(len(names)), means, yerr=stds, color=colors, alpha=0.8)
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels(names, rotation=45, ha='right')
        ax.set(ylabel='Energy (kcal/mol)', title='MM-GBSA Energy Components')
        ax.axhline(0, color='k', lw=0.5)
        plt.tight_layout()
        savefig(fig, 'mmpbsa_components')

# Per-frame energy
per_frame = os.path.join(ANALYSIS_DIR, 'mmpbsa_per_frame.csv')
if os.path.exists(per_frame):
    try:
        data = np.loadtxt(per_frame, delimiter=',', skiprows=1)
        if data.ndim == 2:
            frames = data[:, 0]
            dg = data[:, -1]  # Last column = DELTA TOTAL
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5),
                                            gridspec_kw={{'width_ratios': [3, 1]}})
            ax1.plot(frames * DT_NS, dg, 'b-', alpha=0.5, lw=0.5)
            ax1.axhline(np.mean(dg), color='r', ls='--', lw=1.5,
                       label=f'Mean: {{np.mean(dg):.1f}} kcal/mol')
            ax1.set(xlabel='Time (ns)', ylabel='\\u0394G (kcal/mol)',
                   title='Per-Frame Binding Free Energy')
            ax1.legend()

            ax2.hist(dg, bins=30, orientation='horizontal', color='steelblue', alpha=0.8)
            ax2.axhline(np.mean(dg), color='r', ls='--', lw=1.5)
            ax2.set(xlabel='Frequency', ylabel='\\u0394G (kcal/mol)')
            plt.tight_layout()
            savefig(fig, 'mmpbsa_per_frame')
    except Exception as e:
        print(f"  Could not plot per-frame energy: {{e}}")

# ================================================================
# MODULE 4: Water Analysis Plots
# ================================================================
print("\\n=== Module 4: Water Plots ===")

# Watershell time series
for ws_file in ['watershell_lig.dat', 'watershell_activesite.dat']:
    fpath = os.path.join(WATER_DIR, ws_file)
    if os.path.exists(fpath):
        try:
            data = np.loadtxt(fpath, skiprows=1)
            fig, ax = plt.subplots(figsize=(10, 4))
            t = data[:, 0] * DT_NS
            ax.plot(t, data[:, 1], 'b-', alpha=0.5, lw=0.5, label='1st shell (3.4 \\u00c5)')
            if data.shape[1] > 2:
                ax.plot(t, data[:, 2], 'r-', alpha=0.5, lw=0.5, label='2nd shell (5.0 \\u00c5)')
            ax.set(xlabel='Time (ns)', ylabel='Water Count',
                   title=f'Solvation Shell: {{ws_file.replace(".dat", "")}}')
            ax.legend()
            savefig(fig, ws_file.replace('.dat', ''), WATER_DIR)
        except Exception:
            pass

# Closest water distances
cw_path = os.path.join(WATER_DIR, 'closest_waters.dat')
if os.path.exists(cw_path):
    try:
        data = np.loadtxt(cw_path, skiprows=1)
        n_waters = 10
        n_frames = len(data) // n_waters
        fig, ax = plt.subplots(figsize=(10, 5))
        for rank in range(min(5, n_waters)):
            dists = data[rank::n_waters, 2][:n_frames]
            t = np.arange(len(dists)) * DT_NS
            ax.plot(t, dists, alpha=0.5, lw=0.5, label=f'Water #{{rank+1}}')
        ax.set(xlabel='Time (ns)', ylabel='Distance (\\u00c5)',
               title='Closest Waters to Ligand')
        ax.legend()
        savefig(fig, 'closest_waters', WATER_DIR)
    except Exception:
        pass

print("\\n=== All plots generated ===")
"""
    path = os.path.join(work_dir, 'plot_analysis.py')
    _write_file(path, script)
    return path
