"""
GROMACS MD Analysis — RMSD, RMSF, FEL, H-bonds, distances, MM-PBSA.

All analysis functions work directly with gmx commands and produce
publication-quality matplotlib figures.
"""

import logging
import os
import subprocess
import numpy as np

from .config import GMX_BIN, TIMEOUT_SETUP

logger = logging.getLogger(__name__)


def _find_nonwater_group(work_dir, tpr='md.tpr'):
    """Find the 'non-Water' index group number (varies by system)."""
    groups = _parse_index_groups(work_dir, tpr)
    return groups.get('non-Water', '18')


def _parse_index_groups(work_dir, tpr='md.tpr'):
    """Parse all index group names→numbers from gmx make_ndx output.

    Returns dict like {'Protein': '1', 'Backbone': '4', 'non-Water': '18', ...}
    """
    import re
    out, _ = _run_gmx(f'make_ndx -f {tpr}', work_dir, input_text='q\n')
    groups = {}
    for line in out.splitlines():
        m = re.match(r'\s*(\d+)\s+([\w\-]+)', line)
        if m:
            groups[m.group(2)] = m.group(1)
    return groups


def _find_group(groups, *candidates, fallback='0'):
    """Find group number by trying candidate names in order."""
    for name in candidates:
        if name in groups:
            return groups[name]
    return fallback


def _run_gmx(cmd, work_dir, input_text=None, timeout=TIMEOUT_SETUP):
    """Run gmx command, return (output, success)."""
    full_cmd = f'"{GMX_BIN}" {cmd}'
    try:
        result = subprocess.run(
            full_cmd, shell=True, capture_output=True, text=True,
            input=input_text, timeout=timeout, cwd=work_dir,
        )
        output = (result.stdout or '') + (result.stderr or '')
        return output, result.returncode == 0
    except Exception as e:
        return str(e), False


def _parse_xvg(xvg_path):
    """Parse GROMACS .xvg file → (x_array, y_arrays)."""
    x, ys = [], []
    with open(xvg_path, encoding='utf-8', errors='replace') as f:
        for line in f:
            if line.startswith(('#', '@')):
                continue
            parts = line.split()
            if len(parts) >= 2:
                x.append(float(parts[0]))
                ys.append([float(v) for v in parts[1:]])
    if not x:
        return np.array([]), np.empty((0, 0))
    return np.array(x), np.array(ys)


def compute_rmsd(work_dir, tpr='md.tpr', xtc='md.xtc'):
    """Compute protein backbone RMSD and ligand RMSD.
    Returns dict with xvg paths.
    """
    results = {}
    groups = _parse_index_groups(work_dir, tpr)
    bb_grp = _find_group(groups, 'Backbone', 'MainChain', fallback='4')

    # Protein backbone RMSD
    out, ok = _run_gmx(
        f'rms -s {tpr} -f {xtc} -o rmsd_backbone.xvg -tu ns',
        work_dir, input_text=f'{bb_grp}\n{bb_grp}\n')
    if ok:
        results['backbone'] = os.path.join(work_dir, 'rmsd_backbone.xvg')

    # Ligand RMSD — try common ligand group names
    lig_grp = _find_group(groups, 'MOL', 'LIG', 'UNK', 'Other', fallback=None)
    if lig_grp:
        out, ok = _run_gmx(
            f'rms -s {tpr} -f {xtc} -o rmsd_ligand.xvg -tu ns',
            work_dir, input_text=f'{bb_grp}\n{lig_grp}\n')
        if ok:
            results['ligand'] = os.path.join(work_dir, 'rmsd_ligand.xvg')

    # Protein RMSD vs crystal (using em.tpr as reference)
    if os.path.exists(os.path.join(work_dir, 'em.tpr')):
        out, ok = _run_gmx(
            f'rms -s em.tpr -f {xtc} -o rmsd_vs_xtal.xvg -tu ns',
            work_dir, input_text=f'{bb_grp}\n{bb_grp}\n')
        if ok:
            results['vs_crystal'] = os.path.join(work_dir, 'rmsd_vs_xtal.xvg')

    return results


def compute_rmsf(work_dir, tpr='md.tpr', xtc='md.xtc'):
    """Compute per-residue RMSF. Returns xvg path."""
    groups = _parse_index_groups(work_dir, tpr)
    prot_grp = _find_group(groups, 'Protein', 'Protein-H', fallback='1')
    out, ok = _run_gmx(
        f'rmsf -s {tpr} -f {xtc} -o rmsf.xvg -res',
        work_dir, input_text=f'{prot_grp}\n')
    if ok:
        return os.path.join(work_dir, 'rmsf.xvg')
    return None


def compute_gyration(work_dir, tpr='md.tpr', xtc='md.xtc'):
    """Compute radius of gyration. Returns xvg path."""
    groups = _parse_index_groups(work_dir, tpr)
    prot_grp = _find_group(groups, 'Protein', 'Protein-H', fallback='1')
    out, ok = _run_gmx(
        f'gyrate -s {tpr} -f {xtc} -o gyrate.xvg',
        work_dir, input_text=f'{prot_grp}\n')
    if ok:
        return os.path.join(work_dir, 'gyrate.xvg')
    return None


def compute_hbonds(work_dir, tpr='md.tpr', xtc='md.xtc', ndx='index.ndx'):
    """Compute H-bonds between protein and ligand over time.
    Returns dict with xvg path and average count.
    """
    groups = _parse_index_groups(work_dir, tpr)
    prot_grp = _find_group(groups, 'Protein', fallback='1')
    # Try ligand group names: custom index groups first, then default groups
    lig_grp = _find_group(groups, 'MOL', 'LIG', 'UNK', 'Other', fallback=None)
    if not lig_grp:
        return {}  # No ligand group found, skip H-bond analysis
    ndx_flag = f'-n {ndx}' if os.path.exists(os.path.join(work_dir, ndx)) else ''
    out, ok = _run_gmx(
        f'hbond -s {tpr} -f {xtc} {ndx_flag} -num hbond_prot_lig.xvg',
        work_dir, input_text=f'{prot_grp}\n{lig_grp}\n')
    result = {}
    if ok:
        xvg = os.path.join(work_dir, 'hbond_prot_lig.xvg')
        result['xvg'] = xvg
        if os.path.exists(xvg):
            t, y = _parse_xvg(xvg)
            if len(y) > 0:
                result['mean_hbonds'] = float(np.mean(y[:, 0]))
                result['max_hbonds'] = float(np.max(y[:, 0]))
    return result


def compute_distances(work_dir, tpr='md.tpr', xtc='md.xtc',
                      residues=None, ligand_resname='MOL'):
    """Compute minimum distance between ligand and key residues over time.

    Args:
        residues: list of residue identifiers like ['ASP10', 'LEU151', 'PHE15']
    Returns: dict mapping residue → xvg path
    """
    if not residues:
        return {}

    results = {}
    ndx_path = os.path.join(work_dir, 'dist_groups.ndx')

    for res in residues:
        # Parse residue: e.g. ASP10 → resname ASP, resid 10
        import re
        m = re.match(r'([A-Z]+)(\d+)', res)
        if not m:
            continue
        resname, resid = m.group(1), m.group(2)

        # Create selection-based distance calculation
        out_xvg = f'dist_{res}.xvg'
        # Use gmx pairdist with selections
        out, ok = _run_gmx(
            f'pairdist -s {tpr} -f {xtc} -o {out_xvg} -tu ns '
            f'-ref "resname {ligand_resname}" '
            f'-sel "resid {resid} and name CA"',
            work_dir)

        if ok:
            results[res] = os.path.join(work_dir, out_xvg)
        else:
            # Fallback: use mindist with custom index
            _make_residue_ndx(work_dir, tpr, resid, ligand_resname, res)
            out, ok = _run_gmx(
                f'mindist -s {tpr} -f {xtc} -n dist_{res}.ndx -od dist_{res}.xvg -tu ns',
                work_dir, input_text='0\n1\n')
            if ok:
                results[res] = os.path.join(work_dir, f'dist_{res}.xvg')

    return results


def _make_residue_ndx(work_dir, tpr, resid, ligand_resname, label):
    """Create index file with ligand and specific residue groups."""
    ndx_path = os.path.join(work_dir, f'dist_{label}.ndx')
    # Use gmx select to create custom groups
    _run_gmx(
        f'select -s {tpr} -on {ndx_path} '
        f'-select "resname {ligand_resname}" "resid {resid}"',
        work_dir)


def compute_coordination(work_dir, tpr='md.tpr', xtc='md.xtc'):
    """Compute metal-coordination distances for metalloproteins.

    Auto-detects HEME and other metal cofactors, measures coordination distances.
    Returns dict with xvg paths and statistics, or empty dict if no metal found.
    """
    gro_path = os.path.join(work_dir, 'md.gro')
    if not os.path.exists(gro_path):
        return {}

    # Detect which metal cofactors are present
    _HEME_NAMES = {'HEM', 'HEME', 'HEC', 'HEA', 'HEO2'}
    _METAL_ATOMS = {'FE', 'ZN', 'MN', 'CU', 'CO', 'NI', 'MG'}
    found_cofactors = set()
    found_metals = set()
    with open(gro_path, encoding='utf-8', errors='replace') as f:
        for line in f:
            if len(line) < 30:
                continue
            rn = line[5:10].strip().upper()
            atom = line[10:15].strip().upper()
            if rn in _HEME_NAMES:
                found_cofactors.add('HEME')
            if atom in _METAL_ATOMS:
                found_metals.add(atom)
    if not found_cofactors and not found_metals:
        return {}

    results = {}

    # HEME coordination pairs
    if 'HEME' in found_cofactors:
        hem_sel = '(resname HEM or resname HEME or resname HEC or resname HEA) and name FE'
        pairs = [
            ('Fe_CysSG', hem_sel, '(resname CYS or resname CYS2 or resname CYM) and name SG', 'Fe-Cys(SG)'),
            ('Fe_HisNE2', hem_sel, '(resname HIS or resname HSD or resname HSE or resname HID or resname HIE) and name NE2', 'Fe-His(NE2)'),
            ('Fe_HisND1', hem_sel, '(resname HIS or resname HSD or resname HSE or resname HID or resname HIE) and name ND1', 'Fe-His(ND1)'),
            ('Fe_MetSD', hem_sel, 'resname MET and name SD', 'Fe-Met(SD)'),
            ('Fe_TyrOH', hem_sel, 'resname TYR and name OH', 'Fe-Tyr(OH)'),
        ]
        for tag, ref_sel, sel, label in pairs:
            out_xvg = f'dist_{tag}.xvg'
            out, ok = _run_gmx(
                f'pairdist -s {tpr} -f {xtc} -o {out_xvg} -tu ns '
                f'-ref "{ref_sel}" -sel "{sel}" -type min',
                work_dir)
            xvg_path = os.path.join(work_dir, out_xvg)
            if ok and os.path.exists(xvg_path):
                t, y = _parse_xvg(xvg_path)
                if len(y) > 0:
                    d = y[:, 0] * 10  # nm -> A
                    results[tag] = {
                        'xvg': xvg_path, 'label': label,
                        'mean': float(np.mean(d)), 'std': float(np.std(d)),
                        'min': float(np.min(d)), 'max': float(np.max(d)),
                    }

    # Generic metal coordination (Zn, Cu, Mn, etc.)
    for metal in (found_metals - {'FE'} if 'HEME' in found_cofactors else found_metals):
        metal_sel = f'name {metal}'
        coord_residues = [
            (f'{metal}_CysSG', metal_sel, '(resname CYS or resname CYS2) and name SG', f'{metal}-Cys(SG)'),
            (f'{metal}_HisNE2', metal_sel, '(resname HIS or resname HSD or resname HSE) and name NE2', f'{metal}-His(NE2)'),
            (f'{metal}_AspOD', metal_sel, 'resname ASP and name OD1 OD2', f'{metal}-Asp(OD)'),
            (f'{metal}_GluOE', metal_sel, 'resname GLU and name OE1 OE2', f'{metal}-Glu(OE)'),
        ]
        for tag, ref_sel, sel, label in coord_residues:
            out_xvg = f'dist_{tag}.xvg'
            out, ok = _run_gmx(
                f'pairdist -s {tpr} -f {xtc} -o {out_xvg} -tu ns '
                f'-ref "{ref_sel}" -sel "{sel}" -type min',
                work_dir)
            xvg_path = os.path.join(work_dir, out_xvg)
            if ok and os.path.exists(xvg_path):
                t, y = _parse_xvg(xvg_path)
                if len(y) > 0:
                    d = y[:, 0] * 10
                    results[tag] = {
                        'xvg': xvg_path, 'label': label,
                        'mean': float(np.mean(d)), 'std': float(np.std(d)),
                        'min': float(np.min(d)), 'max': float(np.max(d)),
                    }

    # Protein-cofactor H-bonds (dynamic group detection)
    groups = _parse_index_groups(work_dir, tpr)
    prot_grp = _find_group(groups, 'Protein', fallback='1')
    # Try cofactor/ligand group
    cof_grp = _find_group(groups, 'HEME', 'HEM', 'MOL', 'LIG', 'Other', fallback=None)
    if cof_grp:
        out, ok = _run_gmx(
            f'hbond -s {tpr} -f {xtc} -num hbond_prot_hem.xvg',
            work_dir, input_text=f'{prot_grp}\n{cof_grp}\n')
        hb_path = os.path.join(work_dir, 'hbond_prot_hem.xvg')
        if ok and os.path.exists(hb_path):
            t, y = _parse_xvg(hb_path)
            if len(y) > 0:
                results['hbond_prot_hem'] = {
                    'xvg': hb_path, 'label': 'Protein-Cofactor H-bonds',
                    'mean': float(np.mean(y[:, 0])), 'std': float(np.std(y[:, 0])),
                }

    return results


def compute_energy(work_dir, edr='md.edr'):
    """Extract potential energy over time. Returns xvg path."""
    # Detect 'Potential' energy group number dynamically
    energy_input = 'Potential\n\n'  # GROMACS accepts name or number
    out, ok = _run_gmx(
        f'energy -f {edr} -o energy.xvg',
        work_dir, input_text=energy_input)
    if ok:
        return os.path.join(work_dir, 'energy.xvg')
    # Fallback: try group 10 (common default for Potential)
    out, ok = _run_gmx(
        f'energy -f {edr} -o energy.xvg',
        work_dir, input_text='10\n0\n')
    if ok:
        return os.path.join(work_dir, 'energy.xvg')
    return None


def compute_fel(work_dir, rmsd_xvg=None, rg_xvg=None, temp=300, bins=100):
    """Compute Free Energy Landscape from RMSD and Rg.

    FEL = -kT * ln(P(RMSD, Rg))
    Returns: path to FEL data file and figure.
    """
    if not rmsd_xvg or not rg_xvg:
        return None

    t_rmsd, y_rmsd = _parse_xvg(rmsd_xvg)
    t_rg, y_rg = _parse_xvg(rg_xvg)

    # Align lengths (use shorter)
    n = min(len(t_rmsd), len(t_rg))
    rmsd = y_rmsd[:n, 0]  # nm
    rg = y_rg[:n, 0]      # nm

    # 2D histogram → free energy
    H, xedges, yedges = np.histogram2d(rmsd, rg, bins=bins)
    H = H.T  # transpose for correct orientation

    # Convert to free energy: G = -kT * ln(P)
    kT = 8.314e-3 * temp  # kJ/mol
    # Avoid log(0)
    H_nonzero = np.where(H > 0, H, np.nan)
    G = -kT * np.log(H_nonzero / np.nanmax(H_nonzero))
    G = G - np.nanmin(G)  # shift minimum to 0

    # Save data
    fel_data = os.path.join(work_dir, 'fel_data.npz')
    np.savez(fel_data, rmsd_edges=xedges, rg_edges=yedges, G=G,
             rmsd=rmsd, rg=rg)
    return fel_data


def plot_coordination(work_dir, coord_results):
    """Generate metalloprotein coordination analysis figure.

    Args:
        coord_results: dict from compute_coordination()
    Returns:
        path to saved figure
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    dist_items = {k: v for k, v in coord_results.items()
                  if k != 'hbond_prot_hem' and 'xvg' in v}
    hb_item = coord_results.get('hbond_prot_hem')
    n_panels = len(dist_items) + (1 if dist_items else 0) + (1 if hb_item else 0)
    n_panels = max(n_panels, 2)

    fig, axes = plt.subplots(1, n_panels, figsize=(6 * n_panels, 5))
    if n_panels == 1:
        axes = [axes]
    fig.suptitle('Metalloprotein Coordination Analysis', fontsize=14, fontweight='bold')

    colors = ['#e41a1c', '#377eb8', '#4daf4a', '#984ea3']
    idx = 0

    # Time series for each coordination pair
    for i, (tag, info) in enumerate(dist_items.items()):
        ax = axes[idx]; idx += 1
        t, y = _parse_xvg(info['xvg'])
        d = y[:, 0] * 10
        ax.plot(t, d, color=colors[i % len(colors)], alpha=0.6, linewidth=0.5)
        ax.axhline(info['mean'], color='red', ls='--', alpha=0.6,
                    label=f"Mean: {info['mean']:.2f} A")
        ax.set_xlabel('Time (ns)')
        ax.set_ylabel('Distance (A)')
        ax.set_title(info['label'])
        ax.legend(fontsize=8)

    # Distribution overlay
    if dist_items:
        ax = axes[idx]; idx += 1
        for i, (tag, info) in enumerate(dist_items.items()):
            t, y = _parse_xvg(info['xvg'])
            ax.hist(y[:, 0] * 10, bins=60, alpha=0.5, color=colors[i % len(colors)],
                    label=info['label'], density=True)
        ax.axvline(2.3, color='gray', ls=':', alpha=0.6, label='Fe-S (2.3 A)')
        ax.axvline(2.1, color='gray', ls='-.', alpha=0.6, label='Fe-N (2.1 A)')
        ax.set_xlabel('Distance (A)')
        ax.set_ylabel('Density')
        ax.set_title('Distance Distributions')
        ax.legend(fontsize=7)

    # H-bonds
    if hb_item and idx < len(axes):
        ax = axes[idx]
        t, y = _parse_xvg(hb_item['xvg'])
        ax.plot(t / 1000, y[:, 0], 'purple', alpha=0.6, linewidth=0.5)
        ax.axhline(hb_item['mean'], color='red', ls='--',
                    label=f"Mean: {hb_item['mean']:.1f}")
        ax.set_xlabel('Time (ns)')
        ax.set_ylabel('H-bonds')
        ax.set_title('Protein-HEME H-bonds')
        ax.legend(fontsize=8)

    fig_path = os.path.join(work_dir, 'coordination_analysis.png')
    fig.savefig(fig_path, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    logger.info("Coordination figure saved: %s", fig_path)
    return fig_path


def plot_all(work_dir, key_residues=None, ligand_resname='MOL'):
    """Generate all analysis plots as a single multi-panel figure.

    Returns path to saved figure.
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec

    fig = plt.figure(figsize=(20, 24))
    gs = GridSpec(4, 2, figure=fig, hspace=0.3, wspace=0.3)
    fig.suptitle('MD Simulation Analysis', fontsize=16, fontweight='bold', y=0.98)

    # ── Panel 1: RMSD ──────────────────────────────────────────────────
    ax1 = fig.add_subplot(gs[0, 0])
    rmsd_bb = os.path.join(work_dir, 'rmsd_backbone.xvg')
    rmsd_lig = os.path.join(work_dir, 'rmsd_ligand.xvg')
    if os.path.exists(rmsd_bb):
        t, y = _parse_xvg(rmsd_bb)
        ax1.plot(t, y[:, 0] * 10, 'b-', alpha=0.7, linewidth=0.5, label='Backbone')
    if os.path.exists(rmsd_lig):
        t, y = _parse_xvg(rmsd_lig)
        ax1.plot(t, y[:, 0] * 10, 'r-', alpha=0.7, linewidth=0.5, label='Ligand')
    ax1.set_xlabel('Time (ns)')
    ax1.set_ylabel(u'RMSD (\u00c5)')
    ax1.set_title('RMSD')
    ax1.legend()
    ax1.set_xlim(left=0)

    # ── Panel 2: RMSF ──────────────────────────────────────────────────
    ax2 = fig.add_subplot(gs[0, 1])
    rmsf = os.path.join(work_dir, 'rmsf.xvg')
    if os.path.exists(rmsf):
        t, y = _parse_xvg(rmsf)
        ax2.plot(t, y[:, 0] * 10, 'g-', linewidth=0.8)
        if key_residues:
            import re
            for res in key_residues:
                m = re.match(r'[A-Z]+(\d+)', res)
                if m:
                    rid = int(m.group(1))
                    idx = np.argmin(np.abs(t - rid))
                    if idx < len(y):
                        ax2.axvline(x=t[idx], color='red', alpha=0.3, linestyle='--')
                        ax2.annotate(res, (t[idx], y[idx, 0]*10),
                                    fontsize=7, color='red')
    ax2.set_xlabel('Residue')
    ax2.set_ylabel(u'RMSF (\u00c5)')
    ax2.set_title('Per-Residue RMSF')

    # ── Panel 3: H-bonds ───────────────────────────────────────────────
    ax3 = fig.add_subplot(gs[1, 0])
    hbond_xvg = os.path.join(work_dir, 'hbond_prot_lig.xvg')
    if os.path.exists(hbond_xvg):
        t, y = _parse_xvg(hbond_xvg)
        ax3.plot(t / 1000, y[:, 0], 'purple', alpha=0.7, linewidth=0.5)
        ax3.axhline(y=np.mean(y[:, 0]), color='red', linestyle='--',
                    label=f'Mean: {np.mean(y[:, 0]):.1f}')
        ax3.legend()
    ax3.set_xlabel('Time (ns)')
    ax3.set_ylabel('Number of H-bonds')
    ax3.set_title('Protein-Ligand Hydrogen Bonds')

    # ── Panel 4: Radius of Gyration ────────────────────────────────────
    ax4 = fig.add_subplot(gs[1, 1])
    rg_xvg = os.path.join(work_dir, 'gyrate.xvg')
    if os.path.exists(rg_xvg):
        t, y = _parse_xvg(rg_xvg)
        ax4.plot(t / 1000, y[:, 0], 'orange', alpha=0.7, linewidth=0.5)
        ax4.set_ylabel('Rg (nm)')
    ax4.set_xlabel('Time (ns)')
    ax4.set_title('Radius of Gyration')

    # ── Panel 5: Free Energy Landscape ─────────────────────────────────
    ax5 = fig.add_subplot(gs[2, 0])
    fel_data = os.path.join(work_dir, 'fel_data.npz')
    if os.path.exists(fel_data):
        data = np.load(fel_data)
        G = data['G']
        xedges = data['rmsd_edges']
        yedges = data['rg_edges']
        X, Y = np.meshgrid(
            (xedges[:-1] + xedges[1:]) / 2 * 10,  # nm → A
            (yedges[:-1] + yedges[1:]) / 2,
        )
        im = ax5.contourf(X, Y, G, levels=20, cmap='RdYlBu_r')
        plt.colorbar(im, ax=ax5, label='Free Energy (kJ/mol)')
        ax5.set_xlabel(u'RMSD (\u00c5)')
        ax5.set_ylabel('Rg (nm)')
    ax5.set_title('Free Energy Landscape')

    # ── Panel 6: Key Residue Distances ─────────────────────────────────
    ax6 = fig.add_subplot(gs[2, 1])
    colors = ['#e41a1c', '#377eb8', '#4daf4a', '#984ea3', '#ff7f00']
    if key_residues:
        for i, res in enumerate(key_residues):
            dist_xvg = os.path.join(work_dir, f'dist_{res}.xvg')
            if os.path.exists(dist_xvg):
                t, y = _parse_xvg(dist_xvg)
                color = colors[i % len(colors)]
                ax6.plot(t, y[:, 0] * 10, color=color, alpha=0.6,
                        linewidth=0.5, label=f'{res} ({np.mean(y[:, 0])*10:.1f} A)')
        ax6.legend(fontsize=8)
    ax6.set_xlabel('Time (ns)')
    ax6.set_ylabel(u'Min Distance (\u00c5)')
    ax6.set_title('Ligand-Residue Distances')

    # ── Panel 7: Distance distribution ─────────────────────────────────
    ax7 = fig.add_subplot(gs[3, 0])
    if key_residues:
        for i, res in enumerate(key_residues):
            dist_xvg = os.path.join(work_dir, f'dist_{res}.xvg')
            if os.path.exists(dist_xvg):
                t, y = _parse_xvg(dist_xvg)
                color = colors[i % len(colors)]
                ax7.hist(y[:, 0] * 10, bins=50, alpha=0.5, color=color,
                        label=res, density=True)
        ax7.legend(fontsize=8)
    ax7.set_xlabel(u'Distance (\u00c5)')
    ax7.set_ylabel('Density')
    ax7.set_title('Distance Distributions')

    # ── Panel 8: Summary stats ─────────────────────────────────────────
    ax8 = fig.add_subplot(gs[3, 1])
    ax8.axis('off')
    summary_lines = ['Summary Statistics\n']

    # RMSD stats
    if os.path.exists(rmsd_bb):
        t, y = _parse_xvg(rmsd_bb)
        # Use last 50% for equilibrium stats
        eq = y[len(y)//2:, 0] * 10
        summary_lines.append(f'Backbone RMSD (eq): {np.mean(eq):.2f} +/- {np.std(eq):.2f} A')
    if os.path.exists(rmsd_lig):
        t, y = _parse_xvg(rmsd_lig)
        eq = y[len(y)//2:, 0] * 10
        summary_lines.append(f'Ligand RMSD (eq):   {np.mean(eq):.2f} +/- {np.std(eq):.2f} A')

    # H-bond stats
    if os.path.exists(hbond_xvg):
        t, y = _parse_xvg(hbond_xvg)
        summary_lines.append(f'H-bonds (mean):     {np.mean(y[:, 0]):.1f} +/- {np.std(y[:, 0]):.1f}')
        summary_lines.append(f'H-bonds (max):      {np.max(y[:, 0]):.0f}')

    # Rg stats
    if os.path.exists(rg_xvg):
        t, y = _parse_xvg(rg_xvg)
        eq = y[len(y)//2:, 0]
        summary_lines.append(f'Rg (eq):            {np.mean(eq):.3f} +/- {np.std(eq):.3f} nm')

    # Distance stats
    if key_residues:
        summary_lines.append('')
        summary_lines.append('Key Residue Distances (mean, A):')
        for res in key_residues:
            dist_xvg = os.path.join(work_dir, f'dist_{res}.xvg')
            if os.path.exists(dist_xvg):
                t, y = _parse_xvg(dist_xvg)
                summary_lines.append(f'  {res:8s}: {np.mean(y[:, 0])*10:.2f} +/- {np.std(y[:, 0])*10:.2f}')

    ax8.text(0.05, 0.95, '\n'.join(summary_lines),
             transform=ax8.transAxes, fontsize=10, verticalalignment='top',
             fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

    # Save
    fig_path = os.path.join(work_dir, 'md_analysis.png')
    fig.savefig(fig_path, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close(fig)

    logger.info("Analysis figure saved: %s", fig_path)
    return fig_path


def compare_structures(work_dir, tpr='md.tpr', xtc='md.xtc',
                       interval_ns=None):
    """Compare structures at intervals: per-residue RMSD, cofactor displacement, mobile regions.

    Args:
        work_dir: MD simulation directory
        interval_ns: compare every N ns (None = first vs last only)

    Returns:
        dict with per_residue_rmsd, cofactor_displacement, mobile_regions, report
    """
    import re

    results = {}

    # 1. Extract timepoints as PDB (non-water, fitted)
    # Use absolute paths with forward slashes for GROMACS compatibility on Windows
    work_dir = os.path.abspath(work_dir)
    compare_dir = os.path.join(work_dir, 'structure_comparison')
    os.makedirs(compare_dir, exist_ok=True)

    def _fwd(p):
        """Convert path to forward slashes for GROMACS CLI."""
        return p.replace('\\', '/')

    # Get trajectory length from md.log
    total_ps = 0
    log_path = os.path.join(work_dir, 'md.log')
    if os.path.exists(log_path):
        with open(log_path, encoding='utf-8', errors='replace') as f:
            for line in f:
                m = re.search(r'time\s+([\d.]+)', line)
                if m:
                    total_ps = max(total_ps, float(m.group(1)))
    if total_ps == 0:
        total_ps = 10000  # fallback 10ns

    # Determine timepoints
    if interval_ns and interval_ns > 0:
        interval_ps = interval_ns * 1000
        timepoints = list(range(0, int(total_ps) + 1, int(interval_ps)))
        if timepoints[-1] < total_ps:
            timepoints.append(int(total_ps))
    else:
        timepoints = [0, int(total_ps)]

    # Fix PBC + fit first
    pbc_xtc = os.path.join(compare_dir, 'pbc.xtc')
    _run_gmx(f'trjconv -s {tpr} -f {xtc} -o "{_fwd(pbc_xtc)}" -center -pbc mol -ur compact',
             work_dir, input_text='1\n0\n')
    fit_xtc = os.path.join(compare_dir, 'fit.xtc')
    _run_gmx(f'trjconv -s {tpr} -f "{_fwd(pbc_xtc)}" -o "{_fwd(fit_xtc)}" -fit rot+trans',
             work_dir, input_text='4\n0\n')

    # Find non-Water group number (varies by system)
    nonwater_grp = _find_nonwater_group(work_dir, tpr)

    # Extract frames at each timepoint
    frame_pdbs = []
    for t in timepoints:
        out_pdb = os.path.join(compare_dir, f'frame_{t}ps.pdb')
        _run_gmx(f'trjconv -s {tpr} -f "{_fwd(fit_xtc)}" -o "{_fwd(out_pdb)}" -dump {t}',
                 work_dir, input_text=f'{nonwater_grp}\n')  # non-Water
        if os.path.exists(out_pdb):
            frame_pdbs.append((t, out_pdb))

    if len(frame_pdbs) < 2:
        return {'error': 'Could not extract frames for comparison'}

    results['timepoints'] = [t for t, _ in frame_pdbs]
    results['frame_pdbs'] = [p for _, p in frame_pdbs]

    # 2. Parse CA coordinates + all HETATM from each frame
    cofactor_names = {'HEM', 'HEME', 'HEC', 'HEA', 'HEO2', 'FAD', 'FMN', 'NAD', 'NAP'}

    def parse_frame(pdb_path):
        ca_coords = {}  # resid -> (resname, xyz)
        het_coords = {}  # (resname, atom) -> xyz
        with open(pdb_path, encoding='utf-8', errors='replace') as f:
            for line in f:
                if line.startswith('ENDMDL'):
                    break
                if not (line.startswith('ATOM') or line.startswith('HETATM')):
                    continue
                resname = line[17:20].strip()
                atom = line[12:16].strip()
                try:
                    x, y, z = float(line[30:38]), float(line[38:46]), float(line[46:54])
                except (ValueError, IndexError):
                    continue
                # CA coords for protein residues
                if atom == 'CA' and resname not in cofactor_names:
                    resid = int(line[22:26].strip())
                    ca_coords[resid] = (resname, np.array([x, y, z]))
                # Cofactor atoms (ATOM or HETATM — CHARMM27 writes HEM as ATOM)
                if resname in cofactor_names or line.startswith('HETATM'):
                    het_coords[(resname, atom)] = np.array([x, y, z])
        return ca_coords, het_coords

    frames_data = [(t, *parse_frame(p)) for t, p in frame_pdbs]

    if not frames_data or len(frames_data) < 2:
        results['per_residue'] = []
        results['mobile_regions'] = []
        results['rigid_core'] = []
        results['stats'] = {'mean_displacement': 0, 'std_displacement': 0,
                            'mobile_threshold': 0, 'n_mobile': 0, 'n_rigid': 0, 'n_total': 0}
        results['cofactor_displacement'] = {}
        results['report'] = '# Structure Comparison\n\nInsufficient frames for comparison.'
        return results

    # 3. Per-residue displacement (first vs last)
    first_ca, first_het = frames_data[0][1], frames_data[0][2]
    last_ca, last_het = frames_data[-1][1], frames_data[-1][2]

    per_res = []
    common_resids = sorted(set(first_ca.keys()) & set(last_ca.keys()))
    for rid in common_resids:
        rn1, c1 = first_ca[rid]
        rn2, c2 = last_ca[rid]
        disp = float(np.linalg.norm(c2 - c1))
        per_res.append({'resid': rid, 'resname': rn1, 'displacement': disp})

    results['per_residue'] = per_res

    # 4. Classify mobile regions
    displacements = np.array([r['displacement'] for r in per_res])
    if len(displacements) == 0:
        mean_disp, std_disp = 0.0, 0.0
    else:
        mean_disp = float(np.mean(displacements))
        std_disp = float(np.std(displacements))
    threshold_mobile = mean_disp + 1.5 * std_disp
    threshold_rigid = mean_disp - 0.5 * std_disp

    mobile = [r for r in per_res if r['displacement'] > threshold_mobile]
    rigid = [r for r in per_res if r['displacement'] < threshold_rigid]
    mobile.sort(key=lambda x: -x['displacement'])

    results['mobile_regions'] = mobile[:20]  # top 20
    results['rigid_core'] = rigid[:10]
    results['stats'] = {
        'mean_displacement': mean_disp,
        'std_displacement': std_disp,
        'mobile_threshold': float(threshold_mobile),
        'n_mobile': len(mobile),
        'n_rigid': len(rigid),
        'n_total': len(per_res),
    }

    # 5. Cofactor/HEME displacement
    cofactor_disp = {}
    heme_names = {'HEM', 'HEME', 'HEC'}
    for (rn, atom), c1 in first_het.items():
        key = (rn, atom)
        if key in last_het:
            c2 = last_het[key]
            d = float(np.linalg.norm(c2 - c1))
            if rn in heme_names or atom == 'FE':
                cofactor_disp[f'{rn}_{atom}'] = d

    # HEME center of mass
    hem_first = [c for (rn, _), c in first_het.items() if rn in heme_names]
    hem_last = [c for (rn, _), c in last_het.items() if rn in heme_names]
    if hem_first and hem_last:
        com1 = np.mean(hem_first, axis=0)
        com2 = np.mean(hem_last, axis=0)
        cofactor_disp['HEME_COM'] = float(np.linalg.norm(com2 - com1))
    results['cofactor_displacement'] = cofactor_disp

    # 6. Multi-timepoint comparison (if interval_ns set)
    if len(frames_data) > 2:
        timeline = []
        ref_ca = frames_data[0][1]
        for t, ca, het in frames_data:
            disps = []
            for rid in common_resids:
                if rid in ca and rid in ref_ca:
                    d = np.linalg.norm(ca[rid][1] - ref_ca[rid][1])
                    disps.append(d)
            # HEME COM
            h = [c for (rn, _), c in het.items() if rn in heme_names]
            h_ref = [c for (rn, _), c in frames_data[0][2].items() if rn in heme_names]
            hem_d = np.linalg.norm(np.mean(h, axis=0) - np.mean(h_ref, axis=0)) if h and h_ref else 0
            timeline.append({
                'time_ps': t,
                'mean_ca_disp': float(np.mean(disps)),
                'max_ca_disp': float(np.max(disps)),
                'heme_com_disp': float(hem_d),
            })
        results['timeline'] = timeline

    # 7. Generate comparison figure
    results['figure'] = _plot_structure_comparison(work_dir, results)

    # 8. Text report
    results['report'] = _generate_comparison_report(results, total_ps)

    return results


def _plot_structure_comparison(work_dir, results):
    """Plot structure comparison: per-residue displacement + cofactor + timeline."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec

    has_timeline = 'timeline' in results and len(results['timeline']) > 2
    n_rows = 3 if has_timeline else 2

    fig = plt.figure(figsize=(18, 6 * n_rows))
    gs = GridSpec(n_rows, 2, figure=fig, hspace=0.35, wspace=0.3)
    fig.suptitle('Structure Comparison Analysis', fontsize=16, fontweight='bold', y=0.98)

    per_res = results['per_residue']
    if not per_res:
        plt.close(fig)
        return None
    resids = [r['resid'] for r in per_res]
    disps = [r['displacement'] for r in per_res]
    stats = results['stats']

    # Panel 1: Per-residue displacement bar
    ax1 = fig.add_subplot(gs[0, 0])
    colors_bar = ['#e41a1c' if d > stats['mobile_threshold'] else '#377eb8' for d in disps]
    ax1.bar(resids, disps, width=1.0, color=colors_bar, alpha=0.7)
    ax1.axhline(stats['mean_displacement'], color='green', ls='--', alpha=0.6,
                label=f"Mean: {stats['mean_displacement']:.2f} A")
    ax1.axhline(stats['mobile_threshold'], color='red', ls=':', alpha=0.6,
                label=f"Mobile threshold: {stats['mobile_threshold']:.2f} A")
    ax1.set_xlabel('Residue Number')
    ax1.set_ylabel('CA Displacement (A)')
    ax1.set_title('Per-Residue Displacement (First vs Last)')
    ax1.legend(fontsize=8)

    # Panel 2: Top mobile residues
    ax2 = fig.add_subplot(gs[0, 1])
    mobile = results['mobile_regions'][:15]
    if mobile:
        labels = [f"{r['resname']}{r['resid']}" for r in mobile]
        vals = [r['displacement'] for r in mobile]
        y_pos = range(len(labels))
        ax2.barh(y_pos, vals, color='#e41a1c', alpha=0.7)
        ax2.set_yticks(y_pos)
        ax2.set_yticklabels(labels, fontsize=8)
        ax2.invert_yaxis()
        ax2.set_xlabel('Displacement (A)')
        ax2.set_title('Top Mobile Residues')

    # Panel 3: Cofactor displacement
    ax3 = fig.add_subplot(gs[1, 0])
    cof = results.get('cofactor_displacement', {})
    if cof:
        labels_c = list(cof.keys())
        vals_c = list(cof.values())
        ax3.bar(labels_c, vals_c, color='#ff7f00', alpha=0.8)
        ax3.set_ylabel('Displacement (A)')
        ax3.set_title('Cofactor Atom Displacement')
        ax3.tick_params(axis='x', rotation=45)
    else:
        ax3.text(0.5, 0.5, 'No cofactor detected', transform=ax3.transAxes,
                 ha='center', va='center', fontsize=12, color='gray')
        ax3.set_title('Cofactor Displacement')

    # Panel 4: Displacement distribution
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.hist(disps, bins=40, color='#377eb8', alpha=0.7, density=True)
    ax4.axvline(stats['mean_displacement'], color='green', ls='--',
                label=f"Mean: {stats['mean_displacement']:.2f} A")
    ax4.axvline(stats['mobile_threshold'], color='red', ls=':',
                label=f"Mobile: {stats['mobile_threshold']:.2f} A")
    ax4.set_xlabel('Displacement (A)')
    ax4.set_ylabel('Density')
    ax4.set_title('Displacement Distribution')
    ax4.legend(fontsize=8)

    # Panels 5-6: Timeline (if available)
    if has_timeline:
        tl = results['timeline']
        times = [t['time_ps'] / 1000 for t in tl]

        ax5 = fig.add_subplot(gs[2, 0])
        ax5.plot(times, [t['mean_ca_disp'] for t in tl], 'b-o', markersize=4, label='Mean CA')
        ax5.plot(times, [t['max_ca_disp'] for t in tl], 'r-s', markersize=4, label='Max CA')
        ax5.set_xlabel('Time (ns)')
        ax5.set_ylabel('Displacement vs t=0 (A)')
        ax5.set_title('Structural Drift Over Time')
        ax5.legend(fontsize=8)

        ax6 = fig.add_subplot(gs[2, 1])
        ax6.plot(times, [t['heme_com_disp'] for t in tl], '#ff7f00', marker='D', markersize=5)
        ax6.set_xlabel('Time (ns)')
        ax6.set_ylabel('HEME COM Displacement (A)')
        ax6.set_title('HEME Center-of-Mass Drift')

    fig_path = os.path.join(work_dir, 'structure_comparison.png')
    fig.savefig(fig_path, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    logger.info("Structure comparison figure: %s", fig_path)
    return fig_path


def _generate_comparison_report(results, total_ps):
    """Generate text report for structure comparison."""
    lines = ['# Structure Comparison Report', '']
    stats = results['stats']
    lines.append(f'Simulation time: {total_ps/1000:.1f} ns')
    lines.append(f'Total residues analyzed: {stats["n_total"]}')
    lines.append(f'Mean CA displacement: {stats["mean_displacement"]:.2f} A')
    lines.append(f'Mobile threshold (mean+1.5*std): {stats["mobile_threshold"]:.2f} A')
    pct = 100*stats["n_mobile"]/stats["n_total"] if stats["n_total"] > 0 else 0
    lines.append(f'Mobile residues: {stats["n_mobile"]} ({pct:.1f}%)')
    lines.append(f'Rigid core residues: {stats["n_rigid"]}')
    lines.append('')

    # Mobile regions
    lines.append('## Top Mobile Residues (large structural change)')
    for r in results['mobile_regions'][:15]:
        lines.append(f'  {r["resname"]}{r["resid"]:4d}: {r["displacement"]:.2f} A')

    # Rigid core
    lines.append('')
    lines.append('## Rigid Core (minimal change)')
    for r in results.get('rigid_core', [])[:10]:
        lines.append(f'  {r["resname"]}{r["resid"]:4d}: {r["displacement"]:.2f} A')

    # Cofactor
    cof = results.get('cofactor_displacement', {})
    if cof:
        lines.append('')
        lines.append('## Cofactor Displacement')
        for k, v in sorted(cof.items(), key=lambda x: -x[1]):
            lines.append(f'  {k}: {v:.2f} A')

    # Mobile sequence segments
    lines.append('')
    lines.append('## Mobile Segments (consecutive mobile residues)')
    mobile_ids = {r['resid'] for r in results['mobile_regions']}
    segments = []
    current_seg = []
    for r in sorted(results['per_residue'], key=lambda x: x['resid']):
        if r['resid'] in mobile_ids:
            current_seg.append(r)
        else:
            if len(current_seg) >= 2:
                segments.append(current_seg)
            current_seg = []
    if len(current_seg) >= 2:
        segments.append(current_seg)
    for seg in segments:
        start, end = seg[0], seg[-1]
        mean_d = np.mean([r['displacement'] for r in seg])
        lines.append(f'  {start["resname"]}{start["resid"]}-{end["resname"]}{end["resid"]} '
                     f'({len(seg)} res, mean disp: {mean_d:.2f} A)')

    return '\n'.join(lines)


def run_full_analysis(work_dir, key_residues=None, ligand_resname='MOL',
                      temp=300, visualize=False):
    """Run complete MD analysis suite.

    Args:
        work_dir: MD simulation directory
        key_residues: list like ['ASP10', 'LEU151', 'PHE15']
        ligand_resname: ligand residue name in topology
        temp: simulation temperature (for FEL)
        visualize: if True, also generate trajectory animation + porcupine plot

    Returns:
        dict with all analysis results and figure path
    """
    results = {}

    logger.info("=== RMSD ===")
    results['rmsd'] = compute_rmsd(work_dir)

    logger.info("=== RMSF ===")
    results['rmsf'] = compute_rmsf(work_dir)

    logger.info("=== Radius of Gyration ===")
    results['gyrate'] = compute_gyration(work_dir)

    logger.info("=== Energy ===")
    results['energy'] = compute_energy(work_dir)

    logger.info("=== H-bonds ===")
    results['hbonds'] = compute_hbonds(work_dir)

    logger.info("=== Metal Coordination (auto-detect) ===")
    results['coordination'] = compute_coordination(work_dir)

    logger.info("=== Distances ===")
    results['distances'] = compute_distances(
        work_dir, residues=key_residues, ligand_resname=ligand_resname)

    logger.info("=== Free Energy Landscape ===")
    rmsd_bb = results['rmsd'].get('backbone')
    rg = results['gyrate']
    results['fel'] = compute_fel(work_dir, rmsd_bb, rg, temp=temp)

    logger.info("=== Generating plots ===")
    results['figure'] = plot_all(work_dir, key_residues, ligand_resname)

    if results['coordination']:
        logger.info("=== Metalloprotein coordination figure ===")
        results['coordination_figure'] = plot_coordination(work_dir, results['coordination'])

    logger.info("=== Structure Comparison ===")
    results['comparison'] = compare_structures(work_dir)

    if visualize:
        logger.info("=== Trajectory Animation ===")
        from .visualize import render_trajectory, render_porcupine
        results['animation'] = render_trajectory(work_dir)
        logger.info("=== PCA Porcupine Plot ===")
        results['porcupine'] = render_porcupine(work_dir)

    return results
