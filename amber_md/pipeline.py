"""
AMBER CYP450+Heme MD Pipeline — main orchestrator.
Based on Shahrokh et al., J. Comp. Chem. (2011).

Automated pipeline:
  1. Parse PDB -> identify heme + axial cysteine
  2. Auto-detect heme state (CPDI/DIOXY/IC6)
  3. Kabsch superpose reference heme onto PDB position
  4. Write cleaned PDB (strip H, CYS->CYP, replace heme)
  5. Generate tleap input
  6. Generate MD input files (10-stage Shahrokh protocol)
  7. Generate run script + analysis scripts
"""

import os
import sys

import numpy as np

from .config import VALID_STATES, get_heme_params_base
from .prepare import (
    parse_pdb_atoms,
    extract_sequence,
    find_heme_residue,
    find_heme_anchors,
    identify_axial_cysteine,
    detect_heme_state,
    superpose_heme,
    write_prepared_pdb,
)
from .md_protocol import (
    generate_tleap_input,
    generate_md_inputs,
    generate_run_script,
    generate_analysis_scripts,
)


def run_cyp_heme_md(pdb_path, output_dir=None, state=None, keep_chains=None,
                    prod_ns=10.0, params_base=None, config=None, **kw):
    """Run the full CYP450 + heme AMBER MD preparation pipeline.

    Args:
        pdb_path: Input PDB file path
        output_dir: Output directory (default: <pdb_stem>_amber_md/)
        state: Heme state ('CPDI', 'DIOXY', 'IC6') or None for auto-detect
        keep_chains: List of chain IDs to keep, or None for auto
        prod_ns: Production MD time in nanoseconds (default 10)
        params_base: Heme parameters directory (default from config)
        config: Optional config dict

    Returns:
        dict with state, axial_cys, hem_resid, rmsd, output_dir, next_steps
        or {'error': ..., 'hint': ...} on failure
    """
    # Validate input
    if not os.path.exists(pdb_path):
        return {'error': f'PDB not found: {pdb_path}',
                'hint': 'Provide a valid PDB file path'}

    if state and state not in VALID_STATES:
        return {'error': f'Invalid state: {state}',
                'hint': f'Valid states: {", ".join(VALID_STATES)}'}

    # Output directory
    if output_dir is None:
        stem = os.path.splitext(os.path.basename(pdb_path))[0]
        output_dir = os.path.join(os.getcwd(), f'{stem}_amber_md')
    output_dir = os.path.abspath(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    # Resolve params base
    if params_base is None:
        params_base = get_heme_params_base(config)

    log_lines = []

    def log(msg):
        log_lines.append(msg)
        print(msg)

    try:
        log("=" * 70)
        log("  CYP450 + Heme AMBER MD Pipeline")
        log("  Based on Shahrokh et al., J. Comp. Chem. (2011)")
        log("=" * 70)

        # Step 1: Parse PDB
        log("\n[Step 1] Parsing PDB...")
        atoms = parse_pdb_atoms(pdb_path)
        log(f"  Total atoms: {len(atoms)}")

        # Step 2: Find heme residue
        log("\n[Step 2] Identifying heme residue...")
        heme_resname, heme_chain, heme_resid, heme_atoms = find_heme_residue(atoms)
        if heme_resname is None:
            return {'error': 'No heme residue found in PDB',
                    'hint': 'Ensure PDB contains HEM/HEC/HEA or an Fe atom in a porphyrin residue'}
        log(f"  Found: {heme_resname} chain {heme_chain} resid {heme_resid} ({len(heme_atoms)} atoms)")

        # Step 3: Extract heme anchors
        log("\n[Step 3] Extracting heme anchor atoms (Fe + 4N)...")
        heme_anchors = find_heme_anchors(heme_atoms)
        if 'FE' not in heme_anchors:
            return {'error': 'No Fe atom found in heme residue',
                    'hint': 'Check PDB for Fe/FE/FE1/FE2 atom names'}
        fe_coord = heme_anchors['FE']
        n_keys = [k for k in heme_anchors if k != 'FE']
        log(f"  Fe: [{fe_coord[0]:.3f}, {fe_coord[1]:.3f}, {fe_coord[2]:.3f}]")
        for n in n_keys:
            d = np.linalg.norm(heme_anchors[n] - fe_coord)
            log(f"  {n}: Fe-N = {d:.3f} A")

        # Step 4: Identify axial cysteine
        log("\n[Step 4] Identifying axial cysteine...")
        seq = extract_sequence(atoms)
        try:
            cys_info = identify_axial_cysteine(atoms, heme_anchors, seq)
        except ValueError as e:
            return {'error': str(e),
                    'hint': 'This PDB may not be a CYP450 — no proximal cysteine found'}
        axial_cys_resid = cys_info['resid']
        axial_cys_chain = cys_info['chain']
        log(f"  Axial Cys: resid {axial_cys_resid} chain {axial_cys_chain}")
        log(f"  Method: {cys_info['method']}, Motif: {cys_info['motif']}")
        log(f"  Fe-SG distance: {cys_info['fe_sg_dist']:.3f} A")

        # Step 5: Determine heme state
        log("\n[Step 5] Determining heme state...")
        if state is None:
            detected_state, o_atoms, reason = detect_heme_state(heme_atoms, fe_coord)
            state = detected_state
            log(f"  Auto-detected: {state}")
            log(f"  Reason: {reason}")
        else:
            log(f"  User-specified: {state}")

        params_dir = os.path.join(params_base, state)
        hem_mol2 = os.path.join(params_dir, "HEM.mol2")
        # For WSL paths, skip existence check on Windows
        if not params_dir.startswith('/mnt/') and not os.path.exists(hem_mol2):
            return {'error': f'HEM.mol2 not found at {hem_mol2}',
                    'hint': f'Ensure heme parameters exist in {params_dir}/'}
        log(f"  Parameters: {params_dir}")

        # Step 6: Superpose heme
        log("\n[Step 6] Superposing reference heme onto PDB position (Kabsch)...")
        # For WSL paths, need to convert for local access
        local_mol2 = hem_mol2
        if hem_mol2.startswith('/mnt/'):
            # Convert WSL path to Windows path for local file access
            drive = hem_mol2[5]
            local_mol2 = f"{drive.upper()}:{hem_mol2[6:]}"
            local_mol2 = local_mol2.replace('/', '\\')

        if not os.path.exists(local_mol2):
            return {'error': f'HEM.mol2 not found at {local_mol2}',
                    'hint': f'Ensure heme parameters exist. Tried: {hem_mol2}'}

        try:
            transformed, rmsd, best_perm, mol2_all = superpose_heme(
                heme_anchors, local_mol2)
        except ValueError as e:
            return {'error': f'Heme superposition failed: {e}',
                    'hint': 'Check that PDB heme has Fe + 4 pyrrole N atoms'}
        log(f"  Best N permutation: {best_perm}")
        log(f"  Anchor RMSD: {rmsd:.4f} A")

        fe_new = transformed['FE']
        sg_coord = None
        for a in atoms:
            if (a['chain'] == axial_cys_chain and a['resid'] == axial_cys_resid
                    and a['name'] == 'SG'):
                sg_coord = np.array([a['x'], a['y'], a['z']])
        if sg_coord is not None:
            fe_sg_new = np.linalg.norm(fe_new - sg_coord)
            log(f"  Fe(new) - SG distance: {fe_sg_new:.3f} A")

        # Step 7: Write prepared PDB
        log("\n[Step 7] Writing prepared PDB...")
        pdb_name = "cyp_heme_prepared.pdb"
        if keep_chains is None:
            keep_chains = [axial_cys_chain]

        n_atoms, hem_resid_out = write_prepared_pdb(
            os.path.join(output_dir, pdb_name),
            atoms, heme_resname, heme_chain, heme_resid,
            axial_cys_resid, axial_cys_chain, transformed, mol2_all,
            keep_chains=keep_chains
        )
        log(f"  Output: {pdb_name} ({n_atoms} atoms)")
        log(f"  HEM resid: {hem_resid_out}, CYP resid: {axial_cys_resid}")

        # Step 8: Generate tleap input
        log("\n[Step 8] Generating tleap input...")
        prmtop = "cyp_heme.prmtop"
        inpcrd = "cyp_heme.inpcrd"
        tleap_path = generate_tleap_input(
            output_dir, pdb_name, state, params_dir,
            axial_cys_resid, hem_resid_out, prmtop, inpcrd
        )
        log(f"  Written: {tleap_path}")

        # Step 9: Generate MD inputs
        log("\n[Step 9] Generating MD input files...")
        md_dir = generate_md_inputs(output_dir, state, prod_ns=prod_ns)
        log(f"  Written to: {md_dir}/")

        # Step 10: Generate run script
        log("\n[Step 10] Generating run script...")
        run_path = generate_run_script(output_dir, prmtop, inpcrd)
        log(f"  Written: {run_path}")

        # Step 11: Generate analysis scripts
        log("\n[Step 11] Generating analysis scripts...")
        analyze_path, plot_path = generate_analysis_scripts(
            output_dir, prmtop, axial_cys_resid, hem_resid_out, state
        )
        log(f"  Written: {analyze_path}")
        log(f"  Written: {plot_path}")

        # Save log
        log_path = os.path.join(output_dir, 'pipeline.log')
        with open(log_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(log_lines))

        log("\n" + "=" * 70)
        log("  Pipeline Complete!")
        log("=" * 70)

        return {
            'state': state,
            'axial_cys': axial_cys_resid,
            'axial_cys_chain': axial_cys_chain,
            'hem_resid': hem_resid_out,
            'rmsd': float(rmsd),
            'n_atoms': n_atoms,
            'output_dir': output_dir,
            'prmtop': os.path.join(output_dir, prmtop),
            'inpcrd': os.path.join(output_dir, inpcrd),
            'run_script': run_path,
            'analyze_script': analyze_path,
            'plot_script': plot_path,
            'tleap_input': tleap_path,
            'prepared_pdb': os.path.join(output_dir, pdb_name),
            'next_steps': [
                f'Run tleap: cd {output_dir} && tleap -f tleap.in',
                'Check tleap output for errors',
                'Run MD: bash run_md.sh',
                'After production: bash analyze.sh && python3 plot_analysis.py',
            ],
        }

    except Exception as e:
        return {'error': f'Pipeline failed: {e}',
                'hint': f'Check PDB structure and heme parameters at {params_base}'}
