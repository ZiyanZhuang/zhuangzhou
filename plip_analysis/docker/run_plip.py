#!/usr/bin/env python3
"""
PLIP container entrypoint.

Usage:  python run_plip.py /data/complex.pdb /data/output [--ligand MOL]

Runs PLIP analysis and writes XML + a JSON summary to the output directory.
"""
import json
import os
import sys
import xml.etree.ElementTree as ET
from pathlib import Path


def run_plip_xml(pdb_path: str, output_dir: str):
    """Run PLIP via command-line, producing XML output."""
    os.makedirs(output_dir, exist_ok=True)
    cmd = f'plip -f "{pdb_path}" -o "{output_dir}" -x'
    ret = os.system(cmd)
    return ret


def parse_plip_xml(output_dir: str) -> dict:
    """Parse all PLIP XML files into a structured dict."""
    results = {}
    xml_files = sorted(Path(output_dir).glob('*.xml'))

    for xf in xml_files:
        try:
            tree = ET.parse(str(xf))
            root = tree.getroot()
        except ET.ParseError:
            continue

        for bsite in root.iter('bindingsite'):
            lig_name = _xt(bsite, './/identifiers/longname') or 'UNK'
            lig_chain = _xt(bsite, './/identifiers/chain') or ''
            lig_pos = _xt(bsite, './/identifiers/position') or ''
            key = f"{lig_name}:{lig_chain}:{lig_pos}"

            interactions = {
                'hydrogen_bonds': [],
                'hydrophobic': [],
                'salt_bridges': [],
                'pi_stacking': [],
                'pi_cation': [],
                'halogen_bonds': [],
                'water_bridges': [],
                'metal_complexes': [],
            }

            for hb in bsite.iter('hydrogen_bond'):
                interactions['hydrogen_bonds'].append({
                    'residue': f"{_xt(hb, 'restype')}{_xt(hb, 'resnr')}",
                    'reschain': _xt(hb, 'reschain'),
                    'dist_h_a': _xt(hb, 'dist_h-a'),
                    'dist_d_a': _xt(hb, 'dist_d-a'),
                    'donor_acceptor': _xt(hb, 'protisdon'),
                    'sidechain': _xt(hb, 'sidechain'),
                })

            for hp in bsite.iter('hydrophobic_interaction'):
                interactions['hydrophobic'].append({
                    'residue': f"{_xt(hp, 'restype')}{_xt(hp, 'resnr')}",
                    'reschain': _xt(hp, 'reschain'),
                    'dist': _xt(hp, 'dist'),
                })

            for sb in bsite.iter('salt_bridge'):
                interactions['salt_bridges'].append({
                    'residue': f"{_xt(sb, 'restype')}{_xt(sb, 'resnr')}",
                    'reschain': _xt(sb, 'reschain'),
                    'dist': _xt(sb, 'dist'),
                    'protispos': _xt(sb, 'protispos'),
                })

            for ps in bsite.iter('pi_stack'):
                interactions['pi_stacking'].append({
                    'residue': f"{_xt(ps, 'restype')}{_xt(ps, 'resnr')}",
                    'reschain': _xt(ps, 'reschain'),
                    'type': _xt(ps, 'type'),
                })

            for pc in bsite.iter('pi_cation_interaction'):
                interactions['pi_cation'].append({
                    'residue': f"{_xt(pc, 'restype')}{_xt(pc, 'resnr')}",
                    'reschain': _xt(pc, 'reschain'),
                    'protcharged': _xt(pc, 'protcharged'),
                })

            for hx in bsite.iter('halogen_bond'):
                interactions['halogen_bonds'].append({
                    'residue': f"{_xt(hx, 'restype')}{_xt(hx, 'resnr')}",
                    'reschain': _xt(hx, 'reschain'),
                    'dist': _xt(hx, 'dist'),
                    'acc_donor': _xt(hx, 'acctype'),
                })

            for wb in bsite.iter('water_bridge'):
                interactions['water_bridges'].append({
                    'residue': f"{_xt(wb, 'restype')}{_xt(wb, 'resnr')}",
                    'reschain': _xt(wb, 'reschain'),
                    'dist_a_w': _xt(wb, 'dist_a-w'),
                    'dist_d_w': _xt(wb, 'dist_d-w'),
                })

            for mc in bsite.iter('metal_complex'):
                interactions['metal_complexes'].append({
                    'residue': f"{_xt(mc, 'restype')}{_xt(mc, 'resnr')}",
                    'reschain': _xt(mc, 'reschain'),
                    'metal': _xt(mc, 'metal_type'),
                    'dist': _xt(mc, 'dist'),
                })

            # Remove empty interaction types
            interactions = {k: v for k, v in interactions.items() if v}
            results[key] = interactions

    return results


def _xt(parent, path: str) -> str:
    """Get text from XML element by tag/path."""
    el = parent.find(path)
    return el.text.strip() if el is not None and el.text else ''


def main():
    if len(sys.argv) < 3:
        print("Usage: run_plip.py <pdb_file> <output_dir>", file=sys.stderr)
        sys.exit(1)

    pdb_path = sys.argv[1]
    output_dir = sys.argv[2]

    if not os.path.exists(pdb_path):
        print(json.dumps({'error': f'PDB not found: {pdb_path}'}))
        sys.exit(1)

    # Run PLIP
    ret = run_plip_xml(pdb_path, output_dir)

    # Parse results
    results = parse_plip_xml(output_dir)

    # Write JSON summary
    summary = {
        'pdb': os.path.basename(pdb_path),
        'interactions': results,
        'exit_code': ret,
    }
    summary_path = os.path.join(output_dir, 'plip_results.json')
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2)

    # Print JSON to stdout for the caller to capture
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
