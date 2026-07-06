"""
PLIP Analysis — core logic
===========================
Protein-ligand interaction profiling with multi-backend support.

Backend priority: native → Docker → WSL
"""

import logging
import os
import subprocess
import sys
import tempfile

from .config import (
    DOCKER_IMAGE, DOCKER_DIR, STANDARD_RESIDUES, WATER_RESNAMES,
    NON_LIGAND_HETATM, COFACTOR_RESNAMES, COMMON_IONS, CRYSTALLOGRAPHIC_ADDITIVES,
    DOCKER_TIMEOUT, WSL_TIMEOUT, DOCKER_BUILD_TIMEOUT,
)

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# Public API
# ═══════════════════════════════════════════════════════════════════════════════

def run_plip_analysis(complex_pdb: str, output_dir: str = None,
                      config: dict = None) -> dict:
    """
    Run PLIP interaction analysis on a protein-ligand complex.

    Auto-fixes ATOM→HETATM for non-standard residues (ligands).
    Output is saved to <pdb_dir>/<pdb_stem>_plip/ next to the input PDB.

    Returns:
        dict with interaction types, residues, report, and ligand info
    """
    # Validate input
    if not complex_pdb or not os.path.isfile(complex_pdb):
        return {'error': f'Complex PDB not found: "{complex_pdb}"',
                'hint': 'Provide the full path to a protein-ligand complex PDB file '
                        '(e.g. from smart_docking result["complexes"][0]["path"]).',
                'interactions': {}}

    # Legacy: add scripts path for native plip_analysis import
    if config:
        sd = config.get('scripts_dir')
        if sd and sd not in sys.path:
            sys.path.insert(0, sd)

    # Convention: output defaults to CWD (project workspace), not input file dir
    pdb_stem = os.path.splitext(os.path.basename(complex_pdb))[0]
    if not output_dir:
        output_dir = os.path.join(os.getcwd(), f'{pdb_stem}_plip')
    os.makedirs(output_dir, exist_ok=True)

    # Auto-fix: detect ligands written as ATOM and convert to HETATM
    fixed_pdb = _fix_pdb_hetatm(complex_pdb)
    was_fixed = (fixed_pdb != complex_pdb)

    # Detect actual drug/ligand resnames from HETATM records
    # Exclude: water, cofactors (HEM, FAD...), ions (NA, SO4...), additives (GOL, EDO...)
    ligand_resnames = set()
    cofactors_found = set()
    with open(fixed_pdb, 'r', encoding='utf-8') as f:
        for line in f:
            if line.startswith('HETATM'):
                rn = line[17:20].strip()
                if rn in COFACTOR_RESNAMES:
                    cofactors_found.add(rn)
                elif rn not in NON_LIGAND_HETATM:
                    ligand_resnames.add(rn)
    if cofactors_found:
        logger.info("Cofactors in complex (treated as receptor): %s",
                     ', '.join(sorted(cofactors_found)))

    if not ligand_resnames:
        if was_fixed and os.path.exists(fixed_pdb):
            try: os.unlink(fixed_pdb)
            except OSError: pass
        return {'error': 'No ligand found in PDB',
                'hint': 'PDB has no HETATM records (after auto-fix). '
                        'Ensure the file contains a ligand.',
                'interactions': {}}

    try:
        # --- Strategy 1: native plip_analysis module ---
        try:
            from plip_analysis import run_plip, generate_interaction_report

            all_interactions = {}
            for resname in sorted(ligand_resnames):
                result = run_plip(fixed_pdb, ligand_resname=resname)
                if result:
                    all_interactions.update(result)

            if not all_interactions:
                result = run_plip(fixed_pdb, ligand_resname="")
                if result:
                    all_interactions = result

            if all_interactions:
                return _finalize(all_interactions, complex_pdb, output_dir,
                                 ligand_resnames, was_fixed, 'native')
            logger.info("Native PLIP returned no interactions, trying Docker...")
        except ImportError:
            logger.info("plip_analysis not importable, trying Docker backend...")
        except Exception as e:
            logger.info("Native PLIP failed (%s), trying Docker backend...", e)

        # --- Strategy 2: Docker PLIP (skip if daemon not running) ---
        if _docker_daemon_running():
            try:
                result = _run_plip_docker(fixed_pdb, output_dir, ligand_resnames)
                if result and 'error' not in result:
                    return _finalize(result.get('interactions', {}), complex_pdb,
                                     output_dir, ligand_resnames, was_fixed, 'docker')
                logger.info("Docker PLIP failed (%s), trying WSL...",
                            result.get('error', 'no interactions'))
            except Exception as e:
                logger.info("Docker PLIP failed (%s), trying WSL...", e)
        else:
            logger.info("Docker daemon not running, skipping Docker backend")

        # --- Strategy 3: WSL PLIP ---
        try:
            result = _run_plip_wsl(fixed_pdb, output_dir, ligand_resnames)
            if result and 'error' not in result:
                return _finalize(result.get('interactions', {}), complex_pdb,
                                 output_dir, ligand_resnames, was_fixed, 'wsl')
        except Exception as e:
            logger.warning("WSL PLIP failed: %s", e)

        # --- All backends failed ---
        return {
            'error': 'PLIP failed: no backend available (tried native + docker + WSL)',
            'hint': 'Run: docker build -t zhuangzhou-plip zhuangzhou/plip_analysis/docker/',
            'interactions': {},
            'ligands_detected': sorted(ligand_resnames),
            'hetatm_fixed': was_fixed,
        }
    finally:
        if was_fixed and os.path.exists(fixed_pdb):
            try: os.unlink(fixed_pdb)
            except OSError: pass


# ═══════════════════════════════════════════════════════════════════════════════
# HETATM auto-fix
# ═══════════════════════════════════════════════════════════════════════════════

def _fix_pdb_hetatm(pdb_path: str) -> str:
    """
    Auto-fix PDB: convert non-standard residues from ATOM to HETATM.

    PLIP requires ligands to be HETATM records. Many MD outputs write
    ligands as ATOM. This detects non-standard residue names and converts
    them, writing a fixed copy.

    Returns path to fixed PDB (or original if no fix needed).
    """
    needs_fix = False
    ligand_resnames = set()
    with open(pdb_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.startswith('ATOM  ') or line.startswith('ATOM\t'):
                resname = line[17:20].strip()
                if resname and resname not in STANDARD_RESIDUES:
                    needs_fix = True
                    ligand_resnames.add(resname)

    if not needs_fix:
        return pdb_path

    logger.info("Auto-fix: converting ATOM->HETATM for non-standard residues: %s",
                ', '.join(sorted(ligand_resnames)))

    fixed_dir = os.path.dirname(pdb_path) or '.'
    fd, fixed_path = tempfile.mkstemp(suffix='_fixed.pdb', dir=fixed_dir)
    os.close(fd)

    with open(pdb_path, 'r', encoding='utf-8') as fin, \
         open(fixed_path, 'w', encoding='utf-8') as fout:
        for line in fin:
            if line.startswith('ATOM  ') or line.startswith('ATOM\t'):
                resname = line[17:20].strip()
                if resname in ligand_resnames:
                    line = 'HETATM' + line[6:]
            fout.write(line)

    return fixed_path


# ═══════════════════════════════════════════════════════════════════════════════
# Docker backend
# ═══════════════════════════════════════════════════════════════════════════════

def _docker_daemon_running() -> bool:
    """Quick check if Docker daemon is responsive (< 3s)."""
    try:
        proc = subprocess.run(
            ['docker', 'info'], capture_output=True, timeout=3)
        return proc.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        return False


def _docker_available() -> bool:
    """Check if Docker is running and the PLIP image exists."""
    try:
        proc = subprocess.run(
            ['docker', 'image', 'inspect', DOCKER_IMAGE],
            capture_output=True, timeout=10)
        return proc.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


def _build_plip_docker() -> bool:
    """Build the PLIP Docker image from bundled Dockerfile."""
    if not os.path.isfile(os.path.join(DOCKER_DIR, 'Dockerfile')):
        logger.warning("PLIP Dockerfile not found at %s", DOCKER_DIR)
        return False
    logger.info("Building PLIP Docker image '%s' ...", DOCKER_IMAGE)
    try:
        proc = subprocess.run(
            ['docker', 'build', '-t', DOCKER_IMAGE, DOCKER_DIR],
            capture_output=True, text=True, timeout=DOCKER_BUILD_TIMEOUT)
        if proc.returncode == 0:
            logger.info("PLIP Docker image built successfully.")
            return True
        else:
            logger.error("Docker build failed:\n%s", proc.stderr[-500:])
            return False
    except (FileNotFoundError, subprocess.TimeoutExpired) as e:
        logger.error("Docker build error: %s", e)
        return False


def _run_plip_docker(pdb_path: str, output_dir: str,
                     ligand_resnames: set) -> dict:
    """Run PLIP via Docker container."""
    import json as _json

    pdb_path = os.path.abspath(pdb_path)
    output_dir = os.path.abspath(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    # Auto-build image if missing
    if not _docker_available():
        if not _build_plip_docker():
            return {'error': 'PLIP Docker image not available and build failed'}

    pdb_dir = os.path.dirname(pdb_path)
    pdb_name = os.path.basename(pdb_path)

    cmd = [
        'docker', 'run', '--rm',
        '-v', f'{pdb_dir}:/input:ro',
        '-v', f'{output_dir}:/output',
        DOCKER_IMAGE,
        f'/input/{pdb_name}',
        '/output',
    ]

    try:
        proc = subprocess.run(cmd, capture_output=True, text=True,
                              timeout=DOCKER_TIMEOUT)
    except FileNotFoundError:
        return {'error': 'Docker not installed or not in PATH'}
    except subprocess.TimeoutExpired:
        return {'error': f'PLIP Docker timed out ({DOCKER_TIMEOUT}s)'}

    stdout = proc.stdout.strip()
    if proc.returncode != 0 and not stdout:
        return {'error': f'Docker PLIP failed (exit {proc.returncode})',
                'stderr': proc.stderr[-500:] if proc.stderr else ''}

    # Parse JSON from file or stdout
    json_path = os.path.join(output_dir, 'plip_results.json')
    interactions = {}

    if os.path.exists(json_path):
        try:
            with open(json_path, encoding='utf-8') as f:
                data = _json.load(f)
            interactions = data.get('interactions', {})
        except (ValueError, OSError):
            pass

    if not interactions and stdout:
        try:
            data = _json.loads(stdout)
            interactions = data.get('interactions', {})
        except (ValueError, KeyError):
            pass

    if not interactions:
        return {'error': 'PLIP Docker produced no interactions',
                'stdout': stdout[:500],
                'stderr': proc.stderr[:500] if proc.stderr else ''}

    return {'interactions': interactions}


# ═══════════════════════════════════════════════════════════════════════════════
# WSL backend
# ═══════════════════════════════════════════════════════════════════════════════

def _win_to_wsl_path(win_path: str) -> str:
    """Convert Windows path to WSL /mnt/ path."""
    p = os.path.abspath(win_path).replace('\\', '/')
    if len(p) >= 2 and p[1] == ':':
        drive = p[0].lower()
        return f'/mnt/{drive}{p[2:]}'
    return p


def _run_plip_wsl(pdb_path: str, output_dir: str,
                  ligand_resnames: set) -> dict:
    """Run PLIP via WSL subprocess."""
    import xml.etree.ElementTree as ET

    wsl_pdb = _win_to_wsl_path(pdb_path)

    if output_dir:
        wsl_outdir = _win_to_wsl_path(output_dir)
        os.makedirs(output_dir, exist_ok=True)
    else:
        # Convention: output defaults to CWD/<stem>_plip/
        pdb_stem = os.path.splitext(os.path.basename(pdb_path))[0]
        output_dir = os.path.join(os.getcwd(), f'{pdb_stem}_plip')
        os.makedirs(output_dir, exist_ok=True)
        wsl_outdir = _win_to_wsl_path(output_dir)

    cmd = ['wsl', '-e', 'bash', '-c',
           f'plip -f "{wsl_pdb}" -o "{wsl_outdir}" -x 2>&1']
    try:
        proc = subprocess.run(cmd, capture_output=False, stdout=subprocess.PIPE,
                              stderr=subprocess.STDOUT, timeout=WSL_TIMEOUT)
        plip_stdout = proc.stdout.decode('utf-8', errors='replace')
    except FileNotFoundError:
        return {'error': 'WSL not available'}
    except subprocess.TimeoutExpired:
        return {'error': f'WSL PLIP timed out ({WSL_TIMEOUT}s)'}

    # Parse XML output files
    all_interactions = {}

    xml_files = [f for f in os.listdir(output_dir) if f.endswith('.xml')]
    if not xml_files:
        return _parse_plip_text_output(plip_stdout, ligand_resnames)

    for xf in xml_files:
        xml_path = os.path.join(output_dir, xf)
        try:
            tree = ET.parse(xml_path)
            root = tree.getroot()
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
                }

                for hb in bsite.iter('hydrogen_bond'):
                    interactions['hydrogen_bonds'].append({
                        'residue': f"{_xt(hb, 'restype')}{_xt(hb, 'resnr')}",
                        'dist_h_a': _xt(hb, 'dist_h-a'),
                        'dist_d_a': _xt(hb, 'dist_d-a'),
                    })

                for hp in bsite.iter('hydrophobic_interaction'):
                    interactions['hydrophobic'].append({
                        'residue': f"{_xt(hp, 'restype')}{_xt(hp, 'resnr')}",
                        'dist': _xt(hp, 'dist'),
                    })

                for sb in bsite.iter('salt_bridge'):
                    interactions['salt_bridges'].append({
                        'residue': f"{_xt(sb, 'restype')}{_xt(sb, 'resnr')}",
                        'dist': _xt(sb, 'dist'),
                    })

                for ps in bsite.iter('pi_stack'):
                    interactions['pi_stacking'].append({
                        'residue': f"{_xt(ps, 'restype')}{_xt(ps, 'resnr')}",
                    })

                all_interactions[key] = interactions

        except ET.ParseError as e:
            logger.warning("Failed to parse PLIP XML %s: %s", xf, e)

    if not all_interactions:
        return _parse_plip_text_output(plip_stdout, ligand_resnames)

    return {'interactions': all_interactions}


# ═══════════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════════

def _xt(parent, path: str) -> str:
    """Get text from XML element by tag/path."""
    el = parent.find(path)
    return el.text.strip() if el is not None and el.text else ''


def _parse_plip_text_output(stdout: str, ligand_resnames: set) -> dict:
    """Parse PLIP text stdout as a fallback when XML is not available."""
    if not stdout.strip():
        return {'error': 'PLIP produced no output'}

    interactions = {}
    current_ligand = None
    for line in stdout.splitlines():
        if 'analyzing' in line.lower() or 'ligand' in line.lower():
            for rn in ligand_resnames:
                if rn in line:
                    current_ligand = rn
                    break
        if current_ligand and ':' in line:
            lower = line.lower()
            for itype in ('hydrogen', 'hydrophobic', 'salt', 'pi-stack',
                          'pi-cation', 'halogen', 'water'):
                if itype in lower:
                    if current_ligand not in interactions:
                        interactions[current_ligand] = {}
                    interactions[current_ligand][itype] = line.strip()

    if interactions:
        return {'interactions': interactions, 'report': stdout,
                'note': 'Parsed from text output (XML not available)'}

    return {'interactions': {}, 'report': stdout,
            'note': 'Could not parse PLIP output, raw text included'}


def _generate_plip_report_md(interactions: dict, pdb_name: str,
                              ligand_resnames: list, was_fixed: bool,
                              backend: str) -> str:
    """Generate a formatted Markdown report from PLIP interaction results."""
    lines = [
        '# PLIP Interaction Report',
        '',
        f'**PDB**: `{pdb_name}`  ',
        f'**Backend**: {backend}  ',
        f'**Ligands detected**: {", ".join(ligand_resnames)}  ',
        f'**HETATM auto-fix**: {"Yes" if was_fixed else "No"}',
        '',
    ]

    for lig_key, lig_data in interactions.items():
        lines.append(f'## Ligand: {lig_key}')
        lines.append('')

        if isinstance(lig_data, dict):
            for itype, ilist in lig_data.items():
                if not ilist:
                    continue
                header = itype.replace('_', ' ').title()
                lines.append(f'### {header} ({len(ilist)})')
                lines.append('')
                if isinstance(ilist, list):
                    for ix in ilist:
                        if isinstance(ix, dict):
                            res = ix.get('residue', '?')
                            details = ', '.join(
                                f'{k}={v}' for k, v in ix.items()
                                if k != 'residue' and v)
                            lines.append(f'| {res} | {details} |')
                        else:
                            lines.append(f'- {ix}')
                else:
                    lines.append(f'- {ilist}')
                lines.append('')

    return '\n'.join(lines)


def _finalize(interactions: dict, complex_pdb: str, output_dir: str,
              ligand_resnames: set, was_fixed: bool, backend: str) -> dict:
    """Write report and return standardized result dict."""
    # Filter out cofactor/ion binding sites that PLIP may have detected
    # Keys are like "HEM:A:501" or "GOL:B:1" — check the residue name prefix
    filtered = {}
    removed_sites = []
    for key, data in interactions.items():
        site_resname = key.split(':')[0].strip() if ':' in key else key
        if site_resname in NON_LIGAND_HETATM:
            removed_sites.append(key)
        else:
            filtered[key] = data
    if removed_sites:
        logger.info("Filtered out non-ligand PLIP sites: %s", removed_sites)
    interactions = filtered

    report = _generate_plip_report_md(
        interactions, os.path.basename(complex_pdb),
        sorted(ligand_resnames), was_fixed, backend)
    report_path = os.path.join(output_dir, 'plip_report.md')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report)
    return {
        'interactions': interactions,
        'report': report,
        'report_path': report_path,
        'output_dir': output_dir,
        'ligands_detected': sorted(ligand_resnames),
        'hetatm_fixed': was_fixed,
        'backend': backend,
        'next_steps': [
            'pymol_render: 生成PLIP相互作用可视化图片 (氢键、疏水作用等)',
            'chimerax_render: 用ChimeraX渲染高分辨率结构图',
        ],
    }
