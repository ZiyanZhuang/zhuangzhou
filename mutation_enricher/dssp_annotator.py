"""
DSSP Per-Residue Secondary Structure + RSA Annotator
=====================================================
Replaces the crude B-factor proxy used in mutation analysis.

Primary:    WSLSandbox -> /usr/bin/dssp  (SS + RSA)
Fallback 1: Biopython DSSP wrapper (mkdssp binary)  (SS + RSA)
Fallback 2: pydssp pure-Python  (SS only, no RSA)
Fallback 3: empty dict (caller handles gracefully)
"""

import json
import logging
import os
import tempfile
from pathlib import Path
from typing import Dict, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[1]

logger = logging.getLogger(__name__)

# ── Constants ────────────────────────────────────────────────────────────────

# Tien et al. 2013 empirical max ASA values (Å²)
MAX_ASA = {
    'A': 121, 'R': 265, 'N': 187, 'D': 187, 'C': 148,
    'E': 214, 'Q': 214, 'G': 97,  'H': 216, 'I': 195,
    'L': 191, 'K': 230, 'M': 203, 'F': 228, 'P': 154,
    'S': 143, 'T': 163, 'W': 264, 'Y': 255, 'V': 165,
    'X': 180,
}

SS_LABELS = {
    'H': 'alpha-helix',
    'G': '310-helix',
    'I': 'pi-helix',
    'E': 'beta-strand',
    'B': 'beta-bridge',
    'T': 'turn',
    'S': 'bend',
    'C': 'coil/loop',
    ' ': 'coil/loop',
}

# ── Public API ────────────────────────────────────────────────────────────────


def get_ss_label(code: str) -> str:
    """Map DSSP code to human-readable secondary structure label."""
    return SS_LABELS.get(code, 'coil/loop')


def annotate_dssp(pdb_path: str, cache_dir: str = None) -> dict:
    """Returns {resnum: {ss, ss_label, asa, rsa, aa, chain}} or {} on failure."""
    pdb_path = os.path.abspath(pdb_path)
    pdb_stem = Path(pdb_path).stem

    # Check cache first
    if cache_dir:
        os.makedirs(cache_dir, exist_ok=True)
        cache_file = os.path.join(cache_dir, f'{pdb_stem}_dssp.json')
        if os.path.exists(cache_file):
            try:
                with open(cache_file, 'r', encoding='utf-8') as f:
                    raw = json.load(f)
                # JSON keys are strings; convert to int
                return {int(k): v for k, v in raw.items()}
            except Exception as e:
                logger.warning('DSSP cache read failed for %s: %s', cache_file, e)

    result = _run_wsl_dssp(pdb_path)

    if not result:
        logger.warning('WSL DSSP failed for %s, trying Biopython fallback', pdb_path)
        result = _run_biopython_dssp(pdb_path)

    if not result:
        logger.info('Biopython DSSP unavailable for %s, trying pydssp', pdb_path)
        result = _run_pydssp(pdb_path)

    if not result:
        logger.warning('All DSSP methods failed for %s, returning empty dict', pdb_path)
        return {}

    # pydssp 只提供 ss_type，RSA 标记为 pydssp_no_rsa；
    # 若 BioPython ShrakeRupley 可用，补充真实 RSA，使区域分类正常工作。
    if result and any(v.get('rsa_source') == 'pydssp_no_rsa' for v in result.values()):
        logger.info('pydssp 无 RSA，尝试 BioPython ShrakeRupley 补充')
        rsa_map = _run_shrake_rupley(pdb_path)
        if rsa_map:
            for resnum, data in result.items():
                if resnum in rsa_map:
                    data['asa'] = rsa_map[resnum]['asa']
                    data['rsa'] = rsa_map[resnum]['rsa']
                    data.pop('rsa_source', None)
            logger.info('ShrakeRupley 补充 RSA 完成，覆盖 %d 残基', len(rsa_map))

    # Cache result
    if cache_dir:
        try:
            cache_file = os.path.join(cache_dir, f'{pdb_stem}_dssp.json')
            with open(cache_file, 'w', encoding='utf-8') as f:
                json.dump(result, f, ensure_ascii=False)
        except Exception as e:
            logger.warning('DSSP cache write failed: %s', e)

    return result


# ── WSL DSSP path ─────────────────────────────────────────────────────────────


def _win_to_wsl(path: str) -> str:
    """Convert a Windows path to a WSL path (X:\\foo\\bar -> /mnt/x/foo/bar)."""
    path = path.replace('\\', '/')
    if len(path) >= 2 and path[1] == ':':
        drive = path[0].lower()
        path = f'/mnt/{drive}' + path[2:]
    return path


def _run_wsl_dssp(pdb_path: str) -> Optional[dict]:
    """Run DSSP via WSLSandbox and parse the output file."""
    try:
        from zhuangzhou.envs.sandbox import WSLSandbox
    except ImportError as e:
        logger.warning('WSLSandbox import failed: %s', e)
        return None

    wsl = WSLSandbox(timeout=120)
    if not wsl.available:
        logger.warning('WSL not available')
        return None

    wsl_pdb = _win_to_wsl(pdb_path)

    # Use a unique temp path inside WSL's /tmp
    pdb_stem = Path(pdb_path).stem
    wsl_out = f'/tmp/zz_dssp_{pdb_stem}.dssp'

    check = wsl.run(f'test -x /usr/bin/dssp && echo OK || echo MISSING')
    if not check.success or 'OK' not in check.data:
        logger.warning('DSSP binary not found at /usr/bin/dssp in WSL')
        return None

    run_result = wsl.run(f'dssp -i "{wsl_pdb}" -o "{wsl_out}" 2>&1', timeout=120)
    if not run_result.success:
        logger.warning('DSSP WSL run failed: %s', run_result.error)
        return None

    # Read the output file back from WSL via stdout
    read_result = wsl.run(f'cat "{wsl_out}"')
    if not read_result.success or not read_result.data:
        logger.warning('Could not read DSSP output from WSL: %s', read_result.error)
        return None

    return _parse_dssp_text(read_result.data)


# ── Biopython fallback ────────────────────────────────────────────────────────


def _run_biopython_dssp(pdb_path: str) -> Optional[dict]:
    """Run DSSP via Biopython's wrapper (requires mkdssp in PATH)."""
    try:
        from Bio.PDB import PDBParser, DSSP as BioDSSP
    except ImportError as e:
        logger.warning('Biopython not available: %s', e)
        return None

    try:
        parser = PDBParser(QUIET=True)
        structure = parser.get_structure('prot', pdb_path)
        model = structure[0]
        dssp = BioDSSP(model, pdb_path, dssp='mkdssp')
    except Exception as e:
        logger.warning('Biopython DSSP failed: %s', e)
        return None

    result = {}
    for key in dssp.property_keys:
        try:
            chain_id = key[0]
            resnum = key[1][1]
            record = dssp[key]
            # Biopython DSSP record: (dssp_index, aa, ss, rsa, phi, psi, ...)
            aa = record[1]
            ss_raw = record[2]
            rsa_val = record[3]  # already relative (0-1)

            if aa == 'X' or not aa:
                continue

            ss_code = ss_raw if ss_raw in SS_LABELS else 'C'
            if ss_code == ' ':
                ss_code = 'C'

            max_asa_aa = MAX_ASA.get(aa.upper(), MAX_ASA['X'])
            rsa_clamped = max(0.0, min(1.0, float(rsa_val) if rsa_val is not None else 0.0))
            asa_abs = rsa_clamped * max_asa_aa

            result[int(resnum)] = {
                'ss': ss_code,
                'ss_label': get_ss_label(ss_code),
                'asa': round(asa_abs, 2),
                'rsa': round(rsa_clamped, 4),
                'aa': aa.upper(),
                'chain': chain_id,
            }
        except Exception as e:
            logger.warning('Biopython DSSP record parse error at key %s: %s', key, e)
            continue

    return result if result else None


# ── pydssp pure-Python fallback ───────────────────────────────────────────────

# pydssp c3 output → DSSP-style code
_PYDSSP_C3_TO_CODE = {'H': 'H', 'E': 'E', 'C': 'C'}


def _run_pydssp(pdb_path: str) -> 'dict | None':
    """Compute secondary structure via pydssp (pure Python, no external binary).

    Only provides SS assignment (H/E/C); RSA is set to 0.0 (unknown).
    Requires complete backbone atoms (N, CA, C, O) per residue — automatically
    skips residues with missing backbone atoms.
    """
    try:
        import pydssp
        import numpy as np
    except ImportError:
        logger.warning('pydssp not installed; skipping')
        return None

    try:
        # ── Parse PDB: collect backbone atoms per residue ──────────────────
        AA3TO1 = {
            'ALA': 'A', 'ARG': 'R', 'ASN': 'N', 'ASP': 'D', 'CYS': 'C',
            'GLN': 'Q', 'GLU': 'E', 'GLY': 'G', 'HIS': 'H', 'ILE': 'I',
            'LEU': 'L', 'LYS': 'K', 'MET': 'M', 'PHE': 'F', 'PRO': 'P',
            'SER': 'S', 'THR': 'T', 'TRP': 'W', 'TYR': 'Y', 'VAL': 'V',
            'HSD': 'H', 'HSE': 'H', 'HSP': 'H', 'MSE': 'M',
        }
        BACKBONE = {' N  ': 0, ' CA ': 1, ' C  ': 2, ' O  ': 3}

        # resnum_str+chain → {atom_idx: (x,y,z)}, aa
        from collections import OrderedDict
        residues: 'OrderedDict[tuple, dict]' = OrderedDict()

        with open(pdb_path, encoding='utf-8', errors='replace') as fh:
            for line in fh:
                if not line.startswith('ATOM'):
                    continue
                res_name = line[17:20].strip()
                if res_name not in AA3TO1:
                    continue
                atom_name = line[12:16]
                atom_idx = BACKBONE.get(atom_name)
                if atom_idx is None:
                    continue
                try:
                    resnum = int(line[22:26].strip())
                    chain  = line[21]
                    x = float(line[30:38])
                    y = float(line[38:46])
                    z = float(line[46:54])
                except (ValueError, IndexError):
                    continue
                key = (resnum, chain)
                if key not in residues:
                    residues[key] = {'aa': AA3TO1[res_name], 'atoms': {}}
                residues[key]['atoms'][atom_idx] = (x, y, z)

        # ── Keep only complete residues (all 4 backbone atoms present) ────
        complete_keys = [
            k for k, v in residues.items() if len(v['atoms']) == 4
        ]
        if not complete_keys:
            logger.warning('pydssp: no complete backbone residues found')
            return None

        # Build coords array (N_complete, 4, 3) in N/CA/C/O order
        coords = np.array([
            [residues[k]['atoms'][i] for i in range(4)]
            for k in complete_keys
        ], dtype=np.float32)

        # ── Run pydssp ───────────────────────────────────────────────────
        ss_c3 = pydssp.assign(coords, out_type='c3')  # array of 'H','E','C'

        # ── Build output dict ────────────────────────────────────────────
        result = {}
        for (resnum, chain), ss_char in zip(complete_keys, ss_c3):
            aa = residues[(resnum, chain)]['aa']
            ss_code = _PYDSSP_C3_TO_CODE.get(ss_char, 'C')
            result[resnum] = {
                'ss':        ss_code,
                'ss_label':  get_ss_label(ss_code),
                'asa':       None,  # pydssp does not compute ASA
                'rsa':       None,  # unknown — do NOT use for buried/surface classification
                'aa':        aa,
                'chain':     chain,
                'rsa_source': 'pydssp_no_rsa',  # flag: RSA unavailable, skip RSA-based zone
            }
        logger.info('pydssp: annotated %d residues for %s', len(result), pdb_path)
        return result if result else None

    except Exception as exc:
        logger.warning('pydssp run failed: %s', exc)
        return None


# ── DSSP text parser ──────────────────────────────────────────────────────────


def _parse_dssp_text(dssp_text: str) -> Optional[dict]:
    """
    Parse classic DSSP text format into {resnum: {...}} dict.

    Column layout (1-indexed, inclusive):
      7-11   residue number
      12     chain
      14     one-letter amino acid
      17     secondary structure code
      36-38  ACC (solvent accessible area in Å²)
    """
    result = {}
    in_residues = False

    for line in dssp_text.splitlines():
        if not in_residues:
            if line.strip().startswith('#') and 'RESIDUE' in line:
                in_residues = True
            continue

        if len(line) < 38:
            continue

        try:
            resnum_str = line[6:11].strip()
            if not resnum_str or not resnum_str.lstrip('-').isdigit():
                continue

            resnum = int(resnum_str)
            chain = line[11].strip()
            aa = line[13].strip()

            if not aa or aa == '!':
                # Chain break marker
                continue

            ss_raw = line[16]
            acc_str = line[35:38].strip()

            if not acc_str.isdigit():
                continue

            acc = int(acc_str)
            ss_code = ss_raw if ss_raw in SS_LABELS else 'C'
            if ss_code == ' ':
                ss_code = 'C'

            max_asa_aa = MAX_ASA.get(aa.upper(), MAX_ASA['X'])
            rsa = round(min(1.0, acc / max_asa_aa), 4) if max_asa_aa > 0 else 0.0

            result[resnum] = {
                'ss': ss_code,
                'ss_label': get_ss_label(ss_code),
                'asa': float(acc),
                'rsa': rsa,
                'aa': aa.upper(),
                'chain': chain if chain else 'A',
            }

        except (ValueError, IndexError) as e:
            logger.warning('DSSP line parse error: %s | line: %r', e, line)
            continue

    return result if result else None


# ── BioPython ShrakeRupley SASA 计算 ─────────────────────────────────────────


def _run_shrake_rupley(pdb_path: str) -> 'dict | None':
    """用 BioPython ShrakeRupley 算法计算每残基 SASA 与 RSA。

    仅提供 RSA/ASA，不提供二级结构。用于补充 pydssp 缺失的 RSA 数据。
    返回 {resnum: {'asa': float, 'rsa': float}} 或 None。
    """
    try:
        from Bio.PDB import PDBParser
        from Bio.PDB.SASA import ShrakeRupley
        import warnings
        warnings.filterwarnings('ignore')
    except ImportError:
        logger.warning('BioPython 不可用，无法运行 ShrakeRupley')
        return None

    try:
        parser = PDBParser(QUIET=True)
        structure = parser.get_structure('p', pdb_path)
        sr = ShrakeRupley()
        sr.compute(structure, level='R')

        result = {}
        model = structure[0]
        for chain in model.get_chains():
            for res in chain.get_residues():
                res_id = res.get_id()
                if res_id[0] != ' ':   # 跳过 HETATM（配体、水分子等）
                    continue
                resnum = res_id[1]
                aa3 = res.get_resname().strip()
                aa1 = {
                    'ALA': 'A', 'ARG': 'R', 'ASN': 'N', 'ASP': 'D', 'CYS': 'C',
                    'GLN': 'Q', 'GLU': 'E', 'GLY': 'G', 'HIS': 'H', 'ILE': 'I',
                    'LEU': 'L', 'LYS': 'K', 'MET': 'M', 'PHE': 'F', 'PRO': 'P',
                    'SER': 'S', 'THR': 'T', 'TRP': 'W', 'TYR': 'Y', 'VAL': 'V',
                    'HSD': 'H', 'HSE': 'H', 'HSP': 'H', 'MSE': 'M',
                }.get(aa3, 'X')
                asa = getattr(res, 'sasa', None)
                if asa is None:
                    continue
                max_asa = MAX_ASA.get(aa1, MAX_ASA['X'])
                rsa = round(min(1.0, max(0.0, asa / max_asa)), 4) if max_asa > 0 else 0.0
                result[resnum] = {'asa': round(asa, 2), 'rsa': rsa}

        return result if result else None
    except Exception as e:
        logger.warning('ShrakeRupley 计算失败: %s', e)
        return None
