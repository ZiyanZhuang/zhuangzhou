"""
CGenFF ligand parameterization for CHARMM36 + GROMACS.

Workflow:
  1. Ligand PDB/SDF → MOL2 (OpenBabel, with hydrogens + correct bond orders)
  2. MOL2 → CGenFF .str file (offline binary or web scraper)
  3. .str → GROMACS .itp/.prm (cgenff_charmm2gmx.py, Lemkul Lab)

Produces ligand topology fully compatible with CHARMM36 protein parameters.

References:
  - Vanommeslaeghe et al. 2010 JCC 31:671 (CGenFF)
  - Wacha & Lemkul 2023 JCIM 63:4246 (cgenff_charmm2gmx)
"""

import logging
import os
import re
import shutil
import subprocess
from pathlib import Path

from .config import OBABEL_EXE, TIMEOUT_SETUP

logger = logging.getLogger(__name__)

_HERE = Path(__file__).parent
_CHARMM2GMX_SCRIPT = _HERE / 'cgenff_charmm2gmx' / 'cgenff_charmm2gmx.py'


# ── CGenFF tool detection ────────────────────────────────────────────

def _find_cgenff_binary():
    """Find CGenFF standalone binary (SilcsBio academic license)."""
    cgenff_exe = os.environ.get('CGENFF_EXE')
    if cgenff_exe and os.path.isfile(cgenff_exe):
        return cgenff_exe

    # Check environment
    silcsdir = os.environ.get('SILCSBIODIR', '')
    if silcsdir:
        cgenff = os.path.join(silcsdir, 'programs', 'cgenff')
        if os.path.isfile(cgenff):
            return cgenff

    # Check common locations
    for path in [
        shutil.which('cgenff'),
        r'C:\SilcsBio\programs\cgenff.exe',
    ]:
        if path and os.path.isfile(path):
            return path

    return None


def _find_charmm36_ff():
    """Find CHARMM36 force field directory in GMXLIB."""
    gmxlib = os.environ.get('GMXLIB', '')
    if not gmxlib or not os.path.isdir(gmxlib):
        return None

    # Search for charmm36*.ff directories
    for entry in sorted(Path(gmxlib).iterdir(), reverse=True):
        if entry.is_dir() and entry.name.startswith('charmm36') and entry.name.endswith('.ff'):
            return str(entry)

    return None


# ── Penalty score parsing ─────────────────────────────────────────────

def parse_penalty_scores(str_path):
    """Parse CGenFF .str file for penalty scores.

    Returns:
        dict with 'param_penalty', 'charge_penalty', 'max_penalty',
              'high_penalty_params' (list), 'quality' ('good'/'fair'/'poor')
    """
    result = {
        'param_penalty': 0.0,
        'charge_penalty': 0.0,
        'max_penalty': 0.0,
        'high_penalty_params': [],
        'quality': 'good',
    }

    with open(str_path, encoding='utf-8', errors='replace') as f:
        for line in f:
            # RESI line: param penalty= X ; charge penalty= Y
            m = re.search(r'param\s+penalty\s*=\s*([\d.]+)', line)
            if m:
                result['param_penalty'] = float(m.group(1))
            m = re.search(r'charge\s+penalty\s*=\s*([\d.]+)', line)
            if m:
                result['charge_penalty'] = float(m.group(1))

            # Per-parameter penalties
            m = re.search(r'penalty\s*=\s*([\d.]+)', line)
            if m:
                penalty = float(m.group(1))
                if penalty > result['max_penalty']:
                    result['max_penalty'] = penalty
                if penalty > 50:
                    result['high_penalty_params'].append({
                        'line': line.strip(),
                        'penalty': penalty,
                    })

    # Quality classification
    if result['max_penalty'] > 50:
        result['quality'] = 'poor'
    elif result['max_penalty'] > 10:
        result['quality'] = 'fair'
    else:
        result['quality'] = 'good'

    return result


# ── MOL2 preparation ──────────────────────────────────────────────────

def prepare_mol2(input_path, output_mol2, resname='LIG'):
    """Convert PDB/SDF to MOL2 with correct hydrogens and bond orders.

    CGenFF requires MOL2 with:
    - All hydrogens present
    - Correct bond orders (single/double/aromatic)
    - Correct protonation state

    Args:
        input_path: PDB or SDF file
        output_mol2: output MOL2 path
        resname: residue name to set in MOL2

    Returns:
        (mol2_path, error) — error is None on success
    """
    if not OBABEL_EXE or not os.path.isfile(OBABEL_EXE):
        return None, f'OpenBabel not found at {OBABEL_EXE}'

    # Convert with hydrogen addition and 3D coordinate generation
    cmd = [OBABEL_EXE, input_path, '-O', output_mol2,
           '-h',              # add hydrogens
           '--gen3d',         # generate 3D coords if needed
           '--title', resname]

    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=60)
    except Exception as e:
        return None, f'OpenBabel failed: {e}'

    if not os.path.exists(output_mol2):
        return None, f'OpenBabel did not produce MOL2: {result.stderr}'

    # Fix residue name in MOL2 to match resname
    _fix_mol2_resname(output_mol2, resname)

    logger.info("Prepared MOL2: %s (%s)", output_mol2, resname)
    return output_mol2, None


def _fix_mol2_resname(mol2_path, resname):
    """Set residue name in MOL2 @<TRIPOS>ATOM section."""
    with open(mol2_path, 'r', encoding='utf-8', errors='replace') as f:
        content = f.read()

    # In MOL2, residue info is in column 8 of @<TRIPOS>ATOM section
    # Format: atom_id atom_name x y z atom_type [subst_id subst_name charge]
    lines = content.split('\n')
    in_atom = False
    new_lines = []
    for line in lines:
        if '@<TRIPOS>ATOM' in line:
            in_atom = True
            new_lines.append(line)
            continue
        if line.startswith('@<TRIPOS>') and 'ATOM' not in line:
            in_atom = False

        if in_atom and line.strip():
            parts = line.split()
            if len(parts) >= 8:
                # Replace subst_name (column 8, 0-indexed 7)
                parts[7] = resname
                line = '  '.join(parts)
        new_lines.append(line)

    with open(mol2_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(new_lines))


# ── CGenFF .str generation ────────────────────────────────────────────

def generate_str(mol2_path, str_path, username=None, password=None):
    """Generate CGenFF .str file from MOL2.

    Tries in order:
    1. Pre-existing .str file (user-provided)
    2. Local CGenFF binary (SilcsBio academic license)
    3. ParamChem/CGenFF web service (requires account)

    Args:
        mol2_path: path to MOL2 file
        str_path: desired output .str path
        username: ParamChem/CGenFF account email (optional)
        password: ParamChem/CGenFF account password (optional)

    Returns:
        (str_path, error) — error is None on success
    """
    # Check for pre-existing .str file
    if os.path.exists(str_path):
        logger.info("Using existing CGenFF .str file: %s", str_path)
        return str_path, None

    # Try local binary
    cgenff_bin = _find_cgenff_binary()
    if cgenff_bin:
        return _run_cgenff_binary(cgenff_bin, mol2_path, str_path)

    # Try ParamChem web service
    if username and password:
        result = _fetch_str_from_paramchem(mol2_path, str_path, username, password)
        if result[0]:
            return result

    # Read credentials from environment first. Legacy plaintext file support is
    # intentionally not advertised because it is easy to copy into a repo.
    cfg_user = os.environ.get('CGENFF_USER', '')
    cfg_pass = os.environ.get('CGENFF_PASS', '')
    if not cfg_user:
        from pathlib import Path
        cred_file = Path.home() / '.zhuangzhou' / 'cgenff_credentials'
        if cred_file.exists():
            try:
                if os.name != 'nt' and (cred_file.stat().st_mode & 0o077):
                    logger.warning("Ignoring world/group-readable CGenFF credentials file: %s", cred_file)
                    lines = []
                else:
                    lines = cred_file.read_text().strip().split('\n')
            except OSError:
                lines = []
            if len(lines) >= 2:
                cfg_user = lines[0].strip()
                cfg_pass = lines[1].strip()

    if cfg_user and cfg_pass:
        result = _fetch_str_from_paramchem(mol2_path, str_path, cfg_user, cfg_pass)
        if result[0]:
            return result

    return None, (
        'CGenFF parameter generation requires one of:\n'
        '  1. CGenFF binary (set SILCSBIODIR env var, free academic license from silcsbio.com)\n'
        '  2. Pre-generated .str file (upload MOL2 to https://cgenff.com/ and download .str)\n'
        '  3. ParamChem account credentials:\n'
        '     - Set CGENFF_USER and CGENFF_PASS environment variables\n'
        '     - Register free at https://cgenff.com/ (academic email required)\n'
        f'Place the .str file at: {str_path}'
    )


def _fetch_str_from_paramchem(mol2_path, str_path, username, password):
    """Fetch CGenFF .str file from ParamChem/CGenFF web service.

    Uses mechanize to automate the web form submission.
    Requires a free academic account at https://cgenff.com/.
    """
    try:
        import mechanize
    except ImportError:
        return None, 'mechanize package required for web access (pip install mechanize)'

    logger.info("Fetching CGenFF parameters from web service...")
    try:
        br = mechanize.Browser()
        br.set_handle_robots(False)
        br.set_handle_refresh(False)

        # Login to CGenFF
        br.open('https://cgenff.com/login/')
        br.select_form(nr=0)
        br['username'] = username
        br['password'] = password
        response = br.submit()

        if 'login' in response.geturl().lower():
            return None, 'CGenFF login failed — check credentials'

        # Upload MOL2
        br.open('https://cgenff.com/initguess/')
        br.select_form(nr=0)
        br.form.add_file(
            open(mol2_path, 'rb'),
            'chemical/x-mol2',
            os.path.basename(mol2_path),
        )
        response = br.submit()

        # Extract .str content from response
        content = response.read().decode('utf-8', errors='replace')

        # The response page contains the .str file content
        # Look for CHARMM stream file markers
        if 'RESI' in content and 'END' in content:
            # Extract stream file content between markers
            str_content = _extract_str_from_html(content)
            if str_content:
                with open(str_path, 'w', encoding='utf-8') as f:
                    f.write(str_content)
                logger.info("CGenFF .str downloaded from web service: %s", str_path)
                return str_path, None

        return None, 'CGenFF web service did not return valid .str content'

    except Exception as e:
        return None, f'CGenFF web service error: {e}'


def _extract_str_from_html(html_content):
    """Extract CGenFF .str content from HTML response."""
    try:
        from bs4 import BeautifulSoup
        soup = BeautifulSoup(html_content, 'html.parser')

        # Look for <pre> or <textarea> containing the stream file
        for tag in soup.find_all(['pre', 'textarea']):
            text = tag.get_text()
            if 'RESI' in text or 'read rtf' in text.lower():
                return text

        # Try to find a download link
        for link in soup.find_all('a', href=True):
            if '.str' in link['href']:
                return None  # Would need to download the file

    except ImportError:
        # Fallback: simple regex extraction
        import re
        m = re.search(r'(read rtf.*?END)', html_content, re.DOTALL | re.IGNORECASE)
        if m:
            return m.group(1)

    return None


def _run_cgenff_binary(cgenff_bin, mol2_path, str_path):
    """Run CGenFF standalone binary."""
    try:
        result = subprocess.run(
            [cgenff_bin, mol2_path],
            capture_output=True, text=True, timeout=120)
        if result.returncode != 0:
            return None, f'CGenFF binary failed: {result.stderr}'

        with open(str_path, 'w', encoding='utf-8') as f:
            f.write(result.stdout)

        logger.info("CGenFF .str generated: %s", str_path)
        return str_path, None
    except Exception as e:
        return None, f'CGenFF binary error: {e}'


# ── GROMACS topology conversion ───────────────────────────────────────

def str_to_gmx(resname, mol2_path, str_path, ff_dir, work_dir):
    """Convert CGenFF .str to GROMACS .itp/.prm using cgenff_charmm2gmx.py.

    Args:
        resname: ligand residue name (e.g., 'FLP')
        mol2_path: path to ligand MOL2 file
        str_path: path to CGenFF .str file
        ff_dir: path to charmm36*.ff directory
        work_dir: output directory

    Returns:
        dict with 'itp', 'prm', 'pdb', 'top' paths, or 'error'
    """
    if not _CHARMM2GMX_SCRIPT.exists():
        return {'error': f'cgenff_charmm2gmx.py not found at {_CHARMM2GMX_SCRIPT}'}

    # cgenff_charmm2gmx.py outputs files in the current directory
    # Run from work_dir so outputs land there
    cmd = [
        'python', str(_CHARMM2GMX_SCRIPT),
        resname,
        os.path.abspath(mol2_path),
        os.path.abspath(str_path),
        os.path.abspath(ff_dir),
    ]

    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True,
            timeout=TIMEOUT_SETUP, cwd=work_dir)
    except Exception as e:
        return {'error': f'cgenff_charmm2gmx.py failed: {e}'}

    output = (result.stdout or '') + (result.stderr or '')

    # Check outputs
    itp = os.path.join(work_dir, f'{resname.lower()}.itp')
    prm = os.path.join(work_dir, f'{resname.lower()}.prm')
    pdb = os.path.join(work_dir, f'{resname.lower()}_ini.pdb')

    # cgenff_charmm2gmx uses lowercase resname for output files
    # Try exact case too
    if not os.path.exists(itp):
        itp = os.path.join(work_dir, f'{resname}.itp')
    if not os.path.exists(prm):
        prm = os.path.join(work_dir, f'{resname}.prm')
    if not os.path.exists(pdb):
        pdb = os.path.join(work_dir, f'{resname}_ini.pdb')

    if not os.path.exists(itp):
        return {'error': f'cgenff_charmm2gmx.py did not produce .itp:\n{output[-500:]}'}

    logger.info("CGenFF → GROMACS conversion complete: %s", itp)
    return {
        'itp': itp,
        'prm': prm if os.path.exists(prm) else None,
        'pdb': pdb if os.path.exists(pdb) else None,
        'resname': resname,
    }


# ── Coordinate conversion (PDB → GRO) ────────────────────────────────

def pdb_to_gro(pdb_path, gro_path):
    """Convert PDB to GRO format using gmx editconf."""
    from .prepare import run_gmx
    work_dir = os.path.dirname(gro_path)
    gro_name = os.path.basename(gro_path)
    cmd = f'editconf -f "{pdb_path}" -o "{gro_name}"'
    output, ok = run_gmx(cmd, work_dir)
    if not ok:
        return None, f'editconf PDB→GRO failed: {output[-300:]}'
    return gro_path, None


# ── Full CGenFF parameterization pipeline ─────────────────────────────

def parameterize_ligand_cgenff(work_dir, ligand_pdb, resname='LIG',
                                str_file=None, ff_dir=None):
    """Full CGenFF parameterization pipeline for a small molecule ligand.

    Steps:
      1. PDB/SDF → MOL2 (OpenBabel + hydrogens)
      2. MOL2 → CGenFF .str (offline binary or user-provided)
      3. .str → GROMACS .itp + .prm (cgenff_charmm2gmx.py)
      4. Parse penalty scores for quality assessment
      5. Generate .gro coordinates

    Args:
        work_dir: output directory
        ligand_pdb: ligand PDB or SDF file
        resname: 3-letter residue name
        str_file: pre-generated CGenFF .str file (optional)
        ff_dir: CHARMM36 FF directory (auto-detected if None)

    Returns:
        dict with 'itp', 'prm', 'gro', 'penalties', or 'error'/'hint'
    """
    os.makedirs(work_dir, exist_ok=True)

    # Auto-detect CHARMM36 FF
    if ff_dir is None:
        ff_dir = _find_charmm36_ff()
    if not ff_dir or not os.path.isdir(ff_dir):
        return {
            'error': 'CHARMM36 force field not found',
            'hint': ('Download from http://mackerell.umaryland.edu/charmm_ff.shtml '
                     'and extract to GROMACS top directory'),
        }

    # Step 1: Prepare MOL2
    mol2_path = os.path.join(work_dir, f'{resname}.mol2')
    mol2_path, err = prepare_mol2(ligand_pdb, mol2_path, resname)
    if err:
        return {'error': f'MOL2 preparation failed: {err}'}

    # Step 2: Generate CGenFF .str
    str_path = str_file or os.path.join(work_dir, f'{resname}.str')
    str_path, err = generate_str(mol2_path, str_path)
    if err:
        return {
            'error': f'CGenFF .str generation failed',
            'hint': err,
            'mol2': mol2_path,  # User can upload this to cgenff.com
        }

    # Step 3: Parse penalty scores
    penalties = parse_penalty_scores(str_path)
    if penalties['quality'] == 'poor':
        logger.warning(
            "CGenFF penalty score %.1f > 50 — parameters may be unreliable! "
            "Consider QM refinement with FFParam/FFTK.",
            penalties['max_penalty'])

    # Step 4: Convert .str → GROMACS .itp/.prm
    gmx_result = str_to_gmx(resname, mol2_path, str_path, ff_dir, work_dir)
    if 'error' in gmx_result:
        return gmx_result

    # Step 5: Generate .gro coordinates
    gro_path = os.path.join(work_dir, f'{resname}.gro')
    ini_pdb = gmx_result.get('pdb')
    if ini_pdb and os.path.exists(ini_pdb):
        gro_path, err = pdb_to_gro(ini_pdb, gro_path)
        if err:
            # Fallback: convert original ligand PDB
            gro_path, err = pdb_to_gro(ligand_pdb, gro_path)
    else:
        gro_path, err = pdb_to_gro(ligand_pdb, gro_path)

    if err:
        return {'error': f'GRO generation failed: {err}'}

    # Fix resname in GRO
    _fix_gro_resname(gro_path, resname)

    # Validate charge sum (CGenFF should produce integer total charge)
    _validate_itp_charge(gmx_result['itp'], resname)

    logger.info("CGenFF parameterization complete: %s (quality=%s, max_penalty=%.1f)",
                resname, penalties['quality'], penalties['max_penalty'])

    return {
        'itp': gmx_result['itp'],
        'prm': gmx_result.get('prm'),
        'gro': gro_path,
        'mol2': mol2_path,
        'str': str_path,
        'penalties': penalties,
        'resname': resname,
        'ff_dir': ff_dir,
    }


def _validate_itp_charge(itp_path, resname):
    """Warn if total charge in .itp deviates from integer."""
    total = 0.0
    in_atoms = False
    with open(itp_path, encoding='utf-8', errors='replace') as f:
        for line in f:
            s = line.strip()
            if '[ atoms ]' in s:
                in_atoms = True
                continue
            if in_atoms and s.startswith('['):
                break
            if in_atoms and s and not s.startswith(';'):
                parts = s.split()
                if len(parts) >= 7:
                    try:
                        total += float(parts[6])
                    except (ValueError, IndexError):
                        pass
    nearest = round(total)
    if abs(total - nearest) > 0.01:
        logger.warning(
            "CGenFF %s: total charge %.4f not integer (nearest %d). "
            "Check .str file.", resname, total, nearest)


def _fix_gro_resname(gro_path, resname):
    """Fix residue name in .gro file to match target."""
    if not os.path.exists(gro_path):
        return
    with open(gro_path, 'r', encoding='utf-8', errors='replace') as f:
        lines = f.readlines()
    new_lines = []
    for line in lines:
        if len(line) > 20:
            # GRO format: resid(5) resname(5) atomname(5) atomnr(5) ...
            current_res = line[5:10].strip()
            if current_res and current_res != resname:
                line = line[:5] + f'{resname:>5s}' + line[10:]
        new_lines.append(line)
    with open(gro_path, 'w', encoding='utf-8') as f:
        f.writelines(new_lines)


# ── Topology integration helpers ──────────────────────────────────────

def integrate_cgenff_topology(topol_path, itp_path, prm_path, resname):
    """Insert CGenFF ligand topology into system topol.top.

    For CGenFF, the topology structure is:
      #include "charmm36.ff/forcefield.itp"
      #include "RESNAME.prm"           ← CGenFF-specific parameters
      ...protein topology...
      #include "RESNAME.itp"           ← ligand moleculetype
      [ system ]
      ...
      [ molecules ]
      Protein_chain_A  1
      RESNAME           1              ← add ligand

    Args:
        topol_path: path to topol.top
        itp_path: CGenFF .itp file
        prm_path: CGenFF .prm file (may be None)
        resname: ligand residue name
    """
    with open(topol_path, 'r', encoding='utf-8', errors='replace') as f:
        content = f.read()

    # Insert .prm after forcefield.itp include
    if prm_path and os.path.exists(prm_path):
        prm_basename = os.path.basename(prm_path)
        ff_include_end = content.find('.ff/forcefield.itp"')
        if ff_include_end != -1:
            ff_include_end = content.find('\n', ff_include_end) + 1
            content = (content[:ff_include_end] +
                       f'\n; CGenFF parameters for {resname}\n'
                       f'#include "{prm_basename}"\n' +
                       content[ff_include_end:])

    # Insert .itp before [ system ]
    itp_basename = os.path.basename(itp_path)
    insert_point = content.find('[ system ]')
    if insert_point == -1:
        insert_point = content.find('[system]')
    if insert_point != -1:
        content = (content[:insert_point] +
                   f'; CGenFF ligand topology\n'
                   f'#include "{itp_basename}"\n\n' +
                   content[insert_point:])

    # Add molecule to [ molecules ]
    content = content.rstrip() + f'\n{resname:20s} 1\n'

    with open(topol_path, 'w', encoding='utf-8') as f:
        f.write(content)

    logger.info("Integrated CGenFF ligand %s into topology", resname)
