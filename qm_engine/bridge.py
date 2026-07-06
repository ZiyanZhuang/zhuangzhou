"""
WSL Bridge for PySCF / gpu4pyscf / geomeTRIC.

All QM calculations run in WSL via subprocess.
Input/output via JSON + XYZ files on the shared filesystem.

Key design:
- Scripts are generated as pure-ASCII Python files
- Enzyme solvation: CPCM with eps=4 (Himo, Siegbahn standard)
- Dispersion: D3(BJ) via pyscf-dftd3
- GPU: try gpu4pyscf, fallback to CPU PySCF
"""

import json
import logging
import os
import subprocess
from typing import Optional

from .config import (
    WSL_PYTHON, DEFAULT_FUNCTIONAL, DEFAULT_BASIS_OPT, DEFAULT_BASIS_SP,
    DEFAULT_DISPERSION, DEFAULT_GRID, DEFAULT_MAX_MEMORY,
    TIMEOUT_OPT, TIMEOUT_TS, TIMEOUT_FREQ, TIMEOUT_SP,
    TRY_GPU, DEFAULT_SOLVENT_EPS, USE_SOLVENT,
)

logger = logging.getLogger(__name__)


def _win_to_wsl(path: str) -> str:
    """Convert Windows path to WSL path."""
    path = os.path.abspath(path).replace('\\', '/')
    if len(path) >= 2 and path[1] == ':':
        drive = path[0].lower()
        return f'/mnt/{drive}{path[2:]}'
    return path


def _run_wsl_pyscf(script: str, work_dir: str, timeout: int = 3600) -> dict:
    """Run a PySCF script in WSL, return parsed JSON result."""
    wsl_work = _win_to_wsl(work_dir)

    # Write script to work dir
    script_path = os.path.join(work_dir, '_pyscf_job.py')
    with open(script_path, 'w', encoding='utf-8', newline='\n') as f:
        f.write(script)

    wsl_script = _win_to_wsl(script_path)

    cmd = ['wsl', 'bash', '-c',
           f'cd "{wsl_work}" && {WSL_PYTHON} "{wsl_script}" 2>&1']

    logger.info("[QM] Running PySCF job in WSL (timeout=%ds)", timeout)
    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout)
        output = result.stdout + (result.stderr or '')

        # Parse JSON result from last lines
        for line in reversed(output.strip().split('\n')):
            line = line.strip()
            if line.startswith('{') and line.endswith('}'):
                try:
                    return json.loads(line)
                except json.JSONDecodeError:
                    pass

        if result.returncode != 0:
            return {'error': 'PySCF failed', 'detail': output[-2000:]}
        return {'error': 'No JSON output from PySCF', 'stdout': output[-2000:]}

    except subprocess.TimeoutExpired:
        return {'error': f'PySCF timeout after {timeout}s',
                'hint': 'Increase timeout or reduce system size'}
    except Exception as e:
        return {'error': str(e)}


# =============================================================================
# Public API
# =============================================================================

def run_dft_optimization(
    xyz_path: str,
    output_dir: str,
    charge: int = 0,
    spin: int = 0,
    functional: str = DEFAULT_FUNCTIONAL,
    basis: str = DEFAULT_BASIS_OPT,
    constraints_path: str = None,
    dispersion: str = DEFAULT_DISPERSION,
    max_memory: int = DEFAULT_MAX_MEMORY,
    try_gpu: bool = TRY_GPU,
    solvent_eps: float = None,
    timeout: int = TIMEOUT_OPT,
) -> dict:
    """Run constrained DFT geometry optimization via PySCF + geomeTRIC."""
    os.makedirs(output_dir, exist_ok=True)
    wsl_xyz = _win_to_wsl(xyz_path)
    wsl_out = _win_to_wsl(output_dir)
    wsl_constraints = _win_to_wsl(constraints_path) if constraints_path else ''
    eps = solvent_eps if solvent_eps is not None else (DEFAULT_SOLVENT_EPS if USE_SOLVENT else 0)

    script = _gen_opt_script(
        wsl_xyz, wsl_out, charge, spin, functional, basis,
        wsl_constraints, dispersion, max_memory, try_gpu, eps, ts=False)

    return _run_wsl_pyscf(script, output_dir, timeout)


def run_ts_search(
    xyz_path: str,
    output_dir: str,
    charge: int = 0,
    spin: int = 0,
    functional: str = DEFAULT_FUNCTIONAL,
    basis: str = DEFAULT_BASIS_OPT,
    constraints_path: str = None,
    dispersion: str = DEFAULT_DISPERSION,
    max_memory: int = DEFAULT_MAX_MEMORY,
    try_gpu: bool = TRY_GPU,
    solvent_eps: float = None,
    timeout: int = TIMEOUT_TS,
) -> dict:
    """Run transition state search via PySCF + geomeTRIC (RS-P-RFO)."""
    os.makedirs(output_dir, exist_ok=True)
    wsl_xyz = _win_to_wsl(xyz_path)
    wsl_out = _win_to_wsl(output_dir)
    wsl_constraints = _win_to_wsl(constraints_path) if constraints_path else ''
    eps = solvent_eps if solvent_eps is not None else (DEFAULT_SOLVENT_EPS if USE_SOLVENT else 0)

    script = _gen_opt_script(
        wsl_xyz, wsl_out, charge, spin, functional, basis,
        wsl_constraints, dispersion, max_memory, try_gpu, eps, ts=True)

    return _run_wsl_pyscf(script, output_dir, timeout)


def run_frequency(
    xyz_path: str,
    output_dir: str,
    charge: int = 0,
    spin: int = 0,
    functional: str = DEFAULT_FUNCTIONAL,
    basis: str = DEFAULT_BASIS_OPT,
    dispersion: str = DEFAULT_DISPERSION,
    max_memory: int = DEFAULT_MAX_MEMORY,
    try_gpu: bool = TRY_GPU,
    solvent_eps: float = None,
    timeout: int = TIMEOUT_FREQ,
) -> dict:
    """Run frequency analysis (Hessian) to verify TS or get thermodynamics."""
    os.makedirs(output_dir, exist_ok=True)
    wsl_xyz = _win_to_wsl(xyz_path)
    wsl_out = _win_to_wsl(output_dir)
    eps = solvent_eps if solvent_eps is not None else (DEFAULT_SOLVENT_EPS if USE_SOLVENT else 0)

    script = _gen_freq_script(
        wsl_xyz, wsl_out, charge, spin, functional, basis,
        dispersion, max_memory, try_gpu, eps)

    return _run_wsl_pyscf(script, output_dir, timeout)


def run_single_point(
    xyz_path: str,
    output_dir: str,
    charge: int = 0,
    spin: int = 0,
    functional: str = DEFAULT_FUNCTIONAL,
    basis: str = DEFAULT_BASIS_SP,
    dispersion: str = DEFAULT_DISPERSION,
    max_memory: int = DEFAULT_MAX_MEMORY,
    try_gpu: bool = TRY_GPU,
    solvent_eps: float = None,
    timeout: int = TIMEOUT_SP,
) -> dict:
    """Run high-basis single-point energy calculation."""
    os.makedirs(output_dir, exist_ok=True)
    wsl_xyz = _win_to_wsl(xyz_path)
    wsl_out = _win_to_wsl(output_dir)
    eps = solvent_eps if solvent_eps is not None else (DEFAULT_SOLVENT_EPS if USE_SOLVENT else 0)

    script = _gen_sp_script(
        wsl_xyz, wsl_out, charge, spin, functional, basis,
        dispersion, max_memory, try_gpu, eps)

    return _run_wsl_pyscf(script, output_dir, timeout)


# =============================================================================
# Script generators
# =============================================================================

def _solvent_block(eps):
    """Generate CPCM solvation code block for PySCF scripts."""
    if not eps or eps <= 1.0:
        return '', ''
    # Import and wrapping code
    setup = f'''
# Implicit solvation: CPCM with eps={eps} (enzyme interior)
from pyscf import solvent
mf = solvent.ddCOSMO(mf)
mf.with_solvent.eps = {eps}
'''
    return setup, 'ddCOSMO'


def _gen_opt_script(xyz, out_dir, charge, spin, xc, basis,
                    constraints, disp, mem, try_gpu, eps, ts=False):
    """Generate PySCF optimization script.

    GPU fallback fix (v0.7.1): dispersion + solvent are applied AFTER GPU
    upgrade attempt, ensuring both CPU and GPU paths have identical corrections.
    Follows ORCA's philosophy: method decorators applied to final mf object.
    """
    ts_flag = 'True' if ts else 'False'
    job_type = 'TS' if ts else 'OPT'
    solvent_code, solvent_name = _solvent_block(eps)

    return f'''#!/usr/bin/env python3
# PySCF {job_type} job - auto-generated by zhuangzhou.qm_engine
import json, os, time, warnings
warnings.filterwarnings('ignore')

from pyscf import gto, dft, lib
import numpy as np

t0 = time.time()
result = {{"job": "{job_type}"}}

# Build molecule
with open("{xyz}", encoding="utf-8") as f:
    lines = f.readlines()
natm = int(lines[0].strip())
atom_str = "\\n".join(lines[2:2+natm])

mol = gto.M(
    atom=atom_str,
    basis="{basis}",
    charge={charge},
    spin={spin},
    verbose=4,
    max_memory={mem},
    output=os.path.join("{out_dir}", "{job_type.lower()}.log"),
)

result["n_atoms"] = mol.natm
result["n_electrons"] = mol.nelectron
result["n_basis"] = mol.nao

# DFT setup
if {spin} > 0:
    mf = dft.UKS(mol)
else:
    mf = dft.RKS(mol)
mf.xc = "{xc}"
mf = mf.density_fit()
mf.grids.atom_grid = {DEFAULT_GRID}

# GPU attempt — BEFORE dispersion/solvent so decorators wrap the final mf
gpu_used = False
if {try_gpu}:
    try:
        from gpu4pyscf.dft import rks as gpu_rks, uks as gpu_uks
        if {spin} > 0:
            mf_gpu = gpu_uks.UKS(mol)
        else:
            mf_gpu = gpu_rks.RKS(mol)
        mf_gpu.xc = "{xc}"
        mf_gpu.grids.atom_grid = {DEFAULT_GRID}
        mf = mf_gpu
        gpu_used = True
    except Exception:
        pass

# Dispersion correction (applied to final mf, CPU or GPU)
disp = "{disp}"
if disp:
    try:
        from pyscf import dftd3
        mf = dftd3.dftd3(mf)
    except ImportError:
        pass
{solvent_code}
result["gpu"] = gpu_used
result["solvent"] = "{solvent_name}" if {eps} > 1.0 else "none"

# Geometry optimization with geomeTRIC
from pyscf.geomopt.geometric_solver import optimize

constraints = "{constraints}" if "{constraints}" else None
params = {{}}
if {ts_flag}:
    params["transition"] = True
    params["hessian"] = "first"
if constraints and os.path.exists(constraints):
    params["constraints"] = constraints

try:
    mol_opt = optimize(mf, **params)
    result["success"] = True
    # density_fit wrapper may not store e_tot; re-evaluate on optimized geometry
    if hasattr(mf, 'e_tot') and float(mf.e_tot) != 0.0:
        result["energy"] = float(mf.e_tot)
    else:
        if {spin} > 0:
            mf2 = dft.UKS(mol_opt)
        else:
            mf2 = dft.RKS(mol_opt)
        mf2.xc = "{xc}"
        mf2 = mf2.density_fit()
        mf2.grids.atom_grid = {DEFAULT_GRID}
        result["energy"] = float(mf2.kernel())
    result["energy_kcal"] = result["energy"] * 627.509

    # Save optimized XYZ
    opt_xyz = os.path.join("{out_dir}", "{job_type.lower()}_optimized.xyz")
    coords = mol_opt.atom_coords(unit='ANG') if hasattr(mol_opt, 'atom_coords') else mol.atom_coords(unit='ANG')
    symbols = [mol.atom_symbol(i) for i in range(mol.natm)]
    with open(opt_xyz, 'w', encoding='utf-8') as f:
        f.write(f"{{mol.natm}}\\n")
        f.write(f"{job_type} E={{result['energy']:.10f}} Ha\\n")
        for sym, c in zip(symbols, coords):
            f.write(f"{{sym:>2s}}  {{c[0]:12.6f}}  {{c[1]:12.6f}}  {{c[2]:12.6f}}\\n")
    result["optimized_xyz"] = opt_xyz

except Exception as e:
    result["success"] = False
    result["error"] = str(e)

result["elapsed_s"] = round(time.time() - t0, 1)
print(json.dumps(result))
'''


def _gen_freq_script(xyz, out_dir, charge, spin, xc, basis, disp, mem,
                     try_gpu, eps):
    """Generate PySCF frequency analysis script."""
    solvent_code, solvent_name = _solvent_block(eps)

    return f'''#!/usr/bin/env python3
# PySCF FREQ job - auto-generated by zhuangzhou.qm_engine
import json, os, time, warnings
warnings.filterwarnings('ignore')

from pyscf import gto, dft, lib
import numpy as np

t0 = time.time()
result = {{"job": "FREQ"}}

with open("{xyz}", encoding="utf-8") as f:
    lines = f.readlines()
natm = int(lines[0].strip())
atom_str = "\\n".join(lines[2:2+natm])

mol = gto.M(
    atom=atom_str, basis="{basis}", charge={charge}, spin={spin},
    verbose=4, max_memory={mem},
    output=os.path.join("{out_dir}", "freq.log"),
)

if {spin} > 0:
    mf = dft.UKS(mol)
else:
    mf = dft.RKS(mol)
mf.xc = "{xc}"
mf = mf.density_fit()
mf.grids.atom_grid = {DEFAULT_GRID}

# Dispersion
disp = "{disp}"
if disp:
    try:
        from pyscf import dftd3
        mf = dftd3.dftd3(mf)
    except ImportError:
        pass
{solvent_code}
mf.kernel()

result["energy"] = float(mf.e_tot)
result["solvent"] = "{solvent_name}" if {eps} > 1.0 else "none"

# Hessian + frequencies
try:
    hess = mf.Hessian().kernel()

    from pyscf.hessian.thermo import harmonic_analysis
    freq_info = harmonic_analysis(mol, hess)
    raw_freqs = freq_info['freq_wavenumber']

    # PySCF stores imaginary freqs as complex: real=0, imag=value
    # Convert: if imag component > 0, frequency is imaginary (negative)
    freqs = []
    for f in raw_freqs:
        if hasattr(f, 'imag') and abs(f.imag) > 1.0:
            freqs.append(-abs(f.imag))  # imaginary -> negative
        else:
            freqs.append(float(f.real))

    result["success"] = True
    result["frequencies_cm1"] = [round(f, 1) for f in freqs]
    result["n_imaginary"] = sum(1 for f in freqs if f < -50)
    result["imaginary_freqs"] = [round(f, 1) for f in freqs if f < -50]
    result["lowest_freq"] = round(min(freqs), 1)
    result["is_minimum"] = result["n_imaginary"] == 0
    result["is_ts"] = result["n_imaginary"] == 1

    # Thermodynamic analysis (ZPE, H, G, S at 298.15 K, 1 atm)
    try:
        from pyscf.hessian.thermo import thermo as _thermo
        thermo_res = _thermo(mf, freq_info['freq_au'], 298.15, 101325)

        E_elec = float(mf.e_tot)
        ZPE    = float(thermo_res['ZPE'][0])
        H_tot  = float(thermo_res['H_tot'][0])
        G_tot  = float(thermo_res['G_tot'][0])
        S_tot  = float(thermo_res['S_tot'][0])

        # Thermal corrections (add to SP energy for best estimate)
        # G_correction = G_tot - E_elec (includes ZPE + thermal + entropy)
        # H_correction = H_tot - E_elec (includes ZPE + thermal enthalpy)
        result["ZPE"] = round(ZPE, 8)
        result["H_tot"] = round(H_tot, 8)
        result["G_tot"] = round(G_tot, 8)
        result["S_tot"] = round(S_tot, 10)
        result["G_correction"] = round(G_tot - E_elec, 8)
        result["H_correction"] = round(H_tot - E_elec, 8)
        result["ZPE_correction"] = round(ZPE, 8)
        result["temperature"] = 298.15
    except Exception as te:
        result["thermo_error"] = str(te)

    # Save frequencies
    freq_path = os.path.join("{out_dir}", "frequencies.txt")
    with open(freq_path, 'w', encoding='utf-8') as f:
        f.write("# Vibrational Frequencies (cm-1)\\n")
        for i, fr in enumerate(freqs):
            marker = " ***IMAGINARY***" if fr < -50 else ""
            f.write(f"  {{i+1:4d}}  {{fr:12.1f}}{{marker}}\\n")
        if "ZPE" in result:
            f.write(f"\\n# Thermodynamics (T=298.15 K, P=1 atm)\\n")
            f.write(f"# ZPE          = {{result['ZPE']:.8f}} Eh\\n")
            f.write(f"# H_tot        = {{result['H_tot']:.8f}} Eh\\n")
            f.write(f"# G_tot        = {{result['G_tot']:.8f}} Eh\\n")
            f.write(f"# G_correction = {{result['G_correction']:.8f}} Eh\\n")
            f.write(f"# H_correction = {{result['H_correction']:.8f}} Eh\\n")
    result["freq_file"] = freq_path

except Exception as e:
    result["success"] = False
    result["error"] = str(e)

result["elapsed_s"] = round(time.time() - t0, 1)
print(json.dumps(result))
'''


def _gen_sp_script(xyz, out_dir, charge, spin, xc, basis, disp, mem,
                   try_gpu, eps):
    """Generate PySCF single-point script."""
    solvent_code, solvent_name = _solvent_block(eps)

    return f'''#!/usr/bin/env python3
# PySCF SP job - auto-generated by zhuangzhou.qm_engine
import json, os, time, warnings
warnings.filterwarnings('ignore')

from pyscf import gto, dft, lib
import numpy as np

t0 = time.time()
result = {{"job": "SP"}}

with open("{xyz}", encoding="utf-8") as f:
    lines = f.readlines()
natm = int(lines[0].strip())
atom_str = "\\n".join(lines[2:2+natm])

mol = gto.M(
    atom=atom_str, basis="{basis}", charge={charge}, spin={spin},
    verbose=4, max_memory={mem},
    output=os.path.join("{out_dir}", "sp.log"),
)

result["n_atoms"] = mol.natm
result["n_basis"] = mol.nao
result["basis"] = "{basis}"

if {spin} > 0:
    mf = dft.UKS(mol)
else:
    mf = dft.RKS(mol)
mf.xc = "{xc}"
mf = mf.density_fit()
mf.grids.atom_grid = {DEFAULT_GRID}

# GPU attempt — BEFORE dispersion/solvent (same fix as OPT script)
gpu_used = False
if {try_gpu}:
    try:
        from gpu4pyscf.dft import rks as gpu_rks, uks as gpu_uks
        if {spin} > 0:
            mf_gpu = gpu_uks.UKS(mol)
        else:
            mf_gpu = gpu_rks.RKS(mol)
        mf_gpu.xc = "{xc}"
        mf_gpu.grids.atom_grid = {DEFAULT_GRID}
        mf = mf_gpu
        gpu_used = True
    except Exception:
        pass

# Dispersion (applied to final mf, CPU or GPU)
disp = "{disp}"
if disp:
    try:
        from pyscf import dftd3
        mf = dftd3.dftd3(mf)
    except ImportError:
        pass
{solvent_code}
result["gpu"] = gpu_used
result["solvent"] = "{solvent_name}" if {eps} > 1.0 else "none"

try:
    e = mf.kernel()
    result["success"] = True
    result["energy"] = float(e)
    result["energy_kcal"] = float(e) * 627.509
except Exception as ex:
    result["success"] = False
    result["error"] = str(ex)

result["elapsed_s"] = round(time.time() - t0, 1)
print(json.dumps(result))
'''
