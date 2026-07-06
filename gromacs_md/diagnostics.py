"""
GROMACS MD Diagnostics — Error extraction, analysis, and sanity checks.

Provides automated analysis of GROMACS log files, GRO coordinates,
and simulation outputs to detect common failures:
  - System explosion (box size > threshold)
  - Energy divergence (NaN, Inf)
  - LINCS constraint warnings
  - Pressure instability (Parrinello-Rahman scaling warnings)
  - Missing output files
  - Coordinate overflow (atoms outside box)

Usage:
    from zhuangzhou.gromacs_md.diagnostics import (
        check_stage_sanity,
        analyze_gmx_log,
        diagnose_failure,
    )
"""

import logging
import os
import re

logger = logging.getLogger(__name__)

# Thresholds (overridable via environment variables)
MAX_BOX_NM = float(os.environ.get('ZZ_MAX_BOX_NM', '50.0'))
MAX_COORD_NM = float(os.environ.get('ZZ_MAX_COORD_NM', '500.0'))
MIN_BOX_NM = float(os.environ.get('ZZ_MIN_BOX_NM', '1.0'))
LINCS_WARN_THRESHOLD = int(os.environ.get('ZZ_LINCS_WARN_THRESHOLD', '20'))


def analyze_gmx_log(log_path):
    """
    Parse a GROMACS log file and extract diagnostics.

    Returns:
        {
            'ok': bool,
            'warnings': [str],
            'errors': [str],
            'lincs_warnings': int,
            'pressure_scaling_warnings': int,
            'nan_detected': bool,
            'fatal_error': str or None,
            'performance_ns_day': float or None,
            'final_step': int or None,
            'energy_terms': dict,  # last reported energies
        }
    """
    result = {
        'ok': True,
        'warnings': [],
        'errors': [],
        'lincs_warnings': 0,
        'pressure_scaling_warnings': 0,
        'nan_detected': False,
        'fatal_error': None,
        'performance_ns_day': None,
        'final_step': None,
        'energy_terms': {},
    }

    if not os.path.exists(log_path):
        result['ok'] = False
        result['errors'].append(f'Log file not found: {log_path}')
        return result

    try:
        with open(log_path, encoding='utf-8', errors='replace') as f:
            content = f.read()
    except Exception as e:
        result['ok'] = False
        result['errors'].append(f'Cannot read log: {e}')
        return result

    lines = content.split('\n')

    # Check for GROMACS fatal errors
    fatal_match = re.search(
        r'Fatal error:\s*\n(.*?)(?:\n\n|\nFor more information)', content, re.DOTALL)
    if fatal_match:
        result['fatal_error'] = fatal_match.group(1).strip()[:500]
        result['ok'] = False
        result['errors'].append(f'Fatal error: {result["fatal_error"][:200]}')

    # Count LINCS warnings
    lincs_warns = re.findall(r'LINCS WARNING', content, re.IGNORECASE)
    result['lincs_warnings'] = len(lincs_warns)
    if result['lincs_warnings'] > LINCS_WARN_THRESHOLD:
        result['ok'] = False
        result['errors'].append(
            f'{result["lincs_warnings"]} LINCS warnings — constraints unstable')

    # Count pressure scaling warnings (Parrinello-Rahman instability)
    pressure_warns = re.findall(r'Pressure scaling more than 1%', content)
    result['pressure_scaling_warnings'] = len(pressure_warns)
    if result['pressure_scaling_warnings'] > 0:
        result['warnings'].append(
            f'{result["pressure_scaling_warnings"]} pressure scaling warnings '
            f'— system not equilibrated for Parrinello-Rahman')
    if result['pressure_scaling_warnings'] > 3:
        result['ok'] = False
        result['errors'].append(
            'Excessive pressure scaling — NPT likely exploded. '
            'Use Berendsen or C-rescale for equilibration.')

    # Check for NaN/Inf in energies
    if re.search(r'\bnan\b|\binf\b', content, re.IGNORECASE):
        # Filter out legitimate uses (like epsilon-rf = inf)
        nan_lines = [l for l in lines
                     if re.search(r'\bnan\b', l, re.IGNORECASE)
                     and 'epsilon' not in l.lower()
                     and 'nan(ind)' not in l.lower()]
        inf_lines = [l for l in lines
                     if re.search(r'\binf\b', l, re.IGNORECASE)
                     and 'epsilon' not in l.lower()
                     and 'information' not in l.lower()
                     and 'Inf.' not in l]
        if nan_lines or inf_lines:
            result['nan_detected'] = True
            result['ok'] = False
            result['errors'].append('NaN/Inf detected in energies')

    # Extract performance
    perf_match = re.search(r'Performance:\s+([\d.]+)\s+', content)
    if perf_match:
        result['performance_ns_day'] = float(perf_match.group(1))

    # Extract final step
    step_matches = re.findall(r'^\s+(\d+)\s+[\d.]+\s*$', content, re.MULTILINE)
    if step_matches:
        result['final_step'] = int(step_matches[-1])

    # Extract last energy block
    energy_blocks = re.findall(
        r'Energies \(kJ/mol\)\n(.*?)(?:\n\n|\nStep)', content, re.DOTALL)
    if energy_blocks:
        last_block = energy_blocks[-1]
        # Parse key-value pairs from the energy block
        # Format: alternating header and value lines
        block_lines = [l for l in last_block.split('\n') if l.strip()]
        for i in range(0, len(block_lines) - 1, 2):
            headers = block_lines[i].split()
            try:
                values = block_lines[i + 1].split()
                for h, v in zip(headers, values):
                    try:
                        result['energy_terms'][h] = float(v)
                    except ValueError:
                        pass
            except IndexError:
                pass

    return result


def check_gro_sanity(gro_path):
    """
    Check a GRO file for coordinate/box sanity.

    Returns:
        {
            'ok': bool,
            'errors': [str],
            'warnings': [str],
            'box_dims': (x, y, z) or None,
            'n_atoms': int,
            'max_coord': float,
        }
    """
    result = {
        'ok': True,
        'errors': [],
        'warnings': [],
        'box_dims': None,
        'n_atoms': 0,
        'max_coord': 0.0,
    }

    if not os.path.exists(gro_path):
        result['ok'] = False
        result['errors'].append(f'GRO file not found: {gro_path}')
        return result

    try:
        with open(gro_path, encoding='utf-8', errors='replace') as f:
            lines = f.readlines()
    except Exception as e:
        result['ok'] = False
        result['errors'].append(f'Cannot read GRO: {e}')
        return result

    if len(lines) < 3:
        result['ok'] = False
        result['errors'].append('GRO file too short')
        return result

    # Parse atom count
    try:
        result['n_atoms'] = int(lines[1].strip())
    except ValueError:
        result['ok'] = False
        result['errors'].append('Cannot parse atom count')
        return result

    # Parse box vectors (last line)
    box_line = lines[-1].strip()
    box_parts = box_line.split()
    if len(box_parts) >= 3:
        try:
            bx, by, bz = float(box_parts[0]), float(box_parts[1]), float(box_parts[2])
            result['box_dims'] = (bx, by, bz)

            # Check box size
            for dim, name in [(bx, 'X'), (by, 'Y'), (bz, 'Z')]:
                if dim > MAX_BOX_NM:
                    result['ok'] = False
                    result['errors'].append(
                        f'Box {name}={dim:.1f} nm > {MAX_BOX_NM} nm — system exploded!')
                elif dim < MIN_BOX_NM:
                    result['ok'] = False
                    result['errors'].append(
                        f'Box {name}={dim:.1f} nm < {MIN_BOX_NM} nm — box collapsed')
        except ValueError:
            result['warnings'].append('Cannot parse box vectors')

    # Spot-check a few atom coordinates for overflow
    max_coord = 0.0
    sample_indices = list(range(2, min(102, len(lines) - 1)))  # first 100 atoms
    sample_indices += list(range(max(2, len(lines) - 102), len(lines) - 1))  # last 100
    for i in sample_indices:
        line = lines[i]
        if len(line) < 44:
            continue
        try:
            x = abs(float(line[20:28]))
            y = abs(float(line[28:36]))
            z = abs(float(line[36:44]))
            max_coord = max(max_coord, x, y, z)
        except (ValueError, IndexError):
            pass

    result['max_coord'] = max_coord
    if max_coord > MAX_COORD_NM:
        result['ok'] = False
        result['errors'].append(
            f'Max coordinate {max_coord:.1f} nm — atoms have flown apart')

    return result


def check_stage_sanity(work_dir, stage_name):
    """
    Run comprehensive sanity checks after a simulation stage.

    Args:
        work_dir: simulation directory
        stage_name: 'em', 'nvt', 'npt', or 'md'

    Returns:
        {
            'ok': bool,
            'errors': [str],
            'warnings': [str],
            'log_analysis': dict,
            'gro_analysis': dict,
            'diagnosis': str,  # human-readable summary
        }
    """
    result = {
        'ok': True,
        'errors': [],
        'warnings': [],
        'log_analysis': None,
        'gro_analysis': None,
        'diagnosis': '',
    }

    # Check log file
    log_path = os.path.join(work_dir, f'{stage_name}.log')
    if os.path.exists(log_path):
        log_info = analyze_gmx_log(log_path)
        result['log_analysis'] = log_info
        result['errors'].extend(log_info['errors'])
        result['warnings'].extend(log_info['warnings'])
        if not log_info['ok']:
            result['ok'] = False

    # Check output GRO
    gro_path = os.path.join(work_dir, f'{stage_name}.gro')
    if os.path.exists(gro_path):
        gro_info = check_gro_sanity(gro_path)
        result['gro_analysis'] = gro_info
        result['errors'].extend(gro_info['errors'])
        result['warnings'].extend(gro_info['warnings'])
        if not gro_info['ok']:
            result['ok'] = False

    # Check expected output files exist
    expected = {
        'em': ['em.gro', 'em.log', 'em.edr'],
        'nvt': ['nvt.gro', 'nvt.log', 'nvt.edr', 'nvt.cpt'],
        'npt': ['npt.gro', 'npt.log', 'npt.edr', 'npt.cpt'],
        'md': ['md.xtc', 'md.log', 'md.edr', 'md.cpt', 'md.gro'],
    }
    for fname in expected.get(stage_name, []):
        fpath = os.path.join(work_dir, fname)
        if not os.path.exists(fpath):
            result['warnings'].append(f'Missing expected file: {fname}')

    # Generate diagnosis
    result['diagnosis'] = _diagnose(result, stage_name)

    if result['errors']:
        logger.warning("[%s] Sanity check FAILED: %s",
                       stage_name, '; '.join(result['errors'][:3]))
    elif result['warnings']:
        logger.info("[%s] Sanity warnings: %s",
                    stage_name, '; '.join(result['warnings'][:3]))

    return result


def diagnose_failure(work_dir, failed_stage=None):
    """
    Comprehensive diagnosis of a failed MD simulation.

    Examines all stages and provides actionable recommendations.

    Returns:
        {
            'stage_results': {stage: sanity_result},
            'root_cause': str,
            'recommendations': [str],
        }
    """
    stages = ['em', 'nvt', 'npt', 'md']
    stage_results = {}
    first_failure = None

    for stage in stages:
        log_path = os.path.join(work_dir, f'{stage}.log')
        gro_path = os.path.join(work_dir, f'{stage}.gro')
        if os.path.exists(log_path) or os.path.exists(gro_path):
            sanity = check_stage_sanity(work_dir, stage)
            stage_results[stage] = sanity
            if not sanity['ok'] and first_failure is None:
                first_failure = stage

    recommendations = []
    root_cause = 'Unknown failure'

    if first_failure:
        fail_info = stage_results[first_failure]
        errors = fail_info['errors']

        if any('exploded' in e for e in errors):
            root_cause = f'System exploded during {first_failure}'
            if first_failure == 'npt':
                recommendations.append(
                    'Use C-rescale or Berendsen barostat for NPT equilibration '
                    '(not Parrinello-Rahman)')
                recommendations.append(
                    'Ensure energy minimization converged properly')
                recommendations.append(
                    'Check for missing bonded parameters (especially metal-ligand)')
            recommendations.append('Reduce timestep to 1 fs during equilibration')
            recommendations.append('Increase position restraint force constants')

        elif any('LINCS' in e for e in errors):
            root_cause = f'LINCS constraint failure during {first_failure}'
            recommendations.append('Reduce timestep (try dt=0.001)')
            recommendations.append('Check for clashes or missing parameters')
            recommendations.append('Increase lincs_iter to 2 and lincs_order to 6')

        elif any('NaN' in e for e in errors):
            root_cause = f'Energy divergence (NaN) during {first_failure}'
            recommendations.append('Check topology for missing parameters')
            recommendations.append('Ensure EM converged (emtol < 1000)')
            recommendations.append('Reduce timestep')

        elif any('Fatal error' in e for e in errors):
            log_info = fail_info.get('log_analysis', {})
            fatal = log_info.get('fatal_error', '') if log_info else ''
            root_cause = f'GROMACS fatal error in {first_failure}: {fatal[:100]}'
            if 'not found' in fatal.lower():
                recommendations.append('Check residue/atom naming matches force field')
            elif 'distance' in fatal.lower():
                recommendations.append('Check distance restraint atom indices')

    elif failed_stage:
        root_cause = f'{failed_stage} failed (no detailed diagnostics available)'
        # Check if it's a timeout issue
        log_path = os.path.join(work_dir, f'{failed_stage}.log')
        if os.path.exists(log_path):
            log_info = analyze_gmx_log(log_path)
            if log_info.get('final_step') is None and not log_info.get('fatal_error'):
                root_cause = f'{failed_stage} likely timed out or crashed silently'
                recommendations.append('Increase timeout or run mdrun manually')
                recommendations.append('Check GPU compatibility (try without -update gpu)')

    return {
        'stage_results': stage_results,
        'root_cause': root_cause,
        'recommendations': recommendations,
        'first_failure': first_failure,
    }


def _diagnose(sanity_result, stage_name):
    """Generate human-readable diagnosis from sanity check."""
    if sanity_result['ok']:
        return f'{stage_name} completed successfully'

    parts = []
    for err in sanity_result['errors']:
        if 'exploded' in err:
            parts.append(f'CRITICAL: {err}')
        elif 'LINCS' in err:
            parts.append(f'UNSTABLE: {err}')
        elif 'NaN' in err or 'Inf' in err:
            parts.append(f'DIVERGED: {err}')
        else:
            parts.append(err)

    return f'{stage_name} FAILED: ' + '; '.join(parts)


def format_diagnosis_report(diagnosis):
    """Format a diagnosis dict as a readable string."""
    lines = ['# MD Simulation Diagnostics', '']
    lines.append(f'**Root cause:** {diagnosis["root_cause"]}')
    lines.append('')

    if diagnosis.get('recommendations'):
        lines.append('## Recommendations')
        for i, rec in enumerate(diagnosis['recommendations'], 1):
            lines.append(f'{i}. {rec}')
        lines.append('')

    lines.append('## Stage Analysis')
    for stage, info in diagnosis.get('stage_results', {}).items():
        status = 'OK' if info['ok'] else 'FAILED'
        lines.append(f'### {stage.upper()} — {status}')
        if info.get('log_analysis'):
            la = info['log_analysis']
            if la.get('performance_ns_day'):
                lines.append(f'  Performance: {la["performance_ns_day"]:.1f} ns/day')
            if la.get('lincs_warnings'):
                lines.append(f'  LINCS warnings: {la["lincs_warnings"]}')
            if la.get('pressure_scaling_warnings'):
                lines.append(f'  Pressure scaling warnings: {la["pressure_scaling_warnings"]}')
        if info.get('gro_analysis') and info['gro_analysis'].get('box_dims'):
            bx, by, bz = info['gro_analysis']['box_dims']
            lines.append(f'  Box: {bx:.2f} x {by:.2f} x {bz:.2f} nm')
        for err in info.get('errors', []):
            lines.append(f'  ERROR: {err}')
        for warn in info.get('warnings', []):
            lines.append(f'  WARN: {warn}')
        lines.append('')

    return '\n'.join(lines)
