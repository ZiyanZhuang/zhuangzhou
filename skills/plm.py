"""PLM (Protein Language Model) skills — ESM-2, ESMFold, Tranception, etc.

Backends:
  - Local: py310_DPLM2 conda env (CPU/GPU) or WSL
  - Cloud: Google Colab T4/L4 GPU via cloudflared HTTP tunnel (v0.9.0)
"""

import os
import sys

from .registry import Skill, SkillParam, SkillRegistry


def _scripts_dir(config=None):
    root = (config or {}).get('unimol_root', 'E:/Code/Uni-Mol-main')
    return os.path.join(root, 'scripts')


def _add_path(config=None):
    sd = _scripts_dir(config)
    if sd not in sys.path:
        sys.path.insert(0, sd)


def esm2_scan(fasta_path: str, output_dir: str = None, model: str = 'esm2_t33_650M_UR50D',
              config: dict = None, **kw) -> dict:
    """Run ESM-2 zero-shot mutation scan using the py310_DPLM2 conda environment.

    Primary: Windows CondaSandbox (py310_DPLM2) — PyTorch + ESM installed.
    Fallback: WSLSandbox (if WSL has ESM installed).
    Script uses positional args: <fasta> <output_dir> [--model] [--device].
    """
    import subprocess
    if output_dir is None:
        stem = os.path.splitext(os.path.basename(fasta_path))[0]
        output_dir = os.path.join(os.getcwd(), f'{stem}_esm2')
    os.makedirs(output_dir, exist_ok=True)
    script = os.path.join(_scripts_dir(config), 'esm_mutation_scan.py')

    # --- Primary: Windows py310_DPLM2 conda environment ---
    from zhuangzhou.envs.sandbox import CONDA_ENVS
    dplm2_info = CONDA_ENVS.get('py310_DPLM2', {})
    python_exe = dplm2_info.get('python', '')
    if python_exe and os.path.isfile(python_exe):
        cmd = [python_exe, script, fasta_path, output_dir,
               '--model', model, '--device', 'cpu']
        try:
            result = subprocess.run(cmd, capture_output=True, text=True,
                                    timeout=3600, errors='replace')
            beneficial_json = os.path.join(output_dir, 'esm2_beneficial_mutations.json')
            return {
                'success': result.returncode == 0,
                'stdout': result.stdout or '',
                'stderr': result.stderr or '',
                'output_dir': output_dir,
                'beneficial_mutations_json': beneficial_json if os.path.isfile(beneficial_json) else None,
                'next_steps': [
                    f'generate_report: 生成突变分析Word报告 → '
                    f'generate_report(report_type="mutation", '
                    f'results_json="{beneficial_json}", '
                    f'output_dir="{output_dir}")',
                ],
            }
        except subprocess.TimeoutExpired:
            return {'error': 'ESM-2 scan timed out (>3600s)', 'output_dir': output_dir}
        except Exception as e:
            pass  # fall through to WSL

    # --- Fallback: WSL ---
    try:
        from zhuangzhou.envs.sandbox import WSLSandbox
        sandbox = WSLSandbox()
        # Convert Windows path to WSL path for all path arguments
        def _to_wsl(p):
            import re
            p = p.replace('\\', '/')
            return re.sub(r'^([A-Za-z]):', lambda m: f'/mnt/{m.group(1).lower()}', p)

        fasta_wsl  = _to_wsl(fasta_path)
        out_wsl    = _to_wsl(output_dir)
        script_wsl = _to_wsl(script)
        cmd_str = f'python3 "{script_wsl}" "{fasta_wsl}" "{out_wsl}" --model {model} --device cpu'
        result = sandbox.run(cmd_str, timeout=3600)
        # SandboxResult uses attribute access, NOT .get()
        beneficial_json = os.path.join(output_dir, 'esm2_beneficial_mutations.json')
        return {
            'success': result.returncode == 0,
            'stdout': result.stdout or '',
            'stderr': result.stderr or '',
            'output_dir': output_dir,
            'beneficial_mutations_json': beneficial_json if os.path.isfile(beneficial_json) else None,
            'next_steps': [
                f'generate_report: 生成突变分析Word报告 → '
                f'generate_report(report_type="mutation", '
                f'results_json="{beneficial_json}", '
                f'output_dir="{output_dir}")',
            ],
        }
    except Exception as e:
        return {'error': f'esm2_scan failed (both primary and WSL): {e}',
                'output_dir': output_dir}


def load_esm2_results(result_dirs: list, config: dict = None, **kw) -> dict:
    """Load pre-computed ESM-2 results from multiple model directories."""
    from zhuangzhou.tools import load_esm2_results as _load
    mutations = _load(result_dirs)
    return {'mutations': mutations, 'n_mutations': len(mutations), 'sources': list(result_dirs)}


# ── Colab ESM Functions (v0.9.0 — HTTP tunnel mode) ──────────────────────

def _read_sequence_from_fasta(fasta_path: str) -> str:
    """Read first sequence from a FASTA file."""
    lines = []
    with open(fasta_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line.startswith('>'):
                if lines:
                    break  # Only first sequence
                continue
            lines.append(line)
    return ''.join(lines)


def colab_esm_scan(sequence: str = None, fasta_path: str = None,
                   positions: list = None, model: str = 'esm2_t33_650M_UR50D',
                   top_k: int = 20, device: str = 'cuda',
                   config: dict = None, **kw) -> dict:
    """Run ESM-2/ESM-1v zero-shot mutation scan on Google Colab GPU.

    Uses the dedicated /esm_scan HTTP endpoint — model stays loaded in GPU
    memory across calls, no repeated loading overhead.
    """
    # Resolve sequence
    if not sequence and fasta_path:
        if not os.path.isfile(fasta_path):
            return {'error': f'FASTA file not found: {fasta_path}'}
        sequence = _read_sequence_from_fasta(fasta_path)
    if not sequence:
        return {'error': 'No sequence provided. Pass sequence= or fasta_path=.'}

    from zhuangzhou.envs.sandbox import get_sandbox
    colab = get_sandbox('colab')
    if not colab.available:
        return {'error': 'Colab tunnel URL not configured. '
                'Set ZZ_COLAB_TUNNEL env var or tunnel_url in '
                '~/.zhuangzhou/colab_config.json. '
                'Run scripts/colab_gpu_server.py in a Colab notebook first.',
                'hint': 'See scripts/colab_gpu_server.py for setup instructions.'}

    # Use dedicated HTTP endpoint (model stays loaded)
    result = colab.esm_scan(
        sequence=sequence,
        positions=positions,
        model=model,
        top_k=top_k,
        timeout=600,
    )
    if not result.success:
        return {'error': f'Colab ESM scan failed: {result.error}',
                'hint': 'Check that the Colab notebook is still running.'}

    data = result.data
    if isinstance(data, dict):
        return data
    return {'error': 'Unexpected response format', 'raw': str(data)[:500]}


def colab_esm_consensus_scan(sequence: str = None, fasta_path: str = None,
                             positions: list = None,
                             models: list = None, top_k: int = 20,
                             config: dict = None, **kw) -> dict:
    """Run multi-model consensus mutation scan on Google Colab GPU.

    Averages log-likelihood ratio scores across multiple ESM models.
    Default: ESM-2 + ESM-1v (model 1).
    For full ESM-1v 5-model ensemble, pass all 5 model names.

    ESM-1v (Meier et al. 2021) was specifically designed for variant effect
    prediction, validated on 41 DMS datasets. ESM-2 provides a complementary
    signal trained on different data (UR50D vs UR90S).
    """
    # Resolve sequence
    if not sequence and fasta_path:
        if not os.path.isfile(fasta_path):
            return {'error': f'FASTA file not found: {fasta_path}'}
        sequence = _read_sequence_from_fasta(fasta_path)
    if not sequence:
        return {'error': 'No sequence provided. Pass sequence= or fasta_path=.'}

    from zhuangzhou.envs.sandbox import get_sandbox
    colab = get_sandbox('colab')
    if not colab.available:
        return {'error': 'Colab tunnel URL not configured. '
                'Set ZZ_COLAB_TUNNEL env var or tunnel_url in '
                '~/.zhuangzhou/colab_config.json. '
                'Run scripts/colab_gpu_server.py in a Colab notebook first.',
                'hint': 'See scripts/colab_gpu_server.py for setup instructions.'}

    result = colab.esm_consensus_scan(
        sequence=sequence,
        positions=positions,
        models=models,
        top_k=top_k,
        timeout=600,
    )
    if not result.success:
        return {'error': f'Colab ESM consensus scan failed: {result.error}',
                'hint': 'Check that the Colab notebook is still running.'}

    data = result.data
    if isinstance(data, dict):
        return data
    return {'error': 'Unexpected response format', 'raw': str(data)[:500]}


def colab_esm_union_scan(sequence: str = None, fasta_path: str = None,
                         positions: list = None,
                         models: list = None, top_k: int = 96,
                         threshold: float = 0.0, weights: list = None,
                         output_dir: str = None,
                         config: dict = None, **kw) -> dict:
    """Union-filtered dual-model scan on Google Colab GPU.

    Implements the Zero_Shot.ipynb wet-lab candidate selection strategy:
      1. Run ESM-2 + ESM-1v independently on Colab GPU
      2. Keep mutations where ANY model score > threshold (union filter)
      3. Classify: Tier 1 (consensus, all positive) vs Tier 2 (single positive)
      4. Sort: Tier 1 first, then by max potential
      5. Export top_k candidates (default 96 for 96-well plate)

    This is the recommended endpoint for selecting mutations for wet-lab validation.
    The union strategy ensures no promising mutation is missed, while tier classification
    guides prioritization of experimental resources.
    """
    # Resolve sequence
    if not sequence and fasta_path:
        if not os.path.isfile(fasta_path):
            return {'error': f'FASTA file not found: {fasta_path}'}
        sequence = _read_sequence_from_fasta(fasta_path)
    if not sequence:
        return {'error': 'No sequence provided. Pass sequence= or fasta_path=.'}

    from zhuangzhou.envs.sandbox import get_sandbox
    colab = get_sandbox('colab')
    if not colab.available:
        return {'error': 'Colab tunnel URL not configured. '
                'Set ZZ_COLAB_TUNNEL env var or tunnel_url in '
                '~/.zhuangzhou/colab_config.json.',
                'hint': 'Run colab_auto_setup or colab_set_url first, or manually run '
                        'scripts/colab_gpu_server.py in a Colab notebook.'}

    result = colab.esm_union_scan(
        sequence=sequence,
        positions=positions,
        models=models,
        top_k=top_k,
        threshold=threshold,
        weights=weights,
        timeout=600,
    )
    if not result.success:
        return {'error': f'Colab ESM union scan failed: {result.error}',
                'hint': 'Check that the Colab notebook is still running.'}

    data = result.data
    if not isinstance(data, dict):
        return {'error': 'Unexpected response format', 'raw': str(data)[:500]}

    # Export CSV if output_dir specified
    if output_dir and data.get('mutations'):
        os.makedirs(output_dir, exist_ok=True)
        import csv
        csv_path = os.path.join(output_dir, 'union_candidates_plate.csv')
        try:
            with open(csv_path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(['Well', 'Mutation', 'Position', 'WT', 'Mutant',
                                 'Tier', 'Max_Potential', 'Weighted_Score',
                                 *[m.split('_')[0] for m in (models or ['esm2', 'esm1v'])]])
                for i, m in enumerate(data['mutations'], 1):
                    well_row = chr(64 + (i - 1) // 12 + 1)  # A-H
                    well_col = (i - 1) % 12 + 1
                    well = f'{well_row}{well_col}'
                    per_model = m.get('per_model_scores', {})
                    writer.writerow([
                        well, m['mutation'], m['position'], m['wt'], m['mut'],
                        m['tier'], m['max_potential'], m['weighted_score'],
                        *[per_model.get(mn, '') for mn in (models or ['esm2_t33_650M_UR50D', 'esm1v_t33_650M_UR90S_1'])],
                    ])
            data['csv_path'] = csv_path
        except Exception as e:
            data['csv_export_error'] = str(e)

    return data


def colab_gpu_info(config: dict = None, **kw) -> dict:
    """Check Google Colab GPU availability and specs."""
    from zhuangzhou.envs.sandbox import get_sandbox
    colab = get_sandbox('colab')
    if not colab.available:
        return {'error': 'Colab tunnel URL not configured. '
                'Run colab_auto_setup or colab_set_url first, or manually run '
                'scripts/colab_gpu_server.py in a Colab notebook.'}

    result = colab.gpu_info(timeout=30)
    if not result.success:
        return {'error': f'Failed to query Colab GPU: {result.error}'}
    return result.data if isinstance(result.data, dict) else {'raw': str(result.data)}


def colab_auto_setup(config: dict = None, **kw) -> dict:
    """Automatically set up Google Colab GPU server.

    Uses colab-mcp to:
      1. Open ONE Colab notebook (one browser tab, OAuth cached)
      2. Inject FastAPI + cloudflared server code
      3. Execute and retrieve tunnel URL
      4. Save URL to config for all subsequent GPU calls

    Prerequisites: pip install uv
    """
    from zhuangzhou.envs.sandbox import get_sandbox
    colab = get_sandbox('colab')
    result = colab.auto_setup(timeout=180)
    if result.success:
        data = result.data if isinstance(result.data, dict) else {}
        return {
            'success': True,
            'tunnel_url': data.get('tunnel_url', ''),
            'status': data.get('status', 'connected'),
            'message': 'Colab GPU server is ready. '
                       'All colab_esm_scan / colab_esm_consensus_scan calls will now use this server.',
        }
    else:
        return {
            'error': f'Colab auto-setup failed: {result.error}',
            'hint': 'Ensure "uv" is installed (pip install uv). '
                    'Check ~/.zhuangzhou/colab_bootstrap.log for details. '
                    'Alternative: manually open scripts/colab_gpu_server.ipynb in Colab.',
        }


def colab_set_url(config: dict = None, **kw) -> dict:
    """Manually set the Colab GPU server tunnel URL.

    Use this when colab_auto_setup fails, or when the user has manually started
    the Colab notebook and has a tunnel URL to paste.
    """
    url = kw.get('url', '') or kw.get('tunnel_url', '')
    if not url:
        return {'error': 'No URL provided. Pass url="https://xxx.trycloudflare.com"'}

    url = url.strip().rstrip('/')
    if not url.startswith('https://'):
        return {'error': f'Invalid URL: {url}. Must start with https://'}

    # Save to config
    from zhuangzhou.envs.colab_config import set_tunnel_url, get_config
    import zhuangzhou.envs.colab_config as _cc
    _cc._config = None  # Reset singleton
    set_tunnel_url(url)

    # Verify connection
    from zhuangzhou.envs.colab_sandbox import ColabSandbox
    sandbox = ColabSandbox()
    sandbox.tunnel_url = url
    result = sandbox.connect(timeout=15)

    if result.success:
        data = result.data if isinstance(result.data, dict) else {}
        return {
            'success': True,
            'tunnel_url': url,
            'gpu_name': data.get('gpu_name', '?'),
            'message': f'Connected to Colab GPU: {data.get("gpu_name", "?")}. '
                       'All colab_esm_* calls will use this server.',
        }
    else:
        return {
            'success': True,
            'tunnel_url': url,
            'warning': f'URL saved but server not reachable yet: {result.error}. '
                       'It may still be starting up — try colab_gpu_info in a few seconds.',
        }


def register_skills(registry: SkillRegistry):
    registry.register(Skill(
        name='esm2_scan',
        description='Run ESM-2 zero-shot mutation scanning via WSL (CPU; GPU used if WSL CUDA available)',
        category='plm',
        fn=esm2_scan,
        inputs=[
            SkillParam('fasta_path', 'path', 'Input FASTA file', required=True),
            SkillParam('output_dir', 'path', 'Output directory (default: CWD/<fasta_stem>_esm2/)',
                       required=False, is_output_path=True),
            SkillParam('model', 'string', 'ESM-2 model name', required=False,
                       default='esm2_t33_650M_UR50D'),
        ],
        outputs=[SkillParam('output_dir', 'path', 'Directory with mutation scores')],
        requires_gpu=False,   # GPU optional inside WSL; CPU fallback works
        requires_wsl=True,
        source='scripts/esm_mutation_scan.py (WSLSandbox)',
    ))

    registry.register(Skill(
        name='esm2_load',
        description='Load and merge ESM-2 multi-model results',
        category='plm',
        fn=load_esm2_results,
        inputs=[
            SkillParam('result_dirs', 'list', 'List of ESM-2 result directories'),
        ],
        outputs=[SkillParam('mutations', 'list', 'Merged mutation list with consensus scores')],
        source='zhuangzhou/tools.py',
    ))

    # ── Colab GPU Skills (v0.9.0 — HTTP tunnel, single notebook) ─────────

    registry.register(Skill(
        name='colab_esm_scan',
        description='Run ESM-2 or ESM-1v zero-shot mutation scan on Google Colab T4/L4 GPU '
                    '(single model, free cloud GPU, model stays loaded). '
                    'For multi-model consensus, use colab_esm_consensus_scan instead.',
        category='plm',
        fn=colab_esm_scan,
        when_to_use='When user wants ESM mutation scanning but has no local GPU, or when local ESM-2 scan fails. '
                     'Requires: Colab notebook running colab_gpu_server.py with tunnel URL configured.',
        inputs=[
            SkillParam('sequence', 'string', 'Protein sequence (or pass fasta_path)', required=False),
            SkillParam('fasta_path', 'path', 'FASTA file path (alternative to sequence)', required=False),
            SkillParam('positions', 'list', 'Specific positions to scan (1-indexed, default: all)', required=False),
            SkillParam('model', 'string', 'ESM model name', required=False,
                       default='esm2_t33_650M_UR50D'),
            SkillParam('top_k', 'integer', 'Return top K beneficial mutations', required=False,
                       default='20'),
        ],
        outputs=[
            SkillParam('mutations', 'list', 'Top mutations with scores'),
            SkillParam('model', 'string', 'Model used'),
            SkillParam('gpu_name', 'string', 'Colab GPU type'),
        ],
        requires_gpu=False,   # GPU is on Colab, not local
        source='Colab GPU server /esm_scan endpoint',
    ))

    registry.register(Skill(
        name='colab_esm_consensus_scan',
        description='Run multi-model consensus ESM mutation scan on Google Colab GPU '
                    '(averages ESM-2 + ESM-1v scores for more reliable predictions)',
        category='plm',
        fn=colab_esm_consensus_scan,
        when_to_use='When user wants higher-confidence mutation predictions by combining multiple PLM models. '
                     'ESM-1v was designed for variant effect prediction (41 DMS datasets validated). '
                     'Consensus of ESM-2 + ESM-1v provides more reliable zero-shot predictions than either alone.',
        inputs=[
            SkillParam('sequence', 'string', 'Protein sequence (or pass fasta_path)', required=False),
            SkillParam('fasta_path', 'path', 'FASTA file path (alternative to sequence)', required=False),
            SkillParam('positions', 'list', 'Specific positions to scan (1-indexed, default: all)', required=False),
            SkillParam('models', 'list',
                       'List of ESM model names to ensemble. Default: [esm2_t33_650M_UR50D, esm1v_t33_650M_UR90S_1]. '
                       'For full ESM-1v ensemble: all 5 esm1v_t33_650M_UR90S_1..5',
                       required=False),
            SkillParam('top_k', 'integer', 'Return top K beneficial mutations', required=False,
                       default='20'),
        ],
        outputs=[
            SkillParam('mutations', 'list', 'Top mutations with consensus scores'),
            SkillParam('models', 'list', 'Models used in consensus'),
            SkillParam('gpu_name', 'string', 'Colab GPU type'),
        ],
        requires_gpu=False,
        source='Colab GPU server /esm_consensus_scan endpoint',
    ))

    registry.register(Skill(
        name='colab_esm_union_scan',
        description='Union-filtered ESM-2+ESM-1v dual scan on Colab GPU: '
                    'keeps mutations positive in ANY model, classifies Tier 1 (consensus) '
                    'vs Tier 2 (single positive), exports 96-well plate CSV. '
                    'Recommended for wet-lab candidate selection.',
        category='plm',
        fn=colab_esm_union_scan,
        when_to_use='When user wants to select mutations for wet-lab experiments. '
                     'This is the PREFERRED method over colab_esm_scan or colab_esm_consensus_scan '
                     'when the goal is to fill a 96-well plate with candidates. '
                     'Also use when user mentions "并集", "union", "96孔板", "96-well", "湿实验候选".',
        when_not_to_use='When user only wants to analyze single-model predictions without wet-lab intent.',
        inputs=[
            SkillParam('sequence', 'string', 'Protein sequence (or pass fasta_path)', required=False),
            SkillParam('fasta_path', 'path', 'FASTA file path (alternative to sequence)', required=False),
            SkillParam('positions', 'list', 'Specific positions to scan (1-indexed, default: all)', required=False),
            SkillParam('models', 'list',
                       'ESM models for union scan. Default: [esm2_t33_650M_UR50D, esm1v_t33_650M_UR90S_1]',
                       required=False),
            SkillParam('top_k', 'int', 'Return top K candidates (default: 96 for plate)',
                       required=False, default=96),
            SkillParam('threshold', 'float', 'LL threshold for positive mutations (default: 0.0)',
                       required=False, default=0.0),
            SkillParam('output_dir', 'path', 'Export CSV for plate ordering',
                       required=False, is_output_path=True),
        ],
        outputs=[
            SkillParam('mutations', 'list', 'Union-filtered mutations with tier classification'),
            SkillParam('tier1_consensus', 'int', 'Number of Tier 1 (all models positive) mutations'),
            SkillParam('csv_path', 'path', '96-well plate CSV file path'),
        ],
        requires_gpu=False,
        source='Colab GPU server /esm_union_scan endpoint',
    ))

    registry.register(Skill(
        name='colab_gpu_info',
        description='Check Google Colab GPU availability, type (T4/L4), VRAM, and CUDA version',
        category='plm',
        fn=colab_gpu_info,
        when_to_use='When user asks about Colab GPU status, or before running GPU-intensive tasks on Colab.',
        inputs=[],
        outputs=[
            SkillParam('gpu_available', 'boolean', 'Whether GPU is available'),
            SkillParam('gpu_name', 'string', 'GPU model name'),
            SkillParam('gpu_memory_total_gb', 'number', 'Total GPU memory in GB'),
        ],
        requires_gpu=False,
        source='Colab GPU server /gpu_info endpoint',
    ))

    registry.register(Skill(
        name='colab_auto_setup',
        description='Automatically set up Google Colab GPU server (one-click: opens ONE browser tab, '
                    'starts GPU server, retrieves tunnel URL, saves config)',
        category='plm',
        fn=colab_auto_setup,
        when_to_use='When user wants to use Colab GPU but no tunnel URL is configured yet. '
                     'Call this BEFORE colab_esm_scan or colab_esmfold if Colab is not set up. '
                     'Also call when the existing Colab connection has expired.',
        inputs=[],
        outputs=[
            SkillParam('tunnel_url', 'string', 'The cloudflared tunnel URL'),
            SkillParam('status', 'string', 'Setup status: bootstrapped/reconnected/already connected'),
        ],
        requires_gpu=False,
        source='envs/colab_bootstrap.py + envs/colab_sandbox.py',
    ))

    registry.register(Skill(
        name='colab_set_url',
        description='Manually set Colab GPU tunnel URL (use when auto_setup fails or user provides URL)',
        category='plm',
        fn=colab_set_url,
        when_to_use='When colab_auto_setup fails and user provides a tunnel URL manually. '
                     'Also when user says "set colab url" or pastes a trycloudflare.com URL.',
        inputs=[
            SkillParam('url', 'string', 'Cloudflared tunnel URL (https://xxx.trycloudflare.com)',
                       required=True),
        ],
        outputs=[
            SkillParam('tunnel_url', 'string', 'The configured tunnel URL'),
            SkillParam('gpu_name', 'string', 'GPU name if connection succeeded'),
        ],
        requires_gpu=False,
        source='envs/colab_config.py',
    ))
