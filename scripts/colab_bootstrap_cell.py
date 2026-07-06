"""
Colab Bootstrap Cell (v0.9.2)
=============================
Injected into Colab via colab-mcp. Starts FastAPI + cloudflared in background.

Key design:
  - Two PLM families: ESM-2 (general) + ESM-1v (variant-specialized, 5-model ensemble)
  - ESM-1v ensemble: average log-likelihood ratios from 5 independently trained models
  - VRAM monitoring on every operation
  - Anti-idle keepalive
  - Prints ZZ_TUNNEL_URL=<url> for automatic parsing

No ESMFold — structure prediction handled separately (AlphaFold DB / local).
"""

# ── Install dependencies ─────────────────────────────────────────────────
import subprocess, sys, os

def _install():
    pkgs = ['fastapi', 'uvicorn', 'pydantic', 'nest_asyncio']
    for p in pkgs:
        try:
            __import__(p)
        except ImportError:
            subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', p],
                           capture_output=True)
    # fair-esm for ESM-2 and ESM-1v
    try:
        import esm
    except ImportError:
        subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'fair-esm'],
                       capture_output=True)
    cf = '/usr/local/bin/cloudflared'
    if not os.path.isfile(cf):
        subprocess.run([
            'wget', '-q',
            'https://github.com/cloudflare/cloudflared/releases/latest/'
            'download/cloudflared-linux-amd64',
            '-O', cf], check=True)
        os.chmod(cf, 0o755)

_install()

# Pre-download ESM-2 model weights (avoids 502 timeout on first API call)
# ESM-1v models download on first use (~2.5GB each, cached by torch hub)
print('Pre-downloading ESM-2 model (if not cached)...')
try:
    import esm
    esm.pretrained.load_model_and_alphabet('esm2_t33_650M_UR50D')
    print('ESM-2 (650M) ready')
except Exception as e:
    print(f'ESM-2 pre-download note: {e} (will retry on first call)')

# ── FastAPI Server ────────────────────────────────────────────────────────

import gc, json, time, threading, logging, re, io, socket
from typing import Optional, List
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger('zz-colab')

app = FastAPI(title='Zhuangzhou Colab GPU Server', version='0.9.2')
_model_cache = {}
_cache_lock = threading.Lock()
_start_time = time.time()

# Valid ESM model names
_VALID_MODELS = {
    'esm2_t33_650M_UR50D',
    'esm1v_t33_650M_UR90S_1', 'esm1v_t33_650M_UR90S_2',
    'esm1v_t33_650M_UR90S_3', 'esm1v_t33_650M_UR90S_4',
    'esm1v_t33_650M_UR90S_5',
}


# ── VRAM management ──────────────────────────────────────────────────────

def _gpu_mem():
    import torch
    if not torch.cuda.is_available():
        return {'total': 0, 'used': 0, 'free': 0}
    t = torch.cuda.get_device_properties(0).total_memory
    r = torch.cuda.memory_reserved(0)
    return {'total': round(t/1e9, 2), 'used': round(torch.cuda.memory_allocated(0)/1e9, 2),
            'reserved': round(r/1e9, 2), 'free': round((t-r)/1e9, 2)}


def _unload_all():
    """Unload all cached models."""
    keys = list(_model_cache.keys())
    for k in keys:
        logger.info(f'Unloading {k}')
        del _model_cache[k]
    if keys:
        import torch
        gc.collect()
        torch.cuda.empty_cache()
        logger.info(f'After unload: {_gpu_mem()}')


# ── Model loading ────────────────────────────────────────────────────────

def _get_esm(name='esm2_t33_650M_UR50D'):
    """Load an ESM model (ESM-2 or ESM-1v). Models share VRAM, all are ~2.5GB."""
    import torch
    key = f'esm:{name}'
    with _cache_lock:
        if key not in _model_cache:
            # Check VRAM — if loading another model would exceed capacity, unload first
            mem = _gpu_mem()
            n_loaded = len(_model_cache)
            # T4 has ~15GB; each model ~2.5GB; keep headroom for inference
            if n_loaded >= 4 or (mem['free'] < 3.5 and n_loaded > 0):
                logger.info(f'VRAM tight ({mem["free"]:.1f}GB free, {n_loaded} models loaded), unloading oldest')
                oldest_key = next(iter(_model_cache))
                logger.info(f'Unloading {oldest_key}')
                del _model_cache[oldest_key]
                gc.collect()
                torch.cuda.empty_cache()

            logger.info(f'Loading {name} ...')
            import esm
            model, alphabet = esm.pretrained.load_model_and_alphabet(name)
            dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            model = model.to(dev).eval()
            _model_cache[key] = {'model': model, 'alphabet': alphabet, 'device': dev}
            logger.info(f'{name} loaded. VRAM: {_gpu_mem()}')
    return _model_cache[key]


def _run_esm_scan(model_name, sequence, positions, top_k):
    """Run single-model ESM scan, return mutation list."""
    import torch
    c = _get_esm(model_name)
    model, alphabet, dev = c['model'], c['alphabet'], c['device']
    _, _, toks = alphabet.get_batch_converter()([('wt', sequence)])
    toks = toks.to(dev)
    with torch.no_grad():
        lp = torch.log_softmax(model(toks)['logits'][0], dim=-1)
    aa = 'ACDEFGHIKLMNPQRSTVWY'
    pos_list = positions or list(range(1, len(sequence)+1))
    muts = {}
    for p in pos_list:
        if p < 1 or p > len(sequence): continue
        wt = sequence[p-1]; wid = alphabet.get_idx(wt)
        for m in aa:
            if m == wt: continue
            s = (lp[p, alphabet.get_idx(m)] - lp[p, wid]).item()
            key = f'{wt}{p}{m}'
            muts[key] = {'position': p, 'wt': wt, 'mut': m,
                         'mutation': key, 'score': round(s, 4)}
    del toks, lp; torch.cuda.empty_cache()
    return muts


# ── Request models ───────────────────────────────────────────────────────

class ESMScanReq(BaseModel):
    sequence: str
    positions: Optional[List[int]] = None
    model: str = 'esm2_t33_650M_UR50D'
    top_k: int = 20

class ESMConsensusReq(BaseModel):
    sequence: str
    positions: Optional[List[int]] = None
    models: List[str] = ['esm2_t33_650M_UR50D', 'esm1v_t33_650M_UR90S_1']
    top_k: int = 20

class ESMUnionScanReq(BaseModel):
    """Request for union-filtered dual-model scan (notebook Zero_Shot pattern)."""
    sequence: str
    positions: Optional[List[int]] = None
    models: List[str] = ['esm2_t33_650M_UR50D', 'esm1v_t33_650M_UR90S_1']
    top_k: int = 96           # Default: fill a 96-well plate
    threshold: float = 0.0    # LL threshold for positive mutations
    weights: Optional[List[float]] = None  # Per-model weights for ensemble score

class ExecReq(BaseModel):
    code: str
    timeout: int = 300


# ── Endpoints ────────────────────────────────────────────────────────────

@app.get('/health')
def health():
    import torch
    info = {'status': 'ok', 'uptime_seconds': round(time.time() - _start_time, 1),
            'models_loaded': list(_model_cache.keys()), 'gpu_available': False,
            'version': '0.9.2'}
    if torch.cuda.is_available():
        info['gpu_available'] = True
        info['gpu_name'] = torch.cuda.get_device_name(0)
        info.update(_gpu_mem())
        info['cuda_version'] = torch.version.cuda or ''
    return info

@app.get('/gpu_info')
def gpu_info():
    import torch
    r = {'gpu_available': False}
    if torch.cuda.is_available():
        r['gpu_available'] = True
        r['gpu_name'] = torch.cuda.get_device_name(0)
        r.update(_gpu_mem())
        r['cuda_version'] = torch.version.cuda or ''
        r['models_loaded'] = list(_model_cache.keys())
    return r

@app.post('/esm_scan')
def esm_scan(req: ESMScanReq):
    """Single-model ESM mutation scan (ESM-2 or ESM-1v)."""
    import torch
    if not req.sequence: raise HTTPException(400, 'No sequence')
    if req.model not in _VALID_MODELS:
        raise HTTPException(400, f'Invalid model: {req.model}. Valid: {sorted(_VALID_MODELS)}')
    t0 = time.time()
    muts = _run_esm_scan(req.model, req.sequence, req.positions, req.top_k)
    mut_list = sorted(muts.values(), key=lambda x: x['score'], reverse=True)
    gn = torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu'
    return {'mutations': mut_list[:req.top_k], 'total_mutations_scanned': len(mut_list),
            'n_positions_scanned': len(req.positions or range(1, len(req.sequence)+1)),
            'sequence_length': len(req.sequence),
            'model': req.model, 'device': 'cuda', 'gpu_name': gn,
            'gpu_memory': _gpu_mem(), 'elapsed_seconds': round(time.time()-t0, 2)}

@app.post('/esm_consensus_scan')
def esm_consensus_scan(req: ESMConsensusReq):
    """Multi-model consensus mutation scan.

    Averages log-likelihood ratio scores across specified models.
    Default: ESM-2 + ESM-1v (model 1). For full ESM-1v ensemble, pass all 5 models.

    Consensus score = mean(score_model_1, score_model_2, ...).
    """
    import torch
    if not req.sequence: raise HTTPException(400, 'No sequence')
    for m in req.models:
        if m not in _VALID_MODELS:
            raise HTTPException(400, f'Invalid model: {m}. Valid: {sorted(_VALID_MODELS)}')
    if len(req.models) < 2:
        raise HTTPException(400, 'Need at least 2 models for consensus')

    t0 = time.time()
    all_scores = {}  # mutation_key -> list of scores

    for model_name in req.models:
        logger.info(f'Consensus scan: running {model_name}...')
        muts = _run_esm_scan(model_name, req.sequence, req.positions, req.top_k)
        for key, info in muts.items():
            if key not in all_scores:
                all_scores[key] = {'position': info['position'], 'wt': info['wt'],
                                   'mut': info['mut'], 'mutation': key, 'scores': []}
            all_scores[key]['scores'].append(info['score'])

    # Compute consensus (average)
    consensus = []
    for key, info in all_scores.items():
        scores = info['scores']
        avg = sum(scores) / len(scores)
        consensus.append({
            'position': info['position'], 'wt': info['wt'], 'mut': info['mut'],
            'mutation': key, 'consensus_score': round(avg, 4),
            'per_model_scores': scores, 'n_models': len(scores),
        })
    consensus.sort(key=lambda x: x['consensus_score'], reverse=True)

    gn = torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu'
    return {'mutations': consensus[:req.top_k],
            'total_mutations_scanned': len(consensus),
            'n_positions_scanned': len(req.positions or range(1, len(req.sequence)+1)),
            'sequence_length': len(req.sequence),
            'models': req.models, 'n_models': len(req.models),
            'device': 'cuda', 'gpu_name': gn,
            'gpu_memory': _gpu_mem(), 'elapsed_seconds': round(time.time()-t0, 2)}

@app.post('/esm_union_scan')
def esm_union_scan(req: ESMUnionScanReq):
    """Union-filtered dual-model scan: keep mutations positive in ANY model.

    Implements the notebook Zero_Shot.ipynb strategy:
      1. Run each model independently
      2. Keep mutations where ANY model score > threshold (union)
      3. Classify into tiers:
         - Tier 1 (Consensus): ALL models score > threshold
         - Tier 2 (Single Positive): at least one model > threshold
      4. Sort: Tier 1 first, then by max_potential descending
      5. Return top_k for 96-well plate ordering

    This is the recommended endpoint for wet-lab candidate selection.
    """
    import torch
    if not req.sequence: raise HTTPException(400, 'No sequence')
    for m in req.models:
        if m not in _VALID_MODELS:
            raise HTTPException(400, f'Invalid model: {m}. Valid: {sorted(_VALID_MODELS)}')
    if len(req.models) < 2:
        raise HTTPException(400, 'Need at least 2 models for union scan')

    t0 = time.time()
    n_models = len(req.models)
    weights = req.weights or [1.0 / n_models] * n_models
    if len(weights) != n_models:
        weights = [1.0 / n_models] * n_models

    # Run each model
    model_results = {}
    for i, model_name in enumerate(req.models):
        logger.info(f'Union scan: running {model_name} ({i+1}/{n_models})...')
        muts = _run_esm_scan(model_name, req.sequence, req.positions, top_k=99999)
        model_results[model_name] = muts

    # Merge: for each mutation, collect per-model scores
    all_keys = set()
    for muts in model_results.values():
        all_keys.update(muts.keys())

    merged = []
    threshold = req.threshold
    for key in all_keys:
        per_model = {}
        info = None
        n_positive = 0
        scores = []
        for j, model_name in enumerate(req.models):
            m_data = model_results[model_name].get(key)
            if m_data:
                s = m_data['score']
                per_model[model_name] = s
                scores.append(s)
                if info is None:
                    info = m_data
                if s > threshold:
                    n_positive += 1
            else:
                per_model[model_name] = None
                scores.append(None)

        if info is None:
            continue

        # Union filter: keep if ANY model is positive
        if n_positive == 0:
            continue

        # Tier classification
        if n_positive == n_models:
            tier = 'Tier 1: Consensus'
        else:
            tier = 'Tier 2: Single Positive'

        # Scores
        valid_scores = [s for s in scores if s is not None]
        weighted_score = sum(s * w for s, w in zip(scores, weights) if s is not None)
        max_potential = max(valid_scores) if valid_scores else 0.0

        merged.append({
            'position': info['position'],
            'wt': info['wt'],
            'mut': info['mut'],
            'mutation': key,
            'tier': tier,
            'n_models_positive': n_positive,
            'per_model_scores': per_model,
            'weighted_score': round(weighted_score, 4),
            'max_potential': round(max_potential, 4),
        })

    # Sort: Tier 1 first, then by max_potential descending
    merged.sort(key=lambda x: (0 if 'Consensus' in x['tier'] else 1, -x['max_potential']))

    # Stats
    n_tier1 = sum(1 for x in merged if 'Consensus' in x['tier'])
    n_tier2 = len(merged) - n_tier1

    gn = torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu'
    return {
        'mutations': merged[:req.top_k],
        'total_union_candidates': len(merged),
        'tier1_consensus': n_tier1,
        'tier2_single_positive': n_tier2,
        'plate_size': min(req.top_k, len(merged)),
        'sequence_length': len(req.sequence),
        'models': req.models,
        'n_models': n_models,
        'threshold': threshold,
        'device': 'cuda',
        'gpu_name': gn,
        'gpu_memory': _gpu_mem(),
        'elapsed_seconds': round(time.time() - t0, 2),
    }


@app.post('/execute')
def execute_code(req: ExecReq):
    import traceback
    from contextlib import redirect_stdout, redirect_stderr
    so, se = io.StringIO(), io.StringIO()
    try:
        with redirect_stdout(so), redirect_stderr(se):
            exec(req.code, {'__builtins__': __builtins__})
        stdout = so.getvalue(); data = stdout
        lines = stdout.strip().split('\n')
        if lines:
            try: data = json.loads(lines[-1])
            except: pass
        return {'success': True, 'data': data, 'stdout': stdout, 'stderr': se.getvalue()}
    except Exception as e:
        return {'success': False, 'error': str(e), 'traceback': traceback.format_exc(),
                'stdout': so.getvalue(), 'stderr': se.getvalue()}

@app.get('/models')
def list_models():
    return {'models': list(_model_cache.keys()), 'count': len(_model_cache),
            'available': sorted(_VALID_MODELS)}

@app.post('/unload')
def unload():
    import torch
    _unload_all()
    return {'status': 'ok', 'gpu_memory': _gpu_mem()}

@app.post('/preload_esm')
def preload_esm(model: str = 'esm2_t33_650M_UR50D'):
    if model not in _VALID_MODELS:
        raise HTTPException(400, f'Invalid model: {model}. Valid: {sorted(_VALID_MODELS)}')
    _get_esm(model)
    return {'status': 'loaded', 'model': model, 'gpu_memory': _gpu_mem()}


# ── Anti-idle keepalive ──────────────────────────────────────────────────
def _keepalive():
    import torch
    while True:
        time.sleep(300)
        try:
            if torch.cuda.is_available():
                x = torch.ones(1, device='cuda'); del x
        except: pass

threading.Thread(target=_keepalive, daemon=True).start()


# ── Find free port + start cloudflared + start uvicorn ───────────────────

def _find_free_port(start=8080, attempts=20):
    for offset in range(attempts):
        p = start + offset
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(('0.0.0.0', p))
                return p
        except OSError:
            continue
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('0.0.0.0', 0))
        return s.getsockname()[1]

PORT = _find_free_port()
print(f'Using port {PORT}')

# Kill any leftover cloudflared from previous cell runs
subprocess.run(['pkill', '-f', 'cloudflared'], capture_output=True)
time.sleep(1)

# nest_asyncio needed for uvicorn inside Colab's event loop
try:
    import nest_asyncio
    nest_asyncio.apply()
except ImportError:
    subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'nest_asyncio'],
                   capture_output=True)
    import nest_asyncio
    nest_asyncio.apply()

# Start uvicorn FIRST so it's ready when tunnel connects
def _run_server():
    import uvicorn
    try:
        uvicorn.run(app, host='0.0.0.0', port=PORT, log_level='warning')
    except Exception as e:
        print(f'SERVER ERROR: {e}')

_server_thread = threading.Thread(target=_run_server, daemon=True)
_server_thread.start()
time.sleep(2)  # Let uvicorn bind

# Start cloudflared (no invalid flags — use only documented options)
cf_proc = subprocess.Popen(
    ['/usr/local/bin/cloudflared', 'tunnel', '--url', f'http://localhost:{PORT}'],
    stdout=subprocess.PIPE, stderr=subprocess.PIPE)

_tunnel_url = None
_deadline = time.time() + 60  # 60s timeout
while time.time() < _deadline:
    if cf_proc.poll() is not None:
        err = cf_proc.stderr.read().decode('utf-8', errors='replace')
        print(f'cloudflared exited with code {cf_proc.returncode}: {err[:500]}')
        break
    line = cf_proc.stderr.readline().decode('utf-8', errors='replace')
    if not line:
        time.sleep(0.2)
        continue
    match = re.search(r'(https://[a-zA-Z0-9-]+\.trycloudflare\.com)', line)
    if match:
        _tunnel_url = match.group(1)
        break

# Drain stderr in background
threading.Thread(target=lambda: [_ for _ in cf_proc.stderr], daemon=True).start()

# Print tagged URL for automatic parsing
if _tunnel_url:
    with open('/content/tunnel_url.txt', 'w') as f:
        f.write(_tunnel_url)
    print(f'ZZ_TUNNEL_URL={_tunnel_url}')
    print(f'Server running on {_tunnel_url}')
    print(f'Endpoints: /esm_scan /esm_consensus_scan /health /gpu_info /models /unload')
    print('Keep this notebook alive!')
else:
    print('ZZ_TUNNEL_URL=FAILED')
    print('cloudflared tunnel failed to start')
