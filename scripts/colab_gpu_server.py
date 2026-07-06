"""
Zhuangzhou Colab GPU Server (v0.9.2)
=====================================
Colab T4/L4 GPU 上的持久化蛋白质语言模型推理服务。

关键设计：
  - 双模型家族：ESM-2 (通用PLM) + ESM-1v (变体效应专用，5模型集成)
  - ESM-1v 集成：5个独立训练的650M模型，平均log-likelihood ratio
  - 智能VRAM管理：T4 15GB可同时驻留~4个650M模型，超限自动卸载最早模型
  - 显存监控：每个操作前后报告显存使用
  - Anti-idle：后台线程每5分钟做轻量GPU操作，防止Colab超时

架构：
  本地 Zhuangzhou ──HTTP──> cloudflared tunnel ──> FastAPI (Colab GPU)
                                                      ├── /health
                                                      ├── /gpu_info
                                                      ├── /esm_scan           (单模型扫描)
                                                      ├── /esm_consensus_scan (多模型共识)
                                                      ├── /execute            (任意代码)
                                                      ├── /models             (已加载模型)
                                                      ├── /unload             (释放显存)
                                                      └── /preload_esm
"""

import gc
import json
import logging
import os
import re
import socket
import subprocess
import sys
import threading
import time
from typing import Optional, List

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger('zz-colab')

app = FastAPI(title='Zhuangzhou Colab GPU Server', version='0.9.2')

# ── 全局状态 ──────────────────────────────────────────────────────────────

_model_cache = {}           # key -> {model, alphabet, device}
_cache_lock = threading.Lock()
_start_time = time.time()

# Valid ESM model names
_VALID_MODELS = {
    'esm2_t33_650M_UR50D',
    'esm1v_t33_650M_UR90S_1', 'esm1v_t33_650M_UR90S_2',
    'esm1v_t33_650M_UR90S_3', 'esm1v_t33_650M_UR90S_4',
    'esm1v_t33_650M_UR90S_5',
}


# ── 显存管理 ──────────────────────────────────────────────────────────────

def _gpu_mem_info() -> dict:
    """返回 GPU 显存信息 (GB)。"""
    import torch
    if not torch.cuda.is_available():
        return {'total': 0, 'used': 0, 'free': 0}
    total = torch.cuda.get_device_properties(0).total_memory
    used = torch.cuda.memory_allocated(0)
    reserved = torch.cuda.memory_reserved(0)
    return {
        'total': round(total / 1e9, 2),
        'used': round(used / 1e9, 2),
        'reserved': round(reserved / 1e9, 2),
        'free': round((total - reserved) / 1e9, 2),
    }


def _unload_all():
    """卸载所有模型，释放显存。"""
    import torch
    keys = list(_model_cache.keys())
    for key in keys:
        logger.info(f'Unloading model: {key}')
        del _model_cache[key]
    if keys:
        gc.collect()
        torch.cuda.empty_cache()
        logger.info(f'After unload: {_gpu_mem_info()}')


# ── 模型加载 ──────────────────────────────────────────────────────────────

def _get_esm_model(model_name: str = 'esm2_t33_650M_UR50D'):
    """加载 ESM 模型 (ESM-2 或 ESM-1v)。每个~2.5GB，T4可同时驻留~4个。"""
    import torch
    key = f'esm:{model_name}'
    with _cache_lock:
        if key not in _model_cache:
            # Check VRAM — if loading another model would exceed capacity, unload oldest
            mem = _gpu_mem_info()
            n_loaded = len(_model_cache)
            if n_loaded >= 4 or (mem['free'] < 3.5 and n_loaded > 0):
                oldest_key = next(iter(_model_cache))
                logger.info(f'VRAM tight ({mem["free"]:.1f}GB free), unloading {oldest_key}')
                del _model_cache[oldest_key]
                gc.collect()
                torch.cuda.empty_cache()

            logger.info(f'Loading {model_name} ...')

            try:
                import esm
            except ImportError:
                subprocess.run([sys.executable, '-m', 'pip', 'install', '-q',
                                'fair-esm'], capture_output=True)
                import esm

            model, alphabet = esm.pretrained.load_model_and_alphabet(model_name)
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            model = model.to(device).eval()

            _model_cache[key] = {
                'model': model, 'alphabet': alphabet, 'device': device,
            }
            logger.info(f'{model_name} loaded. VRAM: {_gpu_mem_info()}')
    return _model_cache[key]


def _run_esm_scan(model_name, sequence, positions, top_k):
    """Run single-model ESM scan, return mutation dict."""
    import torch
    cache = _get_esm_model(model_name)
    model, alphabet, device = cache['model'], cache['alphabet'], cache['device']

    _, _, wt_tokens = alphabet.get_batch_converter()([('wt', sequence)])
    wt_tokens = wt_tokens.to(device)

    with torch.no_grad():
        log_probs = torch.log_softmax(model(wt_tokens)['logits'][0], dim=-1)

    aa_list = 'ACDEFGHIKLMNPQRSTVWY'
    pos_list = positions or list(range(1, len(sequence) + 1))
    muts = {}
    for pos in pos_list:
        if pos < 1 or pos > len(sequence):
            continue
        wt_aa = sequence[pos - 1]
        wt_tid = alphabet.get_idx(wt_aa)
        for mut_aa in aa_list:
            if mut_aa == wt_aa:
                continue
            score = (log_probs[pos, alphabet.get_idx(mut_aa)]
                     - log_probs[pos, wt_tid]).item()
            key = f'{wt_aa}{pos}{mut_aa}'
            muts[key] = {
                'position': pos, 'wt': wt_aa, 'mut': mut_aa,
                'mutation': key, 'score': round(score, 4),
            }

    del wt_tokens, log_probs
    torch.cuda.empty_cache()
    return muts


# ── Request Models ────────────────────────────────────────────────────────

class ESMScanRequest(BaseModel):
    sequence: str
    positions: Optional[List[int]] = None
    model: str = 'esm2_t33_650M_UR50D'
    top_k: int = 20

class ESMConsensusRequest(BaseModel):
    sequence: str
    positions: Optional[List[int]] = None
    models: List[str] = ['esm2_t33_650M_UR50D', 'esm1v_t33_650M_UR90S_1']
    top_k: int = 20

class ESMUnionScanRequest(BaseModel):
    """Request for union-filtered dual-model scan (Zero_Shot.ipynb pattern)."""
    sequence: str
    positions: Optional[List[int]] = None
    models: List[str] = ['esm2_t33_650M_UR50D', 'esm1v_t33_650M_UR90S_1']
    top_k: int = 96
    threshold: float = 0.0
    weights: Optional[List[float]] = None

class ExecuteRequest(BaseModel):
    code: str
    timeout: int = 300


# ── API Endpoints ─────────────────────────────────────────────────────────

@app.get('/health')
def health():
    import torch
    info = {
        'status': 'ok',
        'version': '0.9.2',
        'uptime_seconds': round(time.time() - _start_time, 1),
        'models_loaded': list(_model_cache.keys()),
        'gpu_available': False,
    }
    if torch.cuda.is_available():
        info['gpu_available'] = True
        info['gpu_name'] = torch.cuda.get_device_name(0)
        info.update(_gpu_mem_info())
        info['cuda_version'] = torch.version.cuda or ''
    return info


@app.get('/gpu_info')
def gpu_info():
    import torch
    result = {'gpu_available': False}
    if torch.cuda.is_available():
        result['gpu_available'] = True
        result['gpu_name'] = torch.cuda.get_device_name(0)
        result.update(_gpu_mem_info())
        result['cuda_version'] = torch.version.cuda or ''
        result['models_loaded'] = list(_model_cache.keys())
    return result


@app.post('/esm_scan')
def esm_scan(req: ESMScanRequest):
    """单模型 ESM 零样本突变扫描 (ESM-2 或 ESM-1v)。"""
    import torch
    if not req.sequence:
        raise HTTPException(400, 'No sequence')
    if req.model not in _VALID_MODELS:
        raise HTTPException(400, f'Invalid model: {req.model}. Valid: {sorted(_VALID_MODELS)}')

    t0 = time.time()
    muts = _run_esm_scan(req.model, req.sequence, req.positions, req.top_k)
    mut_list = sorted(muts.values(), key=lambda x: x['score'], reverse=True)

    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu'
    return {
        'mutations': mut_list[:req.top_k],
        'total_mutations_scanned': len(mut_list),
        'n_positions_scanned': len(req.positions or range(1, len(req.sequence) + 1)),
        'sequence_length': len(req.sequence),
        'model': req.model, 'device': 'cuda', 'gpu_name': gpu_name,
        'gpu_memory': _gpu_mem_info(),
        'elapsed_seconds': round(time.time() - t0, 2),
    }


@app.post('/esm_consensus_scan')
def esm_consensus_scan(req: ESMConsensusRequest):
    """多模型共识突变扫描。平均多个模型的log-likelihood ratio分数。

    默认: ESM-2 + ESM-1v(模型1)。完整ESM-1v集成需传入全部5个模型。
    """
    import torch
    if not req.sequence:
        raise HTTPException(400, 'No sequence')
    for m in req.models:
        if m not in _VALID_MODELS:
            raise HTTPException(400, f'Invalid model: {m}. Valid: {sorted(_VALID_MODELS)}')
    if len(req.models) < 2:
        raise HTTPException(400, 'Need at least 2 models for consensus')

    t0 = time.time()
    all_scores = {}

    for model_name in req.models:
        logger.info(f'Consensus scan: running {model_name}...')
        muts = _run_esm_scan(model_name, req.sequence, req.positions, req.top_k)
        for key, info in muts.items():
            if key not in all_scores:
                all_scores[key] = {
                    'position': info['position'], 'wt': info['wt'],
                    'mut': info['mut'], 'mutation': key, 'scores': [],
                }
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

    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu'
    return {
        'mutations': consensus[:req.top_k],
        'total_mutations_scanned': len(consensus),
        'n_positions_scanned': len(req.positions or range(1, len(req.sequence) + 1)),
        'sequence_length': len(req.sequence),
        'models': req.models, 'n_models': len(req.models),
        'device': 'cuda', 'gpu_name': gpu_name,
        'gpu_memory': _gpu_mem_info(),
        'elapsed_seconds': round(time.time() - t0, 2),
    }


@app.post('/esm_union_scan')
def esm_union_scan(req: ESMUnionScanRequest):
    """Union-filtered dual-model scan: keep mutations positive in ANY model.

    Implements Zero_Shot.ipynb strategy:
      - Tier 1 (Consensus): ALL models positive
      - Tier 2 (Single Positive): at least one model positive
      - Sort: Tier 1 first, then by max_potential
      - Default top_k=96 for 96-well plate ordering
    """
    import torch
    if not req.sequence:
        raise HTTPException(400, 'No sequence')
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

    model_results = {}
    for i, model_name in enumerate(req.models):
        logger.info(f'Union scan: running {model_name} ({i+1}/{n_models})...')
        muts = _run_esm_scan(model_name, req.sequence, req.positions, top_k=99999)
        model_results[model_name] = muts

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
        for model_name in req.models:
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

        if info is None or n_positive == 0:
            continue

        tier = 'Tier 1: Consensus' if n_positive == n_models else 'Tier 2: Single Positive'
        valid_scores = [s for s in scores if s is not None]
        weighted_score = sum(s * w for s, w in zip(scores, weights) if s is not None)
        max_potential = max(valid_scores) if valid_scores else 0.0

        merged.append({
            'position': info['position'], 'wt': info['wt'], 'mut': info['mut'],
            'mutation': key, 'tier': tier, 'n_models_positive': n_positive,
            'per_model_scores': per_model,
            'weighted_score': round(weighted_score, 4),
            'max_potential': round(max_potential, 4),
        })

    merged.sort(key=lambda x: (0 if 'Consensus' in x['tier'] else 1, -x['max_potential']))
    n_tier1 = sum(1 for x in merged if 'Consensus' in x['tier'])

    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu'
    return {
        'mutations': merged[:req.top_k],
        'total_union_candidates': len(merged),
        'tier1_consensus': n_tier1,
        'tier2_single_positive': len(merged) - n_tier1,
        'plate_size': min(req.top_k, len(merged)),
        'sequence_length': len(req.sequence),
        'models': req.models, 'n_models': n_models, 'threshold': threshold,
        'device': 'cuda', 'gpu_name': gpu_name,
        'gpu_memory': _gpu_mem_info(),
        'elapsed_seconds': round(time.time() - t0, 2),
    }


@app.post('/execute')
def execute_code(req: ExecuteRequest):
    import io, traceback
    from contextlib import redirect_stdout, redirect_stderr
    so, se = io.StringIO(), io.StringIO()
    try:
        with redirect_stdout(so), redirect_stderr(se):
            exec(req.code, {'__builtins__': __builtins__})
        stdout = so.getvalue()
        data = stdout
        lines = stdout.strip().split('\n')
        if lines:
            try:
                data = json.loads(lines[-1])
            except (json.JSONDecodeError, ValueError):
                pass
        return {'success': True, 'data': data,
                'stdout': stdout, 'stderr': se.getvalue()}
    except Exception as e:
        return {'success': False, 'error': str(e),
                'traceback': traceback.format_exc(),
                'stdout': so.getvalue(), 'stderr': se.getvalue()}


@app.get('/models')
def list_models():
    return {
        'models': list(_model_cache.keys()),
        'count': len(_model_cache),
        'available': sorted(_VALID_MODELS),
    }


@app.post('/unload')
def unload():
    """卸载所有模型释放显存。"""
    _unload_all()
    return {'status': 'ok', 'gpu_memory': _gpu_mem_info()}


@app.post('/preload_esm')
def preload_esm(model: str = 'esm2_t33_650M_UR50D'):
    if model not in _VALID_MODELS:
        raise HTTPException(400, f'Invalid model: {model}. Valid: {sorted(_VALID_MODELS)}')
    _get_esm_model(model)
    return {'status': 'loaded', 'model': model, 'gpu_memory': _gpu_mem_info()}


# ── Anti-Idle Keepalive ───────────────────────────────────────────────────

def _keepalive_worker():
    import torch
    while True:
        time.sleep(300)
        try:
            if torch.cuda.is_available():
                x = torch.ones(1, device='cuda')
                del x
        except Exception:
            pass

threading.Thread(target=_keepalive_worker, daemon=True).start()


# ── Tunnel + Server Startup ──────────────────────────────────────────────

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


def _start_cloudflared(port: int) -> str:
    cf_path = '/usr/local/bin/cloudflared'
    if not os.path.isfile(cf_path):
        logger.info('Installing cloudflared...')
        subprocess.run([
            'wget', '-q',
            'https://github.com/cloudflare/cloudflared/releases/latest/'
            'download/cloudflared-linux-amd64',
            '-O', cf_path
        ], check=True)
        os.chmod(cf_path, 0o755)

    # Kill any leftover cloudflared
    subprocess.run(['pkill', '-f', 'cloudflared'], capture_output=True)
    time.sleep(1)

    proc = subprocess.Popen(
        [cf_path, 'tunnel', '--url', f'http://localhost:{port}'],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    tunnel_url = None
    deadline = time.time() + 60
    while time.time() < deadline:
        if proc.poll() is not None:
            err = proc.stderr.read().decode('utf-8', errors='replace')
            logger.error(f'cloudflared exited: {err[:500]}')
            break
        line = proc.stderr.readline().decode('utf-8', errors='replace')
        if not line:
            time.sleep(0.2)
            continue
        match = re.search(r'(https://[a-zA-Z0-9-]+\.trycloudflare\.com)', line)
        if match:
            tunnel_url = match.group(1)
            break

    if not tunnel_url:
        logger.error('cloudflared tunnel URL not found within 60s')
        return ''

    threading.Thread(
        target=lambda: [_ for _ in proc.stderr], daemon=True).start()
    return tunnel_url


def start_server(port: int = None):
    import uvicorn
    if port is None:
        port = _find_free_port()
    logger.info(f'Starting server on port {port}')

    tunnel_url = _start_cloudflared(port)
    if tunnel_url:
        with open('/content/tunnel_url.txt', 'w') as f:
            f.write(tunnel_url)
        print(f'\nZZ_TUNNEL_URL={tunnel_url}')
        print(f'Server ready: {tunnel_url}')
        print(f'Endpoints: /esm_scan /esm_consensus_scan /health /gpu_info /models /unload')
    else:
        print(f'[WARNING] Tunnel failed. Server on localhost:{port} only.')

    uvicorn.run(app, host='0.0.0.0', port=port, log_level='info')


# ── 入口 ──────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    start_server()
else:
    try:
        import google.colab
        start_server()
    except ImportError:
        pass
