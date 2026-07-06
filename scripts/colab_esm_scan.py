"""
ESM-1v / ESM-2 Zero-Shot Mutation Scan — Colab Bridge
=====================================================
Designed to be sent as code to ColabSandbox.execute_code().
Reads `_payload` dict (injected by run_json_bridge), writes JSON to stdout.

_payload = {
    "sequence": "MKTL...",          # Wild-type protein sequence
    "positions": [21, 45, ...],     # Optional: specific positions to scan (1-indexed)
    "model": "esm2_t33_650M_UR50D", # or "esm1v_t33_650M_UR90S_1"
    "top_k": 20,                    # Return top K beneficial mutations
    "device": "cuda",               # "cuda" or "cpu"
}

Output JSON:
{
    "mutations": [{"position": 21, "wt": "V", "mut": "I", "score": 1.23}, ...],
    "model": "esm2_t33_650M_UR50D",
    "n_positions_scanned": 150,
    "sequence_length": 150,
    "device": "cuda",
    "gpu_name": "Tesla T4"
}
"""

import json
import sys
import os
import time

def run_esm_scan(payload):
    import torch

    sequence = payload.get('sequence', '')
    if not sequence:
        return {'error': 'No sequence provided', 'hint': 'Pass sequence in payload'}

    model_name = payload.get('model', 'esm2_t33_650M_UR50D')
    positions = payload.get('positions', None)  # None = scan all
    top_k = payload.get('top_k', 20)
    device_pref = payload.get('device', 'cuda')

    device = torch.device(device_pref if torch.cuda.is_available() else 'cpu')
    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu'

    # Load ESM model
    try:
        import esm
        model, alphabet = esm.pretrained.load_model_and_alphabet(model_name)
    except (ImportError, Exception):
        # Fallback: install fair-esm on the fly
        import subprocess
        subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'fair-esm'],
                       capture_output=True)
        import esm
        model, alphabet = esm.pretrained.load_model_and_alphabet(model_name)

    model = model.to(device)
    model.eval()
    batch_converter = alphabet.get_batch_converter()

    # Tokenize wild-type
    data = [('wt', sequence)]
    _, _, wt_tokens = batch_converter(data)
    wt_tokens = wt_tokens.to(device)

    # Get WT log-probabilities
    with torch.no_grad():
        output = model(wt_tokens)
        logits = output['logits'][0]  # (L+2, vocab)
        log_probs = torch.log_softmax(logits, dim=-1)

    # Scan positions
    aa_list = 'ACDEFGHIKLMNPQRSTVWY'
    if positions is None:
        positions = list(range(1, len(sequence) + 1))

    mutations = []
    for pos in positions:
        if pos < 1 or pos > len(sequence):
            continue
        wt_aa = sequence[pos - 1]
        token_idx = pos  # +1 for BOS token, but ESM uses 1-indexed internally
        wt_token_id = alphabet.get_idx(wt_aa)

        for mut_aa in aa_list:
            if mut_aa == wt_aa:
                continue
            mut_token_id = alphabet.get_idx(mut_aa)
            # Log-likelihood ratio: log P(mut) - log P(wt)
            score = (log_probs[token_idx, mut_token_id] -
                     log_probs[token_idx, wt_token_id]).item()
            mutations.append({
                'position': pos,
                'wt': wt_aa,
                'mut': mut_aa,
                'mutation': f'{wt_aa}{pos}{mut_aa}',
                'score': round(score, 4),
            })

    # Sort by score (higher = more beneficial), take top_k
    mutations.sort(key=lambda x: x['score'], reverse=True)
    top_mutations = mutations[:top_k]

    return {
        'mutations': top_mutations,
        'total_mutations_scanned': len(mutations),
        'n_positions_scanned': len(positions),
        'sequence_length': len(sequence),
        'model': model_name,
        'device': str(device),
        'gpu_name': gpu_name,
        'top_k': top_k,
    }


# Entry point — called when injected by ColabSandbox.run_json_bridge()
if '_payload' in dir() or '_payload' in globals():
    _result = run_esm_scan(_payload)
    print(json.dumps(_result, ensure_ascii=False))
