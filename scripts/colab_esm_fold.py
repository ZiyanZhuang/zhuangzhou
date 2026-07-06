"""
ESMFold Structure Prediction — Colab Bridge
============================================
Designed to be sent as code to ColabSandbox.execute_code().
Reads `_payload` dict (injected by run_json_bridge), writes JSON to stdout.

_payload = {
    "sequence": "MKTL...",           # Protein sequence
    "output_name": "my_protein",     # Stem name for output PDB
    "device": "cuda",                # "cuda" or "cpu"
    "chunk_size": 64,                # Attention chunk size (lower = less VRAM)
}

Output JSON:
{
    "pdb_content": "ATOM  1  N  ...",  # Full PDB file content
    "mean_plddt": 85.3,
    "sequence_length": 150,
    "model": "esmfold_v1",
    "device": "cuda",
    "gpu_name": "Tesla T4",
    "elapsed_seconds": 12.5
}
"""

import json
import sys
import time


def run_esmfold(payload):
    import torch

    sequence = payload.get('sequence', '')
    if not sequence:
        return {'error': 'No sequence provided'}

    # ESMFold max practical length on T4 (16GB) ≈ 400 residues
    if len(sequence) > 800:
        return {
            'error': f'Sequence too long ({len(sequence)} aa). ESMFold on Colab T4 supports ≤800 aa.',
            'hint': 'Use AlphaFold2/Boltz-2 for longer sequences, or split into domains.',
        }

    device_pref = payload.get('device', 'cuda')
    chunk_size = payload.get('chunk_size', 64)
    output_name = payload.get('output_name', 'protein')

    device = torch.device(device_pref if torch.cuda.is_available() else 'cpu')
    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu'

    # Install if needed
    try:
        from transformers import EsmForProteinFolding, AutoTokenizer
    except ImportError:
        import subprocess
        subprocess.run([sys.executable, '-m', 'pip', 'install', '-q',
                        'transformers', 'accelerate'], capture_output=True)
        from transformers import EsmForProteinFolding, AutoTokenizer

    t0 = time.time()

    tokenizer = AutoTokenizer.from_pretrained('facebook/esmfold_v1')
    model = EsmForProteinFolding.from_pretrained(
        'facebook/esmfold_v1', low_cpu_mem_usage=True)
    model = model.to(device)
    model.eval()

    # Set chunk size for memory efficiency
    model.trunk.set_chunk_size(chunk_size)

    inputs = tokenizer([sequence], return_tensors='pt', add_special_tokens=False)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        output = model(**inputs)

    # Convert to PDB string
    from transformers.models.esm.openfold_utils.protein import to_pdb, Protein
    from transformers.models.esm.openfold_utils import feats as of_feats

    # Build Protein object from model output
    pdb_str = _output_to_pdb(output, sequence)

    # Compute mean pLDDT
    plddt = output['plddt'][0, :len(sequence), 1].cpu().numpy()
    mean_plddt = round(float(plddt.mean()), 2)

    elapsed = round(time.time() - t0, 2)

    # Also save to /content/ on Colab
    pdb_path = f'/content/{output_name}_esmfold.pdb'
    with open(pdb_path, 'w') as f:
        f.write(pdb_str)

    return {
        'pdb_content': pdb_str,
        'pdb_path': pdb_path,
        'mean_plddt': mean_plddt,
        'sequence_length': len(sequence),
        'model': 'esmfold_v1',
        'device': str(device),
        'gpu_name': gpu_name,
        'elapsed_seconds': elapsed,
    }


def _output_to_pdb(output, sequence):
    """Convert ESMFold output to PDB string."""
    try:
        # Try the standard transformers utility
        from transformers.models.esm.openfold_utils.protein import to_pdb, Protein
        import numpy as np

        # Extract atom positions and mask
        positions = output['positions'][-1][0].cpu().numpy()  # (L, 37, 3)
        atom_mask = output['atom37_atom_exists'][0].cpu().numpy()  # (L, 37)
        aatype = output['aatype'][0].cpu().numpy()
        residue_index = np.arange(len(sequence))
        b_factors = output['plddt'][0, :len(sequence), 1].cpu().numpy()
        b_factors = np.repeat(b_factors[:, None], 37, axis=1)

        protein = Protein(
            aatype=aatype[:len(sequence)],
            atom_positions=positions[:len(sequence)],
            atom_mask=atom_mask[:len(sequence)],
            residue_index=residue_index,
            b_factors=b_factors,
            chain_index=np.zeros(len(sequence), dtype=np.int32),
        )
        return to_pdb(protein)
    except Exception:
        # Fallback: use model's built-in method if available
        try:
            return output.get('pdb', 'ERROR: Could not convert to PDB')
        except Exception as e:
            return f'REMARK  ESMFold output conversion failed: {e}'


# Entry point
if '_payload' in dir() or '_payload' in globals():
    _result = run_esmfold(_payload)
    print(json.dumps(_result, ensure_ascii=False))
