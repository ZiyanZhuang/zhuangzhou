"""
Boltz-2 Structure Prediction Bridge Script
===========================================
stdin:  JSON {"fasta_path": "...", "model": "boltz2", "output_dir": "..."}
stdout: JSON {"pdb_path": "...", "confidence": 85.3, "model": "boltz2"}
        or   {"error": "...", "hint": "..."}

Run in py310_DPLM2 conda environment.
Install: pip install boltz
"""

import json
import os
import sys
import glob

sys.stdout.reconfigure(encoding='utf-8')


def _out(data: dict):
    print(json.dumps(data, ensure_ascii=False))
    sys.stdout.flush()


def main():
    raw = sys.stdin.read()
    try:
        payload = json.loads(raw)
    except Exception as e:
        _out({'error': f'Invalid JSON input: {e}'})
        return

    fasta_path = payload.get('fasta_path', '')
    model = payload.get('model', 'boltz2').lower()
    output_dir = payload.get('output_dir', os.getcwd())

    if not fasta_path or not os.path.exists(fasta_path):
        _out({'error': f'FASTA file not found: {fasta_path}'})
        return

    os.makedirs(output_dir, exist_ok=True)

    # ---- ESMFold fallback ----
    if model == 'esmfold':
        return _run_esmfold(fasta_path, output_dir)

    # ---- Boltz-2 ----
    try:
        import boltz
    except ImportError:
        _out({
            'error': 'boltz not installed in py310_DPLM2 environment',
            'hint': 'Run: pip install boltz  (inside py310_DPLM2 conda env)',
        })
        return

    try:
        # Boltz CLI: boltz predict <fasta> --out_dir <dir> --accelerator gpu
        import subprocess
        python_exe = sys.executable
        cmd = [
            python_exe, '-m', 'boltz', 'predict',
            fasta_path,
            '--out_dir', output_dir,
            '--accelerator', 'gpu',
            '--override',
        ]
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=600)

        if proc.returncode != 0:
            # Try CPU fallback
            cmd[-1] = 'cpu'
            cmd[-2] = '--accelerator'
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=1200)

        if proc.returncode != 0:
            _out({
                'error': f'Boltz-2 exited with code {proc.returncode}',
                'stderr': proc.stderr[-500:],
            })
            return

        # Find output PDB — boltz writes to out_dir/predictions/<stem>/<stem>.pdb
        stem = os.path.splitext(os.path.basename(fasta_path))[0]
        pdb_candidates = (
            glob.glob(os.path.join(output_dir, 'predictions', stem, '*.pdb'))
            + glob.glob(os.path.join(output_dir, '**', '*.pdb'), recursive=True)
        )
        pdb_path = pdb_candidates[0] if pdb_candidates else None

        # Compute mean pLDDT from B-factor column if PDB exists
        confidence = _mean_plddt_from_pdb(pdb_path) if pdb_path else None

        _out({
            'pdb_path': pdb_path,
            'confidence': confidence,
            'model': 'boltz2',
            'output_dir': output_dir,
        })

    except subprocess.TimeoutExpired:
        _out({'error': 'Boltz-2 timed out (>600s). Try with a shorter sequence.'})
    except Exception as e:
        _out({'error': f'Boltz-2 error: {e}'})


def _run_esmfold(fasta_path: str, output_dir: str):
    """ESMFold fallback using transformers pipeline."""
    try:
        from transformers import EsmForProteinFolding, AutoTokenizer
        import torch
    except ImportError:
        _out({
            'error': 'ESMFold requires transformers and torch in py310_DPLM2',
            'hint': 'pip install transformers torch',
        })
        return

    try:
        # Read sequence from FASTA
        with open(fasta_path, 'r', encoding='utf-8') as f:
            lines = [l.strip() for l in f if not l.startswith('>')]
        sequence = ''.join(lines)

        tokenizer = AutoTokenizer.from_pretrained('facebook/esmfold_v1')
        model = EsmForProteinFolding.from_pretrained('facebook/esmfold_v1',
                                                       low_cpu_mem_usage=True)
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model = model.to(device)

        inputs = tokenizer([sequence], return_tensors='pt', add_special_tokens=False)
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            output = model(**inputs)

        from transformers.models.esm.openfold_utils.protein import to_pdb, Protein
        pdb_str = model.infer_pdb(inputs)

        stem = os.path.splitext(os.path.basename(fasta_path))[0]
        pdb_path = os.path.join(output_dir, f'{stem}_esmfold.pdb')
        with open(pdb_path, 'w', encoding='utf-8') as f:
            f.write(pdb_str)

        confidence = _mean_plddt_from_pdb(pdb_path)
        _out({'pdb_path': pdb_path, 'confidence': confidence, 'model': 'esmfold'})

    except Exception as e:
        _out({'error': f'ESMFold error: {e}'})


def _mean_plddt_from_pdb(pdb_path: str):
    """Read mean pLDDT from B-factor column of ATOM records."""
    if not pdb_path or not os.path.exists(pdb_path):
        return None
    scores = []
    try:
        with open(pdb_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.startswith('ATOM'):
                    try:
                        bfactor = float(line[60:66])
                        scores.append(bfactor)
                    except (ValueError, IndexError):
                        pass
        return round(sum(scores) / len(scores), 2) if scores else None
    except Exception:
        return None


if __name__ == '__main__':
    main()
