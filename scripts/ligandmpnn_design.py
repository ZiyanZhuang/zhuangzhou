"""
LigandMPNN Sequence Design Bridge Script
==========================================
stdin:  JSON {
    "pdb_path": "...",
    "ligand_resname": "HEM",
    "chain": "A",
    "num_sequences": 10,
    "temperature": 0.1,
    "output_dir": "..."
}
stdout: JSON {
    "sequences": [{"sequence": "...", "score": -1.52}, ...],
    "fasta_path": "..."
}
        or   {"error": "...", "hint": "..."}

Run in py310_DPLM2 conda environment.
Install: pip install git+https://github.com/dauparas/LigandMPNN
"""

import json
import os
import sys

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

    pdb_path = payload.get('pdb_path', '')
    ligand_resname = payload.get('ligand_resname', '')
    chain = payload.get('chain', 'A')
    num_sequences = payload.get('num_sequences', 10)
    temperature = payload.get('temperature', 0.1)
    output_dir = payload.get('output_dir', os.getcwd())

    if not pdb_path or not os.path.exists(pdb_path):
        _out({'error': f'PDB file not found: {pdb_path}'})
        return
    if not ligand_resname:
        _out({'error': 'ligand_resname required (e.g. "LIG", "HEM")'})
        return

    os.makedirs(output_dir, exist_ok=True)

    # Try package-based API first
    try:
        import ligand_mpnn
        return _run_ligandmpnn_package(
            pdb_path, ligand_resname, chain, num_sequences, temperature, output_dir)
    except ImportError:
        pass

    # Fall back to subprocess calling the LigandMPNN run.py directly
    return _run_ligandmpnn_subprocess(
        pdb_path, ligand_resname, chain, num_sequences, temperature, output_dir)


def _run_ligandmpnn_package(pdb_path, ligand_resname, chain, num_sequences,
                              temperature, output_dir):
    """Use installed LigandMPNN Python package."""
    try:
        from ligand_mpnn import LigandMPNN
        import torch

        model = LigandMPNN()
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model = model.to(device)

        results = model.design(
            pdb_path=pdb_path,
            chain_id=chain,
            ligand_residue=ligand_resname,
            num_seq_per_target=num_sequences,
            sampling_temp=temperature,
        )

        sequences = []
        for seq, score in zip(results.get('sequences', []), results.get('scores', [])):
            sequences.append({'sequence': seq, 'score': round(float(score), 4)})

        stem = os.path.splitext(os.path.basename(pdb_path))[0]
        fasta_path = os.path.join(output_dir, f'{stem}_ligandmpnn_designs.fasta')
        with open(fasta_path, 'w', encoding='utf-8') as f:
            for i, s in enumerate(sequences):
                f.write(f'>design_{i+1}_score_{s["score"]:.4f}\n{s["sequence"]}\n')

        _out({'sequences': sequences, 'fasta_path': fasta_path, 'n_sequences': len(sequences)})

    except Exception as e:
        _out({'error': f'LigandMPNN package error: {e}'})


def _run_ligandmpnn_subprocess(pdb_path, ligand_resname, chain, num_sequences,
                                  temperature, output_dir):
    """
    Run LigandMPNN by calling run.py directly.
    Searches common installation paths.
    """
    import subprocess
    import glob as globlib

    search_dirs = [
        os.environ.get('LIGANDMPNN_ROOT', ''),
        os.path.expanduser(r'~\LigandMPNN'),
        os.path.expanduser(r'~/LigandMPNN'),
    ]
    run_script = None
    for d in search_dirs:
        candidate = os.path.join(d, 'run.py')
        if os.path.exists(candidate):
            run_script = candidate
            break

    if not run_script:
        _out({
            'error': 'LigandMPNN not found. Install with:',
                'hint': (
                    'Option 1: pip install git+https://github.com/dauparas/LigandMPNN\n'
                    'Option 2: set LIGANDMPNN_ROOT to a local LigandMPNN checkout'
                ),
        })
        return

    # LigandMPNN requires ligand_mpnn_model_010.pt weights
    weights_dir = os.path.join(os.path.dirname(run_script), 'model_params')

    cmd = [
        sys.executable, run_script,
        '--model_type', 'ligand_mpnn',
        '--pdb_path', pdb_path,
        '--out_folder', output_dir,
        '--chains_to_design', chain,
        '--ligand_mpnn_use_side_chain_context', '1',
        '--number_of_batches', str(num_sequences),
        '--temperature', str(temperature),
        '--seed', '42',
    ]
    if os.path.isdir(weights_dir):
        cmd += ['--model_weights_dir', weights_dir]

    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        if proc.returncode != 0:
            _out({
                'error': f'LigandMPNN exited with code {proc.returncode}',
                'stderr': proc.stderr[-500:],
            })
            return

        # Parse output FASTA
        stem = os.path.splitext(os.path.basename(pdb_path))[0]
        fasta_candidates = (
            globlib.glob(os.path.join(output_dir, 'seqs', f'{stem}.fa'))
            + globlib.glob(os.path.join(output_dir, '**', '*.fa'), recursive=True)
        )

        sequences = []
        fasta_path = None
        if fasta_candidates:
            fasta_path = fasta_candidates[0]
            sequences = _parse_mpnn_fasta(fasta_path)

        _out({
            'sequences': sequences,
            'fasta_path': fasta_path,
            'n_sequences': len(sequences),
            'ligand': ligand_resname,
        })

    except subprocess.TimeoutExpired:
        _out({'error': 'LigandMPNN timed out (>300s)'})
    except Exception as e:
        _out({'error': f'LigandMPNN subprocess error: {e}'})


def _parse_mpnn_fasta(fasta_path: str) -> list:
    """Parse MPNN output FASTA, extracting score from header."""
    import re
    sequences = []
    try:
        with open(fasta_path, 'r', encoding='utf-8') as f:
            header = None
            for line in f:
                line = line.strip()
                if line.startswith('>'):
                    header = line[1:]
                elif header and line:
                    score = None
                    m = re.search(r'score=(-?[\d.]+)', header)
                    if m:
                        score = round(float(m.group(1)), 4)
                    sequences.append({'sequence': line, 'score': score})
                    header = None
    except Exception:
        pass
    return sequences


if __name__ == '__main__':
    main()
