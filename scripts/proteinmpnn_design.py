"""
ProteinMPNN Sequence Design Bridge Script
==========================================
stdin:  JSON {
    "pdb_path": "...",
    "chain": "A",
    "fixed_positions": [10, 25, 50],
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
Install: pip install git+https://github.com/dauparas/ProteinMPNN
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
    chain = payload.get('chain', 'A')
    fixed_positions = payload.get('fixed_positions', [])
    num_sequences = payload.get('num_sequences', 10)
    temperature = payload.get('temperature', 0.1)
    output_dir = payload.get('output_dir', os.getcwd())

    if not pdb_path or not os.path.exists(pdb_path):
        _out({'error': f'PDB file not found: {pdb_path}'})
        return

    os.makedirs(output_dir, exist_ok=True)

    # Try package-based API first
    try:
        import protein_mpnn
        return _run_proteinmpnn_package(
            pdb_path, chain, fixed_positions, num_sequences, temperature, output_dir)
    except ImportError:
        pass

    # Fall back to subprocess calling the ProteinMPNN script directly
    return _run_proteinmpnn_subprocess(
        pdb_path, chain, fixed_positions, num_sequences, temperature, output_dir)


def _run_proteinmpnn_package(pdb_path, chain, fixed_positions, num_sequences,
                              temperature, output_dir):
    """Use installed ProteinMPNN Python package."""
    try:
        from protein_mpnn import ProteinMPNN
        import torch

        model = ProteinMPNN()
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model = model.to(device)

        results = model.design(
            pdb_path=pdb_path,
            chain_id=chain,
            num_seq_per_target=num_sequences,
            sampling_temp=temperature,
            fixed_positions=fixed_positions or [],
        )

        sequences = []
        for seq, score in zip(results.get('sequences', []), results.get('scores', [])):
            sequences.append({'sequence': seq, 'score': round(float(score), 4)})

        # Write FASTA output
        stem = os.path.splitext(os.path.basename(pdb_path))[0]
        fasta_path = os.path.join(output_dir, f'{stem}_mpnn_designs.fasta')
        with open(fasta_path, 'w', encoding='utf-8') as f:
            for i, s in enumerate(sequences):
                f.write(f'>design_{i+1}_score_{s["score"]:.4f}\n{s["sequence"]}\n')

        _out({'sequences': sequences, 'fasta_path': fasta_path, 'n_sequences': len(sequences)})

    except Exception as e:
        _out({'error': f'ProteinMPNN package error: {e}'})


def _run_proteinmpnn_subprocess(pdb_path, chain, fixed_positions, num_sequences,
                                  temperature, output_dir):
    """
    Run ProteinMPNN by calling protein_mpnn_run.py directly.
    Searches common installation paths.
    """
    import subprocess
    import glob as globlib
    import json as jsonlib

    # Search for ProteinMPNN installation
    search_dirs = [
        os.environ.get('PROTEINMPNN_ROOT', ''),
        os.path.expanduser(r'~\ProteinMPNN'),
        os.path.expanduser(r'~/ProteinMPNN'),
    ]
    run_script = None
    for d in search_dirs:
        candidate = os.path.join(d, 'protein_mpnn_run.py')
        if os.path.exists(candidate):
            run_script = candidate
            break

    if not run_script:
        # Try pip show to find package location
        result = subprocess.run(
            [sys.executable, '-m', 'pip', 'show', 'proteinmpnn'],
            capture_output=True, text=True)
        if result.returncode == 0:
            for line in result.stdout.splitlines():
                if line.startswith('Location:'):
                    loc = line.split(':', 1)[1].strip()
                    # Validate: must be absolute path, must exist, no traversal
                    if (os.path.isabs(loc)
                            and '..' not in loc.split(os.sep)
                            and os.path.isdir(loc)):
                        candidate = os.path.join(loc, 'protein_mpnn_run.py')
                        if os.path.exists(candidate):
                            run_script = candidate
                            break

    if not run_script:
        _out({
            'error': 'ProteinMPNN not found. Install with:',
                'hint': (
                    'Option 1: pip install git+https://github.com/dauparas/ProteinMPNN\n'
                    'Option 2: set PROTEINMPNN_ROOT to a local ProteinMPNN checkout'
                ),
        })
        return

    # Prepare JSON input for ProteinMPNN
    pdb_dir = os.path.dirname(os.path.abspath(pdb_path))
    out_folder = output_dir

    # Build fixed positions JSON if provided
    fixed_json = {}
    if fixed_positions:
        stem = os.path.splitext(os.path.basename(pdb_path))[0]
        # ProteinMPNN format: {"stem": {"chain": [pos1, pos2, ...]}}
        fixed_json[stem] = {chain: fixed_positions}

    fixed_path = None
    if fixed_json:
        fixed_path = os.path.join(output_dir, 'fixed_positions.json')
        with open(fixed_path, 'w', encoding='utf-8') as f:
            jsonlib.dump(fixed_json, f)

    cmd = [
        sys.executable, run_script,
        '--pdb_path', pdb_path,
        '--pdb_path_chains', chain,
        '--out_folder', out_folder,
        '--num_seq_per_target', str(num_sequences),
        '--sampling_temp', str(temperature),
        '--seed', '42',
        '--batch_size', '1',
    ]
    if fixed_path:
        cmd += ['--fixed_positions_jsonl', fixed_path]

    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        if proc.returncode != 0:
            _out({
                'error': f'ProteinMPNN exited with code {proc.returncode}',
                'stderr': proc.stderr[-500:],
            })
            return

        # Parse output FASTA
        stem = os.path.splitext(os.path.basename(pdb_path))[0]
        fasta_candidates = globlib.glob(os.path.join(out_folder, 'seqs', f'{stem}.fa'))
        if not fasta_candidates:
            fasta_candidates = globlib.glob(os.path.join(out_folder, '**', '*.fa'), recursive=True)

        sequences = []
        fasta_path = None
        if fasta_candidates:
            fasta_path = fasta_candidates[0]
            sequences = _parse_mpnn_fasta(fasta_path)

        _out({
            'sequences': sequences,
            'fasta_path': fasta_path,
            'n_sequences': len(sequences),
        })

    except subprocess.TimeoutExpired:
        _out({'error': 'ProteinMPNN timed out (>300s)'})
    except Exception as e:
        _out({'error': f'ProteinMPNN subprocess error: {e}'})


def _parse_mpnn_fasta(fasta_path: str) -> list:
    """Parse ProteinMPNN output FASTA, extracting score from header."""
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
