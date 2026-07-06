"""
Enzyme Specificity Prediction — DEPRECATED
=============================================
Cross-attention GNN model for predicting enzyme-substrate compatibility.
Based on: "Enzyme specificity prediction using cross-attention graph neural networks" (Nature)

Architecture: ESM-2 (protein) + Grover (substrate) + EGNN (3D structure) + Cross-Attention

STATUS: DISABLED — The saved checkpoints (complete-full-random-all-*-complex) were trained
exclusively with protein 3D pocket structures (full_data=False, mode=complex). The EGNN
expects protein+ligand atom graph with KNN edges (k=48). Without real protein 3D structure,
predictions are meaningless (benchmark: 1/14 TP, all logits << 0).
To revive: retrain with ligand-only mode, or provide AlphaFold pocket structures at inference.
"""

import json
import logging
import os
import shutil
import subprocess
import sys
from typing import Any, Dict, List, Optional, Union

logger = logging.getLogger(__name__)

# Paths
_MODULE_DIR = os.path.dirname(os.path.abspath(__file__))
_BRIDGE_SCRIPT = os.path.join(_MODULE_DIR, 'bridge_script.py')
_EZS_PYTHON = os.environ.get('ZHUANGZHOU_EZS_PYTHON', 'python')

# Source repo
REPO_ROOT = os.environ.get('ZHUANGZHOU_ENZYME_SPECIFICITY_ROOT', '')
GROVER_MODEL = os.path.join(REPO_ROOT, 'data', 'pretrained_model', 'grover_large.pt')
SS_CHECKPOINTS = [
    os.path.join(REPO_ROOT, 'saved_model', f'complete-full-random-all-{i}-complex',
                 'run_0', 'models', 'best-checkpoint.ckpt')
    for i in range(4)
]


def predict_specificity(
    protein_sequence: str,
    substrate_smiles: str,
    model_idx: int = 0,
    ensemble: bool = False,
    timeout: int = 600,
) -> Dict[str, Any]:
    """
    Predict enzyme-substrate compatibility.

    Args:
        protein_sequence: Amino acid sequence (max 1000 residues)
        substrate_smiles: Substrate SMILES string (max 275 atoms)
        model_idx: Which checkpoint to use (0-3), ignored if ensemble=True
        ensemble: Average predictions from all 4 checkpoints
        timeout: Max seconds to wait

    Returns:
        dict with keys:
            probability: float (0~1), higher = more likely to catalyze
            logit: float, raw model output
            protein_length: int
            substrate_atoms: int
            model_used: str
        or on error:
            error: str, hint: str
    """
    if not shutil.which(_EZS_PYTHON) and not os.path.exists(_EZS_PYTHON):
        return {'error': 'ezs_env not found', 'hint': f'Expected at {_EZS_PYTHON}'}
    if not REPO_ROOT or not os.path.isdir(REPO_ROOT):
        return {
            'error': 'enzyme specificity repo not configured',
            'hint': 'Set ZHUANGZHOU_ENZYME_SPECIFICITY_ROOT to the model repository path',
        }

    if not protein_sequence or not substrate_smiles:
        return {'error': 'Empty input', 'hint': 'Provide both protein_sequence and substrate_smiles'}

    if len(protein_sequence) > 1000:
        return {'error': f'Protein sequence too long ({len(protein_sequence)} > 1000)',
                'hint': 'ESM-2 model supports up to 1000 residues'}

    payload = {
        'action': 'predict',
        'protein_sequence': protein_sequence,
        'substrate_smiles': substrate_smiles,
        'model_idx': model_idx,
        'ensemble': ensemble,
        'repo_root': REPO_ROOT,
    }

    try:
        result = subprocess.run(
            [_EZS_PYTHON, _BRIDGE_SCRIPT],
            input=json.dumps(payload).encode('utf-8'),
            capture_output=True,
            timeout=timeout,
        )
        stdout = result.stdout.decode('utf-8', errors='replace')
        stderr = result.stderr.decode('utf-8', errors='replace')

        if result.returncode == 0:
            try:
                return json.loads(stdout)
            except json.JSONDecodeError:
                return {'error': 'Invalid JSON output', 'hint': stdout[:500]}
        else:
            return {'error': f'Bridge exited with code {result.returncode}',
                    'hint': stderr[:500]}
    except subprocess.TimeoutExpired:
        return {'error': f'Timeout after {timeout}s',
                'hint': 'Try a shorter protein sequence or increase timeout'}
    except Exception as e:
        return {'error': str(e), 'hint': 'Check ezs_env installation'}


def batch_predict(
    pairs: List[Dict[str, str]],
    model_idx: int = 0,
    timeout: int = 1200,
) -> Dict[str, Any]:
    """
    Batch predict enzyme-substrate compatibility.

    Args:
        pairs: List of dicts with 'protein_sequence' and 'substrate_smiles'
        model_idx: Which checkpoint to use (0-3)
        timeout: Max seconds to wait

    Returns:
        dict with 'results': list of prediction dicts
    """
    if not shutil.which(_EZS_PYTHON) and not os.path.exists(_EZS_PYTHON):
        return {'error': 'ezs_env not found'}
    if not REPO_ROOT or not os.path.isdir(REPO_ROOT):
        return {
            'error': 'enzyme specificity repo not configured',
            'hint': 'Set ZHUANGZHOU_ENZYME_SPECIFICITY_ROOT to the model repository path',
        }

    payload = {
        'action': 'batch_predict',
        'pairs': pairs,
        'model_idx': model_idx,
        'repo_root': REPO_ROOT,
    }

    try:
        result = subprocess.run(
            [_EZS_PYTHON, _BRIDGE_SCRIPT],
            input=json.dumps(payload).encode('utf-8'),
            capture_output=True,
            timeout=timeout,
        )
        stdout = result.stdout.decode('utf-8', errors='replace')
        stderr = result.stderr.decode('utf-8', errors='replace')

        if result.returncode == 0:
            try:
                return json.loads(stdout)
            except json.JSONDecodeError:
                return {'error': 'Invalid JSON', 'hint': stdout[:500]}
        else:
            return {'error': f'Bridge error (code {result.returncode})',
                    'hint': stderr[:500]}
    except subprocess.TimeoutExpired:
        return {'error': f'Timeout after {timeout}s'}
    except Exception as e:
        return {'error': str(e)}
