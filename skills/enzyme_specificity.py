"""
Enzyme Specificity Skills
==========================
Skills for predicting enzyme-substrate compatibility using
cross-attention graph neural networks (Nature paper).
"""

import logging
from .registry import Skill, SkillParam, SkillRegistry

logger = logging.getLogger(__name__)


def _predict_specificity(
    protein_sequence: str,
    substrate_smiles: str,
    model_idx: int = 0,
    ensemble: bool = False,
    timeout: int = 600,
    config: dict = None,
    **kw,
) -> dict:
    from zhuangzhou.enzyme_specificity import predict_specificity
    return predict_specificity(
        protein_sequence=protein_sequence,
        substrate_smiles=substrate_smiles,
        model_idx=model_idx,
        ensemble=ensemble,
        timeout=timeout,
    )


def _batch_specificity(
    pairs: list,
    model_idx: int = 0,
    timeout: int = 1200,
    config: dict = None,
    **kw,
) -> dict:
    from zhuangzhou.enzyme_specificity import batch_predict
    return batch_predict(pairs=pairs, model_idx=model_idx, timeout=timeout)


def register_skills(registry: SkillRegistry):
    registry.register(Skill(
        name='enzyme_specificity',
        description='Predict enzyme-substrate catalytic compatibility (cross-attention GNN, ESM-2 + Grover + EGNN)',
        category='plm',
        fn=_predict_specificity,
        inputs=[
            SkillParam('protein_sequence', 'string', 'Enzyme amino acid sequence (max 1000 residues)', required=True),
            SkillParam('substrate_smiles', 'string', 'Substrate SMILES string', required=True),
            SkillParam('model_idx', 'int', 'Checkpoint index 0-3', required=False, default=0),
            SkillParam('ensemble', 'bool', 'Average all 4 checkpoints', required=False, default=False),
        ],
        outputs=[
            SkillParam('probability', 'float', 'Compatibility probability 0~1'),
            SkillParam('logit', 'float', 'Raw model logit'),
        ],
        requires_gpu=True,
        source='zhuangzhou/enzyme_specificity/bridge_script.py',
    ))

    registry.register(Skill(
        name='batch_enzyme_specificity',
        description='Batch predict enzyme-substrate compatibility',
        category='plm',
        fn=_batch_specificity,
        inputs=[
            SkillParam('pairs', 'list', 'List of {protein_sequence, substrate_smiles} dicts', required=True),
            SkillParam('model_idx', 'int', 'Checkpoint index 0-3', required=False, default=0),
        ],
        outputs=[
            SkillParam('results', 'list', 'List of prediction results'),
        ],
        requires_gpu=True,
        source='zhuangzhou/enzyme_specificity/bridge_script.py',
    ))
