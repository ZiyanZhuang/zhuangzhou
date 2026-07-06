#!/usr/bin/env python
"""
Enzyme Specificity Bridge Script
==================================
Runs in ezs_env. Reads JSON from stdin, writes JSON to stdout.

Constructs model inputs directly in memory (no LMDB needed):
  1. ESM-2 per-residue embeddings (1280D)
  2. Grover atom fingerprints (2400D) + total fingerprint (4885D)
  3. Morgan fingerprint (1024 bits)
  4. 3D molecular graph from SMILES (RDKit)
  5. Cross-attention GNN inference
"""

import json
import os
import sys
import warnings
import traceback

# Suppress noisy logs
warnings.filterwarnings("ignore")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np
import torch

# ── Global model cache ──────────────────────────────────────────────────────
_esm_model = None
_esm_alphabet = None
_grover_model = None
_grover_args = None
_ss_models = {}


def _get_device():
    return torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


# ── ESM-2 Embeddings ────────────────────────────────────────────────────────

def _load_esm():
    global _esm_model, _esm_alphabet
    if _esm_model is None:
        import esm
        _esm_model, _esm_alphabet = esm.pretrained.esm2_t33_650M_UR50D()
        _esm_model = _esm_model.to(_get_device())
        _esm_model.eval()
    return _esm_model, _esm_alphabet


def generate_esm_embedding(sequence: str) -> np.ndarray:
    """Generate ESM-2 per-residue embedding. Returns [L, 1280]."""
    model, alphabet = _load_esm()
    batch_converter = alphabet.get_batch_converter()
    data = [("protein", sequence)]
    _, _, batch_tokens = batch_converter(data)
    batch_tokens = batch_tokens.to(_get_device())

    with torch.no_grad():
        results = model(batch_tokens, repr_layers=[33], return_contacts=False)
    token_repr = results["representations"][33].cpu()
    # token 0 is BOS, so residue i is at position i+1
    embedding = token_repr[0, 1:1+len(sequence)].numpy()
    return embedding  # [L, 1280]


# ── Grover Fingerprints ─────────────────────────────────────────────────────

def _load_grover(repo_root):
    global _grover_model, _grover_args
    if _grover_model is not None:
        return _grover_model, _grover_args

    grover_dir = os.path.join(repo_root, 'src', 'other_softwares', 'grover_software')
    if grover_dir not in sys.path:
        sys.path.insert(0, grover_dir)

    from grover.util.utils import load_checkpoint
    from argparse import Namespace

    checkpoint_path = os.path.join(repo_root, 'data', 'pretrained_model', 'grover_large.pt')
    args = Namespace(
        cuda=torch.cuda.is_available(),
        gpu=0,
        no_cuda=not torch.cuda.is_available(),
        fingerprint_source='both',
        features_path=None,
        no_cache=True,
        bond_drop_rate=0,
        parser_name='fingerprint',
        fingerprint=True,
    )
    _grover_model = load_checkpoint(checkpoint_path, cuda=args.cuda, current_args=args, logger=None)
    _grover_model.eval()
    _grover_args = args
    return _grover_model, _grover_args


def _generate_fgtasklabel(smiles: str) -> np.ndarray:
    """Generate 85 functional group binary features matching training pipeline."""
    from rdkit import Chem
    from descriptastorus.descriptors import rdDescriptors

    RDKIT_PROPS = ['fr_Al_COO', 'fr_Al_OH', 'fr_Al_OH_noTert', 'fr_ArN',
                   'fr_Ar_COO', 'fr_Ar_N', 'fr_Ar_NH', 'fr_Ar_OH', 'fr_COO', 'fr_COO2',
                   'fr_C_O', 'fr_C_O_noCOO', 'fr_C_S', 'fr_HOCCN', 'fr_Imine', 'fr_NH0',
                   'fr_NH1', 'fr_NH2', 'fr_N_O', 'fr_Ndealkylation1', 'fr_Ndealkylation2',
                   'fr_Nhpyrrole', 'fr_SH', 'fr_aldehyde', 'fr_alkyl_carbamate', 'fr_alkyl_halide',
                   'fr_allylic_oxid', 'fr_amide', 'fr_amidine', 'fr_aniline', 'fr_aryl_methyl',
                   'fr_azide', 'fr_azo', 'fr_barbitur', 'fr_benzene', 'fr_benzodiazepine',
                   'fr_bicyclic', 'fr_diazo', 'fr_dihydropyridine', 'fr_epoxide', 'fr_ester',
                   'fr_ether', 'fr_furan', 'fr_guanido', 'fr_halogen', 'fr_hdrzine', 'fr_hdrzone',
                   'fr_imidazole', 'fr_imide', 'fr_isocyan', 'fr_isothiocyan', 'fr_ketone',
                   'fr_ketone_Topliss', 'fr_lactam', 'fr_lactone', 'fr_methoxy', 'fr_morpholine',
                   'fr_nitrile', 'fr_nitro', 'fr_nitro_arom', 'fr_nitro_arom_nonortho',
                   'fr_nitroso', 'fr_oxazole', 'fr_oxime', 'fr_para_hydroxylation', 'fr_phenol',
                   'fr_phenol_noOrthoHbond', 'fr_phos_acid', 'fr_phos_ester', 'fr_piperdine',
                   'fr_piperzine', 'fr_priamide', 'fr_prisulfonamd', 'fr_pyridine', 'fr_quatN',
                   'fr_sulfide', 'fr_sulfonamd', 'fr_sulfone', 'fr_term_acetylene', 'fr_tetrazole',
                   'fr_thiazole', 'fr_thiocyan', 'fr_thiophene', 'fr_unbrch_alkane', 'fr_urea']

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return np.zeros(85, dtype=np.float32)
    canon = Chem.MolToSmiles(mol, isomericSmiles=True)
    generator = rdDescriptors.RDKit2D(RDKIT_PROPS)
    features = generator.process(canon)[1:]
    features = np.array(features, dtype=np.float32)
    features[features != 0] = 1
    return features


def generate_grover_fingerprint(smiles: str, repo_root: str):
    """Generate Grover fingerprints. Returns (atom_embedding[N,2400], total_embedding[4885])."""
    import tempfile
    import pandas as pd

    grover_dir = os.path.join(repo_root, 'src', 'other_softwares', 'grover_software')
    if grover_dir not in sys.path:
        sys.path.insert(0, grover_dir)

    from grover.data import MolCollator, MoleculeDataset
    from grover.util.utils import get_data

    model, args = _load_grover(repo_root)

    # Write temp CSV for Grover input
    tmp = tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False, encoding='utf-8')
    try:
        tmp.write('smiles\n')
        tmp.write(f'{smiles}\n')
        tmp.close()

        test_data = get_data(path=tmp.name, args=args, use_compound_names=False,
                             max_data_size=float("inf"), skip_invalid_smiles=False)
        test_data = MoleculeDataset(test_data)
        mol_collator = MolCollator(args=args, shared_dict={})
        from torch.utils.data import DataLoader
        loader = DataLoader(test_data, batch_size=1, shuffle=False,
                            num_workers=0, collate_fn=mol_collator)

        for item in loader:
            _, batch, features_batch, _, _ = item
            with torch.no_grad():
                _, _, _, _, _, a_scope, _, _ = batch
                preds = model(batch, features_batch)
                i = a_scope[0]
                atom_emb = preds[0][i[0]:i[0]+i[1]].data.cpu().numpy()
                total_emb = preds[2][0].cpu().numpy()  # 4800D from Grover
                # Double atom embeddings to match training convention (1200 -> 2400)
                atom_emb = np.concatenate([atom_emb, atom_emb], axis=1)
                # Append 85 fgtasklabel features to match training (4800 + 85 = 4885)
                fg_features = _generate_fgtasklabel(smiles)
                total_emb = np.concatenate([total_emb, fg_features])
                return atom_emb, total_emb
    finally:
        os.unlink(tmp.name)

    raise RuntimeError("Grover fingerprint generation failed")


# ── Morgan Fingerprint ───────────────────────────────────────────────────────

def generate_morgan_fingerprint(smiles: str, n_bits: int = 1024) -> np.ndarray:
    """Generate Morgan fingerprint. Returns [1024]."""
    from rdkit import Chem
    from rdkit.Chem import AllChem
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles}")
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=n_bits)
    return np.array(fp, dtype=np.float32)


# ── Reaction Features (atom graph from SMILES) ──────────────────────────────

def parse_smiles_to_reaction_features(smiles: str) -> dict:
    """Parse SMILES into reaction features matching the training pipeline."""
    from rdkit import Chem
    from rdkit.Chem.rdchem import BondType, HybridizationType
    from torch_scatter import scatter

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles}")

    # Canonicalize
    tmp = Chem.MolFromSmiles(smiles)
    Chem.SanitizeMol(tmp)
    for atom in tmp.GetAtoms():
        atom.SetAtomMapNum(0)
    canonical = Chem.MolFromSmiles(Chem.MolToSmiles(tmp))

    num_atoms = canonical.GetNumAtoms()

    # Element array
    element = np.array([canonical.GetAtomWithIdx(i).GetAtomicNum()
                        for i in range(num_atoms)], dtype=np.int64)

    # Atom features: [atomic_number, aromatic, degree, num_hs, sp, sp2, sp3]
    atomic_number, aromatic, degree = [], [], []
    sp, sp2, sp3 = [], [], []
    for i in range(num_atoms):
        atom = canonical.GetAtomWithIdx(i)
        atomic_number.append(atom.GetAtomicNum())
        aromatic.append(1 if atom.GetIsAromatic() else 0)
        degree.append(atom.GetDegree())
        hyb = atom.GetHybridization()
        sp.append(1 if hyb == HybridizationType.SP else 0)
        sp2.append(1 if hyb == HybridizationType.SP2 else 0)
        sp3.append(1 if hyb == HybridizationType.SP3 else 0)

    node_type = torch.tensor(atomic_number, dtype=torch.long)
    row, col = [], []
    BOND_TYPES_MAP = {t: i for i, t in enumerate(BondType.names.values())}
    edge_type_list = []
    for bond in canonical.GetBonds():
        s, e = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        row += [s, e]
        col += [e, s]
        edge_type_list += 2 * [BOND_TYPES_MAP[bond.GetBondType()]]

    row_t = torch.tensor(row, dtype=torch.long)
    col_t = torch.tensor(col, dtype=torch.long)
    hs = (node_type == 1).to(torch.float)
    num_hs = scatter(hs[row_t], col_t, dim_size=num_atoms).numpy()

    atom_feature = np.array([atomic_number, aromatic, degree, num_hs, sp, sp2, sp3],
                            dtype=np.int64).transpose()

    edge_index = np.array([row, col], dtype=np.int64)
    edge_type = np.array(edge_type_list, dtype=np.int64)
    if len(row) > 0:
        perm = (edge_index[0] * num_atoms + edge_index[1]).argsort()
        edge_index = edge_index[:, perm]
        edge_type = edge_type[perm]

    return {
        'element': element,
        'edge_index': edge_index,
        'edge_type': edge_type,
        'atom_feature': atom_feature,
        'reaction_attention_label': np.zeros(num_atoms),
        'num_nodes': np.array(num_atoms),
    }


# ── 3D Structure from SMILES ────────────────────────────────────────────────

def generate_3d_structure(smiles: str):
    """Generate 3D coordinates from SMILES using RDKit. Returns ligand dict."""
    from rdkit import Chem
    from rdkit.Chem import AllChem
    from rdkit.Chem.rdchem import HybridizationType

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles}")
    mol = Chem.AddHs(mol)
    AllChem.EmbedMolecule(mol, AllChem.ETKDG())
    if mol.GetNumConformers() == 0:
        AllChem.EmbedMolecule(mol, AllChem.ETKDGv3())
    if mol.GetNumConformers() == 0:
        # Fallback: random coords
        AllChem.EmbedMolecule(mol, useRandomCoords=True)
    mol = Chem.RemoveHs(mol)

    num_atoms = mol.GetNumAtoms()
    pos = mol.GetConformer().GetPositions().astype(np.float32) if mol.GetNumConformers() > 0 else np.zeros((num_atoms, 3), dtype=np.float32)

    element = []
    atomic_number_list, aromatic_list, degree_list, hybrid_list = [], [], [], []
    for i in range(num_atoms):
        atom = mol.GetAtomWithIdx(i)
        element.append(atom.GetAtomicNum())
        atomic_number_list.append(atom.GetAtomicNum())
        aromatic_list.append(1 if atom.GetIsAromatic() else 0)
        degree_list.append(atom.GetDegree())
        HYBRID_TYPES = {t: idx for idx, t in enumerate(HybridizationType.names.values())}
        hybrid_list.append(HYBRID_TYPES[atom.GetHybridization()])

    from torch_scatter import scatter
    node_type = torch.tensor(element, dtype=torch.long)
    row, col, edge_type = [], [], []
    BOND_TYPES = {}
    from rdkit.Chem.rdchem import BondType
    BOND_TYPES[BondType.SINGLE] = 1
    BOND_TYPES[BondType.DOUBLE] = 2
    BOND_TYPES[BondType.TRIPLE] = 3
    BOND_TYPES[BondType.AROMATIC] = 4
    BOND_TYPES[BondType.UNSPECIFIED] = 5

    for bond in mol.GetBonds():
        s, e = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        row += [s, e]
        col += [e, s]
        edge_type += 2 * [BOND_TYPES.get(bond.GetBondType(), 5)]

    row_t = torch.tensor(row, dtype=torch.long)
    col_t = torch.tensor(col, dtype=torch.long)
    hs = (node_type == 1).to(torch.float)
    num_hs = scatter(hs[row_t], col_t, dim_size=num_atoms).numpy() if len(row) > 0 else np.zeros(num_atoms)

    atom_feature = np.array([atomic_number_list, aromatic_list, degree_list,
                             num_hs, hybrid_list], dtype=np.int64).transpose()

    edge_index = np.array([row, col], dtype=np.int64) if len(row) > 0 else np.zeros((2, 0), dtype=np.int64)
    edge_type_arr = np.array(edge_type, dtype=np.int64) if len(edge_type) > 0 else np.zeros(0, dtype=np.int64)
    if len(row) > 0:
        perm = (edge_index[0] * num_atoms + edge_index[1]).argsort()
        edge_index = edge_index[:, perm]
        edge_type_arr = edge_type_arr[perm]

    ligand_dict = {
        'element': np.array(element, dtype=np.int64),
        'pos': pos,
        'bond_index': edge_index,
        'bond_type': edge_type_arr,
        'atom_feature': atom_feature,
        'center_of_mass': pos.mean(axis=0) if len(pos) > 0 else np.zeros(3, dtype=np.float32),
        'index': np.arange(num_atoms, dtype=np.int64),
    }
    return ligand_dict


# ── Construct Full Input Batch ───────────────────────────────────────────────

def build_input_batch(
    esm_embedding: np.ndarray,
    reaction_features: dict,
    grover_atom: np.ndarray,
    grover_total: np.ndarray,
    morgan_fp: np.ndarray,
    ligand_dict: dict,
    max_enzyme_length: int = 1450,
    max_substrate_length: int = 280,
):
    """Build a single-sample batch for the SS model."""
    from torch_geometric.data import Data, Batch
    from torch_geometric.nn import knn_graph
    import torch.nn.functional as F

    device = 'cpu'  # Build on CPU, move to device later

    # ── Sequence features (ESM embedding) ──
    L = esm_embedding.shape[0]
    padded_emb = np.zeros((max_enzyme_length, 1280), dtype=np.float32)
    padded_emb[:L] = esm_embedding
    enzyme_padding_mask = np.zeros(max_enzyme_length, dtype=bool)
    enzyme_padding_mask[L:] = True

    # ── Reaction features (atom graph) ──
    N_atoms = reaction_features['element'].shape[0]
    reaction_padding_mask = np.zeros(max_substrate_length, dtype=bool)
    reaction_padding_mask[N_atoms:] = True

    # Pad grover atom embeddings
    grover_padded = np.zeros((max_substrate_length, 2400), dtype=np.float32)
    grover_padded[:min(N_atoms, max_substrate_length)] = grover_atom[:min(N_atoms, max_substrate_length)]

    # ── Structure features (3D graph) ──
    # Zero protein (ligand-only mode)
    protein_dict = {
        'element': np.array([], dtype=np.int64),
        'pos': np.zeros((0, 3), dtype=np.float32),
        'is_backbone': np.array([], dtype=bool),
        'atom_name': [],
        'atom_to_aa_type': np.array([], dtype=np.int64),
    }

    # Build structure Data object
    N_lig = ligand_dict['element'].shape[0]
    N_prot = 0

    # Protein atom features: one-hot element [H,C,N,O,S,Se] + one-hot AA type (21) + is_backbone
    # For zero protein, these are empty
    protein_atom_feature = torch.zeros((0, 28), dtype=torch.float)

    # Ligand atom features
    LIGAND_ELEMENTS = torch.LongTensor([1, 6, 7, 8, 9, 15, 16, 17, 26, 35, 53])
    lig_element = torch.from_numpy(ligand_dict['element']).long()
    element_feat = (lig_element.view(-1, 1) == LIGAND_ELEMENTS.view(1, -1)).float()

    # Atom feature columns: AtomicNumber(1), Aromatic(1), Degree(6), NumHs(6), Hybridization(len)
    from rdkit.Chem.rdchem import HybridizationType
    ATOM_FEATS = {'AtomicNumber': 1, 'Aromatic': 1, 'Degree': 6, 'NumHs': 6,
                  'Hybridization': len(HybridizationType.values)}
    lig_atom_feat_raw = torch.from_numpy(ligand_dict['atom_feature']).long()
    atom_feat_parts = []
    for i, (k, v) in enumerate(ATOM_FEATS.items()):
        feat = lig_atom_feat_raw[:, i:i+1]
        if v > 1:
            feat = (feat == torch.LongTensor(list(range(v))).view(1, -1))
        else:
            if k == 'AtomicNumber':
                feat = feat.float() / 100.0
            else:
                feat = feat.float()
        atom_feat_parts.append(feat.float())
    lig_atom_feature_full = torch.cat([element_feat] + atom_feat_parts, dim=-1)

    # Pad for complex graph
    protein_x = torch.cat([torch.zeros((N_lig, protein_atom_feature.shape[1] if protein_atom_feature.shape[1] > 0 else 28)),
                           protein_atom_feature], dim=0)
    ligand_x = torch.cat([lig_atom_feature_full,
                          torch.zeros((N_prot, lig_atom_feature_full.shape[1]))], dim=0)
    ligand_mask = torch.cat([torch.ones(N_lig), torch.zeros(N_prot)])
    protein_mask = torch.cat([torch.zeros(N_lig), torch.ones(N_prot)])

    # Edge features
    lig_pos = torch.from_numpy(ligand_dict['pos']).float()
    lig_bond_index = torch.from_numpy(ligand_dict['bond_index']).long()
    lig_bond_type = torch.from_numpy(ligand_dict['bond_type']).long()

    # Distance expansion (Gaussian smearing)
    pos_all = lig_pos  # No protein atoms
    if lig_bond_index.shape[1] > 0:
        dst_idx, src_idx = lig_bond_index
        dist = torch.norm(pos_all[dst_idx] - pos_all[src_idx], p=2, dim=-1)
    else:
        dist = torch.zeros(0)

    # Gaussian smearing (cutoff=10, num_gaussians=32 matching training config)
    num_r_gaussian = 32
    cutoff = 10.0
    offset = torch.linspace(0, cutoff, num_r_gaussian)
    coeff = -0.5 / (offset[1] - offset[0]).item()**2
    dist_feat = torch.exp(coeff * (dist.view(-1, 1) - offset.view(1, -1))**2)

    # Bond type one-hot (6 classes)
    lig_bond_type_clamped = lig_bond_type.clone()
    lig_bond_type_clamped[lig_bond_type_clamped == 12] = 5
    bond_feat = F.one_hot(lig_bond_type_clamped - 1, num_classes=6).float() if len(lig_bond_type) > 0 else torch.zeros((0, 6))

    # Cross bond indicator (all False for ligand-only)
    cross_bond = torch.zeros((lig_bond_index.shape[1], 1))

    complex_edge_attr = torch.cat([bond_feat, dist_feat, cross_bond], dim=-1)
    complex_edge_index = lig_bond_index

    # Ligand index (atom -> substrate position mapping)
    ligand_index = torch.arange(N_lig).long()

    # Build batch directly (single sample, no PyG Batch to avoid str collation issues)
    data = Data()
    # Sequence features (already batch dim = 1)
    data.embedding = torch.from_numpy(padded_emb).float().view(1, max_enzyme_length, 1280)
    data.enzyme_padding_mask = torch.from_numpy(enzyme_padding_mask).view(1, max_enzyme_length)
    data.reaction_padding_mask = torch.from_numpy(reaction_padding_mask).view(1, max_substrate_length)
    data.grover = torch.from_numpy(grover_padded).float().view(1, max_substrate_length, 2400)
    data.grover_mean = torch.from_numpy(grover_total).float().unsqueeze(0)
    data.morgan = torch.from_numpy(morgan_fp).float().unsqueeze(0)

    # Structure features
    data.protein_x = protein_x
    data.ligand_x = ligand_x
    data.ligand_mask = ligand_mask
    data.protein_mask = protein_mask
    data.complex_edge_index = complex_edge_index
    data.complex_edge_attr = complex_edge_attr.float()
    data.ligand_index = ligand_index
    data.mask_use_complex = torch.ones(N_lig + N_prot)
    data.mask_use_ligand = torch.zeros(N_lig + N_prot)

    # String tags stored as tensors for batching
    data.str_tag = 'ligand'
    data.tag = torch.tensor([0], dtype=torch.long)

    # Labels (dummy)
    data.label = torch.tensor(0.0)
    data.sample_weight = torch.tensor(1.0)

    # PyG batch index for ligand_index (all belong to graph 0)
    data.ligand_index_batch = torch.zeros(N_lig + N_prot, dtype=torch.long)

    # Batch dimension = 1 for single sample
    data.batch = torch.zeros(N_lig + N_prot, dtype=torch.long)
    data.num_graphs = 1

    return data


# ── Load SS Model ────────────────────────────────────────────────────────────

def _load_ss_model(checkpoint_path: str, repo_root: str):
    if checkpoint_path in _ss_models:
        return _ss_models[checkpoint_path]

    src_dir = os.path.join(repo_root, 'src')
    if src_dir not in sys.path:
        sys.path.insert(0, src_dir)

    from Models.ss import SS
    from easydict import EasyDict

    # Config matching the training setup
    config = EasyDict({
        'model': {
            'hidden_dim': 128,
            'use_gnn': True,
            'feature_norm': False,
            'header': {'act_fn': 'relu', 'norm': True, 'num_layers': 3},
            'graph': {
                'name': 'egnn', 'hidden_dim': 128, 'act_fn': 'relu',
                'num_layers': 3, 'attention': True, 'residual': True,
                'norm': 'layer', 'mode': 'complex',
            },
            'cross_attention': {'n_head': 8, 'dropout': 0.9},
        },
        'data': {
            'max_substrate_length': 280,
            'max_enzyme_length': 1450,
            'features': ['morgan', 1024, 'grover_mean', 4885],
            'atom_features': ['grover', 2400],
        },
        'transform': {
            'dist_noise': False, 'cutoff': 10,
            'num_r_gaussian': 32, 'k': 48,
        },
        'training': {
            'optimizer': 'adamW', 'lr': 3e-4, 'weight_decay': 0,
            'warmup_epochs': 8, 'min_lr': 5e-6,
            'sched_factor': 0.5, 'sched_patience': 10,
            'val_frequency': 1, 'gradient_clip_val': 8,
            'accumulate_grad_batches': 2,
        },
    })

    model = SS.load_from_checkpoint(checkpoint_path, config=config, map_location='cpu')
    model.eval()
    device = _get_device()
    model = model.to(device)
    _ss_models[checkpoint_path] = model
    return model


# ── Main Prediction Function ─────────────────────────────────────────────────

def predict_single(protein_sequence: str, substrate_smiles: str,
                   model_idx: int, repo_root: str) -> dict:
    """Run full prediction pipeline for a single enzyme-substrate pair."""
    from torch_geometric.data import Batch

    device = _get_device()

    # 1. ESM-2 embedding
    esm_emb = generate_esm_embedding(protein_sequence)

    # 2. Reaction features
    rxn_feat = parse_smiles_to_reaction_features(substrate_smiles)

    # 3. Grover fingerprints
    grover_atom, grover_total = generate_grover_fingerprint(substrate_smiles, repo_root)

    # 4. Morgan fingerprint
    morgan_fp = generate_morgan_fingerprint(substrate_smiles, n_bits=1024)

    # 5. 3D structure
    lig_dict = generate_3d_structure(substrate_smiles)

    # 6. Build batch
    batch = build_input_batch(
        esm_embedding=esm_emb,
        reaction_features=rxn_feat,
        grover_atom=grover_atom,
        grover_total=grover_total,
        morgan_fp=morgan_fp,
        ligand_dict=lig_dict,
    )

    # 7. Load model
    checkpoints = [
        os.path.join(repo_root, 'saved_model', f'complete-full-random-all-{i}-complex',
                     'run_0', 'models', 'best-checkpoint.ckpt')
        for i in range(4)
    ]
    model = _load_ss_model(checkpoints[model_idx], repo_root)

    # 8. Inference
    batch = batch.to(device)
    with torch.no_grad():
        logit, _ = model(batch)
    logit_val = logit.squeeze().cpu().item()
    prob = torch.sigmoid(torch.tensor(logit_val)).item()

    return {
        'probability': round(prob, 6),
        'logit': round(logit_val, 6),
        'protein_length': len(protein_sequence),
        'substrate_atoms': rxn_feat['element'].shape[0],
        'model_used': f'complete-full-random-all-{model_idx}-complex',
        'device': str(device),
    }


def predict_ensemble(protein_sequence: str, substrate_smiles: str,
                     repo_root: str) -> dict:
    """Average predictions from all 4 checkpoints."""
    results = []
    logits = []
    for i in range(4):
        r = predict_single(protein_sequence, substrate_smiles, i, repo_root)
        if 'error' in r:
            return r
        results.append(r)
        logits.append(r['logit'])

    avg_logit = np.mean(logits)
    avg_prob = float(torch.sigmoid(torch.tensor(avg_logit)).item())

    return {
        'probability': round(avg_prob, 6),
        'logit': round(float(avg_logit), 6),
        'protein_length': results[0]['protein_length'],
        'substrate_atoms': results[0]['substrate_atoms'],
        'model_used': 'ensemble_4_models',
        'individual_logits': [round(l, 6) for l in logits],
        'device': results[0]['device'],
    }


# ── Entry Point ──────────────────────────────────────────────────────────────

def main():
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')

    # Save real stdout for final JSON output; redirect to stderr during computation
    # to prevent library prints (Grover, ESM, PyTorch) from contaminating JSON
    _real_stdout = sys.stdout

    try:
        payload = json.loads(sys.stdin.read())
    except Exception as e:
        _real_stdout.write(json.dumps({'error': f'Failed to parse input: {e}'}) + '\n')
        sys.exit(0)

    action = payload.get('action', 'predict')
    repo_root = payload.get('repo_root') or os.environ.get('ZHUANGZHOU_ENZYME_SPECIFICITY_ROOT', '')
    if not repo_root:
        _real_stdout.write(json.dumps({
            'error': 'enzyme specificity repo not configured',
            'hint': 'Set ZHUANGZHOU_ENZYME_SPECIFICITY_ROOT or pass repo_root',
        }) + '\n')
        sys.exit(0)

    # Redirect stdout to stderr during model computation
    sys.stdout = sys.stderr

    try:
        if action == 'predict':
            seq = payload['protein_sequence']
            smi = payload['substrate_smiles']
            idx = payload.get('model_idx', 0)
            ensemble = payload.get('ensemble', False)

            if ensemble:
                result = predict_ensemble(seq, smi, repo_root)
            else:
                result = predict_single(seq, smi, idx, repo_root)
            _real_stdout.write(json.dumps(result) + '\n')

        elif action == 'batch_predict':
            pairs = payload['pairs']
            idx = payload.get('model_idx', 0)
            results = []
            for pair in pairs:
                try:
                    r = predict_single(pair['protein_sequence'],
                                       pair['substrate_smiles'], idx, repo_root)
                    results.append(r)
                except Exception as e:
                    results.append({'error': str(e)})
            _real_stdout.write(json.dumps({'results': results}) + '\n')

        else:
            _real_stdout.write(json.dumps({'error': f'Unknown action: {action}'}) + '\n')

    except Exception as e:
        _real_stdout.write(json.dumps({'error': str(e), 'traceback': traceback.format_exc()}) + '\n')
    finally:
        sys.stdout = _real_stdout


if __name__ == '__main__':
    main()
