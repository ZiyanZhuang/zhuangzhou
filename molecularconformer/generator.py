"""
Conformer Generator — Core Engine
====================================
SMILES -> 3D conformer(s) -> SDF/MOL file.

Two modes:
  1. RDKit (default, no GPU): EmbedMultipleConfs + MMFF/UFF optimization
  2. UniMol-enhanced (optional, GPU): Better quality via dplm2 bridge

Design: self-contained, no external imports beyond RDKit + numpy.
"""

import logging
import os
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np

logger = logging.getLogger(__name__)

try:
    from rdkit import Chem, RDLogger
    from rdkit.Chem import AllChem, Descriptors, rdMolDescriptors
    from rdkit.Chem import rdDistGeom, rdForceFieldHelpers

    RDLogger.DisableLog("rdApp.*")
    RDKIT_AVAILABLE = True
except ImportError:
    RDKIT_AVAILABLE = False


def smiles_to_3d(
    smiles: str,
    output_path: str = None,
    num_conformers: int = 1,
    seed: int = 42,
    mode: str = "fast",
    optimize: str = "mmff",
    remove_hs: bool = False,
    prune_rms_thresh: float = 0.5,
    energy_window: float = 50.0,
    add_props: bool = True,
) -> dict:
    """
    Convert SMILES to 3D conformer(s) and optionally save as SDF.

    Args:
        smiles: SMILES string
        output_path: Output SDF path (auto-generated if None)
        num_conformers: Number of conformers to generate
        seed: Random seed
        mode: 'fast' (1 attempt) or 'heavy' (5000 max attempts)
        optimize: Force field for optimization: 'mmff', 'uff', or None
        remove_hs: Whether to remove hydrogens from output
        prune_rms_thresh: RMSD threshold for pruning similar conformers
        energy_window: Keep conformers within this energy window (kcal/mol)
        add_props: Add molecular properties (MW, logP, etc.) to SDF

    Returns:
        dict with keys: output_path, smiles, num_conformers, energies, success
    """
    if not RDKIT_AVAILABLE:
        return {"error": "RDKit not installed", "hint": "pip install rdkit"}

    # Parse SMILES
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return {"error": f"Invalid SMILES: {smiles}", "hint": "Check SMILES syntax"}

    # Add hydrogens (required for 3D embedding)
    mol = AllChem.AddHs(mol)

    # Generate conformer(s)
    if num_conformers == 1:
        result = _embed_single(mol, seed, mode)
    else:
        result = _embed_multiple(mol, num_conformers, seed, mode, prune_rms_thresh)

    if result.get("error"):
        return result

    mol = result["mol"]
    conf_ids = result["conf_ids"]

    # Optimize with force field
    energies = []
    if optimize:
        energies = _optimize_conformers(mol, conf_ids, optimize)
        # Filter by energy window
        if energies and energy_window > 0 and len(energies) > 1:
            mol, conf_ids, energies = _filter_by_energy(
                mol, conf_ids, energies, energy_window
            )

    # Remove hydrogens if requested (after optimization)
    if remove_hs:
        mol = AllChem.RemoveAllHs(mol)
        conf_ids = [c.GetId() for c in mol.GetConformers()]

    # Add molecular properties
    if add_props:
        _add_mol_properties(mol, smiles)

    # Determine output path
    if output_path is None:
        # Generate name from SMILES (sanitized)
        safe_name = _sanitize_filename(smiles)
        output_path = os.path.join(os.getcwd(), f"{safe_name}_3d.sdf")

    output_path = str(Path(output_path).resolve())
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Write SDF
    writer = Chem.SDWriter(output_path)
    for i, cid in enumerate(conf_ids):
        if energies:
            mol.SetProp("Energy_kcal_mol", f"{energies[i]:.2f}")
            mol.SetProp("Conformer_ID", str(i))
        writer.write(mol, confId=cid)
    writer.close()

    return {
        "output_path": output_path,
        "smiles": smiles,
        "num_conformers": len(conf_ids),
        "num_atoms": mol.GetNumAtoms(),
        "energies": [round(e, 2) for e in energies] if energies else [],
        "method": f"rdkit_{optimize or 'no_opt'}",
        "success": True,
    }


def batch_smiles_to_sdf(
    smiles_list: List[str],
    output_dir: str = None,
    output_combined: str = None,
    num_conformers: int = 1,
    seed: int = 42,
    optimize: str = "mmff",
    remove_hs: bool = False,
    **kwargs,
) -> dict:
    """
    Batch convert multiple SMILES to 3D SDF files.

    Args:
        smiles_list: List of SMILES strings
        output_dir: Directory for individual SDF files (one per molecule)
        output_combined: Path for single combined SDF (all molecules)
        num_conformers: Conformers per molecule
        seed, optimize, remove_hs: Passed to smiles_to_3d

    Returns:
        dict with keys: results (list), succeeded, failed, output_dir/output_combined
    """
    if not smiles_list:
        return {"error": "Empty SMILES list"}

    if output_dir is None and output_combined is None:
        output_dir = os.path.join(os.getcwd(), "batch_conformers")

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    results = []
    succeeded = 0
    failed = 0

    # Combined writer
    combined_writer = None
    if output_combined:
        os.makedirs(os.path.dirname(os.path.abspath(output_combined)), exist_ok=True)
        combined_writer = Chem.SDWriter(output_combined)

    for i, smi in enumerate(smiles_list):
        if output_dir:
            safe_name = _sanitize_filename(smi)
            out_path = os.path.join(output_dir, f"{safe_name}_{i:04d}.sdf")
        else:
            out_path = None

        res = smiles_to_3d(
            smi,
            output_path=out_path,
            num_conformers=num_conformers,
            seed=seed,
            optimize=optimize,
            remove_hs=remove_hs,
            **kwargs,
        )

        if res.get("success"):
            succeeded += 1
            # Add to combined SDF
            if combined_writer and out_path and os.path.exists(out_path):
                suppl = Chem.SDMolSupplier(out_path, removeHs=False)
                for mol in suppl:
                    if mol is not None:
                        combined_writer.write(mol)
        else:
            failed += 1

        results.append({"smiles": smi, **res})

    if combined_writer:
        combined_writer.close()

    return {
        "results": results,
        "succeeded": succeeded,
        "failed": failed,
        "total": len(smiles_list),
        "output_dir": output_dir,
        "output_combined": output_combined,
    }


def sdf_from_mol(mol, output_path: str, remove_hs: bool = False) -> dict:
    """Save an RDKit Mol object with conformer(s) to SDF."""
    if not RDKIT_AVAILABLE:
        return {"error": "RDKit not installed"}

    if remove_hs:
        mol = AllChem.RemoveAllHs(mol)

    output_path = str(Path(output_path).resolve())
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    writer = Chem.SDWriter(output_path)
    for conf in mol.GetConformers():
        writer.write(mol, confId=conf.GetId())
    writer.close()

    return {
        "output_path": output_path,
        "num_conformers": mol.GetNumConformers(),
        "success": True,
    }


def validate_smiles(smiles: str) -> dict:
    """Validate a SMILES string and return basic molecular info."""
    if not RDKIT_AVAILABLE:
        return {"error": "RDKit not installed"}

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return {"valid": False, "error": f"Cannot parse SMILES: {smiles}"}

    return {
        "valid": True,
        "canonical_smiles": Chem.MolToSmiles(mol),
        "num_atoms": mol.GetNumAtoms(),
        "num_heavy_atoms": mol.GetNumHeavyAtoms(),
        "molecular_formula": rdMolDescriptors.CalcMolFormula(mol),
        "molecular_weight": round(Descriptors.MolWt(mol), 2),
    }


# ── Internal helpers ──────────────────────────────────────────────────────────


def _embed_single(mol, seed: int, mode: str) -> dict:
    """Embed a single conformer (mirrors unimol_tools inner_smi2coords logic)."""
    try:
        res = AllChem.EmbedMolecule(mol, randomSeed=seed)
        if res == 0:
            return {"mol": mol, "conf_ids": [0]}
        elif res == -1 and mode == "heavy":
            res2 = AllChem.EmbedMolecule(mol, maxAttempts=5000, randomSeed=seed)
            if res2 == 0:
                return {"mol": mol, "conf_ids": [0]}
            # Fall back to 2D
            AllChem.Compute2DCoords(mol)
            logger.warning("3D embedding failed, using 2D coordinates")
            return {"mol": mol, "conf_ids": [0]}
        else:
            AllChem.Compute2DCoords(mol)
            logger.warning("3D embedding failed, using 2D coordinates")
            return {"mol": mol, "conf_ids": [0]}
    except Exception as e:
        return {"error": f"Embedding failed: {e}", "hint": "Try mode='heavy'"}


def _embed_multiple(
    mol, num_conformers: int, seed: int, mode: str, prune_rms_thresh: float
) -> dict:
    """Embed multiple conformers using ETKDG."""
    try:
        params = rdDistGeom.ETKDGv3()
        params.randomSeed = seed
        params.pruneRmsThresh = prune_rms_thresh
        params.numThreads = 0  # Use all cores
        if mode == "heavy":
            params.maxIterations = 5000

        conf_ids = AllChem.EmbedMultipleConfs(mol, num_conformers, params)
        conf_ids = list(conf_ids)

        if not conf_ids:
            # Fallback to single conformer
            logger.warning("Multi-conformer failed, trying single")
            return _embed_single(mol, seed, mode)

        return {"mol": mol, "conf_ids": conf_ids}
    except Exception as e:
        return {"error": f"Multi-conformer embedding failed: {e}"}


def _optimize_conformers(
    mol, conf_ids: List[int], method: str = "mmff"
) -> List[float]:
    """Optimize conformers with force field. Returns energies."""
    energies = []
    for cid in conf_ids:
        try:
            if method == "mmff":
                res = AllChem.MMFFOptimizeMolecule(mol, confId=cid, maxIters=500)
                ff = AllChem.MMFFGetMoleculeForceField(
                    mol, AllChem.MMFFGetMoleculeProperties(mol), confId=cid
                )
                energy = ff.CalcEnergy() if ff else 0.0
            elif method == "uff":
                AllChem.UFFOptimizeMolecule(mol, confId=cid, maxIters=500)
                ff = AllChem.UFFGetMoleculeForceField(mol, confId=cid)
                energy = ff.CalcEnergy() if ff else 0.0
            else:
                energy = 0.0
            energies.append(energy)
        except Exception:
            energies.append(float("inf"))
    return energies


def _filter_by_energy(
    mol, conf_ids: List[int], energies: List[float], window: float
) -> Tuple:
    """Keep only conformers within energy window of minimum."""
    min_e = min(energies)
    keep = [(cid, e) for cid, e in zip(conf_ids, energies) if e - min_e <= window]
    if not keep:
        keep = [(conf_ids[0], energies[0])]

    # Sort by energy
    keep.sort(key=lambda x: x[1])
    new_ids = [k[0] for k in keep]
    new_energies = [k[1] for k in keep]
    return mol, new_ids, new_energies


def _add_mol_properties(mol, smiles: str):
    """Add standard molecular properties to the mol object."""
    try:
        mol_no_h = AllChem.RemoveAllHs(mol)
        mol.SetProp("SMILES", smiles)
        mol.SetProp("Canonical_SMILES", Chem.MolToSmiles(mol_no_h))
        mol.SetProp("MolecularFormula", rdMolDescriptors.CalcMolFormula(mol_no_h))
        mol.SetDoubleProp("MolecularWeight", Descriptors.MolWt(mol_no_h))
        mol.SetDoubleProp("LogP", Descriptors.MolLogP(mol_no_h))
        mol.SetIntProp("HBA", rdMolDescriptors.CalcNumHBA(mol_no_h))
        mol.SetIntProp("HBD", rdMolDescriptors.CalcNumHBD(mol_no_h))
        mol.SetDoubleProp("TPSA", rdMolDescriptors.CalcTPSA(mol_no_h))
        mol.SetIntProp("RotatableBonds", rdMolDescriptors.CalcNumRotatableBonds(mol_no_h))
    except Exception:
        pass


def _sanitize_filename(smiles: str, max_len: int = 40) -> str:
    """Convert SMILES to a safe filename."""
    safe = smiles.replace("/", "_").replace("\\", "_").replace(":", "_")
    safe = safe.replace("(", "").replace(")", "").replace("[", "").replace("]", "")
    safe = safe.replace("=", "").replace("#", "").replace("@", "")
    safe = safe.replace("+", "p").replace("-", "m")
    if len(safe) > max_len:
        import hashlib

        h = hashlib.md5(smiles.encode()).hexdigest()[:8]
        safe = safe[: max_len - 9] + "_" + h
    return safe
