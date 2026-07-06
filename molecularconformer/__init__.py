"""
Conformer Generator — SMILES to 3D SDF
========================================
Self-contained module for generating correct 3D molecular conformations.

Methods:
  - RDKit (default): EmbedMultipleConfs + MMFF/UFF optimization
  - UniMol-enhanced (optional): via dplm2 bridge for improved quality

Usage:
    from zhuangzhou.molecularconformer import smiles_to_3d, batch_smiles_to_sdf, validate_smiles

    # Single molecule
    result = smiles_to_3d("CC(=O)OC1=CC=CC=C1C(=O)O", output_path="aspirin_3d.sdf")

    # Multiple conformers
    result = smiles_to_3d("c1ccccc1", num_conformers=10, optimize="mmff")

    # Batch processing
    result = batch_smiles_to_sdf(
        ["CCO", "c1ccccc1", "CC(=O)O"],
        output_dir="./conformers/",
        num_conformers=5,
    )

    # Validate SMILES
    info = validate_smiles("CC(=O)OC1=CC=CC=C1C(=O)O")
"""

from .generator import smiles_to_3d, batch_smiles_to_sdf, sdf_from_mol, validate_smiles

__all__ = ["smiles_to_3d", "batch_smiles_to_sdf", "sdf_from_mol", "validate_smiles"]
