# CYPamber

**Automated Orchestration of High-Fidelity Cytochrome P450 and Ligand MD Parameterization via PySCF and Shahrokh Protocol.**

---

## Overview
`CYPamber` is a Python-native toolkit designed for computational biologists and chemists to automate the complex pre-processing required for Molecular Dynamics (MD) simulations of metalloproteins, specifically Cytochrome P450 (CYP) systems. 

Manual parameterization of metal-binding motifs and high-precision charges for diverse ligands remains a bottleneck. `CYPamber` bridges this gap by integrating state-of-the-art quantum chemistry (PySCF) with established force field protocols (Shahrokh et al.).

## Key Features
- **Shahrokh Protocol Automation**: Automates Heme state detection (CPDI, DIOXY, IC6), Kabsch-based structure superposition, and Fe-S coordination bonding.
- **PySCF-driven RESP Charges**: First-principles ligand parameterization using HF/6-31G* RESP charges calculated via PySCF and Multiwfn, replacing traditional Gaussian dependencies.
- **End-to-End AMBER Integration**: Generates production-ready `tleap` inputs and multi-stage MD equilibration scripts.
- **WSL Bridge**: Seamless execution of Linux-based Amber/QM tools on Windows systems.

## Installation
```bash
git clone https://github.com/yourusername/CYPamber.git
cd CYPamber
pip install -r requirements.txt
```

## Quick Start (CYP450 MD Setup)
```python
from cypamber.heme.pipeline import HemePipeline
from cypamber.ligand.parameterize import LigandParameterizer

# 1. Parameterize Heme and Protein
heme_pl = HemePipeline(params_base="./data/heme_params")
res = heme_pl.run(pdb_path="2CPP.pdb", output_dir="./prep_cyp")

# 2. Parameterize Ligand with PySCF RESP
lig_pl = LigandParameterizer()
res_lig = lig_pl.run(ligand_path="camphor.mol2", output_dir="./prep_lig", charge=0)
```

## Citation
If you use `CYPamber` in your research, please cite our Bioinformatics Application Note:
> *Your Name et al., CYPamber: An Automated Framework for High-Fidelity Metalloprotein Simulation Prep. Bioinformatics (2026).*
