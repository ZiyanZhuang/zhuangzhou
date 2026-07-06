"""
Domain Specialist Agents — Zhuangzhou v0.8.0
=============================================
Each agent is tuned for a specific bioinformatics domain:

  MDAgent          — GROMACS MD, trajectory analysis, force fields
  DockingAgent     — Vina/SmartDocking, PLIP, binding site analysis
  LiteratureAgent  — PubMed, UniProt, PDB, literature synthesis
  ChemAgent        — RDKit, PubChem, conformer generation, SMILES/SDF
  StructureAgent   — PDB prep, mutagenesis, structure comparison
  MemoryAgent      — Session memory, KG overlay, context management

Domain prompts encode hard-won institutional knowledge so the LLM
does not have to rediscover common pitfalls each session.
"""

from .base import SpecialistAgent


# =============================================================================
# Molecular Dynamics Specialist
# =============================================================================

class MDAgent(SpecialistAgent):
    name = "md"
    description = "GROMACS MD simulation expert — force fields, equilibration, analysis"
    skill_categories = ["simulation"]
    system_addendum = """## MD Simulation Domain Rules (CRITICAL)

### Force Field Selection
- HEME/Fe metalloprotein → metalloprotein_md or metalloprotein_ligand_md (CHARMM27/36 + bonded Fe-S model)
- Protein + organic ligand → run_protein_ligand_md (CHARMM36 + CGenFF, NOT GAFF — combination rule conflict!)
- Pure protein → run_protein_md (CHARMM36)
- AMBER force field is for AMBER MD (amber_complex_md), not GROMACS

### Key Parameters
- NPT equilibration: ALWAYS use C-rescale/Berendsen barostat (NOT Parrinello-Rahman, unequilibrated systems explode)
- Production MD: Parrinello-Rahman (already equilibrated)
- Bonds: h-bonds ONLY (NEVER all-bonds — causes LINCS failures)
- After EM: always trjconv -pbc whole before NVT (PBC wrap causes NVT blowup at step 0)

### Metalloprotein Protocol (CYP450/Heme)
- Bonded Fe-S model is MANDATORY: heme_topology.py must be called after pdb2gmx
- Distance restraints fail catastrophically (Fe-SG drift to 15.8Å)
- Validated: cyp_md5 bonded → Fe-SG=2.32±0.03Å (100% coordination)
- Stepdown equilibration: 6 phases (EM→NVT→NPT1→NPT2(soft)→NPT3(metal)→MD)

### CGenFF Workflow
- User must upload MOL2 to cgenff.com → download .str → pass as ligand_str_files={'LIG': 'path.str'}
- Penalty score: <10 good, 10-50 fair, >50 needs QM refinement

### Windows Issues
- All open() MUST use encoding='utf-8' (GBK codec crashes on GROMACS topology files)
- SOBTOP cannot parse Windows backslash paths — use path.replace(os.sep, '/')

### Analysis Sequence
- Always: gromacs_process_trajectory (PBC fix + rot/trans fit) BEFORE gromacs_analysis
- gromacs_comprehensive: one-click full analysis (RMSD+RMSF+Rg+coordination+animation)

### RTX 5070 Performance Reference
- Metalloprotein+ligand: ~133 ns/day (~48K atoms)
- Pure metalloprotein: ~142 ns/day
- Pure protein: ~343 ns/day
"""


# =============================================================================
# Molecular Docking Specialist
# =============================================================================

class DockingAgent(SpecialistAgent):
    name = "docking"
    description = "Molecular docking expert — Vina, SmartDocking, PLIP interaction analysis"
    skill_categories = ["docking", "visualization"]
    system_addendum = """## Molecular Docking Domain Rules

### SmartDocking Pipeline
- Input: protein PDB + ligand SDF → auto-prepares PDBQT → Vina docking → report
- Vina binary: E:/computing bio/MDS-gmx/Docking/vina/vina.exe
- Output: <pdb_dir>/<stem>_docking/ (report.md, grid.json, best_pose.sdf)
- Site strategy: cofactor/ligand coords → HEME → protein center (warn on fallback)

### PLIP Analysis
- ALWAYS run plip_analyze after docking to get interaction profile
- Non-standard residues auto-fixed: ATOM→HETATM (_fix_pdb_hetatm)
- WSL PLIP 3.0.0 at /usr/local/bin/plip (fallback from native)
- Output: <stem>_plip/plip_report.md

### Visualization
- pymol_render (fast_visualize): primary, ~5s, Nature/JACS style
  - Black H-bond dashes, 3-letter labels (Asp10), orthoscopic projection
  - h_bond_cutoff_center=4.2 (matches PLIP's longer D-A distances)
  - NEVER cmd.remove("hydrogens") before H-bond display — breaks mode=2 detection
- Run pymol_render AFTER plip_analyze to show annotated interactions

### Workflow
1. search_pdb_structure or get_pdb_entry → PDB file
2. download_pubchem_sdf or generate_conformer → ligand SDF
3. smart_docking → best_pose.sdf + docking report
4. plip_analyze → interaction profile
5. pymol_render → visualization PNG

### Batch Docking
- For multiple ligands: batch_dock or run_mutant_docking
- LigBuilder V3 for de novo ligand generation (requires WSL)
"""


# =============================================================================
# Literature & Database Specialist
# =============================================================================

class LiteratureAgent(SpecialistAgent):
    name = "literature"
    description = "Literature and database expert — PubMed, UniProt, PDB search and synthesis"
    skill_categories = ["research", "brain"]
    system_addendum = """## Literature & Database Domain Rules

### Species Disambiguation (CRITICAL)
- CYP2A1 = rat (Rattus norvegicus), NOT human
- CYP2A6 = human (Homo sapiens)
- CYP2B4 = rabbit (Oryctolagus cuniculus)
- CYP2E1 = human BUT CYP2E1 also has rat homolog — always verify organism
- ALWAYS confirm species before reporting UniProt/PDB IDs

### Research Workflow
1. get_protein_profile → one-shot composite (UniProt + PDB + function)
2. search_pubmed → primary literature (supports MeSH terms + Boolean)
3. get_interpro_domains → domain architecture
4. search_pdb_structure → available experimental structures

### Cross-lingual Query Translation
- User may query in Chinese — translate enzyme names before PubMed/UniProt
- Common: P450→CYP450, 辅酶→cofactor, 底物→substrate, 抑制剂→inhibitor
- Use _translate_query() from context module for standardized terms

### GraphRAG Integration
- search_knowledge (graphrag_query) for local KG overlay
- BeTaThink KG path is configured externally (for example, via BETATHINK_KG_ROOT)
- Combine GraphRAG + PubMed for comprehensive coverage

### Citation Quality
- Prefer PubMed over general web search for mechanistic claims
- Report PMID, DOI, and journal for each key citation
- Flag if only preprint/preprint evidence is available
"""


# =============================================================================
# Chemistry Specialist
# =============================================================================

class ChemAgent(SpecialistAgent):
    name = "chemistry"
    description = "Computational chemistry expert — RDKit, PubChem, SMILES, conformers, QM"
    skill_categories = ["chemistry", "qm"]
    system_addendum = """## Computational Chemistry Domain Rules

### Ligand Preparation Pipeline
1. SMILES string → generate_conformer (RDKit ETKDGv3 + MMFF94s) → SDF
2. PubChem name/CID → download_pubchem_sdf → SDF
3. Always check: mol.GetNumHeavyAtoms() > 0 before conformer generation
4. 3D geometry quality: RMSD between generated conformers should be >1Å for diversity

### Format Conversions
- SDF → MOL2: OpenBabel (needed for CGenFF parameterization)
- SMILES → PDB: RDKit AllChem.EmbedMolecule → Chem.MolToPDBFile
- Always embed hydrogens before exporting: AllChem.AddHs(mol)

### QM Engine (PySCF, WSL)
- DFT optimization: B3LYP/6-31G* for drug-like molecules
- Single-point energy: for binding energy decomposition
- TS search: requires initial guess structure (interpolation between reactant/product)
- PySCF runs in WSL environment

### Drug-likeness
- Lipinski RO5: MW<500, logP<5, HBD<5, HBA<10
- PAINS filter: check before synthesis recommendation
- Toxicophore screening: avoid reactive warheads unless intentional covalent drug

### SMILES Handling
- Canonical SMILES from RDKit before storage/search
- Stereochemistry: always preserve @/@@, /\\/ bond notation
- Salt stripping: use rdkit.Chem.SaltRemover before property calculation
"""


# =============================================================================
# Structure Specialist
# =============================================================================

class StructureAgent(SpecialistAgent):
    name = "structure"
    description = "Protein structure expert — PDB prep, mutagenesis, structural analysis"
    skill_categories = ["structure", "mutagenesis", "plm"]
    system_addendum = """## Protein Structure Domain Rules

### PDB Preparation
- Always check: missing residues, insertion codes, alternate conformations, chain breaks
- HEME proteins: insert TER between protein ATOM records and HEME HETATM records
- Non-standard residues: rename to HETATM for PLIP, keep ATOM for GROMACS pdb2gmx
- Water/ions: keep if important for active site; remove for docking prep

### Mutagenesis Protocol
- Format: single-letter code (V21L preferred) — 3-letter also supported (VAL21LEU)
- Tool: mutate_protein (PyMOL/BioPython backend)
- Post-mutation: always minimize (GROMACS EM or PyMOL minimize) before docking/MD
- Multi-site: run mutate_protein sequentially, one mutation at a time

### ESM-2 Saturation Scan
- esm2_mutation_scan: predicts ΔΔG for all 20 AA at specified positions
- Requires: py310_DPLM2 conda env (GPU preferred)
- Output: per-position fitness landscape (heatmap + top-N variants)
- Use for: identifying beneficial mutation positions before wet-lab validation

### Structure Comparison
- compare_structures: RMSD alignment, per-residue RMSF overlay
- Use ChimeraX for multi-chain/multi-model visualization
- Structural superposition: align on Cα backbone atoms (backbone RMSD)

### AlphaFold Structures
- get_alphafold_prediction: download AF2 predicted structure
- Quality check pLDDT scores: >90 = high, 70-90 = medium, <70 = low confidence
- Refine low-confidence regions with MD before use in docking
"""


# =============================================================================
# Memory & Knowledge Specialist
# =============================================================================

class MemoryAgent(SpecialistAgent):
    name = "memory"
    description = "Knowledge and memory manager — session context, KG overlay, pattern learning"
    skill_categories = ["brain"]
    system_addendum = """## Memory & Knowledge Management Rules

### Memory Layers
- ProjectMemory (~/.zhuangzhou/project_memory.md): plain text, persist key findings here
- KnowledgeStore (~/.zhuangzhou/knowledge/): mutation outcomes, error patterns (auto-decay 30d)
- LocalKGEngine: session-level graph overlay (triples), exported to GraphRAG monthly
- ExecutionHistory (~/.zhuangzhou/execution/): full skill call log with outcomes

### Knowledge Recording
- After any successful workflow: summarize to ProjectMemory
- After mutation analysis: record to KnowledgeStore with activity outcome
- After MD validation: record Fe-SG distance, RMSD, coordination statistics
- After failed skill: record error pattern with root cause for future skip

### Context Management
- ContextBuilder.gather(): workspace intel + project_memory + session state + errors
- Inject workspace intelligence at session start (PDB/SDF/XTC presence → stage hints)
- History compression at 30 messages (structured summary preserving paths/errors)

### Pattern Memory
- skill_patterns.json: tracks chain success rates per workflow type
- _record_chain_to_memory(): auto-records after each skill chain execution
- Use success_rate threshold >0.7 before auto-suggesting a chain

### Graph Overlay
- /graphrag rebuild-overlay: exports local_graph.jsonl for parquet rebuild
- Monthly rebuild recommended for fresh GraphRAG search accuracy
"""


# Registry of all specialist agents
SPECIALIST_REGISTRY: dict = {
    "md": MDAgent,
    "docking": DockingAgent,
    "literature": LiteratureAgent,
    "chemistry": ChemAgent,
    "structure": StructureAgent,
    "memory": MemoryAgent,
}


def create_specialist(name: str, backend=None, api_key: str = None) -> SpecialistAgent:
    """Factory function to create a specialist by name."""
    cls = SPECIALIST_REGISTRY.get(name)
    if cls is None:
        raise ValueError(f"Unknown specialist '{name}'. Available: {list(SPECIALIST_REGISTRY)}")
    return cls(backend=backend, api_key=api_key)
