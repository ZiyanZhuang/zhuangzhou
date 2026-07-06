# Zhuangzhou

Zhuangzhou is a rational enzyme engineering and synthetic biology design system.
It combines deterministic bioinformatics, molecular modeling, metabolic modeling,
sequence design, and an optional natural-language control plane for DBTL-style
engineering workflows.

The core idea is simple: language models may help decide what to try next, but
the scientific outputs should come from explicit algorithms, structured data, and
reproducible tool calls.

## What Problem It Targets

Synthetic biology projects often fail between the idea and the bench because the
design evidence is scattered across incompatible layers:

- pathway selection is separated from enzyme feasibility;
- enzyme mutations are proposed without structural or MD context;
- expression cassettes are selected without burden or host constraints;
- codon optimization is performed without pathway-level design intent;
- wet-lab design packages are manually assembled from notebooks, scripts, and
  fragmented database searches.

Zhuangzhou aims to make these steps part of one traceable computational design
loop for enzyme and pathway engineering.

Representative questions it is designed to support:

- Which pathway should be engineered for a target metabolite in E. coli or yeast?
- Which enzymes are likely bottlenecks, toxic nodes, or poor expression targets?
- Which active-site or stability mutations are worth testing first?
- Which promoter, RBS/Kozak, terminator, and integration strategy match the
  desired expression balance?
- How should a protein sequence be codon-optimized for a production chassis?
- What evidence package should be handed to the experimental team?

## System Scope

Zhuangzhou is organized as a modular computational stack rather than a single
model wrapper.

| Layer | Modules | Role |
| --- | --- | --- |
| Natural language control | `nlp`, `agents`, `commands`, `ui` | Optional REPL, routing, planning, specialist prompts, CLI |
| Protein and enzyme design | `mutation_enricher`, `mutagenesis`, `molecularconformer`, `enzyme_specificity` | Mutation annotation, structural context, conformer generation, enzyme specificity checks |
| Molecular modeling | `smartdocking`, `gromacs_md`, `amber_md`, `CYPamber`, `nonstandard_res`, `qm_engine` | Docking, MD, CYP/heme workflows, non-standard residue handling, QM preflight and analysis |
| Synthetic biology | `synbio`, `codon_opt`, `workflow_options` | Parts selection, CRISPR guide design, codon optimization, expression balance, design packaging |
| Metabolic reasoning | `metakg`, `brain`, `huizi` | GEM/FBA bridge, GraphRAG/knowledge context, session-aware protein context |
| Reporting and memory | `report`, `memory`, `workflow` | Structured reports, project memory, reusable workflow templates |

## Synthetic Biology Capabilities

Zhuangzhou currently focuses on practical early-stage design support for enzyme
and pathway engineering.

### Pathway and Host Design

- Genome-scale model bridge for E. coli and S. cerevisiae workflows.
- Flux balance analysis, flux variability analysis, gene essentiality, bottleneck
  search, and production-envelope style reasoning.
- Pathway search and substrate/reaction lookup for connecting target products to
  feasible host interventions.

### Genetic Part Selection

- Curated promoter, RBS/Kozak, terminator, and integration-site libraries.
- Expression-strength matching for balanced pathway assembly.
- Separate bacterial and yeast part logic where expression control differs.
- Design package generation for handing off a proposed construct strategy.

### CRISPR and Perturbation Design

- SpCas9 guide search with NGG PAM handling.
- Knockout and CRISPRi-oriented design helpers.
- HDR template design utilities for edit planning.

### Codon Optimization

- Komagataella phaffii-oriented codon optimization engine.
- DNAchisel backend when installed, with a frequency fallback.
- GC-content constraints, translation enforcement, and avoid-pattern hooks.

### Enzyme Engineering

- Mutation enrichment with secondary structure, RSA, active-site distance,
  channel proximity, domain annotations, and literature evidence.
- Docking, MD, and QM modules for ranking mutations beyond sequence-only
  heuristics.
- CYP/heme-aware Amber and GROMACS support for metalloprotein engineering.

## Technical Highlights

- **Deterministic skills first.** Tool outputs are computed by Python modules,
  database APIs, simulation engines, or structured models, not invented as text.
- **Pluggable LLM layer.** Claude, OpenAI-compatible APIs, DeepSeek-compatible
  endpoints, and local Ollama-style backends are treated as optional planners.
- **Environment isolation.** Native tools, WSL tools, conda environments, and
  remote SSH compute can be configured independently.
- **Privacy-conscious configuration.** API keys, host paths, SSH settings, and
  heavyweight model files are externalized and ignored by Git.
- **Reproducible packaging.** Source distributions exclude caches, logs, local
  credentials, private storage exports, and large checkpoint files.
- **Structured reporting.** Design outputs can be converted into Word-style
  reports and reusable project memory records.

## Installation

Requires Python 3.10 or newer.

```bash
git clone https://github.com/ZiyanZhuang/zhuangzhou.git
cd zhuangzhou
python -m pip install -e .
```

Optional feature groups:

```bash
python -m pip install -e ".[nlp]"
python -m pip install -e ".[chem]"
python -m pip install -e ".[codon]"
python -m pip install -e ".[viz]"
python -m pip install -e ".[all]"
```

Some workflows require external scientific software such as GROMACS, AmberTools,
PyMOL, ChimeraX, OpenBabel, AutoDock Vina, RDKit, or WSL-side tools. These are
configured through environment variables or local config files, not committed
repository paths.

## Configuration

Common environment variables:

| Variable | Purpose |
| --- | --- |
| `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `DEEPSEEK_API_KEY` | Optional LLM backends |
| `GMX_BIN`, `GMXLIB` | GROMACS executable and topology directory |
| `VINA_BIN` | AutoDock Vina executable |
| `PYMOL_PYTHON`, `CHIMERAX_BIN` | Visualization backends |
| `SOBTOP_DIR`, `OBABEL_EXE`, `FFMPEG_BIN` | Ligand preparation and media tools |
| `ZHUANGZHOU_GRAPHRAG_ROOT` | Optional local GraphRAG data root |
| `BETATHINK_KG_ROOT` | Optional local knowledge-graph root |

Example config files are provided, for example:

```text
envs/ssh_compute.ini.example
gromacs_md/gromacs_md.ini.example
```

Do not commit real SSH, API, host, or private data files.

## Usage

CLI entry points:

```bash
zhuangzhou
zz
zz-cli
zz-chat
```

Python examples:

```python
from zhuangzhou.synbio.parts import PartSelector

parts = PartSelector.select_promoter(
    organism="ecoli",
    strength=0.6,
    inducible=False,
    n_candidates=3,
)
print(parts)
```

```python
from zhuangzhou.codon_opt.optimizer import optimize_codon

result = optimize_codon(
    protein_seq="MSTNPKPQRKTKRNTNRRPQDVKFPGG",
    backend="frequency",
)
print(result["dna_sequence"])
```

```python
from zhuangzhou.mutation_enricher import enrich_mutations

enriched = enrich_mutations(
    mutations=[{"position": 57, "from": "H", "to": "A"}],
    pdb_path="protein.pdb",
    uniprot_id="P00000",
    active_site={49, 57, 71},
    catalytic={57},
)
```

## Example DBTL Workflow

1. Define a target molecule and host chassis.
2. Search candidate pathways and host reactions with MetaKG/FBA tools.
3. Identify bottleneck enzymes and possible knockout or CRISPRi interventions.
4. Retrieve protein sequence, structure, domains, and literature context.
5. Generate mutation candidates and enrich them with structure and active-site
   features.
6. Run docking, MD, or QM checks for high-risk enzyme variants.
7. Select expression parts and codon-optimize the final sequence.
8. Export a design package and report for experimental validation.

## Current Status

Zhuangzhou is a research engineering project. Many modules are usable as Python
libraries today, while full end-to-end workflows depend on locally installed
scientific software and configured data resources.

Validated before public release:

- local source compile pass for project modules;
- targeted regression tests for CLI startup, tool fallback, and parallel skill
  execution;
- source distribution audit excluding secrets, caches, logs, storage exports,
  and large model checkpoints.

## Roadmap

- Soft DBTL loop with experiment-result ingestion and next-round design ranking.
- Expanded chassis support, including Yarrowia lipolytica and Komagataella
  phaffii production workflows.
- SaCas9 and broader CRISPR nuclease support.
- Stronger CAI and expression-burden models for multi-gene pathways.
- Public example notebooks with small, dependency-light demonstration datasets.
- CI checks for privacy scanning, package build, and core regression tests.

## Repository Notes

Large model weights, private SSH configs, local storage exports, logs, and cache
folders are intentionally ignored. The repository contains source code, templates,
small dictionaries, and example configs needed to reproduce the software layout.

## License

MIT. See project metadata for package details.
