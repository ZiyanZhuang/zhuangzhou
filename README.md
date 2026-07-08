# Zhuangzhou

**English** | [简体中文](README.zh-CN.md)

[![License: MIT](https://img.shields.io/badge/license-MIT-3F7A63.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-3F7A63.svg)](#installation)

**From biological intent to executable DBTL design.**
[Project showcase](docs/index.html) · [Wiki](../../wiki) · [Issues](../../issues)

Zhuangzhou is an AI Scientist Workbench for synthetic biology, enzyme
engineering, and computational DBTL design. It turns a biological intent such as
"improve this enzyme", "build this pathway", or "prepare this construct for a
host chassis" into an executable, auditable design workflow.

The product direction is a specialized Claude Science for bioengineering: a
domain agent that combines language-model planning with real scientific tools,
structured biological knowledge, pathway reasoning, molecular simulation,
sequence design, and experiment-ready reporting.

The principle is simple: the AI proposes and coordinates; Zhuangzhou computes,
checks, records, and packages. Scientific outputs should come from explicit
algorithms, structured data, database calls, and reproducible tool execution, not
from unsupported text generation.

## Table of Contents

- [Product Positioning](#product-positioning)
- [What It Tries To Solve](#what-it-tries-to-solve)
- [AI-Native DBTL Workflow](#ai-native-dbtl-workflow)
- [System Scope](#system-scope)
- [Agent Architecture](#agent-architecture)
- [Synthetic Biology Capabilities](#synthetic-biology-capabilities)
- [Technical Highlights](#technical-highlights)
- [Installation](#installation)
- [Usage](#usage)
- [Current Status](#current-status)
- [Roadmap](#roadmap)
- [Positioning Summary](#positioning-summary)
- [License](#license)

## Product Positioning

Zhuangzhou is not a general chatbot and not just a folder of bioinformatics
scripts. It is designed as a domain-specific AI scientist for the DBTL loop:

```text
User intent
  -> AI task planning
  -> tool-routed computational design
  -> evidence-ranked variants / pathways / constructs
  -> report and experimental handoff
  -> test-result memory for the next design round
```

The long-term product goal is to make synthetic biology design feel like a
scientific operating system:

- **Design:** reason over pathways, enzymes, structures, parts, and host context.
- **Build:** generate construct-level suggestions, codon-optimized sequences,
  CRISPR edits, and expression-control choices.
- **Test:** organize simulation outputs, assay hypotheses, and validation plans.
- **Learn:** preserve decisions, failures, experimental feedback, and genomics
  evidence as reusable project memory.

Zhuangzhou is therefore best described as:

> An AI-native DBTL workbench for enzyme and pathway engineering.

## What It Tries To Solve

Synthetic biology projects often fail between the idea and the bench because the
design evidence is scattered across incompatible layers:

- pathway selection is separated from enzyme feasibility;
- enzyme mutations are proposed without structural or MD context;
- expression cassettes are selected without burden or host constraints;
- codon optimization is performed without pathway-level design intent;
- wet-lab design packages are manually assembled from notebooks, scripts, and
  fragmented database searches.

Zhuangzhou aims to turn these fragments into one traceable design loop. The user
should not need to manually jump between literature search, sequence analysis,
structure inspection, docking, MD, FBA, codon optimization, CRISPR design,
plasmid-part selection, and report writing just to answer one engineering
question.

Representative questions it is designed to support:

- Which pathway should be engineered for a target metabolite in E. coli or yeast?
- Which enzymes are likely bottlenecks, toxic nodes, or poor expression targets?
- Which active-site or stability mutations are worth testing first?
- Which promoter, RBS/Kozak, terminator, and integration strategy match the
  desired expression balance?
- How should a protein sequence be codon-optimized for a production chassis?
- What evidence package should be handed to the experimental team?

## AI-Native DBTL Workflow

Zhuangzhou treats DBTL as an executable workflow rather than a slide-deck
framework.

| DBTL phase | AI role | Deterministic tools |
| --- | --- | --- |
| Design | Decompose the biological goal, choose analyses, rank uncertainty | pathway search, FBA/FVA, protein context, mutation enrichment, docking, MD, QM |
| Build | Translate designs into construct-level plans | promoter/RBS/Kozak selection, codon optimization, CRISPR guide design, integration strategy |
| Test | Prepare validation hypotheses and measurable outputs | assay checklist, simulation metrics, report generation, result schema |
| Learn | Preserve what worked, what failed, and why | project memory, execution history, knowledge context, next-round ranking |

This makes Zhuangzhou closer to a scientific workflow agent than a question
answering assistant. The AI layer is useful because it can decide which tools to
call and how to interpret intermediate results, but the workflow stays grounded
in inspectable outputs.

## System Scope

Zhuangzhou is organized as a modular computational stack rather than a single
model wrapper.

| Layer | Modules | Role |
| --- | --- | --- |
| AI control plane | `nlp`, `agents`, `commands`, `ui` | ReAct-style planning, tool routing, specialist prompts, CLI/REPL |
| Protein and enzyme design | `mutation_enricher`, `mutagenesis`, `molecularconformer`, `enzyme_specificity` | Mutation annotation, structural context, conformer generation, enzyme specificity checks |
| Molecular modeling | `smartdocking`, `gromacs_md`, `amber_md`, `CYPamber`, `cypforge`, `cypforge_core`, `nonstandard_res`, `qm_engine` | Docking, MD, CYP/heme workflows, CYPForge heme-ligand preparation gates, non-standard residue handling, QM preflight and analysis |
| Synthetic biology | `synbio`, `codon_opt`, `workflow_options` | Parts selection, Assembor yeast promoter/terminator candidates, CRISPR guide design, codon optimization, expression balance, design packaging |
| Metabolic reasoning | `metakg`, `brain`, `huizi` | GEM/FBA bridge, GraphRAG/knowledge context, session-aware protein context |
| Genomics computation | `genomics` | VCF/PLINK population-genomics command planning, sample-map validation, diversity/Fst/PCA/ADMIXTURE/TreeMix/IQ-TREE handoff |
| Reporting and memory | `report`, `memory`, `workflow` | Structured reports, project memory, reusable workflow templates |

## Agent Architecture

Zhuangzhou is built around a separation between reasoning and execution.

```text
Natural-language request
    |
    v
AI planner / router
    |
    v
Domain skill graph
    |
    +-- pathway and FBA tools
    +-- enzyme mutation and structure tools
    +-- docking / MD / QM tools
    +-- codon / CRISPR / parts tools
    +-- report and memory tools
    |
    v
Auditable design package
```

The system can be used without an LLM as a Python package, but the AI interface
is intended to make multi-step design work more natural: a scientist states the
biological objective, and the agent assembles the relevant computational path.

## Synthetic Biology Capabilities

Zhuangzhou currently focuses on practical early-stage design support for enzyme
and pathway engineering. Full detail for each area is in the
[Wiki: Capabilities](../../wiki/Capabilities) page.

- **Pathway & Host Design** — genome-scale model bridge for E. coli and
  S. cerevisiae, flux balance/variability analysis, gene essentiality, and
  pathway search connecting target products to feasible host interventions.
- **Genetic Part Selection** — curated promoter/RBS/Kozak/terminator libraries,
  an Assembor-style S. cerevisiae promoter/terminator table, and
  expression-strength matching for balanced assembly.
- **CRISPR & Perturbation Design** — SpCas9 guide search with NGG PAM handling,
  knockout/CRISPRi helpers, and HDR template design utilities.
- **Codon Optimization** — a Komagataella phaffii-oriented engine with a
  DNAchisel backend when installed, GC-content and avoid-pattern constraints.
- **Genomics Computation** — VCF/population-map input checks and auditable
  command plans for diversity, Fst, PCA, ADMIXTURE, TreeMix, and IQ-TREE.
- **Enzyme Engineering** — mutation enrichment (structure, RSA, active-site
  distance, literature evidence), docking/MD/QM ranking, and CYP/heme-aware
  Amber and GROMACS support via CYPForge.

## Technical Highlights

- **AI-native but tool-grounded.** The LLM plans and coordinates; scientific
  outputs are computed by Python modules, database APIs, simulation engines, or
  structured models.
- **DBTL memory direction.** The system is designed to keep design rationale,
  execution history, and future test results connected across rounds.
- **Pluggable model layer.** Claude, OpenAI-compatible APIs, DeepSeek-compatible
  endpoints, and local Ollama-style backends are treated as optional planners.
- **Environment isolation.** Native tools, WSL tools, conda environments, and
  remote SSH compute can be configured independently.
- **Genomics-aware DBTL feedback.** VCF/PLINK workflows are represented as
  inspectable command plans with sample validation, QC parameters, and methods
  text rather than opaque notebook fragments.
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
PyMOL, ChimeraX, OpenBabel, AutoDock Vina, RDKit, or WSL-side tools, configured
through environment variables or local config files. Full variable reference:
[Wiki: Installation & Configuration](../../wiki/Installation-and-Configuration).

## Usage

CLI entry points:

```bash
zhuangzhou
zz
zz-cli
zz-chat
```

An intended AI workflow looks like:

```text
zz-chat
> Improve production of a target compound in yeast.
> Compare pathway options, identify likely enzyme bottlenecks,
> propose expression parts, and prepare a DBTL design package.
```

Python example:

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

More examples (Assembor yeast parts, codon optimization, mutation enrichment,
CYPForge bridge, genomics workflows): [Wiki: CLI & Usage](../../wiki/CLI-and-Usage).

## Current Status

Zhuangzhou is a research engineering project. Many modules are usable as Python
libraries today, while full end-to-end workflows depend on locally installed
scientific software and configured data resources. See
[Wiki: Example Workflow & Roadmap](../../wiki/Example-Workflow-and-Roadmap) for
the full 9-step example workflow and pre-release validation notes.

## Roadmap

- AI DBTL loop with experiment-result ingestion and next-round design ranking.
- Expanded chassis support, including Yarrowia lipolytica and Komagataella
  phaffii production workflows.
- SaCas9 and broader CRISPR nuclease support.
- Stronger CAI and expression-burden models for multi-gene pathways.
- Public example notebooks with small, dependency-light demonstration datasets.
- CI checks for privacy scanning, package build, and core regression tests.

## Positioning Summary

Zhuangzhou is the beginning of a specialized AI science product for synthetic
biology. The aim is not to replace domain scientists, but to give them a
scientific agent that can connect reasoning, tools, data, and reports across the
DBTL cycle.

Short version:

> From biological intent to executable DBTL design.

## Repository Notes

Large model weights, private SSH configs, local storage exports, logs, and cache
folders are intentionally ignored. The repository contains source code, templates,
small dictionaries, and example configs needed to reproduce the software layout.

## License

MIT. See project metadata for package details.
