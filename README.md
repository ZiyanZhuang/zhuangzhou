<h1 align="center">Zhuangzhou · 庄周</h1>

<p align="center">An agent workbench for enzyme and metabolic engineering.</p>

<p align="center">
  <strong>English</strong> · <a href="README.zh-CN.md">简体中文</a>
</p>

<p align="center">
  <a href="https://github.com/ZiyanZhuang/zhuangzhou/actions/workflows/tests.yml"><img src="https://github.com/ZiyanZhuang/zhuangzhou/actions/workflows/tests.yml/badge.svg?branch=main" alt="Offline engineering checks" /></a>
  <a href="#quick-start"><img src="https://img.shields.io/badge/Python-3.10%2B-00A7B5?style=flat" alt="Python 3.10 or later" /></a>
  <a href="pyproject.toml"><img src="https://img.shields.io/badge/License-MIT-D7A923?style=flat" alt="License declared in package metadata: MIT" /></a>
</p>

<p align="center">
  <a href="#quick-start">Quick start</a> ·
  <a href="#usage">Usage</a> ·
  <a href="#modules-and-requirements">Modules</a> ·
  <a href="#architecture">Architecture</a> ·
  <a href="#validation">Validation</a>
</p>

<p align="center">
  <img src="zhuangzhou.png" width="960" alt="Original Zhuangzhou architecture illustration" />
</p>

<p align="center"><sub>Original architecture illustration. It describes the broader design direction; implementation and validation vary by module.</sub></p>

Zhuangzhou connects a terminal agent, scientific tools, workflow templates and
execution records. It supports work on protein variants, molecular modeling,
metabolic networks and sequence design. You can use individual Python modules
or configure a model backend to plan and call tools from the terminal.

This is research software. Available analyses depend on installed tools and data;
a completed tool call does not establish that a scientific conclusion is correct.

## Quick start

Use Python 3.10 or later in a dedicated environment.

```bash
git clone https://github.com/ZiyanZhuang/zhuangzhou.git
cd zhuangzhou
python -m venv .venv
```

Activate the environment for your shell:

| Shell | Command |
| --- | --- |
| Windows CMD | `.venv\Scripts\activate` |
| Windows PowerShell | `.venv\Scripts\Activate.ps1` |
| Linux / macOS | `source .venv/bin/activate` |

Install the core package and open the terminal:

```bash
python -m pip install -e .
zhuangzhou --version
zhuangzhou
```

Startup, `/help`, `/version`, `/status` and `/quit` work without a model backend.
`/status` shows the imported package path, which helps distinguish installations
when several Python environments are present.

### Optional dependencies

Install only the groups needed for your work:

| Group | Install | Purpose |
| --- | --- | --- |
| `nlp` | `python -m pip install -e ".[nlp]"` | Model adapters for the terminal agent |
| `chem` | `python -m pip install -e ".[chem]"` | RDKit and molecular preparation helpers |
| `codon` | `python -m pip install -e ".[codon]"` | DNAchisel and codon-table tooling |
| `viz` | `python -m pip install -e ".[viz]"` | Plotting dependencies |
| `report` | `python -m pip install -e ".[report]"` | Word report generation |
| `dev` | `python -m pip install -e ".[dev]"` | Tests, lint and package-build tools |

These extras install Python dependencies. GROMACS, AmberTools, Vina, PyMOL,
ChimeraX and other external programs need their own installations and configuration.
See [installation and configuration](../../wiki/Installation-and-Configuration)
for environment settings; the declared Python dependencies are in
[pyproject.toml](pyproject.toml).

## Usage

<p align="center">
  <img src="docs/assets/zhuangzhou-terminal.png" width="960" alt="Zhuangzhou terminal showing its gradient banner, version, startup time and command prompt" />
</p>

<p align="center"><sub>Rendered from the v0.8.6 terminal UI. The workspace path is an example; Started is the session start time. No model call or scientific simulation is shown.</sub></p>

### Terminal

| Entry point | Interface |
| --- | --- |
| `zhuangzhou` or `zz` | Main terminal: local commands and agent conversations |
| `zz-chat` | Dedicated agent REPL |
| `zz-cli` | Argument-based CLI; use `zz-cli --help` for its commands |

The main terminal accepts natural-language requests as well as commands:

| Command | Action |
| --- | --- |
| `/help` | List registered commands |
| `/status` | Show package, Python and dependency information |
| `/skills` | List registered scientific skills |
| `/model` | Select a configured model backend |
| `/confirm <skill>` | Execute one pending high-risk call with its reviewed parameters |
| `/resume` | Restore saved session context |
| `/quit` | Leave the terminal |

Agent commands need a configured backend. The adapters read `ANTHROPIC_API_KEY`,
`DEEPSEEK_API_KEY` or `OPENAI_API_KEY` for the corresponding cloud service;
Ollama uses a local service. Keep credentials in your environment or local
configuration. Backend selection is implemented in [nlp/backends.py](nlp/backends.py).

Before a high-risk tool runs, the terminal shows its name, credential-redacted
parameters and confirmation command. An interrupted or failed run retains its
failure state. `/resume` restores context; it does not restart an external simulation.

### Python

Individual modules can be used without a model. This example queries the bundled
parts library:

```python
from zhuangzhou.synbio.parts import PartSelector

parts = PartSelector.select_promoter(
    organism="ecoli",
    strength=0.6,
    inducible=False,
    n_candidates=3,
)
for part in parts:
    print(part["name"], part["strength"])
```

The strength values are relative library annotations, not measurements for a new
construct. More examples are in the [usage guide](../../wiki/CLI-and-Usage).

## Modules and requirements

| Area | Code | Scope and requirements |
| --- | --- | --- |
| Agent and tools | [nlp](nlp/), [skills](skills/), [agents](agents/) | Model adapters, routing, parameter validation, specialist agents and bounded tool execution. Model services are optional for library use. |
| Protein and enzyme analysis | [mutation_enricher](mutation_enricher/), [mutagenesis](mutagenesis/), [enzyme_specificity](enzyme_specificity/) | Mutation annotation and structural context; some analyses need protein structures, model weights or external tools. |
| Docking and molecular dynamics | [smartdocking](smartdocking/), [gromacs_md](gromacs_md/), [amber_md](amber_md/) | Preparation, execution wrappers and output analysis. Scientific engines and valid inputs must be supplied separately. |
| CYP, heme and QM | [cypforge_core](cypforge_core/), [CYPamber](CYPamber/), [nonstandard_res](nonstandard_res/), [qm_engine](qm_engine/) | Heme/ligand preparation, nonstandard residues and QM helpers. Chemical state, parameters and backend readiness need case-specific checks. |
| Synthetic biology | [synbio](synbio/), [codon_opt](codon_opt/) | Parts selection, sequence constraints and codon optimization. Library scores and design heuristics require experimental validation. |
| Metabolic models and knowledge | [metakg](metakg/), [brain](brain/) | Metabolic-model interfaces and knowledge retrieval. Availability depends on model files, databases and configured services. |
| Population genomics | [genomics](genomics/) | Input validation and command planning for VCF/PLINK and downstream tools. A generated plan is not a completed analysis. |
| Workflows and records | [workflow](workflow/), [memory](memory/), [report](report/) | JSON workflows, execution history, session checkpoints and reports. Some reporting paths need the `report` extra. |

The [capabilities guide](../../wiki/Capabilities) contains domain examples.
Metabolic models describe selected network constraints; they are not a complete
whole-cell simulator. Docking scores, model rankings and model-generated
explanations remain separate from experimental evidence.

## Architecture

The runtime follows a request through planning, tool execution, observation and
the next decision. Registered skills validate inputs; workflows resolve step
dependencies and propagate failures. Domain-specific review gates check supported
artifacts. Coverage depends on the workflow.

```text
Request → agent / router → registered skill or workflow → result checks
             ↑                                               ↓
             └──────── context, history and checkpoints ──────┘
```

- [Agent runtime](nlp/agent.py): tool calls, confirmation and iteration limits.
- [Workflow engine](workflow/engine.py): ordered steps, references and failure states.
- [Execution history](memory/execution_history.py): outcomes and interruption checkpoints.
- [Review gates](nlp/supervisor.py): deterministic checks and model-assisted review.

Concurrency is bounded. A timeout limits the caller's wait; it cannot forcibly
terminate arbitrary running Python code. Checkpoints preserve context and progress,
not a portable restart image of every external process. These boundaries are
documented in the [architecture findings](docs/engineering/ARCHITECTURE_FINDINGS.md).

## Validation

The [offline CI workflow](.github/workflows/tests.yml) runs on Ubuntu/Python 3.10
and Windows/Python 3.12. It includes unit and integration tests, failure injection,
built-wheel installation checks, critical lint and a CLI smoke test.

```bash
python -m pip install -e ".[nlp,report,dev]"
python -m pytest -q
```

Live API tests require explicit opt-in. Offline model responses are test fixtures;
they do not establish live-model performance or validate a scientific simulation.
Manually operated scientific runs are excluded from automatic offline collection.

| Record | Contents |
| --- | --- |
| [Engineering report](docs/engineering/FINAL_REPORT.md) | Fixes, executed checks and remaining limits |
| [Terminal UI audit](docs/engineering/UI_AUDIT.md) | CMD behavior, rendering checks and version/status semantics |
| [Execution log](docs/engineering/EXECUTION_LOG.md) | Reproductions, changes and verification evidence |

## Next steps

Current priorities are reproducible examples with public datasets, backend-specific
timeout and cancellation checks, broader scientific integration coverage, and
experimental-result ingestion for the next design round. Cross-scale enzyme and
metabolic optimization remains a development direction; it is not a validated
general-purpose design service.

## Project links

[Project page](docs/index.html) · [Wiki](../../wiki) ·
[Issues](https://github.com/ZiyanZhuang/zhuangzhou/issues)

Large weights, local service settings, credentials and generated simulation
outputs are excluded from the repository. The package metadata declares an
[MIT license](pyproject.toml).
