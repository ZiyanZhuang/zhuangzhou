# Zhuangzhou v0.8.5 — Technical Report
> *Rational Enzyme Engineering System*
> Ziyan Zhuang — Tianjin University | Qianyu Zhao — Xinjiang University | Zifan Zhao — Lanzhou University | Shengfei Wang — University of Chinese Academy of Sciences; Wuhan Institute of Virology, Chinese Academy of Sciences

---

## 0. Design Principle

> **Zhuangzhou IS the system. LLM IS a pluggable reasoning engine.**

The LLM decides *what* to do; Zhuangzhou decides *how* to do it safely, efficiently, and reproducibly.
All architectural decisions derive from this axiom.

Zhuangzhou is NOT a wrapper around an LLM that calls APIs.
It is a scientific computation system with a natural language control plane.

---

## 1. Version History

| Version | Key Milestone |
|---------|--------------|
| v0.6.0 | NLP interface, nonstandard residues, codon optimization |
| v0.7.0 | Word report generation, 7-layer memory system, conversation engine |
| v0.8.0 | RouterAgent (50-token routing), FixedMode, Supervisor gates added |
| **v0.8.5** | **Hardcoded routing removed; LLMClarifier replaces IntentAnalyzer; QM pre-flight guards; silent fallback bugs fixed; code review (17 issues resolved)** |
| v0.9.0 *(planned)* | Multi-agent parallelism, adaptive planning from execution outcomes |

---

## 2. Architecture Overview

### 2.1 System Layers

```
┌─────────────────────────────────────────────────────┐
│                   Natural Language                   │
│           User (Chinese / English)                   │
└────────────────────────┬────────────────────────────┘
                         │
┌────────────────────────▼────────────────────────────┐
│              NLP Control Plane (nlp/)                │
│  REPL → BioDesignMode? → RouterAgent → FixedMode    │
│                                      → PlanMode      │
│                                      → Direct Agent  │
└────────────────────────┬────────────────────────────┘
                         │
┌────────────────────────▼────────────────────────────┐
│            AgentExecutor (ReAct v5)                  │
│  Thought → Action → Observation (MAX_ITER=20)        │
│  Native tool_use OR manual ReAct format              │
└────────────────────────┬────────────────────────────┘
                         │
┌────────────────────────▼────────────────────────────┐
│              Skills Registry (150+ skills)           │
│  18 categories  │  preflight validation              │
│  environment requirements declared at registration   │
└──────┬──────────┬────────────────┬──────────────────┘
       │          │                │
  ┌────▼───┐ ┌───▼────┐  ┌────────▼──────┐
  │ native │ │  WSL   │  │  conda envs   │
  │ Vina   │ │ PLIP   │  │ graphrag      │
  │ GROMACS│ │PySCF   │  │ py310_DPLM2   │
  │ RDKit  │ │LigBuild│  │ ezs_env       │
  └────────┘ └────────┘  └───────────────┘
```

### 2.2 Token Economy

| Execution Path | LLM Tokens | Trigger Condition |
|---------------|-----------|-------------------|
| Fast-path (slash cmd, y/n, number) | **0** | Rule-based pattern match |
| RouterAgent only | **~50** | Any non-trivial query |
| Direct (router → direct) | **~50 + agent** | Simple factual / single skill |
| FixedMode | **~950–1100** | Standard docking/MD/mutation pipeline |
| PlanMode | **~2000–3000** | Complex multi-step, novel workflow |

FixedMode is **65–70% cheaper** than PlanMode for standard pipelines.

---

## 3. NLP System (`nlp/`, 17 modules)

### 3.1 REPL Entry Point (`repl.py`)

```
run_repl()
├── ZhuangzhouAgent.__init__()      # backend + registry
├── PlanMode.__init__()             # lazy: router + fixed mode
├── ZZInterface.render_logo()       # v0.8.5 banner
├── PromptSession (prompt_toolkit)  # history + tab completion
│
└── [loop]
    ├── /cmd → slash_manager.dispatch()
    ├── biodesign keywords → BioDesignMode.run()
    ├── everything else → PlanMode.process()
    └── → AgentExecutor.run_loop()
```

**Status Line** (shown after banner):
```
  Backend: deepseek  │  Tools: 150 loaded  │  Model: deepseek-chat  │  Plan: ON
```

### 3.2 RouterAgent (`router_agent.py`, 155 lines)

**Purpose**: Classify query intent in ~50 tokens. Gateway before expensive LLM calls.

**Decision Flow**:
```python
@dataclass
class RouteDecision:
    mode: str           # "direct" | "fixed" | "plan"
    pipeline: str|None  # "docking" | "md" | "mutation_scan"
    confidence: float   # 0.0–1.0
    reason: str
    from_fast_path: bool
```

**Fast-path rules (0 token)**:
- Starts with `/` → direct (slash command)
- Matches `^(yes|no|y|n|是|否|确认|取消)$` → direct
- Pure digits `^[1-9]$` → direct
- Query length < 8 chars → direct

**LLM-path (~50 tokens)**:
- Single call with constrained JSON schema
- Enum-constrained output (no hallucination)
- `confidence < 0.65` → downgrade `fixed` → `plan`

### 3.3 FixedMode (`fixed_mode.py`, 516 lines)

**Purpose**: Execute deterministic pipeline templates with research-informed parameters.
Anti-GIGO: PubMed research happens *before* any computation.

**Architecture**:
```
FixedMode.run(query, pipeline)
├── 1. ResearchPhase (~800 tokens)
│   ├── PubMedSearch × N papers (0 tokens, stdlib urllib)
│   └── LLM extraction → ResearchResult
│       {active_site_residues, cofactors, md_duration_ns,
│        forcefield_notes, literature_support_count}
│
├── 2. PipelineSupervisor.gate_post_research() (~100 tokens)
│   └── Validates literature support before pipeline starts
│
├── 3. Build steps (fixed topology, no LLM)
│   ├── docking:  get_pdb_entry → smart_docking → plip_analysis → plip_visualize
│   ├── md:       get_pdb_entry → metalloprotein_md|auto_md → gromacs_analysis
│   └── mutation: get_pdb_entry → esm2_mutation_scan
│
├── 4. TypeA gate before each step (0 token)
│   └── metal detection, additive filtering, chain selection
│
└── 5. Execute with {from_step_N} reference resolution
    └── Return {"success", "results", "gate_issues", "research"}
```

**Token Budget**: ~950–1100 total (research 800 + post-research gate 100 + overhead).

### 3.4 PipelineSupervisor (`supervisor.py`, 384 lines)

**Purpose**: 2-tier scientific validation. Block garbage inputs before expensive compute.

**TypeA — Rule-based (0 LLM tokens)**:

| Validator | What it checks |
|-----------|---------------|
| `check_pdb_has_metalloprotein(pdb)` | Detect HEME/Fe/Zn → auto-select `metalloprotein_md` |
| `check_crystallographic_additives(pdb)` | Exclude GOL/PEG/EDO from active site detection |
| `check_chain_completeness(pdb)` | Recommend best chain for multimers |
| `check_docking_output(dir)` | Sanity: poses exist, affinity in –20→0 kcal/mol range |

**TypeB — LLM-based (~100–200 tokens per gate)**:

| Gate | Validates |
|------|-----------|
| `gate_post_research` | Literature parameters are scientifically reasonable |
| `gate_pre_md` | Forcefield + duration match protein/ligand type |

**GateResult**:
```python
@dataclass
class GateResult:
    passed: bool
    issues: List[str]        # Specific scientific concerns
    corrections: Dict        # Proposed fixes (e.g. forcefield swap)
    blocking: bool           # True = stop execution
    confidence: float
    gate_name: str
```

### 3.5 PlanMode (`plan_mode.py`, 851 lines, v0.9.0)

**Key Change from v0.8.0**: `IntentAnalyzer` + `AlignmentTree` (hardcoded option trees)
→ **LLMClarifier** (LLM self-determines what to ask).

**Architecture**:
```
PlanMode.process(query)
├── _should_skip()           # 0 token, fast patterns
├── RouterAgent.route()      # ~50 token
├── _dispatch_fixed()        # if mode=="fixed"
│
└── [mode=="plan"]
    ├── LLMClarifier.run()   # dynamic Q&A, ≤3 rounds
    │   └── LLM reads domain knowledge → decides what's unclear → asks
    │
    ├── PlanGenerator.generate()  # maps entities → skill chain
    │   └── LLM reads skill catalog → selects appropriate tools
    │
    └── display_plan() + confirm()
        └── User approves → returns agent_message
```

**LLMClarifier v0.9.0 design**:
- `_DOMAIN_KNOWLEDGE` = factual reference only (species tables, MD timing, structure sources)
- **No** `分析类型→技能映射` section (was hardcoded skill routing — removed in v0.8.5)
- **No** ENTITY_OPTION_TREES (removed — LLM self-determines)
- LLM output: `{"goal": "自然语言描述", "resolved": [...], "questions": [...]}`

**Removed anti-patterns (v0.8.5)**:
- `_detect_domain_guidance()` in agent.py (keyword-triggered expert injection)
- `PlanGenerator` rules 4–5 (hardcoded `必须用metalloprotein_md`)
- `_fallback_plan` 5-step docking pipeline (replaced with error + skip)
- Broad `_SKIP_PATTERNS` (show/list/display/check/查询/搜索 removed)

### 3.6 AgentExecutor (`agent.py`, 949 lines, ReAct v5)

**10 Core Features**:

| # | Feature | Detail |
|---|---------|--------|
| 1 | Deferred Tools | Dynamic schema generation; `load_tools` meta-tool; MAX_TOOL_SCHEMAS=50 |
| 2 | Dual-Mode Reasoning | Native `tool_use` (Claude) OR manual ReAct format (Ollama, DeepSeek) |
| 3 | Expert Domain Context | Injected via system prompt, not keyword-triggered |
| 4 | Dynamic Pattern Memory | `KnowledgeStore.get_skill_failure_hints()` → avoids repeat mistakes |
| 5 | Workspace Awareness | Auto-detect project stage from CWD (PDB files, MD dirs, etc.) |
| 6 | Self-Correction | On error: LLM reads traceback → retries with fix |
| 7 | Pre-flight Validation | `skill.preflight(**params)` before execution (path checks, choices) |
| 8 | Timeout Guards | 10 min per skill via ThreadPoolExecutor |
| 9 | Context Compression | >30 msgs → summary + last 5 (species validation preserved) |
| 10 | Live UI | ZZStatus monitor, ToolSpinner, StreamRenderer |

**ReAct Loop** (MAX_ITERATIONS=20):
```
[User query]
    ↓
Thought: [LLM analyzes, selects next tool]
Action: {"name": "skill_name", "input": {...}}
    ↓ preflight() → execute() → timeout guard
Observation: {"result": ..., "success": true}
    ↓
[Repeat until Final Answer or max iterations]
```

### 3.7 LLM Backends (`backends.py`, 512 lines)

| Backend | Protocol | Tool Support | Best For |
|---------|----------|-------------|---------|
| Claude (Anthropic) | Native | ✓ `tool_use` | Highest quality, complex reasoning |
| DeepSeek | OpenAI-compat | ✓ `tool_calls` | Cost-effective, Chinese-optimized |
| OpenAI (GPT-4o) | Native | ✓ `tool_calls` | General purpose |
| Ollama | OpenAI-compat | Partial | Offline, local inference |

**Unified interface**:
```python
@dataclass
class LLMResponse:
    text: str
    tool_calls: List[ToolCallRequest]
    stop_reason: str    # "end_turn" | "tool_use" | "max_tokens"
    usage: dict         # {input_tokens, output_tokens}
```

**v0.8.5 fix**: `chat_stream` fallback loop eliminated (was `yield from self.chat_stream(...)` → infinite recursion risk; fixed to `tools=None`).

---

## 4. Skills System (`skills/`, 21 modules, 150+ functions)

### 4.1 Skill Registry Design

```python
@dataclass
class Skill:
    name: str
    description: str          # LLM reads this to decide when to use
    category: str             # 18 categories
    fn: Callable
    inputs: List[SkillParam]
    outputs: List[SkillParam]

    # Environment requirements (declared at registration)
    requires_gpu: bool = False
    requires_wsl: bool = False
    remote_capable: bool = False
    requires_network: bool = False

    # Planning metadata
    depends_on: List[str] = field(default_factory=list)
    when_to_use: str = ""      # LLM reads this for routing
    workflow_stage: str = ""   # fetch|prepare|compute|analyze|visualize|report
    output_type: str = ""      # pdb|docking|md|analysis|report|visualization

    def preflight(**kwargs) → Optional[dict]  # Validates before execution
    def execute(**kwargs) → dict              # Always returns dict
```

**Execution contract**: All skills return `{'error': '...', 'hint': '...'}` on failure — never raise.

### 4.2 Skill Categories (18)

| Category | Count | Key Skills |
|----------|-------|-----------|
| **research** | 21 | `search_pubmed`, `search_protein_databases`, `search_compound` |
| **biodbapi** | 18 | `uniprot_query`, `ncbi_query`, `download_pdb_file`, `alphafold_structure` |
| **simulation** | 14 | `metalloprotein_md`, `auto_md`, `gromacs_analysis`, `amber_md` |
| **visualization** | 11 | `pymol_view`, `plip_visualize` *(ChimeraX/ChatMol disabled in v0.8.5)* |
| **qm** | 11 | `qm_cluster`, `dft_optimization`, `ts_search`, `frequency_analysis` |
| **unimol** | 10 | `unimol_binding_affinity`, `generate_conformer`, `property_predict` |
| **report** | 10 | `generate_docking_report`, `generate_mutation_report`, `enriched_docking_report` |
| **metakg** | 12 | `cyp_metabolism_query`, `pathway_query`, `substrate_query` |
| **docking** | 6 | `smart_docking`, `vina_docking`, `batch_mutant_docking` |
| **synbio** | 8 | `codon_optimize`, `promoter_design`, `chassis_selection` |
| **plm** | 5 | `esm2_scan`, `esm2_load` |
| **structure** | 5 | `plip_analysis`, `cofactor_detection`, `pocket_detection` |
| **mutagenesis** | 2 | `mutate_protein` (PyMOL + GROMACS EM option) |
| **chemistry** | 3 | `pubchem_query`, `validate_smiles`, `molecular_properties` |
| **enzyme_specificity** | 3 | EzReac GNN inference |
| **ligbuilder** | 4 | `ligbuilder_design`, `ligbuilder_cavity`, `ligbuilder_inhibitor` |
| **nonstandard** | 4 | Non-standard AA parameterization |
| **brain_skills** | 7 | `graphrag_query`, `document_upload`, `literature_search` |

### 4.3 v0.8.5 Bug Fixes (Skills Layer)

| Finding | File | Fix Applied |
|---------|------|------------|
| Report skills in `category='visualization'` | `skills/report.py`, `skills/unified.py` | → `category='report'` (7 skills) |
| Silent Vina fallback in `unimol_binding_affinity` | `skills/unimol.py:60–70` | Removed; now returns explicit error dict |
| `smart_docking` missing `resolution_method` in outputs | `skills/docking.py` | Added output param |
| ESM-2 GPU mode no pre-check | `skills/unified.py:224` | Added `Dplm2Sandbox` import guard |
| `pocket_detection` bare subprocess | `skills/structure.py` | → `WSLSandbox().run()` with fallback |
| `esm2_scan` bare subprocess | `skills/plm.py` | → `WSLSandbox().run()` |
| QM functions: no bridge check | `skills/qm.py` (9 functions) | Added `_check_qm_engine_available()` guard |
| Duplicate choices validation | `skills/registry.py` | Removed dead code in `execute()` |
| `str \| None` syntax (Python 3.9) | `smartdocking/pipeline.py` | → String annotation |

### 4.4 Disabled Skills (v0.8.5)

ChimeraX and ChatMol visualization temporarily disabled pending Windows offscreen rendering solution:

```python
# skills/visualization.py
_DISABLED_VIZ_SKILLS: frozenset = frozenset({
    'chimerax_plip', 'chimerax_render', 'chimerax_trajectory',
    'chimerax_defattr', 'chimerax_comparison', 'chimerax_porcupine',
    'chimerax_morph', 'chatmol_viz',
})
```

Active visualization: `pymol_view` (fast_render.py, ~5s, Nature/JACS style).

---

## 5. Environment Isolation (`envs/`)

### 5.1 Environments

| Environment | Conda/Runtime | Key Tools |
|-------------|---------------|-----------|
| `graphrag` | `<GRAPHRAG_CONDA_ENV>` | pandas, networkx, AutoGraphRAG parquet |
| `py310_DPLM2` | `<DPLM2_CONDA_ENV>` | UniMol, ESM-2, PyTorch/GPU |
| `ezs_env` | `<EZS_CONDA_ENV>` | EzReac GNN (enzyme specificity) |
| `WSL` | Windows Subsystem for Linux | PLIP 3.0.0, PySCF, LigBuilder V3, fpocket |
| `native` | System Python | Vina, GROMACS, RDKit, dnachisel |

### 5.2 WSLSandbox (architectural standard)

All cross-environment calls go through `envs/sandbox.py`:

```python
class WSLSandbox:
    def run(cmd: str, timeout: int = 300) -> dict:
        # Manages WSL subprocess, captures stdout/stderr, returns dict
        # Always returns {'returncode', 'stdout', 'stderr'}
```

**v0.8.5 enforcement**: All bare `subprocess(['wsl', ...])` calls in skills layer
now use `WSLSandbox().run()` with subprocess fallback if import fails.

### 5.3 Remote SSH Compute

Config at `~/.zhuangzhou/ssh_compute.ini`.
`RemoteJobManager` handles submit/poll/retrieve for HPC nodes.

---

## 6. Memory System (7 Layers)

### 6.1 Layer Architecture

```
Layer 1: Journal  (~/.zhuangzhou/journal/)
  ├── Append-only JSON event log
  └── Every tool call, result, error recorded

Layer 2: Context  (~/.zhuangzhou/context/)
  └── Session state: CWD, current_protein, mutations_tried

Layer 3: Knowledge  (~/.zhuangzhou/knowledge/)
  ├── skill_patterns.json     — success rate, typical params per skill
  ├── error_patterns.json     — common failures + solutions
  └── mutation_insights.json  — species, role, effect patterns

Layer 4: Archive  (~/.zhuangzhou/archive/)
  └── Full interaction recordings (per session, compressed)

Layer 5: ExecutionHistory  (~/.zhuangzhou/execution/)
  ├── Persistent JSONL skill log (all executions)
  └── Task checkpoints (resume interrupted pipelines)

Layer 6: WorkspaceIntelligence
  └── Auto-detects project stage from CWD:
      PDB files → early; MD dirs → mid; reports → late

Layer 7: ProjectMemory  (~/.zhuangzhou/project_memory.md)
  └── Human-readable accumulating knowledge base
```

**Design**: No vector DB. Plain text, signal-over-noise, LLM-consumable directly.

**v0.8.5 fix**: `execution_history.py` — corrupt JSONL lines now log warning instead of silently skipping.

---

## 7. Molecular Dynamics — Critical Knowledge

### 7.1 GROMACS MD (`gromacs_md/`, 11 files)

**MANDATORY for CYP450**: Bonded Fe-S model.

| Issue | Wrong approach | Correct approach | Result |
|-------|---------------|-----------------|--------|
| Fe-SG coordination | Distance restraints alone | **Bonded topology** (proper bond terms) | 100% stability @ 0.232nm ± 0.003nm |
| PBC errors at NVT | — | `trjconv -pbc whole` after EM | No LINCS blowup |
| NPT equilibration | Parrinello-Rahman | **C-rescale barostat** | Stable P coupling |
| Force field mixing | CGenFF+GAFF | **CGenFF+CGenFF** or **AMBER+GAFF** | Compatible |
| File encoding (Windows) | `open(file)` | `open(file, encoding='utf-8')` | No UnicodeDecodeError |

**Validated runs**:
- `cyp_md5` (bonded Fe-SG): 0.232nm ± 0.003nm, 100% coordination
- `camphor_md9`: RMSD=1.04Å, ligand RMSD=2.63Å, Rg=2.171nm

### 7.2 AMBER MD (`amber_md/`, 10 files)

- Protocol: Shahrokh for CYP450+Heme
- Ligand prep: RESP charges + GAFF
- Files: `pipeline.py`, `general_pipeline.py`, `prepare.py`, `ligand_prep.py`

### 7.3 SmartDocking (`smartdocking/`, 6 files)

**Auto-ligand resolution chain**:
```
Input: protein_query (PDB file, PDB ID, or path)
       ligand_query  (SDF path, SMILES, name, CID)
    ↓
1. If protein is 4-char ID → download_pdb_file()
2. If ligand has SMILES chars → generate_conformer()
3. If ligand is name/CID → PubChem download → conformer
    ↓
smart_docking()
    ↓ PreProcessor → site detection → Vina → complex
```

**Active site strategy** (priority order):
1. Ligand coordinates → binding residues
2. HEME if present → centered at Fe
3. Protein geometric center (fallback)

**Critical**: Exclude crystallographic additives (GOL, PEG, EDO, MPD, GOL, PO4) from site detection.

---

## 8. Quantum Mechanics (`qm_engine/`, 5 files)

### 8.1 Supported Backends

| Backend | Mode | Notes |
|---------|------|-------|
| PySCF | CPU (default) | RTX 5070 (SM 12.0) not supported by gpu4pyscf |
| PySCF | GPU (via gpu4pyscf) | Requires older CUDA SM |
| Gaussian | WSL/Docker | Commercial |
| ORCA | WSL | Academic free |

### 8.2 Available Skills

| Skill | Function | Default Level |
|-------|----------|--------------|
| `qm_cluster` | Extract QM region from PDB | — |
| `dft_optimization` | Geometry optimization | B3LYP/def2-SVP |
| `ts_search` | Transition state (RS-P-RFO) | B3LYP/def2-SVP |
| `frequency_analysis` | Vibrational frequencies | Same as opt |
| `dft_single_point` | Single-point energy | B3LYP/def2-TZVP |
| `ts_pipeline` | PDB→cluster→opt→TS→freq→SP | B3LYP |
| `qm_energy_profile` | Reaction energy diagram | — |
| `qm_reaction_analysis` | Stationary points analysis | — |
| `qm_report` | QM report generation | — |

**Validated**: SN2 F⁻+CH₃F → TS barrier 8.1 kcal/mol (literature: ~9 kcal/mol, reasonable).

**v0.8.5 fix**: All 9 QM functions now call `_check_qm_engine_available()` before import.
No more ImportError at runtime — clear error dict returned instead.

---

## 9. Visualization

### 9.1 PyMOL (`pymol_viz/`, PRIMARY)

**Entry point**: `fast_render.py` — Nature/JACS publication style, ~5s.

**Key lessons**:
- `cmd.remove("hydrogens")` breaks H-bond detection → keep H atoms, only hide display
- `h_bond_cutoff_center = 4.2` (matches PLIP D–A distances)
- Black H-bond dashes, 3-letter residue labels
- Version: PyMOL 3.03 @ `<PYMOL_HOME>`

### 9.2 ChimeraX (`chimerax_viz/`, DISABLED in v0.8.5)

Reason: Windows has no OSMesa → `--offscreen` fails.
Currently requires GUI mode (`--cmd`), unsuitable for headless pipeline execution.

Planned: X virtual framebuffer via WSL for headless rendering.

**Lessons when re-enabled**:
- Quote all paths in .cxc scripts (spaces)
- `contacts` has no `dashes`/`radius` params → use `hbonds`
- Lighting: separate lines (`lighting soft` then `lighting ambientIntensity 0.45`)

---

## 10. Knowledge Brain (`brain/`)

**GraphRAG** (AutoGraphRAG parquet queries):
- BeTaThink Neo4j-based KG at `<BETATHINK_KG_ROOT>`
- English-only KG; Chinese users should use PubMed search first
- Query via `graphrag.py` → `connector.py` → parquet files
- Conda env: `graphrag`

**Research Skills** (preferred for Chinese input):
- `search_pubmed`: stdlib urllib, no external dependencies
- ResearchPhase in FixedMode extracts structured parameters from PubMed results

---

## 11. Report Generation (`report/`, v0.7.0+)

All reports output Word `.docx` (python-docx):

| Skill | Report Type |
|-------|------------|
| `generate_mutation_report` | Mutation analysis with structure table |
| `generate_docking_report` | Docking results with affinity table |
| `generate_md_report` | MD simulation with RMSD/RMSF plots |
| `generate_general_report` | Free-form sections (LLM-driven) |
| `full_pipeline_report` | End-to-end: PDB → mutate → dock → analyze → .docx |
| `enriched_docking_report` | + UniProt active site + KEGG + mechanistic hypothesis |

---

## 12. UI System (`ui/core.py`)

### 12.1 Color Palette (Cyan-Gold)

| Token | Hex | Use |
|-------|-----|-----|
| `CYAN` | `#00E5FF` | Primary: prompts, tool names, borders |
| `GOLD` | `#FFC107` | Accent: chevron, timing, highlights |
| `MID` | `#00D4E0` | Gradient midpoint, separators |
| `BG` | `#12151C` | Deep blue-black panel background |
| `GREEN` | `#66BB6A` | Success states |
| `LAVA` | `#FF8F00` | Warnings |
| `RED` | `#FF5252` | Errors |
| `DIM` | `#888888` | Secondary text |

### 12.2 Banner (v0.8.5 updates)

ASCII art: full `ZHUANGZHOU` logo with Cyan→Gold gradient.
Right side of ZHOU block:
```
   庄周 v0.8.5
   Router → Fixed|Plan|Direct
   Date: 2026-03-15
   Path: .../project
```

Tagline: `Rational Enzyme Engineering  |  Chemistry over consensus  |  LLM as engine, not oracle.`

### 12.3 Key UI Components

**ZZStatus** — live monitor (context manager):
```python
with ZZStatus(action="Reasoning") as monitor:
    monitor.update(tokens=150, action="search_pubmed", phase="act")
    # Auto-renders: ⠋ ◆ 12.3s  │  1.5k tokens 120/s  │  search_pubmed
```

Features: Braille spinner (10 frames, 80ms), smooth token interpolation (ease-out), phase icons.

**ToolSpinner** — per-tool spinner:
```
  ┃ ⠙ ACT  smart_docking  2.1s
  ┃ ✓ DONE smart_docking  45.2s
```

**StreamRenderer** — token-by-token LLM output:
- Rich Panel with Cyan border, Markdown rendering
- Header: `time 12.3s  │  token 2.5k (in:1.8k out:700)  │  ctx 45k/200k (22%)  │  model deepseek-chat`
- Footer: `mem journal+context+knowledge  │  msgs 8`

**ZZInterface.print_action()** — tool execution status:
```
  ┃ ▶ ACT   smart_docking
  ┃ ✓ DONE  smart_docking  45.2s
  ┃ ✗ FAIL  smart_docking
```

**ZZInterface.select_one()** — interactive vertical selection:
```
  ┃ Select protein species:
   ➤ Human (Homo sapiens)
     Mouse (Mus musculus)
     [Up/Down to select, Enter to confirm]
```

**Context bar** (shown before prompt):
```
  ctx 45k/200k (22%)  │  token 2.5k (in:1.8k out:700)  │  mem journal+context  │  msgs 8  │  model deepseek-chat
```

---

## 13. Code Quality — v0.8.5 Review Summary

### 13.1 Architecture Issues Resolved (17 total)

**HIGH severity (3 fixed)**:
1. Report skills miscategorized as `visualization` → `report` (7 skills)
2. `pocket_detection` bare WSL subprocess → `WSLSandbox`
3. `esm2_scan` bare WSL subprocess → `WSLSandbox`

**HIGH severity — Anti-patterns removed**:
4. `_detect_domain_guidance()` keyword-triggered injection (violates LLM reasoning)
5. `PlanGenerator` hardcoded `必须用metalloprotein_md` rule
6. `_fallback_plan` hardcoded 5-step docking sequence
7. Broad `_SKIP_PATTERNS` (show/list/display/check)

**MEDIUM severity (4 fixed)**:
8. `unimol_binding_affinity` silent Vina fallback → explicit error dict
9. `smart_docking` missing `resolution_method` output field
10. ESM-2 GPU mode: no dplm2 bridge pre-check
11. QM functions: 9 functions missing bridge availability check

**LOW severity (3 fixed)**:
12. Duplicate choices validation in `registry.execute()` (dead code after preflight)
13. `str | None` PEP 604 syntax → string annotation (Python 3.9 compat)
14. `backends.py chat_stream` fallback re-entry risk

**Code safety (3 fixed)**:
15. `_parse_json()` infinite loop guard (`max_tries = min(len(text), 200)`)
16. `_parse_manual_tool_calls()` nested JSON truncation (regex → raw_decode scan)
17. `_execute_skill_with_ui()` non-dict return not guarded (`isinstance` check added)

### 13.2 Remaining Known Issues

| Issue | Priority | Plan |
|-------|----------|------|
| ChimeraX offscreen rendering | MEDIUM | WSL X virtual framebuffer |
| `esm2_scan` sys.path injection | LOW | Module-level import instead |
| `chimerax_defattr` coupling to `gromacs_md.analysis` | LOW | Contained in disabled skill |
| `ligbuilder.py` WSLSandbox enforcement | LOW | Implementation layer handles it |

---

## 14. File Structure Summary

### NLP System (`nlp/`)

| File | Lines | Role |
|------|-------|------|
| `agent.py` | 949 | ReAct v5 engine + ZhuangzhouAgent |
| `backends.py` | 512 | Claude, DeepSeek, OpenAI, Ollama |
| `plan_mode.py` | 851 | LLMClarifier + PlanGenerator (v0.9.0) |
| `fixed_mode.py` | 516 | Template pipelines + ResearchPhase |
| `supervisor.py` | 384 | TypeA/TypeB validation gates |
| `biodesign_mode.py` | 771 | Metabolic engineering workflows |
| `router_agent.py` | 155 | 50-token intent router |
| `router.py` | 240 | Specialist lesson router |
| `slash.py` | 1100 | Slash command dispatcher |
| `repl.py` | 282 | Interactive REPL |
| `specialist.py` | 220 | Expert domain guidance |
| `context.py` | 117 | Session context manager |
| `completer.py` | 53 | Tab completion |
| `tools.py` | 116 | Skill schema generation |

### Computation Modules

| Directory | Files | Purpose |
|-----------|-------|---------|
| `gromacs_md/` | 11 | GROMACS MD pipeline (bonded Fe-SG, restraints, analysis) |
| `amber_md/` | 10 | AMBER MD pipeline (Shahrokh protocol) |
| `smartdocking/` | 6 | Smart docking (auto-ligand, site detection) |
| `qm_engine/` | 5 | DFT/QM (PySCF, Gaussian, ORCA) |
| `pymol_viz/` | 5 | PyMOL visualization (fast_render.py PRIMARY) |
| `chimerax_viz/` | 5 | ChimeraX (disabled, pending offscreen fix) |
| `plip_analysis/` | 3 | Protein-ligand interactions |
| `mutagenesis/` | 4 | Protein mutation (PyMOL + GROMACS EM) |
| `nonstandard_res/` | 8 | Non-standard AA parameterization |
| `codon_opt/` | 5 | Codon optimization |
| `ligbuilder_gen/` | 3 | LigBuilder V3 de novo design |
| `molecularconformer/` | 4 | 3D conformer generation (RDKit) |
| `metakg/` | 4 | Metabolic KG queries |
| `enzyme_specificity/` | 3 | EzReac GNN inference |

### Infrastructure

| Directory | Files | Purpose |
|-----------|-------|---------|
| `skills/` | 21 | Skill wrappers + registry |
| `envs/` | 4 | Sandbox isolation (conda, WSL, SSH) |
| `memory/` | 10 | 7-layer memory system |
| `brain/` | 4 | GraphRAG + BeTaThink KG |
| `ui/` | 2 | Rich console UI (Cyan-Gold theme) |
| `report/` | 5 | Word .docx generation |
| `commands/` | 10 | CLI commands |
| `agents/` | 5 | Multi-agent orchestration |
| `workflow/` | 2 | JSON pipeline templates |

---

## 15. Configuration & Key Paths

| Component | Path |
|-----------|------|
| Vina binary | `<VINA_BIN>` |
| GROMACS | `<GMX_BIN>` |
| PyMOL | `<PYMOL_HOME>` |
| ChimeraX 1.9 | `<CHIMERAX_BIN>` |
| LigBuilder V3 | `<LIGBUILDER_HOME>` |
| Uni-Mol | `<UNIMOL_HOME>` |
| BeTaThink KG | `<BETATHINK_KG_ROOT>` |
| Graphrag env | `<GRAPHRAG_CONDA_ENV>` |
| DPLM2 env | `<DPLM2_CONDA_ENV>` |
| Project memory | `~/.zhuangzhou/project_memory.md` |
| Session history | `~/.zhuangzhou/history.txt` |
| SSH compute config | `~/.zhuangzhou/ssh_compute.ini` |

---

## 16. Usage Reference

### CLI
```bash
# Interactive REPL (default backend auto-detected)
python -m zhuangzhou.nlp

# Specific backend
python -m zhuangzhou.nlp --backend deepseek

# One-shot
python -m zhuangzhou.nlp --oneshot "分析CYP3A4的V21L突变"
```

### Python API
```python
from zhuangzhou.nlp import chat, create_agent

# One-shot
result = chat("分析V21L突变对CYP3A4的影响")

# Multi-turn agent
agent = create_agent(backend='deepseek')
r1 = agent.chat("帮我对接咖啡因和CYP3A4")
r2 = agent.chat("再做频率分析")
```

### Slash Commands
| Command | Action |
|---------|--------|
| `/help` | Show all commands |
| `/init` | Scan environment, update ZHUANGZHOU.md |
| `/plan` | Toggle plan mode ON/OFF |
| `/status` | Show tool availability |
| `/skills` | List all registered skills |
| `/clear` | Clear conversation history |
| `/save` | Save session memory |
| `/reset` | Reset agent state |

---

*Report generated: 2026-03-15 | Zhuangzhou v0.8.5 | Tianjin / Xinjiang Universities*
