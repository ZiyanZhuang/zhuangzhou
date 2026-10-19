# Zhuangzhou v0.8.6 — Technical Report
> *Rational Enzyme Engineering & Metabolic Design System*
> Ziyan Zhuang — Tianjin University | Qianyu Zhao — Xinjiang University | Zifan Zhao — Lanzhou University | Shengfei Wang — University of Chinese Academy of Sciences; Wuhan Institute of Virology, Chinese Academy of Sciences
> Report date: 2026-03-17

---

## 0. Design Principle

> **Zhuangzhou IS the system. LLM IS a pluggable reasoning engine.**
> **Real algorithms, not LLM wrappers. Chemistry over consensus.**

The LLM decides *what* to do; Zhuangzhou decides *how* to do it safely, efficiently, and reproducibly.
All architectural decisions derive from this axiom. Every skill function is a **deterministic computation** —
numerical outputs (flux values, docking scores, mutation effects) are never LLM-generated text.

---

## 1. Version History

| Version | Key Milestone |
|---------|--------------|
| v0.6.0 | NLP interface, nonstandard residues, codon optimization |
| v0.7.0 | Word report generation, 7-layer memory system, conversation engine |
| v0.8.0 | RouterAgent, FixedMode, Supervisor gates, BioDesign 6-stage pipeline |
| v0.8.5 | LLMClarifier replaces IntentAnalyzer; QM pre-flight guards; 17 code review issues resolved |
| **v0.8.6** | **HuiZi context companion (v0.8.6); MetaKG COBRApy integration; SynBio engine; 11 critical bugs fixed; 98 skills** |
| v0.9.0 *(planned)* | Soft DBTL loop, SaCas9, real CAI, Y. lipolytica model, GraphRAG corpus loading |

---

## 2. Codebase Statistics (v0.8.6)

| Metric | Value |
|--------|-------|
| Python source files | 214 |
| Total lines of code | 79,142 |
| Registered skills | **98** across **14 categories** |
| Top-level modules | 24 |
| LLM backends supported | 4 (Claude, DeepSeek, OpenAI, Ollama) |
| Genome-scale models | 2 (iML1515 E. coli, iMM904 S. cerevisiae) |
| Supported chassis | E. coli, S. cerevisiae, (Y. lipolytica in v0.9.0) |

---

## 3. Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                    NATURAL LANGUAGE CONTROL PLANE                    │
│  nlp/repl.py → BioDesignMode → PlanMode → ZhuangzhouAgent          │
│  (ReAct loop, MAX_ITER=50, MAX_TOOL_SCHEMAS=50)                     │
└──────────────────────────┬──────────────────────────────────────────┘
                           │ tool calls
┌──────────────────────────▼──────────────────────────────────────────┐
│                         SKILL LAYER (98 skills)                      │
│  structure · docking · simulation · plm · chemistry · qm            │
│  metakg · synbio · research · report · brain · visualization         │
│  sequence · mutagenesis                                              │
└──────┬──────────────┬──────────────┬──────────────┬─────────────────┘
       │              │              │              │
┌──────▼───┐   ┌──────▼───┐   ┌─────▼────┐   ┌────▼──────┐
│ MetaKG   │   │  SynBio  │   │  HuiZi   │   │   Brain   │
│ (COBRApy)│   │ (CRISPR/ │   │ (session │   │(GraphRAG/ │
│ iML1515  │   │  Parts)  │   │ context) │   │ BeTaThink)│
│ iMM904   │   │          │   │          │   │           │
└──────────┘   └──────────┘   └──────────┘   └───────────┘
       │                             │
┌──────▼─────────────────────────────▼─────────────────────────────────┐
│                      ENVIRONMENT ISOLATION LAYER                      │
│  native (Vina/GROMACS/RDKit) · graphrag conda · py310_DPLM2 conda   │
│  WSL (PLIP/LigBuilder/PySCF) · ssh:<host> remote compute            │
└───────────────────────────────────────────────────────────────────────┘
```

---

## 4. Module Inventory

### 4.1 NLP Interface Layer (`zhuangzhou/nlp/`, v0.8.6)

| File | Description |
|------|-------------|
| `agent.py` | ZhuangzhouAgent v5 — ReAct engine with MAX_ITER=50, dual-mode (native tools + manual fallback), context summarization at 30 msgs, dependency graph reasoning |
| `backends.py` | LLM abstraction: Claude (tool_use native), DeepSeek (OpenAI-compat), OpenAI, Ollama. Unified `chat(system, messages, tools) -> LLMResponse` |
| `biodesign_mode.py` | 6-stage metabolic engineering pipeline: target→pathway mining→FBA→interference→parts→package. Intercepts metabolic engineering queries before NL agent |
| `plan_mode.py` | LLMClarifier: up to 3 clarification rounds before task execution. Fast-path for direct commands |
| `repl.py` | Interactive REPL (prompt_toolkit): slash commands, auto-complete, history, Plan+BioDesign integration |
| `router.py` | ToolRouter: category-based tool schema filtering (MAX_TOOL_SCHEMAS=50) |
| `slash.py` | Slash command manager: /help, /status, /context, /init, /plan |
| `backends.py` | 4 backends with unified `LLMResponse` dataclass |
| `specialist.py` | Domain-specialist prompt injection (MD/Docking/QM/Mutation) |
| `supervisor.py` | Multi-turn retry and self-correction gates |

### 4.2 HuiZi Context Companion (`zhuangzhou/huizi/`, v0.8.6)

| File | Description |
|------|-------------|
| `__init__.py` | HuiZi class: session context awareness, protein/ligand registration, report context builder |
| `context.py` | SessionContext + AnalysisTarget dataclass: tracks protein identity, sequence, ligand, analyses done, key findings |
| `protein_resolver.py` | Multi-source protein resolution: PDB file → UniProt → FASTA extraction |
| `structure_fetcher.py` | Auto-download from AFDB v4 (AlphaFold DB) or ESMFold; pLDDT quality tier |
| `query_translator.py` | Chinese scientific query → English keywords for database search |
| `literature_fetcher.py` | PubMed/Europe PMC literature fetch (default disabled, `fetch_literature=False`) |

**v0.8.6 improvements:**
- `get_report_context(fetch_literature=False)` — default changed from True to False (15s → 0.02s)
- `ensure_structure()` — auto-fetches AFDB structure when no local PDB available
- `get_system_prompt_block()` — compact context injection into agent system prompt
- 6 registered HuiZi skills in skill registry

### 4.3 MetaKG Module (`zhuangzhou/metakg/`, v0.8.6)

| File | Description |
|------|-------------|
| `bridge.py` | MetaKGBridge: loads BiGG GEM (COBRApy), BFS pathway finding, metabolite/reaction search, gene essentiality |
| `flux.py` | FluxAnalyzer: FBA, FVA, gene essentiality, production envelope, bottleneck identification |
| `config.py` | MODELS dict: iML1515 (E. coli) and iMM904 (S. cerevisiae) JSON paths |

**v0.8.6 improvements:**
- FBA growth_rate detection: multi-ID loop for cross-model biomass reactions (BIOMASS_Ec_iML1515, BIOMASS_SC5_notrace, etc.)
- `search_nodes()` None guard prevents AttributeError on empty queries
- Verified: E. coli FBA = 5.64 mmol/gDW/h, yeast FBA = 3.74 mmol/gDW/h

**10 registered MetaKG skills:**
`metakg_query`, `metakg_path`, `metakg_flux`, `metakg_fva`, `metakg_bottleneck`, `metakg_essentiality`, `metakg_compare`, `metakg_production_envelope`, `metakg_gene_knockout`, `metakg_substrates`

### 4.4 SynBio Module (`zhuangzhou/synbio/`, v0.8.6)

| File | Description |
|------|-------------|
| `parts.py` | PartSelector: promoter/RBS/terminator libraries for E. coli + yeast; expression balancing |
| `crispr.py` | SpCas9 guide finding (NGG PAM: `[ACGT]{20}[ACGT]GG`), HDR template design, CRISPRi design |
| `config.py` | Part catalog configuration |

**v0.8.6 fixes:**
- SpCas9 PAM regex corrected: `[ACGT]{20}[ACGT]GG` (was missing N position)
- BBa_B0033 RBS sequence corrected: `TCAAAGAGGAAAC` (was duplicate of B0032)

**7 registered SynBio skills:**
`synbio_parts_select`, `synbio_crispr_design`, `synbio_expression_balance`, `synbio_codon_optimize`, `synbio_burden_assess`, `synbio_design_package`, `synbio_knockout_strategy`

### 4.5 Brain / Knowledge Backend (`zhuangzhou/brain/`, v0.8.6)

| File | Description |
|------|-------------|
| `graphrag.py` | GraphRAGBridge: AutoGraphRAG parquet queries + Chinese query translation. BFS KG traversal. Requires `graphrag` conda env |
| `api.py` | BeTaThinkClient: HTTP REST to BeTaThink PostgreSQL/Neo4j/Milvus backend |
| `connector.py` | BrainConnector: unified online/offline mode with health monitoring |

**5 registered Brain skills:**
`graphrag_query`, `graphrag_entity`, `graphrag_path`, `knowledge_search`, `graphrag_report`

### 4.6 Report Generation (`zhuangzhou/report/`, v0.8.6)

| File | Description |
|------|-------------|
| `docx_builder.py` | Low-level .docx builder (python-docx): tables, headings, code blocks, FASTA appendix |
| `templates.py` | High-level builders: mutation report, docking report, MD report, QM report, SynBio report, general report |

**v0.8.6 fixes:**
- FBA key fallback: `objective_flux` → `objective_value` → `flux` (multi-key chain)
- `top_shadow_prices` / `top_fluxes` unified dict renderer
- Context-aware enrichment: HuiZi protein context injected into all report types

**4 registered Report skills:**
`generate_report`, `generate_mutation_report`, `generate_synbio_report`, `generate_docking_report`

### 4.7 Computational Skills (remaining categories)

| Category | Count | Key Skills |
|----------|-------|-----------|
| simulation | 20 | GROMACS MD, AMBER MD, PLIP interaction analysis, NPT/NVT protocols |
| docking | 9 | smart_docking (AutoVina), conformer generation, pocket detection, PLIP |
| research | 13 | graphrag_query, literature_search, uniprot_fetch, pdb_fetch, pubchem_search |
| structure | 7 | pdb_to_fasta, enrich_mutations, pocket_detection, mutagenesis |
| qm | 9 | pyscf_sp, pyscf_opt, pyscf_freq, ts_scan, nbo_analysis, solvation |
| chemistry | 5 | rdkit_descriptors, fingerprint, similarity, scaffold, lipinski |
| sequence | 4 | codon_optimize, pdb_to_fasta, esm2_scan, mutation_classification |
| plm | 1 | esm2_scan (ESM-2 650M zero-shot fitness scan) |
| visualization | 3 | pymol_render, chimerax_render, fast_render |
| mutagenesis | 1 | analyze_mutation (chemical intuition first-principles) |
| brain | 5 | (see §4.5) |

### 4.8 Memory System (7 layers, v0.7.4+)

| Layer | Path | Description |
|-------|------|-------------|
| Journal | `~/.zhuangzhou/journal/` | Append-only raw event log per session |
| Context | `~/.zhuangzhou/context/` | Session state: current protein, analyses, findings |
| Knowledge | `~/.zhuangzhou/knowledge/` | Learned patterns: skill success rates, mutation insights, error patterns |
| Archive | `~/.zhuangzhou/archive/` | Full interaction recordings with tool outputs |
| ProjectMemory | `~/.zhuangzhou/project_memory.md` | Researcher-facing plain-text knowledge accumulator |
| ExecutionHistory | `~/.zhuangzhou/execution/` | Persistent skill log + task checkpoint/resume |
| WorkspaceIntelligence | CWD scan | Auto-detects project stage from PDB/SDF/output files in CWD |

### 4.9 Environment Isolation Layer (`zhuangzhou/envs/`)

| Environment | Activation | Contents |
|-------------|-----------|---------|
| native | default | Vina, GROMACS, RDKit, COBRApy, python-docx |
| graphrag | `E:\conda\envs_dirs\graphrag` | AutoGraphRAG, pandas, networkx |
| py310_DPLM2 | `E:\conda\envs_dirs\py310_DPLM2` | UniMol, ESM-2, PyTorch GPU |
| wsl | WSL subprocess | PLIP 3.0.0, LigBuilder V3, PySCF, fpocket |
| ssh:<host> | `~/.zhuangzhou/ssh_compute.ini` | Remote HPC compute nodes |

---

## 5. v0.8.6 Bug Fixes (11 Critical/High)

All fixed 2026-03-17 in high-concurrency stress testing session (#S389).

| # | File | Severity | Bug | Fix |
|---|------|----------|-----|-----|
| 1 | `nlp/biodesign_mode.py:514` | CRITICAL | LLM fallback `backend.chat(prompt)` — single-arg call crashes all real backends. ALL heterologous compound pathway mining silently failed | Changed to `backend.chat(system, [{'role':'user','content':prompt}])` with `.text` extraction |
| 2 | `skills/metakg.py` | HIGH | `metakg_bottleneck` accepted string input, iterated char-by-char → garbage output | `isinstance(pathway_reactions, list)` type guard |
| 3 | `skills/synbio.py` | HIGH | `synbio_burden_assess` accepted string → `zip()` over chars produced 10 single-char "genes" | `isinstance(heterologous_genes, list)` type guard |
| 4 | `skills/metakg.py` | HIGH | `metakg_query("")` matches entire database (`"" in any_str` is always True) | Explicit empty/None/whitespace guard |
| 5 | `skills/metakg.py` | HIGH | `metakg_compare("")` same issue | Same guard |
| 6 | `skills/synbio.py` | HIGH | `synbio_crispr_design(host=...)` silently ignored — agent uses `host=` but function only accepted `organism=` | Added `host: Optional[str]` parameter with mapping |
| 7 | `synbio/crispr.py:135` | HIGH | SpCas9 PAM regex `[ACGT]{20}GG` — 20th spacer base consumed as PAM N, producing 19nt spacers | Corrected to `[ACGT]{20}[ACGT]GG` |
| 8 | `synbio/parts.py:47` | MEDIUM | BBa_B0033 sequence identical to BBa_B0032 (copy-paste error) | Corrected to `TCAAAGAGGAAAC` |
| 9 | `metakg/flux.py:100-112` | HIGH | Yeast FBA `growth_rate` always 0.0 — pandas `.get()` chain returns NaN for absent E. coli ID, never falls through to yeast biomass ID | Multi-ID loop with `_val > 0` check |
| 10 | `metakg/bridge.py:208` | MEDIUM | `search_nodes(None)` → `AttributeError: 'NoneType' has no .lower()` | `if not query: return []` guard |
| 11 | `report/templates.py:4461,4639,4649` | HIGH | Template used `objective_value` + `top_fluxes` but `metakg_flux` returns `objective_flux` + `top_shadow_prices` — FBA section always blank | Multi-key fallback chain; `top_shadow_prices \|\| top_fluxes` with dict check |

### v0.8.6 Performance Fix

| Fix | Before | After |
|-----|--------|-------|
| `HuiZi.get_report_context(fetch_literature=False)` default | ~15s (PubMed network call) | 0.02s |

### v0.8.6 Test Coverage

```
=== FINAL VALIDATION (post v0.8.6 fixes) ===
[Group 1] 8/8 Input Validation Guards    ✅
[Group 2] 6/6 Functional Correctness     ✅ (E.coli FBA: 5.64 mmol/gDW/h; yeast: 3.74)
[Group 3] 3/3 Report Templates FBA Fix   ✅ (Word doc 39KB generated)
[Group 4] 1/1 HuiZi Performance Fix      ✅ (0.02s)
[Group 5] 1/1 LLM Fallback Fix           ✅

Overall: 19/19 PASS
```

---

## 6. BioDesign Mode: Metabolic Engineering Pipeline (v0.8.6)

The 6-stage pipeline (`nlp/biodesign_mode.py`) executes automatically when metabolic engineering keywords are detected:

```
Stage 1: 目标定义    → product identification, BiGG metabolite ID, chassis selection
Stage 2: 通路挖掘    → KNOWN_PATHWAYS dict → MetaKG BFS → LLM fallback (fixed v0.8.6)
Stage 3: 流量分析    → FBA (COBRApy) theoretical yield + shadow prices + bottlenecks
Stage 4: 干扰消除    → competitive pathways + toxicity screening → knockout candidates
Stage 5: 元件设计    → promoter/RBS/codon opt + metabolic burden assessment
Stage 6: 设计封装    → JSON design package + bilingual Word .docx report
```

**Validated end-to-end** on resveratrol biosynthesis:
- 9-step pathway from glucose correctly reconstructed
- 5 enzyme candidates with UniProt IDs (FjTAL, At4CL1, VvSTS, AccABCD, AroG_fbr)
- CRISPR knockout targets: tyrR, pheA, trpD
- FBA: E. coli yield 5.64 mmol/gDW/h; yeast (TAL route) 3.74 mmol/gDW/h
- 44,772-byte bilingual Word report generated: `resveratrol_full_report.docx`

---

## 7. Known Limitations (v0.8.6)

| # | Component | Limitation | Impact |
|---|-----------|-----------|--------|
| 1 | `synbio/crispr.py` | SaCas9 guide-finding uses SpCas9 `[ACGT]{20}[ACGT]GG` logic instead of 21nt+NNGRRT | Wrong guide sequences for SaCas9 experiments |
| 2 | `skills/synbio.py` | CAI (Codon Adaptation Index) hardcoded at 0.85 (E. coli) / 0.60 (yeast) | Codon optimization reports false precision |
| 3 | `synbio/crispr.py` | `base_edit` has no CBE/ABE-specific editing window logic (C4-8 position not enforced) | Base editing design incorrect |
| 4 | `brain/graphrag.py` | GraphRAG KG corpus empty — no literature loaded | GraphRAG queries return no results; all knowledge retrieval falls back to LLM |
| 5 | `metakg/config.py` | Y. lipolytica iYLI647 model not in MetaKG | No FBA support for industrial yeast chassis (world record: 12.4 g/L resveratrol) |
| 6 | `skills/metakg.py` | `metakg_essentiality` accepts common gene names (aroG, tyrA) but COBRApy requires b-numbers (b0871, b4054) — silent false essentiality results | Gene essentiality analysis unreliable |
| 7 | `skills/synbio.py` | `synbio_design_package(enzymes=...)` parameter silently ignored (correct key is `pathway_enzymes=`) | Design package missing enzyme list |
| 8 | `skills/synbio.py` | `synbio_expression_balance` target_ratios > 2.0 silently clamped without warning | Expression imbalance designs truncated silently |

---

## 8. Industry Benchmarking (2026)

Research conducted 2026-03-17 against state-of-the-art synthetic biology platforms.

### 8.1 Competitive Position

| Capability | TeselaGen | Benchling | novoStoic 2.0 | CRISPR-GPT | **Zhuangzhou v0.8.6** |
|-----------|-----------|-----------|---------------|------------|----------------------|
| Natural language interface | ✅ (domain-specific) | ✅ (data queries only) | ❌ | ✅ (CRISPR only) | ✅ **multi-domain** |
| Automated pathway mining | ❌ | ❌ | ✅ | ❌ | ✅ (KNOWN + MetaKG BFS + LLM) |
| FBA / Flux analysis | ❌ | ❌ | ✅ (thermodynamic) | ❌ | ✅ (COBRApy iML1515/iMM904) |
| Multi-organism GEMs | ❌ | ❌ | ✅ (stoich only) | ❌ | ✅ (2 models) |
| CRISPR guide design | ❌ | ❌ | ❌ | ✅ | ✅ (SpCas9, partially SaCas9) |
| Codon optimization | ❌ | ❌ | ❌ | ❌ | ✅ (CAI hardcoded, v0.9.0 fix) |
| Metabolic burden | ❌ | ❌ | ❌ | ❌ | ✅ (heuristic) |
| Protein structure integration | ❌ | ❌ | ❌ | ❌ | ✅ (AFDB/ESMFold/Vina/MD) |
| Mutation engineering | ❌ | ❌ | ❌ | ❌ | ✅ (ESM-2 + chemical intuition) |
| Word report generation | ❌ | ❌ | ❌ | ❌ | ✅ (bilingual .docx) |
| Full DBTL loop | ✅ (robotic) | ❌ | ❌ | ❌ | ⚠️ (computational only) |
| Enterprise pricing | $$$ | $$$ | Free | Free | **Free/Open** |

**Key insight**: Zhuangzhou is the only open platform with integrated enzyme engineering + metabolic design + NLP. Its multi-domain NLP interface (biology → chemistry → physics → reporting) is architecturally ahead of single-domain tools.

### 8.2 Critical Industry Gaps Zhuangzhou Can Close in v0.9.0

Based on the industry research, the following represent unmet needs across all platforms:

1. **Host-aware pathway-to-expression pipeline** — No tool chains retrosynthesis → GEM host flux → metabolic burden → codon opt → expression vector design. Zhuangzhou's skill architecture makes this feasible.

2. **Metabolic burden and toxicity prediction pre-construction** — All platforms (including Zhuangzhou's heuristic model) fail to predict ribosome competition, ATP/NADPH drain, intermediate accumulation toxicity before lab construction.

3. **Retrosynthesis-to-protein-engineering bridge** — novoStoic/ATLAS identify novel reactions but cannot assess engineering feasibility. Connecting novel reaction → enzyme scaffold → engineering difficulty ranking is an open research gap with direct practical impact.

4. **Manual experiment DBTL feedback loop** — Self-driving labs require robotic biofoundries. Most academic labs need a "soft" closed loop: manual experiment data → Bayesian recommendation update. No platform provides this.

5. **Non-model organism support** — Y. lipolytica, Pseudomonas putida, Streptomyces, Bacillus subtilis are industrial workhorses with almost no integrated computational support.

---

## 9. v0.9.0 Improvement Roadmap

Priority ranked by research impact and implementation feasibility.

### 9.1 Critical Bug Fixes (Must Ship in v0.9.0)

| Priority | Fix | File | Effort |
|----------|-----|------|--------|
| P0 | SaCas9 21nt+NNGRRT guide logic | `synbio/crispr.py` | 2h |
| P0 | `metakg_essentiality` gene name → b-number mapping table | `skills/metakg.py` | 4h |
| P0 | `synbio_design_package` `enzymes=` alias for `pathway_enzymes=` | `skills/synbio.py` | 30m |
| P1 | Real CAI calculation (CodonUsage table, not hardcoded) | `codon_opt/` | 6h |
| P1 | Base editing CBE/ABE C4-8 window validation | `synbio/crispr.py` | 3h |
| P1 | `synbio_expression_balance` target_ratio cap warning | `skills/synbio.py` | 1h |

### 9.2 High-Impact New Features (v0.9.0 Core)

#### Feature 1: Real CAI Codon Optimization
Replace hardcoded 0.85/0.60 with actual Codon Adaptation Index calculation using host codon usage tables (E. coli K-12, S. cerevisiae S288C, Y. lipolytica CLIB89). Implement in `codon_opt/` using published codon frequency tables. Reference: CodonTransformer (Nature Comms 2025) for transformer-based optimization.

#### Feature 2: Y. lipolytica iYLI647 Integration
Add iYLI647 JSON model to MetaKG. Enables FBA for the industrial chassis with the current resveratrol world record (12.4 g/L). Critical for Phase 3 scale-up design in metabolic engineering reports.

#### Feature 3: GraphRAG Corpus Loading
Load 200+ metabolic engineering papers into the BeTaThink KG. Pipeline: PDF → entity extraction → Neo4j/Milvus → GraphRAG parquet. Enable literature-based strategy emergence (e.g., "malonyl-CoA supply strategy from Kim et al. 2020").

#### Feature 4: Soft DBTL Feedback Loop
Implement `log_experiment_result(compound, yield_g_L, od600, notes)` + `update_pathway_recommendation()` skill pair. Use Bayesian updating to revise enzyme selection and flux targets from manual experimental data. Fills the industry gap no platform currently addresses.

#### Feature 5: SaCas9 + Additional CRISPR Systems
Extend `synbio/crispr.py` with:
- SaCas9: 21nt spacer + NNGRRT PAM
- Cas12a/Cpf1: TTTN PAM, 23nt spacer
- CRISPRi (dCas9-KRAB): repression efficiency prediction
- Off-target scoring: CFD score or Doench 2016 model

#### Feature 6: Metabolic Burden Estimator v2
Replace heuristic burden assessment with stoichiometry-driven model:
- NADPH/ATP balance from pathway stoichiometry (COBRApy shadow prices)
- Ribosome load estimate from estimated expression level × gene count
- Intermediate toxicity flags: cross-reference against known toxic metabolite database (HMDB toxicity data)

### 9.3 Infrastructure Upgrades (v0.9.0)

| Item | Description | Rationale |
|------|-------------|-----------|
| Surrogate FBA model | Train ML surrogate on iML1515 FBA outputs for 100x faster local pathway flux queries | Enable real-time metabolic checking inside NLP agent loops without COBRApy startup latency |
| Multi-model GEM cache | Lazy-load + cache all GEM models (ecoli/yeast/lipolytica/bacillus) in process memory | Eliminate 3-5s model loading on each skill call |
| novoStoic API integration | Add `metakg_retrosynthesis(target_compound)` skill calling novoStoic 2.0 API | Enables non-natural pathway design beyond KNOWN_PATHWAYS |
| RetroRules 2026 template DB | Local mirror of 1.17M reaction templates for offline retrosynthesis | Eliminates dependency on network access; enables chemoenzymatic route planning |
| ProteinGPT-style context | Feed AFDB structure embeddings into agent context alongside sequence | Multi-modal protein reasoning per 2025 ProteinGPT architecture |

### 9.4 NLP Interface Improvements (v0.9.0)

| Item | Description |
|------|-------------|
| BioDesign keyword expansion | Add more trigger phrases for automatic BioDesign mode activation |
| Multi-agent parallel pathway mining | Spawn parallel agent tasks for retrosynthesis + FBA + literature simultaneously |
| Experiment data ingestion | Natural language parsing of experimental results (e.g., "OD reached 8.5, resveratrol yield was 230 mg/L") |
| Session export | `/export` command to save full session (queries + results + report) as reproducible script |

---

## 10. UI System (v0.8.6)

### 10.1 Banner (corrected in v0.8.6)

```
  ███████╗██╗  ██╗██╗   ██╗ █████╗ ███╗   ██╗ ██████╗
  ╚══███╔╝██║  ██║██║   ██║██╔══██╗████╗  ██║██╔════╝
    ███╔╝ ███████║██║   ██║███████║██╔██╗ ██║██║  ███╗
   ...

  ███████╗██╗  ██╗ ██████╗ ██╗   ██╗   庄周 v0.8.6
  ...                                  Router → Fixed|Plan|Direct
```

**Fix applied**: `ui/core.py:764` — hardcoded version `"0.8.5"` → `"0.8.6"`.

### 10.2 Color Palette

| Token | Hex | Usage |
|-------|-----|-------|
| Electric Cyan | `#00E5FF` | Primary text, prompt, borders |
| Amber Gold | `#FFC107` | Accents, chevron, tool call highlighting |
| Gradient Mid | `#00D4E0` | Banner gradient midpoint |
| Deep BG | `#12151C` | Response panel background |
| Success Green | `#66BB6A` | Successful operations |
| Lava Orange | `#FF8F00` | Warnings and errors |

---

## 11. Integration Interfaces

### 11.1 Python API

```python
from zhuangzhou.nlp import chat, create_agent
from zhuangzhou.huizi import HuiZi
from zhuangzhou.metakg import get_bridge, FluxAnalyzer
from zhuangzhou.synbio import PartSelector, find_guides
from zhuangzhou.report import build_synbio_report

# One-shot metabolic engineering query
result = chat("为大肠杆菌设计白藜芦醇合成通路", backend="deepseek")

# Full session with context
hz = HuiZi()
hz.set_protein(name="STS", organism="Vitis vinifera")
hz.set_ligand(name="resveratrol", smiles="Oc1ccc(/C=C/c2cc(O)cc(O)c2)cc1")
ctx = hz.get_report_context()

# Direct FBA
fa = FluxAnalyzer("ecoli")
result = fa.run_fba("EX_tyr__L_e", carbon_source="glc__D_e")
```

### 11.2 CLI REPL

```bash
python -m zhuangzhou.nlp --backend deepseek
python -m zhuangzhou.nlp --backend claude --verbose
python -m zhuangzhou.nlp --oneshot "分析V21L突变对CYP3A4的影响"
```

### 11.3 Slash Commands

| Command | Description |
|---------|-------------|
| `/help` | Show all commands and skill categories |
| `/status` | Show current session state and loaded models |
| `/context` | Show HuiZi session context (current protein/ligand/analyses) |
| `/init` | Scan CWD and update ZHUANGZHOU.md project memory |
| `/plan on\|off` | Toggle Plan Mode (LLM clarification before execution) |

---

## 12. Development Practices

### Error Convention
All skill functions return `{'error': '...', 'hint': '...'}` on failure — never raise exceptions.
This enables the ReAct agent to self-correct without crashing.

### Output Directory Convention
When `output_dir=None`, all modules default to `os.getcwd()/<stem>_<suffix>/`.
Researchers `cd` to their project folder; all outputs (structures, docking, MD, reports) stay organized.

### Windows Compatibility
All `open()` calls use `encoding='utf-8'`. All console output uses `sys.stdout.reconfigure(encoding='utf-8')`.
Forward slashes in paths (WSL/Linux-style) in `.cxc` scripts.

### Security
No external code execution except through sandboxed environments (`envs/sandbox.py`).
No secrets stored in code; API keys from environment variables only.

---

*End of Zhuangzhou v0.8.6 Technical Report*
*Generated: 2026-03-17 | Next version: v0.9.0*
