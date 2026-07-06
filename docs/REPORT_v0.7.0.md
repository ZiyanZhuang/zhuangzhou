# Zhuangzhou v0.7.0 技术报告

**版本**: 0.7.0
**日期**: 2026-03-08
**作者**: BeTaThink 团队

---

## 一、版本概述

v0.7.0 是以**系统鲁棒性工程**为核心的质量强化版本。在 v0.6.0 完成"被动工具箱→主动智能体"架构转变的基础上，本版本完成了：

1. **全系统 Unicode 安全**：91 个 bare `open()` 调用全部修复为显式 UTF-8 编码
2. **API 容错层**：所有 LLM 后端（Claude/DeepSeek/OpenAI/Ollama）加入 try-except 错误恢复
3. **生物数据库 API 集成**：UniProt/NCBI/InterPro/PDB/AlphaFold 联网检索（11 个新技能）
4. **MD 管线鲁棒性**：13 项关键修复（动态分组、预飞检查、参数自适应）
5. **可视化修复**：HEME/配体在轨迹动画中正确显示
6. **陈旧文件清理**：删除 ~600 MB 过期测试数据和归档文档
7. **技能总数**：从 55 增至 **81 个注册技能**（11 个类别）

**代码规模**：145 个 Python 文件，40,445 行代码，25 个包目录

---

## 二、系统架构（v0.7.0 更新）

```
+-------------------------------------------------------------------+
|  用户层（自然语言 / Python API / CLI）                               |
+-------------------------------------------------------------------+
|  NLP Agent（可插拔 LLM: Claude/DeepSeek/OpenAI/Ollama）              |
|  +-- nlp/agent.py — 多轮对话 + tool_use 循环（max 15 rounds）       |
|  +-- nlp/backends.py — 4 后端 + 错误恢复（v0.7.0 加固）            |
|  +-- nlp/tools.py — 81 个技能 → LLM tool_use schema                |
+-------------------------------------------------------------------+
|  Skills Registry — 81 个注册技能（11 类别）                          |
|  +-- research (17): BioDBAPI 联网检索（v0.7.0 新增）                 |
|  +-- simulation (14): GROMACS 全流程 + CGenFF + 非标准残基           |
|  +-- docking (9): SmartDocking + Vina + LigBuilder                  |
|  +-- qm (9): DFT/TS/频率/能量剖面                                   |
|  +-- visualization (8): ChimeraX + PyMOL + PLIP                    |
|  +-- chemistry (6): ADMET + 构象 + UniMol                           |
|  +-- brain (5): GraphRAG KG 查询                                    |
|  +-- structure (4): 口袋/辅因子/PLIP                                |
|  +-- sequence (4): 密码子优化                                        |
|  +-- plm (3): ESM-2 突变扫描                                        |
|  +-- mutagenesis (2): 点突变                                         |
+-------------------------------------------------------------------+
|  计算环境隔离（envs/sandbox.py）                                     |
|  +-- native: Vina, GROMACS, RDKit, DNAchisel                       |
|  +-- graphrag: AutoGraphRAG parquet                                 |
|  +-- dplm2: UniMol, ESM-2, torch GPU                               |
|  +-- WSL: PLIP, LigBuilder, PySCF, Linux 工具链                    |
|  +-- SSH: 远程 HPC 计算节点（v0.7.0 新增）                          |
+-------------------------------------------------------------------+
|  记忆系统（5 层）                                                    |
|  Journal → Context → Knowledge → Archive → ProjectMemory           |
+-------------------------------------------------------------------+
```

---

## 三、v0.7.0 新增功能

### 3.1 生物数据库 API 联网检索（skills/biodbapi.py）

**概述**：实现了 5 大生物数据库的 REST API 集成，使系统能够实时获取蛋白质序列、结构、结构域、催化反应等详细信息，为下游 MD/对接/突变分析提供精确输入。

**API 覆盖**：

| 数据库 | API 端点 | 功能 | 技能名 |
|--------|---------|------|--------|
| UniProt | REST v2 | 序列、结构域、全量注释 | `get_uniprot_sequence`, `get_uniprot_domains`, `get_uniprot_full`, `search_uniprot`, `get_uniprot_entry` |
| NCBI Entrez | E-utilities | 蛋白质检索、FASTA、基因信息 | `search_ncbi_protein`, `get_ncbi_protein_fasta`, `search_ncbi_gene` |
| EMBL-EBI InterPro | REST | 蛋白结构域预测、GO 注释 | `get_interpro_domains` |
| RCSB PDB | Search API v2 | 晶体结构检索 | `search_pdb_structure`, `get_pdb_entry` |
| AlphaFold DB | REST | AI 预测结构下载 | `get_alphafold_prediction` |

**复合查询**：`get_protein_profile(identifier)` — 一键调用 UniProt + NCBI + InterPro + PDB，返回统一结构化结果。

**工程特性**：
- 纯 stdlib（`urllib`），无外部依赖
- 3 次重试 + 429 退避 + `socket.timeout` 处理
- NCBI 3 req/s 限速（`_ncbi_throttle()`）
- 所有返回值为 dict，失败返回 `{'error': '...', 'hint': '...'}`

**已验证**：CYP3A4 (P08684) 全量检索，11/11 校验项通过，4 个数据源均正常返回。

### 3.2 SSH 远程计算支持（envs/ssh_config.py + sandbox.py）

新增 `SSHSandbox` 类，支持通过 SSH 在远程 HPC 节点上执行计算任务：
- 配置文件：`~/.zhuangzhou/ssh_compute.ini`
- 功能：命令执行、文件上传/下载、远程环境检测
- 超时：连接超时 + 命令超时独立控制
- 错误处理：`TimeoutExpired` + 通用 `Exception` 全覆盖

### 3.3 CGenFF 配体参数化（gromacs_md/cgenff_param.py）

完整的 CGenFF 小分子参数化管线，用于 CHARMM36 力场兼容的配体-蛋白 MD：
- PDB/SDF → MOL2 (OpenBabel) → CGenFF .str → GROMACS .itp/.prm
- 3 种 .str 来源：本地 CGenFF binary → ParamChem web → 用户提供
- `cgenff_charmm2gmx.py`（Lemkul Lab 脚本）集成
- 惩罚分数验证：<10 good, 10-50 fair, >50 需 ffTK QM 优化
- `integrate_cgenff_topology()` 自动注入 .prm + .itp 到拓扑

---

## 四、v0.7.0 鲁棒性工程（核心主题）

### 4.1 全系统 UTF-8 编码安全

**问题**：Windows 默认编码 GBK，GROMACS 输出、PDB 文件等含非 ASCII 字符时 `open()` 会崩溃。

**修复**：跨 31 个文件，91 个 bare `open()` 调用全部添加 `encoding='utf-8'`：
- 读操作：`encoding='utf-8', errors='replace'`（容忍损坏数据）
- 写操作：`encoding='utf-8'`（确保输出正确编码）

**覆盖模块**：

| 模块 | 修复数 |
|------|--------|
| gromacs_md/ (7 files) | 36 |
| nonstandard_res/ (4 files) | 26 |
| qm_engine/ (4 files) | 8 |
| ligbuilder_gen/ (2 files) | 3 |
| 其他 (14 files) | 18 |
| **合计** | **91** |

### 4.2 LLM 后端 API 容错

**修复前**：API 调用失败（网络超时、限速、服务端错误）导致整个对话崩溃。

**修复后**（nlp/backends.py）：
- **ClaudeBackend**：`client.messages.create()` 包裹 try-except → 返回 `LLMResponse(stop_reason='error')`
- **OpenAICompatibleBackend**：同上 + 空 `resp.choices` 防护
- **OllamaBackend**：`urllib.request.urlopen()` 包裹 try-except → 返回错误 LLMResponse

Agent 层面收到 `stop_reason='error'` 后可以选择重试或向用户报告，不会崩溃。

### 4.3 GROMACS MD 管线 13 项修复（S57）

| # | 严重度 | 修复 | 影响 |
|---|--------|------|------|
| 1 | CRITICAL | `_build_tau_t_ref_t()` 动态生成 tau_t/ref_t | 1 或 3+ tc_grps 不再崩溃 |
| 2 | CRITICAL | `_parse_index_groups()` + `_find_group()` | 分析不再硬编码组号 |
| 3 | CRITICAL | `split_complex()` 4 字符残基名支持 | HEME 等长名称不再截断 |
| 4 | CRITICAL | `_fix_resname()` 正则匹配 | MOL/ligand/UNK/LIG 全覆盖 |
| 5 | CRITICAL | `_preflight_check()` 预飞验证 | 4 个管线入口全部加入 |
| 6 | HIGH | `solvate_system()` 动态水组检测 | Water/SOL/W 自适应 |
| 7 | HIGH | `compute_coordination()` 多金属支持 | Zn/Cu/Mn/Co/Ni/Mg |
| 8 | HIGH | `_fix_pbc_after_em()` 动态 System 组 | 不再硬编码 '0' |
| 9 | HIGH | `compute_energy()` 名称优先查找 | Potential 组名→序号回退 |
| 10 | MODERATE | 辅因子检测列表扩展 | +12 种辅因子 |
| 11 | MODERATE | 金属检测按原子名 | FE/ZN/CU/MN 不限残基 |
| 12 | MODERATE | UTF-8 编码（12 处） | Windows 兼容 |
| 13 | MODERATE | H-bond 动态组查找 | 蛋白+辅因子/配体自适应 |

### 4.4 分析模块数据安全

- `_parse_xvg()`：空 XVG 文件返回空数组，不再导致下游 IndexError
- PDB 坐标解析（analysis.py, visualize.py）：`float(line[30:38])` 包裹 try-except
- 结构比较：不足 2 帧时提前退出

### 4.5 配置与诊断增强

- **diagnostics.py**：LINCS 阈值从 5 提升到 20（stepdown 协议兼容），全部阈值支持环境变量覆盖（`ZZ_LINCS_WARN_THRESHOLD` 等）
- **config.py**：GMXLIB 不再设为空字符串，正确的 None/验证链
- **restraint_protocol.py**：posre 文件发现改用 glob，支持任意链名

### 4.6 HEME/配体可视化修复

**问题**：GROMACS `trjconv` 将所有原子写为 ATOM（不使用 HETATM），且 4 字符残基名 HEME 在 PDB 中存储为 HEMEA（列 17-21）。

**修复**（visualize.py）：
- 同时解析 `line[17:20]` 和 `line[17:21].strip()`
- 移除 `line.startswith('HETATM')` 门控
- 已知辅因子列表优先匹配

**验证**：CYP P450cam 轨迹动画正确显示 HEME（salmon sticks）和 LIG（cyan sticks）。

---

## 五、技能注册表（81 个技能，11 类别）

### 5.1 完整技能清单

| 类别 | 数量 | 技能列表 |
|------|------|---------|
| **research** | **17** | get_alphafold_prediction, get_interpro_domains, get_mutation_evidence, get_ncbi_protein_fasta, get_pdb_entry, get_protein_profile, get_uniprot_domains, get_uniprot_entry, get_uniprot_full, get_uniprot_sequence, search_graphrag, search_ncbi_gene, search_ncbi_protein, search_pdb_structure, search_pubmed, search_uniprot, synthesis_knowledge |
| **simulation** | **14** | auto_md, build_nonstandard_topology, cgenff_parameterize, detect_nonstandard, gromacs_analysis, gromacs_comprehensive, gromacs_md, gromacs_process_trajectory, gromacs_visualize, metalloprotein_ligand_md, metalloprotein_md, minimize_structure, parameterize_nonstandard, unimol_check_gpu |
| **docking** | **9** | batch_mutant_docking, generate_complex, generate_docking_grid, ligbuilder_cavity, ligbuilder_design, ligbuilder_inhibitor, smart_docking, unimol_binding_affinity, vina_docking |
| **qm** | **9** | dft_optimization, dft_single_point, frequency_analysis, qm_cluster, qm_energy_profile, qm_reaction_analysis, qm_report, ts_pipeline, ts_search |
| **visualization** | **8** | chimerax_comparison, chimerax_defattr, chimerax_morph, chimerax_porcupine, chimerax_render, chimerax_trajectory, plip_visualize, pymol_view |
| **chemistry** | **6** | batch_conformer_generate, chemical_analysis, conformer_generate, unimol_conformer, unimol_mol_property, zhuangzhou_pipeline |
| **brain** | **5** | kg_explore, kg_query, literature_search, synbio_pipeline, upload_document |
| **structure** | **4** | cofactor_detection, mutation_classification, plip_analysis, pocket_detection |
| **sequence** | **4** | analyze_codon_usage, batch_codon_optimize, codon_optimize, compare_codon_usage |
| **plm** | **3** | esm2_load, esm2_scan, esm2_scan_gpu |
| **mutagenesis** | **2** | mutate_minimize, mutate_protein |

### 5.2 v0.7.0 技能变化

- **新增 17 个**：research 类别完整新增（BioDBAPI 11 个 + PubMed/GraphRAG/Mutation Evidence 等 6 个）
- **重分类**：部分技能从旧类别迁移至更合理的分类
- **净增**：81 - 55 = **+26 个技能**

---

## 六、公共 API（agent.py）

### 6.1 新增函数（v0.7.0）

| 函数 | 参数 | 说明 |
|------|------|------|
| `get_protein_profile(identifier)` | UniProt ID 或 PDB ID | 一站式蛋白信息检索（序列+结构域+结构+GO） |
| `get_protein_sequence(accession)` | UniProt/NCBI accession | 智能路由：NP_/XP_ → NCBI，其他 → UniProt |
| `search_protein_database(query, database, organism, max_results)` | 搜索词 + 库名 | UniProt/NCBI/PDB 统一搜索接口 |

### 6.2 保留函数（v0.6.0）

| 函数 | 说明 |
|------|------|
| `analyze_mutation()` | 第一性原理突变分析 |
| `search_knowledge()` | GraphRAG + BeTaThink KG 搜索 |
| `run_pipeline()` | 5 阶段酶工程流水线 |
| `check_environments()` | 沙箱状态检查 |
| `list_skills()` | 查看注册技能 |
| `execute_skill()` | 执行任意技能 |
| `graphrag_query()` | 直接 GraphRAG 查询 |

---

## 七、文件结构（v0.7.0）

```
zhuangzhou/                          # 145 个 Python 文件, 40,445 行
+-- __init__.py                      # v0.7.0
+-- agent.py                         # 10 个公共 API 函数
+-- app.py                           # 交互式 REPL
+-- core.py                          # 5 阶段酶工程流水线
+-- conversation.py                  # Gather->Act->Verify 引擎
+-- chemical_intuition.py            # 第一性原理突变分析
+-- pyproject.toml                   # 项目配置（pip install .）
|
+-- nlp/                             # NLP 自然语言控制
|   +-- agent.py                     # ZhuangzhouAgent 多轮对话
|   +-- backends.py                  # 4 LLM 后端 + 错误恢复 (v0.7.0)
|   +-- tools.py                     # 81 技能 → tool_use schema
|   +-- repl.py                      # 命令行 REPL
|
+-- skills/                          # 技能注册系统
|   +-- registry.py                  # SkillRegistry 单例
|   +-- biodbapi.py                  # 生物数据库 API (v0.7.0 新增)
|   +-- simulation.py, research.py, docking.py, ...
|
+-- gromacs_md/                      # GROMACS MD 全流程 (v0.7.0 加固)
|   +-- pipeline.py                  # 4 管线 + stepdown + preflight
|   +-- prepare.py                   # 体系制备 + 动态残基名
|   +-- analysis.py                  # 分析 + 动态组检测
|   +-- visualize.py                 # 轨迹动画 + PCA + HEME 修复
|   +-- restraint_protocol.py        # 约束管理 + glob posre
|   +-- diagnostics.py               # 增强诊断 + 可配阈值
|   +-- heme_topology.py             # Fe-S 键注入
|   +-- cgenff_param.py              # CGenFF 参数化
|   +-- config.py                    # 自动检测 + GMXLIB 修复
|   +-- templates/                   # MDP 模板
|
+-- envs/                            # 计算环境隔离
|   +-- sandbox.py                   # Conda/WSL/SSH 沙箱
|   +-- ssh_config.py                # SSH HPC 配置 (v0.7.0)
|
+-- nonstandard_res/                 # 非标准残基参数化
+-- qm_engine/                       # QM 量子力学计算
+-- smartdocking/                    # 分子对接
+-- pymol_viz/                       # PyMOL 渲染
+-- chimerax_viz/                    # ChimeraX 可视化
+-- plip_analysis/                   # PLIP 分析
+-- codon_opt/                       # 密码子优化
+-- ligbuilder_gen/                  # De Novo 设计
+-- molecularconformer/              # 构象生成
+-- mutate_minimize/                 # 突变+EM
+-- mutagenesis/                     # 点突变
+-- brain/                           # BeTaThink 知识后端
+-- memory/                          # 5 层记忆系统
+-- workflow/                        # JSON 工作流引擎
+-- commands/                        # CLI 命令
+-- tests/                           # 测试套件
```

---

## 八、外部依赖状态

### 8.1 软件检测结果

| 软件 | 状态 | 路径 |
|------|------|------|
| GROMACS 2025.1 | OK | `<GMX_BIN>` |
| GMXLIB | OK | `<GMXLIB>` |
| SOBTOP 1.0(dev5) | OK | `<SOBTOP_DIR>/sobtop.exe` |
| OpenBabel 3.x | OK | `<OBABEL_EXE>` |
| PyMOL 3.03 | OK | `<PYMOL_PYTHON>` |
| FFmpeg | OK | `<FFMPEG_BIN>` |
| MDP Templates | OK | `gromacs_md/templates/` |
| CHARMM36 | OK | `charmm36-feb2021` |

### 8.2 新增 Python 依赖

无新增外部依赖。`biodbapi.py` 使用纯 stdlib（`urllib`, `json`, `xml`, `socket`）。

---

## 九、性能基准

| 操作 | 耗时 | 硬件 |
|------|------|------|
| 10ns 纯蛋白 MD (~40K atoms) | ~25 min | RTX 5070 |
| 10ns 金属蛋白 MD (~48K atoms) | ~45 min | RTX 5070 |
| SmartDocking 全流程 | ~30s | CPU |
| PLIP 分析 + PyMOL 渲染 | ~10s | CPU |
| 蛋白全量信息检索 (get_protein_profile) | ~3s | 网络 |
| DFT 单点 (水分子, def2-TZVP) | ~3.2s | WSL CPU |
| 密码子优化 (DNAchisel) | <1s | CPU |
| 构象生成 (RDKit) | ~2s/分子 | CPU |

---

## 十、v0.7.0 变更日志

### 新增模块
- **BioDBAPI**（`skills/biodbapi.py`）：UniProt/NCBI/InterPro/PDB/AlphaFold 联网检索，17 个研究类技能
- **SSH 远程计算**（`envs/ssh_config.py`）：HPC 节点远程执行支持

### 新增功能
- **CGenFF 参数化管线**（`gromacs_md/cgenff_param.py`）：CHARMM36 兼容配体参数化
- **HEME/配体可视化修复**：轨迹动画正确显示金属辅因子和配体
- **蛋白数据库统一搜索**：`agent.py` 新增 3 个便捷函数

### 鲁棒性工程（v0.7.0 核心）
- **编码安全**：91 个 bare `open()` → `encoding='utf-8'`（31 个文件）
- **API 容错**：3 个 LLM 后端加入 try-except 错误恢复
- **MD 管线**：13 项关键修复（动态分组、预飞检查、参数自适应）
- **数据安全**：空 XVG 防护、PDB 坐标解析防护、数组边界检查
- **配置增强**：诊断阈值可配、GMXLIB 验证、posre glob 发现
- **清理**：删除 ~600 MB 陈旧测试数据、213 个 GROMACS 备份文件、7 个过期文档

### 技能变化
- 新增：26 个技能（research x17, simulation +cgenff, 重分类调整）
- 总计：**81 个注册技能**，**11 个类别**

### 已知问题
- Pichia-CLM 权重文件缺失（GitHub 仓库不完整）
- gpu4pyscf 不支持 RTX 5070（Blackwell SM 12.0）
- gmx_MMPBSA 不可用（parmed 构建失败）
- 酶特异性 GNN 仍为 DEPRECATED

---

## 十一、v0.7.0→v0.8.0 路线图建议

### P0（Critical）
1. **核心模块单元测试** — 81 个技能仅有少量集成测试，需要 pytest 覆盖
2. **错误处理统一** — 部分模块仍混用 exception 和 error dict

### P1（High）
3. **FEP 自由能微扰** — 集成 alchemlyb + GROMACS FEP 工作流
4. **OpenFF/Espaloma 集成** — 下一代小分子力场参数化（参考力场参数化生产实践报告）
5. **批量并行调度** — asyncio/concurrent.futures 批量 MD/对接
6. **日志系统** — `logging` 替代 `print()`，按模块命名

### P2（Medium）
7. **增强采样** — REST2/metadynamics MDP 模板
8. **CI/CD** — GitHub Actions CPU-only 测试
9. **类型注解** — mypy 检查覆盖核心模块

---

*Zhuangzhou v0.7.0 — 庄周梦蝶，系统鲁棒，万法归一*
