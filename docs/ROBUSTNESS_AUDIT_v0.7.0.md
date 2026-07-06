# Zhuangzhou v0.7.0 — 多智能体鲁棒性审计报告

**审计日期**: 2026-03-08
**审计范围**: 全系统 9 个核心模块，145 个 Python 文件
**审计方法**: 逐文件代码审查 + 跨模块交互分析

---

## 审计总览

| 严重性 | 主报告 | 智能体补充 | 合计 | 说明 |
|--------|--------|-----------|------|------|
| **CRITICAL** | 5 | 8 | **13** | 可能导致崩溃、数据损坏或静默错误结果 |
| **HIGH** | 11 | 9 | **20** | 边缘场景下降低可靠性 |
| **MEDIUM** | 9 | 10 | **19** | 影响可维护性或引起混淆 |
| **LOW** | 6 | 10 | **16** | 小幅改进建议 |
| | | | **68** | |

**审计架构**: 1 主审计 + 3 并行子智能体（MD管线、NLP/技能、专项模块）

---

## 一、CRITICAL 级别（5 项）

### C1. NativeSandbox shell 注入风险
**文件**: `envs/sandbox.py:261-276`
**描述**: `NativeSandbox.run()` 使用 `shell=True` 并直接将 `command` 参数传入 `subprocess.run()`。如果 LLM 构造了包含 shell 元字符的命令（如 `; rm -rf /`），将被直接执行。
**建议修复**: 对 NativeSandbox 的命令输入进行白名单验证，或使用 `shlex.split()` + `shell=False`。对于通过 LLM 工具调用传入的命令，应添加前缀检查（仅允许已知安全的工具如 `gmx`, `vina`, `obabel`）。

### C2. QM 引擎 GPU fallback 丢失色散/溶剂校正
**文件**: `qm_engine/bridge.py:266-279` (OPT 脚本), `bridge.py:504-518` (SP 脚本)
**描述**: GPU fallback 路径创建新的 `mf_gpu` 对象后直接替换 `mf`，但此前已应用的 D3 色散校正和 ddCOSMO 溶剂化包装被丢弃。导致 GPU 路径计算的能量缺少色散和溶剂贡献，与 CPU 路径结果不一致。
**建议修复**: 将 GPU mf 对象创建移到色散/溶剂校正之前，或在 GPU 对象上重新应用这些修饰。

### C3. ReAct 循环技能执行无超时限制
**文件**: `nlp/agent.py:409-413`
**描述**: `_execute_skill_with_ui()` 中 `_run()` 函数直接调用 `skill.execute(**params)` 无超时保护。如果某个技能（如 GROMACS MD）挂起，整个 ReAct 循环将永久阻塞，无法被 `MAX_ITERATIONS` 或 3 次错误断路器保护。
**建议修复**: 使用 `concurrent.futures.ThreadPoolExecutor` 配合 `future.result(timeout=skill.estimated_timeout or 600)` 为每个技能执行添加超时。

### C4. WSLSandbox.run_python JSON 转义不安全
**文件**: `envs/sandbox.py:243-248`
**描述**: `run_python()` 中使用 `json_str.replace('"', '\\"')` 进行 JSON 转义后嵌入 bash 命令字符串。这无法正确处理包含 `$`, `` ` ``, `\` 等 bash 特殊字符的 JSON 值，可能导致命令注入或参数截断。
**建议修复**: 使用 `shlex.quote()` 包裹整个 JSON 字符串，或通过临时文件传递 JSON 数据（参考 `_run_wsl_pyscf` 的文件传递模式）。

### C5. mutate_minimize genion 硬编码水分子组名
**文件**: `mutate_minimize/engine.py:337`
**描述**: `_solvate_for_em()` 中 `genion` 命令硬编码 `input_text='SOL\n'`，但 CHARMM 力场使用 "TIP3" 或 "Water" 作为水分组名。使用 CHARMM 力场时 genion 会失败。
**建议修复**: 参考 `gromacs_md/prepare.py:498-505` 的动态水分子组检测逻辑（`make_ndx` 解析），或添加力场→组名映射。

---

## 二、HIGH 级别（11 项）

### H1. 拓扑文件编辑使用字符串查找而非解析器
**文件**: `gromacs_md/prepare.py:403-426` (merge_system)
**描述**: `merge_system()` 通过 `content.find('[ system ]')` 和 `content.find('.ff/forcefield.itp"')` 定位拓扑文件插入点。如果注释行中包含这些文本，或者格式略有不同（如 `[system]` vs `[ system ]`），定位将失败并可能损坏拓扑。
**建议修复**: 编写基于行的拓扑解析器，按 `re.match(r'\[\s*system\s*\]')` 匹配节头。

### H2. split_complex CONECT 记录无条件归入配体
**文件**: `gromacs_md/prepare.py:79-80`
**描述**: 所有 `CONECT` 记录被归入 `ligand_lines`，包括蛋白质内部的 CONECT 记录（如二硫键）。这可能导致蛋白质 PDB 缺少关键连接信息。
**建议修复**: 仅保留引用配体原子的 CONECT 记录。

### H3. Ollama 后端 _convert_messages 与 OpenAI 后端代码重复
**文件**: `nlp/backends.py:374-449` vs `backends.py:191-279`
**描述**: Ollama 的 `_convert_messages()` 与 OpenAI 的版本几乎完全相同（~80 行重复代码）。维护时容易修复一处遗漏另一处。
**建议修复**: 提取为 `LLMBackend` 基类的共享方法，子类仅覆盖差异部分。

### H4. 确认门控使用简单关键词匹配
**文件**: `nlp/agent.py:507-519`
**描述**: 检测用户确认使用 `confirm_keywords = {'确认', '执行', '好的', 'yes', ...}`，在消息中做子串匹配。"不要执行" 包含 "执行" 也会触发确认。同时，向后搜索 `needs_confirmation` 的逻辑只检查第一个匹配的 user message，可能误关联。
**建议修复**: 改用 LLM 意图分类，或要求用户发送特定格式的确认（如 `/confirm <skill_name>`）。

### H5. 上下文摘要使用额外 LLM 调用但无速率限制
**文件**: `nlp/agent.py:469-502`
**描述**: `_summarize_history()` 在消息数 >15 时调用 LLM 进行摘要。如果 ReAct 循环频繁触发（如每轮 2-3 个工具调用），可能在单次对话中多次触发摘要，额外消耗 API 配额和时间。
**建议修复**: 添加冷却期（如最近 5 分钟内不重复摘要），或使用本地截断策略替代 LLM 摘要。

### H6. SkillRegistry 单例非线程安全
**文件**: `skills/registry.py:319-323`
**描述**: `get_registry()` 使用全局 `_registry_instance` 但无锁保护。在 `AgentExecutor` 的 `ThreadPoolExecutor` 并发执行技能时，如果首次调用触发 `load_all()`，可能出现竞争条件导致重复注册或部分加载。
**建议修复**: 使用 `threading.Lock` 保护单例创建和 `load_all()` 方法。

### H7. gromacs_md pipeline 缺少 PBC 修复后的坐标验证
**文件**: `gromacs_md/pipeline.py` (多处)
**描述**: `_fix_pbc_after_em()` 调用 `trjconv -pbc whole` 后不验证输出文件的完整性。如果 trjconv 失败（如索引组不匹配），后续步骤将使用包裹的坐标，可能导致 LINCS 爆炸。
**建议修复**: 检查 trjconv 输出文件大小 > 0，并与输入文件原子数一致。

### H8. build_system_prompt 读取 ZHUANGZHOU.md 无 errors='replace'
**文件**: `nlp/agent.py:324`
**描述**: `open(local_md, 'r', encoding='utf-8')` 缺少 `errors='replace'`。如果 ZHUANGZHOU.md 包含非 UTF-8 字符（如用户从 Word 复制的文本），将抛出 UnicodeDecodeError 并中断整个系统提示词构建。
**建议修复**: 添加 `errors='replace'`。

### H9. CGenFF parameterize 回退到 SOBTOP/GAFF 的力场不兼容
**文件**: `gromacs_md/prepare.py:652-671`
**描述**: `parameterize_ligand_cgenff()` 文档说"Falls back to SOBTOP/GAFF if CGenFF is unavailable"，但代码中并无实际回退逻辑。如果开发者后续添加回退，将导致 CHARMM蛋白+GAFF配体 的力场不兼容组合。
**建议修复**: 删除误导性文档注释，或添加回退时的力场兼容性检查。

### H10. SmartDocking 不验证外部 SDF 文件格式
**文件**: `smartdocking/pipeline.py:109-111`
**描述**: 当用户提供 `ligand_sdf` 时，代码直接使用该文件而不验证其是否为有效的 SDF/MOL 文件。无效文件将在后续 `sdf_to_pdbqt()` 中以不明确的错误失败。
**建议修复**: 在使用前检查文件头（`$$$$` 标记）或用 RDKit 快速解析验证。

### H11. _fix_resname_gro 仅替换 'MOL'
**文件**: `gromacs_md/prepare.py:341-345`
**描述**: `_fix_resname_gro()` 仅在前 10 字符中查找 'MOL' 并替换。不处理 ACPYPE 生成的 'UNK' 或 'LIG' 等默认名称。且 `replace('MOL', ...)` 的第一次匹配可能替换错误位置。
**建议修复**: 参照 `_fix_resname()` 的正则表达式逻辑，按 GRO 格式列位精确替换残基名。

---

## 三、MEDIUM 级别（9 项）

### M1. ThreadPoolExecutor max_workers 设为工具调用数
**文件**: `nlp/agent.py:583`
**描述**: `ThreadPoolExecutor(max_workers=len(other_calls))` 允许无限制并发。如果 LLM 一次性请求 10+ 个工具调用，将创建等量线程并发执行 IO/计算密集型技能。
**建议修复**: 设置 `max_workers=min(len(other_calls), 4)` 上限。

### M2. ToolGenerator.generate_schema 缺少对 'path' 类型的特殊处理
**文件**: `nlp/agent.py:84-94`
**描述**: `_map_type()` 将 'path' 映射为 'string'，但不为 path 类型添加 `format: "path"` 提示，LLM 无法区分普通字符串和文件路径参数。
**建议修复**: 为 path 类型添加 `description` 后缀 "(file path)" 或 JSON Schema `format` 字段。

### M3. check_packages 的 exec 嵌套字符串构造复杂
**文件**: `envs/sandbox.py:186-203`
**描述**: `check_packages()` 使用多层转义的 `exec()` 字符串来构建包检查代码，难以理解和调试。
**建议修复**: 生成临时 Python 脚本文件执行，或使用 `python -c "import pkg; print('OK')"` 单独检查。

### M4. SmartDocking.run 步骤编号不一致
**文件**: `smartdocking/pipeline.py:85-169`
**描述**: 日志显示 "[1/5]" 到 "[5/5]"，但实际有 6 步（包括 [6/6] 复合物生成）。且步骤 6 仅在无错误时执行。
**建议修复**: 将日志改为 "[1/6]" 到 "[6/6]"，或将步骤 6 合并到步骤 5。

### M5. KnowledgeStore 在每个 ReAct 迭代中重新实例化
**文件**: `nlp/agent.py:525`
**描述**: `ks = KnowledgeStore()` 在 `run_loop()` 的循环内部创建。KnowledgeStore 初始化可能涉及文件 I/O（读取知识库），每轮重建浪费资源。
**建议修复**: 将 `KnowledgeStore` 提升为 `AgentExecutor` 的实例属性，在 `__init__` 中初始化一次。

### M6. split_complex 不处理多链配体
**文件**: `gromacs_md/prepare.py:96`
**描述**: `resname = list(ligand_resnames)[0]` 仅取第一个配体残基名。如果 PDB 包含多种配体（如底物+辅因子都被误判为配体），只有第一个被使用，其他被静默丢弃。
**建议修复**: 返回所有检测到的配体残基名，让调用方选择。

### M7. _generate_index_python 硬编码离子名称
**文件**: `gromacs_md/prepare.py:581`
**描述**: `res in ('SOL', 'NA', 'CL', 'Na+', 'Cl-')` 不包含 CHARMM 格式的 'SOD', 'CLA', 'POT' 等离子名，也不包含 Mg²⁺、Zn²⁺ 等二价离子。
**建议修复**: 使用更完整的离子名集合，参考 `split_complex._SKIP_RESNAMES`。

### M8. QM 脚本模板可能包含用户提供的路径
**文件**: `qm_engine/bridge.py` (所有 _gen_*_script 函数)
**描述**: f-string 模板直接嵌入用户提供的路径（如 xyz_path, output_dir）。虽然 WSL 隔离降低了风险，但路径中的引号或特殊字符可能破坏生成的 Python 脚本。
**建议修复**: 对路径进行转义处理，或在脚本中通过 `sys.argv` / 环境变量传递。

### M9. heme_topology _find_gro_atom 使用枚举索引 (1-based) 但无文档
**文件**: `gromacs_md/heme_topology.py:273-277`
**描述**: `enumerate(atoms, 1)` 返回 1-based 索引以匹配 GROMACS 原子编号，但这与 Python 惯例不同，容易在后续使用中产生 off-by-one 错误（如 `gro_atoms[fe_idx - 1]` at line 500）。
**建议修复**: 添加函数文档说明返回值为 GROMACS 1-based atom number。

---

## 四、LOW 级别（6 项）

### L1. LLMResponse 缺少 `__repr__` 方法
**文件**: `nlp/backends.py:36-43`
**描述**: 调试时打印 LLMResponse 对象不显示有意义的信息。
**建议**: 添加 `__repr__` 显示 stop_reason、工具调用数和文本长度。

### L2. AgentExecutor._extract_paths 路径检测过于宽松
**文件**: `nlp/agent.py:446-453`
**描述**: 检查 `'/' in v or '\\' in v` 后再检查文件扩展名，可能错误匹配 URL、错误消息中的路径片段等。
**建议**: 使用 `os.path.isabs()` 或正则匹配常见科学文件后缀。

### L3. CondaSandbox 硬编码环境路径
**文件**: `envs/sandbox.py:27-43`
**描述**: 环境路径硬编码为 `E:\conda\envs_dirs\*`，在其他机器上不可用。
**建议**: 从配置文件或环境变量读取，回退到 `conda info --envs` 动态发现。

### L4. SmartDocking._generate_report 未处理缺失字段
**文件**: `smartdocking/pipeline.py:194-273`
**描述**: 报告模板中多处使用 `.get('key', '?')` 但格式化字符串 `f'{grid.get("center_x", 0):.1f}'` 会在值为 None 时抛出 TypeError。
**建议**: 统一使用默认值 0 或 0.0，并在格式化前检查类型。

### L5. 全局 _default_agent 单例缺少重置接口
**文件**: `nlp/agent.py:681-690`
**描述**: `chat()` 便利函数使用全局 `_default_agent`，但一旦创建后无法更换后端或重置状态（除非调用 `_default_agent.reset()`）。
**建议**: 添加 `reset_agent()` 公共函数。

### L6. Skill.check_available 仅检查 GPU
**文件**: `skills/registry.py:57-69`
**描述**: `check_available()` 仅检查 GPU 可用性，不检查 WSL、网络、外部工具路径等其他依赖。
**建议**: 扩展检查逻辑覆盖 `requires_wsl` 和 `requires_network`。

---

## 五、修复状态

### 已修复 ✓（P0 CRITICAL — 全部完成）
| ID | 修复 | 文件 |
|----|------|------|
| **C1** ✓ | NativeSandbox 命令白名单（ALLOWED_COMMANDS + _validate_command） | `envs/sandbox.py` |
| **C2** ✓ | QM GPU fallback：重构脚本模板，GPU 尝试在 D3/ddCOSMO 之前 | `qm_engine/bridge.py` |
| **C3** ✓ | 技能执行 `_run_with_timeout()` + per-skill 超时（含 verbose 分支） | `nlp/agent.py` |
| **C4** ✓ | WSL JSON 注入：改用临时文件传递（Snakemake 模式） | `envs/sandbox.py` |
| **C5** ✓ | genion 动态水分组检测（SOL/HOH/WAT/OPC/TIP3/SPC） | `mutate_minimize/engine.py` |

### 已修复 ✓（P1 HIGH/MEDIUM — 全部完成）
| ID | 修复 | 文件 |
|----|------|------|
| **BC1** ✓ | ReAct 循环执行预算：30min 超时 + 200K token 上限（Airflow 模式） | `nlp/agent.py` |
| **BC2** ✓ | OpenAI/Ollama 后端 malformed tool_call JSON 容错 | `nlp/backends.py` |
| **BC3** ✓ | 技能注册 `except ImportError` 添加 `logger.debug()` 日志 | `skills/registry.py` |
| **BC4** ✓ | SSHSandbox 可用性缓存 5min TTL（Consul 健康检查模式） | `envs/sandbox.py` |
| **AC3** ✓ | `hg1_charge` 初始化改为 0.0（安全默认值） | `gromacs_md/heme_topology.py` |
| **H6** ✓ | `get_registry()` 双检锁单例（`threading.Lock`） | `skills/registry.py` |
| **H8** ✓ | ZHUANGZHOU.md 读取添加 `errors='replace'` | `nlp/agent.py` |
| **BH2** ✓ | `_ncbi_throttle()` 线程安全（`threading.Lock`） | `skills/biodbapi.py` |
| **M1** ✓ | ThreadPoolExecutor `max_workers=min(n, 4)` 上限 | `nlp/agent.py` |
| **CC1** ✓ | SmartDocking 构造函数延迟错误（不抛异常） | `smartdocking/pipeline.py` |

### 待修复（P2 版本迭代）
- **H1**: 拓扑编辑使用正则 → 行级解析器
- **H2**: split_complex CONECT 按原子归属
- **H3**: 消息转换代码重复 → 基类共享
- **H4**: 确认门控 → 结构化确认协议
- **H5**: 上下文摘要优化 → 冷却期
- **H7**: PBC 修复后坐标验证
- **H9**: CGenFF 回退文档修正
- **H10**: SmartDocking SDF 格式验证
- **H11**: _fix_resname_gro 扩展
- **M2-M9**: 中等级别改进
- **L1-L6**: 低优先级建议

---

## 六、系统架构层面建议

### 6.1 错误传播标准化
当前错误处理模式不统一：
- gromacs_md: 返回 `(output, bool)` 元组
- skills: 返回 `{'error': ..., 'hint': ...}` 字典
- sandbox: 返回 `SandboxResult` 数据类

**建议**: 统一为 `Result[T]` 模式（类似 Rust Result），或至少保证所有公开 API 使用 `{'error': ..., 'hint': ...}` 字典格式。

### 6.2 测试覆盖
当前缺少系统化测试：
- 无单元测试覆盖 NLP 消息转换
- 无集成测试验证 CGenFF 完整工作流
- 无压力测试验证并发技能执行
- 无模糊测试验证 PDB/GRO 解析器

**建议**: 为 CRITICAL 和 HIGH 级别问题的修复同时编写回归测试。

### 6.3 配置外部化
多个模块硬编码路径和参数（sandbox 环境路径、QM 默认参数、GROMACS 超时等）。

**建议**: 统一使用 `~/.zhuangzhou/config.ini` 或环境变量，配合 `gromacs_md/config.py` 的 auto-detect 模式。

---

## 七、结论

Zhuangzhou v0.7.0 的 S58 鲁棒性工程（91 编码修复 + API 容错 + 数据安全）显著提升了系统稳定性。本次审计在此基础上发现了 **5 项 CRITICAL** 和 **11 项 HIGH** 级别问题，主要集中在：

1. **安全层面**: Shell 注入（C1, C4）— 需立即修复
2. **正确性层面**: QM 计算结果不一致（C2）— 影响科学结论
3. **可靠性层面**: 超时保护缺失（C3）、硬编码假设（C5, H2, H11）
4. **可维护性层面**: 代码重复（H3）、测试缺失

建议在 v0.7.1 中优先修复所有 CRITICAL 项和安全相关 HIGH 项，然后在 v0.8.0 中系统性解决架构层面问题（错误传播标准化、测试覆盖、配置外部化）。

---

## 附录：多智能体补充审计发现

以下为 3 个并行审计智能体的额外发现（去除与主报告重复项后）。

### 附录 A：MD 管线审计智能体（20 项发现）

#### CRITICAL 补充

**AC1. cgenff_param.py:286 — 文件句柄泄漏**
`open(mol2_path, 'rb')` 直接传给 `br.form.add_file()` 但未关闭。建议使用 context manager。

**AC2. cgenff_param.py:273,283 — ParamChem 网页爬虫对网站改版脆弱**
硬编码 URL 和表单选择器 `br.select_form(nr=0)`，网站结构变化时静默失败。建议添加表单验证和 fallback 错误消息。

**AC3. heme_topology.py:333-345 — 电荷中和逻辑在 HG1 未找到时仍转移电荷**
`hg1_charge` 初始化为 0.16，若实际 HG1 不在拓扑中则 SG 会错误吸收 0.16 电荷。建议初始化为 0.0 并设 `found_hg1` 标志。

#### HIGH 补充

**AH1. pipeline.py:385 — _fix_pbc_after_em 中 shutil.move 前不验证文件完整性**
trjconv 部分失败时可能移动损坏的 GRO 文件。建议检查行数 = natoms + 3。

**AH2. prepare.py:179-182 — SOBTOP 输出仅检查存在性，不验证内容**
空文件通过验证。建议检查 `os.path.getsize() > 100`。

**AH3. analysis.py:687 — compare_structures 错误返回字典键不一致**
后续代码期望完整 dict 结构但错误返回中缺少 `per_residue`, `figure` 等键。

#### MEDIUM 补充

**AM1. prepare.py:399-400 — merge_system 永久修改输入 ligand.itp**
`_remove_atomtypes_from_itp()` 就地修改源文件，二次运行时 atomtypes 丢失。建议操作副本。

**AM2. analysis.py:357-365 — FEL 直方图空数据导致 NaN 除法**
`np.log(0)` 被 nan 抑制，但 `np.nanmax(nan_array)` 返回 NaN → 除以 NaN。

**AM3. pipeline.py:75 — 预飞检查中 test file 在 Windows 上可能因锁定残留**

### 附录 B：NLP/技能审计智能体（14 项发现）

#### CRITICAL 补充

**BC1. agent.py:527 — ReAct 循环无计算预算上限**
20 轮 × 每轮 15 个工具 = 300 次技能调用，无 token/成本/时间预算控制。恶意 LLM 可触发无限制计算。建议添加 `execution_budget` 追踪（token 数、墙钟时间、成本估算）。

**BC2. backends.py:327 — 畸形 JSON 工具参数静默变为空 dict**
`except (json.JSONDecodeError, TypeError): args = {}` 导致工具以错误参数执行，产生难调试的 "missing required parameters" 错误。建议将解析错误作为 tool_result 返回。

**BC3. registry.py:201-258 — 模块导入失败被静默忽略**
`except ImportError: pass` 不记录哪些类别加载失败。用户不知道 QM 技能为何不可用。建议 `logger.warning(f"Failed to load {category}: {e}")`。

**BC4. sandbox.py:313 — SSHSandbox.available 缓存永不失效**
网络恢复后仍返回 False，SSH 任务永久不可用。建议添加 5 分钟过期的时间戳缓存。

#### HIGH 补充

**BH1. agent.py:469-502 — 上下文摘要可能损坏 tool_results**
摘要提示词中用字符串拼接解析 tool_use/tool_result 块，特殊字符可能导致乱码。

**BH2. biodbapi.py:82 — NCBI 速率限制器非线程安全**
`global _last_ncbi_call` 在 ThreadPoolExecutor 并发中无锁保护，多线程同时通过速率检查。建议添加 `threading.Lock`。

#### MEDIUM 补充

**BM1. backends.py:299 — 无 max_tokens 预检**
不检查 system + messages 是否超出模型上下文窗口。大对话可能在 API 端失败。

**BM2. biodbapi.py:757-826 — get_protein_profile 无错误聚合**
4 个子 API 部分失败时无整体成功/失败标志。建议添加 `profile_completeness: 3/4`。

### 附录 C：专项模块审计智能体（19 项发现）

#### CRITICAL 补充

**CC1. smartdocking/pipeline.py:42 — 构造函数抛异常而非返回错误 dict**
`raise FileNotFoundError(...)` 不符合项目"返回 error dict，不 raise"的约定。

**CC2. pymol_viz/fast_render.py:100-104 — PyMOL 子进程崩溃未捕获**
timeout 被捕获但 ProcessError/OSError 未被捕获，异常直接传播。

**CC3. ligbuilder_gen/pipeline.py:500 — WSL 超时后无进程清理**
timeout 时不调用 `process.kill()`，僵尸进程和临时文件残留。

#### HIGH 补充

**CH1. smartdocking/prepare.py:194-214 — RDKit mol 解析 None 值未检查**
`Chem.MolFromPDBBlock()` 返回 None 后继续调用 `SanitizeMol()` 导致崩溃。

**CH2. qm_engine/pipeline.py:141,160 — TS 搜索在优化失败后仍执行**
OPT 失败但 TS 仍用未优化坐标运行，必然失败且浪费计算时间。

**CH3. plip_analysis/analysis.py:52 — HETATM 修复后配体集为空时继续执行**
所有 PLIP 后端将在空配体列表上静默失败。

**CH4. smartdocking/vina.py:104 — Vina 日志解析不检查文件存在**
log 和 pdbqt 解析均失败时返回空结果而非错误 dict。

#### MEDIUM 补充

**CM1. smartdocking/site.py:74-81 — 空 ATOM 数组导致 np.array([]).mean() 崩溃**
无 ATOM 记录的 PDB 触发 ValueError。

**CM2. chimerax_viz/render.py:25-30 — GUI fallback 不检查显示可用性**
SSH 无头环境中 ChimeraX GUI 模式打开失败但无错误处理。

**CM3. mutate_minimize/engine.py:227-237 — trjconv 失败时静默回退到 editconf**
用户认为得到了溶剂化蛋白提取，实际得到完整体系（含水和离子）。

### 附录 D：边缘场景总结

| 场景 | 模块 | 影响 |
|------|------|------|
| PDB 含多种非标残基 | nonstandard_res | 参数文件互相覆盖 |
| Vina box > 60Å | smartdocking | 静默截断无警告 |
| PLIP 找到 0 个交互 | plip_analysis | 空 dict vs 错误 dict 不可区分 |
| PyMOL H-bond 阈值与 PLIP 不匹配 | pymol_viz | 可视化与报告不一致 |
| 蛋白无 ATOM 记录 | smartdocking/site | NumPy 崩溃 |
| CHARMM36 + genion | mutate_minimize | 水分组名不匹配 |
| GPU 计算路径 | qm_engine | 缺失色散/溶剂校正 |
| 并发工具调用 > 10 | nlp/agent | 线程爆炸 |

---

## 完整优先修复路线图（更新后）

### P0：立即修复（阻断性/安全性）
| ID | 问题 | 预计工作量 |
|----|------|-----------|
| C1 | NativeSandbox shell 注入 | 30 min |
| C2 | QM GPU fallback 丢失校正 | 45 min |
| C4 | WSL JSON 转义 | 30 min |
| BC2 | 畸形 JSON 静默为 {} | 20 min |
| CC1 | SmartDocking 构造函数 raise | 15 min |

### P1：近期修复（可靠性）
| ID | 问题 | 预计工作量 |
|----|------|-----------|
| C3 | 技能执行超时 | 45 min |
| C5 | genion 水分组硬编码 | 30 min |
| BC1 | ReAct 循环预算控制 | 1 hr |
| BC3 | 模块加载失败日志 | 20 min |
| BC4 | SSH 可用性缓存过期 | 15 min |
| AC3 | HG1 电荷中和逻辑 | 30 min |
| H6 | SkillRegistry 线程安全 | 20 min |
| H8 | ZHUANGZHOU.md 编码 | 5 min |
| BH2 | NCBI 速率限制器锁 | 15 min |

### P2：版本迭代（v0.8.0）
- H1-H11 其余项
- M1-M9 + 附录 MEDIUM 项
- 架构改进：错误传播标准化、测试覆盖、配置外部化

---

*本报告由 1 主审计 + 3 并行子智能体（MD管线 90K tokens、NLP/技能 80K tokens、专项模块 90K tokens）协作生成，总计分析 ~260K tokens 代码。*
