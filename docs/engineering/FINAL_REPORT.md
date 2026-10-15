# Engineering final report — 2026-10-15

基线：`b55b88ca75f5ab12dd299bb4e7ac49a1ba78c177`。
工作分支：`codex/runtime-hardening`。工程提交：`dc7db85`、`7d6971a`、`d609050`、`00a931b`。
本轮预算上限 7 小时；实际执行时间以执行日志为准，不把预算写成已消耗时间。

## A. Executive Summary

本轮实际修复了 Workflow 的假成功、失效引用和宽松条件；统一 Skill 参数和
执行失败契约；修复 Agent 确认、超时、恢复、工具协议与上下文完整性；加强
检查点、历史、归档和知识来源；恢复打包后的模板和 CYPamber 导入。

代表性闭环已验证：输入 → 明确标记的确定性 Planner → 真实表格筛选 → 规则复核
→ 真实 DOCX → 执行历史与归档 → 下一轮上下文。另有受限真实 DeepSeek 调用，
两次完成筛选和报告工具调用。筛选只涉及已有元数据，不构成生物学性能验证。

两位独立只读审查者发现本轮引入的任务身份与恢复缺陷，已补失败回归、修复并
复核通过。工具执行成功、格式有效、模型复核与科学结论验证保持区分。

## B. Baseline vs Final

| 项目 | 初始观察 | 本轮结果 |
| --- | --- | --- |
| 原有离线子集 | 37 passed | 纳入完整离线回归 |
| 原有 Agent 子集 | 5 failed / 9 passed / 5 deselected | 对应问题修复，完整回归无失败 |
| 完整离线回归 | 导入期科学脚本使无差别收集不安全 | 205 passed / 5 skipped / 0 failed |
| 失败状态 | 错误字典可能标成功；缺失 Review 可能通过 | 显式失败传播；必需 Review 不可用时阻断 |
| 超时与取消 | 每次建池并等待；队列取消可死锁 | 有界共享池、可跟踪超时、诚实的排队取消 |
| 输入与结果身份 | 工具名级状态不足 | 参数类型一致才复用；调用和 task ID 关联恢复 |
| 已安装 Wheel | 缺 CYPamber 和 workflow JSON | 实际构建并脱离 editable finder 验证；干净环境补齐构建依赖 |
| 科学后端 | 未确认运行条件 | Vina/PLIP/MD/QM/GPU 仍未真实验证 |

测试集合扩大，初始子集与最终全套的分母不同，不据此计算可比通过率。
默认跳过 5 项 live API 测试；5 个有导入期副作用的手动科学脚本保留独立入口，
不在默认 pytest 收集内。13 条旧测试返回值警告仍存在，断言未删除或放宽。

合成历史读取复测（Windows / Python 3.12.14；10 万行、20,988,890 bytes、末尾 50 条）：

| 指标 | 基线 | 修改后 |
| --- | ---: | ---: |
| Wall time | 0.795544 s | 0.529168 s |
| tracemalloc 峰值 Python 分配 | 85,602,138 bytes | 64,949 bytes |
| 返回结果 | 与修改后相同 | 与基线相同 |

这是单次本地合成基准，非原生内存、真实科研吞吐或 GPU 利用率测量。
此前一次独立测量方向一致，数值保存在执行日志；不外推至其他硬件。

## C. Code Changes

| 文件/模块 | 修改及原因 | 风险和验证 |
| --- | --- | --- |
| `skills/registry.py`, `skills/results.py` | 规范化后校验、有限数值、严格布尔/容器、独立默认值、共享失败判定 | 无效旧输入现在明确拒绝；参数与状态回归 |
| `workflow/engine.py` | 预校验 ID/输出、严格引用/条件、错误字典失败、回调隔离、输入默认值 | 旧的宽松条件不再放行；正常路径和失败注入 |
| `nlp/agent.py`, `nlp/slash.py` | 一次性确认、共享任务池、真实执行日志、任务身份、错误检查点、成对上下文压缩 | 模型不能自行确认；线程运行超时不等于取消 |
| `nlp/parallel.py` | UUID、参数快照、有界保留、取消锁修复、严格 JSON 身份 | 不同输入不会串用结果；线程/Event 和子进程死锁回归 |
| `nlp/backends.py` | 空 choices、精确工具回退、流结束/关闭、输出预算和 usage | 真实模型受限 smoke 与离线协议回归 |
| `nlp/supervisor.py`, `nlp/fixed_mode.py` | 复核失败阻断、严格模型 JSON、实际 PDB/pose 检查、失败步骤传播 | 模型复核只标 model_review；规则/异常注入 |
| `memory/*`, `utils/robustness.py` | 原子 JSON、递归字段脱敏、历史有界读取、真实归档字段、KG 来源/队列/进程锁 | 无跨进程事务保证；写失败、并发、污染与来源回归 |
| `tools.py`, `skills/plm.py` | ESM 输入元数据一致性；分类后端失败显式返回 | 聚合公式保留；科学模型本身未改动 |
| `smartdocking/*`, `skills/docking.py` | 独立输出目录、当前 pose/分数检查、实际 complex、参数及 PLIP 接通 | 外部进程 fixtures；不声称真实对接验证 |
| `workflow/templates/*` | ESM 结构引用和实际 docked complex 接通 | 模板小集成验证 |
| `nlp/tools.py`, `skills/report.py`, `skills/unified.py` | 单一 schema 转换，结构化数组/JSON 报告支持 | schema 与真实 DOCX 测试 |
| `pyproject.toml`, CI, `tests/*` | 包数据、report/dev extras、离线隔离与两平台 CI | Wheel、干净 venv、CLI、参数、失败注入及 E2E |
| `report/templates.py` | 修复两处 Python 3.12 专属 f-string，恢复声明的 Python 3.10 支持 | py310 Ruff 先失败后通过；E2E 和真实 CI |
| `skills/workflow_skill.py` | 修复状态查询中的缺失 os 导入 | 确定性 status-only 回归 |

兼容性说明：低层 `tools.load_esm2_results` 继续返回列表；注册 Skill `esm2_load`
改为 `{mutations, n_mutations, sources}`，避免原先列表违反 Agent 的字典输出契约。
模板已同步；直接调用该 Skill 的外部消费者需读取 `mutations`。
`docking_only` v1.1 使用 native SmartDocking 的输出，不再提供旧的未使用 grid/complex
步骤输出名。`run_vina=False` 明确拒绝，不再静默忽略“跳过计算”要求。

## D. Agentic Infrastructure Improvements

- 复用已有 Registry、BackgroundTaskManager、Checkpoint 和 Archive，未引入新编排框架。
- 统一前台、后台、批次、Workflow 的失败语义，减少跨层假成功。
- 共享有界执行池；保留任务 ID、排队/完成/失败状态与输入快照。超时保留可收集句柄。
- 模型输出不再自动充当高风险操作授权；最终文字不能擦掉未解决失败或未收集任务。
- 历史与归档记录实际工具、参数、耗时、输出引用和结果；执行成功标 execution_only。
- KG 蒸馏保持 unverified，队列拥塞不阻塞 Agent；进程内读改写不再覆盖并发新增证据。
- 包安装与默认离线 CI 的边界可重复验证。实际应用入口仍走现有交互式 NLP 路径。

## E. Test Evidence

在仓库根目录执行；以下命令不需要私人目录或模型凭证：

```powershell
.venv/Scripts/python.exe -m pytest -q -ra
.venv/Scripts/python.exe -m ruff check nlp skills workflow memory utils tests tools.py smartdocking/vina.py smartdocking/pipeline.py report --select E9,F63,F7,F82 --target-version py310
.venv/Scripts/python.exe -m zhuangzhou --help
.venv/Scripts/python.exe -m pip check
.venv/Scripts/python.exe scripts/benchmark_history.py --entries 100000
git diff --check
```

最终本地完整回归 205 passed / 5 skipped / 13 warnings（7.31s），独立复核相关
56 项通过。Wheel 测试执行真实 `pip wheel --no-deps --no-build-isolation`，
解包后禁用 editable finder，验证模块、4 个模板及 parts 数据。关键 lint、CLI、
依赖一致性与最终 296 个第一方 Python 文件 AST 扫描通过。AST 扫描排除第三方
vendored engines 和 CGenFF converter（其中保留旧 Python 2 脚本）。

真实模型调试使用不输出凭证的临时进程：每次最多 4 请求、600 输出 tokens/请求、
45 秒/请求、SDK 自动重试为零。共 8 请求：第一次筛选成功但报告未完成，原诊断
未留存；后两次各 3 请求完成筛选与真实 DOCX。最后一次 final input 3577 tokens，
累计 output 737 tokens。离线模型桩和外部进程 fixtures 均在测试中明确标记。

## F. Remaining Issues

- **未修复的系统限制**：任意运行中的 Python Skill 不能被强制终止；底层科学进程
  需要自行实现 deadline 和进程树回收。队列取消只保证尚未开始的任务。
- **持久化技术债**：默认检查点及 MemoryRegistry 仍有全局/进程边界；原子替换和
  进程锁不构成跨进程一致性协议。后台任务句柄仅在当前进程有效，不能透明重启续跑。
- **架构差异**：fixed、legacy conversation、public facade 和交互式 NLP 尚未统一。
  本轮修复共享边界，没有宣称全部入口同等验证。
- **环境阻碍/未测试**：真实 Vina、PLIP、MD、QM、ESM 推理、GPU/WSL、SSH 远端、
  大规模任务、真实生物学结论均未验证。未修改生产或真实远端配置。
- **隐私边界**：字段名脱敏降低结构化凭证泄露风险，无法保证任意自由文本不含秘密。
  提交内容独立扫描结果记录于交付；本地忽略的调试脚本和输出不进入仓库。
- **测试债**：13 条旧返回值警告；live/manual 科学测试需独立配置与预算。

## G. Recommended Next Engineering Iteration

- **P0**：本轮已验证范围未留下已知 P0；若真实后端产生错误科学文件，应先冻结该输出链。
- **P1**：逐个科学后端落实进程组超时/取消与资源预算；增加具有版本固定和小型真实数据的
  Vina → complex → PLIP 契约测试；设计 session/run 独立检查点及跨进程原子更新。
- **P1**：为其他 legacy 入口补同等确认、失败与恢复集成测试，决定是否收敛到现有执行边界。
- **P2**：为真实硬件测量任务吞吐/峰值资源，之后再引入有依据的并行或缓存；整理旧测试
  返回值和手动科学运行入口；建立更完整的科学软件版本及输入文件校验清单。

## Final delivery verification

- 本地全套：205 passed / 5 skipped / 13 warnings，7.31s；关键 lint、CLI、
  pip check、git diff --check 通过；52 个变更文件严格 UTF-8 解码通过。
- 两位独立审查者已确认原发现修复；Spec 轴最终相关 56 项通过，无新增可操作发现。
- 独立 Standards 扫描与主代理复查未发现新增私人路径、常见真实凭证格式或编码异常。
  扫描不读取凭据源，不把忽略的本地调试内容纳入提交；提交使用公开 noreply 身份。
- GitHub 现有认证已验证仓库写入/维护权限；上传后两平台 CI 结果记录在执行日志中。
- 首次远端 CI 发现 Python 3.10 报告导入及干净 Windows 构建依赖问题，已实际修复；
  初次失败保留在日志，没有以本地绿色结果代替跨平台验证。
- 代码提交 `00a931b` 的 [push CI](https://github.com/ZiyanZhuang/zhuangzhou/actions/runs/38679796199)
  及 [PR CI](https://github.com/ZiyanZhuang/zhuangzhou/actions/runs/38679799941) 全部成功：
  Ubuntu/Python 3.10、Windows/Python 3.12 两平台的 pytest、Wheel、关键 lint、CLI 均通过。
  后续仅补充文档证据，最终提交的检查状态可在 [PR #1](https://github.com/ZiyanZhuang/zhuangzhou/pull/1) 核验。
- 干净环境构建的 Wheel 含 284 个条目、约 1.07 MB；未发现本地环境、调试目录、
  日志或敏感扩展名。最终源码未包含本地调试凭据和运行产物。
