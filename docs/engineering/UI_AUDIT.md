# CMD UI 核验与修复 — 2026-10-15

基线 `ccbe823`，分支 `codex/terminal-ui-clarity`。本记录是前次工程硬化后的
UI 跟进；不把历史 CI 失败隐藏成成功，也不把渲染样例当成科学计算结果。

## 1. 用户报告的 PR run failure

重新通过 GitHub API 核验，结果如下：

| 提交 | 真实 run | 状态 |
| --- | --- | --- |
| `997e517` | [早期 PR run](https://github.com/ZiyanZhuang/zhuangzhou/actions/runs/38679190234) | failure，历史通知对应此处 |
| `00a931b` | [修复后 PR run](https://github.com/ZiyanZhuang/zhuangzhou/actions/runs/38679799941) | success |
| `1eec5db` | [PR 最终 run](https://github.com/ZiyanZhuang/zhuangzhou/actions/runs/38680154062) | success |
| `ccbe823` | [合并后 main run](https://github.com/ZiyanZhuang/zhuangzhou/actions/runs/38680414409) | success |

旧失败原因是 Python 3.10 无法解析报告模板中的 Python 3.12 f-string，以及
干净 Windows 环境未安装离线 Wheel 测试所需的构建后端。均已在此前修复。
PR #1 已合并，旧通知不会因后续成功而消失；没有重新运行旧代码或删除失败历史。

## 2. UI 事实与改动

| 观察到的问题 | 本轮修复 / 验证 |
| --- | --- |
| 80 列下路由说明、路径值折到字标下一行，破坏布局 | 60/80 列紧凑标题；100/120 列保留渐变字标；元信息用真实表格约束宽度 |
| Banner 硬编码版本；Date 是当天日期，容易误认成更新日期 | 读取运行包的 `__version__`；显示 `Started` 本地启动时间 |
| Banner 宣称 Fixed/Plan/Direct 路由，与当前 CMD App 入口不符 | 改为能力层面的 `Interface: Commands + Agent`，不声称模型已配置 |
| 帮助漏列 `/model`、`/confirm` 等，又列出没有 dispatcher 的 `/envs` | 复用实际命令注册表生成帮助；主 App 补全复用相同命令说明 |
| 启动、退出及 `/status` 会提前初始化模型 | 本地交互懒加载模型；无模型也可进入提示符、查看状态、退出 |
| `--version` 可能进入 Agent 路径 | `--version` / `-V` / `/version` 本地处理，不构造 App/模型 |
| 未知命令、错误引号、模型初始化失败可能触发无关模型调用或退出 | 显式诊断并保留提示符；已注册技能的斜杠 Agent 路径保持可用 |
| 状态摘要在异常和等待确认时也显示绿色成功勾 | 错误显示红色 `✗`，等待/中断显示金色 `◇`；Agent 返回路径同步状态 |
| 具体确认提示只作为返回值，被主 App 忽略 | 独立审查后补齐一次字面渲染的确认面板：技能、完整脱敏参数与 `/confirm` 命令；不截断待审查的调用参数 |
| 流式最终文字可能掩盖尚未解决的执行失败 | 最终执行未完成时额外显示带 `Execution incomplete` 的实际结论 |
| 把输入 + 输出 tokens 除以时间当生成速率 | 速率仅使用输出 tokens；样例 1,600 input / 400 output / 2s 应为 200/s |
| 估计上下文与流式估计 usage 缺少区分 | `ctx~` 与估计 token 数的 `~`；后端确切 usage 到达后去除估计标记 |
| `/status` 无法确认实际导入来源，Rich 版本只显示 installed | 展示真实 Package Path；缺少模块版本属性时查包元数据；路径括号按字面显示 |

**没有发布新版本或 Release，版本仍为 0.8.6。** Wheel 元数据与运行时版本有
实际构建测试约束。启动时间不等于代码更新日期；判断实际安装来源请看 `/status`。
UI 的 input 计数沿用当前最后一次请求，output 为本轮累计；不将其等同多请求账单总额。
`out …/s` 是输出 tokens 除以本轮经过时间的平均值，包含工具等待，不是纯模型解码基准。

## 3. 实际运行与视觉验证

环境：Windows 11，Python 3.12.14，Rich 15.0.0。当前执行器初始 PATH 未找到裸命令；
选定本仓库 `.venv/Scripts` 后，`where zhuangzhou` 指向该虚拟环境入口，版本为 0.8.6。
没有修改系统 PATH 或替换其他 Python 环境。
用户确认此电脑仅用于测试，尚未日常启动；没有待保留的运行实例或另一个已指定安装环境。

实际通过 CMD 启动真实 console entry point，并在 PTY 内执行：

```text
zhuangzhou --version
zhuangzhou
/version
/help
/status
/unknown-ui-command
/confirm fixture
/quit
```

观察到：启动提示符正常；版本、帮助和实际包路径正确；未知命令清晰报错；无模型时
确认命令给出明确错误且提示符仍可用；`/quit` 正常退出，返回码 0。
另用新的进程执行 `zhuangzhou /status`，观察到 Rich 15.0.0 的真实包版本。
此轮未调用模型 API：运行入口隔离到临时 home，并阻断网络，仅允许 Windows 事件循环
内部 socketpair。首次过宽的网络隔离误拦截该内部通信，修正测试隔离器后重跑成功。

使用同一 `ZZInterface` / Rich 代码生成 60、80、100、120 列的前后 SVG/PNG 样例，
固定宽高并断言真实宽度，避免宿主 `TERM=dumb` 把不同预览都固定成 80 列。
预览使用 Consolas 与中文字体回退、合成路径和明确的 fixture-model；没有真实计算结果。
已查看紧凑布局、宽屏字标、中文响应、估计标记和错误摘要。

**这些是 Rich 渲染预览，不是原生 CMD 窗口截图。** computer-use 的 Windows 规则禁止
终端 UI 自动化，因而采用真实命令/PTY与同源渲染两种证据；未核验用户任意字体、缩放
和终端主题组合。预览及隔离辅助脚本保存在忽略的本地调试目录，没有加入版本控制。

## 4. 测试与交付证据

初始新增检查：14 failed。后续调用链补测再发现 4 个失败（App 捕获异常后的假成功、
模型初始化失败退出、流式结果未纠正）；修复后相关检查通过。最终本地：

```text
python -m pytest -q --disable-warnings
242 passed, 5 skipped, 13 existing warnings in 7.15s

python -m ruff check ui app.py commands/status.py nlp skills workflow memory utils tests tools.py smartdocking/vina.py smartdocking/pipeline.py report --select E9,F63,F7,F82 --target-version py310
All checks passed!

git diff --check
passed
```

包括实际 Wheel 构建、版本一致性、安装导入、模型桩状态、确认与中断、流式结果、
输入/输出速率及四宽度布局检查。CI 的关键静态检查现覆盖 UI、App 和状态命令。
五个 live 测试仍是显式跳过；科学软件和真实模型在此 UI 跟进中没有重新运行。

独立 Spec 复核通过；Standards 复核发现上述确认面板缺口，新增真实 App → Agent →
高风险工具的回归先失败、后修复通过；模型响应明确为桩，工具未执行。
继续核验后台提交包装路径时，新增参数化测试复现 `/confirm submit_background_task`
指向错误目标；现从执行器保存的真实待确认技能和参数生成面板及命令，前台/后台两条
路径均通过，仍需用户显式确认。
初轮独立隐私检查：12 文件严格 UTF-8 / Python 3.10 语法通过，516 新增行无有效
私有路径、凭证或乱码命中。两位审查者随后批准代码 `8f08966`，各自运行相关回归
**77 passed**；补充修复再次独立通过隐私、严格 UTF-8 和 Python 3.10 语法检查。
全工程变更的 58 文件严格 UTF-8、297 个第一方 Python 语法检查通过；扫描没有读取
凭据源或忽略目录。

## 5. 有界离线并发回归与机器人测试边界

新增 `test_repeated_saturation_preserves_results_and_releases_capacity`：使用真实
`BackgroundTaskManager`，1/2/4 工作线程分别执行 8 批，每批容量 16；总计 384 次
已接受提交，288 成功、72 次预定异常、24 次排队取消。另验证 24 次超容量提交被
显式拒绝。每轮核对参数/结果对应、任务表清空与容量恢复，峰值运行数等于配置上限。
所有任务为内存中的确定性 fixture，无网络、GPU 或科学计算，等待各限 5 秒。

```text
python -m pytest -q tests/test_parallel_failures.py -k repeated_saturation --durations=3
3 passed, 12 deselected in 0.16s
```

新增测试放置时一次语法错误由 collection/lint 检出，已修正并完整重跑；没有跳过失败项。
这是执行基础设施的离线压力回归，不是吞吐优化或真实模型性能基准。
用户希望同时测试 ChatGPT 产品里的机器人，但尚未提供具体链接或可访问 API；
该真实入口测试未开始，不能用上述 fixture 结果代替。

## 6. 远端验证

[PR #2](https://github.com/ZiyanZhuang/zhuangzhou/pull/2) 的代码提交
`8f08966af173ff405e5b03ed653a44230e82dd66` 已通过四项远端检查：

| 事件 | Ubuntu / Python 3.10 | Windows / Python 3.12 |
| --- | --- | --- |
| [Push run](https://github.com/ZiyanZhuang/zhuangzhou/actions/runs/38686590167) | success | success |
| [PR run](https://github.com/ZiyanZhuang/zhuangzhou/actions/runs/38686604941) | success | success |

通过 `gh pr checks 2 --json name,state,link` 实际核验。各 job 包含完整离线测试、
Wheel 安装验证、关键 Ruff 和 CLI smoke。随后提交仅追加验证记录；合并前还需
  针对最终 PR SHA 核验全部检查，不能用上述代码提交的成功代替最终 head 状态。

## 7. 指导教师署名

启动界面在原作者信息下增加独立的 `Academic supervisors` 区块：

- Delvin Ce Zhang — University of Sheffield / School of Computer Science。
- Guang-Rong Zhao — Tianjin University / School of Synthetic Biology and Biomanufacturing。

用户确认采用 Sheffield 官网的 Delvin 拼写；指导关系由用户提供。中英文 README
在项目链接之前分别列出指导教师表格，终端预览同步更新，原插图保持不变。
没有增加新 Logo、头衔或背书措辞。

实际 Rich 渲染在 60/80/100/120 列均通过新增的署名完整性回归；生产预览和
60 列预览已目视检查，姓名与完整学院名称没有省略。UI 与 Wheel 相关检查
35 passed；全套离线检查 246 passed / 5 skipped / 13 existing warnings。
这次验证没有调用真实模型或科学软件。
