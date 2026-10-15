# Engineering baseline — 2026-10-15

Started 10:50 Asia/Shanghai; maximum execution deadline 17:50. Source baseline:
`b55b88ca75f5ab12dd299bb4e7ac49a1ba78c177` (`main`, clean clone).
Work branch: `codex/runtime-hardening`. No existing user changes were overwritten.

## Environment (observed)

- Windows, 20 logical CPUs, 31.3 GiB RAM, NVIDIA RTX 5060 reported by Windows.
- Default Python 3.13.15 has no pytest/numpy/OpenAI installation.
- Existing development Python 3.12.14 provides pytest 9.1.1, numpy 2.5.2,
  pandas 3.0.5, rich 15.0.0, ruff 0.16.6.
- Local `.venv` reuses those packages; installing editable project plus NLP extras.
- No scientific executable or GPU computation has been started.
- GitHub CLI is unauthenticated. Git credential manager exists; publishing not yet verified.

## Architecture facts (source inspection; runtime verification pending)

| Path | Status | Evidence / boundary |
| --- | --- | --- |
| `app.py`, `cli.py`, `nlp/repl.py` | Implemented | Interactive and command entry points declared in pyproject |
| `nlp/agent.py`, `nlp/backends.py` | Implemented | Iterative tool calling; OpenAI/Claude adapters; verification pending |
| `skills/registry.py`, `nlp/tools.py` | Partially Implemented | Discovery, schemas, coercion, preflight; validation gaps under investigation |
| `workflow/engine.py` | Broken | Returned error dictionaries counted as completed; missing references silently become None; invalid conditions allow execution |
| `nlp/parallel.py` | Partially Implemented | Thread batches, background manager; cancellation/timeouts need verification |
| `memory/execution_history.py` | Implemented | JSONL history, checkpoints; atomicity and result semantics need verification |
| `nlp/supervisor.py` | Partially Implemented | Deterministic and model review gates; unavailable review defaults to pass |
| `report/` | Implemented | Report builders exist; end-to-end connection not yet verified |
| MD/docking/QM backends | Environment Blocked | No real simulation attempted; external software readiness unverified |

## Baseline quality checks

- Reviewed offline files (parallel, genomics, local integrations, CLI): **37 passed in 3.54s**.
- Existing agent robustness + fallback, excluding five live tests and using `-s`:
  **5 failed, 9 passed, 5 deselected**. Failures: choices validation preempted by
  file check, missing high-risk confirmation, empty in-memory execution log,
  missing unknown-tool diagnostics, missing required prompt/history context.
- New workflow reproductions against original engine: **13 failed, 1 passed**.
- New skill contract reproductions against original registry: **20 failed, 9 passed**.
- New agent runtime reproductions before agent fix: **12 failed, 1 passed**.
- Editable install with NLP extras succeeded. Syntax/CLI/wheel checks pending.
Unrestricted pytest collection was unsafe: several legacy `tests/test_cyp_*.py`
scripts perform live computation or local integration at import time. Baseline
will use explicitly reviewed offline files until test isolation is fixed.

## Initial priorities

1. P1: workflow false success, unresolved inputs, permissive conditions.
2. P1: reliable skill input normalization before scientific dispatch.
3. P1: safe offline test collection and reproducible CI.
4. P1: agent result/review propagation and persistent state consistency.
5. P2: background execution lifecycle and measured overhead/resource use.

These are source findings, not claims of completed fixes. See execution log for
red/green evidence and final report for validated scope.
