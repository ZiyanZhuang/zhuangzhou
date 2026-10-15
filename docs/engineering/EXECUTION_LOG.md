# Engineering execution log

All times Asia/Shanghai on 2026-10-15. Scientific computations and model outputs
must be identified separately from deterministic tests and mocks.

## 10:50 — Establish isolated baseline

- Empty initial workspace; cloned the requested public repository into `zhuangzhou/`.
- Verified clean source baseline `b55b88c`; created `codex/runtime-hardening`.
- Inspected packaging, entry points, registry, workflow, parallel runtime, history,
  supervisor, and existing tests. No repository AGENTS.md or CI configuration found.
- Reused existing Python 3.12 development packages through ignored local `.venv`.
- Found import-time scientific scripts in pytest discovery; selected offline tests first.
- Initial targets: workflow false success and fail-open input/condition handling.
- Next: record baseline tests, add reproductions, fix shared boundaries.

## Round 1 — Workflow correctness

- Red: 13 failed / 1 passed in new workflow tests.
- Root causes: only exceptions were errors; missing references became None;
  invalid conditions evaluated true; progress callback could erase execution state.
- Added shared explicit result failure classifier, strict nested references,
  pre-execution ID/output-key/error-policy validation, declared optional inputs,
  fail-closed conditions and diagnostic callback isolation.
- Green: workflow plus original offline baseline **51 passed in 3.04s**.

## Round 2 — Skill trust boundary and safe test entry

- Red: 20 failures in 29 parameter-contract cases. Confirmed non-finite numbers,
  fractional integer truncation, permissive booleans, shared mutable defaults,
  incorrect enum normalization and unvalidated container types.
- Normalize then validate once before dispatch; support pathlib, canonical enums,
  finite numerics, independent defaults, required non-empty values.
- Added offline pytest boundary: five manual scientific scripts excluded from
  collection (retained as direct scripts), five model tests explicitly opt in;
  default tests block sockets and redirect agent state to a temporary home.
- Existing disabled ChimeraX manifest test now registers that manifest in an
  isolated registry; no GUI capability was re-enabled. Assertions retained.

## Round 3 — Agent execution integrity

- Red: 12 failed / 1 passed in deterministic runtime tests.
- Restored one-shot high-risk confirmation with exact pending params and trusted
  `/confirm` UI entry, memory/persistent status agreement, real execution history
  in prompt, explicit unknown-tool alternatives, positive iteration budgets and
  interruption checkpoints. Added collision-resistant session suffixes.
- Model tools cannot self-confirm. Execution success remains distinct from
  scientific validation. Model stubs are explicitly marked in tests.
- Full regression and timeout/concurrency hardening next.

## Rounds 4–5 — Bounded execution and model protocol

- Red: nine background/batch failures, including a reproduced queued-cancel
  deadlock. The initial in-process reproduction itself hung; terminated only that
  test process, then contained the regression in a five-second subprocess.
- Replaced time-derived task IDs with UUIDs, snapshotted params, bounded retained
  tasks at 32, removed redundant completion cache, isolated serial/single-call
  exceptions, and used the same explicit failure semantics across batches/status.
- Replaced per-call waiting thread pools with the existing bounded task manager.
  Wait timeout returns a tracked, possibly-running task, never claims the thread
  was killed. Unsafe skills serialize; duplicate running unsafe skills reuse the
  tracked task. Disk/log/router failures cannot erase a completed computation.
- Backend regressions: empty choices, generic 400 fallback, unfinished stream,
  stream closure, output budget, usage on final choice, manual fallback history.
  Native-mode text examples no longer execute as tool calls.
- Long-context tests reproduced orphaned tool results after both compaction paths;
  retain assistant/result pairs and use deterministic summaries on model errors.
- Unrelated chat no longer clears another interrupted request; unresolved failures
  and uncollected background work remain resumable. Background work retains its
  submission-time user context.

## Rounds 6–7 — Review, persistence, provenance, package integrity

- Red: 19 review/persistence failures. Strictly validate model review JSON/types,
  confidence and rejection flags; unavailable review blocks required gates.
  Missing PDBs/read failures and empty pose files cannot pass verification.
- Fixed pipeline propagates missing tools, exceptions, rejected gates and explicit
  result failure; executes through the agent boundary and records failed step.
- Atomic JSON replacement preserves an old checkpoint on write/replace failure;
  nested credential keys are redacted. JSON cache returns independent objects.
  History tail reads retain only the requested number of matching entries.
- Distilled KG claims remain `unverified` through storage, query, export and prompt
  patterns. Extraction queue is bounded/nonblocking; worker starts lazily.
- Real wheel tests first failed: CYPamber bridge and workflow JSON absent. Added
  missing package entries/data and tested imports with editable finder disabled.
- Consolidated duplicate tool-schema generators; arrays allow structured/numeric
  elements; report contracts retain structured and serialized JSON compatibility.
- Added explicit report extra; installed python-docx/lxml only in local environment.
- CI now targets Ubuntu/Python 3.10 and Windows/Python 3.12, with offline tests,
  installed-wheel verification, critical lint and CLI smoke. Not yet run remotely.
- Broader critical lint found pre-existing undefined `os` in Pareto harness. A
  status-only reproduction failed, then passed after the missing import fix.

## Round 8 — Real end-to-end evidence and measurement

- Deterministic model stub drives real curated table selection, rule validation,
  real DOCX generation, execution history, archive and next-context tests.
  No MD/docking/QM/PLM simulation is represented by this test.
- E2E reproduced `unknown_tool` archive names and absent result bodies. Fixed
  field mapping, seconds-to-ms conversion, per-turn archival and bounded valid
  JSON observations (raw string truncation previously broke JSON).
- Git push **dry-run only** confirmed write authentication without publishing.
- DeepSeek debugging used an opaque credential loader in an ignored local
  process; no key was printed or stored in repository files. Budget: at most four
  requests per attempt, 600 output tokens/request, 45s/request, zero SDK retries.
  Attempt 1: two requests, selection succeeded, report incomplete; original
  diagnostic unavailable. Attempts 2 and 3: three requests each, both selection
  and DOCX tools succeeded and final response completed. Total eight requests.
  Latest exact final input usage 3577 tokens, total output usage 737 tokens.
- Synthetic history benchmark: 100,000 rows / 20,988,890 bytes, last 50 requested.
  Baseline `b55b88c`: 0.807898s, 85,602,138 peak Python bytes. Current: 0.526559s,
  64,708 bytes. Results identical. Single local tracemalloc run, native memory
  excluded; no scientific/GPU throughput claim.
- First-party syntax scan: 279 tracked Python files, zero syntax failures;
  third-party vendored engines/converter excluded. CLI `python -m zhuangzhou
  --help` passed. Final comprehensive regression and review still pending.
