# Architecture findings

## Execution contracts

The active NLP loop, JSON workflow engine, public agent facade and legacy
conversation engine are separate entry paths. They share the skill registry but
do not yet share consistent failure semantics. A skill can return an error dict
without raising; the workflow currently treats that as a successful output.

References also resolve to None on missing keys, and malformed workflow conditions
default to true. These combine to dispatch downstream scientific work with invalid
inputs. Fix the existing engine and registry rather than adding an orchestrator.

## Review provenance

Supervisor model review currently defaults to passed when unavailable. Tool
execution success, structural output validity and scientific validation need to
remain distinct. Specific implementation and compatibility impacts pending tests.

## Test boundary

Legacy manual integration scripts live under pytest's default filename patterns
and can run scientific work during collection. Offline tests need an explicit,
reliable boundary before CI can be enabled safely.

## State and concurrency

Existing background manager and persistent execution history provide useful seams.
Their error, cancellation, ID and persistence behavior will be tested before any
new infrastructure is introduced.

## Implemented changes and evidence

- One failure classifier now serves workflow, foreground, parallel and background
  execution. No scientific algorithm is substituted for an execution failure.
- Foreground and background skill runs share an existing bounded pool instead of
  creating a waiting thread pool for each invocation. Waiting timeout leaves an
  explicit task handle; already-running Python code cannot be forcibly cancelled.
- High-risk execution requires a one-shot trusted UI/API confirmation. Model/tool
  output cannot authorize itself, and pending parameter changes are rejected.
- Duplicate schema generation now delegates to one converter. JSON report inputs
  keep supported object/list/string forms; list items are no longer falsely typed
  as strings for coordinate and record arrays.
- Required model review is fail-closed and labelled `model_review`; this remains a
  model judgement, not experimental validation. Structural gates check readable
  inputs and nonempty poses. Missing review never receives a passed flag.
- Checkpoints are atomically replaced, and completed prose cannot clear unresolved
  execution failures or another request's checkpoint. This is interruption state,
  not transparent process restart or automatic scientific-job resumption.
- Session archive preserves real tool names, outputs, timing and current-turn
  boundaries. Credential field redaction covers nested structured metadata.
- KG extraction stays explicitly unverified, including exports and text patterns.
  Queue capacity is bounded; original turns remain archived if extraction queue is full.
- Tests isolate state/network. Manual scientific scripts remain runnable directly
  and are excluded explicitly from automatic offline collection. CI includes real
  wheel import/data tests so editable-install masking does not hide missing assets.

## Remaining architectural limits

- Runtime cancellation cannot terminate an already-running arbitrary Python skill
  or all child processes. Scientific backends must enforce their own deadlines.
- Memory still has process-local/global registries and a shared default checkpoint;
  concurrent independent processes need a separate consistency design. Atomic
  replacement prevents torn writes but does not provide a distributed transaction.
- Fixed, legacy conversation, public facade and interactive NLP paths still differ;
  this iteration hardens shared boundaries without claiming full runtime unification.
- External simulation readiness, scientific force-field choices and real GPU
  throughput remain unverified. No live remote compute target was modified.
