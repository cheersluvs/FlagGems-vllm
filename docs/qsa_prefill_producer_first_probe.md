<!-- SPDX-License-Identifier: Apache-2.0 -->
<!-- Copyright 2026 FlagOS Contributors -->
# Producer-first schedule feasibility probe

## Pre-code project board (2026-10-26)

|Field|Frozen choice|
|---|---|
|Task/evidence|QSA score only; validated pass trace2dea5593, four exact original PTX/TTGIR replays; coarse consumer-first clustering leaves next-K copy behind dot/store|
|Editable|This board, private compile-only driver, CPU tests, upload runner|
|Read-only|Production0d7e80e2, configs, references, all prior probes, captured trace and installed compiler files|
|One hypothesis|Reverse cluster0/1 for the original two-stage production schedule, preserving stage numbers and all operations/operands; test whether next-K async copying moves before current dot|
|Active set|First captured BF16 R32 and R64 production chunk; each gets unmodified-input control and producer-first candidate|
|Build|CPU tests, CLI help, bash syntax, changed-file pre-commit; remote four CPU compilations through the pinned FlagTree library|
|Validation|Unmodified control must reproduce captured PTX exactly; MLIR verifier and full original compiler tail run; preserve WGMMA work/waits; inspect K copy placement, buffers, resources and spills|
|Benchmark|None yet. No tensor creation, GPU binary loading, kernel launch or timer; score/TopK/pipeline performance remains unmeasured|
|Gate|Zero/known spill bytes, unchanged MMA count, identifiable loop and hardware async copy; real K copy before first dot required; extra K buffers reported explicitly, not called free|
|Timeout|300s per child; own process-group termination on timeout; upload partial evidence|
|Autotune|Exempt diagnostic: original captured launch options only, no search or production config change|
|Profiler|Pass dumps, PTXAS verbose resource log and final TTGIR/PTX; no new hardware counters|
|Scope|NVIDIA compiler diagnostic adapter; not yet a portable production implementation. A successful schedule must later be expressed/ported and numerically validated|

## Contract board

|Field|Preserved|
|---|---|
|API/math/dtype/layout|All original captured operations, pointer arithmetic, masks, reductions, stores, signatures and constants remain identical; only two cluster IDs exchange|
|Memory/synchronization|Original compiler decides buffer count, async waits, fences and barriers; do not delete waits or edit generated PTX|
|Inputs|No tensor inputs or Torch compute; scalar metadata and captured intermediate IR only|
|Production paths|All public/general/compact/FP8/TopK paths unchanged and unexecuted|
|Coverage limit|Compile verification is not a proof of runtime correctness; no speedup claim or production integration|

## Implementation paths board

|Path|Action/limit|
|---|---|
|Prepare|Require successful four-way prior trace with exact PTX hashes; read original baseline pre-TritonGPUPipeline snapshot; confirm clusters/stages are exactly0/1; make control/candidate copies|
|Adapter|Hash-check original compiler.py; derive its exact AST tail starting with the existing SM90 pipeline call; retain every subsequent pass and metadata action; skip already executed prefix|
|Resume|Use a private process-local static-method replacement and explicit SM90 target; feed captured pre-pipeline IR into the unchanged pipeline and tail; no installed-file edits|
|Control|Exact original PTX equality is mandatory before interpreting the altered schedule|
|Candidate|Change only loop.cluster IDs0↔1; do not change stage numbers or insert/remove/move source operations manually|
|Resources|Enable TRITON_DUMP_PTXAS_LOG only (assembler already invokes -v); parse actual register/spill bytes and metadata shared size; never infer unknown spills as zero|
|Result|compile_feasible_pending_accuracy, not_promising or failure; report buffer growth and retain all artifacts; no automatic execution even on feasibility|
|Delivery|Terminal placement/resource results plus JSON/TXT and compressed pass artifacts; automatic upload in child bash|

The captured compiler hashes fix the Python adapter's input. C++ source suggests
producer-first ordering can increase stageDiff and add a K buffer; the actual
compiled buffer allocation is decisive. Do not clamp that allocation or remove
its synchronization merely to meet a resource budget.

Unmodified control equality checks the resumed compiler tail, including layout,
allocation, fences and target lowering. The experiment does not patch C++ code,
force CUDA policy hints or add backend instructions to the operator.

## Local validation

Seven new CPU tests plus nine existing replay tests pass. Captured production
snapshots from report2dea5593 prepare all four jobs without original-file edits:
R32 exchanges60 cluster attributes and R64 exchanges64; applying the same
exchange twice restores the original text exactly. Stage IDs, operations and
all other text are unchanged. The adapter calibrated against the pinned real
compiler.py preserves all27 top-level tail statements exactly in AST form.
CLI help, bash syntax and applicable pre-commit checks pass.

The remote resumed control compilation and candidate compilation have not run
locally. This is a feasibility experiment with runtime correctness and all
score/TopK/pipeline timing deliberately pending.
