<!-- SPDX-License-Identifier: Apache-2.0 -->
<!-- Copyright 2026 FlagOS Contributors -->
# QSA pipeline dependency localization (2026-10-26)

## Frozen pre-code boards

|Project truth|Choice|
|---|---|
|Task|Locate why next-K async copies remain after current score/store; no optimization or acceptance claim|
|Editable|This document, compile-only cache replay driver, CPU tests and upload runner|
|Read-only|Production, configs, all probes, captured source report and its Triton cache|
|Input|Report6aa7cb8f11be6f8a455f40e649f52514eb5a145e, code63ae460, source0d7e80e2; two BF16 cases and first captured chunk per case|
|Build/validation|Python CLI/CPU tests/bash syntax/pre-commit; remote IR compilation only|
|Execution|Four fresh child compiles: original/stages3 times R32/R64; explicit cuda/SM90 target; no tensor allocation, warmup, kernel launch, timing or GPU binary load|
|Compiler|Verify original module and libtriton hashes and assembler; exact original cached TTIR/options; isolated new output cache|
|Evidence|Require replay PTX SHA equal to captured PTX; preserve pass dumps and compiler stderr; summarize matching pass sections; no claim of exact binary provenance of a C++ release checkout|
|Timeout|300 seconds per compile, terminate process group on timeout and upload partial evidence|
|Autotune|Exempt: replay only original captured options, no candidate search or production change|
|Profiler/metric|Existing TTGIR/PTX and fresh compiler pass dumps; no new hardware metrics or performance comparison|

|Contract|Choice|
|---|---|
|Semantics|No operator/API/math/dtype/layout changes; reuse captured IR, no test-tolerance change|
|Inputs|Do not generate tensor inputs or copy tensors; scalar compilation metadata only|
|Reference|Frozen production versus existing private stages3; no new reference implementation|
|Unsupported|Missing/mismatched original cache, compiler or assembler is an error; never silently regenerate a workload or use another toolchain|

|Path|Action|
|---|---|
|Coordinator|Verify report/source/runtime, locate cache by PTX hash, copy TTIR and metadata to report, launch one child per variant|
|Child|Set dump/cache environment before importing pinned FlagTree; triton.compile(IRSource,target=SM90,options=captured); inspect asm only|
|Success|Exact PTX identity and nonempty pass trace for all four compiles|
|Mismatch/failure|Nonzero; keep diagnostics; no timing/integration claims|
|Delivery|Terminal pass/event summaries, JSON identity proofs and compressed traces; safe child-bash upload on success or failure|

## Established from archived artifacts

Independent SSA and buffer-index tracing covered36 IR/PTX pairs across production,
loop stages2, stages3 and baseline controls from the explicit-K-carry report.
The next-K copy's same-iteration operands have no dependency on the current
WGMMA result or its wait result. Read/write indices alternate0/1 versus1/0 over
32 simulated iterations; K uses two distinct16KiB slots. This is a dataflow
finding, not a proof that arbitrary post-allocation instruction hoisting is safe.
The archived steady-state PTX also places next-K cp.async after output stores.

For R32, the K allocation is at shared offset32768, size32768; output layout
scratch begins at65536. For R64, the K allocation begins at65536, size32768;
output scratch begins at98304. In the stages3 candidate, the two extra256-byte
metadata allocations start at73728/73984(R32),106496/106752(R64).
Current allocation does not show scratch overlapping the K slots. That does
not prove a rescheduled kernel has identical liveness or occupancy.

The observed steady-state order is:

    wait for K[i] -> WGMMA[i] -> wait WGMMA[i]
       -> ReLU/head reduction -> layout conversion/store
       -> address K[i+1] -> copy K[i+1] -> backedge

The WGMMA wait before ReLU is a real dependency. There is no current-score SSA
edge forcing the next-K address/copy behind that wait. The unexploited placement
is an independent next-K copy before the wait, subject to correct buffer reuse,
memory effects, barriers and tail predicates. Removing waits is not a valid fix.

## Compiler-source explanation to confirm with pass traces

FlagTree0.6.1 tag7d5dbab643fd8846b88c5bb054f9953783d7be4b has compiler.py SHA256
7357b379581baf59b088e939853c57e75bffc6b158d3059eb217be972d59ee68,
exactly matching the loaded H800 Python backend. Its relevant C++ source:

- AssignLatencies.cpp distributes load latency across indirection levels:
  `(numStages-1)/(maxIndirectionLevel+1)`. More stages can add page-ID lead
  without increasing the K-to-dot distance.
- ScheduleLoops.cpp:getInitialSchedule assigns clusters in reverse stage
  order. Consumers precede producers in the emitted steady-state body.
- LowerLoops.cpp:loadRequiresAdditionalBuffer adds a WGMMA operand buffer.
  Two K slots alone do not prove useful K-copy/compute overlap.
- SoftwarePipeliner.cpp lowers loads, expands the modulo schedule, removes
  pipeline attributes, then schedules WGMMA waits.
- WGMMAPipeline.cpp routes __TLE__ builds to scheduleTleWgmmaAsyncLaunch.
  drainForMaterializedOperands inserts waits before the first user requiring
  materialized scores. It scans the existing order and does not hoist later
  independent K copies ahead of that user.

These rules explain the final IR/PTX pattern. The Python backend hash match
alone does not prove the installed libtriton was built from this exact C++
checkout. Existing reports lack intermediate IR, so the first pass responsible
for the order remains to be checked using the compile-only replay.

Pinned source:
https://github.com/flagos-ai/FlagTree/tree/7d5dbab643fd8846b88c5bb054f9953783d7be4b/lib/Dialect/TritonGPU/Transforms/Pipeliner

## Local validation and next evidence gate

Eight CPU tests pass, including immutable cache selection by PTX SHA, four-job
input preparation, explicit-target compilation through a stub compiler, mismatch
classification, required pass coverage, no-launch AST checks and archive upload.
CLI help, bash syntax and applicable pre-commit checks pass. No GPU benchmark
was run. The remote pass replay has not yet run.

Each child has a separate dump stream and cache. Standard pass dumps and
SoftwarePipeliner internal dumps retain separate origin labels; their concatenated
file order must not be interpreted as a unified chronological pass sequence.
The collector requires latency assignment, loop scheduling, LowerLoops and
ExpandLoops coverage, and exact original PTX identity. Missing original TTIR
cache is an explicit failure, not permission to recreate or retime the workload.

## Replay failure c8fa151d and fix

The first R32 baseline child reached LLVM lowering and failed in
`get_features(options, arch)`: frozen CUDAOptions hashing saw JSON lists in
supported_fp8_dtypes/deprecated_fp8_dot_operand_dtypes/allowed_dot_input_precisions.
The replay script failed to restore the tuple types. This is a diagnostic-driver
error, not an operator compilation regression. No GPU kernel was launched.

Restore the five tuple-valued CUDAOptions fields (also cluster_dims and the
nested extern_libs pairs), preserving every element, option and original JSON.
A CPU regression reproduces the frozen-options hash failure with the captured
field values, then verifies a hashable reconstruction and identical JSON values.
Nine CPU tests and applicable style checks pass after the fix.

The partial R32 trace is retained: lower-loads emits K copy at(stage0,cluster1),
dot/store at(stage1,cluster0). ExpandLoops places the steady-state dot/store
before next-K copy, before WGMMA wait insertion. This directly corroborates the
consumer-first coarse scheduling explanation. The full replay stopped before
PTX identity validation, so these remain partial localization evidence; the
other three children and complete four-way identity proof still require replay.
