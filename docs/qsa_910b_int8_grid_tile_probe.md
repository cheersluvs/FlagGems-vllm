<!--
 Copyright 2026 FlagOS Contributors

 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

     http://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
 -->

# INT8 score grid and tile probes on Ascend 910B

## Frozen scope

| Item | Value |
| --- | --- |
| Input | Existing synthetic INT8 migration of NV workload geometry |
| Cases | All ten migrated INT8 workloads; see full regression table below |
| Runtime | Pinned CANN 9.1 / FlagTree ascend3.5 environment |
| Profiling evidence | 8dda14206542006974f4c72394e316d5d4a70782; corrected isolated trace parser |
| Hypothesis | Eight programs and 32-key tiles leave parallelism and repeated per-tile work to improve |
| Production | No changes; INT8 remains a standalone trial |

## Contract

| Item | Value |
| --- | --- |
| Q/K | INT8, H4 D128; page192 shuffled physical cache |
| Scales/weights | FP16 inputs; INT32 dot, FP32 dequantization and reduction |
| Output | FP32 scores; invisible columns negative infinity |
| CPU reference | Independent INT32 dot and FP32 reference, rtol/atol 1e-3 |
| Full comparison | Every candidate row against frozen score, full visible/mask checks |
| Postcheck | Middle chunk, complete masks plus first/middle/last CPU rows |
| No torch fallback | Device score uses the frozen Triton kernel; torch checks are outside timing |

## Paths and measurement

| Path | Grid | BLOCK_N | Interpretation |
| --- | ---: | ---: | --- |
| g8_n32 | 8 | 32 | Original score() host entry; frozen source SHA checked |
| g20_n32 | 20 | 32 | Only program count changes |
| g20_n128 | 20 | 128 | Tile width changes relative to g20_n32 |

The candidates launch the same unchanged `_quant_score_trial` JIT function with
explicit parameters. No copied kernel body or compiler modifications are used.
All three specializations must compile to one score store and INT8-to-INT32 dot.
Input shapes, chunk sizes (128/112), masks, scales, weights, padding, warp count
and scratch reuse are held fixed. The initial independent probe used G8/N128;
the current probe replaces it with G20/N128 to test additional benefit over
G20/N32. No library autotuner is added: this is a three-point
hypothesis test; production tuning/dispatch is deferred until measured results.

The default is two rounds, warmup=1 and active=3, using the existing
`do_bench_npu` timer with invocation-sum calibration. Each invocation runs the
complete score workload, summing device kernel durations; host checks/copies and
TopK are excluded. Round two reverses the order. Reports retain raw samples,
paired ratios and aggregate ranges; two rounds are exploratory evidence, not a
stability guarantee. No device exclusivity is claimed.

Speedup is frozen score time / candidate score time. These are neither speedups
against native fused QuantLightningIndexer nor whole-operator speedups. TopK
is unchanged and not timed by this focused score probe; use the previous
stage benchmark for TopK data. A successful probe is not automatically promoted
to production. Recheck promising candidates independently before integration.

Local checks: black, isort, flake8, py_compile, bash syntax, missing-prerequisite
report path on Mac (exit 2). Compilation, numeric correctness and timing require
the card run. The wrapper uploads report JSON, IR text and console output even
when the worker fails, and prints a retry command if upload fails.

## Explicit trace timer after library fallback failure

Report a229b154f8828e5071a86df4afd86b5c674184ac passed all preflight checks
but failed before timing samples: the library returned infinity after profiler
reported missing TASK data. Installing pandas resolved the preceding import
error, not the underlying collection failure. The library hides the initial
MSPTI failure reason, so its exact cause is not established.

`--timer trace` explicitly selects the CPU+NPU Chrome capture API already proven
in the frozen score profiling. It has no silent fallback and preserves each raw
trace as a top-level gzip artifact. Warmups run outside the capture; active
iterations run inside. Parsing requires both expected score host launch counts
and expected score device task counts; only MIX_AIC or named score tasks are
accepted in the isolated capture. Each result is the mean sum of device task
durations over active invocations. Calibration still tests one versus three
score calls. Counter collection is disabled. Default library timing is unchanged.

This changes collection method; rebuild all three paired baselines for the case.
Do not merge raw timings with the previous library-timer samples or claim formal
production speedup from this instrumented exploratory result. Start with only
budget_exact; long100k's existing library-timer paired result remains separate.
Local recorded-trace replay reproduced 929785.352 and 3831887.336 microseconds;
wrong counts and missing named host launches are rejected. Actual new device
capture remains to be verified on the card. No scorer, TopK, or production
implementation changes are included.


## Combined candidate follow-up

Successful trace-timer report d3212957685d120244c6521b355e5ebb93ac2c58 measured
budget_exact at 931.081 ms (G8/N32), 491.017 ms (G20/N32), and 759.847 ms
(G8/N128). All variants passed full score comparisons and postchecks. The two
individual changes are promising, but their gains must not be multiplied.

The current active set is G8/N32, G20/N32, G20/N128 on both budget_exact and
long100k, all with `--timer trace`. Keep two reversed-order rounds, warmup 1,
active 3. Both original-score and G20/N32-relative ratios are printed and saved.
The latter isolates the incremental benefit of N128 after increasing the grid.
No timer, kernel body, TopK or production source changes are included this round.
Promotion requires an independent retest and broader shape coverage after this
screen; successful exit only indicates validity of the probe, not acceptance of
the candidate. Historical library-timer long100k results are not merged with
these new trace-timer samples.


## Full ten-shape screening

The two-case combined probe in report a11f493e929e54e829d4634d654bd6e3869fc578
passed all score checks and measured G20/N128 at 3.116x and 2.996x versus G8/N32.
The next screen keeps the same three variants and explicit trace timer, and
extends only workload coverage. Default selected cases now include all ten.

| Case | Query rows | Logits width | Score chunk rows |
| --- | ---: | ---: | ---: |
| int8_first_chunk | 16384 | 4096 | 128 |
| int8_budget_exact | 16384 | 8192 | 128 |
| int8_dominant_long | 16331 | 8512 | 128 |
| int8_tail112 | 16368 | 8256 | 128 |
| int8_tail113 | 16369 | 8256 | 128 |
| int8_two_balanced | 16329 | 8256 | 128 |
| int8_two_skewed | 16384 | 8256 | 128 |
| int8_short_tail | 2316 | 8512 | 128 |
| int8_compact | 8896 | 1088 | 128 |
| int8_long100k | 16384 | 29696 | 112 |

Keep full-row score comparisons, independent CPU samples, masks, no-debug IR
checks and postchecks. Print a final per-case table and equal-weight arithmetic
mean of selected-case score speedups only if every selected case passed.
Missing/failed cases must not silently disappear into the average. The JSON
records the selected case names so subset runs cannot be mistaken for ten-case
acceptance. This does not time TopK or native fused V1 and changes no production
source, scorer source, tensor generator, timer or TopK implementation. Screening
uses two rounds, warmup 1, active 3; raw traces remain uploaded.

## Incomplete device traces during ten-case screen

Report a6f757e82bc14d419452760930b6800665860730 passed full preflight score
comparisons for all ten cases and all three variants. Six cases completed timing
and postchecks (dominant_long, tail112, tail113, two_balanced, two_skewed,
short_tail). Their combined candidate speedups range from 3.002x to 3.152x.
first_chunk, budget_exact, compact and long100k failed during timing with zero
device task events despite complete named host launch counts. Raw failed trace
inspection confirms absent Ascend Hardware events; this is not a name parser
mismatch. The underlying profiler/runtime cause remains unresolved.

The timer now releases each profiler after export and permits at most three
capture attempts for incomplete host/device task counts only. Every rejected
trace is preserved and recorded as TRACE_REJECTED in JSON/console. Successful
valid timings are never resampled or selected by speed. Unexpected device task
names, calibration failure and other exceptions still fail immediately. This
is bounded recovery from invalid captures, not a fix for the underlying CANN
collection issue or a guarantee of device isolation.

Rerun only the four failed cases with two rounds. Keep the six completed cases
as separate source-report evidence; do not report the four-case mean as a
complete ten-case acceptance. Local real-failure-trace parsing confirms rejection;
a synthetic profiler test verifies failed-first/valid-next recovery and artifact
retention. Actual retry behavior must be verified on the card. Score, TopK,
production, workload definitions, task-count validation and timing arithmetic
are unchanged.
