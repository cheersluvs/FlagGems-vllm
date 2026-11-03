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
| Cases | int8_budget_exact, int8_long100k |
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
| g8_n128 | 8 | 128 | Only key tile width changes |

The candidates launch the same unchanged `_quant_score_trial` JIT function with
explicit parameters. No copied kernel body or compiler modifications are used.
All three specializations must compile to one score store and INT8-to-INT32 dot.
Input shapes, chunk sizes (128/112), masks, scales, weights, padding, warp count
and scratch reuse are held fixed. G20/N128 is deliberately excluded to isolate
one change at a time. No library autotuner is added: this is a three-point
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
