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

# Full INT8 score scale-order experiment: compile and time in one run

## Frozen project

| Item | Choice |
| --- | --- |
| Target | 910B, pinned CANN9.1 / FlagTree ascend3.5 environment |
| Evidence | 09e4821b7e2a422b87eea753202f2fbb5c9c79dc: stable budget_exact scale prefixes; long100k device trace missing |
| Question | Can equivalent full-score multiplication organization reduce the key-scale-associated prefix increment? |
| Cases | budget_exact and long100k middle chunks, original fixture bytes |
| Geometry | G20/N128, H4/D128/page192, same chunk sizes and masks |
| Scope | Full-score diagnostic candidates, not a production patch or full-prefill acceptance |

## Contract

| Item | Choice |
| --- | --- |
| Inputs | INT8 Q/K, FP16 stored query/key scales and weights |
| Dot/postprocess | INT32 dot, FP32 scale math, ReLU, weights and head reduction |
| Output | Original FP32 scores, -inf visibility mask |
| Accuracy | Sampled independent CPU reference plus all output entries against integrated baseline, rtol/atol1e-3 |
| Boundaries | Exact same queries, page mapping, keys, scales, weights, loops and output buffers' shapes |
| Unsupported | Source drift, unexpected dot/store/mode/reduction or load count, invalid scores or incomplete captures fail explicitly |

## Active paths

Let x denote the INT32 dot converted to FP32, q the query scale and k the key
scale, with the original broadcasts. All arms execute the complete score body.

| Arm | Scale expression | Purpose |
| --- | --- | --- |
| integrated_full | (x*q)*k | Current unchanged prepared adapter |
| full | (x*q)*k | Generated original-body control |
| key_first | (x*k)*q | Change broadcast/multiply order only |
| scale_product | x*(q*k) | Form scale product first |

These are equivalent over real arithmetic, not guaranteed bitwise equal in
FP32. Keep original tolerances and validate actual outputs. They do not introduce
FP8 conversion or INT8 requantization. No TopK or BF16 production source changes.
No new autotuning or registration is introduced for this standalone trial.

One user command runs generation, compilation, artifact export, numerical
validation and timing automatically. No compile-only handoff or second card run
is requested. The AST builder changes only the dequant assignment's RHS; the
remaining original kernel body is preserved, including scale loads and final
ReLU/weight/head sum. Every variant must retain the INT8 dot and one score store,
all seven loads, mixed mode and one FP32 reduction. These gates guard the intended
experiment. Source changes never patch the live production module.

Compare location-stripped ttir/ttadapter/bcmlir hashes and exact npubin hashes
when the compiler exposes bytes. Identical IR does not skip timing. Different
IR/binary hashes alone do not prove a speedup or instruction-level difference:
SSA order and embedded metadata can differ. Raw exported IR remains reviewable.

The existing single-capture timer interleaves all four arms, synchronizing at
every marked invocation. Each arm owns a separate warmed output; after capture,
compare all scores against integrated_full and inspect last timed CPU-reference
samples/masks. Preserve all valid active durations, including slow values.
Only invalid/incomplete captures retry (at most three attempts); the one/three
score-call calibration and original full-control ratio gate remain active.
Recommended run: two rounds, two warmup passes, three active invocations per arm.
Print all artifact comparisons and complete-score speedups in the same run.
This is representative-chunk exploratory device timing, not native overall or
formal full-ten benchmark acceptance. Do not interpret full variants as
cumulative prefixes: their adjacent prefix differences are intentionally omitted.

The parent also collects bounded diagnostics from its own worker's console-named
export_only_prof_dir/*_ascend_pt directories: at most five directories, 2MiB log
text and eight SQLite schema-name reads. It does not export table rows/tensors.
Missing/deleted directories are explicit. No changes to CANN/profiler libraries
are attempted, and this collection does not promise to fix missing device events.
Logs, IR, raw traces and result JSON upload together, including on failure.

Local validation: AST comparison proves only the scale expression changes;
original generated full AST remains identical. Normalization replays previous
real full/control bcmlir equivalence. Synthetic diagnostics verify log retention
and schema-only DB reads. Lint/syntax and two-case missing-dependency execution
pass. New device compilations, full-score numerical alignment and timing still
require the one card run below.
