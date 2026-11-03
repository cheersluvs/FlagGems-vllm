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

# Full-score key-scale layout including preparation cost

## Project truth

| Item | Frozen choice |
| --- | --- |
| Target | 910B, pinned CANN9.1 / FlagTree ascend3.5 |
| Evidence | 84b912de885a670894cc5f160c491560aa992239: multiplication order alone gave 0%; scale-product gave 2–3% |
| Question | Does loading an expanded per-token key scale improve the complete score after charging layout preparation? |
| Cases | budget_exact and long100k middle chunks |
| Geometry | G20/N128, H4/D128/page192; all Q/K/table/visibility/weight values unchanged |
| Scope | One-command compile, artifacts, numerical checks, device preparation/score timing, diagnostics/upload |

## Contract

| Item | Original | Layout trial |
| --- | --- | --- |
| Q/K | INT8 | Same INT8 tensors |
| Key scales | FP16 [pages,192,1] | FP16 [pages,192,16], each value duplicated 16 times |
| Query scales / weights | FP16 [rows,4] | Same tensors |
| Dot / postprocessing | INT32 / FP32 | Same types and full score semantics |
| Key-scale read | Vector token load plus column broadcast | Matrix load over token x padded-head lanes |
| Per-token scale bytes | 2 | 32 |
| Output | FP32 masked logits | Same shape/mask and rtol/atol1e-3 contract |

Expanded scales match all 16 padded-head lanes, including lanes not used by H4.
This is a data organization hypothesis, not proof of a particular DMA lowering.
There is no FP8 conversion or requantization. Keep source scale and destination
buffers disjoint. The pack helper rejects noncontiguous, non-FP16 or non-NPU
operands and validates dimensions. Production paths are not redirected.

## Measured jobs

| Job | Work in one timed invocation |
| --- | --- |
| integrated_full | Current original-layout score |
| scale_product | Original layout, scale-product arithmetic control |
| expanded_key | Expanded layout, original multiplication order; no packing included |
| expanded_product | Expanded layout, scale-product arithmetic; no packing included |
| pack_only | Duplicate every physical-cache scale on NPU |
| pack_plus_key | Pack entire physical scale cache, then expanded_key |
| pack_plus_product | Pack entire physical scale cache, then expanded_product |

The two arithmetic choices form a layout/formula control: compare expanded_key
with integrated_full, expanded_product with scale_product, and then include the
pack cost. This avoids attributing the preceding 2–3% arithmetic gain to layout.
Never present isolated expanded-score timing as the cost of the original-input
operator. Packing is charged anew for every combined representative-chunk
invocation; no unmeasured one-time/amortized cache update is assumed.

Packing is a separate persistent Triton vector kernel (grid cap20, BLOCK1024),
copying each FP16 scale to 16 output columns without rounding. It allocates no
inputs and performs no torch compute fallback. The generated full score changes
only the key_scale load layout and matching broadcast expression, optionally
using the already-tested scale-product association. The original frozen-source
hash guard remains. No production, BF16 or TopK source changes or registrations
are introduced; autotune is not added to this diagnostic experiment.

One run exports score/pack source and IR, audits seven score loads, one INT8 dot,
one score store, mixed mode and FP32 reduction, and records compiler hashes.
Every expanded scale value is compared exactly to CPU replication. All score
entries in the representative chunk are compared with integrated_full; each job
also receives first/middle/last-row independent CPU references and full masks,
before and after timing. Combined outputs are read from their last timed calls.
Metadata allocation and reference checks occur outside measurement.

The existing interleaved trace collector now optionally accepts explicit host
kernel counts per job. Default score-only behavior is unchanged. Each marked
window must contain the declared one or two device tasks; total named pack/score
host launches must match the plan. The parser still rejects unassigned/empty
windows and missing task data. The combined result is the sum of both device
kernel durations, excluding host gaps, not a per-kernel average. All valid slow
samples are retained. Invalid complete captures have at most three attempts,
with unique artifacts; the original one/three score calibration still applies.

Two rounds reverse job order, with two warmups and three active repetitions per
round. Each score job owns an output buffer. Report original/expanded scale
bytes (16x), score-output allocations, prepare-only time, isolated score time,
combined time, speedups versus integrated_full, and matched-formula layout-only
speedups. Preserve profiler logs/schema names via the same bounded diagnostics.
This is not a full-prefill, full-ten or native-overall acceptance benchmark.
Any beneficial candidate still needs independent and broader-shape validation,
plus an integration design that accounts for expanded-cache lifetime/memory.

Local checks: AST restoration proves only key-scale load/dequant expressions
change; syntax of generated layout variants; score-only timer regression;
synthetic mixed one/two-kernel windows verify preparation+score summation and
exact host counts; lint/syntax and two-case missing-prerequisite report paths.
Actual pack/score compilation, numerical checks and timing require the card run.
