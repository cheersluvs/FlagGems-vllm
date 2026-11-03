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

# Re-profile integrated INT8 score before query packing probes

## Project truth

| Item | Frozen choice |
| --- | --- |
| Target | Pinned 910B CANN 9.1 / FlagTree environment |
| Entry | PreparedInt8Selection.score_into, grid20/N128 |
| Evidence | Integrated ten-case report 9fe50d9f277f61877af3cbc06e6b3cfe20a1e8df |
| Motivation | Correctness passes, but native overall mean speedup is only 0.0530x |
| Scope | Score profiling only; no new score optimization or TopK timing |
| Cases | budget_exact and long100k, unchanged fixtures |

## Contract

| Item | Value |
| --- | --- |
| Q/K | INT8 H4 D128, page192 cache, per-query INT32 tables/visibility |
| Scales/weights | FP16, dot accumulates INT32 then FP32 score operations |
| Output | FP32 scores, negative infinity for invisible columns |
| Validation | Independent CPU references, complete representative masks, postchecks; rtol/atol 1e-3 |
| Scorer | Frozen source SHA guard remains active; debug store absent in compiled IR |

## Paths and collection

| Item | Value |
| --- | --- |
| Integrated CLI | profile_qsa_int8_score_910b.py --integrated |
| Original CLI | Without flag still profiles original grid8/N32 |
| Selection adapter | Same class as integrated stage benchmark; only score_into invoked |
| Timeline | One full logical workload; require expected score host/device launch counts |
| Hardware counters | Two calls of the same middle chunk per metric group |
| Groups | PipeUtilization, ArithmeticUtilization, Memory, MemoryUB, ResourceConflictRatio |
| IR | Same export of ttir/ttadapter/bcmlir/source where available |
| Fallback/autotune | No compute fallback or new tuning; only profiling route changes |
| Production | No registrations, source changes or BF16 dispatch changes |

The adapter constructor requires output/workspace metadata. Profiling allocates
these unused buffers, records their bytes, and passes no TopK module. No TopK
kernel is loaded or called. The timed score buffer stays bounded to 128/112
rows. Dispatch logging records the true dot shape [128,128,16], physical work
tiles and min/max per-program counts (not a false evenly-divisible count).

Unusable device timelines or exported counter archives without useful numeric
rows may retry at most three total attempts, with unique names and every attempt
recorded. Numeric counters must contain exactly the requested score-row count.
Valid captures are retained immediately; never choose captures based on speed or
utilization. Missing hardware enums and failures without exported evidence remain
explicit and are not repeatedly retried. Exit zero requires usable timeline and
PipeUtilization plus successful checks; optional group availability is reported
separately and must be inspected before drawing conclusions from it.

Inspect whether the previous high AIV scalar activity persists, the AIC MAC
activity, transfer activity, UB traffic and conflicts. Pipeline ratios overlap;
static IR token counts do not prove dynamic stall cost. Do not interpret block
coverage (cube_utilization) as MAC activity. Counter passes and instrumentation
can change durations; this run does not produce benchmark speedups.

Only after reviewing the new measurements should a query-packing/K-reuse probe
be designed. Grouping different queries must preserve their page tables,
visibility, scales, weights and request boundaries. No assumption that adjacent
query rows share a cache mapping is introduced in this round.

Local validation: formatting/lint/syntax, original recorded trace/counter parser
replay, and two-case integrated missing-prerequisite reports on Mac. Hardware
compilation/execution and actual collection remain for the card run.
