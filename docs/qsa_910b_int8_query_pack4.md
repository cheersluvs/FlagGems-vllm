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

# Shared-map four-query score diagnostic on 910B

## Project truth

| Item | Frozen choice |
| --- | --- |
| Target | 910B, pinned CANN9.1 and FlagTree ascend3.5 environment |
| Evidence | 8567ddf76d026dea77924c0d64ecc5209b05d412 rejected external 16x scale expansion, even excluding preparation |
| Hypothesis | Reuse K/table/key_scale across four queries and fill all 16 dot columns with useful heads |
| Baselines | Current G20/N128 adapter and original-layout scale_product control |
| Cases | budget_exact and long100k representative middle chunks |
| One command | Compile, export/check artifacts, numerical checks, interleaved timing, diagnostics and upload |
| Status | Standalone diagnostic; no production/TopK/cache-layout changes |

## Contract

| Item | Value |
| --- | --- |
| Q/K | Original contiguous INT8, H4/D128, page192 |
| Scale/weight | Original FP16 storage; FP32 dequant, ReLU and weighted head reduction |
| Output | FP32 [rows,width], original per-query visibility and -inf mask |
| Grouping | Up to four consecutive rows with identical complete page tables and one request owner per group |
| Mapping proof | CPU fixture table/boundaries checked before running; unequal mappings or cross-request groups rejected |
| Tail | Final group of 1/2/3 rows masks all Q/scale/weight reads and output writes |
| Accuracy | Complete output comparison with current adapter, sampled independent CPU scores; rtol/atol1e-3 |

CPU metadata proof is outside timing and is a restriction of this diagnostic.
The GPU kernel does not validate arbitrary page-table equality. This is not a
production route for unknown mappings. A future general integration must design
and validate grouping/guard/fallback costs; none are silently omitted from a
production acceptance claim here. Existing broader operators are not replaced.

## Compared paths

| Arm | Query grouping | Scale expression |
| --- | --- | --- |
| integrated_full | One Q, H4 padded to16 | (dot_fp32 * QS) * KS |
| scale_product | One Q, H4 padded to16 | dot_fp32 * (QS * KS) |
| pack4 | Four Q x H4 in16 columns | (dot_fp32 * QS) * KS |
| pack4_product | Four Q x H4 in16 columns | dot_fp32 * (QS * KS) |

Packing is implemented by indexing the original Q data inside the kernel, not
by building a new Q tensor. A group loads K and key_scale once per key tile using
the maximum of its four visible lengths. Each query retains its own QS, weights,
visibility and output row. The INT32 dot result [128,16] is interpreted as
[128,4 queries,4 heads], then FP32 reduction runs only over the four head lanes.
Store pointers map the resulting [128,4] values back to row-major query logits.
This does not assume visibility is monotonic and does not replace scales with
constants or change the data. Shared mapping is essential for maximum-visible
K reads through the group's anchor table to be valid.

Same G20/N128/num_warps4 config as the screened baseline is used to answer one
algorithm question. For the main cases the number of logical work tiles is
four times smaller; this is not a promised fourfold timing improvement. The
3D reshape/reduction and strided four-query store may introduce new layout and
synchronization costs. No autotuner is introduced in this diagnostic; promotion
and broader tuning depend on measured benefits and full coverage. Exports,
registration and production tests are not changed by this standalone trial.

Before timing, require one INT8 dot, one score store, seven loads, mixed mode
and one FP32 head reduction. The packed kernel may additionally have one INT32
visibility max reduction. Export source and ttir/ttadapter/bcmlir plus binary
hashes where available. Compare compiled artifacts and timings in this same run;
no intermediate compile-only handoff is requested.

Functional checks additionally use n=5/6/7 with width256: a shuffled two-page
mapping, nonmonotonic visibility [193,0,255,1,128,256,191], nonuniform weights
including negatives, different mappings between groups, and a fully empty
four-query group. Both scale associations are checked against CPU for every row.
Four guard rows detect stores beyond the valid final queries. These checks
reuse the same n/width specializations across the two main cases where possible.
They remain outside profiling. A failed correctness/compile gate stops timing.

The existing collector interleaves all four arms in one CPU+NPU trace, reversing
round2 order, with two warmups and three active invocations per round. Each
window must contain exactly one device kernel; named host launch counts for
_quant_score_trial and _qsa_query_pack4_score must match the plan. Retain all
valid slow samples and bounded retries only for incomplete captures. Independent
outputs preserve each arm's last timed scores for complete postchecks. The
baseline one/three-call calibration is retained. Print score times, speedups
versus integrated_full and packing-only speedups versus the matched formula.
No TopK, CANN fused baseline or full-prefill performance is measured this round.

Local validation: shared-group boundary/mapping rejection; independent packed
matrix algebra on dyadic data for tails1/2/3, shuffled page boundaries, empty and
nonmonotonic visibility, signed weights and both associations; lint/syntax and
the two-case missing-dependency report path. These prove host/algebra checks,
not NPU compilation or lowering. Device compilation, numerical references,
guard rows and timings must pass the card run before considering the candidate.
