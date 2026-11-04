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

## Follow-up: isolate packed postprocessing

Report a109a6921f3e7bc8b7cf0d40648ab0bf8987b9ed passed correctness but
rejected the packed implementation: budget_exact was 15.177ms versus 2.301ms
for the integrated chunk, and long100k was 48.004ms versus 8.709ms.
Its lowering added a [128,4,4] FP32 reduction and [128,4] to [4,128]
transpose, with workspace increasing from 98304 to 133184 bytes. These are
candidate-specific observations; they do not prove a particular instruction
is responsible for the integrated score's bottleneck.

Use `--postprocess` to replace the scale-order comparison with these four arms:

| Arm | Postprocessing | Expected TTIR dot/store/load counts |
| --- | --- | --- |
| integrated_full | Current single-query integrated control | 1/1/7 |
| pack4 | Original 3D reduction, mask then strided store | 1/1/7 |
| pack4_transpose | Same 3D reduction, explicit transpose before mask, contiguous key dimension | 1/1/7 |
| pack4_rows | Four masked 1D FP32 reductions over 16 lanes, four contiguous row stores | 1/4/11 |

All packed arms preserve the same K sharing, INT8 dot, launch geometry and
original `(dot_fp32 * QS) * KS` association. The row variant loads each
query's visibility separately in addition to the group max load. Its four
reductions include zeroed lanes from the other queries; this is a complete
equivalent alternative, not a measurement of one isolated stage. Explicit
transpose can lower identically to the original; normalized IR comparison
and measured times determine whether it changes anything.

The same run exports source/IR/binary hashes, prints lowering lines and
metadata, checks all main-chunk outputs, then times all four arms in the
same interleaved trace. Tail/mapping/empty-group/signed-weight checks run
for all three store modes with the original scale order only. FP32 reduction
count must be four for pack4_rows and one otherwise; visibility max may add
one INT32 reduction. Artifact auditing retains the seven-load gate by default
for existing probes, with an explicit eleven-load expectation for pack4_rows.

Two cases, two rounds, two warmups and three active samples are the default.
Print both integrated-relative speedups and packed-control-relative speedups.
No production entry, TopK implementation or compiler is changed. Reject any
candidate that fails compilation, numerical comparisons or guarded tail stores.

## Follow-up: flattened Q address and mask

Report 68fd99876af020d4682793f040e210291685f3e0 passed all comparisons.
Explicit transpose improved packed timing by 1.208x/1.205x, but remained
5.72x/4.83x slower than integrated scoring. Four separate row reductions
were slower still. All packed variants retained a pre-dot Q transpose
[4,4,128] to [128,4,4], followed by reshape to [128,16]. Its cost is not
isolated by that report.

Use `--flat-query` for three arms: integrated_full, pack4_transpose, and
pack4_flat_q. The packed control and candidate both use STORE_MODE=1,
the original scale association, and identical K/scale/weight operations.
Only Q's pointer/mask expressions change:

- Control: `(base + lane // 4) * 4 + lane % 4`, mask `base + lane // 4 < ROWS`.
- Candidate: `base * 4 + lane`, mask `base * 4 + lane < ROWS * 4`.

For nonnegative group bases and lane0..15 these are equivalent. Masked tail
lanes still load zero. This aims to expose a direct 2D Q load/transpose;
exported lowering lines determine whether it actually changes compilation.
Unchanged or still-3D lowering is recorded and timed, not treated as evidence
of an optimization. No external packing or new preparation kernel is used.

Require the same one dot, one store, seven loads and mixed-mode FP32 head
reduction for all arms. Both packed versions run n5/6/7 and empty-group checks,
including nonmonotonic visibility, page boundaries, signed weights and guard
rows. Main-chunk complete comparisons and independent CPU references remain.
Print the candidate's speedup versus the packed transpose control as well as
all three integrated-relative speedups. The single command retains artifact
export, correctness, two-round interleaved timing and automatic report upload.
Production and TopK remain untouched. Fixed G20/N128/warps4 answers this
source-expression question; the diagnostic does not add an autotuner.

## Follow-up: Q times transposed K orientation

Report 27c91051c4e83f1224342659c1177c0424f8cf0a validated flattened Q.
The matched packed speedups were 1.352x and 1.358x, but remained 3.99x
and 3.33x slower than integrated scoring. This does not justify promotion.

`--qk-direction` compares integrated_full, pack4_flat_q, and pack4_qk.
All packed inputs, shared-map guards, G20/N128/warps4, loads and original
scale association are unchanged. The candidate transposes the operands
inside the kernel and computes [16,128] Q times [128,128] transposed K.
INT32 output [16,128] is dequantized in FP32, weighted and reshaped to
[4 queries,4 heads,128 keys]. Head reduction along axis1 produces [4,128]
and stores directly in query/key order. No output transpose is expressed.
Q/K original storage is retained; there is no preparation kernel.

This borrows matrix orientation from the upstream CANN arch22 implementation,
not its FP16/Fixpipe/second-Cube weighted reduction or explicit memory pipeline.
The installed native binary is not proven source-identical to upstream master.
Precision and compiler-generated layouts must be checked, not assumed.

Require one compiled INT8 dot, one store, seven loads, mixed mode and FP32
head reduction. Source contains constexpr alternatives; only one dot is live
per specialization. Preserve all IR and print transposes/reductions/metadata,
including a surviving compiler-inserted output transpose if present. Compare
full representative chunk scores and sampled CPU references before timing,
then recheck after timing. Both packed directions run n5/6/7 and empty-group
checks, signed weights, nonmonotonic visibility and guard rows. Print matched
QK-versus-flat-Q speedup and integrated-relative speedups in the same run.
Two cases/two rounds/two warmups/three active samples remain fixed diagnostic
settings, exempt from autotuning for this direction comparison. No production
or TopK changes, and no predicted speedup or exact per-stage cost claim.

## Ten-workload restricted regression

Report f7f3464dc53d7d6c62508336357c4127c804d966 passed two-case QK checks;
matched score speedups over integrated_full were 3.393x and 3.702x.
Use `--qk-direction --ten-shapes` for all ten migrated workloads. It retains
three timed arms and original two-round protocol, compiler/precision gates,
and synthetic guarded tail checks. No scorer kernel is modified this round.

For each workload additionally compare first/last chunks against the integrated
scorer, including real final chunks of 12/32/64/73/75/112/113 queries as applicable.
Check a four-row window starting one row before the first real request boundary
when present. The CPU mapping/owner guard must run before the candidate; on
rejection the probe calls the integrated scorer and records integrated_fallback.
Also require rejection of a synthetic unequal mapping inside a same-owner group.
Boundary checks are outside timing and include full output comparisons, mask
checks and sampled CPU references. This demonstrates a probe-level fallback,
not a production dispatch integration or a measurement of dispatch costs.

Main timing still uses each workload's middle chunk with its mapping proof;
it is not a full-prefill/TopK benchmark or native-CANN comparison. The terminal
prints all per-case score times/ratios and the arithmetic mean only when every
requested case passes. Promotion needs independent evidence and an eventual
full-entry regression, including guard/fallback costs. TopK and production are
unchanged. Ten cases use separate workers and per-case timeout1200 seconds.
