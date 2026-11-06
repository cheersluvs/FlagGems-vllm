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

# Ascend INT8 score throughput references

| Project fact | Value |
| --- | --- |
| Path | Current prepared query16/N64 page reuse, grid20, eight key tiles per unit |
| Runtime | Pinned 910B4 FlagTree/CANN environment from the existing prefix runner |
| Cases | Profiled bounded `budget_exact` (row 8192, 128 rows) and full-dot `long100k` (row 8176, 112 rows) |
| Timing | Interleaved device task timing with one named host launch and one device task per arm and sample |
| Integration | Diagnostic only. No production dispatch, tuning or compiler changes |

| Contract | Check |
| --- | --- |
| Current score | INT8 Q/K, INT32 dot, FP16 scales/weights, FP32 scoring and dense output with exact negative infinity at invisible positions |
| Current validation | Entire visibility mask and three CPU score reference rows per chunk at rtol/atol 1e-3 before and after timing |
| Calibration validation | Entire copy, three exact integer accumulator samples and three CPU vector-output samples including masks, before and after timing |

| Arm | Work and observed output |
| --- | --- |
| Current | Actual prepared scorer with its existing bounded/full-dot policy |
| Copy | 64 MiB contiguous INT8 source and destination, 8 KiB tiles, both read and write counted |
| Cube | Same M64/N64/K128 INT8 dot geometry, independent K panels, Q reused over eight panels, one accumulated INT32 output per eight panels |
| Vector | INT32 tile replay with original FP32 cast, scale order, ReLU, weight, head sum and visibility selection, writing one FP32 tile per dot tile |

## Static accounting

The [current score profile][profile] reports about 111 and 304 microseconds for
these chunks. The BCMLIR for both variants shows four input staging copies
(Q, QS, W and visibility) in the unit loop, a scalar page-table load in the
page loop, and a K and KS staging copy in the tile loop. Each dot tile has
a logical K transpose, an INT32 matmul, FP32 conversion, three broadcasts
feeding multiplications, ReLU, and a head reduction. The score is written
with the original row stride. The bounded kernel has a separate fill loop.
BCMLIR precedes the final device schedule and does not expose physical
placement, the number of Cube-to-Vector transfers or synchronization waits.

| Source-level work per chunk | budget_exact | long100k |
| --- | ---: | ---: |
| Issued dot tiles | 776 | 3,248 |
| Fully invisible dot tiles | 0 | 218 |
| Fill-only tiles | 248 | 0 |
| MACs issued | 406,847,488 | 1,702,887,424 |
| MACs contributing to visible scores | 403,685,376 | 1,586,006,016 |
| Fraction of MACs for invisible scores | 0.78% | 6.86% |
| K and KS requested bytes | 6,408,480 | 25,170,600 |
| Unique useful K and KS bytes | 802,880 | 3,597,360 |
| Dense FP32 score bytes written | 4,194,304 | 13,303,808 |
| Logical INT32 tile results before head reduction | 12,713,984 B | 53,215,232 B |

One MAC is a multiply plus an accumulate. A full dot tile has 524,288 MACs
and 16 KiB of logical INT32 results. The repeated K and KS payload is a count
of source requests before caching. The logical INT32 result volume does not
establish the volume transferred through GM. Page-table sites whose masks
are false count in the source-side upper byte model but may never load.

## Interpreting the calibrated target

The probe scales each component by the actual chunk work. It reports
contiguous-copy-equivalent time for the unique inputs plus dense output and
separately for source traffic without cache reuse. It reports both issued
and useful MAC equivalents and a vector equivalent for all issued dot tiles.
The `ideal_overlap_reference_us` is the maximum of perfect-reuse copy
equivalent, issued Cube equivalent and issued vector equivalent. It gives
a same-card target for the current amount of work assuming full overlap.
`current_over_ideal_overlap_reference` gives the measured scorer-to-target ratio.

These are empirical reference times. The Cube calibration amortizes its
sink and retains a loop-carried accumulator. The vector calibration reads
materialized INT32 from memory and uses tile-major output, unlike production.
The copy runs with contiguous access. The reference assumes full overlap
and does not include page lookup, partial masks, tail-fill control or mixed
pipeline dependencies. The ratio is neither a physical roofline nor a
prediction of obtainable speedup. Timings for the four arms cannot be
subtracted to attribute per-stage costs. Use a source-level bottleneck
probe only after checking the compiled arms and comparing the references.

[profile]: https://github.com/cheersluvs/FlagGems-vllm/commit/b9159913dcd9dd6a6103eab9199b5cb90b32739f

## Matched Vector output replay

The first [throughput calibration][sol-calibration] measured 111.22 and
313.29 microseconds for current score and produced 54.04 and 227.44 microsecond
ideal-overlap references. Both references were limited by the synthetic
Vector arm. That arm used a large calibration batch and a compact store.
The `--score-sol --output-replay` mode uses the actual two chunk geometries
to estimate how much of the gap comes from the output layout and tail fill.

| Arm | Output |
| --- | --- |
| Current | Existing score kernel on the same chunk |
| Compact dense | Tile-major score and fill for every output position |
| Row dense | Row-major with the original WIDTH stride, same scores and fill |
| Compact scored | Bounded case only. Compact writes for scored tiles, sentinels preserved in skipped tiles |

The three Vector replays share INT32 input tiles, query/key scales, weights,
actual per-row visibility, grid20, query16, eight-tile units and 192-key page
loop boundaries. Key scales and dot values are synthetic for replay, while the
visibility, query scales and weights are from the fixture. Replays perform the
same FP32 postprocessing and differ in output strides and, for Compact scored,
the fill loop. Complete replay outputs and masks are checked against a CPU
reference before and after timing. Compact scored also checks that every
skipped tile retains its sentinel.

Report paired `row_dense - compact_dense`, `compact_dense - compact_scored`
(bounded case), and `current - row_dense` device times. The first two
comparisons include any changes in lowering caused by layout and fill.
The last comparison is a residual against a replay that reads materialized
INT32. It includes differences in data sources and scheduling, and does not
isolate Cube-to-Vector transfer or wait time.

[sol-calibration]: https://github.com/cheersluvs/FlagGems-vllm/commit/b86c2e6653624a649ad12a6093924523f167e183

## Real INT32 handoff probe

The [matched replay report][output-replay] found a 0.47 us row-stride delta
and 0.75 us fill delta in budget_exact. The stride delta changed sign between
rounds in long100k. The fused scorer exceeded row-strided replay by 45.69 and
107.31 us in paired medians.

| Arm | Work |
| --- | --- |
| Current | Unmodified fused scorer with its real inputs |
| Producer | Original page traversal, Q/K and INT8 dot. Write each issued INT32 tile to global scratch |
| Consumer | Original page traversal, actual scales/weights, load real INT32 tiles, and run the original FP32 postprocessing and masked output |
| Split | Producer followed by Consumer on one stream. Count both device tasks in the window |

The two diagnostic kernels are generated from the current bounded and full
scorer source. The consumer retains the output coordinates and fill loop.
The producer leaves skipped tiles untouched in the bounded case. The transfer
payload is 16 KiB per issued tile in each direction, about 12.13 and 50.75 MiB
per direction for these cases. Compile, compare sampled INT32 tiles against an
independent CPU dot, compare every consumer/split FP32 value and mask with
current, and time all arms in one run. The split timing excludes the interval
between its two tasks. Compare it directly with the fused timing. The
difference includes global scratch traffic and changes in scheduling, so it
does not measure the fused kernel's internal transfer latency.

Capture the actual compiled objects with SHA-256 and try locally available
`llvm-objdump` on each object outside timing. A failed decode is recorded
and the binary is retained. Validate the architecture and opcodes in any
disassembly before using it as device instruction evidence. Earlier
TTIR/ttadapter/BCMLIR alone do not expose the final handoff schedule.
The compiler stage boundary is described in the [Triton-Ascend debugging
guide][ascend-debug].

[output-replay]: https://github.com/cheersluvs/FlagGems-vllm/commit/06eb3e25b3db6b749d6dba4db7f5a0790ffe8ac9
[ascend-debug]: https://github.com/triton-lang/triton-ascend/blob/main/docs/en/debug_guide/debugging.md

The [first handoff run][handoff-first] compiled both cases but failed the
FP32 comparison before recording any timings. The raw-tile check ran after
that comparison and did not execute, so the failure was not localized.
Both lowered sides used the same tile-major address expression. That
pre-codegen agreement does not establish a correct device store or load.

The retry uses the earlier raw INT32 probe's head-major store with a
per-query visibility select and an INT32_MIN sentinel. The consumer reads
the same address layout and preserves the original FP32 output contract.
The raw check now runs first and samples tiles on both sides of page and
work-unit boundaries, partial visibility, and the tail. On failure it
records coordinates and the actual and expected integers. Each case still
requires correct complete FP32 output before taking any timings. Preserve
objdump section and symbol output even when the local tool exposes no
instruction text.

[handoff-first]: https://github.com/cheersluvs/FlagGems-vllm/commit/ff058498e0a14c1cdfcf2f789782de591e126566

## Instruction timeline of the fused scorer

The [validated handoff report][handoff-validated] showed that the explicit
INT32 split is 1.48x and 1.81x as slow as the current bounded and full scorers.
The binary export preserved both original fused objects. The bundled objdump
recognized their ELF sections and symbols, but did not emit device instructions.

Run the same two representative chunks with `--score-instructions`. The
wrapper selects the pinned environment and discovers its `msopprof` or
`msprof op`. It first runs an unprofiled single-launch check on each chunk,
including independent CPU score samples and the complete visibility mask.
Then it runs CANN 9.1 `TimelineDetail` filtered to the exact kernel name, with
one requested launch and logical core 0 by default. The profiled launch must
have the same binary hash as the reference launch. The report also compares
that hash to the previous fused object and checks the kernel name against
OpBasicInfo when that table is available.

TimelineDetail generates a simulator-derived instruction timeline from a
device capture. The [CANN 9.1 msOpProf guide] documents direct `instrTimeLine` as a separate mode
for Ascend 950. The report preserves raw per-core traces and instruction CSVs
within its upload budget, along with a manifest and examples of wait, signal,
Fixpipe, transfer, Cube and Vector activity. Trace durations remain in the
tool's original units. Overlapping instruction durations must not be added
to infer a kernel critical path or substituted for device benchmark latency.
A zero exit from the profiler alone is insufficient. Missing instruction
evidence, target mismatch or a timeout is reported as incomplete.

[handoff-validated]: https://github.com/cheersluvs/FlagGems-vllm/commit/35abcfc0b5c22002aa6ba0460ac77bd7a527805b
[CANN 9.1 msOpProf guide]: https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/910/devaids/optool/docs/zh/user_guide/msopprof_user_guide.md

The [first TimelineDetail run][timeline-detail-first] matched the target
kernel and binary on both cases. It captured operator basic information,
but the profiler failed to save the kernel arguments and emitted no
instruction trace or CSV. The CANN 9.1 guide explicitly excludes Triton
from this on-board mode.

The runner now uses standalone `msopprof simulator` (or
`msprof op simulator`) in application mode, as described by the
[Triton-Ascend profiling guide][triton-ascend-profiling]. It checks the
simulator CLI before starting and defaults to the Ascend910B4 model and
simulated device 0. The independent correctness and binary reference still
uses the selected physical device. Only a matching simulated launch with
instruction evidence counts as captured.

[timeline-detail-first]: https://github.com/cheersluvs/FlagGems-vllm/commit/71c502af8f8600e86d68725ebe4293ac11188187
[triton-ascend-profiling]: https://github.com/triton-lang/triton-ascend/blob/main/docs/en/debug_guide/profiling.md

The [first standalone simulation][simulator-first] reached the target kernel
on both cases. The simulator reported 64 and 130 invalid load/store addresses
and neither run returned from the kernel before the outer timeout. Both
physical-device references passed and matched the historical fused binaries.
The report does not establish which argument or internal address triggered
the simulator fault.

The next diagnostic gives the per-query arguments independent NPU buffers
with storage offset zero and disables the Torch NPU allocator cache inside
the report process and its children. The tensors, launch specialization
and physical reference binary are checked before simulation. The simulator
gets its own one-minute timeout to allow it to parse partial output before
the outer watchdog. An invalid-address log always invalidates a capture.
The report now records fault counts, instruction PCs, hashes of dumped
binaries, and launch identity before device synchronization when available.
Start with budget_exact only.

[simulator-first]: https://github.com/cheersluvs/FlagGems-vllm/commit/d5664bce1972488ecaa87b7ac6971ed447ebccd4

## HIVM synchronization after the graph solver

The [independent-buffer simulator retry][simulator-base] reproduced all 64
address faults at the same PCs. The captured object matched the hardware
reference. Three fault PCs mapped to device stores whose base register was
zero. It came from the first launch-argument slot, used by the pinned
FlagTree 910B launcher for the FFTS control address. The partial trace
stopped before an executed matrix multiply.

The `--score-instructions --sync-ir` mode runs the same two score chunks
on the pinned physical-device environment. It checks the ordinary scorer
against CPU samples and all visibility masks. A second process compiles
the same scorer with an isolated Triton cache. Within the compiler module
it appends only the pinned backend's existing
`--bishengir-print-ir-after=hivm-graph-sync-solver` option to the final
BiSheng invocation. It preserves the original command and compiler output.
A successful capture requires an identified post-solver dump, correct
device output and a binary hash matching both the ordinary launch and the
historical scorer.

The summary identifies functions, static synchronization operations and
memory operations with source lines from the dump. This IR can show
synchronization and buffer relationships. Instruction frequency, overlap
and runtime wait costs require separate measurement.

[simulator-base]: https://github.com/cheersluvs/FlagGems-vllm/commit/72dd9a7fa3e4cee6e36a3eb27622e8e51c6f91b9

The [first IR capture][post-sync-first] confirmed the bounded scorer at
exact binary identity. Cube writes a 64-by-64 INT32 tile from its accumulator
to GM workspace using Fixpipe NZ-to-ND, then signals flag 1. Vector waits,
loads the 16 KiB result into UB, then signals flag 2 so Cube can overwrite
the workspace slot. Vector also passes the maximum visible length per work
unit and the physical page number per page segment to Cube through separate
scalar workspace slots. These are static data dependencies.

The long scorer's dump instead showed a four-slot ring for INT32 results.
It passed score validation but its object hash differed from the historical
binary. The `--compare-fresh-binary` control retains the historical
reference, a normal rebuild from a separate empty cache, and the dump
rebuild. It compares full hashes and fingerprints the allocated sections
and ABI records. Section comparison diagnoses a mismatch and does not
relax the exact-binary acceptance criterion.

[post-sync-first]: https://github.com/cheersluvs/FlagGems-vllm/commit/f8223be8875429308e190ccca66e2a3b54b578fa

## Precomputed maximum visibility probe

Both [bounded][post-sync-first] and [full][post-sync-long] exact-binary IR
dumps have two scalar INT32 workspace sites on each side of the Cube/Vector
boundary. One carries the group maximum visibility and the other carries
the physical page. The full scorer has a four-slot INT32 result buffer.

The `--score-sol --group-visible` probe replaces the in-kernel reduction
over 16 query rows with a load of one precomputed INT32 maximum. It retains
the same route, grid, page traversal, QK dot, FP32 scoring, per-row mask and
output layout. Both arms compile in an isolated cache with the same
post-solver print option, and the reported binaries are the ones used for
paired device timing. Record whether the candidate loses the workspace sites
and inspect the actual IR to see which data still crosses the boundary. A
scalar load may still be lowered to Vector and forwarded to Cube.

The fixture's visibility is available on CPU. Report the incremental CPU
reduction and separately the synchronized allocation and H2D cost for the
chunk metadata. The paired kernel measurement uses already prepared metadata.
Check all candidate scores and visibility masks before and after timing.
The experiment covers complete query16 groups in the two profiled chunks.

[post-sync-long]: https://github.com/cheersluvs/FlagGems-vllm/commit/7ee5420bc6fcce95cbe2fdc1b201bc1b6447ec56

## One-copy GroupVisible request probe

The [chunk probe][group-visible-chunk] removed the maximum-visibility
scalar handoff, while retaining the page-number handoff. It saved paired
medians of 7.01 and 17.37 microseconds. A separate allocation/copy/sync per
chunk exceeded those savings.

`--group-visible-request-samples` uses the current prepared selection
adapter on both arms. The candidate packs the maxima of eligible, complete
query16 groups from the CPU visibility already available during route
preparation. It aligns chunk views to 16 bytes and copies the packed INT32
metadata once per request. Chunks that miss the existing page-reuse guard
or have incomplete query groups use the original scorer. A zero-hit
request performs no metadata allocation or transfer.

Run budget_exact, long100k, and the zero-hit first_chunk. Check all FP32
scores against current, CPU samples, and full persistent TopK indices.
Measure the prepared score device task sums in alternating order. Measure
wall time in separate alternating samples, with both arms creating fresh
adapters on already available inputs and scratch, followed by one complete
score+TopK execution and a final synchronization. The wall scope includes
routing and metadata preparation. It excludes model projections, cache
updates, and the original input/output buffer allocations.

[group-visible-chunk]: https://github.com/cheersluvs/FlagGems-vllm/commit/5c703b205566f66ed07ac50fde33389fb2aa6faa

The [first request report][group-visible-request] found stable prepared score
gains of 1.029x and 1.045x, but fresh request wall time did not improve
consistently. The metadata path cost 1.91 and 2.29 ms in the two hit cases.

The revised candidate captures the exact maxima already evaluated by the
visibility policy, and packs them into one request buffer without alignment
padding or device tensor views per chunk. The candidate kernel loads from
that shared buffer using a runtime chunk offset. FlagTree's
`do_not_specialize` applies to the offset. The benchmark records distinct
compiled score binaries and checks every offset and all metadata against
independent maxima. Measure packing and allocation/copy separately from
request wall time. The packing counter excludes the route-policy calculation
that both arms perform.

[group-visible-request]: https://github.com/cheersluvs/FlagGems-vllm/commit/984a34820fa9823dcd92db2dc305cfc6de98a2a2
