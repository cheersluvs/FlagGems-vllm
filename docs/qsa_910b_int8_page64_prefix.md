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

# Current single-page N64 scoring prefix probe

Full-input report e4c9830b89f189ec34112eaf46c93909fcd83d60 passed ten workloads.
Eight changed workloads averaged1.179x scoring speedup against previous
QK/group4 dispatch. Overall native_fused/combined averaged0.181x. This probe
locates remaining score costs on the new path; it does not time native stages.

Use budget_exact and long100k representative middle chunks, not all query
chunks. Keep N64/group8, grid cap20, warps4, page192, actual input tables/scales,
original FP32 arithmetic, default buffering and original output/mask layout.
Generate prefixes from a frozen AST hash of _qsa_qk_single_page_score. Refuse
source drift. The generated full version is AST-identical except its function
name; current invokes the actual unchanged kernel. No production, integrated
selection, TopK or compiler changes.

| Arm | Observable sink | Expected static loads/dots/stores |
| --- | --- | --- |
| loads | Per-query Q/QS/weight sums plus per-key K/KS sums | 7/0/1 |
| dot | FP32 sum of four INT32 head dot results | 4/1/1 |
| query_scaled | Sum of four dot*QS results | 5/1/1 |
| dequant | Sum of four dot*QS*KS results | 6/1/1 |
| full | Original ReLU, weight, head sum, mask and store | 7/1/1 |
| current | Actual unchanged single-page scorer | 7/1/1 |

Every arm writes a real [chunk,width] FP32 result with original -inf visibility
masks and padded-row guards. CPU checks cover first/middle/last rows and all
masks. Full/current additionally compare every score against the legacy INT8
scorer. A7-row/width256 test exercises query padding and key-block padding,
nonmonotonic visibility, then an empty first query group; check sentinel rows
for every arm. Keep rtol/atol1e-3. Compile, export/check IR and metadata, validate,
time and upload in one command. Separate output buffers preserve each last
timed result for postchecks; full/current outputs must agree after timing.

Checksum reductions/stores are included in timing. Prefix removal can change
DCE, layouts, buffering, synchronization and resource pressure. Dot prefixes
do not retain unused QS/KS/weight loads; report actual static load counts and
lowered reductions/copies/loops. No claim of isolated DMA or pure dot time.
Adjacent timing differences are diagnostic observations, not additive operation
costs; negative differences must not be clipped. Full/current pairing exposes
whether generated-name/source context alone changes lowering or timing.

Two interleaved rounds, warmup2/active3, preserve all valid slow samples. Use
existing one/three-call trace calibration and named host/task count checks
(one device kernel per arm invocation). Print per-arm kernel_us and
relative_to_current plus diagnostic adjacent deltas. Export generated source,
TTIR/ttadapter/bcmlir, binary/source hashes, raw compressed traces and JSON
summary. Automatic upload uses the existing report workflow and explicit Git
identity. This is a fixed diagnostic configuration, not an autotuned deployment.

Local checks: AST-preserved loop/address/mask prefixes, exact full clone,
source freeze, unchanged scorer, syntax/lint, shell syntax, and two missing-NPU
worker reports. These do not establish device compilation or performance.

## Cube boundary and packed-K diagnostic

Prefix report d9dd91093dfca91bf3df1f54cfd3916f1a75a31a passed both cases.
Budget_exact/current539.307us versus dot559.810us; long100k/current1712.570us
versus dot1756.373us. Full-prefix/current differed0.40%/0.06%. Adding scale,
ReLU and weights did not increase time. The load-checksum arm was AIV-only;
dot and full arms were mixed Cube/Vector with20576-byte workspace. Thus
subtracting loads from dot does not isolate Cube arithmetic or synchronization.

Use --cube-boundary to retain default two-case/two-round execution and replace
the six arms with dot, raw_i32, raw_fp32, packed_full, full and current.
No changes to the actual scorer, prepared trial adapter, TopK or compiler.

- dot: original K layout, dot followed by FP32 head-sum checksum.
- raw_i32: preserve original loop, page addressing and QK dot; write all four
  head results as INT32. Invalid positions use INT32_MIN, outside the possible
  INT8/D128 dot range. Mask every padded head and column.
- raw_fp32: same raw-head output but convert INT32 to FP32; invalid positions
  use -inf. Both raw arms write4x output bytes versus dot/full, so their timing
  difference versus dot is not a pure reduction or conversion cost.
- packed_full: explicitly transpose the entire immutable K cache from
  [pages,192,1,128] to contiguous [pages,128,192] before score timing. Load
  [128,64] directly using the same bounded scalar page ID and page-local
  offsets, eliminating the source tl.trans. Keep all score math unchanged.
  Record transpose+allocation synchronized wall time and extra cache bytes;
  verify the cache roundtrip exactly. No layout change is applied to current.
- full/current: retain generated and actual unchanged full-score controls.

Export compiler mode/workspace, copies/loops, transpose/matmul/reduction lines,
TTIR/ttadapter/bcmlir and binary hashes. Do not assume removing tl.trans removes
all hardware conversions. Packed addressing can lower poorly; valid slow
measurements remain results, not timing failures. Every arm has one live dot,
one output store; raw/dot have four static input loads, full arms seven. Raw
visibility reuses the original four-element vector, adding no input load.

CPU reference checks require exact INT32 and FP32 raw dot values for sampled
queries. All visible raw-head sums also compare exactly to the dot checksum
for every query after timing. Packed/full/current compare every main score
against legacy before timing, and packed/full compare all last timed values
against current. Retain rtol/atol1e-3 full-score checks, all masks and guarded
7-row tails/empty groups for all arms. Record actual output bytes. Preparation
wall time and representative device-kernel time have different scopes; do not
sum them to manufacture full-workload or model-inference speedup. No CANN split
or overall acceptance claim. One command compiles, checks, times and uploads.

Local checks cover generated AST loops/math/guards, original prefix backward
compatibility, exact full clone, all page-local packed addresses, raw-head
ownership/tails, static checks and two missing-NPU child workers. Device
compilation and execution remain to be tested on910B.

## Current/raw INT32 hardware boundary capture

Report f621cbb3b4ea12861586a10ccc2036c80abd14da passed both cases. Packed-full
was0.949x/0.932x versus current despite removing the BCMLIR transpose, excluding
preparation. Raw INT32/FP32/dot-sum timings were similar; all remained mix mode
with20576-byte workspace. These controls do not isolate Cube/Vector transfer
or synchronization cost. Single cold packing wall times are not steady-state
layout-conversion measurements.

--hardware-boundary selects only raw_i32 and current, mutually exclusive with
--cube-boundary. Keep original middle chunks, N64/group8 and score formula;
compile/check/time both, then collect separate hardware passes. Default metrics
are PipeUtilization, ArithmeticUtilization and Memory, two counter iterations
per arm/group. Additional MemoryUB/ResourceConflictRatio can be requested.
PipeUtilization is required. Record unsupported/missing optional groups without
inventing values; require usable numeric PipeUtilization and exact invocation
row count for both arms to return0. Failed primary collection is partial_capture.
Retain up to two attempts only when exported tables lack complete usable
metrics; never retry valid timings. Per-pass CPU/mask checks remain mandatory.

Reuse qsa_timeline_910b.collect_hardware. Generalize summarize_tables with an
optional kernel_substring argument, retaining the old scorer as default. Match
the exact requested kernel substring, choose one detailed table, preserve raw
column units, report medians/ranges/counts and retain compressed original CSVs.
Do not mix raw/current rows or summary and detail tables. Counter-profiler
latencies are separate from the ordinary interleaved timing measurement.

Export every available textual asm entry up to4MiB with truncation flags,
including lower IR/assembly when the wheel exposes it; retain asm-key inventory,
metadata and binary hashes. Static barrier/flag/fixpipe/mmad counts are compiler
artifact evidence, not measured wait times. If only TTIR/adapter/BCMLIR exists,
state that lower instruction-level evidence is unavailable. No disassembly or
hardware attribution is fabricated. Terminal prints each metric's collection
status and numeric statistics, plus the actual textual artifact inventory.
Production, TopK, compiler and kernel source remain unchanged. The original
prefix and Cube-boundary modes retain their previous defaults.

## Static inner-loop unroll probe

Hardware report41633e3a811e7f4c168b13b631a38fa04b80ecbb captured both current
and raw INT32 arms. Current aic_mac_ratio0.015/0.018, aic_scalar_ratio0.356/0.365,
aiv_scalar_ratio0.455/0.445. Removing FP32 scoring lowered vector activity but
not latency; raw remained mixed mode. Pipeline counters overlap and do not
prove a particular synchronization instruction is dominant. cube_utilization
and MAC active ratio have different definitions; do not conflate them. The
wheel exposed no lower textual assembly beyond available MLIR/source entries.

--unroll-loop isolates exactly one source change: original inner
range(KEY_TILES) becomes tl.static_range(KEY_TILES), with KEY_TILES8. Keep
outer work ownership, N64/group8, Q/K layouts, bounded scalar page addressing,
visibility masks and original full FP32 scoring. No page-offset recurrence,
extra preparation, arithmetic reassociation or compiler flag is composed.
AST checks require identity to the full clone after restoring this iterator.
This is a diagnostic source variant; current and the trial adapter are unchanged.

Use two arms, unrolled_full/current; compile/check/export/time and collect
separate counters in one run for budget_exact/long100k. Require eight static
dots/stores in expanded TTIR; record actual loads (20..28 allows reused table
loads/common subexpressions) versus seven in current. Export lower artifacts,
loop/copy/reduction counts, binary hash and resource metadata, including any
workspace/code-size changes. Resource or compilation failure is a result,
not a reason to silently alter tile sizes or disable buffering.

Retain all-score preflight versus legacy, per-arm CPU/mask checks and 7-row
partial query/key-block plus empty-group guarded tests. Compare all last timed
unrolled/current scores. Named host/task gates require one device invocation
per arm/window; code expansion must not be confused with eight launches.
Two paired rounds, warmup2/active3. Print UNROLL_SPEEDUP_VS_CURRENT; hardware
passes use the existing two-iteration collector, exact-kernel filtering and
PipeUtilization availability gate. Use PipeUtilization/ArithmeticUtilization
first to limit runtime, with optional memory groups supported. Counter-pass
durations are not benchmark times. No TopK or full-input/native-overall claim.

## Query grouping probe (2026-11-05)

Run `--query-pack-sweep` to compare the current four-query M16 kernel with
M32/eight-query and M64/sixteen-query kernels. N64, K128, eight key tiles per
work unit, grid cap 20, FP32 reduction, addressing and output semantics stay
fixed. The work grid uses each arm's query group count. CPU fixture guards
verify identical page mappings and request owners within every larger group.
These checks are outside timing; this is a representative-chunk experiment,
not a production dispatch or full-input/native comparison.

The same invocation exports compiler IR/metadata, checks all scores against
the legacy scorer, samples CPU reference rows at rtol/atol 1e-3, exercises
19-row tails including empty groups and output sentinels, then times two
rounds in opposite order. Post-timing scores must match current scores.
Console output includes kernel microseconds and current/candidate speedups.
Hardware counter passes are omitted in this screening round to reduce runtime.

Local validation: grouping-only AST transformations and partial-row ownership
passed for both arms; formatting/lint and Python compilation passed. On Mac,
both child processes report prerequisites_missing (no Torch/NPU), exit 2;
compilation, numerical correctness and performance require the card run.

## Restricted ten-shape query16 trial

The 2026-11-05 report `621fbf25bf51773a52b76cf81872061a3c2da8a7`
measured query16 speedups of 3.1654x (budget_exact) and 3.3068x (long100k)
against four-query N64 on representative chunks. These are score-only results.

`benchmark_qsa_int8_kernel_910b.py --integrated --shared-map-qk
--key-group4-dispatch --single-page64-dispatch --query16-dispatch` now tests
all ten complete inputs. The candidate kernel matches the tested generated
query16 AST. Only existing N64 chunks with identical page maps and request
owners within each sixteen-query group take this path. All other chunks use
the current implementation, including N128 and legacy fallback routes.
Preparation cost is reported separately and inputs must remain immutable.

The dispatch_score comparison is the current four-query N64 dispatch when
query16 is enabled. Native remains the fused CANN baseline; no native score
or TopK split is claimed. Candidate score, unchanged TopK and directly timed
combined execution are reported separately. Full-input scores are checked
against legacy, selection is checked against the baseline, and threshold
checks cover nineteen-row tails and groups safe for four queries but unsafe
for sixteen. Candidate M64 IR and fallback route artifacts are exported in
this same run. Local AST equivalence, mocked route guards, lint, Python
compilation and CLI checks passed; device execution remains pending.

## Integrated query16 entry (2026-11-05)

Independent full-input replay `2931dfec4e3cd11d913428fdf7978fed9e3d44d4`
confirmed budget_exact/two_balanced/long100k score speedups of
3.2266x/3.0688x/3.2949x. Candidate two-round score ranges were
0.0165%/0.1555%/0.0279%; all-chunk maximum score error was 1.5259e-5.

The tested M64 kernel now lives in `qsa_int8_query_pack4_910b.py`.
`PreparedSharedMapInt8Selection` enables proven query16 routes by default
for N64 chunks, reusing its existing CPU table copy. `query16=False` retains
the old four-query entry for paired measurements. The former candidate class
is only a compatibility subclass. This remains an unregistered prepared INT8
entry; preparation is outside device timing and requires immutable inputs.

Run the full ten-shape benchmark with `--integrated --shared-map-qk
--key-group4-dispatch --single-page64-dispatch`; no candidate switch is needed.
The old `--query16-dispatch` spelling remains accepted. All-chunk score and
selection checks, route boundary checks and IR export run before stage timing.
Local checks establish kernel AST equivalence to the device-tested candidate,
shared-map/owner guards, partial groups, explicit opt-out, unknown-chunk
fallback and one CPU metadata copy. NPU validation of the integrated entry
is pending this run.

## Visibility-bounded dot-loop probe

`--skip-invisible` compares the integrated sixteen-query N64 scorer with a
candidate whose dot loop ends at the group's maximum visible column. Each
work unit still owns eight key tiles. The remaining tiles run a store-only
loop writing -inf, including output masks; this fill is inside the timed
kernel. There is no conditional assignment in the inner loop and active
arithmetic, FP32 reduction, page addressing and layout stay unchanged.

The report prints issued/useful/skippable tile counts for the representative
chunk and full-input query16 geometry. These are theoretical counts, including
padded key tiles, rather than measured instructions or production-route counts.
The same invocation exports IR/metadata, requires the dynamic loops in TTIR,
checks complete chunk scores against legacy, samples the CPU reference and
checks nineteen-row tails with empty groups and output guards. Timings use two
oppositely ordered rounds. No separate hardware counter passes are required.

Initial cases: first_chunk, budget_exact, long100k. first_chunk currently takes
N128 in the integrated entry: its N64/query16 timings here are diagnostic only,
not an integrated-route performance claim. All cases require shared page maps
and request owners within the representative query groups.

Local validation passed AST equivalence for active scoring, dot-free fill,
empty/partial/oversized visibility coverage, lint and compilation. The Mac
smoke run correctly returns prerequisites_missing (no Torch/NPU); device
compilation, actual skip effectiveness and numerical checks remain pending.
