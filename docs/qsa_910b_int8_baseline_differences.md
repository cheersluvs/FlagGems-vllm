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

# Sequential CANN-method score probes on 910B

| Item | Current evidence |
| --- | --- |
| Control | QK direction, FP32 scoring, shared-map representative chunks |
| Baseline source | Locally inspected upstream CANN arch22 Cube/Vector/kernel headers, 2026-11-04 |
| Version limit | Upstream source is not proven identical to installed CANN9.1 binary |
| Production | No production/TopK/compiler changes; prepared adapter remains unchanged |

| Order | Difference | Isolated proposal |
| --- | --- | --- |
| Done | Matrix direction | Q times transposed K; full-input score mean3.452x vs legacy |
| 1 | Key scale per head vs after weighted reduction | Move only KS multiplication, keeping original QS and FP32 reduction |
| 2 | Repeated QS and weight multiplication vs combined coefficient | Separate later probe; rounding and ReLU constraints need checks |
| 3 | Q loaded per key tile vs L1 reuse across key scan | Later loop/reuse probe; no compiler patch |
| 4 | Vector head sum vs second Cube weighted matmul | Later algorithm/precision probe; FP16 intermediates cannot be assumed equivalent |
| 5 | N128 tile vs larger key block/pipelined native loop | Later tiling/dispatch probe after addressing redundant operations |

| Contract | First probe |
| --- | --- |
| CLI | --late-key-scale, default two representative workloads |
| Arms | integrated_full, pack4_qk, pack4_qk_late_ks |
| Control formula | sum_head(ReLU((dot_fp32 * QS) * KS) * W) |
| Candidate formula | sum_head(ReLU(dot_fp32 * QS) * W) * KS |
| Precondition | KS finite and nonnegative, checked on all CPU fixture scales before device run |
| Precision | INT8 dot to INT32; all following math FP32; no intermediate FP16 or second dot |
| Shape | Same shared-map H4/D128/page192, G20/N128/warps4 diagnostic config |

CANN ProcessVec1 multiplies key scale after Cube has performed weighted head
aggregation. This trial borrows only that placement, not CANN Fixpipe conversion
or weight/QS preparation. The coefficient is independent of head, and ReLU(a*s)
=ReLU(a)*s for s>=0. Negative weights and query scales do not invalidate this
identity. Floating-point reassociation can change rounding, so strict existing
rtol/atol1e-3 comparisons remain mandatory. Negative/nonfinite KS is rejected
by this diagnostic, not silently accepted or restricted in existing adapters.

All output scores in both main chunks are compared against the legacy scorer,
with independent CPU score samples and masks. Both QK variants additionally run
n5/6/7, shuffled pages, different maps between groups, nonmonotonic visibility,
empty groups and guard rows. For this probe small tests set each page's first
key scale to zero, alternate QS signs and retain signed weights; both variants
see the same modified tensors. All checks precede timing and postchecks remain.

Require one compiled INT8 dot, one store, seven loads, mixed mode, FP32 head sum.
Export IR/source/hash/metadata and print multiplication/reduction/transpose lines.
The candidate removes KS multiplication on [16,N], adding it on [4,N]; actual
lowering and timing decide whether this matters. Fixed configuration is an
explicit diagnostic autotuning exemption. Run compile, correctness, two-round
interleaved timing and report upload together. No TopK/full-prefill acceptance
claim; a winning probe needs independent and broader integration validation.

## Second probe: combine QS and weight in FP32

The first probe report 3e13a0fa0a548384a97940500788a211b60b8d25 passed,
with matched late-KS speedups1.0138x/1.0119x. It remains a candidate only.
This next probe does not compose late-KS with coefficient combination.

| Item | Independent coefficient probe |
| --- | --- |
| CLI | --combine-query-weight |
| Arms | integrated_full, pack4_qk, pack4_qk_query_weight |
| Control | sum_head(ReLU((dot_fp32*QS)*KS)*W) |
| Candidate | sum_head(ReLU(dot_fp32*KS)*(QS*W)) |
| Preconditions | All QS finite/nonnegative, CPU fixture proof outside timing |
| Key scale | Still before ReLU and head reduction; negative KS is supported |
| Coefficient | 16 FP32 products within kernel, no FP16 round trip or extra preparation kernel |
| Scope | G20/N128/warps4 fixed diagnostic; original grouping and launch schedule |

For QS>=0, ReLU(x*QS)=QS*ReLU(x). The weights may be negative and must not
move inside ReLU. The candidate reduces one [16,N] product to one [16]
product. CANN ProcessVec0 prepares weight times QS in FP16 for reuse; this
probe isolates combining coefficients in FP32. It does not reproduce the
native FP16 preparation or claim reuse across key tiles. These are separate
precision/reuse questions for later experiments.

All main outputs and masks compare against legacy scores with independent CPU
samples, rtol/atol1e-3. Rounding may differ and any failed precision gate stops
timing. Small n5/6/7 and empty-group checks use zero alternate QS heads, signed
KS at each page's first key and signed weights, identical for both variants.
Guard rows and shuffled/nonmonotonic visibility checks remain. Negative QS
is rejected by this diagnostic, without restricting production or adapter inputs.

Keep one compiled INT8 dot, seven loads, one store and FP32 head reduction.
Print multiply/reduce/transpose lines to confirm vector coefficient formation.
Compile, numerical gates, two-round paired timing and upload stay in one run.
Report QUERY_WEIGHT_SPEEDUP_VS_QK separately from the legacy integrated ratio.
No TopK, compiler or prepared selection adapter changes.

## Third probe: reuse Q group across key tiles

Report 66a59bad936f4413dfd5bd14d35698d70e9550e1 passed coefficient checks,
with speedups1.0135x/1.0221x; no candidate is composed or promoted here.

| Item | Q reuse experiment |
| --- | --- |
| CLI | --query-reuse |
| Arms | integrated_full, pack4_qk, pack4_qk_reuse |
| Formula | Original FP32 QS then KS before ReLU, original weighted FP32 head sum |
| Control schedule | Strided flattened query-group/key-tile work across programs |
| Candidate schedule | Strided query groups across programs, sequential key tiles within each group |
| Load reuse | Q/QS/weight/visibility once per outer group, table/K/KS per key tile |
| Resources | Grid cap20, N128, warps4; no manually specified L1/L0 allocation |

The candidate expresses Q[16,128] and K[128,N] directly, algebraically identical
to the flattened control's transposed operand views. It changes loop ownership
and lifetime of input tensors, not the score formula or input layout. Compiler
lowering still decides whether reuse remains effective in device memory; source
placement alone does not prove L1 residency or final instruction reduction.
Exported TTIR loop/load lines and full ttadapter/bcmlir are retained. For both
main chunks the launch remains20 programs, but 28/32 query groups may distribute
unevenly; loss of balanced key-tile scheduling is an explicit tradeoff. All width
tiles, including invisible ones, are still processed and written, so no work is
silently skipped. Tail tests use min20/groups programs; tail timing is excluded.

The static contract remains seven loads, one dot, one score store, FP32 head
sum and visibility INT32 max. Compile, full output/CPU/mask checks, nonmonotonic
visibility, shuffled mapping, signed weights and signed QS, empty groups, query
tails1/2/3, guard rows, two-round timing and upload run together. A scope-preserving
loop rewrite can still change lowering and synchronization; timing compares the
entire equivalent score kernel rather than claiming a separately measured load
cost. Print QUERY_REUSE_SPEEDUP_VS_QK. No arithmetic candidate, production adapter,
TopK, external preparation kernel, compiler patch or autotuner is introduced.

### Repair the reuse candidate's K-load confound

Report daab0c18ebdc26b943c2edc4d253809b46511cbf passed correctness, but
reuse measured33.551ms/150.486ms vs0.670ms/2.333ms. TTIR confirmed Q-related
loads outside the inner tile loop. However, direct [D,N] K loading introduced
an additional change: bcmlir contains nested D/key loops and scalar one-byte
memref.load/tensor.insert. The control loads each token's128-byte row with
memref.copy, followed by linalg.transpose. Therefore those timings cannot
attribute the regression to Q reuse alone.

The repaired reuse candidate restores token-major K load followed by explicit
transpose for dot, matching the control's addressing/mask and matrix orientation.
All loop ownership, outer-group loads, numerical gates, launch configuration
and timing remain the same. Re-run compile/correctness/timing together using
--query-reuse. Inspect whether the continuous row-copy lowering is restored
before interpreting matched timings. The previous valid slow samples remain
in the earlier report; no timing filters or resampling policy changes are made.

## Fourth probe: second Cube head aggregation

Report d6a9bd1686b30fbb069787c54372bf6f17531daf restored token-major K
loading and passed correctness, but group-loop reuse was slower (0.921x/0.770x).
Keep the existing flattened QK schedule for the next independent experiment.

| Item | Native-inspired Cube reduction probe |
| --- | --- |
| CLI | --cube-reduce |
| Arms | integrated_full, pack4_qk, pack4_qk_cube |
| Dot1 | Same INT8 QK to INT32 with original K row loading/transpose |
| Intermediate | ReLU(raw INT32 dot /1024), cast FP16 |
| Coefficient | QS*W in FP32 then cast FP16, matching FP16 product storage |
| Dot2 | FP16 block-diagonal head weights times FP16 activation, FP32 accumulation |
| Output | Multiply KS then1024 in FP32, restoring original score units |
| Preconditions | QS/KS finite nonnegative; coefficient finite and representable as FP16 |

The inspected native arch22 source FixpSToL1 uses DEQF16, reluPre=1 and
SetFixpipePreQuantFlag(0x3a800000), whose FP32 value is1/1024. ProcessVec0
forms FP16 QS*weight; ComputeWs uses half inputs and float accumulation;
ProcessVec1 applies KS after weighted aggregation. Source correspondence to
the installed CANN binary remains unproven. This experiment borrows those
arithmetic choices; it is not a byte-for-byte native implementation or proof
of native instruction scheduling. Restoring1024 compensates the common scale
for full-score comparison rather than relying solely on TopK invariance.

The second dot uses [16,16] weights, with four block-diagonal active rows and
zeros elsewhere, multiplying [16,N] activation. Each active row selects exactly
its query's four head lanes. This keeps minimum Cube-compatible dimensions and
computes all four query sums in one padded matmul. Native code instead invokes
its own per-query weighted matmul/format conversion; performance is not promised
equivalent. Output masks exclude the twelve padded query rows and invalid tails.
The visibility expansion adds one INT32 reduction, not an FP32 head sum.

Compiler gates require two live dots, seven loads, one store, zero FP32 reductions
and mixed mode. Both first-dot INT32 and second-dot FP32 accumulators are explicit
in source; full IR is exported before numerical checks, so precision rejection
still preserves compilation evidence. All main outputs compare to original
FP32 scores at unchanged rtol/atol1e-3. Failure stops timing and records main
sample rejection/error where available. FP16 rounding, cancellation/underflow
or altered scaling must not be excused by loosening tolerance. No production
fallback/replacement is installed.

Small n5/6/7, empty-group and guarded-tail tests use zero alternate QS heads,
zero KS at page boundaries, signed weights and varied scales. Page mappings,
visibility and all score masks are checked. Two cases/two rounds, compile,
correctness, timing and automatic upload remain one command. This is one
native-method experiment with inherent precision changes, not a claim that
only instruction selection differs. Do not compose earlier arithmetic/reuse
candidates; no TopK/compiler or integration entry changes.

### Cube compile repair: flatten coefficients and reduce N

Report 2e03f0f7984a2b1b0457676279ff4cd997590e37 failed before candidate
execution/precision/timing. BiShengIR required2238464bits of UB (273.25KiB)
against1572864bits available (192KiB). It also reported alignment failures on
coefficient_matrix, whose head-index expressions expanded to [16,4,4].
Compiler automatic code-motion/multibuffer retries did not resolve it.

The operator-only repair uses flat query/head pointer and mask expressions
for QS/W in the Cube arm, and range comparisons for block-diagonal coefficient
membership instead of lane//4. This aims to preserve [16] and [16,16] compiler
shapes; device compilation must verify it. Reduce key tile toN64 to lower
intermediate lifetimes/UB demand, retaining grid20/warps4 and all score math.
Neither compiler code nor precision tolerance is changed.

Compare four arms in one run: integrated_full(N128), pack4_qk(N128),
pack4_qk_n64(FP32 reduction N64), pack4_qk_cube(second Cube N64).
CUBE_REDUCE_SPEEDUP_VS_QK compares matched N64 arms; additionally print
CUBE_REDUCE_SPEEDUP_VS_QK_N128 against the existing candidate configuration.
Compiler records per-arm block_n. Tail checks use N64 for both mathematical
controls. Reducing N changes tiling, so matched and original-config ratios
must not be conflated. Compile/correctness/timing/upload remain in one command;
precision rejection still stops timing and production remains unchanged.

## Fifth probe: bounded key blocks with unchanged N128 scoring

Report c9860c5608c7ffed3417d29fc8240257e46aafc6 compiled the second Cube
candidate and passed the mixed tolerance, but was41.6--41.8x slower than its
matched N64 FP32 control. FP16 score errors increased. Do not compose it here.

| Item | Key block experiment |
| --- | --- |
| CLI | --key-groups |
| Arms | integrated_full, pack4_qk, pack4_qk_g2/g4/g16 |
| Dot block | N128 for all current QK controls and candidates |
| Scheduled work | Query group x key block, grid cap20/warps4 |
| Block sizes | 2/4/16 consecutive key tiles =256/512/2048 keys |
| Math | Original FP32 scales/ReLU/weighted head reduction; one INT8 dot |
| Loading | Original token-major K row load followed by transpose |

Inspected CANN arch22 code has S2_BASE_SIZE2048 outside and
S2_BASIC_BLOCK_L0=128 inside. This is not a2048-wide matmul. The probe borrows
this two-level work organization. Each work unit loads its group's Q/QS/W and
visibility once, then scans only its bounded block. Different blocks belonging
to one query group can run on different programs. This trades bounded reuse
and locality against task granularity and load balance, unlike the prior
whole-key-range group ownership. This does not reproduce native explicit
memory buffers, events or pipelines and adds no prefetch/multibuffer directive.

At the end of a key range, the final block is padded to KEY_TILES iterations.
Extra tiles perform masked K/table/scale reads and no output writes. They are
not silently excluded from timing: compiler records padded_key_tiles per group.
For long100k,232 key tiles round to240 for g16, adding8 masked tiles per group.
Widths and numerical results remain unchanged. All tile coverage/uniqueness
and tail masks must be validated; this experiment does not skip invisible work.

Require seven static loads, one dot, one store, FP32 head sum, mixed mode.
Export loop/load IR and full lowered artifacts to inspect actual reuse and K
row loading. Check every main output vs original scores, CPU sample references,
all masks, n5/6/7 query tails, signed QS/weights, nonmonotonic visibility,
shuffled maps, different maps between groups, empty groups and guard rows.
Tail checks run all three block sizes, including key-block padding cases.
One run compiles, validates, times two rounds and uploads. Print all three
KEY_GROUP_SPEEDUP_VS_QK values; no production/TopK/compiler change. These are
three bounded diagnostic configurations, not hardcoded production dispatch
or a claim of native-equivalent scheduling. The same fixed-config diagnostic
autotuning exemption applies. Pipeline changes are reserved for a later probe.

### Four-tile ten-workload regression

Report 07c49cf054a74c8edae9117645a652d5595232aa passed all two-case checks.
The g4 candidate measured1.1746x/1.0498x vs current QK, outperforming g2/g16
on both cases. Extend with --key-groups --ten-shapes: only integrated_full,
pack4_qk and pack4_qk_g4 remain timed, limiting cold compilation and duration.
The scorer kernel is unchanged from the measured candidate.

All ten migrated INT8 workloads retain their own widths, mappings and chunk
sizes. Main timing remains representative middle chunks. Synthetic guarded
tail tests compare control with g4 only. Real first/last chunks execute g4,
not an accidental plain QK substitute. At the first real request boundary,
a failing map/owner guard routes to the original scorer, recorded explicitly.
The unequal-map rejection check remains. This is probe-level fallback testing,
not production dispatch. Signed scales/weights and masked key-block padding
remain supported by the FP32 formula. Original strict score/IR gates remain.

The terminal table compares g4 directly with current pack4_qk and prints the
arithmetic mean only if all requested cases pass. It is not the older legacy
integrated ratio, full-prefill or TopK/native benchmark. Two rounds, warmup2,
active3 per arm; compile/check/time/upload are still one command. Candidate
promotion and full-entry measurements require subsequent validation.

## Sixth probe: existing buffering/preload options on group4

Full-input dispatch report db4b904b1fd2aa809d892eaaf851ea49b4991d42 passed;
paired full-score speedup vs QK mean1.061x, compact remains the original QK.
Keep the validated group4 kernel body and FP32 formula unchanged for this probe.

| Arm | Existing kernel launch options |
| --- | --- |
| pack4_qk_g4 | Unchanged defaults; recorded multibuffer=True |
| pack4_qk_g4_no_mb | multibuffer=False only |
| pack4_qk_g4_preload | enable_preload=True only |

Use --buffering. integrated_full is retained only as a reference/calibration
context; the primary ratios compare these three identical group4 algorithms.
The current backend already requests automatic buffering, so enabling it again
is not a new optimization. Do not claim num_stages changes add a new pipeline.
No manually allocated double-buffer or compiler implementation patch is added.
This first diagnostic asks whether existing buffering and preload materially
change generated code/resources/performance before a manual source rewrite.

Official FlagTree source inspected via GitHub API on branch
0.7.0-rc2-triton3.5, third_party/ascend/backend/compiler.py, maps multibuffer=False
to --enable-auto-multi-buffer=False and enable_preload to --enable-preload.
That branch is not assumed source-identical to the installed wheel. The run
therefore records the installed compiler module path/hash and relevant source
excerpts, and requires each requested option to appear in kernel metadata.
All input shapes, map guards, grid20/N128/warps4, KEY_TILES4, math, output stores
and work ordering remain fixed. No QS/W or KS reassociation is composed.

Export source/IR/metadata/binary hashes, and normalized IR/binary comparison
against group4 rather than interpreting a changed flag alone as overlap.
Metadata is a requested compiler configuration, not proof of a particular
buffer allocation or actual DMA/Cube concurrency. Backend internal fallbacks
may affect realization; timing and artifacts must be interpreted together.

All main outputs and masks compare to existing FP32 scores and CPU samples;
all three options also execute n5/6/7 and empty-group guarded checks with signed
QS/weights, shuffled maps and nonmonotonic visibility. Same exact one-dot/store,
seven-load and FP32 reduction gates. One command compiles, checks, captures two
rounds interleaved, prints BUFFERING_SPEEDUP_VS_GROUP4 and uploads logs.
No production/TopK changes or full-entry/native speedup claims. A failed option
is reported, not silently replaced or subjected to a weaker accuracy/timing gate.

### Complete buffering pair without unsupported preload

Report ee7d5897ec7b65bb11424fd812a01e7bc89eece6 failed while compiling
preload=True on both workloads: scope.scope op Failed to collect vector loop
tiling info. The compiler command included --enable-preload=True. Default
and multibuffer=False arms compiled and passed sampled main-score checks;
workspace changed98336 to24608 bytes, and binary hashes differed. No timing
or tail checks ran, so no performance conclusion is available.

Use --buffering --buffer-no-preload to explicitly omit that failed arm and
finish default/off comparison. Record excluded_options with the original report
commit; this is a changed requested set, not a silent fallback or a claim that
preload passed. Main full comparisons, artifact audits, both-options tail checks,
two-round timing and upload still run together. No scorer/compiler changes;
default invocation without the exclusion flag retains all three original arms.

## Seventh probe: explicit next-key preload in operator source

Report 01bc22e4bb1bd099f52b669647b2f3ee07e14454 passed the default/off
comparison: disabling automatic buffering slowed both cases (0.897x/0.979x).
Keep default multibuffer=True. Backend enable_preload=True failed compilation
in the earlier report and is not requested here.

| Item | Explicit preload experiment |
| --- | --- |
| CLI | --explicit-prefetch |
| Arms | integrated_full, pack4_qk_g4, pack4_qk_g4_prefetch |
| Constants | N128, KEY_TILES4, grid cap20, warps4, unchanged FP32 formula |
| Initial load | First table/K/KS tile outside the four-tile loop |
| Loop | Load next table/K/KS before current dot, then carry tensors to next iteration |
| End | Last prefetch has all reads masked off; carried dummy tensors unused |

All K reads retain token-major contiguous D128 row loading, then transpose
for QK. No direct column gather, new arithmetic candidate or preparation kernel
is introduced. Work units, masks, query-group guards and output ownership match
the validated group4 control, including padding at the key block end. Candidate
TTIR has ten static loads (four group inputs, three initial tile, three next
tile) versus seven in control; runtime useful tile loads still cover the same
four tiles. The final masked prefetch is included in timing, not filtered out.
No runtime conditional changes loop-carried tensors.

Compiler gates retain one dot/store, FP32 head sum and visibility max. Require
expected per-arm load counts explicitly without weakening other audits. Export
all IR/source/metadata/hash and loop/load lines. Source order does not prove
DMA/Cube overlap; the compiler may reschedule or add spill/synchronization and
live current/next tiles can increase resource pressure. No compiler source or
pipeline launch option changes. The existing default buffering remains enabled.

Main outputs/masks compare fully against FP32 scores, with CPU samples.
Both options run n5/6/7, empty groups, shuffled/different group maps, signed
QS/weights, nonmonotonic visibility and guarded query tails. Width256 tests
include a partial four-tile work unit and masked next reads beyond width.
One command compiles, validates, captures two rounds and uploads; precision or
compile failure stops timing. Print PREFETCH_SPEEDUP_VS_GROUP4. No production,
TopK or full-entry claim; the same fixed-config diagnostic exemption applies.

## Eighth probe: scalar-page contiguous K/KS fragments

Report 5092a965741cd15dcf3e656045306947f5236a64 passed explicit preload checks
but showed no gain (0.994x/0.997x). Do not compose it. Keep the default group4
kernel, N128, auto-buffering and original FP32 arithmetic.

| Item | Page fragment experiment |
| --- | --- |
| CLI | --page-segments |
| Arms | integrated_full, pack4_qk_g4, pack4_qk_g4_pages |
| Geometry | N128/page192, at most two page fragments per key tile |
| Table loads | Two masked scalar page IDs instead of128 per-token addresses |
| Data loads | Two affine token-major K/KS loads, disjoint row masks, then select original order |
| Arithmetic | Identical INT32 QK/FP32 dequant/ReLU/head sum |

For tile start s, page=s//192, offset=s%192, first_count=min(128,192-offset).
First fragment uses physical0*192+offset+lane for lane<first_count. Second uses
physical1*192+offset+lane-192 for lane>=first_count. Both intersect width and
maximum query visibility. The second page ID is read only when the tile crosses
a page and its second fragment is visible/in bounds. Fully invisible/padded
tiles do not read either page. Pointer arithmetic on masked-off lanes may lie
outside storage; no such lane is read, as required by the load mask contract.

This borrows contiguous within-page transfer from native KeyNd2NzForPA and
GetKeyScale, not native direct ND2NZ/DMA instructions. The compiler determines
whether affine masked loads lower to block copies. It may retain scalar loops
or add costly fragment merges; two K temporaries may increase UB pressure.
Do not infer hardware DMA granularity from source alone. All complete IR and
loop/load lines are exported for interpretation, and all timings are retained.

Ten static loads are expected:4 group inputs +2 scalar page IDs +2 K +2 KS.
Useful K/KS traffic equals the original, not twice the tile: fragment masks
are disjoint and partition every live key lane. One dot/store and FP32 head sum
remain required. Shared-map/owner proof, grouping and KEY_TILES4 stay fixed.

Compile, main full-score checks/CPU samples/masks, guarded n5/6/7 and empty-group
checks, signed QS/weights, shuffled/different group maps and visibility checks
precede two-round timing. Width256 and visibility193/255 exercise page192
crossings; padded tiles exercise masked out-of-range page IDs. Print
PAGE_SEGMENT_SPEEDUP_VS_GROUP4 and upload in the same command. No arithmetic,
preload flag, TopK, production entry or compiler changes. The fixed-config
probe exemption applies; this remains an operator-level diagnostic.

### Prefix-only fragment load repair

Report 73db8a59c7830db0d1858a1795c27b5af1df6a36 compiled both candidates
but page-fragment execution failed before numerical checks: ACL507015,
MTE on-chip destination out of bounds. Saved bcmlir contains a definite invalid
second-fragment subview: extent=min(valid_count,128)-first_count, without a
zero clamp. For an invisible tile valid_count0 and first_count64/128 produce
negative extents. This is a concrete lowering defect consistent with the MTE
failure; the exact faulting device PC has not been mapped to that copy.
The same IR also speculatively reads masked scalar page pointers before select,
so masking alone does not guarantee suppression of invalid table addresses.

The operator-only repair computes second_count=max(0,min(128-first_count,
min(width,max_visible)-start-first_count)). Load page1 K/KS from its beginning
with prefix mask lane<second_count. Align the loaded prefix to the output
suffix with in-range gather(max(lane-first_count,0)), then select fragment0/1.
This avoids empty lower-bound/upper-bound intersections in tl.load masks.
Both scalar table addresses clamp to COLUMNS-1 while retaining their original
validity masks. Thus speculative page ID reads stay in bounds even if the
backend does not suppress them; extra invalid logical-page traffic may remain.
All valid lanes and FP32 arithmetic are unchanged. Negative pointer bases for
page1 are removed; all load extents should be nonnegative by construction.

Gather/alignment adds possible overhead, so performance is not predicted.
Keep N128/group4/default buffering, ten static loads, one dot/store and all
original precision/edge tests. Same --page-segments command exports compiled
artifacts and runs correctness/timing/upload together. No compiler, production
or TopK changes. If another runtime failure occurs, retain its evidence rather
than claiming the first diagnosed defect was the only cause.

## Ninth probe: page-aligned N64 tiles without fragment merges

Report 75d92ffec98eea92a02becd16fe3bbcce4d384d5 passed the prefix repair's
preflight and tail checks. Strict timer window validation failed; raw traces
contained18 tasks/18 windows but some task ends exceeded CPU windows. Raw
candidate tasks were roughly2.23s/7.03s, not accepted benchmark measurements.
The lowered program added triton_gather calls. Do not repeat it for a precise
speedup or treat that implementation as a useful candidate.

| Arm | Block N | Tiles per work unit | Page addressing |
| --- | --- | --- | --- |
| pack4_qk_g4 | 128 | 4 | Existing per-token table indexing |
| pack4_qk_g8 | 64 | 8 | Matched N64 control, original table indexing |
| pack4_qk_g8_page64 | 64 | 8 | One bounded scalar page ID, contiguous K/KS prefix loads |

Use --single-page64; integrated_full remains a separate reference. All work
units cover512 key positions, with identical ceil(width/512) unit counts and
grid cap20/warps4. N64 divides page192 and tile starts are multiples of64,
so every tile stays within one page: offsets0/64/128, never offset+lane>=192.
The scalar page pointer clamps to COLUMNS-1 before masking, avoiding invalid
speculative table reads. The returned page ID is masked tozero when invisible;
K and KS use prefix visibility masks and safe within-page offsets. There is
no second page, negative copy extent, merge, TL gather or prefetch.

The arithmetic following loads is AST-identical to the group4 scorer except
for the N64 specialization. Keep INT32 dot and FP32 dequant/ReLU/head reduction,
seven static loads, one dot/store, default buffering. Export all IR and record
any backend-generated triton_gather call count rather than assuming a particular
DMA implementation. Match N64 controls to isolate addressing; compare against
N128/group4 separately to assess whether the smaller dot tile's overhead offsets
benefits. Do not use the old failing N128 fragment loader as the comparator.

Main full-score/CPU/mask gates and guarded n5/6/7/empty-group checks remain.
Both N64 options test shuffled pages, signed QS/weights, nonmonotonic visibility
and padded8-tile work units at width256. One command compiles, validates,
times two rounds and uploads. Print SINGLE_PAGE_SPEEDUP_VS_N64 and
SINGLE_PAGE_SPEEDUP_VS_GROUP4_N128. No production/TopK/compiler changes and
no accuracy/timer relaxation. This is a fixed-config diagnostic, not deployment.

### Single-page N64: ten-workload restricted regression

Report 2835f3afda9d51e64aee132a795c5b4d53d96db1 passed both representative
workloads. Scalar-page N64/group8 took 540.583us (budget_exact) and 1695.540us
(long100k), versus N128/group4 591.377us/2232.137us: 1.094x/1.316x.
Matched per-token N64 controls took 685.923us/2450.750us: 1.269x/1.445x.
The maximum score difference was 7.63e-6; guarded tails passed. Candidate
workspace was 20576 bytes versus 49184 bytes for matched N64. The exported
bcmlir reduced five scf.for loops to two, retaining contiguous K/KS copies.

Extend with --single-page64 --ten-shapes. Each workload times three arms:
legacy integrated_full as correctness reference, the current shared-map
selection path as performance control, and pack4_qk_g8_page64. Mirror the
existing integrated dispatcher: ceil(logits_width/128)<16 selects plain QK
(compact only); all other migrated workloads select N128/group4. Report the
selected control explicitly and calculate candidate speedup against it, never
against the slower legacy reference. Omit the extra matched-N64 timing arm
because the two-workload experiment already isolated the addressing change.

Retain compile-artifact export, all-score pre/post checks at rtol/atol1e-3,
CPU reference checks, signed-scale/weight n5/6/7 and empty-group padded tails.
Real first/last chunk checks launch the actual N64/group8 single-page scorer
when shared-map/owner guards hold; a crossing request group falls back to the
legacy scorer. Explicit unequal-map rejection remains. Time representative
middle chunks for two interleaved rounds, warmup2/active3, print each control
and candidate latency plus their arithmetic mean speedup. These measurements
exclude TopK and do not represent full-prefill or combined-operator speedup.
No production, integrated dispatch, TopK or compiler changes in this step.

### Restricted single-page dispatch integration

Independent report 117fe89a78dcbf43f87eb38a64eed55f972d1fee confirmed scoring
ratios versus current dispatch: first_chunk0.877x, budget_exact1.078x,
compact1.015x, long100k1.326x. All accuracy/tail/guard gates passed;
maximum score error7.63e-6. Keep the smaller-width paths unchanged.

Add opt-in --single-page64-dispatch to the full-input kernel benchmark;
it requires --integrated --shared-map-qk --key-group4-dispatch. The prepared
trial adapter selects N64/group8 scalar-page loads for width>=8192 only after
its existing immutable device-page-table and request-owner proof. Below8192
retain group4 at >=16 N128 tiles or plain QK below that threshold. Unproved
chunks and unknown slices fall back to the single-query scorer. This selects
eight migrated workloads; first_chunk and compact retain previous routes.
The width threshold is a conservative trial policy, not a proven crossover
for arbitrary workload widths or visibility distributions. Default flags
preserve existing behavior; no registered production or compiler changes.

The same run checks compilation artifacts, every chunk's scores against the
legacy scorer, all-row selection pairing and sampled CPU references. Extend
the existing budget_exact threshold gate with widths8191/8192/8193 and actual
N64 execution, plus cross-request and unequal-map fallbacks at8192. Existing
QK/group4 threshold checks remain. Record route counts and preparation cost
outside kernel timing. No input-name lookup or extra timed tensor preparation.

Measure all query chunks, not a representative middle chunk. Retain score,
TopK on frozen logits, directly executed score+TopK combined, legacy_score,
and one full native_fused V1 invocation. Add dispatch_score for the previous
QK/group4 policy with identical chunk guards; in this mode omit the redundant
plain-QK comparator to limit runtime. Print all device sums, current-dispatch
score speedup and native_fused/combined overall speedup; include them in CSV.
Native stage times remain unavailable. Combined duration is measured directly,
not synthesized by adding isolated score and TopK measurements. This remains
an unregistered prepared-input trial with exploratory profiler device sums.
