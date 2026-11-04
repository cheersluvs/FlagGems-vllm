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
