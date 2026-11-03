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

# Validated INT8 score prefix diagnostics

## Project truth

| Item | Frozen choice |
| --- | --- |
| Target | 910B, pinned CANN9.1 / FlagTree ascend3.5 |
| Baseline | Integrated grid20/N128 score entry |
| Source | Original scorer SHA13718b109b2ffd93164b87c077c9d991b59caaa7f073302f3b7aced74a787334 |
| Cases | Middle chunk of budget_exact (128x8192) and long100k (112x29696) |
| Launch | Same grid20/N128, num_warps4, H4 padded16, no raw-dot store |
| Question | Which retained prefix increases latency enough to justify a replacement probe? |
| Scope | Diagnostic tools only; no query packing or production changes |

## Contract

| Item | Value |
| --- | --- |
| Q/K | Existing INT8 synthetic fixture, D128, page192, same row mappings |
| Scale/weight | Original FP16 operands; FP32 postprocessing |
| Work | Same persistent work loop and logical tile count for every arm |
| Mask/output | Same visibility and final masked FP32 store; future columns -inf |
| References | CPU reconstruction of each prefix for first/middle/last chunk rows |
| Checks | Full-chunk finite/mask checks, exact integer checksums, FP32 tolerance1e-3, repeated postchecks |
| Unsupported | Drifted source, invalid cut anchors, bad checks or missing dot/store IR refuse interpretation |

## Paths

| Prefix | Retained original work | Observable diagnostic sink |
| --- | --- | --- |
| mask_store | Row/column loop and visibility | Zero for visible columns, -inf for invisible |
| index | Above + page table and physical token index | Physical token number |
| k_load | Above + INT8 K gather | Sum K along D |
| q_load | Above + INT8 Q load | Sum K along D plus sum Q along D/heads |
| dot | Above + INT8 QK to INT32 | Sum INT32 dot over heads |
| dequant | Above + key/query scales and FP32 dequantization | Sum dequantized dot over heads |
| full | Complete original body | Original ReLU/weight/head-reduction score |
| integrated_full | Unmodified current adapter entry | Original score; controls generated-full equivalence |

The recovered Claude process explicitly says PREFIX TRUNCATION, validating every
arm up to the cut, rather than wrong-answer middle deletions that change later
control flow. Source: local Claude session b98f4645-5c41-4dbe-9f3c-d466f3491692,
2026-09-23 memory `hygon-prefill-optimization-process.md`, step4. The cp-gather
cloud excerpt was also rechecked, but it does not document this specific
attribution method; do not incorrectly attribute the prefix rule to that excerpt.

Builder uses guarded AST cuts from the actual source, not hand-written replicas.
The full generated function's AST is identical to the original. Each generated
module is isolated; no live kernel is rebound or patched. Save source text, IR,
load/dot/store counts and compiler metadata for every arm. Kernels share the
original function name only so the existing strict timer can recognize host
launches; each arm is captured separately, with its own artifacts and samples.

Observable sinks are necessary to keep values live. They include their own
reductions and conversions. Therefore timing differences are diagnostic prefix
contrasts, NOT exact independent operation costs. The mask/store arm is a lower
bound including the loop and masking, not a measurement of writeback alone.
Constant zero output may vectorize differently. Removing the dot changes Cube/
Vector lowering and synchronization; eliminating consumers can change earlier
scheduling and buffering. Negative or non-additive differences are retained,
not clipped into a misleading cost budget. Never replace middle values with
zero and continue the original downstream computation to attribute its latency.

Only H4/D128 fixtures are used. Integer checksums have a worst-case magnitude
below 2^24 (dot sum <=4*128*128*128), so FP32 storage is exact. These intermediates
are valid prefix checksums; they are not selectable logits or production outputs.
No TopK runs. There is no compute fallback or new autotuner. Ops exports and
registration do not apply to this diagnostic tool.

Timing uses the existing strict CPU+NPU trace timer: two reversed-order rounds,
one warmup, two active invocations. Measure one representative chunk per callable,
not the full logical prefill. Preserve rejected captures with the same bounded
retry policy. Print chunk_us, ns per logical output slot, raw samples and the
adjacent prefix contrast. Compare generated-full with integrated-full; a timing
ratio outside 0.9..1.1 marks the probe failed for attribution. Even a passing
control does not prove the earlier prefixes retained identical scheduling.

The result selects the NEXT isolated replacement/microbenchmark, which must then
be retested on the original integrated kernel. Do not ship diagnostic prefixes
or treat their contrast as a promised optimization speedup.

Local checks: source drift rejection, AST equality of full control, syntax of all
seven generated prefixes, shared loop/mask invariants, lint and two-case missing-
prerequisite report flow. NPU compilation and checksum checks are still pending
the card run. Keep production and TopK untouched throughout.

## Repair checksum reduction control before attributing the mixed path

Report 526a144c4e27b867da5d772cb9d8dde12074a023 passes all prefix checks and
full-control equivalence (timing ratio about0.99989), but is nonmonotonic:
INT32 dot-checksum prefix is much slower than the full score. Its metadata shows
pure AIV for early prefixes versus mixed Cube/Vector for dot and later prefixes.
The dot sink reduces INT32 before casting, whereas full score casts before FP32
postprocessing/reduction. These differences prevent direct cost attribution.

`--tail-only` selects integrated_full, dot, dot_fp32, dequant and full on the same
two middle chunks. The new sink is `tl.sum(dot.to(tl.float32), axis=1)`; it keeps
the integer dot unchanged and converts before reduction. Its checksum is verified
against CPU `dot.float().sum(1)` exactly for the bounded H4 fixture, including
mask/postchecks. The old INT32 sink remains a measured control.

Before timing, every selected specialization must report mix_mode='mix'. Its
exported bcmlir must contain exactly one INT32 reduction for dot, or one FP32
reduction for dot_fp32/dequant/full/integrated_full. Fail rather than interpreting
an unrecognized lowering. Store/dot checks, raw source/IR exports, invocation-sum
calibration, invalid-capture retries and reversed two-round timing are unchanged.
The original full function AST and frozen source guard remain unchanged.

Interpret dot_fp32 versus dot as checksum/lowering sensitivity, not an operator
optimization. Dequant minus dot_fp32 includes scale loads and FP32 multiplies;
full minus dequant includes weights/ReLU changes. Even with matching high-level
mode and reduction dtype, layouts, scheduling and synchronization may differ;
these contrasts do not become exact isolated instruction costs. Leave early
cross-mode prefixes out of this round. No query packing, TopK or production
changes are introduced. Use resulting contrasts to select a subsequent probe
that is tested on the original full integrated score.

Local checks: syntax of all generated prefixes, exact cast/reduction AST ordering,
full AST identity, extraction of actual previous bcmlir reduction types, lint,
and two-case tail-only missing-prerequisite reports. New specialization compilation
and runtime equivalence are pending the card run.

## Query-scale prefix follow-up

Report 265ac396910e63eaa900eb7a6eac93c981b08b17 validated the FP32 sink control:
dot_fp32 is 1.421/5.019 ms versus dequant 2.136/8.297 ms on the two middle
chunks. The next question is which part of scale handling changes the cost.

`--scale-only` selects integrated_full, dot_fp32, query_scaled, dequant, full.
The new prefix cuts the actual group body after query_scale is loaded, retains
the original FP32 multiplication by query_scale, then reduces FP32 for its sink.
The CPU reference is `(dot.float() * query_scale.float()).sum(1)`, using the same
non-unit fixture scales. Key-scale arithmetic and weights/ReLU are not executed
by this prefix. All inputs remain INT8 Q/K with FP16 scale storage; no FP8 input
conversion, re-quantization or dtype policy change is introduced.

Require mixed execution and FP32 reductions on all selected specializations.
Additionally require TTIR load counts 4/5/6/7 for dot_fp32/query_scaled/dequant/full
(and 7 for integrated_full). This checks that dead key-scale and weight loads
were removed as intended. An unexpected compiler result fails before timing.
Keep sampled prefix references, all-chunk masks, postchecks, no-debug IR,
full-control equivalence and the same strict trace timer. Tail-only and
scale-only flags are mutually exclusive; earlier subsets remain available.

Inspect query_scaled minus dot_fp32, then dequant minus query_scaled. These
contrasts include scale accesses, broadcasts, arithmetic and any changed layout
or synchronization; they are not isolated multiply timings. Use them to select
an equivalent full-kernel replacement probe before changing the actual adapter.
The original QK math, adapter, production registration and TopK stay unchanged.
Two reversed-order rounds with two active invocations on the two representative
chunks are sufficient for this diagnostic screen, not full-shape acceptance.

Local validation covers every generated prefix's syntax, the actual query-scale
cut and FP32 sink, original-full AST identity, source-drift rejection, lint and
the two-case scale-only missing-prerequisite path. Compilation and numerical
checks on NPU remain pending the card run.

## Interleaved single-trace scale-prefix retest

Report d5572a46b7e2ee0809a0614a19db0b4fe4bfa15f passes numerical and lowering
checks for both cases. Budget_exact stopped in round2 after three empty-device
captures; its complete round1 is not a completed two-round result. Long100k
completed, but query_scaled was 21.616 ms versus 5.203 ms in its two rounds;
individual active values in the slow round were 17.563 and 25.669 ms. Those valid
slow values must not be dropped or silently labeled collection failures. The
cause of this duration variation is not established; the scale-side attribution
is therefore not yet supported.

`--scale-only --single-capture` reduces profiler start/stop cycles by capturing
all prefix samples in one CPU+NPU trace. Within a round, each active repetition
cycles through all five arms; round2 reverses the order. Each launch has a CPU
record-function window and synchronizes before closing it. The existing stage
parser requires complete, nonoverlapping windows and all device tasks covered;
the new collector additionally requires exactly one device task and named score
host launch per window. Both rounds and all raw active durations are retained.
The existing separate score one/three-call calibration remains in force.

Only incomplete whole captures retry, at most three attempts. The first valid
capture is accepted regardless of its durations. Rejected traces and the exact
window-to-arm/round/active mapping are saved. This reduces exposure to repeated
profiler lifecycle failures; it does not fix or identify the underlying CANN
collection issue and cannot guarantee performance stability or exclusivity.

Each arm gets its own FP32 output, warmed before recording, so postchecks inspect
its last timed output directly after the shared capture. This changes output
buffer addresses relative to the old probe; compare current paired arms only,
not pooled old/new raw times. No CPU data copies occur inside timed windows.
All numerical, mode/reduction/load-count gates and full-control timing checks
remain unchanged. No score kernel, TopK or production changes are introduced.

Recommended run: both representative chunks, two rounds, two warmup passes over
all arms, three active repetitions per round. This collects six valid timings
per arm with one main trace per case rather than separate traces per arm/round.
Inspect every raw sample before interpreting prefix contrasts. Exit0 confirms
structural/numerical checks, not low variance or performance acceptance.

Local validation includes synthetic missing-device-first/valid-next capture,
window ownership, interleaving order, output routing, rejection-artifact retention
and deliberately slow valid sample retention. Syntax/lint and the two-case
missing-prerequisite report path also pass. Actual new collection needs the
card run. Hardware isolation remains unverified.
