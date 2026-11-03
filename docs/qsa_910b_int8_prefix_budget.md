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
