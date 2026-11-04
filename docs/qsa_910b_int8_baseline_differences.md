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
