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
