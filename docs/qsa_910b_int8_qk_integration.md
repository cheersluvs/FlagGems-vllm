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

# Guarded QK INT8 integrated-entry validation

| Item | Scope |
| --- | --- |
| Evidence | Ten representative chunks passed, paired mean score speedup3.451x; not full-entry acceptance |
| Entry | PreparedSharedMapInt8Selection, optional --shared-map-qk on integrated benchmark |
| Hardware | Pinned910B/CANN9.1/FlagTree environment |
| Production | Unregistered tools adapter; BF16/NV production unchanged |

| Contract | Value |
| --- | --- |
| Inputs | Existing INT8 Q/K H4 D128 page192; FP16 scales/weights |
| Scores | Original association, INT32 dot and FP32 dequant/ReLU/head sum |
| Fast path | QK direction only when every four-row group shares page table and request owner |
| Fallback | Whole chunk uses unchanged grid20/N128 scorer; arbitrary unprepared slices also fall back |
| Lifetime | Prepared inputs/page tables must remain immutable; reconstruct after mutation |
| TopK | Unchanged persistent_topk, max32 rows per call |

| Measurement | Validation |
| --- | --- |
| Preparation | Copy actual device row table to CPU, validate request boundaries, cache per-chunk guards; record wall time separately |
| Preflight | All chunks/all score values vs legacy, all masks; CPU samples and full native selected-score/index checks |
| Timing | Full logical score/legacy_score/TopK/combined/native jobs, device task sums |
| Calibration | Existing single/triple calibration on legacy scorer; no candidate-host-name ambiguity |
| Limits | Prepared kernel timing excludes metadata preparation and host launch gaps; native has no score/TopK split |

The existing integrated benchmark is opt-in; default callers retain the legacy
adapter. Both inherited score_all and __call__ dispatch through the new
score_into, preserving bounded scratch. No external Q/K packing or compiler
changes. Construction copies the real table, not an unverified CPU fixture.
The request boundaries are caller-supplied prepared metadata. Their validity
and the input immutability contract are prerequisites, not inferred semantics.
Unknown slices fall back. Fast/fallback counts, reasons and representative IR
for each used route are uploaded. Compiler gates require a single score store
and INT8-to-INT32 dot. Legacy full-score timing is rebuilt on the same device.

Use the existing integrated runner with --shared-map-qk. Initial ten-case full
validation uses one round, one warmup and two active samples to limit runtime;
it is an exploratory full-entry check, not final statistical acceptance.
All raw traces and rejected incomplete captures remain. Valid slow timings are
not filtered. Terminal prints every stage in milliseconds, score speedup versus
legacy and combined speedup versus native. CANN stage speedups remain absent
because its public baseline is fused. Baseline/output mutation, TopK changes,
and production registration are outside this change.

## Summary-only recovery

Report cbb40d92e4fa5c6860fdbdfab9bc2a44854ad8df saved ten passed workers,
all correctness checks and complete stage timings. The parent then raised
ValueError writing CSV because legacy_score_us was missing from fieldnames.
The schema now includes legacy_score_us, score_speedup_vs_legacy and preparation_ms.

Set QSA_SUMMARIZE_REPORT to the retained report directory and run the integrated
wrapper. It invokes --summarize-only, validates saved workers passed with zero
subprocess codes, regenerates summary/CSV, and uploads. No NPU import/execution
or profiler capture is required. Original console/code commit/failed exit code
are retained; summary-console and summary-code-commit identify this recovery.
Preparation and device timing remain separate; a single round's zero range
is not evidence of stability. Native score/TopK remain inseparable.

## Optional four-tile dispatch integration

Representative ten-case report a2e7a1fa261c5a7f1a7ecd218d50d43bd0e80c90 passed
all checks, with group4 mean1.077x vs QK and one compact regression0.895x.
Use --shared-map-qk --key-group4-dispatch for the next full-input trial.

| Condition | Prepared chunk route |
| --- | --- |
| Unequal page mappings or mixed request owner inside a four-row group | Legacy single-query scorer |
| Shared mapping, fewer than16 N128 key tiles | Existing QK scorer |
| Shared mapping, at least16 N128 key tiles | Four-tile QK block scorer |

The16-tile threshold is an explicit candidate policy between the observed
9-tile compact and >=32-tile larger workloads, not an empirically optimized
universal cutoff. It uses ceil(width/128), not case names or request counts.
Map checks continue to use actual device table snapshots during preparation;
inputs remain immutable for the adapter lifetime. Default callers do not enable
this policy. Guard costs remain separately recorded, excluded from kernel sums.
Unknown slices retain legacy fallback.

Before full validation, budget_exact additionally exercises widths1920/1921/2049
(15/16/17 tiles) using seven query rows, clamped visibility and original pages.
Compare every score against legacy plus CPU samples/masks. Two further1921
checks cross a request boundary or change one query's physical page mapping;
both must select legacy and match the corresponding modified reference data.
These checks are outside timing and do not change the main workload fixtures.

Run all ten full logical workloads, including TopK. Add qk_score to the existing
legacy_score/score/topk/combined/native jobs so policy benefit is paired on the
same card. CSV explicitly includes qk_score_us and score_speedup_vs_qk. The
legacy scorer still supplies timer single/triple calibration; stage counts
remain one score kernel per chunk. Upload route counts, full score comparisons,
index/selected-score checks against CANN, traces and each stage time.
Initial validation uses one round, one warmup and two active samples. It does
not establish timing stability or native stage speedups. No earlier arithmetic,
Cube reduction, compiler or TopK changes are composed. The wrapper's optional
summary flag also avoids an empty-array nounset issue with macOS Bash3.
