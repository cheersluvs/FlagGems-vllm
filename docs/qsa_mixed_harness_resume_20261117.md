# Mixed full-input harness: failed device initialization and selective resume

| Project truth | Value |
| --- | --- |
| Report | 977a75785eeced690e2f369dce8bc928766f6402 |
| Code | 4ed23178f07eb8f95c1d0bb8d1f0db01ffaffd98 |
| Result | Nine complete pre/post checks and timings passed; long100k failed before kernel execution |
| Failure | set_device(0),507033/E39007 device subprocess startup timeout; HDC diagnosis suggested by runtime |
| Acceptance | No complete ten-case arithmetic mean available |

| Contract | Value |
| --- | --- |
| Frozen | Candidate, native baseline, packing, fixture, timing harness and compiler |
| Editable | Master resume/report bookkeeping and CPU regressions |
| Resume | Copy verified passed case files to fresh destination; only execute missing/failed cases |
| Code gate | Git diff from source code commit allows only master, its tests and docs |
| Provenance | Original report/code and reused cases explicitly retained; combined-session summary labelled |
| Isolation | Fresh compiler preparation files are never replaced by old report files |

| Paths | Purpose |
| --- | --- |
| --resume-report | Reuse valid prior cases and run remaining cases |
| --summarize-only --source-report | Existing CPU-only recovery, unchanged |
| Production | Unchanged, no promotion |

All times below are milliseconds. Separate online TopK measurements are staged
standalone diagnostics, not the fused kernel's internal Vector duration.

| Case | CANN V1/mixed | Serial/mixed | Persistent TopK | Online TopK |
| --- | ---: | ---: | ---: | ---: |
| first_chunk | 0.5872 | 1.3959 | 7.6081 | 10.7554 |
| budget_exact | 0.5090 | 0.9498 | 11.7098 | 24.4110 |
| dominant_long | 0.5070 | 0.9683 | 14.9109 | 25.3199 |
| tail112 | 0.5082 | 1.0251 | 14.9048 | 24.5025 |
| tail113 | 0.5080 | 1.0196 | 14.9134 | 24.4846 |
| two_balanced | 0.4889 | 0.9306 | 14.9771 | 27.9890 |
| two_skewed | 0.5039 | 0.9739 | 14.9261 | 24.7643 |
| short_tail | 0.4823 | 0.8164 | 2.1174 | 4.3511 |
| compact | 0.3607 | 1.0051 | 4.0344 | 6.0703 |
| long100k | incomplete | incomplete | incomplete | incomplete |

These results do not demonstrate a broad whole-input gain over raw serial.
Complete long100k first, then assess native-call granularity independently.
Local validation:49 raw/ABI/boundary/scheduler/summary tests and focused
pre-commit passed. Device re-execution remains the user's next card run.
