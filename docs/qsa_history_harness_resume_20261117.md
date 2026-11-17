# Resume history-prefix harness after runtime/profiler failures

| Frozen | Contract |
| --- | --- |
| Candidate/baseline/timer/compiler | Unchanged from 10c0a209 |
| Editable | Runner, runner tests, this evidence note |
| Reuse | Revalidate seven complete cases and the same boundary policy |
| Remeasure | short_tail,compact,long100k; no boundary retuning |
| Acceptance | All ten complete stock harness kernel-mode CANN/candidate ratios, arithmetic mean |

Report: `06b9218b0d0b4f6c00d0a07a20cb2834126e3f94`. Boundary checks passed; threshold four, upper bound 58 updates.

`short_tail` failed during NPU set_device with 507033/E39007 startup timeout, before candidate execution. `long100k` passed prechecks but the profiler reported missing device tasks/TASK table and an invocation count mismatch; baseline latency became infinity. Postchecks did not complete. `compact` passed correctness but its CANN measurement was 1234.856 ms (62.456x), and both snapshot paths differed despite identical dispatch. This result is flagged for independent remeasurement, not counted as a confirmed gain. No ten-case mean is available.

| Case | CANN ms | Candidate ms | CANN/candidate | Snapshot/candidate | Persistent TopK ms | Snapshot online TopK ms | Dispatched online TopK ms |
| --- | --- | --- | --- | --- | --- | --- | --- |
| int8_first_chunk | 8.038763 | 12.286518 | 0.654275 | 1.050166 | 7.598672 | 10.339808 | 9.690563 |
| int8_budget_exact | 14.278199 | 23.537126 | 0.606625 | 1.130941 | 11.742276 | 23.624398 | 20.523934 |
| int8_dominant_long | 14.747095 | 24.381375 | 0.604851 | 1.131679 | 14.910739 | 24.548242 | 21.250422 |
| int8_tail112 | 14.318461 | 23.650416 | 0.605421 | 1.128871 | 14.914104 | 23.746339 | 20.593131 |
| int8_tail113 | 14.299264 | 23.691429 | 0.603563 | 1.128454 | 14.911627 | 23.744248 | 20.593645 |
| int8_two_balanced | 15.574070 | 26.574390 | 0.586056 | 1.139049 | 14.979031 | 27.108070 | 23.309500 |
| int8_two_skewed | 14.343478 | 23.895928 | 0.600248 | 1.130332 | 14.913139 | 23.991418 | 20.801076 |
| int8_compact (suspect) | 1234.855810 | 19.771649 | 62.455882 | 0.914183 | 145.409952 | 8.850590 | 8.863715 |

TopK values above are independent harness measurements on staged full raw scores; they are not CANN stage timings or internal fused-pipeline durations. Tie-aware rank and exact raw-score checks passed for the eight completed cases; differing tied index sets are allowed.

The new `--rerun-cases` option requires `--resume-report`. Forced cases are excluded from both copied artifacts and retained outcomes, while source reports are left untouched. Provenance records every forced rerun. Existing source-diff and correctness gates remain in effect.

Local checks: runner/policy tests, pre-commit and syntax checks; replay of this actual report retained exactly the first seven cases. No new NPU measurements were performed locally.
