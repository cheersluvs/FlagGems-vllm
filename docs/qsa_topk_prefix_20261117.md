# Online TopK prefix diagnostics

| Project truth | Contract |
| --- | --- |
| Goal | Identify local-sort versus merge-helper pressure after Vector-dominated critical-path evidence |
| Frozen | Block512 Cube, query16, ready/free/drain events, masks, update count, sort and UB-snapshot merge helpers, compiler |
| Editable | Diagnostic AST prefix source, probe flag, validation and tests |
| Timing | Same three prepared fixtures,two interleaved rounds; compile/validate/time in one card run |
| Acceptance | No CANN ratio or ten-input acceptance mean; production and autotune unchanged |

| Contract | Value |
| --- | --- |
| full | Original running UB history,sort,merge,commit and final unpack |
| sort | Per-tile original sort,rank-weighted finite sorted-value checksum; no history |
| merge | Same sort plus original snapshot/fallback merge against fixed nonempty512-element seed,checksum; no running commit/final unpack |
| Seed | FP32 descending0.5..-0.5,valid packed indices0..511; loaded to UB once per row before updates |
| Observable work | Checksum every row/block in both trace modes; rank weighting observes ordering; guards and CPU value oracle |
| Boundaries | Preserve row/key masks,including zero visibility; seed is synthetic diagnostic history,not model output |
| Interpretation | Changing state evolution/export changes workload; no subtraction into exact stage durations; snapshot/fallback remains inside merge helper |

| Path | Validation |
| --- | --- |
| full | Existing exact trace and selected values,rank and guards |
| sort/merge | Exact score trace,CPU sorted-value checksum with1e-3 tolerance,untouched final-output buffers and guards |
| --topk-prefix | Requires critical-path,cube-block512,boundaries,raw-mixed-online |
| Later | Only after evidence,split the merge helper snapshot/fallback versus raw merge; do not assume this run isolates commit alone |

Local validation:63 raw/ABI/boundary/aggregation/prefix checks and five
complete-input schedule checks passed. Prefix tests compare the actual Cube
and event ASTs,bind all twelve pointer operands before constexpr configs,and
check cut boundaries plus observable outputs. Python compilation,Bash syntax,
focused pre-commit and diff checks passed. No NPU result is claimed locally.
