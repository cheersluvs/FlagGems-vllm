# Batch local query sorting with existing TLE primitives

## Frozen project and contract

| Field | Decision |
| --- | --- |
| Current work | Step2 after packing-cost census; candidate only |
| Read-only | Production, CANN baseline, native Cube/bitcode source, compiler/packages, input generation and timing helper |
| Editable | New AST source factory, optional micro-worker mode, wrapper routing and tests |
| Reference | Existing raw score oracle and independent CPU per-row full sort |
| Input | Prepared INT8 Q/K, original query16 packing, FP16 weights/scales, four heads,D128 |
| Numerical semantics | Same FP16 two-Cube scoring and FP32 scaled scores; int32 compressed-group indices, TOP512; tie-aware ranking |
| Sampling | Two reversed paired rounds, three active invocations per arm, strict host/device counts, same trace capture guard |
| Active set | rows16,width512; rows17,width8256; rows112,width29696 |
| Acceptance | Prepared micro self-comparison only; no CANN/full-input acceptance mean |
| Unsupported | Batch other than2/4 rejected; no production fallback or registration |
| Autotune | Fixed unregistered diagnostic; no NVIDIA/autotune change |
| Timeout | Outer1800 seconds; compile, correctness, IR and timing in the same run |

Installed sources captured in report041640ab (raw-tle-sort512-20261117-025829)
show sort32 forwards repeat_times to vbitsort, and mrgsort forwards its repeat
count to vmrgsort4. No native source or library modification is needed. The
existing fixed512 network already relies on repeated independent lists for
32-to128 sorting. Candidate extends those repeats over2/4 contiguous query rows:
sort32 repeat16*batch, mrgsort repeat4*batch, mrgsort repeatbatch.
CPU execution of the actual generated call fragment checks independent sorted
scores/indices per query with distinct rows,ties,masked tails and all-invisible
rows. This verifies declared layout; NPU execution remains necessary.

## Paths

| Path | Behavior |
| --- | --- |
| prefix control | Existing proven prefix mixed kernel with one-query local sort |
| batch2 | Two queries' local sorting in three primitive calls, then two separate history updates |
| batch4 | Source/model available, but compile only when explicitly selected; start with batch2 |
| Frozen stages | Cube body,ring ready/free events,state layout,per-query history helper/commit,final unpack |
| Changed stages | Batch score gather/mask/KS broadcast,flattened local sort,constant per-query slices |

Each Vector subcore still handles eight queries. Local sorting calls fall from
24 per key block to12 with batch2, or6 with batch4. History updates remain eight.
This is a source schedule count, not a hardware instruction count. Input/index
flat ordering is row-major, and every512-element output list belongs to its
original query. Batch slices are compile-time static. Existing row/key masks,
trace sentinel initialization,rank/value/guard checks run before and after timing.
No claim is made that grouped broadcasting lowers efficiently or fits UB until
compiled and run. The IR gate checks retained local primitives,unrolled history
call counts,no fallback gather,and original mixed task modes; all artifacts are
saved. Larger temporary buffers and spills can outweigh reduced calls.

One command selects batch2 or batch4 with --query-batch2 expressed as
`--query-batch 2` (or4), together with existing history micro flags. Failed
compilation/correctness/count checks stop timing and still upload diagnostics.
No unmeasured fallback is presented as a successful candidate.

Packing census is closed: four representative fixtures passed across reports
b84d02a74e74841bbdeec19925e25c0477ffcd4f and
eba67a4cf84eff93c8552c3a658d3f72eb693d28. Independent K/KS stage costs were
6.02-6.59% of complete candidate timing. No packing reuse was implemented.

Local validation: three new source/network/IR checks plus81 existing raw checks,
Python/Bash syntax and focused pre-commit. NPU compile and performance pending.
