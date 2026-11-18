# Bind prefix merge output to the UB history slot

| Project | Configuration |
| --- | --- |
| Scope | Ascend 910B prepared mixed-kernel probe in tools |
| Control | Existing query16, two-slot, block512, one-query local sort and prefix merge |
| Candidate | Same producer and Vector schedule with the merge destination bound to its own query's persistent UB slot |
| Fixtures | 17 queries by 8256 keys and 112 queries by 29696 keys |
| Timing | Two interleaved rounds with three measured invocations per arm, validated device and host task counts, and a 3600-second outer timeout |
| Metric | Control latency divided by candidate latency in the prepared mixed kernel |
| Compiler | Existing FlagTree corefix environment and registered TLE merge primitive |
| Autotune | Fixed diagnostic configuration, outside production registration |

| Contract | Check |
| --- | --- |
| Input | Existing prepared INT8 Q/K, FP16 weights/scales, query and key tails, and shared page mapping |
| Output | Existing FP32 scores and INT32 Top512 compressed-group indices with exact ranking by value, valid unique indices, and masked tails |
| Workspace | 2048 FP32 words per query for up to 1024 output proposals, 64 KiB of persistent history per Vector subcore |
| Aliasing | The old 1024-word prefix is inserted into independent packed input before merge output writes the same history slot |
| Synchronization | Existing post-update, tile and cross-core synchronization stays in place |
| Validation | Compare both arms with the existing CPU/standalone score oracle and pre/post score trace, ranking and guard checks |

| Path | Check before timing |
| --- | --- |
| Prefix control | IR contains initialization and hot-loop copies into history, with a temporary merge destination |
| Direct candidate | IR contains only the initialization copy into history and the merge output resolves to the same UB allocation read by final unpack |
| Both | Expected primitive calls and task mode, saved compilation artifacts, exact device oracle checks |

The direct arm initializes the full enlarged slot and reads only its first
1024 words for the next update and final unpack. An IR failure is reported with
the available compiled artifacts. The compile gate verifies the intent at
pre-device IR level. Card correctness checks test the generated executable.
Timing isolates the prepared mixed kernel and excludes packing.

Enable with `--history-direct` alongside the existing raw mixed, boundaries,
block512, and history merge flags. Upload uses the existing wrapper.

## First card build and explicit result binding

Report 0de21c22c10fc607e41e1635cbfdf1e3816a3831 compiled both prefix control
variants. The first direct variant failed in TritonToLinalgIncubated
with an unresolved materialization for tensor<512xi32>. The preserved TTIR
contains the merge with the UB destination operand, but its results have no
SSA consumers. This is a lowering failure before device execution. The report
does not identify the exact conversion bug.

The follow-up adds an SSA consumer by binding the full merge result back
to the same destination. The IR gate still requires a direct destination
and no update copy into history. It will reject an extra materialization
even if compilation succeeds. Whether explicit binding resolves the
conversion and elides that copy requires a card run.
