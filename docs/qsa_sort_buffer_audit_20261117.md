# Audit sort512 and test explicit scratch/output lifetime

| Project truth | Contract |
| --- | --- |
| Evidence | Report29a5b32184eb1aa4e70d0083f455a5e6c64f89b7 full trace0 bcmlir calls custom_sort_1d_pack_float on512xf32,2048xf32 scratch and1024xf32 output with descending=true,TOPK512,impl0 |
| Observed IR | Scratch and output have zero-filled tensor operands before custom lowering; emitted device fill/copy cost remains unproven |
| Upstream reference | sort_base_impl prepares index,block-sorts and merges into scratch,then copies all1024 output words; source may differ from installed wheel |
| Goal | Capture actual installed sources/bitcode and test whether initialization can be omitted without changing sort math |
| Frozen | Cube,query16,two slots,TopK algorithm,sort_impl0,history helper,masks,events,compiler,fixtures |
| Editable | Diagnostic source factory,installed audit,probe flag/tests and runner |
| Acceptance | Prepared micro self-comparison only; no production or autotune changes |

| Contract | Value |
| --- | --- |
| zero | Original tl.zeros scratch/output |
| uninit | Explicit Vector UB scratch2048 and output1024,allocated once and reused in sequential row/update calls |
| poison | Same explicit buffers,NaN-filled before every sort; validation only,not timed |
| Lifetime | Sort scratch/output are distinct from persistent TopK state and merge scratch; reuse only after prior merge and original barriers |
| Numerical gate | Pre/post exact score trace and selected values,rank checks,guards for all three arms |
| Poison limitation | Empirical initialized-read detection,not a proof for all possible payloads |
| Compiler evidence | Save all IR/binaries; verify actual custom sort call and shapes remain; do not equate high-level fills with hardware cost |

| Path | Behavior |
| --- | --- |
| --sort-buffers | Requires raw-mixed-online,boundaries,cube-block512; excludes critical-path/topk-prefix |
| Audit | Installed source/hash,registry,library hash and compressed bitcode; LLVM sort-related excerpt where available |
| Timing | zero/uninit only,three representative fixtures,two interleaved rounds; poison gate runs first |
| Fallback | No hidden fallback or compiler/library patch; failures remain failures |

## Audit findings before the card run

The upstream triton_v3.5.x implementation dispatches sort_impl0 to sort_base_impl.
For N512 descending it generates512 indices,uses vbitsort on sixteen32-item
lists,then merges32-to128-to512 with vmrgsort4 and copies1024 packed words to
the destination. The generic padding function still contains Vector/Scalar
ordering even when no padding is needed. These are reference-source findings;
the installed header/library capture must identify whether they apply locally.

The previous actual full-kernel bcmlir has zero-filled2048-word scratch and
1024-word destination operands passed to the sort custom op. It also has
separate UB state snapshot copies for the later history merge. This probe
changes only sort scratch/output,not those history copies. No device-level
fill/copy overhead or removable barrier cost is claimed from this IR alone.

Local validation:67 raw/ABI/boundary/aggregation/source-mutation checks and five
complete-input schedule checks passed. Tests bind the real generated signatures,
verify unchanged Cube/events/sort arguments,poison both buffers before each
sort,and reject missing sort IR evidence. Python compilation,Bash syntax,
focused pre-commit and diff checks passed. Installed implementation evidence,
NaN-poison validation and NPU timings remain the next card run.
