# Replace exhausted-prefix history merge with complete two-way merge

| Project truth | Contract |
| --- | --- |
| Goal | Remove snapshot/fallback overhead while keeping real evolving history |
| Frozen | Fixed TLE local sort,block512 Cube,query16,two slots,scaling/masks,outer UB state commit,final output and compiler |
| Editable | Diagnostic helper/source,probe option,transition oracle and CPU tests |
| Strategies | Prefix-only exhausted merge and complete non-exhausted merge,extract first512 proposals |
| Acceptance | Prepared micro comparison only;production and autotune unchanged |

| Contract | Value |
| --- | --- |
| snapshot | Existing merge_exhaust_sort4,consumed counts,UB snapshot and tail gather |
| prefix | Same exhausted merge but return first512 immediately;no snapshot,consumed-count scalar work or tail gather |
| complete | Existing mrgsort with lengths512/512/0/0,valid3,exhaustion false,repeat1;2048 output words then extract1024 |
| Equality | Same sorted Top512 values and valid unique indices;tie order may differ |
| State | Every update consumes the preceding update result;no fixed history seed in the mixed kernels |
| Commit | Same outer to_buffer commit and barriers,not optimized in this probe |
| Tradeoff | Complete merge emits1024 proposals;may do more merge work than exhausted mode but eliminates counts/snapshot/fallback |
| No new native | Reuse existing registered mrgsort;no compiler or bitcode modification |

| Path | Behavior |
| --- | --- |
| Transition gate | Five scenarios,four evolving updates;new tiles all lower/all higher/interleaved/tied/with-infinity,exact CPU rank oracle and guards |
| Mixed gate | Same three fixtures,exact trace/selected values,guards and task counts before/after pair timing |
| Compile | Snapshot retains exhausted merge;complete has three mrgsort calls including two fixed local-sort stages,no exhausted merge |
| --history-full-merge | Requires raw-mixed-online,boundaries,cube-block512;excludes other diagnostic variants |
| Limits | One equivalent helper strategy removes coupled snapshot/fallback work;does not separately quantify each component |

For equal512-element input lists,exhaustion consumes512 entries from at least
one input. Its valid sorted output therefore contains at least512 proposals.
The prefix-only arm retains the exact exhausted merge and removes only the
now-unneeded fallback path. This reasoning requires fixed TOP=CHUNK=512 and
valid complete512-element packed lists,including padded -infinity entries.
Both new helpers statically reject other TOP/CHUNK values. All three arms keep
the original outer history commit and event sequence.

Local validation:79 raw/ABI/boundary/aggregation and helper checks plus five
complete-input schedule checks passed. CPU tests execute actual helper ASTs
with merge models through four evolving updates,verify the exhausted-prefix
lower bound,and check frozen outer commits and call binding. Python compilation,
Bash syntax,focused pre-commit and diff checks passed. NPU transition and mixed
kernel correctness/performance remain pending;no speedup claimed locally.
