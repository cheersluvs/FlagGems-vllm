# Mixed critical-path ablations

| Project truth | Contract |
| --- | --- |
| Goal | Determine which reduction of work can shorten the prepared mixed kernel |
| Frozen | Block512 score body, query16 grouping, two slots, event sequence, masking, online512 TopK and compiler |
| Editable | Diagnostic AST source, lightweight AIC tick, probe mode and CPU tests |
| Fixtures | rows16,width512;rows17,width8256;rows112,width29696;two interleaved rounds |
| Acceptance | Diagnostic only; no CANN ratio, production promotion or phase subtraction |
| Build | Same isolated compiler copy; normal/.cube bitcode exports; mixed metadata checked before launch |

| Contract | Value |
| --- | --- |
| full | Existing block512 Cube plus unchanged online TopK |
| light | Same Cube, ring loads, scaling/masks/events; replace sort/merge/state with per-row/block finite-score checksum |
| replay | Lightweight AIC tick preserves ready/free/drain events; original Vector TopK reads immutable precomputed raw score tiles directly |
| Replay limitation | Bypasses ring payload writes and changes payload address/cache footprint; retains event/control overhead, not a pure Vector duration |
| Light limitation | Checksum adds a reduction and GM stores; not a pure Cube duration |
| No DCE | Exact trace in traced mode; checksum output in timed light mode; replay tick per group and exact TopK in timed replay mode |
| Integrity | Pre/post score checks, checksum/rank checks, guards and expected task counts |
| Timing | Prepared-device pair/triple ablation; preparation excluded, no additive/subtractive stage attribution |

| Path | Behavior |
| --- | --- |
| --critical-path | Requires raw-mixed-online,boundaries,cube-block512; three diagnostic arms |
| full output | Exact trace/selected values plus guards against original score reference |
| light output | Exact trace, tolerant checksum of same visible scores, untouched index/value buffers |
| replay output | Exact trace/selected values and final per-group progress tick |
| Production/autotune | Unchanged; diagnostic fixed configs, no registration or tuning claims |

Local validation:58 raw/ABI/boundary/aggregation and ablation checks passed,
plus five complete-input schedule checks. Tests compare exact AST bodies and
event sequences, verify visible checksum stores remain outside TRACE branches,
and execute the native scalar tick body with host counting stubs. Host checks
are not CANN numerical or scheduling validation. Python compilation,Bash syntax,
focused pre-commit and diff checks passed. Card validation remains pending.

## Argument binding correction

Report cae4bbd6c65d9cb848fdd80185f66f1302e4cbff failed at the first
full-kernel warmup:both appended positional payload pointers consumed WIDTH and
SLOTS before WIDTH was also supplied by keyword. No target kernel was compiled
or executed and no timing was produced. Replay and Audit now precede the first
constexpr configuration argument. A regression binds all eleven payload pointers
and four keyword configs for each actual generated signature,and verifies the
previous ordering reproduces the duplicate-WIDTH failure. Arithmetic,events,
fixtures and measurement options are unchanged.
