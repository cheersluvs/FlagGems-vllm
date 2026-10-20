# QSA evict_first matched-configuration experiment

Pre-coding plan, 2026-10-20. The retained baseline is the eight-choice
key-major/compact candidate validated by report 4b77a6f, code b60b59f.

| Project truth | Frozen choice |
| --- | --- |
| Hypothesis | Adding evict_first only to the rectangular scorer's K-cache load may improve streaming-cache behavior; repeated K reuse can also make it slower. No directional gain is assumed |
| Editable | Private hinted score copy/factory, matched-config preparation, CPU/GPU tests, runner selection/incremental stage and docs |
| Read-only | Production SHA256 d70faeb6928c2c7a56636ded833f3359368871e353ecf4c66c6dfd35cae93e73; retained key-major/compact/reuse modules; all eight configs; native pinned source; TopK; all 22 workloads and tolerances |
| Single change | eviction_policy="evict_first" on keys=tl.load; query and table loads, masks, dot/reduction/store, loop staging, dtype and grid remain identical |
| Matched selection | Warm the retained eight-choice public entry in actual chunk order; derive its compiled launch config, then construct a private one-choice hinted tuner using that exact config. Reject ambiguous/mismatched configs rather than silently retuning |
| Autotune | Baseline uses existing qsa_select_paged_prefill_reuse_score (8); standalone correctness tests can use all8 on the hinted JIT. The measured hint candidate deliberately inherits baseline's choice; no new YAML space |
| Build / validation | Syntax/CPU contracts/pre-commit locally; full existing GPU suites plus every hinted config and direct native/FP32/full-entry checks on H800 |
| Measurement | Same verified FlagTree/ptxas13.4/CUDA13.4, kernel mode, score excluding TopK; native score/TopK/pipeline, frozen-production score and matched score_vs_reuse |
| Timing / admission | 3 alternating rounds, warmup200ms, rep100ms, initial limit80C, actual thermal slowdown/foreign-process rejection; worker1800s, full accuracy7200s |
| Profiling | No ncu/nsys in preceding run. Capture actual matched configs, PTX/CUBIN hashes, shared/register/spill metadata and static policy/hint sites outside timing |
| Keep/revert | Use score_vs_reuse for this trial's increment, upstream12 and observed8 separately. Three-sample control noise is not an algorithmic improvement |

| Alignment contract | Preserved behavior |
| --- | --- |
| Inputs / output | BF16/FP8, current parameterized H/D and layouts, metadata types, compressed selection and -1 padding |
| Numerics | Same FP32 accumulation, real-head ReLU/sum and validity masks; independent FP32/native checks and unchanged tolerances |
| Host / memory | Same cloned public/launch bytecode, validation, per-call allocations, workspace shapes/chunks, stream and lifetime |
| Fallback | NO_TORCH_COMPUTE_FALLBACK; no host device-value reads or copies in the candidate |

| Path | Choice |
| --- | --- |
| Empty / select all | Original behavior; no score/TopK measurement |
| Narrow compact | Unchanged compact kernel without a new hint; ragged54 is a control |
| Rectangular | Key-major copy differing only in the K-load hint, with baseline-matched config in benchmark |
| Baselines | Original native and frozen production; retained reuse module uses all8 choices and an independent outer cache |

Matching is benchmark preparation outside timing. It prevents a change in
tile choice from being mistaken for a cache-hint effect. Record the selected
configuration and verify both sides still use it after warmup and after all
timed stages. Workloads without a rectangular score report matching N/A.
The full-entry hinted factory retains an eight-choice tuner for correctness
coverage; performance conclusions in this experiment apply to matched tiles,
not a separately retuned hinted implementation. No new export is needed for
this private experiment. Production and retained-best files remain untouched.

## Run and inspect

Use the existing automatic-upload wrapper with `--candidate
key-major-evict-first --suite combined-prefill --mode kernel --rounds 3
--kernel-warmup-ms 200 --kernel-rep-ms 100 --max-initial-temp-c 80` in the
verified vLLM0.31/FlagTree environment. The full gate includes all previous
tests plus 26 hint/matching tests; expected on one SM90 is 268 passed and the
existing second-device skip. This is a run expectation, not a GPU result.

The incremental score stage is `score_vs_reuse`, baseline_kind
`retained_reuse`; it replaces `score_vs_key_major` for this candidate. The
native score, common-logits TopK, sequential pipeline and frozen-production
score stages remain, for 102 timed stages on the combined matrix. Select-all
and compact controls remain explicit.

`eviction_config_match` records the baseline-selected configuration and
before/after-timing verification. `retained_baseline_score_configs` and
`retained_baseline_kind` identify the eight-choice baseline; the older
`retained_key_major_score_configs` field is an equal legacy alias, not a
six-choice baseline in this experiment. Resource records likewise include
the retained baseline kind. `ptx_cache_policy_sites` contains static text-site
counts for fractional createpolicy, evict_first qualifiers and cp.async L2
hints; absence of a lowering artifact is not converted into a performance
claim. These counts and all matching/JIT operations occur outside timing.

No hint is added to the compact scorer in this trial. A favorable rectangular
result would justify a separate integration/retuning decision, not automatic
enablement of the policy on every path or future backend.

Local verification: 92 CPU tests ran, 91 passed and one optional Torch test
skipped; syntax/pre-commit passed. Tests remove the one K-load keyword and
compare the complete scoring AST to retained key-major; execute the actual
matching/factory code and worker preparation order; reject ambiguous,
unknown, prematurely warmed or changed configurations; preserve compact and
select-all controls; check static policy-site parsing, stage labels, full
accuracy coverage and success/failure report upload. Production, TopK, all
eight YAML configurations and retained candidate files match the validated
4b77a6f source snapshots. H800 compilation, correctness and hint performance
remain pending; no numerical or performance result is claimed from CPU tests.
