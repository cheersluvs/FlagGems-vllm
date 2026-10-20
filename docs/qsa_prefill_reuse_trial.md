# QSA key-major query-tile reuse trial

Pre-coding plan, 2026-10-20. This is a configuration/work-distribution
experiment on the retained key-major scorer, not a new score algorithm.

| Project truth | Frozen choice |
| --- | --- |
| Best baseline | Key-major + narrow compact, code 83a19d9, report 6b3b7c0; same JIT arithmetic and original six configs |
| Production/native | Production SHA256 d70faeb6928c2c7a56636ded833f3359368871e353ecf4c66c6dfd35cae93e73; unchanged pinned native vLLM 7a877ae |
| Evidence | Key-major TR64/BN64/KT16/warps4 uses 232 registers/thread on upstream cases, 255 on two observed cases; no spills. TR32 uses 118/120 registers in measured variants. This is resource evidence, not proof of a hardware bottleneck |
| Hypothesis | Distributing a 64-query tile across eight warps may make its K/page-table reuse more efficient than the existing four-warp choice. KT4 and KT16 bound the per-program query reuse/launch tradeoff |
| Editable | Separate trial factory, separate YAML tuning section, GPU/CPU tests, benchmark selection/direct-best-baseline stage, docs |
| Read-only | Score JIT source, original six configs, compact gate/kernel, public production source, native source, TopK, tolerances and 22 workloads |
| Autotune | New qsa_select_paged_prefill_reuse_score: original six plus TR64/BN64/KT4/warps8 and TR64/BN64/KT16/warps8; stages2 throughout; same original keys/strategy |
| Build/validation | CPU contracts, syntax/pre-commit; full production/compact/key-major tests plus new configs and full reuse entry on H800 |
| Benchmark | Same verified FlagTree/ptxas13.4/CUDA13.4; score/kernel acceptance, plus unchanged TopK/pipeline and direct score vs frozen production and retained key-major |
| Timing | 3 alternating rounds, warmup200ms, rep100ms, initial temperature<=80C; actual thermal slowdown and foreign processes still reject; full accuracy7200s, worker1800s |
| Profiling | Prior H800 has no ncu/nsys. Capture actual configs, registers/spills/shared bytes/static PTX and before/after correctness; no hardware-counter claim |
| Keep/revert | Evaluate incremental score_vs_key_major, separate upstream12/observed8; original-native score remains acceptance metric. No on-card improvement claimed yet |

| Alignment contract | Preserved behavior |
| --- | --- |
| Numerics | Exact existing key-major JITFunction object, FP32 accumulation, real-head ReLU/sum, unchanged dtype/mask/bounds/tolerances |
| Inputs/output | Original BF16/FP8, arbitrary supported H/D and strides, device metadata, compressed indices and -1 padding |
| Resources | Same public bytecode, allocations, chunk sizes, stream and workspace lifetime; no global workspace or Torch compute fallback |
| Caches | Separate LibEntry instances/dictionaries and separate config-set tuner hashes; retained baseline warmed in public chunk order before candidate |

| Path | Choice |
| --- | --- |
| Empty/select-all | Original entry; score/TopK N/A |
| Narrow compact gate | Existing compact scorer unchanged |
| Rectangular | Same key-major JIT, private eight-choice libtuner; original six choices remain available |
| Unsupported/fallback | Original validation and errors, no Torch compute/copy/cast |

This trial does not add eviction hints, explicit loop staging, page-table
hoisting or a new precision mode. Increasing TILE_R permits one K/page-table
tile to serve more query rows; increasing K_TILES permits a loaded query tile
to serve more columns. The actual best choices and occupancy tradeoff must
come from H800 measurements. The new baseline stage prevents attributing the
previous 1.393x layout gain to this trial. All applicable stages, including
TopK, remain in the report; automatic upload runs on success and failure.

## Current implementation versus pinned native vLLM

| Aspect | Native | Frozen production | Retained candidate |
| --- | --- | --- | --- |
| Dot layout | keys x query | query x keys | keys x query on rectangular path; compact retains prior arithmetic |
| Score tiles | BF16 TR64/BN64/KT16/warps4; FP8 TR32/BN64/KT16/warps8 | Six libtuner choices | Same six; this private trial adds two alternatives |
| Loop | tl.range(..., num_stages=2) | range, launch config num_stages=2 | Same as production; these two staging knobs are not equivalent |
| Key load | eviction_policy="evict_first" | No explicit hint | No explicit hint |
| Scheduling | Rectangular request/query/key grid | Rectangular | Narrow compact route for sparse ragged batches |
| Bounds/layout | Native trusted page-table indexing, unit table inner stride | Explicit table stride, valid physical-page masks, extra small-H/D padding | Preserved |
| Host/allocations | Flat worst-case workspace and per-chunk setup | Retained validation/metadata/2D workspace changes and select-all shortcut | Preserved |
| TopK | Native CUDA dispatcher | Repository persistent_topk | Same repository persistent_topk |

`evict_first` is a performance hint, not a semantic prerequisite or a command
to bypass caching. NVIDIA describes it as lower eviction priority suitable
for streaming data. The same K can also be reused across query tiles, so
restoring it might help cache pollution or hurt reuse; the sign is unmeasured.
It should be evaluated in a separate hint-only A/B after this tile trial.
The verified newer toolchain already runs unchanged native code with its hint;
the historical ptxas12.8 fractional-policy fault is not a reason to claim it
must be permanently removed. Portability requires backend-specific validation.

Sources: [Triton load API](https://triton-lang.org/main/python-api/generated/triton.language.load.html),
[NVIDIA PTX cache eviction hints](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#cache-eviction-priority-hints).

## Run and evaluate

Use `tools/run_qsa_prefill_flagtree_ab_h800.sh --flagtree /opt/flagtree
--candidate key-major-reuse --suite combined-prefill --mode kernel --rounds 3
--kernel-warmup-ms 200 --kernel-rep-ms 100 --max-initial-temp-c 80` in the
existing vLLM0.31 environment. Keep the wrapper's automatic report upload.
Full accuracy adds 24 cases; expected on one SM90 device is 242 passed plus
the existing second-device skip, subject to the actual GPU result.

The new `score_vs_key_major` stage is the incremental comparison against the
retained best scorer with the original six configurations; its baseline_kind
is `retained_key_major`. It uses the same preallocated score buffer, full
public baseline warmup order, independent outer caches and score checks as
`score_vs_production`. Both baseline config sets are checked for mutation
after timing. `score`, `topk`, `pipeline` and `score_vs_production` remain.
The combined matrix has 102 timed stages, with select-all still pipeline-only.

Capture `retained_key_major_score_configs` and compiled resource records next
to the new candidate's selections. If both choose an original config their
PTX/CUBIN should agree; any small timing difference is a control observation,
not evidence that the arithmetic changed. A selected new config is also not
itself proof of improvement: use the paired incremental stage and report any
regression. No best-config whitelist is chosen before on-card evidence.

Local verification: 83 CPU tests ran, 82 passed and one optional Torch test
skipped because Torch is absent on the Mac; syntax/pre-commit passed. Tests
execute the actual factory, preserve the exact JIT object/keys and original
config choices, execute LibTuner's actual config-hash properties to verify
cache separation, check the incremental report and required correctness
coverage, and retain success/failure report-upload coverage. No production
Torch compute fallback or new kernel arithmetic was introduced. CUDA
compilation, full correctness, actual winning configurations and incremental
speedups remain pending H800; do not infer a gain from the resource hypothesis.

## H800 decision, report 4b77a6f

Report `4b77a6f0726549275bc5d928e513b092499aa30e`, code `b60b59f`:
242 passed, one optional second-SM90 skip and all 22 cases complete. All 44
source identities, same compiler/runtime/GPU, cache checks, 612 raw timing
samples and 408 timing telemetry points were verified. See
`docs/qsa_prefill_reuse_h800_results.md` for all score/TopK/pipeline latencies.

| Group | Score/native | Score/frozen production | Incremental score/retained key-major | TopK/native | Pipeline/native |
| --- | ---: | ---: | ---: | ---: | ---: |
| Upstream12 | 1.05793 | 1.40956 | 1.01522 | 0.89457 | 0.97100 |
| Observed8 scored / 10 pipeline | 2.00673 | 1.14843 | 1.00714 | 0.89945 | 1.93582 |

Keep the private eight-choice candidate as the next trial baseline, not a
production integration or a guarantee across workloads/vendors. Seven
upstream cases select a new choice; six have all three paired incremental
ratios above one, one is essentially flat. Twelve rectangular controls use
identical old configurations/PTX/CUBIN on both sides. Observed workloads
select no new choice; their small aggregate shift is not credited to this
optimization. Ragged54 retains the same compact scorer.

The n1/q16384/p51200 pair changes only warps4 to warps8 at TR64/BN64/KT16:
232 to 118 registers/thread, 106496 to 114688 shared bytes, no spills, 1.07049x
incremental speedup. Twice as many threads means the per-CTA register budget
has not halved; there are no hardware counters proving an occupancy cause.

The shorter protocol was used. Full correctness took 514.82s; first-to-last
benchmark telemetry spanned 1015.80s, versus 1929.79s in the preceding run,
despite twenty additional incremental comparison stages. These spans include
startup/compile/validation, not just timers, and are not an equal-work total
runtime experiment. Temperature32–52C, SM clocks1965–1980MHz; only one
trailing-utilization admission waited10s, with no temperature cooldown.

No eviction hint was added or timed. A separate hint-only A/B remains the
next proposed experiment; TopK and the score-only NV acceptance rule stay
unchanged.
