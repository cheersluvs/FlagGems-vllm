# QSA observed workloads and compact request scheduling

Pre-coding plan, 2026-10-20. Baseline production SHA256:
`d70faeb6928c2c7a56636ded833f3359368871e353ecf4c66c6dfd35cae93e73`.
The user requests ten observed-envelope workloads and the first portable
optimization. `optimization.md` and `deep_opt.md` are absent from this checkout;
`AGENTS.md` and `workflow.md` apply. No performance improvement is claimed yet.

| Project truth | Frozen choice |
| --- | --- |
| Operator/reference | Existing QSA prefill API; native vLLM commit 7a877ae34dcf4a7d46de08eb16db4699e1e0047c, SHA256-pinned by the runner. No corresponding single Torch API |
| Editable | Observed workload module, benchmark integration, isolated compact scheduling trial, CPU/GPU tests, runner plumbing and documentation |
| Read-only | Production operator, TopK, native reference, six existing score tuning candidates and numerical tolerances |
| Build | Python sources/JIT; local compileall/pre-commit; GPU compilation and autotuning are untimed on H800 |
| Validation | Existing full production suite, every compact tile config against FP32 Torch, full candidate vs native selection, per-case score/selection checks before/after timing |
| Active set | Upstream 12 plus observed-envelope 10, BF16 H4 D128 R4 block_k512 page196 budget512MiB; two observed cases select all |
| Benchmark | Same verified FlagTree + ptxas13.4 + CUDA13.4 compatibility runtime, --mode kernel, score excluding TopK; paired seven rounds, latency us, baseline/candidate >1 is better |
| Aggregation | Separate upstream12 and observed8 scored geometric means; two fast paths pipeline only; compare trial scoring directly against unchanged production and native vLLM |
| Timeout | Full correctness 7200s, each worker 1800s, cooldown 900s; all configurable |
| Profiler | Mac has no GPU; H800 ncu availability unknown. First evidence: actual CTA grid/configs, score-only paired times, per-chunk dimensions and source identities |
| Autotune | Trial reuses qsa_select_paged_prefill_score six configs, identical keys/strategy and constexpr mapping, separate kernel identity/cache. Fill retains existing no-autotune exemption |
| Fallback | NO_TORCH_COMPUTE_FALLBACK; trial falls back to actual production scorer for equal-length/single-request chunks |
| Decision | Experimental private module until on-card correctness and stable benefit; never infer acceptance from shared-card results |

| Contract | Preserved behavior |
| --- | --- |
| Input | Current SM90 CUDA, BF16/FP8 E4M3FN matching Q/cache; arbitrary positive H/D, existing nonoverlapping strided layouts and integer metadata |
| Output | Existing None API, caller int32 block indices, logical compressed indices, -1 padding, unspecified tie ordering |
| Numerics | Existing dot, FP32 accumulation, ReLU per head then sum heads, finite-input precondition; no tolerance changes |
| Metadata | Device starts can include empty requests and nonzero base; chunk intersection must be respected; no CPU reads in trial production-like path |
| Workspace | Original logits width, chunk budget, scratch, stream and allocation behavior preserved |
| Boundaries | Existing empty/select-all, invalid-page handling and unsupported checks retained; forward-only, no autograd |
| Reference Torch | Test/benchmark only. Trial contains metadata reads and Triton arithmetic; no Torch compute/copy/cast |

| Path | Implementation / test |
| --- | --- |
| Empty/select-all | Original public bytecode; capture actual fill launch for pipeline timing; score/TopK N/A |
| Equal-length, one request, or chunk_rows <= max_query_len | Original score-launch bytecode/JIT with an isolated outer LibEntry cache, no compact lookup overhead |
| Ragged envelope with larger chunks and rectangular row capacity >= 4 * chunk_rows | Metadata-only routing; compact tile slots use floor(clipped request start/TILE_R)+request, binary-search device starts |
| Compact kernel | Grid floor(chunk_rows/TILE_R)+requests bounds row tiles plus at most one empty slot per request. All score arithmetic after row selection is identical to baseline |
| Baselines | Unmodified native vLLM and current production, same compiler/input/budget/timer. TopK is unchanged |
| Tests | Request coverage exactly once under empty requests, arbitrary offsets, every tile size, chunk cuts and nonzero base; each GPU config and full native alignment |

The ten recorded envelopes come from upload commit
`6cbcf092785f266ac39f1f68f110d45b1f519489`, archive SHA256
`55944bc31b07cff46498c5ee36559f1290955fdefc4ebc7131336ad83a57eb7a`.
Missing dtype/layout/page allocation and per-request context lengths are
synthetic using the previous defaults, explicitly authorized by the user.
All contexts equal the recorded max_seqlen. The 54-request split is synthetic:
`[3200] + [112]*38 + [96]*15`; its sum, maximum and request count match the log.
This reconstructs recorded envelopes, not an exact runtime replay. Counts from
overlapping captures are not combined or used as benchmark weights.

For the 54-request envelope and TILE_R=16 the rectangular row grid has
54 * ceil(3200/16) = 10800 slots; the compact grid has floor(8896/16) + 54 = 610.
Key tile count is unchanged. This is a CTA-count comparison, not a predicted
17.7x speedup: binary search and useful computation still cost time. The two
32K boundary cases exercise a large compact chunk; the final 64-row chunk of
the second case retains the original production grid.

## Run and interpret

The normal pytest benchmark now uses all 22 cases for full-entry measurements,
and 20 for explicit score/TopK stages. The same-compiler runner accepts
`--suite upstream-prefill` (12), `observed-prefill` (10), or `combined-prefill`
(22). Its backward-compatible default remains upstream-prefill. The old
source-derived inference probes remain diagnostic fixtures, outside this set.

Run `tools/run_qsa_prefill_flagtree_ab_h800.sh --flagtree /opt/flagtree
--mode kernel --suite combined-prefill --candidate compact-ragged` with the
existing vLLM 0.31 environment. The runner performs full correctness first,
then reports native-vLLM score/TopK/pipeline comparisons and a fourth
`score_vs_production` comparison that isolates this optimization. Production
and trial use the same preallocated score buffer for this fourth comparison.
Native TopK comparisons still use frozen native logits, never candidate logits.
Before candidate warmup, the actual production public entry warms its own cache
in real chunk order. The candidate's original-kernel fallback has independent
per-device LibEntry dictionaries; the unchanged inner tuner still keys on rows.
`score_cache_isolation`, `baseline_production_score_configs` and
`candidate_fallback_score_configs` expose this separation. The runner rejects
shared outer caches and any baseline configuration change after public warmup.

`triton-stage-ab.tsv` includes `baseline_kind` (native_vllm or production).
`grouped-speedups.json` and the summary separate upstream and observed means.
Select-all cases have score/TopK status `not_applicable`, with blank ratios;
only complete validated groups receive means. Raw paired rounds, actual
compiled kernel configs/hashes, theoretical row grids and routed compact
chunk offsets are saved. The compact binary is compiled/autotuned before any
timer starts. H800 correctness is expected to report 194 passed plus the
existing second-GPU skip when exactly one GPU is exposed; actual results
must come from the card.

The wrapper automatically uploads JSON/TXT/TSV on success and failure. Existing
`--shared-gpu-exploratory --shared-min-free-mib 5120` is supported when sharing
a card; all such measurements remain ineligible for performance acceptance.
Use an idle card without that flag for an acceptance run. No host driver or
virtualenv package is replaced by the verified compatibility launcher.

## Local verification and pending decision

CPU suite: 56 tests, 55 passed and one optional Torch-dependent test skipped
because Torch is absent on the Mac. Checks execute the actual kernel mapping
AST across randomized request/chunk boundaries, compare the complete scoring
arithmetic AST to production, verify private routing/fallback and identical
allocation/TopK calls, and test isolated CLI, stage labels and report upload.
Pre-commit checks cover all edited Python/document files. These are not GPU
numerical or performance results; JIT compilation and the keep/revert decision
are pending the H800 run.

```bash
PYTHONPATH=tools python3 -m unittest \
  test_qsa_prefill_observed_compact test_qsa_prefill_shared_gpu \
  test_qsa_prefill_flagtree_ab test_qsa_prefill_triton_ab \
  test_qsa_prefill_triton_runner test_qsa_prefill_upstream_workloads \
  test_qsa_prefill_inference_workloads
```

No-Torch production-path audit: the candidate has no Torch API calls. Its
`.to(tl.int64)` is existing Triton device arithmetic for page pointers, not a
Tensor conversion. Host `.view` and slicing are no-copy views, and production
validation/allocation/TopK callables are shared unchanged. The existing public
registration is retained; this private trial needs no new registration/export.

## First H800 result and cache-isolation correction

Report commit `518675671f7b475675f20bbe6c9696c0fbce57e1`, code `90c14c4`:
194 correctness tests passed and only the second-GPU test skipped. All 22 cases
completed. Same compiler identities, 1148 timing samples, 656 telemetry points,
no sampled co-resident processes or thermal slowdown; shared exploratory mode
remains ineligible for formal acceptance.

| Scored compact path | Production us | Trial us | Ratio | Decision |
| --- | ---: | ---: | ---: | --- |
| ragged54 | 82.976 | 73.712 | 1.12568 | Promising; seven paired ratios 1.12451–1.12864 |
| unequal2 | 755.264 | 738.400 | 1.02284 | Unstable; paired ratios cross 1 |
| chunk_boundary | 670.992 | 693.056 | 0.96816 | Regression in all seven pairs |
| chunk_tail64 | 940.464 | 683.584 | 1.37578 | Invalid production-baseline attribution; replay required |

The old trial shared its fallback LibEntry instance with the production
baseline. It called that entry first with the 64-row tail; the captured
production config is TILE_R=8. LibEntry's outer key reduces do_not_specialize
integer scalars to their type, so the subsequent 16304-row production baseline
reuses this tail config without calling the inner row-keyed tuner. That is not
the actual production public call order (large chunk first). A CPU regression
test executes the real LibEntry.key and reproduces this key aliasing before
checking the isolated-cache correction. Production code and global LibEntry
are unchanged; only the private trial wrapper and benchmark preparation change.

The observed-eight direct-production mean 1.05180 includes this invalid row and
must not be used. Native comparisons and correctness remain useful, subject to
the shared-run policy: upstream12 score/native mean 0.74410; observed8
score/native mean 1.75047 includes pre-existing short-query advantages and is
not all attributable to compact scheduling. The broad ragged gate is not
accepted for production: one genuine regression, one unstable case and one
invalid comparison remain. Keep the candidate experimental and replay the four
compact cases after the cache correction; do not promote from the mean alone.

## Isolated-cache replay and next routing trial

Report `5c226fedda37a454fdf62f39176cb9309c26278f`, code `5e2e7fe`:
194 correctness passes, one second-GPU skip, all four timed cases complete.
Baseline outer caches were cold, device caches independent, and the actual
production public configuration remained unchanged throughout each worker.

| Case | Production us | Compact us | Production/compact | Seven paired ratios |
| --- | ---: | ---: | ---: | --- |
| ragged54 | 83.136 | 73.824 | 1.12614 | 1.12386–1.12928 |
| unequal2 | 732.528 | 729.792 | 1.00375 | 0.96579–1.05567 |
| chunk_boundary | 670.752 | 687.024 | 0.97631 | 0.96907–0.97752 |
| chunk_tail64 | 680.832 | 679.136 | 1.00250 | 0.98922–1.00749 |

The old 1.37578 tail ratio is not reproduced with proper cache isolation.
Its independent production big chunk now selects TILE_R=32 and the private
candidate fallback tail selects TILE_R=8. The 54-request benefit reproduces
across two runs on different cards; no direct cross-card latency comparison
is used. The broad gate is rejected: two two-request cases have no stable
benefit and one regresses. Keep production unchanged.

Pre-coding scope for the next trial: change only the private metadata gate,
retaining the now-validated cache isolation, numerical kernel, six configs,
TopK and complete 22-case benchmark matrix. The candidate requires
`requests * max_query_len >= 4 * chunk_rows` as well as its previous ragged,
multi-request and chunk_rows > max_query_len conditions. This tests a portable
low-utilization policy instead of dispatching to an exact recorded shape.
The factor 4 is a provisional host routing threshold, not an empirically
established universal crossover or a new kernel autotune parameter. Padded
row capacity is the proxy; rounded actual CTA counts are reported separately.
In this matrix only ragged54 uses compact scheduling; the three two-request
cases and all upstream cases use original score bytecode/JIT in the private
fallback cache. Functional scope, unsupported behavior and output contract do
not change. Update CPU routing/boundary tests and the full-entry GPU fixture
to exercise a sparse eight-request batch, then replay all 22 on H800. No new
numerical algorithm or claimed new measured speedup is introduced by this gate.

The replay used GPU2, UUID GPU-2d17872f-4327-2beb-8049-604ecf6a3d66,
the same FlagTree/ptxas/CUDA13.4 identities in all workers, 224 raw timings and
128 telemetry points. Temperature 32–53C, SM clocks 1965–1980MHz, no sampled
co-residents or thermal slowdown. Its shared exploratory acceptance flag stays
false, including the four-case direct-production mean 1.02559.

## Strict-mode validation of the narrowed gate: keep the candidate

Report `c090220d3467b0d1b336717eb978b049c2057086`, code
`0115b4443a1c5547763584f509e2fe34f6baddfc`, completes the narrowed trial:
194 correctness passes, one second-GPU skip, all 22 cases, 82 measured stages,
1148 raw time samples and 656 timing telemetry points. Source identities,
compiler/GPU identities and outer cache isolation match throughout. All timed
stages pass the exclusive-idle policy and have accepted measurements; this
does not mean the overall native-vLLM performance target has been achieved.

| Score-only kernel comparison | Production us | Candidate us | Production/candidate |
| --- | ---: | ---: | ---: |
| ragged54, compact enabled | 83.296 | 73.984 | 1.12587 |
| unequal2, original kernel | 764.288 | 742.752 | 1.02899 |
| chunk_boundary, original kernel | 668.720 | 669.936 | 0.99818 |
| chunk_tail64, original kernel | 680.144 | 680.144 | 1.00000 |

Only ragged54 enters compact scheduling. Its paired ratios span
1.12079–1.12840 (rounded), and its score/native-vLLM ratio is 1.05640.
The other 19 scored cases use identical PTX, CUBIN and configurations to their
production baselines; each has a seven-pair ratio range including 1. The small
control differences, including unequal2's apparent 1.02899 and short96's
0.98127, are not claimed as kernel optimization benefits. Identical binaries
do not establish zero Python-wrapper overhead.

| Matrix subset | Score/native vLLM geometric mean | Score/current production geometric mean |
| --- | ---: | ---: |
| Upstream12 | 0.74144 | 1.00082 |
| Observed8 scored cases | 1.76396 | 1.01786 |

Keep the narrowed private candidate as the measured baseline for subsequent
work. It has not been integrated into the public production file. The old
broad gate remains rejected. No global LibEntry, numerical kernel, TopK or
tuning candidate changes are part of this routing trial. Upstream12 still
uses the original scorer and all 12 score/native ratios are below 1; the
existing short-query advantages in Observed8 must not hide that gap.

Strict-run GPU6 UUID was GPU-83ccf064-1bbb-6765-e356-c2a85323cbee. Same verified
FlagTree/ptxas13.4/CUDA13.4 profile; temperature 32–60C, SM clocks 1950–1980MHz,
only allowed self PIDs in timing samples, no sampled thermal rejection. One
pre-timing 8% utilization observation triggered a 10-second cooldown and then
passed at 0%; the wait is outside timing. Peak allocated memory was
4825.83MiB. Future work should focus on the long-prefill scoring path for
Upstream12 and establish profiling evidence before changing its algorithm or
tile schedule; TopK and the active matrix remain frozen.
