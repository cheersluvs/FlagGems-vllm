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
| Equal-length, one request, or chunk_rows <= max_query_len | Actual original score launcher, no compact lookup overhead |
| Ragged envelope with larger chunks | Metadata gate T < requests * max_query_len; compact tile slots use floor(clipped request start/TILE_R)+request, binary-search device starts |
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

CPU suite: 54 tests, 53 passed and one optional Torch-dependent test skipped
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
