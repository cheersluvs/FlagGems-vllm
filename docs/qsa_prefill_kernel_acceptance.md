# NVIDIA QSA scoring acceptance: --mode kernel, Top-K excluded

The user's 2026-10-16 instruction supersedes the earlier choice of full-entry
wall time as the NVIDIA optimization criterion. Acceptance uses scoring-stage
benchmark speedup in `--mode kernel`, with no Top-K in either timed callable.
Full-entry wall, event and CUDA Graph measurements are diagnostics only.

| Field | Frozen choice |
| --- | --- |
| Baseline | Current retained production score launcher/kernel, same process and actual FlagTree as the candidate; native vLLM is a separate final alignment/comparison subject to compiler compatibility |
| Candidate | Existing single-query vector prototype; no kernel/config changes in establishing this metric |
| Timer | Actual repository benchmark.base.Benchmark.get_latency with BenchMode.KERNEL, hence triton.testing.do_bench; default warmup 1000 ms and rep 100 ms, return_mode median, default cache policy |
| Scope | Same preallocated logits workspace and original chunk sequence/offsets on both sides; score launches only, no Top-K, fill, public validation or allocation inside the callable |
| Active set | Existing 38 cases. The zero-visible case has no score and is N/A, not an artificial speedup |
| Aggregation | Seven rounds, alternate baseline/candidate order; ratio of latency medians and paired ratios, per case; no averaging unlike workloads |
| Correctness | Existing complete 161-pass suite plus independent FP32 score checks outside timing, unchanged tolerances |
| Guard / transport | Same explicit GPU UUID, idle/thermal/process guards, fresh caches and automatic JSON/TXT/TSV upload |
| Frozen files | Production operator, prototype, Top-K, YAML configurations, reference helpers and benchmark/base.py |

The previous report `0025302fefd2edbdb18edf8d07f6f29cd6dc3dbe` did not time
baseline and candidate scores with this timer. Its candidate-only score timing
and full Graph timings cannot establish this speedup. Do not subtract isolated
Top-K time from full latency. The previous decision not to adopt the candidate
based on full wall latency is superseded. The complete score-only result
below rejects the broad gate on the updated criterion. Production routing
remains unchanged.

`kernel` here means the repository's actual do_bench mode, not cudagraph. CUDA
events can still observe GPU starvation from host launch gaps; this metric
must not be described as a perfect subtraction of all host overhead. Score
chunks are measured as one score-stage invocation, preserving real chunking.

## Implementation and run

Use the existing upload runner with `QSA_PREFILL_AB_KIND=single-row` and append
`--require-idle-gpu --mode kernel`. Its correctness/transport behavior is
unchanged. The explicit UUID and `/opt/flagtree` environment remain required.
This mode calls the actual shared `Benchmark.get_latency` rather than a
separate approximation; the shared base file is not patched on disk. The
worker temporarily supplies the same mode/warmup/rep configuration for each
measurement and restores the in-process config afterward. It records shared
timer source hashes and the actual imported triton.testing source hash.

The full public operator/Top-K may run in untimed setup and correctness gates.
Neither timed function calls them. Both timed closures bind only a score
launcher and identical chunk arguments, with equal host wrapper structure.
Before and after timing, score outputs are independently checked against FP32
references for sampled rows of every chunk, using the existing BF16/FP8 score
tolerances. The original complete correctness suite still runs first.

Acceptance results are in `score-kernel-ab.tsv` and per-case JSON
`score_kernel`: baseline/candidate latency samples in microseconds, speedup
`baseline_median / candidate_median`, and paired round ratios. `score_timer`
records the kernel-mode parameters and source identity. Empty/select-all
cases report `not_applicable_no_score`, with no fabricated speedup. The
38-case set therefore contains 37 measurable score cases and one N/A case.
Optional full-entry host profiles and Graph diagnostics are not collected by
this score-only mode. Historical full-entry wall gains also do not establish
NVIDIA kernel-mode performance gains under the updated criterion.

No operator code/config or GPU arithmetic changed in the measurement update.
The card runs below provide evidence from the score-only benchmark entry.


Local checks at introduction: 62 unittest methods passed, including shared-kernel-branch
parameter/unit checks, score-only call boundaries, alternating ratio arithmetic,
N/A cases, CLI/TSV propagation, environment guards and upload success/failure.
Ten frozen operator/config/reference/shared-benchmark files and six existing
case/timer/stage functions were verified unchanged. Pre-commit passed.


## Complete result

Report `573a3aa6a1c8c9957f6d1ac86de949c126d71c89` completes all 38 cases:
37 timed and one N/A, with exit 0. This final session added 13 and retained
25 on the same GPU/runtime/timer. The broad vector gate is not accepted:
small single-query H/D fixtures improve, while H=8,D=256 FP8 and mixed
32/32/1 tails regress. Production routing remains unchanged. See
[qsa_prefill_kernel_complete_results.md](qsa_prefill_kernel_complete_results.md)
for all case latencies, provenance and the proposed narrower experiment.

## First card run

Report `4a195ce7676d516f1c838a607f2a9c6a30c16d97` passed correctness and
completed 17 scored cases plus one N/A case. It stopped with exit 2 after an
additional GPU PID appeared during the next case. Upload succeeded. The
active set remains incomplete and production routing stays pending. See
[qsa_prefill_kernel_partial_results.md](qsa_prefill_kernel_partial_results.md).


## Shared-machine continuation

A second run on GPU 2, report `597f7021c72acc13322bcbe5b5c738575657d099`,
also passed correctness but stopped on an additional GPU process. The user
confirmed that the host is shared. New checkpoint support retains only complete,
guarded measurements on the same GPU/runtime/timer and retries failed/unstarted
cases. Reused data remain explicitly labelled. See
[qsa_prefill_shared_gpu_resume.md](qsa_prefill_shared_gpu_resume.md).

The first resume, report `6a2bf187fb61bd79e9cd273d315726d7f8aca749`, retained
13 cases and completed 12 more before another Python process appeared during
timing. Upload succeeded. Progress is 25/38 including one N/A case, leaving
13 to measure. Its checkpoint validates for chained reuse on the same GPU.
The single H=1/D=32 BF16 case measured 1.0478x; the BF16/FP8 36-chunk cases
measured 0.9924x/0.9902x. These partial results do not support adopting the
entire vector region. Production routing remains unchanged.
