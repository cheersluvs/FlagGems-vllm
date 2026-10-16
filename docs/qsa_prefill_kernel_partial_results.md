# QSA score-only kernel benchmark: interrupted H800 run

Status: incomplete because the GPU process guard rejected an additional PID
after the first round of `hd_5_96_float8_e4m3fn`. Correctness passed. Preserve
the earlier guarded measurements as partial observations; do not adopt a
production route from this incomplete active set. No kernel, configuration,
timing method, validation or guard rule is changed in this report update.

## Provenance and validation

- Report branch: `codex/qsa-h800-results-20261016-010539`.
- Report commit: `4a195ce7676d516f1c838a607f2a9c6a30c16d97`.
- Code commit: `4a16f81fb4a0576e0fb58b4745b24be3e826ecc6`.
- Report directory: `results/qsa_select_paged_prefill/h800/stages-20261016-010539/`.
- Benchmark exit: 2; report upload succeeded.
- Correctness: 161 passed, 1 second-SM90-device skip, 33 native-vLLM cases deselected, 325.22 seconds.
- Completed: 17 measured score cases and 1 N/A zero-visible case.
- Interrupted: 1 case, with no accepted partial timing/speedup; 19 subsequent cases were not started.
- Audited 47 files/30 unique Git blob hashes, all 19 source snapshots against the code commit, all 238 accepted latency samples and their statistics/paired ratios.

Both scoring callables exclude Top-K and use the actual shared
`benchmark.base.Benchmark.get_latency` in `BenchMode.KERNEL`, with
triton.testing.do_bench defaults, warmup 1000 ms, rep 100 ms, seven alternating
rounds, and milliseconds converted to microseconds. Timer source hashes
match across the measured cases. The source of do_bench is
`/opt/flagtree/triton/testing.py`, SHA256
`54b1a09e41f716151e46143f94c4a816090f2f4dea72752bf5f8071293540e7f`.
The shared benchmark base file SHA256 is
`936f41e070e3baab8b5f445ea79c4695c65b164eae7410046d5e8571d5d507fb`.

Both sides use the same actual FlagTree runtime (Triton 3.6.0), Torch
2.13.0+cu129, ptxas 12.8.93 and H800 UUID
`GPU-497db188-9a65-ed86-3516-2de784681f95` (physical GPU 1). Compiler/tool
hashes are consistent across every started case. This remains a production
score self-baseline comparison, not a native-vLLM speedup.

## Why the run stopped

All 20 idle checks passed, including the check before the interrupted case.
Before its timed measurement, only the identified benchmark PID 3516038
appears in the selected GPU process query. After round 1, the query returns:

| PID | Used GPU memory | Identity in report |
| --- | ---: | --- |
| 3516038 | 1214 MiB | Identified benchmark process |
| 3516814 | 522 MiB | Additional PID outside identified self PID set |

The guard saves this query, marks the case rejected, and stops later cases.
No score_kernel speedup is recorded for the interrupted case. The additional
process command line, parent PID and ownership are not captured; its origin
cannot be established from this report. Do not treat it as an allowed child
or stop it based only on its appearance.

There were 138 timing telemetry points: 137 passed and 1 rejected. All
sampled SM clocks are 1980 MHz, temperatures 30–48 C, and SW/HW thermal
slowdown is inactive. The immediate reason for exit 2 is the extra process,
not a thermal rejection, a numerical mismatch or a compiler failure.

## Partial score-only observations

These earlier cases passed the sampled environment checks. Speedup is
production baseline median / vector-candidate median in the exact kernel
benchmark mode. The complete active set has not passed, and most eligible
single-query H/D cases have not run. Keep the candidate pending.

| Case | Vector used | Production us | Candidate us | Speedup | Paired ratio median |
| --- | --- | ---: | ---: | ---: | ---: |
| core_0 | yes | 8.160 | 7.680 | 1.06250 | 1.05372 |
| core_1 | no | 11.136 | 11.104 | 1.00288 | 1.00288 |
| core_2 | no | 68.928 | 69.152 | 0.99676 | 0.99722 |
| core_3 | no | 712.208 | 709.088 | 1.00440 | 1.00112 |
| core_4 | no | 11.168 | 11.008 | 1.01453 | 1.01453 |
| core_5 | no | 14.592 | 14.816 | 0.98488 | 0.98491 |
| upstream_bfloat16_s0 | no | 8.992 | 9.184 | 0.97909 | 0.97909 |
| upstream_bfloat16_s1792 | no | 9.184 | 9.184 | 1.00000 | 0.99653 |
| upstream_float8_e4m3fn_s0 | no | 8.352 | 8.128 | 1.02756 | 1.02756 |
| upstream_float8_e4m3fn_s1792 | no | 8.160 | 8.384 | 0.97328 | 0.97328 |
| hd_1_32_bfloat16 | no | 8.416 | 8.512 | 0.98872 | 0.98872 |
| hd_3_64_bfloat16 | no | 9.664 | 9.472 | 1.02027 | 1.02027 |
| hd_5_96_bfloat16 | no | 11.136 | 11.104 | 1.00288 | 1.00288 |
| hd_8_256_bfloat16 | no | 16.768 | 16.704 | 1.00383 | 1.00383 |
| one_row_bfloat16 | yes | 977.248 | 985.760 | 0.99137 | 0.99001 |
| hd_1_32_float8_e4m3fn | no | 8.224 | 8.224 | 1.00000 | 1.00000 |
| hd_3_64_float8_e4m3fn | no | 8.928 | 8.928 | 1.00000 | 1.00000 |

Only core_0 and one_row_bfloat16 among the measured cases use the vector
kernel; their speedups are respectively 1.06250x and 0.99137x. The other
measured cases share the dot kernel and are controls, not vector-kernel wins.
Do not average these partial cases into an acceptance score.

## Repeat

No software repair or PID-guard relaxation is supported by the evidence.
Repeat the unchanged full 38-case set on a currently available GPU, ensuring
both sides retune and run on that same selected device. The last captured
snapshot shows GPU 2 at 28 C, 1 MiB and 0% utilization; this is historical
evidence only. Resolve its UUID and let the existing guard check live state.
If selecting a different GPU, rerun both sides for the full active set rather
than treating previous absolute latencies as its baseline.

The prior code commit can be run unchanged. The report-recording commit
contains only documentation; all operator, timer and guard sources match
the prior code commit exactly. Use the same --require-idle-gpu --mode kernel
runner, with automatic report upload, and retain the guarded failure status
if a competing or unattributed GPU process appears again.
