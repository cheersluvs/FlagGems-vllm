# QSA prefill H800: host overlap validation A/B result

Decision: **keep the output byte-range validation optimization** from
`7739abad071f0b64c6d3861a0e593798b56e7ec2`. Short workloads show repeatable
median reductions in host-inclusive latency, corroborated by the isolated
validation microbench. No correctness failure or clear material performance
regression was observed. This accepts one host optimization, not completion
of all QSA performance work or a speedup over native vLLM.

Raw report: [summary](https://github.com/cheersluvs/FlagGems-vllm/blob/27f984c2176b285bd44354ffb9846ec9566efdfe/results/qsa_select_paged_prefill/h800/stages-20261015-040800/stage-summary.json),
[full-entry A/B TSV](https://github.com/cheersluvs/FlagGems-vllm/blob/27f984c2176b285bd44354ffb9846ec9566efdfe/results/qsa_select_paged_prefill/h800/stages-20261015-040800/validation-ab.tsv),
[host probes](https://github.com/cheersluvs/FlagGems-vllm/blob/27f984c2176b285bd44354ffb9846ec9566efdfe/results/qsa_select_paged_prefill/h800/stages-20261015-040800/host-summary.tsv),
[correctness log](https://github.com/cheersluvs/FlagGems-vllm/blob/27f984c2176b285bd44354ffb9846ec9566efdfe/results/qsa_select_paged_prefill/h800/stages-20261015-040800/accuracy-full.txt).
Result branch: `codex/qsa-h800-results-20261015-040800`;
result commit: `27f984c2176b285bd44354ffb9846ec9566efdfe`.

## Verification and comparison scope

- All 74 downloaded report files passed Git blob SHA-1 and size verification.
  All 10 source snapshots passed SHA-256 verification and exactly matched the
  tested code commit; the producer worktree was clean.
- The independent correctness gate passed **125 tests**, including all 21 new
  real-storage alias/adjacency/padding-hole cases. One test requiring a second
  visible SM90 GPU skipped; 33 native-vLLM tests were deliberately deselected.
  This run does not establish new native-vLLM functional alignment.
- All 22 benchmark cases passed. Aggregate and individual JSON matched, and
  statistics were recomputed from all 2,247 formal stage/metric samples.
- Both entries used the same public bytecode and shared score and Top-K
  callables. Only the validator differed: frozen helpers from `416083e2` vs
  the candidate. The baseline is validation-only, not an independently
  recompiled snapshot of the entire old operator.
- All case workers reported the same H800 UUID
  `198a9496-3d7a-f01d-ee60-190fe0bc5b51`, Torch `2.13.0+cu129`, actual imported
  Triton `3.6.0` under `/opt/flagtree` (FlagTree package `0.6.1`), identical
  compiler/library hashes and ptxas `12.8.93`; TF32 was disabled. Installed
  distribution metadata also lists Triton `3.7.1`, which was not the imported
  compiler in this run.
- Schema 2: seven rounds, 104 calls per stage and mode per round; Graph uses
  13 replays of eight captured calls. Stage and mode order rotate each round.
  Warmup/JIT and cProfile are outside formal measurements. These are warm,
  repeated inputs; no application-level serving throughput is measured.

## Full operator latency

Wall time includes host dispatch and synchronization amortized over the batch.
Values are medians in microseconds per call. Reduction is `1 - new / old`;
speedup is `old / new`. Paired ratio is the median of seven same-round ratios;
stages are separate batches within a round, not simultaneous measurements.
No aggregate speedup is used to hide workload differences.

| Case | Old wall us | New wall us | Latency reduction | Speedup | Paired ratio | Interpretation |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| core_0 | 162.690 | 151.978 | 6.58% | 1.0705 | 1.0715 | Host benefit |
| core_1 | 120.600 | 111.252 | 7.75% | 1.0840 | 1.0831 | Host benefit |
| core_2 | 129.034 | 128.757 | 0.21% | 1.0021 | 0.9927 | No resolved benefit; paired ratio below 1 |
| core_3 | 1543.161 | 1484.496 | 3.80% | 1.0395 | 1.0129 | Drift; do not adopt speedup |
| core_4 | 120.510 | 111.225 | 7.70% | 1.0835 | 1.0775 | Host benefit |
| core_5 | 120.398 | 111.535 | 7.36% | 1.0795 | 1.0827 | Host benefit |
| upstream_bfloat16_s0 | 119.514 | 110.039 | 7.93% | 1.0861 | 1.0854 | Host benefit |
| upstream_bfloat16_s1792 | 121.584 | 112.463 | 7.50% | 1.0811 | 1.0801 | Host benefit |
| upstream_float8_e4m3fn_s0 | 126.252 | 115.998 | 8.12% | 1.0884 | 1.0929 | Host benefit |
| upstream_float8_e4m3fn_s1792 | 120.655 | 110.336 | 8.55% | 1.0935 | 1.0895 | Host benefit |
| zero_visible | 41.844 | 31.087 | 25.71% | 1.3460 | 1.3460 | Host benefit; noisy wall/Graph samples |
| hd_1_32_bfloat16 | 120.620 | 110.796 | 8.14% | 1.0887 | 1.0903 | Host benefit |
| hd_3_64_bfloat16 | 122.823 | 113.367 | 7.70% | 1.0834 | 1.0885 | Host benefit |
| hd_5_96_bfloat16 | 123.362 | 112.565 | 8.75% | 1.0959 | 1.0959 | Host benefit |
| hd_8_256_bfloat16 | 124.799 | 115.538 | 7.42% | 1.0802 | 1.0781 | Host benefit |
| one_row_bfloat16 | 3775.602 | 3766.826 | 0.23% | 1.0023 | 1.0035 | Small difference; benefit unresolved |
| hd_1_32_float8_e4m3fn | 123.959 | 113.544 | 8.40% | 1.0917 | 1.0917 | Host benefit |
| hd_3_64_float8_e4m3fn | 145.335 | 131.800 | 9.31% | 1.1027 | 1.0899 | Host benefit |
| hd_5_96_float8_e4m3fn | 120.654 | 111.227 | 7.81% | 1.0848 | 1.0854 | Host benefit |
| hd_8_256_float8_e4m3fn | 124.414 | 114.877 | 7.67% | 1.0830 | 1.0818 | Host benefit |
| one_row_float8_e4m3fn | 4002.312 | 3998.307 | 0.10% | 1.0010 | 1.0010 | Small difference; benefit unresolved |
| mixed_padded | 193.762 | 181.220 | 6.47% | 1.0692 | 1.0660 | Host benefit |

The 17 short scored cases show median latency reductions of **6.47–9.31%**
(1.069–1.103x), with same-round median ratios favoring the candidate in every
one of those cases. Zero-visible selection decreases **25.71%** (1.346x).
Wall outliers remain: core_1 candidate samples span 109.800–145.064 us and
baseline samples 120.232–146.400 us. The table describes median behavior;
it is not a tail-latency guarantee or a confidence interval.

Core_2 is device-work limited in this test: wall is 129.034 vs 128.757 us,
while the paired median ratio is 0.9927. This is not evidence of a speedup;
the roughly 0.7% paired slowdown also is not resolved from the several-percent
sample variation. One-row chunks show only 0.23% BF16 and 0.10% FP8 median
reductions. Their differences are small relative to variation; no reliable
end-to-end benefit is claimed for these cases.

Core_3 baseline wall samples span 1311.213–1839.613 us and candidate samples
1363.298–1786.675 us. The ratio of medians is 1.0395 but paired median is
1.0129; even unchanged Graph work differs by 4.45% in median. Temperature
boundary snapshots rise from 70 to 86 Celsius. SM clocks at those boundaries
remain 1590 MHz with no active slowdown reason. This does not establish the
in-kernel cause. Its apparent 3.80% wall reduction is excluded from the
accepted benefit, and stable long-case timing remains unresolved.

## Mechanism and device-work check

For core_1, unprofiled validation falls from **20.842 to 13.157 us** (36.87%
less). Full-entry wall falls by 9.348 us. These measurements corroborate the
same mechanism but are not additive component accounting: probes and full
entries run in separate batches, with different interpreter/cache conditions.
The mixed padded case's validation falls from 25.481 to 15.886 us.

cProfile confirms five old `_overlaps` calls and ten span sums per validation
were replaced by one `_output_overlaps_inputs` call. Its instrumented durations
are not production latency. Production still rejects conservative byte-hull
overlap, including a strided padding hole, and allows touching/disjoint slices.

Core_1 Graph replay is **19.114 vs 19.137 us**, consistent with unchanged
scoring and Top-K work. Outside core_2, core_3 and zero-visible, full Graph
median changes are within 0.37%. Zero-visible is 1.461 vs 1.488 us (about
28 ns, 1.9%) with large sample scatter; core_2 is 122.279 vs 124.326 us with
several-percent scatter. These differences do not support a GPU kernel gain
or a diagnosed device regression. CPU validation occurs during capture and
is absent from each replay; this optimization primarily benefits eager calls.

## Retained scope and remaining work

The retained production change uses contiguous output bounds once, contiguous
input numel bounds when possible, and the original strided byte hull otherwise.
It adds no global cache, tensor compute/copy/cast, launch, workspace ownership
change, or vendor intrinsic. Kernels, public dispatch, Top-K, autotune keys
and the six scoring candidates remain unchanged. Existing exports and
registration remain in place. The optimization is portable host logic, but
correctness and performance on other vendors remain untested; the operator's
current supported device gate is still NVIDIA SM90 and inference-only.

This trial closes with **keep**, without expanding its hypothesis into a
kernel or shared-runtime edit. Host launch/dispatch cost remains a candidate
for a separate measured trial; long-shape timing stability is a separate
measurement issue. The native-vLLM FlagTree compiler issue remains unresolved,
and none of these results are a native-vLLM speed comparison.
