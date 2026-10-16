# H800 single-query vector scoring: complete kernel-mode results

## Decision

The original trial gate (`one scored chunk row, H<=8, D<=256`) is not accepted
for production. Single-query small-H/D fixtures improve, while the H=8,D=256
FP8 case and mixed 32/32/1 tail cases regress. Production scoring and Top-K
remain unchanged. This completes measurement of this trial's 38-case active
set; it does not validate an unmeasured narrower dispatch policy.

## Provenance and verification

- Final report branch: `codex/qsa-h800-results-20261016-025807`.
- Final report commit: `573a3aa6a1c8c9957f6d1ac86de949c126d71c89`.
- Report directory: `results/qsa_select_paged_prefill/h800/stages-20261016-025807/`.
- Code commit: `407e598492882b300750f8e4e9024fa0227d1df9`, clean tree;
  executable sources unchanged from `a887b324cfa50257a64546f17a804e9200dd57b8`.
- Baseline: current production scoring pinned at `454074da0c079cc649d139489ce11fd3141585b7`,
  operator SHA256 `d70faeb6928c2c7a56636ded833f3359368871e353ecf4c66c6dfd35cae93e73`.
  This is not a native-vLLM performance comparison.
- Actual compiler: `/opt/flagtree/triton`, reporting Triton 3.6.0; same compiler,
  libtriton, ptxas 12.8.93 and source hashes for both sides and every retained case.
- GPU: H800 `GPU-2d17872f-4327-2beb-8049-604ecf6a3d66` throughout this checkpoint chain.
- Measurement: actual shared `Benchmark.get_latency`, `--mode kernel`,
  score launches only, Top-K excluded; warmup 1000 ms, rep 100 ms, seven rounds
  with alternating baseline/candidate order. Speedup is baseline latency
  median divided by candidate latency median. Default do_bench cache policy.
- Progress: 38/38 complete = 37 scored cases and one zero-visible N/A case.
  This final run added 13, retaining 25 from the prior checkpoint. Exit 0;
  automatic report upload succeeded.
- Correctness: prior full suite 161 passed, one second-device skip is retained;
  the two resumes explicitly skipped a new full-suite run. Every new scored
  case independently checked scores against FP32 before and after timing.
- Final-run guard: all 14 idle checks and all 104 timing samples passed;
  sampled temperature 30–33 C, SM clock 1980 MHz, no rejected process/thermal
  sample. Across accepted scored cases, all 296 retained/new timing samples pass.
- Local audit: all 20 captured source hashes match; all 38 canonical definitions
  pass the checkpoint validator; runtime/timer fingerprints match across the
  chain; 25 retained records exactly match the preceding report apart from
  explicit provenance. Seven-round medians and ratios were independently
  recomputed by the validator. Failed partial timings from earlier attempts
  are excluded.

The machine is shared. Passing sampled process/thermal checks does not prove
continuous isolation or remove host scheduling noise. Earlier completed
multi-launch cases contain long individual rounds; raw data remain unchanged.
Kernel mode here is the repository do_bench mode, which can observe host launch
starvation. No Graph results or Top-K subtraction enter this decision.

## Single-query sweep

These fixtures have exactly one query, context 8192, and one score launch.
The H=17,D=33 cases use the identical production dot kernel on both sides and
are controls, not vector improvements.

| H | D | BF16 baseline us | BF16 vector us | BF16 speedup | FP8 baseline us | FP8 vector us | FP8 speedup |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 32 | 7.712 | 7.360 | 1.0478x | 7.392 | 7.072 | 1.0452x |
| 3 | 64 | 7.840 | 7.328 | 1.0699x | 7.744 | 7.360 | 1.0522x |
| 4 | 128 | 9.152 | 8.672 | 1.0554x | 8.000 | 7.712 | 1.0373x |
| 5 | 96 | 8.896 | 8.736 | 1.0183x | 8.256 | 8.032 | 1.0279x |
| 8 | 256 | 11.456 | 11.456 | 1.0000x | 9.824 | 10.528 | 0.9331x |

The four smaller tested H/D pairs improve for both dtypes in this fixture.
The (5,96) gain is smaller; it should not be generalized to every H<=5,D<=128
combination. H=8,D=256 BF16 is flat; FP8 is 0.9331x (candidate latency about
7.17% higher). The existing core_0 BF16 H=4,D=128, context 2048 fixture measures
1.0711x and provides a second context length for that pair.

## Chunking matters

| Fixture | BF16 speedup | FP8 speedup | Candidate paths |
| --- | ---: | ---: | --- |
| 36 single-row chunks | 0.9924x | 0.9902x | vector for all 36 |
| 32/32/1 chunks | 0.9865x | 0.9621x | dot / dot / vector |

A one-row chunk is not sufficient evidence that vector dispatch improves the
whole scoring stage. The multi-chunk fixtures differ in context lengths and
request structure as well as launch count, so these results do not isolate a
single cause. H=17 and multi-row dot fallback controls also show small measured
differences despite sharing GPU math; do not count them as algorithmic gains.

The next candidate should first restrict vector dispatch to total query count
one and a measured small-H/D region, retaining dot for multi-query calls,
single-row tails and H=8,D=256. Validate the *actual new gate* with the same
correctness suite, kernel-mode timer and controls; existing broad-gate samples
cannot serve as measurements of that new implementation. Expand context-length,
H/D-boundary and layout coverage before generalizing the region. Use workload
metadata for dispatch; keep Top-K frozen and avoid NVIDIA intrinsics. Each
future vendor needs its own performance validation of the generic algorithm.

## Complete per-case data

Origin below identifies the session in which a case was measured. All timed
pairs were measured together on the same GPU/compiler; no ratio is formed
from latencies in different sessions. `zero_visible` has no score timing.
The uploaded `score-kernel-ab.tsv` and JSON contain full precision, seven
raw rounds, paired ratios, tuned configs, binaries and provenance.

| Case | Measurement report | Baseline us | Candidate us | Speedup |
| --- | --- | ---: | ---: | ---: |
| core_0 | 012924 | 8.192 | 7.648 | 1.0711x |
| core_1 | 012924 | 11.136 | 11.120 | 1.0014x |
| core_2 | 012924 | 69.056 | 69.056 | 1.0000x |
| core_3 | 012924 | 711.648 | 711.456 | 1.0003x |
| core_4 | 012924 | 11.104 | 11.040 | 1.0058x |
| core_5 | 012924 | 14.400 | 14.432 | 0.9978x |
| upstream_bfloat16_s0 | 012924 | 8.960 | 8.928 | 1.0036x |
| upstream_bfloat16_s1792 | 012924 | 9.056 | 8.832 | 1.0254x |
| upstream_float8_e4m3fn_s0 | 012924 | 8.064 | 8.256 | 0.9767x |
| upstream_float8_e4m3fn_s1792 | 012924 | 8.032 | 8.000 | 1.0040x |
| zero_visible | 012924 | N/A | N/A | N/A |
| hd_1_32_bfloat16 | 012924 | 8.352 | 8.288 | 1.0077x |
| hd_3_64_bfloat16 | 012924 | 9.408 | 9.440 | 0.9966x |
| hd_5_96_bfloat16 | 022710 | 11.040 | 11.136 | 0.9914x |
| hd_8_256_bfloat16 | 022710 | 16.800 | 16.768 | 1.0019x |
| one_row_bfloat16 | 022710 | 987.008 | 994.608 | 0.9924x |
| hd_1_32_float8_e4m3fn | 022710 | 8.256 | 8.256 | 1.0000x |
| hd_3_64_float8_e4m3fn | 022710 | 8.832 | 8.608 | 1.0260x |
| hd_5_96_float8_e4m3fn | 022710 | 10.080 | 10.080 | 1.0000x |
| hd_8_256_float8_e4m3fn | 022710 | 14.016 | 13.824 | 1.0139x |
| one_row_float8_e4m3fn | 022710 | 968.784 | 978.416 | 0.9902x |
| mixed_padded | 022710 | 8.448 | 8.544 | 0.9888x |
| tail_chunk_bfloat16 | 022710 | 31.392 | 31.360 | 1.0010x |
| tail_chunk_float8_e4m3fn | 022710 | 38.592 | 38.144 | 1.0117x |
| single_1_32_bfloat16 | 022710 | 7.712 | 7.360 | 1.0478x |
| single_3_64_bfloat16 | 025807 | 7.840 | 7.328 | 1.0699x |
| single_4_128_bfloat16 | 025807 | 9.152 | 8.672 | 1.0554x |
| single_5_96_bfloat16 | 025807 | 8.896 | 8.736 | 1.0183x |
| single_8_256_bfloat16 | 025807 | 11.456 | 11.456 | 1.0000x |
| single_17_33_bfloat16 | 025807 | 16.736 | 16.736 | 1.0000x |
| tail_one_bfloat16 | 025807 | 32.672 | 33.120 | 0.9865x |
| single_1_32_float8_e4m3fn | 025807 | 7.392 | 7.072 | 1.0452x |
| single_3_64_float8_e4m3fn | 025807 | 7.744 | 7.360 | 1.0522x |
| single_4_128_float8_e4m3fn | 025807 | 8.000 | 7.712 | 1.0373x |
| single_5_96_float8_e4m3fn | 025807 | 8.256 | 8.032 | 1.0279x |
| single_8_256_float8_e4m3fn | 025807 | 9.824 | 10.528 | 0.9331x |
| single_17_33_float8_e4m3fn | 025807 | 12.928 | 12.704 | 1.0176x |
| tail_one_float8_e4m3fn | 025807 | 31.712 | 32.960 | 0.9621x |
