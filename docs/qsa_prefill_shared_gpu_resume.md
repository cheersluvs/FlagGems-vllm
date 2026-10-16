# QSA score-only benchmarking on a shared H800 machine

The user confirmed that the machine is shared and other workloads are not
known. Report `597f7021c72acc13322bcbe5b5c738575657d099` stopped on a second
unattributed GPU PID, after switching from physical GPU 1 to GPU 2. Repeated
full-suite restarts discard useful progress. This update adds checkpoint
reuse and rejection diagnostics; it does not change scoring, Top-K, tuning,
input cases, the kernel-mode timer or the process-ownership rule.

## Initial checkpoint

- Code: `746394082edb858784adfed964ff066296d2580e` (same executable sources as the original kernel-mode run).
- Report branch: `codex/qsa-h800-results-20261016-012924`.
- Report commit: `597f7021c72acc13322bcbe5b5c738575657d099`.
- Path: `results/qsa_select_paged_prefill/h800/stages-20261016-012924/`.
- Correctness: 161 passed, 1 second-device skip, 33 native-vLLM cases deselected, 327.48 s.
- Complete: 12 scored cases and one N/A zero-visible case.
- Interrupted: hd_5_96_bfloat16 before its first timing round; 24 later cases unstarted.
- GPU UUID: GPU-2d17872f-4327-2beb-8049-604ecf6a3d66 (physical GPU 2).
- Test PID 3580516 and an additional PID 3580992 (610 MiB) both appear in the query. The latter is outside the recorded self PID set.
- At rejection: 32 C, 1980 MHz; no sampled SW/HW thermal slowdown.

The old report did not capture the extra process's command line or parent PID.
It cannot establish whether that process is another user's task or a helper
related to the test. Repository code inspection finds only a multiprocessing
lock in libentry and the normal nvidia-smi vendor query, not a GPU child-process
launcher in this QSA path. External runtime behavior remains unverified.
Unknown GPU processes still invalidate the affected timing, even if later
identified as a helper. No PID is automatically allowed or terminated.

## Checkpoint contract

| Field | Policy |
| --- | --- |
| Enable | --resume-from prior stage-summary.json, comprehensive single-row --mode kernel with --require-idle-gpu |
| Skip completed | Only status=passed with valid seven-round score samples/medians/speedup, passing guard/telemetry, or the known N/A zero-visible case |
| Retry interrupted | Entire case is measured afresh; partial rounds are never retained |
| Frozen identity | Same GPU UUID, Python/Torch/compiler/tool identities and actual do_bench source hash; same score/public/Top-K/reference/config/shared-benchmark sources and case definitions |
| Allowed changes | Benchmark scheduling/reporting and PID diagnostics; production/math/timer files remain frozen |
| Parameters | Same round count, kernel warmup and rep; no cross-mode conversion |
| Provenance | Retained full case data include original checkpoint SHA256/code commit; new TSV column measurement_origin distinguishes retained/current |
| Further interruption | Returned JSON preserves retained cases and all newly completed cases for another resume |
| Correctness | --skip-accuracy explicitly avoids repeating the previous 161-pass full suite. It records that no new full-suite run occurred, while retained_accuracy preserves the earlier passing result. Frozen test/source checks are required; each new case still independently validates scores before/after timing |
| Rejection details | Capture executable/script basename, PPid/NSpid/status and ps/NVML program identity where visible; omit argument values; missing permissions or exited PIDs remain diagnostic failures, never a whitelist |

This preserves observations from one GPU, not a reservation. The GPU must be
idle at preflight and before each new case. If another job appears, progress
is saved and the run exits 2. Original kernel-mode speedup remains the sole
performance criterion; historical wall/Graph observations are diagnostics.

The new summary separates retained_cases from newly run cases and records
score_case_progress for the 38-case active set. The score-kernel-ab.tsv lists
both groups with origin tags. Do not present retained samples as newly measured.
The first shared-machine resume retained 13 cases and attempted the remaining
25, using the same GPU UUID. Measurements from the earlier GPU 1 report are
not mixed into the GPU 2 checkpoint.

## First resume result: 25 of 38 complete

- Code: `a887b324cfa50257a64546f17a804e9200dd57b8`, clean working tree.
- Report branch: `codex/qsa-h800-results-20261016-022710`.
- Report commit: `6a2bf187fb61bd79e9cd273d315726d7f8aca749`.
- Path: `results/qsa_select_paged_prefill/h800/stages-20261016-022710/`.
- Upload succeeded; benchmark exit 2 is an environment rejection.
- The 13 retained records exactly match the previous report, apart from their
  explicit provenance field. Twelve new cases passed, bringing progress to
  24 scored cases plus one N/A zero-visible case; 13 cases remain.
- No new full correctness suite was requested. The checkpoint preserves the
  earlier 161-pass, one-skip result. New completed cases passed independent
  score checks before and after timing; this is not another 161-pass run.
- All 20 captured source hashes match the local tree. Actual runtime, compiler,
  ptxas, GPU UUID and timer identities match the prior checkpoint.
- All 14 idle checks passed. Of 99 new timing telemetry samples, 98 passed and
  one rejected. Sampled temperature was 31–34 C and SM clock was 1980 MHz;
  the rejection was caused by a process, not thermal slowdown.

The interrupted case is `single_3_64_bfloat16`, after round 2. The benchmark
PID was 3756517, parent 3731789. Another Python PID 3757524, parent 3757495,
appeared during timing. Its reported memory rose from 516 MiB in the first
process query to 6536 MiB in the later diagnostic query. Its parent differs
from the benchmark's parent and it is not a direct child of the benchmark.
The report does not establish its owner or full ancestry. No PID is allowed
on that evidence, and the interrupted case has no accepted speedup.

Selected score-only results (same FlagTree, Top-K excluded):

| Case | Origin | Baseline us | Candidate us | Speedup |
| --- | --- | ---: | ---: | ---: |
| core_0 | retained | 8.192 | 7.648 | 1.0711x |
| single_1_32_bfloat16 | new | 7.712 | 7.360 | 1.0478x |
| one_row_bfloat16, 36 single-row chunks | new | 987.008 | 994.608 | 0.9924x |
| one_row_float8_e4m3fn, 36 single-row chunks | new | 968.784 | 978.416 | 0.9902x |

These observations do not support adopting the whole vector dispatch region.
The incomplete H/D sweep must finish before choosing a narrower region.
Keep the raw rounds: the BF16 36-chunk candidate includes a 1208.224 us round,
and the shared-matmul tail controls also contain isolated long rounds. Passing
the sampled process guard is not proof of continuous isolation or absence of
host scheduling noise. No outliers were deleted or measurements fabricated.

The actual returned JSON passes the chained checkpoint validator: it retains
exactly 25 cases and retries the interrupted case plus 12 unstarted cases.
The production operator, prototype, Top-K, timer and tuning remain unchanged.

## Run and automatically return the report

Fetch the returned report into a temporary file without checking out its older
code. Select the checkpoint's exact GPU UUID. The source tree stays on the new
resume-capable code branch. Full correctness evidence is reused explicitly;
remove --skip-accuracy if a new full-suite run is desired.

```bash
if bash <<'QSA_SCORE_RESUME'
set -euo pipefail
source /tmp/qsa-vllm031-h800.PQnVpV/activate_qsa.sh
qsa_repo=$(mktemp -d /tmp/flaggems-qsa-resume.XXXXXX)
git clone --depth 1 --single-branch \
  --branch codex/qsa-select-paged-prefill-h800 \
  https://github.com/cheersluvs/FlagGems-vllm.git "$qsa_repo"
git -C "$qsa_repo" rev-parse HEAD
git -C "$qsa_repo" fetch --depth 1 \
  https://github.com/cheersluvs/FlagGems-vllm.git \
  codex/qsa-h800-results-20261016-022710
test "$(git -C "$qsa_repo" rev-parse FETCH_HEAD)" = \
  6a2bf187fb61bd79e9cd273d315726d7f8aca749
qsa_checkpoint=$(mktemp /tmp/qsa-score-checkpoint.XXXXXX)
git -C "$qsa_repo" show \
  FETCH_HEAD:results/qsa_select_paged_prefill/h800/stages-20261016-022710/stage-summary.json \
  > "$qsa_checkpoint"
qsa_gpu_uuid=$(python -c 'import json,sys; r=json.load(open(sys.argv[1])); print("GPU-"+r["cases"][0]["runtime"]["uuid"].removeprefix("GPU-"))' "$qsa_checkpoint")
CUDA_VISIBLE_DEVICES="$qsa_gpu_uuid" \
QSA_TRITON_RUNTIME_PATH=/opt/flagtree \
QSA_PREFILL_AB_KIND=single-row \
  bash "$qsa_repo/tools/run_qsa_prefill_validation_ab_h800.sh" \
    --require-idle-gpu --mode kernel --resume-from "$qsa_checkpoint" --skip-accuracy
QSA_SCORE_RESUME
then
  echo "补测和报告回传均完成。"
else
  echo "补测或回传未完成；当前会话已保留，请检查上方结果分支及诊断。" >&2
fi
```

Local verification for the resume implementation: 74 unittest methods passed.
The first actual card resume confirmed retention and process diagnostics.
The next command reuses 25 completed cases and attempts only the remaining 13,
including a fresh full measurement of the interrupted case.
