# QSA score-only benchmarking on a shared H800 machine

The user confirmed that the machine is shared and other workloads are not
known. Report `597f7021c72acc13322bcbe5b5c738575657d099` stopped on a second
unattributed GPU PID, after switching from physical GPU 1 to GPU 2. Repeated
full-suite restarts discard useful progress. This update adds checkpoint
reuse and rejection diagnostics; it does not change scoring, Top-K, tuning,
input cases, the kernel-mode timer or the process-ownership rule.

## Latest evidence

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
The first shared-machine resume retains 13 cases and measures the remaining
25, using the same GPU UUID. Measurements from the earlier GPU 1 report are
not mixed into the GPU 2 checkpoint.

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
  codex/qsa-h800-results-20261016-012924
test "$(git -C "$qsa_repo" rev-parse FETCH_HEAD)" = \
  597f7021c72acc13322bcbe5b5c738575657d099
qsa_checkpoint=$(mktemp /tmp/qsa-score-checkpoint.XXXXXX)
git -C "$qsa_repo" show \
  FETCH_HEAD:results/qsa_select_paged_prefill/h800/stages-20261016-012924/stage-summary.json \
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

Local verification: 74 unittest methods passed. The actual returned checkpoint
was parsed and verified to retain exactly 13 cases, leaving 25 pending. GPU
execution of the new resume/diagnostic path remains pending.
