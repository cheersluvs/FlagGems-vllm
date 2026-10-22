#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 FlagOS Contributors
# Run as a child bash; do not source into the interactive container shell.
set -euo pipefail
trap 'qsa_rc=$?; printf "失败：退出码=%s，行=%s，命令=%s\n" "$qsa_rc" "$LINENO" "$BASH_COMMAND" >&2; exit "$qsa_rc"' ERR
qsa_repo=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
qsa_python=${QSA_STOCK_PYTHON:-${PYTHON:-python}}
qsa_kind=profiling
qsa_report=$(mktemp -d "${TMPDIR:-/tmp}/qsa-profiling-h800.XXXXXX")
echo "运行目录：$qsa_repo"
echo "报告目录：$qsa_report"
echo "Python：$qsa_python"
git -C "$qsa_repo" status --short
git -C "$qsa_repo" rev-parse HEAD
unset PYTHONPATH PYTHONHOME QSA_TRITON_RUNTIME_PATH QSA_NATIVE_REFERENCE_PYTHON QSA_NATIVE_REPORT_DIR QSA_NATIVE_RESULT_PATH CUDA_LAUNCH_BLOCKING
export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 DNN_VENDOR=nvidia VLLM_PLUGINS=""
qsa_status=0
if "$qsa_python" -I "$qsa_repo/tools/with_cuda_compat134.py" \
  --script "$qsa_repo/tools/qsa_prefill_profiling.py" \
  --report-dir "$qsa_report" "$@" \
  2>&1 | tee "$qsa_report/runner.txt"
then
  echo "小规模计时、A/A 和 profiling 完成，开始回传。"
else
  qsa_status=$?
  echo "测试退出码：${qsa_status}；回传已有报告。"
  for qsa_log in preflight.txt; do
    if [[ -s "$qsa_report/$qsa_log" ]]; then
      tail -n 60 "$qsa_report/$qsa_log"
    fi
  done
fi
printf 'benchmark_exit_code=%s\n' "$qsa_status" > "$qsa_report/run-status.txt"
if [[ ! -s "$qsa_report/profiling-summary.json" ]]; then
  printf '{"status":"failed","error":"profiling coordinator did not complete; see runner.txt"}\n' > "$qsa_report/profiling-summary.json"
fi
if [[ ! -s "$qsa_report/stage-summary.json" ]]; then
  printf '{"status":"failed","error":"runner failed before Python summary; see runner.txt"}\n' > "$qsa_report/stage-summary.json"
fi
qsa_stamp=$(date -u +%Y%m%d-%H%M%S)
qsa_branch="codex/qsa-h800-results-$qsa_stamp"
qsa_tree=$(mktemp -d "${TMPDIR:-/tmp}/flaggems-qsa-flagtree134-ab-results.XXXXXX")
git -C "$qsa_repo" worktree add -b "$qsa_branch" "$qsa_tree" HEAD
qsa_dest="results/qsa_select_paged_prefill/h800/$qsa_kind-$qsa_stamp"
mkdir -p "$qsa_tree/$qsa_dest"
shopt -s nullglob
qsa_files=("$qsa_report"/*.json "$qsa_report"/*.txt "$qsa_report"/*.tsv "$qsa_report"/*.csv)
cp -- "${qsa_files[@]}" "$qsa_tree/$qsa_dest/"
git -C "$qsa_tree" add -- "$qsa_dest"
git -C "$qsa_tree" commit -m "Report H800 QSA $qsa_kind stage A/B"
git -C "$qsa_tree" push https://github.com/cheersluvs/FlagGems-vllm.git "HEAD:refs/heads/$qsa_branch"
echo "报告回传完成。"
echo "结果分支：$qsa_branch"
git -C "$qsa_tree" rev-parse HEAD
echo "测试退出码：$qsa_status"
exit "$qsa_status"
