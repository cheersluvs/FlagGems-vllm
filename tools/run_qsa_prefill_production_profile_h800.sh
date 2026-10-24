#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 FlagOS Contributors
# Run via bash, never source into the interactive container shell.
set -euo pipefail
qsa_repo=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
qsa_python=${QSA_STOCK_PYTHON:-${PYTHON:-python}}
qsa_report=$(mktemp -d "${TMPDIR:-/tmp}/qsa-current-profile-h800.XXXXXX")
echo "运行目录：$qsa_repo"
echo "报告目录：$qsa_report"
echo "Python：$qsa_python"
git -C "$qsa_repo" rev-parse HEAD
git -C "$qsa_repo" status --short
unset PYTHONPATH PYTHONHOME QSA_TRITON_RUNTIME_PATH QSA_NATIVE_REFERENCE_PYTHON QSA_NATIVE_REPORT_DIR QSA_NATIVE_RESULT_PATH CUDA_LAUNCH_BLOCKING
export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 DNN_VENDOR=nvidia VLLM_PLUGINS=""
qsa_kind=production-profile
qsa_status=0
if "$qsa_python" -I "$qsa_repo/tools/with_cuda_compat134.py" \
  --script "$qsa_repo/tools/qsa_prefill_production_profile.py" \
  --report-dir "$qsa_report" "$@" 2>&1 | tee "$qsa_report/runner.txt"
then
  echo "Production hardware capture 完成，开始回传。"
else
  qsa_status=$?
  echo "测试退出码：$qsa_status；回传已有报告。"
fi
printf 'probe_exit_code=%s\n' "$qsa_status" > "$qsa_report/run-status.txt"
if "$qsa_python" -I "$qsa_repo/tools/qsa_prefill_production_profile.py" \
  --show-report "$qsa_report" > "$qsa_report/terminal-summary.txt"
then
  echo "数据与结果已保存：$qsa_report/terminal-summary.txt"
else
  echo "离线展示未完成；原始报告保留。"
fi
qsa_stamp=$(date -u +%Y%m%d-%H%M%S)
qsa_branch="codex/qsa-h800-results-$qsa_stamp"
qsa_tree=$(mktemp -d "${TMPDIR:-/tmp}/flaggems-qsa-probe-results.XXXXXX")
git -C "$qsa_repo" worktree add -b "$qsa_branch" "$qsa_tree" HEAD
qsa_dest="results/qsa_select_paged_prefill/h800/$qsa_kind-$qsa_stamp"
mkdir -p "$qsa_tree/$qsa_dest"
shopt -s nullglob
qsa_files=("$qsa_report"/*.json "$qsa_report"/*.txt "$qsa_report"/*.tsv "$qsa_report"/*.csv)
cp -- "${qsa_files[@]}" "$qsa_tree/$qsa_dest/"
git -C "$qsa_tree" add -- "$qsa_dest"
qsa_name=$(git -C "$qsa_repo" config user.name || printf '%s' 'cheersluvs')
qsa_email=$(git -C "$qsa_repo" config user.email || printf '%s' 'yuqingwu51@gmail.com')
git -C "$qsa_tree" -c user.name="$qsa_name" -c user.email="$qsa_email" commit -m "Report H800 QSA current-production current-production hardware counters"
if git -C "$qsa_tree" push https://github.com/cheersluvs/FlagGems-vllm.git "HEAD:refs/heads/$qsa_branch"
then
  echo "报告回传完成。"
else
  echo "回传失败；无需重测。报告目录：$qsa_report"
  echo "重试：git -C '$qsa_tree' push https://github.com/cheersluvs/FlagGems-vllm.git 'HEAD:refs/heads/$qsa_branch'"
  exit 1
fi
echo "结果分支：$qsa_branch"
git -C "$qsa_tree" rev-parse HEAD
echo "测试退出码：$qsa_status"
exit "$qsa_status"
