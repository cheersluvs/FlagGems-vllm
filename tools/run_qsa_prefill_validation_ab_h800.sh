#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 FlagOS Contributors
# Invoke with bash; errors affect this process, not the caller's shell options.
set -euo pipefail
trap 'qsa_exit=$?; printf "失败（退出码=%s，行=%s）：%s\n" "$qsa_exit" "$LINENO" "$BASH_COMMAND" >&2; exit "$qsa_exit"' ERR

qsa_repo=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
qsa_python=${PYTHON:-python}
qsa_report=$(mktemp -d "${TMPDIR:-/tmp}/qsa-stage-h800.XXXXXX")
echo "运行目录：$qsa_repo"
echo "报告目录：$qsa_report"
git -C "$qsa_repo" status --short
git -C "$qsa_repo" rev-parse HEAD
unset CUDA_LAUNCH_BLOCKING
case "${QSA_PREFILL_AB_KIND:-validation}" in
  validation) qsa_ab_flag=--validation-ab ;;
  host) qsa_ab_flag=--host-ab ;;
  single-row) qsa_ab_flag=--single-row-ab ;;
  *) echo "QSA_PREFILL_AB_KIND must be validation, host or single-row" >&2; exit 2 ;;
esac

qsa_run_status=0
if "$qsa_python" "$qsa_repo/tools/benchmark_qsa_prefill_stages.py" \
  "$qsa_ab_flag" --host-profile --suite inference-prefill \
  --report-dir "$qsa_report" "$@"
then
  echo "测试完成，开始回传。"
else
  qsa_run_status=$?
  echo "测试退出码：${qsa_run_status}；尝试回传已有报告。"
fi
printf 'benchmark_exit_code=%s\n' "$qsa_run_status" > "$qsa_report/run-status.txt"
if [[ ! -s "$qsa_report/stage-summary.json" ]]; then
  echo "尚未生成汇总报告，请保留终端错误及目录：$qsa_report" >&2
  exit 2
fi

qsa_stamp=$(date -u +%Y%m%d-%H%M%S)
qsa_branch="codex/qsa-h800-results-$qsa_stamp"
qsa_tree=$(mktemp -d "${TMPDIR:-/tmp}/flaggems-qsa-stage-results.XXXXXX")
git -C "$qsa_repo" worktree add -b "$qsa_branch" "$qsa_tree" HEAD
qsa_dest="results/qsa_select_paged_prefill/h800/stages-$qsa_stamp"
mkdir -p "$qsa_tree/$qsa_dest"
shopt -s nullglob
qsa_files=("$qsa_report"/*.json "$qsa_report"/*.txt "$qsa_report"/*.tsv)
cp -- "${qsa_files[@]}" "$qsa_tree/$qsa_dest/"
git -C "$qsa_tree" add -- "$qsa_dest"
git -C "$qsa_tree" commit -m "Report H800 QSA host A/B"
git -C "$qsa_tree" push https://github.com/cheersluvs/FlagGems-vllm.git \
  "HEAD:refs/heads/$qsa_branch"
echo "报告回传完成。"
echo "结果分支：$qsa_branch"
git -C "$qsa_tree" rev-parse HEAD
echo "测试退出码：$qsa_run_status"
exit "$qsa_run_status"
