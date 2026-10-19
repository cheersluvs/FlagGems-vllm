#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 FlagOS Contributors
# Run as a child bash so a failure preserves the interactive container session.
set -euo pipefail
trap 'qsa_rc=$?; printf "失败：退出码=%s，行=%s，命令=%s\n" "$qsa_rc" "$LINENO" "$BASH_COMMAND" >&2; exit "$qsa_rc"' ERR
qsa_repo=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
qsa_python=${QSA_STOCK_PYTHON:-${PYTHON:-python}}
qsa_report=$(mktemp -d "${TMPDIR:-/tmp}/qsa-ptxas-h800.XXXXXX")
echo "运行目录：$qsa_repo"
echo "报告目录：$qsa_report"
git -C "$qsa_repo" status --short
git -C "$qsa_repo" rev-parse HEAD
unset PYTHONPATH PYTHONHOME
qsa_entry=validate_qsa_ptxas_h800.py
if [[ "${QSA_PTXAS_LOADER_ONLY:-0}" == 1 ]]; then
  qsa_entry=diagnose_qsa_cubin_loader_h800.py
fi
qsa_status=0
if "$qsa_python" -I "$qsa_repo/tools/$qsa_entry" \
  --report-dir "$qsa_report" "$@" 2>&1 | tee "$qsa_report/runner.txt"
then
  echo "新版 ptxas 配置验证完成，开始回传。"
else
  qsa_status=$?
  echo "验证退出码：${qsa_status}；回传已有报告。"
fi
printf 'validation_exit_code=%s\n' "$qsa_status" > "$qsa_report/run-status.txt"
if [[ ! -s "$qsa_report/ptxas-summary.json" ]]; then
  printf '{"status":"failed","error":"see runner.txt"}\n' > "$qsa_report/ptxas-summary.json"
fi
qsa_stamp=$(date -u +%Y%m%d-%H%M%S)
qsa_branch="codex/qsa-h800-results-$qsa_stamp"
qsa_tree=$(mktemp -d "${TMPDIR:-/tmp}/flaggems-qsa-ptxas-results.XXXXXX")
git -C "$qsa_repo" worktree add -b "$qsa_branch" "$qsa_tree" HEAD
qsa_dest="results/qsa_select_paged_prefill/h800/ptxas-$qsa_stamp"
mkdir -p "$qsa_tree/$qsa_dest"
# Transfer nested diagnostics and compressed compiler bundles, not CUDA caches,
# large tensor inputs, raw binaries or the independently downloaded toolchain.
"$qsa_python" -I - "$qsa_report" "$qsa_tree/$qsa_dest" <<'PY'
import shutil, sys
from pathlib import Path
source, target = map(Path, sys.argv[1:])
for path in source.rglob('*'):
    relative = path.relative_to(source)
    if any(p in {'caches', 'dumps'} for p in relative.parts):
        continue
    if path.is_file() and path.suffix in {'.txt', '.json'}:
        destination = target / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, destination)
PY
git -C "$qsa_tree" add -- "$qsa_dest"
git -C "$qsa_tree" commit -m "Report H800 QSA identical PTX with ptxas 13.4.59"
git -C "$qsa_tree" push https://github.com/cheersluvs/FlagGems-vllm.git "HEAD:refs/heads/$qsa_branch"
echo "报告回传完成。"
echo "结果分支：$qsa_branch"
git -C "$qsa_tree" rev-parse HEAD
echo "验证退出码：$qsa_status"
exit "$qsa_status"
