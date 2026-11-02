#!/usr/bin/env bash
# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Execute this file with bash; never source it into the container login shell.
set -uo pipefail
qsa_repo=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd) || exit 1
# Re-activate after shell startup files have run; lock the environment at execution.
if [[ -n "${QSA_ENV_ROOT:-}" ]]; then
    qsa_env_root=$(realpath -- "$QSA_ENV_ROOT") || exit 1
    test -f "$qsa_env_root/activate.sh" || exit 1
    source "$qsa_env_root/activate.sh" || exit 1
    unset BASH_ENV ENV
    if [[ -n "${QSA_OPP_PATH:-}" ]]; then
        qsa_opp_path=$(realpath -- "$QSA_OPP_PATH") || exit 1
        test -d "$qsa_opp_path" || exit 1
        if [[ "$qsa_opp_path" == *.* ]]; then
            echo 'QSA_OPP_PATH must resolve to a directory without dots for the CANN workaround'
            exit 1
        fi
        export ASCEND_OPP_PATH="$qsa_opp_path"
        printf 'Pinned OPP directory: %s\n' "$ASCEND_OPP_PATH"
    fi
    export PYTHONPATH="$qsa_repo/src:$qsa_env_root/python-deps:$qsa_env_root/flagtree${PYTHONPATH:+:$PYTHONPATH}"
    printf 'Pinned runtime environment: %s\n' "$qsa_env_root"
fi
qsa_python=${QSA_PYTHON:-python3}
qsa_report=$(mktemp -d /tmp/qsa-ascend910b-quant-v1.XXXXXXXX) || exit 1
qsa_commit=$(git -C "$qsa_repo" rev-parse HEAD) || exit 1
printf 'Code commit: %s\nReport: %s\n' "$qsa_commit" "$qsa_report"
printf '%s\n' "$qsa_commit" > "$qsa_report/code-commit.txt"
"$qsa_python" -u "$qsa_repo/tools/probe_qsa_quant_v1_910b.py" \
    --report "$qsa_report" "$@" 2>&1 | tee "$qsa_report/console.txt"
qsa_status=${PIPESTATUS[0]}
printf '%s\n' "$qsa_status" > "$qsa_report/exit-code.txt"
if [[ ${QSA_UPLOAD_REPORT:-1} == 1 ]]; then
    qsa_stamp=$(date -u +%Y%m%d-%H%M%S)
    qsa_branch="codex/qsa-ascend910b-results-$qsa_stamp-$$"
    qsa_tree=$(mktemp -d /tmp/flaggems-qsa-910b-results.XXXXXXXX) || exit 1
    git -C "$qsa_repo" worktree add -b "$qsa_branch" "$qsa_tree" "$qsa_commit" || exit 1
    qsa_dest="results/qsa_select_paged_prefill/ascend910b/quant-v1-$qsa_stamp"
    mkdir -p "$qsa_tree/$qsa_dest" || exit 1
    find "$qsa_report" -maxdepth 1 -type f \( -name '*.json' -o -name '*.txt' -o -name '*.json.gz' \) \
        -exec cp -- {} "$qsa_tree/$qsa_dest/" \;
    git -C "$qsa_tree" add -- "$qsa_dest" || exit 1
    git -C "$qsa_tree" -c user.name="QSA GPU reports" \
        -c user.email="qsa-reports@users.noreply.github.com" \
        commit -m "Report Ascend 910B INT8 QuantLightningIndexer V1 probe" || exit 1
    git -C "$qsa_tree" push https://github.com/cheersluvs/FlagGems-vllm.git \
        "HEAD:refs/heads/$qsa_branch" || {
        printf 'Upload failed; report retained: %s\nResult worktree: %s\n' "$qsa_report" "$qsa_tree"
        printf "Retry: git -C '%s' push https://github.com/cheersluvs/FlagGems-vllm.git 'HEAD:refs/heads/%s'\n" "$qsa_tree" "$qsa_branch"
        exit 1
    }
    printf '报告回传完成。\n结果分支：%s\n' "$qsa_branch"
    git -C "$qsa_tree" rev-parse HEAD
fi
printf 'Quant V1 probe exit code: %s (2 means prerequisites missing)\n' "$qsa_status"
exit "$qsa_status"
