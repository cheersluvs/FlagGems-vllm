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
qsa_kind=int8-integrated
if [[ ${1:-} == --harness || ${1:-} == --harness-diagnose ]]; then
    qsa_kind="int8-${1#--}"
    shift
    if [[ -n "${QSA_SUMMARIZE_REPORT:-}" ]]; then
        echo 'Harness summaries use tools.summarize_qsa_int8_harness_910b'
        exit 1
    fi
fi
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
qsa_summary_flag=
qsa_console=console.txt
if [[ -n "${QSA_SUMMARIZE_REPORT:-}" ]]; then
    qsa_report=$(realpath -- "$QSA_SUMMARIZE_REPORT") || exit 1
    test -f "$qsa_report/summary.json" || exit 1
    qsa_summary_flag=--summarize-only
    qsa_console=summary-console.txt
    qsa_commit_file=summary-code-commit.txt
    if [[ -f "$qsa_report/exit-code.txt" && ! -f "$qsa_report/original-exit-code.txt" ]]; then
        cp -- "$qsa_report/exit-code.txt" "$qsa_report/original-exit-code.txt" || exit 1
    fi
else
    qsa_report=$(mktemp -d "/tmp/qsa-ascend910b-$qsa_kind.XXXXXXXX") || exit 1
    qsa_commit_file=code-commit.txt
fi
qsa_commit=$(git -C "$qsa_repo" rev-parse HEAD) || exit 1
printf 'Code commit: %s\nReport: %s\n' "$qsa_commit" "$qsa_report"
printf '%s\n' "$qsa_commit" > "$qsa_report/$qsa_commit_file"
qsa_test=benchmark/test_qsa_select_paged_prefill_int8_perf.py::test_qsa_select_paged_prefill_int8_kernel
qsa_harness_diagnose() (
    cd -- "$qsa_repo" || exit 1
    ulimit -c 0 || exit 1
    qsa_diagnostic_case=$1
    shift
    ASCEND_LAUNCH_BLOCKING=1 QSA_INT8_HARNESS_PREFLIGHT_ONLY=1 \
        PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 "$qsa_python" -u -m pytest \
        -q -s --tb=short "$qsa_test[$qsa_diagnostic_case]" --mode kernel "$@"
)
if [[ "$qsa_kind" == int8-harness-diagnose ]]; then
    qsa_harness_diagnose "${QSA_HARNESS_DIAG_CASE:-int8_dominant_long}" "$@" \
        2>&1 | tee "$qsa_report/$qsa_console"
    qsa_status=${PIPESTATUS[0]}
elif [[ "$qsa_kind" == int8-harness ]]; then
    (
        cd -- "$qsa_repo" || exit 1
        ulimit -c 0 || exit 1
        qsa_cases=$("$qsa_python" -c '
import os
from tools.qsa_int8_ten_workloads_910b import cases, compact_layout_cases, compact_width_cases
from tools.summarize_qsa_int8_harness_910b import SHORT_SCORE_CASES, STAGE_CASES
if os.environ.get("QSA_HARNESS_SORTED_TOPK_LAYOUT") == "1":
    names = compact_layout_cases()
elif os.environ.get("QSA_HARNESS_SORTED_TOPK_COMPACT") == "1":
    names = compact_width_cases()
elif os.environ.get("QSA_HARNESS_SORTED_TOPK_SHORT") == "1":
    names = SHORT_SCORE_CASES
elif os.environ.get("QSA_HARNESS_STAGES") == "1":
    names = STAGE_CASES
else:
    names = [case["name"] for case in cases()]
    names.sort(key=lambda name: name != "int8_dominant_long")
print("\n".join(names))
') || exit 1
        qsa_sorted_topk_short_flag=
        if [[ ${QSA_HARNESS_SORTED_TOPK_LAYOUT:-0} == 1 ]]; then
            export QSA_HARNESS_SORTED_TOPK=1
            qsa_sorted_topk_short_flag=--sorted-topk-layout
        elif [[ ${QSA_HARNESS_SORTED_TOPK_COMPACT:-0} == 1 ]]; then
            export QSA_HARNESS_SORTED_TOPK=1
            qsa_sorted_topk_short_flag=--sorted-topk-compact
        elif [[ ${QSA_HARNESS_SORTED_TOPK_SHORT:-0} == 1 ]]; then
            export QSA_HARNESS_SORTED_TOPK=1
            qsa_sorted_topk_short_flag=--sorted-topk-short
        fi
        qsa_sorted_topk_flag=
        if [[ ${QSA_HARNESS_SORTED_TOPK:-0} == 1 ]]; then
            qsa_sorted_topk_flag=--sorted-topk
        fi
        qsa_stage_flag=
        qsa_short_score_flag=
        if [[ ${QSA_HARNESS_STAGES:-0} == 1 ]]; then
            qsa_stage_flag=--stages
            printf 'QSA INT8 stage profile: four diagnostic cases; no acceptance mean\n'
        fi
        if [[ ${QSA_HARNESS_SHORT_SCORE:-0} == 1 ]]; then
            qsa_short_score_flag=--short-score
            printf 'QSA INT8 trial: guarded N64 query16 scoring for widths (1024, 4096]\n'
        fi
        qsa_pytest_status=0
        while IFS= read -r qsa_case; do
            printf 'QSA INT8 harness worker: %s\n' "$qsa_case"
            qsa_case_console="$qsa_report/harness-$qsa_case-console.txt"
            QSA_INT8_HARNESS_PREFLIGHT_ONLY=0 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
                "$qsa_python" -u -m pytest -q -s \
                "$qsa_test[$qsa_case]" --mode kernel "$@" \
                --record json --output "$qsa_report/harness-$qsa_case.json" \
                2>&1 | tee "$qsa_case_console"
            qsa_pytest_status=${PIPESTATUS[0]}
            if [[ "$qsa_pytest_status" != 0 ]]; then
                printf 'QSA INT8 harness worker failed: %s (pytest exit %s)\n' \
                    "$qsa_case" "$qsa_pytest_status"
                if [[ "$qsa_pytest_status" == 134 ]] || \
                    grep -Eiq '507035|vector core exception|VEC instruction parameter is invalid' "$qsa_case_console"; then
                    printf 'QSA INT8 synchronized preflight diagnostic: %s\n' "$qsa_case"
                    qsa_harness_diagnose "$qsa_case" \
                        2>&1 | tee "$qsa_report/harness-$qsa_case-diagnostic.txt"
                    printf 'QSA INT8 preflight diagnostic exit: %s\n' "${PIPESTATUS[0]}"
                fi
                break
            fi
        done <<< "$qsa_cases"
        if [[ ${QSA_HARNESS_REUSE_TOPK_SCRATCH:-0} == 1 && "$qsa_pytest_status" == 0 ]]; then
            if ! grep -q '^QSA_INT8_HARNESS_TOPK_SCRATCH ' \
                "$qsa_report/harness-int8_long100k-console.txt"; then
                echo 'TopK scratch audit missing for long100k'
                qsa_pytest_status=1
            fi
        fi
        "$qsa_python" -m tools.summarize_qsa_int8_harness_910b \
            "$qsa_report/harness.json" "$qsa_report/summary.json" "$qsa_pytest_status" \
            --merge-case-reports ${qsa_stage_flag:+"$qsa_stage_flag"} \
            ${qsa_short_score_flag:+"$qsa_short_score_flag"} \
            ${qsa_sorted_topk_flag:+"$qsa_sorted_topk_flag"} \
            ${qsa_sorted_topk_short_flag:+"$qsa_sorted_topk_short_flag"}
        qsa_summary_status=$?
        if [[ "$qsa_pytest_status" != 0 ]]; then
            exit "$qsa_pytest_status"
        fi
        exit "$qsa_summary_status"
    ) 2>&1 | tee "$qsa_report/$qsa_console"
    qsa_status=${PIPESTATUS[0]}
else
    "$qsa_python" -u "$qsa_repo/tools/benchmark_qsa_int8_kernel_910b.py" \
        --report "$qsa_report" --integrated ${qsa_summary_flag:+"$qsa_summary_flag"} "$@" 2>&1 | tee "$qsa_report/$qsa_console"
    qsa_status=${PIPESTATUS[0]}
fi
printf '%s\n' "$qsa_status" > "$qsa_report/exit-code.txt"
if [[ ${QSA_UPLOAD_REPORT:-1} == 1 ]]; then
    qsa_stamp=$(date -u +%Y%m%d-%H%M%S)
    qsa_branch="codex/qsa-ascend910b-results-$qsa_stamp-$$"
    qsa_tree=$(mktemp -d /tmp/flaggems-qsa-910b-results.XXXXXXXX) || exit 1
    git -C "$qsa_repo" worktree add -b "$qsa_branch" "$qsa_tree" "$qsa_commit" || exit 1
    qsa_dest="results/qsa_select_paged_prefill/ascend910b/$qsa_kind-$qsa_stamp"
    mkdir -p "$qsa_tree/$qsa_dest" || exit 1
    find "$qsa_report" -maxdepth 1 -type f \( -name '*.json' -o -name '*.txt' -o -name '*.json.gz' -o -name '*.csv' \) \
        -exec cp -- {} "$qsa_tree/$qsa_dest/" \;
    git -C "$qsa_tree" add -- "$qsa_dest" || exit 1
    git -C "$qsa_tree" -c user.name="QSA GPU reports" \
        -c user.email="qsa-reports@users.noreply.github.com" \
        commit -m "Report Ascend 910B $qsa_kind comparison" || exit 1
    git -C "$qsa_tree" push https://github.com/cheersluvs/FlagGems-vllm.git \
        "HEAD:refs/heads/$qsa_branch" || {
        printf 'Upload failed; report retained: %s\nResult worktree: %s\n' "$qsa_report" "$qsa_tree"
        printf "Retry: git -C '%s' push https://github.com/cheersluvs/FlagGems-vllm.git 'HEAD:refs/heads/%s'\n" "$qsa_tree" "$qsa_branch"
        exit 1
    }
    printf '报告回传完成。\n结果分支：%s\n' "$qsa_branch"
    git -C "$qsa_tree" rev-parse HEAD
fi
if [[ "$qsa_kind" == int8-harness-diagnose ]]; then
    printf 'INT8 harness synchronized diagnostic exit code: %s\n' "$qsa_status"
elif [[ "$qsa_kind" == int8-harness ]]; then
    printf 'INT8 harness kernel benchmark exit code: %s\n' "$qsa_status"
else
    printf 'INT8 kernel benchmark exit code: %s (2 means prerequisites missing)\n' "$qsa_status"
fi
exit "$qsa_status"
