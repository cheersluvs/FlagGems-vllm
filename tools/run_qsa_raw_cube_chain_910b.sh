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

# Execute with bash; preserve the login shell and existing GPU reports.
set -uo pipefail
qsa_repo=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd) || exit 1
qsa_kind=raw-cube-chain
qsa_corefix=0
if [[ ${1:-} == --compiler-corefix ]]; then
    qsa_corefix=1
    qsa_kind=raw-cube-corefix
    shift
fi
qsa_score_chunks=0
qsa_profile_vectors=0
qsa_probe_k_pack=0
qsa_probe_finish=0
qsa_probe_qw_pack=0
qsa_probe_finish_group4=0
qsa_probe_cube_query_pack=0
qsa_probe_finish_prefix=0
qsa_probe_finish_merge=0
qsa_probe_finish_blocked=0
qsa_ten_score=0
qsa_finish_routing=0
qsa_empty_pair=0
qsa_empty_ten=0
qsa_serial_harness=0
qsa_mixed_harness=0
qsa_raw_mixed=0
qsa_boundaries=0
qsa_block512=0
qsa_critical=0
qsa_topk_prefix=0
qsa_sort_buffers=0
qsa_tle_sort=0
qsa_tle_harness=0
qsa_history_full=0
qsa_history_harness=0
qsa_query_batch=0
qsa_history_direct=0
qsa_single_block=0
qsa_first_block_seed=0
for qsa_argument in "$@"; do
    if [[ "$qsa_argument" == --first-block-seed ]]; then
        qsa_first_block_seed=1
    fi
    if [[ "$qsa_argument" == --single-block ]]; then
        qsa_single_block=1
    fi
    if [[ "$qsa_argument" == --history-direct ]]; then
        qsa_history_direct=1
    fi
    if [[ "$qsa_argument" == --query-batch ]]; then
        qsa_query_batch=1
    fi
    if [[ "$qsa_argument" == --history-prefix-harness ]]; then
        qsa_history_harness=1
    fi
    if [[ "$qsa_argument" == --history-full-merge ]]; then
        qsa_history_full=1
    fi
    if [[ "$qsa_argument" == --tle-sort-harness ]]; then
        qsa_tle_harness=1
    fi
    if [[ "$qsa_argument" == --tle-sort512 ]]; then
        qsa_tle_sort=1
    fi
    if [[ "$qsa_argument" == --sort-buffers ]]; then
        qsa_sort_buffers=1
    fi
    if [[ "$qsa_argument" == --topk-prefix ]]; then
        qsa_topk_prefix=1
    fi
    if [[ "$qsa_argument" == --critical-path ]]; then
        qsa_critical=1
    fi
    if [[ "$qsa_argument" == --cube-block512 ]]; then
        qsa_block512=1
    fi
    if [[ "$qsa_argument" == --mixed-harness ]]; then
        qsa_mixed_harness=1
        qsa_serial_harness=1
        qsa_score_chunks=1
    fi
    if [[ "$qsa_argument" == --boundaries ]]; then
        qsa_boundaries=1
    fi
    if [[ "$qsa_argument" == --raw-mixed-online ]]; then
        qsa_raw_mixed=1
        qsa_score_chunks=1
    fi
    if [[ "$qsa_argument" == --serial-harness ]]; then
        qsa_serial_harness=1
        qsa_score_chunks=1
    fi
    if [[ "$qsa_argument" == --score-chunks ]]; then
        qsa_score_chunks=1
        qsa_kind=raw-score-chunks
    fi
    if [[ "$qsa_argument" == --profile-vectors ]]; then
        qsa_profile_vectors=1
    fi
    if [[ "$qsa_argument" == --probe-k-pack ]]; then
        qsa_probe_k_pack=1
    fi
    if [[ "$qsa_argument" == --probe-finish ]]; then
        qsa_probe_finish=1
    fi
    if [[ "$qsa_argument" == --probe-qw-pack ]]; then
        qsa_probe_qw_pack=1
    fi
    if [[ "$qsa_argument" == --probe-finish-group4 ]]; then
        qsa_probe_finish_group4=1
    fi
    if [[ "$qsa_argument" == --probe-cube-query-pack ]]; then
        qsa_probe_cube_query_pack=1
    fi
    if [[ "$qsa_argument" == --probe-finish-prefix ]]; then
        qsa_probe_finish_prefix=1
    fi
    if [[ "$qsa_argument" == --probe-finish-merge ]]; then
        qsa_probe_finish_merge=1
    fi
    if [[ "$qsa_argument" == --probe-finish-blocked ]]; then
        qsa_probe_finish_blocked=1
    fi
    if [[ "$qsa_argument" == --finish-empty-ten ]]; then
        qsa_empty_ten=1
        qsa_empty_pair=1
        qsa_finish_routing=1
        qsa_ten_score=1
        qsa_score_chunks=1
    fi
    if [[ "$qsa_argument" == --finish-empty-pair ]]; then
        qsa_empty_pair=1
        qsa_finish_routing=1
        qsa_ten_score=1
        qsa_score_chunks=1
    fi
    if [[ "$qsa_argument" == --finish-routing-probe ]]; then
        qsa_finish_routing=1
        qsa_ten_score=1
        qsa_score_chunks=1
    fi
    if [[ "$qsa_argument" == --ten-score-suite ]]; then
        qsa_ten_score=1
        qsa_score_chunks=1
    fi
done
if [[ "$qsa_profile_vectors" == 1 ]]; then
    qsa_kind=raw-vector-profile
fi
if [[ "$qsa_probe_k_pack" == 1 ]]; then
    qsa_kind=raw-k-pack-2d
fi
if [[ "$qsa_probe_finish" == 1 ]]; then
    qsa_kind=raw-finish-page64
fi
if [[ "$qsa_probe_qw_pack" == 1 ]]; then
    qsa_kind=raw-qw-pack-2d
fi
if [[ "$qsa_probe_finish_group4" == 1 ]]; then
    qsa_kind=raw-finish-group4
fi
if [[ "$qsa_probe_cube_query_pack" == 1 ]]; then
    qsa_kind=raw-cube-query-pack
fi
if [[ "$qsa_probe_finish_prefix" == 1 ]]; then
    qsa_kind=raw-finish-prefix
fi
if [[ "$qsa_probe_finish_merge" == 1 ]]; then
    qsa_kind=raw-finish-merge
fi
if [[ "$qsa_probe_finish_blocked" == 1 ]]; then
    qsa_kind=raw-fused-ks-pack
fi
if [[ "$qsa_ten_score" == 1 ]]; then
    qsa_kind=raw-ten-score-suite
fi
if [[ "$qsa_finish_routing" == 1 ]]; then
    qsa_kind=raw-finish-routing
fi
if [[ "$qsa_empty_pair" == 1 ]]; then
    qsa_kind=raw-empty-finish-pair
fi
if [[ "$qsa_empty_ten" == 1 ]]; then
    qsa_kind=raw-empty-finish-ten
fi
if [[ "$qsa_serial_harness" == 1 ]]; then
    qsa_kind=raw-serial-harness
fi
if [[ "$qsa_mixed_harness" == 1 ]]; then
    qsa_kind=raw-mixed-harness
fi
if [[ "$qsa_raw_mixed" == 1 ]]; then
    qsa_kind=raw-mixed-online
fi
if [[ "$qsa_raw_mixed" == 1 && "$qsa_boundaries" == 1 ]]; then
    qsa_kind=raw-mixed-boundaries
fi
if [[ "$qsa_block512" == 1 ]]; then
    if [[ "$qsa_raw_mixed" != 1 || "$qsa_boundaries" != 1 || "$qsa_serial_harness" == 1 ]]; then
        echo '--cube-block512 requires --raw-mixed-online --boundaries without harness flags'
        exit 2
    fi
    qsa_kind=raw-cube-block512
fi
if [[ "$qsa_critical" == 1 ]]; then
    if [[ "$qsa_block512" != 1 ]]; then
        echo '--critical-path requires --cube-block512'
        exit 2
    fi
    qsa_kind=raw-critical-path
fi
if [[ "$qsa_topk_prefix" == 1 ]]; then
    if [[ "$qsa_critical" != 1 ]]; then
        echo '--topk-prefix requires --critical-path'
        exit 2
    fi
    qsa_kind=raw-topk-prefix
fi
if [[ "$qsa_sort_buffers" == 1 ]]; then
    if [[ "$qsa_block512" != 1 || "$qsa_critical" == 1 || "$qsa_topk_prefix" == 1 ]]; then
        echo '--sort-buffers requires --cube-block512 without critical/prefix modes'
        exit 2
    fi
    qsa_kind=raw-sort-buffers
fi
if [[ "$qsa_tle_sort" == 1 ]]; then
    if [[ "$qsa_block512" != 1 || "$qsa_critical" == 1 || "$qsa_topk_prefix" == 1 || "$qsa_sort_buffers" == 1 ]]; then
        echo '--tle-sort512 requires --cube-block512 without other diagnostic variants'
        exit 2
    fi
    qsa_kind=raw-tle-sort512
fi
if [[ "$qsa_tle_harness" == 1 ]]; then
    if [[ "$qsa_serial_harness" != 1 || "$qsa_mixed_harness" != 1 || "$qsa_tle_sort" == 1 || "$qsa_raw_mixed" == 1 ]]; then
        echo '--tle-sort-harness requires --serial-harness --mixed-harness without micro flags'
        exit 2
    fi
    qsa_kind=raw-tle-sort-harness
fi
if [[ "$qsa_history_full" == 1 ]]; then
    if [[ "$qsa_block512" != 1 || "$qsa_critical" == 1 || "$qsa_topk_prefix" == 1 || "$qsa_sort_buffers" == 1 || "$qsa_tle_sort" == 1 || "$qsa_tle_harness" == 1 ]]; then
        echo '--history-full-merge requires --cube-block512 without other diagnostic variants'
        exit 2
    fi
    qsa_kind=raw-history-full-merge
fi
if [[ "$qsa_history_harness" == 1 ]]; then
    if [[ "$qsa_tle_harness" != 1 || "$qsa_history_full" == 1 ]]; then
        echo '--history-prefix-harness requires --tle-sort-harness without micro flags'
        exit 2
    fi
    qsa_kind=raw-history-prefix-harness
fi
if [[ "$qsa_query_batch" == 1 ]]; then
    if [[ "$qsa_history_full" != 1 || "$qsa_history_harness" == 1 ]]; then
        echo '--query-batch requires --history-full-merge without harness flags'
        exit 2
    fi
    qsa_kind=raw-query-batch
fi
if [[ "$qsa_history_direct" == 1 ]]; then
    if [[ "$qsa_history_full" != 1 || "$qsa_query_batch" == 1 || "$qsa_history_harness" == 1 ]]; then
        echo '--history-direct requires --history-full-merge without other variants'
        exit 2
    fi
    qsa_kind=raw-history-direct
fi
if [[ "$qsa_single_block" == 1 ]]; then
    if [[ "$qsa_history_full" != 1 || "$qsa_query_batch" == 1 || "$qsa_history_direct" == 1 ]]; then
        echo '--single-block requires --history-full-merge without other history variants'
        exit 2
    fi
    qsa_kind=raw-single-block
fi
if [[ "$qsa_first_block_seed" == 1 ]]; then
    if [[ "$qsa_history_full" != 1 || "$qsa_query_batch" == 1 || "$qsa_history_direct" == 1 || "$qsa_single_block" == 1 ]]; then
        echo '--first-block-seed requires --history-full-merge without other history variants'
        exit 2
    fi
    qsa_kind=raw-first-block-seed
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
qsa_report=$(mktemp -d /tmp/qsa-raw-cube-chain.XXXXXXXX) || exit 1
qsa_commit=$(git -C "$qsa_repo" rev-parse HEAD) || exit 1
printf '%s\n' "$qsa_commit" > "$qsa_report/code-commit.txt"
printf 'Code: %s\nReport: %s\n' "$qsa_repo" "$qsa_report"
qsa_timeout=${QSA_PROBE_TIMEOUT:-900}
export TRITON_CACHE_DIR="$qsa_report/triton-cache"
export QSA_CAPTURE_COMPILER_OUTPUTS=1
qsa_status=0
if [[ "$qsa_corefix" == 1 ]]; then
    if [[ -z ${qsa_env_root:-} ]]; then
        echo '--compiler-corefix requires QSA_ENV_ROOT' | tee "$qsa_report/preparation-console.txt"
        qsa_status=2
    else
        "$qsa_python" -u "$qsa_repo/tools/test_qsa_core_mode_fix.py" -v \
            2>&1 | tee "$qsa_report/corefix-cpu-tests.txt"
        qsa_status=${PIPESTATUS[0]}
        if [[ "$qsa_status" == 0 ]]; then
            "$qsa_python" -u "$qsa_repo/tools/prepare_qsa_core_mode_fix_910b.py" \
                --source-root "$qsa_env_root/flagtree" \
                --destination "$qsa_report/flagtree-corefix" --report "$qsa_report" \
                2>&1 | tee "$qsa_report/preparation-console.txt"
            qsa_status=${PIPESTATUS[0]}
        fi
        if [[ "$qsa_status" == 0 ]]; then
            export QSA_CORE_MODE_FIX_MANIFEST="$qsa_report/compiler-core-mode-fix.json"
            export PYTHONPATH="$qsa_report/flagtree-corefix:$PYTHONPATH"
            printf 'Experimental compiler copy: %s\n' "$qsa_report/flagtree-corefix"
        fi
    fi
fi
printf 'Import check: %s\n' "$qsa_python"
if [[ "$qsa_status" == 0 && "$qsa_score_chunks" == 1 ]]; then
    "$qsa_python" -u "$qsa_repo/tools/test_qsa_raw_score_chunks.py" -v \
        2>&1 | tee "$qsa_report/score-chunk-cpu-tests.txt"
    qsa_status=${PIPESTATUS[0]}
    if [[ "$qsa_status" == 0 && ( "$qsa_probe_cube_query_pack" == 1 || "$qsa_probe_finish_prefix" == 1 || "$qsa_probe_finish_merge" == 1 || "$qsa_probe_finish_blocked" == 1 ) ]]; then
        "$qsa_python" -u "$qsa_repo/tools/test_qsa_cube_query_pack_probe.py" -v \
            2>&1 | tee "$qsa_report/cube-query-pack-cpu-tests.txt"
        qsa_status=${PIPESTATUS[0]}
    fi
    if [[ "$qsa_status" == 0 && "$qsa_probe_finish_prefix" == 1 ]]; then
        "$qsa_python" -u "$qsa_repo/tools/test_qsa_finish_prefix_probe.py" -v \
            2>&1 | tee "$qsa_report/finish-prefix-cpu-tests.txt"
        qsa_status=${PIPESTATUS[0]}
    fi
    if [[ "$qsa_status" == 0 && "$qsa_probe_finish_merge" == 1 ]]; then
        "$qsa_python" -u "$qsa_repo/tools/test_qsa_finish_merge_probe.py" -v \
            2>&1 | tee "$qsa_report/finish-merge-cpu-tests.txt"
        qsa_status=${PIPESTATUS[0]}
    fi
fi
if [[ "$qsa_status" == 0 && "$qsa_probe_finish_blocked" == 1 ]]; then
    "$qsa_python" -u "$qsa_repo/tools/test_qsa_finish_blocked_probe.py" -v \
        2>&1 | tee "$qsa_report/finish-blocked-cpu-tests.txt"
    qsa_status=${PIPESTATUS[0]}
fi
if [[ "$qsa_status" == 0 && "$qsa_probe_finish_blocked" == 1 ]]; then
    "$qsa_python" -u "$qsa_repo/tools/test_qsa_finish_full_block_probe.py" -v \
        2>&1 | tee "$qsa_report/finish-full-block-cpu-tests.txt"
    qsa_status=${PIPESTATUS[0]}
fi
if [[ "$qsa_status" == 0 && "$qsa_probe_finish_blocked" == 1 ]]; then
    "$qsa_python" -u "$qsa_repo/tools/test_qsa_fused_ks_pack_probe.py" -v \
        2>&1 | tee "$qsa_report/fused-ks-cpu-tests.txt"
    qsa_status=${PIPESTATUS[0]}
fi
if [[ "$qsa_status" == 0 && "$qsa_ten_score" == 1 ]]; then
    "$qsa_python" -u "$qsa_repo/tools/test_qsa_ten_score_suite.py" -v \
        2>&1 | tee "$qsa_report/ten-score-cpu-tests.txt"
    qsa_status=${PIPESTATUS[0]}
fi
if [[ "$qsa_status" == 0 && "$qsa_finish_routing" == 1 ]]; then
    "$qsa_python" -u "$qsa_repo/tools/test_qsa_finish_routing_probe.py" -v \
        2>&1 | tee "$qsa_report/finish-routing-cpu-tests.txt"
    qsa_status=${PIPESTATUS[0]}
fi
if [[ "$qsa_status" == 0 && "$qsa_empty_pair" == 1 ]]; then
    "$qsa_python" -u "$qsa_repo/tools/test_qsa_empty_finish_pair.py" -v \
        2>&1 | tee "$qsa_report/empty-finish-pair-cpu-tests.txt"
    qsa_status=${PIPESTATUS[0]}
fi
if [[ "$qsa_status" == 0 && "$qsa_serial_harness" == 1 ]]; then
    "$qsa_python" -u "$qsa_repo/tools/test_qsa_raw_serial_selection.py" -v \
        2>&1 | tee "$qsa_report/raw-serial-cpu-tests.txt"
    qsa_status=${PIPESTATUS[0]}
fi
if [[ "$qsa_status" == 0 && "$qsa_mixed_harness" == 1 ]]; then
    "$qsa_python" -u "$qsa_repo/tools/test_qsa_raw_mixed_selection.py" -v \
        2>&1 | tee "$qsa_report/raw-mixed-selection-cpu-tests.txt"
    qsa_status=${PIPESTATUS[0]}
fi
if [[ "$qsa_status" == 0 && "$qsa_raw_mixed" == 1 ]]; then
    "$qsa_python" -u "$qsa_repo/tools/test_qsa_raw_mixed_online.py" -v \
        2>&1 | tee "$qsa_report/raw-mixed-online-cpu-tests.txt"
    qsa_status=${PIPESTATUS[0]}
    if [[ "$qsa_status" == 0 ]]; then
        "$qsa_python" -u "$qsa_repo/tools/test_qsa_raw_mixed_abi.py" -v \
            2>&1 | tee "$qsa_report/raw-mixed-abi-cpu-tests.txt"
        qsa_status=${PIPESTATUS[0]}
    fi
    if [[ "$qsa_status" == 0 && "$qsa_boundaries" == 1 ]]; then
        "$qsa_python" -u "$qsa_repo/tools/test_qsa_raw_mixed_boundaries.py" -v \
            2>&1 | tee "$qsa_report/raw-mixed-boundaries-cpu-tests.txt"
        qsa_status=${PIPESTATUS[0]}
    fi
fi
if [[ "$qsa_status" == 0 && "$qsa_block512" == 1 ]]; then
    "$qsa_python" -u "$qsa_repo/tools/test_qsa_raw_cube_block512.py" -v \
        2>&1 | tee "$qsa_report/raw-block512-cpu-tests.txt"
    qsa_status=${PIPESTATUS[0]}
fi
if [[ "$qsa_status" == 0 && "$qsa_critical" == 1 ]]; then
    "$qsa_python" -u "$qsa_repo/tools/test_qsa_raw_critical_path.py" -v \
        2>&1 | tee "$qsa_report/raw-critical-cpu-tests.txt"
    qsa_status=${PIPESTATUS[0]}
fi
if [[ "$qsa_status" == 0 && "$qsa_topk_prefix" == 1 ]]; then
    "$qsa_python" -u "$qsa_repo/tools/test_qsa_raw_topk_prefix.py" -v \
        2>&1 | tee "$qsa_report/raw-topk-prefix-cpu-tests.txt"
    qsa_status=${PIPESTATUS[0]}
fi
if [[ "$qsa_status" == 0 && "$qsa_sort_buffers" == 1 ]]; then
    "$qsa_python" -u "$qsa_repo/tools/test_qsa_raw_sort_buffers.py" -v \
        2>&1 | tee "$qsa_report/raw-sort-buffer-cpu-tests.txt"
    qsa_status=${PIPESTATUS[0]}
fi
if [[ "$qsa_status" == 0 && "$qsa_tle_sort" == 1 ]]; then
    "$qsa_python" -u "$qsa_repo/tools/test_qsa_raw_tle_sort512.py" -v \
        2>&1 | tee "$qsa_report/raw-tle-sort512-cpu-tests.txt"
    qsa_status=${PIPESTATUS[0]}
fi
if [[ "$qsa_status" == 0 && "$qsa_tle_harness" == 1 ]]; then
    "$qsa_python" -u "$qsa_repo/tools/test_qsa_raw_tle_sort_harness.py" -v \
        2>&1 | tee "$qsa_report/raw-tle-sort-harness-cpu-tests.txt"
    qsa_status=${PIPESTATUS[0]}
fi
if [[ "$qsa_status" == 0 && "$qsa_history_full" == 1 ]]; then
    "$qsa_python" -u "$qsa_repo/tools/test_qsa_raw_history_merge.py" -v \
        2>&1 | tee "$qsa_report/raw-history-cpu-tests.txt"
    qsa_status=${PIPESTATUS[0]}
fi
if [[ "$qsa_status" == 0 && "$qsa_query_batch" == 1 ]]; then
    "$qsa_python" -u "$qsa_repo/tools/test_qsa_query_batch_probe.py" -v \
        2>&1 | tee "$qsa_report/query-batch-cpu-tests.txt"
    qsa_status=${PIPESTATUS[0]}
fi
if [[ "$qsa_status" == 0 && "$qsa_history_direct" == 1 ]]; then
    "$qsa_python" -u "$qsa_repo/tools/test_qsa_history_direct_probe.py" -v \
        2>&1 | tee "$qsa_report/history-direct-cpu-tests.txt"
    qsa_status=${PIPESTATUS[0]}
fi
if [[ "$qsa_status" == 0 && "$qsa_single_block" == 1 ]]; then
    "$qsa_python" -u "$qsa_repo/tools/test_qsa_single_block_probe.py" -v \
        2>&1 | tee "$qsa_report/single-block-cpu-tests.txt"
    qsa_status=${PIPESTATUS[0]}
fi
if [[ "$qsa_status" == 0 && "$qsa_first_block_seed" == 1 ]]; then
    "$qsa_python" -u "$qsa_repo/tools/test_qsa_first_block_seed_probe.py" -v \
        2>&1 | tee "$qsa_report/first-block-seed-cpu-tests.txt"
    qsa_status=${PIPESTATUS[0]}
fi
if [[ "$qsa_status" == 0 && "$qsa_history_harness" == 1 ]]; then
    "$qsa_python" -u "$qsa_repo/tools/test_qsa_history_policy.py" -v \
        2>&1 | tee "$qsa_report/history-policy-cpu-tests.txt"
    qsa_status=${PIPESTATUS[0]}
fi
qsa_worker="$qsa_repo/tools/probe_qsa_raw_cube_chain_910b.py"
if [[ "$qsa_serial_harness" == 1 ]]; then
    qsa_worker="$qsa_repo/tools/run_qsa_raw_serial_harness_910b.py"
fi
if [[ "$qsa_raw_mixed" == 1 ]]; then
    qsa_worker="$qsa_repo/tools/probe_qsa_raw_mixed_online_910b.py"
fi
for qsa_argument in "$@"; do
    if [[ "$qsa_argument" == --packing-probe ]]; then
        qsa_worker="$qsa_repo/tools/run_qsa_packing_probe_910b.py"
        qsa_kind=raw-packing-cost-probe
    fi
    if [[ "$qsa_argument" == --short-history || "$qsa_argument" == --short-history-compact ]]; then
        qsa_worker="$qsa_repo/tools/run_qsa_packing_probe_910b.py"
        qsa_kind=raw-short-history
    fi
    if [[ "$qsa_argument" == --paired-history ]]; then
        qsa_worker="$qsa_repo/tools/run_qsa_packing_probe_910b.py"
        qsa_kind=raw-paired-history
    fi
    if [[ "$qsa_argument" == --paired-handoff ]]; then
        qsa_worker="$qsa_repo/tools/run_qsa_packing_probe_910b.py"
        qsa_kind=raw-paired-handoff
    fi
    if [[ "$qsa_argument" == --sort32-barrier-ten ]]; then
        qsa_worker="$qsa_repo/tools/run_qsa_packing_probe_910b.py"
        qsa_kind=raw-sort32-barrier-ten
    fi
    if [[ "$qsa_argument" == --sort-barrier-ten ]]; then
        qsa_worker="$qsa_repo/tools/run_qsa_packing_probe_910b.py"
        qsa_kind=raw-sort-barrier-ten
    fi
    if [[ "$qsa_argument" == --paired-sort32-barrier ]]; then
        qsa_kind=raw-paired-sort32-barrier
    fi
    if [[ "$qsa_argument" == --paired-sort-barrier ]]; then
        qsa_kind=raw-paired-sort-barrier
    fi
    if [[ "$qsa_argument" == --paired-vector-profile ]]; then
        qsa_kind=raw-paired-vector-profile
    fi
    if [[ "$qsa_argument" == --early-release ]]; then
        qsa_worker="$qsa_repo/tools/run_qsa_packing_probe_910b.py"
        qsa_kind=raw-early-release
    fi
    if [[ "$qsa_argument" == --combined-history ]]; then
        qsa_worker="$qsa_repo/tools/run_qsa_packing_probe_910b.py"
        qsa_kind=raw-combined-history
    fi
done
command -v timeout >/dev/null || { echo 'timeout is required'; exit 2; }
if [[ "$qsa_status" == 0 ]]; then
    timeout --signal=TERM --kill-after=20 "$qsa_timeout" \
        "$qsa_python" -u "$qsa_worker" \
        --report "$qsa_report" "$@" 2>&1 | tee "$qsa_report/console.txt"
    qsa_status=${PIPESTATUS[0]}
else
    printf 'Compiler candidate preparation failed; see preparation report.\n' | tee "$qsa_report/console.txt"
fi
printf '%s\n' "$qsa_status" > "$qsa_report/exit-code.txt"
"$qsa_python" - "$qsa_report" "$qsa_status" <<'PY'
import json
import sys
from pathlib import Path
root = Path(sys.argv[1])
path = root / "raw-cube-chain.json"
record = json.loads(path.read_text()) if path.exists() else {}
serial = root / "serial-harness-summary.json"
if serial.exists():
    record = json.loads(serial.read_text())
record["process_exit_code"] = int(sys.argv[2])
if record.get("status") in (None, "running"):
    record["status"] = "process_failed"
    record["pipeline_proven"] = False
path.write_text(json.dumps(record, indent=2))
print("Report:", path, "status:", record["status"], "process_exit:", sys.argv[2], flush=True)
PY
if [[ ${QSA_UPLOAD_REPORT:-1} == 1 ]]; then
    qsa_stamp=$(date -u +%Y%m%d-%H%M%S)
    qsa_branch="codex/qsa-ascend910b-results-$qsa_stamp-$$"
    qsa_tree=$(mktemp -d /tmp/flaggems-qsa-910b-results.XXXXXXXX) || exit 1
    git -C "$qsa_repo" worktree add -b "$qsa_branch" "$qsa_tree" "$qsa_commit" || exit 1
    qsa_dest="results/qsa_select_paged_prefill/ascend910b/$qsa_kind-$qsa_stamp"
    mkdir -p "$qsa_tree/$qsa_dest" || exit 1
    find "$qsa_report" -maxdepth 1 -type f \( -name '*.json' -o -name '*.txt' -o -name '*.json.gz' -o -name '*.bc.gz' -o -name '*.bin.gz' -o -name '*.csv' \) \
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
printf 'Raw Cube chain exit code: %s\n' "$qsa_status"
exit "$qsa_status"
