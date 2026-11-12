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

set -uo pipefail
qsa_repo=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd) || exit 1
qsa_container=${QSA_CONTAINER:-zhouxt-metax-qsa-py312}
qsa_report=$(mktemp -d /tmp/qsa-metax-native.XXXXXXXX) || exit 1
qsa_inside=$(docker exec "$qsa_container" mktemp -d /tmp/qsa-metax-native.XXXXXXXX) || exit 1
printf 'Container: %s\nHost report: %s\nContainer report: %s/report\n' \
    "$qsa_container" "$qsa_report" "$qsa_inside"
git -C "$qsa_repo" rev-parse HEAD > "$qsa_report/code-commit.txt" || exit 1
docker inspect --format 'ImageID={{.Image}} Image={{.Config.Image}}' \
    "$qsa_container" > "$qsa_report/container.txt" || exit 1
docker cp "$qsa_repo/tools/probe_qsa_metax_native.py" \
    "$qsa_container:$qsa_inside/probe.py" || exit 1
docker exec -w /tmp "$qsa_container" env VLLM_PLUGINS=metax \
    /opt/conda/bin/python -u "$qsa_inside/probe.py" \
    --report "$qsa_inside/report" --device "${QSA_DEVICE:-0}" \
    --timeout "${QSA_CASE_TIMEOUT:-600}" 2>&1 | tee "$qsa_report/console.txt"
qsa_status=${PIPESTATUS[0]}
docker cp "$qsa_container:$qsa_inside/report/." "$qsa_report/" || qsa_status=1
printf '%s\n' "$qsa_status" > "$qsa_report/exit-code.txt"
qsa_stamp=$(date -u +%Y%m%d-%H%M%S)
qsa_branch="codex/qsa-metax-results-$qsa_stamp-$$"
qsa_tree=$(mktemp -d /tmp/flaggems-qsa-metax-results.XXXXXXXX) || exit 1
git -C "$qsa_repo" worktree add -b "$qsa_branch" "$qsa_tree" HEAD || exit 1
qsa_dest="results/qsa_select_paged_prefill/metax/native-probe-$qsa_stamp"
mkdir -p "$qsa_tree/$qsa_dest" || exit 1
cp -R "$qsa_report/." "$qsa_tree/$qsa_dest/" || exit 1
git -C "$qsa_tree" add -- "$qsa_dest" || exit 1
git -C "$qsa_tree" -c user.name="QSA GPU reports" \
    -c user.email="qsa-reports@users.noreply.github.com" \
    commit -m "Report MetaX native QSA smoke correctness" || exit 1
printf 'Report: %s\nResult worktree: %s\n' "$qsa_report" "$qsa_tree"
git -C "$qsa_tree" push https://github.com/cheersluvs/FlagGems-vllm.git \
    "HEAD:refs/heads/$qsa_branch" || {
    printf "Retry: git -C '%s' push https://github.com/cheersluvs/FlagGems-vllm.git 'HEAD:refs/heads/%s'\n" \
        "$qsa_tree" "$qsa_branch"
    exit 1
}
printf '报告回传完成。\n结果分支：%s\n' "$qsa_branch"
git -C "$qsa_tree" rev-parse HEAD
printf 'MetaX native probe exit code: %s\n' "$qsa_status"
exit "$qsa_status"
