#!/usr/bin/env bash
# On the Hygon experiment worktree, validate and benchmark the opt-in carry arm.
# Always preserve the report, including a failed validation or benchmark.
set -uo pipefail

ROOT=$(cd "$(dirname "$0")/.." && pwd)
cd "$ROOT" || exit 2
BRANCH=codex/hygon-prefill-audit
REMOTE=https://github.com/cheersluvs/FlagGems-vllm.git
NAME=${1:-hygon_prefill_carry_v1}
if [[ ! "$NAME" =~ ^[A-Za-z0-9_-]+$ ]]; then
    echo "Invalid report name."
    exit 2
fi
if [ "$(git symbolic-ref --quiet --short HEAD)" != "$BRANCH" ]; then
    echo "Run in the dedicated $BRANCH worktree."
    exit 2
fi
if ! git diff --cached --quiet || ! git diff --quiet -- src tools; then
    echo "Commit intended source/tool changes before measuring; index and source must be clean."
    exit 2
fi
if ! command -v timeout >/dev/null; then
    echo "GNU timeout is required for bounded remote trials."
    exit 2
fi

OUT="reports/${NAME}.txt"
if [ -e "$OUT" ]; then
    echo "Report already exists: $OUT. Choose a new name."
    exit 2
fi
mkdir -p reports
export PYTHONPATH="src${PYTHONPATH:+:$PYTHONPATH}"
{
    echo "### branch $BRANCH @ $(git rev-parse HEAD)"
    echo "### $(date -Is) host $(uname -n)"
    echo "### python ${PY:-python}"
    echo "### HIP_VISIBLE_DEVICES=${HIP_VISIBLE_DEVICES:-unset} CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
} | tee "$OUT"

record() {
    echo "### RUN $*" | tee -a "$OUT"
    "$@" 2>&1 | tee -a "$OUT"
    local code=${PIPESTATUS[0]}
    echo "### EXIT $code" | tee -a "$OUT"
    return "$code"
}

record hy-smi || true
overall=0
SOURCE_CHECK='import hashlib; import importlib; from pathlib import Path; ov=importlib.import_module("flaggems_vllm.runtime.backend._hygon.fused.top_k_per_row_prefill"); expected="869cf66cd5408b353333684c63a35a29f1f0d68f082eca50635beb3a7ccde777"; assert ov._ONESCAN_PATH and ov._ENABLED and ov._GEOMETRY; assert ov._dense_carry is not None and ov._CARRY_PATH; actual=hashlib.sha256(Path(ov._CARRY_PATH).read_bytes()).hexdigest(); print("carry source sha256", actual); assert actual == expected, (actual, expected)'
if record timeout 1200 env FLAGGEMS_HYGON_TOPK_CARRY=1 "${PY:-python}" -c "$SOURCE_CHECK"; then
    if ! record timeout 1200 env FLAGGEMS_HYGON_TOPK_CARRY=1 "${PY:-python}" -m pytest -q tests/test_top_k_per_row_prefill.py; then
        overall=1
    fi
else
    overall=1
fi

if [ "$overall" -eq 0 ]; then
    # Separate processes load the correct env-gated module. B-C-C-B limits
    # drift; never quote a chosen minimum as the result.
    for mode in 0 1 1 0; do
        if ! record timeout 1200 env FLAGGEMS_HYGON_TOPK_CARRY="$mode" "${PY:-python}" -m pytest -q -s benchmark/test_top_k_per_row_prefill.py --mode kernel; then
            overall=1
            break
        fi
    done
fi
record hy-smi || true
echo "### trial_exit=$overall" | tee -a "$OUT"

IDENT=()
if [ -z "$(git config user.email || true)" ]; then
    IDENT=(-c user.name=cheersluvs -c user.email=yuqingwu51@gmail.com)
fi
git add -- "$OUT" || exit 1
if ! git "${IDENT[@]}" commit -m "reports: $NAME from $(uname -n)"; then
    echo "COMMIT FAILED: $OUT remains local."
    exit 1
fi
CRED=()
if [ -n "${GH_TOKEN:-${GITHUB_TOKEN:-}}" ]; then
    CRED=(-c 'credential.helper=!f() { if [ "$1" = get ]; then printf "%s\n" "username=cheersluvs" "password=${GH_TOKEN:-$GITHUB_TOKEN}"; fi; }; f')
fi
if ! git "${CRED[@]}" push "$REMOTE" "HEAD:refs/heads/$BRANCH"; then
    echo "PUSH FAILED: $OUT is committed locally but has not reached GitHub."
    exit 1
fi
echo "Report pushed: $OUT; trial exit=$overall"
exit "$overall"
