#!/usr/bin/env bash
# Run one ordered BW1000 experiment stage and push its complete report.
# Usage: tools/hygon_prefill_next_run.sh launch [report-name]
#        tools/hygon_prefill_next_run.sh combo  [report-name]
#        tools/hygon_prefill_next_run.sh bitonic [report-name]
#        tools/hygon_prefill_next_run.sh private_hist [report-name]
#        tools/hygon_prefill_next_run.sh gaps [report-name]
#        tools/hygon_prefill_next_run.sh focus [report-name]
#        tools/hygon_prefill_next_run.sh collisions [report-name]
#        tools/hygon_prefill_next_run.sh pivot [report-name]
#        tools/hygon_prefill_next_run.sh codegen [report-name]
#        tools/hygon_prefill_next_run.sh step0 [report-name]
#        tools/hygon_prefill_next_run.sh vec [report-name]
#        tools/hygon_prefill_next_run.sh vec2 [report-name]
#        tools/hygon_prefill_next_run.sh split_workset [report-name]
set -uo pipefail

ROOT=$(cd "$(dirname "$0")/.." && pwd)
cd "$ROOT" || exit 2
BRANCH=codex/hygon-prefill-audit
REMOTE=https://github.com/cheersluvs/FlagGems-vllm.git
STAGE=${1:-}
case "$STAGE" in
    launch|combo|bitonic|private_hist|gaps|focus|collisions|pivot|codegen|step0|vec|vec2|split_workset) ;;
    *) echo "Usage: $0 {launch|combo|bitonic|private_hist|gaps|focus|collisions|pivot|codegen|step0|vec|vec2|split_workset} [report-name]"; exit 2 ;;
esac
NAME=${2:-hygon_prefill_${STAGE}_device_v1}
if [[ ! "$NAME" =~ ^[A-Za-z0-9_-]+$ ]]; then
    echo "Invalid report name."
    exit 2
fi
if [ "$(git symbolic-ref --quiet --short HEAD)" != "$BRANCH" ]; then
    echo "Run in the dedicated $BRANCH worktree."
    exit 2
fi
if ! git diff --cached --quiet || ! git diff --quiet -- src tools; then
    echo "Commit intended source/tool changes and clear the index before measuring."
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
    echo "### stage $STAGE"
    echo "### HIP_VISIBLE_DEVICES=${HIP_VISIBLE_DEVICES:-unset} CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
} | tee "$OUT"

if [ "$STAGE" = bitonic ]; then
    PROBE=(tools/hygon_prefill_bitonic.py)
elif [ "$STAGE" = private_hist ]; then
    PROBE=(tools/hygon_prefill_private_hist.py)
elif [ "$STAGE" = gaps ]; then
    PROBE=(tools/hygon_prefill_gaps.py all)
elif [ "$STAGE" = focus ]; then
    PROBE=(tools/hygon_prefill_focus.py)
elif [ "$STAGE" = collisions ]; then
    PROBE=(tools/hygon_prefill_collisions.py)
elif [ "$STAGE" = pivot ]; then
    PROBE=(tools/hygon_prefill_pivot.py)
elif [ "$STAGE" = codegen ]; then
    PROBE=(tools/hygon_prefill_codegen.py)
elif [ "$STAGE" = step0 ]; then
    PROBE=(tools/hygon_prefill_step0_only.py)
elif [ "$STAGE" = vec ]; then
    PROBE=(tools/hygon_prefill_vec.py)
elif [ "$STAGE" = vec2 ]; then
    PROBE=(tools/hygon_prefill_vec2_trial.py)
elif [ "$STAGE" = split_workset ]; then
    PROBE=(tools/hygon_prefill_split_workset.py)
else
    PROBE=(tools/hygon_prefill_next.py "$STAGE")
fi
echo "### RUN timeout 14400 ${PY:-python} -u ${PROBE[*]}" | tee -a "$OUT"
timeout 14400 "${PY:-python}" -u "${PROBE[@]}" 2>&1 | tee -a "$OUT"
STATUS=("${PIPESTATUS[@]}")
PROBE_STATUS=${STATUS[0]}
if [ "${STATUS[1]}" -ne 0 ]; then
    echo "Report write failed; probe status $PROBE_STATUS."
    exit 1
fi
echo "### probe_exit=$PROBE_STATUS" | tee -a "$OUT"

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
echo "Report pushed: $OUT; probe exit=$PROBE_STATUS"
exit "$PROBE_STATUS"
