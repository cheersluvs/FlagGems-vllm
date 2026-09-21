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
#        tools/hygon_prefill_next_run.sh split_alloc_fair [report-name]
#        tools/hygon_prefill_next_run.sh short_special [report-name]
#        tools/hygon_prefill_next_run.sh short_bins_crossover [report-name]
#        tools/hygon_prefill_next_run.sh short_bins_verify [report-name]
#        tools/hygon_prefill_next_run.sh three_remaining [report-name]
#        tools/hygon_prefill_next_run.sh scratch_reuse [report-name]
#        tools/hygon_prefill_next_run.sh scratch_verify [report-name]
#        tools/hygon_prefill_next_run.sh three_new [report-name]
#        tools/hygon_prefill_next_run.sh lane16 [report-name]
#        tools/hygon_prefill_next_run.sh primitives [report-name]
#        tools/hygon_prefill_next_run.sh budget [report-name]
#        tools/hygon_prefill_next_run.sh algorithms [report-name]
#        tools/hygon_prefill_next_run.sh alg_{threshold,streaming,final,delegate} [report-name]
#        tools/hygon_prefill_next_run.sh alg_final_network [report-name]
#        tools/hygon_prefill_next_run.sh alg_final_cached [report-name]
#        tools/hygon_prefill_next_run.sh final_production [report-name]
#        tools/hygon_prefill_next_run.sh scan_paths [report-name]
set -uo pipefail

ROOT=$(cd "$(dirname "$0")/.." && pwd)
cd "$ROOT" || exit 2
BRANCH=codex/hygon-prefill-audit
REMOTE=https://github.com/cheersluvs/FlagGems-vllm.git
STAGE=${1:-}
case "$STAGE" in
    launch|combo|bitonic|private_hist|gaps|focus|collisions|pivot|codegen|step0|vec|vec2|split_workset|split_alloc_fair|short_special|short_bins_crossover|short_bins_verify|three_remaining|scratch_reuse|scratch_verify|three_new|lane16|primitives|budget|algorithms|alg_threshold|alg_streaming|alg_final|alg_delegate|alg_final_network|alg_final_cached|final_production|scan_paths) ;;
    *) echo "Usage: $0 {launch|combo|bitonic|private_hist|gaps|focus|collisions|pivot|codegen|step0|vec|vec2|split_workset|split_alloc_fair|short_special|short_bins_crossover|short_bins_verify|three_remaining|scratch_reuse|scratch_verify|three_new|lane16|primitives|budget|algorithms|alg_threshold|alg_streaming|alg_final|alg_delegate|alg_final_network|alg_final_cached|final_production|scan_paths} [report-name]"; exit 2 ;;
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
elif [ "$STAGE" = split_alloc_fair ]; then
    PROBE=(tools/hygon_prefill_split_alloc_fair.py)
elif [ "$STAGE" = short_special ]; then
    PROBE=(tools/hygon_prefill_short_special.py)
elif [ "$STAGE" = short_bins_crossover ]; then
    PROBE=(tools/hygon_prefill_short_bins_crossover.py)
elif [ "$STAGE" = short_bins_verify ]; then
    PROBE=(tools/hygon_prefill_short_bins_verify.py)
elif [ "$STAGE" = three_remaining ]; then
    PROBE=(tools/hygon_prefill_three_remaining.py)
elif [ "$STAGE" = scratch_reuse ]; then
    PROBE=(tools/hygon_prefill_scratch_reuse.py)
elif [ "$STAGE" = scratch_verify ]; then
    PROBE=(tools/hygon_prefill_scratch_verify.py)
elif [ "$STAGE" = three_new ]; then
    PROBE=(tools/hygon_prefill_three_new.py)
elif [ "$STAGE" = lane16 ]; then
    PROBE=(tools/hygon_prefill_lane16.py)
elif [ "$STAGE" = primitives ]; then
    PROBE=(tools/hygon_prefill_primitives.py)
elif [ "$STAGE" = budget ]; then
    PROBE=(tools/hygon_prefill_budget.py)
elif [ "$STAGE" = algorithms ]; then
    PROBE=(tools/hygon_prefill_algorithms.py all)
elif [ "$STAGE" = alg_final_network ]; then
    PROBE=(tools/hygon_prefill_algorithms.py final --variant network)
elif [ "$STAGE" = alg_final_cached ]; then
    PROBE=(tools/hygon_prefill_algorithms.py final --variant network --cache-final)
elif [ "$STAGE" = final_production ]; then
    PROBE=(tools/hygon_prefill_final_production.py)
elif [ "$STAGE" = scan_paths ]; then
    PROBE=(tools/hygon_prefill_scan_paths.py)
elif [[ "$STAGE" == alg_* ]]; then
    PROBE=(tools/hygon_prefill_algorithms.py "${STAGE#alg_}")
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