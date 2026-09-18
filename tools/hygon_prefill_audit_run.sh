#!/usr/bin/env bash
# Run in the dedicated experiment worktree, commit only this report, and return
# it to the Mac's fork. A failed probe is reported and still preserves its log.
set -uo pipefail

ROOT=$(cd "$(dirname "$0")/.." && pwd)
cd "$ROOT" || exit 2
BRANCH=codex/hygon-prefill-audit
REMOTE=https://github.com/cheersluvs/FlagGems-vllm.git
NAME=${1:-hygon_prefill_audit_v1}
if [ "$#" -gt 0 ]; then shift; fi
if [[ ! "$NAME" =~ ^[A-Za-z0-9_-]+$ ]]; then
    echo "Invalid report name: use letters, numbers, underscore or hyphen."
    exit 2
fi
if [ "$(git symbolic-ref --quiet --short HEAD)" != "$BRANCH" ]; then
    echo "Run this probe in the dedicated $BRANCH worktree."
    exit 2
fi
if ! git diff --cached --quiet; then
    echo "The index already has staged changes; preserve them and use a clean worktree."
    exit 2
fi
if ! git diff --quiet -- src tools; then
    echo "Tracked source/tool changes exist. Commit the intended probe version before measuring."
    exit 2
fi

OUT="reports/${NAME}.txt"
if [ -e "$OUT" ]; then
    echo "Report already exists: $OUT. Choose a new report name to preserve it."
    exit 2
fi
mkdir -p reports
export PYTHONPATH="src${PYTHONPATH:+:$PYTHONPATH}"
{
    echo "### branch $BRANCH @ $(git rev-parse HEAD)"
    echo "### $(date -Is) host $(uname -n)"
    echo "### python ${PY:-python}"
} | tee "$OUT"
"${PY:-python}" -u tools/hygon_prefill_audit.py "$@" 2>&1 | tee -a "$OUT"
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
    echo "COMMIT FAILED: report remains at $OUT (probe exit $PROBE_STATUS)."
    exit 1
fi
CRED=()
if [ -n "${GH_TOKEN:-${GITHUB_TOKEN:-}}" ]; then
    # Expand the token only within the helper; do not put its value in argv,
    # the remote URL, git configuration or the report.
    CRED=(-c 'credential.helper=!f() { if [ "$1" = get ]; then printf "%s\n" "username=cheersluvs" "password=${GH_TOKEN:-$GITHUB_TOKEN}"; fi; }; f')
fi
if ! git "${CRED[@]}" push "$REMOTE" "HEAD:refs/heads/$BRANCH"; then
    echo "PUSH FAILED: $OUT is committed locally; report has not reached GitHub."
    exit 1
fi
echo "Report pushed: $OUT; probe exit=$PROBE_STATUS (0 means all requested shapes passed)."
exit "$PROBE_STATUS"
