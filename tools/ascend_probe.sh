#!/usr/bin/env bash
# Run a probe on the Ascend box and ship its output back through GitHub.
#
#   tools/ascend_probe.sh tools/ascend_tle_surface.py
#
# Writes reports/<name>.txt, commits it on the current branch, pushes to origin.
# If the push fails (no credentials on the box), the report is still committed
# locally and the path is printed -- paste it as a fallback.
set -uo pipefail

ROOT=$(cd "$(dirname "$0")/.." && pwd)
cd "$ROOT"

# usage: tools/ascend_probe.sh <probe.py> [report-name] [args passed to the probe...]
PROBE=${1:?usage: tools/ascend_probe.sh <probe.py> [report-name] [probe args...]}
shift
NAME=${1:-$(basename "$PROBE" .py)}
[ $# -gt 0 ] && shift
OUT="reports/${NAME}.txt"
mkdir -p reports

# CANN first, then src -- appending is mandatory: a bare PYTHONPATH=src clobbers
# CANN's own entries and GE init dies with AclSetCompileopt ... 500001.
if [ -f /usr/local/Ascend/cann/set_env.sh ]; then
    # shellcheck disable=SC1091
    source /usr/local/Ascend/cann/set_env.sh
fi
export PYTHONPATH="src${PYTHONPATH:+:$PYTHONPATH}"

echo "### branch $(git rev-parse --abbrev-ref HEAD) @ $(git rev-parse --short HEAD)" | tee "$OUT"
echo "### $(date -Is)  host $(hostname)" | tee -a "$OUT"
echo "### PYTHONPATH=$PYTHONPATH" | tee -a "$OUT"
echo | tee -a "$OUT"

python "$PROBE" "$@" 2>&1 | tee -a "$OUT"
echo
echo "=== report written to $OUT ($(wc -l < "$OUT") lines) ==="

BRANCH=$(git rev-parse --abbrev-ref HEAD)
git add -A reports/
if git diff --cached --quiet; then
    echo "=== report unchanged, nothing to commit ==="
    exit 0
fi
# Use whatever identity this box has; fall back only if it has none, because
# a commit with no mappable email is one GitHub cannot attribute.
IDENT=()
if [ -z "$(git config user.email || true)" ]; then
    IDENT=(-c user.name=cheersluvs -c user.email=yuqingwu51@gmail.com)
fi
git "${IDENT[@]+"${IDENT[@]}"}" commit -q -m "reports: ${NAME} from the Ascend box"
if git push -q origin "HEAD:refs/heads/${BRANCH}"; then
    echo "=== pushed to origin/${BRANCH} ==="
else
    echo "=== PUSH FAILED -- report is committed locally at $OUT; paste it instead ==="
fi
