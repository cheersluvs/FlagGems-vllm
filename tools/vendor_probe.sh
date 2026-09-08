#!/usr/bin/env bash
# Run a probe on a vendor card and ship its output back through GitHub.
#
#   tools/vendor_probe.sh tools/topk_preflight.py
#   tools/vendor_probe.sh tools/topk_preflight.py preflight_run --run prefill
#
# Writes reports/<name>.txt, commits it on the current branch, pushes to origin.
# Every step that can fail says so; nothing is reported as done that was not.
set -uo pipefail

ROOT=$(cd "$(dirname "$0")/.." && pwd)
cd "$ROOT"

# A detached HEAD makes `git rev-parse --abbrev-ref HEAD` return the literal
# string "HEAD", so the push at the end would create a branch called HEAD on
# the fork. Refuse up front rather than after a long probe.
if ! BRANCH=$(git symbolic-ref --quiet --short HEAD); then
    echo "!! detached HEAD -- check out the working branch first:"
    echo "     git fetch origin <branch> && git checkout -B <branch> origin/<branch>"
    exit 2
fi

# usage: tools/vendor_probe.sh <probe.py> [report-name] [args passed to the probe...]
PROBE=${1:?usage: tools/vendor_probe.sh <probe.py> [report-name] [probe args...]}
shift
NAME=${1:-$(basename "$PROBE" .py)}
[ $# -gt 0 ] && shift
OUT="reports/${NAME}.txt"
mkdir -p reports

# Vendor environment first, then src. APPENDING is mandatory, not cosmetic: a
# bare PYTHONPATH=src clobbers CANN's own entries and GE init dies with
# `AclSetCompileopt ... 500001`, whose real cause (`No module named 'tbe'`)
# only appears in ~/ascend/log/debug/plog/. Harmless on the other cards, which
# is why the shorter form looks fine until it is not.
if [ -f /usr/local/Ascend/cann/set_env.sh ]; then
    # shellcheck disable=SC1091
    source /usr/local/Ascend/cann/set_env.sh
fi
export PYTHONPATH="src${PYTHONPATH:+:$PYTHONPATH}"

{
    echo "### branch ${BRANCH} @ $(git rev-parse --short HEAD)"
    echo "### $(date -Is)  host $(uname -n)"
    echo "### PYTHONPATH=$PYTHONPATH"
    echo
} | tee "$OUT"

python "$PROBE" "$@" 2>&1 | tee -a "$OUT"
echo
echo "=== report written to $OUT ($(wc -l < "$OUT") lines) ==="

git add -A reports/
if git diff --cached --quiet; then
    echo "=== report unchanged, nothing to commit ==="
    exit 0
fi

# Use whatever identity this box has; fall back only if it has none, because a
# commit with no mappable email is one GitHub cannot attribute. NOTE the -c
# form covers only THIS commit -- `git pull --rebase` replays commits and reads
# the config, so a box with no identity still needs it set globally.
IDENT=()
if [ -z "$(git config user.email || true)" ]; then
    IDENT=(-c user.name=cheersluvs -c user.email=yuqingwu51@gmail.com)
fi
if ! git "${IDENT[@]+"${IDENT[@]}"}" commit -q -m "reports: ${NAME} from $(uname -n)"; then
    echo "=== COMMIT FAILED -- the report is on disk at $OUT but is NOT committed."
    echo "=== Set an identity, then commit by hand:"
    echo "===   git config --global user.name cheersluvs"
    echo "===   git config --global user.email <the github email>"
    exit 1
fi

# These links drop connections; one refusal is not a verdict.
export GIT_TERMINAL_PROMPT=0
for attempt in 1 2 3; do
    if git push -q origin "HEAD:refs/heads/${BRANCH}"; then
        echo "=== pushed to origin/${BRANCH} (attempt ${attempt}) ==="
        exit 0
    fi
    sleep 4
done
echo "=== PUSH FAILED after 3 attempts -- the report IS committed locally at"
echo "=== $OUT ; paste it, or push again once the link is back."
