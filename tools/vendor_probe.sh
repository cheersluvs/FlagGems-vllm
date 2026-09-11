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

# Dispatch on extension: a .sh probe handed to python fails with a syntax
# error on `set -uo pipefail`, which reads as a broken probe rather than as a
# wrong interpreter.
case "$PROBE" in
    *.sh) bash "$PROBE" "$@" 2>&1 | tee -a "$OUT" ;;
    # PY picks the interpreter: on MetaX the mctle FlagTree build lives in a
    # separate venv, and PATH's python would silently measure the other triton.
    *)    echo "### python: ${PY:-python} -> $(command -v "${PY:-python}")" | tee -a "$OUT"
          "${PY:-python}" "$PROBE" "$@" 2>&1 | tee -a "$OUT" ;;
esac
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

# --- pushing ------------------------------------------------------------
#
# The repo is public, so fetch works anonymously and a box can look fully wired
# up while having no push credentials at all. Rather than three cryptic auth
# failures, work out up front what this box has.
#
# GH_TOKEN / GITHUB_TOKEN, if exported, is used through an inline credential
# helper: the value never enters the URL, the config, the report, or the
# reflog, and nothing here ever echoes it.
CRED=()
WHY=""
if [ -n "${GH_TOKEN:-${GITHUB_TOKEN:-}}" ]; then
    CRED=(-c "credential.helper=!f(){ echo username=cheersluvs; echo \"password=${GH_TOKEN:-$GITHUB_TOKEN}\"; };f")
    WHY="token from the environment"
fi

# Attempt 1 allows a terminal prompt: `credential.helper store` needs exactly
# one interactive answer before it has anything stored, and refusing that is
# how the setup path gets blocked. Retries then go quiet, so a box with no
# credentials costs one prompt rather than three.
pushed=0
for attempt in 1 2 3; do
    if [ "$attempt" -gt 1 ]; then export GIT_TERMINAL_PROMPT=0; fi
    if git "${CRED[@]+"${CRED[@]}"}" push -q origin "HEAD:refs/heads/${BRANCH}"; then
        echo "=== pushed to origin/${BRANCH} (attempt ${attempt})${WHY:+, ${WHY}} ==="
        pushed=1
        break
    fi
    sleep 4
done

if [ "$pushed" -eq 0 ]; then
    echo "=== PUSH FAILED -- the report IS committed locally at $OUT ."
    echo "=== Paste it, or set up credentials once and re-push:"
    cat <<'MSG'
===
===   A. gh, if installed:  gh auth login && gh auth setup-git
===   B. a fine-grained PAT (Contents: Read and write on this repo):
===        git config --global credential.helper store
===        git push origin HEAD        # username: cheersluvs, password: the PAT
===      That writes the token to ~/.git-credentials in PLAIN TEXT; on a shared
===      box prefer  credential.helper 'cache --timeout=3600'
===   C. per-shell, nothing on disk:  export GH_TOKEN=<token>
===
=== Never paste the token into the conversation -- it is not needed there.
MSG
fi
