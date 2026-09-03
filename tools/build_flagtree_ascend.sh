#!/usr/bin/env bash
# Build FlagTree for Ascend into a WHEEL, leaving the installed toolchain alone.
#
#   nohup tools/build_flagtree_ascend.sh 0.6.1rc1+ascend3.5 > ~/ft-build.log 2>&1 &
#   tail -f ~/ft-build.log
#
# A wheel, not `pip install .`: the running environment is the only one known to
# work (25 tests pass on 0.5.0+ascend3.2), and a wheel can be installed into a
# throwaway venv, verified, and only then promoted.  It is also how the current
# install arrived -- someone built it and passed the file along.
set -euo pipefail

TAG=${1:-0.6.1rc1+ascend3.5}
SRC=${FT_SRC:-$HOME/flagtree-src}
OUT=${FT_OUT:-$HOME/flagtree-build}
JOBS=${MAX_JOBS:-64}

# The device and the CPU both have to be idle: a build saturating 192 threads
# next to a benchmark makes that benchmark's numbers unquotable, and those
# numbers are the only evidence the histogram change produced anything.
if pgrep -af "pytest.*benchmark|pytest.*test_top_k" | grep -v $$ | grep -q .; then
    echo "!! a benchmark or test run is still alive -- refusing to start:"
    pgrep -af "pytest.*benchmark|pytest.*test_top_k" | sed 's/^/     /'
    exit 1
fi

# Domestic hosts (the LLVM tarball on ksyuncs) are reachable ONLY without the
# proxy; the proxy is the way out to GitHub.  Nothing here needs GitHub.
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY all_proxy ALL_PROXY || true

echo "=== $(date -Is) building FlagTree $TAG"
echo "=== source $SRC   output $OUT   MAX_JOBS $JOBS"
mkdir -p "$OUT/wheels"
cd "$SRC"
git checkout -q "$TAG"
echo "=== checked out $(git describe --tags --always) $(git log -1 --format=%h)"

export FLAGTREE_BACKEND=ascend
export MAX_JOBS="$JOBS"
python -c "import sys; print('=== python', sys.version)"

echo "=== downloading deps and building (this is the long part)"
time python -m pip wheel . \
     --no-deps --no-build-isolation \
     -w "$OUT/wheels" \
     -v 2>&1

echo
echo "=== built:"
ls -lh "$OUT/wheels"/*.whl
for W in "$OUT/wheels"/*.whl; do sha256sum "$W"; done
echo "=== $(date -Is) done.  NOTHING has been installed; verify in a venv next."
