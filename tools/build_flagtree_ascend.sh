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

echo "=== $(date -Is) building FlagTree $TAG"
echo "=== source $SRC   output $OUT   MAX_JOBS $JOBS"
mkdir -p "$OUT/wheels"
cd "$SRC"
git checkout -q "$TAG"
echo "=== checked out $(git describe --tags --always) $(git log -1 --format=%h)"

# This build needs BOTH network directions at once, which no single proxy
# setting gives:
#   * the 2.15 GB LLVM tarball (ksyuncs) and the Huawei mirrors answer ONLY
#     with the proxy unset;
#   * setup fetches third_party/ascend/AscendNPU-IR by cloning
#     github.com/Ascend/AscendNPU-IR at 4c304921 (python/setup_tools/utils/
#     ascend.py) -- and github.com answers ONLY through the proxy.
# It is not a git submodule, so `git submodule update` is a silent no-op; the
# first attempt left an empty directory with a bare .git in it and CMake then
# stopped at add_subdirectory, after the whole download had succeeded.
#
# So: drop the proxy from the environment, and hand git a github.com-only proxy
# through GIT_CONFIG_* -- which git reads without writing ~/.gitconfig, so the
# credentials in that URL are not persisted to disk.
PROXY_SAVED="${https_proxy:-${http_proxy:-}}"
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY all_proxy ALL_PROXY || true
if [ -n "$PROXY_SAVED" ]; then
    export GIT_CONFIG_COUNT=1
    export GIT_CONFIG_KEY_0="http.https://github.com/.proxy"
    export GIT_CONFIG_VALUE_0="$PROXY_SAVED"
    echo "=== git will reach github.com through the proxy; everything else direct"
fi

# Anything left from a previous tag has to go.  Three separate failures came
# from stale state, each surfacing later and less clearly than the last:
#   * a build/ tree configured for another tag makes CMake fail to regenerate
#     build.ninja ("ninja: error: rebuilding 'build.ninja'"), with no CMake
#     error of its own to read;
#   * that same stale tree is why 0.6.1 could not find a generated
#     bishengir/InitAllDialects.h -- not, as it looked, a wrong AscendNPU-IR
#     commit: 0.6.1 and 0.6.1rc1 pin the identical 4c304921.
# setup deletes the .git of its AscendNPU-IR clone, so its commit cannot be
# read back and a stamp file is the only reliable record of what is there.
STAMP="$SRC/.flagtree_built_tag"
LAST=$(cat "$STAMP" 2>/dev/null || true)
if [ "${LAST:-}" != "$TAG" ]; then
    echo "=== tag changed (${LAST:-none} -> $TAG): wiping build tree and AscendNPU-IR"
    rm -rf "$SRC/build" "$SRC/third_party/ascend/AscendNPU-IR"
else
    echo "=== same tag as the last build, keeping the build tree"
fi
printf '%s' "$TAG" > "$STAMP"

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
