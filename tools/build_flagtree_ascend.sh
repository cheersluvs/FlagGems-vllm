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

# Submodules FIRST, and while the proxy is still set: third_party/ascend/
# AscendNPU-IR is a submodule, and without it CMake stops at
#   add_subdirectory: .../AscendNPU-IR does not contain a CMakeLists.txt
# after the whole 2.15 GB LLVM download has already succeeded.  Their remotes
# may be domestic or on GitHub, so try with the proxy and fall back without it.
echo "=== initialising submodules"
if ! git submodule update --init --recursive --depth 1 2>&1 | tail -5; then
    echo "=== retrying submodules without the proxy"
    env -u http_proxy -u https_proxy -u HTTP_PROXY -u HTTPS_PROXY \
        -u all_proxy -u ALL_PROXY \
        git submodule update --init --recursive --depth 1 2>&1 | tail -5
fi
for D in third_party/ascend/AscendNPU-IR; do
    [ -f "$D/CMakeLists.txt" ] && echo "=== $D ok" \
        || { echo "!! $D still empty -- build would fail at CMake"; exit 1; }
done

# Only now drop the proxy: the LLVM tarball on ksyuncs and the Huawei mirrors
# answer ONLY without it, while github.com answers only with it.
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY all_proxy ALL_PROXY || true

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
