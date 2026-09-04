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
FT_REPO=${FT_REPO:-https://github.com/flagos-ai/FlagTree.git}
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

# Fail on a missing tool now, not forty minutes into a CMake configure.
MISSING=""
for T in cmake ninja git python; do
    command -v "$T" >/dev/null || MISSING="$MISSING $T"
done
# Either compiler will do: LLVM arrives prebuilt, so only Triton and BiShengIR
# are compiled here.  A CANN container typically has gcc and no clang.
if command -v clang++ >/dev/null; then
    CXX_FOUND="clang++ $(clang++ --version | head -1)"
elif command -v g++ >/dev/null; then
    CXX_FOUND="g++ $(g++ --version | head -1)"
else
    MISSING="$MISSING clang++/g++"
fi
if [ -n "$MISSING" ]; then
    echo "!! missing build tools:$MISSING"
    exit 1
fi
echo "=== compiler: ${CXX_FOUND:-?}"
echo "=== cmake: $(cmake --version | head -1)"

mkdir -p "$OUT/wheels"
if [ ! -d "$SRC/.git" ]; then
    echo "=== cloning FlagTree into $SRC (GitHub needs the proxy, still set here)"
    git clone -q --filter=blob:none "$FT_REPO" "$SRC" || {
        echo "!! clone failed"; exit 1; }
fi
cd "$SRC"
git fetch -q --tags origin 2>/dev/null || true
git checkout -q "$TAG" || { echo "!! no such tag: $TAG"; exit 1; }
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
# Fetch the GitHub-hosted dependencies while the proxy is still in the
# environment.  setup downloads nlohmann/json from github.com with urllib, which
# ignores the git-only proxy configured below, so with the proxy gone it times
# out -- after the 2.15 GB LLVM and six CUDA archives have already succeeded.
JSON_DIR=${JSON_DIR:-$HOME/.triton/json}
if [ ! -d "$JSON_DIR/include" ]; then
    echo "=== prefetching nlohmann/json into $JSON_DIR (needs the proxy)"
    mkdir -p "$JSON_DIR"
    python - "$JSON_DIR" <<'PYEOF' || echo "!! json prefetch failed; the build will try on its own"
import io, sys, urllib.request, zipfile
url = "https://github.com/nlohmann/json/releases/download/v3.11.3/include.zip"
with urllib.request.urlopen(url, timeout=120) as r:
    data = r.read()
zipfile.ZipFile(io.BytesIO(data)).extractall(sys.argv[1])
print(f"    extracted {len(data)} bytes")
PYEOF
    ls -d "$JSON_DIR/include" >/dev/null 2>&1 && echo "=== json ok" || echo "!! json still missing"
else
    echo "=== nlohmann/json already present at $JSON_DIR"
fi

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

# LLVM_SYSPATH short-circuits the 2.15 GB download and uses a local LLVM
# instead -- setup_helper's ascend entry has `pre_hock=check_env('LLVM_SYSPATH')`.
# That is almost certainly how the vendor's container build was matched to
# CANN's own bishengir-opt: FlagTree's default pin produces MLIR 22 bytecode,
# which a bishengir-opt built on an older MLIR refuses outright.
if [ -n "${LLVM_SYSPATH:-}" ]; then
    echo "=== LLVM_SYSPATH=$LLVM_SYSPATH (skipping the LLVM download)"
    export LLVM_SYSPATH
else
    echo "=== no LLVM_SYSPATH: FlagTree will download its pinned LLVM (MLIR 22)."
    echo "===   If the resulting wheel dies in bishengir-opt with"
    echo "===   'bytecode version N produced by MLIR22', rebuild with LLVM_SYSPATH"
    echo "===   pointing at an LLVM matching this CANN's bishengir-opt."
fi

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
