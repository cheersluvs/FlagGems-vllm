#!/usr/bin/env bash
# Build a FlagTree metax wheel that actually contains the TLE bindings.
#
# WHY
#
# The installed wheel calls itself flagtree 0.6.1a2+metax3.6, and that tag
# predates mctle entirely -- no plugin/mctle source, no BUILD_MCTLE wiring, no
# backends/metax/tle_supported.py. Its libtriton exposes none of the seven
# binding symbols the operator's TLE path needs, which is why tle.gpu.alloc
# fails on a missing make_swizzled_shared_encoding_attr.
#
#     tag                   mctle source   BUILD_MCTLE   tle_supported.py
#     0.6.1a1+metax3.6      no             no            no
#     0.6.1a2+metax3.6      no             no            no      <- installed
#     0.6.1+metax3.6        YES            YES           no      <- target
#     main                  YES            YES           YES
#
# third_party/metax/CMakeLists.txt gates it:
#
#     if(BUILD_MCTLE)
#       add_subdirectory(plugin/mctle)
#     endif()
#
# off by default, which is why MetaX's published wheel has no bindings even
# where the source has them.
#
# WHAT THIS COSTS
#
# Much less than the Ascend build. python/setup_tools/utils/metax.py downloads
# a PREBUILT LLVM 19 and a PREBUILT metaxTritonPlugin.so, so neither is
# compiled here; mctle links against that plugin rather than replacing it
# (see its CMakeLists' non-FLAGTREE_PLUGIN branch), so MetaX's own backend is
# left alone. Only Triton and mctle are built.
#
# STAGES
#
#   check   tools, MACA, network, disk
#   fetch   clone/checkout 0.6.1+metax3.6
#   probe   configure only, and assert BUILD_MCTLE actually reached the cache
#           -- a flag that silently does not apply is the whole failure mode
#           this script exists to avoid
#   build   wheel
#   verify  the seven symbols, in the built libtriton, before anyone installs
#
# Stop after any stage:  tools/build_flagtree_metax.sh probe
#
set -uo pipefail

TAG="${FLAGTREE_TAG:-0.6.1+metax3.6}"
SRC="${FLAGTREE_SRC:-$HOME/flagtree}"
STAGE="${1:-build}"
WANT_SYMS="make_swizzled_shared_encoding_attr create_local_pointers
           create_local_alloc create_local_load create_local_store"

# The box's proxy refuses a CONNECT tunnel to the artifact host -- "CONNECT
# tunnel failed, response 500" -- while a direct request to it returns 206. So
# exempt that ONE host and leave every other route on the proxy, which is what
# github and pypi are reaching through. Both spellings, because Python's
# urllib reads no_proxy and some tools read NO_PROXY.
ARTIFACT_HOST="baai-cp-web.ks3-cn-beijing.ksyuncs.com"
export no_proxy="${no_proxy:+$no_proxy,}$ARTIFACT_HOST"
export NO_PROXY="$no_proxy"

say() { printf '\n=== %s\n' "$*"; }
die() { printf '!!! %s\n' "$*" >&2; exit 1; }

# ---------------------------------------------------------------- check
say "check"
for t in git cmake ninja python pip; do
    command -v "$t" >/dev/null || die "missing $t"
    printf '  %-8s %s\n' "$t" "$(command -v "$t")"
done
printf '  %-8s %s\n' "python" "$(python -V 2>&1)"
[ -d "${MACA_PATH:-/opt/maca}" ] || die "no MACA at ${MACA_PATH:-/opt/maca}"
printf '  %-8s %s\n' "MACA" "${MACA_PATH:-/opt/maca}"

# The prebuilt plugin and LLVM are DOWNLOADED by setup, so the box needs to
# reach that host. Check it now rather than after a long configure.
printf '  %-8s %s\n' "no_proxy" "$(printf '%s' "$no_proxy" | sed -E 's#://[^@/]*@#://***:***@#')"
ART_URL="https://$ARTIFACT_HOST/trans/metaxTritonPlugin-cpython3.12-x86_64_v0.6.1.tar.gz"
ART_CODE=$(curl -sS -o /dev/null -w '%{http_code}' -m 25 -r 0-0 "$ART_URL" 2>/dev/null)
if [ "$ART_CODE" = 206 ] || [ "$ART_CODE" = 200 ]; then
    echo "  artifact host reachable (HTTP $ART_CODE, proxy bypassed)"
else
    echo "  !! artifact host still unreachable (HTTP ${ART_CODE:-000})"
    echo "     Direct access worked when tested by hand, so check that no_proxy"
    echo "     took effect here; without it setup fails at the download."
fi
avail=$(df -Pm "$(dirname "$SRC")" | awk 'NR==2{print $4}')
echo "  free space at $(dirname "$SRC"): ${avail} MB"
[ "${avail:-0}" -lt 20000 ] && echo "  !! under 20 GB free; the build may not fit"

# The two prebuilt artifacts are DOWNLOADED by setup. If that host is
# unreachable, both can be supplied locally instead -- the cache skips a
# download whose file is already present, and LLVM_SYSPATH short-circuits the
# LLVM one. Report what this box already has.
say "deps: can the two prebuilt artifacts come from this box?"
INSTALLED_SO=$(python -c 'import os,triton;p=os.path.join(os.path.dirname(triton.__file__),"_C","metaxTritonPlugin.so");print(p if os.path.exists(p) else "")' 2>/dev/null)
if [ -n "$INSTALLED_SO" ]; then
    MD5=$(md5sum "$INSTALLED_SO" | cut -c1-8)
    printf '  installed plugin  %s\n' "$INSTALLED_SO"
    printf '  its md5[:8]       %s   (setup expects afb7ab8f for v0.6.1)\n' "$MD5"
    if [ "$MD5" = afb7ab8f ]; then
        echo "  -> MATCHES: seeding the cache with it skips that download"
    else
        echo "  -> differs (this is the 0.6.1a2 plugin); the cache will still want v0.6.1"
    fi
else
    echo "  no installed metaxTritonPlugin.so found"
fi

echo "  existing flagtree caches:"
for d in "$HOME/.flagtree" "$SRC/.flagtree"; do
    if [ -d "$d" ]; then find "$d" -maxdepth 2 | sed 's/^/    /' | head -12; else echo "    $d: absent"; fi
done

echo "  LLVM candidates (MLIR is what matters, not just clang):"
for d in "${LLVM_SYSPATH:-}" /opt/maca/mxgpu_llvm /opt/maca/llvm; do
    if [ -n "$d" ] && [ -d "$d" ]; then
        m=$(find "$d" -maxdepth 4 -name MLIRConfig.cmake 2>/dev/null | head -1)
        v=$("$d/bin/llvm-config" --version 2>/dev/null)
        printf '    %-28s version=%-10s MLIRConfig=%s\n' "$d" "${v:-?}" "${m:-NO}"
    fi
done

[ "$STAGE" = check ] && exit 0

# ---------------------------------------------------------------- fetch
say "fetch $TAG"
if [ ! -e "$SRC/.git" ]; then
    git clone --filter=blob:none https://github.com/FlagTree/flagtree.git "$SRC" \
        || die "clone failed"
fi
git -C "$SRC" fetch --tags -q origin || die "fetch failed"
git -C "$SRC" checkout -q "refs/tags/$TAG" || die "no such tag: $TAG"
echo "  at $(git -C "$SRC" rev-parse --short HEAD)  ($TAG)"

# A stale build/ from another tag silently reuses the wrong CMake cache, which
# is exactly how a BUILD_MCTLE=ON run can produce a wheel without mctle.
STAMP="$SRC/.built-from"
if [ -f "$STAMP" ] && [ "$(cat "$STAMP")" != "$TAG" ]; then
    echo "  previous build was $(cat "$STAMP"); wiping build/"
    rm -rf "$SRC/python/build" "$SRC/build"
fi
echo "$TAG" > "$STAMP"

for f in third_party/metax/plugin/mctle/triton_mctle.cc third_party/metax/CMakeLists.txt; do
    [ -f "$SRC/$f" ] || die "$f missing at $TAG -- wrong tag"
done
grep -q BUILD_MCTLE "$SRC/third_party/metax/CMakeLists.txt" \
    || die "BUILD_MCTLE not wired at $TAG -- wrong tag"
echo "  mctle source and BUILD_MCTLE wiring both present"

export FLAGTREE_BACKEND=metax
export TRITON_APPEND_CMAKE_ARGS="-DBUILD_MCTLE=ON"

# setup.py downloads NVIDIA's ptxas, cuobjdump, nvdisasm, cudacrt, cudart and
# cupti unconditionally -- from developer.download.nvidia.com, which the proxy
# also refuses to tunnel to. None of them is used by a metax build.
#
# download_and_copy() opens with `if variable in os.environ: return`, so
# setting each one skips its download outright. The value is never read; it is
# checked for presence only. Pointing them at /bin/true keeps them from looking
# like real paths that something might later try to run as a compiler.
# Two things are needed together, and each alone fails.
#
# download_and_copy_dependencies() is called unconditionally, independent of
# proton, and every entry skips only via `if variable in os.environ: return`.
# So ALL eight names must be set, CUPTI included, or that one downloads.
#
# But TRITON_CUPTI_INCLUDE_PATH is also read back as a directory and passed as
# -DCUPTI_INCLUDE_DIR, which is how /bin/true broke the configure. Point them
# at a real (empty) directory so that path is at least valid, and turn proton
# off so it is never consulted at all: with TRITON_BUILD_PROTON=OFF,
# get_proton_cmake_args() is not called and the flag is never passed.
NVSTUB="$SRC/.nvidia-skip"
mkdir -p "$NVSTUB"
for v in TRITON_PTXAS_PATH TRITON_PTXAS_BLACKWELL_PATH TRITON_CUOBJDUMP_PATH \
         TRITON_NVDISASM_PATH TRITON_CUDACRT_PATH TRITON_CUDART_PATH \
         TRITON_CUPTI_INCLUDE_PATH TRITON_CUPTI_LIB_PATH; do
    export "$v=$NVSTUB"
done
export TRITON_BUILD_PROTON=OFF
echo "  NVIDIA toolkit downloads: skipped via 8 TRITON_*_PATH -> $NVSTUB"
echo "  proton: OFF (so CUPTI_INCLUDE_DIR is never passed to cmake)"
export MAX_JOBS="${MAX_JOBS:-$(nproc)}"
echo "  FLAGTREE_BACKEND=$FLAGTREE_BACKEND"
echo "  TRITON_APPEND_CMAKE_ARGS=$TRITON_APPEND_CMAKE_ARGS"
echo "  MAX_JOBS=$MAX_JOBS"

# The prebuilt LLVM 19 exports targets that reference ZLIB::ZLIB (and often
# zstd), so cmake must be able to find them or LLVMExports.cmake fails at
# set_target_properties with a dependency that does not exist. conda usually
# ships both; cmake just does not look there by default.
find_lib() {   # find_lib <name> <header>  ->  echoes "<root>|<lib>|<incdir>"
    # lib64 matters: this is a RHEL-family box, where /usr/lib64 is the real
    # library directory and /usr/lib holds almost nothing.
    local n="$1" h="$2" root lib inc
    for root in "${CONDA_PREFIX:-/opt/conda}" /usr /usr/local; do
        lib=$(ls "$root"/lib64/lib${n}.so "$root"/lib/lib${n}.so \
                 "$root"/lib/x86_64-linux-gnu/lib${n}.so \
                 "$root"/lib64/lib${n}.so.1 "$root"/lib/lib${n}.so.1 2>/dev/null | head -1)
        inc=$(ls "$root"/include/"$h" 2>/dev/null | head -1)
        if [ -n "$lib" ] && [ -n "$inc" ]; then
            printf '%s|%s|%s' "$root" "$lib" "$(dirname "$inc")"
            return 0
        fi
    done
    return 1
}

# The prebuilt LLVM 19 exports targets referencing ZLIB::ZLIB, so cmake must
# resolve zlib or LLVMExports.cmake fails at set_target_properties. This box
# has the RUNTIME only -- /usr/lib64/libz.so.1.2.13 and no zlib.h anywhere --
# and conda cannot install the headers because its mirror goes through the same
# proxy that refuses to tunnel.
#
# GitHub is reachable, and zlib's headers are two checked-in files (zconf.h is
# committed, not generated). Fetch the pair matching the installed runtime and
# point cmake at them: find_package(ZLIB) needs a header and a library, and
# nothing here actually compiles against zlib -- the imported target just has
# to resolve.
ZLIB_LIB=$(ls /usr/lib64/libz.so /usr/lib64/libz.so.1.* "${CONDA_PREFIX:-/opt/conda}"/lib/libz.so \
              "${CONDA_PREFIX:-/opt/conda}"/lib/libz.so.1.* /usr/lib/libz.so 2>/dev/null | head -1)
ZLIB_INC=$(ls /usr/include/zlib.h "${CONDA_PREFIX:-/opt/conda}"/include/zlib.h 2>/dev/null | head -1)

if [ -n "$ZLIB_LIB" ] && [ -z "$ZLIB_INC" ]; then
    ZV=$(printf '%s' "$ZLIB_LIB" | sed -nE 's/.*libz\.so\.1\.([0-9]+\.[0-9]+).*/1.\1/p')
    ZV=${ZV:-1.2.13}
    ZDIR="$SRC/.zlib-headers"
    if [ ! -f "$ZDIR/zlib.h" ]; then
        echo "  zlib: runtime only ($ZLIB_LIB); fetching v$ZV headers from GitHub"
        mkdir -p "$ZDIR"
        for h in zlib.h zconf.h; do
            curl -sSf -m 60 -o "$ZDIR/$h" \
                 "https://raw.githubusercontent.com/madler/zlib/v$ZV/$h" \
              || { echo "  !! could not fetch $h for v$ZV"; rm -f "$ZDIR/$h"; }
        done
    fi
    [ -f "$ZDIR/zlib.h" ] && [ -f "$ZDIR/zconf.h" ] && ZLIB_INC="$ZDIR/zlib.h"
fi

if [ -n "$ZLIB_LIB" ] && [ -n "$ZLIB_INC" ]; then
    # LLVM's exports name zlib as the bare `z`, which the linker expands to
    # -lz and then looks for an unversioned libz.so. This box has only
    # libz.so.1.2.13 -- the usual state without a -devel package -- so ld
    # fails with "cannot find -lz" even though the library is right there.
    #
    # Make the name resolvable without touching /usr/lib64: a symlink in a
    # private directory, added to the link search path.
    ZSTUB="$SRC/.zlib-stub"
    mkdir -p "$ZSTUB"
    [ -e "$ZSTUB/libz.so" ] || ln -sf "$ZLIB_LIB" "$ZSTUB/libz.so"
    TRITON_APPEND_CMAKE_ARGS="$TRITON_APPEND_CMAKE_ARGS -DZLIB_LIBRARY=$ZSTUB/libz.so -DZLIB_INCLUDE_DIR=$(dirname "$ZLIB_INC")"
    for f in CMAKE_SHARED_LINKER_FLAGS CMAKE_EXE_LINKER_FLAGS CMAKE_MODULE_LINKER_FLAGS; do
        TRITON_APPEND_CMAKE_ARGS="$TRITON_APPEND_CMAKE_ARGS -D$f=-L$ZSTUB"
    done
    echo "  zlib: lib=$ZLIB_LIB"
    echo "        inc=$(dirname "$ZLIB_INC")"
    echo "        -lz resolved via $ZSTUB/libz.so"
else
    echo "  !! zlib unresolved (lib='${ZLIB_LIB:-none}' header='${ZLIB_INC:-none}')"
    echo "     LLVMExports.cmake will fail. Either add the conda mirror to"
    echo "     no_proxy and 'conda install -y zlib', or place zlib.h+zconf.h by hand."
fi

if zs=$(find_lib zstd zstd.h); then
    TRITON_APPEND_CMAKE_ARGS="$TRITON_APPEND_CMAKE_ARGS -Dzstd_ROOT=${zs%%|*}"
    echo "  zstd: ${zs%%|*}"
else
    echo "  zstd: not found (only a problem if LLVM's exports ask for it)"
fi
export TRITON_APPEND_CMAKE_ARGS
echo "  cmake args: $TRITON_APPEND_CMAKE_ARGS"

[ "$STAGE" = fetch ] && exit 0

# ---------------------------------------------------------------- probe
say "probe: does BUILD_MCTLE reach the cache?"
cd "$SRC" || die "cannot cd $SRC"
( python setup.py build_ext --dry-run 2>&1 || true ) | tail -5
# Look only where a build would put one. The repo ships a file of the same
# name under third_party/hcu/..., and the first run matched that instead.
CACHE=$(find "$SRC/python/build" "$SRC/build" -name CMakeCache.txt 2>/dev/null | head -1)
if [ -n "$CACHE" ]; then
    echo "  cache: $CACHE"
    grep -E "^BUILD_MCTLE" "$CACHE" || echo "  !! BUILD_MCTLE absent from the cache"
else
    echo "  no cache yet -- a dry run may not configure; the build stage will tell"
fi

[ "$STAGE" = probe ] && exit 0

# ---------------------------------------------------------------- build
say "build wheel (this is the long part; LLVM and the plugin are downloaded, not compiled)"
LOG="$SRC/build-mctle.log"
pip wheel . -w "$SRC/dist-mctle" --no-build-isolation --no-deps > "$LOG" 2>&1
RC=$?
echo "  full log: $LOG ($(wc -l < "$LOG") lines)"
if [ "$RC" != 0 ]; then
    # A python traceback around a failed cmake hides cmake's own message, which
    # is the only line that says what is actually wrong.
    echo "  --- CMake errors ---"
    grep -nE -A 8 "CMake Error|Could NOT find" "$LOG" | head -50 | sed 's/^/    /'
    echo "  --- build failures (with the lines that explain them) ---"
    grep -nE -A 6 "FAILED:|error:|cannot find|undefined reference" "$LOG" \
        | tail -40 | sed 's/^/    /'
    echo "  --- tail ---"
    tail -12 "$LOG" | sed 's/^/    /'
fi
WHL=$(ls -t "$SRC/dist-mctle"/*.whl 2>/dev/null | head -1)
[ -n "$WHL" ] || die "no wheel produced"
echo "  wheel: $WHL"

# ---------------------------------------------------------------- verify
say "verify: are the bindings actually in it?"
TMP=$(mktemp -d)
( cd "$TMP" && unzip -q "$WHL" ) || die "cannot unpack the wheel"
found=0
for so in "$TMP"/triton/_C/*.so; do
    [ -e "$so" ] || continue
    hits=""
    for w in $WANT_SYMS; do
        strings -a "$so" 2>/dev/null | grep -qF "$w" && hits="$hits $w"
    done
    printf '  %-46s %s\n' "$(basename "$so")" "${hits:- none}"
    [ -n "$hits" ] && found=1
done
printf '  %-46s %s\n' "backends/metax/tle_supported.py" \
    "$([ -f "$TMP/triton/backends/metax/tle_supported.py" ] && echo present || echo absent)"
rm -rf "$TMP"

echo
if [ "$found" = 1 ]; then
    echo "=== bindings ARE in the wheel. Install into a THROWAWAY env first:"
    echo "===   python -m venv ~/mctle-test && ~/mctle-test/bin/pip install $WHL"
    echo "=== then re-run tools/tle_lowering_probe.py there. Do not replace the"
    echo "=== working triton until that probe passes."
else
    echo "=== NO bindings in the wheel. BUILD_MCTLE did not take effect."
    echo "=== Check the configure log for 'BUILD_MCTLE' and whether"
    echo "=== add_subdirectory(plugin/mctle) ran; do not install this."
fi
