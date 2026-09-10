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
export MAX_JOBS="${MAX_JOBS:-$(nproc)}"
echo "  FLAGTREE_BACKEND=$FLAGTREE_BACKEND"
echo "  TRITON_APPEND_CMAKE_ARGS=$TRITON_APPEND_CMAKE_ARGS"
echo "  MAX_JOBS=$MAX_JOBS"

[ "$STAGE" = fetch ] && exit 0

# ---------------------------------------------------------------- probe
say "probe: does BUILD_MCTLE reach the cache?"
cd "$SRC" || die "cannot cd $SRC"
( python setup.py build_ext --dry-run 2>&1 || true ) | tail -5
CACHE=$(find "$SRC" -name CMakeCache.txt -newermt '-1 hour' 2>/dev/null | head -1)
if [ -n "$CACHE" ]; then
    echo "  cache: $CACHE"
    grep -E "^BUILD_MCTLE" "$CACHE" || echo "  !! BUILD_MCTLE absent from the cache"
else
    echo "  no cache yet -- a dry run may not configure; the build stage will tell"
fi

[ "$STAGE" = probe ] && exit 0

# ---------------------------------------------------------------- build
say "build wheel (this is the long part; LLVM and the plugin are downloaded, not compiled)"
pip wheel . -w "$SRC/dist-mctle" --no-build-isolation --no-deps 2>&1 | tail -25
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
