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
# a PREBUILT LLVM 19, so LLVM is never compiled here. The metaxTritonPlugin.so
# is ALSO prebuilt by default -- but that prebuilt one cannot carry __MCTLE__
# (see "BUILD_MCTLE=ON compiles mctle in" below), so this script now compiles
# the plugin from source as well. MCTLE_PLUGIN_SRC=0 restores the download;
# MCTLE_DEFINE=0 builds without the macro, i.e. the first, broken wheel.
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

# Everything under a PERSISTENT directory. On the C550 box $HOME is not
# persistent: between sessions it lost the built wheel, the throwaway venv, the
# FlagTree clone AND the 1.35 GB of downloaded LLVM and plugin in ~/.flagtree --
# all at once, silently, which surfaced as "No such file or directory" on the
# venv's python. The repo lives under /data and survives; so does its parent.
PERSIST="${FLAGTREE_PERSIST:-$(cd "$(dirname "$0")/../.." && pwd)}"
TAG="${FLAGTREE_TAG:-0.6.1+metax3.6}"
# 0.6.1+metax3.6 is dated 2026-08-13 and #971, "[Metax][TLE] Metax TLE support
# local pointer", landed 2026-08-20 -- a week after it. Without that commit
# mctle.local_pointers exists as an op but its LLVM lowering emits llvm.bitcast
# across address spaces, which the verifier rejects with "use
# 'llvm.addrspacecast' instead". Cherry-pick it rather than moving to main,
# which is a moving target whose metax backend may expect a newer plugin.
PICK="${FLAGTREE_PICK:-52678a8b}"
SRC="${FLAGTREE_SRC:-$PERSIST/flagtree}"
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

# FlagTree caches its downloads in $HOME/.flagtree and offers no knob to move
# them. Point that name at persistent storage instead, so a lost $HOME costs a
# symlink rather than 1.35 GB of downloads.
mkdir -p "$PERSIST/.flagtree-cache"
if [ -d "$HOME/.flagtree" ] && [ ! -L "$HOME/.flagtree" ]; then
    cp -a "$HOME/.flagtree/." "$PERSIST/.flagtree-cache/" 2>/dev/null
    rm -rf "$HOME/.flagtree"
fi
ln -sfn "$PERSIST/.flagtree-cache" "$HOME/.flagtree"
VENV="${MCTLE_VENV:-$PERSIST/mctle-test}"
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

if [ -n "$PICK" ]; then
    git -C "$SRC" fetch -q origin main || die "cannot fetch main for the pick"
    for c in $PICK; do
        if git -C "$SRC" merge-base --is-ancestor "$c" HEAD 2>/dev/null; then
            echo "  $c already in $TAG"
        elif git -C "$SRC" cherry-pick -n "$c" 2>/dev/null; then
            echo "  picked $c  $(git -C "$SRC" log -1 --format=%s "$c" | cut -c1-58)"
        else
            git -C "$SRC" cherry-pick --abort 2>/dev/null
            die "cherry-pick of $c failed -- resolve by hand or set FLAGTREE_PICK="
        fi
    done
fi

# A stale build/ from another tag silently reuses the wrong CMake cache, which
# is exactly how a BUILD_MCTLE=ON run can produce a wheel without mctle.
STAMP="$SRC/.built-from"
# The macro and plugin mode are in the stamp too: tablegen output generated
# without -D__MCTLE__ would otherwise be reused, and that is the defect itself.
WANT_STAMP="$TAG${PICK:++$PICK}+def${MCTLE_DEFINE:-1}+plugsrc${MCTLE_PLUGIN_SRC:-1}"
if [ -f "$STAMP" ] && [ "$(cat "$STAMP")" != "$WANT_STAMP" ]; then
    echo "  previous build was $(cat "$STAMP"); wiping build/"
    rm -rf "$SRC/python/build" "$SRC/build"
fi
echo "$WANT_STAMP" > "$STAMP"

for f in third_party/metax/plugin/mctle/triton_mctle.cc third_party/metax/CMakeLists.txt; do
    [ -f "$SRC/$f" ] || die "$f missing at $TAG -- wrong tag"
done
grep -q BUILD_MCTLE "$SRC/third_party/metax/CMakeLists.txt" \
    || die "BUILD_MCTLE not wired at $TAG -- wrong tag"
echo "  mctle source and BUILD_MCTLE wiring both present"

export FLAGTREE_BACKEND=metax
export TRITON_APPEND_CMAKE_ARGS="-DBUILD_MCTLE=ON"

# BUILD_MCTLE=ON compiles mctle in, but #971's fixes are ALSO wrapped in
# `#ifdef __MCTLE__` -- in metax's own TritonOps.td (atomic_rmw/atomic_cas
# constraints, shared-memory effects), Dialect.h, and seven blocks of the
# plugin's LoadStoreOpToLLVM.cpp -- and NOTHING defines that macro: not a
# CMakeLists, not the tablegen rule. The first mctle wheel proved it: its
# verifier rejected tt.atomic_rmw on a !tt.ptr<i32, 3> with "ptr type matches
# value type", the #else constraint, whose getPointerTypeSameShape hardcodes
# address space 1.
#
# Two consumers need it, and they read different flags:
#   C++       CMAKE_CXX_FLAGS -- the top CMakeLists only appends to it
#   TableGen  LLVM_TABLEGEN_FLAGS -- TableGen.cmake splices it into every
#             tablegen command, and mlir-tblgen honours -D for .td #ifdef
if [ "${MCTLE_DEFINE:-1}" != 0 ]; then
    TRITON_APPEND_CMAKE_ARGS="$TRITON_APPEND_CMAKE_ARGS -DCMAKE_CXX_FLAGS=-D__MCTLE__ -DLLVM_TABLEGEN_FLAGS=-D__MCTLE__"
fi

# And the macro is useless to the PLUGIN unless the plugin is compiled here.
# By default setup downloads a prebuilt metaxTritonPlugin.so (v0.6.2) --
# LoadStoreOpToLLVM.cpp, the shared-pointer load/store/atomic lowering, lives
# in it, and whatever macros MetaX built it with are baked in. FLAGTREE_PLUGIN
# does two things and needs both spellings:
#   env    skips the prebuilt download and the copy of it into triton/_C
#   -D     makes third_party/metax/CMakeLists.txt add_subdirectory(plugin)
# The plugin links MLIRMACADialect and MLIRGPUToMACATransforms, which are not
# in the FlagTree source; they must come from the prebuilt metax LLVM.
if [ "${MCTLE_PLUGIN_SRC:-1}" != 0 ]; then
    export FLAGTREE_PLUGIN=1
    TRITON_APPEND_CMAKE_ARGS="$TRITON_APPEND_CMAKE_ARGS -DFLAGTREE_PLUGIN=ON"
    MACA_LIBS=$(find "$PERSIST/.flagtree-cache" -maxdepth 6 \
                  \( -name 'libMLIRMACADialect*' -o -name 'libMLIRGPUToMACATransforms*' \) 2>/dev/null)
    if [ -n "$MACA_LIBS" ]; then
        echo "  plugin deps found in the prebuilt LLVM:"
        printf '%s\n' "$MACA_LIBS" | sed 's/^/    /'
    else
        echo "  !! MLIRMACADialect / MLIRGPUToMACATransforms not found under"
        echo "     $PERSIST/.flagtree-cache -- a plugin source build will fail at link."
        echo "     Retry with MCTLE_PLUGIN_SRC=0 to keep the prebuilt plugin (then only"
        echo "     the libtriton half of __MCTLE__ applies)."
    fi
else
    unset FLAGTREE_PLUGIN
fi

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
# In python, not with strings+grep. This image has no unzip and very likely no
# binutils either, and `strings ... 2>/dev/null | grep -q` on a missing binary
# yields nothing -- which reads as "no bindings" rather than as "no strings".
# That silent negative already cost one round trip here.
python - "$WHL" <<'PYV'
import sys, zipfile

WANT = [
    "make_swizzled_shared_encoding_attr",   # tle.gpu.alloc, non-MMA layout
    "create_local_pointers",                # tle.gpu.local_ptr
    "create_local_alloc", "create_local_load", "create_local_store",
    "make_nv_mma_shared_encoding_attr",     # NVIDIA-only, absence expected
    "create_exclusive_cumsum",              # tle.cumsum, absent on metax
]
whl = sys.argv[1]
z = zipfile.ZipFile(whl)
sos = [n for n in z.namelist() if n.endswith(".so") and "/_C/" in n]
if not sos:
    print("  !! no .so under triton/_C in the wheel")
    sys.exit(2)
hit_any = False
for n in sos:
    data = z.read(n)
    hits = [w for w in WANT if w.encode() in data]
    print(f"  {n.split('/')[-1]:<28} {len(data)//1048576:>4} MB")
    for w in WANT:
        mark = "HIT " if w in hits else "  - "
        print(f"      {mark}{w}")
    if any(w in hits for w in ("make_swizzled_shared_encoding_attr",
                               "create_local_pointers")):
        hit_any = True
ts = any("backends/metax/tle_supported.py" in n for n in z.namelist())
print(f"  tle_supported.py present: {ts}  (absent is expected before main)")

# Did __MCTLE__ reach TableGen? The atomic_rmw/atomic_cas type constraint's
# description is compiled into the verifier's error text, one string per branch.
blob = b"".join(z.read(n) for n in sos)
new = b"value type matches ptr type" in blob     # #ifdef __MCTLE__
old = b"ptr type matches value type" in blob     # #else
print(f"  atomic constraint, __MCTLE__ branch: {'HIT' if new else 'absent'}")
print(f"  atomic constraint, #else branch:     {'HIT' if old else 'absent'}"
      "  (may remain from the non-metax TritonOps.td)")
if not new:
    print("  !! __MCTLE__ did not reach TableGen -- atomics on local_ptr will still fail")
plug = [n for n in z.namelist() if n.endswith("metaxTritonPlugin.so")]
print(f"  metaxTritonPlugin.so in wheel: {plug or 'NO (plugin compiled into libtriton?)'}")
sys.exit(0 if hit_any else 1)
PYV
VRC=$?

echo
if [ "$VRC" = 0 ]; then
    echo "=== the two bindings the operator needs ARE in the wheel."
    echo "=== Install into a THROWAWAY env, never over the working triton:"
    echo "===   python -m venv --system-site-packages $VENV"
    echo "===   $VENV/bin/pip install --no-deps --force-reinstall $WHL"
    echo "=== then, in that venv:"
    echo "===   PYTHONPATH=src:\$PYTHONPATH $VENV/bin/python tools/metax_tle_ptr_forms.py"
else
    echo "=== The bindings are NOT in the wheel (exit $VRC)."
    echo "=== Check whether add_subdirectory(plugin/mctle) ran; BUILD_MCTLE being"
    echo "=== in the cache is not the same as the target having been built."
fi
