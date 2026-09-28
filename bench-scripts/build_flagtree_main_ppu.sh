#!/usr/bin/env bash
# Build FlagTree main (which contains PR #1116) for the PPU backend into a NEW
# venv, then check that it is the build we think it is, then run the ZW810E
# fp8 probes on it WITHOUT the shim.
#
# Why a new venv: the whole chain of failures on this box started with one
# install over a working one (flagtree 0.7.0rc1+ppu3.6 over the vendor's
# triton 3.5.0+ppu2.0.0). Nothing here writes outside $SRC, $VENV, $BUILD_HOME
# and $LOG_DIR -- not /usr/local, not /root/wuyuqing/venv, not ~/.triton/cache.
#
# The recipe is the one FlagTree's own PPU CI runs
# (.github/workflows/ppu3.6-build-and-test.yml):
#     export FLAGTREE_BACKEND=ppu; MAX_JOBS=32 python3 -m pip install . --no-build-isolation -v
#
#     bash bench-scripts/build_flagtree_main_ppu.sh            # preflight + build + verify
#     bash bench-scripts/build_flagtree_main_ppu.sh preflight  # network/toolchain checks only
#     bash bench-scripts/build_flagtree_main_ppu.sh validate   # after a build: probes, no shim
#
# The build takes tens of minutes; run it detached so a dropped session does
# not kill it:
#     nohup bash bench-scripts/build_flagtree_main_ppu.sh > /root/ftmain.out 2>&1 &
#     tail -f /root/ftmain.out
set -uo pipefail

REPO_URL=${REPO_URL:-https://github.com/flagos-ai/FlagTree.git}
REF=${REF:-main}
FIX_COMMIT=a20284766d9a                       # merge commit of FlagTree PR #1116
SRC=${SRC:-/root/flagtree-main}
VENV=${VENV:-/root/venv-ft-main}
PY=${PY:-/usr/local/bin/python3}              # 3.12, and the site-packages with the working torch
JOBS=${JOBS:-20}                              # 22 cores; leave two for the shell
BUILD_HOME=${BUILD_HOME:-/root/.triton-build} # TRITON_HOME for the build's LLVM / nvidia downloads only
PIP_INDEX_USER=${PIP_INDEX:-}                 # empty: preflight picks the first index that really downloads
# the box's pip.conf points at aiextpypi, which returned 502 before; keep it out of every pip call
export PIP_CONFIG_FILE=/dev/null
LOG_DIR=${LOG_DIR:-/root/ftmain-logs}
HERE=$(cd "$(dirname "$0")" && pwd)
REPO=${REPO:-$(cd "$HERE/.." && pwd)}         # the FlagGems-vllm worktree the probes run against

mkdir -p "$LOG_DIR"
stage=${1:-all}
ok()   { printf '  \033[32mok\033[0m    %s\n' "$*"; }
warn() { printf '  \033[33mwarn\033[0m  %s\n' "$*"; }
die()  { printf '  \033[31mFAIL\033[0m  %s\n' "$*"; echo "[RESULT] FAILED at stage: $cur"; exit 1; }
hdr()  { cur=$1; printf '\n==== %s ====\n' "$1"; }

# reachable URL -> 0. Any HTTP answer counts (403/404 on a directory is still
# "the host answers"); only a transport failure is fatal.
reach() { curl -sS -m 25 -o /dev/null -w '%{http_code}' -I "$1" 2>/dev/null | grep -qE '^[1-5][0-9][0-9]$'; }

# ----------------------------------------------------------------------------
preflight() {
    hdr preflight
    "$PY" -c 'import sys; assert sys.version_info[:2] == (3, 12), sys.version' 2>/dev/null \
        && ok "python $("$PY" -V 2>&1) at $PY" || die "$PY is not python 3.12"
    "$PY" -c 'import torch; print(torch.__version__)' >/dev/null 2>&1 \
        && ok "torch $("$PY" -c 'import torch; print(torch.__version__)') visible to $PY (the venv inherits it)" \
        || die "no torch in $PY's site-packages; the new venv would have none"
    command -v g++ >/dev/null && ok "g++ $(g++ -dumpversion)" || die "no g++"
    command -v git >/dev/null && ok "git $(git --version | cut -d' ' -f3)" || die "no git"
    for h in /usr/include/zlib.h; do
        [ -f "$h" ] && ok "zlib headers" || warn "no $h (README asks for zlib1g-dev; LLVM links zlib)"
    done
    avail=$(df -BG --output=avail "$(dirname "$SRC")" | tail -1 | tr -dc 0-9)
    [ "${avail:-0}" -ge 40 ] && ok "${avail}G free under $(dirname "$SRC")" || die "need >= 40G free, have ${avail}G"

    echo "  -- network (every host the build touches) --"
    reach https://github.com && ok "github.com (clone, nlohmann/json, FlagPrism submodule)" \
        || die "github.com unreachable -- nothing else matters"
    local glibc suffix
    glibc=$("$PY" -c 'import platform; v=platform.libc_ver()[1].split("."); print(int(v[0])*100+int(v[1]))')
    suffix=$([ "$glibc" -gt 228 ] && echo ubuntu-x64 || echo almalinux-x64)
    LLVM_URL_BASE="https://oaitriton.blob.core.windows.net/public/llvm-builds/llvm-<hash8>-$suffix.tar.gz"
    reach https://oaitriton.blob.core.windows.net/public/llvm-builds/ \
        && ok "oaitriton.blob.core.windows.net (prebuilt LLVM, $suffix, glibc $glibc)" \
        || die "LLVM host unreachable. Workaround: fetch $LLVM_URL_BASE elsewhere, unpack it, and set LLVM_SYSPATH (+ LLVM_INCLUDE_DIRS, LLVM_LIBRARY_DIR) before the build"
    if reach https://developer.download.nvidia.com/compute/cuda/redist/; then
        ok "developer.download.nvidia.com (ptxas/cupti -- downloaded for every backend but xpu)"
        SKIP_NV=0
    else
        warn "NVIDIA redist unreachable -> will skip those downloads (TRITON_*_PATH set) and build without proton; PPU does not use them"
        SKIP_NV=1
    fi
    # An index answering is not enough: pypi.org's index answered here while its
    # file host (files.pythonhosted.org) timed out mid-download and failed the
    # first build. So download a real (small) package through each candidate
    # and keep the first one that completes. DSW runs on Aliyun, whose mirror
    # is on the internal network.
    local cands=() idx PIP_INDEX=""
    [ -n "$PIP_INDEX_USER" ] && cands+=("$PIP_INDEX_USER")
    cands+=(https://mirrors.aliyun.com/pypi/simple https://pypi.tuna.tsinghua.edu.cn/simple https://pypi.org/simple)
    for idx in "${cands[@]}"; do
        rm -rf "$LOG_DIR/piptest"
        if timeout 120 "$PY" -m pip download -q --no-deps --no-cache-dir --timeout 30 \
                -d "$LOG_DIR/piptest" -i "$idx" lit >/dev/null 2>&1; then
            PIP_INDEX=$idx; break
        fi
        warn "pip index $idx could not deliver a package"
    done
    rm -rf "$LOG_DIR/piptest"
    [ -n "$PIP_INDEX" ] && ok "pip index $PIP_INDEX (a real download completed)" \
        || die "no pip index could deliver a package; set PIP_INDEX=<reachable mirror>/simple"
    { echo "SKIP_NV=$SKIP_NV"; echo "PIP_INDEX=$PIP_INDEX"; } > "$LOG_DIR/preflight.env"
    echo "[RESULT] PREFLIGHT_OK"
}

# ----------------------------------------------------------------------------
fetch_source() {
    hdr source
    if [ -d "$SRC/.git" ]; then
        git -C "$SRC" fetch -q origin "$REF" || die "git fetch failed"
        git -C "$SRC" checkout -q FETCH_HEAD || die "checkout failed (local changes in $SRC?)"
    else
        # blob-less: full history for the ancestry check, blobs on demand
        git clone -q --filter=blob:none "$REPO_URL" "$SRC" || die "clone failed"
        git -C "$SRC" checkout -q "$REF" || die "no ref $REF"
    fi
    git -C "$SRC" merge-base --is-ancestor "$FIX_COMMIT" HEAD \
        || die "$REF @ $(git -C "$SRC" rev-parse --short HEAD) does NOT contain #1116 ($FIX_COMMIT)"
    SHA=$(git -C "$SRC" rev-parse HEAD)
    ok "$REF @ ${SHA:0:12} ($(git -C "$SRC" log -1 --format=%cs)) contains #1116"
    ok "LLVM hash $(head -c 8 "$SRC/cmake/llvm-hash.txt")"
}

# ----------------------------------------------------------------------------
build() {
    hdr venv
    if [ ! -x "$VENV/bin/python" ]; then
        # --system-site-packages: torch (and FlagGems-vllm's deps) come from the
        # working /usr/local install. pip inside the venv refuses to uninstall
        # anything outside it, so /usr/local's triton 3.5.0 survives -- it is
        # only shadowed, and verify() checks the shadowing actually happened.
        "$PY" -m venv --system-site-packages "$VENV" || die "venv creation failed"
    fi
    ok "venv $VENV"
    [ -f "$LOG_DIR/preflight.env" ] && . "$LOG_DIR/preflight.env"
    [ -n "${PIP_INDEX:-}" ] || die "no pip index recorded -- run the preflight stage first"
    "$VENV/bin/python" -m pip install -q -i "$PIP_INDEX" --timeout 60 --retries 8 \
        "setuptools>=40.8.0" wheel "cmake>=3.20,<4.0" "ninja>=1.11.1" "pybind11>=2.13.1" lit \
        || die "build requirements (python/requirements.txt) failed to install"
    ok "build requirements (from $PIP_INDEX)"

    hdr build
    local extra=()
    if [ "${SKIP_NV:-0}" = 1 ]; then
        for v in PTXAS PTXAS_BLACKWELL CUOBJDUMP NVDISASM CUDACRT CUDART CUPTI_INCLUDE CUPTI_LIB; do
            extra+=("TRITON_${v}_PATH=/nonexistent")
        done
        extra+=("TRITON_BUILD_PROTON=OFF")
        warn "NVIDIA downloads skipped, proton off"
    fi
    echo "  log: $LOG_DIR/build.log  (tens of minutes; tail -f it)"
    local t0=$SECONDS
    ( cd "$SRC" && env FLAGTREE_BACKEND=ppu MAX_JOBS="$JOBS" TRITON_HOME="$BUILD_HOME" \
        PATH="$VENV/bin:$PATH" ${extra[@]+"${extra[@]}"} \
        "$VENV/bin/python" -m pip install . --no-build-isolation -v ) > "$LOG_DIR/build.log" 2>&1
    local rc=$?
    if [ $rc -ne 0 ]; then
        echo "  --- last 30 lines of the build log ---"
        tail -30 "$LOG_DIR/build.log" | sed 's/^/    /'
        die "pip install exited $rc after $(( (SECONDS - t0) / 60 )) min"
    fi
    ok "built in $(( (SECONDS - t0) / 60 )) min"
}

# ----------------------------------------------------------------------------
verify() {
    hdr verify
    local VP="$VENV/bin/python"
    local where
    where=$("$VP" -c 'import triton, os; print(os.path.dirname(triton.__file__))' 2>&1) \
        || die "import triton fails in the new venv: $where"
    case "$where" in
        "$VENV"/*) ok "triton $("$VP" -c 'import triton; print(triton.__version__)') from $where" ;;
        *) die "import triton resolves to $where, NOT the new build (shadowing failed)" ;;
    esac
    ok "flagtree $("$VP" -m pip show flagtree 2>/dev/null | sed -n 's/^Version: //p')"
    # #1116's PPU software cast is the function this whole exercise is about
    local f
    f=$(grep -rl "def convert_custom_float8_sub89" "$where" 2>/dev/null | head -1)
    [ -n "$f" ] && ok "#1116 software cast present: ${f#$where/}" \
        || die "convert_custom_float8_sub89 not in the installed package -- this is not a #1116 build"
    local knob
    knob=$("$VP" -c 'from triton import knobs; print(knobs.language.low_precision_float)' 2>&1)
    [ "$knob" = "True" ] && ok "knobs.language.low_precision_float = True (the cap80 cast is enabled)" \
        || warn "knobs.language.low_precision_float = $knob -- set FLAGTREE_LOW_PRECISION_FLOAT=1 or the cast stays off"
    { echo "flagtree_sha=$(git -C "$SRC" rev-parse HEAD)"; echo "venv=$VENV"; echo "triton=$where"; } \
        > "$LOG_DIR/build-info.txt"
    ok "recorded $LOG_DIR/build-info.txt"
    echo "[RESULT] BUILD_VERIFIED"
}

# ----------------------------------------------------------------------------
# Probes on the new build, SDK 2.1.0 user space, WITHOUT the shim: if #1116 is
# really in, the installed compiler casts f32 -> fp8e4nv on its own and the A/B
# reports mode=native. Separate triton cache so no earlier build's binaries can
# be picked up.
validate() {
    hdr validate
    local VP="$VENV/bin/python" W="$HERE/with_sdk21"
    [ -x "$W" ] || die "no $W"
    export TRITON_CACHE_DIR=/root/.triton/cache-ftmain REPO
    local PP="$REPO/src"

    echo "  -- trivial kernel (proves this build + SDK 2.1.0 compile and load) --"
    cat > "$LOG_DIR/trivial.py" <<'EOF'
import torch, triton, triton.language as tl

@triton.jit
def k(s, d, n, B: tl.constexpr):
    o = tl.arange(0, B)
    m = o < n
    tl.store(d + o, tl.load(s + o, mask=m, other=0.0) * 2.0, mask=m)

x = torch.randn(256, device="cuda"); y = torch.empty_like(x)
k[(1,)](x, y, 256, B=256, num_warps=1); torch.cuda.synchronize()
print("triton", triton.__version__, "from", triton.__file__)
print("trivial kernel correct:", bool((y == x * 2).all()))
EOF
    "$W" env PYTHONPATH="$PP" "$VP" "$LOG_DIR/trivial.py" 2>&1 | tail -3 | sed 's/^/    /'

    echo "  -- cast pieces, PIECES_SHIM=0 (the build's own cast) --"
    "$W" env PYTHONPATH="$PP" PIECES_SHIM=0 "$VP" "$HERE/probe_ppu_cast_pieces.py" 2>&1 \
        | tee "$LOG_DIR/pieces.log" | tail -14 | sed 's/^/    /'

    echo "  -- A/B, 22 shapes, generic (native #1116) vs the removed override --"
    "$W" env PYTHONPATH="$PP" OVERRIDE_REV="${OVERRIDE_REV:-bd8eb77^}" ROUNDS="${ROUNDS:-5}" \
        "$VP" "$HERE/probe_thead_fp8_ab.py" > "$LOG_DIR/ab.log" 2>&1
    grep -E "cast:|two arms|differing|same encoding|DIFFERENT" "$LOG_DIR/ab.log" | sed 's/^/    /'
    grep -qE "cast: installed flagtree provides it" "$LOG_DIR/ab.log" \
        && ok "A/B ran on the NATIVE #1116 cast (no shim)" \
        || warn "A/B did not report the native cast -- read $LOG_DIR/ab.log before quoting anything"
    echo "  full table: $LOG_DIR/ab.log"
    tail -1 "$LOG_DIR/ab.log"
}

cur=start
case "$stage" in
    preflight) preflight ;;
    build)     preflight && fetch_source && build && verify ;;
    verify)    verify ;;
    validate)  validate ;;
    all)       preflight && fetch_source && build && verify && validate ;;
    *) echo "usage: $0 [preflight|build|verify|validate|all]"; exit 2 ;;
esac
