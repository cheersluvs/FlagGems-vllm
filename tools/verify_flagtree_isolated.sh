#!/usr/bin/env bash
# Verify the FlagTree 0.6.1 (Triton 3.5.1) wheel in a FULLY isolated venv --
# its own torch and torch_npu too, not the system ones.
#
#   tools/verify_flagtree_isolated.sh ~/flagtree-build/wheels/flagtree-*.whl [torch_ver]
#
# Why a torch of its own: torch 2.6.0 needs AttrsDescriptor, which Triton 3.5.1
# removed.  torch's _inductor/runtime/hints.py falls back from
# triton.backends.compiler to triton.compiler.compiler, and that second import
# re-enters triton mid-initialisation -- the "circular import" everything
# reported is only the symptom.  Triton 3.5 pairs with torch 2.9.
#
# This matters because the gpu TLE surface the generic operator wants
# (tle.gpu.alloc / local_ptr / smem) exists ONLY on the 3.5 line: 0.6.0+ascend3.2
# still ships nothing but dsa.  So torch cannot stay at 2.6 and get that surface.
#
# The system install is never touched: it stays the known-good environment where
# 25 tests pass, and every result here is reproducible against it.
set -uo pipefail

WHL=${1:?usage: tools/verify_flagtree_isolated.sh <wheel> [torch_version]}
TORCH_VER=${2:-2.9.1}
NPU_VER=${3:-$TORCH_VER}
VENV=${FT_VENV:-$HOME/ft-isolated}
ROOT=$(cd "$(dirname "$0")/.." && pwd)
OUT=${FT_REPORTS:-$HOME/ft-reports}/$(basename "$WHL" .whl)
mkdir -p "$OUT"

if [ -f /usr/local/Ascend/cann/set_env.sh ]; then
    # shellcheck disable=SC1091
    source /usr/local/Ascend/cann/set_env.sh
fi
# Domestic indexes answer only without the proxy on this box.
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY all_proxy ALL_PROXY || true

echo "=== building an isolated venv at $VENV (torch $TORCH_VER, torch_npu $NPU_VER)"
rm -rf "$VENV"
python -m venv "$VENV"          # NO --system-site-packages, on purpose
PY="$VENV/bin/python"
"$VENV/bin/pip" install -q --upgrade pip setuptools wheel || exit 1
echo "=== installing torch"
"$VENV/bin/pip" install -q "torch==${TORCH_VER}" || { echo "!! torch install failed"; exit 1; }
echo "=== installing torch_npu"
"$VENV/bin/pip" install -q "torch_npu==${NPU_VER}" || { echo "!! torch_npu install failed"; exit 1; }
echo "=== installing the flagtree wheel"
"$VENV/bin/pip" install -q "$WHL" || { echo "!! flagtree install failed"; exit 1; }

echo
echo "=== versions:"
"$PY" - <<'PYEOF'
import importlib.metadata as m
for n in ("torch", "torch_npu", "flagtree"):
    try:
        print(f"  {n:<10} {m.version(n)}")
    except Exception as e:
        print(f"  {n:<10} !! {e}")
PYEOF

echo
echo "=== does it import at all?  (this is what failed on torch 2.6)"
"$PY" -c "
import torch, torch_npu, triton, triton.language as tl
print('  torch', torch.__version__, '| triton', triton.__version__)
print('  npu available:', torch.npu.is_available())
" 2>&1 | tail -20 || { echo "!! still cannot import -- stopping here"; exit 1; }

export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"

echo
echo "=== A. TLE surface -- has gpu.alloc/local_ptr/smem/cumsum appeared?"
"$PY" "$ROOT/tools/ascend_tle_surface.py" 2>&1 | tee "$OUT/tle_surface.txt" | \
    grep -E "gpu|cumsum|local_ptr|smem|has_triton_tle|tle_enabled|HAS_TLE|triton " | head -25

echo
echo "=== B. the eight workarounds -- which are now unnecessary?"
DSA_CASE_SCRIPT="$ROOT/tools/ascend_defect_case.py" \
    "$PY" "$ROOT/tools/ascend_dsa_run.py" \
    reduce_or assume uint16_shift atomic_unique dup_store row_mask \
    where_in_loop reshape_2d 2>&1 | tee "$OUT/defects.txt" | grep -E "^RESULT|TIMED"

echo
echo "=== C. UB surface and tl.histogram"
"$PY" "$ROOT/tools/ascend_dsa_run.py" \
    copy ptr_store atomic2 hist hist_accum 2>&1 | tee "$OUT/dsa.txt" | grep -E "^RESULT|TIMED"

echo
echo "=== reports in $OUT  (outside the repo, so the work tree stays clean)"
echo "=== the system toolchain was NOT touched"
