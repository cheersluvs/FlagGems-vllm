#!/usr/bin/env bash
# Verify a freshly built FlagTree wheel in a throwaway venv.  Nothing here can
# touch the installed toolchain: the venv is created with --system-site-packages
# so torch/torch_npu are reused, but pip writes only inside the venv, whose
# site-packages precedes the system one.
#
#   tools/verify_flagtree_wheel.sh ~/flagtree-build/wheels/flagtree-*.whl
#
# The question is not "does it run" -- it is which of the workarounds in
# runtime/backend/_ascend/fused/ are now dead weight, and whether the gpu.*
# TLE surface the generic operator wants has appeared.
set -uo pipefail

WHL=${1:?usage: tools/verify_flagtree_wheel.sh <wheel>}
VENV=${FT_VENV:-$HOME/ft-verify}
ROOT=$(cd "$(dirname "$0")/.." && pwd)
OUT="$ROOT/reports/verify_$(basename "$WHL" .whl)"
mkdir -p "$OUT"

if [ -f /usr/local/Ascend/cann/set_env.sh ]; then
    # shellcheck disable=SC1091
    source /usr/local/Ascend/cann/set_env.sh
fi

echo "=== creating $VENV"
rm -rf "$VENV"
python -m venv --system-site-packages "$VENV"
# The domestic index answers only without the proxy on this box.
env -u http_proxy -u https_proxy -u HTTP_PROXY -u HTTPS_PROXY \
    "$VENV/bin/pip" install -q "$WHL" || { echo "!! install failed"; exit 1; }

# torch's automatic backend loading makes FlagTree 0.6.1's import graph
# circular: triton -> torch -> torch_npu -> triton, which fails halfway through
# triton.backends.  The probes import torch_npu explicitly instead.
export TORCH_DEVICE_BACKEND_AUTOLOAD=0
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
PY="$VENV/bin/python"

echo "=== installed:"
"$PY" -c "import triton, importlib.metadata as m; print('  triton', triton.__version__); print('  flagtree', m.version('flagtree'))"

echo
echo "=== A. TLE surface -- has gpu.alloc/local_ptr/smem/cumsum appeared?"
"$PY" "$ROOT/tools/ascend_tle_surface.py" 2>&1 | tee "$OUT/tle_surface.txt" | \
    grep -E "^  (triton|flagtree)|gpu|cumsum|has_triton_tle|tle_enabled|HAS_TLE" | head -20

echo
echo "=== B. the eight workarounds -- which are now unnecessary?"
DSA_CASE_SCRIPT="$ROOT/tools/ascend_defect_case.py" \
    "$PY" "$ROOT/tools/ascend_dsa_run.py" \
    reduce_or assume uint16_shift atomic_unique dup_store row_mask \
    where_in_loop reshape_2d 2>&1 | tee "$OUT/defects.txt" | grep -E "^RESULT|^--- .*(exit|TIMED)"

echo
echo "=== C. DSA/UB surface, unchanged cases"
"$PY" "$ROOT/tools/ascend_dsa_run.py" \
    copy ptr_store atomic2 hist hist_accum 2>&1 | tee "$OUT/dsa.txt" | \
    grep -E "^RESULT|^--- .*(exit|TIMED)"

echo
echo "=== reports in $OUT"
echo "=== the installed toolchain was NOT modified; run the test suite with"
echo "===   PYTHONPATH=src:\$PYTHONPATH $PY -m pytest tests/test_top_k_per_row_*.py -q"
