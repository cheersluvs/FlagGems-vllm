#!/usr/bin/env bash
# The MTT prefill override with one environment switch, FLAGGEMS_MTT_TOPK_PREFILL,
# in place of FLAGGEMS_MTT_PREFILL_SSTRIDE / _MIN_SPAN / _WIDE_MAX_ROWS, on
# main + that change + the new top_k_per_row tests. Run from this worktree:
#
#     bash tools/mtt_switch_check.sh 2>&1 | tee /tmp/mtt_switch_check.txt
#
# 1. The switch: default (override, sampled + wide block), =0 (generic op
#    called), and the removed variables set to garbage (must be ignored).
# 2. Both test suites, and the narrow-band cases.
# 3. Both benchmarks, kernel mode. Decode is the harness control (MTT has no
#    decode override); compare the prefill FlagGems column with the table the
#    fix was submitted with: 0.0296 0.0374 0.1553 0.3873 1.7776 2.4732 2.3316.
set -u
ROOT=$(git rev-parse --show-toplevel)
cd "$ROOT"
export VLLM_PLUGINS=musa
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
unset FLAGGEMS_MTT_TOPK_PREFILL FLAGGEMS_MTT_PREFILL_SSTRIDE \
  FLAGGEMS_MTT_PREFILL_MIN_SPAN FLAGGEMS_MTT_PREFILL_WIDE_MAX_ROWS

echo "### $(git log --oneline -1)  host $(hostname)  $(date -Iseconds)"
case "$(python -c 'import flaggems_vllm; print(flaggems_vllm.__file__)')" in
  "$ROOT"/*) ;;
  *) echo "!!! flaggems_vllm is NOT imported from this worktree"; exit 1 ;;
esac

CHILD=$(cat <<'EOF'
import torch
from importlib import import_module
import flaggems_vllm
mtt = import_module("flaggems_vllm.runtime.backend._mthreads.fused.top_k_per_row_prefill")
calls = []
orig = mtt._generic_prefill
def spy(*a, **k):
    calls.append(1)
    return orig(*a, **k)
mtt._generic_prefill = spy
dev = flaggems_vllm.device
print("  _ENABLED", mtt._ENABLED, " SSTRIDE", mtt.SSTRIDE, " MIN_SPAN", mtt.MIN_SPAN,
      " wide_max_rows", mtt._wide_max_rows(), " HAS_TLE", mtt.HAS_TLE)
for rows, vocab, k in ((64, 129280, 1024), (4, 8193, 512), (4100, 1025, 512)):
    torch.manual_seed(1)
    x = torch.randn(rows, vocab, device=dev)
    st = torch.zeros(rows, dtype=torch.int32, device=dev)
    en = torch.full((rows,), vocab, dtype=torch.int32, device=dev)
    out = torch.full((rows, k), -1, dtype=torch.int32, device=dev)
    n0 = len(calls)
    mtt.top_k_per_row_prefill(x, st, en, out, rows, x.stride(0), 1, k)
    ref = torch.topk(x, k, dim=1).values
    got = torch.gather(x, 1, out.long().clamp(min=0)).sort(dim=1, descending=True)[0]
    ok = float((got - ref).abs().max()) == 0.0 and bool((out >= 0).all())
    print(f"  {rows}x{vocab} k{k}: {'ok' if ok else 'WRONG'}  generic called: {len(calls) > n0}")
EOF
)

echo; echo "### 1. the switch"
echo "[default]"; python -c "$CHILD" 2>&1 | grep -v "Platform plugin"
echo "[FLAGGEMS_MTT_TOPK_PREFILL=0]"
FLAGGEMS_MTT_TOPK_PREFILL=0 python -c "$CHILD" 2>&1 | grep -v "Platform plugin"
echo "[removed variables set to garbage]"
FLAGGEMS_MTT_PREFILL_SSTRIDE=x FLAGGEMS_MTT_PREFILL_MIN_SPAN=? \
  FLAGGEMS_MTT_PREFILL_WIDE_MAX_ROWS=many python -c "$CHILD" 2>&1 | grep -v "Platform plugin"

echo; echo "### 2. tests"
python -m pytest tests/test_top_k_per_row_prefill.py tests/test_top_k_per_row_decode.py \
  -q -p no:cacheprovider -rs 2>&1 | grep -E "passed|failed|SKIPPED|FAILED|Error" | tail -20
python -m pytest tests/test_top_k_per_row_prefill.py -k narrow_band -q -p no:cacheprovider 2>&1 \
  | grep -E "passed|failed|FAILED"

echo; echo "### 3. benchmarks, kernel mode"
for op in decode prefill; do
  echo "[$op]"
  python -m pytest "benchmark/test_top_k_per_row_$op.py" -s -q --mode kernel \
    -p no:cacheprovider 2>&1 | grep -E "SUCCESS" | sed -E 's/^ *//' | cut -c1-120
done
