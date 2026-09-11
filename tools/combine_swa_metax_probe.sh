#!/usr/bin/env bash
# MetaX acceptance run for PR #753 (vendor-agnostic combine_topk_swa_indices test
# and benchmark). Run through tools/vendor_probe.sh so the report is committed and
# pushed back:
#
#   tools/vendor_probe.sh tools/combine_swa_metax_probe.sh combine_swa_metax
#
# Sections are independent: a failing one is reported, the next still runs --
# except section 0, which stops everything if flaggems_vllm is not imported from
# this checkout (a run against another tree produces plausible, meaningless output).
set -uo pipefail
PY=${PY:-python}
T=tests/test_deepseek_v4_attention_combine_topk_swa_indices.py
B=benchmark/test_deepseek_v4_attention_combine_topk_swa_indices.py
section() { printf '\n%s\n=== %s\n%s\n' "$(printf '=%.0s' {1..78})" "$1" "$(printf '=%.0s' {1..78})"; }

section "0. environment (refuses to continue if flaggems_vllm is not this checkout)"
"$PY" - <<'PYEOF'
import os, sys
import torch, triton
import flaggems_vllm
root = os.path.realpath(os.getcwd())
f = os.path.realpath(flaggems_vllm.__file__)
m = getattr(torch, flaggems_vllm.device, None)
ok = m is not None and m.is_available()
print("python        ", sys.executable)
print("torch         ", torch.__version__)
print("triton        ", triton.__version__, os.path.dirname(triton.__file__))
print("flaggems_vllm ", f)
print("device/vendor ", flaggems_vllm.device, "/", flaggems_vllm.vendor_name)
print("is_available  ", ok, "| count", m.device_count() if m is not None else 0,
      "|", m.get_device_name(0) if ok else "-")
try:
    from vllm.v1.attention.ops.deepseek_v4_ops import combine_topk_swa_indices  # noqa: F401
    print("vLLM op        IMPORTABLE -> benchmark pass 1 uses the vLLM baseline")
except Exception as e:
    print(f"vLLM op        unavailable ({type(e).__name__}: {e})")
    print("               -> benchmark pass 1 already uses the torch fallback")
if not f.startswith(root + os.sep):
    print(f"!! REFUSING: flaggems_vllm comes from {f}, not from this checkout {root}")
    sys.exit(3)
PYEOF
rc=$?
if [ $rc -ne 0 ]; then echo "!! section 0 failed (rc=$rc) -- nothing else was run"; exit $rc; fi

section "1. functional tests (tests/)"
"$PY" -m pytest "$T" -v -rs -p no:cacheprovider
echo "(pytest exit $?)"

section "2. benchmark as shipped (--mode kernel)"
"$PY" -m pytest "$B" -s --mode kernel -p no:cacheprovider
echo "(pytest exit $?)"

if "$PY" -c "import vllm.v1.attention.ops.deepseek_v4_ops" >/dev/null 2>&1; then
    section "3. benchmark with the baseline FORCED to the torch fallback (--mode kernel)"
    echo "vLLM op is importable here, so pass 2 never ran the fallback the PR adds."
    PYTHONPATH="tools:${PYTHONPATH}" "$PY" -m pytest "$B" -s --mode kernel \
        -p no:cacheprovider -p combine_swa_force_torch_baseline
    echo "(pytest exit $?)"
else
    section "3. (skipped) vLLM op unavailable -- pass 2 already ran the torch fallback"
fi

section "4. on-device exactness: gems / torch fallback / vLLM vs a CPU loop oracle"
"$PY" tools/combine_swa_metax_check.py
echo "(check exit $?)"
