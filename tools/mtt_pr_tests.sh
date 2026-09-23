#!/usr/bin/env bash
# The top_k_per_row test files as they will stand once both open PRs merge,
# on a Moore Threads card. Run from the root of this worktree:
#
#     bash tools/mtt_pr_tests.sh 2>&1 | tee /tmp/mtt_pr_tests.txt
#
# 1. This tree: main + the MTT retry fix + the new test files. Both suites.
# 2. The narrow-band cases one by one.
# 3. Control: the same tests against main's MTT override (no fix), in an
#    exported copy of this tree with that one file swapped -- a whole tree,
#    because pytest.ini's `pythonpath = src` puts the rootdir's src/ first.
#    Narrow-band cases are EXPECTED to fail here; that is what the test is for.
set -u
ROOT=$(git rev-parse --show-toplevel)
cd "$ROOT"
MAIN=5f4dc53cc31b99e83967a2b19195ab1a295aeef6
MTT=src/flaggems_vllm/runtime/backend/_mthreads/fused/top_k_per_row_prefill.py
export VLLM_PLUGINS=musa
ORIG_PP="${PYTHONPATH:-}"
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"

echo "### $(git log --oneline -1)  host $(hostname)  $(date -Iseconds)"
python - <<'EOF'
import importlib, torch
m = importlib.import_module("flaggems_vllm.runtime.backend._mthreads.fused.top_k_per_row_prefill")
print("mtt override :", m.__file__)
print("HAS_TLE      :", m.HAS_TLE, " can_sample(64,129280,1,1024):", m._can_sample(64, 129280, 1, 1024))
try:
    import vllm._custom_ops  # noqa: F401
    print("vLLM op      :", hasattr(torch.ops._C, "top_k_per_row_prefill"))
except Exception as e:
    print("vLLM op      : import failed:", repr(e)[:120])
EOF
case "$(python -c 'import flaggems_vllm; print(flaggems_vllm.__file__)')" in
  "$ROOT"/*) ;;
  *) echo "!!! flaggems_vllm is NOT imported from this worktree (an editable install shadows it?)"; exit 1 ;;
esac

echo; echo "### 1. both suites, this tree"
python -m pytest tests/test_top_k_per_row_prefill.py tests/test_top_k_per_row_decode.py \
  -q -p no:cacheprovider -rs 2>&1 | grep -E "passed|failed|SKIPPED|FAILED|Error" | tail -20

echo; echo "### 2. narrow band, this tree"
python -m pytest tests/test_top_k_per_row_prefill.py -k narrow_band -v -p no:cacheprovider 2>&1 \
  | grep -E "narrow_band|passed|failed"

echo; echo "### 3. control: main's MTT override, no fix"
CTRL=$(mktemp -d)
git archive HEAD | tar -x -C "$CTRL"
git show "$MAIN:$MTT" > "$CTRL/$MTT"
cmp -s "$CTRL/$MTT" "$MTT" && echo "!!! control file equals the fixed one"
cd "$CTRL"
PYTHONPATH="$CTRL/src${ORIG_PP:+:$ORIG_PP}" python -c \
  "import importlib; print('mtt override :', importlib.import_module('flaggems_vllm.runtime.backend._mthreads.fused.top_k_per_row_prefill').__file__)"
PYTHONPATH="$CTRL/src${ORIG_PP:+:$ORIG_PP}" python -m pytest tests/test_top_k_per_row_prefill.py \
  -k narrow_band -q -p no:cacheprovider 2>&1 | grep -E "^FAILED|passed|failed" | sed 's/ - .*//'
cd "$ROOT"
rm -rf "$CTRL"
