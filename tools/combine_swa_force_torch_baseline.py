"""pytest plugin: force the combine_topk_swa_indices benchmark onto its torch
fallback baseline, so the fallback path is exercised on a box where the vLLM op
IS importable (there the benchmark would otherwise never run it).

    PYTHONPATH=tools:$PYTHONPATH pytest -p combine_swa_force_torch_baseline <bench file>

The baseline is chosen inside the test, from a module global, so flipping that
global after collection is enough. Fails loudly if the module is not found,
rather than silently benchmarking the vLLM baseline under a misleading label.
"""
import sys

_SUFFIX = "test_deepseek_v4_attention_combine_topk_swa_indices"
_FLAG = "_HAS_VLLM_COMBINE_TOPK_SWA_INDICES"


def pytest_collection_modifyitems(session, config, items):
    hit = False
    for name, mod in list(sys.modules.items()):
        if name.endswith(_SUFFIX) and hasattr(mod, _FLAG):
            setattr(mod, _FLAG, False)
            hit = True
            print(f"\n[force-torch-baseline] {name}: baseline forced to the torch fallback")
    if not hit:
        raise RuntimeError(
            "[force-torch-baseline] benchmark module not found -- nothing was forced"
        )
