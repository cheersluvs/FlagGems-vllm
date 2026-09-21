"""Stdlib-only parity checks for the production final-network source patch."""

import ast
import importlib.util
from pathlib import Path
from types import SimpleNamespace

from hygon_prefill_algorithms_source import final_variant
from hygon_prefill_audit_source import function_text
from hygon_prefill_vec_source import variants

ROOT = Path(__file__).resolve().parents[1]
FUSED = ROOT / "src/flaggems_vllm/runtime/backend/_hygon/fused"


def _builder():
    path = FUSED / "_top_k_per_row_prefill_final_source.py"
    spec = importlib.util.spec_from_file_location("hygon_final_source_check", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.build_final_source


def _tree(source):
    return ast.dump(ast.parse(source), include_attributes=False)


def check():
    baseline = variants(ROOT, True)[2]
    production = _builder()(baseline)
    experiment = final_variant(baseline, "network")
    prod_job = function_text(production, "_top_k_per_row_job")
    exp_job = function_text(experiment, "_top_k_per_row_job")
    assert _tree(
        prod_job.replace("_hygon_final_network", "_probe_final_network")
    ) == _tree(exp_job)
    # The audited overflow selector and unrelated passes remain unchanged.
    for name in ("_process_histogram_step", "non_tle_top_k_per_row_prefill"):
        assert _tree(function_text(production, name)) == _tree(
            function_text(baseline, name)
        )
    assert "for j in tl.range(0, final_cnt):" in prod_job

    helper = (FUSED / "_top_k_per_row_prefill_final_network.py").read_text()
    probe = (ROOT / "tools/hygon_prefill_algorithms_kernel.py").read_text()
    prod_helper = function_text(helper, "final_network")
    exp_helper = function_text(probe, "final_network")
    assert _tree(prod_helper.replace("_ordered_key", "ordered_key")) == _tree(
        exp_helper
    )
    assert _tree(function_text(helper, "_ordered_key")) == _tree(
        function_text(probe, "ordered_key").replace("ordered_key", "_ordered_key", 1)
    )

    override = (FUSED / "top_k_per_row_prefill.py").read_text()
    assert "FLAGGEMS_HYGON_TOPK_FINAL_NETWORK" in override
    assert "_dense_vec2_final" in override
    final = SimpleNamespace(HAS_TLE=False)
    scope = dict(
        torch=SimpleNamespace(float32="f32"),
        _ENABLED=True,
        DENSE_VOCAB_PER_TOPK=10,
        SHORT_BINS_TOPK=512,
        SHORT_BINS_MAX_VOCAB=1536,
        _dense_short_bins="short",
        _dense_vec2_final=final,
        _dense_vec2="vec2",
        _dense_carry="carry",
        _dense="dense",
        _sparse="sparse",
    )
    exec(function_text(override, "_select_module"), scope)
    select = scope["_select_module"]
    for rows, vocab in ((16383, 4095), (12961, 4100), (16380, 5115)):
        assert (
            select(SimpleNamespace(shape=(rows, vocab), dtype="f32"), rows, 512)
            is final
        )
    assert select(SimpleNamespace(shape=(4, 4095), dtype="f32"), 4, 512) == "vec2"
    assert (
        select(SimpleNamespace(shape=(4100, 1025), dtype="f32"), 4100, 512) == "short"
    )
    assert (
        select(SimpleNamespace(shape=(8192, 8193), dtype="f32"), 8192, 512) == "sparse"
    )
    assert (
        select(SimpleNamespace(shape=(16383, 4095), dtype="f16"), 16383, 512) == "vec2"
    )
    scope["_dense_vec2_final"] = None
    assert (
        select(SimpleNamespace(shape=(16383, 4095), dtype="f32"), 16383, 512) == "vec2"
    )
    compile(production, "<hygon-final-production-check>", "exec")
    print(
        "PASS: production selector matches audited network, overflow and other passes"
    )


if __name__ == "__main__":
    check()
