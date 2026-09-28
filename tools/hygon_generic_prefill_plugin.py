"""pytest plugin: run top_k_per_row_prefill through the GENERIC operator.

The Hygon override has no run-time switch, so a "before" measurement swaps the
package's top-level entry back to flaggems_vllm.ops.top_k_per_row_prefill at
configure time -- the benchmark reads flaggems_vllm.top_k_per_row_prefill when
its test runs, after this. Load with `-p hygon_generic_prefill_plugin` and
tools/ on PYTHONPATH.
"""

from importlib import import_module


def pytest_configure(config):
    import flaggems_vllm

    generic = import_module("flaggems_vllm.ops.top_k_per_row_prefill")
    flaggems_vllm.top_k_per_row_prefill = generic.top_k_per_row_prefill
