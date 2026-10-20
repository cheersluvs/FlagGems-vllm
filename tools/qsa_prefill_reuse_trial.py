# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 FlagOS Contributors
"""Private tuning-space experiment on the exact retained key-major JIT."""

from flaggems_vllm import runtime
from flaggems_vllm.utils import libentry, libtuner
from tools import qsa_prefill_key_major_trial as key_major


def score_entry():
    template = key_major._qsa_prefill_key_major_score_kernel.fn
    # LibTuner hashes the config list as well as JIT source, so the expanded
    # search has its own tuning cache. Never mutate the original tuner/configs.
    tuner = libtuner(
        configs=runtime.get_tuned_config("qsa_select_paged_prefill_reuse_score"),
        key=list(template.keys),
        strategy="default",
    )(template.fn)
    return libentry()(tuner)


def candidate_module(production):
    module = key_major.candidate_module(production)
    module.__name__ = production.__name__ + ".key_major_reuse_trial"
    # The cloned launcher's globals are already this module's namespace.
    module._qsa_prefill_score_kernel = score_entry()
    module.retained_baseline = key_major.candidate_module(production)
    return module
