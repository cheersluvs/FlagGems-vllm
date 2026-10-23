# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 FlagOS Contributors
"""Test/benchmark helpers for score-cache alignment; no production import."""

import hashlib
from contextlib import ExitStack
from unittest.mock import patch

ROUND1_SOURCE_SHA256 = (
    "d345e5e0eeec5de84958f0ffe81aa2ce286ce982448f6d167f94d87e02f7c849"
)


def round1_source(source):
    """Remove only the local entry class/import/decorator and require exact pin."""
    from pathlib import Path

    from tools import qsa_prefill_triton_ab as ab

    if hashlib.sha256(source.encode()).hexdigest() == ab.INTEGRATED_PRODUCTION_SHA256:
        snapshot = Path(__file__).resolve().parents[1] / ab.PRE_STORE_PRODUCTION_PATH
        frozen = snapshot.read_text()
        if (
            hashlib.sha256(frozen.encode()).hexdigest()
            != ab.PRE_STORE_PRODUCTION_SHA256
        ):
            raise RuntimeError("Pre-store source changed")
        source = frozen
    if hashlib.sha256(source.encode()).hexdigest() == ROUND1_SOURCE_SHA256:
        return source
    try:
        if source.count("@_QSAScoreEntry\n") != 1:
            raise ValueError("unexpected score decorator count")
        text = source.replace(
            "from flaggems_vllm.utils.libentry import LibEntry\n", "", 1
        )
        begin = text.index("\n\nclass _QSAScoreEntry(")
        end = text.index("\n\n# Fixed-size", begin)
        text = (text[:begin] + text[end:]).replace(
            "@_QSAScoreEntry\n", "@libentry()\n", 1
        )
    except ValueError as error:
        raise RuntimeError("Cache alignment source normalization mismatch") from error
    if hashlib.sha256(text.encode()).hexdigest() != ROUND1_SOURCE_SHA256:
        raise RuntimeError("Code outside the operator-local cache entry changed")
    return text


def previous_module(production, kind):
    from pathlib import Path

    from tools import qsa_prefill_triton_ab as ab
    from tools import qsa_prefill_tuning_round1 as round1

    baseline = getattr(production, "production_baseline", production)
    if (
        hasattr(baseline, "__file__")
        and hashlib.sha256(Path(baseline.__file__).read_bytes()).hexdigest()
        == ab.INTEGRATED_PRODUCTION_SHA256
    ):
        production = ab.pre_store_production()
    if kind == "stage-joint":
        from tools import qsa_prefill_stage_joint_trial as joint

        return joint.candidate_module(
            getattr(production, "production_baseline", production)
        )
    if kind == "combined":
        from pathlib import Path

        from tools import qsa_prefill_loop_stage_trial as stage
        from tools import qsa_prefill_query_page_trial as combined

        module = combined.candidate_module(
            getattr(production, "production_baseline", production)
        )
        if (
            hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest()
            != stage.KERNEL_SHA256
        ):
            raise RuntimeError("Previous combined kernel source changed")
        return module
    if kind not in ("eight", "round1"):
        raise ValueError("Unknown previous tuning baseline: " + kind)
    group = round1.CONFIG_GROUP if kind == "round1" else round1.PREVIOUS_CONFIG_GROUP
    return round1.previous_module(production, rectangular_group=group)


def previous_identity(kind):
    from tools import qsa_prefill_tuning_round1 as round1

    if kind == "stage-joint":
        from tools import qsa_prefill_stage_joint_trial as joint

        return dict(
            source_equivalent_sha256=joint.KERNEL_SHA256,
            config_group=joint.CONFIG_GROUP,
            baseline_kind="joint_stage_combined_score",
        )
    if kind == "combined":
        from tools import qsa_prefill_loop_stage_trial as stage

        return dict(
            source_equivalent_sha256=stage.KERNEL_SHA256,
            config_group="qsa_select_paged_prefill_query_joint_score",
            baseline_kind=stage.BASELINE_KIND,
        )
    if kind == "round1":
        return dict(
            source_equivalent_sha256=ROUND1_SOURCE_SHA256,
            config_group=round1.CONFIG_GROUP,
            baseline_kind="previous_production_ten_configs",
        )
    if kind == "eight":
        return dict(
            source_equivalent_sha256=round1.PREVIOUS_SOURCE_SHA256,
            config_group=round1.PREVIOUS_CONFIG_GROUP,
            baseline_kind="previous_production_eight_configs",
        )
    raise ValueError("Unknown previous tuning baseline: " + kind)


def observe(module, args):
    """Capture actual compiled configs during the public call; no extra launch."""
    from tools import qsa_prefill_kernel_resources as resources
    from tools import qsa_prefill_triton_ab as ab

    traces = []

    class Capture:
        def __init__(self, entry, route):
            self.entry, self.route = entry, route

        def __getitem__(self, grid):
            launch = self.entry[grid]

            def invoke(*values, **kwargs):
                result = launch(*values, **kwargs)
                if not isinstance(result, tuple) or len(result) != 2:
                    raise RuntimeError("Score entry did not return kernel/constexprs")
                kernel, constants = result
                named = dict(zip(self.entry.arg_names, values))
                named.update(kwargs)
                traces.append(
                    dict(
                        offset=named["query_offset"],
                        rows=named["num_rows"],
                        width=named["logits_width"],
                        route=self.route,
                        requested_tuner_key_repr=repr(self.entry.fn.get_key(named)),
                        constants=dict(constants),
                        entry_count=sum(
                            len(cache) for cache in self.entry.kernel_cache
                        ),
                        compiled=resources.compiled_record(kernel),
                    )
                )
                return result

            return invoke

    with ExitStack() as stack:
        for name, route in (
            ("_qsa_prefill_score_kernel", "rectangular"),
            ("_qsa_prefill_compact_score_kernel", "compact"),
            ("_qsa_prefill_packed_score_kernel", "packed"),
        ):
            if hasattr(module, name):
                stack.enter_context(
                    patch.object(module, name, Capture(getattr(module, name), route))
                )
        chunks = ab.observe(module, "persistent_topk", args)
    if [(t["offset"], t["rows"], t["width"]) for t in traces] != ab.signatures(chunks):
        raise RuntimeError("Score trace differs from actual public chunk geometry")
    return chunks, traces
