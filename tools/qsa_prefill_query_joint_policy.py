# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 FlagOS Contributors
"""CPU-safe dtype pruning for the explicit YAML joint query-mode choices."""

CONFIG_GROUP = "qsa_select_paged_prefill_query_joint_score"


def prune_configs(configs, named_args, **kwargs):
    args = {**named_args, **kwargs}
    fp8 = bool(args["IS_FP8"])
    return [
        c
        for c in configs
        if c.kwargs["QUERY_PARTS"] == 1
        or (fp8 and c.kwargs["QUERY_PARTS"] == 2 and c.kwargs["TILE_R"] >= 32)
    ]


def validate_choices(traces, fp8):
    for trace in traces:
        if trace["route"] != "rectangular":
            continue
        config = trace["constants"]
        parts = config["QUERY_PARTS"]
        if parts not in (1, 2) or (parts == 2 and (not fp8 or config["TILE_R"] < 32)):
            raise RuntimeError("Joint tuner selected an ineligible query mode")


def same_tiles(left, right):
    def normalized(traces):
        return [
            (
                t["rows"],
                t["offset"],
                t["width"],
                t["route"],
                {k: v for k, v in t["constants"].items() if k != "QUERY_PARTS"},
            )
            for t in traces
        ]

    return normalized(left) == normalized(right)
