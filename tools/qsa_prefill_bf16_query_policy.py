# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 FlagOS Contributors
"""Private fixed-chunk policy; exact original JIT for excluded BF16 tiles."""

POLICIES = ("all", "large-tile")


def parts(constants, policy, side="candidate"):
    if policy not in POLICIES or side not in ("baseline", "candidate"):
        raise ValueError("Unknown BF16 query policy or side")
    if side == "baseline" or constants["TILE_R"] < 32:
        return 1
    if policy == "all":
        return 2
    return 2 if constants["TILE_R"] == 64 and constants["K_TILES"] == 16 else 1


def pair_label(policy):
    if policy not in POLICIES:
        raise ValueError("Unknown BF16 query policy")
    return [1, 2] if policy == "all" else [1, "gated"]


class RoutedJIT:
    def __init__(self, baseline, split):
        self.baseline, self.split = baseline, split

    def __getitem__(self, grid):
        def launch(*values, **kwargs):
            mode = kwargs["QUERY_PARTS"]
            if mode not in (1, 2):
                raise RuntimeError("Invalid private query mode")
            jit = self.baseline if mode == 1 else self.split
            return jit[grid](*values, **kwargs)

        return launch


def apply(pair, policy):
    pair_label(policy)
    if policy == "all":
        return pair
    baseline, candidate = (m._qsa_prefill_score_kernel for m in pair)
    if set(baseline.configs) != set(candidate.configs):
        raise RuntimeError("Selective policy requires matching chunk configs")
    for key, config in candidate.configs.items():
        config["QUERY_PARTS"] = parts(config, policy)
        expected = dict(baseline.configs[key], QUERY_PARTS=config["QUERY_PARTS"])
        if config != expected or baseline.configs[key]["QUERY_PARTS"] != 1:
            raise RuntimeError("Selective policy changed non-mode params")
    candidate.jit = RoutedJIT(baseline.jit, candidate.jit)
    return pair


def freeze(pair, policy):
    from tools import qsa_prefill_query_split_matched as fixed

    if policy == "all":
        return fixed.freeze_and_records(pair, query_parts=(1, 2))
    pair_label(policy)
    records = [fixed.freeze_and_records((m, m))["baseline"] for m in pair]
    maps = []
    for side, values in zip(("baseline", "candidate"), records):
        mapping = {}
        for record in values:
            config = dict(record["constants"])
            if config.pop("QUERY_PARTS") != parts(record["constants"], policy, side):
                raise RuntimeError("Compiled query mode differs from selective policy")
            key = tuple(record["chunk"])
            if key in mapping:
                raise RuntimeError("Duplicate selective chunk identity")
            mapping[key] = config
        maps.append(mapping)
    if maps[0] != maps[1]:
        raise RuntimeError("Selective fixed configs differ outside query mode")
    verify_fallback(*records, policy)
    return dict(
        baseline=records[0],
        candidate=records[1],
        configs_match=True,
        query_parts_pair=pair_label(policy),
        query_policy=policy,
    )


def verify_fallback(baseline, candidate, policy):
    originals = {tuple(r["chunk"]): r for r in baseline}
    for record in candidate:
        if parts(record["constants"], policy) == 1:
            other = originals[tuple(record["chunk"])]
            for key in ("cubin_sha256", "ptx_sha256"):
                if not record.get(key) or record[key] != other.get(key):
                    raise RuntimeError(
                        "Excluded chunk differs from the original P1 binary"
                    )
