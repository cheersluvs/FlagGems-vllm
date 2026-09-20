"""Measure the *opportunity* for wave64 histogram-atomic aggregation.

This is not a wave-aggregation implementation or a speedup measurement. It
uses the exact STEP-0 fp16 key mapping, then counts distinct bins in logical
groups corresponding to two plausible [BLOCK, VEC=4] lane mappings. The real
physical mapping and aggregation cost still require a kernel experiment.
"""

import argparse
import json
import statistics


SHAPES = ((64, 129280, 1024), (16383, 4095, 512), (4, 16385, 512))
SEEDS = (42, 43)


def emit(kind, **fields):
    print(json.dumps(dict(kind=kind, **fields), sort_keys=True), flush=True)


def step0_bins(values):
    """Torch spelling of ops.top_k_per_row_prefill._extract_bin_idx(STEP=0)."""
    import torch

    bits = values.to(torch.float16).view(torch.int16).to(torch.int32) & 0xFFFF
    mapped = torch.where((bits & 0x8000) != 0, bits, (~bits) & 0x7FFF)
    return mapped >> 5


def groups(row, mapping):
    """Yield 64 concurrent logical histogram addresses; exclude tail lanes."""
    if mapping == "flat64":
        for off in range(0, len(row), 64):
            yield row[off : off + 64]
    elif mapping == "vec4_component":
        for off in range(0, len(row), 256):
            for component in range(4):
                group = row[off + component : min(off + 256, len(row)) : 4]
                if group:
                    yield group
    else:
        raise ValueError(mapping)


def summarize(rows, mapping):
    unique = []
    lanes = []
    for row in rows:
        for group in groups(row, mapping):
            if group:
                lanes.append(len(group))
                unique.append(len(set(group)))
    total_lanes = sum(lanes)
    total_unique = sum(unique)
    ordered = sorted(unique)
    return {
        "groups": len(unique),
        "lane_updates": total_lanes,
        "ideal_group_updates": total_unique,
        "ideal_reduction": round(1 - total_unique / total_lanes, 4),
        "median_unique": statistics.median(unique),
        "p90_unique": ordered[int(0.9 * (len(ordered) - 1))],
    }


def self_test():
    assert list(groups(list(range(65)), "flat64"))[-1] == [64]
    assert [len(x) for x in groups(list(range(257)), "vec4_component")] == [64] * 4 + [1]
    assert summarize([[1] * 64], "flat64")["ideal_reduction"] == round(63 / 64, 4)
    try:
        import torch
    except ModuleNotFoundError:
        return False
    keys = step0_bins(torch.tensor([0.0, -0.0, 1.0, -1.0]))
    assert keys.tolist() == [1023, 1024, 543, 1504], keys.tolist()
    return True


def main():
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("Run on BW1000 with HIP PyTorch")
    props = torch.cuda.get_device_properties(0)
    emit("device", name=props.name, gcn_arch=getattr(props, "gcnArchName", None))
    for shape in SHAPES:
        rows, vocab, top_k = shape
        samples = min(rows, 8)
        for seed in SEEDS:
            torch.manual_seed(seed)
            normal = torch.randn((samples, vocab), device="cuda", dtype=torch.float32)
            for case in ("normal", "tied", "constant"):
                if case == "normal":
                    values = normal
                elif case == "tied":
                    values = (normal * 4).round() / 4
                else:
                    values = torch.zeros_like(normal)
                bins = step0_bins(values).cpu().tolist()
                for mapping in ("flat64", "vec4_component"):
                    emit(
                        "collision_opportunity",
                        shape=shape,
                        sampled_rows=samples,
                        seed=seed,
                        case=case,
                        mapping=mapping,
                        **summarize(bins, mapping),
                    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    key_checked = self_test()
    if args.self_test:
        emit("self_test", ok=True, key_checked=key_checked)
    else:
        main()
