"""CPU semantic models and source-construction checks, not device validation."""

import ast
import math
import random
import struct
from pathlib import Path

from hygon_prefill_algorithms_source import final_variant
from hygon_prefill_audit_source import build, function_text
from hygon_prefill_vec_source import variants


def key(x):
    if math.isnan(x):
        raise ValueError("NaN is outside this probe contract")
    if x == 0:
        x = 0.0
    bits = struct.unpack("<I", struct.pack("<f", x))[0]
    return bits ^ (0xFFFFFFFF if bits >> 31 else 0x80000000)


def search(keys, k, quaternary):
    low, high, rounds = min(keys), max(keys), 0
    while low < high:
        span = high - low
        if quaternary:
            p1 = low + (span >> 2) + 1
            p2 = low + (span >> 1) + 1
            p3 = low + span - (span >> 2)
            c1, c2, c3 = (sum(x >= p for x in keys) for p in (p1, p2, p3))
            if c3 >= k:
                low = p3
            elif c2 >= k:
                low, high = p2, p3 - 1
            elif c1 >= k:
                low, high = p1, p2 - 1
            else:
                high = p1 - 1
        else:
            p = low + (span >> 1) + 1
            if sum(x >= p for x in keys) >= k:
                low = p
            else:
                high = p - 1
        assert high - low < span
        rounds += 1
        assert rounds <= 32
    return low


def stream(values, k, block=None, filtering=True):
    if len(values) <= k:
        return list(range(len(values))) + [-1] * (k - len(values))
    keys = [key(v) for v in values]
    maxima = None
    bound = 0
    if block:
        maxima = [max(keys[i : i + block]) for i in range(0, len(keys), block)]
        if len(maxima) >= k:
            bound = search(maxima, k, False)
    queue = [0] * k
    tau = 0
    for start in range(0, len(values), k):
        codes = []
        for i in range(start, min(start + k, len(values))):
            if block and (maxima[i // block] < bound or keys[i] < bound):
                continue
            codes.append((keys[i] << 32) | (0xFFFFFFFF - i))
        if not filtering or any(c > tau for c in codes):
            queue = sorted(queue + codes, reverse=True)[:k]
            tau = queue[-1]
    return [0xFFFFFFFF - (c & 0xFFFFFFFF) for c in queue]


def check():
    root = Path(__file__).resolve().parents[1]
    rng = random.Random(531)
    # Adjacent keys exercise quartile duplication and pivot +/-1 edges.
    samples = [
        [0, 1],
        [0, 2],
        [1, 2, 3],
        [0xFFFFFFFF] * 7,
        [0, 0xFFFFFFFF],
        [0, 0, 1, 1, 2],
    ]
    samples += [
        [rng.getrandbits(32) for _ in range(rng.randrange(1, 150))] for _ in range(80)
    ]
    for keys in samples:
        for k in range(1, len(keys) + 1):
            want = sorted(keys, reverse=True)[k - 1]
            for q in (False, True):
                assert search(keys, k, q) == want
    assert key(-0.0) == key(0.0)
    assert key(-float("inf")) > 0  # packed invalid sentinel is always smaller
    try:
        key(float("nan"))
    except ValueError:
        pass
    else:
        raise AssertionError("NaN accepted")
    arrays = [
        [],
        [float("inf"), -float("inf"), 0.0, -0.0] * 9,
        [1.0] * 129,
        list(range(129)),
        list(reversed(range(129))),
    ]
    arrays += [[round(rng.gauss(0, 1), 1) for _ in range(n)] for n in (1, 7, 33, 257)]
    for values in arrays:
        for k in (1, 2, 4, 8, 16, 32):
            for block in (None, 2, 4, 8):
                for filtering in (False, True):
                    idx = stream(values, k, block, filtering)
                    live = [i for i in idx if i >= 0]
                    assert len(live) == min(k, len(values))
                    assert len(set(live)) == len(live)
                    assert (
                        sorted(values[i] for i in live)
                        == sorted(values, reverse=True)[:k][::-1]
                    )

    # Every source family that can reach the public Hygon dispatcher must patch.
    sources = [build(root, False), build(root, True, "carry")]
    sources.extend(variants(root, True).values())
    dense_vec2 = variants(root, True)[2]
    short = dense_vec2.replace(
        "bin_idx = (mapped >> 5).to(tl.uint32)",
        "bin_idx = (mapped >> 7).to(tl.uint32)",
        1,
    )
    short = short.replace(
        "RADIX_SIZE: tl.constexpr = RADIX10_SIZE if STEP == 3 else RADIX11_SIZE",
        "RADIX_SIZE: tl.constexpr = (RADIX10_SIZE if STEP == 3 else (512 if STEP == 0 else RADIX11_SIZE))",
        1,
    )
    sources.append(short)
    for source in sources:
        original = function_text(source, "_top_k_per_row_job")
        for mode in ("network", "prefix"):
            changed = final_variant(source, mode)
            ast.parse(changed)
            assert changed != source
            assert (
                "for j in tl.range(0, final_cnt):" in changed
            )  # exact fallback retained
            # The patch must not alter histogram/refinement/launch functions.
            for name in ("_process_histogram_step", "non_tle_top_k_per_row_prefill"):
                assert function_text(source, name) == function_text(changed, name)
            assert original != function_text(changed, "_top_k_per_row_job")
    for file in Path(__file__).parent.glob("hygon_prefill_algorithms*.py"):
        compile(file.read_text(), str(file), "exec")
    print(
        "PASS: exact-key search, streaming/delegate scalar models, source patches and syntax; HIP still required."
    )


if __name__ == "__main__":
    check()
