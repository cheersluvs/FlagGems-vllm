# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""Pure-Python experiment geometry; also testable on a machine without a GPU."""


def positive_list(value, powers_of_two=False):
    numbers = [int(x) for x in value.split(",")]
    if not numbers or len(set(numbers)) != len(numbers) or any(x <= 0 for x in numbers):
        raise ValueError("expected distinct positive comma-separated integers")
    if powers_of_two and any(x & (x - 1) for x in numbers):
        raise ValueError("tile/warps must be powers of two")
    return numbers


def geometry(arm, n, h, hpp):
    if min(n, h, hpp) <= 0:
        raise ValueError("N/H/HPP must be positive")
    groups = (h + hpp - 1) // hpp
    if arm in ("A", "A2"):
        return ((n + 3) // 4) * (h + 1), 4, False
    if arm == "B":
        return n * groups + (n + 3) // 4, 4, False
    if arm == "B1":
        return n * groups + n, 1, False
    if arm == "C":
        return n * groups, 1, True
    raise ValueError(f"unknown arm: {arm}")


def cache_stride(bs):
    if bs <= 0:
        raise ValueError("block size must be positive")
    return ((bs * 584 + 575) // 576) * 576


def addresses(slot, bs):
    block, offset = divmod(slot, bs)
    start = block * cache_stride(bs)
    return start + offset * 576, start + bs * 576 + offset * 8
