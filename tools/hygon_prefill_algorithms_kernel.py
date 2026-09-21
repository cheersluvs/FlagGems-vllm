"""Experimental exact selection kernels, never imported by production."""

import triton
import triton.language as tl


@triton.jit
def ordered_key(x):
    # Numeric equality treats both signs of zero as tied.
    x = tl.where(x == 0, 0.0, x)
    bits = x.to(tl.uint32, bitcast=True)
    return bits ^ tl.where((bits >> 31) != 0, 0xFFFFFFFF, 0x80000000).to(tl.uint32)


@triton.jit
def kth_key(keys, valid, k, QUATERNARY: tl.constexpr):
    low = tl.min(tl.where(valid, keys, 0xFFFFFFFF).to(tl.uint32), 0)
    high = tl.max(tl.where(valid, keys, 0).to(tl.uint32), 0)
    rounds = 0
    while low < high:
        # Upper midpoint, no uint32 overflow. Each update strictly shrinks.
        span = high - low
        if QUATERNARY:
            p1 = low + (span >> 2) + 1
            p2 = low + (span >> 1) + 1
            p3 = low + span - (span >> 2)
            c1 = tl.sum((valid & (keys >= p1)).to(tl.int32), 0)
            c2 = tl.sum((valid & (keys >= p2)).to(tl.int32), 0)
            c3 = tl.sum((valid & (keys >= p3)).to(tl.int32), 0)
            next_low = tl.where(
                c3 >= k, p3, tl.where(c2 >= k, p2, tl.where(c1 >= k, p1, low))
            )
            next_high = tl.where(
                c3 >= k,
                high,
                tl.where(c2 >= k, p3 - 1, tl.where(c1 >= k, p2 - 1, p1 - 1)),
            )
            low = next_low
            high = next_high
        else:
            pivot = low + (span >> 1) + 1
            count = tl.sum((valid & (keys >= pivot)).to(tl.int32), 0)
            low = tl.where(count >= k, pivot, low)
            high = tl.where(count >= k, high, pivot - 1)
        rounds += 1
    return low, rounds


@triton.jit
def emit_indices(keys, valid, threshold, count, indices, output):
    better = valid & (keys > threshold)
    equal = valid & (keys == threshold)
    needed = count - tl.sum(better.to(tl.int32), 0)
    eq_rank = tl.cumsum(equal.to(tl.int32), 0)
    take = better | (equal & (eq_rank <= needed))
    rank = tl.cumsum(take.to(tl.int32), 0) - 1
    tl.store(output + rank, indices, mask=take)


@triton.jit
def threshold_select(
    X,
    Starts,
    Ends,
    Out,
    Stats,
    Stride,
    K: tl.constexpr,
    B: tl.constexpr,
    QUATERNARY: tl.constexpr,
    DIAG: tl.constexpr,
):
    row = tl.program_id(0)
    start = tl.load(Starts + row)
    n = tl.load(Ends + row) - start
    out = Out + row.to(tl.int64) * K
    if n <= K:
        short_pos = tl.arange(0, K)
        tl.store(out + short_pos, tl.where(short_pos < n, short_pos, -1))
        if DIAG:
            tl.store(Stats + row, 0)
    else:
        search_pos = tl.arange(0, B)
        valid = search_pos < n
        x = tl.load(
            X + row.to(tl.int64) * Stride + start + search_pos,
            mask=valid,
            other=0.0,
        )
        keys = ordered_key(x)
        threshold, rounds = kth_key(keys, valid, K, QUATERNARY)
        emit_indices(keys, valid, threshold, K, search_pos, out)
        if DIAG:
            tl.store(Stats + row, rounds)


@triton.jit
def delegate_max(
    X, Starts, Ends, Maxima, Stride, GROUPS: tl.constexpr, BLOCK: tl.constexpr
):
    row = tl.program_id(0)
    group = tl.program_id(1)
    start = tl.load(Starts + row)
    n = tl.load(Ends + row) - start
    p = group * BLOCK + tl.arange(0, BLOCK)
    x = tl.load(X + row.to(tl.int64) * Stride + start + p, mask=p < n, other=0.0)
    key = tl.max(tl.where(p < n, ordered_key(x), 0).to(tl.uint32), 0)
    tl.store(Maxima + row * GROUPS + group, key)


@triton.jit
def delegate_bound(
    Maxima,
    Starts,
    Ends,
    Bounds,
    Stats,
    K: tl.constexpr,
    GROUPS: tl.constexpr,
    BLOCK: tl.constexpr,
    B: tl.constexpr,
    DIAG: tl.constexpr,
):
    row = tl.program_id(0)
    n = tl.load(Ends + row) - tl.load(Starts + row)
    count = tl.cdiv(n, BLOCK)
    p = tl.arange(0, B)
    keys = tl.load(Maxima + row * GROUPS + p, mask=p < count, other=0)
    bound = tl.full((), 0, tl.uint32)
    rounds = 0
    if count >= K:
        bound, rounds = kth_key(keys, p < count, K, False)
    tl.store(Bounds + row, bound)
    if DIAG:
        survivors = tl.sum(((p < count) & (keys >= bound)).to(tl.int32), 0)
        tl.store(Stats + row * 3, count)
        tl.store(Stats + row * 3 + 1, survivors)
        tl.store(Stats + row * 3 + 2, rounds)


@triton.jit
def streaming_select(
    X,
    Starts,
    Ends,
    Out,
    Maxima,
    Bounds,
    Stats,
    Stride,
    K: tl.constexpr,
    FILTER: tl.constexpr,
    DELEGATE: tl.constexpr,
    GROUPS: tl.constexpr,
    BLOCK: tl.constexpr,
    DIAG: tl.constexpr,
):
    row = tl.program_id(0)
    start = tl.load(Starts + row)
    n = tl.load(Ends + row) - start
    out = Out + row.to(tl.int64) * K
    if n <= K:
        p = tl.arange(0, K)
        tl.store(out + p, tl.where(p < n, p, -1))
        if DIAG:
            tl.store(Stats + row, 0)
    else:
        lane = tl.arange(0, 2 * K)
        queue = tl.full((2 * K,), 0, tl.uint64)
        tau = tl.full((), 0, tl.uint64)
        merges = 0
        bound = tl.full((), 0, tl.uint32)
        if DELEGATE:
            bound = tl.load(Bounds + row)
        for tile in tl.range(0, tl.cdiv(n, K)):
            p = tile * K + lane - K
            valid = (lane >= K) & (p < n)
            if DELEGATE:
                group_max = tl.load(
                    Maxima + row * GROUPS + p // BLOCK, mask=valid, other=0
                )
                valid = valid & (group_max >= bound)
            if tl.sum(valid.to(tl.int32), 0) > 0:
                x = tl.load(
                    X + row.to(tl.int64) * Stride + start + p, mask=valid, other=0.0
                )
                key = ordered_key(x)
                if DELEGATE:
                    valid = valid & (key >= bound)
                # Larger packed keys win, lower relative indices break ties.
                code = (key.to(tl.uint64) << 32) | (0xFFFFFFFF - p.to(tl.uint32)).to(
                    tl.uint64
                )
                code = tl.where(valid, code, 0).to(tl.uint64)
                should_merge = tl.full((), True, tl.int1)
                if FILTER:
                    should_merge = tl.sum((valid & (code > tau)).to(tl.int32), 0) > 0
                if should_merge:
                    queue = tl.sort(tl.where(lane < K, queue, code), descending=True)
                    tau = tl.min(
                        tl.where(lane < K, queue, 0xFFFFFFFFFFFFFFFF).to(tl.uint64), 0
                    )
                    merges += 1
        pos = (0xFFFFFFFF - (queue & 0xFFFFFFFF).to(tl.uint32)).to(tl.int32)
        tl.store(out + lane, pos, mask=lane < K)
        if DIAG:
            tl.store(Stats + row, merges)


@triton.jit
def final_network(Values, Indices, Output, count, base, remain, CAP: tl.constexpr):
    p = tl.arange(0, CAP)
    valid = p < count
    x = tl.load(Values + p, mask=valid, other=0.0)
    # Match generic ranking: later candidate position wins equal-value ties.
    code = (ordered_key(x).to(tl.uint64) << 32) | p.to(tl.uint64)
    code = tl.where(valid, code, 0).to(tl.uint64)
    if remain == count:
        direct_idx = tl.load(Indices + p, mask=valid, other=0)
        tl.store(Output + base + p, direct_idx, mask=valid)
    elif remain == 1:
        best = tl.max(code, 0)
        best_pos = (best & 0xFFFFFFFF).to(tl.int32)
        best_idx = tl.load(Indices + best_pos)
        tl.store(Output + base, best_idx)
    elif remain > 0:
        ranked = tl.sort(code, descending=True)
        sorted_pos = (ranked & 0xFFFFFFFF).to(tl.int32)
        sorted_idx = tl.load(Indices + sorted_pos, mask=p < remain, other=0)
        tl.store(Output + base + p, sorted_idx, mask=p < remain)


@triton.jit
def final_prefix(Values, Indices, Output, count, base, remain, CAP: tl.constexpr):
    p = tl.arange(0, CAP)
    valid = p < count
    x = tl.load(Values + p, mask=valid, other=0.0)
    keys = ordered_key(x)
    if remain == count:
        idx = tl.load(Indices + p, mask=valid, other=0)
        tl.store(Output + base + p, idx, mask=valid)
    elif remain > 0:
        threshold, rounds = kth_key(keys, valid, remain, False)
        better = valid & (keys > threshold)
        equal = valid & (keys == threshold)
        needed = remain - tl.sum(better.to(tl.int32), 0)
        # Reverse scan preserves the original later-position tie rule.
        eq_rank = tl.cumsum(equal.to(tl.int32), 0, reverse=True)
        take = better | (equal & (eq_rank <= needed))
        rank = tl.cumsum(take.to(tl.int32), 0) - 1
        idx = tl.load(Indices + p, mask=take, other=0)
        tl.store(Output + base + rank, idx, mask=take)
