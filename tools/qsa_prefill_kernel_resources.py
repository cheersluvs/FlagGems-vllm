# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 FlagOS Contributors
"""Untimed compiled-resource diagnostics; no profiling or kernel launches."""

import hashlib
import re
import shutil


def compiled_record(kernel):
    ptx = kernel.asm.get("ptx", "")
    # Counts are static PTX sites, not executed instructions or memory traffic.
    ptx = re.sub(r"/\*.*?\*/|//[^\n]*", "", ptx, flags=re.S)
    instructions = re.findall(r"^\s*(?:@!?%\w+\s+)?([a-z][\w.]*)\s", ptx, re.M)
    families = {
        "dot": ("mma.", "wgmma.mma", "tcgen05.mma."),
        "shared_load": ("ld.shared.", "ldmatrix."),
        "shared_store": ("st.shared.",),
        "global_load": ("ld.global.",),
        "global_store": ("st.global.",),
        "async_copy": ("cp.async.",),
        "shuffle": ("shfl.",),
        "barrier": ("bar.", "mbarrier."),
        "local_load": ("ld.local.",),
        "local_store": ("st.local.",),
    }
    return dict(
        name=getattr(kernel.metadata, "name", None),
        registers_per_thread=getattr(kernel, "n_regs", None),
        spill_count=getattr(kernel, "n_spills", None),
        shared_bytes=getattr(kernel.metadata, "shared", None),
        num_warps=getattr(kernel.metadata, "num_warps", None),
        ptx_sha256=hashlib.sha256(kernel.asm.get("ptx", "").encode()).hexdigest(),
        cubin_sha256=hashlib.sha256(kernel.asm.get("cubin", b"")).hexdigest(),
        ptx_static_sites={
            name: sum(op.startswith(prefixes) for op in instructions)
            for name, prefixes in families.items()
        },
    )


def libentry_records(entry):
    records = []
    for cache in entry.kernel_cache:
        for kernel, constants, *_ in cache.values():
            records.append(dict(constants=constants, **compiled_record(kernel)))
    return records


def native_records(jit):
    # FlagTree/Triton 3.6 JITFunction caches store (kernel_cache, target,
    # backend, binder). Do not launch/warmup a second reconstructed native call.
    caches = getattr(jit, "device_caches", None)
    if not isinstance(caches, dict):
        return dict(status="unavailable", reason="JIT cache interface unavailable")
    records = []
    seen = set()
    for value in caches.values():
        cache = value[0] if isinstance(value, tuple) and value else None
        if not isinstance(cache, dict):
            return dict(status="unavailable", reason="Unsupported JIT cache structure")
        for kernel in cache.values():
            if not hasattr(kernel, "asm") or not hasattr(kernel, "metadata"):
                return dict(status="unavailable", reason="Unsupported cached kernel")
            record = compiled_record(kernel)
            digest = (record["ptx_sha256"], record["cubin_sha256"])
            if digest not in seen:
                records.append(record)
                seen.add(digest)
    return dict(status="collected" if records else "not_launched", kernels=records)


def collect(native, production, candidate):
    return dict(
        phase="after_warmup_before_timing",
        ncu=shutil.which("ncu"),
        nsys=shutil.which("nsys"),
        limitations=(
            "Compiled resources and static PTX counts only; no hardware counters. "
            "Null attributes mean unavailable, never zero. Native cache discovery "
            "is optional and does not modify the reference or launch kernels."
        ),
        production=libentry_records(production._qsa_prefill_score_kernel),
        candidate_rectangular=libentry_records(candidate._qsa_prefill_score_kernel),
        candidate_compact=libentry_records(candidate._qsa_prefill_compact_score_kernel),
        native=native_records(native._qsa_mqa_paged_prefill_kernel),
    )
