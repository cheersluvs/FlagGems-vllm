# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 FlagOS Contributors
"""Replay a captured failing SM90 kernel without vLLM or FlagGems.

The captured PTX/control experiments are GPU-verified; this standalone
packaging still needs a card recheck. Run each variant in a fresh process.
"""

import argparse
import hashlib
import json
import re
import subprocess
import tempfile
from pathlib import Path

import torch
import triton
import triton.language as tl
from triton.backends.nvidia.compiler import get_ptxas
from triton.compiler.compiler import ASTSource, CompiledKernel

HERE = Path(__file__).resolve().parent


@triton.jit
def abi_stub(
    q_ptr,
    k_cache_ptr,
    page_table_ptr,
    query_start_loc_ptr,
    visible_blocks_ptr,
    logits_ptr,
    stride_q_row,
    stride_q_head,
    stride_cache_block,
    stride_cache_token,
    stride_table_req,
    stride_logits_row,
    num_rows,
    query_offset,
    page_table_width,
    PAGE_SIZE: tl.constexpr,
    NUM_HEADS: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    TILE_R: tl.constexpr,
    BLOCK_N: tl.constexpr,
    K_TILES: tl.constexpr,
    STAGES: tl.constexpr,
):
    # Never compiled/launched. Only its argument names describe the saved ABI.
    pass


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def transform(text, variant):
    if variant == "original":
        return text, 0
    if variant == "no-hint":

        def replace(match):
            operands, policy = match[3].rsplit(",", 1)
            assert re.fullmatch(r"\s*%rd\d+\s*", policy)
            assert operands.count(",") == 3
            return match[1] + match[2] + " " + operands + ";"

        result, count = re.subn(
            r"^(\s*)(cp\.async\.cg\.shared\.global)\.L2::cache_hint\s+(.+);[ \t]*$",
            replace,
            text,
            flags=re.MULTILINE,
        )
        assert count == 4
    else:
        result, count = re.subn(
            r"(createpolicy\.fractional\.L2::evict_first\.b64\s+%rd\d+,\s*)1\.0;",
            r"\g<1>0.5;",
            text,
        )
        assert count == 2
    for op in ("cp.async.commit_group", "cp.async.wait_group", "wgmma."):
        assert text.count(op) == result.count(op)
    return result, count


def inputs():
    # Exact H1/D32 input recipe from the original GPU-verified case.
    torch.manual_seed(20261013)
    q = torch.randn((10, 1, 32), device="cuda", dtype=torch.bfloat16)
    cache = torch.randn((162, 16, 1, 32), device="cuda", dtype=torch.bfloat16)
    pages = torch.randperm(162, device="cuda").to(torch.int32)
    table = torch.full((2, 97), -1, device="cuda", dtype=torch.int32)
    table[0, :65] = pages[:65]
    table[1, :97] = pages[65:]
    starts = torch.tensor([19, 22, 29], device="cuda", dtype=torch.int32)
    counts = [
        (length * 4 - rows + row + 1) // 4
        for rows, length in ((3, 1025), (7, 1537))
        for row in range(rows)
    ]
    visible = torch.tensor(counts, device="cuda", dtype=torch.int32)
    logits = torch.full((10, 1552), -torch.inf, device="cuda", dtype=torch.float32)
    expected = torch.full_like(logits, -torch.inf)
    for request, (begin, end, count) in enumerate(((0, 3, 1025), (3, 10, 1537))):
        columns = torch.arange(count, device="cuda")
        physical = table[request, columns // 16].long()
        keys = cache[physical, columns % 16, 0].float()
        expected[begin:end, :count] = (q[begin:end, 0].float() @ keys.T).clamp_min(0)
    runtime = (
        q,
        cache,
        table,
        starts,
        visible,
        logits,
        q.stride(0),
        q.stride(1),
        cache.stride(0),
        cache.stride(1),
        table.stride(0),
        logits.stride(0),
        10,
        0,
        97,
    )
    return runtime, expected


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--variant",
        choices=("original", "no-hint", "fraction-half"),
        default="original",
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--verify-inputs", type=Path, help="optional original card h1d32-input.pt"
    )
    args = parser.parse_args()
    folder = args.output or Path(tempfile.mkdtemp(prefix="flagtree-cp-async-replay."))
    folder.mkdir(parents=True, exist_ok=True)
    print("Reports:", folder, flush=True)
    print("torch:", torch.__version__, torch.__file__)
    print("triton:", triton.__version__, triton.__file__, flush=True)
    assert torch.cuda.is_available() and torch.cuda.get_device_capability() == (9, 0)
    torch.backends.cuda.matmul.allow_tf32 = False
    info = json.loads((HERE / "captured.json").read_text())
    assert digest(HERE / "captured.ptx") == info["ptx_sha256"]
    assert abi_stub.arg_names == info["arg_names"]
    text, edits = transform((HERE / "captured.ptx").read_text(), args.variant)
    (folder / "kernel.ptx").write_text(text)
    (folder / "kernel.json").write_text(json.dumps(info["metadata"]) + "\n")
    assembler = get_ptxas(90)
    assembler = str(getattr(assembler, "path", assembler))
    command = [
        assembler,
        "-lineinfo",
        "-v",
        "--gpu-name=sm_90a",
        str(folder / "kernel.ptx"),
        "-o",
        str(folder / "kernel.cubin"),
    ]
    state = {
        "variant": args.variant,
        "ptx_edits": edits,
        "passed": False,
        "device": torch.cuda.get_device_name(),
        "assembler": assembler,
    }
    try:
        state["phase"] = "assemble"
        result = subprocess.run(command, capture_output=True, text=True, timeout=120)
        (folder / "ptxas.txt").write_text(
            repr(command) + "\n" + result.stdout + result.stderr
        )
        assert result.returncode == 0, result.stderr
        state["cubin_sha256"] = digest(folder / "kernel.cubin")
        state["matches_captured_cubin"] = (
            state["cubin_sha256"] == info["original_cubin_sha256"]
        )
        state["phase"] = "input preparation"
        runtime, expected = inputs()
        if args.verify_inputs:
            saved = torch.load(
                args.verify_inputs, map_location="cpu", weights_only=True
            )
            for actual, original in zip(runtime[:5], saved["args"][:5]):
                assert actual.stride() == original.stride()
                assert torch.equal(actual.cpu(), original)
            state["original_inputs_equal"] = True
        torch.cuda.synchronize()
        state["phase"] = "binary load"
        src = ASTSource(
            abi_stub, info["signature"], {tuple(k): v for k, v in info["constants"]}
        )
        group = {
            "kernel." + ext: str(folder / ("kernel." + ext))
            for ext in ("json", "ptx", "cubin")
        }
        compiled = CompiledKernel(src, group, info["hash"])
        compiled._init_handles()
        constants = {path[0]: value for path, value in info["constants"]}
        bound = list(runtime) + [constants[i] for i in range(15, 22)]
        state["phase"] = "launch"
        compiled[tuple(info["grid"])](*bound)
        state["phase"] = "synchronize"
        torch.cuda.synchronize()
        state["phase"] = "validate"
        logits, visible = runtime[5], runtime[4]
        mask = torch.arange(1552, device="cuda")[None, :] < visible[:, None]
        torch.testing.assert_close(logits[mask], expected[mask], rtol=2e-4, atol=2e-3)
        state.update(phase="passed", passed=True)
        print("PASS", args.variant, flush=True)
        return 0
    except Exception as exc:
        state["error"] = repr(exc)
        raise
    finally:
        (folder / "result.json").write_text(json.dumps(state, indent=2) + "\n")


if __name__ == "__main__":
    raise SystemExit(main())
