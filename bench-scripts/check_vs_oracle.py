

# ------------------------------------------------- baseline against the ORACLE
import importlib.util
import sys
import traceback

REPO = "/home/secure/wuyuqing/workspace/FlagGems-vllm"
sys.path.insert(0, REPO)
sys.path.insert(0, REPO + "/src")

CACHE_BLOCK = 64


def load_oracle():
    """The test file's torch reference, imported directly.

    This is the independent check. Comparing the baseline to the OPERATOR only
    shows they agree; if both made the same mistake they would agree and both be
    wrong. The oracle was written separately, and it is what the suite judges
    the operator by.
    """
    import types
    pkg = types.ModuleType("tests")
    pkg.__path__ = [REPO + "/tests"]
    sys.modules.setdefault("tests", pkg)
    spec = importlib.util.spec_from_file_location(
        "tests.test_fused_deepseek_v4_qnorm_rope_kv_rope_quant_insert",
        REPO + "/tests/test_fused_deepseek_v4_qnorm_rope_kv_rope_quant_insert.py",
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def make(n, h, neg=False):
    dev = "npu"
    nb = (n + CACHE_BLOCK - 1) // CACHE_BLOCK + 1
    bb = CACHE_BLOCK * (TOKEN_DATA_BYTES + SCALE_BYTES_PER_TOKEN)
    torch.manual_seed(0)
    q = torch.randn(n, h, HEAD_DIM, dtype=torch.float32).to(torch.bfloat16).npu()
    kv = torch.randn(n, HEAD_DIM, dtype=torch.float32).to(torch.bfloat16).npu()
    slot = torch.arange(n, dtype=torch.int64, device=dev)
    if neg and n > 3:
        slot[1] = -1
        slot[n // 2] = -1
    pos = torch.arange(n, dtype=torch.int64, device=dev)
    cs = torch.randn(max(4096, n), ROPE_DIM, dtype=torch.float32).npu()
    cache = torch.zeros(nb, bb, dtype=torch.uint8, device=dev)
    return q, kv, cache, slot, pos, cs


def main():
    t = load_oracle()
    print("oracle loaded from the test file\n")

    print("### the eager BASELINE against the test file's ORACLE\n")
    print("  {:<24} {:>14} {:>18} {:>12}".format(
        "shape", "q differs", "k_cache differs", "q rel<=1e-2"))
    for n, h, neg in ((17, 64, False), (64, 128, False), (1024, 64, False),
                      (64, 64, True)):
        q1, kv1, c1, sl, po, cs = make(n, h, neg)
        q2, kv2, c2 = q1.clone(), kv1.clone(), c1.clone()

        # oracle mutates q, kv and k_cache in place
        t.ref_impl(q1, kv1, c1, sl.clone(), po.clone(), cs.clone(), 1e-6,
                   CACHE_BLOCK)
        eager_fused_deepseek_v4(q2, kv2, c2, sl, po, cs, 1e-6, CACHE_BLOCK)
        torch.npu.synchronize()

        a, b = q1.cpu().float(), q2.cpu().float()
        dq = int((q1.cpu() != q2.cpu()).sum())
        dc = int((c1.cpu() != c2.cpu()).sum())
        close = torch.allclose(a, b, rtol=1e-2, atol=1e-2)
        print("  {:<24} {:>14} {:>18} {:>12}".format(
            "{}x{}{}".format(n, h, " (-1 slots)" if neg else ""), dq, dc,
            str(close)))
        del q1, q2, kv1, kv2, c1, c2
        torch.npu.empty_cache()

    print("\n### is npu_rms_norm exact at the operator's real widths?\n")
    torch.manual_seed(1)
    for n in (1024, 65536):
        x = torch.randn(n, HEAD_DIM, dtype=torch.float32).npu()
        g = torch.ones(HEAD_DIM, dtype=torch.float32, device="npu")
        ours = x * torch.rsqrt((x * x).mean(-1, keepdim=True) + 1e-6)
        out = torch_npu.npu_rms_norm(x, g, epsilon=1e-6)
        got = out[0] if isinstance(out, (tuple, list)) else out
        d = (got.view(torch.int32) != ours.view(torch.int32))
        print("  {:>6} rows: {} of {} float32 words differ".format(
            n, int(d.sum()), d.numel()))
        del x, g, ours, got
        torch.npu.empty_cache()

    print("\n[RESULT] DONE")


try:
    main()
except Exception:
    traceback.print_exc()
    print("\n[RESULT] FAILED")
sys.stdout.flush()
