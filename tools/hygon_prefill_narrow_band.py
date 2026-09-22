"""Does the narrow-band key collapse reach a path that is actually shipped?

WHY. The 11-bit fp16 STEP-0 key resolves magnitude/32, so a row whose values
sit inside a band narrower than that maps to ONE bin. Then "an overflow can
only drop what shares the k-th element's key" is true but vacuous. This has
now been found and fixed twice -- MTT (PR #831) and Hygon decode (PR #827) --
and tools/hygon_prefill_sample_tight.py just confirmed the Hygon SAMPLED
prefill path has it too, wrong at every TARGET_MULT.

What has never been checked is whether any path that IS shipped has it. The
generic operator is supposed to escape through STEP 1-3: STEP 0 fails to
converge, STEP 1 (sign + 8 exponent + 2 mantissa bits of the fp32 key) also
collapses on such a band, and STEP 2 (the next 11 bits, ~0.001 of a binade)
finally resolves it. That is the theory; nobody has run it.

`tests/test_top_k_per_row_prefill.py` has eight tests and NO narrow-band case,
which is why several clean suite runs said nothing about this.

ARMS
    gen     the shipped override with every route switched off by env
            (SLOTSCAN, GEOMETRY, ONESCAN all 0) -- i.e. the generic operator
    ship    the shipped override as production runs it
    audit   codex/hygon-prefill-audit's override and its three companions

CASES, on every benchmark shape
    normal    randn
    tied      randn rounded to 1/4
    narrow    10.0 + 0.2*u   -- one fp16 binade, spread 0.2 against a key
                                resolution of 0.25, so ~1 bin
    narrower  10.0 + 0.02*u  -- ten times tighter again
    constant  all 3.5        -- reported for contrast: it passes even when the
                                key has collapsed, because any k of equal
                                values is a correct answer. A constant row does
                                NOT discriminate; a band does.

Each arm runs in its own process, because the override is chosen at import.

RECOVERY, if killed between the swap and the restore:

    git checkout -- src/flaggems_vllm/runtime/backend/_hygon/fused/top_k_per_row_prefill.py
    rm -f src/flaggems_vllm/runtime/backend/_hygon/fused/_top_k_per_row_prefill_*.py

    tools/vendor_probe.sh tools/hygon_prefill_narrow_band.py hygon_prefill_narrow_band
"""

import json
import os
import pathlib
import subprocess
import sys
import tempfile

FUSED = pathlib.Path("src/flaggems_vllm/runtime/backend/_hygon/fused")
OVERRIDE = FUSED / "top_k_per_row_prefill.py"
COMPANIONS = [
    FUSED / "_top_k_per_row_prefill_carry_source.py",
    FUSED / "_top_k_per_row_prefill_final_network.py",
    FUSED / "_top_k_per_row_prefill_final_source.py",
]
AUDIT_REFS = ("origin/codex/hygon-prefill-audit", "0b05008")
SHAPES = [
    (4, 8193, 512, 8456),
    (4, 16385, 512, 16648),
    (64, 129280, 1024, 129280),
    (4100, 1025, 512, 1288),
    (12961, 4100, 512, 4360),
    (16383, 4095, 512, 4352),
    (16380, 5115, 512, 5376),
]
OFF = {
    "FLAGGEMS_HYGON_TOPK_SLOTSCAN": "0",
    "FLAGGEMS_HYGON_TOPK_GEOMETRY": "0",
    "FLAGGEMS_HYGON_TOPK_ONESCAN": "0",
}

CHECKER = """
import json, sys, torch, flaggems_vllm
SHAPES = json.loads(sys.argv[1])
out = []
for rows, vocab, top_k, stride0 in SHAPES:
    torch.manual_seed(42)
    u = torch.rand(rows, vocab, device="cuda", dtype=torch.float32)
    n = torch.randn(rows, vocab, device="cuda", dtype=torch.float32)
    cases = {
        "normal": n,
        "tied": (n * 4).round() / 4,
        "narrow": 10.0 + 0.2 * u,
        "narrower": 10.0 + 0.02 * u,
        "constant": torch.full((rows, vocab), 3.5, device="cuda"),
    }
    starts = torch.zeros(rows, dtype=torch.int32, device="cuda")
    ends = torch.full((rows,), vocab, dtype=torch.int32, device="cuda")
    idx = torch.empty((rows, top_k), dtype=torch.int32, device="cuda")
    for name, src in cases.items():
        want = torch.topk(src, top_k, dim=1).values.sort(dim=1).values
        idx.fill_(-9)
        flaggems_vllm.top_k_per_row_prefill(
            src, starts, ends, idx, rows, stride0, 1, top_k
        )
        torch.cuda.synchronize()
        got = src.gather(1, idx.long().clamp(0, vocab - 1)).sort(dim=1).values
        ok = bool(torch.allclose(got, want)) and bool((idx >= 0).all())
        bad = int((~torch.isclose(got, want)).sum()) if not ok else 0
        ev = [torch.cuda.Event(enable_timing=True) for _ in range(2)]
        for _ in range(3):
            flaggems_vllm.top_k_per_row_prefill(
                src, starts, ends, idx, rows, stride0, 1, top_k
            )
        torch.cuda.synchronize(); ev[0].record()
        for _ in range(5):
            flaggems_vllm.top_k_per_row_prefill(
                src, starts, ends, idx, rows, stride0, 1, top_k
            )
        ev[1].record(); torch.cuda.synchronize()
        out.append({"shape": [rows, vocab, top_k], "case": name, "ok": ok,
                    "bad": bad, "ms": ev[0].elapsed_time(ev[1]) / 5})
print("RESULT " + json.dumps(out))
"""


def sh(*a, **k):
    return subprocess.run(a, capture_output=True, text=True, **k)


def audit_ref():
    for r in AUDIT_REFS:
        if sh("git", "show", f"{r}:{OVERRIDE}").returncode == 0:
            return r
    raise SystemExit("fetch it first:  git fetch origin codex/hygon-prefill-audit")


def install(ref):
    if ref is None:
        assert sh("git", "checkout", "--", str(OVERRIDE)).returncode == 0
        for c in COMPANIONS:
            if c.exists():
                c.unlink()
        return
    for p in [OVERRIDE] + COMPANIONS:
        r = sh("git", "show", f"{ref}:{p}")
        assert r.returncode == 0 and r.stdout, f"cannot read {p} from {ref}"
        p.write_text(r.stdout)


def run_checker(script, env_extra):
    env = dict(os.environ)
    env.update(env_extra)
    r = subprocess.run(
        [sys.executable, script, json.dumps(SHAPES)],
        capture_output=True,
        text=True,
        env=env,
    )
    for line in r.stdout.splitlines():
        if line.startswith("RESULT "):
            return json.loads(line[len("RESULT ") :])
    print(r.stdout[-2500:])
    print(r.stderr[-2000:])
    raise SystemExit("the checker produced no result")


def main():
    ref = audit_ref()
    dirty = sh(
        "git", "status", "--porcelain", "--", *[str(p) for p in [OVERRIDE] + COMPANIONS]
    ).stdout
    if dirty.strip():
        raise SystemExit("those paths are already modified:\n" + dirty)
    script = pathlib.Path(tempfile.mkdtemp(prefix="nb_")) / "check.py"
    script.write_text(CHECKER)
    arms = [("gen", None, OFF), ("ship", None, {}), ("audit", ref, {})]
    res = {}
    try:
        for tag, r, env in arms:
            install(r)
            print(f"### arm {tag}", flush=True)
            res[tag] = run_checker(str(script), env)
    finally:
        install(None)
        left = sh(
            "git",
            "status",
            "--porcelain",
            "--",
            *[str(p) for p in [OVERRIDE] + COMPANIONS],
        ).stdout
        print(f"### restored; git status: {left.strip() or 'clean'}")

    cases = ["normal", "tied", "narrow", "narrower", "constant"]
    print("\ncorrectness by path and input\n")
    print(f"  {'shape':>13} {'case':>10}" + "".join(f"{t:>9}" for t, _, _ in arms))
    wrong = []
    for sh_ in SHAPES:
        key = sh_[:3]
        for c in cases:
            line = f"  {f'{key[0]}x{key[1]}':>13} {c:>10}"
            for tag, _, _ in arms:
                row = next(
                    x for x in res[tag] if x["shape"] == list(key) and x["case"] == c
                )
                line += f"{'OK' if row['ok'] else 'WRONG':>9}"
                if not row["ok"]:
                    wrong.append((tag, key, c, row["bad"]))
            print(line)
    print("\nper-call ms on the narrow band (the cost of escaping through STEP 1-3)\n")
    print(f"  {'shape':>13} {'normal':>10} {'narrow':>10} {'x':>7}   arm")
    for tag, _, _ in arms:
        for sh_ in SHAPES:
            key = sh_[:3]
            n = next(
                x for x in res[tag] if x["shape"] == list(key) and x["case"] == "normal"
            )["ms"]
            w = next(
                x for x in res[tag] if x["shape"] == list(key) and x["case"] == "narrow"
            )["ms"]
            print(
                f"  {f'{key[0]}x{key[1]}':>13} {n:>10.3f} {w:>10.3f} {w / n:>7.2f}   {tag}"
            )
    print()
    if wrong:
        print(f"  {len(wrong)} WRONG results:")
        for tag, key, c, bad in wrong:
            print(f"    {tag:>6}  {key[0]}x{key[1]}  {c}  ({bad} values differ)")
    else:
        print("  every shipped path is correct on every input tested.")
    print(
        "\n  'constant' is reported for contrast only: it passes even when the"
        "\n  key has collapsed, because any k of equal values is correct."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
