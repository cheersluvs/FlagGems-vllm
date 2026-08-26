"""Skip the shapes the composed eager baseline cannot fit, before it OOMs.

WHAT THIS IS FOR. The torch_npu eager composition needs about six times q's
own size in scratch -- `q.float()`, npu_rms_norm's fp32 output, the fp32 rope
intermediate and the bf16 write-back are all live at once -- so on a 60.96 GiB
910B it dies at 98304x128 (75 GiB) and 131072x128 (100 GiB). Two of 22 shapes.

PREDICT, DO NOT CATCH. Catching the OOM would be simpler and is wrong here. On
this backend a failed allocation does not always land on the call that caused
it, and an error left in flight poisons everything after it in the process --
that is exactly how a stray `aicpu timeout` 507017 corrupted a whole eager
sweep before and forced one-shape-per-process isolation. Predicting means the
bad shape is never launched, so there is nothing to poison. The OOM catch below
is a backstop for a wrong prediction, not the mechanism.

**THE PEAK MODEL IS A COPY OF THE COMPOSITION AND WILL GO STALE.** The list in
`eager_peak_terms` mirrors the temporaries in `eager_baseline.py` term by term.
If that composition changes -- someone drops the `.float()`, adds an `out=`,
fuses two steps -- this predicate keeps answering for the old code and will
skip a shape that now fits, or launch one that no longer does. It is written as
a named checklist rather than a `6 *` constant so that it can be read against
the source. Re-read it when you touch the baseline.

WHY SKIPPING IS LEGITIMATE HERE, AND THE ONE WAY IT STOPS BEING. The two
missing cells are not the large end: 131072 tokens at 64 heads runs and is
reported, so the table is not quietly dropping the shapes where a ratio is
hardest to win. And "the framework composition needs 100 GiB of scratch for a
16 GiB q" is itself the result -- the fused kernel does it in place. What turns
this from a stated limit into a hole is aggregating over the survivors without
saying so: any geomean or range computed from these runs must be labelled
20/22 and name the two absent cells. `summary_note()` exists to make that hard
to forget.

Say "the eager composition does not fit", never "the card cannot do this
shape". The card runs it fine -- the fused kernel is measured on it.
"""

import functools
import os

HEAD_DIM = 512
ROPE_DIM = 64

# Fraction of free memory we are willing to plan against. The allocator
# fragments and the harness holds its own buffers, so a shape predicted to need
# 99% of free will fail in practice.
USABLE_FRACTION = float(os.environ.get("EAGER_SKIP_USABLE", "0.85"))


def eager_peak_terms(num_tokens, num_heads, elem_bytes=2):
    """The composition's live temporaries, term by term. Bytes.

    Mirrors eager_baseline.py. Each entry names the line it comes from so the
    two can be diffed by eye.
    """
    body = num_tokens * num_heads * HEAD_DIM
    #                                                      bytes, already resident?
    return [
        ("q, the caller's tensor", body * elem_bytes, True),
        ("q.reshape(-1, HEAD_DIM).float()", body * 4, False),
        ("npu_rms_norm output, fp32", body * 4, False),
        ("rope intermediate on the last ROPE_DIM, fp32",
         num_tokens * num_heads * ROPE_DIM * 4, False),
        ("the .to(q.dtype) temporary before copy_", body * elem_bytes, False),
    ]


def eager_peak_bytes(num_tokens, num_heads, elem_bytes=2):
    """Total footprint at the peak, q included. For explaining, not deciding."""
    return sum(b for _, b, _ in eager_peak_terms(num_tokens, num_heads, elem_bytes))


def eager_additional_bytes(num_tokens, num_heads, elem_bytes=2):
    """What must be NEWLY allocated. This is the number to compare against free.

    q is already on the device when the wrapper runs, so it has already been
    subtracted from the free figure. Comparing the full peak (which counts q)
    against free (which does not) double-counts q and makes the predicate too
    conservative by exactly q's size -- at 131072x64 that is 8 GiB against a
    2 GiB true margin, so it would have deleted a shape that is known to run.
    """
    return sum(b for _, b, resident in
               eager_peak_terms(num_tokens, num_heads, elem_bytes) if not resident)


def device_free_bytes():
    """Free bytes on the device, or None if it cannot be determined.

    Goes through flaggems_vllm's own device handle rather than torch.cuda:
    on MUSA the namespace is torch.musa and torch.cuda reports zero devices,
    which has already cost one probe a bogus `device_count = 0`. Returning None
    rather than guessing a card size is deliberate -- an unknown budget must
    disable the predicate, not invent one.
    """
    try:
        import flaggems_vllm

        fn = flaggems_vllm.runtime.torch_device_fn
    except Exception:
        return None
    try:
        free, _total = fn.mem_get_info()
        return int(free)
    except Exception:
        pass
    try:
        total = fn.get_device_properties(0).total_memory
        return int(total) - int(fn.memory_reserved())
    except Exception:
        return None


def will_fit(num_tokens, num_heads, elem_bytes=2, free_override=None):
    """(fits, reason). `fits` is True when the budget is unknown.

    Failing open is the right default -- a missing device reading must not
    silently delete shapes from a table -- but it means a caller that never
    checks the reason cannot tell "it fits" from "I could not tell". Callers
    that need the distinction should read the reason string, and the self-test
    below shows what the predicate does under a stated budget rather than
    letting a device-less machine report that everything fits.
    """
    need = eager_additional_bytes(num_tokens, num_heads, elem_bytes)
    free = device_free_bytes() if free_override is None else free_override
    gib = 1024.0 ** 3
    body = num_tokens * num_heads * HEAD_DIM * elem_bytes
    if free is None:
        return True, "device free memory unknown; predicate disabled"
    budget = free * USABLE_FRACTION
    if need <= budget:
        return True, "needs {:.1f} GiB of {:.1f} GiB usable".format(
            need / gib, budget / gib)
    return False, (
        "eager composition needs ~{:.1f} GiB of fresh scratch for a {:.1f} GiB q; "
        "{:.1f} GiB free ({:.1f} GiB usable at {:.0%}). The card runs this "
        "shape -- the fused kernel is measured on it -- the composition is "
        "what does not fit.".format(
            need / gib, body / gib, free / gib, budget / gib, USABLE_FRACTION))


def explain(num_tokens, num_heads, elem_bytes=2):
    """Print the term-by-term account. For when a skip needs justifying."""
    gib = 1024.0 ** 3
    print("  {}x{} eager peak:".format(num_tokens, num_heads))
    for name, b, resident in eager_peak_terms(num_tokens, num_heads, elem_bytes):
        print("    {:<48}{:>8.1f} GiB{}".format(
            name, b / gib, "   (already resident)" if resident else ""))
    print("    {:<48}{:>8.1f} GiB".format(
        "peak total", eager_peak_bytes(num_tokens, num_heads, elem_bytes) / gib))
    print("    {:<48}{:>8.1f} GiB   <- compared against free".format(
        "of which newly allocated",
        eager_additional_bytes(num_tokens, num_heads, elem_bytes) / gib))


class WontFit(Exception):
    """Raised by the wrapper when a shape is skipped by prediction."""


def skip_if_wont_fit(ref, use_pytest_skip=True):
    """Wrap the composed baseline so an unfittable shape skips instead of OOMs.

    Mirrors `_skip_if_unrunnable` in the benchmark file: pytest.skip raises
    Skipped, which derives from BaseException and so passes intact through the
    harness's `except (RuntimeError, Exception)`. Standalone runners that want
    to keep going instead should pass use_pytest_skip=False and catch WontFit.
    """
    skipped = []

    @functools.wraps(ref)
    def wrapper(*args, **kwargs):
        q = args[0] if args else kwargs["q"]
        n, h = int(q.shape[0]), int(q.shape[1])
        elem = q.element_size()
        fits, reason = will_fit(n, h, elem)
        if not fits:
            skipped.append((n, h, reason))
            if use_pytest_skip:
                import pytest

                pytest.skip("{}x{}: {}".format(n, h, reason))
            raise WontFit("{}x{}: {}".format(n, h, reason))
        try:
            return ref(*args, **kwargs)
        except RuntimeError as e:
            # Backstop only. A prediction that says "fits" and then OOMs means
            # the model above is wrong -- report it as a model failure, not as
            # a tidy skip, so it gets fixed rather than absorbed.
            if "out of memory" not in str(e).lower():
                raise
            raise RuntimeError(
                "{}x{} OOMed despite being predicted to fit -- eager_skip's "
                "peak model is out of date with the composition. Original: {}"
                .format(n, h, str(e).splitlines()[0])) from e

    wrapper.skipped = skipped
    return wrapper


def summary_note(total_shapes, skipped):
    """The label any aggregate over a partial sweep has to carry."""
    if not skipped:
        return "all {} shapes ran".format(total_shapes)
    names = ", ".join("{}x{}".format(n, h) for n, h, _ in skipped)
    return (
        "{}/{} shapes; {} omitted because the eager composition does not fit "
        "({}). Any geomean or range here is over the {} that ran."
        .format(total_shapes - len(skipped), total_shapes, len(skipped), names,
                total_shapes - len(skipped)))


# What the 910B actually did, on the run that produced the eager table. The
# predicate has to reproduce this: too loose and a shape OOMs and poisons the
# process, too tight and it deletes a shape that is known to run. 131072x64 is
# the one that pins the tolerance -- it fits with about 7% to spare, and an
# earlier version of this file failed it by double-counting q.
OBSERVED_OOM = {(98304, 128), (131072, 128)}


def check_against_observed(card_free_gib=60.96, verbose=True):
    gib = 1024.0 ** 3
    bad = []
    for n in (8192, 32768, 65536, 98304, 131072):
        for h in (64, 128):
            resident = sum(b for _, b, r in eager_peak_terms(n, h) if r)
            resident += n * 512 * 2 + (n // 64 + 1) * 64 * 584
            fits, _ = will_fit(n, h,
                               free_override=int(card_free_gib * gib) - resident)
            expected = (n, h) not in OBSERVED_OOM
            if fits != expected:
                bad.append((n, h, fits, expected))
    if bad and verbose:
        for n, h, got, want in bad:
            print("  DISAGREES at {}x{}: predicate says {}, the card said {}"
                  .format(n, h, "fits" if got else "skip",
                          "ran" if want else "OOM"))
    return bad


if __name__ == "__main__":
    gib = 1024.0 ** 3
    detected = device_free_bytes()
    if detected is None:
        # A device-less machine would otherwise print "yes" for every row,
        # including the 100 GiB one, and look like a passing self-test.
        free = int(60.96 * gib)
        print("No device visible. Exercising the predicate against a STATED "
              "HYPOTHETICAL budget of {:.2f} GiB free (a 910B).".format(free / gib))
        print("These rows are the logic, not a measurement of this machine.\n")
    else:
        free = detected
        print("Detected {:.2f} GiB free; planning against {:.0%} of it.\n"
              .format(free / gib, USABLE_FRACTION))

    skipped = []
    print("  {:>7} {:>5} {:>10} {:>8}".format("tokens", "heads", "peak GiB", "fits?"))
    shapes = [(n, h) for n in (8192, 32768, 65536, 98304, 131072)
              for h in (64, 128)]
    for n, h in shapes:
        # Model the real call site: the wrapper runs after q/kv/k_cache are
        # allocated, so free is already reduced by them. Passing the whole
        # card here would test a situation that never occurs.
        resident = sum(b for _, b, r in eager_peak_terms(n, h) if r)
        resident += n * 512 * 2 + (n // 64 + 1) * 64 * 584   # kv, k_cache
        fits, reason = will_fit(n, h, free_override=free - resident)
        if not fits:
            skipped.append((n, h, reason))
        print("  {:>7} {:>5} {:>10.1f} {:>8}".format(
            n, h, eager_peak_bytes(n, h) / gib, "yes" if fits else "SKIP"))

    print("\n" + summary_note(len(shapes), skipped) + "\n")
    if skipped:
        print("Why the first skip:\n  " + skipped[0][2] + "\n")
    explain(131072, 128)

    bad = check_against_observed()
    print()
    if bad:
        print("SELF-TEST FAILED: the predicate no longer matches what the card "
              "did. Fix the peak model before trusting any skip.")
        raise SystemExit(1)
    print("Self-test: predicate reproduces all 10 observed outcomes.")
