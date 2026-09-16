"""Turn per-shape table logs into the markdown table PR #684 uses, with spreads.

Reads the output of run_c550_table.py, run_thead_table.py or
run_asshipped_table.py, collects every `<ratio>x` on each 22-shape row (tokens
in the benchmark list, heads 64/128) and prints the mean per shape to three
decimals.

**A shape's spread across readings is printed next to it.** A single global
"largest spread" hides which rows it belongs to, and on these parts the small
shapes are where the measurement stops being usable -- a row that moved 128%
between two rounds is not a number to quote, and averaging it does not make it
one. Rows above SPREAD_WARN are listed again at the end.

    python3 md_table.py run1.log [run2.log ...]
"""
import re
import statistics
import sys

TOKENS = (1, 4, 17, 64, 1024, 2048, 8192, 32768, 65536, 98304, 131072)
SPREAD_WARN = 0.05
ROW = re.compile(r"^\s*\|?\s*(\d+)\s*\|?\s+(64|128)\b(.*)$")

vals = {}
for line in (l for f in sys.argv[1:] for l in open(f, errors="replace")):
    m = ROW.match(line)
    if not m or int(m.group(1)) not in TOKENS:
        continue
    for r in re.findall(r"(\d+\.\d{3})x", m.group(3)):
        vals.setdefault((int(m.group(1)), int(m.group(2))), []).append(float(r))


def spread(v):
    return max(v) / min(v) - 1 if v and min(v) > 0 else 0.0


print("| tokens | 64 heads | 128 heads |")
print("|---|---|---|")
missing, noisy = [], []
for n in TOKENS:
    cells = []
    for h in (64, 128):
        v = vals.get((n, h))
        if not v:
            cells.append("—")
            missing.append("{}x{}".format(n, h))
            continue
        cells.append("{:.3f}".format(statistics.mean(v)))
        if spread(v) > SPREAD_WARN:
            noisy.append((n, h, spread(v), v))
    print("| {} | {} | {} |".format(n, *cells))

print("\nper-shape readings and spread:")
for n in TOKENS:
    for h in (64, 128):
        v = vals.get((n, h))
        if v:
            print("  {:>7} x {:<3} {:>7.1%}   {}".format(
                n, h, spread(v), " ".join("{:.3f}".format(x) for x in v)))
print("\nshapes found {}/22{}".format(22 - len(missing),
                                      (", missing " + " ".join(missing)) if missing else ""))
if noisy:
    print("DO NOT QUOTE without a decision — spread over {:.0%}:".format(SPREAD_WARN))
    for n, h, s, v in noisy:
        print("  {}x{}  {:.1%}  {}".format(n, h, s, " ".join("{:.3f}".format(x) for x in v)))
