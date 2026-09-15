"""Turn a per-shape table log into the markdown table PR #684 uses for S5000.

Reads the output of run_c550_table.py, run_thead_table.py or
run_asshipped_table.py, collects every `<ratio>x` on each 22-shape row
(tokens in the benchmark list, heads 64/128), and prints the mean per shape
to three decimals -- both rounds when a runner prints two. Missing shapes are
reported, not filled in.

    python3 md_table.py run1.log [run2.log ...]
"""
import re
import statistics
import sys

TOKENS = (1, 4, 17, 64, 1024, 2048, 8192, 32768, 65536, 98304, 131072)
ROW = re.compile(r"^\s*\|?\s*(\d+)\s*\|?\s+(64|128)\b(.*)$")
vals = {}
for line in (l for f in sys.argv[1:] for l in open(f, errors="replace")):
    m = ROW.match(line)
    if not m or int(m.group(1)) not in TOKENS:
        continue
    for r in re.findall(r"(\d+\.\d{3})x", m.group(3)):
        vals.setdefault((int(m.group(1)), int(m.group(2))), []).append(float(r))

print("| tokens | 64 heads | 128 heads |")
print("|---|---|---|")
missing = []
for n in TOKENS:
    cells = []
    for h in (64, 128):
        v = vals.get((n, h))
        cells.append("{:.3f}".format(statistics.mean(v)) if v else "—")
        if not v:
            missing.append("{}x{}".format(n, h))
    print("| {} | {} | {} |".format(n, *cells))
spread = [max(v) / min(v) - 1 for v in vals.values() if len(set(v)) > 1]
print("\nshapes found {}/22{}; largest spread between readings {:.1%}".format(
    22 - len(missing), (", missing " + " ".join(missing)) if missing else "",
    max(spread) if spread else 0.0))
