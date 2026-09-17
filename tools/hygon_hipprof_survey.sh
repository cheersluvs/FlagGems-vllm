#!/usr/bin/env bash
# Can hipprof count anything on the real prefill kernel?
#
# Round 1 found no rocprof/rocprofv2/rocprofv3/omniperf on this box -- the only
# profiler is DTK's own /opt/dtk/bin/hipprof. It also wasted its rocminfo
# budget on the CPU agents (every "Hygon C86 Processor, Compute Unit: 16" line
# is a CPU core as an HSA agent) and never reached the GPU agent, so the card's
# own parameters are read through torch here instead of by parsing rocminfo.
#
# Still not guessing counter names: ask hipprof what it accepts, then try, in
# order, the forms rocprof-derived tools usually take. Every attempt prints the
# command it ran and its output, so a failure says which form failed rather
# than leaving a blank section.
#
#   tools/vendor_probe.sh tools/hygon_hipprof_survey.sh hygon_hipprof_survey
set -uo pipefail

HP=/opt/dtk/bin/hipprof
PY=${PY:-python}
DRIVER=tools/hygon_prefill_one.py
SHAPE="16383 4095 512 4352 20"
OUTDIR=/tmp/hipprof_survey

say() { printf '\n=== %s\n' "$*"; }
run() { printf '\n--- $ %s\n' "$*"; timeout 600 "$@" 2>&1 | head -40; }

say "the GPU agent, via torch (rocminfo lists every CPU core as an agent too)"
"$PY" - <<'PYEOF'
import torch
p = torch.cuda.get_device_properties(0)
for k in sorted(dir(p)):
    if k.startswith("_"):
        continue
    try:
        print(f"  {k} = {getattr(p, k)}")
    except Exception as exc:  # noqa: BLE001
        print(f"  {k} !! {exc!r}")
PYEOF

say "GPU agent from rocminfo, this time actually selecting it"
if command -v rocminfo >/dev/null 2>&1; then
    rocminfo 2>/dev/null | grep -B4 -A28 'Device Type:.*GPU' | head -60
else
    echo "  rocminfo absent"
fi

say "hipprof interface"
ls -la "$HP" "$(dirname "$HP")"/hipprof* 2>/dev/null
run "$HP" --help
run "$HP" --version

say "what hipprof says it can count"
run "$HP" --list-basic
run "$HP" --list-derived
run "$HP" --list-counters
ls /opt/dtk/lib/rocprofiler /opt/dtk/rocprofiler 2>/dev/null | head -20
find /opt/dtk -maxdepth 4 -name '*metrics*.xml' -o -maxdepth 4 -name '*counter*.xml' 2>/dev/null | head

say "kernel trace of the real operator"
rm -rf "$OUTDIR" && mkdir -p "$OUTDIR"
run "$HP" --stats -o "$OUTDIR/stats.csv" "$PY" "$DRIVER" $SHAPE
echo "--- files"
ls -la "$OUTDIR" 2>/dev/null | head -20
for f in "$OUTDIR"/*.csv "$OUTDIR"/*.db "$OUTDIR"/*.json; do
    [ -f "$f" ] || continue
    echo "--- $f"
    head -12 "$f"
done

say "if --stats did not work, the plainest form"
run "$HP" "$PY" "$DRIVER" $SHAPE

say "survey done"
