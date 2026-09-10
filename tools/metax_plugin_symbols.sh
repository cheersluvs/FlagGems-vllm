#!/usr/bin/env bash
# Does MetaX's prebuilt Triton plugin carry the TLE bindings?
#
# FlagTree's metax backend links a PREBUILT metaxTritonPlugin.so rather than
# compiling third_party/metax/plugin/mctle, which it only builds when
# FLAGTREE_PLUGIN is set. So the tag 0.6.1a2+metax3.6 has the mctle source --
# make_swizzled_shared_encoding_attr, create_local_pointers and the rest -- but
# the wheel's bindings come from whatever .so MetaX shipped.
#
# The installed libtriton exposes none of them, so either that .so is older
# than the source, or it is not the one being loaded. Look at the symbols.
set -uo pipefail

echo "### $(date -Is)  host $(uname -n)"
echo

PY=$(python -c "import triton,os;print(os.path.dirname(triton.__file__))" 2>/dev/null)
echo "triton package: ${PY:-<not found>}"
echo

echo "=== candidate plugin libraries ==="
for d in "$PY/_C" "$HOME/.flagtree/metax" "$HOME/.flagtree" /opt/maca; do
    [ -d "$d" ] || continue
    find "$d" -maxdepth 3 -name "*.so" 2>/dev/null | while read -r f; do
        printf "  %-72s %8s KB  %s\n" "$f" "$(( $(stat -c%s "$f" 2>/dev/null || echo 0) / 1024 ))" "$(stat -c%y "$f" 2>/dev/null | cut -d. -f1)"
    done
done
echo

WANT="make_swizzled_shared_encoding_attr make_nv_mma_shared_encoding_attr create_local_pointers create_local_alloc create_local_load create_local_store create_exclusive_cumsum"

echo "=== which .so, if any, mentions the TLE bindings ==="
for d in "$PY/_C" "$HOME/.flagtree/metax" "$HOME/.flagtree"; do
    [ -d "$d" ] || continue
    find "$d" -maxdepth 3 -name "*.so" 2>/dev/null | while read -r f; do
        hits=""
        for w in $WANT; do
            if strings -a "$f" 2>/dev/null | grep -qF "$w"; then hits="$hits $w"; fi
        done
        if [ -n "$hits" ]; then
            echo "  $f"
            for h in $hits; do echo "      $h"; done
        else
            echo "  $f  -- none"
        fi
    done
done
echo
echo "=== is mctle even in this wheel's python side? ==="
for f in "$PY/backends/metax/tle_supported.py" "$PY/experimental/tle/__init__.py"; do
    printf "  %-70s %s\n" "$f" "$([ -f "$f" ] && echo present || echo MISSING)"
done
