#!/usr/bin/env bash
# Which outbound channels does this box actually have?
#
# The FlagTree metax build needs two prebuilt artifacts from a Kingsoft KS3
# endpoint -- a 1.23 GB LLVM 19 and a 0.12 GB plugin -- and that host is
# unreachable here while github.com is not. Both are reachable from the Mac, so
# this is an egress restriction rather than a dead host.
#
# Find a channel that works, so 1.35 GB can be moved through it rather than
# guessed at. Prints no credentials: proxy variables are shown with the
# user:password stripped.
set -uo pipefail

echo "### $(date -Is)  host $(uname -n)"
echo
echo "=== proxy settings visible here (values masked) ==="
found=0
for v in http_proxy https_proxy HTTP_PROXY HTTPS_PROXY no_proxy NO_PROXY all_proxy ALL_PROXY; do
    val="${!v:-}"
    if [ -n "$val" ]; then
        found=1
        printf '  %-12s %s\n' "$v" "$(printf '%s' "$val" | sed -E 's#://[^@/]*@#://***:***@#')"
    fi
done
[ "$found" = 0 ] && echo "  none set"
echo "  git http.proxy: $(git config --get http.proxy 2>/dev/null | sed -E 's#://[^@/]*@#://***:***@#' || echo unset)"
echo

probe() {
    local name="$1" url="$2"
    local dns code
    dns=$(getent hosts "$(printf '%s' "$url" | sed -E 's#https?://([^/]+).*#\1#')" 2>/dev/null | head -1 | awk '{print $1}')
    code=$(curl -sS -o /dev/null -w '%{http_code}' -m 20 -r 0-0 "$url" 2>/dev/null)
    printf '  %-34s dns=%-16s http=%s\n' "$name" "${dns:-FAIL}" "${code:-unreachable}"
}

echo "=== reachability (range request, one byte) ==="
probe "github.com"                 "https://github.com/robots.txt"
probe "codeload.github.com"        "https://codeload.github.com/FlagTree/flagtree/tar.gz/refs/heads/main"
probe "objects.githubusercontent.com" "https://objects.githubusercontent.com/"
probe "raw.githubusercontent.com"  "https://raw.githubusercontent.com/FlagTree/flagtree/main/README.md"
probe "pypi.org"                   "https://pypi.org/simple/"
probe "files.pythonhosted.org"     "https://files.pythonhosted.org/"
probe "KS3 (the artifacts)"        "https://baai-cp-web.ks3-cn-beijing.ksyuncs.com/trans/metaxTritonPlugin-cpython3.12-x86_64_v0.6.1.tar.gz"
echo
echo "A working objects.githubusercontent.com means release assets can carry"
echo "the 1.35 GB; a working KS3 with a proxy means nothing has to be moved."
