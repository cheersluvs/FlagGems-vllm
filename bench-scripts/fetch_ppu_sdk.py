"""Pull PPU SDK 2.1.0 out of the published flagtree image without docker.

This box has ppu-llc 2.0.0 and flagtree's ppu3.6 line needs 2.1.0, which is
published only inside
  harbor.baai.ac.cn/flagtree/flagtree-ppu-py312-torch2.10.0-sdk2.1.0-cu130-ubuntu24.04
(anonymous read is allowed). There is no docker here, so talk to the registry
v2 API directly -- and do NOT download all 18.8 GB: the image config carries a
`history` entry per layer, so the layer that added the SDK can be identified
first and fetched alone.

    python3 bench-scripts/fetch_ppu_sdk.py --list
    python3 bench-scripts/fetch_ppu_sdk.py --extract 12 --out /root/ppu_sdk_2.1.0

`--extract` streams that one layer and writes only the files matching --pattern
(default: ppu-llc and llvm-irformatter), so nothing but the binaries lands on
disk. Nothing installs itself: point TRITON_PPU_LLC_PATH at the result, which is
first in the backend's search order, so /usr/local/PPU_SDK -- the only working
toolchain on this box -- is left untouched.
"""

import argparse
import gzip
import json
import os
import re
import sys
import tarfile
import urllib.error
import urllib.request

HOST = "harbor.baai.ac.cn"
REPO = "flagtree/flagtree-ppu-py312-torch2.10.0-sdk2.1.0-cu130-ubuntu24.04"
TAG = "202608-3.6-base"

MANIFEST_ACCEPT = ", ".join([
    "application/vnd.docker.distribution.manifest.v2+json",
    "application/vnd.docker.distribution.manifest.list.v2+json",
    "application/vnd.oci.image.manifest.v1+json",
    "application/vnd.oci.image.index.v1+json",
])


def _open(url, token=None, accept=None):
    req = urllib.request.Request(url)
    if token:
        req.add_header("Authorization", "Bearer " + token)
    if accept:
        req.add_header("Accept", accept)
    return urllib.request.urlopen(req, timeout=120)


def get_token(repo):
    """Anonymous pull token. Harbor answers the standard 401 + Www-Authenticate."""
    url = "https://{}/v2/".format(HOST)
    try:
        _open(url).read()
        return None
    except urllib.error.HTTPError as e:
        if e.code != 401:
            raise
        chal = e.headers.get("Www-Authenticate", "")
    realm = re.search(r'realm="([^"]+)"', chal)
    service = re.search(r'service="([^"]+)"', chal)
    if not realm:
        return None
    url = "{}?service={}&scope=repository:{}:pull".format(
        realm.group(1), service.group(1) if service else "harbor-registry", repo)
    return json.loads(_open(url).read())["token"]


def fetch_json(path, token, accept=None):
    url = "https://{}/v2/{}/{}".format(HOST, REPO, path)
    return json.loads(_open(url, token, accept).read())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default=TAG)
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--extract", type=int, metavar="LAYER_INDEX")
    ap.add_argument("--pattern", default=r"(^|/)(ppu-llc|llvm-irformatter)$")
    ap.add_argument("--out", default="/root/ppu_sdk_2.1.0")
    args = ap.parse_args()

    token = get_token(REPO)
    print("registry {}  repo {}  tag {}".format(HOST, REPO, args.tag))
    print("anonymous token: {}".format("obtained" if token else "not needed"))

    man = fetch_json("manifests/" + args.tag, token, MANIFEST_ACCEPT)
    if "manifests" in man:  # an index: take linux/amd64
        pick = [m for m in man["manifests"]
                if m.get("platform", {}).get("architecture") == "amd64"]
        assert pick, "no amd64 manifest in the index"
        man = fetch_json("manifests/" + pick[0]["digest"], token, MANIFEST_ACCEPT)

    cfg = fetch_json("blobs/" + man["config"]["digest"], token)
    layers = man["layers"]
    hist = [h for h in cfg.get("history", []) if not h.get("empty_layer")]
    print("{} layers, {} non-empty history entries\n".format(len(layers), len(hist)))

    if args.extract is None or args.list:
        for i, lay in enumerate(layers):
            made = (hist[i].get("created_by", "") if i < len(hist) else "").strip()
            made = re.sub(r"\s+", " ", made)
            mark = "  <-- SDK?" if re.search(r"sdk|ppu", made, re.I) else ""
            print("  [{:>2}] {:>8.2f} MB  {}{}".format(
                i, lay["size"] / 2**20, made[:110], mark))
        if args.extract is None:
            print("\nPick a layer and rerun with --extract N.")
            return

    i = args.extract
    lay = layers[i]
    pat = re.compile(args.pattern)
    os.makedirs(args.out, exist_ok=True)
    print("streaming layer [{}] {} ({:.2f} MB) -- writing only {}".format(
        i, lay["digest"][:19], lay["size"] / 2**20, args.pattern))
    url = "https://{}/v2/{}/blobs/{}".format(HOST, REPO, lay["digest"])
    resp = _open(url, token)
    raw = gzip.GzipFile(fileobj=resp) if lay["mediaType"].endswith("gzip") else resp
    found = []
    with tarfile.open(fileobj=raw, mode="r|") as tf:
        for m in tf:
            if not m.isfile() or not pat.search(m.name):
                continue
            dst = os.path.join(args.out, os.path.basename(m.name))
            with open(dst, "wb") as f:
                src = tf.extractfile(m)
                while True:
                    chunk = src.read(1 << 20)
                    if not chunk:
                        break
                    f.write(chunk)
            os.chmod(dst, 0o755)
            found.append((m.name, os.path.getsize(dst)))
            print("  extracted {}  ({:.2f} MB)  from {}".format(
                os.path.basename(m.name), os.path.getsize(dst) / 2**20, m.name))
    if not found:
        print("\n  nothing matched in this layer; try another index from --list")
        print("[RESULT] NOT_IN_THIS_LAYER")
        return
    print("\n  now:  {}/ppu-llc --version".format(args.out))
    print("  then: TRITON_PPU_LLC_PATH={}/ppu-llc  (leaves /usr/local/PPU_SDK alone)".format(args.out))
    print("[RESULT] EXTRACTED")


main()
