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
    ap.add_argument("--grep", default="", metavar="REGEX",
                    help="only list layers whose command matches this")
    ap.add_argument("--out", default="/root/ppu_sdk_2.1.0")
    ap.add_argument("--keep-paths", action="store_true",
                    help="recreate the tree (and its symlinks) under --out instead of "
                         "flattening -- needed when ppu-llc turns out to load SDK .so files")
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
            # buildkit prefixes RUN layers with the whole build-arg environment,
            # which is what truncated the useful half out of the first listing
            cut = made.find("/bin/sh -c")
            if cut > 0:
                made = made[cut + len("/bin/sh -c"):].strip()
            if args.grep and not re.search(args.grep, made, re.I):
                continue
            print("  [{:>2}] {:>8.2f} MB  {}".format(i, lay["size"] / 2**20, made[:190]))
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
            if not pat.search(m.name):
                continue
            if args.keep_paths:
                dst = os.path.join(args.out, m.name.lstrip("./"))
                os.makedirs(os.path.dirname(dst), exist_ok=True)
                if m.issym():
                    if os.path.lexists(dst):
                        os.remove(dst)
                    os.symlink(m.linkname, dst)
                    continue
                if m.isdir():
                    os.makedirs(dst, exist_ok=True)
                    continue
            if not m.isfile():
                continue
            if not args.keep_paths:
                dst = os.path.join(args.out, os.path.basename(m.name))
            with open(dst, "wb") as f:
                src = tf.extractfile(m)
                while True:
                    chunk = src.read(1 << 20)
                    if not chunk:
                        break
                    f.write(chunk)
            os.chmod(dst, m.mode if args.keep_paths else 0o755)
            found.append((m.name, os.path.getsize(dst)))
            if not args.keep_paths or re.search(r"/bin/[^/]+$", m.name):
                print("  extracted {}  ({:.2f} MB)  from {}".format(
                    os.path.basename(m.name), os.path.getsize(dst) / 2**20, m.name))
    if args.keep_paths and found:
        print("  {} files, {:.1f} MB total under {}".format(
            len(found), sum(sz for _, sz in found) / 2**20, args.out))
    if not found:
        print("\n  nothing matched in this layer; try another index from --list")
        print("[RESULT] NOT_IN_THIS_LAYER")
        return
    print("\n  now:  {}/ppu-llc --version".format(args.out))
    print("  then: TRITON_PPU_LLC_PATH={}/ppu-llc  (leaves /usr/local/PPU_SDK alone)".format(args.out))
    print("[RESULT] EXTRACTED")


main()
