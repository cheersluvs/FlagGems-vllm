"""Where did the installed mcoplib come from? Find the index before concluding.

THE PREVIOUS PROBE CONCLUDED WRONGLY AND THIS EXISTS BECAUSE OF IT. It ran
`pip index versions mcoplib`, got "No matching distribution found", and reported
that MetaX has published no >= 0.4.9 wheel. But pip on this box points at
`mirrors.aliyun.com`, a generic PyPI mirror -- which could never have carried
mcoplib in the first place. The installed version is
`0.4.6+maca3.7.1.5.torch2.8`; a local version segment like that does not come
from a generic mirror. So the query proved nothing about MetaX and everything
about which index was asked.

This is the same mistake as T-Head, where the installed vllm was a stock
0.19.0+cu130 and the vendor build sat on a vendor index the whole time. The rule
that came out of that -- "check the vendor index, not just what is installed" --
has a corollary this box just demonstrated: **a negative answer from the wrong
index is not a negative answer.**

So: find the provenance rather than guess a URL. `direct_url.json` in the
dist-info records the URL a package was installed from, `INSTALLER` says which
tool put it there, and pip's own config may have per-user or per-env files that
override the global one seen earlier. A wheel cached on disk is just as good --
its filename and any sibling wheels name the version series that was available.

Read-only. Nothing is installed, downloaded, or changed.
"""

import glob
import os
import subprocess

def sh(cmd, timeout=120):
    try:
        p = subprocess.run(cmd, shell=True, capture_output=True, text=True,
                           timeout=timeout)
        return ((p.stdout or "") + (p.stderr or "")).strip()
    except subprocess.TimeoutExpired:
        return "(超时)"

def section(n, t):
    print("\n" + "=" * 74)
    print("{}. {}".format(n, t))
    print("=" * 74)

section(1, "dist-info:这个包是从哪个 URL 装进来的")
for d in glob.glob("/opt/conda/lib/python3.12/site-packages/mcoplib*.dist-info"):
    print("  " + d)
    for f in ("direct_url.json", "INSTALLER", "WHEEL", "METADATA"):
        p = os.path.join(d, f)
        if not os.path.exists(p):
            continue
        with open(p, errors="replace") as fh:
            body = fh.read()
        if f == "METADATA":
            body = "\n".join(l for l in body.splitlines()
                             if l.startswith(("Name:", "Version:", "Home-page:",
                                              "Download-URL:", "Project-URL:")))
        print("    --- {}\n{}".format(
            f, "\n".join("      " + l for l in body.strip().splitlines()[:12])))
if not glob.glob("/opt/conda/lib/python3.12/site-packages/mcoplib*.dist-info"):
    print("  没找到 dist-info —— 可能不是 pip 装的")

section(2, "pip 的全部配置文件(全局配置之外还有没有别的)")
print(sh("python3 -m pip config debug 2>&1 | head -40"))
print("\n环境变量:")
print(sh("env | grep -i '^PIP_\\|INDEX_URL\\|EXTRA_INDEX' || echo '  (无)'"))
for p in ("/etc/pip.conf", os.path.expanduser("~/.pip/pip.conf"),
          os.path.expanduser("~/.config/pip/pip.conf"),
          "/opt/conda/pip.conf"):
    if os.path.exists(p):
        print("\n{}:".format(p))
        print(sh("cat " + p))

section(3, "磁盘上有没有 mcoplib 的 wheel(文件名会带版本序列)")
print(sh("find / -maxdepth 6 -name 'mcoplib*.whl' 2>/dev/null | head -20 "
         "|| true") or "  (没找到)")
print("\npip 缓存:")
print(sh("python3 -m pip cache list 2>/dev/null | grep -i mcoplib | head "
         "|| echo '  (缓存里没有)'"))

section(4, "MACA 安装目录里有没有厂商源的线索")
print(sh("ls /opt/maca 2>/dev/null | head -20 || echo '  (无 /opt/maca)'"))
print(sh("grep -rIl 'index-url\\|pypi\\|mcoplib' /opt/maca/*.txt /opt/maca/*.md "
         "/opt/maca/conf 2>/dev/null | head -10 || true"))

section(5, "这台机器能不能出网(决定源码重建这条路走不走得通)")
for host in ("github.com", "pypi.org", "mirrors.aliyun.com"):
    rc = sh("timeout 8 curl -sS -o /dev/null -w '%{{http_code}}' "
            "https://{} 2>&1 | tail -1".format(host))
    print("  https://{:<22} -> {}".format(host, rc))

section(6, "对照:上一次的查询错在哪")
print("""  pip 指向 mirrors.aliyun.com(通用 PyPI 镜像),而装着的是
  0.4.6+maca3.7.1.5.torch2.8 —— 带 +maca 本地版本段的包不可能来自通用镜像。
  所以那句 "No matching distribution found" 说明的是索引不对,
  不是 MetaX 没发布。""")
print("\n[RESULT] PROVENANCE_DONE")
