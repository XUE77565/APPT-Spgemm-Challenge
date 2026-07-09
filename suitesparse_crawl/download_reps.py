#!/usr/bin/env python3
"""
把 representatives.csv 里的代表矩阵下载为 .mtx 到 ../data/rep/。

网络要点(本环境实测):
  - 文件真正主机 = http://sparse-files.engr.tamu.edu/MM/<group>/<name>.tar.gz
    (sparse.tamu.edu / herokuapp.com 的 MM 链接都是 301 跳转到这里)
  - 必须经本机代理 127.0.0.1:7890;直连会被 ICT 网关(gw.ict.ac.cn)劫持。
  - tar.gz 里 .mtx 在 <name>/<name>.mtx。
"""

import csv
import os
import sys
import tarfile
import tempfile
from pathlib import Path

import requests

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
REPS_CSV = HERE / "representatives.csv"
OUT_DIR = REPO / "data" / "rep"
FILE_HOST = "http://sparse-files.engr.tamu.edu/MM"
PROXY = (os.environ.get("SSC_PROXY") or os.environ.get("HTTPS_PROXY")
         or os.environ.get("https_proxy") or "http://127.0.0.1:7890")
PROXIES = {"http": PROXY, "https": PROXY}
UA = ("Mozilla/5.0 (X11; Linux x64) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/120 Safari/537.36")


def download_one(name, group, dst):
    url = f"{FILE_HOST}/{group}/{name}.tar.gz"
    with requests.get(url, headers={"User-Agent": UA}, proxies=PROXIES,
                      stream=True, timeout=180, allow_redirects=True) as resp:
        resp.raise_for_status()
        with tempfile.NamedTemporaryFile(delete=False, suffix=".tar.gz") as tf:
            total = 0
            for chunk in resp.iter_content(1 << 16):
                if chunk:
                    tf.write(chunk)
                    total += len(chunk)
            tmppath = tf.name
    try:
        with tarfile.open(tmppath, "r:gz") as tar:
            members = [m for m in tar.getmembers()
                       if m.isfile() and m.name.endswith(".mtx")]
            target = next((m for m in members
                           if Path(m.name).name == f"{name}.mtx"), None)
            if target is None and members:
                target = members[0]
            if target is None:
                raise RuntimeError("tar.gz 里找不到 .mtx")
            f = tar.extractfile(target)
            dst.write_bytes(f.read())
    finally:
        os.unlink(tmppath)
    return total


def main():
    if not REPS_CSV.exists():
        print(f"找不到 {REPS_CSV};先运行 classify_by_sparsity.py", file=sys.stderr)
        sys.exit(1)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    reps = list(csv.DictReader(open(REPS_CSV)))
    print(f"下载 {len(reps)} 个代表 -> {OUT_DIR}  (经代理 {PROXY})\n")

    ok, skipped, fail = [], [], []
    for i, r in enumerate(reps, 1):
        name, group, cls = r["name"], r["group"], r["class"]
        dst = OUT_DIR / f"{name}.mtx"
        if dst.exists() and dst.stat().st_size > 0:
            print(f"  [{i:>2}/{len(reps)}] skip  {name:<26}(已存在)")
            skipped.append(name)
            continue
        try:
            tgz = download_one(name, group, dst)
            print(f"  [{i:>2}/{len(reps)}] ok    {name:<26}"
                  f"{dst.stat().st_size:>11,} B  (tar.gz {tgz:>9,} B)  [{cls}]")
            ok.append(name)
        except Exception as e:  # noqa: BLE001
            print(f"  [{i:>2}/{len(reps)}] FAIL  {name:<26}{e}")
            fail.append((name, str(e)))

    print(f"\n完成: {len(ok)} 成功 / {len(skipped)} 已存在 / {len(fail)} 失败")
    print(f"输出目录: {OUT_DIR}")
    if fail:
        print("失败列表:")
        for n, e in fail:
            print(f"  - {n}: {e}")


if __name__ == "__main__":
    main()
