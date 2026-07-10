#!/usr/bin/env python3
"""
下载 SuiteSparse 里 id 1..100 的前 100 个矩阵(.mtx)到 ../data/first100/。
- 元数据:单次 bulk 请求 ?page=1&per_page=3000,解析表格取 id≤100。
- 文件:经本机代理 127.0.0.1:7890,从 sparse-files.engr.tamu.edu/MM/<group>/<name>.tar.gz
  下载并解包出 <name>/<name>.mtx → data/first100/<name>.mtx。
"""
import csv
import os
import re
import sys
import tarfile
import tempfile
from pathlib import Path

import requests
from bs4 import BeautifulSoup

BASE = "https://sparse.tamu.edu"
FILE_HOST = "http://sparse-files.engr.tamu.edu/MM"
UA = ("Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/120 Safari/537.36")
PROXY = (os.environ.get("SSC_PROXY") or os.environ.get("HTTPS_PROXY")
         or os.environ.get("https_proxy") or "http://127.0.0.1:7890")
PROXIES = {"http": PROXY, "https": PROXY}

HERE = Path(__file__).resolve().parent
OUT_DIR = HERE.parent / "data" / "first100"


def fetch_index():
    print(f"抓元数据 {BASE}?page=1&per_page=3000 ...")
    r = requests.get(BASE, params={"page": "1", "per_page": "3000"},
                     headers={"User-Agent": UA}, timeout=120)
    r.raise_for_status()
    soup = BeautifulSoup(r.text, "html.parser")
    tbl = soup.find("table")
    rows = tbl.find_all("tr")
    out = []
    for tr in rows[1:]:
        cells = tr.find_all("td")
        if len(cells) < 8:
            continue
        id_raw = cells[0].get_text(strip=True)
        if not re.fullmatch(r"\d+", id_raw):
            continue
        out.append({"id": int(id_raw), "name": cells[1].get_text(" ", strip=True),
                    "group": cells[2].get_text(" ", strip=True)})
    return out


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
            members = [m for m in tar.getmembers() if m.isfile() and m.name.endswith(".mtx")]
            target = next((m for m in members if Path(m.name).name == f"{name}.mtx"), None)
            if target is None and members:
                target = members[0]
            if target is None:
                raise RuntimeError("tar.gz 里找不到 .mtx")
            dst.write_bytes(tar.extractfile(target).read())
    finally:
        os.unlink(tmppath)
    return total


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    meta = fetch_index()
    first100 = sorted([m for m in meta if m["id"] <= 100], key=lambda m: m["id"])
    print(f"id 1..100 共 {len(first100)} 个;输出 {OUT_DIR}(经代理 {PROXY})\n")

    # 存一份元数据 CSV
    with open(HERE / "first100_meta.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["id", "name", "group"])
        w.writeheader()
        w.writerows(first100)

    ok = skip = fail = 0
    for i, m in enumerate(first100, 1):
        dst = OUT_DIR / f"{m['name']}.mtx"
        if dst.exists() and dst.stat().st_size > 0:
            print(f"[{i:>3}/100] skip  {m['name']:<24}(已存在)")
            skip += 1
            continue
        try:
            tgz = download_one(m["name"], m["group"], dst)
            print(f"[{i:>3}/100] ok    {m['name']:<24}{dst.stat().st_size:>11,} B  "
                  f"(tar.gz {tgz:>9,} B)  [{m['group']}]")
            ok += 1
        except Exception as e:  # noqa: BLE001
            print(f"[{i:>3}/100] FAIL  {m['name']:<24}{e}")
            fail += 1
    print(f"\n完成: {ok} 成功 / {skip} 已存在 / {fail} 失败 → {OUT_DIR}")
    print(f"元数据: suitesparse_crawl/first100_meta.csv")


if __name__ == "__main__":
    main()
