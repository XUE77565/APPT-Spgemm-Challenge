#!/usr/bin/env python3
"""下载 dispatcher 重拟合的【不相交训练集】:SuiteSparse 方阵,与两个测试集无交集。

  测试集 A = first100 (id 1..100)
  测试集 B = ocean337 (data/ocean/square/,Ocean benchmark)
  训练集   = id 101..700 的方阵,排除 A/B 同名,nnz ≤ 2000 万、n ≤ 400 万,
             按 nnz 对数分层采样 N_TRAIN 个 → data/train_disp/

输出 data/train_disp/<name>.mtx + train_disp_meta.csv(id,name,group,rows,nnz)。
经本机代理下载(同 download_first100.py)。
用法:.venv/bin/python suitesparse_crawl/download_train.py [N_TRAIN]   (默认 220)
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
OUT_DIR = HERE.parent / "data" / "train_disp"

MAX_NNZ = 20_000_000      # 存储 nnz 上限(对称阵展开 ×2 后 ≤ 4000 万)
MAX_DIM = 4_000_000
ID_LO, ID_HI = 101, 700


def fetch_index():
    print(f"抓元数据 {BASE}?page=1&per_page=3000 ...")
    r = requests.get(BASE, params={"page": "1", "per_page": "3000"},
                     headers={"User-Agent": UA}, timeout=120)
    r.raise_for_status()
    soup = BeautifulSoup(r.text, "html.parser")
    rows = soup.find("table").find_all("tr")
    out = []
    for tr in rows[1:]:
        cells = tr.find_all("td")
        if len(cells) < 8:
            continue
        id_raw = cells[0].get_text(strip=True)
        if not re.fullmatch(r"\d+", id_raw):
            continue
        try:
            rows_n = int(cells[3].get_text(strip=True).replace(",", ""))
            cols_n = int(cells[4].get_text(strip=True).replace(",", ""))
            nnz = int(cells[5].get_text(strip=True).replace(",", ""))
        except ValueError:
            continue
        out.append({"id": int(id_raw), "name": cells[1].get_text(" ", strip=True),
                    "group": cells[2].get_text(" ", strip=True),
                    "rows": rows_n, "cols": cols_n, "nnz": nnz})
    return out


def download_one(name, group, dst):
    url = f"{FILE_HOST}/{group}/{name}.tar.gz"
    with requests.get(url, headers={"User-Agent": UA}, proxies=PROXIES,
                      stream=True, timeout=300, allow_redirects=True) as resp:
        resp.raise_for_status()
        with tempfile.NamedTemporaryFile(delete=False, suffix=".tar.gz") as tf:
            for chunk in resp.iter_content(1 << 16):
                if chunk:
                    tf.write(chunk)
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


def main():
    n_train = int(sys.argv[1]) if len(sys.argv) > 1 else 220
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    test_names = set()
    f100 = HERE / "first100_meta.csv"
    if f100.exists():
        test_names |= {r["name"] for r in csv.DictReader(open(f100))}
    oc = HERE.parent / "data" / "ocean" / "square"
    if oc.exists():
        test_names |= {p.stem for p in oc.glob("*.mtx")}

    meta = fetch_index()
    pool = []
    skipped = {"testset": 0, "nonsquare": 0, "toobig": 0}
    seen = set()
    for m in sorted(meta, key=lambda x: x["id"]):
        if not (ID_LO <= m["id"] <= ID_HI):
            continue
        if m["name"] in test_names or m["name"] in seen:
            skipped["testset"] += 1
            continue
        seen.add(m["name"])
        if m["rows"] != m["cols"]:
            skipped["nonsquare"] += 1
            continue
        if m["rows"] > MAX_DIM or m["nnz"] > MAX_NNZ:
            skipped["toobig"] += 1
            continue
        pool.append(m)
    print(f"id {ID_LO}..{ID_HI} 方阵池 {len(pool)} 个(排除 {skipped});按 nnz 分层采样 {n_train}")

    # 对数分层:按 nnz 排序后等距采样,覆盖稀疏→稠密谱
    pool.sort(key=lambda m: m["nnz"])
    if len(pool) > n_train:
        step = len(pool) / n_train
        sample = [pool[int(i * step)] for i in range(n_train)]
    else:
        sample = pool

    with open(HERE / "train_disp_meta.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["id", "name", "group", "rows", "nnz"])
        w.writeheader()

    ok = skip = fail = 0
    for i, m in enumerate(sample, 1):
        dst = OUT_DIR / f"{m['name']}.mtx"
        if dst.exists() and dst.stat().st_size > 0:
            skip += 1
            continue
        try:
            download_one(m["name"], m["group"], dst)
            ok += 1
            print(f"[{i:>3}/{len(sample)}] ok    {m['name']:<28} n={m['rows']:<9} nnz={m['nnz']}")
        except Exception as e:
            fail += 1
            print(f"[{i:>3}/{len(sample)}] FAIL  {m['name']:<28} {e}")

    # 重写 meta(只含实际到手的)
    got = [m for m in sample if (OUT_DIR / f"{m['name']}.mtx").exists()]
    with open(HERE / "train_disp_meta.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["id", "name", "group", "rows", "nnz"])
        w.writeheader()
        w.writerows(got)
    print(f"\n完成:ok={ok} skip={skip} fail={fail};meta={HERE/'train_disp_meta.csv'}")


if __name__ == "__main__":
    main()
