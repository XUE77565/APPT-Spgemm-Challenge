#!/usr/bin/env python3
"""
Crawl metadata for the SuiteSparse Matrix Collection (sparse.tamu.edu).

The collection index renders all matrices in one HTML table when a large
`per_page` is requested, so a SINGLE polite request to
    https://sparse.tamu.edu/?page=1&per_page=3000
returns every matrix (~2904).  No .mtx files are downloaded — only the index
metadata (Id, Name, Group, Rows, Cols, Nonzeros, Kind, Date).

Derived columns added for later analysis:
    num_entries  = rows * cols
    density      = nnz / num_entries          (fraction of nonzeros)
    sparsity_pct = (1 - density) * 100
    nnz_per_row  = nnz / rows
    is_square    = rows == cols
    year         = year parsed from the Date column
    url_mm       = canonical Matrix-MMarket .tar.gz download URL (for reference)

Usage:
    .venv/bin/python scrape_suitesparse.py [--per-page 3000] [--out suitesparse_metadata.csv]
"""

import argparse
import csv
import re
import sys
import time

import requests
from bs4 import BeautifulSoup

BASE = "https://sparse.tamu.edu"
UA = ("Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/120 Safari/537.36")


def fetch(session, params, retries=4, timeout=60):
    """GET BASE with retries + exponential backoff. Returns HTML text."""
    last = None
    for i in range(retries):
        try:
            r = session.get(BASE, params=params, timeout=timeout)
            r.raise_for_status()
            return r.text
        except Exception as e:  # noqa: BLE001
            last = e
            wait = 2 ** i
            print(f"  fetch error ({e}); retry {i + 1}/{retries} in {wait}s",
                  file=sys.stderr)
            time.sleep(wait)
    raise RuntimeError(f"fetch failed after {retries} retries: {last}")


def parse_table(html):
    """Parse the index HTML table -> list of dict rows (raw, no enrichment)."""
    soup = BeautifulSoup(html, "html.parser")
    tbl = soup.find("table")
    if tbl is None:
        raise RuntimeError("no <table> found in page")
    rows = tbl.find_all("tr")

    header = [c.get_text(strip=True) for c in rows[0].find_all(["th", "td"])]

    out = []
    for tr in rows[1:]:
        cells = tr.find_all("td")
        if len(cells) < 8:
            continue

        def txt(i):
            return cells[i].get_text(" ", strip=True)

        id_raw = txt(0)
        if not re.fullmatch(r"\s*\d+\s*", id_raw):
            continue  # not a data row

        a = cells[1].find("a", href=True)
        detail = a["href"].strip() if a else ""

        out.append({
            "id": int(id_raw),
            "name": txt(1),
            "group": txt(2),
            "rows": int(txt(3).replace(",", "")),
            "cols": int(txt(4).replace(",", "")),
            "nnz": int(txt(5).replace(",", "")),
            "kind": txt(6),
            "date": txt(7),
            "detail_url": detail,
        })
    return header, out


def enrich(rows):
    """Add derived columns in place."""
    for r in rows:
        entries = r["rows"] * r["cols"]
        r["num_entries"] = entries
        r["density"] = r["nnz"] / entries if entries else 0.0
        r["sparsity_pct"] = (1.0 - r["density"]) * 100.0
        r["nnz_per_row"] = r["nnz"] / r["rows"] if r["rows"] else 0.0
        r["is_square"] = bool(r["rows"] == r["cols"])
        m = re.search(r"(\d{4})", r["date"])
        r["year"] = int(m.group(1)) if m else None
        r["url_mm"] = f"{BASE}/MM/{r['group']}/{r['name']}.tar.gz"
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--per-page", type=int, default=3000,
                    help="rows per request (default 3000 = single bulk fetch)")
    ap.add_argument("--out", default="suitesparse_metadata.csv")
    ap.add_argument("--paginate-fallback", action="store_true",
                    help="if the bulk fetch looks truncated, paginate per_page=100")
    ap.add_argument("--square-only", action="store_true",
                    help="drop rectangular matrices; keep only rows==cols")
    args = ap.parse_args()

    session = requests.Session()
    session.headers.update({"User-Agent": UA, "Accept-Language": "en"})

    print(f"Fetching {BASE}?page=1&per_page={args.per_page} ...")
    html = fetch(session, {"page": "1", "per_page": str(args.per_page)})
    _, rows = parse_table(html)
    print(f"  parsed {len(rows)} matrices from bulk page")

    if args.paginate_fallback and len(rows) < 2900:
        print("Bulk looks small; paginating per_page=100 as fallback ...")
        have = {r["id"] for r in rows}
        page = 2
        while page <= 300:
            html = fetch(session, {"page": str(page), "per_page": "100"})
            _, prows = parse_table(html)
            fresh = [r for r in prows if r["id"] not in have]
            if not fresh:
                break
            rows.extend(fresh)
            have |= {r["id"] for r in fresh}
            print(f"  page {page}: +{len(fresh)} (total {len(rows)})")
            page += 1
            time.sleep(0.5)

    rows = enrich(rows)
    rows.sort(key=lambda r: r["id"])

    if args.square_only:
        before = len(rows)
        rows = [r for r in rows if r["is_square"]]
        print(f"  --square-only: kept {len(rows)} square "
              f"(dropped {before - len(rows)} rectangular)")

    cols = ["id", "name", "group", "rows", "cols", "nnz", "kind", "date", "year",
            "num_entries", "density", "sparsity_pct", "nnz_per_row", "is_square",
            "detail_url", "url_mm"]
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)

    print(f"Saved {len(rows)} matrices -> {args.out}")
    # quick sanity peek
    if rows:
        tot_nnz = sum(r["nnz"] for r in rows)
        sq = sum(1 for r in rows if r["is_square"])
        print(f"  total nnz across collection: {tot_nnz:,}")
        print(f"  square: {sq}   rectangular: {len(rows) - sq}")
        big = max(rows, key=lambda r: r["num_entries"])
        print(f"  largest by rows*cols: {big['name']} "
              f"({big['rows']:,} x {big['cols']:,}, nnz={big['nnz']:,})")


if __name__ == "__main__":
    main()
