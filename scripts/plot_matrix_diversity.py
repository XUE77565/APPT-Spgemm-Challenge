#!/usr/bin/env python3
"""Single-column paper figure for the Ocean square-matrix benchmark set."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.ticker import LogLocator, NullFormatter
import numpy as np


BLUE = "#4D87B5"
ORANGE = "#D17A32"
GREEN = "#659A4B"
RED = "#C0504D"
INK = "#202020"
MID = "#666666"
GRID = "#D9D9D9"
WHITE = "#FFFFFF"

REPRESENTATIVES = (
    ("circuit_4", "banded / coupled", BLUE),
    ("pf2177", "block structured", ORANGE),
    ("email-Enron", "irregular network", GREEN),
    ("Reuters911", "dense cross-linked", RED),
)


def parse_args() -> argparse.Namespace:
    repo = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=repo / "ocean" / "utils" / "square.csv")
    parser.add_argument("--data-dir", type=Path, default=repo / "data" / "ocean" / "square")
    parser.add_argument("--output-dir", type=Path, default=repo / "fig")
    return parser.parse_args()


def read_manifest(path: Path) -> list[dict[str, float | str]]:
    rows: list[dict[str, float | str]] = []
    with path.open(newline="", encoding="utf-8") as stream:
        for raw in csv.DictReader(stream):
            n = int(raw["rows"])
            cols = int(raw["cols"])
            nnz = int(raw["entries"])
            if n != cols:
                raise ValueError(f"Expected square matrix, got {raw['Name']}: {n} x {cols}")
            rows.append(
                {
                    "name": raw["Name"],
                    "n": float(n),
                    "nnz": float(nnz),
                    "density": 100.0 * nnz / (n * n),
                }
            )
    if len(rows) != 337:
        raise ValueError(f"Expected 337 square matrices, found {len(rows)}")
    return rows


def binned_pattern(path: Path, bins: int = 144) -> np.ndarray:
    """Aggregate a Matrix Market sparsity pattern without materializing the matrix."""
    with path.open("rt", encoding="utf-8", errors="replace") as stream:
        banner = stream.readline().lower()
        if "matrixmarket matrix coordinate" not in banner:
            raise ValueError(f"Unsupported Matrix Market banner in {path}")
        symmetric = any(kind in banner for kind in ("symmetric", "hermitian", "skew-symmetric"))

        line = stream.readline()
        while line.startswith("%"):
            line = stream.readline()
        nrows, ncols, _ = map(int, line.split()[:3])

        counts = np.zeros((bins, bins), dtype=np.uint32)
        for line in stream:
            fields = line.split()
            if len(fields) < 2:
                continue
            row = int(fields[0]) - 1
            col = int(fields[1]) - 1
            br = min(bins - 1, row * bins // nrows)
            bc = min(bins - 1, col * bins // ncols)
            counts[br, bc] += 1
            if symmetric and row != col:
                counts[bc, br] += 1
    return counts


def add_panel_label(fig: plt.Figure, ax: plt.Axes, label: str, *, x: float | None = None) -> None:
    box = ax.get_position()
    fig.text(box.x0 if x is None else x, box.y1 + 0.018, label, ha="left", va="bottom", weight="bold", fontsize=7.6)


def draw_scatter(fig: plt.Figure, ax: plt.Axes, rows: list[dict[str, float | str]]) -> None:
    n = np.array([float(row["n"]) for row in rows])
    density = np.array([float(row["density"]) for row in rows])
    by_name = {str(row["name"]): row for row in rows}

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.scatter(n, density, s=8.8, color="#87939C", alpha=0.70, linewidths=0, rasterized=False, zorder=2)

    for index, (name, _, color) in enumerate(REPRESENTATIVES, 1):
        row = by_name[name]
        x = float(row["n"])
        y = float(row["density"])
        ax.scatter([x], [y], s=38, marker="s", color=color, edgecolors=WHITE, linewidths=0.75, zorder=4)
        ax.annotate(
            str(index),
            (x, y),
            xytext=(4.0, 3.5),
            textcoords="offset points",
            color=color,
            fontsize=8.2,
            weight="bold",
            zorder=5,
        )

    ax.set_xlim(3.6e3, 4.0e7)
    ax.set_ylim(3.5e-6, 2.0e1)
    ax.set_xlabel("Matrix dimension, n", labelpad=1.5)
    ax.set_ylabel(r"Density, nnz / n$^2$ (%)", labelpad=1.5)
    ax.xaxis.set_major_locator(LogLocator(base=10, numticks=6))
    ax.yaxis.set_major_locator(LogLocator(base=10, numticks=8))
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.yaxis.set_minor_formatter(NullFormatter())
    ax.grid(True, which="major", color=GRID, linewidth=0.42, zorder=0)
    ax.tick_params(which="both", width=0.55, length=2.2, pad=1.2)
    ax.tick_params(which="minor", length=1.2)
    for spine in ax.spines.values():
        spine.set_linewidth(0.65)
        spine.set_color(INK)

    add_panel_label(fig, ax, "(a) Scale and density", x=0.035)


def draw_patterns(fig: plt.Figure, axes: list[plt.Axes], data_dir: Path, rows: list[dict[str, float | str]]) -> None:
    for index, (ax, (name, _, color)) in enumerate(zip(axes, REPRESENTATIVES), 1):
        path = data_dir / f"{name}.mtx"
        if not path.exists():
            raise FileNotFoundError(path)
        counts = binned_pattern(path)
        image = np.log1p(counts.astype(float))
        positive = image[image > 0]
        vmax = float(np.percentile(positive, 98)) if positive.size else 1.0
        cmap = LinearSegmentedColormap.from_list(f"matrix-{index}", [WHITE, color])
        ax.imshow(image, cmap=cmap, origin="upper", interpolation="nearest", vmin=0, vmax=max(vmax, 1.0))
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_linewidth(0.65)
            spine.set_color(INK)
        ax.text(
            0.025,
            0.96,
            f"{index}  {name}",
            transform=ax.transAxes,
            ha="left",
            va="top",
            color=color,
            fontsize=6.8,
            weight="bold",
            bbox={"facecolor": WHITE, "edgecolor": "none", "alpha": 0.88, "pad": 0.6},
        )

    add_panel_label(fig, axes[0], "(b) Sparsity patterns")


def main() -> None:
    args = parse_args()
    rows = read_manifest(args.manifest)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 7.0,
            "axes.labelsize": 7.4,
            "axes.titlesize": 7.4,
            "xtick.labelsize": 6.5,
            "ytick.labelsize": 6.5,
            "axes.edgecolor": INK,
            "axes.labelcolor": INK,
            "text.color": INK,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )

    fig = plt.figure(figsize=(3.50, 2.15), facecolor=WHITE)
    scatter_ax = fig.add_axes([0.14, 0.23, 0.375, 0.63])

    # Independent right-hand geometry lets the 2x2 pattern block extend down
    # to the x-axis-title baseline while preserving square matrix thumbnails.
    right_x0, right_x1, gap_x = 0.530, 0.985, 0.015
    cell_w = (right_x1 - right_x0 - gap_x) / 2.0
    cell_h = cell_w * 3.50 / 2.15
    bottom_y, top_y = 0.120, 0.86 - cell_h
    pattern_axes = [
        fig.add_axes([right_x0, top_y, cell_w, cell_h]),
        fig.add_axes([right_x0 + cell_w + gap_x, top_y, cell_w, cell_h]),
        fig.add_axes([right_x0, bottom_y, cell_w, cell_h]),
        fig.add_axes([right_x0 + cell_w + gap_x, bottom_y, cell_w, cell_h]),
    ]

    draw_scatter(fig, scatter_ax, rows)
    draw_patterns(fig, pattern_axes, args.data_dir, rows)

    stem = args.output_dir / "matrix_diversity"
    fig.savefig(stem.with_suffix(".pdf"), facecolor=WHITE)
    fig.savefig(stem.with_suffix(".svg"), facecolor=WHITE)
    fig.savefig(stem.with_suffix(".png"), dpi=600, facecolor=WHITE)
    plt.close(fig)

    n = [float(row["n"]) for row in rows]
    density = [float(row["density"]) for row in rows]
    print(
        f"Saved {stem}.{{pdf,svg,png}} | matrices={len(rows)} | "
        f"n=[{min(n):.0f}, {max(n):.0f}] | density=[{min(density):.6g}, {max(density):.6g}]%"
    )


if __name__ == "__main__":
    main()
