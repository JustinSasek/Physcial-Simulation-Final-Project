#!/usr/bin/env python3
import csv
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt


def plot_csv(path: Path) -> None:
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    if not rows:
        print(f"No rows in {path}")
        return
    # determine parameter columns (exclude iteration, loss and gt.*)
    fieldnames = reader.fieldnames or []
    params = [
        f
        for f in fieldnames
        if f not in ("iteration", "loss") and not f.startswith("gt.")
    ]
    gt_map = {f[3:]: None for f in fieldnames if f.startswith("gt.")}
    for g in [f for f in fieldnames if f.startswith("gt.")]:
        key = g[3:]
        val = rows[0].get(g, "")
        try:
            gt_map[key] = float(val) if val != "" and val.lower() != "none" else None
        except Exception:
            gt_map[key] = None

    iters = [
        int(r.get("iteration", i))
        for i, r in enumerate(rows)
        if int(r.get("iteration", i)) >= 0
    ]
    if not iters:
        print(f"No iterations found in {path}")
        return

    for param in params:
        ys = []
        xs = []
        for r in rows:
            try:
                it = int(r.get("iteration", -1))
            except Exception:
                continue
            if it < 0:
                continue
            v = r.get(param, "")
            try:
                y = float(v) if v != "" else None
            except Exception:
                y = None
            xs.append(it)
            ys.append(y)
        if not xs:
            continue
        plt.figure()
        plt.plot(xs, ys, marker="o", label=param)
        gt = gt_map.get(param)
        if gt is not None:
            plt.axhline(gt, color="gray", linestyle="--", label=f"gt {gt}")
        plt.xlabel("iteration")
        plt.ylabel(param)
        plt.title(f"{path.name} — {param}")
        plt.legend()
        out = path.with_suffix("").with_name(
            path.stem + "_" + param.replace("/", "_") + ".png"
        )
        plt.tight_layout()
        plt.savefig(out, dpi=150)
        plt.close()
        print(f"Wrote {out}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: plot_optimization.py <file1.csv> [file2.csv ...]")
        sys.exit(2)
    for p in sys.argv[1:]:
        plot_csv(Path(p))
