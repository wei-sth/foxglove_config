#!/usr/bin/env python3
import os
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


def read_total_csv(path: str) -> pd.Series:
    df = pd.read_csv(path)
    if "total_ms" not in df.columns:
        raise ValueError(f"{path} missing column 'total_ms'. Columns: {list(df.columns)}")
    s = df["total_ms"].dropna()
    s = pd.to_numeric(s, errors="coerce").dropna()
    return s


def default_label(path: str) -> str:
    return Path(path).stem


def main():
    # before optimization
    csv_a = "/home/weizh/data/slam_total_20260317_165054.csv"  # slam_total_20260317_165054 | slam_total_20260317_170510
    # after optimization
    csv_b = "/home/weizh/data/slam_total_20260317_175011.csv"
    out_path = "/home/weizh/data/compare_slam_total.png"

    label_a = default_label(csv_a)
    label_b = default_label(csv_b)
    title = "slam total(ms) comparison"
    dpi = 160
    a = read_total_csv(csv_a)
    b = read_total_csv(csv_b)
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(range(len(a)), a.values, linewidth=1.2, label=f"{label_a} (n={len(a)})")
    ax.plot(range(len(b)), b.values, linewidth=1.2, label=f"{label_b} (n={len(b)})")
    ax.set_xlabel("frame index")
    ax.set_ylabel("total_ms")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)

    print(f"Saved plot: {out_path}")
    

if __name__ == "__main__":
    main()
