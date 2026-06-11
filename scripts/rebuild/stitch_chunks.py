#!/usr/bin/env python3
"""Stitch parallel replay chunks into segment_lfc.parquet.

Each chunk re-computed OVERLAP bars before its true start (warm state);
those rows are dropped using the chunk's .discard_before.txt marker.
Validates the stitched index is gap-free hourly before writing.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

OUT_DIR = Path(__file__).resolve().parents[2] / "results/rebuild"
OUT = OUT_DIR / "segment_lfc.parquet"


def main():
    frames = []
    for chunk in sorted(OUT_DIR.glob("segment_chunk_*.parquet"),
                        key=lambda p: int(p.stem.split("_")[-1])):
        df = pd.read_parquet(chunk)
        marker = chunk.with_suffix(".discard_before.txt")
        if marker.exists():
            cutoff = pd.Timestamp(marker.read_text().strip())
            before = len(df)
            df = df.loc[df.index >= cutoff]
            print(f"{chunk.name}: {before} rows → {len(df)} after discarding warm-state overlap")
        else:
            print(f"{chunk.name}: {len(df)} rows (no discard marker)")
        frames.append(df)

    seg = pd.concat(frames).sort_index()
    dupes = seg.index.duplicated().sum()
    seg = seg[~seg.index.duplicated(keep="last")]

    # Gap check: hourly continuity
    diffs = seg.index.to_series().diff().dropna()
    gaps = diffs[diffs != pd.Timedelta(hours=1)]
    print(f"\nstitched: {len(seg)} rows, {seg.shape[1]} cols, {seg.index[0]} → {seg.index[-1]}")
    print(f"duplicates removed: {dupes}; non-hourly gaps: {len(gaps)}")
    if len(gaps):
        print(gaps.head(10).to_string())
        raise SystemExit("ABORT: gaps in stitched segment")

    seg.to_parquet(OUT)
    print(f"WROTE {OUT}")


if __name__ == "__main__":
    main()
