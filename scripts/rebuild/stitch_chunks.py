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
    # Drop non-hour-aligned stragglers (e.g. Binance's 2018-02-09 09:28 post-outage bar)
    misaligned = seg.index[(seg.index.minute != 0) | (seg.index.second != 0)]
    if len(misaligned):
        print(f"dropping {len(misaligned)} non-hour-aligned bars: {list(misaligned[:3])}")
        seg = seg[(seg.index.minute == 0) & (seg.index.second == 0)]

    # Completeness check: the segment must cover every RAW kline bar after the
    # global warmup. Gaps inherited from the exchange (Binance outages) are
    # expected and fine — V12 has 13 of its own; the backtester tolerates them.
    klines = pd.read_parquet(
        OUT_DIR.parents[1] / "data/cache/binance_vision/klines/BTCUSDT_1h.parquet")
    kidx = klines.index.tz_localize(None)
    kidx = kidx[(kidx.minute == 0) & (kidx.second == 0)]
    expected = kidx[kidx >= seg.index[0]]
    missing = expected.difference(seg.index)
    extra = seg.index.difference(expected)
    print(f"\nstitched: {len(seg)} rows, {seg.shape[1]} cols, {seg.index[0]} → {seg.index[-1]}")
    print(f"duplicates removed: {dupes}; missing vs raw klines: {len(missing)}; extra: {len(extra)}")
    if len(missing) or len(extra):
        print("missing:", list(missing[:5]), "extra:", list(extra[:5]))
        raise SystemExit("ABORT: stitched segment does not match raw kline coverage")

    seg.to_parquet(OUT)
    print(f"WROTE {OUT}")


if __name__ == "__main__":
    main()
