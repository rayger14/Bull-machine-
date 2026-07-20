#!/usr/bin/env python3
"""V16 deep-daily-context patch — recompute the Wyckoff feature family with
real daily history (2026-07-20 upgrade).

V15's daily Wyckoff layer resampled the 1,000-bar 1H buffer → ~42 daily bars,
of which the detectors' 20-30 bar rolling stats burn most as warmup. This
patch replays ONLY _wyckoff_features() with the deep daily buffer spliced in
(completed days from full history, capped 300, + current partial day), and a
90-day context/score horizon — exactly mirroring the live change in
bin/live/live_feature_computer.py.

Output V16 = V15 with the wyckoff-family columns replaced.

Usage:
  python3 scripts/rebuild/patch_v16_daily.py --chunk 1 12
  python3 scripts/rebuild/patch_v16_daily.py --limit 300      # smoke
  python3 scripts/rebuild/patch_v16_daily.py --stitch
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

STORE = REPO / "data/features_mtf/BTC_1H_FEATURES_V15_STRUCTURE.parquet"
OUT_DIR = REPO / "results/rebuild/v16_daily"
V16 = REPO / "data/features_mtf/BTC_1H_FEATURES_V16_DEEPDAILY.parquet"
WARMUP = 1000
DEEP_DAILY = 300
CHECKPOINT_EVERY = 2000

logging.basicConfig(level=logging.WARNING)
log = logging.getLogger("v16patch")
log.setLevel(logging.INFO)


def load_ohlcv() -> pd.DataFrame:
    df = pd.read_parquet(STORE, columns=["open", "high", "low", "close", "volume"])
    if df.index.tz is not None:
        df.index = df.index.tz_localize(None)
    return df


def daily_series(ohlcv: pd.DataFrame) -> pd.DataFrame:
    d = ohlcv.resample("1D").agg({"open": "first", "high": "max", "low": "min",
                                  "close": "last", "volume": "sum"}).dropna()
    return d


def compute_rows(ohlcv: pd.DataFrame, daily: pd.DataFrame, start: int, end: int,
                 out_path: Path) -> pd.DataFrame:
    from bin.live.live_feature_computer import LiveFeatureComputer

    done = None
    if out_path.exists():
        done = pd.read_parquet(out_path)
        if len(done) >= end - start:
            log.info("%s already complete", out_path.name)
            return done
        start = start + len(done)
        log.info("resuming %s at +%d", out_path.name, len(done))

    lfc = LiveFeatureComputer(buffer_size=WARMUP)
    warm_lo = max(0, start - WARMUP)
    if start > warm_lo:
        lfc.ingest_candles(ohlcv.iloc[warm_lo:start])

    rows, t0 = [], time.time()
    idx = ohlcv.index
    for i in range(start, end):
        ts = idx[i]
        bar = ohlcv.iloc[i]
        new_row = pd.DataFrame([{c: float(bar[c]) for c in
                                 ("open", "high", "low", "close", "volume")}],
                               index=[ts])
        lfc._buf = new_row.copy() if lfc._buf is None else pd.concat([lfc._buf, new_row])
        if len(lfc._buf) > lfc.buffer_size:
            lfc._buf = lfc._buf.iloc[-lfc.buffer_size:]

        # deep daily buffer: completed days strictly before ts's day
        cur_day = ts.normalize()
        lfc._daily_buf = daily[daily.index < cur_day].tail(DEEP_DAILY + 20)

        feats: dict = {"__ts": ts}
        feats.update(lfc._wyckoff_features())
        rows.append(feats)

        n = i - start + 1
        if n % CHECKPOINT_EVERY == 0 or i == end - 1:
            chunk = pd.DataFrame(rows).set_index("__ts")
            merged = pd.concat([done, chunk]) if done is not None else chunk
            out_path.parent.mkdir(parents=True, exist_ok=True)
            merged.to_parquet(out_path)
            done, rows = merged, []
            rate = n / (time.time() - t0)
            eta_h = (end - 1 - i) / rate / 3600 if rate > 0 else float("inf")
            log.info("%s: %d/%d (%.1f bars/s, eta %.1fh)",
                     out_path.name, n, end - start, rate, eta_h)
    return done


def stitch(index: pd.Index) -> None:
    chunks = sorted(OUT_DIR.glob("chunk_*.parquet"))
    if not chunks:
        sys.exit("no chunks")
    patch = pd.concat([pd.read_parquet(c) for c in chunks])
    patch = patch[~patch.index.duplicated(keep="last")].sort_index()
    patch = patch.drop(columns=[c for c in patch.columns
                                if patch[c].dtype == object], errors="ignore")
    missing = index.difference(patch.index)
    if len(missing):
        sys.exit(f"incomplete: {len(missing)} rows missing (first {missing[0]})")
    patch = patch.reindex(index)
    store = pd.read_parquet(STORE)
    if store.index.tz is not None:
        store.index = store.index.tz_localize(None)
    replaced = [c for c in patch.columns if c in store.columns]
    added = [c for c in patch.columns if c not in store.columns]
    for c in patch.columns:
        store[c] = patch[c].values
    store.to_parquet(V16)
    print(f"V16 written: {V16} — replaced {len(replaced)}, added {len(added)}")
    for c in ["tf1d_wyckoff_score", "tf1d_wyckoff_bullish_score", "tf1d_daily_bars",
              "tf4h_wyckoff_phase_score", "wyckoff_bullish_score"]:
        if c in store.columns:
            s = store[c].fillna(0)
            print(f"  {c:28s} nonzero={float((s != 0).mean()):6.1%} mean={float(s.mean()):.3f}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--chunk", nargs=2, type=int, metavar=("I", "N"))
    ap.add_argument("--limit", type=int)
    ap.add_argument("--stitch", action="store_true")
    args = ap.parse_args()

    ohlcv = load_ohlcv()
    if args.stitch:
        stitch(ohlcv.index)
        return
    daily = daily_series(ohlcv)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    n = len(ohlcv)
    if args.limit:
        out = OUT_DIR / "smoke.parquet"
        out.unlink(missing_ok=True)
        df = compute_rows(ohlcv, daily, n - args.limit, n, out)
        for c in ["tf1d_wyckoff_score", "tf1d_wyckoff_bullish_score",
                  "tf1d_wyckoff_bearish_score", "tf1d_daily_bars"]:
            if c in df.columns:
                s = df[c].fillna(0)
                print(f"{c:30s} nonzero={float((s != 0).mean()):6.1%} "
                      f"mean={float(s.mean()):.3f} max={float(s.max()):.3f}")
        out.unlink(missing_ok=True)
        return
    if not args.chunk:
        sys.exit("need --chunk I N, --limit, or --stitch")
    i, total = args.chunk
    per = (n + total - 1) // total
    start, end = (i - 1) * per, min(i * per, n)
    log.info("chunk %d/%d: %d..%d", i, total, start, end - 1)
    compute_rows(ohlcv, daily, start, end, OUT_DIR / f"chunk_{i:02d}.parquet")


if __name__ == "__main__":
    main()
