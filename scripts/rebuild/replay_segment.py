#!/usr/bin/env python3
"""Replay Binance 1h klines through LiveFeatureComputer to extend the feature
store past 2024-12-31 (its current end) using the LIVE feature path.

External touch points are replaced with HISTORICAL data:
  - MacroDataFetcher.get_features  → per-day values from prefetched yfinance
    daily history (90d rolling z-scores) + alternative.me fear/greed history
  - _binance_futures_features      → per-hour values from data.binance.vision
    metrics/funding (same pipeline as scripts/data/backfill_binance_vision_derivatives.py)
  - Dominance / eth_btc / mcap     → unavailable historically (free) → None
    (champion-critical features do not consume them)

Checkpoints every 2000 bars; resumable. Output:
  results/rebuild/segment_lfc.parquet   (2024-06 warmup tail → present)

Usage:
  python3 scripts/rebuild/replay_segment.py            # full run (resumable)
  python3 scripts/rebuild/replay_segment.py --limit 50 # smoke test
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts/data"))

KLINES = REPO / "data/cache/binance_vision/klines/BTCUSDT_1h.parquet"
MACRO_CACHE = REPO / "data/cache/macro_daily_history.parquet"
DERIV_CACHE = REPO / "data/cache/derivatives_hourly_2024_2026.parquet"
OUT_DIR = REPO / "results/rebuild"
OUT = OUT_DIR / "segment_lfc.parquet"
WARMUP = 1000  # == WARMUP_CANDLES in coinbase_runner.py (live parity)

logging.basicConfig(level=logging.WARNING)
log = logging.getLogger("replay")
log.setLevel(logging.INFO)


# ----------------------------------------------------------------------------
# Macro history (daily)
# ----------------------------------------------------------------------------
def build_macro_history() -> pd.DataFrame:
    if MACRO_CACHE.exists():
        return pd.read_parquet(MACRO_CACHE)

    import requests
    import yfinance as yf

    tickers = {"VIX": "^VIX", "DXY": "DX-Y.NYB", "YIELD_10Y": "^TNX",
               "YIELD_5Y": "^FVX", "GOLD": "GC=F", "OIL": "CL=F"}
    frames = {}
    for name, tk in tickers.items():
        h = yf.download(tk, start="2023-10-01", progress=False, auto_adjust=True)
        s = h["Close"]
        if isinstance(s, pd.DataFrame):
            s = s.iloc[:, 0]
        frames[name] = s
        log.info("macro %s: %d days", name, len(s))
    df = pd.DataFrame(frames)
    df.index = pd.to_datetime(df.index).tz_localize(None)
    df = df.ffill()

    # 90-day rolling z-scores (mirrors MacroDataFetcher's 90d history z)
    for name in ["VIX", "DXY", "GOLD", "OIL"]:
        r = df[name].rolling(90, min_periods=5)
        df[f"{name}_Z"] = (df[name] - r.mean()) / (r.std(ddof=0) + 1e-10)
    df["YIELD_CURVE"] = df["YIELD_10Y"] - df["YIELD_5Y"]

    # Fear & Greed full history (free)
    r = requests.get("https://api.alternative.me/fng/?limit=0&format=json", timeout=30)
    fg = pd.DataFrame(r.json()["data"])
    fg["ts"] = pd.to_datetime(fg["timestamp"].astype(int), unit="s")
    fg = fg.set_index("ts").sort_index()
    df["FEAR_GREED"] = fg["value"].astype(float).reindex(df.index, method="ffill")
    df["FEAR_GREED_LABEL"] = fg["value_classification"].reindex(df.index, method="ffill")

    MACRO_CACHE.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(MACRO_CACHE)
    return df


# ----------------------------------------------------------------------------
# Derivatives history (hourly) — reuse the proven backfill pipeline
# ----------------------------------------------------------------------------
def build_derivatives_history(btc_close: pd.Series) -> pd.DataFrame:
    if DERIV_CACHE.exists():
        d = pd.read_parquet(DERIV_CACHE)
        if getattr(d.index, "tz", None) is not None:
            d.index = d.index.tz_localize(None)
        return d

    import backfill_binance_vision_derivatives as bf

    start, end = date(2024, 4, 1), date.today()
    log.info("downloading binance.vision metrics %s → %s (cached files skipped)", start, end)
    mpaths = bf.download_metrics(start, end)
    fpaths = bf.download_funding(date(2024, 1, 1), end)
    metrics = bf.aggregate_metrics(mpaths)
    funding = bf.aggregate_funding(fpaths)
    close = btc_close.copy()
    close.index = close.index.tz_localize(None)
    feats = bf.compute_derived_features(metrics, funding, close)
    DERIV_CACHE.parent.mkdir(parents=True, exist_ok=True)
    feats.to_parquet(DERIV_CACHE)
    return feats


# ----------------------------------------------------------------------------
# Replay
# ----------------------------------------------------------------------------
OVERLAP = 168  # bars re-computed at each chunk start and discarded — absorbs
               # update-accumulated state (funding_Z history, detector smoothing)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None, help="bars to process (smoke test)")
    ap.add_argument("--chunk", type=str, default=None, metavar="I/N",
                    help="process chunk i of n in parallel (e.g. 2/8); writes segment_chunk_i.parquet")
    args = ap.parse_args()

    klines = pd.read_parquet(KLINES)
    klines.index = klines.index.tz_localize(None)
    log.info("klines: %d bars %s → %s", len(klines), klines.index[0], klines.index[-1])

    macro = build_macro_history()
    deriv = build_derivatives_history(klines["close"])
    log.info("macro: %d days; derivatives: %d hours", len(macro), len(deriv))

    from bin.live.live_feature_computer import LiveFeatureComputer

    lfc = LiveFeatureComputer(buffer_size=WARMUP)
    cur = {"ts": None}  # mutable holder read by the stubs

    # ---- Stub 1: historical macro per replay-day --------------------------
    def macro_features():
        ts = cur["ts"]
        try:
            row = macro.loc[:ts.normalize()].iloc[-1]
        except (IndexError, KeyError):
            return {}
        out = {f"{n}_Z": float(row[f"{n}_Z"]) for n in ["VIX", "DXY", "GOLD", "OIL"]
               if pd.notna(row.get(f"{n}_Z"))}
        out["YIELD_CURVE"] = float(row["YIELD_CURVE"]) if pd.notna(row["YIELD_CURVE"]) else 0.0
        out["BTC.D"] = None
        out["USDT.D"] = None
        out["USDC.D"] = None
        if pd.notna(row["FEAR_GREED"]):
            out["FEAR_GREED"] = float(row["FEAR_GREED"])
            out["fear_greed_norm"] = float(row["FEAR_GREED"]) / 100.0
            out["FEAR_GREED_LABEL"] = str(row["FEAR_GREED_LABEL"])
        if pd.notna(row.get("GOLD")):
            out["gold_price"] = float(row["GOLD"])
        if pd.notna(row.get("OIL")):
            out["oil_price"] = float(row["OIL"])
        return out

    lfc._macro_fetcher.get_features = macro_features

    # ---- Stub 2: historical derivatives per replay-hour --------------------
    DERIV_KEYS = ["oi_value", "oi_change_4h", "oi_change_24h", "oi_price_divergence",
                  "binance_funding_rate", "funding_oi_divergence",
                  "ls_ratio_extreme", "taker_imbalance"]
    DEFAULTS = {k: 0.0 for k in DERIV_KEYS}
    DEFAULTS["oi_price_divergence"] = 0
    DEFAULTS["funding_oi_divergence"] = 0

    def deriv_features():
        ts = cur["ts"]
        try:
            row = deriv.loc[ts]
        except KeyError:
            return dict(DEFAULTS)
        out = {}
        for k in DERIV_KEYS:
            v = row.get(k, 0.0)
            out[k] = 0.0 if (v is None or v != v) else (int(v) if "divergence" in k else float(v))
        return out

    lfc._binance_futures_features = deriv_features

    # ---- Warmup + resume ----------------------------------------------------
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    global OUT
    discard_before = None
    if args.chunk:
        i, n = (int(x) for x in args.chunk.split("/"))
        OUT = OUT_DIR / f"segment_chunk_{i}.parquet"
        all_todo = klines.index[WARMUP:]
        size = (len(all_todo) + n - 1) // n
        chunk_idx = all_todo[(i - 1) * size: i * size]
        if len(chunk_idx) == 0:
            log.info("chunk %s empty — nothing to do", args.chunk)
            return
        discard_before = chunk_idx[0]
        # start OVERLAP bars early (discarded later); warm buffer on prior WARMUP bars
        start_pos = max(0, klines.index.get_loc(chunk_idx[0]) - OVERLAP)
        todo = klines.iloc[start_pos: klines.index.get_loc(chunk_idx[-1]) + 1]
        lfc.ingest_candles(klines.iloc[max(0, start_pos - WARMUP): start_pos])
        done = None
        if OUT.exists():
            done = pd.read_parquet(OUT)
            last = done.index[-1]
            lfc.ingest_candles(klines.loc[:last].tail(WARMUP))
            todo = todo.loc[todo.index > last]
        log.info("chunk %s: %d bars (%s → %s), discard before %s",
                 args.chunk, len(todo), todo.index[0], todo.index[-1], discard_before)
    else:
        done = None
        if OUT.exists():
            done = pd.read_parquet(OUT)
            log.info("resuming: %d rows already computed (last %s)", len(done), done.index[-1])
        warmup_df = klines.iloc[:WARMUP]
        lfc.ingest_candles(warmup_df)
        todo = klines.iloc[WARMUP:]
        if done is not None and len(done):
            last = done.index[-1]
            upto = klines.loc[:last]
            lfc.ingest_candles(upto.tail(WARMUP))
            todo = klines.loc[klines.index > last]
    if args.limit:
        todo = todo.iloc[:args.limit]
    log.info("replaying %d bars (%s → %s)", len(todo), todo.index[0], todo.index[-1])

    rows, t0 = [], time.time()
    n_done = 0
    try:
        for ts, bar in todo.iterrows():
            cur["ts"] = ts
            series = lfc.update({
                "timestamp": ts, "open": bar["open"], "high": bar["high"],
                "low": bar["low"], "close": bar["close"], "volume": bar["volume"],
            })
            rows.append(series)
            n_done += 1
            if n_done % 200 == 0:
                rate = n_done / (time.time() - t0)
                eta = (len(todo) - n_done) / rate / 60
                log.info("  %d/%d bars (%.1f bars/s, ETA %.0f min)", n_done, len(todo), rate, eta)
            if n_done % 2000 == 0:
                _checkpoint(done, rows)
                done = pd.read_parquet(OUT)
                rows = []
    finally:
        if rows:
            _checkpoint(done, rows)

    final = pd.read_parquet(OUT)
    if discard_before is not None:
        OUT.with_suffix(".discard_before.txt").write_text(str(discard_before))
    log.info("DONE: %d rows, %d cols, %s → %s", len(final), final.shape[1],
             final.index[0], final.index[-1])


def _checkpoint(done, rows):
    new = pd.DataFrame(rows)
    # object columns (labels) → keep as str for parquet
    for c in new.columns:
        if new[c].dtype == object:
            new[c] = new[c].astype(str)
    merged = pd.concat([done, new]) if done is not None and len(done) else new
    merged = merged[~merged.index.duplicated(keep="last")].sort_index()
    tmp = OUT.with_suffix(".tmp.parquet")
    merged.to_parquet(tmp)
    tmp.rename(OUT)
    log.info("  checkpoint: %d rows total", len(merged))


if __name__ == "__main__":
    main()
