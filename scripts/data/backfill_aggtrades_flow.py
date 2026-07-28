#!/usr/bin/env python3
"""Phase-1 tick-flow backfill — true aggressor features from Binance aggTrades.

The hourly taker_imbalance proxy washed out on CVD divergence (sign-flip,
2026-07-29). This backfill computes the TRUE tick-derived versions from
data.binance.vision monthly aggTrades archives (BTCUSDT spot), aggregated
to 1H, so the pre-registered splits can be rerun on real flow data BEFORE
any live infrastructure is built.

PRE-REGISTERED Phase-1 test features (defined here, before any results):
  true_delta        taker buy vol - taker sell vol (base asset), per hour
  cvd_true          cumulative sum of true_delta (built at load time)
  delta_at_low      signed volume of trades within 0.1% of the hour's low
                    (who was aggressing AT the extreme)
  delta_after_low   signed volume after the low's timestamp (did buyers
                    take over after the flush point)
  t_low_frac        when the low occurred within the hour (0..1)
Splits (same clean wick_trap trade logs, train+holdout, n>=30/cell):
  S1 cvd_true 24h divergence (same definition as the failed proxy test)
  S2 delta_at_low <= 0 (sellers aggressed the extreme)
  S3 delta_after_low > 0 (recovery aggression flipped to buyers)

Output: results/rebuild/aggtrades_flow/BTCUSDT-YYYY-MM.parquet (hourly rows).
Resumable: existing month files are skipped. ~100 months, network-bound.

Usage:
  python3 scripts/data/backfill_aggtrades_flow.py                 # 2018-01..2026-06
  python3 scripts/data/backfill_aggtrades_flow.py --start 2025-01 --end 2026-06
"""
from __future__ import annotations

import argparse
import csv
import io
import sys
import tempfile
import time
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
OUT_DIR = REPO / "results/rebuild/aggtrades_flow"
URL = ("https://data.binance.vision/data/spot/monthly/aggTrades/BTCUSDT/"
       "BTCUSDT-aggTrades-{ym}.zip")
LOW_BAND = 1.001  # trades with price <= hour_low * 1.001 count as "at the low"


def month_range(start: str, end: str):
    cur = pd.Period(start, "M")
    last = pd.Period(end, "M")
    while cur <= last:
        yield str(cur)
        cur += 1


def process_month(ym: str, out_path: Path) -> None:
    url = URL.format(ym=ym)
    t0 = time.time()
    with tempfile.NamedTemporaryFile(suffix=".zip") as tmp:
        with urllib.request.urlopen(url, timeout=120) as r:
            while True:
                chunk = r.read(1 << 22)
                if not chunk:
                    break
                tmp.write(chunk)
        tmp.flush()
        dl_s = time.time() - t0

        rows = []
        with zipfile.ZipFile(tmp.name) as zf:
            name = zf.namelist()[0]
            with zf.open(name) as f:
                text = io.TextIOWrapper(f, encoding="utf-8", newline="")
                reader = csv.reader(text)
                cur_hour = None
                # per-hour accumulators
                prices, qtys, signs, times = [], [], [], []

                def flush_hour(hour):
                    if not prices:
                        return
                    p = np.asarray(prices)
                    q = np.asarray(qtys)
                    s = np.asarray(signs)          # +1 taker buy, -1 taker sell
                    t = np.asarray(times)
                    low = p.min()
                    i_low = int(p.argmin())
                    t_low = t[i_low]
                    at_low = p <= low * LOW_BAND
                    after = t > t_low
                    rows.append({
                        "ts": pd.Timestamp(hour, unit="h"),
                        "buy_vol": float(q[s > 0].sum()),
                        "sell_vol": float(q[s < 0].sum()),
                        "true_delta": float((q * s).sum()),
                        "delta_at_low": float((q[at_low] * s[at_low]).sum()),
                        "delta_after_low": float((q[after] * s[after]).sum()),
                        "t_low_frac": float((t_low - hour * 3600_000) / 3600_000),
                        "n_trades": int(len(p)),
                    })

                for rec in reader:
                    if rec[0] == "agg_trade_id" or rec[0].startswith("a"):
                        continue  # header line variants
                    ts_ms = int(rec[5])
                    if ts_ms > 10**14:      # some archives use microseconds
                        ts_ms //= 1000
                    hour = ts_ms // 3600_000
                    if cur_hour is None:
                        cur_hour = hour
                    if hour != cur_hour:
                        flush_hour(cur_hour)
                        prices, qtys, signs, times = [], [], [], []
                        cur_hour = hour
                    prices.append(float(rec[1]))
                    qtys.append(float(rec[2]))
                    # is_buyer_maker == true -> taker SOLD
                    signs.append(-1.0 if rec[6] in ("true", "True", "1") else 1.0)
                    times.append(ts_ms)
                flush_hour(cur_hour)

    df = pd.DataFrame(rows).set_index("ts").sort_index()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path)
    print(f"[{ym}] {len(df)} hours, {df['n_trades'].sum():,} trades "
          f"(dl {dl_s:.0f}s, total {time.time()-t0:.0f}s)", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2018-01")
    ap.add_argument("--end", default="2026-06")
    args = ap.parse_args()
    months = list(month_range(args.start, args.end))
    print(f"backfilling {len(months)} months -> {OUT_DIR}", flush=True)
    for ym in months:
        out = OUT_DIR / f"BTCUSDT-{ym}.parquet"
        if out.exists():
            continue
        for attempt in range(3):
            try:
                process_month(ym, out)
                break
            except Exception as e:
                print(f"[{ym}] attempt {attempt+1} failed: {e}", flush=True)
                time.sleep(20)
        else:
            print(f"[{ym}] GIVING UP after 3 attempts", flush=True)
    done = len(list(OUT_DIR.glob("*.parquet")))
    print(f"DONE: {done}/{len(months)} months on disk", flush=True)


if __name__ == "__main__":
    main()
