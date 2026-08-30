#!/usr/bin/env python3
"""V23 PARITY STORE builder — one code path, one reality.

Metrology audit 2026-08-28 found the production store (V12) and the study
lineage (V14-V22) broken in opposite ways, neither matching live. This
builder produces the replacement by running THE ACTUAL LiveFeatureComputer
(with TA-Lib, verified numerically identical to the server) bar-by-bar over
historical OHLCV — parity with live BY CONSTRUCTION for every price-derived
feature.

Deliberately OFFLINE: all network-dependent feature groups (binance/okx
derivatives snapshots, macro, news, alt-basket) are no-op'd; the validated
historical derivatives witnesses (Binance Vision aggregates 2020-09..2024-12,
CME OI) are spliced in a separate assembly step, keeping witness provenance
explicit instead of pretending a live fetch happened in 2021.

Usage:
  python3 bin/build_v23_parity_store.py --start 2023-06-01 --end 2024-06-01 \
      --out data/features_mtf/V23_pilot.parquet [--warmup 1000] [--checkpoint 1000]
Resumes from checkpoint automatically if the output's .ckpt.parquet exists.
"""
import argparse, logging, sys, time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
logging.disable(logging.CRITICAL)

from bin.live.live_feature_computer import LiveFeatureComputer  # noqa: E402


def _neuter_network(fc: LiveFeatureComputer) -> None:
    """No-op every network-touching method so offline builds are fast and
    deterministic. Missing-feature semantics = NaN/absent (nan_policy paths),
    matching a live bar where the fetch failed."""
    noops = {}
    for name in dir(fc):
        if any(k in name for k in ('binance', 'okx', 'coinglass', 'macro',
                                   'news', 'alt_basket', 'fear_greed',
                                   'coinbase_funding', 'eth_btc')):
            attr = getattr(fc, name, None)
            if callable(attr) and name.startswith('_'):
                noops[name] = attr
    for name in noops:
        try:
            setattr(fc, name, lambda *a, **k: {})
        except Exception:
            pass
    # derivative snapshot cache: permanently empty
    if hasattr(fc, '_binance_cache'):
        fc._binance_cache = {}
        fc._binance_fetch_interval = 10**12
        fc._binance_last_fetch = time.time()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--start', required=True)
    ap.add_argument('--end', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--source', default='data/features_mtf/BTC_1H_FEATURES_V12_ENHANCED.parquet',
                    help='OHLCV source (uses open/high/low/close/volume only)')
    ap.add_argument('--warmup', type=int, default=1000)
    ap.add_argument('--checkpoint', type=int, default=1000)
    args = ap.parse_args()

    src = pd.read_parquet(args.source)[['open', 'high', 'low', 'close', 'volume']]
    if src.index.tz is None:
        src.index = src.index.tz_localize('UTC')
    start = pd.Timestamp(args.start, tz='UTC'); end = pd.Timestamp(args.end, tz='UTC')
    body = src.loc[start:end]
    warm = src.loc[:start].tail(args.warmup)
    if len(warm) < args.warmup:
        print(f"WARN: only {len(warm)} warmup bars available")
    print(f"build: {len(body)} bars ({body.index[0]} -> {body.index[-1]}), warmup {len(warm)}")

    ckpt = Path(args.out + '.ckpt.parquet')
    done: list = []
    resume_from = None
    if ckpt.exists():
        prev = pd.read_parquet(ckpt)
        done = [prev]
        resume_from = prev.index[-1]
        print(f"resuming after {resume_from} ({len(prev)} rows)")

    fc = LiveFeatureComputer()
    _neuter_network(fc)
    fc.ingest_candles(warm if resume_from is None else src.loc[:resume_from].tail(args.warmup + 1).iloc[:-1])

    rows = {}
    t0 = time.time(); n = 0
    it = body.itertuples()
    for bar in it:
        ts = bar.Index
        if resume_from is not None and ts <= resume_from:
            # replay through computer to maintain buffers, discard output
            fc.update({'timestamp': ts, 'open': bar.open, 'high': bar.high,
                       'low': bar.low, 'close': bar.close, 'volume': bar.volume})
            continue
        f = fc.update({'timestamp': ts, 'open': bar.open, 'high': bar.high,
                       'low': bar.low, 'close': bar.close, 'volume': bar.volume})
        rows[ts] = f
        n += 1
        if n % args.checkpoint == 0:
            chunk = pd.DataFrame.from_dict(rows, orient='index')
            allp = pd.concat(done + [chunk]) if done else chunk
            allp.to_parquet(ckpt)
            done = [allp]; rows = {}
            rate = n / (time.time() - t0)
            print(f"  {n} bars  ({rate:.1f} bars/s, eta {((len(body)-n)/rate)/60:.0f} min)", flush=True)
    chunk = pd.DataFrame.from_dict(rows, orient='index') if rows else None
    allp = pd.concat(done + ([chunk] if chunk is not None else [])) if done or chunk is not None else pd.DataFrame()
    allp.to_parquet(args.out)
    if ckpt.exists():
        ckpt.unlink()
    print(f"DONE: {len(allp)} rows x {len(allp.columns)} cols -> {args.out}  ({(time.time()-t0)/60:.1f} min)")


if __name__ == '__main__':
    main()
