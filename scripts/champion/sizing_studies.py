#!/usr/bin/env python3
"""Pre-registered sizing studies on wick_trap_v14rq (V14 live-path store).

wick_trap_v14rq is the best honest asset (holdout +$6.8K PF 1.43) but bleeds in
macro-bear years. These studies test whether REGIME/LEVEL-AWARE SIZING (never
filtering — Rule 8) compresses the bear bleed without damaging the holdout.

  S1  macro-bear sizing: long size *= k when close < N-day mean
        grid: k in {0.25, 0.50} x N in {150, 200, 250}
  S2  day-type sizing: long size *= k when day_type == -1 (trend-down day)
        grid: k in {0.25, 0.50}    (counter-trend longs on confirmed downtrend days)
  S3  level-quality sizing: long size *= k when level_quality_low < train-tercile cutoff
        grid: k in {0.25, 0.50} x cutoff in {q33, q50}

All sizing keys are CONCURRENT-STATE (Lesson 41) and point-in-time. Tercile
cutoffs computed on TRAIN (2018-2022) only — no lookahead. Baseline k=1.0 included.

Windows: per-year 2018-2024, wfo_train 2018-2022, holdout_2025_26.
Acceptance (a champion): C1 positive every year 2018-2024, C2 holdout PF>=1.3,
C3 train PF>=1.3, C4 holdout n>=30, C5 full MaxDD<=10%, AND train+holdout co-move.

Usage: python3 scripts/champion/sizing_studies.py [--study S1|S2|S3|all]
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts/champion"))

from run_battery import score  # noqa: E402

STORE = REPO / "data/features_mtf/BTC_1H_FEATURES_V14_FULL_LIVE_PATH.parquet"
LEVELF = REPO / "results/rebuild/level_features_full.parquet"
CONFIG = REPO / "configs/champion/champion_wick_trap_v14rq.json"
OUT_ROOT = REPO / "results/champion_v14_sizing"

WINDOWS = {
    "y2018": ("2018-01-01", "2018-12-31"), "y2019": ("2019-01-01", "2019-12-31"),
    "y2020": ("2020-01-01", "2020-12-31"), "y2021": ("2021-01-01", "2021-12-31"),
    "y2022": ("2022-01-01", "2022-12-31"), "y2023": ("2023-01-01", "2023-12-31"),
    "y2024": ("2024-01-01", "2024-12-31"),
    "wfo_train": ("2018-01-01", "2022-12-31"),
    "holdout_2025_26": ("2025-01-01", "2026-06-10"),
    "full": ("2018-01-01", "2024-12-31"),
}

_CACHE = {}


def macro_bear(days: int) -> pd.Series:
    key = f"macro{days}"
    if key not in _CACHE:
        close = pd.read_parquet(STORE, columns=["close"])["close"]
        if getattr(close.index, "tz", None) is not None:
            close.index = close.index.tz_localize(None)
        bars = days * 24
        _CACHE[key] = close < close.rolling(bars, min_periods=bars // 2).mean()
    return _CACHE[key]


def level_col(col: str) -> pd.Series:
    if col not in _CACHE:
        _CACHE[col] = pd.read_parquet(LEVELF, columns=[col])[col]
    return _CACHE[col]


def install(size_fn):
    """Patch _open_position: long size *= size_fn(timestamp)."""
    from bin.backtest_v11_standalone import StandaloneBacktestEngine
    state = {"entries": 0, "scaled": 0}
    original = StandaloneBacktestEngine._open_position

    def patched(self, *args, **kwargs):
        direction = kwargs.get("direction") or (args[2] if len(args) > 2 else None)
        ts = kwargs.get("timestamp") or (args[0] if len(args) > 0 else None)
        state["entries"] += 1
        if direction == "long" and ts is not None and "allocated_size_pct" in kwargs:
            mult = size_fn(ts)
            if mult != 1.0:
                kwargs["allocated_size_pct"] *= mult
                state["scaled"] += 1
        return original(self, *args, **kwargs)

    StandaloneBacktestEngine._open_position = patched
    return state, lambda: setattr(StandaloneBacktestEngine, "_open_position", original)


def run_window(size_fn, start, end, out_dir) -> dict:
    from bin.backtest_v11_standalone import StandaloneBacktestEngine
    out_dir.mkdir(parents=True, exist_ok=True)
    sf = out_dir / "performance_stats.json"
    if sf.exists():
        return json.loads(sf.read_text())
    state, restore = install(size_fn)
    try:
        cfg = json.loads(CONFIG.read_text())
        eng = StandaloneBacktestEngine(config=cfg, initial_cash=100_000.0,
            commission_rate=0.0002, slippage_bps=3.0, feature_store_path=str(STORE))
        eng.run(start_date=start, end_date=end)
        stats = eng.get_performance_stats()
    finally:
        restore()
    clean = {}
    for k, v in stats.items():
        try:
            if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
                clean[k] = str(v)
            else:
                json.dumps(v); clean[k] = v
        except (TypeError, ValueError):
            clean[k] = str(v)
    sf.write_text(json.dumps(clean, indent=2, default=str))
    return clean


def run_variant(name, size_fn) -> dict:
    print(f"\n=== {name} ===", flush=True)
    results = {}
    total_scaled = 0
    for w, (s, e) in WINDOWS.items():
        results[w] = run_window(size_fn, s, e, OUT_ROOT / name / w)
    card = score(name, results)
    card["variant"] = name
    (OUT_ROOT / name / "scorecard.json").write_text(json.dumps(card, indent=2, default=str))
    d = card["detail"]
    print(f"  {'window':<16}{'pnl':>9}{'pf':>7}{'n':>6}{'mdd%':>6}")
    for w in ["y2018","y2019","y2020","y2021","y2022","y2023","y2024","holdout_2025_26","full"]:
        if w in d:
            r = d[w]; print(f"  {w:<16}{r['pnl']:>9}{r['pf']:>7}{r['trades']:>6}{r['maxdd_pct']:>6}")
    print(f"  -> {'PASS' if card['pass'] else 'FAIL'} ({sum(1 for v in card['criteria'].values() if v)}/5)")
    return card


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--study", default="all", choices=["S1", "S2", "S3", "all"])
    args = ap.parse_args()
    OUT_ROOT.mkdir(parents=True, exist_ok=True)

    cards = []

    # Baseline (k=1.0, no sizing) for reference
    cards.append(run_variant("baseline", lambda ts: 1.0))

    if args.study in ("S1", "all"):
        for k in (0.25, 0.50):
            for N in (150, 200, 250):
                bear = macro_bear(N)
                def fn(ts, bear=bear, k=k):
                    try: return k if bool(bear.at[ts]) else 1.0
                    except KeyError: return 1.0
                cards.append(run_variant(f"S1_macro{N}_k{int(k*100)}", fn))

    if args.study in ("S2", "all"):
        dt = level_col("day_type")
        for k in (0.25, 0.50):
            def fn(ts, dt=dt, k=k):
                try: return k if dt.at[ts] == -1 else 1.0
                except KeyError: return 1.0
            cards.append(run_variant(f"S2_daytype_k{int(k*100)}", fn))

    if args.study in ("S3", "all"):
        lq = level_col("level_quality_low")
        train = lq.loc["2018-01-01":"2022-12-31"]
        for cut_name, cut in (("q33", train.quantile(0.33)), ("q50", train.quantile(0.50))):
            for k in (0.25, 0.50):
                def fn(ts, lq=lq, cut=cut, k=k):
                    try:
                        v = lq.at[ts]
                        return k if (v == v and v < cut) else 1.0
                    except KeyError: return 1.0
                cards.append(run_variant(f"S3_lq{cut_name}_k{int(k*100)}", fn))

    summary = OUT_ROOT / f"summary_{args.study}.json"
    summary.write_text(json.dumps(cards, indent=2, default=str))
    print(f"\n{'='*70}\nSUMMARY ({summary.name})")
    print(f"{'variant':<22}{'pass':<6}{'2018':>8}{'2021':>8}{'2022':>8}{'hold_pf':>8}{'full_pf':>8}{'mdd':>6}")
    for c in cards:
        d = c["detail"]
        print(f"{c['variant']:<22}{str(c['pass']):<6}"
              f"{d['y2018']['pnl']:>8}{d['y2021']['pnl']:>8}{d['y2022']['pnl']:>8}"
              f"{d['holdout_2025_26']['pf']:>8}{d['full']['pf']:>8}{d['full']['maxdd_pct']:>6}")


if __name__ == "__main__":
    main()
