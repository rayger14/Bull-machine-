#!/usr/bin/env python3
"""Offline validation of wick_trap_v14rq against the REAL live feature stream.

The disciplined pre-deploy test (user decision 2026-06-24): before changing the
live engine, prove the validated champion actually fires + performs on the
features the live engine has computed since the feature logger went live, using
V14's tail for warmup so the short live window doesn't cold-start.

What it does:
  1. live_features/*.jsonl  ->  align to V14 schema
  2. concat [V14 tail for warmup] + [live rows]  -> temp parquet
  3. run the backtester with configs/champion/champion_wick_trap_v14rq.json over
     the live date range
  4. report wick_trap signals/trades on live data vs the backtested expectation
     (holdout 2025-26 was PF 1.43); flag insufficient-n honestly.

Mirrors the validated battery conditions exactly (e.g. the `instability` gate is
absent in V14 and is left absent here — it has nan_policy: skip — so behavior
matches the holdout result we are validating). Run on demand or nightly; the
verdict firms up as live logs accumulate (need ~30+ wick_trap setups, weeks).

Usage: python3 scripts/champion/validate_live_offline.py [--pull] [--warmup 600]
"""
from __future__ import annotations

import argparse
import glob
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
V14 = REPO / "data/features_mtf/BTC_1H_FEATURES_V14_FULL_LIVE_PATH.parquet"
LIVE_DIR = REPO / "results/coinbase_paper/live_features"
CONFIG = REPO / "configs/champion/champion_wick_trap_v14rq.json"
OUT_DIR = REPO / "results/champion_v14/live_offline"
SSH_KEY = str(Path.home() / ".ssh/oracle_bullmachine")
SERVER = "ubuntu@165.1.79.19"


def load_live() -> pd.DataFrame:
    files = sorted(glob.glob(str(LIVE_DIR / "*.jsonl")))
    frames = [pd.read_json(f, lines=True) for f in files if Path(f).stat().st_size > 0]
    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames, ignore_index=True)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    return df.set_index("timestamp").sort_index()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pull", action="store_true")
    ap.add_argument("--warmup", type=int, default=600, help="V14 tail bars for warmup")
    args = ap.parse_args()

    if args.pull:
        LIVE_DIR.mkdir(parents=True, exist_ok=True)
        subprocess.run(["scp", "-i", SSH_KEY, "-q",
                        f"{SERVER}:Bull-machine-/results/coinbase_paper/live_features/*.jsonl",
                        str(LIVE_DIR) + "/"], check=False)

    live = load_live()
    if live.empty:
        print("No live logs yet."); return 0
    v14 = pd.read_parquet(V14)
    if getattr(v14.index, "tz", None) is None:
        v14.index = v14.index.tz_localize("UTC")

    # Align live to V14 schema; numeric coercion (drop the regime_label so the
    # backtester re-derives it consistently, as the battery did).
    live_aligned = live.reindex(columns=v14.columns)
    for c in live_aligned.columns:
        if c != "regime_label":
            live_aligned[c] = pd.to_numeric(live_aligned[c], errors="coerce")
    live_aligned["regime_label"] = float("nan")

    warmup = v14.tail(args.warmup)
    combined = pd.concat([warmup, live_aligned])
    combined = combined[~combined.index.duplicated(keep="last")].sort_index()
    for c in combined.columns:
        if combined[c].dtype == object:
            m = combined[c].notna()
            combined.loc[m, c] = combined.loc[m, c].astype(str)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    tmp = Path(tempfile.mkdtemp()) / "live_plus_warmup.parquet"
    combined.to_parquet(tmp)

    start = live.index[0].strftime("%Y-%m-%d")
    end = live.index[-1].strftime("%Y-%m-%d")
    print(f"live window: {start} -> {end} ({len(live)} bars, {(live.index[-1]-live.index[0]).days}d); "
          f"warmup {len(warmup)} V14 bars")

    cmd = [sys.executable, str(REPO / "bin/backtest_v11_standalone.py"),
           "--config", str(CONFIG), "--feature-store", str(tmp),
           "--start-date", start, "--end-date", end,
           "--output-dir", str(OUT_DIR),
           "--commission-rate", "0.0002", "--slippage-bps", "3", "--initial-cash", "100000"]
    res = subprocess.run(cmd, capture_output=True, text=True, cwd=REPO)
    stats_file = OUT_DIR / "performance_stats.json"
    if res.returncode != 0 or not stats_file.exists():
        print("backtest failed:\n", res.stderr[-1500:]); return 1

    stats = json.loads(stats_file.read_text())
    n = stats.get("total_trades", 0)
    pf = stats.get("profit_factor", "n/a")
    pnl = stats.get("total_pnl", 0)
    wr = stats.get("win_rate", "n/a")
    print(f"\n=== wick_trap_v14rq on LIVE features ({start}..{end}) ===")
    print(f"  trades={n}  PF={pf}  PnL=${pnl:,.0f}  WR={wr}")
    print(f"  backtest holdout reference: PF 1.43 (2025-26)")
    if n < 30:
        print(f"  ** INSUFFICIENT: {n} trades (<30). Accumulating — a real verdict "
              f"needs weeks of live logs. Harness is ready; rerun as data grows. **")
    summary = {"window": [start, end], "live_bars": int(len(live)),
               "trades": n, "pf": pf, "pnl": pnl, "win_rate": wr,
               "verdict": "INSUFFICIENT_N" if n < 30 else "READ_AVAILABLE"}
    (OUT_DIR / "live_validation_summary.json").write_text(json.dumps(summary, indent=2, default=str))
    print(f"  summary -> {OUT_DIR / 'live_validation_summary.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
