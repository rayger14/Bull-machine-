#!/usr/bin/env python3
"""Risk/exit overlay study on the honest core (V14, WFO).

The live engine: 67% WR but PF 0.71 — direction is fine, losers > winners.
Sizing is RISK-BASED (notional = risk$ / stop_distance), so every stop-out costs
~2% regardless of stop width; the asymmetry comes from book composition (trading
proven losers) and regime (long-only into bears). This study tests the cleanest,
Rule-8-compliant levers — composition + regime sizing + stop width — NOT the
exit-logic R-levels (those were tried on V12 and reverted; separate work).

Phase 1 — COMPOSITION (the "cut the book" thesis):
  full16     all archetypes, thresholds enforced (best case for the full book)
  core2      wick_trap_v14rq + liquidity_sweep   (the 2 holdout-positive assets)
  core4      core2 + exhaustion_reversal + funding_divergence

Phase 2 — RISK OVERLAYS on the best composition:
  regime sizing: long size *= k in macro-bear (k in {1.0, 0.5, 0.25}, 200d)
  stop width:    stop distance *= f  (f in {0.7, 1.0, 1.3}); risk-neutral, so this
                 changes stop-out FREQUENCY, not per-stop $ — the lever for "too
                 many full-risk stop-outs".

Windows: per-year 2018-2024, wfo_train 2018-2022, holdout 2025-26.
Reports per-window PF/PnL/MaxDD/WR + the composition comparison. Honest n flags.

Usage: python3 scripts/champion/risk_overlay_study.py [--phase 1|2|all]
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts/champion"))

from run_battery import ALL_ARCHETYPES, BASE_CONFIG, CONFIG_DIR, score  # noqa: E402

STORE = REPO / "data/features_mtf/BTC_1H_FEATURES_V14_FULL_LIVE_PATH.parquet"
V14RQ_DIR = "configs/champion/archetypes_v14rq/"   # has wick_trap @ 0.432
OUT_ROOT = REPO / "results/champion_v14_risk"

WINDOWS = {
    "y2018": ("2018-01-01", "2018-12-31"), "y2019": ("2019-01-01", "2019-12-31"),
    "y2020": ("2020-01-01", "2020-12-31"), "y2021": ("2021-01-01", "2021-12-31"),
    "y2022": ("2022-01-01", "2022-12-31"), "y2023": ("2023-01-01", "2023-12-31"),
    "y2024": ("2024-01-01", "2024-12-31"),
    "wfo_train": ("2018-01-01", "2022-12-31"),
    "holdout_2025_26": ("2025-01-01", "2026-06-10"),
    "full": ("2018-01-01", "2024-12-31"),
}

PORTFOLIOS = {
    "full16": ALL_ARCHETYPES,
    "core2":  ["wick_trap", "liquidity_sweep"],
    "core4":  ["wick_trap", "liquidity_sweep", "exhaustion_reversal", "funding_divergence"],
}

_BEAR = {}


def macro_bear(days=200):
    if days not in _BEAR:
        c = pd.read_parquet(STORE, columns=["close"])["close"]
        if getattr(c.index, "tz", None) is not None:
            c.index = c.index.tz_localize(None)
        _BEAR[days] = c < c.rolling(days * 24, min_periods=days * 12).mean()
    return _BEAR[days]


def make_portfolio_config(name, archetypes) -> Path:
    cfg = json.loads(BASE_CONFIG.read_text())
    cfg["disabled_archetypes"] = [a for a in ALL_ARCHETYPES if a not in archetypes]
    cfg.setdefault("adaptive_fusion", {})["bypass_threshold"] = False
    cfg["archetype_config_dir"] = V14RQ_DIR  # wick_trap re-quantiled to 0.432
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    p = CONFIG_DIR / f"risk_{name}.json"
    p.write_text(json.dumps(cfg, indent=2))
    return p


def install_overlay(bear_k=1.0, stop_f=1.0):
    """Patch _open_position: long size *= bear_k in macro-bear; stop distance *= stop_f."""
    from bin.backtest_v11_standalone import StandaloneBacktestEngine
    bear = macro_bear() if bear_k != 1.0 else None
    orig = StandaloneBacktestEngine._open_position
    st = {"n": 0, "bear_sized": 0}

    def patched(self, *a, **k):
        ts = k.get("timestamp") or (a[0] if a else None)
        direction = k.get("direction") or (a[2] if len(a) > 2 else None)
        entry = k.get("entry_price") or (a[3] if len(a) > 3 else None)
        st["n"] += 1
        # stop width (risk-neutral: sizing recomputes notional from stop distance)
        if stop_f != 1.0 and entry and "stop_loss" in k and k["stop_loss"]:
            k["stop_loss"] = entry - (entry - k["stop_loss"]) * stop_f
        # regime sizing
        if bear is not None and direction == "long" and ts is not None and "allocated_size_pct" in k:
            try:
                if bool(bear.at[ts]):
                    k["allocated_size_pct"] *= bear_k
                    st["bear_sized"] += 1
            except KeyError:
                pass
        return orig(self, *a, **k)

    StandaloneBacktestEngine._open_position = patched
    return st, lambda: setattr(StandaloneBacktestEngine, "_open_position", orig)


def run_window(cfg_path, start, end, out_dir, bear_k=1.0, stop_f=1.0) -> dict:
    from bin.backtest_v11_standalone import StandaloneBacktestEngine
    out_dir.mkdir(parents=True, exist_ok=True)
    sf = out_dir / "performance_stats.json"
    if sf.exists():
        return json.loads(sf.read_text())
    st, restore = install_overlay(bear_k, stop_f)
    try:
        eng = StandaloneBacktestEngine(config=json.loads(cfg_path.read_text()),
            initial_cash=100_000.0, commission_rate=0.0002, slippage_bps=3.0,
            feature_store_path=str(STORE))
        eng.run(start_date=start, end_date=end)
        stats = eng.get_performance_stats()
        eng.save_trade_log(str(out_dir / "trade_log.csv"))
    finally:
        restore()
    clean = {}
    for kk, v in stats.items():
        try:
            clean[kk] = str(v) if (isinstance(v, float) and (math.isnan(v) or math.isinf(v))) else (json.loads(json.dumps(v)) or v)
        except (TypeError, ValueError):
            clean[kk] = str(v)
    sf.write_text(json.dumps(clean, indent=2, default=str))
    return clean


def run_variant(label, cfg_path, bear_k=1.0, stop_f=1.0):
    print(f"\n=== {label} ===", flush=True)
    results = {w: run_window(cfg_path, s, e, OUT_ROOT / label / w, bear_k, stop_f)
               for w, (s, e) in WINDOWS.items()}
    card = score(label, results)
    card["variant"] = label
    (OUT_ROOT / label / "scorecard.json").write_text(json.dumps(card, indent=2, default=str))
    d = card["detail"]
    for w in ["y2018", "y2021", "y2022", "holdout_2025_26", "full"]:
        r = d.get(w, {})
        print(f"  {w:<16} pnl={r.get('pnl'):>8} pf={r.get('pf'):>6} n={r.get('trades'):>5} mdd={r.get('maxdd_pct')}%")
    return card


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", default="all", choices=["1", "2", "all"])
    args = ap.parse_args()
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    cards = []

    if args.phase in ("1", "all"):
        print("\n##### PHASE 1: COMPOSITION #####")
        for name, arch in PORTFOLIOS.items():
            cards.append(run_variant(f"comp_{name}", make_portfolio_config(name, arch)))

    if args.phase in ("2", "all"):
        print("\n##### PHASE 2: RISK OVERLAYS on core4 #####")
        core = make_portfolio_config("core4", PORTFOLIOS["core4"])
        for bk in (0.5, 0.25):
            cards.append(run_variant(f"risk_core4_bear{int(bk*100)}", core, bear_k=bk))
        for sf in (0.7, 1.3):
            cards.append(run_variant(f"risk_core4_stop{int(sf*100)}", core, stop_f=sf))
        # best-guess combo: bear de-size + wider stops
        cards.append(run_variant("risk_core4_bear50_stop130", core, bear_k=0.5, stop_f=1.3))

    summary = OUT_ROOT / f"summary_phase{args.phase}.json"
    summary.write_text(json.dumps(cards, indent=2, default=str))
    print(f"\n{'='*72}\nSUMMARY")
    print(f"{'variant':<26}{'2018':>8}{'2021':>8}{'2022':>8}{'holdPF':>8}{'fullPF':>8}{'fullPnL':>9}{'mdd':>6}")
    for c in cards:
        d = c["detail"]
        print(f"{c['variant']:<26}{d['y2018']['pnl']:>8}{d['y2021']['pnl']:>8}{d['y2022']['pnl']:>8}"
              f"{d['holdout_2025_26']['pf']:>8}{d['full']['pf']:>8}{d['full']['pnl']:>9}{d['full']['maxdd_pct']:>6}")


if __name__ == "__main__":
    main()
