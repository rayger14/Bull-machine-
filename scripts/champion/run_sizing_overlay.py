#!/usr/bin/env python3
"""
Champion regime-sizing overlay — crisis-as-sizing study (Rule 8: sizing, not filters).

Monkey-patches StandaloneBacktestEngine._open_position (same pattern as
scripts/composite_boost/run_variant.py) to scale LONG position size by `k`
when the entry bar's ENGINE-DERIVED regime_label (SMA50/200 + vol, computed
at engine init) is in the configured set. No entries are blocked; no
production code or YAML changes.

Variants swept (selected on train 2020-2022, which contains the bear failure
year; OOS 2023/2024 must remain undamaged — Rule 9 co-movement):

  k025_bear2   k=0.25, regimes={risk_off, crisis}
  k050_bear2   k=0.50, regimes={risk_off, crisis}
  k025_crisis  k=0.25, regimes={crisis}

Usage:
  python3 scripts/champion/run_sizing_overlay.py --candidate wick_trap
  python3 scripts/champion/run_sizing_overlay.py --candidate liquidity_sweep --variants k025_bear2
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts/champion"))

from run_battery import WINDOWS, make_variant_config, score  # noqa: E402

VARIANTS = {
    "k025_bear2":  {"k": 0.25, "regimes": {"risk_off", "crisis"}},
    "k050_bear2":  {"k": 0.50, "regimes": {"risk_off", "crisis"}},
    "k025_crisis": {"k": 0.25, "regimes": {"crisis"}},
    # Macro-horizon variants: engine regime labels are 200-HOUR SMAs (~8 days)
    # and call 2022 bear rallies "risk_on". These use price vs 200-DAY
    # (4800-bar) mean from the full feature store instead.
    "k025_macro200d": {"k": 0.25, "macro_sma_days": 200},
    "k050_macro200d": {"k": 0.50, "macro_sma_days": 200},
    # Regime-complementary pair: each archetype sized down OUTSIDE its element.
    # wick_trap bleeds in macro-bear (buys dips); exhaustion_reversal bleeds in
    # macro-bull (buys capitulation that never comes). Sizing only, no filters.
    "pair_k025": {"macro_sma_days": 200, "rules": {
        "wick_trap":           {"k": 0.25, "when": "macro_bear"},
        "exhaustion_reversal": {"k": 0.25, "when": "macro_bull"},
    }},
    "pair_k050": {"macro_sma_days": 200, "rules": {
        "wick_trap":           {"k": 0.50, "when": "macro_bear"},
        "exhaustion_reversal": {"k": 0.50, "when": "macro_bull"},
    }},
    # --- Adjudication sensitivity battery (2026-06-10) ---
    # pair_k100 = no-overlay PAIR baseline (k=1.0 is identity) for Rule 9 deltas.
    "pair_k100": {"macro_sma_days": 200, "rules": {
        "wick_trap":           {"k": 1.00, "when": "macro_bear"},
        "exhaustion_reversal": {"k": 1.00, "when": "macro_bull"},
    }},
    "pair_k020": {"macro_sma_days": 200, "rules": {
        "wick_trap":           {"k": 0.20, "when": "macro_bear"},
        "exhaustion_reversal": {"k": 0.20, "when": "macro_bull"},
    }},
    "pair_k030": {"macro_sma_days": 200, "rules": {
        "wick_trap":           {"k": 0.30, "when": "macro_bear"},
        "exhaustion_reversal": {"k": 0.30, "when": "macro_bull"},
    }},
    "pair_k035": {"macro_sma_days": 200, "rules": {
        "wick_trap":           {"k": 0.35, "when": "macro_bear"},
        "exhaustion_reversal": {"k": 0.35, "when": "macro_bull"},
    }},
    "pair_k025_150d": {"macro_sma_days": 150, "rules": {
        "wick_trap":           {"k": 0.25, "when": "macro_bear"},
        "exhaustion_reversal": {"k": 0.25, "when": "macro_bull"},
    }},
    "pair_k025_250d": {"macro_sma_days": 250, "rules": {
        "wick_trap":           {"k": 0.25, "when": "macro_bear"},
        "exhaustion_reversal": {"k": 0.25, "when": "macro_bull"},
    }},
}

OUT_ROOT = REPO / "results/champion_overlay"


_MACRO_BEAR_CACHE = {}


def macro_bear_series(days: int):
    """Boolean Series over the FULL store: close < rolling 200-day mean.

    Computed from the full parquet (not the engine's window-filtered frame)
    so warmup is always available. Concurrent-state, deterministic.
    """
    if days not in _MACRO_BEAR_CACHE:
        import pandas as pd
        close = pd.read_parquet(
            REPO / "data/features_mtf/BTC_1H_FEATURES_V12_ENHANCED.parquet",
            columns=["close"],
        )["close"]
        bars = days * 24
        _MACRO_BEAR_CACHE[days] = close < close.rolling(bars, min_periods=bars // 2).mean()
    return _MACRO_BEAR_CACHE[days]


def install_overlay(k: float = None, regimes: set = None, macro_sma_days: int = None,
                    rules: dict = None):
    """Patch _open_position with a sizing overlay (never blocks entries).

    Modes:
      - regimes:  longs entered while engine regime_label in `regimes` → size * k
      - macro:    longs entered while close < N-day mean (full store) → size * k
      - rules:    per-archetype {k, when: macro_bear|macro_bull} using the
                  N-day macro state (regime-complementary pair mode)
    """
    from bin.backtest_v11_standalone import StandaloneBacktestEngine

    state = {"entries": 0, "scaled": 0, "regime_counts": {}, "scaled_by_arch": {}}
    original_open = StandaloneBacktestEngine._open_position
    bear = macro_bear_series(macro_sma_days) if macro_sma_days else None

    def macro_state(timestamp):
        try:
            return "macro_bear" if bool(bear.at[timestamp]) else "macro_bull"
        except KeyError:
            return None

    def patched_open(self, *args, **kwargs):
        direction = kwargs.get("direction") or (args[2] if len(args) > 2 else None)
        timestamp = kwargs.get("timestamp") or (args[0] if len(args) > 0 else None)
        archetype = kwargs.get("archetype") or (args[1] if len(args) > 1 else None)
        state["entries"] += 1
        if direction == "long" and timestamp is not None:
            if rules is not None:
                ms = macro_state(timestamp)
                if ms:
                    state["regime_counts"][ms] = state["regime_counts"].get(ms, 0) + 1
                rule = rules.get(archetype)
                if rule and ms == rule["when"] and "allocated_size_pct" in kwargs:
                    kwargs["allocated_size_pct"] *= rule["k"]
                    state["scaled"] += 1
                    state["scaled_by_arch"][archetype] = state["scaled_by_arch"].get(archetype, 0) + 1
            elif bear is not None:
                ms = macro_state(timestamp)
                if ms:
                    state["regime_counts"][ms] = state["regime_counts"].get(ms, 0) + 1
                if ms == "macro_bear" and "allocated_size_pct" in kwargs:
                    kwargs["allocated_size_pct"] *= k
                    state["scaled"] += 1
            else:
                try:
                    regime = self.features_df.at[timestamp, "regime_label"]
                except KeyError:
                    regime = None
                if regime is not None:
                    state["regime_counts"][regime] = state["regime_counts"].get(regime, 0) + 1
                if regime in regimes and "allocated_size_pct" in kwargs:
                    kwargs["allocated_size_pct"] *= k
                    state["scaled"] += 1
        return original_open(self, *args, **kwargs)

    StandaloneBacktestEngine._open_position = patched_open
    return state, lambda: setattr(StandaloneBacktestEngine, "_open_position", original_open)


def run_window_inproc(config_path: Path, start: str, end: str, out_dir: Path) -> dict:
    """Run one backtest window in-process (so the patch applies)."""
    from bin.backtest_v11_standalone import StandaloneBacktestEngine

    out_dir.mkdir(parents=True, exist_ok=True)
    stats_file = out_dir / "performance_stats.json"
    if stats_file.exists():
        return json.loads(stats_file.read_text())

    config = json.loads(config_path.read_text())
    engine = StandaloneBacktestEngine(
        config=config, initial_cash=100_000.0,
        commission_rate=0.0002, slippage_bps=3.0,
        feature_store_path=str(REPO / "data/features_mtf/BTC_1H_FEATURES_V12_ENHANCED.parquet"),
    )
    engine.run(start_date=start, end_date=end)
    stats = engine.get_performance_stats()
    engine.save_trade_log(str(out_dir / "trade_log.csv"))

    clean = {}
    for kk, v in stats.items():
        try:
            if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
                clean[kk] = str(v)
            else:
                json.dumps(v)
                clean[kk] = v
        except (TypeError, ValueError):
            clean[kk] = str(v)
    stats_file.write_text(json.dumps(clean, indent=2, default=str))
    return clean


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidate", nargs="+", required=True,
                    help="one archetype, or several for a pair/portfolio test")
    ap.add_argument("--variants", nargs="+", default=list(VARIANTS))
    args = ap.parse_args()

    logging.basicConfig(level=logging.WARNING)

    candidate = args.candidate[0] if len(args.candidate) == 1 else args.candidate
    label = "__".join(args.candidate)
    cfg = make_variant_config(candidate)
    cards = []
    for vname in args.variants:
        v = VARIANTS[vname]
        if "rules" in v:
            desc = f"rules={v['rules']}"
        elif "regimes" in v:
            desc = f"k={v['k']}, regimes={sorted(v['regimes'])}"
        else:
            desc = f"k={v['k']}, macro_sma={v['macro_sma_days']}d"
        print(f"\n=== {label} + {vname} ({desc}) ===")
        state, restore = install_overlay(v.get("k"), v.get("regimes"),
                                         v.get("macro_sma_days"), v.get("rules"))
        try:
            results = {}
            for w, (s, e) in WINDOWS.items():
                out_dir = OUT_ROOT / label / vname / w
                print(f"  [run ] {w}", flush=True)
                results[w] = run_window_inproc(cfg, s, e, out_dir)
        finally:
            restore()

        card = score(f"{label}+{vname}", results)
        card["overlay"] = {"variant": vname, **{kk: (sorted(vv) if isinstance(vv, set) else vv) for kk, vv in v.items()},
                           "entries": state["entries"], "scaled": state["scaled"],
                           "regime_counts": state["regime_counts"]}
        cards.append(card)
        (OUT_ROOT / label / vname / "scorecard.json").write_text(json.dumps(card, indent=2, default=str))
        for kk, vv in card["criteria"].items():
            print(f"    {'PASS' if vv else 'FAIL'}  {kk}")
        print(f"  OVERALL: {'PASS' if card['pass'] else 'FAIL'} | scaled {state['scaled']}/{state['entries']} entries")

    print(f"\n{'variant':<16}{'pass':<6}{'2020':>8}{'2021':>8}{'2022':>8}{'2023':>8}{'2024':>8}{'full_pf':>8}{'mdd':>6}")
    for c in cards:
        d = c["detail"]
        print(f"{c['overlay']['variant']:<16}{str(c['pass']):<6}"
              f"{d['y2020']['pnl']:>8}{d['y2021']['pnl']:>8}{d['y2022']['pnl']:>8}"
              f"{d['y2023']['pnl']:>8}{d['y2024']['pnl']:>8}"
              f"{d['full']['pf']:>8}{d['full']['maxdd_pct']:>6}")


if __name__ == "__main__":
    main()
