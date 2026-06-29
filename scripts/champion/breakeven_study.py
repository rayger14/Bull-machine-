#!/usr/bin/env python3
"""Breakeven-stop-after-1R study (V14, WFO).

MFE analysis (mfe_analysis_2026_06_28) found we already capture 84% of winners'
high-water mark, but 14% of LOSERS (exhaustion_reversal 24%) were +1R before
reversing to a loss. Hypothesis: moving the stop to breakeven once a trade
reaches +1R converts those give-back losers into scratches, lifting PF.

Mechanism (monkey-patch _check_exits, no production change): once a bar's high
(long) reaches entry + trigger*R, raise the stop to entry + buffer*R — applied
from the NEXT bar (current-bar exits processed first; no intrabar lookahead).
R = entry-to-initial-stop, captured once per position.

Trade-off being measured: saved give-back losers vs trades that dip to breakeven
then recover (cut early). Only WFO settles it. Acceptance: train AND holdout PF
both improve (Rule 9), per-archetype.

Grid: baseline | BE@1.0R buf0 | BE@1.0R buf+0.1R | BE@1.5R buf0
Archetypes: exhaustion_reversal, wick_trap_v14rq, liquidity_sweep, spring.
Usage: python3 scripts/champion/breakeven_study.py
"""
from __future__ import annotations

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
V14RQ_DIR = "configs/champion/archetypes_v14rq/"
OUT_ROOT = REPO / "results/champion_v14_breakeven"

WINDOWS = {
    "y2018": ("2018-01-01", "2018-12-31"), "y2019": ("2019-01-01", "2019-12-31"),
    "y2020": ("2020-01-01", "2020-12-31"), "y2021": ("2021-01-01", "2021-12-31"),
    "y2022": ("2022-01-01", "2022-12-31"), "y2023": ("2023-01-01", "2023-12-31"),
    "y2024": ("2024-01-01", "2024-12-31"),
    "wfo_train": ("2018-01-01", "2022-12-31"),
    "holdout_2025_26": ("2025-01-01", "2026-06-10"),
    "full": ("2018-01-01", "2024-12-31"),
}

ARCHETYPES = ["exhaustion_reversal", "wick_trap", "liquidity_sweep", "spring"]
VARIANTS = {  # (trigger_R, buffer_R) or None for baseline
    "baseline": None,
    "be10_b0": (1.0, 0.0),
    "be10_b10": (1.0, 0.1),
    "be15_b0": (1.5, 0.0),
}


def cfg_for(archetype: str) -> Path:
    cfg = json.loads(BASE_CONFIG.read_text())
    cfg["disabled_archetypes"] = [a for a in ALL_ARCHETYPES if a != archetype]
    cfg.setdefault("adaptive_fusion", {})["bypass_threshold"] = False
    cfg["archetype_config_dir"] = V14RQ_DIR
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    p = CONFIG_DIR / f"be_{archetype}.json"
    p.write_text(json.dumps(cfg, indent=2))
    return p


def install_be(trigger_R, buffer_R):
    from bin.backtest_v11_standalone import StandaloneBacktestEngine
    orig = StandaloneBacktestEngine._check_all_exits
    R0 = {}  # pos_id -> initial R; moved flag tracked in a set
    moved = set()

    def patched(self, row, ts, bar_idx):
        result = orig(self, row, ts, bar_idx)  # current-bar exits FIRST (no lookahead)
        high, low = row.get("high"), row.get("low")
        for pid, pos in self.positions.items():
            if pid not in R0:
                R0[pid] = abs(pos.entry_price - pos.stop_loss)
            R = R0[pid]
            if pid in moved or R <= 0:
                continue
            if pos.direction == "long" and high is not None and high >= pos.entry_price + trigger_R * R:
                be = pos.entry_price + buffer_R * R
                pos.stop_loss = max(pos.stop_loss, be)
                if pos.trailing_stop is not None and pos.trailing_stop < be:
                    pos.trailing_stop = be
                moved.add(pid)
            elif pos.direction == "short" and low is not None and low <= pos.entry_price - trigger_R * R:
                be = pos.entry_price - buffer_R * R
                pos.stop_loss = min(pos.stop_loss, be)
                if pos.trailing_stop is not None and pos.trailing_stop > be:
                    pos.trailing_stop = be
                moved.add(pid)
        return result

    StandaloneBacktestEngine._check_all_exits = patched
    return lambda: setattr(StandaloneBacktestEngine, "_check_all_exits", orig)


def run_window(cfg_path, start, end, out_dir, be):
    from bin.backtest_v11_standalone import StandaloneBacktestEngine
    out_dir.mkdir(parents=True, exist_ok=True)
    sf = out_dir / "performance_stats.json"
    if sf.exists():
        return json.loads(sf.read_text())
    restore = install_be(*be) if be else (lambda: None)
    try:
        eng = StandaloneBacktestEngine(config=json.loads(cfg_path.read_text()),
            initial_cash=100_000.0, commission_rate=0.0002, slippage_bps=3.0,
            feature_store_path=str(STORE))
        eng.run(start_date=start, end_date=end)
        stats = eng.get_performance_stats()
    finally:
        restore()
    clean = {}
    for k, v in stats.items():
        try:
            clean[k] = str(v) if (isinstance(v, float) and (math.isnan(v) or math.isinf(v))) else (json.loads(json.dumps(v)) if False else v)
        except (TypeError, ValueError):
            clean[k] = str(v)
    sf.write_text(json.dumps(clean, indent=2, default=str))
    return clean


def main():
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    table = []
    for a in ARCHETYPES:
        cfg = cfg_for(a)
        per_variant = {}
        for vname, be in VARIANTS.items():
            results = {w: run_window(cfg, s, e, OUT_ROOT / a / vname / w, be)
                       for w, (s, e) in WINDOWS.items()}
            card = score(f"{a}/{vname}", results)
            (OUT_ROOT / a / vname / "scorecard.json").write_text(json.dumps(card, indent=2, default=str))
            per_variant[vname] = card["detail"]
        table.append((a, per_variant))
        print(f"\n=== {a} ===")
        print(f"  {'variant':<12}{'train_pf':>9}{'hold_pf':>8}{'hold_pnl':>9}{'full_pf':>8}{'full_pnl':>9}{'2022':>8}{'mdd':>6}")
        for vname in VARIANTS:
            d = per_variant[vname]
            print(f"  {vname:<12}{d['wfo_train']['pf']:>9}{d['holdout_2025_26']['pf']:>8}"
                  f"{d['holdout_2025_26']['pnl']:>9}{d['full']['pf']:>8}{d['full']['pnl']:>9}"
                  f"{d['y2022']['pnl']:>8}{d['full']['maxdd_pct']:>6}")

    (OUT_ROOT / "summary.json").write_text(json.dumps(
        [{"archetype": a, "variants": v} for a, v in table], indent=2, default=str))
    print(f"\nreport -> {OUT_ROOT / 'summary.json'}")
    # verdict: variants beating baseline on BOTH train and holdout PF, per archetype
    print("\n=== verdict (BE beats baseline on train AND holdout PF?) ===")
    for a, pv in table:
        b = pv["baseline"]
        for vname in VARIANTS:
            if vname == "baseline":
                continue
            d = pv[vname]
            better = (d["wfo_train"]["pf"] > b["wfo_train"]["pf"] and
                      d["holdout_2025_26"]["pf"] > b["holdout_2025_26"]["pf"])
            if better:
                print(f"  {a}/{vname}: train {b['wfo_train']['pf']}->{d['wfo_train']['pf']}, "
                      f"holdout {b['holdout_2025_26']['pf']}->{d['holdout_2025_26']['pf']}  CO-MOVE UP")


if __name__ == "__main__":
    main()
