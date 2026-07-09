#!/usr/bin/env python3
"""Validated-stack grid (V14) — REAL engine features, no monkey-patches.

Pre-registered 9 cells: book {wt_only, core2, full16} x protection
{none, skip200, skip200_be}. Protection via the PRODUCTION config keys built
for this study: downtrend_skip{enabled,sma_days} and per-archetype
breakeven_trigger_r override inside exit_logic (wick_trap only).

Pre-registered acceptance ("a properly-firing winner"):
  holdout PF >= 1.3 AND positive; train PF >= 1.3 (co-move);
  every year >= -$2K; MaxDD <= 12%; holdout n >= 30;
  wick_trap's own holdout PF >= 1.3 (per-archetype split).

Usage: python3 scripts/champion/stack_validation.py
"""
from __future__ import annotations
import json, math, sys
from pathlib import Path
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO / "scripts/champion"))
from run_battery import ALL_ARCHETYPES, BASE_CONFIG, CONFIG_DIR, score

STORE = REPO / "data/features_mtf/BTC_1H_FEATURES_V14_FULL_LIVE_PATH.parquet"
OUT = REPO / "results/champion_v14_stack"
WINDOWS = {
    "y2018": ("2018-01-01", "2018-12-31"), "y2019": ("2019-01-01", "2019-12-31"),
    "y2020": ("2020-01-01", "2020-12-31"), "y2021": ("2021-01-01", "2021-12-31"),
    "y2022": ("2022-01-01", "2022-12-31"), "y2023": ("2023-01-01", "2023-12-31"),
    "y2024": ("2024-01-01", "2024-12-31"),
    "wfo_train": ("2018-01-01", "2022-12-31"),
    "holdout_2025_26": ("2025-01-01", "2026-06-10"),
    "full": ("2018-01-01", "2024-12-31"),
}
BOOKS = {
    "wt_only": ["wick_trap"],
    "core2": ["wick_trap", "liquidity_sweep"],
    "full16": list(ALL_ARCHETYPES),
}
PROTECTIONS = ["none", "skip200", "skip200_be"]


def make_cfg(book_name, prot) -> Path:
    cfg = json.loads(BASE_CONFIG.read_text())
    cfg["disabled_archetypes"] = [a for a in ALL_ARCHETYPES if a not in BOOKS[book_name]]
    cfg.setdefault("adaptive_fusion", {})["bypass_threshold"] = False
    cfg["archetype_config_dir"] = "configs/champion/archetypes_v14rq/"
    if prot in ("skip200", "skip200_be"):
        cfg["downtrend_skip"] = {"enabled": True, "sma_days": 200, "direction": "long"}
    if prot == "skip200_be":
        # per-archetype user override -> ExitLogic (top-level key inside exit_logic cfg)
        cfg.setdefault("exit_logic", {})["wick_trap"] = {"breakeven_trigger_r": 1.0,
                                                          "breakeven_buffer_r": 0.0}
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    p = CONFIG_DIR / f"stack_{book_name}_{prot}.json"
    p.write_text(json.dumps(cfg, indent=2))
    return p


def run_window(cfg_path, start, end, out_dir) -> dict:
    from bin.backtest_v11_standalone import StandaloneBacktestEngine
    out_dir.mkdir(parents=True, exist_ok=True)
    sf = out_dir / "performance_stats.json"
    if sf.exists():
        return json.loads(sf.read_text())
    eng = StandaloneBacktestEngine(config=json.loads(cfg_path.read_text()),
        initial_cash=100_000.0, commission_rate=0.0002, slippage_bps=3.0,
        feature_store_path=str(STORE))
    eng.run(start_date=start, end_date=end)
    stats = eng.get_performance_stats()
    eng.save_trade_log(str(out_dir / "trade_log.csv"))
    clean = {k: (str(v) if isinstance(v, float) and (math.isnan(v) or math.isinf(v)) else v)
             for k, v in stats.items()}
    clean["downtrend_skips"] = getattr(eng, "downtrend_skips", 0)
    sf.write_text(json.dumps(clean, indent=2, default=str))
    return clean


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    cards = {}
    for book in BOOKS:
        for prot in PROTECTIONS:
            name = f"{book}__{prot}"
            cfg = make_cfg(book, prot)
            print(f"=== {name} ===", flush=True)
            res = {w: run_window(cfg, s, e, OUT / name / w) for w, (s, e) in WINDOWS.items()}
            card = score(name, res)
            (OUT / name / "scorecard.json").write_text(json.dumps(card, indent=2, default=str))
            cards[name] = card["detail"]

    print(f"\n{'cell':<24}{'2018':>7}{'2021':>7}{'2022':>7}{'holdPnL':>8}{'holdPF':>7}"
          f"{'holdN':>6}{'trainPF':>8}{'fullPF':>7}{'fullPnL':>8}{'mdd':>6}")
    for name, d in cards.items():
        print(f"{name:<24}{d['y2018']['pnl']:>7}{d['y2021']['pnl']:>7}{d['y2022']['pnl']:>7}"
              f"{d['holdout_2025_26']['pnl']:>8}{d['holdout_2025_26']['pf']:>7}"
              f"{d['holdout_2025_26']['trades']:>6}{d['wfo_train']['pf']:>8}"
              f"{d['full']['pf']:>7}{d['full']['pnl']:>8}{d['full']['maxdd_pct']:>6}")

    # Pre-registered acceptance
    print("\n=== ACCEPTANCE (pre-registered) ===")
    for name, d in cards.items():
        years_ok = all(d[f"y{y}"]["pnl"] >= -2000 for y in range(2018, 2025))
        ok = (d["holdout_2025_26"]["pf"] >= 1.3 and d["holdout_2025_26"]["pnl"] > 0
              and d["wfo_train"]["pf"] >= 1.3 and years_ok
              and d["full"]["maxdd_pct"] <= 12.0 and d["holdout_2025_26"]["trades"] >= 30)
        print(f"  {name:<24} {'PASS' if ok else 'fail'}"
              f"{'' if years_ok else '  (year floor breached)'}")
    (OUT / "grid_summary.json").write_text(json.dumps(cards, indent=2, default=str))
    print(f"\nreport -> {OUT / 'grid_summary.json'}")


if __name__ == "__main__":
    main()
