#!/usr/bin/env python3
"""Unified-strategy grid: does LEVEL ANCHORING improve the validated wick_trap?

Pre-registered variants (gates ADDED to repaired wick_trap; nothing else changes):
  baseline  repaired wick_trap as-is (holdout PF 1.43)
  U1_sweep  + sweep_low_event bool_true      (flush must sweep prior-day/swing low)
  U2_qual   + level_quality_low min 0.5      (flush at an earned level; nan skip)
  U3_near   + dist_to_support_atr max 2.0    (flush NEAR support, no-chase; nan skip)
Store: V14L (V14 + level features). Windows: per-year + train + holdout + full.
Win condition: train AND holdout PF >= baseline, holdout n >= 30 (Rule 9 co-move).
Honest outcome "levels add nothing" is expected-possible (prior evidence mixed).
"""
import json, math, shutil, sys
from pathlib import Path
import yaml
REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO/"scripts/champion"))
from run_battery import ALL_ARCHETYPES, BASE_CONFIG, CONFIG_DIR
from bin.backtest_v11_standalone import StandaloneBacktestEngine

STORE = REPO/"data/features_mtf/BTC_1H_FEATURES_V14L_LEVELS.parquet"
UDIR = REPO/"configs/champion/archetypes_unified"
OUT = REPO/"results/champion_unified"
WINDOWS = {**{f"y{y}": (f"{y}-01-01", f"{y}-12-31") for y in range(2018, 2025)},
           "wfo_train": ("2018-01-01","2022-12-31"),
           "holdout_2025_26": ("2025-01-01","2026-06-10"),
           "full": ("2018-01-01","2024-12-31")}
VARIANTS = {
    "baseline": None,
    "U1_sweep": {"feature":"sweep_low_event","op":"bool_true","description":"UNIFIED: flush must sweep a tracked level (prior-day/swing low) [pre-registered 2026-07-13]"},
    "U2_qual":  {"feature":"level_quality_low","op":"min","value":0.5,"nan_policy":"skip","description":"UNIFIED: flush at an earned level (quality>=0.5) [pre-registered 2026-07-13]"},
    "U3_near":  {"feature":"dist_to_support_atr","op":"max","value":2.0,"nan_policy":"skip","description":"UNIFIED: entry within 2 ATR of support (no-chase) [pre-registered 2026-07-13]"},
}
def set_variant(extra_gate):
    src = REPO/"configs/champion/archetypes_v14rq/wick_trap.yaml"
    c = yaml.safe_load(open(src))
    if extra_gate: c["hard_gates"] = list(c["hard_gates"]) + [extra_gate]
    yaml.safe_dump(c, open(UDIR/"wick_trap.yaml","w"), sort_keys=False)
def cfg():
    c = json.loads(Path(BASE_CONFIG).read_text())
    c["disabled_archetypes"] = [a for a in ALL_ARCHETYPES if a != "wick_trap"]
    c.setdefault("adaptive_fusion",{})["bypass_threshold"] = False
    c["archetype_config_dir"] = "configs/champion/archetypes_unified/"
    p = CONFIG_DIR/"unified_wt.json"; p.write_text(json.dumps(c, indent=2)); return p
def run(cp, s, e, od):
    od.mkdir(parents=True, exist_ok=True); sf = od/"performance_stats.json"
    if sf.exists(): return json.loads(sf.read_text())
    eng = StandaloneBacktestEngine(config=json.loads(cp.read_text()), initial_cash=100000.0,
        commission_rate=0.0002, slippage_bps=3.0, feature_store_path=str(STORE))
    eng.run(start_date=s, end_date=e)
    st = eng.get_performance_stats()
    clean = {k:(str(v) if isinstance(v,float) and (math.isnan(v) or math.isinf(v)) else v) for k,v in st.items()}
    sf.write_text(json.dumps(clean, indent=2, default=str)); return clean
def num(v):
    try: f=float(v); return 0.0 if f!=f else f
    except (TypeError,ValueError): return 0.0
def main():
    OUT.mkdir(parents=True, exist_ok=True); table = {}
    for vn, gate in VARIANTS.items():
        set_variant(gate); cp = cfg()
        print(f"=== {vn} ===", flush=True)
        res = {w: run(cp, s, e, OUT/vn/w) for w,(s,e) in WINDOWS.items()}
        table[vn] = {w: {"pf": round(num(r.get("profit_factor")),2),
                         "pnl": round(num(r.get("total_pnl"))),
                         "n": int(num(r.get("total_trades")))} for w,r in res.items()}
    print(f"\n{'variant':<10}{'trainPF':>8}{'holdPF':>7}{'holdPnL':>8}{'holdN':>6}{'fullPF':>7}{'fullPnL':>8}{'2022':>7}")
    b = table["baseline"]
    for vn,d in table.items():
        print(f"{vn:<10}{d['wfo_train']['pf']:>8}{d['holdout_2025_26']['pf']:>7}"
              f"{d['holdout_2025_26']['pnl']:>8}{d['holdout_2025_26']['n']:>6}"
              f"{d['full']['pf']:>7}{d['full']['pnl']:>8}{d['y2022']['pnl']:>7}")
    print("\n=== verdict (train AND holdout PF >= baseline, holdN >= 30) ===")
    for vn,d in table.items():
        if vn == "baseline": continue
        ok = (d['wfo_train']['pf'] >= b['wfo_train']['pf'] and
              d['holdout_2025_26']['pf'] >= b['holdout_2025_26']['pf'] and
              d['holdout_2025_26']['n'] >= 30)
        print(f"  {vn}: {'PASS' if ok else 'fail'}")
    (OUT/"grid.json").write_text(json.dumps(table, indent=2))
    set_variant(None)  # restore dir to baseline copy
    print("GRID DONE")
if __name__ == "__main__": main()
