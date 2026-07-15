#!/usr/bin/env python3
"""Pre-registered early-weakness TIME-CUT grid (Part 2 of path-conditional study).

Policy: exit at market if a position's max favorable excursion has not reached
+X R within H hours. Grid (pre-registered, no extensions): X in {0.25, 0.5} x
H in {12, 24, 48}. Per archetype (wick_trap, liquidity_compression), windows
train/mid/holdout. Real backtest via the _check_all_exits patch pattern.
Acceptance: train AND holdout PF >= baseline (Rule 9 co-move), per archetype.
Motivation: P(win | MFE<0.25R after 12h) = 0%/0% for LC (train/mid).
"""
import json, math, sys
from pathlib import Path
import pandas as pd
REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO/"scripts/champion"))
from run_battery import ALL_ARCHETYPES, BASE_CONFIG, CONFIG_DIR
from bin.backtest_v11_standalone import StandaloneBacktestEngine

STORE = REPO/"data/features_mtf/BTC_1H_FEATURES_V14L_LEVELS.parquet"
OUT = REPO/"results/timecut_grid"
W = {"train":("2018-01-01","2022-12-31"),"mid":("2023-01-01","2024-12-31"),"holdout":("2025-01-01","2026-06-10")}
CELLS = [None] + [(x,h) for x in (0.25,0.5) for h in (12,24,48)]

def cfg_for(arch):
    c = json.loads(Path(BASE_CONFIG).read_text())
    c["disabled_archetypes"] = [a for a in ALL_ARCHETYPES if a != arch]
    c.setdefault("adaptive_fusion",{})["bypass_threshold"] = False
    c["archetype_config_dir"] = "configs/champion/archetypes_v14rq/"
    p = CONFIG_DIR/f"tc_{arch}.json"; p.write_text(json.dumps(c, indent=2)); return p

def install(x_r, h_bars):
    orig = StandaloneBacktestEngine._check_all_exits
    state = {}  # pid -> [bars_held, max_fav_R]
    def patched(self, row, ts, bar_idx):
        # update path state and apply time-cut BEFORE normal exits this bar
        for pid in list(self.positions.keys()):
            pos = self.positions[pid]
            R = abs(pos.entry_price - pos.stop_loss) or 1e-9
            fav = (row["high"] - pos.entry_price)/R if pos.direction == "long" else (pos.entry_price - row["low"])/R
            bars, mx = state.get(pid, [0, -9.9])
            bars += 1; mx = max(mx, fav); state[pid] = [bars, mx]
            if bars >= h_bars and mx < x_r:
                self._close_position(pid, float(row["close"]), ts,
                                     exit_reason=f"timecut_{x_r}R_{h_bars}h", exit_pct=1.0)
        return orig(self, row, ts, bar_idx)
    StandaloneBacktestEngine._check_all_exits = patched
    return lambda: setattr(StandaloneBacktestEngine, "_check_all_exits", orig)

def run(cp, s, e, od, cell):
    od.mkdir(parents=True, exist_ok=True); sf = od/"performance_stats.json"
    if sf.exists(): return json.loads(sf.read_text())
    restore = install(*cell) if cell else (lambda: None)
    try:
        eng = StandaloneBacktestEngine(config=json.loads(cp.read_text()), initial_cash=100000.0,
            commission_rate=0.0002, slippage_bps=3.0, feature_store_path=str(STORE))
        eng.run(start_date=s, end_date=e)
        st = eng.get_performance_stats()
    finally:
        restore()
    clean = {k:(str(v) if isinstance(v,float) and (math.isnan(v) or math.isinf(v)) else v) for k,v in st.items()}
    sf.write_text(json.dumps(clean, indent=2, default=str)); return clean

def num(v):
    try: f=float(v); return 0.0 if f!=f else f
    except (TypeError,ValueError): return 0.0

def main():
    OUT.mkdir(parents=True, exist_ok=True)
    for arch in ("wick_trap","liquidity_compression"):
        cp = cfg_for(arch)
        print(f"\n===== {arch} =====", flush=True)
        base = {}
        rows = []
        for cell in CELLS:
            name = "baseline" if cell is None else f"x{cell[0]}_h{cell[1]}"
            res = {w: run(cp, s, e, OUT/arch/name/w, cell) for w,(s,e) in W.items()}
            r = {w: (round(num(v.get("profit_factor")),2), round(num(v.get("total_pnl")))) for w,v in res.items()}
            if cell is None: base = r
            rows.append((name, r))
            print(f"  {name:<12} train PF {r['train'][0]} (${r['train'][1]:,}) | mid {r['mid'][0]} | holdout {r['holdout'][0]} (${r['holdout'][1]:,})", flush=True)
        print(f"  --- verdict (train AND holdout PF >= baseline) ---", flush=True)
        for name, r in rows[1:]:
            ok = r['train'][0] >= base['train'][0] and r['holdout'][0] >= base['holdout'][0]
            print(f"    {name}: {'PASS' if ok else 'fail'}", flush=True)
    print("TIMECUT DONE", flush=True)

if __name__ == "__main__": main()
