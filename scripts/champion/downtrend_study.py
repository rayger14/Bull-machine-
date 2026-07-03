#!/usr/bin/env python3
"""Downtrend scoping study (V14) — the direction fix.

Every prior study says the bleed is long-only-into-downtrends, not bad signals.
This scopes the fix in three parts, cheapest first:

  STEP 1 — DOWNTREND DETECTOR: a few candidate definitions from V14 price.
  STEP 2 — DEFENSIVE SKIP: skip long entries when the detector fires. Does it
           cut the loss / drawdown across 2018-2026 (esp. bears)? Real skip
           (position never opens — no size=0 accounting bug).
  STEP 3 — OFFENSIVE FEASIBILITY: during detected downtrends, what does price do
           next (fwd 24h/72h return)? If reliably negative, shorting has edge —
           the go/no-go for building a short side.

Read-only research; full-16 book, thresholds as configured. No production change.
Usage: python3 scripts/champion/downtrend_study.py
"""
from __future__ import annotations
import json, math, sys
from pathlib import Path
import numpy as np, pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO/"scripts/champion"))
from run_battery import ALL_ARCHETYPES, BASE_CONFIG, CONFIG_DIR, score

STORE = REPO/"data/features_mtf/BTC_1H_FEATURES_V14_FULL_LIVE_PATH.parquet"
OUT = REPO/"results/champion_v14_downtrend"
WINDOWS = {"y2018":("2018-01-01","2018-12-31"),"y2019":("2019-01-01","2019-12-31"),
           "y2020":("2020-01-01","2020-12-31"),"y2021":("2021-01-01","2021-12-31"),
           "y2022":("2022-01-01","2022-12-31"),"y2023":("2023-01-01","2023-12-31"),
           "y2024":("2024-01-01","2024-12-31"),"wfo_train":("2018-01-01","2022-12-31"),
           "holdout_2025_26":("2025-01-01","2026-06-10"),"full":("2018-01-01","2024-12-31")}

_DET = {}
def detectors():
    """Return dict name -> boolean downtrend Series over the full store."""
    if _DET: return _DET
    df = pd.read_parquet(STORE, columns=["close","ema_50_above_200"])
    if df.index.tz is not None: df.index = df.index.tz_localize(None)
    c = df["close"]
    _DET["dt_200d"] = c < c.rolling(200*24, min_periods=100*24).mean()   # macro
    _DET["dt_50d"]  = c < c.rolling(50*24,  min_periods=25*24).mean()    # faster
    _DET["dt_death"] = df["ema_50_above_200"].fillna(1) == 0             # death cross
    # combo: 200d AND price falling (below 50d too) = high-conviction downtrend
    _DET["dt_combo"] = _DET["dt_200d"] & _DET["dt_50d"]
    return _DET

def full_cfg():
    cfg = json.loads(BASE_CONFIG.read_text()); cfg["disabled_archetypes"]=[]
    cfg.setdefault("adaptive_fusion",{})["bypass_threshold"]=False  # enforced (clean read)
    cfg["archetype_config_dir"]="configs/champion/archetypes_v14rq/"
    CONFIG_DIR.mkdir(parents=True,exist_ok=True)
    p=CONFIG_DIR/"downtrend_full.json"; p.write_text(json.dumps(cfg,indent=2)); return p

def install_skip(dt: pd.Series):
    """Patch _open_position to SKIP long opens when downtrend fires (real skip)."""
    from bin.backtest_v11_standalone import StandaloneBacktestEngine
    orig = StandaloneBacktestEngine._open_position
    st = {"seen":0,"skipped":0}
    def patched(self,*a,**k):
        ts = k.get("timestamp") or (a[0] if a else None)
        direction = k.get("direction") or (a[2] if len(a)>2 else None)
        st["seen"]+=1
        if direction=="long" and ts is not None:
            try:
                if bool(dt.at[ts]):
                    st["skipped"]+=1; return None    # skip: position never opens
            except KeyError: pass
        return orig(self,*a,**k)
    StandaloneBacktestEngine._open_position = patched
    return st, lambda: setattr(StandaloneBacktestEngine,"_open_position",orig)

def run(cfg,start,end,out_dir,dt=None):
    from bin.backtest_v11_standalone import StandaloneBacktestEngine
    out_dir.mkdir(parents=True,exist_ok=True); sf=out_dir/"performance_stats.json"
    if sf.exists(): return json.loads(sf.read_text())
    restore = install_skip(dt)[1] if dt is not None else (lambda:None)
    try:
        eng=StandaloneBacktestEngine(config=json.loads(cfg.read_text()),initial_cash=100000.0,
            commission_rate=0.0002,slippage_bps=3.0,feature_store_path=str(STORE))
        eng.run(start_date=start,end_date=end); stats=eng.get_performance_stats()
    finally: restore()
    clean={kk:(str(v) if isinstance(v,float) and (math.isnan(v) or math.isinf(v)) else v) for kk,v in stats.items()}
    sf.write_text(json.dumps(clean,indent=2,default=str)); return clean

def main():
    OUT.mkdir(parents=True,exist_ok=True); cfg=full_cfg(); dets=detectors()
    # coverage: what fraction of bars each detector flags, and per-year
    print("=== STEP 1: detector coverage (% bars flagged downtrend) ===")
    for n,s in dets.items():
        by=s.groupby(s.index.year).mean()
        print(f"  {n:<10} overall {100*s.mean():4.0f}% | 2018 {100*by.get(2018,0):3.0f} 2020 {100*by.get(2020,0):3.0f} 2022 {100*by.get(2022,0):3.0f} 2025 {100*by.get(2025,0):3.0f}")

    print("\n=== STEP 3: SHORT FEASIBILITY — fwd returns during detected downtrends ===")
    px = pd.read_parquet(STORE, columns=["close"]);
    if px.index.tz is not None: px.index=px.index.tz_localize(None)
    c=px["close"]; f24=c.shift(-24)/c-1; f72=c.shift(-72)/c-1
    print(f"  {'detector':<10}{'baseline_all':>14}{'in_downtrend_24h':>18}{'72h':>10}{'%neg_72h':>10}")
    base24=f24.mean()
    for n,s in dets.items():
        m=s & f72.notna()
        print(f"  {n:<10}{100*base24:>13.2f}%{100*f24[m].mean():>17.2f}%{100*f72[m].mean():>9.2f}%{100*(f72[m]<0).mean():>9.0f}%")
    print("  (if in-downtrend fwd returns are reliably NEGATIVE, a short side has edge)")

    print("\n=== STEP 2: DEFENSIVE SKIP — full-16 book, skip longs in downtrend ===")
    variants={"baseline":None, **{f"skip_{n}":s for n,s in dets.items()}}
    cards={}
    for vn,dt in variants.items():
        res={w:run(cfg,s,e,OUT/vn/w,dt) for w,(s,e) in WINDOWS.items()}
        cards[vn]=score(vn,res)["detail"]
    print(f"{'variant':<16}{'2018':>8}{'2022':>8}{'hold_pnl':>9}{'full_pnl':>9}{'full_pf':>8}{'mdd':>6}")
    for vn,d in cards.items():
        print(f"{vn:<16}{d['y2018']['pnl']:>8}{d['y2022']['pnl']:>8}{d['holdout_2025_26']['pnl']:>9}"
              f"{d['full']['pnl']:>9}{d['full']['pf']:>8}{d['full']['maxdd_pct']:>6}")
    (OUT/"summary.json").write_text(json.dumps(cards,indent=2,default=str))
    print(f"\nreport -> {OUT/'summary.json'}")

if __name__=="__main__": main()
