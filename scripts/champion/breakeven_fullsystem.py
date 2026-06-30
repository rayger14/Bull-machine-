#!/usr/bin/env python3
"""Full-system validation of the wick_trap breakeven-at-1R exit.

The isolated study confounded the exit change with entry reshuffling. This runs
the FULL 16-archetype book (thresholds enforced) and applies BE@1R ONLY to the
named archetypes' positions — the deployment-realistic test the quant verdict
required: does wick_trap's clean edge survive in the competitive book, and does
liquidity_sweep's "reshuffle" gain persist or evaporate?

Variants: full_baseline | be_wicktrap | be_wt_plus_liqsweep
Reports full-system PnL/PF/MaxDD + per-archetype PnL deltas (esp. wick_trap).
"""
from __future__ import annotations
import json, math, sys
from pathlib import Path
REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO/"scripts/champion"))
from run_battery import ALL_ARCHETYPES, BASE_CONFIG, CONFIG_DIR, score

STORE = REPO/"data/features_mtf/BTC_1H_FEATURES_V14_FULL_LIVE_PATH.parquet"
OUT = REPO/"results/champion_v14_be_fullsystem"
WINDOWS = {"y2022":("2022-01-01","2022-12-31"),"y2023":("2023-01-01","2023-12-31"),
           "y2024":("2024-01-01","2024-12-31"),"wfo_train":("2018-01-01","2022-12-31"),
           "holdout_2025_26":("2025-01-01","2026-06-10"),"full":("2018-01-01","2024-12-31")}

def full_cfg():
    cfg=json.loads(BASE_CONFIG.read_text())
    cfg["disabled_archetypes"]=[]
    cfg.setdefault("adaptive_fusion",{})["bypass_threshold"]=False
    cfg["archetype_config_dir"]="configs/champion/archetypes_v14rq/"
    CONFIG_DIR.mkdir(parents=True,exist_ok=True)
    p=CONFIG_DIR/"be_fullsystem.json"; p.write_text(json.dumps(cfg,indent=2)); return p

def install_be(archset, trig=1.0, buf=0.0):
    from bin.backtest_v11_standalone import StandaloneBacktestEngine
    orig=StandaloneBacktestEngine._check_all_exits; R0={}; moved=set()
    def patched(self,row,ts,bar_idx):
        res=orig(self,row,ts,bar_idx); high,low=row.get("high"),row.get("low")
        for pid,pos in self.positions.items():
            if getattr(pos,"archetype",None) not in archset: continue
            if pid not in R0: R0[pid]=abs(pos.entry_price-pos.stop_loss)
            R=R0[pid]
            if pid in moved or R<=0: continue
            if pos.direction=="long" and high is not None and high>=pos.entry_price+trig*R:
                be=pos.entry_price+buf*R; pos.stop_loss=max(pos.stop_loss,be)
                if pos.trailing_stop is not None and pos.trailing_stop<be: pos.trailing_stop=be
                moved.add(pid)
            elif pos.direction=="short" and low is not None and low<=pos.entry_price-trig*R:
                be=pos.entry_price-buf*R; pos.stop_loss=min(pos.stop_loss,be)
                if pos.trailing_stop is not None and pos.trailing_stop>be: pos.trailing_stop=be
                moved.add(pid)
        return res
    StandaloneBacktestEngine._check_all_exits=patched
    return lambda: setattr(StandaloneBacktestEngine,"_check_all_exits",orig)

def run(cfg,start,end,out_dir,archset):
    from bin.backtest_v11_standalone import StandaloneBacktestEngine
    out_dir.mkdir(parents=True,exist_ok=True); sf=out_dir/"performance_stats.json"
    if sf.exists(): return json.loads(sf.read_text())
    restore=install_be(archset) if archset else (lambda:None)
    try:
        eng=StandaloneBacktestEngine(config=json.loads(cfg.read_text()),initial_cash=100000.0,
            commission_rate=0.0002,slippage_bps=3.0,feature_store_path=str(STORE))
        eng.run(start_date=start,end_date=end); stats=eng.get_performance_stats()
        eng.save_trade_log(str(out_dir/"trade_log.csv"))
    finally: restore()
    clean={k:(str(v) if isinstance(v,float) and (math.isnan(v) or math.isinf(v)) else v) for k,v in stats.items()}
    sf.write_text(json.dumps(clean,indent=2,default=str)); return clean

VARIANTS={"full_baseline":None,"be_wicktrap":{"wick_trap"},"be_wt_liqsweep":{"wick_trap","liquidity_sweep"}}
def main():
    OUT.mkdir(parents=True,exist_ok=True); cfg=full_cfg(); cards={}
    for vn,archset in VARIANTS.items():
        res={w:run(cfg,s,e,OUT/vn/w,archset) for w,(s,e) in WINDOWS.items()}
        cards[vn]=score(vn,res)["detail"]
    print(f"{'variant':<16}{'train_pf':>9}{'hold_pf':>8}{'hold_pnl':>9}{'full_pf':>8}{'full_pnl':>9}{'2022':>8}{'mdd':>6}")
    for vn in VARIANTS:
        d=cards[vn]
        print(f"{vn:<16}{d['wfo_train']['pf']:>9}{d['holdout_2025_26']['pf']:>8}{d['holdout_2025_26']['pnl']:>9}"
              f"{d['full']['pf']:>8}{d['full']['pnl']:>9}{d['y2022']['pnl']:>8}{d['full']['maxdd_pct']:>6}")
    # per-archetype wick_trap PnL in full vs holdout, baseline vs be
    import pandas as pd
    print("\n=== wick_trap PnL within full-system (full window) ===")
    for vn in VARIANTS:
        tl=OUT/vn/"full"/"trade_log.csv"
        if tl.exists():
            df=pd.read_csv(tl); wt=df[df.archetype=="wick_trap"]
            print(f"  {vn:<16} wick_trap: n={len(wt)} pnl={wt.pnl.sum():>9.0f}  | ALL: pnl={df.pnl.sum():>9.0f}")
    (OUT/"summary.json").write_text(json.dumps(cards,indent=2,default=str))
    print(f"report -> {OUT/'summary.json'}")
if __name__=="__main__": main()
