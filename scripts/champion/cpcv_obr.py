#!/usr/bin/env python3
"""CPCV evaluation of standalone order_block_retest (V15, post index-fix) (no parameters fitted).

Splits 2018-01-01..2026-06-10 into k=6 contiguous time groups; evaluates all
C(6,2)=15 two-group test combinations. Since the strategy is FIXED (no
optimization), each group is backtested once and combinations aggregate
group-level results. A real edge: most combinations positive, no catastrophe.

Acceptance (pre-registered): >=10/15 combinations positive PnL; median
combination PF >= 1.2; worst combination loss bounded (> -$8K on $100K).
"""
from __future__ import annotations
import itertools, json, math, sys
from pathlib import Path
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO/"scripts/champion"))

STORE = REPO/"data/features_mtf/BTC_1H_FEATURES_V15_STRUCTURE.parquet"
CFG = REPO/"configs/champion/champion_order_block_retest.json"
OUT = REPO/"results/champion_v15_cpcv_obr"

def run_window(start, end, out_dir):
    from bin.backtest_v11_standalone import StandaloneBacktestEngine
    out_dir.mkdir(parents=True, exist_ok=True)
    sf = out_dir/"performance_stats.json"
    if sf.exists(): return json.loads(sf.read_text())
    eng = StandaloneBacktestEngine(config=json.loads(CFG.read_text()), initial_cash=100000.0,
        commission_rate=0.0002, slippage_bps=3.0, feature_store_path=str(STORE))
    eng.run(start_date=start, end_date=end)
    stats = eng.get_performance_stats()
    clean = {k:(str(v) if isinstance(v,float) and (math.isnan(v) or math.isinf(v)) else v) for k,v in stats.items()}
    sf.write_text(json.dumps(clean, indent=2, default=str)); return clean

def num(v, d=0.0):
    try:
        f=float(v); return d if f!=f else f
    except (TypeError,ValueError): return d

def main():
    OUT.mkdir(parents=True, exist_ok=True)
    edges = pd.date_range("2018-01-01","2026-06-10",periods=7)
    groups=[]
    for i in range(6):
        s=edges[i].strftime("%Y-%m-%d"); e=(edges[i+1]-pd.Timedelta(days=1)).strftime("%Y-%m-%d")
        st=run_window(s,e,OUT/f"g{i+1}")
        groups.append({"g":i+1,"span":f"{s}..{e}","pnl":num(st.get("total_pnl")),
            "gp":num(st.get("gross_profit")),"gl":num(st.get("gross_loss")),
            "n":int(num(st.get("total_trades")))})
        # derive gross P/L if absent: reconstruct from PF+PnL
        if groups[-1]["gp"]==0 and num(st.get("profit_factor"))>0 and groups[-1]["pnl"]!=0:
            pf=num(st.get("profit_factor")); pnl=groups[-1]["pnl"]
            gl=pnl/(pf-1) if pf!=1 else 0.0
            groups[-1]["gl"]=abs(gl); groups[-1]["gp"]=abs(gl)*pf
        print(f"  g{i+1} {groups[-1]['span']}: n={groups[-1]['n']} PnL=${groups[-1]['pnl']:.0f}")
    combos=[]
    for a,b in itertools.combinations(range(6),2):
        pnl=groups[a]["pnl"]+groups[b]["pnl"]; n=groups[a]["n"]+groups[b]["n"]
        gp=groups[a]["gp"]+groups[b]["gp"]; gl=groups[a]["gl"]+groups[b]["gl"]
        combos.append({"combo":f"g{a+1}+g{b+1}","pnl":round(pnl),"n":n,
                       "pf":round(gp/gl,2) if gl>0 else None})
    combos.sort(key=lambda c:c["pnl"])
    pos=sum(1 for c in combos if c["pnl"]>0)
    pfs=[c["pf"] for c in combos if c["pf"]]
    med_pf=sorted(pfs)[len(pfs)//2] if pfs else None
    print(f"\n{'combo':<10}{'pnl':>9}{'pf':>7}{'n':>6}")
    for c in combos: print(f"{c['combo']:<10}{c['pnl']:>9}{str(c['pf']):>7}{c['n']:>6}")
    worst=combos[0]["pnl"]
    verdict=(pos>=10 and (med_pf or 0)>=1.2 and worst>-8000)
    print(f"\npositive combos: {pos}/15 | median PF: {med_pf} | worst: ${worst}")
    print(f"CPCV VERDICT: {'PASS' if verdict else 'FAIL'} (need >=10/15 pos, med PF>=1.2, worst>-$8K)")
    (OUT/"cpcv_summary.json").write_text(json.dumps({"groups":groups,"combos":combos,
        "positive":pos,"median_pf":med_pf,"worst":worst,"pass":verdict},indent=2))

if __name__=="__main__": main()
