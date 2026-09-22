"""wick_trap population entry-skill test (PRE-REGISTERED 2026-09-22).
Q: are wick_trap entries better than random longs in the same regime, under
identical exits? Excess = real - mean(20 same-month random entries), common
ladder engine (0.5R/1R/2R @ 10/30/50 + 2.5ATR trail after 1R, 120h cap,
R = 4.94*ATR14, $52.5K notional, 3bps slip each way + 2bps comm each way).
PASS: mean excess>0 AND same sign >=5/7 years AND bootstrap 90% CI excludes 0.
Event tags: FOMC/CPI/NFP (engine calendar, 2022+) within +/-2h of entry.
"""
import pandas as pd, numpy as np, sys
sys.path.insert(0,'/Users/rayghandchi/Bull Machine/Bull-machine-')
S='/private/tmp/claude-501/-Users-rayghandchi-Bull-Machine-Bull-machine-/833eefef-c5b5-45bb-af8d-fd9afb9e129c/scratchpad'
v=pd.read_parquet('/Users/rayghandchi/Bull Machine/Bull-machine-/data/features_mtf/BTC_1H_FEATURES_V23_PARITY_2018_2026.parquet',
                  columns=['open','high','low','close'])
c,h,l=v.close.values,v.high.values,v.low.values
tr=np.maximum(h-l,np.maximum(np.abs(h-np.roll(c,1)),np.abs(l-np.roll(c,1))))
atr=pd.Series(tr).rolling(14).mean().values
ts=v.index; n=len(v); NOTIONAL=52500.0; COST=0.0005+0.0005  # slip+comm each side lumped per leg

def sim(i):
    """Common ladder engine. Returns net $ PnL at $52.5K notional."""
    if i+2>=n or not np.isfinite(atr[i]) or atr[i]<=0: return None
    entry=c[i]*1.0003
    R=4.94*atr[i]; stop=c[i]-R
    qty=NOTIONAL/entry
    rungs=[(entry+0.5*R,0.10),(entry+1.0*R,0.30),(entry+2.0*R,0.50)]
    filled=[False]*3; rem=1.0; pnl=0.0; trail=None; hit1R=False; hwm=entry
    for j in range(i+1,min(i+121,n)):
        # stop first (conservative)
        eff_stop=max(stop,trail) if trail is not None else stop
        if l[j]<=eff_stop:
            fill=eff_stop*0.9997
            pnl+=rem*qty*(fill-entry)-rem*qty*fill*COST
            return pnl
        for k,(trg,pct) in enumerate(rungs):
            if not filled[k] and c[j]>=trg:
                fill=c[j]*0.9997
                pnl+=pct*qty*(fill-entry)-pct*qty*fill*COST
                rem-=pct; filled[k]=True
                if k>=1: hit1R=True
        if hit1R:
            hwm=max(hwm,c[j]); trail=hwm-2.5*atr[j] if np.isfinite(atr[j]) else trail
        if rem<1e-9: return pnl
    fill=c[min(i+120,n-1)]*0.9997
    pnl+=rem*qty*(fill-entry)-rem*qty*fill*COST
    return pnl

# real entries from the silo log
t=pd.read_csv(f'{S}/wick_base/trade_log.csv')
g=t.groupby('position_id').agg(ts_=('timestamp','first'))
g['ts_']=pd.to_datetime(g.ts_,utc=True)
idx={tt:i for i,tt in enumerate(ts)}
entries=[idx[x] for x in g.ts_ if x in idx]
print(f"real wick_trap entries: {len(entries)}")
# event calendar
from engine.regime_detector import FOMC_DATES, CPI_DATES, NFP_DATES
ev=set(pd.Timestamp(d,tz='UTC').date() for d in FOMC_DATES+CPI_DATES+NFP_DATES)
rng=np.random.default_rng(42)
rows=[]
month_pool={}
for i in entries:
    real=sim(i)
    if real is None: continue
    key=(ts[i].year,ts[i].month)
    if key not in month_pool:
        m=np.where((ts.year==key[0])&(ts.month==key[1]))[0]
        month_pool[key]=m[(m>200)&(m<n-130)]
    pool=month_pool[key]
    pool=pool[np.abs(pool-i)>24]
    picks=rng.choice(pool,20,replace=False) if len(pool)>=20 else pool
    base=[sim(int(p)) for p in picks]
    base=[b for b in base if b is not None]
    if not base: continue
    jit=[]
    for d in (-24,-12,-6,6,12,24):
        r2=sim(i+d)
        if r2 is not None: jit.append(r2)
    rows.append({'ts':ts[i],'yr':ts[i].year,'real':real,'rand':np.mean(base),
                 'excess':real-np.mean(base),'jit':np.mean(jit) if jit else np.nan,
                 'event':ts[i].date() in ev})
d=pd.DataFrame(rows)
d.to_csv(f'{S}/entry_skill_results.csv',index=False)
print(f"\nn={len(d)} | mean real ${d.real.mean():,.0f} | mean random ${d['rand'].mean():,.0f}")
print(f"MEAN EXCESS: ${d.excess.mean():,.0f}/trade | median ${d.excess.median():,.0f}")
yr=d.groupby('yr').excess.mean()
pos=sum(1 for x in yr if x>0)
print("by year:", ", ".join(f"{y}:{x:+,.0f}" for y,x in yr.items()), f"-> {pos}/{len(yr)} positive")
boot=[rng.choice(d.excess.values,len(d),replace=True).mean() for _ in range(2000)]
lo5,hi95=np.percentile(boot,[5,95])
print(f"bootstrap 90% CI on mean excess: [${lo5:,.0f}, ${hi95:,.0f}]")
print(f"jitter (timing): real ${d.real.mean():,.0f} vs +/-6-24h ${d.jit.mean():,.0f} -> timing worth ${d.real.mean()-d.jit.mean():,.0f}")
print(f"event-bar entries (2022+ calendar): {d.event.sum()} of {len(d)} | excess on-event ${d[d.event].excess.mean():,.0f} vs off ${d[~d.event].excess.mean():,.0f}")
print("\nBAR: mean>0 AND >=5/7 years AND CI excludes 0")
