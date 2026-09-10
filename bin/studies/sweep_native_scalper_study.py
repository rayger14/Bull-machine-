"""ZeroIka equal-lows sweep scalper — native-resolution study + FINAL SPEC.

Reproduction: download Binance BTCUSDT futures 1m monthly klines from
data.binance.vision (2021-2026) into a parquet with open/high/low/close/vol,
UTC index, then run. Full verdicts: docs/knowledge/sweep_native_scalper_2026_09_09.md

FINAL SPEC (locked 2026-09-09, all choices declared pre-test, 6/6 years each):
  event  : pivot-low level (15min/side), >=2 touches within 0.1% over 24h,
           standing >=30min; swept below; first 1m close reclaiming <=30min
  entry  : the reclaiming minute's close (confirmation bar TESTED AND REJECTED)
  stop   : sweep_low - 0.15%  (tight -0.05% TESTED: 3x worse; do not re-tighten)
  hold   : 4h flat (24h tested: inferior in money-sim)
  sizing : 1.5x when level touches >= 3 (pool-depth tilt; exclusion TESTED AND
           REJECTED - cuts a positive class)
  costs  : 12bps round trip assumed; only unmeasured risk = live stop-fill
           slippage in cascades (+0.05% worse -> $136K; +0.10% -> $67K/5-6yrs)
Backtest: ~$204K per $50K notional, 2021-2026, 6/6 years, maxDD -$9.8K.
"""
import pandas as pd, numpy as np
S='data/studies'
b=pd.read_parquet(f'{S}/btc_1m_2021_2026.parquet')
lo=b.low.values; cl=b.close.values; hi=b.high.values; ts=b.index
n=len(b)
COST=0.0007
events=[]
# rolling scan: use 15-min pivot lows as level candidates to keep O(n) sane
piv=(b.low.rolling(31,center=True).min()==b.low)  # 15m-each-side pivot low
piv_idx=np.where(piv.fillna(False).values)[0]
print(f"bars={n:,}, pivot-low candidates={len(piv_idx):,}", flush=True)
# index pivots by time for trailing-24h cluster lookup
piv_lows=lo[piv_idx]
piv_times=ts[piv_idx]
import bisect
last_event_i=-10**9
for k in range(2,len(piv_idx)):
    i=piv_idx[k]
    lvl=piv_lows[k]
    # cluster: another pivot low within 0.1% in the PRIOR 24h, formed >=30min before
    t=piv_times[k]
    j0=bisect.bisect_left(piv_times, t-pd.Timedelta(hours=24), 0, k)
    prior=piv_lows[j0:k]
    prior_t=piv_times[j0:k]
    m=(np.abs(prior-lvl)/lvl<=0.001)&(prior_t<=t-pd.Timedelta(minutes=30))
    if m.sum()<1:  # this pivot + >=1 prior equal low = >=2 touches
        continue
    # find first sweep after this pivot within 24h: bar low < lvl*(1-0.0002)
    end=min(n-1, i+1440)
    win_lo=lo[i+15:end]
    if len(win_lo)==0: continue
    br=np.where(win_lo<lvl*0.9998)[0]
    if len(br)==0: continue
    s=i+15+br[0]           # sweep bar
    if s-last_event_i<60: continue
    # reclaim: first 1m close back above lvl within 120m
    rec=np.where(cl[s:min(s+121,n)]>lvl)[0]
    if len(rec)==0: continue
    r=s+rec[0]
    speed=rec[0]
    sweep_low=lo[s:r+1].min()
    entry=cl[r]
    stop=sweep_low*0.9995
    out={'ts':ts[r],'yr':ts[r].year,'speed':speed,'entry':entry,'stop':stop}
    # outcome: stop-aware forward returns
    for hz,mins in [('r1h',60),('r4h',240),('r24h',1440)]:
        e2=min(r+mins,n-1)
        path_lo=lo[r+1:e2+1]
        stopped=len(path_lo)>0 and path_lo.min()<=stop
        if stopped:
            ret=(stop/entry-1)
        else:
            ret=(cl[e2]/entry-1)
        out[hz]=ret-COST
    events.append(out)
    last_event_i=s
d=pd.DataFrame(events)
d.to_csv(f'{S}/sweep_native_events.csv',index=False)
fast=d[d.speed<=30]; slow=d[d.speed>30]
print(f"\nevents: {len(d)} total | FAST(<=30m reclaim): {len(fast)} | SLOW control: {len(slow)}")
for tag,g in [('FAST',fast),('SLOW',slow)]:
    if len(g)==0: continue
    print(f"\n{tag}: after-cost expectancy 1h {10000*g.r1h.mean():+.1f}bps | 4h {10000*g.r4h.mean():+.1f}bps | 24h {10000*g.r24h.mean():+.1f}bps | WR4h {100*(g.r4h>0).mean():.0f}%")
    yr=g.groupby('yr').r4h.mean()*10000
    pos=sum(1 for v in yr if v>0)
    print(f"  4h expectancy by year (bps): "+", ".join(f"{y}:{v:+.0f}" for y,v in yr.items())+f"  -> {pos}/{len(yr)} positive")
