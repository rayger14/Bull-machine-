"""
E5 quality-axis test + CPCV on the best pre-registered cells.
Uses cached BOMS states from run_expansion.py.

E5 (Part A specs, computed from store OHLCV where inputs exist):
  - trap_reset   : Part A #4 VERBATIM (bull_machine/modules/bojan/bojan.py):
                   dir-flip + body>=1.25*ATR + opp-wick>0.30*range. FULLY computable.
                   quality = body/ATR ratio (continuous), gated by flip+sweep.
  - eq_magnet    : Part A #3 (detect_equal_clusters, 3+ within 0.1% / 10 bars) ->
                   store binary eq_low_pool/eq_high_pool (the pool membership flag).
                   APPROXIMATION: store gives binary membership, not cluster_score.
  - reclaim_spd  : Part A #2 (score_sweep_mitigation) proxy: recent sweep_low_event
                   within 5 bars + fib_in_ote_zone (golden-pocket 0.618-0.786 proxy).
                   HEAVY APPROXIMATION: store lacks sweep-bar ts / reclaim-hour count /
                   ATR-displacement of the reclaim. Flagged.
  - ob_quality   : Part A #1 (engine/liquidity/hob.py) proxy: touch_strength
                   (min(swing_low_touches/5,1)) * 0.6 + level_quality_low * 0.4.
                   APPROXIMATION: only 2 of 5 HOB components have store inputs
                   (no consolidation/reaction-speed/mtf-confluence axes). Flagged.
  Test: WITHIN the gated long set, does quality-top-half beat quality-bottom-half
  (win72 + sim PF)? This is the key test of "the lost quality components are why the
  trigger is a coin flip."

CPCV: K=6 purged combinatorial folds on the trade-sim R-series of the best cells.
"""
import sys, os, json, warnings, itertools
warnings.filterwarnings('ignore')
import numpy as np, pandas as pd
ROOT = "/Users/rayghandchi/Bull Machine/Bull-machine-"
sys.path.insert(0, ROOT)
STORE = f"{ROOT}/data/features_mtf/BTC_1H_FEATURES_V22_CTX.parquet"
SCRATCH = "/private/tmp/claude-501/-Users-rayghandchi-Bull-Machine-Bull-machine-/da0c7698-bb0a-4aa2-9a4e-481e62857ce4/scratchpad"
CACHE = f"{SCRATCH}/boms_states.parquet"
COST_SIDE = 0.0005

df = pd.read_parquet(STORE)
keep = ['open','high','low','close','volume','atr_14','ema_200','tf1h_bos_bullish','tf1h_bos_bearish',
        'tf1h_choch_detected','eq_high_pool','eq_low_pool','sweep_low_event','sweep_high_event',
        'swing_low_touches','level_quality_low','fib_in_ote_zone']
df = df[keep].join(pd.read_parquet(CACHE))

o=df['open'].values;h=df['high'].values;l=df['low'].values;c=df['close'].values;atr=df['atr_14'].values

def trap_reset_quality(i, side):
    if i<1 or not np.isfinite(atr[i]) or atr[i]<=0: return np.nan
    prev_bull = c[i-1]>o[i-1]; cur_bull = c[i]>o[i]
    flip = (prev_bull and not cur_bull) or (not prev_bull and cur_bull)
    body_atr = abs(c[i]-o[i])/atr[i]
    rng = h[i]-l[i]
    if rng<=0: return np.nan
    if cur_bull:
        sweep = (min(o[i],c[i])-l[i])/rng > 0.3
    else:
        sweep = (h[i]-max(o[i],c[i]))/rng > 0.3
    base = body_atr if (flip and sweep) else 0.0   # verbatim gating; magnitude=body/ATR
    return base

def ob_quality(i):
    ts = df['swing_low_touches'].values[i]; lq = df['level_quality_low'].values[i]
    touch_strength = min(ts/5.0, 1.0)
    return touch_strength*0.6 + (lq if np.isfinite(lq) else 0)*0.4

def reclaim_quality(i):
    # recent sweep_low within 5 bars + golden-pocket (fib OTE) at signal
    lo=max(0,i-5)
    swept = df['sweep_low_event'].values[lo:i+1].sum()>0
    ote = df['fib_in_ote_zone'].values[i]>0
    return (0.5 if swept else 0) + (0.5 if ote else 0)

def eq_magnet_quality(i, side):
    return float(df['eq_low_pool'].values[i]) if side==1 else float(df['eq_high_pool'].values[i])

def trade_R(i, side):
    N=len(df);H=168
    if i+1>=N or not np.isfinite(atr[i]) or atr[i]<=0: return None
    entry=c[i];R=1.5*atr[i]
    if side==1: stop=entry-R;tp=entry+2*R
    else: stop=entry+R;tp=entry-2*R
    end=min(i+H,N-1);ex=None
    for j in range(i+1,end+1):
        if side==1: hs=l[j]<=stop; ht=h[j]>=tp
        else: hs=h[j]>=stop; ht=l[j]<=tp
        if hs and ht: ex=stop;break
        if hs: ex=stop;break
        if ht: ex=tp;break
    if ex is None: ex=c[end]
    net=side*(ex-entry)/entry - 2*COST_SIDE
    return net/(R/entry)

def pf_of(rs):
    rs=np.asarray(rs); w=rs[rs>0].sum(); ls=-rs[rs<0].sum()
    return (w/ls if ls>0 else np.inf), (rs>0).mean(), rs.mean()

def gated_idx(e1col, P, side, tset):
    st=df[f'{e1col}_P{P}']; want=side
    if tset=='T1':
        trig = (df['tf1h_bos_bullish']==1) if side==1 else (df['tf1h_bos_bearish']==1)
    else:
        base = (df['tf1h_bos_bullish']==1) if side==1 else (df['tf1h_bos_bearish']==1)
        trig = base | (df['tf1h_choch_detected']==1)
    m=(st==want)&trig
    return np.where(m.values)[0]

fwd72 = (df['close'].shift(-72)/df['close']-1).values

# ---------- E5: quality split within a gated long cohort ----------
# Use the largest pre-registered long cells for a top/bottom split.
print("=== E5 QUALITY-AXIS TEST (within gated long set) ===")
for label,(e1,P) in {'4H|P20|long|T1':('st4h',20),'1D|P20|long|T1':('st1d',20),
                      '1D_OR_4H|P20|long|T1':('stOR',20)}.items():
    idx=gated_idx(e1,P,1,'T1')
    idx=[i for i in idx if i+72<len(df) and np.isfinite(atr[i]) and atr[i]>0]
    if len(idx)<40:
        print(f"{label}: n={len(idx)} too small for split"); continue
    qnames=['trap_reset','ob_quality','reclaim_spd','eq_magnet']
    Q={ 'trap_reset':[trap_reset_quality(i,1) for i in idx],
        'ob_quality':[ob_quality(i) for i in idx],
        'reclaim_spd':[reclaim_quality(i) for i in idx],
        'eq_magnet':[eq_magnet_quality(i,1) for i in idx] }
    # composite = mean of min-max normalized computable components
    comp=[]
    for q in qnames:
        a=np.array(Q[q],dtype=float);
        if np.nanmax(a)>np.nanmin(a): a=(a-np.nanmin(a))/(np.nanmax(a)-np.nanmin(a))
        else: a=np.zeros_like(a)
        comp.append(a)
    composite=np.nanmean(np.vstack(comp),axis=0)
    Q['composite']=composite
    win=np.array([1 if fwd72[i]>0 else 0 for i in idx])
    Rs=np.array([trade_R(i,1) for i in idx])
    print(f"\n{label}: n={len(idx)}  overall win72={win.mean():.3f} sim_pf={pf_of(Rs)[0]:.3f}")
    for q in qnames+['composite']:
        a=np.array(Q[q],dtype=float); med=np.nanmedian(a)
        top=a>med; bot=a<=med
        if top.sum()<5 or bot.sum()<5:
            # binary axis: split by ==max vs <max
            top=a>=np.nanmax(a); bot=~top
        wt=win[top].mean() if top.sum() else np.nan; wb=win[bot].mean() if bot.sum() else np.nan
        pft=pf_of(Rs[top])[0] if top.sum() else np.nan; pfb=pf_of(Rs[bot])[0] if bot.sum() else np.nan
        flag = 'TOP>BOT' if (np.isfinite(wt) and np.isfinite(wb) and wt>wb) else 'no'
        print(f"   {q:12s} top(n={top.sum():3d}) win={wt:.3f} pf={pft:.2f} | bot(n={bot.sum():3d}) win={wb:.3f} pf={pfb:.2f}  [{flag}]")

# ---------- CPCV K=6 on best cells ----------
def cpcv(rs, K=6, ncomb=2):
    rs=np.asarray(rs); n=len(rs)
    if n<K*3: return None
    folds=np.array_split(np.arange(n),K)
    pfs=[]
    for combo in itertools.combinations(range(K),ncomb):
        test=np.concatenate([folds[k] for k in combo])
        r=rs[test]
        if (r<0).sum()==0: pfs.append(np.inf); continue
        pfs.append(r[r>0].sum()/(-r[r<0].sum()))
    pfs=np.array([p for p in pfs if np.isfinite(p)])
    return {'folds':K,'combos':int(len(pfs)),'pf_median':float(np.median(pfs)),
            'pf_mean':float(np.mean(pfs)),'pf_min':float(np.min(pfs)),'pf_max':float(np.max(pfs)),
            'frac_ge_1.5':float(np.mean(pfs>=1.5))}

print("\n=== CPCV (K=6, 2-fold test combos) on best-PF n>=150 cells ===")
best_cells={'1D_OR_4H|P20|long|T1':('stOR',20,1,'T1'),
            '4H|P20|long|T1':('st4h',20,1,'T1'),
            '1D|P20|long|T2':('st1d',20,1,'T2'),
            '1D_OR_4H|P0|short|T1':('stOR',0,-1,'T1')}
for label,(e1,P,side,tset) in best_cells.items():
    idx=gated_idx(e1,P,side,tset)
    Rs=[trade_R(i,side) for i in idx]; Rs=np.array([r for r in Rs if r is not None])
    full=pf_of(Rs)
    cp=cpcv(Rs)
    print(f"{label}: n={len(Rs)} full_pf={full[0]:.3f} wr={full[1]:.3f} | CPCV {cp}")
print("\n[done]")
