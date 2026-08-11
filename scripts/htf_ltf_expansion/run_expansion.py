"""
HTF-state + LTF-trigger EXPANSION VALIDATION  (study/htf-ltf-expansion)
=======================================================================
Pre-registered before measuring. No per-cell tuning. No grids beyond the
pre-registered P values and axis combos. Cells are NOT silently dropped.

CONTEXT (wyckoff_audit.md add.44-52)
------------------------------------
- The 1H bull-BOS trigger ALONE is a coin flip (~51% win; order_block_retest PF 0.75).
- add.51 FIXED the BOMS-direction dead-signal bug (engine/structure/boms_detector.py):
  displacement-only breaks now emit the break-side `direction` instead of 'none'.
- add.52 first fair probe: gating the 1H bull-BOS trigger by HTF BOMS-bull state
  lifted win 51%->77% (72h) and BEAT a plain 1D-EMA200 trend filter (54%) -- but
  only n=35 signals over 8y (state covers 1.1% of bars), clustered, "beyond trend"
  evidence was n=4. OPEN QUESTION: does the edge survive EXPANSION to a tradeable
  signal count, or was 77% the tightest-slice artifact?

DATA
----
data/features_mtf/BTC_1H_FEATURES_V22_CTX.parquet (2018-01-01..2026-06-10, 73,829 1H bars).
The store's tf1d_boms_direction / tf4h_boms_direction are DEAD (all-zero, verified) --
they were computed with the pre-fix detector. We RECOMPUTE BOMS direction from store
OHLCV resampled to 4H and 1D, running the FIXED detect_boms() on trailing windows
(causal), broadcast to 1H with closed-bar discipline (available_at = HTF bar close).

PRE-REGISTERED EXPANSION AXES (report the edge-vs-n curve ACROSS all; do not cherry-pick)
----------------------------------------------------------------------------------------
E1 state timeframe : 1D only (n=35 baseline) | 4H only | 1D-OR-4H
E2 state persistence: P in {0, 5, 10, 20} days after the last directional break
                      (P=0 = break HTF-bar only). A break sets a regime lasting P days.
E3 direction        : bull-state + long-trigger  AND  bear-state + short-trigger (mirror)
E4 trigger set      : T1 tf1h_bos_(bull/bear)            [store col, ~1.9% fire]
                      T2 T1 OR tf1h_choch_detected
                      T3 T2 OR OB-tap (tf1h_ob_high/low)  -> UNAVAILABLE: store col all-NaN
                                                            (flagged, cannot run as specified)
E5 quality axes (Part A specs, computed from store OHLCV where inputs exist):
   - reclaim_speed  : recent sweep + fast reclaim (Part A #2 proxy; flagged approximations)
   - eq_magnet_prox : eq_high_pool/eq_low_pool proximity (Part A #3; store has the pools)
   - trap_reset     : dir-flip + body>=1.25*ATR + opp-wick>30% (Part A #4; fully computable)
   - ob_quality     : swing touches + level_quality + reaction speed (Part A #1 proxy)
   Test: WITHIN the gated set, does quality-top-half beat quality-bottom-half win%?

METRICS per cell: n, independent episodes (signals >72h apart), win% @24/72/168h,
mean/median fwd return, cost-aware trade-sim PF (entry@close, stop 1.5*ATR, TP 2R,
168h time-exit, 5bps/side).
CONTROLS every cell is compared against: (a) unconditional same-horizon returns,
(b) plain 1D-EMA200 trend filter, (c) the SAME trigger ungated.

VERDICT RULE (pre-registered): VALIDATES for escalation iff some cell reaches
n>=150 AND >=40 independent episodes AND win%>=60% @72h AND beats both controls
AND sim PF>=1.5 AND quality-top-half>=quality-bottom-half. Else report which
criterion fails and where on the expansion curve the edge dies.

Costs: single asset (BTC); many folds already spent on longs on this store; fwd
returns are NOT tradeable edge (the cost-aware sim is the anchor). Nothing ships.
"""
import sys, os, json, warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd

ROOT = "/Users/rayghandchi/Bull Machine/Bull-machine-"
sys.path.insert(0, ROOT)
from engine.structure.boms_detector import detect_boms

STORE = f"{ROOT}/data/features_mtf/BTC_1H_FEATURES_V22_CTX.parquet"
SCRATCH = "/private/tmp/claude-501/-Users-rayghandchi-Bull-Machine-Bull-machine-/da0c7698-bb0a-4aa2-9a4e-481e62857ce4/scratchpad"
os.makedirs(SCRATCH, exist_ok=True)
CACHE = f"{SCRATCH}/boms_states.parquet"

FWD_H = [24, 72, 168]
COST_SIDE = 0.0005  # 2bps commission + 3bps slippage per side

# ---------------------------------------------------------------------------
def resample_ohlcv(df1h, rule):
    o = df1h[['open','high','low','close','volume']].resample(rule, label='left', closed='left').agg(
        {'open':'first','high':'max','low':'min','close':'last','volume':'sum'}).dropna()
    return o

def boms_dir_series(ohlc, timeframe, win=90):
    """Per-bar BOMS direction via trailing-window detect_boms (causal).
    Returns int series: +1 bull, -1 bear, 0 none, indexed at HTF bar start."""
    dirs = np.zeros(len(ohlc), dtype=int)
    vals = ohlc.reset_index(drop=True)
    for i in range(len(vals)):
        lo = max(0, i - win + 1)
        w = vals.iloc[lo:i+1].reset_index(drop=True)
        sig = detect_boms(w, timeframe=timeframe)
        if sig.direction == 'bullish':
            dirs[i] = 1
        elif sig.direction == 'bearish':
            dirs[i] = -1
    return pd.Series(dirs, index=ohlc.index)

def persist_state(raw_dir, tf_delta, P_days):
    """A directional break at time tb sets state=dir for [tb, tb+P_days].
    Later opposite break overrides. P_days=0 -> the break bar only.
    Returns series of {+1,-1,0} at the HTF index."""
    idx = raw_dir.index
    out = np.zeros(len(idx), dtype=int)
    last_dir = 0
    last_tb = None
    Pd = pd.Timedelta(days=P_days)
    for i, t in enumerate(idx):
        d = raw_dir.iloc[i]
        if d != 0:
            last_dir = d; last_tb = t
            out[i] = d
        else:
            if last_tb is not None and (t - last_tb) <= Pd:
                out[i] = last_dir
            else:
                out[i] = 0
    return pd.Series(out, index=idx)

def broadcast(state_htf, tf_delta, target_index):
    """Closed-bar discipline: HTF bar labeled at t0 closes at t0+tf_delta; only
    available to 1H bars at/after that close. merge_asof backward on available_at."""
    av = pd.DataFrame({'available_at': state_htf.index + tf_delta,
                       'state': state_htf.values}).sort_values('available_at')
    tgt = pd.DataFrame({'ts': target_index}).sort_values('ts')
    m = pd.merge_asof(tgt, av, left_on='ts', right_on='available_at', direction='backward')
    return pd.Series(m['state'].fillna(0).astype(int).values, index=m['ts'].values).reindex(target_index).fillna(0).astype(int)

# ---------------------------------------------------------------------------
def build_states(df1h):
    if os.path.exists(CACHE):
        print("[cache] loading recomputed BOMS states")
        return pd.read_parquet(CACHE)
    print("[compute] resampling + causal BOMS recompute (fixed detector)")
    o4 = resample_ohlcv(df1h, '4h')
    o1d = resample_ohlcv(df1h, '1D')
    print(f"  4H bars={len(o4)}  1D bars={len(o1d)}")
    raw4 = boms_dir_series(o4, '4H')
    raw1d = boms_dir_series(o1d, '1D')
    print(f"  raw 4H dir nonzero={np.mean(raw4!=0):.4f}  1D dir nonzero={np.mean(raw1d!=0):.4f}")

    # causality check (3-point truncation agreement on 1D)
    causal_ok = causality_check(o1d, '1D', raw1d)

    out = pd.DataFrame(index=df1h.index)
    for P in [0, 5, 10, 20]:
        s4 = persist_state(raw4, pd.Timedelta(hours=4), P)
        s1d = persist_state(raw1d, pd.Timedelta(days=1), P)
        b4 = broadcast(s4, pd.Timedelta(hours=4), df1h.index)
        b1d = broadcast(s1d, pd.Timedelta(days=1), df1h.index)
        out[f'st4h_P{P}'] = b4
        out[f'st1d_P{P}'] = b1d
        # OR combine: bull if any HTF bull and no HTF bear; bear if any bear and no bull
        both = pd.DataFrame({'a': b4, 'b': b1d})
        bull = ((both['a']==1)|(both['b']==1)) & ~((both['a']==-1)|(both['b']==-1))
        bear = ((both['a']==-1)|(both['b']==-1)) & ~((both['a']==1)|(both['b']==1))
        oror = np.where(bull, 1, np.where(bear, -1, 0))
        out[f'stOR_P{P}'] = oror.astype(int)
    out.attrs = {}
    out.to_parquet(CACHE)
    with open(f"{SCRATCH}/causality.json","w") as f:
        json.dump(causal_ok, f, indent=2)
    return out

def causality_check(ohlc, timeframe, full_dir):
    """Recompute the direction series at 3 truncation points; verify that for
    bars present in an earlier truncation, the value never changes (no repaint)."""
    n = len(ohlc)
    cuts = [int(n*0.6), int(n*0.8), n]
    series = {}
    for c in cuts:
        series[c] = boms_dir_series(ohlc.iloc[:c], timeframe)
    res = {'timeframe': timeframe, 'checks': []}
    ok_all = True
    for a, b in [(cuts[0],cuts[1]),(cuts[1],cuts[2]),(cuts[0],cuts[2])]:
        sa, sb = series[a], series[b]
        overlap = sa.index
        eq = bool((sa.reindex(overlap).values == sb.reindex(overlap).values).all())
        ok_all = ok_all and eq
        res['checks'].append({'trunc_a':int(a),'trunc_b':int(b),'bars_compared':int(len(overlap)),'identical':eq})
    res['no_repaint'] = ok_all
    print(f"  [causality {timeframe}] no_repaint={ok_all}")
    return res

# ---------------------------------------------------------------------------
def fwd_returns(close):
    return {h: close.shift(-h)/close - 1.0 for h in FWD_H}

def trade_sim(df1h, sig_mask, side):
    """Entry@close of signal bar; stop 1.5*ATR; TP 2R; 168h time-exit; 5bps/side.
    side=+1 long, -1 short. Returns dict metrics + per-trade net-R list."""
    close = df1h['close'].values; high = df1h['high'].values; low = df1h['low'].values
    atr = df1h['atr_14'].values
    idx = np.where(sig_mask.values)[0]
    N = len(df1h); H = 168
    rs = []
    for i in idx:
        if i + 1 >= N or not np.isfinite(atr[i]) or atr[i] <= 0:
            continue
        entry = close[i]; R = 1.5*atr[i]
        if side == 1:
            stop = entry - R; tp = entry + 2*R
        else:
            stop = entry + R; tp = entry - 2*R
        exit_px = None
        end = min(i+H, N-1)
        for j in range(i+1, end+1):
            if side == 1:
                hit_stop = low[j] <= stop; hit_tp = high[j] >= tp
            else:
                hit_stop = high[j] >= stop; hit_tp = low[j] <= tp
            if hit_stop and hit_tp:   # same-bar ambiguity -> stop first (conservative)
                exit_px = stop; break
            if hit_stop: exit_px = stop; break
            if hit_tp: exit_px = tp; break
        if exit_px is None:
            exit_px = close[end]
        gross = side*(exit_px-entry)/entry
        net = gross - 2*COST_SIDE
        rs.append(net/(R/entry))  # net in R units
    rs = np.array(rs)
    if len(rs)==0:
        return {'n':0,'pf':np.nan,'winrate':np.nan,'mean_R':np.nan,'sum_R':np.nan}
    wins = rs[rs>0].sum(); losses = -rs[rs<0].sum()
    pf = wins/losses if losses>0 else np.inf
    return {'n':int(len(rs)),'pf':float(pf),'winrate':float((rs>0).mean()),
            'mean_R':float(rs.mean()),'sum_R':float(rs.sum()),'rs':rs}

def episodes(ts_index, sig_mask, gap_h=72):
    ts = ts_index[sig_mask.values]
    if len(ts)==0: return 0
    ts = pd.Series(ts).sort_values().values
    ep = 1
    for k in range(1,len(ts)):
        if (ts[k]-ts[k-1])/np.timedelta64(1,'h') > gap_h:
            ep += 1
    return ep

def cell_metrics(df1h, fwd, sig_mask, side):
    n = int(sig_mask.sum())
    row = {'n':n, 'episodes':episodes(df1h.index, sig_mask)}
    for h in FWD_H:
        r = fwd[h][sig_mask.values]
        r = r.dropna()*side
        row[f'win{h}'] = float((r>0).mean()) if len(r) else np.nan
        row[f'mean{h}'] = float(r.mean()) if len(r) else np.nan
        row[f'med{h}'] = float(r.median()) if len(r) else np.nan
    sim = trade_sim(df1h, sig_mask, side)
    row['sim_n']=sim['n']; row['sim_pf']=sim['pf']; row['sim_wr']=sim['winrate']; row['sim_meanR']=sim['mean_R']
    return row, sim

# ---------------------------------------------------------------------------
def main():
    df = pd.read_parquet(STORE)
    df = df[['open','high','low','close','volume','atr_14','ema_200',
             'tf1h_bos_bullish','tf1h_bos_bearish','tf1h_choch_detected',
             'eq_high_pool','eq_low_pool','sweep_low_event','sweep_high_event',
             'swing_low_touches','level_quality_low','fib_in_ote_zone']].copy()
    states = build_states(df)
    df = df.join(states)
    fwd = fwd_returns(df['close'])

    # triggers
    trig = {
        ('long','T1'): df['tf1h_bos_bullish']==1,
        ('long','T2'): (df['tf1h_bos_bullish']==1)|(df['tf1h_choch_detected']==1),
        ('short','T1'): df['tf1h_bos_bearish']==1,
        ('short','T2'): (df['tf1h_bos_bearish']==1)|(df['tf1h_choch_detected']==1),
    }
    side_of = {'long':1,'short':-1}
    dir_of = {'long':1,'short':-1}

    # controls
    ema_ok = {'long': df['close']>df['ema_200'], 'short': df['close']<df['ema_200']}

    rows = []
    sims = {}
    for e1, stcol in [('1D','st1d'),('4H','st4h'),('1D_OR_4H','stOR')]:
        for P in [0,5,10,20]:
            st = df[f'{stcol}_P{P}']
            for dside in ['long','short']:
                want = dir_of[dside]
                state_ok = st==want
                for tset in ['T1','T2']:
                    tmask = trig[(dside,tset)]
                    gated = state_ok & tmask
                    row, sim = cell_metrics(df, fwd, gated, side_of[dside])
                    key = f"{e1}|P{P}|{dside}|{tset}"
                    row.update({'cell':key,'E1':e1,'P':P,'dir':dside,'trig':tset})
                    # control (c) ungated trigger
                    ug, _ = cell_metrics(df, fwd, tmask, side_of[dside])
                    row['ctrl_ungated_win72']=ug['win72']; row['ctrl_ungated_pf']=ug['sim_pf']; row['ctrl_ungated_n']=ug['n']
                    # control (b) trend filter (ema200) + same trigger
                    tf_mask = tmask & ema_ok[dside]
                    tfm, _ = cell_metrics(df, fwd, tf_mask, side_of[dside])
                    row['ctrl_ema200_win72']=tfm['win72']; row['ctrl_ema200_pf']=tfm['sim_pf']; row['ctrl_ema200_n']=tfm['n']
                    rows.append(row)
                    sims[key]=sim
    res = pd.DataFrame(rows)

    # control (a) unconditional same-horizon
    uncond = {}
    for dside in ['long','short']:
        s = side_of[dside]
        for h in FWD_H:
            r=(fwd[h].dropna()*s); uncond[f'{dside}_win{h}']=float((r>0).mean()); uncond[f'{dside}_mean{h}']=float(r.mean())
    with open(f"{SCRATCH}/uncond_control.json","w") as f: json.dump(uncond,f,indent=2)

    res.to_parquet(f"{SCRATCH}/expansion_grid.parquet")
    res.to_csv(f"{SCRATCH}/expansion_grid.csv", index=False)
    # print compact
    pd.set_option('display.width',240); pd.set_option('display.max_columns',40); pd.set_option('display.max_rows',200)
    show=['cell','n','episodes','win24','win72','win168','mean72','sim_n','sim_pf','sim_wr',
          'ctrl_ungated_win72','ctrl_ungated_pf','ctrl_ema200_win72','ctrl_ema200_pf']
    print("\n=== UNCONDITIONAL CONTROL (a) ===")
    print(json.dumps(uncond,indent=0))
    print("\n=== EXPANSION GRID (E1 x E2 x E3 x E4) ===")
    print(res[show].round(3).to_string(index=False))

    # save sim R-arrays for the best cell CPCV step (separate script)
    import pickle
    with open(f"{SCRATCH}/sims.pkl","wb") as f:
        pickle.dump({k:(v.get('rs') if isinstance(v,dict) else None) for k,v in sims.items()}, f)
    print("\n[done] grid + sims written to scratchpad")

if __name__ == "__main__":
    main()
