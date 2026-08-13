"""
H2 SKEPTICAL DIAGNOSTICS (add.61) — before any adopt verdict, stress the ob_quality pass.
STUDY ONLY. 0-for-9 base rate; a pass must be UNAMBIGUOUS. Interrogate:
  (1) which of the 5 ob_quality components actually drives the top/bottom discrimination
  (2) does the effect survive on the INDEPENDENT non-crypto family (GOLD+NDX)? (gold inverted)
  (3) crypto episode-clustering: how many INDEPENDENT crypto episodes back the crypto pass
  (4) split-threshold sensitivity (tertile top vs bottom; is the pass median-artifact-fragile)
  (5) is the crypto pass carried by a few large-R winners (drop-top robustness)
"""
from __future__ import annotations
import os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from audit_common import load_basket, basket_stats_from_R, FIXED_RISK, CRYPTO
from run_paired_sizing import (compute_eq_magnet, ob_quality_proxy, paired_dR_ci,
                               decompose, family_of, OBW, OB_RECENT, OB_VOLTHR, OB_MINPIPS)
from trend_continuation_door import TrendContinuationDoor
from backtester import run_backtest


def ob_components(o, h, l, c, v, i, level_price):
    lo = max(0, i - OB_RECENT + 1)
    wh = h[lo:i + 1]; wl = l[lo:i + 1]; wv = v[lo:i + 1]
    if level_price is None or not np.isfinite(level_price) or level_price <= 0:
        level_price = c[i]
    rng_pct = (np.nanmax(wh) - np.nanmin(wl)) / level_price
    consol = min(1.0, max(0.0, 1.0 - rng_pct / 0.05))
    recent_vol = np.nanmean(wv[-5:]); avg_vol = np.nanmean(wv)
    vr = recent_vol / avg_vol if avg_vol and avg_vol > 0 else 1.0
    vol_s = min(1.0, vr / 3.0) if vr >= OB_VOLTHR else vr / OB_VOLTHR * 0.5
    rp = (c[i] - level_price) / level_price * 10000.0
    react = min(1.0, rp / (OB_MINPIPS * 2)) if rp >= OB_MINPIPS else max(0.0, rp / OB_MINPIPS)
    body = abs(c[i] - o[i]); lower_wick = (o[i] - l[i]) if c[i] > o[i] else (c[i] - l[i])
    wick = min(1.0, (lower_wick / body if body > 0 else 0.0) / 2.0)
    return dict(consol=consol, vol=vol_s, react=react, wick=wick,
                q=consol*OBW[0]+vol_s*OBW[1]+react*OBW[2]+wick*OBW[3])


def build():
    assets = load_basket(with_campaigns=False, verbose=False)
    rows = []
    for name, a in assets.items():
        df = a["df"]
        strat = TrendContinuationDoor(df, a["sr"], a["bj"], a["eye"], variant="struct", conviction=False)
        res = run_backtest(df, strat, label="v1")
        o=df["open"].to_numpy(float); h=df["high"].to_numpy(float); l=df["low"].to_numpy(float)
        c=df["close"].to_numpy(float); v=df["volume"].to_numpy(float)
        elog = {e["entry_time"]: e for e in strat.entries_log}
        idxmap = {ts:k for k,ts in enumerate(df.index)}
        for t in res["trades"]:
            et=t["entry_time"]; i=idxmap.get(et)
            if i is None: i=int(df.index.get_indexer([et],method="nearest")[0])
            bl=elog.get(et,{}).get("break_level",np.nan)
            comp=ob_components(o,h,l,c,v,i,bl)
            rows.append(dict(name=name, fam=family_of(name), R=float(t["R"]),
                             ts=et, **comp))
    return rows


def main():
    rows = build()
    R = np.array([r["R"] for r in rows])
    q = np.array([r["q"] for r in rows])
    med = np.median(q); h2 = q >= med
    print("="*92)
    print("H2 SKEPTICAL DIAGNOSTICS  |  ob_quality top-half sizing")
    print("="*92)

    # (1) component drivers: corr(component, R) and top/bottom meanR per component
    print("\n(1) COMPONENT DRIVERS — Pearson corr(component,R) and which component carries the split")
    for comp in ["consol","vol","react","wick","q"]:
        x = np.array([r[comp] for r in rows])
        if x.std() < 1e-12:
            print(f"    {comp:<7} CONSTANT (std~0) -> inert"); continue
        r_pear = np.corrcoef(x, R)[0,1]
        hi = x >= np.median(x)
        print(f"    {comp:<7} corr(R)={r_pear:+.3f}   topHalf meanR={R[hi].mean():+.3f} "
              f"botHalf meanR={R[~hi].mean():+.3f}   (top cover {hi.mean():.0%})")

    # (2) INDEPENDENT non-crypto family only (GOLD+NDX): does H2 survive?
    print("\n(2) INDEPENDENT NON-CRYPTO (GOLD+NDX) — the real portability test (gold inverted in main)")
    for subset, mask in [("ALL", np.ones(len(rows),bool)),
                         ("crypto", np.array([r["fam"]=="crypto" for r in rows])),
                         ("noncrypto(GOLD+NDX)", np.array([r["fam"]!="crypto" for r in rows]))]:
        Rs=R[mask]; qs=q[mask]
        # use the POOLED median flag (as the strategy would apply it) to avoid re-fitting per subset
        fl=h2[mask]
        dec=decompose(Rs,fl); ci=paired_dR_ci(Rs,fl)
        print(f"    {subset:<20} n={len(Rs):>3} cover={dec['cover']:.0%} "
              f"meanR_flag={dec['mean_flag']:+.3f} meanR_unflag={dec['mean_unflag']:+.3f} "
              f"ΔR_CI=[{ci['ci_lo']:+.4f},{ci['ci_hi']:+.4f}] {'PASS' if ci['ci_lo']>0 else 'fail'}")

    # (3) crypto episode-clustering: independent episodes behind the crypto pass
    print("\n(3) CRYPTO EPISODE-CLUSTERING (±5d cross-asset) — effective independent n")
    cr = [r for r in rows if r["fam"]=="crypto"]
    ev = sorted(cr, key=lambda x: x["ts"])
    episodes=[]; cur=[]
    for e in ev:
        if not cur: cur=[e]; continue
        if (e["ts"]-cur[-1]["ts"]).days<=5: cur.append(e)
        else: episodes.append(cur); cur=[e]
    if cur: episodes.append(cur)
    n_cr=len(cr); n_ep=len(episodes)
    flagged_ep=sum(1 for ep in episodes if any((r["q"]>=med) for r in ep))
    print(f"    crypto trades={n_cr} -> independent episodes(±5d)={n_ep}  "
          f"(episodes touching a flagged trade={flagged_ep})")
    print(f"    => the crypto per-trade bootstrap treats {n_cr} as independent; the honest count is ~{n_ep}.")

    # (4) tertile sensitivity (top third vs bottom third meanR)
    print("\n(4) SPLIT-THRESHOLD SENSITIVITY (is the pass a median artifact?)")
    order=np.argsort(q); k=len(q)//3
    bot=order[:k]; top=order[-k:]
    print(f"    tertile: top{k} meanR={R[top].mean():+.3f}  bottom{k} meanR={R[bot].mean():+.3f}")
    for thr_name,thr in [("40th pct",40),("median",50),("60th pct",60)]:
        cut=np.percentile(q,thr); fl=q>=cut
        dec=decompose(R,fl); ci=paired_dR_ci(R,fl)
        print(f"    {thr_name:<9} cover={fl.mean():.0%} meanR_flag={dec['mean_flag']:+.3f} "
              f"unflag={dec['mean_unflag']:+.3f} ΔR_CI=[{ci['ci_lo']:+.4f},{ci['ci_hi']:+.4f}] "
              f"{'PASS' if ci['ci_lo']>0 else 'fail'}")

    # (5) drop-top robustness on crypto (is the crypto pass a few big winners?)
    print("\n(5) DROP-TOP ROBUSTNESS (crypto only): remove the largest-R flagged crypto winners")
    cr_mask=np.array([r["fam"]=="crypto" for r in rows])
    Rc=R[cr_mask]; flc=h2[cr_mask]
    fl_R=Rc[flc]; unfl_R=Rc[~flc]
    order=np.argsort(fl_R)
    for drop in [0,1,2,3]:
        keep=order[:len(order)-drop] if drop>0 else order
        m_flag=fl_R[keep].mean() if len(keep) else float('nan')
        print(f"    drop top{drop} flagged winners: flagged meanR={m_flag:+.3f} "
              f"(vs unflagged {unfl_R.mean():+.3f}, n_flag={len(keep)})")


if __name__ == "__main__":
    main()
