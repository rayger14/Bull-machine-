"""
DELIVERABLE 3 — SELECTION-BIAS / NEAR-MISS audit (wyckoff_audit add.66; STUDY ONLY)
==================================================================================
Batch-7 (add.65) tested WI's confluence as AND-overlays on fires ALREADY selected by
the door's 5 structural checks — conditioning on a collider, a frame that tilts toward
NULL. This does the empirical check that frame missed: build the NEAR-MISS population —
retest events that failed EXACTLY ONE of the door's five checks — and ask whether high
confluence RESCUES them (confluent near-misses profitable/comparable to real fires;
non-confluent near-misses bad). If yes, WI's confluence is a COMPLEMENT to marginal
structure (a widening tool) and batch-7's frame genuinely missed it.

THE DOOR'S FIVE CHECKS (M2-broad, per bar i, given R0 regime + causal break):
  C1  eye_dir[i] == 'bull'
  C2  eye_state[i] in {MODEL_FORMING, IN_RANGE, MANIPULATION}   (non-extension)
  C3  a bull CONFIRMED_BREAK within trailing M2_SOS_WIN bars    -> break_level exists
  C4  close[i] >= break_level                                   (retest HELD)
  C5  low[i]   <= break_level + RTZ_ATR*ATR                     (pulled INTO the zone)

NEAR-MISS DEFINITION (pre-registered): a bar that (i) passes R0, (ii) has a break_level
(C3 true — otherwise there is no retest event at all; the C3-miss family is NOT a retest
population and is reported separately as such), and (iii) fails EXACTLY ONE of {C1,C2,C4,C5}.
Tagged by which check failed. REAL fires = all of {C1,C2,C4,C5} pass (n_fail=0). Events are
deduped within each family by DEDUP_K=3 bars per asset (first of a cluster).

TRADE SIM (parity-clean): every event (real + near-miss) is simulated in ISOLATION as a
single position using the door's own _plan geometry (entry at close[i], stop = created_low
- 0.25*ATR, struct TP1/BE/runner, MAX_HOLD) and the SAME exit logic as backtester.run_backtest
(an inline single-trade simulator, asserted bit-identical to run_backtest on the real fires
-> 0.00% referee parity inherited). Isolation removes the one-position-scan confound so
near-miss R and real-fire R are apples-to-apples (both isolated single positions).

CONFLUENCE = the add.65 causal combined score over {N1 freshness<=10, N2a sweep-retest,
N2b candle anatomy, N4 origin-discount}, computed post-hoc from the same causal arrays.

PRE-REGISTERED PASS RULE (fixed before measuring; the rescue hypothesis):
  For a check-family, confluence RESCUES iff, with confluent := score>=K (K=2):
    (R1) confluent-near-miss meanR bootstrap 95% CI-lo > 0, AND
    (R2) confluent-near-miss meanR > non-confluent-near-miss meanR (same family),
  reported per check-family. A GLOBAL rescue verdict requires (R1)&(R2) to hold on the
  POOLED near-miss set AND on >=2 individual check-families (not a single-family artifact).
CAUTION (adjacent to Lesson #54 / fusion): near-miss + confluence-rescue is compensatory
logic. If it PASSES, adoption must be a strictly-bounded two-condition rule
(exactly-one-check-missed AND score>=K), NEVER a weighted score. If it FAILS, the batch-7
null stands FRAME-INDEPENDENT.
"""
from __future__ import annotations
import os, sys, json, warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(HERE, ".."))

from xasset_spx_port import load_spx
from trend_continuation_door import build_daily_sensors, TrendContinuationDoor
from unified_archetype_v2 import (
    UnifiedArchetypeV2, RTZ_ATR, M2_SOS_WIN, LPS_LOOKBACK, STOP_BUF_ATR,
    MAX_HOLD, MIN_TP1_R, DEDUP_K,
)
from engine.features.eye_state import (
    MODEL_FORMING, CONFIRMED_BREAK, TRENDING, IN_RANGE, MANIPULATION,
)
from backtester import run_backtest, _slip_fill, COMMISSION_RATE, INITIAL_CASH, RISK_PCT
from run_wi_batch7 import (
    breadth300_universe, RECENT, pf_of, confirmed_break_events,
    N1_F, N2A_LO, N2A_HI, N2B_WICK, N2B_CLOSEPOS, N4_DISCOUNT, N4_LOOKBACK_DAYS,
)

K_CONF = 2          # pre-registered confluence threshold (score>=2 = "confluent")
BOOT = 10_000
SEED = 20260866
FIXED_RISK = RISK_PCT * INITIAL_CASH


# ---------------------------------------------------------------------------
# inline single-trade simulator — MUST match backtester.run_backtest exactly
# (verified per-asset by parity_check below on the REAL fires)
# ---------------------------------------------------------------------------
def sim_one(highs, lows, closes, entry_idx, stop, tp1, runner_target, be_level, max_hold):
    """Struct-plan single long position; mirrors run_backtest lines 122-223.
    Returns R (pnl / risk_dollars) with the SAME slip/commission/fill discipline."""
    entry_raw = closes[entry_idx]
    entry_fill = _slip_fill(entry_raw, side_is_buy=True)
    stop_dist = abs(entry_fill - stop)
    if stop_dist <= 0 or not np.isfinite(stop_dist):
        return None
    risk_dollars = INITIAL_CASH * RISK_PCT
    qty = risk_dollars / stop_dist
    orig_qty = qty
    remaining = orig_qty
    realized_gross = 0.0
    exit_comm_total = 0.0
    entry_comm = qty * entry_fill * COMMISSION_RATE
    cur_stop = stop
    targets = [(tp1, 0.40)]
    first_tp_done = False
    n = len(closes)

    def _take(px_raw, frac_qty):
        nonlocal realized_gross, exit_comm_total
        fill = _slip_fill(px_raw, side_is_buy=False)
        realized_gross += (fill - entry_fill) * frac_qty
        exit_comm_total += abs(frac_qty) * fill * COMMISSION_RATE
        return fill

    j = entry_idx + 1
    end = min(entry_idx + 1 + max_hold, n)
    last_exit_j = None
    while j < end and remaining > 1e-12:
        lo, hi, cl = lows[j], highs[j], closes[j]
        if lo <= cur_stop:                      # stop first (wick)
            _take(cur_stop, remaining); remaining = 0.0; last_exit_j = j; break
        fired = True
        while fired and targets and remaining > 1e-12:
            fired = False
            lvl, frac = targets[0]
            if cl >= lvl:
                q = min(frac * orig_qty, remaining)
                _take(cl, q); remaining -= q; targets.pop(0); fired = True
                if not first_tp_done:
                    first_tp_done = True
                    if be_level is not None:
                        cur_stop = float(be_level)
                last_exit_j = j
        if runner_target is not None and remaining > 1e-12 and not targets:
            if cl >= runner_target:
                _take(cl, remaining); remaining = 0.0; last_exit_j = j; break
        j += 1
    if remaining > 1e-12:
        jn = min(end - 1, n - 1)
        _take(closes[jn], remaining); last_exit_j = jn; remaining = 0.0
    pnl = realized_gross - entry_comm - exit_comm_total
    return pnl / risk_dollars if risk_dollars > 0 else 0.0


def build_plan_geo(u, i):
    """Recompute the struct-plan geometry the door would use at bar i (verbatim _plan)."""
    atr = u.atr[i]; entry_raw = u.closes[i]
    lps_lo = max(0, i - LPS_LOOKBACK + 1)
    created_low = np.nanmin(u.lows[lps_lo:i + 1])
    if not np.isfinite(created_low):
        return None
    stop = created_low - STOP_BUF_ATR * atr
    R = entry_raw - stop
    if R <= 0:
        return None
    rhigh = u.rhigh[i]; rlow = u.rlow[i]; swh = u.swh50[i]
    tp1 = None
    for cand in (rhigh, swh):
        if np.isfinite(cand) and cand >= entry_raw + MIN_TP1_R * R:
            tp1 = cand; break
    if tp1 is None:
        tp1 = entry_raw + 1 * R
    measured = rhigh + (rhigh - rlow) if (np.isfinite(rhigh) and np.isfinite(rlow) and rhigh > rlow) else -np.inf
    tt = max(entry_raw + 2 * R, measured)
    return dict(stop=stop, tp1=tp1, runner=tt, be=entry_raw)


# ---------------------------------------------------------------------------
# per-asset enumeration + flags
# ---------------------------------------------------------------------------
def enumerate_asset(key, path, recent_only=True):
    df = load_spx(path)
    dfs, sr, bj, eye = build_daily_sensors(df)
    u = UnifiedArchetypeV2(dfs, sr, bj, eye, variant="struct", conviction=False, m2_mode="broad")
    idx = dfs.index
    highs = dfs["high"].to_numpy(float); lows = dfs["low"].to_numpy(float)
    closes = dfs["close"].to_numpy(float); opens = dfs["open"].to_numpy(float)
    atr = dfs["atr_14"].to_numpy(float)
    es = eye["eye_state"].to_numpy(object); ed = eye["eye_dir"].to_numpy(object)
    ru = eye["range_upper_1d"].to_numpy(float)
    ts = np.asarray(idx.values, dtype="datetime64[ns]")
    n = len(dfs)
    recent_cut = np.datetime64(RECENT.to_datetime64())

    events = []   # dicts: tag in {REAL, missC1, missC2, missC4, missC5}
    last_by_fam = {}
    for i in range(n):
        if recent_only and ts[i] < recent_cut:
            continue
        a = atr[i]
        if not (np.isfinite(a) and a > 0):
            continue
        if not u._regime_ok(i):
            continue
        # break_level = most recent bull CONFIRMED_BREAK in window (C3)
        lo = max(0, i - M2_SOS_WIN); bl = np.nan; bidx = None
        for c in range(i - 1, lo - 1, -1):
            if es[c] == CONFIRMED_BREAK and ed[c] == "bull":
                bl = ru[c]; bidx = c; break
        if not np.isfinite(bl):
            continue   # C3 fails -> not a retest event (reported as such in the doc)
        C1 = (ed[i] == "bull")
        C2 = (es[i] in (MODEL_FORMING, IN_RANGE, MANIPULATION))
        C4 = (closes[i] >= bl)
        C5 = (lows[i] <= bl + RTZ_ATR * a)
        checks = {"C1": C1, "C2": C2, "C4": C4, "C5": C5}
        nfail = sum(1 for v in checks.values() if not v)
        if nfail == 0:
            tag = "REAL"
        elif nfail == 1:
            failed = [kk for kk, v in checks.items() if not v][0]
            tag = "miss" + failed
        else:
            continue
        # dedup within family (3 bars per asset)
        if i - last_by_fam.get(tag, -10_000) < DEDUP_K:
            continue
        last_by_fam[tag] = i

        geo = build_plan_geo(u, i)
        if geo is None:
            continue
        Rr = sim_one(highs, lows, closes, i, geo["stop"], geo["tp1"], geo["runner"],
                     geo["be"], MAX_HOLD)
        if Rr is None:
            continue

        # ---- batch-7 causal flags (N1,N2a,N2b,N4) ----
        n1 = bool((i - bidx) <= N1_F)
        n2a = False
        for b in (i, i - 1):
            if b < 0 or not np.isfinite(atr[b]) or atr[b] <= 0:
                continue
            uc = bl - lows[b]
            if (N2A_LO * atr[b] <= uc <= N2A_HI * atr[b]) and (closes[b] >= bl):
                n2a = True; break
        rng = highs[i] - lows[i]
        if rng > 0:
            lw = min(opens[i], closes[i]) - lows[i]
            cp = (closes[i] - lows[i]) / rng
            n2b = bool((lw / rng >= N2B_WICK) and (cp >= N2B_CLOSEPOS))
        else:
            n2b = False
        n4 = False
        t0 = ts[bidx] - np.timedelta64(N4_LOOKBACK_DAYS, "D")
        w = (ts <= ts[bidx]) & (ts >= t0)
        if w.sum() >= 2:
            hi365 = np.nanmax(highs[w]); lo365 = np.nanmin(lows[w])
            if hi365 > lo365:
                n4 = bool((bl - lo365) / (hi365 - lo365) <= N4_DISCOUNT)
        score = int(n1) + int(n2a) + int(n2b) + int(n4)
        events.append(dict(key=key, entry_time=str(idx[i]), tag=tag, R=float(Rr),
                           score=int(score), below=bool(closes[i] < u.ema200[i])))

    # ---- parity check: REAL events via sim_one vs run_backtest door fires ----
    strat = TrendContinuationDoor(dfs, sr, bj, eye, variant="struct", conviction=False)
    res = run_backtest(dfs, strat, label=key)
    door_R = {str(pd.Timestamp(t["entry_time"])): float(t["R"]) for t in res["trades"]}
    return events, door_R


# ---------------------------------------------------------------------------
def boot_ci_mean(x, B=BOOT, seed=SEED):
    x = np.asarray(x, float)
    if len(x) == 0:
        return (float("nan"), float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    idxb = rng.integers(0, len(x), size=(B, len(x)))
    means = x[idxb].mean(1)
    return float(x.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def fam_report(tag, ev):
    R = np.array([e["R"] for e in ev], float)
    sc = np.array([e["score"] for e in ev], int)
    conf = sc >= K_CONF
    m_all, lo_all, hi_all = boot_ci_mean(R)
    if conf.sum() > 0:
        m_c, lo_c, hi_c = boot_ci_mean(R[conf])
    else:
        m_c = lo_c = hi_c = float("nan")
    m_nc = R[~conf].mean() if (~conf).sum() else float("nan")
    r1 = bool(np.isfinite(lo_c) and lo_c > 0)
    r2 = bool(np.isfinite(m_c) and np.isfinite(m_nc) and m_c > m_nc)
    print(f"  {tag:<8} n={len(R):5d}  meanR {m_all:+.3f} [{lo_all:+.3f},{hi_all:+.3f}]  PF {pf_of(R):.2f}"
          f" | conf(n={int(conf.sum())}) meanR {m_c:+.3f} CIlo {lo_c:+.3f}"
          f" | nonconf meanR {m_nc:+.3f} | R1 {r1} R2 {r2} "
          f"{'RESCUE' if (r1 and r2) else '-'}")
    return dict(tag=tag, n=len(R), meanR=m_all, ci_lo=lo_all, pf=pf_of(R),
                n_conf=int(conf.sum()), conf_meanR=m_c, conf_ci_lo=lo_c,
                nonconf_meanR=m_nc, R1=r1, R2=r2, rescue=bool(r1 and r2))


def main():
    print("=" * 100)
    print("DELIVERABLE 3 — NEAR-MISS / SELECTION-BIAS audit (add.66)")
    print("476-stock recent 2018+ window | does confluence RESCUE exactly-one-check-missed retests?")
    print("=" * 100)
    uni = breadth300_universe()
    print(f"universe: {len(uni)} names")

    all_ev = []; parity_fail = []; parity_checked = 0
    for j, (key, path, sec) in enumerate(uni):
        try:
            ev, door_R = enumerate_asset(key, path, recent_only=True)
        except Exception as ex:
            print(f"  ERR {key}: {ex}"); continue
        # parity: REAL events at door entry_times must match door_R (recent subset)
        for e in ev:
            if e["tag"] == "REAL" and e["entry_time"] in door_R:
                parity_checked += 1
                if abs(e["R"] - door_R[e["entry_time"]]) > 1e-6:
                    parity_fail.append((key, e["entry_time"], e["R"], door_R[e["entry_time"]]))
        all_ev += ev
        if (j + 1) % 100 == 0:
            print(f"  ...{j+1}/{len(uni)}  events {len(all_ev)}", flush=True)

    print(f"\nSIM PARITY (REAL events vs run_backtest): {parity_checked} checked, "
          f"{len(parity_fail)} mismatches -> {'0.00% PASS' if not parity_fail else 'FAIL'}")
    if parity_fail[:3]:
        print("  sample mismatches:", parity_fail[:3])

    # census
    from collections import Counter
    cen = Counter(e["tag"] for e in all_ev)
    print("\nEVENT CENSUS:", dict(cen))

    real = [e for e in all_ev if e["tag"] == "REAL"]
    nm = [e for e in all_ev if e["tag"] != "REAL"]
    Rreal = np.array([e["R"] for e in real], float)
    m_r, lo_r, hi_r = boot_ci_mean(Rreal)
    print(f"\nREAL fires (isolated): n={len(real)}  meanR {m_r:+.3f} [{lo_r:+.3f},{hi_r:+.3f}]  PF {pf_of(Rreal):.2f}")

    print("\nPER-CHECK-FAMILY near-miss + confluence split (K>=%d confluent):" % K_CONF)
    fam_out = {}
    for fam in ["missC1", "missC2", "missC4", "missC5"]:
        ev = [e for e in nm if e["tag"] == fam]
        if ev:
            fam_out[fam] = fam_report(fam, ev)
    # pooled near-miss
    print("\nPOOLED near-miss:")
    pooled = fam_report("POOLED", nm)

    # confluence-score monotonicity on near-misses (does R rise with score?)
    print("\nNEAR-MISS meanR by confluence score:")
    scn = np.array([e["score"] for e in nm], int); Rn = np.array([e["R"] for e in nm], float)
    for s in range(5):
        m = scn == s
        if m.sum():
            print(f"   score {s}: n={int(m.sum()):5d}  meanR {Rn[m].mean():+.3f}  PF {pf_of(Rn[m]):.2f}")

    # GLOBAL verdict
    fams_rescue = [f for f, v in fam_out.items() if v["rescue"]]
    global_rescue = bool(pooled["rescue"] and len(fams_rescue) >= 2)
    print("\n" + "=" * 100)
    print(f"GLOBAL RESCUE VERDICT: pooled R1&R2={pooled['rescue']}  families-with-rescue={fams_rescue}")
    _msg = ("CONFLUENCE RESCUES marginal structure (COMPLEMENT) — batch-7 frame missed it"
            if global_rescue else "NO RESCUE — batch-7 null stands FRAME-INDEPENDENT")
    print(f"  => {_msg}")
    print("=" * 100)

    dump = dict(census=dict(cen), n_real=len(real), real_meanR=m_r, real_ci_lo=lo_r,
                real_pf=pf_of(Rreal), families=fam_out, pooled=pooled,
                parity_checked=parity_checked, parity_fail=len(parity_fail),
                global_rescue=global_rescue, fams_rescue=fams_rescue)
    outp = os.path.join(os.path.dirname(__file__), "near_miss_results.json")
    json.dump(dump, open(outp, "w"), indent=2, default=str)
    print(f"\nresults -> {outp}")


if __name__ == "__main__":
    main()
