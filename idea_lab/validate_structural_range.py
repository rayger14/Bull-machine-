"""
VALIDATION of the structural-range object (STUDY ONLY).
Runs the pre-registered checks A-E from the task and prints a report to stdout.

A. CAUSALITY  — truncation no-repaint at 3 cut points.
B. STABILITY  — change-frac of struct_range_low vs rolling POC ~97%; lifetime in DAYS.
C. STRUCTURAL-SPRING re-test — structural spring = struct_sweep_low inside active range;
   quantify mislocation of the OLD rolling wyckoff_spring_a/b vs concurrent struct_range_low,
   and how many rolling springs fired when NO active range existed.
D. STRUCTURAL-POC re-test — bar-to-bar drift, weekly drift, trend-contamination phi,
   vs a faithfully-reconstructed rolling-60d POC baseline (addendum 33: ~97% move, phi .46-.55).
E. EPISODE SANITY — overlay drawn ranges on known BTC structures.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from structural_range import (
    build_structural_range, build_structural_poc, ACTIVE, FORMING,
    BROKEN_UP, BROKEN_DOWN, MIN_WIDTH_ATR, TIGHTEN_MIN_WIDTH_ATR,
)

STORE = "/Users/rayghandchi/Bull Machine/Bull-machine-/data/features_mtf/BTC_1H_FEATURES_V22_CTX.parquet"


def hr(t):
    print("\n" + "=" * 78)
    print(t)
    print("=" * 78)


# ---------------------------------------------------------------- rolling POC
def rolling_poc_60d(df, window_bars=60 * 24, bin_pct=0.0025):
    """Faithful reconstruction of the audit's rolling-60d POC: max-volume 0.25%
    bin over the trailing window's [min low, max high]. Vectorized-ish."""
    close = df["close"].to_numpy(float)
    high = df["high"].to_numpy(float)
    low = df["low"].to_numpy(float)
    vol = df["volume"].to_numpy(float)
    n = len(df)
    poc = np.full(n, np.nan)
    step = 200  # only evaluate every `step` bars for speed of weekly-drift metric
    # We need bar-to-bar change-frac, so evaluate every bar but keep it cheap:
    for i in range(n):
        if i < window_bars:
            continue
        s = i - window_bars + 1
        w_lo = low[s:i + 1].min()
        w_hi = high[s:i + 1].max()
        if w_hi <= w_lo:
            continue
        binsz = close[i] * bin_pct
        if binsz <= 0:
            continue
        idx = ((close[s:i + 1] - w_lo) / binsz).astype(np.int64)
        # accumulate volume per bin
        order = np.argsort(idx)
        idx_s = idx[order]
        vol_s = vol[s:i + 1][order]
        uniq, start = np.unique(idx_s, return_index=True)
        sums = np.add.reduceat(vol_s, start)
        best = uniq[np.argmax(sums)]
        poc[i] = w_lo + (best + 0.5) * binsz
    return pd.Series(poc, index=df.index, name="rolling_poc")


def phi_coef(a, b):
    """Matthews/phi correlation between two boolean arrays."""
    a = a.astype(bool); b = b.astype(bool)
    n11 = np.sum(a & b); n10 = np.sum(a & ~b)
    n01 = np.sum(~a & b); n00 = np.sum(~a & ~b)
    num = n11 * n00 - n10 * n01
    den = np.sqrt((n11 + n10) * (n01 + n00) * (n11 + n01) * (n10 + n00))
    return num / den if den > 0 else np.nan


def episodes(sr):
    """Return list of (start_idx, end_idx, formed_low, formed_high, final_low,
    lifetime_bars) for each active episode (contiguous run in ACTIVE, allowing
    intra-episode floor tightening; an episode ends at a break bar)."""
    state = sr["struct_range_state"].to_numpy(object)
    rl = sr["struct_range_low"].to_numpy(float)
    rh = sr["struct_range_high"].to_numpy(float)
    idx = sr.index
    eps = []
    i = 0
    n = len(sr)
    while i < n:
        if state[i] == ACTIVE:
            start = i
            formed_low, formed_high = rl[i], rh[i]
            j = i
            while j < n and state[j] == ACTIVE:
                j += 1
            end = j - 1
            final_low = rl[end]
            eps.append(dict(start=start, end=end, t0=idx[start], t1=idx[end],
                            formed_low=formed_low, formed_high=formed_high,
                            final_low=final_low, bars=end - start + 1,
                            break_kind=state[j] if j < n else "eod"))
            i = j
        else:
            i += 1
    return eps


def main():
    df = pd.read_parquet(STORE)
    print(f"loaded {len(df):,} bars  {df.index[0]} -> {df.index[-1]}")
    sr = build_structural_range(df)
    active = (sr["struct_range_state"] == ACTIVE).to_numpy()

    # ------------------------------------------------------------------ A
    hr("A. CAUSALITY — truncation no-repaint (3 cut points)")
    cut_points = [int(len(df) * f) for f in (0.35, 0.60, 0.85)]
    all_ok = True
    for cp in cut_points:
        sr_trunc = build_structural_range(df.iloc[:cp].copy())
        # compare emitted range levels for bars [0..cp-1]
        a_lo = sr["struct_range_low"].iloc[:cp].to_numpy()
        b_lo = sr_trunc["struct_range_low"].to_numpy()
        a_hi = sr["struct_range_high"].iloc[:cp].to_numpy()
        b_hi = sr_trunc["struct_range_high"].to_numpy()
        eq_lo = np.array_equal(np.nan_to_num(a_lo, nan=-1), np.nan_to_num(b_lo, nan=-1))
        eq_hi = np.array_equal(np.nan_to_num(a_hi, nan=-1), np.nan_to_num(b_hi, nan=-1))
        st_a = sr["struct_range_state"].iloc[:cp].to_numpy()
        st_b = sr_trunc["struct_range_state"].to_numpy()
        eq_st = np.array_equal(st_a, st_b)
        ok = eq_lo and eq_hi and eq_st
        all_ok &= ok
        print(f"  cut @ bar {cp:>6} ({df.index[cp-1].date()}): "
              f"range_low={'OK' if eq_lo else 'REPAINT'}  "
              f"range_high={'OK' if eq_hi else 'REPAINT'}  "
              f"state={'OK' if eq_st else 'REPAINT'}  -> {'PASS' if ok else 'FAIL'}")
    print(f"  VERDICT: {'NO REPAINT (causal)' if all_ok else 'REPAINT DETECTED'}")

    # ------------------------------------------------------------------ B
    hr("B. STABILITY — is it a drawn box, not a sliding window?")
    rl = sr["struct_range_low"]
    age = sr["struct_range_age_bars"].to_numpy()
    chg_any = ((rl != rl.shift(1)) & rl.notna() & rl.shift(1).notna()).to_numpy()
    # classify each change as redraw (new episode, age==0) vs in-range tighten (age>0)
    redraw = chg_any & (age == 0)
    tighten = chg_any & (age > 0)
    print(f"  struct_range_low changes on {chg_any.mean()*100:.2f}% of ALL bars "
          f"(rolling-60d POC baseline: ~97%)")
    print(f"    of which redraws (new box):     {redraw.sum():>4}  ({redraw.mean()*100:.2f}% of bars)")
    print(f"    of which in-range tightenings:  {tighten.sum():>4}  ({tighten.mean()*100:.2f}% of bars)")
    # within-range stability: among ACTIVE bars with age>0, how often does floor move?
    active_aged = active & (age > 0)
    within_chg = tighten.sum() / max(active_aged.sum(), 1)
    print(f"  WITHIN-RANGE floor stability: floor holds on "
          f"{(1-within_chg)*100:.2f}% of in-range bars")
    eps = episodes(sr)
    lifetimes_days = np.array([e["bars"] for e in eps]) / 24.0
    print(f"  episodes: {len(eps)}   median lifetime = {np.median(lifetimes_days):.1f} days "
          f"(mean {lifetimes_days.mean():.1f}, max {lifetimes_days.max():.1f})")
    print(f"  active-bar coverage: {active.mean()*100:.1f}%   "
          f"forming: {(sr['struct_range_state']=='forming').mean()*100:.1f}%")

    # ------------------------------------------------------------------ C
    hr("C. STRUCTURAL-SPRING re-test vs rolling wyckoff_spring_a/b")
    sweep_lo = (sr["struct_sweep_low"] == 1).to_numpy()
    struct_springs = sweep_lo & active
    print(f"  structural springs (struct_sweep_low inside active range): {struct_springs.sum()}")
    print(f"    -> by construction these sit AT the drawn range low (mislocation = 0 by design)")
    srl = sr["struct_range_low"].to_numpy()
    atr = df["atr_14"].to_numpy()
    close = df["close"].to_numpy()
    low = df["low"].to_numpy()
    for col in ["wyckoff_spring_a", "wyckoff_spring_b"]:
        if col not in df.columns:
            continue
        fires = (df[col] > 0).to_numpy()
        nfire = int(fires.sum())
        # how many fired with NO active range = structurally invalid
        invalid = fires & ~active
        n_invalid = int(invalid.sum())
        # for those WITH an active range, mislocation of the spring bar's low vs struct_range_low
        valid = fires & active
        mis_atr = np.abs(low[valid] - srl[valid]) / atr[valid]
        mis_atr = mis_atr[np.isfinite(mis_atr)]
        # also close-based distance
        misc_atr = np.abs(close[valid] - srl[valid]) / atr[valid]
        misc_atr = misc_atr[np.isfinite(misc_atr)]
        # fraction sitting within 1 ATR of the drawn low ("correctly located")
        within1 = (mis_atr <= 1.0).mean() if len(mis_atr) else np.nan
        print(f"\n  {col}: {nfire} fires")
        print(f"    fired with NO active range (structurally invalid): "
              f"{n_invalid} ({n_invalid/max(nfire,1)*100:.0f}%)")
        if len(mis_atr):
            print(f"    when a range WAS active (n={valid.sum()}): spring-low vs drawn range_low")
            print(f"      median mislocation = {np.median(mis_atr):.2f} ATR   "
                  f"mean = {mis_atr.mean():.2f} ATR   "
                  f"within 1 ATR = {within1*100:.0f}%")
            print(f"      (close-based median = {np.median(misc_atr):.2f} ATR)")

    # ------------------------------------------------------------------ D
    hr("D. STRUCTURAL-POC re-test vs rolling-60d POC (addendum 33)")
    spoc = build_structural_poc(df, sr)
    print("  computing rolling-60d POC baseline (this takes ~1-2 min)...")
    rpoc = rolling_poc_60d(df)
    ema200 = df["ema_200"].to_numpy()
    up = close > ema200  # trend proxy

    # bar-to-bar drift (change frac)
    def change_frac(series, mask=None):
        s = series.to_numpy()
        chg = (s[1:] != s[:-1]) & np.isfinite(s[1:]) & np.isfinite(s[:-1])
        if mask is not None:
            chg = chg & mask[1:]
        base = mask[1:].sum() if mask is not None else np.isfinite(s[1:]).sum()
        return chg.sum() / max(base, 1)

    sp_chg = change_frac(spoc, active)
    rp_chg = change_frac(rpoc)
    print(f"  bar-to-bar change frac:")
    print(f"    structural POC (within active range): {sp_chg*100:.2f}%")
    print(f"    rolling-60d POC (reconstructed):      {rp_chg*100:.2f}%  (audit said ~97%)")

    # weekly drift in ATR (change over 168 bars)
    def weekly_drift_atr(series, mask):
        s = series.to_numpy()
        w = 168
        d = []
        for i in range(w, len(s)):
            if mask[i] and mask[i - w] and np.isfinite(s[i]) and np.isfinite(s[i - w]) and atr[i] > 0:
                d.append(abs(s[i] - s[i - w]) / atr[i])
        return np.median(d) if d else np.nan, len(d)

    sp_wd, sp_n = weekly_drift_atr(spoc, active)
    rp_wd, rp_n = weekly_drift_atr(rpoc, np.isfinite(rpoc.to_numpy()))
    print(f"  weekly drift (median ATR/week):")
    print(f"    structural POC: {sp_wd:.2f} ATR/wk  (n={sp_n})")
    print(f"    rolling-60d POC: {rp_wd:.2f} ATR/wk  (n={rp_n})  (audit said 0.3-1.1)")

    # trend contamination phi: above_POC vs close>ema200
    sp_valid = np.isfinite(spoc.to_numpy()) & active & np.isfinite(ema200)
    rp_valid = np.isfinite(rpoc.to_numpy()) & np.isfinite(ema200)
    sp_above = (close > spoc.to_numpy())[sp_valid]
    sp_up = up[sp_valid]
    rp_above = (close > rpoc.to_numpy())[rp_valid]
    rp_up = up[rp_valid]
    phi_sp = phi_coef(sp_above, sp_up)
    phi_rp = phi_coef(rp_above, rp_up)
    agree_sp = (sp_above == sp_up).mean()
    agree_rp = (rp_above == rp_up).mean()
    print(f"  trend contamination (above-POC vs close>ema200):")
    print(f"    structural POC: phi={phi_sp:+.3f}  agreement={agree_sp*100:.1f}%  (n={sp_valid.sum()})")
    print(f"    rolling-60d POC: phi={phi_rp:+.3f}  agreement={agree_rp*100:.1f}%  "
          f"(audit said phi .46-.55, agree 72-77%)")
    print(f"    -> lower phi = LESS trend contamination (more mean-reversion-usable)")

    # ------------------------------------------------------------------ E
    hr("E. EPISODE SANITY — drawn ranges over known BTC structures")
    windows = {
        "2021-H2 top range":   ("2021-05-01", "2021-12-31"),
        "2022 bear ranges":    ("2022-01-01", "2022-12-31"),
        "2023 accumulation":   ("2023-01-01", "2023-12-31"),
        "2024-03 ATH region":  ("2024-01-01", "2024-06-30"),
        "2025 cycle":          ("2025-01-01", "2025-12-31"),
    }
    for name, (a, b) in windows.items():
        wa, wb = pd.Timestamp(a), pd.Timestamp(b)
        sub = [e for e in eps if not (e["t1"] < wa or e["t0"] > wb)]
        print(f"\n  {name}  [{a} .. {b}]  ({len(sub)} boxes)")
        for e in sub[:8]:
            lo0, lo1 = e["formed_low"], e["final_low"]
            tighten_note = f" (floor {lo0:,.0f}->{lo1:,.0f})" if abs(lo1 - lo0) > 1 else ""
            print(f"    {e['t0'].date()} -> {e['t1'].date()}  "
                  f"[{lo1:>8,.0f} .. {e['formed_high']:>8,.0f}]  "
                  f"{e['bars']/24:>5.1f}d  end={e['break_kind']}{tighten_note}")
        if len(sub) > 8:
            print(f"    ... (+{len(sub)-8} more)")

    hr("DONE")
    print("STUDY ONLY — sensor-fix prototype, awaiting user decision.")


if __name__ == "__main__":
    main()
