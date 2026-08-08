"""
VALIDATION of the HTF-ANCHORED structural-range object (STUDY ONLY).

Re-runs addendum-34's checks A-E, but the make-or-break criterion is now STABILITY
(check B): does an HTF-anchored box live for WEEKS instead of the 0.4-day 1H box?

Variants (all four pre-registered, ledger C.2 flagged hypotheses):
    4H N=3, 4H N=5, 1D N=3, 1D N=5
plus the 1H swing_50 baseline (addendum 34) for reference.

The structural_range machine RULES are UNCHANGED. We only swap the anchor pivots
(swing_low_50/swing_high_50 <- HTF fractal pivots) via htf_pivots.reanchor_frame.
"""
from __future__ import annotations
import sys
import numpy as np
import pandas as pd

from structural_range import build_structural_range, build_structural_poc, ACTIVE
from htf_pivots import reanchor_frame, resample_htf, detect_fractal_pivots, build_htf_pivots
from validate_structural_range import rolling_poc_60d, phi_coef, episodes

STORE = "/Users/rayghandchi/Bull Machine/Bull-machine-/data/features_mtf/BTC_1H_FEATURES_V22_CTX.parquet"
VARIANTS = [("4H", 3), ("4H", 5), ("1D", 3), ("1D", 5)]


def hr(t):
    print("\n" + "=" * 82)
    print(t)
    print("=" * 82)


# ---------------------------------------------------------------- A: causality
def causality_check(df, freq, N):
    """No-repaint truncation for HTF pivots AND the re-anchored range (3 cut points)."""
    df_full = reanchor_frame(df, freq, N)
    sr_full = build_structural_range(df_full)
    cut_points = [int(len(df) * f) for f in (0.35, 0.60, 0.85)]
    all_ok = True
    rows = []
    for cp in cut_points:
        df_tr = reanchor_frame(df.iloc[:cp].copy(), freq, N)
        # pivots must match on the overlap
        piv_ok = (
            np.array_equal(np.nan_to_num(df_full["swing_low_50"].iloc[:cp].to_numpy(), nan=-1),
                           np.nan_to_num(df_tr["swing_low_50"].to_numpy(), nan=-1))
            and np.array_equal(np.nan_to_num(df_full["swing_high_50"].iloc[:cp].to_numpy(), nan=-1),
                               np.nan_to_num(df_tr["swing_high_50"].to_numpy(), nan=-1))
        )
        sr_tr = build_structural_range(df_tr)
        rng_ok = (
            np.array_equal(np.nan_to_num(sr_full["struct_range_low"].iloc[:cp].to_numpy(), nan=-1),
                           np.nan_to_num(sr_tr["struct_range_low"].to_numpy(), nan=-1))
            and np.array_equal(np.nan_to_num(sr_full["struct_range_high"].iloc[:cp].to_numpy(), nan=-1),
                               np.nan_to_num(sr_tr["struct_range_high"].to_numpy(), nan=-1))
            and np.array_equal(sr_full["struct_range_state"].iloc[:cp].to_numpy(),
                               sr_tr["struct_range_state"].to_numpy())
        )
        ok = piv_ok and rng_ok
        all_ok &= ok
        rows.append((cp, df.index[cp - 1].date(), piv_ok, rng_ok, ok))
    return all_ok, rows, sr_full


# ---------------------------------------------------------------- run one variant
def run_variant(df, freq, N, rpoc_cache):
    tag = f"{freq} N={N}"
    hr(f"VARIANT  {tag}")

    # ---- A CAUSALITY
    all_ok, rows, sr = causality_check(df, freq, N)
    print("A. CAUSALITY (no-repaint truncation, 3 cut points):")
    for cp, dt, piv_ok, rng_ok, ok in rows:
        print(f"   cut@{cp:>6} ({dt}): HTF-pivots={'OK' if piv_ok else 'REPAINT'}  "
              f"range={'OK' if rng_ok else 'REPAINT'}  -> {'PASS' if ok else 'FAIL'}")
    print(f"   VERDICT: {'NO REPAINT (causal)' if all_ok else 'REPAINT DETECTED'}")

    active = (sr["struct_range_state"] == ACTIVE).to_numpy()
    age = sr["struct_range_age_bars"].to_numpy()
    rl = sr["struct_range_low"]

    # ---- B STABILITY (the make-or-break test)
    chg_any = ((rl != rl.shift(1)) & rl.notna() & rl.shift(1).notna()).to_numpy()
    redraw = chg_any & (age == 0)
    tighten = chg_any & (age > 0)
    active_aged = active & (age > 0)
    within_hold = 1 - (tighten.sum() / max(active_aged.sum(), 1))
    eps = episodes(sr)
    lt_days = np.array([e["bars"] for e in eps]) / 24.0
    print("\nB. STABILITY  (target: WEEKS; 1H baseline failed at median 0.4d):")
    print(f"   episodes: {len(eps)}   median lifetime = {np.median(lt_days):.1f}d   "
          f"mean = {lt_days.mean():.1f}d   p25/p75 = {np.percentile(lt_days,25):.1f}/"
          f"{np.percentile(lt_days,75):.1f}d   max = {lt_days.max():.1f}d")
    print(f"   struct_range_low changes on {chg_any.mean()*100:.2f}% of ALL bars "
          f"(redraw {redraw.mean()*100:.2f}% + tighten {tighten.mean()*100:.2f}%)")
    print(f"   within-range floor holds on {within_hold*100:.2f}% of in-range bars   "
          f"| active coverage {active.mean()*100:.1f}%")
    weeks = np.median(lt_days) >= 7.0
    print(f"   -> {'WEEKS ACHIEVED' if weeks else 'still sub-week'} "
          f"(median {np.median(lt_days):.1f}d)")

    # ---- C SPRING LOCATION
    close = df["close"].to_numpy(); low = df["low"].to_numpy()
    atr = df["atr_14"].to_numpy(); srl = sr["struct_range_low"].to_numpy()
    struct_springs = ((sr["struct_sweep_low"] == 1).to_numpy()) & active
    print("\nC. STRUCTURAL-SPRING location:")
    print(f"   structural springs (sweep_low in active HTF range): {struct_springs.sum()}  "
          f"(sit AT the drawn low by construction)")
    for col in ["wyckoff_spring_a", "wyckoff_spring_b"]:
        if col not in df.columns:
            continue
        fires = (df[col] > 0).to_numpy()
        nfire = int(fires.sum())
        invalid = fires & ~active
        valid = fires & active
        mis = np.abs(low[valid] - srl[valid]) / atr[valid]
        mis = mis[np.isfinite(mis)]
        within1 = (mis <= 1.0).mean() if len(mis) else np.nan
        print(f"   {col}: {nfire} fires | no-active-range {int(invalid.sum())} "
              f"({invalid.sum()/max(nfire,1)*100:.0f}%) | in-range n={int(valid.sum())} "
              f"median misloc {np.median(mis) if len(mis) else float('nan'):.2f} ATR, "
              f"within-1ATR {within1*100 if len(mis) else float('nan'):.0f}%")

    # ---- D STRUCTURAL-POC decontamination
    spoc = build_structural_poc(df, sr)
    ema200 = df["ema_200"].to_numpy(); up = close > ema200
    # bar-to-bar change frac within active
    s = spoc.to_numpy()
    chg = (s[1:] != s[:-1]) & np.isfinite(s[1:]) & np.isfinite(s[:-1]) & active[1:]
    sp_chg = chg.sum() / max(active[1:].sum(), 1)
    # weekly drift
    w = 168; d = []
    for i in range(w, len(s)):
        if active[i] and active[i - w] and np.isfinite(s[i]) and np.isfinite(s[i - w]) and atr[i] > 0:
            d.append(abs(s[i] - s[i - w]) / atr[i])
    sp_wd = np.median(d) if d else float("nan")
    sp_valid = np.isfinite(s) & active & np.isfinite(ema200)
    phi_sp = phi_coef((close > s)[sp_valid], up[sp_valid])
    agree_sp = ((close > s)[sp_valid] == up[sp_valid]).mean()
    print("\nD. STRUCTURAL-POC trend-decontamination:")
    print(f"   bar-to-bar change {sp_chg*100:.2f}%  | weekly drift {sp_wd:.2f} ATR/wk (n={len(d)})")
    print(f"   trend-contam phi(above-POC vs close>ema200) = {phi_sp:+.3f}  "
          f"agree {agree_sp*100:.1f}%  (target LOW like 1H struct +0.075, NOT rolling +0.515)")

    # ---- E EPISODE SANITY
    print("\nE. EPISODE SANITY (known BTC structures):")
    windows = {
        "2021-H2 top":       ("2021-05-01", "2021-12-31"),
        "2022 bear":         ("2022-01-01", "2022-12-31"),
        "2023 accumulation": ("2023-01-01", "2023-12-31"),
        "2024-03 ATH":       ("2024-01-01", "2024-06-30"),
    }
    for name, (a, b) in windows.items():
        wa, wb = pd.Timestamp(a), pd.Timestamp(b)
        sub = [e for e in eps if not (e["t1"] < wa or e["t0"] > wb)]
        lts = [e["bars"] / 24 for e in sub]
        med = np.median(lts) if lts else float("nan")
        print(f"   {name:<18} {len(sub):>3} boxes, median {med:.1f}d")
        for e in sorted(sub, key=lambda x: -x["bars"])[:3]:
            note = f" floor {e['formed_low']:,.0f}->{e['final_low']:,.0f}" if abs(e['final_low']-e['formed_low'])>1 else ""
            print(f"      {e['t0'].date()}->{e['t1'].date()}  "
                  f"[{e['final_low']:>8,.0f}..{e['formed_high']:>8,.0f}]  "
                  f"{e['bars']/24:>5.1f}d  end={e['break_kind']}{note}")

    return dict(tag=tag, causal=all_ok, n_eps=len(eps),
                med_lt=float(np.median(lt_days)), mean_lt=float(lt_days.mean()),
                max_lt=float(lt_days.max()), chg_frac=float(chg_any.mean()),
                within_hold=float(within_hold), active_cov=float(active.mean()),
                phi=float(phi_sp), poc_wd=float(sp_wd), poc_chg=float(sp_chg),
                weeks=bool(np.median(lt_days) >= 7.0))


def baseline_1h(df):
    """The addendum-34 1H swing_50 anchor, for the comparison row."""
    sr = build_structural_range(df)  # df already carries swing_*_50
    eps = episodes(sr)
    lt = np.array([e["bars"] for e in eps]) / 24.0
    active = (sr["struct_range_state"] == ACTIVE).to_numpy()
    rl = sr["struct_range_low"]
    chg = ((rl != rl.shift(1)) & rl.notna() & rl.shift(1).notna()).to_numpy()
    return dict(tag="1H swing_50 (a34)", med_lt=float(np.median(lt)), mean_lt=float(lt.mean()),
                max_lt=float(lt.max()), n_eps=len(eps), chg_frac=float(chg.mean()),
                active_cov=float(active.mean()))


def main():
    df = pd.read_parquet(STORE)
    print(f"loaded {len(df):,} bars  {df.index[0]} -> {df.index[-1]}")
    base = baseline_1h(df)
    results = [run_variant(df, f, n, None) for f, n in VARIANTS]

    hr("THE STABILITY TABLE  (make-or-break: lifetime in DAYS)")
    print(f"{'variant':<20}{'eps':>5}{'med_d':>8}{'mean_d':>8}{'max_d':>8}"
          f"{'chg%':>8}{'floorHold%':>12}{'actCov%':>9}{'phi':>8}{'weeks?':>8}")
    print(f"{base['tag']:<20}{base['n_eps']:>5}{base['med_lt']:>8.1f}{base['mean_lt']:>8.1f}"
          f"{base['max_lt']:>8.1f}{base['chg_frac']*100:>8.2f}{'--':>12}"
          f"{base['active_cov']*100:>9.1f}{'--':>8}{'NO':>8}")
    for r in results:
        print(f"{r['tag']:<20}{r['n_eps']:>5}{r['med_lt']:>8.1f}{r['mean_lt']:>8.1f}"
              f"{r['max_lt']:>8.1f}{r['chg_frac']*100:>8.2f}{r['within_hold']*100:>12.2f}"
              f"{r['active_cov']*100:>9.1f}{r['phi']:>+8.3f}{('YES' if r['weeks'] else 'no'):>8}")
    print("\n1H baseline (a34): median 0.4d, weekly POC drift 6.35 ATR, contam phi rose to 0.222 when tuned")
    print("target: median lifetime >= 7d (weeks), phi LOW (~+0.075 like the clean 1H struct-POC)")

    hr("DONE")
    print("STUDY ONLY — sensor-fix (HTF anchor), awaiting user decision.")


if __name__ == "__main__":
    main()
