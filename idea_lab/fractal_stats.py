"""Paired per-trade delta statistics for the 2x2 execution study (STUDY ONLY)."""
from __future__ import annotations
import numpy as np

RNG = np.random.default_rng(20260811)   # fixed seed — reproducible bootstrap


def bootstrap_ci(deltas, n_boot=10_000, alpha=0.05):
    """Bootstrap 95% CI on the MEAN of paired deltas (10k resamples, fixed seed)."""
    d = np.asarray(deltas, float)
    if len(d) == 0:
        return (np.nan, np.nan)
    idx = RNG.integers(0, len(d), size=(n_boot, len(d)))
    means = d[idx].mean(axis=1)
    return float(np.quantile(means, alpha / 2)), float(np.quantile(means, 1 - alpha / 2))


def paired_summary(arm_R, a_R, label):
    """arm_R, a_R aligned per-fire R arrays. Returns a dict of paired stats."""
    arm = np.asarray(arm_R, float); a = np.asarray(a_R, float)
    d = arm - a
    lo, hi = bootstrap_ci(d)
    ci_excl0 = (lo > 0) or (hi < 0)
    return {
        "label": label, "n": len(d),
        "mean_dR": float(d.mean()) if len(d) else np.nan,
        "median_dR": float(np.median(d)) if len(d) else np.nan,
        "pct_improved": float((d > 1e-9).mean() * 100) if len(d) else np.nan,
        "pct_worse": float((d < -1e-9).mean() * 100) if len(d) else np.nan,
        "ci_lo": lo, "ci_hi": hi, "ci_excludes_0": bool(ci_excl0),
        "wr_arm": float((arm > 0).mean() * 100) if len(arm) else np.nan,
        "wr_a": float((a > 0).mean() * 100) if len(a) else np.nan,
        "sum_dR": float(d.sum()),
        "totR_arm": float(arm.sum()), "totR_a": float(a.sum()),
        "maxR_arm": float(arm.max()) if len(arm) else np.nan,
        "maxR_a": float(a.max()) if len(a) else np.nan,
    }


def fmt_row(s):
    ci = f"[{s['ci_lo']:+.3f},{s['ci_hi']:+.3f}]"
    star = "*" if s["ci_excludes_0"] else " "
    return (f"{s['label']:<22} n={s['n']:>3}  meanDR={s['mean_dR']:+.3f}{star} "
            f"medDR={s['median_dR']:+.3f}  impr={s['pct_improved']:4.0f}%  "
            f"CI95={ci:<20} WR {s['wr_a']:4.0f}->{s['wr_arm']:4.0f}%  "
            f"maxR {s['maxR_a']:.2f}->{s['maxR_arm']:.2f}  sumDR={s['sum_dR']:+.1f}")
