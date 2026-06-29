#!/usr/bin/env python3
"""Per-archetype winner-vs-loser forensic — "what makes each archetype win?"

Goal (user, 2026-06-28): make each archetype a BETTER TRIGGER — fire mostly in
the conditions where it actually wins — rather than cutting the book to 2.

For each archetype, split its trades into winners (pnl>0) and losers, then for
each entry-time feature measure how well that feature SEPARATES winners from
losers (rank-AUC: 0.5 = no signal, >0.60 or <0.40 = meaningful, directional).
Compute on TRAIN (2018-2024) and require SIGN-CONSISTENCY on the pristine
HOLDOUT (2025-26) — a discriminator that flips sign OOS is noise (Lesson #54:
fusion has NEGATIVE predictive power; this harness will show that explicitly).

Features tested = curated entry-state from the trade log (fusion, regime/CMI,
domain scores) + the level-awareness features we built (level_quality_low,
day_type, acceptance, sweep, dist_to_support, swing touches) joined by entry ts.

Output: per-archetype ranked discriminators + the directional "win condition",
flagged train/holdout consistent. Hypothesis-generating — survivors go to WFO
boosts (Rule 8), never deployed as raw filters.

Usage: python3 scripts/champion/winner_loser_forensic.py
"""
from __future__ import annotations

import glob
import json
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
CHAMP = REPO / "results/champion_v14"
V14 = REPO / "data/features_mtf/BTC_1H_FEATURES_V14_FULL_LIVE_PATH.parquet"
LEVELF = REPO / "results/rebuild/level_features_full.parquet"
OUT = REPO / "results/champion_v14/winner_loser_forensic.json"

# entry-state features already in the trade log (no lookahead — all known at entry)
LOG_FEATS = ["fusion_score", "instability", "crisis_prob", "trend_align",
             "trend_strength", "dd_score", "chop", "adx_weakness", "wick_sc",
             "vol_instab", "wyckoff_score", "liquidity_score", "momentum_score",
             "smc_score", "atr_at_entry", "threshold_margin", "risk_temp",
             "duration_hours"]
# level-awareness features joined from level_features_full
LEVEL_FEATS = ["level_quality_low", "day_type", "acceptance_bars_above",
               "sweep_low_event", "dist_to_support_atr", "swing_low_touches",
               "eq_low_pool"]
MIN_TRADES = 40  # per archetype for a meaningful split


def auc(feat: np.ndarray, win: np.ndarray) -> float:
    """Rank-AUC: P(feature higher for a winner than a loser). 0.5 = no signal."""
    ok = ~np.isnan(feat)
    f, w = feat[ok], win[ok]
    if w.sum() < 5 or (~w).sum() < 5:
        return float("nan")
    order = np.argsort(f, kind="mergesort")
    ranks = np.empty(len(f)); ranks[order] = np.arange(1, len(f) + 1)
    # average ranks for ties
    nw = int(w.sum()); nl = len(f) - nw
    return float((ranks[w].sum() - nw * (nw + 1) / 2) / (nw * nl))


def load_trades(archetype: str, window: str) -> pd.DataFrame:
    f = CHAMP / archetype / window / "trade_log.csv"
    if not f.exists():
        return pd.DataFrame()
    df = pd.read_csv(f)
    df["ts"] = pd.to_datetime(df["timestamp"], utc=True, format="mixed")
    return df


def enrich(df: pd.DataFrame, level: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    idx = df["ts"].dt.tz_localize(None)
    for c in LEVEL_FEATS:
        if c in level.columns:
            df[c] = level[c].reindex(idx, method="ffill").values
    return df


def analyze(archetype: str, level: pd.DataFrame) -> dict:
    # train = full 2018-2024; holdout = 2025-26 (separate dirs)
    train = enrich(load_trades(archetype, "full"), level)
    hold = enrich(load_trades(archetype, "holdout_2025_26"), level)
    if len(train) < MIN_TRADES:
        return {"archetype": archetype, "n_train": len(train), "skip": "n<40"}
    train["win"] = train["pnl"] > 0
    wr = float(train["win"].mean())
    feats = [c for c in LOG_FEATS + LEVEL_FEATS if c in train.columns]
    rows = []
    for c in feats:
        a_tr = auc(pd.to_numeric(train[c], errors="coerce").to_numpy(float), train["win"].to_numpy(bool))
        if np.isnan(a_tr):
            continue
        rec = {"feature": c, "auc_train": round(a_tr, 3),
               "win_mean": round(float(pd.to_numeric(train.loc[train.win, c], errors="coerce").mean()), 4),
               "loss_mean": round(float(pd.to_numeric(train.loc[~train.win, c], errors="coerce").mean()), 4)}
        if len(hold) >= 15:
            hold["win"] = hold["pnl"] > 0
            a_h = auc(pd.to_numeric(hold[c], errors="coerce").to_numpy(float), hold["win"].to_numpy(bool))
            rec["auc_holdout"] = None if np.isnan(a_h) else round(a_h, 3)
            # sign-consistent = both point the same side of 0.5
            rec["consistent"] = (not np.isnan(a_h)) and ((a_tr - 0.5) * (a_h - 0.5) > 0) and abs(a_tr - 0.5) >= 0.08
        else:
            rec["auc_holdout"] = None; rec["consistent"] = None
        rec["strength"] = round(abs(a_tr - 0.5), 3)
        rows.append(rec)
    rows.sort(key=lambda r: r["strength"], reverse=True)
    return {"archetype": archetype, "n_train": len(train), "n_holdout": len(hold),
            "win_rate_train": round(wr, 3), "discriminators": rows}


def main():
    level = pd.read_parquet(LEVELF)
    archetypes = sorted({Path(p).parents[1].name for p in glob.glob(str(CHAMP / "*/full/trade_log.csv"))})
    results = []
    for a in archetypes:
        r = analyze(a, level)
        results.append(r)
        if "skip" in r:
            print(f"\n=== {a} — SKIP ({r['skip']}, n={r['n_train']}) ==="); continue
        print(f"\n=== {a}  (n_train={r['n_train']}, WR {r['win_rate_train']:.0%}, n_holdout={r['n_holdout']}) ===")
        print(f"  {'feature':<22}{'AUC_tr':>8}{'AUC_ho':>8}{'winµ':>10}{'lossµ':>10}  consistent")
        for d in r["discriminators"][:6]:
            cons = "YES" if d["consistent"] else ("—" if d["consistent"] is None else "no(OOS flip)")
            ah = d["auc_holdout"] if d["auc_holdout"] is not None else "n/a"
            print(f"  {d['feature']:<22}{d['auc_train']:>8}{str(ah):>8}{d['win_mean']:>10}{d['loss_mean']:>10}  {cons}")
    OUT.write_text(json.dumps(results, indent=2, default=str))
    print(f"\nreport -> {OUT}")
    # cross-archetype: which features are consistently discriminative?
    from collections import Counter
    cc = Counter()
    for r in results:
        for d in r.get("discriminators", []):
            if d.get("consistent"):
                cc[d["feature"]] += 1
    print("\n=== features that separate winners OOS-consistently across archetypes ===")
    for f, n in cc.most_common():
        print(f"  {f}: {n} archetypes")


if __name__ == "__main__":
    main()
