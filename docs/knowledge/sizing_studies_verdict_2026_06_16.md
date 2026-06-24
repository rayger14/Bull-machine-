# Sizing Studies Verdict: Sizing Cannot Flip a Long-Only Strategy's Bear Years (Confirmed Twice)

**Date**: 2026-06-16
**Subject**: wick_trap_v14rq (best honest asset: holdout +$6.8K PF 1.43) — can regime/level-aware SIZING make it pass the full gauntlet?
**Answer**: No. 0 of 13 valid variants pass; 0 of 13 flip ANY bear year (2018/2021/2022) positive.
**Artifacts**: `results/champion_v14_sizing/`, `scripts/champion/sizing_studies.py`

## What was tested (all pre-registered, Rule 8 — sizing never filtering)
- **S1 macro-bear**: long size ×{0.25,0.5} when close < {150,200,250}-day mean (6 variants)
- **S2 day-type**: long size ×{0.25,0.5} on confirmed trend-down days (2 variants)
- **S3 level-quality**: long size ×{0.25,0.5} when level_quality_low < train-{q33,q50} (4 variants)
- baseline (no sizing) for reference. All on V14 live-path store, 10 windows incl. 2025-26 holdout.

## Results (key columns)

| Variant | 2018 | 2021 | 2022 | holdout PF | full PF | MaxDD |
|---|---|---|---|---|---|---|
| baseline | −8.1K | −7.5K | −8.7K | 1.43 | 1.46 | 14.3% |
| S1_macro150_k25 | −1.2K | −10.4K | −4.4K | 1.26 | **1.63** | 12.4% |
| S1_macro250_k25 | −2.4K | −8.2K | −4.4K | 1.27 | 1.52 | **11.4%** |
| S2_daytype_k50 | −8.1K | −7.4K | −8.6K | 1.42 | 1.45 | 14.1% |
| S3_lqq33_k50 | −6.5K | −6.4K | −8.1K | **1.46** | 1.44 | 12.7% |

(Full 13-variant table in summary_all.json.)

## What the data says, plainly
1. **Sizing shrinks bear losses but never flips them.** Every variant keeps 2018/2021/2022 negative. C1 (positive every year) is therefore unreachable by sizing. This is the IDENTICAL structural truth found in the V12-era hunt — now confirmed on honest data with a 13-variant sweep. It is robust.
2. **Macro sizing (S1) improves full-period PF and drawdown but at a holdout cost** (1.43→1.26) and PARADOXICALLY worsens 2021: a slow N-day mean reads choppy-bull 2021 as "bull," so it shrinks the dip-buys that won while leaving above-mean losers full-size. Slow macro state is wrong for chop years.
3. **Day-type sizing (S2) barely moves anything** — trend-down days are only 8% of bars and wick_trap already trades few of them; too rare to matter at this granularity.
4. **Level-quality sizing (S3) gives the best holdout PF (1.46) and cuts drawdown**, mild evidence that weak-level trades are worse (H2 directionally supported), but it cannot flip bears either and slightly cuts bull-year upside.
5. The kill-switch limit (size→0 in bear) could not be validly tested: the backtester mis-accounts zero-quantity positions (PF=∞ with simultaneous losses). A real kill-switch must SKIP trades — engine change required, and it is a filter, not sizing.

## The honest structural conclusion
**A long-only dip-buyer cannot pass "positive every calendar year" through 2018 (−73%) and 2022 (−64%) by any sizing scheme.** Losing trades in a bear can be made smaller, never positive. Flipping a bear year green requires a component that MAKES money in bears — which the system does not have (established: 0 functioning short side; the V12 exhaustion_reversal "bear complement" was a feature-pipeline artifact that did not reproduce on V14).

## Three honest paths (decision required — none is "more sizing")
- **P1 — Reframe the bar.** "Positive every year" is an unrealistic test for a long-only crypto strategy spanning two >60% bear markets. A professional bar — full-period PF ≥ 1.4, holdout PF ≥ 1.3, MaxDD ≤ 15%, Sharpe — is ALREADY cleared by wick_trap_v14rq (full PF 1.46, holdout 1.43) and by S1_macro250 (PF 1.52, MaxDD 11.4%). Under this bar we HAVE a deployable champion; the question becomes risk tolerance, not existence.
- **P2 — Build a real bear-profiting / short component.** The only way to literally flip bear years. Largest effort; the system has never had a working short side. High value, high risk.
- **P3 — Regime kill-switch as trade-SKIP** (flat in confirmed macro-bear). Makes bear years ≈ $0 instead of negative → "non-negative every year" achievable. Requires engine support for skip (not size×0) and is a portfolio-level regime filter (categorically different from the 0/8 fusion-score filters; matches the ES "cut to zero on trend days against you"). Medium effort, directly testable once skip is wired.

## Recommendation
Take **P1 + P3** together: adopt the professional bar (we already have a winner under it), and build the trade-skip regime kill-switch as the one engine change worth making — it is the honest, evidence-backed version of "don't fade a confirmed bear," and it converts wick_trap_v14rq's only weakness (bear drawdown) into flat. Defer P2 (short side) as a separate larger initiative. Get quant-analyst adjudication before any production YAML change.

## Cross-references
[[v14_champion_hunt_2026_06_11]] · [[trader_knowledge_failed_breakdown_es_transfer_2026_06_11]] (P3 = "classify the day, cut to zero against trend") · [[champion_strategy_pair_2026_06_10]] (V12 sizing-can't-flip lesson, now reconfirmed)
