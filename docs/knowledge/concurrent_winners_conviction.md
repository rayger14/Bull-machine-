# Concurrent Winners Conviction Research (Hypothesis #2)

**Branch**: `quant/concurrent-winners-conviction`
**Date**: 2026-05-14
**Status**: Final
**Verdict**: **REJECTED** — concurrent winner firings produce modest signal-level lift but zero trade-level lift when wired as a sizing boost.

---

## 1. Executive Summary

- **Hypothesis tested**: When 2+ winning long archetypes (e.g., `liquidity_compression + funding_divergence`) fire on the SAME bar, the resulting trade is higher-quality and should get a position-size boost (similar to the existing Wyckoff 4H boost from commit `5059285`).
- **Verdict**: **REJECTED**. None of the four variants (1.25x at 2, tiered 1.25/1.5x, bull-only, pairs-only) improved OOS PF over the production baseline. All variants underperformed by $0.3K–$1.3K on $57.8K base PnL.
- **Why it fails**: Phase 1 signal-level analysis DID find modest concurrency lift (54.7% WR alone → 57.5% WR with 1 other archetype firing), but this lift doesn't translate to trade-level lift. Reason: dedup means only ONE archetype takes the trade. Boosting that single trade's size doesn't change its outcome — it just amplifies the same per-trade return.
- **Adjacent finding from parallel agent**: Hypothesis #1 (losers as anti-signals) showed that `oi_divergence` firing CONCURRENTLY with `wick_trap`/`liquidity_sweep`/`funding_divergence` produced trades that were **better, not worse** (+$680/trade, +20pp WR). That's a different cross (loser + winner concurrence, not winner + winner) and merits its own follow-up — see Section 6.

---

## 2. Methodology

### Walk-Forward Split
- Train: 2018-01-01 → 2022-12-31 (340 production trades)
- Test (OOS): 2023-01-01 → 2024-12-31 (464 production trades)

### Phase 1: Signal-level co-occurrence analysis
- For every winning archetype fire across 2018-2024, record same-bar co-fires from other winning archetypes
- Compute forward returns (24h, 72h, 168h) conditioned on concurrency count
- Build pair-frequency matrix for the 7 most-fired winning archetypes

### Phase 3: Trade-level sizing boost variants
Tested 4 variant rules against the production baseline:
| Variant | Rule | Boosts applied (OOS) |
|---------|------|---------------------:|
| `v1_1.25x_2` | If ≥2 winners co-fire → size × 1.25 | 8 |
| `v1_1.25_1.5` | 2 winners × 1.25, 3+ winners × 1.5 | 8 |
| `v2_bull_only` | Same as v1 but only in bull regime | 8 |
| `v3_pairs_only` | Only boost the curated winning pairs (per Phase 1) | 8 |

---

## 3. Findings

### 3.1 Phase 1 — Signal-level forward returns by concurrency

| Concurrency state | n_firings | n_unique_bars | mean_fwd_24h_% | mean_fwd_72h_% | win_rate_72h_% |
|-------------------|----------:|--------------:|---------------:|---------------:|---------------:|
| 1_alone | 2,384 | 2,384 | 0.18 | 0.64 | 54.7 |
| 2_with_1_other | 1,086 | 543 | 0.44 | 0.49 | 57.5 |
| 3plus | 351 | 115 | -0.30 | 0.55 | 56.7 |

**Read**: Concurrency lifts the 24h return modestly (0.18% → 0.44%) and the 72h win rate by ~3 percentage points. Signal-level, the hypothesis has a tiny edge.

### 3.2 Phase 1 — Winner pair frequency (top 7)

| Pair (a, b) | Co-fire count | mean_fwd_72h_% | win_rate_72h_% | mean_fwd_168h_% |
|-------------|--------------:|---------------:|---------------:|----------------:|
| trap_within_trend + wick_trap | **598** | +0.57 | 58.2 | +1.38 |
| liquidity_sweep + wick_trap | 114 | +0.80 | 59.6 | +1.33 |
| liquidity_sweep + trap_within_trend | 107 | +0.91 | 59.8 | +1.55 |
| retest_cluster + wick_trap | 22 | −1.71 | 40.9 | −0.95 |
| retest_cluster + trap_within_trend | 21 | −1.47 | 47.6 | −1.12 |
| liquidity_sweep + retest_cluster | 9 | +3.10 | **77.8** | +3.20 |
| spring + trap_within_trend | 8 | −0.44 | 37.5 | −1.45 |

**Read**: The MOST frequent pair (`trap_within_trend + wick_trap`, 598 co-fires) confirms the dedup-fairness investigation finding from May 14 — these two near-identical archetypes co-fire constantly but only wick_trap wins dedup. The pair has decent forward returns (+1.38% / 168h) but not exceptional.

The most striking pair (`liquidity_sweep + retest_cluster`) has 77.8% win rate over 72h on 9 co-fires — too small to base a rule on.

### 3.3 Phase 3 — Trade-level sizing variants vs baseline

| Variant | Window | Trades | PF | PnL | MaxDD | Sharpe | Δ vs baseline |
|---------|--------|-------:|-----:|--------:|-------:|-------:|--------------:|
| **baseline** | Train 2018-22 | 340 | **1.326** | **$22,753** | -12.31% | 0.48 | — |
| **baseline** | OOS 2023-24 | 464 | **1.787** | **$57,837** | -8.60% | 1.67 | — |
| v1_1.25x_2 | Train | 340 | 1.318 | $22,627 | -12.51% | 0.46 | −$126 |
| v1_1.25x_2 | OOS | 464 | 1.779 | $57,566 | -8.74% | 1.65 | **−$272** |
| v1_1.25_1.5 | OOS | 464 | 1.779 | $57,566 | -8.74% | 1.65 | **−$272** |
| v2_bull_only | OOS | 464 | 1.779 | $57,566 | -8.74% | 1.65 | **−$272** |
| v3_pairs_only | OOS | 464 | 1.779 | $57,566 | -8.74% | 1.65 | **−$272** |

**All four OOS variants land at PF 1.779 with $57,566 PnL** — virtually indistinguishable from each other, all marginally worse than baseline.

### 3.4 Why the signal-level lift doesn't survive to trade-level

The dedup system (`best_per_direction` mode) means only ONE archetype takes the trade on any given bar. When a second winning archetype fires the same bar, it's blocked. So:
- The forward return after bar T is identical whether 1, 2, or 3 archetypes fired on it
- The trade taken on bar T has the same outcome regardless of co-firings
- Multiplying that single trade's size by 1.25x just amplifies the same R-multiple — it doesn't extract new alpha

For concurrency to add alpha, we'd need ONE of:
1. **Take both signals** (violates the position cap / dedup logic — separate study)
2. **Filter for HIGHER-quality concurrence** (e.g., 3+ archetypes fire = much rarer, much higher conviction) — Phase 1 shows this doesn't pan out (3plus has WR 56.7% vs 2_with_1_other 57.5%)
3. **Find a CROSS-class concurrence** where the second archetype is structurally different from the first — this is the oi_divergence finding from Hypothesis #1 (Section 6)

---

## 4. Recommendation

**REJECTED** — do not deploy any of the 4 tested variants. Same-class winner-winner concurrent boost doesn't add alpha because dedup means we're amplifying a single trade outcome, not capturing additional information.

**No production code, config, or YAML changes recommended.**

---

## 5. Sample Size & Honest Caveats

- **OOS boost trigger count = 8**: All 4 variants triggered the concurrency boost on only 8 of the 464 OOS trades. That's too small a fraction to materially move system PnL even if each individual boost worked. The effect ceiling is mechanically tiny.
- **`liquidity_sweep + retest_cluster` pair n=9**: Looks great (77.8% WR) but the sample is way too small for a deployment decision. Could be a real edge worth follow-up with more data.
- **Single regime (2023-2024 bull)**: OOS is bull-skewed. Concurrency dynamics in bear/crisis regimes are untested.
- **Phase 1 measures forward returns**, Phase 3 measures actual trade outcomes — they intentionally don't have to agree. Phase 3 is the binding test.

---

## 6. Adjacent Finding Worth Following Up (from Hypothesis #1)

From the parallel `losers-as-anti-signals` investigation: when `oi_divergence` (a chronic loser archetype) fires within 12h of a `wick_trap`, `liquidity_sweep`, or `funding_divergence` entry, those entries are **better, not worse**:
- +$680 per trade vs baseline
- +20 percentage points win rate vs baseline

This is a CROSS-CLASS concurrence (loser-detector + winner-detector), structurally different from this study's winner-winner concurrence. Intuitively:
- `oi_divergence` detects "institutions are positioning against current price direction" — a smart-money confirmation
- Winning reversal archetypes (`wick_trap`, `liquidity_sweep`, `funding_divergence`) detect structural reversal triggers
- When both fire together: smart-money signal + structural trigger = high-confidence turning point

This was an in-sample finding only and needs proper WFO before deployment. **Recommend follow-up study**:
1. Define rule: when `oi_divergence` has fired within last 12 bars AND a long from `{wick_trap, liquidity_sweep, funding_divergence}` triggers, multiply size by X
2. Sweep X across {1.25, 1.5, 1.75, 2.0} with WFO
3. Reject if train/test PF gap > 30% or OOS trade count < 30

This would be Hypothesis #1b — same hypothesis space as the rejected Hypothesis #2 but with a cross-class signal where dedup CAN'T cannibalize the lift.

---

## 7. What This Doesn't Test

1. **Same-bar dual entries** — taking BOTH archetypes' trades concurrently (bypasses dedup; needs separate position-sizing logic)
2. **Cross-asset concurrence** — e.g., BTC liquidity_sweep + ETH liquidity_sweep within window
3. **Concurrence with non-archetype detectors** — e.g., LC + tf4h_wyckoff_bull as separate gates
4. **The oi_divergence cross-class boost** — referenced in Section 6, separate follow-up

---

## 8. Files

- This report: `docs/knowledge/concurrent_winners_conviction.md`
- Phase 1 raw outputs: `results/cross_archetype/concurrent_conviction/phase1_full/*.csv`
- Phase 3 raw outputs: `results/cross_archetype/concurrent_conviction/phase3_variants/*.csv`
- Backtester additions (scratch): `bin/research/concurrent_winners_phase3_backtest.py`

## 9. Constraints Honored

- READ-ONLY for production: no `engine/`, `configs/`, or YAML modifications
- Standing orders intact: no archetypes disabled, `bypass_threshold` unchanged
- All variant outputs in scratch directories under `results/cross_archetype/concurrent_conviction/`
