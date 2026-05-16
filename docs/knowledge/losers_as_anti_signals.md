# Losers-as-Anti-Signals (Hypothesis #1)

**Branch**: `quant/losers-as-anti-signals`
**Backtester**: `bin/backtest_v11_standalone.py` (with new `--save-signal-log` and `--anti-signal-rules` flags)
**Date**: 2026-05-14
**Window**: BTC 1H, 2020-01-01 → 2024-12-31, $100K, 2bps comm, 3bps slip

---

## 1. Executive Summary

**Hypothesis**: When chronic-loser archetypes (`oi_divergence`, `long_squeeze`, `order_block_retest`, `fvg_continuation`) FIRE on the same bar (or within K bars) as a winning archetype's entry, those winner entries have worse outcomes. A binary anti-gate on the loser's fire would be a free system-wide filter.

**Verdict: REJECTED for production. The hypothesis does not generalize OOS.**

- In-sample (2020-2022) showed a borderline-credible negative effect for `confluence_breakout` entries with `oi_divergence` firing in the prior 12 bars while in `risk_off` regime (n=21, Δ-PnL ≈ -$255/trade).
- Out-of-sample (2023-2024), the same pattern **inverts**: concurrent (n=11) actually had +$99.7 mean PnL and 54.5% WR vs +$149.8/43.6% baseline — both positive, but the sign-flip plus tiny n kills any structural read.
- Full-period (2020-2024) Rule A delivered PF +0.05, PnL +$2.9K, MaxDD -2.9pp Sharpe +0.06 vs baseline — entirely sourced from the train period; OOS contribution was -$1.2K PnL, -0.01 PF.
- 2 of the 4 "chronic loser" archetypes named in the brief (`order_block_retest`, `fvg_continuation`) **never fired** in the 2020-2024 baseline (they fired only 3 and 6 times respectively in 2023-2024 OOS). Only `oi_divergence` (1,332 fires) and `long_squeeze` (202 fires) had usable sample sizes.

**Headline co-occurrence finding** (counter to hypothesis): when `oi_divergence` fires within 12h of a `wick_trap`, `liquidity_sweep`, or `funding_divergence` entry, those entries are **better**, not worse. `oi_divergence` is acting as a confluence/timing signal, not an anti-signal, for the top wave-following archetypes. The *one* possible exception (`confluence_breakout` in `risk_off`) did not survive WFO.

---

## 2. Methodology

### Phase 1 — Signal-log capture

Patched `bin/backtest_v11_standalone.py`:
- Added `_signal_log_enabled` toggle and an in-loop hook that records every raw archetype fire (timestamp, bar_idx, archetype, direction, fusion_score, regime_label) BEFORE any threshold/dedup/disabled filter.
- New CLI flag `--save-signal-log` writes `signal_log.csv` next to `trade_log.csv`.

This is necessary because the standard `trade_log.csv` only captures executed entries, not the underlying signal fires used as the anti-signal trigger.

### Phase 2 — Co-occurrence + conditional outcomes

`scripts/cross_archetype/analyze_anti_signals.py`:
- For each (winner, loser) pair, flag each winner entry by whether the loser fired within `{0, 4, 12}` hours.
- Aggregate trade outcomes (PnL, R-multiple, win-rate) by flag.
- Cells with `n_concurrent ≥ 5` are recorded; "credible" cells require `n_concurrent ≥ 10`.

R-multiple defined as `position_pnl / (qty × |entry − stop|)` per `position_id` (summed across scale-out rows).

### Phase 3 — Anti-signal rule injection

Patched the backtester with an `--anti-signal-rules <json>` flag. Each rule = `{winner, loser, window_bars, regimes?}`. Inside the run loop:
- Maintain `_anti_signal_last_fire_bar` per archetype.
- After regime is resolved and BEFORE the threshold filter, drop any winner-signal matching an active rule whose loser fired within `window_bars` (and whose regime matches, if filtered).
- Logs and counts blocks; no production config touched.

### Phase 4 — WFO

Train 2020-2022, OOS test 2023-2024. Baseline and rule variants run with identical config (`configs/bull_machine_isolated_v11_fixed.json`), commission, slippage, and capital.

Anti-signal rules tested:
- **Rule A**: block `confluence_breakout` when `oi_divergence` fired in last 12 bars AND regime=`risk_off`.
- **Rule B**: block `confluence_breakout` when `oi_divergence` fired in last 4 bars, ALL regimes.
- **Rule C**: Rule A + same for `long_squeeze` (within 12 bars, `risk_off`).

---

## 3. Findings

### 3.1 Signal-fire census (full 2020-2024 baseline)

| Archetype | Fires | Trade entries (final) |
|---|---|---|
| oi_divergence | 1,332 | 10 (PF 0.08) |
| liquidity_sweep | 1,317 | 53 (PF 2.85) |
| wick_trap | 638 | 56 (PF 5.10) |
| trap_within_trend | 548 | 6 |
| confluence_breakout | 443 | 264 (PF 1.31) |
| exhaustion_reversal | 279 | 21 |
| long_squeeze | 202 | 34 (PF 0.69) |
| funding_divergence | 157 | 21 |
| retest_cluster | 150 | 0 (all filtered) |
| spring | 69 | 0 |
| liquidity_compression | 40 | 14 |
| liquidity_vacuum | 20 | 0 |
| order_block_retest | 0 | 0 |
| fvg_continuation | 0 | 0 |

`order_block_retest` and `fvg_continuation` listed as chronic underperformers in `MEMORY.md` did **not fire at all** in this baseline — current production gating already suppresses them entirely. They appear only in 2023-2024 OOS (3 and 6 fires respectively), still too few for analysis. Hypothesis #1 effectively narrows to `oi_divergence` and `long_squeeze` only.

### 3.2 Co-occurrence matrix (winner-entry conditioned on loser-fire, full sample 2020-2024)

| Winner | Loser | Window | n_entries | n_concurrent | % concurrent |
|---|---|---|---|---|---|
| wick_trap | oi_divergence | 12h | 23 | 12 | 52% |
| liquidity_sweep | oi_divergence | 12h | 21 | 10 | 48% |
| confluence_breakout | oi_divergence | 12h | 264 | 35 | 13% |
| confluence_breakout | oi_divergence | 4h | 264 | 10 | 4% |
| funding_divergence | oi_divergence | 12h | 21 | 8 | 38% |
| liquidity_compression | oi_divergence | 12h | 14 | 6 | 43% |
| confluence_breakout | long_squeeze | 12h | 264 | 9 | 3% |
| trap_within_trend | long_squeeze | 12h | 3 | 2 | 67% |

### 3.3 Conditional outcome deltas (Δ = concurrent − baseline, full sample)

Cells with `n_concurrent ≥ 10` only (credibility floor):

| Winner | Loser | Window | n_conc | n_base | Δ-PnL/trade | Δ-R | Δ-WR |
|---|---|---|---|---|---|---|---|
| wick_trap | oi_divergence | 12h | 12 | 11 | **+$680** | +0.97 | +20pp |
| liquidity_sweep | oi_divergence | 12h | 10 | 11 | **+$1,232** | +1.08 | +45pp |
| confluence_breakout | oi_divergence | 12h | 35 | 229 | -$14 | -0.36 | +3pp |
| confluence_breakout | oi_divergence | 4h | 10 | 254 | **-$116** | -1.06 | -0pp |

**Reading**: For the top reversal archetypes (`wick_trap`, `liquidity_sweep`, `funding_divergence`), concurrent `oi_divergence` is *positively* correlated with outcomes — it's a confluence signal, not an anti-signal. Only `confluence_breakout` shows a meaningful negative tilt, and only in the tight 4-bar window with n=10.

### 3.4 Regime stratification (the *only* surviving credible candidate)

`confluence_breakout` × `oi_divergence` within 12 bars, **stratified by regime** (full 2020-2024):

| Regime | Concurrent flag | n | PnL_sum | PnL_mean | R_mean | WR |
|---|---|---|---|---|---|---|
| risk_off | 0 (baseline) | 119 | $12,587 | +$105.8 | +1.77 | 42.0% |
| risk_off | 1 (concurrent) | **21** | **-$3,142** | **-$149.6** | +0.33 | 28.6% |
| risk_on | 0 | 76 | $9,432 | +$124.1 | +1.96 | 38.2% |
| risk_on | 1 | 9 | $3,188 | +$354.2 | +2.67 | 66.7% |
| neutral | 0 | 34 | -$2,613 | -$76.9 | +1.07 | 35.3% |
| neutral | 1 | 5 | $2,444 | +$488.8 | +3.44 | 60.0% |

The negative effect is **confined to `risk_off`**, n=21. This was the rule target.

### 3.5 WFO results

| Variant | Window | Trades | PF | PnL | MaxDD | Sharpe | WR |
|---|---|---|---|---|---|---|---|
| Baseline | Train 2020-2022 | 302 | 1.55 | $27,716 | -11.84% | 0.943 | 63.24% |
| **Rule A** | Train 2020-2022 | 288 | **1.66** | **$31,326** | -10.01% | 1.068 | 66.32% |
| Baseline | OOS 2023-2024 | 464 | **1.79** | **$57,837** | -8.60% | **1.669** | 68.53% |
| Rule A | OOS 2023-2024 | 451 | 1.78 | $56,599 | -7.79% | 1.645 | 68.74% |
| Rule C | OOS 2023-2024 | 450 | 1.795 | $57,303 | -7.79% | 1.666 | 68.89% |
| Rule A | Full 2020-2024 | 730 | 1.71 | $85,221 | -12.24% | 1.284 | 67.40% |
| Rule C | Full 2020-2024 | 729 | 1.72 | $85,925 | -12.24% | 1.294 | 67.49% |

**Train→Test delta (Rule A − Baseline)**:
- Train: PF +7.1%, PnL +$3,610, Sharpe +0.13
- OOS:  PF -0.5%, PnL -$1,238, Sharpe -0.02

The train-side gain did **not** replicate OOS. Per WFO acceptance criterion (gap > 30% of train improvement is a reject), this rule fails.

### 3.6 OOS co-occurrence verification

Re-ran Phase 2 analysis on 2023-2024 OOS data alone:

`confluence_breakout × oi_divergence × 12h × risk_off`:
- **Train (2020-2022)**: concurrent n=21, mean -$149.6, WR 28.6%
- **OOS (2023-2024)**: concurrent n=11, **mean +$99.7**, WR 54.5%

The sign of the conditional effect **inverted** from train to test. This is the smoking gun: the pattern is not a stable structural relationship; it is sampling noise (n=21 in a noisy 3-year window).

### 3.7 Effect of Rule A on `confluence_breakout` OOS by regime

| Regime | Baseline n | Baseline PnL | Rule A n | Rule A PnL |
|---|---|---|---|---|
| risk_off | 89 | +$12,783 | 82 (-7) | +$12,064 (-$720) |
| risk_on | 50 | +$17,106 | 51 (+1) | +$16,587 (-$519) |
| neutral | 30 | +$813 | 30 | +$813 |

Rule A blocked 7 CB-in-risk_off entries OOS that collectively had ~+$720 of edge — i.e., we **threw away a small positive expectancy** in exchange for a slight MaxDD reduction. The +1 in risk_on is a downstream re-allocation effect (CB blocked → portfolio slot freed → another CB entry later, which itself happened to be slightly less profitable, hence -$519).

---

## 4. Recommendation

**Do not deploy any anti-signal cross-archetype block.** No candidate survived WFO.

If we ever revisit this hypothesis, the methodology to use:

1. Capture signal_log over a larger sample (Optuna 6-group already produced longer fold data; reuse).
2. Require both:
   - in-sample effect-size > 1.5× the baseline std of per-trade PnL in the targeted cell
   - **same-sign effect** in OOS with n ≥ 30
3. Filter candidates *before* WFO by demanding n_concurrent ≥ 30, not 10.

In the current data, only one (winner, loser, window, regime) cell met even the weakest sample criterion, and it failed OOS — so there is essentially nothing to deploy.

---

## 5. Sample Size & Honest Caveats

- The "chronic losers" list in the brief was 4 archetypes; only 2 (`oi_divergence`, `long_squeeze`) had usable fire counts in 2020-2024. `order_block_retest` and `fvg_continuation` were already suppressed to ~0 trades by current production gating.
- The most-negative cell in the full sample (CB × oi_div × 4h, all regimes) had only n=10 concurrent. Even the regime-stratified candidate (CB × oi_div × 12h × risk_off) had only n=21 in 3 years.
- The OOS confluence_breakout had 169 trades — large enough that a real anti-signal would have shown up. The 7-trade block from Rule A produced a near-zero economic effect (-$720 PnL).
- WFO was a single fold (train/test split), not full rolling WFO. A more aggressive rolling-WFO might surface seasonality but is unlikely to rescue this hypothesis given the OOS sign-flip.

---

## 6. What This Doesn't Test

- **Loser features as gates** (Hypothesis #3). Whether the loser's *underlying feature state* (e.g., the OI-divergence pattern itself, not the archetype's binary fire) predicts winner-archetype outcomes. The archetype-fire is a heavily quantized representation of the underlying feature space; the latter may carry signal that the former destroys.
- **Direction-mismatched anti-signals**. `long_squeeze` is a short signal; this analysis treated its fire as a generic "loser fired" event. A direction-aware test (loser is short, winner is long: opposing forces) was not separately built — the data was too sparse (long_squeeze fires concurrent with confluence_breakout n=9 in 12h, all regimes).
- **Effect on archetypes other than `confluence_breakout`**. The reversal archetypes (wick_trap, liquidity_sweep, funding_divergence) showed *positive* deltas; we did not try to construct a "concurrent loser = boost size" rule. That could be Hypothesis #1b.
- **Loser bar-level features as predictors (not the fire)**. e.g., `oi_change_4h < -0.1` may itself be useful even without the archetype-detection layer.
- **Crisis-regime stratification**. n=7 OOS crisis trades, n=0 in train. Sample too small to evaluate.

---

## 7. Files Modified

### Production-touching (READ-ONLY guidance — patches confined to backtester analysis hooks; no archetype configs, no archetype code, no fusion changes)

- `bin/backtest_v11_standalone.py`:
  - Added `_signal_log_enabled`, `_signal_log_rows`, `_anti_signal_rules`, `_anti_signal_last_fire_bar`, `_anti_signal_blocked_count` state.
  - Inserted SIGNAL_LOG_HOOK after raw signal generation (captures every fire, pre-filter).
  - Inserted ANTI_SIGNAL_HOOK that updates last-fire-bar map for cross-archetype rules.
  - Inserted ANTI_SIGNAL_FILTER block that drops winner signals matching active rules.
  - Added CLI flags: `--save-signal-log`, `--anti-signal-rules <json>`.
  - Save block: writes `signal_log.csv` when enabled.

  The hooks are guarded by `getattr(..., '_signal_log_enabled', False)` and `if self._anti_signal_rules:` — they are no-ops in default runs. **No production behavior changes when flags are absent.** Verified by baseline parity check (default-flag run produces same trade count/PF as pre-patch).

### New scratch / research artifacts

- `scripts/cross_archetype/analyze_anti_signals.py` — Phase 1+2 co-occurrence + conditional outcomes.
- `scripts/cross_archetype/regime_stratify.py` — regime-stratified breakdown for candidates.
- `results/cross_archetype/anti_signals/rule_A.json` — focused rule.
- `results/cross_archetype/anti_signals/rule_B.json` — broad rule.
- `results/cross_archetype/anti_signals/rule_C.json` — combined rule.
- `results/cross_archetype/anti_signals/{baseline,baseline_with_log,rule_A_*,rule_B_*,rule_C_*,baseline_train_2020_2022,rule_A_train_2020_2022,baseline_test_2023_2024,rule_A_test_2023_2024,rule_C_test_2023_2024,rule_C_train_2020_2022,baseline_test_with_log}/` — backtest outputs.
- `results/cross_archetype/anti_signals/analysis/` — Phase 1+2 outputs (full sample).
- `results/cross_archetype/anti_signals/analysis_oos/` — Phase 1+2 outputs (OOS only).

### Production configs/YAMLs touched

**None.** No archetype yaml, no config json, no engine code modified.

---

## 8. Lessons (proposed for MEMORY.md)

- **Lesson 61**: Archetype-fire co-occurrence is too quantized to deliver stable anti-signal effects. `oi_divergence` fires 1,332 times in 5 years but only 10 reach trade execution; the fires that don't reach execution are already losing signals filtered by hard gates. Using the noisy fires as anti-signal triggers picks up too much noise relative to the small structural delta.
- **Lesson 62**: When a "chronic loser" archetype already has near-zero firing rate under current production gates (`order_block_retest`, `fvg_continuation`), the hypothesis "use loser as anti-signal" is vacuously inapplicable — there's no loser fire to use.
- **Lesson 63**: For BTC 1H, n=21 in a 3-year training window is **not enough** to assert a regime-stratified effect. Sign-flips in OOS are the rule, not the exception, at that sample size.
