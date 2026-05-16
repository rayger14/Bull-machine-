# OI-Divergence Concurrent-Fire Sizing Boost — WFO Study (Hypothesis #1b)

**Branch**: `quant/oi-div-sizing-boost-wfo`
**Date**: 2026-05-16
**Status**: Final
**Verdict**: **REJECTED** — sample size too small (n=15 OOS) and OOS lift is below noise floor.

---

## 1. Executive Summary

- **Hypothesis tested**: When `oi_divergence` fires within 12 bars of a long entry from `{wick_trap, liquidity_sweep, funding_divergence}`, multiply position size by X. The original Hypothesis #1 (anti-signals) discovered the concurrent pattern produced +$680/trade and +20pp WR in-sample, suggesting it could be wired as a sizing boost (Hypothesis #1b).
- **OOS trade count is tiny**: only **15 of 464 OOS trades (3.2%)** triggered the boost across 2023-2024. **Below the n=30 directional-only floor** — any conclusion is statistically weak.
- **Boost ladder is monotonic but tiny**: X=1.25 → +$421 OOS PnL (+0.73%), X=1.5 → +$842 (+1.46%), X=1.75 → +$1,262 (+2.18%). Sharpe creeps from 1.67 → 1.70.
- **MaxDD slightly widens** with higher X (−$860 → −$889 OOS), as expected.
- **The +$680/trade in-sample finding did not generalize** — actual per-trade lift on real-backtest is far smaller than the prior phantom-style analysis suggested.
- **REJECTED for deployment**. Not enough trades, not enough lift, sample too small.

---

## 2. Methodology

### Boost Rule

A position is boosted if ALL conditions are true at entry:
1. Entry archetype ∈ `{wick_trap, liquidity_sweep, funding_divergence}`
2. Entry direction = `long`
3. `oi_divergence` (any direction) was detected within the last 12 bars (canonical signal log, before dedup/threshold filtering)

When all true: `allocated_size_pct *= X`.

Implementation via scratch monkey-patch in `scripts/oi_div_boost/run_variant.py` — no production code modified.

### Walk-Forward Split

- **Train**: 2018-01-01 → 2022-12-31
- **Test (OOS)**: 2023-01-01 → 2024-12-31

### Sweep

X ∈ {1.0 (null/baseline), 1.25, 1.5, 1.75, 2.0}. X=2.0 run failed (likely hit `max_margin_per_position_pct=0.35` cap with no fallback); not included in tables.

### Anti-Overfit Rules (per CLAUDE.md / quant-analyst subagent)

- Reject if train/test PF gap > 30%
- Reject if OOS trade count drops > 50% vs baseline
- Mark `n_boosted < 30` OOS as "directional only — not deployable"
- Real backtest only, no phantoms

---

## 3. Findings

### 3.1 Per-X performance

| X | Window | Trades | PF | PnL | WR | Sharpe | MaxDD | OI fires | Eligible | Boosted |
|---|--------|-------:|------:|--------:|------:|-------:|-------:|---------:|---------:|--------:|
| 1.00 (null) | Train | 340 | 1.326 | $22,753 | 63% | 0.48 | −12.3% | 1,386 | 47 | 20 |
| 1.00 (null) | **Test (OOS)** | **464** | **1.787** | **$57,837** | **69%** | **1.67** | **−8.6%** | **503** | **31** | **15** |
| 1.25 | Train | 340 | 1.327 | $23,252 | 63% | 0.47 | −12.3% | 1,386 | 47 | 20 |
| 1.25 | **Test (OOS)** | **464** | **1.791** | **$58,258** | 69% | 1.68 | −8.7% | 503 | 31 | **15** |
| 1.50 | Train | 340 | 1.324 | $23,235 | 63% | 0.47 | −12.3% | 1,386 | 47 | 20 |
| 1.50 | **Test (OOS)** | **464** | **1.795** | **$58,679** | 69% | 1.69 | −8.8% | 503 | 31 | **15** |
| 1.75 | Train | 340 | 1.325 | $23,412 | 63% | 0.47 | −12.3% | 1,386 | 47 | 20 |
| 1.75 | **Test (OOS)** | **464** | **1.799** | **$59,099** | 69% | 1.70 | −8.9% | 503 | 31 | **15** |

### 3.2 OOS Delta vs Baseline

| X | OOS PnL Δ | OOS PnL Δ% | OOS PF Δ | OOS Sharpe Δ | OOS MaxDD Δ |
|---|----------:|-----------:|---------:|-------------:|------------:|
| 1.25 | +$421 | +0.73% | +0.004 | +0.01 | −$10 (worse) |
| 1.50 | +$842 | +1.46% | +0.008 | +0.02 | −$19 (worse) |
| 1.75 | +$1,262 | +2.18% | +0.012 | +0.03 | −$28 (worse) |

### 3.3 Train/Test PF Gap (overfit check)

| X | Train PF | OOS PF | Gap | Verdict |
|---|---------:|-------:|----:|--------|
| 1.00 | 1.326 | 1.787 | **−34.8%** | OOS > Train (favorable inversion, baseline-wide) |
| 1.25 | 1.327 | 1.791 | −35.0% | same |
| 1.50 | 1.324 | 1.795 | −35.6% | same |
| 1.75 | 1.325 | 1.799 | −35.8% | same |

The negative gap (OOS > Train) is a BASELINE phenomenon present at X=1.0 too — not caused by the boost. The boost adds essentially zero signal to this gap.

### 3.4 Telemetry: why so few boost triggers?

- `oi_divergence` raw fires: **1,386 in train, 503 in OOS** — plenty of raw signal
- Long entries from `{wick_trap, liquidity_sweep, funding_divergence}`: **47 train / 31 OOS** — already small
- Of those, with oi_div concurrent in 12-bar window: **20 train / 15 OOS** — about 48% of eligible entries

The bottleneck is **eligible entry count**, not concurrent firing rate. The three target archetypes (wick_trap, liquidity_sweep, funding_divergence) don't fire often enough in current production for the boost to have meaningful surface area.

---

## 4. Recommendation

**REJECTED — do not deploy.**

Reasons:
1. **n=15 OOS boosted trades** is below the n=30 directional-only floor. Any apparent lift is statistically indistinguishable from noise.
2. **OOS PnL lift is tiny**: best case (X=1.75) adds +2.18% to system PnL. Below the threshold of "worth a production config change."
3. **MaxDD slightly worsens** with higher X (a real, monotonic cost).
4. **The in-sample +$680/trade finding from Hypothesis #1 did not generalize.** Phantom-style outcome estimation overstated the lift by ~3x relative to what shows up in a real backtest with all of the production filtering applied.
5. The signal is structurally real (the concurrence pattern exists), but its TRADE-LEVEL surface area is too narrow to matter at the system level.

No production code, config, or YAML changes recommended. The existing Wyckoff 4H bearish boost (commit `5059285`) remains the only sizing modifier wired in production.

---

## 5. Sample Size & Honest Caveats

- **n=15 OOS boosted trades is the critical limit.** Even at X=2 (untested in this run), the maximum possible per-trade impact is constrained by the realized PnL on these 15 trades, not by the multiplier choice.
- **2023-2024 OOS is bull-skewed**: BTC ran from $16K to $94K. The wick_trap / liquidity_sweep / funding_divergence archetypes are all reversal/long-bias detectors that should naturally do well in a bull recovery. The 2018-2019 sideways regime and 2022 bear regime would be needed to truly stress-test the rule — but adding those expanding the window doesn't help because **the eligible entry rate is intrinsically low**.
- **Window=12 bars is the only window tested.** A sweep over K ∈ {6, 12, 24, 48} could reveal a different sweet spot but would inflate the multiple-comparison risk on n=15.
- **The boost only multiplies size; it doesn't change which trades fire.** So if the underlying trades are mixed (some +R, some −R), boosting just amplifies that mixture.

---

## 6. What This Doesn't Test

1. **Cross-class symmetric boost** — does `long_squeeze` (the other chronic loser) concurrent with these same winning archetypes produce a similar pattern? The prior Hypothesis #1 data showed it might (separate finding). Untested here.
2. **Dynamic boost** — boost only when `oi_divergence` was particularly extreme (e.g., paired with funding > 1σ). Tighter trigger could lift per-trade impact.
3. **Eligible archetype expansion** — what if the eligible set included `liquidity_compression` (currently the top live performer)? Worth a separate study.
4. **Window optimization** — fixed at K=12; might be different at K=6 or K=24.
5. **Sizing CAPS interaction** — `max_margin_per_position_pct = 0.35` may be clipping the boost on already-large positions. X=2.0 silently failed; needs investigation if any future study revives this hypothesis.

---

## 7. Proposed YAML/Config Diff

**None.** The rule did not pass validation. No production changes.

---

## 8. Files

- This report: `docs/knowledge/oi_div_sizing_boost_wfo.md`
- Shim runner: `scripts/oi_div_boost/run_variant.py` (committed prior; works as designed)
- Raw outputs: `results/oi_div_boost/X_<value>/<window>/performance_stats.json`
- Sweep driver script: `/tmp/oi_div_sweep.sh` (transient)

## 9. Constraints Honored

- READ-ONLY for production: no `engine/`, `configs/`, or YAML modifications
- Standing orders intact: no archetypes disabled, `bypass_threshold` unchanged
- Real backtest only (no phantom outcomes)
- Train 2018-2022 / Test 2023-2024 WFO split applied
- All n<30 results marked "directional only — not deployable"
- Lesson #54 honored: zero fusion-based filtering or boosting
