# Regime Soft Penalties System - Complete Guide

**Version:** 2.0
**Last Updated:** 2025-12-19
**Author:** Agent 3 (System Architect)
**Status:** Production Ready

---

## Table of Contents
1. [Overview](#overview)
2. [Architecture](#architecture)
3. [Penalty Scheme](#penalty-scheme)
4. [Configuration](#configuration)
5. [Monitoring](#monitoring)
6. [Best Practices](#best-practices)
7. [Troubleshooting](#troubleshooting)

---

## Overview

### What Are Regime Soft Penalties?

Regime soft penalties are **confidence multipliers** applied to archetype scores based on the current market regime. Unlike hard vetoes (which completely block signals), soft penalties gracefully reduce or boost confidence to improve regime-specific performance.

### Why Use Soft Penalties?

**Problem:** Without soft penalties, archetypes fire indiscriminately across all market regimes, leading to:
- Bull patterns triggering in bear markets (poor performance)
- Bear patterns triggering in bull markets (poor performance)
- Excessive archetype overlap and signal confusion

**Solution:** Soft penalties adjust confidence based on regime fit:
- **Bonuses** (>1.0x) when regime aligns with archetype nature
- **Penalties** (<1.0x) when regime conflicts with archetype nature
- **Neutral** (1.0x) in transitional markets

### System Coverage

- **Total Archetypes:** 16 (A, B, C, G, H, K, L, M, S1, S2, S3, S4, S5, S8)
- **With Soft Penalties:** 16/16 (100% coverage)
- **Hard Veto Coverage:** 16/16 (100% coverage)

---

## Architecture

### Implementation Location

```
engine/archetypes/logic_v2_adapter.py
│
├── _apply_regime_soft_penalty()  # Centralized helper method
│   ├── Input: score, context, archetype_type
│   ├── Output: (adjusted_score, multiplier, tags)
│   └── Called by: All _check_X() methods
│
└── _check_X() methods  # Individual archetype logic
    ├── Pattern detection
    ├── Base scoring
    ├── Domain engine boosts
    ├── ⚡ REGIME SOFT PENALTY (applied here)
    └── Fusion threshold gate
```

### Execution Order

```
1. Pattern Detection       → Base pattern match (gates)
2. Base Scoring           → Component score calculation
3. Archetype Weight       → Per-archetype multiplier
4. Domain Engine Boosts   → Wyckoff, SMC, Temporal, etc.
5. ⚡ REGIME SOFT PENALTY → Confidence adjustment (THIS STEP)
6. Fusion Threshold Gate  → Final qualification check
```

**Critical:** Soft penalties are applied **AFTER** domain engines but **BEFORE** the fusion threshold gate. This allows:
- Domain engines to boost weak patterns that become viable after regime bonus
- Regime penalties to filter out marginal wrong-regime signals

---

## Penalty Scheme

### Bull Archetypes (A, B, C, G, H, K, L, M)

**Direction:** Long bias
**Best Regimes:** risk_on, neutral

| Regime    | Multiplier | Rationale                                    | Example Impact        |
|-----------|------------|----------------------------------------------|-----------------------|
| `risk_on` | **1.20x**  | Bull archetypes thrive in bull markets       | Score 0.50 → 0.60     |
| `neutral` | **1.00x**  | No adjustment in transitional markets        | Score 0.50 → 0.50     |
| `risk_off`| **0.50x**  | Penalize bull patterns in bear markets       | Score 0.50 → 0.25     |
| `crisis`  | **0.30x**  | Heavy penalty - unreliable in crisis         | Score 0.50 → 0.15     |

**Why This Works:**
- Bull patterns (spring, order block retest, momentum continuation) require bullish market structure
- In bear markets, these patterns fail frequently (whipsaws, false breakouts)
- Penalties prevent bull archetypes from competing with bear archetypes in wrong regimes

### Bear Archetypes (S1, S2, S3, S4, S8)

**Direction:** Short bias or counter-trend long
**Best Regimes:** crisis, risk_off

| Regime    | Multiplier | Rationale                                    | Example Impact        |
|-----------|------------|----------------------------------------------|-----------------------|
| `crisis`  | **1.30x**  | Bear archetypes thrive in crisis             | Score 0.50 → 0.65     |
| `risk_off`| **1.20x**  | Bonus in bear markets                        | Score 0.50 → 0.60     |
| `neutral` | **1.00x**  | No adjustment in transitional markets        | Score 0.50 → 0.50     |
| `risk_on` | **0.50x**  | Penalize bear patterns in bull markets       | Score 0.50 → 0.25     |

**Why This Works:**
- Bear patterns (liquidity vacuum, failed rally, funding divergence) require bearish conditions
- In bull markets, these patterns have low win rates (market absorbs selling pressure)
- Bonuses in crisis amplify confidence when patterns are most reliable

### Contrarian Short (S5 Only)

**Direction:** Short (contrarian)
**Best Regimes:** risk_on, neutral
**Special Case:** S5 shorts overleveraged longs during bull market exhaustion

| Regime    | Multiplier | Rationale                                    | Example Impact        |
|-----------|------------|----------------------------------------------|-----------------------|
| `risk_on` | **1.20x**  | THIS IS WHERE OVERLEVERAGED LONGS FORM       | Score 0.50 → 0.60     |
| `neutral` | **1.00x**  | No adjustment                                | Score 0.50 → 0.50     |
| `risk_off`| **0.70x**  | Penalty - no overleveraged longs in bears    | Score 0.50 → 0.35     |
| `crisis`  | **0.50x**  | Heavy penalty - crisis has different dynamics| Score 0.50 → 0.25     |

**Why This Works:**
- S5 detects extreme positive funding (longs paying shorts) + overbought conditions
- These conditions ONLY occur in bull markets (risk_on)
- Bear markets don't have overleveraged longs to squeeze
- This is **counterintuitive but correct** - S5 is a bear pattern that needs bull setups

**CRITICAL FIX (2025-12-19):** Previous implementation had this BACKWARDS:
- ❌ Old: Bonus in crisis (1.25x), penalty in risk_on (0.65x)
- ✅ New: Bonus in risk_on (1.20x), penalty in crisis (0.50x)

---

## Configuration

### Default Configuration (Built-in)

Soft penalties are **always active** with the default multipliers shown above. No configuration required.

### Customizing Multipliers (Advanced)

To customize penalties for a specific archetype, modify the `_apply_regime_soft_penalty()` call in its `_check_X()` method:

```python
# Example: Custom penalties for archetype A
score, regime_penalty, regime_tags = self._apply_regime_soft_penalty(
    score, context, archetype_type='bull'
)

# Override with custom multipliers (if needed)
if archetype_id == 'A':
    # Custom regime logic for archetype A
    if current_regime == 'crisis' and special_condition:
        regime_penalty *= 0.5  # Extra crisis penalty
        score *= regime_penalty
```

### Per-Archetype Configuration (Future)

Future enhancement: Add regime penalty multipliers to archetype configs:

```json
{
  "archetypes": {
    "thresholds": {
      "spring": {
        "regime_penalties": {
          "risk_on": 1.20,
          "neutral": 1.00,
          "risk_off": 0.50,
          "crisis": 0.30
        }
      }
    }
  }
}
```

**Status:** Not yet implemented (current implementation uses centralized defaults)

### Disabling Soft Penalties (Emergency)

If soft penalties cause issues, disable by modifying `_apply_regime_soft_penalty()`:

```python
def _apply_regime_soft_penalty(self, score, context, archetype_type):
    """EMERGENCY DISABLE: Always return 1.0x multiplier"""
    return score, 1.0, ['regime_penalties_disabled']
```

**WARNING:** This removes regime discrimination and may cause performance degradation.

---

## Monitoring

### Regime Mismatch Alerts

Use `bin/monitor_regime_mismatches.py` to detect regime classifier failures:

```bash
# Analyze backtest results for regime mismatches
python3 bin/monitor_regime_mismatches.py results/backtest_2022_crisis.csv

# Save report to JSON
python3 bin/monitor_regime_mismatches.py results/backtest_2022_crisis.csv --output report.json
```

### Alert Types

| Severity   | Alert Type                      | Trigger Condition                            |
|------------|---------------------------------|----------------------------------------------|
| CRITICAL   | `bull_archetypes_in_crisis`     | Bull archetypes >50% of crisis signals       |
| CRITICAL   | `bear_archetypes_in_bull`       | Bear archetypes >50% of risk_on signals      |
| WARNING    | `bear_archetypes_low_in_crisis` | Bear archetypes <20% of crisis signals       |
| WARNING    | `s5_high_in_crisis`             | S5 >20% of crisis signals (should be low)    |
| INFO       | `s5_high_in_risk_on`            | S5 >30% of risk_on signals (expected)        |

### Expected Distributions

**2022 Crisis Period:**
- Bear archetypes: >70% of signals (target)
- Bull archetypes: <30% of signals (penalty effect)
- S5: <20% of signals (crisis penalty working)

**2024 Bull Period:**
- Bull archetypes: >70% of signals (target)
- Bear archetypes: <30% of signals (penalty effect)
- S5: 20-40% of signals (acceptable - contrarian shorts)

**2023 Neutral Period:**
- Balanced distribution: 40-60% bull, 40-60% bear
- S5: 10-30% of signals

### Metadata Tracking

Each archetype signal includes regime penalty metadata:

```python
meta = {
    "score": 0.65,                    # Final score after penalties
    "score_before_regime": 0.50,      # Score before regime penalty
    "regime_penalty": 1.30,            # Multiplier applied
    "regime_tags": ["regime_crisis_bonus"],  # Tags explaining adjustment
    "current_regime": "crisis"
}
```

Use this to analyze penalty impact:

```python
# Example: Extract penalty impact from backtest
regime_impact = df['score'] / df['score_before_regime']
avg_impact_by_regime = regime_impact.groupby(df['regime_label']).mean()
```

---

## Best Practices

### When to Use Hard Veto vs Soft Penalty

| Scenario                              | Use Hard Veto   | Use Soft Penalty |
|---------------------------------------|-----------------|------------------|
| Pattern fundamentally broken          | ✅ Yes          | ❌ No            |
| Pattern unreliable in regime          | ❌ No           | ✅ Yes           |
| Want to preserve 80%+ signals         | ❌ No           | ✅ Yes           |
| Want complete regime blocking         | ✅ Yes          | ❌ No            |
| Need granular confidence adjustment   | ❌ No           | ✅ Yes           |

**Example:**
- Hard veto: Block bull archetypes in crisis completely (allowed_regimes = ['risk_on', 'neutral'])
- Soft penalty: Reduce bull confidence by 70% in crisis (multiplier = 0.30)

**Best Approach:** Use **BOTH**:
1. Hard veto blocks extreme mismatches (>90% expected failure)
2. Soft penalty adjusts confidence for marginal cases (30-70% expected failure)

### Penalty Strength Guidelines

| Confidence Reduction | Multiplier | Use Case                                    |
|----------------------|------------|---------------------------------------------|
| 0-10%                | 0.90-1.00  | Minor regime preference                     |
| 10-30%               | 0.70-0.90  | Moderate regime mismatch                    |
| 30-50%               | 0.50-0.70  | Strong regime mismatch                      |
| 50-70%               | 0.30-0.50  | Severe regime mismatch (near hard veto)     |
| 70%+                 | <0.30      | Extreme mismatch (consider hard veto)       |

**Guideline:** Keep penalties ≥0.30 to preserve ~20% of marginal signals (avoid false negatives).

### Regime Classifier Quality

Soft penalties are only as good as the regime classifier:

**High Quality Regime Labels:**
- Smooth transitions (not flickering)
- Align with price action (crisis during crashes, risk_on during rallies)
- Stable over 5-10 bars (persistence)

**Low Quality Regime Labels:**
- Frequent transitions (every 1-2 bars)
- Contradictory labels (crisis during strong uptrend)
- Lag or lead price action excessively

**Fix:** If regime labels are poor quality, soft penalties will hurt performance. Improve regime classifier first.

---

## Troubleshooting

### Issue: Bull Archetypes Still Firing in Crisis

**Symptoms:**
- Bull archetypes >50% of crisis signals
- Crisis penalty not reducing signals

**Diagnosis:**
```bash
python3 bin/monitor_regime_mismatches.py results/backtest_2022_crisis.csv
```

**Possible Causes:**
1. Soft penalties not applied (check code injection)
2. Fusion threshold too low (penalties not enough to block signals)
3. Domain engine boosts overriding penalties
4. Regime classifier mislabeling crisis as neutral

**Fixes:**
1. Verify soft penalty code in `_check_X()` method
2. Raise fusion_threshold (e.g., 0.33 → 0.40)
3. Reduce domain boost strength in crisis
4. Improve regime classifier accuracy

### Issue: Bear Archetypes Not Firing in Crisis

**Symptoms:**
- Bear archetypes <20% of crisis signals
- Expected S1, S2, S4 patterns missing

**Diagnosis:**
```bash
# Check if crisis bars are even being processed
python3 bin/monitor_regime_mismatches.py results/backtest_2022_crisis.csv
```

**Possible Causes:**
1. Thresholds too strict for bear archetypes
2. Crisis bonus not strong enough (1.30x → consider 1.50x)
3. Hard veto blocking signals (check allowed_regimes)
4. Missing required features in crisis periods

**Fixes:**
1. Relax bear archetype thresholds (fusion_threshold, gate thresholds)
2. Increase crisis bonus (modify `_apply_regime_soft_penalty()`)
3. Review hard veto configuration (should allow crisis)
4. Check feature availability (funding, OI, liquidity)

### Issue: S5 Firing in Crisis Instead of Risk_On

**Symptoms:**
- S5 >20% of crisis signals
- S5 should fire in risk_on where overleveraged longs exist

**Diagnosis:**
```python
# Check S5 regime distribution
df_s5 = df[df['archetype_id'] == 'S5']
s5_by_regime = df_s5['regime_label'].value_counts(normalize=True)
print(s5_by_regime)
# Expected: risk_on 60-80%, neutral 10-30%, crisis <10%
```

**Possible Causes:**
1. S5 soft penalties STILL backwards (check fix applied)
2. Regime classifier mislabeling bull exhaustion as crisis
3. OI data missing (graceful degradation firing more often)

**Fixes:**
1. Verify S5 uses `archetype_type='contrarian_short'` (not 'bear')
2. Review regime classifier - bull exhaustion should be risk_on, not crisis
3. Check OI availability - add fallback logic if missing

### Issue: Penalties Too Strong - No Signals

**Symptoms:**
- Total signal count dropped >50% after adding soft penalties
- Expected profitable setups being filtered

**Diagnosis:**
```python
# Compare before/after soft penalties
signals_before = df[df['score_before_regime'] >= fusion_threshold]
signals_after = df[df['score'] >= fusion_threshold]

print(f"Signals before penalties: {len(signals_before)}")
print(f"Signals after penalties: {len(signals_after)}")
print(f"Filtered by penalties: {len(signals_before) - len(signals_after)}")
```

**Fixes:**
1. Reduce penalty strength (0.30 → 0.50, 0.50 → 0.70)
2. Lower fusion_threshold to compensate
3. Check if regime classifier is over-labeling crisis/risk_off
4. Review if penalties are being double-applied (bug)

---

## Production Readiness Checklist

- [x] S5 contradiction fixed (hard veto + soft penalties aligned)
- [x] Soft penalties implemented for all 16 archetypes
- [x] Penalty scheme consistent with hard veto boundaries
- [x] Centralized helper function `_apply_regime_soft_penalty()`
- [x] Metadata tracking (score_before_regime, regime_penalty, regime_tags)
- [x] Regime mismatch monitoring script (`bin/monitor_regime_mismatches.py`)
- [x] Unit tests passing (`bin/test_regime_soft_penalties.py`)
- [x] Documentation complete (this guide)

**Status:** ✅ Production Ready

---

## Quick Reference

### Penalty Multipliers Summary

```
Bull Archetypes (A, B, C, G, H, K, L, M):
  risk_on  : 1.20x (bonus)
  neutral  : 1.00x (no adjustment)
  risk_off : 0.50x (penalty)
  crisis   : 0.30x (heavy penalty)

Bear Archetypes (S1, S2, S3, S4, S8):
  crisis   : 1.30x (bonus)
  risk_off : 1.20x (bonus)
  neutral  : 1.00x (no adjustment)
  risk_on  : 0.50x (penalty)

Contrarian Short (S5):
  risk_on  : 1.20x (bonus - overleveraged longs)
  neutral  : 1.00x (no adjustment)
  risk_off : 0.70x (penalty)
  crisis   : 0.50x (heavy penalty)
```

### Common Commands

```bash
# Test soft penalty implementation
python3 bin/test_regime_soft_penalties.py

# Monitor regime mismatches in backtest
python3 bin/monitor_regime_mismatches.py results/backtest.csv

# Add soft penalties to new archetype (if needed)
python3 bin/add_regime_soft_penalties.py

# Check git diff to review changes
git diff engine/archetypes/logic_v2_adapter.py
```

---

## Support

**Questions?** Contact Agent 3 (System Architect)

**Bugs?** File an issue with:
- Backtest results showing mismatch
- Expected vs actual behavior
- Regime distribution analysis

**Enhancements?** Propose changes to penalty multipliers with:
- Rationale (why current multipliers don't work)
- Backtested performance comparison
- Impact on signal distribution
