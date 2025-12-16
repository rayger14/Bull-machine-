# ACCEPTANCE CRITERIA EXAMPLES
## Detailed Model Evaluation Case Studies

**Purpose:** Provide concrete examples of how to apply the 6 Rules across various scenarios, including edge cases and common failure modes.

**Last Updated:** 2025-12-07

---

## TABLE OF CONTENTS

1. [Clear Winners (KEEP Examples)](#clear-winners-keep-examples)
2. [Needs Improvement (IMPROVE Examples)](#needs-improvement-improve-examples)
3. [Kill Immediately (KILL Examples)](#kill-immediately-kill-examples)
4. [Edge Cases](#edge-cases)
5. [Common Failure Modes](#common-failure-modes)

---

## CLEAR WINNERS (KEEP EXAMPLES)

### Example 1: Ideal Archetype - S4 Funding Divergence

**Model Type:** Archetype (funding rate + OI divergence)
**Complexity:** Medium (2 primary features)

**Metrics:**
```
Train:  PF 2.45 | WR 52% | DD 12% | Trades 95
Test:   PF 2.35 | WR 51% | DD 15% | Trades 78
OOS:    PF 2.20 | WR 49% | DD 14% | Trades 68
```

**Best Baseline:**
```
Test PF: 2.12 | Max DD: 18%
```

**Rule Evaluation:**

**Rule 1: Beat Baselines**
- Test PF: 2.35 > (2.12 + 0.1 = 2.22) ✓
- Delta: 0.23 (well above buffer)
- **PASS**

**Rule 2: Low Overfit**
- Overfit: 2.45 - 2.35 = 0.10
- Interpretation: Minimal overfit (GOOD)
- **PASS**

**Rule 3: Statistical Significance**
- Total Trades: 95 + 78 + 68 = 241
- 241 >> 50
- **PASS**

**Rule 4: OOS Validation**
- OOS PF: 2.20 > 1.2 ✓
- OOS/Test: 2.20/2.35 = 0.94
- Interpretation: Excellent consistency
- **PASS**

**Rule 5: Risk-Adjusted**
- Max DD: 15% < (2 × 18% = 36%) ✓
- Actually lower than baseline!
- **PASS**

**Rule 6: Costs Included**
- Slippage: 5 bps
- Fees: 3 bps
- Total: 16 bps round-trip
- **PASS**

**Decision:** ✅ **KEEP** (6/6 rules passed)

**Why This Works:**
- Simple logic (funding + OI divergence)
- Consistent across all periods
- Low overfit (conservative tuning)
- Actually beats baseline in risk AND return

**Next Steps:**
1. Deploy to paper trading immediately
2. Monitor for 2 weeks (minimum 15 trades)
3. Compare paper vs OOS performance
4. Graduate to production at 25% size

---

### Example 2: Negative Overfit (Better on Test) - S2 SMC Break of Structure

**Model Type:** Archetype (SMC + volume)
**Complexity:** Medium

**Metrics:**
```
Train:  PF 2.15 | WR 48% | DD 16% | Trades 88
Test:   PF 2.28 | WR 50% | DD 14% | Trades 75
OOS:    PF 2.10 | WR 48% | DD 15% | Trades 71
```

**Best Baseline:**
```
Test PF: 2.12 | Max DD: 18%
```

**Rule Evaluation:**

**Rule 1: Beat Baselines**
- Test PF: 2.28 > 2.22 ✓
- Delta: 0.16
- **PASS**

**Rule 2: Low Overfit**
- Overfit: 2.15 - 2.28 = **-0.13**
- Interpretation: NEGATIVE overfit (EXCELLENT!)
- Model actually performs BETTER on unseen test data
- **PASS**

**Rule 3: Statistical Significance**
- Total: 234 trades
- **PASS**

**Rule 4: OOS Validation**
- OOS PF: 2.10 > 1.2 ✓
- OOS/Test: 2.10/2.28 = 0.92
- **PASS**

**Rule 5: Risk-Adjusted**
- Max DD: 14% < 36% ✓
- **PASS**

**Rule 6: Costs Included**
- Yes, 16 bps round-trip
- **PASS**

**Decision:** ✅ **KEEP** (6/6 rules passed)

**Why Negative Overfit is GOOD:**
- Indicates conservative parameter selection
- Wasn't optimized to fit training noise
- Logic generalizes well
- Random variation made test slightly better (luck)

**Key Insight:** Negative overfit is a GREEN FLAG, not a red flag. It suggests the model wasn't curve-fitted.

---

### Example 3: High PF Compensates for Higher Risk - S3 Order Block Retest

**Model Type:** Archetype (order block + Fibonacci)
**Complexity:** Medium-High

**Metrics:**
```
Train:  PF 3.80 | WR 58% | DD 22% | Trades 102
Test:   PF 3.50 | WR 56% | DD 25% | Trades 85
OOS:    PF 3.20 | WR 54% | DD 24% | Trades 78
```

**Best Baseline:**
```
Test PF: 2.12 | Max DD: 18%
```

**Rule Evaluation:**

**Rule 1: Beat Baselines**
- Test PF: 3.50 > 2.22 ✓
- Delta: 1.38 (massive improvement!)
- **PASS**

**Rule 2: Low Overfit**
- Overfit: 3.80 - 3.50 = 0.30
- Interpretation: Moderate overfit (ACCEPTABLE)
- **PASS**

**Rule 3: Statistical Significance**
- Total: 265 trades
- **PASS**

**Rule 4: OOS Validation**
- OOS PF: 3.20 > 1.2 ✓
- OOS/Test: 3.20/3.50 = 0.91
- **PASS**

**Rule 5: Risk-Adjusted**
- Max DD: 25% > (2 × 18% = 36%)? NO, 25% < 36% ✓
- BUT, even if DD was higher, PF 3.5 > 3.0 would compensate
- **PASS**

**Rule 6: Costs Included**
- Yes
- **PASS**

**Decision:** ✅ **KEEP** (6/6 rules passed)

**Why This Works:**
- High PF (3.5) justifies slightly higher risk
- Still within 2x baseline DD limit
- Overfit is moderate but acceptable given performance
- Consistent degradation (not collapse) from train → test → OOS

**Monitoring Note:** Higher DD means tighter position sizing in production. Start at 15% size instead of 25%.

---

## NEEDS IMPROVEMENT (IMPROVE EXAMPLES)

### Example 4: High Overfit - S1 Liquidity Vacuum

**Model Type:** Archetype (order book + liquidity scoring)
**Complexity:** High (15+ features)

**Metrics:**
```
Train:  PF 3.20 | WR 62% | DD 10% | Trades 92
Test:   PF 1.80 | WR 48% | DD 18% | Trades 68
OOS:    PF 1.50 | WR 45% | DD 20% | Trades 63
```

**Best Baseline:**
```
Test PF: 2.12 | Max DD: 18%
```

**Rule Evaluation:**

**Rule 1: Beat Baselines**
- Test PF: 1.80 < 2.22 ✗
- WORSE than baseline!
- **FAIL**

**Rule 2: Low Overfit**
- Overfit: 3.20 - 1.80 = **1.40**
- Interpretation: HIGH overfit (curve-fitted)
- **FAIL**

**Rule 3: Statistical Significance**
- Total: 223 trades
- **PASS**

**Rule 4: OOS Validation**
- OOS PF: 1.50 > 1.2 ✓
- OOS/Test: 1.50/1.80 = 0.83
- **PASS**

**Rule 5: Risk-Adjusted**
- Max DD: 18% < 36% ✓
- **PASS**

**Rule 6: Costs Included**
- Yes
- **PASS**

**Score:** 4/6 rules passed

**Decision:** 🔧 **IMPROVE**

**Diagnosis:**
1. **Overfit Issue:** Train PF 3.2 → Test PF 1.8 is massive degradation
2. **Too Many Features:** 15+ features = too many degrees of freedom
3. **Below Baseline:** Simple baseline beats this complex model

**Remediation Plan:**
1. **Reduce Features:** Use feature importance analysis, keep only top 5
2. **Simplify Logic:** Remove complex confluence requirements
3. **Add Regularization:** If using ML, add L1/L2 penalties
4. **Re-optimize Conservatively:** Use walk-forward instead of single train period
5. **Re-test:** Must show Test PF > 2.22 and Overfit < 0.5

**Re-test Deadline:** 2025-12-21 (2 weeks)

**If re-test fails:** KILL and try simpler approach

---

### Example 5: Low Sample Size - Macro Crisis Detector

**Model Type:** Archetype (volatility spike + correlation breakdown)
**Complexity:** Medium

**Metrics:**
```
Train:  PF 2.80 | WR 75% | DD 8% | Trades 12
Test:   PF 2.50 | WR 71% | DD 10% | Trades 7
OOS:    PF 2.30 | WR 67% | DD 12% | Trades 6
```

**Best Baseline:**
```
Test PF: 2.12 | Max DD: 18%
```

**Rule Evaluation:**

**Rule 1: Beat Baselines**
- Test PF: 2.50 > 2.22 ✓
- **PASS**

**Rule 2: Low Overfit**
- Overfit: 2.80 - 2.50 = 0.30
- **PASS**

**Rule 3: Statistical Significance**
- Total: 12 + 7 + 6 = 25 < 50 ✗
- **FAIL** (unless exception applies)

**Rule 4: OOS Validation**
- OOS PF: 2.30 > 1.2 ✓
- OOS/Test: 2.30/2.50 = 0.92 ✓
- **PASS**

**Rule 5: Risk-Adjusted**
- Max DD: 12% < 36% ✓
- **PASS**

**Rule 6: Costs Included**
- Yes
- **PASS**

**Score:** 5/6 rules passed (Rule 3 failed)

**Decision:** 🔧 **IMPROVE** (but could PASS with exception)

**Path 1: Multi-Asset Validation**
Run same logic on BTC, ETH, SOL:
```
BTC: 25 trades, PF 2.30
ETH: 18 trades, PF 2.15
SOL: 22 trades, PF 2.45
Total: 65 effective trades across 3 independent markets
```
If this works → Tag as "low-freq/macro" → **PASS with exception**

**Path 2: Walk-Forward Validation**
Divide history into 5 regimes:
```
Bull 2020: 4 trades, PF 2.1
Bear 2022: 8 trades, PF 2.8
Bull 2023: 5 trades, PF 2.3
Sideways 2024: 6 trades, PF 1.9
Crisis periods: 2 trades, PF 3.5
```
Show consistency across regimes → **PASS with exception**

**Path 3: Kill It**
If neither multi-asset nor walk-forward works, 25 trades is just too low.

**Recommended:** Try Path 1 first (easiest to implement)

---

### Example 6: OOS Collapse - H Trap Within Trend

**Model Type:** Archetype (Wyckoff + trend + trap logic)
**Complexity:** High

**Metrics:**
```
Train:  PF 2.60 | WR 54% | DD 14% | Trades 102
Test:   PF 2.40 | WR 52% | DD 16% | Trades 88
OOS:    PF 1.05 | WR 39% | DD 28% | Trades 75
```

**Best Baseline:**
```
Test PF: 2.12 | Max DD: 18%
```

**Rule Evaluation:**

**Rule 1: Beat Baselines**
- Test PF: 2.40 > 2.22 ✓
- **PASS**

**Rule 2: Low Overfit**
- Overfit (Train-Test): 2.60 - 2.40 = 0.20
- **PASS**

**Rule 3: Statistical Significance**
- Total: 265 trades
- **PASS**

**Rule 4: OOS Validation**
- OOS PF: 1.05 < 1.2 ✗
- OOS/Test: 1.05/2.40 = 0.44 ✗ (catastrophic degradation!)
- **FAIL**

**Rule 5: Risk-Adjusted**
- Max DD: 28% < 36% ✓ (barely)
- **PASS**

**Rule 6: Costs Included**
- Yes
- **PASS**

**Score:** 5/6 rules passed (Rule 4 failed badly)

**Decision:** 🔧 **IMPROVE** (but close to KILL)

**Diagnosis:**
- Test looks great (PF 2.4)
- OOS catastrophically collapses (PF 1.05)
- This screams: regime shift or data leakage

**Investigation Required:**
1. **Check for Data Leakage:**
   - Are any features using future data?
   - Is Wyckoff logic peeking ahead?
   - Are order book features realistic?

2. **Regime Analysis:**
   - What regime was Test period? (2022 bear)
   - What regime was OOS period? (2023 bull)
   - Does strategy only work in bear markets?

3. **Feature Validation:**
   - Which features work in OOS vs not?
   - Can we identify what broke?

**Possible Fixes:**
- **If Regime-Specific:** Deploy only in bear markets (add regime filter)
- **If Data Leakage:** Fix the leak, re-validate from scratch
- **If Broken Logic:** Simplify or redesign

**Re-test Deadline:** 2 weeks

**If can't fix:** KILL (OOS PF 1.05 is barely breaking even)

---

## KILL IMMEDIATELY (KILL EXAMPLES)

### Example 7: Worse Than Baseline Everywhere - S5 Long Squeeze

**Model Type:** Archetype (funding + OI squeeze detection)
**Complexity:** Medium

**Metrics:**
```
Train:  PF 1.85 | WR 44% | DD 20% | Trades 45
Test:   PF 1.30 | WR 38% | DD 25% | Trades 38
OOS:    PF 0.90 | WR 32% | DD 32% | Trades 35
```

**Best Baseline:**
```
Test PF: 2.12 | Max DD: 18%
```

**Rule Evaluation:**

**Rule 1: Beat Baselines**
- Test PF: 1.30 << 2.22 ✗
- FAR below baseline
- **FAIL**

**Rule 2: Low Overfit**
- Overfit: 1.85 - 1.30 = 0.55 ✗
- **FAIL**

**Rule 3: Statistical Significance**
- Total: 118 trades
- **PASS**

**Rule 4: OOS Validation**
- OOS PF: 0.90 < 1.2 ✗ (LOSING MONEY!)
- OOS/Test: 0.90/1.30 = 0.69 ✓ (ratio OK but absolute PF terrible)
- **FAIL**

**Rule 5: Risk-Adjusted**
- Max DD: 25% < 36% ✓
- **PASS**

**Rule 6: Costs Included**
- Yes
- **PASS**

**Score:** 3/6 rules passed

**Decision:** ❌ **KILL**

**Why Kill:**
1. Worse than simple baseline on EVERY metric
2. OOS PF 0.90 means LOSING MONEY after costs
3. High overfit suggests it barely worked even on train
4. Simple "buy dips" strategy crushes this

**Post-Mortem:**
- "Long squeeze" signal doesn't reliably predict reversals
- Funding + OI combination needs complete redesign
- OR this edge doesn't exist at all

**Lessons Learned:**
1. Just because "squeeze" sounds good doesn't mean it works
2. Need to validate signal on simple backtests BEFORE building complex archetypes
3. Sometimes the edge just isn't there

**Salvageable Components:**
- None. Archive and move on.

**Time Saved:** Don't waste 2 more weeks trying to fix this. Kill now.

---

### Example 8: Extreme Overfit - ML Random Forest v1

**Model Type:** ML Model (Random Forest, 50+ features)
**Complexity:** Very High

**Metrics:**
```
Train:  PF 8.50 | WR 78% | DD 5% | Trades 220
Test:   PF 2.10 | WR 51% | DD 22% | Trades 198
OOS:    PF 1.60 | WR 47% | DD 28% | Trades 185
```

**Best Baseline:**
```
Test PF: 2.12 | Max DD: 18%
```

**Rule Evaluation:**

**Rule 1: Beat Baselines**
- Test PF: 2.10 < 2.22 ✗
- Barely below, but still fails
- **FAIL**

**Rule 2: Low Overfit**
- Overfit: 8.50 - 2.10 = **6.40** ✗
- EXTREME overfit (>>1.5)
- **KILL IMMEDIATELY**

**Rule 3-6:** Don't even bother checking

**Score:** 0/6 (auto-fail on Rule 2)

**Decision:** ❌ **KILL**

**Why Kill:**
- Train PF 8.5 vs Test PF 2.1 is absurd
- 75% degradation from train to OOS
- Model memorized training data noise
- Classic ML overfitting

**Post-Mortem:**
1. 50+ features with Random Forest = recipe for overfitting
2. No regularization or pruning
3. Probably fitting to random correlations in training data
4. "Great" train performance was a trap

**Lessons Learned:**
1. More features ≠ better model
2. Complex ML needs MORE validation, not less
3. If train PF >> test PF, it's overfitted (period)
4. Should have used feature selection BEFORE training

**Next Steps:**
1. Try simpler model (logistic regression)
2. Reduce to top 5-10 features only
3. Add cross-validation during training
4. Use walk-forward optimization
5. Re-validate from scratch

**Don't try to fix this version.** Start fresh with lessons learned.

---

### Example 9: Data Leakage Detected - Liquidity Score Bug

**Model Type:** Archetype (liquidity + order flow)
**Complexity:** Medium

**Metrics:**
```
Backtest: PF 3.80 | WR 65% | DD 8%
Paper Trading: PF 1.10 | WR 42% | DD 25%
Deviation: -71% PF collapse
```

**Rule Evaluation:**
- Passed all 6 rules in backtest
- CATASTROPHIC failure in paper trading

**Decision:** ❌ **KILL** (data leakage)

**Root Cause:**
```python
# WRONG (has lookahead bias):
def calculate_liquidity_score(df):
    # Uses order book data from END of bar
    # In backtest: Had access to future fills
    # In live: Only has current order book
    return df['liquidity_at_close']

# CORRECT:
def calculate_liquidity_score(df):
    # Uses order book data from START of bar
    # Only uses information available at entry time
    return df['liquidity_at_open']
```

**Why Kill:**
1. Backtest results are INVALID (used future data)
2. Need to re-backtest with fixed logic
3. Can't trust any of the original metrics

**Post-Mortem:**
- Liquidity feature was using close-of-bar data for entry decisions
- In backtest: Had access to future order book state
- In live: Only had current order book
- This inflated backtest performance by ~70%

**Lessons Learned:**
1. ALWAYS validate features for lookahead bias
2. Paper trading is critical (catches data leaks)
3. If paper PF << backtest PF, suspect data leakage
4. Order book features are HIGH RISK for this bug

**Fix Required:**
1. Rewrite liquidity scoring (only use available data)
2. Re-run FULL validation (train/test/OOS)
3. Re-evaluate against 6 rules
4. If still passes → new paper test

**Don't deploy the broken version.** Fix first.

---

## EDGE CASES

### Example 10: Regime-Specific Strategy (Bear Only)

**Model Type:** Archetype - Bear Market Specialist
**Complexity:** Medium

**Metrics (Full History):**
```
Train:  PF 1.65 | WR 45% | DD 18%
Test:   PF 1.45 | WR 42% | DD 22%
OOS:    PF 1.30 | WR 40% | DD 25%
```

**Fails Rule 1:** Test PF 1.45 < 2.22 ✗

**BUT... in Bear Regime Only:**
```
Train (Bear): PF 2.90 | WR 58% | DD 12% | Trades 45
Test (Bear):  PF 2.85 | WR 56% | DD 15% | Trades 38
OOS (Bear):   PF 2.70 | WR 55% | DD 14% | Trades 32
```

**Performance in Bull/Neutral:**
```
Bull:    PF 1.05 | WR 38% | DD 8%  (near break-even, OK)
Neutral: PF 1.10 | WR 40% | DD 6%  (small profit, OK)
```

**Regime-Specific Evaluation:**

**Within Bear Regime:**
- Rule 1: 2.85 > 2.22 ✓ (beats baseline in bear)
- Rule 2: Overfit 0.05 ✓
- Rule 3: 115 trades total ✓
- Rule 4: OOS/Test = 0.95 ✓
- Rule 5: DD 15% < 36% ✓
- Rule 6: Costs included ✓

**Outside Bear Regime:**
- Not losing money (PF > 1.0)
- Low DD (< 10%)
- Acceptable to go flat

**Decision:** ✅ **KEEP** (with regime filter)

**Deployment Strategy:**
1. Only activate when GMM regime = "bear"
2. Go flat (or use different strategy) in bull/neutral
3. Monitor regime classifier accuracy
4. Set alerts for regime transitions

**Why This Works:**
- Strategy is DESIGNED for bear markets
- Excellent performance in target regime
- Doesn't lose money in other regimes
- Regime classifier is validated (95% OOS accuracy)

---

### Example 11: Low-Frequency with Multi-Asset Validation

**Model Type:** Macro Strategy (crisis alpha)
**Complexity:** Medium

**Single Asset (BTC):**
```
Total Trades: 18 (FAILS Rule 3)
Train: PF 2.50 | Trades 6
Test:  PF 2.30 | Trades 7
OOS:   PF 2.20 | Trades 5
```

**Multi-Asset Validation:**
```
BTC: 18 trades | PF 2.20
ETH: 22 trades | PF 2.10
SOL: 16 trades | PF 2.35
Total: 56 trades (PASSES Rule 3 with exception)
```

**Portfolio Performance:**
```
Train:  PF 2.55 | WR 68% | DD 10%
Test:   PF 2.40 | WR 65% | DD 12%
OOS:    PF 2.25 | WR 63% | DD 11%
```

**Rule Evaluation:**
- Rule 1: 2.40 > 2.22 ✓
- Rule 2: Overfit 0.15 ✓
- Rule 3: 56 trades (multi-asset) ✓ (with exception)
- Rule 4: OOS 2.25, ratio 0.94 ✓
- Rule 5: DD 12% < 36% ✓
- Rule 6: Costs included ✓

**Special Conditions:**
- Tagged as "low-freq/macro"
- Strategy logic: Detects vol spikes > 3 sigma (inherently rare)
- Works independently on 3 uncorrelated assets
- Effective sample: 56 independent events

**Decision:** ✅ **KEEP** (with low-freq exception)

**Why Exception Applies:**
1. Strategy is INHERENTLY low-frequency (crisis detection)
2. Works on multiple independent markets (BTC/ETH/SOL)
3. Each asset shows positive PF
4. Total sample (56) exceeds threshold when combined

**Monitoring:**
- Track performance on each asset separately
- If one asset starts failing, investigate
- Quarterly review of crisis detection logic

---

## COMMON FAILURE MODES

### Failure Mode 1: "Great Train, Terrible Test" (Overfit)

**Symptoms:**
- Train PF >> Test PF (difference > 0.5)
- Train WR >> Test WR
- Test performance barely beats baseline (or doesn't)

**Causes:**
- Too many parameters
- Over-optimization on training period
- Fitting to noise, not signal
- Using in-sample data for parameter selection

**Fixes:**
1. Reduce parameters (simplify model)
2. Add regularization
3. Use walk-forward optimization
4. Increase training period length
5. Use ensemble methods (average across parameter sets)

**Prevention:**
- Set parameter bounds BEFORE seeing data
- Use economic logic to guide parameter selection
- Cross-validate during optimization
- Monitor Train vs Test gap throughout development

---

### Failure Mode 2: "Test Great, OOS Collapse" (Regime Shift)

**Symptoms:**
- Test PF looks good
- OOS PF << Test PF (ratio < 0.6)
- Often coincides with market regime change

**Causes:**
- Strategy works in specific regime (bear/bull) only
- Test period not representative of OOS period
- Market microstructure changed
- Data distribution shifted

**Fixes:**
1. **If Regime-Specific:** Add regime filter, only trade in target regime
2. **If Microstructure:** Recalibrate for new market conditions
3. **If Fundamental Shift:** May need to KILL (edge disappeared)

**Prevention:**
- Test across multiple regimes
- Use walk-forward validation
- Monitor regime characteristics of train/test/OOS
- Build regime-aware strategies from start

---

### Failure Mode 3: "Backtest Great, Paper Terrible" (Data Leakage)

**Symptoms:**
- Backtest passes all 6 rules
- Paper trading PF << Backtest OOS PF (> 30% deviation)
- Features behave differently in live vs backtest

**Causes:**
- Lookahead bias (using future data)
- Survivorship bias (only backtesting on current assets)
- Order book features using close-of-bar data
- Rebalancing on not-yet-available data

**Fixes:**
1. **Audit ALL features** for lookahead bias
2. **Fix data leakage**, re-validate from scratch
3. **Re-run paper trading** with fixed logic

**Prevention:**
- Code review for temporal logic
- Use strict "available data" checks
- Validate order book features carefully
- Paper trade BEFORE production (always)

---

### Failure Mode 4: "Low Sample Size Luck" (Statistical Noise)

**Symptoms:**
- Total trades < 50
- High PF but low trade count
- 1-2 lucky/unlucky trades dominate results
- High variance in PF across folds

**Causes:**
- Strategy too selective (overly strict filters)
- Low-frequency by design (macro)
- Short backtest period
- Rare event strategy

**Fixes:**
1. **If Too Selective:** Relax filters, reduce confluence requirements
2. **If Low-Freq:** Apply multi-asset or walk-forward validation
3. **If Short Period:** Extend backtest history
4. **If Rare Event:** Tag as low-freq, require multi-asset validation

**Prevention:**
- Aim for 100+ trades in backtest (not just 50)
- Use bootstrap resampling to assess variance
- Multi-asset validation for low-freq strategies
- Longer backtest periods

---

### Failure Mode 5: "Complexity Theater" (Doesn't Beat Simple Baseline)

**Symptoms:**
- Complex model (10+ features, ML, ensemble)
- Test PF ≈ Simple baseline PF
- High development/maintenance cost
- Marginal improvement (< 0.1 PF)

**Causes:**
- Complexity doesn't add value
- Simple edge is sufficient
- Over-engineering the problem

**Fixes:**
1. **KILL complex version**
2. **Use simple baseline instead**
3. **Save development time**

**Prevention:**
- Always build simple baseline FIRST
- Complexity must justify itself (>0.1 PF improvement)
- Prefer simple over complex (Occam's Razor)
- Maintenance cost matters

**Philosophy:** If a simple "buy dips" strategy works as well as your 50-feature ML model, use the simple strategy.

---

## SUMMARY DECISION TREE

```
Start
  │
  ▼
Does it beat baseline + 0.1?
  ├─ No → KILL (or IMPROVE if close)
  │
  ▼ Yes
Is overfit < 0.5?
  ├─ No → KILL (if >1.5) or IMPROVE
  │
  ▼ Yes
Does it have 50+ trades OR qualify for exception?
  ├─ No → IMPROVE (add multi-asset or walk-forward)
  │
  ▼ Yes
Is OOS PF > 1.2 AND ratio > 0.6?
  ├─ No → IMPROVE (if fixable) or KILL
  │
  ▼ Yes
Is risk acceptable (DD < 2x baseline OR PF > 3.0)?
  ├─ No → IMPROVE (add risk controls) or KILL
  │
  ▼ Yes
Are costs included (16 bps)?
  ├─ No → Re-run with costs
  │
  ▼ Yes
✅ KEEP → Deploy to paper trading
```

---

## FINAL CHECKLIST

Before making ANY decision, verify:

1. [ ] All 6 rules evaluated explicitly (no skipping)
2. [ ] Baseline comparison done (apples-to-apples)
3. [ ] Overfit score calculated (Train - Test)
4. [ ] Sample size checked (50+ or exception documented)
5. [ ] OOS validated (PF > 1.2, ratio > 0.6)
6. [ ] Risk assessed (DD within limits)
7. [ ] Costs verified (16 bps minimum)
8. [ ] Decision documented (KEEP/IMPROVE/KILL)
9. [ ] If IMPROVE: Specific action plan created
10. [ ] If KILL: Lessons learned documented

**No shortcuts. No self-deception. No exceptions without documentation.**

---

**"The goal is not to have the most complex model.**
**The goal is to have the model that makes the most money."**
**— Bull Machine Lab**
