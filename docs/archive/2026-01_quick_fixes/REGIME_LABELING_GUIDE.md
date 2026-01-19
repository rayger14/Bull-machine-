# Regime Labeling Guide for Supervised Learning

## Overview

This guide helps you manually label crisis periods for training a supervised regime classifier. The classifier will use crisis features + macro features to automatically classify market regimes in real-time.

**Time Commitment:** ~10 hours total
- 5 major crisis events: ~5 hours
- Surrounding context (optional): ~5 hours

**Why This Matters:**
- Supervised learning outperforms HMMs for event-driven crypto markets
- Your labels will train a model that detects future crises automatically
- Target: >70% crisis detection with <20% false positives

---

## 1. Regime Definitions

### 🚨 Crisis
**Definition:** Extreme market stress with systemic risk, rapid price collapse, and mass liquidations.

**When to Label as Crisis:**
- Multiple crisis indicators triggered simultaneously (crisis_composite_score ≥ 3)
- Flash crashes: >4% drop in 1H, >8% in 4H, or >12% in 1D
- Volume spikes: >3 sigma above normal
- Funding extremes: Absolute z-score >3.0
- Market structure breakdown (liquidity vacuum, OI cascades)

**Examples:**
- LUNA collapse (May 9-12, 2022): -60% in 3 days, Terra ecosystem implosion
- FTX collapse (Nov 8-11, 2022): -20% in 24H, exchange bankruptcy
- June 2022 dump (June 13-18): -40% in 5 days, cascading liquidations

**What NOT to Label as Crisis:**
- Normal corrections (<10% over multiple days)
- Single indicator triggers (e.g., just a volume spike)
- Slow grinds lower without panic

---

### ⚠️ Risk-Off
**Definition:** Elevated volatility and risk aversion, but not systemic crisis.

**When to Label as Risk-Off:**
- Crisis score = 2 (some indicators triggered, but not confluence)
- VIX z-score >1.5 (elevated fear)
- Realized volatility >5% (RV_20 >0.05)
- Sharp but controlled selloffs (-5% to -15%)
- Single flash crash or volume spike (no cascade)

**Examples:**
- Days immediately before/after crisis events
- Macro shocks (Fed hawkishness, geopolitical events)
- Technical breakdowns without cascades

**Difference from Crisis:**
- Risk-off = controlled selloff, crisis = panic liquidation cascade
- Risk-off = 1-2 indicators, crisis = 3+ indicators

---

### 🟢 Risk-On
**Definition:** Low volatility, bullish sentiment, greed phase.

**When to Label as Risk-On:**
- VIX z-score <-0.5 (low fear)
- Realized volatility <3% (RV_20 <0.03)
- No crisis indicators triggered
- Stable uptrends or consolidation at highs
- Positive funding rates (long bias)

**Examples:**
- Q1 2023 recovery (post-FTX, pre-banking crisis)
- Late 2023 bull run (ETF anticipation)
- Any period with sustained low volatility + positive price action

---

### ⚪ Neutral
**Definition:** Normal market conditions, no extreme fear or greed.

**When to Label as Neutral:**
- VIX z-score between -0.5 and +1.5
- Realized volatility 3-5% (normal range)
- No crisis indicators
- Rangebound or slow trends
- Mixed signals from macro

**Examples:**
- Most of 2023 (between crisis events)
- Consolidation periods after rallies
- Boring sideways chop

**Default Choice:**
When in doubt, label as neutral. Better to be conservative.

---

## 2. Labeling Interface Usage

### Step-by-Step Workflow

**1. Launch the Labeling Tool:**
```bash
python bin/label_crisis_periods.py
```

**2. Main Menu Options:**
- `[1]` Label crisis events - Start here, label 5 major events
- `[2]` Label surrounding context - Add context after events (optional)
- `[3]` Show progress - Check how many hours labeled
- `[4]` Save and exit - Save labels to CSV
- `[5]` Exit without saving - Discard changes

**3. Labeling Modes (for each event):**
- `[1]` Label entire event as one regime - Fast, good for clear-cut events
- `[2]` Label hour-by-hour - Granular, better for mixed signals
- `[3]` Auto-label based on crisis scores - Uses crisis_composite_score as guide

**4. Keyboard Shortcuts (hour-by-hour mode):**
- `c` = crisis
- `r` = risk_off
- `n` = neutral
- `o` = risk_on
- `a` = auto (use suggested label)
- `s` = skip this hour
- `q` = quit and save

---

## 3. What to Look For

### Crisis Indicators Display

The tool shows you these key metrics for each hour:

```
📊 Price Context:
   Current: $20,830.53
   Change (24h before → now): -15.2%
   Change (now → 24h after): -8.1%

🚨 Crisis Indicators (current hour):
   Composite Score: 4
   Flash Crash 1H:  1
   Flash Crash 4H:  1
   Flash Crash 1D:  1
   Volume Spike:    1
   OI Cascade:      0
   Funding Extreme: 0

🌍 Macro Context:
   VIX (z-score):   2.35
   DXY (z-score):   0.82
   Funding (z):     -1.42
   RV 20d:          0.0687

💡 Suggested Label: CRISIS (crisis_score=4 (HIGH))
```

### Decision Tree for Labeling

**Start here:**
1. **Check crisis_composite_score:**
   - ≥3 → Likely crisis (check price action to confirm)
   - =2 → Likely risk_off (elevated risk, not full panic)
   - ≤1 → Check macro context

2. **If composite score ≤1, check macro:**
   - VIX_Z >1.5 OR RV_20 >0.05 → risk_off
   - VIX_Z <-0.5 → risk_on
   - Otherwise → neutral

3. **Validate with price action:**
   - Crisis: Sharp drop (>5% in hours), continued weakness
   - Risk-off: Moderate drop (2-5%), controlled
   - Risk-on: Stable or rising
   - Neutral: Sideways chop

4. **Trust your judgment:**
   - The tool suggests, but you decide
   - If it feels like panic, label as crisis
   - If it feels controlled, label as risk_off

---

## 4. Example Labeled Data

### Example 1: LUNA Collapse (May 9, 2022, 08:00 UTC)

**Context:**
- Terra/LUNA stablecoin de-peg triggered cascade
- BTC dropped from $34,400 to $29,300 in 24H (-14.8%)

**Metrics:**
- Crisis composite score: 4
- Flash crash 1D: 1 (>12% drop)
- Volume spike: 1 (panic selling)
- VIX_Z: 2.1 (high fear)

**Label: CRISIS** ✅

**Reasoning:**
- Multiple indicators triggered (score=4)
- Extreme price drop with panic
- Systemic risk (Terra ecosystem collapse)

---

### Example 2: Post-FTX (Nov 12, 2022, 14:00 UTC)

**Context:**
- 4 days after FTX bankruptcy
- Market still stressed but panic subsiding
- BTC at $16,800 (down from $21,000)

**Metrics:**
- Crisis composite score: 1
- Flash crash 1D: 0 (no new crash)
- Volume spike: 0 (normalizing)
- VIX_Z: 1.8 (still elevated)
- RV_20: 0.058 (high volatility)

**Label: RISK_OFF** ✅

**Reasoning:**
- Crisis passed, but fear remains high
- No active cascade, but elevated volatility
- Not back to normal yet

---

### Example 3: March 2023 Recovery (March 25, 2023, 10:00 UTC)

**Context:**
- Post-banking crisis, market stabilizing
- BTC at $28,000 (recovering from $20,000)

**Metrics:**
- Crisis composite score: 0
- VIX_Z: -0.3 (normalizing)
- RV_20: 0.038 (moderate volatility)
- Funding_Z: 0.5 (slight long bias)

**Label: NEUTRAL** ✅

**Reasoning:**
- No crisis signals
- VIX not extreme in either direction
- Normal volatility, steady recovery

---

### Example 4: Aug 2024 Carry Unwind (Aug 5, 2024, 04:00 UTC)

**Context:**
- BoJ rate hike triggered global carry unwind
- BTC dropped from $62,000 to $54,000 in hours (-13%)

**Metrics:**
- Crisis composite score: 3
- Flash crash 1D: 1
- Volume spike: 1
- VIX_Z: 2.5 (elevated)

**Label: CRISIS** ✅

**Reasoning:**
- Rapid drawdown with global contagion
- Multiple indicators triggered
- Not crypto-specific, but systemic macro shock

---

## 5. Best Practices

### DO:
✅ **Label conservatively** - When in doubt, use neutral or risk_off (not crisis)
✅ **Use auto-label as starting point** - Crisis_composite_score is a good guide
✅ **Look at price context** - 24H before/after shows the full picture
✅ **Save frequently** - Tool auto-saves after each event
✅ **Take breaks** - Labeling for hours is tedious, break it up
✅ **Trust the data** - If crisis_score ≥3 and price dropped hard, it's a crisis

### DON'T:
❌ **Over-label as crisis** - Reserve for true panic events (score ≥3)
❌ **Ignore price action** - Crisis score alone isn't enough, check price
❌ **Rush through events** - Take time to understand each period
❌ **Label based on hindsight** - Use only info available at that hour
❌ **Skip surrounding context** - Context helps model learn regime transitions

---

## 6. Time Estimates

### Phase 1: Label 5 Major Crisis Events (~5 hours)

**Recommended approach:** Auto-label + manual review

1. **LUNA Collapse** (May 9-12, 2022)
   - 96 hours total
   - Auto-label, review outliers
   - Estimated time: 1 hour

2. **FTX Collapse** (Nov 8-11, 2022)
   - 96 hours total
   - Auto-label, review outliers
   - Estimated time: 1 hour

3. **June 2022 Dump** (June 13-18, 2022)
   - 144 hours total
   - Auto-label, review outliers
   - Estimated time: 1.5 hours

4. **March 2023 Banking Crisis** (March 10-13, 2023)
   - 96 hours total
   - Less clear-cut (BTC rallied), manual review needed
   - Estimated time: 1 hour

5. **Aug 2024 Carry Unwind** (Aug 5-8, 2024)
   - 96 hours total
   - Auto-label, review outliers
   - Estimated time: 0.5 hours

**Total: ~5 hours**

---

### Phase 2: Label Surrounding Context (~5 hours, OPTIONAL)

**Purpose:** Help model learn regime transitions (crisis → risk_off → neutral)

**Approach:**
- Label 1 week before each event (168 hours × 5 = 840 hours)
- Label 1 week after each event (168 hours × 5 = 840 hours)
- Use auto-label for most, manual review for transitions

**Estimated time:** ~5 hours (mostly auto-labeling)

---

## 7. Output Format

Labels are saved to: `data/regime_labels/crisis_labels_manual.csv`

**Format:**
```csv
timestamp,regime_label,confidence,labeled_at
2022-05-09 00:00:00+00:00,crisis,high,2024-12-19T10:30:00
2022-05-09 01:00:00+00:00,crisis,high,2024-12-19T10:30:00
2022-05-09 02:00:00+00:00,crisis,auto,2024-12-19T10:30:00
...
```

**Columns:**
- `timestamp`: Hour timestamp (UTC)
- `regime_label`: crisis, risk_off, neutral, or risk_on
- `confidence`: high (manual), auto (auto-labeled)
- `labeled_at`: When label was created

---

## 8. After Labeling

Once you've labeled the data, train the model:

```bash
# Train with default parameters (fast)
python bin/train_regime_classifier.py

# Train with hyperparameter optimization (slow, better performance)
python bin/train_regime_classifier.py --optimize --n-trials 100
```

**Training time:**
- Default: ~5 minutes
- With optimization: ~30-60 minutes

**Evaluate the model:**
```bash
# Evaluate on test set (2024 data)
python bin/evaluate_regime_classifier.py --model ensemble

# Validate on Aug 2024 OOS
python bin/validate_regime_classifier_oos.py --model ensemble
```

---

## 9. Target Performance Metrics

After training, the model should achieve:

**Overall:**
- Test accuracy: >0.75 (75%+)
- F1 score (weighted): >0.70

**Crisis Class (most important):**
- Precision: >0.60 (when model says crisis, it's right 60%+ of time)
- Recall: >0.70 (detect 70%+ of actual crisis hours)
- F1 score: >0.50

**False Positive Rate:**
- <20% (don't over-trigger crisis alerts)

**OOS Validation (Aug 2024):**
- Detect crisis during carry unwind event
- Combined crisis + risk_off detection: >70%

If these metrics aren't met, consider:
1. Labeling more crisis examples
2. Adjusting crisis thresholds
3. Adding more features (temporal, technical)

---

## 10. FAQ

### Q: How strict should I be with crisis labels?
**A:** Very strict. Crisis should be rare (<5% of data). If in doubt, use risk_off instead.

### Q: Can I change labels later?
**A:** Yes! The tool saves to CSV, you can edit manually or re-run the tool to update.

### Q: What if crisis_composite_score suggests crisis but price didn't drop much?
**A:** Use your judgment. If there's no significant price impact, label as risk_off instead. Crisis should have real market impact.

### Q: Should I label based on what I know happened later?
**A:** No! Label based only on information available at that hour. Avoid hindsight bias.

### Q: How do I handle transitions (crisis → risk_off → neutral)?
**A:** Label hour-by-hour during transitions. The model needs to learn these gradients.

### Q: Is 10 hours of labeling worth it?
**A:** Yes! Manual labels are the foundation of supervised learning. Good labels = good model.

---

## 11. Support

**Issues with the labeling tool?**
- Check that feature store exists: `data/features_mtf/BTC_1H_2022-01-01_to_2024-12-31.parquet`
- Verify crisis features are present (run `bin/analyze_crisis_features.py`)

**Questions about labels?**
- Review the 4 example labeled periods in Section 4
- When in doubt, be conservative (neutral or risk_off, not crisis)

**Model performance issues?**
- Ensure you labeled at least 100+ hours (minimum for training)
- Balance classes: crisis should be ~5%, risk_off ~20%, neutral/risk_on ~75%
- Add surrounding context to improve regime transition learning

---

## 12. Next Steps

1. **Start labeling:** `python bin/label_crisis_periods.py`
2. **Label Phase 1:** 5 major crisis events (~5 hours)
3. **(Optional) Label Phase 2:** Surrounding context (~5 hours)
4. **Train model:** `python bin/train_regime_classifier.py`
5. **Evaluate:** `python bin/evaluate_regime_classifier.py`
6. **Validate OOS:** `python bin/validate_regime_classifier_oos.py`
7. **Deploy:** Integrate with existing regime discriminators

**Goal:** Train a model that detects future crises automatically, improving regime-aware strategy performance by 20-30% (based on Option A research).

Good luck! Your labels will power the next generation of regime detection. 🚀
