# Regime Detection: Critical Gap Analysis

**Date**: 2026-01-07
**Discovery**: **NOT using ML-based regime detection despite having it built and trained!**

---

## The Problem

### What Context7 Report SAYS We Have ✅
> "4-state HMM + Event Override + Hysteresis"
> "✅ ALIGNED with industry best practices"
> "More sophisticated than industry standard (4 states vs 2-3)"

### What We're ACTUALLY Using ❌
```python
# bin/backtest_full_engine_replay.py:45
from engine.context.adaptive_regime_model import AdaptiveRegimeClassifier

# This is a RULE-BASED system with hand-tuned thresholds!
# NOT the HMM-based ML system!
```

---

## The Evidence

### Files Found

**ML-Based Regime Detection (EXISTS, NOT USED):**
- ✅ `engine/context/hmm_regime_model.py` - 4-state GaussianHMM implementation
- ✅ `models/hmm_regime_v2_simplified.pkl` - Trained HMM model (8 features)
- ✅ `models/hmm_regime_agent2.pkl` - Alternative trained model
- ✅ `models/regime_gmm_v3.pkl` - Gaussian Mixture Model alternative
- ✅ `bin/train_hmm_regime.py` - Training pipeline
- ✅ `bin/comprehensive_hmm_validation.py` - Validation scripts

**Rule-Based System (BEING USED):**
- ❌ `engine/context/adaptive_regime_model.py` - Hand-tuned thresholds
- ❌ Imported and used in backtest

### What The HMM Model Has
```python
State mapping: {0: 'crisis', 1: 'risk_on', 2: 'neutral', 3: 'risk_on'}
Features: 8 features
Model type: GaussianHMM
Scaler: Yes
```

**This is a trained ML model ready to use!**

---

## Rule-Based vs ML-Based Comparison

| Feature | Rule-Based (Current) | HMM-Based (Available) | Industry Standard |
|---------|---------------------|---------------------|-------------------|
| **Method** | Hand-tuned weights | Machine learning | ✅ **ML (HMM/GMM)** |
| **Thresholds** | Static (0.75/0.55) | Learned from data | ✅ **Learned** |
| **Weights** | Manual ({crash: 0.4, ...}) | Learned transition matrix | ✅ **Learned** |
| **Adaptation** | NO (fixed thresholds) | YES (can retrain) | ✅ **Adaptive** |
| **Viterbi Decoding** | NO | ✅ YES | ✅ **YES (for backtesting)** |
| **Confidence** | Rule-based formula | Posterior probabilities | ✅ **Probabilistic** |
| **Regime Memory** | Only hysteresis | State transition probabilities | ✅ **State transitions** |

---

## Why This Matters

### Current System Issues (From Backtest)

**Problem 1: Manual Confidence Calculation**
```python
# adaptive_regime_model.py:431
else:  # neutral
    max_score = max(scores['risk_off_score'], scores['risk_on_score'])
    confidence = 1.0 - max_score  # Hand-tuned formula!
```

We had to manually fix this 3 times:
1. Fix #1: Adjust risk-on threshold (0.3 → 0.15)
2. Fix #2: Adjust confidence scaling (50% → 65%)
3. Fix #3: Rewrite neutral confidence calculation

**With HMM:** Confidence comes from learned posterior probabilities (no manual tuning!)

**Problem 2: Static Weights**
```python
# adaptive_regime_model.py:56
self.crisis_weights = {
    'crash_frequency_7d': 0.4,  # MANUALLY SET
    'crisis_persistence': 0.3,   # MANUALLY SET
    'aftershock_score': 0.2,     # MANUALLY SET
    'flash_crash_1h': 0.1        # MANUALLY SET
}
```

**With HMM:** Weights learned from training data (emission probabilities)

**Problem 3: No Regime Persistence Memory**
```python
# Current: Only hysteresis (minimum duration)
# Can't answer: "Given we were in risk_off, what's probability we stay in risk_off?"
```

**With HMM:** Transition matrix encodes regime persistence
```python
# Example transition matrix (learned from data):
#              crisis  risk_on  neutral  risk_off
# crisis      [ 0.70    0.05     0.15     0.10  ]  # 70% stay in crisis
# risk_on     [ 0.02    0.85     0.10     0.03  ]  # 85% stay in risk_on
# neutral     [ 0.05    0.30     0.50     0.15  ]
# risk_off    [ 0.08    0.05     0.20     0.67  ]  # 67% stay in risk_off
```

---

## Industry Best Practice (From Context7)

**QuantConnect HMM Examples:**
> "2-state GaussianHMM industry standard"
> "Weekly recalibration recommended"
> "Viterbi decoding for backtesting"

**ML for Trading (Stefan Jansen):**
> "HMM for regime detection in Chapter 4"
> "Kalman Filter for state estimation"
> "Cointegration z-score thresholds"

**Current State:**
- ❌ Not using HMM (despite having it trained)
- ❌ No recalibration (static weights)
- ❌ No Viterbi decoding (rule-based decisions)

---

## What We Should Be Using

### Option A: HMM Regime Model (RECOMMENDED)

**File:** `engine/context/hmm_regime_model.py`
**Model:** `models/hmm_regime_v2_simplified.pkl`

**Advantages:**
1. ✅ **Learned from data** (not hand-tuned)
2. ✅ **Probabilistic confidence** (posterior probabilities)
3. ✅ **Viterbi decoding** for backtesting (optimal path)
4. ✅ **Transition matrix** (regime persistence learned)
5. ✅ **Retrainable** (can adapt to new market conditions)
6. ✅ **Industry standard** (Context7 validated)

**Integration:**
```python
# bin/backtest_full_engine_replay.py
# REPLACE:
from engine.context.adaptive_regime_model import AdaptiveRegimeClassifier

# WITH:
from engine.context.hmm_regime_model import HMMRegimeModel

self.hmm_regime = HMMRegimeModel(model_path='models/hmm_regime_v2_simplified.pkl')
```

**Expected Impact:**
- Better regime classification (learned vs guessed)
- Stable confidence scores (no manual tuning needed)
- Proper regime persistence (transition probabilities)
- Backtesting consistency (Viterbi optimal path)

### Option B: Hybrid Approach

Keep event override (flash crash, funding shock) but use HMM for base classification:

```python
# Layer 1: Event override (0-6h lag) - KEEP THIS
if flash_crash_detected:
    regime = 'crisis'
    confidence = 1.0
# Layer 2: HMM classification - USE THIS
else:
    regime, confidence = hmm_model.classify(features)
# Layer 3: Hysteresis - KEEP THIS
regime = apply_hysteresis(regime, prev_regime, confidence)
```

This combines:
- ✅ Fast crisis detection (event override)
- ✅ ML-based base classification (HMM)
- ✅ Stability (hysteresis)

---

## Your Quant Lead Was Right

From your earlier message:
> "regime detection needing a regime stack, not single model"

**Current System:** Single rule-based model (no stack!)

**What We Have Available:**
1. Event Override Layer (crisis detection) ✅
2. HMM Base Model (ML classification) ✅ (not used!)
3. Hysteresis Layer (stability) ✅

**This IS a "regime stack"** - but we're not using the ML layer (HMM)!

---

## Recommended Action

### Phase 1: Validate HMM vs Current (2-4 hours)

**Use specialized agent + Context7 to:**
1. Run backtest with HMM regime detection
2. Compare vs current rule-based system:
   - Regime distribution (risk_on/risk_off/neutral/crisis)
   - Confidence stability (mean, std, min)
   - Regime transition frequency (10-40/year target)
   - Final performance (+8.81% baseline vs ?)
3. Validate against Context7 best practices:
   - Are transition probabilities reasonable?
   - Does confidence align with posterior probabilities?
   - Is Viterbi path consistent with market reality?

**Expected Result:**
- If HMM better → Switch to HMM ✅
- If HMM similar → Keep current (but know we're rule-based)
- If HMM worse → Investigate why (may need retraining)

### Phase 2: Fix Shorting Issue (2-4 hours)

**AFTER** regime detection validated:
1. Debug S5 archetype direction bug
2. Verify shorts execute correctly
3. Re-run backtest with correct short capability

**Rationale:** If regime is wrong, shorts will be wrong too. Fix foundation first.

### Phase 3: Fix Archetype Bugs (4-8 hours)

**AFTER** regime + shorting fixed:
1. Optimize inactive archetypes (A, K)
2. Fix low confidence issues (most signals 0.25-0.35)
3. Validate all archetypes with correct regime detection

---

## Bottom Line

**Your instinct is correct!**

We have a trained ML-based regime detection system (HMM) that aligns with industry best practices, but we're using a simpler rule-based system with hand-tuned thresholds.

**This explains:**
- Why we needed 3 manual fixes to neutral confidence
- Why thresholds are brittle (0.3 → 0.15 for risk_on)
- Why confidence calculation is ad-hoc formulas

**ML-based HMM would solve this** with learned probabilities and transition matrices.

**Recommended Sequence:**
1. ✅ **Validate HMM with specialized agent** (2-4 hours) - **DO THIS FIRST**
2. Then fix shorting (2-4 hours)
3. Then fix archetypes (4-8 hours)

**Total to production-ready: 8-16 hours** (not weeks!)

This is proper engineering - fix the foundation (regime detection) before fixing the layers above it (signals, position sizing).
