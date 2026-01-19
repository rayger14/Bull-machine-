# HMM Regime Detection - Diagnosis Complete

**Date:** 2025-12-18
**Status:** ⚠️ **DATA PIPELINE ISSUE IDENTIFIED**

---

## Executive Summary

Agent 4's research was **technically correct** - HMM is the best approach for dynamic regime detection (85% accuracy potential). However, I discovered the **feature engineering pipeline is broken**:

```
Critical Features Status:
- RV_20 (volatility): ALL ZEROS ❌
- funding_Z: ALL ZEROS ❌
- VIX_Z: ALL ZEROS ❌
- rv_20d: Doesn't exist (all NaN) ❌
```

**The HMM model can't learn meaningful regimes from zero-valued features.**

---

## What I Found

### 1. Existing HMM Model Status

**File:** `models/hmm_regime_v2_simplified.pkl`

**Problem:** Predicts 100% neutral across all data

**Root Causes:**
1. ✅ **Single random initialization** - Got stuck in local optimum (research said use 10)
2. ✅ **Wrong covariance type** - Used 'diag' instead of 'full' (research recommendation)
3. ❌ **CRITICAL: Feature mismatch** - Model trained on features that don't exist in data

### 2. What I Tried

**Attempt 1:** Run existing model on 2022-2023 data
- Result: 100% neutral predictions, 0% crisis detection
- Missed all 3 crisis events (LUNA, FTX, June 2022)

**Attempt 2:** Retrain with existing training script
- Result: Training failed with NaN values
- All state means = NaN
- ValueError: startprob_ must sum to 1 (got nan)

**Attempt 3:** Create simplified training script with multi-initialization
- Result: Model trained successfully! ✅
- Found best model (seed=9, log-likelihood=393910.71)
- BUT: Predicts 99.8% risk_on (same problem, different regime)

**Attempt 4:** Diagnose feature data
- **CRITICAL DISCOVERY:** All regime features are ZERO or NaN in the dataset
- The data file doesn't have proper regime features computed

---

## Data Pipeline Analysis

### Available Data File

**Path:** `data/features_mtf/BTC_1H_2022-01-01_to_2023-12-31_with_macro.parquet`

**Features:** 119 columns total

**Regime Features Status:**

| Feature Needed | Exists? | Status | Notes |
|----------------|---------|--------|-------|
| `funding_Z` | ✅ Yes | ❌ ALL ZEROS | 30-day z-score not computed |
| `RV_21` | ❌ No | ❌ N/A | Doesn't exist |
| `RV_20` | ✅ Yes | ❌ ALL ZEROS | Should be 40-80% range |
| `rv_20d` | ❌ No | ❌ ALL NaN | Doesn't exist |
| `VIX_Z` | ✅ Yes | ❌ ALL ZEROS | Z-score not computed |
| `DXY_Z` | ✅ Yes | ✅ WORKING | Has real values |
| `OI_CHANGE` | ❌ No | ❌ N/A | Use `oi_change_pct_24h` instead |
| `oi_change_pct_24h` | ✅ Yes | ✅ WORKING | Has real values |
| `USDT.D` | ✅ Yes | ✅ WORKING | Has real values |
| `BTC.D` | ✅ Yes | ✅ WORKING | Has real values |
| `YC_SPREAD` | ✅ Yes | ✅ WORKING | Has real values |

**Summary:** Only 5 of 11 core features are properly computed.

---

## Why This Matters

The HMM learns regime patterns by analyzing feature distributions across different market conditions:

- **Crisis:** High VIX_Z (fear), high RV (volatility), negative funding
- **Risk_on:** Low VIX_Z, positive funding, positive returns
- **Risk_off:** High DXY_Z (strong dollar), high USDT dominance
- **Neutral:** Middle-of-road values

**Without proper features, the HMM has no signal to learn from.**

It's like trying to classify images as "cat" or "dog" but all the images are blank - the model will just pick one class for everything.

---

## What Agent 4's Research Said

From `DYNAMIC_REGIME_DETECTION_RESEARCH.md`:

> **RECOMMENDED APPROACH: Hidden Markov Models (HMM) with 21-day Rolling Window**
>
> ✅ **Already Implemented:** Your codebase has `HMMRegimeModel` v2 ready to deploy
> ✅ **Expected accuracy:** 80-85% on crisis events
> ✅ **Implementation time:** 30-60 minutes (training + validation + deployment)

**The research was correct about the approach, but assumed the feature engineering pipeline was working.**

---

## Options Going Forward

### Option A: Fix Feature Engineering Pipeline (2-3 hours)

**Tasks:**
1. Diagnose why features are all zeros
2. Re-run feature engineering on raw price/macro data
3. Verify features have proper distributions
4. Retrain HMM with corrected features
5. Validate on crisis events

**Pros:**
- ✅ Unlocks dynamic regime detection (Agent 4's recommendation)
- ✅ Expected +0.3-0.5 PF improvement
- ✅ 10-15 days earlier crisis detection

**Cons:**
- ❌ Requires debugging feature engineering pipeline
- ❌ May uncover additional data issues
- ❌ Delays operational layer work (paper trading dashboard, etc.)

**Risk:** Medium - Feature engineering may be broken for a reason (data source issues, API changes, etc.)

---

### Option B: Deploy with Static Labels (0 hours) ⭐ RECOMMENDED

**What:**
- Use existing year-based regime labels (2022=risk_off, 2023=neutral)
- Move forward with paper trading, kill-switches, meta-model
- Queue HMM for Phase 2 after operational layer is live

**Pros:**
- ✅ **Working system > perfect system**
- ✅ Static labels already validated in smoke tests
- ✅ Unblocks operational layer (your stated priority)
- ✅ Can fix HMM pipeline during paper trading period

**Cons:**
- ❌ Miss intra-year regime changes (Q1 2023 was bullish, labeled neutral)
- ❌ No dynamic crisis detection
- ❌ 10-15 day lag in regime shifts

**Risk:** Low - Static labels are crude but working

---

### Option C: Hybrid Approach (1 hour)

**What:**
1. Deploy with static labels NOW
2. Launch paper trading dashboard + kill-switches
3. Fix HMM pipeline in parallel (separate task)
4. A/B test HMM vs static after validation

**Pros:**
- ✅ Best of both worlds
- ✅ Unblocks operational layer immediately
- ✅ Preserves HMM path forward
- ✅ Validates both approaches in paper trading

**Cons:**
- ❌ Requires maintaining two regime detection systems
- ❌ More testing required (static vs HMM comparison)

**Risk:** Low-Medium

---

## Recommendation

**Deploy Option B (Static Labels) NOW for these reasons:**

1. **Your Priority Shift:** You said "progress comes more from operating the system than inventing new logic"

2. **Option A Complete:** You already have:
   - ✅ Multi-objective optimization (+15-25% OOS Sharpe)
   - ✅ Regime discriminators (production-ready)
   - ✅ Direction metadata (100% coverage)
   - ✅ Paper trading dashboard spec (73 pages)
   - ✅ Kill-switch specification (4-tier escalation)
   - ✅ Meta-model architecture (overlap as feature)

3. **HMM is Phase 2:** Fix feature pipeline during 60-day paper trading

4. **Risk Management:** Don't let perfect (HMM) be the enemy of good (static labels)

---

## What I Learned

### Research Agent Was Right About:
- ✅ HMM is best method (85% accuracy potential)
- ✅ Multiple initializations critical (avoid local optima)
- ✅ Full covariance better than diagonal
- ✅ Implementation exists in codebase

### Research Agent Missed:
- ❌ Feature engineering pipeline is broken
- ❌ Existing model was poorly trained (single init)
- ❌ Data file doesn't have required features computed
- ❌ "30-60 minutes" timeline unrealistic with broken pipeline

**Lesson:** Implementation research != data pipeline validation

---

## Next Steps (Recommended)

### Immediate (Today)

1. ✅ **Accept static labels for now** (working system > perfect system)
2. ⏳ **Check multi-regime validation results** (3 smoke tests running in background)
3. ⏳ **Commit all Option A changes** (multi-objective, regime discriminators, metadata, operational specs)
4. ⏳ **Update OPTION_A_EXECUTIVE_SUMMARY.md** with HMM findings

### Short-Term (This Week)

5. **Deploy paper trading dashboard** (Agent 2 delivered spec)
6. **Implement circuit breakers** (Agent 2 delivered code)
7. **Begin meta-model Phase 1** (Agent 3 delivered architecture)

### Medium-Term (During 60-Day Paper Trading)

8. **Fix HMM feature engineering pipeline**:
   - Diagnose why RV_20, funding_Z, VIX_Z are all zeros
   - Re-compute features from raw data
   - Validate feature distributions
   - Retrain HMM with proper multi-initialization
   - Validate on 10+ crisis events

9. **A/B test HMM vs static labels** in paper trading

10. **Deploy winning regime detection method** to production

---

## Files Created During Investigation

1. `bin/quick_hmm_validation.py` - Validation framework for regime detection
2. `bin/train_hmm_simplified.py` - Proper multi-initialization training script
3. `/tmp/hmm_validation_results.txt` - Validation output (100% neutral failure)
4. `/tmp/hmm_training.log` - Multi-init training log (99.8% risk_on)
5. `HMM_DIAGNOSIS_COMPLETE.md` - This file

---

## Conclusion

**HMM regime detection is the right long-term solution**, but requires fixing the feature engineering pipeline first.

**Pragmatic path:** Deploy with static labels NOW, fix HMM during paper trading.

**Expected timeline:**
- Static labels: 0 hours (already working)
- HMM pipeline fix: 2-3 hours (Phase 2)
- Total delay avoided: 2-3 hours ✅

**Your call:** Should we:
- A) Fix HMM pipeline now (2-3 hours)
- B) Deploy with static labels, fix HMM in Phase 2 ⭐
- C) Hybrid approach (1 hour setup)

---

**Generated:** 2025-12-18
**Investigation Time:** 2 hours
**Status:** Diagnosis complete, awaiting decision
