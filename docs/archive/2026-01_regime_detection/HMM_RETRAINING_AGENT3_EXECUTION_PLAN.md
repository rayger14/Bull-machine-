# Agent 3: HMM Retraining & Deployment - Execution Plan

## Current Status (Pre-Agent 2 Delivery)

### Baseline Performance (9 lagging features)
```
Crisis Detection: 20% (1/5 events) ❌
  - LUNA: 0% detected
  - FTX: 0% detected
  - June 2022: 0% detected

Model Quality:
  - Silhouette: 0.089 (target: >0.50) ❌
  - Transitions/year: 13.7 (target: 10-20) ✅

Verdict: NOT DEPLOYMENT READY
```

### Stage 2: Existing Crisis Features (14 features)
```
Crisis Detection: 20% (1/5 events) ❌
  - LUNA: 10% detected (slight improvement)
  - FTX: 23% detected (slight improvement)
  - June 2022: 0% detected

Model Quality:
  - Silhouette: 0.080 (target: >0.50) ❌
  - Transitions/year: 152.6 (target: 10-20) ❌ THRASHING

Verdict: NOT DEPLOYMENT READY
Issue: Existing crisis features are noisy, causing regime thrashing
```

## Root Cause Analysis

**Why existing crisis features fail:**

1. **crisis_composite & crisis_context**: Too generic, don't spike sharply during actual crises
   - LUNA window: mean=0.51, max=0.64 (not strong enough signal)
   - Need REAL-TIME crash detection (flash_crash_1h, flash_crash_4h)

2. **volume_z**: Wrong calculation window
   - Current: 7-day z-score (too slow to react)
   - Need: 1h volume spike detection for immediate crashes

3. **volatility_spike**: Binary flag but poorly tuned
   - Not sensitive enough to crisis-level volatility
   - Need: Multiple timeframe volatility cascade detection

4. **Missing critical features from Agent 2:**
   - ❌ flash_crash_1h/4h/1d: Detect crashes AS THEY HAPPEN
   - ❌ oi_cascade: OI waterfall liquidations
   - ❌ funding_extreme/flip: Extreme funding rate moves
   - ❌ oi_delta_1h_z: 1-hour OI change z-score

## Agent 2 Dependency Specification

### Required Features (Agent 2 must deliver)

**Priority 1: Flash Crash Detection (CRITICAL)**
```python
# Column name: flash_crash_1h
# Type: binary (0 or 1)
# Logic: 1 if price drop >5% in 1 hour OR volatility >3σ in 1h
# Coverage: 100% (2022-2024)
# Expected spike during LUNA: >80% of hours during May 9-12

# Column name: flash_crash_4h
# Type: binary
# Logic: 1 if price drop >10% in 4 hours OR sustained volatility >2.5σ
# Coverage: 100%

# Column name: flash_crash_1d
# Type: binary
# Logic: 1 if price drop >15% in 24 hours
# Coverage: 100%
```

**Priority 2: Volume Surge Detection (CRITICAL)**
```python
# Column name: volume_spike
# Type: binary OR continuous (0-1)
# Logic: 1 if volume >3σ above 24h rolling mean
# Coverage: 100%
# Expected: Spike during all crisis events

# Column name: volume_z_7d (if not already correct)
# Type: continuous
# Logic: z-score of volume vs 7-day rolling window
# Coverage: 100%
```

**Priority 3: OI Cascade Detection (HIGH)**
```python
# Column name: oi_delta_1h_z
# Type: continuous
# Logic: z-score of 1h OI change (to detect liquidation cascades)
# Coverage: 100%
# Expected: Negative spike during LUNA/FTX

# Column name: oi_cascade
# Type: binary
# Logic: 1 if OI drops >5% in 1 hour (mass liquidations)
# Coverage: 100%
```

**Priority 4: Funding Extreme Detection (MEDIUM)**
```python
# Column name: funding_extreme
# Type: binary
# Logic: 1 if funding_rate >99th percentile or <1st percentile
# Coverage: 100%

# Column name: funding_flip
# Type: binary
# Logic: 1 if funding rate changes sign AND magnitude >0.5σ in <4h
# Coverage: 100%
```

**Priority 5: Crisis Composite Score (NICE TO HAVE)**
```python
# Column name: crisis_composite_score (if different from existing)
# Type: continuous (0-7)
# Logic: Count of triggered crisis indicators
# Coverage: 100%
# Expected during LUNA: 4-6 indicators active simultaneously
```

### Data Quality Requirements

**All features must:**
- ✅ 100% coverage (2022-2024, no NaNs)
- ✅ Correct timestamps (aligned with 1H bars)
- ✅ Validated on LUNA event (May 9-12, 2022): Show values spike
- ✅ Documented in feature_store with clear descriptions

**Acceptance Test:**
```python
# Agent 3 will run this test on delivery:
df = pd.read_parquet('data/features_mtf/BTC_1H_2022-01-01_to_2024-12-31.parquet')

luna_window = df.loc['2022-05-09':'2022-05-12']

# MUST PASS:
assert luna_window['flash_crash_1h'].mean() > 0.5  # >50% of hours detected
assert luna_window['volume_spike'].mean() > 0.4    # >40% volume surge
assert luna_window['oi_cascade'].sum() >= 3         # At least 3 cascade events
```

## Agent 3 Execution Timeline (POST Agent 2 Delivery)

### Phase 1: Feature Validation (15 minutes)

**Step 1.1: Verify feature delivery**
```bash
python3 -c "
import pandas as pd
df = pd.read_parquet('data/features_mtf/BTC_1H_2022-01-01_to_2024-12-31.parquet')

required = ['flash_crash_1h', 'flash_crash_4h', 'flash_crash_1d',
            'volume_spike', 'oi_delta_1h_z', 'oi_cascade',
            'funding_extreme', 'funding_flip']

for feat in required:
    if feat in df.columns:
        coverage = df[feat].notna().sum() / len(df) * 100
        print(f'✅ {feat}: {coverage:.1f}% coverage')
    else:
        print(f'❌ {feat}: MISSING')
"
```

**Step 1.2: LUNA validation test**
```bash
python3 bin/validate_agent2_crisis_features.py
# Expected output:
# ✅ LUNA flash_crash_1h: 73% triggered (58/80 hours)
# ✅ LUNA volume_spike: 67% triggered
# ✅ LUNA oi_cascade: 8 events detected
```

**If validation fails:** Report to Agent 2, wait for fixes

### Phase 2: HMM Retraining (20 minutes)

**Step 2.1: Train Stage 3 (Agent 2 features)**
```bash
# Full ensemble with 10 random initializations
python3 bin/train_hmm_with_crisis_features.py \
  --stage agent2 \
  --n_init 10 \
  --n_iter 1000 \
  > results/hmm_training_stage3.log 2>&1

# Expected improvements:
# - Crisis detection: 20% → 80%+
# - Silhouette: 0.089 → 0.50+
# - Transitions: 13.7/year (should stay stable)
```

**Step 2.2: Run comprehensive validation**
```bash
python3 bin/comprehensive_hmm_validation.py \
  --model models/hmm_regime_agent2.pkl \
  > results/hmm_validation_comprehensive.txt

# Success criteria:
# ✅ LUNA detection: >80% (vs current 0%)
# ✅ FTX detection: >80% (vs current 0%)
# ✅ June detection: >70% (vs current 0%)
# ✅ False positive rate: <2%
# ✅ Silhouette: >0.50
```

**If Stage 3 fails (<80% crisis detection):**
- Try Stage 4 (full feature set)
- Increase n_init to 20 for more robust training
- Adjust feature weights in interpretation logic

### Phase 3: Deployment Readiness (10 minutes)

**Step 3.1: Generate deployment artifacts**
```bash
python3 bin/generate_hmm_deployment_guide.py \
  --model models/hmm_regime_agent2.pkl \
  --output HMM_PRODUCTION_DEPLOYMENT_GUIDE.md
```

**Step 3.2: Create monitoring script**
```bash
# Will monitor HMM regime changes in live trading
python3 bin/create_hmm_monitoring_script.py
```

**Step 3.3: Create rollback plan**
```bash
# Document how to revert to static labels if HMM fails
python3 bin/create_hmm_rollback_plan.py
```

## Deliverables (When Agent 2 completes)

### Immediate Deliverables (45 min after Agent 2)

**1. Training Report**
```
File: results/hmm_training_report_agent2_YYYYMMDD_HHMMSS.txt
Contains:
  - Best model seed & log-likelihood
  - Feature importance (which features mattered most)
  - Regime distribution
  - Silhouette score
  - Transition frequency
```

**2. Validation Report**
```
File: results/hmm_validation_comprehensive_YYYYMMDD_HHMMSS.txt
Contains:
  - Event-by-event breakdown (LUNA, FTX, June, etc.)
  - Detection accuracy per event
  - Early detection hours per event
  - False positive analysis
  - PASS/FAIL verdict
```

**3. Trained Model**
```
File: models/hmm_regime_agent2.pkl
Contains:
  - GaussianHMM model (best of 10 inits)
  - StandardScaler
  - State mapping {0: 'crisis', 1: 'risk_on', ...}
  - Feature list (in order)
  - Metadata (training date, metrics, etc.)
```

**4. Regime Labels**
```
File: data/regime_labels_agent2.parquet
Contains:
  - timestamp (index)
  - regime_label (crisis, risk_off, neutral, risk_on)
  - 2022-2024 hourly labels
```

### Production Deployment Deliverables (if >80% accuracy)

**5. Deployment Guide**
```
File: HMM_PRODUCTION_DEPLOYMENT_GUIDE.md
Sections:
  - Validation Results Summary
  - Production Integration Steps
  - Config changes required
  - Monitoring setup
  - Rollback procedure
  - Expected performance impact
```

**6. Monitoring Script**
```
File: bin/monitor_hmm_regime_live.py
Purpose:
  - Log regime transitions
  - Alert on crisis detection
  - Track confidence scores
  - Compare HMM vs static labels
```

**7. Rollback Plan**
```
File: HMM_ROLLBACK_PLAN.md
Contains:
  - Rollback triggers (when to abort HMM)
  - Revert procedure (< 5 minutes)
  - Backup configs with static labels
  - Post-rollback validation steps
```

### Failure Analysis Deliverable (if <80% accuracy)

**8. Failure Report**
```
File: HMM_FAILURE_ANALYSIS_YYYYMMDD.md
Contains:
  - Which events still missed?
  - Which features didn't help?
  - Feature correlation analysis
  - Recommended next steps:
    * More features needed?
    * Different model architecture (Hidden Semi-Markov)?
    * Manual regime labeling instead?
```

## Success Metrics

### Deployment Ready Criteria (ALL must pass)

| Metric | Current | Target | Status |
|--------|---------|--------|--------|
| LUNA detection | 0% | >80% | ❌ WAITING FOR AGENT 2 |
| FTX detection | 0% | >80% | ❌ WAITING FOR AGENT 2 |
| June detection | 0% | >70% | ❌ WAITING FOR AGENT 2 |
| Overall crisis accuracy | 20% | >80% | ❌ WAITING FOR AGENT 2 |
| Bull event accuracy | 100% | >70% | ✅ PASSING |
| Silhouette score | 0.089 | >0.50 | ❌ WAITING FOR AGENT 2 |
| Transitions/year | 13.7 | 10-20 | ✅ PASSING |
| False positive rate | TBD | <2% | ⏳ PENDING |
| Crisis distribution | 16.7% | 5-15% | ⚠️  MARGINAL |

### Expected Impact (if deployed)

**Crisis Regime Performance:**
```
Current (static 2022=risk_off):
  - Crisis PF: 2.1
  - Crisis accuracy: 0% (lumps all 2022 together)
  - Early detection: 0 days (year-based labels)

After HMM (estimated):
  - Crisis PF: 2.4-2.6 (+15-24%)
  - Crisis accuracy: 80-90%
  - Early detection: 10-15 days (real-time detection)
```

**Overall Portfolio:**
```
Current:
  - Overall PF: ~2.0
  - Max drawdown: -25%
  - Regime routing: Coarse (year-based)

After HMM:
  - Overall PF: 2.3-2.5 (+15-25%)
  - Max drawdown: -15-20% (-5-10% improvement)
  - Regime routing: Dynamic (hour-by-hour)
```

## Communication Protocol

### When Agent 2 Completes

**Agent 2 → Agent 3 handoff message:**
```
🚀 Agent 2 Complete: Crisis features delivered

Features added:
  - flash_crash_1h, flash_crash_4h, flash_crash_1d
  - volume_spike (retrained), oi_delta_1h_z, oi_cascade
  - funding_extreme, funding_flip

Data file: data/features_mtf/BTC_1H_2022-01-01_to_2024-12-31.parquet
Validation: LUNA shows 78% flash_crash_1h triggering (62/80 hours)

Agent 3: You're cleared for HMM retraining
Timeline: 45 minutes to deployment decision
```

**Agent 3 → User status updates:**

Every 15 minutes:
```
[T+0]  🔄 Feature validation starting...
[T+15] ✅ Features validated. Starting HMM training (10 inits)...
[T+30] ✅ Training complete. Running comprehensive validation...
[T+40] ✅ Validation complete. Generating deployment artifacts...
[T+45] 🎯 DEPLOYMENT DECISION: [READY/NOT READY]
```

## Waiting State Actions (NOW)

**What Agent 3 is doing while waiting:**

1. ✅ Created training infrastructure (`train_hmm_with_crisis_features.py`)
2. ✅ Created validation framework (`comprehensive_hmm_validation.py`)
3. ✅ Established baseline (Stage 1: 20% crisis detection)
4. ✅ Tested existing features (Stage 2: still 20%, thrashing issues)
5. ✅ Documented Agent 2 requirements (this file)
6. ⏳ **READY TO EXECUTE** when Agent 2 delivers

**Pre-staged artifacts:**
- `bin/train_hmm_with_crisis_features.py` (READY)
- `bin/comprehensive_hmm_validation.py` (READY)
- Feature requirement spec (READY)
- Validation tests (READY)
- Deployment templates (READY)

**Execution trigger:**
```bash
# Agent 3 will run this when Agent 2 completes:
./bin/execute_hmm_retraining_pipeline.sh
```

## Risk Mitigation

### What if Stage 3 fails? (Contingency plan)

**Scenario 1: <80% crisis detection even with Agent 2 features**

Options:
- A) Try more random initializations (n_init=20)
- B) Try different covariance structures (full vs diagonal)
- C) Add more features from Agent 1 research
- D) Switch to supervised learning (manual regime labeling)

**Scenario 2: High false positive rate (>2%)**

Options:
- A) Adjust crisis threshold (require more indicators)
- B) Add smoothing (require 2+ consecutive crisis hours)
- C) Add post-processing filters

**Scenario 3: Regime thrashing (>30 transitions/year)**

Options:
- A) Add transition penalties to HMM
- B) Increase minimum regime duration (e.g., 24 hours)
- C) Use Viterbi smoothing with stronger priors

---

**Status: READY FOR AGENT 2 DELIVERY**
**Estimated execution time: 45 minutes from Agent 2 completion**
**Deployment decision: 80%+ crisis detection → DEPLOY, <80% → INVESTIGATE**
