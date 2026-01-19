# HMM Regime Detection Integration Design

**Status:** Design & Planning Phase
**Author:** Claude Code (Backend Architect)
**Date:** 2026-01-07
**Target:** Replace rule-based regime with ML-based HMM

---

## Executive Summary

This document provides a complete integration plan for switching from rule-based regime detection (`AdaptiveRegimeClassifier`) to ML-based HMM regime detection (`HMMRegimeModel`).

**Key Findings:**
- **Data Compatibility:** ✅ All 8 HMM features available in production data
- **Interface Compatibility:** ⚠️ Partial - requires wrapper adapter
- **Performance Impact:** 🟡 Unknown - needs benchmarking
- **Risk Level:** 🟡 Medium - requires careful rollout and testing

**Recommended Approach:** **Option B - Hybrid Integration** (maintain 3-layer architecture while replacing state-based layer with HMM)

---

## Table of Contents

1. [Current System Analysis](#1-current-system-analysis)
2. [HMM Model Analysis](#2-hmm-model-analysis)
3. [Integration Options](#3-integration-options)
4. [Recommended Approach](#4-recommended-approach)
5. [Implementation Plan](#5-implementation-plan)
6. [Testing Strategy](#6-testing-strategy)
7. [Risk Assessment](#7-risk-assessment)
8. [Rollback Plan](#8-rollback-plan)

---

## 1. Current System Analysis

### 1.1 Current Architecture

The Bull Machine uses a **3-layer adaptive regime detection system**:

```
┌─────────────────────────────────────────────────────────────┐
│                    LAYER 1: EVENT OVERRIDE                   │
│  Crisis detection via event triggers (flash crash, funding   │
│  shock, OI cascade) → Immediate regime override (12h TTL)    │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│              LAYER 2: STATE-BASED CLASSIFICATION             │
│  Rule-based scoring using state features:                    │
│  - Crisis score (crash_freq + crisis_persist + aftershock)   │
│  - Risk-off score (drawdown + VIX + funding + RV)           │
│  - Neutral/Risk-on (derived from risk-off inverse)          │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│               LAYER 3: HYSTERESIS STATE MACHINE              │
│  Dual threshold + minimum duration requirements:             │
│  - Crisis: enter=0.75, exit=0.55, min=6h                    │
│  - Risk-off: enter=0.65, exit=0.45, min=24h                │
│  - Risk-on: enter=0.70, exit=0.50, min=48h                 │
└─────────────────────────────────────────────────────────────┘
```

### 1.2 Current Integration Points

**File:** `bin/backtest_full_engine_replay.py`
**Line 178-181:**
```python
self.adaptive_regime = AdaptiveRegimeClassifier(
    enable_adaptive=self.enable_adaptive_regime,
    fallback_to_static=True
)
```

**Usage Pattern:**
```python
# Single bar classification
result = self.adaptive_regime.classify(features, timestamp)
# Returns: {'regime': 'risk_off', 'confidence': 0.75, 'proba': {...}, ...}

# Batch classification
regime_df = self.adaptive_regime.classify_series(df)
# Returns: DataFrame with regime, confidence, transition columns
```

### 1.3 Downstream Dependencies

**Systems that consume regime classification:**

1. **Backtest Engine** (`bin/backtest_full_engine_replay.py`):
   - Uses `regime_label` for archetype filtering
   - Uses `regime_confidence` for position sizing
   - Uses `regime_transition` for risk adjustment

2. **Archetype Logic** (`engine/archetypes/logic_v2_adapter.py`):
   - Checks `ARCHETYPE_REGIMES` mapping
   - Applies regime soft penalties (0.5x confidence if mismatch)

3. **Circuit Breaker** (`engine/risk/circuit_breaker.py`):
   - Tightens risk controls in crisis mode
   - Loosens controls in risk-on mode

4. **Direction Balance** (`engine/risk/direction_balance.py`):
   - Scales position sizes based on regime confidence

5. **Trade Metadata** (saved to CSV):
   - Currently shows `regime: unknown` (BUG)
   - Should save regime classification with each trade

### 1.4 Expected Interface

```python
# REQUIRED INTERFACE (for backward compatibility)
class RegimeClassifier:
    def classify(self, features: Dict, timestamp: pd.Timestamp) -> Dict:
        return {
            'regime': str,              # 'crisis' | 'risk_off' | 'neutral' | 'risk_on'
            'confidence': float,        # 0.0-1.0
            'proba': Dict[str, float],  # Probability per regime
            'transition': bool,         # True if regime changed
            'override': bool,           # True if event override active
            'features_used': int,       # Number of features
            'adaptive': bool            # True if adaptive mode
        }

    def classify_series(self, df: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame with columns: [regime, confidence, transition, override, proba]
```

---

## 2. HMM Model Analysis

### 2.1 Trained Model Specifications

**Model File:** `models/hmm_regime_v2_simplified.pkl`

**Architecture:**
- **Type:** GaussianHMM (hmmlearn library)
- **States:** 4 states (0: crisis, 1: risk_on, 2: neutral, 3: risk_on)
- **Features:** 8 features
- **Scaler:** StandardScaler (sklearn)

**State Mapping:**
```python
{
    0: 'crisis',
    1: 'risk_on',
    2: 'neutral',
    3: 'risk_on'  # Note: States 1 and 3 both map to risk_on
}
```

### 2.2 Feature Requirements

**Required Features (8 total):**
```python
[
    'funding_Z',           # ✅ Available (30-day z-score of funding rate)
    'oi_change_pct_24h',   # ✅ Available (24h open interest % change)
    'rv_20d',              # ✅ Available (21-day realized volatility)
    'USDT.D',              # ✅ Available (USDT dominance %)
    'BTC.D',               # ✅ Available (BTC dominance %)
    'VIX_Z',               # ✅ Available (VIX z-score)
    'DXY_Z',               # ✅ Available (DXY z-score)
    'YC_SPREAD'            # ✅ Available (10Y - 2Y yield spread)
]
```

**Feature Availability:** ✅ **100% coverage** in production data

### 2.3 HMM Interface

```python
class HMMRegimeModel:
    def load_model(self, model_path: str) -> None:
        # Load trained HMM from pickle
        pass

    def classify_batch(self, df: pd.DataFrame) -> pd.DataFrame:
        # Viterbi decode entire sequence
        # Returns: df with regime_label, regime_confidence, regime_proba_* columns
        pass

    def classify_stream(self, bar: dict) -> Tuple[str, float]:
        # Single bar classification (simplified)
        # Returns: (regime_label, confidence)
        pass
```

### 2.4 Key Differences vs Rule-Based

| Feature | Rule-Based (Current) | HMM (Target) |
|---------|---------------------|--------------|
| **State representation** | Continuous scores (0-1) | Hidden states (discrete) |
| **Decision logic** | Hand-crafted thresholds | Learned from data |
| **Crisis detection** | Event-triggered (instant) | Statistical (smooth) |
| **Hysteresis** | Explicit dual thresholds | Implicit in transition matrix |
| **Lag** | 0-6 hours (event override) | Unknown - needs testing |
| **Interpretability** | High (explicit rules) | Medium (learned weights) |
| **Confidence** | Distance from threshold | Posterior probability |
| **Transitions/year** | ~20 (with hysteresis) | Unknown - needs testing |

---

## 3. Integration Options

### Option A: Drop-In Replacement

**Approach:** Replace `AdaptiveRegimeClassifier` entirely with HMM

**Pros:**
- Clean architecture
- No legacy code
- Fully ML-based

**Cons:**
- ❌ Loses event override layer (crisis detection lag increases)
- ❌ No hysteresis (may cause thrashing)
- ❌ Interface incompatibility (needs wrapper)
- ❌ High risk (all-or-nothing switch)

**Verdict:** ❌ **Not Recommended** - loses critical crisis detection

---

### Option B: Hybrid Integration (RECOMMENDED)

**Approach:** Replace Layer 2 (state-based) with HMM, keep Layers 1 & 3

```
┌─────────────────────────────────────────────────────────────┐
│              LAYER 1: EVENT OVERRIDE (KEEP)                  │
│  Flash crash, funding shock, OI cascade → Crisis override    │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│            LAYER 2: HMM CLASSIFICATION (NEW)                 │
│  GaussianHMM with 8 features → regime probabilities          │
│  Replace RegimeScorer with HMM posterior probabilities       │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│         LAYER 3: HYSTERESIS STATE MACHINE (KEEP)             │
│  Use HMM scores with existing dual threshold logic           │
└─────────────────────────────────────────────────────────────┘
```

**Implementation:**
1. Load HMM model in `AdaptiveRegimeClassifier.__init__()`
2. Replace `RegimeScorer.compute_scores()` with HMM posterior probabilities
3. Keep `EventOverrideDetector` unchanged
4. Keep `RegimeEngine` hysteresis logic unchanged
5. Maintain exact same interface for downstream systems

**Pros:**
- ✅ Keeps crisis detection (event override)
- ✅ Keeps hysteresis (prevents thrashing)
- ✅ Backward compatible interface
- ✅ Low risk (incremental change)
- ✅ Easy rollback (toggle flag)

**Cons:**
- 🟡 More complex (3 layers)
- 🟡 HMM hysteresis + explicit hysteresis may interact

**Verdict:** ✅ **RECOMMENDED** - best risk/reward

---

### Option C: Parallel Deployment

**Approach:** Run both systems side-by-side, compare outputs

**Pros:**
- ✅ Safe (no production impact)
- ✅ Empirical validation
- ✅ Easy A/B testing

**Cons:**
- ❌ 2x compute cost
- ❌ Delayed integration
- ❌ More code complexity

**Verdict:** 🟡 **Use for validation only** - good for testing phase

---

## 4. Recommended Approach

### 4.1 Hybrid Integration Design

**New File:** `engine/context/hmm_adaptive_classifier.py`

```python
"""
HMM-Enhanced Adaptive Regime Classifier

Hybrid approach:
- Layer 1: Event override (keep existing)
- Layer 2: HMM classification (NEW - replace rule-based)
- Layer 3: Hysteresis (keep existing)

Backward compatible with AdaptiveRegimeClassifier interface.
"""

class HMMRegimeScorer:
    """
    Replace RegimeScorer with HMM-based scoring.

    Uses trained GaussianHMM model to compute regime probabilities.
    """

    def __init__(self, model_path: str):
        self.hmm = HMMRegimeModel(model_path)
        self.feature_order = self.hmm.feature_order

    def compute_scores(self, features: Dict[str, float]) -> Dict[str, float]:
        """
        Compute regime scores using HMM posterior probabilities.

        Args:
            features: Dict of feature values

        Returns:
            Dict with crisis_score, risk_off_score, neutral_score, risk_on_score
        """
        # Extract HMM features
        x = np.array([features.get(f, 0.0) for f in self.feature_order])

        # Scale features
        if self.hmm.scaler:
            x = self.hmm.scaler.transform([x])[0]

        # Get posterior probabilities
        probs = self.hmm.model.predict_proba([x])[0]

        # Map states to regime scores
        crisis_score = probs[0]  # State 0 = crisis
        neutral_score = probs[2]  # State 2 = neutral
        risk_on_score = probs[1] + probs[3]  # States 1,3 = risk_on

        # Risk-off is intermediate (between crisis and neutral)
        risk_off_score = max(crisis_score * 0.8, 1.0 - risk_on_score - neutral_score)

        return {
            'crisis_score': crisis_score,
            'risk_off_score': risk_off_score,
            'neutral_score': neutral_score,
            'risk_on_score': risk_on_score
        }


class HMMAdaptiveRegimeClassifier:
    """
    Drop-in replacement for AdaptiveRegimeClassifier using HMM.

    Maintains 3-layer architecture:
    1. Event override (existing EventOverrideDetector)
    2. HMM classification (NEW HMMRegimeScorer)
    3. Hysteresis (existing RegimeEngine logic)
    """

    def __init__(
        self,
        config: Optional[Dict] = None,
        enable_adaptive: bool = True,
        enable_hmm: bool = True,
        hmm_model_path: str = 'models/hmm_regime_v2_simplified.pkl',
        fallback_to_static: bool = False
    ):
        self.enable_adaptive = enable_adaptive
        self.enable_hmm = enable_hmm

        if self.enable_adaptive and self.enable_hmm:
            # Initialize HMM-based regime engine
            hmm_scorer = HMMRegimeScorer(hmm_model_path)

            # Create RegimeEngine with HMM scorer instead of rule-based scorer
            self.engine = RegimeEngine(config)
            self.engine.scorer = hmm_scorer  # REPLACE SCORER

            logger.info("HMMAdaptiveRegimeClassifier initialized (HMM mode)")
        elif self.enable_adaptive:
            # Fallback to rule-based (existing AdaptiveRegimeClassifier)
            from engine.context.adaptive_regime_model import AdaptiveRegimeClassifier
            self.engine = AdaptiveRegimeClassifier(config, enable_adaptive=True)
            logger.info("HMMAdaptiveRegimeClassifier initialized (rule-based fallback)")
        else:
            # Static mode
            self.engine = None
            logger.info("HMMAdaptiveRegimeClassifier initialized (static mode)")

    def classify(self, features: Dict, timestamp: pd.Timestamp) -> Dict:
        """Compatible with AdaptiveRegimeClassifier.classify() interface."""
        if self.engine:
            return self.engine.classify(features, timestamp)
        else:
            return self._classify_static(timestamp)

    def classify_series(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compatible with AdaptiveRegimeClassifier.classify_series() interface."""
        if self.engine:
            return self.engine.classify_series(df)
        else:
            return self._classify_series_static(df)
```

### 4.2 Integration Points

**Files to Modify:**

1. **`bin/backtest_full_engine_replay.py`** (Line 178-181):
   ```python
   # OLD:
   self.adaptive_regime = AdaptiveRegimeClassifier(
       enable_adaptive=self.enable_adaptive_regime,
       fallback_to_static=True
   )

   # NEW:
   from engine.context.hmm_adaptive_classifier import HMMAdaptiveRegimeClassifier

   self.adaptive_regime = HMMAdaptiveRegimeClassifier(
       enable_adaptive=self.enable_adaptive_regime,
       enable_hmm=self.config.get('enable_hmm_regime', True),  # NEW FLAG
       hmm_model_path='models/hmm_regime_v2_simplified.pkl',
       fallback_to_static=True
   )
   ```

2. **`engine/context/regime_manager.py`** (Line 99-105):
   ```python
   # OLD:
   self.classifier = AdaptiveRegimeClassifier(...)

   # NEW:
   from engine.context.hmm_adaptive_classifier import HMMAdaptiveRegimeClassifier

   self.classifier = HMMAdaptiveRegimeClassifier(
       config=config,
       enable_adaptive=True,
       enable_hmm=True,  # TOGGLE HERE
       fallback_to_static=self.fallback_to_static
   )
   ```

3. **New Config Flag** (`configs/production_config.json`):
   ```json
   {
       "regime_detection": {
           "enable_adaptive": true,
           "enable_hmm": true,
           "hmm_model_path": "models/hmm_regime_v2_simplified.pkl",
           "fallback_to_rule_based": true
       }
   }
   ```

### 4.3 Regime Metadata Fix

**Problem:** Trades currently save `regime: unknown`

**Root Cause:** Regime not passed to trade metadata in `_close_position()`

**Fix in `bin/backtest_full_engine_replay.py` (Line 795):**
```python
# BEFORE:
regime=position.metadata.get('regime', 'unknown'),  # Always returns 'unknown'

# AFTER:
regime=signal['regime'],  # Use regime from signal metadata
```

**Additional Change in `_process_signal()` (Line 602):**
```python
# Save regime to pending order metadata
order = PendingOrder(
    archetype_id=archetype_id,
    direction=direction,
    entry_bar_index=bar_index + 1,
    signal_bar_index=bar_index,
    signal_time=timestamp,
    confidence=confidence,
    metadata={
        **signal.get('metadata', {}),
        'regime': signal['regime'],  # ADD THIS
        'regime_confidence': regime_confidence  # ADD THIS
    }
)
```

---

## 5. Implementation Plan

### Phase 1: Foundation (Week 1)

**Tasks:**
1. Create `engine/context/hmm_adaptive_classifier.py`
2. Implement `HMMRegimeScorer` class
3. Implement `HMMAdaptiveRegimeClassifier` wrapper
4. Write unit tests for HMM scorer
5. Verify feature extraction works correctly

**Deliverable:** Working HMM classifier with backward-compatible interface

**Test:**
```bash
python -m pytest tests/unit/context/test_hmm_adaptive_classifier.py
```

---

### Phase 2: Integration (Week 2)

**Tasks:**
1. Add config flag `enable_hmm_regime` to backtest
2. Modify `RegimeManager` to support HMM mode
3. Fix regime metadata bug (save regime to trades)
4. Add HMM mode toggle to CLI arguments
5. Integration tests with backtest engine

**Deliverable:** End-to-end integration with toggle flag

**Test:**
```bash
# Test HMM mode
python bin/backtest_full_engine_replay.py --enable-hmm

# Test rule-based fallback
python bin/backtest_full_engine_replay.py --disable-hmm
```

---

### Phase 3: Validation (Week 3)

**Tasks:**
1. Run parallel backtests (HMM vs Rule-based)
2. Compare regime classification accuracy
3. Compare PnL performance
4. Compare transition frequency
5. Benchmark inference speed

**Deliverable:** Validation report with A/B comparison

**Metrics to Compare:**
- Regime distribution (% time in each regime)
- Transition count (per year)
- Crisis detection lag (vs known events)
- PnL impact (same signals, different regimes)
- Inference latency (ms per classification)

**Test Script:**
```bash
python bin/validate_hmm_vs_rule_based.py \
    --start 2022-01-01 \
    --end 2024-12-31 \
    --output results/hmm_validation/
```

---

### Phase 4: Production Rollout (Week 4)

**Tasks:**
1. Set `enable_hmm_regime=True` as default
2. Update documentation
3. Monitor production performance
4. Prepare rollback plan if issues arise

**Rollout Strategy:**
- Start with paper trading (1 week)
- Monitor regime transitions and PnL
- If metrics improve → enable in live trading
- If metrics degrade → rollback to rule-based

---

## 6. Testing Strategy

### 6.1 Unit Tests

**File:** `tests/unit/context/test_hmm_adaptive_classifier.py`

```python
def test_hmm_scorer_initialization():
    """Test HMM scorer loads model correctly."""
    scorer = HMMRegimeScorer('models/hmm_regime_v2_simplified.pkl')
    assert scorer.hmm.model is not None
    assert len(scorer.feature_order) == 8

def test_hmm_scorer_compute_scores():
    """Test HMM scorer computes valid regime scores."""
    scorer = HMMRegimeScorer('models/hmm_regime_v2_simplified.pkl')

    features = {
        'funding_Z': -1.5,
        'oi_change_pct_24h': -10.0,
        'rv_20d': 0.65,
        'USDT.D': 6.5,
        'BTC.D': 48.0,
        'VIX_Z': 1.2,
        'DXY_Z': 0.8,
        'YC_SPREAD': -50.0
    }

    scores = scorer.compute_scores(features)

    assert 'crisis_score' in scores
    assert 'risk_off_score' in scores
    assert 0.0 <= scores['crisis_score'] <= 1.0
    assert sum(scores.values()) <= 4.0  # Probabilities don't need to sum to 1

def test_hmm_classifier_interface_compatibility():
    """Test HMM classifier maintains backward compatibility."""
    classifier = HMMAdaptiveRegimeClassifier(enable_hmm=True)

    features = {...}  # Same as above
    timestamp = pd.Timestamp('2024-06-15')

    result = classifier.classify(features, timestamp)

    # Check interface compliance
    assert 'regime' in result
    assert result['regime'] in ['crisis', 'risk_off', 'neutral', 'risk_on']
    assert 'confidence' in result
    assert 0.0 <= result['confidence'] <= 1.0
    assert 'proba' in result
    assert 'transition' in result
```

### 6.2 Integration Tests

**Test Scenarios:**

1. **Smoke Test** (2022 Q2):
   - Run 3-month backtest with HMM
   - Verify no crashes
   - Verify trades have regime metadata

2. **Crisis Detection Test** (May 2022 - Luna crash):
   - Verify HMM detects crisis within 24 hours
   - Compare to rule-based (should detect within 6 hours)

3. **Transition Test** (2023 Q1 - Bull recovery):
   - Count regime transitions
   - Verify hysteresis prevents thrashing (< 40 transitions/year)

4. **Feature Missing Test**:
   - Remove one HMM feature
   - Verify graceful degradation (fills with 0)

### 6.3 Performance Benchmarks

**Metrics:**

1. **Inference Speed:**
   - Rule-based: ~0.1ms per classification
   - HMM target: < 1ms per classification
   - Benchmark: 10,000 classifications

2. **Memory Usage:**
   - Rule-based: ~50 MB
   - HMM target: < 200 MB (model + scaler)

3. **Accuracy (vs labeled data):**
   - Crisis detection: > 85% recall
   - Regime alignment: > 70% accuracy

---

## 7. Risk Assessment

### 7.1 Critical Risks

| Risk | Impact | Probability | Mitigation |
|------|--------|-------------|------------|
| **HMM lag in crisis detection** | High | Medium | Keep event override layer (Layer 1) |
| **Transition thrashing** | High | Low | Keep hysteresis layer (Layer 3) |
| **Feature missing in live data** | High | Low | Graceful degradation (fill 0) + alerts |
| **HMM overfitting to train period** | Medium | Medium | Walk-forward validation, regular retraining |
| **Performance degradation** | Medium | Low | Benchmark + rollback plan |
| **Interface breaking changes** | High | Low | Strict interface compliance tests |

### 7.2 Validation Gates

**Gate 1: Unit Tests**
- ✅ All unit tests pass
- ✅ Interface compatibility verified

**Gate 2: Backtest Validation**
- ✅ PnL >= rule-based baseline
- ✅ Transitions < 40/year
- ✅ Crisis detection lag < 24 hours
- ✅ No runtime errors

**Gate 3: Paper Trading**
- ✅ 1 week paper trading with no issues
- ✅ Regime metadata saves correctly
- ✅ Performance metrics stable

**Gate 4: Production**
- ✅ Gradual rollout (10% → 50% → 100%)
- ✅ Monitor regime transitions
- ✅ Monitor PnL impact

---

## 8. Rollback Plan

### 8.1 Immediate Rollback (< 5 minutes)

**Trigger:** Critical failure (crashes, wrong regimes, PnL collapse)

**Action:**
```bash
# Set config flag
echo '{"enable_hmm_regime": false}' > configs/override.json

# Restart backtest
python bin/backtest_full_engine_replay.py --config configs/override.json
```

**Result:** System reverts to rule-based regime detection

### 8.2 Gradual Rollback (< 1 hour)

**Trigger:** Performance degradation (lower PnL, more transitions)

**Action:**
1. Disable HMM for new positions (keep existing)
2. Monitor for 24 hours
3. If stable → full rollback
4. If unstable → investigate root cause

### 8.3 Code Rollback (< 1 day)

**Trigger:** Unfixable bugs, architectural issues

**Action:**
```bash
# Revert to previous commit
git revert <hmm_integration_commit>

# Redeploy
./deploy.sh
```

---

## 9. Success Criteria

### 9.1 Technical Success

- ✅ All unit tests pass
- ✅ All integration tests pass
- ✅ Inference speed < 1ms per classification
- ✅ Zero runtime errors in 1M+ classifications
- ✅ Regime metadata saves correctly to trades

### 9.2 Performance Success

- ✅ PnL >= rule-based baseline (no degradation)
- ✅ Transition count: 10-40/year (not thrashing)
- ✅ Crisis detection lag: < 24 hours (vs known events)
- ✅ Regime alignment: > 70% accuracy (vs labeled data)

### 9.3 Operational Success

- ✅ 1 week paper trading with no issues
- ✅ Easy toggle between HMM and rule-based
- ✅ Clear monitoring dashboards
- ✅ Documentation complete

---

## 10. File-by-File Changes

### New Files

1. **`engine/context/hmm_adaptive_classifier.py`** (NEW)
   - `HMMRegimeScorer` class
   - `HMMAdaptiveRegimeClassifier` class
   - ~300 lines

2. **`tests/unit/context/test_hmm_adaptive_classifier.py`** (NEW)
   - Unit tests for HMM scorer
   - Interface compatibility tests
   - ~200 lines

3. **`bin/validate_hmm_vs_rule_based.py`** (NEW)
   - A/B comparison script
   - Parallel backtest runner
   - ~150 lines

### Modified Files

1. **`bin/backtest_full_engine_replay.py`**
   - Line 178-181: Switch to `HMMAdaptiveRegimeClassifier`
   - Line 602: Add regime to order metadata
   - Line 795: Fix regime metadata bug
   - ~10 line changes

2. **`engine/context/regime_manager.py`**
   - Line 99-105: Support HMM classifier
   - Add config flag for HMM mode
   - ~5 line changes

3. **`configs/production_config.json`** (NEW SECTION)
   - Add `regime_detection` section
   - ~10 lines

### Unchanged Files (Interface Preserved)

- `engine/context/adaptive_regime_model.py` - Keep as fallback
- `engine/archetypes/logic_v2_adapter.py` - No changes needed
- `engine/risk/circuit_breaker.py` - No changes needed
- `engine/risk/direction_balance.py` - No changes needed

---

## 11. Timeline Estimate

| Phase | Duration | Dependencies | Risk |
|-------|----------|--------------|------|
| **Phase 1: Foundation** | 3-5 days | None | Low |
| **Phase 2: Integration** | 3-5 days | Phase 1 | Medium |
| **Phase 3: Validation** | 5-7 days | Phase 2 | Medium |
| **Phase 4: Production Rollout** | 7-14 days | Phase 3 | High |
| **Total** | **3-4 weeks** | - | - |

**Critical Path:** Validation phase (must pass all gates)

---

## 12. Next Steps

### Immediate Actions (DO NOT IMPLEMENT YET)

1. ⏸️ **WAIT** for validation agent results
2. 📋 Review this design document
3. 🤔 Discuss approach with team
4. ✅ Get approval before proceeding

### After Approval

1. Create feature branch: `feature/hmm-regime-integration`
2. Implement Phase 1 (Foundation)
3. Write unit tests
4. Request code review
5. Proceed to Phase 2

---

## Appendix A: HMM Feature Mapping

| HMM Feature | Data Column | Availability | Notes |
|-------------|-------------|--------------|-------|
| `funding_Z` | `funding_Z` | ✅ Available | 30-day z-score of funding rate |
| `oi_change_pct_24h` | `oi_change_pct_24h` | ✅ Available | 24h open interest % change |
| `rv_20d` | `rv_20d` | ✅ Available | 21-day realized volatility |
| `USDT.D` | `USDT.D` | ✅ Available | USDT dominance % |
| `BTC.D` | `BTC.D` | ✅ Available | BTC dominance % |
| `VIX_Z` | `VIX_Z` | ✅ Available | VIX z-score (252d window) |
| `DXY_Z` | `DXY_Z` | ✅ Available | DXY z-score (252d window) |
| `YC_SPREAD` | `YC_SPREAD` | ✅ Available | 10Y - 2Y yield spread (bps) |

**Total:** 8/8 features available (100% coverage)

---

## Appendix B: Interface Specification

### Required Interface (Backward Compatibility)

```python
class IRegimeClassifier(Protocol):
    """Interface that all regime classifiers must implement."""

    def classify(
        self,
        features: Dict[str, float],
        timestamp: Optional[pd.Timestamp] = None
    ) -> Dict[str, Any]:
        """
        Classify regime from features.

        Returns:
            {
                'regime': str,              # 'crisis' | 'risk_off' | 'neutral' | 'risk_on'
                'confidence': float,        # 0.0-1.0
                'proba': Dict[str, float],  # {'crisis': 0.1, 'risk_off': 0.3, ...}
                'transition': bool,         # True if regime changed
                'override': bool,           # True if event override active
                'features_used': int,       # Number of non-null features
                'adaptive': bool            # True if adaptive mode
            }
        """
        ...

    def classify_series(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Classify regime for entire DataFrame.

        Returns:
            DataFrame with columns:
            - regime: regime label
            - confidence: confidence score
            - transition: transition flag
            - override: override flag
            - proba: probability dict
        """
        ...
```

---

## Appendix C: Validation Queries

### Crisis Detection Test

```python
# Test case: May 2022 Luna crash
luna_crash = pd.Timestamp('2022-05-09')

# Get regime classifications
hmm_regimes = hmm_classifier.classify_series(df.loc['2022-05-01':'2022-05-31'])
rule_regimes = rule_classifier.classify_series(df.loc['2022-05-01':'2022-05-31'])

# Check crisis detection lag
hmm_crisis_start = hmm_regimes[hmm_regimes['regime'] == 'crisis'].index[0]
rule_crisis_start = rule_regimes[rule_regimes['regime'] == 'crisis'].index[0]

hmm_lag = (hmm_crisis_start - luna_crash).total_seconds() / 3600
rule_lag = (rule_crisis_start - luna_crash).total_seconds() / 3600

print(f"HMM crisis lag: {hmm_lag:.1f} hours")
print(f"Rule-based crisis lag: {rule_lag:.1f} hours")

# PASS CRITERIA: hmm_lag < 24 hours
assert hmm_lag < 24, f"HMM lag too high: {hmm_lag:.1f}h"
```

---

**END OF DESIGN DOCUMENT**

---

**Status:** 📋 Design Complete - Awaiting Approval
**Next Action:** Wait for validation results, then proceed to Phase 1
**Estimated Completion:** 3-4 weeks after approval
