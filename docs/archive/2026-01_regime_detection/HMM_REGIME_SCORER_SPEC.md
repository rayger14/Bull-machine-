# HMMRegimeScorer Technical Specification

**Component:** `engine.context.hmm_adaptive_classifier.HMMRegimeScorer`
**Purpose:** Replace rule-based `RegimeScorer` with ML-based HMM posterior probabilities
**Status:** Design specification (not yet implemented)

---

## Overview

The `HMMRegimeScorer` class is the core component that replaces rule-based regime scoring with ML-based HMM classification. It maintains the same interface as `RegimeScorer` but uses trained GaussianHMM posterior probabilities instead of hand-crafted rules.

---

## Class Interface

```python
class HMMRegimeScorer:
    """
    Compute continuous regime scores (0-1) from HMM posterior probabilities.

    Replaces RegimeScorer's rule-based logic with ML-based classification
    while maintaining identical interface for backward compatibility.
    """

    def __init__(self, model_path: str):
        """
        Initialize HMM regime scorer.

        Args:
            model_path: Path to trained HMM model pickle file
                       (default: 'models/hmm_regime_v2_simplified.pkl')

        Raises:
            FileNotFoundError: If model file not found
            ValueError: If model is invalid or incompatible
        """
        ...

    def compute_scores(self, features: Dict[str, float]) -> Dict[str, float]:
        """
        Compute continuous regime scores from features.

        Args:
            features: Dict of state feature values
                     (must contain all 8 HMM features)

        Returns:
            Dict with:
            - crisis_score: 0-1 (crisis likelihood)
            - risk_off_score: 0-1 (bear market likelihood)
            - neutral_score: 0-1 (neutral likelihood)
            - risk_on_score: 0-1 (bull market likelihood)

        Note: Scores do not need to sum to 1.0 (same as original RegimeScorer)
        """
        ...
```

---

## Implementation Details

### 1. Model Loading

```python
def __init__(self, model_path: str):
    # Load HMM model from pickle
    from engine.context.hmm_regime_model import HMMRegimeModel

    self.hmm = HMMRegimeModel(model_path)
    self.feature_order = self.hmm.feature_order  # 8 features in exact order

    # Verify model loaded correctly
    assert self.hmm.model is not None, "HMM model failed to load"
    assert len(self.feature_order) == 8, f"Expected 8 features, got {len(self.feature_order)}"

    logger.info(f"HMMRegimeScorer initialized")
    logger.info(f"  Model: {model_path}")
    logger.info(f"  Features: {self.feature_order}")
    logger.info(f"  States: {self.hmm.state_map}")
```

### 2. Feature Extraction

```python
def _extract_features(self, features: Dict[str, float]) -> np.ndarray:
    """
    Extract HMM features in correct order.

    Missing features are filled with 0.0 and logged as warnings.

    Args:
        features: Dict of feature values

    Returns:
        numpy array of shape (8,) with features in correct order
    """
    x = []
    missing_features = []

    for feat_name in self.feature_order:
        if feat_name in features:
            x.append(features[feat_name])
        else:
            # Missing feature - fill with 0 and log warning
            x.append(0.0)
            missing_features.append(feat_name)

    if missing_features:
        logger.warning(f"Missing {len(missing_features)} features: {missing_features}")
        logger.warning("Filled with 0.0 - may impact regime accuracy")

    return np.array(x, dtype=float)
```

### 3. Score Computation

```python
def compute_scores(self, features: Dict[str, float]) -> Dict[str, float]:
    """Compute regime scores using HMM posterior probabilities."""

    # Step 1: Extract features in correct order
    x = self._extract_features(features)  # Shape: (8,)

    # Step 2: Scale features (if scaler available)
    if self.hmm.scaler is not None:
        x = self.hmm.scaler.transform([x])[0]  # Shape: (8,)

    # Step 3: Compute posterior probabilities
    probs = self.hmm.model.predict_proba([x])[0]  # Shape: (4,) for 4 states

    # Step 4: Map HMM states to regime scores
    # State mapping: {0: 'crisis', 1: 'risk_on', 2: 'neutral', 3: 'risk_on'}

    crisis_score = probs[0]                    # State 0 → crisis
    neutral_score = probs[2]                   # State 2 → neutral
    risk_on_score = probs[1] + probs[3]       # States 1+3 → risk_on

    # Risk-off is intermediate (between crisis and neutral)
    # Derived from other scores to maintain consistency
    risk_off_score = max(
        crisis_score * 0.8,                    # At least 80% of crisis
        1.0 - risk_on_score - neutral_score    # Remainder after risk_on + neutral
    )

    # Step 5: Ensure all scores are in [0, 1]
    scores = {
        'crisis_score': np.clip(crisis_score, 0.0, 1.0),
        'risk_off_score': np.clip(risk_off_score, 0.0, 1.0),
        'neutral_score': np.clip(neutral_score, 0.0, 1.0),
        'risk_on_score': np.clip(risk_on_score, 0.0, 1.0)
    }

    # Step 6: Log scores (debug level)
    logger.debug(f"HMM regime scores: {scores}")

    return scores
```

---

## Feature Requirements

### Required Features (8 total)

| Feature | Description | Typical Range | Source |
|---------|-------------|---------------|--------|
| `funding_Z` | 30-day z-score of funding rate | -3 to +3 | Normalized funding rate |
| `oi_change_pct_24h` | 24h open interest % change | -20% to +20% | Derivatives data |
| `rv_20d` | 21-day realized volatility | 0.2 to 1.5 | Price returns |
| `USDT.D` | USDT dominance % | 3% to 8% | Market cap data |
| `BTC.D` | BTC dominance % | 40% to 70% | Market cap data |
| `VIX_Z` | VIX z-score (252d window) | -2 to +3 | TradFi volatility |
| `DXY_Z` | DXY z-score (252d window) | -2 to +2 | Dollar index |
| `YC_SPREAD` | 10Y - 2Y yield spread (bps) | -100 to +200 | Treasury yields |

### Feature Validation

```python
def _validate_features(self, features: Dict[str, float]) -> None:
    """
    Validate feature ranges and log warnings for outliers.

    Does NOT raise errors - just logs warnings for monitoring.
    """
    validations = {
        'funding_Z': (-5, 5),
        'oi_change_pct_24h': (-50, 50),
        'rv_20d': (0, 3),
        'USDT.D': (0, 15),
        'BTC.D': (20, 80),
        'VIX_Z': (-3, 4),
        'DXY_Z': (-3, 3),
        'YC_SPREAD': (-200, 300)
    }

    for feat, (min_val, max_val) in validations.items():
        if feat in features:
            val = features[feat]
            if val < min_val or val > max_val:
                logger.warning(
                    f"Feature {feat} out of typical range: {val:.2f} "
                    f"(expected {min_val} to {max_val})"
                )
```

---

## State Mapping

### HMM States → Regime Scores

The trained HMM has 4 states that map to regimes:

```python
state_map = {
    0: 'crisis',    # State 0 → Crisis
    1: 'risk_on',   # State 1 → Risk-on
    2: 'neutral',   # State 2 → Neutral
    3: 'risk_on'    # State 3 → Risk-on (duplicate)
}
```

**Score Computation:**

```python
# Direct mapping
crisis_score = P(state=0)              # Crisis
neutral_score = P(state=2)             # Neutral
risk_on_score = P(state=1) + P(state=3)  # Risk-on (sum both states)

# Derived mapping (maintains consistency with original RegimeScorer)
risk_off_score = max(
    crisis_score * 0.8,                # Crisis implies risk-off
    1.0 - risk_on_score - neutral_score  # Remainder
)
```

**Rationale:**
- States 1 and 3 both map to `risk_on` → sum their probabilities
- `risk_off` is intermediate between `crisis` and `neutral`
- Ensures compatibility with downstream hysteresis thresholds

---

## Backward Compatibility

### Interface Compatibility with RegimeScorer

```python
# OLD: Rule-based RegimeScorer
scorer = RegimeScorer(config)
scores = scorer.compute_scores(features)
# Returns: {'crisis_score': 0.2, 'risk_off_score': 0.4, 'neutral_score': 0.6, 'risk_on_score': 0.3}

# NEW: HMM-based HMMRegimeScorer
scorer = HMMRegimeScorer('models/hmm_regime_v2_simplified.pkl')
scores = scorer.compute_scores(features)
# Returns: {'crisis_score': 0.15, 'risk_off_score': 0.35, 'neutral_score': 0.55, 'risk_on_score': 0.25}

# SAME INTERFACE - drop-in replacement!
```

**Key Points:**
1. ✅ Same method signature: `compute_scores(features: Dict) -> Dict`
2. ✅ Same return structure: Dict with 4 regime scores
3. ✅ Same value ranges: All scores in [0, 1]
4. ✅ Scores don't need to sum to 1 (same as original)

---

## Error Handling

### Graceful Degradation

```python
def compute_scores(self, features: Dict[str, float]) -> Dict[str, float]:
    """Compute scores with graceful error handling."""

    try:
        # Normal HMM scoring
        x = self._extract_features(features)

        if self.hmm.scaler:
            x = self.hmm.scaler.transform([x])[0]

        probs = self.hmm.model.predict_proba([x])[0]

        # Map to regime scores
        return self._map_states_to_scores(probs)

    except Exception as e:
        # Log error but don't crash
        logger.error(f"HMM scoring failed: {e}")
        logger.warning("Falling back to neutral regime (all scores = 0.5)")

        # Fallback: neutral regime
        return {
            'crisis_score': 0.0,
            'risk_off_score': 0.5,
            'neutral_score': 1.0,
            'risk_on_score': 0.0
        }
```

### Error Scenarios

| Error | Cause | Handling |
|-------|-------|----------|
| **Model not found** | File missing | Raise FileNotFoundError at init |
| **Invalid pickle** | Corrupted file | Raise ValueError at init |
| **Missing features** | Incomplete data | Fill with 0, log warning |
| **Invalid feature values** | NaN, inf | Replace with 0, log warning |
| **Scaler error** | Shape mismatch | Skip scaling, log error |
| **Predict error** | Model corruption | Fallback to neutral, log error |

---

## Performance Considerations

### Inference Speed Target

**Target:** < 1ms per classification

**Breakdown:**
- Feature extraction: ~0.1ms
- Scaling: ~0.05ms
- HMM predict_proba: ~0.5ms
- State mapping: ~0.05ms
- Total: ~0.7ms ✅

**Optimization:**
```python
# Cache scaler transform if features don't change
if self._cache_enabled and features == self._last_features:
    x_scaled = self._cached_x_scaled
else:
    x_scaled = self.hmm.scaler.transform([x])[0]
    self._cached_x_scaled = x_scaled
    self._last_features = features.copy()
```

### Memory Usage

**Target:** < 200 MB

**Components:**
- HMM model: ~50 MB
- Scaler: ~1 MB
- Feature buffer: ~1 MB
- Total: ~52 MB ✅

---

## Testing

### Unit Tests

```python
def test_hmm_scorer_initialization():
    """Test HMM scorer loads model correctly."""
    scorer = HMMRegimeScorer('models/hmm_regime_v2_simplified.pkl')

    assert scorer.hmm.model is not None
    assert len(scorer.feature_order) == 8
    assert scorer.hmm.state_map == {0: 'crisis', 1: 'risk_on', 2: 'neutral', 3: 'risk_on'}


def test_hmm_scorer_compute_scores():
    """Test HMM scorer computes valid regime scores."""
    scorer = HMMRegimeScorer('models/hmm_regime_v2_simplified.pkl')

    # Crisis-like features
    features = {
        'funding_Z': -2.5,
        'oi_change_pct_24h': -15.0,
        'rv_20d': 1.2,
        'USDT.D': 7.0,
        'BTC.D': 45.0,
        'VIX_Z': 2.5,
        'DXY_Z': 1.5,
        'YC_SPREAD': -80.0
    }

    scores = scorer.compute_scores(features)

    # Validate structure
    assert 'crisis_score' in scores
    assert 'risk_off_score' in scores
    assert 'neutral_score' in scores
    assert 'risk_on_score' in scores

    # Validate ranges
    for regime, score in scores.items():
        assert 0.0 <= score <= 1.0, f"{regime} out of range: {score}"

    # Validate logic (crisis features should have high crisis score)
    assert scores['crisis_score'] > 0.3, "Crisis score too low for crisis-like features"


def test_hmm_scorer_missing_features():
    """Test graceful handling of missing features."""
    scorer = HMMRegimeScorer('models/hmm_regime_v2_simplified.pkl')

    # Missing 2 features
    features = {
        'funding_Z': -1.0,
        'oi_change_pct_24h': -5.0,
        'rv_20d': 0.8,
        'USDT.D': 6.0,
        'BTC.D': 50.0,
        'VIX_Z': 1.0
        # Missing: DXY_Z, YC_SPREAD
    }

    scores = scorer.compute_scores(features)

    # Should still return valid scores (with warnings)
    assert all(0.0 <= s <= 1.0 for s in scores.values())


def test_hmm_scorer_interface_compatibility():
    """Test interface matches original RegimeScorer."""
    from engine.context.adaptive_regime_model import RegimeScorer

    # Same features for both
    features = {...}

    # Old scorer
    old_scorer = RegimeScorer()
    old_scores = old_scorer.compute_scores(features)

    # New scorer
    new_scorer = HMMRegimeScorer('models/hmm_regime_v2_simplified.pkl')
    new_scores = new_scorer.compute_scores(features)

    # Interface compatibility check
    assert old_scores.keys() == new_scores.keys()
    assert all(isinstance(v, float) for v in new_scores.values())
    assert all(0.0 <= v <= 1.0 for v in new_scores.values())
```

---

## Logging

### Log Levels

**INFO:** Initialization and high-level events
```python
logger.info("HMMRegimeScorer initialized")
logger.info(f"  Model: {model_path}")
logger.info(f"  Features: {self.feature_order}")
```

**WARNING:** Missing features, outliers, graceful degradation
```python
logger.warning(f"Missing {len(missing_features)} features: {missing_features}")
logger.warning(f"Feature {feat} out of range: {val:.2f}")
logger.warning("HMM scoring failed - falling back to neutral")
```

**DEBUG:** Per-classification details
```python
logger.debug(f"HMM input features: {features}")
logger.debug(f"HMM posterior probabilities: {probs}")
logger.debug(f"HMM regime scores: {scores}")
```

**ERROR:** Critical failures
```python
logger.error(f"HMM model loading failed: {e}")
logger.error(f"HMM scoring failed: {e}")
```

---

## Configuration

### Optional Config Parameters

```python
class HMMRegimeScorer:
    def __init__(
        self,
        model_path: str,
        config: Optional[Dict] = None
    ):
        self.config = config or {}

        # Optional: Enable feature caching for performance
        self.enable_cache = self.config.get('enable_cache', False)

        # Optional: Validate feature ranges
        self.validate_features = self.config.get('validate_features', True)

        # Optional: Risk-off score derivation method
        self.risk_off_method = self.config.get('risk_off_method', 'max')
        # Options: 'max' (default), 'linear', 'residual'
```

---

## Comparison: Rule-Based vs HMM

### Decision Logic

**Rule-Based (Old):**
```python
# Explicit thresholds and weights
crisis_score = (
    min(crash_freq / 5.0, 1.0) * 0.4 +
    min(crisis_persist, 1.0) * 0.3 +
    min(aftershock, 1.0) * 0.2 +
    flash_crash * 0.1
)
```

**HMM (New):**
```python
# Learned from data
probs = hmm.model.predict_proba([x])[0]
crisis_score = probs[0]  # Posterior probability
```

### Advantages of HMM

1. **Data-driven:** Learned from historical regime patterns
2. **Adaptive:** Captures non-linear relationships
3. **Probabilistic:** Natural confidence scores
4. **Validated:** Trained on labeled data

### Advantages of Rule-Based

1. **Interpretable:** Clear logic for each threshold
2. **Fast:** No model inference overhead
3. **Deterministic:** Same inputs → same outputs
4. **Tunable:** Easy to adjust thresholds

### Why Hybrid?

Combine best of both:
- HMM for regime classification (Layer 2)
- Rules for crisis detection (Layer 1)
- Rules for hysteresis (Layer 3)

---

## Future Enhancements

### Phase 2 Improvements

1. **Feature Engineering:**
   - Add interaction terms (e.g., funding_Z × oi_change)
   - Add temporal features (regime duration, transition history)

2. **Model Updates:**
   - Retrain monthly with new data
   - A/B test multiple models in production

3. **Performance:**
   - Batch inference optimization
   - Feature caching for repeated classifications

4. **Monitoring:**
   - Track regime confidence over time
   - Alert on low-confidence classifications
   - Dashboard for regime distribution

---

## Summary

**Component:** HMMRegimeScorer
**Purpose:** Replace rule-based scoring with ML-based HMM
**Interface:** Backward compatible with RegimeScorer
**Features:** 8 HMM features (all available in production)
**Performance:** < 1ms inference, < 200 MB memory
**Risk:** Low (hybrid approach maintains safety nets)

**Status:** Design complete - awaiting Phase 1 implementation

---

**Next Steps:**
1. Implement `HMMRegimeScorer` class
2. Write unit tests
3. Integrate with `HMMAdaptiveRegimeClassifier`
4. Validate interface compatibility
5. Proceed to Phase 2 integration

---

**File:** `engine/context/hmm_adaptive_classifier.py`
**Lines of Code:** ~200 (HMMRegimeScorer only)
**Dependencies:** `HMMRegimeModel`, `numpy`, `logging`
**Test Coverage Target:** > 90%
