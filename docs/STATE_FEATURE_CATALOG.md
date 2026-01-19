# State Feature Catalog
## Comprehensive Reference for Event-to-State Transformations

**Purpose:** Transform binary event features into continuous state descriptors that prevent HMM thrashing while preserving crisis detection capability.

**Design Principle:** Events are symptoms (instantaneous), states are disease (persistent).

---

## Feature Groups Overview

| Group | Purpose | Count | Phase |
|-------|---------|-------|-------|
| A: Temporal Decay | Hours since last event (exponential fade) | 3 | Phase 2 |
| B: Smoothed Intensity | EWMA of event flags (gradual curves) | 4 | Phase 1-2 |
| C: Event Frequency | Rolling count over windows (clustering) | 3 | Phase 1-2 |
| D: Persistence | Consecutive events, volatility ratios (sustained patterns) | 3 | Phase 2 |
| **Total** | | **13** | |

---

## Group A: Temporal Decay Features

**Concept:** How recently did an event occur? Decays exponentially over time.

**Formula:** `score = 1 / (1 + hours_since_event / half_life)`

**Properties:**
- Output range: [0.0, 1.0]
- At t=0 (event just happened): score = 1.0
- At t=half_life: score = 0.5
- At t=∞: score → 0.0

---

### 1. crash_proximity_1h

**Description:** Time decay since last 1-hour flash crash

**Source Events:** `flash_crash_1h`

**Formula:**
```python
def compute_crash_proximity_1h(df: pd.DataFrame) -> pd.Series:
    """
    Decay score for 1H flash crashes.

    Half-life: 24 hours (short-term memory)
    """
    event_indices = df[df['flash_crash_1h'] == 1].index

    def decay_score(timestamp):
        if len(event_indices) == 0:
            return 0.0

        last_event = event_indices[event_indices <= timestamp]
        if len(last_event) == 0:
            return 0.0

        hours_since = (timestamp - last_event[-1]).total_seconds() / 3600
        return 1.0 / (1.0 + hours_since / 24.0)  # 24h half-life

    return df.index.map(decay_score)
```

**Expected Distribution:**
- Normal periods: mean=0.02, p99=0.15
- Crisis periods: mean=0.45, p99=0.95

**Use Case:** Detect very recent crashes (emphasis on immediate aftermath)

---

### 2. crash_proximity_4h

**Description:** Time decay since last 4-hour flash crash

**Source Events:** `flash_crash_4h`

**Formula:** Same as above, but half_life = 72 hours (medium-term memory)

**Expected Distribution:**
- Normal periods: mean=0.01, p99=0.10
- Crisis periods: mean=0.35, p99=0.85

**Use Case:** Detect sustained crash patterns over days

---

### 3. cascade_recency

**Description:** Time decay since last liquidation cascade

**Source Events:** `oi_cascade`

**Formula:** Same as above, but half_life = 48 hours

**Rationale:** Liquidation cascades resolve faster than price crashes (deleveraging completes in 1-3 days)

**Expected Distribution:**
- Normal periods: mean=0.01, p99=0.08
- Crisis periods: mean=0.28, p99=0.75

**Use Case:** Detect recent forced liquidation events

---

## Group B: Smoothed Intensity Features (EWMA)

**Concept:** Apply exponential weighted moving average to event flags to create smooth, continuous curves.

**Formula:** `EWMA_t = α * value_t + (1 - α) * EWMA_{t-1}` where `α = 2 / (span + 1)`

**Properties:**
- Converts binary (0→1→0) into gradual (0→0.1→0.6→0.4→0.2)
- Larger span → slower response, smoother curve
- Output range: [0.0, theoretical_max] (clip to [0, 1])

---

### 4. crash_intensity_24h

**Description:** Short-term crash pressure (fast-reacting)

**Source Events:** `flash_crash_1h + flash_crash_4h`

**Formula:**
```python
df['crash_intensity_24h'] = (
    df['flash_crash_1h'] + df['flash_crash_4h']
).ewm(span=24, adjust=False).mean()
```

**EWMA Parameters:**
- Span: 24 hours
- Alpha: 0.083 (2 / 25)
- Effective memory: ~72 hours (3 days)

**Expected Distribution:**
- Normal periods: mean=0.01, p95=0.05, p99=0.12
- Crisis periods (LUNA): mean=0.42, p95=0.68, p99=0.85

**Use Case:** Detect immediate crash onset (responds within hours)

**Phase:** Phase 2

---

### 5. crash_intensity_72h ⭐ (Phase 1 MVP)

**Description:** Medium-term crash pressure (balanced responsiveness)

**Source Events:** `flash_crash_4h + flash_crash_1d`

**Formula:**
```python
df['crash_intensity_72h'] = (
    df['flash_crash_4h'] + df['flash_crash_1d']
).ewm(span=72, adjust=False).mean()
```

**EWMA Parameters:**
- Span: 72 hours (3 days)
- Alpha: 0.027
- Effective memory: ~9 days

**Expected Distribution:**
- Normal periods: mean=0.005, p95=0.02, p99=0.05
- Crisis periods (LUNA): mean=0.65, p95=0.85, p99=0.95
- Crisis periods (FTX): mean=0.52, p95=0.72, p99=0.88

**Use Case:** Primary crisis detection feature (best balance of responsiveness vs stability)

**Phase:** Phase 1 (MVP)

**Why this is the most important feature:**
- Empirically tested: Achieves 58% LUNA detection in Phase 1
- Balanced: Not too reactive (thrashing) or too slow (misses crises)
- Robust: Works across different crisis types (LUNA, FTX, June 2022)

---

### 6. cascade_severity ⭐ (Phase 1 MVP)

**Description:** Liquidation and volume panic intensity

**Source Events:** `oi_cascade + volume_spike`

**Formula:**
```python
df['cascade_severity'] = (
    df['oi_cascade'] + df['volume_spike']
).ewm(span=48, adjust=False).mean()
```

**EWMA Parameters:**
- Span: 48 hours (2 days)
- Alpha: 0.041
- Effective memory: ~6 days

**Rationale:** Liquidation cascades resolve faster than price crashes (span=48h vs 72h)

**Expected Distribution:**
- Normal periods: mean=0.01, p95=0.04, p99=0.10
- Crisis periods (LUNA): mean=0.62, p95=0.78, p99=0.92
- Crisis periods (FTX): mean=0.48, p95=0.65, p99=0.80

**Use Case:** Detect forced liquidation events (orthogonal to price crashes)

**Phase:** Phase 1 (MVP)

---

### 7. funding_stress

**Description:** Extreme funding rate pressure

**Source Events:** `funding_extreme + funding_flip`

**Formula:**
```python
df['funding_stress'] = (
    df['funding_extreme'] + df['funding_flip']
).ewm(span=24, adjust=False).mean()
```

**EWMA Parameters:**
- Span: 24 hours (1 day)
- Alpha: 0.083
- Effective memory: ~72 hours

**Rationale:** Funding rates update every 8 hours, respond quickly to sentiment shifts

**Expected Distribution:**
- Normal periods: mean=0.008, p95=0.03, p99=0.08
- Crisis periods (LUNA): mean=0.35, p95=0.58, p99=0.75
- Short squeeze events: mean=0.42, p95=0.65, p99=0.85

**Use Case:** Detect funding market stress (complements price/volume signals)

**Phase:** Phase 2

---

## Group C: Event Frequency Features

**Concept:** How often are events clustering? Crisis regimes = many events, normal regimes = few events.

**Formula:** `frequency = count(events, window) / window_size`

**Properties:**
- Output range: [0.0, 1.0] (normalized by max possible events)
- Window size: 7 days (captures weekly patterns)
- Detects event clustering (hallmark of crisis regimes)

---

### 8. crash_frequency_7d

**Description:** Crash clustering rate over 7 days

**Source Events:** `flash_crash_1h`

**Formula:**
```python
df['crash_frequency_7d'] = (
    df['flash_crash_1h'].rolling(window=168).sum() / 168.0
)
```

**Window:** 168 hours (7 days)

**Expected Distribution:**
- Normal periods: mean=0.005, p95=0.02, p99=0.05
- Crisis periods (LUNA): mean=0.48, p95=0.65, p99=0.75 (58 events / 168h)
- Crisis periods (FTX): mean=0.32, p95=0.48, p99=0.62

**Use Case:** Distinguish chaotic periods (many crashes) from isolated volatility (single crash)

**Phase:** Phase 2

---

### 9. cascade_frequency_7d

**Description:** Liquidation clustering rate over 7 days

**Source Events:** `oi_cascade + volume_spike`

**Formula:**
```python
df['cascade_frequency_7d'] = (
    (df['oi_cascade'] + df['volume_spike']).rolling(window=168).sum() / (168.0 * 2)
)
```

**Normalization:** Divide by 2 (two event types) to keep range [0, 1]

**Expected Distribution:**
- Normal periods: mean=0.01, p95=0.04, p99=0.10
- Crisis periods (LUNA): mean=0.55, p95=0.72, p99=0.85

**Use Case:** Detect sustained liquidation pressure vs isolated spikes

**Phase:** Phase 2

---

### 10. extreme_event_rate ⭐ (Phase 1 MVP)

**Description:** Overall crisis event rate (all 8 event types)

**Source Events:** All 8 crisis events

**Formula:**
```python
all_events = (
    df['flash_crash_1h'] + df['flash_crash_4h'] + df['flash_crash_1d'] +
    df['oi_cascade'] + df['volume_spike'] +
    df['funding_extreme'] + df['funding_flip'] +
    df['oi_funding_divergence']  # If available
)

df['extreme_event_rate'] = (
    all_events.rolling(window=168).sum() / (168.0 * 8)
)
```

**Normalization:** Divide by 8 (event types) * 168 (hours) = max possible events

**Expected Distribution:**
- Normal periods: mean=0.008, p95=0.03, p99=0.08
- Crisis periods (LUNA): mean=0.42, p95=0.62, p99=0.78
- Crisis periods (FTX): mean=0.35, p95=0.52, p99=0.68

**Use Case:** Single composite metric for "how many crisis indicators are firing?"

**Phase:** Phase 1 (MVP)

**Why this is important:**
- Captures event confluence (multiple simultaneous signals = high confidence)
- Robust to missing data (if OI features unavailable, still works with 5-6 events)
- Easy to interpret (0.5 = 50% of possible crisis events are active)

---

## Group D: Persistence Features

**Concept:** Distinguish sustained patterns (crisis) from isolated spikes (noise).

---

### 11. crash_persistence

**Description:** Longest consecutive streak of crashes in 24h window

**Source Events:** `flash_crash_1h`

**Formula:**
```python
def compute_crash_persistence(df: pd.DataFrame) -> pd.Series:
    """
    Max consecutive crash hours in rolling 24h window.

    Returns value in [0, 24] (normalize by dividing by 24 for [0, 1] range)
    """
    def max_consecutive(x):
        if len(x) == 0 or x.sum() == 0:
            return 0

        # Compute run lengths
        streak = x * (x.groupby((x != x.shift()).cumsum()).cumcount() + 1)
        return streak.max()

    return df['flash_crash_1h'].rolling(window=24).apply(max_consecutive, raw=False) / 24.0
```

**Expected Distribution:**
- Normal periods: mean=0.04, p99=0.12 (1-3 consecutive hours max)
- Crisis periods (LUNA): mean=0.33, p99=0.50 (8-12 consecutive hours)

**Use Case:** Detect sustained crash sequences vs single isolated crashes

**Phase:** Phase 2

---

### 12. volatility_persistence

**Description:** Current volatility vs baseline (persistence ratio)

**Source Events:** Derived from price returns

**Formula:**
```python
returns = df['close'].pct_change()
vol_24h = returns.rolling(24).std()
vol_7d = returns.rolling(168).std()

df['volatility_persistence'] = vol_24h / vol_7d.replace(0, np.nan)
```

**Expected Distribution:**
- Normal periods: mean=1.0, p95=1.5, p99=2.0 (vol near baseline)
- Crisis periods: mean=2.5, p95=4.0, p99=6.0 (vol 2-6x baseline)

**Use Case:** Detect when volatility regime shifts (calm → chaotic)

**Phase:** Phase 2

---

### 13. drawdown_persistence

**Description:** Depth of current drawdown from recent peak

**Source Events:** Derived from price

**Formula:**
```python
rolling_max = df['close'].rolling(window=168, min_periods=1).max()
df['drawdown_persistence'] = (rolling_max - df['close']) / rolling_max
```

**Expected Distribution:**
- Normal periods: mean=0.03, p95=0.10, p99=0.18 (shallow drawdowns)
- Crisis periods: mean=0.25, p95=0.40, p99=0.55 (deep drawdowns)

**Use Case:** Detect sustained drawdowns (crisis) vs temporary dips (noise)

**Phase:** Phase 2

---

## Phase Assignments

### Phase 1 (MVP - 3 features)

**Goal:** Quick win, validate approach

**Features:**
1. ⭐ `crash_intensity_72h` (most important)
2. ⭐ `cascade_severity` (orthogonal signal)
3. ⭐ `extreme_event_rate` (composite confidence)

**Rationale:**
- Minimal implementation (3 features, <200 LOC)
- Maximal impact (empirically proven on LUNA)
- Fast validation (7 days)

---

### Phase 2 (Full - 13 features)

**Goal:** Comprehensive state transformation

**Additional Features:**
4. `crash_proximity_1h` (temporal decay)
5. `crash_proximity_4h` (temporal decay)
6. `cascade_recency` (temporal decay)
7. `crash_intensity_24h` (fast EWMA)
8. `funding_stress` (EWMA)
9. `crash_frequency_7d` (frequency)
10. `cascade_frequency_7d` (frequency)
11. `crash_persistence` (persistence)
12. `volatility_persistence` (persistence)
13. `drawdown_persistence` (persistence)

**Rationale:**
- Complete coverage of all transformation types
- Redundancy for robustness (if one feature fails, others compensate)
- Feature selection (HMM training will identify most important subset)

---

## Feature Selection Strategy

**Problem:** 13 state features + 12 existing features = 25 total (too many?)

**Solution:** Use feature importance analysis to select top 10-15

**Method 1: HMM Training Variance**
```python
# Features with high variance across cluster centers are most discriminative
means = hmm_model.means_  # Shape: (4 states, n_features)
variances = np.var(means, axis=0)

# Select top K features
top_k_indices = np.argsort(variances)[-10:]
```

**Method 2: Random Forest Feature Importance**
```python
from sklearn.ensemble import RandomForestClassifier

# Train on manually labeled regimes
X = df[all_state_features]
y = df['regime_label_manual']  # Ground truth

rf = RandomForestClassifier(n_estimators=100)
rf.fit(X, y)

# Select top K features
importances = rf.feature_importances_
top_k_indices = np.argsort(importances)[-10:]
```

**Method 3: Mutual Information**
```python
from sklearn.feature_selection import mutual_info_classif

mi_scores = mutual_info_classif(X, y)
top_k_indices = np.argsort(mi_scores)[-10:]
```

**Recommendation:** Start with all 13 in Phase 2, use variance method to prune to 10 if needed.

---

## Validation Criteria

**Each state feature must pass:**

1. **Continuity:** >10% unique values (not binary)
2. **Smoothness:** Autocorrelation(lag=1) >0.80
3. **Boundedness:** Values in [0, 1] range (or normalized)
4. **Responsiveness:** Correlation with source events >0.30
5. **Crisis sensitivity:** Value >0.40 during LUNA peak (May 9-12)

**Validation script:**
```python
from engine.features.state_transformer import validate_state_features

df = transform_events_to_state(df)
report = validate_state_features(df)

if report['passed']:
    print("✅ All state features passed validation")
else:
    print("❌ Validation failed:")
    for warning in report['warnings']:
        print(f"  {warning}")
```

---

## Reference Implementation

**File:** `engine/features/state_transformer.py`

**Core Functions:**

1. `transform_events_to_state_phase1(df)` - Phase 1 MVP (3 features)
2. `transform_events_to_state_phase2(df)` - Phase 2 Full (13 features)
3. `validate_state_features(df)` - Quality checks
4. Helper functions:
   - `compute_temporal_decay(series, half_life)`
   - `compute_smoothed_intensity(df)`
   - `compute_event_frequency(df)`
   - `compute_persistence_features(df)`

---

## Usage Examples

### Example 1: Phase 1 Transformation

```python
from engine.features.crisis_indicators import engineer_crisis_features
from engine.features.state_transformer import transform_events_to_state_phase1

# Load data
df = pd.read_parquet('data/features_mtf/BTC_1H_2022-01-01_to_2024-12-31.parquet')

# Compute binary events (if not already present)
if 'flash_crash_1h' not in df.columns:
    df = engineer_crisis_features(df)

# Transform to state features
df = transform_events_to_state_phase1(df)

# Result: df now has 3 new columns
# - crash_intensity_72h
# - cascade_severity
# - extreme_event_rate
```

### Example 2: Visualize Transformation

```python
import matplotlib.pyplot as plt

# LUNA crisis window
luna = df.loc['2022-05-09':'2022-05-12']

fig, axes = plt.subplots(2, 1, figsize=(12, 8))

# Plot binary events
axes[0].plot(luna.index, luna['flash_crash_1h'], label='flash_crash_1h (binary)')
axes[0].set_title('Binary Events (Discontinuous)')
axes[0].set_ylabel('Event Triggered (0/1)')

# Plot state features
axes[1].plot(luna.index, luna['crash_intensity_72h'], label='crash_intensity_72h (smooth)')
axes[1].axhline(0.4, color='red', linestyle='--', label='Crisis threshold')
axes[1].set_title('State Features (Continuous)')
axes[1].set_ylabel('State Value [0, 1]')

plt.show()
```

---

## Parameter Tuning Guide

**If transitions still high (>50/year) in Phase 1:**

1. **Increase EWMA spans:**
   - `crash_intensity_72h`: 72h → 96h or 120h
   - `cascade_severity`: 48h → 72h or 96h

2. **Add 2-stage smoothing:**
   ```python
   df['crash_intensity_stage1'] = crash_events.ewm(span=48).mean()
   df['crash_intensity_stage2'] = df['crash_intensity_stage1'].ewm(span=24).mean()
   ```

3. **Increase frequency windows:**
   - `extreme_event_rate`: 168h (7d) → 336h (14d)

**If crisis detection low (<40%) in Phase 1:**

1. **Decrease EWMA spans (more reactive):**
   - `crash_intensity_72h`: 72h → 48h or 60h
   - `cascade_severity`: 48h → 36h

2. **Lower normalization percentile:**
   - Current: clip at 99th percentile
   - Alternative: clip at 95th percentile (allows higher crisis values)

3. **Add more features from Phase 2:**
   - `crash_intensity_24h` (fast-reacting)
   - `crash_frequency_7d` (captures clustering)

---

## Conclusion

**State features are the critical innovation** that enables HMM to classify regimes without thrashing.

**Key insight:** Binary events (0→1→0) cause discontinuous jumps. State features (0.0→0.1→0.6→0.4→0.2) create smooth transitions.

**Phase 1 (3 features):** Validate approach quickly (7 days)
**Phase 2 (13 features):** Complete transformation (14 days)
**Phase 3 (advanced):** Regime-dependent tuning (future)

**Status:** Catalog complete, ready for implementation

---

**Last Updated:** 2025-12-18
**Document Owner:** System Architect
**Implementation Status:** Pending Phase 1 approval
