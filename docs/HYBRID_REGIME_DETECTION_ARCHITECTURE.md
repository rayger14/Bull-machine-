# Hybrid Regime Detection Architecture
## Two-Layer Event-to-State Design

**Status:** Architecture Design v1.0
**Date:** 2025-12-18
**Purpose:** Separate fast event detection from slow state classification to eliminate regime thrashing

---

## Executive Summary

### The Problem
Current HMM approach fails because **binary event features** (flash crashes, liquidations) cause **regime thrashing**:
- 117 transitions/year (expected: 10-20)
- 0% crisis detection despite having crisis indicators
- Silhouette score: 0.11 (expected: >0.40)
- Root cause: Binary flags (0/1) create discontinuous jumps that HMM interprets as regime changes

### The Solution
**Two-layer architecture** that separates concerns:

```
Layer 1: EVENT DETECTION (Fast, Binary)
         ↓ transformation
Layer 2: STATE CLASSIFICATION (Slow, Continuous)
         ↓ HMM inference
         REGIME LABELS (crisis, risk_off, neutral, risk_on)
```

**Key Insight:** Events are discrete shocks (0-6 hours). Regimes are persistent states (days to weeks). Don't mix them.

### Success Criteria
- HMM transitions: 117/year → 10-20/year
- Crisis detection: 0% → >60%
- Silhouette score: 0.11 → >0.40
- Implementation time: 1-2 sprints

---

## Architecture Overview

### Component Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                     RAW MARKET DATA                              │
│  (price, volume, OI, funding, liquidations, macro)              │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│                LAYER 1: EVENT DETECTOR                           │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐          │
│  │ Flash Crash  │  │ Liquidation  │  │ Volume Spike │  ...     │
│  │  Detection   │  │   Cascade    │  │  Detection   │          │
│  └──────────────┘  └──────────────┘  └──────────────┘          │
│                                                                  │
│  Output: Binary flags (0 or 1)                                  │
│  Latency: 0-6 hours                                             │
│  Trigger rate: 0.05-2% (rare events)                            │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│            LAYER 1.5: STATE TRANSFORMER                          │
│  ┌──────────────────────────────────────────────────────┐       │
│  │  Convert binary events → continuous state features   │       │
│  │                                                       │       │
│  │  - Hours since last event (decay)                    │       │
│  │  - EWMA of event flags (24-72h smoothing)            │       │
│  │  - Rolling frequency (event count / 7 days)          │       │
│  │  - Persistence scores (consecutive events)           │       │
│  └──────────────────────────────────────────────────────┘       │
│                                                                  │
│  Output: Continuous features (0.0-1.0)                          │
│  Latency: Days to weeks (smoothed)                              │
│  Active rate: 10-30% (frequent signals)                         │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│          LAYER 2: REGIME CLASSIFIER (HMM)                        │
│  ┌──────────────────────────────────────────────────────┐       │
│  │  Input: State features + macro features              │       │
│  │                                                       │       │
│  │  States: [crisis, risk_off, neutral, risk_on]        │       │
│  │  Window: 21 days (504 hours)                         │       │
│  │  Algorithm: Gaussian HMM with Viterbi decoding       │       │
│  └──────────────────────────────────────────────────────┘       │
│                                                                  │
│  Output: Regime label + confidence                              │
│  Latency: Days to weeks (regime-level)                          │
│  Transition frequency: 10-20/year                               │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│              ARCHETYPE SYSTEM (Downstream)                       │
│  - Position sizing (crisis → reduce leverage)                   │
│  - Archetype selection (route to regime-specific systems)       │
│  - Risk management (circuit breakers, kill switches)            │
└─────────────────────────────────────────────────────────────────┘
```

### Data Flow

```
1. Raw Features (hourly bars)
   ├── close, volume, oi, funding, liquidations
   ├── VIX, DXY, yields, dominance
   └── timestamp

2. Event Layer (binary 0/1)
   ├── flash_crash_1h = 1 if price drop >4% in 1H
   ├── flash_crash_4h = 1 if price drop >8% in 4H
   ├── flash_crash_1d = 1 if price drop >12% in 24H
   ├── volume_spike = 1 if volume >3σ
   ├── oi_cascade = 1 if OI drops >5% in 1H
   ├── oi_funding_divergence = 1 if OI↓ + funding↑ (squeeze)
   ├── funding_extreme = 1 if funding >3σ
   └── funding_flip = 1 if rapid sign change

3. State Layer (continuous 0.0-1.0)
   ├── crash_proximity = hours_since_flash_crash / 168h (1 week decay)
   ├── crash_intensity = EWMA(flash_crash_*, alpha=0.1, span=24h)
   ├── crash_frequency = count(flash_crash_1h, window=7d) / 168h
   ├── cascade_severity = EWMA(oi_cascade, alpha=0.05, span=72h)
   ├── funding_stress = max(EWMA(funding_extreme), EWMA(funding_flip))
   ├── volatility_persistence = std(close.pct_change(), window=7d) / mean(7d)
   ├── liquidity_withdrawal = EWMA(oi_cascade + volume_spike, span=48h)
   └── market_fragility = weighted_sum(state_features, learned_weights)

4. HMM Input Features (state + macro)
   ├── State features (7-10 from step 3)
   ├── Macro z-scores (VIX_Z, DXY_Z, YC_SPREAD, M2_GROWTH)
   ├── Market structure (BTC.D, USDT.D, TOTAL_RET_21d)
   └── Normalized to [0, 1] or z-scored

5. Regime Output
   ├── regime_label ∈ {crisis, risk_off, neutral, risk_on}
   ├── regime_confidence ∈ [0.0, 1.0]
   └── regime_proba_{crisis, risk_off, neutral, risk_on}
```

---

## Layer 1: Event Detection (Unchanged)

### Purpose
Detect discrete shock events in real-time (0-6 hours latency).

### Implementation
**Already implemented** in `engine/features/crisis_indicators.py`. **No changes needed.**

### Event Catalog (8 features)

| Feature | Type | Threshold | Trigger Rate | Latency | Use Case |
|---------|------|-----------|--------------|---------|----------|
| `flash_crash_1h` | Binary | Price drop >4% in 1H | 0.5% | 0H | Immediate crash detection |
| `flash_crash_4h` | Binary | Price drop >8% in 4H | 0.3% | 0-4H | Sustained crash detection |
| `flash_crash_1d` | Binary | Price drop >12% in 24H | 0.2% | 0-24H | Crisis confirmation |
| `volume_spike` | Binary | Volume >3σ (7d baseline) | 1.0% | 0H | Panic selling / capitulation |
| `oi_cascade` | Binary | OI drops >5% in 1H | 0.5% | 0H | Liquidation cascade |
| `oi_funding_divergence` | Binary | OI↓ + funding↑ >2σ | 0.3% | 0-8H | Short squeeze setup |
| `funding_extreme` | Binary | Funding >3σ or <-3σ | 0.8% | 0-8H | Extreme sentiment |
| `funding_flip` | Binary | Sign change + magnitude >0.5σ | 0.2% | 0H | Regime shift signal |

**Total active time:** 0.5-2% of hours (rare, sharp signals)

### Validation
**LUNA crisis (May 9-12, 2022):** 73% of hours triggered `flash_crash_1h` (pass)

### Interface
```python
def compute_crisis_events(df: pd.DataFrame) -> pd.DataFrame:
    """
    Layer 1: Event detection (already implemented)

    Input: Raw OHLCV + OI + funding
    Output: 8 binary columns (0 or 1)
    """
    df = compute_flash_crash_indicators(df)  # 3 features
    df = compute_volume_anomalies(df)        # 1 feature
    df = compute_oi_deltas(df)               # 2 features
    df = compute_funding_extremes(df)        # 2 features
    return df
```

**Status:** ✅ Production ready (no changes)

---

## Layer 1.5: State Transformer (NEW - Core Innovation)

### Purpose
Convert binary event flags into **continuous state descriptors** that represent persistent market conditions.

### Design Principles

1. **Decay over time:** Events lose relevance as time passes (exponential decay)
2. **Smooth aggregation:** Use EWMA instead of raw sums to avoid discontinuities
3. **Persistence detection:** Distinguish single events from sustained patterns
4. **Bounded outputs:** All features normalized to [0, 1] or [-3, 3] z-scores

### State Feature Catalog (12 features)

#### Group A: Temporal Decay Features (hours since last event)

**Purpose:** Capture how recently a shock occurred (decays exponentially)

| Feature | Formula | Decay Half-Life | Interpretation |
|---------|---------|-----------------|----------------|
| `crash_proximity_1h` | `1 / (1 + hours_since_flash_crash_1h / 24)` | 24 hours | Recent crash influence (0=old, 1=just now) |
| `crash_proximity_4h` | `1 / (1 + hours_since_flash_crash_4h / 72)` | 72 hours | Medium-term crash memory |
| `cascade_recency` | `1 / (1 + hours_since_oi_cascade / 48)` | 48 hours | Recent liquidation cascade |

**Implementation:**
```python
def compute_temporal_decay(event_series: pd.Series, half_life_hours: int) -> pd.Series:
    """
    Compute exponentially decaying proximity score.

    Args:
        event_series: Binary event flags (0 or 1)
        half_life_hours: Hours until score decays to 0.5

    Returns:
        Continuous score [0, 1] with exponential decay
    """
    # Find index of last event occurrence
    event_indices = event_series[event_series == 1].index

    def decay_score(timestamp):
        if len(event_indices) == 0:
            return 0.0

        # Hours since last event
        last_event = event_indices[event_indices <= timestamp]
        if len(last_event) == 0:
            return 0.0

        hours_since = (timestamp - last_event[-1]).total_seconds() / 3600
        return 1.0 / (1.0 + hours_since / half_life_hours)

    return event_series.index.map(decay_score)
```

#### Group B: Smoothed Intensity Features (EWMA of event flags)

**Purpose:** Capture sustained stress vs single spikes (smooths discontinuities)

| Feature | Formula | Span | Interpretation |
|---------|---------|------|----------------|
| `crash_intensity_24h` | `EWMA(flash_crash_1h + flash_crash_4h, span=24h)` | 24h | Short-term crash pressure |
| `crash_intensity_72h` | `EWMA(flash_crash_4h + flash_crash_1d, span=72h)` | 72h | Medium-term crisis intensity |
| `cascade_severity` | `EWMA(oi_cascade + volume_spike, span=48h, alpha=0.05)` | 48h | Liquidation stress level |
| `funding_stress` | `EWMA(funding_extreme + funding_flip, span=24h)` | 24h | Funding market stress |

**Implementation:**
```python
def compute_smoothed_intensity(df: pd.DataFrame) -> pd.DataFrame:
    """
    Group B: EWMA smoothing of event flags.

    Converts binary 0/1 events into continuous [0, 1] intensity scores.
    """
    df = df.copy()

    # Crash intensity (multiple timeframes)
    df['crash_intensity_24h'] = (
        df['flash_crash_1h'] + df['flash_crash_4h']
    ).ewm(span=24, adjust=False).mean()

    df['crash_intensity_72h'] = (
        df['flash_crash_4h'] + df['flash_crash_1d']
    ).ewm(span=72, adjust=False).mean()

    # Cascade severity (liquidations + volume panic)
    df['cascade_severity'] = (
        df['oi_cascade'] + df['volume_spike']
    ).ewm(span=48, adjust=False).mean()

    # Funding stress (extreme rates + rapid flips)
    df['funding_stress'] = (
        df['funding_extreme'] + df['funding_flip']
    ).ewm(span=24, adjust=False).mean()

    return df
```

**Alpha parameter:** `alpha = 2 / (span + 1)`
- Span=24h → alpha≈0.08 (slow smoothing)
- Span=48h → alpha≈0.04 (very slow)
- Span=72h → alpha≈0.03 (regime-level)

#### Group C: Frequency Features (event count over window)

**Purpose:** Distinguish chaotic periods (many events) from calm (few events)

| Feature | Formula | Window | Interpretation |
|---------|---------|--------|----------------|
| `crash_frequency_7d` | `count(flash_crash_1h, window=7d) / 168h` | 7 days | Crash clustering rate |
| `cascade_frequency_7d` | `count(oi_cascade + volume_spike, window=7d) / 168h` | 7 days | Liquidation clustering |
| `extreme_event_rate` | `sum(all_events, window=7d) / (168h * 8)` | 7 days | Overall chaos index |

**Implementation:**
```python
def compute_event_frequency(df: pd.DataFrame) -> pd.DataFrame:
    """
    Group C: Event frequency over rolling windows.

    Measures how often events cluster together (crisis regimes).
    """
    df = df.copy()

    # Crash frequency (7-day window)
    df['crash_frequency_7d'] = (
        df['flash_crash_1h'].rolling(window=168).sum() / 168.0
    )

    # Cascade frequency (OI + volume events)
    df['cascade_frequency_7d'] = (
        (df['oi_cascade'] + df['volume_spike']).rolling(window=168).sum() / 168.0
    )

    # Extreme event rate (all 8 events normalized)
    all_events = (
        df['flash_crash_1h'] + df['flash_crash_4h'] + df['flash_crash_1d'] +
        df['volume_spike'] + df['oi_cascade'] + df['oi_funding_divergence'] +
        df['funding_extreme'] + df['funding_flip']
    )
    df['extreme_event_rate'] = all_events.rolling(window=168).sum() / (168.0 * 8)

    return df
```

#### Group D: Persistence Features (consecutive events)

**Purpose:** Detect sustained patterns vs isolated spikes (crisis vs noise)

| Feature | Formula | Logic | Interpretation |
|---------|---------|-------|----------------|
| `crash_persistence` | `max_consecutive(flash_crash_1h, window=24h)` | Longest streak | Sustained crash vs single spike |
| `volatility_persistence` | `std(returns_24h) / mean(std(returns_7d))` | Ratio | Current vol vs baseline |
| `drawdown_persistence` | `(peak - current) / peak, rolling 7d` | % off peak | How deep in drawdown |

**Implementation:**
```python
def compute_persistence_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Group D: Persistence and momentum features.

    Detect sustained patterns (crisis) vs isolated spikes (noise).
    """
    df = df.copy()

    # Crash persistence (longest consecutive streak in 24h)
    df['crash_persistence'] = (
        df['flash_crash_1h']
        .rolling(window=24)
        .apply(lambda x: (x * (x.groupby((x != x.shift()).cumsum()).cumcount() + 1)).max(), raw=False)
    )

    # Volatility persistence (current vs baseline)
    returns = df['close'].pct_change()
    vol_24h = returns.rolling(24).std()
    vol_7d = returns.rolling(168).std()
    df['volatility_persistence'] = vol_24h / vol_7d.replace(0, np.nan)

    # Drawdown persistence (% off peak in 7d window)
    rolling_max = df['close'].rolling(window=168, min_periods=1).max()
    df['drawdown_persistence'] = (rolling_max - df['close']) / rolling_max

    return df
```

### Complete State Transformer Pipeline

```python
def transform_events_to_state(df: pd.DataFrame) -> pd.DataFrame:
    """
    Layer 1.5: Transform binary events → continuous state features.

    Input: df with 8 binary event columns
    Output: df with 12 continuous state columns (0.0-1.0 normalized)

    This is the CRITICAL transformation that prevents HMM thrashing.
    """
    logger.info("Transforming events to state features...")

    # Group A: Temporal decay
    df['crash_proximity_1h'] = compute_temporal_decay(df['flash_crash_1h'], half_life=24)
    df['crash_proximity_4h'] = compute_temporal_decay(df['flash_crash_4h'], half_life=72)
    df['cascade_recency'] = compute_temporal_decay(df['oi_cascade'], half_life=48)

    # Group B: Smoothed intensity
    df = compute_smoothed_intensity(df)

    # Group C: Event frequency
    df = compute_event_frequency(df)

    # Group D: Persistence
    df = compute_persistence_features(df)

    # Normalize all state features to [0, 1]
    state_features = [
        'crash_proximity_1h', 'crash_proximity_4h', 'cascade_recency',
        'crash_intensity_24h', 'crash_intensity_72h', 'cascade_severity', 'funding_stress',
        'crash_frequency_7d', 'cascade_frequency_7d', 'extreme_event_rate',
        'crash_persistence', 'volatility_persistence', 'drawdown_persistence'
    ]

    for feat in state_features:
        # Clip outliers to 99th percentile
        p99 = df[feat].quantile(0.99)
        df[feat] = df[feat].clip(upper=p99)

        # Min-max normalize to [0, 1]
        feat_min = df[feat].min()
        feat_max = df[feat].max()
        if feat_max > feat_min:
            df[feat] = (df[feat] - feat_min) / (feat_max - feat_min)
        else:
            df[feat] = 0.0

    logger.info(f"  ✅ Created {len(state_features)} state features")
    return df
```

**Expected Output Distribution:**
- Normal periods: state features ∈ [0.0, 0.2] (calm)
- Elevated risk: state features ∈ [0.2, 0.5] (stress)
- Crisis: state features ∈ [0.5, 1.0] (alarm)

**Active time:** 10-30% of hours (frequent, smooth signals) vs 0.5-2% for raw events

---

## Layer 2: Regime Classifier (HMM - Modified Inputs)

### Purpose
Classify persistent market regimes using **state features** (not raw events).

### HMM Input Features (15-20 features)

**NEW Feature Set (prevents thrashing):**

| Category | Features | Source | Distribution |
|----------|----------|--------|--------------|
| State features (NEW) | 7-10 selected from Layer 1.5 | Event transformation | Continuous, 10-30% active |
| Macro z-scores | VIX_Z, DXY_Z, YC_SPREAD, M2_GROWTH | Existing pipeline | Continuous, mean=0, std=1 |
| Market structure | BTC.D, USDT.D, TOTAL_RET_21d, ALT_ROTATION | Existing pipeline | Continuous, slow-moving |
| Derived volatility | RV_21 (21d realized vol) | Existing pipeline | Continuous, 20-80% annualized |
| Derived funding | funding_Z (30d z-score) | Existing pipeline | Continuous, mean=0, std=1 |

**OLD Feature Set (REMOVED - causes thrashing):**
- ❌ `flash_crash_1h` (binary) → Replaced by `crash_intensity_24h` (continuous)
- ❌ `flash_crash_4h` (binary) → Replaced by `crash_intensity_72h` (continuous)
- ❌ `oi_cascade` (binary) → Replaced by `cascade_severity` (EWMA)
- ❌ `volume_spike` (binary) → Included in `cascade_severity`
- ❌ `funding_extreme` (binary) → Replaced by `funding_stress` (EWMA)

### Feature Selection Strategy

**Phase 1 (Quick Fix):** Top 3 state features + existing 12 features
```python
PHASE1_STATE_FEATURES = [
    'crash_intensity_72h',    # Most important: sustained crash detection
    'cascade_severity',       # Second: liquidation stress
    'extreme_event_rate',     # Third: overall chaos index
]

PHASE1_FEATURES = PHASE1_STATE_FEATURES + [
    'funding_Z', 'OI_CHANGE', 'RV_21', 'LIQ_VOL_24h',  # Tier 1: Crypto-native
    'USDT.D', 'BTC.D', 'TOTAL_RET_21d', 'ALT_ROTATION',  # Tier 2: Market structure
    'VIX_Z', 'DXY_Z', 'YC_SPREAD', 'M2_GROWTH',          # Tier 3: Macro
]
# Total: 15 features (vs 21 before, removed raw event binaries)
```

**Phase 2 (Full Hybrid):** 10 state features + existing 12 features
```python
PHASE2_STATE_FEATURES = [
    'crash_intensity_72h', 'cascade_severity', 'funding_stress',  # Intensity
    'crash_frequency_7d', 'extreme_event_rate',                   # Frequency
    'crash_persistence', 'volatility_persistence',                # Persistence
    'crash_proximity_4h', 'cascade_recency',                      # Recency
    'drawdown_persistence',                                        # Drawdown
]

PHASE2_FEATURES = PHASE2_STATE_FEATURES + [
    # Same 12 existing features as Phase 1
]
# Total: 22 features (10 state + 12 existing)
```

**Phase 3 (Advanced):** Regime-dependent decay + GARCH volatility
```python
# State transformer uses different decay rates per regime
# - Normal regime: half_life = 72h (slow decay)
# - Stressed regime: half_life = 24h (fast decay)
# - Crisis regime: half_life = 12h (very fast decay)
```

### HMM Configuration

```python
from hmmlearn.hmm import GaussianHMM

# 4-state HMM (same as before)
n_states = 4
state_names = ['crisis', 'risk_off', 'neutral', 'risk_on']

# Covariance structure: diagonal (assumes independence)
# This prevents feature correlations from causing instability
covariance_type = 'diag'

# Training parameters
n_iter = 1000          # EM iterations
n_init = 10            # Random initializations (ensemble)
tol = 1e-4             # Convergence tolerance

# Transition constraints (prevent thrashing)
# - Minimum regime duration: encourage staying in same state
# - Self-transition bias: P(stay) > P(switch)
min_regime_duration_hours = 24  # Crisis must last ≥24 hours
transition_prior = np.array([
    #  crisis  risk_off  neutral  risk_on
    [0.90,    0.08,     0.01,    0.01],    # from crisis
    [0.05,    0.85,     0.08,    0.02],    # from risk_off
    [0.01,    0.05,     0.89,    0.05],    # from neutral
    [0.01,    0.02,     0.07,    0.90],    # from risk_on
])
# Diagonal dominant → encourages staying in current state
```

**Key Change:** Input features are **continuous and smooth** (not binary and jumpy).

### Training Protocol

```python
def train_hmm_with_state_features(
    df: pd.DataFrame,
    state_features: list,
    n_init: int = 10
) -> GaussianHMM:
    """
    Train HMM using state features (not raw events).

    Args:
        df: DataFrame with state features + macro features
        state_features: List of state feature names to use
        n_init: Number of random initializations

    Returns:
        Trained HMM model
    """
    # Extract features
    feature_cols = state_features + MACRO_FEATURES + MARKET_FEATURES
    X = df[feature_cols].values

    # Handle NaNs (forward fill, then zero fill)
    X = pd.DataFrame(X, columns=feature_cols).fillna(method='ffill').fillna(0).values

    # Standardize (mean=0, std=1)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Train HMM with ensemble (best of n_init)
    best_model = None
    best_score = -np.inf

    for seed in range(n_init):
        model = GaussianHMM(
            n_components=4,
            covariance_type='diag',
            n_iter=1000,
            random_state=seed,
            verbose=False
        )

        model.fit(X_scaled)
        score = model.score(X_scaled)

        if score > best_score:
            best_score = score
            best_model = model
            logger.info(f"  New best model: seed={seed}, score={score:.2f}")

    # Interpret states (map HMM states to regime labels)
    state_map = interpret_hmm_states(best_model, X_scaled, df)

    return best_model, scaler, state_map
```

**Expected Improvements:**
- Transitions: 117/year → 10-20/year (smooth features prevent thrashing)
- Crisis detection: 0% → 60-80% (state features capture sustained crises)
- Silhouette: 0.11 → 0.40-0.60 (better cluster separation)

---

## Regime Transition Logic (Anti-Thrashing Mechanisms)

### Problem: Regime Thrashing
- **Definition:** Rapid regime switches (>50 transitions/year) caused by noisy features
- **Impact:** Unstable position sizing, excessive rebalancing, poor performance
- **Root cause:** Binary event features create discontinuous jumps

### Solution: Multi-Layer Defense

#### Defense 1: Smooth Input Features (Layer 1.5)
**Already addressed** by state transformation (EWMA, decay, frequency).

#### Defense 2: Hysteresis Bands
**Concept:** Different thresholds for entering vs exiting a regime.

```python
# Example: Crisis regime entry/exit bands
CRISIS_ENTRY_THRESHOLD = 0.6    # Need 60% crisis probability to enter
CRISIS_EXIT_THRESHOLD = 0.4     # Need to drop below 40% to exit

def apply_hysteresis(
    regime_probs: pd.DataFrame,
    current_regime: str,
    hysteresis_bands: dict
) -> str:
    """
    Apply hysteresis to prevent rapid regime switching.

    Args:
        regime_probs: Dict of {regime: probability}
        current_regime: Current regime label
        hysteresis_bands: Dict of {regime: (entry, exit)}

    Returns:
        New regime label (may be same as current if in band)
    """
    # Get highest probability regime
    next_regime = max(regime_probs, key=regime_probs.get)
    next_prob = regime_probs[next_regime]

    # If staying in same regime, no hysteresis needed
    if next_regime == current_regime:
        return current_regime

    # If switching regimes, check hysteresis bands
    entry_threshold, exit_threshold = hysteresis_bands[next_regime]

    # Must exceed entry threshold to switch
    if next_prob >= entry_threshold:
        # Also check if we should stay in current regime
        current_prob = regime_probs[current_regime]
        current_exit = hysteresis_bands[current_regime][1]

        if current_prob >= current_exit:
            # Stay in current regime (haven't crossed exit band)
            return current_regime
        else:
            # Switch to new regime (crossed both bands)
            return next_regime
    else:
        # Not strong enough signal to switch
        return current_regime

# Hysteresis configuration
HYSTERESIS_BANDS = {
    'crisis':   (0.60, 0.40),  # Need 60% to enter, drop below 40% to exit
    'risk_off': (0.50, 0.35),  # Need 50% to enter, drop below 35% to exit
    'neutral':  (0.45, 0.30),  # Need 45% to enter, drop below 30% to exit
    'risk_on':  (0.50, 0.35),  # Need 50% to enter, drop below 35% to exit
}
```

#### Defense 3: Minimum Regime Duration
**Concept:** Force regimes to last at least N hours before allowing switch.

```python
MIN_REGIME_DURATION = {
    'crisis': 24,    # Crisis must last ≥24 hours
    'risk_off': 48,  # Risk-off must last ≥48 hours
    'neutral': 72,   # Neutral must last ≥72 hours (most stable)
    'risk_on': 48,   # Risk-on must last ≥48 hours
}

def enforce_minimum_duration(
    regime_history: pd.Series,
    new_regime: str,
    timestamp: pd.Timestamp
) -> str:
    """
    Prevent regime switches if minimum duration not met.

    Args:
        regime_history: Series of past regime labels
        new_regime: Proposed new regime
        timestamp: Current timestamp

    Returns:
        Regime label (may override new_regime if too soon)
    """
    # Find when current regime started
    current_regime = regime_history.iloc[-1]

    if new_regime == current_regime:
        return current_regime  # No switch

    # Find last regime change
    regime_change_indices = regime_history != regime_history.shift(1)
    last_change_idx = regime_change_indices[regime_change_indices].index[-1]
    hours_in_regime = (timestamp - last_change_idx).total_seconds() / 3600

    # Check minimum duration
    min_duration = MIN_REGIME_DURATION[current_regime]

    if hours_in_regime < min_duration:
        # Override: stay in current regime
        logger.debug(f"Blocking regime switch: {current_regime} → {new_regime} "
                     f"(only {hours_in_regime:.1f}h, need {min_duration}h)")
        return current_regime
    else:
        # Allow switch
        return new_regime
```

#### Defense 4: Regime Smoothing (Post-Processing)
**Concept:** Apply moving average to regime probabilities over 6-12 hour window.

```python
def smooth_regime_probabilities(
    regime_probs_df: pd.DataFrame,
    window: int = 12
) -> pd.DataFrame:
    """
    Smooth regime probabilities using rolling average.

    This prevents single-bar spikes from triggering regime changes.

    Args:
        regime_probs_df: DataFrame with columns [crisis, risk_off, neutral, risk_on]
        window: Smoothing window in hours (default: 12)

    Returns:
        Smoothed probability DataFrame
    """
    return regime_probs_df.rolling(window=window, min_periods=1).mean()
```

### Combined Anti-Thrashing Pipeline

```python
def classify_regime_anti_thrashing(
    df: pd.DataFrame,
    hmm_model: GaussianHMM,
    scaler: StandardScaler,
    feature_cols: list
) -> pd.DataFrame:
    """
    Full regime classification with anti-thrashing mechanisms.

    Layers of defense:
    1. Smooth state features (Layer 1.5)
    2. HMM with transition priors
    3. Probability smoothing (12h rolling average)
    4. Hysteresis bands
    5. Minimum regime duration

    Args:
        df: DataFrame with features
        hmm_model: Trained HMM
        scaler: Feature scaler
        feature_cols: Feature names

    Returns:
        DataFrame with regime labels + probabilities
    """
    # Extract and scale features
    X = df[feature_cols].fillna(method='ffill').fillna(0).values
    X_scaled = scaler.transform(X)

    # HMM prediction (Viterbi decoding)
    states = hmm_model.predict(X_scaled)
    probs = hmm_model.predict_proba(X_scaled)

    # Map states to regime labels
    regime_labels_raw = [STATE_MAP[s] for s in states]

    # Step 1: Smooth probabilities (12h window)
    probs_df = pd.DataFrame(
        probs,
        columns=['crisis', 'risk_off', 'neutral', 'risk_on'],
        index=df.index
    )
    probs_smoothed = smooth_regime_probabilities(probs_df, window=12)

    # Step 2: Apply hysteresis + minimum duration
    regime_labels_final = []
    for i, timestamp in enumerate(df.index):
        if i == 0:
            # First bar: use raw HMM output
            regime_labels_final.append(regime_labels_raw[i])
            continue

        # Get smoothed probabilities
        probs_dict = probs_smoothed.iloc[i].to_dict()
        current_regime = regime_labels_final[-1]

        # Apply hysteresis
        regime_with_hysteresis = apply_hysteresis(
            probs_dict,
            current_regime,
            HYSTERESIS_BANDS
        )

        # Enforce minimum duration
        regime_final = enforce_minimum_duration(
            pd.Series(regime_labels_final),
            regime_with_hysteresis,
            timestamp
        )

        regime_labels_final.append(regime_final)

    # Add to dataframe
    df['regime_label'] = regime_labels_final
    df['regime_proba_crisis'] = probs_smoothed['crisis']
    df['regime_proba_risk_off'] = probs_smoothed['risk_off']
    df['regime_proba_neutral'] = probs_smoothed['neutral']
    df['regime_proba_risk_on'] = probs_smoothed['risk_on']
    df['regime_confidence'] = probs_smoothed.max(axis=1)

    return df
```

**Expected Transition Frequency:**
- Raw HMM: 117 transitions/year
- + Smoothing: 60 transitions/year
- + Hysteresis: 30 transitions/year
- + Min duration: **10-20 transitions/year** ✅

---

## Production Integration

### Archetype System Interface

**No breaking changes** to archetype system. Regime classifier API remains identical.

```python
# Before (static labels)
regime = '2022' in timestamp ? 'risk_off' : 'neutral'

# After (HMM with state features)
regime_result = regime_classifier.classify(macro_features, timestamp)
regime = regime_result['regime']  # Same interface
```

### Stream Mode Compatibility

**Challenge:** Can state features be computed in real-time?

**Answer:** Yes, with minor latency (<1 bar).

| Feature Type | Real-Time Feasibility | Latency |
|--------------|----------------------|---------|
| Temporal decay | ✅ Yes | 0 bars (instant) |
| EWMA smoothing | ✅ Yes | 0 bars (incremental update) |
| Rolling frequency | ✅ Yes | 0 bars (maintain buffer) |
| Persistence | ✅ Yes | 0 bars (stateful counter) |

**Implementation:**
```python
class StreamStateTransformer:
    """
    Stateful state transformer for live trading.

    Maintains buffers for rolling computations.
    """

    def __init__(self, window_sizes: dict):
        self.event_buffers = {}
        self.ewma_states = {}
        self.window_sizes = window_sizes
        self.last_event_times = {}

    def update(self, bar: dict, timestamp: pd.Timestamp) -> dict:
        """
        Update state features with new bar.

        Args:
            bar: Dict with event features (binary 0/1)
            timestamp: Bar timestamp

        Returns:
            Dict with state features (continuous 0.0-1.0)
        """
        state_features = {}

        # Group A: Temporal decay
        for event_name in ['flash_crash_1h', 'flash_crash_4h', 'oi_cascade']:
            if bar.get(event_name, 0) == 1:
                self.last_event_times[event_name] = timestamp

            if event_name in self.last_event_times:
                hours_since = (timestamp - self.last_event_times[event_name]).total_seconds() / 3600
                half_life = self.window_sizes.get(f'{event_name}_decay', 48)
                state_features[f'{event_name}_proximity'] = 1.0 / (1.0 + hours_since / half_life)
            else:
                state_features[f'{event_name}_proximity'] = 0.0

        # Group B: EWMA smoothing (incremental update)
        for metric_name, components in [
            ('crash_intensity_24h', ['flash_crash_1h', 'flash_crash_4h']),
            ('cascade_severity', ['oi_cascade', 'volume_spike']),
            ('funding_stress', ['funding_extreme', 'funding_flip'])
        ]:
            # Initialize if first bar
            if metric_name not in self.ewma_states:
                self.ewma_states[metric_name] = 0.0

            # Update EWMA incrementally
            value = sum(bar.get(c, 0) for c in components)
            span = self.window_sizes.get(f'{metric_name}_span', 24)
            alpha = 2.0 / (span + 1)

            self.ewma_states[metric_name] = (
                alpha * value + (1 - alpha) * self.ewma_states[metric_name]
            )
            state_features[metric_name] = self.ewma_states[metric_name]

        # Group C: Rolling frequency (maintain buffer)
        for event_name in ['flash_crash_1h', 'oi_cascade', 'volume_spike']:
            buffer_key = f'{event_name}_buffer'
            if buffer_key not in self.event_buffers:
                self.event_buffers[buffer_key] = deque(maxlen=168)  # 7 days

            self.event_buffers[buffer_key].append(bar.get(event_name, 0))
            state_features[f'{event_name}_freq_7d'] = (
                sum(self.event_buffers[buffer_key]) / len(self.event_buffers[buffer_key])
            )

        return state_features
```

**Memory footprint:** <10 KB per symbol (deques + EWMA states)

### Fallback Behavior

**Scenario:** HMM fails or produces invalid output.

**Fallback Hierarchy:**
1. **Primary:** HMM with state features (this architecture)
2. **Fallback 1:** Static year-based labels (2022=risk_off, others=neutral)
3. **Fallback 2:** VIX-based simple rules (VIX >30 → risk_off)
4. **Fallback 3:** Neutral regime (safest default)

```python
def get_regime_with_fallback(
    features: dict,
    timestamp: pd.Timestamp,
    hmm_classifier: Optional[HMMRegimeModel] = None
) -> dict:
    """
    Get regime with graceful degradation.

    Returns:
        {
            'regime': str,
            'confidence': float,
            'source': str  # 'hmm', 'static', 'vix', 'neutral'
        }
    """
    try:
        # Try HMM first
        if hmm_classifier is not None:
            result = hmm_classifier.classify(features, timestamp)
            if result.get('fallback', False):
                raise ValueError("HMM returned fallback mode")
            return {**result, 'source': 'hmm'}
    except Exception as e:
        logger.warning(f"HMM failed: {e}, using fallback")

    # Fallback 1: Static labels
    year = timestamp.year
    if year == 2022:
        return {'regime': 'risk_off', 'confidence': 0.7, 'source': 'static'}

    # Fallback 2: VIX-based rules
    vix = features.get('VIX', 0)
    if vix > 30:
        return {'regime': 'risk_off', 'confidence': 0.6, 'source': 'vix'}
    elif vix < 15:
        return {'regime': 'risk_on', 'confidence': 0.6, 'source': 'vix'}

    # Fallback 3: Neutral
    return {'regime': 'neutral', 'confidence': 0.5, 'source': 'neutral'}
```

---

## Monitoring & Observability

### System Health Metrics

**Layer 1: Event Detection Metrics**
```python
EVENT_HEALTH_METRICS = {
    'event_trigger_rate': "% of bars with any event (expect 1-3%)",
    'crash_detection_rate': "% of bars with flash_crash_* (expect 0.3-1%)",
    'event_coverage': "% non-NaN values (expect 100%)",
    'luna_validation': "% LUNA hours with flash_crash_1h (expect >70%)",
}
```

**Layer 1.5: State Transformation Metrics**
```python
STATE_HEALTH_METRICS = {
    'state_active_rate': "% of bars with state features >0.1 (expect 10-30%)",
    'state_smoothness': "Correlation(state_t, state_t-1) (expect >0.90)",
    'state_discontinuity': "Count(abs(state_t - state_t-1) > 0.3) (expect <5%)",
    'state_normalization': "Min/max of state features (expect [0, 1])",
}
```

**Layer 2: HMM Metrics**
```python
HMM_HEALTH_METRICS = {
    'transition_frequency': "Transitions per year (expect 10-20)",
    'silhouette_score': "Cluster quality (expect >0.40)",
    'crisis_detection_rate': "% of crisis events detected (expect >60%)",
    'regime_distribution': "% of time in each regime (crisis <10%, neutral 40-60%)",
    'confidence_score': "Mean regime probability (expect >0.60)",
}
```

### Early Warning System

**Regime Thrashing Detector:**
```python
def detect_regime_thrashing(regime_history: pd.Series, window_hours: int = 168) -> dict:
    """
    Detect regime thrashing (early warning).

    Args:
        regime_history: Series of regime labels
        window_hours: Rolling window for detection

    Returns:
        {
            'is_thrashing': bool,
            'transitions_per_week': float,
            'severity': str  # 'normal', 'elevated', 'critical'
        }
    """
    # Count transitions in rolling window
    transitions = (regime_history != regime_history.shift(1)).rolling(window_hours).sum()
    transitions_per_week = transitions.iloc[-1]

    # Severity levels
    if transitions_per_week <= 2:
        severity = 'normal'
        is_thrashing = False
    elif transitions_per_week <= 5:
        severity = 'elevated'
        is_thrashing = False
    else:
        severity = 'critical'
        is_thrashing = True

    return {
        'is_thrashing': is_thrashing,
        'transitions_per_week': transitions_per_week,
        'severity': severity,
        'timestamp': regime_history.index[-1]
    }
```

**Event vs State Divergence Monitor:**
```python
def monitor_event_state_divergence(df: pd.DataFrame) -> dict:
    """
    Detect when events fire but state features don't respond.

    This indicates Layer 1.5 transformation issues.
    """
    # Check correlation between events and state features
    event_cols = ['flash_crash_1h', 'oi_cascade', 'volume_spike']
    state_cols = ['crash_intensity_24h', 'cascade_severity']

    # Compute cross-correlation with 0-12h lags
    correlations = {}
    for event_col in event_cols:
        for state_col in state_cols:
            max_corr = 0
            best_lag = 0
            for lag in range(13):
                corr = df[event_col].corr(df[state_col].shift(-lag))
                if abs(corr) > abs(max_corr):
                    max_corr = corr
                    best_lag = lag

            correlations[f'{event_col}_{state_col}'] = {
                'correlation': max_corr,
                'lag_hours': best_lag
            }

    # Check if any correlations are too weak (<0.3)
    weak_correlations = [
        k for k, v in correlations.items() if abs(v['correlation']) < 0.3
    ]

    return {
        'correlations': correlations,
        'weak_links': weak_correlations,
        'health': 'ok' if len(weak_correlations) == 0 else 'warning'
    }
```

### Logging & Alerts

```python
class RegimeMonitor:
    """
    Production monitoring for hybrid regime detection.
    """

    def __init__(self, alert_thresholds: dict):
        self.alert_thresholds = alert_thresholds
        self.regime_history = deque(maxlen=720)  # 30 days
        self.transition_times = []

    def log_regime_update(
        self,
        timestamp: pd.Timestamp,
        regime: str,
        confidence: float,
        state_features: dict,
        event_flags: dict
    ):
        """
        Log regime update for monitoring.
        """
        # Detect regime change
        if len(self.regime_history) > 0 and regime != self.regime_history[-1]['regime']:
            self.transition_times.append(timestamp)
            logger.info(f"🔄 Regime transition: {self.regime_history[-1]['regime']} → {regime} "
                        f"(confidence: {confidence:.1%})")

            # Alert if crisis detected
            if regime == 'crisis':
                logger.warning(f"⚠️  CRISIS REGIME DETECTED at {timestamp}")
                self._send_alert('crisis_detected', timestamp, confidence)

        # Store history
        self.regime_history.append({
            'timestamp': timestamp,
            'regime': regime,
            'confidence': confidence,
            'state_features': state_features,
            'event_flags': event_flags
        })

        # Check for thrashing
        if len(self.transition_times) >= 10:
            recent_transitions = [
                t for t in self.transition_times if (timestamp - t).total_seconds() / 3600 <= 168
            ]
            if len(recent_transitions) > 5:
                logger.error(f"🚨 REGIME THRASHING DETECTED: {len(recent_transitions)} transitions in 7 days")
                self._send_alert('thrashing_detected', timestamp, len(recent_transitions))

    def _send_alert(self, alert_type: str, timestamp: pd.Timestamp, value: float):
        """Send alert to monitoring system."""
        # Implement your alerting (Slack, PagerDuty, etc.)
        pass
```

---

## Implementation Roadmap

### Phase 1: Quick Fix (1 week - MVP)

**Goal:** Add 3 state features, retrain HMM, validate improvement

**Tasks:**
1. ✅ **Day 1-2:** Implement Layer 1.5 transformer (core 3 features)
   ```python
   # File: engine/features/state_transformer.py
   def transform_events_to_state_phase1(df: pd.DataFrame) -> pd.DataFrame:
       df['crash_intensity_72h'] = (df['flash_crash_4h'] + df['flash_crash_1d']).ewm(span=72).mean()
       df['cascade_severity'] = (df['oi_cascade'] + df['volume_spike']).ewm(span=48).mean()
       df['extreme_event_rate'] = (all_events).rolling(168).sum() / (168 * 8)
       return df
   ```

2. ✅ **Day 3:** Modify HMM training script
   ```bash
   # Update: bin/train_regime_hmm_v2.py
   # Add state features to feature list
   # Remove binary event features
   ```

3. ✅ **Day 4:** Retrain HMM with new features
   ```bash
   python bin/train_regime_hmm_v2.py --features phase1 --n_init 10
   ```

4. ✅ **Day 5:** Validate improvements
   ```bash
   python bin/validate_regime_classifier.py --model models/hmm_regime_phase1.pkl
   # Check: transitions/year, crisis detection %, silhouette
   ```

5. ✅ **Day 6-7:** Integrate into archetype system + smoke test
   ```bash
   python bin/run_multi_regime_smoke_tests.py --regime-model hmm_phase1
   ```

**Success Criteria:**
- Transitions: 117 → <50/year
- Crisis detection: 0% → >40%
- Silhouette: 0.11 → >0.30

**Rollback Plan:** If metrics worse, revert to static labels (configs already exist)

---

### Phase 2: Full Hybrid (2 weeks)

**Goal:** Implement all 12 state features + anti-thrashing mechanisms

**Tasks:**
1. **Week 1:** Complete state transformer
   - All 4 feature groups (decay, intensity, frequency, persistence)
   - Unit tests for each feature type
   - Stream mode compatibility

2. **Week 2:** Advanced HMM features
   - Hysteresis bands
   - Minimum regime duration
   - Probability smoothing
   - Comprehensive validation (walk-forward, cross-regime)

**Success Criteria:**
- Transitions: <20/year
- Crisis detection: >60%
- Silhouette: >0.40
- No thrashing during normal volatility

---

### Phase 3: Advanced Features (Future - 1 month)

**Goal:** Regime-dependent decay + GARCH volatility + ML feature selection

**Tasks:**
1. **Regime-dependent decay:**
   ```python
   # Different decay rates per regime
   if current_regime == 'crisis':
       half_life = 12  # Fast decay (crisis fades quickly)
   elif current_regime == 'risk_off':
       half_life = 48  # Medium decay
   else:
       half_life = 72  # Slow decay (calm periods)
   ```

2. **GARCH volatility clustering:**
   ```python
   # Use GARCH(1,1) for volatility persistence instead of rolling std
   from arch import arch_model
   model = arch_model(returns, vol='Garch', p=1, q=1)
   result = model.fit()
   df['volatility_persistence_garch'] = result.conditional_volatility
   ```

3. **ML feature selection:**
   ```python
   # Use RandomForest to identify most important state features
   from sklearn.ensemble import RandomForestClassifier
   from sklearn.feature_selection import SelectKBest, mutual_info_classif

   # Train on manually labeled regimes
   X = df[all_state_features]
   y = df['regime_label_manual']  # Ground truth labels

   # Select top K features
   selector = SelectKBest(mutual_info_classif, k=10)
   X_selected = selector.fit_transform(X, y)
   ```

**Success Criteria:**
- Crisis detection: >80%
- Silhouette: >0.50
- Early detection: 6-12 hours before peak crisis

---

## Risk Mitigation

### What if State Features Still Cause Thrashing?

**Scenario:** Phase 1 shows <50% improvement (still >60 transitions/year)

**Root Cause Analysis:**
1. State features not smooth enough (EWMA span too short)
2. State features still too correlated with events (lag too short)
3. HMM model hyperparameters need tuning

**Contingency Actions:**
1. **Increase EWMA spans:** 24h → 72h, 48h → 96h
2. **Add more smoothing layers:** 2-stage EWMA (EWMA of EWMA)
3. **Reduce feature dimensionality:** PCA to decorrelate state features
4. **Switch to alternative models:** Hidden Semi-Markov Model (HSMM) with explicit duration modeling

**Fallback to Static Labels:**
If all above fail, document findings and revert to year-based labels until better solution found.

---

### What if Crisis Detection Remains Low?

**Scenario:** Phase 2 shows <60% crisis detection

**Root Cause Analysis:**
1. State features don't spike sharply enough during crises
2. HMM state interpretation incorrect (crisis mapped to wrong cluster)
3. Training data insufficient (only 3-4 crisis events in 2022-2024)

**Contingency Actions:**
1. **Supervised learning:** Manually label regimes, train Random Forest
2. **Hybrid model:** HMM for normal regimes, rule-based for crisis (if state features >0.7 → crisis)
3. **Feature engineering:** Add more crisis-specific features (correlation breakdowns, contagion metrics)

---

### A/B Testing Framework

**Goal:** Compare state-based HMM vs static labels vs GMM

```python
def run_ab_test_regimes(
    df: pd.DataFrame,
    models: dict,
    test_window: tuple
) -> pd.DataFrame:
    """
    A/B test different regime classifiers.

    Args:
        df: Test data
        models: {
            'hmm_state': HMMRegimeModel with state features,
            'hmm_events': HMMRegimeModel with raw events,
            'static': Static year-based labels,
            'gmm': GMM classifier
        }
        test_window: (start_date, end_date)

    Returns:
        Comparison metrics DataFrame
    """
    results = {}

    for name, model in models.items():
        # Classify regimes
        if name == 'static':
            df[f'regime_{name}'] = df.index.map(lambda t: 'risk_off' if t.year == 2022 else 'neutral')
        else:
            df_classified = model.classify_batch(df)
            df[f'regime_{name}'] = df_classified['regime_label']

        # Compute metrics
        transitions = (df[f'regime_{name}'] != df[f'regime_{name}'].shift(1)).sum()
        crisis_pct = (df[f'regime_{name}'] == 'crisis').sum() / len(df) * 100

        # Crisis detection accuracy (manual labels required)
        if 'regime_label_manual' in df.columns:
            crisis_bars_manual = df['regime_label_manual'] == 'crisis'
            crisis_detected = df.loc[crisis_bars_manual, f'regime_{name}'] == 'crisis'
            detection_rate = crisis_detected.sum() / crisis_bars_manual.sum() * 100
        else:
            detection_rate = np.nan

        results[name] = {
            'transitions_per_year': transitions / (len(df) / 8760),
            'crisis_pct': crisis_pct,
            'crisis_detection_rate': detection_rate
        }

    return pd.DataFrame(results).T
```

**Deployment Decision:**
- Run A/B test on 2024 data (out-of-sample)
- Deploy model with best crisis_detection_rate AND transitions <20/year
- If tie, choose most interpretable model

---

## Success Metrics Summary

### Quantitative Targets

| Metric | Current (Binary Events) | Target (State Features) | Phase 1 Goal | Phase 2 Goal |
|--------|-------------------------|-------------------------|--------------|--------------|
| Transitions/year | 117 | 10-20 | <50 | 10-20 |
| Crisis detection % | 0% | >60% | >40% | >60% |
| Silhouette score | 0.11 | >0.40 | >0.30 | >0.40 |
| False positive % | TBD | <2% | <5% | <2% |
| LUNA detection % | 0% | >80% | >60% | >80% |
| FTX detection % | 0% | >80% | >50% | >80% |
| June 2022 detection % | 0% | >70% | >40% | >70% |

### Qualitative Success Criteria

✅ **System is production-ready if:**
1. Regime transitions feel natural (not random jumps)
2. Crisis regime activates during major crashes (LUNA, FTX, June)
3. Normal volatility doesn't trigger false crisis alarms
4. State features are interpretable (analysts can explain why regime changed)
5. Stream mode produces same results as batch mode
6. Monitoring dashboards show healthy metrics (no thrashing alerts)

❌ **System is NOT ready if:**
1. Regime thrashing continues (>30 transitions/year)
2. Crisis detection remains low (<40%)
3. State features cause new bugs in production
4. System is too complex for operators to understand

---

## Conclusion

### Key Innovations

1. **Two-layer architecture** separates fast events from slow states
2. **State transformation** converts binary flags → continuous features (prevents thrashing)
3. **Multi-layer anti-thrashing** (smoothing, hysteresis, min duration)
4. **Pragmatic phasing** (3 state features → 12 features → advanced)

### Why This Will Work

**Root Cause:** Binary event features (0/1) create discontinuous jumps that HMM interprets as regime changes.

**Solution:** Transform events into **continuous, smooth state features** that represent persistent market conditions, not instantaneous shocks.

**Evidence:**
- EWMA smoothing (span=72h) creates gradual transitions, not jumps
- Decay functions (half_life=48h) slowly forget events over days, not instantly
- Frequency metrics (7-day windows) aggregate events into trends, not spikes

**Expected Outcome:**
- HMM sees smooth, continuous signals → interprets as gradual regime transitions
- Regime changes become infrequent (10-20/year) and meaningful (aligned with actual market shifts)
- Crisis detection improves because state features capture **sustained stress**, not single events

### Implementation Confidence: HIGH

**Why 1-2 sprints is realistic:**
- Layer 1 (events): Already implemented ✅
- Layer 1.5 (state transformer): ~200 lines of code, well-defined formulas
- Layer 2 (HMM): Minimal changes (swap features, add post-processing)
- Validation: Existing test harness (LUNA, FTX validation)

**Biggest Risk:** State features still cause some thrashing (60 transitions/year instead of 117)

**Mitigation:** Phase 1 validates the approach quickly. If <50% improvement, pivot to Contingency Plan B (model tuning) before investing in Phase 2.

---

**Next Step:** Review this architecture, approve Phase 1 implementation, assign to engineer.

**Estimated LOE:**
- Phase 1 (MVP): 5-7 days
- Phase 2 (Full): 10-14 days
- Phase 3 (Advanced): 20-30 days (future)

**Status:** Ready for implementation approval.
