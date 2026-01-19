# Wisdom Time Layer Architecture (v2.0)

**Component**: Temporal Fusion Intelligence Layer
**Purpose**: Integrate time-based market intelligence (Gann cycles, Fibonacci time clusters, temporal confluence) to add "when" dimension to pattern detection
**Philosophy**: Time is NOT prediction—it's PRESSURE. Markets have rhythm, periodicity, emotional cycles that create confluence windows.
**Status**: Production-Ready Design (Phase 2 Integration)

---

## Executive Summary

The Wisdom Time Layer adds temporal intelligence to the Bull Machine's pattern detection by calculating confluence scores when multiple time cycles align. This layer modulates archetype fusion weights (±5-15%) based on:

1. **Fibonacci Time Clusters** - Bars since pivots/Wyckoff events (21, 34, 55, 89, 144 bars)
2. **Gann Cycles** - Square of 9, 30/60/90 day cycles, 360° rotations
3. **Volatility Cycles** - 30-day rolling std, phase detection (accumulation/distribution)
4. **Emotional Cycles** - Market psychology phases (euphoria, capitulation, disbelief)

**Key Design Principles**:
- Soft adjustments only (no hard vetoes)
- Bounded multipliers (0.85x - 1.15x range)
- Feature parity (batch = stream mode)
- Optional/tunable (can be disabled for A/B testing)
- Interpretable (operators understand temporal scores)

---

## 1. Feature Specifications

### 1.1 Fibonacci Time Cluster Features

**Source**: `engine/temporal/fib_time_clusters.py` (existing implementation + enhancements)

| Feature Name | Type | Range | Description | Formula |
|-------------|------|-------|-------------|---------|
| `bars_since_sc` | int | 0-500+ | Bars since Selling Climax | `current_idx - last_sc_idx` |
| `bars_since_bc` | int | 0-500+ | Bars since Buying Climax | `current_idx - last_bc_idx` |
| `bars_since_spring_a` | int | 0-500+ | Bars since Spring-A (trap) | `current_idx - last_spring_idx` |
| `bars_since_utad` | int | 0-500+ | Bars since UTAD | `current_idx - last_utad_idx` |
| `bars_since_lps` | int | 0-500+ | Bars since Last Point Support | `current_idx - last_lps_idx` |
| `bars_since_last_pivot` | int | 0-200 | Bars since nearest swing pivot | Min of all pivot distances |
| `fib_time_cluster_score` | float | 0.0-1.0 | Temporal confluence strength | See Algorithm 1.1 |
| `is_fib_time_cluster_zone` | bool | T/F | High confluence flag | `score >= 0.7` |
| `fib_time_cluster_match_count` | int | 0-10 | Number of Fib alignments | Count of matches within tolerance |

**Algorithm 1.1: Fibonacci Time Cluster Score**

```python
def compute_fib_time_cluster_score(
    bars_since_events: Dict[str, int],  # {'sc': 55, 'bc': 144, 'pivot_1': 89}
    fib_levels: List[int] = [21, 34, 55, 89, 144, 233],
    tolerance: int = 2,
    wyckoff_event_weight: float = 1.3,
    pivot_weight: float = 1.0
) -> Tuple[float, Dict]:
    """
    Calculate temporal confluence score based on Fibonacci alignment.

    Returns:
        (score, metadata) where score ∈ [0, 1]
        - 0.0-0.3: Weak/no confluence
        - 0.3-0.6: Moderate confluence (1-2 matches)
        - 0.6-0.8: Strong confluence (3+ matches)
        - 0.8-1.0: Extreme confluence (4+ matches, tight convergence)
    """
    score = 0.0
    matches = []

    for event_name, event_bars in bars_since_events.items():
        for fib_level in fib_levels:
            # Check alignment within tolerance
            if abs(event_bars - fib_level) <= tolerance:
                # Weight by Fibonacci importance (higher = more significant)
                fib_weight = fib_level / max(fib_levels)  # 0.09-1.0

                # Weight by event importance
                if 'wyckoff' in event_name.lower() or 'sc' in event_name or 'spring' in event_name:
                    event_weight = wyckoff_event_weight  # 1.3x for trap events
                else:
                    event_weight = pivot_weight  # 1.0x for swing pivots

                match_score = fib_weight * event_weight
                matches.append({
                    'event': event_name,
                    'fib_level': fib_level,
                    'event_bars': event_bars,
                    'distance': abs(event_bars - fib_level),
                    'score': match_score
                })
                score += match_score

    # Normalize by theoretical maximum
    if len(bars_since_events) > 0:
        max_possible = len(bars_since_events) * wyckoff_event_weight
        score = score / max_possible

    # Confluence multiplier (bonus for multiple matches)
    if len(matches) >= 4:
        score *= 1.3  # 30% bonus for 4+ matches (extreme confluence)
    elif len(matches) >= 3:
        score *= 1.2  # 20% bonus for 3 matches
    elif len(matches) >= 2:
        score *= 1.1  # 10% bonus for 2 matches

    # Tightness bonus (all matches within ±1 bar)
    if len(matches) >= 2:
        max_distance = max(m['distance'] for m in matches)
        if max_distance <= 1:
            score *= 1.15  # 15% bonus for very tight cluster

    # Cap at 1.0
    score = min(score, 1.0)

    metadata = {
        'match_count': len(matches),
        'matches': matches,
        'dominant_fib': max(matches, key=lambda m: m['score'])['fib_level'] if matches else None
    }

    return score, metadata
```

**Pivot Detection Logic**:

```python
def detect_pivot_points(
    df: pd.DataFrame,
    window: int = 5,
    min_strength: float = 0.6
) -> pd.DataFrame:
    """
    Detect swing high/low pivot points for time cluster analysis.

    Args:
        df: OHLCV DataFrame
        window: Lookback window for pivot detection
        min_strength: Minimum relative strength (0-1)

    Returns:
        DataFrame with columns: ['bar_index', 'price', 'type', 'strength']
    """
    pivots = []

    for i in range(window, len(df) - window):
        # Check for swing high
        center_high = df['high'].iloc[i]
        left_highs = df['high'].iloc[i-window:i]
        right_highs = df['high'].iloc[i+1:i+window+1]

        is_high = (center_high >= left_highs.max()) and (center_high >= right_highs.max())

        if is_high:
            # Calculate strength (how much higher than neighbors)
            avg_neighbor = (left_highs.mean() + right_highs.mean()) / 2
            strength = (center_high - avg_neighbor) / avg_neighbor if avg_neighbor > 0 else 0

            if strength >= min_strength:
                pivots.append({
                    'bar_index': i,
                    'price': center_high,
                    'type': 'swing_high',
                    'strength': min(strength, 1.0)
                })

        # Check for swing low
        center_low = df['low'].iloc[i]
        left_lows = df['low'].iloc[i-window:i]
        right_lows = df['low'].iloc[i+1:i+window+1]

        is_low = (center_low <= left_lows.min()) and (center_low <= right_lows.min())

        if is_low:
            avg_neighbor = (left_lows.mean() + right_lows.mean()) / 2
            strength = (avg_neighbor - center_low) / avg_neighbor if avg_neighbor > 0 else 0

            if strength >= min_strength:
                pivots.append({
                    'bar_index': i,
                    'price': center_low,
                    'type': 'swing_low',
                    'strength': min(strength, 1.0)
                })

    return pd.DataFrame(pivots)
```

### 1.2 Gann Cycle Features

**Source**: `engine/temporal/gann_cycles.py` (existing implementation)

| Feature Name | Type | Range | Description | Formula |
|-------------|------|-------|-------------|---------|
| `gann_cycle_phase` | float | 0-360 | Degrees in 360° cycle | `(days_since_anchor % 360) * 360/360` |
| `gann_square9_score` | float | 0.0-1.0 | Proximity to Square of 9 level | `1 - (abs(price - nearest_level) / tolerance)` |
| `gann_angle_score` | float | 0.0-1.0 | Adherence to 1×1 Gann angle | `1 - (abs(slope - 1) / 2)` |
| `gann_major_period_active` | bool | T/F | Within major Gann period | Check 3/7/8/9/12/144 day windows |
| `gann_confluence_score` | float | 0.0-1.0 | Multiple Gann cycles aligning | Weighted avg of cycle alignments |
| `acf_30d_score` | float | 0.0-1.0 | 30-day cycle autocorrelation | ACF peak at 30±5 days |
| `acf_60d_score` | float | 0.0-1.0 | 60-day cycle autocorrelation | ACF peak at 60±5 days |
| `acf_90d_score` | float | 0.0-1.0 | 90-day cycle autocorrelation | ACF peak at 90±5 days |

**Algorithm 1.2: Gann Confluence Score**

```python
def compute_gann_confluence_score(
    gann_features: Dict[str, float],
    weights: Dict[str, float] = {
        'square9': 0.30,
        'angle': 0.20,
        'acf_30d': 0.20,
        'acf_60d': 0.15,
        'acf_90d': 0.15
    }
) -> float:
    """
    Calculate Gann cycle confluence score from multiple indicators.

    Returns:
        score ∈ [0, 1] representing multi-cycle alignment strength
    """
    score = (
        weights['square9'] * gann_features.get('gann_square9_score', 0.0) +
        weights['angle'] * gann_features.get('gann_angle_score', 0.0) +
        weights['acf_30d'] * gann_features.get('acf_30d_score', 0.0) +
        weights['acf_60d'] * gann_features.get('acf_60d_score', 0.0) +
        weights['acf_90d'] * gann_features.get('acf_90d_score', 0.0)
    )

    # Bonus for major period alignment
    if gann_features.get('gann_major_period_active', False):
        score *= 1.1  # 10% bonus

    return min(score, 1.0)
```

**Gann Major Periods** (configurable):

```python
GANN_MAJOR_PERIODS = [
    3,    # Short-term reversal window
    7,    # Weekly cycle
    8,    # Gann octave
    9,    # Square of 9 base
    12,   # Zodiac/monthly cycle
    30,   # Monthly cycle
    60,   # Bi-monthly
    90,   # Quarterly
    144,  # Fibonacci × Gann (12²)
    180,  # Semi-annual
    360   # Annual cycle
]
```

### 1.3 Volatility Cycle Features

**Source**: `engine/temporal/volatility_cycles.py` (NEW)

| Feature Name | Type | Range | Description | Formula |
|-------------|------|-------|-------------|---------|
| `volatility_30d_std` | float | 0.0+ | 30-day rolling volatility | `std(log_returns[-30:])` |
| `volatility_phase` | str | enum | Current volatility regime | "low" / "rising" / "high" / "declining" |
| `volatility_z_score` | float | -3 to +3 | Z-score vs historical vol | `(current_vol - mean) / std` |
| `volatility_cycle_score` | float | 0.0-1.0 | Volatility cycle quality | Based on phase + z-score |
| `days_since_vol_extreme` | int | 0-200 | Days since vol spike/crash | Track volatility extremes |

**Algorithm 1.3: Volatility Cycle Score**

```python
def compute_volatility_cycle_score(
    df_1d: pd.DataFrame,
    lookback_window: int = 180
) -> Tuple[float, str, Dict]:
    """
    Calculate volatility cycle score and phase.

    Volatility cycles have 4 phases:
    1. Low Vol (Compression) - Coiling energy, breakout imminent
    2. Rising Vol (Expansion) - Trend accelerating
    3. High Vol (Climax) - Exhaustion approaching
    4. Declining Vol (Consolidation) - Returning to equilibrium

    Returns:
        (score, phase, metadata)
    """
    if len(df_1d) < lookback_window:
        return 0.0, "insufficient_data", {}

    # Calculate 30-day rolling volatility
    log_returns = np.log(df_1d['close'] / df_1d['close'].shift(1))
    vol_30d = log_returns.rolling(30).std().iloc[-30:]

    current_vol = vol_30d.iloc[-1]
    prev_vol = vol_30d.iloc[-10]  # 10 days ago

    # Historical volatility stats (180 days)
    hist_vol = log_returns.rolling(30).std().iloc[-lookback_window:]
    vol_mean = hist_vol.mean()
    vol_std = hist_vol.std()

    # Z-score
    z_score = (current_vol - vol_mean) / vol_std if vol_std > 0 else 0

    # Phase detection
    vol_trend = (current_vol - prev_vol) / prev_vol if prev_vol > 0 else 0

    if z_score < -0.5:  # Below mean volatility
        if vol_trend > 0.05:  # Starting to rise
            phase = "rising"
            score = 0.7  # Good for breakout setups
        else:
            phase = "low"
            score = 0.8  # Excellent for compression patterns
    elif z_score > 0.5:  # Above mean volatility
        if vol_trend < -0.05:  # Starting to decline
            phase = "declining"
            score = 0.6  # Moderate (consolidation phase)
        else:
            phase = "high"
            score = 0.3  # Poor (exhaustion/chop risk)
    else:  # Normal volatility
        phase = "normal"
        score = 0.5

    # Track days since extreme
    vol_extremes = hist_vol[hist_vol > vol_mean + 2 * vol_std]
    days_since_extreme = len(df_1d) - vol_extremes.index[-1] if len(vol_extremes) > 0 else 999

    metadata = {
        'current_vol': current_vol,
        'vol_mean': vol_mean,
        'vol_std': vol_std,
        'z_score': z_score,
        'vol_trend': vol_trend,
        'days_since_extreme': days_since_extreme
    }

    return score, phase, metadata
```

### 1.4 Emotional Cycle Features

**Source**: `engine/temporal/emotional_cycles.py` (NEW)

| Feature Name | Type | Range | Description | Formula |
|-------------|------|-------|-------------|---------|
| `emotional_phase` | str | enum | Current market psychology | "optimism" / "euphoria" / "anxiety" / "capitulation" / "disbelief" / "hope" |
| `emotional_cycle_score` | float | 0.0-1.0 | Cycle quality for timing | Based on phase + momentum |
| `days_in_current_phase` | int | 0-100 | Duration of current phase | Track phase persistence |
| `fear_greed_proxy` | float | 0-100 | Synthesized F&G index | Based on RSI, vol, sentiment |

**Algorithm 1.4: Emotional Cycle Score**

```python
def compute_emotional_cycle_score(
    df_1d: pd.DataFrame,
    regime_label: str = 'neutral'
) -> Tuple[float, str, Dict]:
    """
    Calculate emotional cycle phase and score.

    Uses price action + RSI + volume to infer market psychology.
    Maps to Wall Street Cheat Sheet emotional stages.

    Phases (in order):
    1. Disbelief (best buy) → 2. Hope → 3. Optimism → 4. Belief →
    5. Thrill → 6. Euphoria (worst buy) → 7. Complacency →
    8. Anxiety → 9. Denial → 10. Panic → 11. Capitulation (best buy) →
    12. Anger → 13. Depression → back to 1. Disbelief

    Returns:
        (score, phase, metadata)

    Score interpretation:
    - 0.8-1.0: Capitulation/Disbelief (strong buy zone)
    - 0.6-0.8: Hope/Optimism (good entry)
    - 0.4-0.6: Neutral (wait for clarity)
    - 0.2-0.4: Thrill/Complacency (reduce exposure)
    - 0.0-0.2: Euphoria/Panic (avoid new positions)
    """
    if len(df_1d) < 100:
        return 0.5, "insufficient_data", {}

    # Calculate indicators
    rsi_14 = compute_rsi(df_1d['close'], 14)
    current_rsi = rsi_14.iloc[-1]

    # Price momentum (90-day return)
    price_90d_return = (df_1d['close'].iloc[-1] / df_1d['close'].iloc[-90] - 1) * 100

    # Volume trend (recent vs historical)
    vol_recent = df_1d['volume'].iloc[-30:].mean()
    vol_hist = df_1d['volume'].iloc[-180:].mean()
    vol_ratio = vol_recent / vol_hist if vol_hist > 0 else 1.0

    # Volatility (30-day std)
    log_returns = np.log(df_1d['close'] / df_1d['close'].shift(1))
    current_vol = log_returns.iloc[-30:].std()
    hist_vol = log_returns.iloc[-180:].std()
    vol_spike = current_vol / hist_vol if hist_vol > 0 else 1.0

    # Phase detection logic
    if current_rsi > 75 and price_90d_return > 50:
        if vol_spike > 1.5 and vol_ratio > 1.3:
            phase = "euphoria"  # Peak excitement, dangerous
            score = 0.1
        else:
            phase = "thrill"  # Early excitement
            score = 0.3

    elif current_rsi > 60 and price_90d_return > 20:
        phase = "optimism"  # Healthy uptrend
        score = 0.7

    elif current_rsi > 50 and price_90d_return > 0:
        phase = "hope"  # Early recovery
        score = 0.75

    elif current_rsi < 25 and price_90d_return < -30:
        if vol_spike > 2.0 and vol_ratio > 1.5:
            phase = "capitulation"  # Peak fear, best buy
            score = 0.95
        else:
            phase = "panic"  # Fear intensifying
            score = 0.85

    elif current_rsi < 40 and price_90d_return < -10:
        phase = "anxiety"  # Growing concern
        score = 0.6

    elif current_rsi < 50 and price_90d_return < 0:
        phase = "disbelief"  # Early recovery skepticism
        score = 0.9

    else:
        phase = "neutral"
        score = 0.5

    # Fear & Greed proxy (0-100)
    # Simplified: RSI + vol + momentum weighted
    fg_proxy = (current_rsi * 0.5 +
                min(100, (1 - vol_spike) * 50) +
                min(50, max(0, price_90d_return + 50)))

    metadata = {
        'rsi': current_rsi,
        'price_90d_return': price_90d_return,
        'vol_ratio': vol_ratio,
        'vol_spike': vol_spike,
        'fear_greed_proxy': fg_proxy
    }

    return score, phase, metadata
```

---

## 2. Temporal Confluence Engine

### 2.1 Integration Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     ARCHETYPE DETECTION                      │
│                    (logic_v2_adapter.py)                     │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│                    BASE FUSION SCORE (K2)                    │
│              wyckoff × liquidity × momentum                  │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│                    GLOBAL SOFT FILTERS                       │
│          (liquidity, regime, session penalties)              │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│            ★ TEMPORAL FUSION LAYER (NEW) ★                  │
│                                                               │
│  ┌─────────────────┐  ┌─────────────────┐                  │
│  │  Fibonacci Time │  │   Gann Cycles   │                  │
│  │    Clusters     │  │  (Square of 9)  │                  │
│  │  (0-1 score)    │  │  (0-1 score)    │                  │
│  └────────┬────────┘  └────────┬────────┘                  │
│           │                     │                            │
│           └──────────┬──────────┘                            │
│                      ↓                                       │
│  ┌─────────────────────────────────────┐                    │
│  │     Volatility     Emotional        │                    │
│  │       Cycle          Cycle          │                    │
│  │    (0-1 score)    (0-1 score)       │                    │
│  └──────────────┬──────────────────────┘                    │
│                 ↓                                            │
│  ┌────────────────────────────────────────┐                 │
│  │  TEMPORAL CONFLUENCE SCORE (0-1)      │                 │
│  │  = weighted avg of 4 components       │                 │
│  └────────────────────────────────────────┘                 │
│                 ↓                                            │
│  ┌────────────────────────────────────────┐                 │
│  │  FUSION SCORE ADJUSTMENT               │                 │
│  │  fusion_score *= multiplier [0.85-1.15] │                 │
│  └────────────────────────────────────────┘                 │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│              ARCHETYPE THRESHOLD CHECK                       │
│              (adjusted fusion vs threshold)                  │
└─────────────────────────────────────────────────────────────┘
                              ↓
                        TRADE EXECUTION
```

### 2.2 Confluence Scoring Algorithm

```python
def calculate_temporal_confluence_score(
    fib_cluster_score: float,
    gann_confluence_score: float,
    volatility_cycle_score: float,
    emotional_cycle_score: float,
    weights: Dict[str, float] = {
        'fib_clusters': 0.40,
        'gann_cycles': 0.30,
        'volatility': 0.20,
        'emotional': 0.10
    }
) -> float:
    """
    Calculate overall temporal confluence score.

    Weighted average of 4 time-based components:
    1. Fibonacci time clusters (40%) - Primary temporal pressure
    2. Gann cycles (30%) - Classical harmonic timing
    3. Volatility cycles (20%) - Market energy state
    4. Emotional cycles (10%) - Psychology confirmation

    Returns:
        confluence_score ∈ [0, 1]
    """
    confluence = (
        weights['fib_clusters'] * fib_cluster_score +
        weights['gann_cycles'] * gann_confluence_score +
        weights['volatility'] * volatility_cycle_score +
        weights['emotional'] * emotional_cycle_score
    )

    return min(confluence, 1.0)
```

### 2.3 Fusion Weight Adjustment Rules

**Rule Engine**: Soft multiplicative adjustments based on temporal confluence

```python
def apply_temporal_fusion_adjustment(
    fusion_score: float,
    temporal_confluence: float,
    temporal_features: Dict,
    config: Dict
) -> Tuple[float, Dict]:
    """
    Apply temporal context adjustments to fusion score.

    Adjustment rules (multiplicative, bounded):
    1. High confluence (>0.70) + bullish phase → +10-15% boost
    2. Low confluence (<0.30) + ranging → -5-10% penalty
    3. Extreme volatility → -10% penalty (chop protection)
    4. Capitulation/disbelief phase → +8% boost
    5. Euphoria phase → -10% penalty

    Args:
        fusion_score: Base fusion score from K2 [0-1]
        temporal_confluence: Overall temporal score [0-1]
        temporal_features: Dict with all temporal sub-scores
        config: Temporal fusion configuration

    Returns:
        (adjusted_fusion_score, adjustment_metadata)
    """
    original_score = fusion_score
    adjustments_applied = []
    multiplier = 1.0

    # Extract features
    fib_score = temporal_features.get('fib_time_cluster_score', 0.0)
    gann_score = temporal_features.get('gann_confluence_score', 0.0)
    vol_phase = temporal_features.get('volatility_phase', 'normal')
    emotional_phase = temporal_features.get('emotional_phase', 'neutral')
    wyckoff_phase = temporal_features.get('wyckoff_phase_abc', 'neutral')

    # ═══════════════════════════════════════════════════════════
    # RULE 1: High Temporal Confluence + Bullish Wyckoff Phase
    # ═══════════════════════════════════════════════════════════
    if temporal_confluence > 0.70 and wyckoff_phase in ['C', 'D']:
        # Strong temporal pressure in accumulation/markup phase
        boost = config.get('high_confluence_boost', 1.15)
        multiplier *= boost
        adjustments_applied.append({
            'rule': 'high_confluence_bullish_phase',
            'multiplier': boost,
            'reason': f'Temporal confluence {temporal_confluence:.2f} in Phase {wyckoff_phase}'
        })

    # ═══════════════════════════════════════════════════════════
    # RULE 2: Low Confluence + Ranging Phase
    # ═══════════════════════════════════════════════════════════
    elif temporal_confluence < 0.30 and wyckoff_phase == 'B':
        # No temporal setup in consolidation → slight penalty
        penalty = config.get('low_confluence_penalty', 0.95)
        multiplier *= penalty
        adjustments_applied.append({
            'rule': 'low_confluence_ranging',
            'multiplier': penalty,
            'reason': f'Low confluence {temporal_confluence:.2f} in Phase B'
        })

    # ═══════════════════════════════════════════════════════════
    # RULE 3: Fibonacci Cluster + Gann Alignment (Double Confluence)
    # ═══════════════════════════════════════════════════════════
    if fib_score > 0.70 and gann_score > 0.65:
        # Both time systems agree → extra boost
        boost = config.get('double_confluence_boost', 1.12)
        multiplier *= boost
        adjustments_applied.append({
            'rule': 'fib_gann_double_confluence',
            'multiplier': boost,
            'reason': f'Fib {fib_score:.2f} + Gann {gann_score:.2f}'
        })

    # ═══════════════════════════════════════════════════════════
    # RULE 4: Extreme Volatility Phase
    # ═══════════════════════════════════════════════════════════
    if vol_phase == 'high':
        # High volatility = chop risk → penalty
        penalty = config.get('high_volatility_penalty', 0.90)
        multiplier *= penalty
        adjustments_applied.append({
            'rule': 'extreme_volatility',
            'multiplier': penalty,
            'reason': f'High volatility phase detected'
        })

    # ═══════════════════════════════════════════════════════════
    # RULE 5: Compression Phase (Low Vol + High Confluence)
    # ═══════════════════════════════════════════════════════════
    elif vol_phase in ['low', 'rising'] and temporal_confluence > 0.60:
        # Coiling energy before breakout → boost
        boost = config.get('compression_boost', 1.10)
        multiplier *= boost
        adjustments_applied.append({
            'rule': 'volatility_compression',
            'multiplier': boost,
            'reason': f'Low vol + confluence {temporal_confluence:.2f}'
        })

    # ═══════════════════════════════════════════════════════════
    # RULE 6: Emotional Cycle - Capitulation/Disbelief
    # ═══════════════════════════════════════════════════════════
    if emotional_phase in ['capitulation', 'disbelief', 'panic']:
        # Peak fear = best buy opportunity → boost
        boost = config.get('capitulation_boost', 1.08)
        multiplier *= boost
        adjustments_applied.append({
            'rule': 'emotional_capitulation',
            'multiplier': boost,
            'reason': f'Capitulation phase ({emotional_phase})'
        })

    # ═══════════════════════════════════════════════════════════
    # RULE 7: Emotional Cycle - Euphoria
    # ═══════════════════════════════════════════════════════════
    elif emotional_phase in ['euphoria', 'thrill']:
        # Peak greed = dangerous → penalty
        penalty = config.get('euphoria_penalty', 0.90)
        multiplier *= penalty
        adjustments_applied.append({
            'rule': 'emotional_euphoria',
            'multiplier': penalty,
            'reason': f'Euphoria phase ({emotional_phase})'
        })

    # ═══════════════════════════════════════════════════════════
    # APPLY BOUNDS: Force multiplier into [0.85, 1.15] range
    # ═══════════════════════════════════════════════════════════
    min_multiplier = config.get('min_multiplier', 0.85)
    max_multiplier = config.get('max_multiplier', 1.15)

    if multiplier < min_multiplier:
        adjustments_applied.append({
            'rule': 'floor_enforcement',
            'multiplier': min_multiplier / multiplier,
            'reason': f'Capped at {min_multiplier}x floor'
        })
        multiplier = min_multiplier

    if multiplier > max_multiplier:
        adjustments_applied.append({
            'rule': 'ceiling_enforcement',
            'multiplier': max_multiplier / multiplier,
            'reason': f'Capped at {max_multiplier}x ceiling'
        })
        multiplier = max_multiplier

    # Apply adjustment
    adjusted_score = fusion_score * multiplier

    # Final clipping to [0, 1]
    adjusted_score = max(0.0, min(1.0, adjusted_score))

    # Build metadata
    metadata = {
        'original_score': original_score,
        'final_score': adjusted_score,
        'total_multiplier': multiplier,
        'adjustment_pct': (multiplier - 1.0) * 100,
        'rules_triggered': len(adjustments_applied),
        'adjustments': adjustments_applied,
        'temporal_confluence': temporal_confluence,
        'temporal_features': temporal_features
    }

    return adjusted_score, metadata
```

---

## 3. Implementation Plan

### 3.1 File Structure

```
engine/
├── temporal/
│   ├── __init__.py                      # Module exports
│   ├── fib_time_clusters.py            # ENHANCED (add v2 algorithm)
│   ├── gann_cycles.py                   # EXISTS (use as-is)
│   ├── volatility_cycles.py            # NEW
│   ├── emotional_cycles.py              # NEW
│   ├── temporal_confluence.py           # NEW (main integration)
│   └── README.md                        # Documentation
│
├── fusion/
│   └── temporal.py                      # NEW (fusion adjustment layer)
│
└── features/
    ├── registry.py                      # UPDATE (add temporal features)
    └── builder.py                       # UPDATE (compute temporal features)
```

### 3.2 Phase 1: Feature Computation (Batch Mode)

**Goal**: Compute all temporal features for historical data and store in feature store.

**Script**: `bin/build_temporal_features.py`

```python
"""
Build temporal features for feature store (batch mode).

Usage:
    python bin/build_temporal_features.py \
        --input data/processed/features_mtf/btc_1h_2022_2024.parquet \
        --output data/processed/features_mtf/btc_1h_2022_2024_temporal.parquet \
        --config configs/temporal_config.json
"""

import pandas as pd
from engine.temporal.temporal_confluence import TemporalFeatureBuilder

def main():
    # Load OHLCV data
    df = pd.read_parquet(args.input)

    # Initialize builder
    builder = TemporalFeatureBuilder(config)

    # Compute all temporal features (vectorized where possible)
    df_temporal = builder.build_all_features(df)

    # Save to feature store
    df_temporal.to_parquet(args.output)

    print(f"Temporal features built: {len(df_temporal)} rows")
    print(f"Features added: {[c for c in df_temporal.columns if c not in df.columns]}")
```

**TemporalFeatureBuilder** (batch mode):

```python
class TemporalFeatureBuilder:
    """
    Batch mode temporal feature calculator.

    Computes all temporal features for historical data in one pass.
    Optimized for vectorized operations where possible.
    """

    def __init__(self, config: Dict):
        self.config = config
        self.temporal_config = config.get('temporal', {})

    def build_all_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Build all temporal features for DataFrame.

        Steps:
        1. Detect pivot points
        2. Extract Wyckoff event indices
        3. Calculate Fibonacci time cluster features
        4. Calculate Gann cycle features
        5. Calculate volatility cycle features
        6. Calculate emotional cycle features
        7. Calculate overall temporal confluence

        Returns:
            DataFrame with original + temporal columns
        """
        df = df.copy()

        # 1. Detect pivots
        pivots = detect_pivot_points(
            df,
            window=self.temporal_config.get('pivot_window', 5)
        )

        # 2. Process Wyckoff events (already in feature store)
        wyckoff_events = self._extract_wyckoff_events(df)

        # 3. Fibonacci time clusters (bar-by-bar)
        fib_features = []
        for idx in range(len(df)):
            fib_feat = self._calculate_fib_features_at_bar(
                df, pivots, wyckoff_events, idx
            )
            fib_features.append(fib_feat)

        fib_df = pd.DataFrame(fib_features, index=df.index)
        df = pd.concat([df, fib_df], axis=1)

        # 4. Gann cycles (vectorized where possible)
        gann_features = self._calculate_gann_features(df)
        df = pd.concat([df, gann_features], axis=1)

        # 5. Volatility cycles (rolling window)
        vol_features = self._calculate_volatility_features(df)
        df = pd.concat([df, vol_features], axis=1)

        # 6. Emotional cycles (rolling window)
        emotional_features = self._calculate_emotional_features(df)
        df = pd.concat([df, emotional_features], axis=1)

        # 7. Overall temporal confluence
        df['temporal_confluence_score'] = df.apply(
            lambda row: calculate_temporal_confluence_score(
                row['fib_time_cluster_score'],
                row['gann_confluence_score'],
                row['volatility_cycle_score'],
                row['emotional_cycle_score']
            ), axis=1
        )

        return df
```

### 3.3 Phase 2: Stream Mode Integration

**Goal**: Compute temporal features incrementally during live trading/backtesting.

**Integration Point**: `engine/archetypes/logic_v2_adapter.py`

```python
# In LogicV2Adapter.detect() method

def detect(self, context: RuntimeContext) -> Optional[TradeSignal]:
    """
    Enhanced detection with temporal fusion layer.
    """
    # ... existing base fusion logic ...

    fusion_score = self._fusion(context.row)
    fusion_score, wyckoff_meta = self._apply_wyckoff_event_boosts(context.row, fusion_score)

    # Apply global soft filters (existing)
    if use_soft_liquidity and liquidity_score < self.min_liquidity:
        fusion_score *= 0.7

    # ═══════════════════════════════════════════════════════════
    # NEW: Apply Temporal Fusion Layer
    # ═══════════════════════════════════════════════════════════
    if self.config.get('temporal_fusion', {}).get('enabled', False):
        from engine.fusion.temporal import apply_temporal_fusion_adjustment

        # Calculate temporal confluence (incremental)
        temporal_features = self._get_temporal_features(context)
        temporal_confluence = calculate_temporal_confluence_score(
            temporal_features['fib_time_cluster_score'],
            temporal_features['gann_confluence_score'],
            temporal_features['volatility_cycle_score'],
            temporal_features['emotional_cycle_score']
        )

        # Apply adjustment
        fusion_score, temporal_meta = apply_temporal_fusion_adjustment(
            fusion_score,
            temporal_confluence,
            temporal_features,
            self.config.get('temporal_fusion', {})
        )

        # Log if adjustment significant
        if abs(temporal_meta['adjustment_pct']) > 5.0:
            logger.info(f"[TEMPORAL FUSION] {temporal_meta['adjustment_pct']:+.1f}% "
                       f"adjustment: {temporal_meta['original_score']:.3f} → "
                       f"{temporal_meta['final_score']:.3f}")

    # Continue with archetype dispatch...
    return self._dispatch_to_archetypes(context, fusion_score)
```

### 3.4 Feature Parity Validation

**Critical**: Batch and stream mode MUST produce identical results.

**Validation Script**: `bin/validate_temporal_parity.py`

```python
"""
Validate temporal feature parity between batch and stream modes.

Tests:
1. Batch vs stream feature comparison (row-by-row)
2. Temporal confluence score consistency
3. Fusion adjustment parity
4. Performance benchmarking

Acceptance criteria:
- Max difference < 1e-6 for float features
- 100% match for boolean/categorical features
- Stream mode ≤ 2x slower than batch mode
"""

def test_parity(df_batch: pd.DataFrame, df_stream: pd.DataFrame):
    """Compare batch vs stream results."""

    temporal_cols = [c for c in df_batch.columns if 'temporal_' in c or 'fib_' in c or 'gann_' in c]

    for col in temporal_cols:
        diff = (df_batch[col] - df_stream[col]).abs()
        max_diff = diff.max()

        assert max_diff < 1e-6, f"Parity violation in {col}: max_diff={max_diff}"

    print("✓ Feature parity validated")
```

---

## 4. Configuration

### 4.1 Configuration Schema

```json
{
  "temporal_fusion": {
    "enabled": true,
    "version": "v2.0",

    "fibonacci_time_clusters": {
      "enabled": true,
      "fib_levels": [21, 34, 55, 89, 144, 233],
      "tolerance_bars": 2,
      "min_wyckoff_confidence": 0.65,
      "lookback_bars": 500,
      "pivot_window": 5,
      "pivot_min_strength": 0.6,
      "wyckoff_event_weight": 1.3,
      "pivot_weight": 1.0
    },

    "gann_cycles": {
      "enabled": true,
      "square9_step": 9.0,
      "square9_tolerance_pct": 2.0,
      "gann_angle_lookback": 24,
      "major_periods": [3, 7, 8, 9, 12, 30, 60, 90, 144, 180, 360],
      "acf_lookback_days": 180,
      "target_cycles": [30, 60, 90],
      "cycle_tolerance_days": 5
    },

    "volatility_cycles": {
      "enabled": true,
      "rolling_window": 30,
      "historical_lookback": 180,
      "phase_transition_threshold": 0.05
    },

    "emotional_cycles": {
      "enabled": true,
      "rsi_period": 14,
      "momentum_window": 90,
      "volume_lookback": 180
    },

    "confluence_weights": {
      "fib_clusters": 0.40,
      "gann_cycles": 0.30,
      "volatility": 0.20,
      "emotional": 0.10
    },

    "fusion_adjustments": {
      "min_multiplier": 0.85,
      "max_multiplier": 1.15,
      "high_confluence_boost": 1.15,
      "low_confluence_penalty": 0.95,
      "double_confluence_boost": 1.12,
      "high_volatility_penalty": 0.90,
      "compression_boost": 1.10,
      "capitulation_boost": 1.08,
      "euphoria_penalty": 0.90
    },

    "logging": {
      "log_adjustments_above_pct": 5.0,
      "log_high_confluence_events": true,
      "telemetry_enabled": true
    }
  }
}
```

### 4.2 Example Production Config

```json
{
  "temporal_fusion": {
    "enabled": true,

    "fibonacci_time_clusters": {
      "enabled": true,
      "fib_levels": [21, 34, 55, 89, 144],
      "tolerance_bars": 2,
      "min_wyckoff_confidence": 0.70,
      "lookback_bars": 500,
      "pivot_window": 5,
      "pivot_min_strength": 0.65
    },

    "gann_cycles": {
      "enabled": true,
      "square9_step": 9.0,
      "square9_tolerance_pct": 2.0
    },

    "volatility_cycles": {
      "enabled": true
    },

    "emotional_cycles": {
      "enabled": false
    },

    "confluence_weights": {
      "fib_clusters": 0.50,
      "gann_cycles": 0.35,
      "volatility": 0.15,
      "emotional": 0.00
    },

    "fusion_adjustments": {
      "min_multiplier": 0.90,
      "max_multiplier": 1.12,
      "high_confluence_boost": 1.12,
      "low_confluence_penalty": 0.95
    }
  }
}
```

---

## 5. Validation & Testing

### 5.1 Unit Tests

**File**: `tests/unit/temporal/test_temporal_confluence.py`

```python
def test_fib_cluster_score_bounds():
    """Test Fibonacci cluster score is bounded [0, 1]."""

def test_gann_confluence_calculation():
    """Test Gann confluence combines sub-scores correctly."""

def test_temporal_fusion_adjustment_bounds():
    """Test fusion adjustment multiplier is bounded [0.85, 1.15]."""

def test_capitulation_phase_boost():
    """Test capitulation emotional phase triggers boost."""

def test_euphoria_phase_penalty():
    """Test euphoria emotional phase triggers penalty."""

def test_double_confluence_bonus():
    """Test Fib + Gann alignment triggers extra boost."""

def test_feature_parity_batch_vs_stream():
    """Test batch and stream modes produce identical features."""
```

### 5.2 Integration Tests

**File**: `tests/integration/test_temporal_fusion_integration.py`

```python
def test_temporal_fusion_in_backtest():
    """Test temporal fusion integrates cleanly with full backtest."""

def test_temporal_fusion_on_luna_crash():
    """Test temporal layer detects LUNA crash confluence."""

def test_temporal_fusion_on_ftx_collapse():
    """Test temporal layer detects FTX collapse signals."""

def test_temporal_fusion_on_june_18_2022():
    """Test temporal layer on 3AC liquidation cascade."""
```

### 5.3 Historical Validation Scenarios

**Test Case 1: LUNA Crash (May 2022)**
- **Expected**: High Fibonacci cluster score (55 bars from BC)
- **Expected**: Capitulation emotional phase detected
- **Expected**: 10-15% boost to short entries

**Test Case 2: FTX Collapse (November 2022)**
- **Expected**: Gann 30-day cycle alignment with funding extreme
- **Expected**: High volatility phase (VIX spike)
- **Expected**: 8-12% boost to panic-driven shorts

**Test Case 3: June 18, 2022 (3AC Liquidation)**
- **Expected**: Double confluence (Fib + Gann) near capitulation
- **Expected**: Extreme volatility + emotional panic
- **Expected**: 12-15% boost (all systems aligned)

### 5.4 A/B Testing Plan

**Baseline**: Archetypes WITHOUT temporal fusion (current production)

**Test**: Same config WITH temporal fusion enabled

**Test Period**: 2022-2024 full dataset (bull + bear + chop)

**Metrics**:

| Metric | Without Temporal | With Temporal | Target Δ |
|--------|-----------------|---------------|---------|
| Profit Factor | 2.37 | ? | +2-5% |
| Win Rate | 65.8% | ? | +1-3% |
| Sharpe Ratio | 1.83 | ? | +5-10% |
| Max Drawdown | -12.4% | ? | -10-15% (smaller) |
| Trade Count | 1,234 | ? | ±10% max |
| Avg Trade Duration | 4.2 days | ? | -5-10% (faster) |

**Success Criteria**:
- PF improvement ≥ +2%
- Trade count change < 15%
- No Sharpe degradation
- At least 1 metric improves by ≥ 5%

### 5.5 Ablation Study

**Goal**: Determine which temporal components contribute most to performance.

**Method**: Disable features one at a time and re-run backtest.

| Feature Disabled | PF Δ | WR Δ | Sharpe Δ | Interpretation |
|-----------------|------|------|----------|----------------|
| Fibonacci clusters | ? | ? | ? | Critical / Minor / Neutral |
| Gann cycles | ? | ? | ? | Critical / Minor / Neutral |
| Volatility cycles | ? | ? | ? | Critical / Minor / Neutral |
| Emotional cycles | ? | ? | ? | Critical / Minor / Neutral |
| All temporal | -X% | -X% | -X% | (baseline comparison) |

**Action**: Remove features with minimal/negative impact to simplify production config.

---

## 6. Example Scenarios

### Scenario 1: Strong Temporal Confluence (Bullish)

**Context**:
- Current bar is 55 bars from Spring-A event
- Price at Gann Square of 9 level (54,000)
- Volatility compressing (low phase)
- Emotional phase = "disbelief"
- Wyckoff Phase = D (markup starting)

**Temporal Features**:
```python
{
    'fib_time_cluster_score': 0.82,  # 55 bars = Fib level
    'gann_confluence_score': 0.76,    # Square 9 + angle alignment
    'volatility_cycle_score': 0.85,   # Compression phase
    'emotional_cycle_score': 0.90     # Disbelief = buy zone
}
```

**Temporal Confluence**: `0.40×0.82 + 0.30×0.76 + 0.20×0.85 + 0.10×0.90 = 0.786`

**Fusion Adjustment**:
- Rule 1 (High confluence + Phase D): 1.15× boost
- Rule 3 (Fib + Gann double confluence): 1.12× boost
- Rule 5 (Compression): 1.10× boost
- Rule 6 (Disbelief): 1.08× boost
- Combined multiplier: `1.15 × 1.12 × 1.10 × 1.08 = 1.52` → **CAPPED at 1.15×**

**Result**: Base fusion score 0.42 → **0.483** (+15%)

---

### Scenario 2: Weak Temporal Setup (Ranging)

**Context**:
- No recent Wyckoff events (200+ bars)
- Price between Square of 9 levels
- Normal volatility
- Emotional phase = "complacency"
- Wyckoff Phase = B (ranging)

**Temporal Features**:
```python
{
    'fib_time_cluster_score': 0.18,  # No cluster
    'gann_confluence_score': 0.32,    # Weak alignment
    'volatility_cycle_score': 0.50,   # Normal
    'emotional_cycle_score': 0.45     # Complacency
}
```

**Temporal Confluence**: `0.40×0.18 + 0.30×0.32 + 0.20×0.50 + 0.10×0.45 = 0.313`

**Fusion Adjustment**:
- Rule 2 (Low confluence + Phase B): 0.95× penalty

**Result**: Base fusion score 0.42 → **0.399** (-5%)

---

### Scenario 3: Dangerous Timing (Euphoria)

**Context**:
- Price 2.5× above 90-day MA
- Recent BC + UTAD (5 bars ago)
- Extreme volatility (spike)
- Emotional phase = "euphoria"
- Wyckoff Phase = E (markdown imminent)

**Temporal Features**:
```python
{
    'fib_time_cluster_score': 0.28,  # Weak
    'gann_confluence_score': 0.41,    # Moderate
    'volatility_cycle_score': 0.15,   # Extreme (danger)
    'emotional_cycle_score': 0.10     # Euphoria (danger)
}
```

**Temporal Confluence**: `0.40×0.28 + 0.30×0.41 + 0.20×0.15 + 0.10×0.10 = 0.266`

**Fusion Adjustment**:
- Rule 4 (High volatility): 0.90× penalty
- Rule 7 (Euphoria): 0.90× penalty
- Combined multiplier: `0.90 × 0.90 = 0.81` → **FLOORED at 0.85×**

**Result**: Base fusion score 0.42 → **0.357** (-15%)

---

## 7. Risk Mitigation

### Risk 1: Overfitting to Historical Cycles

**Problem**: Time cycles may be regime-specific (e.g., 4-year halving only applies to BTC).

**Mitigation**:
- Make cycle periods configurable per asset
- A/B test with temporal layer disabled
- Monitor performance degradation post-deployment
- Use walk-forward validation (train on 2022, test on 2023, validate on 2024)

### Risk 2: Lookahead Bias in Pivot Detection

**Problem**: Detecting pivots requires future bars (window=5 means 5 bars ahead).

**Mitigation**:
- Use confirmed pivots only (wait for window to complete)
- Stream mode: pivot detection lags by `window` bars
- Feature parity test: batch = stream (no sneaky future data)
- Document pivot lag explicitly in feature metadata

### Risk 3: Computational Cost (Stream Mode)

**Problem**: Temporal features require heavy computation (ACF, rolling stats).

**Mitigation**:
- Cache intermediate results (pivots, Wyckoff events)
- Incremental updates (don't recompute full history each bar)
- Pre-compute ACF in batch mode, interpolate in stream mode
- Feature flag: `temporal.simplified_mode = True` (skip expensive features)

### Risk 4: Config Complexity

**Problem**: 30+ tunable parameters → overfitting / unmaintainable.

**Mitigation**:
- Start with defaults (no optimization)
- Only tune top 5 most sensitive parameters (via ablation)
- Use hierarchical config (global defaults → asset overrides)
- Document parameter sensitivity analysis

### Risk 5: Temporal Features Missing

**Problem**: Feature store may not have Wyckoff events / pivots pre-computed.

**Mitigation**:
- Graceful fallbacks: `row.get('fib_time_cluster_score', 0.0)`
- Log warnings if < 80% feature coverage
- Auto-disable temporal fusion if insufficient data
- Backfill script: `bin/backfill_temporal_features.py`

---

## 8. Rollout Plan

### Phase 1: Feature Development (Week 1-2)
- [ ] Implement `engine/temporal/volatility_cycles.py`
- [ ] Implement `engine/temporal/emotional_cycles.py`
- [ ] Enhance `engine/temporal/fib_time_clusters.py` (add v2 algorithm)
- [ ] Implement `engine/temporal/temporal_confluence.py` (batch mode)
- [ ] Write unit tests (target: 90%+ coverage)

### Phase 2: Integration (Week 3)
- [ ] Implement `engine/fusion/temporal.py` (adjustment layer)
- [ ] Integrate into `logic_v2_adapter.py`
- [ ] Add feature columns to `engine/features/registry.py`
- [ ] Update `engine/features/builder.py` (batch computation)
- [ ] Feature parity validation script

### Phase 3: Validation (Week 4)
- [ ] Backfill temporal features for 2022-2024 data
- [ ] Run A/B backtest (temporal ON vs OFF)
- [ ] Historical scenario validation (LUNA, FTX, June 18)
- [ ] Ablation study (determine critical features)
- [ ] Performance benchmarking (batch vs stream)

### Phase 4: Production Deployment (Week 5)
- [ ] Create production config (conservative settings)
- [ ] Deploy to staging environment
- [ ] Paper trading validation (1 week live data)
- [ ] Rollout to 10% of production traffic
- [ ] Monitor telemetry (adjustment frequency, magnitude)
- [ ] Full rollout if metrics green

---

## 9. Success Metrics

### Performance Improvements (vs Baseline)
- ✅ **Profit Factor**: +2-5%
- ✅ **Win Rate**: +1-3%
- ✅ **Sharpe Ratio**: +5-10%
- ✅ **Max Drawdown**: -10-15% (smaller)
- ✅ **Trade Quality**: Higher R-multiples on temporal confluence trades

### Operational Metrics
- ✅ **Feature Parity**: 100% match (batch = stream, max diff < 1e-6)
- ✅ **Computation Time**: Stream mode ≤ 2× batch mode
- ✅ **Trade Count**: ±10% max (no over-filtering)
- ✅ **Adjustment Frequency**: 20-30% of trades get ≥5% adjustment
- ✅ **Config Stability**: No parameter changes needed for 90 days

### Integration Quality
- ✅ **Test Coverage**: ≥90% for temporal modules
- ✅ **Documentation**: Complete (architecture + API docs)
- ✅ **Backward Compatibility**: No breaking changes to existing archetypes
- ✅ **Feature Flag Control**: Can enable/disable without code changes
- ✅ **Interpretability**: Operators understand temporal scores

---

## 10. Future Enhancements (Phase 3+)

### 10.1 Machine Learning Rule Discovery
- Train XGBoost to learn optimal fusion adjustments
- Input: All temporal features + regime + archetype type
- Output: Optimal multiplier [0.85, 1.15]
- Use SHAP to explain model decisions

### 10.2 Adaptive Cycle Periods
- Auto-detect dominant cycles via FFT/wavelets
- Asset-specific cycle tuning (ETH ≠ BTC ≠ SOL)
- Regime-aware cycle weights (bull market ≠ bear market)

### 10.3 Cross-Asset Temporal Correlation
- Track temporal phase across BTC, ETH, SPX, DXY
- Confluence bonus when multiple assets align
- Leading/lagging relationship detection

### 10.4 Event-Driven Overrides
- NFP days: Disable temporal fusion (too volatile)
- Fed meetings: Boost Wyckoff phase signals
- Options expiry: Boost temporal cluster signals (pinning effects)
- Halving events: Increase Gann cycle weights

### 10.5 Temporal Prediction Layer
- Forecast next Fibonacci cluster window
- Predict volatility phase transitions
- Emotional cycle phase prediction (ML-based)
- "Oracle Whisper": Alert when multi-cycle alignment approaching

---

## 11. References & Resources

### Internal Documentation
- **Fibonacci Time Clusters**: `docs/FIB_TIME_CLUSTER_SPEC.md`
- **Temporal Fusion Layer**: `docs/TEMPORAL_FUSION_SPEC.md`
- **Wyckoff Events**: `engine/wyckoff/events.py`, `WYCKOFF_EVENTS_IMPLEMENTATION_PLAN.md`
- **Feature Registry**: `engine/features/registry.py`
- **Archetype Logic**: `engine/archetypes/logic_v2_adapter.py`

### External Research
- **Gann Square of 9**: Bramesh's Technical Analysis, 2024 (Web search results)
- **Fibonacci Time Analysis**: CME Group, OANDA educational resources
- **Bitcoin Volatility Cycles**: Fidelity Digital Assets, Bitso research
- **Emotional Cycles**: Wall Street Cheat Sheet, Market Psychology Chart

### Academic Papers
- **LPPLS Models**: Sornette et al., "Log-Periodic Power Laws"
- **Autocorrelation Analysis**: Box-Jenkins time series methodology
- **Volatility Regimes**: Ang & Bekaert (2002), "Regime Switches in Interest Rates"

### Trader Sources
- **@Wyckoff_Insider**: Accumulation/distribution cycles, rhythm patterns
- **@Moneytaur**: Smart money timing, institutional re-entry windows
- **@ZeroIKA**: Frequency-domain analysis, harmonic cycles

---

## 12. Appendix: Mathematical Foundations

### A. Autocorrelation Function (ACF)

```
ACF(k) = E[(X_t - μ)(X_{t-k} - μ)] / σ²

where:
- k = lag (in bars/days)
- μ = mean of time series
- σ² = variance
```

**Interpretation**: ACF(k) ∈ [-1, 1]
- ACF > 0.15 at k = 30 → 30-day cycle present
- Peak detection: Find local maxima in ACF around target lags

### B. Gann Square of 9

```
Level_n = Start_Price + (n × Step)

where:
- Start_Price = anchor (e.g., major low)
- Step = 9 (configurable)
- n = integer multiplier

Proximity Score = 1 - (|Price - Nearest_Level| / Price) / Tolerance
```

**Example**: Price = 54,005, Nearest Level = 54,000, Tolerance = 2%
- Distance = |54,005 - 54,000| / 54,005 = 0.0093%
- Score = 1 - (0.0093% / 2%) = 0.9954 ≈ 1.0

### C. Volatility Z-Score

```
Z = (σ_current - μ_hist) / σ_hist

where:
- σ_current = current 30-day volatility
- μ_hist = mean of historical volatilities (180 days)
- σ_hist = std of historical volatilities

Interpretation:
- Z < -0.5: Low volatility (compression)
- |Z| < 0.5: Normal volatility
- Z > 0.5: High volatility (expansion)
- Z > 2.0: Extreme volatility (danger)
```

### D. Temporal Confluence Score

```
C = w_1·S_fib + w_2·S_gann + w_3·S_vol + w_4·S_emo

where:
- S_* = component scores ∈ [0, 1]
- w_* = weights (sum to 1.0)
- C ∈ [0, 1] (final confluence)

Default weights:
- w_1 = 0.40 (Fibonacci clusters)
- w_2 = 0.30 (Gann cycles)
- w_3 = 0.20 (Volatility cycles)
- w_4 = 0.10 (Emotional cycles)
```

### E. Fusion Score Adjustment

```
F_adjusted = F_base × M

where:
- F_base = base fusion score ∈ [0, 1]
- M = multiplier ∈ [0.85, 1.15] (bounded)
- F_adjusted ∈ [0, 1] (clipped)

Multiplier calculation:
M = Π(m_i) for all triggered rules
m_i ∈ {0.90, 0.95, 1.00, 1.05, 1.08, 1.10, 1.12, 1.15}

Bounds enforcement:
M_final = clip(M, 0.85, 1.15)
```

---

**Document Version**: 2.0
**Author**: System Architect (Claude)
**Date**: 2025-11-24
**Status**: Production-Ready Design (Pending Implementation)
**Next Review**: After Phase 1 completion (feature development)
