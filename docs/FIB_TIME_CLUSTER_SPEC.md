# Fibonacci Time Cluster Specification

## Executive Summary

**Purpose**: Identify temporal confluence zones around Fibonacci intervals from major Wyckoff events and pivot points to detect high-probability reversal/continuation zones.

**Status**: PARTIAL (Legacy implementation exists in `bull_machine/strategy/temporal_fib_clusters.py`, needs integration with Wyckoff events and feature store)

**Implementation Location**: `engine/temporal/fib_time_clusters.py` (new v2 implementation)

**Feature Store Integration**: Phase 2 (Ghost Module Revival)

---

## Background

Time is NOT prediction—it's PRESSURE. Fibonacci time clusters identify when accumulated temporal tension must resolve into price movement. This concept derives from:

1. **Wyckoff Insider's Rhythm Patterns**: Markets move in natural cycles, not random walks
2. **Elliott Wave Time Projections**: Fibonacci ratios appear in time just as in price
3. **Institutional Order Flow**: Smart money operates on predictable time windows (options expiry, rebalancing, etc.)

The existing implementation (v1.6.1) uses swing pivots only. **Phase 2 upgrade** integrates Wyckoff events (SC, BC, Spring-A, UTAD) as primary reference points, dramatically increasing signal quality.

---

## Input Requirements

### 1. Wyckoff Events (from Feature Store)
**Source**: `engine/wyckoff/events.py` → Feature store columns

| Event Type | Column Name | Confidence Column | Description |
|-----------|-------------|-------------------|-------------|
| Selling Climax | `wyckoff_sc` | `wyckoff_sc_confidence` | Capitulation at lows (accumulation start) |
| Buying Climax | `wyckoff_bc` | `wyckoff_bc_confidence` | Euphoria at highs (distribution start) |
| Spring Type A | `wyckoff_spring_a` | `wyckoff_spring_a_confidence` | Deep fake breakdown (trap reversal) |
| Spring Type B | `wyckoff_spring_b` | `wyckoff_spring_b_confidence` | Shallow spring (test of support) |
| UTAD | `wyckoff_utad` | `wyckoff_utad_confidence` | Upthrust After Distribution (bull trap) |
| Last Point of Support | `wyckoff_lps` | `wyckoff_lps_confidence` | Final test before markup |
| Last Point of Supply | `wyckoff_lpsy` | `wyckoff_lpsy_confidence` | Final rally before markdown |

**Minimum Confidence Threshold**: 0.65 (configurable)

### 2. Pivot Points (Legacy, still used)
**Source**: `detect_pivot_points()` function

- Swing highs/lows detected via windowed local extrema (default window=5 bars)
- Used as secondary reference points when Wyckoff events are sparse

### 3. Current Bar Context
- **Bar index**: Current position in time series (integer)
- **Timeframe**: `1H`, `4H`, `1D` (affects tolerance and scoring)

---

## Fibonacci Time Intervals

### Base Fibonacci Sequence (1H timeframe)
```python
FIB_LEVELS_1H = [21, 34, 55, 89, 144, 233]
```

### Tolerance Bands (±N bars)
| Timeframe | Tolerance | Reasoning |
|-----------|-----------|-----------|
| 1H | ±2 bars | 2 hours slack for noise |
| 4H | ±1 bar | 4 hours slack (already smoothed) |
| 1D | ±1 bar | 1 day slack (weekly cycles) |

**Design Choice**: Tighter tolerance on HTF prevents false clusters from market noise.

---

## Output Features

All features written to feature store with prefix `fib_time_`:

### 1. Time Distance Features (Integer Bars)
| Feature Name | Type | Description | Range |
|-------------|------|-------------|-------|
| `bars_since_sc` | int | Bars since last Selling Climax | 0 - 500+ |
| `bars_since_bc` | int | Bars since last Buying Climax | 0 - 500+ |
| `bars_since_spring_a` | int | Bars since last Spring-A | 0 - 500+ |
| `bars_since_utad` | int | Bars since last UTAD | 0 - 500+ |
| `bars_since_last_wyckoff_event` | int | Min of all above (closest event) | 0 - 500+ |
| `bars_since_last_pivot` | int | Bars since nearest swing pivot | 0 - 200 |

### 2. Cluster Score (Float 0-1)
**Feature**: `fib_time_cluster_score`

**Formula**:
```python
def compute_fib_time_cluster_score(
    bars_since_events: Dict[str, int],  # {'sc': 55, 'bc': 144, 'pivot_1': 89, ...}
    fib_levels: List[int] = [21, 34, 55, 89, 144, 233],
    tolerance: int = 2
) -> float:
    """
    Calculate temporal confluence score based on Fibonacci alignment.

    Returns:
        0.0 - 1.0, where:
        - 0.0-0.3: Weak/no temporal confluence
        - 0.3-0.6: Moderate confluence (1-2 matches)
        - 0.6-0.8: Strong confluence (3+ matches)
        - 0.8-1.0: Extreme confluence (4+ matches, tight convergence)
    """
    score = 0.0
    matches = []

    for event_name, event_bars in bars_since_events.items():
        for fib_level in fib_levels:
            # Check if event distance aligns with Fib level (within tolerance)
            if abs(event_bars - fib_level) <= tolerance:
                # Weight by:
                # 1. Fibonacci importance (higher = more important)
                # 2. Event importance (Wyckoff > pivot)
                fib_weight = fib_level / max(fib_levels)  # 0.09 (21) to 1.0 (233)

                event_weight = 1.0
                if 'wyckoff' in event_name.lower():
                    event_weight = 1.2  # 20% bonus for Wyckoff events
                elif 'sc' in event_name or 'spring' in event_name:
                    event_weight = 1.3  # 30% bonus for trap events

                match_score = fib_weight * event_weight
                matches.append({
                    'event': event_name,
                    'fib_level': fib_level,
                    'event_bars': event_bars,
                    'score': match_score
                })
                score += match_score

    # Normalize by number of events checked (prevent score inflation)
    if len(bars_since_events) > 0:
        score = score / (len(bars_since_events) * 0.8)  # Divide by ~max possible

    # Bonus for multiple matches (confluence multiplier)
    if len(matches) >= 3:
        score *= 1.2  # 20% bonus for 3+ matches
    elif len(matches) >= 2:
        score *= 1.1  # 10% bonus for 2 matches

    # Cap at 1.0
    return min(score, 1.0)
```

**Example Calculation**:
```python
# Scenario: Current bar is 55 bars from SC, 89 bars from last pivot
bars_since_events = {
    'wyckoff_sc': 55,      # Matches Fib(55) EXACTLY
    'pivot_high_1': 89,    # Matches Fib(89) EXACTLY
    'wyckoff_bc': 200      # No match
}

# Calculation:
# - SC match: (55/233) * 1.3 = 0.307 (SC event bonus)
# - Pivot match: (89/233) * 1.0 = 0.382
# - Base score: (0.307 + 0.382) / (3 * 0.8) = 0.287
# - Confluence bonus (2 matches): 0.287 * 1.1 = 0.316
#
# Result: fib_time_cluster_score = 0.316 (moderate confluence)
```

### 3. Cluster Zone Flag (Boolean)
**Feature**: `is_fib_time_cluster_zone`

**Logic**:
```python
is_fib_time_cluster_zone = (fib_time_cluster_score >= 0.7)
```

**Interpretation**:
- `True`: High temporal pressure zone (expect acceleration/reversal)
- `False`: No significant temporal confluence

### 4. Cluster Metadata (for logging/debugging)
**Features** (optional, for telemetry):
- `fib_time_cluster_match_count` (int): Number of Fib alignments detected
- `fib_time_cluster_dominant_fib` (int): Most common Fib level in cluster (e.g., 55, 89)
- `fib_time_cluster_tightness` (float 0-1): How tightly clustered matches are (1.0 = all within ±1 bar)

---

## Algorithm Pseudocode

```python
def calculate_fib_time_features(
    df: pd.DataFrame,
    wyckoff_events: pd.DataFrame,
    pivot_points: pd.DataFrame,
    current_idx: int,
    config: Dict
) -> Dict[str, Any]:
    """
    Calculate Fibonacci time cluster features for current bar.

    Args:
        df: OHLCV DataFrame (full history)
        wyckoff_events: Boolean columns for Wyckoff events
        pivot_points: Bar indices of swing pivots
        current_idx: Current bar index
        config: {
            'fib_levels': [21, 34, 55, 89, 144, 233],
            'tolerance': 2,
            'min_confidence': 0.65,
            'lookback_bars': 500
        }

    Returns:
        Dict with all fib_time_* features
    """
    # STEP 1: Extract Wyckoff event bar indices (with confidence filtering)
    wyckoff_bars = {}

    for event in ['sc', 'bc', 'spring_a', 'spring_b', 'utad', 'lps', 'lpsy']:
        event_col = f'wyckoff_{event}'
        conf_col = f'wyckoff_{event}_confidence'

        # Find most recent occurrence with confidence >= threshold
        lookback_slice = df.iloc[max(0, current_idx - config['lookback_bars']):current_idx]

        if event_col in lookback_slice.columns:
            # Filter by confidence
            valid_events = lookback_slice[
                (lookback_slice[event_col] == True) &
                (lookback_slice[conf_col] >= config['min_confidence'])
            ]

            if len(valid_events) > 0:
                last_event_idx = valid_events.index[-1]
                bars_since = current_idx - last_event_idx
                wyckoff_bars[f'wyckoff_{event}'] = bars_since

    # STEP 2: Extract pivot bar indices (most recent N pivots)
    pivot_bars = {}
    if pivot_points is not None:
        recent_pivots = pivot_points[pivot_points['bar_index'] < current_idx].tail(10)
        for i, pivot in enumerate(recent_pivots.itertuples()):
            bars_since = current_idx - pivot.bar_index
            pivot_bars[f'pivot_{i}'] = bars_since

    # STEP 3: Compute cluster score
    all_events = {**wyckoff_bars, **pivot_bars}
    cluster_score = compute_fib_time_cluster_score(
        all_events,
        config['fib_levels'],
        config['tolerance']
    )

    # STEP 4: Build output features
    return {
        # Distance features
        'bars_since_sc': wyckoff_bars.get('wyckoff_sc', 999),
        'bars_since_bc': wyckoff_bars.get('wyckoff_bc', 999),
        'bars_since_spring_a': wyckoff_bars.get('wyckoff_spring_a', 999),
        'bars_since_utad': wyckoff_bars.get('wyckoff_utad', 999),
        'bars_since_last_wyckoff_event': min(wyckoff_bars.values()) if wyckoff_bars else 999,
        'bars_since_last_pivot': min(pivot_bars.values()) if pivot_bars else 999,

        # Cluster features
        'fib_time_cluster_score': cluster_score,
        'is_fib_time_cluster_zone': cluster_score >= 0.7,

        # Metadata (for logging)
        'fib_time_cluster_match_count': len([
            1 for bars in all_events.values()
            for fib in config['fib_levels']
            if abs(bars - fib) <= config['tolerance']
        ]),
    }
```

---

## Integration Points

### 1. Feature Builder Pipeline
**File**: `engine/features/builder.py`

```python
# Add to Tier 2 (MTF) feature building
from engine.temporal.fib_time_clusters import calculate_fib_time_features

def build_tier2_features(df: pd.DataFrame, config: Dict) -> pd.DataFrame:
    # ... existing MTF features ...

    # Add Fibonacci time clusters
    pivot_points = detect_pivot_points(df, window=5)

    for idx in range(len(df)):
        fib_features = calculate_fib_time_features(
            df, df, pivot_points, idx, config.get('temporal', {})
        )

        # Write to feature store
        for col, value in fib_features.items():
            df.at[df.index[idx], col] = value

    return df
```

### 2. Archetype Logic (Temporal Fusion)
**File**: `engine/archetypes/logic_v2_adapter.py`

```python
# Read fib cluster features in archetype checks
def _check_A(self, context: RuntimeContext) -> tuple:
    # ... existing spring trap logic ...

    # ENHANCEMENT: Boost score if in Fib time cluster zone
    fib_cluster_score = self.g(context.row, 'fib_time_cluster_score', 0.0)
    is_cluster_zone = self.g(context.row, 'is_fib_time_cluster_zone', False)

    if is_cluster_zone:
        # 10% bonus for temporal confluence
        score *= 1.10
        meta['fib_time_boost'] = True
        meta['fib_cluster_score'] = fib_cluster_score

    return matched, score, meta
```

### 3. Temporal Fusion Layer (Soft Adjustments)
**File**: `engine/fusion/temporal.py` (NEW, see TEMPORAL_FUSION_SPEC.md)

```python
def adjust_fusion_temporal(
    fusion_score: float,
    row: pd.Series,
    config: Dict
) -> float:
    """Apply small temporal adjustments to fusion score."""

    fib_cluster_score = row.get('fib_time_cluster_score', 0.0)
    wyckoff_phase = row.get('wyckoff_phase_abc', 'neutral')

    # Rule 1: Boost if Fib cluster + accumulation/markup phase
    if fib_cluster_score > 0.7 and wyckoff_phase in ['C', 'D']:
        fusion_score *= 1.10  # 10% boost

    # Rule 2: Suppress if no temporal confluence in ranging phase
    elif fib_cluster_score < 0.2 and wyckoff_phase == 'B':
        fusion_score *= 0.95  # 5% penalty

    return fusion_score
```

---

## Validation Plan

### Phase 1: Historical Validation (2024 Data)
**Script**: `results/fib_time_cluster_validation.md`

**Test Cases**:
1. **March 2024 ATH ($73k BTC)**:
   - Check if `is_fib_time_cluster_zone` = `True` within ±3 days of peak
   - Expect `fib_cluster_score` > 0.7 from BC/UTAD alignment

2. **Major Pullbacks (May, August 2024)**:
   - Verify Spring-A events align with Fib(55) or Fib(89) from previous SC
   - Expect cluster scores > 0.6 at reversal bottoms

3. **Ranging Periods (June-July 2024)**:
   - Confirm low cluster scores (< 0.3) during chop
   - No false positives from random pivot noise

### Phase 2: Real-Time Logging
**Implementation**: Add telemetry to backtest runner

```python
# Log cluster detections
if row['is_fib_time_cluster_zone']:
    logger.info(f"[FIB TIME] Cluster zone detected: score={row['fib_time_cluster_score']:.3f}, "
                f"bars_since_sc={row['bars_since_sc']}, "
                f"bars_since_spring_a={row['bars_since_spring_a']}")
```

### Phase 3: A/B Testing
**Baseline**: Archetype fusion WITHOUT temporal adjustments
**Test**: Same archetypes WITH temporal fusion layer enabled

**Metrics**:
- Win rate change
- Average profit factor
- Sharpe ratio
- Trade count (ensure no over-filtering)

**Target**: +2-5% improvement in PF without significant trade reduction (<10%)

---

## Implementation Checklist

- [ ] Create `engine/temporal/fib_time_clusters.py` with `calculate_fib_time_features()`
- [ ] Add feature columns to `engine/features/registry.py`
- [ ] Integrate into `engine/features/builder.py` (Tier 2 pipeline)
- [ ] Update archetype logic to read `fib_time_cluster_score`
- [ ] Create temporal fusion layer (`engine/fusion/temporal.py`)
- [ ] Write unit tests (`tests/unit/test_fib_time_clusters.py`)
- [ ] Backfill feature store for 2022-2024 data
- [ ] Run validation script and document results
- [ ] Update `CHANGELOG.md` with Phase 2 completion

---

## Configuration Example

```json
{
  "temporal": {
    "fib_time_clusters": {
      "enabled": true,
      "fib_levels": [21, 34, 55, 89, 144, 233],
      "tolerance_bars": 2,
      "min_wyckoff_confidence": 0.65,
      "lookback_bars": 500,
      "pivot_window": 5,
      "cluster_threshold": 0.7
    }
  }
}
```

---

## References

- **Existing Implementation**: `bull_machine/strategy/temporal_fib_clusters.py` (v1.6.1 legacy)
- **Wyckoff Events**: `engine/wyckoff/events.py`, Feature Registry lines 154-254
- **Feature Store**: `engine/features/builder.py`
- **Archetype Logic**: `engine/archetypes/logic_v2_adapter.py`
