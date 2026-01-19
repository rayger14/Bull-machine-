# PTI (Psychological Trap Index) Specification

## Executive Summary

**Purpose**: Quantify emotional trap conditions (fakeouts, wicks, volume spikes, funding extremes) to detect retail herd trapping WITHOUT requiring sentiment data.

**Status**: LIVE (Implemented in `engine/psychology/pti.py`, needs feature store integration + archetype enhancements)

**Feature Store Integration**: PARTIAL (PTI calculation exists, needs runtime enrichment features)

**Key Innovation**: Observable price action → psychological state inference (no Twitter sentiment required!)

---

## Background

Markets punish emotional decisions. PTI detects when retail traders are trapped by:

1. **Failed Breakouts** → Late FOMO entries get stopped out
2. **Extreme Wicks** → Stop hunts above/below key levels
3. **Volume Climaxes** → Exhaustion buying/selling at tops/bottoms
4. **Funding Rate Extremes** → Overcrowded positioning (crypto-specific)
5. **RSI Divergence** → Price vs momentum disconnect (smart money fading)

**Smart Money Edge**: When retail is trapped (high PTI), the opposite move is likely. Fade the herd.

---

## Input Features

### 1. Core Price Action (from OHLCV)
| Feature | Source | Description |
|---------|--------|-------------|
| `upper_wick_ratio` | `(high - max(open, close)) / (high - low)` | Upper wick as % of total candle range |
| `lower_wick_ratio` | `(min(open, close) - low) / (high - low)` | Lower wick as % of total candle range |
| `volume_zscore` | Rolling z-score of volume | Volume spike detection (3-sigma = extreme) |
| `rsi_14` | RSI(14) | Overbought/oversold conditions |

### 2. Advanced Indicators (optional, boost accuracy)
| Feature | Source | Description | Coverage |
|---------|--------|-------------|----------|
| `funding_rate` | Binance/Bybit API | Perpetual futures funding rate | Crypto only, 100% (2022+) |
| `rsi_bearish_div` | Runtime enrichment | TRUE if price new high, RSI lower high | Needs backfill |
| `rsi_bullish_div` | Runtime enrichment | TRUE if price new low, RSI higher low | Needs backfill |
| `volume_fade_flag` | Runtime enrichment | TRUE if 3-bar volume declining sequence | Needs backfill |
| `ob_retest_flag` | Runtime enrichment | TRUE if price near order block resistance/support | Needs backfill |

### 3. Multi-Timeframe Context (for confluence)
| Feature | Source | Description |
|---------|--------|-------------|
| `tf4h_external_trend` | 4H timeframe | 'up' / 'down' / 'neutral' |
| `wyckoff_bc` | Wyckoff events | Buying Climax (trap at top) |
| `wyckoff_utad` | Wyckoff events | UTAD (upthrust trap) |
| `wyckoff_spring_a` | Wyckoff events | Spring (downthrust trap) |

---

## PTI Formula (v2.0 Enhanced)

### Base PTI Score (0-1)

```python
def compute_pti_score(row: pd.Series) -> float:
    """
    Calculate PTI score from observable price action.

    Returns:
        0.0-1.0 where:
        - 0.0-0.3: Calm (no traps detected)
        - 0.3-0.5: Agitated (minor trap signals)
        - 0.5-0.7: Emotional (significant traps)
        - 0.7-1.0: Panic/Euphoria (extreme traps)
    """
    components = []

    # COMPONENT 1: Wick Traps (0-1)
    # Large wicks = stop hunts / rejection candles
    upper_wick_ratio = row.get('upper_wick_ratio', 0.0)
    lower_wick_ratio = row.get('lower_wick_ratio', 0.0)

    # Average wick size (symmetric traps)
    wick_score = (upper_wick_ratio + lower_wick_ratio) / 2.0
    wick_score = min(wick_score, 1.0)  # Cap at 1.0

    components.append(wick_score)

    # COMPONENT 2: Volume Climax (0-1)
    # Unusual volume = emotional buying/selling
    volume_z = row.get('volume_zscore', 0.0)

    # Map z-score to 0-1 (3-sigma = max)
    vol_score = min(abs(volume_z) / 3.0, 1.0)

    components.append(vol_score)

    # COMPONENT 3: Funding Rate Extremes (0-1) [Crypto only]
    funding_rate = row.get('funding_rate', None)

    if funding_rate is not None:
        # Funding > +0.01% or < -0.01% = overcrowded positioning
        # Positive funding = longs pay shorts (long squeeze risk)
        # Negative funding = shorts pay longs (short squeeze risk)
        funding_score = min(abs(funding_rate) / 0.01, 1.0)  # 1% = extreme
        components.append(funding_score)
    else:
        # No funding data (stocks / older crypto data)
        # Use alternative: BB width squeeze as proxy for volatility trap
        bb_width = row.get('bb_width', 0.0)
        atr = row.get('atr_20', 1.0)
        bb_squeeze = max(0.0, 1.0 - (bb_width / (2.0 * atr)))  # Low BB width = high squeeze
        components.append(min(bb_squeeze, 1.0))

    # COMPONENT 4: RSI Extremes (0-1)
    # Overbought (>70) or oversold (<30) = emotional positioning
    rsi = row.get('rsi_14', 50.0)

    # Distance from neutral (50) normalized to 0-1
    rsi_extreme = max(abs(rsi - 50.0) - 20.0, 0.0) / 30.0  # >70 or <30 = extreme
    rsi_extreme = min(rsi_extreme, 1.0)

    components.append(rsi_extreme)

    # WEIGHTED AVERAGE
    weights = [0.30, 0.25, 0.25, 0.20]  # Wick, Volume, Funding/Squeeze, RSI
    pti_score = sum(c * w for c, w in zip(components, weights))

    return pti_score


def compute_pti_state(pti_score: float) -> str:
    """
    Classify PTI state for human readability.

    Args:
        pti_score: PTI score (0-1)

    Returns:
        State label: 'calm' | 'agitated' | 'emotional' | 'euphoric' | 'panic'
    """
    if pti_score < 0.3:
        return 'calm'
    elif pti_score < 0.5:
        return 'agitated'
    elif pti_score < 0.7:
        return 'emotional'
    else:
        # Distinguish euphoria (bullish extreme) from panic (bearish extreme)
        rsi = row.get('rsi_14', 50.0)
        if rsi > 70:
            return 'euphoric'  # Retail FOMO at top
        elif rsi < 30:
            return 'panic'  # Retail capitulation at bottom
        else:
            return 'emotional'  # Extreme but not clear direction
```

---

## Enhanced PTI Features (Runtime Enrichment)

### 1. RSI Divergence Detection

**Feature**: `rsi_bearish_div` (bool), `rsi_bullish_div` (bool)

**Logic**:
```python
def detect_rsi_divergence(df: pd.DataFrame, lookback: int = 14) -> Tuple[bool, bool]:
    """
    Detect RSI divergence over lookback period.

    Returns:
        (bearish_div, bullish_div)
    """
    # Bearish divergence: Price new high, RSI lower high
    price_new_high_idx = df['close'].iloc[-lookback:].idxmax()
    price_new_high = df['close'].iloc[-lookback:].max()

    # Find previous high
    prev_highs = df['close'].iloc[-lookback:-5]
    if len(prev_highs) > 0:
        prev_high_idx = prev_highs.idxmax()
        prev_high = prev_highs.max()

        # Compare RSI at both highs
        rsi_at_new_high = df.loc[price_new_high_idx, 'rsi_14']
        rsi_at_prev_high = df.loc[prev_high_idx, 'rsi_14']

        bearish_div = (price_new_high > prev_high * 1.01) and (rsi_at_new_high < rsi_at_prev_high - 5)
    else:
        bearish_div = False

    # Bullish divergence: Price new low, RSI higher low
    price_new_low_idx = df['close'].iloc[-lookback:].idxmin()
    price_new_low = df['close'].iloc[-lookback:].min()

    prev_lows = df['close'].iloc[-lookback:-5]
    if len(prev_lows) > 0:
        prev_low_idx = prev_lows.idxmin()
        prev_low = prev_lows.min()

        rsi_at_new_low = df.loc[price_new_low_idx, 'rsi_14']
        rsi_at_prev_low = df.loc[prev_low_idx, 'rsi_14']

        bullish_div = (price_new_low < prev_low * 0.99) and (rsi_at_new_low > rsi_at_prev_low + 5)
    else:
        bullish_div = False

    return bearish_div, bullish_div
```

**Integration**: Run vectorized during feature store build, cache in column.

### 2. Volume Fade Detection

**Feature**: `volume_fade_flag` (bool)

**Logic**:
```python
def detect_volume_fade(df: pd.DataFrame, window: int = 3) -> bool:
    """
    Detect volume fading over N bars (declining buying/selling pressure).

    Returns:
        True if volume declining for N consecutive bars
    """
    recent_volume = df['volume'].iloc[-window:]

    # Check if monotonically decreasing
    is_fading = all(recent_volume.iloc[i] < recent_volume.iloc[i-1]
                    for i in range(1, len(recent_volume)))

    return is_fading
```

**Integration**: Vectorized with rolling window, cached in feature store.

### 3. Order Block Retest Detection

**Feature**: `ob_retest_flag` (bool)

**Logic**:
```python
def detect_ob_retest(row: pd.Series, tolerance_pct: float = 0.02) -> bool:
    """
    Detect if price is retesting order block resistance/support.

    Args:
        row: Current bar
        tolerance_pct: Proximity threshold (2% default)

    Returns:
        True if price within tolerance of OB level
    """
    close = row['close']

    # Check bullish OB (support)
    ob_bull_top = row.get('tf1h_ob_bull_top', None)
    if ob_bull_top is not None and close >= ob_bull_top * (1 - tolerance_pct):
        return True

    # Check bearish OB (resistance)
    ob_bear_bottom = row.get('tf1h_ob_bear_bottom', None)
    if ob_bear_bottom is not None and close <= ob_bear_bottom * (1 + tolerance_pct):
        return True

    return False
```

**Integration**: Runtime calculation (cheap, no backfill needed).

---

## PTI Confluence Features

### 1. PTI-Wyckoff Confluence

**Feature**: `pti_confluence_with_wyckoff` (bool)

**Logic**:
```python
pti_confluence_with_wyckoff = (
    (pti_score > 0.6) AND
    (wyckoff_spring_a OR wyckoff_utad OR wyckoff_bc)
)
```

**Interpretation**:
- High PTI + Spring-A → Bullish trap reversal (BUY signal)
- High PTI + UTAD/BC → Bearish trap reversal (SELL/avoid longs)

### 2. PTI-Order Block Confluence

**Feature**: `pti_confluence_with_ob` (bool)

**Logic**:
```python
pti_confluence_with_ob = (
    (pti_score > 0.6) AND
    ob_retest_flag
)
```

**Interpretation**:
- High PTI + OB retest → Fakeout/reversal at key level (high probability)

---

## Output Features

| Feature Name | Type | Range | Description |
|-------------|------|-------|-------------|
| `pti_score` | float | 0.0-1.0 | Overall psychological trap intensity |
| `pti_state` | str | enum | 'calm' / 'agitated' / 'emotional' / 'euphoric' / 'panic' |
| `pti_confluence_with_wyckoff` | bool | - | PTI + Wyckoff trap event |
| `pti_confluence_with_ob` | bool | - | PTI + order block retest |
| `rsi_bearish_div` | bool | - | Bearish divergence detected |
| `rsi_bullish_div` | bool | - | Bullish divergence detected |
| `volume_fade_flag` | bool | - | Volume declining (3-bar sequence) |
| `ob_retest_flag` | bool | - | Price near order block |

---

## Integration with Archetypes

### Archetype A (Trap Reversal)
**Current Logic** (uses PTI trap type):
```python
pti_trap = self.g(context.row, "pti_trap_type", '')
if pti_trap not in ['spring', 'utad']:
    return False, 0.0, {"reason": "no_pti_trap"}
```

**Enhanced Logic** (uses PTI score + confluence):
```python
pti_score = self.g(context.row, "pti_score", 0.0)
pti_wyckoff_confluence = self.g(context.row, "pti_confluence_with_wyckoff", False)

# Gate 1: High PTI score OR Wyckoff confluence
if pti_score < 0.6 and not pti_wyckoff_confluence:
    return False, 0.0, {"reason": "pti_score_low"}

# Gate 2: RSI divergence bonus (if available)
rsi_bearish_div = self.g(context.row, "rsi_bearish_div", False)
if rsi_bearish_div:
    score *= 1.15  # 15% bonus for TRUE divergence (not just overbought)
```

### Archetype S2 (Failed Rally Rejection)
**Already uses runtime enrichment** (see `_check_S2_enhanced()` in logic_v2_adapter.py):
```python
# Enhanced S2 uses:
# - wick_upper_ratio (runtime calc)
# - volume_fade_flag (3-bar sequence)
# - rsi_bearish_div (true divergence)
# - ob_retest_flag (OB proximity)

if not ob_retest_flag:
    return False, 0.0, {"reason": "no_ob_retest"}

if wick_upper_ratio < 0.4:
    return False, 0.0, {"reason": "weak_wick"}

# Compute enhanced score
components = {
    "ob_retest": 1.0,
    "wick_rejection": min(wick_upper_ratio / 0.6, 1.0),
    "volume_fade": 1.0 if volume_fade_flag else 0.3,
    "rsi_divergence": 1.0 if rsi_bearish_div else 0.4
}

score = sum(components[k] * weights[k] for k in components)
```

### Global PTI Filter (Temporal Fusion Layer)
**File**: `engine/fusion/temporal.py`

```python
# Rule: High PTI + wrong direction = penalty
pti_score = row.get('pti_score', 0.0)
pti_state = row.get('pti_state', 'calm')
direction = 'long'  # From archetype

if pti_score > 0.7:
    # Check if trading INTO the trap (bad) or FADING the trap (good)
    rsi = row.get('rsi_14', 50.0)

    if direction == 'long' and rsi > 70:
        # Trying to buy at euphoric top (BAD)
        fusion_score *= 0.85  # 15% penalty
        logger.warning(f"[PTI FILTER] Euphoric trap (PTI={pti_score:.2f}, RSI={rsi:.0f}), penalizing long entry")

    elif direction == 'short' and rsi < 30:
        # Trying to short at panic bottom (BAD)
        fusion_score *= 0.85  # 15% penalty
        logger.warning(f"[PTI FILTER] Panic trap (PTI={pti_score:.2f}, RSI={rsi:.0f}), penalizing short entry")

    else:
        # Trading AGAINST the trap (GOOD - fade the herd)
        fusion_score *= 1.05  # 5% bonus
        logger.info(f"[PTI FILTER] Fading trap (PTI={pti_score:.2f}), boosting entry")
```

---

## Implementation Checklist

### Phase 1: Feature Store Integration (Priority 1)
- [x] PTI calculation exists in `engine/psychology/pti.py` (DONE)
- [ ] Add `pti_score` column to feature store (vectorized calculation)
- [ ] Add `pti_state` column (binned from score)
- [ ] Backfill 2022-2024 data with PTI features

### Phase 2: Runtime Enrichment (Priority 2)
- [ ] Implement `detect_rsi_divergence()` (vectorized, backfill)
- [ ] Implement `detect_volume_fade()` (vectorized, backfill)
- [ ] Implement `detect_ob_retest()` (runtime, no backfill needed)
- [ ] Add columns: `rsi_bearish_div`, `rsi_bullish_div`, `volume_fade_flag`, `ob_retest_flag`

### Phase 3: Confluence Features (Priority 3)
- [ ] Add `pti_confluence_with_wyckoff` (boolean logic)
- [ ] Add `pti_confluence_with_ob` (boolean logic)
- [ ] Test on 2024 data (expect higher precision at reversals)

### Phase 4: Archetype Integration (Priority 4)
- [ ] Update Archetype A to use `pti_score` instead of `pti_trap_type`
- [ ] Add RSI divergence bonus to Archetype A scoring
- [ ] Verify S2 enhanced logic uses runtime enrichment correctly
- [ ] Add PTI filter to temporal fusion layer

### Phase 5: Validation (Priority 5)
- [ ] Run backtest with PTI features (2022-2024)
- [ ] Document trap detection accuracy in `results/pti_validation.md`
- [ ] A/B test: With vs without RSI divergence enrichment

---

## Validation Plan

### Test Case 1: March 2024 ATH ($73k BTC)
**Expected**:
- `pti_score` > 0.7 at peak (euphoric)
- `pti_state` = 'euphoric'
- `rsi_bearish_div` = TRUE (price new high, RSI divergence)
- `wyckoff_bc` = TRUE (buying climax)
- `pti_confluence_with_wyckoff` = TRUE

**Action**: Archetype A should AVOID longs, consider shorts

### Test Case 2: Bear Market Bottom (June 2022, ~$17k)
**Expected**:
- `pti_score` > 0.7 at bottom (panic)
- `pti_state` = 'panic'
- `rsi_bullish_div` = TRUE (price new low, RSI higher low)
- `wyckoff_sc` = TRUE (selling climax)

**Action**: Archetype A should BOOST longs (spring reversal)

### Test Case 3: Ranging Chop (Summer 2023)
**Expected**:
- `pti_score` < 0.4 (calm)
- `pti_state` = 'calm' or 'agitated'
- No divergences detected
- No Wyckoff confluence

**Action**: PTI filter has minimal impact (neutral)

---

## Configuration Example

```json
{
  "pti": {
    "enabled": true,
    "timeframe": "4H",
    "lookback_divergence": 14,
    "lookback_volume_fade": 3,

    "weights": {
      "wick_trap": 0.30,
      "volume_climax": 0.25,
      "funding_extreme": 0.25,
      "rsi_extreme": 0.20
    },

    "thresholds": {
      "calm": 0.3,
      "agitated": 0.5,
      "emotional": 0.7
    },

    "confluence": {
      "pti_min": 0.6,
      "wyckoff_events": ["spring_a", "utad", "bc", "sc"],
      "ob_tolerance_pct": 0.02
    },

    "fusion_adjustments": {
      "trap_penalty": 0.85,
      "fade_bonus": 1.05,
      "divergence_bonus": 1.15
    }
  }
}
```

---

## References

- **Existing Implementation**: `engine/psychology/pti.py`
- **Archetype A (Trap Reversal)**: `engine/archetypes/logic_v2_adapter.py` lines 731-791
- **Archetype S2 (Failed Rally)**: `engine/archetypes/logic_v2_adapter.py` lines 1207-1557
- **Feature Registry**: `engine/features/registry.py` lines 248-254 (Wyckoff-PTI integration)
- **Temporal Fusion Layer**: `docs/TEMPORAL_FUSION_SPEC.md`
