# Complete Knowledge Architecture - Bull Machine v2.0

**Branch**: `feature/ml-meta-optimizer`
**Date**: 2025-10-16
**Status**: Unifying All Wisdom Layers

---

## 🎯 Mission

Complete the Bull Machine with ALL accumulated wisdom:
- Wyckoff/Bojan SMC mastery
- Moneytaur balance→imbalance philosophy
- ZeroIKA funding/OI traps
- Astronomer macro echoes
- Trader Dune HTF wisdom
- Internal structure awareness
- Psychology trap detection
- Temporal confluence

**Goal**: One unified system where PyTorch learns from complete market understanding.

---

## 📚 Knowledge Layers (Complete Stack)

### Layer 1: Market Structure (SMC/Wyckoff Foundation)

#### 1.1 Internal vs External Structure ⚠️ **NEW**
**Purpose**: Distinguish micro reversals from macro trend continuation

**Implementation**:
```python
# engine/structure/internal_external.py

@dataclass
class StructureState:
    internal_phase: str  # 'accumulation', 'distribution', 'transition'
    external_trend: str  # 'bullish', 'bearish', 'range'
    alignment: bool      # True if internal matches external
    conflict_score: float  # 0-1, higher = more conflict

def detect_structure_state(df_1h, df_4h, df_1d) -> StructureState:
    """
    Detect nested structure states.

    Logic:
    - External (1D): Wyckoff phase + trend SMA
    - Internal (1H/4H): Local BOS/CHOCH patterns
    - Conflict: When 1H shows accumulation but 1D in distribution

    Returns:
        StructureState with alignment flag
    """
    # 1D external structure (macro)
    wyckoff_1d = detect_wyckoff_phase(df_1d)
    trend_1d = get_trend(df_1d, period=50)

    # 1H/4H internal structure (micro)
    bos_4h = detect_bos_choch(df_4h)
    ob_1h = detect_order_blocks(df_1h)

    # Alignment check
    alignment = (wyckoff_1d['phase'] in ['accumulation', 'markup'] and
                 bos_4h['direction'] == 'bullish') or \
                (wyckoff_1d['phase'] in ['distribution', 'markdown'] and
                 bos_4h['direction'] == 'bearish')

    # Conflict score (for early reversal detection)
    conflict = abs(wyckoff_score_1d - bos_score_1h) / 2.0

    return StructureState(
        internal_phase=bos_4h['direction'],
        external_trend=trend_1d,
        alignment=alignment,
        conflict_score=conflict
    )
```

**Feature Store Columns**:
- `internal_phase`: str
- `external_trend`: str
- `structure_alignment`: bool
- `conflict_score`: float

**Fusion Impact**:
- If `alignment=True`: No penalty
- If `alignment=False` and `conflict_score > 0.6`: Threshold +0.05 (require higher confluence)
- Use conflict for early reversal detection

---

#### 1.2 BOMS (Break of Market Structure) ⚠️ **NEW**
**Purpose**: HTF volume-confirmed structural breaks (stronger than BOS)

**Implementation**:
```python
# engine/structure/boms_detector.py

def detect_boms(df, timeframe='4H') -> Dict:
    """
    Detect Break of Market Structure (BOMS).

    Criteria (more stringent than BOS):
    1. Close beyond prior swing high/low
    2. Volume > 1.5x mean (institutional participation)
    3. FVG left behind (imbalance trailing)
    4. No immediate reversal (confirmed)

    Returns:
        {
            'boms_detected': bool,
            'direction': 'bullish' | 'bearish',
            'volume_surge': float,
            'fvg_present': bool,
            'confirmation_bars': int
        }
    """
    swings = find_swing_points(df, window=20)

    for i in range(len(df) - 5):
        close = df['close'].iloc[i]
        volume = df['volume'].iloc[i]
        vol_mean = df['volume'].rolling(20).mean().iloc[i]

        # Check if broke swing high with volume
        if close > swings['high'].iloc[i]:
            if volume > vol_mean * 1.5:
                # Check for FVG trail
                fvg = detect_fvg(df.iloc[i-3:i+1])
                if fvg['bullish_fvg']:
                    # Confirm no immediate reversal
                    if not reversed_in_n_bars(df.iloc[i:i+3], direction='bullish'):
                        return {
                            'boms_detected': True,
                            'direction': 'bullish',
                            'volume_surge': volume / vol_mean,
                            'fvg_present': True,
                            'confirmation_bars': 3
                        }

    return {'boms_detected': False}
```

**Feature Store Columns**:
- `boms_detected`: bool
- `boms_direction`: str
- `boms_volume_surge`: float
- `boms_confirmation`: int

**Fusion Impact**:
- BOMS on 4H/1D → Fusion +0.10 (strong HTF confirmation)
- Require BOMS for entries >2R risk

---

#### 1.3 1-2-3 Squiggle Pattern ⚠️ **NEW**
**Purpose**: Classic SMC entry pattern (BOS → Retest → Continuation)

**Implementation**:
```python
# engine/patterns/squiggle_123.py

@dataclass
class SquigglePattern:
    stage: int  # 1, 2, or 3
    pattern_id: str
    entry_window: bool
    confidence: float

def detect_squiggle_123(df) -> SquigglePattern:
    """
    Detect 1-2-3 Squiggle Pattern.

    Stages:
    1. BOS (Break of Structure) - Impulse move
    2. Retest (OB/FVG mitigation) - Entry window ✅
    3. Continuation (Follow-through) - Re-entry or exit

    Returns:
        SquigglePattern with stage and entry signal
    """
    # Stage 1: Detect BOS
    bos = detect_bos_choch(df)
    if not bos['bos_detected']:
        return SquigglePattern(stage=0, pattern_id='', entry_window=False, confidence=0.0)

    # Stage 2: Check for retest (pullback to OB/FVG)
    obs = get_active_order_blocks(df)
    fvgs = get_active_fvgs(df)
    current_price = df['close'].iloc[-1]

    for ob in obs:
        if ob['zone_low'] <= current_price <= ob['zone_high']:
            # In retest zone - ENTRY WINDOW ✅
            return SquigglePattern(
                stage=2,
                pattern_id=f"123_{bos['direction']}_{ob['id']}",
                entry_window=True,
                confidence=ob['strength'] * bos['displacement']
            )

    # Stage 3: Check for continuation
    if bos['direction'] == 'bullish' and current_price > bos['break_level'] * 1.02:
        return SquigglePattern(stage=3, pattern_id='continuation', entry_window=False, confidence=0.8)

    return SquigglePattern(stage=1, pattern_id='bos_only', entry_window=False, confidence=0.5)
```

**Feature Store Columns**:
- `squiggle_stage`: int (0-3)
- `squiggle_entry_window`: bool
- `squiggle_confidence`: float
- `squiggle_pattern_id`: str

**Fusion Impact**:
- Stage 2 (retest) + `entry_window=True`: Fusion +0.05
- Stage 3 (continuation): Filter re-entries (only if new setup)

---

#### 1.4 Range Outcomes Classification ⚠️ **NEW**
**Purpose**: Predict range breakout vs fake-out probability

**Implementation**:
```python
# engine/structure/range_classifier.py

def classify_range_outcome(df, range_bounds: tuple) -> Dict:
    """
    Classify range outcome: breakout, retest, fake-out, or rejection.

    Criteria:
    - Breakout: Close beyond range + volume + FVG trail
    - Break & Retest: Break → pullback to range edge → continue
    - Fake-out: Break without volume → immediate reversal
    - Rejection: Wick beyond range → close inside

    Returns:
        {
            'outcome': str,
            'probability': float,
            'volume_confirmation': bool,
            'displacement': float
        }
    """
    low, high = range_bounds
    close = df['close'].iloc[-1]
    volume = df['volume'].iloc[-1]
    vol_mean = df['volume'].rolling(20).mean().iloc[-1]

    # Check if price broke range
    if close > high:
        # Potential bullish breakout
        if volume > vol_mean * 1.3:
            # Volume confirmed - likely real breakout
            fvg = detect_fvg(df.tail(5))
            if fvg['bullish_fvg']:
                return {
                    'outcome': 'breakout',
                    'probability': 0.8,
                    'volume_confirmation': True,
                    'displacement': (close - high) / high
                }
        else:
            # Weak volume - potential fake-out
            return {
                'outcome': 'fakeout',
                'probability': 0.7,
                'volume_confirmation': False,
                'displacement': (close - high) / high
            }

    # Check for rejection (wick beyond range, close inside)
    if df['high'].iloc[-1] > high and close < high:
        return {
            'outcome': 'rejection',
            'probability': 0.75,
            'volume_confirmation': volume > vol_mean,
            'displacement': 0.0
        }

    return {'outcome': 'inside_range', 'probability': 0.5}
```

**Feature Store Columns**:
- `range_outcome`: str
- `range_outcome_prob`: float
- `range_volume_conf`: bool
- `range_displacement`: float

**Fusion Impact**:
- `fakeout` with prob > 0.6: Fusion -0.05 (reduce confidence)
- `breakout` with volume: Fusion +0.05

---

### Layer 2: Psychology & Trap Detection

#### 2.1 PTI (Psychology Trap Index) ⚠️ **NEW**
**Purpose**: Quantify herd-trap risk without external sentiment

**Implementation**:
```python
# engine/psychology/trap_index.py

def calculate_pti(df, lookback=20) -> Dict:
    """
    Psychology Trap Index - measures late-belief traps.

    Inputs:
    - grind_slope: Gradual price rise (low vol)
    - pullback_depth: Shallow pullbacks (optimism)
    - R_distribution: Tight reward distribution (consensus)
    - wick_ratio: Climactic wicks (exhaustion)
    - session_stretch: Extended hours activity (FOMO)

    Output:
    - trap_score: -1 (bearish trap) to +1 (bullish trap)
    - tags: ['late_belief', 'grand_finale', 'exhaustion']
    """
    # 1. Grind slope (gradual rise = potential bull trap)
    prices = df['close'].tail(lookback)
    slope = np.polyfit(range(len(prices)), prices, 1)[0]
    atr = df['close'].rolling(14).std().iloc[-1]
    grind_score = np.clip(slope / atr, -1, 1)  # Normalize by volatility

    # 2. Pullback depth (shallow = optimism)
    pullbacks = []
    for i in range(1, lookback):
        if df['close'].iloc[-i] < df['close'].iloc[-i-1]:
            depth = abs(df['close'].iloc[-i] - df['high'].iloc[-i-1]) / df['high'].iloc[-i-1]
            pullbacks.append(depth)
    avg_pullback = np.mean(pullbacks) if pullbacks else 0.02
    pullback_score = 1.0 - np.clip(avg_pullback / 0.05, 0, 1)  # Shallow = high score

    # 3. R-distribution width (tight = consensus)
    returns = df['close'].pct_change().tail(lookback)
    r_std = returns.std()
    r_width_score = 1.0 - np.clip(r_std / 0.02, 0, 1)  # Tight = high score

    # 4. Wick ratio (climactic = exhaustion)
    wick_ratios = []
    for i in range(lookback):
        body = abs(df['close'].iloc[-i] - df['open'].iloc[-i])
        total = df['high'].iloc[-i] - df['low'].iloc[-i]
        wick_ratios.append((total - body) / total if total > 0 else 0)
    avg_wick = np.mean(wick_ratios)
    wick_score = np.clip(avg_wick / 0.5, 0, 1)  # High wick = exhaustion

    # Composite trap score
    trap_score = (grind_score * 0.3 +
                  pullback_score * 0.25 +
                  r_width_score * 0.25 +
                  wick_score * 0.20)

    # Tags
    tags = []
    if trap_score > 0.6:
        tags.append('late_belief')
    if wick_score > 0.7 and trap_score > 0.5:
        tags.append('grand_finale')
    if trap_score > 0.7:
        tags.append('exhaustion')

    return {
        'trap_score': float(np.clip(trap_score, -1, 1)),
        'trap_tags': tags,
        'grind_component': float(grind_score),
        'pullback_component': float(pullback_score),
        'wick_component': float(wick_score)
    }
```

**Feature Store Columns**:
- `pti_trap_score`: float (-1 to 1)
- `pti_tags`: str (JSON list)
- `pti_grind`: float
- `pti_pullback`: float
- `pti_wick`: float

**Fusion Impact**:
- PTI > 0.6 (bullish trap): Threshold +0.05, Size ×0.75
- PTI < -0.6 (bearish trap): Threshold +0.05 (for shorts)
- In trade + PTI spike + climactic wick: Take partial profit

---

#### 2.2 Fake-out Intensity ⚠️ **NEW**
**Purpose**: Quantify fake-out probability for better filtering

**Implementation**:
```python
# engine/psychology/fakeout_detector.py

def calculate_fakeout_intensity(df) -> float:
    """
    Fake-out intensity = wick_ratio × spread / ATR

    High intensity = likely fake-out (enter opposite or wait)
    Low intensity = likely real move
    """
    last_candle = df.iloc[-1]

    # Wick ratio
    body = abs(last_candle['close'] - last_candle['open'])
    total_range = last_candle['high'] - last_candle['low']
    wick_ratio = (total_range - body) / total_range if total_range > 0 else 0

    # Spread
    spread = total_range / last_candle['close']

    # ATR
    atr = df['close'].rolling(14).std().iloc[-1] / df['close'].iloc[-1]

    # Intensity
    intensity = (wick_ratio * spread) / (atr + 1e-9)

    return float(np.clip(intensity, 0, 2))
```

**Feature Store Column**: `fakeout_intensity`

**Fusion Impact**:
- Intensity > 1.2: Reduce entry confidence (-0.05) or delay confirmation

---

### Layer 3: Volume & Liquidity DNA

#### 3.1 FRVP (Fixed Range Volume Profile) ⚠️ **NEW**
**Purpose**: Identify POC, HVN, LVN for entries/exits

**Implementation**:
```python
# engine/volume/frvp_engine.py

@dataclass
class FRVPAnalysis:
    poc: float  # Point of Control (highest volume)
    hvn_zones: List[tuple]  # High Volume Nodes
    lvn_zones: List[tuple]  # Low Volume Nodes
    absorption_flag: bool
    current_zone: str  # 'poc', 'hvn', 'lvn', 'neutral'

def calculate_frvp(df, lookback=100) -> FRVPAnalysis:
    """
    Calculate Fixed Range Volume Profile.

    Logic:
    1. Split price range into bins (e.g., 50 bins)
    2. Sum volume per bin
    3. Find POC (max volume bin)
    4. Find HVN (bins > 80th percentile)
    5. Find LVN (bins < 20th percentile)
    6. Detect absorption (large wick into HVN + no close through)
    """
    price_range = df['high'].tail(lookback).max() - df['low'].tail(lookback).min()
    bin_size = price_range / 50

    # Create bins
    bins = {}
    for i in range(50):
        bin_low = df['low'].tail(lookback).min() + i * bin_size
        bin_high = bin_low + bin_size
        bins[i] = {'range': (bin_low, bin_high), 'volume': 0}

    # Accumulate volume
    for idx, row in df.tail(lookback).iterrows():
        for i, bin_data in bins.items():
            if bin_data['range'][0] <= row['close'] <= bin_data['range'][1]:
                bins[i]['volume'] += row['volume']

    # Find POC
    poc_bin = max(bins.items(), key=lambda x: x[1]['volume'])
    poc = (poc_bin[1]['range'][0] + poc_bin[1]['range'][1]) / 2

    # Find HVN/LVN
    volumes = [b['volume'] for b in bins.values()]
    hvn_threshold = np.percentile(volumes, 80)
    lvn_threshold = np.percentile(volumes, 20)

    hvn_zones = [b['range'] for b in bins.values() if b['volume'] > hvn_threshold]
    lvn_zones = [b['range'] for b in bins.values() if b['volume'] < lvn_threshold]

    # Check absorption
    current_price = df['close'].iloc[-1]
    last_wick = df['high'].iloc[-1] - max(df['open'].iloc[-1], df['close'].iloc[-1])
    absorption = False
    for hvn in hvn_zones:
        if hvn[0] <= current_price + last_wick <= hvn[1] and df['close'].iloc[-1] < hvn[1]:
            absorption = True

    # Determine current zone
    current_zone = 'neutral'
    if abs(current_price - poc) < bin_size:
        current_zone = 'poc'
    elif any(z[0] <= current_price <= z[1] for z in hvn_zones):
        current_zone = 'hvn'
    elif any(z[0] <= current_price <= z[1] for z in lvn_zones):
        current_zone = 'lvn'

    return FRVPAnalysis(
        poc=poc,
        hvn_zones=hvn_zones,
        lvn_zones=lvn_zones,
        absorption_flag=absorption,
        current_zone=current_zone
    )
```

**Feature Store Columns**:
- `frvp_poc`: float
- `frvp_hvn_count`: int
- `frvp_lvn_count`: int
- `frvp_absorption`: bool
- `frvp_current_zone`: str

**Fusion Impact**:
- Near POC reclaim: Fusion +0.05
- Absorption at HVN: Exit partial (25-50%)
- In LVN gap: Caution (fast moves expected)

---

### Layer 4: Temporal & Cycles

#### 4.1 Temporal Cycles (Gann Time) ⚠️ **NEW**
**Purpose**: Time-based confluence using Fibonacci periods

**Implementation**:
```python
# engine/temporal/gann_cycles.py

def calculate_time_clusters(df, pivots: List[datetime]) -> Dict:
    """
    Calculate Gann time clusters.

    Fibonacci periods: 21, 34, 55, 89, 144 bars

    Logic:
    1. From last 2-3 pivots, project Fib periods forward
    2. Find cluster windows (±TF-scaled tolerance)
    3. Score confluence (more overlaps = higher score)

    Returns:
        {
            'time_cluster_score': float (0-1),
            'cluster_window': tuple (start, end),
            'contributing_periods': list
        }
    """
    fib_periods = [21, 34, 55, 89, 144]
    current_idx = len(df) - 1

    # Project from pivots
    projections = []
    for pivot_time in pivots[-3:]:
        pivot_idx = df.index.get_loc(pivot_time)
        for period in fib_periods:
            target_idx = pivot_idx + period
            projections.append(target_idx)

    # Find clusters (within ±3 bars)
    tolerance = 3
    clusters = []
    for proj in projections:
        cluster = [p for p in projections if abs(p - proj) <= tolerance]
        if len(cluster) >= 2 and cluster not in clusters:
            clusters.append(cluster)

    # Score current time
    time_score = 0.0
    active_cluster = None
    for cluster in clusters:
        if any(abs(proj - current_idx) <= tolerance for proj in cluster):
            time_score = len(cluster) / 5.0  # Normalize to 0-1
            active_cluster = cluster
            break

    return {
        'time_cluster_score': float(np.clip(time_score, 0, 1)),
        'cluster_window': (current_idx - tolerance, current_idx + tolerance) if active_cluster else None,
        'contributing_periods': [p - current_idx for p in active_cluster] if active_cluster else []
    }
```

**Feature Store Columns**:
- `time_cluster_score`: float (0-1)
- `time_cluster_active`: bool
- `time_cluster_periods`: str (JSON list)

**Fusion Impact**:
- Time cluster + structure alignment: Fusion +0.05 (capped)
- Time cluster alone: No impact (plus-one only)

---

#### 4.2 Price Clusters (Fib Confluence) ⚠️ **ENHANCE EXISTING**
**Purpose**: Multiple Fib levels converging at same price

**Implementation**:
```python
# engine/fibonacci/price_clusters.py

def detect_price_clusters(df, swings: List[tuple]) -> Dict:
    """
    Detect Fib price clusters from multiple swing ranges.

    Logic:
    1. Calculate Fib retracements from last 3 swings
    2. Find price zones where 2+ fibs overlap (±0.5%)
    3. Score by number of fibs + TF diversity

    Returns:
        {
            'cluster_zones': List[tuple],
            'cluster_scores': List[float],
            'current_near_cluster': bool
        }
    """
    fib_levels = [0.236, 0.382, 0.5, 0.618, 0.786, 1.272, 1.618]
    all_fibs = []

    # Calculate fibs from each swing
    for swing_low, swing_high in swings[-3:]:
        range_size = swing_high - swing_low
        for level in fib_levels:
            fib_price = swing_low + range_size * level
            all_fibs.append(fib_price)

    # Find clusters (±0.5% tolerance)
    tolerance = 0.005
    clusters = []
    for fib in all_fibs:
        cluster_fibs = [f for f in all_fibs if abs(f - fib) / fib < tolerance]
        if len(cluster_fibs) >= 2:
            zone = (min(cluster_fibs), max(cluster_fibs))
            score = len(cluster_fibs) / 6.0  # Normalize
            if zone not in [c[0] for c in clusters]:
                clusters.append((zone, score))

    # Check if current price near cluster
    current_price = df['close'].iloc[-1]
    near_cluster = any(zone[0] <= current_price <= zone[1] for zone, _ in clusters)

    return {
        'cluster_zones': [z for z, _ in clusters],
        'cluster_scores': [s for _, s in clusters],
        'current_near_cluster': near_cluster
    }
```

**Feature Store Columns**:
- `price_cluster_count`: int
- `price_cluster_score`: float
- `near_price_cluster`: bool

**Fusion Impact**:
- Near cluster + structure: Fusion +0.05
- Target magnet for exits

---

### Layer 5: Macro Echo Rules ⚠️ **NEW**

#### 5.1 DXY/Oil/Yield Correlations
**Purpose**: Macro intermarket relationships

**Implementation**:
```python
# engine/macro/correlation_matrix.py

def calculate_macro_echo(macro_data: Dict) -> Dict:
    """
    Macro correlation echo rules.

    Correlations:
    - DXY ↑ → BTC ↓ (inverse)
    - Oil ↑ → Inflation ↑ → Risk ↓ (negative)
    - Gold ↑ + DXY ↓ → Risk-on (positive)
    - Yield ↑ + VIX ↑ → Risk-off (veto)

    Returns:
        {
            'macro_score': float (-1 to 1),
            'veto': bool,
            'regime': 'risk_on' | 'risk_off' | 'mixed' | 'crisis',
            'reasons': List[str]
        }
    """
    dxy = macro_data.get('DXY', 100)
    oil = macro_data.get('OIL', 70)
    gold = macro_data.get('GOLD', 2000)
    vix = macro_data.get('VIX', 15)
    yields = macro_data.get('YIELD_10Y', 4.0)

    score = 0.0
    reasons = []
    veto = False

    # DXY inverse correlation
    if dxy > 105:
        score -= 0.15
        reasons.append('DXY_strong_headwind')
    elif dxy < 95:
        score += 0.10
        reasons.append('DXY_weak_tailwind')

    # Oil inflation impact
    if oil > 90:
        score -= 0.10
        reasons.append('Oil_inflation_risk')

    # Gold + DXY risk-on signal
    if gold > 2100 and dxy < 100:
        score += 0.15
        reasons.append('Gold_DXY_risk_on')

    # Yield + VIX veto
    if yields > 4.5 and vix > 30:
        veto = True
        score = -1.0
        reasons.append('Yield_VIX_crisis')

    # DXY + Oil stagflation
    if dxy > 105 and oil > 85:
        veto = True
        reasons.append('Stagflation_risk')

    # Determine regime
    if veto:
        regime = 'crisis'
    elif score > 0.1:
        regime = 'risk_on'
    elif score < -0.1:
        regime = 'risk_off'
    else:
        regime = 'mixed'

    return {
        'macro_score': float(np.clip(score, -1, 1)),
        'veto': veto,
        'regime': regime,
        'reasons': reasons
    }
```

**Feature Store Columns**:
- `macro_echo_score`: float
- `macro_echo_veto`: bool
- `macro_regime`: str
- `macro_reasons`: str (JSON list)

**Fusion Impact**:
- Macro veto: Hard block (no trades)
- Risk-on (score > 0.1): Threshold -0.02, Risk ×1.1
- Risk-off (score < -0.1): Threshold +0.03, Risk ×0.8

---

### Layer 6: Exit System Enhancement ⚠️ **NEW**

#### 6.1 Multi-Modal Exit Logic
**Purpose**: Context-aware exits (not just R-ladder)

**Implementation**:
```python
# engine/execution/smart_exits.py

@dataclass
class ExitPlan:
    tp_levels: List[float]  # Price levels
    tp_sizes: List[float]   # % to close at each
    trail_atr_mult: float   # ATR trail multiplier
    time_stop_bars: int     # Max hold time
    structural_stop: float  # Opposite BOS level
    liquidity_target: float # Next pool level

def create_exit_plan(entry_price: float, side: str, df: pd.DataFrame,
                     frvp: FRVPAnalysis, ob_zones: List, fib_clusters: List) -> ExitPlan:
    """
    Create multi-modal exit plan.

    Exit modes:
    1. R-Ladder: 1R (50%), 2R (30%), 3R (20%)
    2. Liquidity: Next pool / fib extension
    3. Structural: Opposite BOS/CHOCH
    4. Absorption: HVN + climactic wick
    5. Time: Stall detected (EFE later)

    Returns:
        Complete exit plan with all levels
    """
    atr = df['close'].rolling(14).std().iloc[-1]

    # 1. R-Ladder (baseline)
    if side == 'long':
        tp1 = entry_price + 1.0 * atr  # 1R
        tp2 = entry_price + 2.0 * atr  # 2R
        tp3 = entry_price + 3.0 * atr  # 3R
    else:
        tp1 = entry_price - 1.0 * atr
        tp2 = entry_price - 2.0 * atr
        tp3 = entry_price - 3.0 * atr

    # 2. Find next liquidity pool
    liquidity_target = find_next_pool(df, entry_price, side)

    # 3. Fib extension targets
    if fib_clusters:
        nearest_cluster = min(fib_clusters, key=lambda x: abs(x - entry_price))
        if (side == 'long' and nearest_cluster > entry_price) or \
           (side == 'short' and nearest_cluster < entry_price):
            # Use cluster as target
            tp2 = nearest_cluster

    # 4. Structural stop (opposite BOS)
    structural_stop = find_opposite_bos_level(df, side)

    # 5. Time stop (based on typical hold duration)
    time_stop = 18  # 18 bars default (can be ML-optimized)

    return ExitPlan(
        tp_levels=[tp1, tp2, tp3],
        tp_sizes=[0.5, 0.3, 0.2],  # 50% @ TP1, 30% @ TP2, 20% @ TP3
        trail_atr_mult=1.5,
        time_stop_bars=time_stop,
        structural_stop=structural_stop,
        liquidity_target=liquidity_target
    )
```

**Feature Store Columns** (per trade):
- `exit_plan`: str (JSON)
- `exit_mode`: str  # 'tp1', 'tp2', 'structural', 'absorption', 'time'
- `exit_price`: float
- `exit_r_multiple`: float

---

### Layer 7: Learning Loop (Nightly Meta-Analysis) ⚠️ **NEW**

#### 7.1 Self-Optimization Framework
**Purpose**: Propose config patches based on recent performance

**Implementation**:
```python
# engine/learning/meta_optimizer.py

def nightly_meta_analysis(trade_log: pd.DataFrame, config: Dict) -> Dict:
    """
    Analyze recent trades and propose bounded config adjustments.

    Guards:
    - Only adjust weights/thresholds (never hard rules)
    - Require OOS confirmation (train on 80%, test on 20%)
    - Bound all changes (±0.10 max)

    Returns:
        {
            'proposed_patches': Dict,
            'confidence': float,
            'ablation_results': Dict,
            'oos_sharpe': float
        }
    """
    # Split data
    train_size = int(len(trade_log) * 0.8)
    train = trade_log.iloc[:train_size]
    test = trade_log.iloc[train_size:]

    # Current performance
    baseline_sharpe = calculate_sharpe(train)

    # Propose adjustments using Optuna
    def objective(trial):
        # Bound parameter search
        wyckoff_weight = trial.suggest_float('wyckoff_weight',
                                              max(0.1, config['wyckoff_weight'] - 0.10),
                                              min(0.5, config['wyckoff_weight'] + 0.10))
        threshold = trial.suggest_float('threshold',
                                        config['threshold'] - 0.05,
                                        config['threshold'] + 0.05)

        # Simulate with new params
        sim_trades = simulate_with_params(train, {
            'wyckoff_weight': wyckoff_weight,
            'threshold': threshold
        })
        return calculate_sharpe(sim_trades)

    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=50)

    # Get best params
    best_params = study.best_params

    # Validate on OOS
    oos_trades = simulate_with_params(test, best_params)
    oos_sharpe = calculate_sharpe(oos_trades)

    # Only accept if OOS Sharpe > baseline
    if oos_sharpe > baseline_sharpe * 1.05:
        return {
            'proposed_patches': best_params,
            'confidence': oos_sharpe / baseline_sharpe,
            'ablation_results': study.trials_dataframe(),
            'oos_sharpe': oos_sharpe,
            'accept': True
        }
    else:
        return {
            'proposed_patches': {},
            'confidence': 0.0,
            'accept': False,
            'reason': 'OOS validation failed'
        }
```

**Execution**: Cron job runs nightly, creates `config_patch.json` for review

---

## 📊 Complete Feature Store Schema

**New columns to add** (40+ total features):

```python
# Structure
'internal_phase', 'external_trend', 'structure_alignment', 'conflict_score',
'boms_detected', 'boms_direction', 'boms_volume_surge', 'boms_confirmation',
'squiggle_stage', 'squiggle_entry_window', 'squiggle_confidence', 'squiggle_pattern_id',
'range_outcome', 'range_outcome_prob', 'range_volume_conf', 'range_displacement',

# Psychology
'pti_trap_score', 'pti_tags', 'pti_grind', 'pti_pullback', 'pti_wick',
'fakeout_intensity',

# Volume/Liquidity
'frvp_poc', 'frvp_hvn_count', 'frvp_lvn_count', 'frvp_absorption', 'frvp_current_zone',

# Temporal
'time_cluster_score', 'time_cluster_active', 'time_cluster_periods',
'price_cluster_count', 'price_cluster_score', 'near_price_cluster',

# Macro
'macro_echo_score', 'macro_echo_veto', 'macro_regime', 'macro_reasons',

# Exits
'exit_plan', 'exit_mode', 'exit_price', 'exit_r_multiple'
```

---

## 🧪 Scenario Tests (15 Critical Tests)

```python
# tests/scenarios/test_complete_wisdom.py

def test_internal_external_conflict():
    """1D distribution but 1H accumulation → conflict_score high → threshold +0.05"""

def test_boms_confirmation():
    """4H BOMS with volume → fusion +0.10"""

def test_squiggle_entry_window():
    """Stage 2 retest of OB → entry_window=True → fusion +0.05"""

def test_range_fakeout():
    """Weak volume breakout → fakeout prob > 0.6 → fusion -0.05"""

def test_pti_late_belief():
    """PTI > 0.6 + climactic wick → threshold +0.05, size ×0.75"""

def test_frvp_absorption():
    """Wick into HVN + no close → absorption=True → take partial"""

def test_time_cluster_plus_structure():
    """Time cluster + structure align → fusion +0.05"""

def test_dxy_oil_stagflation():
    """DXY > 105 + Oil > 85 → macro veto=True"""

def test_yield_vix_crisis():
    """Yield > 4.5 + VIX > 30 → crisis regime → pause"""

def test_structural_exit():
    """Opposite BOS → exit at structural_stop"""

def test_liquidity_exit():
    """Price hits next pool → partial exit"""

def test_oos_patch_validation():
    """Proposed patch must improve OOS Sharpe by 5%"""

def test_fib_price_cluster():
    """2+ fibs overlap → cluster detected → target magnet"""

def test_smt_divergence_persistence():
    """SMT divergence ≥ 3 bars → signal valid"""

def test_htf_wick_magnet():
    """Unfinished 1D wick → magnet_strength increases → partial exit"""
```

---

## 🚀 Implementation Priority

### Phase 1 (Week 1): Core Structure
1. ✅ Internal/External structure tracking
2. ✅ BOMS detection
3. ✅ 1-2-3 Squiggle pattern
4. ✅ Range outcome classification
5. ✅ Update feature store schema

### Phase 2 (Week 2): Psychology & Volume
6. ✅ PTI (Psychology Trap Index)
7. ✅ Fake-out intensity
8. ✅ FRVP integration
9. ✅ Price/time clusters

### Phase 3 (Week 3): Macro & Exits
10. ✅ Macro echo rules
11. ✅ Enhanced exit system
12. ✅ Learning loop foundation

### Phase 4 (Week 4): Integration & Testing
13. ✅ Scenario tests (15 tests)
14. ✅ Wiring report
15. ✅ Full system validation

---

## 📝 Next Steps

1. **Immediate**: Start implementing internal/external structure
2. **Today**: Complete BOMS and squiggle pattern
3. **This week**: Finish all structure enhancements
4. **Next week**: Psychology + volume layers
5. **Final week**: Macro echo + exits + learning loop

---

**Status**: Architecture complete, ready for systematic implementation
**Branch**: `feature/ml-meta-optimizer`
**Estimated completion**: 3-4 weeks for full knowledge integration
