# Bull Machine Fusion Score & Threshold Architecture

## Feature Store Analysis

### Precomputed Fusion Columns (V12_ENHANCED)
All fusion columns are **precomputed at feature store build time** and stored in `/data/features_mtf/BTC_1H_FEATURES_V12_ENHANCED.parquet`:

| Column | Coverage | Mean | Max | Purpose |
|--------|----------|------|-----|---------|
| **fusion_total** | 100% | 0.4395 | 0.7794 | Composite score (all 4 domains) |
| **fusion_liquidity** | 100% | 0.4830 | 1.0000 | BOMS/volume-based (main work horse) |
| **fusion_wyckoff** | 100% | 0.4961 | 0.7000 | Max(spring, LPS, SOS) from structure events |
| **fusion_momentum** | 100% | 0.2811 | 0.8189 | ADX, RSI, MACD-based momentum |
| **fusion_smc** | 14.3% | 0.0631 | 1.0000 | Supply/demand, BOS/CHoCH detection (sparse) |
| **k2_fusion_score** | 100% | 0.2669 | 0.4129 | Alternative combined score |
| **tf1h_fusion_score** | 62.1% | 0.0122 | 0.2000 | 1H timeframe composite (low values) |
| **tf4h_fusion_score** | 100% | 0.4428 | 0.7426 | 4H timeframe composite (backtest primary) |
| **tf1d_fusion_score** | 100% | 0.4433 | 0.6565 | Daily timeframe composite |

### Key Insight: Precomputed Doesn't Mean Static
- `fusion_total`, `fusion_wyckoff`, `fusion_momentum` are built at feature store creation
- `fusion_smc` has sparse coverage (14.3%) because SMC events (BOS/CHoCH) are rare
- At **runtime in the backtester**, these are FURTHER modified by:
  - **Wyckoff event boosts** (detect Spring, LPS, SOS, UT patterns live)
  - **SMC confluence boosts** (detect live BOS/CHoCH)
  - **Temporal adjustments** (Fibonacci clusters, hour-of-day)
  - **Crisis penalty** (multiply by `1 - crisis_prob * 0.4`)

---

## Fusion Score Calculation Pipeline

### Step 1: Raw Archetype Pattern Detection
**Location**: `engine/archetypes/logic_v2_adapter.py`

Each archetype evaluates 8-15 pattern conditions (gates). Example for `trap_within_trend`:
```
gate_1: ADX > 15 (trend confirmation)
gate_2: RSI < 40 or RSI > 60 (extremes)
gate_3: Volume > SMA_20 (liquidity)
gate_4: Wick ratio < 1.5 (trap setup)
gate_5: Fib cluster detected (confluence)
...
```

**Result**: Boolean pattern match. If ANY gate fails → VETO, no signal generated.

### Step 2: Base Fusion Score Lookup
**Location**: Logic adapter line ~234

```python
# Prefer precomputed scores from feature store
# Fallback chain: fusion_score → tf4h_fusion_score → k2_fusion_score → default 0.3
fusion_score = self.g(row, "fusion_score", 
                     self.g(row, "tf4h_fusion_score", 
                     self.g(row, "k2_fusion_score", 0.3)))
```

**Result**: `fusion_score ∈ [0, 0.78]` (typically 0.3-0.65 for live signals)

### Step 3: Runtime Wyckoff Event Boosts
**Location**: Logic adapter `_apply_wyckoff_event_boosts()` line ~415

```python
# Detect 5 live events: Spring, LPS, SOS, UT, UTAD
spring_score = _compute_spring_confidence(...)      # 0-1
sos_lps_score = max(_compute_sos_conf(), _lps_conf())  # 0-1
utad_score = _compute_utad_confidence(...)          # 0-1

# Max boost: multiply original fusion by max(1, event_boosts)
# Example: fusion_score=0.4 + spring=0.8 → boosted=0.4 * 1.8 = 0.72
if spring_score > 0.5 or sos_lps_score > 0.5:
    total_boost = max(spring_score, sos_lps_score, utad_score)
    boosted_fusion = fusion_score * (1.0 + total_boost)
    return min(boosted_fusion, 1.0), metadata
```

**Result**: `fusion_score ∈ [0.3, 1.0]` (can exceed original)

### Step 4: Live SMC Confluence Boost
**Location**: Logic adapter `_apply_smc_confluence_boost()` line ~490

```python
# Detect live BOS/CHoCH patterns
bos_strength = _detect_break_of_structure(...)      # 0-1
choch_strength = _detect_choch(...)                 # 0-1

# Add SMC bonus if confluence detected
if bos_strength > 0.3 or choch_strength > 0.3:
    smc_boost = max(bos_strength, choch_strength) * 0.5  # Cap boost at 0.5x
    adjusted_fusion = fusion_score + smc_boost
    return adjusted_fusion, metadata
```

**Result**: Additional +0.15-0.50 added to fusion score

### Step 5: Temporal Confluences
**Location**: Logic adapter `adjust_fusion_weight()` line ~560

```python
# Fibonacci cluster bonus: if multiple timeframes align
# Hour-of-day bias: US market hours (13-22 UTC) premium
# Day-of-week bias: avoid Sundays/Mondays

temporal_weight = 1.0
if fib_cluster_count >= 3:
    temporal_weight *= 1.2
if hour_in_US_session:
    temporal_weight *= 1.05
adjusted_fusion = original_fusion * temporal_weight
```

**Result**: ±5-20% adjustment to fusion score

### Step 6: Crisis Penalty (Dynamic Thresholds)
**Location**: Backtester `backtest_v11_standalone.py` line ~650

```python
# Compute crisis_prob [0-1] from 3 dimensions
crisis_signals = (drawdown_persist > 0.96) + (crash_freq >= 2) + (crisis_persist > 0.55)
if crisis_signals >= 2:
    base_crisis = min(0.7 + 0.1 * crisis_signals, 1.0)
else:
    base_crisis = 0.02-0.10

vol_shock = min(max(rv_20d - 0.8, 0) / 0.4, 1.0)
sentiment_crisis = max(0.0, (0.20 - fear_greed) / 0.20)

crisis_prob = 0.6 * base_crisis + 0.2 * vol_shock + 0.2 * sentiment_crisis

# Apply penalty: multiplier that reduces all fusion scores in crisis
crisis_penalty = 1.0 - crisis_prob * 0.4  # Max penalty = 0.6x when crisis_prob=1.0
adjusted_fusion = original_fusion * crisis_penalty
```

**Result**: `fusion_score *= [0.6, 1.0]` depending on crisis state

---

## Dynamic Threshold Architecture

### CMI v0 (Contextual Market Intelligence)
The threshold is **NOT fixed**. It **adapts continuously** based on market conditions.

**Location**: Backtester line ~560-650

### Step 1: Risk Temperature [0-1]
Measures market bias: 0=bear, 1=bull

```python
# Trend alignment (45% weight)
if price > ema50 and ema50 > ema200:
    trend_align = 1.0      # Bull regime
elif price > ema50:
    trend_align = 0.6      # Early recovery
elif ema50 > ema200:
    trend_align = 0.4      # Distribution
else:
    trend_align = 0.0      # Bear regime

# Momentum health (25% weight)
trend_strength = min(adx / 40.0, 1.0)  # 0=choppy, 1=strong trend

# Sentiment contrarian (15% weight)
sentiment_score = fear_greed_norm      # 0=extreme fear, 1=extreme greed

# Drawdown context (10% weight)
dd_score = max(1.0 - drawdown_persistence, 0.0)

# Derivatives heat (5% weight) — only if OI data available
derivatives_heat = 0.4*oi_momentum + 0.3*funding_health + 0.3*taker_conviction

risk_temp = 0.45*trend_align + 0.25*trend_strength + 0.15*sentiment_score + 0.10*dd_score + 0.05*derivatives_heat
```

**Range**: `risk_temp ∈ [0, 1]`

### Step 2: Instability Score [0-1]
Measures market choppiness/risk: 0=stable trending, 1=choppy/unstable

```python
# Choppiness (35% weight)
chop = chop_score    # 0=trending, 1=choppy

# ADX weakness (25% weight)
adx_weakness = max(1.0 - adx / 40.0, 0.0)  # High ADX = stable

# Wick extremes (20% weight)
wick_sc = min(wick_ratio / 5.0, 1.0)  # Large wicks = unstable

# Volume extremes (20% weight)
vol_instab = min(abs(volume_z_7d) / 2.5, 1.0)  # Extreme vol = unstable

instability = 0.35*chop + 0.25*adx_weakness + 0.20*wick_sc + 0.20*vol_instab
```

**Range**: `instability ∈ [0, 1]`

### Step 3: Crisis Probability [0-1]
Measures pure stress/drawdown: 0=healthy, 1=crisis

```python
# Drawdown signals (60% weight)
crash_frequency = count(drawdown > 2%) in last 7 days
crisis_persistence = duration(current_drawdown) / max_duration
drawdown_persistence = (current_dd - min_recent_dd) / (max_recent_dd - min_recent_dd)

crisis_signals = (dd_persist > 0.96) + (crash_freq >= 2) + (crisis_persist > 0.55)
if crisis_signals >= 2:
    base_crisis = min(0.7 + 0.1*crisis_signals, 1.0)
elif crisis_signals == 1:
    base_crisis = 0.10
else:
    base_crisis = 0.02

# Volatility shock (20% weight)
vol_shock = min(max(rv_20d - 0.8, 0) / 0.4, 1.0)  # rv > 0.8 = shock

# Sentiment extreme (20% weight)
sentiment_crisis = max(0.0, (0.20 - fear_greed) / 0.20)  # F&G < 20 = crisis

crisis_prob = 0.6*base_crisis + 0.2*vol_shock + 0.2*sentiment_crisis
```

**Range**: `crisis_prob ∈ [0, 1]`

### Step 4: Threshold Composition
The threshold is **PER-ARCHETYPE and DYNAMIC**:

```python
# Per-archetype base thresholds (discovered via Optuna)
per_arch_thresholds = {
    "trap_within_trend": 0.15,
    "liquidity_vacuum": 0.15,
    "retest_cluster": 0.12,
    "wick_trap": 0.18,
    "failed_continuation": 0.15,
    "liquidity_sweep": 0.15,
}

# Adaptive adjustment
base_threshold = per_arch_thresholds.get(archetype_id, 0.18)  # Fallback
temp_range = 0.48  # Bear penalty range (Variant A)
instab_range = 0.15  # Chop penalty range

# Dynamic threshold for each archetype
arch_threshold = base_threshold + (1.0 - risk_temp) * temp_range + instability * instab_range

# Global threshold (for logging/averaging)
flat_threshold = base_threshold_global + (1.0 - risk_temp) * temp_range + instability * instab_range
```

### Step 5: Signal Filtering
Each signal's fusion score is **compared against archetype-specific dynamic threshold**:

```python
for signal in raw_signals:
    arch_base = per_arch_thresholds.get(signal.archetype_id, 0.18)
    arch_threshold = arch_base + (1.0 - risk_temp) * temp_range + instability * instab_range
    
    # Apply crisis penalty
    crisis_penalty = 1.0 - crisis_prob * 0.4
    adjusted_fusion = signal.fusion_score * crisis_penalty
    
    # Compare
    if adjusted_fusion >= arch_threshold:
        # Signal passed — store adjusted fusion for allocation
        signal.fusion_score = adjusted_fusion
        accepted_signals.append(signal)
    else:
        # Signal rejected — log reason
        logger.debug(f"Rejected {signal.archetype_id}: {adjusted_fusion:.3f} < {arch_threshold:.3f}")
        signals_rejected += 1
```

---

## Threshold Evolution Examples

### Example 1: Bull Market (2023-Q1 Data)
```
Conditions:
  price=23,500 > ema50=22,800 > ema200=22,100
  adx=28 (strong trend)
  fear_greed_norm=0.65 (moderate greed)
  drawdown_persistence=0.20 (healthy)
  rv_20d=0.55 (calm)

Risk Temperature:
  trend_align=1.0 (price > ema50 > ema200)
  trend_strength=0.70 (adx=28, max=40)
  sentiment_score=0.65
  dd_score=0.80 (healthy)
  derivatives_heat=0.65
  → risk_temp = 0.45*1.0 + 0.25*0.70 + 0.15*0.65 + 0.10*0.80 + 0.05*0.65 = 0.80

Instability:
  chop=0.35 (trending)
  adx_weakness=0.30 (adx=28, strong)
  wick_sc=0.20 (moderate wicks)
  vol_instab=0.15 (calm vol)
  → instability = 0.35*0.35 + 0.25*0.30 + 0.20*0.20 + 0.20*0.15 = 0.23

Crisis:
  drawdown_persistence=0.20 (low)
  crash_freq=0 (no recent crashes)
  vol_shock=0.0 (rv=0.55 < 0.8)
  sentiment_crisis=0.0 (no extreme fear)
  → crisis_prob = 0.6*0.02 + 0.2*0 + 0.2*0 = 0.012

Thresholds (per archetype):
  trap_within_trend:    0.15 + (1-0.80)*0.48 + 0.23*0.15 = 0.15 + 0.096 + 0.035 = 0.281
  retest_cluster:       0.12 + (1-0.80)*0.48 + 0.23*0.15 = 0.12 + 0.096 + 0.035 = 0.251
  wick_trap:            0.18 + (1-0.80)*0.48 + 0.23*0.15 = 0.18 + 0.096 + 0.035 = 0.311

Fusion scores come in at 0.45 (mean) → ALL PASS these permissive thresholds
```

### Example 2: Bear Market (2022-Q3 Data)
```
Conditions:
  price=16,500 < ema50=18,200, ema50 < ema200=19,800
  adx=18 (weak trend)
  fear_greed_norm=0.25 (fear)
  drawdown_persistence=0.85 (significant)
  rv_20d=0.92 (elevated)

Risk Temperature:
  trend_align=0.0 (bear regime)
  trend_strength=0.45 (adx=18)
  sentiment_score=0.25
  dd_score=0.15 (drawdown=0.85)
  derivatives_heat=0.35 (OI negative, funding high)
  → risk_temp = 0.45*0.0 + 0.25*0.45 + 0.15*0.25 + 0.10*0.15 + 0.05*0.35 = 0.17

Instability:
  chop=0.70 (choppy)
  adx_weakness=0.55 (adx=18, weak)
  wick_sc=0.40 (volatile wicks)
  vol_instab=0.65 (vol_z=1.6)
  → instability = 0.35*0.70 + 0.25*0.55 + 0.20*0.40 + 0.20*0.65 = 0.58

Crisis:
  drawdown_persistence=0.85 (high)
  crash_freq=1 (some recent crashes)
  vol_shock=0.30 (rv=0.92 > 0.8)
  sentiment_crisis=0.20 (fear_greed=0.25)
  → crisis_prob = 0.6*0.10 + 0.2*0.30 + 0.2*0.20 = 0.16

Thresholds (per archetype):
  trap_within_trend:    0.15 + (1-0.17)*0.48 + 0.58*0.15 = 0.15 + 0.398 + 0.087 = 0.635
  retest_cluster:       0.12 + (1-0.17)*0.48 + 0.58*0.15 = 0.12 + 0.398 + 0.087 = 0.605
  wick_trap:            0.18 + (1-0.17)*0.48 + 0.58*0.15 = 0.18 + 0.398 + 0.087 = 0.665

Fusion scores ~0.45 at PENALTY of (1 - 0.16*0.4) = 0.936
  → adjusted_fusion = 0.45 * 0.936 = 0.421

0.421 < 0.635 → REJECTED for trap_within_trend (too selective in bear)
0.421 < 0.605 → REJECTED for retest_cluster
0.421 < 0.665 → REJECTED for wick_trap
```

### Example 3: Crisis (2020-03-12 Flash Crash)
```
Conditions:
  price=7,800 (down 40% in 3 days)
  adx=15 (confused)
  fear_greed_norm=0.10 (extreme fear)
  drawdown_persistence=0.99 (SEVERE)
  crash_frequency=3 (multiple 5%+ drops)
  rv_20d=1.20 (extreme vol)

Risk Temperature:
  → risk_temp ≈ 0.05 (nearly bear extreme)

Instability:
  → instability ≈ 0.85 (very choppy)

Crisis:
  base_crisis = min(0.7 + 0.1*3, 1.0) = 1.0 (3 crisis signals)
  vol_shock = min(max(1.20 - 0.8, 0) / 0.4, 1.0) = 1.0
  sentiment_crisis = max(0, (0.20 - 0.10) / 0.20) = 0.5
  → crisis_prob = 0.6*1.0 + 0.2*1.0 + 0.2*0.5 = 0.9

Thresholds:
  trap_within_trend: 0.15 + (1-0.05)*0.48 + 0.85*0.15 = 0.15 + 0.456 + 0.128 = 0.734

Crisis Penalty:
  crisis_penalty = 1.0 - 0.9*0.4 = 0.64

Fusion score 0.45 → adjusted = 0.45 * 0.64 = 0.288
0.288 < 0.734 → ALL REJECTED (emergency protection)

Separate Emergency Cap:
  if crisis_prob (0.9) > emergency_threshold (0.7):
    Any surviving signals multiplied by 0.5x sizing
```

---

## Key Takeaways

1. **Fusion Scores Are Not Precomputed Only**
   - Base scores in feature store are modified 3-5x at runtime
   - Wyckoff events, SMC patterns, temporal confluences, and crisis all affect the final fusion

2. **Thresholds Are Continuous, Not Binary**
   - No more "bull regime = 0.24, bear regime = 0.30"
   - Instead: threshold = base + (1 - risk_temp) * temp_range + instability * instab_range
   - Ranges from ~0.25 (permissive bull) to ~0.75+ (strict crisis)

3. **Per-Archetype Thresholds Are Critical**
   - Liquidity_vacuum needs 0.15 (low gates, needs help)
   - Retest_cluster needs 0.12 (very sparse pattern)
   - Wick_trap stays at 0.18 (frequent signals, can afford selectivity)
   - Uniform threshold = many false signals from low-gate archetypes

4. **Crisis Penalty Is Orthogonal to Threshold**
   - Threshold rises in crisis (higher bar to pass)
   - Fusion score itself multiplied by crisis_penalty (0.6-1.0x)
   - Double protection: harder threshold + weaker fusion

5. **Risk Temperature vs Instability Are Different**
   - risk_temp = bias/direction (0=bear, 1=bull) — determines threshold level
   - instability = noise/chop (0=trending, 1=choppy) — adds to threshold
   - Both independent, both matter

6. **Variant A Enhancement**
   - Original: temp_range = 0.40
   - Variant A: temp_range = 0.48 (+20% stronger bear penalty)
   - Result: bear threshold increased by 0.08, surgically rejecting marginal trades

