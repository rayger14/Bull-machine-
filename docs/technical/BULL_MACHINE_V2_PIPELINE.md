# Bull Machine v2 Signal Pipeline Guide

**Version:** 2.0
**Last Updated:** 2025-11-12
**Target Audience:** Technical contributors with trading knowledge

---

## Table of Contents

1. [Overview](#overview)
2. [Input Stage](#input-stage)
3. [Regime Detection](#regime-detection)
4. [Archetype Detection](#archetype-detection)
5. [Routing & Gates](#routing--gates)
6. [Position Sizing & Exits](#position-sizing--exits)
7. [Validation & Benchmarks](#validation--benchmarks)
8. [Architecture Diagrams](#architecture-diagrams)
9. [Code Examples](#code-examples)

---

## Overview

Bull Machine v2 is a regime-aware algorithmic trading system that combines macro analysis, market microstructure, and technical patterns to identify high-conviction entry points. The system processes 114 features across 4 domains (Wyckoff, Liquidity, Momentum, PTI) to generate fusion scores, then applies archetype-specific filters to execute trades.

### Signal Flow (High-Level)
```
Feature Store (114 features)
    ↓
Regime Classification (GMM v3.2)
    ↓
Fusion Score Computation (4 domains)
    ↓
Archetype Detection (19 patterns)
    ↓
Routing & Gates (regime-aware)
    ↓
Entry Decision (threshold check)
    ↓
Position Sizing (ATR-based)
    ↓
Exit Management (trailing stops)
```

### Key Characteristics
- **Deterministic:** PYTHONHASHSEED=0 ensures reproducible results
- **Regime-Aware:** Parameters morph based on market conditions
- **Archetype-Driven:** 19 distinct patterns with custom thresholds
- **Risk-Managed:** ATR-based sizing with dynamic trailing stops

---

## Input Stage

### Feature Store Overview

The engine consumes a consolidated feature store with 114 columns (excluding metadata):

**Technical Indicators (94 columns):**
- Price action: OHLC, HL2, HLC3, OHLC4
- Trend: SMA (20, 50, 200), EMA (9, 21), ADX, DMI
- Momentum: RSI (14), MACD (12, 26, 9), Stochastic
- Volatility: ATR (14), Bollinger Bands, Keltner Channels
- Volume: Volume SMA, Volume Z-score, OBV
- Microstructure: Order flow imbalance, wick ratios
- Wyckoff: Phase labels (accumulation, markup, distribution)
- Liquidity: BOMS strength, BOS proximity, FVG presence
- PTI: Power Trend Index (1D, 1H timeframes)

**Macro Features (20 columns):**
*Note: Total features = 114 (94 technical + 20 macro). Total columns = 119 (114 features + 5 metadata: open, high, low, close, volume).*
- Volatility: VIX, MOVE (bond vol)
- Currencies: DXY (dollar), EUR/USD
- Rates: US 2Y, 10Y yields, yield curve slope
- Crypto-specific: BTC.D, USDT.D, funding rates, OI premium
- Market cap: TOTAL, TOTAL2, TOTAL3

### Data Loading

**Location:** `engine/context/loader.py`

```python
from engine.context.loader import load_macro_data

# Load macro time series from consolidated parquet
macro_data = load_macro_data(
    data_dir="data",
    asset_type="crypto"  # Auto-selects relevant series
)

# Returns dict: {'VIX': DataFrame, 'DXY': DataFrame, ...}
```

**Consolidated Feature Store:**
```python
import pandas as pd

# Load pre-computed feature store (created by bin/cache_features.py)
df = pd.read_parquet('data/btc_features_2022_2024.parquet')

print(df.columns)
# ['timestamp', 'open', 'high', 'low', 'close', 'volume',
#  'rsi_14', 'adx_14', 'atr_14', 'macd', 'signal', 'hist',
#  'boms_strength', 'bos_proximity', 'fvg_present', 'pti_1d', 'pti_1h',
#  'VIX', 'DXY', 'MOVE', 'YIELD_2Y', 'YIELD_10Y', 'USDT.D', 'BTC.D',
#  'funding', 'oi', ...]
```

### Feature Validation

The backtest engine validates required features on initialization:

```python
# bin/backtest_knowledge_v2.py:226-230
required_features = ['atr_14', 'rsi_14', 'adx_14', 'boms_strength', 'pti_1d']
missing = [f for f in required_features if f not in df.columns]
if missing:
    raise ValueError(f"Missing required features: {missing}")
```

---

## Regime Detection

### GMM-Based Classification

**Model:** Gaussian Mixture Model (GMM) with 5 clusters mapped to 4 regimes
**Location:** `engine/context/regime_classifier.py`
**Model File:** `models/regime_gmm_v3.2_balanced.pkl`

#### Regime Definitions

| Regime | Cluster IDs | Description | Macro Signature |
|--------|-------------|-------------|-----------------|
| **risk_on** | 0, 1 | Bull market, low vol, strong flows | Low VIX (<20), weak DXY, positive funding |
| **neutral** | 2 | Range-bound, uncertain | Mid VIX (20-30), flat DXY, neutral funding |
| **risk_off** | 3 | Bear pressure, risk aversion | High VIX (30-45), strong DXY, negative funding |
| **crisis** | 4 | Market panic, extreme volatility | VIX >45, DXY spike, funding collapse |

#### Feature Order (19 inputs)

```python
MACRO_FEATURE_ORDER = [
    'VIX',           # Volatility index (S&P 500 implied vol)
    'DXY',           # US Dollar index
    'MOVE',          # Bond market volatility
    'YIELD_2Y',      # 2-year Treasury yield
    'YIELD_10Y',     # 10-year Treasury yield
    'USDT.D',        # USDT dominance (stablecoin coil)
    'BTC.D',         # Bitcoin dominance
    'TOTAL',         # Total crypto market cap
    'TOTAL2',        # Alt market cap (ex BTC)
    'funding',       # Perpetual futures funding rate
    'oi',            # Open interest premium
    'rv_20d',        # 20-day realized volatility
    'rv_60d',        # 60-day realized volatility
    # ... (6 more features)
]
```

#### Classification Process

```python
# 1. Extract macro features from current bar
macro_row = {
    'VIX': 18.5,
    'DXY': 102.0,
    'MOVE': 85.0,
    'YIELD_2Y': 4.2,
    'YIELD_10Y': 4.0,
    'USDT.D': 6.8,
    'BTC.D': 54.5,
    'TOTAL': 1000,
    'TOTAL2': 400,
    'funding': 0.008,
    'oi': 0.012,
    'rv_20d': 0.02,
    'rv_60d': 0.025,
    # ...
}

# 2. Load classifier and classify
from engine.context.regime_classifier import RegimeClassifier

rc = RegimeClassifier.load(
    model_path='models/regime_gmm_v3.2_balanced.pkl',
    feature_order=MACRO_FEATURE_ORDER,
    zero_fill_missing=True  # Fill NaNs with 0 instead of fallback
)

result = rc.classify(macro_row)

print(result)
# {
#   'regime': 'risk_on',
#   'proba': {'risk_on': 0.68, 'neutral': 0.22, 'risk_off': 0.08, 'crisis': 0.02},
#   'features_used': 19,
#   'fallback': False
# }
```

#### Zero-Fill vs Fallback

**Zero-Fill Mode (Recommended):**
- Missing features → filled with 0.0
- Allows classification to proceed
- Used in production for real-time feeds

**Fallback Mode (Conservative):**
- Missing features → return 'neutral' regime
- Prevents classification with incomplete data
- Used during data validation

#### Regime Override (Testing Only)

Force specific regime for parity testing:

```python
rc = RegimeClassifier.load(
    model_path='models/regime_gmm_v3.2_balanced.pkl',
    feature_order=MACRO_FEATURE_ORDER,
    regime_override={'2022': 'risk_off', '2024': 'risk_on'}
)

# All 2022 bars → forced to 'risk_off'
# All 2024 bars → forced to 'risk_on'
```

---

## Archetype Detection

### 19 Archetype Patterns

Bull Machine v2 implements 19 distinct market archetypes using rule-based heuristics.

**Location:** `engine/archetypes/logic.py`

#### Bull-Biased Archetypes (11)

| Code | Name | Description | Key Features |
|------|------|-------------|--------------|
| A | Spring | Wyckoff spring/UTAD reversal | PTI divergence, volume spike, wick rejection |
| B | Order Block Retest | BOMS + Wyckoff + BOS proximity | BOMS strength >0.6, BOS <5 bars ago |
| C | Wick Trap | Moneytaur-style wick anomaly | Wick ratio >2.0, ADX >25, trend alignment |
| D | Failed Continuation | FVG present + weak momentum | FVG valid, RSI <50, falling ADX |
| E | Volume Exhaustion | Extreme RSI + volume spike | RSI >70, volume_z >2.0, momentum fade |
| F | Exhaustion Reversal | Climax volume + reversal | High ATR, extreme RSI, wick against trend |
| G | Liquidity Sweep | Hunt stops then reverse | Price pierces swing low/high, quick reversal |
| H | Momentum Continuation | Displacement + FVG | Strong ADX, FVG created, BOS recent |
| K | Trap Within Trend | HTF trend + LTF trap | HTF trend up, 1H wick down, liquidity drop |
| L | Retest Cluster | Multiple structure retests | Price returns to OB/FVG/POC within 3 bars |
| M | Confluence Breakout | Low ATR coil + BOMS + POC | ATR_pctile <30, near POC, BOMS strength high |

#### Bear-Biased Archetypes (8)

| Code | Name | Description | Key Features |
|------|------|-------------|--------------|
| S1 | Breakdown | Support break with volume | Price <support, volume spike, RSI <40 |
| S2 | Rejection | Resistance rejection | Price tests resistance, wick rejection, RSI >60 |
| S3 | Whipsaw | False breakout trap | BOS then immediate reversal |
| S4 | Distribution | Wyckoff distribution phase | High volume at top, weak RSI, SOW present |
| S5 | Short Squeeze | Extreme funding + shorts | Funding <-0.03%, OI spike, RSI <30 |
| S6 | Alt Rotation Down | BTC.D rising, alts bleeding | BTC.D +2% 7D, TOTAL2 -5% 7D |
| S7 | Curve Inversion | Yield curve inversion | 2Y yield > 10Y yield, VIX rising |
| S8 | Volume Fade Chop | Low volume range | ATR_pctile <20, volume_z <0, ADX <15 |

### Archetype Detection Logic

**Priority-Based Selection:**
Archetypes are checked in order (A → M, then S1 → S8). First matching pattern wins.

**Example: Wick Trap (Archetype C)**

```python
# engine/archetypes/logic.py:detect_wick_trap_moneytaur()

def detect_wick_trap_moneytaur(self, row, ctx) -> Tuple[Optional[str], float, float]:
    """
    Wick Trap (Moneytaur): Wick anomaly + ADX + BOS context

    Entry when:
    - Wick ratio >2.0 (body small, wick large)
    - ADX >25 (strong trend)
    - Recent BOS (<5 bars ago)
    - Wick direction against prevailing trend
    """
    # Extract features
    wick_upper = row.get('wick_upper_ratio', 0.0)
    wick_lower = row.get('wick_lower_ratio', 0.0)
    adx = row.get('adx_14', 0.0)
    bos_bars_ago = row.get('bos_bars_ago', 99)
    trend = row.get('trend', 0)  # 1=up, -1=down

    # Thresholds from config
    min_wick_ratio = self.thresh_C.get('min_wick_ratio', 2.0)
    min_adx = self.thresh_C.get('min_adx', 25.0)
    max_bos_bars = self.thresh_C.get('max_bos_bars', 5)

    # Check wick trap conditions
    if trend == 1:  # Uptrend → look for bullish wick trap (lower wick)
        wick_ratio = wick_lower
        wick_direction = 'lower'
    elif trend == -1:  # Downtrend → look for bearish wick trap (upper wick)
        wick_ratio = wick_upper
        wick_direction = 'upper'
    else:
        return (None, 0.0, 0.0)  # No clear trend

    # Pattern match
    if (wick_ratio >= min_wick_ratio and
        adx >= min_adx and
        bos_bars_ago <= max_bos_bars):

        # Compute fusion score (weighted domain scores)
        fusion_score = self._compute_fusion(row, ctx)
        liquidity_score = ctx.get('liquidity_score', 0.0)

        # Check thresholds
        fusion_threshold = self.thresh_C.get('fusion', 0.35)
        liquidity_threshold = self.thresh_C.get('liquidity', 0.20)

        if fusion_score >= fusion_threshold and liquidity_score >= liquidity_threshold:
            return ('wick_trap', fusion_score, liquidity_score)

    return (None, 0.0, 0.0)
```

### Soft Filters

**Global Liquidity Gate:**
All archetypes require minimum liquidity score:

```python
# engine/archetypes/logic.py:detect_all()

min_liquidity = self.config.get('thresholds', {}).get('min_liquidity', 0.30)

if liquidity_score < min_liquidity:
    # Suppress entry even if archetype matches
    return (None, 0.0, 0.0)
```

**Enable/Disable Flags:**

```json
// configs/baseline_btc_bull_strict.json
{
  "archetypes": {
    "use_archetypes": true,
    "enable_A": true,
    "enable_B": true,
    "enable_C": true,
    "enable_S1": false,  // Disable bear archetypes
    "enable_S2": false,
    // ...
  }
}
```

---

## Routing & Gates

### Regime-Aware Routing

**Location:** `engine/archetypes/threshold_policy.py`

Thresholds adjust based on regime probabilities:

```python
# ThresholdPolicy.resolve()

# 1. Start from base thresholds (config)
base_thresholds = {
    'order_block_retest': {'fusion': 0.35, 'liquidity': 0.20},
    'wick_trap': {'fusion': 0.38, 'liquidity': 0.22},
    # ...
}

# 2. Blend regime gates (weighted by probability)
regime_probs = {'risk_on': 0.68, 'neutral': 0.22, 'risk_off': 0.08, 'crisis': 0.02}

blended_fusion_floor = (
    0.68 * 0.28 +  # risk_on floor
    0.22 * 0.32 +  # neutral floor
    0.08 * 0.40 +  # risk_off floor
    0.02 * 0.50    # crisis floor
) = 0.304

# 3. Apply regime floors
for archetype in base_thresholds:
    base_thresholds[archetype]['fusion'] = max(
        base_thresholds[archetype]['fusion'],
        blended_fusion_floor
    )

# 4. Apply archetype-specific overrides (optional)
# 5. Clamp to global guardrails (0.20 - 0.65)
```

**Regime Profiles (Example):**

```json
// configs/baseline_btc_bull_strict.json
{
  "gates_regime_profiles": {
    "risk_on": {
      "final_fusion_floor": 0.28,
      "min_liquidity": 0.18
    },
    "neutral": {
      "final_fusion_floor": 0.32,
      "min_liquidity": 0.22
    },
    "risk_off": {
      "final_fusion_floor": 0.40,
      "min_liquidity": 0.28
    },
    "crisis": {
      "final_fusion_floor": 0.50,
      "min_liquidity": 0.35
    }
  }
}
```

### State-Aware Gates

**Location:** `engine/archetypes/state_aware_gates.py`

Dynamic threshold adjustments based on market state:

```python
# StateAwareGates.compute_gate()

base_gate = 0.35  # From ThresholdPolicy

adjustments = {
    'adx_weak': +0.06 if adx < 18 else 0.0,        # Chop penalty
    'atr_low': +0.05 if atr_pctile < 25 else 0.0,  # Low vol penalty
    'funding_high': +0.05 if funding_z > 1.0 else 0.0,  # Late long penalty
    'tf4h_misalign': +0.03 if tf4h != tf1h else 0.0,    # MTF conflict
    'adx_strong': -0.03 if adx > 30 else 0.0,      # Strong trend bonus
    'funding_low': -0.02 if funding_z < 0 else 0.0     # Short covering bonus
}

total_adjustment = sum(adjustments.values())
total_adjustment = np.clip(total_adjustment, -0.15, +0.15)  # Max ±15%

adjusted_gate = base_gate + total_adjustment
adjusted_gate = np.clip(adjusted_gate, 0.25, 0.75)  # Hard clamps

return adjusted_gate
```

**State Features:**

| Feature | Source | Purpose |
|---------|--------|---------|
| ADX | Technical (trend strength) | Detect chop (ADX <18) vs trend (ADX >30) |
| ATR percentile | Technical (volatility) | Avoid micro-scalps in tight ranges |
| Funding z-score | Macro (leverage) | Detect late-long risk (funding >1σ) |
| 4H trend | MTF (higher timeframe) | Detect MTF misalignment |

### Final Entry Decision

```python
# bin/backtest_knowledge_v2.py:entry_logic()

# 1. Get base threshold from ThresholdPolicy
ctx = RuntimeContext(
    ts=timestamp,
    row=row,
    regime_probs=regime_probs,
    regime_label=regime_label,
    adapted_params=adapted_params,
    thresholds=threshold_policy.resolve(regime_probs, regime_label)
)

base_threshold = ctx.get_threshold('wick_trap', 'fusion', default=0.35)

# 2. Apply state-aware gate adjustment
from engine.archetypes.state_aware_gates import apply_state_aware_gate

final_threshold = apply_state_aware_gate(
    archetype='wick_trap',
    base_gate=base_threshold,
    ctx=ctx,
    gate_module=state_aware_gates
)

# 3. Check if fusion score meets final threshold
if fusion_score >= final_threshold and liquidity_score >= min_liquidity:
    # ENTER TRADE
    execute_entry(archetype='wick_trap', fusion_score=fusion_score)
```

---

## Position Sizing & Exits

### ATR-Based Position Sizing

**Location:** `bin/backtest_knowledge_v2.py:compute_position_size()`

```python
def compute_position_size(self, current_price: float, atr: float) -> Tuple[float, float]:
    """
    Compute position size using ATR-based risk management.

    Risk Model:
    - Max 2% risk per trade
    - Stop loss = 2.5× ATR below entry
    - Position size = (equity × 0.02) / (stop_distance)

    Args:
        current_price: Entry price
        atr: Current ATR value

    Returns:
        (position_size_usd, stop_loss_price)
    """
    # 1. Compute stop distance
    stop_mult = self.params.atr_stop_mult  # 2.5
    stop_distance = stop_mult * atr
    stop_loss = current_price - stop_distance

    # 2. Compute position size
    risk_per_trade = self.equity * self.params.max_risk_pct  # 2%
    position_size = risk_per_trade / stop_distance

    # 3. Apply volatility scaling (reduce size in high VIX)
    if self.params.volatility_scaling:
        vix = self.df.iloc[-1].get('VIX', 20.0)
        if vix > 30.0:
            scale_factor = 0.5  # 50% size in high vol
            position_size *= scale_factor

    return position_size, stop_loss
```

**Example:**
```
Equity: $10,000
Entry: $50,000
ATR: $1,000
Stop mult: 2.5

Stop distance = 2.5 × $1,000 = $2,500
Stop loss = $50,000 - $2,500 = $47,500

Risk per trade = $10,000 × 0.02 = $200
Position size = $200 / $2,500 = 0.08 BTC = $4,000 USD

If stop hit: Loss = 0.08 × $2,500 = $200 (2% of equity)
```

### Exit Management

**Location:** `bin/backtest_knowledge_v2.py:check_exit()`

#### Exit Types

1. **Stop Loss (Fixed)**
   - Price hits initial stop (entry - 2.5× ATR)
   - Always active

2. **Trailing Stop (Dynamic)**
   - Trails price at 2.0× ATR from peak
   - Activated after +1R profit
   - Locks in gains

3. **Partial Exits (Optional)**
   - 33% at TP1 (+1R)
   - 33% at TP2 (+2R)
   - 34% trails to exit

4. **Max Hold (Time-Based)**
   - Default: 168 hours (7 days)
   - Adaptive: Extends in strong Wyckoff phases

5. **Archetype-Specific**
   - Each archetype can override default exits
   - Example: Wick trap uses tighter trailing stop

#### Exit Logic Flow

```python
def check_exit(self, current_price: float, row: pd.Series) -> Optional[Tuple[str, float]]:
    """
    Check all exit conditions for current position.

    Returns:
        (exit_reason, exit_price) or None
    """
    if self.current_position is None:
        return None

    # 1. Stop loss (fixed)
    if current_price <= self.current_position.initial_stop:
        return ('stop_loss', current_price)

    # 2. Update trailing stop
    profit = current_price - self.current_position.entry_price
    if profit >= self.current_position.atr_at_entry:  # +1R
        peak_price = max(self.current_position.peak_profit, current_price)
        trailing_stop = peak_price - (self.params.trailing_atr_mult * self.current_position.atr_at_entry)

        if current_price <= trailing_stop:
            return ('trailing_stop', current_price)

    # 3. Partial exits (if enabled)
    if self.params.use_smart_exits:
        if profit >= 2.0 * self.current_position.atr_at_entry:  # TP2 (+2R)
            self._execute_partial_exit(0.33, 'tp2')
        elif profit >= 1.0 * self.current_position.atr_at_entry:  # TP1 (+1R)
            self._execute_partial_exit(0.33, 'tp1')

    # 4. Max hold (time-based)
    bars_in_trade = len(self.df) - self.current_position.entry_bar
    max_hold = self._get_adaptive_max_hold(row)
    if bars_in_trade >= max_hold:
        return ('max_hold', current_price)

    # 5. Archetype-specific exits
    archetype_exit = self._check_archetype_exit(row)
    if archetype_exit:
        return archetype_exit

    return None  # Hold
```

#### Adaptive Max Hold

```python
def _get_adaptive_max_hold(self, row: pd.Series) -> int:
    """
    Adjust max hold based on Wyckoff phase and regime.

    Extensions:
    - Markup phase + risk_on: +50% hold time
    - Distribution phase: -30% hold time
    - Crisis regime: -50% hold time
    """
    base_hold = self.params.max_hold_bars  # 168 hours

    if not self.params.adaptive_max_hold:
        return base_hold

    wyckoff_phase = row.get('wyckoff_phase', 'none')
    regime = row.get('regime', 'neutral')

    multiplier = 1.0

    # Phase-based adjustments
    if wyckoff_phase == 'markup' and regime == 'risk_on':
        multiplier = 1.5  # Extend in strong bull phase
    elif wyckoff_phase == 'distribution':
        multiplier = 0.7  # Exit faster in distribution

    # Regime-based adjustments
    if regime == 'crisis':
        multiplier *= 0.5  # Cut hold time in half during crisis

    adjusted_hold = int(base_hold * multiplier)
    return adjusted_hold
```

---

## Validation & Benchmarks

### Gold Standard Test

**Location:** `results/bench_v2/GOLD_STANDARD_REPORT.md`

#### Run Command

```bash
# Set deterministic seed
export PYTHONHASHSEED=0

# Run full backtest (2022-2024)
python bin/backtest_knowledge_v2.py \
  --config configs/baseline_btc_bull_strict.json \
  --start 2022-01-01 \
  --end 2024-12-31 \
  --soft-filters-off \
  --output results/bench_v2/gold_standard_run.json
```

#### Expected Metrics

| Year | Trades | Win Rate | Profit Factor | Net PNL | Max DD |
|------|--------|----------|---------------|---------|--------|
| 2022 | 13 | 15.4% | 0.15 | -$598 | -18.2% |
| 2023 | 21 | 66.7% | 3.85 | +$1,246 | -8.4% |
| 2024 | 17 | 76.5% | 6.17 | +$1,285 | -4.2% |
| **Total** | **51** | **56.9%** | **2.09** | **+$1,933** | **-18.2%** |

#### Regression Checks

```python
# tests/test_gold_standard.py

def test_gold_standard_parity():
    """Verify backtest matches gold standard results."""
    results = run_backtest(config='baseline_btc_bull_strict.json')

    # Trade count (exact match)
    assert results['2022']['trade_count'] == 13
    assert results['2023']['trade_count'] == 21
    assert results['2024']['trade_count'] == 17

    # PNL (within $1 due to rounding)
    assert abs(results['2022']['net_pnl'] - (-598)) < 1.0
    assert abs(results['2023']['net_pnl'] - 1246) < 1.0
    assert abs(results['2024']['net_pnl'] - 1285) < 1.0

    # Profit factor (within 5%)
    assert abs(results['2022']['pf'] - 0.15) < 0.01
    assert abs(results['2023']['pf'] - 3.85) < 0.20
    assert abs(results['2024']['pf'] - 6.17) < 0.30
```

### Known Edge Cases

1. **Regime Boundary Transitions**
   - Hysteresis in regime classification may cause ±1 trade variance
   - Expected when testing near regime shift dates

2. **Zero-Fill vs NaN Handling**
   - Zero-fill mode may classify differently than fallback mode
   - Always use same mode for parity testing

3. **Archetype Priority**
   - If multiple archetypes score equally, first in order wins
   - Trade count may vary by ±1 in rare cases

### Archetype Performance Analysis

```python
# bin/analyze_archetype_perf.py

import pandas as pd

trades_df = pd.read_csv('results/gold_standard_run.csv')

# Group by archetype
arch_stats = trades_df.groupby('entry_archetype').agg({
    'net_pnl': ['sum', 'mean'],
    'gross_pnl': 'sum',
    'trade_id': 'count'
}).round(2)

print(arch_stats)
#                          net_pnl          gross_pnl  trade_id
#                              sum   mean        sum     count
# entry_archetype
# order_block_retest          450.2  32.2      520.0        14
# wick_trap                   380.5  27.2      440.0        14
# spring                      220.0  31.4      250.0         7
# trap_within_trend           185.0  46.3      200.0         4
# ...
```

---

## Architecture Diagrams

### Full Signal Pipeline (Detailed)

```
┌─────────────────────────────────────────────────────────────┐
│                     Feature Store Loader                     │
│  (engine/context/loader.py)                                 │
│  • Load 114 features from parquet (119 cols total)          │
│  • Validate required columns                                │
│  • Zero-fill missing macro features                         │
└─────────────────┬───────────────────────────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────────────────────────┐
│                   Regime Classifier                          │
│  (engine/context/regime_classifier.py)                      │
│  • Extract 19 macro features                                │
│  • GMM v3.2 predict (5 clusters → 4 regimes)                │
│  • Return regime label + probability distribution            │
└─────────────────┬───────────────────────────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────────────────────────┐
│                   Adaptive Fusion                            │
│  (engine/fusion/adaptive.py)                                │
│  • Morph domain weights by regime                           │
│  • Adjust ML threshold by regime                            │
│  • Smooth transitions with EMA                              │
└─────────────────┬───────────────────────────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────────────────────────┐
│                  Fusion Score Compute                        │
│  (bin/backtest_knowledge_v2.py:compute_advanced_fusion)     │
│  • Wyckoff domain (33%)                                     │
│  • Liquidity domain (39%)                                   │
│  • Momentum domain (21%)                                    │
│  • PTI domain (7%)                                          │
│  • Return weighted fusion score (0.0-1.0)                   │
└─────────────────┬───────────────────────────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────────────────────────┐
│                  Archetype Detection                         │
│  (engine/archetypes/logic.py)                               │
│  • Check 19 patterns in priority order                      │
│  • First match wins                                         │
│  • Return (archetype, fusion_score, liquidity_score)        │
└─────────────────┬───────────────────────────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────────────────────────┐
│                  Threshold Policy                            │
│  (engine/archetypes/threshold_policy.py)                    │
│  • Get base thresholds from config                          │
│  • Blend regime gates (weighted by proba)                   │
│  • Apply archetype-specific overrides                       │
│  • Clamp to global guardrails                               │
└─────────────────┬───────────────────────────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────────────────────────┐
│                  State-Aware Gates                           │
│  (engine/archetypes/state_aware_gates.py)                   │
│  • Adjust threshold based on ADX, ATR, funding              │
│  • Penalize chop, low vol, late longs                       │
│  • Reward strong trends, short covering                     │
└─────────────────┬───────────────────────────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────────────────────────┐
│                    Entry Decision                            │
│  • fusion_score >= final_threshold?                         │
│  • liquidity_score >= min_liquidity?                        │
│  • ML filter pass (if enabled)?                             │
│  ├─ YES → Execute Entry                                     │
│  └─ NO  → Skip                                              │
└─────────────────┬───────────────────────────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────────────────────────┐
│                   Position Sizing                            │
│  • Compute stop distance (2.5× ATR)                         │
│  • Size for 2% max risk                                     │
│  • Apply volatility scaling (if VIX high)                   │
└─────────────────┬───────────────────────────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────────────────────────┐
│                    Exit Management                           │
│  • Fixed stop loss (entry - 2.5× ATR)                       │
│  • Trailing stop (peak - 2.0× ATR)                          │
│  • Partial exits (33% @ TP1, 33% @ TP2)                     │
│  • Max hold (168 hours, adaptive)                           │
│  • Archetype-specific exits                                 │
└─────────────────┬───────────────────────────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────────────────────────┐
│                      PNL Tracking                            │
│  • Record trade entry/exit                                  │
│  • Compute gross/net PNL (fees + slippage)                  │
│  • Update equity and drawdown                               │
│  • Log telemetry                                            │
└─────────────────────────────────────────────────────────────┘
```

### Regime Classification Flow

```
Macro Features (19)
    │
    ├─ VIX, DXY, MOVE
    ├─ YIELD_2Y, YIELD_10Y
    ├─ USDT.D, BTC.D, TOTAL
    ├─ funding, oi
    └─ rv_20d, rv_60d, ...
    │
    ▼
Feature Normalization
    │  (zero-fill NaNs)
    │
    ▼
GMM Predict (5 clusters)
    │
    ├─ Cluster 0 → risk_on (p=0.30)
    ├─ Cluster 1 → risk_on (p=0.38)
    ├─ Cluster 2 → neutral (p=0.22)
    ├─ Cluster 3 → risk_off (p=0.08)
    └─ Cluster 4 → crisis (p=0.02)
    │
    ▼
Aggregate Probabilities
    │
    ├─ risk_on: 0.68 (0.30 + 0.38)
    ├─ neutral: 0.22
    ├─ risk_off: 0.08
    └─ crisis: 0.02
    │
    ▼
Argmax → regime_label = 'risk_on'
```

---

## Code Examples

### Complete Backtest Example

```python
#!/usr/bin/env python3
"""
Run Bull Machine v2 backtest with full pipeline.
"""

import pandas as pd
from bin.backtest_knowledge_v2 import KnowledgeAwareBacktest, KnowledgeParams
import json

# 1. Load feature store
df = pd.read_parquet('data/btc_features_2022_2024.parquet')
print(f"Loaded {len(df)} bars with {len(df.columns)} features")

# 2. Configure parameters
params = KnowledgeParams(
    wyckoff_weight=0.331,
    liquidity_weight=0.392,
    momentum_weight=0.205,
    macro_weight=0.0,
    pti_weight=0.072,
    tier3_threshold=0.374,
    max_risk_pct=0.02,
    atr_stop_mult=2.5,
    trailing_atr_mult=2.0,
    max_hold_bars=168
)

# 3. Load runtime config
with open('configs/baseline_btc_bull_strict.json') as f:
    config = json.load(f)

# 4. Initialize backtest
bt = KnowledgeAwareBacktest(
    df=df,
    params=params,
    starting_capital=10000.0,
    asset='BTC',
    runtime_config=config
)

# 5. Run backtest
results = bt.run()

# 6. Print results
print(f"\n{'='*60}")
print(f"BACKTEST RESULTS")
print(f"{'='*60}")
print(f"Total Trades:     {results['trade_count']}")
print(f"Win Rate:         {results['win_rate']:.1%}")
print(f"Profit Factor:    {results['profit_factor']:.2f}")
print(f"Net PNL:          ${results['net_pnl']:,.2f}")
print(f"Max Drawdown:     {results['max_drawdown']:.1%}")
print(f"Sharpe Ratio:     {results['sharpe_ratio']:.2f}")

# 7. Export trades
trades_df = pd.DataFrame([vars(t) for t in results['trades']])
trades_df.to_csv('results/trades_2022_2024.csv', index=False)
print(f"\nExported {len(trades_df)} trades to results/trades_2022_2024.csv")

# 8. Archetype breakdown
arch_stats = trades_df.groupby('entry_archetype')['net_pnl'].agg(['sum', 'count', 'mean'])
print(f"\nArchetype Performance:")
print(arch_stats.sort_values('sum', ascending=False))
```

### Custom Archetype Implementation

```python
# engine/archetypes/logic.py

def detect_custom_pattern(self, row, ctx) -> Tuple[Optional[str], float, float]:
    """
    Custom archetype: Your pattern here.

    Args:
        row: Current bar (pd.Series)
        ctx: Runtime context dict

    Returns:
        (archetype_name, fusion_score, liquidity_score) or (None, 0, 0)
    """
    # 1. Extract features
    rsi = row.get('rsi_14', 50.0)
    adx = row.get('adx_14', 0.0)
    boms = row.get('boms_strength', 0.0)
    volume_z = row.get('volume_z', 0.0)

    # 2. Define pattern logic
    pattern_match = (
        rsi < 30.0 and        # Oversold
        adx > 25.0 and        # Strong trend
        boms > 0.6 and        # Strong order block
        volume_z > 1.5        # Volume spike
    )

    if not pattern_match:
        return (None, 0.0, 0.0)

    # 3. Compute fusion score
    fusion_score = self._compute_fusion(row, ctx)
    liquidity_score = ctx.get('liquidity_score', 0.0)

    # 4. Check thresholds
    min_fusion = self.thresh_CUSTOM.get('fusion', 0.40)
    min_liquidity = self.thresh_CUSTOM.get('liquidity', 0.25)

    if fusion_score >= min_fusion and liquidity_score >= min_liquidity:
        return ('custom_pattern', fusion_score, liquidity_score)

    return (None, 0.0, 0.0)
```

### Regime-Specific Strategy

```python
# Example: Only trade in risk_on regime

# configs/risk_on_only.json
{
  "gates_regime_profiles": {
    "risk_on": {
      "final_fusion_floor": 0.30,
      "min_liquidity": 0.20
    },
    "neutral": {
      "final_fusion_floor": 0.80,  // Extremely high (no entries)
      "min_liquidity": 0.80
    },
    "risk_off": {
      "final_fusion_floor": 0.80,
      "min_liquidity": 0.80
    },
    "crisis": {
      "final_fusion_floor": 0.80,
      "min_liquidity": 0.80
    }
  }
}

# Result: Only enters trades when regime = 'risk_on'
```

---

## Summary

Bull Machine v2's signal pipeline transforms 114 raw features into actionable trading signals through a sophisticated multi-stage process:

1. **Input Stage:** Feature store with 94 technical + 20 macro indicators (114 total features, 119 columns including metadata)
2. **Regime Detection:** GMM-based classification (4 regimes)
3. **Fusion Scoring:** Weighted combination of 4 domains
4. **Archetype Detection:** 19 rule-based pattern matchers
5. **Routing & Gates:** Regime-aware + state-aware threshold adjustments
6. **Position Management:** ATR-based sizing + adaptive exits

The system is fully deterministic (PYTHONHASHSEED=0), enabling reproducible backtests and regression testing. Gold standard metrics (2022-2024) serve as the baseline for performance validation.

---

## Next Steps

1. **Extend to Multi-Asset:** Replicate pipeline for ETH, SOL
2. **Add Bear Archetypes:** Enable S1-S8 for short-biased strategies
3. **Live Trading Prep:** Real-time regime detection integration
4. **ML Enhancement:** Train archetype classifier on labeled data

---

**Document Version:** 1.0
**Last Updated:** 2025-11-12
**Maintained By:** Bull Machine v2 Integration Team
**Related Docs:**
- `results/bench_v2/GOLD_STANDARD_REPORT.md` - Validated baseline metrics
- `docs/PR6B_REGIME_AWARE_REFACTOR.md` - Regime-aware architecture
- `docs/TESTING_METHODOLOGY.md` - Validation procedures
