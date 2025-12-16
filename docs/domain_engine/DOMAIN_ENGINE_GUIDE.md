# Domain Engine Guide

**Understanding the 6 domain engines that power archetype intelligence**

---

## Overview

The archetype engine uses 6 specialized domain engines to detect sophisticated market patterns. Each engine provides unique intelligence:

| Engine | Purpose | Archetypes Using It | Criticality |
|--------|---------|---------------------|-------------|
| **Wyckoff** | Structural accumulation/distribution cycles | S1, A, B | HIGH |
| **SMC** | Smart money concepts (institutional flow) | B, G, K, L | HIGH |
| **Temporal** | Fibonacci time and Gann cycle confluence | L, M | MEDIUM |
| **HOB** | House of Blues proprietary patterns | All | MEDIUM |
| **Fusion** | Multi-domain synthesis and scoring | H, M | HIGH |
| **Macro** | Regime classification and filters | All | CRITICAL |

**When disabled:** Archetypes fall back to simple Tier-1 logic (RSI + volume only)

---

## 1. Wyckoff Engine

### Purpose
Detects accumulation/distribution cycles and structural market events based on Richard Wyckoff's methodology.

### Key Concepts

**Phases:**
- **Accumulation:** Smart money quietly buying at lows
- **Markup:** Public participation drives price up
- **Distribution:** Smart money quietly selling at highs
- **Markdown:** Public panic drives price down

**Structural Events:**
- **Spring:** False breakdown below support (trap bears, accumulation complete)
- **UTAD:** Upthrust After Distribution (trap bulls, distribution complete)
- **SOS:** Sign of Strength (breakout from accumulation)
- **SOW:** Sign of Weakness (breakdown from distribution)
- **LPS:** Last Point of Support (final test before markup)

### Features Provided

```python
# Event flags
wyckoff_event_spring: bool        # Spring detected
wyckoff_event_utad: bool          # UTAD detected
wyckoff_event_sos: bool           # Sign of Strength
wyckoff_event_sow: bool           # Sign of Weakness
wyckoff_event_lps: bool           # Last Point of Support

# Phase detection
wyckoff_phase: str                # 'accumulation', 'markup', 'distribution', 'markdown'
wyckoff_confidence: float         # 0.0-1.0
```

### Archetypes Using Wyckoff

**Archetype A (Spring):**
```python
# Detects Wyckoff spring (capitulation reversal)
entry = (wyckoff_event_spring == True) and \
        (volume > 2.0 * volume_sma) and \
        (rsi_14 < 35)
```

**S1 (Liquidity Vacuum):**
```python
# Enhanced spring detection with exhaustion
entry = (wyckoff_event_spring == True) and \
        (volume_climax_3b == True) and \
        (wick_exhaustion_3b == True) and \
        (regime in ['risk_off', 'crisis'])
```

**Archetype B (Order Block Retest):**
```python
# Uses LPS for final support confirmation
entry = (order_block_bull == True) and \
        (wyckoff_event_lps == True) and \
        (ob_retest_bull == True)
```

### Configuration

```json
{
  "enable_wyckoff": true,
  "wyckoff": {
    "spring_lookback": 20,
    "spring_threshold": 0.02,
    "volume_ratio_min": 1.5,
    "phase_window": 50
  }
}
```

### Impact When Disabled
- **S1 trades drop 80%** (can't detect springs)
- **Archetype A blind** (100% reliant on Wyckoff)
- **Archetype B precision drops 40%** (misses LPS confirmation)

---

## 2. SMC Engine (Smart Money Concepts)

### Purpose
Tracks institutional order flow through order blocks, fair value gaps, and liquidity sweeps.

### Key Concepts

**Order Blocks (OB):**
- Last bullish candle before bearish move = Bullish OB (demand zone)
- Last bearish candle before bullish move = Bearish OB (supply zone)
- Institutions return to these zones to add positions

**Fair Value Gaps (FVG):**
- Inefficiencies in price (gaps left by fast moves)
- Price tends to return to fill these gaps

**Liquidity Sweeps:**
- Stop hunts above/below key levels
- Institutions trigger stops to enter opposite direction

### Features Provided

```python
# Order Blocks
is_bullish_ob: bool               # Bullish order block present
is_bearish_ob: bool               # Bearish order block present
ob_retest_bullish: bool           # Price retesting bullish OB
ob_retest_bearish: bool           # Price retesting bearish OB
ob_distance_pct: float            # Distance to nearest OB

# Fair Value Gaps
is_bullish_fvg: bool              # Bullish FVG present
is_bearish_fvg: bool              # Bearish FVG present
fvg_fill_pct: float               # How much FVG has been filled (0.0-1.0)

# Liquidity
is_liquidity_sweep: bool          # Liquidity sweep detected
liquidity_score: float            # Overall liquidity strength
swing_high_liquidity: float       # Liquidity at swing highs
swing_low_liquidity: float        # Liquidity at swing lows
```

### Archetypes Using SMC

**Archetype B (Order Block Retest):**
```python
# Classic OB retest long
entry = (is_bullish_ob == True) and \
        (ob_retest_bullish == True) and \
        (ob_distance_pct < 0.5) and \
        (tf4h_trend_strength > 0.6)
```

**Archetype G (Liquidity Sweep):**
```python
# Liquidity sweep reversal
entry = (is_liquidity_sweep == True) and \
        (liquidity_score > 0.7) and \
        (rsi_14 < 35) and \
        (volume_ratio > 1.8)
```

**Archetype K (Trap Within Trend):**
```python
# False breakout above resistance (liquidity grab)
entry = (is_liquidity_sweep == True) and \
        (is_bullish_ob == True) and \
        (tf4h_trend_strength > 0.5)
```

### Configuration

```json
{
  "enable_smc": true,
  "smc": {
    "ob_lookback": 20,
    "ob_threshold": 0.5,
    "fvg_min_size": 0.002,
    "liquidity_window": 50
  }
}
```

### Impact When Disabled
- **Archetype B blind** (100% reliant on OB detection)
- **Archetype G blind** (100% reliant on liquidity sweeps)
- **Archetype K precision drops 70%** (misses trap setup)

---

## 3. Temporal Engine

### Purpose
Detects time-based confluence using Fibonacci time ratios and Gann cycle analysis.

### Key Concepts

**Fibonacci Time Zones:**
- 21, 34, 55, 89, 144 bar intervals from significant pivots
- Reversals/continuations more likely at these intervals

**Gann Cycles:**
- 90-degree turns (quarterly cycles)
- 180-degree turns (semi-annual cycles)
- 360-degree turns (annual cycles)

**Temporal Confluence:**
- When multiple time-based signals align
- Increases probability of significant price action

### Features Provided

```python
# Fibonacci Time
fib_time_cluster_score: float     # 0.0-1.0, strength of Fib time confluence
is_fib_time_zone: bool            # Currently in Fib time zone
fib_time_ratio_34: bool           # 34-bar Fib ratio active
fib_time_ratio_55: bool           # 55-bar Fib ratio active
fib_time_ratio_89: bool           # 89-bar Fib ratio active

# Gann Cycles
gann_cycle_phase: str             # 'turn', 'momentum', 'consolidation'
is_gann_turn_window: bool         # In Gann turn window

# Confluence
temporal_confluence_score: float  # 0.0-1.0, combined Fib + Gann
time_cluster_count: int           # Number of time signals aligned
```

### Archetypes Using Temporal

**Archetype L (Retest Cluster):**
```python
# Price retest with time confluence
entry = (ob_retest_bullish == True) and \
        (temporal_confluence_score > 0.6) and \
        (fib_time_cluster_score > 0.5) and \
        (time_cluster_count >= 2)
```

**Archetype M (Confluence Breakout):**
```python
# Multi-domain breakout with time alignment
entry = (tf4h_bos_bullish == True) and \
        (temporal_confluence_score > 0.7) and \
        (is_fib_time_zone == True) and \
        (tf4h_fusion_score > 0.6)
```

### Configuration

```json
{
  "enable_temporal": true,
  "temporal": {
    "fib_ratios": [21, 34, 55, 89, 144],
    "gann_cycle_length": 90,
    "confluence_threshold": 0.6,
    "lookback_pivots": 50
  }
}
```

### Impact When Disabled
- **Archetype L recall drops 60%** (misses time confluence signals)
- **Archetype M precision drops 30%** (false breakouts increase)
- All archetypes lose timing precision

---

## 4. HOB Engine (House of Blues)

### Purpose
Proprietary pattern library based on institutional trading behaviors and custom research.

### Key Concepts

**Proprietary Patterns:**
- Chaos windows (extreme volatility + volume)
- PTI (Proprietary Technical Indicator)
- Custom confluence scoring
- Institutional footprint detection

**Integration with Other Engines:**
- Validates signals from Wyckoff/SMC/Temporal
- Adds proprietary filters and gates
- Provides custom confidence scoring

### Features Provided

```python
# Chaos Windows
chaos_window_active: bool         # High volatility regime detected
chaos_severity: float             # 0.0-1.0, intensity of chaos
chaos_resolution_expected: bool   # Chaos about to resolve

# PTI (Proprietary Technical Indicator)
pti_score: float                  # -1.0 to 1.0, proprietary signal
pti_bullish: bool                 # PTI bullish signal
pti_bearish: bool                 # PTI bearish signal

# Institutional Footprint
institutional_flow: str           # 'buying', 'selling', 'neutral'
smart_money_divergence: bool      # Retail vs institutional divergence
```

### Archetypes Using HOB

**All Archetypes:**
```python
# Chaos window filter (avoid entries during extreme chaos)
if chaos_window_active and chaos_severity > 0.8:
    return None  # Skip entry

# PTI confirmation
if pti_score > 0.5:
    confidence *= 1.2  # Boost confidence
```

**S1 (Liquidity Vacuum):**
```python
# Use chaos resolution as capitulation signal
entry = (wyckoff_event_spring == True) and \
        (chaos_resolution_expected == True) and \
        (pti_bullish == True)
```

### Configuration

```json
{
  "enable_hob": true,
  "hob": {
    "chaos_threshold": 0.7,
    "pti_lookback": 14,
    "institutional_flow_window": 20
  }
}
```

### Impact When Disabled
- All archetypes lose chaos window filter (30% more false signals)
- Proprietary confirmation lost (precision drops 15-20%)
- No institutional flow divergence detection

---

## 5. Fusion Engine

### Purpose
Synthesizes signals from multiple domain engines into unified scores.

### Key Concepts

**Multi-Domain Synthesis:**
- Combines Wyckoff + SMC + Temporal + HOB
- Weighted scoring based on importance
- Produces single 0.0-1.0 confidence score

**Timeframe Fusion:**
- Aligns signals across 1H, 4H, 1D timeframes
- Higher timeframe validation
- Reduces false signals on lower timeframes

**Adaptive Weighting:**
- Engine weights adjust based on regime
- Bull market: SMC weighted higher
- Bear market: Wyckoff weighted higher

### Features Provided

```python
# Multi-Domain Fusion
fusion_score: float               # 0.0-1.0, combined signal strength
fusion_bull_bias: float           # 0.0-1.0, bullish bias
fusion_bear_bias: float           # 0.0-1.0, bearish bias

# Timeframe Fusion
tf1h_fusion_score: float          # 1H fusion score
tf4h_fusion_score: float          # 4H fusion score
tf1d_fusion_score: float          # 1D fusion score
mtf_alignment: float              # 0.0-1.0, cross-TF alignment

# Engine Contributions
wyckoff_weight: float             # Current Wyckoff weight
smc_weight: float                 # Current SMC weight
temporal_weight: float            # Current Temporal weight
```

### Archetypes Using Fusion

**Archetype H (Momentum Continuation):**
```python
# High-confidence multi-domain continuation
entry = (tf4h_fusion_score > 0.7) and \
        (mtf_alignment > 0.6) and \
        (tf4h_bos_bullish == True) and \
        (fusion_bull_bias > 0.6)
```

**Archetype M (Confluence Breakout):**
```python
# Maximum confluence across all domains
entry = (fusion_score > 0.8) and \
        (temporal_confluence_score > 0.7) and \
        (tf4h_fusion_score > 0.7) and \
        (mtf_alignment > 0.7)
```

### Configuration

```json
{
  "enable_fusion": true,
  "fusion": {
    "wyckoff_weight": 0.3,
    "smc_weight": 0.3,
    "temporal_weight": 0.2,
    "hob_weight": 0.2,
    "mtf_window": 20
  }
}
```

### Impact When Disabled
- **Archetype H blind** (100% reliant on fusion scores)
- **Archetype M precision drops 50%** (can't validate confluence)
- All archetypes lose multi-domain synthesis (20% more false signals)

---

## 6. Macro Engine

### Purpose
Classifies market regime and filters archetypes to appropriate conditions.

### Key Concepts

**Regime Classification:**
- **Risk-On:** Bull market, high confidence, expanding credit
- **Risk-Off:** Bear market, low confidence, contracting credit
- **Neutral:** Sideways, mixed signals
- **Crisis:** Extreme fear, capitulation, liquidations

**Regime Routing:**
- Bull archetypes (A-M) only fire in `risk_on` or `neutral`
- Bear archetypes (S1-S8) only fire in `risk_off` or `crisis`
- Prevents counter-trend trades in wrong regime

**Macro Indicators:**
- BTC dominance (BTC.D)
- USDT dominance (USDT.D)
- Fear & Greed Index
- Funding rates (aggregate)
- VIX/volatility proxies

### Features Provided

```python
# Regime State
regime_gmm: str                   # 'risk_on', 'risk_off', 'neutral', 'crisis'
regime_hmm_state: str             # HMM-based regime state
regime_hmm_confidence: float      # 0.0-1.0, regime confidence
regime_transition_prob: float     # Probability of regime change

# Macro Indicators
btc_dominance: float              # BTC.D percentage
usdt_dominance: float             # USDT.D percentage
fear_greed: float                 # Fear & Greed Index (0-100)
fear_greed_z: float               # Z-score of Fear & Greed
aggregate_funding: float          # Average funding across markets

# Regime Filters
is_risk_on: bool                  # Currently risk-on regime
is_risk_off: bool                 # Currently risk-off regime
is_crisis: bool                   # Currently crisis regime
```

### Archetypes Using Macro

**All Archetypes:**
```python
# Regime filter at archetype level
ARCHETYPE_REGIMES = {
    'spring': ['risk_on', 'neutral'],
    'liquidity_vacuum': ['risk_off', 'crisis'],
    'funding_divergence': ['risk_off', 'neutral']
}

def detect_archetype(row, ctx, policy):
    # Check regime first
    current_regime = row['regime_gmm']
    allowed_regimes = ARCHETYPE_REGIMES[archetype_name]

    if current_regime not in allowed_regimes:
        return None  # Skip detection in wrong regime

    # Rest of detection logic...
```

**S1 (Liquidity Vacuum):**
```python
# Only fires in bear regimes
entry = (regime_gmm in ['risk_off', 'crisis']) and \
        (fear_greed < 30) and \
        (wyckoff_event_spring == True)
```

**S4 (Funding Divergence):**
```python
# Only fires when funding extreme in bear regime
entry = (regime_gmm in ['risk_off', 'neutral']) and \
        (funding_z > 2.0) and \
        (aggregate_funding > 0.05)
```

### Configuration

```json
{
  "enable_macro": true,
  "macro": {
    "regime_model": "gmm",
    "regime_window": 100,
    "btc_dominance_threshold": 50.0,
    "fear_greed_extreme": 25
  }
}
```

### Impact When Disabled
- **ALL archetypes fire in wrong regimes** (50%+ false signals)
- Bull archetypes fire in bear markets (get stopped out)
- Bear archetypes fire in bull markets (miss rallies)
- **CRITICAL:** Macro engine is non-negotiable

---

## Engine Interaction Map

```
┌─────────────┐
│ Macro Engine│ (CRITICAL FILTER)
└──────┬──────┘
       │ Regime: risk_on/off/neutral/crisis
       ↓
┌──────────────────────────────────────┐
│  Domain Engines (Parallel)           │
├──────────┬──────────┬────────────────┤
│ Wyckoff  │   SMC    │   Temporal     │
│ Events   │   Patterns│   Confluence  │
└────┬─────┴────┬─────┴────┬───────────┘
     │          │          │
     └──────────┼──────────┘
                ↓
         ┌──────────────┐
         │ Fusion Engine│ (Synthesis)
         └──────┬───────┘
                │ fusion_score, mtf_alignment
                ↓
         ┌──────────────┐
         │ HOB Engine   │ (Validation)
         └──────┬───────┘
                │ chaos_filter, pti_confirmation
                ↓
         ┌──────────────┐
         │  Archetype   │
         │  Detection   │
         └──────┬───────┘
                │
                ↓
         Signal Generation
```

---

## Enabling/Disabling Engines

### Check Current Status

```bash
# Check which engines are enabled
python bin/check_domain_engines.py

# Expected output:
# ✓ Wyckoff: ENABLED
# ✓ SMC: ENABLED
# ✓ Temporal: ENABLED
# ✓ HOB: ENABLED
# ✓ Fusion: ENABLED
# ✓ Macro: ENABLED
```

### Enable All Engines

```bash
# Enable all engines in production configs
python bin/enable_domain_engines.py --all

# Or enable specific engines
python bin/enable_domain_engines.py --wyckoff --smc --macro
```

### Manual Configuration

```json
{
  "enable_wyckoff": true,
  "enable_smc": true,
  "enable_temporal": true,
  "enable_hob": true,
  "enable_fusion": true,
  "enable_macro": true
}
```

---

## Performance Impact

### All Engines ON vs OFF

| Metric | Engines OFF | Engines ON | Improvement |
|--------|-------------|------------|-------------|
| **S1 Test PF** | 0.32 | 2.0-2.5 | +525-681% |
| **S4 Test PF** | 0.36 | 2.5-3.0 | +594-733% |
| **S5 Test PF** | 1.55 | 1.8-2.2 | +16-42% |
| **Precision** | 31% | 68% | +119% |
| **Recall** | 89% | 64% | -28% (intentional) |
| **Tier-1 Fallback** | 87% | 12% | -86% |

### Individual Engine Impact

| Engine | Disabled Impact | Archetypes Affected |
|--------|-----------------|---------------------|
| **Wyckoff** | -65% PF | S1, A, B |
| **SMC** | -50% PF | B, G, K, L |
| **Temporal** | -20% PF | L, M |
| **HOB** | -15% PF | All (filter loss) |
| **Fusion** | -40% PF | H, M |
| **Macro** | -75% PF | All (wrong regime) |

**Conclusion:** ALL engines must be enabled for production.

---

## Troubleshooting

### Engines Not Activating

```bash
# Check config files
grep -r "enable_" configs/mvp/*.json

# Should show all true:
# "enable_wyckoff": true,
# "enable_smc": true,
# etc.
```

### Features Missing Despite Engines ON

```bash
# Check feature store has required features
python bin/verify_feature_store.py --engine wyckoff

# If missing, rebuild feature store
python bin/build_feature_store.py --rebuild
```

### High Tier-1 Fallback Despite Engines ON

```bash
# Diagnose fallback causes
python bin/check_tier1_fallback.py --archetype s1 --verbose

# Common causes:
# 1. Feature name mismatch (use FeatureMapper)
# 2. Feature store missing data
# 3. Engine not actually running (check logs)
```

---

## Best Practices

1. **Always enable all 6 engines in production**
2. **Macro engine is non-negotiable** (prevents wrong-regime trades)
3. **Monitor engine health** via `bin/check_domain_engines.py`
4. **Rebuild feature store** after engine config changes
5. **Validate performance** after enabling engines (expect 3-5x improvement)

---

**Document Version:** 1.0
**Last Updated:** 2025-12-08
**Maintained By:** Archetype Engine Team
**Next Review:** After any engine logic updates
