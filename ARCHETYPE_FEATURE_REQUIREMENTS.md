# Archetype Feature Requirements

**Generated:** 2025-12-03
**Purpose:** Complete feature dependency mapping for S1 (Liquidity Vacuum) and S4 (Funding Divergence) archetypes
**Source:** Code inspection of runtime enrichment modules and detection logic

---

## S1: Liquidity Vacuum Reversal

**Archetype Type:** Bear market capitulation reversal (long bias)
**Target Frequency:** 10-15 trades/year
**Edge:** Deep liquidity drain + multi-bar exhaustion = violent bounce
**Implementation:** `engine/strategies/archetypes/bear/liquidity_vacuum_runtime.py`

### Detection Modes

S1 operates in **dual mode** depending on feature availability:

1. **V2 MODE (preferred):** Multi-bar capitulation detection with runtime features
2. **V1 MODE (fallback):** Single-bar detection using base features only

---

### CRITICAL FEATURES (Hard Requirements)

These features are **REQUIRED** for S1 to function. Missing any blocks pattern detection.

#### Base Market Data
| Feature | Type | Source | Usage |
|---------|------|--------|-------|
| `open` | float | OHLCV | Candle body calculation |
| `high` | float | OHLCV | Candle range calculation |
| `low` | float | OHLCV | Candle range calculation |
| `close` | float | OHLCV | Price level, drawdown calculation |

#### Core Indicators (Base Features)
| Feature | Type | Source | Usage |
|---------|------|--------|-------|
| `liquidity_score` | float [0,1] | Feature store | **PRIMARY SIGNAL** - Orderbook depth gauge |
| `volume_zscore` | float | Feature store | Panic selling detection (z > 2.0 = climax) |
| `rsi_14` | float [0,100] | Technical | Oversold condition (< 30 = extreme) |
| `atr_20` | float | Technical | Volatility measure, position sizing |
| `atr_percentile` | float [0,1] | Feature store | Volatility spike detection |

#### Macro Context Features
| Feature | Type | Source | Usage |
|---------|------|--------|-------|
| `VIX_Z` | float | Macro | Fear/volatility indicator (> 1.0 = elevated) |
| `DXY_Z` | float | Macro | Dollar strength (> 0.8 = risk-off) |
| `funding_Z` | float | Funding | Negative funding = short squeeze fuel |

#### Regime Classification
| Feature | Type | Source | Usage |
|---------|------|--------|-------|
| `regime_label` | str | Regime classifier | Fast-fail filter (allowed: risk_off, crisis) |
| `tf4h_external_trend` | str | MTF | Downtrend confirmation (context signal) |

---

### V2 RUNTIME FEATURES (Computed On-Demand)

These features are **calculated by `LiquidityVacuumRuntimeFeatures.enrich_dataframe()`** and are NOT in the feature store.

#### V2 Single-Bar Features
| Feature | Calculation | Purpose | Threshold |
|---------|-------------|---------|-----------|
| `wick_lower_ratio` | `(body_low - low) / (high - low)` | Seller exhaustion signal | > 0.30 (30% wick) |
| `liquidity_vacuum_score` | `1.0 - (liquidity_score / 0.15)` | Inverted liquidity (low = high vacuum) | > 0.5 |
| `volume_panic` | `(volume_zscore - 2.0) / 2.0` | Panic selling intensity [0,1] | > 0.5 |
| `crisis_context` | `0.6*vix_component + 0.4*dxy_component` | Macro stress composite | > 0.5 |
| `liquidity_vacuum_fusion` | Weighted combination (8 components) | **V1 LEGACY SCORE** | > 0.30 (V1 mode) |

#### V2 Multi-Bar Features (FIXES LIQUIDITY PARADOX)
| Feature | Lookback | Calculation | Purpose |
|---------|----------|-------------|---------|
| `liquidity_drain_pct` | 168h (7d) | `(liq - liq_7d_avg) / liq_7d_avg` | **KEY FIX** - Relative drain vs baseline |
| `liquidity_velocity` | 6h | `(liq - liq_6h_ago) / liq_6h_ago` | Rate of drain (active vs quiet) |
| `liquidity_persistence` | Rolling | Consecutive bars with drain < -20% | Multi-bar confirmation counter |
| `capitulation_depth` | 720h (30d) | `(close - rolling_high_30d) / rolling_high_30d` | Drawdown from recent high |
| `crisis_composite` | Multi-factor | Weighted: VIX(30%), Funding(25%), RV(20%), DD(25%) | Enhanced crisis score |
| `volume_climax_last_3b` | 3 bars | `rolling_max(volume_panic, 3)` | Max volume spike in 3-bar window |
| `wick_exhaustion_last_3b` | 3 bars | `rolling_max(wick_lower_ratio, 3)` | Max wick rejection in 3-bar window |

**Why Multi-Bar Logic:**
Real capitulations are MESSY. Signals rarely align on same bar:
- FTX bottom: volume + wick (same bar)
- 2022-06-18: wick bar 1, volume bar 2 (SEPARATED!)
- 2024-08-05: wick bar 1, entry bar 2

---

### S1 V2 DETECTION LOGIC

#### STEP 1: Regime Filter (Optional - Fast Fail)
```python
if use_regime_filter:
    allowed_regimes = ['risk_off', 'crisis']  # Config: allowed_regimes
    drawdown_override_pct = 0.10  # Config: drawdown_override_pct

    # PASS if: (regime in allowed_regimes) OR (drawdown > 10%)
    # This prevents bull market false positives while catching flash crashes
```

**Required Features:**
- `regime_label` (primary) or `crisis_composite` (fallback)
- `capitulation_depth` (for drawdown override)

---

#### STEP 2: Hard Gates (V2 Mode)
```python
# Gate 1: Capitulation Depth
capitulation_depth < -0.20  # Default threshold (20% drawdown)

# Gate 2: Crisis Environment
crisis_composite >= 0.40  # Default threshold (FTX = 0.34, needs relaxing)

# Both must pass to continue
```

**Required Features:**
- `capitulation_depth` (V2 runtime)
- `crisis_composite` (V2 runtime)

---

#### STEP 3: Confluence Logic (V2 Probabilistic Mode)
```python
# Score each condition [0-1]
depth_score = abs(cap_depth) / 0.30      # Deeper = higher score
crisis_score = crisis / 0.50             # More stress = higher
vol_score = vol_climax_3b / 0.70         # Bigger spike = higher
wick_score = wick_exhaust_3b / 0.80      # Deeper wick = higher

# Count binary conditions
conditions_met = sum([
    cap_depth < -0.20,           # Depth threshold
    crisis > 0.35,               # Crisis threshold
    vol_climax_3b > 0.50,        # Volume threshold (RAISED from 0.25)
    wick_exhaust_3b > 0.60       # Wick threshold (RAISED from 0.30)
])

# Require 3-of-4 conditions
confluence_min_conditions = 3  # Config: confluence_min_conditions

# Weighted confluence score
confluence_score = (
    depth_score * 0.20 +
    crisis_score * 0.15 +
    vol_score * 0.08 +
    wick_score * 0.07 +
    # ... liquidity dynamics (25%) ...
    # ... additional signals (25%) ...
)

# Final gate
confluence_score >= 0.65  # Config: confluence_threshold
```

**Required Features (V2 Confluence):**
- `capitulation_depth` (CRITICAL)
- `crisis_composite` (CRITICAL)
- `volume_climax_last_3b` (CRITICAL)
- `wick_exhaustion_last_3b` (CRITICAL)
- `liquidity_drain_pct` (optional enhancement)
- `liquidity_velocity` (optional enhancement)
- `liquidity_persistence` (optional enhancement)
- `funding_Z` (optional enhancement)
- `rsi_14` (optional enhancement)
- `atr_percentile` (optional enhancement)

---

#### STEP 4: V1 Fallback (Binary Mode)
```python
# If V2 features missing, fall back to V1 logic
liquidity_score < 0.20  # Absolute liquidity threshold

# AND (Volume spike OR Wick rejection)
(volume_zscore > 1.5) OR (wick_lower_ratio > 0.28)

# AND fusion score
liquidity_vacuum_fusion > 0.30
```

**Required Features (V1 Fallback):**
- `liquidity_score` (base)
- `volume_zscore` (base)
- `wick_lower_ratio` (runtime)
- `liquidity_vacuum_fusion` (runtime)

---

### S1 Configuration Thresholds

**Production Config:** `configs/s1_v2_production.json`

```json
{
  "liquidity_vacuum": {
    "use_v2_logic": true,
    "use_confluence": true,
    "use_regime_filter": true,

    // Hard Gates
    "capitulation_depth_max": -0.20,     // 20% drawdown minimum
    "crisis_composite_min": 0.35,        // Macro stress minimum

    // Confluence System
    "confluence_min_conditions": 3,      // Require 3-of-4 conditions
    "confluence_threshold": 0.65,        // 65% weighted score
    "volume_climax_3b_min": 0.50,       // Volume threshold (RAISED)
    "wick_exhaustion_3b_min": 0.60,     // Wick threshold (RAISED)

    // Regime Filter
    "allowed_regimes": ["risk_off", "crisis"],
    "drawdown_override_pct": 0.10,      // 10% drawdown = bypass regime

    // Legacy V1 (kept for compatibility)
    "fusion_threshold": 0.30,
    "liquidity_max": 0.20,
    "volume_z_min": 1.5,
    "wick_lower_min": 0.28
  }
}
```

---

### S1 Feature Priority Classification

#### TIER 1: CRITICAL (Missing = Pattern Fails)
1. `close` - Price data
2. `liquidity_score` - Primary signal
3. `capitulation_depth` (V2) - Drawdown filter
4. `crisis_composite` (V2) - Macro context

#### TIER 2: HARD GATES (Missing = Degraded Performance)
5. `volume_climax_last_3b` (V2) - Exhaustion signal
6. `wick_exhaustion_last_3b` (V2) - Exhaustion signal
7. `volume_zscore` (V1 fallback)
8. `wick_lower_ratio` (V1 fallback)

#### TIER 3: CONFLUENCE SIGNALS (Nice-to-Have)
9. `liquidity_drain_pct` - Relative drain
10. `liquidity_velocity` - Drain speed
11. `liquidity_persistence` - Multi-bar confirmation
12. `funding_Z` - Funding reversal
13. `VIX_Z`, `DXY_Z` - Crisis context
14. `rsi_14` - Oversold
15. `atr_percentile` - Volatility spike

#### TIER 4: CONTEXT (Optional)
16. `regime_label` - Regime routing
17. `tf4h_external_trend` - Trend context

---

## S4: Funding Divergence (Short Squeeze)

**Archetype Type:** Bear market short squeeze (long bias)
**Target Frequency:** 6-10 trades/year
**Edge:** Overcrowded shorts + price strength = violent squeeze UP
**Implementation:** `engine/strategies/archetypes/bear/funding_divergence_runtime.py`

### Key Difference from S5
- **S5 (Long Squeeze):** Positive funding → longs overcrowded → cascade DOWN
- **S4 (Funding Divergence):** Negative funding → shorts overcrowded → squeeze UP

---

### CRITICAL FEATURES (Hard Requirements)

#### Base Market Data
| Feature | Type | Source | Usage |
|---------|------|--------|-------|
| `close` | float | OHLCV | Price resilience calculation |
| `funding_rate` | float | Derivatives | **PRIMARY SIGNAL** - Funding rate level |

#### Core Indicators
| Feature | Type | Source | Usage |
|---------|------|--------|-------|
| `funding_Z` | float | Feature store | Z-score of funding (< -1.2 = extreme negative) |
| `liquidity_score` | float [0,1] | Feature store | Thin liquidity = amplified squeeze |
| `volume_zscore` | float | Feature store | Volume quiet detection (< -0.5 = coiled spring) |

---

### S4 RUNTIME FEATURES (Computed On-Demand)

These features are **calculated by `S4RuntimeFeatures.enrich_dataframe()`**.

| Feature | Lookback | Calculation | Purpose |
|---------|----------|-------------|---------|
| `funding_z_negative` | 24h | Rolling z-score of funding_rate | Negative funding extreme detection |
| `price_resilience` | 12h | `(price_pct_change - expected_change) / 0.04` | Price NOT falling despite bearish funding |
| `volume_quiet` | N/A | `volume_zscore < -0.5` | Low volume before squeeze (boolean) |
| `s4_fusion_score` | N/A | Weighted combination (4 components) | Final score [0,1] |

#### Price Resilience Calculation
```python
price_pct_change = (close - close.shift(12)) / close.shift(12)
expected_price_change = funding * 12  # Negative funding should push price down
resilience = price_pct_change - expected_price_change
resilience_norm = clip((resilience + 0.02) / 0.04, 0.0, 1.0)
```

**Interpretation:**
- High resilience (> 0.6) = price STRONGER than funding suggests = divergence signal
- Low resilience (< 0.5) = price following funding expectations = no divergence

---

### S4 DETECTION LOGIC

#### STEP 1: Negative Funding Extreme (REQUIRED)
```python
funding_z < -1.2  # Default threshold (more negative = more shorts)
```

**Required Features:**
- `funding_Z` or `funding_rate` (with runtime calculation)

---

#### STEP 2: Low Liquidity (REQUIRED)
```python
liquidity_score < 0.30  # Default threshold (thin orderbook)
```

**Required Features:**
- `liquidity_score`

---

#### STEP 3: Price Resilience (REQUIRED if runtime features available)
```python
price_resilience >= 0.5  # Default threshold (price showing strength)
```

**Required Features:**
- `price_resilience` (runtime) or skip gate if not available

---

#### STEP 4: Fusion Score
```python
components = {
    "funding_negative": (-funding_z - 1.0) / 2.0,     # 40% weight
    "price_resilience": price_resilience,             # 30% weight
    "volume_quiet": 1.0 if volume_quiet else 0.0,     # 15% weight
    "liquidity_thin": 1.0 - (liquidity / 0.5)         # 15% weight
}

s4_fusion_score = sum(components[k] * weights[k])

# Final gate
s4_fusion_score >= 0.40  # Default threshold (optimized: 0.78)
```

**Required Features:**
- All components above

---

### S4 Configuration Thresholds

**Production Config:** `configs/s4_optimized_oos_test.json`

```json
{
  "funding_divergence": {
    "use_runtime_features": true,
    "funding_lookback": 24,       // Hours for funding z-score
    "price_lookback": 12,         // Hours for resilience calc

    // Hard Gates
    "funding_z_max": -1.976,      // Optimized (more negative than default)
    "resilience_min": 0.555,      // Optimized
    "liquidity_max": 0.348,       // Optimized

    // Fusion
    "fusion_threshold": 0.7824,   // Optimized (stricter than default)
    "final_fusion_gate": 0.7824,

    // Weights
    "weights": {
      "funding_negative": 0.40,
      "price_resilience": 0.30,
      "volume_quiet": 0.15,
      "liquidity_thin": 0.15
    }
  }
}
```

---

### S4 Feature Priority Classification

#### TIER 1: CRITICAL (Missing = Pattern Fails)
1. `funding_rate` or `funding_Z` - Primary signal
2. `close` - Price data for resilience
3. `liquidity_score` - Amplification factor

#### TIER 2: RUNTIME COMPUTED (Missing = Degraded)
4. `funding_z_negative` - Runtime z-score calculation
5. `price_resilience` - Runtime divergence signal
6. `volume_quiet` - Runtime coiled spring detection

#### TIER 3: NICE-TO-HAVE
7. `volume_zscore` - Base volume data
8. `s4_fusion_score` - Final weighted score

---

## COMMON FEATURES (Both S1 & S4)

### Shared Base Features
| Feature | S1 Usage | S4 Usage |
|---------|----------|----------|
| `close` | Drawdown calculation | Price resilience |
| `liquidity_score` | Vacuum detection | Thin orderbook amplification |
| `volume_zscore` | Panic selling | Volume quiet detection |
| `atr_20` | Position sizing, stops | Position sizing, stops |
| `funding_Z` | Optional enhancement | **PRIMARY SIGNAL** |

### Shared Macro Features
| Feature | S1 Usage | S4 Usage |
|---------|----------|----------|
| `VIX_Z` | Crisis context | (not used) |
| `DXY_Z` | Crisis context | (not used) |
| `regime_label` | Regime filtering | Regime routing |

---

## FEATURE STORE REQUIREMENTS

### Must Have in Feature Store
1. `liquidity_score` - Critical for both patterns
2. `volume_zscore` - Critical for S1, helpful for S4
3. `funding_Z` - Critical for S4, helpful for S1
4. `atr_20`, `atr_percentile` - Position sizing
5. `rsi_14` - S1 oversold detection
6. `VIX_Z`, `DXY_Z` - S1 crisis context
7. `regime_label` - Regime routing

### Runtime Computed (NOT in Feature Store)
1. S1 V2 features (12 features)
2. S4 runtime features (4 features)

---

## DEPLOYMENT CHECKLIST

### For S1 Liquidity Vacuum
- [ ] Base OHLCV data available
- [ ] `liquidity_score` in feature store
- [ ] `volume_zscore` in feature store
- [ ] `VIX_Z`, `DXY_Z` macro data
- [ ] `funding_Z` (optional but recommended)
- [ ] Regime classifier enabled (if using regime filter)
- [ ] Runtime enrichment called BEFORE backtest:
  ```python
  from engine.strategies.archetypes.bear.liquidity_vacuum_runtime import apply_liquidity_vacuum_enrichment
  df_enriched = apply_liquidity_vacuum_enrichment(df, lookback=24)
  ```

### For S4 Funding Divergence
- [ ] Base OHLCV data available
- [ ] `funding_rate` or `funding_Z` in feature store
- [ ] `liquidity_score` in feature store
- [ ] `volume_zscore` in feature store (for volume_quiet)
- [ ] Runtime enrichment called BEFORE backtest:
  ```python
  from engine.strategies.archetypes.bear.funding_divergence_runtime import apply_s4_enrichment
  df_enriched = apply_s4_enrichment(df, funding_lookback=24, price_lookback=12)
  ```

---

## MISSING FEATURE HANDLING

### S1 Graceful Degradation
1. If V2 features missing → fall back to V1 logic
2. If macro features missing → use 0.0 defaults
3. If `regime_label` missing → infer from `crisis_composite` or skip filter

### S4 Graceful Degradation
1. If `price_resilience` missing → skip resilience gate
2. If `volume_quiet` missing → default to False
3. If `liquidity_score` missing → use volume z-score as proxy

---

## OPTIMIZATION GUIDANCE

### S1 High-Impact Parameters
1. `confluence_threshold` (0.60-0.70) - Trade frequency vs precision
2. `capitulation_depth_max` (-0.15 to -0.25) - Severity filter
3. `crisis_composite_min` (0.30-0.45) - Macro filter sensitivity
4. `volume_climax_3b_min`, `wick_exhaustion_3b_min` - Exhaustion gates

### S4 High-Impact Parameters
1. `fusion_threshold` (0.40-0.80) - Trade frequency
2. `funding_z_max` (-1.0 to -2.5) - Funding extreme threshold
3. `resilience_min` (0.4-0.7) - Divergence strength
4. `liquidity_max` (0.2-0.4) - Liquidity thinness

---

## KNOWN ISSUES & GOTCHAS

### S1 Liquidity Vacuum
1. **Liquidity Paradox:** Absolute liquidity can be HIGH during capitulations (institutions stepping in). Use `liquidity_drain_pct` (relative) instead.
2. **Multi-bar Signals:** Volume spike and wick rejection rarely align on same bar. Use 3-bar lookback windows.
3. **2023 Bull Market:** Expected 0 trades (by design). Not a bug.
4. **FTX Detection:** `crisis_composite_min=0.40` misses FTX (0.34). Use 0.35.

### S4 Funding Divergence
1. **Funding Sign:** Threshold is NEGATIVE (e.g., -1.2). More negative = more shorts.
2. **Price Resilience:** Requires price_lookback=12h. Shorter = noisy, longer = lag.
3. **OI Data:** Not required (unlike S5). Pattern works without OI.
4. **Regime Sensitivity:** Works best in risk_off/neutral, not crisis (shorts already liquidated).

---

**END OF REQUIREMENTS DOCUMENT**
