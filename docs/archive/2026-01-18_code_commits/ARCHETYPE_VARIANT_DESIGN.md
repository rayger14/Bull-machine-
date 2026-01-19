# Archetype Variant Design - Simplified Intelligence Layers

**Mission**: Create testable variants of each archetype that isolate core logic from noise, enabling meta-model feature attribution.

**Date**: 2025-12-08
**Status**: Design Complete - Ready for Implementation

---

## Design Philosophy

Each archetype has a **special forces capability** - a unique market signature it detects better than any other pattern. Variants strip away the noise to test if that core capability alone is profitable.

**Variant Hierarchy**:
1. **Core** - Bare minimum archetype logic (the "special forces" capability)
2. **Core + Time** - Core + temporal confluence (Fibonacci time, session awareness)
3. **Core + Macro** - Core + regime filters (BTC.D, DXY, funding trends)
4. **Full** - Current production config (all filters, all noise)

**Key Principle**: Each variant uses the SAME runtime feature enrichment class, just different threshold flags. No code duplication.

---

## 1. LIQUIDITY VACUUM (S1) - Capitulation Reversal Specialist

### Core Logic
**Special Forces Capability**: Detecting panic exhaustion at exact capitulation bottoms when liquidity evaporates.

**Canonical Pattern**:
- Price reverses into vacuum zone (liquidity drain)
- Volume climax (capitulation selling)
- Deep lower wick (sellers exhausted, buyers stepping in)

**Historical Wins**:
- 2022-06-18: LUNA capitulation → -70% → violent 25% bounce in 24h
- 2022-11-09: FTX collapse → liquidity vacuum → explosive reversal
- 2022-05-12: LUNA death spiral → extreme capitulation → sharp bounce

### Variants

#### a) LV_core - Pure Vacuum Logic
**Features Used**:
- `liquidity_vacuum_score` (low liquidity = high vacuum)
- `volume_panic` (volume z-score > 2.0)
- `wick_lower_ratio` (> 0.30 = deep rejection)

**Thresholds**:
```json
{
  "liquidity_vacuum_core": {
    "direction": "long",
    "archetype_weight": 1.0,
    "fusion_threshold": 0.35,
    "liquidity_max": 0.15,
    "volume_z_min": 2.0,
    "wick_lower_min": 0.30,
    "max_risk_pct": 0.02,
    "atr_stop_mult": 2.5,
    "cooldown_bars": 12,
    "regime_filter": false,
    "macro_filter": false,
    "time_filter": false
  }
}
```

**Expected Behavior**: 5-8 trades/year, PF 1.5-2.0 (if vacuum logic is sound)

#### b) LV_core_plus_time - Core + Temporal Confluence
**Additional Features**:
- `fib_time_cluster` (Fibonacci time confluence)
- Session awareness (Asian/London/NY session detection)

**Thresholds**: Same as core, plus:
```json
{
  "fib_time_cluster_min": 0.6,
  "preferred_sessions": ["london_close", "ny_open"]
}
```

**Expected Behavior**: 3-5 trades/year, PF 2.0-2.5 (time filtering improves hit rate)

#### c) LV_core_plus_macro - Core + Regime Filters
**Additional Features**:
- `crisis_composite` (VIX + funding extreme + drawdown depth)
- `btc_dominance_z` (BTC.D z-score)
- `dxy_z` (DXY strength)

**Thresholds**: Same as core, plus:
```json
{
  "crisis_composite_min": 0.5,
  "btc_d_z_min": 0.5,
  "dxy_z_min": 0.3,
  "regime_filter": true,
  "allowed_regimes": ["risk_off", "crisis"]
}
```

**Expected Behavior**: 2-4 trades/year, PF 2.5-3.0 (crisis filter = extreme quality)

#### d) LV_full - Production Config
**Current Implementation**: All filters enabled (V2 multi-bar capitulation encoding)

**Config Path**: `configs/system_s1_production.json` (if exists) or extract from MVP configs

---

## 2. FUNDING DIVERGENCE (S4) - Short Squeeze Specialist

### Core Logic
**Special Forces Capability**: Detecting overcrowded shorts about to be liquidated during violent rallies.

**Canonical Pattern**:
- Extremely negative funding rate (shorts paying longs)
- Price resilience despite bearish sentiment (strength signal)
- Low liquidity (thin bids = violent cascade up)

**Historical Wins**:
- 2022-08-15: Funding -0.15% → +18% rally in 48h (violent short squeeze)
- 2023-01-14: Negative funding + price strength → 12% rally
- 2022-03-28: Overcrowded shorts → forced covering cascade

### Variants

#### a) FD_core - Pure Funding Divergence
**Features Used**:
- `funding_z_negative` (< -1.5 sigma = extreme shorts)
- `price_resilience` (price strong despite bearish funding)
- `liquidity_score` (< 0.20 = thin orderbook)

**Thresholds**:
```json
{
  "funding_divergence_core": {
    "direction": "long",
    "archetype_weight": 1.0,
    "fusion_threshold": 0.40,
    "funding_z_max": -1.5,
    "resilience_min": 0.6,
    "liquidity_max": 0.20,
    "max_risk_pct": 0.02,
    "atr_stop_mult": 3.0,
    "cooldown_bars": 8,
    "regime_filter": false,
    "macro_filter": false
  }
}
```

**Expected Behavior**: 6-10 trades/year, PF 1.8-2.2

#### b) FD_core_plus_time - Core + Temporal
**Additional Features**:
- Volume quiet detection (calm before storm)
- Multi-bar funding persistence (not just 1-bar spike)

**Thresholds**: Same as core, plus:
```json
{
  "volume_quiet": true,
  "funding_persistence_bars": 3
}
```

**Expected Behavior**: 4-7 trades/year, PF 2.0-2.5

#### c) FD_core_plus_macro - Core + Regime
**Additional Features**:
- BTC dominance (BTC.D decline = alt weakness = more shorts)
- Macro stress (VIX spike = extreme positioning)

**Thresholds**: Same as core, plus:
```json
{
  "btc_d_z_max": -0.5,
  "vix_z_min": 1.0,
  "regime_filter": true,
  "allowed_regimes": ["risk_off", "neutral"]
}
```

**Expected Behavior**: 3-5 trades/year, PF 2.5-3.0

---

## 3. LONG SQUEEZE (S5) - Overleveraged Longs Specialist

### Core Logic
**Special Forces Capability**: Detecting overleveraged longs about to cascade down during bull market corrections.

**Canonical Pattern**:
- Extremely high positive funding rate (longs paying shorts)
- Rising open interest (many new longs entering = liquidation fuel)
- RSI overbought (> 70)
- Low liquidity (cascading liquidations)

**Historical Wins**:
- Bull market corrections where funding > +2 sigma
- Late-cycle leverage flushes
- Altcoin capitulation events

### Variants

#### a) LS_core - Pure Long Squeeze
**Features Used**:
- `funding_z_score` (> 2.0 sigma = extreme longs)
- `rsi_overbought` (> 70)
- `liquidity_score` (< 0.25 = cascade risk)

**Thresholds**:
```json
{
  "long_squeeze_core": {
    "direction": "short",
    "archetype_weight": 1.0,
    "fusion_threshold": 0.45,
    "funding_z_min": 2.0,
    "rsi_min": 70,
    "liquidity_max": 0.25,
    "max_risk_pct": 0.015,
    "atr_stop_mult": 3.0,
    "cooldown_bars": 8,
    "regime_filter": false,
    "oi_filter": false
  }
}
```

**Expected Behavior**: 7-12 trades/year, PF 1.3-1.5

#### b) LS_core_plus_time - Core + OI Dynamics
**Additional Features**:
- `oi_change` (rising OI > 10% = more liquidations)
- Volume quiet before explosion

**Thresholds**: Same as core, plus:
```json
{
  "oi_change_min": 0.10,
  "volume_quiet": true,
  "oi_filter": true
}
```

**Expected Behavior**: 5-9 trades/year, PF 1.5-1.8

#### c) LS_core_plus_macro - Core + Regime
**Additional Features**:
- Regime routing (higher weight in risk_on)
- BTC dominance (BTC.D high = alt risk)

**Thresholds**: Same as core, plus:
```json
{
  "btc_d_z_min": 0.8,
  "regime_filter": true,
  "allowed_regimes": ["risk_on", "neutral"]
}
```

**Expected Behavior**: 4-7 trades/year, PF 1.8-2.2

---

## Implementation Architecture

### Shared Components

All variants use the SAME runtime enrichment classes:
- `LiquidityVacuumRuntimeFeatures` (S1)
- `S4RuntimeFeatures` (Funding Divergence)
- `S5RuntimeFeatures` (Long Squeeze)

**No code duplication** - only config differences.

### Config Structure

Each variant is a standalone config in `/configs/variants/`:

```
configs/variants/
├── liquidity_vacuum_core.json
├── liquidity_vacuum_core_plus_time.json
├── liquidity_vacuum_core_plus_macro.json
├── liquidity_vacuum_full.json
├── funding_divergence_core.json
├── funding_divergence_core_plus_time.json
├── funding_divergence_core_plus_macro.json
├── funding_divergence_full.json
├── long_squeeze_core.json
├── long_squeeze_core_plus_time.json
├── long_squeeze_core_plus_macro.json
├── long_squeeze_full.json
└── README.md
```

### Feature Flags

Each variant config includes:
```json
{
  "variant_name": "liquidity_vacuum_core",
  "variant_tier": "core",
  "parent_archetype": "S1",
  "enable_regime_filter": false,
  "enable_macro_filter": false,
  "enable_time_filter": false,
  "enable_oi_filter": false,
  "log_variant_features": true
}
```

### Meta-Model Integration

Each variant logs its signals as separate features:
```python
# In backtest results
features_logged = {
  'S1_core_signal': 0.45,
  'S1_core_plus_time_signal': 0.62,
  'S1_core_plus_macro_signal': 0.78,
  'S1_full_signal': 0.55,
}
```

Meta-model learns:
- Which variant fires most often
- Which variant has highest PF
- Whether macro filters help or hurt
- Optimal feature combinations

---

## Testing Strategy

### Phase 1: Individual Variant Testing
Run each variant through quant suite on 2020-2024 data:
```bash
python bin/run_quant_suite.py --config configs/variants/liquidity_vacuum_core.json
python bin/run_quant_suite.py --config configs/variants/liquidity_vacuum_core_plus_time.json
python bin/run_quant_suite.py --config configs/variants/liquidity_vacuum_core_plus_macro.json
```

**Success Criteria**:
- Core variant trades > 0 (core logic works)
- Core PF > 1.0 (core logic is profitable)
- Each tier adds value (PF increases or trade count decreases with quality improvement)

### Phase 2: Variant Comparison
Compare metrics across tiers:
- Trade count (should decrease as filters tighten)
- Profit Factor (should increase as quality improves)
- Win Rate (should increase with better filtering)
- Drawdown (should decrease with safer entries)

### Phase 3: Meta-Model Training
Feed all variant signals to meta-fusion engine:
- Input: 12 variant signals (3 archetypes × 4 variants each)
- Output: Combined confidence score
- Learn: Which variant combinations work best in which regimes

---

## Variant Summary Table

| Archetype | Core Capability | Core Features | Core PF Target | Time Tier PF | Macro Tier PF | Full PF |
|-----------|----------------|---------------|----------------|--------------|---------------|---------|
| **S1 (Liquidity Vacuum)** | Capitulation exhaustion detection | vacuum + volume + wick | 1.5-2.0 | 2.0-2.5 | 2.5-3.0 | TBD |
| **S4 (Funding Divergence)** | Short squeeze anticipation | funding_z + resilience + liquidity | 1.8-2.2 | 2.0-2.5 | 2.5-3.0 | TBD |
| **S5 (Long Squeeze)** | Overleveraged long cascade | funding_z + rsi + liquidity | 1.3-1.5 | 1.5-1.8 | 1.8-2.2 | TBD |

**Total Variants**: 12 (3 archetypes × 4 variants each)

---

## Expected Deliverables

1. **12 Config Files** - One for each variant in `/configs/variants/`
2. **Variant README** - Usage guide for running and comparing variants
3. **Test Results** - Quant suite results for each variant (2020-2024 backtest)
4. **Feature Attribution Report** - Which features add value vs noise
5. **Meta-Model Training Data** - Variant signals logged for ensemble learning

---

## Next Steps

1. **Generate Configs** - Create all 12 variant configs based on templates above
2. **Test Core Variants** - Validate that core logic alone is profitable
3. **Run Comparison Suite** - Compare all variants to find optimal feature combinations
4. **Train Meta-Model** - Feed variant signals to ensemble learner
5. **Production Deployment** - Deploy best-performing variant or ensemble

---

## References

- **Runtime Features**:
  - `/engine/strategies/archetypes/bear/liquidity_vacuum_runtime.py`
  - `/engine/strategies/archetypes/bear/funding_divergence_runtime.py`
  - `/engine/strategies/archetypes/bear/long_squeeze_runtime.py`
- **Archetype Logic**: `/engine/archetypes/logic_v2_adapter.py`
- **MVP Configs**: `/configs/mvp/mvp_bear_market_v1.json`
- **Feature Registry**: `/engine/features/registry.py`
- **Name Mapping**: `/ARCHETYPE_NAME_MAPPING_REFERENCE.md`

---

**Status**: Design Complete - Ready for Config Generation
**Author**: System Architect (Claude Code)
**Date**: 2025-12-08
