# Feature Mapping Reference

**Complete canonical name → feature store mapping**

---

## Overview

The archetype engine expects canonical feature names, but the feature store uses different naming conventions. The `FeatureMapper` class (in `engine/features/feature_mapper.py`) translates between these naming systems.

---

## Critical Mappings

These mappings fix the most common archetype failures:

| Canonical Name | Store Name | Archetype Impact |
|----------------|------------|------------------|
| `funding_z` | `funding_Z` | S4 (Funding Divergence) core signal |
| `oi_change_1h` | `oi_change_1h` | S4, S5 open interest tracking |
| `oi_change_4h` | `oi_change_4h` | S4, S5 open interest tracking |
| `volume_climax_3b` | `volume_climax_last_3b` | S1 (Liquidity Vacuum) exhaustion gate |
| `wick_exhaustion_3b` | `wick_exhaustion_last_3b` | S1 (Liquidity Vacuum) exhaustion gate |
| `btc_d` | `BTC.D` | Macro regime filter |
| `usdt_d` | `USDT.D` | Macro regime filter |
| `order_block_bull` | `is_bullish_ob` | Archetype B (Order Block Retest) |
| `order_block_bear` | `is_bearish_ob` | Bear archetype S2 |
| `tf4h_bos_flag` | `tf4h_bos_bullish` | 4H structure detection |
| `tf4h_bos_bearish` | `tf4h_bos_bearish` | 4H bearish structure |
| `tf1d_trend` | `tf1d_trend_strength` | Daily trend filter |

---

## Funding and OI Features

**Funding Rate:**
- `funding_rate` → `funding_rate`
- `funding_z` → `funding_Z` (note case difference)
- `funding_rate_1h` → `funding_rate_1h`
- `funding_rate_4h` → `funding_rate_4h`
- `funding_rate_ma` → `funding_rate_ma_8h`

**Open Interest:**
- `oi_change_1h` → `oi_change_1h`
- `oi_change_4h` → `oi_change_4h`
- `oi_delta` → `oi_change_1h` (alias)
- `oi_spike` → Calculated from `oi_change_1h > 0.05`

**Usage in Archetypes:**
- S4 (Funding Divergence): `funding_z`, `oi_change_4h`
- S5 (Long Squeeze): `funding_z`, `oi_change_1h`

---

## Volume and Exhaustion Features

**Volume Patterns:**
- `volume` → `volume`
- `volume_sma` → `volume_sma_20`
- `volume_ratio` → `volume / volume_sma_20`
- `volume_spike` → Calculated from `volume_ratio > 2.0`
- `volume_climax_3b` → `volume_climax_last_3b`
- `volume_climax_5b` → `volume_climax_last_5b`

**Wick Exhaustion:**
- `wick_exhaustion_3b` → `wick_exhaustion_last_3b`
- `wick_exhaustion_5b` → `wick_exhaustion_last_5b`
- `wick_ratio_upper` → `wick_ratio_upper`
- `wick_ratio_lower` → `wick_ratio_lower`

**Usage in Archetypes:**
- S1 (Liquidity Vacuum): `volume_climax_3b`, `wick_exhaustion_3b`
- Archetype E (Volume Exhaustion): `volume_climax_5b`

---

## Macro and Regime Features

**Dominance:**
- `btc_d` → `BTC.D`
- `usdt_d` → `USDT.D`
- `eth_d` → `ETH.D`
- `btc_dominance` → `BTC.D` (alias)

**Regime:**
- `regime` → `regime_gmm`
- `regime_state` → `regime_hmm_state`
- `regime_confidence` → `regime_hmm_confidence`
- `is_risk_on` → Calculated from `regime_gmm == 'risk_on'`
- `is_risk_off` → Calculated from `regime_gmm == 'risk_off'`

**Fear and Greed:**
- `fear_greed_index` → `fear_greed`
- `fear_greed_z` → `fear_greed_z`

**Usage in Archetypes:**
- All archetypes: Regime filtering via `ARCHETYPE_REGIMES` map
- S1: Requires `risk_off` or `crisis` regime
- S4: Requires `risk_off` or `neutral` regime

---

## SMC (Smart Money Concepts) Features

**Order Blocks:**
- `order_block_bull` → `is_bullish_ob`
- `order_block_bear` → `is_bearish_ob`
- `ob_retest_bull` → `ob_retest_bullish`
- `ob_retest_bear` → `ob_retest_bearish`
- `ob_distance` → `ob_distance_pct`

**Fair Value Gaps:**
- `fvg_bull` → `is_bullish_fvg`
- `fvg_bear` → `is_bearish_fvg`
- `fvg_filled` → `fvg_fill_pct > 0.5`

**Liquidity:**
- `liquidity_sweep` → `is_liquidity_sweep`
- `liquidity_score` → `liquidity_score`
- `liquidity_high` → `swing_high_liquidity`
- `liquidity_low` → `swing_low_liquidity`

**Usage in Archetypes:**
- Archetype B (Order Block Retest): `order_block_bull`, `ob_retest_bull`
- Archetype G (Liquidity Sweep): `liquidity_sweep`, `liquidity_score`

---

## Multi-Timeframe Features

**4H Timeframe:**
- `tf4h_fusion_score` → `tf4h_fusion_score`
- `tf4h_trend_strength` → `tf4h_trend_strength`
- `tf4h_bos_flag` → `tf4h_bos_bullish`
- `tf4h_bos_bearish` → `tf4h_bos_bearish`
- `tf4h_choch` → `tf4h_choch_detected`

**1D Timeframe:**
- `tf1d_trend` → `tf1d_trend_strength`
- `tf1d_fusion` → `tf1d_fusion_score`
- `tf1d_regime` → `tf1d_regime_state`

**1H Timeframe:**
- `tf1h_bos` → `tf1h_bos_bullish`
- `tf1h_choch` → `tf1h_choch_detected`

**Usage in Archetypes:**
- Archetype H (Momentum Continuation): `tf4h_trend_strength`, `tf4h_bos_flag`
- Archetype M (Confluence Breakout): `tf4h_fusion_score`, `tf1d_trend`

---

## Wyckoff Features

**Structural Events:**
- `wyckoff_spring` → `wyckoff_event_spring`
- `wyckoff_utad` → `wyckoff_event_utad`
- `wyckoff_sos` → `wyckoff_event_sos` (Sign of Strength)
- `wyckoff_sow` → `wyckoff_event_sow` (Sign of Weakness)
- `wyckoff_lps` → `wyckoff_event_lps` (Last Point of Support)

**Phase Detection:**
- `wyckoff_phase` → `wyckoff_phase`
- `wyckoff_accumulation` → `wyckoff_phase == 'accumulation'`
- `wyckoff_distribution` → `wyckoff_phase == 'distribution'`

**Usage in Archetypes:**
- Archetype A (Spring): `wyckoff_spring`
- S1 (Liquidity Vacuum): `wyckoff_spring` (capitulation reversal)

---

## Temporal Features

**Fibonacci Time:**
- `fib_time_cluster` → `fib_time_cluster_score`
- `fib_time_zone` → `is_fib_time_zone`
- `fib_ratio_34` → `fib_time_ratio_34`
- `fib_ratio_55` → `fib_time_ratio_55`

**Gann Cycles:**
- `gann_cycle` → `gann_cycle_phase`
- `gann_turn_window` → `is_gann_turn_window`

**Temporal Confluence:**
- `temporal_confluence` → `temporal_confluence_score`
- `time_cluster` → `time_cluster_count`

**Usage in Archetypes:**
- Archetype L (Retest Cluster): `temporal_confluence`, `fib_time_cluster`
- Archetype M (Confluence Breakout): `temporal_confluence`

---

## Technical Indicators

**RSI:**
- `rsi` → `rsi_14`
- `rsi_9` → `rsi_9`
- `rsi_oversold` → `rsi_14 < 30`
- `rsi_overbought` → `rsi_14 > 70`

**MACD:**
- `macd` → `macd`
- `macd_signal` → `macd_signal`
- `macd_hist` → `macd_hist`
- `macd_cross` → Calculated from `macd > macd_signal`

**Bollinger Bands:**
- `bb_upper` → `bb_upper`
- `bb_lower` → `bb_lower`
- `bb_width` → `bb_width`
- `bb_pct` → `(close - bb_lower) / (bb_upper - bb_lower)`

**ADX:**
- `adx` → `adx_14`
- `adx_strong` → `adx_14 > 25`

**ATR:**
- `atr` → `atr_20`
- `atr_ratio` → `atr_20 / close`

---

## Fallback and Tier-1 Features

When domain engines are disabled or features missing, archetypes fall back to Tier-1 simple logic:

**Tier-1 Required Features:**
- `rsi_14` (always available)
- `volume` (always available)
- `volume_sma_20` (always available)
- `close` (always available)
- `atr_20` (always available)

**Tier-1 Logic:**
```python
# Simple reversal logic (ALL archetypes use this as fallback)
entry = (rsi_14 < 30) and (volume > 2.0 * volume_sma_20)
exit = (rsi_14 > 70)
```

**Goal:** Minimize Tier-1 fallback to < 30% of trades

---

## FeatureMapper Usage

**Python API:**

```python
from engine.features.feature_mapper import FeatureMapper

# Initialize mapper
mapper = FeatureMapper()

# Get feature value (handles name translation)
funding_z = mapper.get('funding_z', row)  # Returns row['funding_Z']

# Get multiple features
features = mapper.get_many(['funding_z', 'oi_change_1h', 'volume_climax_3b'], row)

# Check if feature exists
if mapper.has('funding_z', row):
    value = mapper.get('funding_z', row)

# Get all aliases for a canonical name
aliases = mapper.get_aliases('funding_z')  # Returns ['funding_Z', 'funding_zscore']
```

**In Archetype Logic:**

```python
def detect_s4_funding_divergence(row, ctx, policy):
    """S4: Funding Divergence with FeatureMapper."""
    from engine.features.feature_mapper import FeatureMapper
    mapper = FeatureMapper()

    # OLD: Hardcoded lookup (fails if name doesn't match)
    # funding_z = row.get('funding_z', 0.0)  # ❌ Returns 0.0

    # NEW: Canonical mapping
    funding_z = mapper.get('funding_z', row)  # ✓ Returns row['funding_Z']
    oi_change = mapper.get('oi_change_4h', row)

    # Rest of logic...
```

---

## Validation

**Check Feature Coverage:**

```bash
# Run feature coverage check
python bin/check_domain_engines.py --verbose

# Expected output:
# Feature coverage: 98.1% (461/470 features)
# Missing features: ['custom_feature_x', 'experimental_y']
```

**Check Tier-1 Fallback Rate:**

```bash
# Check how often archetypes use fallback
python bin/check_tier1_fallback.py --archetype s4

# Expected output:
# S4 Tier-1 fallback: 12.3% (23/187 trades)
# Target: < 30%
# Status: ✓ PASS
```

**Verify Mapping Correctness:**

```bash
# Test all mappings
python bin/test_feature_mapper.py

# Expected output:
# ✓ All 470 canonical names mapped
# ✓ No orphaned aliases
# ✓ No circular mappings
# ✓ All critical features accessible
```

---

## Adding New Mappings

**1. Update FeatureMapper:**

```python
# In engine/features/feature_mapper.py

class FeatureMapper:
    def _initialize_mappings(self):
        # Add new mapping
        self._add_mapping('new_canonical_name', 'actual_store_name')

        # Add aliases
        self._add_alias('new_canonical_name', 'legacy_name')
        self._add_alias('new_canonical_name', 'alternate_name')
```

**2. Test Mapping:**

```bash
python bin/test_feature_mapper.py --feature new_canonical_name
```

**3. Update This Document:**

Add new mapping to appropriate section above.

**4. Re-run Validation:**

```bash
./bin/validate_archetype_engine.sh --full
```

---

## Common Issues

**Issue: Feature not found despite being in store**

**Diagnosis:**
```bash
# Check actual store column names
python bin/verify_feature_store.py --columns

# Check if mapper has translation
python -c "from engine.features.feature_mapper import FeatureMapper; print(FeatureMapper().get_store_name('your_feature'))"
```

**Solution:**
Add mapping to `FeatureMapper._initialize_mappings()`

---

**Issue: Tier-1 fallback rate too high (> 30%)**

**Diagnosis:**
```bash
# Check which features are missing
python bin/check_tier1_fallback.py --archetype s1 --verbose
```

**Solution:**
1. Check feature store has required features
2. Add missing mappings to FeatureMapper
3. Ensure domain engines are enabled

---

**Issue: Wrong feature value returned**

**Diagnosis:**
```bash
# Check feature value in store
python bin/verify_feature_store.py --feature funding_Z --sample

# Check mapper translation
python -c "from engine.features.feature_mapper import FeatureMapper; print(FeatureMapper().get_store_name('funding_z'))"
```

**Solution:**
Verify mapping is correct. If store name changed, update FeatureMapper.

---

## Appendix: Complete Mapping Table

**Generated from FeatureMapper:**

```bash
# Export all mappings to CSV
python bin/export_feature_mappings.py --output feature_mappings.csv

# View in terminal
python bin/export_feature_mappings.py --format table | less
```

---

**Document Version:** 1.0
**Last Updated:** 2025-12-08
**Maintained By:** Feature Store Team
**Next Review:** After feature store schema updates
