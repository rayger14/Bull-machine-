# Feature Store Final Verification Report

**Generated:** 2025-12-11 16:15:51

**Feature Store:** `data/features_mtf/BTC_1H_2022-01-01_to_2024-12-31.parquet`

## Executive Summary

- **Total columns:** 202
- **Total rows:** 26236
- **Date range:** 2022-01-01 19:00:00+00:00 to 2024-12-31 00:00:00+00:00
- **Overall quality:** **POOR** (19.8% good features)

## Critical Feature Status

### V2 Features (OI Change Spikes)

❌ **Status:** INCOMPLETE (0/4 found)

**Missing:**
- oi_change_spike_3h
- oi_change_spike_6h
- oi_change_spike_12h
- oi_change_spike_24h
### Wyckoff Features

⚠ **Status:** INCOMPLETE (6/15 found)

**Working:** 5/6

**Quality:**

| Feature | Non-Null % | Signal Count | Signal % | Quality |
|---------|------------|--------------|----------|----------|
| ✓ wyckoff_spring_a | 100.00% | 8 | 0.03% | RARE |
| ❌ wyckoff_spring_b | 100.00% | 0 | 0.00% | BROKEN |
| ✓ wyckoff_sos | 100.00% | 125 | 0.48% | GOOD |
| ✓ wyckoff_sow | 100.00% | 119 | 0.45% | GOOD |
| ✓ wyckoff_ar | 100.00% | 2043 | 7.79% | GOOD |
| ✓ wyckoff_st | 100.00% | 16184 | 61.69% | FREQUENT |

### SMC Features

⚠ **Status:** INCOMPLETE (3/7 found)

**Working:** 3/3

**Quality:**

| Feature | Non-Null % | Signal Count | Signal % | Quality |
|---------|------------|--------------|----------|----------|
| ✓ smc_bos | 100.00% | 282 | 1.07% | GOOD |
| ✓ smc_choch | 100.00% | 90 | 0.34% | GOOD |
| ✓ smc_liquidity_sweep | 100.00% | 187 | 0.71% | GOOD |

## Problematic Features

### ❌ BROKEN Features (26)

| Feature | Non-Null % | Constant | Issue |
|---------|------------|----------|-------|
| k2_threshold_delta | 100.00% | True | Constant value |
| mtf_alignment_ok | 100.00% | True | Constant value |
| oi_change_24h | 32.92% | False | Low coverage (32.9%) |
| oi_change_pct_24h | 32.92% | False | Low coverage (32.9%) |
| oi_z | 32.93% | False | Low coverage (32.9%) |
| tf1d_boms_detected | 100.00% | True | Constant value |
| tf1d_boms_direction | 100.00% | True | Constant value |
| tf1h_fvg_high | 46.47% | False | Low coverage (46.5%) |
| tf1h_fvg_low | 49.04% | False | Low coverage (49.0%) |
| tf1h_kelly_hint | 100.00% | True | Constant value |
| tf1h_pti_trap_type | 100.00% | True | Constant value |
| tf4h_boms_direction | 100.00% | True | Constant value |
| tf4h_choch_flag | 100.00% | True | Constant value |
| tf4h_fvg_present | 100.00% | True | Constant value |
| tf4h_range_breakout_strength | 100.00% | True | Constant value |
| tf4h_structure_alignment | 100.00% | True | Constant value |
| funding | 33.39% | False | Low coverage (33.4%) |
| oi | 33.02% | False | Low coverage (33.0%) |
| rv_20d | 33.39% | False | Low coverage (33.4%) |
| rv_60d | 33.39% | False | Low coverage (33.4%) |
| macro_oil_trend | 100.00% | True | Constant value |
| wyckoff_spring_b | 100.00% | True | Constant value |
| wyckoff_spring_b_confidence | 100.00% | True | Constant value |
| wyckoff_pti_confluence | 100.00% | True | Constant value |
| temporal_confluence | 100.00% | True | Constant value |
| fib_time_target | 30.78% | False | Low coverage (30.8%) |

### ⚠ ALWAYS_FIRES Features (108)

These features are always True (may indicate broken logic):

- BTC.D_Z
- DXY_Z
- RV_20
- RV_30
- RV_60
- RV_7
- TOTAL2_RET
- TOTAL_RET
- USDT.D_Z
- VIX_Z
- YC_SPREAD
- YC_Z
- atr_14
- atr_20
- close
- funding_Z
- funding_rate
- high
- k2_fusion_score
- k2_score_delta
- low
- macro_correlation_score
- mtf_conflict_score
- open
- rsi_14
- sma_20
- tf1d_boms_strength
- tf1d_frvp_poc
- tf1d_frvp_position
- tf1d_frvp_va_high
- tf1d_frvp_va_low
- tf1d_fusion_score
- tf1d_pti_score
- tf1d_range_confidence
- tf1d_range_direction
- tf1d_range_outcome
- tf1d_wyckoff_phase
- tf1d_wyckoff_score
- tf1h_fakeout_direction
- tf1h_fakeout_intensity
- tf1h_frvp_distance_to_poc
- tf1h_frvp_poc
- tf1h_frvp_position
- tf1h_frvp_va_high
- tf1h_frvp_va_low
- tf1h_fusion_score
- tf1h_kelly_atr_pct
- tf1h_kelly_volatility_ratio
- tf1h_pti_confidence
- tf1h_pti_score
- tf4h_boms_displacement
- tf4h_conflict_score
- tf4h_external_trend
- tf4h_fusion_score
- tf4h_internal_phase
- tf4h_range_outcome
- tf4h_squiggle_confidence
- tf4h_squiggle_direction
- tf4h_squiggle_stage
- volume
- volume_zscore
- USDT.D
- BTC.D
- macro_vix_level
- macro_dxy_trend
- macro_yields_trend
- macro_regime
- macro_correlation
- adaptive_threshold
- volume_ratio
- ob_confidence
- volume_z
- range_position
- wyckoff_sc_confidence
- wyckoff_bc_confidence
- wyckoff_ar_confidence
- wyckoff_as_confidence
- wyckoff_st_confidence
- wyckoff_sos_confidence
- wyckoff_sow_confidence
- wyckoff_spring_a_confidence
- wyckoff_ut_confidence
- wyckoff_utad_confidence
- wyckoff_lps_confidence
- wyckoff_lpsy_confidence
- wyckoff_phase_abc
- wyckoff_sequence_position
- liquidity_score
- wick_lower_ratio
- liquidity_vacuum_score
- volume_panic
- crisis_context
- liquidity_vacuum_fusion
- liquidity_drain_pct
- liquidity_velocity
- liquidity_persistence
- capitulation_depth
- crisis_composite
- volume_climax_last_3b
- wick_exhaustion_last_3b
- resilience
- smc_score
- hob_imbalance
- wyckoff_pti_score
- temporal_support_cluster
- temporal_resistance_cluster
- fib_time_score
- volatility_cycle

## Date Coverage

- **Start date:** 2022-01-01 19:00:00+00:00
- **End date:** 2024-12-31 00:00:00+00:00
- **Total timespan:** 1094 days
- **Expected frequency:** 1H
- **Gaps detected:** 2 (largest: 1 days 00:00:00)

⚠ **Warning:** Date coverage has gaps. This may affect backtest accuracy.

## Production Readiness Assessment

❌ **V2 Features:** NOT production ready (missing features or incomplete data)

✅ **Critical Systems:** Production ready (Wyckoff & SMC >80% working)

### ⚠ **Overall: NEEDS IMPROVEMENT**

Feature store has issues that should be addressed before production deployment.

## Next Steps

1. **Fix broken features:** Investigate and repair features with constant values or low coverage
3. **Add missing V2 features:** Regenerate feature store with all OI change spike features
4. **Fill date gaps:** Ensure continuous hourly data coverage

---

*Report generated by verify_feature_store_quality.py on 2025-12-11 16:15:51*
