================================================================================
LOGIC TREE AUDIT REPORT
================================================================================

EXECUTIVE SUMMARY
--------------------------------------------------------------------------------
GREEN (Wired & Used):     50 features
YELLOW (Unwired):         18 features
RED (Ghost/Idea-Only):    45 features


GREEN - WIRED & USED FEATURES
================================================================================

BOS_CHOCH (4 features):
----------------------------------------
  ✓ atr
  ✓ boms_disp
  ✓ pti_score
  ✓ pti_trap_type

FAILED_RALLY (2 features):
----------------------------------------
  ✓ atr_percentile
  ✓ vol_z

FUNDING_DIVERGENCE (3 features):
----------------------------------------
  ✓ atr_percentile
  ✓ rsi
  ✓ vol_z

LIQUIDITY_VACUUM (1 features):
----------------------------------------
  ✓ boms_strength

LONG_SQUEEZE (2 features):
----------------------------------------
  ✓ fvg_present_1h
  ✓ rsi

ORDER_BLOCK_RETEST (3 features):
----------------------------------------
  ✓ boms_strength
  ✓ bos_bullish
  ✓ wyckoff_score

S1 (22 features):
----------------------------------------
  ✓ DXY_Z
  ✓ VIX_Z
  ✓ atr_percentile
  ✓ capitulation_depth
  ✓ crisis_composite
  ✓ funding_Z
  ✓ hob_demand_zone
  ✓ liquidity_drain_pct
  ✓ liquidity_persistence
  ✓ liquidity_velocity
  ✓ regime_label
  ✓ rsi_14
  ✓ smc_score
  ✓ tf4h_external_trend
  ✓ volume_climax_last_3b
  ✓ volume_zscore
  ✓ wick_exhaustion_last_3b
  ✓ wick_lower_ratio
  ✓ wyckoff_ps
  ✓ wyckoff_pti_confluence
  ✓ wyckoff_spring_a
  ✓ wyckoff_spring_b

S2 (12 features):
----------------------------------------
  ✓ close
  ✓ high
  ✓ low
  ✓ ob_retest_flag
  ✓ open
  ✓ rsi_14
  ✓ rsi_bearish_div
  ✓ tf1h_ob_high
  ✓ tf4h_external_trend
  ✓ volume_fade_flag
  ✓ volume_zscore
  ✓ wick_upper_ratio

S4 (10 features):
----------------------------------------
  ✓ funding_Z
  ✓ price_resilience
  ✓ smc_score
  ✓ volume_quiet
  ✓ wyckoff_phase_abc
  ✓ wyckoff_pti_confluence
  ✓ wyckoff_sow
  ✓ wyckoff_spring_a
  ✓ wyckoff_spring_b
  ✓ wyckoff_utad

S5 (9 features):
----------------------------------------
  ✓ funding_Z
  ✓ oi_change_24h
  ✓ rsi_14
  ✓ smc_score
  ✓ wyckoff_phase_abc
  ✓ wyckoff_pti_confluence
  ✓ wyckoff_pti_score
  ✓ wyckoff_sow
  ✓ wyckoff_utad

TRAP_WITHIN_TREND (4 features):
----------------------------------------
  ✓ atr
  ✓ boms_disp
  ✓ fusion_score
  ✓ fvg_present_4h


YELLOW - UNWIRED FEATURES (Exist but Not Used)
================================================================================

Found 18 unwired features:

  ⚠ Range
  ⚠ Validation
  ⚠ adx_14
  ⚠ atr_20
  ⚠ float64
  ⚠ int64
  ⚠ liquidity_score
  ⚠ momentum_score
  ⚠ parameter_bounds
  ⚠ tf1d_trend_direction
  ⚠ tf1h_bos_bearish
  ⚠ tf1h_bos_bullish
  ⚠ tf1h_fvg_bear
  ⚠ tf1h_fvg_bull
  ⚠ tf4h_bos_bearish
  ⚠ tf4h_bos_bullish
  ⚠ tf4h_fusion_score
  ⚠ tf4h_trend_strength


RED - GHOST FEATURES (Referenced but Don't Exist)
================================================================================

Found 45 ghost features:

  ✗ adaptive_fusion
  ✗ bos_choch_reversal
  ✗ buy_threshold
  ✗ capitulation_depth_max
  ✗ capitulation_depth_score
  ✗ confluence_threshold
  ✗ confluence_weights
  ✗ crisis_composite_min
  ✗ crisis_environment
  ✗ crisis_fuse
  ✗ drawdown_override_pct
  ✗ ema_alpha
  ✗ entry_threshold_confidence
  ✗ failed_rally
  ✗ final_fusion_gate
  ✗ final_gate_delta
  ✗ funding_reversal
  ✗ funding_z_min
  ✗ fusion_adapt
  ✗ fusion_threshold
  ✗ liquidity_drain_severity
  ✗ liquidity_max
  ✗ liquidity_persistence_score
  ✗ liquidity_vacuum
  ✗ liquidity_velocity_score
  ✗ long_squeeze
  ✗ lookback_hours
  ✗ monthly_share_cap
  ✗ order_block_retest
  ✗ regime_classifier
  ✗ regime_override
  ✗ risk_off
  ✗ risk_on
  ✗ rsi_min
  ✗ signal_ttl_bars
  ✗ system_name
  ✗ trap_within_trend
  ✗ vol_z_max
  ✗ volatility_spike
  ✗ volume_z_min
  ✗ wick_exhaustion_3b
  ✗ wick_exhaustion_3b_min
  ✗ wick_lower_min
  ✗ wick_ratio_min
  ✗ wick_trap_moneytaur


DOMAIN ENGINES
================================================================================

WYCKOFF (10 methods):
----------------------------------------
  • __init__
  • _analyze_price_structure
  • _basic_phase_logic
  • _calculate_volume_quality
  • _get_expected_next_events
  • analyze
  • crt_smr_check
  • detect_wyckoff_events
  • detect_wyckoff_phase
  • get_wyckoff_sequence_context

SMC (6 methods):
----------------------------------------
  • __init__
  • _empty_signal
  • _generate_unified_signal
  • _identify_entry_zones
  • analyze
  • analyze_smc

TEMPORAL (10 methods):
----------------------------------------
  • __init__
  • _compute_emotional_cycle_score
  • _compute_fib_cluster_score
  • _compute_gann_cycle_score
  • _compute_volatility_cycle_score
  • adjust_fusion_weight
  • compute_bars_since_wyckoff_events
  • compute_temporal_confluence
  • compute_temporal_features_batch
  • get_component_scores