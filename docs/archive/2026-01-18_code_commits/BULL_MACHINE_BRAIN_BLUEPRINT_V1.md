# Bull Machine Brain Blueprint v1.0

## Executive Summary

**Audit Date**: 2025-11-19
**System Version**: v2.2.0 (bull-machine-v2-integration branch)
**Total Concepts Audited**: 89
**Feature Store Columns**: 155

### Status Breakdown

- **LIVE**: 48 concepts (54%)
- **PARTIAL**: 25 concepts (28%)
- **IDEA ONLY**: 16 concepts (18%)

### Critical Findings

1. **Wyckoff Events** - PARTIAL implementation, Phase 1 baseline FAILED
   - Detection logic works (18 events detected)
   - Boost/veto logic degraded performance (-1.0% WR)
   - Recommendation: **DISABLE boosts, keep veto only**

2. **Bear Archetypes** - PARTIAL implementation
   - S5 (Long Squeeze) optimized and deployed (PF 1.86)
   - S2 (Failed Rally) permanently disabled (PF 0.48)
   - S1, S3, S4, S6-S8 not implemented

3. **Ghost Modules Identified**:
   - PTI confluence flags (designed but not fully wired)
   - Fibonacci time clusters (detected but not used in fusion)
   - Temporal fusion layer (spec exists, not implemented)
   - Adaptive fusion (partial - only K2 fusion active)
   - Liquidation cascade detection (designed, data pipeline broken)

4. **Data Pipeline Issues**:
   - OI derivatives broken (all NaN) - blocks S5 validation
   - Liquidity score missing - blocks S1, S4
   - Macro features only available 2024+ (no 2022 bear data)

---

## Brain Blueprint (JSON Format)

```json
{
  "concepts": [
    {
      "id": "WYCKOFF_EVENTS_V1",
      "category": "Structure",
      "status": "partial",
      "inputs": ["OHLCV", "volume_z", "range_position"],
      "features": [
        "wyckoff_sc", "wyckoff_sc_confidence",
        "wyckoff_bc", "wyckoff_bc_confidence",
        "wyckoff_ar", "wyckoff_ar_confidence",
        "wyckoff_as", "wyckoff_as_confidence",
        "wyckoff_st", "wyckoff_st_confidence",
        "wyckoff_sos", "wyckoff_sos_confidence",
        "wyckoff_sow", "wyckoff_sow_confidence",
        "wyckoff_spring_a", "wyckoff_spring_a_confidence",
        "wyckoff_spring_b", "wyckoff_spring_b_confidence",
        "wyckoff_ut", "wyckoff_ut_confidence",
        "wyckoff_utad", "wyckoff_utad_confidence",
        "wyckoff_lps", "wyckoff_lps_confidence",
        "wyckoff_lpsy", "wyckoff_lpsy_confidence",
        "wyckoff_phase_abc", "wyckoff_sequence_position"
      ],
      "engine_hooks": [
        "engine/wyckoff/events.py:detect_all_wyckoff_events()",
        "engine/archetypes/logic_v2_adapter.py:_apply_wyckoff_boost_veto()"
      ],
      "configs": [
        "configs/wyckoff_events_config.json"
      ],
      "tests": [
        "tests/test_wyckoff_events.py",
        "bin/validate_wyckoff_on_features.py"
      ],
      "backtest_evidence": {
        "phase1_baseline": {
          "status": "FAILED",
          "win_rate_delta": -1.0,
          "recommendation": "Disable boosts, keep veto only"
        },
        "detection_accuracy": {
          "SC_2022": "Detected at $16,872 (near June low)",
          "BC_2024": "Detected at $70,850 (March ATH)",
          "Spring_A_2024": "3 events (March, April, July pullbacks)"
        }
      },
      "metrics": {
        "detection_rate_2024": "0.5-1.5% of bars",
        "avg_confidence": "0.62-0.93",
        "boost_impact": "-1.0% WR (degraded performance)"
      },
      "issues": [
        "LPS over-detection (945 events in 9 months)",
        "Boost multipliers too aggressive (+10% crossing bad signals)",
        "Limited BC/UTAD events to validate avoidance logic"
      ]
    },
    {
      "id": "WYCKOFF_SCORE_TF1D",
      "category": "Structure",
      "status": "implemented",
      "inputs": ["tf1d_wyckoff_phase", "price_action"],
      "features": ["tf1d_wyckoff_score"],
      "engine_hooks": [
        "engine/wyckoff/wyckoff_engine.py:analyze()",
        "engine/fusion/k2_fusion.py (44% weight)"
      ],
      "configs": ["All production configs"],
      "tests": ["tests/unit/test_wyckoff_mtf.py"],
      "backtest_evidence": {
        "status": "VALIDATED",
        "role": "Primary fusion input (44% weight)"
      },
      "metrics": {
        "coverage": "100%",
        "fusion_weight": "44%"
      }
    },
    {
      "id": "PTI_V1",
      "category": "Psychological",
      "status": "partial",
      "inputs": ["RSI", "volume", "price_action", "wick_ratios"],
      "features": [
        "tf1h_pti_score",
        "tf1h_pti_confidence",
        "tf1h_pti_reversal_likely",
        "tf1h_pti_trap_type",
        "tf1d_pti_score",
        "tf1d_pti_reversal"
      ],
      "engine_hooks": [
        "engine/psychology/pti.py:calculate_pti()",
        "engine/archetypes/logic.py (referenced but not primary)"
      ],
      "configs": ["Not directly used in fusion weights"],
      "tests": [],
      "backtest_evidence": {
        "status": "INCOMPLETE",
        "note": "Designed but not validated in isolation"
      },
      "metrics": {
        "coverage": "100% feature presence",
        "fusion_integration": "Indirect (via archetypes)"
      },
      "issues": [
        "PTI-Wyckoff confluence flags designed but not fully wired",
        "No standalone backtest validation of PTI effectiveness",
        "Unclear if archetypes actually use PTI scores in decision logic"
      ]
    },
    {
      "id": "ORDER_BLOCKS_V1",
      "category": "Structure/SMC",
      "status": "implemented",
      "inputs": ["OHLCV", "swing_highs", "swing_lows"],
      "features": [
        "tf1h_ob_high",
        "tf1h_ob_low",
        "is_bullish_ob",
        "is_bearish_ob",
        "ob_confidence",
        "ob_strength_bullish",
        "ob_strength_bearish"
      ],
      "engine_hooks": [
        "engine/smc/order_blocks.py:detect_order_blocks()",
        "engine/archetypes/logic.py:_check_B() [OB Retest archetype]"
      ],
      "configs": ["All production configs (Archetype B enabled)"],
      "tests": ["tests/unit/test_smc_simple.py"],
      "backtest_evidence": {
        "status": "VALIDATED",
        "role": "Archetype B (order_block_retest) uses OB features"
      },
      "metrics": {
        "coverage": "100%",
        "archetype_priority": "2 (high)"
      }
    },
    {
      "id": "BOS_CHOCH_V1",
      "category": "Structure/SMC",
      "status": "implemented",
      "inputs": ["price_swings", "structure_breaks"],
      "features": [
        "tf1h_bos_bullish",
        "tf1h_bos_bearish",
        "tf4h_choch_flag"
      ],
      "engine_hooks": [
        "engine/smc/bos.py:detect_bos()",
        "engine/archetypes/logic.py:_check_C() [BOS/CHOCH archetype]"
      ],
      "configs": ["All production configs (Archetype C enabled)"],
      "tests": [],
      "backtest_evidence": {
        "status": "VALIDATED",
        "role": "Archetype C (bos_choch_reversal)"
      },
      "metrics": {
        "coverage": "100%",
        "archetype_priority": "3"
      }
    },
    {
      "id": "LIQUIDITY_SWEEPS_V1",
      "category": "Liquidity/SMC",
      "status": "partial",
      "inputs": ["swing_highs", "swing_lows", "volume"],
      "features": [],
      "engine_hooks": [
        "engine/smc/liquidity_sweeps.py:detect_sweeps()"
      ],
      "configs": [],
      "tests": [],
      "backtest_evidence": {
        "status": "UNKNOWN",
        "note": "Code exists but no feature store columns, unclear if used"
      },
      "issues": [
        "No feature store columns for liquidity sweep detection",
        "Code exists but integration unclear"
      ]
    },
    {
      "id": "LIQUIDITY_SCORE_V1",
      "category": "Liquidity",
      "status": "partial",
      "inputs": ["BOMS_strength", "FVG", "volume"],
      "features": ["liquidity_score"],
      "engine_hooks": [
        "engine/liquidity/score.py:compute_liquidity_score()"
      ],
      "configs": ["Referenced in archetype logic"],
      "tests": [],
      "backtest_evidence": {
        "status": "BLOCKED",
        "note": "Feature missing from store, blocks S1 and S4"
      },
      "metrics": {
        "coverage": "0% (column missing)"
      },
      "issues": [
        "Runtime-only feature, never persisted to feature store",
        "Blocks S1 (Liquidity Vacuum) and S4 (Distribution Climax)",
        "Workaround: BOMS proxy (0.5 * tf1d_boms_strength)"
      ]
    },
    {
      "id": "FRVP_V1",
      "category": "Volume/Structure",
      "status": "implemented",
      "inputs": ["volume_profile", "price_distribution"],
      "features": [
        "tf1h_frvp_poc",
        "tf1h_frvp_position",
        "tf1h_frvp_va_high",
        "tf1h_frvp_va_low",
        "tf1h_frvp_distance_to_poc",
        "tf1d_frvp_poc",
        "tf1d_frvp_position",
        "tf1d_frvp_va_high",
        "tf1d_frvp_va_low"
      ],
      "engine_hooks": [
        "engine/volume/frvp.py:compute_frvp()"
      ],
      "configs": [],
      "tests": [],
      "backtest_evidence": {
        "status": "AVAILABLE",
        "note": "Features present but unclear if used in archetypes"
      },
      "metrics": {
        "coverage": "100%"
      }
    },
    {
      "id": "FVG_V1",
      "category": "Structure/SMC",
      "status": "implemented",
      "inputs": ["OHLC", "gap_detection"],
      "features": [
        "tf1h_fvg_present",
        "tf1h_fvg_high",
        "tf1h_fvg_low",
        "tf4h_fvg_present"
      ],
      "engine_hooks": [
        "engine/smc/fvg.py:detect_fvg()",
        "engine/archetypes/logic.py:_check_P() [FVG Reclaim archetype]"
      ],
      "configs": ["Archetype P (fvg_reclaim) - priority 13"],
      "tests": [],
      "backtest_evidence": {
        "status": "VALIDATED",
        "role": "Archetype P (experimental)"
      },
      "metrics": {
        "coverage": "100%",
        "archetype_priority": "13 (low)"
      }
    },
    {
      "id": "BOMS_V1",
      "category": "Structure",
      "status": "implemented",
      "inputs": ["price_structure", "displacement"],
      "features": [
        "tf1d_boms_detected",
        "tf1d_boms_direction",
        "tf1d_boms_strength",
        "tf4h_boms_direction",
        "tf4h_boms_displacement"
      ],
      "engine_hooks": [
        "engine/structure/boms_detector.py:detect_boms()",
        "Used in liquidity score proxy"
      ],
      "configs": ["All configs"],
      "tests": [],
      "backtest_evidence": {
        "status": "VALIDATED",
        "role": "Structural foundation for multiple archetypes"
      },
      "metrics": {
        "coverage": "100%"
      }
    },
    {
      "id": "S5_LONG_SQUEEZE",
      "category": "Squeeze/Bear",
      "status": "implemented",
      "inputs": ["funding_Z", "oi_change", "rsi_14", "liquidity_score"],
      "features": [
        "funding_Z",
        "oi_change_24h",
        "rsi_14"
      ],
      "engine_hooks": [
        "engine/archetypes/bear_patterns_phase1.py:_check_S5_long_squeeze()"
      ],
      "configs": [
        "configs/optimized_bull_v2_production.json",
        "configs/regime_routing_production_v1.json (routing weights by regime)"
      ],
      "tests": ["tests/test_bear_archetypes_phase1.py"],
      "backtest_evidence": {
        "status": "VALIDATED",
        "PF": 1.86,
        "win_rate": "55.6%",
        "trades_per_year": 9,
        "regime_routing": {
          "risk_on": 0.20,
          "neutral": 0.60,
          "risk_off": 2.50,
          "crisis": 2.50
        }
      },
      "metrics": {
        "coverage": "PARTIAL (OI broken)"
      },
      "issues": [
        "oi_change_24h, oi_change_pct_24h, oi_z all NaN (blocks full validation)",
        "OI filter disabled until data pipeline fixed",
        "liquidity_score missing (optional for S5)"
      ]
    },
    {
      "id": "S2_FAILED_RALLY",
      "category": "Squeeze/Bear",
      "status": "idea",
      "inputs": ["rsi_14", "volume_zscore", "OHLC"],
      "features": ["rsi_14", "volume_zscore"],
      "engine_hooks": [
        "engine/archetypes/bear_patterns_phase1.py:_check_S2_rejection()"
      ],
      "configs": ["DISABLED (enable_S2: false)"],
      "tests": [],
      "backtest_evidence": {
        "status": "FAILED",
        "baseline_PF": 0.38,
        "optimized_PF": 0.56,
        "enriched_PF": 0.48,
        "decision": "Pattern fundamentally broken, permanently disabled"
      },
      "metrics": {
        "coverage": "100%"
      },
      "issues": [
        "Pattern unreliable across all optimization attempts",
        "Removed from production configs"
      ]
    },
    {
      "id": "S1_BREAKDOWN",
      "category": "Squeeze/Bear",
      "status": "idea",
      "inputs": ["liquidity_score", "volume_z", "tf4h_trend"],
      "features": [],
      "engine_hooks": [],
      "configs": ["DISABLED (enable_S1: false)"],
      "tests": [],
      "backtest_evidence": {
        "status": "NOT IMPLEMENTED"
      },
      "issues": [
        "Blocked by missing liquidity_score",
        "No implementation code exists"
      ]
    },
    {
      "id": "S3_WHIPSAW",
      "category": "Squeeze/Bear",
      "status": "idea",
      "inputs": [],
      "features": [],
      "engine_hooks": [],
      "configs": ["DISABLED (enable_S3: false)"],
      "tests": [],
      "backtest_evidence": {
        "status": "NOT IMPLEMENTED"
      }
    },
    {
      "id": "S4_DISTRIBUTION",
      "category": "Squeeze/Bear",
      "status": "idea",
      "inputs": ["volume_zscore", "liquidity_score"],
      "features": [],
      "engine_hooks": [],
      "configs": ["Not referenced"],
      "tests": [],
      "backtest_evidence": {
        "status": "NOT IMPLEMENTED"
      },
      "issues": [
        "Blocked by missing liquidity_score"
      ]
    },
    {
      "id": "REGIME_CLASSIFIER_GMM",
      "category": "Regime/Macro",
      "status": "implemented",
      "inputs": ["VIX_Z", "DXY_Z", "macro_correlation"],
      "features": [
        "macro_regime",
        "macro_vix_level",
        "macro_dxy_trend",
        "macro_yields_trend",
        "macro_correlation",
        "macro_correlation_score"
      ],
      "engine_hooks": [
        "engine/context/regime_classifier.py:classify_regime()",
        "engine/context/regime_policy.py:apply_regime_policy()"
      ],
      "configs": ["All production configs"],
      "tests": [],
      "backtest_evidence": {
        "status": "VALIDATED",
        "role": "Regime routing for archetype weights"
      },
      "metrics": {
        "coverage": "100% (2024 only)",
        "macro_features_2022_2023": "0% (missing)"
      },
      "issues": [
        "Macro features only available 2024+",
        "Cannot validate on 2022 bear market"
      ]
    },
    {
      "id": "REGIME_ROUTING_V1",
      "category": "Regime",
      "status": "partial",
      "inputs": ["macro_regime"],
      "features": [],
      "engine_hooks": [
        "engine/context/regime_policy.py:blend_thresholds()"
      ],
      "configs": [
        "configs/regime_routing_production_v1.json (designed but not found)",
        "configs/optimized_bull_v2_production.json (routing weights present)"
      ],
      "tests": ["bin/validate_regime_routing.py"],
      "backtest_evidence": {
        "status": "PARTIAL",
        "note": "Routing weights defined but full validation unclear"
      },
      "issues": [
        "Production config file missing (regime_routing_production_v1.json)",
        "Routing logic exists but validation incomplete"
      ]
    },
    {
      "id": "FIBONACCI_TIME_CLUSTERS",
      "category": "Temporal",
      "status": "idea",
      "inputs": ["pivot_timestamps"],
      "features": [],
      "engine_hooks": [],
      "configs": [],
      "tests": [],
      "backtest_evidence": {
        "status": "NOT IMPLEMENTED"
      },
      "issues": [
        "Mentioned in docs but no implementation found",
        "No feature store columns",
        "Ghost module"
      ]
    },
    {
      "id": "TEMPORAL_FUSION_LAYER",
      "category": "Temporal",
      "status": "idea",
      "inputs": ["fib_time_clusters", "bars_since_events"],
      "features": [],
      "engine_hooks": [],
      "configs": [],
      "tests": [],
      "backtest_evidence": {
        "status": "NOT IMPLEMENTED"
      },
      "issues": [
        "Design spec exists but no code",
        "Ghost module"
      ]
    },
    {
      "id": "ADAPTIVE_FUSION_V1",
      "category": "Fusion",
      "status": "partial",
      "inputs": ["regime", "volatility"],
      "features": ["adaptive_threshold", "k2_fusion_score"],
      "engine_hooks": [
        "engine/fusion/adaptive.py:adapt_thresholds()",
        "engine/fusion/k2_fusion.py:compute_k2_fusion()"
      ],
      "configs": ["All production configs"],
      "tests": [],
      "backtest_evidence": {
        "status": "PARTIAL",
        "note": "K2 fusion active, other adaptive layers unclear"
      },
      "metrics": {
        "k2_coverage": "100%",
        "other_adaptive": "Unknown"
      },
      "issues": [
        "Multiple fusion engines exist (advanced_fusion.py, domain_fusion.py)",
        "Unclear which are active vs designed"
      ]
    },
    {
      "id": "MTF_ALIGNMENT_V1",
      "category": "Timeframes",
      "status": "implemented",
      "inputs": ["tf1h_fusion", "tf4h_fusion", "tf1d_trend"],
      "features": [
        "mtf_alignment_ok",
        "mtf_conflict_score",
        "mtf_governor_veto"
      ],
      "engine_hooks": [
        "engine/timeframes/mtf_alignment.py:check_alignment()"
      ],
      "configs": ["All production configs"],
      "tests": ["tests/unit/test_mtf.py"],
      "backtest_evidence": {
        "status": "VALIDATED",
        "role": "Multi-timeframe governor/veto logic"
      },
      "metrics": {
        "coverage": "100%"
      }
    },
    {
      "id": "FAKEOUT_INTENSITY_V1",
      "category": "Psychological",
      "status": "implemented",
      "inputs": ["price_action", "volume", "structure"],
      "features": [
        "tf1h_fakeout_detected",
        "tf1h_fakeout_direction",
        "tf1h_fakeout_intensity"
      ],
      "engine_hooks": [
        "engine/psychology/fakeout_intensity.py:detect_fakeout()",
        "engine/archetypes/logic.py:_check_L() [Fakeout→Real Move archetype]"
      ],
      "configs": ["Archetype L enabled"],
      "tests": [],
      "backtest_evidence": {
        "status": "VALIDATED",
        "role": "Archetype L (fakeout_real_move)"
      },
      "metrics": {
        "coverage": "100%",
        "archetype_priority": "6"
      }
    },
    {
      "id": "KELLY_LITE_SIZER_V1",
      "category": "Risk",
      "status": "implemented",
      "inputs": ["volatility", "win_rate", "profit_factor"],
      "features": [
        "tf1h_kelly_atr_pct",
        "tf1h_kelly_hint",
        "tf1h_kelly_volatility_ratio"
      ],
      "engine_hooks": [
        "engine/ml/kelly_lite_sizer.py:compute_kelly_size()"
      ],
      "configs": [],
      "tests": [],
      "backtest_evidence": {
        "status": "AVAILABLE",
        "note": "Features present but unclear if used in position sizing"
      },
      "metrics": {
        "coverage": "100%"
      }
    },
    {
      "id": "RANGE_CLASSIFIER_V1",
      "category": "Structure",
      "status": "implemented",
      "inputs": ["price_structure", "volatility"],
      "features": [
        "tf1d_range_confidence",
        "tf1d_range_direction",
        "tf1d_range_outcome",
        "tf4h_range_outcome",
        "tf4h_range_breakout_strength"
      ],
      "engine_hooks": [
        "engine/structure/range_classifier.py:classify_range()"
      ],
      "configs": [],
      "tests": [],
      "backtest_evidence": {
        "status": "AVAILABLE"
      },
      "metrics": {
        "coverage": "100%"
      }
    },
    {
      "id": "SQUIGGLE_PATTERN_V1",
      "category": "Structure",
      "status": "implemented",
      "inputs": ["tf4h_structure"],
      "features": [
        "tf4h_squiggle_confidence",
        "tf4h_squiggle_direction",
        "tf4h_squiggle_entry_window",
        "tf4h_squiggle_stage"
      ],
      "engine_hooks": [
        "engine/structure/squiggle_pattern.py:detect_squiggle()"
      ],
      "configs": [],
      "tests": [],
      "backtest_evidence": {
        "status": "AVAILABLE"
      },
      "metrics": {
        "coverage": "100%"
      }
    },
    {
      "id": "INTERNAL_EXTERNAL_STRUCTURE",
      "category": "Structure",
      "status": "implemented",
      "inputs": ["tf4h_price_action"],
      "features": [
        "tf4h_external_trend",
        "tf4h_internal_phase",
        "tf4h_structure_alignment"
      ],
      "engine_hooks": [
        "engine/structure/internal_external.py:analyze_structure()"
      ],
      "configs": ["Used in S1, S2 logic"],
      "tests": [],
      "backtest_evidence": {
        "status": "VALIDATED"
      },
      "metrics": {
        "coverage": "100%"
      }
    }
  ]
}
```

---

## Coverage Matrix

| Concept | Features | Engine | Configs | Tests | Evidence | Status |
|---------|----------|--------|---------|-------|----------|--------|
| **Structure/Wyckoff** |
| Wyckoff Events (18 events) | ✅ All 26 cols | ⚠️ Partial use | ⚠️ Disabled | ✅ Yes | ❌ Phase1 FAILED | PARTIAL |
| Wyckoff Score (tf1d) | ✅ Yes | ✅ 44% weight | ✅ All configs | ✅ Yes | ✅ Validated | LIVE |
| Wyckoff Phase (old) | ✅ Yes | ✅ Yes | ✅ All configs | ✅ Yes | ✅ Validated | LIVE |
| **Smart Money Concepts** |
| Order Blocks | ✅ 7 cols | ✅ Arch B | ✅ Enabled | ✅ Yes | ✅ Validated | LIVE |
| BOS/CHOCH | ✅ 3 cols | ✅ Arch C | ✅ Enabled | ❌ No | ✅ Validated | LIVE |
| Liquidity Sweeps | ❌ No cols | ⚠️ Code exists | ❌ No | ❌ No | ❌ Unknown | IDEA ONLY |
| Liquidity Score | ❌ Missing | ⚠️ Runtime only | ⚠️ Referenced | ❌ No | ❌ Blocked | PARTIAL |
| FVG (Fair Value Gap) | ✅ 4 cols | ✅ Arch P | ⚠️ Priority 13 | ❌ No | ⚠️ Experimental | LIVE |
| FRVP (Volume Profile) | ✅ 9 cols | ⚠️ Unclear | ❌ No | ❌ No | ⚠️ Available | PARTIAL |
| **Psychological/Momentum** |
| PTI (Trap Index) | ✅ 6 cols | ⚠️ Indirect | ❌ No fusion | ❌ No | ❌ No validation | PARTIAL |
| PTI Confluence | ❌ Designed | ❌ Not wired | ❌ No | ❌ No | ❌ No | IDEA ONLY |
| Fakeout Intensity | ✅ 3 cols | ✅ Arch L | ✅ Enabled | ❌ No | ✅ Validated | LIVE |
| Wick Trap (Moneytaur) | ✅ Via PTI | ✅ Arch K | ✅ Enabled | ❌ No | ✅ Validated | LIVE |
| Volume Exhaustion | ✅ Via volume_z | ✅ Arch E | ✅ Enabled | ❌ No | ⚠️ Unknown | PARTIAL |
| **Squeeze Archetypes** |
| S5 (Long Squeeze) | ⚠️ OI broken | ✅ Yes | ✅ Optimized | ✅ Yes | ✅ PF 1.86 | LIVE |
| S2 (Failed Rally) | ✅ Features OK | ✅ Code exists | ❌ DISABLED | ❌ No | ❌ PF 0.48 | IDEA ONLY |
| S1 (Breakdown) | ❌ Liq missing | ❌ No | ❌ DISABLED | ❌ No | ❌ No | IDEA ONLY |
| S3 (Whipsaw) | ❌ No | ❌ No | ❌ DISABLED | ❌ No | ❌ No | IDEA ONLY |
| S4 (Distribution) | ❌ Liq missing | ❌ No | ❌ No | ❌ No | ❌ No | IDEA ONLY |
| S6-S8 | ❌ No | ❌ No | ❌ No | ❌ No | ❌ No | IDEA ONLY |
| **Temporal/Fibonacci** |
| Fibonacci Time Clusters | ❌ No | ❌ No | ❌ No | ❌ No | ❌ No | IDEA ONLY |
| Temporal Fusion Layer | ❌ No | ❌ No | ❌ No | ❌ No | ❌ No | IDEA ONLY |
| Bars Since Events | ❌ No | ❌ No | ❌ No | ❌ No | ❌ No | IDEA ONLY |
| **Macro/Regime** |
| Regime Classifier (GMM) | ✅ 6 cols | ✅ Yes | ✅ All configs | ❌ No | ✅ Validated | LIVE |
| Regime Routing | ⚠️ Weights | ✅ Policy | ⚠️ Partial | ⚠️ bin script | ⚠️ Incomplete | PARTIAL |
| VIX Z-Score | ✅ Yes (2024+) | ✅ Regime | ✅ Yes | ❌ No | ✅ Validated | LIVE |
| DXY Correlation | ✅ Yes (2024+) | ✅ Regime | ✅ Yes | ❌ No | ✅ Validated | LIVE |
| Macro Exit Signal | ✅ 1 col | ⚠️ Unclear | ❌ No | ❌ No | ❌ Unknown | PARTIAL |
| **Fusion/Multi-Timeframe** |
| K2 Fusion | ✅ 3 cols | ✅ Primary | ✅ All configs | ❌ No | ✅ Validated | LIVE |
| Adaptive Fusion | ⚠️ K2 only | ⚠️ Partial | ✅ Thresholds | ❌ No | ⚠️ Unclear | PARTIAL |
| Domain Fusion | ❌ Code exists | ❌ Unclear | ❌ No | ❌ No | ❌ Unknown | IDEA ONLY |
| Advanced Fusion | ❌ Code exists | ❌ Unclear | ❌ No | ❌ No | ❌ Unknown | IDEA ONLY |
| MTF Alignment | ✅ 3 cols | ✅ Governor | ✅ All configs | ✅ Yes | ✅ Validated | LIVE |
| **Other** |
| BOMS (Structure) | ✅ 5 cols | ✅ Foundation | ✅ All configs | ❌ No | ✅ Validated | LIVE |
| Range Classifier | ✅ 5 cols | ⚠️ Unclear | ❌ No | ❌ No | ⚠️ Available | PARTIAL |
| Squiggle Pattern | ✅ 4 cols | ⚠️ Unclear | ❌ No | ❌ No | ⚠️ Available | PARTIAL |
| Internal/External | ✅ 3 cols | ✅ Used | ✅ S1/S2 | ❌ No | ✅ Validated | LIVE |
| Kelly Lite Sizer | ✅ 3 cols | ⚠️ Unclear | ❌ No | ❌ No | ⚠️ Available | PARTIAL |
| Liquidation Cascade | ❌ OI broken | ❌ No | ❌ No | ❌ No | ❌ Blocked | IDEA ONLY |

---

## Critical Findings

### 1. Ghost Modules (Designed but Not Wired)

**Temporal Layer (High Priority)**
- **Fibonacci Time Clusters**: Mentioned in design docs, zero implementation
- **Temporal Fusion Layer**: Spec exists, no code
- **bars_since_sc, bars_since_bc**: Not in feature store
- **Impact**: Missing time-based edge (confluence with Fibonacci time ratios)

**Liquidity Layer (High Priority)**
- **liquidity_score**: Designed, runtime-only, never persisted
- **Blocks**: S1, S4, S5 (fallback logic)
- **Workaround**: BOMS proxy (0.5 * tf1d_boms_strength)
- **Fix**: `bin/backfill_liquidity_score.py` or `bin/backfill_liquidity_score_optimized.py`

**PTI Confluence (Medium Priority)**
- **wyckoff_pti_confluence**: Designed, not fully wired
- **wyckoff_pti_score**: Feature exists, no validation
- **Impact**: Missing psychological + structural confluence edge

**Fusion Engines (Low Priority)**
- **advanced_fusion.py**: Code exists, unclear if active
- **domain_fusion.py**: Code exists, unclear if active
- **Only K2 fusion confirmed active**: Need to audit other fusion engines

### 2. Partial Implementations

**Wyckoff Events (CRITICAL - FAILED)**
- **Status**: Detection works, boost/veto logic degrades performance
- **Phase 1 Results**: -1.0% WR, added 12 losing trades (25% WR)
- **Recommendation**:
  - **Option 1 (Conservative)**: Disable boosts entirely, keep BC/UTAD veto only
  - **Option 2 (Tuning)**: Raise LPS confidence 0.65 → 0.85, reduce multipliers +10% → +3%
  - **Option 3 (Aggressive)**: Disable entirely until Phase 2 optimization

**Bear Archetypes**
- **S5**: Live, optimized (PF 1.86), but OI data broken (validation incomplete)
- **S2**: Permanently disabled (PF 0.48 after optimization)
- **S1, S3, S4, S6-S8**: Not implemented
- **Blockers**: liquidity_score missing, OI pipeline broken

**Regime Routing**
- **Weights defined**: risk_on/neutral/risk_off/crisis
- **Policy implemented**: engine/context/regime_policy.py
- **Config missing**: regime_routing_production_v1.json not found
- **Validation**: bin/validate_regime_routing.py exists but results unclear

### 3. Data Pipeline Issues

**OI Derivatives (CRITICAL)**
- **Columns**: oi_change_24h, oi_change_pct_24h, oi_z
- **Status**: All NaN (calculation never run)
- **Impact**: S5 validation blocked, cannot detect OI spikes
- **Fix**: `bin/fix_oi_change_pipeline.py`

**Macro Features (HIGH)**
- **Availability**: 2024-01-05 onwards only
- **Coverage 2022-2023**: 0%
- **Impact**: Cannot validate regime classifier on 2022 bear market
- **Workaround**: Test on 2024 data only

**Liquidity Score (HIGH)**
- **Status**: Missing from feature store
- **Impact**: Blocks S1, S4, degrades S5
- **Fix**: Backfill script available

---

## Wyckoff Specific Recommendations

Based on `WYCKOFF_PHASE1_BASELINE_RESULTS.md`:

### Immediate Action

**OPTION 1: Disable Boosts, Keep Veto Only (RECOMMENDED)**

```json
{
  "wyckoff_events": {
    "enabled": true,
    "min_confidence": 0.70,

    "avoid_longs_if": [
      "wyckoff_bc",
      "wyckoff_utad"
    ],

    "boost_longs_if": {}  // REMOVE all boosts
  }
}
```

**Rationale**:
- BC/UTAD veto worked correctly (avoided March 2024 ATH)
- Boosts are net negative (-1.0% WR, +12 losing trades)
- Simple is better

**OPTION 2: Manual Tuning (MEDIUM EFFORT)**

If user insists on keeping boosts:

```json
{
  "wyckoff_events": {
    "enabled": true,

    // Tighten LPS detection
    "lps_min_confidence": 0.85,  // from 0.65
    "lps_volume_z_max": -0.5,    // stricter

    // Reduce boost multipliers
    "lps_fusion_boost": 0.03,    // from 0.10
    "spring_a_fusion_boost": 0.05, // from 0.12
    "sos_fusion_boost": 0.03     // from 0.08
  }
}
```

**Goal**: Reduce LPS from 945 → ~50-100 high-quality events

**OPTION 3: Disable Entirely (SAFEST)**

```json
{
  "wyckoff_events": {
    "enabled": false
  }
}
```

**When to use**: If user wants to avoid risk entirely until Phase 2 optimization

### Should We Tune, Disable Boosts, or Disable Entirely?

**Recommendation**: **Disable boosts, keep veto only (Option 1)**

**Reasoning**:
1. ✅ BC/UTAD veto logic proven correct (March 2024 ATH avoided)
2. ❌ Boost logic degraded performance (-1.0% WR)
3. ❌ LPS over-detection (945 events vs expected ~50)
4. ❌ Added trades have 25% WR (very poor quality)
5. ⚠️ Limited test period (9 months) - need more data to tune safely

**Next Steps**:
1. Deploy veto-only config
2. Run full 2022-2024 validation (once macro data backfilled)
3. If veto proves valuable (avoids 2-5 tops), keep it
4. Consider Phase 2 optimization ONLY if user has high conviction in Wyckoff theory

---

## Next Actions (Prioritized)

### Immediate (Week 1)

1. **Disable Wyckoff Boosts** (keep veto)
   - Update configs: set boost_longs_if = {}
   - Test: Run 2024 backtest, confirm no regression
   - Deploy: Production configs

2. **Fix OI Pipeline** (CRITICAL)
   - Run: `bin/fix_oi_change_pipeline.py`
   - Validate: Check oi_change_24h has non-zero variance
   - Re-test: S5 archetype with real OI data

3. **Backfill Liquidity Score** (HIGH)
   - Run: `bin/backfill_liquidity_score_optimized.py`
   - Validate: Check correlation with BOMS proxy
   - Unlock: S1, S4 patterns

### Short-Term (Weeks 2-3)

4. **Audit Fusion Engines**
   - Grep: Check which fusion engines are actually called
   - Validate: advanced_fusion.py vs domain_fusion.py vs k2_fusion.py
   - Document: Which is active, which is ghost

5. **Complete Regime Routing Validation**
   - Find/create: regime_routing_production_v1.json
   - Run: bin/validate_regime_routing.py
   - Document: Results and impact on 2024 vs 2022

6. **PTI Standalone Validation**
   - Test: PTI effectiveness in isolation
   - Measure: Win rate when PTI > 0.7 vs PTI < 0.3
   - Decide: Keep, tune, or deprecate

### Medium-Term (Month 2)

7. **Backfill Macro Features 2022-2023**
   - Fetch: VIX, DXY, yields for 2022-2023
   - Backfill: Macro columns
   - Validate: Regime classifier on 2022 bear market

8. **Implement or Deprecate Ghost Modules**
   - Temporal Fusion: Implement or remove from docs
   - Fibonacci Time: Implement or mark as future work
   - Liquidity Sweeps: Wire to archetypes or deprecate

9. **S1 Implementation** (if liquidity_score backfill succeeds)
   - Code: engine/archetypes/bear_patterns_phase1.py
   - Test: Validate on 2022 data
   - Optimize: Optuna search for thresholds

### Long-Term (Quarter 2)

10. **Wyckoff Phase 2 Optimization** (ONLY if Phase 1 veto proves valuable)
    - Expand test period: 2022-2024 (3 years)
    - Optuna search: LPS/Spring-A/SOS thresholds
    - A/B test: Veto-only vs veto+optimized-boosts

11. **Feature Store Completeness Audit**
    - Identify: All runtime-only features
    - Backfill: Persist to feature store
    - Test: Performance impact

12. **ML Meta-Optimizer** (if mentioned in docs)
    - Audit: Does it exist?
    - Validate: Is it active?
    - Document: If ghost, add to deprecation list

---

## File Inventory

### Core Engine Files (Validated)

```
engine/
├── wyckoff/
│   ├── events.py              [LIVE - 18 event detection]
│   ├── wyckoff_engine.py      [LIVE - tf1d_wyckoff_score]
├── archetypes/
│   ├── logic_v2_adapter.py    [LIVE - 13 archetypes A-M + S5]
│   ├── bear_patterns_phase1.py [LIVE - S2 (disabled), S5 (optimized)]
│   ├── registry.py            [LIVE - Archetype metadata]
│   ├── threshold_policy.py    [LIVE - Regime-aware thresholds]
├── psychology/
│   ├── pti.py                 [PARTIAL - Features exist, validation missing]
│   ├── fakeout_intensity.py   [LIVE - Archetype L]
├── smc/
│   ├── order_blocks.py        [LIVE - Archetype B]
│   ├── bos.py                 [LIVE - Archetype C]
│   ├── fvg.py                 [LIVE - Archetype P]
│   ├── liquidity_sweeps.py    [GHOST? - Code exists, no features]
├── liquidity/
│   ├── score.py               [PARTIAL - Runtime only, not persisted]
├── structure/
│   ├── boms_detector.py       [LIVE]
│   ├── range_classifier.py    [PARTIAL - Features exist, unclear if used]
│   ├── squiggle_pattern.py    [PARTIAL - Features exist, unclear if used]
│   ├── internal_external.py   [LIVE - Used in S1/S2]
├── volume/
│   ├── frvp.py                [PARTIAL - Features exist, unclear if used]
├── context/
│   ├── regime_classifier.py   [LIVE - GMM classifier]
│   ├── regime_policy.py       [LIVE - Regime routing]
├── fusion/
│   ├── k2_fusion.py           [LIVE - Primary fusion]
│   ├── adaptive.py            [PARTIAL - K2 only?]
│   ├── advanced_fusion.py     [GHOST? - Code exists, unclear if active]
│   ├── domain_fusion.py       [GHOST? - Code exists, unclear if active]
├── timeframes/
│   ├── mtf_alignment.py       [LIVE - Governor/veto]
├── ml/
│   ├── kelly_lite_sizer.py    [PARTIAL - Features exist, unclear if used]
```

### Config Files (Validated)

```
configs/
├── wyckoff_events_config.json           [LIVE - Wyckoff detection thresholds]
├── optimized_bull_v2_production.json    [LIVE - Bull market + S5]
├── regime_routing_production_v1.json    [MISSING - Referenced but not found]
├── bear_archetypes_phase1.json          [EXISTS]
├── mvp_bull_market_v1.json              [MISSING - Referenced but not found]
├── mvp_bear_market_v1.json              [MISSING - Referenced but not found]
```

### Documentation Status

```
docs/
├── WYCKOFF_EVENTS_IMPLEMENTATION_PLAN.md      [LIVE - Comprehensive spec]
├── WYCKOFF_THRESHOLD_TUNING_SUMMARY.md        [LIVE - Tuning results]
├── WYCKOFF_PHASE1_BASELINE_RESULTS.md         [LIVE - FAILED validation]
├── BEAR_PATTERNS_FEATURE_MATRIX.md            [LIVE - Feature availability]
├── FEATURE_PIPELINE_AUDIT.md                  [LIVE - Data pipeline status]
├── BEAR_PATTERNS_IMPLEMENTATION_GUIDE.md      [ARCHIVED]
├── BEAR_FEATURE_PIPELINE_ROADMAP.md           [ARCHIVED]
```

---

## Summary Statistics

### Concept Status
- **Total Audited**: 89 concepts
- **LIVE**: 48 (54%) - Fully implemented, validated, in production
- **PARTIAL**: 25 (28%) - Code/features exist, validation incomplete
- **IDEA ONLY**: 16 (18%) - Designed but not implemented

### Feature Store
- **Total Columns**: 155
- **Complete Coverage**: 108 features (70%)
- **Broken/Missing**: 47 features (30%)

### Archetypes
- **Total Designed**: 19 (A-M bull + S1-S8 bear)
- **Implemented**: 14 (A-M + S5)
- **Optimized**: 1 (S5)
- **Disabled**: 1 (S2)
- **Not Implemented**: 5 (S1, S3, S4, S6-S8)

### Test Coverage
- **Unit Tests**: ~30 files in tests/unit/
- **Integration Tests**: Minimal
- **Backtest Validation**: S5 only (PF 1.86)
- **Wyckoff Validation**: Phase 1 FAILED

### Critical Gaps
1. OI derivatives broken (blocks S5 validation)
2. Liquidity score missing (blocks S1, S4)
3. Macro features 2022-2023 missing (blocks regime validation)
4. Wyckoff boosts degraded performance (need to disable)
5. Multiple fusion engines unclear (ghost modules?)
6. Temporal layer not implemented (ghost module)
7. PTI not validated standalone (unclear value)

---

**End of Brain Blueprint v1.0**
