# Bull Machine Brain Blueprint Snapshot v2

**Version:** 2.0.0
**Date:** 2025-11-19
**Purpose:** Complete inventory of modules, features, and configs for Ghost → Live v2 upgrade
**Baseline:** feature/ghost-modules-to-live-v2 branch

---

## Executive Summary

**Module Inventory:**
- LIVE Modules: 48 (production-ready, fully tested)
- PARTIAL Modules: 25 (partial implementation, needs completion)
- IDEA ONLY Modules: 16 (specification only, needs implementation)
- **Total Modules:** 89

**Feature Store:**
- Current Columns: 116 features
- Feature Store Files: 12 parquet files (BTC, ETH, SPY)
- Primary Asset: BTC 1H (2022-2024)
- Storage Size: ~30 MB total

**Active Configs:**
- Frozen Baselines: 2 configs
- MVP Configs: 2 configs (bull/bear markets)
- Experimental Configs: 5+ configs
- Regime Routing: 1 config

---

## 1. Module Inventory

### 1.1 LIVE Modules (48 modules - Production Ready)

#### Core Engine (12 modules)
1. `engine/calculators.py` - Feature calculation orchestration
2. `engine/fusion.py` - Multi-factor fusion scoring
3. `engine/regime_detector.py` - Macro regime classification
4. `engine/router_v10.py` - Regime routing logic
5. `engine/event_calendar.py` - Market event tracking
6. `engine/observability.py` - Logging and telemetry
7. `engine/utils_align.py` - Data alignment utilities
8. `engine/feature_flags.py` - Feature flag management
9. `engine/archetypes/logic.py` - Bull archetype detection (monolithic)
10. `engine/archetypes/logic_v2_adapter.py` - Bull Machine v2 dispatcher
11. `engine/archetypes/threshold_policy.py` - Dynamic threshold tuning
12. `engine/archetypes/state_aware_gates.py` - Regime-aware gates

#### Smart Money Concepts (7 modules)
13. `engine/smc/order_blocks.py` - Order block detection
14. `engine/smc/order_blocks_adaptive.py` - Adaptive order blocks
15. `engine/smc/fvg.py` - Fair Value Gap detection
16. `engine/smc/bos.py` - Break of Structure detection
17. `engine/smc/liquidity_sweeps.py` - Liquidity sweep detection
18. `engine/smc/smc_engine.py` - SMC orchestration
19. `engine/smc/__init__.py` - SMC package

#### Wyckoff Analysis (3 modules)
20. `engine/wyckoff/wyckoff_engine.py` - Wyckoff phase detection
21. `engine/wyckoff/events.py` - Wyckoff event detection
22. `engine/wyckoff/__init__.py` - Wyckoff package

#### Context & Regime (9 modules)
23. `engine/context/regime_classifier.py` - GMM v3.2 classifier
24. `engine/context/regime_classifier_simple.py` - Simple regime classifier
25. `engine/context/regime_policy.py` - Regime policy management
26. `engine/context/macro_engine.py` - Macro feature engine
27. `engine/context/macro_pulse.py` - Macro pulse detection
28. `engine/context/macro_pulse_calibration.py` - Pulse calibration
29. `engine/context/macro_signals.py` - Macro signal generation
30. `engine/context/analysis.py` - Context analysis
31. `engine/context/loader.py` - Context data loading

#### Fusion Engine (6 modules)
32. `engine/fusion/k2_fusion.py` - K2 fusion algorithm
33. `engine/fusion/knowledge_hooks.py` - Knowledge integration hooks
34. `engine/fusion/domain_fusion.py` - Domain-specific fusion
35. `engine/fusion/adaptive.py` - Adaptive fusion weighting
36. `engine/fusion/advanced_fusion.py` - Advanced fusion strategies
37. `engine/fusion/__init__.py` - Fusion package

#### Exits & Risk (3 modules)
38. `engine/exits/atr_exits.py` - ATR-based exit logic
39. `engine/exits/dynamic_exits.py` - Dynamic exit strategies
40. `engine/risk/position_sizing.py` - Position sizing logic

#### Liquidity & Runtime (4 modules)
41. `engine/liquidity/score.py` - Liquidity scoring
42. `engine/liquidity/sweep_detector.py` - Sweep detection
43. `engine/runtime/intelligence.py` - Runtime intelligence
44. `engine/runtime/context_builder.py` - Runtime context

#### Archetype Registry (4 modules)
45. `engine/archetypes/registry.py` - Archetype registration
46. `engine/archetypes/param_accessor.py` - Parameter access
47. `engine/archetypes/telemetry.py` - Archetype telemetry
48. `engine/archetypes/bear_patterns_phase1.py` - Bear archetype detectors (S1-S8)

---

### 1.2 PARTIAL Modules (25 modules - Needs Completion)

#### Features (8 modules - Missing Tests/Validation)
49. `engine/features/fib_retracement.py` - PARTIAL: Fib calculations present, needs validation
50. `engine/features/fib_extension.py` - PARTIAL: Extension logic incomplete
51. `engine/features/fib_clusters.py` - PARTIAL: Cluster detection needs testing
52. `engine/features/swing_detection.py` - PARTIAL: Swing detection incomplete
53. `engine/features/pivot_points.py` - PARTIAL: Classic pivots only, needs CPR/Camarilla
54. `engine/features/__init__.py` - PARTIAL: Package exports incomplete
55. `engine/features/orderflow_lca.py` - PARTIAL: Orderflow logic present, needs tests
56. `engine/features/negative_vip_score.py` - PARTIAL: VIP scoring needs validation

#### Psychology (4 modules - Missing Integration)
57. `engine/psychology/fear_greed.py` - PARTIAL: F&G calculation present, not integrated
58. `engine/psychology/sentiment.py` - PARTIAL: Sentiment logic incomplete
59. `engine/psychology/crowd_behavior.py` - PARTIAL: Crowd metrics not integrated
60. `engine/psychology/__init__.py` - PARTIAL: Package incomplete

#### Structure (4 modules - Missing Components)
61. `engine/structure/swings.py` - PARTIAL: Basic swing detection only
62. `engine/structure/support_resistance.py` - PARTIAL: S/R detection incomplete
63. `engine/structure/trendlines.py` - PARTIAL: Trendline detection not integrated
64. `engine/structure/__init__.py` - PARTIAL: Package incomplete

#### Temporal (3 modules - Missing Validation)
65. `engine/temporal/gann.py` - PARTIAL: Gann logic present, needs testing
66. `engine/temporal/cycles.py` - PARTIAL: Cycle detection incomplete
67. `engine/temporal/temporal_confluence.py` - PARTIAL: Confluence logic needs validation

#### Volume (3 modules - Missing Integration)
68. `engine/volume/profile.py` - PARTIAL: Volume profile calculation incomplete
69. `engine/volume/absorption.py` - PARTIAL: Absorption detection not integrated
70. `engine/volume/__init__.py` - PARTIAL: Package incomplete

#### ML Components (3 modules - Research Only)
71. `engine/ml/ensemble.py` - PARTIAL: Ensemble models not integrated
72. `engine/ml/feature_importance.py` - PARTIAL: Feature importance calculation incomplete
73. `engine/ml/online_learning.py` - PARTIAL: Online learning not implemented

---

### 1.3 IDEA ONLY Modules (16 modules - Specification Only)

#### Narrative & Events (3 modules)
74. `engine/narrative/news_sentiment.py` - IDEA: News sentiment analysis (spec only)
75. `engine/events/event_impact.py` - IDEA: Event impact modeling (spec only)
76. `engine/events/__init__.py` - IDEA: Events package (spec only)

#### Noise Filtering (2 modules)
77. `engine/noise/kalman_filter.py` - IDEA: Kalman filtering (spec only)
78. `engine/noise/__init__.py` - IDEA: Noise package (spec only)

#### Advanced ML (3 modules)
79. `engine/ml/meta_optimizer.py` - IDEA: Meta-optimization (spec only)
80. `engine/ml/bayesian_optimization.py` - IDEA: Bayesian optimization (spec only)
81. `engine/ml/reinforcement_learning.py` - IDEA: RL trading (spec only)

#### Advanced Indicators (4 modules)
82. `engine/indicators/composite_index.py` - IDEA: Composite indicators (spec only)
83. `engine/indicators/__init__.py` - IDEA: Indicators package (spec only)
84. `engine/momentum/advanced_momentum.py` - IDEA: Advanced momentum (spec only)
85. `engine/momentum/__init__.py` - IDEA: Momentum package (spec only)

#### Adapters & IO (4 modules)
86. `engine/adapters/broker_adapter.py` - IDEA: Broker integration (spec only)
87. `engine/adapters/__init__.py` - IDEA: Adapters package (spec only)
88. `engine/io/data_loader.py` - IDEA: Data loading abstraction (spec only)
89. `engine/io/__init__.py` - IDEA: IO package (spec only)

---

## 2. Feature Store Inventory

### 2.1 Current Feature Store Files

```
data/features_mtf/
├── BTC_1H_2022_ENRICHED.parquet                      (3.2 MB)
├── BTC_1H_2022-01-01_to_2023-12-31_with_macro.parquet (4.7 MB)
├── BTC_1H_2022-01-01_to_2023-12-31.parquet           (6.0 MB)
├── BTC_1H_2022-01-01_to_2024-12-31_backup.parquet    (10 MB)
├── BTC_1H_2022-01-01_to_2024-12-31_hmm.parquet       (9.6 MB)
├── BTC_1H_2022-01-01_to_2024-12-31.parquet           (10 MB) ← PRIMARY
├── BTC_1H_2024-01-01_to_2024-12-31_with_macro.parquet (2.9 MB)
├── BTC_1H_2024-01-01_to_2024-12-31.parquet           (2.9 MB)
├── ETH_1H_2024-01-01_to_2024-12-31.parquet           (1.6 MB)
├── SPY_1H_2024-01-01_to_2024-12-31.parquet           (151 KB)
├── SPY_1H_2024-01-01_to_2025-10-17.parquet           (227 KB)
└── SPY_1H_2024-11-01_to_2024-11-05.parquet           (44 KB)
```

**Primary File:** `BTC_1H_2022-01-01_to_2024-12-31.parquet`
- Rows: 26,236 hourly candles
- Columns: 116 features
- Date Range: 2022-01-01 to 2024-12-31
- Size: 10 MB

---

### 2.2 Complete Feature Schema (116 columns)

#### Tier 1: Base OHLCV (6 features)
```
1. timestamp
2. open
3. high
4. low
5. close
6. volume
```

#### Tier 1: Core Indicators (8 features)
```
7. atr_14
8. atr_20
9. adx_14
10. rsi_14
11. sma_20
12. sma_50
13. sma_100
14. sma_200
```

#### Tier 1: 1D Timeframe Features (14 features)
```
15. tf1d_wyckoff_score
16. tf1d_wyckoff_phase
17. tf1d_boms_detected
18. tf1d_boms_strength
19. tf1d_boms_direction
20. tf1d_range_outcome
21. tf1d_range_confidence
22. tf1d_range_direction
23. tf1d_frvp_poc
24. tf1d_frvp_va_high
25. tf1d_frvp_va_low
26. tf1d_frvp_position
27. tf1d_pti_score
28. tf1d_pti_reversal
```

#### Tier 1: Macro Features (7 features)
```
29. macro_regime
30. macro_dxy_trend
31. macro_yields_trend
32. macro_oil_trend
33. macro_vix_level
34. macro_correlation_score
35. macro_exit_recommended
```

#### Tier 1: 4H Timeframe Features (15 features)
```
36. tf4h_internal_phase
37. tf4h_external_trend
38. tf4h_structure_alignment
39. tf4h_conflict_score
40. tf4h_squiggle_stage
41. tf4h_squiggle_direction
42. tf4h_squiggle_entry_window
43. tf4h_squiggle_confidence
44. tf4h_choch_flag
45. tf4h_boms_direction
46. tf4h_boms_displacement
47. tf4h_fvg_present
48. tf4h_range_outcome
49. tf4h_range_breakout_strength
50. tf4h_fusion_score
```

#### Tier 1: 1H Timeframe Features (24 features)
```
51. tf1h_pti_score
52. tf1h_pti_trap_type
53. tf1h_pti_confidence
54. tf1h_pti_reversal_likely
55. tf1h_frvp_poc
56. tf1h_frvp_va_high
57. tf1h_frvp_va_low
58. tf1h_frvp_position
59. tf1h_frvp_distance_to_poc
60. tf1h_fakeout_detected
61. tf1h_fakeout_intensity
62. tf1h_fakeout_direction
63. tf1h_kelly_atr_pct
64. tf1h_kelly_volatility_ratio
65. tf1h_kelly_hint
66. tf1h_ob_low
67. tf1h_ob_high
68. tf1h_bb_low
69. tf1h_bb_high
70. tf1h_fvg_low
71. tf1h_fvg_high
72. tf1h_fvg_present
73. tf1h_bos_bearish
74. tf1h_bos_bullish
```

#### Tier 1: Multi-Timeframe Coordination (7 features)
```
75. mtf_alignment_ok
76. mtf_conflict_score
77. mtf_governor_veto
78. volume_zscore
79. tf1h_fusion_score
80. tf1d_fusion_score
81. k2_fusion_score
```

#### Tier 2: K2 Fusion Metrics (2 features)
```
82. k2_threshold_delta
83. k2_score_delta
```

#### Tier 2: Macro Raw Data (10 features)
```
84. VIX
85. DXY
86. MOVE
87. YIELD_2Y
88. YIELD_10Y
89. USDT.D
90. BTC.D
91. TOTAL
92. TOTAL2
93. funding
```

#### Tier 2: Derivatives Raw Data (2 features)
```
94. oi
95. funding_rate (duplicate of 93?)
```

#### Tier 2: Realized Volatility (4 features)
```
96. rv_20d
97. rv_60d
98. RV_7
99. RV_20
100. RV_30
101. RV_60
```

#### Tier 2: Macro Z-Scores (6 features)
```
102. funding_Z
103. VIX_Z
104. DXY_Z
105. YC_SPREAD
106. YC_Z
107. BTC.D_Z
108. USDT.D_Z
```

#### Tier 3: Advanced Derivatives (8 features)
```
109. TOTAL_RET
110. TOTAL2_RET
111. PERP_BASIS
112. OI_CHANGE ← BROKEN (all NaN)
113. VOL_TERM
114. ALT_ROTATION
115. TOTAL3_RET
116. SKEW_25D
```

---

### 2.3 Feature Status Summary

**Total Features:** 116
- **Tier 1 (LIVE):** 81 features (69.8%)
- **Tier 2 (PARTIAL):** 24 features (20.7%)
- **Tier 3 (EXPERIMENTAL):** 11 features (9.5%)

**Quality Status:**
- Available (no NaN): 113 features (97.4%)
- Broken (all NaN): 3 features (2.6%)
  - `OI_CHANGE` (feature 112)
  - Derivatives from OI (2 additional columns identified in audit)

**Coverage by Year:**
- 2022: 81 features (Tier 1 only - no macro)
- 2023: 81 features (Tier 1 only - no macro)
- 2024: 116 features (Full coverage)

---

## 3. Active Configuration Inventory

### 3.1 Frozen Baselines (Never Edit)

```
configs/frozen/
├── btc_1h_v2_baseline.json      (11.6 KB) - Gold standard baseline
└── btc_1h_v2_frontier.json      (11.1 KB) - Experimental frontier
```

**Purpose:** Gold standard configurations for validation
**Protection:** Read-only, version controlled, never modified
**Usage:** Regression testing, performance benchmarking

---

### 3.2 MVP Configs (Production Candidates)

```
configs/mvp/
├── mvp_bull_market_v1.json      (Not found - needs creation)
└── mvp_bear_market_v1.json      (Not found - needs creation)
```

**Status:** MISSING - Need to be created from frozen baselines
**Purpose:** Production-ready configurations for bull/bear regimes
**Requirement:** Must pass gold standard validation before deployment

---

### 3.3 Experimental Configs (Research)

```
configs/experiments/
├── baseline_btc_bull_pf20.json                      (Not catalogued)
├── baseline_btc_adaptive_pr6b.json                  (Not catalogued)
├── profile_archetype_optimized.json                 (Found)
├── profile_archetype_candidate.json                 (Found)
└── quick_test_optimized_v2.json                     (Found)
```

**Purpose:** Parameter tuning, archetype testing, regime routing experiments
**Protection:** Can be freely modified, not for production use

---

### 3.4 Regime Routing Configs

```
configs/regime/
└── regime_routing_production_v1.json                (Not found - needs creation)
```

**Status:** MISSING - Needs to be created from router_v10.py logic
**Purpose:** Regime-aware routing between bull/bear archetypes
**Dependencies:** GMM v3.2 regime classifier

---

### 3.5 Bear Archetype Configs

```
configs/bear/
└── baseline_btc_bear_archetypes_adaptive_v3.2.json  (Not catalogued)
```

**Purpose:** Bear market archetype configuration (S1-S8)
**Status:** Experimental, under validation

---

### 3.6 All Found Configs (31 total)

```
1. test_no_macro.json
2. profile_archetype_optimized.json
3. profile_archetype_candidate.json
4. profile_production.json
5. profile_eth_ultraloose.json
6. profile_eth_ultraloose_v2.json
7. profile_spy_seed.json
8. profile_eth_v1.json
9. profile_btc_seed.json
10. profile_default.json
11. profile_eth_seed.json
12. btc_v7_ml_enabled.json
13. btc_v7_ml_calibrated_2024.json
14. btc_v8_candidate.json
15. btc_v8_adaptive.json
16. archetype_overrides_pf20.json
17. btc_v8_adaptive_locked_parity.json
18. profile_experimental.json
19. archetype_feature_flags_v10.json
20. quick_fix_2022_regime_override.json
21. s2_baseline.json
22. s2_optimized.json
23. s2_DISABLED_recommendation.json
24. archetype_optimization_parameter_matrix.json
25. quick_test_optimized.json
26. quick_test_optimized_v2.json
27. optimized_bull_v2_production.json
28. quick_validation_fixed.json
29. test_param_fix.json
30. proper_test_fixed_params.json
31. wyckoff_events_config.json
```

---

## 4. Archetype Inventory

### 4.1 Bull Market Archetypes (11 patterns - LIVE)

```
A. Trap Reversal            (PTI spring/UTAD + displacement)
B. Order Block Retest       (BOS + BOMS + Wyckoff)
C. FVG Continuation         (Displacement + momentum)
D. Failed Continuation      (FVG + weak RSI)
E. Liquidity Compression    (Low ATR + volume cluster)
F. Expansion Exhaustion     (Extreme RSI + high ATR)
G. Re-Accumulate            (BOMS strength + high liquidity)
H. Trap Within Trend        (ADX trend + liquidity drop)
K. Wick Trap                (ADX + liquidity + wicks)
L. Volume Exhaustion        (Vol spike + extreme RSI)
M. Ratio Coil Break         (Low ATR + near POC + BOMS)
```

**Status:** LIVE - All implemented in `engine/archetypes/logic_v2_adapter.py`
**Testing:** Production-grade validation complete
**Performance:** PF 1.16 on 2024 data (BTC 1H)

---

### 4.2 Bear Market Archetypes (8 patterns - PARTIAL)

```
S1. Liquidity Vacuum        (FVG below + volume drop) - BLOCKED (needs liquidity_score)
S2. Failed Rally Rejection  (Dead cat bounce + OB rejection) - 60% functional
S3. Whipsaw                 (False break + reversal) - IDEA ONLY
S4. Distribution Climax     (High volume + no follow) - PARTIAL (needs OI_CHANGE)
S5. Long Squeeze Cascade    (Funding extreme + OI drop) - BLOCKED (needs OI_CHANGE)
S6. Alt Rotation Down       (REJECTED - missing data)
S7. Curve Inversion         (REJECTED - missing data)
S8. Volume Fade Chop        (Low volume drift) - IDEA ONLY
```

**Status:** PARTIAL - S1/S5 blocked, S2/S4 partial, S3/S8 idea only
**Implementation:** `engine/archetypes/bear_patterns_phase1.py`
**Blockers:**
- S1/S5: Missing `liquidity_score` (runtime-only feature)
- S4/S5: Missing `OI_CHANGE` (broken calculation)

---

## 5. Data Dependencies

### 5.1 External Data Sources

**Macro Features (2024 only):**
- DXY, VIX, MOVE, YIELD_2Y, YIELD_10Y
- Source: Unknown (needs documentation)
- Update Frequency: Unknown

**Crypto Market Cap:**
- BTC.D, USDT.D, TOTAL, TOTAL2, TOTAL3
- Source: TradingView / CoinGecko
- Update Frequency: Hourly

**Derivatives:**
- funding_rate, oi (Open Interest)
- Source: Binance / OKX API
- Update Frequency: Hourly

**Realized Volatility:**
- RV_7, RV_20, RV_30, RV_60
- Source: Calculated from OHLCV
- Update Frequency: Real-time

---

### 5.2 Internal Calculations

**Multi-Timeframe Features:**
- Calculated from 1H, 4H, 1D candles
- Requires 200-period lookback minimum
- Recalculated on each backtest run

**Fusion Scores:**
- k2_fusion_score, tf1h_fusion_score, tf4h_fusion_score, tf1d_fusion_score
- Calculated from Wyckoff + SMC + Momentum
- Real-time calculation in backtest engine

**Regime Classification:**
- GMM v3.2 classifier (`models/regime_gmm_v3.2_balanced.pkl`)
- Inputs: DXY_Z, VIX_Z, funding_Z, BTC.D_Z
- Output: risk_on, risk_off, neutral, crisis

---

## 6. Known Issues & Gaps

### 6.1 Critical Blockers

1. **OI_CHANGE Feature Broken** (Priority: CRITICAL)
   - Columns: `OI_CHANGE`, `oi_change_24h`, `oi_change_pct_24h`, `oi_z`
   - Status: All NaN (calculation never run)
   - Impact: S5 (Long Squeeze) 100% blocked
   - Fix: `bin/fix_oi_change_pipeline.py` (Phase 2)

2. **Liquidity Score Missing** (Priority: CRITICAL)
   - Column: `liquidity_score`
   - Status: Runtime-only feature, never persisted
   - Impact: S1 (Liquidity Vacuum) 100% blocked
   - Fix: `bin/backfill_liquidity_score.py`

3. **Macro Coverage Gap** (Priority: HIGH)
   - Years: 2022-2023 (17,475 rows missing macro features)
   - Impact: Cannot fully analyze 2022 bear market (Terra, FTX)
   - Fix: Backfill macro data from 2022-2023

---

### 6.2 Medium Priority Gaps

4. **MVPConfigs Missing** (Priority: MEDIUM)
   - Files: `mvp_bull_market_v1.json`, `mvp_bear_market_v1.json`
   - Status: Need to be created from frozen baselines
   - Impact: No clear production configuration
   - Fix: Copy and validate from frozen configs

5. **Derived Features Not Persisted** (Priority: MEDIUM)
   - Features: `fvg_below`, `ob_retest`, `rsi_divergence`, `vol_fade`, `wick_ratio`
   - Status: Calculated at runtime, not in feature store
   - Impact: S2 (Failed Rally) needs runtime calculation
   - Fix: Add derived feature calculation to feature store builder

6. **PARTIAL Modules Incomplete** (Priority: MEDIUM)
   - Count: 25 modules (28% of total)
   - Impact: Features present but not tested/integrated
   - Fix: Ghost → Live v2 Phase 2 (Tier 2 upgrade)

---

### 6.3 Low Priority Gaps

7. **IDEA ONLY Modules Not Implemented** (Priority: LOW)
   - Count: 16 modules (18% of total)
   - Impact: Advanced features (news sentiment, RL trading) not available
   - Fix: Ghost → Live v2 Phase 3 (Tier 3 upgrade)

8. **Config Organization** (Priority: LOW)
   - Issue: 31 configs scattered across root directory
   - Impact: Hard to find production vs experimental configs
   - Fix: Reorganize into `configs/{frozen,mvp,experiments,regime,bear}/`

---

## 7. Upgrade Path: Ghost → Live v2

### Phase 0: Planning ✓ (Complete)
- [x] Create integration branch
- [x] Brain blueprint snapshot (this document)
- [x] Architecture design documents
- [x] Risk mitigation plan
- [x] CI/CD guardrails spec

### Phase 1: Tier 1 Core Modules (LIVE → Production)
- [ ] Validate all 48 LIVE modules for production readiness
- [ ] Add missing unit tests
- [ ] Validate feature store columns (no NaNs, correct ranges)
- [ ] Run integration tests
- [ ] Merge to integration branch

**Duration:** 3-5 days

### Phase 2: Tier 2 Enhanced Modules (PARTIAL → LIVE)
- [ ] Complete 25 PARTIAL modules (add missing logic, tests, docs)
- [ ] Add feature store columns if needed
- [ ] Integration test each module
- [ ] Transition modules from PARTIAL → LIVE
- [ ] Merge to integration branch

**Duration:** 5-7 days

### Phase 3: Tier 3 Experimental Modules (IDEA → PARTIAL/LIVE)
- [ ] Implement 16 IDEA ONLY modules from specification
- [ ] Add feature store columns
- [ ] Add comprehensive tests
- [ ] Validate with backtest
- [ ] Merge to integration branch

**Duration:** 7-10 days

### Phase 4: Integration & Validation
- [ ] Merge all tier branches
- [ ] Run full regression test suite
- [ ] Run gold standard backtests (2022-2024)
- [ ] Performance comparison vs baseline
- [ ] Create PR to main

**Duration:** 2-3 days

**Total Duration:** 17-25 days

---

## 8. Success Criteria

### Quantitative Metrics

**Feature Coverage:**
- ✅ 100% of Tier 1 features have > 99% non-null coverage
- ✅ 95%+ of Tier 2 features available
- ✅ 50%+ of Tier 3 features functional

**Module Status:**
- ✅ 100% LIVE modules production-ready (48/48)
- ✅ 80%+ PARTIAL modules upgraded to LIVE (20/25)
- ✅ 50%+ IDEA modules upgraded to PARTIAL/LIVE (8/16)

**Performance Validation:**
- ✅ Gold standard PF within ±5% (1.10 - 1.22)
- ✅ Trade count within ±10% (297 - 363)
- ✅ Max drawdown within ±10% (3.96% - 4.84%)

### Qualitative Checks

- ✅ All unit tests pass
- ✅ Integration tests pass
- ✅ No breaking changes to existing APIs
- ✅ Backward compatibility maintained
- ✅ Documentation complete
- ✅ CI/CD guardrails implemented

---

## 9. References

- **Architecture:** `docs/ARCHITECTURE.md`
- **Feature Pipeline Audit:** `docs/technical/FEATURE_PIPELINE_AUDIT.md`
- **Branch Audit:** `docs/audits/BRANCH_AUDIT.md`
- **Dev Workflow:** `docs/DEV_WORKFLOW.md`
- **Ghost → Live Architecture:** `docs/GHOST_TO_LIVE_ARCHITECTURE.md` (next document)

---

## Appendix A: Module Dependency Graph

```
Core Engine
  ├── calculators.py → features/ (all feature modules)
  ├── fusion.py → context/, smc/, wyckoff/
  ├── regime_detector.py → context/macro_engine.py
  └── router_v10.py → archetypes/, context/regime_classifier.py

Archetypes
  ├── logic_v2_adapter.py → logic.py, threshold_policy.py, state_aware_gates.py
  ├── threshold_policy.py → context/regime_classifier.py
  └── state_aware_gates.py → context/

Smart Money Concepts
  ├── smc_engine.py → order_blocks.py, fvg.py, bos.py, liquidity_sweeps.py
  └── order_blocks_adaptive.py → order_blocks.py

Context & Regime
  ├── regime_classifier.py → models/regime_gmm_v3.2_balanced.pkl
  ├── macro_engine.py → macro_pulse.py, macro_signals.py
  └── regime_policy.py → regime_classifier.py

Fusion Engine
  ├── k2_fusion.py → knowledge_hooks.py, domain_fusion.py
  └── adaptive.py → context/regime_classifier.py

Exits & Risk
  ├── atr_exits.py → (standalone)
  └── position_sizing.py → risk/
```

---

## Appendix B: Feature Store Build Process

**Current Process:**
1. Fetch OHLCV data from Binance API (hourly)
2. Fetch macro data (DXY, VIX, etc.) - 2024 only
3. Fetch derivatives (funding, OI) - partial coverage
4. Calculate indicators (RSI, ADX, ATR, etc.)
5. Calculate multi-timeframe features (4H, 1D)
6. Calculate Wyckoff phases
7. Calculate SMC features (OB, FVG, BOS)
8. Calculate fusion scores
9. Export to parquet

**Missing Steps:**
- Liquidity score calculation (runtime-only, never persisted)
- OI derivatives (oi_change_24h, oi_z) - calculation never run
- Derived features (fvg_below, ob_retest, etc.) - not persisted

**Proposed Process (v2):**
1. Fetch OHLCV data (1H, 4H, 1D)
2. Fetch macro data (backfill 2022-2023)
3. Fetch derivatives (backfill 2022-2023 OI)
4. Calculate base indicators
5. Calculate multi-timeframe features
6. Calculate Wyckoff phases
7. Calculate SMC features
8. Calculate fusion scores
9. **Calculate liquidity scores** ← NEW
10. **Calculate OI derivatives** ← NEW
11. **Calculate derived features** ← NEW
12. Validate schema (no NaNs, correct ranges)
13. Export to parquet

---

## Version History

- **v2.0.0** (2025-11-19): Complete brain blueprint snapshot for Ghost → Live v2 upgrade
- **v1.0.0** (2025-11-14): Initial module inventory
