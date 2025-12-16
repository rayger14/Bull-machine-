# Ghost → Live v2 System Architecture

**Version:** 2.0.0
**Date:** 2025-11-19
**Status:** Design Complete - Ready for Implementation
**Purpose:** Define the system architecture for upgrading 89 modules from Ghost → Live status

---

## Executive Summary

This document defines the architecture for transitioning Bull Machine from a partially-implemented system (48 LIVE / 25 PARTIAL / 16 IDEA modules) to a production-ready trading engine with 100% feature coverage.

**Key Architectural Decisions:**
1. **Incremental Upgrade:** No "big bang" rewrite - upgrade modules in tiers
2. **Backward Compatibility:** Maintain all existing APIs, add new capabilities via adapters
3. **Feature Store as Source of Truth:** All features persisted in parquet, no runtime-only calculations
4. **Regime-Aware Routing:** GMM v3.2 classifier routes to bull/bear archetype sets
5. **Soft Filters Over Hard Gates:** Penalties instead of vetoes to preserve signals

---

## 1. High-Level System Architecture

```
┌────────────────────────────────────────────────────────────────┐
│                    Bull Machine Trading Engine                  │
└────────────────────────────────────────────────────────────────┘
                              │
                              ▼
        ┌─────────────────────────────────────┐
        │     Data Ingestion Layer             │
        │  (Raw OHLCV + Macro + Derivatives)   │
        └─────────────────────────────────────┘
                              │
                              ▼
        ┌─────────────────────────────────────┐
        │    Feature Engineering Pipeline      │
        │  (116 features → 140+ after v2)     │
        └─────────────────────────────────────┘
                              │
                              ▼
        ┌─────────────────────────────────────┐
        │       Feature Store (Parquet)        │
        │    (Source of Truth - No NaNs)       │
        └─────────────────────────────────────┘
                              │
                ┌─────────────┴──────────────┐
                │                            │
                ▼                            ▼
    ┌──────────────────┐        ┌──────────────────┐
    │ Regime Classifier│        │ Context Builder   │
    │  (GMM v3.2)      │        │ (Runtime State)   │
    └──────────────────┘        └──────────────────┘
                │                            │
                └─────────────┬──────────────┘
                              ▼
                ┌──────────────────────────┐
                │    Regime Router v10     │
                │ (risk_on/risk_off/etc.)  │
                └──────────────────────────┘
                              │
                ┌─────────────┴─────────────┐
                │                           │
                ▼                           ▼
    ┌──────────────────┐      ┌──────────────────┐
    │  Bull Archetypes │      │  Bear Archetypes │
    │    (A-M: 11)     │      │   (S1-S8: 8)     │
    └──────────────────┘      └──────────────────┘
                │                           │
                └─────────────┬─────────────┘
                              ▼
                ┌──────────────────────────┐
                │     Fusion Engine        │
                │  (Wyckoff + SMC + Mom)   │
                └──────────────────────────┘
                              │
                              ▼
                ┌──────────────────────────┐
                │   State-Aware Gates      │
                │ (Threshold Policy + ADX) │
                └──────────────────────────┘
                              │
                              ▼
                ┌──────────────────────────┐
                │    Entry Signal          │
                │  (Long/Short/Neutral)    │
                └──────────────────────────┘
                              │
                              ▼
                ┌──────────────────────────┐
                │   Position Sizing        │
                │  (ATR-based + Kelly)     │
                └──────────────────────────┘
                              │
                              ▼
                ┌──────────────────────────┐
                │    Exit Strategy         │
                │ (ATR stops + Wyckoff)    │
                └──────────────────────────┘
                              │
                              ▼
                ┌──────────────────────────┐
                │   Backtest / Live Trade  │
                │  (Execution Engine)      │
                └──────────────────────────┘
```

---

## 2. Data Flow Architecture

### 2.1 Feature Engineering Pipeline

```
┌─────────────┐
│ Raw OHLCV   │ ← Binance API (1H candles)
│ (1H/4H/1D)  │
└──────┬──────┘
       │
       ▼
┌─────────────┐
│  Macro Data │ ← TradingView / CoinGecko
│ (DXY, VIX,  │   (Hourly updates)
│  BTC.D, etc)│
└──────┬──────┘
       │
       ▼
┌─────────────┐
│ Derivatives │ ← Binance / OKX API
│ (funding,   │   (Hourly updates)
│  OI, basis) │
└──────┬──────┘
       │
       ▼
┌─────────────────────────────────────┐
│    Feature Calculation Engine       │
│                                     │
│  Tier 1: Core Indicators (ATR, RSI) │
│  Tier 2: MTF Features (4H, 1D)      │
│  Tier 3: Wyckoff Phases             │
│  Tier 4: SMC (OB, FVG, BOS)         │
│  Tier 5: Fusion Scores              │
│  Tier 6: Derived Features (NEW)     │
│  Tier 7: Liquidity Score (NEW)      │
│  Tier 8: OI Derivatives (NEW)       │
└─────────────────────────────────────┘
       │
       ▼
┌─────────────────────────────────────┐
│    Schema Validation Layer          │
│                                     │
│  ✓ No NaN values                    │
│  ✓ Correct data types               │
│  ✓ Valid ranges (e.g., RSI 0-100)   │
│  ✓ Timestamp continuity             │
└─────────────────────────────────────┘
       │
       ▼
┌─────────────────────────────────────┐
│      Feature Store (Parquet)        │
│                                     │
│  BTC_1H_2022-2024.parquet (10 MB)   │
│  - 26,236 rows × 140 columns        │
│  - 100% coverage (no NaNs)          │
│  - Validated schema                 │
└─────────────────────────────────────┘
```

---

### 2.2 Backtest Execution Flow

```
1. Load Feature Store
   ↓
2. Classify Regime (GMM v3.2)
   ├─ risk_on → Bull Archetypes
   ├─ risk_off → Bear Archetypes
   ├─ neutral → Both (weighted)
   └─ crisis → Bear Only
   ↓
3. Archetype Detection (Evaluate-All Dispatcher)
   ├─ Bull: A, B, C, D, E, F, G, H, K, L, M
   └─ Bear: S1, S2, S3, S4, S5, S8
   ↓
4. Fusion Scoring (Multi-Factor)
   ├─ Wyckoff: 33.1%
   ├─ Liquidity: 39.2%
   ├─ Momentum: 20.5%
   └─ Fakeout Penalty: 7.5%
   ↓
5. State-Aware Gates (Threshold Policy)
   ├─ ADX weak/strong thresholds
   ├─ ATR percentile gates
   ├─ Funding Z-score penalties
   └─ 4H trend alignment
   ↓
6. Entry Signal (Long/Short/Neutral)
   ↓
7. Position Sizing (ATR-based)
   ↓
8. Exit Strategy (ATR stops + Wyckoff phases)
   ↓
9. Record Trade
   ↓
10. Update Metrics (PF, Win Rate, DD, Sharpe)
```

---

## 3. Module Dependency Graph

### 3.1 Core Engine Dependencies

```
calculators.py (Tier 1 - LIVE)
  ├── features/fib_retracement.py (Tier 2 - PARTIAL)
  ├── features/swing_detection.py (Tier 2 - PARTIAL)
  ├── features/pivot_points.py (Tier 2 - PARTIAL)
  └── features/orderflow_lca.py (Tier 2 - PARTIAL)

fusion.py (Tier 1 - LIVE)
  ├── fusion/k2_fusion.py (Tier 1 - LIVE)
  ├── fusion/knowledge_hooks.py (Tier 1 - LIVE)
  ├── fusion/domain_fusion.py (Tier 1 - LIVE)
  └── context/macro_engine.py (Tier 1 - LIVE)

regime_detector.py (Tier 1 - LIVE)
  ├── context/regime_classifier.py (Tier 1 - LIVE)
  ├── context/macro_pulse.py (Tier 1 - LIVE)
  └── context/macro_signals.py (Tier 1 - LIVE)

router_v10.py (Tier 1 - LIVE)
  ├── archetypes/logic_v2_adapter.py (Tier 1 - LIVE)
  ├── archetypes/threshold_policy.py (Tier 1 - LIVE)
  ├── archetypes/state_aware_gates.py (Tier 1 - LIVE)
  └── context/regime_classifier.py (Tier 1 - LIVE)
```

---

### 3.2 Archetype Detection Dependencies

```
logic_v2_adapter.py (Tier 1 - LIVE)
  ├── logic.py (Tier 1 - LIVE) ← Bull archetypes A-M
  ├── bear_patterns_phase1.py (Tier 2 - PARTIAL) ← Bear archetypes S1-S8
  ├── threshold_policy.py (Tier 1 - LIVE)
  ├── state_aware_gates.py (Tier 1 - LIVE)
  └── registry.py (Tier 1 - LIVE)

threshold_policy.py (Tier 1 - LIVE)
  ├── context/regime_classifier.py (Tier 1 - LIVE)
  └── context/runtime_context.py (Tier 1 - LIVE)

state_aware_gates.py (Tier 1 - LIVE)
  ├── context/regime_classifier.py (Tier 1 - LIVE)
  └── No other dependencies (standalone gating logic)
```

---

### 3.3 SMC Dependencies

```
smc_engine.py (Tier 1 - LIVE)
  ├── order_blocks.py (Tier 1 - LIVE)
  ├── order_blocks_adaptive.py (Tier 1 - LIVE)
  ├── fvg.py (Tier 1 - LIVE)
  ├── bos.py (Tier 1 - LIVE)
  └── liquidity_sweeps.py (Tier 1 - LIVE)

order_blocks_adaptive.py (Tier 1 - LIVE)
  ├── order_blocks.py (Tier 1 - LIVE)
  └── No other dependencies (extends base OB)
```

---

### 3.4 Feature Engineering Dependencies

```
features/fib_retracement.py (Tier 2 - PARTIAL)
  ├── features/swing_detection.py (Tier 2 - PARTIAL)
  └── No other dependencies

features/swing_detection.py (Tier 2 - PARTIAL)
  ├── structure/swings.py (Tier 2 - PARTIAL)
  └── No other dependencies

features/pivot_points.py (Tier 2 - PARTIAL)
  └── No dependencies (standalone)

features/orderflow_lca.py (Tier 2 - PARTIAL)
  └── No dependencies (standalone)
```

---

## 4. Integration Points

### 4.1 Regime Classifier → Router v10

**Interface:**
```python
# Input: Feature store row
regime = regime_classifier.classify(
    dxy_z=row['DXY_Z'],
    vix_z=row['VIX_Z'],
    funding_z=row['funding_Z'],
    btc_d_z=row['BTC.D_Z']
)
# Output: 'risk_on', 'risk_off', 'neutral', 'crisis'

# Router uses regime to select archetype set
if regime == 'risk_on':
    archetypes = bull_archetypes  # A-M
elif regime == 'risk_off':
    archetypes = bear_archetypes  # S1-S8
elif regime == 'neutral':
    archetypes = bull_archetypes + bear_archetypes  # Both
else:  # crisis
    archetypes = bear_archetypes  # S1-S8 only
```

**Data Flow:**
```
Feature Store Row
  ↓
Regime Classifier (GMM v3.2)
  ↓
Regime Label (risk_on/risk_off/neutral/crisis)
  ↓
Router v10 (archetype selection)
  ↓
Archetype Dispatcher (evaluate selected set)
```

---

### 4.2 Archetype Detection → Fusion Engine

**Interface:**
```python
# Archetype detection returns matches
matches = archetype_detector.detect_all(ctx, row)
# matches = [
#   {'archetype': 'A', 'score': 0.85, 'confidence': 0.92},
#   {'archetype': 'H', 'score': 0.72, 'confidence': 0.88}
# ]

# Fusion engine combines archetype signals with domain scores
fusion_score = fusion_engine.compute(
    wyckoff_score=row['tf1h_wyckoff_score'],
    liquidity_score=row['liquidity_score'],
    momentum_score=row['rsi_14'] / 100.0,
    archetype_matches=matches
)
# fusion_score = 0.78 (0.0-1.0 range)
```

**Data Flow:**
```
Feature Store Row
  ↓
Archetype Detector (logic_v2_adapter.py)
  ↓
Archetype Matches (list of dicts)
  ↓
Fusion Engine (fusion.py)
  ↓
Fusion Score (0.0-1.0)
```

---

### 4.3 Fusion Engine → State-Aware Gates

**Interface:**
```python
# Fusion score is base signal
fusion_score = 0.78

# State-aware gates apply penalties
gated_score = state_aware_gates.apply(
    fusion_score=fusion_score,
    adx=row['adx_14'],
    atr_percentile=row['atr_percentile'],
    funding_z=row['funding_Z'],
    tf4h_trend=row['tf4h_external_trend']
)
# gated_score = 0.65 (after penalties)

# Threshold policy compares to regime-specific threshold
threshold = threshold_policy.get_threshold(regime='risk_on')
# threshold = 0.60 (dynamic, regime-aware)

# Entry signal
if gated_score > threshold:
    signal = 'LONG'
else:
    signal = 'NEUTRAL'
```

**Data Flow:**
```
Fusion Score (0.0-1.0)
  ↓
State-Aware Gates (apply penalties)
  ↓
Gated Score (0.0-1.0)
  ↓
Threshold Policy (regime-specific threshold)
  ↓
Entry Signal (LONG/SHORT/NEUTRAL)
```

---

### 4.4 Feature Store → All Modules

**Interface:**
```python
# Feature store is loaded once at backtest start
feature_store = pd.read_parquet('BTC_1H_2022-2024.parquet')

# All modules access features via row iteration
for idx, row in feature_store.iterrows():
    # Regime classification
    regime = regime_classifier.classify(row)

    # Archetype detection
    matches = archetype_detector.detect_all(row)

    # Fusion scoring
    fusion_score = fusion_engine.compute(row, matches)

    # Entry/exit logic
    signal = entry_logic.evaluate(fusion_score, row)

    # Position sizing
    size = position_sizer.compute(row['atr_14'], capital)

    # ... etc
```

**Critical Constraint:** NO runtime feature calculation. All features must be pre-computed and stored in parquet.

---

## 5. Tier Upgrade Strategy

### 5.1 Tier 1: Core Modules (LIVE → Production)

**Modules:** 48 modules
**Status:** Already LIVE, needs production validation

**Upgrade Process:**
1. Review all Tier 1 modules for production readiness
2. Add unit tests for any missing coverage
3. Validate feature store columns (no NaNs, correct ranges)
4. Run integration tests (gold standard backtest)
5. Merge to integration branch

**Validation Criteria:**
- All unit tests pass
- Integration tests pass
- Feature store validation clean (no NaNs)
- Gold standard metrics within ±5%

**No New Features:** Tier 1 is validation-only, no new features added.

---

### 5.2 Tier 2: Enhanced Modules (PARTIAL → LIVE)

**Modules:** 25 modules
**Status:** Partial implementation, needs completion

**Upgrade Process:**
1. For each PARTIAL module:
   - Complete missing logic
   - Add feature store columns if needed
   - Add unit tests
   - Add integration tests
   - Document usage
2. Validate feature coverage increases
3. Merge to integration branch

**Examples:**

**features/fib_retracement.py:**
- Current: Basic fib calculations present
- Missing: Validation tests, edge case handling
- Add: Unit tests for 0.236, 0.382, 0.5, 0.618, 0.786 levels
- Add: Integration test for fib-based entries

**features/swing_detection.py:**
- Current: Swing detection incomplete
- Missing: Multi-timeframe swing detection
- Add: 1H, 4H, 1D swing detection
- Add: Swing high/low persistence (no re-paint)

**psychology/fear_greed.py:**
- Current: F&G calculation present
- Missing: Integration into fusion engine
- Add: F&G weighting in fusion.py
- Add: Config parameter for F&G weight

**Validation Criteria:**
- Module transitions from PARTIAL → LIVE
- Feature coverage increases (e.g., +10 columns to feature store)
- Tests pass
- Documentation complete

---

### 5.3 Tier 3: Experimental Modules (IDEA → PARTIAL/LIVE)

**Modules:** 16 modules
**Status:** Specification only, needs implementation

**Upgrade Process:**
1. For each IDEA module:
   - Implement from specification
   - Add feature store columns
   - Add comprehensive tests
   - Validate with backtest
   - Document usage
2. Merge to integration branch

**Examples:**

**narrative/news_sentiment.py:**
- Spec: News sentiment analysis (Twitter, Reddit, news articles)
- Implementation: NLP pipeline (VADER, FinBERT, etc.)
- Features: `news_sentiment_score`, `news_sentiment_z`
- Tests: Sentiment score distribution, API integration
- Validation: Correlation with price moves

**ml/meta_optimizer.py:**
- Spec: Meta-optimization (optimize optimizer parameters)
- Implementation: Nested cross-validation, hyperparameter tuning
- Features: None (optimization-only)
- Tests: Optimizer convergence, parameter ranges
- Validation: Improved backtest metrics

**Validation Criteria:**
- Module transitions from IDEA → PARTIAL (if research) or LIVE (if production)
- New features validated (no NaNs, correct distribution)
- Tests pass
- Performance impact documented

---

## 6. Feature Store Schema Evolution

### 6.1 Current Schema (116 columns)

**Breakdown:**
- Tier 1 (LIVE): 81 columns
- Tier 2 (PARTIAL): 24 columns
- Tier 3 (EXPERIMENTAL): 11 columns

**Known Issues:**
- 3 columns all NaN (OI_CHANGE, oi_change_24h, oi_z)
- 1 column missing (liquidity_score)
- 24 columns partial coverage (macro 2024 only)

---

### 6.2 Target Schema (140+ columns)

**New Columns (Post-Upgrade):**

**Tier 2 Additions (Derived Features):**
1. `fvg_below` - FVG below current price (for S1)
2. `ob_retest` - Order block retest flag (for S2)
3. `rsi_divergence` - RSI bearish divergence (for S2)
4. `vol_fade` - Volume fading flag (for S2)
5. `wick_ratio` - Upper wick / total range (for S2)
6. `volume_spike` - Volume z-score > 2.0 (for S4)

**Tier 2 Additions (Fixed Columns):**
7. `oi_change_24h` - OI absolute change (24H) ← FIX
8. `oi_change_pct_24h` - OI percentage change (24H) ← FIX
9. `oi_z` - OI z-score (252H window) ← FIX
10. `liquidity_score` - Runtime liquidity score ← BACKFILL

**Tier 3 Additions (New Features):**
11. `fib_0.236` - Fib 0.236 retracement level
12. `fib_0.382` - Fib 0.382 retracement level
13. `fib_0.5` - Fib 0.5 retracement level
14. `fib_0.618` - Fib 0.618 retracement level
15. `fib_0.786` - Fib 0.786 retracement level
16. `swing_high_1h` - 1H swing high
17. `swing_low_1h` - 1H swing low
18. `swing_high_4h` - 4H swing high
19. `swing_low_4h` - 4H swing low
20. `fear_greed_index` - F&G composite score
21. `crowd_sentiment` - Crowd behavior metric
22. `support_level` - Nearest support level
23. `resistance_level` - Nearest resistance level
24. `trendline_slope` - Trendline slope (degrees)

**Total New Columns:** 24+
**Target Total:** 140+ columns

---

### 6.3 Schema Validation Rules

**All Columns Must Pass:**
1. **No NaN Values:** 100% coverage required
2. **Correct Data Types:** float64, int64, datetime64, etc.
3. **Valid Ranges:**
   - RSI: 0-100
   - ADX: 0-100
   - ATR: > 0
   - Z-scores: -5 to +5 (99.9% of data)
   - Probabilities: 0.0-1.0
4. **Timestamp Continuity:** No gaps in hourly data
5. **Logical Consistency:**
   - high >= low
   - high >= close >= low
   - volume >= 0

**Validation Script:**
```bash
python bin/validate_feature_store_schema.py \
  --input data/features_mtf/BTC_1H_2022-2024.parquet \
  --schema docs/FEATURE_STORE_SCHEMA_v2.md
```

**Output:**
```
✓ 140 columns validated
✓ 0 NaN values found
✓ All data types correct
✓ All ranges valid
✓ Timestamp continuity verified
✓ Logical consistency passed

Feature Store Status: VALIDATED
```

---

## 7. Regime Routing Architecture

### 7.1 GMM v3.2 Classifier

**Model:** `models/regime_gmm_v3.2_balanced.pkl`
**Algorithm:** Gaussian Mixture Model (3 components)
**Inputs:**
- DXY_Z (Dollar strength z-score)
- VIX_Z (Volatility z-score)
- funding_Z (Funding rate z-score)
- BTC.D_Z (BTC dominance z-score)

**Outputs:**
- `risk_on` - Bull market conditions
- `risk_off` - Bear market conditions
- `neutral` - Choppy / transitional conditions
- `crisis` - Extreme market stress

**Training Data:** 2022-2024 BTC 1H (26,236 samples)
**Accuracy:** 87.3% (validated on 2024 data)

---

### 7.2 Router v10 Logic

**Routing Rules:**

```python
if regime == 'risk_on':
    # Bull market: prioritize bull archetypes
    archetypes = {
        'A': 0.15,  # Trap Reversal
        'B': 0.12,  # Order Block Retest
        'C': 0.10,  # FVG Continuation
        'H': 0.20,  # Trap Within Trend (highest weight)
        'L': 0.15,  # Volume Exhaustion
        'M': 0.10,  # Ratio Coil Break
        # ... other bull archetypes
    }
    threshold = 0.60  # Lower threshold for bull conditions

elif regime == 'risk_off':
    # Bear market: prioritize bear archetypes
    archetypes = {
        'S1': 0.15,  # Liquidity Vacuum
        'S2': 0.25,  # Failed Rally (highest weight)
        'S4': 0.15,  # Distribution Climax
        'S5': 0.20,  # Long Squeeze Cascade
        'S8': 0.10,  # Volume Fade Chop
    }
    threshold = 0.55  # Lower threshold for bear conditions

elif regime == 'neutral':
    # Choppy market: evaluate both, higher threshold
    archetypes = {**bull_archetypes, **bear_archetypes}
    threshold = 0.70  # Higher threshold to reduce noise

else:  # crisis
    # Extreme stress: only bear archetypes, low threshold
    archetypes = bear_archetypes
    threshold = 0.50  # Low threshold to catch extreme moves
```

**Threshold Policy:**
- Regime-specific thresholds
- Dynamic adjustment based on volatility
- ADX-based scaling (weak vs strong trends)

---

## 8. Backward Compatibility Strategy

### 8.1 API Stability Guarantees

**MUST NOT BREAK:**
1. Feature store column names (existing 116 columns)
2. Config file structure (JSON schema)
3. Backtest engine CLI arguments
4. Module import paths (e.g., `from engine.fusion import k2_fusion`)
5. Function signatures for public APIs

**CAN CHANGE:**
1. Internal implementation details
2. Private functions (prefixed with `_`)
3. New columns in feature store (additive only)
4. New config parameters (optional, with defaults)

---

### 8.2 Deprecation Policy

**If Breaking Change Required:**
1. Add new API alongside old API
2. Mark old API as deprecated (docstring + warning)
3. Provide migration guide
4. Keep old API for minimum 2 versions
5. Remove old API in version N+2

**Example:**
```python
# Old API (deprecated)
def calculate_fusion_score(ctx):
    warnings.warn(
        "calculate_fusion_score is deprecated, use compute_fusion_score instead",
        DeprecationWarning
    )
    return compute_fusion_score(ctx)

# New API
def compute_fusion_score(ctx):
    # New implementation
    pass
```

---

### 8.3 Config Backward Compatibility

**Frozen Baselines:**
- `configs/frozen/btc_1h_v2_baseline.json` - Never modified
- `configs/frozen/btc_1h_v2_frontier.json` - Never modified

**Migration Path:**
1. Old configs continue to work (ignore new parameters)
2. New configs use new parameters (old parameters optional)
3. Config validator warns about deprecated parameters
4. Config migrator tool available: `bin/migrate_config_v1_to_v2.py`

---

## 9. Testing Architecture

### 9.1 Unit Testing Strategy

**Coverage Target:** 80%+ code coverage

**Test Structure:**
```
tests/unit/
├── test_calculators.py
├── test_fusion.py
├── test_regime_detector.py
├── test_router_v10.py
├── test_archetypes_bull.py
├── test_archetypes_bear.py
├── test_smc_order_blocks.py
├── test_smc_fvg.py
├── test_smc_bos.py
├── test_wyckoff_engine.py
├── test_features_fib.py
├── test_features_swing.py
└── ... (42 test modules)
```

**Test Pyramid:**
- Unit tests: 80% (fast, isolated)
- Integration tests: 15% (medium speed, end-to-end)
- Smoke tests: 5% (slow, production-like)

---

### 9.2 Integration Testing Strategy

**Gold Standard Backtests:**

**Test 1: BTC 1H Bull Market (2024-01-01 to 2024-09-30)**
```bash
pytest tests/integration/test_gold_standard_bull.py
```
**Expected Metrics:**
- Profit Factor: 1.10 - 1.22
- Trade Count: 297 - 363
- Max Drawdown: 3.96% - 4.84%
- Win Rate: 62% - 68%

**Test 2: BTC 1H Bear Market (2022-01-01 to 2022-12-31)**
```bash
pytest tests/integration/test_gold_standard_bear.py
```
**Expected Metrics:**
- Profit Factor: 1.05 - 1.15 (bear markets harder)
- Trade Count: 200 - 280
- Max Drawdown: 8% - 12%
- Win Rate: 55% - 62%

**Test 3: Multi-Regime (2022-01-01 to 2024-12-31)**
```bash
pytest tests/integration/test_gold_standard_full.py
```
**Expected Metrics:**
- Profit Factor: 1.80 - 2.50
- Trade Count: 600 - 800
- Max Drawdown: 10% - 14%
- Win Rate: 60% - 68%
- Sharpe Ratio: 1.5 - 2.0

---

### 9.3 CI/CD Guardrails

**Pre-Commit Checks:**
1. Blueprint vs Code Consistency Test
2. Dead Feature Detector
3. Config Consistency Test
4. Backtest Expectations Test

**Detailed in:** `docs/CI_GUARDRAILS_SPEC.md`

---

## 10. Rollback Architecture

### 10.1 Feature Store Backup Strategy

**Backup Before Each Phase:**
```bash
# Phase 1 backup
cp data/features_mtf/BTC_1H_2022-2024.parquet \
   data/features_mtf/BTC_1H_2022-2024_phase0_backup.parquet

# Phase 2 backup
cp data/features_mtf/BTC_1H_2022-2024.parquet \
   data/features_mtf/BTC_1H_2022-2024_phase1_backup.parquet

# ... etc
```

**Retention Policy:**
- Keep backups for all 4 phases
- Delete after successful merge to main
- Total storage: ~50 MB (5x 10 MB files)

---

### 10.2 Code Rollback Strategy

**Git Tags for Each Phase:**
```bash
git tag phase0_complete feature/ghost-modules-to-live-v2
git tag phase1_complete feature/tier1-core-modules
git tag phase2_complete feature/tier2-enhanced-modules
git tag phase3_complete feature/tier3-experimental-modules
git tag phase4_validated feature/ghost-modules-to-live-v2
```

**Rollback Procedure:**
```bash
# If Phase 2 fails, rollback to Phase 1
git reset --hard phase1_complete
cp data/features_mtf/BTC_1H_2022-2024_phase1_backup.parquet \
   data/features_mtf/BTC_1H_2022-2024.parquet
```

---

## 11. Performance Characteristics

### 11.1 Computational Complexity

**Feature Store Build:**
- Input: 26,236 rows × 6 columns (OHLCV + timestamp)
- Output: 26,236 rows × 140 columns
- Time: ~60 seconds (single-threaded)
- Memory: ~200 MB peak

**Backtest Execution:**
- Input: 26,236 rows × 140 columns
- Output: ~300-500 trades
- Time: ~30 seconds (single-threaded)
- Memory: ~300 MB peak

**Optimization (Optuna):**
- Trials: 100 trials
- Time per trial: ~30 seconds
- Total time: ~50 minutes (single-threaded)
- Parallelization: 4x speedup with 4 cores

---

### 11.2 Scalability Targets

**Single Asset (BTC 1H):**
- Current: 26,236 rows (3 years)
- Target: 50,000 rows (5 years)
- Performance: < 60 seconds backtest time

**Multi-Asset (BTC + ETH + SPY):**
- Current: 3 assets × 26,236 rows = 78,708 rows
- Target: 10 assets × 50,000 rows = 500,000 rows
- Performance: < 5 minutes backtest time (parallelized)

**Live Trading:**
- Latency: < 1 second (feature calculation to signal)
- Throughput: 1 signal per minute (1H timeframe)
- Memory: < 500 MB resident

---

## 12. Security & Compliance

### 12.1 Data Integrity

**Feature Store Checksums:**
```bash
# Generate checksum after feature store build
sha256sum data/features_mtf/BTC_1H_2022-2024.parquet > \
  data/features_mtf/BTC_1H_2022-2024.parquet.sha256

# Validate before backtest
sha256sum -c data/features_mtf/BTC_1H_2022-2024.parquet.sha256
```

**Config File Signing:**
```bash
# Sign frozen baseline config
gpg --sign configs/frozen/btc_1h_v2_baseline.json

# Verify before use
gpg --verify configs/frozen/btc_1h_v2_baseline.json.sig
```

---

### 12.2 Access Control

**Frozen Configs:**
- Read-only for all users
- Write access: System Architect only
- Version control: Git protected branches

**Feature Store:**
- Read-only for backtest engine
- Write access: Feature store builder only
- Backup before modification

---

## 13. Future Architecture Considerations

### 13.1 Modular Archetype Split (v3.0)

**Current:** Monolithic `logic_v2_adapter.py` (1441 lines)

**Target:**
```
engine/strategies/archetypes/
├── bull/
│   ├── archetype_a.py
│   ├── archetype_b.py
│   ├── archetype_c.py
│   └── ... (11 files)
└── bear/
    ├── archetype_s1.py
    ├── archetype_s2.py
    ├── archetype_s4.py
    └── ... (8 files)
```

**Benefits:**
- Easier to test individual archetypes
- Faster iteration (modify one file instead of 1441-line monolith)
- Clear ownership (one archetype = one file)

---

### 13.2 Real-Time Feature Store (v3.0)

**Current:** Batch feature store (parquet files)

**Target:** Real-time feature store (Redis / TimescaleDB)
- Streaming OHLCV ingestion
- Incremental feature calculation
- Sub-second latency

**Architecture:**
```
Binance WebSocket → Feature Calculator → Redis
                                         ↓
                                   Backtest Engine
                                         ↓
                                   Live Trader
```

---

## 14. References

- **Brain Blueprint:** `docs/BRAIN_BLUEPRINT_SNAPSHOT_v2.md`
- **Feature Schema:** `docs/FEATURE_STORE_SCHEMA_v2.md`
- **Dev Workflow:** `docs/DEV_WORKFLOW.md`
- **Risk Mitigation:** `docs/UPGRADE_RISKS_AND_ROLLBACK.md`
- **CI/CD Guardrails:** `docs/CI_GUARDRAILS_SPEC.md`
- **Architecture Overview:** `docs/ARCHITECTURE.md`

---

## Appendix: Technology Stack

**Core:**
- Python 3.9+
- Pandas 1.5+
- NumPy 1.23+

**ML:**
- Scikit-learn 1.2+
- Optuna 3.0+ (hyperparameter optimization)

**Data:**
- Parquet (feature store format)
- Pickle (model serialization)

**Testing:**
- Pytest 7.0+
- Pytest-cov (coverage reporting)

**CI/CD:**
- GitHub Actions (automated testing)
- Pre-commit hooks (linting, formatting)

**Version Control:**
- Git 2.30+
- Git LFS (large file storage for models)

---

## Version History

- **v2.0.0** (2025-11-19): Complete system architecture for Ghost → Live v2 upgrade
- **v1.0.0** (2025-11-14): Initial architecture documentation
