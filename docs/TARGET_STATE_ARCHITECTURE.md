# Bull Machine: TARGET STATE ARCHITECTURE

**Version:** 1.0.0
**Date:** 2025-12-03
**Status:** DESIGN COMPLETE - Implementation Roadmap Defined
**Purpose:** Define the production-ready architecture for Bull Machine trading engine

---

## Executive Summary

This document defines the **TARGET STATE** architecture for Bull Machine - a production-ready quantitative trading system designed for institutional-grade feature engineering, model-agnostic backtesting, and eventual live trading deployment.

**Key Design Principles:**
1. **Clean Versioned Feature Store** - Single source of truth, no NaNs, full lineage
2. **Model-Agnostic Architecture** - Support archetypes, ML models, ensembles equally
3. **Offline Feature Engineering** - Features computed ONCE, used by ALL models
4. **Horizontal Scalability** - Add models without modifying infrastructure
5. **Production-Ready** - Live trading capability, monitoring, rollback safety

**Benefits Over Current State:**
- Eliminates "runtime enrichment" technical debt (10+ scattered modules)
- Enables clean model comparison (apples-to-apples on same features)
- Unblocks ML/ensemble models (clean training data)
- Reduces backtest time by 60% (no duplicate feature computation)
- Prepares for live trading (streaming feature pipeline)

---

## Table of Contents

1. [Current State Assessment](#1-current-state-assessment)
2. [Target Architecture Overview](#2-target-architecture-overview)
3. [Data Layer - Versioned Feature Store](#3-data-layer---versioned-feature-store)
4. [Model Layer - Unified Interface](#4-model-layer---unified-interface)
5. [Backtesting Layer - Model Agnostic](#5-backtesting-layer---model-agnostic)
6. [Feature Pipeline - Build System](#6-feature-pipeline---build-system)
7. [Separation of Concerns](#7-separation-of-concerns)
8. [Migration Path](#8-migration-path)
9. [Production Benefits](#9-production-benefits)
10. [Appendices](#10-appendices)

---

## 1. Current State Assessment

### 1.1 Current Architecture (As-Is)

```
┌─────────────────────────────────────────────────────────────┐
│                CURRENT STATE (Partially Broken)              │
└─────────────────────────────────────────────────────────────┘

DATA LAYER:
  ├── BTC_1H_2022-2024.parquet (167 columns, ~10MB)
  │   ├── ✓ 140 columns clean
  │   ├── ✗ 27 columns with NaNs (OI, macro 2022-2023, etc.)
  │   └── ✗ No versioning (unclear what changed)
  │
  ├── Runtime Enrichment (10+ scattered modules)
  │   ├── liquidity_vacuum_runtime.py
  │   ├── failed_rally_runtime.py
  │   ├── funding_divergence_runtime.py
  │   └── long_squeeze_runtime.py
  │   └── → Computed ON-THE-FLY during backtest (slow!)
  │
  └── → PROBLEM: Features scattered across disk + runtime

MODEL LAYER:
  ├── ArchetypeModel (wraps legacy system)
  │   ├── Uses ArchetypeLogic (1441 lines monolith)
  │   ├── RuntimeContext (regime-aware parameters)
  │   └── → Works, but tightly coupled to archetypes
  │
  ├── SimpleClassifier (basic baseline)
  └── → No ML models (blocked by feature mess)

BACKTEST LAYER:
  ├── BacktestEngine (model-agnostic) ✓
  ├── BUT: Each model may compute different features
  └── → Can't compare apples-to-apples

ISSUES:
  ❌ Runtime enrichment = 10x slowdown + code duplication
  ❌ NaN values in 27 columns (blocking some archetypes)
  ❌ No feature versioning (can't reproduce old results)
  ❌ Can't add ML models (training data unclear)
  ❌ Can't compare models fairly (different features)
```

### 1.2 Technical Debt Summary

| Issue | Impact | Blockers |
|-------|--------|----------|
| **Runtime enrichment** | 10x slower backtests, code duplication | Adding new models requires copying enrichment logic |
| **NaN values (27 cols)** | Some archetypes fail silently | OI features (S5 blocked), macro 2022-2023 |
| **No versioning** | Can't reproduce results from 3 months ago | Features change, no lineage tracking |
| **Scattered features** | Hard to understand "what data exists" | No schema documentation |
| **Model coupling** | Can't add ML models cleanly | Training data pipeline unclear |

### 1.3 Current Feature Store

**File:** `data/features_mtf/BTC_1H_2022-01-01_to_2024-12-31.parquet`

**Stats:**
- Shape: 26,236 rows × 167 columns (~10 MB)
- Coverage: 140 columns clean (83.8%), 27 columns with NaNs (16.2%)
- Time Range: 2022-01-01 to 2024-12-31 (3 years hourly data)

**Known Issues:**
```python
# NaN Columns (27 total):
oi_change_24h           17,598 NaNs  (67.1% missing) ← BROKEN CALC
oi_change_pct_24h       17,598 NaNs  (67.1% missing) ← BROKEN CALC
oi_z                    17,597 NaNs  (67.1% missing) ← BROKEN CALC
funding                 17,475 NaNs  (66.6% missing) ← 2022-2023 missing
rv_20d, rv_60d          17,475 NaNs  (66.6% missing) ← 2022-2023 missing
VIX, DXY, YIELD_*          29 NaNs  (0.1% missing)  ← Data gaps
tf1h_fvg_*, tf1h_ob_*   8-14k NaNs  (30-53% missing) ← No FVG/OB detected
```

---

## 2. Target Architecture Overview

### 2.1 High-Level Design

```
┌─────────────────────────────────────────────────────────────┐
│                TARGET STATE (Production-Ready)               │
└─────────────────────────────────────────────────────────────┘

ETL LAYER (Build Once):
  ┌──────────────────────────────────────────┐
  │   Feature Engineering Pipeline           │
  │   ├── Raw OHLCV (Binance API)            │
  │   ├── Macro (DXY, VIX, Yields, BTC.D)    │
  │   ├── Derivatives (Funding, OI, Basis)   │
  │   └── Wyckoff Events (18 events)         │
  └──────────────────────────────────────────┘
                    ↓
  ┌──────────────────────────────────────────┐
  │  Versioned Feature Store (Parquet)       │
  │  ├── v1: BTC_1H_2022-2024_v1.parquet     │
  │  │   └── 167 columns (base features)     │
  │  ├── v2: BTC_1H_2022-2024_v2.parquet     │
  │  │   └── 195 columns (+ S1/S4 features)  │
  │  └── v3: BTC_1H_2022-2024_v3.parquet     │
  │      └── 220+ columns (+ Wyckoff/ML)     │
  │                                           │
  │  ✓ 100% coverage (no NaNs)               │
  │  ✓ Schema validation enforced            │
  │  ✓ Git-tracked manifests                 │
  └──────────────────────────────────────────┘
                    ↓
MODEL LAYER (Read Only):
  ┌──────────────────────────────────────────┐
  │  All models read SAME v2 parquet         │
  │  ├── ArchetypeModel (S1-S8, A-M)         │
  │  ├── SimpleClassifier (baselines)        │
  │  ├── LSTMModel (future)                  │
  │  ├── XGBoostModel (future)               │
  │  └── EnsembleModel (future)              │
  │                                           │
  │  ✓ NO runtime enrichment                 │
  │  ✓ All models use same features          │
  │  ✓ Apples-to-apples comparison           │
  └──────────────────────────────────────────┘
                    ↓
BACKTEST LAYER (Model Agnostic):
  ┌──────────────────────────────────────────┐
  │  Single Backtest Engine                  │
  │  ├── Load v2 features (once)             │
  │  ├── For each model:                     │
  │  │   ├── model.fit(train_data)           │
  │  │   ├── model.predict(test_data)        │
  │  │   └── evaluate(predictions)           │
  │  ├── Multi-model comparison dashboard    │
  │  └── Walk-forward validation framework   │
  │                                           │
  │  ✓ No model-specific logic               │
  │  ✓ Clean train/test/OOS split            │
  └──────────────────────────────────────────┘
                    ↓
LIVE LAYER (Future):
  ┌──────────────────────────────────────────┐
  │  Streaming Feature Pipeline              │
  │  ├── Binance WebSocket → feature calc    │
  │  ├── model.predict(latest_bar)           │
  │  ├── Order execution (CCXT)              │
  │  └── Monitoring + alerts                 │
  └──────────────────────────────────────────┘
```

### 2.2 Key Design Decisions

| Decision | Rationale | Trade-off |
|----------|-----------|-----------|
| **Offline feature engineering** | 10x faster backtests, clean model comparison | Larger disk usage (~50MB per asset) |
| **Versioned parquet files** | Reproducibility, lineage tracking | Need migration scripts |
| **No runtime enrichment** | Simplify codebase, eliminate duplication | Must rebuild store for new features |
| **BaseModel interface** | Model-agnostic backtester, easy to add ML | All models must conform to interface |
| **100% coverage requirement** | Eliminate silent failures from NaNs | Must backfill 2022-2023 macro data |

---

## 3. Data Layer - Versioned Feature Store

### 3.1 Feature Store Structure

```
data/features_mtf/
├── BTC_1H_2022-2024_v1.parquet      # Base features (167 cols)
├── BTC_1H_2022-2024_v2.parquet      # + Derived features (195 cols)
├── BTC_1H_2022-2024_v3.parquet      # + Wyckoff/ML features (220+ cols)
├── ETH_1H_2022-2024_v2.parquet      # Multi-asset support
│
├── manifests/                        # Git-tracked metadata
│   ├── v1_manifest.json             # Column list, build date, data sources
│   ├── v2_manifest.json             # Diff from v1, new columns, validation
│   └── v3_manifest.json
│
└── validation_reports/               # Automated validation
    ├── v2_validation_report.json    # NaN check, range check, consistency
    └── v2_vs_v1_diff.json           # What changed from v1 → v2
```

### 3.2 Feature Store v2 Schema (195 columns)

**Design Goal:** Support all current archetypes (S1-S8, A-M) with 100% coverage

**Tier 1: Base OHLCV + Technical (14 columns)**
```
OHLCV:         open, high, low, close, volume (5)
Technical:     atr_14, atr_20, adx_14, rsi_14 (4)
SMAs:          sma_20, sma_50, sma_100, sma_200 (4)
Timestamp:     timestamp (1)
```

**Tier 2: Multi-Timeframe Features (45 columns)**
```
1D Timeframe:  tf1d_wyckoff_*, tf1d_boms_*, tf1d_frvp_*, tf1d_pti_* (14)
4H Timeframe:  tf4h_internal_phase, tf4h_fusion_score, tf4h_squiggle_* (15)
1H Timeframe:  tf1h_pti_*, tf1h_frvp_*, tf1h_fakeout_*, tf1h_kelly_* (24)
Alignment:     mtf_alignment_ok, mtf_conflict_score, mtf_governor_veto (3)
```

**Tier 3: Smart Money Concepts (12 columns)**
```
Order Blocks:  tf1h_ob_high, tf1h_ob_low, is_bullish_ob, is_bearish_ob (4)
FVGs:          tf1h_fvg_high, tf1h_fvg_low, tf1h_fvg_present (3)
BOS:           tf1h_bos_bullish, tf1h_bos_bearish (2)
Bollinger:     tf1h_bb_high, tf1h_bb_low (2)
Swings:        is_swing_high, is_swing_low (2)
```

**Tier 4: Wyckoff Events (36 columns)**
```
Phase A:       wyckoff_sc, wyckoff_sc_confidence, wyckoff_bc, wyckoff_bc_confidence (4)
               wyckoff_ar, wyckoff_ar_confidence, wyckoff_as, wyckoff_as_confidence (4)
               wyckoff_st, wyckoff_st_confidence (2)
Phase B:       wyckoff_sos, wyckoff_sos_confidence, wyckoff_sow, wyckoff_sow_confidence (4)
Phase C:       wyckoff_spring_a, wyckoff_spring_a_confidence, wyckoff_spring_b, ... (8)
               wyckoff_ut, wyckoff_ut_confidence, wyckoff_utad, wyckoff_utad_confidence (4)
Phase D:       wyckoff_lps, wyckoff_lps_confidence, wyckoff_lpsy, wyckoff_lpsy_confidence (4)
Classification: wyckoff_phase_abc, wyckoff_sequence_position (2)
```

**Tier 5: Macro + Derivatives (30 columns)**
```
Macro Raw:     VIX, DXY, MOVE, YIELD_2Y, YIELD_10Y, BTC.D, USDT.D, TOTAL, TOTAL2 (9)
Funding:       funding, funding_rate, funding_Z (3)
OI:            oi, oi_change_24h, oi_change_pct_24h, oi_z (4)
Realized Vol:  RV_7, RV_20, RV_30, RV_60, rv_20d, rv_60d (6)
Z-Scores:      VIX_Z, DXY_Z, YC_Z, BTC.D_Z, USDT.D_Z (5)
Macro Logic:   macro_regime, macro_dxy_trend, macro_exit_recommended (3)
```

**Tier 6: Fusion + Composite Scores (12 columns)**
```
Fusion:        tf1h_fusion_score, tf4h_fusion_score, tf1d_fusion_score, k2_fusion_score (4)
K2 Delta:      k2_threshold_delta, k2_score_delta (2)
Liquidity:     liquidity_score, liquidity_velocity, liquidity_persistence (3)
Volume:        volume_z, volume_zscore, volume_ratio (3)
```

**Tier 7: NEW - Derived Features for Archetypes (28 columns)**
```
S1 Features:   fvg_below, liquidity_drain_pct, liquidity_velocity (3)
S2 Features:   wick_ratio, vol_fade, rsi_divergence (3)
S4 Features:   ob_retest, volume_spike, capitulation_depth (3)
S5 Features:   (uses existing OI features - no new columns) (0)
Multi-bar:     volume_climax_last_3b, wick_exhaustion_last_3b (2)
               liquidity_vacuum_score, liquidity_vacuum_fusion (2)
               volume_panic, crisis_composite, crisis_context (3)
Structural:    ob_confidence, ob_strength_bullish, ob_strength_bearish (3)
               bullish_displacement, bearish_displacement (2)
               adaptive_threshold, range_position (2)
Psychology:    macro_correlation, macro_correlation_score (2)
```

**Tier 8: FUTURE - ML/Temporal Features (18 columns)**
```
Fibonacci:     fib_0.236, fib_0.382, fib_0.5, fib_0.618, fib_0.786 (5)
Swings:        swing_high_1h, swing_low_1h, swing_high_4h, swing_low_4h (4)
Psychology:    fear_greed_index, crowd_sentiment (2)
Structure:     support_level, resistance_level, trendline_slope (3)
Temporal:      fib_time_cluster, gann_angle, temporal_confluence (4)
```

**TOTAL: v2 = 195 columns | v3 (future) = 220+ columns**

### 3.3 Feature Versioning Strategy

**Version Naming Convention:**
```
{ASSET}_{TIMEFRAME}_{START}-{END}_v{VERSION}.parquet

Examples:
  BTC_1H_2022-2024_v1.parquet   # Base features (167 cols)
  BTC_1H_2022-2024_v2.parquet   # + Derived features (195 cols)
  BTC_1H_2022-2024_v3.parquet   # + Wyckoff/ML features (220+ cols)
  ETH_1H_2022-2024_v2.parquet   # Multi-asset
```

**Version Manifest (JSON):**
```json
{
  "version": "v2",
  "created_at": "2025-12-03T10:00:00Z",
  "built_by": "bin/build_feature_store_v2.py",
  "source_version": "v1",
  "row_count": 26236,
  "column_count": 195,
  "coverage": 1.0,
  "validation_status": "PASSED",

  "new_columns": [
    "fvg_below", "ob_retest", "rsi_divergence", "vol_fade",
    "wick_ratio", "volume_spike", "liquidity_drain_pct", ...
  ],

  "removed_columns": [],

  "data_sources": {
    "ohlcv": "binance_api_2025-12-01",
    "macro": "tradingview_2025-12-01",
    "oi": "binance_futures_2025-12-01"
  },

  "validation_results": {
    "nan_check": "PASS (0 NaNs)",
    "range_check": "PASS (all valid)",
    "consistency_check": "PASS (high >= low, etc.)",
    "timestamp_check": "PASS (no gaps)"
  }
}
```

**Benefits:**
- Reproducibility: Can rebuild any historical version
- Lineage: Know exactly what changed between versions
- Rollback: If v2 is broken, revert to v1
- Documentation: Manifest is self-documenting

### 3.4 Feature Store Build Process

**Script:** `bin/build_feature_store_v2.py`

```python
#!/usr/bin/env python3
"""
Build Feature Store v2 - Complete feature pipeline.

Produces: BTC_1H_2022-2024_v2.parquet (195 columns, 100% coverage)

Steps:
1. Load raw OHLCV (Binance API)
2. Load macro data (TradingView, backfill 2022-2023)
3. Load derivatives (Binance, OKX - backfill OI)
4. Calculate Tier 1: Base technical indicators
5. Calculate Tier 2: MTF features (1H/4H/1D)
6. Calculate Tier 3: SMC features (OB, FVG, BOS)
7. Calculate Tier 4: Wyckoff events (18 events)
8. Calculate Tier 5: Macro features
9. Calculate Tier 6: Fusion scores
10. Calculate Tier 7: Derived archetype features (NEW)
11. Validate schema (100% coverage, correct ranges)
12. Export parquet + manifest
"""

# Pseudocode flow:
df = load_ohlcv('BTC', '1H', '2022-01-01', '2024-12-31')
macro = load_macro_backfilled('2022-01-01', '2024-12-31')  # NEW
oi = load_oi_backfilled('2022-01-01', '2024-12-31')       # NEW

# Tier 1-6: Existing features (from v1)
df = calculate_technical_indicators(df)
df = calculate_mtf_features(df)
df = calculate_smc_features(df)
df = calculate_wyckoff_events(df)
df = merge_macro_features(df, macro)
df = calculate_fusion_scores(df)

# Tier 7: NEW derived features
df = calculate_s1_features(df)  # fvg_below, liquidity_drain_pct, etc.
df = calculate_s2_features(df)  # wick_ratio, vol_fade, rsi_divergence
df = calculate_s4_features(df)  # ob_retest, volume_spike, etc.
df = calculate_s5_features(df)  # Uses existing OI features

# Validate schema
validate_no_nans(df)           # STRICT: No NaNs allowed
validate_ranges(df, SCHEMA_V2)  # All columns within expected ranges
validate_consistency(df)        # Logical checks (high >= low, etc.)

# Export
df.to_parquet('data/features_mtf/BTC_1H_2022-2024_v2.parquet')
save_manifest('manifests/v2_manifest.json', df, validation_results)
```

**Key Improvements from v1:**
1. **Backfill macro data (2022-2023):** Fixes 17,475 NaNs in funding, VIX, DXY, etc.
2. **Fix OI derivatives:** Fixes 17,598 NaNs in `oi_change_24h`, `oi_change_pct_24h`, `oi_z`
3. **Add derived features:** 28 new columns for S1/S2/S4/S5 archetypes
4. **Strict validation:** NO NaNs allowed - build fails if coverage < 100%

---

## 4. Model Layer - Unified Interface

### 4.1 BaseModel Interface (Already Implemented)

**File:** `engine/models/base.py`

```python
class BaseModel(ABC):
    """
    Abstract base class for all trading models.

    All models must implement:
    - fit(train_data)       # Calibrate model
    - predict(bar, position) # Generate signal
    - get_position_size()   # Risk management
    """

    @abstractmethod
    def fit(self, train_data: pd.DataFrame, **kwargs) -> None:
        """Train/calibrate model on historical data."""
        pass

    @abstractmethod
    def predict(self, bar: pd.Series, position: Optional[Position] = None) -> Signal:
        """Generate trading signal for current bar."""
        pass

    @abstractmethod
    def get_position_size(self, bar: pd.Series, signal: Signal) -> float:
        """Calculate position size for signal."""
        pass
```

**Signal Output:**
```python
@dataclass
class Signal:
    direction: Literal['long', 'short', 'hold']
    confidence: float  # 0.0 to 1.0
    entry_price: float
    stop_loss: Optional[float]
    take_profit: Optional[float]
    metadata: Dict[str, Any]  # Model-specific context
```

### 4.2 Model Implementations

**Current Models:**

```python
# 1. ArchetypeModel (wraps existing archetype system)
model = ArchetypeModel(
    config_path='configs/mvp/mvp_bear_market_v1.json',
    archetype_name='S2',  # Single archetype
    name='S2-Failed-Rally-Optimized'
)
model.fit(train_data)  # No-op (params from config)
signal = model.predict(bar, position)

# 2. SimpleClassifier (baseline)
model = SimpleClassifier(
    features=['rsi_14', 'adx_14', 'volume_z'],
    rules={'rsi_14': ('<', 30), 'adx_14': ('>', 25)}
)
model.fit(train_data)  # Optimize thresholds
signal = model.predict(bar)
```

**Future Models:**

```python
# 3. LSTMModel (deep learning)
model = LSTMModel(
    features=['close', 'volume', 'rsi_14', 'wyckoff_phase_abc'],
    lookback=24,  # 24-hour sequence
    hidden_size=128,
    num_layers=2
)
model.fit(train_data, epochs=50)
signal = model.predict(bar)

# 4. XGBoostModel (gradient boosting)
model = XGBoostModel(
    features=registry.get_tier_features(tier=2),  # All Tier 2 features
    n_estimators=200,
    max_depth=6
)
model.fit(train_data)
signal = model.predict(bar)

# 5. EnsembleModel (portfolio of models)
model = EnsembleModel(
    models=[
        ArchetypeModel('configs/s2.json', 'S2'),
        ArchetypeModel('configs/s5.json', 'S5'),
        XGBoostModel(features=...)
    ],
    weights=[0.4, 0.3, 0.3]  # Weighted voting
)
model.fit(train_data)
signal = model.predict(bar)
```

### 4.3 Model Feature Consumption

**PRINCIPLE:** All models read SAME v2 parquet (apples-to-apples comparison)

```python
# Load v2 feature store (once)
df = pd.read_parquet('data/features_mtf/BTC_1H_2022-2024_v2.parquet')

# Split data
train = df['2022':'2023']
test = df['2024']

# Model 1: ArchetypeModel
model1 = ArchetypeModel('configs/s2.json', 'S2')
model1.fit(train)  # Uses features defined in config

# Model 2: XGBoostModel
features = ['rsi_14', 'adx_14', 'wyckoff_phase_abc', 'liquidity_score']
model2 = XGBoostModel(features=features)
model2.fit(train[features])  # Selects subset of columns

# Model 3: LSTMModel
seq_features = ['close', 'volume', 'rsi_14', 'atr_14']
model3 = LSTMModel(features=seq_features, lookback=24)
model3.fit(train[seq_features])  # Time series data

# All models use SAME underlying data (v2 parquet)
# NO model computes features at runtime
```

### 4.4 Runtime Enrichment - ELIMINATED

**OLD WAY (Current - DEPRECATED):**
```python
# PROBLEM: Each archetype has runtime enrichment module
from engine.strategies.archetypes.bear.liquidity_vacuum_runtime import (
    LiquidityVacuumRuntimeFeatures
)

enricher = LiquidityVacuumRuntimeFeatures()
df = enricher.enrich_dataframe(df)  # Computes features during backtest (SLOW)

# Issues:
# - 10x slower (recompute same features for every backtest)
# - Code duplication (4+ enrichers with overlapping logic)
# - Hard to compare models (different features)
# - Blocks ML models (unclear what training data is)
```

**NEW WAY (Target - RECOMMENDED):**
```python
# SOLUTION: All features in v2 parquet (computed once)
df = pd.read_parquet('data/features_mtf/BTC_1H_2022-2024_v2.parquet')

# Features already present:
assert 'liquidity_drain_pct' in df.columns  # S1 feature
assert 'wick_ratio' in df.columns           # S2 feature
assert 'ob_retest' in df.columns            # S4 feature
assert 'oi_z' in df.columns                 # S5 feature

# Models just read features (no enrichment)
model = ArchetypeModel('configs/s1.json', 'S1')
model.fit(df)  # No enrichment needed - features already there
```

**Exception: Experimental Features**

```python
# ONLY exception: Experimental features not yet in v2
# Use experimental_features/ module (NOT production)

if feature_flags.EXPERIMENTAL_FIB_TIME_CLUSTERS:
    from engine.experimental_features.fib_time import calculate_fib_clusters
    df['fib_cluster_score'] = calculate_fib_clusters(df)

# Once validated, promote to v3:
# 1. Add to build_feature_store_v3.py
# 2. Rebuild parquet with new feature
# 3. Remove experimental flag
```

---

## 5. Backtesting Layer - Model Agnostic

### 5.1 Backtest Engine (Already Implemented)

**File:** `engine/backtesting/engine.py`

**Design:** Model-agnostic (doesn't know about archetypes, fusion, etc.)

```python
class BacktestEngine:
    """
    Model-agnostic backtesting engine.

    Works with ANY model that implements BaseModel interface.
    """

    def __init__(
        self,
        model: BaseModel,
        data: pd.DataFrame,
        initial_capital: float = 10000.0,
        commission_pct: float = 0.001
    ):
        self.model = model
        self.data = data
        self.capital = initial_capital
        self.commission = commission_pct

    def run(
        self,
        start: Optional[str] = None,
        end: Optional[str] = None
    ) -> BacktestResults:
        """
        Run backtest on historical data.

        Flow:
        1. Filter data by date range
        2. For each bar:
           a. signal = model.predict(bar, position)
           b. If entry signal: open position
           c. If in position: check stop/target
        3. Calculate metrics (PF, WR, Sharpe, etc.)
        4. Return BacktestResults
        """
        # Implementation already exists
        ...
```

**Usage:**
```python
# Load v2 feature store
df = pd.read_parquet('data/features_mtf/BTC_1H_2022-2024_v2.parquet')

# Model 1: S2 Archetype
model1 = ArchetypeModel('configs/s2.json', 'S2')
engine1 = BacktestEngine(model1, df)
results1 = engine1.run(start='2024-01-01', end='2024-12-31')

# Model 2: XGBoost
model2 = XGBoostModel(features=['rsi_14', 'adx_14', 'liquidity_score'])
model2.fit(df['2022':'2023'])
engine2 = BacktestEngine(model2, df)
results2 = engine2.run(start='2024-01-01', end='2024-12-31')

# Compare results
print(results1.summary())
print(results2.summary())
```

### 5.2 Multi-Model Comparison Framework

**File:** `engine/backtesting/comparison.py`

```python
class ModelComparison:
    """
    Compare multiple models on same data (apples-to-apples).

    Ensures:
    - All models use same v2 feature store
    - Same train/test split
    - Same evaluation metrics
    - Fair position sizing (same risk per trade)
    """

    def __init__(
        self,
        models: List[BaseModel],
        data: pd.DataFrame,
        train_period: Tuple[str, str],
        test_period: Tuple[str, str]
    ):
        self.models = models
        self.data = data
        self.train_start, self.train_end = train_period
        self.test_start, self.test_end = test_period

    def run_comparison(self) -> pd.DataFrame:
        """
        Run all models on same data, return comparison table.

        Returns DataFrame:
        | Model | Trades | WR% | PF | Sharpe | MaxDD | Total PnL |
        |-------|--------|-----|----|---------| ------|-----------|
        | S2    | 15     | 60% | 1.8| 1.2    | -8%   | $1,200    |
        | S5    | 8      | 75% | 2.4| 1.8    | -5%   | $1,800    |
        | XGB   | 42     | 55% | 1.5| 1.0    | -12%  | $900      |
        """
        results = []

        train = self.data[self.train_start:self.train_end]
        test = self.data[self.test_start:self.test_end]

        for model in self.models:
            # Fit on training data
            model.fit(train)

            # Backtest on test data
            engine = BacktestEngine(model, test)
            result = engine.run()

            # Collect metrics
            results.append({
                'Model': model.name,
                'Trades': result.total_trades,
                'WR%': result.win_rate,
                'PF': result.profit_factor,
                'Sharpe': result.metrics.get('sharpe', 0),
                'MaxDD': result.metrics.get('max_dd_pct', 0),
                'Total PnL': result.total_pnl
            })

        return pd.DataFrame(results)
```

**Usage:**
```python
# Define models to compare
models = [
    ArchetypeModel('configs/s2.json', 'S2', name='S2-Optimized'),
    ArchetypeModel('configs/s5.json', 'S5', name='S5-Optimized'),
    XGBoostModel(features=['rsi_14', 'adx_14', 'liquidity_score'], name='XGB-Baseline'),
    SimpleClassifier(features=['rsi_14'], rules={'rsi_14': ('<', 30)}, name='RSI-Oversold')
]

# Run comparison
comparison = ModelComparison(
    models=models,
    data=df,
    train_period=('2022-01-01', '2023-12-31'),
    test_period=('2024-01-01', '2024-12-31')
)

results = comparison.run_comparison()
print(results)

# Output:
# | Model         | Trades | WR%  | PF  | Sharpe | MaxDD | Total PnL |
# |---------------|--------|------|-----|--------|-------|-----------|
# | S2-Optimized  | 15     | 60.0 | 1.8 | 1.2    | -8.0  | 1200.00   |
# | S5-Optimized  | 8      | 75.0 | 2.4 | 1.8    | -5.0  | 1800.00   |
# | XGB-Baseline  | 42     | 55.0 | 1.5 | 1.0    | -12.0 | 900.00    |
# | RSI-Oversold  | 28     | 45.0 | 0.9 | 0.5    | -15.0 | -300.00   |

# Clear winner: S5 (highest PF, Sharpe, lowest DD)
```

### 5.3 Walk-Forward Validation

**File:** `engine/backtesting/validator.py`

```python
class WalkForwardValidator:
    """
    Walk-forward validation to prevent overfitting.

    Splits data into:
    - Train window (e.g., 6 months)
    - Test window (e.g., 1 month)
    - Rolls forward, refitting model each period

    Example:
    Train: 2022-01 to 2022-06 → Test: 2022-07
    Train: 2022-02 to 2022-07 → Test: 2022-08
    Train: 2022-03 to 2022-08 → Test: 2022-09
    ...
    """

    def run_walk_forward(
        self,
        model: BaseModel,
        data: pd.DataFrame,
        train_window_months: int = 6,
        test_window_months: int = 1,
        step_months: int = 1
    ) -> Dict:
        """
        Run walk-forward validation.

        Returns:
        - List of test period results
        - Aggregate metrics (avg PF, WR, Sharpe)
        - Stability metrics (std of PF across periods)
        """
        ...
```

---

## 6. Feature Pipeline - Build System

### 6.1 Feature Builder Architecture

```
bin/
├── build_feature_store_v2.py     # Master builder (v1 → v2)
├── build_feature_store_v3.py     # Future builder (v2 → v3)
├── validate_feature_store.py     # Schema validation
├── backfill_macro_data.py        # Backfill 2022-2023 macro
└── backfill_oi_data.py           # Backfill OI derivatives

engine/features/
├── base.py                       # OHLCV, ATR, RSI, ADX, SMAs
├── mtf.py                        # Multi-timeframe features
├── smc.py                        # Order blocks, FVGs, BOS
├── wyckoff.py                    # Wyckoff events (18 events)
├── macro.py                      # Macro features (DXY, VIX, etc.)
├── fusion.py                     # Fusion scores
├── derived.py                    # NEW: Derived archetype features
└── registry.py                   # Feature metadata + validation
```

### 6.2 Feature Calculation Modules

**Tier 1: Base Technical Indicators**

```python
# engine/features/base.py

def calculate_base_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate Tier 1 features: OHLCV + technical indicators.

    Inputs: OHLCV (5 columns)
    Outputs: 14 columns (OHLCV + ATR + ADX + RSI + SMAs)
    """
    df['atr_14'] = calculate_atr(df, period=14)
    df['atr_20'] = calculate_atr(df, period=20)
    df['adx_14'] = calculate_adx(df, period=14)
    df['rsi_14'] = calculate_rsi(df, period=14)
    df['sma_20'] = df['close'].rolling(20).mean()
    df['sma_50'] = df['close'].rolling(50).mean()
    df['sma_100'] = df['close'].rolling(100).mean()
    df['sma_200'] = df['close'].rolling(200).mean()

    # Fill NaNs with forward fill (warmup period)
    df = df.fillna(method='ffill')

    return df
```

**Tier 7: Derived Archetype Features (NEW)**

```python
# engine/features/derived.py

def calculate_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate Tier 7 derived features for S1/S2/S4 archetypes.

    Inputs: v1 features (167 columns)
    Outputs: +28 columns for archetype detection
    """

    # S1 Features (Liquidity Vacuum Reversal)
    df['fvg_below'] = (df['tf1h_fvg_high'] < df['close']).astype(int)
    df['liquidity_drain_pct'] = (
        (df['liquidity_score'] - df['liquidity_score'].rolling(168).mean()) /
        df['liquidity_score'].rolling(168).mean()
    )
    df['liquidity_velocity'] = df['liquidity_score'].diff(4) / 4

    # S2 Features (Failed Rally Rejection)
    df['wick_ratio'] = (df['high'] - df['close']) / (df['high'] - df['low'])
    df['vol_fade'] = (
        df['volume_z'] < df['volume_z'].shift(4)
    ).astype(int)
    df['rsi_divergence'] = detect_rsi_divergence(df)  # Price HH, RSI LH

    # S4 Features (Distribution)
    df['ob_retest'] = (
        (df['high'] >= df['tf1h_ob_low']) &
        (df['low'] <= df['tf1h_ob_high'])
    ).astype(int)
    df['volume_spike'] = (df['volume_z'] > 2.0).astype(int)

    # Multi-bar features (capitulation depth)
    df['capitulation_depth'] = (
        (df['close'].rolling(24).max() - df['close']) /
        df['close'].rolling(24).max()
    )
    df['volume_climax_last_3b'] = df['volume_z'].rolling(3).max()
    df['wick_exhaustion_last_3b'] = df['wick_ratio'].rolling(3).max()

    # Fill NaNs (edge cases)
    df = df.fillna(0)

    return df
```

### 6.3 Feature Store Builder Script

**File:** `bin/build_feature_store_v2.py`

```bash
#!/usr/bin/env python3
"""
Build Feature Store v2 - Complete Pipeline

Usage:
    python bin/build_feature_store_v2.py --asset BTC --timeframe 1H --start 2022-01-01 --end 2024-12-31

Outputs:
    - data/features_mtf/BTC_1H_2022-2024_v2.parquet
    - data/features_mtf/manifests/v2_manifest.json
    - data/features_mtf/validation_reports/v2_validation_report.json

Validation:
    - NO NaNs allowed (build fails if any NaN)
    - All columns within expected ranges
    - Logical consistency checks (high >= low, etc.)
    - Timestamp continuity (no gaps)

Time Estimate: 5-10 minutes for 3 years of hourly data
"""

# Step 1: Load raw data
df = load_ohlcv('BTC', '1H', '2022-01-01', '2024-12-31')
macro = load_macro_backfilled('2022-01-01', '2024-12-31')
oi = load_oi_backfilled('2022-01-01', '2024-12-31')

# Step 2: Calculate features (Tier 1-7)
df = calculate_base_features(df)           # Tier 1
df = calculate_mtf_features(df)            # Tier 2
df = calculate_smc_features(df)            # Tier 3
df = calculate_wyckoff_events(df)          # Tier 4
df = merge_macro_features(df, macro)       # Tier 5
df = calculate_fusion_scores(df)           # Tier 6
df = calculate_derived_features(df)        # Tier 7 (NEW)

# Step 3: Validate schema
validation_results = validate_feature_store(df, schema='v2')

if not validation_results['passed']:
    raise ValueError(f"Validation FAILED: {validation_results['errors']}")

# Step 4: Export
df.to_parquet('data/features_mtf/BTC_1H_2022-2024_v2.parquet')
save_manifest('manifests/v2_manifest.json', df, validation_results)

print("✓ Feature Store v2 built successfully")
print(f"  Rows: {len(df):,}")
print(f"  Columns: {len(df.columns)}")
print(f"  Coverage: {(1 - df.isnull().sum().sum() / (len(df) * len(df.columns))) * 100:.1f}%")
```

### 6.4 Validation Script

**File:** `bin/validate_feature_store.py`

```python
#!/usr/bin/env python3
"""
Validate Feature Store Schema

Usage:
    python bin/validate_feature_store.py --input data/features_mtf/BTC_1H_2022-2024_v2.parquet --schema v2

Checks:
1. No NaN values (100% coverage required)
2. All columns within expected ranges
3. Logical consistency (high >= low, etc.)
4. Timestamp continuity (no gaps)
5. Data type validation
"""

def validate_feature_store(df: pd.DataFrame, schema: str = 'v2') -> Dict:
    """
    Comprehensive feature store validation.

    Returns:
        {
            'passed': True/False,
            'errors': [...],
            'warnings': [...],
            'stats': {...}
        }
    """
    errors = []
    warnings = []

    # 1. No NaN check (CRITICAL)
    nan_counts = df.isnull().sum()
    if nan_counts.sum() > 0:
        errors.append(f"NaN values detected: {nan_counts[nan_counts > 0].to_dict()}")

    # 2. Range validation
    if schema == 'v2':
        range_checks = {
            'rsi_14': (0, 100),
            'adx_14': (0, 100),
            'liquidity_score': (0, 1),
            'wyckoff_sc_confidence': (0, 1),
            # ... all columns
        }

        for col, (min_val, max_val) in range_checks.items():
            if col in df.columns:
                if (df[col] < min_val).any() or (df[col] > max_val).any():
                    errors.append(f"{col} out of range [{min_val}, {max_val}]")

    # 3. Logical consistency
    if not (df['high'] >= df['low']).all():
        errors.append("high < low in some rows")
    if not (df['high'] >= df['close']).all():
        errors.append("high < close in some rows")

    # 4. Timestamp continuity
    time_diffs = df.index.to_series().diff()
    expected_diff = pd.Timedelta(hours=1)
    gaps = time_diffs[time_diffs != expected_diff].dropna()
    if len(gaps) > 0:
        warnings.append(f"{len(gaps)} timestamp gaps detected")

    return {
        'passed': len(errors) == 0,
        'errors': errors,
        'warnings': warnings,
        'stats': {
            'rows': len(df),
            'columns': len(df.columns),
            'coverage': 1 - (df.isnull().sum().sum() / (len(df) * len(df.columns)))
        }
    }
```

---

## 7. Separation of Concerns

### 7.1 Architecture Layers

```
┌─────────────────────────────────────────────────────────┐
│                    LAYER SEPARATION                      │
└─────────────────────────────────────────────────────────┘

ETL LAYER (Offline):
  ├── Responsibility: Raw data → Versioned features
  ├── Location: bin/build_feature_store_v2.py
  ├── Runs: Once per day (batch job)
  ├── Output: BTC_1H_2022-2024_v2.parquet
  └── NO cross-contamination with model layer

MODEL LAYER (Online/Offline):
  ├── Responsibility: Features → Signals
  ├── Location: engine/models/*.py
  ├── Reads: v2 parquet (READ-ONLY)
  ├── Outputs: Signal(direction, confidence, stop, target)
  └── NO feature computation (except experimental)

BACKTEST LAYER (Offline):
  ├── Responsibility: Signals → Performance metrics
  ├── Location: engine/backtesting/engine.py
  ├── Inputs: Model + v2 parquet
  ├── Outputs: BacktestResults (trades, PnL, metrics)
  └── NO model-specific logic (model-agnostic)

LIVE LAYER (Online - Future):
  ├── Responsibility: Streaming features → Orders
  ├── Location: engine/live/executor.py
  ├── Inputs: Binance WebSocket → real-time features
  ├── Outputs: CCXT orders (limit/market)
  └── NO backtesting logic (separate execution path)
```

### 7.2 Data Flow (No Cross-Contamination)

```
DAY 1: Feature Engineering (Batch)
├── 1. Raw OHLCV (Binance API)
├── 2. Macro data (TradingView)
├── 3. Calculate ALL features (195 columns)
├── 4. Validate schema (no NaNs, correct ranges)
└── 5. Export: BTC_1H_2022-2024_v2.parquet

DAY 2: Model Training (Batch)
├── 1. Load v2 parquet (READ-ONLY)
├── 2. Split: train (2022-2023), test (2024)
├── 3. For each model:
│   ├── model.fit(train_data)
│   └── Checkpoint: model_S2_2025-12-03.pkl
└── NO feature computation

DAY 3: Backtesting (Batch)
├── 1. Load v2 parquet (READ-ONLY)
├── 2. Load trained models (checkpoints)
├── 3. For each model:
│   ├── signals = model.predict(test_data)
│   ├── trades = execute_signals(signals)
│   └── metrics = evaluate(trades)
└── Compare results (apples-to-apples)

DAY 4+: Live Trading (Streaming - Future)
├── 1. Binance WebSocket → real-time OHLCV
├── 2. Calculate features (streaming pipeline)
├── 3. Load best model (from backtest)
├── 4. signal = model.predict(latest_bar)
├── 5. If entry: place_order(signal)
└── Monitor position, update stops
```

### 7.3 Module Responsibilities

| Module | Reads | Writes | Computes | Can Modify |
|--------|-------|--------|----------|------------|
| **build_feature_store_v2.py** | Raw OHLCV, Macro, OI | v2 parquet, manifest | ALL features | Feature store |
| **ArchetypeModel** | v2 parquet | None (read-only) | None (uses existing features) | Config params |
| **XGBoostModel** | v2 parquet | Model checkpoint | None (trains on features) | Model weights |
| **BacktestEngine** | v2 parquet | BacktestResults | Trade execution logic | None |
| **Live Executor** | Binance WebSocket | Orders (CCXT) | Streaming features | None |

**PRINCIPLE:** Features are computed ONCE (ETL layer), used MANY times (all models)

---

## 8. Migration Path

### 8.1 Migration Strategy (Incremental)

**Phase 0: Planning (COMPLETE)**
- [x] Document current state (167 columns, 27 with NaNs)
- [x] Define target state (195 columns, 100% coverage)
- [x] Design v2 schema
- [x] Create migration plan

**Phase 1: Fix Broken Columns (Week 1)**
```bash
# Goal: Eliminate 27 NaN columns from v1

Tasks:
1. Backfill macro data (2022-2023)
   - Script: bin/backfill_macro_data.py
   - Sources: TradingView API (DXY, VIX, Yields)
   - Output: data/macro/macro_backfilled_2022-2023.csv

2. Fix OI derivatives
   - Script: bin/backfill_oi_data.py
   - Sources: Binance Futures API, OKX API
   - Calculate: oi_change_24h, oi_change_pct_24h, oi_z

3. Rebuild v1 parquet (with fixes)
   - Script: bin/build_feature_store_v1_fixed.py
   - Output: BTC_1H_2022-2024_v1_fixed.parquet
   - Validation: validate_feature_store.py --strict

Success Criteria:
✓ 0 NaN columns (100% coverage)
✓ All 167 columns valid ranges
✓ Logical consistency checks pass
```

**Phase 2: Add Derived Features (Week 2)**
```bash
# Goal: Add 28 derived columns for S1/S2/S4/S5

Tasks:
1. Implement derived feature calculator
   - File: engine/features/derived.py
   - Functions:
     * calculate_s1_features() # fvg_below, liquidity_drain_pct, etc.
     * calculate_s2_features() # wick_ratio, vol_fade, rsi_divergence
     * calculate_s4_features() # ob_retest, volume_spike, etc.
     * calculate_multi_bar_features() # capitulation_depth, etc.

2. Build v2 parquet
   - Script: bin/build_feature_store_v2.py
   - Input: v1_fixed.parquet
   - Output: BTC_1H_2022-2024_v2.parquet (195 columns)

3. Validate v2
   - Script: bin/validate_feature_store.py --schema v2
   - Check: 195 columns, 100% coverage

Success Criteria:
✓ 195 columns (167 + 28 new)
✓ 100% coverage (no NaNs)
✓ S1/S2/S4 archetypes work without runtime enrichment
```

**Phase 3: Eliminate Runtime Enrichment (Week 3)**
```bash
# Goal: Remove all runtime enrichment modules

Tasks:
1. Update archetype configs
   - Point to v2 features (not runtime-computed)
   - Example: 'fvg_below' instead of 'compute_fvg_below()'

2. Deprecate runtime modules
   - Mark DEPRECATED:
     * liquidity_vacuum_runtime.py
     * failed_rally_runtime.py
     * funding_divergence_runtime.py
     * long_squeeze_runtime.py

3. Update backtest scripts
   - Remove enrichment calls
   - Load v2 parquet directly

4. Run regression tests
   - Script: pytest tests/integration/test_v2_vs_v1.py
   - Verify: Same results with v2 (no runtime enrichment)

Success Criteria:
✓ All archetypes work with v2 parquet (no enrichment)
✓ Backtest speedup: 60% faster (measured)
✓ No behavioral regressions (same PF/WR)
```

**Phase 4: Model Comparison Framework (Week 4)**
```bash
# Goal: Enable apples-to-apples model comparison

Tasks:
1. Implement ModelComparison class
   - File: engine/backtesting/comparison.py
   - Run all models on same v2 data
   - Generate comparison table

2. Create comparison dashboard
   - Script: bin/compare_models.py
   - Inputs: List of models
   - Output: HTML report with charts

3. Run baseline comparison
   - Models: S2, S5, SimpleClassifier (RSI), XGBoost (basic)
   - Report: docs/backtests/model_comparison_baseline.md

Success Criteria:
✓ 4+ models compared on same data
✓ Clear winner identified (by PF/Sharpe)
✓ HTML dashboard with equity curves
```

**Phase 5: ML Model Integration (Week 5-6)**
```bash
# Goal: Add first ML model (XGBoost)

Tasks:
1. Implement XGBoostModel
   - File: engine/models/xgboost_model.py
   - Inherits: BaseModel
   - Features: Tier 1-3 features (60+ columns)

2. Train on 2022-2023 data
   - Script: bin/train_xgboost_baseline.py
   - Hyperparams: n_estimators=200, max_depth=6
   - Checkpoint: models/xgboost_baseline_2025-12-03.pkl

3. Backtest on 2024 data
   - Compare to S2/S5 archetypes
   - Report: docs/backtests/xgboost_vs_archetypes.md

Success Criteria:
✓ XGBoost trains successfully on v2 data
✓ Comparable performance to archetypes (PF > 1.5)
✓ Feature importance analysis (top 10 features)
```

**Phase 6: Live Trading Prep (Week 7-8)**
```bash
# Goal: Prepare for paper trading deployment

Tasks:
1. Implement streaming feature pipeline
   - File: engine/live/streaming_features.py
   - Input: Binance WebSocket (real-time OHLCV)
   - Output: Real-time feature calculation (1H bars)

2. Create paper trading executor
   - File: engine/live/paper_executor.py
   - Load best model from backtest
   - Simulate orders (no real money)

3. Run 2-week paper trading test
   - Monitor: signal quality, execution latency
   - Report: docs/live/paper_trading_results.md

Success Criteria:
✓ Streaming features match backtest features (validated)
✓ Paper trading matches backtest results (within 5%)
✓ Average latency < 100ms (bar close → signal)
```

### 8.2 Backward Compatibility

**Strategy:** NO breaking changes - use adapters/facades

```python
# OLD CODE (still works):
from engine.strategies.archetypes.bear.liquidity_vacuum_runtime import (
    LiquidityVacuumRuntimeFeatures
)
enricher = LiquidityVacuumRuntimeFeatures()
df = enricher.enrich_dataframe(df)

# NEW CODE (recommended):
df = pd.read_parquet('data/features_mtf/BTC_1H_2022-2024_v2.parquet')
# Features already present, no enrichment needed

# TRANSITION ADAPTER (for backward compat):
class LiquidityVacuumRuntimeFeatures:
    """DEPRECATED: Use v2 feature store instead."""

    def enrich_dataframe(self, df):
        # Check if features already present (v2)
        if 'fvg_below' in df.columns:
            warnings.warn(
                "Features already present in v2 store. "
                "Runtime enrichment is deprecated and will be removed in v3.",
                DeprecationWarning
            )
            return df  # No-op

        # Fallback: Compute features (legacy path)
        return self._legacy_enrich(df)
```

**Deprecation Timeline:**
- **v2.0 (Dec 2025):** Runtime enrichment deprecated (warnings)
- **v2.5 (Mar 2026):** Runtime enrichment raises errors (must use v2 store)
- **v3.0 (Jun 2026):** Runtime enrichment removed entirely

### 8.3 Rollback Plan

**If v2 breaks production:**

```bash
# 1. Identify issue
pytest tests/integration/test_v2_validation.py  # Fails

# 2. Revert to v1_fixed
ln -sf BTC_1H_2022-2024_v1_fixed.parquet BTC_1H_2022-2024.parquet

# 3. Re-enable runtime enrichment
git revert <commit-hash>  # Revert "Remove runtime enrichment"

# 4. Investigate root cause
python bin/debug_v2_diff.py --v1 v1_fixed.parquet --v2 v2.parquet

# 5. Fix v2 builder
# Edit: bin/build_feature_store_v2.py
# Test: pytest tests/unit/test_derived_features.py

# 6. Rebuild v2
python bin/build_feature_store_v2.py --asset BTC --validate-strict

# 7. Re-test
pytest tests/integration/test_v2_validation.py  # Pass

# 8. Re-deploy v2
ln -sf BTC_1H_2022-2024_v2.parquet BTC_1H_2022-2024.parquet
```

**Safety Measures:**
1. **Ground Truth Results:** Keep frozen backtest results for regression testing
2. **Git-Tracked Configs:** All configs versioned, can rollback
3. **Parquet Backups:** Keep v1_fixed.parquet as fallback
4. **Automated Tests:** 55+ tests catch regressions before production

---

## 9. Production Benefits

### 9.1 Performance Improvements

| Metric | Current (v1 + Runtime) | Target (v2 Offline) | Improvement |
|--------|------------------------|---------------------|-------------|
| **Backtest Time** | 15 min (single archetype) | 6 min (all models) | **60% faster** |
| **Feature Recomputation** | Every backtest (10x waste) | Once per day (batch) | **90% reduction** |
| **Model Comparison** | Sequential (30 min for 4 models) | Parallel (8 min for 4 models) | **73% faster** |
| **Storage** | 10 MB (v1) + runtime code | 12 MB (v2) only | **Simpler** |
| **Code Complexity** | 10+ enrichment modules | 0 enrichment modules | **-1000 LOC** |

### 9.2 Developer Experience

**Before (v1 + Runtime):**
```python
# Want to add new archetype? Must:
# 1. Write feature logic in archetype detector
# 2. Write DUPLICATE logic in runtime enricher
# 3. Update backtest script to call enricher
# 4. Hope features match between detector and enricher

# Example: Adding S6 (Alt Rotation Down)
class S6Detector:
    def detect(self, df):
        # Need 'alt_rotation_score'
        return df['alt_rotation_score'] > 0.7  # ERROR: Column doesn't exist

class S6RuntimeEnricher:
    def enrich(self, df):
        # Must duplicate calculation
        df['alt_rotation_score'] = ...  # Copy-paste from detector

# Issues:
# - Code duplication (2 places to update)
# - Easy to get out of sync (bugs)
# - Slow (recompute every backtest)
```

**After (v2 Offline):**
```python
# Want to add new archetype? Easy:
# 1. Add feature to bin/build_feature_store_v3.py
# 2. Rebuild parquet (once)
# 3. Use feature in archetype detector

# Example: Adding S6 (Alt Rotation Down)
# Step 1: Add to feature builder
def calculate_v3_features(df):
    df['alt_rotation_score'] = (
        df['TOTAL2_RET'] - df['BTC_RET']
    ) / df['RV_20']
    return df

# Step 2: Rebuild parquet
python bin/build_feature_store_v3.py

# Step 3: Use in detector
class S6Detector:
    def detect(self, df):
        return df['alt_rotation_score'] > 0.7  # Works!

# Benefits:
# - Single source of truth (v3 parquet)
# - No code duplication
# - Fast (feature precomputed)
```

### 9.3 ML Model Enablement

**Before (v1 + Runtime):**
```python
# Can't train ML models - unclear what training data is

# Q: What features should XGBoost use?
# A: Uhh... whatever's in v1 parquet + whatever runtime enrichers compute?

# Q: How to get training data for 2022-2023?
# A: Must run backtests on historical data, capture enriched features (SLOW)

# Q: Can I compare XGBoost to S2 archetype fairly?
# A: No - they use different features (enrichers may compute differently)

# Result: ML models BLOCKED
```

**After (v2 Offline):**
```python
# ML models unblocked - clean training data

# Load v2 parquet (ALL features, 100% coverage)
df = pd.read_parquet('data/features_mtf/BTC_1H_2022-2024_v2.parquet')

# Train XGBoost
features = registry.get_tier_features(tier=2)  # 45 MTF features
X_train = df[features]['2022':'2023']
y_train = df['future_return_24h']['2022':'2023']  # Target

model = XGBoostModel(features=features)
model.fit(X_train, y_train)

# Compare to S2 archetype (apples-to-apples)
comparison = ModelComparison(
    models=[model, ArchetypeModel('configs/s2.json', 'S2')],
    data=df,
    train_period=('2022', '2023'),
    test_period=('2024', '2024')
)

# Both models use SAME v2 features → fair comparison
results = comparison.run_comparison()
```

### 9.4 Live Trading Preparation

**Before (v1 + Runtime):**
```
Live Trading Flow (MESSY):
1. Binance WebSocket → real-time OHLCV
2. Calculate base features (ATR, RSI, etc.)
3. Run 10+ runtime enrichers (S1, S2, S4, S5)
4. Run archetype detector
5. Generate signal

Issues:
- Latency: 10+ enrichers = slow (500ms+)
- Inconsistency: Runtime features may differ from backtest
- Complexity: 10+ modules to monitor in production
```

**After (v2 Offline):**
```
Live Trading Flow (CLEAN):
1. Binance WebSocket → real-time OHLCV
2. Streaming feature pipeline (same logic as v2 builder)
3. model.predict(latest_bar) → signal

Benefits:
- Fast: Single feature pipeline (< 100ms)
- Consistent: Same features as backtest (validated)
- Simple: One module to monitor

Validation:
- Test: Compare streaming features to v2 parquet (should match)
- Paper trade: Run 2 weeks, compare to backtest results (within 5%)
```

---

## 10. Appendices

### Appendix A: Complete v2 Schema (195 Columns)

**Tier 1: Base (14 columns)**
```
timestamp, open, high, low, close, volume,
atr_14, atr_20, adx_14, rsi_14,
sma_20, sma_50, sma_100, sma_200
```

**Tier 2: MTF (45 columns)**
```
tf1d_wyckoff_score, tf1d_wyckoff_phase, tf1d_boms_detected, tf1d_boms_strength,
tf1d_boms_direction, tf1d_range_outcome, tf1d_range_confidence, tf1d_range_direction,
tf1d_frvp_poc, tf1d_frvp_va_high, tf1d_frvp_va_low, tf1d_frvp_position,
tf1d_pti_score, tf1d_pti_reversal,

tf4h_internal_phase, tf4h_external_trend, tf4h_structure_alignment,
tf4h_conflict_score, tf4h_squiggle_stage, tf4h_squiggle_direction,
tf4h_squiggle_entry_window, tf4h_squiggle_confidence, tf4h_choch_flag,
tf4h_boms_direction, tf4h_boms_displacement, tf4h_fvg_present,
tf4h_range_outcome, tf4h_range_breakout_strength, tf4h_fusion_score,

tf1h_pti_score, tf1h_pti_trap_type, tf1h_pti_confidence, tf1h_pti_reversal_likely,
tf1h_frvp_poc, tf1h_frvp_va_high, tf1h_frvp_va_low, tf1h_frvp_position,
tf1h_frvp_distance_to_poc, tf1h_fakeout_detected, tf1h_fakeout_intensity,
tf1h_fakeout_direction, tf1h_kelly_atr_pct, tf1h_kelly_volatility_ratio,
tf1h_kelly_hint,

mtf_alignment_ok, mtf_conflict_score, mtf_governor_veto
```

**Tier 3: SMC (12 columns)**
```
tf1h_ob_high, tf1h_ob_low, tf1h_fvg_high, tf1h_fvg_low, tf1h_fvg_present,
tf1h_bos_bullish, tf1h_bos_bearish, tf1h_bb_high, tf1h_bb_low,
is_bullish_ob, is_bearish_ob, is_swing_high, is_swing_low
```

**Tier 4: Wyckoff (36 columns)**
```
wyckoff_sc, wyckoff_sc_confidence, wyckoff_bc, wyckoff_bc_confidence,
wyckoff_ar, wyckoff_ar_confidence, wyckoff_as, wyckoff_as_confidence,
wyckoff_st, wyckoff_st_confidence,
wyckoff_sos, wyckoff_sos_confidence, wyckoff_sow, wyckoff_sow_confidence,
wyckoff_spring_a, wyckoff_spring_a_confidence, wyckoff_spring_b, wyckoff_spring_b_confidence,
wyckoff_ut, wyckoff_ut_confidence, wyckoff_utad, wyckoff_utad_confidence,
wyckoff_lps, wyckoff_lps_confidence, wyckoff_lpsy, wyckoff_lpsy_confidence,
wyckoff_phase_abc, wyckoff_sequence_position
```

**Tier 5: Macro (30 columns)**
```
VIX, DXY, MOVE, YIELD_2Y, YIELD_10Y, BTC.D, USDT.D, TOTAL, TOTAL2,
funding, funding_rate, funding_Z,
oi, oi_change_24h, oi_change_pct_24h, oi_z,
RV_7, RV_20, RV_30, RV_60, rv_20d, rv_60d,
VIX_Z, DXY_Z, YC_Z, BTC.D_Z, USDT.D_Z,
macro_regime, macro_dxy_trend, macro_exit_recommended
```

**Tier 6: Fusion (12 columns)**
```
tf1h_fusion_score, tf4h_fusion_score, tf1d_fusion_score, k2_fusion_score,
k2_threshold_delta, k2_score_delta,
liquidity_score, liquidity_velocity, liquidity_persistence,
volume_z, volume_zscore, volume_ratio
```

**Tier 7: Derived (28 columns) - NEW**
```
fvg_below, liquidity_drain_pct, liquidity_velocity,
wick_ratio, vol_fade, rsi_divergence,
ob_retest, volume_spike, capitulation_depth,
volume_climax_last_3b, wick_exhaustion_last_3b,
liquidity_vacuum_score, liquidity_vacuum_fusion,
volume_panic, crisis_composite, crisis_context,
ob_confidence, ob_strength_bullish, ob_strength_bearish,
bullish_displacement, bearish_displacement,
adaptive_threshold, range_position,
macro_correlation, macro_correlation_score
```

**Tier 8: Future (18 columns) - v3**
```
fib_0.236, fib_0.382, fib_0.5, fib_0.618, fib_0.786,
swing_high_1h, swing_low_1h, swing_high_4h, swing_low_4h,
fear_greed_index, crowd_sentiment,
support_level, resistance_level, trendline_slope,
fib_time_cluster, gann_angle, temporal_confluence
```

**TOTAL: 195 columns (v2) | 213 columns (v3)**

### Appendix B: Key Files Reference

**Feature Store:**
```
data/features_mtf/BTC_1H_2022-2024_v2.parquet   # Main feature store
data/features_mtf/manifests/v2_manifest.json     # Version metadata
```

**Build Scripts:**
```
bin/build_feature_store_v2.py                    # v2 builder
bin/validate_feature_store.py                    # Validator
bin/backfill_macro_data.py                       # Macro backfill
bin/backfill_oi_data.py                          # OI backfill
```

**Feature Modules:**
```
engine/features/base.py                          # Tier 1 features
engine/features/mtf.py                           # Tier 2 features
engine/features/smc.py                           # Tier 3 features
engine/features/wyckoff.py                       # Tier 4 features
engine/features/macro.py                         # Tier 5 features
engine/features/fusion.py                        # Tier 6 features
engine/features/derived.py                       # Tier 7 features (NEW)
engine/features/registry.py                      # Feature metadata
```

**Model Layer:**
```
engine/models/base.py                            # BaseModel interface
engine/models/archetype_model.py                 # Archetype wrapper
engine/models/simple_classifier.py               # Baseline models
engine/models/xgboost_model.py                   # ML model (future)
engine/models/ensemble_model.py                  # Ensemble (future)
```

**Backtesting:**
```
engine/backtesting/engine.py                     # Backtest engine
engine/backtesting/comparison.py                 # Model comparison
engine/backtesting/validator.py                  # Walk-forward validation
engine/backtesting/metrics.py                    # Performance metrics
```

### Appendix C: Validation Checklist

**Pre-Release Validation (Before v2 Goes Live):**

- [ ] **Feature Store Validation**
  - [ ] 195 columns present
  - [ ] 0 NaN values (100% coverage)
  - [ ] All columns within expected ranges
  - [ ] Logical consistency (high >= low, etc.)
  - [ ] Timestamp continuity (no gaps)

- [ ] **Backtest Regression**
  - [ ] S2 archetype: Same results as v1 (within 1%)
  - [ ] S5 archetype: Same results as v1 (within 1%)
  - [ ] All archetypes: No crashes on v2 data

- [ ] **Performance Benchmarks**
  - [ ] Backtest time: 60% faster than v1+runtime
  - [ ] Memory usage: < 200 MB for 3 years data
  - [ ] Model comparison: 4 models in < 10 minutes

- [ ] **Code Quality**
  - [ ] All tests pass (55+ tests)
  - [ ] No deprecation warnings
  - [ ] Code coverage > 80%

- [ ] **Documentation**
  - [ ] v2 schema documented (this file)
  - [ ] Migration guide written
  - [ ] Rollback plan tested

### Appendix D: Future Roadmap (Beyond v2)

**v3.0 (Q2 2026): ML Model Integration**
- Add 18 columns (Tier 8): Fibonacci, swings, psychology, structure
- Implement LSTMModel, XGBoostModel, EnsembleModel
- Feature importance analysis
- Online learning adaptation

**v4.0 (Q3 2026): Live Trading**
- Streaming feature pipeline
- Paper trading executor
- Real-money trading (manual approval)
- Monitoring + alerting dashboard

**v5.0 (Q4 2026): Multi-Asset**
- Extend to ETH, SOL, multi-crypto
- Cross-asset correlation features
- Portfolio optimization
- Risk parity weighting

**v6.0 (Q1 2027): Advanced Features**
- Temporal fusion (Fib time clusters, Gann cycles)
- Alternative data (Reddit sentiment, GitHub commits)
- Order flow analysis (Level 2 data)
- High-frequency features (1min bars)

---

## Conclusion

This TARGET STATE architecture defines a **production-ready** trading system with:

1. **Clean Data Layer:** Versioned feature stores (v2: 195 cols, 100% coverage)
2. **Model-Agnostic Layer:** BaseModel interface supports archetypes, ML, ensembles
3. **Offline Feature Engineering:** Compute once, use many times (60% faster)
4. **Horizontal Scalability:** Add models without modifying infrastructure
5. **Live Trading Ready:** Streaming pipeline, paper trading, monitoring

**Key Benefits:**
- Eliminates runtime enrichment debt (-1000 LOC, 60% faster)
- Enables ML models (clean training data)
- Fair model comparison (apples-to-apples)
- Reproducibility (versioned features + manifests)
- Live trading capable (streaming pipeline designed)

**Next Steps:**
1. Implement Phase 1 (fix 27 NaN columns)
2. Build v2 feature store (195 columns)
3. Eliminate runtime enrichment
4. Add ML model comparison
5. Prepare for live trading

**Timeline:** 8 weeks to v2 production-ready

---

**Document Status:** APPROVED - Ready for Implementation
**Last Updated:** 2025-12-03
**Authors:** Claude Code (Architect), Raymond Ghandchi (Product Owner)
