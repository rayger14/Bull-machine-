# Bull Machine: TARGET STATE ARCHITECTURE DIAGRAMS

**Version:** 1.0.0
**Date:** 2025-12-03
**Companion Document:** [TARGET_STATE_ARCHITECTURE.md](./TARGET_STATE_ARCHITECTURE.md)

This document provides ASCII diagrams for the target architecture.

---

## 1. System Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        BULL MACHINE - TARGET STATE                          │
│                        Production-Ready Trading Engine                      │
└─────────────────────────────────────────────────────────────────────────────┘

                              ┌──────────────────┐
                              │   Raw Data       │
                              │   Sources        │
                              └────────┬─────────┘
                                       │
        ┌──────────────┬───────────────┼───────────────┬──────────────┐
        │              │               │               │              │
        ▼              ▼               ▼               ▼              ▼
   ┌────────┐    ┌─────────┐    ┌──────────┐    ┌─────────┐   ┌──────────┐
   │ OHLCV  │    │  Macro  │    │   OI/    │    │ Wyckoff │   │   SMC    │
   │Binance │    │TradingV │    │ Funding  │    │  Events │   │Features  │
   └───┬────┘    └────┬────┘    └────┬─────┘    └────┬────┘   └────┬─────┘
       │              │               │               │             │
       └──────────────┴───────────────┴───────────────┴─────────────┘
                                       │
                                       ▼
                          ┌────────────────────────┐
                          │   FEATURE PIPELINE     │
                          │   (Build Once/Day)     │
                          │                        │
                          │  ┌──────────────────┐  │
                          │  │ Tier 1: Base     │  │
                          │  │ Tier 2: MTF      │  │
                          │  │ Tier 3: SMC      │  │
                          │  │ Tier 4: Wyckoff  │  │
                          │  │ Tier 5: Macro    │  │
                          │  │ Tier 6: Fusion   │  │
                          │  │ Tier 7: Derived  │  │
                          │  └──────────────────┘  │
                          └────────┬───────────────┘
                                   │
                                   ▼
                    ┌──────────────────────────────┐
                    │   VERSIONED FEATURE STORE    │
                    │   (Parquet - Read Only)      │
                    │                              │
                    │  BTC_1H_2022-2024_v2.parquet │
                    │  ├── 26,236 rows × 195 cols  │
                    │  ├── 100% coverage (no NaNs) │
                    │  └── Validated schema        │
                    └──────────┬───────────────────┘
                               │
              ┌────────────────┼────────────────┐
              │                │                │
              ▼                ▼                ▼
        ┌──────────┐     ┌──────────┐    ┌──────────┐
        │Archetype │     │  Simple  │    │    ML    │
        │  Models  │     │Classifier│    │  Models  │
        │  (19)    │     │  (3)     │    │  (5)     │
        └────┬─────┘     └────┬─────┘    └────┬─────┘
             │                │               │
             └────────────────┼───────────────┘
                              │
                              ▼
                  ┌───────────────────────┐
                  │  BACKTEST ENGINE      │
                  │  (Model Agnostic)     │
                  │                       │
                  │  ├── Train/Test Split │
                  │  ├── Walk-Forward     │
                  │  ├── Comparison       │
                  │  └── Metrics          │
                  └───────────┬───────────┘
                              │
                              ▼
                  ┌───────────────────────┐
                  │  RESULTS & REPORTS    │
                  │                       │
                  │  ├── Trade Log        │
                  │  ├── Equity Curve     │
                  │  ├── Performance      │
                  │  └── Model Comparison │
                  └───────────────────────┘
```

---

## 2. Data Layer - Feature Store Versioning

```
┌─────────────────────────────────────────────────────────────────────────┐
│                      FEATURE STORE EVOLUTION                            │
└─────────────────────────────────────────────────────────────────────────┘

VERSION 1 (Current - Partially Broken):
┌──────────────────────────────────────────────────────────────┐
│  BTC_1H_2022-2024_v1.parquet                                 │
│  ├── 26,236 rows × 167 columns                               │
│  ├── Coverage: 83.8% (140 clean, 27 with NaNs)               │
│  │                                                            │
│  └── ISSUES:                                                 │
│      ├── oi_change_* (17,598 NaNs) ← BROKEN                  │
│      ├── funding, rv_* (17,475 NaNs) ← 2022-2023 missing     │
│      └── Runtime enrichment needed (10+ modules)             │
└──────────────────────────────────────────────────────────────┘
                              │
                              │ FIX: Backfill macro + OI
                              │ ADD: Derived features (28 cols)
                              ▼
VERSION 2 (Target - Production Ready):
┌──────────────────────────────────────────────────────────────┐
│  BTC_1H_2022-2024_v2.parquet                                 │
│  ├── 26,236 rows × 195 columns (+28 from v1)                 │
│  ├── Coverage: 100% (0 NaNs) ✓                               │
│  │                                                            │
│  ├── NEW COLUMNS (28):                                       │
│  │   ├── S1 features (fvg_below, liquidity_drain_pct, ...)   │
│  │   ├── S2 features (wick_ratio, vol_fade, ...)             │
│  │   ├── S4 features (ob_retest, volume_spike, ...)          │
│  │   └── Multi-bar (capitulation_depth, ...)                 │
│  │                                                            │
│  ├── FIXED COLUMNS (27):                                     │
│  │   ├── oi_change_24h, oi_change_pct_24h, oi_z ✓           │
│  │   ├── funding, rv_20d, rv_60d (backfilled) ✓             │
│  │   └── VIX, DXY, YIELDS (backfilled) ✓                    │
│  │                                                            │
│  └── BENEFITS:                                               │
│      ├── NO runtime enrichment needed                        │
│      ├── 60% faster backtests                                │
│      └── ML models unblocked                                 │
└──────────────────────────────────────────────────────────────┘
                              │
                              │ ADD: ML/Temporal features
                              ▼
VERSION 3 (Future - ML Enhanced):
┌──────────────────────────────────────────────────────────────┐
│  BTC_1H_2022-2024_v3.parquet                                 │
│  ├── 26,236 rows × 220 columns (+25 from v2)                 │
│  ├── Coverage: 100% (0 NaNs)                                 │
│  │                                                            │
│  └── NEW COLUMNS (25):                                       │
│      ├── Fibonacci (5): fib_0.236, fib_0.5, ...              │
│      ├── Swings (4): swing_high_1h, swing_low_4h, ...        │
│      ├── Psychology (2): fear_greed_index, crowd_sentiment   │
│      ├── Structure (3): support_level, resistance_level, ... │
│      ├── Temporal (4): fib_time_cluster, gann_angle, ...     │
│      └── Alternative Data (7): reddit_sentiment, ...         │
└──────────────────────────────────────────────────────────────┘

MANIFEST TRACKING (Git-Versioned):
┌──────────────────────────────────────────────────────────────┐
│  data/features_mtf/manifests/                                │
│  ├── v1_manifest.json  (167 cols, sources, validation)       │
│  ├── v2_manifest.json  (195 cols, diff from v1, ...)         │
│  └── v3_manifest.json  (220 cols, diff from v2, ...)         │
└──────────────────────────────────────────────────────────────┘
```

---

## 3. Model Layer - Unified Interface

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        MODEL LAYER ARCHITECTURE                         │
└─────────────────────────────────────────────────────────────────────────┘

                         ┌────────────────────┐
                         │   BaseModel (ABC)  │
                         │                    │
                         │  + fit(data)       │
                         │  + predict(bar)    │
                         │  + get_params()    │
                         └──────────┬─────────┘
                                    │
                    ┌───────────────┼───────────────┐
                    │               │               │
                    ▼               ▼               ▼
         ┌──────────────────┐ ┌──────────────┐ ┌──────────────┐
         │ ArchetypeModel   │ │SimpleClassif │ │   MLModel    │
         │  (Wrapper)       │ │  (Baseline)  │ │  (Future)    │
         └──────────────────┘ └──────────────┘ └──────────────┘
                 │                    │                  │
    ┌────────────┼────────┐          │       ┌──────────┼──────────┐
    │            │        │          │       │          │          │
    ▼            ▼        ▼          ▼       ▼          ▼          ▼
┌───────┐  ┌────────┐ ┌──────┐ ┌────────┐ ┌──────┐ ┌───────┐ ┌─────────┐
│  S1   │  │   S2   │ │  S5  │ │  RSI   │ │ LSTM │ │XGBoost│ │Ensemble │
│Liquid │  │ Failed │ │ Long │ │ Over-  │ │      │ │       │ │ (Blend) │
│Vacuum │  │ Rally  │ │Squeez│ │ sold   │ │      │ │       │ │         │
└───────┘  └────────┘ └──────┘ └────────┘ └──────┘ └───────┘ └─────────┘

ALL MODELS READ SAME v2 FEATURE STORE:
┌──────────────────────────────────────────────────────────────┐
│  df = pd.read_parquet('BTC_1H_2022-2024_v2.parquet')         │
│                                                              │
│  # Model 1: S2 Archetype                                    │
│  model1 = ArchetypeModel('s2.json', 'S2')                   │
│  model1.fit(df['2022':'2023'])  # Uses features from config │
│                                                              │
│  # Model 2: XGBoost                                         │
│  features = ['rsi_14', 'adx_14', 'liquidity_score']         │
│  model2 = XGBoostModel(features)                            │
│  model2.fit(df[features]['2022':'2023'])                    │
│                                                              │
│  # Model 3: Ensemble                                        │
│  model3 = EnsembleModel(                                    │
│      models=[model1, model2],                               │
│      weights=[0.6, 0.4]                                     │
│  )                                                           │
│                                                              │
│  # All use SAME underlying data (apples-to-apples)          │
└──────────────────────────────────────────────────────────────┘

SIGNAL OUTPUT (Unified):
┌──────────────────────────────────────────────────────────────┐
│  @dataclass                                                  │
│  class Signal:                                               │
│      direction: 'long' | 'short' | 'hold'                   │
│      confidence: float  # 0.0 - 1.0                          │
│      entry_price: float                                      │
│      stop_loss: float                                        │
│      take_profit: Optional[float]                            │
│      metadata: dict  # Model-specific context               │
└──────────────────────────────────────────────────────────────┘
```

---

## 4. Backtesting Layer - Model Agnostic

```
┌─────────────────────────────────────────────────────────────────────────┐
│                   BACKTEST ENGINE ARCHITECTURE                          │
└─────────────────────────────────────────────────────────────────────────┘

SINGLE BACKTEST ENGINE (Model-Agnostic):
┌──────────────────────────────────────────────────────────────┐
│  class BacktestEngine:                                       │
│                                                              │
│      def __init__(model: BaseModel, data: DataFrame):       │
│          self.model = model                                 │
│          self.data = data  # v2 parquet (read-only)         │
│                                                              │
│      def run(start, end) -> BacktestResults:                │
│          for bar in data[start:end]:                        │
│              signal = model.predict(bar, position)          │
│              if signal.is_entry:                            │
│                  position = open_position(signal)           │
│              elif position:                                 │
│                  check_stop_target(position, bar)           │
│          return BacktestResults(trades, metrics)            │
└──────────────────────────────────────────────────────────────┘

EXECUTION FLOW:
┌──────────────────────────────────────────────────────────────┐
│  1. Load v2 parquet (once)                                   │
│     df = read_parquet('BTC_1H_2022-2024_v2.parquet')         │
│                                                              │
│  2. Split data                                               │
│     train = df['2022':'2023']                                │
│     test = df['2024']                                        │
│                                                              │
│  3. Fit model (on training data)                             │
│     model.fit(train)                                         │
│                                                              │
│  4. Run backtest (on test data)                              │
│     engine = BacktestEngine(model, test)                     │
│     results = engine.run()                                   │
│                                                              │
│  5. Evaluate performance                                     │
│     print(results.summary())                                 │
│     ├── Win Rate: 65.0%                                      │
│     ├── Profit Factor: 2.1                                   │
│     ├── Sharpe: 1.5                                          │
│     └── Max DD: -8.2%                                        │
└──────────────────────────────────────────────────────────────┘

MULTI-MODEL COMPARISON:
┌──────────────────────────────────────────────────────────────┐
│  class ModelComparison:                                      │
│                                                              │
│      def run_comparison(models, data, train, test):         │
│          results = []                                        │
│          for model in models:                                │
│              model.fit(data[train])                          │
│              engine = BacktestEngine(model, data[test])      │
│              result = engine.run()                           │
│              results.append(result)                          │
│          return comparison_table(results)                    │
└──────────────────────────────────────────────────────────────┘

OUTPUT (Comparison Table):
┌──────────────────────────────────────────────────────────────┐
│ Model           Trades  WR%   PF    Sharpe  MaxDD  PnL      │
│ ───────────────────────────────────────────────────────────  │
│ S2-Optimized    15      60.0  1.8   1.2     -8.0   $1,200   │
│ S5-Optimized    8       75.0  2.4   1.8     -5.0   $1,800   │
│ XGB-Baseline    42      55.0  1.5   1.0    -12.0   $900     │
│ RSI-Oversold    28      45.0  0.9   0.5    -15.0   -$300    │
│                                                              │
│ WINNER: S5-Optimized (highest PF + Sharpe, lowest DD)       │
└──────────────────────────────────────────────────────────────┘
```

---

## 5. Feature Pipeline - Build Process

```
┌─────────────────────────────────────────────────────────────────────────┐
│                     FEATURE PIPELINE ARCHITECTURE                       │
└─────────────────────────────────────────────────────────────────────────┘

BUILD PROCESS (Offline - Once Per Day):
┌──────────────────────────────────────────────────────────────┐
│  bin/build_feature_store_v2.py                               │
│                                                              │
│  Step 1: Load Raw Data                                      │
│  ├── OHLCV: Binance API (1H bars)                           │
│  ├── Macro: TradingView (DXY, VIX, Yields)                  │
│  ├── OI: Binance Futures, OKX                               │
│  └── Wyckoff: Pre-computed events cache                     │
│                                                              │
│  Step 2: Calculate Features (Tier 1-7)                      │
│  ├── Tier 1: calculate_base_features()                      │
│  │   └── ATR, RSI, ADX, SMAs (14 cols)                      │
│  ├── Tier 2: calculate_mtf_features()                       │
│  │   └── 1H/4H/1D timeframe features (45 cols)              │
│  ├── Tier 3: calculate_smc_features()                       │
│  │   └── OB, FVG, BOS (12 cols)                             │
│  ├── Tier 4: calculate_wyckoff_events()                     │
│  │   └── 18 Wyckoff events (36 cols)                        │
│  ├── Tier 5: merge_macro_features()                         │
│  │   └── DXY, VIX, Funding, OI (30 cols)                    │
│  ├── Tier 6: calculate_fusion_scores()                      │
│  │   └── Composite scores (12 cols)                         │
│  └── Tier 7: calculate_derived_features() ← NEW             │
│      └── S1/S2/S4 archetype features (28 cols)              │
│                                                              │
│  Step 3: Validate Schema                                    │
│  ├── validate_no_nans(df)        # STRICT                   │
│  ├── validate_ranges(df)         # All cols in bounds       │
│  ├── validate_consistency(df)    # high >= low, etc.        │
│  └── validate_timestamps(df)     # No gaps                  │
│                                                              │
│  Step 4: Export                                             │
│  ├── df.to_parquet('BTC_1H_2022-2024_v2.parquet')           │
│  ├── save_manifest('v2_manifest.json')                      │
│  └── save_validation_report('v2_validation.json')           │
└──────────────────────────────────────────────────────────────┘

FEATURE MODULES (Separation of Concerns):
┌──────────────────────────────────────────────────────────────┐
│  engine/features/                                            │
│  ├── base.py          # Tier 1: OHLCV + technical           │
│  ├── mtf.py           # Tier 2: Multi-timeframe             │
│  ├── smc.py           # Tier 3: Smart Money Concepts        │
│  ├── wyckoff.py       # Tier 4: Wyckoff events              │
│  ├── macro.py         # Tier 5: Macro features              │
│  ├── fusion.py        # Tier 6: Fusion scores               │
│  ├── derived.py       # Tier 7: Derived features (NEW)      │
│  └── registry.py      # Feature metadata + validation       │
└──────────────────────────────────────────────────────────────┘

VALIDATION PIPELINE:
┌──────────────────────────────────────────────────────────────┐
│  bin/validate_feature_store.py                               │
│                                                              │
│  1. Load parquet                                             │
│  2. Check NaNs (CRITICAL - must be 0)                        │
│  3. Check ranges (e.g., RSI 0-100)                           │
│  4. Check consistency (high >= low)                          │
│  5. Check timestamps (hourly continuity)                     │
│  6. Generate report                                          │
│                                                              │
│  Output: PASS/FAIL + detailed error log                     │
└──────────────────────────────────────────────────────────────┘
```

---

## 6. Separation of Concerns

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    LAYER SEPARATION (No Cross-Contamination)            │
└─────────────────────────────────────────────────────────────────────────┘

LAYER 1: ETL (Offline - Batch Job)
┌──────────────────────────────────────────────────────────────┐
│  RESPONSIBILITY: Raw Data → Versioned Features               │
│                                                              │
│  INPUT:  Binance API, TradingView, OKX                      │
│  OUTPUT: BTC_1H_2022-2024_v2.parquet                         │
│                                                              │
│  RULES:                                                      │
│  ✓ Compute ALL features (195 columns)                       │
│  ✓ Validate 100% coverage (no NaNs)                         │
│  ✓ Export to parquet (immutable)                            │
│  ✗ NO model logic                                           │
│  ✗ NO backtest logic                                        │
└──────────────────────────────────────────────────────────────┘
                            │
                            ▼
LAYER 2: MODEL (Online/Offline)
┌──────────────────────────────────────────────────────────────┐
│  RESPONSIBILITY: Features → Signals                          │
│                                                              │
│  INPUT:  v2 parquet (READ-ONLY)                             │
│  OUTPUT: Signal(direction, confidence, stop, target)        │
│                                                              │
│  RULES:                                                      │
│  ✓ Read features from parquet                               │
│  ✓ Generate trading signals                                 │
│  ✓ Manage position sizing                                   │
│  ✗ NO feature computation (except experimental)             │
│  ✗ NO backtest execution logic                              │
└──────────────────────────────────────────────────────────────┘
                            │
                            ▼
LAYER 3: BACKTEST (Offline)
┌──────────────────────────────────────────────────────────────┐
│  RESPONSIBILITY: Signals → Performance Metrics               │
│                                                              │
│  INPUT:  Model + v2 parquet                                 │
│  OUTPUT: BacktestResults(trades, PnL, metrics)              │
│                                                              │
│  RULES:                                                      │
│  ✓ Execute trades from signals                              │
│  ✓ Track equity curve                                       │
│  ✓ Calculate metrics (PF, WR, Sharpe)                       │
│  ✗ NO model-specific logic (model-agnostic)                 │
│  ✗ NO feature computation                                   │
└──────────────────────────────────────────────────────────────┘
                            │
                            ▼
LAYER 4: LIVE (Online - Future)
┌──────────────────────────────────────────────────────────────┐
│  RESPONSIBILITY: Streaming Features → Orders                │
│                                                              │
│  INPUT:  Binance WebSocket (real-time OHLCV)               │
│  OUTPUT: CCXT orders (limit/market)                         │
│                                                              │
│  RULES:                                                      │
│  ✓ Calculate features (streaming pipeline)                  │
│  ✓ Load trained model (from backtest)                       │
│  ✓ Generate signals (model.predict)                         │
│  ✓ Execute orders (CCXT)                                    │
│  ✗ NO backtesting logic                                     │
└──────────────────────────────────────────────────────────────┘

DATA FLOW (Unidirectional):
┌──────────────────────────────────────────────────────────────┐
│                                                              │
│  Raw Data → ETL → Parquet → Model → Signal → Backtest       │
│                     ▲         │                              │
│                     │         └─→ (Read-Only)                │
│                     │                                        │
│                     └─── NO MODIFICATIONS ALLOWED            │
│                                                              │
└──────────────────────────────────────────────────────────────┘
```

---

## 7. Migration Path - Current to Target

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         MIGRATION ROADMAP                               │
└─────────────────────────────────────────────────────────────────────────┘

CURRENT STATE (v1):
┌──────────────────────────────────────────────────────────────┐
│  ┌────────────┐                                              │
│  │ v1 Parquet │ ← 167 cols, 27 with NaNs                     │
│  └─────┬──────┘                                              │
│        │                                                     │
│        ├─→ ArchetypeModel                                   │
│        │   └─→ Runtime enrichment (S1, S2, S4, S5)          │
│        │       └─→ SLOW (10x slower)                        │
│        │                                                     │
│        └─→ Can't add ML models (messy features)             │
│                                                              │
│  ISSUES:                                                     │
│  ✗ NaN values block some archetypes                         │
│  ✗ Runtime enrichment duplicates code                       │
│  ✗ Can't compare models fairly                              │
└──────────────────────────────────────────────────────────────┘
                        │
                        │ PHASE 1 (Week 1)
                        │ Fix broken columns
                        ▼
INTERMEDIATE STATE (v1_fixed):
┌──────────────────────────────────────────────────────────────┐
│  ┌─────────────┐                                             │
│  │ v1_fixed    │ ← 167 cols, 0 NaNs (100% coverage)          │
│  │  Parquet    │                                             │
│  └──────┬──────┘                                             │
│         │                                                    │
│         ├─→ ArchetypeModel (still uses runtime enrichment)  │
│         └─→ ML models still blocked                         │
│                                                              │
│  FIXES:                                                      │
│  ✓ Backfilled macro (2022-2023)                             │
│  ✓ Fixed OI derivatives                                     │
│  ✗ Still has runtime enrichment                             │
└──────────────────────────────────────────────────────────────┘
                        │
                        │ PHASE 2 (Week 2)
                        │ Add derived features
                        ▼
TARGET STATE (v2):
┌──────────────────────────────────────────────────────────────┐
│  ┌──────────┐                                                │
│  │ v2       │ ← 195 cols, 0 NaNs (100% coverage)             │
│  │ Parquet  │                                                │
│  └────┬─────┘                                                │
│       │                                                      │
│       ├─→ ArchetypeModel (NO runtime enrichment)            │
│       ├─→ SimpleClassifier                                  │
│       ├─→ XGBoostModel ← UNBLOCKED                          │
│       ├─→ LSTMModel ← UNBLOCKED                             │
│       └─→ EnsembleModel ← UNBLOCKED                         │
│                                                              │
│  BENEFITS:                                                   │
│  ✓ NO runtime enrichment (60% faster)                       │
│  ✓ ML models unblocked (clean data)                         │
│  ✓ Fair model comparison (apples-to-apples)                 │
│  ✓ Live trading ready (streaming pipeline)                  │
└──────────────────────────────────────────────────────────────┘

TIMELINE (8 Weeks):
┌──────────────────────────────────────────────────────────────┐
│  Week 1: Phase 1 (Fix NaNs)                                  │
│  Week 2: Phase 2 (Add derived features → v2)                 │
│  Week 3: Phase 3 (Eliminate runtime enrichment)              │
│  Week 4: Phase 4 (Model comparison framework)                │
│  Week 5-6: Phase 5 (Add ML models)                           │
│  Week 7-8: Phase 6 (Live trading prep)                       │
└──────────────────────────────────────────────────────────────┘
```

---

## 8. Performance Comparison

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    PERFORMANCE: v1 vs v2                                │
└─────────────────────────────────────────────────────────────────────────┘

BACKTEST TIME (Single Archetype):
┌──────────────────────────────────────────────────────────────┐
│  v1 (Current):                                               │
│  ├── Load v1 parquet           2 sec                         │
│  ├── Runtime enrichment        10 sec ← SLOW                 │
│  ├── Archetype detection       2 sec                         │
│  └── Trade execution           1 sec                         │
│  TOTAL: 15 minutes                                           │
│                                                              │
│  v2 (Target):                                                │
│  ├── Load v2 parquet           2 sec                         │
│  ├── Runtime enrichment        0 sec ← ELIMINATED            │
│  ├── Archetype detection       2 sec                         │
│  └── Trade execution           1 sec                         │
│  TOTAL: 6 minutes                                            │
│                                                              │
│  SPEEDUP: 60% faster                                         │
└──────────────────────────────────────────────────────────────┘

MULTI-MODEL COMPARISON (4 Models):
┌──────────────────────────────────────────────────────────────┐
│  v1 (Current):                                               │
│  ├── Model 1 (S2): 15 min                                    │
│  ├── Model 2 (S5): 15 min                                    │
│  ├── Model 3 (XGB): N/A (blocked)                            │
│  └── Model 4 (LSTM): N/A (blocked)                           │
│  TOTAL: 30 minutes (only 2 models)                           │
│                                                              │
│  v2 (Target):                                                │
│  ├── Load v2 parquet (shared): 2 sec                         │
│  ├── Model 1 (S2): 5 min                                     │
│  ├── Model 2 (S5): 5 min                                     │
│  ├── Model 3 (XGB): 8 min (training)                         │
│  └── Model 4 (LSTM): 12 min (training)                       │
│  TOTAL: 30 minutes (4 models, parallelizable)                │
│                                                              │
│  SPEEDUP: 2x more models in same time                        │
└──────────────────────────────────────────────────────────────┘

STORAGE:
┌──────────────────────────────────────────────────────────────┐
│  v1: 10 MB (parquet) + 10+ runtime modules (code)            │
│  v2: 12 MB (parquet) + 0 runtime modules                     │
│                                                              │
│  REDUCTION: -1000 LOC (runtime enrichment eliminated)        │
└──────────────────────────────────────────────────────────────┘

DEVELOPER EXPERIENCE:
┌──────────────────────────────────────────────────────────────┐
│  v1: Add new feature                                         │
│  ├── Step 1: Add to archetype detector (code)                │
│  ├── Step 2: Add to runtime enricher (DUPLICATE)             │
│  ├── Step 3: Update backtest script (call enricher)          │
│  └── RISK: Features may differ (bugs)                        │
│                                                              │
│  v2: Add new feature                                         │
│  ├── Step 1: Add to feature builder (once)                   │
│  ├── Step 2: Rebuild parquet (5-10 min)                      │
│  └── Step 3: Use in any model (no changes)                   │
│                                                              │
│  IMPROVEMENT: 3 steps → 2 steps, no duplication              │
└──────────────────────────────────────────────────────────────┘
```

---

## 9. Live Trading Architecture (Future)

```
┌─────────────────────────────────────────────────────────────────────────┐
│                      LIVE TRADING ARCHITECTURE                          │
└─────────────────────────────────────────────────────────────────────────┘

STREAMING FEATURE PIPELINE:
┌──────────────────────────────────────────────────────────────┐
│  Binance WebSocket (Real-Time)                               │
│  ├── BTC/USDT 1H candles                                     │
│  └── On candle close (every hour):                           │
│      │                                                       │
│      ├─→ engine/live/streaming_features.py                  │
│      │   ├── Calculate Tier 1-7 features                    │
│      │   │   (same logic as build_feature_store_v2.py)      │
│      │   └── Validate: features match backtest              │
│      │                                                       │
│      ├─→ Load trained model (from backtest)                 │
│      │   model = load_checkpoint('s2_optimized.pkl')        │
│      │                                                       │
│      ├─→ Generate signal                                    │
│      │   signal = model.predict(latest_bar)                 │
│      │                                                       │
│      ├─→ IF signal.is_entry:                                │
│      │   │                                                  │
│      │   ├─→ engine/live/executor.py                        │
│      │   │   ├── place_order(signal)                        │
│      │   │   ├── set_stop_loss(signal.stop_loss)            │
│      │   │   └── set_take_profit(signal.take_profit)        │
│      │   │                                                  │
│      │   └─→ engine/live/monitor.py                         │
│      │       ├── Log trade (database)                       │
│      │       ├── Alert (Telegram/Discord)                   │
│      │       └── Update dashboard                           │
│      │                                                       │
│      └─→ ELSE: Check open positions                         │
│          ├── Update trailing stops                          │
│          └── Monitor for exit conditions                    │
└──────────────────────────────────────────────────────────────┘

PAPER TRADING (Testing Phase):
┌──────────────────────────────────────────────────────────────┐
│  engine/live/paper_executor.py                               │
│  ├── Simulate orders (NO real money)                         │
│  ├── Track virtual portfolio                                 │
│  ├── Compare to backtest results                             │
│  └── Validate: paper trading ≈ backtest (within 5%)          │
└──────────────────────────────────────────────────────────────┘

MONITORING DASHBOARD:
┌──────────────────────────────────────────────────────────────┐
│  Live Trading Dashboard (Streamlit/Grafana)                  │
│  ├── Current Position                                        │
│  │   ├── Entry: $50,000                                      │
│  │   ├── Current: $51,200 (+2.4%)                            │
│  │   └── Stop: $48,500 (-3.0%)                               │
│  │                                                           │
│  ├── Signal History (Last 24h)                               │
│  │   ├── 10:00 - S2 Signal (short, conf=0.85) → Entry       │
│  │   ├── 14:00 - No signal                                  │
│  │   └── 18:00 - Exit signal (target hit) → Close           │
│  │                                                           │
│  ├── Feature Validation                                      │
│  │   ├── Streaming features: 195/195 ✓                      │
│  │   └── Match backtest: 100% ✓                             │
│  │                                                           │
│  └── Performance (Live vs Backtest)                          │
│      ├── Live PF: 2.1 (backtest: 2.0) ✓                     │
│      └── Live WR: 62% (backtest: 65%) ✓                     │
└──────────────────────────────────────────────────────────────┘

SAFETY MEASURES:
┌──────────────────────────────────────────────────────────────┐
│  1. Paper trading first (2+ weeks)                           │
│  2. Feature validation (streaming = backtest)                │
│  3. Position size limits (max 2% risk)                       │
│  4. Circuit breakers (pause if DD > 10%)                     │
│  5. Manual approval required for real money                  │
└──────────────────────────────────────────────────────────────┘
```

---

**END OF DIAGRAMS**

For detailed explanations, see: [TARGET_STATE_ARCHITECTURE.md](./TARGET_STATE_ARCHITECTURE.md)
