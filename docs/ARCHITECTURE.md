# Bull Machine Architecture

**Last Updated:** 2025-11-14
**Version:** v2 (Post-Restructuring)

## Overview

Bull Machine is a quantitative trading engine implementing multi-timeframe technical analysis, Smart Money Concepts (SMC), and regime-aware archetype detection for cryptocurrency markets.

## Repository Structure

```
Bull-machine-/
├── bull_machine/          # High-level orchestration & package entrypoints
├── engine/                # Core trading engine
│   ├── strategies/        # Trading strategies (NEW - modular refactoring)
│   │   └── archetypes/    # Archetype detection
│   │       ├── bull/      # Bull market patterns (A-M) [Future]
│   │       └── bear/      # Bear market patterns (S1-S8) [Future]
│   ├── archetypes/        # Archetype logic (current monolith)
│   ├── smc/               # Smart Money Concepts
│   ├── fusion/            # Multi-factor fusion engine
│   ├── exits/             # Exit strategy logic
│   ├── features/          # Feature engineering
│   ├── ml/                # Machine learning components
│   └── runtime/           # Runtime context & execution
├── configs/               # Configuration files (REORGANIZED)
│   ├── frozen/            # Frozen baselines (never edit)
│   ├── mvp/               # Production MVP configs
│   ├── experiments/       # Research & experimental configs
│   ├── regime/            # Regime routing configs
│   └── bear/              # Bear archetype configs
├── data/                  # Feature stores & data (gitignored)
│   ├── processed/         # Processed features
│   │   ├── features_mtf/  # Multi-timeframe features
│   │   └── macro/         # Macro features
│   ├── raw/               # Raw market data
│   └── archive/           # Historical data archives
├── tests/                 # Test suite (REORGANIZED)
│   ├── unit/              # Unit tests (42 files)
│   ├── integration/       # Integration tests (13 files)
│   ├── smoke/             # Smoke tests
│   └── robustness/        # Robustness tests
├── docs/                  # Documentation
│   ├── technical/         # Technical documentation
│   ├── backtests/         # Backtest reports
│   ├── analysis/          # Analysis reports
│   ├── reports/           # Execution reports
│   └── guides/            # User guides
├── results_reference/     # Curated ground truth results
├── bin/                   # Executable scripts & tools
├── scripts/               # Automation scripts
└── tools/                 # Development tools
```

## Core Components

### 1. Archetype Detection (`engine/archetypes/`)

**Status:** Monolithic (1441 lines in `logic_v2_adapter.py`)
**Future:** Modular refactoring into `engine/strategies/archetypes/{bull,bear}/`

**Bull Market Archetypes (11 patterns):**
- **A (Trap Reversal):** PTI spring/UTAD + displacement
- **B (Order Block Retest):** BOS + BOMS + Wyckoff
- **C (FVG Continuation):** Displacement + momentum
- **D (Failed Continuation):** FVG + weak RSI
- **E (Liquidity Compression):** Low ATR + volume cluster
- **F (Expansion Exhaustion):** Extreme RSI + high ATR
- **G (Re-Accumulate):** BOMS strength + high liquidity
- **H (Trap Within Trend):** ADX trend + liquidity drop
- **K (Wick Trap):** ADX + liquidity + wicks
- **L (Volume Exhaustion):** Vol spike + extreme RSI
- **M (Ratio Coil Break):** Low ATR + near POC + BOMS

**Bear Market Archetypes (8 patterns):**
- **S1 (Breakdown):** Support break + volume
- **S2 (Failed Rally Rejection):** Dead cat bounce (58.5% WR)
- **S3 (Whipsaw):** False break + reversal
- **S4 (Distribution):** High volume + no follow
- **S5 (Long Squeeze Cascade):** Funding extreme + exhaustion
- **S6 (Alt Rotation Down):** [REJECTED - missing data]
- **S7 (Curve Inversion):** [REJECTED - missing data]
- **S8 (Volume Fade Chop):** Low volume drift

**Key Features:**
- **RuntimeContext API:** Regime-aware parameter adaptation
- **ThresholdPolicy:** Dynamic gate adjustment by market state
- **StateAwareGates:** Bull Machine v2 adaptive gating
- **Evaluate-All Dispatcher:** Prevents archetype starvation
- **Soft Filters:** Penalties instead of hard vetoes (critical for choppy conditions)

### 2. Fusion Engine (`engine/fusion.py`)

Weighted multi-factor scoring combining:
- Wyckoff analysis (0.331 weight)
- Liquidity score (0.392 weight)
- Momentum indicators (0.205 weight)
- Fakeout penalty (0.075 multiplier)

### 3. Smart Money Concepts (`engine/smc/`)

- **Order Blocks:** Institutional demand/supply zones
- **FVGs:** Fair Value Gaps (imbalance zones)
- **BOS:** Break of Structure detection
- **Liquidity Pools:** Sweep detection

### 4. Regime Detection (`engine/regime_detector.py`)

Multi-factor macro regime classification:
- Risk-on / Risk-off / Neutral / Crisis
- Inputs: DXY, VIX, Gold, Funding, OI

### 5. Exit Strategy (`engine/exits/`)

- **ATR-based exits:** Dynamic stop-loss/take-profit
- **Phase-based stops:** Accumulation/distribution phases
- **Broker-managed:** SL/TP orders

## Configuration System

### Config Organization

```
configs/
├── frozen/              # Frozen baselines (never modify)
│   └── btc_1h_v2_baseline.json
├── mvp/                 # Production MVP configs
│   ├── mvp_bull_market_v1.json
│   └── mvp_bear_market_v1.json
├── experiments/         # Research configs
│   ├── baseline_btc_bull_pf20.json
│   └── baseline_btc_adaptive_pr6b.json
├── regime/              # Regime routing
│   └── regime_routing_production_v1.json
└── bear/                # Bear archetype configs
    └── baseline_btc_bear_archetypes_adaptive_v3.2.json
```

### Config Structure

```json
{
  "archetypes": {
    "use_archetypes": true,
    "enable_A": true,
    "enable_S2": true,
    "thresholds": {...},
    "routing": {
      "risk_on": {"weights": {...}},
      "risk_off": {"weights": {...}}
    }
  },
  "fusion": {
    "weights": {
      "wyckoff": 0.331,
      "liquidity": 0.392,
      "momentum": 0.205
    }
  },
  "exits": {...},
  "risk": {...}
}
```

## Testing Strategy

### Unit Tests (`tests/unit/`)
- Feature engineering tests
- Indicator validation
- Component isolation tests
- 42 test modules

### Integration Tests (`tests/integration/`)
- Full backtest validation
- Multi-asset reproducibility
- Determinism verification
- Performance benchmarks
- 13 test modules

### Test Execution

```bash
# Run all tests
pytest tests/

# Run unit tests only
pytest tests/unit/

# Run integration tests
pytest tests/integration/

# Run smoke tests
pytest tests/smoke/
```

## Data Pipeline

### Feature Store Structure

```
data/processed/
├── features_mtf/        # Multi-timeframe features
│   ├── btc_1h_*.parquet
│   ├── btc_4h_*.parquet
│   └── btc_1d_*.parquet
└── macro/               # Macro features
    ├── regime_labels_*.parquet
    ├── funding_*.csv
    └── oi_*.csv
```

### Feature Categories

1. **Price Action:** OHLCV, structure, swings
2. **SMC:** Order blocks, FVGs, liquidity
3. **Momentum:** RSI, ADX, volume z-score
4. **Wyckoff:** Phase detection, effort/result
5. **Macro:** Regime, funding, OI, correlations

## Execution Flow

```
1. Load Data → Feature Store (data/processed/)
2. Regime Detection → Classify market state
3. Archetype Detection → Pattern matching (engine/archetypes/)
4. Fusion Scoring → Multi-factor weighting (engine/fusion.py)
5. Entry Signal → Gate validation + threshold checks
6. Position Sizing → Risk management
7. Exit Strategy → ATR/phase-based stops
8. Execution → Backtest or live trading
```

## Future Modularization Plan

### Phase 1: Archetype Split (Planned)
- Extract bull archetypes to `engine/strategies/archetypes/bull/`
- Extract bear archetypes to `engine/strategies/archetypes/bear/`
- Individual detector classes per archetype
- Maintain backward compatibility via facades

### Phase 2: Feature Engineering (Planned)
- Separate feature calculators by domain
- Unified feature pipeline orchestration
- Cached feature computation

### Phase 3: ML Integration (Planned)
- Scikit-learn ensemble models
- Feature importance analysis
- Online learning adaptation

## Development Workflow

### Adding New Archetypes

1. Define archetype in `engine/archetypes/logic_v2_adapter.py`
2. Add threshold parameters in config
3. Add unit tests in `tests/unit/`
4. Add integration test in `tests/integration/`
5. Validate with backtest
6. Document in `docs/technical/`

### Modifying Configs

1. **NEVER** edit frozen configs (`configs/frozen/`)
2. Create new config in `configs/experiments/`
3. Run backtest validation
4. If approved, promote to `configs/mvp/`

### Running Backtests

```bash
# Single asset backtest
python bin/backtest_knowledge_v2.py --config configs/mvp/mvp_bull_market_v1.json

# Multi-asset validation
pytest tests/integration/test_real_performance.py
```

## Key Design Principles

1. **Backward Compatibility:** Never break existing imports/APIs
2. **Incremental Refactoring:** Small, safe changes with validation
3. **Feature Flags:** Bull/Bear separate evaluation paths
4. **Soft Filters:** Penalties over hard vetoes (preserves signals)
5. **Regime Awareness:** Dynamic thresholds by market state
6. **Data Integrity:** Frozen baselines + ground truth results

## Performance Characteristics

### Gold Standard Performance (BTC 2022-2024)
- Win Rate: 65.8%
- Profit Factor: 2.37
- Max Drawdown: -12.4%
- Sharpe Ratio: 1.83

### Bear Archetype Performance (2022 Bear Market)
- S2 (Failed Rally): 58.5% WR, 1.4 PF
- S5 (Long Squeeze): Under validation

## References

- **Smart Money Concepts:** ICT, Wyckoff methodology
- **Regime Detection:** Macro factor analysis
- **Feature Engineering:** Multi-timeframe technical analysis
- **Risk Management:** ATR-based position sizing

## Contact & Support

For questions about architecture:
- See `docs/technical/` for detailed component docs
- See `docs/guides/` for usage guides
- See `CHANGELOG.md` for version history
