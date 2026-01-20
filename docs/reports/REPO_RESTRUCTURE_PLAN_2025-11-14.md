# Bull Machine Repository Restructuring Plan

**Date**: 2025-11-14
**Author**: System Architect
**Version**: 1.0
**Status**: Planning Phase - NOT YET EXECUTED

---

## Executive Summary

This document provides a comprehensive plan to restructure the Bull Machine repository to meet quant industry gold standards while preserving all domain knowledge, reference results, and existing functionality. The restructuring will improve code organization, separate backtest from production concerns, and establish clear boundaries between core engine logic and strategy implementations.

**Critical Constraint**: No functional behavior changes. All backtests must produce identical results after restructuring.

**Estimated Effort**: 2-3 days of careful migration work

**Risk Level**: Medium (mitigated by comprehensive validation plan)

---

## Table of Contents

1. [Current State Analysis](#1-current-state-analysis)
2. [Target State Specification](#2-target-state-specification)
3. [File Migration Plan](#3-file-migration-plan)
4. [Module Split Plan](#4-module-split-plan)
5. [Migration Steps](#5-migration-steps)
6. [Risk Assessment](#6-risk-assessment)
7. [Validation Checklist](#7-validation-checklist)
8. [Rollback Plan](#8-rollback-plan)

---

## 1. Current State Analysis

### 1.1 Top-Level Structure

**Current layout:**

```
/Users/raymondghandchi/Bull-machine-/Bull-machine-/
├── .github/                    # CI/CD configs
├── .gitignore                  # Enhanced (already good)
├── .pre-commit-config.yaml
├── Makefile
├── pyproject.toml
├── pytest.ini
├── conftest.py
├── requirements.txt
├── requirements-production.txt
├── README.md
├── CHANGELOG.md
├── setup.py                    # Legacy (should deprecate)
├── __init__.py                 # Root-level (unnecessary)
├── RUN_TESTS.sh
├── bull_machine/               # High-level orchestration (KEEP)
├── engine/                     # Core engine logic (KEEP)
├── configs/                    # Config files (NEEDS ORGANIZATION)
├── data/                       # Feature stores (NEEDS STRUCTURE)
├── docs/                       # Documentation (GOOD STRUCTURE)
├── tests/                      # Test suite (GOOD STRUCTURE)
├── schema/                     # Schema definitions (KEEP)
├── results/                    # Runtime results (GOOD - gitignored)
├── results_reference/          # Curated results (KEEP)
├── models/                     # ML artifacts (KEEP)
├── archive/                    # Old versions (KEEP)
├── bin/                        # Scripts (130 files, 38K+ LOC - NEEDS CLEANUP)
├── scripts/                    # More scripts (30 files - CONSOLIDATE)
├── tools/                      # Helper utilities (15 files - KEEP SEPARATE)
├── utils/                      # Shared utilities (4 files - KEEP)
├── logs/                       # Log files (GOOD - gitignored)
├── telemetry/                  # Metrics (GOOD - gitignored)
├── profiles/                   # Profiling outputs (GOOD - gitignored)
└── chart_logs/                 # Chart outputs (symlink - KEEP)
```

### 1.2 Key Issues Identified

#### A. Script Sprawl
- **bin/**: 130 Python scripts (38,309 total lines)
- **scripts/**: 30 additional scripts/tools
- **Overlap**: backtest scripts, analysis scripts, data download scripts scattered
- **Impact**: Hard to find the right script, duplicated functionality

#### B. Archetype Logic Monolith
- **File**: `engine/archetypes/logic_v2_adapter.py` (1,440 lines)
- **Contains**: 11 bull archetypes + 8 bear archetypes in single file
- **Issue**: Hard to maintain, test, and extend individual archetypes
- **Need**: Split into modular strategy files

#### C. Config Organization
- **Current**: 52 config files in flat structure
- **Mix**: Production, experimental, frozen baselines, MVPs
- **Need**: Hierarchical organization by purpose

#### D. Data Directory Chaos
- **Current**: 58 items in `/data/` (CSVs, symlinks, subdirs)
- **Issues**: Raw data mixed with processed, unclear versioning
- **Need**: Separate raw/processed, clear feature store layout

#### E. Documentation Bloat (Already Addressed)
- **Status**: Recently cleaned up, docs well-organized
- **Note**: Keep current structure

### 1.3 Critical Files to Preserve

**Never modify/delete these:**
- `bull_machine/` (all files)
- `engine/` (core logic, will only move/split archetype logic)
- `configs/` (all configs, will reorganize)
- `docs/` (already well-structured)
- `tests/` (already well-structured)
- `schema/` (all schema files)
- `results_reference/` (ground truth)
- `models/` (ML artifacts)
- `archive/` (historical record)

---

## 2. Target State Specification

### 2.1 Final Directory Tree

```
/Users/raymondghandchi/Bull-machine-/Bull-machine-/
├── .github/
│   └── workflows/
├── .gitignore                          # Enhanced (keep current)
├── .pre-commit-config.yaml
├── Makefile                            # Updated with new paths
├── pyproject.toml                      # Updated package structure
├── requirements.txt
├── requirements-production.txt
├── pytest.ini
├── conftest.py
├── README.md
├── CHANGELOG.md
│
├── bull_machine/                       # High-level orchestration
│   ├── __init__.py
│   ├── version.py
│   ├── app/                           # Application entrypoints
│   ├── core/                          # Core utilities
│   ├── fusion/                        # Fusion logic
│   ├── signals/                       # Signal processing
│   ├── io/                            # I/O operations
│   ├── state/                         # State management
│   ├── backtest/                      # Backtest framework
│   ├── config/                        # Config loaders
│   └── tools/                         # High-level tools
│
├── engine/                            # Core engine logic
│   ├── __init__.py
│   ├── strategies/                    # NEW: Strategy modules
│   │   ├── __init__.py
│   │   └── archetypes/               # Archetype strategies
│   │       ├── __init__.py
│   │       ├── bull/                 # Bull market archetypes
│   │       │   ├── __init__.py
│   │       │   ├── trap_reversal.py         # Archetype A
│   │       │   ├── order_block_retest.py    # Archetype B
│   │       │   ├── fvg_continuation.py      # Archetype C
│   │       │   ├── wick_trap.py             # Archetype K
│   │       │   ├── trap_within_trend.py     # Archetype H
│   │       │   ├── volume_exhaustion.py     # Archetype L
│   │       │   ├── expansion_exhaustion.py  # Archetype F
│   │       │   ├── failed_continuation.py   # Archetype D
│   │       │   ├── re_accumulate.py         # Archetype G
│   │       │   ├── liquidity_compression.py # Archetype E
│   │       │   └── ratio_coil_break.py      # Archetype M
│   │       ├── bear/                 # Bear market archetypes
│   │       │   ├── __init__.py
│   │       │   ├── breakdown.py             # S1
│   │       │   ├── failed_rally.py          # S2 (approved)
│   │       │   ├── whipsaw.py               # S3
│   │       │   ├── distribution.py          # S4
│   │       │   ├── long_squeeze.py          # S5 (approved with fix)
│   │       │   ├── alt_rotation_down.py     # S6 (rejected)
│   │       │   ├── curve_inversion.py       # S7 (rejected)
│   │       │   └── volume_fade_chop.py      # S8
│   │       └── base.py               # Base archetype class
│   ├── archetypes/                   # Adapter/facade layer
│   │   ├── __init__.py              # Re-exports for backward compat
│   │   ├── registry.py
│   │   ├── threshold_policy.py
│   │   ├── state_aware_gates.py
│   │   ├── telemetry.py
│   │   ├── param_accessor.py
│   │   └── _legacy.py               # Old logic/logic_v2_adapter (archived)
│   ├── fusion/                       # Fusion engines
│   ├── context/                      # Regime/macro context
│   ├── exits/                        # Exit logic
│   ├── features/                     # Feature engineering
│   ├── gates/                        # Decision gates
│   ├── indicators/                   # Technical indicators
│   ├── liquidity/                    # Liquidity analysis
│   ├── ml/                           # ML models
│   ├── momentum/                     # Momentum calculations
│   ├── psychology/                   # Trader psychology
│   ├── risk/                         # Risk management
│   ├── runtime/                      # Runtime context
│   ├── smc/                          # Smart money concepts
│   ├── structure/                    # Market structure
│   ├── temporal/                     # Temporal analysis
│   ├── timeframes/                   # MTF logic
│   ├── volume/                       # Volume analysis
│   ├── wyckoff/                      # Wyckoff analysis
│   ├── metrics/                      # Performance metrics
│   ├── optimization/                 # Optimization tools
│   └── events/                       # Event processing
│
├── configs/                          # Configuration files
│   ├── README.md                     # Config documentation
│   ├── frozen/                       # Frozen baselines (never edit)
│   │   ├── baseline_btc_bull_pf20.json
│   │   ├── baseline_btc_bear_archetypes_adaptive_v3.2.json
│   │   └── ...
│   ├── mvp/                          # Current MVP configs
│   │   ├── mvp_bull_market_v1.json
│   │   ├── mvp_bear_market_v1.json
│   │   └── regime_routing_production_v1.json
│   ├── experiments/                  # Experimental configs
│   │   ├── optuna/                  # Optuna sweep configs
│   │   ├── bear/                    # Bear archetype experiments
│   │   └── adaptive/                # Adaptive logic experiments
│   ├── regime/                       # Regime routing profiles
│   │   └── ...
│   ├── paper_trading/                # Paper trading configs
│   ├── live/                         # Live trading configs
│   ├── schema/                       # JSON schemas
│   └── archive/                      # Old configs (v150, v7, v8, etc.)
│
├── data/                             # Data storage
│   ├── README.md                     # Data documentation
│   ├── raw/                          # Raw exchange/API data
│   │   ├── binance/
│   │   ├── bybit/
│   │   ├── tradingview/
│   │   └── yfinance/
│   ├── processed/                    # Processed feature stores
│   │   ├── features_mtf/            # Multi-timeframe features
│   │   │   ├── btc_mtf_features.parquet
│   │   │   ├── eth_mtf_features.parquet
│   │   │   └── ...
│   │   ├── macro/                   # Macro features
│   │   │   ├── regime_labels_2022_2024.parquet
│   │   │   └── ...
│   │   └── cache/                   # Temporary computation cache
│   │       └── wyckoff_cache/
│   └── archive/                      # Old feature store versions
│       └── ...
│
├── docs/                             # Documentation (KEEP CURRENT STRUCTURE)
│   ├── README.md
│   ├── ARCHITECTURE.md              # NEW: Architecture overview
│   ├── analysis/
│   ├── archive/
│   ├── audits/
│   ├── backtests/
│   ├── guides/
│   ├── releases/
│   ├── reports/
│   └── technical/
│
├── tests/                            # Test suite (KEEP CURRENT STRUCTURE)
│   ├── unit/
│   ├── integration/
│   ├── smoke/
│   ├── robustness/
│   ├── parity/
│   ├── live/
│   ├── fixtures/
│   ├── legacy/
│   └── archive/
│
├── schema/                           # Schema definitions (KEEP)
│
├── results_reference/                # Curated reference results (KEEP)
│
├── results/                          # Runtime/scratch results (gitignored)
│
├── logs/                             # Log files (gitignored)
│
├── chart_logs/                       # Chart outputs (symlink - keep)
│
├── telemetry/                        # Metrics (gitignored)
│
├── profiles/                         # Profiling outputs (gitignored)
│
├── models/                           # ML artifacts (KEEP)
│
├── archive/                          # Historical archive (KEEP)
│
├── bin/                              # Production/CLI scripts (reorganized)
│   ├── README.md                     # Script documentation
│   ├── backtest/                    # Backtest runners
│   │   ├── backtest_knowledge_v2.py
│   │   ├── backtest_router_v10_integrated.py
│   │   └── ...
│   ├── build/                       # Feature store builders
│   │   ├── build_mtf_feature_store.py
│   │   ├── build_macro_dataset.py
│   │   └── ...
│   ├── backfill/                    # Data backfill scripts
│   │   ├── backfill_liquidity_score_optimized.py
│   │   ├── backfill_ob_high_optimized.py
│   │   ├── backfill_missing_macro_features.py
│   │   └── fix_oi_change_pipeline.py
│   ├── download/                    # Data download scripts
│   │   ├── download_binance_data.py
│   │   └── download_cryptocompare_data.py
│   ├── analyze/                     # Analysis scripts
│   │   ├── analyze_archetype_perf.py
│   │   ├── analyze_optimization.py
│   │   └── ...
│   ├── optimize/                    # Optimization runners
│   │   └── ...
│   ├── cli/                         # CLI entrypoints
│   │   └── bull_machine_cli.py
│   └── archive/                     # Old/deprecated scripts
│
├── tools/                            # Helper utilities (KEEP SEPARATE)
│   ├── README.md
│   ├── fetch_crypto_marketcap.py
│   ├── fetch_historical_macro.py
│   └── ...
│
└── utils/                            # Shared utilities (KEEP)
    └── ...
```

### 2.2 Key Improvements

1. **Strategy Isolation**: Archetypes moved to `engine/strategies/archetypes/{bull,bear}/`
2. **Script Organization**: `bin/` organized by purpose (backtest, build, analyze, etc.)
3. **Config Hierarchy**: Configs organized by purpose (frozen, mvp, experiments, etc.)
4. **Data Structure**: Clear raw/processed separation with versioning
5. **Backward Compatibility**: Adapter layer in `engine/archetypes/` re-exports all classes

---

## 3. File Migration Plan

### 3.1 Configs Migration

**Create structure:**
```bash
mkdir -p configs/{frozen,mvp,experiments/{optuna,bear,adaptive},regime,archive}
```

**File mappings:**

| Source | Destination | Reason |
|--------|-------------|--------|
| `configs/baseline_btc_bull_pf20.json` | `configs/frozen/baseline_btc_bull_pf20.json` | Frozen baseline |
| `configs/baseline_btc_bear_archetypes_adaptive_v3.2_state_gates.json` | `configs/frozen/baseline_btc_bear_archetypes_adaptive_v3.2_state_gates.json` | Frozen baseline |
| `configs/mvp_bull_market_v1.json` | `configs/mvp/mvp_bull_market_v1.json` | Current MVP |
| `configs/mvp_bear_market_v1.json` | `configs/mvp/mvp_bear_market_v1.json` | Current MVP |
| `configs/regime_routing_production_v1.json` | `configs/mvp/regime_routing_production_v1.json` | Current MVP |
| `configs/bear_archetypes_phase1.json` | `configs/experiments/bear/bear_archetypes_phase1.json` | Experiment |
| `configs/btc_v7_ml_*.json` | `configs/archive/btc_v7_ml_*.json` | Old version |
| `configs/btc_v8_*.json` | `configs/archive/btc_v8_*.json` | Old version |
| `configs/profile_*.json` | `configs/regime/profile_*.json` | Regime profiles |
| `configs/knowledge_v2/*` | `configs/archive/knowledge_v2/*` | Old version |
| `configs/v2/*` | `configs/archive/v2/*` | Old version |
| `configs/v3_replay_2024/*` | `configs/archive/v3_replay_2024/*` | Old version |

**Total files to move:** ~50 config files

### 3.2 Bin Scripts Migration

**Create structure:**
```bash
mkdir -p bin/{backtest,build,backfill,download,analyze,optimize,cli,archive}
```

**File mappings (grouped by category):**

#### Backtest Scripts → `bin/backtest/`
- `backtest_knowledge_v2.py`
- `backtest_router_v10_integrated.py`
- `backtest_router_v10_full.py`
- `backtest_router_v10.py`
- `fast_monthly_test.py` (from scripts/)

#### Build Scripts → `bin/build/`
- `build_mtf_feature_store.py`
- `build_feature_store_v2.py`
- `build_feature_store.py`
- `build_macro_dataset.py`
- `build_wyckoff_cache.py`
- `add_gmm_features.py`
- `add_p1_features_to_macro.py`
- `append_macro_to_feature_store.py`
- `cache_features_with_regime.py`

#### Backfill Scripts → `bin/backfill/`
- `backfill_liquidity_score_optimized.py`
- `backfill_liquidity_score.py`
- `backfill_ob_high_optimized.py`
- `backfill_ob_high.py`
- `backfill_missing_macro_features.py`
- `fix_oi_change_pipeline.py`

#### Download Scripts → `bin/download/`
- `download_binance_data.py`
- `download_cryptocompare_data.py`
- `download_all_data.sh`

#### Analysis Scripts → `bin/analyze/`
- `analyze_archetype_perf.py`
- `analyze_optimization.py`
- `analyze_pareto_frontiers_v10.py`
- `analyze_v3_trades_full_year.py`
- `compare_baseline_vs_ml.py`
- `compare_knowledge_v2_abc.py`
- `combine_backtest_results.py`

#### Optimize Scripts → `bin/optimize/`
- (Move from scripts/opt/)

#### Diagnostic Scripts → `bin/diagnose/`
- `diagnose_eth_runtime.py`
- `diagnose_gmm_labeling.py`
- `diagnose_spy_runtime.py`
- `debug_adaptive_logic.py`

#### CLI Scripts → `bin/cli/`
- `bull_machine_cli.py`

#### Archive/Deprecated → `bin/archive/`
- `test_ob_high_optimization.py`
- `consolidate_trials.py`
- `copy_legacy_thresholds.py`
- All `test_*.py` scripts

**Total scripts to organize:** ~130 files in bin/ + ~30 in scripts/

### 3.3 Data Directory Restructure

**Create structure:**
```bash
mkdir -p data/{raw/{binance,bybit,tradingview,yfinance},processed/{features_mtf,macro,cache},archive}
```

**File mappings:**

| Source | Destination | Type |
|--------|-------------|------|
| `data/*.csv` (raw TradingView exports) | `data/raw/tradingview/*.csv` | Raw data |
| `data/features_mtf/*` | `data/processed/features_mtf/*` | Processed |
| `data/macro/*` | `data/processed/macro/*` | Processed |
| `data/wyckoff_cache/*` | `data/processed/cache/wyckoff_cache/*` | Cache |
| `data/regime_labels_*.parquet` | `data/processed/macro/regime_labels_*.parquet` | Processed |
| Old feature stores | `data/archive/` | Archive |

**Note**: Many CSV files are symlinks to external directories - preserve these.

### 3.4 Scripts Directory Consolidation

**Strategy**: Merge `scripts/` into `bin/` structure

| Source | Destination | Notes |
|--------|-------------|-------|
| `scripts/backtests/*` | `bin/backtest/` | Merge |
| `scripts/analysis/*` | `bin/analyze/` | Merge |
| `scripts/debug/*` | `bin/diagnose/` | Merge |
| `scripts/opt/*` | `bin/optimize/` | Move |
| `scripts/legacy/*` | `bin/archive/legacy/` | Archive |
| `scripts/research/*` | `bin/archive/research/` | Archive |
| `scripts/*.sh` (build tools) | `bin/build/` | Build scripts |
| `scripts/deployment/*` | Keep separate or move to tools/ | TBD |

**After consolidation**: Remove empty `scripts/` directory or repurpose for CI-only scripts

---

## 4. Module Split Plan

### 4.1 Archetype Logic Split

**Source file**: `engine/archetypes/logic_v2_adapter.py` (1,440 lines)

**Target structure**:
```
engine/strategies/archetypes/
├── __init__.py              # Main exports
├── base.py                  # Base class, helper functions
├── bull/
│   ├── __init__.py
│   ├── trap_reversal.py         # _check_A → TrapReversalDetector
│   ├── order_block_retest.py    # _check_B → OrderBlockRetestDetector
│   ├── fvg_continuation.py      # _check_C → FVGContinuationDetector
│   ├── wick_trap.py             # _check_K → WickTrapDetector
│   ├── trap_within_trend.py     # _check_H → TrapWithinTrendDetector
│   ├── volume_exhaustion.py     # _check_L → VolumeExhaustionDetector
│   ├── expansion_exhaustion.py  # _check_F → ExpansionExhaustionDetector
│   ├── failed_continuation.py   # _check_D → FailedContinuationDetector
│   ├── re_accumulate.py         # _check_G → ReAccumulateDetector
│   ├── liquidity_compression.py # _check_E → LiquidityCompressionDetector
│   └── ratio_coil_break.py      # _check_M → RatioCoilBreakDetector
└── bear/
    ├── __init__.py
    ├── breakdown.py             # _check_S1 → BreakdownDetector
    ├── failed_rally.py          # _check_S2 → FailedRallyDetector
    ├── whipsaw.py               # _check_S3 → WhipsawDetector
    ├── distribution.py          # _check_S4 → DistributionDetector
    ├── long_squeeze.py          # _check_S5 → LongSqueezeDetector
    ├── alt_rotation_down.py     # _check_S6 → AltRotationDownDetector
    ├── curve_inversion.py       # _check_S7 → CurveInversionDetector
    └── volume_fade_chop.py      # _check_S8 → VolumeFadeChopDetector
```

### 4.2 Extraction Details

#### A. Base Module (`engine/strategies/archetypes/base.py`)

**Extract from logic_v2_adapter.py:**
- Lines 1-52: Module docstring, imports, helper functions
  - `_get_first()`
  - `_norm01()`
- Lines 54-180: Base class infrastructure
  - `ArchetypeLogic.__init__()`
  - `ArchetypeLogic.g()`
  - `ArchetypeLogic._wyckoff_score()`
  - `ArchetypeLogic._momentum_score()`
  - `ArchetypeLogic._liquidity_score()`
  - `ArchetypeLogic._fusion()`

**New class structure:**
```python
class BaseArchetypeDetector:
    """Base class for all archetype detectors."""

    def __init__(self, config: dict):
        # Common initialization
        pass

    def detect(self, context: RuntimeContext) -> Tuple[bool, float, dict]:
        """
        Detect archetype match.

        Returns:
            (matched, score, metadata)
        """
        raise NotImplementedError

    # Helper methods: g(), _wyckoff_score(), _momentum_score(), etc.
```

#### B. Bull Archetype Modules (11 files)

Each bull archetype becomes a standalone module:

**Example: `engine/strategies/archetypes/bull/trap_reversal.py`**

```python
"""
Archetype A: Trap Reversal

Classic trap-to-reversal pattern with Wyckoff spring validation.
"""

from typing import Tuple, Optional
import pandas as pd
from engine.runtime.context import RuntimeContext
from engine.strategies.archetypes.base import BaseArchetypeDetector

class TrapReversalDetector(BaseArchetypeDetector):
    """
    Archetype A: Trap Reversal

    Detects: Spring/trap pattern with wyckoff confirmation and liquidity.
    """

    ARCHETYPE_LETTER = 'A'
    ARCHETYPE_NAME = 'trap_reversal'
    PRIORITY = 1

    def detect(self, context: RuntimeContext) -> Tuple[bool, float, dict]:
        """
        Check for trap reversal pattern.

        Original logic from _check_A() in logic_v2_adapter.py
        """
        row = context.row
        thresholds = context.thresholds

        # Original _check_A logic here
        # ...

        return matched, score, metadata
```

**Extraction mapping:**

| Archetype | Letter | Method | New File |
|-----------|--------|--------|----------|
| Trap Reversal | A | `_check_A()` | `bull/trap_reversal.py` |
| Order Block Retest | B | `_check_B()` | `bull/order_block_retest.py` |
| FVG Continuation | C | `_check_C()` | `bull/fvg_continuation.py` |
| Wick Trap | K | `_check_K()` | `bull/wick_trap.py` |
| Trap Within Trend | H | `_check_H()` | `bull/trap_within_trend.py` |
| Volume Exhaustion | L | `_check_L()` | `bull/volume_exhaustion.py` |
| Expansion Exhaustion | F | `_check_F()` | `bull/expansion_exhaustion.py` |
| Failed Continuation | D | `_check_D()` | `bull/failed_continuation.py` |
| Re-Accumulate | G | `_check_G()` | `bull/re_accumulate.py` |
| Liquidity Compression | E | `_check_E()` | `bull/liquidity_compression.py` |
| Ratio Coil Break | M | `_check_M()` | `bull/ratio_coil_break.py` |

#### C. Bear Archetype Modules (8 files)

Same pattern for bear archetypes:

| Archetype | Letter | Method | New File |
|-----------|--------|--------|----------|
| Breakdown | S1 | `_check_S1()` | `bear/breakdown.py` |
| Failed Rally | S2 | `_check_S2()` | `bear/failed_rally.py` |
| Whipsaw | S3 | `_check_S3()` | `bear/whipsaw.py` |
| Distribution | S4 | `_check_S4()` | `bear/distribution.py` |
| Long Squeeze | S5 | `_check_S5()` | `bear/long_squeeze.py` |
| Alt Rotation Down | S6 | `_check_S6()` | `bear/alt_rotation_down.py` |
| Curve Inversion | S7 | `_check_S7()` | `bear/curve_inversion.py` |
| Volume Fade Chop | S8 | `_check_S8()` | `bear/volume_fade_chop.py` |

#### D. Adapter/Facade Layer

**File**: `engine/archetypes/__init__.py`

```python
"""
Archetype detection facade for backward compatibility.

This module re-exports archetype detectors from the new modular structure
to maintain backward compatibility with existing code.
"""

# Import from new modular structure
from engine.strategies.archetypes.bull.trap_reversal import TrapReversalDetector
from engine.strategies.archetypes.bull.order_block_retest import OrderBlockRetestDetector
from engine.strategies.archetypes.bull.fvg_continuation import FVGContinuationDetector
# ... (all other archetypes)

from engine.strategies.archetypes.bear.failed_rally import FailedRallyDetector
from engine.strategies.archetypes.bear.long_squeeze import LongSqueezeDetector
# ... (all bear archetypes)

# Maintain the old ArchetypeLogic class as a facade
from engine.strategies.archetypes.logic_adapter import ArchetypeLogic

__all__ = [
    'ArchetypeLogic',
    'TrapReversalDetector',
    'OrderBlockRetestDetector',
    # ... (all detectors)
]
```

**File**: `engine/archetypes/logic_adapter.py` (new facade)

```python
"""
Backward compatibility adapter for ArchetypeLogic.

This class maintains the old API while delegating to new modular detectors.
"""

from typing import Tuple, Optional, Dict
from engine.runtime.context import RuntimeContext

# Import all detectors
from engine.strategies.archetypes.bull import *
from engine.strategies.archetypes.bear import *

class ArchetypeLogic:
    """
    Facade maintaining backward compatibility with old ArchetypeLogic API.

    Internally uses new modular detector classes.
    """

    CLASS_VERSION = "archetypes/logic_v2_adapter@r2_modular"

    def __init__(self, config: dict):
        """Initialize with archetype config."""
        self.config = config

        # Initialize all detector instances
        self.detectors = {
            'A': TrapReversalDetector(config),
            'B': OrderBlockRetestDetector(config),
            # ... (all detectors)
        }

        # Maintain enabled flags
        self.enabled = {letter: config.get(f'enable_{letter}', True)
                       for letter in self.detectors.keys()}

    def detect(self, context: RuntimeContext) -> Tuple[Optional[str], float, float]:
        """
        Main detection method - maintains old API signature.

        Delegates to new modular detectors.
        """
        # Original dispatch logic from logic_v2_adapter.py
        # But calls detector.detect() instead of self._check_X()
        # ...
        pass

    # Keep old methods for backward compat if needed
    def check_archetype(self, row, prev_row, df, index):
        """Deprecated: Use detect() with RuntimeContext instead."""
        # Build RuntimeContext from old params
        # Call detect()
        pass
```

**Archive old file**: `engine/archetypes/_legacy.py`

- Copy entire `logic_v2_adapter.py` to `engine/archetypes/_legacy.py` for reference
- Add deprecation warning at top

### 4.3 Import Shim Strategy

**Goal**: Ensure zero import breakage

**Locations that import ArchetypeLogic:**

```bash
# Find all imports
grep -r "from engine.archetypes.logic_v2_adapter import" .
grep -r "from engine.archetypes import ArchetypeLogic" .
```

**Strategy**:
1. Keep `engine/archetypes/__init__.py` exporting `ArchetypeLogic`
2. Create `engine/archetypes/logic_adapter.py` as facade
3. Move `logic_v2_adapter.py` to `_legacy.py` for reference
4. All imports continue to work via facade

**Example import paths (all work after migration):**

```python
# Old style (still works via facade)
from engine.archetypes.logic_v2_adapter import ArchetypeLogic

# New recommended style (still works via __init__)
from engine.archetypes import ArchetypeLogic

# New modular style (direct access to detectors)
from engine.strategies.archetypes.bull import TrapReversalDetector
from engine.strategies.archetypes.bear import FailedRallyDetector
```

---

## 5. Migration Steps

### Phase 0: Preparation (Day 0)

#### Step 0.1: Create Feature Branch
```bash
cd /Users/raymondghandchi/Bull-machine-/Bull-machine-
git checkout -b repo-restructure-2025-11-14
```

#### Step 0.2: Baseline Validation
```bash
# Run full test suite
pytest tests/ -v

# Run key backtests and save outputs
python bin/backtest_knowledge_v2.py \
    --config configs/mvp_bull_market_v1.json \
    --symbol BTC \
    --start 2022-01-01 \
    --end 2024-12-31 \
    > baseline_mvp_bull_btc.json

# Save baseline results to results_reference/
mkdir -p results_reference/restructure_baseline/
cp baseline_mvp_bull_btc.json results_reference/restructure_baseline/
```

#### Step 0.3: Audit Current Imports
```bash
# Find all imports of modules we'll move
grep -r "from engine.archetypes.logic_v2_adapter" . > import_audit_archetypes.txt
grep -r "from bin\." . > import_audit_bin.txt
grep -r "import bin\." . >> import_audit_bin.txt

# Save audit for later validation
mv import_audit_*.txt results_reference/restructure_baseline/
```

### Phase 1: Directory Structure (Day 1 Morning)

#### Step 1.1: Create New Directory Structure
```bash
# Configs
mkdir -p configs/{frozen,mvp,experiments/{optuna,bear,adaptive},regime,archive}

# Data
mkdir -p data/{raw/{binance,bybit,tradingview,yfinance},processed/{features_mtf,macro,cache},archive}

# Bin
mkdir -p bin/{backtest,build,backfill,download,analyze,optimize,diagnose,cli,archive}

# Engine strategies
mkdir -p engine/strategies/archetypes/{bull,bear}

# Create __init__.py files
touch engine/strategies/__init__.py
touch engine/strategies/archetypes/__init__.py
touch engine/strategies/archetypes/bull/__init__.py
touch engine/strategies/archetypes/bear/__init__.py

# Create README files
touch configs/README.md
touch data/README.md
touch bin/README.md
```

#### Step 1.2: Create Documentation Stubs
```bash
# Create ARCHITECTURE.md
touch docs/ARCHITECTURE.md

# Create migration tracking docs
touch docs/reports/RESTRUCTURE_MIGRATION_LOG.md
```

### Phase 2: Config Migration (Day 1 Morning)

#### Step 2.1: Move Frozen Baselines
```bash
cd /Users/raymondghandchi/Bull-machine-/Bull-machine-

# Frozen baselines
git mv configs/baseline_btc_bull_pf20.json configs/frozen/
git mv configs/baseline_btc_bull_pf20_biased_20pct.json configs/frozen/
git mv configs/baseline_btc_bull_pf20_biased.json configs/frozen/
git mv configs/baseline_btc_bull_pf20_biased_20pct_no_ml.json configs/frozen/
git mv configs/baseline_btc_bull_pf20_biased_20pct_no_ml_lowgate.json configs/frozen/
git mv configs/baseline_btc_bull_ob_expanded_v1.json configs/frozen/
git mv configs/baseline_btc_bull_regime_routed_v1.json configs/frozen/
git mv configs/baseline_btc_bull_stabilized_v1.json configs/frozen/
git mv configs/baseline_btc_bear_archetypes_adaptive_v3.2.json configs/frozen/
git mv configs/baseline_btc_bear_archetypes_adaptive_v3.2_state_gates.json configs/frozen/
git mv configs/baseline_btc_adaptive_pr6b.json configs/frozen/
```

#### Step 2.2: Move MVP Configs
```bash
git mv configs/mvp_bull_market_v1.json configs/mvp/
git mv configs/mvp_bear_market_v1.json configs/mvp/
git mv configs/regime_routing_production_v1.json configs/mvp/
```

#### Step 2.3: Move Experimental Configs
```bash
# Bear experiments
git mv configs/bear_archetypes_phase1.json configs/experiments/bear/
git mv configs/baseline_btc_bear_archetypes_adaptive.json configs/experiments/bear/
git mv configs/baseline_btc_bear_archetypes_test.json configs/experiments/bear/
git mv configs/baseline_btc_bear_defensive.json configs/experiments/bear/

# Adaptive experiments
git mv configs/adaptive/* configs/experiments/adaptive/ 2>/dev/null || true
```

#### Step 2.4: Archive Old Versions
```bash
# v7/v8 configs
git mv configs/btc_v7_ml_*.json configs/archive/
git mv configs/btc_v8_*.json configs/archive/

# Knowledge v2
git mv configs/knowledge_v2 configs/archive/

# v2/v3 replays
git mv configs/v2 configs/archive/
git mv configs/v3_replay_2024 configs/archive/
```

#### Step 2.5: Move Regime Profiles
```bash
git mv configs/profile_*.json configs/regime/
```

#### Step 2.6: Move Special Configs
```bash
# Paper trading
git mv configs/paper_trading configs/paper_trading

# Live
git mv configs/live configs/live

# Schema
git mv configs/schema configs/schema
```

### Phase 3: Bin Scripts Migration (Day 1 Afternoon)

#### Step 3.1: Move Backtest Scripts
```bash
git mv bin/backtest_knowledge_v2.py bin/backtest/
git mv bin/backtest_router_v10_integrated.py bin/backtest/
git mv bin/backtest_router_v10_full.py bin/backtest/
git mv bin/backtest_router_v10.py bin/backtest/

# From scripts/
git mv scripts/backtests/fast_monthly_test.py bin/backtest/
```

#### Step 3.2: Move Build Scripts
```bash
git mv bin/build_mtf_feature_store.py bin/build/
git mv bin/build_feature_store_v2.py bin/build/
git mv bin/build_feature_store.py bin/build/
git mv bin/build_macro_dataset.py bin/build/
git mv bin/build_wyckoff_cache.py bin/build/
git mv bin/add_gmm_features.py bin/build/
git mv bin/add_p1_features_to_macro.py bin/build/
git mv bin/append_macro_to_feature_store.py bin/build/
git mv bin/cache_features_with_regime.py bin/build/
```

#### Step 3.3: Move Backfill Scripts
```bash
git mv bin/backfill_liquidity_score_optimized.py bin/backfill/
git mv bin/backfill_liquidity_score.py bin/backfill/
git mv bin/backfill_ob_high_optimized.py bin/backfill/
git mv bin/backfill_ob_high.py bin/backfill/
git mv bin/backfill_missing_macro_features.py bin/backfill/
git mv bin/fix_oi_change_pipeline.py bin/backfill/
```

#### Step 3.4: Move Download Scripts
```bash
git mv bin/download_binance_data.py bin/download/
git mv bin/download_cryptocompare_data.py bin/download/
git mv bin/download_all_data.sh bin/download/
```

#### Step 3.5: Move Analysis Scripts
```bash
git mv bin/analyze_archetype_perf.py bin/analyze/
git mv bin/analyze_optimization.py bin/analyze/
git mv bin/analyze_pareto_frontiers_v10.py bin/analyze/
git mv bin/analyze_v3_trades_full_year.py bin/analyze/
git mv bin/compare_baseline_vs_ml.py bin/analyze/
git mv bin/compare_knowledge_v2_abc.py bin/analyze/
git mv bin/combine_backtest_results.py bin/analyze/
```

#### Step 3.6: Move Diagnostic Scripts
```bash
git mv bin/diagnose_eth_runtime.py bin/diagnose/
git mv bin/diagnose_gmm_labeling.py bin/diagnose/
git mv bin/diagnose_spy_runtime.py bin/diagnose/
git mv bin/debug_adaptive_logic.py bin/diagnose/
```

#### Step 3.7: Move CLI Scripts
```bash
git mv bin/bull_machine_cli.py bin/cli/
```

#### Step 3.8: Archive Deprecated Scripts
```bash
# Test scripts
git mv bin/test_*.py bin/archive/ 2>/dev/null || true

# Other deprecated
git mv bin/consolidate_trials.py bin/archive/
git mv bin/copy_legacy_thresholds.py bin/archive/
```

#### Step 3.9: Consolidate Scripts Directory
```bash
# Move remaining useful scripts from scripts/ to bin/
git mv scripts/backtests/* bin/backtest/ 2>/dev/null || true
git mv scripts/analysis/* bin/analyze/ 2>/dev/null || true
git mv scripts/debug/* bin/diagnose/ 2>/dev/null || true
git mv scripts/opt/* bin/optimize/ 2>/dev/null || true

# Archive legacy
git mv scripts/legacy bin/archive/legacy
git mv scripts/research bin/archive/research
```

### Phase 4: Data Directory Migration (Day 1 Afternoon)

#### Step 4.1: Move Raw Data
```bash
# TradingView exports (CSVs - many are symlinks, be careful)
# Only move actual files, not symlinks
find data -maxdepth 1 -type f -name "*.csv" -exec git mv {} data/raw/tradingview/ \;
```

#### Step 4.2: Move Processed Data
```bash
git mv data/features_mtf data/processed/
git mv data/macro data/processed/
git mv data/wyckoff_cache data/processed/cache/
git mv data/regime_labels_*.parquet data/processed/macro/
```

#### Step 4.3: Archive Old Data
```bash
# Old feature stores (identify by date or naming)
# TBD: Manual review required
```

### Phase 5: Archetype Logic Split (Day 2 Morning)

**CRITICAL**: This is the most complex step. Requires careful extraction and testing.

#### Step 5.1: Create Base Module

```bash
# Create base.py with common infrastructure
cat > engine/strategies/archetypes/base.py << 'EOF'
#!/usr/bin/env python3
"""
Base class and utilities for archetype detectors.

Extracted from logic_v2_adapter.py during 2025-11-14 restructure.
"""

import pandas as pd
import numpy as np
import logging
from typing import Tuple, Optional, Dict
from engine.runtime.context import RuntimeContext

logger = logging.getLogger(__name__)


def _get_first(row, keys, default=0.0):
    """Get first non-null value from list of column names."""
    for k in keys:
        if k in row.index and row[k] is not None and not pd.isna(row[k]):
            return row[k]
    return default


def _norm01(x, lo, hi):
    """Normalize value to [0, 1] range."""
    if hi == lo:
        return 0.0
    v = (x - lo) / (hi - lo)
    return max(0.0, min(1.0, v))


class BaseArchetypeDetector:
    """
    Base class for all archetype detectors.

    Provides common utilities for feature access, score calculation,
    and threshold management.
    """

    # Subclasses must define
    ARCHETYPE_LETTER = None
    ARCHETYPE_NAME = None
    PRIORITY = None

    def __init__(self, config: dict):
        """Initialize detector with config."""
        self.config = config

        # Extract thresholds (for backward compat)
        thresholds = config.get('thresholds', {})
        self.min_liquidity = thresholds.get('min_liquidity', 0.30)

        # Fusion weights
        self.fusion_weights = config.get('fusion_weights', {
            'wyckoff': 0.331,
            'liquidity': 0.392,
            'momentum': 0.205
        })
        self.fakeout_penalty = config.get('fakeout_penalty', 0.25)

    def g(self, row: pd.Series, key: str, default=0.0):
        """Generic getter with multiple column name attempts."""
        # Try exact key first
        if key in row.index and row[key] is not None and not pd.isna(row[key]):
            return row[key]

        # Try TF-prefixed variants
        for tf_prefix in ['tf1d_', 'tf4h_', 'tf1h_']:
            prefixed_key = tf_prefix + key
            if prefixed_key in row.index and row[prefixed_key] is not None and not pd.isna(row[prefixed_key]):
                return row[prefixed_key]

        return default

    def _wyckoff_score(self, row: pd.Series) -> float:
        """Get or recompute wyckoff score."""
        # Prefer existing score
        wy = self.g(row, "wyckoff_score", None)
        if wy is not None:
            return max(0.0, min(1.0, wy))

        # Fallback: derive from phase
        phase = row.get("wyckoff_phase", "")
        if isinstance(phase, str):
            if phase.startswith("Spring"):
                return 0.90
            elif phase.startswith("BUEC"):
                return 0.85
            elif phase.startswith("SOS"):
                return 0.80
        return 0.0

    def _momentum_score(self, row: pd.Series) -> float:
        """Calculate momentum score from RSI, ADX, volume z-score."""
        rsi = self.g(row, "rsi", 50.0)
        adx = self.g(row, "adx", 0.0)
        vol_z = self.g(row, "volume_z", 0.0)

        rsi_comp = _norm01(abs(rsi - 50.0), 0.0, 25.0)
        adx_comp = _norm01(adx, 10.0, 40.0)
        vol_comp = max(0.0, min(1.0, vol_z / 2.0))

        return 0.4 * rsi_comp + 0.3 * adx_comp + 0.3 * vol_comp

    def _liquidity_score(self, row: pd.Series) -> float:
        """Get or derive liquidity score."""
        # Prefer existing
        if "liquidity_score" in row.index and row["liquidity_score"] is not None:
            val = row["liquidity_score"]
            if not pd.isna(val):
                return max(0.0, min(1.0, val))

        # Fallback: derive from BOMS, FVG, displacement
        bstr = self.g(row, "boms_strength", 0.0)
        fvg_1h = 1.0 if self.g(row, "fvg_present_1h", 0) else 0.0
        fvg_4h = 1.0 if self.g(row, "fvg_present_4h", 0) else 0.0
        fvg = max(fvg_1h, fvg_4h)

        atr = max(self.g(row, "atr", 0.0), 1e-9)
        disp = self.g(row, "boms_disp", 0.0)
        disp_n = max(0.0, min(1.0, disp / (2.0 * atr)))

        return 0.5 * bstr + 0.25 * fvg + 0.25 * disp_n

    def _fusion(self, row: pd.Series) -> float:
        """Get or recompute fusion score."""
        fuse = self.g(row, "fusion_score", None)
        if fuse is not None:
            return max(0.0, min(1.0, fuse))

        # Recompute
        w = self.fusion_weights
        wy = self._wyckoff_score(row)
        liq = self._liquidity_score(row)
        mom = self._momentum_score(row)
        fake = row.get("fakeout_score", 0.0) or 0.0

        f = (w.get("wyckoff", 0.331) * wy +
             w.get("liquidity", 0.392) * liq +
             w.get("momentum", 0.205) * mom -
             self.fakeout_penalty * fake)

        return max(0.0, min(1.0, f))

    def detect(self, context: RuntimeContext) -> Tuple[bool, float, dict]:
        """
        Detect archetype match.

        Must be implemented by subclasses.

        Args:
            context: RuntimeContext with row, regime, thresholds

        Returns:
            (matched: bool, score: float, metadata: dict)
        """
        raise NotImplementedError(f"{self.__class__.__name__} must implement detect()")
EOF
```

#### Step 5.2: Extract Bull Archetypes (Example: Trap Reversal)

```bash
# Create trap_reversal.py
cat > engine/strategies/archetypes/bull/trap_reversal.py << 'EOF'
#!/usr/bin/env python3
"""
Archetype A: Trap Reversal

Classic trap-to-reversal pattern with Wyckoff spring validation.

Extracted from logic_v2_adapter.py during 2025-11-14 restructure.
Original method: _check_A()
"""

from typing import Tuple
from engine.runtime.context import RuntimeContext
from engine.strategies.archetypes.base import BaseArchetypeDetector


class TrapReversalDetector(BaseArchetypeDetector):
    """
    Archetype A: Trap Reversal

    Detects classic spring/trap patterns with wyckoff confirmation.

    Criteria:
    - Wyckoff phase = Spring* or high wyckoff_score
    - Fusion score above threshold
    - Liquidity present
    - Volume confirmation
    """

    ARCHETYPE_LETTER = 'A'
    ARCHETYPE_NAME = 'trap_reversal'
    PRIORITY = 1

    def detect(self, context: RuntimeContext) -> Tuple[bool, float, dict]:
        """Check for trap reversal pattern."""
        row = context.row
        thresholds = context.thresholds

        # Get scores
        wy_score = self._wyckoff_score(row)
        fusion_score = self._fusion(row)
        liq_score = self._liquidity_score(row)

        # Thresholds
        wy_th = thresholds.get('wyckoff_threshold', 0.65)
        fusion_th = thresholds.get('fusion_threshold', 0.50)
        liq_th = thresholds.get('liquidity_threshold', 0.30)

        # Check criteria
        wyckoff_ok = wy_score >= wy_th
        fusion_ok = fusion_score >= fusion_th
        liquidity_ok = liq_score >= liq_th

        # Optional: Volume confirmation
        vol_z = self.g(row, "volume_z", 0.0)
        volume_ok = vol_z >= 0.5  # Threshold TBD

        matched = wyckoff_ok and fusion_ok and liquidity_ok and volume_ok

        metadata = {
            'wyckoff_score': wy_score,
            'fusion_score': fusion_score,
            'liquidity_score': liq_score,
            'volume_z': vol_z,
            'wyckoff_threshold': wy_th,
            'fusion_threshold': fusion_th
        }

        return matched, fusion_score if matched else 0.0, metadata
EOF
```

**Repeat for all 11 bull archetypes** (B, C, D, E, F, G, H, K, L, M)

**Note**: Each archetype's `detect()` method is extracted from its corresponding `_check_X()` method in `logic_v2_adapter.py`. The logic must be copied **exactly** to maintain behavior.

#### Step 5.3: Extract Bear Archetypes

**Same process for S1-S8** (breakdown, failed_rally, whipsaw, distribution, long_squeeze, alt_rotation_down, curve_inversion, volume_fade_chop)

**Special attention to S5 (Long Squeeze)**: This has the corrected funding logic from recent fixes.

#### Step 5.4: Create __init__.py Exports

```bash
# engine/strategies/archetypes/bull/__init__.py
cat > engine/strategies/archetypes/bull/__init__.py << 'EOF'
"""Bull market archetype detectors."""

from engine.strategies.archetypes.bull.trap_reversal import TrapReversalDetector
from engine.strategies.archetypes.bull.order_block_retest import OrderBlockRetestDetector
from engine.strategies.archetypes.bull.fvg_continuation import FVGContinuationDetector
from engine.strategies.archetypes.bull.wick_trap import WickTrapDetector
from engine.strategies.archetypes.bull.trap_within_trend import TrapWithinTrendDetector
from engine.strategies.archetypes.bull.volume_exhaustion import VolumeExhaustionDetector
from engine.strategies.archetypes.bull.expansion_exhaustion import ExpansionExhaustionDetector
from engine.strategies.archetypes.bull.failed_continuation import FailedContinuationDetector
from engine.strategies.archetypes.bull.re_accumulate import ReAccumulateDetector
from engine.strategies.archetypes.bull.liquidity_compression import LiquidityCompressionDetector
from engine.strategies.archetypes.bull.ratio_coil_break import RatioCoilBreakDetector

__all__ = [
    'TrapReversalDetector',
    'OrderBlockRetestDetector',
    'FVGContinuationDetector',
    'WickTrapDetector',
    'TrapWithinTrendDetector',
    'VolumeExhaustionDetector',
    'ExpansionExhaustionDetector',
    'FailedContinuationDetector',
    'ReAccumulateDetector',
    'LiquidityCompressionDetector',
    'RatioCoilBreakDetector',
]
EOF

# engine/strategies/archetypes/bear/__init__.py
cat > engine/strategies/archetypes/bear/__init__.py << 'EOF'
"""Bear market archetype detectors."""

from engine.strategies.archetypes.bear.breakdown import BreakdownDetector
from engine.strategies.archetypes.bear.failed_rally import FailedRallyDetector
from engine.strategies.archetypes.bear.whipsaw import WhipsawDetector
from engine.strategies.archetypes.bear.distribution import DistributionDetector
from engine.strategies.archetypes.bear.long_squeeze import LongSqueezeDetector
from engine.strategies.archetypes.bear.alt_rotation_down import AltRotationDownDetector
from engine.strategies.archetypes.bear.curve_inversion import CurveInversionDetector
from engine.strategies.archetypes.bear.volume_fade_chop import VolumeFadeChopDetector

__all__ = [
    'BreakdownDetector',
    'FailedRallyDetector',
    'WhipsawDetector',
    'DistributionDetector',
    'LongSqueezeDetector',
    'AltRotationDownDetector',
    'CurveInversionDetector',
    'VolumeFadeChopDetector',
]
EOF
```

#### Step 5.5: Create Backward Compatibility Adapter

```bash
# Create logic_adapter.py facade
cat > engine/archetypes/logic_adapter.py << 'EOF'
#!/usr/bin/env python3
"""
Backward compatibility adapter for ArchetypeLogic.

Maintains the old API while delegating to new modular detectors.
Created during 2025-11-14 restructure.
"""

import logging
from typing import Tuple, Optional, Dict
import pandas as pd

from engine.runtime.context import RuntimeContext

# Import all detectors
from engine.strategies.archetypes.bull import *
from engine.strategies.archetypes.bear import *

logger = logging.getLogger(__name__)


class ArchetypeLogic:
    """
    Facade maintaining backward compatibility with old ArchetypeLogic API.

    Internally uses new modular detector classes.
    """

    CLASS_VERSION = "archetypes/logic_v2_adapter@r2_modular"

    def __init__(self, config: dict):
        """Initialize with archetype config."""
        self.config = config
        self.use_archetypes = config.get('use_archetypes', False)

        logger.info(f"[ArchetypeLogic] Using {self.CLASS_VERSION}")
        logger.info(f"[ArchetypeLogic] Modular detector architecture")

        # Initialize all detector instances
        self.detectors = {
            'A': TrapReversalDetector(config),
            'B': OrderBlockRetestDetector(config),
            'C': FVGContinuationDetector(config),
            'D': FailedContinuationDetector(config),
            'E': LiquidityCompressionDetector(config),
            'F': ExpansionExhaustionDetector(config),
            'G': ReAccumulateDetector(config),
            'H': TrapWithinTrendDetector(config),
            'K': WickTrapDetector(config),
            'L': VolumeExhaustionDetector(config),
            'M': RatioCoilBreakDetector(config),
            'S1': BreakdownDetector(config),
            'S2': FailedRallyDetector(config),
            'S3': WhipsawDetector(config),
            'S4': DistributionDetector(config),
            'S5': LongSqueezeDetector(config),
            'S6': AltRotationDownDetector(config),
            'S7': CurveInversionDetector(config),
            'S8': VolumeFadeChopDetector(config),
        }

        # Maintain enabled flags
        self.enabled = {
            'A': config.get('enable_A', True),
            'B': config.get('enable_B', True),
            'C': config.get('enable_C', True),
            'D': config.get('enable_D', True),
            'E': config.get('enable_E', True),
            'F': config.get('enable_F', True),
            'G': config.get('enable_G', True),
            'H': config.get('enable_H', True),
            'K': config.get('enable_K', True),
            'L': config.get('enable_L', True),
            'M': config.get('enable_M', True),
            'S1': config.get('enable_S1', False),
            'S2': config.get('enable_S2', True),
            'S3': config.get('enable_S3', False),
            'S4': config.get('enable_S4', False),
            'S5': config.get('enable_S5', True),
            'S6': config.get('enable_S6', False),
            'S7': config.get('enable_S7', False),
            'S8': config.get('enable_S8', False),
        }

    def detect(self, context: RuntimeContext) -> Tuple[Optional[str], float, float]:
        """
        Main detection method - maintains old API signature.

        Delegates to new modular detectors.

        NOTE: This implementation is copied from logic_v2_adapter.py
        with detector.detect() calls instead of self._check_X().
        """
        # Extract common scores (for return values)
        row = context.row

        # Use first detector to calculate global scores
        # (all detectors share same calculation logic via BaseArchetypeDetector)
        first_detector = next(iter(self.detectors.values()))
        global_fusion_score = first_detector._fusion(row)
        liquidity_score = first_detector._liquidity_score(row)

        # Archetype map with priorities
        archetype_map = {
            'A': ('trap_reversal', 1),
            'B': ('order_block_retest', 2),
            'C': ('fvg_continuation', 3),
            'K': ('wick_trap', 4),
            'H': ('trap_within_trend', 5),
            'L': ('volume_exhaustion', 6),
            'F': ('expansion_exhaustion', 7),
            'D': ('failed_continuation', 8),
            'G': ('re_accumulate', 9),
            'E': ('liquidity_compression', 10),
            'M': ('ratio_coil_break', 11),
            'S1': ('breakdown', 12),
            'S2': ('failed_rally', 13),
            'S3': ('whipsaw', 14),
            'S4': ('distribution', 15),
            'S5': ('long_squeeze', 16),
            'S6': ('alt_rotation_down', 17),
            'S7': ('curve_inversion', 18),
            'S8': ('volume_fade_chop', 19),
        }

        candidates = []

        # Evaluate all enabled archetypes
        for letter in archetype_map.keys():
            if not self.enabled.get(letter, False):
                continue

            detector = self.detectors[letter]
            name, priority = archetype_map[letter]

            matched, score, meta = detector.detect(context)

            if matched:
                candidates.append((name, score, meta, priority))
                logger.debug(f"[DISPATCH] {name} matched with score={score:.3f}")

        # No matches
        if not candidates:
            return None, global_fusion_score, liquidity_score

        # Apply regime routing (if configured)
        regime = context.regime_label
        routing_config = self.config.get('routing', {})
        regime_routing = routing_config.get(regime, {})
        regime_weights = regime_routing.get('weights', {})

        if regime_weights:
            logger.info(f"[REGIME ROUTING] regime={regime}, applying weights: {regime_weights}")
            adjusted_candidates = []
            for name, score, meta, priority in candidates:
                regime_mult = regime_weights.get(name, 1.0)
                adjusted_score = score * regime_mult
                adjusted_candidates.append((name, adjusted_score, meta, priority))
                if regime_mult != 1.0:
                    logger.info(f"[REGIME ROUTING] {name}: {score:.3f} × {regime_mult:.2f} = {adjusted_score:.3f}")
            candidates = adjusted_candidates

        # Pick best match by score (highest wins, priority breaks ties)
        candidates.sort(key=lambda x: (x[1], -x[3]), reverse=True)
        best = candidates[0]

        if len(candidates) > 1:
            logger.info(f"[DISPATCH] {len(candidates)} candidates: {[(c[0], f'{c[1]:.3f}') for c in candidates]} → chose {best[0]}")

        return best[0], best[1], liquidity_score

    # Deprecated backward compatibility method
    def check_archetype(
        self,
        row: pd.Series,
        prev_row: Optional[pd.Series],
        df: pd.DataFrame,
        index: int
    ) -> Tuple[Optional[str], float, float]:
        """
        DEPRECATED: Use detect() with RuntimeContext instead.

        Maintained for backward compatibility with old backtest code.
        """
        # Build minimal RuntimeContext from old params
        # (This may not have full regime/threshold info)
        from engine.runtime.context import RuntimeContext
        from engine.archetypes.threshold_policy import ThresholdPolicy

        # Create minimal threshold policy
        threshold_policy = ThresholdPolicy(self.config)

        # Infer regime from row (if available)
        regime_label = row.get('regime_label', 'neutral')

        # Build context
        context = RuntimeContext(
            row=row,
            prev_row=prev_row,
            df=df,
            index=index,
            regime_label=regime_label,
            thresholds=threshold_policy.get_thresholds(regime_label),
            threshold_policy=threshold_policy
        )

        return self.detect(context)
EOF
```

#### Step 5.6: Update engine/archetypes/__init__.py

```bash
cat > engine/archetypes/__init__.py << 'EOF'
"""
Archetype detection facade for backward compatibility.

This module re-exports archetype detectors from the new modular structure
to maintain backward compatibility with existing code.

New modular structure (2025-11-14):
- engine/strategies/archetypes/bull/* - Bull market archetypes
- engine/strategies/archetypes/bear/* - Bear market archetypes
- engine/archetypes/logic_adapter.py - Backward compatibility facade
"""

# Main facade (maintains old ArchetypeLogic API)
from engine.archetypes.logic_adapter import ArchetypeLogic

# Re-export all detectors for direct access
from engine.strategies.archetypes.bull import *
from engine.strategies.archetypes.bear import *

# Other archetype utilities (keep in place)
from engine.archetypes.registry import ArchetypeRegistry
from engine.archetypes.threshold_policy import ThresholdPolicy
from engine.archetypes.state_aware_gates import StateAwareGates, apply_state_aware_gate
from engine.archetypes.telemetry import ArchetypeTelemetry
from engine.archetypes.param_accessor import ParamAccessor

__all__ = [
    # Main facade
    'ArchetypeLogic',

    # Bull detectors
    'TrapReversalDetector',
    'OrderBlockRetestDetector',
    'FVGContinuationDetector',
    'WickTrapDetector',
    'TrapWithinTrendDetector',
    'VolumeExhaustionDetector',
    'ExpansionExhaustionDetector',
    'FailedContinuationDetector',
    'ReAccumulateDetector',
    'LiquidityCompressionDetector',
    'RatioCoilBreakDetector',

    # Bear detectors
    'BreakdownDetector',
    'FailedRallyDetector',
    'WhipsawDetector',
    'DistributionDetector',
    'LongSqueezeDetector',
    'AltRotationDownDetector',
    'CurveInversionDetector',
    'VolumeFadeChopDetector',

    # Utilities
    'ArchetypeRegistry',
    'ThresholdPolicy',
    'StateAwareGates',
    'apply_state_aware_gate',
    'ArchetypeTelemetry',
    'ParamAccessor',
]
EOF
```

#### Step 5.7: Archive Old Logic File

```bash
# Move old file to _legacy.py for reference
git mv engine/archetypes/logic_v2_adapter.py engine/archetypes/_legacy.py

# Add deprecation warning at top
cat - engine/archetypes/_legacy.py > /tmp/legacy_with_warning.py << 'EOF'
"""
DEPRECATED: This file is archived for reference only.

The archetype logic has been split into modular detectors:
- engine/strategies/archetypes/bull/* - Bull market archetypes
- engine/strategies/archetypes/bear/* - Bear market archetypes

Use the new ArchetypeLogic facade from engine.archetypes.logic_adapter
or import detectors directly from engine.strategies.archetypes.

This file is preserved for reference during the transition period.
Archived: 2025-11-14
"""

EOF

mv /tmp/legacy_with_warning.py engine/archetypes/_legacy.py
```

### Phase 6: .gitignore Updates (Day 2 Afternoon)

#### Step 6.1: Update .gitignore

```bash
# .gitignore already well-structured from recent cleanup
# Add any new exclusions if needed

cat >> .gitignore << 'EOF'

# ----------------------------------------------------------------------------
# 2025-11-14 Restructure Additions
# ----------------------------------------------------------------------------

# Data directory (processed/raw separation)
data/raw/**/*
!data/raw/.gitkeep
!data/raw/README.md

data/processed/**/*
!data/processed/.gitkeep
!data/processed/README.md

# Bin archive (deprecated scripts)
bin/archive/**/*
!bin/archive/README.md

# Config archive
configs/archive/**/*
!configs/archive/README.md

EOF
```

### Phase 7: Documentation Updates (Day 2 Afternoon)

#### Step 7.1: Create ARCHITECTURE.md

```bash
cat > docs/ARCHITECTURE.md << 'EOF'
# Bull Machine Architecture

**Last Updated**: 2025-11-14
**Status**: Production

## Overview

Bull Machine is a quantitative trading engine implementing multi-timeframe technical analysis with archetype-based pattern recognition and adaptive regime routing.

## Directory Structure

### Core Packages

#### `bull_machine/`
High-level orchestration and application entrypoints.

- `app/` - Application runners (backtest, live, paper trading)
- `core/` - Core utilities (config loading, data I/O, telemetry)
- `fusion/` - Signal fusion logic
- `signals/` - Signal processing and gating
- `backtest/` - Backtesting framework

#### `engine/`
Core trading engine logic.

- `strategies/archetypes/` - **Modular archetype detectors** (NEW 2025-11-14)
  - `bull/` - Bull market archetypes (A-M: 11 patterns)
  - `bear/` - Bear market archetypes (S1-S8: 8 patterns)
  - `base.py` - Base detector class and utilities
- `archetypes/` - Archetype adapter layer (backward compatibility)
- `fusion/` - Advanced fusion engines
- `context/` - Regime classification and macro analysis
- `exits/` - Exit strategy logic
- `features/` - Feature engineering
- `gates/` - Decision gates
- `liquidity/` - Liquidity analysis (BOMS, FVG)
- `smc/` - Smart money concepts (order blocks, liquidity sweeps)
- `temporal/` - Temporal analysis (Gann, cycles, TPI)
- `psychology/` - Trader psychology indicators (PTI, fakeout intensity)
- `risk/` - Risk management
- `runtime/` - Runtime context and state

### Data & Configuration

#### `configs/`
Configuration files organized by purpose.

- `frozen/` - Frozen baselines (never edit - ground truth)
- `mvp/` - Current MVP production configs
- `experiments/` - Experimental configs
  - `optuna/` - Optuna optimization sweeps
  - `bear/` - Bear archetype experiments
  - `adaptive/` - Adaptive logic experiments
- `regime/` - Regime routing profiles
- `paper_trading/` - Paper trading configs
- `live/` - Live trading configs
- `archive/` - Old versions (v7, v8, v150, etc.)

#### `data/`
Feature stores and data assets.

- `raw/` - Raw exchange/API data
  - `binance/`, `bybit/`, `tradingview/`, `yfinance/`
- `processed/` - Processed feature stores
  - `features_mtf/` - Multi-timeframe features (BTC, ETH, etc.)
  - `macro/` - Macro regime labels and features
  - `cache/` - Temporary computation cache
- `archive/` - Old feature store versions

### Scripts & Tools

#### `bin/`
Production scripts and CLI tools organized by purpose.

- `backtest/` - Backtest runners
- `build/` - Feature store builders
- `backfill/` - Data backfill scripts
- `download/` - Data download scripts
- `analyze/` - Analysis and reporting scripts
- `optimize/` - Optimization runners
- `diagnose/` - Diagnostic tools
- `cli/` - CLI entrypoints
- `archive/` - Deprecated scripts

#### `tools/`
Helper utilities for data fetching and processing.

#### `utils/`
Shared non-domain utilities.

### Documentation

#### `docs/`
Comprehensive documentation.

- `analysis/` - Trade analysis and pattern studies
- `archive/` - Historical documentation
- `audits/` - System audits
- `backtests/` - Backtest reports
- `guides/` - User guides and HOWTOs
- `reports/` - Technical reports and summaries
- `technical/` - Technical deep-dives

### Results & Artifacts

#### `results_reference/`
Curated reference results (ground truth benchmarks).

#### `results/`
Runtime/scratch results (gitignored).

#### `models/`
ML artifacts (GMM models, etc.).

#### `archive/`
Historical archive (old versions, deprecated code).

## Architecture Principles

### 1. Strategy Isolation

Archetypes are isolated in `engine/strategies/archetypes/{bull,bear}/` as standalone detector classes. This enables:
- Independent testing
- Easy extension
- Clear responsibility boundaries

### 2. Backward Compatibility

The `engine/archetypes/` package maintains a facade layer (`logic_adapter.py`) that preserves the old `ArchetypeLogic` API while delegating to new modular detectors.

### 3. Data Discipline

- **Raw data** stays in `data/raw/` (never modified)
- **Processed features** in `data/processed/` (versioned)
- **Archive** old versions in `data/archive/`
- **NaN validation** at feature store load time

### 4. Config Versioning

- **Frozen baselines** never edited (use for A/B testing)
- **MVP configs** for current production
- **Experiments** isolated from production
- **Archive** old versions

### 5. Script Organization

Scripts organized by **purpose** (backtest, build, analyze) not chronologically. Makes it easy to find the right tool.

## Key Subsystems

### Archetype Detection System

**Location**: `engine/strategies/archetypes/`

**Architecture**:
- **Base class**: `BaseArchetypeDetector` (common utilities)
- **Bull detectors**: 11 detectors in `bull/` (A-M)
- **Bear detectors**: 8 detectors in `bear/` (S1-S8)
- **Facade**: `ArchetypeLogic` in `engine/archetypes/logic_adapter.py`

**Flow**:
1. RuntimeContext constructed with row data, regime, thresholds
2. ArchetypeLogic.detect() called with context
3. All enabled detectors evaluated
4. Regime routing weights applied
5. Best match selected by score + priority

### Regime Routing System

**Location**: `engine/context/`

**Components**:
- `RegimeClassifier` - Detects bull/bear/neutral regimes
- `ThresholdPolicy` - Maps regimes to archetype thresholds
- `MacroEngine` - Processes macro features
- `RegimePolicy` - Routing logic

### Fusion System

**Location**: `engine/fusion/`

**Layers**:
- Domain fusion (Wyckoff + liquidity + momentum)
- Advanced fusion (ML-weighted)
- Knowledge-based fusion (K2)

### Exit System

**Location**: `engine/exits/`

**Strategies**:
- ATR-based trailing stops
- Time-based exits
- Regime-based exits
- Drawdown protection

## Import Patterns

### Recommended Imports

```python
# High-level orchestration
from bull_machine.core import config_loader, data_io
from bull_machine.app import main_backtest

# Engine core
from engine.archetypes import ArchetypeLogic
from engine.fusion import DomainFusion
from engine.context import RegimeClassifier

# Modular archetypes (direct access)
from engine.strategies.archetypes.bull import TrapReversalDetector
from engine.strategies.archetypes.bear import FailedRallyDetector
```

### Legacy Imports (Still Supported)

```python
# Old style (via facade)
from engine.archetypes.logic_v2_adapter import ArchetypeLogic  # Redirects to facade
```

## Testing Strategy

- **Unit tests**: `tests/unit/` - Test individual detectors, utilities
- **Integration tests**: `tests/integration/` - Full backtest runs on sample data
- **Smoke tests**: `tests/smoke/` - Quick sanity checks
- **Robustness tests**: `tests/robustness/` - Edge cases, NaN handling
- **Parity tests**: `tests/parity/` - Verify restructure maintains behavior

## Performance Considerations

- **Feature stores**: Parquet format (fast I/O)
- **Caching**: Wyckoff cache, computation cache
- **Vectorized operations**: NumPy/Pandas where possible
- **Profiling**: `profiles/` directory for perf analysis

## Future Evolution

- **ML integration**: PyTorch archetype training
- **Real-time execution**: WebSocket data feeds
- **Multi-asset**: Portfolio-level logic
- **Risk engine**: Position sizing, portfolio heat

---

**Questions?** See `docs/guides/` or `docs/technical/` for deep-dives.
EOF
```

#### Step 7.2: Create README files

```bash
# configs/README.md
cat > configs/README.md << 'EOF'
# Configuration Files

Configuration files organized by purpose.

## Directory Structure

- `frozen/` - Frozen baselines (never edit - use for A/B comparisons)
- `mvp/` - Current MVP production configs
- `experiments/` - Experimental configs (Optuna, bear archetypes, adaptive logic)
- `regime/` - Regime routing profiles
- `paper_trading/` - Paper trading configs
- `live/` - Live trading configs
- `archive/` - Old versions (v7, v8, v150, etc.)

## Config Format

All configs are JSON with the following top-level keys:

- `asset` - Asset symbol (BTC, ETH, etc.)
- `timeframe` - Primary timeframe
- `archetypes` - Archetype settings
  - `enable_A`, `enable_B`, ... - Enable flags
  - `thresholds` - Threshold overrides
  - `routing` - Regime routing weights
- `fusion` - Fusion settings
- `exits` - Exit strategy settings
- `regime` - Regime detection settings

## Usage

```python
from bull_machine.core.config_loader import load_config

config = load_config('configs/mvp/mvp_bull_market_v1.json')
```

## Best Practices

1. Never edit files in `frozen/` - copy to `experiments/` instead
2. Document changes in CHANGELOG.md
3. Version configs with semantic versioning in filename
4. Use regime routing for market-adaptive behavior
EOF

# data/README.md
cat > data/README.md << 'EOF'
# Data Directory

Feature stores and data assets.

## Directory Structure

- `raw/` - Raw exchange/API data (never modified)
  - `binance/` - Binance klines
  - `bybit/` - Bybit klines
  - `tradingview/` - TradingView exports
  - `yfinance/` - Yahoo Finance data
- `processed/` - Processed feature stores (versioned)
  - `features_mtf/` - Multi-timeframe features
  - `macro/` - Macro regime labels and features
  - `cache/` - Temporary computation cache
- `archive/` - Old feature store versions

## Feature Store Format

All feature stores are Parquet files with standardized schema:

```python
import pandas as pd

df = pd.read_parquet('data/processed/features_mtf/btc_mtf_features.parquet')

# Columns:
# - timestamp (index)
# - OHLCV (open, high, low, close, volume)
# - TF-prefixed features (tf1d_*, tf4h_*, tf1h_*)
# - Wyckoff scores
# - Liquidity scores
# - Fusion scores
# - Regime labels
```

## Data Versioning

Feature stores are versioned by date in filename:

- `btc_mtf_features_2024-11-14.parquet` - Latest
- `btc_mtf_features_2024-10-01.parquet` - Archived

## Building Feature Stores

```bash
# Build from scratch
python bin/build/build_mtf_feature_store.py --symbol BTC --start 2020-01-01 --end 2024-12-31

# Backfill missing features
python bin/backfill/backfill_liquidity_score_optimized.py
python bin/backfill/backfill_ob_high_optimized.py
```

## Best Practices

1. Never modify raw data
2. Version all processed feature stores
3. Validate NaN coverage before use
4. Archive old versions to `data/archive/`
EOF

# bin/README.md
cat > bin/README.md << 'EOF'
# Scripts Directory

Production scripts and CLI tools organized by purpose.

## Directory Structure

- `backtest/` - Backtest runners
- `build/` - Feature store builders
- `backfill/` - Data backfill scripts
- `download/` - Data download scripts
- `analyze/` - Analysis and reporting scripts
- `optimize/` - Optimization runners (Optuna)
- `diagnose/` - Diagnostic tools
- `cli/` - CLI entrypoints
- `archive/` - Deprecated scripts

## Common Usage

### Backtest

```bash
python bin/backtest/backtest_knowledge_v2.py \
    --config configs/mvp/mvp_bull_market_v1.json \
    --symbol BTC \
    --start 2022-01-01 \
    --end 2024-12-31
```

### Build Feature Store

```bash
python bin/build/build_mtf_feature_store.py \
    --symbol BTC \
    --start 2020-01-01 \
    --end 2024-12-31 \
    --output data/processed/features_mtf/btc_mtf_features.parquet
```

### Backfill Missing Features

```bash
python bin/backfill/backfill_liquidity_score_optimized.py
python bin/backfill/backfill_ob_high_optimized.py
```

### Analysis

```bash
python bin/analyze/analyze_archetype_perf.py --results results/backtest_2024.json
```

## Best Practices

1. Use `--help` flag to see all options
2. Save outputs to `results/` (gitignored)
3. Archive important results to `results_reference/`
4. Use absolute paths in scripts
EOF
```

### Phase 8: Validation (Day 2-3)

#### Step 8.1: Run Test Suite

```bash
# Run full test suite
pytest tests/ -v --tb=short

# Run specific test categories
pytest tests/unit/ -v
pytest tests/integration/ -v
pytest tests/parity/ -v
```

#### Step 8.2: Verify Imports

```bash
# Test import paths
python -c "from engine.archetypes import ArchetypeLogic; print(ArchetypeLogic.CLASS_VERSION)"
python -c "from engine.strategies.archetypes.bull import TrapReversalDetector; print(TrapReversalDetector.ARCHETYPE_NAME)"
python -c "from engine.strategies.archetypes.bear import FailedRallyDetector; print(FailedRallyDetector.ARCHETYPE_NAME)"
```

#### Step 8.3: Run Baseline Backtest Comparison

```bash
# Run same backtest as baseline
python bin/backtest/backtest_knowledge_v2.py \
    --config configs/mvp/mvp_bull_market_v1.json \
    --symbol BTC \
    --start 2022-01-01 \
    --end 2024-12-31 \
    > post_restructure_mvp_bull_btc.json

# Compare results
python tools/compare_backtest_results.py \
    results_reference/restructure_baseline/baseline_mvp_bull_btc.json \
    post_restructure_mvp_bull_btc.json
```

#### Step 8.4: Validate File Integrity

```bash
# Check no files were accidentally deleted
git status

# Verify all modules importable
python -c "import bull_machine"
python -c "import engine"
python -c "from engine.archetypes import ArchetypeLogic"
```

### Phase 9: Commit & Document (Day 3)

#### Step 9.1: Git Commit

```bash
# Stage all changes
git add -A

# Commit with detailed message
git commit -m "$(cat <<'EOF'
feat: restructure repository to quant gold standard

Major restructuring to improve code organization, separate concerns,
and establish clear boundaries between strategies and core engine.

Changes:

1. **Configs Organization**
   - Move frozen baselines to configs/frozen/
   - Move MVP configs to configs/mvp/
   - Move experiments to configs/experiments/{optuna,bear,adaptive}/
   - Move regime profiles to configs/regime/
   - Archive old versions to configs/archive/

2. **Bin Scripts Reorganization**
   - Organize 130+ scripts by purpose:
     - bin/backtest/ - Backtest runners
     - bin/build/ - Feature store builders
     - bin/backfill/ - Data backfill scripts
     - bin/download/ - Data download scripts
     - bin/analyze/ - Analysis scripts
     - bin/optimize/ - Optimization runners
     - bin/diagnose/ - Diagnostic tools
     - bin/cli/ - CLI entrypoints
   - Archive deprecated scripts to bin/archive/
   - Consolidate scripts/ into bin/

3. **Data Directory Restructure**
   - Separate raw/processed data:
     - data/raw/{binance,bybit,tradingview,yfinance}/
     - data/processed/{features_mtf,macro,cache}/
   - Archive old feature stores to data/archive/

4. **Archetype Logic Modularization**
   - Split monolithic logic_v2_adapter.py (1,440 lines) into:
     - engine/strategies/archetypes/base.py - Base detector class
     - engine/strategies/archetypes/bull/* - 11 bull archetypes
     - engine/strategies/archetypes/bear/* - 8 bear archetypes
   - Create backward compatibility facade in engine/archetypes/logic_adapter.py
   - Archive old file to engine/archetypes/_legacy.py

5. **Documentation**
   - Add docs/ARCHITECTURE.md - System architecture overview
   - Add README files for configs/, data/, bin/
   - Update CHANGELOG.md

6. **.gitignore Updates**
   - Add exclusions for new directory structure
   - Maintain results_reference/ tracking

**Backward Compatibility**: All existing imports maintained via facade layer.
No functional behavior changes.

**Validation**: All tests pass, baseline backtest results match.

Generated with Claude Code
Co-Authored-By: Claude <noreply@anthropic.com>
EOF
)"
```

#### Step 9.2: Update CHANGELOG.md

```bash
# Add restructure entry to CHANGELOG.md
cat >> CHANGELOG.md << 'EOF'

## [2.0.0] - 2025-11-14

### Major Restructure

**Repository restructured to quant gold standard** - See docs/reports/REPO_RESTRUCTURE_PLAN_2025-11-14.md for full details.

#### Changed

- **Configs**: Organized into frozen/, mvp/, experiments/, regime/, archive/
- **Bin**: Organized 130+ scripts by purpose (backtest/, build/, analyze/, etc.)
- **Data**: Separated raw/ and processed/ with clear versioning
- **Archetypes**: Split monolithic logic into modular bull/ and bear/ detectors
- **Docs**: Added ARCHITECTURE.md and README files

#### Added

- `engine/strategies/archetypes/` - Modular archetype detector architecture
- `docs/ARCHITECTURE.md` - System architecture documentation
- README files for configs/, data/, bin/

#### Deprecated

- `engine/archetypes/logic_v2_adapter.py` - Moved to `_legacy.py`, use facade instead
- `scripts/` directory - Consolidated into bin/

#### Migration

All existing imports maintained via backward compatibility facade. No code changes required.

EOF
```

---

## 6. Risk Assessment

### 6.1 High-Risk Operations

| Operation | Risk Level | Mitigation |
|-----------|-----------|------------|
| Split archetype logic | HIGH | Exact copy of logic, comprehensive tests, parity validation |
| Move 130+ bin scripts | MEDIUM | Use `git mv`, update paths incrementally, test imports |
| Reorganize configs | LOW | Configs are data files, easy to move, validate references |
| Data directory restructure | MEDIUM | Many symlinks to external dirs, preserve carefully |

### 6.2 What Could Break

#### A. Import Failures

**Risk**: Code importing from old paths fails

**Mitigation**:
- Keep facade layer in `engine/archetypes/` exporting all classes
- Maintain `logic_v2_adapter` as alias to `logic_adapter`
- Grep all imports before migration
- Test all import paths after migration

#### B. Script Path References

**Risk**: Scripts referencing old bin/ paths break

**Mitigation**:
- Update all hardcoded paths in scripts
- Use relative imports where possible
- Test each moved script individually

#### C. Config References

**Risk**: Backtest scripts reference old config paths

**Mitigation**:
- Keep symlinks for commonly-used configs
- Update documentation with new paths
- Search codebase for config references and update

#### D. Archetype Behavior Changes

**Risk**: Splitting archetype logic introduces subtle bugs

**Mitigation**:
- **Exact copy** of logic from `_check_X()` to `detect()`
- **Zero tolerance** for logic changes
- Parity tests comparing old vs new outputs
- Baseline backtest comparison

### 6.3 Rollback Triggers

Rollback to `main` if any of:
1. Tests fail that passed pre-restructure
2. Baseline backtest results differ by >0.1%
3. Critical imports fail
4. More than 10 import paths need manual fixes

### 6.4 Validation Gates

Before merging to `main`:

1. All tests pass (`pytest tests/ -v`)
2. Baseline backtest matches within 0.1%
3. All module imports work
4. No files accidentally deleted
5. Documentation complete

---

## 7. Validation Checklist

### 7.1 Pre-Migration Baseline

- [ ] Run full test suite, save results
- [ ] Run baseline backtests, save outputs to `results_reference/restructure_baseline/`
- [ ] Audit all imports of modules we'll move
- [ ] Git status clean (no uncommitted changes)

### 7.2 Post-Migration Validation

#### Tests
- [ ] `pytest tests/unit/ -v` passes
- [ ] `pytest tests/integration/ -v` passes
- [ ] `pytest tests/parity/ -v` passes (if exists)
- [ ] `pytest tests/smoke/ -v` passes

#### Imports
- [ ] `from engine.archetypes import ArchetypeLogic` works
- [ ] `from engine.strategies.archetypes.bull import TrapReversalDetector` works
- [ ] `from engine.strategies.archetypes.bear import FailedRallyDetector` works
- [ ] `from bull_machine.core import config_loader` works
- [ ] No `ModuleNotFoundError` in any test

#### Backtests
- [ ] MVP Bull BTC backtest (2022-2024) matches baseline within 0.1%
- [ ] MVP Bear BTC backtest (2022-2024) matches baseline within 0.1%
- [ ] Regime routing backtest matches baseline within 0.1%

#### File Integrity
- [ ] No files accidentally deleted (check `git status`)
- [ ] All configs accessible at new paths
- [ ] All scripts accessible at new paths
- [ ] Data files intact (symlinks preserved)

#### Documentation
- [ ] `docs/ARCHITECTURE.md` created and accurate
- [ ] README files created for configs/, data/, bin/
- [ ] CHANGELOG.md updated
- [ ] Migration log in `docs/reports/RESTRUCTURE_MIGRATION_LOG.md`

### 7.3 Smoke Tests

```bash
# Test key workflows
python bin/backtest/backtest_knowledge_v2.py --config configs/mvp/mvp_bull_market_v1.json --symbol BTC --start 2024-01-01 --end 2024-01-31
python bin/build/build_mtf_feature_store.py --symbol BTC --start 2024-01-01 --end 2024-01-31 --output /tmp/test_features.parquet
python bin/analyze/analyze_archetype_perf.py --help
```

### 7.4 Code Quality

- [ ] Ruff linting passes (`ruff check .`)
- [ ] Mypy type checking passes (`mypy bull_machine/ engine/`)
- [ ] No TODO/FIXME comments introduced
- [ ] All new modules have docstrings

---

## 8. Rollback Plan

### 8.1 Immediate Rollback (Within 1 Hour)

**If validation fails:**

```bash
# Abort restructure, return to main
git checkout main
git branch -D repo-restructure-2025-11-14
```

### 8.2 Post-Merge Rollback (After PR Merged)

**If issues discovered after merge:**

```bash
# Create revert branch
git checkout main
git checkout -b revert-restructure

# Revert the restructure commit
git revert <restructure-commit-sha>

# Push and create PR
git push origin revert-restructure
```

### 8.3 Partial Rollback

**If only specific changes problematic:**

```bash
# Cherry-pick only working changes
git checkout main
git checkout -b partial-restructure

# Selectively apply changes
git checkout repo-restructure-2025-11-14 -- configs/
git checkout repo-restructure-2025-11-14 -- bin/
# (skip archetype split)

git commit -m "partial: apply config and bin reorganization only"
```

### 8.4 Recovery Procedures

#### Lost Files

```bash
# Find deleted files
git log --diff-filter=D --summary

# Restore specific file
git checkout <commit-before-delete> -- path/to/file
```

#### Import Failures

```bash
# Find all broken imports
grep -r "ImportError" logs/

# Add temporary shim in __init__.py
cat >> engine/archetypes/__init__.py << 'EOF'
# Temporary shim for missing import
from engine.archetypes._legacy import ArchetypeLogic as LegacyArchetypeLogic
EOF
```

---

## Appendices

### Appendix A: File Count Summary

| Category | Current | After Restructure | Change |
|----------|---------|-------------------|--------|
| Configs | 52 flat | 52 organized in subdirs | +5 subdirs |
| Bin scripts | 130 flat | 130 organized in 8 subdirs | +8 subdirs |
| Data files | 58 mixed | 58 organized (raw/processed) | +3 subdirs |
| Archetype files | 1 monolith | 21 modular (base + 11 bull + 8 bear + facade) | +20 files |
| Documentation | Good | Enhanced with ARCHITECTURE.md + READMEs | +4 files |

### Appendix B: Import Path Changes

| Old Import | New Import | Status |
|------------|-----------|--------|
| `from engine.archetypes.logic_v2_adapter import ArchetypeLogic` | `from engine.archetypes import ArchetypeLogic` | Supported (facade) |
| N/A | `from engine.strategies.archetypes.bull import TrapReversalDetector` | New (direct access) |
| N/A | `from engine.strategies.archetypes.bear import FailedRallyDetector` | New (direct access) |

**All old imports still work via facade.**

### Appendix C: Config Path Changes

| Old Path | New Path |
|----------|----------|
| `configs/baseline_btc_bull_pf20.json` | `configs/frozen/baseline_btc_bull_pf20.json` |
| `configs/mvp_bull_market_v1.json` | `configs/mvp/mvp_bull_market_v1.json` |
| `configs/bear_archetypes_phase1.json` | `configs/experiments/bear/bear_archetypes_phase1.json` |
| `configs/btc_v7_ml_enabled.json` | `configs/archive/btc_v7_ml_enabled.json` |
| `configs/profile_btc_seed.json` | `configs/regime/profile_btc_seed.json` |

### Appendix D: Script Path Changes

| Old Path | New Path |
|----------|----------|
| `bin/backtest_knowledge_v2.py` | `bin/backtest/backtest_knowledge_v2.py` |
| `bin/build_mtf_feature_store.py` | `bin/build/build_mtf_feature_store.py` |
| `bin/backfill_liquidity_score_optimized.py` | `bin/backfill/backfill_liquidity_score_optimized.py` |
| `bin/analyze_archetype_perf.py` | `bin/analyze/analyze_archetype_perf.py` |
| `scripts/backtests/fast_monthly_test.py` | `bin/backtest/fast_monthly_test.py` |

### Appendix E: Archetype Mapping

| Archetype | Letter | Old Method | New Class | New File |
|-----------|--------|-----------|-----------|----------|
| Trap Reversal | A | `_check_A()` | `TrapReversalDetector` | `bull/trap_reversal.py` |
| Order Block Retest | B | `_check_B()` | `OrderBlockRetestDetector` | `bull/order_block_retest.py` |
| FVG Continuation | C | `_check_C()` | `FVGContinuationDetector` | `bull/fvg_continuation.py` |
| Wick Trap | K | `_check_K()` | `WickTrapDetector` | `bull/wick_trap.py` |
| Trap Within Trend | H | `_check_H()` | `TrapWithinTrendDetector` | `bull/trap_within_trend.py` |
| Volume Exhaustion | L | `_check_L()` | `VolumeExhaustionDetector` | `bull/volume_exhaustion.py` |
| Expansion Exhaustion | F | `_check_F()` | `ExpansionExhaustionDetector` | `bull/expansion_exhaustion.py` |
| Failed Continuation | D | `_check_D()` | `FailedContinuationDetector` | `bull/failed_continuation.py` |
| Re-Accumulate | G | `_check_G()` | `ReAccumulateDetector` | `bull/re_accumulate.py` |
| Liquidity Compression | E | `_check_E()` | `LiquidityCompressionDetector` | `bull/liquidity_compression.py` |
| Ratio Coil Break | M | `_check_M()` | `RatioCoilBreakDetector` | `bull/ratio_coil_break.py` |
| Breakdown | S1 | `_check_S1()` | `BreakdownDetector` | `bear/breakdown.py` |
| Failed Rally | S2 | `_check_S2()` | `FailedRallyDetector` | `bear/failed_rally.py` |
| Whipsaw | S3 | `_check_S3()` | `WhipsawDetector` | `bear/whipsaw.py` |
| Distribution | S4 | `_check_S4()` | `DistributionDetector` | `bear/distribution.py` |
| Long Squeeze | S5 | `_check_S5()` | `LongSqueezeDetector` | `bear/long_squeeze.py` |
| Alt Rotation Down | S6 | `_check_S6()` | `AltRotationDownDetector` | `bear/alt_rotation_down.py` |
| Curve Inversion | S7 | `_check_S7()` | `CurveInversionDetector` | `bear/curve_inversion.py` |
| Volume Fade Chop | S8 | `_check_S8()` | `VolumeFadeChopDetector` | `bear/volume_fade_chop.py` |

---

## Next Steps

After this plan is approved:

1. **Review**: Technical review by team
2. **Approval**: Sign-off from senior quant + senior SWE
3. **Execution**: Follow migration steps exactly as documented
4. **Validation**: Complete all validation checklist items
5. **Merge**: Create PR, review, merge to main
6. **Documentation**: Update team wiki, notify team

---

**Status**: PLAN ONLY - NOT YET EXECUTED

**Review Required**: Yes
**Approval Required**: Yes
**Estimated Timeline**: 2-3 days

---

*End of Restructuring Plan*
