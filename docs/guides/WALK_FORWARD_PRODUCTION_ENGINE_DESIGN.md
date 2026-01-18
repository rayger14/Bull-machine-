# Walk-Forward Validation with Production Backtest Engine

**Design Document**
**Date**: 2026-01-16
**Author**: Claude Code - Performance Engineer
**Status**: Ready for Implementation

---

## Executive Summary

This document specifies a **TRUE walk-forward validation system** using the **REAL production backtest engine** (`FullEngineBacktest`) to validate the 6 recently optimized archetype configurations.

**Key Principle**: NO SIMPLIFIED LOGIC. Use the actual production execution flow with all systems enabled.

---

## Current State Analysis

### ✅ What We Have

1. **Production Backtest Engine**: `/Users/raymondghandchi/Bull-machine-/Bull-machine-/bin/backtest_full_engine_replay.py`
   - Class: `FullEngineBacktest`
   - Features:
     - Next-bar execution (signal on bar T → entry on bar T+1)
     - Real archetype implementations via `ArchetypeFactory`
     - Regime detection (`RegimeService` with logistic model + hysteresis)
     - Circuit breakers (`CircuitBreakerEngine`)
     - Direction balance tracking (`DirectionBalanceTracker`)
     - Transaction costs (0.06% fees + 0.08% slippage)
     - Position management (stops/targets)
     - Portfolio constraints (max positions, cooldowns)

2. **Real Archetype Implementations**:
   - Bull: `engine/strategies/archetypes/bull/*.py` (B, H, K)
   - Bear: `engine/strategies/archetypes/bear/*_runtime.py` (S1, S4, S5)
   - Factory: `engine/archetypes/archetype_factory.py` (loads from `archetype_registry.yaml`)

3. **Optimized Configurations**: `results/optimization_2026-01-16/*/best_config.json`
   - S1 (Liquidity Vacuum): Sharpe 1.78, Sortino 1.64
   - S4 (Funding Divergence): Configs ready
   - S5 (Long Squeeze): Configs ready
   - B (Order Block Retest): Configs ready
   - H (Trap Within Trend): Configs ready
   - K (Wick Trap Moneytaur): Configs ready

4. **Feature Data**: `data/features_mtf/BTC_1H_2022-01-01_to_2024-12-31.parquet`
   - 3 years hourly OHLCV + features + regime labels

### ❌ What's Broken

**Current walk-forward script** (`bin/walk_forward_validation.py`):
- ❌ Uses **simplified archetype logic** (lines 414-455)
- ❌ Uses **simplified execution** (no circuit breakers, no direction balance)
- ❌ Uses **placeholder signal generation** (not real archetype implementations)
- ❌ Missing portfolio constraints
- ❌ Missing regime systems

**This is NOT acceptable for production validation.**

---

## Architecture Design

### Walk-Forward Framework Integration

```
┌─────────────────────────────────────────────────────────────────┐
│                  Walk-Forward Orchestrator                       │
│                                                                   │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐             │
│  │  Window 1   │  │  Window 2   │  │  Window N   │             │
│  │ Train: 365d │  │ Train: 365d │  │ Train: 365d │             │
│  │ Embargo:48h │  │ Embargo:48h │  │ Embargo:48h │             │
│  │ Test: 90d   │  │ Test: 90d   │  │ Test: 90d   │             │
│  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘             │
│         │                 │                 │                     │
│         └─────────────────┴─────────────────┘                     │
│                           │                                       │
│                           ▼                                       │
│         ┌─────────────────────────────────────┐                  │
│         │   FullEngineBacktest (PRODUCTION)   │                  │
│         │                                     │                  │
│         │  ┌────────────────────────────────┐│                  │
│         │  │   ArchetypeFactory             ││                  │
│         │  │   - Real implementations       ││                  │
│         │  │   - Load from registry         ││                  │
│         │  └────────────────────────────────┘│                  │
│         │  ┌────────────────────────────────┐│                  │
│         │  │   RegimeService                ││                  │
│         │  │   - Logistic model             ││                  │
│         │  │   - Hysteresis                 ││                  │
│         │  └────────────────────────────────┘│                  │
│         │  ┌────────────────────────────────┐│                  │
│         │  │   CircuitBreakerEngine         ││                  │
│         │  └────────────────────────────────┘│                  │
│         │  ┌────────────────────────────────┐│                  │
│         │  │   DirectionBalanceTracker      ││                  │
│         │  └────────────────────────────────┘│                  │
│         │  ┌────────────────────────────────┐│                  │
│         │  │   TransactionCostModel         ││                  │
│         │  │   - Fees: 0.06%                ││                  │
│         │  │   - Slippage: 0.08%            ││                  │
│         │  └────────────────────────────────┘│                  │
│         └─────────────────────────────────────┘                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## Implementation Plan

### Step 1: Adapt FullEngineBacktest for Walk-Forward

**Current Signature**:
```python
backtest = FullEngineBacktest(config)
results = backtest.run(data, start_date, end_date, archetypes_to_test)
```

**Needed Changes**:
1. **Parameter injection**: Accept optimized params per archetype
2. **Archetype filtering**: Test specific archetypes (not all 16)
3. **Config override**: Allow runtime parameter updates

**Proposed Enhancement**:
```python
class FullEngineBacktest:
    def __init__(self, config: Dict):
        # Existing initialization...

        # NEW: Allow archetype-specific param overrides
        self.archetype_param_overrides = config.get('archetype_param_overrides', {})

    def _get_archetype_config(self, archetype_id: str) -> Dict:
        """Get archetype config with walk-forward overrides"""
        base_config = self.archetype_factory._get_archetype_config(archetype_id)

        # Apply walk-forward overrides if present
        if archetype_id in self.archetype_param_overrides:
            overrides = self.archetype_param_overrides[archetype_id]
            base_config['thresholds'].update(overrides)

        return base_config
```

### Step 2: Create Walk-Forward Orchestrator

**New File**: `bin/walk_forward_production_engine.py`

```python
#!/usr/bin/env python3
"""
Walk-Forward Validation with Production Backtest Engine
========================================================

CRITICAL: Uses REAL FullEngineBacktest - NO simplified logic!

Architecture:
- Window generation: 365d train / 48h embargo / 90d test / 90d step
- Execution: FullEngineBacktest with real archetypes
- Parameters: Optimized configs from results/optimization_2026-01-16/
- Systems: All production systems enabled (regime, circuit breakers, etc.)
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
import json
import logging
from typing import Dict, List, Tuple
from dataclasses import dataclass

# Import PRODUCTION backtest engine
from backtest_full_engine_replay import FullEngineBacktest


@dataclass
class WalkForwardWindow:
    """Walk-forward window definition"""
    id: int
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    embargo_start: pd.Timestamp
    embargo_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp


class ProductionWalkForwardValidator:
    """
    Walk-forward validation using PRODUCTION backtest engine.

    NO SIMPLIFIED LOGIC - uses FullEngineBacktest with real archetypes.
    """

    def __init__(
        self,
        train_days: int = 365,
        embargo_hours: int = 48,
        test_days: int = 90,
        step_days: int = 90,
        initial_capital: float = 10000.0
    ):
        self.train_days = train_days
        self.embargo_hours = embargo_hours
        self.test_days = test_days
        self.step_days = step_days
        self.initial_capital = initial_capital

    def generate_windows(
        self,
        data: pd.DataFrame
    ) -> List[WalkForwardWindow]:
        """Generate non-overlapping walk-forward windows"""
        windows = []
        window_id = 0

        current_start = data.index[0]
        data_end = data.index[-1]

        while True:
            # Calculate boundaries
            train_start = current_start
            train_end = train_start + pd.Timedelta(days=self.train_days)
            embargo_start = train_end
            embargo_end = embargo_start + pd.Timedelta(hours=self.embargo_hours)
            test_start = embargo_end
            test_end = test_start + pd.Timedelta(days=self.test_days)

            # Check if we've run out of data
            if test_end > data_end:
                break

            window_id += 1
            windows.append(WalkForwardWindow(
                id=window_id,
                train_start=train_start,
                train_end=train_end,
                embargo_start=embargo_start,
                embargo_end=embargo_end,
                test_start=test_start,
                test_end=test_end
            ))

            # Step forward (non-overlapping test windows)
            current_start += pd.Timedelta(days=self.step_days)

        return windows

    def run_window(
        self,
        window: WalkForwardWindow,
        data: pd.DataFrame,
        archetype_id: str,
        optimized_params: Dict,
        config: Dict
    ) -> Dict:
        """
        Run backtest for single window using PRODUCTION engine.

        Args:
            window: Walk-forward window definition
            data: Full dataset
            archetype_id: Archetype to test (e.g., 'S1')
            optimized_params: Optimized parameters from training
            config: Base backtest configuration

        Returns:
            Results dictionary from FullEngineBacktest
        """
        # Extract test data (OOS period only)
        test_data = data[
            (data.index >= window.test_start) &
            (data.index < window.test_end)
        ].copy()

        # Build config with archetype-specific overrides
        backtest_config = config.copy()
        backtest_config['archetype_param_overrides'] = {
            archetype_id: optimized_params
        }

        # Initialize PRODUCTION backtest engine
        backtest = FullEngineBacktest(backtest_config)

        # Run backtest on OOS test window
        results = backtest.run(
            data=test_data,
            archetypes_to_test=[archetype_id]
        )

        # Add window metadata
        results['window_id'] = window.id
        results['train_period'] = (
            window.train_start.strftime('%Y-%m-%d'),
            window.train_end.strftime('%Y-%m-%d')
        )
        results['test_period'] = (
            window.test_start.strftime('%Y-%m-%d'),
            window.test_end.strftime('%Y-%m-%d')
        )

        return results

    def validate_archetype(
        self,
        archetype_id: str,
        optimized_config_path: str,
        data: pd.DataFrame
    ) -> Dict:
        """
        Run complete walk-forward validation for archetype.

        Args:
            archetype_id: Archetype identifier (S1, S4, S5, B, H, K)
            optimized_config_path: Path to best_config.json from optimization
            data: Full historical dataset

        Returns:
            Aggregated walk-forward results
        """
        # Load optimized parameters
        with open(optimized_config_path, 'r') as f:
            optimized_config = json.load(f)

        optimized_params = optimized_config['best_params']

        # Base backtest configuration (PRODUCTION settings)
        base_config = {
            'symbol': 'BTC',
            'initial_capital': self.initial_capital,
            'max_positions': 5,
            'position_size_pct': 0.12,  # 12% per position
            'fee_pct': 0.0006,  # 0.06% Binance taker
            'slippage_pct': 0.0008,  # 0.08%
            'cooldown_bars': 12,  # 12 hours
            # ENABLE ALL PRODUCTION SYSTEMS
            'enable_circuit_breakers': True,
            'enable_direction_balance': True,
            'enable_regime_penalties': True,
            'enable_adaptive_regime': True,
            # Archetype selection
            f'enable_{archetype_id}': True  # Enable only this archetype
        }

        # Generate windows
        windows = self.generate_windows(data)

        # Run backtest on each window
        window_results = []
        for window in windows:
            result = self.run_window(
                window=window,
                data=data,
                archetype_id=archetype_id,
                optimized_params=optimized_params,
                config=base_config
            )
            window_results.append(result)

        # Aggregate results
        aggregated = self._aggregate_results(
            archetype_id=archetype_id,
            window_results=window_results,
            optimized_config=optimized_config
        )

        return aggregated

    def _aggregate_results(
        self,
        archetype_id: str,
        window_results: List[Dict],
        optimized_config: Dict
    ) -> Dict:
        """Aggregate metrics across all windows"""

        # Extract metrics from each window
        total_trades = sum(r['total_trades'] for r in window_results)
        total_pnl = sum(r['total_pnl'] for r in window_results)

        # Calculate aggregate statistics
        sharpes = [r['sharpe_ratio'] for r in window_results if r['total_trades'] > 0]
        sortinos = [r['sortino_ratio'] for r in window_results if r['total_trades'] > 0]
        win_rates = [r['win_rate'] for r in window_results if r['total_trades'] > 0]

        profitable_windows = sum(1 for r in window_results if r['total_pnl'] > 0)

        # Compute OOS degradation
        in_sample_sharpe = optimized_config['metrics']['sharpe']
        oos_sharpe = np.mean(sharpes) if sharpes else 0.0
        degradation_pct = ((in_sample_sharpe - oos_sharpe) / in_sample_sharpe * 100) if in_sample_sharpe > 0 else 0.0

        return {
            'archetype': archetype_id,
            'config_path': optimized_config_path,
            'total_windows': len(window_results),
            'aggregate_metrics': {
                'total_trades': total_trades,
                'total_pnl': total_pnl,
                'avg_sharpe': oos_sharpe,
                'avg_sortino': np.mean(sortinos) if sortinos else 0.0,
                'avg_win_rate': np.mean(win_rates) if win_rates else 0.0,
                'profitable_windows': profitable_windows,
                'profitable_pct': (profitable_windows / len(window_results) * 100) if window_results else 0.0
            },
            'oos_analysis': {
                'in_sample_sharpe': in_sample_sharpe,
                'oos_sharpe': oos_sharpe,
                'degradation_pct': degradation_pct,
                'robust': degradation_pct < 20.0  # <20% degradation = robust
            },
            'window_details': window_results
        }


def main():
    """Execute walk-forward validation"""
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    logger.info("="*80)
    logger.info("WALK-FORWARD VALIDATION - PRODUCTION ENGINE")
    logger.info("="*80)

    # Load data
    data_path = "data/features_mtf/BTC_1H_2022-01-01_to_2024-12-31.parquet"
    logger.info(f"Loading {data_path}...")
    data = pd.read_parquet(data_path)

    # Initialize validator
    validator = ProductionWalkForwardValidator(
        train_days=365,
        embargo_hours=48,
        test_days=90,
        step_days=90,
        initial_capital=10000.0
    )

    # Test all optimized archetypes
    archetypes = ['S1', 'S4', 'S5', 'B', 'H', 'K']

    all_results = {}

    for archetype in archetypes:
        config_path = f"results/optimization_2026-01-16/{archetype}/best_config.json"

        if not Path(config_path).exists():
            logger.warning(f"Skipping {archetype} - config not found")
            continue

        logger.info(f"\n{'='*60}")
        logger.info(f"VALIDATING {archetype}")
        logger.info(f"{'='*60}")

        results = validator.validate_archetype(
            archetype_id=archetype,
            optimized_config_path=config_path,
            data=data
        )

        all_results[archetype] = results

        # Log summary
        logger.info(f"\nRESULTS: {archetype}")
        logger.info(f"  Windows: {results['total_windows']}")
        logger.info(f"  Total Trades: {results['aggregate_metrics']['total_trades']}")
        logger.info(f"  Total PnL: ${results['aggregate_metrics']['total_pnl']:.2f}")
        logger.info(f"  OOS Sharpe: {results['aggregate_metrics']['avg_sharpe']:.3f}")
        logger.info(f"  OOS Degradation: {results['oos_analysis']['degradation_pct']:.1f}%")
        logger.info(f"  Robust: {'✅' if results['oos_analysis']['robust'] else '❌'}")

    # Save results
    output_dir = Path('results/walk_forward_production')
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(output_dir / 'all_results.json', 'w') as f:
        json.dump(all_results, f, indent=2, default=str)

    logger.info(f"\n✅ Results saved: {output_dir}")


if __name__ == '__main__':
    main()
```

---

## Configuration Requirements

### Archetype Parameter Mapping

The walk-forward system needs to map optimized parameters to archetype implementations:

**S1 (Liquidity Vacuum)**:
```json
{
  "fusion_threshold": 0.35,
  "liquidity_weight": 0.32,
  "volume_weight": 0.23,
  "wick_weight": 0.18
}
```

Maps to: `engine/strategies/archetypes/bear/liquidity_vacuum_runtime.py`

**S4, S5, B, H, K**: Similar mapping from `best_config.json` to implementation.

### FullEngineBacktest Configuration

```python
{
    'symbol': 'BTC',
    'initial_capital': 10000.0,
    'max_positions': 5,
    'position_size_pct': 0.12,  # 12% per position
    'fee_pct': 0.0006,  # 0.06% Binance taker
    'slippage_pct': 0.0008,  # 0.08%
    'cooldown_bars': 12,  # 12 hours

    # PRODUCTION SYSTEMS - ALL ENABLED
    'enable_circuit_breakers': True,
    'enable_direction_balance': True,
    'enable_regime_penalties': True,
    'enable_adaptive_regime': True,

    # ARCHETYPE-SPECIFIC OVERRIDES (injected per window)
    'archetype_param_overrides': {
        'S1': {
            'fusion_threshold': 0.35,
            'liquidity_weight': 0.32,
            # ... from best_config.json
        }
    }
}
```

---

## Validation Criteria

### Success Metrics

**Per Window**:
- ✅ Trades > 5 (sufficient sample)
- ✅ Sharpe > 0.5 (positive risk-adjusted return)
- ✅ Max DD < 50% (no catastrophic failure)

**Aggregate (All Windows)**:
- ✅ OOS Degradation < 20% (robust generalization)
- ✅ Profitable Windows > 60% (consistency)
- ✅ Aggregate Sharpe > 0.5 (profitable overall)
- ✅ Zero catastrophic failures (no >50% DD windows)

### Production Readiness Assessment

```python
def assess_production_readiness(results: Dict) -> bool:
    """
    Determine if archetype is production-ready based on walk-forward results.

    Returns True if ALL criteria met:
    - OOS degradation <20%
    - >60% windows profitable
    - Aggregate Sharpe >0.5
    - No catastrophic failures
    """
    oos_analysis = results['oos_analysis']
    aggregate = results['aggregate_metrics']

    checks = [
        oos_analysis['degradation_pct'] < 20.0,
        aggregate['profitable_pct'] > 60.0,
        aggregate['avg_sharpe'] > 0.5,
        # No windows with >50% DD
        all(w['max_drawdown_pct'] < 50 for w in results['window_details'])
    ]

    return all(checks)
```

---

## Expected Outputs

### 1. Per-Archetype Results

`results/walk_forward_production/S1_results.json`:
```json
{
  "archetype": "S1",
  "config_path": "results/optimization_2026-01-16/S1/best_config.json",
  "total_windows": 8,
  "aggregate_metrics": {
    "total_trades": 73,
    "total_pnl": 1247.32,
    "avg_sharpe": 1.42,
    "avg_sortino": 1.56,
    "avg_win_rate": 52.3,
    "profitable_windows": 6,
    "profitable_pct": 75.0
  },
  "oos_analysis": {
    "in_sample_sharpe": 1.78,
    "oos_sharpe": 1.42,
    "degradation_pct": 20.2,
    "robust": false
  },
  "production_ready": false,
  "notes": [
    "❌ OOS degradation 20.2% (>20% threshold)",
    "✅ 75% windows profitable (>60% threshold)",
    "✅ Aggregate Sharpe 1.42 (>0.5)",
    "✅ No catastrophic failures"
  ],
  "window_details": [...]
}
```

### 2. Comparison Report

`results/walk_forward_production/comparison_report.md`:
```markdown
# Walk-Forward Validation Results - Production Engine

## Summary

| Archetype | Windows | Total Trades | Total PnL | OOS Sharpe | Degradation | Production Ready |
|-----------|---------|--------------|-----------|------------|-------------|------------------|
| S1        | 8       | 73           | $1,247    | 1.42       | 20.2%       | ❌               |
| S4        | 8       | 45           | $892      | 1.18       | 15.3%       | ✅               |
| S5        | 8       | 58           | $1,105    | 1.35       | 12.7%       | ✅               |
| B         | 8       | 82           | $1,420    | 1.51       | 18.9%       | ✅               |
| H         | 8       | 67           | $1,234    | 1.44       | 16.2%       | ✅               |
| K         | 8       | 91           | $1,678    | 1.62       | 14.5%       | ✅               |

## Key Findings

- **5/6 archetypes** passed production readiness criteria
- **Average OOS degradation**: 16.3% (within acceptable range)
- **Total OOS trades**: 416
- **Total OOS PnL**: $7,576
```

---

## Implementation Steps

### Phase 1: Minimal Viable Implementation (Day 1)

1. ✅ Create `bin/walk_forward_production_engine.py` with `ProductionWalkForwardValidator`
2. ✅ Implement window generation
3. ✅ Integrate with `FullEngineBacktest`
4. ✅ Test on S1 archetype (1 archetype, all windows)

**Acceptance**: Successfully runs walk-forward on S1 with real engine.

### Phase 2: Full Validation Suite (Day 2)

1. ✅ Add all 6 archetypes (S1, S4, S5, B, H, K)
2. ✅ Generate comparison reports
3. ✅ Implement production readiness assessment
4. ✅ Create visualization scripts

**Acceptance**: All 6 archetypes validated, production readiness assessed.

### Phase 3: Reporting & Analysis (Day 3)

1. ✅ Generate detailed window-by-window analysis
2. ✅ Regime-stratified performance
3. ✅ Degradation analysis by market condition
4. ✅ Final recommendation report

**Acceptance**: Complete documentation with actionable recommendations.

---

## Risk Mitigation

### Known Risks

1. **Data Leakage**:
   - Risk: Embargo too short, temporal features leak
   - Mitigation: 48-hour embargo (validated in literature)

2. **Insufficient Windows**:
   - Risk: <8 windows, statistical noise dominates
   - Mitigation: 2022-2024 data should yield ~8-10 windows

3. **Regime Bias**:
   - Risk: All test windows in same regime
   - Mitigation: Track regime distribution per window

4. **Overfitting to Training**:
   - Risk: Optimized params too specific to train period
   - Mitigation: Monitor OOS degradation (alert if >20%)

---

## Success Criteria

### Minimum Viable Success

- ✅ At least **3/6 archetypes** production-ready
- ✅ Average OOS degradation **<25%**
- ✅ Zero catastrophic failures
- ✅ System runs end-to-end without errors

### Ideal Success

- ✅ **5/6 archetypes** production-ready
- ✅ Average OOS degradation **<20%**
- ✅ >70% profitable windows across all archetypes
- ✅ Aggregate Sharpe >1.0

---

## Next Steps

1. **Implement** `bin/walk_forward_production_engine.py` (follow specification above)
2. **Run validation** on all 6 archetypes
3. **Analyze results** and identify production-ready configs
4. **Generate report** with recommendations
5. **Deploy** validated configs to live system (if passing criteria)

---

## References

- Production Backtest Engine: `bin/backtest_full_engine_replay.py`
- Archetype Factory: `engine/archetypes/archetype_factory.py`
- Optimized Configs: `results/optimization_2026-01-16/*/best_config.json`
- Legacy Walk-Forward (DO NOT USE): `bin/walk_forward_validation.py`

---

**Status**: Ready for implementation
**Estimated Time**: 2-3 days
**Dependencies**: None (all systems in place)
**Blocker Risk**: Low
