#!/usr/bin/env python3
"""
Validate trap_within_trend Optimization Results

Compares baseline vs optimized trap_within_trend parameters to validate
whether the optimization actually improved the problematic 2022-2023 performance.

Baseline problem (from diagnostic backtest):
- archetype_trap_within_trend: 104 trades, -$352.95 PnL, 46.15% WR
- 2022 performance: 25% WR, catastrophic losses

This script runs both configs and compares:
- Overall PnL, WR, PF
- 2022 specific performance (the problematic period)
- Trap_within_trend archetype-specific metrics
- Trade-level comparison

Usage:
    python3 bin/validate_trap_optimization.py \
      --cache data/cached/btc_features_2022-01-01_2024-12-31_cached.parquet \
      --output results/trap_validation
"""

import sys
import json
import argparse
from pathlib import Path
from datetime import datetime
import pandas as pd
import numpy as np
from typing import Dict, Any

sys.path.insert(0, str(Path(__file__).parent.parent))

from bin.backtest_knowledge_v2 import KnowledgeAwareBacktest, KnowledgeParams


class TrapValidationRunner:
    """
    Validation runner for trap_within_trend optimization.

    Runs baseline and optimized configs side-by-side to validate improvement.
    """

    def __init__(self, cache_path: str, output_dir: str):
        self.cache_path = Path(cache_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Load cached features
        print("📂 Loading cached features...")
        self.df_full = pd.read_parquet(self.cache_path)
        print(f"  ✓ Loaded {len(self.df_full)} bars")

        # Define validation windows
        self.windows = {
            '2022_H1': ('2022-01-01', '2022-06-30'),
            '2022_H2': ('2022-07-01', '2022-12-31'),
            '2023': ('2023-01-01', '2023-12-31'),
            '2024': ('2024-01-01', '2024-12-31'),
            'full': ('2022-01-01', '2024-12-31')
        }

        self.capital = 10000.0

        # Load baseline config
        self.baseline_config = self._load_baseline_config()

        # Load optimized params
        self.optimized_params = self._load_optimized_params()

        print(f"  ✓ Output: {self.output_dir}")

    def _load_baseline_config(self) -> Dict[str, Any]:
        """Load baseline config with original trap_within_trend parameters."""
        baseline_path = Path('configs/baseline_btc_bull_pf20.json')
        with open(baseline_path) as f:
            config = json.load(f)

        # Set baseline trap parameters (original values)
        if 'archetypes' not in config:
            config['archetypes'] = {}

        config['archetypes']['trap_within_trend'] = {
            'quality_threshold': 0.55,
            'liquidity_threshold': 0.30,
            'adx_threshold': 25.0,
            'fusion_threshold': 0.35,
            'wick_multiplier': 2.0
        }

        # Fixed sizing for fair comparison
        config['position_sizing'] = {
            'mode': 'fixed_fractional',
            'base_risk_per_trade_pct': 0.8,
            'max_risk_per_trade_pct': 0.8,
            'kelly_fraction': 1.0,
            'confidence_scaling': False,
            'archetype_quality_weight': 0.0
        }

        return config

    def _load_optimized_params(self) -> Dict[str, Any]:
        """Load optimized parameters from Optuna results."""
        optuna_path = Path('results/optuna_trap_200trial_PRODUCTION/best_params.json')

        if not optuna_path.exists():
            raise FileNotFoundError(f"Optimized params not found: {optuna_path}")

        with open(optuna_path) as f:
            data = json.load(f)

        best_params = data['best_params']

        # Map from Optuna param names to config param names
        return {
            'quality_threshold': best_params['trap_quality_threshold'],
            'liquidity_threshold': best_params['trap_liquidity_threshold'],
            'adx_threshold': best_params['trap_adx_threshold'],
            'fusion_threshold': best_params['trap_fusion_threshold'],
            'wick_multiplier': best_params['trap_wick_multiplier']
        }

    def _create_optimized_config(self) -> Dict[str, Any]:
        """Create config with optimized trap_within_trend parameters."""
        config = json.loads(json.dumps(self.baseline_config))  # Deep copy

        # Replace with optimized params
        config['archetypes']['trap_within_trend'] = self.optimized_params.copy()

        return config

    def _run_backtest(self, config: Dict, df: pd.DataFrame, name: str) -> Dict[str, Any]:
        """Run backtest with given config."""
        print(f"\n🔬 Running {name} backtest...")

        backtest = KnowledgeAwareBacktest(
            df=df,
            params=KnowledgeParams(),
            starting_capital=self.capital,
            asset='BTC',
            runtime_config=config
        )

        results = backtest.run()

        # Extract key metrics
        total_pnl = results.get('total_pnl', 0.0)
        total_trades = results.get('total_trades', 0)
        win_rate = results.get('win_rate', 0.0)
        profit_factor = results.get('profit_factor', 0.0)
        max_dd = results.get('max_drawdown_pct', 0.0)

        trades_df = results.get('trades_df', pd.DataFrame())

        # Extract trap_within_trend specific trades
        trap_trades = trades_df[trades_df['archetype'] == 'trap_within_trend'].copy() if len(trades_df) > 0 else pd.DataFrame()

        trap_metrics = {
            'trades': len(trap_trades),
            'total_pnl': trap_trades['net_pnl'].sum() if len(trap_trades) > 0 else 0.0,
            'avg_pnl': trap_trades['net_pnl'].mean() if len(trap_trades) > 0 else 0.0,
            'win_rate': (trap_trades['net_pnl'] > 0).mean() if len(trap_trades) > 0 else 0.0
        }

        print(f"  ✓ {name}: {total_trades} trades, ${total_pnl:.2f} PnL, {win_rate:.1%} WR")
        print(f"    Trap archetype: {trap_metrics['trades']} trades, ${trap_metrics['total_pnl']:.2f} PnL, {trap_metrics['win_rate']:.1%} WR")

        return {
            'total_pnl': total_pnl,
            'total_trades': total_trades,
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'max_drawdown_pct': max_dd,
            'trap_metrics': trap_metrics,
            'trades_df': trades_df,
            'trap_trades_df': trap_trades
        }

    def run_validation(self):
        """Execute validation comparison."""
        print("\n" + "="*80)
        print("🔬 TRAP_WITHIN_TREND OPTIMIZATION VALIDATION")
        print("="*80)
        print(f"\nBaseline Parameters:")
        for k, v in self.baseline_config['archetypes']['trap_within_trend'].items():
            print(f"  • {k}: {v}")

        print(f"\nOptimized Parameters:")
        for k, v in self.optimized_params.items():
            print(f"  • {k}: {v}")

        # Run on all windows
        results_comparison = {}

        for window_name, (start, end) in self.windows.items():
            print(f"\n{'='*80}")
            print(f"Window: {window_name} ({start} to {end})")
            print(f"{'='*80}")

            df_window = self.df_full[start:end]

            # Run baseline
            baseline_results = self._run_backtest(
                self.baseline_config,
                df_window,
                f"BASELINE ({window_name})"
            )

            # Create optimized config
            optimized_config = self._create_optimized_config()

            # Run optimized
            optimized_results = self._run_backtest(
                optimized_config,
                df_window,
                f"OPTIMIZED ({window_name})"
            )

            # Calculate deltas
            delta_pnl = optimized_results['total_pnl'] - baseline_results['total_pnl']
            delta_wr = optimized_results['win_rate'] - baseline_results['win_rate']
            delta_trap_pnl = optimized_results['trap_metrics']['total_pnl'] - baseline_results['trap_metrics']['total_pnl']
            delta_trap_wr = optimized_results['trap_metrics']['win_rate'] - baseline_results['trap_metrics']['win_rate']

            print(f"\n📊 Delta Summary:")
            print(f"  Overall PnL: ${delta_pnl:+.2f}")
            print(f"  Overall WR: {delta_wr:+.1%}")
            print(f"  Trap PnL: ${delta_trap_pnl:+.2f}")
            print(f"  Trap WR: {delta_trap_wr:+.1%}")

            results_comparison[window_name] = {
                'baseline': {
                    'total_pnl': baseline_results['total_pnl'],
                    'total_trades': baseline_results['total_trades'],
                    'win_rate': baseline_results['win_rate'],
                    'profit_factor': baseline_results['profit_factor'],
                    'max_drawdown_pct': baseline_results['max_drawdown_pct'],
                    'trap_trades': baseline_results['trap_metrics']['trades'],
                    'trap_pnl': baseline_results['trap_metrics']['total_pnl'],
                    'trap_wr': baseline_results['trap_metrics']['win_rate']
                },
                'optimized': {
                    'total_pnl': optimized_results['total_pnl'],
                    'total_trades': optimized_results['total_trades'],
                    'win_rate': optimized_results['win_rate'],
                    'profit_factor': optimized_results['profit_factor'],
                    'max_drawdown_pct': optimized_results['max_drawdown_pct'],
                    'trap_trades': optimized_results['trap_metrics']['trades'],
                    'trap_pnl': optimized_results['trap_metrics']['total_pnl'],
                    'trap_wr': optimized_results['trap_metrics']['win_rate']
                },
                'delta': {
                    'pnl': delta_pnl,
                    'wr': delta_wr,
                    'trap_pnl': delta_trap_pnl,
                    'trap_wr': delta_trap_wr
                }
            }

        # Save results
        self._save_validation_results(results_comparison)

        # Generate report
        self._generate_report(results_comparison)

    def _save_validation_results(self, results: Dict):
        """Save validation results to JSON."""
        output_path = self.output_dir / 'validation_results.json'

        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)

        print(f"\n💾 Validation results saved: {output_path}")

    def _generate_report(self, results: Dict):
        """Generate human-readable validation report."""
        print("\n" + "="*80)
        print("📊 VALIDATION REPORT - TRAP_WITHIN_TREND OPTIMIZATION")
        print("="*80)

        # Focus on 2022 (the problematic period)
        print("\n🎯 2022 Performance (Critical Period):")
        print("-" * 80)

        for period in ['2022_H1', '2022_H2']:
            if period in results:
                r = results[period]
                print(f"\n{period}:")
                print(f"  Baseline:  {r['baseline']['trap_trades']:3d} trades, ${r['baseline']['trap_pnl']:8.2f} PnL, {r['baseline']['trap_wr']:5.1%} WR")
                print(f"  Optimized: {r['optimized']['trap_trades']:3d} trades, ${r['optimized']['trap_pnl']:8.2f} PnL, {r['optimized']['trap_wr']:5.1%} WR")
                print(f"  Delta:     ${r['delta']['trap_pnl']:+8.2f} PnL, {r['delta']['trap_wr']:+5.1%} WR")

                # Color code improvement
                if r['delta']['trap_pnl'] > 0:
                    print(f"  ✅ IMPROVED by ${r['delta']['trap_pnl']:.2f}")
                else:
                    print(f"  ❌ DEGRADED by ${abs(r['delta']['trap_pnl']):.2f}")

        # Full period summary
        print("\n📈 Full Period (2022-2024) Summary:")
        print("-" * 80)

        if 'full' in results:
            r = results['full']
            print(f"\nBaseline Trap Performance:")
            print(f"  Trades: {r['baseline']['trap_trades']}")
            print(f"  Total PnL: ${r['baseline']['trap_pnl']:.2f}")
            print(f"  Avg PnL: ${r['baseline']['trap_pnl'] / max(r['baseline']['trap_trades'], 1):.2f}")
            print(f"  Win Rate: {r['baseline']['trap_wr']:.1%}")

            print(f"\nOptimized Trap Performance:")
            print(f"  Trades: {r['optimized']['trap_trades']}")
            print(f"  Total PnL: ${r['optimized']['trap_pnl']:.2f}")
            print(f"  Avg PnL: ${r['optimized']['trap_pnl'] / max(r['optimized']['trap_trades'], 1):.2f}")
            print(f"  Win Rate: {r['optimized']['trap_wr']:.1%}")

            print(f"\nDelta:")
            print(f"  PnL Change: ${r['delta']['trap_pnl']:+.2f}")
            print(f"  WR Change: {r['delta']['trap_wr']:+.1%}")

        # Acceptance criteria
        print("\n✅ Acceptance Criteria:")
        print("-" * 80)

        acceptance_passed = True

        # Criterion 1: 2022 improvement
        if 'full' in results:
            combined_2022_delta = sum(
                results.get(p, {}).get('delta', {}).get('trap_pnl', 0)
                for p in ['2022_H1', '2022_H2']
            )

            print(f"\n1. 2022 PnL Improvement: ${combined_2022_delta:+.2f}")
            if combined_2022_delta > 0:
                print(f"   ✅ PASS - 2022 performance improved")
            else:
                print(f"   ❌ FAIL - 2022 performance degraded")
                acceptance_passed = False

        # Criterion 2: Overall PnL improvement
        if 'full' in results:
            full_delta = results['full']['delta']['trap_pnl']
            print(f"\n2. Overall PnL Improvement: ${full_delta:+.2f}")
            if full_delta > 0:
                print(f"   ✅ PASS - Overall performance improved")
            else:
                print(f"   ❌ FAIL - Overall performance degraded")
                acceptance_passed = False

        # Criterion 3: Win rate improvement
        if 'full' in results:
            wr_delta = results['full']['delta']['trap_wr']
            print(f"\n3. Win Rate Change: {wr_delta:+.1%}")
            if wr_delta >= 0:
                print(f"   ✅ PASS - Win rate maintained or improved")
            else:
                print(f"   ⚠️  WARNING - Win rate declined")

        print("\n" + "="*80)
        if acceptance_passed:
            print("🎉 VALIDATION PASSED - Optimization improved trap_within_trend performance!")
        else:
            print("❌ VALIDATION FAILED - Optimization did not improve performance")
        print("="*80)


def main():
    parser = argparse.ArgumentParser(description='Validate trap_within_trend optimization')
    parser.add_argument('--cache', required=True, help='Path to cached feature parquet')
    parser.add_argument('--output', default='results/trap_validation', help='Output directory')

    args = parser.parse_args()

    validator = TrapValidationRunner(
        cache_path=args.cache,
        output_dir=args.output
    )

    validator.run_validation()

    return 0


if __name__ == '__main__':
    exit(main())
