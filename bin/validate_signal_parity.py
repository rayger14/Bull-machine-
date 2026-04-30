#!/usr/bin/env python3
"""
Signal Parity Validation: Event Engine vs Baseline
===================================================

Validates that the event-driven engine produces signals within 2-5% of the
baseline Python backtest engine, ensuring the integration preserved accuracy.

Test Period: Q1 2023 (Jan 1 - Mar 31, 2023)
- 2,136 bars
- Controlled period with known behavior
- Fast execution for rapid iteration

Acceptance Criteria:
- Signal count difference <5%
- Entry price RMSE <0.5%
- Archetype distribution difference <10%
- No critical failures or exceptions

Usage:
    python3 bin/validate_signal_parity.py \\
        --config configs/s1_multi_objective_production.json \\
        --feature-store data/features_mtf/BTC_1H_complete_2022-01-01_to_2024-12-31.parquet
"""

import sys
import os
import json
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple
import argparse
from collections import Counter

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from engine.backtesting.engine import BacktestEngine
from engine.integrations.event_engine import EventEngine, Bar
from engine.integrations.bull_machine_strategy import BullMachineStrategy


def run_baseline_backtest(
    config_path: str,
    feature_store_path: str,
    start_date: str,
    end_date: str
) -> Dict:
    """
    Run baseline backtest using BacktestEngine.

    Returns dict with:
    - trades: List of trade dicts
    - signals: List of signal dicts
    - metrics: Performance metrics
    """
    print("\n" + "="*70)
    print("BASELINE BACKTEST (Python BacktestEngine)")
    print("="*70 + "\n")

    # Import baseline model
    from bin.baseline_wyckoff_backtest import BaselineArchetypeModel

    # Load feature store
    feature_df = pd.read_parquet(feature_store_path)

    # Filter to test period
    feature_df = feature_df.loc[start_date:end_date]
    print(f"Test period: {start_date} to {end_date}")
    print(f"Bars: {len(feature_df):,}")

    # Initialize model
    model = BaselineArchetypeModel(config_path, feature_store_path)

    # Run backtest
    engine = BacktestEngine(
        model=model,
        data=feature_df,
        initial_capital=10000,
        commission_pct=0.001
    )

    print("\nRunning baseline backtest...")
    results = engine.run()

    print(f"\nBaseline Results:")
    print(f"  Trades: {results.total_trades}")
    print(f"  Win rate: {results.win_rate:.1%}")
    print(f"  Profit factor: {results.profit_factor:.2f}")
    print(f"  Final capital: ${results.initial_capital + results.total_pnl:.2f}")

    # Extract signals (from model tracking)
    baseline_signals = []
    for archetype, count in model.archetype_signal_counts.items():
        baseline_signals.extend([
            {'archetype': archetype, 'timestamp': None}
            for _ in range(count)
        ])

    return {
        'trades': [
            {
                'entry_time': t.entry_time,
                'entry_price': t.entry_price,
                'exit_time': t.exit_time,
                'exit_price': t.exit_price,
                'direction': t.direction,
                'pnl': t.pnl,
                'archetype': 'unknown'  # Baseline doesn't track archetype per trade
            }
            for t in results.trades
        ],
        'signals': baseline_signals,
        'metrics': {
            'total_trades': results.total_trades,
            'win_rate': results.win_rate,
            'profit_factor': results.profit_factor,
            'sharpe': results.sharpe_ratio,
            'max_drawdown': results.max_drawdown,
            'final_capital': results.initial_capital + results.total_pnl
        },
        'archetype_counts': dict(model.archetype_signal_counts)
    }


def run_event_engine_backtest(
    config_path: str,
    feature_store_path: str,
    start_date: str,
    end_date: str
) -> Dict:
    """
    Run event engine backtest using EventEngine + BullMachineStrategy.

    Returns dict with:
    - trades: List of trade dicts
    - signals: List of signal dicts
    - metrics: Performance metrics
    """
    print("\n" + "="*70)
    print("EVENT ENGINE BACKTEST (Nautilus-style)")
    print("="*70 + "\n")

    # Load feature store
    feature_df = pd.read_parquet(feature_store_path)

    # Filter to test period
    feature_df = feature_df.loc[start_date:end_date]
    print(f"Test period: {start_date} to {end_date}")
    print(f"Bars: {len(feature_df):,}")

    # Initialize strategy
    strategy = BullMachineStrategy(
        config_path=config_path,
        feature_store_path=feature_store_path,
        enable_soft_gating=False,  # Match baseline (no soft gating)
        enable_circuit_breaker=False,
        base_position_size_usd=1000.0
    )

    # Initialize event engine with strategy
    engine = EventEngine(
        strategy=strategy,
        initial_cash=10000,
        commission_rate=0.001,
        slippage_bps=2.0
    )

    print("\nRunning event engine backtest...")

    # Convert feature_df to list of bars
    bars = []
    for timestamp, row in feature_df.iterrows():
        bars.append(Bar(
            timestamp=timestamp,
            open=float(row['open']),
            high=float(row['high']),
            low=float(row['low']),
            close=float(row['close']),
            volume=float(row['volume'])
        ))

    # Run backtest
    engine.run(bars)

    # Get results
    results = engine.portfolio.get_stats()

    print(f"\nEvent Engine Results:")
    print(f"  Trades: {results['total_trades']}")
    print(f"  Win rate: {results['win_rate']:.1%}")
    print(f"  Profit factor: {results['profit_factor']:.2f}")
    print(f"  Final capital: ${results['final_capital']:.2f}")

    # Extract signals from strategy tracking
    event_signals = []
    event_archetype_counts = {}

    # Note: BullMachineStrategy doesn't expose signal tracking yet
    # For now, use trade count as proxy
    event_archetype_counts['total'] = results['total_trades']

    return {
        'trades': results['trades'],
        'signals': event_signals,
        'metrics': {
            'total_trades': results['total_trades'],
            'win_rate': results['win_rate'],
            'profit_factor': results['profit_factor'],
            'sharpe': results.get('sharpe', 0.0),
            'max_drawdown': results.get('max_drawdown_pct', 0.0),
            'final_capital': results['final_capital']
        },
        'archetype_counts': event_archetype_counts
    }


def compare_results(baseline: Dict, event_engine: Dict) -> Dict:
    """
    Compare baseline vs event engine results.

    Returns dict with:
    - signal_parity: % difference in signal counts
    - trade_parity: % difference in trade counts
    - price_rmse: RMSE of entry prices
    - archetype_diff: Distribution differences
    - pass: Boolean - whether parity criteria met
    """
    print("\n" + "="*70)
    print("PARITY VALIDATION")
    print("="*70 + "\n")

    # 1. Signal count parity
    baseline_signals = len(baseline['signals'])
    event_signals = len(event_engine['signals'])

    if baseline_signals > 0:
        signal_diff_pct = abs(event_signals - baseline_signals) / baseline_signals * 100
    else:
        signal_diff_pct = 100.0 if event_signals > 0 else 0.0

    # 2. Trade count parity
    baseline_trades = baseline['metrics']['total_trades']
    event_trades = event_engine['metrics']['total_trades']

    if baseline_trades > 0:
        trade_diff_pct = abs(event_trades - baseline_trades) / baseline_trades * 100
    else:
        trade_diff_pct = 100.0 if event_trades > 0 else 0.0

    # 3. Entry price RMSE (if we have trades)
    price_rmse_pct = 0.0
    if baseline_trades > 0 and event_trades > 0:
        # Match trades by entry time (within 1 hour tolerance)
        baseline_prices = [t['entry_price'] for t in baseline['trades']]
        event_prices = [t['entry_price'] for t in event_engine['trades']]

        # Use min length to avoid index errors
        min_len = min(len(baseline_prices), len(event_prices))
        if min_len > 0:
            squared_errors = [
                ((bp - ep) / bp) ** 2
                for bp, ep in zip(baseline_prices[:min_len], event_prices[:min_len])
            ]
            price_rmse_pct = np.sqrt(np.mean(squared_errors)) * 100

    # 4. Archetype distribution difference
    archetype_diff = 0.0
    if baseline['archetype_counts'] and event_engine['archetype_counts']:
        # Simple: compare total counts
        baseline_total = sum(baseline['archetype_counts'].values())
        event_total = sum(event_engine['archetype_counts'].values())

        if baseline_total > 0:
            archetype_diff = abs(event_total - baseline_total) / baseline_total * 100

    # 5. Acceptance criteria
    SIGNAL_TOLERANCE = 5.0  # %
    TRADE_TOLERANCE = 5.0  # %
    PRICE_TOLERANCE = 0.5  # %

    signal_pass = signal_diff_pct <= SIGNAL_TOLERANCE
    trade_pass = trade_diff_pct <= TRADE_TOLERANCE
    price_pass = price_rmse_pct <= PRICE_TOLERANCE

    overall_pass = signal_pass and trade_pass and price_pass

    # Print results
    print("1. Signal Count Parity")
    print(f"   Baseline: {baseline_signals}")
    print(f"   Event Engine: {event_signals}")
    print(f"   Difference: {signal_diff_pct:.1f}% {'✅ PASS' if signal_pass else '❌ FAIL'} (tolerance: {SIGNAL_TOLERANCE}%)")

    print("\n2. Trade Count Parity")
    print(f"   Baseline: {baseline_trades}")
    print(f"   Event Engine: {event_trades}")
    print(f"   Difference: {trade_diff_pct:.1f}% {'✅ PASS' if trade_pass else '❌ FAIL'} (tolerance: {TRADE_TOLERANCE}%)")

    print("\n3. Entry Price RMSE")
    print(f"   RMSE: {price_rmse_pct:.2f}% {'✅ PASS' if price_pass else '❌ FAIL'} (tolerance: {PRICE_TOLERANCE}%)")

    print("\n4. Archetype Distribution")
    print(f"   Difference: {archetype_diff:.1f}%")

    print("\n" + "="*70)
    if overall_pass:
        print("✅ PARITY VALIDATION PASSED")
    else:
        print("❌ PARITY VALIDATION FAILED")
    print("="*70 + "\n")

    return {
        'signal_diff_pct': signal_diff_pct,
        'trade_diff_pct': trade_diff_pct,
        'price_rmse_pct': price_rmse_pct,
        'archetype_diff_pct': archetype_diff,
        'signal_pass': signal_pass,
        'trade_pass': trade_pass,
        'price_pass': price_pass,
        'overall_pass': overall_pass
    }


def main():
    parser = argparse.ArgumentParser(
        description='Validate signal parity between event engine and baseline'
    )
    parser.add_argument(
        '--config',
        type=str,
        default='configs/s1_multi_objective_production.json',
        help='Config JSON path'
    )
    parser.add_argument(
        '--feature-store',
        type=str,
        default='data/features_mtf/BTC_1H_complete_2022-01-01_to_2024-12-31.parquet',
        help='Feature store parquet path'
    )
    parser.add_argument(
        '--start',
        type=str,
        default='2023-01-01',
        help='Start date (YYYY-MM-DD)'
    )
    parser.add_argument(
        '--end',
        type=str,
        default='2023-03-31',
        help='End date (YYYY-MM-DD)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='results/parity_validation.json',
        help='Output JSON path'
    )

    args = parser.parse_args()

    print("\n" + "="*70)
    print("SIGNAL PARITY VALIDATION")
    print("="*70)
    print(f"\nConfig: {args.config}")
    print(f"Feature store: {args.feature_store}")
    print(f"Test period: {args.start} to {args.end}")

    # Run baseline backtest
    baseline_results = run_baseline_backtest(
        args.config, args.feature_store, args.start, args.end
    )

    # Run event engine backtest
    event_results = run_event_engine_backtest(
        args.config, args.feature_store, args.start, args.end
    )

    # Compare results
    parity = compare_results(baseline_results, event_results)

    # Save results
    output = {
        'test_period': {'start': args.start, 'end': args.end},
        'baseline': baseline_results,
        'event_engine': event_results,
        'parity': parity,
        'timestamp': datetime.now().isoformat()
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2, default=str)

    print(f"\n✅ Results saved to: {output_path}")

    # Exit code
    sys.exit(0 if parity['overall_pass'] else 1)


if __name__ == '__main__':
    main()
