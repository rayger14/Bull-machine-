#!/usr/bin/env python3
"""
Test Archetype Variants - Complexity vs Performance Analysis
=============================================================
Runs backtests on Core, Core+, and Full variants for S1, S4, S5 archetypes.
Compares performance to determine optimal complexity level for ML ensemble.

Test Protocol:
- Period: 2024 only (1 year, full OI data available)
- Asset: BTC/USDT 1H
- Metrics: PF, WR, Trades, Max DD, Sharpe

Expected Pattern:
- Core: High frequency, lower precision
- Core+: Medium frequency, medium precision
- Full: Low frequency, high precision
"""

import sys
import json
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from engine.models.archetype_model import ArchetypeModel


def calculate_metrics(trades_list, equity_history, initial_capital=10000.0):
    """Calculate comprehensive backtest metrics."""
    if not trades_list:
        return {
            'total_trades': 0,
            'win_rate': 0.0,
            'profit_factor': 0.0,
            'total_pnl': 0.0,
            'max_drawdown': 0.0,
            'sharpe': 0.0,
            'avg_win': 0.0,
            'avg_loss': 0.0,
            'avg_bars_in_trade': 0.0,
            'final_equity': initial_capital
        }

    # Basic trade stats
    total_trades = len(trades_list)
    wins = [t for t in trades_list if t.get('pnl', 0) > 0]
    losses = [t for t in trades_list if t.get('pnl', 0) <= 0]

    win_rate = len(wins) / total_trades if total_trades > 0 else 0.0

    # PnL stats
    gross_profit = sum(t['pnl'] for t in wins) if wins else 0.0
    gross_loss = abs(sum(t['pnl'] for t in losses)) if losses else 0.0
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else (float('inf') if gross_profit > 0 else 0.0)
    total_pnl = sum(t.get('pnl', 0) for t in trades_list)

    # Drawdown
    equity_values = np.array([e['equity'] for e in equity_history])
    running_max = np.maximum.accumulate(equity_values)
    drawdown = (equity_values - running_max) / running_max
    max_drawdown = drawdown.min()

    # Sharpe (annualized)
    if len(equity_values) > 1:
        returns = np.diff(equity_values) / equity_values[:-1]
        if returns.std() != 0:
            # Assuming hourly data, annualize: sqrt(24*365)
            sharpe = (returns.mean() / returns.std()) * np.sqrt(24 * 365)
        else:
            sharpe = 0.0
    else:
        sharpe = 0.0

    # Average metrics
    avg_win = gross_profit / len(wins) if wins else 0.0
    avg_loss = -gross_loss / len(losses) if losses else 0.0

    # Trade duration
    durations = []
    for t in trades_list:
        if 'entry_time' in t and 'exit_time' in t:
            entry = pd.to_datetime(t['entry_time'])
            exit = pd.to_datetime(t['exit_time'])
            durations.append((exit - entry).total_seconds() / 3600)
    avg_bars_in_trade = np.mean(durations) if durations else 0.0

    final_equity = equity_values[-1] if len(equity_values) > 0 else initial_capital

    return {
        'total_trades': total_trades,
        'win_rate': win_rate,
        'profit_factor': profit_factor,
        'total_pnl': total_pnl,
        'max_drawdown': max_drawdown,
        'sharpe': sharpe,
        'avg_win': avg_win,
        'avg_loss': avg_loss,
        'avg_bars_in_trade': avg_bars_in_trade,
        'final_equity': final_equity
    }


@dataclass
class Position:
    """Simple position tracker."""
    direction: str
    entry_price: float
    entry_time: pd.Timestamp
    size: float
    stop_loss: float
    entry_bar_idx: int


def run_variant_backtest(config_path, start_date, end_date):
    """Run backtest for a single variant config."""
    print(f"\n{'='*80}")
    print(f"Testing: {config_path.stem}")
    print(f"{'='*80}")

    # Load config
    with open(config_path) as f:
        config = json.load(f)

    # Determine archetype name from config
    archetype_map = {
        's1': 'S1',
        's4': 'S4',
        's5': 'S5'
    }
    archetype_name = None
    for key, value in archetype_map.items():
        if key in config_path.stem.lower():
            archetype_name = value
            break

    if not archetype_name:
        print(f"ERROR: Could not determine archetype from filename: {config_path.stem}")
        return None

    print(f"Archetype: {archetype_name}")

    # Initialize model
    try:
        model = ArchetypeModel(
            config_path=str(config_path),
            archetype_name=archetype_name,
            name=f"{archetype_name}-{config_path.stem}"
        )
    except Exception as e:
        print(f"ERROR: Failed to initialize model: {e}")
        import traceback
        traceback.print_exc()
        return None

    # Load feature store data
    print(f"Loading feature store data for {start_date} to {end_date}...")

    # Find appropriate feature store file
    feature_files = list(project_root.glob('data/*features*.parquet'))
    if not feature_files:
        print("ERROR: No feature store files found!")
        return None

    # Use the most recent one
    feature_path = max(feature_files, key=lambda p: p.stat().st_mtime)
    print(f"Using feature store: {feature_path.name}")

    df = pd.read_parquet(feature_path)

    # Ensure timestamp index
    if 'timestamp' in df.columns:
        df.set_index('timestamp', inplace=True)
    elif not isinstance(df.index, pd.DatetimeIndex):
        print("ERROR: DataFrame has no timestamp index!")
        return None

    # Filter to date range
    df = df.loc[start_date:end_date]

    if df.empty:
        print(f"ERROR: No data in range {start_date} to {end_date}!")
        return None

    print(f"Loaded {len(df)} candles with {len(df.columns)} features")

    # Fit model (no-op for pre-configured)
    model.fit(df)

    # Run backtest
    print("Running backtest...")
    initial_capital = 10000.0
    equity = initial_capital
    position = None
    trades = []
    equity_history = [{'timestamp': df.index[0], 'equity': equity}]

    for i, (timestamp, bar) in enumerate(df.iterrows()):
        # Check for exit signals if in position
        if position is not None:
            # Simple stop loss check
            if position.direction == 'long' and bar['low'] <= position.stop_loss:
                # Exit at stop loss
                exit_price = position.stop_loss
                pnl = (exit_price - position.entry_price) * position.size
                equity += pnl

                trades.append({
                    'entry_time': position.entry_time,
                    'exit_time': timestamp,
                    'direction': position.direction,
                    'entry_price': position.entry_price,
                    'exit_price': exit_price,
                    'pnl': pnl,
                    'exit_reason': 'stop_loss'
                })

                position = None

            elif position.direction == 'short' and bar['high'] >= position.stop_loss:
                # Exit at stop loss
                exit_price = position.stop_loss
                pnl = (position.entry_price - exit_price) * position.size
                equity += pnl

                trades.append({
                    'entry_time': position.entry_time,
                    'exit_time': timestamp,
                    'direction': position.direction,
                    'entry_price': position.entry_price,
                    'exit_price': exit_price,
                    'pnl': pnl,
                    'exit_reason': 'stop_loss'
                })

                position = None

            # Simple time-based exit (72 hours for S1, 48 for S4, 24 for S5)
            elif position is not None:
                bars_held = i - position.entry_bar_idx
                max_hold_hours = {'S1': 72, 'S4': 48, 'S5': 24}
                if bars_held >= max_hold_hours.get(archetype_name, 48):
                    exit_price = bar['close']
                    if position.direction == 'long':
                        pnl = (exit_price - position.entry_price) * position.size
                    else:
                        pnl = (position.entry_price - exit_price) * position.size
                    equity += pnl

                    trades.append({
                        'entry_time': position.entry_time,
                        'exit_time': timestamp,
                        'direction': position.direction,
                        'entry_price': position.entry_price,
                        'exit_price': exit_price,
                        'pnl': pnl,
                        'exit_reason': 'time_limit'
                    })

                    position = None

        # Check for entry signals if no position
        if position is None:
            signal = model.predict(bar)

            if signal.is_entry:
                # Calculate position size
                position_size = model.get_position_size(bar, signal)

                position = Position(
                    direction=signal.direction,
                    entry_price=signal.entry_price,
                    entry_time=timestamp,
                    size=position_size,
                    stop_loss=signal.stop_loss,
                    entry_bar_idx=i
                )

        # Update equity history
        equity_history.append({'timestamp': timestamp, 'equity': equity})

    # Close any remaining position
    if position is not None:
        final_bar = df.iloc[-1]
        exit_price = final_bar['close']
        if position.direction == 'long':
            pnl = (exit_price - position.entry_price) * position.size
        else:
            pnl = (position.entry_price - exit_price) * position.size
        equity += pnl

        trades.append({
            'entry_time': position.entry_time,
            'exit_time': df.index[-1],
            'direction': position.direction,
            'entry_price': position.entry_price,
            'exit_price': exit_price,
            'pnl': pnl,
            'exit_reason': 'backtest_end'
        })

    # Calculate metrics
    metrics = calculate_metrics(trades, equity_history, initial_capital)

    # Add metadata
    metrics['config'] = config_path.stem
    metrics['variant_type'] = config.get('_production_metadata', {}).get('variant_type', 'UNKNOWN')
    metrics['enabled_engines'] = config.get('_production_metadata', {}).get('enabled_domain_engines', [])

    # Print summary
    print(f"\nResults for {config_path.stem}:")
    print(f"  Variant Type: {metrics['variant_type']}")
    print(f"  Enabled Engines: {', '.join(metrics['enabled_engines']) if isinstance(metrics['enabled_engines'], list) else metrics['enabled_engines']}")
    print(f"  Trades: {metrics['total_trades']}")
    print(f"  Win Rate: {metrics['win_rate']:.1%}")
    print(f"  Profit Factor: {metrics['profit_factor']:.2f}")
    print(f"  Total PnL: ${metrics['total_pnl']:.2f}")
    print(f"  Max DD: {metrics['max_drawdown']:.1%}")
    print(f"  Sharpe: {metrics['sharpe']:.2f}")

    return metrics


def compare_variants(results, archetype_name):
    """Compare variant results and determine winner."""
    print(f"\n{'='*80}")
    print(f"{archetype_name.upper()} VARIANT COMPARISON")
    print(f"{'='*80}")

    # Create comparison table
    table_data = []
    for r in results:
        engines = r['enabled_engines']
        if isinstance(engines, list):
            engine_str = f"{len(engines)}"
        else:
            engine_str = "?"

        table_data.append({
            'Variant': r['variant_type'],
            'Engines': engine_str,
            'PF': r['profit_factor'],
            'WR': r['win_rate'],
            'Trades': r['total_trades'],
            'MaxDD': r['max_drawdown'],
            'Sharpe': r['sharpe']
        })

    df = pd.DataFrame(table_data)

    # Determine winner (highest Sharpe as primary metric)
    winner_idx = df['Sharpe'].idxmax()
    df['Winner'] = ''
    df.loc[winner_idx, 'Winner'] = '✓'

    # Format table
    print("\n| Variant      | Engines | PF   | WR    | Trades | MaxDD   | Sharpe | Winner |")
    print("|--------------|---------|------|-------|--------|---------|--------|--------|")

    for _, row in df.iterrows():
        print(f"| {row['Variant']:<12} | {row['Engines']:<7} | "
              f"{row['PF']:.2f} | {row['WR']:.1%} | "
              f"{row['Trades']:<6} | {row['MaxDD']:>6.1%} | "
              f"{row['Sharpe']:>6.2f} | {row['Winner']:<6} |")

    # Winner summary
    winner = df.loc[winner_idx]
    print(f"\nWINNER: {winner['Variant']} ({winner['Engines']} engines)")
    print(f"RATIONALE: Best Sharpe ({winner['Sharpe']:.2f}), "
          f"PF {winner['PF']:.2f}, "
          f"{winner['Trades']} trades")

    # Recommendation
    config_name = results[winner_idx]['config']
    print(f"\nRECOMMENDATION: Use configs/variants/{config_name}.json for ML ensemble")

    return winner


def main():
    """Run all variant comparisons."""
    print("="*80)
    print("ARCHETYPE VARIANT TESTING - STEP 3")
    print("="*80)
    print("\nTest Protocol:")
    print("- Period: 2022 (Full Year - Bear Market)")
    print("- Asset: BTC/USDT 1H")
    print("- Variants: Core, Core+, Full for S1, S4, S5")
    print("- Objective: Find optimal complexity per archetype")
    print("\nNOTE: Using 2022 bear market as OOS test (most archetypes calibrated on other periods)")

    # Test period - Using 2022 as we have full feature store data for this year
    start_date = '2022-01-01'
    end_date = '2022-12-31'

    variants_dir = project_root / 'configs' / 'variants'

    # Test sets
    test_groups = {
        'S1 Liquidity Vacuum': [
            's1_core.json',
            's1_core_plus_time.json',
            's1_full.json'
        ],
        'S4 Funding Divergence': [
            's4_core.json',
            's4_core_plus_macro.json',
            's4_full.json'
        ],
        'S5 Long Squeeze': [
            's5_core.json',
            's5_core_plus_wyckoff.json',
            's5_full.json'
        ]
    }

    all_winners = {}

    # Run tests for each archetype
    for archetype_name, variant_files in test_groups.items():
        print(f"\n\n{'#'*80}")
        print(f"# TESTING: {archetype_name}")
        print(f"{'#'*80}")

        results = []
        for variant_file in variant_files:
            config_path = variants_dir / variant_file
            if not config_path.exists():
                print(f"WARNING: Config not found: {config_path}")
                continue

            try:
                metrics = run_variant_backtest(config_path, start_date, end_date)
                if metrics:
                    results.append(metrics)
            except Exception as e:
                print(f"ERROR testing {variant_file}: {e}")
                import traceback
                traceback.print_exc()

        # Compare and find winner
        if results:
            winner = compare_variants(results, archetype_name)
            all_winners[archetype_name] = winner

    # Overall summary
    print("\n\n" + "="*80)
    print("OVERALL SUMMARY - OPTIMAL COMPLEXITY PER ARCHETYPE")
    print("="*80)

    for arch_name, winner in all_winners.items():
        arch_code = arch_name.split()[0]  # S1, S4, or S5
        print(f"\n{arch_name}:")
        print(f"  Best Variant: {winner['Variant']}")
        print(f"  Metrics: PF={winner['PF']:.2f}, Sharpe={winner['Sharpe']:.2f}, "
              f"WR={winner['WR']:.1%}, Trades={winner['Trades']}")

    # Insight
    print("\n" + "="*80)
    print("KEY INSIGHT: Does Complexity Help or Hurt?")
    print("="*80)

    complexity_levels = {'CORE': 0, 'CORE+TIME': 1, 'CORE+MACRO': 1, 'CORE+WYCKOFF': 1, 'FULL': 2}
    winner_complexities = [complexity_levels.get(w['Variant'], 0) for w in all_winners.values()]
    avg_complexity = sum(winner_complexities) / len(winner_complexities) if winner_complexities else 0

    if avg_complexity < 0.5:
        print("FINDING: Simple CORE variants perform best")
        print("IMPLICATION: Ghost features may be adding noise, not signal")
    elif avg_complexity > 1.5:
        print("FINDING: Complex FULL variants perform best")
        print("IMPLICATION: Ghost features provide valuable context filtering")
    else:
        print("FINDING: Intermediate CORE+ variants perform best")
        print("IMPLICATION: Selective ghost features add value, but full complexity may overfit")

    print("\n" + "="*80)
    print("NEXT STEPS:")
    print("="*80)
    print("1. Use winning variant configs as ML ensemble inputs")
    print("2. Train meta-learner on selected complexity levels")
    print("3. Validate ensemble performance on 2025 OOS data")

    # Save results
    output_file = project_root / 'STEP3_VARIANT_TEST_RESULTS.json'
    with open(output_file, 'w') as f:
        results_data = {
            'test_date': datetime.now().isoformat(),
            'test_period': f"{start_date} to {end_date}",
            'winners': {k: dict(v) for k, v in all_winners.items()},
            'avg_complexity': avg_complexity
        }
        json.dump(results_data, f, indent=2, default=str)

    print(f"\nResults saved to: {output_file}")


if __name__ == '__main__':
    main()
