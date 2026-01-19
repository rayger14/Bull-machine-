#!/usr/bin/env python3
"""Extract metrics from all validation logs and create summary table."""

import re
import pandas as pd
from pathlib import Path

def parse_backtest_log(log_path):
    """Parse backtest log and extract key metrics."""
    with open(log_path) as f:
        content = f.read()

    metrics = {}

    # Extract metrics using regex
    patterns = {
        'profit_factor': r'Profit Factor[:\s]+([0-9.]+)',
        'win_rate': r'Win Rate[:\s]+([0-9.]+)',
        'total_trades': r'Total Trades[:\s]+(\d+)',
        'max_drawdown': r'Max(?:imum)? Drawdown[:\s]+([0-9.]+)',
        'sharpe': r'Sharpe[:\s]+([0-9.-]+)',
        'total_return': r'Total Return[:\s]+([0-9.-]+)',
    }

    for key, pattern in patterns.items():
        match = re.search(pattern, content, re.IGNORECASE)
        if match:
            metrics[key] = float(match.group(1))
        else:
            metrics[key] = None

    # Count archetype vs tier1 trades
    archetype_trades = len(re.findall(r'archetype_\w+', content))
    tier1_trades = len(re.findall(r'tier1_market', content))
    metrics['archetype_trades'] = archetype_trades
    metrics['tier1_trades'] = tier1_trades
    metrics['fallback_rate'] = tier1_trades / max(archetype_trades + tier1_trades, 1)

    return metrics

def create_validation_summary():
    """Create comprehensive validation summary."""

    results = []

    # Chaos window logs
    chaos_logs = [
        ('S4', 'terra_collapse', 'results/validation/s4_terra_collapse.log'),
        ('S1', 'ftx_collapse', 'results/validation/s1_ftx_collapse.log'),
        ('S5', 'cpi_shock', 'results/validation/s5_cpi_shock.log'),
    ]

    # Full period logs
    full_logs = [
        ('S4', 'train_2020_2022', 'results/validation/s4_train_2020_2022.log'),
        ('S4', 'test_2023', 'results/validation/s4_test_2023.log'),
        ('S4', 'oos_2024', 'results/validation/s4_oos_2024.log'),
        ('S1', 'train_2020_2022', 'results/validation/s1_train_2020_2022.log'),
        ('S1', 'test_2023', 'results/validation/s1_test_2023.log'),
        ('S1', 'oos_2024', 'results/validation/s1_oos_2024.log'),
        ('S5', 'train_2020_2022', 'results/validation/s5_train_2020_2022.log'),
        ('S5', 'test_2023', 'results/validation/s5_test_2023.log'),
        ('S5', 'oos_2024', 'results/validation/s5_oos_2024.log'),
    ]

    all_logs = chaos_logs + full_logs

    for archetype, period, log_path in all_logs:
        if Path(log_path).exists():
            metrics = parse_backtest_log(log_path)
            results.append({
                'Archetype': archetype,
                'Period': period,
                'PF': metrics.get('profit_factor'),
                'WR': f"{metrics.get('win_rate', 0):.1f}%" if metrics.get('win_rate') else 'N/A',
                'Trades': metrics.get('total_trades'),
                'Archetype': metrics.get('archetype_trades', 0),
                'Tier1': metrics.get('tier1_trades', 0),
                'Fallback%': f"{metrics.get('fallback_rate', 0) * 100:.1f}%",
                'MaxDD': f"{metrics.get('max_drawdown', 0):.1f}%" if metrics.get('max_drawdown') else 'N/A',
                'Sharpe': metrics.get('sharpe'),
                'Return': f"{metrics.get('total_return', 0):.1f}%" if metrics.get('total_return') else 'N/A',
            })
        else:
            print(f"Warning: {log_path} not found")

    df = pd.DataFrame(results)

    # Save to CSV
    df.to_csv('results/validation/archetype_validation_summary.csv', index=False)

    # Print summary
    print("\n" + "="*100)
    print("ARCHETYPE VALIDATION SUMMARY")
    print("="*100 + "\n")
    print(df.to_string(index=False))

    return df

if __name__ == "__main__":
    create_validation_summary()
