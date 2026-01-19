#!/usr/bin/env python3
"""
Validate Optimized Archetype Configuration

Runs comprehensive validation suite on optimized config:
1. In-sample (2024 Q1-Q3): Should match optimization results
2. Out-of-sample (2024 Q4): True generalization test
3. Per-archetype breakdown: Individual archetype performance
4. Comparison to baseline: Show improvement vs unoptimized config

Usage:
    python bin/validate_optimized_config.py --config configs/optimized_archetypes.json
"""

import argparse
import json
import re
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Dict, List


def run_validation_backtest(
    config_path: str,
    start_date: str,
    end_date: str,
    label: str
) -> Dict:
    """
    Run validation backtest and extract metrics.

    Args:
        config_path: Path to config JSON
        start_date: Start date YYYY-MM-DD
        end_date: End date YYYY-MM-DD
        label: Label for this validation run

    Returns:
        Dict with metrics
    """
    print(f"\n🔄 Running {label} backtest ({start_date} to {end_date})...")

    cmd = [
        "python3",
        "bin/backtest_knowledge_v2.py",
        "--asset", "BTC",
        "--start", start_date,
        "--end", end_date,
        "--config", config_path
    ]

    result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    output = result.stdout + result.stderr

    # Extract metrics
    metrics = {
        'label': label,
        'period': f"{start_date} to {end_date}",
        'pnl': 0.0,
        'trades': 0,
        'roi': 0.0,
        'win_rate': 0.0,
        'drawdown': 0.0,
        'profit_factor': 0.0,
        'sharpe': 0.0
    }

    # Parse output
    pnl_match = re.search(r'Total PNL:\s+\$?([-\d,\.]+)', output)
    if pnl_match:
        metrics['pnl'] = float(pnl_match.group(1).replace(',', ''))

    trades_match = re.search(r'Total Trades:\s+(\d+)', output)
    if trades_match:
        metrics['trades'] = int(trades_match.group(1))

    roi_match = re.search(r'ROI:\s+([-\d\.]+)%', output)
    if roi_match:
        metrics['roi'] = float(roi_match.group(1))

    wr_match = re.search(r'Win Rate:\s+([\d\.]+)%', output)
    if wr_match:
        metrics['win_rate'] = float(wr_match.group(1))

    dd_match = re.search(r'Max Drawdown:\s+([\d\.]+)%', output)
    if dd_match:
        metrics['drawdown'] = float(dd_match.group(1))

    pf_match = re.search(r'Profit Factor:\s+([\d\.]+)', output)
    if pf_match:
        metrics['profit_factor'] = float(pf_match.group(1))

    sharpe_match = re.search(r'Sharpe:\s+([-\d\.]+)', output)
    if sharpe_match:
        metrics['sharpe'] = float(sharpe_match.group(1))

    return metrics


def print_validation_report(
    optimized_metrics: List[Dict],
    baseline_metrics: List[Dict] = None
):
    """
    Print comprehensive validation report.

    Args:
        optimized_metrics: List of validation results for optimized config
        baseline_metrics: Optional baseline results for comparison
    """
    print("\n" + "=" * 100)
    print("OPTIMIZED CONFIG VALIDATION REPORT")
    print("=" * 100)

    # Summary table
    print("\n📊 PERFORMANCE SUMMARY")
    print("-" * 100)
    print(f"{'Period':<30} {'PNL':>12} {'Trades':>8} {'PF':>8} {'WR':>8} {'DD':>8} {'Sharpe':>8}")
    print("-" * 100)

    for metrics in optimized_metrics:
        print(
            f"{metrics['label']:<30} "
            f"${metrics['pnl']:>11,.2f} "
            f"{metrics['trades']:>8} "
            f"{metrics['profit_factor']:>8.2f} "
            f"{metrics['win_rate']:>7.1f}% "
            f"{metrics['drawdown']:>7.1f}% "
            f"{metrics['sharpe']:>8.2f}"
        )

    # Comparison to baseline
    if baseline_metrics:
        print("\n" + "=" * 100)
        print("COMPARISON TO BASELINE")
        print("=" * 100)
        print(f"{'Period':<30} {'PNL Δ':>12} {'PF Δ':>10} {'WR Δ':>10} {'Trades Δ':>10}")
        print("-" * 100)

        for opt, base in zip(optimized_metrics, baseline_metrics):
            pnl_delta = opt['pnl'] - base['pnl']
            pf_delta = opt['profit_factor'] - base['profit_factor']
            wr_delta = opt['win_rate'] - base['win_rate']
            trades_delta = opt['trades'] - base['trades']

            pnl_emoji = "📈" if pnl_delta > 0 else "📉"
            pf_emoji = "📈" if pf_delta > 0 else "📉"

            print(
                f"{opt['label']:<30} "
                f"{pnl_emoji} ${pnl_delta:>10,.2f} "
                f"{pf_emoji} {pf_delta:>9.2f} "
                f"{'+' if wr_delta >= 0 else ''}{wr_delta:>8.1f}% "
                f"{'+' if trades_delta >= 0 else ''}{trades_delta:>9}"
            )

    print("\n" + "=" * 100)


def main():
    parser = argparse.ArgumentParser(description='Validate Optimized Archetype Config')
    parser.add_argument('--config', type=str, required=True,
                        help='Optimized config path')
    parser.add_argument('--baseline', type=str, default='configs/profile_production.json',
                        help='Baseline config for comparison')
    parser.add_argument('--skip-baseline', action='store_true',
                        help='Skip baseline comparison')

    args = parser.parse_args()

    if not Path(args.config).exists():
        print(f"❌ Error: Config '{args.config}' not found")
        return 1

    # Define validation periods
    validation_periods = [
        ("2024-01-01", "2024-09-30", "In-Sample (Q1-Q3 2024)"),
        ("2024-10-01", "2024-12-31", "Out-of-Sample (Q4 2024)"),
        ("2024-01-01", "2024-12-31", "Full Year 2024"),
    ]

    # Run validation on optimized config
    optimized_metrics = []
    for start, end, label in validation_periods:
        metrics = run_validation_backtest(args.config, start, end, label)
        optimized_metrics.append(metrics)

    # Run baseline comparison if requested
    baseline_metrics = None
    if not args.skip_baseline and Path(args.baseline).exists():
        print("\n" + "=" * 100)
        print("RUNNING BASELINE COMPARISON")
        print("=" * 100)

        baseline_metrics = []
        for start, end, label in validation_periods:
            metrics = run_validation_backtest(args.baseline, start, end, f"Baseline {label}")
            baseline_metrics.append(metrics)

    # Print report
    print_validation_report(optimized_metrics, baseline_metrics)

    # Save results to JSON
    results = {
        'timestamp': datetime.now().isoformat(),
        'optimized_config': args.config,
        'baseline_config': args.baseline if baseline_metrics else None,
        'optimized_metrics': optimized_metrics,
        'baseline_metrics': baseline_metrics
    }

    output_path = Path('results') / f"validation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    output_path.parent.mkdir(exist_ok=True)

    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n💾 Results saved to {output_path}")

    # Determine success
    oos_metrics = optimized_metrics[1]  # Q4 2024 out-of-sample
    if oos_metrics['profit_factor'] >= 2.0 and oos_metrics['win_rate'] >= 60.0:
        print("\n✅ VALIDATION PASSED - Config ready for production!")
        return 0
    else:
        print("\n⚠️  VALIDATION WARNING - Out-of-sample performance below target")
        print(f"   Target: PF >= 2.0, WR >= 60%")
        print(f"   Actual: PF = {oos_metrics['profit_factor']:.2f}, WR = {oos_metrics['win_rate']:.1f}%")
        return 1


if __name__ == '__main__':
    exit(main())
