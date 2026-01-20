#!/usr/bin/env python3
"""
Knowledge v2.0 A/B/C Comparison Tool

Runs 3 test scenarios and compares performance:
1. BASELINE: knowledge_v2.enabled = false
2. SHADOW: knowledge_v2 logs only (should match baseline PNL)
3. ACTIVE: knowledge_v2 modifies decisions

Usage:
    python3 bin/compare_knowledge_v2_abc.py \
        --asset ETH \
        --start 2024-07-01 \
        --end 2024-09-30 \
        --configs configs/knowledge_v2/ETH_baseline.json,configs/knowledge_v2/ETH_shadow_mode.json,configs/knowledge_v2/ETH_v2_active.json \
        --output reports/v2_ab_test
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import json
import argparse
import subprocess
from datetime import datetime
from typing import Dict, List
import pandas as pd


def run_backtest(asset: str, start: str, end: str, config_path: str, output_dir: Path) -> Dict:
    """
    Run hybrid_runner backtest with given config.

    Returns:
        Dict with performance metrics extracted from portfolio summary
    """
    config_name = Path(config_path).stem
    log_file = output_dir / f'{config_name}.log'

    print(f"\n{'='*70}")
    print(f"Running: {config_name}")
    print(f"Config: {config_path}")
    print(f"Log: {log_file}")
    print(f"{'='*70}\n")

    # Run hybrid_runner
    cmd = [
        'python3', 'bin/live/hybrid_runner.py',
        '--asset', asset,
        '--start', start,
        '--end', end,
        '--config', config_path
    ]

    with open(log_file, 'w') as f:
        result = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, text=True)

    if result.returncode != 0:
        print(f"❌ Test failed with exit code {result.returncode}")
        print(f"   Check log: {log_file}")
        return None

    print(f"✅ Test complete")

    # Parse performance metrics from log
    metrics = parse_metrics_from_log(log_file, config_name)

    return metrics


def parse_metrics_from_log(log_file: Path, config_name: str) -> Dict:
    """
    Parse performance metrics from hybrid_runner log output.

    Extracts:
    - Final Balance
    - Total P&L
    - Total Trades
    - Win Rate
    - Profit Factor
    - Max Drawdown
    - Sharpe Ratio (if available)
    """
    metrics = {
        'config': config_name,
        'final_balance': 0.0,
        'total_pnl': 0.0,
        'total_trades': 0,
        'win_rate': 0.0,
        'profit_factor': 0.0,
        'max_drawdown': 0.0,
        'sharpe_ratio': 0.0,
        'avg_trade_pnl': 0.0,
        'largest_win': 0.0,
        'largest_loss': 0.0
    }

    try:
        with open(log_file, 'r') as f:
            content = f.read()

        # Parse key metrics from portfolio summary
        lines = content.split('\n')

        for i, line in enumerate(lines):
            # Final Balance
            if 'Final Balance:' in line:
                try:
                    balance_str = line.split('$')[1].split()[0].replace(',', '')
                    metrics['final_balance'] = float(balance_str)
                except:
                    pass

            # Total P&L
            if 'Total P&L:' in line:
                try:
                    pnl_str = line.split('$')[1].split()[0].replace(',', '')
                    metrics['total_pnl'] = float(pnl_str)
                except:
                    pass

            # Total Trades
            if 'Total Trades:' in line:
                try:
                    trades = int(line.split(':')[1].strip().split()[0])
                    metrics['total_trades'] = trades
                except:
                    pass

            # Win Rate
            if 'Win Rate:' in line:
                try:
                    win_rate = float(line.split(':')[1].strip().replace('%', ''))
                    metrics['win_rate'] = win_rate
                except:
                    pass

            # Profit Factor
            if 'Profit Factor:' in line:
                try:
                    pf = float(line.split(':')[1].strip().split()[0])
                    metrics['profit_factor'] = pf
                except:
                    pass

            # Max Drawdown
            if 'Max Drawdown:' in line:
                try:
                    dd_str = line.split(':')[1].strip().replace('%', '').replace('$', '').replace(',', '')
                    metrics['max_drawdown'] = float(dd_str.split()[0])
                except:
                    pass

            # Sharpe Ratio
            if 'Sharpe Ratio:' in line or 'Sharpe:' in line:
                try:
                    sharpe = float(line.split(':')[1].strip().split()[0])
                    metrics['sharpe_ratio'] = sharpe
                except:
                    pass

            # Avg Trade P&L
            if 'Avg Trade P&L:' in line or 'Average Trade:' in line:
                try:
                    avg_str = line.split('$')[1].split()[0].replace(',', '')
                    metrics['avg_trade_pnl'] = float(avg_str)
                except:
                    pass

        # Calculate return percentage
        if metrics['total_pnl'] != 0:
            metrics['return_pct'] = (metrics['total_pnl'] / 10000) * 100  # Assuming $10k initial

    except Exception as e:
        print(f"⚠️  Error parsing metrics from {log_file}: {e}")

    return metrics


def generate_comparison_report(results: List[Dict], output_dir: Path):
    """
    Generate comparison report showing baseline vs shadow vs active performance.

    Validates:
    - Baseline == Shadow PNL (proves shadow mode works)
    - Active vs Baseline performance delta
    """
    output_file = output_dir / 'comparison_summary.json'
    report_file = output_dir / 'comparison_report.txt'

    # Save raw metrics
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n💾 Raw metrics saved: {output_file}")

    # Create readable report
    baseline = next((r for r in results if 'baseline' in r['config'].lower()), None)
    shadow = next((r for r in results if 'shadow' in r['config'].lower()), None)
    active = next((r for r in results if 'active' in r['config'].lower()), None)

    with open(report_file, 'w') as f:
        f.write("="*80 + "\n")
        f.write("Knowledge v2.0 A/B/C Test Comparison Report\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("="*80 + "\n\n")

        # Performance Summary Table
        f.write("PERFORMANCE SUMMARY\n")
        f.write("-"*80 + "\n")
        f.write(f"{'Metric':<25} {'Baseline':<15} {'Shadow':<15} {'Active':<15}\n")
        f.write("-"*80 + "\n")

        metrics_to_show = [
            ('Final Balance', 'final_balance', '$'),
            ('Total P&L', 'total_pnl', '$'),
            ('Return %', 'return_pct', '%'),
            ('Total Trades', 'total_trades', ''),
            ('Win Rate', 'win_rate', '%'),
            ('Profit Factor', 'profit_factor', ''),
            ('Max Drawdown', 'max_drawdown', '%'),
            ('Sharpe Ratio', 'sharpe_ratio', ''),
            ('Avg Trade P&L', 'avg_trade_pnl', '$')
        ]

        for label, key, suffix in metrics_to_show:
            base_val = baseline.get(key, 0) if baseline else 0
            shadow_val = shadow.get(key, 0) if shadow else 0
            active_val = active.get(key, 0) if active else 0

            if suffix == '$':
                base_str = f"${base_val:,.2f}"
                shadow_str = f"${shadow_val:,.2f}"
                active_str = f"${active_val:,.2f}"
            elif suffix == '%':
                base_str = f"{base_val:.2f}%"
                shadow_str = f"{shadow_val:.2f}%"
                active_str = f"{active_val:.2f}%"
            elif key == 'total_trades':
                base_str = f"{int(base_val)}"
                shadow_str = f"{int(shadow_val)}"
                active_str = f"{int(active_val)}"
            else:
                base_str = f"{base_val:.3f}"
                shadow_str = f"{shadow_val:.3f}"
                active_str = f"{active_val:.3f}"

            f.write(f"{label:<25} {base_str:<15} {shadow_str:<15} {active_str:<15}\n")

        f.write("-"*80 + "\n\n")

        # Validation Checks
        f.write("VALIDATION CHECKS\n")
        f.write("-"*80 + "\n")

        if baseline and shadow:
            # Check if baseline == shadow (proves shadow mode works)
            pnl_diff = abs(baseline.get('total_pnl', 0) - shadow.get('total_pnl', 0))
            pnl_diff_pct = (pnl_diff / abs(baseline.get('total_pnl', 1))) * 100 if baseline.get('total_pnl') else 0

            if pnl_diff_pct < 0.1:  # Less than 0.1% difference
                f.write("✅ PASS: Baseline == Shadow PNL (shadow mode validated)\n")
                f.write(f"   Difference: ${pnl_diff:.2f} ({pnl_diff_pct:.3f}%)\n\n")
            else:
                f.write("❌ FAIL: Baseline != Shadow PNL (shadow mode broken!)\n")
                f.write(f"   Difference: ${pnl_diff:.2f} ({pnl_diff_pct:.2f}%)\n\n")

        # Performance Comparison: Active vs Baseline
        if baseline and active:
            f.write("ACTIVE MODE PERFORMANCE vs BASELINE\n")
            f.write("-"*80 + "\n")

            # P&L comparison
            pnl_delta = active.get('total_pnl', 0) - baseline.get('total_pnl', 0)
            pnl_delta_pct = (pnl_delta / abs(baseline.get('total_pnl', 1))) * 100 if baseline.get('total_pnl') else 0

            f.write(f"P&L Delta:        ${pnl_delta:+,.2f} ({pnl_delta_pct:+.2f}%)\n")

            # Profit Factor comparison
            pf_delta = active.get('profit_factor', 0) - baseline.get('profit_factor', 0)
            f.write(f"Profit Factor:    {pf_delta:+.3f}\n")

            # Sharpe Ratio comparison
            sharpe_delta = active.get('sharpe_ratio', 0) - baseline.get('sharpe_ratio', 0)
            f.write(f"Sharpe Delta:     {sharpe_delta:+.3f}\n")

            # Max Drawdown comparison (lower is better)
            dd_delta = active.get('max_drawdown', 0) - baseline.get('max_drawdown', 0)
            f.write(f"Drawdown Delta:   {dd_delta:+.2f}% (negative = better)\n")

            # Trade count comparison
            trade_delta = active.get('total_trades', 0) - baseline.get('total_trades', 0)
            trade_delta_pct = (trade_delta / baseline.get('total_trades', 1)) * 100 if baseline.get('total_trades') else 0
            f.write(f"Trade Count:      {trade_delta:+d} ({trade_delta_pct:+.1f}%)\n\n")

            # Acceptance Gates
            f.write("ACCEPTANCE GATES (Must meet ≥3 of 4)\n")
            f.write("-"*80 + "\n")

            gates_passed = 0
            total_gates = 4

            # Gate 1: Profit Factor +0.10 uplift
            pf_pass = pf_delta >= 0.10
            gates_passed += pf_pass
            status = "✅" if pf_pass else "❌"
            f.write(f"{status} Profit Factor:  {active.get('profit_factor', 0):.3f} vs {baseline.get('profit_factor', 0):.3f} (target: +0.10)\n")

            # Gate 2: Sharpe Ratio +0.10 uplift
            sharpe_pass = sharpe_delta >= 0.10
            gates_passed += sharpe_pass
            status = "✅" if sharpe_pass else "❌"
            f.write(f"{status} Sharpe Ratio:   {active.get('sharpe_ratio', 0):.3f} vs {baseline.get('sharpe_ratio', 0):.3f} (target: +0.10)\n")

            # Gate 3: Max Drawdown ≤ baseline
            dd_pass = active.get('max_drawdown', 999) <= baseline.get('max_drawdown', 0)
            gates_passed += dd_pass
            status = "✅" if dd_pass else "❌"
            f.write(f"{status} Max Drawdown:   {active.get('max_drawdown', 0):.2f}% vs {baseline.get('max_drawdown', 0):.2f}% (target: ≤ baseline)\n")

            # Gate 4: Trade Count ≥ 80% of baseline
            trade_pass = active.get('total_trades', 0) >= (baseline.get('total_trades', 999) * 0.80)
            gates_passed += trade_pass
            status = "✅" if trade_pass else "❌"
            f.write(f"{status} Trade Count:    {active.get('total_trades', 0)} vs {baseline.get('total_trades', 0)} (target: ≥80%)\n\n")

            # Overall verdict
            f.write(f"GATES PASSED: {gates_passed}/{total_gates}\n")
            if gates_passed >= 3:
                f.write("✅ VERDICT: Knowledge v2.0 PASSES acceptance criteria\n")
            else:
                f.write("❌ VERDICT: Knowledge v2.0 FAILS acceptance criteria - needs tuning\n")

        f.write("\n" + "="*80 + "\n")

    # Print to console
    print(f"\n📊 Comparison report generated: {report_file}")
    print("\n" + open(report_file).read())


def main():
    parser = argparse.ArgumentParser(
        description='Compare baseline vs ML Knowledge v2.0 performance',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    parser.add_argument('--asset', default='ETH', help='Asset to test (BTC, ETH, SOL)')
    parser.add_argument('--start', default='2024-07-01', help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end', default='2024-09-30', help='End date (YYYY-MM-DD)')
    parser.add_argument('--configs', required=True,
                       help='Comma-separated list of config paths (baseline,shadow,active)')
    parser.add_argument('--output', default='reports/v2_ab_test',
                       help='Output directory for results')

    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("="*80)
    print("Knowledge v2.0 A/B/C Comparison Tool")
    print("="*80)
    print(f"Asset:        {args.asset}")
    print(f"Period:       {args.start} → {args.end}")
    print(f"Output:       {output_dir}")
    print("="*80)

    # Parse config paths
    config_paths = [c.strip() for c in args.configs.split(',')]

    print(f"\nRunning {len(config_paths)} tests:")
    for i, path in enumerate(config_paths, 1):
        print(f"  {i}. {path}")

    # Run all tests
    results = []
    for config_path in config_paths:
        metrics = run_backtest(args.asset, args.start, args.end, config_path, output_dir)
        if metrics:
            results.append(metrics)

    # Generate comparison report
    if len(results) == len(config_paths):
        generate_comparison_report(results, output_dir)
        print(f"\n✅ All {len(results)} tests completed successfully")
    else:
        print(f"\n⚠️  Only {len(results)}/{len(config_paths)} tests completed successfully")
        print("   Check logs for errors")

    return 0


if __name__ == '__main__':
    sys.exit(main())
