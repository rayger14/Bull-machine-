#!/usr/bin/env python3
"""
Bull Machine v1.7 CLI Runner
Implements battle-tested tiered optimization strategy
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import argparse
import json
import time
from datetime import datetime, timedelta
from pathlib import Path

from scripts.tiered_testing import TieredTester
from scripts.config_sweep import ConfigSweep

def run_preflight(args):
    """Run Tier 0 preflight checks"""
    print("🔍 RUNNING PREFLIGHT CHECKS")
    print("=" * 40)

    assets = args.assets.split(',') if args.assets else ['ETH_4H', 'ETH_1D']
    assets_with_tf = []

    for asset in assets:
        if '_' not in asset:
            # Add default timeframes
            assets_with_tf.extend([f"{asset}_4H", f"{asset}_1D"])
        else:
            assets_with_tf.append(asset)

    tester = TieredTester()
    results = tester.tier0_preflight(assets_with_tf, timeout=args.timeout)

    if results['status'] == 'pass':
        print("✅ All preflight checks passed")
        return 0
    else:
        print("❌ Preflight checks failed")
        for issue in results['issues']:
            print(f"   - {issue}")
        return 1

def run_smoke(args):
    """Run Tier 1 smoke slice test"""
    print("🧪 RUNNING SMOKE SLICE TEST")
    print("=" * 40)

    config = args.config or 'configs/v170/assets/ETH_v17_tuned.json'
    asset = args.assets or 'ETH_4H'

    tester = TieredTester(config)
    results = tester.tier1_smoke_slice(
        config=config,
        asset=asset,
        start_date=args.start,
        end_date=args.end,
        timeout=args.timeout
    )

    print(f"Status: {results['status']}")
    if results['status'] == 'pass':
        print(f"Return: {results.get('total_return', 0):+.2f}%")
        print(f"Trades: {results.get('total_trades', 0)}")
        print(f"Duration: {results.get('duration', 0):.1f}s")
        return 0
    else:
        print(f"Error: {results.get('error', 'Unknown')}")
        return 1

def run_walk_forward(args):
    """Run Tier 2 walk-forward validation"""
    print("🚶 RUNNING WALK-FORWARD VALIDATION")
    print("=" * 40)

    # Parse windows
    windows = []
    for window_str in args.windows.split(','):
        start, end = window_str.split(':')
        windows.append((start.strip(), end.strip()))

    config = args.config or 'configs/v170/assets/ETH_v17_tuned.json'

    tester = TieredTester(config)
    results = tester.tier2_walk_forward(config, windows, timeout=args.timeout)

    print(f"Status: {results['status']}")
    print(f"Windows passed: {results['windows_passed']}/{results['total_windows']}")

    if results['status'] == 'pass':
        print(f"Average return: {results.get('avg_return', 0):+.2f}%")
        print(f"Consistency: {results.get('return_consistency', 0):.3f}")
        return 0
    else:
        return 1

def run_full(args):
    """Run Tier 3 full backtest"""
    print("🏁 RUNNING FULL BACKTEST")
    print("=" * 40)

    config = args.config or 'configs/v170/assets/ETH_v17_tuned.json'
    assets = args.assets.split(',') if args.assets else ['ETH_4H']

    # Add timeframe suffix if missing
    assets_with_tf = []
    for asset in assets:
        if '_' not in asset:
            assets_with_tf.append(f"{asset}_4H")
        else:
            assets_with_tf.append(asset)

    tester = TieredTester(config)
    results = tester.tier3_full_backtest(
        config=config,
        assets=assets_with_tf,
        months=args.months,
        timeout=args.timeout
    )

    print(f"Status: {results['status']}")
    print(f"Assets passed: {results['assets_passed']}/{results['total_assets']}")

    if results['status'] == 'pass':
        print(f"Portfolio return: {results.get('portfolio_return', 0):+.2f}%")
        print(f"Duration: {results.get('duration', 0):.1f}s")
        return 0
    else:
        return 1

def run_sweep(args):
    """Run config parameter sweep"""
    print("🔬 RUNNING CONFIG SWEEP")
    print("=" * 40)

    # Define search space based on args or use default
    search_space = {
        'confidence_threshold': (0.25, 0.35),
        'strength_threshold': (0.35, 0.45),
        'smc_params': {
            'ob_threshold': (0.3, 0.7)
        },
        'momentum_params': {
            'rsi_period': (12, 16)
        }
    }

    # Override with custom search space if provided
    if args.search_space:
        with open(args.search_space, 'r') as f:
            search_space = json.load(f)

    sweeper = ConfigSweep()
    results = sweeper.run_sweep(
        search_space=search_space,
        max_configs=args.max_configs,
        keep_ratio=args.keep_ratio,
        parallel=args.parallel,
        max_workers=args.workers
    )

    print(f"Status: {results['status']}")
    print(f"Finalists: {results['finalists']}")

    if results['finalists'] > 0:
        print(f"Best config: {results['top_configs'][0]['config_id']}")
        return 0
    else:
        return 1

def run_quick_test(args):
    """Run quick end-to-end test"""
    print("⚡ RUNNING QUICK TEST")
    print("=" * 40)

    tester = TieredTester()

    # Tier 0: Preflight
    print("Step 1: Preflight...")
    preflight = tester.tier0_preflight(['ETH_4H', 'ETH_1D'], timeout=30)
    if preflight['status'] != 'pass':
        print("❌ Preflight failed")
        return 1

    # Tier 1: Quick smoke
    print("Step 2: Smoke test...")
    smoke = tester.tier1_smoke_slice(
        config='configs/v170/assets/ETH_v17_tuned.json',
        start_date='2025-08-01',
        end_date='2025-09-01',
        timeout=300
    )

    print(f"✅ Quick test complete")
    print(f"Status: {smoke['status']}")
    if smoke['status'] == 'pass':
        print(f"Return: {smoke.get('total_return', 0):+.2f}%")
        return 0
    else:
        return 1

def main():
    """Main CLI entry point"""
    parser = argparse.ArgumentParser(
        description='Bull Machine v1.7 Tiered Testing CLI',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Quick preflight check
  python scripts/run.py preflight --assets ETH,BTC

  # Smoke test (60-90 days)
  python scripts/run.py smoke --start 2025-07-01 --end 2025-09-01

  # Walk-forward validation
  python scripts/run.py walk --windows "2025-07-01:2025-08-15,2025-08-15:2025-09-30"

  # Full backtest (12+ months)
  python scripts/run.py full --assets ETH,BTC --months 18

  # Config sweep
  python scripts/run.py sweep --max-configs 40 --parallel

  # Quick test (preflight + smoke)
  python scripts/run.py quick
        """
    )

    subparsers = parser.add_subparsers(dest='command', help='Available commands')

    # Preflight command
    preflight_parser = subparsers.add_parser('preflight', help='Run Tier 0 preflight checks')
    preflight_parser.add_argument('--assets', default='ETH,BTC',
                                 help='Assets to check (comma-separated)')
    preflight_parser.add_argument('--timeout', type=int, default=30,
                                 help='Timeout in seconds')

    # Smoke command
    smoke_parser = subparsers.add_parser('smoke', help='Run Tier 1 smoke slice test')
    smoke_parser.add_argument('--config', help='Config file path')
    smoke_parser.add_argument('--assets', default='ETH_4H', help='Asset to test')
    smoke_parser.add_argument('--start', default='2025-07-01', help='Start date')
    smoke_parser.add_argument('--end', default='2025-09-01', help='End date')
    smoke_parser.add_argument('--timeout', type=int, default=600, help='Timeout in seconds')

    # Walk-forward command
    walk_parser = subparsers.add_parser('walk', help='Run Tier 2 walk-forward validation')
    walk_parser.add_argument('--config', help='Config file path')
    walk_parser.add_argument('--windows', required=True,
                           help='Windows as "start1:end1,start2:end2"')
    walk_parser.add_argument('--timeout', type=int, default=1200, help='Timeout in seconds')

    # Full command
    full_parser = subparsers.add_parser('full', help='Run Tier 3 full backtest')
    full_parser.add_argument('--config', help='Config file path')
    full_parser.add_argument('--assets', default='ETH', help='Assets (comma-separated)')
    full_parser.add_argument('--months', type=int, default=18, help='Months to backtest')
    full_parser.add_argument('--timeout', type=int, default=1800, help='Timeout in seconds')

    # Sweep command
    sweep_parser = subparsers.add_parser('sweep', help='Run config parameter sweep')
    sweep_parser.add_argument('--search-space', help='JSON file with search space')
    sweep_parser.add_argument('--max-configs', type=int, default=40,
                             help='Max configs to test')
    sweep_parser.add_argument('--keep-ratio', type=float, default=0.25,
                             help='Ratio of configs to keep')
    sweep_parser.add_argument('--parallel', action='store_true',
                             help='Run in parallel')
    sweep_parser.add_argument('--workers', type=int, help='Number of workers')

    # Quick command
    quick_parser = subparsers.add_parser('quick', help='Run quick test (preflight + smoke)')

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return 1

    # Route to appropriate function
    command_map = {
        'preflight': run_preflight,
        'smoke': run_smoke,
        'walk': run_walk_forward,
        'full': run_full,
        'sweep': run_sweep,
        'quick': run_quick_test
    }

    start_time = time.time()
    exit_code = command_map[args.command](args)
    duration = time.time() - start_time

    print(f"\n⏱️  Total duration: {duration:.1f}s")
    return exit_code

if __name__ == "__main__":
    exit(main())