#!/usr/bin/env python3
"""
Performance profiler for backtest_knowledge_v2.py

Runs cProfile and generates detailed performance reports to identify optimization targets.
"""

import sys
from pathlib import Path
import cProfile
import pstats
from pstats import SortKey
import argparse
import time
import os

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Profile backtest_knowledge_v2.py')
    parser.add_argument('--asset', default='BTC', help='Asset to backtest')
    parser.add_argument('--start', default='2024-01-01', help='Start date')
    parser.add_argument('--end', default='2024-09-30', help='End date')
    parser.add_argument('--config', default='configs/frozen/btc_1h_v2_baseline.json', help='Config file')
    parser.add_argument('--output', default='profile_baseline.prof', help='Output profile file')

    args = parser.parse_args()

    # Setup profiler
    profiler = cProfile.Profile()

    # Mock sys.argv for the backtest script
    sys.argv = [
        'backtest_knowledge_v2.py',
        '--asset', args.asset,
        '--start', args.start,
        '--end', args.end,
        '--config', args.config
    ]

    print(f"🔍 Starting profiler for {args.asset} {args.start} to {args.end}...")
    start_time = time.time()

    # Run the backtest script with profiling
    profiler.enable()
    try:
        # Execute the backtest script as if running it directly
        script_path = 'bin/backtest_knowledge_v2.py'
        with open(script_path) as f:
            code = compile(f.read(), script_path, 'exec')
            # Provide __file__ in the execution context
            exec_globals = {
                '__name__': '__main__',
                '__file__': os.path.abspath(script_path)
            }
            exec(code, exec_globals)
    finally:
        profiler.disable()

    elapsed = time.time() - start_time
    print(f"\n✅ Backtest completed in {elapsed:.2f}s")

    # Generate statistics
    print("\n" + "="*80)
    print("TOP 30 FUNCTIONS BY CUMULATIVE TIME")
    print("="*80)
    ps = pstats.Stats(profiler)
    ps.sort_stats(SortKey.CUMULATIVE)
    ps.print_stats(30)

    print("\n" + "="*80)
    print("TOP 30 FUNCTIONS BY SELF TIME")
    print("="*80)
    ps = pstats.Stats(profiler)
    ps.sort_stats(SortKey.TIME)
    ps.print_stats(30)

    print("\n" + "="*80)
    print("TOP 30 MOST CALLED FUNCTIONS")
    print("="*80)
    ps = pstats.Stats(profiler)
    ps.sort_stats(SortKey.CALLS)
    ps.print_stats(30)

    # Save detailed stats to file
    profiler.dump_stats(args.output)
    print(f"\n📊 Detailed profile saved to: {args.output}")
    print(f"   Analyze with: python -m pstats {args.output}")
    print(f"\n⏱️  Total execution time: {elapsed:.2f}s")
