#!/usr/bin/env python3
"""
PR-A: Parity Test - Legacy vs Adaptive Code Paths

This test ensures that the adaptive RuntimeContext path produces identical
trades to the legacy check_archetype() path when locked to static mode.

If this test fails, it means the two code paths diverge, and we must
fix the adaptive path to match legacy before proceeding with any regime work.

Usage:
    python3 tests/test_parity_legacy_vs_adaptive.py --asset BTC --start 2024-01-01 --end 2024-12-31

Expected Output:
    PASS: Both configs produce identical trade counts and archetype selections

    If FAIL:
        - Trade count mismatch
        - Archetype mismatch at specific bars
        - Diagnostic output showing where divergence occurred
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import json
import argparse
import pandas as pd
import logging
from typing import Dict, List, Tuple

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


def run_backtest(config_path: str, asset: str, start: str, end: str) -> pd.DataFrame:
    """
    Run backtest and return DataFrame of trade entries with timestamps and archetypes.

    Returns:
        DataFrame with columns: timestamp, archetype, fusion_score
    """
    import subprocess
    import tempfile

    # Create temp file for trade export
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        trade_file = f.name

    cmd = [
        'python3', 'bin/backtest_knowledge_v2.py',
        '--asset', asset,
        '--start', start,
        '--end', end,
        '--config', config_path,
        '--export-trades', trade_file
    ]

    logger.info(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        logger.error(f"Backtest failed:\n{result.stderr}")
        raise RuntimeError(f"Backtest failed for {config_path}")

    # Parse trade file
    try:
        trades = pd.read_csv(trade_file)
        logger.info(f"Config {config_path}: {len(trades)} trades")
        return trades
    except Exception as e:
        logger.error(f"Failed to parse trades from {trade_file}: {e}")
        raise


def compare_trade_lists(
    legacy_trades: pd.DataFrame,
    adaptive_trades: pd.DataFrame,
    tolerance: int = 0
) -> Tuple[bool, Dict]:
    """
    Compare two trade lists for parity.

    Args:
        legacy_trades: Trades from legacy static config
        adaptive_trades: Trades from adaptive locked config
        tolerance: Allowed difference in trade count

    Returns:
        (passed, diagnostics)
    """
    diagnostics = {}
    passed = True

    # 1. Trade count comparison
    legacy_count = len(legacy_trades)
    adaptive_count = len(adaptive_trades)
    count_diff = abs(legacy_count - adaptive_count)

    diagnostics['legacy_count'] = legacy_count
    diagnostics['adaptive_count'] = adaptive_count
    diagnostics['count_diff'] = count_diff

    if count_diff > tolerance:
        logger.error(f"FAIL: Trade count mismatch (legacy={legacy_count}, adaptive={adaptive_count}, diff={count_diff})")
        passed = False
    else:
        logger.info(f"PASS: Trade counts match (legacy={legacy_count}, adaptive={adaptive_count})")

    # 2. Timestamp alignment (sample check)
    if 'entry_time' in legacy_trades.columns and 'entry_time' in adaptive_trades.columns:
        legacy_times = set(legacy_trades['entry_time'].values)
        adaptive_times = set(adaptive_trades['entry_time'].values)

        missing_in_adaptive = legacy_times - adaptive_times
        extra_in_adaptive = adaptive_times - legacy_times

        diagnostics['missing_in_adaptive'] = len(missing_in_adaptive)
        diagnostics['extra_in_adaptive'] = len(extra_in_adaptive)

        if missing_in_adaptive:
            logger.warning(f"  {len(missing_in_adaptive)} trades missing in adaptive:")
            for ts in sorted(list(missing_in_adaptive))[:5]:
                logger.warning(f"    - {ts}")

        if extra_in_adaptive:
            logger.warning(f"  {len(extra_in_adaptive)} extra trades in adaptive:")
            for ts in sorted(list(extra_in_adaptive))[:5]:
                logger.warning(f"    - {ts}")

        if missing_in_adaptive or extra_in_adaptive:
            passed = False

    # 3. Archetype distribution comparison
    if 'archetype' in legacy_trades.columns and 'archetype' in adaptive_trades.columns:
        legacy_archs = legacy_trades['archetype'].value_counts().to_dict()
        adaptive_archs = adaptive_trades['archetype'].value_counts().to_dict()

        diagnostics['legacy_archetypes'] = legacy_archs
        diagnostics['adaptive_archetypes'] = adaptive_archs

        all_archs = set(legacy_archs.keys()) | set(adaptive_archs.keys())
        for arch in sorted(all_archs):
            legacy_n = legacy_archs.get(arch, 0)
            adaptive_n = adaptive_archs.get(arch, 0)

            if legacy_n != adaptive_n:
                logger.warning(f"  Archetype {arch}: legacy={legacy_n}, adaptive={adaptive_n}")
                passed = False

    return passed, diagnostics


def create_locked_config(base_config_path: str, output_path: str):
    """
    Create a locked version of the adaptive config for parity testing.

    Adds "locked_regime": "static" to bypass regime blending.
    """
    with open(base_config_path) as f:
        config = json.load(f)

    config['locked_regime'] = 'static'
    config['description'] = config.get('description', '') + ' [PR-A: LOCKED to static for parity test]'

    with open(output_path, 'w') as f:
        json.dump(config, f, indent=2)

    logger.info(f"Created locked config: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='PR-A Parity Test: Legacy vs Adaptive')
    parser.add_argument('--asset', default='BTC', help='Asset to test')
    parser.add_argument('--start', default='2024-01-01', help='Start date')
    parser.add_argument('--end', default='2024-12-31', help='End date')
    parser.add_argument('--tolerance', type=int, default=0, help='Allowed trade count difference')
    parser.add_argument('--legacy-config', default='configs/baseline_btc_bull_pf20.json',
                        help='Legacy static config')
    parser.add_argument('--adaptive-config', default='configs/btc_v8_adaptive.json',
                        help='Adaptive config to lock')

    args = parser.parse_args()

    logger.info("="*80)
    logger.info("PR-A: Parity Test - Legacy vs Adaptive Code Paths")
    logger.info("="*80)

    # Step 1: Create locked version of adaptive config
    locked_config_path = '/tmp/btc_v8_adaptive_locked.json'
    create_locked_config(args.adaptive_config, locked_config_path)

    # Step 2: Run legacy config
    logger.info("\n--- Running LEGACY config (static) ---")
    legacy_trades = run_backtest(args.legacy_config, args.asset, args.start, args.end)

    # Step 3: Run locked adaptive config
    logger.info("\n--- Running ADAPTIVE config (locked to static) ---")
    adaptive_trades = run_backtest(locked_config_path, args.asset, args.start, args.end)

    # Step 4: Compare results
    logger.info("\n--- Parity Analysis ---")
    passed, diagnostics = compare_trade_lists(legacy_trades, adaptive_trades, args.tolerance)

    # Step 5: Report
    logger.info("\n" + "="*80)
    if passed:
        logger.info("✅ PARITY TEST PASSED")
        logger.info("   Legacy and adaptive paths produce identical results when locked.")
        logger.info("   Safe to proceed with regime-aware work.")
    else:
        logger.error("❌ PARITY TEST FAILED")
        logger.error("   Code paths diverge. Must fix adaptive path before proceeding.")
        logger.error("\nDiagnostics:")
        logger.error(json.dumps(diagnostics, indent=2))
        sys.exit(1)

    logger.info("="*80)


if __name__ == '__main__':
    main()
