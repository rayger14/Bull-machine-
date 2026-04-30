#!/usr/bin/env python3
"""
Validate v11 Overtrading Fix

Tests that v11 fixes produce expected trade count reduction:
- Before: 931 trades in Jan 2023 (3,750/year)
- After: 40-80 trades in Jan 2023 (150-300/year)

Usage:
    python bin/validate_v11_overtrading_fix.py
"""

import sys
import pandas as pd
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from engine.integrations.isolated_archetype_engine import IsolatedArchetypeEngine
from engine.context.regime_service import RegimeService
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def load_test_data():
    """Load Q1 2023 test data."""
    data_path = Path("data/btc_1h_2023_Q1.csv")

    if not data_path.exists():
        logger.error(f"Test data not found: {data_path}")
        logger.info("Please create test data file or use existing feature store")
        return None

    df = pd.read_csv(data_path)
    logger.info(f"Loaded {len(df)} bars from {data_path}")
    return df


def run_v11_fixed_test():
    """
    Test v11 with overtrading fixes.

    Expected outcomes:
    - Total signals: 40-150 (was 931 in broken v11)
    - Multiple archetypes firing
    - Cooling period blocks visible in stats
    """
    logger.info("="*80)
    logger.info("TESTING v11 WITH OVERTRADING FIXES")
    logger.info("="*80)

    # Load config
    import json
    config_path = Path("configs/bull_machine_isolated_v11_fixed.json")

    if not config_path.exists():
        logger.error(f"Config not found: {config_path}")
        return None

    with open(config_path) as f:
        config = json.load(f)

    logger.info(f"Config: {config['version']}")
    logger.info(f"Relative score percentile: {config['portfolio_allocation']['relative_score_percentile']}")
    logger.info(f"Fusion thresholds: {config['fusion_thresholds_by_regime']}")

    # Initialize engine
    engine = IsolatedArchetypeEngine(
        archetype_config_dir=config['archetype_config_dir'],
        portfolio_config=config['portfolio_allocation'],
        enable_regime=config['regime_classifier']['enabled'],
        regime_model_path=config['regime_classifier'].get('model_path')
    )

    logger.info(f"Initialized engine with {len(engine.archetypes)} archetypes")

    # Load test data
    df = load_test_data()
    if df is None:
        logger.error("Cannot proceed without test data")
        return None

    # Set timestamp as index
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df.set_index('timestamp', inplace=True)

    # Process bars
    all_signals = []

    logger.info(f"\nProcessing {len(df)} bars...")

    for bar_idx, (timestamp, bar) in enumerate(df.iterrows()):
        # Get signals
        signals = engine.get_signals(bar, bar_index=bar_idx)

        if signals:
            logger.info(
                f"[BAR {bar_idx}] {timestamp}: {len(signals)} signals "
                f"({', '.join(s.archetype_id for s in signals)})"
            )
            all_signals.extend(signals)

    # Print summary
    logger.info("\n" + "="*80)
    logger.info("TEST RESULTS")
    logger.info("="*80)

    stats = engine.get_stats()

    logger.info(f"Total bars processed: {stats['total_bars']}")
    logger.info(f"Total signals generated: {stats['total_signals']}")
    logger.info(f"Signals filtered by score: {stats['signals_filtered_by_score']}")
    logger.info(f"Signals blocked by cooling: {stats['signals_blocked_by_cooling']}")
    logger.info(f"Signal rate: {stats['signal_rate']:.4f} per bar")

    logger.info("\nSignals by archetype:")
    for archetype, count in sorted(stats['signals_by_archetype'].items(), key=lambda x: -x[1]):
        if count > 0:
            pct = count / stats['total_signals'] * 100 if stats['total_signals'] > 0 else 0
            logger.info(f"  {archetype:25s}: {count:4d} ({pct:.1f}%)")

    # Evaluate success
    logger.info("\n" + "="*80)
    logger.info("SUCCESS CRITERIA")
    logger.info("="*80)

    total_signals = stats['total_signals']

    # Success criteria
    criteria = {
        "Signal count in range (40-150)": 40 <= total_signals <= 150,
        "Multiple archetypes firing (>3)": sum(1 for c in stats['signals_by_archetype'].values() if c > 0) > 3,
        "Cooling period active": stats['signals_blocked_by_cooling'] > 0,
        "Score filtering active": stats['signals_filtered_by_score'] > 0,
    }

    for criterion, passed in criteria.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        logger.info(f"{status}: {criterion}")

    all_passed = all(criteria.values())

    if all_passed:
        logger.info("\n🎉 ALL TESTS PASSED - v11 overtrading fix is working!")
    else:
        logger.warning("\n⚠️  SOME TESTS FAILED - review implementation")

    return {
        'stats': stats,
        'signals': all_signals,
        'passed': all_passed
    }


if __name__ == "__main__":
    try:
        result = run_v11_fixed_test()

        if result is None:
            logger.error("Test failed to complete")
            sys.exit(1)

        sys.exit(0 if result['passed'] else 1)

    except Exception as e:
        logger.exception(f"Test failed with exception: {e}")
        sys.exit(1)
