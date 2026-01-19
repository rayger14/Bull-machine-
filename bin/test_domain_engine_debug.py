#!/usr/bin/env python3
"""
Domain Engine Debug Script
Tests whether feature_flags control S1 domain engine behavior
Compares s1_core.json (Wyckoff only) vs s1_full.json (all 6 engines)
"""

import sys
import json
import logging
import pandas as pd
from pathlib import Path

# Setup paths
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from engine.archetypes import ArchetypeLogic
from engine.runtime.context import RuntimeContext
from engine.archetypes.threshold_policy import ThresholdPolicy

# Configure logging to see DOMAIN_DEBUG messages
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[logging.StreamHandler()]
)

logger = logging.getLogger(__name__)


def load_config(path: str) -> dict:
    """Load JSON config file"""
    with open(ROOT / path, 'r') as f:
        return json.load(f)


def run_quick_test(config_path: str, label: str, max_bars: int = 200):
    """Run quick test on first N bars of 2022 data"""

    logger.info(f"\n{'='*80}")
    logger.info(f"Testing: {label}")
    logger.info(f"Config: {config_path}")
    logger.info(f"{'='*80}\n")

    # Load config
    config = load_config(config_path)

    # Log feature_flags from config
    feature_flags = config.get('feature_flags', {})
    logger.info(f"[CONFIG] Feature Flags: {json.dumps(feature_flags, indent=2)}")

    # Load 2022 data from parquet
    logger.info("[DATA] Loading 2022 data from parquet...")
    data_path = ROOT / 'data' / 'features_mtf' / 'BTC_1H_2022_ENRICHED.parquet'

    if not data_path.exists():
        logger.error(f"[DATA] Data file not found: {data_path}")
        return None

    df = pd.read_parquet(data_path)
    logger.info(f"[DATA] Loaded {len(df)} bars from {data_path.name}")

    # Filter to early 2022 (bear market crash period where S1 should fire)
    df = df[(df.index >= '2022-01-01') & (df.index < '2022-03-01')]
    logger.info(f"[DATA] Filtered to Jan-Feb 2022: {len(df)} bars")

    # Limit to first N bars
    df = df.head(max_bars)
    logger.info(f"[TEST] Testing on first {len(df)} bars\n")

    # Initialize archetype logic
    archetype_logic = ArchetypeLogic(config)
    policy = ThresholdPolicy(config)

    # Track detections
    detections = []

    # Scan for S1 matches
    logger.info("[SCAN] Scanning for S1 Liquidity Vacuum patterns...")

    for i in range(len(df)):
        row = df.iloc[i]

        # Create runtime context (mimicking backtest_knowledge_v2.py structure)
        regime_probs = {'risk_on': 0.0, 'neutral': 0.0, 'risk_off': 0.5, 'crisis': 0.5}
        thresholds_resolved = policy.resolve(row.name, regime_probs)

        context = RuntimeContext(
            ts=row.name,
            row=row,
            regime_probs=regime_probs,
            regime_label='risk_off',  # Assume risk_off for 2022 bear market
            adapted_params={},
            thresholds=thresholds_resolved,
            metadata={
                'df': df,
                'index': i,
                'feature_flags': feature_flags
            }
        )

        # Use the unified detect() method
        archetype_name, score, liquidity_score = archetype_logic.detect(context)

        # Check if it's S1
        matched = (archetype_name == 'liquidity_vacuum')

        if matched:
            detections.append({
                'timestamp': row.name,
                'price': row['close'],
                'archetype': archetype_name,
                'score': score,
                'liquidity_score': liquidity_score
            })

    logger.info(f"\n{'='*80}")
    logger.info(f"[RESULTS] {label}")
    logger.info(f"Total Detections: {len(detections)}")

    if len(detections) > 0:
        logger.info(f"\nFirst 5 detections:")
        for i, det in enumerate(detections[:5]):
            logger.info(f"  #{i+1}: {det['timestamp']} | Price: {det['price']:.2f} | Score: {det['score']:.3f}")
    else:
        logger.info("No patterns detected!")

    logger.info(f"{'='*80}\n")

    return {
        'label': label,
        'config': config_path,
        'feature_flags': feature_flags,
        'detection_count': len(detections),
        'detections': detections[:5]  # Return first 5 for comparison
    }


def main():
    """Run comparison test"""

    logger.info("\n" + "="*80)
    logger.info("DOMAIN ENGINE DEBUG TEST")
    logger.info("Goal: Prove whether feature_flags control S1 behavior")
    logger.info("="*80 + "\n")

    # Test 1: s1_core.json (Wyckoff only)
    core_results = run_quick_test(
        'configs/variants/s1_core.json',
        'S1 CORE (Wyckoff Only)',
        max_bars=200
    )

    if core_results is None:
        logger.error("Core test failed!")
        return

    # Test 2: s1_full.json (All 6 engines)
    full_results = run_quick_test(
        'configs/variants/s1_full.json',
        'S1 FULL (All 6 Engines)',
        max_bars=200
    )

    if full_results is None:
        logger.error("Full test failed!")
        return

    # Comparison
    logger.info("\n" + "="*80)
    logger.info("COMPARISON ANALYSIS")
    logger.info("="*80 + "\n")

    logger.info(f"Core Detection Count: {core_results['detection_count']}")
    logger.info(f"Full Detection Count: {full_results['detection_count']}")

    if core_results['detection_count'] == full_results['detection_count']:
        logger.warning("\nWARNING: Detection counts are IDENTICAL!")
        logger.warning("This suggests feature_flags are NOT affecting trade decisions.")
        logger.warning("Domain engines may be modifying scores but not gates.\n")
    else:
        logger.info(f"\nDetection count DIFFERS: Core={core_results['detection_count']}, Full={full_results['detection_count']}")
        logger.info("Feature flags ARE affecting behavior!")

    # Conclusion
    logger.info("\nNOTE: Look for [DOMAIN_DEBUG] messages in logs above to see if:")
    logger.info("1. Feature flags are being read correctly")
    logger.info("2. Domain boost is being calculated")
    logger.info("3. Domain signals are being triggered")

    logger.info("\n" + "="*80)
    logger.info("NEXT STEPS:")
    logger.info("="*80)
    logger.info("1. Review logs above for [DOMAIN_DEBUG] messages")
    logger.info("2. Check if domain_boost is being calculated")
    logger.info("3. Verify domain_boost affects GATES (True/False) not just scores")
    logger.info("4. If scores change but gates don't, need architectural fix")
    logger.info("="*80 + "\n")


if __name__ == '__main__':
    main()
