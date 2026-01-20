#!/usr/bin/env python3
"""
Test script to demonstrate the domain engine gate ordering bug.

HYPOTHESIS: Domain boosts modify scores AFTER the fusion threshold gate,
so marginal signals that would pass with domain context are rejected.

EXPECTED RESULT: Log messages showing [ARCH_BUG] warnings when domain
engines would have saved signals but couldn't due to gate ordering.
"""

import sys
import os
import pandas as pd
import logging
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from engine.archetypes.logic_v2_adapter import ArchetypeLogic
from engine.runtime.context import RuntimeContext
from engine.context.regime_classifier import RegimeClassifier

# Configure logging to see the [ARCH_BUG] warnings
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def create_test_row_marginal_signal():
    """
    Create a test row with:
    - Score JUST BELOW fusion threshold (e.g., 0.38 when threshold is 0.40)
    - Strong Wyckoff Spring signal (2.5x boost)
    - Expected: WITHOUT fix, rejected. WITH fix, accepted.

    IMPORTANT: Must have V2 features to trigger V2 logic path.
    """
    return pd.Series({
        # S1 V2 features (marginal capitulation) - REQUIRED for V2 logic
        'capitulation_depth_pct': -0.22,  # 22% drawdown (moderate)
        'crisis_composite': 0.63,  # Crisis but not extreme (just above min 0.60)
        'volume_climax_3bar': 0.32,  # Marginal volume spike (just above min 0.30)
        'wick_exhaustion_3bar': 0.22,  # Marginal wick exhaustion (just above min 0.20)
        'liquidity_drain_pct': -0.15,  # 15% liquidity drain
        'liquidity_velocity': -0.05,  # Moderate velocity
        'liquidity_persistence_bars': 5,  # 5 bars of stress
        'funding_Z': -0.6,  # Funding reversal
        'rsi_14': 28,  # Oversold
        'atr_percentile': 0.75,  # High volatility

        # DOMAIN ENGINE: Strong Wyckoff Spring signal
        'wyckoff_spring_a': True,  # 2.5x boost
        'wyckoff_phase_abc': 'accumulation',
        'wyckoff_sc': False,
        'wyckoff_lps': False,
        'wyckoff_distribution': False,
        'wyckoff_utad': False,
        'wyckoff_bc': False,

        # Other required features
        'close': 50000,
        'liquidity_score': 0.20,  # Low liquidity (capitulation)
        'volume_zscore': 2.5,
        'regime': 'risk_off',

        # Ensure bar has high/low for wick calc
        'high': 51000,
        'low': 49000,
        'open': 50500
    })


def create_test_row_passing_signal():
    """
    Create a test row that ALREADY passes without domain boost.
    Expected: Domain boost increases confidence but doesn't change decision.
    """
    return pd.Series({
        # S1 V2 features (strong capitulation)
        'capitulation_depth_pct': -0.35,  # 35% drawdown
        'crisis_composite': 0.80,  # Strong crisis
        'volume_climax_3bar': 0.55,  # Strong volume spike
        'wick_exhaustion_3bar': 0.45,  # Strong wick exhaustion
        'liquidity_drain_pct': -0.30,  # 30% liquidity drain
        'liquidity_velocity': -0.08,  # Strong velocity
        'liquidity_persistence_bars': 8,  # 8 bars of stress
        'funding_Z': -1.2,  # Strong funding reversal
        'rsi_14': 22,  # Deep oversold
        'atr_percentile': 0.85,  # Very high volatility

        # DOMAIN ENGINE: Strong Wyckoff Spring signal
        'wyckoff_spring_a': True,  # 2.5x boost
        'wyckoff_phase_abc': 'accumulation',

        # Other required features
        'close': 48000,
        'liquidity_score': 0.15,  # Very low liquidity
        'volume_zscore': 3.5,
        'regime': 'crisis'
    })


def run_test():
    """Run test to demonstrate the gate ordering bug."""

    logger.info("=" * 80)
    logger.info("DOMAIN ENGINE GATE ORDERING BUG TEST")
    logger.info("=" * 80)

    # Create S1-only config with domain engines enabled
    config = {
        'archetypes': {
            'enabled': ['S1'],  # Only S1 liquidity vacuum
            'thresholds': {
                'liquidity_vacuum': {
                    'fusion_threshold': 0.40,  # Set threshold that marginal signal will miss
                    'liquidity_max': 0.30,  # Allow low liquidity
                    'capitulation_depth_max': -0.15,  # Allow moderate drawdowns
                    'crisis_composite_min': 0.60,  # Allow moderate crisis
                    'volume_climax_3bar_min': 0.30,  # Lower threshold
                    'wick_exhaustion_3bar_min': 0.20,  # Lower threshold
                }
            }
        },
        'feature_flags': {
            'enable_wyckoff': True,  # Enable Wyckoff domain engine
            'enable_smc': False,
            'enable_temporal': False,
            'enable_hob': False,
            'enable_macro': False
        }
    }

    # Initialize archetype logic
    archetype_logic = ArchetypeLogic(config)

    # Test 1: Marginal signal (should show [ARCH_BUG] warning)
    logger.info("\n" + "=" * 80)
    logger.info("TEST 1: MARGINAL SIGNAL (score ~0.38, threshold 0.40)")
    logger.info("Expected: [ARCH_BUG] warning - domain boost would save it but can't")
    logger.info("=" * 80)

    row_marginal = create_test_row_marginal_signal()
    df_marginal = pd.DataFrame([row_marginal])

    context_marginal = RuntimeContext(
        ts=pd.Timestamp('2022-01-01'),
        row=row_marginal,
        regime_probs={'risk_off': 0.8, 'neutral': 0.15, 'risk_on': 0.05},
        regime_label='risk_off',
        adapted_params={},
        thresholds=config['archetypes']['thresholds'],
        metadata={
            'feature_flags': config['feature_flags'],
            'df': df_marginal,
            'index': 0
        }
    )

    result_marginal = archetype_logic._check_S1(context_marginal)
    matched_marginal, score_marginal, meta_marginal = result_marginal

    logger.info(f"\nRESULT: matched={matched_marginal}, score={score_marginal:.3f}")
    logger.info(f"Metadata: {meta_marginal}")

    # Test 2: Strong signal (already passes, no warning expected)
    logger.info("\n" + "=" * 80)
    logger.info("TEST 2: STRONG SIGNAL (score ~0.55, threshold 0.40)")
    logger.info("Expected: NO warning - signal already passes, domain boost just increases confidence")
    logger.info("=" * 80)

    row_strong = create_test_row_passing_signal()
    df_strong = pd.DataFrame([row_strong])

    context_strong = RuntimeContext(
        ts=pd.Timestamp('2022-06-01'),
        row=row_strong,
        regime_probs={'crisis': 0.9, 'risk_off': 0.08, 'neutral': 0.02},
        regime_label='crisis',
        adapted_params={},
        thresholds=config['archetypes']['thresholds'],
        metadata={
            'feature_flags': config['feature_flags'],
            'df': df_strong,
            'index': 0
        }
    )

    result_strong = archetype_logic._check_S1(context_strong)
    matched_strong, score_strong, meta_strong = result_strong

    logger.info(f"\nRESULT: matched={matched_strong}, score={score_strong:.3f}")
    logger.info(f"Metadata: {meta_strong}")

    # Summary
    logger.info("\n" + "=" * 80)
    logger.info("TEST SUMMARY")
    logger.info("=" * 80)
    logger.info(f"Test 1 (Marginal): matched={matched_marginal}, score={score_marginal:.3f}")
    if 'debug_would_pass_with_boost' in meta_marginal:
        logger.info(f"  Would pass with boost: {meta_marginal['debug_would_pass_with_boost']}")
        logger.info(f"  Domain boost: {meta_marginal.get('debug_domain_boost', 1.0):.2f}x")
        logger.info(f"  Boosted score: {meta_marginal.get('debug_boosted_score', 0.0):.3f}")

    logger.info(f"\nTest 2 (Strong): matched={matched_strong}, score={score_strong:.3f}")

    if matched_marginal:
        logger.error("\n⚠️  TEST FAILED: Marginal signal should have been rejected!")
    else:
        if meta_marginal.get('debug_would_pass_with_boost', False):
            logger.info("\n✅ TEST PASSED: Bug confirmed - domain boost would have saved marginal signal")
        else:
            logger.info("\n⚠️  TEST INCONCLUSIVE: Marginal signal rejected but domain boost wouldn't help")

    if matched_strong:
        logger.info("✅ Strong signal passed as expected")
    else:
        logger.error("⚠️  TEST FAILED: Strong signal should have passed!")

    logger.info("\n" + "=" * 80)
    logger.info("RECOMMENDATION:")
    logger.info("Move domain engine calculations BEFORE fusion threshold gate")
    logger.info("This allows domain context to influence pattern detection, not just scoring")
    logger.info("=" * 80)


if __name__ == '__main__':
    run_test()
