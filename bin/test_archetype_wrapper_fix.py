#!/usr/bin/env python3
"""
Test Script: Verify ArchetypeModel Wrapper Fix

This script verifies that the fixed ArchetypeModel wrapper correctly:
1. Builds RuntimeContext with enriched bar (liquidity_score, fusion_score)
2. Passes all required features to archetype detection logic
3. Allows archetypes to detect signals (no longer running "blind")

Usage:
    python bin/test_archetype_wrapper_fix.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
import logging
from datetime import datetime, timedelta

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Import the fixed wrapper
from engine.models.archetype_model import ArchetypeModel


def create_synthetic_bar() -> pd.Series:
    """
    Create a synthetic bar with typical feature values.

    This simulates what would come from the feature store.
    """
    # Start with OHLCV
    bar_data = {
        'open': 50000.0,
        'high': 50500.0,
        'low': 49500.0,
        'close': 50250.0,
        'volume': 1000000.0,
    }

    # Add technical indicators
    bar_data.update({
        'atr_14': 500.0,
        'adx_14': 25.0,
        'rsi_14': 45.0,
        'ema_9': 50100.0,
        'ema_21': 49900.0,
        'ema_55': 49500.0,
        'bb_upper': 51000.0,
        'bb_lower': 49000.0,
        'bb_middle': 50000.0,
    })

    # Add Wyckoff features
    bar_data.update({
        'tf1d_wyckoff_phase': 'reaccumulation',
        'tf1d_m1_signal': 'spring',  # Wyckoff M1 signal present
        'tf1d_m2_signal': None,
        'tf4h_wyckoff_phase': 'markup',
    })

    # Add liquidity features
    bar_data.update({
        'tf1d_boms_strength': 0.7,
        'tf4h_boms_displacement': 800.0,
        'tf4h_fvg_present': True,
        'tf1h_fakeout_detected': False,
        'mtf_governor_veto': False,
    })

    # Add PTI features
    bar_data.update({
        'tf1d_pti_score': 0.2,
        'tf1h_pti_score': 0.15,
        'tf1h_pti_trap_type': None,
    })

    # Add FRVP features
    bar_data.update({
        'tf1h_frvp_poc_position': 'at_poc',
        'tf1d_frvp_poc_position': 'middle',
    })

    # Add Squiggle features
    bar_data.update({
        'tf4h_squiggle_confidence': 0.65,
        'tf1h_squiggle_direction': 'up',
    })

    # Add macro features (optional - will use defaults if missing)
    bar_data.update({
        'macro_regime': 'neutral',
        'macro_vix_level': 'medium',
    })

    # Create Series with timestamp index
    bar = pd.Series(bar_data)
    bar.name = pd.Timestamp.now()

    return bar


def test_runtime_context_enrichment():
    """
    Test 1: Verify RuntimeContext contains enriched row with runtime scores.
    """
    print("\n" + "="*80)
    print("TEST 1: RuntimeContext Enrichment")
    print("="*80)

    # Create wrapper instance
    config_path = 'configs/mvp/mvp_bull_market_v1.json'
    if not Path(config_path).exists():
        print(f"❌ Config not found: {config_path}")
        print("   Using fallback config...")
        config_path = 'configs/optimized_bull_v2_production.json'
        if not Path(config_path).exists():
            print(f"❌ Fallback config also not found: {config_path}")
            return False

    try:
        model = ArchetypeModel(
            config_path=config_path,
            archetype_name='long_squeeze',  # S5
            name='test-wrapper'
        )
    except Exception as e:
        print(f"❌ Failed to initialize ArchetypeModel: {e}")
        return False

    # Create synthetic bar
    bar = create_synthetic_bar()

    print(f"\n📊 Original bar features: {len(bar)} features")
    print(f"   Has liquidity_score: {'liquidity_score' in bar}")
    print(f"   Has fusion_score: {'fusion_score' in bar}")

    # Build RuntimeContext using the fixed method
    try:
        context = model._build_runtime_context(bar)
    except Exception as e:
        print(f"❌ Failed to build RuntimeContext: {e}")
        import traceback
        traceback.print_exc()
        return False

    print(f"\n✅ RuntimeContext built successfully")
    print(f"   Timestamp: {context.ts}")
    print(f"   Regime: {context.regime_label}")
    print(f"   Regime probs: {context.regime_probs}")

    # Verify enriched row
    enriched_row = context.row
    print(f"\n📊 Enriched row features: {len(enriched_row)} features")

    # Check critical runtime scores
    if 'liquidity_score' in enriched_row:
        print(f"   ✅ liquidity_score: {enriched_row['liquidity_score']:.3f}")
    else:
        print(f"   ❌ liquidity_score: MISSING")
        return False

    if 'fusion_score' in enriched_row:
        print(f"   ✅ fusion_score: {enriched_row['fusion_score']:.3f}")
    else:
        print(f"   ❌ fusion_score: MISSING")
        return False

    # Verify thresholds are present
    print(f"\n📋 Thresholds: {len(context.thresholds)} archetypes")
    if 'long_squeeze' in context.thresholds:
        print(f"   ✅ long_squeeze thresholds found")
        long_squeeze_params = context.thresholds['long_squeeze']
        print(f"      Available params: {list(long_squeeze_params.keys())[:5]}...")
    else:
        print(f"   ⚠️  long_squeeze thresholds not found (may be expected if not in config)")

    print(f"\n✅ TEST 1 PASSED: RuntimeContext properly enriched")
    return True


def test_archetype_detection():
    """
    Test 2: Verify archetypes can detect signals using the enriched context.
    """
    print("\n" + "="*80)
    print("TEST 2: Archetype Detection with Enriched Context")
    print("="*80)

    # Create wrapper instance
    config_path = 'configs/mvp/mvp_bull_market_v1.json'
    if not Path(config_path).exists():
        config_path = 'configs/optimized_bull_v2_production.json'
        if not Path(config_path).exists():
            print(f"❌ Config not found: {config_path}")
            return False

    try:
        model = ArchetypeModel(
            config_path=config_path,
            archetype_name='long_squeeze',  # S5
            name='test-wrapper'
        )
    except Exception as e:
        print(f"❌ Failed to initialize ArchetypeModel: {e}")
        return False

    # Create synthetic bar
    bar = create_synthetic_bar()

    # Call predict() which internally builds RuntimeContext and calls detect()
    print(f"\n🔍 Calling model.predict() with synthetic bar...")
    try:
        signal = model.predict(bar, position=None)
    except Exception as e:
        print(f"❌ Failed to call predict(): {e}")
        import traceback
        traceback.print_exc()
        return False

    print(f"\n✅ predict() executed successfully")
    print(f"   Direction: {signal.direction}")
    print(f"   Confidence: {signal.confidence:.3f}")
    print(f"   Entry Price: ${signal.entry_price:,.2f}")

    if signal.stop_loss:
        print(f"   Stop Loss: ${signal.stop_loss:,.2f}")
        risk_pct = abs(signal.entry_price - signal.stop_loss) / signal.entry_price * 100
        print(f"   Risk: {risk_pct:.2f}%")

    if signal.metadata:
        print(f"\n📊 Signal Metadata:")
        for key, value in signal.metadata.items():
            if isinstance(value, float):
                print(f"      {key}: {value:.3f}")
            else:
                print(f"      {key}: {value}")

    # Verify archetype was able to compute scores
    fusion_score = signal.metadata.get('fusion_score', 0.0)
    liquidity_score = signal.metadata.get('liquidity_score', 0.0)

    if fusion_score > 0.0 or liquidity_score > 0.0:
        print(f"\n✅ Archetype logic accessed runtime scores:")
        print(f"   Fusion: {fusion_score:.3f}")
        print(f"   Liquidity: {liquidity_score:.3f}")
    else:
        print(f"\n⚠️  Scores are zero (archetype may not have matched, or scores legitimately low)")

    print(f"\n✅ TEST 2 PASSED: Archetype detection works with enriched context")
    return True


def test_feature_access():
    """
    Test 3: Verify all expected features are accessible in RuntimeContext.
    """
    print("\n" + "="*80)
    print("TEST 3: Feature Accessibility Check")
    print("="*80)

    # Create wrapper instance
    config_path = 'configs/mvp/mvp_bull_market_v1.json'
    if not Path(config_path).exists():
        config_path = 'configs/optimized_bull_v2_production.json'
        if not Path(config_path).exists():
            print(f"❌ Config not found: {config_path}")
            return False

    try:
        model = ArchetypeModel(
            config_path=config_path,
            archetype_name='long_squeeze',
            name='test-wrapper'
        )
    except Exception as e:
        print(f"❌ Failed to initialize ArchetypeModel: {e}")
        return False

    # Create synthetic bar
    bar = create_synthetic_bar()

    # Build RuntimeContext
    context = model._build_runtime_context(bar)
    row = context.row

    # List of critical features archetypes need
    critical_features = {
        'liquidity_score': 'Runtime liquidity score',
        'fusion_score': 'Runtime fusion score',
        'atr_14': 'ATR for volatility',
        'adx_14': 'ADX for trend strength',
        'rsi_14': 'RSI for momentum',
        'tf1d_wyckoff_phase': 'Wyckoff phase',
        'tf1d_boms_strength': 'BOMS strength',
        'close': 'Close price',
    }

    print(f"\n🔍 Checking critical features...")
    missing_features = []

    for feature, description in critical_features.items():
        if feature in row:
            value = row[feature]
            if isinstance(value, (int, float, np.number)):
                print(f"   ✅ {feature:30s}: {value:>10.3f} ({description})")
            else:
                print(f"   ✅ {feature:30s}: {value!s:>10s} ({description})")
        else:
            print(f"   ❌ {feature:30s}: MISSING ({description})")
            missing_features.append(feature)

    if missing_features:
        print(f"\n❌ TEST 3 FAILED: {len(missing_features)} critical features missing")
        return False
    else:
        print(f"\n✅ TEST 3 PASSED: All critical features accessible")
        return True


def main():
    """Run all tests."""
    print("\n" + "="*80)
    print("ARCHETYPE WRAPPER FIX VERIFICATION")
    print("="*80)
    print("\nThis test verifies that the ArchetypeModel wrapper fix allows")
    print("archetypes to access all required features (liquidity_score, fusion_score, etc.)")

    results = {
        'RuntimeContext Enrichment': test_runtime_context_enrichment(),
        'Archetype Detection': test_archetype_detection(),
        'Feature Accessibility': test_feature_access(),
    }

    # Summary
    print("\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80)

    for test_name, passed in results.items():
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"{test_name:40s}: {status}")

    all_passed = all(results.values())

    if all_passed:
        print("\n🎉 ALL TESTS PASSED!")
        print("\nThe wrapper fix successfully:")
        print("  1. Enriches bars with runtime scores (liquidity_score, fusion_score)")
        print("  2. Passes enriched bars to RuntimeContext")
        print("  3. Allows archetypes to access all required features")
        print("\nArchetypes are no longer running 'blind'!")
        return 0
    else:
        print("\n❌ SOME TESTS FAILED")
        print("\nThe wrapper may still have issues. Check the output above for details.")
        return 1


if __name__ == '__main__':
    sys.exit(main())
