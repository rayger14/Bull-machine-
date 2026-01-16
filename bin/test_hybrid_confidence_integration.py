#!/usr/bin/env python3
"""
Test hybrid confidence integration in RegimeService.

Tests:
1. Load RegimeService with calibration enabled
2. Make single prediction
3. Verify both ensemble_agreement (raw) and regime_stability_forecast (calibrated) are returned
4. Check calibrated value is different from raw
5. Test with calibration disabled
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
import pandas as pd
from engine.context.regime_service import RegimeService
from datetime import datetime

def test_hybrid_confidence():
    """Test hybrid confidence metrics integration."""

    print("=" * 80)
    print("TEST: Hybrid Confidence Integration")
    print("=" * 80)

    # Test 1: Load RegimeService with calibration enabled
    print("\n[Test 1] Loading RegimeService with calibration ENABLED...")

    regime_service = RegimeService(
        mode='dynamic_ensemble',
        model_path="models/ensemble_regime_v1.pkl",
        enable_calibration=True,
        calibrator_path="models/confidence_calibrator_v1.pkl"
    )

    print(f"✓ RegimeService loaded")
    print(f"  - Calibration enabled: {regime_service.enable_calibration}")
    print(f"  - Calibrator loaded: {regime_service.confidence_calibrator is not None}")

    # Test 2: Make a prediction with sample features
    print("\n[Test 2] Making prediction with sample features...")

    # Create sample features (realistic values)
    sample_features = {
        'btc_return_1h': 0.002,
        'btc_return_24h': 0.015,
        'btc_volatility_24h': 0.025,
        'funding_rate': 0.0001,
        'funding_rate_ema': 0.00008,
        'oi_change_1h': 0.01,
        'liquidation_ratio': 0.02,
        'volume_imbalance': 0.1,
        'price_momentum': 0.5,
        'market_breadth': 0.6,
    }

    result = regime_service.get_regime(sample_features)

    print(f"✓ Prediction made")
    print(f"  - Regime: {result['regime_label']}")
    print(f"  - Risk score: {result['risk_score']:.4f}")

    # Test 3: Verify both metrics are returned
    print("\n[Test 3] Verifying hybrid metrics...")

    assert 'ensemble_agreement' in result, "Missing ensemble_agreement (raw metric)"
    assert 'regime_stability_forecast' in result, "Missing regime_stability_forecast (calibrated metric)"

    raw_agreement = result['ensemble_agreement']
    calibrated_forecast = result['regime_stability_forecast']

    print(f"✓ Both metrics present:")
    print(f"  - ensemble_agreement (raw):          {raw_agreement:.4f}")
    print(f"  - regime_stability_forecast (calib): {calibrated_forecast:.4f}")

    # Test 4: Check calibration effect
    print("\n[Test 4] Checking calibration effect...")

    assert calibrated_forecast is not None, "Calibrated forecast is None (should have value)"
    assert calibrated_forecast != raw_agreement, "Calibrated forecast equals raw (should be different)"

    delta = calibrated_forecast - raw_agreement
    delta_pct = (delta / raw_agreement) * 100 if raw_agreement > 0 else 0

    print(f"✓ Calibration applied:")
    print(f"  - Delta: {delta:+.4f} ({delta_pct:+.2f}%)")
    print(f"  - Expected: Calibrated > Raw (shifts toward stability base rate)")

    # Test 5: Load with calibration disabled
    print("\n[Test 5] Loading RegimeService with calibration DISABLED...")

    regime_service_no_cal = RegimeService(
        mode='dynamic_ensemble',
        model_path="models/ensemble_regime_v1.pkl",
        enable_calibration=False
    )

    result_no_cal = regime_service_no_cal.get_regime(sample_features)

    print(f"✓ Prediction without calibration:")
    print(f"  - ensemble_agreement: {result_no_cal['ensemble_agreement']:.4f}")
    print(f"  - regime_stability_forecast: {result_no_cal['regime_stability_forecast']}")

    assert result_no_cal['regime_stability_forecast'] is None, "Forecast should be None when calibration disabled"

    # Test 6: Backward compatibility check
    print("\n[Test 6] Backward compatibility check...")

    # regime_confidence should still exist (for backward compatibility)
    assert 'regime_confidence' in result, "Missing regime_confidence (backward compatibility)"

    print(f"✓ Backward compatible:")
    print(f"  - regime_confidence: {result['regime_confidence']:.4f} (still exists, computed by hysteresis)")
    print(f"  - ensemble_agreement: {result['ensemble_agreement']:.4f} (raw quality indicator)")
    print(f"  - Note: regime_confidence is probability gap from hysteresis, not raw agreement")

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY: All tests passed ✓")
    print("=" * 80)
    print("\nHybrid Confidence Integration:")
    print("  ✓ Calibrator loads successfully from pickle")
    print("  ✓ Both metrics (raw + calibrated) are returned")
    print("  ✓ Calibration applies transformation to raw agreement")
    print("  ✓ Can disable calibration (returns None for forecast)")
    print("  ✓ Backward compatible (regime_confidence still exists)")
    print("\nNext Steps:")
    print("  → Phase 2.3: Validate in streaming + batch backtest")
    print("  → Phase 2.4: Document usage patterns for archetypes")


if __name__ == '__main__':
    test_hybrid_confidence()
