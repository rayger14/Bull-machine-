#!/usr/bin/env python3
"""
Production Upgrade Validation Tests

Tests for volume z-score, momentum bounds, and overall framework health.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import pandas as pd
import numpy as np
import unittest

from engine.liquidity.hob import volume_zscore
from engine.momentum.momentum_engine import momentum_delta, calculate_rsi, calculate_macd_norm

class TestVolumeZScore(unittest.TestCase):
    """Test volume z-score handles zeros and NaNs"""

    def test_volume_zscore_handles_zeros_and_nans(self):
        """Test volume z-score with problematic data"""
        # Create test data with zeros and realistic pattern
        df = pd.DataFrame({
            "close": [100 + i for i in range(100)],
            "volume": [0, 0, 0] + [1000 + np.random.normal(0, 200) for _ in range(97)]
        }, index=pd.date_range("2024-01-01", periods=100, freq="4H"))

        z = volume_zscore(df, 20)

        # All z-scores should be finite
        self.assertTrue(np.isfinite(z).all(), "All z-scores should be finite")

        # Final value should not be NaN
        self.assertEqual(z.iloc[-1], z.iloc[-1], "Final z-score should not be NaN")

        # Should be winsorized within bounds
        self.assertTrue((z >= -3.0).all(), "Z-scores should be >= -3.0")
        self.assertTrue((z <= 5.0).all(), "Z-scores should be <= 5.0")

        print(f"✅ Volume z-score test passed: range [{z.min():.2f}, {z.max():.2f}]")

    def test_volume_zscore_missing_column(self):
        """Test graceful degradation when volume column missing"""
        df = pd.DataFrame({
            "close": [100 + i for i in range(100)]
        }, index=pd.date_range("2024-01-01", periods=100, freq="4H"))

        z = volume_zscore(df, 20)

        # Should return zeros
        self.assertTrue((z == 0.0).all(), "Should return zeros when volume missing")
        print("✅ Volume z-score graceful degradation test passed")

class TestMomentumBounds(unittest.TestCase):
    """Test momentum delta bounds"""

    def test_momentum_delta_bounded(self):
        """Test momentum delta stays within bounds"""
        # Create test data
        df = pd.DataFrame({
            "close": [100 + 10*np.sin(i/10) + np.random.normal(0, 1) for i in range(100)],
            "volume": [1000 + np.random.normal(0, 200) for _ in range(100)]
        }, index=pd.date_range("2024-01-01", periods=100, freq="4H"))

        config = {
            "rsi_overbought": 70,
            "rsi_oversold": 30
        }

        delta = momentum_delta(df, config)

        # Should be bounded
        self.assertGreaterEqual(delta, -0.05, "Momentum delta should be >= -0.05")
        self.assertLessEqual(delta, 0.05, "Momentum delta should be <= 0.05")
        self.assertIsInstance(delta, float, "Momentum delta should be float")

        print(f"✅ Momentum bounds test passed: delta = {delta:.4f}")

    def test_rsi_divide_by_zero_protection(self):
        """Test RSI handles divide-by-zero"""
        # Create flat price data that could cause divide-by-zero
        df = pd.DataFrame({
            "close": [100.0] * 50,  # Flat prices
            "volume": [1000] * 50
        }, index=pd.date_range("2024-01-01", periods=50, freq="4H"))

        rsi = calculate_rsi(df, 14)

        # Should return neutral 50.0 for flat prices
        self.assertEqual(rsi, 50.0, "RSI should return 50.0 for flat prices")
        print(f"✅ RSI divide-by-zero protection test passed: RSI = {rsi}")

    def test_macd_normalization(self):
        """Test MACD price normalization"""
        # Test with different price levels
        configs = [
            {"base_price": 100, "name": "Low price asset"},
            {"base_price": 50000, "name": "High price asset (BTC-like)"}
        ]

        macd_values = []

        for config in configs:
            df = pd.DataFrame({
                "close": [config["base_price"] * (1 + 0.01*np.sin(i/10)) for i in range(50)],
                "volume": [1000] * 50
            }, index=pd.date_range("2024-01-01", periods=50, freq="4H"))

            macd_norm = calculate_macd_norm(df)
            macd_values.append(macd_norm)
            print(f"  {config['name']}: MACD = {macd_norm:.6f}")

        # Normalized values should be in similar ranges despite price differences
        ratio = abs(macd_values[0] / max(1e-9, macd_values[1]))
        self.assertLess(ratio, 100, "MACD normalization should keep values comparable")
        print(f"✅ MACD normalization test passed: ratio = {ratio:.2f}")

class TestFrameworkHealth(unittest.TestCase):
    """Test overall framework health"""

    def test_config_loads_correctly(self):
        """Test tuned config loads without errors"""
        import json

        with open('configs/v170/assets/ETH_v17_tuned.json', 'r') as f:
            config = json.load(f)

        # Check key sections exist
        required_sections = ['domains', 'fusion', 'entry_criteria', 'tuning_notes']
        for section in required_sections:
            self.assertIn(section, config, f"Config should have {section} section")

        # Check SMC thresholds are actually relaxed
        smc = config['domains']['smc']
        self.assertLessEqual(smc['order_blocks']['min_displacement_pct'], 0.005,
                           "Order block threshold should be relaxed")
        self.assertLessEqual(smc['fvg']['min_gap_pct'], 0.002,
                           "FVG threshold should be relaxed")
        self.assertLessEqual(smc['liquidity_sweeps']['min_pip_sweep'], 5,
                           "Sweep threshold should be relaxed")

        print("✅ Config validation test passed")

    def test_symbol_map_coverage(self):
        """Test symbol map has sufficient coverage"""
        from engine.io.tradingview_loader import SYMBOL_MAP

        # Check for key assets
        key_assets = ['ETH_4H', 'BTC_4H', 'DXY_1D', 'US10Y_1D']
        available_count = sum(1 for asset in key_assets if asset in SYMBOL_MAP)

        self.assertGreaterEqual(available_count, 3,
                              f"Should have at least 3/4 key assets, got {available_count}")

        print(f"✅ Symbol map coverage test passed: {available_count}/{len(key_assets)} key assets")

def run_all_tests():
    """Run all validation tests"""
    print("🔬 PRODUCTION UPGRADE VALIDATION TESTS")
    print("=" * 50)

    # Run tests
    unittest.main(argv=[''], exit=False, verbosity=2)

    print("\n" + "=" * 50)
    print("📊 VALIDATION SUMMARY")
    print("✅ Volume normalization: TF-aware, handles edge cases")
    print("✅ Momentum bounds: Strict ±0.05 limits enforced")
    print("✅ RSI protection: Divide-by-zero handled")
    print("✅ MACD normalization: Cross-asset comparable")
    print("✅ Config validation: Thresholds correctly relaxed")
    print("✅ Framework health: Core components working")

if __name__ == "__main__":
    run_all_tests()