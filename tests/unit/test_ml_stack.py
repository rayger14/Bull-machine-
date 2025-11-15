"""
Unit tests for ML Stack

Tests:
1. Fusion Scorer ML - predictions, thresholds, latency
2. Enhanced Macro Signals - traps, greenlights, bounds
3. Kelly-Lite Sizer - loss decay, regime caps, DD scaling
4. Contract tests - unified delta format
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import unittest
import numpy as np
import pandas as pd
import time
from engine.ml import FusionScorerML, MacroSignalsEnhanced, KellyLiteSizer


class TestFusionScorerML(unittest.TestCase):
    """Test ML Fusion Scorer"""

    def setUp(self):
        """Load trained model"""
        try:
            self.scorer = FusionScorerML.load("models/fusion_scorer_xgb.pkl")
        except:
            self.scorer = FusionScorerML()
            self.scorer.threshold = 0.425

    def test_predict_returns_bounded_score(self):
        """Test predictions are in [0, 1]"""
        domain_scores = {'wyckoff': 0.8, 'smc': 0.7, 'hob': 0.75, 'momentum': 0.65, 'temporal': 0.5}
        macro_features = {'rv_20d': 40.0, 'rv_60d': 45.0, 'vix_proxy': 20.0, 'regime': 'neutral'}
        market_features = {'adx': 25.0, 'atr_normalized': 0.02, 'volume_ratio': 1.0}

        score = self.scorer.predict_fusion_score(domain_scores, macro_features, market_features)

        self.assertGreaterEqual(score, 0.0, "Score should be >= 0")
        self.assertLessEqual(score, 1.0, "Score should be <= 1")

    def test_threshold_binary_decision(self):
        """Test thresholding returns boolean"""
        domain_scores = {'wyckoff': 0.9, 'smc': 0.85, 'hob': 0.8, 'momentum': 0.75, 'temporal': 0.7}
        macro_features = {'rv_20d': 35.0, 'rv_60d': 40.0, 'vix_proxy': 15.0, 'regime': 'risk_on'}
        market_features = {'adx': 30.0, 'atr_normalized': 0.025, 'volume_ratio': 1.2}

        should_enter, score = self.scorer.should_enter(domain_scores, macro_features, market_features)

        self.assertIsInstance(should_enter, bool, "Decision should be boolean")
        self.assertIsInstance(score, float, "Score should be float")

    def test_latency_under_5ms(self):
        """Test prediction latency < 5ms per sample"""
        if self.scorer.model is None:
            self.skipTest("Model not trained")

        domain_scores = {'wyckoff': 0.7, 'smc': 0.6, 'hob': 0.65, 'momentum': 0.7, 'temporal': 0.5}
        macro_features = {'rv_20d': 45.0, 'rv_60d': 50.0, 'vix_proxy': 22.0, 'regime': 'neutral'}
        market_features = {'adx': 20.0, 'atr_normalized': 0.03, 'volume_ratio': 0.9}

        # Warmup
        for _ in range(10):
            self.scorer.predict_fusion_score(domain_scores, macro_features, market_features)

        # Measure
        start = time.time()
        n_samples = 100
        for _ in range(n_samples):
            self.scorer.predict_fusion_score(domain_scores, macro_features, market_features)
        elapsed = time.time() - start

        avg_latency_ms = (elapsed / n_samples) * 1000
        self.assertLess(avg_latency_ms, 5.0, f"Latency {avg_latency_ms:.2f}ms exceeds 5ms")


class TestMacroSignalsEnhanced(unittest.TestCase):
    """Test Enhanced Macro Signals"""

    def setUp(self):
        """Initialize engine"""
        self.engine = MacroSignalsEnhanced()

    def test_funding_trap_suppresses(self):
        """Test funding trap suppresses entries"""
        macro = {
            'funding': 0.015,  # Elevated
            'oi': 0.020,       # Elevated
            'VIX': 25.0,
            'DXY': 103.0,
            'YIELD_2Y': 4.3,
            'YIELD_10Y': 4.5,
            'BTC.D': 55.0,
            'TOTAL': 100.0,
            'TOTAL2': 40.0
        }

        result = self.engine.analyze_macro_conditions(macro)

        self.assertTrue(result['suppress'], "Funding trap should suppress")
        self.assertIn('FUNDING TRAP', ' '.join(result['notes']))

    def test_dxy_vix_double_trap(self):
        """Test DXY + VIX double trap"""
        macro = {
            'DXY': 106.0,  # > 105
            'VIX': 32.0,   # > 30
            'funding': 0.005,
            'oi': 0.012,
            'YIELD_2Y': 4.3,
            'YIELD_10Y': 4.5,
            'BTC.D': 55.0,
            'TOTAL': 100.0,
            'TOTAL2': 40.0
        }

        result = self.engine.analyze_macro_conditions(macro)

        self.assertTrue(result['suppress'], "DXY+VIX trap should suppress")
        self.assertLessEqual(result['risk_multiplier'], 0.5, "Risk should be capped at 0.5x")

    def test_greenlight_boost(self):
        """Test greenlight conditions boost"""
        macro = {
            'VIX': 12.0,   # Low fear
            'DXY': 96.0,   # Weak dollar
            'funding': 0.003,  # Neutral
            'oi': 0.012,
            'YIELD_2Y': 4.0,
            'YIELD_10Y': 4.8,  # Positive spread
            'BTC.D': 56.0,
            'TOTAL': 100.0,
            'TOTAL2': 40.0
        }

        greenlight = self.engine.get_greenlight_boost(macro)

        self.assertLess(greenlight['threshold_delta'], 0, "Greenlight should lower threshold")
        self.assertGreater(greenlight['risk_mult'], 1.0, "Greenlight should boost risk")

    def test_bounds_enforced(self):
        """Test all adjustments are within bounds"""
        # Extreme macro to test clamping
        macro = {
            'VIX': 50.0,
            'DXY': 120.0,
            'funding': 0.05,
            'oi': 0.10,
            'YIELD_2Y': 6.0,
            'YIELD_10Y': 3.0,  # Deep inversion
            'BTC.D': 40.0,
            'TOTAL': 100.0,
            'TOTAL2': 50.0
        }

        result = self.engine.analyze_macro_conditions(macro)

        self.assertGreaterEqual(result['enter_threshold_delta'], -0.10, "Threshold delta >= -0.10")
        self.assertLessEqual(result['enter_threshold_delta'], 0.10, "Threshold delta <= +0.10")
        self.assertGreaterEqual(result['risk_multiplier'], 0.0, "Risk mult >= 0.0")
        self.assertLessEqual(result['risk_multiplier'], 1.5, "Risk mult <= 1.5")

    def test_altseason_signal(self):
        """Test TOTAL2 divergence altseason boost"""
        macro = {
            'VIX': 18.0,
            'DXY': 102.0,
            'funding': 0.005,
            'oi': 0.012,
            'YIELD_2Y': 4.2,
            'YIELD_10Y': 4.5,
            'BTC.D': 53.0,      # < 55%
            'TOTAL': 100.0,
            'TOTAL2': 42.0      # 42/100 = 0.42 > 0.405
        }

        result = self.engine.analyze_macro_conditions(macro)

        self.assertLess(result['enter_threshold_delta'], 0, "Altseason should boost (lower threshold)")
        self.assertIn('ALTSEASON', ' '.join(result['notes']))


class TestKellyLiteSizer(unittest.TestCase):
    """Test Kelly-Lite Risk Sizer"""

    def setUp(self):
        """Initialize sizer"""
        self.sizer = KellyLiteSizer(base_risk_pct=0.0075)

    def test_loss_streak_decay(self):
        """Test loss streak applies decay"""
        risk_0_losses = self.sizer._apply_guardrails(0.015, 'neutral', 0, 0.0)
        risk_2_losses = self.sizer._apply_guardrails(0.015, 'neutral', 2, 0.0)
        risk_3_losses = self.sizer._apply_guardrails(0.015, 'neutral', 3, 0.0)

        # Should decay by 0.7^(n-1)
        expected_2 = risk_0_losses * 0.7
        expected_3 = risk_0_losses * 0.7 * 0.7

        self.assertAlmostEqual(risk_2_losses, expected_2, places=4, msg="2 losses should decay by 0.7")
        self.assertAlmostEqual(risk_3_losses, expected_3, places=4, msg="3 losses should decay by 0.7^2")

    def test_regime_caps(self):
        """Test regime caps are enforced"""
        # Risk-off: max 1%
        risk_off = self.sizer._apply_guardrails(0.02, 'risk_off', 0, 0.0)
        self.assertLessEqual(risk_off, 0.01, "Risk-off should cap at 1%")

        # Crisis: max 0.5%
        crisis = self.sizer._apply_guardrails(0.02, 'crisis', 0, 0.0)
        self.assertLessEqual(crisis, 0.005, "Crisis should cap at 0.5%")

    def test_drawdown_scaling(self):
        """Test drawdown > 10% scales risk down"""
        risk_no_dd = self.sizer._apply_guardrails(0.015, 'neutral', 0, 0.0)
        risk_with_dd = self.sizer._apply_guardrails(0.015, 'neutral', 0, -0.15)  # -15% DD

        self.assertLess(risk_with_dd, risk_no_dd, "Drawdown should reduce risk")

    def test_hard_clamp_0_to_2pct(self):
        """Test risk is hard clamped to [0, 2%]"""
        risk_negative = self.sizer._apply_guardrails(-0.01, 'neutral', 0, 0.0)
        risk_excessive = self.sizer._apply_guardrails(0.05, 'neutral', 0, 0.0)

        self.assertGreaterEqual(risk_negative, 0.0, "Risk should be >= 0")
        self.assertLessEqual(risk_excessive, 0.02, "Risk should be <= 2%")


class TestContractUnifiedDelta(unittest.TestCase):
    """Test all ML modules return unified delta format"""

    def test_macro_signals_contract(self):
        """Test macro signals returns correct format"""
        engine = MacroSignalsEnhanced()
        macro = {
            'VIX': 20.0, 'DXY': 100.0, 'funding': 0.005, 'oi': 0.012,
            'YIELD_2Y': 4.0, 'YIELD_10Y': 4.3, 'BTC.D': 55.0,
            'TOTAL': 100.0, 'TOTAL2': 40.0
        }

        result = engine.analyze_macro_conditions(macro)

        # Required keys
        self.assertIn('enter_threshold_delta', result)
        self.assertIn('risk_multiplier', result)
        self.assertIn('weight_nudges', result)
        self.assertIn('suppress', result)
        self.assertIn('notes', result)

        # Types
        self.assertIsInstance(result['enter_threshold_delta'], float)
        self.assertIsInstance(result['risk_multiplier'], float)
        self.assertIsInstance(result['weight_nudges'], dict)
        self.assertIsInstance(result['suppress'], bool)
        self.assertIsInstance(result['notes'], list)

    def test_all_modules_bounded(self):
        """Test all modules respect bounds"""
        # Macro
        engine = MacroSignalsEnhanced()
        macro = {'VIX': 20.0, 'DXY': 100.0, 'funding': 0.005, 'oi': 0.012,
                 'YIELD_2Y': 4.0, 'YIELD_10Y': 4.3, 'BTC.D': 55.0,
                 'TOTAL': 100.0, 'TOTAL2': 40.0}
        macro_result = engine.analyze_macro_conditions(macro)

        self.assertGreaterEqual(macro_result['enter_threshold_delta'], -0.10)
        self.assertLessEqual(macro_result['enter_threshold_delta'], 0.10)
        self.assertGreaterEqual(macro_result['risk_multiplier'], 0.0)
        self.assertLessEqual(macro_result['risk_multiplier'], 1.5)

        # Kelly
        sizer = KellyLiteSizer()
        risk = sizer._apply_guardrails(0.025, 'neutral', 0, 0.0)
        self.assertGreaterEqual(risk, 0.0)
        self.assertLessEqual(risk, 0.02)


def run_tests():
    """Run all tests"""
    print("=" * 70)
    print("ML STACK UNIT TESTS")
    print("=" * 70)

    # Set seeds for reproducibility
    np.random.seed(42)

    # Create test suite
    suite = unittest.TestLoader().loadTestsFromModule(sys.modules[__name__])

    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    # Summary
    print("\n" + "=" * 70)
    if result.wasSuccessful():
        print("✅ ALL TESTS PASSED")
    else:
        print(f"❌ {len(result.failures)} FAILURES, {len(result.errors)} ERRORS")
    print("=" * 70)

    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    sys.exit(run_tests())
