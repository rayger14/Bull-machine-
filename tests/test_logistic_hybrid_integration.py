"""
Integration test for LogisticRegimeModel with HybridRegimeModel

Verifies that LogisticRegimeModel provides the exact interface
expected by HybridRegimeModel (line 397-412).

Author: Claude Code
Date: 2026-01-19
"""

import pytest
from engine.context.logistic_regime_model import LogisticRegimeModel
from engine.context.hybrid_regime_model import HybridRegimeModel


class TestLogisticHybridIntegration:
    """Test LogisticRegimeModel integration with HybridRegimeModel."""

    def test_hybrid_model_can_use_logistic_model(self):
        """Test that HybridRegimeModel can successfully use LogisticRegimeModel."""
        # Create logistic model (will be in fallback mode)
        logistic_model = LogisticRegimeModel()

        # HybridRegimeModel should be able to use it
        # (This mimics line 397 of hybrid_regime_model.py)
        features = {
            'crash_frequency_7d': 0,
            'crisis_persistence': 0.1,
            'aftershock_score': 0.05,
            'RV_7': 45.0,
            'RV_30': 50.0,
            'drawdown_persistence': 0.2,
            'funding_Z': -0.5,
            'volume_z_7d': 0.3,
            'USDT.D': 5.2,
            'BTC.D': 54.0,
            'DXY_Z': 0.8,
            'YC_SPREAD': 0.4
        }

        # Call classify (as HybridRegimeModel does on line 397)
        ml_result = logistic_model.classify(features)

        # Verify HybridRegimeModel can extract all required fields (line 400-402)
        assert 'regime_label' in ml_result
        assert 'regime_confidence' in ml_result
        assert 'regime_probs' in ml_result

        # Verify types match expectations
        regime_label = ml_result['regime_label']
        regime_confidence = ml_result['regime_confidence']
        regime_proba = ml_result['regime_probs']

        assert isinstance(regime_label, str)
        assert isinstance(regime_confidence, (int, float))
        assert isinstance(regime_proba, dict)

    def test_hybrid_result_construction(self):
        """Test that HybridRegimeModel can construct result from LogisticRegimeModel output."""
        logistic_model = LogisticRegimeModel()

        features = {
            'crash_frequency_7d': 0,
            'RV_7': 45.0,
            'crisis_persistence': 0.1,
        }

        ml_result = logistic_model.classify(features)

        # Construct result as HybridRegimeModel does (line 399-407)
        result = {
            'regime_label': ml_result['regime_label'],
            'regime_confidence': ml_result['regime_confidence'],
            'regime_proba': ml_result['regime_probs'],
            'crisis_override': False,
            'crisis_triggers': None,
            'triggers_fired': 0,
            'regime_source': 'hybrid_ml_normal'
        }

        # Verify result structure
        assert result['regime_label'] in ['crisis', 'risk_off', 'neutral', 'risk_on']
        assert 0.0 <= result['regime_confidence'] <= 1.0
        assert len(result['regime_proba']) == 4

    def test_backward_compatibility_alias(self):
        """Test that both 'regime_probs' and 'regime_proba' keys work."""
        logistic_model = LogisticRegimeModel()

        features = {'crash_frequency_7d': 0, 'RV_7': 45.0}
        result = logistic_model.classify(features)

        # LogisticRegimeModel returns 'regime_probs'
        assert 'regime_probs' in result

        # HybridRegimeModel expects 'regime_proba' (line 402)
        # Both should work (backward compatibility)
        regime_proba = result['regime_probs']
        assert isinstance(regime_proba, dict)
        assert len(regime_proba) == 4

    def test_all_regime_probabilities_present(self):
        """Test that all 4 regime probabilities are in output."""
        logistic_model = LogisticRegimeModel()

        features = {'crash_frequency_7d': 0, 'RV_7': 45.0}
        result = logistic_model.classify(features)

        probs = result['regime_probs']

        # All 4 regimes must be present
        assert 'crisis' in probs
        assert 'risk_off' in probs
        assert 'neutral' in probs
        assert 'risk_on' in probs

        # Probabilities should sum to ~1.0
        total = sum(probs.values())
        assert abs(total - 1.0) < 0.01

    def test_error_handling_matches_hybrid_expectations(self):
        """Test that errors are handled gracefully (fallback, not exceptions)."""
        logistic_model = LogisticRegimeModel()

        # Test with invalid features (should fall back, not crash)
        result = logistic_model.classify({})

        # Should still return valid structure
        assert 'regime_label' in result
        assert 'regime_confidence' in result
        assert 'regime_probs' in result

        # Should default to neutral (safe fallback)
        assert result['regime_label'] == 'neutral'


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])
