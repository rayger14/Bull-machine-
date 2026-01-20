"""
Unit tests for LogisticRegimeModel

Tests cover:
1. Initialization and fallback mode
2. Feature extraction and validation
3. Classification interface (single and batch)
4. Model artifact loading
5. HybridRegimeModel compatibility

Author: Claude Code
Date: 2026-01-19
"""

import pytest
import numpy as np
import pandas as pd
from pathlib import Path
import pickle
import tempfile
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import CalibratedClassifierCV

from engine.context.logistic_regime_model import LogisticRegimeModel


class TestLogisticRegimeModelInitialization:
    """Test model initialization and configuration."""

    def test_init_fallback_mode_no_model(self):
        """Test initialization with no trained model (fallback mode)."""
        model = LogisticRegimeModel(model_path="nonexistent_model.pkl")

        assert model.fallback_mode is True
        assert len(model.feature_order) == 12
        assert len(model.regime_labels) == 4
        assert model.regime_labels == ['crisis', 'risk_off', 'neutral', 'risk_on']

    def test_default_feature_order(self):
        """Test that default feature order is correct."""
        model = LogisticRegimeModel()

        expected_features = [
            'crash_frequency_7d',
            'crisis_persistence',
            'aftershock_score',
            'RV_7',
            'RV_30',
            'drawdown_persistence',
            'funding_Z',
            'volume_z_7d',
            'USDT.D',
            'BTC.D',
            'DXY_Z',
            'YC_SPREAD'
        ]

        assert model.feature_order == expected_features

    def test_metadata_retrieval(self):
        """Test get_metadata returns correct structure."""
        model = LogisticRegimeModel()
        metadata = model.get_metadata()

        assert 'fallback_mode' in metadata
        assert 'feature_count' in metadata
        assert 'feature_order' in metadata
        assert 'regime_labels' in metadata
        assert 'use_calibration' in metadata
        assert 'training_metadata' in metadata

        assert metadata['feature_count'] == 12
        assert len(metadata['regime_labels']) == 4


class TestFeatureExtraction:
    """Test feature extraction and validation."""

    def test_extract_features_all_present(self):
        """Test feature extraction when all features present."""
        model = LogisticRegimeModel()

        features = {
            'crash_frequency_7d': 1.0,
            'crisis_persistence': 0.5,
            'aftershock_score': 0.3,
            'RV_7': 45.0,
            'RV_30': 50.0,
            'drawdown_persistence': 0.4,
            'funding_Z': -0.5,
            'volume_z_7d': 0.8,
            'USDT.D': 5.2,
            'BTC.D': 54.0,
            'DXY_Z': 0.7,
            'YC_SPREAD': 0.4
        }

        X = model._extract_features(features)

        assert X.shape == (12,)
        assert X[0] == 1.0  # crash_frequency_7d
        assert X[3] == 45.0  # RV_7
        assert X[11] == 0.4  # YC_SPREAD

    def test_extract_features_missing_values_zerofill(self):
        """Test zero-filling of missing features."""
        model = LogisticRegimeModel()

        features = {
            'crash_frequency_7d': 1.0,
            'RV_7': 45.0,
            # Missing other features
        }

        X = model._extract_features(features)

        assert X.shape == (12,)
        assert X[0] == 1.0  # crash_frequency_7d present
        assert X[1] == 0.0  # crisis_persistence missing
        assert X[3] == 45.0  # RV_7 present
        assert X[4] == 0.0  # RV_30 missing

    def test_extract_features_nan_handling(self):
        """Test NaN values are converted to zero."""
        model = LogisticRegimeModel()

        features = {
            'crash_frequency_7d': np.nan,
            'crisis_persistence': 0.5,
            'RV_7': None,
            'RV_30': float('inf')
        }

        X = model._extract_features(features)

        assert not np.isnan(X).any()
        assert not np.isinf(X).any()
        assert X[0] == 0.0  # NaN -> 0
        assert X[3] == 0.0  # None -> 0
        assert X[4] == 0.0  # inf -> 0

    def test_extract_features_batch(self):
        """Test batch feature extraction."""
        model = LogisticRegimeModel()

        df = pd.DataFrame([
            {'crash_frequency_7d': 0, 'crisis_persistence': 0.1, 'RV_7': 40.0},
            {'crash_frequency_7d': 1, 'crisis_persistence': 0.5, 'RV_7': 60.0},
            {'crash_frequency_7d': 2, 'crisis_persistence': 0.8, 'RV_7': 80.0}
        ])

        X = model._extract_features_batch(df)

        assert X.shape == (3, 12)
        assert X[0, 0] == 0  # First row, crash_frequency_7d
        assert X[1, 0] == 1  # Second row, crash_frequency_7d
        assert X[2, 0] == 2  # Third row, crash_frequency_7d

    def test_validate_features_valid(self):
        """Test feature validation with valid features."""
        model = LogisticRegimeModel()

        features = {
            'crash_frequency_7d': 0,
            'crisis_persistence': 0.1,
            'RV_7': 45.0,
            'RV_30': 50.0,
            'funding_Z': -0.5,
            'BTC.D': 54.0,
            'DXY_Z': 0.8
        }

        assert model._validate_features(features) is True

    def test_validate_features_too_few(self):
        """Test validation fails with too few features."""
        model = LogisticRegimeModel()

        features = {
            'crash_frequency_7d': 0,
            'RV_7': 45.0
        }

        # Only 2 features, need at least 6
        assert model._validate_features(features) is False

    def test_validate_features_wrong_type(self):
        """Test validation fails with wrong input type."""
        model = LogisticRegimeModel()

        assert model._validate_features("not a dict") is False
        assert model._validate_features([1, 2, 3]) is False
        assert model._validate_features(None) is False


class TestClassificationInterface:
    """Test classification methods (single and batch)."""

    def test_classify_fallback_mode(self):
        """Test classify returns neutral in fallback mode."""
        model = LogisticRegimeModel()

        features = {
            'crash_frequency_7d': 0,
            'crisis_persistence': 0.1,
            'RV_7': 45.0
        }

        result = model.classify(features)

        assert 'regime_label' in result
        assert 'regime_confidence' in result
        assert 'regime_probs' in result

        assert result['regime_label'] == 'neutral'
        assert result['regime_confidence'] == 0.50
        assert result['regime_probs']['neutral'] == 0.50

    def test_classify_interface_contract(self):
        """Test classify returns correct dict structure (HybridRegimeModel contract)."""
        model = LogisticRegimeModel()

        features = {'crash_frequency_7d': 0, 'RV_7': 45.0}
        result = model.classify(features)

        # Verify required keys
        assert 'regime_label' in result
        assert 'regime_confidence' in result
        assert 'regime_probs' in result

        # Verify types
        assert isinstance(result['regime_label'], str)
        assert isinstance(result['regime_confidence'], float)
        assert isinstance(result['regime_probs'], dict)

        # Verify regime_probs has all 4 regimes
        assert 'crisis' in result['regime_probs']
        assert 'risk_off' in result['regime_probs']
        assert 'neutral' in result['regime_probs']
        assert 'risk_on' in result['regime_probs']

        # Verify probabilities sum to 1.0
        total_prob = sum(result['regime_probs'].values())
        assert abs(total_prob - 1.0) < 0.01

    def test_classify_batch_fallback_mode(self):
        """Test batch classification in fallback mode."""
        # Force fallback mode by using nonexistent model
        model = LogisticRegimeModel(model_path='nonexistent_model.pkl')

        df = pd.DataFrame([
            {'crash_frequency_7d': 0, 'RV_7': 40.0},
            {'crash_frequency_7d': 1, 'RV_7': 60.0},
            {'crash_frequency_7d': 2, 'RV_7': 80.0}
        ])

        result = model.classify_batch(df)

        assert len(result) == 3
        assert 'regime_label' in result.columns
        assert 'regime_confidence' in result.columns
        assert 'regime_probs' in result.columns

        # All should be neutral in fallback mode
        assert (result['regime_label'] == 'neutral').all()
        assert (result['regime_confidence'] == 0.50).all()

    def test_classify_batch_returns_copy(self):
        """Test that classify_batch doesn't modify original dataframe."""
        model = LogisticRegimeModel()

        df = pd.DataFrame([
            {'crash_frequency_7d': 0, 'RV_7': 40.0},
            {'crash_frequency_7d': 1, 'RV_7': 60.0}
        ])

        original_columns = df.columns.tolist()
        result = model.classify_batch(df)

        # Original should be unchanged
        assert df.columns.tolist() == original_columns

        # Result should have new columns
        assert 'regime_label' in result.columns
        assert result.columns.tolist() != original_columns


class TestModelArtifactLoading:
    """Test loading of trained model artifacts."""

    def test_create_model_artifact(self):
        """Test create_model_artifact helper method."""
        # Create dummy model components
        model = LogisticRegression()
        scaler = StandardScaler()
        feature_order = ['feat1', 'feat2', 'feat3']

        artifact = LogisticRegimeModel.create_model_artifact(
            model=model,
            scaler=scaler,
            feature_order=feature_order,
            training_metadata={'version': 'test_v1'}
        )

        assert 'model' in artifact
        assert 'scaler' in artifact
        assert 'feature_order' in artifact
        assert 'regime_labels' in artifact
        assert 'use_calibration' in artifact
        assert 'training_metadata' in artifact

        assert artifact['feature_order'] == feature_order
        assert artifact['training_metadata']['version'] == 'test_v1'

    def test_load_invalid_pickle_format(self):
        """Test loading with invalid pickle format."""
        with tempfile.NamedTemporaryFile(mode='wb', suffix='.pkl', delete=False) as f:
            # Write invalid artifact (missing required keys)
            pickle.dump({'invalid': 'artifact'}, f)
            temp_path = f.name

        try:
            # Should fall back to neutral mode
            model = LogisticRegimeModel(model_path=temp_path)
            assert model.fallback_mode is True
        finally:
            Path(temp_path).unlink()

    def test_load_model_convenience_function(self):
        """Test load_model convenience function."""
        from engine.context.logistic_regime_model import load_model

        # Should work even with nonexistent path (fallback mode)
        model = load_model("nonexistent.pkl")
        assert isinstance(model, LogisticRegimeModel)
        assert model.fallback_mode is True


class TestHybridRegimeModelCompatibility:
    """Test compatibility with HybridRegimeModel interface."""

    def test_classify_output_format_matches_hybrid_contract(self):
        """Test that classify output matches HybridRegimeModel expectations."""
        model = LogisticRegimeModel()

        features = {
            'crash_frequency_7d': 0,
            'crisis_persistence': 0.1,
            'RV_7': 45.0
        }

        result = model.classify(features)

        # These are the keys HybridRegimeModel expects (line 397-402)
        assert 'regime_label' in result
        assert 'regime_confidence' in result
        assert 'regime_probs' in result

        # Verify it can be used as in HybridRegimeModel line 400-402
        regime_label = result['regime_label']
        regime_confidence = result['regime_confidence']
        regime_proba = result['regime_probs']

        assert isinstance(regime_label, str)
        assert isinstance(regime_confidence, (int, float))
        assert isinstance(regime_proba, dict)

    def test_regime_labels_match_hybrid_expectations(self):
        """Test regime labels match what HybridRegimeModel expects."""
        model = LogisticRegimeModel()

        # HybridRegimeModel expects these 4 regime labels
        expected_regimes = {'crisis', 'risk_off', 'neutral', 'risk_on'}

        features = {'crash_frequency_7d': 0, 'RV_7': 45.0}
        result = model.classify(features)

        # Check all expected regimes are in probs dict
        assert set(result['regime_probs'].keys()) == expected_regimes

        # Check predicted regime is one of the expected values
        assert result['regime_label'] in expected_regimes


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_empty_features_dict(self):
        """Test with empty features dict."""
        model = LogisticRegimeModel()
        result = model.classify({})

        # Should fall back to neutral
        assert result['regime_label'] == 'neutral'

    def test_all_nan_features(self):
        """Test with all NaN features."""
        # Force fallback mode to get predictable result
        model = LogisticRegimeModel(model_path='nonexistent_model.pkl')
        features = {f: np.nan for f in model.feature_order}

        result = model.classify(features)

        # Should handle gracefully (zero-fill) and return neutral in fallback mode
        assert result['regime_label'] == 'neutral'

    def test_empty_dataframe_batch(self):
        """Test classify_batch with empty dataframe."""
        model = LogisticRegimeModel()
        df = pd.DataFrame()

        result = model.classify_batch(df)

        assert len(result) == 0
        assert 'regime_label' in result.columns

    def test_single_row_batch(self):
        """Test classify_batch with single row."""
        # Force fallback mode for predictable result
        model = LogisticRegimeModel(model_path='nonexistent_model.pkl')
        df = pd.DataFrame([{'crash_frequency_7d': 0, 'RV_7': 45.0}])

        result = model.classify_batch(df)

        assert len(result) == 1
        assert result['regime_label'].iloc[0] == 'neutral'

    def test_feature_importance_fallback_mode_raises(self):
        """Test get_feature_importance raises in fallback mode."""
        # Force fallback mode
        model = LogisticRegimeModel(model_path='nonexistent_model.pkl')
        assert model.fallback_mode is True

        with pytest.raises(ValueError, match="Model not loaded"):
            model.get_feature_importance()


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])
