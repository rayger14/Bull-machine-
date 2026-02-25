"""
Unit Tests for ArchetypeInstance - Proving Archetype Isolation

These tests demonstrate that each archetype calculates its own fusion score
independently, solving the coupling issue where all archetypes shared fusion_total.

Test Coverage:
1. Per-archetype fusion calculation with custom weights
2. Independence: Changing archetype A's weights doesn't affect archetype B
3. Different fusion weights produce different scores
4. Config validation catches invalid parameters
"""

import pytest
import pandas as pd
from engine.archetypes.archetype_instance import ArchetypeInstance, ArchetypeConfig


class TestArchetypeConfig:
    """Test ArchetypeConfig validation and defaults."""

    def test_default_fusion_weights(self):
        """Test that default fusion weights are set if not provided."""
        config = ArchetypeConfig(name='test', direction='long')

        assert config.fusion_weights is not None
        assert sum(config.fusion_weights.values()) == pytest.approx(1.0, abs=0.01)

    def test_custom_fusion_weights(self):
        """Test that custom fusion weights are preserved."""
        custom_weights = {
            'wyckoff': 0.70,
            'liquidity': 0.15,
            'momentum': 0.10,
            'smc': 0.05
        }
        config = ArchetypeConfig(
            name='spring',
            direction='long',
            fusion_weights=custom_weights
        )

        assert config.fusion_weights == custom_weights

    def test_validation_fusion_weights_sum(self):
        """Test that fusion weights must sum to 1.0."""
        config = ArchetypeConfig(
            name='bad_weights',
            direction='long',
            fusion_weights={'wyckoff': 0.5, 'liquidity': 0.3}  # Sums to 0.8, not 1.0
        )

        with pytest.raises(AssertionError, match="Fusion weights sum"):
            config.validate()

    def test_validation_invalid_direction(self):
        """Test that direction must be 'long' or 'short'."""
        config = ArchetypeConfig(
            name='bad_direction',
            direction='neutral'  # Invalid
        )

        with pytest.raises(AssertionError, match="Invalid direction"):
            config.validate()

    def test_validation_threshold_range(self):
        """Test that thresholds must be in valid ranges."""
        # Entry threshold > 1.0
        config = ArchetypeConfig(
            name='bad_threshold',
            direction='long',
            entry_threshold=1.5
        )

        with pytest.raises(AssertionError, match="Entry threshold"):
            config.validate()


class TestArchetypeInstance:
    """Test ArchetypeInstance fusion calculation and isolation."""

    @pytest.fixture
    def sample_features(self):
        """Sample feature dict for testing."""
        return {
            'close': 50000.0,
            'atr_14': 1000.0,
            # Wyckoff features
            'tf1d_wyckoff_m1_signal': True,
            'wyckoff_event_confidence': 0.8,
            # Liquidity features
            'liquidity_score': 0.6,
            'tf1d_boms_strength': 0.7,
            'tf4h_fvg_present': True,
            'tf4h_boms_displacement': 1500.0,
            # Momentum features
            'adx_14': 30.0,
            'rsi_14': 65.0,
            'tf4h_squiggle_confidence': 0.7,
            # SMC features
            'fusion_smc': 0.5,
            # Penalties
            'tf1h_fakeout_detected': False,
            'tf1d_pti_score': 0.0,
            'tf1h_pti_score': 0.0
        }

    def test_fusion_calculation_wyckoff_heavy(self, sample_features):
        """Test fusion calculation for Wyckoff-heavy archetype (e.g., spring)."""
        config = ArchetypeConfig(
            name='spring',
            direction='long',
            fusion_weights={
                'wyckoff': 0.70,  # Heavy Wyckoff emphasis
                'liquidity': 0.15,
                'momentum': 0.10,
                'smc': 0.05
            },
            entry_threshold=0.35
        )

        archetype = ArchetypeInstance(config)
        fusion = archetype.compute_fusion_score(sample_features)

        # Expected: High fusion score due to strong Wyckoff signals (1.0 or 0.8)
        # fusion = 0.70 * 1.0 + 0.15 * 0.6 + 0.10 * ~0.6 + 0.05 * 0.5
        #        ≈ 0.70 + 0.09 + 0.06 + 0.025 = 0.875
        assert 0.8 <= fusion <= 1.0
        assert fusion > 0.35  # Above entry threshold

    def test_fusion_calculation_liquidity_heavy(self, sample_features):
        """Test fusion calculation for liquidity-heavy archetype (e.g., wick_trap)."""
        config = ArchetypeConfig(
            name='wick_trap',
            direction='long',
            fusion_weights={
                'wyckoff': 0.20,
                'liquidity': 0.60,  # Heavy liquidity emphasis
                'momentum': 0.15,
                'smc': 0.05
            },
            entry_threshold=0.28
        )

        archetype = ArchetypeInstance(config)
        fusion = archetype.compute_fusion_score(sample_features)

        # Expected: Moderate fusion score, liquidity-driven
        # fusion = 0.20 * 1.0 + 0.60 * 0.6 + 0.15 * ~0.6 + 0.05 * 0.5
        #        ≈ 0.20 + 0.36 + 0.09 + 0.025 = 0.675
        assert 0.6 <= fusion <= 0.8
        assert fusion > 0.28  # Above entry threshold

    def test_isolation_different_weights_different_scores(self, sample_features):
        """
        CRITICAL TEST: Prove that different archetypes with different weights
        produce different fusion scores from the SAME features.

        This proves archetypes are ISOLATED - no shared fusion_total.
        """
        # Spring: Wyckoff-heavy
        spring_config = ArchetypeConfig(
            name='spring',
            direction='long',
            fusion_weights={'wyckoff': 0.70, 'liquidity': 0.15, 'momentum': 0.10, 'smc': 0.05}
        )
        spring = ArchetypeInstance(spring_config)
        spring_fusion = spring.compute_fusion_score(sample_features)

        # Wick Trap: Liquidity-heavy
        wick_trap_config = ArchetypeConfig(
            name='wick_trap',
            direction='long',
            fusion_weights={'wyckoff': 0.20, 'liquidity': 0.60, 'momentum': 0.15, 'smc': 0.05}
        )
        wick_trap = ArchetypeInstance(wick_trap_config)
        wick_trap_fusion = wick_trap.compute_fusion_score(sample_features)

        # Whipsaw: Momentum-heavy
        whipsaw_config = ArchetypeConfig(
            name='whipsaw',
            direction='short',
            fusion_weights={'wyckoff': 0.10, 'liquidity': 0.30, 'momentum': 0.50, 'smc': 0.10}
        )
        whipsaw = ArchetypeInstance(whipsaw_config)
        whipsaw_fusion = whipsaw.compute_fusion_score(sample_features)

        # All three archetypes should have DIFFERENT fusion scores
        assert spring_fusion != wick_trap_fusion
        assert spring_fusion != whipsaw_fusion
        assert wick_trap_fusion != whipsaw_fusion

        # Spring should have highest (strong Wyckoff signal)
        assert spring_fusion > wick_trap_fusion
        assert spring_fusion > whipsaw_fusion

    def test_isolation_modifying_one_archetype_doesnt_affect_others(self, sample_features):
        """
        CRITICAL TEST: Prove that modifying archetype A's fusion weights
        does NOT affect archetype B's fusion score.

        This is the core isolation guarantee.
        """
        # Create archetype A with initial weights
        config_a = ArchetypeConfig(
            name='archetype_a',
            direction='long',
            fusion_weights={'wyckoff': 0.50, 'liquidity': 0.30, 'momentum': 0.15, 'smc': 0.05}
        )
        archetype_a = ArchetypeInstance(config_a)

        # Create archetype B with different weights
        config_b = ArchetypeConfig(
            name='archetype_b',
            direction='long',
            fusion_weights={'wyckoff': 0.25, 'liquidity': 0.25, 'momentum': 0.25, 'smc': 0.25}
        )
        archetype_b = ArchetypeInstance(config_b)

        # Calculate initial fusion scores
        fusion_b_before = archetype_b.compute_fusion_score(sample_features)

        # Modify archetype A's weights (simulate optimization)
        config_a_modified = ArchetypeConfig(
            name='archetype_a',
            direction='long',
            fusion_weights={'wyckoff': 0.10, 'liquidity': 0.70, 'momentum': 0.15, 'smc': 0.05}
        )
        archetype_a_modified = ArchetypeInstance(config_a_modified)
        fusion_a_modified = archetype_a_modified.compute_fusion_score(sample_features)

        # Recalculate archetype B's fusion score
        fusion_b_after = archetype_b.compute_fusion_score(sample_features)

        # CRITICAL: Archetype B's fusion score should be UNCHANGED
        assert fusion_b_before == fusion_b_after

        # Archetype A's fusion score should have changed
        fusion_a_original = archetype_a.compute_fusion_score(sample_features)
        assert fusion_a_modified != fusion_a_original

    def test_signal_generation_respects_threshold(self, sample_features):
        """Test that signals are only generated when fusion exceeds threshold."""
        config = ArchetypeConfig(
            name='test',
            direction='long',
            fusion_weights={'wyckoff': 0.70, 'liquidity': 0.15, 'momentum': 0.10, 'smc': 0.05},
            entry_threshold=0.90  # Very high threshold
        )

        archetype = ArchetypeInstance(config)
        signal = archetype.detect(sample_features, regime='neutral')

        # Should NOT generate signal (fusion ~0.875 < threshold 0.90)
        # Note: Depending on exact calculation, this might be close
        # Let's use a clearly too-high threshold
        config_high = ArchetypeConfig(
            name='test_high',
            direction='long',
            fusion_weights={'wyckoff': 0.70, 'liquidity': 0.15, 'momentum': 0.10, 'smc': 0.05},
            entry_threshold=0.95  # Definitely too high
        )

        archetype_high = ArchetypeInstance(config_high)
        signal = archetype_high.detect(sample_features, regime='neutral')

        assert signal is None

    def test_signal_generation_respects_liquidity_requirement(self, sample_features):
        """Test that signals require minimum liquidity."""
        # Set very high liquidity requirement
        config = ArchetypeConfig(
            name='test',
            direction='long',
            fusion_weights={'wyckoff': 0.70, 'liquidity': 0.15, 'momentum': 0.10, 'smc': 0.05},
            entry_threshold=0.35,
            min_liquidity=0.90  # Very high
        )

        archetype = ArchetypeInstance(config)
        signal = archetype.detect(sample_features, regime='neutral')

        # Should NOT generate signal (liquidity 0.6 < min 0.90)
        assert signal is None

    def test_signal_generation_success(self, sample_features):
        """Test successful signal generation with reasonable thresholds."""
        config = ArchetypeConfig(
            name='spring',
            direction='long',
            fusion_weights={'wyckoff': 0.70, 'liquidity': 0.15, 'momentum': 0.10, 'smc': 0.05},
            entry_threshold=0.35,
            min_liquidity=0.12
        )

        archetype = ArchetypeInstance(config)
        signal = archetype.detect(sample_features, regime='neutral')

        # Should generate signal
        assert signal is not None
        assert signal.direction == 'long'
        assert signal.confidence > 0
        assert signal.stop_loss < signal.entry_price  # Long position
        assert signal.take_profit > signal.entry_price
        assert signal.metadata['archetype'] == 'spring'

    def test_position_sizing_atr_based(self, sample_features):
        """Test ATR-based position sizing."""
        config = ArchetypeConfig(
            name='test',
            direction='long',
            fusion_weights={'wyckoff': 0.50, 'liquidity': 0.30, 'momentum': 0.15, 'smc': 0.05},
            entry_threshold=0.35,
            max_risk_pct=0.02  # 2% risk
        )

        archetype = ArchetypeInstance(config)
        signal = archetype.detect(sample_features, regime='neutral')

        assert signal is not None

        portfolio_value = 10000.0
        size = archetype.get_position_size(portfolio_value, signal, regime='neutral')

        # Should be reasonable size
        assert 0 < size <= portfolio_value * 0.12  # Max 12% of portfolio

        # Verify risk calculation
        # Risk = 2% of $10k = $200
        # Stop distance = entry - stop = ~2.5 ATR = ~2500
        # Expected size = $200 / (2500/50000) = $200 / 0.05 = $4000
        expected_size = portfolio_value * 0.02 / abs(signal.entry_price - signal.stop_loss) * signal.entry_price
        assert size == pytest.approx(min(expected_size, portfolio_value * 0.12), abs=100)

    def test_penalties_reduce_fusion_score(self, sample_features):
        """Test that penalties (PTI, fakeouts) reduce fusion score."""
        config = ArchetypeConfig(
            name='test',
            direction='long',
            fusion_weights={'wyckoff': 0.70, 'liquidity': 0.15, 'momentum': 0.10, 'smc': 0.05}
        )

        archetype = ArchetypeInstance(config)

        # Calculate fusion without penalties
        fusion_clean = archetype.compute_fusion_score(sample_features)

        # Add PTI penalty
        features_with_pti = sample_features.copy()
        features_with_pti['tf1d_pti_score'] = 0.8
        fusion_with_pti = archetype.compute_fusion_score(features_with_pti)

        # PTI should reduce fusion
        assert fusion_with_pti < fusion_clean
        assert fusion_with_pti == pytest.approx(fusion_clean - 0.08, abs=0.01)

        # Add fakeout penalty
        features_with_fakeout = sample_features.copy()
        features_with_fakeout['tf1h_fakeout_detected'] = True
        fusion_with_fakeout = archetype.compute_fusion_score(features_with_fakeout)

        # Fakeout should reduce fusion
        assert fusion_with_fakeout < fusion_clean
        assert fusion_with_fakeout == pytest.approx(fusion_clean - 0.10, abs=0.01)


class TestArchetypeIsolationIntegration:
    """Integration tests proving complete archetype isolation."""

    def test_multiple_archetypes_independent_optimization(self):
        """
        Simulate optimizing multiple archetypes independently.

        This is the real-world use case: We want to optimize spring, wick_trap,
        and whipsaw independently without interference.
        """
        features = {
            'close': 50000.0,
            'atr_14': 1000.0,
            'tf1d_wyckoff_m1_signal': True,
            'wyckoff_event_confidence': 0.8,
            'liquidity_score': 0.6,
            'adx_14': 30.0,
            'rsi_14': 65.0,
            'tf4h_squiggle_confidence': 0.7,
            'fusion_smc': 0.5,
            'tf1h_fakeout_detected': False,
            'tf1d_pti_score': 0.0,
            'tf1h_pti_score': 0.0
        }

        # Initial configurations
        spring_v1 = ArchetypeInstance(ArchetypeConfig(
            name='spring',
            direction='long',
            fusion_weights={'wyckoff': 0.50, 'liquidity': 0.30, 'momentum': 0.15, 'smc': 0.05},
            entry_threshold=0.35
        ))

        wick_trap_v1 = ArchetypeInstance(ArchetypeConfig(
            name='wick_trap',
            direction='long',
            fusion_weights={'wyckoff': 0.25, 'liquidity': 0.50, 'momentum': 0.20, 'smc': 0.05},
            entry_threshold=0.28
        ))

        # Calculate initial fusion scores
        spring_fusion_v1 = spring_v1.compute_fusion_score(features)
        wick_trap_fusion_v1 = wick_trap_v1.compute_fusion_score(features)

        # Optimize spring (increase wyckoff emphasis)
        spring_v2 = ArchetypeInstance(ArchetypeConfig(
            name='spring',
            direction='long',
            fusion_weights={'wyckoff': 0.70, 'liquidity': 0.15, 'momentum': 0.10, 'smc': 0.05},
            entry_threshold=0.35
        ))

        spring_fusion_v2 = spring_v2.compute_fusion_score(features)

        # Recalculate wick_trap fusion (should be unchanged)
        wick_trap_fusion_v2 = wick_trap_v1.compute_fusion_score(features)

        # CRITICAL: Spring changed, wick_trap unchanged
        assert spring_fusion_v2 != spring_fusion_v1
        assert wick_trap_fusion_v2 == wick_trap_fusion_v1

        # Now optimize wick_trap (increase liquidity emphasis)
        wick_trap_v2 = ArchetypeInstance(ArchetypeConfig(
            name='wick_trap',
            direction='long',
            fusion_weights={'wyckoff': 0.20, 'liquidity': 0.60, 'momentum': 0.15, 'smc': 0.05},
            entry_threshold=0.28
        ))

        wick_trap_fusion_v3 = wick_trap_v2.compute_fusion_score(features)

        # Recalculate spring fusion (should be unchanged)
        spring_fusion_v3 = spring_v2.compute_fusion_score(features)

        # CRITICAL: Wick trap changed, spring unchanged
        assert wick_trap_fusion_v3 != wick_trap_fusion_v1
        assert spring_fusion_v3 == spring_fusion_v2
