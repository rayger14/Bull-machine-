"""
Comprehensive Macro Pulse Tests - Intermarket Relationships and Veto Logic

Tests the complete macro pulse system including DXY/Oil/Gold/Bonds relationships,
veto/boost logic, and fusion integration.
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from engine.context.macro_pulse import (
    MacroPulseEngine, MacroPulse, MacroRegime,
    dxy_breakout_strength, oil_dxy_stagflation, yields_spike, vix_move_spike,
    usd_jpy_carry_break, hyg_credit_stress, usdt_sfp_wolfe, total3_vs_total
)
from engine.fusion import FusionEngine, FusionSignal
import json

class TestMacroPulseHelpers:
    """Test individual macro pulse helper functions"""

    def test_dxy_breakout_detection(self):
        """Test DXY breakout strength calculation"""
        # Create DXY data with clear breakout
        dates = pd.date_range('2024-01-01', periods=30, freq='1D')

        # Stable range then breakout
        prices = [100] * 20 + [102, 103, 104, 105, 106, 107, 108, 109, 110, 111]

        dxy_data = pd.DataFrame({
            'close': prices,
            'high': [p + 0.5 for p in prices],
            'low': [p - 0.5 for p in prices]
        }, index=dates)

        breakout = dxy_breakout_strength(dxy_data, 20)
        assert breakout > 0.5  # Should detect strong breakout
        assert breakout <= 1.0  # Bounded

    def test_oil_dxy_stagflation_detection(self):
        """Test Oil↑ + DXY↑ stagflation detection"""
        dates = pd.date_range('2024-01-01', periods=30, freq='1D')

        # Both rising significantly
        oil_rising = pd.DataFrame({
            'close': np.linspace(70, 80, 30)  # +14% rise
        }, index=dates)

        dxy_rising = pd.DataFrame({
            'close': np.linspace(100, 102, 30)  # +2% rise
        }, index=dates)

        stagflation = oil_dxy_stagflation(oil_rising, dxy_rising)
        assert stagflation is True

        # Test non-stagflation case
        oil_falling = pd.DataFrame({
            'close': np.linspace(80, 70, 30)  # Falling
        }, index=dates)

        no_stagflation = oil_dxy_stagflation(oil_falling, dxy_rising)
        assert no_stagflation is False

    def test_yield_spike_detection(self):
        """Test yield spike detection"""
        dates = pd.date_range('2024-01-01', periods=30, freq='1D')

        # Normal yields then spike
        normal_yields = [0.045] * 25
        spike_yields = [0.045, 0.047, 0.055, 0.058, 0.060]  # Sharp spike

        us2y = pd.DataFrame({
            'close': normal_yields + spike_yields
        }, index=dates)

        us10y = pd.DataFrame({
            'close': [0.042] * 30  # Stable
        }, index=dates)

        spike = yields_spike(us2y, us10y, 2.0)
        assert spike > 0.5  # Should detect spike
        assert spike <= 1.0  # Bounded

    def test_vix_move_spike_detection(self):
        """Test VIX/MOVE volatility spike detection"""
        dates = pd.date_range('2024-01-01', periods=10, freq='1D')

        # VIX spike scenario
        vix_spike = pd.DataFrame({
            'close': [15, 18, 22, 28, 35, 32, 28, 25, 22, 20]
        }, index=dates)

        move_normal = pd.DataFrame({
            'close': [100] * 10
        }, index=dates)

        spike = vix_move_spike(vix_spike, move_normal, 24.0, 130.0)
        assert spike > 0.4  # Should detect VIX spike above 24

    def test_usdjpy_carry_unwind(self):
        """Test USDJPY carry trade unwind detection"""
        dates = pd.date_range('2024-01-01', periods=10, freq='1D')

        # Sharp drop below 145 level
        usdjpy_crash = pd.DataFrame({
            'close': [148, 147, 146, 145, 144, 142, 140, 138, 136, 135]
        }, index=dates)

        carry_unwind = usd_jpy_carry_break(usdjpy_crash, 145.0)
        assert carry_unwind is True

    def test_hyg_credit_stress(self):
        """Test HYG credit stress detection"""
        dates = pd.date_range('2024-01-01', periods=20, freq='1D')

        # Credit stress scenario - HYG dropping
        hyg_stress = pd.DataFrame({
            'close': np.linspace(80, 75, 20)  # >6% drop
        }, index=dates)

        stress = hyg_credit_stress(hyg_stress, 15)
        assert stress > 0.8  # Should detect high stress
        assert stress <= 1.0  # Bounded

    def test_usdt_sfp_detection(self):
        """Test USDT.D SFP/Wolfe wave detection"""
        dates = pd.date_range('2024-01-01', periods=60, freq='4H')

        # Create SFP pattern - false breakout then failure
        highs = [4.0] * 50 + [4.05, 4.08, 4.12, 4.10, 4.05, 4.02, 3.98, 3.95, 3.92, 3.90]
        closes = [3.98] * 50 + [4.02, 4.05, 4.08, 4.06, 4.00, 3.96, 3.94, 3.91, 3.88, 3.86]

        usdt_sfp = pd.DataFrame({
            'high': highs,
            'close': closes,
            'low': [h - 0.05 for h in highs]
        }, index=dates)

        sfp_score = usdt_sfp_wolfe(usdt_sfp, 50)
        assert sfp_score > 0.5  # Should detect SFP pattern

    def test_total3_divergence(self):
        """Test TOTAL3 vs TOTAL divergence detection"""
        dates = pd.date_range('2024-01-01', periods=20, freq='4H')

        # TOTAL3 outperforming TOTAL
        total3_strong = pd.DataFrame({
            'close': np.linspace(4.5e11, 5.0e11, 20)  # +11% gain
        }, index=dates)

        total_weak = pd.DataFrame({
            'close': np.linspace(1.8e12, 1.85e12, 20)  # +2.8% gain
        }, index=dates)

        divergence = total3_vs_total(total3_strong, total_weak)
        assert divergence > 0.5  # Should detect alt leadership
        assert divergence <= 1.0  # Bounded

class TestMacroPulseEngine:
    """Test complete Macro Pulse Engine"""

    def setup_method(self):
        """Setup test configuration"""
        with open('configs/v170/context.json', 'r') as f:
            self.config = json.load(f)
        self.engine = MacroPulseEngine(self.config)

    def create_mock_series(self, scenario: str = 'neutral') -> dict:
        """Create mock market data for different scenarios"""
        dates_1d = pd.date_range('2024-01-01', periods=250, freq='1D')
        dates_4h = pd.date_range('2024-05-01', periods=300, freq='4H')

        def make_df(values, dates):
            return pd.DataFrame({
                'open': values,
                'high': [v * 1.01 for v in values],
                'low': [v * 0.99 for v in values],
                'close': values,
                'volume': [1000000] * len(values)
            }, index=dates)

        if scenario == 'stagflation_veto':
            # Oil and DXY both rising
            return {
                'DXY_1D': make_df(np.linspace(100, 105, 250), dates_1d),  # Strong DXY
                'WTI_1D': make_df(np.linspace(70, 85, 250), dates_1d),    # Oil surge
                'GOLD_1D': make_df([2000] * 250, dates_1d),
                'US2Y_1D': make_df([0.045] * 250, dates_1d),
                'US10Y_1D': make_df([0.042] * 250, dates_1d),
                'VIX_1D': make_df([15] * 250, dates_1d),
                'MOVE_1D': make_df([100] * 250, dates_1d),
                'USDJPY_1D': make_df([148] * 250, dates_1d),
                'HYG_1D': make_df([78] * 250, dates_1d),
                'ETH.D_1D': make_df([18] * 250, dates_1d),
                'USDT.D_4H': make_df([4.0] * 300, dates_4h),
                'TOTAL_4H': make_df([1.8e12] * 300, dates_4h),
                'TOTAL3_4H': make_df([4.5e11] * 300, dates_4h),
                'ETHBTC_1D': make_df([0.055] * 250, dates_1d)
            }

        elif scenario == 'risk_on_boost':
            # DXY weak, alt leadership
            return {
                'DXY_1D': make_df(np.linspace(105, 98, 250), dates_1d),   # DXY breakdown
                'WTI_1D': make_df([75] * 250, dates_1d),
                'GOLD_1D': make_df([2000] * 250, dates_1d),
                'US2Y_1D': make_df(np.linspace(0.055, 0.040, 250), dates_1d),  # Yields falling
                'US10Y_1D': make_df(np.linspace(0.052, 0.038, 250), dates_1d),
                'VIX_1D': make_df([12] * 250, dates_1d),                   # Low vol
                'MOVE_1D': make_df([85] * 250, dates_1d),
                'USDJPY_1D': make_df([148] * 250, dates_1d),
                'HYG_1D': make_df([78] * 250, dates_1d),
                'ETH.D_1D': make_df(np.linspace(17, 19, 250), dates_1d),  # ETH.D rising
                'USDT.D_4H': make_df(np.linspace(4.2, 3.8, 300), dates_4h),  # USDT.D falling
                'TOTAL_4H': make_df(np.linspace(1.8e12, 2.0e12, 300), dates_4h),
                'TOTAL3_4H': make_df(np.linspace(4.5e11, 5.5e11, 300), dates_4h),  # TOTAL3 leadership
                'ETHBTC_1D': make_df(np.linspace(0.055, 0.065, 250), dates_1d)   # ETH/BTC rising
            }

        else:  # neutral
            return {
                'DXY_1D': make_df([100] * 250, dates_1d),
                'WTI_1D': make_df([75] * 250, dates_1d),
                'GOLD_1D': make_df([2000] * 250, dates_1d),
                'US2Y_1D': make_df([0.045] * 250, dates_1d),
                'US10Y_1D': make_df([0.042] * 250, dates_1d),
                'VIX_1D': make_df([15] * 250, dates_1d),
                'MOVE_1D': make_df([100] * 250, dates_1d),
                'USDJPY_1D': make_df([148] * 250, dates_1d),
                'HYG_1D': make_df([78] * 250, dates_1d),
                'ETH.D_1D': make_df([18] * 250, dates_1d),
                'USDT.D_4H': make_df([4.0] * 300, dates_4h),
                'TOTAL_4H': make_df([1.8e12] * 300, dates_4h),
                'TOTAL3_4H': make_df([4.5e11] * 300, dates_4h),
                'ETHBTC_1D': make_df([0.055] * 250, dates_1d)
            }

    def test_stagflation_veto_scenario(self):
        """Test stagflation veto scenario"""
        series_data = self.create_mock_series('stagflation_veto')
        pulse = self.engine.analyze_macro_pulse(series_data)

        assert pulse.regime == MacroRegime.STAGFLATION
        assert pulse.suppression_flag is True
        assert pulse.veto_strength > 0.7
        assert "stagflation" in ' '.join(pulse.notes).lower()

    def test_risk_on_boost_scenario(self):
        """Test risk-on boost scenario"""
        series_data = self.create_mock_series('risk_on_boost')
        pulse = self.engine.analyze_macro_pulse(series_data)

        assert pulse.regime == MacroRegime.RISK_ON
        assert pulse.suppression_flag is False
        assert pulse.boost_strength > 0.0
        assert pulse.risk_bias == 'risk_on'
        assert pulse.macro_delta > 0.0

    def test_neutral_scenario(self):
        """Test neutral macro scenario"""
        series_data = self.create_mock_series('neutral')
        pulse = self.engine.analyze_macro_pulse(series_data)

        assert pulse.regime == MacroRegime.NEUTRAL
        assert pulse.suppression_flag is False
        assert abs(pulse.macro_delta) < 0.05
        assert pulse.risk_bias == 'neutral'

    def test_macro_signal_generation(self):
        """Test macro signal generation and classification"""
        series_data = self.create_mock_series('stagflation_veto')
        pulse = self.engine.analyze_macro_pulse(series_data)

        # Should have multiple active signals
        assert len(pulse.active_signals) >= 2

        # Should have stagflation signal
        stagflation_signals = [s for s in pulse.active_signals if s.name == 'STAGFLATION_VETO']
        assert len(stagflation_signals) > 0

        # Signals should have proper structure
        for signal in pulse.active_signals:
            assert hasattr(signal, 'name')
            assert hasattr(signal, 'value')
            assert hasattr(signal, 'direction')
            assert hasattr(signal, 'confidence')
            assert 0 <= signal.value <= 1
            assert 0 <= signal.confidence <= 1

class TestFusionMacroIntegration:
    """Test Fusion Engine integration with Macro Pulse"""

    def setup_method(self):
        """Setup fusion engine with macro pulse"""
        # Load both fusion and context configs
        with open('configs/v170/assets/ETH_v17_baseline.json', 'r') as f:
            config = json.load(f)

        # Add context config
        with open('configs/v170/context.json', 'r') as f:
            config['context'] = json.load(f)

        self.engine = FusionEngine(config)

    def create_domain_signals(self, direction: str = 'long') -> dict:
        """Create mock domain signals"""
        return {
            'wyckoff': {
                'signal_type': 'accumulation',
                'direction': direction,
                'strength': 0.7,
                'confidence': 0.8,
                'stop_loss': 49000,
                'targets': [51000, 52000, 53000]
            },
            'liquidity': {
                'signal_type': 'hob_reaction',
                'direction': direction,
                'strength': 0.75,
                'confidence': 0.85,
                'stop_loss': 49200,
                'targets': [51500, 52500]
            },
            'momentum': {
                'signal_type': 'momentum_shift',
                'direction': direction,
                'strength': 0.65,
                'confidence': 0.7,
                'targets': [51200]
            }
        }

    def create_market_data_with_macro(self, macro_scenario: str = 'neutral') -> dict:
        """Create market data including macro series"""
        dates_1h = pd.date_range('2024-01-01', periods=100, freq='1H')
        dates_1d = pd.date_range('2024-01-01', periods=50, freq='1D')
        dates_4h = pd.date_range('2024-01-01', periods=200, freq='4H')

        def make_df(values, dates):
            return pd.DataFrame({
                'open': values,
                'high': [v * 1.005 for v in values],
                'low': [v * 0.995 for v in values],
                'close': values,
                'volume': [1000000] * len(values)
            }, index=dates)

        # Base crypto data
        market_data = {
            '1H': make_df([50000] * 100, dates_1h),
            '4H': make_df([50000] * 200, dates_4h),
            '1D': make_df([50000] * 50, dates_1d)
        }

        # Add macro data based on scenario
        if macro_scenario == 'stagflation_veto':
            market_data.update({
                'DXY_1D': make_df(np.linspace(100, 105, 50), dates_1d),
                'WTI_1D': make_df(np.linspace(70, 85, 50), dates_1d),
                'VIX_1D': make_df([30] * 50, dates_1d),  # High VIX
            })
        elif macro_scenario == 'risk_on_boost':
            market_data.update({
                'DXY_1D': make_df(np.linspace(105, 98, 50), dates_1d),
                'WTI_1D': make_df([75] * 50, dates_1d),
                'VIX_1D': make_df([12] * 50, dates_1d),  # Low VIX
                'USDT.D_4H': make_df(np.linspace(4.2, 3.8, 200), dates_4h),
                'TOTAL3_4H': make_df(np.linspace(4.5e11, 5.5e11, 200), dates_4h),
            })
        else:  # neutral
            market_data.update({
                'DXY_1D': make_df([100] * 50, dates_1d),
                'WTI_1D': make_df([75] * 50, dates_1d),
                'VIX_1D': make_df([15] * 50, dates_1d),
            })

        return market_data

    def test_macro_veto_blocks_signal(self):
        """Test that hard macro veto blocks signal generation"""
        domain_signals = self.create_domain_signals('long')
        market_data = self.create_market_data_with_macro('stagflation_veto')

        # Should return None due to macro veto
        result = self.engine.aggregate_signals(domain_signals, market_data)
        assert result is None

    def test_macro_boost_enhances_signal(self):
        """Test that macro boost enhances signal quality"""
        domain_signals = self.create_domain_signals('long')

        # Test with neutral macro
        neutral_data = self.create_market_data_with_macro('neutral')
        neutral_result = self.engine.aggregate_signals(domain_signals, neutral_data)

        # Test with risk-on boost
        boost_data = self.create_market_data_with_macro('risk_on_boost')
        boost_result = self.engine.aggregate_signals(domain_signals, boost_data)

        if neutral_result and boost_result:
            # Boosted signal should be stronger
            assert boost_result.strength >= neutral_result.strength
            assert boost_result.confidence >= neutral_result.confidence
            assert boost_result.macro_delta > 0
            assert boost_result.risk_bias == 'risk_on'

    def test_explainable_factors_generation(self):
        """Test explainable factors in fusion signal"""
        domain_signals = self.create_domain_signals('long')
        market_data = self.create_market_data_with_macro('risk_on_boost')

        result = self.engine.aggregate_signals(domain_signals, market_data)

        if result:
            factors = result.explainable_factors

            # Should have all required sections
            assert 'signal_summary' in factors
            assert 'domain_breakdown' in factors
            assert 'macro_context' in factors
            assert 'quality_factors' in factors
            assert 'risk_considerations' in factors

            # Signal summary should be descriptive
            assert 'LONG' in factors['signal_summary']
            assert '3 domains' in factors['signal_summary']

            # Macro context should be populated
            macro_context = factors['macro_context']
            assert 'regime' in macro_context
            assert 'risk_bias' in macro_context
            assert 'macro_delta' in macro_context

    def test_macro_pulse_integration_complete(self):
        """Test complete macro pulse integration in fusion signal"""
        domain_signals = self.create_domain_signals('long')
        market_data = self.create_market_data_with_macro('risk_on_boost')

        result = self.engine.aggregate_signals(domain_signals, market_data)

        if result:
            # Should have macro pulse data
            assert result.macro_pulse is not None
            assert isinstance(result.macro_delta, float)
            assert result.risk_bias in ['risk_on', 'risk_off', 'neutral']

            # Macro pulse should be populated
            pulse = result.macro_pulse
            assert pulse.regime in [r for r in MacroRegime]
            assert isinstance(pulse.veto_strength, float)
            assert isinstance(pulse.boost_strength, float)
            assert isinstance(pulse.active_signals, list)

class TestMacroWisdomIntegration:
    """Test integration of macro wisdom and edge cases"""

    def test_dxy_heartbeat_wisdom(self):
        """Test 'DXY is the heartbeat of global liquidity' wisdom"""
        config = {'dxy_breakout_len': 20, 'veto_threshold': 0.7}
        engine = MacroPulseEngine(config)

        dates = pd.date_range('2024-01-01', periods=30, freq='1D')

        # Strong DXY breakout should dominate other signals
        dxy_breakout = pd.DataFrame({
            'close': [100] * 20 + [105, 107, 109, 111, 113, 115, 117, 119, 121, 123],
            'high': [101] * 20 + [106, 108, 110, 112, 114, 116, 118, 120, 122, 124],
            'low': [99] * 20 + [104, 106, 108, 110, 112, 114, 116, 118, 120, 122]
        }, index=dates)

        series = {'DXY_1D': dxy_breakout}
        pulse = engine.analyze_macro_pulse(series)

        # Should generate strong veto
        assert pulse.veto_strength > 0.7
        assert "DXY" in ' '.join(pulse.notes)

    def test_oil_dxy_poison_wisdom(self):
        """Test 'Oil and DXY rising together is poison' wisdom"""
        config = {'oil_dxy_stagflation_veto': 0.85, 'veto_threshold': 0.7}
        engine = MacroPulseEngine(config)

        dates = pd.date_range('2024-01-01', periods=30, freq='1D')

        # Both oil and DXY rising
        oil_rising = pd.DataFrame({'close': np.linspace(70, 85, 30)}, index=dates)
        dxy_rising = pd.DataFrame({'close': np.linspace(100, 103, 30)}, index=dates)

        series = {'WTI_1D': oil_rising, 'DXY_1D': dxy_rising}
        pulse = engine.analyze_macro_pulse(series)

        assert pulse.regime == MacroRegime.STAGFLATION
        assert "poison" in ' '.join(pulse.notes).lower() or "stagflation" in ' '.join(pulse.notes).lower()

    def test_bounded_deltas_enforcement(self):
        """Test that macro deltas are properly bounded"""
        config = {
            'weights': {'boost_context_max': 0.10},
            'veto_threshold': 0.7
        }
        engine = MacroPulseEngine(config)

        # Create extreme scenario
        dates = pd.date_range('2024-01-01', periods=50, freq='1D')
        extreme_series = {
            'DXY_1D': pd.DataFrame({'close': np.linspace(105, 85, 50)}, index=dates),  # Massive decline
            'TOTAL3_4H': pd.DataFrame({'close': np.linspace(4e11, 6e11, 200)},
                                    index=pd.date_range('2024-01-01', periods=200, freq='4H')),
        }

        pulse = engine.analyze_macro_pulse(extreme_series)

        # Delta should be bounded
        assert -0.10 <= pulse.macro_delta <= 0.10

if __name__ == '__main__':
    pytest.main([__file__, '-v'])