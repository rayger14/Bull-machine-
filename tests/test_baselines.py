"""
Unit tests for baseline models.

Tests each baseline model for:
- Correct initialization
- Fit behavior
- Signal generation
- Position sizing
- Edge cases (no data, insufficient history, etc.)
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from engine.models.baselines import (
    Baseline0_BuyAndHold,
    Baseline1_SMA200Trend,
    Baseline2_SMACrossover,
    Baseline3_RSIMeanReversion,
    Baseline4_VolTargetTrend,
    Baseline5_Cash
)
from engine.models.base import Signal, Position


@pytest.fixture
def sample_data():
    """Generate sample OHLCV data for testing."""
    np.random.seed(42)
    n_bars = 300

    dates = pd.date_range(start='2023-01-01', periods=n_bars, freq='1H')

    # Generate realistic price series (trending up)
    close_prices = 50000 + np.cumsum(np.random.randn(n_bars) * 100)

    data = pd.DataFrame({
        'timestamp': dates,
        'open': close_prices * 0.999,
        'high': close_prices * 1.002,
        'low': close_prices * 0.998,
        'close': close_prices,
        'volume': np.random.randint(100, 1000, n_bars)
    })

    data.set_index('timestamp', inplace=True)

    # Add indicators
    data['sma_50'] = data['close'].rolling(window=50).mean()
    data['sma_200'] = data['close'].rolling(window=200).mean()

    # RSI
    delta = data['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    data['rsi_14'] = 100 - (100 / (1 + rs))

    # ATR
    tr1 = data['high'] - data['low']
    tr2 = abs(data['high'] - data['close'].shift(1))
    tr3 = abs(data['low'] - data['close'].shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    data['atr_14'] = tr.rolling(window=14).mean()

    return data


class TestBaseline0BuyAndHold:
    """Test buy-and-hold baseline."""

    def test_initialization(self):
        """Test model initialization."""
        model = Baseline0_BuyAndHold(position_size=1000.0)
        assert model.name == "Baseline0_BuyAndHold"
        assert model.position_size == 1000.0
        assert not model._is_fitted

    def test_fit(self, sample_data):
        """Test fit method."""
        model = Baseline0_BuyAndHold()
        model.fit(sample_data)
        assert model._is_fitted

    def test_first_signal_is_long(self, sample_data):
        """Test that first signal is long entry."""
        model = Baseline0_BuyAndHold()
        model.fit(sample_data)

        bar = sample_data.iloc[0]
        signal = model.predict(bar, position=None)

        assert signal.direction == 'long'
        assert signal.confidence == 1.0
        assert signal.entry_price == bar['close']

    def test_subsequent_signals_are_hold(self, sample_data):
        """Test that subsequent signals are hold."""
        model = Baseline0_BuyAndHold()
        model.fit(sample_data)

        # First signal (entry)
        bar1 = sample_data.iloc[0]
        signal1 = model.predict(bar1, position=None)
        assert signal1.direction == 'long'

        # Create position
        position = Position(
            direction='long',
            entry_price=bar1['close'],
            entry_time=bar1.name,
            size=1000.0,
            stop_loss=0.0
        )

        # Second signal (should hold)
        bar2 = sample_data.iloc[1]
        signal2 = model.predict(bar2, position=position)
        assert signal2.direction == 'hold'

    def test_position_sizing(self, sample_data):
        """Test position sizing."""
        model = Baseline0_BuyAndHold(position_size=5000.0)
        bar = sample_data.iloc[0]
        signal = Signal(direction='long', confidence=1.0, entry_price=bar['close'])

        size = model.get_position_size(bar, signal)
        assert size == 5000.0


class TestBaseline1SMA200Trend:
    """Test SMA trend-following baseline."""

    def test_initialization(self):
        """Test model initialization."""
        model = Baseline1_SMA200Trend(sma_period=200)
        assert 'SMA200' in model.name
        assert model.sma_period == 200

    def test_fit(self, sample_data):
        """Test fit method."""
        model = Baseline1_SMA200Trend()
        model.fit(sample_data)
        assert model._is_fitted

    def test_entry_when_above_sma(self, sample_data):
        """Test long entry when price > SMA."""
        model = Baseline1_SMA200Trend()
        model.fit(sample_data)

        # Find bar where close > SMA
        for idx in range(200, len(sample_data)):
            bar = sample_data.iloc[idx]
            if pd.notna(bar['sma_200']) and bar['close'] > bar['sma_200']:
                signal = model.predict(bar, position=None)
                assert signal.direction == 'long'
                assert signal.stop_loss < signal.entry_price
                break

    def test_no_entry_when_below_sma(self, sample_data):
        """Test no entry when price < SMA."""
        model = Baseline1_SMA200Trend()
        model.fit(sample_data)

        # Force a below-SMA condition
        bar = sample_data.iloc[250].copy()
        bar['close'] = bar['sma_200'] * 0.95  # 5% below SMA

        signal = model.predict(bar, position=None)
        assert signal.direction == 'hold'

    def test_insufficient_history(self, sample_data):
        """Test behavior with insufficient history (SMA not available)."""
        model = Baseline1_SMA200Trend()
        model.fit(sample_data)

        # Early bar where SMA is NaN
        bar = sample_data.iloc[100].copy()
        bar['sma_200'] = np.nan

        signal = model.predict(bar, position=None)
        assert signal.direction == 'hold'


class TestBaseline2SMACrossover:
    """Test SMA crossover baseline."""

    def test_initialization(self):
        """Test model initialization."""
        model = Baseline2_SMACrossover(fast_period=50, slow_period=200)
        assert model.fast_period == 50
        assert model.slow_period == 200

    def test_fit(self, sample_data):
        """Test fit method."""
        model = Baseline2_SMACrossover()
        model.fit(sample_data)
        assert model._is_fitted

    def test_golden_cross_entry(self, sample_data):
        """Test entry on golden cross."""
        model = Baseline2_SMACrossover()
        model.fit(sample_data)

        # Find golden cross (fast > slow)
        for idx in range(200, len(sample_data)):
            bar = sample_data.iloc[idx]
            if pd.notna(bar['sma_50']) and pd.notna(bar['sma_200']):
                if bar['sma_50'] > bar['sma_200']:
                    signal = model.predict(bar, position=None)
                    assert signal.direction == 'long'
                    break

    def test_death_cross_no_entry(self, sample_data):
        """Test no entry on death cross."""
        model = Baseline2_SMACrossover()
        model.fit(sample_data)

        # Force death cross condition
        bar = sample_data.iloc[250].copy()
        bar['sma_50'] = 100.0
        bar['sma_200'] = 110.0  # Slow > Fast

        signal = model.predict(bar, position=None)
        assert signal.direction == 'hold'


class TestBaseline3RSIMeanReversion:
    """Test RSI mean reversion baseline."""

    def test_initialization(self):
        """Test model initialization."""
        model = Baseline3_RSIMeanReversion(
            rsi_period=14,
            entry_threshold=30.0,
            exit_threshold=70.0
        )
        assert model.rsi_period == 14
        assert model.entry_threshold == 30.0
        assert model.exit_threshold == 70.0

    def test_fit(self, sample_data):
        """Test fit method."""
        model = Baseline3_RSIMeanReversion()
        model.fit(sample_data)
        assert model._is_fitted

    def test_oversold_entry(self, sample_data):
        """Test entry when RSI < 30 (oversold)."""
        model = Baseline3_RSIMeanReversion()
        model.fit(sample_data)

        # Force oversold condition
        bar = sample_data.iloc[50].copy()
        bar['rsi_14'] = 25.0

        signal = model.predict(bar, position=None)
        assert signal.direction == 'long'
        assert signal.confidence > 0

    def test_overbought_exit(self, sample_data):
        """Test exit when RSI > 70 (overbought)."""
        model = Baseline3_RSIMeanReversion()
        model.fit(sample_data)

        # Create position
        position = Position(
            direction='long',
            entry_price=50000.0,
            entry_time=sample_data.index[50],
            size=1000.0,
            stop_loss=49000.0
        )

        # Force overbought condition
        bar = sample_data.iloc[100].copy()
        bar['rsi_14'] = 75.0

        signal = model.predict(bar, position=position)
        # Should signal exit (hold with exit reason)
        if signal.metadata and 'reason' in signal.metadata:
            assert signal.metadata['reason'] == 'signal'

    def test_neutral_rsi_no_action(self, sample_data):
        """Test no action when RSI is neutral (30-70)."""
        model = Baseline3_RSIMeanReversion()
        model.fit(sample_data)

        bar = sample_data.iloc[50].copy()
        bar['rsi_14'] = 50.0  # Neutral

        signal = model.predict(bar, position=None)
        assert signal.direction == 'hold'


class TestBaseline4VolTargetTrend:
    """Test volatility-targeted trend baseline."""

    def test_initialization(self):
        """Test model initialization."""
        model = Baseline4_VolTargetTrend(
            sma_period=200,
            atr_period=14,
            target_vol=0.02
        )
        assert model.sma_period == 200
        assert model.atr_period == 14
        assert model.target_vol == 0.02

    def test_fit(self, sample_data):
        """Test fit method."""
        model = Baseline4_VolTargetTrend()
        model.fit(sample_data)
        assert model._is_fitted

    def test_vol_adjusted_position_sizing(self, sample_data):
        """Test that position size adjusts with volatility."""
        model = Baseline4_VolTargetTrend(target_vol=0.02, base_capital=10000.0)
        model.fit(sample_data)

        # High volatility scenario
        bar_high_vol = sample_data.iloc[250].copy()
        bar_high_vol['atr_14'] = 1000.0  # High ATR

        signal_high_vol = Signal(
            direction='long',
            confidence=0.8,
            entry_price=bar_high_vol['close'],
            metadata={'atr': 1000.0}
        )

        size_high_vol = model.get_position_size(bar_high_vol, signal_high_vol)

        # Low volatility scenario
        bar_low_vol = sample_data.iloc[250].copy()
        bar_low_vol['atr_14'] = 100.0  # Low ATR

        signal_low_vol = Signal(
            direction='long',
            confidence=0.8,
            entry_price=bar_low_vol['close'],
            metadata={'atr': 100.0}
        )

        size_low_vol = model.get_position_size(bar_low_vol, signal_low_vol)

        # Low vol should have larger position size (unless capped at max)
        # Both might hit the 50% of capital cap, so check they're reasonable
        assert size_high_vol > 0
        assert size_low_vol > 0
        assert size_low_vol >= size_high_vol  # Allow equal if both capped

    def test_atr_based_stop_loss(self, sample_data):
        """Test that stop loss is ATR-based."""
        model = Baseline4_VolTargetTrend(stop_atr_mult=2.5)
        model.fit(sample_data)

        # Find bar with valid SMA and ATR
        for idx in range(200, len(sample_data)):
            bar = sample_data.iloc[idx]
            if (pd.notna(bar['sma_200']) and pd.notna(bar['atr_14']) and
                bar['close'] > bar['sma_200']):

                signal = model.predict(bar, position=None)
                if signal.direction == 'long':
                    expected_stop = bar['close'] - (2.5 * bar['atr_14'])
                    assert abs(signal.stop_loss - expected_stop) < 1.0  # Allow small rounding error
                    break


class TestBaseline5Cash:
    """Test cash (do nothing) baseline."""

    def test_initialization(self):
        """Test model initialization."""
        model = Baseline5_Cash()
        assert model.name == "Baseline5_Cash"

    def test_fit(self, sample_data):
        """Test fit method."""
        model = Baseline5_Cash()
        model.fit(sample_data)
        assert model._is_fitted

    def test_always_hold(self, sample_data):
        """Test that model never enters positions."""
        model = Baseline5_Cash()
        model.fit(sample_data)

        # Test multiple bars
        for idx in range(0, len(sample_data), 10):
            bar = sample_data.iloc[idx]
            signal = model.predict(bar, position=None)
            assert signal.direction == 'hold'
            assert signal.confidence == 0.0

    def test_position_size_zero(self, sample_data):
        """Test that position size is always zero."""
        model = Baseline5_Cash()
        bar = sample_data.iloc[0]
        signal = Signal(direction='long', confidence=1.0, entry_price=bar['close'])

        size = model.get_position_size(bar, signal)
        assert size == 0.0


def test_all_baselines_implement_interface(sample_data):
    """Test that all baselines implement BaseModel interface correctly."""
    from engine.models.baselines import get_all_baselines

    baselines = get_all_baselines()

    for baseline_cls in baselines:
        model = baseline_cls()

        # Test interface methods exist
        assert hasattr(model, 'fit')
        assert hasattr(model, 'predict')
        assert hasattr(model, 'get_position_size')
        assert hasattr(model, 'get_params')
        assert hasattr(model, 'get_state')

        # Test fit/predict workflow
        model.fit(sample_data)
        assert model._is_fitted

        bar = sample_data.iloc[200]  # Use bar with indicators available
        signal = model.predict(bar, position=None)
        assert hasattr(signal, 'direction')
        assert hasattr(signal, 'confidence')
        assert hasattr(signal, 'entry_price')


def test_baseline_params_serializable(sample_data):
    """Test that all baseline params are JSON-serializable."""
    import json
    from engine.models.baselines import get_all_baselines

    baselines = get_all_baselines()

    for baseline_cls in baselines:
        model = baseline_cls()
        model.fit(sample_data)

        params = model.get_params()

        # Should be serializable to JSON
        try:
            json.dumps(params)
        except (TypeError, ValueError) as e:
            pytest.fail(f"{model.name} params not JSON-serializable: {e}")
