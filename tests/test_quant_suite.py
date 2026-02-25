"""
Integration tests for Quant Suite runner.

Tests the complete end-to-end workflow:
- Load configuration
- Load data
- Run backtests on all baselines
- Generate results and reports
"""

import pytest
import pandas as pd
import numpy as np
import json
import tempfile
import shutil
from pathlib import Path
from datetime import datetime, timedelta

from engine.models.baselines import get_all_baselines
from engine.backtesting.engine import BacktestEngine


@pytest.fixture
def test_config():
    """Create test configuration."""
    return {
        "experiment_name": "Test Experiment",
        "asset": "BTC",
        "timeframe": "1H",
        "data_path": "test_data.parquet",
        "periods": {
            "train": {
                "start": "2023-01-01",
                "end": "2023-06-30",
                "description": "Train period"
            },
            "test": {
                "start": "2023-07-01",
                "end": "2023-12-31",
                "description": "Test period"
            },
            "oos": {
                "start": "2024-01-01",
                "end": "2024-06-30",
                "description": "OOS period"
            }
        },
        "regime_filter": "all",
        "costs": {
            "slippage_bps": 5,
            "fee_bps": 3,
            "total_bps": 8
        },
        "initial_capital": 10000.0,
        "acceptance_criteria": {
            "min_test_pf": 1.5,
            "min_test_sharpe": 0.5,
            "max_overfit": 0.5,
            "min_trades": 50
        },
        "baseline_config": {
            "enabled": True
        },
        "archetype_config": {
            "enabled": False
        },
        "output": {
            "results_dir": "test_results",
            "save_csv": True,
            "save_report": True,
            "save_trades": False,
            "plot_equity_curves": False
        }
    }


@pytest.fixture
def test_data():
    """Generate test OHLCV data covering all periods."""
    np.random.seed(42)

    # Generate 18 months of hourly data
    start = pd.Timestamp('2023-01-01')
    end = pd.Timestamp('2024-06-30')
    dates = pd.date_range(start=start, end=end, freq='1H')

    # Generate trending price series
    n_bars = len(dates)
    close_prices = 50000 + np.cumsum(np.random.randn(n_bars) * 50)

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


class TestQuantSuiteIntegration:
    """Integration tests for Quant Suite."""

    def test_all_baselines_run_successfully(self, test_data):
        """Test that all baselines run without errors."""
        baselines = get_all_baselines()

        for baseline_cls in baselines:
            model = baseline_cls()

            # Fit on train data
            train_data = test_data['2023-01-01':'2023-06-30']
            model.fit(train_data)

            # Run backtest
            engine = BacktestEngine(
                model=model,
                data=test_data,
                initial_capital=10000.0,
                commission_pct=0.0008
            )

            results = engine.run(start='2023-07-01', end='2023-12-31', verbose=False)

            # Validate results
            assert results.model_name == model.name
            assert results.start_date == pd.Timestamp('2023-07-01')
            assert results.end_date >= pd.Timestamp('2023-12-31')
            assert len(results.equity_curve) > 0

    def test_baseline0_has_exactly_one_trade(self, test_data):
        """Test that buy-and-hold has exactly one trade."""
        from engine.models.baselines import Baseline0_BuyAndHold

        model = Baseline0_BuyAndHold()
        model.fit(test_data)

        engine = BacktestEngine(
            model=model,
            data=test_data,
            initial_capital=10000.0,
            commission_pct=0.0008
        )

        results = engine.run(start='2023-07-01', end='2023-12-31', verbose=False)

        # Buy-and-hold should have exactly 1 trade (entry at start, exit at end)
        assert results.total_trades == 1

    def test_baseline5_has_zero_trades(self, test_data):
        """Test that cash strategy has zero trades."""
        from engine.models.baselines import Baseline5_Cash

        model = Baseline5_Cash()
        model.fit(test_data)

        engine = BacktestEngine(
            model=model,
            data=test_data,
            initial_capital=10000.0,
            commission_pct=0.0008
        )

        results = engine.run(start='2023-07-01', end='2023-12-31', verbose=False)

        # Cash should have exactly 0 trades
        assert results.total_trades == 0
        assert results.total_pnl == 0.0

    def test_baseline5_pnl_is_exactly_zero(self, test_data):
        """Test that cash strategy has exactly $0 PnL (sanity check)."""
        from engine.models.baselines import Baseline5_Cash

        model = Baseline5_Cash()
        model.fit(test_data)

        engine = BacktestEngine(
            model=model,
            data=test_data,
            initial_capital=10000.0,
            commission_pct=0.0008
        )

        results = engine.run(start='2023-07-01', end='2023-12-31', verbose=False)

        # This is critical - if cash shows any PnL, engine is broken
        assert abs(results.total_pnl) < 0.01  # Allow tiny floating point error

    def test_metrics_are_calculated(self, test_data):
        """Test that all metrics are calculated correctly."""
        from engine.models.baselines import Baseline1_SMA200Trend

        model = Baseline1_SMA200Trend()
        model.fit(test_data)

        engine = BacktestEngine(
            model=model,
            data=test_data,
            initial_capital=10000.0,
            commission_pct=0.0008
        )

        results = engine.run(start='2023-07-01', end='2023-12-31', verbose=False)

        # Check that all metrics are available
        assert hasattr(results, 'profit_factor')
        assert hasattr(results, 'win_rate')
        assert hasattr(results, 'max_drawdown')
        assert hasattr(results, 'sharpe_ratio')
        assert hasattr(results, 'total_return_pct')
        assert hasattr(results, 'avg_win')
        assert hasattr(results, 'avg_loss')

        # Metrics should be valid numbers (not NaN)
        if results.total_trades > 0:
            assert not np.isnan(results.profit_factor)
            assert not np.isnan(results.win_rate)
            assert not np.isnan(results.max_drawdown)

    def test_to_dict_serialization(self, test_data):
        """Test that results can be serialized to dict (for CSV export)."""
        from engine.models.baselines import Baseline1_SMA200Trend

        model = Baseline1_SMA200Trend()
        model.fit(test_data)

        engine = BacktestEngine(
            model=model,
            data=test_data,
            initial_capital=10000.0,
            commission_pct=0.0008
        )

        results = engine.run(start='2023-07-01', end='2023-12-31', verbose=False)

        # Convert to dict
        result_dict = results.to_dict(period='test')

        # Check required fields
        assert 'model_name' in result_dict
        assert 'period' in result_dict
        assert 'profit_factor' in result_dict
        assert 'win_rate' in result_dict
        assert 'num_trades' in result_dict
        assert 'total_pnl' in result_dict
        assert 'sharpe_ratio' in result_dict
        assert 'max_drawdown' in result_dict

        # Values should be JSON-serializable
        import json
        try:
            json.dumps(result_dict)
        except (TypeError, ValueError) as e:
            pytest.fail(f"Results dict not JSON-serializable: {e}")

    def test_multiple_periods_same_model(self, test_data):
        """Test running same model on train/test/oos periods."""
        from engine.models.baselines import Baseline1_SMA200Trend

        model = Baseline1_SMA200Trend()

        # Fit on train
        train_data = test_data['2023-01-01':'2023-06-30']
        model.fit(train_data)

        # Run on train
        engine = BacktestEngine(
            model=model,
            data=test_data,
            initial_capital=10000.0,
            commission_pct=0.0008
        )
        train_results = engine.run(start='2023-01-01', end='2023-06-30', verbose=False)

        # Run on test
        test_results = engine.run(start='2023-07-01', end='2023-12-31', verbose=False)

        # Run on OOS
        oos_results = engine.run(start='2024-01-01', end='2024-06-30', verbose=False)

        # All should complete
        assert train_results.total_trades >= 0
        assert test_results.total_trades >= 0
        assert oos_results.total_trades >= 0

    def test_costs_are_applied(self, test_data):
        """Test that transaction costs reduce PnL."""
        from engine.models.baselines import Baseline1_SMA200Trend

        model = Baseline1_SMA200Trend()
        model.fit(test_data)

        # Run with no costs
        engine_no_cost = BacktestEngine(
            model=model,
            data=test_data,
            initial_capital=10000.0,
            commission_pct=0.0
        )
        results_no_cost = engine_no_cost.run(start='2023-07-01', end='2023-12-31', verbose=False)

        # Reset model state
        model = Baseline1_SMA200Trend()
        model.fit(test_data)

        # Run with costs
        engine_with_cost = BacktestEngine(
            model=model,
            data=test_data,
            initial_capital=10000.0,
            commission_pct=0.01  # 1% costs (high)
        )
        results_with_cost = engine_with_cost.run(start='2023-07-01', end='2023-12-31', verbose=False)

        # With costs should have lower PnL (if there were trades)
        if results_no_cost.total_trades > 0:
            assert results_with_cost.total_pnl < results_no_cost.total_pnl

    def test_equity_curve_monotonic_for_buy_hold(self, test_data):
        """Test that buy-and-hold equity curve tracks price."""
        from engine.models.baselines import Baseline0_BuyAndHold

        model = Baseline0_BuyAndHold()
        model.fit(test_data)

        engine = BacktestEngine(
            model=model,
            data=test_data,
            initial_capital=10000.0,
            commission_pct=0.0
        )

        results = engine.run(start='2023-07-01', end='2023-12-31', verbose=False)

        # Equity curve should exist and have values
        assert len(results.equity_curve) > 0

        # Final equity should reflect price change
        test_period = test_data['2023-07-01':'2023-12-31']
        price_start = test_period.iloc[0]['close']
        price_end = test_period.iloc[-1]['close']
        price_change_pct = (price_end - price_start) / price_start

        # Equity should change roughly the same (allowing for position sizing)
        # This is a rough check, not exact
        final_equity = results.equity_curve.iloc[-1]
        equity_change_pct = (final_equity - 10000.0) / 10000.0

        # Should be same sign at minimum
        assert np.sign(price_change_pct) == np.sign(equity_change_pct)


class TestQuantSuiteFileOutput:
    """Test file output generation."""

    def test_config_file_creation(self, test_config):
        """Test that config file can be created and loaded."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / 'test_config.json'

            # Write config
            with open(config_path, 'w') as f:
                json.dump(test_config, f, indent=2)

            # Read config
            with open(config_path, 'r') as f:
                loaded_config = json.load(f)

            assert loaded_config == test_config

    def test_results_csv_creation(self, test_data):
        """Test that results can be exported to CSV."""
        from engine.models.baselines import Baseline1_SMA200Trend

        model = Baseline1_SMA200Trend()
        model.fit(test_data)

        engine = BacktestEngine(
            model=model,
            data=test_data,
            initial_capital=10000.0,
            commission_pct=0.0008
        )

        results = engine.run(start='2023-07-01', end='2023-12-31', verbose=False)

        # Create DataFrame from results
        df = pd.DataFrame([results.to_dict(period='test')])

        # Save to temp CSV
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = Path(tmpdir) / 'results.csv'
            df.to_csv(csv_path, index=False)

            # Read back
            loaded_df = pd.read_csv(csv_path)

            assert len(loaded_df) == 1
            assert loaded_df['model_name'][0] == model.name
            assert loaded_df['period'][0] == 'test'


def test_baselines_complete_workflow(test_data):
    """
    Test complete workflow for all baselines:
    1. Fit on train
    2. Run on test
    3. Generate metrics
    4. Export to dict
    """
    baselines = get_all_baselines()

    for baseline_cls in baselines:
        # Initialize
        model = baseline_cls()

        # Fit on train
        train_data = test_data['2023-01-01':'2023-06-30']
        model.fit(train_data)

        # Run backtest
        engine = BacktestEngine(
            model=model,
            data=test_data,
            initial_capital=10000.0,
            commission_pct=0.0008
        )

        results = engine.run(start='2023-07-01', end='2023-12-31', verbose=False)

        # Generate dict
        result_dict = results.to_dict(period='test')

        # Validate
        assert result_dict['model_name'] == model.name
        assert result_dict['period'] == 'test'
        assert 'profit_factor' in result_dict
        assert 'num_trades' in result_dict

        print(f"✓ {model.name}: {result_dict['num_trades']} trades, PF={result_dict['profit_factor']:.2f}")
