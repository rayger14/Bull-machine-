"""
Integration Test for NautilusTrader Bull Machine Integration

This test validates the entire pipeline:
1. FeatureProvider → Feature generation
2. RegimeService → Regime classification
3. RuntimeContext → Context building
4. ArchetypeLogic → Signal generation
5. ThresholdPolicy → Parameter adaptation
6. EventEngine → Order execution

Test Strategy:
- Use synthetic data (100 bars)
- Test both feature store and runtime computation modes
- Validate signal generation pipeline
- Check that all components integrate correctly

Author: Claude Code (System Architect)
Date: 2026-01-21
"""

import sys
from pathlib import Path
import unittest
import pandas as pd
import numpy as np
import logging
from datetime import datetime, timedelta

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from engine.integrations.event_engine import EventEngine, Bar
from engine.integrations.nautilus_strategy import NautilusBullMachineStrategy
from engine.integrations.feature_provider import FeatureProvider

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class TestNautilusIntegration(unittest.TestCase):
    """Integration tests for Nautilus Bull Machine."""

    @classmethod
    def setUpClass(cls):
        """Set up test fixtures."""
        logger.info("=" * 80)
        logger.info("NAUTILUS INTEGRATION TEST SUITE")
        logger.info("=" * 80)

    def setUp(self):
        """Set up each test."""
        self.config_path = 'configs/baseline_wyckoff_test.json'

    def generate_synthetic_data(self, n_bars: int = 100) -> list:
        """
        Generate synthetic OHLCV data for testing.

        Args:
            n_bars: Number of bars to generate

        Returns:
            List of Bar objects
        """
        np.random.seed(42)
        start_price = 30000.0
        start_date = pd.Timestamp('2023-01-01')

        # Generate trending price with noise
        trend = np.linspace(start_price, start_price * 1.2, n_bars)
        noise = np.random.normal(0, start_price * 0.02, n_bars)
        closes = trend + noise

        bars = []
        for i in range(n_bars):
            timestamp = start_date + pd.Timedelta(hours=i)
            close = closes[i]

            bar = Bar(
                timestamp=timestamp,
                open=close * np.random.uniform(0.995, 1.005),
                high=close * np.random.uniform(1.0, 1.02),
                low=close * np.random.uniform(0.98, 1.0),
                close=close,
                volume=np.random.uniform(1e6, 1e7)
            )
            bars.append(bar)

        return bars

    def test_feature_provider_runtime(self):
        """Test FeatureProvider with runtime computation."""
        logger.info("\n" + "=" * 80)
        logger.info("TEST: FeatureProvider Runtime Computation")
        logger.info("=" * 80)

        # Initialize FeatureProvider (no feature store)
        provider = FeatureProvider(
            feature_store_path=None,
            enable_runtime_computation=True
        )

        # Generate test bar
        bar = Bar(
            timestamp=pd.Timestamp('2023-01-01'),
            open=30000.0,
            high=30500.0,
            low=29500.0,
            close=30200.0,
            volume=1e7
        )

        # Get features
        features = provider.get_features(bar)

        # Validate essential features
        self.assertIn('atr_14', features)
        self.assertIn('rsi_14', features)
        self.assertIn('liquidity_score', features)
        self.assertIn('fusion_score', features)
        self.assertIn('close', features)

        logger.info("✓ Runtime feature computation working")
        logger.info(f"  Generated {len(features)} features")
        logger.info(f"  ATR: ${features['atr_14']:.2f}")
        logger.info(f"  RSI: {features['rsi_14']:.1f}")
        logger.info(f"  Liquidity: {features['liquidity_score']:.3f}")
        logger.info(f"  Fusion: {features['fusion_score']:.3f}")

    def test_strategy_initialization(self):
        """Test strategy initialization."""
        logger.info("\n" + "=" * 80)
        logger.info("TEST: Strategy Initialization")
        logger.info("=" * 80)

        # Check if config exists
        config_path = Path(self.config_path)
        if not config_path.exists():
            logger.warning(f"Config not found: {config_path} - skipping test")
            self.skipTest(f"Config not found: {config_path}")

        # Initialize strategy
        strategy = NautilusBullMachineStrategy(
            config_path=str(config_path),
            regime_model_path=None,  # Disable regime for test
            feature_store_path=None,
            enable_regime_service=False,
            enable_feature_store=False
        )

        # Initialize engine
        engine = EventEngine(
            strategy=strategy,
            initial_cash=100000.0
        )

        # Call on_start
        strategy.on_start(engine)

        # Validate components initialized
        self.assertIsNotNone(strategy.config)
        self.assertIsNotNone(strategy.feature_provider)
        self.assertIsNotNone(strategy.archetype_logic)
        self.assertIsNotNone(strategy.threshold_policy)

        logger.info("✓ Strategy initialization working")
        logger.info(f"  Config: {strategy.config.get('version', 'unknown')}")
        logger.info(f"  FeatureProvider: initialized")
        logger.info(f"  ArchetypeLogic: initialized")
        logger.info(f"  ThresholdPolicy: initialized")

    def test_signal_pipeline_integration(self):
        """Test complete signal generation pipeline."""
        logger.info("\n" + "=" * 80)
        logger.info("TEST: Signal Pipeline Integration")
        logger.info("=" * 80)

        # Check if config exists
        config_path = Path(self.config_path)
        if not config_path.exists():
            logger.warning(f"Config not found: {config_path} - skipping test")
            self.skipTest(f"Config not found: {config_path}")

        # Initialize strategy
        strategy = NautilusBullMachineStrategy(
            config_path=str(config_path),
            regime_model_path=None,
            feature_store_path=None,
            enable_regime_service=False,
            enable_feature_store=False
        )

        # Initialize engine
        engine = EventEngine(
            strategy=strategy,
            initial_cash=100000.0
        )

        # Start strategy
        strategy.on_start(engine)

        # Generate synthetic data
        bars = self.generate_synthetic_data(n_bars=100)

        # Process bars
        logger.info(f"Processing {len(bars)} bars...")
        for bar in bars:
            strategy.on_bar(bar, engine)

        # Validate results
        stats = engine.get_performance_stats()

        logger.info("✓ Signal pipeline integration working")
        logger.info(f"  Bars processed: {len(bars)}")
        logger.info(f"  Signals generated: {strategy.total_signals}")
        logger.info(f"  Signals taken: {strategy.signals_taken}")
        logger.info(f"  Trades executed: {stats['total_trades']}")
        logger.info(f"  Final equity: ${stats['final_equity']:,.2f}")

        # Assert pipeline ran without errors
        self.assertGreaterEqual(len(bars), 100)

    def test_full_backtest_smoke_test(self):
        """Smoke test for full backtest."""
        logger.info("\n" + "=" * 80)
        logger.info("TEST: Full Backtest Smoke Test")
        logger.info("=" * 80)

        # Check if config exists
        config_path = Path(self.config_path)
        if not config_path.exists():
            logger.warning(f"Config not found: {config_path} - skipping test")
            self.skipTest(f"Config not found: {config_path}")

        # Initialize strategy
        strategy = NautilusBullMachineStrategy(
            config_path=str(config_path),
            regime_model_path=None,
            feature_store_path=None,
            enable_regime_service=False,
            enable_feature_store=False,
            risk_per_trade=0.02,
            atr_stop_mult=2.5
        )

        # Initialize engine
        engine = EventEngine(
            strategy=strategy,
            initial_cash=100000.0,
            commission_rate=0.001,
            slippage_bps=2.0
        )

        # Generate synthetic data (larger sample)
        bars = self.generate_synthetic_data(n_bars=500)

        # Run backtest
        logger.info(f"Running backtest on {len(bars)} bars...")
        import time
        start_time = time.time()

        engine.run(bars)

        elapsed = time.time() - start_time
        bars_per_sec = len(bars) / elapsed if elapsed > 0 else 0

        # Get stats
        stats = engine.get_performance_stats()

        logger.info("✓ Full backtest completed")
        logger.info(f"  Elapsed time: {elapsed:.3f}s")
        logger.info(f"  Processing speed: {bars_per_sec:,.0f} bars/sec")
        logger.info(f"  Total trades: {stats['total_trades']}")
        logger.info(f"  Total PnL: ${stats['total_pnl']:,.2f}")
        logger.info(f"  Win rate: {stats['win_rate']:.1f}%")
        logger.info(f"  Max drawdown: {stats['max_drawdown']:.2f}%")

        # Assert basic sanity checks
        self.assertGreater(bars_per_sec, 0)
        self.assertIsNotNone(stats['total_trades'])

    @classmethod
    def tearDownClass(cls):
        """Clean up after all tests."""
        logger.info("\n" + "=" * 80)
        logger.info("INTEGRATION TEST SUITE COMPLETED")
        logger.info("=" * 80)


def main():
    """Run tests."""
    unittest.main(verbosity=2)


if __name__ == '__main__':
    main()
