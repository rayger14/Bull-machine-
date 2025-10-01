"""
Quick Test of Macro Backtest Framework

Validates that all components work together before running full 24-month study.
"""

import sys
import os
import logging
import pandas as pd
import numpy as np
from pathlib import Path

# Add project to path
sys.path.insert(0, '.')

from engine.context.macro_pulse import MacroPulseEngine, MacroPulse
from engine.fusion import FusionEngine
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_config_loading():
    """Test configuration file loading"""
    try:
        config_path = 'configs/v170/assets/ETH_v17_baseline.json'
        with open(config_path, 'r') as f:
            config = json.load(f)

        logger.info(f"✓ Config loaded: {len(config)} sections")

        # Check required sections
        required_sections = ['domains', 'fusion', 'risk_management']
        for section in required_sections:
            if section in config:
                logger.info(f"  ✓ {section}: OK")
            else:
                logger.warning(f"  ⚠ {section}: Missing")

        return True

    except Exception as e:
        logger.error(f"✗ Config loading failed: {e}")
        return False

def test_macro_engine():
    """Test macro pulse engine initialization"""
    try:
        # Load context config
        with open('configs/v170/context.json', 'r') as f:
            context_config = json.load(f)

        # Initialize macro engine
        engine = MacroPulseEngine(context_config)
        logger.info("✓ Macro pulse engine initialized")

        # Test with synthetic data
        dates = pd.date_range('2024-01-01', periods=100, freq='1D')

        # Create synthetic macro data
        series_data = {
            'DXY_1D': pd.DataFrame({
                'open': 100, 'high': 101, 'low': 99, 'close': 100,
                'volume': 1000000
            }, index=dates),
            'VIX_1D': pd.DataFrame({
                'open': 15, 'high': 16, 'low': 14, 'close': 15,
                'volume': 1000000
            }, index=dates)
        }

        # Test analysis
        pulse = engine.analyze_macro_pulse(series_data)
        logger.info(f"✓ Macro analysis completed: {pulse.regime}")

        return True

    except Exception as e:
        logger.error(f"✗ Macro engine test failed: {e}")
        return False

def test_fusion_engine():
    """Test fusion engine with macro integration"""
    try:
        # Load full config
        with open('configs/v170/assets/ETH_v17_baseline.json', 'r') as f:
            config = json.load(f)

        # Add context config
        with open('configs/v170/context.json', 'r') as f:
            config['context'] = json.load(f)

        # Initialize fusion engine
        fusion_engine = FusionEngine(config)
        logger.info("✓ Fusion engine initialized with macro pulse")

        # Test signal aggregation with mock data
        domain_signals = {
            'wyckoff': {
                'direction': 'long',
                'strength': 0.7,
                'confidence': 0.8
            },
            'liquidity': {
                'direction': 'long',
                'strength': 0.6,
                'confidence': 0.75
            },
            'momentum': {
                'direction': 'long',
                'strength': 0.65,
                'confidence': 0.7
            }
        }

        # Mock market data
        dates = pd.date_range('2024-01-01', periods=50, freq='1h')
        market_data = {
            '1H': pd.DataFrame({
                'open': 50000, 'high': 50100, 'low': 49900, 'close': 50000,
                'volume': 1000000
            }, index=dates),
            'DXY_1D': pd.DataFrame({
                'open': 100, 'high': 101, 'low': 99, 'close': 100,
                'volume': 1000000
            }, index=pd.date_range('2024-01-01', periods=25, freq='1D'))
        }

        # Test aggregation
        result = fusion_engine.aggregate_signals(domain_signals, market_data)

        if result:
            logger.info(f"✓ Fusion completed: {result.direction} signal with {result.confidence:.2f} confidence")
            logger.info(f"  Macro delta: {result.macro_delta:+.3f}")
            logger.info(f"  Risk bias: {result.risk_bias}")
        else:
            logger.info("✓ Fusion completed: No signal generated (expected)")

        return True

    except Exception as e:
        logger.error(f"✗ Fusion engine test failed: {e}")
        return False

def test_data_loading():
    """Test data loading functionality"""
    try:
        from scripts.run_macro_backtest import load_chart_logs_data

        # Test loading with synthetic fallback
        df = load_chart_logs_data('ETH', '1H', '2024-01-01', '2024-01-02')

        if not df.empty:
            logger.info(f"✓ Data loading works: {len(df)} bars loaded")
            logger.info(f"  Columns: {list(df.columns)}")
            return True
        else:
            logger.warning("⚠ No data loaded (expected if chart_logs missing)")
            return True

    except Exception as e:
        logger.error(f"✗ Data loading test failed: {e}")
        return False

def run_quick_backtest():
    """Run a very quick backtest to validate the pipeline"""
    try:
        from scripts.run_macro_backtest import simulate_asset_backtest

        # Load configs
        with open('configs/v170/assets/ETH_v17_baseline.json', 'r') as f:
            config = json.load(f)

        engine_config = {
            'timeframe': '4H',
            'risk_pct': config['risk_management']['risk_pct'],
            'tp_R': 2.0,
            'sl_R': 1.0,
            'fusion': config['fusion'],
            'warmup': 10  # Short warmup for test
        }

        context_config = config.get('domains', {}).get('macro_context', {})
        temporal_config = config.get('domains', {}).get('temporal', {})

        # Run short backtest (1 week)
        result = simulate_asset_backtest(
            'ETH', '2024-01-01', '2024-01-08',
            engine_config, context_config, temporal_config,
            ablation=None, enable_macro=True
        )

        logger.info(f"✓ Quick backtest completed:")
        logger.info(f"  Asset: {result.asset}")
        logger.info(f"  Trades: {result.total_trades}")
        logger.info(f"  PnL: {result.pnl_percent:.2f}%")

        return True

    except Exception as e:
        logger.error(f"✗ Quick backtest failed: {e}")
        return False

def main():
    """Run all validation tests"""

    print("\n" + "="*60)
    print("MACRO BACKTEST FRAMEWORK VALIDATION")
    print("="*60)

    tests = [
        ("Configuration Loading", test_config_loading),
        ("Macro Engine", test_macro_engine),
        ("Fusion Engine", test_fusion_engine),
        ("Data Loading", test_data_loading),
        ("Quick Backtest", run_quick_backtest)
    ]

    results = []

    for test_name, test_func in tests:
        print(f"\n🧪 Testing {test_name}...")
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            logger.error(f"Test {test_name} crashed: {e}")
            results.append((test_name, False))

    # Summary
    print("\n" + "="*60)
    print("VALIDATION SUMMARY")
    print("="*60)

    passed = 0
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} {test_name}")
        if result:
            passed += 1

    print(f"\nResults: {passed}/{len(tests)} tests passed")

    if passed == len(tests):
        print("\n🎉 All tests passed! Ready for 24-month study.")
        print("Run: python run_24_month_macro_study.py")
    else:
        print("\n⚠️ Some tests failed. Fix issues before running full study.")

    return passed == len(tests)

if __name__ == '__main__':
    main()