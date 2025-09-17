#!/usr/bin/env python3
"""
Comprehensive test suite for Bull Machine v1.2.1
Tests all 6 confluence layers and integration
"""

import json
import os
import sys
import tempfile
from unittest.mock import Mock, patch

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bull_machine.core.types import Series, Bar, WyckoffResult, Signal, RiskPlan
from bull_machine.modules.fusion.advanced import AdvancedFusionEngine
from bull_machine.modules.liquidity.advanced import AdvancedLiquidityAnalyzer
from bull_machine.modules.risk.advanced import AdvancedRiskManager
from bull_machine.modules.structure.advanced import AdvancedStructureAnalyzer
from bull_machine.modules.momentum.advanced import AdvancedMomentumAnalyzer
from bull_machine.modules.volume.advanced import AdvancedVolumeAnalyzer
from bull_machine.modules.context.advanced import AdvancedContextAnalyzer
from bull_machine.app.main import run_bull_machine_v1_2_1

def create_test_series(num_bars=100):
    """Create test series with realistic price data"""
    bars = []
    base_price = 50000.0
    for i in range(num_bars):
        # Simple trending data
        trend = i * 10
        open_price = base_price + trend + (i % 5) * 5
        high = open_price + 50 + (i % 3) * 20
        low = open_price - 50 - (i % 4) * 15
        close = open_price + (i % 7 - 3) * 10
        volume = 1000 + (i % 100) * 10

        bars.append(Bar(
            ts=1640000000 + i * 3600,  # Hourly bars
            open=open_price,
            high=high,
            low=low,
            close=close,
            volume=volume
        ))

    return Series(bars=bars, timeframe="1H", symbol="BTCUSD")

def create_test_wyckoff():
    """Create test Wyckoff result"""
    return WyckoffResult(
        regime='accumulation',
        phase='C',
        bias='long',
        phase_confidence=0.75,
        trend_confidence=0.8,
        range=None
    )

def test_config_loads():
    """Test v1.2.1 config loads correctly"""
    try:
        config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                                  'bull_machine', 'config', 'config_v1_2_1.json')
        with open(config_path, 'r') as f:
            config = json.load(f)

        assert config['version'] == '1.2.1'
        assert 'fusion' in config
        assert 'enter_threshold' in config['fusion']
        assert config['fusion']['enter_threshold'] == 0.35
        print("✅ Config loads and has correct structure")
        return True
    except Exception as e:
        print(f"❌ Config test failed: {e}")
        return False

def test_weights_sum():
    """Test fusion weights sum to 1.0"""
    try:
        config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                                  'bull_machine', 'config', 'config_v1_2_1.json')
        with open(config_path, 'r') as f:
            config = json.load(f)

        weights = config['fusion']['weights']
        total = sum(weights.values())

        assert abs(total - 1.0) < 0.01, f"Weights sum to {total}, not 1.0"
        print(f"✅ Weights sum correctly: {total:.3f}")
        return True
    except Exception as e:
        print(f"❌ Weights test failed: {e}")
        return False

def test_all_modules_import():
    """Test all 6 modules can be imported and instantiated"""
    try:
        config = {
            'fusion': {'weights': {'wyckoff': 0.3, 'liquidity': 0.25, 'structure': 0.2,
                                  'momentum': 0.1, 'volume': 0.1, 'context': 0.05}},
            'liquidity': {'tick_size': 0.01},
            'risk': {'account_risk_percent': 1.0}
        }

        # Test all modules can be instantiated
        fusion = AdvancedFusionEngine(config)
        liquidity = AdvancedLiquidityAnalyzer(config)
        risk = AdvancedRiskManager(config)
        structure = AdvancedStructureAnalyzer(config)
        momentum = AdvancedMomentumAnalyzer(config)
        volume = AdvancedVolumeAnalyzer(config)
        context = AdvancedContextAnalyzer(config)

        print("✅ All 6 modules import and instantiate successfully")
        return True
    except Exception as e:
        print(f"❌ Module import test failed: {e}")
        return False

def test_module_analysis():
    """Test each module can run analysis"""
    try:
        config = {
            'fusion': {'weights': {'wyckoff': 0.3, 'liquidity': 0.25, 'structure': 0.2,
                                  'momentum': 0.1, 'volume': 0.1, 'context': 0.05}},
            'liquidity': {'tick_size': 0.01, 'sweep_recent_bars': 5},
            'momentum': {'rsi_period': 14},
            'volume': {'sma_period': 20}
        }

        series = create_test_series()
        wyckoff = create_test_wyckoff()

        # Test each module analysis
        liquidity = AdvancedLiquidityAnalyzer(config)
        lres = liquidity.analyze(series, wyckoff)
        assert 'overall_score' in lres

        structure = AdvancedStructureAnalyzer(config)
        sres = structure.analyze(series, wyckoff)
        assert 'bos_strength' in sres

        momentum = AdvancedMomentumAnalyzer(config)
        mres = momentum.analyze(series, wyckoff)
        assert 'score' in mres

        volume = AdvancedVolumeAnalyzer(config)
        vres = volume.analyze(series, wyckoff)
        assert 'score' in vres

        context = AdvancedContextAnalyzer(config)
        cres = context.analyze(series, wyckoff)
        assert 'score' in cres

        print("✅ All modules can run analysis and return expected structure")
        return True
    except Exception as e:
        print(f"❌ Module analysis test failed: {e}")
        return False

def test_fusion_score_calculation():
    """Test fusion engine calculates scores correctly"""
    try:
        config = {
            'fusion': {
                'weights': {'wyckoff': 0.3, 'liquidity': 0.25, 'structure': 0.2,
                           'momentum': 0.1, 'volume': 0.1, 'context': 0.05},
                'enter_threshold': 0.35
            }
        }

        fusion = AdvancedFusionEngine(config)

        # Create test modules data
        wyckoff = WyckoffResult(
            regime='accumulation', phase='C', bias='long',
            phase_confidence=0.7, trend_confidence=0.8, range=None
        )

        modules_data = {
            'wyckoff': wyckoff,
            'liquidity': {'overall_score': 0.6},
            'structure': {'bos_strength': 0.5},
            'momentum': {'score': 0.4},
            'volume': {'score': 0.3},
            'context': {'score': 0.2}
        }

        scores = fusion._calculate_module_scores(modules_data)

        # Test Wyckoff score calculation (should be combined confidence)
        expected_wy = (0.7 + 0.8) / 2
        assert abs(scores['wyckoff'] - expected_wy) < 0.01

        # Test fusion score calculation
        fusion_score = fusion._calculate_fusion_score(scores)
        expected = (expected_wy * 0.3 + 0.6 * 0.25 + 0.5 * 0.2 +
                   0.4 * 0.1 + 0.3 * 0.1 + 0.2 * 0.05)
        assert abs(fusion_score - expected) < 0.01

        print(f"✅ Fusion score calculation correct: {fusion_score:.3f}")
        return True
    except Exception as e:
        print(f"❌ Fusion score test failed: {e}")
        return False

def test_veto_system():
    """Test veto system works correctly"""
    try:
        config = {
            'features': {'veto_system': True, 'trend_filter': True},
            'fusion': {
                'weights': {'wyckoff': 0.3, 'liquidity': 0.25, 'structure': 0.2,
                           'momentum': 0.1, 'volume': 0.1, 'context': 0.05},
                'enter_threshold': 0.35,
                'volatility_shock_sigma': 4.0
            }
        }

        fusion = AdvancedFusionEngine(config)

        # Test early phase veto (Phase A with low confidence)
        wyckoff_early = WyckoffResult(
            regime='accumulation', phase='A', bias='long',
            phase_confidence=0.5, trend_confidence=0.6, range=None
        )

        modules_data = {
            'wyckoff': wyckoff_early,
            'liquidity': {'overall_score': 0.3, 'sweeps': []},
            'series': create_test_series()
        }

        vetoes = fusion._check_vetoes(modules_data)
        assert 'early_wyckoff_phase' in vetoes

        print("✅ Veto system correctly blocks early phase trades")
        return True
    except Exception as e:
        print(f"❌ Veto system test failed: {e}")
        return False

def test_risk_manager():
    """Test risk manager calculates proper trade plans"""
    try:
        config = {
            'risk': {
                'account_risk_percent': 1.0,
                'max_risk_percent': 1.25,
                'max_risk_per_trade': 150.0,
                'stop': {
                    'method': 'swing_with_atr_guardrail',
                    'atr_mult': 3.0,
                    'swing_buffer': 0.001,
                    'volatility_scaling': True,
                    'target_volatility': 0.012
                }
            }
        }

        risk_manager = AdvancedRiskManager(config)
        series = create_test_series()
        signal = Signal(ts=0, side='long', confidence=0.75, reasons=['test'], ttl_bars=20)

        plan = risk_manager.plan_trade(series, signal, 10000)

        assert hasattr(plan, 'entry')
        assert hasattr(plan, 'stop')
        assert hasattr(plan, 'size')
        assert hasattr(plan, 'tp_levels')
        assert plan.size > 0
        assert len(plan.tp_levels) == 3  # tp1, tp2, tp3

        print(f"✅ Risk manager generates valid trade plan")
        print(f"   Entry: {plan.entry:.2f}, Stop: {plan.stop:.2f}, Size: {plan.size:.4f}")
        return True
    except Exception as e:
        print(f"❌ Risk manager test failed: {e}")
        return False

def test_full_integration():
    """Test full v1.2.1 integration with temporary CSV"""
    try:
        # Create temporary CSV file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            f.write("timestamp,open,high,low,close,volume\n")
            base_ts = 1640000000
            base_price = 50000.0

            for i in range(100):
                ts = base_ts + i * 3600
                price = base_price + i * 10 + (i % 5) * 20
                open_p = price
                high = price + 100
                low = price - 80
                close = price + (i % 3 - 1) * 30
                volume = 1000

                f.write(f"{ts},{open_p},{high},{low},{close},{volume}\n")

            temp_file = f.name

        try:
            # Test with lowered threshold to ensure signal generation
            result = run_bull_machine_v1_2_1(
                temp_file,
                account_balance=10000,
                override_signals={'enter_threshold': 0.25}  # Lower threshold
            )

            # Should generate either trade or no_trade, not error
            assert result['action'] in ['enter_trade', 'no_trade']
            assert result['version'] == '1.2.1'

            if result['action'] == 'enter_trade':
                assert 'signal' in result
                assert 'risk_plan' in result
                print("✅ Full integration test: Trade generated")
            else:
                print(f"✅ Full integration test: No trade ({result.get('reason', 'unknown')})")

            return True

        finally:
            os.unlink(temp_file)

    except Exception as e:
        print(f"❌ Full integration test failed: {e}")
        return False

def test_score_bounds():
    """Test all module scores are within 0-1 bounds"""
    try:
        config = {
            'fusion': {'weights': {'wyckoff': 0.3, 'liquidity': 0.25, 'structure': 0.2,
                                  'momentum': 0.1, 'volume': 0.1, 'context': 0.05}},
            'liquidity': {'tick_size': 0.01},
            'momentum': {'rsi_period': 14},
            'volume': {'sma_period': 20}
        }

        series = create_test_series()
        wyckoff = create_test_wyckoff()

        # Test all modules return scores in valid range
        modules = [
            ('liquidity', AdvancedLiquidityAnalyzer(config)),
            ('structure', AdvancedStructureAnalyzer(config)),
            ('momentum', AdvancedMomentumAnalyzer(config)),
            ('volume', AdvancedVolumeAnalyzer(config)),
            ('context', AdvancedContextAnalyzer(config))
        ]

        for name, module in modules:
            result = module.analyze(series, wyckoff)

            if name == 'liquidity':
                score = result.get('overall_score', 0)
            else:
                score = result.get('score', 0)

            assert 0.0 <= score <= 1.0, f"{name} score {score} out of bounds"
            print(f"   {name}: {score:.3f}")

        print("✅ All module scores within valid bounds")
        return True
    except Exception as e:
        print(f"❌ Score bounds test failed: {e}")
        return False

def main():
    """Run all comprehensive tests"""
    print("=" * 80)
    print("BULL MACHINE v1.2.1 - COMPREHENSIVE TEST SUITE")
    print("=" * 80)

    tests = [
        ("Config Loading", test_config_loads),
        ("Weights Validation", test_weights_sum),
        ("Module Imports", test_all_modules_import),
        ("Module Analysis", test_module_analysis),
        ("Fusion Calculation", test_fusion_score_calculation),
        ("Veto System", test_veto_system),
        ("Risk Manager", test_risk_manager),
        ("Score Bounds", test_score_bounds),
        ("Full Integration", test_full_integration)
    ]

    passed = 0
    failed = 0

    for test_name, test_func in tests:
        print(f"\n🧪 Running: {test_name}")
        print("-" * 40)

        try:
            if test_func():
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"❌ {test_name} crashed: {e}")
            failed += 1

    print("\n" + "=" * 80)
    print("TEST RESULTS")
    print("=" * 80)
    print(f"✅ Passed: {passed}")
    print(f"❌ Failed: {failed}")
    print(f"📊 Total:  {passed + failed}")

    if failed == 0:
        print("\n🎉 ALL TESTS PASSED! v1.2.1 ready for production")
        return 0
    else:
        print(f"\n⚠️  {failed} test(s) failed - fix before production")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)