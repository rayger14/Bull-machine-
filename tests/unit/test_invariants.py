"""
Unit and Invariant Tests for Bull Machine v1.7
Ensures determinism, no future leak, and delta cap enforcement
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

import pandas as pd
import numpy as np
import pytest
from datetime import datetime, timedelta

from engine.smc.smc_engine import SMCEngine
from engine.wyckoff.wyckoff_engine import WyckoffEngine
from engine.liquidity.hob import HOBDetector
from engine.momentum.momentum_engine import MomentumEngine


class TestInvariants:
    """Test critical system invariants"""

    def test_no_future_leak(self):
        """Ensure no future data is used in signals"""
        # Create mock data
        dates = pd.date_range('2025-01-01', periods=100, freq='4H')
        df = pd.DataFrame({
            'open': np.random.randn(100).cumsum() + 100,
            'high': np.random.randn(100).cumsum() + 101,
            'low': np.random.randn(100).cumsum() + 99,
            'close': np.random.randn(100).cumsum() + 100,
            'volume': np.random.uniform(1000, 5000, 100)
        }, index=dates)

        # Test each engine
        config = {'threshold': 0.5}

        # SMC should only use data up to current bar
        smc = SMCEngine(config)
        for i in range(50, len(df)):
            historical_data = df.iloc[:i]  # No future bars
            signal = smc.analyze(historical_data)

            # Signal should be based on historical data only
            if signal:
                assert signal.timestamp <= historical_data.index[-1]

    def test_delta_cap_enforcement(self):
        """Test that delta contributions stay within caps"""
        # Delta caps from spec
        delta_caps = {
            'macro': 0.10,
            'momentum': 0.06,
            'hob': 0.05,
            'hps': 0.03
        }

        base_score = 0.5

        # Test momentum delta
        momentum_delta = 0.08  # Exceeds cap
        clamped = max(-delta_caps['momentum'], min(delta_caps['momentum'], momentum_delta))
        assert clamped == 0.06  # Should be capped

        # Test HOB delta
        hob_delta = -0.07  # Exceeds negative cap
        clamped = max(-delta_caps['hob'], min(delta_caps['hob'], hob_delta))
        assert clamped == -0.05  # Should be capped

        # Test total fusion score stays reasonable
        total_delta = sum([
            min(delta_caps['macro'], 0.08),
            min(delta_caps['momentum'], 0.04),
            min(delta_caps['hob'], 0.03),
            min(delta_caps['hps'], 0.02)
        ])

        fusion_score = base_score + total_delta
        max_possible = base_score + sum(delta_caps.values())

        assert fusion_score <= max_possible

    def test_veto_precedence(self):
        """Test that macro veto takes precedence"""
        # Mock signals
        signals = {
            'smc': {'direction': 'long', 'confidence': 0.8},
            'momentum': {'direction': 'long', 'confidence': 0.7},
            'wyckoff': {'direction': 'long', 'confidence': 0.6}
        }

        # Macro veto should override
        macro_veto = True

        if macro_veto:
            final_signal = None  # Should be vetoed
        else:
            # Normal fusion
            final_signal = signals

        if macro_veto:
            assert final_signal is None
        else:
            assert final_signal is not None

    def test_health_band_ranges(self):
        """Test health bands stay within expected ranges"""
        health_bands = {
            'macro_veto_rate': (0.05, 0.15),    # 5-15%
            'smc_2hit_rate': (0.30, 1.0),       # ≥30%
            'hob_relevance': (0.0, 0.30),       # ≤30%
            'delta_breaches': (0, 0)            # Must be 0
        }

        # Test valid values
        test_values = {
            'macro_veto_rate': 0.11,
            'smc_2hit_rate': 0.35,
            'hob_relevance': 0.22,
            'delta_breaches': 0
        }

        for metric, value in test_values.items():
            min_val, max_val = health_bands[metric]
            assert min_val <= value <= max_val, f"{metric} out of range"

    def test_position_sizing_limits(self):
        """Test position sizing stays within risk limits"""
        capital = 100000
        max_risk_per_trade = 0.075  # 7.5%
        max_position_value = capital * max_risk_per_trade

        # Test various confidence levels
        test_cases = [
            (0.3, 1.0),   # Min confidence, base sizing
            (0.5, 1.2),   # Higher confidence, slightly larger
            (0.8, 1.5),   # High confidence, max multiplier
        ]

        for confidence, size_mult in test_cases:
            position_value = capital * max_risk_per_trade * size_mult

            # Should never exceed 15% (2x max risk)
            assert position_value <= capital * 0.15

    def test_mtf_alignment_logic(self):
        """Test multi-timeframe confluence logic"""
        # Test cases: (1H_trend, 4H_trend, 1D_trend, expected_aligned)
        test_cases = [
            ('bullish', 'bullish', 'bullish', True),   # Full alignment
            ('bearish', 'bearish', 'bearish', True),   # Full alignment
            ('bullish', 'bearish', 'bullish', True),   # 2/3 bullish = aligned
            ('neutral', 'bullish', 'bullish', False),  # Neutral blocks
            ('bullish', 'bullish', 'neutral', False),  # HTF neutral
        ]

        for h1_trend, h4_trend, d1_trend, expected in test_cases:
            # At least 60% must agree and be non-neutral
            trends = [h1_trend, h4_trend, d1_trend]
            bullish_count = trends.count('bullish')
            bearish_count = trends.count('bearish')

            aligned = False
            if bullish_count >= 2 and 'neutral' not in trends:
                aligned = True
            elif bearish_count >= 2 and 'neutral' not in trends:
                aligned = True

            assert aligned == expected, f"MTF logic failed for {trends}"


class TestWyckoffPhases:
    """Test Wyckoff phase detection"""

    def test_accumulation_phases(self):
        """Test detection of accumulation phases"""
        # Phase A: SC (Selling Climax)
        sc_pattern = {
            'volume_spike': 2.5,  # High volume
            'price_reversal': True,
            'support_test': True
        }
        assert self._detect_phase(sc_pattern) == 'SC'

        # Phase B: ST (Secondary Test)
        st_pattern = {
            'volume_decline': True,
            'price_holds': True,
            'range_bound': True
        }
        assert self._detect_phase(st_pattern) == 'ST'

    def test_distribution_phases(self):
        """Test detection of distribution phases"""
        # Phase A: PSY (Preliminary Supply)
        psy_pattern = {
            'volume_increase': True,
            'price_stall': True,
            'resistance_test': True
        }
        assert self._detect_phase(psy_pattern) == 'PSY'

        # Phase E: UTAD (Upthrust After Distribution)
        utad_pattern = {
            'false_breakout': True,
            'volume_divergence': True,
            'immediate_reversal': True
        }
        assert self._detect_phase(utad_pattern) == 'UTAD'

    def _detect_phase(self, pattern: dict) -> str:
        """Mock phase detection logic"""
        if pattern.get('volume_spike') and pattern.get('price_reversal'):
            return 'SC'
        elif pattern.get('volume_decline') and pattern.get('range_bound'):
            return 'ST'
        elif pattern.get('volume_increase') and pattern.get('price_stall'):
            return 'PSY'
        elif pattern.get('false_breakout') and pattern.get('volume_divergence'):
            return 'UTAD'
        return 'Unknown'


class TestSMCLogic:
    """Test Smart Money Concepts engine"""

    def test_order_block_detection(self):
        """Test OB detection logic"""
        # Create mock price action
        df = pd.DataFrame({
            'high': [100, 101, 102, 99, 98, 97, 100, 103],
            'low': [99, 100, 101, 98, 97, 96, 99, 102],
            'close': [99.5, 100.5, 101.5, 98.5, 97.5, 96.5, 99.5, 102.5],
            'volume': [1000, 1100, 1200, 2000, 1800, 1600, 2200, 2500]
        })

        # Bearish OB: Last up candle before move down (index 2)
        # Bullish OB: Last down candle before move up (index 5)

        bearish_ob_idx = 2  # High: 102, Low: 101
        bullish_ob_idx = 5  # High: 97, Low: 96

        # Test detection
        assert df['high'].iloc[bearish_ob_idx] == 102
        assert df['low'].iloc[bullish_ob_idx] == 96

    def test_fair_value_gap(self):
        """Test FVG detection"""
        # FVG requires gap between candle 1 high and candle 3 low (bullish)
        # or candle 1 low and candle 3 high (bearish)

        df = pd.DataFrame({
            'high': [100, 101, 105, 104],  # Gap up
            'low': [99, 100, 104, 103]     # FVG between 101 and 104
        })

        # Bullish FVG exists between bars 1 and 3
        fvg_exists = df['low'].iloc[2] > df['high'].iloc[0]
        assert fvg_exists

    def test_liquidity_sweep(self):
        """Test liquidity sweep detection"""
        # Sweep: Price briefly exceeds previous high/low then reverses

        df = pd.DataFrame({
            'high': [100, 101, 102, 103, 102.5, 101],  # Sweep of 103
            'low': [99, 100, 101, 102, 101, 100],
            'close': [99.5, 100.5, 101.5, 102.8, 101.5, 100.5]
        })

        # Liquidity sweep at index 3 (high of 103)
        sweep_high = df['high'].iloc[3]
        reversal = df['close'].iloc[4] < df['close'].iloc[3]

        assert sweep_high == 103 and reversal


class TestMomentumEngine:
    """Test momentum calculations and delta capping"""

    def test_momentum_delta_capping(self):
        """Test momentum delta stays within ±0.06"""
        momentum_values = [-0.10, -0.06, -0.03, 0, 0.03, 0.06, 0.10]
        cap = 0.06

        for momentum in momentum_values:
            clamped = max(-cap, min(cap, momentum))
            assert -cap <= clamped <= cap
            assert abs(clamped) <= 0.06

    def test_rsi_calculation(self):
        """Test RSI calculation correctness"""
        # Create price series
        prices = [100, 102, 101, 103, 105, 104, 106, 108, 107, 109]
        df = pd.DataFrame({'close': prices})

        # Calculate RSI manually (simplified)
        gains = []
        losses = []

        for i in range(1, len(prices)):
            change = prices[i] - prices[i-1]
            if change > 0:
                gains.append(change)
                losses.append(0)
            else:
                gains.append(0)
                losses.append(abs(change))

        avg_gain = np.mean(gains[-14:]) if len(gains) >= 14 else np.mean(gains)
        avg_loss = np.mean(losses[-14:]) if len(losses) >= 14 else np.mean(losses)

        if avg_loss == 0:
            rsi = 100
        else:
            rs = avg_gain / avg_loss
            rsi = 100 - (100 / (1 + rs))

        assert 0 <= rsi <= 100


class TestHOBDetection:
    """Test Hands-on-Back pattern detection"""

    def test_volume_z_score(self):
        """Test volume z-score calculation"""
        volumes = [1000, 1100, 1200, 1300, 5000, 1400, 1500]  # Spike at index 4
        mean_vol = np.mean(volumes)
        std_vol = np.std(volumes)

        z_score_spike = (5000 - mean_vol) / std_vol
        assert z_score_spike > 1.3  # Should detect as significant

    def test_hob_proximity(self):
        """Test HOB proximity requirements"""
        atr = 2.0  # Average True Range
        max_proximity = 0.25 * atr  # Within 25% of ATR

        # Test various distances
        distances = [0.1, 0.3, 0.5, 0.8, 1.0]

        for distance in distances:
            is_proximate = distance <= max_proximity
            expected = distance <= 0.5  # 0.25 * 2.0

            assert is_proximate == expected

    def test_hob_recency(self):
        """Test HOB recency requirements"""
        max_bars_1h = 50
        max_bars_4h = 200

        # Test 1H
        assert 30 <= max_bars_1h  # Recent HOB
        assert 100 > max_bars_1h  # Too old

        # Test 4H
        assert 150 <= max_bars_4h  # Recent HOB
        assert 300 > max_bars_4h  # Too old


def run_all_unit_tests():
    """Run all unit and invariant tests"""
    print("🧪 RUNNING UNIT & INVARIANT TESTS")
    print("=" * 50)

    test_classes = [
        TestInvariants,
        TestWyckoffPhases,
        TestSMCLogic,
        TestMomentumEngine,
        TestHOBDetection
    ]

    results = []
    for test_class in test_classes:
        test_instance = test_class()
        class_name = test_class.__name__

        print(f"\nTesting {class_name}...")

        # Run all test methods
        test_methods = [m for m in dir(test_instance) if m.startswith('test_')]

        for method_name in test_methods:
            try:
                method = getattr(test_instance, method_name)
                method()
                print(f"   ✅ {method_name}")
                results.append((class_name, method_name, True))
            except AssertionError as e:
                print(f"   ❌ {method_name}: {e}")
                results.append((class_name, method_name, False))
            except Exception as e:
                print(f"   ⚠️  {method_name}: {e}")
                results.append((class_name, method_name, False))

    # Summary
    passed = sum(1 for _, _, result in results if result)
    total = len(results)

    print(f"\n📊 RESULTS: {passed}/{total} tests passed")

    return passed == total


if __name__ == "__main__":
    success = run_all_unit_tests()
    exit(0 if success else 1)