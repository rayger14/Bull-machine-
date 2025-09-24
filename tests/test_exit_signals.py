#!/usr/bin/env python3
"""
Test Exit Signal Framework
Tests the exit signal detection and integration system.
"""

import sys
import pandas as pd
import numpy as np
import logging
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from bull_machine.strategy.exits import (
    ExitSignalEvaluator, CHoCHAgainstDetector, MomentumFadeDetector,
    TimeStopEvaluator, create_default_exit_config
)
from bull_machine.strategy.exits.types import ExitType, ExitAction
from bull_machine.backtest.broker import PaperBroker

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def create_test_data_with_choch():
    """Create test data with clear CHoCH pattern."""
    dates = pd.date_range('2023-01-01', periods=100, freq='H')

    # Create uptrend followed by breakdown (CHoCH)
    prices = []
    base_price = 50000

    for i in range(100):
        if i < 40:
            # Strong uptrend
            trend = i * 50
            noise = (i % 10 - 5) * 20
        elif i < 60:
            # Consolidation at top
            trend = 40 * 50
            noise = (i % 15 - 7) * 100
        else:
            # Breakdown (CHoCH against long positions)
            trend = 40 * 50 - (i - 60) * 80
            noise = (i % 8 - 4) * 50

        price = base_price + trend + noise
        prices.append(max(price, 10000))  # Keep prices reasonable

    # Create OHLCV data
    data = []
    for i, close_price in enumerate(prices):
        if i == 0:
            open_price = close_price
        else:
            open_price = prices[i-1] + (close_price - prices[i-1]) * 0.2

        volatility = abs(close_price - open_price) + 30
        high_price = max(open_price, close_price) + volatility * 0.5
        low_price = min(open_price, close_price) - volatility * 0.3

        data.append({
            'timestamp': dates[i],
            'open': open_price,
            'high': high_price,
            'low': low_price,
            'close': close_price,
            'volume': 1000 + (i % 20) * 50
        })

    df = pd.DataFrame(data)
    df = df.set_index('timestamp')
    return df

def test_choch_detector():
    """Test CHoCH-Against detection."""
    print("\n" + "="*60)
    print("TEST: CHoCH-Against Detection")
    print("="*60)

    config = create_default_exit_config()
    detector = CHoCHAgainstDetector(config['choch_against'])

    # Create test data with CHoCH pattern
    df = create_test_data_with_choch()

    # Test on the breakdown portion (should detect CHoCH against long position)
    mtf_data = {'4H': df}

    signal = detector.evaluate(
        symbol="TESTBTC",
        position_bias="long",  # We have a long position
        mtf_data=mtf_data,
        current_bar=df.index[-5]  # During breakdown
    )

    if signal:
        print(f"✅ CHoCH signal detected: {signal.exit_type.value}")
        print(f"   Confidence: {signal.confidence:.2f}")
        print(f"   Action: {signal.action.value}")
        print(f"   Reasons: {signal.reasons}")
        return True
    else:
        print("❌ No CHoCH signal detected")
        return False

def test_momentum_fade_detector():
    """Test momentum fade detection."""
    print("\n" + "="*60)
    print("TEST: Momentum Fade Detection")
    print("="*60)

    config = create_default_exit_config()
    detector = MomentumFadeDetector(config['momentum_fade'])

    # Create data with momentum fade (RSI divergence)
    df = create_test_data_with_choch()

    mtf_data = {'1H': df}

    signal = detector.evaluate(
        symbol="TESTBTC",
        position_bias="long",
        mtf_data=mtf_data,
        current_bar=df.index[-10]
    )

    if signal:
        print(f"✅ Momentum fade signal detected: {signal.exit_type.value}")
        print(f"   Confidence: {signal.confidence:.2f}")
        print(f"   Action: {signal.action.value}")
        return True
    else:
        print("❌ No momentum fade signal detected")
        return False

def test_time_stop_evaluator():
    """Test time-based stop."""
    print("\n" + "="*60)
    print("TEST: Time Stop Evaluation")
    print("="*60)

    config = create_default_exit_config()
    evaluator = TimeStopEvaluator(config['time_stop'])

    # Create position data that's been open too long
    old_time = pd.Timestamp.now() - pd.Timedelta(hours=200)  # Longer than max_bars_1h
    position_data = {
        'entry_time': old_time,
        'pnl_pct': 0.05  # Small gain, not enough to justify time
    }

    signal = evaluator.evaluate(
        symbol="TESTBTC",
        position_data=position_data,
        current_bar=pd.Timestamp.now()
    )

    if signal:
        print(f"✅ Time stop signal detected: {signal.exit_type.value}")
        print(f"   Confidence: {signal.confidence:.2f}")
        print(f"   Action: {signal.action.value}")
        return True
    else:
        print("❌ No time stop signal detected")
        return False

def test_exit_evaluator_integration():
    """Test full exit evaluator integration."""
    print("\n" + "="*60)
    print("TEST: Exit Evaluator Integration")
    print("="*60)

    config = create_default_exit_config()
    evaluator = ExitSignalEvaluator(config)

    # Create test scenario
    df = create_test_data_with_choch()
    mtf_data = {'1H': df, '4H': df}

    position_data = {
        'bias': 'long',
        'entry_time': df.index[0],
        'pnl_pct': 0.02,
        'entry_price': 50000,
        'stop_price': 49000
    }

    # Evaluate at breakdown point
    result = evaluator.evaluate_exits(
        symbol="TESTBTC",
        position_data=position_data,
        mtf_data=mtf_data,
        current_bar=df.index[-5]
    )

    if result.has_signals():
        print(f"✅ Exit evaluation generated {len(result.signals)} signals")

        recommended = evaluator.get_action_recommendation(result)
        if recommended:
            print(f"   Recommended: {recommended.exit_type.value}")
            print(f"   Confidence: {recommended.confidence:.2f}")
            print(f"   Urgency: {recommended.urgency:.2f}")

        return True
    else:
        print("❌ Exit evaluation generated no signals")
        return False

def test_broker_exit_integration():
    """Test broker integration with exit signals."""
    print("\n" + "="*60)
    print("TEST: Broker Exit Integration")
    print("="*60)

    from bull_machine.strategy.exits.types import ExitSignal, ExitType, ExitAction

    broker = PaperBroker()

    # Open a position
    broker.submit(
        ts=pd.Timestamp.now(),
        symbol="TESTBTC",
        side="long",
        size=1.0,
        price_hint=50000.0
    )

    # Create mock exit signal
    exit_signal = ExitSignal(
        timestamp=pd.Timestamp.now(),
        symbol="TESTBTC",
        exit_type=ExitType.CHOCH_AGAINST,
        action=ExitAction.PARTIAL_EXIT,
        confidence=0.8,
        urgency=0.7,
        exit_percentage=0.5,
        reasons=["Test exit signal"]
    )

    # Process exit signal
    fills = broker.mark(pd.Timestamp.now(), "TESTBTC", 51000.0, exit_signal)

    if fills:
        print(f"✅ Exit signal processed: {len(fills)} fills")
        for fill in fills:
            print(f"   Fill: {fill['side']} @ {fill['price']:.2f}, PnL: {fill.get('pnl', 0):.2f}")

        # Check if position size was reduced
        pos = broker.positions.get("TESTBTC")
        if pos and pos.size < 1.0:
            print(f"   Position size reduced to: {pos.size}")

        return True
    else:
        print("❌ Exit signal processing failed")
        return False

def main():
    """Run all exit signal tests."""
    print("🧪 EXIT SIGNAL FRAMEWORK TESTS")
    print("=" * 80)

    tests = [
        ("CHoCH-Against Detection", test_choch_detector),
        ("Momentum Fade Detection", test_momentum_fade_detector),
        ("Time Stop Evaluation", test_time_stop_evaluator),
        ("Exit Evaluator Integration", test_exit_evaluator_integration),
        ("Broker Exit Integration", test_broker_exit_integration)
    ]

    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
            print(f"\n{test_name}: {'✅ PASSED' if result else '❌ FAILED'}")
        except Exception as e:
            results.append((test_name, False))
            print(f"\n{test_name}: ❌ ERROR - {e}")
            logging.error(f"Test {test_name} failed with error: {e}", exc_info=True)

    print("\n" + "="*80)
    print("EXIT SIGNAL TEST RESULTS")
    print("="*80)

    for test_name, result in results:
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{test_name:<35} {status}")

    passed = sum(1 for _, result in results if result)
    total = len(results)
    print(f"\nOVERALL: {passed}/{total} tests passed")

    if passed == total:
        print("🎉 All exit signal tests passed! Framework ready for production.")
        return True
    else:
        print("⚠️ Some exit signal tests failed. Check implementation.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)