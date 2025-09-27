"""
Minimal unit tests for Bull Machine v1.6.0 enhanced signals
"""

import pandas as pd
import numpy as np
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from bull_machine.strategy.wyckoff_m1m2 import compute_m1m2_scores
from bull_machine.strategy.hidden_fibs import compute_hidden_fib_scores


def test_m1m2_range_and_side():
    """Test M1/M2 signal bounds and side determination."""
    # Create test data
    df = pd.DataFrame({
        'open': [100] * 240,
        'high': [102] * 240,
        'low': [98] * 240,
        'close': [100] * 240,
        'volume': [1000] * 240
    })

    # Compute scores
    scores = compute_m1m2_scores(df, 'H1')

    # Validate bounds
    assert 0.0 <= scores['m1'] <= 1.0, f"M1 out of range: {scores['m1']}"
    assert 0.0 <= scores['m2'] <= 1.0, f"M2 out of range: {scores['m2']}"
    assert scores['side'] in ('long', 'short', 'neutral'), f"Invalid side: {scores['side']}"

    print(f"✅ M1/M2 test passed: M1={scores['m1']:.3f}, M2={scores['m2']:.3f}, Side={scores['side']}")


def test_fibonacci_bounds():
    """Test Fibonacci signal bounds."""
    # Create trending data for Fibonacci
    prices = np.linspace(100, 120, 240)  # Uptrend
    df = pd.DataFrame({
        'open': prices,
        'high': prices * 1.01,
        'low': prices * 0.99,
        'close': prices,
        'volume': [1000] * 240
    })

    # Compute scores
    scores = compute_hidden_fib_scores(df, 'H1')

    # Validate bounds
    assert 0.0 <= scores['fib_retracement'] <= 1.0, f"FibR out of range: {scores['fib_retracement']}"
    assert 0.0 <= scores['fib_extension'] <= 1.0, f"FibX out of range: {scores['fib_extension']}"

    print(f"✅ Fibonacci test passed: Ret={scores['fib_retracement']:.3f}, Ext={scores['fib_extension']:.3f}")


def test_confluence_detection():
    """Test confluence requirement logic."""
    # Test confluence cases
    test_cases = [
        {'m1': 0.65, 'm2': 0.30, 'fib_retracement': 0.50, 'fib_extension': 0.20, 'expected': True},  # M1+FibR
        {'m1': 0.30, 'm2': 0.55, 'fib_retracement': 0.20, 'fib_extension': 0.50, 'expected': True},  # M2+FibX
        {'m1': 0.75, 'm2': 0.20, 'fib_retracement': 0.20, 'fib_extension': 0.20, 'expected': True},  # Solo M1
        {'m1': 0.50, 'm2': 0.40, 'fib_retracement': 0.30, 'fib_extension': 0.30, 'expected': False}, # No confluence
    ]

    M1_TH = 0.60
    M2_TH = 0.50
    FIBR_TH = 0.45
    FIBX_TH = 0.45
    SOLO_M1 = 0.70
    SOLO_M2 = 0.60

    for i, case in enumerate(test_cases):
        m1 = case['m1']
        m2 = case['m2']
        fibr = case['fib_retracement']
        fibx = case['fib_extension']

        # Check confluence logic
        has_confluence = False
        if (m1 > M1_TH and fibr > FIBR_TH) or (m2 > M2_TH and fibx > FIBX_TH):
            has_confluence = True
        elif m1 > SOLO_M1 or m2 > SOLO_M2:
            has_confluence = True

        assert has_confluence == case['expected'], f"Case {i+1} failed: expected {case['expected']}, got {has_confluence}"

    print(f"✅ Confluence test passed: All {len(test_cases)} cases validated")


if __name__ == "__main__":
    print("\n🧪 Running Bull Machine v1.6.0 Unit Tests")
    print("=" * 50)

    test_m1m2_range_and_side()
    test_fibonacci_bounds()
    test_confluence_detection()

    print("=" * 50)
    print("✅ All tests passed!")