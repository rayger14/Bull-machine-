#!/usr/bin/env python3
"""
Simple test to validate v1.6.0 M1/M2 and Fib signal generation
Bypasses most quality floors to isolate the signal flow
"""

import pandas as pd
from typing import Dict
from bull_machine.modules.fusion.v151_core_trader import CoreTraderV151
from bull_machine.strategy.wyckoff_m1m2 import compute_m1m2_scores
from bull_machine.strategy.hidden_fibs import compute_hidden_fib_scores
from bull_machine.backtest.ensemble_mode import EnsembleAligner

# Load test data
def load_test_data():
    timeframes = {}

    # Load 1H
    df_1h = pd.read_csv('chartlogs2/ETH_1H.csv')
    df_1h['timestamp'] = pd.to_datetime(df_1h['timestamp'])
    df_1h.columns = ['timestamp', 'open_1H', 'high_1H', 'low_1H', 'close_1H', 'volume_1H']
    timeframes['1H'] = df_1h

    # Load 4H
    df_4h = pd.read_csv('chartlogs2/ETH_4H.csv')
    df_4h['timestamp'] = pd.to_datetime(df_4h['timestamp'])
    df_4h.columns = ['timestamp', 'open_4H', 'high_4H', 'low_4H', 'close_4H', 'volume_4H']
    timeframes['4H'] = df_4h

    # Load 1D
    df_1d = pd.read_csv('chartlogs2/ETH_1D.csv')
    df_1d['timestamp'] = pd.to_datetime(df_1d['timestamp'])
    df_1d.columns = ['timestamp', 'open_1D', 'high_1D', 'low_1D', 'close_1D', 'volume_1D']
    timeframes['1D'] = df_1d

    return timeframes

def test_enhanced_signals():
    """Test enhanced signal generation on recent data"""

    print("🧪 Testing Bull Machine v1.6.0 Enhanced Signals")
    print("=" * 60)

    timeframes = load_test_data()

    # Test on last 100 bars of each timeframe
    for tf, df in timeframes.items():
        if len(df) < 100:
            continue

        test_df = df.tail(100).reset_index(drop=True)

        print(f"\n🔍 Testing {tf} - Last 100 bars:")
        print(f"   Price Range: ${test_df[f'close_{tf}'].min():.2f} - ${test_df[f'close_{tf}'].max():.2f}")

        try:
            # Test M1/M2 signals
            m1m2_scores = compute_m1m2_scores(test_df, tf)
            print(f"   📊 M1/M2: M1={m1m2_scores['m1']:.3f}, M2={m1m2_scores['m2']:.3f}")

            # Test Fibonacci signals
            fib_scores = compute_hidden_fib_scores(test_df, tf)
            print(f"   📐 Fibs: Ret={fib_scores['fib_retracement']:.3f}, Ext={fib_scores['fib_extension']:.3f}")

            # Test traditional scoring
            trader = CoreTraderV151({
                "features": {
                    "six_candle_leg": True,
                    "mtf_dl2": True,
                    "orderflow_lca": True,
                    "negative_vip": True
                }
            })

            traditional_scores = trader.compute_base_scores(test_df)
            print(f"   🔧 Traditional: wyckoff={traditional_scores.get('wyckoff', 0):.3f}")

            # Enhanced integration
            enhanced_wyckoff = traditional_scores.get('wyckoff', 0.25) + (m1m2_scores['m1'] * 0.3) + (m1m2_scores['m2'] * 0.3)
            print(f"   ⚡ Enhanced Wyckoff: {enhanced_wyckoff:.3f} (vs traditional {traditional_scores.get('wyckoff', 0):.3f})")

        except Exception as e:
            print(f"   ⚠️  Error testing {tf}: {e}")

def test_ensemble_simple():
    """Test ensemble with minimal requirements"""

    print("\n🎯 Testing Simplified Ensemble")
    print("=" * 40)

    # Super minimal config
    config = {
        "entry_threshold": 0.25,  # Very low
        "quality_floors": {
            "wyckoff": 0.1,       # Very low
            "liquidity": 0.1,
            "structure": 0.1,
            "momentum": 0.1,
            "volume": 0.1,
            "context": 0.1,
            "mtf": 0.1,
            "m1": 0.1,
            "m2": 0.1,
            "fib_retracement": 0.1,
            "fib_extension": 0.1
        },
        "ensemble": {
            "enabled": True,
            "min_consensus": 1,    # Only need 1 TF
            "rolling_k": 1,        # Only need 1 bar
            "rolling_n": 5,
            "lead_lag_window": 10, # Very lenient
            "consensus_penalty": 0.0,
            "base_threshold": 0.2  # Very low
        }
    }

    aligner = EnsembleAligner(config)

    timeframes = load_test_data()

    # Create simple scores
    simple_scores = {
        '1H': {'wyckoff': 0.3, 'liquidity': 0.3, 'structure': 0.3, 'momentum': 0.3, 'volume': 0.3, 'context': 0.3, 'mtf': 0.3},
        '4H': {'wyckoff': 0.3, 'liquidity': 0.3, 'structure': 0.3, 'momentum': 0.3, 'volume': 0.3, 'context': 0.3, 'mtf': 0.3},
        '1D': {'wyckoff': 0.3, 'liquidity': 0.3, 'structure': 0.3, 'momentum': 0.3, 'volume': 0.3, 'context': 0.3, 'mtf': 0.3}
    }

    aligner.update(simple_scores)
    fire, score = aligner.fire(simple_scores, timeframes)

    print(f"   🚀 Ensemble Fire: {fire}")
    print(f"   📊 Ensemble Score: {score:.3f}")

    if fire:
        print("   ✅ SUCCESS: Basic ensemble firing works!")
    else:
        print("   ❌ FAIL: Even simple ensemble not firing")

if __name__ == "__main__":
    test_enhanced_signals()
    test_ensemble_simple()

    print("\n" + "=" * 60)
    print("🏁 Test Complete")