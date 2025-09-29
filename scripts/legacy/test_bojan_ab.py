#!/usr/bin/env python3
"""
Bojan A/B Test - Bull Machine v1.6.2
Quick validation of Bojan impact on ETH backtest results
"""

import sys
import os
import json
import pandas as pd
sys.path.append('.')

from bull_machine.modules.fusion.bojan_hook import apply_bojan
from bull_machine.modules.bojan.bojan import compute_bojan_score


def load_eth_config():
    """Load ETH configuration"""
    config_path = "configs/v160/assets/ETH.json"
    with open(config_path, 'r') as f:
        return json.load(f)


def simple_backtest_with_bojan_flag(enable_bojan=True):
    """Simple backtest with Bojan on/off"""
    print(f"\\n=== Testing with Bojan {'ENABLED' if enable_bojan else 'DISABLED'} ===")

    # Load sample data
    try:
        data_path = '/Users/raymondghandchi/Downloads/Chart logs 2/COINBASE_ETHUSD, 1D_64942.csv'
        df = pd.read_csv(data_path).tail(100)  # Last 100 bars for speed

        # Handle Chart Logs 2 format
        if 'time' in df.columns:
            df['timestamp'] = pd.to_datetime(df['time'], unit='s')

        df.columns = df.columns.str.lower()
        if 'buy+sell v' in df.columns:
            df['volume'] = df['buy+sell v']

        df = df.set_index('timestamp').sort_index()
        print(f"Loaded {len(df)} bars for testing")

    except Exception as e:
        print(f"Warning: Could not load Chart Logs 2 data: {e}")
        # Create synthetic data for testing
        dates = pd.date_range('2024-01-01', periods=100, freq='D')
        df = pd.DataFrame({
            'open': 2000 + (dates.dayofyear * 5) + pd.Series(range(100)) * 2,
            'high': 2050 + (dates.dayofyear * 5) + pd.Series(range(100)) * 2,
            'low': 1950 + (dates.dayofyear * 5) + pd.Series(range(100)) * 2,
            'close': 2000 + (dates.dayofyear * 5) + pd.Series(range(100)) * 2,
            'volume': 1000 + pd.Series(range(100)) * 10
        }, index=dates)
        print(f"Using synthetic data: {len(df)} bars")

    # Load config and set Bojan flag
    config = load_eth_config()
    config['features']['bojan'] = enable_bojan

    # Test metrics
    bojan_signals_count = 0
    fusion_boosts_count = 0
    total_bojan_applied = 0.0

    # Simulate trading decisions over the data
    for i in range(20, len(df)):
        window_data = df.iloc[i-20:i+1]

        # Test Bojan scoring
        if enable_bojan:
            bojan_result = compute_bojan_score(window_data, config.get('bojan', {}))
            if bojan_result['bojan_score'] > 0.1:
                bojan_signals_count += 1

        # Test fusion integration
        base_scores = {"structure": 0.25, "wyckoff": 0.25, "volume": 0.25}
        enhanced_scores, telemetry = apply_bojan(
            base_scores, window_data, tf="1D", config=config, last_hooks={}
        )

        if telemetry.get('bojan_applied', 0) > 0:
            fusion_boosts_count += 1
            total_bojan_applied += telemetry['bojan_applied']

    # Results
    print(f"Bojan signals detected: {bojan_signals_count}")
    print(f"Fusion boosts applied: {fusion_boosts_count}")
    print(f"Total Bojan boost: {total_bojan_applied:.4f}")
    print(f"Average boost per signal: {total_bojan_applied / max(fusion_boosts_count, 1):.4f}")

    return {
        'bojan_enabled': enable_bojan,
        'signals_count': bojan_signals_count,
        'fusion_boosts': fusion_boosts_count,
        'total_boost': total_bojan_applied,
        'avg_boost': total_bojan_applied / max(fusion_boosts_count, 1)
    }


def run_ab_test():
    """Run A/B test comparing Bojan on vs off"""
    print("=== BOJAN A/B TEST ===")
    print("Testing Bojan microstructure impact on Bull Machine v1.6.2")

    # Test A: Bojan OFF
    results_off = simple_backtest_with_bojan_flag(enable_bojan=False)

    # Test B: Bojan ON
    results_on = simple_backtest_with_bojan_flag(enable_bojan=True)

    # Compare results
    print("\\n" + "="*50)
    print("A/B COMPARISON RESULTS")
    print("="*50)

    print(f"Bojan OFF: {results_off['signals_count']} signals, {results_off['fusion_boosts']} boosts")
    print(f"Bojan ON:  {results_on['signals_count']} signals, {results_on['fusion_boosts']} boosts")

    if results_on['fusion_boosts'] > 0:
        print(f"\\nBojan Impact:")
        print(f"  - Additional signals: {results_on['signals_count'] - results_off['signals_count']}")
        print(f"  - Fusion boost applications: {results_on['fusion_boosts']}")
        print(f"  - Average boost strength: {results_on['avg_boost']:.4f}")
        print("\\n✅ Bojan integration is WORKING and adding value")
    else:
        print("\\n⚠️  Bojan not triggering - may need threshold adjustment")

    # Telemetry validation
    if results_on['total_boost'] > 0:
        print("✅ Telemetry shows bojan_applied > 0")
    else:
        print("❌ No telemetry for bojan_applied")

    return results_off, results_on


if __name__ == '__main__':
    run_ab_test()