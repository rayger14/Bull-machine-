#!/usr/bin/env python3
"""
Debug v1.5.1 Signal Generation
Analyze why no trades are being generated in the final validation
"""

import sys
import os
sys.path.append(os.path.abspath('.'))

import pandas as pd
import json
from pathlib import Path
from bull_machine.modules.fusion.v151_core_trader import CoreTraderV151

def load_eth_data() -> pd.DataFrame:
    """Load ETH 4H data for debugging"""
    data_path = '/Users/raymondghandchi/Downloads/Chart logs 2/COINBASE_ETHUSD, 240_1d04a.csv'

    df = pd.read_csv(data_path)
    df['time'] = pd.to_datetime(df['time'], unit='s')
    df = df.rename(columns={
        'time': 'timestamp',
        'open': 'open',
        'high': 'high',
        'low': 'low',
        'close': 'close'
    })

    if 'BUY+SELL V' in df.columns:
        df['volume'] = df['BUY+SELL V']
    else:
        df['volume'] = (df['high'] - df['low']) * df['close'] * 1000

    return df.sort_values('timestamp').reset_index(drop=True)

def load_config(profile: str) -> dict:
    """Load profile configuration"""
    config_path = Path(f"configs/v150/assets/{profile}.json")
    with open(config_path, 'r') as f:
        return json.load(f)

def debug_single_signal(df: pd.DataFrame, bar_idx: int, config: dict, trader: CoreTraderV151):
    """Debug signal generation for a single bar"""
    print(f"\n=== Debugging Bar {bar_idx} ===")

    bar_df = df.iloc[:bar_idx+1].copy()
    current_price = df.iloc[bar_idx]['close']
    current_date = df.iloc[bar_idx]['timestamp']

    print(f"Date: {current_date}")
    print(f"Price: ${current_price:.2f}")

    # Try entry check
    try:
        trade_plan = trader.check_entry(bar_df, bar_idx - 100, config, 10000)

        if trade_plan:
            print("✓ ENTRY SIGNAL GENERATED")
            print(f"  Side: {trade_plan['side']}")
            print(f"  Score: {trade_plan['weighted_score']:.3f}")
            print(f"  Layer scores: {trade_plan['layer_scores']}")
            return True
        else:
            print("✗ NO ENTRY SIGNAL")

            # Check individual components that might veto
            print("\nDiagnosing vetoes...")

            # Manual regime check
            if config.get("features", {}).get("regime_filter"):
                try:
                    from bull_machine.modules.regime_filter import regime_ok
                    regime_cfg = config.get("regime", {})
                    tf = config.get("timeframe", "")
                    regime_passed = regime_ok(bar_df, tf, regime_cfg)
                    print(f"  Regime filter: {'PASS' if regime_passed else 'VETO'}")
                except Exception as e:
                    print(f"  Regime filter: ERROR - {e}")

            # Check base layer scores
            try:
                if hasattr(trader, 'compute_base_scores'):
                    base_scores = trader.compute_base_scores(bar_df)
                    print(f"  Base layer scores: {base_scores}")

                    # Check quality floors
                    floors = config.get("quality_floors", {})
                    for layer, score in base_scores.items():
                        if layer in floors:
                            floor = floors[layer]
                            status = "PASS" if score >= floor else "FAIL"
                            print(f"    {layer}: {score:.3f} >= {floor} [{status}]")

                except Exception as e:
                    print(f"  Base score calculation: ERROR - {e}")

            # Check ensemble HTF bias
            if config.get("features", {}).get("ensemble_htf_bias"):
                try:
                    ctx_floor = config.get("quality_floors", {}).get("context", 0.3)
                    mtf_floor = config.get("quality_floors", {}).get("mtf", 0.3)

                    if hasattr(trader, 'compute_base_scores'):
                        scores = trader.compute_base_scores(bar_df)
                        ctx_score = scores.get("context", 0.0)
                        mtf_score = scores.get("mtf", 0.0)
                        combined_score = (ctx_score + mtf_score) / 2.0

                        print(f"  Context: {ctx_score:.3f} >= {ctx_floor} [{'PASS' if ctx_score >= ctx_floor else 'FAIL'}]")
                        print(f"  MTF: {mtf_score:.3f} >= {mtf_floor} [{'PASS' if mtf_score >= mtf_floor else 'FAIL'}]")
                        print(f"  Combined HTF: {combined_score:.3f} >= 0.65 [{'PASS' if combined_score >= 0.65 else 'FAIL'}]")

                except Exception as e:
                    print(f"  HTF ensemble check: ERROR - {e}")

            return False

    except Exception as e:
        print(f"ERROR in entry check: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Debug signal generation across time periods"""
    print("Bull Machine v1.5.1 Signal Generation Debug")
    print("=" * 50)

    # Load data and configs
    df = load_eth_data()
    print(f"Loaded {len(df)} 4H candles")
    print(f"Date range: {df['timestamp'].iloc[0]} to {df['timestamp'].iloc[-1]}")

    # Test both profiles
    for profile_name in ['ETH_4H', 'ETH']:
        print(f"\n{'='*60}")
        print(f"DEBUGGING {profile_name} PROFILE")
        print(f"{'='*60}")

        config = load_config(profile_name)
        trader = CoreTraderV151(config)

        print(f"Entry threshold: {config['entry_threshold']}")
        print(f"Quality floors: {config.get('quality_floors', {})}")
        print(f"Features: {[k for k, v in config.get('features', {}).items() if v]}")

        # Test specific time periods
        test_periods = [
            500,   # Early period
            800,   # Mid period
            1200,  # Later period
            1500,  # Recent period
            len(df) - 10  # Very recent
        ]

        signals_found = 0
        for bar_idx in test_periods:
            if bar_idx < len(df):
                found_signal = debug_single_signal(df, bar_idx, config, trader)
                if found_signal:
                    signals_found += 1

        print(f"\nSUMMARY: {signals_found}/{len(test_periods)} test periods generated signals")

        if signals_found == 0:
            print("⚠️  NO SIGNALS FOUND - System may be too restrictive")

            # Suggest relaxations
            print("\nSuggested relaxations:")
            print("1. Lower entry threshold")
            print("2. Reduce quality floors")
            print("3. Disable HTF ensemble requirement")
            print("4. Check regime filter settings")

        # Quick scan for ANY possible signals
        print(f"\nScanning last 200 bars for any signals...")
        scan_signals = 0
        start_idx = max(100, len(df) - 200)

        for i in range(start_idx, len(df), 10):  # Sample every 10th bar
            try:
                bar_df = df.iloc[:i+1].copy()
                trade_plan = trader.check_entry(bar_df, i - 100, config, 10000)
                if trade_plan:
                    scan_signals += 1
                    print(f"  Signal found at bar {i} ({df.iloc[i]['timestamp']})")
            except:
                continue

        print(f"Quick scan result: {scan_signals} signals in last 200 bars")

if __name__ == "__main__":
    main()