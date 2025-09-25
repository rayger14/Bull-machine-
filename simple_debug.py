#!/usr/bin/env python3
"""
Simple debug for v1.5.1 signals
"""

import sys
import os
sys.path.append(os.path.abspath('.'))

import pandas as pd
import json
from pathlib import Path
from bull_machine.modules.fusion.v151_core_trader import CoreTraderV151

def main():
    print("Simple signal debug test")

    # Load config
    with open("configs/v150/assets/ETH_4H.json", 'r') as f:
        config = json.load(f)

    print(f"Config loaded: threshold={config['entry_threshold']}")

    # Load data
    data_path = '/Users/raymondghandchi/Downloads/Chart logs 2/COINBASE_ETHUSD, 240_1d04a.csv'
    df = pd.read_csv(data_path)
    df['time'] = pd.to_datetime(df['time'], unit='s')
    df = df.rename(columns={'time': 'timestamp'})
    df['volume'] = 1000  # Simple volume

    print(f"Data loaded: {len(df)} rows")

    # Initialize trader
    trader = CoreTraderV151(config)
    print("Trader initialized")

    # Test single entry check
    test_df = df.iloc[:100].copy()
    try:
        result = trader.check_entry(test_df, 10, config, 10000)
        print(f"Entry check result: {result}")
    except Exception as e:
        print(f"Entry check error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()