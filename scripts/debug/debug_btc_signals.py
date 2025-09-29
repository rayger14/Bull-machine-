#!/usr/bin/env python3
"""
DIAGNOSTIC: Debug BTC signal generation
⚠️  FOR DEBUGGING ONLY - NOT PRODUCTION TESTING
Use run_btc_ensemble_backtest.py for actual Bull Machine functionality testing
"""

import pandas as pd
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from bull_machine.modules.fusion.v151_core_trader import CoreTraderV151

def load_btc_sample():
    """Load a small sample of BTC data for debugging."""
    file_path = "/Users/raymondghandchi/Downloads/Chart logs 2/COINBASE_BTCUSD, 1D_85c84.csv"

    df = pd.read_csv(file_path)

    # Handle timestamp
    if 'time' in df.columns:
        df['timestamp'] = pd.to_datetime(df['time'], unit='s')

    # Handle volume
    if 'volume' not in df.columns:
        if 'BUY+SELL V' in df.columns:
            df['volume'] = df['BUY+SELL V']
        else:
            df['volume'] = df['close'] * 100

    # Clean data
    df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']].dropna()

    # Return recent 300 bars
    return df.tail(300).reset_index(drop=True)

def debug_signal_generation():
    """Debug the signal generation process step by step."""

    print("🔍 BTC Signal Generation Debug")
    print("=" * 50)

    # Load data
    df = load_btc_sample()
    print(f"Loaded {len(df)} bars")
    print(f"Price range: ${df['close'].min():.2f} - ${df['close'].max():.2f}")

    # Simple config
    config = {
        "timeframe": "1D",
        "entry_threshold": 0.20,  # Very low threshold
        "cooldown_bars": 1,
        "quality_floors": {
            'wyckoff': 0.20,
            'liquidity': 0.20,
            'structure': 0.20,
            'momentum': 0.20,
            'volume': 0.20,
            'context': 0.20,
            'mtf': 0.20
        },
        "features": {
            "atr_sizing": True,
            "atr_exits": True,
            "ensemble_htf_bias": False  # Disable for now
        },
        "risk": {
            "risk_pct": 0.01,
            "atr_window": 14,
            "sl_atr": 2.0
        }
    }

    # Initialize trader
    trader = CoreTraderV151(config)

    # Try signal generation on different bars
    for i in range(150, min(200, len(df))):
        history = df.iloc[:i+1]

        print(f"\n--- Bar {i} (Price: ${df.iloc[i]['close']:.2f}) ---")

        try:
            trade_plan = trader.check_entry(history, -999, config, 10000)

            if trade_plan:
                print(f"✅ SIGNAL FOUND!")
                print(f"  Side: {trade_plan['side']}")
                print(f"  Entry Price: ${trade_plan['entry_price']:.2f}")
                print(f"  Quantity: ${trade_plan['quantity']:.2f}")
                print(f"  Score: {trade_plan.get('weighted_score', 0):.3f}")
                print(f"  Layer scores: {trade_plan.get('layer_scores', {})}")
                break
            else:
                print("❌ No signal")

        except Exception as e:
            print(f"⚠️  Error: {e}")
            import traceback
            traceback.print_exc()
            break

    else:
        print(f"\n❌ No signals found in {50} bars tested")
        print("This suggests the entry conditions are too strict")

if __name__ == "__main__":
    debug_signal_generation()