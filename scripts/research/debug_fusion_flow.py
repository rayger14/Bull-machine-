#!/usr/bin/env python3
"""
Debug Fusion Signal Flow - Track signal generation step by step
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

from engine.io.tradingview_loader import load_tv
from engine.fusion import FusionEngine, FusionSignal
import pandas as pd
import json
import numpy as np

def debug_fusion_flow():
    """Debug the fusion signal generation flow"""

    # Load ETH data
    print("📊 Loading ETH data...")
    df = load_tv('ETH_4H')

    # Filter to recent period
    start_date = pd.to_datetime('2025-05-10', utc=True)
    end_date = pd.to_datetime('2025-05-25', utc=True)

    if df.index.tz is None:
        df.index = df.index.tz_localize('UTC')

    df = df[(df.index >= start_date) & (df.index <= end_date)]
    print(f"✅ Loaded {len(df)} bars from {df.index.min()} to {df.index.max()}")

    # Load ultra config
    with open('configs/v170/assets/ETH_v17_ultra_calibration.json', 'r') as f:
        config = json.load(f)

    print("\n🔍 Testing Fusion Engine...")

    try:
        # Initialize fusion engine
        fusion_config = config['fusion']
        print(f"Fusion config: min_domains={fusion_config['min_domains']}, min_confidence={fusion_config['min_confidence']}")

        fusion_engine = FusionEngine(fusion_config)
        print("✅ Fusion engine initialized")

        # Test with a simple mock signal for each domain
        print("\n🎯 Creating mock signals for testing...")

        # Create mock signals that should definitely trigger
        mock_signals = {}

        # Mock SMC signal
        from engine.smc.smc_engine import SMCSignal
        mock_smc = SMCSignal(
            timestamp=df.index[50],
            signal_type="LONG",
            confidence=0.8,
            entry_price=df.iloc[50]['close'],
            stop_loss=df.iloc[50]['close'] * 0.98,
            take_profit=df.iloc[50]['close'] * 1.04,
            analysis={
                "pattern": "order_block",
                "confluence": 2,
                "strength": 0.75
            }
        )
        mock_signals['smc'] = [mock_smc]

        # Mock momentum signal
        from engine.momentum.momentum_engine import MomentumSignal
        mock_momentum = MomentumSignal(
            timestamp=df.index[50],
            signal_type="LONG",
            confidence=0.7,
            entry_price=df.iloc[50]['close'],
            stop_loss=df.iloc[50]['close'] * 0.98,
            take_profit=df.iloc[50]['close'] * 1.03,
            rsi=45.0,
            macd_signal=0.02
        )
        mock_signals['momentum'] = [mock_momentum]

        print(f"Created {len(mock_signals)} domain signal groups")

        # Test fusion
        try:
            fusion_signals = fusion_engine.fuse_signals(mock_signals, df.iloc[50])
            print(f"🎉 Fusion generated {len(fusion_signals)} signals!")

            if fusion_signals:
                for i, signal in enumerate(fusion_signals):
                    print(f"  Signal {i+1}: {signal.signal_type}, confidence={signal.confidence:.2f}, strength={signal.strength:.2f}")
            else:
                print("❌ No fusion signals generated despite mock inputs")

        except Exception as e:
            print(f"❌ Fusion error: {e}")

    except Exception as e:
        print(f"❌ Fusion engine error: {e}")
        import traceback
        traceback.print_exc()

    print("\n" + "="*50)
    print("🎯 FUSION DEBUG SUMMARY")
    print("="*50)

if __name__ == "__main__":
    debug_fusion_flow()