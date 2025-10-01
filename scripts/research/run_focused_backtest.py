#!/usr/bin/env python3
"""
Focused ETH Backtest - Bull Machine v1.7
Optimized for speed and signal verification
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

import pandas as pd
import numpy as np
import json
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Import core engines
from engine.io.tradingview_loader import load_tv
from engine.smc.smc_engine import SMCEngine
from engine.wyckoff.wyckoff_engine import WyckoffEngine
from engine.liquidity.hob import HOBDetector
from engine.momentum.momentum_engine import MomentumEngine

def run_focused_backtest():
    """Run focused backtest on recent ETH data"""

    print("🎯 BULL MACHINE v1.7 FOCUSED BACKTEST")
    print("="*60)

    # Load config
    with open('configs/v170/assets/ETH_v17_tuned.json', 'r') as f:
        config = json.load(f)

    # Load ETH data
    print("📊 Loading ETH 4H data...")
    try:
        df = load_tv('ETH_4H')
        print(f"✅ Loaded {len(df)} bars ({df.index[0]} to {df.index[-1]})")
    except Exception as e:
        print(f"❌ Failed to load data: {e}")
        return

    # Use last 200 bars for focused test
    df_test = df.tail(200)
    print(f"🔍 Testing on {len(df_test)} bars ({df_test.index[0]} to {df_test.index[-1]})")

    # Initialize engines
    print("\n🔧 Initializing engines...")
    try:
        smc_engine = SMCEngine(config['domains']['smc'])
        wyckoff_engine = WyckoffEngine(config['domains']['wyckoff'])
        hob_engine = HOBDetector(config['domains']['liquidity']['hob_detection'])
        momentum_engine = MomentumEngine(config['domains']['momentum'])
        print("✅ All engines initialized")
    except Exception as e:
        print(f"❌ Engine initialization failed: {e}")
        return

    # Test signal generation
    print("\n⚡ TESTING SIGNAL GENERATION")
    print("-" * 40)

    signals_found = 0
    trades_generated = 0

    # Get calibration thresholds
    cal_mode = config['fusion'].get('calibration_mode', False)
    if cal_mode:
        cal_thresholds = config['fusion']['calibration_thresholds']
        min_confidence = cal_thresholds['confidence']
        min_strength = cal_thresholds['strength']
    else:
        min_confidence = config['fusion']['entry_threshold_confidence']
        min_strength = config['fusion']['entry_threshold_strength']

    print(f"📏 Thresholds: confidence ≥ {min_confidence}, strength ≥ {min_strength}")
    print(f"🎮 Mode: {'CALIBRATION' if cal_mode else 'PRODUCTION'}")

    # Test on windows of data
    for i in range(50, len(df_test), 10):  # Every 10 bars for speed
        window_data = df_test.iloc[:i+1]
        recent_data = window_data.tail(100)
        current_bar = window_data.iloc[-1]

        try:
            # Test each engine
            domain_signals = {}

            # SMC
            try:
                smc_signal = smc_engine.analyze(recent_data)
                domain_signals['smc'] = smc_signal
                if smc_signal:
                    print(f"   SMC: {smc_signal.direction} (conf: {smc_signal.confidence:.3f})")
            except:
                domain_signals['smc'] = None

            # Momentum
            try:
                momentum_signal = momentum_engine.analyze(recent_data)
                domain_signals['momentum'] = momentum_signal
                if momentum_signal:
                    delta = momentum_engine.get_delta_only(recent_data)
                    print(f"   Momentum: {momentum_signal.direction} (delta: {delta:+.3f})")
            except:
                domain_signals['momentum'] = None

            # Wyckoff
            try:
                wyckoff_signal = wyckoff_engine.analyze(recent_data, usdt_stagnation=0.5)
                domain_signals['wyckoff'] = wyckoff_signal
                if wyckoff_signal:
                    print(f"   Wyckoff: {wyckoff_signal.phase.value} -> {wyckoff_signal.direction}")
            except:
                domain_signals['wyckoff'] = None

            # HOB
            try:
                hob_signal = hob_engine.detect_hob(recent_data)
                domain_signals['hob'] = hob_signal
                if hob_signal:
                    print(f"   HOB: {hob_signal.hob_type.value} (conf: {hob_signal.confidence:.3f})")
            except:
                domain_signals['hob'] = None

            # Simple fusion logic
            active_signals = [s for s in domain_signals.values() if s is not None]
            if active_signals:
                signals_found += 1

                # Get directions and confidences
                directions = []
                confidences = []

                for signal in active_signals:
                    if hasattr(signal, 'direction') and hasattr(signal, 'confidence'):
                        directions.append(signal.direction)
                        confidences.append(signal.confidence)

                if directions and confidences:
                    # Vote on direction
                    long_votes = directions.count('long')
                    short_votes = directions.count('short')

                    if long_votes > short_votes:
                        fusion_direction = 'long'
                        fusion_strength = long_votes / len(directions)
                    elif short_votes > long_votes:
                        fusion_direction = 'short'
                        fusion_strength = short_votes / len(directions)
                    else:
                        fusion_direction = 'neutral'
                        fusion_strength = 0.0

                    avg_confidence = np.mean(confidences)

                    # Check entry criteria
                    if avg_confidence >= min_confidence and fusion_strength >= min_strength:
                        trades_generated += 1
                        print(f"\n🎯 TRADE SIGNAL #{trades_generated} at {current_bar.name}")
                        print(f"   Direction: {fusion_direction.upper()}")
                        print(f"   Price: ${current_bar['close']:.2f}")
                        print(f"   Confidence: {avg_confidence:.3f} (≥ {min_confidence})")
                        print(f"   Strength: {fusion_strength:.3f} (≥ {min_strength})")
                        print(f"   Vote: {long_votes}L, {short_votes}S")
                        print(f"   Active engines: {len(active_signals)}")

                        # Show engine breakdown
                        for domain, signal in domain_signals.items():
                            if signal:
                                dir_str = getattr(signal, 'direction', 'N/A')
                                conf_str = f"{getattr(signal, 'confidence', 0):.3f}"
                                print(f"     {domain}: {dir_str} ({conf_str})")
                        print()

        except Exception as e:
            continue

    # Summary
    print("\n" + "="*60)
    print("🎯 FOCUSED BACKTEST SUMMARY")
    print("="*60)
    print(f"📊 Bars processed: {len(df_test)}")
    print(f"⚡ Signals found: {signals_found}")
    print(f"💰 Trade signals: {trades_generated}")
    print(f"📈 Signal ratio: {(trades_generated/max(1,signals_found))*100:.1f}%")

    if trades_generated > 0:
        print(f"\n✅ System is generating trades with current config!")
        print(f"   Calibration mode: {cal_mode}")
        print(f"   Confidence threshold: {min_confidence}")
        print(f"   Strength threshold: {min_strength}")
    else:
        print(f"\n⚠️ No trade signals generated")
        print(f"   Consider lowering thresholds or checking engine outputs")

    # Test latest bar specifically
    print(f"\n🔬 LATEST BAR ANALYSIS")
    print("-" * 30)
    latest_data = df.tail(100)
    latest_bar = df.iloc[-1]

    print(f"📅 Timestamp: {latest_bar.name}")
    print(f"💰 Price: ${latest_bar['close']:.2f}")

    # Test each engine on latest data
    try:
        smc_signal = smc_engine.analyze(latest_data)
        print(f"SMC: {'✅' if smc_signal else '❌'} {getattr(smc_signal, 'direction', 'None')} (conf: {getattr(smc_signal, 'confidence', 0):.3f})")
    except Exception as e:
        print(f"SMC: ❌ Error: {e}")

    try:
        momentum_signal = momentum_engine.analyze(latest_data)
        print(f"Momentum: {'✅' if momentum_signal else '❌'} {getattr(momentum_signal, 'direction', 'None')} (conf: {getattr(momentum_signal, 'confidence', 0):.3f})")
    except Exception as e:
        print(f"Momentum: ❌ Error: {e}")

    try:
        wyckoff_signal = wyckoff_engine.analyze(latest_data, usdt_stagnation=0.5)
        print(f"Wyckoff: {'✅' if wyckoff_signal else '❌'} {getattr(wyckoff_signal, 'phase', 'None')} -> {getattr(wyckoff_signal, 'direction', 'None')}")
    except Exception as e:
        print(f"Wyckoff: ❌ Error: {e}")

    try:
        hob_signal = hob_engine.detect_hob(latest_data)
        print(f"HOB: {'✅' if hob_signal else '❌'} {getattr(hob_signal, 'hob_type', 'None')} (conf: {getattr(hob_signal, 'confidence', 0):.3f})")
    except Exception as e:
        print(f"HOB: ❌ Error: {e}")

if __name__ == "__main__":
    run_focused_backtest()