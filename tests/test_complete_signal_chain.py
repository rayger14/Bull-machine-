#!/usr/bin/env python3
"""
Test Complete Signal Chain - All Engines + Fusion
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

from engine.io.tradingview_loader import load_tv
from engine.smc.smc_engine import SMCEngine
from engine.wyckoff.wyckoff_engine import WyckoffEngine
from engine.liquidity.hob import HOBDetector
from engine.momentum.momentum_engine import MomentumEngine
from engine.context.macro_pulse import MacroPulseEngine
import pandas as pd
import json

def test_complete_signal_chain():
    """Test the complete signal generation chain"""

    print("🎯 TESTING COMPLETE SIGNAL CHAIN")
    print("="*50)

    # Load real ETH data
    df = load_tv('ETH_4H')
    recent = df.tail(100)  # More data for better analysis
    print(f"📊 Testing with {len(recent)} ETH 4H bars ({recent.index[0]} to {recent.index[-1]})")

    # Load config
    with open('configs/v170/assets/ETH_v17_tuned.json', 'r') as f:
        config = json.load(f)

    print(f"\n🔧 Using config: {config['version']}")

    signals = {}

    # 1. Test SMC Engine
    print("\n📈 1. SMC ANALYSIS")
    try:
        smc = SMCEngine(config['domains']['smc'])
        smc_signal = smc.analyze(recent)
        if smc_signal:
            signals['smc'] = smc_signal
            print(f"   Direction: {smc_signal.direction}")
            print(f"   Confidence: {smc_signal.confidence:.3f}")
            print(f"   Hit counters: OB={smc_signal.hit_counters['ob_hits']}, FVG={smc_signal.hit_counters['fvg_hits']}")
            print(f"   Confluence rate: {smc_signal.confluence_rate:.1%}")
        else:
            print("   No SMC signal generated")
    except Exception as e:
        print(f"   ❌ SMC Error: {e}")

    # 2. Test Momentum Engine
    print("\n📊 2. MOMENTUM ANALYSIS")
    try:
        momentum = MomentumEngine(config['domains']['momentum'])
        momentum_signal = momentum.analyze(recent)
        momentum_delta = momentum.get_delta_only(recent)
        if momentum_signal:
            signals['momentum'] = momentum_signal
            print(f"   Direction: {momentum_signal.direction}")
            print(f"   Delta: {momentum_delta:+.3f}")
            print(f"   RSI: {momentum_signal.rsi:.1f}")
            print(f"   MACD norm: {momentum_signal.macd_normalized:.6f}")
        else:
            print("   No momentum signal")
    except Exception as e:
        print(f"   ❌ Momentum Error: {e}")

    # 3. Test Wyckoff Engine
    print("\n🏗️ 3. WYCKOFF ANALYSIS")
    try:
        wyckoff = WyckoffEngine(config['domains']['wyckoff'])
        wyckoff_signal = wyckoff.analyze(recent, usdt_stagnation=0.5)
        if wyckoff_signal:
            signals['wyckoff'] = wyckoff_signal
            print(f"   Phase: {wyckoff_signal.phase.value}")
            print(f"   Direction: {wyckoff_signal.direction}")
            print(f"   Volume quality: {wyckoff_signal.volume_quality:.3f}")
            print(f"   CRT active: {wyckoff_signal.crt_active}")
        else:
            print("   No Wyckoff phase detected")
    except Exception as e:
        print(f"   ❌ Wyckoff Error: {e}")

    # 4. Test HOB Detector
    print("\n💧 4. LIQUIDITY (HOB) ANALYSIS")
    try:
        hob = HOBDetector(config['domains']['liquidity']['hob_detection'])
        hob_signal = hob.detect_hob(recent)
        if hob_signal:
            signals['hob'] = hob_signal
            print(f"   Type: {hob_signal.hob_type.value}")
            print(f"   Quality: {hob_signal.quality.value}")
            print(f"   Confidence: {hob_signal.confidence:.3f}")
        else:
            print("   No HOB pattern detected")
    except Exception as e:
        print(f"   ❌ HOB Error: {e}")

    # 5. Simple Fusion Logic
    print("\n⚡ 5. SIGNAL FUSION")
    total_signals = len(signals)
    print(f"   Active signals: {total_signals}")

    if total_signals == 0:
        print("   🚫 No signals to fuse")
        return

    # Simple fusion logic
    directions = [s.direction for s in signals.values() if hasattr(s, 'direction')]
    confidences = [s.confidence for s in signals.values() if hasattr(s, 'confidence')]

    if directions:
        # Count direction consensus
        long_votes = sum(1 for d in directions if d == 'long')
        short_votes = sum(1 for d in directions if d == 'short')

        if long_votes > short_votes:
            fusion_direction = 'long'
            fusion_strength = long_votes / len(directions)
        elif short_votes > long_votes:
            fusion_direction = 'short'
            fusion_strength = short_votes / len(directions)
        else:
            fusion_direction = 'neutral'
            fusion_strength = 0.0

        avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0

        print(f"   📊 Fusion Result:")
        print(f"     Direction: {fusion_direction}")
        print(f"     Strength: {fusion_strength:.2f}")
        print(f"     Avg Confidence: {avg_confidence:.3f}")
        print(f"     Vote breakdown: {long_votes}L, {short_votes}S, {len(directions)-(long_votes+short_votes)}N")

        # Entry criteria check (handle calibration mode)
        fusion_config = config['fusion']
        if fusion_config.get('calibration_mode', False):
            cal_thresholds = fusion_config.get('calibration_thresholds', {})
            min_confidence = cal_thresholds.get('confidence', 0.32)
            entry_threshold = cal_thresholds.get('strength', 0.40)
        else:
            min_confidence = fusion_config.get('entry_threshold_confidence', 0.35)
            entry_threshold = fusion_config.get('entry_threshold_strength', 0.40)

        print(f"   🎯 Entry Check:")
        print(f"     Min confidence required: {min_confidence}")
        print(f"     Entry threshold required: {entry_threshold}")
        print(f"     Current avg confidence: {avg_confidence:.3f}")
        print(f"     Current strength: {fusion_strength:.3f}")

        if avg_confidence >= min_confidence and fusion_strength >= entry_threshold:
            print(f"   ✅ TRADE SIGNAL: {fusion_direction.upper()}")
            print(f"      Entry price: {recent['close'].iloc[-1]:.2f}")
            print(f"      Signal quality: HIGH")
        else:
            print(f"   ⚠️ Signal below entry thresholds")
            print(f"      Confidence gap: {min_confidence - avg_confidence:+.3f}")
            print(f"      Strength gap: {entry_threshold - fusion_strength:+.3f}")

    print(f"\n{'='*50}")
    print(f"🎯 SUMMARY")
    print(f"{'='*50}")
    print(f"Active engines: {total_signals}/{4}")
    print(f"Signal generation: {'✅ WORKING' if total_signals > 0 else '❌ NO SIGNALS'}")
    print(f"Fusion logic: {'✅ OPERATIONAL' if directions else '❌ NO DIRECTIONS'}")
    print(f"Entry criteria: {'✅ MET' if avg_confidence >= min_confidence and fusion_strength >= entry_threshold else '⚠️ BELOW THRESHOLD'}")

if __name__ == "__main__":
    test_complete_signal_chain()