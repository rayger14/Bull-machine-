#!/usr/bin/env python3
"""
Debug Domain Engines - Test each domain individually
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

from engine.io.tradingview_loader import load_tv
from engine.liquidity.hob import HOBDetector
from engine.liquidity.bojan_rules import BojanEngine
from engine.smc.smc_engine import SMCEngine
from engine.temporal.tpi import TemporalEngine
import pandas as pd
import json
import numpy as np

def test_domain_engines():
    """Test each domain engine individually"""

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

    # Load config for parameters
    with open('configs/v170/assets/ETH_v17_calibration.json', 'r') as f:
        config = json.load(f)

    # Test HOB Detector
    print("\n🔍 Testing HOB Detector...")
    try:
        hob_detector = HOBDetector(config['domains']['liquidity']['hob_detection'])
        hob_signals = []

        for i in range(50, len(df)):  # Start after 50 bars for lookback
            try:
                signal = hob_detector.detect_hob(df.iloc[:i+1])
                if signal and signal.confidence > 0.5:
                    hob_signals.append(signal)
            except Exception as e:
                continue

        print(f"  📈 HOB Signals: {len(hob_signals)}")
        if hob_signals:
            print(f"    Sample: {hob_signals[0].timestamp}, confidence: {hob_signals[0].confidence:.2f}")

    except Exception as e:
        print(f"  ❌ HOB Error: {e}")

    # Test Bojan Engine
    print("\n🔍 Testing Bojan Engine...")
    try:
        bojan_engine = BojanEngine(config['domains']['liquidity']['bojan_engine'])
        bojan_signals = []

        for i in range(50, len(df)):
            try:
                signal = bojan_engine.analyze_reaction(df.iloc[:i+1])
                if signal and signal.strength > 0.5:
                    bojan_signals.append(signal)
            except Exception as e:
                continue

        print(f"  📈 Bojan Signals: {len(bojan_signals)}")
        if bojan_signals:
            print(f"    Sample: {bojan_signals[0].timestamp}, strength: {bojan_signals[0].strength:.2f}")

    except Exception as e:
        print(f"  ❌ Bojan Error: {e}")

    # Test SMC Engine
    print("\n🔍 Testing SMC Engine...")
    try:
        smc_config = config['domains']['smc']
        smc_engine = SMCEngine(smc_config)
        smc_signals = []

        for i in range(100, len(df)):  # SMC needs more lookback
            try:
                signals = smc_engine.analyze(df.iloc[:i+1])
                for signal in signals:
                    if signal.confidence > 0.5:
                        smc_signals.append(signal)
            except Exception as e:
                continue

        print(f"  📈 SMC Signals: {len(smc_signals)}")
        if smc_signals:
            print(f"    Sample: {smc_signals[0].timestamp}, confidence: {smc_signals[0].confidence:.2f}")

    except Exception as e:
        print(f"  ❌ SMC Error: {e}")

    # Test Temporal Engine
    print("\n🔍 Testing Temporal Engine...")
    try:
        temporal_engine = TemporalEngine(config['domains']['temporal'])
        temporal_signals = []

        for i in range(200, len(df)):  # Temporal needs even more data
            try:
                signals = temporal_engine.analyze(df.iloc[:i+1])
                for signal in signals:
                    if signal.confidence > 0.5:
                        temporal_signals.append(signal)
            except Exception as e:
                continue

        print(f"  📈 Temporal Signals: {len(temporal_signals)}")
        if temporal_signals:
            print(f"    Sample: {temporal_signals[0].timestamp}, confidence: {temporal_signals[0].confidence:.2f}")

    except Exception as e:
        print(f"  ❌ Temporal Error: {e}")

    print("\n" + "="*50)
    print("🎯 DOMAIN ENGINE SUMMARY")
    print("="*50)
    try:
        print(f"HOB Signals: {len(hob_signals)}")
    except:
        print("HOB Signals: Failed")
    try:
        print(f"Bojan Signals: {len(bojan_signals)}")
    except:
        print("Bojan Signals: Failed")
    try:
        print(f"SMC Signals: {len(smc_signals)}")
    except:
        print("SMC Signals: Failed")
    try:
        print(f"Temporal Signals: {len(temporal_signals)}")
    except:
        print("Temporal Signals: Failed")

if __name__ == "__main__":
    test_domain_engines()