#!/usr/bin/env python3
"""
Demo: Multi-Timeframe Confluence Analysis
Shows how ETH 4H signals are validated against 12H/1D trends + macro sentiment
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

from engine.io.tradingview_loader import load_tv, SYMBOL_MAP
import pandas as pd
import numpy as np

def demo_multi_timeframe_confluence():
    """Demonstrate multi-timeframe + macro confluence analysis"""

    print("🎯 MULTI-TIMEFRAME CONFLUENCE ANALYSIS")
    print("="*60)

    # 1. Primary timeframe (trading signals)
    print("\n📊 1. PRIMARY TIMEFRAME (ETH 4H)")
    try:
        eth_4h = load_tv('ETH_4H')
        print(f"   ✅ ETH 4H: {len(eth_4h)} bars ({eth_4h.index.min()} to {eth_4h.index.max()})")
        print(f"   📈 Price range: ${eth_4h['low'].min():.2f} - ${eth_4h['high'].max():.2f}")
        print(f"   📊 Avg daily range: {((eth_4h['high'] - eth_4h['low']) / eth_4h['close'] * 100).mean():.2f}%")
    except Exception as e:
        print(f"   ❌ ETH 4H error: {e}")

    # 2. Higher timeframes (trend context)
    print("\n📊 2. HIGHER TIMEFRAMES (Trend Context)")

    # Check if higher timeframes are available
    available_timeframes = []
    for symbol in ['ETH_12H', 'ETH_1D']:
        if symbol in SYMBOL_MAP:
            available_timeframes.append(symbol)
            print(f"   ✅ {symbol}: Available in symbol map")
        else:
            print(f"   ⚠️ {symbol}: Not in symbol map (would use synthetic)")

    # 3. Cross-asset analysis (BTC correlation, dominance effects)
    print("\n📊 3. CROSS-ASSET ANALYSIS")

    cross_assets = {
        'BTC_4H': 'Bitcoin correlation',
        'BTC.D_1D': 'Bitcoin dominance effects',
        'ETH.D_1D': 'Ethereum dominance',
        'TOTAL_4H': 'Total crypto market cap',
        'TOTAL3_4H': 'Altcoin market cap'
    }

    for symbol, description in cross_assets.items():
        if symbol in SYMBOL_MAP:
            try:
                data = load_tv(symbol)
                print(f"   ✅ {symbol}: {len(data)} bars - {description}")
            except:
                print(f"   ⚠️ {symbol}: Available but loading error - {description}")
        else:
            print(f"   ❌ {symbol}: Not available - {description}")

    # 4. Macro sentiment (risk-on/risk-off context)
    print("\n📊 4. MACRO SENTIMENT ANALYSIS")

    macro_indicators = {
        'DXY_1D': 'US Dollar strength (inverse correlation)',
        'VIX_1D': 'Fear/greed sentiment',
        'US10Y_1D': 'Risk-free rate (opportunity cost)',
        'WTI_1D': 'Oil prices (stagflation risk)',
        'GOLD_1D': 'Flight to safety indicator'
    }

    loaded_macro = 0
    for symbol, description in macro_indicators.items():
        if symbol in SYMBOL_MAP:
            try:
                data = load_tv(symbol)
                loaded_macro += 1

                # Calculate recent trend
                recent_change = data['close'].pct_change(10).iloc[-1] * 100
                direction = "📈" if recent_change > 1 else "📉" if recent_change < -1 else "➡️"

                print(f"   {direction} {symbol}: {len(data)} bars, recent 10-day: {recent_change:+.1f}% - {description}")
            except Exception as e:
                print(f"   ⚠️ {symbol}: Error loading - {description}")
        else:
            print(f"   ❌ {symbol}: Not available - {description}")

    # 5. Confluence scoring example
    print("\n🎯 5. CONFLUENCE ANALYSIS EXAMPLE")
    print("="*40)

    print("When analyzing an ETH 4H signal:")
    print("  📊 Primary: ETH 4H breakout pattern detected")
    print("  📈 Trend context: ETH 1D confirms uptrend")
    print("  🔗 Cross-asset: BTC showing strength (+correlation)")
    print("  🌍 Macro context: DXY weakening (bullish for crypto)")
    print("  📉 Risk sentiment: VIX declining (risk-on environment)")
    print("  ⚖️ Confluence score: 4/5 domains aligned")
    print("  ✅ Signal validation: HIGH CONFIDENCE")

    print("\n🚀 MULTI-TIMEFRAME IMPACT:")
    print("- 4H signals validated by 1D trend = Higher success rate")
    print("- Macro headwinds detected = Position size reduction")
    print("- Cross-asset divergence = Additional confirmation needed")
    print("- Risk regime alignment = Optimal entry timing")

    print(f"\n📈 MACRO DATA COVERAGE: {loaded_macro}/{len(macro_indicators)} indicators loaded")

    if loaded_macro >= 3:
        print("✅ Sufficient macro coverage for regime analysis")
    else:
        print("⚠️ Limited macro data - recommend expanding symbol map")

if __name__ == "__main__":
    demo_multi_timeframe_confluence()