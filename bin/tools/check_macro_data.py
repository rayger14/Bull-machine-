#!/usr/bin/env python3
"""
Check for macro data availability and create diagnostic report
"""
import os
import sys
from pathlib import Path

def main():
    # Required macro series from configs
    required_series = {
        "DXY": ["TVC_DXY", "DXY"],
        "VIX": ["VIX", "CBOE_VIX"],
        "WTI": ["WTI", "OILUSD", "EASYMARKETS_OILUSD"],
        "GOLD": ["GOLD", "TVC_GOLD", "COMEX_GC1!"],
        "US10Y": ["US10Y", "TVC_US10Y", "TNX"],
        "US2Y": ["US2Y", "TVC_US2Y", "IRX"],
        "MOVE": ["MOVE", "TVC_MOVE"],
        "USDT.D": ["CRYPTOCAP_USDT.D"],
        "TOTAL": ["CRYPTOCAP_TOTAL"],
        "TOTAL2": ["CRYPTOCAP_TOTAL2"],
        "TOTAL3": ["CRYPTOCAP_TOTAL3"]
    }

    chart_logs = Path("/Users/raymondghandchi/Desktop/Chart Logs")
    data_dir = Path("data")

    print("🔍 Bull Machine Macro Data Availability Check")
    print("=" * 50)

    # Check what's available in Chart Logs
    available_files = {}
    if chart_logs.exists():
        for file in chart_logs.glob("*.csv"):
            available_files[file.name] = file

    print(f"\\n📁 Chart Logs Directory: {len(available_files)} CSV files found")

    # Check each required series
    found = {}
    missing = []

    for series, possible_names in required_series.items():
        series_found = False
        found_files = []

        for possible in possible_names:
            # Look for 1D timeframe files
            pattern_matches = [f for f in available_files.keys()
                             if possible in f and ("1D_" in f or "1W_" in f)]
            if pattern_matches:
                series_found = True
                found_files.extend(pattern_matches)

        if series_found:
            found[series] = found_files
        else:
            missing.append(series)

    # Print results
    print(f"\\n✅ FOUND ({len(found)} series):")
    for series, files in found.items():
        print(f"   {series:10} → {files[0]}")
        for extra in files[1:]:
            print(f"   {' ' * 12} → {extra}")

    print(f"\\n❌ MISSING ({len(missing)} series):")
    for series in missing:
        alternatives = required_series[series]
        print(f"   {series:10} → Need one of: {', '.join(alternatives)}")

    # Crypto proxies available
    crypto_proxies = [s for s in found.keys() if s.startswith(('TOTAL', 'USDT'))]
    print(f"\\n🔄 Crypto Proxies Available: {', '.join(crypto_proxies)}")

    # Critical vs Optional
    critical = ["DXY", "VIX", "WTI", "US10Y"]
    critical_missing = [s for s in missing if s in critical]

    if critical_missing:
        print(f"\\n⚠️  CRITICAL MISSING: {', '.join(critical_missing)}")
        print("   → Macro engine will default to neutral (0% veto)")
    else:
        print(f"\\n🎯 All critical macro series found!")

    # Recommendation
    print(f"\\n📋 Recommendations:")
    if missing:
        print("   1. Create symlinks for found files:")
        for series, files in found.items():
            src = chart_logs / files[0]
            dst = data_dir / f"{series}_1D.csv"
            print(f"      ln -sf '{src}' '{dst}'")

        if missing:
            print("   2. Source missing series:")
            for series in missing:
                print(f"      - {series}: {', '.join(required_series[series])}")
    else:
        print("   1. All series found! Create symlinks to enable macro engine.")

    print(f"\\n🚀 After linking, macro engine will fire with:")
    print(f"   • Veto strength: 5-15% (vs current 0%)")
    print(f"   • Regime detection: Risk On/Off, Stagflation, etc.")
    print(f"   • Flight-to-safety signals")
    print(f"   • VIX hysteresis (real vs proxy)")

if __name__ == "__main__":
    main()