#!/usr/bin/env python3
"""
Multi-Asset Profile Generator - Bull Machine v1.7.2
Generates adaptive configurations for ETH, SOL, XRP, and any other asset
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from engine.adapters.asset_profiler import AssetProfiler
from data.real_data_loader import RealDataLoader
import pandas as pd
import json
from datetime import datetime

def generate_asset_profiles():
    """Generate profiles for all available assets."""
    print("🏭 Multi-Asset Profile Generator v1.7.2")
    print("=" * 70)
    print("Building adaptive configurations for all assets...")
    print()

    loader = RealDataLoader()
    assets = ['ETHUSD', 'SOLUSD', 'XRPUSD']
    profiles = {}
    configs = {}

    for symbol in assets:
        try:
            print(f"🔍 Processing {symbol}...")

            # Load 4H data for profiling
            data = loader.load_raw_data(f'COINBASE_{symbol}', '4H')

            if data is None or len(data) < 100:
                print(f"❌ Insufficient data for {symbol}")
                continue

            # Create profiler
            profiler = AssetProfiler(symbol, 'COINBASE')

            # Load BTC as benchmark for crypto correlations
            btc_data = loader.load_raw_data('COINBASE_BTCUSD', '4H')

            # Build profile
            profile = profiler.build_profile(data, btc_data)
            config = profiler.generate_config()

            # Save files
            profiler.save_profile()
            profiler.save_config()

            profiles[symbol] = profile
            configs[symbol] = config

            # Display key metrics
            vol = profile['volatility']
            liq = profile['liquidity']
            print(f"✅ {symbol} Profile Complete:")
            print(f"   ATR p50: {vol['atr']['p50']:.2f}% | Regime: {vol['regime']}")
            print(f"   Volume Z70: {liq['volume_z_scores']['z70']:.2f}")
            print(f"   Risk Scalar: {config['risk']['position_size_scalar']:.2f}")
            if 'correlations' in profile:
                corr = profile['correlations']['full_period']
                print(f"   BTC Correlation: {corr:.2f}")
            print()

        except Exception as e:
            print(f"❌ Error processing {symbol}: {e}")
            print()
            continue

    # Generate comparison report
    generate_comparison_report(profiles, configs)

    return profiles, configs

def generate_comparison_report(profiles: dict, configs: dict):
    """Generate a comparison report across all assets."""
    print("📊 MULTI-ASSET COMPARISON REPORT")
    print("=" * 70)

    if not profiles:
        print("❌ No profiles generated")
        return

    # Create comparison table
    comparison = {}

    for symbol, profile in profiles.items():
        vol = profile['volatility']
        liq = profile['liquidity']
        config = configs[symbol]

        comparison[symbol] = {
            'atr_p50': vol['atr']['p50'],
            'atr_p80': vol['atr']['p80'],
            'volatility_regime': vol['regime'],
            'volume_z70': liq['volume_z_scores']['z70'],
            'volume_z85': liq['volume_z_scores']['z85'],
            'spread_estimate': liq['spread_estimate'],
            'slippage_factor': liq['slippage_factor'],
            'risk_scalar': config['risk']['position_size_scalar'],
            'spring_wick_atr': config['wyckoff']['spring_wick_min_atr'],
            'volume_spike_z': config['wyckoff']['volume_spike_z'],
            'stop_atr_mult': config['risk']['atr_multiplier_stop'],
            'target_atr_mult': config['risk']['atr_multiplier_target'],
            'btc_correlation': profiles[symbol].get('correlations', {}).get('full_period', 0)
        }

    # Display formatted table
    print("\n🎯 VOLATILITY CHARACTERISTICS:")
    print("-" * 70)
    print(f"{'Asset':<8} {'ATR p50':<8} {'ATR p80':<8} {'Regime':<8} {'Risk Scalar':<12}")
    print("-" * 70)
    for symbol, data in comparison.items():
        print(f"{symbol:<8} {data['atr_p50']:<8.2f} {data['atr_p80']:<8.2f} "
              f"{data['volatility_regime']:<8} {data['risk_scalar']:<12.2f}")

    print("\n💧 LIQUIDITY CHARACTERISTICS:")
    print("-" * 70)
    print(f"{'Asset':<8} {'Vol Z70':<8} {'Vol Z85':<8} {'Spread %':<10} {'Slippage':<10}")
    print("-" * 70)
    for symbol, data in comparison.items():
        print(f"{symbol:<8} {data['volume_z70']:<8.2f} {data['volume_z85']:<8.2f} "
              f"{data['spread_estimate']:<10.3f} {data['slippage_factor']:<10.3f}")

    print("\n⚙️ ADAPTIVE PARAMETERS:")
    print("-" * 70)
    print(f"{'Asset':<8} {'Spring ATR':<10} {'Vol Spike Z':<12} {'Stop Mult':<10} {'Target Mult':<12}")
    print("-" * 70)
    for symbol, data in comparison.items():
        print(f"{symbol:<8} {data['spring_wick_atr']:<10.2f} {data['volume_spike_z']:<12.2f} "
              f"{data['stop_atr_mult']:<10.1f} {data['target_atr_mult']:<12.1f}")

    print("\n🔗 CORRELATION TO BTC:")
    print("-" * 70)
    for symbol, data in comparison.items():
        corr = data['btc_correlation']
        corr_strength = "Strong" if abs(corr) > 0.7 else "Moderate" if abs(corr) > 0.4 else "Weak"
        print(f"{symbol:<8} {corr:<8.3f} ({corr_strength})")

    # Save comparison to file
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    comparison_file = f'multi_asset_comparison_{timestamp}.json'

    with open(comparison_file, 'w') as f:
        json.dump(comparison, f, indent=2)

    print(f"\n💾 Comparison saved to: {comparison_file}")

    # Key insights
    print("\n🔍 KEY INSIGHTS:")
    print("-" * 70)

    # Find highest/lowest volatility
    vol_sorted = sorted(comparison.items(), key=lambda x: x[1]['atr_p50'])
    print(f"🔸 Lowest Volatility: {vol_sorted[0][0]} ({vol_sorted[0][1]['atr_p50']:.2f}% ATR)")
    print(f"🔸 Highest Volatility: {vol_sorted[-1][0]} ({vol_sorted[-1][1]['atr_p50']:.2f}% ATR)")

    # Risk scaling insights
    risk_sorted = sorted(comparison.items(), key=lambda x: x[1]['risk_scalar'])
    print(f"🔸 Most Conservative: {risk_sorted[0][0]} ({risk_sorted[0][1]['risk_scalar']:.2f}x size)")
    print(f"🔸 Most Aggressive: {risk_sorted[-1][0]} ({risk_sorted[-1][1]['risk_scalar']:.2f}x size)")

    # Correlation insights
    if any(data['btc_correlation'] != 0 for data in comparison.values()):
        corr_sorted = sorted([(k, v['btc_correlation']) for k, v in comparison.items()
                             if v['btc_correlation'] != 0], key=lambda x: abs(x[1]), reverse=True)
        if corr_sorted:
            print(f"🔸 Most BTC-Correlated: {corr_sorted[0][0]} ({corr_sorted[0][1]:.3f})")

    print("\n🎯 DEPLOYMENT RECOMMENDATIONS:")
    print("-" * 70)

    for symbol, data in comparison.items():
        if data['volatility_regime'] == 'low' and data['risk_scalar'] >= 0.75:
            rec = "✅ RECOMMENDED - Low vol, stable"
        elif data['volatility_regime'] == 'medium' and data['risk_scalar'] >= 0.5:
            rec = "⚠️  MODERATE - Needs monitoring"
        else:
            rec = "🔴 HIGH RISK - Reduce size or avoid"

        print(f"{symbol:<8} {rec}")

def main():
    """Main execution function."""
    profiles, configs = generate_asset_profiles()

    print("\n🚀 MULTI-ASSET PROFILING COMPLETE!")
    print("=" * 70)
    print("Next steps:")
    print("1. Review generated profiles in profiles/ directory")
    print("2. Test adaptive configs in configs/adaptive/ directory")
    print("3. Run backtests with asset-specific parameters")
    print("4. Monitor performance across different volatility regimes")
    print("\nBull Machine v1.7.2 is now ready for multi-asset deployment! 🎯")

if __name__ == "__main__":
    main()