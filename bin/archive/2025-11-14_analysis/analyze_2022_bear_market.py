#!/usr/bin/env python3
"""
Analyze 2022 bear market conditions and validate proposed bear archetypes.

This script examines:
1. Market regime characteristics (funding, vol, liquidity, macro)
2. Why current bull archetypes failed
3. Pattern frequency and forward performance for proposed bear archetypes
4. Feature availability for implementation
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path
from typing import Dict, List, Tuple

# Load data
print("Loading 2022 feature data...")
df = pd.read_parquet('data/features_mtf/BTC_1H_2022-01-01_to_2023-12-31.parquet')
df_2022 = df[df.index < '2023-01-01'].copy()

print(f"Loaded {len(df_2022)} bars from 2022")
print(f"Date range: {df_2022.index[0]} to {df_2022.index[-1]}")

# Load trade data (CSV format despite .json extension)
trades = pd.read_csv('results/bear_patterns/2022_baseline_trades.json')
print(f"\nLoaded {len(trades)} trades from baseline backtest")
print(f"Win rate: {trades['trade_won'].mean():.1%}")
pf_denom = abs(trades[trades['r_multiple'] < 0]['r_multiple'].sum())
if pf_denom > 0:
    pf = trades[trades['r_multiple'] > 0]['r_multiple'].sum() / pf_denom
else:
    pf = 0.0
print(f"Profit factor: {pf:.2f}")

# ============================================================================
# 1. MARKET REGIME ANALYSIS
# ============================================================================
print("\n" + "="*80)
print("1. MARKET REGIME CHARACTERISTICS")
print("="*80)

regime_stats = {}

# Funding rates
if 'funding_Z' in df_2022.columns:
    funding_z = df_2022['funding_Z'].dropna()
    regime_stats['funding'] = {
        'mean': float(funding_z.mean()),
        'median': float(funding_z.median()),
        'std': float(funding_z.std()),
        'pct_positive': float((funding_z > 0).mean()),
        'pct_extreme_positive': float((funding_z > 1.5).mean()),
        'pct_extreme_negative': float((funding_z < -1.5).mean()),
        'max': float(funding_z.max()),
        'min': float(funding_z.min())
    }
    print("\nFunding Rate Z-Score:")
    print(f"  Mean: {regime_stats['funding']['mean']:.3f}")
    print(f"  Median: {regime_stats['funding']['median']:.3f}")
    print(f"  % Positive (longs pay shorts): {regime_stats['funding']['pct_positive']:.1%}")
    print(f"  % Extreme Positive (>1.5σ): {regime_stats['funding']['pct_extreme_positive']:.1%}")
    print(f"  % Extreme Negative (<-1.5σ): {regime_stats['funding']['pct_extreme_negative']:.1%}")

# Open Interest changes
if 'OI_CHANGE' in df_2022.columns:
    oi_change = df_2022['OI_CHANGE'].dropna()
    regime_stats['oi'] = {
        'mean': float(oi_change.mean()),
        'median': float(oi_change.median()),
        'std': float(oi_change.std()),
        'pct_large_drop': float((oi_change < -0.08).mean()),
        'pct_large_spike': float((oi_change > 0.08).mean())
    }
    print("\nOpen Interest Changes:")
    print(f"  Mean: {regime_stats['oi']['mean']:.3%}")
    print(f"  % Large Drops (< -8%): {regime_stats['oi']['pct_large_drop']:.1%}")
    print(f"  % Large Spikes (> 8%): {regime_stats['oi']['pct_large_spike']:.1%}")

# Volatility
if 'VIX_Z' in df_2022.columns:
    vix_z = df_2022['VIX_Z'].dropna()
    regime_stats['vix'] = {
        'mean': float(vix_z.mean()),
        'median': float(vix_z.median()),
        'pct_elevated': float((vix_z > 0.5).mean()),
        'pct_extreme': float((vix_z > 1.5).mean())
    }
    print("\nVIX Z-Score:")
    print(f"  Mean: {regime_stats['vix']['mean']:.3f}")
    print(f"  % Elevated (>0.5σ): {regime_stats['vix']['pct_elevated']:.1%}")
    print(f"  % Extreme (>1.5σ): {regime_stats['vix']['pct_extreme']:.1%}")

# Volume
if 'volume_zscore' in df_2022.columns:
    vol_z = df_2022['volume_zscore'].dropna()
    regime_stats['volume'] = {
        'mean': float(vol_z.mean()),
        'median': float(vol_z.median()),
        'pct_climax': float((vol_z > 1.8).mean()),
        'pct_very_low': float((vol_z < -1.0).mean())
    }
    print("\nVolume Z-Score:")
    print(f"  Mean: {regime_stats['volume']['mean']:.3f}")
    print(f"  % Climax (>1.8σ): {regime_stats['volume']['pct_climax']:.1%}")
    print(f"  % Very Low (<-1.0σ): {regime_stats['volume']['pct_very_low']:.1%}")

# Liquidity (if available)
if 'liquidity_score' in df_2022.columns:
    liq = df_2022['liquidity_score'].dropna()
    regime_stats['liquidity'] = {
        'mean': float(liq.mean()),
        'median': float(liq.median()),
        'pct_very_low': float((liq < 0.20).mean()),
        'pct_low': float((liq < 0.30).mean())
    }
    print("\nLiquidity Score:")
    print(f"  Mean: {regime_stats['liquidity']['mean']:.3f}")
    print(f"  % Very Low (<0.20): {regime_stats['liquidity']['pct_very_low']:.1%}")
    print(f"  % Low (<0.30): {regime_stats['liquidity']['pct_low']:.1%}")

# Macro backdrop
if 'DXY_Z' in df_2022.columns:
    dxy_z = df_2022['DXY_Z'].dropna()
    regime_stats['dxy'] = {
        'mean': float(dxy_z.mean()),
        'median': float(dxy_z.median()),
        'pct_strong': float((dxy_z > 0.8).mean())
    }
    print("\nDXY Z-Score (Dollar Strength):")
    print(f"  Mean: {regime_stats['dxy']['mean']:.3f}")
    print(f"  % Strong Dollar (>0.8σ): {regime_stats['dxy']['pct_strong']:.1%}")

if 'YC_SPREAD' in df_2022.columns:
    yc = df_2022['YC_SPREAD'].dropna()
    regime_stats['yield_curve'] = {
        'mean': float(yc.mean()),
        'median': float(yc.median()),
        'pct_inverted': float((yc < -0.01).mean())
    }
    print("\nYield Curve Spread:")
    print(f"  Mean: {regime_stats['yield_curve']['mean']:.4f}")
    print(f"  % Inverted (<-0.01): {regime_stats['yield_curve']['pct_inverted']:.1%}")

# ============================================================================
# 2. WHY BULL ARCHETYPES FAILED
# ============================================================================
print("\n" + "="*80)
print("2. WHY BULL ARCHETYPES FAILED")
print("="*80)

# Analyze losing trades
losing_trades = trades[trades['trade_won'] == 0]
print(f"\nLosing Trades: {len(losing_trades)} of {len(trades)} ({len(losing_trades)/len(trades):.1%})")

# Which archetypes triggered the losses?
archetype_cols = [c for c in trades.columns if c.startswith('archetype_') and c != 'archetype_false_break_reversal']
for col in archetype_cols:
    count = losing_trades[col].sum()
    if count > 0:
        arch_name = col.replace('archetype_', '')
        print(f"  {arch_name}: {count} losing trades")

# Analyze market conditions at losing trade entries
print("\nMarket Conditions at Losing Trade Entries:")
if len(losing_trades) > 0:
    print(f"  Avg Volume Z: {losing_trades['volume_zscore'].mean():.2f}")
    print(f"  Avg VIX Z: {losing_trades['vix_z_score'].mean():.2f}")
    print(f"  Avg Fusion: {losing_trades['entry_fusion_score'].mean():.3f}")
    print(f"  Avg Liquidity: {losing_trades['entry_liquidity_score'].mean():.3f}")

    # Regime distribution
    print("\n  Regime Distribution:")
    for regime in ['risk_on', 'neutral', 'risk_off', 'crisis']:
        col = f'macro_regime_{regime}'
        if col in losing_trades.columns:
            pct = losing_trades[col].mean()
            print(f"    {regime}: {pct:.1%}")

# ============================================================================
# 3. PROPOSED BEAR PATTERN VALIDATION
# ============================================================================
print("\n" + "="*80)
print("3. PROPOSED BEAR PATTERN VALIDATION")
print("="*80)

# Helper to calculate forward returns
def calc_forward_returns(df, horizons=[1, 4, 24]):
    """Calculate forward returns at various horizons"""
    results = {}
    for h in horizons:
        fwd = df['close'].shift(-h) / df['close'] - 1
        results[f'{h}h'] = fwd
    return pd.DataFrame(results)

# Calculate forward returns for all 2022 data
fwd_rets = calc_forward_returns(df_2022)
df_2022 = pd.concat([df_2022, fwd_rets], axis=1)

# Create liquidity proxy for all patterns
if 'liquidity_score' in df_2022.columns:
    liq_proxy = df_2022['liquidity_score']
else:
    # Approximate liquidity score from BOMS strength
    liq_proxy = df_2022['tf1d_boms_strength'].fillna(0) * 0.5

# S1: Liquidity Vacuum Cascade
print("\n--- S1: Liquidity Vacuum Cascade ---")
print("Logic: liq < 0.20 + vol_z > 1.0 + tf4h trend down")
s1_matches = pd.Series(False, index=df_2022.index)
liq_condition = liq_proxy < 0.20

vol_condition = df_2022.get('volume_zscore', pd.Series(0, index=df_2022.index)) > 1.0
tf4h_down = df_2022.get('tf4h_external_trend', pd.Series('', index=df_2022.index)) == 'down'

s1_matches = liq_condition & vol_condition & tf4h_down
print(f"Occurrences: {s1_matches.sum()} ({s1_matches.mean():.1%} of bars)")
if s1_matches.sum() > 0:
    s1_fwd = df_2022.loc[s1_matches, ['1h', '4h', '24h']].dropna()
    print(f"Forward Returns (mean):")
    print(f"  1h: {s1_fwd['1h'].mean():.2%}")
    print(f"  4h: {s1_fwd['4h'].mean():.2%}")
    print(f"  24h: {s1_fwd['24h'].mean():.2%}")
    print(f"Win Rate (1h < 0): {(s1_fwd['1h'] < 0).mean():.1%}")

# S2: Failed Rally Rejection
print("\n--- S2: Failed Rally Rejection ---")
print("Logic: rsi > 70 + vol_z < 0.5 + wick > 2.0")
rsi_condition = df_2022.get('rsi_14', pd.Series(50, index=df_2022.index)) > 70
vol_low = df_2022.get('volume_zscore', pd.Series(0, index=df_2022.index)) < 0.5
# Wick ratio = (high - max(open, close)) / (high - low)
df_2022['wick_ratio_upper'] = (df_2022['high'] - df_2022[['open', 'close']].max(axis=1)) / (df_2022['high'] - df_2022['low'] + 1e-9)
wick_condition = df_2022['wick_ratio_upper'] > 0.4  # Upper wick > 40% of candle range

s2_matches = rsi_condition & vol_low & wick_condition
print(f"Occurrences: {s2_matches.sum()} ({s2_matches.mean():.1%} of bars)")
if s2_matches.sum() > 0:
    s2_fwd = df_2022.loc[s2_matches, ['1h', '4h', '24h']].dropna()
    print(f"Forward Returns (mean):")
    print(f"  1h: {s2_fwd['1h'].mean():.2%}")
    print(f"  4h: {s2_fwd['4h'].mean():.2%}")
    print(f"  24h: {s2_fwd['24h'].mean():.2%}")
    print(f"Win Rate (1h < 0): {(s2_fwd['1h'] < 0).mean():.1%}")

# S4: Distribution Climax
print("\n--- S4: Distribution Climax ---")
print("Logic: vol_z > 1.5 + liq < 0.25")
vol_climax = df_2022.get('volume_zscore', pd.Series(0, index=df_2022.index)) > 1.5
liq_low = liq_proxy < 0.25

s4_matches = vol_climax & liq_low
print(f"Occurrences: {s4_matches.sum()} ({s4_matches.mean():.1%} of bars)")
if s4_matches.sum() > 0:
    s4_fwd = df_2022.loc[s4_matches, ['1h', '4h', '24h']].dropna()
    print(f"Forward Returns (mean):")
    print(f"  1h: {s4_fwd['1h'].mean():.2%}")
    print(f"  4h: {s4_fwd['4h'].mean():.2%}")
    print(f"  24h: {s4_fwd['24h'].mean():.2%}")
    print(f"Win Rate (1h < 0): {(s4_fwd['1h'] < 0).mean():.1%}")

# S5: Long Squeeze (FIXED LOGIC)
print("\n--- S5: Long Squeeze Cascade (FIXED) ---")
print("Logic: funding_Z > 1.5 + oi_spike + rsi > 75")
funding_high = df_2022.get('funding_Z', pd.Series(0, index=df_2022.index)) > 1.5
oi_spike = df_2022.get('OI_CHANGE', pd.Series(0, index=df_2022.index)) > 0.05
rsi_high = df_2022.get('rsi_14', pd.Series(50, index=df_2022.index)) > 75

s5_matches = funding_high & oi_spike & rsi_high
print(f"Occurrences: {s5_matches.sum()} ({s5_matches.mean():.1%} of bars)")
if s5_matches.sum() > 0:
    s5_fwd = df_2022.loc[s5_matches, ['1h', '4h', '24h']].dropna()
    print(f"Forward Returns (mean):")
    print(f"  1h: {s5_fwd['1h'].mean():.2%}")
    print(f"  4h: {s5_fwd['4h'].mean():.2%}")
    print(f"  24h: {s5_fwd['24h'].mean():.2%}")
    print(f"Win Rate (1h < 0): {(s5_fwd['1h'] < 0).mean():.1%}")

# S8: Trend Exhaustion Fade
print("\n--- S8: Trend Exhaustion Fade ---")
print("Logic: adx > 35 + rsi > 70 + vol_z < 0.3")
adx_high = df_2022.get('adx_14', pd.Series(0, index=df_2022.index)) > 35
rsi_extreme = df_2022.get('rsi_14', pd.Series(50, index=df_2022.index)) > 70
vol_fade = df_2022.get('volume_zscore', pd.Series(0, index=df_2022.index)) < 0.3

s8_matches = adx_high & rsi_extreme & vol_fade
print(f"Occurrences: {s8_matches.sum()} ({s8_matches.mean():.1%} of bars)")
if s8_matches.sum() > 0:
    s8_fwd = df_2022.loc[s8_matches, ['1h', '4h', '24h']].dropna()
    print(f"Forward Returns (mean):")
    print(f"  1h: {s8_fwd['1h'].mean():.2%}")
    print(f"  4h: {s8_fwd['4h'].mean():.2%}")
    print(f"  24h: {s8_fwd['24h'].mean():.2%}")
    print(f"Win Rate (1h < 0): {(s8_fwd['1h'] < 0).mean():.1%}")

# ============================================================================
# 4. FEATURE AVAILABILITY CHECK
# ============================================================================
print("\n" + "="*80)
print("4. FEATURE AVAILABILITY CHECK")
print("="*80)

required_features = {
    'S1_liquidity_vacuum': ['liquidity_score', 'volume_zscore', 'tf4h_external_trend'],
    'S2_failed_rally': ['rsi_14', 'volume_zscore', 'high', 'low', 'open', 'close'],
    'S4_distribution': ['volume_zscore', 'liquidity_score'],
    'S5_long_squeeze': ['funding_Z', 'OI_CHANGE', 'rsi_14'],
    'S8_exhaustion': ['adx_14', 'rsi_14', 'volume_zscore']
}

print("\nPattern | Required Features | Available?")
print("-" * 70)
for pattern, features in required_features.items():
    available = all(f in df_2022.columns for f in features)
    missing = [f for f in features if f not in df_2022.columns]
    status = "✓ All available" if available else f"✗ Missing: {', '.join(missing)}"
    print(f"{pattern:25} | {', '.join(features[:3])}... | {status}")

# ============================================================================
# 5. EXPORT RESULTS
# ============================================================================
print("\n" + "="*80)
print("5. EXPORTING RESULTS")
print("="*80)

results = {
    'analysis_period': '2022-01-01 to 2022-12-31',
    'baseline_performance': {
        'trades': int(len(trades)),
        'profit_factor': float(pf),
        'win_rate': float(trades['trade_won'].mean()),
        'avg_r': float(trades['r_multiple'].mean())
    },
    'regime_characteristics': regime_stats,
    'proposed_patterns': {
        'S1_liquidity_vacuum': {
            'exists_in_data': bool(s1_matches.sum() > 0),
            'occurrences': int(s1_matches.sum()),
            'frequency_pct': float(s1_matches.mean()),
            'forward_performance': {
                '1h': float(df_2022.loc[s1_matches, '1h'].mean()) if s1_matches.sum() > 0 else None,
                '4h': float(df_2022.loc[s1_matches, '4h'].mean()) if s1_matches.sum() > 0 else None,
                '24h': float(df_2022.loc[s1_matches, '24h'].mean()) if s1_matches.sum() > 0 else None
            },
            'win_rate_1h': float((df_2022.loc[s1_matches, '1h'] < 0).mean()) if s1_matches.sum() > 0 else None
        },
        'S2_failed_rally': {
            'exists_in_data': bool(s2_matches.sum() > 0),
            'occurrences': int(s2_matches.sum()),
            'frequency_pct': float(s2_matches.mean()),
            'forward_performance': {
                '1h': float(df_2022.loc[s2_matches, '1h'].mean()) if s2_matches.sum() > 0 else None,
                '4h': float(df_2022.loc[s2_matches, '4h'].mean()) if s2_matches.sum() > 0 else None,
                '24h': float(df_2022.loc[s2_matches, '24h'].mean()) if s2_matches.sum() > 0 else None
            },
            'win_rate_1h': float((df_2022.loc[s2_matches, '1h'] < 0).mean()) if s2_matches.sum() > 0 else None
        },
        'S4_distribution': {
            'exists_in_data': bool(s4_matches.sum() > 0),
            'occurrences': int(s4_matches.sum()),
            'frequency_pct': float(s4_matches.mean()),
            'forward_performance': {
                '1h': float(df_2022.loc[s4_matches, '1h'].mean()) if s4_matches.sum() > 0 else None,
                '4h': float(df_2022.loc[s4_matches, '4h'].mean()) if s4_matches.sum() > 0 else None,
                '24h': float(df_2022.loc[s4_matches, '24h'].mean()) if s4_matches.sum() > 0 else None
            },
            'win_rate_1h': float((df_2022.loc[s4_matches, '1h'] < 0).mean()) if s4_matches.sum() > 0 else None
        },
        'S5_long_squeeze': {
            'exists_in_data': bool(s5_matches.sum() > 0),
            'occurrences': int(s5_matches.sum()),
            'frequency_pct': float(s5_matches.mean()),
            'forward_performance': {
                '1h': float(df_2022.loc[s5_matches, '1h'].mean()) if s5_matches.sum() > 0 else None,
                '4h': float(df_2022.loc[s5_matches, '4h'].mean()) if s5_matches.sum() > 0 else None,
                '24h': float(df_2022.loc[s5_matches, '24h'].mean()) if s5_matches.sum() > 0 else None
            },
            'win_rate_1h': float((df_2022.loc[s5_matches, '1h'] < 0).mean()) if s5_matches.sum() > 0 else None
        },
        'S8_exhaustion': {
            'exists_in_data': bool(s8_matches.sum() > 0),
            'occurrences': int(s8_matches.sum()),
            'frequency_pct': float(s8_matches.mean()),
            'forward_performance': {
                '1h': float(df_2022.loc[s8_matches, '1h'].mean()) if s8_matches.sum() > 0 else None,
                '4h': float(df_2022.loc[s8_matches, '4h'].mean()) if s8_matches.sum() > 0 else None,
                '24h': float(df_2022.loc[s8_matches, '24h'].mean()) if s8_matches.sum() > 0 else None
            },
            'win_rate_1h': float((df_2022.loc[s8_matches, '1h'] < 0).mean()) if s8_matches.sum() > 0 else None
        }
    },
    'feature_availability': {
        pattern: {
            'available': all(f in df_2022.columns for f in features),
            'missing': [f for f in features if f not in df_2022.columns]
        }
        for pattern, features in required_features.items()
    }
}

# Export to JSON
output_path = Path('results/bear_patterns/validation_2022.json')
output_path.parent.mkdir(parents=True, exist_ok=True)
with open(output_path, 'w') as f:
    json.dump(results, f, indent=2)

print(f"\nResults exported to: {output_path}")
print("\nAnalysis complete!")
