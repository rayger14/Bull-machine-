#!/usr/bin/env python3
"""
System B0 Diagnostic Analysis

Quick diagnostic to understand why System B0 performs differently across regimes.
Analyzes:
- Volatility patterns (ATR distributions)
- Drawdown characteristics
- Win/loss distributions
- Regime-specific behaviors

Usage:
    python bin/diagnose_system_b0.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
from datetime import datetime
import json

def load_feature_data():
    """Load BTC feature data for analysis."""
    print("Loading feature data...")
    data_path = Path("data/features_mtf/BTC_1H_2022-01-01_to_2024-12-31.parquet")

    if not data_path.exists():
        print(f"ERROR: Data file not found: {data_path}")
        print("Please ensure feature data is available.")
        return None

    df = pd.read_parquet(data_path)
    print(f"Loaded {len(df)} bars from {df.index[0]} to {df.index[-1]}")
    return df


def analyze_volatility_regimes(df):
    """Analyze ATR distributions across different periods."""
    print("\n" + "="*80)
    print("VOLATILITY ANALYSIS (ATR)")
    print("="*80)

    periods = {
        '2022 (Bear)': ('2022-01-01', '2022-12-31'),
        '2023 (Recovery)': ('2023-01-01', '2023-12-31'),
        '2024 (Bull)': ('2024-01-01', '2024-12-31')
    }

    for name, (start, end) in periods.items():
        period_df = df[start:end]

        if 'atr_14' not in period_df.columns:
            print(f"\n{name}:")
            print("  ATR_14 not available in dataset")
            continue

        atr = period_df['atr_14']
        close = period_df['close']
        atr_pct = (atr / close * 100).dropna()

        print(f"\n{name}:")
        print(f"  ATR (absolute):  Mean: ${atr.mean():,.0f}, Median: ${atr.median():,.0f}")
        print(f"  ATR (%):         Mean: {atr_pct.mean():.2f}%, Median: {atr_pct.median():.2f}%")
        print(f"  ATR (% range):   Min: {atr_pct.min():.2f}%, Max: {atr_pct.max():.2f}%")
        print(f"  Volatility:      Std: {atr_pct.std():.2f}%")


def analyze_drawdowns(df):
    """Analyze drawdown patterns and recovery characteristics."""
    print("\n" + "="*80)
    print("DRAWDOWN ANALYSIS")
    print("="*80)

    periods = {
        '2022 (Bear)': ('2022-01-01', '2022-12-31'),
        '2023 (Recovery)': ('2023-01-01', '2023-12-31'),
        '2024 (Bull)': ('2024-01-01', '2024-12-31')
    }

    for name, (start, end) in periods.items():
        period_df = df[start:end]
        close = period_df['close']

        # Calculate drawdown from 30-day rolling high
        rolling_high = close.rolling(720).max()  # 30 days * 24 hours
        drawdown = (close - rolling_high) / rolling_high

        # Count -15% drawdown events (System B0 entry threshold)
        deep_drawdowns = drawdown <= -0.15
        num_signals = deep_drawdowns.sum()

        # Analyze drawdown distribution
        dd_values = drawdown.dropna()

        print(f"\n{name}:")
        print(f"  Entry signals (-15% DD):  {num_signals} bars")
        print(f"  Average drawdown:         {dd_values.mean():.2%}")
        print(f"  Median drawdown:          {dd_values.median():.2%}")
        print(f"  Max drawdown:             {dd_values.min():.2%}")
        print(f"  Time in drawdown > -10%:  {(dd_values <= -0.10).sum()} bars ({(dd_values <= -0.10).mean():.1%})")

        # Analyze recovery from deep drawdowns
        if num_signals > 0:
            analyze_recovery_patterns(close, drawdown, name)


def analyze_recovery_patterns(close, drawdown, period_name):
    """Analyze recovery patterns after deep drawdowns."""
    deep_dd_indices = np.where(drawdown <= -0.15)[0]

    if len(deep_dd_indices) == 0:
        return

    recoveries_8pct = 0
    recoveries_10pct = 0
    avg_bars_to_8pct = []

    for idx in deep_dd_indices[:100]:  # Sample first 100 to avoid long computation
        if idx + 720 >= len(close):  # Need at least 30 days forward
            continue

        entry_price = close.iloc[idx]
        forward_prices = close.iloc[idx:idx+720]  # Next 30 days

        # Check if 8% profit target hit
        target_8pct = entry_price * 1.08
        if (forward_prices >= target_8pct).any():
            recoveries_8pct += 1
            bars_to_target = np.argmax(forward_prices >= target_8pct)
            avg_bars_to_8pct.append(bars_to_target)

        # Check if 10% profit target hit
        target_10pct = entry_price * 1.10
        if (forward_prices >= target_10pct).any():
            recoveries_10pct += 1

    sample_size = min(100, len(deep_dd_indices))
    print(f"  Recovery analysis (sample: {sample_size} signals):")
    print(f"    Hit +8% target:    {recoveries_8pct}/{sample_size} ({recoveries_8pct/sample_size:.1%})")
    print(f"    Hit +10% target:   {recoveries_10pct}/{sample_size} ({recoveries_10pct/sample_size:.1%})")
    if avg_bars_to_8pct:
        print(f"    Avg time to +8%:   {np.mean(avg_bars_to_8pct):.0f} hours")


def analyze_price_action(df):
    """Analyze basic price action statistics."""
    print("\n" + "="*80)
    print("PRICE ACTION ANALYSIS")
    print("="*80)

    periods = {
        '2022 (Bear)': ('2022-01-01', '2022-12-31'),
        '2023 (Recovery)': ('2023-01-01', '2023-12-31'),
        '2024 (Bull)': ('2024-01-01', '2024-12-31')
    }

    for name, (start, end) in periods.items():
        period_df = df[start:end]
        close = period_df['close']

        returns = close.pct_change()

        print(f"\n{name}:")
        print(f"  Start price:     ${close.iloc[0]:,.0f}")
        print(f"  End price:       ${close.iloc[-1]:,.0f}")
        print(f"  Total return:    {(close.iloc[-1] / close.iloc[0] - 1):.1%}")
        print(f"  Volatility:      {returns.std() * np.sqrt(24*365):.1%} (annualized)")
        print(f"  Positive bars:   {(returns > 0).sum()} ({(returns > 0).mean():.1%})")
        print(f"  Negative bars:   {(returns < 0).sum()} ({(returns < 0).mean():.1%})")


def generate_recommendations():
    """Generate diagnostic-based recommendations."""
    print("\n" + "="*80)
    print("DIAGNOSTIC RECOMMENDATIONS")
    print("="*80)

    print("""
Based on this diagnostic analysis, consider:

1. VOLATILITY ADAPTATION:
   - If ATR varies significantly across periods, implement regime-based parameter scaling
   - Adjust buy_threshold based on current volatility regime
   - Scale profit_target based on ATR

2. RECOVERY PATTERN ANALYSIS:
   - If recovery rates differ across periods, adjust profit_target per regime
   - Consider adaptive profit targets based on recent recovery patterns
   - Test multiple profit target levels (6%, 8%, 10%, 12%)

3. SIGNAL FREQUENCY:
   - If signal count varies dramatically, may need additional filters
   - High signal count + low PF suggests poor signal quality
   - Consider adding volume confirmation or Wyckoff filters

4. REGIME DETECTION:
   - Implement GMM-based regime classifier
   - Route to different parameter sets based on regime
   - Bear market: different thresholds than bull market

5. PARAMETER OPTIMIZATION:
   - Re-optimize buy_threshold and profit_target on FULL period (2022-2024)
   - Use walk-forward validation
   - Target realistic PF: 1.5-2.0 instead of 3.17

Next Steps:
1. Review the analysis above
2. Identify which regime characteristics differ most
3. Implement adaptive parameters or abandon simple baseline
4. Re-validate on full period with new approach
""")


def main():
    """Main diagnostic entry point."""
    print("="*80)
    print("SYSTEM B0 DIAGNOSTIC ANALYSIS")
    print("="*80)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    print("Analyzing why System B0 performs differently across market regimes...")
    print()

    # Load data
    df = load_feature_data()
    if df is None:
        return 1

    # Run analyses
    analyze_volatility_regimes(df)
    analyze_drawdowns(df)
    analyze_price_action(df)
    generate_recommendations()

    print("\n" + "="*80)
    print("DIAGNOSTIC COMPLETE")
    print("="*80)
    print(f"Completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    print("Review the analysis above to understand regime-specific behaviors.")
    print("Use findings to guide System B0 improvements or pivot to different approach.")

    return 0


if __name__ == '__main__':
    sys.exit(main())
