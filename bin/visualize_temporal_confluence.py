#!/usr/bin/env python3
"""
Visualize Temporal Fibonacci Confluence Events

Creates visualization showing:
1. Price chart with Fibonacci confluence markers
2. bars_since_* timing features
3. fib_time_score heatmap
4. Major Wyckoff events
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime
import sys

def plot_temporal_confluence(df, start_date=None, end_date=None, output_file='temporal_confluence_chart.png'):
    """
    Create comprehensive temporal confluence visualization.

    Args:
        df: DataFrame with temporal timing features
        start_date: Optional start date for visualization
        end_date: Optional end date for visualization
        output_file: Path to save output image
    """
    # Filter date range if provided
    if start_date:
        df = df[df.index >= start_date]
    if end_date:
        df = df[df.index <= end_date]

    if len(df) == 0:
        print("No data in selected date range")
        return

    # Create figure with subplots
    fig, axes = plt.subplots(4, 1, figsize=(16, 12), sharex=True)
    fig.suptitle(f'Temporal Fibonacci Confluence Analysis\n{df.index[0].date()} to {df.index[-1].date()}',
                 fontsize=14, fontweight='bold')

    # Subplot 1: Price with Fib confluence markers
    ax1 = axes[0]
    ax1.plot(df.index, df['close'], linewidth=1, color='black', label='BTC Price')

    # Mark high-quality Fibonacci confluences
    high_fib = df[df['fib_time_score'] >= 0.667]
    if len(high_fib) > 0:
        ax1.scatter(high_fib.index, high_fib['close'],
                   c='red', s=100, marker='*',
                   label=f'High Fib Confluence (n={len(high_fib)})', zorder=5)

    # Mark medium Fibonacci confluences
    med_fib = df[(df['fib_time_score'] >= 0.333) & (df['fib_time_score'] < 0.667)]
    if len(med_fib) > 0:
        ax1.scatter(med_fib.index, med_fib['close'],
                   c='orange', s=50, marker='o', alpha=0.6,
                   label=f'Medium Fib Confluence (n={len(med_fib)})', zorder=4)

    ax1.set_ylabel('Price (USD)', fontweight='bold')
    ax1.legend(loc='upper left')
    ax1.grid(True, alpha=0.3)
    ax1.set_title('Price Chart with Fibonacci Time Confluences')

    # Subplot 2: Key bars_since timing features
    ax2 = axes[1]

    # Plot 3-4 key timing features
    bars_since_features = ['bars_since_ar', 'bars_since_sos_long', 'bars_since_spring']
    colors = ['blue', 'green', 'purple']

    for feat, color in zip(bars_since_features, colors):
        if feat in df.columns:
            ax2.plot(df.index, df[feat], linewidth=1, color=color,
                    label=feat.replace('bars_since_', '').upper(), alpha=0.7)

    # Mark Fibonacci levels
    for fib_level in [13, 21, 34, 55, 89]:
        ax2.axhline(y=fib_level, color='red', linestyle='--', linewidth=0.5, alpha=0.5)
        ax2.text(df.index[0], fib_level, f' Fib {fib_level}',
                va='bottom', fontsize=8, color='red')

    ax2.set_ylabel('Bars Since Event', fontweight='bold')
    ax2.legend(loc='upper right')
    ax2.grid(True, alpha=0.3)
    ax2.set_title('Wyckoff Event Timing (bars_since_*)')
    ax2.set_ylim(0, 150)  # Focus on short-term patterns

    # Subplot 3: Fibonacci time score heatmap
    ax3 = axes[2]

    # Create bar chart of fib_time_score
    colors_score = ['green' if score >= 0.667 else 'orange' if score >= 0.333 else 'gray'
                    for score in df['fib_time_score']]
    ax3.bar(df.index, df['fib_time_score'], width=0.04, color=colors_score, alpha=0.6)

    ax3.axhline(y=0.667, color='red', linestyle='--', linewidth=1, label='High Quality (≥0.667)')
    ax3.axhline(y=0.333, color='orange', linestyle='--', linewidth=1, label='Medium Quality (≥0.333)')

    ax3.set_ylabel('Confluence Score', fontweight='bold')
    ax3.set_ylim(0, 1.1)
    ax3.legend(loc='upper right')
    ax3.grid(True, alpha=0.3)
    ax3.set_title('Fibonacci Time Confluence Score')

    # Subplot 4: Cycle indicators
    ax4 = axes[3]

    # Plot volatility cycle
    ax4.plot(df.index, df['volatility_cycle'], linewidth=1, color='purple',
            label='Volatility Cycle', alpha=0.7)
    ax4.axhline(y=0.5, color='gray', linestyle='--', linewidth=0.5)

    # Mark Gann cycles
    gann_events = df[df['gann_cycle']]
    if len(gann_events) > 0:
        ax4.scatter(gann_events.index, [0.95] * len(gann_events),
                   c='red', s=50, marker='v', label=f'Gann Cycle (n={len(gann_events)})')

    ax4.set_ylabel('Cycle Indicator', fontweight='bold')
    ax4.set_xlabel('Date', fontweight='bold')
    ax4.set_ylim(0, 1.0)
    ax4.legend(loc='upper right')
    ax4.grid(True, alpha=0.3)
    ax4.set_title('Market Cycles')

    # Format x-axis
    ax4.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    ax4.xaxis.set_major_locator(mdates.AutoDateLocator())
    plt.setp(ax4.xaxis.get_majorticklabels(), rotation=45, ha='right')

    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"Chart saved to: {output_file}")
    print(f"Visualization complete!")

    return fig


def print_confluence_summary(df):
    """Print summary statistics of confluence events."""
    print("\n" + "=" * 70)
    print("TEMPORAL CONFLUENCE SUMMARY")
    print("=" * 70)
    print()

    # Overall stats
    total_bars = len(df)
    fib_events = df['fib_time_cluster'].sum()
    high_quality = len(df[df['fib_time_score'] >= 0.667])
    gann_events = df['gann_cycle'].sum()

    print(f"Total bars analyzed: {total_bars:,}")
    print(f"Fibonacci confluence events: {fib_events:,} ({fib_events/total_bars*100:.1f}%)")
    print(f"High-quality confluences (≥0.667): {high_quality:,} ({high_quality/total_bars*100:.1f}%)")
    print(f"Gann cycle events: {gann_events:,} ({gann_events/total_bars*100:.1f}%)")
    print()

    # Top confluence events
    print("TOP 10 FIBONACCI CONFLUENCE EVENTS:")
    top_events = df.nlargest(10, 'fib_time_score')

    for i, (idx, row) in enumerate(top_events.iterrows(), 1):
        print(f"\n{i}. {idx}")
        print(f"   Price: ${row['close']:,.2f}")
        print(f"   Fib Score: {row['fib_time_score']:.3f}")
        print(f"   Fib Target: {row['fib_time_target']}")

        # Show aligned events
        bars_since_cols = [c for c in df.columns if c.startswith('bars_since_')]
        aligned = []
        for col in bars_since_cols:
            val = row[col]
            if pd.notna(val):
                for fib in [13, 21, 34, 55, 89, 144]:
                    if abs(val - fib) <= 1:
                        event_name = col.replace('bars_since_', '').upper()
                        aligned.append(f"{event_name}={int(val)}")
                        break

        if aligned:
            print(f"   Aligned: {' | '.join(aligned)}")

    print()


if __name__ == '__main__':
    # Load feature store
    print("Loading feature store...")
    df = pd.read_parquet('data/features_mtf/BTC_1H_2022-01-01_to_2024-12-31.parquet')
    print(f"Loaded {len(df):,} bars from {df.index[0]} to {df.index[-1]}")
    print()

    # Print summary
    print_confluence_summary(df)

    # Create visualizations for different periods
    periods = [
        ('2022 Bear Market', '2022-05-01', '2022-06-30', 'temporal_confluence_2022_bear.png'),
        ('2024 Bull Market', '2024-01-01', '2024-03-31', 'temporal_confluence_2024_bull.png'),
        ('Recent Period', '2024-10-01', '2024-12-31', 'temporal_confluence_recent.png'),
    ]

    print("\n" + "=" * 70)
    print("GENERATING VISUALIZATIONS")
    print("=" * 70)
    print()

    for name, start, end, filename in periods:
        print(f"Creating chart: {name}...")
        try:
            plot_temporal_confluence(df, start_date=start, end_date=end, output_file=filename)
        except Exception as e:
            print(f"  Error: {e}")

    print()
    print("=" * 70)
    print("VISUALIZATION COMPLETE")
    print("=" * 70)
    print()
    print("Generated files:")
    for name, start, end, filename in periods:
        print(f"  - {filename} ({name})")
