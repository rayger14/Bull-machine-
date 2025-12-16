#!/usr/bin/env python3
"""
Validate Temporal Confluence Layer on Historical Events

Tests the temporal fusion layer's ability to detect high confluence zones at
major market turning points:
    - LUNA collapse (May 2022)
    - FTX collapse (Nov 2022)
    - June 18, 2022 capitulation bottom
    - Other major bottoms/tops

Expected behavior:
    - High confluence (>0.70) at major bottoms
    - Low confluence (<0.30) at climax tops
    - Neutral confluence (0.30-0.70) during trends

Usage:
    python bin/validate_temporal_confluence.py \
        --data data/features/btc_1h_temporal.parquet \
        --config configs/temporal_fusion_config.json \
        --output results/temporal_validation.csv

Author: Bull Machine v2.0 - Temporal Intelligence Validation
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from datetime import datetime, timedelta

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from engine.temporal.temporal_fusion import TemporalFusionEngine

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# Major historical events to validate
HISTORICAL_EVENTS = [
    {
        'name': 'June 18, 2022 - Capitulation Bottom',
        'date': '2022-06-18',
        'type': 'bottom',
        'expected_confluence': 'high',  # >0.70
        'window_hours': 48,  # Check ±48 hours
        'description': 'Three Arrows Capital liquidation, extreme fear'
    },
    {
        'name': 'LUNA Collapse - May 2022',
        'date': '2022-05-12',
        'type': 'bottom',
        'expected_confluence': 'high',
        'window_hours': 72,
        'description': 'UST depeg, LUNA death spiral'
    },
    {
        'name': 'FTX Collapse - Nov 2022',
        'date': '2022-11-09',
        'type': 'bottom',
        'expected_confluence': 'high',
        'window_hours': 72,
        'description': 'FTX bankruptcy, Alameda liquidations'
    },
    {
        'name': 'March 2023 Banking Crisis',
        'date': '2023-03-10',
        'type': 'bottom',
        'expected_confluence': 'high',
        'window_hours': 48,
        'description': 'SVB collapse, Silvergate exit'
    },
    {
        'name': 'Nov 2021 Top',
        'date': '2021-11-10',
        'type': 'top',
        'expected_confluence': 'low',  # <0.30
        'window_hours': 48,
        'description': 'BTC all-time high $69k, extreme greed'
    },
    {
        'name': 'April 2021 Top',
        'date': '2021-04-14',
        'type': 'top',
        'expected_confluence': 'low',
        'window_hours': 48,
        'description': 'First major top before May crash'
    }
]


def load_data(data_path: Path) -> pd.DataFrame:
    """Load feature data with temporal scores."""
    logger.info(f"Loading data: {data_path}")

    if data_path.suffix == '.parquet':
        df = pd.read_parquet(data_path)
    elif data_path.suffix == '.csv':
        df = pd.read_csv(data_path, index_col=0, parse_dates=True)
    else:
        raise ValueError(f"Unsupported format: {data_path.suffix}")

    # Ensure datetime index
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)

    logger.info(f"Loaded {len(df)} rows from {df.index[0]} to {df.index[-1]}")

    return df


def validate_event(df: pd.DataFrame, event: dict) -> dict:
    """
    Validate temporal confluence around a historical event.

    Returns:
        Dict with validation results
    """
    event_date = pd.Timestamp(event['date'])
    window = timedelta(hours=event['window_hours'])

    # Get window around event
    start_date = event_date - window
    end_date = event_date + window

    window_df = df.loc[start_date:end_date]

    if len(window_df) == 0:
        logger.warning(f"No data found for {event['name']}")
        return {
            'event': event['name'],
            'status': 'NO_DATA',
            'passed': False
        }

    # Get temporal confluence scores
    if 'temporal_confluence' not in window_df.columns:
        logger.error(f"temporal_confluence column not found!")
        return {
            'event': event['name'],
            'status': 'MISSING_FEATURE',
            'passed': False
        }

    confluence = window_df['temporal_confluence']

    # Find max and min confluence in window
    max_conf = confluence.max()
    min_conf = confluence.min()
    mean_conf = confluence.mean()

    # Find exact event date confluence (or closest)
    if event_date in confluence.index:
        event_conf = confluence.loc[event_date]
    else:
        # Find closest
        closest_idx = confluence.index.get_indexer([event_date], method='nearest')[0]
        event_conf = confluence.iloc[closest_idx]

    # Validate based on expected confluence
    expected = event['expected_confluence']
    passed = False

    if expected == 'high':
        # Expect max confluence >0.70 in window
        passed = max_conf >= 0.70
        threshold = 0.70
        actual = max_conf
        condition = "max_conf >= 0.70"
    elif expected == 'low':
        # Expect min confluence <0.30 in window
        passed = min_conf <= 0.30
        threshold = 0.30
        actual = min_conf
        condition = "min_conf <= 0.30"
    else:
        # Neutral
        passed = (mean_conf >= 0.30) and (mean_conf <= 0.70)
        threshold = (0.30, 0.70)
        actual = mean_conf
        condition = "0.30 <= mean_conf <= 0.70"

    result = {
        'event': event['name'],
        'date': event_date.strftime('%Y-%m-%d'),
        'type': event['type'],
        'expected': expected,
        'passed': passed,
        'max_confluence': max_conf,
        'min_confluence': min_conf,
        'mean_confluence': mean_conf,
        'event_confluence': event_conf,
        'threshold': threshold,
        'actual_value': actual,
        'condition': condition,
        'description': event['description'],
        'window_bars': len(window_df)
    }

    # Log result
    status = "✓ PASS" if passed else "✗ FAIL"
    logger.info(
        f"{status} | {event['name']:<40} | "
        f"Expected: {expected:>7} | Actual: {actual:.3f} | "
        f"Threshold: {threshold}"
    )

    return result


def plot_event_confluence(df: pd.DataFrame, event: dict, output_dir: Path):
    """Plot temporal confluence around event."""
    event_date = pd.Timestamp(event['date'])
    window = timedelta(hours=event['window_hours'] * 2)  # Wider window for context

    start_date = event_date - window
    end_date = event_date + window

    window_df = df.loc[start_date:end_date]

    if len(window_df) == 0:
        logger.warning(f"No data for plotting {event['name']}")
        return

    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)

    # Plot 1: Price
    axes[0].plot(window_df.index, window_df['close'], color='blue', linewidth=1.5)
    axes[0].axvline(event_date, color='red', linestyle='--', linewidth=2, alpha=0.7, label='Event')
    axes[0].set_ylabel('Price (USD)', fontsize=12)
    axes[0].set_title(f'{event["name"]} - Temporal Confluence Analysis', fontsize=14, fontweight='bold')
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    # Plot 2: Temporal Confluence
    axes[1].plot(window_df.index, window_df['temporal_confluence'], color='purple', linewidth=2)
    axes[1].axhline(0.70, color='green', linestyle='--', alpha=0.5, label='High Threshold (0.70)')
    axes[1].axhline(0.30, color='orange', linestyle='--', alpha=0.5, label='Low Threshold (0.30)')
    axes[1].axvline(event_date, color='red', linestyle='--', linewidth=2, alpha=0.7)
    axes[1].fill_between(window_df.index, 0.70, 1.0, alpha=0.2, color='green', label='High Confluence')
    axes[1].fill_between(window_df.index, 0.0, 0.30, alpha=0.2, color='red', label='Low Confluence')
    axes[1].set_ylabel('Temporal Confluence', fontsize=12)
    axes[1].set_ylim(0, 1)
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    # Plot 3: Component Scores
    if all(col in window_df.columns for col in ['temporal_fib_score', 'temporal_gann_score', 'temporal_vol_score', 'temporal_emotional_score']):
        axes[2].plot(window_df.index, window_df['temporal_fib_score'], label='Fib (40%)', alpha=0.7)
        axes[2].plot(window_df.index, window_df['temporal_gann_score'], label='Gann (30%)', alpha=0.7)
        axes[2].plot(window_df.index, window_df['temporal_vol_score'], label='Vol (20%)', alpha=0.7)
        axes[2].plot(window_df.index, window_df['temporal_emotional_score'], label='Emotional (10%)', alpha=0.7)
        axes[2].axvline(event_date, color='red', linestyle='--', linewidth=2, alpha=0.7)
        axes[2].set_ylabel('Component Scores', fontsize=12)
        axes[2].set_xlabel('Date', fontsize=12)
        axes[2].set_ylim(0, 1)
        axes[2].grid(True, alpha=0.3)
        axes[2].legend(loc='upper right')

    plt.tight_layout()

    # Save plot
    output_dir.mkdir(parents=True, exist_ok=True)
    safe_name = event['name'].replace(' ', '_').replace(',', '').replace('-', '_')
    plot_path = output_dir / f'{safe_name}.png'
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()

    logger.info(f"  Saved plot: {plot_path}")


def generate_summary_report(results: list, output_path: Path):
    """Generate summary validation report."""
    total = len(results)
    passed = sum(1 for r in results if r['passed'])
    failed = total - passed

    report = []
    report.append("="*80)
    report.append("TEMPORAL CONFLUENCE VALIDATION REPORT")
    report.append("="*80)
    report.append("")
    report.append(f"Total Events Tested: {total}")
    report.append(f"Passed:              {passed} ({passed/total*100:.1f}%)")
    report.append(f"Failed:              {failed} ({failed/total*100:.1f}%)")
    report.append("")
    report.append("="*80)
    report.append("DETAILED RESULTS")
    report.append("="*80)
    report.append("")

    for r in results:
        status = "✓ PASS" if r['passed'] else "✗ FAIL"
        report.append(f"{status} | {r['event']}")
        report.append(f"  Date:        {r['date']}")
        report.append(f"  Type:        {r['type']}")
        report.append(f"  Expected:    {r['expected']} confluence")
        report.append(f"  Condition:   {r['condition']}")
        report.append(f"  Actual:      {r['actual_value']:.3f}")
        report.append(f"  Range:       [{r['min_confluence']:.3f}, {r['max_confluence']:.3f}]")
        report.append(f"  Mean:        {r['mean_confluence']:.3f}")
        report.append(f"  Event Score: {r['event_confluence']:.3f}")
        report.append(f"  Description: {r['description']}")
        report.append("")

    report.append("="*80)
    report.append("INTERPRETATION")
    report.append("="*80)
    report.append("")
    report.append("High confluence (>0.70) zones indicate:")
    report.append("  - Multiple time cycles aligned (Fib, Gann, Volatility, Emotional)")
    report.append("  - Strong temporal pressure for price resolution")
    report.append("  - Higher conviction for signals in these zones")
    report.append("")
    report.append("Low confluence (<0.30) zones indicate:")
    report.append("  - Time cycles out of phase")
    report.append("  - Climax conditions (expansion exhaustion)")
    report.append("  - Reduced conviction, avoid entries")
    report.append("")

    # Write report
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        f.write('\n'.join(report))

    logger.info(f"\nValidation report saved to: {output_path}")

    # Print summary to console
    print("\n" + '\n'.join(report))


def main():
    parser = argparse.ArgumentParser(
        description='Validate temporal confluence on historical events'
    )
    parser.add_argument(
        '--data',
        type=Path,
        required=True,
        help='Input data with temporal features (from compute_temporal_features.py)'
    )
    parser.add_argument(
        '--output',
        type=Path,
        default=Path('results/temporal_validation_report.txt'),
        help='Output report path'
    )
    parser.add_argument(
        '--plots',
        type=Path,
        default=Path('results/temporal_plots'),
        help='Output directory for plots'
    )
    parser.add_argument(
        '--no-plots',
        action='store_true',
        help='Disable plot generation'
    )

    args = parser.parse_args()

    # Load data
    df = load_data(args.data)

    # Validate required features
    required = ['temporal_confluence', 'close']
    missing = [f for f in required if f not in df.columns]
    if missing:
        logger.error(f"Missing required features: {missing}")
        logger.error("Run compute_temporal_features.py first!")
        sys.exit(1)

    # Run validation on each event
    logger.info("\n" + "="*80)
    logger.info("VALIDATING HISTORICAL EVENTS")
    logger.info("="*80 + "\n")

    results = []
    for event in HISTORICAL_EVENTS:
        result = validate_event(df, event)
        results.append(result)

        # Generate plot
        if not args.no_plots and result.get('passed') is not None:
            plot_event_confluence(df, event, args.plots)

    # Generate summary report
    generate_summary_report(results, args.output)

    # Save results as CSV
    results_df = pd.DataFrame(results)
    csv_path = args.output.parent / 'temporal_validation_results.csv'
    results_df.to_csv(csv_path, index=False)
    logger.info(f"Results CSV saved to: {csv_path}")

    # Exit with appropriate code
    total_passed = sum(1 for r in results if r['passed'])
    if total_passed == len(results):
        logger.info("\n✓ All validations PASSED")
        sys.exit(0)
    else:
        logger.warning(f"\n⚠ {len(results) - total_passed} validations FAILED")
        sys.exit(1)


if __name__ == '__main__':
    main()
