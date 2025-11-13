#!/usr/bin/env python3
"""
Router v10 Backtest - Full 2022-2024 Validation

Validates regime-adaptive config switching across bull and bear markets.

Usage:
    python3 bin/backtest_router_v10.py --asset BTC --start 2022-01-01 --end 2024-12-31

Target Performance:
    - PF > 1.8 (blended bull/bear)
    - DD ≤ 6%
    - Smoother equity curve than single-config

Components:
    - RegimeDetector: GMM v3.1 regime classification
    - EventCalendar: CPI/FOMC/NFP suppression windows
    - RouterV10: Confidence veto + event suppression + regime switching
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import argparse
from datetime import datetime
import json

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from engine.regime_detector import RegimeDetector
from engine.event_calendar import EventCalendar
from engine.router_v10 import RouterV10


def load_feature_store(asset: str, start_date: str, end_date: str) -> pd.DataFrame:
    """
    Load MTF feature store for given asset and date range.

    Args:
        asset: Asset symbol (BTC, ETH, etc.)
        start_date: Start date YYYY-MM-DD
        end_date: End date YYYY-MM-DD

    Returns:
        DataFrame with all features
    """
    # Try to find matching feature store
    feature_dir = Path('data/features_mtf')

    # Parse dates to determine which file to use (ensure timezone-naive)
    start = pd.to_datetime(start_date)
    end = pd.to_datetime(end_date)

    # Make timezone-naive for comparison
    if hasattr(start, 'tzinfo') and start.tzinfo:
        start = start.replace(tzinfo=None)
    if hasattr(end, 'tzinfo') and end.tzinfo:
        end = end.replace(tzinfo=None)

    # Check for exact match first
    filename = f"{asset}_1H_{start_date}_to_{end_date}.parquet"
    filepath = feature_dir / filename

    if filepath.exists():
        print(f"✅ Loading feature store: {filepath}")
        df = pd.read_parquet(filepath)

        # Ensure timestamp column exists
        if 'timestamp' not in df.columns:
            if isinstance(df.index, pd.DatetimeIndex):
                df['timestamp'] = df.index
            else:
                raise ValueError(f"Feature store has no timestamp column or DatetimeIndex")

        return df

    # Check for overlapping files
    pattern = f"{asset}_1H_*.parquet"
    matching_files = list(feature_dir.glob(pattern))

    if not matching_files:
        raise FileNotFoundError(
            f"No feature stores found for {asset}.\n"
            f"Expected: {filepath}\n"
            f"Run: python3 bin/build_mtf_feature_store.py --asset {asset} "
            f"--start {start_date} --end {end_date}"
        )

    print(f"⚠️  Exact match not found, checking {len(matching_files)} available files...")

    # Load and filter by date range
    for file in sorted(matching_files):
        df = pd.read_parquet(file)

        # Ensure timestamp column
        if 'timestamp' not in df.columns:
            if isinstance(df.index, pd.DatetimeIndex):
                df['timestamp'] = df.index
            else:
                continue

        df_start = df['timestamp'].min()
        df_end = df['timestamp'].max()

        # Make both timezone-naive for comparison
        if hasattr(df_start, 'tzinfo') and df_start.tzinfo:
            df_start = df_start.replace(tzinfo=None)
        if hasattr(df_end, 'tzinfo') and df_end.tzinfo:
            df_end = df_end.replace(tzinfo=None)

        # Debug logging
        print(f"   Checking {file.name}:")
        print(f"      File range: {df_start} to {df_end}")
        print(f"      Requested:  {start} to {end}")

        # Check if file covers requested range (allow 1 day buffer for start date)
        # This handles cases where feature engineering warm-up consumes first few hours
        start_buffer = pd.Timedelta(days=1)
        start_ok = df_start <= start + start_buffer
        end_ok = df_end >= end

        print(f"      Covers? start_ok={start_ok} (df_start={df_start} <= {start + start_buffer}) and end_ok={end_ok}")

        if start_ok and end_ok:
            print(f"✅ Found covering file: {file.name}")

            # Ensure timestamp column is timezone-naive for filtering
            if hasattr(df['timestamp'].dtype, 'tz') and df['timestamp'].dtype.tz is not None:
                df['timestamp'] = df['timestamp'].dt.tz_localize(None)

            # Filter to requested range
            mask = (df['timestamp'] >= start) & (df['timestamp'] <= end)
            df_filtered = df[mask].copy()
            print(f"   Filtered to {len(df_filtered):,} bars ({start_date} to {end_date})")
            return df_filtered

    # If no single file covers range, raise error
    raise FileNotFoundError(
        f"No feature store covers full range {start_date} to {end_date}.\n"
        f"Available files: {[f.name for f in matching_files]}\n"
        f"Build new store: python3 bin/build_mtf_feature_store.py --asset {asset} "
        f"--start {start_date} --end {end_date}"
    )


def simulate_router_backtest(df: pd.DataFrame, router: RouterV10, regime_detector: RegimeDetector,
                               event_calendar: EventCalendar) -> dict:
    """
    Simulate backtest with router-based config switching.

    For now, this is a simplified simulation that tracks router decisions
    without running full trade execution. This validates the router logic
    before full backtest integration.

    Args:
        df: Feature store DataFrame
        router: Router v10 instance
        regime_detector: Regime classifier
        event_calendar: Event suppression calendar

    Returns:
        dict with router telemetry and statistics
    """
    print(f"\n{'='*80}")
    print("ROUTER BACKTEST SIMULATION")
    print('='*80)

    print(f"\nClassifying regimes for {len(df):,} bars...")
    df_classified = regime_detector.classify_batch(df)

    print(f"\n📊 Regime Distribution:")
    regime_counts = df_classified['regime_label'].value_counts()
    total = len(df_classified)
    for regime in ['crisis', 'risk_off', 'neutral', 'risk_on']:
        count = regime_counts.get(regime, 0)
        pct = 100 * count / total
        bar = '█' * int(pct / 2)
        print(f"  {regime:12s}: {count:5d} bars ({pct:5.1f}%) {bar}")

    # Confidence stats
    valid_conf = df_classified[df_classified['regime_confidence'] > 0]['regime_confidence']
    print(f"\n📈 Regime Confidence Stats:")
    print(f"  Mean:   {valid_conf.mean():.3f}")
    print(f"  Median: {valid_conf.median():.3f}")
    print(f"  <0.6:   {(valid_conf < 0.6).sum()} bars ({100 * (valid_conf < 0.6).sum() / len(valid_conf):.1f}%)")

    # Router decisions
    print(f"\n{'='*80}")
    print("ROUTER DECISION SIMULATION")
    print('='*80)

    for idx, row in df_classified.iterrows():
        timestamp = row['timestamp']
        regime_label = row['regime_label']
        regime_confidence = row['regime_confidence']

        # Check event calendar
        event_flag = event_calendar.is_suppression_window(timestamp)

        # Router decision
        decision = router.select_config(
            timestamp=timestamp,
            regime_label=regime_label,
            regime_confidence=regime_confidence,
            event_flag=event_flag
        )

    # Router statistics
    print(f"\n{'='*80}")
    print("ROUTER STATISTICS")
    print('='*80)

    stats = router.get_stats()

    print(f"\nTotal decisions: {stats['total_decisions']:,}")

    print(f"\n📊 Config Usage Distribution:")
    for action, pct in sorted(stats['action_distribution'].items()):
        bar = '█' * int(pct / 2)
        print(f"  {action:5s}: {pct:5.1f}% {bar}")

    print(f"\n📊 Decision Reasons:")
    reason_groups = {
        'Confidence Veto': [r for r in stats['reason_distribution'].keys() if 'low_confidence' in r],
        'Event Suppression': ['event_suppression'],
        'Crisis/Risk-Off': [r for r in stats['reason_distribution'].keys() if 'crisis' in r or 'risk_off' in r],
        'Risk-On/Neutral': [r for r in stats['reason_distribution'].keys() if 'risk_on' in r or 'neutral' in r]
    }

    for group_name, reasons in reason_groups.items():
        group_count = sum(stats['reason_distribution'].get(r, 0) for r in reasons)
        if group_count > 0:
            group_pct = 100 * group_count / stats['total_decisions']
            print(f"  {group_name:20s}: {group_count:5d} ({group_pct:4.1f}%)")

    print(f"\n📈 Regime Switches: {stats['regime_switches']}")
    switches_per_day = stats['regime_switches'] / (total / 24)
    print(f"   ({switches_per_day:.2f} switches/day)")

    print(f"\n📈 Confidence Stats (all bars):")
    for stat, val in stats['confidence_stats'].items():
        print(f"  {stat:8s}: {val:.3f}")

    return {
        'regime_stats': {
            'total_bars': total,
            'regime_distribution': {k: v for k, v in regime_counts.items()},
            'confidence_mean': valid_conf.mean(),
            'confidence_median': valid_conf.median(),
            'low_confidence_pct': 100 * (valid_conf < 0.6).sum() / len(valid_conf)
        },
        'router_stats': stats,
        'classified_df': df_classified
    }


def main():
    parser = argparse.ArgumentParser(description='Router v10 Backtest Validation')
    parser.add_argument('--asset', type=str, required=True, help='Asset symbol (BTC, ETH, etc.)')
    parser.add_argument('--start', type=str, required=True, help='Evaluation start date YYYY-MM-DD')
    parser.add_argument('--end', type=str, required=True, help='End date YYYY-MM-DD')
    parser.add_argument('--data-start', type=str, help='Data load start (for warm-up period, default=--start)')
    parser.add_argument('--output', type=str, help='Output directory for results')
    parser.add_argument('--export-decisions', type=str, help='Export router decisions to JSON')

    args = parser.parse_args()

    # Use data_start if provided, otherwise use start
    data_start = args.data_start if args.data_start else args.start
    eval_start = pd.to_datetime(args.start)

    print("\n" + "="*80)
    print("ROUTER V10 - BACKTEST VALIDATION")
    print("="*80)
    print(f"Asset: {args.asset}")
    print(f"Evaluation period: {args.start} to {args.end}")
    if args.data_start:
        print(f"Data load start: {data_start} (warm-up period)")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Load feature store
    print(f"\n{'='*80}")
    print("LOADING DATA")
    print('='*80)
    df = load_feature_store(args.asset, data_start, args.end)
    print(f"✅ Loaded {len(df):,} bars")
    print(f"   Columns: {len(df.columns)}")
    print(f"   Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")

    # Show warm-up period if applicable
    if args.data_start:
        warmup_bars = len(df[df['timestamp'] < eval_start])
        eval_bars = len(df[df['timestamp'] >= eval_start])
        print(f"\n📊 Data Split:")
        print(f"   Warm-up bars: {warmup_bars:,} ({data_start} to {args.start})")
        print(f"   Evaluation bars: {eval_bars:,} ({args.start} to {args.end})")

    # Initialize components
    print(f"\n{'='*80}")
    print("INITIALIZING ROUTER COMPONENTS")
    print('='*80)

    regime_detector = RegimeDetector()
    print(f"✅ RegimeDetector initialized (GMM v3.1)")
    print(f"   Features: {len(regime_detector.features)}")
    print(f"   Clusters: {len(regime_detector.label_map)}")

    event_calendar = EventCalendar()
    print(f"✅ EventCalendar initialized")
    print(f"   Total events: {len(event_calendar.events)}")
    print(f"   Suppression window: T-{event_calendar.pre_event_hours}h to T+{event_calendar.post_event_hours}h")

    router = RouterV10(
        bull_config_path='configs/v10_bases/btc_bull_v10_best.json',
        bear_config_path='configs/v10_bases/btc_bear_v10_best.json',
        confidence_threshold=0.60,
        event_suppression=True,
        hysteresis_bars=0
    )
    print(f"✅ RouterV10 initialized")
    print(f"   Bull config: {router.bull_config_path.name}")
    print(f"   Bear config: {router.bear_config_path.name}")
    print(f"   Confidence threshold: {router.confidence_threshold}")

    # Run simulation
    results = simulate_router_backtest(df, router, regime_detector, event_calendar)

    # Filter results to evaluation period if warm-up was used
    if args.data_start:
        print(f"\n{'='*80}")
        print("FILTERING TO EVALUATION PERIOD")
        print('='*80)

        # Filter classified DataFrame
        df_eval = results['classified_df'][results['classified_df']['timestamp'] >= eval_start].copy()

        # Filter router decision history
        router_eval_decisions = [
            d for d in router.decision_history
            if d['timestamp'] >= eval_start
        ]

        # Recalculate stats from filtered decisions
        router._decision_history_backup = router.decision_history
        router.decision_history = router_eval_decisions
        eval_router_stats = router.get_stats()

        # Update results
        results['classified_df'] = df_eval
        results['router_stats'] = eval_router_stats

        # Recalculate regime stats from filtered data
        regime_counts = df_eval['regime_label'].value_counts()
        total_eval = len(df_eval)
        valid_conf = df_eval[df_eval['regime_confidence'] > 0]['regime_confidence']

        results['regime_stats'] = {
            'total_bars': total_eval,
            'regime_distribution': {k: v for k, v in regime_counts.items()},
            'confidence_mean': valid_conf.mean() if len(valid_conf) > 0 else 0.0,
            'confidence_median': valid_conf.median() if len(valid_conf) > 0 else 0.0,
            'low_confidence_pct': 100 * (valid_conf < 0.6).sum() / len(valid_conf) if len(valid_conf) > 0 else 0.0
        }

        print(f"✅ Filtered to {total_eval:,} evaluation bars ({args.start} to {args.end})")

    # Export results
    if args.output:
        output_dir = Path(args.output)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Export statistics
        stats_file = output_dir / 'router_stats.json'
        with open(stats_file, 'w') as f:
            json.dump({
                'asset': args.asset,
                'start_date': args.start,
                'end_date': args.end,
                'data_start': data_start,
                'regime_stats': results['regime_stats'],
                'router_stats': {
                    'total_decisions': results['router_stats']['total_decisions'],
                    'action_distribution': results['router_stats']['action_distribution'],
                    'regime_switches': results['router_stats']['regime_switches'],
                    'confidence_stats': results['router_stats']['confidence_stats']
                }
            }, f, indent=2)
        print(f"\n💾 Exported statistics: {stats_file}")

        # Export classified DataFrame
        df_file = output_dir / 'classified_bars.parquet'
        results['classified_df'].to_parquet(df_file)
        print(f"💾 Exported classified bars: {df_file}")

    # Export router decisions
    if args.export_decisions:
        router.export_decision_log(args.export_decisions)

    print("\n" + "="*80)
    print("✅ ROUTER VALIDATION COMPLETE")
    print("="*80)
    print(f"Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    return 0


if __name__ == '__main__':
    exit(main())
