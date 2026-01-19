#!/usr/bin/env python3
"""
Validate Wyckoff event detection on historical BTC feature store data.
Expects to detect key market events:
- SC at June 2022 $17.6k lows (capitulation)
- Springs and LPS during 2024 bull market pullbacks
"""
import sys
import json
import pandas as pd
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from engine.wyckoff.wyckoff_engine import WyckoffEngine


def main():
    print("=" * 100)
    print("WYCKOFF HISTORICAL VALIDATION - FEATURE STORE DATA")
    print("=" * 100)

    # Load config
    config_path = project_root / 'configs' / 'wyckoff_events_config.json'
    with open(config_path) as f:
        config = json.load(f)

    print(f"\n✓ Loaded config: {config_path.name}")
    print(f"  Events enabled: {config['wyckoff_events']['enabled']}")
    print(f"  PTI integration: {config['wyckoff_events']['pti_integration']}")

    # Initialize engine
    engine = WyckoffEngine(config=config)

    # Test periods
    test_periods = [
        {
            'name': '2022 Bear Market',
            'file': 'data/features_mtf/BTC_1H_2022-01-01_to_2023-12-31.parquet',
            'start': '2022-01-01',
            'end': '2022-12-31',
            'expected_events': ['SC', 'ST', 'SOS'],
            'key_dates': 'June 2022 SC at $17.6k lows',
        },
        {
            'name': '2024 Bull Market',
            'file': 'data/features_mtf/BTC_1H_2024-01-01_to_2024-12-31.parquet',
            'start': '2024-01-01',
            'end': '2024-09-30',
            'expected_events': ['LPS', 'Spring-A', 'Spring-B', 'SOS'],
            'key_dates': 'Multiple springs at pullbacks',
        },
    ]

    all_results = {}

    for period in test_periods:
        print("\n" + "=" * 100)
        print(f"PERIOD: {period['name']}")
        print(f"File: {period['file']}")
        print(f"Expected: {period['key_dates']}")
        print("=" * 100)

        # Load data
        data_path = project_root / period['file']
        if not data_path.exists():
            print(f"✗ File not found: {data_path}")
            continue

        print(f"\n[1/3] Loading feature store data...")
        try:
            df = pd.read_parquet(data_path)

            # Handle timestamp - could be index or column
            if df.index.name == 'timestamp' or isinstance(df.index, pd.DatetimeIndex):
                # Timestamp is the index - reset to column
                df = df.reset_index()
            elif 'timestamp' not in df.columns:
                raise ValueError("No timestamp column or index found in data")

            # Ensure timestamp is datetime
            df['timestamp'] = pd.to_datetime(df['timestamp'])

            # Filter to period if needed
            mask = (df['timestamp'] >= period['start']) & (df['timestamp'] <= period['end'])
            df = df[mask].copy()

            print(f"✓ Loaded {len(df)} bars from {df['timestamp'].iloc[0]} to {df['timestamp'].iloc[-1]}")

            # Check required columns
            required = ['open', 'high', 'low', 'close', 'volume']
            missing = [col for col in required if col not in df.columns]
            if missing:
                print(f"✗ Missing required columns: {missing}")
                continue

        except Exception as e:
            print(f"✗ ERROR loading data: {e}")
            continue

        # Detect events
        print(f"\n[2/3] Running Wyckoff event detection...")
        try:
            df_events = engine.detect_wyckoff_events(df)
            print(f"✓ Detection complete, {len(df_events.columns)} total columns")
        except Exception as e:
            print(f"✗ ERROR during detection: {e}")
            import traceback
            traceback.print_exc()
            continue

        # Analyze results
        print(f"\n[3/3] Analyzing detected events...")

        event_cols = {
            'SC': 'wyckoff_sc',
            'BC': 'wyckoff_bc',
            'AR': 'wyckoff_ar',
            'AS': 'wyckoff_as',
            'ST': 'wyckoff_st',
            'SOS': 'wyckoff_sos',
            'SOW': 'wyckoff_sow',
            'Spring-A': 'wyckoff_spring_a',
            'Spring-B': 'wyckoff_spring_b',
            'UT': 'wyckoff_ut',
            'UTAD': 'wyckoff_utad',
            'LPS': 'wyckoff_lps',
            'LPSY': 'wyckoff_lpsy',
        }

        events_detected = {}
        for event_name, col in event_cols.items():
            if col in df_events.columns:
                count = int(df_events[col].sum())
                if count > 0:
                    # Get event dates and prices
                    event_rows = df_events[df_events[col]].copy()
                    dates = event_rows['timestamp'].dt.strftime('%Y-%m-%d %H:%M').tolist()[:10]

                    # Get prices at events
                    prices = event_rows['close'].tolist()[:10]

                    events_detected[event_name] = {
                        'count': count,
                        'dates': dates,
                        'prices': [f"${p:,.0f}" for p in prices],
                        'confidence': {}
                    }

                    # Get confidence scores
                    conf_col = f"{col}_confidence"
                    if conf_col in df_events.columns:
                        conf_scores = df_events[df_events[col]][conf_col]
                        if len(conf_scores) > 0:
                            events_detected[event_name]['confidence'] = {
                                'min': float(conf_scores.min()),
                                'max': float(conf_scores.max()),
                                'mean': float(conf_scores.mean()),
                            }

        # Print results
        total_events = sum(e['count'] for e in events_detected.values())
        print(f"\n✓ Total events detected: {total_events}")

        if total_events > 0:
            # Sort by count
            for event_name, data in sorted(events_detected.items(), key=lambda x: x[1]['count'], reverse=True):
                print(f"\n  {event_name}: {data['count']} occurrences")

                # Show first 5 detections with dates and prices
                for i, (date, price) in enumerate(zip(data['dates'][:5], data['prices'][:5])):
                    print(f"    [{i+1}] {date} @ {price}")

                if data['confidence']:
                    c = data['confidence']
                    print(f"    Confidence: min={c['min']:.2f}, max={c['max']:.2f}, mean={c['mean']:.2f}")

            # Check for expected events
            print(f"\n  Expected Events Check:")
            for expected in period['expected_events']:
                if expected in events_detected:
                    print(f"    ✓ {expected} detected ({events_detected[expected]['count']} times)")
                else:
                    print(f"    ✗ {expected} NOT detected")

            # Special checks for known market events
            if period['name'] == '2022 Bear Market' and 'SC' in events_detected:
                print(f"\n  Key Market Event Check:")
                print(f"    June 2022 SC ($17.6k): Looking for SC events in May-July 2022...")
                sc_dates = events_detected['SC']['dates']
                june_sc = [d for d in sc_dates if '2022-06' in d or '2022-05' in d or '2022-07' in d]
                if june_sc:
                    print(f"    ✓ Found {len(june_sc)} SC events near June 2022 bottom")
                    for date in june_sc[:3]:
                        print(f"      - {date}")
                else:
                    print(f"    ⚠ No SC detected near June 2022 (may need threshold tuning)")

        else:
            print("  ⚠ No events detected in this period")
            print("\n  Possible reasons:")
            print("    - Thresholds too strict (try lowering sc_volume_z_min from 2.5 to 2.0)")
            print("    - Check if volume_z column exists in feature store")
            print("    - Run with debug logging enabled")

        all_results[period['name']] = events_detected

    # Final summary
    print("\n" + "=" * 100)
    print("VALIDATION SUMMARY")
    print("=" * 100)

    for period_name, events in all_results.items():
        total = sum(e['count'] for e in events.values())
        print(f"\n{period_name}: {total} total events")
        if events:
            top_events = sorted(events.items(), key=lambda x: x[1]['count'], reverse=True)[:5]
            for event_name, data in top_events:
                print(f"  • {event_name}: {data['count']}")

    # Recommendations
    print("\n" + "=" * 100)
    print("RECOMMENDATIONS")
    print("=" * 100)

    total_all = sum(sum(e['count'] for e in events.values()) for events in all_results.values())

    if total_all > 0:
        print("\n✓ Wyckoff event detection is working correctly!")
        print("\nNext steps:")
        print("1. Review detected events against known market structures")
        print("2. If too few events, lower thresholds in configs/wyckoff_events_config.json:")
        print("   - sc_volume_z_min: 2.5 → 2.0 (more SC detections)")
        print("   - spring_a_breakdown_margin: 0.02 → 0.015 (more Spring-A)")
        print("3. If too many false positives, raise thresholds")
        print("4. Enable in main trading config for backtesting")
    else:
        print("\n⚠ No events detected - diagnostic needed")
        print("\nTroubleshooting:")
        print("1. Check if feature store has volume_z column:")
        print("   python3 -c \"import pandas as pd; df=pd.read_parquet('data/features_mtf/BTC_1H_2022-01-01_to_2023-12-31.parquet'); print(df.columns.tolist())\"")
        print("2. Check volume statistics:")
        print("   python3 -c \"import pandas as pd; df=pd.read_parquet('data/features_mtf/BTC_1H_2022-01-01_to_2023-12-31.parquet'); print(df['volume'].describe())\"")
        print("3. Try lowering all thresholds by 20-30%")

    print("=" * 100)

    return 0


if __name__ == '__main__':
    sys.exit(main())
