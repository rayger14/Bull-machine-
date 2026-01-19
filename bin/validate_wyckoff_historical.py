#!/usr/bin/env python3
"""
Validate Wyckoff event detection on historical BTC data (2022-2024).
Expected to detect key market events:
- SC at June 2022 $17.6k lows (capitulation)
- BC at November 2021 $69k ATH (euphoria)
- Springs and LPS during 2024 bull market pullbacks
"""
import sys
import json
import pandas as pd
from pathlib import Path
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from engine.wyckoff.wyckoff_engine import WyckoffEngine
from engine.data.binance_fetcher import BinanceFetcher


def main():
    print("=" * 100)
    print("WYCKOFF HISTORICAL VALIDATION - 2022-2024 BTC")
    print("=" * 100)

    # Load config
    config_path = project_root / 'configs' / 'wyckoff_events_config.json'
    with open(config_path) as f:
        config = json.load(f)

    print(f"\n✓ Loaded config: {config_path}")
    print(f"  Events enabled: {config['wyckoff_events']['enabled']}")
    print(f"  PTI integration: {config['wyckoff_events']['pti_integration']}")

    # Initialize components
    fetcher = BinanceFetcher()
    engine = WyckoffEngine(config=config)

    # Test periods
    test_periods = [
        {
            'name': '2022 Bear Market (SC expected)',
            'start': '2022-01-01',
            'end': '2022-12-31',
            'expected_events': ['SC', 'ST', 'SOS'],
            'key_event': 'SC at June 2022 $17.6k lows',
        },
        {
            'name': '2024 Bull Market (LPS/Spring expected)',
            'start': '2024-01-01',
            'end': '2024-09-30',
            'expected_events': ['LPS', 'Spring-A', 'Spring-B', 'SOS'],
            'key_event': 'Multiple springs at pullbacks',
        },
    ]

    all_results = {}

    for period in test_periods:
        print("\n" + "=" * 100)
        print(f"PERIOD: {period['name']}")
        print(f"Range: {period['start']} to {period['end']}")
        print(f"Expected: {period['key_event']}")
        print("=" * 100)

        # Fetch data
        print(f"\n[1/3] Fetching BTC 1H data from Binance...")
        try:
            df = fetcher.fetch_ohlcv(
                symbol='BTC/USDT',
                timeframe='1h',
                since=period['start'],
                until=period['end']
            )
            print(f"✓ Fetched {len(df)} bars")
        except Exception as e:
            print(f"✗ ERROR fetching data: {e}")
            print(f"  Skipping period {period['name']}")
            continue

        # Detect events
        print(f"\n[2/3] Running Wyckoff event detection...")
        try:
            df_events = engine.detect_wyckoff_events(df)
            print(f"✓ Detection complete, {len(df_events.columns)} columns")
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
                count = df_events[col].sum()
                if count > 0:
                    events_detected[event_name] = {
                        'count': int(count),
                        'dates': df_events[df_events[col]]['timestamp'].dt.strftime('%Y-%m-%d %H:%M').tolist()[:10],  # First 10
                        'confidence': []
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
        print(f"\nTotal events detected: {total_events}")

        if total_events > 0:
            for event_name, data in sorted(events_detected.items(), key=lambda x: x[1]['count'], reverse=True):
                print(f"\n  • {event_name}: {data['count']} occurrences")
                print(f"    First detections: {', '.join(data['dates'][:5])}")
                if data['confidence']:
                    print(f"    Confidence: min={data['confidence']['min']:.2f}, max={data['confidence']['max']:.2f}, mean={data['confidence']['mean']:.2f}")

            # Check for expected events
            print(f"\n  Expected Events Check:")
            for expected in period['expected_events']:
                if expected in events_detected:
                    print(f"    ✓ {expected} detected ({events_detected[expected]['count']} times)")
                else:
                    print(f"    ✗ {expected} NOT detected")

        else:
            print("  ⚠ No events detected in this period")
            print("  Possible reasons:")
            print("    - Thresholds too strict for this market regime")
            print("    - Events require specific confluence conditions")
            print("    - Check configs/wyckoff_events_config.json for threshold settings")

        all_results[period['name']] = events_detected

    # Final summary
    print("\n" + "=" * 100)
    print("VALIDATION SUMMARY")
    print("=" * 100)

    for period_name, events in all_results.items():
        total = sum(e['count'] for e in events.values())
        print(f"\n{period_name}: {total} total events")
        if events:
            top_events = sorted(events.items(), key=lambda x: x[1]['count'], reverse=True)[:3]
            for event_name, data in top_events:
                print(f"  • {event_name}: {data['count']}")

    print("\n" + "=" * 100)
    print("✓ VALIDATION COMPLETE")
    print("\nNext steps:")
    print("1. Review detected events against known market structures")
    print("2. Tune thresholds if needed in configs/wyckoff_events_config.json")
    print("3. Enable Wyckoff events in main trading config for backtesting")
    print("=" * 100)

    return 0


if __name__ == '__main__':
    sys.exit(main())
