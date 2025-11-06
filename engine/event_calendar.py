#!/usr/bin/env python3
"""
Event Calendar v10 - Macro Event Suppression

Tracks major macro events (CPI, FOMC, NFP) and provides suppression windows
to avoid trading during high-volatility event windows.

Usage:
    from engine.event_calendar import EventCalendar

    calendar = EventCalendar()

    # Check if timestamp is in suppression window
    should_suppress = calendar.is_suppression_window(timestamp)

    # Get event details
    event_info = calendar.get_event_info(timestamp)
"""

import pandas as pd
from datetime import datetime, timedelta
from typing import Tuple, Optional, Dict, List
import warnings
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────────────────────────────────────
# Event Schedules (2022-2025)
# ─────────────────────────────────────────────────────────────────────────────

# FOMC meetings (2:00 PM ET decision day)
FOMC_DATES = [
    # 2022
    '2022-01-26', '2022-03-16', '2022-05-04', '2022-06-15', '2022-07-27',
    '2022-09-21', '2022-11-02', '2022-12-14',
    # 2023
    '2023-02-01', '2023-03-22', '2023-05-03', '2023-06-14', '2023-07-26',
    '2023-09-20', '2023-11-01', '2023-12-13',
    # 2024
    '2024-01-31', '2024-03-20', '2024-05-01', '2024-06-12', '2024-07-31',
    '2024-09-18', '2024-11-07', '2024-12-18',
    # 2025
    '2025-01-29', '2025-03-19', '2025-05-07', '2025-06-18', '2025-07-30'
]

# CPI releases (8:30 AM ET, typically ~13th of month)
CPI_DATES = [
    # 2022
    '2022-01-12', '2022-02-10', '2022-03-10', '2022-04-12', '2022-05-11', '2022-06-10',
    '2022-07-13', '2022-08-10', '2022-09-13', '2022-10-13', '2022-11-10', '2022-12-13',
    # 2023
    '2023-01-12', '2023-02-14', '2023-03-14', '2023-04-12', '2023-05-10', '2023-06-13',
    '2023-07-12', '2023-08-10', '2023-09-13', '2023-10-12', '2023-11-14', '2023-12-12',
    # 2024
    '2024-01-11', '2024-02-13', '2024-03-12', '2024-04-10', '2024-05-15', '2024-06-12',
    '2024-07-11', '2024-08-14', '2024-09-11', '2024-10-10', '2024-11-13', '2024-12-11',
    # 2025
    '2025-01-15', '2025-02-12', '2025-03-12', '2025-04-10', '2025-05-13', '2025-06-11'
]

# NFP (Non-Farm Payrolls) - First Friday of month, 8:30 AM ET
NFP_DATES = [
    # 2022
    '2022-01-07', '2022-02-04', '2022-03-04', '2022-04-01', '2022-05-06', '2022-06-03',
    '2022-07-08', '2022-08-05', '2022-09-02', '2022-10-07', '2022-11-04', '2022-12-02',
    # 2023
    '2023-01-06', '2023-02-03', '2023-03-10', '2023-04-07', '2023-05-05', '2023-06-02',
    '2023-07-07', '2023-08-04', '2023-09-01', '2023-10-06', '2023-11-03', '2023-12-08',
    # 2024
    '2024-01-05', '2024-02-02', '2024-03-08', '2024-04-05', '2024-05-03', '2024-06-07',
    '2024-07-05', '2024-08-02', '2024-09-06', '2024-10-04', '2024-11-01', '2024-12-06',
    # 2025
    '2025-01-10', '2025-02-07', '2025-03-07', '2025-04-04', '2025-05-02', '2025-06-06'
]


class EventCalendar:
    """
    Macro event calendar with suppression window logic.

    Suppression windows (default):
    - CPI: 8:30 AM ET ± 12h (T-12h to T+2h)
    - FOMC: 2:00 PM ET ± 12h (T-12h to T+2h)
    - NFP: 8:30 AM ET ± 12h (T-12h to T+2h)
    """

    def __init__(
        self,
        pre_event_hours: float = 12.0,
        post_event_hours: float = 2.0
    ):
        """
        Initialize event calendar.

        Args:
            pre_event_hours: Hours before event to start suppression (default 12h)
            post_event_hours: Hours after event to end suppression (default 2h)
        """
        self.pre_event_hours = pre_event_hours
        self.post_event_hours = post_event_hours

        # Convert event dates to datetime (midnight UTC)
        # Note: Actual event times are ET (8:30 AM or 2:00 PM), but we use
        # conservative 24h windows that capture ±12h from event time
        self.fomc_dates = pd.to_datetime(FOMC_DATES)
        self.cpi_dates = pd.to_datetime(CPI_DATES)
        self.nfp_dates = pd.to_datetime(NFP_DATES)

        # Build unified event list for efficient lookup
        self._build_event_index()

    def _build_event_index(self):
        """Build unified index of all events for fast lookup."""
        events = []

        for date in self.fomc_dates:
            events.append({
                'date': date,
                'type': 'FOMC',
                'description': 'Federal Reserve Interest Rate Decision'
            })

        for date in self.cpi_dates:
            events.append({
                'date': date,
                'type': 'CPI',
                'description': 'Consumer Price Index Release'
            })

        for date in self.nfp_dates:
            events.append({
                'date': date,
                'type': 'NFP',
                'description': 'Non-Farm Payrolls Report'
            })

        # Sort by date
        self.events = sorted(events, key=lambda x: x['date'])

    def is_suppression_window(self, timestamp: pd.Timestamp) -> bool:
        """
        Check if timestamp falls within event suppression window.

        Args:
            timestamp: Timestamp to check

        Returns:
            True if within suppression window, False otherwise
        """
        # Convert to naive datetime for comparison
        ts_naive = timestamp.replace(tzinfo=None) if hasattr(timestamp, 'tzinfo') and timestamp.tzinfo else timestamp

        # Check all events
        for event in self.events:
            event_date = event['date']

            # Define suppression window
            window_start = event_date - timedelta(hours=self.pre_event_hours)
            window_end = event_date + timedelta(hours=24 + self.post_event_hours)

            if window_start <= ts_naive <= window_end:
                return True

        return False

    def get_event_info(self, timestamp: pd.Timestamp) -> Optional[Dict]:
        """
        Get details about event if timestamp is in suppression window.

        Args:
            timestamp: Timestamp to check

        Returns:
            Event info dict if in window, None otherwise:
            {
                'in_window': bool,
                'event_type': str,
                'event_date': pd.Timestamp,
                'description': str,
                'hours_to_event': float,
                'window_start': pd.Timestamp,
                'window_end': pd.Timestamp
            }
        """
        # Convert to naive datetime
        ts_naive = timestamp.replace(tzinfo=None) if hasattr(timestamp, 'tzinfo') and timestamp.tzinfo else timestamp

        # Find closest event in suppression window
        for event in self.events:
            event_date = event['date']
            window_start = event_date - timedelta(hours=self.pre_event_hours)
            window_end = event_date + timedelta(hours=24 + self.post_event_hours)

            if window_start <= ts_naive <= window_end:
                hours_to_event = (event_date - ts_naive).total_seconds() / 3600

                return {
                    'in_window': True,
                    'event_type': event['type'],
                    'event_date': event_date,
                    'description': event['description'],
                    'hours_to_event': hours_to_event,
                    'window_start': window_start,
                    'window_end': window_end
                }

        return {
            'in_window': False,
            'event_type': None,
            'event_date': None,
            'description': None,
            'hours_to_event': None,
            'window_start': None,
            'window_end': None
        }

    def get_next_event(self, timestamp: pd.Timestamp) -> Optional[Dict]:
        """
        Get next upcoming event after timestamp.

        Args:
            timestamp: Reference timestamp

        Returns:
            Event info dict or None if no upcoming events
        """
        ts_naive = timestamp.replace(tzinfo=None) if hasattr(timestamp, 'tzinfo') and timestamp.tzinfo else timestamp

        for event in self.events:
            if event['date'] > ts_naive:
                hours_until = (event['date'] - ts_naive).total_seconds() / 3600

                return {
                    'event_type': event['type'],
                    'event_date': event['date'],
                    'description': event['description'],
                    'hours_until': hours_until
                }

        return None

    def count_suppressed_bars(
        self,
        start_date: str,
        end_date: str,
        bar_frequency_hours: float = 1.0
    ) -> Dict:
        """
        Count how many bars would be suppressed in date range.

        Args:
            start_date: Start date string 'YYYY-MM-DD'
            end_date: End date string 'YYYY-MM-DD'
            bar_frequency_hours: Hours between bars (default 1.0 for 1H)

        Returns:
            {
                'total_bars': int,
                'suppressed_bars': int,
                'suppressed_pct': float,
                'events_in_range': int,
                'by_event_type': {'FOMC': int, 'CPI': int, 'NFP': int}
            }
        """
        start = pd.to_datetime(start_date)
        end = pd.to_datetime(end_date)

        # Generate bar timestamps
        date_range = pd.date_range(start=start, end=end, freq=f'{bar_frequency_hours}H')

        # Count suppressions
        suppressed_count = 0
        suppressed_by_type = {'FOMC': 0, 'CPI': 0, 'NFP': 0}
        events_in_range = 0

        for ts in date_range:
            event_info = self.get_event_info(ts)
            if event_info['in_window']:
                suppressed_count += 1
                event_type = event_info['event_type']
                if event_type in suppressed_by_type:
                    suppressed_by_type[event_type] += 1

        # Count events in range
        for event in self.events:
            if start <= event['date'] <= end:
                events_in_range += 1

        total_bars = len(date_range)
        suppressed_pct = 100 * suppressed_count / total_bars if total_bars > 0 else 0

        return {
            'total_bars': total_bars,
            'suppressed_bars': suppressed_count,
            'suppressed_pct': suppressed_pct,
            'events_in_range': events_in_range,
            'by_event_type': suppressed_by_type
        }


if __name__ == '__main__':
    """Quick test of event calendar."""
    print("\n" + "="*80)
    print("EVENT CALENDAR V10 - QUICK TEST")
    print("="*80)

    # Initialize calendar
    calendar = EventCalendar()
    print(f"\n✅ Initialized EventCalendar")
    print(f"   Suppression window: T-{calendar.pre_event_hours}h to T+{calendar.post_event_hours}h")
    print(f"   Total events: {len(calendar.events)}")
    print(f"   - FOMC: {len(calendar.fomc_dates)}")
    print(f"   - CPI: {len(calendar.cpi_dates)}")
    print(f"   - NFP: {len(calendar.nfp_dates)}")

    # Test suppression windows on 2024
    print(f"\n{'='*80}")
    print("2024 SUPPRESSION ANALYSIS")
    print('='*80)

    stats = calendar.count_suppressed_bars('2024-01-01', '2024-12-31', bar_frequency_hours=1.0)

    print(f"\nTotal bars (1H): {stats['total_bars']:,}")
    print(f"Suppressed bars: {stats['suppressed_bars']:,} ({stats['suppressed_pct']:.1f}%)")
    print(f"Events in 2024: {stats['events_in_range']}")

    print(f"\nSuppressed bars by event type:")
    for event_type, count in stats['by_event_type'].items():
        pct = 100 * count / stats['total_bars']
        print(f"  {event_type:6s}: {count:5d} bars ({pct:4.1f}%)")

    # Test specific event windows
    print(f"\n{'='*80}")
    print("EXAMPLE EVENT WINDOWS")
    print('='*80)

    test_dates = [
        ('2024-01-11', 'CPI Release'),  # CPI day
        ('2024-03-20', 'FOMC Decision'),  # FOMC day
        ('2024-05-03', 'NFP Report'),  # NFP day
    ]

    for date_str, description in test_dates:
        ts = pd.to_datetime(date_str)
        event_info = calendar.get_event_info(ts)

        print(f"\n{description} ({date_str}):")
        print(f"  In suppression window: {event_info['in_window']}")
        if event_info['in_window']:
            print(f"  Event type: {event_info['event_type']}")
            print(f"  Hours to event: {event_info['hours_to_event']:.1f}h")
            print(f"  Window: {event_info['window_start']} to {event_info['window_end']}")

    print("\n" + "="*80)
    print("✅ EVENT CALENDAR READY")
    print("="*80)
