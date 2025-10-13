"""
Event-Driven Backtesting - Bull Machine v1.8.5

Tags major crypto events (conferences, halving, etc.) and detects
pump/dump patterns around these catalysts. Flags high-leverage periods
via funding/OI metrics.

Trader Alignment:
- @Wyckoff_Insider: Event-driven accumulation/distribution (post:30)
- @Moneytaur: Institutional positioning around catalysts (post:33)
"""

from datetime import datetime, timedelta
from typing import Dict, List, Optional
import pandas as pd


# Major crypto events calendar (2024-2025)
EVENT_CALENDAR = [
    {
        'name': 'Bitcoin Halving',
        'date': datetime(2024, 4, 20),
        'type': 'halving',
        'impact': 'pump_then_chop',
        'window_days': 30
    },
    {
        'name': 'Bitcoin Miami 2025',
        'date': datetime(2025, 1, 24),
        'type': 'conference',
        'impact': 'pump_dump',
        'window_days': 14
    },
    {
        'name': 'ETHDenver 2025',
        'date': datetime(2025, 2, 27),
        'type': 'conference',
        'impact': 'pump_dump',
        'window_days': 14
    },
    {
        'name': 'Consensus 2025',
        'date': datetime(2025, 5, 14),
        'type': 'conference',
        'impact': 'pump_dump',
        'window_days': 14
    },
    {
        'name': 'TOKEN2049 Singapore',
        'date': datetime(2025, 9, 17),
        'type': 'conference',
        'impact': 'pump_dump',
        'window_days': 14
    },
    {
        'name': 'TOKEN2049 Dubai',
        'date': datetime(2025, 10, 1),
        'type': 'conference',
        'impact': 'pump_dump',
        'window_days': 14
    }
]


def get_active_events(timestamp: datetime, config: Dict) -> List[Dict]:
    """
    Get events active at given timestamp.

    Args:
        timestamp: Current timestamp
        config: Config dict with 'calendar_enabled'

    Returns:
        List of active events with metadata

    Example:
        >>> ts = datetime(2025, 5, 10)
        >>> events = get_active_events(ts, {'calendar_enabled': True})
        >>> # Returns Consensus 2025 (4 days before event)
    """
    if not config.get('calendar_enabled', False):
        return []

    active_events = []

    for event in EVENT_CALENDAR:
        # Define event window (before and after)
        window_days = event.get('window_days', 14)
        window_start = event['date'] - timedelta(days=window_days)
        window_end = event['date'] + timedelta(days=7)  # 7 days after

        # Normalize timestamp to naive datetime for comparison
        ts_naive = timestamp.replace(tzinfo=None) if hasattr(timestamp, 'tzinfo') and timestamp.tzinfo else timestamp

        if window_start <= ts_naive <= window_end:
            days_to_event = (event['date'] - ts_naive).days
            phase = 'pre' if days_to_event > 0 else 'post'

            active_events.append({
                'name': event['name'],
                'type': event['type'],
                'impact': event['impact'],
                'phase': phase,
                'days_to_event': abs(days_to_event)
            })

    return active_events


def tag_events(df: pd.DataFrame, macro_data: Dict, config: Dict) -> Dict:
    """
    Tag price action with event metadata and leverage signals.

    Args:
        df: OHLCV DataFrame with timestamps
        macro_data: Macro data dict (FUNDING, OI, etc.)
        config: Config dict

    Returns:
        Dict with event_signals and funding_oi_tags

    Example:
        >>> df = pd.DataFrame({'close': [100, 102], 'timestamp': [...]})
        >>> macro = {'FUNDING': pd.DataFrame({'value': [0.01], 'timestamp': [...]})}
        >>> tags = tag_events(df, macro, {'calendar_enabled': True})
    """
    if df.empty:
        return {'event_signals': [], 'funding_oi_tags': {}}

    # Get current timestamp
    current_time = df.index[-1] if isinstance(df.index, pd.DatetimeIndex) else df['timestamp'].iloc[-1]
    if not isinstance(current_time, datetime):
        current_time = pd.to_datetime(current_time)

    # Check for active events
    active_events = get_active_events(current_time, config)

    # Analyze funding and OI if available
    funding_oi_tags = {}

    if 'FUNDING' in macro_data:
        funding_df = macro_data['FUNDING']
        if not funding_df.empty:
            recent_funding = funding_df['value'].iloc[-1] if 'value' in funding_df else 0.0
            funding_max = config.get('funding_max', 0.01)

            if abs(recent_funding) > funding_max:
                funding_oi_tags['high_leverage'] = True
                funding_oi_tags['funding_rate'] = float(recent_funding)

    if 'OI' in macro_data:
        oi_df = macro_data['OI']
        if not oi_df.empty and len(oi_df) > 1:
            oi_change = (oi_df['value'].iloc[-1] - oi_df['value'].iloc[-2]) / oi_df['value'].iloc[-2]
            oi_spot_max = config.get('oi_spot_max', 0.015)

            if abs(oi_change) > oi_spot_max:
                funding_oi_tags['high_oi_change'] = True
                funding_oi_tags['oi_change_pct'] = float(oi_change)

    # Build event signals
    event_signals = []
    for event in active_events:
        event_signals.append({
            'event': event['name'],
            'type': event['type'],
            'impact': event['impact'],
            'phase': event['phase'],
            'days_to_event': event['days_to_event']
        })

    return {
        'event_signals': event_signals,
        'funding_oi_tags': funding_oi_tags
    }


def should_veto_event(signal: Dict, event_tags: Dict, config: Dict) -> tuple:
    """
    Determine if trade should be vetoed based on event context.

    Args:
        signal: Trade signal dict
        event_tags: Event tags from tag_events()
        config: Config dict

    Returns:
        (should_veto, reason)

    Example:
        >>> tags = {'funding_oi_tags': {'high_leverage': True}}
        >>> veto, reason = should_veto_event({}, tags, {})
        >>> # Returns (True, "High leverage detected")
    """
    # Veto on extreme leverage
    if event_tags['funding_oi_tags'].get('high_leverage', False):
        return True, "High leverage detected - funding rate extreme"

    if event_tags['funding_oi_tags'].get('high_oi_change', False):
        return True, "High OI change - potential manipulation"

    # Check event phase risk
    for event_signal in event_tags['event_signals']:
        if event_signal['impact'] == 'pump_dump':
            # Be cautious in post-event dump phase
            if event_signal['phase'] == 'post' and event_signal['days_to_event'] < 3:
                return True, f"Post-event dump risk: {event_signal['event']}"

    return False, ""


if __name__ == '__main__':
    # Quick validation
    print("Testing event engine...")

    config = {
        'calendar_enabled': True,
        'funding_max': 0.01,
        'oi_spot_max': 0.015
    }

    # Test 1: Get active events
    test_time = datetime(2025, 5, 10)  # 4 days before Consensus
    events = get_active_events(test_time, config)
    print(f"Active events on {test_time.date()}: {len(events)}")
    if events:
        print(f"  {events[0]['name']}: {events[0]['phase']} phase, {events[0]['days_to_event']} days")

    # Test 2: Tag with high leverage
    df = pd.DataFrame({
        'close': [100, 102, 104],
        'timestamp': pd.date_range('2025-05-10', periods=3, freq='1H')
    })
    df.set_index('timestamp', inplace=True)

    macro_data = {
        'FUNDING': pd.DataFrame({
            'value': [0.015],  # High funding
            'timestamp': [pd.Timestamp('2025-05-10')]
        })
    }

    tags = tag_events(df, macro_data, config)
    print(f"\nEvent tags: {len(tags['event_signals'])} events")
    print(f"Funding/OI tags: {tags['funding_oi_tags']}")

    # Test 3: Veto check
    veto, reason = should_veto_event({}, tags, config)
    print(f"\nShould veto? {veto}")
    if veto:
        print(f"  Reason: {reason}")

    assert len(events) > 0, "Should find Consensus event"
    assert tags['funding_oi_tags'].get('high_leverage', False), "Should flag high leverage"
    assert veto, "Should veto on high leverage"

    print("\n✅ Event engine validated")
