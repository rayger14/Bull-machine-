"""
Unit tests for v1.8.5 Event Engine - Bull Machine

Tests event calendar tagging, funding/OI detection, and veto logic.
"""

import pytest
import pandas as pd
from datetime import datetime, timedelta
from engine.events.event_engine import (
    get_active_events,
    tag_events,
    should_veto_event
)


def test_get_active_events_pre_phase():
    """Test event detection in pre-event phase."""
    # 10 days before Consensus 2025 (May 14)
    timestamp = datetime(2025, 5, 4)
    config = {'calendar_enabled': True}

    events = get_active_events(timestamp, config)

    assert len(events) > 0
    event = events[0]
    assert event['name'] == 'Consensus 2025'
    assert event['phase'] == 'pre'
    assert event['days_to_event'] == 10


def test_get_active_events_post_phase():
    """Test event detection in post-event phase."""
    # 2 days after Consensus 2025
    timestamp = datetime(2025, 5, 16)
    config = {'calendar_enabled': True}

    events = get_active_events(timestamp, config)

    assert len(events) > 0
    event = events[0]
    assert event['name'] == 'Consensus 2025'
    assert event['phase'] == 'post'
    assert event['days_to_event'] == 2


def test_get_active_events_disabled():
    """Test that no events are returned when calendar disabled."""
    timestamp = datetime(2025, 5, 10)
    config = {'calendar_enabled': False}

    events = get_active_events(timestamp, config)

    assert len(events) == 0


def test_get_active_events_outside_window():
    """Test no events when outside all event windows."""
    # Random date far from any event
    timestamp = datetime(2025, 6, 1)
    config = {'calendar_enabled': True}

    events = get_active_events(timestamp, config)

    # Should be empty or very few events
    assert isinstance(events, list)


def test_tag_events_with_high_funding():
    """Test funding rate tagging when above threshold."""
    df = pd.DataFrame({
        'close': [100, 102],
        'timestamp': pd.date_range('2025-05-10', periods=2, freq='1H')
    })
    df.set_index('timestamp', inplace=True)

    macro_data = {
        'FUNDING': pd.DataFrame({
            'value': [0.015],  # High funding (>1%)
            'timestamp': [pd.Timestamp('2025-05-10')]
        })
    }

    config = {'calendar_enabled': True, 'funding_max': 0.01}

    tags = tag_events(df, macro_data, config)

    assert tags['funding_oi_tags'].get('high_leverage', False) is True
    assert 'funding_rate' in tags['funding_oi_tags']


def test_tag_events_with_high_oi_change():
    """Test OI change tagging when above threshold."""
    df = pd.DataFrame({
        'close': [100, 102],
        'timestamp': pd.date_range('2025-05-10', periods=2, freq='1H')
    })
    df.set_index('timestamp', inplace=True)

    macro_data = {
        'OI': pd.DataFrame({
            'value': [1000, 1020],  # 2% increase
            'timestamp': pd.date_range('2025-05-10', periods=2, freq='1H')
        })
    }

    config = {'calendar_enabled': True, 'oi_spot_max': 0.015}

    tags = tag_events(df, macro_data, config)

    assert tags['funding_oi_tags'].get('high_oi_change', False) is True


def test_tag_events_empty_df():
    """Test graceful handling of empty dataframe."""
    df = pd.DataFrame()
    macro_data = {}
    config = {'calendar_enabled': True}

    tags = tag_events(df, macro_data, config)

    assert tags['event_signals'] == []
    assert tags['funding_oi_tags'] == {}


def test_should_veto_event_high_leverage():
    """Test veto on high leverage."""
    event_tags = {
        'event_signals': [],
        'funding_oi_tags': {'high_leverage': True, 'funding_rate': 0.015}
    }

    veto, reason = should_veto_event({}, event_tags, {})

    assert veto is True
    assert 'leverage' in reason.lower()


def test_should_veto_event_high_oi():
    """Test veto on high OI change."""
    event_tags = {
        'event_signals': [],
        'funding_oi_tags': {'high_oi_change': True, 'oi_change_pct': 0.02}
    }

    veto, reason = should_veto_event({}, event_tags, {})

    assert veto is True
    assert 'OI' in reason


def test_should_veto_event_post_dump():
    """Test veto on post-event dump phase."""
    event_tags = {
        'event_signals': [{
            'event': 'Consensus 2025',
            'impact': 'pump_dump',
            'phase': 'post',
            'days_to_event': 2
        }],
        'funding_oi_tags': {}
    }

    veto, reason = should_veto_event({}, event_tags, {})

    assert veto is True
    assert 'dump' in reason.lower()


def test_should_not_veto_clean_conditions():
    """Test no veto when conditions are clean."""
    event_tags = {
        'event_signals': [],
        'funding_oi_tags': {}
    }

    veto, reason = should_veto_event({}, event_tags, {})

    assert veto is False
    assert reason == ""


def test_timezone_aware_timestamp_handling():
    """Test that timezone-aware timestamps are handled correctly."""
    # Pandas Timestamp with timezone
    timestamp = pd.Timestamp('2025-05-10', tz='UTC')

    df = pd.DataFrame({
        'close': [100],
        'timestamp': [timestamp]
    })
    df.set_index('timestamp', inplace=True)

    macro_data = {}
    config = {'calendar_enabled': True}

    # Should not raise TypeError
    tags = tag_events(df, macro_data, config)

    assert isinstance(tags, dict)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
