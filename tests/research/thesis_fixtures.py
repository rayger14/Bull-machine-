"""Literal prices and clocks shared by the new thesis lab tests."""
import pandas as pd


def candle_event(start, timeframe='1h', *, low=101., high=109., close=104.,
                 opened=103., status='known'):
    from scripts.research.thesis_contract import event
    start = pd.Timestamp(start)
    return event('candle', timeframe, start, start + pd.Timedelta(timeframe.lower()),
                 {'open': opened, 'high': high, 'low': low, 'close': close,
                  'volume': 10.} if status == 'known' else None, status=status,
                 stream_id='fixture')


def base_episode():
    return {'stream_id': 'fixture', 'parent': {'id': 'parent1', 'lineage_id': 'line1',
                'available_at': '2023-12-31T23:00:00+00:00',
                'range_low': 100., 'range_high': 120.},
            'origin': candle_event('2024-01-01T00:00Z', '4h', low=98., high=108., close=102.),
            'atr4h': 10., 'daily_context': {'status': 'absent'},
            'source_status': 'known', 'execution_authorized': False}


def sequence_events():
    return [candle_event('2024-01-01T04:00Z', low=99., high=105., close=102., opened=102.),
            candle_event('2024-01-01T05:00Z', low=102., high=111., close=109.),
            candle_event('2024-01-01T06:00Z', low=107., high=110., close=109., opened=109.),
            candle_event('2024-01-01T07:00Z', '1min', low=109., high=112., close=111., opened=109.)]


def packet(events=None):
    from scripts.research.thesis_sequence import compile_episode
    return compile_episode(base_episode(), sequence_events() if events is None else events)


def minutes(start='2024-01-01T04:00Z', periods=500, price=104.):
    return pd.DataFrame({'open': price, 'high': price+.1, 'low': price-.1,
                         'close': price, 'volume': 10.},
                        index=pd.date_range(start, periods=periods, freq='min'))
