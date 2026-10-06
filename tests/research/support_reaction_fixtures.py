"""Literal contrast paths, not fitted historical examples."""
import pandas as pd

from scripts.research.thesis_contract import event
from scripts.research.thesis_sequence import compile_episode

T0 = pd.Timestamp('2024-01-02T04:00Z')


def fixture(*, missing_volume=False, ceiling=150.):
    idx = pd.date_range(T0-pd.Timedelta('24h'), T0+pd.Timedelta('7d'), freq='min')
    bars = pd.DataFrame({'open': 104., 'high': 108., 'low': 100.,
                         'close': 104., 'volume': 10.}, index=idx)
    spring = (bars.index >= T0-pd.Timedelta('4h')) & (bars.index < T0)
    bars.loc[spring, ['open', 'high', 'low', 'close']] = [103., 108., 98., 103.]
    bars.loc[T0:T0+pd.Timedelta('59min'), ['open', 'high', 'low', 'close', 'volume']] = [103., 112., 102., 111., 20.]
    bars.loc[T0+pd.Timedelta('1h'):T0+pd.Timedelta('119min'),
             ['open', 'high', 'low', 'close', 'volume']] = [111., 111.5, 107., 109.5, 5.]
    start = T0+pd.Timedelta('2h')
    bars.loc[start:, ['open', 'high', 'low', 'close']] = [110.2, 110.5, 109.5, 110.2]
    child = [
        (109., 109.2, 108.8, 109.), (109., 109.5, 108.9, 109.2),
        (109.5, 110., 109.2, 109.6), (109.1, 109.7, 108.9, 109.1),
        (109., 109.4, 108.8, 109.), (108.9, 109.2, 108.5, 108.9),
        (108.9, 109.3, 108.7, 109.1), (109.1, 109.4, 108.8, 109.2),
        (109.2, 110.3, 109.2, 110.2),
    ]
    for i, row in enumerate(child):
        bars.loc[start+pd.Timedelta(minutes=i), ['open', 'high', 'low', 'close']] = row
    if missing_volume:
        bars.loc[T0, 'volume'] = float('nan')
    origin = event('candle', '4h', T0-pd.Timedelta('4h'), T0,
                   dict(open=103., high=108., low=98., close=103., volume=2400.), stream_id='fixture')
    base = {'stream_id': 'fixture', 'parent': {'id': 'parent', 'lineage_id': 'lineage',
            'range_low': 100., 'range_high': ceiling, 'available_at': '2024-01-01T23:00:00+00:00'},
            'origin': origin, 'atr4h': 10., 'daily_context': {'status': 'absent'},
            'source_status': 'known', 'execution_authorized': False,
            'observed_through': (T0+pd.Timedelta('7d')).isoformat()}
    # A genuinely has no old mandatory floor-test in this path.
    hours = []
    for at, values in bars.loc[T0:].resample('1h').agg(
            {'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum'}).iterrows():
        if at+pd.Timedelta('1h') <= T0+pd.Timedelta('7d'):
            hours.append(event('candle', '1h', at, at+pd.Timedelta('1h'),
                               {k: float(v) for k, v in values.items()}, stream_id='fixture'))
    return compile_episode(base, hours), bars
