"""Small hand-derived sources for the offline LC context contract."""
from copy import deepcopy

import pandas as pd

T = pd.Timestamp('2024-01-03T12:00Z')


def minutes(start=None, periods=2880, price=100.):
    start = T - pd.Timedelta('2d') if start is None else pd.Timestamp(start)
    return pd.DataFrame({'open': price, 'high': price + 1., 'low': price - 1.,
                         'close': price, 'volume': 1.},
                        index=pd.date_range(start, periods=periods, freq='min'))


def source():
    bars = minutes()
    bars.loc[T-pd.Timedelta('1h'):, ['high', 'close']] = [103., 102.]
    raw = {'candidate_id': 'lc:test', 'decision_time': T.isoformat(),
           'feature_available_at': T.isoformat(),
           'native_diagnostic': {'native_signal': {'direction': 'long'}},
           'features': {'open': 100., 'high': 103., 'low': 99., 'close': 102.,
                        'volume': 60., 'atr_14': 2.},
           'previous_features': {'open': 100., 'high': 101., 'low': 99.,
                                 'close': 100., 'volume': 60., 'atr_14': 2.}}
    provenance = {'instrument': 'BTC', 'data_stream_id': 'fixture',
                  'source_artifact_sha256': 'fixture-source',
                  'limitations': ['synthetic'], 'reconstruction_verified': True}
    return raw, bars, {'4H_N3': ledger(), '1D_N3': ledger('1D')}, provenance


def ledger(tf='4H'):
    available = T - pd.Timedelta('1d')
    bound = {'id': 'range-old', 'lineage_id': 'lineage', 'creation_reason': 'formation',
             'formation_hour': (available-pd.Timedelta('1h')).isoformat(),
             'available_at': available.isoformat(), 'range_low': 90., 'range_high': 110.,
             'low_pivot_id': 'low', 'high_pivot_id': 'high', 'predecessor_version_id': None}
    return {'manifest': {'instrument': 'BTC', 'data_stream_id': 'fixture',
                         'parameters': {'anchor_timeframe': tf, 'pivot_n': 3},
                         'contract_id': 'constructor'},
            'coverage': {'first_open': '2023-12-01T00:00Z',
                         'last_processed_close': T.isoformat()},
            'versions': [bound],
            'pivots': [{'id': side, 'available_at': (available-pd.Timedelta('4h')).isoformat(),
                        'side': side, 'level': level, 'data_stream_id': 'fixture'}
                       for side, level in [('low', 90.), ('high', 110.)]],
            'transitions': [{'available_at': (T-pd.Timedelta('1h')).isoformat(),
                             'post_state': 'broken_up', 'pre_version_id': 'range-old',
                             'evaluated_version_id': 'range-old', 'post_version_id': None}]}


def clone(value):
    return deepcopy(value)
