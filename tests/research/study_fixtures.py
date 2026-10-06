"""Small synthetic OHLC and complete-as-needed causal parent fixtures."""
from copy import deepcopy

import pandas as pd


def minutes(count=120, start='2024-01-01T00:00:00Z', price=101.0):
    index = pd.date_range(start, periods=count, freq='min')
    return pd.DataFrame({'open': price, 'high': price + 1, 'low': price - 1,
                         'close': price, 'volume': 1.0}, index=index)


def parent_ledger():
    return {
        'manifest': {'instrument': 'BTC', 'data_stream_id': 'fixture',
                     'parameters': {'anchor_timeframe': '4H', 'pivot_n': 3}},
        'coverage': {'first_open': '2023-12-01T00:00:00Z',
                     'query_exclusive_end': '2024-02-02T00:00:00Z'},
        'versions': [{
            'id': 'parent-v1', 'lineage_id': 'lineage-1', 'predecessor_version_id': None,
            'creation_reason': 'formation', 'formation_hour': '2023-12-31T19:00:00Z',
            'available_at': '2023-12-31T20:00:00Z', 'range_low': 90.0,
            'range_high': 110.0, 'low_pivot_id': 'low-pivot', 'high_pivot_id': 'high-pivot',
        }],
        'transitions': [{
            'id': 'parent-formed', 'source_hour': '2023-12-31T19:00:00Z',
            'available_at': '2023-12-31T20:00:00Z', 'pre_state': 'absent',
            'pre_lineage_id': None, 'pre_version_id': None, 'post_state': 'active',
            'post_lineage_id': 'lineage-1', 'post_version_id': 'parent-v1',
            'source_break_direction': None,
        }],
    }


def with_parent_break(ledger, when, direction='down'):
    result = deepcopy(ledger)
    at = pd.Timestamp(when)
    result['transitions'].append({
        'id': 'break-' + at.isoformat(), 'source_hour': (at - pd.Timedelta('1h')).isoformat(),
        'available_at': at.isoformat(), 'pre_state': 'active', 'pre_lineage_id': 'lineage-1',
        'pre_version_id': 'parent-v1', 'post_state': 'broken', 'post_lineage_id': None,
        'post_version_id': None, 'source_break_direction': direction,
    })
    return result


def nested_minutes():
    data = minutes()
    data.iloc[30:35, data.columns.get_indexer(['open', 'high', 'low', 'close'])] = [102.5, 103, 102.2, 102.8]
    data.iloc[35:40, data.columns.get_indexer(['open', 'high', 'low', 'close'])] = [102.5, 103.1, 101.7, 102.5]
    data.iloc[40, data.columns.get_indexer(['open', 'high', 'low', 'close'])] = [102.5, 103.5, 102.2, 103.2]
    return data


def qualified_parent_ledger():
    """Synthetic, internally hash-consistent constructor output; not real evidence."""
    from scripts.research.replay_clock import digest
    ledger = parent_ledger()
    contract = ledger['manifest']['contract_id'] = 'synthetic-parent-contract'
    ledger['pivots'] = []
    for side, level in [('low', 90.), ('high', 110.)]:
        pivot = {'side': side, 'level': level, 'pivot_open': '2023-12-30 12:00:00+00:00',
                 'pivot_close': '2023-12-30 16:00:00+00:00', 'confirming_close': '2023-12-31 04:00:00+00:00',
                 'available_at': '2023-12-31 04:00:00+00:00', 'instrument': 'BTC', 'data_stream_id': 'fixture',
                 'anchor_timeframe': '4H', 'pivot_n': 3, 'evidence_id': side + '-support',
                 'supporting_anchor_ids': [side + str(i) for i in range(7)]}
        pivot['id'] = digest({'kind': 'anchor_pivot', 'contract_id': contract,
                              **{k: pivot[k] for k in ('data_stream_id', 'instrument', 'side', 'pivot_open',
                                                       'pivot_close', 'confirming_close', 'level', 'evidence_id')}})
        ledger['pivots'].append(pivot)
    version = ledger['versions'][0]
    version['low_pivot_id'], version['high_pivot_id'] = [p['id'] for p in ledger['pivots']]
    version['id'] = digest({'kind': 'parent_version', 'contract_id': contract, 'data_stream_id': 'fixture',
                            'source_hour': version['formation_hour'], **{k: version[k] for k in
                            ('lineage_id', 'predecessor_version_id', 'creation_reason', 'range_low',
                             'range_high', 'low_pivot_id', 'high_pivot_id')}})
    ledger['transitions'][0]['post_version_id'] = version['id']
    return ledger
