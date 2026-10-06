from copy import deepcopy

import pandas as pd
import pytest

from scripts.research.thesis_source import aggregate, build_source, atr_series
from tests.research.thesis_fixtures import minutes


def source_fixture():
    bars = minutes('2023-12-28T00:00Z', periods=12*1440)
    bars.loc['2024-01-01T00:00Z':'2024-01-01T03:59Z', 'low'] = 98.
    bars.loc['2024-01-01T00:00Z':'2024-01-01T03:59Z', 'high'] = 108.
    bars.loc['2024-01-01T03:59Z', 'close'] = 102.
    bars.loc['2024-01-01T04:00Z':'2024-01-01T04:59Z', 'low'] = 99.
    bars.loc['2024-01-01T05:00Z':'2024-01-01T05:59Z', ['high', 'close']] = [111., 109.]
    bars.loc['2024-01-01T06:00Z':'2024-01-01T06:59Z', ['open', 'high', 'low', 'close']] = [109., 110., 107., 109.]
    bars.loc['2024-01-01T07:00Z', ['open', 'high', 'low', 'close']] = [109., 112., 109., 111.]
    def ledger(tf, versions):
        return {'manifest': {'instrument': 'BTC', 'data_stream_id': 'fixture',
                             'parameters': {'anchor_timeframe': tf, 'pivot_n': 3}},
                'coverage': {'first_open': '2023-12-28T00:00:00+00:00',
                             'last_processed_close': '2024-01-09T00:00:00+00:00',
                             'query_exclusive_end': '2024-01-09T01:00:00+00:00'},
                'versions': versions, 'transitions': [{'available_at': v['available_at'],
                     'post_state': 'active', 'post_version_id': v['id']} for v in versions]}
    parent = {'id': 'p1', 'lineage_id': 'l1', 'range_low': 100., 'range_high': 120.,
              'available_at': '2023-12-31T23:00:00+00:00'}
    return bars, {'4H_N3': ledger('4H', [parent]), '1D_N3': ledger('1D', [])}


def test_complete_aggregates_and_sma_true_range_are_literal():
    rows = aggregate(minutes(periods=15*240), '4h', 'fixture')
    atr = atr_series(rows)
    assert all(atr[r['id']] is None for r in rows[:14])
    assert atr[rows[14]['id']] == pytest.approx(.2)
    assert rows[0]['available_at'] == '2024-01-01T08:00:00+00:00'


def test_missing_constituent_unknown_not_forward_filled():
    bars = minutes(periods=240).drop(pd.Timestamp('2024-01-01T04:10Z'))
    assert aggregate(bars, '4h', 'fixture')[0]['status'] == 'unknown'


def test_common_raw_census_sequence_citations_and_parent_strictness():
    bars, parents = source_fixture()
    s = build_source(bars, parents, '2024-01-01T00:00Z', '2024-01-02T00:00Z', stream='fixture')
    assert len(s['packets']) == 1
    p = s['packets'][0]
    assert p['parent']['id'] == 'p1'
    assert p['origin']['available_at'] == '2024-01-01T04:00:00+00:00'
    assert p['entry_intents']['thesis']['decision_time'] == '2024-01-01T07:01:00+00:00'
    assert p['daily_context']['status'] == 'absent'
    assert s['economic_outcomes_computed'] is False
    assert s['issues'] == []
    equal = deepcopy(parents)
    equal['4H_N3']['versions'][0]['available_at'] = '2024-01-01T00:00:00+00:00'
    equal['4H_N3']['transitions'][0]['available_at'] = '2024-01-01T00:00:00+00:00'
    assert not build_source(bars, equal, '2024-01-01T00:00Z', '2024-01-01T04:01Z', stream='fixture')['packets']


def test_source_prefix_future_append_and_rebuild_witness():
    bars, parents = source_fixture()
    a = build_source(bars.loc[:'2024-01-01T07:09Z'], parents, '2024-01-01T00:00Z', '2024-01-02T00:00Z', stream='fixture')
    b = build_source(bars, parents, '2024-01-01T00:00Z', '2024-01-02T00:00Z', stream='fixture')
    assert a['packets'][0]['id'] == b['packets'][0]['id']
    assert a['packets'][0]['entry_intents'] == b['packets'][0]['entry_intents']
    assert a == build_source(bars.loc[:'2024-01-01T07:09Z'], parents, '2024-01-01T00:00Z', '2024-01-02T00:00Z', stream='fixture')


def test_foreign_parent_stream_and_origin_coverage_fail_closed():
    bars, parents = source_fixture()
    parents['4H_N3']['manifest']['data_stream_id'] = 'foreign'
    with pytest.raises(ValueError, match='binding'):
        build_source(bars, parents, '2024-01-01T00:00Z', '2024-01-02T00:00Z', stream='fixture')
