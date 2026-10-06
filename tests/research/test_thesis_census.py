"""Guard indexed selection, complete denominators and chronological attrition."""
from copy import deepcopy
import importlib.util

import pandas as pd
import pytest

from scripts.research.causal_parent_ledger import parent_asof
from scripts.research.thesis_contract import signed
from scripts.research.thesis_sequence import compile_episode
from scripts.research.thesis_source import build_source
from tests.research.test_thesis_source import source_fixture
from tests.research.thesis_fixtures import base_episode, sequence_events, candle_event


def api():
    # An explicit missing-feature failure, not an unrelated collection error.
    assert importlib.util.find_spec('scripts.research.thesis_census'), 'census adapter not implemented'
    from scripts.research import thesis_census
    return thesis_census


def build(bars, parents, **kwargs):
    options = dict(start='2024-01-01T00:00Z', end='2024-01-02T00:00Z',
                   seed='2023-12-28T00:00Z', source_end='2024-01-09T00:00Z', stream='fixture')
    options.update(kwargs)
    return api().build_census(bars, parents, **options)


def test_strict_parent_index_excludes_equal_clock_and_preserves_ties():
    _, parents = source_fixture()
    ledger = parents['4H_N3']
    second = dict(ledger['versions'][0], id='p2', range_low=99.)
    ledger['versions'].append(second)
    ledger['transitions'].append(dict(ledger['transitions'][0], post_version_id='p2'))
    index = api().ParentIndex(ledger)
    assert index.asof('2023-12-31T23:00Z') is None
    assert index.asof('2024-01-01T00:00Z')['id'] == 'p2'
    for t in ['2023-12-31T23:00Z', '2023-12-31T23:01Z', '2024-01-08T23:00Z']:
        assert index.asof(t) == parent_asof(ledger, t, strict=True)
    with pytest.raises(ValueError, match='coverage'):
        index.asof('2024-01-09T01:00Z')


def test_parent_index_rejects_unordered_transitions_and_missing_version():
    _, parents = source_fixture()
    ledger = parents['4H_N3']
    ledger['transitions'].append(dict(ledger['transitions'][0], available_at='2023-12-30T23:00:00+00:00'))
    with pytest.raises(ValueError, match='ordered'):
        api().ParentIndex(ledger)
    ledger['transitions'] = ledger['transitions'][:1]
    ledger['transitions'][0]['post_version_id'] = 'missing'
    with pytest.raises(ValueError, match='version'):
        api().ParentIndex(ledger)


def test_complete_census_has_literal_funnel_and_original_packet_parity():
    bars, parents = source_fixture()
    result = build(bars, parents)
    old = build_source(bars, parents, result['start'], result['end'], stream='fixture')
    assert result['packets'] == old['packets']
    assert result['catalog'] == old['catalog']
    assert result['counts'] == old['counts']
    assert len(result['decisions']) == 6
    assert [d['disposition'] for d in result['decisions']] == [
        'no_active_parent', 'raw_episode', 'consumed_lineage', 'consumed_lineage', 'consumed_lineage', 'consumed_lineage']
    summary = api().summarize_census(result)
    assert summary['totals']['raw_episodes'] == 1
    assert summary['totals']['test'] == summary['totals']['strength'] == summary['totals']['last_support'] == 1
    assert summary['totals']['thesis_intents'] == 1
    assert summary['entry_dispositions'] == {'intent_issued': 1}
    assert summary['months'][0]['month'] == '2024-01'
    assert summary['economic_outcomes_computed'] is False
    assert summary['minimum_fill_floor_possible'] is False


@pytest.mark.parametrize('cut', ['leading', 'trailing'])
def test_truncated_source_rejected_instead_of_shortening_calendar(cut):
    bars, parents = source_fixture()
    bars = bars.iloc[1:] if cut == 'leading' else bars.iloc[:-1]
    with pytest.raises(ValueError, match='bounds'):
        build(bars, parents)


def test_entire_missing_origin_bucket_is_visible_and_disqualifies_source():
    bars, parents = source_fixture()
    bars = bars.drop(pd.date_range('2024-01-01T00:00Z', periods=240, freq='min'))
    result = build(bars, parents)
    assert len(result['decisions']) == 6
    assert result['decisions'][1]['disposition'] == 'unknown_origin'
    assert result['coverage']['missing_minutes'] == 240
    assert 'unknown_origin_candles' in result['issues']
    assert api().summarize_census(result)['source_qualified'] is False


def test_incomplete_source_cannot_prove_the_minimum_fill_floor_impossible():
    bars, parents = source_fixture()
    bars = bars.drop(pd.Timestamp('2024-01-01T00:10Z'))
    summary = api().summarize_census(build(bars, parents))
    assert summary['minimum_fill_floor_possible'] is None
    assert summary['fill_floor_reason'] == 'source_unqualified'


def test_first_entry_decision_survives_later_invalidation_or_unknown():
    b, events = base_episode(), sequence_events()
    invalidation = candle_event('2024-01-02T00:00Z', '4h', low=95., high=110., close=99.)
    p = compile_episode(b, events+[invalidation])
    assert p['sequence_status'] == 'invalidated'
    assert api().entry_disposition(p)['kind'] == 'intent_issued'
    # A known no-entry deadline is not retroactively made invalidated/unknown.
    p['entry_intents']['thesis'] = None
    p['milestones'] = []
    p['entry_closed_at'] = '2024-01-02T04:00:00+00:00'
    p['entry_close_reason'] = 'sequence_expired'
    p['terminal_at'] = '2024-01-03T04:00:00+00:00'
    p['unknown_at'] = '2024-01-04T04:00:00+00:00'
    d = api().entry_disposition(p)
    assert (d['kind'], d['stage'], d['at']) == ('sequence_expired', 'test', '2024-01-02T04:00:00+00:00')
    p['unknown_at'] = '2024-01-01T05:00:00+00:00'
    assert api().entry_disposition(p)['kind'] == 'unknown'


def test_cross_month_partition_keeps_consumed_lineage_and_closed_prefix():
    bars, parents = source_fixture()
    # Shift literal fixture so its first episode precedes February.
    shift = pd.Timedelta('30d')
    bars.index += shift
    for ledger in parents.values():
        for k, v in ledger['coverage'].items():
            ledger['coverage'][k] = (pd.Timestamp(v)+shift).isoformat()
        for r in ledger['versions']+ledger['transitions']:
            r['available_at'] = (pd.Timestamp(r['available_at'])+shift).isoformat()
    # Another qualifying spring from the SAME lineage in the next month.
    bars.loc['2024-02-01T00:00Z':'2024-02-01T03:59Z', 'low'] = 98.
    kwargs = dict(start='2024-01-31T00:00Z', end='2024-02-02T00:00Z',
                  seed='2024-01-27T00:00Z', source_end='2024-02-08T00:00Z')
    whole = build(bars, parents, **kwargs)
    left = build(bars, parents, **dict(kwargs, end='2024-02-01T00:00Z'))
    right = build(bars, parents, **dict(kwargs, start='2024-02-01T00:00Z'), resume=left['continuation'])
    assert len(whole['packets']) == 1
    assert whole['packets'] == left['packets']+right['packets']
    assert whole['decisions'] == left['decisions']+right['decisions']
    assert right['counts']['raw_episodes'] == 0
    assert right['continuation'] == whole['continuation']
    bad = signed(dict(left['continuation'], next_origin_close='2024-02-01T04:00:00+00:00'))
    with pytest.raises(ValueError, match='continuation'):
        build(bars, parents, **dict(kwargs, start='2024-02-01T00:00Z'), resume=bad)


def test_clock_availability_counts_are_not_executed_actions():
    bars, parents = source_fixture()
    result = build(bars, parents)
    obs = api().summarize_census(result)['observability']
    assert obs['fib_anchors_defined'] == 1
    assert obs['gann_clocks_scheduled'] == 7
    assert obs['gann_clocks_observed'] == 7
    assert obs['executed_actions_computed'] is False
