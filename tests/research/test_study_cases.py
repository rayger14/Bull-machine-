from copy import deepcopy

import pytest

from scripts.research.study_cases import project_case, render_case, validate_case
from tests.research.test_r3_census import census
from tests.research.study_fixtures import nested_minutes


def test_packet_is_source_backed_asof_and_has_explicit_missing_context():
    result = census()
    oid = result['opportunities'][0]['id']
    packet = project_case(result, oid, '2024-01-01T00:40Z')
    assert validate_case(packet) == []
    assert [e['kind'] for e in packet['events']] == ['armed', 'breakout', 'retest']
    assert packet['execution_authorized'] is False
    assert {'daily_context', 'fib_time_price', 'gann_timing', 'derivatives_receipts'} <= set(packet['unknown_context'])
    assert 'parent-v1' in render_case(packet)
    prefix = census(nested_minutes().iloc[:40])
    assert packet == project_case(prefix, oid, '2024-01-01T00:40Z')


@pytest.mark.parametrize('field', ['pnl', 'net_pnl', 'MFE', 'MAE', 'outcome', 'profit_factor'])
def test_case_rejects_outcome_contamination_at_any_depth(field):
    result = census()
    packet = project_case(result, result['opportunities'][0]['id'], '2024-01-01T00:41Z')
    packet['parent'][field] = 3
    assert any('forbidden' in e for e in validate_case(packet))


def test_case_rejects_future_events_and_future_parent():
    result = census()
    packet = project_case(result, result['opportunities'][0]['id'], '2024-01-01T00:41Z')
    packet['as_of'] = '2024-01-01T00:40:00+00:00'
    assert any('future' in e for e in validate_case(packet))
    bad = deepcopy(packet)
    bad['parent']['available_at'] = '2024-01-01T00:00:00+00:00'
    assert any('parent_not_preexisting' in e for e in validate_case(bad))
    with pytest.raises(ValueError):
        project_case(result, result['opportunities'][0]['id'], '2024-01-01T00:29Z')


@pytest.mark.parametrize('mutation', ['empty_children', 'forged_event', 'unknown_field', 'future_under_unknown_key', 'child_geometry'])
def test_exact_schema_and_citations_reject_adversarial_packets(mutation):
    result = census()
    packet = project_case(result, result['opportunities'][0]['id'], '2024-01-01T00:41Z')
    if mutation == 'empty_children':
        packet['child_bars'] = []
    elif mutation == 'forged_event':
        packet['events'][-1]['id'] = 'forged'
    elif mutation == 'unknown_field':
        packet['controller_notes'] = 'known future result'
    elif mutation == 'future_under_unknown_key':
        packet['parent']['tomorrow_close'] = 200.
    else:
        packet['child_bars'][0]['high'] = 200.
    assert validate_case(packet)


def test_projection_whitelists_child_fields_and_marks_unresolved_source_citations():
    result = census()
    oid = result['opportunities'][0]['id']
    result['opportunities'][0]['child_bars'][0]['controller_notes'] = 'future information'
    packet = project_case(result, oid, '2024-01-01T00:41Z')
    assert 'controller_notes' not in packet['child_bars'][0]
    assert packet['source_citations']['status'] == 'incomplete'
    assert set(packet['source_citations']['missing_ids']) == {'low-pivot', 'high-pivot'}


def test_qualified_parent_hash_and_formation_clock_are_verified():
    from tests.research.study_fixtures import qualified_parent_ledger
    result = census(parents=qualified_parent_ledger())
    packet = project_case(result, result['opportunities'][0]['id'], '2024-01-01T00:41Z')
    assert packet['source_citations']['status'] == 'resolved'
    assert validate_case(packet) == []
    altered = deepcopy(packet)
    altered['parent']['range_high'] = 10000.
    assert any('parent' in e for e in validate_case(altered))
    altered = deepcopy(packet)
    altered['parent']['available_at'] = '2023-12-31T20:01:00Z'
    assert any('parent' in e for e in validate_case(altered))
