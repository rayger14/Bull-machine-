"""Descriptive research labels must not become permission gates."""
from copy import deepcopy
import importlib

import pytest

from tests.research.test_lc_context_facts import price_case, reseal


def module():
    name = 'scripts.research.lc_setup_preflight'
    assert importlib.util.find_spec(name), 'LC preflight adapter missing'
    return importlib.import_module(name)


@pytest.mark.parametrize('low,high,close,want', [
    (95., 112., 98., 'downside_rebound_candidate'),
    (101., 115., 114., 'upside_expansion_candidate'),
    (101., 112., 108., 'unresolved'),
    (99., 110., 105., 'downside_rebound_candidate'),
    (100., 110., 100., 'unresolved'),
])
def test_geometry_labels_thesis_not_trade_permission(low, high, close, want):
    p = price_case(low, high, close)
    before = deepcopy(p)
    result = module().annotate_lc_setup(p)
    assert result['subtype'] == want
    assert result['execution_authorized'] is False
    assert result['context']['native_long'] is True
    assert 'plan_id' not in result
    assert p == before


def test_rsi_does_not_override_price_geometry():
    p = price_case(95., 112., 98.)
    p['current']['features']['rsi_14'] = 80.
    assert module().annotate_lc_setup(reseal(p))['subtype'] == 'downside_rebound_candidate'


def test_untrusted_current_is_unknown_and_retained():
    p = price_case(95., 112., 98.)
    p['current']['validated'] = False
    report = module().build_preflight([reseal(p)], outcome_exposed_ids=set())
    assert report['case_count'] == 1
    row = report['cases'][0]
    assert row['annotation']['subtype'] == 'unresolved'
    assert row['evidence_ready'] is False
    assert 'hourly_unknown' in row['unknown_reasons']


def test_missing_minute_evidence_is_not_a_profitable_skip():
    p = price_case(95., 112., 98.)
    p['evidence']['1m'] = []
    report = module().build_preflight([reseal(p)], outcome_exposed_ids={p['case_id']})
    row = report['cases'][0]
    assert row['evidence_ready'] is False
    assert '1m_unknown' in row['unknown_reasons']
    assert row['exposure_status'] == 'outcome_exposed'
    assert 'pnl' not in row and 'decision' not in row


def test_absent_parent_is_known_state_not_readiness_failure():
    p = price_case(95., 112., 98.)
    p['evidence']['parent_4h'].update(status='fail', bound=None, pivots=[],
        reasons=['absent_parent'], lineage_broken=False)
    report = module().build_preflight([reseal(p)], outcome_exposed_ids=set())
    assert report['cases'][0]['evidence_ready'] is True
    assert report['cases'][0]['annotation']['context']['parent_4h']['lifecycle'] == 'absent'


def test_future_minute_is_unavailable_not_confirmation():
    p = price_case(95., 112., 98.)
    p['evidence']['1m'][-1][0] = p['decision_time']
    row = module().build_preflight([reseal(p)], outcome_exposed_ids=set())['cases'][0]
    assert row['evidence_ready'] is False
    assert '1m_unknown' in row['unknown_reasons']


def test_order_uniqueness_exposure_and_aggregate_counts():
    a = price_case(95., 112., 98.)
    b = price_case(101., 115., 114.)
    a['case_id'] = 'a'; b['case_id'] = 'b'
    a = reseal(a); b = reseal(b)
    report = module().build_preflight([b, a], outcome_exposed_ids={'a'})
    assert [x['case_id'] for x in report['cases']] == ['a', 'b']
    assert report['ready_count'] == 2
    assert report['subtype_counts'] == {'downside_rebound_candidate': 1, 'upside_expansion_candidate': 1}
    assert report['cases'][1]['exposure_status'] == 'no_new_reveal_recorded'
    with pytest.raises(ValueError, match='duplicate'):
        module().build_preflight([a, a], outcome_exposed_ids=set())
    with pytest.raises(ValueError, match='exposure'):
        module().build_preflight([a], outcome_exposed_ids={'not-in-roster'})


def test_seal_tampering_fails():
    p = price_case(95., 112., 98.)
    p['current']['features']['close'] = 1.
    with pytest.raises(ValueError):
        module().annotate_lc_setup(p)
