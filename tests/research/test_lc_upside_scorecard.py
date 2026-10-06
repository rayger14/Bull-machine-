"""Isolation and risk accounting tests use hand-calculated synthetic books."""
from copy import deepcopy
import importlib

import pandas as pd
import pytest


def api():
    name = 'scripts.research.lc_upside_scorecard'
    assert importlib.util.find_spec(name), 'upside scorecard missing'
    return importlib.import_module(name)


def case(cid, subtype, unavailable=False):
    return dict(candidate_id=cid, subtype=subtype,
                decision_time='2024-01-01T00:00:00+00:00',
                plans=None if unavailable else {},
                unavailable_reason='gap' if unavailable else None)


def book(values):
    rows = []
    for i, (net, risk, fees) in enumerate(values):
        rows.append(dict(candidate_id=str(i), decision_time=f'2024-0{i+1}-01T00:00:00+00:00',
                         status='admitted', position=dict(status='closed', net_pnl=net,
                         initial_risk=risk, fees=fees)))
    return dict(ledger=rows, summary=dict(supplied_population_resolved=True))


def metrics(b):
    return api().book_metrics(b, first_month='2024-01', last_month='2024-03')


def test_selection_preserves_unavailable_upside_and_does_not_mutate_input():
    cases = [case('a', 'upside_expansion_candidate', True),
             case('b', 'downside_rebound_candidate'), case('c', 'unresolved')]
    before = deepcopy(cases)
    selected = api().upside_cases(cases)
    assert [c['candidate_id'] for c in selected] == ['a']
    assert selected[0]['plans'] is None
    selected[0]['unavailable_reason'] = 'changed'
    assert cases == before


def test_duplicate_identity_is_not_silently_counted_twice():
    c = case('a', 'upside_expansion_candidate')
    with pytest.raises(ValueError, match='unique'):
        api().upside_cases([c, deepcopy(c)])


def test_risk_normalization_includes_costs_not_just_stop_distance():
    # First case risks 100 including fees; second risks 200. Net R is +2, -1.
    result = metrics(book([(200., 90., 10.), (-200., 190., 10.)]))
    assert result['net_pnl'] == 0.
    assert result['profit_factor_dollars'] == 1.
    assert result['mean_net_r'] == .5
    assert result['total_net_r'] == 1.
    assert result['profit_factor_r'] == 2.
    assert result['wins'] == result['losses'] == 1
    assert result['bootstrap_months'] == 3
    assert result['bootstrap_95_mean_net_r'][0] <= 0
    assert result['bootstrap_95_mean_net_r'][1] >= .5


def test_unknown_case_keeps_partial_subtotal_but_invalidates_policy_metrics():
    b = book([(100., 90., 10.)])
    b['ledger'].append(dict(candidate_id='x', decision_time='2024-02-01T00:00:00+00:00',
                           status='plan_unavailable', position=None))
    b['summary']['supplied_population_resolved'] = False
    result = metrics(b)
    assert result['known_net_subtotal'] == 100.
    assert result['net_pnl'] is None
    assert result['mean_net_r'] is None
    assert result['bootstrap_95_mean_net_r'] is None


def test_empty_and_all_rejected_books_are_not_positive_evidence():
    for b in [book([]), dict(ledger=[dict(candidate_id='r',
                 decision_time='2024-01-01T00:00:00+00:00', status='rejected', position=None)],
                 summary=dict(supplied_population_resolved=True))]:
        result = metrics(b)
        assert result['net_pnl'] == 0
        assert result['mean_net_r'] is None
        assert result['bootstrap_95_mean_net_r'] is None
        assert result['profit_factor_dollars'] is None


def test_bootstrap_is_deterministic_and_reports_constant_return_exactly():
    b = book([(100., 90., 10.), (200., 180., 20.), (50., 45., 5.)])
    first = metrics(b)
    assert first == metrics(b)
    assert first['bootstrap_95_mean_net_r'] == [1., 1.]
    assert first['net_without_top_three_winners'] == 0.


@pytest.mark.parametrize('risk,fees,net', [(0.,0.,1.), (-1.,2.,1.), (10.,-1.,1.),
                                        (10.,1.,float('nan'))])
def test_invalid_economics_cannot_be_published_as_finite_metrics(risk, fees, net):
    with pytest.raises(ValueError):
        metrics(book([(net, risk, fees)]))


def test_declared_calendar_must_contain_all_decisions():
    b = book([(100., 90., 10.)])
    b['ledger'][0]['decision_time'] = '2023-12-31T23:00:00+00:00'
    with pytest.raises(ValueError, match='calendar'):
        metrics(b)


def test_replays_upside_in_own_book_without_downside_capacity():
    bars = pd.DataFrame(dict(open=100., high=101., low=99., close=100.),
                        index=pd.date_range('2024-01-01T00:00Z', periods=61, freq='min'))
    def ready(cid, decision, subtype):
        base = dict(decision_time=decision, stop=95., entry_expiry='2024-01-01T00:15:00+00:00',
                    exit_deadline='2024-01-01T01:00:00+00:00', processing_seconds=90,
                    routing_seconds=0, notional=50000., cost_bps=12)
        return dict(candidate_id=cid, decision_time=decision, subtype=subtype,
                    unavailable_reason=None, plans=dict(immediate=dict(base, action='enter', level=None),
                    mechanical_wait=dict(base, action='wait_close_above', level=102.)))
    cases = [ready('down', '2024-01-01T00:00:00+00:00', 'downside_rebound_candidate'),
             ready('up', '2024-01-01T00:01:00+00:00', 'upside_expansion_candidate')]
    result = api().score_upside(cases, bars)
    b = result['replay']['scenarios']['12bps_90s']['immediate']
    assert b['admission_order'] == ['up']
    assert b['summary']['policy_net_pnl'] == -60.
    assert result['source_case_count'] == 2
    assert result['selected_case_count'] == 1
    assert result['metrics']['12bps_90s']['immediate']['net_pnl'] == -60.
    assert result['pristine_holdout'] is False
    assert result['execution_authorized'] is False
