"""Fixed gate, opportunity-cost and separate-capacity regression tests."""
from copy import deepcopy
import importlib

import pandas as pd
import pytest


def api():
    name = 'scripts.research.lc_room_validation'
    assert importlib.util.find_spec(name), 'room validation helper missing'
    return importlib.import_module(name)


def case(cid='a', minutes=0, subtype='upside_expansion_candidate'):
    d = pd.Timestamp('2026-08-01T00:00Z') + pd.Timedelta(minutes=minutes)
    plan = dict(decision_time=d.isoformat(), action='enter', level=None, stop=95.,
                entry_expiry=(d+pd.Timedelta('15min')).isoformat(),
                exit_deadline=(d+pd.Timedelta('1h')).isoformat(),
                processing_seconds=90, routing_seconds=0, notional=50000., cost_bps=12)
    return dict(candidate_id=cid, decision_time=d.isoformat(), subtype=subtype,
                plans=dict(immediate=plan), unavailable_reason=None)


def context(c, label):
    return dict(candidate_id=c['candidate_id'], decision_time=c['decision_time'],
                labels=dict(mapped_overhead=label))


def bars():
    return pd.DataFrame(dict(open=100., high=101., low=99., close=100.),
                        index=pd.date_range('2026-08-01T00:00Z', periods=121, freq='min'))


@pytest.mark.parametrize('label,action', [('at_least_2r','enter'), ('below_2r','reject'),
                                        ('no_reference','reject'), ('unknown','reject')])
def test_exact_frozen_label_policy_and_input_immutability(label, action):
    c = case(); original = deepcopy(c)
    result = api().compile_room_plans([c], [context(c, label)])
    assert result['baseline'][0]['plan']['action'] == 'enter'
    assert result['known_room'][0]['plan']['action'] == action
    assert result['decisions'][0]['room_label'] == label
    assert result['decisions'][0]['reason'] == (None if action == 'enter' else label)
    assert c == original


@pytest.mark.parametrize('bad', ['missing', 'extra', 'duplicate', 'clock', 'label'])
def test_bad_context_join_is_not_silently_an_abstention(bad):
    c = case(); contexts = [context(c, 'at_least_2r')]
    if bad == 'missing': contexts = []
    if bad == 'extra': contexts.append(context(case('b'), 'no_reference'))
    if bad == 'duplicate': contexts += deepcopy(contexts)
    if bad == 'clock': contexts[0]['decision_time'] = '2026-08-02T00:00Z'
    if bad == 'label': contexts[0]['labels']['mapped_overhead'] = 'infinite'
    with pytest.raises(ValueError):
        api().compile_room_plans([c], contexts)


def test_unavailable_original_plan_stays_unavailable_even_if_rule_abstains():
    c = case(); c.update(plans=None, unavailable_reason='price_gap')
    result = api().compile_room_plans([c], [context(c, 'unknown')])
    assert result['known_room'][0]['plan'] is None
    assert result['known_room'][0]['unavailable_reason'] == 'price_gap'


def test_books_are_separate_and_rejected_candidate_frees_capacity():
    a, b = case('a'), case('b', 1)
    result = api().score_room([a, b], [context(a, 'below_2r'), context(b, 'at_least_2r')],
                              bars(), first_month='2026-08', last_month='2026-08')
    assert set(result['scenarios']) == {'12bps_90s','24bps_90s','12bps_300s','24bps_300s'}
    s = result['scenarios']['12bps_90s']
    assert s['books']['baseline']['admission_order'] == ['a']
    assert s['books']['known_room']['admission_order'] == ['b']
    assert s['attribution']['counts'] == {'avoided_loser': 1, 'new_loser_after_capacity_change': 1}
    assert s['metrics']['baseline']['net_pnl'] == -60.
    assert s['metrics']['known_room']['net_pnl'] == -60.
    assert s['net_pnl_delta'] == 0.
    assert result['profitability_certified'] is False


def test_all_rejected_is_cash_not_success_and_missed_winners_are_counted():
    c = case(); prices = bars()
    prices.loc[prices.index[3], 'high'] = 111.
    r = api().score_room([c], [context(c, 'no_reference')], prices,
                        first_month='2026-08', last_month='2026-08')['scenarios']['12bps_90s']
    assert r['attribution']['counts'] == {'missed_winner': 1}
    assert r['metrics']['known_room']['net_pnl'] == 0.
    assert r['metrics']['known_room']['mean_net_r'] is None
    assert r['metrics']['known_room']['mean_net_r_per_supplied_candidate'] == 0.
    assert r['net_pnl_delta'] < 0


def test_abstentions_remain_in_opportunity_denominator():
    a, b = case('a'), case('b', 70)
    prices = bars(); prices.loc[prices.index[3], 'high'] = 111.
    r = api().score_room([a,b], [context(a,'at_least_2r'),context(b,'no_reference')], prices,
                        first_month='2026-08',last_month='2026-08')['scenarios']['12bps_90s']
    m = r['metrics']['known_room']
    assert m['candidate_count'] == 2 and m['closed_count'] == 1
    assert m['mean_net_r_per_supplied_candidate'] == m['mean_net_r']/2


def test_unknown_future_is_not_zero_or_dropped():
    c = case()
    r = api().score_room([c], [context(c, 'at_least_2r')], bars().iloc[:10],
                        first_month='2026-08', last_month='2026-08')['scenarios']['12bps_90s']
    for m in r['metrics'].values():
        assert m['net_pnl'] is None
        assert m['mean_net_r_per_supplied_candidate'] is None
    assert r['net_pnl_delta'] is None


def test_empty_upside_cohort_is_explicit_not_error_or_positive_evidence():
    c = case(subtype='downside_rebound_candidate')
    r = api().score_room([c], [], bars(), first_month='2026-08', last_month='2026-08')
    assert r['source_case_count'] == 1 and r['selected_count'] == 0
    assert r['scenarios']['12bps_90s']['metrics']['known_room']['mean_net_r'] is None


@pytest.mark.parametrize('bad', ['copied_path', 'changed_bytes'])
def test_actual_consumed_paths_must_be_bound(tmp_path, bad):
    name = 'scripts.research.run_lc_room_validation'
    assert importlib.util.find_spec(name), 'room runner missing'
    runner = importlib.import_module(name)
    from scripts.research.lc_judgment_runner import _save_equal, _sha
    original = _save_equal(tmp_path/'original.json', {'frozen': True})
    files = {str(original): _sha(original)}
    other = _save_equal(tmp_path/'other.json', {'frozen': bad == 'copied_path'})
    consumed = other if bad == 'copied_path' else original
    if bad == 'changed_bytes':
        files[str(original)] = _sha(other)
    with pytest.raises(ValueError, match='consumed'):
        runner.require_bound(files, [consumed])
