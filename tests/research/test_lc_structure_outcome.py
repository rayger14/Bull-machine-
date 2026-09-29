"""Hand-checkable synthetic outcomes, not historical strategy performance."""
import importlib
import json
from copy import deepcopy

import pandas as pd
import pytest

from scripts.research.conditional_assessment import digest
from tests.research.test_lc_structure_preentry import fixture


def api():
    name = 'scripts.research.lc_structure_outcome'
    assert importlib.util.find_spec(name), 'structural-target outcome API missing'
    return importlib.import_module(name)


def minute(at, op=105., hi=106., lo=104., cl=105.):
    return dict(open_time=at, open=op, high=hi, low=lo, close=cl, volume=1.)


def case(wait=False, horizon=3):
    source, packet, policy, answer, execution = fixture(wait)
    policy['horizon_minutes'] = horizon
    answer['plan']['horizon_minutes'] = horizon
    answer['policy_sha256'] = digest(policy)
    at = pd.Timestamp(execution['proposed_fill_at'])
    entry = execution['fill_open']
    rows = [minute((at + pd.Timedelta(minutes=i)).isoformat(),
                   entry, entry + 1, entry - 1, entry) for i in range(horizon)]
    rows.append(dict(open_time=(at + pd.Timedelta(minutes=horizon)).isoformat(),
                     open=entry + 1))
    future = dict(instrument='BTC-USD', data_stream_id='same-stream', minutes=rows)
    return source, packet, policy, answer, execution, future


def score(c):
    s, p, policy, a, e, future = c
    return api().score_structure_outcome(s, p, json.dumps(a), policy, e, future)


@pytest.mark.parametrize('thesis', ['downside_rebound', 'upside_expansion'])
def test_structural_target_not_two_r_and_costs_charged_once(thesis):
    c = case(); c[3]['thesis'] = thesis
    c[5]['minutes'][0]['high'] = 111.
    before = deepcopy(c)
    r = score(c); o = r['outcome']
    assert r['status'] == 'scored' and r['thesis'] == thesis
    assert o['exit_reason'] == 'target' and o['exit_price'] == 110.
    assert o['quantity'] == pytest.approx(100 / 6.126)
    assert o['gross_pnl'] == pytest.approx(5 * 100 / 6.126)
    assert o['modeled_costs'] == pytest.approx(.126 * 100 / 6.126)
    assert o['net_pnl'] == pytest.approx(4.874 * 100 / 6.126)
    assert o['modeled_loss_at_stop'] == pytest.approx(100.)
    assert o['exit_time'] is None  # OHLC does not reveal the intraminute timestamp.
    assert o['exit_bar_open'] == '2026-01-01T04:02:00+00:00'
    assert o['exit_observed_at'] == '2026-01-01T04:03:00+00:00'
    assert r['execution_authorized'] is False and c == before
    assert r['bindings']['packet_sha256'] == c[1]['seal']
    assert r['bindings']['policy_sha256'] == digest(c[2])


@pytest.mark.parametrize('hi,lo,reason,price,ambiguous', [
    (109., 99., 'stop', 99., False), (110., 99., 'stop', 99., True),
    (110., 100., 'target', 110., False)])
def test_first_touch_equality_and_ambiguous_bar_stop_first(hi, lo, reason, price, ambiguous):
    c = case(); c[5]['minutes'][0].update(high=hi, low=lo)
    o = score(c)['outcome']
    assert (o['exit_reason'], o['exit_price'], o['ambiguous_bar']) == (reason, price, ambiguous)
    if reason == 'stop':
        assert o['net_pnl'] == pytest.approx(-100.)
        assert o['net_r'] == pytest.approx(-1.)


@pytest.mark.parametrize('op,reason,exit_price', [(95., 'stop', 95.), (115., 'target', 110.)])
def test_open_gaps_precede_later_extremes_and_cannot_improve_target(op, reason, exit_price):
    c = case()
    c[5]['minutes'][1] = dict(open_time='2026-01-01T04:03:00Z', open=op,
                               high='unobserved', low=None, close=False)
    r = score(c); o = r['outcome']
    assert r['status'] == 'scored'
    assert (o['exit_reason'], o['exit_price']) == (reason, exit_price)
    assert o['exit_time'] == '2026-01-01T04:03:00+00:00'
    assert o['exit_observed_at'] == o['exit_time']
    assert not o['ambiguous_bar']
    if reason == 'stop':
        assert o['stop_gap'] is True and o['net_pnl'] < -100.


@pytest.mark.parametrize('wait,deadline', [(False, '2026-01-01T04:05:00+00:00'),
                                         (True, '2026-01-01T04:06:00+00:00')])
def test_holding_horizon_starts_at_fill_and_reads_deadline_open_only(wait, deadline):
    c = case(wait)
    c[5]['minutes'][-1].update(high=float('nan'), low=False, close='hidden')
    o = score(c)['outcome']
    assert o['exit_reason'] == 'deadline' and o['deadline'] == deadline
    assert o['exit_time'] == deadline
    assert o['exit_price'] == (112. if wait else 106.)


def test_missing_tail_after_resolved_exit_does_not_erase_known_result():
    c = case(); c[5]['minutes'][0]['high'] = 111.
    r = score(c)
    c[5]['minutes'] = c[5]['minutes'][:1] + ['unconsumed corrupt future']
    assert score(c) == r  # Includes the hash of only consumed observations.


@pytest.mark.parametrize('change,reason', [
    ('empty', 'missing_minute'), ('gap', 'coverage_gap'), ('deadline', 'missing_minute'),
    ('duplicate', 'minute_order'), ('reverse', 'minute_order'),
    ('naive', 'invalid_minute'), ('partial', 'invalid_minute'),
    ('boolean', 'invalid_minute'), ('nan', 'invalid_minute'), ('overflow', 'invalid_minute'),
    ('bad_ohlc', 'invalid_minute'), ('volume', 'invalid_minute'),
    ('entry_open', 'entry_open_mismatch'), ('instrument', 'stream_mismatch'),
    ('stream', 'stream_mismatch'), ('shape', 'future_shape')])
def test_unresolved_data_never_becomes_zero_pnl(change, reason):
    c = case(); future = c[5]; rows = future['minutes']
    if change == 'empty': rows.clear()
    if change == 'gap': rows.pop(1)
    if change == 'deadline': rows.pop()
    if change == 'duplicate': rows[1]['open_time'] = rows[0]['open_time']
    if change == 'reverse': rows[1]['open_time'] = '2026-01-01T04:01:00Z'
    if change == 'naive': rows[0]['open_time'] = '2026-01-01T04:02:00'
    if change == 'partial': rows[0]['open_time'] = '2026-01-01T04:02:30Z'
    if change == 'boolean': rows[0]['open'] = True
    if change == 'nan': rows[0]['high'] = float('nan')
    if change == 'overflow': rows[0]['open'] = 10 ** 309
    if change == 'bad_ohlc': rows[0]['low'] = 107.
    if change == 'volume': rows[0]['volume'] = -1.
    if change == 'entry_open': rows[0]['open'] = 105.5
    if change == 'instrument': future['instrument'] = 'ETH-USD'
    if change == 'stream': future['data_stream_id'] = 'different-venue'
    if change == 'shape': future['extra'] = True
    r = score(c)
    assert (r['status'], r['reason']) == ('data_unavailable', reason)
    assert r['outcome']['net_pnl'] is None and r['execution_authorized'] is False


@pytest.mark.parametrize('decision', ['reject', 'insufficient_evidence'])
def test_nonentry_proposals_do_not_receive_a_hypothetical_profit(decision):
    c = case(); a = c[3]; a.update(decision=decision, plan=None)
    if decision == 'insufficient_evidence': a['unknowns'] = a['opposing']
    c[5]['minutes'] = ['must not be read']
    r = score(c)
    assert r['status'] == 'not_scored' and r['reason'] == 'no_entry_proposal'
    assert r['outcome']['net_pnl'] is None


@pytest.mark.parametrize('change,reason', [('policy', 'proposal_invalid'),
    ('cap', 'entry_cap'), ('missing_preentry', 'coverage_gap'),
    ('different_invalidation', 'separate_postentry_invalidation')])
def test_preentry_revalidated_and_unsupported_management_not_silently_ignored(change, reason):
    c = case()
    if change == 'policy': c[2]['risk_budget_usd'] = 50.
    if change == 'cap': c[4]['fill_open'] = 107.
    if change == 'missing_preentry': c[4]['completed_minutes'].pop()
    if change == 'different_invalidation':
        c[3]['plan']['invalidation_level_id'] = next(
            k for k, v in c[1]['levels'].items() if v['price'] == 90.)
    r = score(c)
    assert (r['status'], r['reason']) == ('not_scored', reason)
    assert r['outcome']['net_pnl'] is None


def test_source_tampering_is_controller_error_not_a_profitable_rejection():
    c = case(); c[1]['levels']['bar:5m:11:high']['price'] = 111.
    with pytest.raises(ValueError): score(c)


@pytest.mark.parametrize('field,value,expected_quantity', [
    ('risk_budget_usd', 61.26, 10.), ('max_notional_usd', 1050., 10.),
    ('equity_usd', 525., 10.)])
def test_theoretical_size_respects_each_frozen_risk_cap(field, value, expected_quantity):
    c = case(); c[2][field] = value; c[3]['policy_sha256'] = digest(c[2])
    o = score(c)['outcome']
    assert o['quantity'] == pytest.approx(expected_quantity)
    assert o['net_pnl'] == pytest.approx(8.74)


def test_tick_rounded_destination_is_used_without_moving_source_anchor():
    c = case(); c[2]['tick_size'] = .3; c[3]['policy_sha256'] = digest(c[2])
    c[5]['minutes'][0]['high'] = 110.
    o = score(c)['outcome']
    assert o['target_price'] == pytest.approx(109.8)
    assert o['exit_price'] == pytest.approx(109.8)
    assert c[1]['levels']['bar:5m:11:high']['price'] == 110.


def test_packet_and_execution_bindings_prevent_cross_case_result_confusion():
    c = case(); r = score(c)
    assert r['bindings']['execution_sha256'] == digest(c[4])
    assert len(r['bindings']['raw_response_sha256']) == 64
    assert r['case_id'] == c[1]['case_id']
    assert r['version'] == 'lc_structure_outcome_v1'


@pytest.mark.parametrize('op,reason,price', [(99., 'stop', 99.), (98., 'stop', 98.),
                                          (110., 'target', 110.), (115., 'target', 110.)])
def test_open_barrier_at_deadline_precedes_time_exit_without_reading_later_hlc(op, reason, price):
    c = case(); c[5]['minutes'][-1].update(open=op, high=None, low='unobserved')
    o = score(c)['outcome']
    assert (o['exit_reason'], o['exit_price']) == (reason, price)
    assert o['exit_time'] == '2026-01-01T04:05:00+00:00'


@pytest.mark.parametrize('change,reason', [('policy', 'missing_policy'),
    ('response', 'proposal_invalid'), ('horizon', 'invalid_horizon'),
    ('rounded_stop', 'separate_postentry_invalidation')])
def test_unsupported_inputs_remain_null_without_reading_future(change, reason):
    c = list(case()); c[5]['minutes'] = ['must not be read']
    if change == 'policy': c[2] = None
    if change == 'horizon':
        c[2]['horizon_minutes'] = 10 ** 12
        c[3]['plan']['horizon_minutes'] = 10 ** 12
    if change == 'rounded_stop': c[2]['tick_size'] = .4
    c[3]['policy_sha256'] = digest(c[2]) if c[2] is not None else None
    raw = 'not JSON' if change == 'response' else json.dumps(c[3])
    r = api().score_structure_outcome(c[0], c[1], raw, c[2], c[4], c[5])
    assert (r['status'], r['reason']) == ('not_scored', reason)
    assert r['outcome']['net_pnl'] is None
    assert r['bindings']['consumed_future_sha256'] is None


def test_zero_cost_is_explicit_and_does_not_inherit_a_hidden_fee_default():
    c = case(); c[2]['roundtrip_cost_bps'] = 0.; c[3]['policy_sha256'] = digest(c[2])
    c[5]['minutes'][0]['high'] = 110.
    o = score(c)['outcome']
    assert o['modeled_costs'] == 0.
    assert o['net_pnl'] == pytest.approx(5 * 100 / 6)


def test_binding_changes_for_consumed_valid_prices_but_not_deadline_hlc():
    c = case(); initial = score(c)
    c[5]['minutes'][0]['high'] = 107.
    changed = score(c)
    assert initial['outcome'] == changed['outcome']
    assert initial['bindings']['consumed_future_sha256'] != changed['bindings']['consumed_future_sha256']
    c[5]['minutes'][-1].update(high=float('nan'), low=False, close='not consumed')
    assert score(c) == changed
