"""Guardrail tests, not exit replay or evidence of a trading edge."""
import importlib
import json

import pandas as pd
import pytest

from scripts.research.conditional_assessment import digest
from scripts.research.lc_structure_packet import build_structure_packet
from tests.research.lc_structure_fixtures import structure_source, structure_policy, structure_answer


def api():
    name='scripts.research.lc_structure_preentry'
    assert importlib.util.find_spec(name), 'structure preentry API missing'
    return importlib.import_module(name)


def fixture(wait=False, count=None):
    s=structure_source();p=build_structure_packet(s);policy=structure_policy()
    if wait: policy['max_entry_price']=115.
    a=structure_answer(p,policy,'wait_proposal' if wait else 'enter_proposal')
    n=count if count is not None else (3 if wait else 2)
    start=pd.Timestamp('2026-01-01T04:00:00Z')
    e=dict(response_available_at='2026-01-01T04:01:30Z',
           proposed_fill_at=(start+pd.Timedelta(minutes=n)).isoformat(),
           fill_open=111. if wait else 105., completed_minutes=[])
    for i in range(n):
        close=111. if wait and i>=2 else 105.
        e['completed_minutes'].append(dict(open_time=(start+pd.Timedelta(minutes=i)).isoformat(),
            open=105.,high=max(105.5,close),low=104.5,close=close,volume=1.))
    return s,p,policy,a,e


def check(s,p,policy,a,e):
    return api().check_structure_preentry(s,p,json.dumps(a),policy,e)


@pytest.mark.parametrize('wait',[False,True])
@pytest.mark.parametrize('thesis',['downside_rebound','upside_expansion'])
def test_positive_hypothetical_cases_and_literal_geometry(wait,thesis):
    s,p,policy,a,e=fixture(wait);a['thesis']=thesis
    r=check(s,p,policy,a,e)
    assert r['status']=='eligible_hypothetical' and r['execution_authorized'] is False
    assert r['geometry']['risk_per_unit']==(12. if wait else 6.)
    assert r['geometry']['reward_per_unit']==(9. if wait else 5.)
    if not wait:
        assert r['geometry']['cost_per_unit']==pytest.approx(.126)
        assert r['geometry']['net_rr']==pytest.approx(4.874/6.126)
        assert r['geometry']['quantity_upper_bound']==pytest.approx(100/6.126)
    assert not {'pnl','order','outcome'} & r.keys()


@pytest.mark.parametrize('change,status,reason',[
    ('cap','cancelled','entry_cap'),('target','cancelled','destination_reached'),
    ('stop','cancelled','preentry_invalidation'),('gap_stop','cancelled','preentry_invalidation'),
    ('missing','not_ready','coverage_gap'),('duplicate','invalid','execution_input'),
    ('reverse','invalid','execution_input'),('partial','invalid','execution_input'),
    ('future_bar','invalid','execution_input'),('future_response','invalid','execution_input'),
    ('naive','invalid','execution_input'),('bool','invalid','execution_input'),
    ('nan','invalid','execution_input'),('bar_ohlc','invalid','execution_input'),
    ('fill_high','invalid','execution_input'),('late_response','not_ready','before_arm'),
    ('expiry','cancelled','expired'),('skip','not_ready','missed_entry'),
    ('fees','cancelled','insufficient_room'),('below_rr','cancelled','insufficient_room')])
def test_preentry_failures_are_explicit(change,status,reason):
    n=15 if change=='expiry' else 3 if change=='skip' else 2
    s,p,policy,a,e=fixture(count=n)
    if change=='cap': e['fill_open']=107.
    if change=='target': e['fill_open']=111.;policy['max_entry_price']=115.
    if change=='stop': e['completed_minutes'][0]['low']=98.
    if change=='gap_stop': e['fill_open']=98.
    if change=='missing': e['completed_minutes'].pop(0)
    if change=='duplicate': e['completed_minutes'][1]['open_time']=e['completed_minutes'][0]['open_time']
    if change=='reverse': e['completed_minutes'].reverse()
    if change=='partial': e['completed_minutes'][-1]['open_time']='2026-01-01T04:01:30Z'
    if change=='future_bar': e['completed_minutes'][-1]['open_time']='2026-01-01T04:02:00Z'
    if change=='future_response': e['response_available_at']='2026-01-01T04:03:00Z'
    if change=='naive': e['proposed_fill_at']='2026-01-01T04:02:00'
    if change=='bool': e['fill_open']=True
    if change=='nan': e['fill_open']=float('nan')
    if change=='bar_ohlc': e['completed_minutes'][0]['high']=1.
    if change=='fill_high': e['high']=120.
    if change=='late_response': policy['routing_seconds']=60.
    if change=='fees': policy['roundtrip_cost_bps']=1000.
    if change=='below_rr': policy['minimum_net_rr']=1.
    a['policy_sha256']=digest(policy)
    r=check(s,p,policy,a,e)
    assert (r['status'],r['reason'])==(status,reason)
    assert r['execution_authorized'] is False


@pytest.mark.parametrize('change,reason',[('equal','trigger_not_met'),('prearm','trigger_not_met'),
                                        ('skip','missed_entry'),('expiry','expired')])
def test_wait_uses_first_strict_postarm_close_and_expiry(change,reason):
    s,p,policy,a,e=fixture(True,count=4 if change=='skip' else 15 if change=='expiry' else 3)
    if change=='equal': e['completed_minutes'][2]['close']=110.
    if change=='prearm':
        e['completed_minutes'][1].update(high=111.,close=111.)
        e['completed_minutes'][2]['close']=105.
    assert check(s,p,policy,a,e)['reason']==reason


def test_actual_response_delay_moves_first_eligible_open():
    s,p,policy,a,e=fixture(count=4);e['response_available_at']='2026-01-01T04:03:10Z'
    assert check(s,p,policy,a,e)['status']=='eligible_hypothetical'


def test_structural_invalidation_can_be_nearer_than_stop():
    s,p,policy,a,e=fixture()
    a['plan']['invalidation_level_id']='bar:1h:22:low'  # pre-setup100 vs stop99
    assert p['levels'][a['plan']['invalidation_level_id']]['price']==100.
    e['completed_minutes'][0]['low']=99.5
    assert check(s,p,policy,a,e)['reason']=='preentry_invalidation'


def test_tick_rounding_and_fill_price_recompute_not_indicative_r():
    s,p,policy,a,e=fixture();policy['tick_size']=4.;a['policy_sha256']=digest(policy);e['fill_open']=103.
    r=check(s,p,policy,a,e)
    assert r['status']=='eligible_hypothetical'
    assert r['geometry']['stop']==96. and r['geometry']['destination']==108.
    assert r['geometry']['risk_per_unit']==7. and r['geometry']['reward_per_unit']==5.


def test_quantity_caps_are_applied_without_becoming_order_size():
    s,p,policy,a,e=fixture();policy['risk_budget_usd']=10000.;policy['max_notional_usd']=100.
    a['policy_sha256']=digest(policy)
    assert check(s,p,policy,a,e)['geometry']['quantity_upper_bound']==pytest.approx(100/105.)


def test_missing_policy_or_nonentry_never_executes():
    s,p,policy,a,e=fixture();a['policy_sha256']=None
    assert check(s,p,None,a,e)['reason']=='missing_policy'
    for decision in ['reject','insufficient_evidence']:
        assert check(s,p,policy,structure_answer(p,policy,decision),e)['reason']=='no_entry_proposal'


def test_does_not_trust_invalid_proposal_or_resealed_packet():
    s,p,policy,a,e=fixture();a['packet_sha256']='wrong'
    assert check(s,p,policy,a,e)['status']=='invalid'
    p['levels']['bar:5m:11:high']['price']=999.
    with pytest.raises(ValueError): check(s,p,policy,a,e)


@pytest.mark.parametrize('where',['policy','fill'])
def test_unrepresentable_numeric_input_is_invalid_not_controller_exception(where):
    s,p,policy,a,e=fixture()
    if where=='policy': policy['max_entry_price']=10**309
    else: e['fill_open']=10**309
    assert check(s,p,policy,a,e)['status']=='invalid'
