"""Literal proposals: catch fabricated geometry, weak bindings and hidden obstacles."""
import importlib
import json
import math
from copy import deepcopy

import pytest

from scripts.research.conditional_assessment import digest
from scripts.research.lc_structure_packet import build_structure_packet
from tests.research.lc_structure_fixtures import structure_source, structure_policy, structure_answer


def api():
    name = 'scripts.research.lc_structure_proposal'
    assert importlib.util.find_spec(name), 'structure proposal API missing'
    return importlib.import_module(name)


def inputs(decision='enter_proposal', mutate=None):
    s=structure_source(mutate); p=build_structure_packet(s); policy=structure_policy()
    return s,p,policy,structure_answer(p,policy,decision)


def grade(s,p,policy,a):
    return api().validate_structure_proposal(s,p,json.dumps(a),policy)


@pytest.mark.parametrize('decision,status', [('enter_proposal','valid_proposal'),
    ('wait_proposal','valid_proposal'),('reject','valid_reject'),('insufficient_evidence','insufficient_evidence')])
def test_positive_controls_are_distinct_and_non_authorizing(decision,status):
    s,p,policy,a=inputs(decision); before=deepcopy((s,p,policy,a))
    r=grade(s,p,policy,a)
    assert r['status']==status and r['errors']==[]
    assert r['execution_authorized'] is False
    assert (s,p,policy,a)==before


@pytest.mark.parametrize('field', ['case_id','packet_sha256','contract_sha256','curriculum_sha256','policy_sha256'])
def test_each_binding_is_required(field):
    s,p,policy,a=inputs();a[field]='wrong'
    assert grade(s,p,policy,a)['status']=='invalid'


@pytest.mark.parametrize('change', ['citation','empty_opposition','empty_competing','price','management',
    'retest','future','sequence_order','stop','target','unresolved','reject_plan','unknowns',
    'authorize','confirm_group','duplicate_ids','sequence_group','horizon','expiry','plan_none',
    'wrong_trigger','bad_claim','scalar_plan','bool_duration'])
def test_defective_answers_never_become_rejects(change):
    s,p,policy,a=inputs()
    if change=='citation': a['supporting'][0]['evidence_ids']=['fake']
    if change=='empty_opposition': a['opposing']=[]
    if change=='empty_competing': a['competing_explanation']['text']=' '
    if change=='price': a['plan']['stop_price']=101.
    if change=='management': a['plan']['trail']=True
    if change=='retest': a['plan']['trigger']={'kind':'retest'}
    if change=='future': a['sequence'][-1]['observation_end']='2026-01-02T00:00:00Z'
    if change=='sequence_order': a['sequence'].reverse()
    if change=='stop': a['plan']['stop_level_id']='bar:5m:11:high'
    if change=='target': a['plan']['destination_level_id']='bar:5m:11:low'
    if change=='unresolved': a['thesis']='unresolved'
    if change=='reject_plan': a['decision']='reject'
    if change=='unknowns': a.update(decision='insufficient_evidence',plan=None)
    if change=='authorize': a['execution_authorized']=True
    if change=='confirm_group': a['plan']['confirmation_evidence_ids']=['parent_4h']
    if change=='duplicate_ids': a['opposing'][0]['evidence_ids']=['current','current']
    if change=='sequence_group': a['sequence'][-1]['evidence_ids']=['parent_4h']
    if change=='horizon': a['plan']['horizon_minutes']=61
    if change=='expiry': a['plan']['expiry_minutes']=16
    if change=='plan_none': a['plan']=None
    if change=='wrong_trigger': a['decision']='wait_proposal'
    if change=='bad_claim': a['parent_child']=False
    if change=='scalar_plan': a['plan']=42
    if change=='bool_duration': a['plan']['expiry_minutes']=True
    r=grade(s,p,policy,a)
    assert r['status']=='invalid' and r['proposal'] is None


def test_omitted_obstacles_fail_even_at_duplicate_prices():
    s,p,policy,a=inputs('wait_proposal')
    assert len(a['plan']['obstacle_level_ids'])>1
    a['plan']['obstacle_level_ids'].pop()
    assert grade(s,p,policy,a)['status']=='invalid'


@pytest.mark.parametrize('raw', ['null','[]','{}','{"decision": "reject", "decision":"enter_proposal"}',
                                '{"value":NaN}', '{"value":Infinity}'])
def test_bad_json_is_invalid_not_a_controller_exception(raw):
    s,p,policy,a=inputs()
    assert api().validate_structure_proposal(s,p,raw,policy)['status']=='invalid'


def test_rebound_parent_conflict_and_absence_are_not_permission_gates():
    for state in ('absent','broken'):
        def change(v):
            x=v['evidence']['parent_4h']
            if state=='absent': x.update(bound=None,pivots=[],status='fail',reasons=['absent_parent'],lineage_broken=False)
            else:
                x.update(status='fail',lineage_broken=True,reasons=['bound_lineage_broken'])
                x['updates'][-1]['source_break_direction']='down'
        s,p,policy,a=inputs(mutate=change)
        assert grade(s,p,policy,a)['status']=='valid_proposal'


def test_missing_mandatory_facts_requires_insufficient_evidence():
    s,p,policy,a=inputs(mutate=lambda p:p['current'].update(validated=False))
    assert grade(s,p,policy,a)['status']=='invalid'
    a=structure_answer(p,policy,'insufficient_evidence')
    assert grade(s,p,policy,a)['status']=='insufficient_evidence'


def test_no_policy_is_diagnostic_only_and_not_an_invented_default():
    s,p,policy,a=inputs();a=structure_answer(p,None)
    r=grade(s,p,None,a)
    assert r['status']=='valid_proposal' and r['policy_sha256'] is None
    assert r['execution_authorized'] is False
    assert grade(s,p,None,structure_answer(p,None,'reject'))['status']=='valid_reject'


@pytest.mark.parametrize('field,value', [('max_entry_price',True),('max_entry_price',float('nan')),
    ('tick_size',0),('horizon_minutes',1.5),('instrument','OTHER'),('routing_seconds',-1),
    ('equity_usd',float('inf')),('entry_expiry_minutes',False),('extra',1)])
def test_invalid_policy_rejected_even_when_answer_rebound_to_it(field,value):
    s,p,policy,a=inputs();policy[field]=value
    # A nonfinite object cannot be canonically sealed; still exercise the API.
    a['policy_sha256']=digest(policy) if not isinstance(value,float) or math.isfinite(value) else '0'*64
    assert grade(s,p,policy,a)['status']=='invalid'


def test_mutating_policy_invalidates_prior_answer():
    s,p,policy,a=inputs();policy['max_entry_price']=107.
    assert 'policy_binding' in grade(s,p,policy,a)['errors']


def test_unrepresentable_integer_policy_is_invalid_not_controller_exception():
    s,p,policy,a=inputs();policy['max_entry_price']=10**309
    r=grade(s,p,policy,a)
    assert r['status']=='invalid'
    assert 'policy_number:max_entry_price' in r['errors']
