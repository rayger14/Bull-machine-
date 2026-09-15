"""Literal public fixtures for the separate LC context judgment contract."""
from copy import deepcopy
import importlib
import json

import pandas as pd
import pytest

from scripts.research.conditional_assessment import digest
from scripts.research.conditional_entry import resolve_entry
from tests.research.test_lc_master_assessment import packet


POLICY = 'lc_context_discrimination_v1'
BRIEF_VERSION = 'lc-context-brief-v1'
SETTINGS = dict(entry_expiry_minutes=15, processing_seconds=90, routing_seconds=0)
FORBIDDEN = {
    '6f84d4de456e046fb38e2762d9381e9e02a2093c09abd4858704afaf70d5c8d3',
    'd30b1cf051576cdd67cc4555a4eeaf4fd53d05d8e4e855b3f54a7252540700cd',
}


def api():
    name = 'scripts.research.lc_context_assessment'
    assert importlib.util.find_spec(name), 'separate LC context adapter missing'
    return importlib.import_module(name)


def reseal(value, field='seal'):
    value.pop(field, None)
    value[field] = digest(value)
    return value


def empty_snapshot():
    value = dict(as_of='2026-01-01T04:00:00Z', training_end='2026-01-01T00:00:00Z',
                 excluded_case_ids=['LC1'], tags=[], records=[], review_ids=[])
    value['id'] = digest(value)
    return value


def context_brief(evidence_ids=None):
    return dict(version=BRIEF_VERSION, policy_id=POLICY, author='controller',
                scope='outcome-free methodology; independently reviewed',
                principles=[dict(text='Judge larger context, sequence, horizon and invalidation.',
                                 evidence_ids=[] if evidence_ids is None else evidence_ids)])


def context_request(prior_bb=.04, current_validated=True):
    source = packet()
    source['previous']['features']['bb_width'] = prior_bb
    source['current']['validated'] = current_validated
    reseal(source)
    return api().build_context_request(source, empty_snapshot(), context_brief(), SETTINGS)


def choice(req, interpretation='support', plan_id='enter'):
    item = dict(text='The supplied context supports this bounded interpretation.',
                evidence_ids=['context'])
    return dict(case_id=req['case_id'], packet_sha256=req['packet_sha256'],
                memory_sha256=req['memory_sha256'], interpretation=interpretation,
                plan_id=plan_id,
                supporting=[deepcopy(item)] if interpretation == 'support' else [],
                opposing=[deepcopy(item)] if interpretation == 'oppose' else [],
                unknowns=[deepcopy(item)] if interpretation == 'uncertain' else [],
                structural_invalidation=dict(text='Contrary completed structure invalidates the thesis.',
                                             evidence_ids=['current', 'parent4h']))


def raw_choice(req, **changes):
    return json.dumps(choice(req, **changes), indent=2)


def review(req, answer, *, complete=True, material_errors=None):
    rr = api().build_context_review_request(req, answer)
    return dict(case_id=req['case_id'], reviewed_sha256=rr['reviewed_sha256'],
                complete=complete, material_errors=material_errors or [],
                notes=['No profitability judgment was made.'])


def raw_review(req, answer, **changes):
    return json.dumps(review(req, answer, **changes), indent=2)


def test_context_choice_survives_old_geometry_failure():
    req = context_request(prior_bb=.2)
    assert api().grade_context_choice(req, choice(req, interpretation='support', plan_id='enter')) == []
    assert api().grade_context_choice(
        req, choice(req, interpretation='support', plan_id='wait_5m_high')) == []
    assert set(req['plan_menu']['plans']) == {'enter', 'wait_5m_high', 'reject'}
    assert 'conditions' not in req['source_packet']


def test_unknown_source_is_not_a_profitable_rejection():
    req = context_request(current_validated=False)
    assert api().grade_context_choice(req, choice(req, interpretation='oppose', plan_id='reject'))
    assert api().grade_context_choice(req, choice(req, interpretation='uncertain', plan_id=None)) == []


def test_build_preserves_original_and_owns_context_groups_and_instructions():
    source = packet(); before = deepcopy(source)
    req = api().build_context_request(source, empty_snapshot(), context_brief(), SETTINGS)
    assert source == before
    assert req['version'] == POLICY
    assert req['context']['source_packet_sha256'] != req['packet_sha256']
    assert set(req['source_packet']['group_catalog']) == {
        'current', 'previous', 'native', 'context', 'parent4h', 'parent1d',
        '1m', '5m', '15m', '1h', '4h', '1d', 'economics', 'limitations'
    }
    assert req['instruction'] == req['plan_menu']['instruction']
    assert 'conditions' not in json.dumps(req['source_packet'])
    assert 'lc_nested_child_rejection_v1' not in json.dumps(req['source_packet'])
    assert 'probability' not in json.dumps(req['response_schema'])
    api().validate_context_request(req)


@pytest.mark.parametrize('change', ['settings', 'stop', 'atr', 'notional', 'cost', 'horizon'])
def test_fixed_settings_and_economics_cannot_silently_change(change):
    source = packet(); settings = deepcopy(SETTINGS)
    if change == 'settings': settings['processing_seconds'] = 89
    if change == 'stop': source['plan']['stop'] += 1.
    if change == 'atr': source['current']['features']['atr_14'] += 1.
    if change == 'notional': source['plan']['notional'] = 1.
    if change == 'cost': source['plan']['roundtrip_cost'] = 0.
    if change == 'horizon': source['plan']['horizon_minutes'] = 1
    reseal(source)
    with pytest.raises(ValueError):
        api().build_context_request(source, empty_snapshot(), context_brief(), settings)


def test_known_nonnative_candidate_and_missing_wait_anchor_fail_explicitly():
    source = packet()
    source['native']['diagnostic']['native_signal']['direction'] = 'short'
    reseal(source)
    with pytest.raises(ValueError, match='native'):
        api().build_context_request(source, empty_snapshot(), context_brief(), SETTINGS)
    source = packet(); source['evidence']['5m'] = []; reseal(source)
    with pytest.raises(ValueError, match='wait_5m_high'):
        api().build_context_request(source, empty_snapshot(), context_brief(), SETTINGS)


@pytest.mark.parametrize('where', ['snapshot', 'brief'])
@pytest.mark.parametrize('record_id', sorted(FORBIDDEN))
def test_old_policy_records_are_rejected_from_new_memory_and_brief(where, record_id):
    snapshot = empty_snapshot(); brief = context_brief()
    if where == 'snapshot':
        record = dict(kind='hypothesis', author='fixture', content={'text': 'old policy'},
                      source_refs=['source'], tags=[], available_at=None, event_end=None,
                      case_ids=[])
        record['id'] = record_id
        snapshot['records'] = [record]; snapshot['review_ids'] = ['a' * 64]
        reseal(snapshot, 'id')
    else:
        brief['principles'][0]['evidence_ids'] = [record_id]
    with pytest.raises(ValueError):
        api().build_context_request(packet(), snapshot, brief, SETTINGS)


def test_brief_policy_and_curriculum_evidence_are_bound():
    snapshot = empty_snapshot()
    record = dict(kind='hypothesis', author='fixture', content={'text': 'context method'},
                  source_refs=['source'], tags=[], available_at=None, event_end=None,
                  case_ids=[])
    record['id'] = digest(record)
    snapshot['records'] = [record]; snapshot['review_ids'] = ['a' * 64]; reseal(snapshot, 'id')
    req = api().build_context_request(packet(), snapshot, context_brief([record['id']]), SETTINGS)
    assert req['source_packet']['group_catalog'][record['id']] == [
        'source_packet', 'curriculum', 0]
    answer = choice(req); answer['supporting'][0]['evidence_ids'] = [record['id'], 'context']
    assert api().grade_context_choice(req, answer) == []
    for key, value in [('version', 'source-only-v1'), ('policy_id', 'old-policy')]:
        bad = context_brief([record['id']]); bad[key] = value
        with pytest.raises(ValueError):
            api().build_context_request(packet(), snapshot, bad, SETTINGS)


@pytest.mark.parametrize('change', ['context', 'menu', 'settings', 'source', 'memory'])
def test_resealed_request_mutations_fail_recomputation(change):
    req = context_request()
    if change == 'context':
        req['context']['hourly']['inside_bar'] = not req['context']['hourly']['inside_bar']
        req['context_sha256'] = digest(req['context'])
    if change == 'menu':
        req['plan_menu']['plans'].pop('reject'); reseal(req['plan_menu'])
        req['menu_sha256'] = digest(req['plan_menu'])
    if change == 'settings': req['settings']['routing_seconds'] = 1
    if change == 'source':
        req['source_packet']['current']['features']['close'] += 1.
        reseal(req['source_packet']); req['packet_sha256'] = digest(req['source_packet'])
    if change == 'memory':
        req['memory_snapshot']['as_of'] = '2026-01-02T00:00:00Z'
        reseal(req['memory_snapshot'], 'id'); req['memory_sha256'] = digest(req['memory_snapshot'])
    reseal(req)
    assert api().grade_context_choice(req, choice(req))


@pytest.mark.parametrize('interpretation,plan_id,valid', [
    ('support', 'enter', True), ('support', 'wait_5m_high', True),
    ('oppose', 'reject', True), ('uncertain', None, True),
    ('support', 'reject', False), ('oppose', 'enter', False),
    ('uncertain', 'reject', False), ('support', 'wait_1m_high', False),
])
def test_exact_choice_mapping(interpretation, plan_id, valid):
    req = context_request()
    errors = api().grade_context_choice(req, choice(req, interpretation, plan_id))
    assert (errors == []) is valid


def test_choice_rejects_invalid_citations_and_duplicate_raw_keys():
    req = context_request(); answer = choice(req)
    answer['supporting'][0]['evidence_ids'] = ['missing']
    assert api().grade_context_choice(req, answer)
    raw = json.dumps(choice(req))
    duplicate = raw[:-1] + ',"case_id":"LC1"}'
    assert api().grade_context_choice(req, duplicate)


def test_exact_raw_review_binding_and_review_gate_states():
    req = context_request(); raw = raw_choice(req)
    rr = api().build_context_review_request(req, raw)
    good = review(req, raw)
    assert api().grade_context_review(rr, good) == []
    assert api().gate_context_choice(req, raw, good)['research_plan']['action'] == 'enter'
    assert api().gate_context_choice(req, choice(req), good)['research_plan'] is None
    incomplete = review(req, raw, complete=False)
    assert api().gate_context_choice(req, raw, incomplete)['research_plan'] is None
    material = [dict(category='factual', evidence_ids=['current'],
                     explanation='The exact cited source contradicts the claim.')]
    failed = review(req, raw, material_errors=material)
    assert api().gate_context_choice(req, raw, failed)['research_plan'] is None
    reject = raw_choice(req, interpretation='oppose', plan_id='reject')
    assert api().gate_context_choice(req, reject, review(req, reject))['research_plan']['action'] == 'reject'


def test_real_conditional_resolver_observes_90_second_arm_for_both_choices():
    req = context_request()
    bars = pd.DataFrame(dict(open=105., high=112., low=101., close=109., volume=1.),
                        index=pd.date_range('2026-01-01T04:00Z', periods=5, freq='min'))
    bars.loc[pd.Timestamp('2026-01-01T04:02Z'), 'close'] = 111.
    for plan_id, expected in [('enter', '2026-01-01T04:02:00+00:00'),
                              ('wait_5m_high', '2026-01-01T04:03:00+00:00')]:
        raw = raw_choice(req, plan_id=plan_id)
        plan = api().gate_context_choice(req, raw, review(req, raw))['research_plan']
        resolution = resolve_entry(bars, as_of='2026-01-01T04:04Z', **{
            key: value for key, value in plan.items() if key not in ('notional', 'cost_bps')})
        assert resolution['entry_time'] == expected
