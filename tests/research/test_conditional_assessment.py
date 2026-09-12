from copy import deepcopy
import importlib

import pandas as pd
import pytest

from scripts.research.evidence_id_assessment import build_catalog
from scripts.research.assessment_evidence_guard import build_envelope


SETTINGS = dict(entry_expiry_minutes=15, processing_seconds=0, routing_seconds=0)


def api():
    name = 'scripts.research.conditional_assessment'
    assert importlib.util.find_spec(name), 'conditional assessment adapter missing'
    return importlib.import_module(name)


def packet():
    p = dict(case_id='C1', track='hourly', decision_time='2026-01-01T00:00Z',
        candle_columns=['open_time', 'open', 'high', 'low', 'close', 'volume'],
        evidence={'1m': [['2025-12-31T23:59Z', 100., 102., 99., 101., 5.]],
                  '5m': [['2025-12-31T23:55Z', 100., 104., 98., 101., 20.]],
                  '15m': [['2025-12-31T23:45Z', 100., 105., 97., 101., 50.]],
                  'parent_4h': {'status':'verified_present'},
                  'parent_1d': {'status':'verified_absent'}},
        plan=dict(direction='long', entry='same-source minute OPEN at decision; actual fill withheld',
                  stop=95., target='actual entry+2*(actual entry-stop)', horizon_minutes=60,
                  indicative_close=101., notional=50000., roundtrip_cost=60.),
        indicative_economics={'actual_fill_known':False},
        rulecard={'teaching':'Observe a completed candle before acting.'},
        instruction='Only supplied evidence.', specialist_instructions='No future inputs.',
        input_status={'macro':'unavailable'}, limitations=['synthetic fixture'])
    p['evidence_catalog'] = build_catalog(p)
    return p


def response(p, selected='wait_1m_high'):
    eid = next(k for k,v in p['evidence_catalog'].items() if v == ['evidence','1m',0])
    return dict(case_id='C1', plan_id=selected,
        probability_net_positive=None if selected in (None, 'reject') else .5,
        facts=dict(decision_time=p['decision_time'], last_1m_close=101.,
                   parent_4h_status='verified_present', parent_1d_status='verified_absent',
                   minute_child_level=None),
        claims=[dict(category=c, claim='The last completed minute high was 102.',
                     status='supported', evidence_ids=[eid])
                for c in ('detector','structure','sequence','economics','reason','missing')])


def review_for(p, menu, choice, findings=None, complete=True, verdict='pass', settings=None):
    request = api().build_review_request(p, menu, choice, SETTINGS if settings is None else settings)
    return dict(case_id='C1', reviewed_sha256=request['reviewed_sha256'],
                assessment_complete=complete, findings=findings or [], verdict=verdict)


def test_menu_derives_named_highs_and_preserves_deadline_without_mutation():
    p = packet(); before = deepcopy(p)
    menu = api().compile_menu(p, SETTINGS)
    assert set(menu['plans']) == {'enter','reject','wait_1m_high','wait_5m_high','wait_15m_high'}
    item = menu['plans']['wait_5m_high']
    assert item['parameters']['level'] == 104.
    assert item['parameters']['exit_deadline'] == '2026-01-01T01:00:00+00:00'
    assert item['parameters']['entry_expiry'] == '2026-01-01T00:15:00+00:00'
    assert item['source']['timeframe'] == '5m'
    assert item['source']['available_at'] == '2026-01-01T00:00:00+00:00'
    assert item['confirmation_timeframe'] == '1m'
    assert p == before


def test_named_column_order_not_assumed():
    p = packet()
    p['candle_columns'][1], p['candle_columns'][2] = p['candle_columns'][2], p['candle_columns'][1]
    for tf in ('1m','5m','15m'):
        row = p['evidence'][tf][0]; row[1], row[2] = row[2], row[1]
    p['evidence_catalog'] = build_catalog(p)
    assert api().compile_menu(p, SETTINGS)['plans']['wait_5m_high']['parameters']['level'] == 104.


def test_missing_and_stale_options_explicitly_omitted():
    p = packet(); del p['evidence']['5m']
    p['evidence']['15m'][0][0] = '2025-12-31T23:30Z'
    p['evidence_catalog'] = build_catalog(p)
    menu = api().compile_menu(p, SETTINGS)
    assert set(menu['plans']) == {'enter','reject','wait_1m_high'}
    assert menu['omitted'] == {'wait_5m_high':'missing_candles', 'wait_15m_high':'stale_anchor'}


@pytest.mark.parametrize('change', ['future_anchor','future_other_layer','bad_ohlc','bad_columns','bad_catalog','short','negative_stop'])
def test_unsafe_packet_cannot_compile(change):
    p = packet()
    if change == 'future_anchor': p['evidence']['1m'][0][0] = '2026-01-01T00:00Z'
    if change == 'future_other_layer': p['evidence']['4h'] = [['2026-01-01T00:00Z',100.,102.,99.,101.,5.]]
    if change == 'bad_ohlc': p['evidence']['1m'][0][2] = 50.
    if change == 'bad_columns': p['candle_columns'][2] = 'open'
    if change == 'short': p['plan']['direction'] = 'short'
    if change == 'negative_stop': p['plan']['stop'] = -1.
    p['evidence_catalog'] = build_catalog(p)
    if change == 'bad_catalog': p['evidence_catalog']['fake'] = ['plan','stop']
    with pytest.raises(ValueError): api().compile_menu(p, SETTINGS)


@pytest.mark.parametrize('key,value', [('processing_seconds',True),('routing_seconds',-1),('entry_expiry_minutes',0),('entry_expiry_minutes',61)])
def test_invalid_settings_rejected(key,value):
    settings = dict(SETTINGS, **{key:value})
    with pytest.raises(ValueError): api().compile_menu(packet(),settings)


@pytest.mark.parametrize('key', ['processing_seconds','routing_seconds'])
def test_unrepresentable_delay_cannot_produce_research_ready_plan(key):
    with pytest.raises(ValueError):
        api().compile_menu(packet(),dict(SETTINGS,**{key:10**30}))


def test_factual_findings_must_identify_the_claim():
    p = packet(); menu = api().compile_menu(p,SETTINGS); choice = response(p)
    request = api().build_review_request(p,menu,choice,SETTINGS)
    r = review_for(p,menu,choice,[finding(p,'factual',None)],verdict='fail')
    assert api().grade_review(request,r)


def test_unknown_plan_and_altered_menu_are_rejected():
    p = packet(); menu = api().compile_menu(p,SETTINGS)
    assert api().grade_choice(p,menu,response(p,'invented'),SETTINGS)
    menu['plans']['wait_1m_high']['parameters']['level'] = 1.
    assert api().grade_choice(p,menu,response(p),SETTINGS)


def test_changed_settings_do_not_reuse_old_menu():
    p = packet(); menu = api().compile_menu(p,SETTINGS)
    assert api().grade_choice(p,menu,response(p),dict(SETTINGS,processing_seconds=90))


def test_fact_and_evidence_validation_are_retained():
    p = packet(); menu = api().compile_menu(p,SETTINGS); choice = response(p)
    assert api().grade_choice(p,menu,choice,SETTINGS) == []
    choice['facts']['last_1m_close'] = 500.
    choice['claims'][0]['evidence_ids'] = ['invented']
    errors = api().grade_choice(p,menu,choice,SETTINGS)
    assert 'fact:last_1m_close' in errors and 'claim:0:evidence_ids' in errors


@pytest.mark.parametrize('selected', [None,'reject'])
def test_no_profit_probability_on_nonentry_choice(selected):
    p = packet(); menu = api().compile_menu(p,SETTINGS); choice = response(p,selected)
    assert api().grade_choice(p,menu,choice,SETTINGS) == []
    choice['probability_net_positive'] = .5
    assert api().grade_choice(p,menu,choice,SETTINGS)


def test_review_bound_to_exact_choice_and_settings():
    p = packet(); menu = api().compile_menu(p,SETTINGS); choice = response(p)
    old = review_for(p,menu,choice)
    choice['plan_id'] = 'enter'
    assert api().gate_choice(p,menu,choice,old,SETTINGS)['research_plan'] is None
    settings = dict(SETTINGS,processing_seconds=90); menu2 = api().compile_menu(p,settings)
    assert api().gate_choice(p,menu2,response(p),old,settings)['research_plan'] is None


def test_review_bound_to_claim_text_and_packet_not_just_case():
    p = packet(); menu = api().compile_menu(p,SETTINGS); choice = response(p)
    old = review_for(p,menu,choice)
    choice['claims'][0]['claim'] = 'Changed interpretation.'
    assert api().gate_choice(p,menu,choice,old,SETTINGS)['research_plan'] is None
    p['rulecard']['teaching'] = 'Changed source.'; p['evidence_catalog'] = build_catalog(p)
    menu2 = api().compile_menu(p,SETTINGS)
    assert api().gate_choice(p,menu2,response(p),old,SETTINGS)['research_plan'] is None


def finding(p, rule='ambiguity', index=0):
    return dict(rule_id=rule, claim_index=index,
                evidence_ids=response(p)['claims'][0]['evidence_ids'], explanation='Fixture finding.')


@pytest.mark.parametrize('rule,verdict', [('ambiguity','pass'),('judgment','pass'),('factual','fail'),
                                       ('citation','fail'),('contract','fail'),('polarity','fail')])
def test_materiality_controls_review_verdict(rule,verdict):
    p = packet(); menu = api().compile_menu(p,SETTINGS); choice = response(p)
    request = api().build_review_request(p,menu,choice,SETTINGS)
    review = review_for(p,menu,choice,[finding(p,rule)],verdict=verdict)
    assert api().grade_review(request,review) == []
    review['verdict'] = 'fail' if verdict == 'pass' else 'pass'
    assert api().grade_review(request,review)


@pytest.mark.parametrize('change', ['bad_rule','bad_index','bad_id','missing_support','incomplete_pass'])
def test_invalid_review_findings_cannot_release_plan(change):
    p = packet(); menu = api().compile_menu(p,SETTINGS); choice = response(p)
    f = finding(p,'factual'); review = review_for(p,menu,choice,[f],verdict='fail')
    if change == 'bad_rule': f['rule_id'] = 'made_up'
    if change == 'bad_index': f['claim_index'] = True
    if change == 'bad_id': f['evidence_ids'] = ['missing']
    if change == 'missing_support': f['evidence_ids'] = []
    if change == 'incomplete_pass':
        review.update(findings=[],assessment_complete=False,verdict='pass')
    assert api().grade_review(api().build_review_request(p,menu,choice,SETTINGS),review)
    assert api().gate_choice(p,menu,choice,review,SETTINGS)['research_plan'] is None


def test_abstention_not_promoted_to_valid_rejection():
    p = packet(); menu = api().compile_menu(p,SETTINGS); choice = response(p,None)
    r = api().replay_choice(None,p,menu,choice,review_for(p,menu,choice),SETTINGS,as_of=p['decision_time'])
    assert r['gate']['status'] == 'insufficient_evidence'
    assert r['outcome']['net_pnl'] is None


def test_actual_assessor_and_reviewer_requests_build_transport_envelopes():
    p = packet(); request = api().build_assessor_request(p,SETTINGS)
    assert build_envelope(request)['case_id'] == 'C1'
    menu = request['plan_menu']; choice = response(p)
    review = api().build_review_request(p,menu,choice,SETTINGS)
    assert build_envelope(review)['case_id'] == 'C1'
    assert review['selected_plan']['parameters']['action'] == 'wait_close_above'


def test_reviewed_wait_reaches_real_resolver_without_old_immediate_projection():
    p = packet(); menu = api().compile_menu(p,SETTINGS); choice = response(p)
    b = pd.DataFrame(dict(open=[101.]+[103.]*60, high=[104.]*61,
                          low=[100.]*61, close=[103.]*61),
                     index=pd.date_range('2026-01-01',periods=61,freq='min',tz='UTC'))
    r = api().replay_choice(b,p,menu,choice,review_for(p,menu,choice),SETTINGS,as_of='2026-01-01T00:15Z')
    assert r['gate']['execution_authorized'] is False
    assert r['resolution']['entry_time'] == '2026-01-01T00:01:00+00:00'
    assert r['outcome']['net_pnl'] == pytest.approx(-60.)


def test_valid_reject_can_have_zero_exposure_but_failed_review_cannot():
    p = packet(); menu = api().compile_menu(p,SETTINGS); choice = response(p,'reject')
    review = review_for(p,menu,choice)
    r = api().replay_choice(None,p,menu,choice,review,SETTINGS,as_of=p['decision_time'])
    assert r['outcome']['net_pnl'] == 0.
    review['assessment_complete'] = False; review['verdict'] = 'not_assessable'
    r = api().replay_choice(None,p,menu,choice,review,SETTINGS,as_of=p['decision_time'])
    assert r['outcome']['net_pnl'] is None
