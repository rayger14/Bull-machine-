"""Literal source-only fixtures: no private source files or market outcomes."""
from copy import deepcopy
import importlib
import json

import pandas as pd
import pytest

from scripts.research.conditional_assessment import digest, compile_menu
from scripts.research.assessment_evidence_guard import build_envelope
from scripts.research.conditional_entry import score_conditional
from tests.research.test_parent_context_policy import ledger, _break


def api():
    name = 'scripts.research.lc_master_assessment'
    assert importlib.util.find_spec(name), 'LC assessment adapter missing'
    return importlib.import_module(name)


def inputs():
    minute = pd.DataFrame(dict(open=105., high=110., low=100., close=105., volume=1.),
        index=pd.date_range('2025-12-20', '2026-01-01 05:00', freq='min', tz='UTC'))
    minute.loc['2026-01-01 03:00':'2026-01-01 03:59', 'low'] = 99.
    current = dict(timestamp='2026-01-01T03:00:00Z', open=105., high=110.,
                   low=99., close=105., volume=60., atr_14=2., bb_width=.04)
    previous = dict(current, timestamp='2026-01-01T02:00:00Z', low=100.)
    raw = dict(candidate_id='hourly-lc:fixture', decision_time='2026-01-01T04:00:00Z',
        setup_open='2026-01-01T03:00:00Z', features=current, previous_features=previous,
        feature_available_at='2026-01-01T04:00:00Z', feature_observation_ids={},
        feature_observation_visible_at={}, native_emitted=False,
        native_diagnostic={'native_signal': {'direction': 'long'}})
    parents = {}
    for anchor in ('4H', '1D'):
        p = ledger(anchor_timeframe=anchor)
        p['versions'][0]['range_low'] = 90.
        p['pivots'][0]['level'] = 90.
        for t in p['transitions']:
            for key in ('pre_range_low', 'post_range_low', 'source_range_low'):
                if t[key] is not None:
                    t[key] = 90.
        parents[anchor+'_N3'] = p
    provenance = dict(instrument='BTC-USD', data_stream_id='same-stream',
        source_manifest={'files': {'minute': 'a'*64}},
        code_manifest={'files': {'code': 'b'*64}},
        config_manifest={'files': {'config': 'c'*64}},
        atr_contract={'formula_id': 'fixture-atr', 'version': '1'},
        source_sha256='a'*64, reconstruction_verified=True,
        missing_input_limits=['Historical macro unavailable'],
        replay_blockers=['defaulted_derivative:funding_rate'])
    return raw, minute, parents, provenance


def packet():
    return api().build_lc_packet(*inputs(), case_id='LC1')


def request():
    snapshot = dict(as_of='2026-01-01T04:00:00Z', training_end='2026-01-01T00:00:00Z',
                    excluded_case_ids=['LC1'], tags=[], records=[], review_ids=[])
    snapshot['id'] = digest(snapshot)
    brief = dict(version='source-only-v1', content='Evaluate nested structure using supplied sources.')
    return api().build_lc_request(packet(), snapshot, brief,
        dict(entry_expiry_minutes=15, processing_seconds=90, routing_seconds=0))


def choice(req, interpretation='support', plan='enter'):
    item = dict(text='The supplied structure supports this interpretation.', evidence_ids=['conditions'])
    return dict(case_id='LC1', packet_sha256=req['packet_sha256'],
        memory_sha256=req['memory_sha256'], interpretation=interpretation, plan_id=plan,
        supporting=[deepcopy(item)] if interpretation == 'support' else [],
        opposing=[deepcopy(item)] if interpretation == 'oppose' else [],
        unknowns=[dict(text='Derivative history unavailable.', evidence_ids=['limitations'])],
        structural_invalidation=dict(text='Loss of the nested structure.', evidence_ids=['previous','parent4h']))


def review(req, answer):
    rr = api().build_lc_review_request(req, answer)
    return dict(case_id='LC1', reviewed_sha256=rr['reviewed_sha256'], complete=True,
                material_errors=[], notes=['Discretionary preference for waiting.'])


def test_literal_nested_child_pass_and_exact_inequality_boundaries():
    p = packet()
    assert api().evaluate_reference(p)['status'] == 'pass'
    assert p['current']['features']['low'] == 99.
    for change, want in [('low_equal', 'fail'), ('wide', 'fail'), ('outside', 'fail')]:
        raw, bars, parents, provenance = inputs()
        if change == 'low_equal':
            raw['features']['low'] = 100.
            bars.loc['2026-01-01 03:00':'2026-01-01 03:59','low'] = 100.
        if change == 'wide': raw['previous_features']['bb_width'] = .07
        if change == 'outside':
            raw['previous_features']['high'] = 121.
            bars.loc['2026-01-01 02:00':'2026-01-01 02:59','high'] = 121.
        assert api().evaluate_reference(api().build_lc_packet(raw,bars,parents,provenance,'LC1'))['status'] == want


@pytest.mark.parametrize('change', ['history','previous_clock','previous_available','current_future',
                                  'mismatch','stream','anchor_future','provenance'])
def test_unvalidated_operands_never_create_known_geometry_rejections(change):
    raw, bars, parents, provenance = inputs()
    if change == 'history': parents['4H_N3']['transitions'].pop(2)
    if change == 'previous_clock': raw['previous_features']['timestamp'] = '2026-01-01T01:00:00Z'
    if change == 'previous_available': raw['previous_feature_available_at'] = '2026-01-01T04:00:00Z'
    if change == 'current_future': raw['features']['timestamp'] = '2026-01-01T04:00:00Z'
    if change == 'mismatch': raw['features']['low'] = 105.
    if change == 'stream': parents['4H_N3']['manifest']['data_stream_id'] = 'different'
    if change == 'anchor_future': parents['4H_N3']['pivots'][0]['available_at'] = '2026-01-02T00:00:00Z'
    if change == 'provenance': provenance['reconstruction_verified'] = False
    p = api().build_lc_packet(raw,bars,parents,provenance,'LC1')
    assert api().evaluate_reference(p)['status'] == 'unknown'


def test_known_failure_dominates_missing_parent_but_future_bb_is_unknown():
    raw,bars,parents,prov = inputs()
    parents['4H_N3'] = None
    raw['previous_features']['bb_width'] = .07
    p = api().build_lc_packet(raw,bars,parents,prov,'LC1')
    assert api().evaluate_reference(p)['status'] == 'fail'
    raw['previous_feature_available_at'] = '2026-01-01T04:00:00Z'
    assert api().evaluate_reference(api().build_lc_packet(raw,bars,parents,prov,'LC1'))['status'] == 'unknown'


@pytest.mark.parametrize('bb,expected', [(-.01,'pass'),(.06,'pass'),(.060000001,'fail')])
def test_compression_uses_registered_upper_bound_without_an_extra_lower_gate(bb,expected):
    raw,bars,parents,prov=inputs(); raw['previous_features']['bb_width']=bb
    p=api().build_lc_packet(raw,bars,parents,prov,'LC1')
    assert p['conditions']['conditions']['compression']['status']==expected


def test_future_minute_tail_is_not_part_of_packet_and_limits_survive():
    raw,bars,parents,prov = inputs()
    p = api().build_lc_packet(raw,bars,parents,prov,'LC1')
    bars.loc[bars.index >= pd.Timestamp('2026-01-01T04:00Z'),:] = 999.
    assert api().build_lc_packet(raw,bars,parents,prov,'LC1') == p
    assert 'defaulted_derivative:funding_rate' in p['limitations']
    assert set(p['evidence']).issuperset({'1m','5m','15m','1h','4h','1d'})


def test_request_has_owned_groups_bound_economics_and_real_transport_envelope():
    req = request()
    assert req['plan_menu']['plans'] == compile_menu(req['source_packet'],req['settings'])['plans']
    assert set(req['source_packet']['group_catalog']).issuperset(
        {'current','previous','native','parent4h','parent1d','1m','5m','15m','1h','4h','1d','conditions','economics','limitations'})
    assert build_envelope(req)
    assert 'probability_net_positive' not in json.dumps(req)
    assert api().grade_lc_choice(req,choice(req)) == []
    assert build_envelope(api().build_lc_review_request(req, choice(req)))


@pytest.mark.parametrize('interpretation,plan,valid', [
    ('support','enter',True),('support','wait_5m_high',True),('oppose','reject',True),
    ('uncertain',None,True),('support','reject',False),('oppose',None,False),
    ('uncertain','reject',False),('support','invented',False),('support',True,False)])
def test_choice_mapping(interpretation,plan,valid):
    req = request()
    assert (api().grade_lc_choice(req,choice(req,interpretation,plan)) == []) is valid


@pytest.mark.parametrize('change', ['case','packet','memory','keys','empty_support','citation',
    'duplicate','empty_text','long_text','many_items','invalidation','menu','plan_copy','seal'])
def test_choice_and_request_mutations_fail_closed(change):
    req = request(); answer = choice(req)
    if change == 'case': answer['case_id'] = 'other'
    if change == 'packet': answer['packet_sha256'] = 'x'
    if change == 'memory': answer['memory_sha256'] = 'x'
    if change == 'keys': answer['confidence'] = True
    if change == 'empty_support': answer['supporting'] = []
    if change == 'citation': answer['supporting'][0]['evidence_ids'] = ['fake']
    if change == 'duplicate': answer['supporting'][0]['evidence_ids'] = ['current','current']
    if change == 'empty_text': answer['supporting'][0]['text'] = ' '
    if change == 'long_text': answer['supporting'][0]['text'] = 'x'*1201
    if change == 'many_items': answer['supporting'] *= 9
    if change == 'invalidation': answer['structural_invalidation'] = 'stop'
    if change == 'menu': req['plan_menu']['plans']['enter']['parameters']['stop'] = 10.
    if change == 'plan_copy': req['plan']['stop'] = 10.
    if change == 'seal': req['master_brief']['content'] = 'changed'
    assert api().grade_lc_choice(req,answer)


def test_review_notes_are_nonblocking_but_material_errors_and_false_complete_block():
    req=request(); answer=choice(req); critique=review(req,answer)
    assert api().gate_lc_choice(req,answer,critique)['status'] == 'research_ready'
    critique['complete'] = False
    assert api().gate_lc_choice(req,answer,critique)['research_plan'] is None
    critique['complete'] = True
    critique['material_errors'] = [dict(category='factual', evidence_ids=['current'], explanation='A contradicted claim.')]
    assert api().gate_lc_choice(req,answer,critique)['research_plan'] is None
    critique['complete'] = 1
    assert api().grade_lc_review(api().build_lc_review_request(req,answer),critique)


@pytest.mark.parametrize('plan,entry', [('enter','2026-01-01T04:02:00+00:00'),
                                     ('wait_1m_high','2026-01-01T04:03:00+00:00')])
def test_exact_raw_review_binding_and_conditional_resolver_integration(plan,entry):
    req=request(); answer=choice(req,plan=plan); raw=json.dumps(answer,indent=2)
    critique=review(req,raw)
    assert api().gate_lc_choice(req,answer,critique)['research_plan'] is None
    gate=api().gate_lc_choice(req,raw,critique)
    assert gate['research_plan']['stop'] == 99.6
    bars=pd.DataFrame(dict(open=112.,high=114.,low=111.,close=113.,volume=1.),
        index=pd.date_range('2026-01-01T04:00Z',periods=1442,freq='min'))
    result=score_conditional(bars,as_of='2026-01-02T04:01Z',**gate['research_plan'])
    assert result['resolution']['entry_time']==entry
    assert result['outcome']['exit_time']=='2026-01-02T04:00:00+00:00'
    assert result['outcome']['target_price']==136.8
    assert result['outcome']['net_pnl']==pytest.approx(-60.)


def test_broken_lineage_is_known_fail_and_daily_absence_is_descriptive():
    raw,bars,parents,prov=inputs()
    _break(parents['4H_N3']['transitions'][3])
    parents['4H_N3']['transitions'][3]['post_range_low']=90.
    p=api().build_lc_packet(raw,bars,parents,prov,'LC1')
    assert p['conditions']['status']=='fail'
    assert p['evidence']['parent_4h']['lineage_broken'] is True
    raw,bars,parents,prov=inputs(); parents['1D_N3']=None
    assert api().build_lc_packet(raw,bars,parents,prov,'LC1')['conditions']['status']=='pass'


def test_verified_absent_parent_fails_but_missing_history_is_unknown():
    raw,bars,parents,prov=inputs()
    book=parents['4H_N3']; book['versions']=[]; book['pivots']=[]
    for row in book['transitions']:
        row.update(pre_state='forming',post_state='forming',source_range_state='forming')
        for key in ('pre_version_id','post_version_id','evaluated_version_id','pre_lineage_id',
                    'post_lineage_id','pre_range_low','pre_range_high','post_range_low','post_range_high',
                    'source_range_low','source_range_high','adopted_low_pivot_id','adopted_high_pivot_id'):
            row[key]=None
    p=api().build_lc_packet(raw,bars,parents,prov,'LC1')
    assert p['conditions']['status']=='fail'
    assert p['evidence']['parent_4h']['reasons']==['absent_parent']
    parents['4H_N3']=None
    assert api().build_lc_packet(raw,bars,parents,prov,'LC1')['conditions']['status']=='unknown'


@pytest.mark.parametrize('change', ['future_current_failed_low','mismatch_previous_failed_high',
                                 'future_observation','bool_bb','bool_atr'])
def test_invalid_numbers_and_future_observations_never_authorize(change):
    raw,bars,parents,prov=inputs()
    if change=='future_current_failed_low':
        raw['features'].update(timestamp='2026-01-01T04:00Z',low=110.)
    if change=='mismatch_previous_failed_high': raw['previous_features']['high']=121.
    if change=='future_observation': raw['feature_observation_visible_at']={'flow':'2026-01-02T00:00Z'}
    if change=='bool_bb': raw['previous_features']['bb_width']=True
    if change=='bool_atr': raw['features']['atr_14']=True
    p=api().build_lc_packet(raw,bars,parents,prov,'LC1')
    if change=='bool_atr':
        with pytest.raises(ValueError): compile_menu(p,dict(entry_expiry_minutes=15,processing_seconds=90,routing_seconds=0))
    else: assert p['conditions']['status']=='unknown'


def test_structural_fail_and_unknown_restrict_plan_without_silent_rejection():
    for unknown in (False,True):
        raw,bars,parents,prov=inputs()
        if unknown: parents['4H_N3']=None
        else: raw['previous_features']['bb_width']=.07
        p=api().build_lc_packet(raw,bars,parents,prov,'LC1')
        seed=request()
        req=api().build_lc_request(p,seed['memory_snapshot'],seed['master_brief'],seed['settings'])
        assert api().grade_lc_choice(req,choice(req))
        allowed=choice(req,'uncertain',None) if unknown else choice(req,'oppose','reject')
        assert api().grade_lc_choice(req,allowed)==[]
        gated=api().gate_lc_choice(req,allowed,review(req,allowed))
        if unknown: assert gated['research_plan'] is None
        else: assert gated['research_plan']['action']=='reject'


def test_versioned_structured_master_brief_and_group_catalog_integrity():
    req=request()
    brief={'version':'v2','principles':[{'text':'Source-only '+('x'*1500)}]}
    built=api().build_lc_request(req['source_packet'],req['memory_snapshot'],brief,req['settings'])
    assert built['master_brief']==brief
    req['source_packet']['group_catalog']['invented']=['current','features','low']
    # Even a resealed request must not introduce partial or invented groups.
    req['source_packet'].pop('seal'); req['source_packet']['seal']=digest(req['source_packet'])
    req['packet_sha256']=digest(req['source_packet'])
    menu=compile_menu(req['source_packet'],req['settings']); menu['instruction']=req['instruction']
    menu.pop('seal'); menu['seal']=digest(menu)
    req['plan_menu']=menu; req['menu_sha256']=digest(menu)
    req.pop('seal'); req['seal']=digest(req)
    assert api().grade_lc_choice(req,choice(req))


@pytest.mark.parametrize('change', ['hash','case','extra','unknown_id','empty_ids','bad_category','empty_note','many_notes'])
def test_critic_schema_and_binding_mutations(change):
    req=request(); answer=choice(req); rr=api().build_lc_review_request(req,answer); critique=review(req,answer)
    if change=='hash': critique['reviewed_sha256']='stale'
    if change=='case': critique['case_id']='other'
    if change=='extra': critique['verdict']='pass'
    if change in ('unknown_id','empty_ids','bad_category'):
        critique['material_errors']=[dict(category='factual',evidence_ids=['current'],explanation='Check source.')]
        if change=='unknown_id': critique['material_errors'][0]['evidence_ids']=['made-up']
        if change=='empty_ids': critique['material_errors'][0]['evidence_ids']=[]
        if change=='bad_category': critique['material_errors'][0]['category']='disagreement'
    if change=='empty_note': critique['notes']=['']
    if change=='many_notes': critique['notes']=['preference']*9
    assert api().grade_lc_review(rr,critique)


@pytest.mark.parametrize('change', ['missing_review','timeless_case','future_event','future_available',
                                  'excluded_case','duplicate_record','valid'])
def test_memory_snapshot_records_checked_and_curriculum_citations_work(change):
    req=request(); snapshot=deepcopy(req['memory_snapshot'])
    record=dict(kind='hypothesis',author='fixture',content={'text':'Source-only method'},
                source_refs=['fixture-source'],tags=[],available_at=None,event_end=None,case_ids=[])
    if change=='timeless_case': record['case_ids']=['prior-case']
    if change in ('future_event','future_available','excluded_case'):
        record.update(available_at='2025-12-30T00:00Z',event_end='2025-12-29T00:00Z',case_ids=['prior-case'])
        if change=='future_event': record['event_end']='2026-01-01T00:00Z'
        if change=='future_available': record['available_at']='2026-01-02T00:00Z'
        if change=='excluded_case': record['case_ids']=['LC1']
    record['id']=digest(record)
    snapshot['records']=[record]; snapshot['review_ids']=['d'*64]
    if change=='missing_review': snapshot['review_ids']=[]
    if change=='duplicate_record':
        snapshot['records']*=2; snapshot['review_ids']*=2
    snapshot.pop('id'); snapshot['id']=digest(snapshot)
    if change!='valid':
        with pytest.raises(ValueError):
            api().build_lc_request(packet(),snapshot,req['master_brief'],req['settings'])
    else:
        built=api().build_lc_request(packet(),snapshot,req['master_brief'],req['settings'])
        answer=choice(built)
        answer['supporting'][0]['evidence_ids']=[record['id'],'conditions']
        assert api().grade_lc_choice(built,answer)==[]


@pytest.mark.parametrize('change', ['future_feature','mismatched_current','future_observation','unverified_reconstruction'])
def test_unavailable_current_atr_never_compiles_a_native_economic_plan(change):
    raw,bars,parents,prov=inputs()
    if change=='future_feature': raw['feature_available_at']='2026-01-01T05:00Z'
    if change=='mismatched_current': raw['features']['close']=106.
    if change=='future_observation': raw['feature_observation_visible_at']={'flow':'2026-01-01T05:00Z'}
    if change=='unverified_reconstruction': prov['reconstruction_verified']=False
    p=api().build_lc_packet(raw,bars,parents,prov,'LC1')
    assert p['plan']['stop'] is None
    assert p['indicative_economics']['inputs_valid'] is False
    with pytest.raises(ValueError): compile_menu(p,dict(entry_expiry_minutes=15,processing_seconds=90,routing_seconds=0))
    raw['previous_features']['bb_width']=.07
    p=api().build_lc_packet(raw,bars,parents,prov,'LC1')
    assert p['conditions']['status']==('unknown' if change=='unverified_reconstruction' else 'fail')
