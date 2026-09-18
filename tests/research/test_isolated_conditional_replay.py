"""Behavioral non-interference checks for independently declared research books."""
from copy import deepcopy
import importlib

import pandas as pd
import pytest

from tests.research.test_conditional_occupancy import bars, candidate, clock


def api():
    module = importlib.import_module('scripts.research.conditional_occupancy')
    assert hasattr(module, 'replay_isolated_sleeves'), 'explicit silo replay missing'
    return module


def silo(archetype='liquidity_compression', track='hourly', variant='code', rows=None):
    identity = dict(archetype=archetype, track=track, variant=variant)
    rows = deepcopy(rows if rows is not None else [candidate('same-id')])
    for row in rows:
        row.update(identity)
    return dict(identity, candidates=rows)


def run(silos, b=None, cutoff=10):
    return api().replay_isolated_sleeves(b if b is not None else bars(), silos, as_of=clock(cutoff))


def test_other_archetype_cannot_block_trade_or_change_full_result():
    a, b = silo(), silo('order_block_retest')
    alone = run([a])['silos'][0]
    together = run([b,a])
    assert together['silos'][0] == alone
    assert [s['replay']['admission_order'] for s in together['silos']] == [['same-id'],['same-id']]
    assert [s['replay']['summary']['policy_net_pnl'] for s in together['silos']] == [-60.,-60.]
    assert together['combined_policy_net_pnl'] is None
    assert together['execution_authorized'] is False


@pytest.mark.parametrize('dimension,value', [('track','minute'),('variant','agent')])
def test_same_archetype_other_timeframe_or_variant_has_independent_capacity(dimension,value):
    a = silo()
    b = silo(**{dimension:value})
    together = run([a,b])
    assert len(together['silos']) == 2
    assert all(s['replay']['admission_order'] == ['same-id'] for s in together['silos'])
    assert run([b,a]) == together


def test_busy_skips_remain_local_to_same_silo():
    a = silo(rows=[candidate('first'),candidate('later',1)])
    b = silo('order_block_retest',rows=[candidate('later',1)])
    result = run([a,b])['silos']
    assert result[0]['replay']['admission_order'] == ['first']
    assert result[0]['replay']['ledger'][1]['status'] == 'skipped_busy'
    assert result[1]['replay']['admission_order'] == ['later']


def test_missing_pending_observation_only_contaminates_its_own_book():
    b = bars(); b.loc[b.index[0],'close'] = float('nan')
    a = silo(rows=[candidate('wait',action='wait_close_above',level=102.)])
    other = silo('order_block_retest',rows=[candidate('entry',1)])
    result = run([a,other],b)['silos']
    assert result[0]['replay']['uncertain_from'] == clock(1)
    assert result[0]['replay']['summary']['policy_net_pnl'] is None
    assert result[1] == run([other],b)['silos'][0]
    assert result[1]['replay']['summary']['policy_net_pnl'] == -60.


def test_cost_size_and_unavailable_plan_do_not_leak_between_books():
    missing = candidate('unknown'); missing.update(plan=None,unavailable_reason='review_failed')
    a = silo(rows=[missing])
    b = silo('order_block_retest', rows=[candidate('sized',notional=1000.,cost_bps=5.)])
    result = run([a,b])['silos']
    assert result[0]['replay']['summary']['policy_net_pnl'] is None
    assert result[1]['replay']['summary']['policy_net_pnl'] == -.5
    assert result[1]['replay']['summary']['average_initial_risk'] == 50.


def test_declared_empty_silo_is_retained_without_claiming_source_completeness():
    result = run([silo(rows=[])])
    book = result['silos'][0]['replay']
    assert book['summary']['candidate_count'] == 0
    assert book['source_population_verified'] is False
    assert result['source_population_verified'] is False


def test_inputs_and_other_outputs_are_not_mutated_or_aliased():
    inputs = [silo(),silo('order_block_retest')]; frozen = deepcopy(inputs)
    b = bars(); original_bars = b.copy(deep=True)
    result = run(inputs,b)
    result['silos'][0]['replay']['ledger'][0]['position']['fees'] = 999.
    assert result['silos'][1]['replay']['ledger'][0]['position']['fees'] == 60.
    assert inputs == frozen
    pd.testing.assert_frame_equal(b,original_bars)


@pytest.mark.parametrize('bad', ['duplicate_silo','missing_identity','mixed_archetype','mixed_variant',
    'mixed_track','blank','unhashable','duplicate_candidate','extra_silo_field','nonlist'])
def test_bad_routing_is_rejected_instead_of_silently_repartitioned(bad):
    a = silo(); inputs = [a]
    if bad == 'duplicate_silo': inputs.append(deepcopy(a))
    if bad == 'missing_identity': del a['candidates'][0]['archetype']
    if bad == 'mixed_archetype': a['candidates'][0]['archetype'] = 'other'
    if bad == 'mixed_variant': a['candidates'][0]['variant'] = 'other'
    if bad == 'mixed_track': a['candidates'][0]['track'] = 'minute'
    if bad == 'blank': a['archetype'] = ' '
    if bad == 'unhashable': a['variant'] = []
    if bad == 'duplicate_candidate': a['candidates'] *= 2
    if bad == 'extra_silo_field': a['combined_capital'] = 1.
    if bad == 'nonlist': inputs = None
    with pytest.raises(ValueError): run(inputs)


def reviewed_silo(archetype='liquidity_compression',variant='agent'):
    from scripts.research.conditional_assessment import compile_menu, build_review_request
    from scripts.research.evidence_id_assessment import build_catalog
    identity = dict(archetype=archetype,track='minute',variant=variant)
    p = dict(case_id='X',track='minute',decision_time=clock(0),
        research_identity=dict(identity,candidate_id='same-id'),
        candle_columns=['open_time','open','high','low','close','volume'],
        evidence={'1m':[[clock(-1),100.,101.,99.,100.,1.]],
                  'parent_4h':{'status':'verified_present'},'parent_1d':{'status':'verified_absent'},
                  'setup':{'child_level':99.}},
        plan=dict(direction='long',entry='reference',stop=95.,target='actual entry+2*(actual entry-stop)',
                  horizon_minutes=8,indicative_close=100.,notional=50000.,roundtrip_cost=60.),
        indicative_economics={'actual_fill_known':False})
    p['evidence_catalog'] = build_catalog(p)
    settings = dict(entry_expiry_minutes=4,processing_seconds=0,routing_seconds=0)
    menu = compile_menu(p,settings)
    eid = next(k for k,v in p['evidence_catalog'].items() if v==['evidence','1m',0])
    choice = dict(case_id='X',plan_id='enter',probability_net_positive=.5,
        facts=dict(decision_time=clock(0),last_1m_close=100.,parent_4h_status='verified_present',
                   parent_1d_status='verified_absent',minute_child_level=99.),
        claims=[dict(category=c,claim='Synthetic fixture.',status='supported',evidence_ids=[eid])
                for c in ('detector','structure','sequence','economics','reason','missing')])
    binding = build_review_request(p,menu,choice,settings)
    review = dict(case_id='X',reviewed_sha256=binding['reviewed_sha256'],assessment_complete=True,findings=[],verdict='pass')
    request = dict(identity,candidate_id='same-id',packet=p,menu=menu,choice=choice,review=review,settings=settings)
    return dict(identity,requests=[request])


def run_reviewed(silos):
    return api().replay_reviewed_isolated_sleeves(bars(),silos,as_of=clock(10))


def test_review_gate_is_real_and_failed_review_does_not_contaminate_other_archetype():
    a,b = reviewed_silo(),reviewed_silo('order_block_retest')
    a['requests'][0]['review'].update(assessment_complete=False,verdict='not_assessable')
    result = run_reviewed([a,b])['silos']
    assert result[0]['replay']['admission_order'] == []
    assert result[0]['replay']['summary']['policy_net_pnl'] is None
    assert result[1]['replay']['admission_order'] == ['same-id']
    assert result[1] == run_reviewed([b])['silos'][0]


def test_relabelled_reviewed_packet_cannot_cross_silo_boundary():
    a = reviewed_silo()
    a['variant'] = a['requests'][0]['variant'] = 'other'
    with pytest.raises(ValueError): run_reviewed([a])
    # Changing the bound packet too invalidates its old menu/review, not a new assessment.
    a['requests'][0]['packet']['research_identity']['variant'] = 'other'
    result = run_reviewed([a])['silos'][0]['replay']
    assert result['admission_order'] == []
    assert result['summary']['policy_net_pnl'] is None


def test_missing_reviewed_identity_cannot_be_inferred_from_case_id():
    a = reviewed_silo(); del a['requests'][0]['packet']['research_identity']
    with pytest.raises(ValueError): run_reviewed([a])
