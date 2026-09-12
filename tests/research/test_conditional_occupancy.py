from copy import deepcopy
import importlib

import pandas as pd
import pytest

from scripts.research.entry_case_outcome import score_case


def api():
    name = 'scripts.research.conditional_occupancy'
    assert importlib.util.find_spec(name), 'conditional occupancy replay missing'
    return importlib.import_module(name)


def clock(i):
    return (pd.Timestamp('2026-01-01T00:00Z')+pd.Timedelta(minutes=i)).isoformat()


def bars():
    return pd.DataFrame(dict(open=[100.]*11,high=[101.]*11,low=[99.]*11,close=[100.]*11),
                        index=pd.date_range(clock(0),periods=11,freq='min'))


def candidate(identity, decision=0, **changes):
    plan = dict(decision_time=clock(decision),action='enter',stop=95.,level=None,
                entry_expiry=clock(decision+4),exit_deadline=clock(8),
                processing_seconds=0,routing_seconds=0,notional=50000.,cost_bps=12.)
    plan.update(changes)
    return dict(candidate_id=identity,track='minute',decision_time=clock(decision),
                plan=plan,unavailable_reason=None)


def run(b,candidates,cutoff=10):
    return api().replay_sleeve(b,candidates,track='minute',as_of=clock(cutoff))


def ledger(result):
    return {r['candidate_id']:r for r in result['ledger']}


def test_later_signal_can_enter_first_while_earlier_intent_waits():
    b = bars(); b.loc[b.index[2],['close','high']] = [103.,104.]
    a = candidate('early_wait',action='wait_close_above',level=102.)
    z = candidate('later_enter',1)
    result = run(b,[a,z]); rows = ledger(result)
    assert result['admission_order'] == ['later_enter']
    assert rows['early_wait']['status'] == 'skipped_busy'
    assert rows['early_wait']['blocked_by'] == 'later_enter'
    assert run(b,[z,a]) == result


def test_ready_tie_uses_original_decision_before_id_then_id():
    a = candidate('z',processing_seconds=120)
    b = candidate('a',1,processing_seconds=60)
    assert run(bars(),[b,a])['admission_order'] == ['z']
    assert run(bars(),[candidate('z'),candidate('a')])['admission_order'] == ['a']


def test_busy_skip_never_retries_after_release():
    result = run(bars(),[candidate('first',exit_deadline=clock(4)),candidate('blocked',1)])
    assert result['admission_order'] == ['first']
    assert ledger(result)['blocked']['status'] == 'skipped_busy'


def test_open_stop_gap_releases_before_same_open_entry():
    b = bars(); b.loc[b.index[2]] = [94.,95.,93.,94.]
    result = run(b,[candidate('first'),candidate('second',2,stop=90.)])
    assert result['admission_order'] == ['first','second']
    outcome = ledger(result)['first']['position']
    assert (outcome['exit_phase'],outcome['release_time'],outcome['exit_price']) == ('open',clock(2),94.)
    assert outcome['net_pnl'] == -3060.


def test_intrabar_stop_releases_next_open_not_current_open():
    b = bars(); b.loc[b.index[2],'low'] = 94.
    result = run(b,[candidate('first'),candidate('same_bar',2,stop=90.),candidate('next_bar',3,stop=90.)])
    assert result['admission_order'] == ['first','next_bar']
    assert ledger(result)['same_bar']['status'] == 'skipped_busy'
    outcome = ledger(result)['first']['position']
    assert (outcome['exit_time'],outcome['release_time'],outcome['exit_phase']) == (clock(2),clock(3),'intrabar')


def test_entry_bar_touch_cannot_release_for_second_entry_at_same_open():
    b = bars(); b.loc[b.index[0],['low','high']] = [94.,111.]
    result = run(b,[candidate('a'),candidate('b'),candidate('c',1,stop=90.)])
    assert result['admission_order'] == ['a','c']
    out = ledger(result)['a']['position']
    assert out['exit_reason'] == 'stop' and out['ambiguous_bar'] is True


def test_deadline_open_releases_before_new_entry_without_reading_its_future():
    b = bars(); b.loc[b.index[2],'low'] = 94.
    result = run(b,[candidate('first',entry_expiry=clock(2),exit_deadline=clock(2)),
                    candidate('second',2,stop=90.)],cutoff=2)
    assert result['admission_order'] == ['first','second']
    assert ledger(result)['first']['position']['exit_reason'] == 'deadline'
    assert ledger(result)['second']['position']['status'] == 'open'


def test_pending_data_uncertainty_precedes_same_time_entry():
    b = bars(); b.loc[b.index[0],'close'] = float('nan')
    result = run(b,[candidate('waiting',action='wait_close_above',level=102.),candidate('later',1)])
    assert result['admission_order'] == []
    assert result['uncertain_from'] == clock(1)
    assert ledger(result)['later']['status'] == 'admission_indeterminate'
    assert result['summary']['policy_net_pnl'] is None


def test_active_gap_preserves_admission_but_not_later_certainty():
    b = bars().drop(pd.Timestamp(clock(1)))
    result = run(b,[candidate('first'),candidate('later',2)])
    assert result['admission_order'] == ['first']
    assert ledger(result)['first']['position']['status'] == 'unknown'
    assert ledger(result)['later']['status'] == 'admission_indeterminate'
    assert result['uncertain_from'] == clock(1)


def test_early_exit_does_not_require_future_diagnostic_tail():
    b = bars(); b.loc[b.index[1],'low'] = 94.; b = b.drop(pd.Timestamp(clock(4)))
    result = run(b,[candidate('first')])
    assert ledger(result)['first']['position']['status'] == 'closed'
    assert result['summary']['policy_net_pnl'] == -2560.
    with pytest.raises(ValueError):
        score_case(b,decision_time=clock(0),step_minutes=1,horizon_bars=8,stop=95.)


def test_future_corruption_does_not_change_prefix_admissions():
    b = bars(); candidates = [candidate('a'),candidate('b',1)]
    before = deepcopy(candidates)
    result = run(b,candidates,cutoff=1)
    b.iloc[2:] = float('nan'); b = pd.concat([b,b.iloc[5:6]])
    assert run(b,candidates,cutoff=1) == result
    assert candidates == before


@pytest.mark.parametrize('kind', ['target_gap','target_intrabar','stop_intrabar','deadline'])
def test_complete_single_position_economics_match_existing_scorer(kind):
    b = bars()
    if kind == 'target_gap': b.loc[b.index[1]] = [112.,113.,111.,112.]
    if kind == 'target_intrabar': b.loc[b.index[1],'high'] = 111.
    if kind == 'stop_intrabar': b.loc[b.index[1],'low'] = 94.
    out = ledger(run(b,[candidate('a')]))['a']['position']
    old = score_case(b,decision_time=clock(0),step_minutes=1,horizon_bars=8,stop=95.)
    for key in ('exit_price','exit_time','exit_reason','initial_risk','target_price','quantity','net_pnl'):
        assert out[key] == old[key]


def test_unavailable_assessment_not_zero_return_rejection():
    c = candidate('unknown'); c.update(plan=None,unavailable_reason='review_not_passed')
    result = run(bars(),[c,candidate('valid',1)])
    assert result['admission_order'] == ['valid']
    assert ledger(result)['unknown']['status'] == 'plan_unavailable'
    assert result['summary']['policy_net_pnl'] is None


@pytest.mark.parametrize('change', ['duplicate','mixed_track','clock','missing_reason','cost','size','badplan'])
def test_invalid_candidate_manifests_rejected(change):
    a,b = candidate('a'),candidate('b',1)
    if change == 'duplicate': b['candidate_id'] = 'a'
    if change == 'mixed_track': b['track'] = 'hourly'
    if change == 'clock': b['decision_time'] = clock(0)
    if change == 'missing_reason': b['plan'] = None
    if change == 'cost': b['plan']['cost_bps'] = True
    if change == 'size': b['plan']['notional'] = 1000.
    if change == 'badplan': b['plan']['extra'] = 1
    with pytest.raises(ValueError): run(bars(),[a,b])


def test_invalid_plan_and_pending_remain_in_complete_ledger():
    result = run(bars(),[candidate('invalid',entry_expiry=clock(0)),candidate('future',3)],cutoff=1)
    assert ledger(result)['invalid']['status'] == 'invalid_plan'
    assert ledger(result)['future']['status'] == 'pending'
    assert result['summary']['policy_net_pnl'] is None


def test_reviewed_adapter_retains_failed_review_and_executes_bound_choice():
    # Real compiler/reviewer gate, with explicitly synthetic hand-authored review.
    from scripts.research.conditional_assessment import compile_menu, build_review_request
    from scripts.research.evidence_id_assessment import build_catalog
    p = dict(case_id='X',track='minute',decision_time=clock(0),
        candle_columns=['open_time','open','high','low','close','volume'],
        evidence={'1m':[[clock(-1),100.,101.,99.,100.,1.]],
                  'parent_4h':{'status':'verified_present'},'parent_1d':{'status':'verified_absent'},
                  'setup':{'child_level':99.}},
        plan=dict(direction='long',entry='reference',stop=95.,target='actual entry+2*(actual entry-stop)',
                  horizon_minutes=8,indicative_close=100.,notional=50000.,roundtrip_cost=60.),
        indicative_economics={'actual_fill_known':False})
    p['evidence_catalog'] = build_catalog(p)
    settings = dict(entry_expiry_minutes=4,processing_seconds=0,routing_seconds=0)
    menu = compile_menu(p,settings); eid = next(k for k,v in p['evidence_catalog'].items() if v==['evidence','1m',0])
    choice = dict(case_id='X',plan_id='enter',probability_net_positive=.5,
        facts=dict(decision_time=clock(0),last_1m_close=100.,parent_4h_status='verified_present',
                   parent_1d_status='verified_absent',minute_child_level=99.),
        claims=[dict(category=c,claim='Synthetic fixture.',status='supported',evidence_ids=[eid])
                for c in ('detector','structure','sequence','economics','reason','missing')])
    request = build_review_request(p,menu,choice,settings)
    review = dict(case_id='X',reviewed_sha256=request['reviewed_sha256'],assessment_complete=True,findings=[],verdict='pass')
    inputs = [dict(candidate_id='candidate-X',packet=p,menu=menu,choice=choice,review=review,settings=settings)]
    result = api().replay_reviewed_sleeve(bars(),inputs,track='minute',as_of=clock(10))
    assert result['admission_order'] == ['candidate-X']
    assert result['execution_authorized'] is False
    review.update(assessment_complete=False,verdict='not_assessable')
    result = api().replay_reviewed_sleeve(bars(),inputs,track='minute',as_of=clock(10))
    assert result['admission_order'] == [] and result['summary']['policy_net_pnl'] is None
