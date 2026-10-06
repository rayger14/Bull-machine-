"""Fresh, hand-derived recognition contracts; not the exposed twelve-case exam.

Mirrors use 400-price so every directional comparison is exercised symmetrically.
Events are deliberately pretagged here to isolate sequential recognition; raw
detector/prefix integration is tested separately. No prices/outcomes are fitted.
"""
import copy
import json

import pandas as pd
import pytest

from engine.wyckoff import events as w


START = pd.Timestamp('2025-03-01T00:00Z')


def bar(i, low, high, close, volume=100., z=0., side='accumulation'):
    r = dict(open=close, low=float(low), high=float(high), close=float(close),
             volume=float(volume), volume_z=z, sos_confidence=.8, sow_confidence=.8,
             timestamp=START + pd.Timedelta(hours=i),
             available_at=START + pd.Timedelta(hours=i+1), clock_required=True)
    if side == 'distribution':
        r.update(open=400-close, close=400-close, low=400-high, high=400-low)
    return r


def feed(sm, i, low, high, close, volume=100., z=0., side='accumulation', event=None):
    return sm.process_bar(i, bar(i, low, high, close, volume, z, side),
                          {event: True} if event else {})


def snapshot(sm):
    value = getattr(sm, 'structural_evidence', None)
    assert isinstance(value, dict), 'causal range/level/phase evidence is missing'
    return copy.deepcopy(value)


def developing(side='accumulation', horizon=15):
    sm = w.WyckoffStateMachine({'timeframe': '1h', 'sm_ar_max_bars': horizon,
                              'sm_m2_context_only': True})
    feed(sm, 0, 100, 108, 102, 1000, 4, side, 'sc' if side == 'accumulation' else 'bc')
    feed(sm, 1, 104, 110, 109, 400, 0, side, 'ar' if side == 'accumulation' else 'as')
    feed(sm, 2, 108, 116, 115, 300, 0, side)
    return sm


def locked(side='accumulation', horizon=15):
    sm = developing(side, horizon)
    feed(sm, 3, 106, 113, 107, 200, 0, side)
    return sm


@pytest.mark.parametrize('side,lo,hi', [('accumulation',100,116), ('distribution',284,300)])
def test_reaction_extends_then_locks_on_later_opposite_extreme_close(side, lo, hi):
    sm = developing(side)
    ref = sm.range_ref
    assert (ref.ar_high if side == 'accumulation' else ref.as_low) == (116 if side == 'accumulation' else 284)
    s = snapshot(sm)
    assert s['range']['status'] == 'developing'
    assert s['range']['bound_id'] is None
    parent = sm.parent_snapshot()['id']
    feed(sm, 3, 106, 113, 107, 200, 0, side)
    s = snapshot(sm)
    assert (s['range']['lower'], s['range']['upper']) == (lo, hi)
    assert s['range']['status'] == 'reaction_complete'
    assert s['range']['extreme_index'] == 2
    assert s['range']['locked_index'] == 3
    assert s['range']['locked_at'] == '2025-03-01T04:00:00+00:00'
    assert sm.parent_snapshot()['id'] == parent


@pytest.mark.parametrize('side', ['accumulation','distribution'])
def test_outside_bar_cannot_extend_and_lock_its_own_turn(side):
    sm = developing(side)
    feed(sm, 3, 106, 119, 107, side=side)
    assert snapshot(sm)['range']['status'] == 'developing'
    feed(sm, 4, 107, 119, 108, side=side)  # equal high must not replace extreme candle
    assert snapshot(sm)['range']['extreme_index'] == 3
    feed(sm, 5, 104, 109, 105, side=side)
    assert snapshot(sm)['range']['locked_index'] == 5


@pytest.mark.parametrize('side', ['accumulation','distribution'])
@pytest.mark.parametrize('bad', ['gap','missing_clock','expired'])
def test_uncertain_or_expired_reaction_cannot_lock(side,bad):
    sm = developing(side, horizon=2 if bad == 'expired' else 15)
    if bad == 'expired':
        feed(sm, 3, 109, 115, 110, side=side)
    i = 4 if bad == 'expired' else 3
    r = bar(i, 106, 113, 107, side=side)
    if bad == 'gap':
        r['timestamp'] += pd.Timedelta(hours=1)
        r['available_at'] += pd.Timedelta(hours=1)
    if bad == 'missing_clock': r.pop('available_at')
    sm.process_bar(i, r, {})
    s = snapshot(sm)
    assert s['range']['bound_id'] is None
    assert s['range']['status'] in ('expired','unavailable')


@pytest.mark.parametrize('side', ['accumulation','distribution'])
@pytest.mark.parametrize('bad', [None,'loud','wide','failed','far','missing_volume'])
def test_tested_range_requires_quiet_narrow_level_test_and_later_hold(side,bad):
    sm = locked(side)
    r = bar(4, 101 if bad != 'far' else 106, 104 if bad != 'far' else 109,
            103 if bad != 'far' else 108, 1200 if bad == 'loud' else 100, side=side)
    if bad == 'wide': r = bar(4,101,111,103,100,side=side)
    if bad == 'missing_volume': r.pop('volume')
    sm.process_bar(4,r,{})
    assert snapshot(sm)['range']['status'] == 'reaction_complete'
    feed(sm,5,100.5 if bad == 'failed' else 102,107,106,side=side)
    s = snapshot(sm)
    assert (s['range']['status'] == 'tested') is (bad is None)
    if bad is None:
        assert s['range']['test']['candidate_index'] == 4
        assert s['range']['test']['confirmed_index'] == 5


@pytest.mark.parametrize('side', ['accumulation','distribution'])
def test_local_strength_is_not_a_parent_escape(side):
    sm = locked(side)
    feed(sm,4,106,113,112,600,2,side,'sos' if side == 'accumulation' else 'sow')
    s = snapshot(sm)
    assert s['strength']['role'] == 'local'
    assert s['escape'] is None
    feed(sm,5,112,120,119,600,2,side)
    s = snapshot(sm)
    assert s['escape']['status'] == 'qualified'
    assert s['escape']['strength_index'] == 4
    assert s['escape']['level'] == (116 if side == 'accumulation' else 284)


@pytest.mark.parametrize('side', ['accumulation','distribution'])
def test_price_escape_without_strength_does_not_authorize_retest(side):
    sm = locked(side)
    feed(sm,4,113,122,121,600,2,side)
    assert snapshot(sm)['escape']['status'] == 'price_escape_without_qualified_strength'
    feed(sm,5,116,119,117,100,-.5,side)
    v,_ = feed(sm,6,117,122,121,200,0,side)
    assert not v.get('lps' if side == 'accumulation' else 'lpsy',False)


def escaped(side='accumulation'):
    sm = locked(side)
    feed(sm,4,112,122,121,600,2,side,'sos' if side == 'accumulation' else 'sow')
    return sm


@pytest.mark.parametrize('side', ['accumulation','distribution'])
@pytest.mark.parametrize('bad', [None,'loud','wide','far','failed_hold','lost_boundary','new_parent'])
def test_anchored_retest_needs_quieter_pullback_and_later_confirmation(side,bad):
    sm = escaped(side)
    r = bar(5,116.5,119,117.5,700 if bad == 'loud' else 100,-.5,side)
    if bad == 'wide': r = bar(5,115,126,117,100,-.5,side)
    if bad == 'far': r = bar(5,121,123,121,100,-.5,side)
    if bad == 'lost_boundary': r = bar(5,113,119,115,100,-.5,side)
    retest = 'lps' if side == 'accumulation' else 'lpsy'
    v,_ = sm.process_bar(5,r,{retest: True})  # old rolling detector cannot confirm it
    assert not v.get(retest,False)
    event = ('sc' if side == 'accumulation' else 'bc') if bad == 'new_parent' else None
    v,_ = feed(sm,6,115 if bad == 'failed_hold' else 117,123,122,200,0,side,event)
    assert bool(v.get(retest,False)) is (bad is None)
    if bad is None:
        e = snapshot(sm)['retest']
        assert e['status'] == 'confirmed'
        assert e['role'] == 'post_escape'
        assert e['candidate_index'] == 5 and e['confirmed_index'] == 6
        assert e['level'] == (116 if side == 'accumulation' else 284)
        assert e['confidence'] == .8


@pytest.mark.parametrize('side', ['accumulation','distribution'])
def test_inside_range_legacy_retest_is_explicitly_unverified(side):
    sm = locked(side)
    feed(sm,4,106,113,112,600,2,side,'sos' if side == 'accumulation' else 'sow')
    key = 'lps' if side == 'accumulation' else 'lpsy'
    v,_ = feed(sm,5,105,110,109,100,-.5,side,key)
    assert v[key]
    assert snapshot(sm)['local_retest_role'] == 'local_unverified'
    assert snapshot(sm)['escape'] is None


def spring_test(side='accumulation'):
    sm = locked(side)
    feed(sm,4,99,106,104,500,0,side,'spring_a' if side == 'accumulation' else 'ut')
    feed(sm,5,101,104,103,100,-.5,side)
    return sm


@pytest.mark.parametrize('side', ['accumulation','distribution'])
def test_phase_c_justification_needs_original_spring_and_confirmed_test(side):
    sm = spring_test(side)
    assert snapshot(sm)['phase_justification']['status'] != 'confirmed'
    feed(sm,6,102,107,106,200,0,side)
    j = snapshot(sm)['phase_justification']
    assert j['status'] == 'confirmed'
    assert j['role'] == ('spring_test' if side == 'accumulation' else 'upthrust_test')
    assert j['origin_index'] == 4 and j['candidate_index'] == 5
    assert j['confirmed_index'] == 6
    feed(sm,7,100,107,106,side=side)
    assert snapshot(sm)['phase_justification']['status'] == 'cancelled_failed_hold'


def test_missing_original_volume_or_prelock_spring_cannot_justify_phase_c():
    for mode in ('missing_volume','prelock'):
        sm = developing() if mode == 'prelock' else locked()
        i = 3 if mode == 'prelock' else 4
        r = bar(i,99,106,104,500)
        if mode == 'missing_volume': r.pop('volume')
        sm.process_bar(i,r,{'spring_a':True})
        feed(sm,i+1,101,104,103,100)
        feed(sm,i+2,102,107,106,200)
        assert snapshot(sm)['phase_justification']['status'] != 'confirmed'


def test_bare_phase_c_does_not_earn_conditional_sizing_increase():
    from scripts.research import wyckoff_recognition_exam as exam
    probe = exam.SizingProbe(exam.RUNNER,exam.file_hash(exam.RUNNER))
    assert probe.evaluate('long','C_accum',{'wyckoff_phase_boost':{'enabled':True}})['allocated_ratio'] == 1.


def phase_metadata():
    sm = spring_test()
    feed(sm,6,102,107,106,200)
    return dict(wyckoff_phase_dir=sm.get_phase_dir(),wyckoff_parent_id=sm.parent_snapshot()['id'],
                wyckoff_evidence_status='available',wyckoff_available_at='2025-03-01T07:00:00+00:00',
                wyckoff_structure_evidence=json.dumps(snapshot(sm)))


@pytest.mark.parametrize('bad', [None,'phase','parent','bound','stale','unconfirmed','future','expired',
                                 'unavailable','missing','malformed','no_spring_role'])
def test_phase_guard_requires_current_bound_confirmed_supported_evidence(bad):
    from engine.wyckoff.range_evidence import phase_c_sizing_eligible
    metadata = phase_metadata()
    evidence = json.loads(metadata['wyckoff_structure_evidence'])
    j = evidence['phase_justification']
    if bad == 'phase': metadata['wyckoff_phase_dir'] = 'C_distrib'
    if bad == 'parent': metadata['wyckoff_parent_id'] += 1
    if bad == 'bound': j['bound_id'] = 'other'
    if bad == 'stale': metadata['wyckoff_available_at'] = '2025-03-01T08:00:00+00:00'
    if bad == 'unconfirmed': j['status'] = 'pending'
    if bad == 'future': j['confirmed_at'] = '2025-03-01T08:00:00+00:00'
    if bad == 'expired': j['expires_at'] = '2025-03-01T06:00:00+00:00'
    if bad == 'unavailable': metadata['wyckoff_evidence_status'] = 'unavailable'
    if bad == 'no_spring_role': j['role'] = 'no_spring_higher_low'
    metadata['wyckoff_structure_evidence'] = json.dumps(evidence)
    if bad == 'missing': metadata.pop('wyckoff_structure_evidence')
    if bad == 'malformed': metadata['wyckoff_structure_evidence'] = 'not-json'
    assert phase_c_sizing_eligible(metadata) is (bad is None)


@pytest.mark.parametrize('direction,enabled,want', [('long',True,1.25),('short',True,1.),('long',False,1.)])
def test_actual_sizing_branch_consumes_confirmed_evidence(direction,enabled,want):
    from scripts.research import wyckoff_recognition_exam as exam
    probe = exam.SizingProbe(exam.RUNNER,exam.file_hash(exam.RUNNER))
    result = probe.evaluate(direction,'C_accum',{'wyckoff_phase_boost':{'enabled':enabled}},phase_metadata())
    assert result['allocated_ratio'] == want
    assert result['multiplier'] == want and result['capex_mult'] == want


def test_adapter_exports_confirmed_retest_only_at_availability_and_preserves_raw_flag():
    rows = [bar(0,100,108,102,1000,4),bar(1,104,110,109,400),
            bar(2,108,116,115,300),bar(3,106,113,107,200),
            bar(4,112,122,121,600,2),bar(5,116.5,119,117.5,100,-.5),
            bar(6,117,123,122,200)]
    frame = pd.DataFrame(rows).set_index('timestamp')
    for key in ('sc','bc','ar','as','st','sos','sow','spring_a','spring_b','ut','utad','lps','lpsy'):
        frame[f'wyckoff_{key}'] = False
        frame[f'wyckoff_{key}_confidence'] = 0.
    for i,key in ((0,'sc'),(1,'ar'),(4,'sos'),(5,'lps')):
        frame.loc[frame.index[i],f'wyckoff_{key}'] = True
        frame.loc[frame.index[i],f'wyckoff_{key}_confidence'] = .8
    cfg = {'timeframe':'1h'}
    out = w._apply_state_machine_validation(frame.copy(),cfg)
    assert not out['wyckoff_lps'].iloc[5]
    assert out['wyckoff_lps'].iloc[6]
    assert out['wyckoff_lps_confidence'].iloc[6] == .8
    assert out['wyckoff_lps_raw'].iloc[5] and not out['wyckoff_lps_raw'].iloc[6]
    assert json.loads(out['wyckoff_structure_evidence'].iloc[6])['retest']['confirmed_index'] == 6
    cols = ['wyckoff_lps','wyckoff_lps_confidence','wyckoff_structure_evidence']
    for n in (4,5,6,7):
        prefix = w._apply_state_machine_validation(frame.iloc[:n].copy(),cfg)
        pd.testing.assert_frame_equal(prefix[cols],out.iloc[:n][cols])
    replay = w._apply_state_machine_validation(out.copy(),cfg)
    pd.testing.assert_frame_equal(replay[cols],out[cols])


@pytest.mark.parametrize('side', ['accumulation','distribution'])
def test_raw_candles_mature_range_and_confirm_anchored_retest_without_rolling_flag(side):
    raw = pd.DataFrame(dict(open=120.,high=121.,low=119.,close=120.,volume=1000.),
                       index=pd.date_range(START,periods=75,freq='1h'))
    values = {
        50:(120,121,100,103,10000),51:(105,110,104,109,400),
        52:(109,116,108,115,300),53:(112,113,106,107,200),
        54:(102,104,101,103,100),55:(103,107,102,106,200),
        56:(106,111,105,110,300),57:(110,112,109,111,300),
        58:(112,125,111,124,9000),59:(120,120,116.5,118,150),
        60:(118,124,117,123,200),
    }
    for i,v in values.items(): raw.iloc[i] = v
    raw.iloc[61:] = [123,124,122,123,1000]
    if side == 'distribution':
        orig = raw.copy()
        for dest,src in (('open','open'),('close','close'),('high','low'),('low','high')):
            raw[dest] = 400-orig[src]
    cfg = {'timeframe':'1h','shadow_v2_enabled':False}
    out = w.detect_all_wyckoff_events(raw.copy(),cfg)
    key = 'lps' if side == 'accumulation' else 'lpsy'
    strength = 'sos' if side == 'accumulation' else 'sow'
    assert json.loads(out.wyckoff_structure_evidence.iloc[52])['range']['status'] == 'developing'
    locked_range = json.loads(out.wyckoff_structure_evidence.iloc[53])['range']
    assert locked_range['upper' if side == 'accumulation' else 'lower'] == (116 if side == 'accumulation' else 284)
    assert json.loads(out.wyckoff_structure_evidence.iloc[55])['range']['status'] == 'tested'
    assert out[f'wyckoff_{strength}'].iloc[58]
    assert not out[f'wyckoff_{key}_raw'].iloc[59:61].any()
    assert not out[f'wyckoff_{key}'].iloc[59] and out[f'wyckoff_{key}'].iloc[60]
    assert out[f'wyckoff_{key}_confidence'].iloc[60] == out[f'wyckoff_{strength}_confidence'].iloc[58]
    assert json.loads(out.wyckoff_structure_evidence.iloc[60])['retest']['candidate_index'] == 59
    for n in (52,53,54,56,59,60,61):
        prefix = w.detect_all_wyckoff_events(raw.iloc[:n].copy(),cfg)
        pd.testing.assert_frame_equal(prefix,out.iloc[:n])
    polluted = raw.copy()
    for key in w._ACCUM_EVENTS+w._DISTRIB_EVENTS:
        polluted[f'wyckoff_{key}_raw'] = True
        polluted[f'wyckoff_{key}_raw_confidence'] = .99
    pd.testing.assert_frame_equal(w.detect_all_wyckoff_events(polluted,cfg).sort_index(axis=1),
                                  out.sort_index(axis=1))


@pytest.mark.parametrize('side', ['accumulation','distribution'])
def test_missing_clock_after_escape_cannot_confirm_or_reenable_local_retest(side):
    sm = escaped(side)
    key = 'lps' if side == 'accumulation' else 'lpsy'
    r = bar(5,116.5,119,117.5,100,side=side)
    r.pop('timestamp')
    sm.process_bar(5,r,{key:True})
    v,_ = feed(sm,6,117,123,122,200,side=side,event=key)
    assert not v[key]
    assert snapshot(sm)['range']['status'] == 'unavailable'


@pytest.mark.parametrize('side', ['accumulation','distribution'])
def test_clock_gap_also_blocks_legacy_retest_before_tracker_updates(side):
    sm = locked(side)
    feed(sm,4,106,113,112,600,2,side,'sos' if side == 'accumulation' else 'sow')
    key = 'lps' if side == 'accumulation' else 'lpsy'
    r = bar(6,105,110,109,100,-.5,side)
    v,_ = sm.process_bar(5,r,{key:True})
    assert not v[key]
    assert sm.get_phase() != 'D'
    assert snapshot(sm)['range']['status'] == 'unavailable'


@pytest.mark.parametrize('side', ['accumulation','distribution'])
def test_retest_deadline_is_not_extended_by_repeated_raw_flags(side):
    sm = escaped(side)
    key = 'lps' if side == 'accumulation' else 'lpsy'
    feed(sm,5,116.5,119,117.5,100,side=side,event=key)
    for i in range(6,21):
        feed(sm,i,116.5,119,118,100,side=side,event=key)
    v,_ = feed(sm,21,117,123,122,200,side=side,event=key)
    assert not v[key]
    assert snapshot(sm)['escape']['status'] == 'expired'
    assert snapshot(sm)['retest']['candidate_index'] == 5


def test_confirmed_phase_c_expires_and_does_not_renew_from_legacy_flags():
    sm = spring_test()
    feed(sm,6,102,107,106,200)
    for i in range(7,22): feed(sm,i,102,107,106,200,event='lps')
    assert snapshot(sm)['phase_justification']['status'] == 'expired'


def test_original_parent_replacement_clears_phase_and_range_authority():
    sm = spring_test()
    feed(sm,6,102,107,106,200)
    before = snapshot(sm)['parent_id']
    feed(sm,7,101,108,104,1000,4,event='sc')
    s = snapshot(sm)
    assert s['parent_id'] > before and s['range']['bound_id'] is None
    assert s['phase_justification']['status'] == 'unsupported'


@pytest.mark.parametrize('side', ['accumulation','distribution'])
def test_delayed_recovery_cannot_restore_spring_breached_before_confirmation(side):
    sm = locked(side)
    feed(sm,4,99,106,104,500,side=side)
    feed(sm,5,98,105,103,500,side=side)
    feed(sm,6,101,106,105,300,side=side)
    key = 'spring_a' if side == 'accumulation' else 'ut'
    p = sm.parent_snapshot()
    item = w.DelayedEventEvidence(key,4,7,99 if side == 'accumulation' else 301,
                                  100 if side == 'accumulation' else 300,
                                  p['id'],p['context'],p['status'],START+pd.Timedelta(hours=4),
                                  START+pd.Timedelta(hours=7),START+pd.Timedelta(hours=8))
    v,_ = sm.process_bar(7,bar(7,101,107,106,200,side=side),{key:True},{key:item})
    assert v[key], 'legacy observation preserved, but it is not phase-sizing authority'
    feed(sm,8,101,104,103,100,side=side)
    feed(sm,9,102,107,106,200,side=side)
    assert snapshot(sm)['phase_justification']['status'] != 'confirmed'


def test_runner_metadata_replaces_stale_phase_and_forwards_current_justification():
    import ast
    from pathlib import Path
    from types import SimpleNamespace
    from engine.wyckoff.range_evidence import phase_c_sizing_eligible
    tree = ast.parse(Path('bin/live/v11_shadow_runner.py').read_text())
    loops = [n for n in ast.walk(tree) if isinstance(n,ast.For)
             and isinstance(n.target,ast.Name) and n.target.id == 's'
             and isinstance(n.iter,ast.Name) and n.iter.id == 'signals'
             and any(isinstance(x,ast.Constant) and x.value == 'wyckoff_phase_dir' for x in ast.walk(n))]
    assert len(loops) == 1
    code = compile(ast.fix_missing_locations(ast.Module(body=loops,type_ignores=[])),'metadata-probe','exec')
    for missing in (False,True):
        features = phase_metadata()
        if missing: features.pop('wyckoff_phase_dir')
        signal = SimpleNamespace(metadata={'wyckoff_phase_dir':'C_accum','wyckoff_structure_evidence':'old'})
        scope = dict(features=features,signals=[signal],self=SimpleNamespace(
            last_dd_score=0.,last_risk_temp=0.,last_trend_align=0.))
        exec(code,scope,scope)
        assert phase_c_sizing_eligible(signal.metadata) is not missing


@pytest.mark.parametrize('breached', [False,True])
def test_actual_delayed_spring_detector_cannot_hide_intermediate_breach(breached):
    """Reviewer witness: native spring detector and adapter, pretagged SC/AR only."""
    from engine.wyckoff.range_evidence import phase_c_sizing_eligible
    rows = [bar(i,103,108,105) for i in range(30)]
    for i,lo,hi,cl,vol,z in (
        (20,100,108,102,1000,4),(21,104,110,109,400,0),
        (22,108,116,115,300,0),(23,106,113,107,200,0),
        (24,97.9,106,104,500,2),(25,97 if breached else 99,105,103,500,0),
        (26,101,106,105,300,0),(27,101,107,106,200,0),
        (28,101,104,103,100,0),(29,102,107,106,200,0),
    ): rows[i] = bar(i,lo,hi,cl,vol,z)
    frame = pd.DataFrame(rows).set_index('timestamp')
    for key in w._ACCUM_EVENTS+w._DISTRIB_EVENTS:
        frame[f'wyckoff_{key}'] = False
        frame[f'wyckoff_{key}_confidence'] = 0.
    for i,key in ((20,'sc'),(21,'ar')):
        frame.loc[frame.index[i],f'wyckoff_{key}'] = True
        frame.loc[frame.index[i],f'wyckoff_{key}_confidence'] = .8
    frame['wyckoff_spring_a'],frame['wyckoff_spring_a_confidence'] = w.detect_spring_type_a(frame,{})
    assert frame.wyckoff_spring_a.iloc[27]
    out = w._apply_state_machine_validation(frame.copy(),{'timeframe':'1h'})
    assert out.wyckoff_spring_a.iloc[27]
    features = out.iloc[-1].to_dict()
    features.update(wyckoff_available_at=rows[29]['available_at'],wyckoff_evidence_status='available')
    assert phase_c_sizing_eligible(features) is not breached
    for n in (25,26,28,29,30):
        pd.testing.assert_frame_equal(w._apply_state_machine_validation(frame.iloc[:n].copy(),{'timeframe':'1h'}),
                                      out.iloc[:n])
