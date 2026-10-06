"""Harness safety and observation; no frozen exam candles run in this suite."""
import copy
import hashlib
import importlib.util
from datetime import datetime, timedelta, timezone

import pytest

from scripts.research.wyckoff_recognition_cases import build_packet, digest, write_packet


def api():
    name = 'scripts.research.wyckoff_recognition_exam'
    assert importlib.util.find_spec(name) is not None, 'recognition runner API not built'
    return __import__(name, fromlist=['trace_prefix'])


def flat_rows(n=600):
    start = datetime(2024,1,1,tzinfo=timezone.utc)
    return [[(start+timedelta(hours=i)).isoformat(),100.,101.,99.,100.,100.+i%3] for i in range(n)]


def review(p):
    return {'schema':'wyckoff-recognition-review-v1','packet_sha256':digest(p),
            'reviewer':'unit_test_reviewer','independent_source_only':True,'approved':True,
            'cases':[{'id':c['id'],'status':'approved','interpretation':'Test only',
                      'limitations':'Not an actual independent approval'} for c in p['cases']]}


def test_real_flat_trace_is_observational_and_directional():
    m = api()
    result = m.trace_prefix(flat_rows(), 600)
    assert set(result['timeframes']) == {'1h','4h','1d'}
    assert len(result['timeframes']['1h']['rows']) == 600
    assert result['features']['wyckoff_evidence_status'] == 'available'
    assert result['features']['tf1d_daily_bars'] == 25
    assert result['scores'] == {'long':0.0,'short':0.0}
    assert result['timeframes']['1h']['cfg']['sm_m2_context_only'] is True
    assert all(not r['validated'] for r in result['timeframes']['1h']['rows'])
    assert result['timeframes']['1h']['rows'][-1]['structure']['range']['bound_id'] is None
    assert result['timeframes']['1h']['rows'][-1]['structure']['phase_justification']['status'] == 'unsupported'
    assert m.trace_prefix(flat_rows(), 600, observe=False)['features'] == result['features']


def test_sampled_prefix_does_not_see_future_rows():
    m = api()
    rows = flat_rows(610)
    a = m.trace_prefix(rows,600)
    for row in rows[600:]:
        row[1:5] = [150.,180.,120.,170.]
        row[5] = 999999.
    assert m.trace_prefix(rows,600) == a


@pytest.mark.parametrize('direction,phase,enabled,want',[
    ('long','C_accum',True,1.0),('short','C_accum',True,1.0),
    ('long','C_distrib',True,1.0),('long','D_accum',True,1.0),
    ('long',None,True,1.0),('long','C_accum',False,1.0)])
def test_exact_sizing_branch_effect(direction,phase,enabled,want):
    m = api()
    probe = m.SizingProbe(m.RUNNER, m.file_hash(m.RUNNER))
    answer = probe.evaluate(direction,phase,{'wyckoff_phase_boost':{'enabled':enabled}})
    assert answer['allocated_ratio'] == want
    assert answer['multiplier'] == want
    assert answer['capex_mult'] == want
    assert answer['conditional_only'] is True


def test_sizing_extraction_rejects_drift_or_missing_unique_branch(tmp_path):
    m = api()
    with pytest.raises(ValueError,match='hash'):
        m.SizingProbe(m.RUNNER,'0'*64)
    empty = tmp_path/'empty.py'
    empty.write_text('class V11ShadowRunner:\n def process_bar(self):\n  pass\n')
    with pytest.raises(ValueError,match='unique'):
        m.SizingProbe(empty,m.file_hash(empty))


@pytest.mark.parametrize('mutation',['unapproved','wrong_hash','not_independent','missing_case','duplicate','bad_status'])
def test_review_preflight_rejects_invalid_annotations(mutation):
    m = api()
    p = build_packet()
    r = review(p)
    if mutation == 'unapproved': r['approved'] = False
    if mutation == 'wrong_hash': r['packet_sha256'] = '0'*64
    if mutation == 'not_independent': r['independent_source_only'] = False
    if mutation == 'missing_case': r['cases'].pop()
    if mutation == 'duplicate': r['cases'][-1]['id'] = 'W01'
    if mutation == 'bad_status': r['cases'][0]['status'] = 'pass'
    with pytest.raises(ValueError):
        m.validate_review(p,r)


def test_semantic_summary_does_not_count_config_disabled_as_recognition():
    m = api()
    c = build_packet()['cases'][1]
    empty = {'features':{'wyckoff_phase_dir':'neutral'},'scores':{'long':0.,'short':0.},
             'timeframes':{'1h':{'cfg':{'sm_m2_context_only':True},'rows':[]}},
             'sizing':{'long':{'allocated_ratio':1.0},'short':{'allocated_ratio':1.0}}}
    s = m.summarize_case(c,[empty],{'status':'approved','interpretation':'No-spring positive'})
    assert s['missing_milestones'] == ['sos','lps']
    assert s['recognition_status'] == 'missed_positive'
    assert s['full_m2_enabled'] is False
    assert s['economic_progression'] == 'blocked'
    unresolved = m.summarize_case(c,[empty],{'status':'unresolved','interpretation':'Insufficient source evidence'})
    assert unresolved['recognition_status'] == 'unscored'


def test_seal_detects_source_and_review_drift(tmp_path):
    m = api()
    p = build_packet()
    r = review(p)
    seal = m.make_seal(p,r)
    m.verify_seal(p,r,seal)
    r['cases'][0]['interpretation'] = 'changed'
    with pytest.raises(ValueError,match='review'):
        m.verify_seal(p,r,seal)
    r = review(p)
    seal['files']['../../outside.py'] = '0'*64
    with pytest.raises(ValueError,match='binding'):
        m.verify_seal(p,r,seal)


def test_output_immutable_and_budget_fail_closed(tmp_path):
    m = api()
    with m.BoundedOutput(tmp_path/'small',seconds=1,max_bytes=20) as out:
        with pytest.raises(ValueError,match='budget'):
            out.write('large.json',{'x':'x'*30})
        with pytest.raises(ValueError,match='path'):
            out.write('../escape',{})
    with pytest.raises(FileExistsError):
        with m.BoundedOutput(tmp_path/'small',seconds=1,max_bytes=20): pass


def fabricated_trace(n,rows=(),phase='neutral',long_score=0.,ratio=1.):
    return {'n':n,'features':{'wyckoff_phase_dir':phase},
            'scores':{'long':long_score,'short':0.},
            'timeframes':{'1h':{'cfg':{},'rows':list(rows)}},
            'sizing':{'long':{'allocated_ratio':ratio},'short':{'allocated_ratio':1.}}}


def test_earlier_unrelated_labels_cannot_cover_later_positive_milestones():
    m = api()
    c = build_packet()['cases'][0]
    r = {'i':601,'validated':['spring_a','sos','lps'],'raw':['spring_a','sos','lps']}
    s = m.summarize_case(c,[fabricated_trace(666,[r])],{'status':'approved','interpretation':'source setup'})
    assert s['missing_milestones'] == ['spring','sos','lps']


def test_broken_parent_does_not_invalidate_earlier_candidate_retroactively():
    m = api()
    c = build_packet()['cases'][7]
    traces = [fabricated_trace(636,phase='C_accum'),fabricated_trace(642),fabricated_trace(649)]
    s = m.summarize_case(c,traces,{'status':'approved','interpretation':'break after earlier range'})
    assert not s['contradictions']
    assert not s['phase_claims_requiring_review']


def test_phase_alone_is_review_flag_not_proven_semantic_contradiction():
    m = api()
    c = build_packet()['cases'][4]
    s = m.summarize_case(c,[fabricated_trace(645,phase='C_accum')],
                         {'status':'approved','interpretation':'no range'})
    assert not s['contradictions']
    assert s['phase_claims_requiring_review'] == [{'n':645,'phase':'C_accum'}]


def test_positive_label_without_parent_link_is_not_recognition_pass():
    m = api()
    c = build_packet()['cases'][1]
    rows = [{'i':652,'validated':['sos'],'raw':['sos']},
            {'i':660,'validated':['lps'],'raw':['lps']}]
    s = m.summarize_case(c,[fabricated_trace(666,rows)],{'status':'approved','interpretation':'no spring'})
    assert not s['missing_milestones']
    assert s['parent_association_gaps'] == ['sos','lps']
    assert s['recognition_status'] != 'pass'
    assert s['economic_progression'] == 'blocked'


def test_neutral_phase_cannot_hide_retained_post_failure_directional_evidence():
    m = api()
    c = build_packet()['cases'][6]
    s = m.summarize_case(c,[fabricated_trace(645),fabricated_trace(649,long_score=.6)],
                         {'status':'approved','interpretation':'failed test'})
    assert s['unattributed_post_failure_scores'] == [{'n':649,'long':.6,'short':0.}]
    assert 'score_parent_lineage' in s['unrepresented_distinctions']
    assert s['economic_progression'] == 'blocked'


def test_preflight_rejects_review_before_trace_or_success_output(tmp_path,monkeypatch):
    import json
    m = api()
    p = build_packet()
    packet_path = tmp_path/'packet.json'
    write_packet(packet_path,p)
    review_path = tmp_path/'review.json'
    bad = review(p)
    bad['approved'] = False
    review_path.write_text(json.dumps(bad))
    seal_path = tmp_path/'seal.json'
    seal_path.write_text('{}')
    monkeypatch.setattr(m,'trace_prefix',lambda *_: pytest.fail('must reject before detection'))
    with pytest.raises(ValueError):
        m.run_exam(packet_path,review_path,seal_path,tmp_path/'out')
    assert json.loads((tmp_path/'out/failure.json').read_text())['status'] == 'failed'
    assert not (tmp_path/'out/receipt.json').exists()


def test_deadline_is_not_caught_as_detector_fallback(tmp_path):
    import signal
    m = api()
    with pytest.raises(m.DeadlineExceeded):
        with m.BoundedOutput(tmp_path/'deadline',seconds=1):
            try:
                signal.raise_signal(signal.SIGALRM)
            except Exception:
                pytest.fail('deadline was swallowed')
