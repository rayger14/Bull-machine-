"""Synthetic handshake tests; not market-data or agent-intelligence evaluation."""
from copy import deepcopy
import importlib
from pathlib import Path

import pytest

STAGES = ('inputs', 'parent', 'sequence', 'trade_plan', 'management')


def module():
    assert Path('scripts/research/agent_assessment.py').exists(), 'assessment adapter missing'
    return importlib.import_module('scripts.research.agent_assessment')


def witnesses():
    return [dict(id='e'+str(i), stage=stage, status='pass',
                 available_at='2026-01-01T10:00:00+00:00',
                 source_ref='synthetic-fixture-v1',
                 summary='Synthetic '+stage+' requirement satisfied.')
            for i, stage in enumerate(STAGES)]


def packet(records=None):
    return module().build_packet('case-a', 'liquidity_compression',
                                '2026-01-01T10:00:00+00:00',
                                witnesses() if records is None else records)


def response(p):
    return dict(packet_id=p['packet_id'], conclusion='supported',
                findings=[dict(stage=s, assessment='supported', evidence_ids=['e'+str(i)],
                               explanation='Supplied witness supports this requirement.')
                          for i, s in enumerate(STAGES)])


def pilot_packets():
    m = module()
    a = packet()
    records = witnesses()
    records[1]['status'] = 'reject'
    records[1]['summary'] = 'Synthetic parent formed after the setup; prior-structure requirement failed.'
    b = m.build_packet('case-b', 'minute_sweep_reclaim', a['decision_time'], records)
    c = m.build_packet('case-c', 'liquidity_compression', a['decision_time'], witnesses()[1:])
    return [a, b, c]


def test_supported_response_and_input_nonmutation():
    records = witnesses()
    before = deepcopy(records)
    p = packet(records)
    r = response(p)
    saved = deepcopy((p, r))
    result = module().validate_assessment(p, r)
    assert result == dict(valid=True, errors=[], execution_authorized=False)
    assert records == before and (p, r) == saved


def test_rejection_dominates_unknown():
    p = pilot_packets()[1]
    r = response(p)
    r['findings'][1]['assessment'] = 'contradicted'
    r['conclusion'] = 'contradicted'
    assert module().validate_assessment(p, r)['valid']
    r['conclusion'] = 'supported'
    assert not module().validate_assessment(p, r)['valid']


@pytest.mark.parametrize('future', [False, True])
def test_missing_or_future_evidence_cannot_support(future):
    records = witnesses()
    if future:
        records[0]['available_at'] = '2026-01-01T10:00:01+00:00'
    else:
        records.pop(0)
    p = packet(records)
    assert not any(w['id'] == 'e0' for w in p['witnesses'])
    r = response(p)
    assert not module().validate_assessment(p, r)['valid']
    r['conclusion'] = 'unresolved'
    r['findings'][0].update(assessment='unresolved', evidence_ids=[])
    assert module().validate_assessment(p, r)['valid']


@pytest.mark.parametrize('mutation', ['packet', 'identity', 'citation', 'wrong_stage', 'omit', 'duplicate', 'extra', 'prose'])
def test_reject_invalid_responses_and_tampering(mutation):
    p = packet()
    r = response(p)
    if mutation == 'packet': p['archetype'] = 'changed'
    elif mutation == 'identity': r['packet_id'] = 'wrong'
    elif mutation == 'citation': r['findings'][0]['evidence_ids'] = ['invented']
    elif mutation == 'wrong_stage': r['findings'][0]['evidence_ids'] = ['e1']
    elif mutation == 'omit': r['findings'].pop()
    elif mutation == 'duplicate': r['findings'][0] = deepcopy(r['findings'][1])
    elif mutation == 'extra': r['order'] = 'buy'
    elif mutation == 'prose': r['findings'][0]['explanation'] = ''
    assert not module().validate_assessment(p, r)['valid']


@pytest.mark.parametrize('mutation', ['naive', 'bad_time', 'duplicate', 'status', 'stage', 'extra', 'empty_id'])
def test_reject_malformed_witnesses(mutation):
    w = witnesses()
    if mutation == 'naive': w[0]['available_at'] = '2026-01-01T10:00:00'
    elif mutation == 'bad_time': w[0]['available_at'] = 'bad'
    elif mutation == 'duplicate': w.append(deepcopy(w[0]))
    elif mutation == 'status': w[0]['status'] = True
    elif mutation == 'stage': w[0]['stage'] = 'invented'
    elif mutation == 'extra': w[0]['future_profit'] = 500
    elif mutation == 'empty_id': w[0]['id'] = ''
    with pytest.raises(ValueError): packet(w)


@pytest.mark.parametrize('r', [None, [], {}, {'findings': None}])
def test_malformed_response_fails_closed(r):
    result = module().validate_assessment(packet(), r)
    assert not result['valid'] and not result['execution_authorized']
