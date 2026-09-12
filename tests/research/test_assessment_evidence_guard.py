"""Contract tests for the offline, caller-attested assessment delivery guard."""
from copy import deepcopy
import hashlib
import importlib
import json
from pathlib import Path

import pytest


def module():
    assert Path('scripts/research/assessment_evidence_guard.py').exists(), 'evidence guard missing'
    return importlib.import_module('scripts.research.assessment_evidence_guard')


def packet():
    return {
        'case_id': 'C1',
        'plan': {'indicative_close': 100.0, 'stop': 98.0, 'notional': 1000.0, 'roundtrip_cost': 1.2},
        'evidence': {'parent': {'low': 90.0}, 'bars': [[100.0, 101.0]]},
    }


def records(envelope):
    return [dict(index=chunk['index'], text=chunk['text'], truncated=False)
            for chunk in envelope['chunks']]


def test_build_envelope_canonical_chunks_and_indicative_economics():
    source = packet()
    before = deepcopy(source)
    envelope = module().build_envelope(source, max_chunk_bytes=32)
    canonical = json.dumps(source, sort_keys=True, separators=(',', ':'), ensure_ascii=True)
    economics = envelope['indicative_economics']
    assert envelope['case_id'] == 'C1'
    assert envelope['packet_bytes'] == len(canonical.encode('ascii'))
    assert envelope['packet_sha256'] == hashlib.sha256(canonical.encode('ascii')).hexdigest()
    assert envelope['sections'] == ['case_id', 'evidence', 'plan']
    assert envelope['max_chunk_bytes'] == 32 and envelope['execution_authorized'] is False
    assert ''.join(chunk['text'] for chunk in envelope['chunks']) == canonical
    assert all(chunk['bytes'] == len(chunk['text'].encode('ascii')) <= 32
               for chunk in envelope['chunks'])
    expected_texts = [canonical[start:start + 32] for start in range(0, len(canonical), 32)]
    assert [chunk['index'] for chunk in envelope['chunks']] == list(range(len(expected_texts)))
    assert [chunk['sha256'] for chunk in envelope['chunks']] == [
        hashlib.sha256(text.encode('ascii')).hexdigest() for text in expected_texts
    ]
    assert economics == {
        'basis': 100.0, 'quantity': 10.0, 'initial_risk': 20.0, 'target': 104.0,
        'cost_R': 0.06, 'breakeven': 100.12, 'net_stop': -21.2, 'net_target': 38.8,
        'actual_fill_known': False,
    }
    assert source == before


@pytest.mark.parametrize('mutation', [
    'missing', 'extra', 'swapped', 'duplicate', 'modified', 'truncated_true', 'truncated_null',
])
def test_validate_delivery_rejects_incomplete_or_changed_caller_receipts(mutation):
    source = packet()
    envelope = module().build_envelope(source, max_chunk_bytes=32)
    supplied = records(envelope)
    if mutation == 'missing':
        supplied.pop()
    elif mutation == 'extra':
        supplied.append(dict(index=99, text='extra', truncated=False))
    elif mutation == 'swapped':
        supplied[0], supplied[1] = supplied[1], supplied[0]
    elif mutation == 'duplicate':
        supplied[1] = deepcopy(supplied[0])
    elif mutation == 'modified':
        supplied[0]['text'] = 'changed'
    elif mutation == 'truncated_true':
        supplied[0]['truncated'] = True
    else:
        supplied[0]['truncated'] = None
    result = module().validate_delivery(source, envelope, supplied, case_id='C1')
    assert result['valid'] is False
    assert result['execution_authorized'] is False
    assert result['authority'] == 'caller_attested_transport_only'


@pytest.mark.parametrize('mutation', ['case', 'hash', 'sections', 'economics'])
def test_validate_delivery_rebuilds_the_trusted_packet_envelope(mutation):
    source = packet()
    envelope = module().build_envelope(source, max_chunk_bytes=32)
    altered = deepcopy(envelope)
    if mutation == 'case':
        altered['case_id'] = 'other'
    elif mutation == 'hash':
        altered['packet_sha256'] = '0' * 64
    elif mutation == 'sections':
        altered['sections'] = ['plan']
    else:
        altered['indicative_economics']['target'] = 999.0
    result = module().validate_delivery(source, altered, records(envelope), case_id='C1')
    assert result['valid'] is False and result['execution_authorized'] is False


def test_validate_delivery_accepts_exact_caller_attested_receipts_only():
    source = packet()
    envelope = module().build_envelope(source, max_chunk_bytes=32)
    assert module().validate_delivery(source, envelope, records(envelope), case_id='C1') == {
        'valid': True, 'errors': [], 'execution_authorized': False,
        'authority': 'caller_attested_transport_only',
    }
    assert not module().validate_delivery(source, envelope, records(envelope), case_id='other')['valid']


def test_resolve_evidence_returns_detached_nested_value():
    source = packet()
    value = module().resolve_evidence(source, ['evidence', 'bars', 0])
    assert module().resolve_evidence(source, ['evidence', 'bars', 0, 1]) == 101.0
    value[1] = 0.0
    assert source['evidence']['bars'][0][1] == 101.0


@pytest.mark.parametrize('path', [[], ['missing'], ['evidence', 'missing'],
                                   ['evidence', 'bars', True], ['evidence', 'bars', -1],
                                   ['evidence', 0], ['evidence', 'bars', 0, 'bad']])
def test_resolve_evidence_rejects_invalid_typed_paths(path):
    with pytest.raises(ValueError):
        module().resolve_evidence(packet(), path)


@pytest.mark.parametrize('change', ['case_id', 'bool', 'nonfinite', 'stop', 'cost', 'overflow', 'underflow'])
def test_build_envelope_rejects_invalid_packet_values(change):
    source = packet()
    if change == 'case_id':
        source['case_id'] = ''
    elif change == 'bool':
        source['plan']['notional'] = True
    elif change == 'nonfinite':
        source['plan']['indicative_close'] = float('nan')
    elif change == 'stop':
        source['plan']['stop'] = 100.0
    elif change == 'cost':
        source['plan']['roundtrip_cost'] = -1.0
    elif change == 'overflow':
        source['plan'].update(indicative_close=1e-308, stop=5e-309, notional=1e308)
    else:
        source['plan'].update(indicative_close=1e-200, stop=5e-201, notional=5e-324)
    with pytest.raises(ValueError):
        module().build_envelope(source)


@pytest.mark.parametrize('bound', [0, -1, True, 1.5])
def test_build_envelope_rejects_invalid_chunk_bounds(bound):
    with pytest.raises(ValueError):
        module().build_envelope(packet(), max_chunk_bytes=bound)


def test_build_envelope_rejects_integer_plan_values_that_lose_float_precision():
    source = packet()
    source['plan'].update(indicative_close=9007199254740993, stop=9007199254740991,
                          notional=9007199254740993, roundtrip_cost=0)
    with pytest.raises(ValueError):
        module().build_envelope(source)
