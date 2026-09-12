"""Reject incomplete/misbound actual inner returns without certifying attention."""
from copy import deepcopy
import importlib

import pytest


def fixture():
    guard = importlib.import_module('scripts.research.assessment_evidence_guard')
    packet = {'case_id': 'C2', 'plan': {'indicative_close': 100., 'stop': 98.,
              'notional': 1000., 'roundtrip_cost': 1.2}, 'evidence': 'x' * 220}
    envelope = guard.build_envelope(packet, max_chunk_bytes=256)
    assert len(envelope['chunks']) == 2
    captures = [dict(role='assessor', case_id='C2', packet_sha256=envelope['packet_sha256'],
                chunk_index=c['index'], exec_result=dict(output=c['text'], exit_code=0,
                chunk_id='runtime-' + str(c['index']), wall_time_seconds=.1))
                for c in envelope['chunks']]
    return guard, packet, envelope, captures


def check(guard, packet, envelope, captures, **bindings):
    assert hasattr(guard, 'validate_runtime_returns'), 'runtime-return validation missing'
    return guard.validate_runtime_returns(packet, envelope, captures,
        **dict(dict(case_id='C2', role='assessor'), **bindings))


def test_exact_two_returns_pass_without_mutation_or_overclaim():
    guard, packet, envelope, captures = fixture()
    before = deepcopy((packet, envelope, captures))
    assert check(guard, packet, envelope, captures) == {
        'valid': True, 'errors': [], 'authority': 'captured_inner_runtime_returns_only',
        'execution_authorized': False, 'outer_renderer_unverified': True,
        'model_attention_unverified': True}
    assert (packet, envelope, captures) == before


@pytest.mark.parametrize('mutation', [
    'drop', 'extra', 'swap', 'duplicate', 'text', 'newline', 'role', 'case', 'hash',
    'index', 'bool_index', 'capture_object', 'exec_object', 'output_missing',
    'exit_missing', 'exit_failed', 'exit_bool', 'exit_float', 'session',
    'captures_object', 'envelope_hash', 'envelope_object', 'capture_extra',
    'stale_packet', 'envelope_chunk', 'envelope_index', 'missing_capture_key', 'output_type',
])
def test_invalid_runtime_returns_fail_closed(mutation):
    guard, packet, envelope, captures = fixture()
    c = captures[0]
    if mutation == 'drop': captures.pop()
    elif mutation == 'extra': captures.append(deepcopy(c))
    elif mutation == 'swap': captures.reverse()
    elif mutation == 'duplicate': captures[1] = deepcopy(c)
    elif mutation == 'text': c['exec_result']['output'] = 'wrong'
    elif mutation == 'newline': c['exec_result']['output'] += '\n'
    elif mutation == 'role': c['role'] = 'reviewer'
    elif mutation == 'case': c['case_id'] = 'C3'
    elif mutation == 'hash': c['packet_sha256'] = 'wrong'
    elif mutation == 'index': c['chunk_index'] = 1
    elif mutation == 'bool_index': c['chunk_index'] = False
    elif mutation == 'capture_object': captures[0] = None
    elif mutation == 'exec_object': c['exec_result'] = []
    elif mutation == 'output_missing': del c['exec_result']['output']
    elif mutation == 'exit_missing': del c['exec_result']['exit_code']
    elif mutation == 'exit_failed': c['exec_result']['exit_code'] = 1
    elif mutation == 'exit_bool': c['exec_result']['exit_code'] = False
    elif mutation == 'exit_float': c['exec_result']['exit_code'] = 0.0
    elif mutation == 'session': c['exec_result']['session_id'] = 23
    elif mutation == 'captures_object': captures = {}
    elif mutation == 'envelope_hash': envelope['packet_sha256'] = 'wrong'
    elif mutation == 'envelope_object': envelope = None
    elif mutation == 'stale_packet': packet['evidence'] = 'changed'
    elif mutation == 'envelope_chunk': envelope['chunks'][0]['text'] = 'changed'
    elif mutation == 'envelope_index': envelope['chunks'][0]['index'] = 9
    elif mutation == 'missing_capture_key': del c['role']
    elif mutation == 'output_type': c['exec_result']['output'] = None
    else: c['unexpected'] = True
    result = check(guard, packet, envelope, captures)
    assert result['valid'] is False and result['errors']
    assert result['execution_authorized'] is False


@pytest.mark.parametrize('bindings', [{'case_id': ''}, {'role': ''}, {'role': None}, {'case_id': False}])
def test_empty_or_wrong_type_expected_bindings_fail(bindings):
    guard, packet, envelope, captures = fixture()
    assert check(guard, packet, envelope, captures, **bindings)['valid'] is False


def test_completed_return_allows_null_session_and_extra_runtime_metadata():
    guard, packet, envelope, captures = fixture()
    for c in captures:
        c['exec_result'].update(session_id=None, original_token_count=64)
    assert check(guard, packet, envelope, captures)['valid'] is True
