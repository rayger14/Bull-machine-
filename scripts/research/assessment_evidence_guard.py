"""Pure transport receipts for one offline assessment packet; not execution or truth proof."""
from copy import deepcopy
import hashlib
import json
import math


def _canonical(packet):
    try:
        return json.dumps(packet, sort_keys=True, separators=(',', ':'), ensure_ascii=True,
                          allow_nan=False)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError('packet must be a finite JSON object') from exc


def _number(value, name, *, positive=False, nonnegative=False):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(name + ' must be numeric')
    try:
        number = float(value)
    except OverflowError as exc:
        raise ValueError(name + ' must be finite') from exc
    if isinstance(value, int) and int(number) != value:
        raise ValueError(name + ' must be losslessly representable')
    if not math.isfinite(number) or (positive and number <= 0) or (nonnegative and number < 0):
        raise ValueError(name + ' out of range')
    return number


def _economics(packet):
    if not isinstance(packet, dict) or not isinstance(packet.get('case_id'), str) or not packet['case_id'].strip():
        raise ValueError('case_id must be nonempty text')
    plan = packet.get('plan')
    if not isinstance(plan, dict):
        raise ValueError('plan must be an object')
    close = _number(plan.get('indicative_close'), 'indicative_close', positive=True)
    stop = _number(plan.get('stop'), 'stop', positive=True)
    notional = _number(plan.get('notional'), 'notional', positive=True)
    cost = _number(plan.get('roundtrip_cost'), 'roundtrip_cost', nonnegative=True)
    if stop >= close:
        raise ValueError('stop must be below indicative_close')
    quantity = notional / close
    risk = (close - stop) * quantity
    target = close + 2 * (close - stop)
    if not all(math.isfinite(value) for value in (quantity, risk, target)) or risk <= 0:
        raise ValueError('derived economics must be finite')
    cost_r = cost / risk
    breakeven = close * (1 + cost / notional)
    net_stop, net_target = -risk - cost, 2 * risk - cost
    if not all(math.isfinite(value) for value in (cost_r, breakeven, net_stop, net_target)):
        raise ValueError('derived economics must be finite')
    return {'basis': close, 'quantity': quantity, 'initial_risk': risk, 'target': target,
            'cost_R': cost_r, 'breakeven': breakeven, 'net_stop': net_stop,
            'net_target': net_target, 'actual_fill_known': False}


def _bound(max_chunk_bytes):
    if isinstance(max_chunk_bytes, bool) or not isinstance(max_chunk_bytes, int) or max_chunk_bytes <= 0:
        raise ValueError('max_chunk_bytes must be a positive integer')
    return max_chunk_bytes


def build_envelope(packet, max_chunk_bytes=4096):
    """Canonicalize a validated packet into bounded, caller-deliverable text chunks."""
    bound = _bound(max_chunk_bytes)
    economics = _economics(packet)
    text = _canonical(packet)
    encoded = text.encode('ascii')
    chunks = []
    for index, start in enumerate(range(0, len(text), bound)):
        chunk_text = text[start:start + bound]
        chunk_bytes = chunk_text.encode('ascii')
        chunks.append({'index': index, 'text': chunk_text, 'bytes': len(chunk_bytes),
                       'sha256': hashlib.sha256(chunk_bytes).hexdigest()})
    return {'case_id': packet['case_id'], 'packet_sha256': hashlib.sha256(encoded).hexdigest(),
            'packet_bytes': len(encoded), 'sections': sorted(packet), 'max_chunk_bytes': bound,
            'chunks': chunks, 'indicative_economics': economics, 'execution_authorized': False}


def validate_delivery(packet, envelope, records, *, case_id):
    """Validate caller-attested transport records, never actual rendering or evidence entailment."""
    errors = []
    try:
        if not isinstance(case_id, str) or not case_id.strip():
            raise ValueError('case_id must be nonempty text')
        if not isinstance(envelope, dict):
            raise ValueError('envelope must be an object')
        expected = build_envelope(packet, envelope.get('max_chunk_bytes'))
        if case_id != packet.get('case_id') or envelope.get('case_id') != case_id:
            raise ValueError('case_id mismatch')
        if envelope != expected:
            raise ValueError('envelope mismatch')
        if not isinstance(records, list) or len(records) != len(expected['chunks']):
            raise ValueError('receipt count mismatch')
        for chunk, record in zip(expected['chunks'], records):
            if not isinstance(record, dict) or set(record) != {'index', 'text', 'truncated'}:
                raise ValueError('receipt schema mismatch')
            if (type(record['index']) is not int or record['index'] != chunk['index']
                    or not isinstance(record['text'], str) or record['text'] != chunk['text']
                    or record['truncated'] is not False):
                raise ValueError('receipt content mismatch')
    except (ValueError, TypeError, KeyError, AttributeError) as exc:
        errors.append(str(exc))
    return {'valid': not errors, 'errors': errors, 'execution_authorized': False,
            'authority': 'caller_attested_transport_only'}


def resolve_evidence(packet, path):
    """Return a detached value at a typed packet path; existence is not entailment."""
    if not isinstance(packet, dict) or not isinstance(path, list) or not path:
        raise ValueError('packet and nonempty path required')
    if not isinstance(path[0], str):
        raise ValueError('path must start with a section name')
    value = packet
    for part in path:
        if isinstance(value, dict):
            if not isinstance(part, str) or part not in value:
                raise ValueError('missing dictionary path')
            value = value[part]
        elif isinstance(value, list):
            if type(part) is not int or part < 0 or part >= len(value):
                raise ValueError('invalid list path')
            value = value[part]
        else:
            raise ValueError('path reaches a scalar')
    return deepcopy(value)
