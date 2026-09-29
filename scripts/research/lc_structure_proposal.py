"""Strict structural proposal checks, not semantic truth or trading approval."""
from copy import deepcopy
import json

from scripts.research.conditional_assessment import digest
from scripts.research.lc_structure_packet import (
    RESPONSE_KEYS, PLAN_KEYS, POLICY_KEYS, DECISIONS, THESES,
    validate_structure_packet, clock, number, resolve,
)


VERSION = 'lc_structure_proposal_v1'
CLAIM_KEYS = {'text', 'evidence_ids'}


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError('duplicate JSON key')
        result[key] = value
    return result


def _reject_constant(value):
    raise ValueError('non-finite JSON constant: ' + value)


def _ids(value, catalog, nonempty=True):
    return (isinstance(value, list) and (bool(value) or not nonempty)
            and all(isinstance(x, str) and x in catalog for x in value)
            and len(value) == len(set(value)))


def _claim(value, packet, sequence=False):
    keys = CLAIM_KEYS | ({'observation_end'} if sequence else set())
    return (isinstance(value, dict) and set(value) == keys
            and isinstance(value['text'], str) and bool(value['text'].strip())
            and _ids(value['evidence_ids'], packet['citation_catalog']))


def _observed_time(packet, eid):
    # Groups and free-text teachings cannot certify a dated observation.
    path = packet['citation_catalog'][eid]
    if path[0] not in ('levels', 'candles'):
        raise ValueError('confirmation requires a timed price observation')
    value = resolve(packet, eid)
    end, available = clock(value['observation_end']), clock(value['available_at'])
    if end > available or available > clock(packet['decision_time']):
        raise ValueError('future observation')
    return end


def _policy_errors(policy, packet):
    if policy is None:
        return []
    if not isinstance(policy, dict) or set(policy) != POLICY_KEYS:
        return ['policy_shape']
    if (policy['version'] != 'lc_structure_policy_v1'
            or policy['instrument'] != packet['instrument']
            or policy['data_stream_id'] != packet['data_stream_id']):
        return ['policy_identity']
    errors = []
    zero = {'minimum_net_rr', 'roundtrip_cost_bps', 'processing_seconds', 'routing_seconds'}
    numeric = POLICY_KEYS - {'version', 'instrument', 'data_stream_id'}
    for key in sorted(numeric):
        if not number(policy[key], zero=key in zero):
            errors.append('policy_number:' + key)
    for key in ('entry_expiry_minutes', 'horizon_minutes'):
        if type(policy[key]) is not int:
            errors.append('policy_duration:' + key)
    return errors


def obstacle_ids(packet, entry, target):
    """Every catalogued intervening level, without equating it to real resistance."""
    return sorted(k for k,v in packet['levels'].items() if entry < v['price'] < target)


def _plan_errors(answer, packet, policy):
    plan = answer['plan']; decision = answer['decision']
    if decision not in ('enter_proposal', 'wait_proposal'):
        return [] if plan is None else ['nonentry_plan']
    if not isinstance(plan, dict) or set(plan) != PLAN_KEYS:
        return ['plan_shape']
    errors = []
    if answer['thesis'] == 'unresolved':
        errors.append('unresolved_entry_thesis')
    if not answer['supporting'] or not answer['opposing'] or not answer['sequence']:
        errors.append('entry_evidence')
    ctx = packet['context']
    if not (ctx['current_validated'] is True and ctx['native_long'] is True
            and ctx['hourly']['status'] == 'known'):
        errors.append('mandatory_facts_unknown')
    levels = packet['levels']
    for field in ('stop_level_id', 'invalidation_level_id', 'destination_level_id'):
        if not isinstance(plan[field], str) or plan[field] not in levels:
            errors.append('level_reference:' + field)
    if errors:
        return errors
    stop, invalidation, target = (levels[plan[x]]['price'] for x in
        ('stop_level_id', 'invalidation_level_id', 'destination_level_id'))
    close = ctx['current']['close']
    if not (number(close) and stop < close < target and invalidation < close):
        errors.append('long_geometry')
    if plan['invalidation_operator'] != 'touch_or_below':
        errors.append('invalidation_operator')
    obstacles = plan['obstacle_level_ids']
    if not _ids(obstacles, levels, False) or sorted(obstacles) != obstacle_ids(packet, close, target):
        errors.append('obstacle_membership')
    for field in ('horizon_minutes', 'expiry_minutes'):
        if type(plan[field]) is not int or plan[field] <= 0:
            errors.append('plan_duration:' + field)
    if policy is not None:
        if (plan['horizon_minutes'] != policy['horizon_minutes']
                or plan['expiry_minutes'] != policy['entry_expiry_minutes']):
            errors.append('policy_duration_binding')
    trigger = plan['trigger']
    if not isinstance(trigger, dict):
        return errors + ['trigger_shape']
    expected = 'immediate' if decision == 'enter_proposal' else 'close_above'
    keys = {'kind'} if expected == 'immediate' else {'kind', 'level_id'}
    if set(trigger) != keys or trigger.get('kind') != expected:
        return errors + ['unsupported_trigger']
    if expected == 'close_above':
        identity = trigger['level_id']
        if not isinstance(identity, str) or identity not in levels:
            errors.append('trigger_reference')
        elif not max(stop, invalidation) < levels[identity]['price'] < target:
            errors.append('trigger_geometry')
    confirmation = plan['confirmation_evidence_ids']
    if not _ids(confirmation, packet['citation_catalog'], nonempty=expected=='immediate'):
        errors.append('confirmation_evidence')
    else:
        try:
            for eid in confirmation:
                _observed_time(packet, eid)
        except (KeyError, ValueError, TypeError):
            errors.append('confirmation_time')
    return errors


def validate_structure_proposal(source_request: dict, packet: dict,
                                raw_response: str, policy) -> dict:
    """Source failures raise; malformed answers are invalid, never valid rejects.

    Missing external policy permits diagnostic interpretation only. A successful
    validation establishes schema/citation consistency, not the truth of prose.
    """
    validate_structure_packet(source_request, packet)
    policy_errors = _policy_errors(policy, packet)
    policy_hash = digest(policy) if policy is not None and not policy_errors else None

    def result(status, errors, proposal=None):
        return dict(status=status, errors=sorted(set(errors)), proposal=deepcopy(proposal),
                    policy_sha256=policy_hash, execution_authorized=False)

    if policy_errors:
        return result('invalid', policy_errors)
    try:
        if not isinstance(raw_response, str):
            raise ValueError('raw JSON text required')
        answer = json.loads(raw_response, object_pairs_hook=_unique_object,
                            parse_constant=_reject_constant)
        if not isinstance(answer, dict) or set(answer) != RESPONSE_KEYS:
            return result('invalid', ['response_shape'])
        errors = []
        for key, expected in (('version', VERSION), ('case_id', packet['case_id']),
            ('packet_sha256', packet['seal']), ('contract_sha256', packet['contract_sha256']),
            ('curriculum_sha256', packet['curriculum_sha256']), ('policy_sha256', policy_hash)):
            if answer[key] != expected:
                errors.append('policy_binding' if key=='policy_sha256' else 'binding:' + key)
        if answer['execution_authorized'] is not False:
            errors.append('execution_authority')
        if answer['decision'] not in DECISIONS or answer['thesis'] not in THESES:
            errors.append('decision_or_thesis')
        for field in ('parent_child', 'competing_explanation'):
            if not _claim(answer[field], packet):
                errors.append('claim:' + field)
        for field in ('supporting', 'opposing', 'unknowns', 'sequence'):
            claims = answer[field]
            if not isinstance(claims, list) or not all(_claim(c, packet, field=='sequence') for c in claims):
                errors.append('claims:' + field)
        if errors:
            return result('invalid', errors)
        previous = None
        for item in answer['sequence']:
            when = clock(item['observation_end'])
            if previous is not None and when < previous:
                errors.append('sequence_order')
            previous = when
            if any(_observed_time(packet, eid) != when for eid in item['evidence_ids']):
                errors.append('sequence_time')
        decision = answer['decision']
        if decision == 'reject' and not answer['opposing']:
            errors.append('rejection_explanation')
        if decision == 'insufficient_evidence' and not answer['unknowns']:
            errors.append('missing_unknowns')
        if (packet['context']['hourly']['status'] != 'known'
                or packet['context']['native_long'] is not True
                or packet['context']['current_validated'] is not True):
            if decision != 'insufficient_evidence':
                errors.append('mandatory_facts_unknown')
        errors.extend(_plan_errors(answer, packet, policy))
        if errors:
            return result('invalid', errors)
        status = {'reject':'valid_reject', 'insufficient_evidence':'insufficient_evidence'}.get(
            decision, 'valid_proposal')
        return result(status, [], answer)
    except (ValueError, TypeError, KeyError, IndexError, OverflowError, RecursionError):
        return result('invalid', ['malformed_response'])
