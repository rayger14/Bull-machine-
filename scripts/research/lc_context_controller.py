"""A single contextual entry policy; evidence roles are not weighted votes."""
from copy import deepcopy

from scripts.research.lc_context_contract import SUBTYPES, positive, seal, verify_case


def entry_location_ok(decision, opening):
    if not positive(opening):
        return False
    if decision['scenario'] == 'accepted_expansion':
        return opening > decision['parent_high']
    if decision['scenario'] in ('local_range_expansion', 'range_floor_rebound'):
        return decision['parent_low'] < opening < decision['parent_high']
    return False


def classify(case):
    verify_case(case)
    parent, subtype = case['parent_4h'], case['subtype']
    bound = parent['bound']
    known = parent['status'] == 'known' and parent['state'] != 'unknown'
    low, high = (bound['range_low'], bound['range_high']) if bound else (None, None)
    current, prior = case['current'], case['prior']
    geometry = case['source_status'] == 'known'
    values = {
        'source_known': geometry, 'risk_known': case['risk_status'] == 'known',
        'resolved_subtype': subtype in SUBTYPES,
        'parent_known': True if known else None if parent['status'] == 'unknown' else False,
        'parent_inside': parent['state'] == 'inside' if known else None,
        'parent_accepted_above': parent['state'] == 'accepted_above' if known else None,
        'parent_accepted_below': parent['state'] == 'accepted_below' if known else None,
        'source_close_above_parent': current['close'] > high if known and geometry else None,
        'source_close_inside_parent': low < current['close'] < high if known and geometry else None,
        'source_close_below_upper': current['close'] < high if known and geometry else None,
        'prior_hour_contained': low <= prior['low'] <= prior['high'] <= high if known and geometry else None,
        'source_swept_floor': current['low'] < low if known and geometry else None,
        'h5_known': case['h5_status'] == 'known',
    }
    predicates = {k: 'unknown' if v is None else 'pass' if v else 'fail' for k, v in values.items()}
    scenario, state, action, trigger, reason = 'outside_playbook', 'outside_playbook', 'none', None, 'unmatched_geometry'
    matches = []
    if known and geometry:
        if subtype == SUBTYPES[0]:
            if values['parent_accepted_above'] and values['source_close_above_parent']:
                matches.append(('accepted_expansion', 'eligible', 'immediate', None, 'accepted_parent_expansion'))
            if values['parent_inside'] and values['source_close_inside_parent'] and values['prior_hour_contained']:
                matches.append(('local_range_expansion', 'awaiting_confirmation', 'wait', case['h5'], 'await_h5_close'))
            if values['source_close_above_parent'] and parent['state'] in ('inside', 'boundary', 'not_established'):
                matches.append(('parent_acceptance_unconfirmed', 'watching', 'none', None, 'no_new_4h_close_before_expiry'))
        elif subtype == SUBTYPES[1]:
            if values['source_swept_floor'] and values['source_close_below_upper'] and values['parent_inside']:
                trigger = max(low, case['h5']) if case['h5'] is not None else None
                matches.append(('range_floor_rebound', 'awaiting_confirmation', 'wait', trigger, 'await_floor_and_h5_reclaim'))
            if values['parent_accepted_below']:
                matches.append(('rebound_invalidated', 'invalidated', 'none', None, 'parent_accepted_below_floor'))
    if len(matches) > 1:
        raise ValueError('conflicting contextual scenarios')
    if matches:
        scenario, state, action, trigger, reason = matches[0]
    if not geometry:
        state, reason = 'insufficient_evidence', 'unknown_common_source'
    elif case['risk_status'] != 'known':
        state, reason = 'insufficient_evidence', 'unknown_risk'
    elif subtype not in SUBTYPES:
        state, reason = 'outside_playbook', 'unresolved_subtype'
    elif parent['status'] == 'absent':
        state, reason = 'outside_playbook', 'no_parent_reference'
    elif not known:
        state, reason = 'insufficient_evidence', 'unknown_parent_context'
    elif action == 'wait' and case['h5_status'] != 'known':
        state, reason = 'insufficient_evidence', 'unknown_h5'
    if state in ('insufficient_evidence', 'outside_playbook'):
        scenario, action, trigger = state, 'none', None
    evidence = [e['id'] for e in parent.get('events', [])]
    daily = case['parent_1d']['state']
    result = {'schema': 'lc-context-decision-v1', 'candidate_id': case['candidate_id'],
              'case_seal': case['seal'], 'policy_seal': case['policy_seal'],
              'decision_time': case['decision_time'], 'subtype': subtype,
              'scenario': scenario, 'state': state, 'action': action,
              'reason': reason, 'trigger_level': trigger, 'stop': case['stop'],
              'parent_version_id': bound['id'] if bound else None,
              'parent_lineage_id': bound['lineage_id'] if bound else None,
              'parent_low': low, 'parent_high': high, 'parent_state': parent['state'],
              'reserves_capacity': action in ('immediate', 'wait'),
              'predicates': predicates, 'required_evidence_ids': evidence,
              'failed_predicates': [k for k, v in predicates.items() if v == 'fail'],
              'unknown_predicates': [k for k, v in predicates.items() if v == 'unknown'],
              'annotation_context': {'daily_state': daily, 'optional': deepcopy(case['optional']),
                                     'destinations': deepcopy(case['destinations'])},
              'execution_authorized': False}
    result['card'] = (f"{case['candidate_id']}: {scenario}; {state}: {reason}. "
                      f"Frozen 4H reference {result['parent_version_id']} has originating-close "
                      f"state {parent['state']}; legacy detector state {parent['legacy_state']} is separate. "
                      f"Daily state {daily} is annotation, not a permission gate. "
                      "Optional scores cannot replace the required location and sequence; "
                      "this is not a complete Wyckoff phase diagnosis.")
    return dict(result, seal=seal(result))
