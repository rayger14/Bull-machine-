"""Outcome-free LC thesis labels and evidence readiness, never entry gates.

Consumes sealed, caller-verified research packets. Hash integrity is not live
receipt authentication. This module neither invokes a model nor places orders.
"""
from collections import Counter

from scripts.research.lc_context_facts import describe_lc_context


VERSION = 'lc_setup_preflight_v1'


def annotate_lc_setup(packet):
    context = describe_lc_context(packet)
    hourly = context['hourly']
    subtype = 'unresolved'
    if hourly['status'] == 'known':
        if hourly['close_relation'] == 'above_prior_high':
            subtype = 'upside_expansion_candidate'
        elif (hourly['close_relation'] == 'below_prior_low'
              or hourly['reclaimed_prior_low']):
            subtype = 'downside_rebound_candidate'
    return {
        'version': VERSION,
        'case_id': context['case_id'],
        'decision_time': context['decision_time'],
        'source_packet_sha256': context['source_packet_sha256'],
        'subtype': subtype,
        'subtype_meaning': 'candidate thesis, not confirmed exhaustion or continuation',
        'context': context,
        'execution_authorized': False,
    }


def build_preflight(packets, *, outcome_exposed_ids):
    """Retain every case; unavailable evidence is not a rejection or zero PnL.

Known absent/broken parents are valid observations, not failed entry rules.
Exposure labels describe this caller's records, never certify untouched data.
"""
    annotations = [annotate_lc_setup(packet) for packet in packets]
    ids = [a['case_id'] for a in annotations]
    if len(set(ids)) != len(ids):
        raise ValueError('duplicate case IDs')
    if not isinstance(outcome_exposed_ids, (set, frozenset)):
        raise ValueError('explicit exposure ID set required')
    if not outcome_exposed_ids <= set(ids):
        raise ValueError('exposure IDs outside supplied roster')
    records = []
    for annotation in sorted(annotations, key=lambda a: (a['decision_time'], a['case_id'])):
        context = annotation['context']
        reasons = []
        if context['native_long'] is not True:
            reasons.append('native_long_not_verified')
        if context['hourly']['status'] != 'known':
            reasons.append('hourly_unknown')
        for tf in ('1m', '5m'):
            if context['last_two_' + tf]['status'] != 'known':
                reasons.append(tf + '_unknown')
        for tf in ('4h', '1d'):
            if context['parent_' + tf]['evidence_status'] != 'known':
                reasons.append('parent_' + tf + '_unknown')
        records.append({
            'case_id': annotation['case_id'],
            'decision_time': annotation['decision_time'],
            'annotation': annotation,
            'evidence_ready': not reasons,
            'unknown_reasons': reasons,
            'exposure_status': ('outcome_exposed' if annotation['case_id'] in outcome_exposed_ids
                                else 'no_new_reveal_recorded'),
        })
    return {
        'version': VERSION,
        'cases': records,
        'case_count': len(records),
        'ready_count': sum(row['evidence_ready'] for row in records),
        'subtype_counts': dict(sorted(Counter(a['subtype'] for a in annotations).items())),
        'exposed_count': len(outcome_exposed_ids),
        'execution_authorized': False,
        'assessment_dispatch_authorized': False,
    }
