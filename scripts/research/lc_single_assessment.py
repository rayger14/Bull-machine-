"""Prospective single-specialist gate; schema validity is not semantic review.

Reuses the unchanged published request, citations and fixed plan menu. The old
two-role contract remains intact. Transport must be verified by the caller.
"""
from copy import deepcopy

from scripts.research.lc_context_assessment import _parse
from scripts.research.lc_published_assessment import grade_published_choice


VERSION = 'lc_single_assessment_v1'


def gate_single_choice(source_request, request, raw_choice):
    errors = grade_published_choice(source_request, request, raw_choice)
    result = {
        'version': VERSION,
        'status': 'invalid_assessment',
        'review_status': 'unreviewed',
        'semantic_validity': 'not_independently_verified',
        'research_plan': None,
        'execution_authorized': False,
        'transport_authenticated': False,
        'errors': errors,
    }
    if errors:
        return result
    choice = _parse(raw_choice)
    if choice['plan_id'] is None:
        return dict(result, status='insufficient_evidence')
    selected = source_request['plan_menu']['plans'][choice['plan_id']]
    plan = dict(deepcopy(selected['parameters']), notional=selected['notional'],
                cost_bps=selected['cost_bps'])
    return dict(result, status='schema_valid_unreviewed', research_plan=plan)
