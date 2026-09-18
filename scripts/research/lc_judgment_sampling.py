"""Outcome-independent chronological sampling approved after the LC census.

This policy deliberately permits gaps from prior assessments. Its roster is a
filtered judgment sample, not a continuous detector/portfolio history.
"""

POLICY = 'lc_unassessed_chronological_v1'


def select_unassessed(candidate_ids, excluded, cap=30):
    """Preserve chronological input order, exclude prior cases, take at most cap."""
    if type(cap) is not int or not 1 <= cap <= 30:
        raise ValueError('cap must be an integer from 1 to 30')
    if not isinstance(candidate_ids, list) or not isinstance(excluded, set):
        raise ValueError('ordered ID list and exclusion set required')
    if any(not isinstance(x, str) or not x for x in candidate_ids):
        raise ValueError('nonempty candidate IDs required')
    if len(set(candidate_ids)) != len(candidate_ids):
        raise ValueError('duplicate candidate IDs')
    if any(not isinstance(x, str) or not x for x in excluded):
        raise ValueError('nonempty excluded IDs required')
    return [x for x in candidate_ids if x not in excluded][:cap]
