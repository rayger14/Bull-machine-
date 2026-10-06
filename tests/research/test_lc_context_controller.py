import importlib

import pytest

from scripts.research.lc_context_contract import sealed_case
from scripts.research.lc_context_evidence import prepare_evidence
from tests.research.lc_context_fixtures import clone, source


def changed(case, **fields):
    value = clone(case)
    value.pop('seal')
    value.update(fields)
    return sealed_case(value)


def contextual(subtype='upside_expansion_candidate', state='inside', close=102., low=99.):
    case = prepare_evidence(*source())
    parent = clone(case['parent_4h'])
    parent['state'] = state
    return changed(case, subtype=subtype, parent_4h=parent,
                   current=dict(case['current'], close=close, low=low))


def test_context_scenario_table():
    classify = importlib.import_module('scripts.research.lc_context_controller').classify
    fixtures = [
        (contextual(state='accepted_above', close=112.), 'accepted_expansion', 'eligible', 'immediate', None),
        (contextual(), 'local_range_expansion', 'awaiting_confirmation', 'wait', 103.),
        (contextual(close=112.), 'parent_acceptance_unconfirmed', 'watching', 'none', None),
        (contextual(state='boundary', close=112.), 'parent_acceptance_unconfirmed', 'watching', 'none', None),
        (contextual(state='not_established', close=112.), 'parent_acceptance_unconfirmed', 'watching', 'none', None),
        (contextual('downside_rebound_candidate', low=89.), 'range_floor_rebound', 'awaiting_confirmation', 'wait', 103.),
        (contextual('downside_rebound_candidate', state='accepted_below', low=89.), 'rebound_invalidated', 'invalidated', 'none', None),
        (contextual('downside_rebound_candidate', low=90.), 'outside_playbook', 'outside_playbook', 'none', None),
        (contextual(state='accepted_above', close=102.), 'outside_playbook', 'outside_playbook', 'none', None),
    ]
    for case, scenario, state, action, trigger in fixtures:
        result = classify(case)
        assert (result['scenario'], result['state'], result['action'], result['trigger_level']) == (scenario, state, action, trigger)
        assert result['execution_authorized'] is False
        assert result['parent_version_id'] == 'range-old'
        assert result['predicates']


def test_required_unknown_and_known_absence_not_conflated():
    classify = importlib.import_module('scripts.research.lc_context_controller').classify
    case = contextual()
    for status, expected, reason in [('absent', 'outside_playbook', 'no_parent_reference'),
                                     ('unknown', 'insufficient_evidence', 'unknown_parent_context')]:
        parent = dict(case['parent_4h'], status=status, bound=None, state=status)
        result = classify(changed(case, parent_4h=parent))
        assert result['state'] == expected
        assert result['reason'] == reason
        assert result['reserves_capacity'] is False
    result = classify(changed(case, risk_status='unknown', stop=None))
    assert result['state'] == 'insufficient_evidence'
    assert result['reason'] == 'unknown_risk'


def test_optional_evidence_has_no_permission_vote_and_daily_not_bullish_gate():
    classify = importlib.import_module('scripts.research.lc_context_controller').classify
    case = contextual('downside_rebound_candidate', low=89.)
    first = classify(case)
    other = changed(case, optional={'duplicated_bullish_votes': [1]*100},
                    parent_1d=dict(case['parent_1d'], state='accepted_below'))
    second = classify(other)
    for key in ['state', 'scenario', 'action', 'trigger_level', 'reserves_capacity']:
        assert first[key] == second[key]
    assert 'not a permission gate' in second['card']
    assert second['annotation_context']['daily_state'] == 'accepted_below'


def test_containment_and_floor_reclaim_are_specific_sequence_requirements():
    classify = importlib.import_module('scripts.research.lc_context_controller').classify
    case = contextual()
    assert classify(changed(case, prior=dict(case['prior'], low=89.)))['action'] == 'none'
    rebound = contextual('downside_rebound_candidate', low=89.)
    parent = clone(rebound['parent_4h'])
    parent['bound']['range_low'] = 105.
    assert classify(changed(rebound, parent_4h=parent))['trigger_level'] == 105.


@pytest.mark.parametrize('scenario,price,expected', [('accepted_expansion', 110., False),
    ('accepted_expansion', 111., True), ('local_range_expansion', 90., False),
    ('local_range_expansion', 110., False), ('local_range_expansion', 100., True),
    ('range_floor_rebound', 89., False), ('range_floor_rebound', 100., True)])
def test_actual_fill_location_uses_strict_frozen_boundary(scenario, price, expected):
    api = importlib.import_module('scripts.research.lc_context_controller')
    decision = {'scenario': scenario, 'parent_low': 90., 'parent_high': 110.}
    assert api.entry_location_ok(decision, price) is expected


def test_tampered_case_and_nonfinite_fill_fail_closed():
    api = importlib.import_module('scripts.research.lc_context_controller')
    case = contextual()
    case['stop'] = 1.
    with pytest.raises(ValueError, match='seal'):
        api.classify(case)
    assert api.entry_location_ok({'scenario': 'accepted_expansion', 'parent_low': 90.,
                                  'parent_high': 110.}, float('nan')) is False
