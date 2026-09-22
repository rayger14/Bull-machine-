"""Single-role plans are usable without pretending a critic reviewed them."""
from copy import deepcopy
import importlib

import pytest

from scripts.research.lc_published_assessment import build_published_request, gate_published_choice
from tests.research.test_lc_context_assessment import context_request
from tests.research.test_lc_published_assessment import raw_answer


def api():
    name = 'scripts.research.lc_single_assessment'
    assert importlib.util.find_spec(name), 'single-assessor contract missing'
    return importlib.import_module(name)


@pytest.mark.parametrize('interpretation,plan_id,action', [
    ('support', 'enter', 'enter'),
    ('support', 'wait_5m_high', 'wait_close_above'),
    ('oppose', 'reject', 'reject'),
])
def test_exact_menu_choice_without_fabricated_review(interpretation, plan_id, action):
    source = context_request(); before = deepcopy(source)
    request = build_published_request(source)
    raw = raw_answer(source, request, interpretation=interpretation, plan_id=plan_id)
    result = api().gate_single_choice(source, request, raw)
    assert result['status'] == 'schema_valid_unreviewed'
    assert result['review_status'] == 'unreviewed'
    assert result['semantic_validity'] == 'not_independently_verified'
    assert result['research_plan']['action'] == action
    assert result['research_plan']['notional'] == 50000
    assert result['execution_authorized'] is False
    assert result['transport_authenticated'] is False
    assert source == before
    # The old gate still requires its real critic.
    assert gate_published_choice(source, request, raw, None)['research_plan'] is None


def test_uncertainty_is_null_not_reject():
    source = context_request(current_validated=False)
    request = build_published_request(source)
    raw = raw_answer(source, request, interpretation='uncertain', plan_id=None)
    result = api().gate_single_choice(source, request, raw)
    assert result['status'] == 'insufficient_evidence'
    assert result['research_plan'] is None


@pytest.mark.parametrize('mutation', ['malformed', 'wrong_case', 'unknown_citation', 'unknown_plan'])
def test_invalid_answer_never_becomes_zero_exposure(mutation):
    import json
    source = context_request(); request = build_published_request(source)
    raw = raw_answer(source, request)
    value = json.loads(raw)
    if mutation == 'wrong_case': value['case_id'] = 'OTHER'
    if mutation == 'unknown_citation': value['supporting'][0]['evidence_ids'] = ['FAKE']
    if mutation == 'unknown_plan': value['plan_id'] = 'invented'
    raw = '{' if mutation == 'malformed' else json.dumps(value)
    result = api().gate_single_choice(source, request, raw)
    assert result['status'] == 'invalid_assessment'
    assert result['research_plan'] is None
    assert result['errors']


def test_source_drift_cannot_be_reinterpreted_as_valid_choice():
    source = context_request(); request = build_published_request(source)
    raw = raw_answer(source, request)
    source['plan_menu']['plans']['enter']['notional'] = 1
    assert api().gate_single_choice(source, request, raw)['research_plan'] is None
