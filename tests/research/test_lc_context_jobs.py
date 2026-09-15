"""Immutable-storage regression tests for the separate context namespace."""
from copy import deepcopy
import importlib
import json

import pytest

from tests.research.test_lc_context_assessment import (
    api, choice, context_request, raw_choice, raw_review,
)
from tests.research.test_master_research_jobs import job as v1_job


def context_job(path):
    name = 'scripts.research.lc_context_jobs'
    assert importlib.util.find_spec(name), 'context research job adapter missing'
    return importlib.import_module(name).ContextResearchJob(path)


def valid_transport(role, valid=True):
    return dict(role=role, agent_id=role + '-fixture', requested_model='fixture-model',
                actual_model='fixture-model', capture_sha256='a' * 64,
                transport_valid=valid)


def captures(path, specialist_valid=True, reviewer_valid=True):
    req = context_request(); answer = raw_choice(req)
    critique = raw_review(req, answer)
    context_job(path).prepare(req)
    context_job(path).capture('specialist', answer,
                              valid_transport('specialist', specialist_valid))
    context_job(path).capture('reviewer', critique,
                              valid_transport('reviewer', reviewer_valid))
    return req, answer, critique


def test_context_job_reopens_and_recomputes_grade(tmp_path):
    path = tmp_path / 'case'; req = context_request()
    context_job(path).prepare(req)
    answer = raw_choice(req, plan_id='enter')
    context_job(path).capture('specialist', answer, valid_transport('specialist'))
    review = raw_review(req, answer)
    context_job(path).capture('reviewer', review, valid_transport('reviewer'))
    reopened = context_job(path)
    assert reopened.lock_grade()['research_plan']['action'] == 'enter'


def test_changed_raw_response_and_forged_grade_are_refused(tmp_path):
    path = tmp_path / 'case'; req, answer, critique = captures(path)
    with pytest.raises(ValueError):
        context_job(path).capture('specialist', answer + ' ', valid_transport('specialist'))
    forged = api().gate_context_choice(req, answer, critique)
    forged['research_plan']['stop'] = 1.
    with pytest.raises(ValueError):
        context_job(path).lock_grade(forged)


@pytest.mark.parametrize('bad_role', ['specialist', 'reviewer'])
def test_invalid_transport_nulls_a_valid_research_plan(tmp_path, bad_role):
    path = tmp_path / bad_role
    req, answer, critique = captures(
        path, specialist_valid=bad_role != 'specialist', reviewer_valid=bad_role != 'reviewer')
    positive = api().gate_context_choice(req, answer, critique)
    with pytest.raises(ValueError):
        context_job(path).lock_grade(positive)
    grade = context_job(path).lock_grade()
    assert grade['status'] == 'invalid_transport'
    assert grade['research_plan'] is None


def test_context_job_rejects_v1_directory_on_reopen(tmp_path):
    path = tmp_path / 'v1'
    from tests.research.test_lc_master_assessment import request
    v1_job(path).prepare(request())
    with pytest.raises(ValueError, match='context'):
        context_job(path)


@pytest.mark.parametrize('stage', ['empty', 'request', 'specialist', 'reviewer'])
def test_reveal_is_refused_before_context_grade(tmp_path, stage):
    path = tmp_path / stage; req = context_request(); answer = raw_choice(req)
    runner = context_job(path)
    if stage != 'empty': runner.prepare(req)
    if stage in ('specialist', 'reviewer'):
        runner.capture('specialist', answer, valid_transport('specialist'))
    if stage == 'reviewer':
        runner.capture('reviewer', raw_review(req, answer), valid_transport('reviewer'))
    with pytest.raises(ValueError):
        runner.authorize_reveal()


def test_reopen_revalidates_namespace_even_if_artifact_hash_is_self_consistent(tmp_path):
    path = tmp_path / 'case'; runner = context_job(path); req = context_request()
    runner.prepare(req)
    artifact = json.loads((path / 'request.json').read_text())
    artifact['payload']['version'] = 'lc_nested_child_rejection_v1'
    # The base store canonicalization/hashing is public; simulate a self-consistent
    # foreign namespace rather than relying only on byte-tamper detection.
    from scripts.research.conditional_assessment import digest
    from scripts.research.assessment_evidence_guard import _canonical
    artifact['sha256'] = digest({k: v for k, v in artifact.items() if k != 'sha256'})
    (path / 'request.json').write_text(_canonical(artifact))
    with pytest.raises(ValueError, match='context'):
        context_job(path)


def test_null_choice_and_failed_critic_lock_without_a_research_plan(tmp_path):
    for suffix, kwargs in [('null', dict(interpretation='uncertain', plan_id=None)),
                           ('critic', dict())]:
        path = tmp_path / suffix; req = context_request(); answer = raw_choice(req, **kwargs)
        critique = (raw_review(req, answer, complete=False) if suffix == 'critic'
                    else raw_review(req, answer))
        runner = context_job(path); runner.prepare(req)
        runner.capture('specialist', answer, valid_transport('specialist'))
        runner.capture('reviewer', critique, valid_transport('reviewer'))
        assert runner.lock_grade()['research_plan'] is None
