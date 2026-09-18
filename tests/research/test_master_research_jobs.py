"""Restart and tamper tests use temporary files and declared fixture provenance."""
from copy import deepcopy
import importlib
import json

import pytest

from tests.research.test_lc_master_assessment import api, request, choice, review


def job(path):
    name='scripts.research.master_research_jobs'
    assert importlib.util.find_spec(name), 'restartable research job missing'
    return importlib.import_module(name).ResearchJob(path)


def provenance(role,valid=True):
    return dict(role=role,agent_id=role+'-fixture',requested_model='fixture-model',
                actual_model='fixture-model',capture_sha256='a'*64,transport_valid=valid)


def captures(path,valid=True):
    req=request(); raw=json.dumps(choice(req),indent=2)
    critic=json.dumps(review(req,raw),indent=2)
    job(path).prepare(req)
    job(path).capture('specialist',raw,provenance('specialist',valid))
    job(path).capture('reviewer',critic,provenance('reviewer'))
    return req,raw,critic


def test_restart_at_every_stage_equal_replay_and_frozen_snapshot(tmp_path):
    path=tmp_path/'job'; req,raw,critic=captures(path)
    before=(path/'request.json').read_bytes()
    job(path).prepare(req)
    job(path).capture('specialist',raw,provenance('specialist'))
    grade=api().gate_lc_choice(req,raw,critic)
    assert job(path).lock_grade(grade)['research_plan'] is not None
    job(path).authorize_reveal()
    outcome={'status':'fixture_closed','net_pnl':12.}
    job(path).save_outcome(outcome)
    job(path).save_outcome(outcome)
    assert (path/'request.json').read_bytes()==before
    assert job(path).state=='outcome'


@pytest.mark.parametrize('stage', ['empty','prepared','specialist','reviewer'])
def test_reveal_refused_until_grade_is_locked(tmp_path,stage):
    path=tmp_path/'job'; req=request(); raw=json.dumps(choice(req))
    if stage!='empty': job(path).prepare(req)
    if stage in ('specialist','reviewer'): job(path).capture('specialist',raw,provenance('specialist'))
    if stage=='reviewer': job(path).capture('reviewer',json.dumps(review(req,raw)),provenance('reviewer'))
    with pytest.raises(ValueError): job(path).authorize_reveal()


def test_out_of_order_capture_grade_and_outcome_refused(tmp_path):
    path=tmp_path/'job'; req=request()
    with pytest.raises(ValueError): job(path).capture('specialist','{}',provenance('specialist'))
    job(path).prepare(req)
    with pytest.raises(ValueError): job(path).capture('reviewer','{}',provenance('reviewer'))
    with pytest.raises(ValueError): job(path).lock_grade({})
    with pytest.raises(ValueError): job(path).save_outcome({})


def test_changed_raw_request_provenance_and_grade_are_refused(tmp_path):
    path=tmp_path/'job'; req,raw,critic=captures(path)
    with pytest.raises(ValueError): job(path).capture('specialist',raw+' ',provenance('specialist'))
    meta=provenance('specialist'); meta['actual_model']='changed'
    with pytest.raises(ValueError): job(path).capture('specialist',raw,meta)
    changed=deepcopy(req); changed['memory_snapshot']['records'].append({})
    with pytest.raises(ValueError): job(path).prepare(changed)
    grade=api().gate_lc_choice(req,raw,critic)
    grade['research_plan']['stop']=1.
    with pytest.raises(ValueError): job(path).lock_grade(grade)


def test_tampered_earlier_artifact_refuses_every_later_transition(tmp_path):
    path=tmp_path/'job'; req,raw,critic=captures(path)
    job(path).lock_grade(api().gate_lc_choice(req,raw,critic))
    artifact=path/'specialist.json'
    artifact.write_bytes(artifact.read_bytes().replace(b'fixture-model',b'changed-model'))
    with pytest.raises(ValueError): job(path).authorize_reveal()


def test_unknown_transport_cannot_lock_a_positive_research_plan(tmp_path):
    path=tmp_path/'job'; req,raw,critic=captures(path,valid=False)
    positive=api().gate_lc_choice(req,raw,critic)
    with pytest.raises(ValueError): job(path).lock_grade(positive)
    grade=job(path).lock_grade()
    assert grade['status']=='invalid_transport'
    assert grade['research_plan'] is None
    job(path).authorize_reveal()


@pytest.mark.parametrize('field,value', [('role','reviewer'),('transport_valid',1),
    ('actual_model',False),('capture_sha256','bad'),('agent_id','')])
def test_transport_metadata_exact_schema(tmp_path,field,value):
    path=tmp_path/'job'; job(path).prepare(request())
    meta=provenance('specialist'); meta[field]=value
    with pytest.raises(ValueError): job(path).capture('specialist','{}',meta)


def test_unavailable_model_and_capture_are_honestly_recorded(tmp_path):
    path=tmp_path/'job'; job(path).prepare(request())
    meta=provenance('specialist',False); meta.update(actual_model=None,capture_sha256=None)
    job(path).capture('specialist','invalid raw JSON is still captured',meta)
    assert job(path).state=='specialist'
