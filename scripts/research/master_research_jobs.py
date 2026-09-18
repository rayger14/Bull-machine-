"""Immutable, hash-chained offline jobs with caller-declared role provenance.

This store cannot authenticate model identity, transport truth or the market
meaning of prose. Validate actual delivery captures before attesting transport.
"""
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import tempfile

from scripts.research.assessment_evidence_guard import _canonical
from scripts.research.conditional_assessment import digest
from scripts.research.lc_master_assessment import _validate_request, gate_lc_choice


STAGES = ('request','specialist','reviewer','grade','reveal','outcome')
PROVENANCE_KEYS = {'role','agent_id','requested_model','actual_model','capture_sha256','transport_valid'}


def _sha(raw):
    return hashlib.sha256(raw.encode('utf-8')).hexdigest()


def _metadata(role,provenance):
    if not isinstance(provenance,dict) or set(provenance)!=PROVENANCE_KEYS:
        raise ValueError('exact transport provenance required')
    if provenance['role']!=role or type(provenance['transport_valid']) is not bool:
        raise ValueError('invalid role/transport flag')
    for key in ('agent_id','requested_model'):
        if not isinstance(provenance[key],str) or not provenance[key].strip():
            raise ValueError('invalid '+key)
    if provenance['actual_model'] is not None and (
            not isinstance(provenance['actual_model'],str) or not provenance['actual_model'].strip()):
        raise ValueError('invalid actual_model')
    sha=provenance['capture_sha256']
    if sha is not None and (not isinstance(sha,str) or len(sha)!=64 or any(c not in '0123456789abcdef' for c in sha)):
        raise ValueError('invalid capture_sha256')


class ResearchJob:
    """Reopen at any stage. Unequal replays fail; all existing hashes rechecked.

    Each JSON stage contains its prior stage hash and immutable payload. Capture
    payloads preserve exact raw strings plus raw UTF-8 SHA256. Atomic exclusive
    links prevent overwriting a completed stage even with concurrent writers.
    """
    def __init__(self,directory):
        self.directory=Path(directory)
        self.directory.mkdir(parents=True,exist_ok=True)
        self._read()

    def _read(self):
        stages={}; previous=None; missing=False
        for stage in STAGES:
            path=self.directory/(stage+'.json')
            if not path.exists():
                missing=True
                continue
            if missing: raise ValueError('out-of-order job artifacts')
            try:
                raw=path.read_bytes(); value=json.loads(raw)
                if set(value)!={'stage','previous_sha256','payload','sha256'}:
                    raise ValueError('artifact keys')
                body={k:v for k,v in value.items() if k!='sha256'}
                if (value['stage']!=stage or value['previous_sha256']!=previous
                        or digest(body)!=value['sha256'] or _canonical(value).encode('ascii')!=raw):
                    raise ValueError('artifact hash or bytes changed')
                if stage in ('specialist','reviewer'):
                    capture=value['payload']
                    _metadata(stage,capture['provenance'])
                    if _sha(capture['raw_response'])!=capture['raw_sha256']:
                        raise ValueError('raw capture changed')
                stages[stage]=value; previous=value['sha256']
            except (ValueError,TypeError,KeyError,UnicodeError) as exc:
                raise ValueError('invalid '+stage+' artifact') from exc
        return stages

    @property
    def state(self):
        stages=self._read()
        return next(reversed(stages)) if stages else 'empty'

    def _save(self,stage,payload):
        stages=self._read(); index=STAGES.index(stage)
        if any(k not in stages for k in STAGES[:index]):
            raise ValueError('required prior stage missing')
        previous=stages[STAGES[index-1]]['sha256'] if index else None
        value=dict(stage=stage,previous_sha256=previous,payload=deepcopy(payload))
        value['sha256']=digest(value)
        raw=_canonical(value).encode('ascii')
        path=self.directory/(stage+'.json')
        if stage in stages:
            if stages[stage]!=value: raise ValueError('immutable '+stage+' differs')
            return deepcopy(payload)
        fd,temp=tempfile.mkstemp(prefix='.pending-',dir=self.directory)
        try:
            with os.fdopen(fd,'wb') as stream:
                stream.write(raw); stream.flush(); os.fsync(stream.fileno())
            try:
                os.link(temp,path)
            except FileExistsError:
                if path.read_bytes()!=raw: raise ValueError('concurrent unequal '+stage)
            directory_fd=os.open(self.directory,os.O_RDONLY)
            try: os.fsync(directory_fd)
            finally: os.close(directory_fd)
        finally:
            os.unlink(temp)
        self._read()
        return deepcopy(payload)

    def prepare(self,request):
        self._read()
        _validate_request(request)
        return self._save('request',request)

    def capture(self,role,raw_response,provenance):
        self._read()
        if role not in ('specialist','reviewer') or not isinstance(raw_response,str):
            raise ValueError('exact raw role string required')
        _metadata(role,provenance)
        return self._save(role,dict(raw_response=raw_response,raw_sha256=_sha(raw_response),
                                   provenance=deepcopy(provenance)))

    def lock_grade(self,grade=None):
        """Recompute from pinned raw captures. Optional grade must match exactly."""
        stages=self._read()
        if 'reviewer' not in stages: raise ValueError('both captures required')
        request=stages['request']['payload']
        specialist=stages['specialist']['payload']; reviewer=stages['reviewer']['payload']
        expected=gate_lc_choice(request,specialist['raw_response'],reviewer['raw_response'])
        if not all(c['provenance']['transport_valid'] is True for c in (specialist,reviewer)):
            expected=dict(expected,status='invalid_transport',research_plan=None,
                          errors=expected['errors']+['unverified_declared_transport'])
        if grade is not None and _canonical(grade)!=_canonical(expected):
            raise ValueError('grade differs from pinned captures/transport')
        return self._save('grade',expected)

    def authorize_reveal(self):
        stages=self._read()
        if 'grade' not in stages: raise ValueError('grade required before reveal')
        return self._save('reveal',dict(grade_sha256=stages['grade']['sha256'],authorized=True))

    def save_outcome(self,outcome):
        return self._save('outcome',outcome)
