"""Bounded single-role research campaign, independent of frozen two-role runs.

No model invocation or outcome IO. Keep one controller alive across dispatch and
capture for measured timing. Call limits do not guarantee a credit/dollar cap.
"""
from copy import deepcopy
import fcntl
import hashlib
import math
import os
from pathlib import Path
import time
import uuid

from scripts.research.assessment_evidence_guard import build_envelope, validate_runtime_returns
from scripts.research.lc_context_assessment import _context_input
from scripts.research.lc_judgment_runner import _digest, _load, _save_equal as _publish, _sha, _verify_lock
from scripts.research.lc_published_assessment import build_published_request
from scripts.research.lc_setup_preflight import annotate_lc_setup
from scripts.research.lc_single_assessment import gate_single_choice


VERSION = 'lc_single_campaign_v1'
TIMEOUT_SECONDS = 600
POLICY = {
    'max_assessor_calls': 18, 'max_critic_calls': 0, 'retries': 0,
    'requested_model': 'gpt-6-astra', 'requested_effort': 'high',
    'credit_usage': None, 'credit_cap_enforced': False,
    'review_status': 'unreviewed', 'max_parallel': 3,
    'timeout_seconds': TIMEOUT_SECONDS,
    'scenarios': [[12, 90], [24, 90], [12, 300], [24, 300]],
    'measured_delay_cost_bps': [12, 24],
    'arms': ['immediate', 'mechanical_wait', 'agent', 'reject_all'],
    'notional': 50000, 'horizon_minutes': 1440, 'entry_expiry_minutes': 15,
    'execution_authorized': False,
}


def _sealed(value):
    return dict(value, sha256=_digest(value))


def _save_equal(path, value):
    path = _publish(path, value)
    for directory in (path.parent, path.parent.parent):
        descriptor = os.open(str(directory), os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    return path


def prepare_campaign(directory, source_paths, *, excluded_ids, input_lock, extra_files=()):
    """Freeze chronological remaining cases from a verified pre-existing lock."""
    root = Path(directory).resolve()
    locked = _verify_lock(input_lock)
    if not isinstance(excluded_ids, set):
        raise ValueError('explicit exclusion set required')
    cases = []; ids = []
    files = {str(Path(input_lock).resolve()): _sha(input_lock)}
    files.update({str(Path(p).resolve()): _sha(p) for p in extra_files})
    allowed = {str(Path(p).resolve()): h for p, h in locked['files'].items()}
    for path in source_paths:
        path = Path(path).resolve()
        if allowed.get(str(path)) != _sha(path):
            raise ValueError('source absent from original input lock')
        source = _load(path)
        request = build_published_request(source)
        cid = source['case_id']; ids.append(cid)
        files[str(path)] = _sha(path)
        annotation = annotate_lc_setup(_context_input(source['source_packet']))
        cases.append(dict(case_id=cid, decision_time=annotation['decision_time'],
                          source_path=str(path), subtype=annotation['subtype'],
                          request=request, source=source))
    if len(ids) != len(set(ids)) or set(ids) != set(locked['roster']):
        raise ValueError('source IDs must match entire original roster exactly')
    if not excluded_ids <= set(ids):
        raise ValueError('exclusion outside original roster')
    cases = sorted((c for c in cases if c['case_id'] not in excluded_ids),
                   key=lambda c: (c['decision_time'], c['case_id']))
    if not 1 <= len(cases) <= POLICY['max_assessor_calls']:
        raise ValueError('one to eighteen remaining cases required; never silently truncate')
    frozen = []
    for index, case in enumerate(cases):
        folder = root / 'cases' / f'{index:03d}'
        wrapper = dict(case_id=case['case_id'], plan=case['source']['plan'],
                       request=case['request'], setup_subtype=case['subtype'],
                       subtype_meaning='candidate thesis, not an entry gate')
        wrapper_path = _save_equal(folder / 'packet.json', wrapper)
        envelope_path = _save_equal(folder / 'envelope.json', build_envelope(wrapper, 16000))
        files.update({str(p): _sha(p) for p in (wrapper_path, envelope_path)})
        frozen.append({k: case[k] for k in ('case_id', 'decision_time', 'source_path', 'subtype')})
    # Pin the whole small research-code directory to include transitive imports.
    files.update({str(p.resolve()): _sha(p) for p in Path(__file__).parent.glob('*.py')})
    manifest = _sealed(dict(version=VERSION, policy=POLICY, cases=frozen,
                            excluded_ids=sorted(excluded_ids), files=files,
                            input_lock_path=str(Path(input_lock).resolve()),
                            exposure='filtered retrospective; not pristine holdout'))
    _save_equal(root / 'manifest.json', manifest)
    return deepcopy(manifest)


def prepare_remaining_campaign(directory, preparation_dir):
    """Production boundary: original frozen twenty minus the two captured cases."""
    from scripts.research.lc_published_jobs import PublishedContextResearchJob
    prep = Path(preparation_dir).resolve()
    binding = prep / 'preparation_binding.json'
    _verify_lock(binding)
    lock_path = prep / 'evidence' / 'evidence_lock.json'
    lock = _verify_lock(lock_path)
    excluded = {'hourly-lc:2026-01-20T06:00:00+00:00',
                'hourly-lc:2026-01-25T09:00:00+00:00'}
    roster = lock['roster']
    if len(roster) != 20 or len(set(roster)) != 20 or not excluded <= set(roster):
        raise ValueError('exact frozen twenty and two prior cases required')
    extras = [binding, Path('docs/knowledge/lc_bounded_single_protocol_2026_09_20.md').resolve()]
    checkpoint = prep / 'runtime_checkpoint_v1'
    terminal_paths = list((checkpoint / 'terminals').glob('*.json'))
    if {p.stem for p in terminal_paths} != excluded:
        raise ValueError('prior assessment roster changed; stop and reconcile')
    for cid in sorted(excluded):
        terminal_path = checkpoint / 'terminals' / (cid + '.json')
        terminal = _load(terminal_path)
        job_path = checkpoint / 'jobs' / cid
        if Path(terminal['job_directory']).resolve() != job_path.resolve():
            raise ValueError('prior assessment job directory mismatch')
        stages = PublishedContextResearchJob(job_path)._read()
        if (terminal['kind'] != 'published_grade'
                or terminal['grade_sha256'] != stages['grade']['sha256']
                or stages['request']['payload']['source_request']['case_id'] != cid):
            raise ValueError('prior assessment binding mismatch')
        extras.extend([terminal_path, *job_path.glob('*.json')])
    return prepare_campaign(directory,
        [prep / 'evidence' / (cid + '_source_request.json') for cid in roster],
        excluded_ids=excluded, input_lock=lock_path, extra_files=extras)


class SingleCampaign:
    def __init__(self, directory, *, monotonic_ns=time.monotonic_ns):
        self.root = Path(directory).resolve()
        self.clock = monotonic_ns
        self.runtime_id = str(uuid.uuid4())
        self._owner = (self.root / 'owner.lock').open('a+')
        try:
            fcntl.flock(self._owner, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            self._owner.close()
            raise ValueError('campaign already has an owner') from exc
        try:
            self.verify()
        except Exception:
            self.close()
            raise

    def close(self):
        if not self._owner.closed:
            fcntl.flock(self._owner, fcntl.LOCK_UN)
            self._owner.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def verify(self):
        if self._owner.closed:
            raise ValueError('controller is closed')
        manifest = _verify_lock(self.root / 'manifest.json')
        if manifest['version'] != VERSION or manifest['policy'] != POLICY:
            raise ValueError('campaign policy changed')
        _verify_lock(manifest['input_lock_path'])
        return manifest

    def _case(self, case_id):
        manifest = self.verify()
        for index, case in enumerate(manifest['cases']):
            if case['case_id'] == case_id:
                return case, self.root / 'cases' / f'{index:03d}'
        raise ValueError('unknown case')

    def authorize(self, user_instruction):
        manifest = self.verify()
        if not isinstance(user_instruction, str) or not user_instruction.strip():
            raise ValueError('explicit user authorization required')
        _save_equal(self.root / 'authorization.json', _sealed(dict(
            manifest_sha256=manifest['sha256'], user_instruction=user_instruction,
            call_ceiling=len(manifest['cases']), credit_usage=None, credit_cap_enforced=False)))

    def reserve(self, case_id):
        case, folder = self._case(case_id)
        auth_path = self.root / 'authorization.json'
        if not auth_path.exists():
            raise ValueError('campaign not authorized')
        auth = _verify_lock(auth_path)
        if auth['manifest_sha256'] != self.verify()['sha256']:
            raise ValueError('authorization binding changed')
        if (folder / 'reservation.json').exists():
            raise ValueError('one attempt only; no retry')
        if (self.root / 'terminal_lock.json').exists() or (folder / 'terminal.json').exists():
            raise ValueError('terminal forbids dispatch')
        active = sum(not (p.parent / 'terminal.json').exists()
                     for p in self.root.glob('cases/*/reservation.json'))
        if active >= POLICY['max_parallel']:
            raise ValueError('parallel call ceiling reached')
        value = _sealed(dict(case_id=case_id, runtime_id=self.runtime_id,
                             started_ns=self.clock(), manifest_sha256=self.verify()['sha256']))
        _save_equal(folder / 'reservation.json', value)
        return dict(case_id=case_id, folder=str(folder),
                    wrapper=_load(folder / 'packet.json'), envelope=_load(folder / 'envelope.json'))

    def _open_attempt(self, case_id):
        case, folder = self._case(case_id)
        if (folder / 'terminal.json').exists() or (self.root / 'terminal_lock.json').exists():
            raise ValueError('terminal already recorded')
        reservation = _verify_lock(folder / 'reservation.json')
        if (reservation['case_id'] != case_id
                or reservation['manifest_sha256'] != self.verify()['sha256']):
            raise ValueError('reservation binding changed')
        return case, folder, reservation

    def capture(self, case_id, raw_bytes, metadata, runtime_returns):
        case, folder, reservation = self._open_attempt(case_id)
        if not isinstance(raw_bytes, bytes):
            raise ValueError('exact response bytes required')
        fields = {'agent_id', 'requested_model', 'actual_model', 'actual_runtime_verified'}
        if (not isinstance(metadata, dict) or set(metadata) != fields
                or type(metadata['actual_runtime_verified']) is not bool
                or not isinstance(metadata['agent_id'], str) or not metadata['agent_id'].strip()
                or metadata['requested_model'] != POLICY['requested_model']
                or (metadata['actual_model'] is not None and
                    (not isinstance(metadata['actual_model'], str) or not metadata['actual_model'].strip()))):
            raise ValueError('exact declared role metadata required')
        now = self.clock()
        elapsed = now - reservation['started_ns']
        same_runtime = reservation['runtime_id'] == self.runtime_id and elapsed >= 0
        seconds = math.ceil(elapsed / 1e9) if same_runtime else None
        wrapper = _load(folder / 'packet.json'); envelope = _load(folder / 'envelope.json')
        validation = validate_runtime_returns(wrapper, envelope, runtime_returns,
                                               case_id=case_id, role='specialist')
        unique = all(_load(p)['metadata']['agent_id'] != metadata['agent_id']
                     for p in self.root.glob('cases/*/capture.json') if p.parent != folder)
        capture = dict(raw_hex=raw_bytes.hex(), raw_sha256=hashlib.sha256(raw_bytes).hexdigest(),
                       metadata=metadata, runtime_returns=runtime_returns,
                       validation=validation, agent_unique=unique,
                       reservation_sha256=reservation['sha256'], ended_ns=now,
                       runtime_id=self.runtime_id,
                       prior_captures={str(p): _sha(p) for p in self.root.glob('cases/*/capture.json')
                                       if p.parent != folder})
        _save_equal(folder / 'capture.json', _sealed(capture))
        grade, seconds = self._evaluate_capture(case, folder, reservation, capture)
        return self._terminal(folder, case_id, grade, seconds,
                              [folder / 'reservation.json', folder / 'capture.json'])

    def _evaluate_capture(self, case, folder, reservation, capture):
        wrapper = _load(folder / 'packet.json'); envelope = _load(folder / 'envelope.json')
        raw_bytes = bytes.fromhex(capture['raw_hex'])
        validation = validate_runtime_returns(wrapper, envelope, capture['runtime_returns'],
                                               case_id=case['case_id'], role='specialist')
        for p, expected in capture['prior_captures'].items():
            if _sha(p) != expected:
                raise ValueError('prior capture changed')
        unique = all(_load(p)['metadata']['agent_id'] != capture['metadata']['agent_id']
                     for p in capture['prior_captures'])
        if (capture['raw_sha256'] != hashlib.sha256(raw_bytes).hexdigest()
                or capture['reservation_sha256'] != reservation['sha256']
                or capture['validation'] != validation or capture['agent_unique'] != unique):
            raise ValueError('capture binding changed')
        elapsed = capture['ended_ns'] - reservation['started_ns']
        same_runtime = capture['runtime_id'] == reservation['runtime_id'] and elapsed >= 0
        seconds = math.ceil(elapsed / 1e9) if same_runtime else None
        try:
            raw = raw_bytes.decode('utf-8')
        except UnicodeError:
            raw = '{'  # Raw bytes above are preserved; invalid JSON produces no plan.
        grade = gate_single_choice(_load(case['source_path']), wrapper['request'], raw)
        override = None
        if not validation['valid'] or not unique or not capture['metadata']['actual_runtime_verified']:
            override = 'invalid_transport'
        if not same_runtime:
            override = 'interrupted_runtime'
        elif elapsed > TIMEOUT_SECONDS * 1_000_000_000:
            override = 'timeout'
        if override:
            grade = dict(grade, status=override, research_plan=None,
                         errors=grade['errors'] + [override])
        return grade, seconds

    def _terminal(self, folder, case_id, grade, seconds, files, reason=None):
        terminal = _sealed(dict(case_id=case_id, grade=grade, processing_seconds=seconds,
                                reason=reason, files={str(p): _sha(p) for p in files}))
        _save_equal(folder / 'terminal.json', terminal)
        return terminal

    def fail(self, case_id, reason):
        _, folder, _ = self._open_attempt(case_id)
        if not isinstance(reason, str) or not reason.strip():
            raise ValueError('explicit failure reason required')
        return self._terminal(folder, case_id,
            dict(status='external_failure', research_plan=None, review_status='unreviewed',
                 execution_authorized=False, transport_authenticated=False), None,
            [folder / 'reservation.json'], reason=reason)

    def lock_terminals(self):
        manifest = self.verify(); terminals = {}; files = {}
        for index, case in enumerate(manifest['cases']):
            path = self.root / 'cases' / f'{index:03d}' / 'terminal.json'
            if not path.exists():
                raise ValueError('all case terminals required before outcome reveal')
            terminal = _verify_lock(path)
            if terminal['case_id'] != case['case_id']:
                raise ValueError('terminal identity differs')
            folder = path.parent
            reservation = _verify_lock(folder / 'reservation.json')
            if (reservation['case_id'] != case['case_id']
                    or reservation['manifest_sha256'] != manifest['sha256']):
                raise ValueError('terminal reservation binding differs')
            dependencies = [folder / 'reservation.json']
            if terminal['reason'] is None:
                capture = _verify_lock(folder / 'capture.json')
                grade, seconds = self._evaluate_capture(case, folder, reservation, capture)
                dependencies.append(folder / 'capture.json')
            else:
                if not isinstance(terminal['reason'], str) or not terminal['reason'].strip():
                    raise ValueError('terminal failure reason missing')
                grade = dict(status='external_failure', research_plan=None, review_status='unreviewed',
                             execution_authorized=False, transport_authenticated=False)
                seconds = None
            expected = _sealed(dict(case_id=case['case_id'], grade=grade,
                processing_seconds=seconds, reason=terminal['reason'],
                files={str(p): _sha(p) for p in dependencies}))
            if terminal != expected:
                raise ValueError('terminal differs from captured evidence')
            terminals[case['case_id']] = terminal; files[str(path)] = _sha(path)
        result = _sealed(dict(version=VERSION, manifest_sha256=manifest['sha256'],
                              terminals=terminals, files=files, outcome_reveal_authorized=True,
                              execution_authorized=False))
        _save_equal(self.root / 'terminal_lock.json', result)
        return result

    def assert_reveal_allowed(self):
        self.verify()
        if not (self.root / 'terminal_lock.json').exists():
            raise ValueError('all terminal results must be locked first')
        locked = _verify_lock(self.root / 'terminal_lock.json')
        if self.lock_terminals() != locked:
            raise ValueError('terminal lock changed')
        return locked
