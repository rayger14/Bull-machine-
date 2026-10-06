"""Immutable offline practice run and persistent capture owner. Never dispatches a model."""
from copy import deepcopy
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import time
import uuid

import pandas as pd

from scripts.research.assessment_evidence_guard import _canonical
from scripts.research.lc_context_assessment import _context_input
from scripts.research.lc_judgment_runner import _load, _save_equal, _sha, _verify_lock, _verify_digest
from scripts.research.conditional_assessment import digest
from scripts.research.lc_setup_preflight import annotate_lc_setup
from scripts.research.lc_structure_packet import build_structure_packet, clock
from scripts.research.lc_structure_proposal import validate_structure_proposal
from scripts.research.lc_practice_sources import practice_policy, verify_candles, mechanical_control, structure_request


VERSION = 'lc_practice_run_v1'
ROLE_POLICY = dict(max_calls=12, max_inflight=1, timeout_seconds=600,
    retries=0, critics=0, requested_model='gpt-6-astra', requested_effort='high',
    credit_cap_enforced=False, billed_credits=None)


def sealed(value):
    return dict(value, sha256=digest(value))


def publish(path, value):
    path = _save_equal(path, value)
    fd = os.open(str(path.parent), os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)
    return path


def read_archive(path, start, end):
    """UTC inclusive start, exclusive end; preparation only supplies past windows."""
    return pd.read_parquet(path, filters=[('ts', '>=', start.to_pydatetime()),
        ('ts', '<', end.to_pydatetime())]).rename(columns={'vol': 'volume'})


def prepare_run(directory, source_paths, *, input_lock, archive, limit=12):
    """Reusable <=12-case adapter. CLI additionally pins the exact real roster."""
    if type(limit) is not int or not 1 <= limit <= 12:
        raise ValueError('one to twelve cases required')
    root = Path(directory).resolve(); archive = Path(archive).resolve()
    locked = _verify_lock(input_lock)
    allowed = {str(Path(p).resolve()): h for p, h in locked['files'].items()}
    if allowed.get(str(archive)) != _sha(archive):
        raise ValueError('archive absent from original lock')
    cases = []; all_ids = []
    for path in source_paths:
        path = Path(path).resolve()
        if allowed.get(str(path)) != _sha(path):
            raise ValueError('source absent from original lock')
        source = _load(path); cid = source['case_id']; all_ids.append(cid)
        when = clock(source['source_packet']['decision_time']).isoformat()
        cases.append(dict(case_id=cid, decision_time=when, source_path=str(path), source=source))
    if len(all_ids) != len(set(all_ids)) or set(all_ids) != set(locked['roster']):
        raise ValueError('sources must match entire locked roster')
    if len(cases) < limit:
        raise ValueError('not enough roster cases; no silent substitution')
    cases.sort(key=lambda c: (c['decision_time'], c['case_id']))
    files = {str(Path(input_lock).resolve()): _sha(input_lock), str(archive): _sha(archive)}
    selected = []
    for i, case in enumerate(cases[:limit]):
        folder = root/'cases'/f'{i:03d}'; source = case.pop('source')
        packet = policy = request = control = None; subtype = 'unresolved'
        try:
            packet = build_structure_packet(source)
            subtype = annotate_lc_setup(_context_input(source['source_packet']))['subtype']
            first = min(clock(r['open_time']) for rows in packet['candles'].values() for r in rows)
            alignment = verify_candles(packet, read_archive(archive, first, clock(packet['decision_time'])))
            if alignment['status'] == 'verified':
                policy = practice_policy(packet)
                request = structure_request(packet, policy)
                control = mechanical_control(source, packet, policy)
        except (ValueError, KeyError, TypeError, OSError) as exc:
            alignment = dict(status='unavailable', candles_compared=0, errors=[type(exc).__name__+': '+str(exc)])
        item = dict(case, folder=f'cases/{i:03d}', source_status=alignment['status'], subtype=subtype)
        stored = dict(source=source, packet=packet, policy=policy, alignment=alignment, control=control)
        paths = [publish(folder/'input.json', stored)]
        if request is not None:
            paths.append(publish(folder/'request.json', request))
            item['request_bytes'] = len(_canonical(request).encode('ascii'))
            item['request_sha256'] = _sha(folder/'request.json')
        else:
            item.update(request_bytes=0, request_sha256=None)
        files[case['source_path']] = _sha(case['source_path'])
        files.update({str(p): _sha(p) for p in paths}); selected.append(item)
    # Covers the small research adapter dependency tree without changing frozen code.
    files.update({str(p.resolve()): _sha(p) for p in Path(__file__).parent.glob('*.py')})
    manifest = sealed(dict(version=VERSION, cases=selected, role_policy=ROLE_POLICY,
        archive=str(archive), archive_sha256=_sha(archive), input_lock=str(Path(input_lock).resolve()),
        files=files, exposure='exposed practice/development; not a holdout',
        archive_mapping='source labels bound to this archive; real preset: Binance USD-M BTCUSDT',
        execution_authorized=False))
    publish(root/'manifest.json', manifest)
    for case in selected:
        if case['source_status'] != 'verified':
            folder = root/case['folder']
            publish(folder/'terminal.json', sealed(dict(case_id=case['case_id'],
                status='source_unavailable', reason='source_alignment_failed', elapsed_seconds=None,
                net_pnl=None, files={str(folder/'input.json'): _sha(folder/'input.json')},
                execution_authorized=False)))
    return manifest


class PracticeRun:
    """One owner process from reserve through capture; exact host-reported bytes retained.

    Host metadata/delivery is recorded evidence, not provider attestation. Unknown
    model snapshot and billing stay unknown; contradictory model identity fails.
    """
    def __init__(self, directory, *, monotonic_ns=time.monotonic_ns):
        self.root = Path(directory).resolve(); self.clock = monotonic_ns
        self.runtime_id = uuid.uuid4().hex
        self.owner = (self.root/'owner.lock').open('a+')
        try:
            fcntl.flock(self.owner, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            self.owner.close(); raise ValueError('run already has an owner') from exc
        try:
            self.verify()
            # Any attempt without a terminal is incomplete after an owner restart.
            for case in self.manifest['cases']:
                folder = self.root/case['folder']
                if (folder/'reservation.json').exists() and not (folder/'terminal.json').exists():
                    self._failure(case, 'interrupted', 'owner_restarted')
        except Exception:
            self.close(); raise

    def __enter__(self): return self
    def __exit__(self, *args): self.close()

    def close(self):
        if not self.owner.closed:
            fcntl.flock(self.owner, fcntl.LOCK_UN); self.owner.close()

    def verify(self):
        if self.owner.closed: raise ValueError('owner closed')
        manifest = _verify_lock(self.root/'manifest.json')
        if manifest['version'] != VERSION or manifest['role_policy'] != ROLE_POLICY:
            raise ValueError('practice policy changed')
        _verify_lock(manifest['input_lock'])
        self.manifest = manifest
        return deepcopy(manifest)

    def _case(self, cid):
        for case in self.manifest['cases']:
            if case['case_id'] == cid: return case, self.root/case['folder']
        raise ValueError('unknown case')

    def status(self):
        self.verify(); terminals = {}; inflight = []
        for case in self.manifest['cases']:
            folder = self.root/case['folder']; cid = case['case_id']
            if (folder/'terminal.json').exists(): terminals[cid] = _verify_lock(folder/'terminal.json')['status']
            elif (folder/'reservation.json').exists(): inflight.append(cid)
        return dict(manifest_sha256=self.manifest['sha256'], terminals=terminals, inflight=inflight,
            pending=len(self.manifest['cases'])-len(terminals)-len(inflight),
            authorized=(self.root/'authorization.json').exists(),
            locked=(self.root/'terminal_lock.json').exists(),
            request_bytes={c['case_id']:c['request_bytes'] for c in self.manifest['cases']})

    def authorize(self, user_instruction, manifest_sha256):
        self.verify()
        if (not isinstance(user_instruction, str) or not user_instruction.strip()
                or manifest_sha256 != self.manifest['sha256']):
            raise ValueError('explicit authorization bound to current manifest required')
        return publish(self.root/'authorization.json', sealed(dict(
            manifest_sha256=manifest_sha256, user_instruction=user_instruction,
            max_calls=len(self.manifest['cases']), billed_credits=None, credit_cap_enforced=False)))

    def reserve(self, cid):
        self.verify(); case, folder = self._case(cid)
        auth = self.root/'authorization.json'
        if not auth.exists(): raise ValueError('market calls not authorized')
        if _verify_digest(auth)['manifest_sha256'] != self.manifest['sha256']:
            raise ValueError('authorization binding changed')
        if (folder/'reservation.json').exists(): raise ValueError('one attempt only')
        if ((folder/'terminal.json').exists() or (self.root/'terminal_lock.json').exists()
                or case['source_status'] != 'verified'): raise ValueError('case cannot dispatch')
        if self.status()['inflight']: raise ValueError('one in-flight assessment only')
        reservation = sealed(dict(case_id=cid, runtime_id=self.runtime_id, started_ns=self.clock(),
            manifest_sha256=self.manifest['sha256'], request_sha256=case['request_sha256']))
        publish(folder/'reservation.json', reservation)
        raw = (folder/'request.json').read_bytes()
        return dict(case_id=cid, request_path=str(folder/'request.json'), request_hex=raw.hex(),
                    request_sha256=case['request_sha256'], requested_model=ROLE_POLICY['requested_model'],
                    requested_effort=ROLE_POLICY['requested_effort'])

    def _attempt(self, cid):
        self.verify(); case, folder = self._case(cid)
        if (folder/'terminal.json').exists() or (self.root/'terminal_lock.json').exists():
            raise ValueError('terminal already recorded')
        r = _verify_digest(folder/'reservation.json')
        if r['runtime_id'] != self.runtime_id: raise ValueError('interrupted owner')
        return case, folder, r

    def attach(self, cid, agent_id):
        _, folder, _ = self._attempt(cid)
        if not isinstance(agent_id, str) or not agent_id.strip(): raise ValueError('role identity required')
        for path in self.root.glob('cases/*/dispatch.json'):
            if path.parent != folder and _verify_digest(path)['agent_id'] == agent_id:
                raise ValueError('fresh role identity required')
        publish(folder/'dispatch.json', sealed(dict(case_id=cid, agent_id=agent_id,
            requested_model=ROLE_POLICY['requested_model'], requested_effort=ROLE_POLICY['requested_effort'])))

    def _evaluate(self, case, folder, capture):
        r = _verify_digest(folder/'reservation.json'); dispatched = _verify_digest(folder/'dispatch.json')
        raw = bytes.fromhex(capture['raw_hex']); delivered = bytes.fromhex(capture['delivered_hex'])
        expected = (folder/'request.json').read_bytes(); meta = capture['metadata']
        stored = _load(folder/'input.json')
        try: decoded = raw.decode('utf-8')
        except UnicodeError: decoded = '{'
        checked = validate_structure_proposal(stored['source'], stored['packet'], decoded, stored['policy'])
        elapsed = (capture['ended_ns']-r['started_ns'])/1e9
        status = checked['status']
        fields = {'agent_id','requested_model','requested_effort','observed_model','observed_snapshot','delivery_complete'}
        valid_meta = (isinstance(meta, dict) and set(meta) == fields
            and meta['agent_id'] == dispatched['agent_id'] and meta['delivery_complete'] is True
            and meta['requested_model'] == ROLE_POLICY['requested_model']
            and meta['requested_effort'] == ROLE_POLICY['requested_effort']
            and meta['observed_model'] in (None, ROLE_POLICY['requested_model'])
            and (meta['observed_snapshot'] is None or
                 isinstance(meta['observed_snapshot'], str) and bool(meta['observed_snapshot'].strip())))
        if not valid_meta or delivered != expected: status = 'invalid_transport'
        if capture['runtime_id'] != r['runtime_id'] or not math.isfinite(elapsed) or elapsed < 0:
            status = 'interrupted'; elapsed = None
        elif elapsed > ROLE_POLICY['timeout_seconds']: status = 'timeout'
        if capture['raw_sha256'] != hashlib.sha256(raw).hexdigest(): raise ValueError('raw capture changed')
        return dict(status=status, elapsed_seconds=elapsed, validation=checked)

    def capture(self, cid, raw_bytes, metadata, delivered_bytes):
        case, folder, _ = self._attempt(cid)
        if not isinstance(raw_bytes, bytes) or not isinstance(delivered_bytes, bytes):
            raise ValueError('exact raw and delivered bytes required')
        # Receipt timestamp is taken before hashing, validation and disk work.
        captured = sealed(dict(raw_hex=raw_bytes.hex(), raw_sha256=hashlib.sha256(raw_bytes).hexdigest(),
            delivered_hex=delivered_bytes.hex(), metadata=deepcopy(metadata), ended_ns=self.clock(),
            runtime_id=self.runtime_id))
        publish(folder/'capture.json', captured)
        evaluated = self._evaluate(case, folder, captured)
        return self._terminal(case, evaluated['status'], None, evaluated['elapsed_seconds'])

    def _terminal(self, case, status, reason, elapsed):
        folder = self.root/case['folder']
        files = {str(p):_sha(p) for p in (folder/'reservation.json', folder/'dispatch.json', folder/'capture.json') if p.exists()}
        t = sealed(dict(case_id=case['case_id'], status=status, reason=reason, elapsed_seconds=elapsed,
                        net_pnl=None, files=files, execution_authorized=False))
        publish(folder/'terminal.json', t)
        return t

    def _failure(self, case, status, reason):
        return self._terminal(case, status, reason, None)

    def fail(self, cid, reason):
        case, _, _ = self._attempt(cid)
        if not isinstance(reason, str) or not reason.strip(): raise ValueError('failure reason required')
        return self._failure(case, 'external_failure', reason)

    def mark_not_run(self, cid, reason):
        self.verify(); case, folder = self._case(cid)
        if ((folder/'reservation.json').exists() or (self.root/'terminal_lock.json').exists()
                or not isinstance(reason, str) or not reason.strip()): raise ValueError('cannot mark not-run')
        return self._failure(case, 'not_run', reason)

    def lock_terminals(self):
        self.verify(); terminals = {}; files = {}
        for case in self.manifest['cases']:
            folder = self.root/case['folder']; path = folder/'terminal.json'
            if not path.exists(): raise ValueError('all case terminals required before reveal')
            t = _verify_lock(path)
            if t['case_id'] != case['case_id']: raise ValueError('terminal case changed')
            if t['reason'] is None:
                captured = _verify_digest(folder/'capture.json')
                check = self._evaluate(case, folder, captured)
                if t['status'] != check['status'] or t['elapsed_seconds'] != check['elapsed_seconds']:
                    raise ValueError('terminal differs from original capture')
            elif (t['status'] not in ('source_unavailable','interrupted','external_failure','not_run')
                    or not isinstance(t['reason'], str) or not t['reason'].strip()
                    or t['elapsed_seconds'] is not None): raise ValueError('invalid failure terminal')
            if t['net_pnl'] is not None or t['execution_authorized'] is not False:
                raise ValueError('terminal cannot invent outcome or execution authority')
            terminals[case['case_id']] = t; files[str(path)] = _sha(path)
        locked = sealed(dict(version=VERSION, manifest_sha256=self.manifest['sha256'],
            terminals=terminals, files=files, outcome_reveal_authorized=True, execution_authorized=False))
        publish(self.root/'terminal_lock.json', locked)
        return locked

    def assert_reveal_allowed(self):
        if not (self.root/'terminal_lock.json').exists(): raise ValueError('terminal lock required')
        locked = _verify_lock(self.root/'terminal_lock.json')
        if locked != self.lock_terminals(): raise ValueError('terminal lock changed')
        return locked
