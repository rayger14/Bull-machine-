"""Offline source-bound recognition exam, not entry/portfolio/P&L replay.

Observe native raw events -> state -> features -> directional score. The only
sizing execution is the exact hash-checked phase boost AST with inert inputs.
Run in its own single-threaded process (socket patches and SIGALRM are global).
"""
from __future__ import annotations

import argparse
import ast
from contextlib import ExitStack
from copy import deepcopy
from dataclasses import asdict
from enum import Enum
import hashlib
from importlib import metadata
import json
import logging
import math
from pathlib import Path
import platform
import signal
import time
from types import SimpleNamespace
from unittest.mock import patch

from scripts.research.wyckoff_recognition_cases import (
    COLUMNS, canonical, digest, read_packet, validate_packet)

ROOT = Path(__file__).resolve().parents[2]
RUNNER = ROOT / 'bin/live/v11_shadow_runner.py'
CONFIG = ROOT / 'configs/champion_paper.json'
# 2026-10-05 range repair: explicit same-parent phase justification guard.
# Historical receipts retain the old AST binding; new runs require a new seal.
BRANCH_AST_HASH = 'c76df4c78e406c7993a5ff7eb60dda25468531cbd13f566bf2c92fddd15ca222'
EVENTS = ('sc','bc','ar','as','st','sos','sow','spring_a','spring_b','ut','utad','lps','lpsy')


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def clean(value):
    """Strict JSON, preserving missingness rather than inventing zero evidence."""
    if isinstance(value,Enum):
        return clean(value.value)
    if isinstance(value, dict):
        return {str(k):clean(v) for k,v in value.items()}
    if isinstance(value, (tuple,list)):
        return [clean(v) for v in value]
    if hasattr(value, 'item'):
        return clean(value.item())
    if value is None or isinstance(value,(str,bool,int)):
        return value
    if isinstance(value,float):
        return value if math.isfinite(value) else None
    if hasattr(value,'isoformat'):
        return value.isoformat()
    raise TypeError(f'Unsupported trace value: {type(value).__name__}')


class SizingProbe:
    def __init__(self, path, expected_hash):
        if file_hash(path) != expected_hash:
            raise ValueError('sizing source hash drift')
        tree = ast.parse(Path(path).read_text())
        classes = [n for n in tree.body if isinstance(n,ast.ClassDef) and n.name == 'V11ShadowRunner']
        methods = [n for c in classes for n in c.body if isinstance(n,ast.FunctionDef) and n.name == 'process_bar']
        branches = [n for m in methods for n in ast.walk(m) if isinstance(n,ast.If)
                    and any(isinstance(x,ast.Constant) and x.value == 'wyckoff_phase_boost'
                            for x in ast.walk(n.test))]
        if len(classes) != 1 or len(methods) != 1 or len(branches) != 1:
            raise ValueError('unique phase sizing branch required')
        node = branches[0]
        self.ast_hash = hashlib.sha256(ast.dump(node,include_attributes=False).encode()).hexdigest()
        if self.ast_hash != BRANCH_AST_HASH:
            raise ValueError('sizing AST shape/hash drift')
        self.source_hash = expected_hash
        self.lines = [node.lineno,node.end_lineno]
        self.code = compile(ast.fix_missing_locations(ast.Module(body=[node],type_ignores=[])),
                            str(path),'exec')

    def evaluate(self, direction, phase, config, features=None):
        from engine.wyckoff.range_evidence import phase_c_sizing_eligible
        if direction not in ('long','short'):
            raise ValueError('unknown sizing direction')
        intent = SimpleNamespace(allocated_size_pct=.02,
                                 signal=SimpleNamespace(direction=direction,archetype_id='recognition_probe'))
        meta = {**(features or {}), 'wyckoff_phase_dir':phase,
                'sizing_boosts':{'multiplier':1.0,'capex_mult':1.0,'reasons':[]}}
        scope = {'__builtins__':{'isinstance':isinstance,'str':str},
                 'self':SimpleNamespace(config=deepcopy(config)), 'intent':intent,'sig_meta':meta,
                 'logger':SimpleNamespace(info=lambda *_: None),
                 'phase_c_sizing_eligible':phase_c_sizing_eligible}
        exec(self.code,scope,scope)
        return {'conditional_only':True,'allocated_ratio':intent.allocated_size_pct/.02,
                **meta['sizing_boosts']}


def trace_prefix(candles, n, observe=True):
    if type(n) is not int or not 1 <= n <= len(candles):
        raise ValueError('invalid prefix')
    # Import and every actual-source operation stay inside the run deadline.
    import pandas as pd
    from scripts.research.live_feature_replay import LiveFeatureProcessor, deny_network
    with deny_network():
        from engine.wyckoff import events
        from engine.archetypes.archetype_instance import ArchetypeInstance
        processor = LiveFeatureProcessor('1h')
        frame = pd.DataFrame(candles[:n],columns=COLUMNS)
        frame.index = pd.to_datetime(frame.pop('timestamp'),utc=True)
        processor.fc.ingest_candles(frame)
        traces, active = {}, []
        original_process = events.WyckoffStateMachine.process_bar
        original_detect = processor.module.detect_all_wyckoff_events

        def process(sm, bar_idx, row, raw_events, event_metadata=None):
            before = sm.parent_snapshot()
            range_before = asdict(sm.range_ref)
            answer = original_process(sm,bar_idx,row,raw_events,event_metadata=event_metadata)
            if active:
                active[-1]['rows'].append({
                    'i':bar_idx,'raw':[k for k in EVENTS if raw_events.get(k,False)],
                    'validated':[k for k in EVENTS if answer[0].get(k,False)],
                    'phase':sm.get_phase_dir(),'state':sm.state.value,
                    'parent_before':before,'parent_after':sm.parent_snapshot(),
                    'range_before':range_before,'range':asdict(sm.range_ref),'ohlc_volume_z':dict(row),
                    'structure':deepcopy(sm.structural_evidence),
                    'confidence_modifiers':answer[1]})
            return answer

        def detect(df, *args, **kwargs):
            cfg = kwargs.get('cfg') or {}
            tf = cfg.get('timeframe','unknown')
            if tf in traces:
                raise ValueError('duplicate timeframe observation')
            rec = {'cfg':deepcopy(cfg),'rows':[]}
            traces[tf] = rec
            active.append(rec)
            try:
                result = original_detect(df,*args,**kwargs)
                if len(rec['rows']) != len(result):
                    raise ValueError('state trace missing rows')
                for r,(ts,series) in zip(rec['rows'],result.iterrows()):
                    r['timestamp'] = ts.isoformat()
                    r['volume'] = series['volume']
                    r['confidence'] = {e:series[f'wyckoff_{e}_confidence'] for e in EVENTS
                                       if e in r['raw'] or e in r['validated']}
                    r['provenance'] = {k:v for k,v in series.items()
                                       if k.startswith('wyckoff_') and k.endswith(
                                           ('candidate_index','candidate_extreme','prior_swept_boundary',
                                            'candidate_parent_id','evidence_status','available_at'))
                                       and v is not None and not pd.isna(v)}
                return result
            finally:
                active.pop()

        with ExitStack() as stack:
            if observe:
                stack.enter_context(patch.object(events.WyckoffStateMachine,'process_bar',process))
                stack.enter_context(patch.object(processor.module,'detect_all_wyckoff_events',detect))
            features = processor.fc._wyckoff_features()
        scores = {}
        for direction in ('long','short'):
            instance = object.__new__(ArchetypeInstance)
            instance.direction = direction
            scores[direction] = instance._get_wyckoff_score(features)
        if observe and (set(traces) != {'1h','4h','1d'} or any(
                features.get(p+'wyckoff_evidence_status') != 'available' for p in ('','tf4h_','tf1d_'))):
            raise ValueError('missing/error native timeframe evidence')
        return clean({'n':n,'as_of':(frame.index[-1]+pd.Timedelta(hours=1)).isoformat(),
                      'features':features,'scores':scores,'timeframes':traces})


def validate_review(packet, review):
    validate_packet(packet)
    if (review.get('schema') != 'wyckoff-recognition-review-v1' or review.get('approved') is not True
            or review.get('independent_source_only') is not True or not review.get('reviewer')
            or review.get('packet_sha256') != digest(packet)):
        raise ValueError('unapproved, non-independent or wrong-hash review')
    cards = review.get('cases',[])
    if len(cards) != 12 or {c['id'] for c in cards} != {c['id'] for c in packet['cases']}:
        raise ValueError('review must account for all twelve cases')
    if any(c.get('status') not in ('approved','unresolved') or not c.get('interpretation') for c in cards):
        raise ValueError('invalid review status')


def bindings():
    paths = set((ROOT/'engine').rglob('*.py')) | set((ROOT/'bin/live').rglob('*.py'))
    paths.update([Path(__file__),CONFIG,ROOT/'scripts/research/wyckoff_recognition_cases.py',
                  ROOT/'scripts/research/live_feature_replay.py',ROOT/'scripts/research/replay_clock.py',
                  ROOT/'docs/superpowers/specs/2026-10-05-wyckoff-recognition-exam-design.md',
                  ROOT/'docs/superpowers/plans/2026-10-05-wyckoff-recognition-exam.md',
                  ROOT/'reports/Wyckoff recognition edge cases.md'])
    paths.update((ROOT/'research_notes/Wyckoff recognition edge cases').glob('*.md'))
    model = ROOT/'models/logistic_regime_v4_no_funding_stratified.pkl'
    paths.add(model)
    return {str(p.relative_to(ROOT)):file_hash(p) if p.exists() else None for p in sorted(paths)}


def environment():
    versions = {}
    for name in ('pandas','numpy','TA-Lib','scipy','joblib','scikit-learn','requests','yfinance'):
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = None
    return {'python':platform.python_version(),'packages':versions}


def make_seal(packet, review):
    validate_review(packet,review)
    return {'schema':'wyckoff-recognition-seal-v1','packet_sha256':digest(packet),
            'review_sha256':digest(review),'files':bindings(),'environment':environment(),
            'sizing_ast_sha256':BRANCH_AST_HASH,'grading':grading_ledger(packet)}


def verify_seal(packet, review, seal):
    validate_review(packet,review)
    if seal.get('schema') != 'wyckoff-recognition-seal-v1' or seal.get('packet_sha256') != digest(packet):
        raise ValueError('packet seal mismatch')
    if seal.get('review_sha256') != digest(review):
        raise ValueError('review seal mismatch')
    if seal.get('files') != bindings():
        raise ValueError('source binding drift')
    if seal.get('environment') != environment() or seal.get('sizing_ast_sha256') != BRANCH_AST_HASH:
        raise ValueError('environment/AST drift')
    if seal.get('grading') != grading_ledger(packet):
        raise ValueError('grading policy drift')


def _labels(keys):
    return {'spring' if k in ('spring_a','spring_b') else 'upthrust' if k in ('ut','utad') else k for k in keys}


def grading_ledger(packet):
    """Frozen source-landmark interpretation, never learned from detector outputs."""
    starts = {'W05':645,'W06':645,'W07':649,'W08':639,'W09':581,'W10':581}
    return {'schema':'wyckoff-recognition-grading-v1',
            'authority':'Source-only v2 annotation plus delegated quant pre-run window ruling',
            'rules':[
                'Positive milestone must follow its source landmark, not an earlier range-cycle label.',
                'Spring/upthrust must reference source candidate index through actual delayed metadata.',
                'Parent association requires established matching context and exact synthetic source bounds; numeric tolerance 1e-6 is serialization tolerance, not a market gate.',
                'A missing parent match is a representation gap, not proof an emitted local event is false.',
                'C/D and generic event labels are review flags, not alone proof of false confirmation.',
                'Post-invalidation positive score or size is unassociated unless actual evidence proves parent attribution.',
                'Review time, geometry and role against this ledger; preserve disagreements; no result-dependent label edits.',
                'No recognition outcome establishes entries, profitability or live readiness.'],
            'cases':[{'id':c['id'],'target_parent':c['range'],
                      'required_windows':{e:{'min_i':644 if e in ('spring','upthrust') else
                                            652 if e in ('sos','sow') else 660,
                                            'max_i':c['checkpoints'][-1]['n']-1,
                                            'candidate_i':644 if e in ('spring','upthrust') else None}
                                          for e in c['required_milestones']},
                      'claim_start_n':starts.get(c['id'],c['checkpoints'][0]['n']),
                      'post_failure_start_n':649 if c['id']=='W07' else 639 if c['id']=='W08' else None,
                      'checkpoints':[{'n':cp['n'],'stage':cp['stage'],
                                      'available_through_i':cp['n']-1} for cp in c['checkpoints']],
                      'prohibited_roles':c['prohibited_claims'],
                      'source_interpretation':c['interpretation']}
                     for c in packet['cases']]}


def _milestone_hit(row, event, window):
    if event not in _labels(row['validated']) or not window['min_i'] <= row['i'] <= window['max_i']:
        return False
    if window['candidate_i'] is not None:
        return any(row.get('provenance',{}).get(f'wyckoff_{k}_candidate_index') == window['candidate_i']
                   for k in row['validated'] if event in _labels([k]))
    return True


def _parent_matches(row, case):
    parent,geometry = row.get('parent_before',{}),row.get('range_before',{})
    accum = case['direction']=='long'
    context = 'accumulation' if accum else 'distribution'
    low = geometry.get('sc_low' if accum else 'as_low')
    high = geometry.get('ar_high' if accum else 'bc_high')
    return (parent.get('status') == 'established' and parent.get('context') == context
            and low is not None and high is not None and case['range']['low'] is not None
            and abs(low-case['range']['low']) < 1e-6 and abs(high-case['range']['high']) < 1e-6)


def summarize_case(case, traces, review):
    last = traces[-1]
    native = last['timeframes']['1h']
    policy = grading_ledger({'cases':[case]})['cases'][0]
    rows = [r for r in native['rows'] if r['i'] >= policy['claim_start_n']-1]
    labels = set().union(*(_labels(r['validated']) for r in rows)) if rows else set()
    hits = {e:[r for r in native['rows'] if _milestone_hit(r,e,w)]
            for e,w in policy['required_windows'].items()}
    missing = [e for e,v in hits.items() if not v]
    parent_gaps = [e for e,v in hits.items() if v and not any(_parent_matches(r,case) for r in v)]
    label_flags = [{'i':r['i'],'label':e,'parent':r.get('parent_before'),
                    'provenance':r.get('provenance',{})} for r in rows
                   for e in sorted(_labels(r['validated'])) if e in case['prohibited_claims']]
    # Semantic disagreement needs geometry/role adjudication; don't manufacture
    # either success or contradiction from a generic label or score alone.
    contradictions = []
    phases = [t['features'].get('wyckoff_phase_dir') for t in traces]
    phase_flags = [{'n':t.get('n'),'phase':t['features'].get('wyckoff_phase_dir')} for t in traces
                   if t.get('n',0) >= policy['claim_start_n'] and
                   any(x in case['prohibited_claims'] for x in ('confirmed_C_or_D','active_long_after_invalidation'))
                   and t['features'].get('wyckoff_phase_dir') in ('C_accum','D_accum')]
    later = [t for t in traces if policy['post_failure_start_n'] is not None
             and t.get('n',0) >= policy['post_failure_start_n']]
    stale_scores = [{'n':t['n'],**t['scores']} for t in later if t['scores'].get('long',0)>0]
    stale_boosts = [{'n':t['n'],**t['sizing']['long']} for t in later
                    if t['sizing']['long']['allocated_ratio']>1]
    if review['status'] == 'unresolved':
        status = 'unscored'
    elif missing:
        status = 'missed_positive'
    elif parent_gaps:
        status = 'parent_association_unresolved'
    elif label_flags or phase_flags:
        status = 'requires_semantic_adjudication'
    elif case['category'] == 'positive':
        status = 'native_labels_present_requires_geometry_review'
    else:
        status = 'no_prohibited_claim_observed_not_full_recognition'
    unrepresented = [x for x in case['prohibited_claims'] if x in (
        'parent_range_escape','lps_after_parent_escape','confirmed_nested_accumulation')]
    if stale_scores:
        unrepresented.append('score_parent_lineage')
    if stale_boosts:
        unrepresented.append('sizing_parent_lineage')
    return {'id':case['id'],'name':case['name'],'category':case['category'],
            'source_status':review['status'],'source_interpretation':review['interpretation'],
            'recognition_status':status,'native_labels':sorted(labels),'missing_milestones':missing,
            'matched_milestone_indices':{e:[r['i'] for r in v] for e,v in hits.items()},
            'parent_association_gaps':parent_gaps,'label_claims_requiring_review':label_flags,
            'phase_claims_requiring_review':phase_flags,
            'unattributed_post_failure_scores':stale_scores,'unattributed_post_failure_boosts':stale_boosts,
            'contradictions':contradictions,'phases_at_checkpoints':phases,
            'full_m2_enabled':native['cfg'].get('sm_m2_path',False),
            'context_only_m2':native['cfg'].get('sm_m2_context_only',False),
            'unrepresented_distinctions':unrepresented,
            'scores_at_checkpoints':[t['scores'] for t in traces],
            'conditional_sizing_at_checkpoints':[t['sizing'] for t in traces],
            'economic_progression':'blocked' if missing or parent_gaps or label_flags or phase_flags
                                   or contradictions or unrepresented or status == 'unscored'
                                   else 'not_authorized_by_recognition_exam'}


class DeadlineExceeded(BaseException):
    """Not swallowed by detector's fallback exception handlers."""


class BoundedOutput:
    def __init__(self,path,seconds=600,max_bytes=100*1024*1024):
        if not 0 < seconds <= 600 or type(max_bytes) is not int or not 0 < max_bytes <= 100*1024*1024:
            raise ValueError('invalid resource bounds')
        self.path,self.seconds,self.max_bytes = Path(path),seconds,max_bytes
        self.used,self.artifacts = 0,{}

    def __enter__(self):
        self.started = time.monotonic()
        self.path.mkdir(parents=True,exist_ok=False)
        self.old_handler = signal.getsignal(signal.SIGALRM)
        self.old_timer = signal.getitimer(signal.ITIMER_REAL)
        if self.old_timer[0]:
            raise ValueError('cannot nest process deadline')
        signal.signal(signal.SIGALRM,self._alarm)
        signal.setitimer(signal.ITIMER_REAL,self.seconds)
        return self

    def _alarm(self,*_):
        raise DeadlineExceeded('exam deadline exceeded')

    def check(self):
        if time.monotonic()-self.started > self.seconds:
            raise DeadlineExceeded('exam deadline exceeded')

    def write(self,name,value):
        self.check()
        if Path(name).name != name or name in ('.','..'):
            raise ValueError('artifact path must be a basename')
        data = canonical(value)
        if self.used+len(data) > self.max_bytes:
            raise ValueError('artifact byte budget exceeded')
        with (self.path/name).open('xb') as f:
            f.write(data)
        self.used += len(data)
        self.artifacts[name] = hashlib.sha256(data).hexdigest()

    def __exit__(self,*_):
        signal.setitimer(signal.ITIMER_REAL,0)
        signal.signal(signal.SIGALRM,self.old_handler)


def run_exam(packet_path, review_path, seal_path, output):
    with BoundedOutput(output) as out:
        try:
            packet = read_packet(packet_path)
            review = json.loads(Path(review_path).read_text())
            seal = json.loads(Path(seal_path).read_text())
            verify_seal(packet,review,seal)
            out.write('seal.json',seal)
            probe = SizingProbe(RUNNER,seal['files'][str(RUNNER.relative_to(ROOT))])
            config = json.loads(CONFIG.read_text())
            summaries,witnesses = [],[]
            by_id = {c['id']:c for c in review['cases']}
            for case in packet['cases']:
                traces = []
                for cp in case['checkpoints']:
                    out.check()
                    tr = trace_prefix(case['candles'],cp['n'])
                    tr['sizing'] = {d:probe.evaluate(d,tr['features'].get('wyckoff_phase_dir'),config,tr['features'])
                                    for d in ('long','short')}
                    traces.append(tr)
                # Sample the first decision checkpoint with a nonempty future tail.
                cp = case['checkpoints'][0]
                witness = trace_prefix(case['candles'][:cp['n']],cp['n'])
                changed = deepcopy(case['candles'])
                for row in changed[cp['n']:]:
                    row[1:5] = [v*1.5 for v in row[1:5]]
                    row[5] *= 3
                perturbed = trace_prefix(changed,cp['n'])
                baseline = {k:v for k,v in traces[0].items() if k != 'sizing'}
                hashes = [digest(v) for v in (baseline,witness,perturbed)]
                if len(set(hashes)) != 1:
                    raise ValueError('sampled prefix witness differs')
                witnesses.append({'id':case['id'],'n':cp['n'],'trace_hashes':hashes,
                                  'scope':'sampled prefix boundary only; not exhaustive causality'})
                out.write(case['id']+'.json',{'id':case['id'],'traces':traces})
                summaries.append(summarize_case(case,traces,by_id[case['id']]))
                print(case['id'],summaries[-1]['recognition_status'],flush=True)
            verify_seal(packet,review,seal)
            out.write('summary.json',{'schema':'wyckoff-recognition-result-v1','cases':summaries,
                      'witnesses':witnesses,'sizing_source':{'file':str(RUNNER.relative_to(ROOT)),
                      'hash':probe.source_hash,'ast_hash':probe.ast_hash,'lines':probe.lines},
                      'not_tested':['entry_eligibility','portfolio_sizing','profit','minute_execution','all17_archetypes']})
            out.write('receipt.json',{'status':'complete','seal_sha256':digest(seal),
                      'elapsed_seconds':time.monotonic()-out.started,'bytes_before_receipt':out.used,
                      'artifacts':dict(out.artifacts),'cases':12})
        except BaseException as exc:
            # A deadline cannot be swallowed as a valid empty result. Failure
            # marker remains within budget; stdout is fallback if already full.
            signal.setitimer(signal.ITIMER_REAL,0)
            failure = {'status':'failed','error_type':type(exc).__name__,'error':str(exc)[:400]}
            data = canonical(failure)
            if out.used+len(data) <= out.max_bytes:
                with (out.path/'failure.json').open('xb') as f:
                    f.write(data)
            print(json.dumps(failure),flush=True)
            raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=['seal','run'])
    parser.add_argument('--packet',required=True)
    parser.add_argument('--review',required=True)
    parser.add_argument('--seal',required=True)
    parser.add_argument('--output')
    args = parser.parse_args()
    logging.disable(logging.CRITICAL)
    if args.action == 'seal':
        packet = read_packet(args.packet)
        review = json.loads(Path(args.review).read_text())
        seal = make_seal(packet,review)
        with Path(args.seal).open('xb') as f:
            f.write(canonical(seal))
        print(json.dumps({'seal_sha256':digest(seal),'files':len(seal['files'])}))
    else:
        if not args.output:
            parser.error('--output required for run')
        run_exam(args.packet,args.review,args.seal,args.output)


if __name__ == '__main__':
    main()
