"""Source-only launch, blind semantic packets and explicit economic boundaries."""
from collections import Counter
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import resource
import signal
import sys
import time

import pandas as pd

from scripts.research.run_thesis_census import Output
from scripts.research.support_reaction import assess, origin_record, policy, verify_record
from scripts.research.support_reaction_replay import BLOCKS, folds
from scripts.research.thesis_contract import clock, seal, signed, verify
from scripts.research.thesis_source import ROOT, ARCHIVE, sha, verify_files

SOURCE = ROOT/'results/thesis_census_2026_10_03/source_v1/source.json'
RECEIPT = SOURCE.with_name('receipt.json')
PINNED = {str(SOURCE): 'b27440a9247ea93b77458f08fc13009ab56679a43bbfc66aba3a86d43039730b',
          str(RECEIPT): '1c13cc6eabc94ec8272f92998738242fb94111536f341a833772815ff1328eb7'}
MAX_SECONDS, MAX_BYTES = 600, 268435456
CLAIMS = ('location', 'phase', 'recovery', 'support', 'minute_anchors', 'invalidation', 'room', 'action')


def benchmark_roster(origins):
    if len({o['id'] for o in origins}) != len(origins):
        raise ValueError('duplicate raw roster identity')
    roster = []
    for block, (a, b) in enumerate(zip(BLOCKS[:-1], BLOCKS[1:])):
        candidates = [o for o in origins if clock(a) <= clock(o['origin']['available_at']) < clock(b)]
        rank = lambda o: hashlib.sha256(o['id'].encode()).hexdigest()
        if len(candidates) < 3:
            raise ValueError('fewer than three raw episodes in benchmark block')
        roster.extend(dict(id=o['id'], block=block, rank_hash=rank(o)) for o in sorted(candidates, key=rank)[:3])
    return roster


def blind_packet(record):
    verify_record(record)
    cutoff = clock(record['decisions']['B']['at'])
    if record['decisions']['B']['status'] == 'pending':
        raise ValueError('natural benchmark needs observed terminal source cutoff')
    # Exclude selected/derived pivots and ALL policy labels/decisions. Include
    # raw hourly/minute constituents so a reviewer can disagree independently.
    observations = {k: deepcopy(e) for k, e in record['catalog'].items()
                    if e['kind'] in ('parent', 'candle', 'volume') and clock(e['available_at']) <= cutoff}
    raw = record['origin']
    daily = deepcopy(raw['daily_context'])
    for key in ('candle', 'parent'):
        e = daily.get(key)
        if e:
            if clock(e['available_at']) > cutoff: raise ValueError('future daily context')
            observations[e['id']] = deepcopy(e)
    return signed(dict(schema='support-blind-source-v1', episode_id=raw['id'], cutoff=cutoff.isoformat(),
                       stream_id=raw['stream_id'], parent=deepcopy(raw['parent']),
                       origin_candle_id=raw['origin']['id'], daily_context=daily,
                       original_stop=raw['original_stop'], atr4h=raw['atr4h'], observations=observations,
                       context_limit='pivot range plus available candle context; not a full phase history',
                       rubric=dict(required_claims=list(CLAIMS), source_citations_required=True,
                                   phase_certainty='unclassified_or_uncertain_unless_independently_supported',
                                   no_outcomes_or_project_history=True), execution_authorized=False))


def validate_annotation(packet, annotation):
    verify(packet)
    allowed = {'packet_seal', 'cutoff', 'status', 'citations', *CLAIMS}
    if set(annotation) != allowed or annotation['packet_seal'] != packet['seal'] or annotation['cutoff'] != packet['cutoff']:
        raise ValueError('annotation fields/packet binding mismatch')
    if annotation['status'] not in ('complete', 'unknown', 'disputed'):
        raise ValueError('invalid annotation status')
    cited = []
    for name in CLAIMS:
        claim = annotation[name]
        if not isinstance(claim, dict) or set(claim) != {'state', 'text', 'citations'}:
            raise ValueError('claim fields required')
        states = ('unclassified', 'uncertain', 'disputed') if name == 'phase' else (
            ('enter', 'reject', 'wait', 'unknown', 'disputed') if name == 'action' else ('known', 'unknown', 'disputed'))
        if claim['state'] not in states or not isinstance(claim['text'], str) or not claim['text'].strip():
            raise ValueError('unsupported claim certainty/state')
        if not isinstance(claim['citations'], list) or (not claim['citations'] and claim['state'] not in ('unknown', 'uncertain', 'disputed')):
            raise ValueError('source citation required')
        cited.extend(claim['citations'])
    if not isinstance(annotation['citations'], list) or set(cited) != set(annotation['citations']):
        raise ValueError('annotation citation union mismatch')
    for cid in cited:
        if cid not in packet['observations'] or clock(packet['observations'][cid]['available_at']) > clock(packet['cutoff']):
            raise ValueError('foreign/future annotation evidence')
    return signed(dict(packet_seal=packet['seal'], annotation_seal=seal(annotation),
                       status='valid_schema_not_verified_judgment', independent_agreement_measured=False))


def explain(record, arm):
    d = record['decisions'][arm]; raw = record['origin']; s = record['structure']
    return dict(episode_id=raw['id'], arm=arm, status=d['status'], reason=d['reason'], at=d['at'],
                range=dict(id=raw['parent']['id'], low=raw['parent']['range_low'], high=raw['parent']['range_high'],
                           kind='pivot_range', phase='unclassified'),
                recovery_candle=s['recovery'], support_candle=s['support'],
                demand=(record['demand'] or {}).get('state', 'not_observed'),
                supply=(record['supply'] or {}).get('state', 'not_observed'),
                minute_high=s['child_high'], minute_low=s['child_low'], trigger=s['trigger'],
                original_stop=raw['original_stop'], room_r=d.get('room_r'), evidence_ids=d['citations'],
                interpretation='Local support reaction; no claim of confirmed accumulation or institutional intent.')


def summarize(records, old_packets):
    if len(records) != len(old_packets) or {r['origin']['id'] for r in records} != {p['id'] for p in old_packets}:
        raise ValueError('summary lost raw population')
    arms = {}
    for arm in 'BC':
        decisions = [r['decisions'][arm] for r in records]
        selected = [r for r in records if r['decisions'][arm]['status'] == 'intent']
        arms[arm] = dict(statuses=dict(Counter(d['status'] for d in decisions)),
                         reasons=dict(Counter(d['reason'] for d in decisions)), intents=len(selected),
                         origin_months=len({clock(r['origin']['origin']['available_at']).strftime('%Y-%m') for r in selected}),
                         evidence=dict(demand=dict(Counter((r['demand'] or {}).get('state', 'not_observed') for r in records)),
                                       supply=dict(Counter((r['supply'] or {}).get('state', 'not_observed') for r in records))))
    for r in records:
        verify_record(r)
        b, c = (r['decisions'][a] for a in 'BC')
        if c['status'] == 'intent' and (b['status'] != 'intent' or any(b[k] != c[k] for k in ('at', 'trigger_id', 'origin_id'))):
            raise ValueError('C is not the same-trigger B subset')
    return signed(dict(schema='support-source-summary-v1', policy_seal=seal(policy()),
                       raw_episodes=len(records), old_A_intents=sum(p['entry_intents']['thesis'] is not None for p in old_packets),
                       arms=arms, parent_lineages=len({r['origin']['parent']['lineage_id'] for r in records}),
                       research_floor='50 closed fills in12 origin months; not a power guarantee',
                       independent_annotations_complete=False, economic_outcomes_computed=False,
                       edge_demonstrated=False, pristine_holdout=False, execution_authorized=False))


def bounded_source(output, files, compute, *, max_bytes=MAX_BYTES, prepare=None):
    if not 32768 <= max_bytes <= MAX_BYTES:
        raise ValueError('invalid output bound')
    out = Output(output, limit=max_bytes)
    start = time.monotonic()
    launch = signed(dict(schema='support-launch-v1', policy_seal=seal(policy()), files=files,
                         maximum_seconds=MAX_SECONDS, maximum_output_bytes=max_bytes, stage='source_only',
                         economic_outcomes_computed=False, execution_authorized=False))
    def timeout(signum, frame): raise TimeoutError('support source runtime cap exceeded')
    previous = signal.signal(signal.SIGALRM, timeout)
    signal.alarm(MAX_SECONDS)
    try:
        out.write('launch.json', launch)
        verify_files(files)
        if prepare is not None:
            extra = prepare()
            if any(k in files and files[k] != v for k, v in extra.items()):
                raise ValueError('conflicting prepared dependency')
            files = {**files, **extra}
            verify_files(files)
        out.write('bindings.json', signed(dict(schema='support-bindings-v1', launch_seal=launch['seal'], files=files)))
        artifacts = compute()
        verify_files(files)
        for name, value in artifacts.items(): out.write(name, value)
        hashes = {p.name: sha(p) for p in out.path.iterdir() if p.is_file()}
        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        receipt = signed(dict(schema='support-source-receipt-v1', files=files, artifacts=hashes,
                              launch_seal=launch['seal'], policy_seal=seal(policy()),
                              elapsed_seconds=time.monotonic()-start, bytes_before_receipt=out.used,
                              peak_rss_bytes=int(rss if sys.platform == 'darwin' else rss*1024),
                              python=sys.version, pandas=pd.__version__, status='completed_source_only',
                              independent_annotations_complete=False, economic_outcomes_computed=False,
                              execution_authorized=False))
        out.write('receipt.json', receipt)
        return receipt
    except BaseException as exc:
        signal.alarm(0)
        out.write('failure.json', signed(dict(schema='support-source-failure-v1', error_type=type(exc).__name__,
                  error=str(exc)[:4000], launch_seal=launch['seal'], elapsed_seconds=time.monotonic()-start,
                  execution_authorized=False)), failure=True)
        raise
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, previous)


def run_source(output):
    def prepare():
        previous = json.loads(RECEIPT.read_text()); verify(previous)
        artifacts = {str(SOURCE.parent/k): v for k, v in previous['artifacts'].items()}
        paths = [ROOT/'scripts/research'/name for name in ('support_reaction.py', 'support_reaction_replay.py',
                 'support_reaction_study.py', 'run_support_reaction_study.py', 'event_walkforward.py')]
        paths += [ROOT/'tests/research'/name for name in ('support_reaction_fixtures.py', 'test_support_reaction.py',
                  'test_support_reaction_replay.py', 'test_support_reaction_study.py')]
        paths += [ROOT/'docs/superpowers/specs/2026-10-03-support-reaction-suite-design.md',
                  ROOT/'docs/superpowers/plans/2026-10-03-support-reaction-suite.md']
        return {**previous['files'], **artifacts, **{str(p): sha(p) for p in paths}}
    def compute():
        source = json.loads(SOURCE.read_text()); verify(source)
        packets = source['packets']
        if len(packets) != 183 or len({p['id'] for p in packets}) != 183:
            raise ValueError('frozen raw183 population mismatch')
        origins = [origin_record(p) for p in packets]
        start = min(clock(o['origin']['start'])-pd.Timedelta('20h') for o in origins)
        end = max(clock(o['origin']['available_at'])+pd.Timedelta('73h') for o in origins)
        minutes = pd.read_parquet(ARCHIVE, columns=['open', 'high', 'low', 'close', 'vol'],
                    filters=[('ts', '>=', start), ('ts', '<', end)]).rename(columns={'vol': 'volume'})
        records = [assess(o, minutes, as_of=clock(o['origin']['available_at'])+pd.Timedelta('73h')) for o in origins]
        summary = summarize(records, packets)
        if summary['old_A_intents'] != 3: raise ValueError('frozen A intent parity failed')
        roster = benchmark_roster(origins); by_id = {r['origin']['id']: r for r in records}
        benchmark = [blind_packet(by_id[p['id']]) for p in roster]
        # A source-prefix witness on the preselected roster, never on winners.
        witnesses = []
        for p in roster:
            r = by_id[p['id']]; cut = clock(r['decisions']['B']['at'])
            rebuilt = assess(r['origin'], minutes.loc[minutes.index < cut], as_of=cut)
            if any(rebuilt[k] != r[k] for k in ('decisions', 'catalog', 'structure', 'demand', 'supply')):
                raise ValueError('source prefix witness failed')
            witnesses.append(dict(episode_id=p['id'], cutoff=cut.isoformat(), exact_source_decision_prefix=True))
        return {'source.json': signed(dict(schema='support-source-v1', policy_seal=seal(policy()),
                        frozen_source_hash=PINNED[str(SOURCE)], records=records, execution_authorized=False)),
                'summary.json': summary,
                'benchmark.json': signed(dict(schema='support-benchmark-v1', roster=roster, packets=benchmark,
                        independent_annotations_complete=False, selection='three lowest raw-ID SHA256 per fixed block')),
                'case_explanations.json': signed(dict(cases=[explain(r, a) for r in records for a in 'BC'])),
                'chronology.json': signed(dict(folds=folds(origins), origin_label_horizon_days=7,
                                               fitted=False, pristine_holdout=False)),
                'witnesses.json': signed(dict(source_prefix=witnesses, old_A_intents=3, raw_population=183,
                                               economic_outcomes_computed=False))}
    return bounded_source(output, PINNED, compute, prepare=prepare)
