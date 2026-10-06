"""Finite engineering orchestration, paired attribution and ledger reconciliation."""
from collections import Counter
import json
import math
from pathlib import Path
import time

import pandas as pd

from scripts.research.thesis_contract import clock, protocol, seal, signed, verify
from scripts.research.thesis_execution import replay_book
from scripts.research.thesis_source import (ROOT, ARCHIVE, PARENTS, PINNED, build_source,
    implementation_files, sha, verify_files)


def new_output(path):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=False)
    return path


def _write(path, value):
    data = json.dumps(value, sort_keys=True, allow_nan=False, separators=(',', ':'))
    if len(data.encode()) > protocol()['maximum_output_bytes']:
        raise ValueError('artifact exceeds resource cap')
    with open(path, 'x') as handle:
        handle.write(data+'\n')


def reconcile(book):
    """Separate literal cashflow identities, including unknown positions' ledgers."""
    for p in book['positions'].values():
        flows = p['cashflows']
        entry = [f for f in flows if f['kind'] == 'entry']
        exits = [f for f in flows if f['kind'] in ('partial', 'exit')]
        checks = {'quantity': sum(f['quantity'] for f in entry)-sum(f['quantity'] for f in exits)-p['remaining_qty'],
                  'fee': math.fsum(f['fee'] for f in flows)-p['fees'],
                  'funding': math.fsum(f['funding'] for f in flows)-p['funding'],
                  'gross': math.fsum(f['quantity']*(f['price']-p['entry_price']) for f in exits)-p['gross']}
        for name, difference in checks.items():
            if not math.isclose(difference, 0., abs_tol=1e-8):
                raise ValueError(name+' reconciliation failed')
        if p['status'] == 'closed' and not math.isclose(p['net'], p['gross']-p['fees']-p['funding'], abs_tol=1e-8):
            raise ValueError('net reconciliation failed')
    return len(book['positions'])


def summarize(book):
    counts = Counter(r['status'] for r in book['rows'])
    actions = Counter(a['kind'] for p in book['positions'].values() for a in p['actions'])
    reasons = Counter(f['reason'] for p in book['positions'].values() for f in p['cashflows'] if f['kind'] in ('partial', 'exit'))
    unknown = counts['unknown']+counts['open']+counts['pending']+counts['watch']
    return {'raw_episodes': len(book['rows']), 'status_counts': dict(counts), 'fills': len(book['positions']),
            'net': None if unknown else math.fsum(r['net'] for r in book['rows']),
            'fees': math.fsum(p['fees'] for p in book['positions'].values()),
            'funding': math.fsum(p['funding'] for p in book['positions'].values()),
            'exposure_minutes': sum(p['exposure_minutes'] for p in book['positions'].values()),
            'drawdown': None if unknown else book['drawdown'], 'actions': dict(actions), 'exit_reasons': dict(reasons)}


def compare(packets, minutes):
    books, shadows, pairs = {}, {}, {}
    for entry in ('simple', 'thesis'):
        for management in ('fixed', 'adaptive'):
            name = entry+'_'+management
            books[name] = replay_book(packets, minutes, entry, management)
            shadows[name] = replay_book(packets, minutes, entry, management, capacity=False)
        pairs[entry] = shadows[entry+'_fixed']['entry_tape'] == shadows[entry+'_adaptive']['entry_tape']
        if not pairs[entry]:
            raise ValueError('identical-entry attribution failed')
    count = sum(reconcile(b) for b in list(books.values())+list(shadows.values()))
    deltas = {}
    for entry in ('simple', 'thesis'):
        fixed = {r['episode_id']: r for r in shadows[entry+'_fixed']['rows']}
        deltas[entry] = [{'episode_id': r['episode_id'], 'delta': None if r['net'] is None or fixed[r['episode_id']]['net'] is None
                           else r['net']-fixed[r['episode_id']]['net']} for r in shadows[entry+'_adaptive']['rows']]
    return signed({'schema': 'thesis-engineering-comparison-v1', 'raw_episodes': len(packets),
                   'books': books, 'shadows': shadows, 'paired_entry_equality': pairs,
                   'paired_shadow_deltas': deltas, 'summary': {k: summarize(b) for k, b in books.items()},
                   'reconciliation': {'positions_checked': count},
                   'claim': 'engineering_only_no_edge_verdict', 'execution_authorized': False})


def validate_source_review(source, review, source_file_sha):
    verify(source)
    verify(review)
    if (source['issues'] or source['economic_outcomes_computed'] is not False
            or review.get('source_seal') != source['seal'] or review.get('source_file_sha256') != source_file_sha
            or review.get('verdict') != 'clear_engineering' or review.get('execution_authorized') is not False
            or not review.get('reviewer') or not review.get('witnesses')):
        raise ValueError('source review not cleared/bound')


def _minutes():
    p = protocol()
    return pd.read_parquet(ARCHIVE, columns=['open', 'high', 'low', 'close', 'vol'],
                           filters=[('ts', '>=', clock(p['seed'])), ('ts', '<', clock(p['source_end']))]).rename(columns={'vol': 'volume'})


def run_source(output):
    started = time.monotonic()
    files = {**PINNED, **implementation_files()}
    verify_files(files)
    parents = json.loads(PARENTS.read_text())
    source = build_source(_minutes(), parents, protocol()['start'], protocol()['end'])
    source = signed(dict(source, files=files))
    if time.monotonic()-started > protocol()['maximum_seconds']:
        raise ValueError('source runtime cap')
    out = new_output(output)
    _write(out/'source.json', source)
    receipt = signed({'schema': 'thesis-source-receipt-v1', 'source_seal': source['seal'],
                      'artifacts': {'source.json': sha(out/'source.json')}, 'files': files,
                      'counts': source['counts'], 'issues': source['issues'],
                      'elapsed_seconds': time.monotonic()-started,
                      'economic_outcomes_computed': False, 'economic_books_absent': True,
                      'execution_authorized': False})
    _write(out/'receipt.json', receipt)
    return receipt


def run_engineering(source_dir, output, review_path):
    started = time.monotonic()
    source_path = Path(source_dir)/'source.json'
    source, review = json.loads(source_path.read_text()), json.loads(Path(review_path).read_text())
    validate_source_review(source, review, sha(source_path))
    verify_files(source['files'])
    if source['start'] != protocol()['start'] or source['end'] != protocol()['end'] or source['policy_seal'] != seal(protocol()):
        raise ValueError('unapproved calendar/policy')
    comparison = compare(source['packets'], _minutes())
    if any(s['status_counts'].get('unknown', 0) for s in comparison['summary'].values()):
        raise ValueError('unknown economic paths; no success receipt')
    if time.monotonic()-started > protocol()['maximum_seconds']:
        raise ValueError('engineering runtime cap')
    out = new_output(output)
    _write(out/'comparison.json', comparison)
    receipt = signed({'schema': 'thesis-engineering-receipt-v1', 'source_file_sha256': sha(source_path),
                      'review_sha256': sha(review_path), 'artifacts': {'comparison.json': sha(out/'comparison.json')},
                      'elapsed_seconds': time.monotonic()-started, 'summary': comparison['summary'],
                      'claim': comparison['claim'], 'execution_authorized': False})
    _write(out/'receipt.json', receipt)
    return receipt
