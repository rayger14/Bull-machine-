"""Review-gated accounting using the unchanged fixed execution reducer.

Sparse scheduling is based on source entry eligibility and maximum label windows,
never profitability or realized holding periods. This is not a new trading engine.
"""
import argparse
import json
from pathlib import Path
import resource
import signal
import sys
import time

import pandas as pd

from scripts.research.run_thesis_census import Output
from scripts.research.support_reaction import policy
from scripts.research.support_reaction_replay import replay, folds, _totals, _pairs
from scripts.research.support_reaction_review import SOURCE_DIR, SOURCE_RECEIPT_SHA, verify_source
from scripts.research.support_reaction_study import SOURCE
from scripts.research.thesis_contract import clock, protocol, seal, signed, verify
from scripts.research.thesis_source import ROOT, ARCHIVE, sha, verify_files

MAX_SECONDS, MAX_BYTES = 600, 268435456


class UnknownOccupancy(ValueError):
    def __init__(self, book):
        super().__init__('sticky unknown occupancy prevents independent component reset')
        self.partial_book = book


def components(records):
    ordered = sorted(records, key=lambda r: (clock(r['origin']['origin']['available_at']), r['origin']['id']))
    groups = []
    for r in ordered:
        raw = r['origin']; at, end = clock(raw['origin']['available_at']), clock(raw['deadline'])
        if not groups or at > clock(groups[-1]['end']):
            groups.append(dict(start=at.isoformat(), end=end.isoformat(), ids=[raw['id']]))
        else:
            groups[-1]['end'] = max(end, clock(groups[-1]['end'])).isoformat()
            groups[-1]['ids'].append(raw['id'])
    return groups


def _possible_occupancy(record, packet, arm):
    if packet['source_status'] != 'known': return True
    if arm == 'A': return packet['entry_intents']['thesis'] is not None
    return record['decisions'][arm]['status'] in ('intent', 'unknown')


def _watcher_end(record, packet, arm):
    if arm == 'A':
        clocks = [packet.get(k) for k in ('entry_closed_at', 'terminal_at', 'unknown_at')]
    else:
        d = record['decisions'][arm]
        clocks = [d['at']] if d['status'] != 'pending' else []
    return min([clock(packet['deadline'])]+[clock(x) for x in clocks if x])


def _frame(records, minutes, end):
    start = min(clock(r['origin']['origin']['start'])-pd.Timedelta('20h') for r in records)
    frame = minutes.loc[(minutes.index >= start) & (minutes.index <= end)]
    if frame.empty: raise ValueError('missing component minute coverage')
    return frame


def partitioned_book(records, old_packets, minutes, *, arm, execution=None, capacity=False):
    if arm not in 'ABC' or len(arm) != 1: raise ValueError('invalid arm')
    by_id = {p['id']: p for p in old_packets}; by_record = {r['origin']['id']: r for r in records}
    if len(by_id) != len(old_packets) or len(by_record) != len(records) or set(by_id) != set(by_record):
        raise ValueError('full raw population mismatch')
    active = [r for r in records if _possible_occupancy(r, by_id[r['origin']['id']], arm)]
    passive = [r for r in records if not _possible_occupancy(r, by_id[r['origin']['id']], arm)]
    groups = components(active)
    parts = []
    for r in passive:
        p = by_id[r['origin']['id']]
        parts.append((dict(kind='source_closed_nonoccupying', ids=[p['id']]), [r],
                      _watcher_end(r, p, arm)))
    for group in groups:
        rs = [by_record[eid] for eid in group['ids']]
        parts.append((dict(kind='maximum_horizon_component', **group), rs,
                      clock(group['end'])+pd.Timedelta('1min')))
    rows, positions, admission, tape, audit = {}, {}, {}, [], []
    for grouping, rs, end in parts:
        ps = [by_id[r['origin']['id']] for r in rs]
        book = replay(rs, ps, _frame(rs, minutes, end), arm=arm, execution=execution, capacity=capacity)
        if capacity and book['runtime']['unknown_occupancy']: raise UnknownOccupancy(book)
        for row in book['rows']:
            if row['episode_id'] in rows: raise ValueError('duplicate scheduled episode')
            rows[row['episode_id']] = row
        positions.update(book['runtime']['positions']); admission.update(book['admission'])
        tape.extend(book['runtime']['entry_tape'])
        audit.append(dict(grouping=grouping, book=book))
    if set(rows) != set(by_id): raise ValueError('scheduled raw population lost')
    return signed(dict(schema='support-partitioned-book-v1', arm=arm, capacity=capacity,
                       execution=execution or protocol()['primary'], policy_seal=seal(policy()),
                       source_ids=sorted(by_id), rows=[rows[p['id']] for p in old_packets],
                       positions=positions, admission=admission,
                       entry_tape=sorted(tape, key=lambda t: (clock(t['at']), t['episode_id'])),
                       partitions=audit, global_drawdown=None,
                       drawdown_status='not_reconstructed_across_components',
                       execution_authorized=False, edge_demonstrated=False))


def partitioned_compare(records, old_packets, minutes, *, execution=None, emit=None):
    origins = {r['origin']['id']: r['origin'] for r in records}
    books, occupied = {}, {}
    for arm in 'ABC':
        books[arm] = partitioned_book(records, old_packets, minutes, arm=arm, execution=execution)
        occupied[arm] = partitioned_book(records, old_packets, minutes, arm=arm, execution=execution, capacity=True)
        if emit: emit(arm, books[arm], occupied[arm])
    splits = folds(list(origins.values())); reports=[]
    for split in splits:
        selected = set(split['test_ids'])
        subset = {a: {'rows': [r for r in b['rows'] if r['episode_id'] in selected]} for a,b in books.items()}
        rs = [r for r in records if r['origin']['id'] in selected]
        ps = [p for p in old_packets if p['id'] in selected]
        obooks = {a: partitioned_book(rs, ps, minutes, arm=a, execution=execution, capacity=True) for a in 'ABC'}
        reports.append(dict(split=split, test_raw_episodes=len(selected),
                            arms={a: _totals(b, origins) for a,b in subset.items()}, pairs=_pairs(subset),
                            occupied_arms={a: _totals(b, origins) for a,b in obooks.items()},
                            occupied_books=obooks, occupied_boundary='reset_to_empty_at_test_window',
                            fitted=False, pristine_holdout=False))
    values=list(origins.values())
    overlaps=sum(clock(a['origin']['available_at']) <= clock(b['origin']['available_at']) < clock(a['deadline'])
                 or clock(b['origin']['available_at']) <= clock(a['origin']['available_at']) < clock(b['deadline'])
                 for i,a in enumerate(values) for b in values[i+1:])
    return signed(dict(schema='support-partitioned-comparison-v1', raw_episodes=len(records),
                       arms={a: _totals(b,origins) for a,b in books.items()}, pairs=_pairs(books),
                       occupied_arms={a: _totals(b,origins) for a,b in occupied.items()},
                       attribution_books=books, occupied_books=occupied, folds=splits,
                       chronological_reports=reports, dependence=dict(overlapping_origin_pairs=overlaps,
                       parent_lineages=len({o['parent']['lineage_id'] for o in values})),
                       capacity_free_is_portfolio=False, fitted=False, pristine_holdout=False,
                       edge_demonstrated=False, execution_authorized=False))


def bounded_economics(output, review, files, compute, *, prepare=None):
    """Internal bounded stage; run() enforces the exact natural review/population."""
    out=Output(output); start=time.monotonic()
    launch=signed(dict(schema='support-economic-launch-v1', stage='exploratory_economics',
                       review_seal=review['seal'] if review else None, files=files, policy_seal=seal(policy()),
                       maximum_seconds=MAX_SECONDS, maximum_output_bytes=MAX_BYTES,
                       scenarios={s:protocol()[s] for s in ('primary','stress')}, execution_authorized=False))
    def timeout(signum, frame): raise TimeoutError('support economic runtime cap exceeded')
    previous=signal.signal(signal.SIGALRM,timeout); signal.alarm(MAX_SECONDS)
    try:
        out.write('launch.json',launch)
        if prepare is not None: review,files=prepare()
        verify(review)
        if review['status']!='exploratory_economics_cleared': raise ValueError('semantic review has not cleared economics')
        if any(files.get(k)!=v for k,v in review['files'].items()): raise ValueError('review binding omitted or changed')
        verify_files(files)
        out.write('bindings.json',signed(dict(launch_seal=launch['seal'],review_seal=review['seal'],files=files)))
        compute(out); verify_files(files)
        hashes={p.name:sha(p) for p in out.path.iterdir() if p.is_file()}
        rss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        receipt=signed(dict(schema='support-economic-receipt-v1', status='completed_exploratory_accounting',
                    review_seal=review['seal'], launch_seal=launch['seal'], files=files, artifacts=hashes,
                    elapsed_seconds=time.monotonic()-start, bytes_before_receipt=out.used,
                    peak_rss_bytes=int(rss if sys.platform=='darwin' else rss*1024),
                    edge_demonstrated=False, economic_outcomes_computed=True, execution_authorized=False))
        out.write('receipt.json',receipt); return receipt
    except BaseException as exc:
        signal.alarm(0)
        if isinstance(exc,UnknownOccupancy):
            try: out.write('unknown_partial_book.json',exc.partial_book)
            except ValueError: pass  # reserved failure record still fits
        out.write('failure.json',signed(dict(schema='support-economic-failure-v1', launch_seal=launch['seal'],
                  error_type=type(exc).__name__, error=str(exc)[:4000], elapsed_seconds=time.monotonic()-start,
                  status='incomplete_no_economic_clearance', execution_authorized=False)),failure=True)
        raise
    finally:
        signal.alarm(0); signal.signal(signal.SIGALRM,previous)


def implementation_files():
    names=['scripts/research/support_reaction_review.py', 'scripts/research/support_reaction_economics.py',
           'tests/research/test_support_reaction_review.py', 'tests/research/test_support_reaction_economics.py',
           'docs/superpowers/specs/2026-10-03-support-reaction-validation-design.md',
           'docs/superpowers/plans/2026-10-03-support-reaction-validation.md']
    return {str(ROOT/n):sha(ROOT/n) for n in names}


def run(output, review_path):
    def prepare():
        path=Path(review_path).resolve(); review=json.loads(path.read_text()); verify(review)
        benchmark=json.loads((SOURCE_DIR/'benchmark.json').read_text()); verify(benchmark)
        if (review.get('schema')!='support-semantic-review-v1' or review['source_receipt_sha']!=SOURCE_RECEIPT_SHA
                or review['packet_seals']!=sorted(p['seal'] for p in benchmark['packets']) or len(review['packet_seals'])!=12):
            raise ValueError('wrong semantic/source review')
        required={**verify_source(),**implementation_files()}
        if any(review['files'].get(k)!=v for k,v in required.items()):
            raise ValueError('semantic review does not bind exact source/code')
        return review,{**review['files'],str(path):sha(path)}
    def compute(out):
        source=json.loads((SOURCE_DIR/'source.json').read_text()); verify(source)
        old=json.loads(SOURCE.read_text()); verify(old)
        records,packets=source['records'],old['packets']
        if len(records)!=183 or {r['origin']['id'] for r in records}!={p['id'] for p in packets}:
            raise ValueError('frozen raw183 population mismatch')
        start=min(clock(p['origin']['start'])-pd.Timedelta('20h') for p in packets)
        end=max(clock(p['deadline'])+pd.Timedelta('2min') for p in packets)
        minutes=pd.read_parquet(ARCHIVE,columns=['open','high','low','close','vol'],
                    filters=[('ts','>=',start),('ts','<',end)]).rename(columns={'vol':'volume'})
        for scenario in ('primary','stress'):
            print('starting '+scenario,flush=True)
            def emit(arm,free,occupied):
                out.write(scenario+'_'+arm+'_books.json',signed(dict(attribution=free,occupied=occupied)))
                print(scenario+' '+arm+' complete',flush=True)
            result=partitioned_compare(records,packets,minutes,execution=protocol()[scenario],emit=emit)
            out.write(scenario+'.json',result)
            print(scenario+' chronology complete',flush=True)
    return bounded_economics(output,None,{},compute,prepare=prepare)


def main():
    p=argparse.ArgumentParser(); p.add_argument('--output',required=True); p.add_argument('--review',required=True)
    args=p.parse_args(); receipt=run(args.output,args.review)
    print(json.dumps({k:receipt[k] for k in ('status','elapsed_seconds','seal')}))


if __name__=='__main__': main()
