"""One locked source-only census. No economic or live-trading command exists."""
import argparse
import json
from pathlib import Path
import resource
import signal
import sys
import time

import pandas as pd

from scripts.research.thesis_census import build_census, rule_identity, summarize_census
from scripts.research.thesis_contract import clock, seal, signed, verify
from scripts.research.thesis_source import ROOT, ARCHIVE, PARENTS, sha, verify_files

CALENDAR = {'seed': '2023-12-02T00:00:00+00:00', 'start': '2024-01-01T00:00:00+00:00',
            'end': '2026-08-24T00:00:00+00:00', 'source_end': '2026-09-01T00:00:00+00:00'}
REFERENCE = ROOT/'results/thesis_management_2026_10_02/source_v1/source.json'
REFERENCE_RECEIPT = REFERENCE.with_name('receipt.json')
REFERENCE_FILES = {str(REFERENCE): 'fcf43d779641185f45b33d56c1872360997c9c1b32202976ced7e9c9ee39fa64',
                   str(REFERENCE_RECEIPT): '77e6d0eec18a09f5fede202d4cf23831f0137c2104eb769eaa49503ab25d6ff1'}
MAX_SECONDS, MAX_BYTES = 600, 268435456


class Output:
    """Exclusive artifacts with a cumulative output cap and failure-log reserve."""

    def __init__(self, path, limit=MAX_BYTES, failure_reserve=16384):
        self.path, self.limit, self.reserve, self.used = Path(path), limit, failure_reserve, 0
        self.path.mkdir(parents=True, exist_ok=False)

    def write(self, name, value, *, failure=False):
        if Path(name).name != name or not name.endswith('.json'):
            raise ValueError('invalid artifact name')
        data = (json.dumps(value, sort_keys=True, allow_nan=False, separators=(',', ':'))+'\n').encode()
        ceiling = self.limit if failure else self.limit-self.reserve
        if self.used+len(data) > ceiling:
            raise ValueError('cumulative output cap exceeded')
        with open(self.path/name, 'xb') as handle:
            handle.write(data)
        self.used += len(data)


def bounded_stage(output, files, compute, *, additional_files=None):
    """Internal stage wrapper. Only run() supplies the production computation."""
    started = time.monotonic()
    out = Output(output)
    launch = signed({'schema': 'thesis-census-launch-v1', 'calendar': CALENDAR, 'files': files,
                     'identity': rule_identity(), 'maximum_seconds': MAX_SECONDS,
                     'maximum_output_bytes': MAX_BYTES, 'execution_authorized': False,
                     'economic_outcomes_computed': False, 'stage': 'source_only'})
    def timeout(signum, frame):
        raise TimeoutError('census runtime cap exceeded')
    previous = signal.signal(signal.SIGALRM, timeout)
    signal.alarm(MAX_SECONDS)
    try:
        out.write('launch.json', launch)
        verify_files(files)
        bound_files = dict(files)
        if additional_files is not None:
            extra = additional_files()
            if any(k in bound_files and bound_files[k] != v for k, v in extra.items()):
                raise ValueError('conflicting input bindings')
            bound_files.update(extra)
        verify_files(bound_files)
        out.write('bindings.json', signed({'schema': 'thesis-census-bindings-v1',
                   'launch_seal': launch['seal'], 'files': bound_files}))
        artifacts = compute()
        verify_files(bound_files)
        for name, value in artifacts.items():
            out.write(name, value)
        hashes = {p.name: sha(p) for p in out.path.iterdir() if p.is_file()}
        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        receipt = signed({'schema': 'thesis-census-receipt-v1', 'launch_seal': launch['seal'],
                          'calendar': CALENDAR, 'files': bound_files, 'artifacts': hashes,
                          'legacy_policy_seal': rule_identity()['legacy_policy_seal'],
                          'rule_fingerprint': rule_identity()['rule_fingerprint'],
                          'elapsed_seconds': time.monotonic()-started,
                          'peak_rss_bytes': int(rss if sys.platform == 'darwin' else rss*1024),
                          'bytes_before_receipt': out.used, 'status': 'completed_source_only',
                          'summary_seal': artifacts.get('summary.json', {}).get('seal'),
                          'source_qualified': artifacts.get('summary.json', {}).get('source_qualified'),
                          'economic_outcomes_computed': False, 'execution_authorized': False})
        out.write('receipt.json', receipt)
        return receipt
    except BaseException as exc:
        signal.alarm(0)
        out.write('failure.json', signed({'schema': 'thesis-census-failure-v1', 'launch_seal': launch['seal'],
                   'status': 'failed', 'error_type': type(exc).__name__, 'error': str(exc)[:4000],
                   'elapsed_seconds': time.monotonic()-started, 'execution_authorized': False}), failure=True)
        raise
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, previous)


def january_witness(source, reference):
    verify(source)
    verify(reference)
    packets = [p for p in source['packets'] if clock(reference['start']) <= clock(p['origin']['available_at']) < clock(reference['end'])]
    if packets != reference['packets']:
        raise ValueError('January full-packet parity failed')
    # The full calendar adds future events, but cannot change any saved event.
    if any(source['catalog'].get(k) != v for k, v in reference['catalog'].items()):
        raise ValueError('January catalog parity failed')
    return {'packets_equal': True, 'packets': len(packets), 'catalog_events_equal': len(reference['catalog']),
            'reference_source_seal': reference['seal']}


def partition_witness(minutes, parents, source, boundary):
    boundary = clock(boundary)
    if not clock(source['start']) < boundary < clock(source['end']):
        raise ValueError('partition boundary outside source calendar')
    prefix_end = min(boundary+pd.Timedelta('7d'), clock(source['observed_end']))
    common = {'seed': source['seed'], 'stream': source['stream_id']}
    left = build_census(minutes.loc[minutes.index < prefix_end], parents, source['start'], boundary,
                        source_end=prefix_end, **common)
    right = build_census(minutes, parents, boundary, source['end'], source_end=source['observed_end'],
                         resume=left['continuation'], **common)
    prefix_catalog = {k: v for k, v in source['catalog'].items() if clock(v['available_at']) <= boundary}
    left_catalog = {k: v for k, v in left['catalog'].items() if clock(v['available_at']) <= boundary}
    checks = {'packets_equal': left['packets']+right['packets'] == source['packets'],
              'decisions_equal': left['decisions']+right['decisions'] == source['decisions'],
              'continuation_equal': right['continuation'] == source['continuation'],
              'prefix_catalog_equal': left_catalog == prefix_catalog}
    if not all(checks.values()):
        raise ValueError('prefix/origin-partition witness failed: '+str(checks))
    return dict(checks, boundary=boundary.isoformat(), prefix_source_end=prefix_end.isoformat(),
                left_packets=len(left['packets']), right_packets=len(right['packets']),
                left_source_seal=left['seal'], right_source_seal=right['seal'],
                prefix_catalog_seal=seal(prefix_catalog),
                scope='origin_partition_restart_with_complete_tails_not_online_position_restart')


def run(output):
    prepared = {}
    def prepare():
        reference = json.loads(REFERENCE.read_text())
        verify(reference)
        prepared['reference'] = reference
        new = [ROOT/'scripts/research/thesis_census.py', Path(__file__),
               ROOT/'docs/superpowers/plans/2026-10-03-thesis-source-census.md',
               ROOT/'tests/research/test_thesis_census.py', ROOT/'tests/research/test_thesis_census_run.py']
        return {**reference['files'], **{str(p): sha(p) for p in new}}
    def compute():
        reference = prepared['reference']
        minutes = pd.read_parquet(ARCHIVE, columns=['open', 'high', 'low', 'close', 'vol'],
                   filters=[('ts', '>=', clock(CALENDAR['seed'])), ('ts', '<', clock(CALENDAR['source_end']))]).rename(columns={'vol': 'volume'})
        parents = json.loads(PARENTS.read_text())
        source = build_census(minutes, parents, **CALENDAR)
        witnesses = signed({'schema': 'thesis-census-witnesses-v1', 'source_seal': source['seal'],
            'january': january_witness(source, reference),
            'partition': partition_witness(minutes, parents, source, '2025-01-01T00:00Z'),
            'execution_authorized': False})
        return {'source.json': source, 'summary.json': summarize_census(source), 'witnesses.json': witnesses}
    return bounded_stage(output, REFERENCE_FILES, compute, additional_files=prepare)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    args = parser.parse_args(argv)
    receipt = run(args.output)
    print(json.dumps(receipt, sort_keys=True))


if __name__ == '__main__':
    main()
