"""Shared continuous candle census independent of every native archetype filter."""
from copy import deepcopy
import hashlib
import math
from pathlib import Path

import pandas as pd

from scripts.research.causal_parent_ledger import parent_asof
from scripts.research.thesis_contract import MINUTE, STREAM, clock, event, protocol, seal, signed
from scripts.research.thesis_execution import _validate_minutes
from scripts.research.thesis_sequence import compile_episode

ROOT = Path(__file__).resolve().parents[2]
ARCHIVE = ROOT/'data/recovered_binance_minute_2026_09_14/btc_1m_2021_2026.parquet'
PARENTS = ROOT/'results/archetype_study_2026_10_01/census_v1/parent_ledgers.json'
PINNED = {str(ARCHIVE): '5b8a4533f70b8ccd0dc533984469e6886ec73ce315d48f150c667ccb97e84035',
          str(PARENTS): 'e189c3f1dd4c284613994a1aae4cc0b396f6dc882482d97180f57e1c049c1966'}


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as handle:
        for block in iter(lambda: handle.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def implementation_files():
    paths = list((ROOT/'scripts/research').glob('thesis_*.py'))
    paths += [ROOT/'scripts/research/run_thesis_study.py', ROOT/'scripts/research/causal_parent_ledger.py',
              ROOT/'docs/superpowers/specs/2026-10-02-thesis-management-lab-design.md']
    return {str(p): sha(p) for p in paths}


def verify_files(files):
    for path, expected in files.items():
        if sha(path) != expected:
            raise ValueError('source/code hash changed: '+path)


def aggregate(minutes, timeframe, stream=STREAM):
    _validate_minutes(minutes)
    if (minutes['volume'] < 0).any() or not minutes['volume'].map(math.isfinite).all():
        raise ValueError('invalid volume')
    delta = pd.Timedelta(timeframe)
    grouped = minutes.resample(timeframe, origin='epoch', label='left', closed='left')
    bars = grouped.agg({'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum'})
    counts = grouped['close'].count()
    end = minutes.index[-1]+MINUTE
    rows = []
    for at, values in bars.iterrows():
        if at+delta > end:
            continue  # not available yet, rather than a fabricated missing close
        complete = counts.loc[at] == int(delta/MINUTE) and at >= minutes.index[0]
        payload = {k: float(v) for k, v in values.items()} if complete else None
        rows.append(event('candle', timeframe, at, at+delta, payload,
                          status='known' if complete else 'unknown', stream_id=stream))
    return rows


def atr_series(rows):
    tr, output, previous = [], {}, None
    for e in rows:
        if e['status'] == 'known':
            c = e['payload']
            value = None if previous is None else max(c['high']-c['low'], abs(c['high']-previous), abs(c['low']-previous))
            previous = c['close']
        else:
            value, previous = None, None
        tr.append(value)
        window = tr[-14:]
        output[e['id']] = math.fsum(window)/14 if len(window) == 14 and all(x is not None for x in window) else None
    return output


def _pivots(hourly, stream):
    output = []
    for i in range(2, len(hourly)-2):
        window = hourly[i-2:i+3]
        if any(e['status'] != 'known' for e in window):
            continue
        if clock(window[-1]['end'])-clock(window[0]['start']) != pd.Timedelta('5h'):
            continue
        center = window[2]
        low = center['payload']['low']
        if all(e['payload']['low'] > low for j, e in enumerate(window) if j != 2):
            output.append(event('pivot_low', '1h', center['start'], window[-1]['end'],
                                {'price': low, 'center_start': center['start'], 'decision_close': window[-1]['payload']['close']},
                                input_ids=[e['id'] for e in window], stream_id=stream))
    return output


def _daily_context(daily, ledger, origin):
    at = clock(origin['available_at'])
    rows = [e for e in daily if clock(e['available_at']) <= at]
    last = rows[-1] if rows else None
    if last is None or last['status'] != 'known':
        return {'status': 'unknown', 'reason': 'missing_daily_close', 'candle': last, 'parent': None}
    parent = parent_asof(ledger, origin['start'], strict=True)
    if parent is None:
        return {'status': 'absent', 'candle': last, 'parent': None, 'location': None}
    price = last['payload']['close']
    location = ('above' if price > parent['range_high'] else 'below' if price < parent['range_low']
                else 'inside' if parent['range_low'] < price < parent['range_high'] else 'boundary')
    return {'status': 'known', 'candle': last, 'parent': parent, 'location': location}


def build_source(minutes, parents, start, end, *, stream=STREAM):
    start, end = clock(start), clock(end)
    if start >= end:
        raise ValueError('invalid census calendar')
    for tf, key in [('4H', '4H_N3'), ('1D', '1D_N3')]:
        m = parents[key]['manifest']
        if (m['instrument'] != 'BTC' or m['data_stream_id'] != stream
                or m['parameters']['anchor_timeframe'] != tf or m['parameters']['pivot_n'] != 3):
            raise ValueError('parent source/constructor binding mismatch')
        cov = parents[key]['coverage']
        if clock(cov['first_open']) > start-pd.Timedelta('4h') or clock(cov['last_processed_close']) < min(end, minutes.index[-1]+MINUTE):
            raise ValueError('incomplete parent coverage')
    hourly, four, daily = (aggregate(minutes, tf, stream) for tf in ('1h', '4h', '1d'))
    atr = atr_series(four)
    pivots = _pivots(hourly, stream)
    shared = hourly+four+pivots
    catalog = {e['id']: e for e in shared+daily}
    packets, consumed, issues = [], set(), []
    origin_rows = [e for e in four if start <= clock(e['available_at']) < end]
    if any(e['status'] != 'known' for e in origin_rows):
        issues.append('unknown_origin_candles')
    for origin in origin_rows:
        if origin['status'] != 'known':
            continue
        parent = parent_asof(parents['4H_N3'], origin['start'], strict=True)
        if parent is None or parent['lineage_id'] in consumed:
            continue
        c = origin['payload']
        if not c['low'] < parent['range_low'] < c['close'] < parent['range_high']:
            continue
        consumed.add(parent['lineage_id'])
        t0 = clock(origin['available_at'])
        tail_end = min(t0+pd.Timedelta('7d'), minutes.index[-1]+MINUTE)
        observations = [e for e in shared if t0 < clock(e['available_at']) <= tail_end]
        base = {'stream_id': stream, 'parent': parent, 'origin': origin, 'atr4h': atr[origin['id']],
                'daily_context': _daily_context(daily, parents['1D_N3'], origin),
                'source_status': 'known', 'observed_through': tail_end.isoformat(), 'execution_authorized': False}
        preliminary = compile_episode(base, observations)
        support = next((m for m in preliminary['milestones'] if m['kind'] == 'last_support'), None)
        if support:
            s = clock(support['available_at'])
            for at in pd.date_range(s, min(s+pd.Timedelta('14min'), tail_end-MINUTE), freq='min'):
                if at in minutes.index:
                    e = event('candle', '1min', at, at+MINUTE, {k: float(v) for k, v in minutes.loc[at].items()
                              if k in ('open', 'high', 'low', 'close', 'volume')}, stream_id=stream)
                else:
                    e = event('candle', '1min', at, at+MINUTE, None, status='unknown', stream_id=stream)
                observations.append(e)
                catalog[e['id']] = e
        p = compile_episode(base, observations)
        packets.append(p)
        if p['source_status'] != 'known':
            issues.append('unknown_initial_risk:'+p['id'])
        if tail_end < t0+pd.Timedelta('7d'):
            issues.append('incomplete_tail:'+p['id'])
        if p['unknown_at']:
            issues.append('unknown_required_structure:'+p['id'])
    return signed({'schema': 'thesis-source-v1', 'policy_seal': seal(protocol()),
                   'stream_id': stream, 'start': start.isoformat(), 'end': end.isoformat(),
                   'observed_end': (minutes.index[-1]+MINUTE).isoformat(), 'minute_rows': len(minutes),
                   'packets': packets, 'catalog': catalog, 'issues': sorted(set(issues)),
                   'counts': {'raw_episodes': len(packets), 'lineages': len(consumed), 'origin_candles': len(origin_rows),
                              'simple_intents': sum(p['entry_intents']['simple'] is not None for p in packets),
                              'thesis_intents': sum(p['entry_intents']['thesis'] is not None for p in packets)},
                   'economic_outcomes_computed': False, 'economic_books_absent': True,
                   'execution_authorized': False})
