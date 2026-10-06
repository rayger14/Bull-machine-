"""Source-only calendar extension; frozen thesis rules and packet IDs are reused."""
from bisect import bisect_left, bisect_right
from collections import Counter
from copy import deepcopy

import pandas as pd

from scripts.research.thesis_contract import MINUTE, STREAM, clock, event, protocol, seal, signed, verify
from scripts.research.thesis_sequence import compile_episode
from scripts.research.thesis_source import aggregate, atr_series, _pivots


class ParentIndex:
    """Strict as-of parity with parent_asof, including original-order time ties."""

    def __init__(self, ledger):
        self.end = clock(ledger['coverage']['query_exclusive_end'])
        self.transitions = ledger['transitions']
        self.times = [clock(t['available_at']) for t in self.transitions]
        if self.times != sorted(self.times):
            raise ValueError('parent transitions must be ordered')
        self.versions = {v['id']: v for v in ledger['versions']}
        if len(self.versions) != len(ledger['versions']):
            raise ValueError('duplicate parent version')
        if any(t['post_state'] == 'active' and t['post_version_id'] not in self.versions
               for t in self.transitions):
            raise ValueError('ledger references unknown parent version')

    def asof(self, at):
        at = clock(at)
        if at >= self.end:
            raise ValueError('out_of_coverage')
        i = bisect_left(self.times, at)-1
        if i < 0 or self.transitions[i]['post_state'] != 'active':
            return None
        return deepcopy(self.versions[self.transitions[i]['post_version_id']])


def rule_identity():
    """Calendar-independent annotation; packet IDs retain the entire old protocol."""
    ignored = {'seed', 'start', 'end', 'source_end', 'maximum_seconds', 'maximum_output_bytes',
               'execution_authorized', 'full_campaign_authorized', 'pristine_holdout'}
    rules = {k: v for k, v in protocol().items() if k not in ignored}
    return {'legacy_policy_seal': seal(protocol()), 'rule_fingerprint': seal(rules),
            'legacy_calendar_is_identity_metadata': True, 'rules': rules}


def _context(daily, times, index, origin):
    i = bisect_right(times, clock(origin['available_at']))-1
    last = daily[i] if i >= 0 else None
    if last is None or last['status'] != 'known':
        return {'status': 'unknown', 'reason': 'missing_daily_close', 'candle': last, 'parent': None}
    parent = index.asof(origin['start'])
    if parent is None:
        return {'status': 'absent', 'candle': last, 'parent': None, 'location': None}
    price = last['payload']['close']
    location = ('above' if price > parent['range_high'] else 'below' if price < parent['range_low']
                else 'inside' if parent['range_low'] < price < parent['range_high'] else 'boundary')
    return {'status': 'known', 'candle': last, 'parent': parent, 'location': location}


def build_census(minutes, parents, start, end, *, seed, source_end, stream=STREAM, resume=None):
    """One continuous lineage census; optional continuation partitions origin clocks.

    This continuation resumes origin admission, not an online in-flight sequence.
    Each partition still compiles the complete seven-day tail for its episodes.
    """
    start, end, seed, source_end = map(clock, (start, end, seed, source_end))
    if not seed < start < end <= source_end or any(t != t.floor('4h') for t in (start, end)):
        raise ValueError('invalid census calendar')
    if len(minutes) == 0 or minutes.index[0] != seed or minutes.index[-1]+MINUTE != source_end:
        raise ValueError('source bounds do not match requested calendar')
    indexes = {}
    for tf, key in [('4H', '4H_N3'), ('1D', '1D_N3')]:
        m, cov = parents[key]['manifest'], parents[key]['coverage']
        if (m['instrument'] != 'BTC' or m['data_stream_id'] != stream
                or m['parameters']['anchor_timeframe'] != tf or m['parameters']['pivot_n'] != 3):
            raise ValueError('parent source/constructor binding mismatch')
        if clock(cov['first_open']) > seed or clock(cov['last_processed_close']) < source_end:
            raise ValueError('incomplete parent coverage')
        indexes[key] = ParentIndex(parents[key])
    parent_fingerprint = seal(parents)
    original_start, consumed = start.isoformat(), set()
    if resume is not None:
        verify(resume)
        if (resume.get('schema') != 'thesis-census-continuation-v1'
                or resume.get('next_origin_close') != start.isoformat()
                or resume.get('stream_id') != stream or resume.get('seed') != seed.isoformat()
                or resume.get('parent_fingerprint') != parent_fingerprint
                or resume.get('legacy_policy_seal') != seal(protocol())):
            raise ValueError('invalid census continuation')
        original_start = resume['original_start']
        consumed = set(resume['consumed_lineages'])
    hourly, four, daily = (aggregate(minutes, tf, stream) for tf in ('1h', '4h', '1d'))
    atr = atr_series(four)
    shared = sorted(hourly+four+_pivots(hourly, stream), key=lambda e: clock(e['available_at']))
    times = [clock(e['available_at']) for e in shared]
    daily_times = [clock(e['available_at']) for e in daily]
    catalog = {e['id']: e for e in shared+daily}
    origins = {clock(e['available_at']): e for e in four}
    packets, decisions, issues = [], [], []
    expected_minutes = int((source_end-seed)/MINUTE)
    missing = expected_minutes-len(minutes)
    if missing:
        issues.append('missing_source_minutes')
    expected_closes = pd.date_range(start, end, freq='4h', inclusive='left')
    for at in expected_closes:
        origin = origins.get(at)
        row = {'at': at.isoformat(), 'origin_id': origin['id'] if origin else None,
               'parent_id': None, 'lineage_id': None, 'episode_id': None}
        decisions.append(row)
        if origin is None or origin['status'] != 'known':
            row['disposition'] = 'unknown_origin'
            issues.append('unknown_origin_candles')
            continue
        parent = indexes['4H_N3'].asof(origin['start'])
        if parent is None:
            row['disposition'] = 'no_active_parent'
            continue
        row.update(parent_id=parent['id'], lineage_id=parent['lineage_id'])
        if parent['lineage_id'] in consumed:
            row['disposition'] = 'consumed_lineage'
            continue
        c = origin['payload']
        if not c['low'] < parent['range_low'] < c['close'] < parent['range_high']:
            row['disposition'] = 'not_spring'
            continue
        consumed.add(parent['lineage_id'])
        tail_end = min(at+pd.Timedelta('7d'), source_end)
        observations = shared[bisect_right(times, at):bisect_right(times, tail_end)]
        base = {'stream_id': stream, 'parent': parent, 'origin': origin, 'atr4h': atr[origin['id']],
                'daily_context': _context(daily, daily_times, indexes['1D_N3'], origin),
                'source_status': 'known', 'observed_through': tail_end.isoformat(), 'execution_authorized': False}
        preliminary = compile_episode(base, observations)
        support = next((m for m in preliminary['milestones'] if m['kind'] == 'last_support'), None)
        if support:
            s = clock(support['available_at'])
            for minute in pd.date_range(s, min(s+pd.Timedelta('14min'), tail_end-MINUTE), freq='min'):
                payload = ({k: float(v) for k, v in minutes.loc[minute].items()
                            if k in ('open', 'high', 'low', 'close', 'volume')} if minute in minutes.index else None)
                e = event('candle', '1min', minute, minute+MINUTE, payload,
                          status='known' if payload is not None else 'unknown', stream_id=stream)
                observations.append(e)
                catalog[e['id']] = e
        p = compile_episode(base, observations)
        packets.append(p)
        row.update(disposition='raw_episode', episode_id=p['id'])
        if p['source_status'] != 'known':
            issues.append('unknown_initial_risk:'+p['id'])
        if tail_end < at+pd.Timedelta('7d'):
            issues.append('incomplete_tail:'+p['id'])
        if p['unknown_at']:
            issues.append('unknown_required_structure:'+p['id'])
    continuation = signed({'schema': 'thesis-census-continuation-v1', 'seed': seed.isoformat(),
                           'original_start': original_start, 'next_origin_close': end.isoformat(),
                           'stream_id': stream, 'parent_fingerprint': parent_fingerprint,
                           'legacy_policy_seal': seal(protocol()), 'consumed_lineages': sorted(consumed)})
    return signed({'schema': 'thesis-calendar-census-v1', 'identity': rule_identity(),
                   'stream_id': stream, 'seed': seed.isoformat(), 'start': start.isoformat(), 'end': end.isoformat(),
                   'observed_end': source_end.isoformat(), 'minute_rows': len(minutes), 'coverage': {
                       'expected_minutes': expected_minutes, 'missing_minutes': missing,
                       'expected_origin_closes': len(expected_closes), 'unknown_candles': dict(Counter(
                           e['timeframe'] for e in hourly+four+daily if e['status'] != 'known'))},
                   'packets': packets, 'catalog': catalog, 'decisions': decisions,
                   'issues': sorted(set(issues)), 'continuation': continuation,
                   'counts': {'raw_episodes': len(packets), 'lineages': len({p['parent']['lineage_id'] for p in packets}),
                              'origin_candles': len(expected_closes),
                              'simple_intents': sum(p['entry_intents']['simple'] is not None for p in packets),
                              'thesis_intents': sum(p['entry_intents']['thesis'] is not None for p in packets)},
                   'economic_outcomes_computed': False, 'economic_books_absent': True, 'execution_authorized': False})


def entry_disposition(p):
    """First entry decision/closure, not the final seven-day thesis state."""
    choices = []
    for key, kind, priority in [('unknown_at', 'unknown', 0), ('terminal_at', 'invalidated', 1),
                                ('entry_closed_at', p['entry_close_reason'], 2)]:
        if p.get(key):
            choices.append((clock(p[key]), priority, kind))
    intent = p['entry_intents']['thesis']
    if intent:
        choices.append((clock(intent['decision_time']), 3, 'intent_issued'))
    if choices:
        at, _, kind = min(choices)
    else:
        at, kind = clock(p.get('observed_through', p['deadline'])), 'right_censored'
    kinds = [m['kind'] for m in p['milestones'] if clock(m['available_at']) <= at]
    stage = next((s for s in ('test', 'strength', 'last_support', 'trigger') if s not in kinds), 'complete')
    return {'kind': kind, 'stage': stage, 'at': at.isoformat()}


def summarize_census(source):
    verify(source)
    months = {}
    first = clock(source['start']).strftime('%Y-%m')
    last = (clock(source['end'])-MINUTE).strftime('%Y-%m')
    for month in pd.period_range(first, last, freq='M').astype(str):
        months[month] = {'month': month, 'origin_candles': 0, 'raw_episodes': 0, 'simple_intents': 0,
                         'test': 0, 'strength': 0, 'last_support': 0, 'trigger': 0, 'thesis_intents': 0,
                         'admission_dispositions': Counter(), 'entry_dispositions': Counter()}
    for d in source['decisions']:
        row = months[d['at'][:7]]
        row['origin_candles'] += 1
        row['admission_dispositions'][d['disposition']] += 1
    entries, stages, obs = Counter(), Counter(), Counter()
    details = []
    for p in source['packets']:
        row = months[p['origin']['available_at'][:7]]
        row['raw_episodes'] += 1
        for name in ('simple', 'thesis'):
            row[name+'_intents'] += int(p['entry_intents'][name] is not None)
        for milestone in p['milestones']:
            row[milestone['kind']] += 1
        disposition = entry_disposition(p)
        entries[disposition['kind']] += 1
        stages[disposition['kind']+':'+disposition['stage']] += 1
        row['entry_dispositions'][disposition['kind']] += 1
        details.append(dict(disposition, episode_id=p['id'], origin_month=row['month']))
        obs['fib_anchors_defined'] += int(p['fib'] is not None)
        obs['range_destinations_defined'] += 1
        terminals = [clock(p[k]) for k in ('terminal_at', 'unknown_at') if p.get(k)]
        terminal = min(terminals) if terminals else None
        hourly = {e['available_at']: e for e in p['events'] if e['kind'] == 'candle' and e['timeframe'] == '1h'}
        for review in p['reviews']:
            known = review['at'] in hourly and hourly[review['at']]['status'] == 'known'
            active = terminal is None or clock(review['at']) < terminal
            obs['unique_clocks_scheduled'] += 1
            obs['unique_clocks_observed'] += int(known)
            obs['observed_before_termination' if active else 'observed_at_or_after_termination'] += int(known)
            for family in review['families']:
                obs[family+'_clocks_scheduled'] += 1
                obs[family+'_clocks_observed'] += int(known)
    totals = {key: sum(m[key] for m in months.values()) for key in (
        'origin_candles', 'raw_episodes', 'simple_intents', 'test', 'strength', 'last_support', 'trigger', 'thesis_intents')}
    intent_months = sum(m['thesis_intents'] > 0 for m in months.values())
    return signed({'schema': 'thesis-census-summary-v1', 'source_seal': source['seal'], 'totals': totals,
                   'months': list(months.values()), 'entry_dispositions': dict(entries), 'attrition_by_stage': dict(stages),
                   'entry_decisions': details, 'admission_dispositions': dict(Counter(d['disposition'] for d in source['decisions'])),
                   'observability': dict(obs, executed_actions_computed=False), 'coverage': source['coverage'],
                   'issues': source['issues'], 'source_qualified': not source['issues'], 'thesis_intent_months': intent_months,
                   'minimum_fill_floor_possible': (totals['thesis_intents'] >= 50 and intent_months >= 12) if not source['issues'] else None,
                   'fill_floor_reason': 'intent_upper_bound_only' if not source['issues'] else 'source_unqualified',
                   'claim': 'source_intents_upper_bound_not_fills_or_edge', 'pristine_holdout': False,
                   'economic_outcomes_computed': False, 'execution_authorized': False})
