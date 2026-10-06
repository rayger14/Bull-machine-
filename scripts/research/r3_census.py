"""Causal, arm-neutral R3 source census. No orders, prices-after-decision or PnL."""
from bisect import bisect_left, bisect_right
from copy import deepcopy

import numpy as np
import pandas as pd

from scripts.research.study_contract import finite_number, stable_id, utc_minute

MINUTE = pd.Timedelta('1min')
SCHEMA = 'r3-census-v1'
TRANSITION_FIELDS = ('id', 'source_hour', 'available_at', 'pre_state', 'pre_lineage_id',
                     'pre_version_id', 'post_state', 'post_lineage_id', 'post_version_id',
                     'source_break_direction', 'source_sweep_low', 'source_sweep_high')


def validated_minutes(frame):
    """Reject malformed observations; preserve gaps for explicit unknown receipts."""
    columns = ['open', 'high', 'low', 'close', 'volume']
    if not isinstance(frame.index, pd.DatetimeIndex) or frame.index.tz is None:
        raise ValueError('minute index must be timezone aware')
    if frame.index.hasnans or not frame.index.is_monotonic_increasing or frame.index.has_duplicates:
        raise ValueError('minute index must be unique and increasing')
    if not (frame.index == frame.index.floor('min')).all():
        raise ValueError('unaligned minute index')
    for col in columns:
        if col not in frame or not pd.api.types.is_numeric_dtype(frame[col]) or pd.api.types.is_bool_dtype(frame[col]):
            raise ValueError('numeric OHLCV required: ' + col)
    data = frame[columns].copy()
    data.index = data.index.tz_convert('UTC')
    if not np.isfinite(data.to_numpy(dtype=float)).all():
        raise ValueError('nonfinite OHLCV')
    if ((data[['open', 'high', 'low', 'close']] <= 0).any().any()
            or (data.volume < 0).any()
            or (data.low > data[['open', 'close']].min(axis=1)).any()
            or (data.high < data[['open', 'close']].max(axis=1)).any()):
        raise ValueError('invalid OHLCV geometry')
    return data


class _Parents:
    def __init__(self, ledger, instrument, stream):
        self.ledger = ledger
        manifest = ledger.get('manifest', {})
        if manifest.get('instrument') != instrument or manifest.get('data_stream_id') != stream:
            raise ValueError('foreign parent instrument/stream')
        params = manifest.get('parameters', {})
        if str(params.get('anchor_timeframe')).lower() != '4h' or params.get('pivot_n') != 3:
            raise ValueError('R3 requires 4H_N3 parent contract')
        self.end = utc_minute(ledger['coverage']['query_exclusive_end'])
        self.pivots = {p['id']: p for p in ledger.get('pivots', [])}
        self.versions = {}
        for version in ledger['versions']:
            v = deepcopy(version)
            if not v.get('id') or not v.get('lineage_id') or v['id'] in self.versions:
                raise ValueError('invalid parent version identity')
            v['available_at'] = utc_minute(v['available_at']).isoformat()
            v['range_low'] = finite_number(v['range_low'], positive=True)
            v['range_high'] = finite_number(v['range_high'], positive=True)
            if v['range_low'] >= v['range_high']:
                raise ValueError('invalid parent bounds')
            self.versions[v['id']] = v
        self.transitions = sorted(deepcopy(ledger['transitions']), key=lambda t: (utc_minute(t['available_at']), t['id']))
        self.times = [utc_minute(t['available_at']) for t in self.transitions]
        for t, clock in zip(self.transitions, self.times):
            t['available_at'] = clock.isoformat()
            for side in ('pre', 'post'):
                vid = t.get(side + '_version_id')
                if vid is not None:
                    v = self.versions.get(vid)
                    if v is None or v['lineage_id'] != t.get(side + '_lineage_id'):
                        raise ValueError('unknown or foreign parent version reference')
                    if utc_minute(v['available_at']) > clock:
                        raise ValueError('future parent version reference')
            if t.get('post_state') == 'active' and t.get('post_version_id') is None:
                raise ValueError('active transition without parent version')

    def asof(self, clock):
        if clock >= self.end:
            raise ValueError('parent out_of_coverage')
        idx = bisect_left(self.times, clock) - 1  # strictly pre-existing
        if idx < 0 or self.transitions[idx].get('post_state') != 'active':
            return None
        return self.versions[self.transitions[idx]['post_version_id']]

    def between(self, after, through):
        return self.transitions[bisect_right(self.times, after):bisect_right(self.times, through)]

    def prefix(self, through):
        return stable_id('parent-prefix', {
            'versions': sorted([v for v in self.versions.values() if utc_minute(v['available_at']) <= through], key=lambda v: v['id']),
            'transitions': self.transitions[:bisect_right(self.times, through)],
        })


def build_r3_census(minutes, parent_ledger, *, instrument, data_stream_id,
                    emit_from, end_exclusive, checkpoint=None):
    """Consume minute opens < end_exclusive, revealing bars only at close.

    Results are append-only deltas on restart; checkpoints contain bounded candle
    state, not raw history. Boundary-close events are retained, never made terminal
    solely by a reporting cutoff. Campaign calendar selection is a later view.
    """
    data = validated_minutes(minutes)
    emit_from, end = utc_minute(emit_from), utc_minute(end_exclusive)
    if not instrument or not data_stream_id or emit_from >= end:
        raise ValueError('invalid census identity/window')
    if len(data) and data.index[-1] >= end:
        raise ValueError('minute open at/after exclusive end')
    parents = _Parents(parent_ledger, instrument, data_stream_id)
    if len(data) and data.index[-1] + MINUTE >= parents.end:
        raise ValueError('parent out_of_coverage at source close')
    identity = {'schema': SCHEMA, 'instrument': instrument, 'data_stream_id': data_stream_id,
                'emit_from': emit_from.isoformat()}
    state = deepcopy(checkpoint) if checkpoint else {
        'identity': identity, 'last_close': None, 'partial_5m': [], 'recent_5m': [],
        'active': {}, 'reservations': {}, 'parent_prefix': None,
    }
    if state['identity'] != identity:
        raise ValueError('checkpoint identity mismatch')
    if state['last_close']:
        last = utc_minute(state['last_close'])
        if len(data) and data.index[0] != last:
            raise ValueError('resume requires strictly contiguous suffix')
        if parents.prefix(last) != state['parent_prefix']:
            raise ValueError('parent prefix changed since checkpoint')
    result = {'schema': SCHEMA, 'opportunities': [], 'events': [], 'attempts': [], 'blockers': []}

    def event(active, clock, kind, **evidence):
        if not active['reported']:
            return
        row = {'opportunity_id': active['opportunity']['id'], 'available_at': clock.isoformat(),
               'kind': kind, **evidence}
        row['id'] = stable_id('r3-event', row)
        result['events'].append(row)

    def finish(active, clock, kind, reason, **evidence):
        active['phase'] = 'terminal'
        event(active, clock, kind, reason=reason, **evidence)

    for raw in data.itertuples(name=None):
        opened, o, h, low, c, volume = raw
        clock = opened + MINUTE
        previous = utc_minute(state['last_close']) if state['last_close'] else opened
        if opened != previous:
            gap = {'reason': 'minute_gap', 'from': previous.isoformat(), 'to': opened.isoformat()}
            result['blockers'].append(gap)
            for active in state['active'].values():
                if active['phase'] != 'terminal':
                    finish(active, clock, 'unknown', 'minute_gap')
            state['partial_5m'], state['recent_5m'] = [], []

        for transition in parents.between(previous, clock):
            active = state['active'].get(transition.get('pre_lineage_id'))
            direction = transition.get('source_break_direction')
            if active and active['phase'] != 'terminal':
                citation = {k: transition.get(k) for k in TRANSITION_FIELDS}
                if direction == 'down':
                    finish(active, utc_minute(transition['available_at']), 'cancelled', 'parent_down',
                           transition_id=transition['id'], source_transition=citation)
                elif direction == 'up':
                    event(active, utc_minute(transition['available_at']), 'parent_up',
                          transition_id=transition['id'], source_transition=citation)

        # Retest candle itself cannot stop itself out. Later completed bars can.
        for active in state['active'].values():
            if active['phase'] == 'trigger' and clock > utc_minute(active['retest']['available_at']):
                if low <= active['retest']['low']:
                    finish(active, clock, 'cancelled', 'pre_entry_stop_touch')

        bar = {'open_time': opened.isoformat(), 'available_at': clock.isoformat(),
               'open': float(o), 'high': float(h), 'low': float(low), 'close': float(c), 'volume': float(volume)}
        if opened.minute % 5 == 0:
            state['partial_5m'] = []
        state['partial_5m'].append(bar)
        five = None
        if clock.minute % 5 == 0:
            part = state['partial_5m']
            if len(part) == 5 and utc_minute(part[0]['open_time']) == clock - 5 * MINUTE:
                five = {'open_time': part[0]['open_time'], 'available_at': clock.isoformat(),
                        'open': part[0]['open'], 'high': max(b['high'] for b in part),
                        'low': min(b['low'] for b in part), 'close': part[-1]['close'],
                        'volume': sum(b['volume'] for b in part)}
                five['id'] = stable_id('r3-child-bar', five)
                if state['recent_5m'] and state['recent_5m'][-1]['available_at'] != five['open_time']:
                    state['recent_5m'] = []
                state['recent_5m'] = (state['recent_5m'] + [five])[-6:]
            else:
                state['recent_5m'] = []
            state['partial_5m'] = []

        if five:
            for lineage, active in state['active'].items():
                op = active['opportunity']
                if active['phase'] == 'breakout':
                    if five['close'] < op['child_low']:
                        finish(active, clock, 'cancelled', 'child_close_below')
                    elif five['close'] > op['child_high']:
                        active['phase'], active['breakout_time'] = 'retest', clock.isoformat()
                        state['reservations'][lineage] = (clock + 40 * MINUTE).isoformat()
                        event(active, clock, 'breakout', close=five['close'], stop=op['child_low'])
                    elif clock >= utc_minute(op['origin_time']) + 30 * MINUTE:
                        finish(active, clock, 'expired', 'no_breakout')
                elif active['phase'] == 'retest':
                    if five['low'] <= op['child_high']:
                        if five['close'] <= op['child_high']:
                            finish(active, clock, 'cancelled', 'first_retest_failed')
                        else:
                            active['phase'], active['retest'] = 'trigger', deepcopy(five)
                            event(active, clock, 'retest', high=five['high'], low=five['low'], close=five['close'])
                    elif clock >= utc_minute(active['breakout_time']) + 30 * MINUTE:
                        finish(active, clock, 'expired', 'no_retest')

        for active in state['active'].values():
            if active['phase'] == 'trigger' and clock > utc_minute(active['retest']['available_at']):
                if c > active['retest']['high']:
                    event(active, clock, 'trigger', close=float(c), stop=active['retest']['low'],
                          breakout_time=active['breakout_time'])
                    active['phase'] = 'terminal'
                elif clock >= utc_minute(active['retest']['available_at']) + 5 * MINUTE:
                    finish(active, clock, 'expired', 'no_trigger')

        # Source reservations survive every terminal outcome and every book state.
        state['active'] = {k: a for k, a in state['active'].items()
                           if clock < utc_minute(state['reservations'][k])}
        recent = state['recent_5m']
        if five and len(recent) == 6:
            first = utc_minute(recent[0]['open_time'])
            parent = parents.asof(first)
            child_low, child_high = min(b['low'] for b in recent), max(b['high'] for b in recent)
            reason = 'absent_parent'
            if parent:
                lineage = parent['lineage_id']
                reservation = state['reservations'].get(lineage)
                down = any(t.get('pre_lineage_id') == lineage and t.get('source_break_direction') == 'down'
                           for t in parents.between(first - MINUTE, clock))
                if down:
                    reason = 'parent_down_during_construction'
                elif reservation and first < utc_minute(reservation):
                    reason = 'lineage_reserved_or_child_not_new'
                elif not (parent['range_low'] <= child_low < child_high <= parent['range_high']
                          and child_high - child_low <= (parent['range_high'] - parent['range_low']) / 4):
                    reason = 'ineligible_geometry'
                else:
                    reason = 'armed'
                    fields = {'instrument': instrument, 'parent_lineage_id': lineage,
                              'parent_version_id': parent['id'], 'origin_time': clock.isoformat(),
                              'child_first_open': first.isoformat(), 'child_last_close': clock.isoformat()}
                    op = {'id': stable_id('r3-box', fields), 'family': 'R3', **fields,
                          'data_stream_id': data_stream_id, 'child_low': child_low, 'child_high': child_high,
                          'parent': deepcopy(parent), 'child_bars': deepcopy(recent),
                          'source_evidence': {'contract_id': parent_ledger['manifest'].get('contract_id'),
                                              'pivots': [deepcopy(parents.pivots[pid]) for pid in
                                                         (parent['low_pivot_id'], parent['high_pivot_id'])
                                                         if pid in parents.pivots]}}
                    active = {'phase': 'breakout', 'opportunity': op, 'reported': clock >= emit_from}
                    state['active'][lineage] = active
                    state['reservations'][lineage] = (clock + 30 * MINUTE).isoformat()
                    if active['reported']:
                        result['opportunities'].append(deepcopy(op))
                    event(active, clock, 'armed', parent_version_id=parent['id'])
            if clock >= emit_from:
                result['attempts'].append({'available_at': clock.isoformat(), 'child_first_open': first.isoformat(),
                                           'parent_version_id': parent['id'] if parent else None, 'reason': reason})
        state['last_close'] = clock.isoformat()

    if state['last_close']:
        state['parent_prefix'] = parents.prefix(utc_minute(state['last_close']))
    result['checkpoint'] = state
    result['coverage'] = {'emit_from': emit_from.isoformat(), 'source_open_end_exclusive': end.isoformat(),
                          'consumed_checkpoint_id': stable_id('r3-checkpoint', checkpoint) if checkpoint else None,
                          'first_source_open': data.index[0].isoformat() if len(data) else state['last_close'],
                          'last_available_at': state['last_close'], 'input_minutes': len(data),
                          'right_censored_opportunities': [a['opportunity']['id'] for a in state['active'].values()
                                                          if a['reported'] and a['phase'] != 'terminal']}
    return result


def merge_censuses(*segments):
    """Join contiguous append-only output before compiling signals or projecting cases."""
    if not segments:
        raise ValueError('at least one census segment required')
    result = deepcopy(segments[0])
    known = {op['id'] for op in result['opportunities']}
    for segment in segments[1:]:
        if (segment['schema'] != result['schema']
                or segment['checkpoint']['identity'] != result['checkpoint']['identity']
                or segment['coverage']['first_source_open'] != result['checkpoint']['last_close']):
            raise ValueError('census merge requires contiguous identical stream')
        if segment['coverage']['consumed_checkpoint_id'] != stable_id('r3-checkpoint', result['checkpoint']):
            raise ValueError('census merge checkpoint lineage mismatch')
        ids = {op['id'] for op in segment['opportunities']}
        if known & ids or len(ids) != len(segment['opportunities']):
            raise ValueError('duplicate raw opportunity identity')
        known.update(ids)
        if any(e['opportunity_id'] not in known for e in segment['events']):
            raise ValueError('census event missing immutable opportunity registry')
        for key in ('opportunities', 'events', 'attempts', 'blockers'):
            result[key].extend(deepcopy(segment[key]))
        total = result['coverage']['input_minutes'] + segment['coverage']['input_minutes']
        first = result['coverage']['first_source_open']
        result['coverage'] = dict(deepcopy(segment['coverage']), first_source_open=first, input_minutes=total)
        result['checkpoint'] = deepcopy(segment['checkpoint'])
    return result


def compile_r3_signals(census, arm):
    if arm not in ('baseline', 'repair'):
        raise ValueError('unknown R3 arm')
    ops = {op['id']: op for op in census['opportunities']}
    signals = []
    for event in census['events']:
        if event['kind'] != ('breakout' if arm == 'baseline' else 'trigger'):
            continue
        if event['opportunity_id'] not in ops:
            raise ValueError('merge census deltas before compiling signals')
        op = ops[event['opportunity_id']]
        decision = utc_minute(event['available_at'])
        breakout = utc_minute(event.get('breakout_time', event['available_at']))
        signals.append({'opportunity_id': op['id'], 'family': 'R3', 'arm': arm,
                        'decision_time': decision.isoformat(), 'stop': event['stop'],
                        'entry_expiry': (decision + 5 * MINUTE).isoformat(),
                        'exit_deadline': (breakout + pd.Timedelta('4h')).isoformat(),
                        'parent_lineage_id': op['parent_lineage_id']})
    return signals
