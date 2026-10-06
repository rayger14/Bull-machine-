"""Finite, timestamped offline research contracts; never a live trading adapter."""
from copy import deepcopy
import hashlib
import json
import math
from numbers import Real

import pandas as pd

MINUTE = pd.Timedelta('1min')
STREAM = 'btc_1m_2021_2026_saved_5b8a4533f70b8ccd'


def protocol():
    return {'schema': 'thesis-management-v1', 'playbook': 'established_range_spring_sequence',
            'seed': '2023-12-02T00:00:00+00:00', 'start': '2024-01-01T00:00:00+00:00',
            'end': '2024-02-01T00:00:00+00:00', 'source_end': '2024-02-08T00:00:00+00:00',
            'test_hours': 24, 'strength_hours': 48, 'support_hours': 72,
            'entry_minutes': 15, 'deadline_days': 7, 'atr_window': 14,
            'stop_buffer_atr': .1, 'fib_ratios': [.382, .618, 1., 1.618],
            'fib_time_ratios': [1., 1.618, 2.618], 'gann_hours': list(range(24, 169, 24)),
            'risk_budget': 100., 'notional_cap': 50000., 'target_r': 2.,
            'primary': {'delay_seconds': 90, 'fee_bps': 6., 'funding_bps': 8.},
            'stress': {'delay_seconds': 180, 'fee_bps': 12., 'funding_bps': 8.},
            'range_fraction': .25, 'fib_fraction': .25, 'pivot_side_bars': 2,
            'maximum_seconds': 600, 'maximum_output_bytes': 268435456,
            'trader_formula_certified': False, 'pristine_holdout': False,
            'execution_authorized': False, 'full_campaign_authorized': False}


def seal(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                    allow_nan=False).encode()).hexdigest()


def clock(value):
    if value is None or isinstance(value, (bool, Real)):
        raise ValueError('aware minute clock required')
    t = pd.Timestamp(value)
    if pd.isna(t) or t.tzinfo is None or t != t.floor('min'):
        raise ValueError('aware minute clock required')
    return t.tz_convert('UTC')


def positive(value):
    return isinstance(value, Real) and not isinstance(value, bool) and math.isfinite(value) and value > 0


def signed(record):
    payload = {k: v for k, v in record.items() if k != 'seal'}
    return dict(payload, seal=seal(payload))


def verify(record):
    if record.get('seal') != seal({k: v for k, v in record.items() if k != 'seal'}):
        raise ValueError('record seal mismatch')


def event(kind, timeframe, start, end, payload, *, status='known', input_ids=(), stream_id=STREAM):
    s, t = clock(start), clock(end)
    if s >= t or status not in ('known', 'absent', 'unknown'):
        raise ValueError('invalid event interval/status')
    row = {'kind': kind, 'timeframe': timeframe, 'start': s.isoformat(), 'end': t.isoformat(),
           'available_at': t.isoformat(), 'status': status, 'payload': deepcopy(payload),
           'input_ids': list(input_ids), 'stream_id': stream_id}
    return dict(row, id='event:'+seal(row))


def validate_event(row, stream):
    tf = row.get('timeframe')
    if tf not in ('1min', '1h', '4h', '1d'):
        raise ValueError('noncanonical timeframe')
    if row['kind'] == 'candle' and (clock(row['start']) != clock(row['start']).floor(tf)
            or clock(row['end'])-clock(row['start']) != pd.Timedelta(tf)):
        raise ValueError('invalid candle timeframe/grid')
    payload = {k: v for k, v in row.items() if k != 'id'}
    if (row.get('id') != 'event:'+seal(payload) or row.get('stream_id') != stream
            or clock(row['start']) >= clock(row['end'])
            or clock(row['end']) > clock(row['available_at'])
            or row['status'] not in ('known', 'absent', 'unknown')):
        raise ValueError('invalid event identity/clock/source')
    if row['kind'] == 'candle' and row['status'] == 'known':
        c = row['payload']
        if (any(not positive(c.get(k)) for k in ('open', 'high', 'low', 'close'))
                or c['low'] > min(c['open'], c['close'])
                or c['high'] < max(c['open'], c['close'])
                or clock(row['end'])-clock(row['start']) != pd.Timedelta(row['timeframe'])):
            raise ValueError('invalid event candle geometry')


def verify_packet(p):
    verify(p)
    if p.get('policy_seal') != seal(protocol()) or p.get('execution_authorized') is not False:
        raise ValueError('packet policy/authority mismatch')
    for e in [p['origin']]+p['events']:
        validate_event(e, p['stream_id'])
