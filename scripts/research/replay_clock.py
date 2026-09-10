"""Offline observation clock; selected-processor contract, not a live engine.

Resume deliberately replays the identical original history through a fresh
processor. No partial detector-state restoration, silent sorting or gap filling.
"""
from dataclasses import asdict, dataclass
import hashlib
import json
import math

import numpy as np
import pandas as pd


def utc(value, label='timestamp'):
    if value is None:
        raise ValueError(f'{label} is unknown')
    t = pd.Timestamp(value)
    if pd.isna(t) or t.tz is None:
        raise ValueError(f'{label} must be timezone-aware and finite')
    return t.tz_convert('UTC')


def json_safe(value):
    """Lossless nonfinite markers for manifests/hashes; never normalize inputs."""
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, pd.Timestamp):
        return str(value)
    if isinstance(value, np.generic):
        return json_safe(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return {'__nonfinite__': str(value)}
    return value


def digest(value):
    return hashlib.sha256(json.dumps(json_safe(value), sort_keys=True,
                                    allow_nan=False).encode()).hexdigest()


def duration(value):
    d = pd.Timedelta(value)
    if pd.isna(d) or d <= pd.Timedelta(0) or d > pd.Timedelta('1d'):
        raise ValueError('Timeframe must be positive and at most one day')
    if pd.Timedelta('1d').value % d.value:
        raise ValueError('Timeframe must divide a UTC day')
    return d


@dataclass(frozen=True)
class Observation:
    id: str
    feature: str
    value: object
    instrument: str
    source: str
    units: str
    version: str
    event_time: object
    available_at: object
    status: str = 'valid'
    received_at: object = None
    valid_until: object = None

    def visible_at(self):
        available = utc(self.available_at, 'available_at')
        event = utc(self.event_time, 'event_time')
        if event > available:
            raise ValueError('event_time follows available_at')
        return max(available, utc(self.received_at, 'received_at')) if self.received_at is not None else available


def validate_bars(bars, timeframe):
    step = duration(timeframe)
    if len(bars) == 0:
        raise ValueError('Empty evidence cannot certify replay')
    if not isinstance(bars.index, pd.DatetimeIndex) or bars.index.tz is None or bars.index.hasnans:
        raise ValueError('Candle index must be timezone-aware and finite')
    if not bars.index.is_unique or not bars.index.is_monotonic_increasing:
        raise ValueError('Duplicate or out-of-order candle timestamps')
    times = np.array([t.value for t in bars.index], dtype=np.int64)
    if (times % step.value).any() or (len(times) > 1 and (np.diff(times) != step.value).any()):
        raise ValueError('Candle grid/continuity failure; gaps must not be filled silently')
    cols = ['open','high','low','close','volume']
    try:
        a = bars[cols].to_numpy(dtype=float)
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError('Numeric OHLCV required') from exc
    if not np.isfinite(a).all() or (a[:,:4] <= 0).any() or (a[:,4] < 0).any():
        raise ValueError('Invalid OHLCV value')
    if (a[:,2] > a[:,[0,3]].min(axis=1)).any() or (a[:,1] < a[:,[0,3]].max(axis=1)).any():
        raise ValueError('Invalid OHLC envelope')
    return step


def context_at(bars, decision_time, base_timeframe, target_timeframe):
    base = validate_bars(bars, base_timeframe)
    target = duration(target_timeframe)
    if target < base or target.value % base.value:
        raise ValueError('Target must be an integer multiple of base timeframe')
    now = utc(decision_time)
    visible = bars.loc[bars.index + base <= now].copy()
    result = {'completed': [], 'developing': [], 'incomplete': []}
    if visible.empty:
        return result
    # Explicit UTC anchoring, independent of the input timezone's DST rules.
    visible.index = visible.index.tz_convert('UTC')
    for start, group in visible.groupby(visible.index.floor(target)):
        close_time = start + target
        expected = int(target / base)
        row = dict(open_time=str(start), close_time=str(close_time),
            open=float(group.open.iloc[0]), high=float(group.high.max()),
            low=float(group.low.min()), close=float(group.close.iloc[-1]),
            volume=float(group.volume.sum()), constituents=len(group),
            complete=close_time <= now and len(group) == expected,
            source_open_times=list(map(str, group.index)))
        if row['complete']:
            result['completed'].append(row)
        elif close_time <= now or group.index[0] != start:
            result['incomplete'].append(row)
        else:
            result['developing'].append(row)
    return result


def replay(bars, observations, factory, *, instrument, timeframe,
           emit_from=None, checkpoint=None, expected_versions=None):
    """Processor protocol: contract_id, update(candle, values), snapshot().

    All supplied observations are contract-checked. Certification here concerns
    supplied inputs/clock only, not completeness of a strategy's requirements.
    Raw invalid/stale values reach the reference processor but flag its output.
    """
    step = validate_bars(bars, timeframe)
    if not instrument:
        raise ValueError('Instrument required')
    observations = list(observations)
    times, identities, seen, last_key = [], {}, set(), None
    for o in observations:
        if not isinstance(o, Observation):
            raise ValueError('Observation records required')
        if any(not isinstance(v, str) or not v for v in
               [o.id,o.feature,o.instrument,o.source,o.units,o.version]):
            raise ValueError('Observation identity/source/units/version required')
        if o.instrument != instrument:
            raise ValueError('Observation instrument mismatch')
        if o.status not in ('valid','missing','invalid','stale'):
            raise ValueError('Unknown validity status')
        t = o.visible_at()
        if o.valid_until is not None and utc(o.valid_until) <= utc(o.event_time):
            raise ValueError('Invalid expiry')
        key = (t, o.id)
        if o.id in seen or (last_key is not None and key <= last_key):
            raise ValueError('Duplicate or out-of-order observations (visibility, id)')
        identity = (o.source,o.units,o.version)
        if o.feature in identities and identities[o.feature] != identity:
            raise ValueError(f'Formula/source/units mismatch: {o.feature}')
        if expected_versions and o.feature in expected_versions and o.version != expected_versions[o.feature]:
            raise ValueError(f'Formula version mismatch: {o.feature}')
        identities[o.feature] = identity
        seen.add(o.id); last_key = key; times.append(t)
    processor = factory()
    identity = dict(instrument=instrument,timeframe=str(step),
                    processor=processor.contract_id, expected_versions=expected_versions or {})
    first_emit = utc(emit_from) if emit_from is not None else bars.index[0].tz_convert('UTC')
    identity['emit_from'] = str(first_emit)
    records = [dict(timestamp=str(t.tz_convert('UTC')),**r) for t,r in zip(bars.index,bars.to_dict('records'))]

    def prefix_hash(end):
        return digest(dict(contract=identity, bars=[r for r in records if utc(r['timestamp'])+step <= end],
            observations=[asdict(o) for o,t in zip(observations,times) if t <= end]))

    cursor = None
    if checkpoint is not None:
        cursor = utc(checkpoint['cursor'])
        if cursor not in bars.index + step or checkpoint.get('prefix_hash') != prefix_hash(cursor):
            raise ValueError('Changed or incompatible checkpoint prehistory/contract')
    selected, position, rows, all_issues = {}, 0, [], []
    for candle in records:
        opened = utc(candle['timestamp']); now = opened + step
        while position < len(observations) and times[position] <= now:
            o = observations[position]; selected[o.feature] = o; position += 1
        issues = []
        for name,o in selected.items():
            if o.status != 'valid':
                issues.append(f'{name}: {o.status}')
            elif o.value is None or (isinstance(o.value,(float,np.floating)) and not math.isfinite(o.value)):
                issues.append(f'{name}: invalid value')
            if o.valid_until is not None and now >= utc(o.valid_until):
                issues.append(f'{name}: stale')
        output = processor.update(dict(candle,close_time=str(now)),
                                  {k:o.value for k,o in selected.items()})
        if opened >= first_emit:
            all_issues.extend(issues)
            if cursor is None or now > cursor:
                rows.append(dict(decision_time=str(now),output=output,
                    observation_ids={k:o.id for k,o in selected.items()},
                    certified=not issues,issues=issues))
    end = bars.index[-1].tz_convert('UTC') + step
    return dict(rows=rows,state=processor.snapshot(),
        checkpoint=dict(cursor=str(end),prefix_hash=prefix_hash(end)),
        certified=bool(rows) and not all_issues,issues=sorted(set(all_issues)),
        scope='Supplied observation clock and processor only; not full strategy parity')
