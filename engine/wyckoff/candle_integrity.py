"""Completed, contiguous inputs for Wyckoff only; no generic resampler changes.

Inputs are UTC start-stamped observations (naive timestamps mean UTC). The caller
supplies a causal as-of cutoff. Native daily observations have separate provenance:
their validity does not certify inspection of their 24 hourly constituents.
"""
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd


OHLCV = ['open', 'high', 'low', 'close', 'volume']


@dataclass(frozen=True)
class WyckoffInput:
    frame: pd.DataFrame
    status: str
    reason: str
    source: str
    last_input_close: Optional[pd.Timestamp]
    available_at: Optional[pd.Timestamp]


def _observations(frame):
    if frame is None:
        return pd.DataFrame(columns=OHLCV, index=pd.DatetimeIndex([], tz='UTC'))
    out = frame.loc[:, OHLCV].copy()
    out.index = pd.to_datetime(out.index, utc=True)
    if out.index.hasnans:
        raise ValueError('unknown candle timestamp')
    return out.sort_index()


def _valid_ohlcv(frame):
    values = frame.apply(pd.to_numeric, errors='coerce')
    return (np.isfinite(values).all(axis=1)
            & (values[['open', 'high', 'low', 'close']] > 0).all(axis=1)
            & (values['volume'] >= 0)
            & (values['high'] >= values[['open', 'close', 'low']].max(axis=1))
            & (values['low'] <= values[['open', 'close']].min(axis=1))).all()


def _bins(frame, timeframe, step):
    """Return only valid bins; absent keys are barriers, never compressed away."""
    result = {}
    for start, group in frame.groupby(frame.index.floor(timeframe)):
        expected = pd.date_range(start, periods=int(pd.Timedelta(timeframe) / step), freq=step)
        if not group.index.equals(expected) or not _valid_ohlcv(group):
            continue
        numeric = group.astype(float)
        result[start] = dict(open=numeric['open'].iloc[0], high=numeric['high'].max(),
                             low=numeric['low'].min(), close=numeric['close'].iloc[-1],
                             volume=numeric['volume'].sum())
        if not all(np.isfinite(value) for value in result[start].values()):
            del result[start]
    return result


def prepare_wyckoff_bars(hourly, timeframe, as_of, min_bars=1, native_daily=None):
    """Select the latest valid closed contiguous segment, or explicit unavailable.

    Native daily history is permitted only before the first fully observable UTC
    day of the original hourly coverage. Internal hourly holes cannot be patched
    by another source. Only a contiguous native/hourly seam may extend history.
    This routine does not prove the caller's supplied as-of represents wall time.
    """
    timeframe = str(timeframe).lower()
    if timeframe not in ('1h', '4h', '1d'):
        raise ValueError('Wyckoff timeframe must be 1h, 4h or 1d')
    if not isinstance(min_bars, int) or isinstance(min_bars, bool) or min_bars < 1:
        raise ValueError('min_bars must be a positive integer')
    cutoff = pd.Timestamp(as_of)
    if pd.isna(cutoff):
        raise ValueError('as_of must be a timestamp')
    cutoff = cutoff.tz_localize('UTC') if cutoff.tzinfo is None else cutoff.tz_convert('UTC')
    interval = pd.Timedelta(timeframe)
    latest = cutoff.floor(timeframe) - interval
    empty = pd.DataFrame(columns=OHLCV, index=pd.DatetimeIndex([], tz='UTC'), dtype=float)
    source = 'hourly' if timeframe == '1h' else 'hourly_aggregate'

    def unavailable(reason):
        return WyckoffInput(empty.copy(), 'unavailable', reason, source, None, None)

    try:
        hours = _observations(hourly)
        # Future and unclosed observations cannot affect coverage or validation.
        hours = hours[hours.index + pd.Timedelta(hours=1) <= cutoff]
        valid = _bins(hours, timeframe, pd.Timedelta(hours=1))
        origins = {key: source for key in valid}
        if timeframe == '1d' and native_daily is not None:
            native = _observations(native_daily)
            native = native[native.index + interval <= cutoff]
            # Boundary is computed before any invalid hourly rows are removed.
            if not hours.empty:
                native = native[native.index < hours.index[0].ceil('1d')]
            native_valid = _bins(native, '1d', interval)
            valid.update(native_valid)
            origins.update({key: 'native_daily' for key in native_valid})
            if native_valid:
                source = 'native_daily+hourly_aggregate' if hours.size else 'native_daily'
    except (KeyError, TypeError, ValueError, OverflowError):
        return unavailable('invalid_input_schema_or_timestamp')

    if latest not in valid:
        return unavailable('latest_closed_bin_missing_or_invalid')
    starts = []
    cursor = latest
    while cursor in valid:
        starts.append(cursor)
        cursor -= interval
    starts.reverse()
    used = {origins[key] for key in starts}
    source = next(iter(used)) if len(used) == 1 else 'native_daily+hourly_aggregate'
    if len(starts) < min_bars:
        return unavailable('insufficient_contiguous_history')
    frame = pd.DataFrame([valid[key] for key in starts], index=pd.DatetimeIndex(starts), columns=OHLCV)
    close = latest + interval
    return WyckoffInput(frame, 'available', 'completed_contiguous', source, close, close)
