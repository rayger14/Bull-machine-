"""Small, outcome-free same-stream evidence builders for offline entry studies."""
import math

import numpy as np
import pandas as pd

from scripts.research.minute_sweep_validation import detect_events, validate_bars


LAYERS = {'1m': (1,30), '5m': (5,12), '15m': (15,16),
          '1h': (60,24), '4h': (240,18), '1d': (1440,10)}


def _numeric_prices(bars):
    if any(k not in bars or getattr(bars[k].dtype,'kind',None) not in ('i','u','f')
           for k in ('open','high','low','close')):
        raise ValueError('real numeric OHLC dtypes required')


def _clock(value):
    at=pd.Timestamp(value)
    if pd.isna(at) or at.tz is None or at.value % pd.Timedelta('1min').value:
        raise ValueError('timezone-aware exact minute required')
    return at.tz_convert('UTC')


def completed_candles(bars, decision_time, period_minutes, count):
    """Return [open_time,O,H,L,C,V] rows; never consume a developing bucket."""
    at=_clock(decision_time)
    if (type(period_minutes) is not int or period_minutes<=0 or 1440%period_minutes
            or type(count) is not int or count<=0):
        raise ValueError('positive count and whole-day-dividing period required')
    if not isinstance(bars.index,pd.DatetimeIndex) or bars.index.tz is None:
        raise ValueError('timezone-aware minute source required')
    end=at.floor(pd.Timedelta(minutes=period_minutes))
    start=end-pd.Timedelta(minutes=period_minutes*count)
    window=bars.loc[(bars.index>=start)&(bars.index<end)]
    expected=pd.date_range(start,end,freq='min',inclusive='left')
    if not window.index.equals(expected):
        raise ValueError('missing, duplicate, unsorted or off-grid constituents')
    _numeric_prices(window)
    validate_bars(window)
    if 'volume' not in window or getattr(window.volume.dtype,'kind',None) not in ('i','u','f'):
        raise ValueError('numeric volume required')
    if not np.isfinite(window.volume.to_numpy()).all() or (window.volume<0).any():
        raise ValueError('finite nonnegative volume required')
    grouped=window.resample(str(period_minutes)+'min',origin='epoch').agg(
        dict(open='first',high='max',low='min',close='last',volume='sum'))
    if len(grouped)!=count or not np.isfinite(grouped.to_numpy(dtype=float)).all():
        raise ValueError('invalid aggregate')
    return [[t.isoformat()]+[float(v) for v in row] for t,row in grouped.iterrows()]


def minute_setup_witness(bars, event, decision_time):
    """Recompute selected membership using ONLY the original seed through reclaim.

    The prefix replay proves membership under this detector/reset convention,
    not independent correctness of the strategy or authentic live receipt time.
    Numeric pivot/sequence witnesses permit inspection without sending all bars.
    """
    at=_clock(decision_time)
    index_keys=('pivot_idx','confirmed_idx','sweep_idx','reclaim_idx')
    keys=set(index_keys)|{'level','sweep_low','touches'}
    if not isinstance(event,dict) or set(event)!=keys:
        raise ValueError('exact detector event fields required')
    if any(type(event[k]) is not int for k in index_keys+('touches',)):
        raise ValueError('integer event indices/touches required')
    i,c,s,r=(event[k] for k in index_keys)
    if not 15<=i<c<=s<=r<len(bars) or c!=i+15:
        raise ValueError('invalid detector chronology')
    for k in ('level','sweep_low'):
        v=event[k]
        if type(v) not in (int,float) or not math.isfinite(v) or v<=0:
            raise ValueError('finite positive event price required')
    past=bars.iloc[:r+1]
    _numeric_prices(past)
    validate_bars(past)
    if past.index[-1]+pd.Timedelta('1min')!=at:
        raise ValueError('reclaim must just have closed')
    detected=detect_events(past)
    positions=[j for j,e in enumerate(detected) if e==event]
    if len(positions)!=1:
        raise ValueError('event does not match causal prefix detector')
    j=positions[0]
    prev=detected[j-1] if j else None
    lows=past.low.to_numpy()
    pivots=np.flatnonzero((past.low.rolling(31,center=True).min()==past.low).to_numpy())
    matching=[int(k) for k in pivots if i-1440<=k<=i-30 and abs(lows[k]-lows[i])/lows[i]<=.001]

    def pivot(k):
        return dict(open_time=past.index[k].isoformat(),low=float(lows[k]),
                    window_start=past.index[k-15].isoformat(),
                    window_last_open=past.index[k+15].isoformat(),
                    available_at=(past.index[k+15]+pd.Timedelta('1min')).isoformat(),
                    window_min=float(lows[k-15:k+16].min()))

    def candle(k):
        return dict(open_time=past.index[k].isoformat(),
                    available_at=(past.index[k]+pd.Timedelta('1min')).isoformat(),
                    **{name:float(past.iloc[k][name]) for name in ('open','high','low','close')})

    return dict(source_seed=past.index[0].isoformat(),decision_time=at.isoformat(),
        prefix_membership_recomputed=True,pivot=pivot(i),prior_touches=[pivot(k) for k in matching],
        touches=event['touches'],child_level=event['level'],sweep_threshold=event['level']*.9998,
        pre_sweep_min=float(lows[c:s].min()) if s>c else None,
        first_sweep=candle(s),reclaim=candle(r),sweep_low=event['sweep_low'],
        sequence_bars=[candle(k) for k in range(s,r+1)],
        preceding_selected_sweep=past.index[prev['sweep_idx']].isoformat() if prev else None,
        sweep_spacing_minutes=s-prev['sweep_idx'] if prev else None,
        previous_reclaim=past.index[prev['reclaim_idx']].isoformat() if prev else None)
