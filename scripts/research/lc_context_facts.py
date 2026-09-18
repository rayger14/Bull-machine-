"""Descriptive LC context, without a permission rule or outcome access.

Consumes sealed build_lc_packet output from a caller-verified reconstruction.
Checks the operands it describes, not raw archive authenticity or full ledger
reconstruction. A valid seal is integrity, not independent source authentication.
Old reference conditions are intentionally neither read nor changed.
"""
from copy import deepcopy
import math

import pandas as pd

from scripts.research.conditional_assessment import digest
from scripts.research.conditional_entry import _clock
from scripts.research.lc_master_assessment import _check_seal, _reconstructed


VERSION = 'lc_context_facts_v1'
OHLC = ('open', 'high', 'low', 'close')


def _number(value, positive=True):
    return (type(value) in (int, float) and math.isfinite(value)
            and (value > 0 if positive else value >= 0))


def _prices(row):
    return (all(_number(row.get(k)) for k in OHLC)
            and _number(row.get('volume'), False)
            and row['low'] <= min(row['open'], row['close'])
            and row['high'] >= max(row['open'], row['close']))


def _hour(view, opened, reconstruction):
    try:
        features, source = view['features'], view['source_candle']
        if (not reconstruction or view['validated'] is not True
                or _clock(view['available_at']) != opened + pd.Timedelta('1h')
                or _clock(features['timestamp']) != opened
                or _clock(source['open_time']) != opened
                or not _prices(features) or not _prices(source)
                or any(features[k] != source[k] for k in OHLC + ('volume',))):
            return None
        return source
    except (KeyError, ValueError, TypeError, AttributeError):
        return None


def _hourly(current, previous):
    result = dict(status='unknown', close_relation=None, swept_prior_low=None,
                  swept_prior_high=None, reclaimed_prior_low=None,
                  rejected_prior_high=None, inside_bar=None)
    if current is None or previous is None:
        return result
    cl, lo, hi = current['close'], previous['low'], previous['high']
    return dict(result, status='known', close_relation=(
        'below_prior_low' if cl < lo else 'above_prior_high' if cl > hi
        else 'inside_or_boundary'), swept_prior_low=current['low'] < lo,
        swept_prior_high=current['high'] > hi,
        reclaimed_prior_low=current['low'] < lo < cl,
        rejected_prior_high=current['high'] > hi > cl,
        inside_bar=current['low'] >= lo and current['high'] <= hi)


def _sequence(packet, tf, period, decision, reconstruction):
    result = dict(status='unknown', candles=[], higher_low=None, higher_close=None,
                  last_body=None, last_close_location=None,
                  postdecision_confirmation_evaluated=False)
    try:
        columns = packet['candle_columns']
        if (not reconstruction or columns != ['open_time', 'open', 'high', 'low', 'close', 'volume']):
            return result
        rows = packet['evidence'][tf][-2:]
        if len(rows) != 2 or any(len(r) != 6 for r in rows):
            return result
        candles = [dict(zip(columns, r)) for r in rows]
        end = decision.floor(pd.Timedelta(minutes=period))
        if any(not _prices(c) or _clock(c['open_time']) != end-pd.Timedelta(minutes=period*(2-i))
               for i, c in enumerate(candles)):
            return result
        a, b = candles
        width = b['high']-b['low']
        return dict(result, status='known', candles=deepcopy(candles),
                    higher_low=b['low'] > a['low'], higher_close=b['close'] > a['close'],
                    last_body='up' if b['close'] > b['open'] else 'down' if b['close'] < b['open'] else 'flat',
                    last_close_location=(b['close']-b['low'])/width if width else None)
    except (KeyError, ValueError, TypeError, AttributeError, IndexError):
        return result


def _parent(view, anchor, setup, decision, provenance, current, previous, stop):
    result = dict(evidence_status='unknown', pre_setup='unknown', lifecycle='unknown',
                  lifecycle_scope='strict_before_bound_only', decision_state=None,
                  bound=None, updates=[], child_nested=None, position_in_bound=None,
                  ceiling_distance_r=None, distance_basis='indicative_close_to_frozen_stop',
                  distance_is_unobstructed_room=False)
    try:
        if (not _reconstructed(provenance) or view['status'] not in ('pass', 'fail')
                or view['anchor'] != anchor or view['pivot_n'] != 3
                or _clock(view['strict_before']) != setup):
            return result
        updates = view['updates']
        if [_clock(u['available_at']) for u in updates] != [setup, decision]:
            return result
        if any(u['source_break_direction'] not in (None, 'up', 'down')
               or u['post_state'] not in ('active', 'forming', 'broken_up', 'broken_down')
               for u in updates):
            return result
        bound = view['bound']
        if bound is None:
            if (view['status'] != 'fail' or view['reasons'] != ['absent_parent']
                    or view['lineage_broken'] is not False):
                return result
            return dict(result, evidence_status='known', pre_setup='absent',
                        lifecycle='absent', decision_state=updates[-1]['post_state'],
                        updates=deepcopy(updates))
        low, high = bound['range_low'], bound['range_high']
        if (not _number(low) or not _number(high) or low >= high
                or _clock(bound['available_at']) >= setup):
            return result
        for side, level in (('low', low), ('high', high)):
            pivots = [p for p in view['pivots'] if p['id'] == bound[side+'_pivot_id']]
            if len(pivots) != 1:
                return result
            pivot = pivots[0]
            if (pivot['side'] != side or not _number(pivot['level']) or pivot['level'] != level
                    or pivot['anchor_timeframe'] != anchor or pivot['pivot_n'] != 3
                    or _clock(pivot['available_at']) > _clock(bound['available_at'])
                    or any(pivot[k] != provenance[k] for k in ('instrument', 'data_stream_id'))):
                return result
        directions = {u['source_break_direction'] for u in updates
                      if u['pre_lineage_id'] == bound['lineage_id']
                      and u['source_break_direction'] is not None}
        if (view['lineage_broken'] is not bool(directions)
                or view['status'] != ('fail' if directions else 'pass')):
            return result
        lifecycle = ('intact' if not directions else 'broken_'+next(iter(directions))
                     if len(directions) == 1 else 'broken_multiple')
        result.update(evidence_status='known', pre_setup='present', lifecycle=lifecycle,
                      decision_state=updates[-1]['post_state'],
                      bound=deepcopy(bound), updates=deepcopy(updates))
        if previous is not None:
            result['child_nested'] = low <= previous['low'] < previous['high'] <= high
        if current is not None:
            cl = current['close']
            result['position_in_bound'] = (cl-low)/(high-low)
            if _number(stop) and stop < cl:
                result['ceiling_distance_r'] = (high-cl)/(cl-stop)
        return result
    except (KeyError, ValueError, TypeError, AttributeError):
        return result


def describe_lc_context(packet):
    """Return source-bound facts only; invalid operands stay unknown, never reject.

    The caller must keep the source packet/manifest and verify reconstruction.
    This helper does not grant authorization, redefine native eligibility, fit
    thresholds, revalidate the full archive, or compute post-decision outcomes.
    """
    _check_seal(packet)
    decision, setup = _clock(packet['decision_time']), _clock(packet['setup_open'])
    if setup != setup.floor('h') or decision != setup+pd.Timedelta('1h'):
        raise ValueError('invalid LC clocks')
    provenance = packet['provenance']
    reconstructed = _reconstructed(provenance)
    try:
        available = packet['feature_availability']
        clocks_valid = (available['observations_valid'] is True and
                        all(_clock(v) <= decision for v in available['observation_visible_at'].values()))
    except (KeyError, ValueError, TypeError, AttributeError):
        clocks_valid = False
    current = _hour(packet['current'], setup, reconstructed and clocks_valid)
    previous = _hour(packet['previous'], setup-pd.Timedelta('1h'), reconstructed)
    hourly = _hourly(current, previous)
    # A distance is withheld if either hourly operand is untrusted, even though
    # independently valid parent facts remain reportable.
    distance_current = current if hourly['status'] == 'known' else None
    stop = packet['plan'].get('stop')
    native = packet['native']
    direction = (native.get('diagnostic') or {}).get('native_signal') or {}
    native_long = (direction.get('direction') == 'long' if reconstructed and clocks_valid
                   and native.get('source_valid') is True and direction else None)
    result = dict(version=VERSION, case_id=packet['case_id'], candidate_id=packet['candidate_id'],
                  source_packet_sha256=digest(packet), decision_time=decision.isoformat(),
                  native_long=native_long, hourly=hourly, execution_authorized=False,
                  source_authentication='caller-verified reconstruction; not live receipt authentication')
    for key, anchor in (('parent_4h', '4H'), ('parent_1d', '1D')):
        result[key] = _parent(packet['evidence'].get(key, {}), anchor, setup, decision,
                              provenance, distance_current, previous, stop)
    for tf, period in (('1m', 1), ('5m', 5)):
        result['last_two_'+tf] = _sequence(packet, tf, period, decision, reconstructed)
    return result
