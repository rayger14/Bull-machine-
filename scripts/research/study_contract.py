"""Frozen offline R1/R3 study policy and small input-boundary primitives."""
from copy import deepcopy
import hashlib
import json
import math
from numbers import Real

import pandas as pd


_PROTOCOL = {
    'schema': 'archetype-study-protocol-v1',
    'canonical_spec_sha256': '006375d94a90ec042d431780bb9de9e275d6527d7bad080e4c5815b37557ce28',
    'active_hypotheses': ['R1', 'R3'], 'parked_hypotheses': ['R2'],
    'seed': '2023-12-02T00:00:00+00:00',
    'start': '2024-01-01T00:00:00+00:00',
    'pilot_end': '2024-02-01T00:00:00+00:00',
    'end_exclusive': '2026-08-31T00:00:00+00:00',
    'source_end_exclusive': '2026-09-01T00:00:00+00:00',
    'source_sha256': '5b8a4533f70b8ccd0dc533984469e6886ec73ce315d48f150c667ccb97e84035',
    'availability_basis': 'historical_bar_close_assumption',
    'cost_bps': [12, 24], 'delay_seconds': [5, 65],
    'risk_budget': 100.0, 'notional_cap': 50000.0, 'target_r': 2.0,
    'funding_stress_bps': 8.0, 'funding_hours_utc': [0, 8, 16],
    'bootstrap_draws': 5000, 'bootstrap_seed': 20260930,
    'interval_quantiles': [0.0083333333, 0.9916666667],
    'minimum_repair_fills': 50, 'minimum_origin_months': 12,
    'execution_authorized': False,
}


def protocol():
    return deepcopy(_PROTOCOL)


def finite_number(value, positive=False):
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError('finite nonboolean numeric value required')
    try:
        result = float(value)
    except (OverflowError, ValueError, TypeError) as exc:
        raise ValueError('finite numeric value required') from exc
    if not math.isfinite(result) or (positive and result <= 0):
        raise ValueError('finite positive value required' if positive else 'finite value required')
    return result


def utc_minute(value):
    if value is None or isinstance(value, (bool, Real)):
        raise ValueError('explicit UTC-aware minute clock required')
    try:
        stamp = pd.Timestamp(value)
    except (ValueError, TypeError, OverflowError) as exc:
        raise ValueError('invalid clock') from exc
    if pd.isna(stamp) or stamp.tz is None:
        raise ValueError('UTC-aware clock required')
    stamp = stamp.tz_convert('UTC')
    if stamp != stamp.floor('min'):
        raise ValueError('minute-aligned clock required')
    return stamp


def stable_id(kind, fields):
    if not isinstance(kind, str) or not kind.strip() or not isinstance(fields, dict):
        raise ValueError('nonempty identity kind and field mapping required')
    try:
        raw = json.dumps({'kind': kind, 'fields': fields}, sort_keys=True,
                         separators=(',', ':'), allow_nan=False)
    except (ValueError, TypeError, OverflowError) as exc:
        raise ValueError('identity requires finite JSON values') from exc
    return kind + ':' + hashlib.sha256(raw.encode('utf-8')).hexdigest()
