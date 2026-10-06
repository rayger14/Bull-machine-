"""Frozen offline LC context experiment; no live adapter or search parameters."""
import hashlib
import json
import math
from numbers import Real

import pandas as pd

SUBTYPES = ('upside_expansion_candidate', 'downside_rebound_candidate')
ARMS = ('immediate', 'unconditional_wait', 'context')
MINUTE = pd.Timedelta('1min')


def protocol():
    return {'schema': 'lc-context-policy-v1', 'start': '2024-01-01T00:00:00+00:00',
            'end_exclusive': '2026-09-01T00:00:00+00:00',
            'parent_seed': '2023-12-02T00:00:00+00:00',
            'parent_constructor': 'continuous_saved_N3', 'parent_lookback_days': 30,
            'candidate_atr': 'original_monthly_30day_seed_talib_ATR14',
            'parent_atr': 'continuous_seed_talib_ATR14',
            'stop_atr_multiple': 2.7, 'target_price_r': 2., 'expiry_minutes': 15,
            'deadline_hours': 24, 'risk_budget': 100., 'notional_cap': 50000.,
            'cost_bps': [12, 24], 'delay_seconds': [90, 300],
            'funding_modes': ['adverse_stress', 'zero_diagnostic'],
            'funding_bps': 8., 'funding_hours_utc': [0, 8, 16],
            'bootstrap_draws': 5000, 'bootstrap_seed': 20261002,
            'interval_quantiles': [.025, .975], 'minimum_fills': 50,
            'minimum_filled_months': 12, 'execution_authorized': False,
            'pristine_holdout': False, 'optimization_performed': False}


def seal(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                    allow_nan=False).encode()).hexdigest()


def clock(value):
    at = pd.Timestamp(value)
    if pd.isna(at) or at.tzinfo is None or at != at.floor('min'):
        raise ValueError('aware exact minute clock required')
    return at.tz_convert('UTC')


def positive(value):
    return isinstance(value, Real) and not isinstance(value, bool) and math.isfinite(value) and value > 0


def verify_case(case):
    payload = {k: v for k, v in case.items() if k != 'seal'}
    if case.get('seal') != seal(payload) or case.get('policy_seal') != seal(protocol()):
        raise ValueError('case or policy seal mismatch')
    if case.get('execution_authorized') is not False:
        raise ValueError('offline authority required')


def sealed_case(case):
    return dict(case, seal=seal(case))
