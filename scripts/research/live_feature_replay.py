"""Offline current-source LFC experiment, NOT production startup/book parity.

Run in a dedicated single-threaded research process: network denial temporarily
patches process socket calls. Minute bars advance LFC only on complete hours.
Macro inputs are precomputed snapshots; their provider transforms are excluded.
"""
from contextlib import contextmanager, ExitStack
from copy import deepcopy
from datetime import datetime
import hashlib
import importlib.util
from importlib import metadata
import logging
import math
from pathlib import Path
import platform
import socket
import sys
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from scripts.research.replay_clock import digest, duration, replay, utc

SOURCE = ROOT / 'bin/live/live_feature_computer.py'
CHANNELS = {'derivatives_snapshot', 'macro_features', 'candle_funding_rate',
            'alt_basket_ret_4h', 'stables_rot_rising', 'cme_oi_features'}
MACRO_KEYS = {'VIX_Z', 'DXY_Z', 'YIELD_10Y', 'YIELD_5Y', 'YIELD_CURVE',
              'GOLD_Z', 'OIL_Z', 'BTC.D', 'USDT.D', 'USDC.D', 'FEAR_GREED',
              'fear_greed_norm', 'FEAR_GREED_LABEL', 'eth_btc_ratio',
              'total_market_cap', 'gold_price', 'oil_price'}
CME_KEYS = {'cme_oi_value', 'cme_oi_change_24h', 'cme_oi_price_divergence', 'cme_oi_age_days'}


def validate_channel(name, value):
    if name in ('derivatives_snapshot', 'macro_features', 'cme_oi_features'):
        if not isinstance(value, dict):
            raise ValueError(f'{name} must be a dictionary')
        allowed = MACRO_KEYS if name == 'macro_features' else CME_KEYS if name == 'cme_oi_features' else None
        if allowed is not None and set(value) - allowed:
            raise ValueError(f'Undeclared {name} fields: {sorted(set(value)-allowed)}')


@contextmanager
def deny_network():
    """Fail after the guarded scope even if upstream code catches our denial."""
    attempts = []

    def denied(*args, **kwargs):
        attempts.append('Network access attempted')
        raise RuntimeError(attempts[-1])

    with ExitStack() as stack:
        for obj, name in ((socket, 'create_connection'), (socket, 'getaddrinfo'),
                          (socket.socket, 'connect'), (socket.socket, 'connect_ex'),
                          (socket.socket, 'sendto')):
            stack.enter_context(patch.object(obj, name, denied))
        try:
            yield
        finally:
            if attempts:
                raise RuntimeError('Network access attempted inside offline replay')


class RecordedProvider:
    """Only the transport boundary is replaced; derivative formulas remain real."""
    def __init__(self):
        self.value = {}

    def fetch_all_current(self):
        return deepcopy(self.value)

    def get_features(self):
        return deepcopy(self.value)


def state_view(value, unsupported, active=None):
    """Inspectable state, never pickle/repr addresses or an opaque restore blob."""
    active = set() if active is None else active
    if isinstance(value, np.generic):
        return state_view(value.item(), unsupported, active)
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, (datetime, pd.Timestamp)):
        return pd.Timestamp(value)
    if isinstance(value, pd.Timedelta):
        return {'timedelta_ns': value.value}
    if id(value) in active:
        unsupported.add('cyclic_state')
        return {'unsupported': 'cyclic_state'}
    active.add(id(value))
    try:
        convert = lambda v: state_view(v, unsupported, active)
        if isinstance(value, pd.DataFrame):
            return {'dataframe': convert(value.to_dict('split'))}
        if isinstance(value, pd.Series):
            return {'series': convert(value.to_dict()), 'name': convert(value.name)}
        if isinstance(value, np.ndarray):
            return {'array': convert(value.tolist()), 'dtype': str(value.dtype)}
        if isinstance(value, dict):
            return {k: convert(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [convert(v) for v in value]
        if isinstance(value, set):
            return sorted([convert(v) for v in value], key=digest)
        cls = type(value)
        if any(getattr(base, '__slots__', ()) for base in cls.__mro__):
            unsupported.add(cls.__module__ + '.' + cls.__qualname__ + ':slotted_state')
        if hasattr(value, '__dict__'):
            return {'class': cls.__module__ + '.' + cls.__qualname__,
                    'attributes': convert(vars(value))}
        name = cls.__module__ + '.' + cls.__qualname__
        unsupported.add(name)
        return {'unsupported': name}
    finally:
        active.remove(id(value))


def source_manifest(module, computer):
    paths = {SOURCE, Path(__file__), ROOT / 'scripts/research/replay_clock.py'}
    # Conservatively bind every engine and live source, including lazy imports.
    paths.update((ROOT / 'engine').rglob('*.py'))
    paths.update((ROOT / 'bin/live').rglob('*.py'))
    model = ROOT / 'models/logistic_regime_v4_no_funding_stratified.pkl'
    hashes = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
              for p in sorted(paths)}
    hashes[str(model.relative_to(ROOT))] = hashlib.sha256(model.read_bytes()).hexdigest() if model.exists() else None
    flags = {k: v for k, v in vars(module).items() if k.endswith('_AVAILABLE')}
    versions = {}
    for package in ('pandas', 'numpy', 'TA-Lib', 'scipy', 'joblib', 'scikit-learn', 'requests', 'yfinance'):
        try:
            versions[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            versions[package] = None
    actual_model = getattr(computer._prob_detector, 'crisis_model', None)
    return dict(files=hashes, capabilities=flags, python=platform.python_version(),
                dependencies=versions,
                instantiated_model=type(actual_model).__module__ + '.' + type(actual_model).__qualname__,
                model_file_present=model.exists())


class LiveFeatureProcessor:
    def __init__(self, timeframe='1h'):
        self.step = duration(timeframe)
        if self.step not in (pd.Timedelta('1min'), pd.Timedelta('1h')):
            raise ValueError('Only hourly native or minute aggregation clocks are supported')
        # Separate globals: never alter the production module's clock/API flags.
        spec = importlib.util.spec_from_file_location('research_frozen_lfc', SOURCE)
        self.module = importlib.util.module_from_spec(spec)
        with deny_network():
            spec.loader.exec_module(self.module)
            for name in ('HAS_BINANCE_API', 'HAS_OKX_API', 'HAS_COINGLASS_API'):
                setattr(self.module, name, False)
            self.now = pd.Timestamp('1970-01-01', tz='UTC')
            self.module.time = SimpleNamespace(time=lambda: self.now.timestamp())
            self.fc = self.module.LiveFeatureComputer()
        self.derivatives = RecordedProvider()
        self.macro = RecordedProvider()
        self.fc.binance_api = self.derivatives
        self.fc._derivatives_source = 'recorded_snapshot'
        self.fc._macro_fetcher = self.macro
        self.manifest = source_manifest(self.module, self.fc)
        self.contract_id = digest(dict(policy='lfc-hourly-reference-v1',
                                       timeframe=str(self.step), sources=self.manifest,
                                       startup='cold_replay', deep_daily=False))
        self.context = {tf: dict(developing=None, completed=None, incomplete=None)
                        for tf in ('1h', '4h', '1d')}
        self.pending = {tf: None for tf in self.context}
        self.hourly_updates = 0
        self.features = None
        self.features_at = None
        self.feature_input_values_hash = None
        self.last_open = None
        self.blockers = {'full_decision_book_not_replayed', 'cold_start_not_live_state',
                         'source_availability_assertions_not_independently_verified',
                         'macro_provider_transforms_not_replayed'}

    def _advance_context(self, candle):
        opened = utc(candle['timestamp'])
        for tf, view in self.context.items():
            target = duration(tf)
            start = opened.floor(target)
            row = self.pending[tf]
            if row is None:
                row = dict(open_time=str(start), close_time=str(start+target),
                           first_constituent=str(opened), open=candle['open'],
                           high=candle['high'], low=candle['low'], close=candle['close'],
                           volume=candle['volume'], constituents=1, complete=False)
            else:
                row['high'] = max(row['high'], candle['high'])
                row['low'] = min(row['low'], candle['low'])
                row['close'] = candle['close']
                row['volume'] += candle['volume']
                row['constituents'] += 1
            if self.now == start + target:
                row['complete'] = (row['constituents'] == int(target/self.step)
                                   and utc(row['first_constituent']) == start)
                view['completed' if row['complete'] else 'incomplete'] = row
                view['developing'] = None
                self.pending[tf] = None
            else:
                self.pending[tf] = row
                if utc(row['first_constituent']) != start:
                    view['incomplete'] = row
                    view['developing'] = None
                else:
                    view['developing'] = row

    def update(self, candle, observations):
        unknown = set(observations) - CHANNELS
        if unknown:
            raise ValueError(f'Undeclared LFC input channels: {sorted(unknown)}')
        for name, value in observations.items():
            validate_channel(name, value)
        opened = utc(candle['timestamp'])
        self.now = utc(candle['close_time'])
        if self.now - opened != self.step or (self.last_open is not None and opened != self.last_open + self.step):
            raise ValueError('Processor cadence/continuity mismatch')
        self.last_open = opened
        self._advance_context(candle)
        updated = False
        if self.now == self.now.floor('1h'):
            hour = self.context['1h']['completed']
            if hour is None or utc(hour['close_time']) != self.now:
                self.blockers.add('incomplete_hour_not_ingested')
            else:
                raw = observations.get('derivatives_snapshot', {})
                macro = observations.get('macro_features', {})
                if not isinstance(raw, dict) or not isinstance(macro, dict):
                    raise ValueError('Derivative and macro snapshots must be dictionaries')
                if not raw:
                    self.blockers.add('missing_derivatives_snapshot')
                for key in ('oi_value', 'oi_change_4h', 'oi_change_24h', 'funding_rate', 'ls_ratio'):
                    if key not in raw:
                        self.blockers.add('defaulted_derivative:' + key)
                for key, value in raw.items():
                    if not isinstance(value, (int, float, np.number)) or not math.isfinite(value):
                        self.blockers.add('invalid_derivative:' + key)
                if not (('taker_buy_vol_1h' in raw and 'taker_sell_vol_1h' in raw) or 'taker_buy_sell_ratio' in raw):
                    self.blockers.add('defaulted_derivative:taker_imbalance')
                if not macro:
                    self.blockers.add('missing_macro_snapshot')
                self.derivatives.value = deepcopy(raw)
                self.macro.value = deepcopy(macro)
                hourly = {k: hour[k] for k in ('open', 'high', 'low', 'close', 'volume')}
                hourly['timestamp'] = utc(hour['open_time'])
                for key in ('alt_basket_ret_4h', 'stables_rot_rising'):
                    if key in observations:
                        hourly[key] = observations[key]
                if 'candle_funding_rate' in observations:
                    hourly['funding_rate'] = observations['candle_funding_rate']
                with deny_network():
                    self.fc.set_cme_oi_features(observations.get('cme_oi_features', {}))
                    self.features = self.fc.update(hourly).to_dict()
                self.features_at = self.now
                self.feature_input_values_hash = digest(observations)
                self.hourly_updates += 1
                updated = True
        return dict(features=deepcopy(self.features), hourly_updated=updated,
                    features_available_at=str(self.features_at) if self.features_at is not None else None,
                    feature_age_seconds=(self.now-self.features_at).total_seconds() if self.features_at is not None else None,
                    feature_input_values_hash=self.feature_input_values_hash,
                    feature_timeframe='1h', context=deepcopy(self.context),
                    context_note='LFC tf4h/tf1d features include developing buckets; context view is separate',
                    full_pipeline_certified=False)

    def snapshot(self):
        unsupported = set()
        computer = state_view(self.fc, unsupported)
        if unsupported:
            self.blockers.add('unsupported_state_types')
        return dict(hourly_updates=self.hourly_updates, computer=computer,
                    context=deepcopy(self.context), pending=deepcopy(self.pending), last_open=self.last_open,
                    features_at=self.features_at, features=deepcopy(self.features),
                    feature_input_values_hash=self.feature_input_values_hash,
                    unsupported_state_types=sorted(unsupported), blockers=sorted(self.blockers),
                    state_scope='Current source cold-start experiment; complete prehistory replay, not production state restoration')


def run_replay(bars, observations=(), *, instrument, timeframe, emit_from=None, checkpoint=None):
    """Safe report boundary: generic clock success never certifies full LFC/book."""
    observations = list(observations)
    for record in observations:
        if record.feature not in CHANNELS:
            raise ValueError(f'Undeclared LFC input channel: {record.feature}')
        validate_channel(record.feature, record.value)
    processors = []

    def factory():
        processor = LiveFeatureProcessor(timeframe)
        processors.append(processor)
        return processor

    with deny_network():
        result = replay(bars, observations, factory, instrument=instrument,
                        timeframe=timeframe, emit_from=emit_from, checkpoint=checkpoint)
    processor = processors[0]
    result['clock_certified'] = result['certified']
    result['certified'] = False
    result['full_pipeline_certified'] = False
    for row in result['rows']:
        row['clock_certified'] = row.pop('certified')
        row['certified'] = False
    # Clock IDs describe inputs visible NOW. Held features need their own IDs
    # from the hour when they were actually computed, including during warmup.
    selected, position = {}, 0
    outputs = [row['output'] for row in result['rows']]
    outputs.append(dict(features_available_at=result['state']['features_at']))
    for output in outputs:
        at = output['features_available_at']
        if at is not None:
            at = utc(at)
            while position < len(observations) and observations[position].visible_at() <= at:
                record = observations[position]
                selected[record.feature] = record
                position += 1
        output['feature_observation_ids'] = {k: v.id for k, v in selected.items()}
        output['feature_observation_visible_at'] = {k: str(v.visible_at()) for k, v in selected.items()}
    result['state'].update({k: outputs[-1][k] for k in ('feature_observation_ids', 'feature_observation_visible_at')})
    result['blockers'] = sorted(processor.blockers)
    result['source_manifest'] = processor.manifest
    result['contract_id'] = processor.contract_id
    result['scope'] = 'Actual hourly LFC source; offline transports, cold start, no minute-native trigger or decision/book replay'
    return result
