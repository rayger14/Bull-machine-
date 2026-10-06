"""Observed native diagnostics plus independently stateful TWT/R1 research arms."""
from collections import defaultdict, deque
from contextlib import ExitStack, contextmanager
from copy import deepcopy
from pathlib import Path
import hashlib
import math
from numbers import Real
from unittest.mock import patch

import pandas as pd

from engine.archetypes import archetype_instance as native
from scripts.research.engine_signal_replay import EXPECTED_ARCHETYPES
from scripts.research.replay_clock import json_safe
from scripts.research.study_contract import finite_number, stable_id, utc_minute

TWT = 'trap_within_trend'


def directional_permission(value, status):
    return bool(status == 'observed' and not isinstance(value, bool)
                and isinstance(value, Real) and math.isfinite(value) and value >= 1.)


def _status(provenance, feature):
    value = provenance.get(feature, {})
    return value.get('status', 'unknown') if isinstance(value, dict) else value


class _Reads(dict):
    """Instrument derived .get calls without changing their return/default values."""
    def __init__(self, features, provenance):
        super().__init__(features)
        self.provenance, self.reads = provenance, {}

    def get(self, key, default=None):
        value = super().get(key, default)
        self.reads[key] = {'value': json_safe(value),
                           'status': _status(self.provenance, key) if key in self else 'defaulted'}
        return value


def _gate_receipts(config, features, provenance, calls):
    rows = []
    for gate in config.hard_gates:
        key = gate.get('feature', '')
        nan_policy = gate.get('nan_policy', 'fail')
        row = {'gate': deepcopy(gate), 'value': None, 'source_status': 'unknown',
               'derived_inputs': {}, 'error': None}
        if key.startswith('derived:'):
            name = key.split(':', 1)[1]
            record = calls[name].popleft() if calls[name] else {
                'value': None, 'reads': {}, 'error': {'type': 'UnknownDerivedFeature', 'message': name}}
            value = record['value']
            row.update(derived_inputs=record['reads'], error=record['error'], source_status='derived')
            if row['error']:
                row['native_status'] = 'failed_compute' if nan_policy == 'fail' else 'skipped_compute'
                rows.append(row)
                continue
        else:
            value = features.get(key)
            row['source_status'] = _status(provenance, key) if key in features else 'missing'
            if gate.get('frozen_bypass', False) and key in native.FROZEN_FEATURES:
                row['native_status'] = 'frozen_bypass'
                rows.append(row)
                continue
        row['value'] = json_safe(value)
        if value is None or (isinstance(value, float) and value != value):
            row['native_status'] = 'failed_nan' if nan_policy == 'fail' else 'skipped_nan'
        else:
            # Reconstruct primitive comparisons only, never run derived logic again.
            converted = value if isinstance(value, bool) else native._safe_float(value)
            op, threshold = gate.get('op', 'bool_true'), gate.get('value')
            failed = ((op == 'min' and converted < threshold) or (op == 'max' and converted > threshold)
                      or (op == 'bool_true' and not converted) or (op == 'bool_false' and bool(converted))
                      or (op == 'eq' and converted != threshold))
            if op == 'in_range' and isinstance(threshold, (tuple, list)) and len(threshold) == 2:
                failed = converted < threshold[0] or converted > threshold[1]
            row.update(native_status='failed' if failed else 'passed', compared_value=json_safe(converted))
        rows.append(row)
    return rows


@contextmanager
def _observe_gates(arch, provenance, destination):
    original = arch._evaluate_gates

    def call(features):
        calls = defaultdict(deque)
        wrappers = {}
        for name, fn in native.DERIVED_FEATURES.items():
            def traced(values, name=name, fn=fn):
                trace = _Reads(values, provenance)
                try:
                    value = fn(trace)
                except Exception as exc:
                    calls[name].append({'value': None, 'reads': trace.reads,
                                        'error': {'type': type(exc).__name__, 'message': str(exc)}})
                    raise
                calls[name].append({'value': value, 'reads': trace.reads, 'error': None})
                return value
            wrappers[name] = traced
        with patch.dict(native.DERIVED_FEATURES, wrappers):
            result = original(features)
        receipts = _gate_receipts(arch.config, features, provenance, calls)
        counted = [r for r in receipts if r['native_status'] in ('passed', 'failed', 'failed_nan', 'failed_compute')]
        failed = sum(r['native_status'] != 'passed' for r in counted)
        expected = (failed == 0, 1. - failed / len(counted) if counted else 1.)
        if (result[0], result[2]) != expected:
            raise ValueError('gate receipt reconstruction differs from native result')
        destination[:] = receipts
        return result

    with patch.object(arch, '_evaluate_gates', call):
        yield


class _CachedStructure:
    def __init__(self, result):
        self.result = result

    def check_structure(self, **kwargs):
        if kwargs['archetype_name'] != TWT:
            raise ValueError('foreign cached identity')
        return self.result['passed'], self.result['reason']


class HourlyStudyObserver:
    def __init__(self, signal_engine, *, instrument, data_stream_id):
        self.signal_engine = signal_engine
        if not isinstance(instrument, str) or not instrument.strip() or not isinstance(data_stream_id, str) or not data_stream_id.strip():
            raise ValueError('explicit instrument/source stream required')
        self.instrument, self.data_stream_id = instrument, data_stream_id
        if set(signal_engine.engine.archetypes) != EXPECTED_ARCHETYPES:
            raise ValueError('exact all17 source required')
        if signal_engine.engine.structural_checker is None or signal_engine.bar_index != 0:
            raise ValueError('fresh structural source required')
        source = signal_engine.engine.archetypes[TWT]
        if getattr(source, 'defer_cooldown_arm', False):
            raise ValueError('study requires legacy signal-time cooldown')
        self.arms = {name: native.ArchetypeInstance(deepcopy(source.config)) for name in ('baseline', 'repair')}
        root = Path(__file__).resolve().parents[2]
        self.identity_version = stable_id('twt-native-identity', {
            'sources': {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
                        for p in [root/'engine/archetypes/logic.py', root/'engine/archetypes/structural_check.py']},
            'gate_params': signal_engine.engine.structural_checker.gate_params,
        })

    def update(self, features, decision_time, provenance):
        decision = utc_minute(decision_time)
        if (features.get('instrument', self.instrument) != self.instrument
                or features.get('data_stream_id', self.data_stream_id) != self.data_stream_id):
            raise ValueError('foreign hourly instrument/source stream')
        receipts, detected = {}, {}
        with ExitStack() as stack:
            for name, arch in self.signal_engine.engine.archetypes.items():
                receipts[name] = [{'gate': deepcopy(g), 'native_status': 'not_evaluated'} for g in arch.config.hard_gates]
                stack.enter_context(_observe_gates(arch, provenance, receipts[name]))
                original = arch.detect

                def detect(*args, original=original, name=name, **kwargs):
                    detected[name] = (args, kwargs)
                    return original(*args, **kwargs)

                stack.enter_context(patch.object(arch, 'detect', detect))
            diagnostic = self.signal_engine.update(features, decision)

        identity = diagnostic['archetypes'][TWT]['structural']
        blockers = []
        identity_valid = identity is not None and identity['reason'] in ('structural_passed', 'structural_H_failed')
        if identity and identity['reason'].startswith('error:'):
            blockers.append('R1:structural_error')
        elif not identity_valid:
            blockers.append('R1:unqualified_identity')
        passed = identity_valid and identity['passed']
        opportunity = None
        if passed:
            keys = {'instrument': self.instrument, 'source_hour_close': decision.isoformat(), 'archetype': TWT,
                    'native_identity_version': self.identity_version}
            opportunity = {'id': stable_id('r1-native-identity', keys), 'family': 'R1',
                           'origin_time': decision.isoformat(), 'data_stream_id': self.data_stream_id, **keys}
        above = features.get('price_above_ema_50')
        permission = directional_permission(above, _status(provenance, 'price_above_ema_50'))
        atr_key = 'atr_14' if 'atr_14' in features else ('atr' if 'atr' in features else None)
        atr = {'source': atr_key or 'close_times_0.02',
               'status': _status(provenance, atr_key) if atr_key else 'defaulted',
               'value': features[atr_key] if atr_key else features['close'] * .02}
        if passed:
            try:
                finite_number(atr['value'], positive=True)
                if atr['status'] != 'observed':
                    raise ValueError('unobserved')
            except ValueError:
                blockers.append('R1:unqualified_atr')
        arms = {}
        for name, arch in self.arms.items():
            before = arch.last_signal_bar
            local_receipts = [{'gate': deepcopy(g), 'native_status': 'not_evaluated'} for g in arch.config.hard_gates]
            signal, reason = None, 'identity_rejected'
            if passed and (name == 'baseline' or permission):
                args, kwargs = detected[TWT]
                kwargs = dict(kwargs, structural_checker=_CachedStructure(identity))
                ready = arch.can_signal(self.signal_engine.bar_index)
                with _observe_gates(arch, provenance, local_receipts):
                    signal = arch.detect(*args, **kwargs)
                reason = 'emitted' if signal is not None else ('cooldown' if not ready else 'native_gate_fusion_or_liquidity_rejected')
            elif passed:
                reason = 'directional_permission_rejected'
            entry_signal = None
            if signal is not None:
                entry_signal = {'opportunity_id': opportunity['id'], 'family': 'R1', 'arm': name,
                                'decision_time': decision.isoformat(), 'stop': signal.stop_loss,
                                'entry_expiry': (decision + pd.Timedelta('15min')).isoformat(),
                                'exit_deadline': (decision + pd.Timedelta('24h')).isoformat(),
                                'parent_lineage_id': None}
            if any(r.get('error') for r in local_receipts) or any(r.get('error') for r in receipts[TWT]):
                blockers.append('R1:derived_gate_error')
            arms[name] = {'reason': reason, 'signal': entry_signal,
                          'native_signal': deepcopy(vars(signal)) if signal is not None else None,
                          'gate_receipts': local_receipts, 'last_signal_bar_before': before,
                          'last_signal_bar_after': arch.last_signal_bar}
        return {'schema': 'hourly-study-v1', 'decision_time': decision.isoformat(),
                          'native': diagnostic, 'gate_receipts': receipts, 'identity': identity,
                          'opportunity': opportunity, 'directional_permission': permission, 'atr': atr,
                          'arms': arms, 'blockers': sorted(set(blockers)), 'execution_authorized': False}

    def snapshot(self):
        return {'native': self.signal_engine.snapshot(), 'identity_version': self.identity_version,
                'instrument': self.instrument, 'data_stream_id': self.data_stream_id,
                'arms': {name: {'last_signal_bar': arch.last_signal_bar,
                                'cooling_period_bars': arch.cooling_period_bars} for name, arch in self.arms.items()}}
