"""Offline engine signal generation, excluding runner entry/book/execution.

Observers wrap existing calls once, record their outputs, then restore methods.
No second detection pass, no changed gates or cooldown/dedup policies.
"""
from collections import deque
from contextlib import ExitStack
from copy import deepcopy
import hashlib
from importlib import metadata
import json
from pathlib import Path
import sys
from unittest.mock import patch

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from scripts.research.live_feature_replay import LiveFeatureProcessor, CHANNELS, deny_network, state_view, validate_channel
from scripts.research.replay_clock import digest, replay, utc

EXPECTED_ARCHETYPES = {
    'confluence_breakout', 'exhaustion_reversal', 'failed_continuation',
    'funding_divergence', 'fvg_continuation', 'liquidity_compression',
    'liquidity_sweep', 'liquidity_vacuum', 'long_squeeze', 'oi_divergence',
    'order_block_retest', 'retest_cluster', 'spring', 'trap_within_trend',
    'volume_fade_chop', 'whipsaw', 'wick_trap',
}


def resolved(path):
    path = Path(path)
    return path.resolve() if path.is_absolute() else (ROOT/path).resolve()


class SignalEngine:
    def __init__(self, config=ROOT/'configs/champion_paper.json'):
        config = resolved(config)
        self.config = deepcopy(json.loads(config.read_text()))
        if any(self.config.get(key, False) for key in ('use_ml_fusion', 'use_kelly_sizing')):
            raise ValueError('ML fusion and Kelly extensions are outside this signal replay contract')
        self.config.setdefault('structural_checks', {})['mode_context'] = 'live'
        self.config['structural_checks'].setdefault('enabled', True)
        directory = resolved(self.config.get('archetype_config_dir', 'configs/archetypes'))
        regime_cfg = self.config.get('regime_classifier', {})
        model = resolved(regime_cfg.get('model_path') or 'models/logistic_regime_v4_no_funding_stratified.pkl')
        calibrator = ROOT/'models/confidence_calibrator_v1.pkl'
        with deny_network():
            from engine.integrations.isolated_archetype_engine import IsolatedArchetypeEngine
            from engine.context.regime_service import RegimeService, REGIME_MODE_PROBABILISTIC
            regime = RegimeService(mode=REGIME_MODE_PROBABILISTIC, model_path=str(model),
                                   calibrator_path=str(calibrator)) if regime_cfg.get('enabled', False) else None
            self.engine = IsolatedArchetypeEngine(archetype_config_dir=str(directory),
                portfolio_config=self.config.get('portfolio_allocation', {}),
                regime_service=regime, enable_regime=regime_cfg.get('enabled', False),
                regime_model_path=str(model), config=self.config)
        if set(self.engine.archetypes) != EXPECTED_ARCHETYPES or self.config.get('disabled_archetypes'):
            raise ValueError('This research contract requires all 17 enabled archetypes')
        self.buffer = deque(maxlen=500)
        self.bar_index = 0
        self.last_open = None
        self.blockers = {'runner_threshold_entry_book_not_replayed', 'cold_start_not_live_state',
                         'no_profitability_certificate'}
        paths = {config, Path(__file__), model, calibrator,
                 ROOT/'scripts/research/live_feature_replay.py', ROOT/'scripts/research/replay_clock.py',
                 ROOT/'bin/live/v11_shadow_runner.py'} | set(directory.glob('*.yaml')) | set(directory.glob('*.yml'))
        paths.update((ROOT/'engine').rglob('*.py'))
        dependencies = {}
        for package in ('numpy', 'pandas', 'scipy', 'joblib', 'scikit-learn', 'PyYAML'):
            try:
                dependencies[package] = metadata.version(package)
            except metadata.PackageNotFoundError:
                dependencies[package] = None
        self.manifest = dict(
            files={str(p): hashlib.sha256(p.read_bytes()).hexdigest() if p.exists() else None for p in sorted(paths)},
            effective_config=self.config, archetype_order=list(self.engine.archetypes), dependencies=dependencies,
            model_class=type(getattr(getattr(regime, 'probabilistic_detector', None), 'crisis_model', None)).__qualname__,
            calibrator_class=type(getattr(regime, 'confidence_calibrator', None)).__qualname__,
            structural_mode=getattr(self.engine.structural_checker, 'mode', None))
        if regime_cfg.get('enabled', False) and not model.exists():
            self.blockers.add('regime_model_missing_source_mock_fallback')
        if 'MockCrisisModel' in self.manifest['model_class']:
            self.blockers.add('regime_model_mock_fallback')
        if not calibrator.exists():
            self.blockers.add('confidence_calibrator_missing')
        self.contract_id = digest(self.manifest)

    def update(self, features, decision_time):
        opened = utc(features['timestamp'])
        decision_time = utc(decision_time)
        if opened != opened.floor('1h') or decision_time != opened + pd.Timedelta('1h'):
            raise ValueError('Signals require one completed hourly feature vector')
        if self.last_open is not None and opened != self.last_open + pd.Timedelta('1h'):
            raise ValueError('Duplicate/out-of-order/gapped signal feature history')
        self.last_open = opened
        self.bar_index += 1
        bar = pd.Series(deepcopy(features), name=opened)
        self.buffer.append(bar.copy())
        previous = self.buffer[-2] if len(self.buffer) >= 2 else None
        lookback = pd.DataFrame(list(self.buffer)) if len(self.buffer) > 1 else None
        outcomes = {name: dict(detect_calls=0, structural=None, gates=None,
                              fusion_before_gate_penalty=None, native_signal=None,
                              can_signal_before=arch.can_signal(self.bar_index),
                              last_signal_bar_before=arch.last_signal_bar,
                              gate_mode=arch.config.gate_mode,
                              entry_threshold=arch.config.entry_threshold)
                    for name, arch in self.engine.archetypes.items()}

        def observe(original, record):
            def call(*args, **kwargs):
                value = original(*args, **kwargs)
                record(value, args, kwargs)
                return value
            return call

        def reject_allocation(*args, **kwargs):
            raise RuntimeError('Allocation is prohibited in signal-only replay')

        with deny_network(), ExitStack() as stack:
            stack.enter_context(patch.object(self.engine, 'allocate', reject_allocation))
            stack.enter_context(patch.object(self.engine.portfolio_allocator, 'allocate', reject_allocation))
            checker = self.engine.structural_checker
            if checker is not None:
                def structure_record(value, args, kwargs):
                    name = kwargs['archetype_name']
                    outcomes[name]['structural'] = dict(passed=value[0], reason=value[1])
                stack.enter_context(patch.object(checker, 'check_structure', observe(checker.check_structure, structure_record)))
            for name, arch in self.engine.archetypes.items():
                item = outcomes[name]

                def detect_record(value, args, kwargs, item=item):
                    item['detect_calls'] += 1
                    item['native_signal'] = deepcopy(vars(value)) if value is not None else None

                def gate_record(value, args, kwargs, item=item):
                    item['gates'] = dict(passed=value[0], reason=value[1], penalty=value[2])

                def fusion_record(value, args, kwargs, item=item):
                    item['fusion_before_gate_penalty'] = value

                for method, record in (('detect', detect_record), ('_evaluate_gates', gate_record),
                                       ('compute_fusion_score', fusion_record)):
                    original = getattr(arch, method)
                    stack.enter_context(patch.object(arch, method, observe(original, record)))
            signals = self.engine.get_signals(bar=bar, bar_index=self.bar_index,
                                             prev_row=previous, lookback_df=lookback)
        selected = {signal.archetype_id for signal in signals}
        for name, item in outcomes.items():
            item['selected'] = name in selected
            item['last_signal_bar_after'] = self.engine.archetypes[name].last_signal_bar
        if checker is not None and checker.stats['errors']:
            self.blockers.add('structural_error_permissive_fallback')
        return dict(certified=False, bar_index=self.bar_index, feature_open_time=str(opened),
                    decision_time=str(decision_time), archetypes=outcomes,
                    signals=[deepcopy(vars(signal)) for signal in signals],
                    blockers=sorted(self.blockers),
                    scope='Native get_signals including structure/gates/fusion/cooldown/dedup; not runner entry or execution')

    def snapshot(self):
        unsupported = set()
        state = state_view(dict(bar_index=self.bar_index, last_open=self.last_open,
                     buffer=list(self.buffer), engine_stats=self.engine.stats,
                     archetypes=self.engine.archetypes, regime=self.engine.regime_service,
                     structural=self.engine.structural_checker), unsupported)
        if unsupported:
            self.blockers.add('unsupported_signal_state_types')
        return dict(state=state, blockers=sorted(self.blockers),
                    unsupported_state_types=sorted(unsupported))


class FeatureSignalProcessor(LiveFeatureProcessor):
    """Both retained engines advance together, including every warmup hour."""
    def __init__(self, timeframe='1h', config=ROOT/'configs/champion_paper.json'):
        super().__init__(timeframe)
        self.signal_engine = SignalEngine(config)
        self.contract_id = digest(dict(features=self.contract_id,
                                       signals=self.signal_engine.contract_id))

    def update(self, candle, observations):
        output = super().update(candle, observations)
        output['engine_signal'] = None
        if output['hourly_updated']:
            output['engine_signal'] = self.signal_engine.update(output['features'], candle['close_time'])
        self.blockers.update(self.signal_engine.blockers)
        return output

    def snapshot(self):
        signal_state = self.signal_engine.snapshot()
        self.blockers.update(self.signal_engine.blockers)
        return dict(super().snapshot(), engine_signal=signal_state)


def run_signal_replay(bars, observations=(), *, instrument, timeframe, emit_from=None,
                      checkpoint=None, config=ROOT/'configs/champion_paper.json'):
    observations = list(observations)
    for record in observations:
        if record.feature not in CHANNELS:
            raise ValueError(f'Undeclared feature input channel: {record.feature}')
        validate_channel(record.feature, record.value)
    processors = []

    def factory():
        processor = FeatureSignalProcessor(timeframe, config)
        processors.append(processor)
        return processor

    with deny_network():
        result = replay(bars, observations, factory, instrument=instrument, timeframe=timeframe,
                        emit_from=emit_from, checkpoint=checkpoint)
    processor = processors[0]
    result['clock_certified'] = result['certified']
    result['certified'] = result['full_pipeline_certified'] = False
    # Preserve feature lineage separately from the current clock observations.
    selected, position = {}, 0
    outputs = [row['output'] for row in result['rows']]
    outputs.append(dict(features_available_at=result['state']['features_at']))
    for output in outputs:
        at = output['features_available_at']
        if at is not None:
            while position < len(observations) and observations[position].visible_at() <= utc(at):
                record = observations[position]
                selected[record.feature] = record
                position += 1
        output['feature_observation_ids'] = {k: v.id for k, v in selected.items()}
        output['feature_observation_visible_at'] = {k: str(v.visible_at()) for k, v in selected.items()}
    result['state'].update({k: outputs[-1][k] for k in ('feature_observation_ids', 'feature_observation_visible_at')})
    for row in result['rows']:
        row['clock_certified'] = row.pop('certified')
        row['certified'] = False
    result.update(blockers=sorted(processor.blockers), contract_id=processor.contract_id,
                  source_manifest=dict(features=processor.manifest, signals=processor.signal_engine.manifest),
                  scope='Actual LFC plus engine signal generation; no runner adaptive threshold, allocation, positions or fills')
    return result
