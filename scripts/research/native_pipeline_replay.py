"""Actual LFC -> one runner-owned signal engine -> native virtual paper book.

Research only, all strategy certificates false. Native source uses completed
hour-close features/signal prices yet labels entries with the hour OPEN. Its
virtual stop fills can lie outside the next bar's range. Neither that source
behavior nor the host's later fetch/processing delay is an executable fill model.
No next-open correction is silently applied. Maker measurement, persisted logs,
funding costs and daily host state remain excluded as in VirtualBookFixture.

Use a dedicated single-threaded process. Import, construction, feature updates
and native process_bar are protected by the inherited scoped Python guard, not
an OS sandbox. Restart reconstructs the entire original prehistory, including
warmup positions. No serialized partial runner state is loaded or saved.
"""
from copy import deepcopy
import hashlib
from pathlib import Path
import socket
from unittest.mock import patch

import pandas as pd

from scripts.research.engine_signal_replay import observe_engine_signals
from scripts.research.live_feature_replay import LiveFeatureProcessor, CHANNELS, validate_channel
from scripts.research.replay_clock import digest, replay, utc
from scripts.research.virtual_book_replay import VirtualBookFixture, side_effect_guard

FILL_POLICY = ('Native source virtual fills from completed-hour prices; source hour-open labels; '
               'not executable market results; host delay is not replayed')


class NativeSignalBook(VirtualBookFixture):
    """Reuse the guarded native constructor; never invoke the controlled step."""
    def _manifest(self, initial_cash, commission_rate, slippage_bps):
        manifest = super()._manifest(initial_cash, commission_rate, slippage_bps)
        self.blockers.discard('synthetic_controlled_signals_not_detector_replay')
        self.blockers.update({'source_fill_not_executable', 'backdated_source_labels',
                              'host_delay_not_replayed'})
        path = Path(__file__).resolve()
        manifest['files'][str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
        manifest.update(signal_source='native_runner_owned_get_signals_once_per_bar',
                        signal_observer='same_native_objects; diagnostics_copied_before_runner_mutation',
                        fill_policy=FILL_POLICY, blockers=sorted(self.blockers))
        return manifest

    def step(self, features, decision_time):
        if self.failed:
            raise RuntimeError('This fixture failed; use a fresh whole-prehistory replay')
        opened = utc(features['timestamp'])
        available = utc(decision_time)
        if opened != opened.floor('1h') or available != opened + pd.Timedelta('1h'):
            raise ValueError('Requires completed hourly features and exact hourly availability')
        if self.last_open is not None and opened != self.last_open + pd.Timedelta('1h'):
            raise ValueError('Duplicate/out-of-order/gapped native-book history')
        original = self.runner.engine.get_signals
        calls = 0
        diagnostic = None

        def observed_native(**native_kwargs):
            nonlocal calls, diagnostic
            calls += 1
            if calls != 1:
                raise RuntimeError('Multiple native signal evaluations in one bar are prohibited')
            signals, diagnostic = observe_engine_signals(self.runner.engine, original, **native_kwargs)
            # This is the actual list with actual object identities. Thresholds,
            # allocation and sizing continue in native process_bar after return.
            return signals

        start = len(self.records)
        try:
            with side_effect_guard(self.records), patch.object(
                    self.runner.engine, 'get_signals', observed_native):
                bar = pd.Series(deepcopy(features), name=opened)
                acted = self.runner.process_bar(bar, opened)
                if calls != 1 or diagnostic is None:
                    raise RuntimeError('Native process_bar must evaluate its engine exactly once')
        except BaseException:
            self.failed = True
            raise
        self.last_open = opened
        if diagnostic['structural_errors']:
            self.blockers.add('structural_error_permissive_fallback')
        diagnostic.update(certified=False, bar_index=self.runner.bar_index,
                          feature_open_time=str(opened), decision_time=str(available),
                          blockers=sorted(self.blockers),
                          scope='One native engine call within actual process_bar; selected signals copied before runner mutation')
        return dict(certified=False, contract_id=self.contract_id, source_hour_open=opened,
                    available_at=available, bar_index=self.runner.bar_index,
                    engine_signal=diagnostic, acted_signals=deepcopy(acted),
                    last_bar_signals=deepcopy(self.runner.last_bar_signals),
                    cash=self.runner.cash, equity=self.runner.equity_curve[-1],
                    position_count=len(self.runner.positions), trade_count=len(self.runner.trades),
                    phantom_position_count=len(self.runner.phantom_positions),
                    phantom_trade_count=len(self.runner.phantom_trades),
                    records=deepcopy(self.records[start:]), blockers=sorted(self.blockers),
                    fill_policy=FILL_POLICY)


class NativePipelineProcessor(LiveFeatureProcessor):
    """Adapter advances base bars; actual LFC and book advance completed hours."""
    def __init__(self, timeframe='1h', *, initial_cash, commission_rate, slippage_bps):
        self.feature_records = []
        self.failed = False
        # urllib3 probes local IPv6 binding during requests import. Disable only
        # that optional offline transport capability probe; never permit sockets
        # or import LFC outside the sticky guard. The process flag is restored.
        with side_effect_guard(self.feature_records), patch.object(socket, 'has_ipv6', False):
            super().__init__(timeframe)
        self.book = NativeSignalBook(initial_cash=initial_cash,
                                    commission_rate=commission_rate, slippage_bps=slippage_bps)
        self.blockers.discard('full_decision_book_not_replayed')
        self.blockers.update(self.book.blockers)
        self.manifest = dict(features=self.manifest, native_book=self.book.manifest,
                             fill_policy=FILL_POLICY,
                             offline_ipv6_transport_probe_disabled=True,
                             clock_policy='base-hour or base-minute; book only when hourly_updated')
        self.contract_id = digest(dict(feature_contract=self.contract_id,
                                       native_book_contract=self.book.contract_id,
                                       offline_ipv6_transport_probe_disabled=True,
                                       clock_policy=self.manifest['clock_policy']))

    def update(self, candle, observations):
        if self.failed:
            raise RuntimeError('This pipeline failed; use a fresh whole-prehistory replay')
        try:
            with side_effect_guard(self.feature_records):
                output = super().update(candle, observations)
                output['engine_signal'] = output['native_book'] = None
                if output['hourly_updated']:
                    book_output = self.book.step(output['features'], candle['close_time'])
                    output['native_book'] = book_output
                    output['engine_signal'] = deepcopy(book_output['engine_signal'])
        except BaseException:
            self.failed = self.book.failed = True
            raise
        self.blockers.update(self.book.blockers)
        output.update(blockers=sorted(self.blockers), fill_policy=FILL_POLICY,
                      certified=False, full_pipeline_certified=False)
        return output

    def snapshot(self):
        with side_effect_guard(self.feature_records):
            book_state = self.book.snapshot()
            self.blockers.update(self.book.blockers)
            state = super().snapshot()
        state.update(native_book=book_state, failed=self.failed, blockers=sorted(self.blockers))
        return state


def run_pipeline_replay(bars, observations=(), *, instrument, timeframe,
                        initial_cash, commission_rate, slippage_bps,
                        emit_from=None, checkpoint=None):
    """Clock provenance + full-prehistory state replay, never a trading certificate."""
    observations = list(observations)
    for record in observations:
        if record.feature not in CHANNELS:
            raise ValueError(f'Undeclared feature input channel: {record.feature}')
        validate_channel(record.feature, record.value)
    processors = []

    def factory():
        processor = NativePipelineProcessor(timeframe, initial_cash=initial_cash,
                            commission_rate=commission_rate, slippage_bps=slippage_bps)
        processors.append(processor)
        return processor

    with side_effect_guard([]):
        result = replay(bars, observations, factory, instrument=instrument,
                        timeframe=timeframe, emit_from=emit_from, checkpoint=checkpoint)
    processor = processors[0]
    result['clock_certified'] = result['certified']
    result['certified'] = result['full_pipeline_certified'] = False
    # Held feature vectors retain the observation lineage from their actual
    # hourly calculation, even when later minute observations are now visible.
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
                  source_manifest=processor.manifest, fill_policy=FILL_POLICY,
                  scope='Actual LFC plus one native runner-owned signal engine and native virtual paper book; source virtual fills are not executable market results')
    return result
