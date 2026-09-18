"""Native source composition witnesses; synthetic inputs, no strategy claim."""
from copy import deepcopy
import importlib
from pathlib import Path
import subprocess
import sys
import unittest
from unittest.mock import patch

import pandas as pd

from scripts.research.engine_signal_replay import SignalEngine
from scripts.research.replay_clock import Observation, digest, replay
from scripts.research.virtual_book_replay import VirtualBookFixture, side_effect_guard

ROOT = Path(__file__).resolve().parents[2]
START = pd.Timestamp('2026-01-01T00:00Z')
ECONOMICS = dict(initial_cash=100000., commission_rate=.0004, slippage_bps=5.)


def wick_features(hour=0):
    return dict(timestamp=START+pd.Timedelta(hours=hour),
                open=99., high=101., low=90., close=100., volume=10.,
                wick_lower_ratio=.8, wick_upper_ratio=.05, volume_zscore=1.,
                liquidity_score=1., wyckoff_bullish_score=1.,
                tf4h_wyckoff_bullish_score=1., tf1d_wyckoff_bullish_score=1.,
                rsi_14=50., adx=25., adx_14=25., atr_14=2., atr_20=2.,
                smc_score=1., tf1h_bos_bullish=1., tf1h_bos_bearish=0.,
                tf1h_bos_detected=1., fusion_smc=1., prev_close=100.,
                funding_Z=0., alt_basket_ret_4h=0., stables_rot_rising=0.)


def candles(count=120, frequency='1min'):
    return pd.DataFrame(dict(open=100., high=101., low=99., close=100., volume=10.),
                        index=pd.date_range(START, periods=count, freq=frequency))


class NativePipelineTests(unittest.TestCase):
    def module(self):
        self.assertTrue((ROOT/'scripts/research/native_pipeline_replay.py').exists(),
                        'One-engine native pipeline adapter is missing')
        return importlib.import_module('scripts.research.native_pipeline_replay')

    def run_case(self, bars, observations=(), **kwargs):
        params = dict(ECONOMICS, instrument='BTC-fixture', timeframe='1min')
        params.update(kwargs)
        return self.module().run_pipeline_replay(bars, observations, **params)

    def test_20_hour_native_book_matches_standalone_signals_and_unobserved_book(self):
        module = self.module()
        book = module.NativeSignalBook(**ECONOMICS)
        raw = VirtualBookFixture(**ECONOMICS)
        with side_effect_guard([]):
            standalone = SignalEngine()
        for hour in range(20):
            values = wick_features(hour)
            opened = values['timestamp']
            at = opened+pd.Timedelta('1h')
            native = book.step(values, at)
            with side_effect_guard([]):
                reference = standalone.update(values, at)
            with side_effect_guard(raw.records):
                unobserved = raw.runner.process_bar(pd.Series(values, name=opened), opened)
            raw.last_open = opened
            self.assertEqual(digest(native['engine_signal']['signals']), digest(reference['signals']))
            self.assertEqual(digest(native['engine_signal']['archetypes']), digest(reference['archetypes']))
            self.assertEqual(digest(native['acted_signals']), digest(unobserved))
            self.assertEqual(digest(book.snapshot()['state']), digest(raw.snapshot()['state']))
            self.assertEqual(len(native['engine_signal']['archetypes']), 17)
            self.assertTrue(all(a['detect_calls'] == 1 for a in native['engine_signal']['archetypes'].values()))
        self.assertEqual(book.runner.engine.stats['total_bars'], 20)
        self.assertEqual(book.runner.engine.archetypes['wick_trap'].last_signal_bar, 19)
        self.assertEqual(len(book.runner.engine.regime_service.probabilistic_detector.regime_history), 20)
        self.assertEqual(len(book.runner.trades), 3)
        self.assertEqual(book.snapshot()['unsupported_state_types'], [])

    def test_observer_returns_same_native_objects_and_copies_before_runner_mutation(self):
        module = self.module()
        book = module.NativeSignalBook(**ECONOMICS)
        values = wick_features()
        observed_lists = []
        original = book.runner.engine.get_signals
        def native(**kwargs):
            result = original(**kwargs)
            observed_lists.append(result)
            return result
        with side_effect_guard([]):
            selected, diagnostic = module.observe_engine_signals(book.runner.engine, native,
                bar=pd.Series(values, name=START), bar_index=1, prev_row=None, lookback_df=None)
        self.assertEqual(len(observed_lists), 1)
        self.assertIs(selected, observed_lists[0])
        self.assertIs(selected[0], observed_lists[0][0])
        selected[0].metadata['post_selection_mutation'] = True
        self.assertNotIn('post_selection_mutation', diagnostic['signals'][0]['metadata'])

        fresh = module.NativeSignalBook(**ECONOMICS)
        output = fresh.step(values, START+pd.Timedelta('1h'))
        before = output['engine_signal']['signals'][0]
        after = next(iter(fresh.runner.positions.values()))
        self.assertEqual(before['archetype_id'], 'liquidity_sweep')
        self.assertNotIn('sizing_boosts', before['metadata'])
        self.assertIn('sizing_boosts', after.entry_metadata)
        self.assertNotEqual(before['fusion_score'], after.fusion_score)

    def test_native_order_exits_then_maker_then_threshold_then_detection_then_entry(self):
        book = self.module().NativeSignalBook(**ECONOMICS)
        book.step(wick_features(), START+pd.Timedelta('1h'))
        order = []
        def observe(label, original):
            def call(*args, **kwargs):
                order.append(label)
                return original(*args, **kwargs)
            return call
        with patch.object(book.runner, '_check_all_exits', observe('exits', book.runner._check_all_exits)), \
             patch.object(book.maker, 'on_bar', observe('maker', book.maker.on_bar)), \
             patch.object(book.runner, '_check_phantom_exits', observe('phantoms', book.runner._check_phantom_exits)), \
             patch.object(book.runner, '_compute_adaptive_threshold', observe('threshold', book.runner._compute_adaptive_threshold)), \
             patch.object(book.runner.engine, 'get_signals', observe('signals', book.runner.engine.get_signals)), \
             patch.object(book.runner, '_open_position', observe('entry', book.runner._open_position)):
            row = book.step(wick_features(1), START+pd.Timedelta('2h'))
        self.assertEqual(order, ['exits', 'maker', 'phantoms', 'threshold', 'signals', 'entry'])
        self.assertEqual(row['trade_count'], 1)
        self.assertEqual(row['position_count'], 1)
        self.assertEqual(book.runner.trades[0].exit_reason, 'stop_loss')
        self.assertEqual(book.runner.trades[0].timestamp_entry, START)
        self.assertEqual(row['source_hour_open'], START+pd.Timedelta('1h'))
        self.assertEqual(row['available_at'], START+pd.Timedelta('2h'))

    def test_native_soft_gate_failure_reaches_phantom_tracking_under_bypass(self):
        book = self.module().NativeSignalBook(**ECONOMICS)
        values = dict(wick_features(), liquidity_score=.3)
        row = book.step(values, START+pd.Timedelta('1h'))
        selected = row['engine_signal']['signals'][0]
        self.assertEqual(selected['archetype_id'], 'liquidity_sweep')
        self.assertFalse(selected['metadata']['hard_gates_passed'])
        self.assertEqual(row['last_bar_signals'][0]['rejection_stage'], 'bypass_gate_block')
        self.assertEqual(row['position_count'], 0)
        self.assertEqual(row['phantom_position_count'], 1)
        self.assertEqual(book.runner.cash, 100000.)
        later = dict(wick_features(1), liquidity_score=.3, low=80.)
        book.step(later, START+pd.Timedelta('2h'))
        self.assertEqual(book.runner.phantom_trades[0].exit_reason, 'stop_loss')

    def test_actual_detector_preserves_nonexecutable_gap_fill_and_source_open_label(self):
        book = self.module().NativeSignalBook(**ECONOMICS)
        entry = book.step(wick_features(), START+pd.Timedelta('1h'))
        pos = next(iter(book.runner.positions.values()))
        self.assertEqual(pos.archetype, 'liquidity_sweep')
        self.assertEqual(pos.stop_loss, 94.)
        self.assertAlmostEqual(pos.entry_price, 100.05)
        self.assertEqual(pos.entry_time, START)
        self.assertEqual(entry['available_at'], START+pd.Timedelta('1h'))
        # The completed source bar's low 90 does not stop its later-created
        # entry. On the next bar the source fills the stop above the entire bar.
        self.assertEqual(entry['trade_count'], 0)
        later = dict(wick_features(1), open=80., high=82., low=79., close=81.)
        exit_row = book.step(later, START+pd.Timedelta('2h'))
        trade = book.runner.trades[0]
        self.assertEqual(trade.exit_reason, 'stop_loss')
        self.assertAlmostEqual(trade.exit_price, 93.953)
        self.assertGreater(trade.exit_price, later['high'])
        self.assertFalse(exit_row['certified'])
        self.assertIn('source_fill_not_executable', exit_row['blockers'])
        self.assertIn('backdated_source_labels', exit_row['blockers'])

    def test_120_minutes_hold_book_until_completed_hour_and_preserve_lineage(self):
        observations = [Observation('a', 'stables_rot_rising', 0., 'BTC-fixture',
                            'fixture', 'indicator', 'v1', START, START),
                        Observation('b', 'stables_rot_rising', 1., 'BTC-fixture',
                            'fixture', 'indicator', 'v1', START+pd.Timedelta('61min'),
                            START+pd.Timedelta('61min'))]
        output = self.run_case(candles(), observations)
        rows = output['rows']
        for minute, row in enumerate(rows):
            if minute not in (59, 119):
                self.assertIsNone(row['output']['native_book'])
                self.assertIsNone(row['output']['engine_signal'])
        self.assertEqual(rows[59]['output']['native_book']['bar_index'], 1)
        self.assertEqual(rows[119]['output']['native_book']['bar_index'], 2)
        self.assertEqual(rows[60]['observation_ids']['stables_rot_rising'], 'b')
        self.assertEqual(rows[60]['output']['feature_observation_ids']['stables_rot_rising'], 'a')
        self.assertEqual(output['state']['hourly_updates'], 2)
        self.assertEqual(output['state']['native_book']['state']['bar_index'], 2)
        self.assertFalse(output['certified'])
        self.assertTrue(output['clock_certified'])
        for blocker in ('source_fill_not_executable', 'backdated_source_labels',
                        'host_delay_not_replayed', 'maker_shadow_calculation_and_persistence_excluded'):
            self.assertIn(blocker, output['blockers'])
        self.assertNotIn('synthetic_controlled_signals_not_detector_replay', output['blockers'])
        self.assertNotIn('full_decision_book_not_replayed', output['blockers'])

    def test_minute_prefix_and_fresh_restart_keep_rows_and_complete_state(self):
        full = self.run_case(candles(), emit_from=START+pd.Timedelta('1h'))
        prefix = self.run_case(candles(61), emit_from=START+pd.Timedelta('1h'))
        self.assertEqual(digest(prefix['rows']), digest(full['rows'][:1]))
        resumed = self.run_case(candles(), emit_from=START+pd.Timedelta('1h'),
                                checkpoint=prefix['checkpoint'])
        self.assertEqual(digest(resumed['rows']), digest(full['rows'][1:]))
        self.assertEqual(digest(resumed['state']), digest(full['state']))

    def test_hidden_warmup_position_is_rebuilt_from_prehistory_before_resumed_exit(self):
        module = self.module()
        created = []
        def factory():
            processor = module.NativePipelineProcessor('1h', **ECONOMICS)
            original = processor.fc.update
            # Diagnostic seam fixes only the feature vector; the LFC still
            # advances, and actual detectors/runner retain their own state.
            def wick_vector(candle):
                original(candle)
                hour = int((candle['timestamp']-START)/pd.Timedelta('1h'))
                return pd.Series(wick_features(hour), name=candle['timestamp'])
            processor.fc.update = wick_vector
            created.append(processor)
            return processor
        kwargs = dict(instrument='BTC-fixture', timeframe='1h', emit_from=START+pd.Timedelta('1h'))
        with side_effect_guard([]):
            full = replay(candles(3, '1h'), [], factory, **kwargs)
            prefix = replay(candles(1, '1h'), [], factory, **kwargs)
            self.assertEqual(len(created[-1].book.runner.positions), 1)
            resumed = replay(candles(3, '1h'), [], factory, checkpoint=prefix['checkpoint'], **kwargs)
        self.assertEqual(digest(resumed['rows']), digest(full['rows']))
        self.assertEqual(digest(resumed['state']), digest(full['state']))
        self.assertEqual(resumed['rows'][0]['output']['native_book']['trade_count'], 1)

    def test_changed_explicit_cash_commission_or_slippage_rejects_checkpoint(self):
        prefix = self.run_case(candles(60))
        for field, value in (('initial_cash', 100001.), ('commission_rate', .0005), ('slippage_bps', 6.)):
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, 'checkpoint'):
                self.run_case(candles(), checkpoint=prefix['checkpoint'], **{field: value})

    def test_native_detector_swallowed_write_fails_closed_and_locks_book(self):
        book = self.module().NativeSignalBook(**ECONOMICS)
        target = ROOT/'results/live_signals/native_pipeline_forbidden.txt'
        self.assertFalse(target.exists())
        original = book.runner.engine.archetypes['wick_trap'].detect
        def detector(*args, **kwargs):
            try:
                target.write_text('must never reach disk')
            except Exception:
                pass
            return original(*args, **kwargs)
        with patch.object(book.runner.engine.archetypes['wick_trap'], 'detect', detector):
            with self.assertRaisesRegex(RuntimeError, 'side effect'):
                book.step(wick_features(), START+pd.Timedelta('1h'))
        self.assertFalse(target.exists())
        with self.assertRaisesRegex(RuntimeError, 'failed'):
            book.step(wick_features(1), START+pd.Timedelta('2h'))

    def test_composition_owns_only_runner_engine_and_never_uses_controlled_step(self):
        module = self.module()
        with patch.object(SignalEngine, '__init__', side_effect=AssertionError('second engine prohibited')), \
             patch.object(VirtualBookFixture, 'step', side_effect=AssertionError('controlled step prohibited')):
            output = self.run_case(candles(60))
        self.assertEqual(output['state']['native_book']['state']['engine']['attributes']['stats']['total_bars'], 1)

    def test_fresh_process_imports_and_runs_without_transport_probe_sockets(self):
        self.module()
        script = '''
from scripts.research.native_pipeline_replay import NativePipelineProcessor
from scripts.research.virtual_book_replay import side_effect_guard
import pandas as pd
import socket
ipv6_before = socket.has_ipv6
p = NativePipelineProcessor('1h', initial_cash=100000., commission_rate=.0004, slippage_bps=5.)
assert socket.has_ipv6 == ipv6_before
t = pd.Timestamp('2026-01-01T00:00Z')
out = p.update(dict(timestamp=t, close_time=t+pd.Timedelta('1h'),
                    open=100., high=101., low=99., close=100., volume=10.), {})
assert out['native_book']['bar_index'] == 1
assert not out['certified']
try:
    with side_effect_guard(p.feature_records):
        try:
            socket.socket(socket.AF_INET6)
        except RuntimeError:
            pass
except RuntimeError as error:
    assert 'side effect' in str(error)
else:
    raise AssertionError('deliberate socket attempt was not denied')
'''
        completed = subprocess.run([sys.executable, '-B', '-c', script], cwd=ROOT,
                                   capture_output=True, text=True, timeout=30)
        self.assertEqual(completed.returncode, 0, completed.stderr)


if __name__ == '__main__':
    unittest.main()
