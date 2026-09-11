"""Synthetic characterization of native runner behavior, never strategy results."""
from copy import deepcopy
import importlib
import os
from pathlib import Path
import socket
import unittest
from unittest.mock import patch

import pandas as pd

from scripts.research.replay_clock import digest


ROOT = Path(__file__).resolve().parents[2]
START = pd.Timestamp('2026-01-01T00:00Z')


def features(hour=0):
    return dict(timestamp=START + pd.Timedelta(hours=hour), open=100., high=101.,
                low=99., close=100., volume=10., atr_14=2., regime_label='neutral')


def signal(hour=0, fusion=.95, gates=True):
    from engine.portfolio.archetype_allocator import ArchetypeSignal
    return ArchetypeSignal('wick_trap', 'long', .95, 100., 95., 110., fusion,
                           'neutral', START + pd.Timedelta(hours=hour),
                           {'hard_gates_passed': gates,
                            'hard_gates_failed_reason': 'controlled failure' if not gates else ''})


class VirtualBookReplayTests(unittest.TestCase):
    def module(self):
        self.assertTrue((ROOT/'scripts/research/virtual_book_replay.py').exists(),
                        'Bounded actual runner fixture is missing')
        return importlib.import_module('scripts.research.virtual_book_replay')

    def fixture(self):
        return self.module().VirtualBookFixture(
            initial_cash=100000., commission_rate=.0004, slippage_bps=5.)

    def step(self, fixture, hour=0, signals=(), **changes):
        values = features(hour)
        values.update(changes)
        return fixture.step(values, signals, START + pd.Timedelta(hours=hour+1))

    def test_no_signal_advances_real_book_and_separates_availability_clock(self):
        fixture = self.fixture()
        row = self.step(fixture)
        self.assertEqual(fixture.runner.bar_index, 1)
        self.assertEqual(fixture.runner.equity_curve, [100000.])
        self.assertEqual(row['acted_signals'], [])
        self.assertEqual(row['source_hour_open'], START)
        self.assertEqual(row['available_at'], START + pd.Timedelta('1h'))
        self.assertFalse(row['certified'])
        self.assertIn('maker_shadow_calculation_and_persistence_excluded', row['blockers'])
        self.assertEqual(len(fixture.runner.engine.archetypes), 17)
        self.assertTrue(fixture.runner.bypass_threshold)

    def test_accepted_entry_keeps_native_fill_fees_sizing_and_series_quirks(self):
        fixture = self.fixture()
        incoming = signal()
        original = deepcopy(vars(incoming))
        row = self.step(fixture, signals=[incoming])
        pos = next(iter(fixture.runner.positions.values()))
        # Champion flat $4000 notional is capped at $1750 margin x 1.5 leverage.
        self.assertAlmostEqual(pos.position_size_usd, 2625.)
        self.assertAlmostEqual(pos.margin_used, 1750.)
        self.assertAlmostEqual(pos.entry_price, 100.05)
        self.assertAlmostEqual(pos.current_quantity, 26.23688155922039)
        self.assertAlmostEqual(fixture.runner.cash, 98247.6375)
        self.assertAlmostEqual(fixture.runner.equity_curve[-1], 99996.32565592204)
        self.assertEqual(pos.entry_time, START)
        self.assertEqual(pos.entry_metadata['entry_wick_low'], pos.entry_price)
        self.assertEqual(pos.entry_metadata['entry_volume'], 0.)
        self.assertEqual(row['acted_signals'][0]['action'], 'ENTRY')
        self.assertEqual(digest(vars(incoming)), digest(original))

    def test_later_stop_uses_native_exit_fee_and_source_accounting(self):
        fixture = self.fixture()
        self.step(fixture, signals=[signal()])
        self.step(fixture, 1, low=94., close=96.)
        self.assertEqual(len(fixture.runner.positions), 0)
        trade = fixture.runner.trades[0]
        self.assertEqual(trade.exit_reason, 'stop_loss')
        self.assertEqual(trade.duration_hours, 1.)
        self.assertAlmostEqual(trade.exit_price, 94.9525)
        # Native trade PnL subtracts EXIT commission; entry costs reside in cash.
        self.assertAlmostEqual(trade.pnl, -134.7390067466266)
        self.assertAlmostEqual(fixture.runner.cash, 99862.89849325336)

    def test_stop_and_target_touched_in_same_bar_choose_stop(self):
        fixture = self.fixture()
        self.step(fixture, signals=[signal()])
        self.step(fixture, 1, low=94., high=115., close=112.)
        self.assertEqual(len(fixture.runner.trades), 1)
        self.assertEqual(fixture.runner.trades[0].exit_reason, 'stop_loss')
        self.assertAlmostEqual(fixture.runner.trades[0].exit_price, 94.9525)

    def test_exits_precede_new_entries_and_new_entry_avoids_same_bar_stop(self):
        fixture = self.fixture()
        self.step(fixture, signals=[signal()])
        self.step(fixture, 1, signals=[signal(1)], low=94., close=100.)
        self.assertEqual(len(fixture.runner.trades), 1)
        self.assertEqual(len(fixture.runner.positions), 1)
        self.assertEqual(next(iter(fixture.runner.positions.values())).bars_held, 0)
        events = [r['payload']['action'] for r in fixture.records if r['kind'] == 'signal_log']
        self.assertEqual(events, ['ENTRY', 'EXIT', 'ENTRY'])

    def test_below_threshold_failed_gates_create_phantom_not_real_entry(self):
        fixture = self.fixture()
        row = self.step(fixture, signals=[signal(fusion=.001, gates=False)])
        self.assertEqual(row['acted_signals'], [])
        self.assertEqual(fixture.runner.cash, 100000.)
        self.assertEqual(len(fixture.runner.positions), 0)
        self.assertEqual(len(fixture.runner.phantom_positions), 1)
        self.assertEqual(fixture.runner.last_bar_signals[0]['rejection_stage'], 'bypass_gate_block')
        self.step(fixture, 1, low=94., close=96.)
        self.assertEqual(fixture.runner.phantom_trades[0].exit_reason, 'stop_loss')
        self.assertEqual(fixture.runner.cash, 100000.)

    def test_fresh_whole_prehistory_repeats_outputs_and_retained_state(self):
        first, second = self.fixture(), self.fixture()
        for hour in range(4):
            signals = [signal(hour)] if hour in (0, 2) else []
            changes = {'low': 94., 'close': 96.} if hour in (1, 3) else {}
            self.assertEqual(digest(self.step(first, hour, signals, **changes)),
                             digest(self.step(second, hour, signals, **changes)))
        self.assertEqual(digest(first.snapshot()), digest(second.snapshot()))
        self.assertEqual(first.snapshot()['unsupported_state_types'], [])

    def test_duplicate_gap_and_early_availability_do_not_mutate_state(self):
        fixture = self.fixture()
        self.step(fixture)
        before = digest(fixture.snapshot())
        for values, at in ((features(), START+pd.Timedelta('1h')),
                           (features(2), START+pd.Timedelta('3h')),
                           (features(1), START+pd.Timedelta('90min'))):
            with self.assertRaises(ValueError):
                fixture.step(values, [], at)
            self.assertEqual(before, digest(fixture.snapshot()))

    def test_swallowed_write_is_detected_and_failed_fixture_cannot_continue(self):
        fixture = self.fixture()
        target = ROOT/'results/live_signals/virtual_book_forbidden_write.txt'
        self.assertFalse(target.exists())
        def attempted_write(*args, **kwargs):
            target.write_text('must never reach disk')
        with patch.object(fixture.maker, 'record_entry', attempted_write):
            with self.assertRaisesRegex(RuntimeError, 'side effect'):
                self.step(fixture, signals=[signal()])
        self.assertFalse(target.exists())
        with self.assertRaisesRegex(RuntimeError, 'failed'):
            self.step(fixture, 1)

    def test_swallowed_network_is_detected_during_native_processing(self):
        fixture = self.fixture()
        original = fixture.runner._compute_adaptive_threshold
        def attempt(features):
            try:
                socket.create_connection(('127.0.0.1', 9))
            except Exception:
                pass
            return original(features)
        with patch.object(fixture.runner, '_compute_adaptive_threshold', attempt):
            with self.assertRaisesRegex(RuntimeError, 'side effect|Network'):
                self.step(fixture)
        self.assertTrue(any(r['kind'] == 'denied_side_effect' for r in fixture.records))

    def test_constructor_write_attempt_cannot_escape_by_being_swallowed(self):
        module = self.module()
        runner_module = importlib.import_module('bin.live.v11_shadow_runner')
        original = runner_module.V11ShadowRunner._init_exit_logic
        target = ROOT/'results/live_signals/virtual_book_forbidden_constructor.txt'
        self.assertFalse(target.exists())
        def attempt(runner):
            try:
                target.write_text('must never reach disk')
            except Exception:
                pass
            original(runner)
        with patch.object(runner_module.V11ShadowRunner, '_init_exit_logic', attempt):
            with self.assertRaisesRegex(RuntimeError, 'side effect'):
                module.VirtualBookFixture(initial_cash=100000., commission_rate=.0004,
                                          slippage_bps=5.)
        self.assertFalse(target.exists())

    def test_audit_guard_catches_prebound_write_open_and_unexpected_mkdir(self):
        module = self.module()
        prebound_open = os.open
        target = ROOT/'results/live_signals/virtual_book_forbidden_audit'
        self.assertFalse(target.exists())
        for operation in (lambda: prebound_open(target, os.O_CREAT | os.O_WRONLY, 0o600),
                          lambda: target.mkdir()):
            records = []
            with self.assertRaisesRegex(RuntimeError, 'side effect'):
                with module.side_effect_guard(records):
                    try:
                        operation()
                    except Exception:
                        pass
            self.assertFalse(target.exists())
            self.assertEqual(records[-1]['kind'], 'denied_side_effect')

    def test_contract_binds_economic_parameters_and_model_artifacts(self):
        first = self.fixture()
        second = self.module().VirtualBookFixture(initial_cash=100001., commission_rate=.0004,
                                                   slippage_bps=5.)
        self.assertNotEqual(first.contract_id, second.contract_id)
        files = first.manifest['files']
        for path in ('configs/champion_paper.json',
                     'models/logistic_regime_v4_no_funding_stratified.pkl',
                     'models/confidence_calibrator_v1.pkl',
                     'configs/optimized/cmi_weights_optimized.json'):
            self.assertIn(str(ROOT/path), files)
        self.assertIn('model_class', first.manifest)
        self.assertIn('calibrator_class', first.manifest)


if __name__ == '__main__':
    unittest.main()
