import importlib.util
from copy import deepcopy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import json

import numpy as np
import pandas as pd


SCRIPT = Path(__file__).resolve().parents[2] / 'scripts/research/minute_sweep_validation.py'


def candles(n=200):
    return pd.DataFrame({'open': np.full(n, 112.), 'high': np.full(n, 115.),
                         'low': 110. + np.arange(n)*.001, 'close': np.full(n, 112.)},
                        index=pd.date_range('2021-01-01', periods=n, freq='min', tz='UTC'))


def setup():
    b = candles()
    b.loc[b.index[[20, 60]], 'low'] = 100.
    b.loc[b.index[90], ['low', 'close']] = [99., 101.]
    return b


def event(i=2):
    return dict(pivot_idx=0, sweep_idx=i, reclaim_idx=i, level=100.,
                sweep_low=99., touches=2)


class MinuteValidationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not SCRIPT.exists():
            raise AssertionError('Minute validation implementation is missing')
        spec = importlib.util.spec_from_file_location('minute_validation', SCRIPT)
        cls.m = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.m)

    def test_confirmed_two_touch_level_reclaims_at_expected_minute(self):
        result = self.m.detect_events(setup())
        self.assertEqual([x['reclaim_idx'] for x in result], [90])
        self.assertEqual(result[0]['pivot_idx'], 60)
        self.assertEqual(result[0]['touches'], 2)
        self.assertEqual(result[0]['sweep_low'], 99.)

    def test_unconfirmed_pivot_cannot_create_setup(self):
        self.assertEqual(self.m.detect_events(setup().iloc[:74]), [])

    def test_future_cannot_rewrite_emitted_events(self):
        b = setup()
        expected = self.m.detect_events(b.iloc[:100])
        b.loc[b.index[120:], ['open', 'high', 'low', 'close']] = [200., 210., 190., 200.]
        self.assertEqual([e for e in self.m.detect_events(b) if e['reclaim_idx'] < 100], expected)
        self.assertEqual(len(expected), 1)

    def test_invalid_input_is_rejected(self):
        b = setup()
        variants = [b.drop(b.index[10]), pd.concat([b, b.iloc[-1:]]), b.tz_localize(None)]
        nan = b.copy(); nan.iloc[10, 0] = np.nan; variants.append(nan)
        bad = b.copy(); bad.iloc[10, bad.columns.get_loc('low')] = 999; variants.append(bad)
        for frame in variants:
            with self.subTest(shape=frame.shape), self.assertRaises(ValueError):
                self.m.detect_events(frame)

    def small_bars(self, n=12):
        b = candles(n)
        b[['open', 'high', 'low', 'close']] = [100., 101., 99.5, 100.]
        return b

    def simulate(self, b, events=None, **kw):
        return self.m.simulate_events(b, events or [event()], notional=1000,
                                     stop_buffer=0, hold_minutes=4, **kw)

    def test_entry_minute_stop_is_checked_and_cost_charged_once(self):
        b = self.small_bars(); b.loc[b.index[3], 'low'] = 98.
        r = self.simulate(b)
        t = r['trades'][0]
        self.assertEqual(t['exit_idx'], 3)
        self.assertEqual(t['reason'], 'stop')
        self.assertAlmostEqual(t['pnl'], -11.2)
        self.assertAlmostEqual(t['initial_risk'], 10.)

    def test_gap_stop_fills_at_adverse_open(self):
        b = self.small_bars(); b.loc[b.index[4], ['open','high','low','close']] = [97., 98., 96., 97.]
        t = self.simulate(b)['trades'][0]
        self.assertEqual(t['exit_price'], 97.)
        self.assertAlmostEqual(t['pnl'], -31.2)

    def test_incomplete_horizon_remains_open(self):
        r = self.simulate(self.small_bars(5))
        self.assertEqual(r['trades'], [])
        self.assertEqual(len(r['open_positions']), 1)
        self.assertAlmostEqual(r['open_positions'][0]['marked_pnl'], -.6)

    def test_four_hour_policy_locks_after_early_stop(self):
        b = self.small_bars(); b.loc[b.index[3], 'low'] = 98.
        r = self.simulate(b, [event(2), event(4)])
        self.assertEqual(len(r['trades']), 1)
        self.assertEqual(r['skipped_busy'], 1)

    def test_close_entry_does_not_use_pre_entry_low(self):
        b = self.small_bars(); b.loc[b.index[2], 'low'] = 98.
        t = self.simulate(b, entry_mode='close')['trades'][0]
        self.assertEqual(t['reason'], 'time')
        self.assertAlmostEqual(t['pnl'], -1.2)

    def test_unsorted_events_are_rejected(self):
        with self.assertRaises(ValueError):
            self.simulate(self.small_bars(), [event(5), event(2)])

    def test_next_open_time_exit_is_at_deadline_open_before_later_wick(self):
        b = self.small_bars()
        b.loc[b.index[7], ['open', 'high', 'low', 'close']] = [100., 108., 98., 107.]
        r = self.simulate(b, [event(2), event(6)])
        first = r['trades'][0]
        self.assertEqual(first['reason'], 'time')
        self.assertEqual(first['exit_price'], 100.)
        self.assertEqual(first['exit_phase'], 'open')
        self.assertEqual(first['exit_idx'], 7)
        self.assertEqual(r['trades'][1]['entry_idx'], 7)

    def test_zero_delay_preserves_existing_numeric_fills_and_adds_stable_ledger(self):
        b = self.small_bars()
        defaults = self.simulate(b)
        explicit = self.simulate(b, decision_delay_seconds=0)
        self.assertEqual(defaults, explicit)
        trade = explicit['trades'][0]
        self.assertEqual((trade['entry_idx'], trade['exit_idx']), (3, 7))
        self.assertEqual((trade['entry_price'], trade['exit_price'], trade['stop_price']), (100., 100., 99.))
        self.assertAlmostEqual(trade['pnl'], -1.2)
        self.assertAlmostEqual(trade['fees'], 1.2)
        self.assertAlmostEqual(explicit['summary']['minute_close_mtm_drawdown'], -1.2)
        self.assertEqual(explicit['event_ledger'][0]['status'], 'completed')
        self.assertEqual(explicit['event_ledger'][0]['event_id'], trade['event_id'])
        self.assertEqual(explicit['event_ledger'][0]['sampled_delay_seconds'], 0)

    def test_ninety_second_delay_rounds_up_across_hour_boundary(self):
        b = self.small_bars()
        b.index = pd.date_range('2021-01-01T00:57:00Z', periods=len(b), freq='min')
        events = [event(0)]
        unchanged = deepcopy(events)
        output = self.simulate(b, events, decision_delay_seconds=90)
        trade = output['trades'][0]
        timing = output['event_ledger'][0]
        self.assertEqual(trade['entry_idx'], 3)
        self.assertEqual(trade['entry_time'], '2021-01-01 01:00:00+00:00')
        self.assertEqual(trade['exit_idx'], 7)
        self.assertEqual(timing['signal_available_at'], '2021-01-01 00:58:00+00:00')
        self.assertEqual(timing['order_ready_at'], '2021-01-01 00:59:30+00:00')
        self.assertEqual(timing['eligible_open_at'], '2021-01-01 01:00:00+00:00')
        self.assertEqual(timing['requested_delay_seconds'], 90)
        self.assertEqual(timing['sampled_delay_seconds'], 120)
        self.assertEqual(timing['rounding_delay_seconds'], 30)
        self.assertEqual(events, unchanged)
        self.assertEqual(trade['stop_price'], 99.)

    def test_delay_rejects_missing_or_shifted_minute_grid(self):
        missing = self.small_bars().drop(self.small_bars().index[4])
        shifted = self.small_bars()
        shifted.index += pd.Timedelta(seconds=15)
        for frame in (missing, shifted, shifted.iloc[:1]):
            with self.subTest(first=frame.index[0]), self.assertRaises(ValueError):
                self.m.simulate_events(frame, [event(0)], decision_delay_seconds=90)
        with self.assertRaises(ValueError):
            self.m.detect_events(shifted)

    def test_delay_beyond_data_is_unfilled_and_keeps_event_identity(self):
        b = self.small_bars(5)
        zero = self.simulate(b)
        delayed = self.simulate(b, decision_delay_seconds=90)
        self.assertEqual(zero['event_ledger'][0]['status'], 'open_censored')
        self.assertEqual(delayed['trades'], [])
        self.assertEqual(delayed['open_positions'], [])
        self.assertEqual(delayed['unfilled_at_end'], 1)
        item = delayed['event_ledger'][0]
        self.assertEqual(item['status'], 'unfilled')
        self.assertEqual(item['eligible_open_at'], '2021-01-01 00:05:00+00:00')
        self.assertEqual(item['event_id'], zero['event_ledger'][0]['event_id'])

    def test_stop_remains_frozen_when_delay_invalidates_entry_price(self):
        b = self.small_bars()
        b.loc[b.index[5], ['open', 'high', 'low', 'close']] = [98., 99., 97., 98.]
        output = self.simulate(b, decision_delay_seconds=90)
        self.assertEqual(output['skipped_invalid_stop'], 1)
        self.assertEqual(output['trades'], [])
        item = output['event_ledger'][0]
        self.assertEqual(item['status'], 'invalid_stop')
        self.assertEqual(item['stop_price'], 99.)
        self.assertEqual(item['eligible_entry_price'], 98.)

    def test_delayed_entry_checks_own_bar_stop_and_preserves_fees(self):
        b = self.small_bars()
        b.loc[b.index[3], 'low'] = 95.  # Before the delayed entry: ignored.
        b.loc[b.index[5], 'low'] = 98.
        output = self.simulate(b, decision_delay_seconds=90)
        trade = output['trades'][0]
        self.assertEqual((trade['entry_idx'], trade['exit_idx']), (5, 5))
        self.assertEqual(trade['exit_price'], 99.)
        self.assertEqual(trade['reason'], 'stop')
        self.assertAlmostEqual(trade['pnl'], -11.2)
        self.assertAlmostEqual(trade['fees'], 1.2)

    def test_delayed_stop_gap_uses_adverse_open(self):
        b = self.small_bars()
        b.loc[b.index[6], ['open', 'high', 'low', 'close']] = [97., 98., 96., 97.]
        trade = self.simulate(b, decision_delay_seconds=90)['trades'][0]
        self.assertEqual((trade['entry_idx'], trade['exit_idx']), (5, 6))
        self.assertEqual(trade['exit_price'], 97.)
        self.assertAlmostEqual(trade['pnl'], -31.2)

    def test_delayed_hold_deadline_and_censoring_start_at_actual_entry(self):
        b = self.small_bars(9)
        zero = self.simulate(b)
        delayed = self.simulate(b, decision_delay_seconds=90)
        self.assertEqual(zero['trades'][0]['exit_idx'], 7)
        self.assertEqual(delayed['trades'], [])
        self.assertEqual(delayed['event_ledger'][0]['status'], 'open_censored')
        self.assertEqual(delayed['open_positions'][0]['entry_idx'], 5)
        self.assertAlmostEqual(delayed['open_positions'][0]['marked_pnl'], -.6)
        complete = self.small_bars(10)
        complete.loc[complete.index[9], ['open', 'high', 'low', 'close']] = [101., 105., 98., 102.]
        trade = self.simulate(complete, decision_delay_seconds=90)['trades'][0]
        self.assertEqual(trade['exit_idx'], 9)
        self.assertEqual(trade['exit_price'], 101.)
        self.assertEqual(trade['reason'], 'time')

    def test_delayed_invalid_first_entry_changes_lockout_population_without_changing_ids(self):
        b = self.small_bars(14)
        b.loc[b.index[5], ['open', 'high', 'low', 'close']] = [98., 99., 97., 98.]
        events = [event(2), event(4)]
        zero = self.simulate(b, events)
        delayed = self.simulate(b, events, decision_delay_seconds=90)
        self.assertEqual([x['status'] for x in zero['event_ledger']], ['completed', 'skipped_busy'])
        self.assertEqual([x['status'] for x in delayed['event_ledger']], ['invalid_stop', 'completed'])
        self.assertEqual([x['event_id'] for x in zero['event_ledger']],
                         [x['event_id'] for x in delayed['event_ledger']])
        self.assertEqual(delayed['trades'][0]['event']['reclaim_idx'], 4)
        self.assertEqual(delayed['trades'][0]['entry_idx'], 7)

    def test_delayed_early_stop_keeps_lockout_until_actual_entry_deadline(self):
        b = self.small_bars(14)
        b.loc[b.index[5], 'low'] = 98.
        output = self.simulate(b, [event(2), event(4), event(6)], decision_delay_seconds=90)
        self.assertEqual([x['status'] for x in output['event_ledger']],
                         ['completed', 'skipped_busy', 'completed'])
        self.assertEqual(output['trades'][0]['exit_idx'], 5)
        self.assertEqual(output['trades'][1]['entry_idx'], 9)
        self.assertEqual(output['trades'][1]['exit_idx'], 13)

    def test_invalid_delays_and_nonzero_close_mode_are_rejected(self):
        for delay in (-1, True, np.bool_(False), 1.5, 90., np.nan, np.inf, '90', None):
            with self.subTest(delay=delay), self.assertRaises(ValueError):
                self.simulate(self.small_bars(), decision_delay_seconds=delay)
        with self.assertRaises(ValueError):
            self.simulate(self.small_bars(), entry_mode='close', decision_delay_seconds=90)
        valid = self.simulate(self.small_bars(), decision_delay_seconds=np.int64(90))
        self.assertEqual(valid['trades'][0]['entry_idx'], 5)
        close = self.simulate(self.small_bars(), entry_mode='close', decision_delay_seconds=0)
        self.assertEqual((close['trades'][0]['entry_idx'], close['trades'][0]['exit_idx']), (2, 6))

    def test_cli_binds_delay_to_parameters_and_discloses_sampling_limit(self):
        with tempfile.TemporaryDirectory() as directory:
            bars_path = Path(directory)/'bars.parquet'
            out_path = Path(directory)/'report'
            setup().to_parquet(bars_path)
            with patch('sys.argv', ['minute_sweep_validation', '--bars', str(bars_path),
                                   '--out', str(out_path), '--decision-delay-seconds', '90']):
                self.m.main()
            report = json.loads((out_path/'minute_replay.json').read_text())
        self.assertEqual(report['parameters']['decision_delay_seconds'], 90)
        self.assertEqual(report['event_ledger'][0]['sampled_delay_seconds'], 120)
        self.assertTrue(any('round' in item.lower() and 'minute' in item.lower()
                            for item in report['limitations']))


if __name__ == '__main__':
    unittest.main()
