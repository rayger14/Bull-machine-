import importlib.util
from pathlib import Path
import unittest

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


if __name__ == '__main__':
    unittest.main()
