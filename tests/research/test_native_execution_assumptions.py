"""Execution limitations of source paper fills, not desired market behavior."""
import unittest

import pandas as pd

from engine.portfolio.archetype_allocator import ArchetypeSignal
from scripts.research.virtual_book_replay import VirtualBookFixture


class NativeExecutionAssumptionTests(unittest.TestCase):
    def test_close_known_entry_is_backdated_and_avoids_signal_bar_extremes(self):
        start = pd.Timestamp('2026-01-01T00:00Z')
        book = VirtualBookFixture(initial_cash=100000., commission_rate=.0004, slippage_bps=5.)
        signal = ArchetypeSignal('wick_trap', 'long', .95, 100., 95., 110., .95,
                                'neutral', start, {'hard_gates_passed': True})
        row = book.step(dict(timestamp=start, open=99., high=115., low=94., close=100.,
                             volume=10., atr_14=2., regime_label='neutral'),
                        [signal], start+pd.Timedelta('1h'))
        position = next(iter(book.runner.positions.values()))
        self.assertEqual(position.entry_time, start)
        self.assertEqual(row['available_at']-position.entry_time, pd.Timedelta('1h'))
        self.assertAlmostEqual(position.entry_price, 100.05)
        self.assertEqual(book.runner.trades, [])

    def test_native_gap_stop_price_can_lie_above_entire_next_bar(self):
        start = pd.Timestamp('2026-01-01T00:00Z')
        book = VirtualBookFixture(initial_cash=100000., commission_rate=.0004, slippage_bps=5.)
        signal = ArchetypeSignal('wick_trap', 'long', .95, 100., 95., 110., .95,
                                'neutral', start, {'hard_gates_passed': True})
        book.step(dict(timestamp=start, open=99., high=101., low=99., close=100.,
                       volume=10., atr_14=2., regime_label='neutral'),
                  [signal], start+pd.Timedelta('1h'))
        book.step(dict(timestamp=start+pd.Timedelta('1h'), open=90., high=92., low=89., close=91.,
                       volume=10., atr_14=2., regime_label='neutral'),
                  [], start+pd.Timedelta('2h'))
        trade = book.runner.trades[0]
        self.assertEqual(trade.exit_reason, 'stop_loss')
        self.assertAlmostEqual(trade.exit_price, 94.9525)
        self.assertGreater(trade.exit_price, 92.)
