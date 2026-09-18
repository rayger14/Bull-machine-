"""Native-rule witnesses, not desired strategy behavior or profitability tests.

These pin what the two hypotheses currently distinguish. A later correction
must introduce a separately named contract, not relabel these outcomes as edge.
"""
from pathlib import Path
import unittest

import numpy as np
import pandas as pd
import yaml

from engine.archetypes.logic import ArchetypeLogic
from engine.archetypes.archetype_instance import ArchetypeConfig, ArchetypeInstance
from scripts.research.minute_sweep_validation import detect_events


def minute_setup():
    frame = pd.DataFrame(dict(open=112., high=115., low=110.+np.arange(150)*.001, close=112.),
        index=pd.date_range('2026-01-01', periods=150, freq='1min', tz='UTC'))
    frame.loc[frame.index[[20, 60]], 'low'] = 100.
    frame.loc[frame.index[90], ['low', 'close']] = [99., 101.]
    return frame


class TraderIntentWitnessTests(unittest.TestCase):
    def lc(self, values, history=None):
        path = Path(__file__).resolve().parents[2]/'configs/champion/archetypes_v14rq/liquidity_compression.yaml'
        cfg = yaml.safe_load(path.read_text())
        row = pd.Series(values)
        history = history if history is not None else pd.DataFrame([row])
        identity = ArchetypeLogic({})._check_E(row, history.iloc[-1], history, len(history)-1, 1.)
        arch = ArchetypeInstance(ArchetypeConfig(name=cfg['name'], direction=cfg['direction'],
            hard_gates=cfg['hard_gates'], gate_mode=cfg['gate_mode']))
        return identity, arch._evaluate_gates(values)[0], arch.direction

    def values(self, **changes):
        return dict(dict(volume_zscore=3.5, rsi_14=25., bb_width=.04, chop_score=.2), **changes)

    def test_quiet_coil_and_absorption_below_volume_gate_do_not_qualify(self):
        self.assertEqual(self.lc(self.values(volume_zscore=0., rsi_14=50.)), (False, False, 'long'))
        self.assertEqual(self.lc(self.values(volume_zscore=2., absorption_flag=1)), (True, False, 'long'))

    def test_both_rsi_extremes_pass_the_same_long_identity_and_gates(self):
        self.assertEqual(self.lc(self.values(rsi_14=25.)), (True, True, 'long'))
        self.assertEqual(self.lc(self.values(rsi_14=75.)), (True, True, 'long'))

    def test_previous_compression_sequence_is_not_consumed(self):
        compressed = pd.DataFrame([self.values(bb_width=.01)]*12)
        expanding = pd.DataFrame([self.values(bb_width=.20)]*12)
        self.assertEqual(self.lc(self.values(), compressed), (True, True, 'long'))
        self.assertEqual(self.lc(self.values(), expanding), (True, True, 'long'))

    def test_parent_reclaim_geometry_is_not_consumed_by_lc_identity_or_gates(self):
        reclaimed = self.values(parent_range_low=100., close=101.)
        below = self.values(parent_range_low=100., close=99.)
        self.assertEqual(self.lc(reclaimed), (True, True, 'long'))
        self.assertEqual(self.lc(below), (True, True, 'long'))

    def test_climax_with_missing_gate_evidence_is_permissive_not_confirmation(self):
        values = {key: float('nan') for key in self.values()}
        values['volume_climax_last_3b'] = 1
        self.assertEqual(self.lc(values), (True, True, 'long'))

    def test_minute_first_touch_outside_level_tolerance_invalidates_pair(self):
        frame = minute_setup()
        frame.loc[frame.index[20], 'low'] = 100.20
        self.assertEqual(detect_events(frame), [])

    def test_minute_sweep_before_confirmation_invalidates_pivot(self):
        frame = minute_setup()
        frame.loc[frame.index[74], ['low', 'close']] = [99., 101.]
        self.assertEqual(detect_events(frame), [])

    def test_minute_wick_above_level_does_not_replace_close_reclaim(self):
        frame = minute_setup()
        frame.loc[frame.index[90:], ['open', 'high', 'low', 'close']] = [100., 101., 99., 100.]
        self.assertEqual(detect_events(frame), [])

    def test_minute_first_sweep_deadline_does_not_reset_on_new_wick(self):
        for reclaim, expected in ((120, [120]), (121, [])):
            frame = minute_setup()
            frame.loc[frame.index[90:], ['open', 'high', 'low', 'close']] = [100., 101., 99., 100.]
            frame.loc[frame.index[reclaim], 'close'] = 101.
            self.assertEqual([event['reclaim_idx'] for event in detect_events(frame)], expected)

    def test_minute_selector_does_not_apply_parent_context_permission(self):
        bullish, bearish = minute_setup(), minute_setup()
        bullish['parent_range_direction'] = 'bullish'
        bearish['parent_range_direction'] = 'bearish'
        self.assertEqual([event['reclaim_idx'] for event in detect_events(bullish)], [90])
        self.assertEqual(detect_events(bullish), detect_events(bearish))

    def test_oi_gate_accepts_zero_or_missing_oi_but_rejects_positive_change(self):
        path = Path(__file__).resolve().parents[2]/'configs/champion/archetypes_v14rq/oi_divergence.yaml'
        cfg = yaml.safe_load(path.read_text())
        arch = ArchetypeInstance(ArchetypeConfig(name=cfg['name'], direction=cfg['direction'],
            hard_gates=cfg['hard_gates'], gate_mode=cfg['gate_mode']))
        flat = dict(oi_change_4h=0., oi_change_24h=0., taker_imbalance=0.,
                    volume_zscore=1.1, rsi_14=31.)
        missing = dict(volume_zscore=1.1, rsi_14=31.)
        self.assertTrue(arch._evaluate_gates(flat)[0])
        self.assertTrue(arch._evaluate_gates(missing)[0])
        self.assertFalse(arch._evaluate_gates(dict(flat, oi_change_4h=.0001))[0])
