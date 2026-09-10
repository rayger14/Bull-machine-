"""Actual-source adapter regressions; no exchange or live runner is constructed."""
import importlib.util
from pathlib import Path
import socket
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from scripts.research.replay_clock import Observation, digest

ROOT = Path(__file__).resolve().parents[2]
MODULE = ROOT / 'scripts/research/live_feature_replay.py'


def candles(n=28, freq='1h', start='2026-01-01'):
    index = pd.date_range(start, periods=n, freq=freq, tz='UTC')
    prices = 100 + np.arange(n) * .1
    return pd.DataFrame(dict(open=prices, high=prices+1, low=prices-1,
                             close=prices+.2, volume=10.), index=index)


def observation(feature, value, at='2026-01-01T01:00Z', ident='sample'):
    return Observation(ident, feature, value, 'BTC-test', 'fixture', 'declared',
                       'fixture-v1', at, at)


def derivatives(funding=.001, ls=1.):
    return dict(oi_value=100., oi_change_4h=.02, oi_change_24h=.03,
                funding_rate=funding, ls_ratio=ls,
                taker_buy_vol_1h=75., taker_sell_vol_1h=25.)


class LiveFeatureReplayTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.m = None
        if MODULE.exists():
            spec = importlib.util.spec_from_file_location('lfc_research_test', MODULE)
            cls.m = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(cls.m)

    def run_case(self, frame, observations=(), **kwargs):
        self.assertIsNotNone(self.m, 'Actual-source live feature replay adapter is missing')
        return self.m.run_replay(frame, observations, instrument='BTC-test', **kwargs)

    def test_real_feature_computer_advances_once_and_cannot_certify_full_pipeline(self):
        result = self.run_case(candles(2), timeframe='1h')
        self.assertEqual(result['state']['hourly_updates'], 2)
        self.assertEqual(result['rows'][-1]['output']['features']['close'], 100.3)
        self.assertIn('liquidity_score', result['rows'][-1]['output']['features'])
        self.assertFalse(result['certified'])
        self.assertIn('missing_derivatives_snapshot', result['blockers'])
        self.assertIn('full_decision_book_not_replayed', result['blockers'])

    def test_minute_clock_does_not_advance_hourly_engine_early(self):
        result = self.run_case(candles(61, '1min'), timeframe='1min')
        rows = result['rows']
        self.assertIsNone(rows[58]['output']['features'])
        self.assertEqual(rows[58]['output']['context']['1h']['developing']['constituents'], 59)
        self.assertEqual(rows[59]['output']['features']['volume'], 600.)
        self.assertEqual(rows[59]['output']['feature_age_seconds'], 0.)
        self.assertEqual(rows[60]['output']['feature_age_seconds'], 60.)
        self.assertEqual(result['state']['hourly_updates'], 1)

    def test_hourly_features_equal_same_minute_aggregates(self):
        minute = candles(120, '1min')
        hourly = minute.resample('1h').agg(dict(open='first', high='max', low='min', close='last', volume='sum'))
        a = self.run_case(minute, timeframe='1min')
        b = self.run_case(hourly, timeframe='1h')
        for i, j in ((59, 0), (119, 1)):
            self.assertEqual(digest(a['rows'][i]['output']['features']), digest(b['rows'][j]['output']['features']))

    def test_actual_funding_and_long_short_sampling_transitions(self):
        observations = [observation('derivatives_snapshot', derivatives(), ident='a'),
                        observation('derivatives_snapshot', derivatives(.003, 3.),
                                    at='2026-01-02T00:00Z', ident='b')]
        result = self.run_case(candles(24), observations, timeframe='1h')
        self.assertEqual(result['rows'][8]['output']['features']['funding_Z'], 0.)
        self.assertEqual(len(result['state']['computer']['attributes']['_funding_history']), 24)
        self.assertEqual(len(result['state']['computer']['attributes']['_ls_history']), 24)
        self.assertEqual(result['rows'][22]['output']['features']['ls_ratio_extreme'], 0.)
        self.assertAlmostEqual(result['rows'][23]['output']['features']['ls_ratio_extreme'], np.sqrt(23), places=7)
        self.assertAlmostEqual(result['rows'][23]['output']['features']['taker_imbalance'], .5)
        self.assertAlmostEqual(result['rows'][23]['output']['features']['funding_Z'], np.sqrt(23), places=5)

    def test_zero_derivatives_funding_skipped_but_two_channels_append_twice(self):
        zero = self.run_case(candles(1), [observation('derivatives_snapshot', derivatives(0.))], timeframe='1h')
        self.assertEqual(zero['state']['computer']['attributes']['_funding_history'], [])
        both = self.run_case(candles(1), [observation('candle_funding_rate', 0., ident='a'),
                 observation('derivatives_snapshot', derivatives(.001), ident='b')], timeframe='1h')
        self.assertEqual(both['state']['computer']['attributes']['_funding_history'], [0., .001])

    def test_future_append_and_multi_cut_restart_preserve_features_and_state(self):
        frame = candles(28)
        observations = [observation('derivatives_snapshot', derivatives(), ident='a'),
                        observation('derivatives_snapshot', derivatives(.003, 2.),
                                    at='2026-01-02T03:00Z', ident='b')]
        full = self.run_case(frame, observations, timeframe='1h')
        for cut in (10, 25):
            prefix = self.run_case(frame.iloc[:cut], observations[:1], timeframe='1h')
            self.assertEqual(digest(prefix['rows']), digest(full['rows'][:cut]))
            resumed = self.run_case(frame, observations, timeframe='1h', checkpoint=prefix['checkpoint'])
            self.assertEqual(digest(resumed['rows']), digest(full['rows'][cut:]))
            self.assertEqual(digest(resumed['state']), digest(full['state']))

    def test_completed_context_boundaries_and_incomplete_first_hour(self):
        result = self.run_case(candles(24), timeframe='1h')
        self.assertIsNone(result['rows'][2]['output']['context']['4h']['completed'])
        self.assertEqual(result['rows'][3]['output']['context']['4h']['completed']['constituents'], 4)
        self.assertEqual(result['rows'][23]['output']['context']['1d']['completed']['constituents'], 24)
        partial = self.run_case(candles(59, '1min', '2026-01-01T00:01'), timeframe='1min')
        self.assertEqual(partial['state']['hourly_updates'], 0)
        self.assertIn('incomplete_hour_not_ingested', partial['blockers'])

    def test_network_attempts_fail_even_when_source_swallows_exception(self):
        self.assertIsNotNone(self.m, 'Actual-source live feature replay adapter is missing')
        with self.assertRaisesRegex(RuntimeError, 'Network'):
            with self.m.deny_network():
                try:
                    socket.create_connection(('example.invalid', 80))
                except RuntimeError:
                    pass

    def test_unknown_inputs_and_non_hourly_native_cadence_rejected(self):
        self.assertIsNotNone(self.m, 'Actual-source live feature replay adapter is missing')
        with self.assertRaises(ValueError):
            self.run_case(candles(1), [observation('liquidity_score', .5)], timeframe='1h')
        with self.assertRaises(ValueError):
            self.run_case(candles(1, '5min'), timeframe='5min')

    def test_leading_partial_context_is_incomplete_immediately(self):
        result = self.run_case(candles(1, '1min', '2026-01-01T00:01'), timeframe='1min')
        context = result['rows'][0]['output']['context']['1h']
        self.assertIsNone(context['developing'])
        self.assertEqual(context['incomplete']['constituents'], 1)

    def test_held_feature_lineage_does_not_follow_new_minute_observation(self):
        observations = [observation('derivatives_snapshot', derivatives(), ident='a'),
                        observation('derivatives_snapshot', derivatives(),
                                    at='2026-01-01T01:01Z', ident='b')]
        result = self.run_case(candles(61, '1min'), observations, timeframe='1min')
        held = result['rows'][-1]
        self.assertEqual(held['observation_ids']['derivatives_snapshot'], 'b')
        self.assertEqual(held['output'].get('feature_observation_ids'), {'derivatives_snapshot': 'a'})

    def test_nested_nan_retained_in_reference_but_reported_invalid(self):
        result = self.run_case(candles(1), [observation('derivatives_snapshot', derivatives(float('nan')))], timeframe='1h')
        history = result['state']['computer']['attributes']['_funding_history']
        self.assertTrue(np.isnan(history[0]))
        self.assertIn('invalid_derivative:funding_rate', result['blockers'])

    def test_macro_snapshot_cannot_overwrite_arbitrary_engine_features(self):
        for channel in ('macro_features', 'cme_oi_features'):
            with self.subTest(channel=channel), self.assertRaises(ValueError):
                self.run_case(candles(1), [observation(channel, {'close': 999.})], timeframe='1h')

    def test_state_with_hidden_slots_is_not_silently_complete(self):
        self.assertIsNotNone(self.m)
        class HiddenState:
            __slots__ = ('retained', '__dict__')
        state = HiddenState()
        state.retained = 2
        unsupported = set()
        self.m.state_view(state, unsupported)
        self.assertTrue(unsupported)

    def test_changed_numeric_dependency_rejects_checkpoint(self):
        prefix = self.run_case(candles(1), timeframe='1h')
        original = self.m.metadata.version
        with patch.object(self.m.metadata, 'version', lambda p: 'changed-scipy' if p == 'scipy' else original(p)):
            with self.assertRaisesRegex(ValueError, 'checkpoint'):
                self.run_case(candles(2), timeframe='1h', checkpoint=prefix['checkpoint'])


if __name__ == '__main__':
    unittest.main()
