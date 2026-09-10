import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import pandas as pd

from scripts.research.replay_clock import digest


def wick_fixture():
    return dict(open=99., high=101., low=90., close=100., volume=10.,
                wick_lower_ratio=.8, wick_upper_ratio=.05, volume_zscore=1.,
                liquidity_score=1., wyckoff_bullish_score=1.,
                tf4h_wyckoff_bullish_score=1., tf1d_wyckoff_bullish_score=1.,
                rsi_14=50., adx=25., adx_14=25., atr_14=2., atr_20=2.,
                smc_score=1., tf1h_bos_bullish=1., tf1h_bos_bearish=0.,
                tf1h_bos_detected=1., fusion_smc=1., prev_close=100.,
                funding_Z=0., alt_basket_ret_4h=0., stables_rot_rising=0.)


class EngineSignalReplayTests(unittest.TestCase):
    def module(self):
        path = Path(__file__).resolve().parents[2] / 'scripts/research/engine_signal_replay.py'
        self.assertTrue(path.exists(), 'Actual-source engine signal adapter is missing')
        spec = importlib.util.spec_from_file_location('engine_signal_test', path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    def test_real_detection_runs_all_17_once_without_allocation(self):
        engine = self.module().SignalEngine()
        with patch.object(engine.engine.portfolio_allocator, 'allocate', side_effect=AssertionError('allocation prohibited')):
            outcome = engine.update(dict(wick_fixture(), timestamp=pd.Timestamp('2026-01-01T00:00Z')),
                                    '2026-01-01T01:00Z')
        self.assertEqual(len(outcome['archetypes']), 17)
        self.assertTrue(all(a['detect_calls'] == 1 for a in outcome['archetypes'].values()))
        self.assertEqual(outcome['signals'][0]['archetype_id'], 'liquidity_sweep')
        self.assertEqual(outcome['archetypes']['wick_trap']['last_signal_bar_after'], 1)
        self.assertFalse(outcome['archetypes']['wick_trap']['selected'])
        self.assertEqual(engine.engine.structural_checker.stats['total_checks'], 17)
        self.assertFalse(outcome['certified'])

    def test_deduplicated_signal_still_cools_until_actual_expiry(self):
        engine = self.module().SignalEngine()
        outcomes = []
        for t in pd.date_range('2026-01-01', periods=19, freq='1h', tz='UTC'):
            outcomes.append(engine.update(dict(wick_fixture(), timestamp=t), t+pd.Timedelta('1h')))
        self.assertEqual(outcomes[17]['signals'], [])
        self.assertEqual(outcomes[18]['signals'][0]['archetype_id'], 'wick_trap')
        self.assertEqual(outcomes[18]['archetypes']['wick_trap']['last_signal_bar_after'], 19)

    def test_fresh_original_prehistory_has_identical_state_and_outputs(self):
        module = self.module()
        first, second = module.SignalEngine(), module.SignalEngine()
        for t in pd.date_range('2026-01-01', periods=20, freq='1h', tz='UTC'):
            values = dict(wick_fixture(), timestamp=t)
            a = first.update(values, t+pd.Timedelta('1h'))
            b = second.update(values, t+pd.Timedelta('1h'))
            self.assertEqual(digest(a), digest(b))
        self.assertEqual(digest(first.snapshot()), digest(second.snapshot()))
        self.assertEqual(len(first.engine.regime_service.probabilistic_detector.regime_history), 20)

    def test_duplicate_or_early_hour_cannot_rearm_engine_state(self):
        engine = self.module().SignalEngine()
        values = dict(wick_fixture(), timestamp=pd.Timestamp('2026-01-01T00:00Z'))
        with self.assertRaises(ValueError):
            engine.update(values, '2026-01-01T00:59Z')
        engine.update(values, '2026-01-01T01:00Z')
        with self.assertRaises(ValueError):
            engine.update(values, '2026-01-01T01:00Z')
        self.assertEqual(engine.engine.stats['total_bars'], 1)

    def test_structural_error_is_a_blocker_even_if_source_passes_it(self):
        engine = self.module().SignalEngine()
        # Source catches this detector error and returns a permissive result.
        with patch.object(engine.engine.structural_checker.logic, '_check_K', side_effect=ValueError('fixture detector failure')):
            result = engine.update(dict(wick_fixture(), timestamp=pd.Timestamp('2026-01-01T00:00Z')),
                                   '2026-01-01T01:00Z')
        self.assertIn('structural_error_permissive_fallback', result['blockers'])

    def test_composed_minute_clock_calls_signal_engine_only_at_hour_close(self):
        module = self.module()
        self.assertTrue(hasattr(module, 'run_signal_replay'), 'Full-feature/engine clock composition is missing')
        frame = pd.DataFrame(dict(open=100., high=101., low=99., close=100., volume=10.),
                             index=pd.date_range('2026-01-01', periods=61, freq='1min', tz='UTC'))
        result = module.run_signal_replay(frame, instrument='fixture', timeframe='1min')
        self.assertIsNone(result['rows'][58]['output']['engine_signal'])
        self.assertEqual(len(result['rows'][59]['output']['engine_signal']['archetypes']), 17)
        self.assertIsNone(result['rows'][60]['output']['engine_signal'])
        self.assertEqual(result['state']['engine_signal']['state']['bar_index'], 1)
        self.assertFalse(result['certified'])

    def test_composed_hourly_prefix_restart_preserves_both_engines(self):
        module = self.module()
        self.assertTrue(hasattr(module, 'run_signal_replay'), 'Full-feature/engine clock composition is missing')
        frame = pd.DataFrame(dict(open=100., high=101., low=99., close=100., volume=10.),
                             index=pd.date_range('2026-01-01', periods=3, freq='1h', tz='UTC'))
        full = module.run_signal_replay(frame, instrument='fixture', timeframe='1h')
        prefix = module.run_signal_replay(frame.iloc[:2], instrument='fixture', timeframe='1h')
        resumed = module.run_signal_replay(frame, instrument='fixture', timeframe='1h', checkpoint=prefix['checkpoint'])
        self.assertEqual(digest(prefix['rows']), digest(full['rows'][:2]))
        self.assertEqual(digest(resumed['rows']), digest(full['rows'][2:]))
        self.assertEqual(digest(resumed['state']), digest(full['state']))

    def test_unsupported_ml_and_kelly_switches_rejected(self):
        module = self.module()
        cfg = json.loads((module.ROOT/'configs/champion_paper.json').read_text())
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'config.json'
            for key in ('use_ml_fusion', 'use_kelly_sizing'):
                path.write_text(json.dumps(dict(cfg, **{key: True})))
                with self.subTest(key=key), self.assertRaises(ValueError):
                    module.SignalEngine(path)

    def test_present_but_unloadable_model_is_flagged_as_source_fallback(self):
        module = self.module()
        cfg = json.loads((module.ROOT/'configs/champion_paper.json').read_text())
        with tempfile.TemporaryDirectory() as directory:
            model = Path(directory)/'bad.pkl'
            model.write_bytes(b'not a model')
            cfg['regime_classifier']['model_path'] = str(model)
            path = Path(directory)/'config.json'
            path.write_text(json.dumps(cfg))
            engine = module.SignalEngine(path)
            self.assertIn('regime_model_mock_fallback', engine.blockers)
