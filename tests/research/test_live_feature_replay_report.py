import importlib.util
import inspect
from pathlib import Path
import unittest
import json
import tempfile
import pandas as pd


class FeatureReplayReportTests(unittest.TestCase):
    def module(self):
        path = Path(__file__).resolve().parents[2] / 'scripts/research/live_feature_replay_report.py'
        spec = importlib.util.spec_from_file_location('lfc_report_test', path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    def test_report_exercises_restart_without_certifying_strategy(self):
        path = Path(__file__).resolve().parents[2] / 'scripts/research/live_feature_replay_report.py'
        self.assertTrue(path.exists(), 'Reproducible full-feature experiment report is missing')
        spec = importlib.util.spec_from_file_location('lfc_report_test', path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        frame = pd.DataFrame(dict(open=[100., 101.], high=[102., 103.],
                                  low=[99., 100.], close=[101., 102.], volume=[10., 20.]),
                             index=pd.date_range('2026-01-01', periods=2, freq='1h', tz='UTC'))
        report = module.exercise(frame, [], instrument='fixture-BTC', timeframe='1h', cuts=[1])
        self.assertFalse(report['certified'])
        self.assertTrue(report['checks'][0]['prefix_equal'])
        self.assertTrue(report['checks'][0]['restart_rows_equal'])
        self.assertTrue(report['checks'][0]['restart_state_equal'])
        self.assertEqual(report['coverage']['hourly_updates'], 2)
        self.assertEqual(len(report['archetypes']), 17)
        self.assertEqual(report['archetypes']['liquidity_compression'], 'signal_layer_not_replayed')

    def test_empty_input_rejected(self):
        path = Path(__file__).resolve().parents[2] / 'scripts/research/live_feature_replay_report.py'
        self.assertTrue(path.exists(), 'Reproducible full-feature experiment report is missing')
        spec = importlib.util.spec_from_file_location('lfc_report_test', path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        with self.assertRaises(ValueError):
            module.exercise(pd.DataFrame(), [], instrument='fixture-BTC', timeframe='1h')

    def test_no_hourly_evidence_and_empty_comparisons_are_explicit(self):
        module = self.module()
        frame = pd.DataFrame(dict(open=100., high=101., low=99., close=100., volume=10.),
                             index=pd.date_range('2026-01-01', periods=2, freq='1min', tz='UTC'))
        report = module.exercise(frame, [], instrument='fixture', timeframe='1min', cuts=[1])
        self.assertIn('no_hourly_feature_evidence', report['blockers'])
        self.assertEqual(report['checks'][0].get('prefix_rows_compared'), 1)
        self.assertEqual(report['checks'][0].get('restart_rows_compared'), 1)

    def test_reduced_inventory_is_not_reported_as_all_17(self):
        module = self.module()
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            (directory/'only.yaml').write_text('name: liquidity_compression\nenabled: true\n')
            config = directory/'config.json'
            config.write_text(json.dumps({'archetype_config_dir': str(directory)}))
            frame = pd.DataFrame(dict(open=100., high=101., low=99., close=100., volume=10.),
                                 index=pd.date_range('2026-01-01', periods=2, freq='1h', tz='UTC'))
            report = module.exercise(frame, [], instrument='fixture', timeframe='1h', config=config)
            self.assertIn('archetype_inventory_mismatch', report['blockers'])

    def test_config_changed_during_replay_rejects_attribution(self):
        module = self.module()
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            (directory/'only.yaml').write_text('name: liquidity_compression\n')
            config = directory/'config.json'
            config.write_text(json.dumps({'archetype_config_dir': str(directory)}))
            frame = pd.DataFrame(dict(open=100., high=101., low=99., close=100., volume=10.),
                                 index=pd.date_range('2026-01-01', periods=2, freq='1h', tz='UTC'))
            def mutate(message):
                config.write_text(json.dumps({'archetype_config_dir': str(directory), 'changed': True}))
            with self.assertRaisesRegex(ValueError, 'changed'):
                module.exercise(frame, [], instrument='fixture', timeframe='1h', config=config, progress=mutate)

    def test_signal_report_retains_all_archetype_evaluations(self):
        module = self.module()
        self.assertIn('include_signals', inspect.signature(module.exercise).parameters)
        frame = pd.DataFrame(dict(open=100., high=101., low=99., close=100., volume=10.),
                             index=pd.date_range('2026-01-01', periods=2, freq='1h', tz='UTC'))
        report = module.exercise(frame, [], instrument='fixture', timeframe='1h', include_signals=True)
        self.assertEqual(len(report['archetypes']), 17)
        self.assertEqual(report['archetypes']['liquidity_compression']['evaluations'], 2)
        self.assertEqual(report['coverage']['signal_engine_hours_including_warmup'], 2)
        self.assertFalse(report['certified'])
