"""Report tests target counting/attribution errors, never strategy performance."""
import importlib
from pathlib import Path
import unittest

from scripts.research.engine_signal_replay import EXPECTED_ARCHETYPES


def fixture_result():
    outcomes = {name: dict(native_signal=None, selected=False) for name in EXPECTED_ARCHETYPES}
    outcomes['wick_trap'] = dict(native_signal={'fusion_score': .4}, selected=True)
    return dict(certified=False, blockers=['source_fill_not_executable'], contract_id='fixture',
                scope='fixture', rows=[dict(output=dict(hourly_updated=True,
                    engine_signal=dict(archetypes=outcomes), native_book=dict(
                        acted_signals=[dict(action='ENTRY', archetype='wick_trap')],
                        last_bar_signals=[dict(archetype='wick_trap', status='allocated')],
                        records=[], position_count=1, trade_count=0)))],
                state=dict(hourly_updates=3, native_book=dict(state=dict(
                    cash=98247., equity_curve=[100000., 99996.], positions={'1': {}}, trades=[],
                    phantom_positions={}, phantom_trades=[]))))


class NativePipelineReportTests(unittest.TestCase):
    def module(self):
        self.assertTrue((Path(__file__).resolve().parents[2]/
                         'scripts/research/native_pipeline_report.py').exists(),
                        'Native pipeline report is missing')
        return importlib.import_module('scripts.research.native_pipeline_report')

    def test_counts_emitted_events_not_warmup_or_minute_carry(self):
        result = fixture_result()
        result['rows'].append(dict(output=dict(hourly_updated=False, engine_signal=None, native_book=None)))
        report = self.module().summarize(result)
        self.assertEqual(report['coverage']['emitted_book_hours'], 1)
        self.assertEqual(report['coverage']['book_hours_including_warmup'], 3)
        self.assertEqual(report['archetypes']['wick_trap']['entries'], 1)
        self.assertEqual(report['archetypes']['wick_trap']['selected'], 1)
        self.assertEqual(len(report['archetypes']), 17)
        self.assertEqual(report['archetypes']['spring']['native_candidates'], 0)
        self.assertEqual(report['native_accounting']['open_positions'], 1)
        self.assertFalse(report['execution_causality_certified'])
        self.assertFalse(report['certified'])

    def test_empty_rows_do_not_certify_warmup_positions(self):
        result = fixture_result()
        result['rows'] = []
        report = self.module().summarize(result)
        self.assertIn('no_emitted_book_evidence', report['blockers'])
        self.assertIn('no_emitted_entry_evidence', report['blockers'])
        self.assertEqual(report['native_accounting']['open_positions'], 1)

    def test_rejections_and_scale_outs_are_not_entries_or_independent_positions(self):
        result = fixture_result()
        book = result['rows'][0]['output']['native_book']
        book['acted_signals'] = []
        book['last_bar_signals'] = [dict(archetype='wick_trap', status='rejected',
                                        rejection_stage='bypass_gate_block')]
        book['records'] = [dict(kind='outcome_log', payload=dict(position={'archetype': 'spring'}))]*2
        report = self.module().summarize(result)
        self.assertEqual(report['archetypes']['wick_trap']['entries'], 0)
        self.assertEqual(report['archetypes']['wick_trap']['rejected'], 1)
        self.assertEqual(report['archetypes']['spring']['exit_events'], 2)
        self.assertNotIn('completed_positions', report['native_accounting'])

    def test_missing_archetype_observation_rejects_report(self):
        result = fixture_result()
        del result['rows'][0]['output']['engine_signal']['archetypes']['spring']
        with self.assertRaisesRegex(ValueError, '17'):
            self.module().summarize(result)

    def test_carry_row_cannot_hide_an_entry(self):
        result = fixture_result()
        result['rows'][0]['output']['hourly_updated'] = False
        with self.assertRaisesRegex(ValueError, 'hourly'):
            self.module().summarize(result)

    def test_real_exercise_requires_explicit_costs_and_checks_restart(self):
        import pandas as pd
        bars = pd.DataFrame(dict(open=100., high=101., low=99., close=100., volume=10.),
            index=pd.date_range('2026-01-01', periods=2, freq='1h', tz='UTC'))
        report = self.module().exercise(bars, [], instrument='fixture', timeframe='1h',
            initial_cash=100000., commission_rate=.0004, slippage_bps=5., cuts=[1])
        self.assertTrue(report['checks'][0]['prefix_equal'])
        self.assertTrue(report['checks'][0]['restart_rows_equal'])
        self.assertTrue(report['checks'][0]['restart_state_equal'])
        self.assertEqual(report['coverage']['book_hours_including_warmup'], 2)
        self.assertFalse(report['certified'])

    def test_source_manifest_metadata_is_not_mistaken_for_file_inventory(self):
        from scripts.research.live_feature_replay_report import manifest_files
        root = Path(__file__).resolve().parents[2]
        self.assertEqual(manifest_files(dict(features=dict(files={'example.py': 'sha'}),
            fill_policy='native virtual only', clock_policy='hourly')),
            {str(root/'example.py'): 'sha'})

    def test_output_cannot_overwrite_a_protected_json_configuration(self):
        import tempfile
        with tempfile.TemporaryDirectory() as directory:
            config = Path(directory)/'champion.json'
            config.write_text('{"preserve":true}')
            module = self.module()
            self.assertTrue(hasattr(module, 'validate_output'), 'Protected-output guard missing')
            with self.assertRaisesRegex(ValueError, 'protected'):
                module.validate_output(config, [config])
            self.assertEqual(config.read_text(), '{"preserve":true}')

    def test_output_rejects_live_runtime_state_and_accepts_only_research_roots(self):
        module = self.module()
        root = Path(__file__).resolve().parents[2]
        for relative in ('results/live_signals/state.json', 'results/coinbase_paper/state.json',
                         'results/coinbase_paper/funding_costs.json'):
            with self.subTest(relative=relative), self.assertRaises(ValueError):
                module.validate_output(root/relative, [])
        module.validate_output(root/'results/research_validation_2026_09_10/native_pipeline/fixture.json', [])
