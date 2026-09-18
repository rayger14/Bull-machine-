import importlib.util
from pathlib import Path
import unittest

import pandas as pd

SCRIPT = Path(__file__).resolve().parents[2] / 'scripts/research/gate_observability.py'


class GateObservabilityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not SCRIPT.exists():
            raise AssertionError('Gate observability implementation is missing')
        spec = importlib.util.spec_from_file_location('gate_observability', SCRIPT)
        cls.m = importlib.util.module_from_spec(spec); spec.loader.exec_module(cls.m)

    def test_missing_raw_value_skips_instead_of_counting_pass(self):
        r = self.m.gate_observation({'feature':'funding_Z', 'op':'max', 'value':-.5,
                                     'nan_policy':'skip'}, {})
        self.assertEqual(r['status'], 'skip')
        self.assertEqual(r['missing_inputs'], ['funding_Z'])
        self.assertTrue(r['runtime_passed'])

    def test_constant_zero_funding_fails_instead_of_skipping(self):
        r = self.m.gate_observation({'feature':'funding_Z', 'op':'max', 'value':-.5,
                                     'nan_policy':'skip'}, {'funding_Z':0.})
        self.assertEqual(r['status'], 'fail')
        self.assertEqual(r['missing_inputs'], [])

    def test_derived_default_pass_is_flagged(self):
        r = self.m.gate_observation({'feature':'derived:rsi_extreme_65', 'op':'bool_true'},
                                    {'rsi_14':float('nan')})
        self.assertEqual(r['status'], 'pass')
        self.assertEqual(r['missing_inputs'], ['rsi_14'])
        self.assertTrue(r['derived_defaulted'])

    def test_unknown_derived_is_visible_even_when_runtime_skips(self):
        r = self.m.gate_observation({'feature':'derived:not_registered', 'op':'bool_true',
                                     'nan_policy':'skip'}, {})
        self.assertEqual(r['status'], 'skip')
        self.assertEqual(r['resolution'], 'unknown_derived')

    def test_unknown_operator_is_not_silently_certified(self):
        r = self.m.gate_observation({'feature':'x', 'op':'typo', 'value':1}, {'x':0})
        self.assertEqual(r['status'], 'unsupported_operator')
        self.assertTrue(r['runtime_passed'])

    def test_era_coverage_distinguishes_constant_fail_and_missing_skip(self):
        frame = pd.DataFrame({'funding_Z':[0., 0., float('nan')]},
            index=pd.to_datetime(['2024-01-01','2024-01-02','2025-01-01'], utc=True))
        cfg = {'funding':{'name':'funding', 'direction':'long', 'gate_mode':'soft',
            'hard_gates':[{'feature':'funding_Z','op':'max','value':-.5,'nan_policy':'skip'}]}}
        rows = self.m.audit_frame(frame, cfg, 'fixture')
        old, new = sorted(rows, key=lambda x:x['year'])
        self.assertEqual(old['counts']['fail'], 2)
        self.assertEqual(old['unique_values'], 1)
        self.assertTrue(old['constant_nonmissing'])
        self.assertEqual(new['counts']['skip'], 1)
        self.assertEqual(new['unique_values'], 0)

    def test_empty_frame_preserves_archetype(self):
        cfg={'unused':{'name':'unused','direction':'neutral','hard_gates':[],
                       'enabled':False,'gate_mode':'soft'}}
        rows=self.m.audit_frame(pd.DataFrame(index=pd.DatetimeIndex([], tz='UTC')),cfg,'empty')
        self.assertEqual(rows[0]['archetype'],'unused')
        self.assertEqual(rows[0]['rows'],0)
        self.assertFalse(rows[0]['enabled'])
        self.assertEqual(rows[0]['gate_mode'],'soft')
        self.assertEqual(rows[0]['direction'],'neutral')

    def test_absent_columns_still_evaluate_every_timestamp(self):
        cfg={'squeeze':{'hard_gates':[{'feature':'vol_shock','op':'max',
                                      'value':.1,'nan_policy':'skip'}]}}
        index=pd.date_range('2026-01-01',periods=2,tz='UTC')
        for frame in (pd.DataFrame({'unrelated':[1,2]},index=index),
                      pd.DataFrame(index=index)):
            with self.subTest(columns=list(frame.columns)):
                row=self.m.audit_frame(frame,cfg,'fixture')[0]
                self.assertEqual(row['counts'],{'skip':2})
                self.assertEqual(row['missing_input_rows'],2)
                self.assertEqual(sum(row['counts'].values()),row['rows'])


if __name__ == '__main__':
    unittest.main()
