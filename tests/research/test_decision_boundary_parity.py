import importlib.util
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT/'scripts/research/decision_boundary_parity.py'
LIVE = ROOT/'bin/live/v11_shadow_runner.py'
BACKTEST = ROOT/'bin/backtest_v11_standalone.py'


class BoundaryParityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not SCRIPT.exists():
            raise AssertionError('Decision boundary implementation is missing')
        spec=importlib.util.spec_from_file_location('boundary_parity',SCRIPT)
        cls.m=importlib.util.module_from_spec(spec); spec.loader.exec_module(cls.m)

    def pair(self, **kw):
        case=dict(score=.1, threshold=.3, gates_passed=True, global_bypass=True,
                  archetype_bypass=False, enforce=True, opt_out=False)
        case.update(kw)
        return [self.m.run_boundary(p,k,case)['accepted']
                for p,k in [(LIVE,'live'),(BACKTEST,'backtest')]]

    def test_actual_global_bypass_paths_disagree(self):
        self.assertEqual(self.pair(),[True,False])

    def test_without_bypass_both_reject(self):
        self.assertEqual(self.pair(global_bypass=False),[False,False])

    def test_threshold_equality_passes_both(self):
        self.assertEqual(self.pair(score=.3),[True,True])

    def test_archetype_bypass_only_backtester_recognizes_at_this_boundary(self):
        self.assertEqual(self.pair(global_bypass=False, archetype_bypass=True),[False,True])

    def test_failed_gates_block_global_live_bypass(self):
        self.assertEqual(self.pair(gates_passed=False),[False,False])

    def test_live_gate_opt_out_is_exercised(self):
        self.assertEqual(self.pair(gates_passed=False,opt_out=True),[True,False])

    def test_matrix_has_both_matches_and_mismatches(self):
        r=self.m.compare_boundaries(ROOT)
        self.assertEqual(len(r['cases']),192)
        self.assertGreater(len(r['mismatches']),0)
        self.assertLess(len(r['mismatches']),192)


if __name__=='__main__':
    unittest.main()
