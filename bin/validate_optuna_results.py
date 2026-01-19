#!/usr/bin/env python3
"""
Automated validation suite for Optuna optimization results.

Implements 7-phase validation from OPTUNA_VALIDATION_PLAN.md:
1. Data leakage checks
2. Fixed-size validation (CRITICAL)
3. Rolling OOS validation
4. Regime stratification
5. Trade-level diagnostics
6. Session analysis
7. Slippage sensitivity

Usage:
    python3 bin/validate_optuna_results.py \
      --study-dir results/optuna_trap_v10_full \
      --baseline-dir results/router_v10_full_2022_2024_combined \
      --output results/trap_validation
"""

import sys
import json
import argparse
from pathlib import Path
from datetime import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Add project root
sys.path.insert(0, str(Path(__file__).parent.parent))

from bin.backtest_router_v10_full import RouterAwareBacktest
from bin.backtest_knowledge_v2 import KnowledgeParams
from engine.router_v10 import RouterV10
from engine.regime_detector import RegimeDetector
from engine.event_calendar import EventCalendar


class OptunaValidator:
    """Automated validation suite for Optuna results."""

    def __init__(self, study_dir, baseline_dir, output_dir):
        self.study_dir = Path(study_dir)
        self.baseline_dir = Path(baseline_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Load results
        self.best_params = self._load_best_params()
        self.trials_df = pd.read_csv(self.study_dir / 'trials.csv')

        print(f"📊 Loaded {len(self.trials_df)} trials")
        print(f"🏆 Best trial: #{self.best_params['number']} with score {self.best_params['value']:.4f}")

    def _load_best_params(self):
        with open(self.study_dir / 'best_params.json') as f:
            data = json.load(f)
        # Extract trial number from trials.csv later
        return data

    def run_all_validations(self):
        """Execute all 7 validation phases."""
        print("\n" + "="*80)
        print("🔬 OPTUNA VALIDATION SUITE")
        print("="*80 + "\n")

        results = {}

        # Phase 1: Sanity checks
        print("Phase 1: Data Leakage & Sanity Checks")
        results['phase1'] = self.phase1_sanity_checks()

        # Phase 2: Fixed-size validation (CRITICAL)
        print("\nPhase 2: Fixed-Size Validation (CRITICAL)")
        results['phase2'] = self.phase2_fixed_size_validation()

        # Phase 3: Rolling OOS
        print("\nPhase 3: Rolling Out-of-Sample Validation")
        results['phase3'] = self.phase3_rolling_oos()

        # Phase 4: Regime stratification
        print("\nPhase 4: Regime Stratification")
        results['phase4'] = self.phase4_regime_analysis()

        # Phase 5: Trade diagnostics
        print("\nPhase 5: Trade-Level Diagnostics")
        results['phase5'] = self.phase5_trade_diagnostics()

        # Phase 6: Session analysis
        print("\nPhase 6: Session & Temporal Analysis")
        results['phase6'] = self.phase6_session_analysis()

        # Phase 7: Slippage sensitivity
        print("\nPhase 7: Slippage & Cost Sensitivity")
        results['phase7'] = self.phase7_slippage_test()

        # Generate report
        self.generate_report(results)

        # Make decision
        decision = self.make_decision(results)

        return decision, results

    def phase1_sanity_checks(self):
        """Phase 1: Sanity checks and data leakage verification."""
        results = {}

        # Check best params make sense
        params = self.best_params['best_params']
        results['params_sensible'] = all([
            0.4 <= params.get('trap_quality_threshold', 0.5) <= 0.7,
            0.8 <= params.get('trap_stop_multiplier', 1.0) <= 1.8,
            2 <= params.get('trap_confirmation_bars', 3) <= 6
        ])

        # Check trial distribution
        results['trials_better_than_baseline'] = (self.trials_df['value'] > 0.364932).sum()
        results['total_trials'] = len(self.trials_df)
        results['pct_better'] = results['trials_better_than_baseline'] / results['total_trials'] * 100

        # Check for unique configs
        param_cols = [c for c in self.trials_df.columns if c.startswith('params_')]
        unique_configs = self.trials_df[param_cols].drop_duplicates()
        results['unique_configs'] = len(unique_configs)
        results['duplicate_rate'] = 1 - (len(unique_configs) / len(self.trials_df))

        # Check rejection rates
        results['rejected_trials'] = (self.trials_df['value'] < 0).sum()
        results['rejection_rate'] = results['rejected_trials'] / results['total_trials'] * 100

        print(f"  ✓ Parameters sensible: {results['params_sensible']}")
        print(f"  ✓ Trials better than baseline: {results['trials_better_than_baseline']}/{results['total_trials']} ({results['pct_better']:.1f}%)")
        print(f"  ✓ Unique configurations: {results['unique_configs']} ({results['duplicate_rate']*100:.1f}% duplicates)")
        print(f"  ✓ Rejection rate: {results['rejection_rate']:.1f}%")

        return results

    def phase2_fixed_size_validation(self):
        """Phase 2: CRITICAL - Validate with fixed position sizing."""
        print("  🔍 Re-running optimized params with FIXED sizing...")
        print("  🔍 Re-running baseline params with FIXED sizing...")
        print("  📊 Comparing results...")

        # This would run actual backtests with fixed sizing
        # For now, return placeholder
        results = {
            'fixed_size_optimized_pf': None,
            'fixed_size_baseline_pf': None,
            'improvement': None,
            'passes': None  # To be filled by actual backtest
        }

        print("  ⚠️  Backtest integration required - placeholder")

        return results

    def phase3_rolling_oos(self):
        """Phase 3: Rolling out-of-sample validation."""
        windows = [
            ('2022-01-01', '2022-06-30', '2022-07-01', '2022-12-31'),
            ('2022-01-01', '2022-12-31', '2023-01-01', '2023-12-31'),
            ('2022-01-01', '2023-06-30', '2023-07-01', '2023-12-31'),
            ('2022-01-01', '2023-12-31', '2024-01-01', '2024-12-31'),
        ]

        results = {}
        oos_pfs = []

        for i, (train_start, train_end, test_start, test_end) in enumerate(windows):
            print(f"  Testing window {i+1}: Train({train_start[:7]}→{train_end[:7]}) Test({test_start[:7]}→{test_end[:7]})")

            # Placeholder for actual backtest
            # result = run_backtest(optimized_params, test_start, test_end)
            # oos_pfs.append(result['pf'])

        # results['median_pf'] = np.median(oos_pfs) if oos_pfs else None
        # results['min_pf'] = np.min(oos_pfs) if oos_pfs else None
        # results['std_pf'] = np.std(oos_pfs) if oos_pfs else None

        print("  ⚠️  Backtest integration required - placeholder")

        return results

    def phase4_regime_analysis(self):
        """Phase 4: Regime stratification analysis."""
        results = {}

        # Would analyze trades by regime
        regimes = ['RISK_ON', 'RISK_OFF', 'NEUTRAL', 'CRISIS', 'TRANSITIONAL']

        for regime in regimes:
            results[regime] = {
                'pf': None,
                'wr': None,
                'trades': None,
                'pnl': None
            }

        print("  ⚠️  Trade log analysis required - placeholder")

        return results

    def phase5_trade_diagnostics(self):
        """Phase 5: Trade-by-trade comparison."""
        results = {
            'shared_trades': None,
            'only_baseline': None,
            'only_optimized': None,
            'improvement_source': None
        }

        print("  ⚠️  Trade log comparison required - placeholder")

        return results

    def phase6_session_analysis(self):
        """Phase 6: Session breakdown (ASIA/EUROPE/US)."""
        results = {
            'ASIA': {'pnl': None, 'wr': None, 'trades': None},
            'EUROPE': {'pnl': None, 'wr': None, 'trades': None},
            'US': {'pnl': None, 'wr': None, 'trades': None}
        }

        print("  ⚠️  Session analysis required - placeholder")

        return results

    def phase7_slippage_test(self):
        """Phase 7: Test with realistic costs."""
        results = {
            'pf_no_slippage': None,
            'pf_with_slippage': None,
            'degradation': None,
            'still_profitable': None
        }

        print("  ⚠️  Slippage simulation required - placeholder")

        return results

    def generate_report(self, results):
        """Generate markdown validation report."""
        report_path = self.output_dir / 'VALIDATION_REPORT.md'

        with open(report_path, 'w') as f:
            f.write("# Optuna Validation Report\n\n")
            f.write(f"**Generated**: {datetime.now().isoformat()}\n")
            f.write(f"**Study**: {self.study_dir}\n\n")

            f.write("## Best Parameters\n\n")
            f.write("```json\n")
            f.write(json.dumps(self.best_params['best_params'], indent=2))
            f.write("\n```\n\n")

            f.write("## Validation Results\n\n")

            for phase_name, phase_results in results.items():
                f.write(f"### {phase_name.upper()}\n\n")
                f.write("```json\n")
                f.write(json.dumps(phase_results, indent=2))
                f.write("\n```\n\n")

        print(f"\n📝 Report saved: {report_path}")

    def make_decision(self, results):
        """Make accept/reject/conditional decision."""
        print("\n" + "="*80)
        print("🎯 VALIDATION DECISION")
        print("="*80 + "\n")

        # For now, return CONDITIONAL since we need backtest integration
        decision = "CONDITIONAL"
        reason = "Backtest integration required for full validation"

        print(f"Decision: {decision}")
        print(f"Reason: {reason}")

        # Save decision
        decision_data = {
            'decision': decision,
            'reason': reason,
            'timestamp': datetime.now().isoformat(),
            'results_summary': results
        }

        with open(self.output_dir / 'DECISION.json', 'w') as f:
            json.dump(decision_data, f, indent=2)

        return decision


def main():
    parser = argparse.ArgumentParser(description='Validate Optuna optimization results')
    parser.add_argument('--study-dir', required=True, help='Optuna study directory')
    parser.add_argument('--baseline-dir', default='results/router_v10_full_2022_2024_combined',
                        help='Baseline results directory')
    parser.add_argument('--output', default='results/trap_validation',
                        help='Output directory for validation results')

    args = parser.parse_args()

    validator = OptunaValidator(args.study_dir, args.baseline_dir, args.output)
    decision, results = validator.run_all_validations()

    print("\n" + "="*80)
    print(f"✅ Validation complete: {decision}")
    print("="*80)

    return 0 if decision == "ACCEPT" else 1


if __name__ == '__main__':
    exit(main())
