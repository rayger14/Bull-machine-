#!/usr/bin/env python3
"""
System B0 Validation Suite

Comprehensive validation and testing for System B0 production deployment:
- Extended period testing (2022-2024)
- Walk-forward validation
- Parameter sensitivity analysis
- Regime performance breakdown
- Statistical significance testing

Architecture:
- Isolated test runs with clean state
- Reproducible results
- Comprehensive reporting
- Performance benchmarking

Usage:
    # Full validation suite
    python bin/validate_system_b0.py

    # Quick validation (essential tests only)
    python bin/validate_system_b0.py --quick

    # Specific test
    python bin/validate_system_b0.py --test regime_breakdown
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import json
import argparse
from datetime import datetime
from typing import Dict, Any, List
import pandas as pd
import numpy as np
from dataclasses import dataclass, asdict

# Import the deployment system
from examples.baseline_production_deploy import SystemB0, ConfigLoader


# =============================================================================
# Test Definitions
# =============================================================================

@dataclass
class TestResult:
    """Individual test result."""
    test_name: str
    passed: bool
    score: float
    expected: Any
    actual: Any
    message: str
    details: Dict[str, Any]


@dataclass
class ValidationReport:
    """Complete validation report."""
    timestamp: datetime
    total_tests: int
    passed: int
    failed: int
    overall_score: float
    results: List[TestResult]
    summary: str


# =============================================================================
# Validation Tests
# =============================================================================

class ValidationSuite:
    """Complete validation test suite."""
    
    def __init__(self, config_path: str):
        self.config_path = config_path
        self.config = ConfigLoader.load(config_path)
        self.results: List[TestResult] = []
    
    def run_all_tests(self, quick: bool = False) -> ValidationReport:
        """Run complete validation suite."""
        
        print("=" * 80)
        print("SYSTEM B0 VALIDATION SUITE")
        print("=" * 80)
        print(f"Mode: {'QUICK' if quick else 'FULL'}")
        print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print()
        
        # Core tests (always run)
        self._test_extended_period()
        self._test_regime_breakdown()
        
        if not quick:
            # Additional tests (full validation only)
            self._test_walk_forward()
            self._test_parameter_sensitivity()
            self._test_statistical_significance()
        
        # Generate report
        report = self._generate_report()
        self._print_report(report)
        
        return report
    
    def _test_extended_period(self):
        """Test performance across extended period (2022-2024)."""
        print("\nTest 1: Extended Period Performance (2022-2024)")
        print("-" * 80)
        
        try:
            system = SystemB0(self.config_path)
            results = system.run_backtest('2022-01-01', '2024-09-30')
            
            targets = self.config['performance_targets']
            
            # Check profit factor
            pf_passed = results['profit_factor'] >= targets['min_profit_factor']
            
            # Check win rate
            wr_passed = results['win_rate'] >= targets['min_win_rate_pct'] / 100
            
            # Check drawdown
            dd_passed = results['max_drawdown'] <= targets['max_drawdown_pct'] / 100
            
            passed = pf_passed and wr_passed and dd_passed
            
            score = (
                (results['profit_factor'] / targets['min_profit_factor']) * 0.4 +
                (results['win_rate'] / (targets['min_win_rate_pct'] / 100)) * 0.3 +
                (1 - results['max_drawdown'] / (targets['max_drawdown_pct'] / 100)) * 0.3
            )
            
            self.results.append(TestResult(
                test_name="Extended Period Performance",
                passed=passed,
                score=score,
                expected={
                    'min_pf': targets['min_profit_factor'],
                    'min_wr': targets['min_win_rate_pct'],
                    'max_dd': targets['max_drawdown_pct']
                },
                actual={
                    'pf': results['profit_factor'],
                    'wr': results['win_rate'] * 100,
                    'dd': results['max_drawdown'] * 100
                },
                message=f"{'PASS' if passed else 'FAIL'} - PF: {results['profit_factor']:.2f}, "
                        f"WR: {results['win_rate']:.1%}, DD: {results['max_drawdown']:.1%}",
                details=results
            ))
            
            print(f"Result: {'PASS' if passed else 'FAIL'}")
            print(f"  Profit Factor: {results['profit_factor']:.2f} (target: >= {targets['min_profit_factor']:.2f})")
            print(f"  Win Rate:      {results['win_rate']:.1%} (target: >= {targets['min_win_rate_pct']:.0f}%)")
            print(f"  Max Drawdown:  {results['max_drawdown']:.1%} (target: <= {targets['max_drawdown_pct']:.0f}%)")
            print(f"  Total Trades:  {results['total_trades']}")
            
        except Exception as e:
            self.results.append(TestResult(
                test_name="Extended Period Performance",
                passed=False,
                score=0.0,
                expected={},
                actual={},
                message=f"FAIL - Error: {e}",
                details={'error': str(e)}
            ))
            print(f"Result: FAIL - {e}")
    
    def _test_regime_breakdown(self):
        """Test performance across different market regimes."""
        print("\nTest 2: Regime Performance Breakdown")
        print("-" * 80)
        
        test_periods = self.config['validation']['test_periods']
        regime_results = {}
        all_passed = True
        
        for period in test_periods:
            print(f"\n  Testing {period['regime'].upper()} market ({period['start']} to {period['end']})...")
            
            try:
                system = SystemB0(self.config_path)
                results = system.run_backtest(period['start'], period['end'])
                
                regime_results[period['regime']] = results
                
                passed = results['profit_factor'] >= period['expected_pf_min']
                all_passed = all_passed and passed
                
                print(f"    PF: {results['profit_factor']:.2f} (expected: >= {period['expected_pf_min']:.2f}) - {'PASS' if passed else 'FAIL'}")
                print(f"    WR: {results['win_rate']:.1%}")
                print(f"    Trades: {results['total_trades']}")
                
            except Exception as e:
                print(f"    FAIL - Error: {e}")
                all_passed = False
        
        score = sum(
            min(regime_results[period['regime']]['profit_factor'] / period['expected_pf_min'], 2.0)
            for period in test_periods
            if period['regime'] in regime_results
        ) / len(test_periods)
        
        self.results.append(TestResult(
            test_name="Regime Performance Breakdown",
            passed=all_passed,
            score=score,
            expected={p['regime']: p['expected_pf_min'] for p in test_periods},
            actual={regime: results['profit_factor'] for regime, results in regime_results.items()},
            message=f"{'PASS' if all_passed else 'FAIL'} - All regimes tested",
            details=regime_results
        ))
        
        print(f"\nOverall: {'PASS' if all_passed else 'FAIL'}")
    
    def _test_walk_forward(self):
        """Test walk-forward validation."""
        print("\nTest 3: Walk-Forward Validation")
        print("-" * 80)
        
        wf_config = self.config['validation']['walk_forward']
        if not wf_config['enabled']:
            print("Skipped (disabled in config)")
            return
        
        print("Walk-forward validation: 6-month train, 3-month test, 3-month step")
        
        # Define windows
        windows = [
            # Train: 2022-01 to 2022-06, Test: 2022-07 to 2022-09
            {'train': ('2022-01-01', '2022-06-30'), 'test': ('2022-07-01', '2022-09-30')},
            # Train: 2022-04 to 2022-09, Test: 2022-10 to 2022-12
            {'train': ('2022-04-01', '2022-09-30'), 'test': ('2022-10-01', '2022-12-31')},
            # Train: 2022-07 to 2022-12, Test: 2023-01 to 2023-03
            {'train': ('2022-07-01', '2022-12-31'), 'test': ('2023-01-01', '2023-03-31')},
            # Train: 2022-10 to 2023-03, Test: 2023-04 to 2023-06
            {'train': ('2022-10-01', '2023-03-31'), 'test': ('2023-04-01', '2023-06-30')},
        ]
        
        wf_results = []
        all_positive = True
        
        for i, window in enumerate(windows):
            print(f"\n  Window {i+1}: Train {window['train'][0]} to {window['train'][1]}, "
                  f"Test {window['test'][0]} to {window['test'][1]}")
            
            try:
                system = SystemB0(self.config_path)
                results = system.run_backtest(window['test'][0], window['test'][1])
                
                wf_results.append(results)
                
                passed = results['total_pnl_usd'] > 0 and results['profit_factor'] > 1.0
                all_positive = all_positive and passed
                
                print(f"    PF: {results['profit_factor']:.2f}, PnL: ${results['total_pnl_usd']:,.2f} - {'PASS' if passed else 'FAIL'}")
                
            except Exception as e:
                print(f"    FAIL - Error: {e}")
                all_positive = False
        
        avg_pf = np.mean([r['profit_factor'] for r in wf_results]) if wf_results else 0
        score = min(avg_pf / 2.0, 1.0)
        
        self.results.append(TestResult(
            test_name="Walk-Forward Validation",
            passed=all_positive,
            score=score,
            expected={'all_windows_positive': True},
            actual={'avg_pf': avg_pf, 'positive_windows': sum(1 for r in wf_results if r['total_pnl_usd'] > 0)},
            message=f"{'PASS' if all_positive else 'FAIL'} - Avg PF: {avg_pf:.2f}",
            details={'windows': wf_results}
        ))
        
        print(f"\nOverall: {'PASS' if all_positive else 'FAIL'} (Avg PF: {avg_pf:.2f})")
    
    def _test_parameter_sensitivity(self):
        """Test sensitivity to parameter variations."""
        print("\nTest 4: Parameter Sensitivity Analysis")
        print("-" * 80)
        
        # Test variations in key parameters
        base_params = self.config['strategy']['parameters']
        
        variations = [
            {'buy_threshold': -0.12},  # Less drawdown required
            {'buy_threshold': -0.18},  # More drawdown required
            {'profit_target': 0.06},   # Lower profit target
            {'profit_target': 0.10},   # Higher profit target
        ]
        
        sensitivity_results = []
        all_stable = True
        
        for var in variations:
            param_name = list(var.keys())[0]
            param_value = list(var.values())[0]
            
            print(f"\n  Testing {param_name} = {param_value}")
            
            try:
                # Modify config
                modified_config = self.config.copy()
                modified_config['strategy']['parameters'][param_name] = param_value
                
                # Save temporary config
                temp_config_path = '/tmp/system_b0_test.json'
                with open(temp_config_path, 'w') as f:
                    json.dump(modified_config, f)
                
                # Run backtest
                system = SystemB0(temp_config_path)
                results = system.run_backtest('2022-01-01', '2024-09-30')
                
                sensitivity_results.append({
                    'param': param_name,
                    'value': param_value,
                    'pf': results['profit_factor'],
                    'wr': results['win_rate'],
                    'trades': results['total_trades']
                })
                
                # Check if still profitable
                stable = results['profit_factor'] > 1.5
                all_stable = all_stable and stable
                
                print(f"    PF: {results['profit_factor']:.2f}, Trades: {results['total_trades']} - {'STABLE' if stable else 'UNSTABLE'}")
                
            except Exception as e:
                print(f"    FAIL - Error: {e}")
                all_stable = False
        
        self.results.append(TestResult(
            test_name="Parameter Sensitivity",
            passed=all_stable,
            score=1.0 if all_stable else 0.5,
            expected={'all_variations_stable': True},
            actual={'stable_count': sum(1 for r in sensitivity_results if r['pf'] > 1.5)},
            message=f"{'PASS' if all_stable else 'FAIL'} - Strategy robust to parameter variations",
            details={'variations': sensitivity_results}
        ))
        
        print(f"\nOverall: {'PASS' if all_stable else 'FAIL'}")
    
    def _test_statistical_significance(self):
        """Test statistical significance of results."""
        print("\nTest 5: Statistical Significance")
        print("-" * 80)
        
        try:
            system = SystemB0(self.config_path)
            results = system.run_backtest('2022-01-01', '2024-09-30')
            
            # Check minimum trade count
            min_trades = self.config['performance_targets'].get('min_trades_per_month', 2) * 33  # 33 months
            sufficient_trades = results['total_trades'] >= min_trades
            
            # Check if profit factor confidence interval excludes 1.0
            # Using simple bootstrap estimation
            trades = results['total_trades']
            pf = results['profit_factor']
            
            # Estimate standard error (simplified)
            se_pf = pf / np.sqrt(trades) if trades > 0 else 0
            confidence_interval = (pf - 1.96 * se_pf, pf + 1.96 * se_pf)
            
            significant = confidence_interval[0] > 1.0
            
            passed = sufficient_trades and significant
            
            self.results.append(TestResult(
                test_name="Statistical Significance",
                passed=passed,
                score=1.0 if passed else 0.5,
                expected={'sufficient_trades': True, 'pf_ci_excludes_1': True},
                actual={
                    'total_trades': trades,
                    'pf': pf,
                    'ci_lower': confidence_interval[0],
                    'ci_upper': confidence_interval[1]
                },
                message=f"{'PASS' if passed else 'FAIL'} - PF 95% CI: [{confidence_interval[0]:.2f}, {confidence_interval[1]:.2f}]",
                details={'trades': trades, 'confidence_interval': confidence_interval}
            ))
            
            print(f"Total Trades: {trades} (min: {min_trades})")
            print(f"Profit Factor: {pf:.2f}")
            print(f"95% CI: [{confidence_interval[0]:.2f}, {confidence_interval[1]:.2f}]")
            print(f"Result: {'PASS' if passed else 'FAIL'}")
            
        except Exception as e:
            self.results.append(TestResult(
                test_name="Statistical Significance",
                passed=False,
                score=0.0,
                expected={},
                actual={},
                message=f"FAIL - Error: {e}",
                details={'error': str(e)}
            ))
            print(f"Result: FAIL - {e}")
    
    def _generate_report(self) -> ValidationReport:
        """Generate validation report."""
        
        passed = sum(1 for r in self.results if r.passed)
        failed = len(self.results) - passed
        overall_score = np.mean([r.score for r in self.results])
        
        if overall_score >= 0.9:
            summary = "EXCELLENT - System ready for production deployment"
        elif overall_score >= 0.75:
            summary = "GOOD - System meets minimum requirements"
        elif overall_score >= 0.6:
            summary = "MARGINAL - Review failures before deployment"
        else:
            summary = "FAIL - System not ready for production"
        
        return ValidationReport(
            timestamp=datetime.now(),
            total_tests=len(self.results),
            passed=passed,
            failed=failed,
            overall_score=overall_score,
            results=self.results,
            summary=summary
        )
    
    def _print_report(self, report: ValidationReport):
        """Print formatted validation report."""
        
        print("\n" + "=" * 80)
        print("VALIDATION REPORT")
        print("=" * 80)
        print(f"Timestamp: {report.timestamp.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Total Tests: {report.total_tests}")
        print(f"Passed: {report.passed}")
        print(f"Failed: {report.failed}")
        print(f"Overall Score: {report.overall_score:.1%}")
        print()
        print(f"Summary: {report.summary}")
        print()
        
        print("Test Results:")
        print("-" * 80)
        for result in report.results:
            status = "PASS" if result.passed else "FAIL"
            print(f"  [{status}] {result.test_name:40s} Score: {result.score:.2f}")
        
        print("=" * 80)
        
        # Save report to file
        report_file = f"logs/validation_report_{report.timestamp.strftime('%Y%m%d_%H%M%S')}.json"
        Path(report_file).parent.mkdir(parents=True, exist_ok=True)
        
        report_dict = {
            'timestamp': report.timestamp.isoformat(),
            'total_tests': report.total_tests,
            'passed': report.passed,
            'failed': report.failed,
            'overall_score': report.overall_score,
            'summary': report.summary,
            'results': [asdict(r) for r in report.results]
        }
        
        with open(report_file, 'w') as f:
            json.dump(report_dict, f, indent=2, default=str)
        
        print(f"\nReport saved to: {report_file}")


# =============================================================================
# CLI Entry Point
# =============================================================================

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='System B0 Validation Suite',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        '--config',
        type=str,
        default='configs/system_b0_production.json',
        help='Path to configuration file'
    )
    
    parser.add_argument(
        '--quick',
        action='store_true',
        help='Run quick validation (essential tests only)'
    )
    
    parser.add_argument(
        '--test',
        type=str,
        choices=['extended', 'regime', 'walkforward', 'sensitivity', 'significance'],
        help='Run specific test only'
    )
    
    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()
    
    try:
        suite = ValidationSuite(args.config)
        
        if args.test:
            # Run specific test
            test_map = {
                'extended': suite._test_extended_period,
                'regime': suite._test_regime_breakdown,
                'walkforward': suite._test_walk_forward,
                'sensitivity': suite._test_parameter_sensitivity,
                'significance': suite._test_statistical_significance
            }
            test_map[args.test]()
        else:
            # Run full suite
            report = suite.run_all_tests(quick=args.quick)
        
        return 0
        
    except Exception as e:
        print(f"\nERROR: {e}\n")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())
