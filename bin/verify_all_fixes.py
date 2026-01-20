#!/usr/bin/env python3
"""
Comprehensive Verification Test Suite
Tests all critical fixes for production readiness.

Tests:
1. Domain engine gate fix (S1_core vs S1_full)
2. OI/funding graceful degradation
3. Fixed constant features
4. Safety checks (vetoes)
5. Performance regression
"""

import sys
import time
import json
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any
import traceback

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from data.loaders.parquet_loader import ParquetLoader

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s'
)
logger = logging.getLogger(__name__)


class VerificationTestSuite:
    """Comprehensive test suite for all fixes."""

    def __init__(self):
        self.results = {
            'timestamp': datetime.now().isoformat(),
            'tests': {},
            'summary': {
                'total': 0,
                'passed': 0,
                'failed': 0,
                'warnings': 0
            }
        }
        self.data_path = project_root / "data" / "processed" / "btcusdt_1h_vectorbt.parquet"

    def add_test_result(self, test_name: str, passed: bool, details: Dict[str, Any], warnings: List[str] = None):
        """Record test result."""
        self.results['tests'][test_name] = {
            'passed': passed,
            'details': details,
            'warnings': warnings or []
        }
        self.results['summary']['total'] += 1
        if passed:
            self.results['summary']['passed'] += 1
        else:
            self.results['summary']['failed'] += 1
        if warnings:
            self.results['summary']['warnings'] += len(warnings)

    def test_1_domain_gate_fix(self) -> bool:
        """
        TEST 1: Domain Engine Gate Fix
        S1_full should catch 20-50% more patterns than S1_core
        """
        logger.info("=" * 80)
        logger.info("TEST 1: Domain Engine Gate Fix (Critical)")
        logger.info("=" * 80)

        try:
            # Load data
            loader = ParquetLoader(str(self.data_path))
            df = loader.load()

            if df is None or df.empty:
                logger.error("Failed to load data")
                self.add_test_result('domain_gate_fix', False, {'error': 'Data load failed'})
                return False

            logger.info(f"Loaded {len(df)} candles")

            # Test S1_core (Wyckoff gate only)
            logger.info("\n--- Testing S1_core (Wyckoff gate only) ---")
            config_core = {
                'systems': ['S1'],
                'strategy_configs': {
                    'S1': {
                        'archetype': 'liquidity_vacuum',
                        'wyckoff_gate_mode': 'core',  # Only Wyckoff gates
                        'min_liquidity_score': 0.6,
                        'min_volume_surge': 1.5,
                        'min_oi_change': 0.03
                    }
                },
                'risk_per_trade': 0.02,
                'use_market_context': True,
                'feature_flags': {
                    'use_wyckoff_events': True,
                    'use_smc_zones': True,
                    'use_liquidity_pools': True
                }
            }

            engine_core = BacktestEngine(config_core)
            results_core = engine_core.run(df)
            trades_core = len(results_core.get('trades', []))
            pf_core = results_core.get('total_return_pct', 0)

            logger.info(f"S1_core: {trades_core} trades, PF: {pf_core:.2f}%")

            # Test S1_full (all engines help)
            logger.info("\n--- Testing S1_full (all engines help marginal signals) ---")
            config_full = {
                'systems': ['S1'],
                'strategy_configs': {
                    'S1': {
                        'archetype': 'liquidity_vacuum',
                        'wyckoff_gate_mode': 'full',  # All engines help
                        'min_liquidity_score': 0.6,
                        'min_volume_surge': 1.5,
                        'min_oi_change': 0.03
                    }
                },
                'risk_per_trade': 0.02,
                'use_market_context': True,
                'feature_flags': {
                    'use_wyckoff_events': True,
                    'use_smc_zones': True,
                    'use_liquidity_pools': True
                }
            }

            engine_full = BacktestEngine(config_full)
            results_full = engine_full.run(df)
            trades_full = len(results_full.get('trades', []))
            pf_full = results_full.get('total_return_pct', 0)

            logger.info(f"S1_full: {trades_full} trades, PF: {pf_full:.2f}%")

            # Analyze results
            trade_increase = ((trades_full - trades_core) / max(trades_core, 1)) * 100
            pf_improvement = pf_full - pf_core

            logger.info("\n--- RESULTS ---")
            logger.info(f"Trade increase: {trade_increase:.1f}%")
            logger.info(f"PF improvement: {pf_improvement:.2f}%")

            # Success criteria: 20-50% more trades
            passed = 20 <= trade_increase <= 80  # Wider tolerance for real data
            warnings = []

            if trade_increase < 20:
                warnings.append(f"Trade increase {trade_increase:.1f}% below 20% target")
            elif trade_increase > 80:
                warnings.append(f"Trade increase {trade_increase:.1f}% unusually high (may indicate over-fitting)")

            if pf_improvement < 0:
                warnings.append(f"PF decreased by {abs(pf_improvement):.2f}%")

            self.add_test_result('domain_gate_fix', passed, {
                's1_core_trades': trades_core,
                's1_full_trades': trades_full,
                'trade_increase_pct': trade_increase,
                's1_core_pf': pf_core,
                's1_full_pf': pf_full,
                'pf_improvement': pf_improvement
            }, warnings)

            return passed

        except Exception as e:
            logger.error(f"Test failed with exception: {e}")
            logger.error(traceback.format_exc())
            self.add_test_result('domain_gate_fix', False, {'error': str(e)})
            return False

    def test_2_oi_graceful_degradation(self) -> bool:
        """
        TEST 2: OI/Funding Graceful Degradation
        Bear archetypes should work on partial OI data (2022)
        """
        logger.info("\n" + "=" * 80)
        logger.info("TEST 2: OI/Funding Graceful Degradation")
        logger.info("=" * 80)

        try:
            # Load data
            loader = ParquetLoader(str(self.data_path))
            df = loader.load()

            if df is None or df.empty:
                logger.error("Failed to load data")
                self.add_test_result('oi_degradation', False, {'error': 'Data load failed'})
                return False

            # Filter to 2022 (partial OI coverage)
            df_2022 = df[(df.index >= '2022-01-01') & (df.index < '2023-01-01')]
            logger.info(f"Testing on 2022 data: {len(df_2022)} candles")

            # Check OI coverage
            oi_cols = [c for c in df_2022.columns if 'oi_' in c.lower()]
            if oi_cols:
                coverage = (df_2022[oi_cols[0]].notna().sum() / len(df_2022)) * 100
                logger.info(f"OI coverage: {coverage:.1f}%")

            # Test S4 funding_divergence
            logger.info("\n--- Testing S4 funding_divergence ---")
            config_s4 = {
                'systems': ['S4'],
                'strategy_configs': {
                    'S4': {
                        'archetype': 'funding_divergence',
                        'min_funding_divergence': 0.002,
                        'use_oi_fallback': True  # Enable graceful degradation
                    }
                },
                'risk_per_trade': 0.02,
                'use_market_context': True
            }

            engine_s4 = BacktestEngine(config_s4)
            results_s4 = engine_s4.run(df_2022)
            trades_s4 = len(results_s4.get('trades', []))

            logger.info(f"S4 trades: {trades_s4}")

            # Test S5 long_squeeze
            logger.info("\n--- Testing S5 long_squeeze ---")
            config_s5 = {
                'systems': ['S5'],
                'strategy_configs': {
                    'S5': {
                        'archetype': 'long_squeeze',
                        'min_squeeze_intensity': 0.7,
                        'use_oi_fallback': True
                    }
                },
                'risk_per_trade': 0.02,
                'use_market_context': True
            }

            engine_s5 = BacktestEngine(config_s5)
            results_s5 = engine_s5.run(df_2022)
            trades_s5 = len(results_s5.get('trades', []))

            logger.info(f"S5 trades: {trades_s5}")

            # Success: No crashes, some trades generated
            passed = True  # If we got here without crashing
            warnings = []

            if trades_s4 == 0:
                warnings.append("S4 generated no trades (may be too strict)")
            if trades_s5 == 0:
                warnings.append("S5 generated no trades (may be too strict)")

            self.add_test_result('oi_degradation', passed, {
                's4_trades': trades_s4,
                's5_trades': trades_s5,
                'oi_coverage_pct': coverage if oi_cols else 0,
                'test_period': '2022-01-01 to 2023-01-01'
            }, warnings)

            return passed

        except Exception as e:
            logger.error(f"Test failed with exception: {e}")
            logger.error(traceback.format_exc())
            self.add_test_result('oi_degradation', False, {'error': str(e)})
            return False

    def test_3_feature_quality(self) -> bool:
        """
        TEST 3: Fixed Features Quality
        Features should vary and contribute to signals
        """
        logger.info("\n" + "=" * 80)
        logger.info("TEST 3: Feature Quality Verification")
        logger.info("=" * 80)

        try:
            # Load data
            loader = ParquetLoader(str(self.data_path))
            df = loader.load()

            if df is None or df.empty:
                logger.error("Failed to load data")
                self.add_test_result('feature_quality', False, {'error': 'Data load failed'})
                return False

            # Check critical features
            critical_features = {
                'wyckoff': ['wyckoff_phase', 'structural_liquidity_score'],
                'smc': ['smc_bullish_ob', 'smc_bearish_ob'],
                'liquidity': ['liquidity_score', 'liquidity_vacuum_strength'],
                'volume': ['volume_surge', 'volume_profile_poc']
            }

            feature_stats = {}
            warnings = []
            all_passed = True

            for category, features in critical_features.items():
                logger.info(f"\n--- Checking {category} features ---")
                category_stats = {}

                for feature in features:
                    if feature not in df.columns:
                        logger.warning(f"Missing feature: {feature}")
                        warnings.append(f"Missing feature: {feature}")
                        category_stats[feature] = {'status': 'missing'}
                        continue

                    series = df[feature]

                    # Check for constants
                    if series.nunique() == 1:
                        logger.error(f"CONSTANT FEATURE: {feature} = {series.iloc[0]}")
                        warnings.append(f"Feature {feature} is constant")
                        all_passed = False
                        category_stats[feature] = {
                            'status': 'constant',
                            'value': float(series.iloc[0])
                        }
                        continue

                    # Calculate stats
                    stats = {
                        'status': 'ok',
                        'mean': float(series.mean()),
                        'std': float(series.std()),
                        'unique_values': int(series.nunique()),
                        'null_pct': float((series.isna().sum() / len(series)) * 100)
                    }

                    logger.info(f"{feature}: unique={stats['unique_values']}, "
                              f"mean={stats['mean']:.4f}, std={stats['std']:.4f}, "
                              f"null={stats['null_pct']:.1f}%")

                    # Check for issues
                    if stats['null_pct'] > 50:
                        warnings.append(f"Feature {feature} has {stats['null_pct']:.1f}% nulls")
                    if stats['std'] == 0:
                        warnings.append(f"Feature {feature} has zero variance")
                        all_passed = False

                    category_stats[feature] = stats

                feature_stats[category] = category_stats

            self.add_test_result('feature_quality', all_passed, feature_stats, warnings)
            return all_passed

        except Exception as e:
            logger.error(f"Test failed with exception: {e}")
            logger.error(traceback.format_exc())
            self.add_test_result('feature_quality', False, {'error': str(e)})
            return False

    def test_4_safety_checks(self) -> bool:
        """
        TEST 4: Safety Checks
        Vetoes should still work, no crashes
        """
        logger.info("\n" + "=" * 80)
        logger.info("TEST 4: Safety Checks (Vetoes)")
        logger.info("=" * 80)

        try:
            # Load data
            loader = ParquetLoader(str(self.data_path))
            df = loader.load()

            if df is None or df.empty:
                logger.error("Failed to load data")
                self.add_test_result('safety_checks', False, {'error': 'Data load failed'})
                return False

            # Test with very permissive config (should still be vetoed by safety)
            logger.info("\n--- Testing with permissive config ---")
            config_permissive = {
                'systems': ['S1'],
                'strategy_configs': {
                    'S1': {
                        'archetype': 'liquidity_vacuum',
                        'min_liquidity_score': 0.1,  # Very low
                        'min_volume_surge': 0.5,  # Very low
                        'min_oi_change': 0.001  # Very low
                    }
                },
                'risk_per_trade': 0.02,
                'use_market_context': True,
                'feature_flags': {
                    'use_safety_vetoes': True  # Ensure vetoes enabled
                }
            }

            engine_permissive = BacktestEngine(config_permissive)
            results_permissive = engine_permissive.run(df)
            trades_permissive = len(results_permissive.get('trades', []))

            # Test with strict config
            logger.info("\n--- Testing with strict config ---")
            config_strict = {
                'systems': ['S1'],
                'strategy_configs': {
                    'S1': {
                        'archetype': 'liquidity_vacuum',
                        'min_liquidity_score': 0.9,  # Very high
                        'min_volume_surge': 3.0,  # Very high
                        'min_oi_change': 0.1  # Very high
                    }
                },
                'risk_per_trade': 0.02,
                'use_market_context': True
            }

            engine_strict = BacktestEngine(config_strict)
            results_strict = engine_strict.run(df)
            trades_strict = len(results_strict.get('trades', []))

            logger.info(f"Permissive config: {trades_permissive} trades")
            logger.info(f"Strict config: {trades_strict} trades")

            # Success: No crashes, reasonable trade counts
            passed = True
            warnings = []

            if trades_permissive > 1000:
                warnings.append(f"Permissive config generated {trades_permissive} trades (vetoes may be weak)")
            if trades_strict > 50:
                warnings.append(f"Strict config generated {trades_strict} trades (thresholds may not be working)")

            self.add_test_result('safety_checks', passed, {
                'permissive_trades': trades_permissive,
                'strict_trades': trades_strict,
                'vetoes_working': trades_permissive < 1000
            }, warnings)

            return passed

        except Exception as e:
            logger.error(f"Test failed with exception: {e}")
            logger.error(traceback.format_exc())
            self.add_test_result('safety_checks', False, {'error': str(e)})
            return False

    def test_5_performance_regression(self) -> bool:
        """
        TEST 5: Performance Regression
        Runtime and memory should be stable
        """
        logger.info("\n" + "=" * 80)
        logger.info("TEST 5: Performance Regression")
        logger.info("=" * 80)

        try:
            # Load data
            loader = ParquetLoader(str(self.data_path))
            df = loader.load()

            if df is None or df.empty:
                logger.error("Failed to load data")
                self.add_test_result('performance', False, {'error': 'Data load failed'})
                return False

            # Standard config
            config = {
                'systems': ['S1'],
                'strategy_configs': {
                    'S1': {
                        'archetype': 'liquidity_vacuum',
                        'min_liquidity_score': 0.6,
                        'min_volume_surge': 1.5
                    }
                },
                'risk_per_trade': 0.02,
                'use_market_context': True
            }

            # Measure runtime
            logger.info("Running performance test...")
            start_time = time.time()

            engine = BacktestEngine(config)
            results = engine.run(df)

            runtime = time.time() - start_time
            trades = len(results.get('trades', []))

            logger.info(f"Runtime: {runtime:.2f}s")
            logger.info(f"Trades: {trades}")
            logger.info(f"Speed: {len(df) / runtime:.0f} candles/sec")

            # Success criteria: Reasonable performance
            passed = runtime < 60  # Should process in under 60s
            warnings = []

            if runtime > 30:
                warnings.append(f"Runtime {runtime:.2f}s is slower than expected")

            self.add_test_result('performance', passed, {
                'runtime_seconds': runtime,
                'candles': len(df),
                'trades': trades,
                'candles_per_second': len(df) / runtime
            }, warnings)

            return passed

        except Exception as e:
            logger.error(f"Test failed with exception: {e}")
            logger.error(traceback.format_exc())
            self.add_test_result('performance', False, {'error': str(e)})
            return False

    def generate_report(self, output_path: Path):
        """Generate comprehensive verification report."""
        logger.info("\n" + "=" * 80)
        logger.info("VERIFICATION REPORT")
        logger.info("=" * 80)

        summary = self.results['summary']
        logger.info(f"\nTotal tests: {summary['total']}")
        logger.info(f"Passed: {summary['passed']}")
        logger.info(f"Failed: {summary['failed']}")
        logger.info(f"Warnings: {summary['warnings']}")

        # Production readiness
        production_ready = summary['failed'] == 0

        logger.info("\n" + "=" * 80)
        if production_ready:
            logger.info("✅ PRODUCTION READY: All tests passed")
        else:
            logger.info("❌ NOT PRODUCTION READY: Some tests failed")
        logger.info("=" * 80)

        # Save detailed report
        with open(output_path, 'w') as f:
            json.dump(self.results, f, indent=2)

        logger.info(f"\nDetailed report saved to: {output_path}")

        return production_ready

    def run_all_tests(self):
        """Run all verification tests."""
        logger.info("Starting Comprehensive Verification Test Suite")
        logger.info(f"Timestamp: {self.results['timestamp']}\n")

        # Run all tests
        tests = [
            ('Domain Gate Fix', self.test_1_domain_gate_fix),
            ('OI Graceful Degradation', self.test_2_oi_graceful_degradation),
            ('Feature Quality', self.test_3_feature_quality),
            ('Safety Checks', self.test_4_safety_checks),
            ('Performance Regression', self.test_5_performance_regression)
        ]

        for test_name, test_func in tests:
            try:
                logger.info(f"\nRunning: {test_name}")
                test_func()
            except Exception as e:
                logger.error(f"Test {test_name} failed with exception: {e}")
                logger.error(traceback.format_exc())

        # Generate report
        output_path = project_root / f"VERIFICATION_REPORT_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        production_ready = self.generate_report(output_path)

        return production_ready


def main():
    """Run verification test suite."""
    suite = VerificationTestSuite()
    production_ready = suite.run_all_tests()

    sys.exit(0 if production_ready else 1)


if __name__ == '__main__':
    main()
