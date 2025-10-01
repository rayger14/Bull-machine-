#!/usr/bin/env python3
"""
Tiered Testing Framework for Bull Machine v1.7
Battle-tested optimization strategy with fast→slow validation
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import pandas as pd
import numpy as np
import json
import time
import multiprocessing as mp
from datetime import datetime, timedelta
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

from engine.io.tradingview_loader import load_tv, RealDataRequiredError

class TieredTester:
    def __init__(self, config_base='configs/v170/assets/ETH_v17_tuned.json'):
        """Initialize tiered testing framework"""
        self.config_base = config_base
        self.results_dir = Path('results/tiered_tests')
        self.results_dir.mkdir(parents=True, exist_ok=True)

        print("🚀 BULL MACHINE v1.7 TIERED TESTING FRAMEWORK")
        print("=" * 60)

    def tier0_preflight(self, assets=['ETH_4H', 'ETH_1D'], timeout=30):
        """
        Tier 0: Preflight checks (seconds)
        - Data presence, NaNs, flat volume
        - Indicator caches warm
        - Fail fast on basic issues
        """
        print(f"\n🔍 TIER 0: PREFLIGHT CHECKS")
        print("-" * 40)

        start_time = time.time()
        results = {'status': 'pass', 'issues': [], 'assets_checked': {}}

        for asset in assets:
            try:
                print(f"   Checking {asset}...")

                # Load data with timeout
                data = load_tv(asset)

                # Basic data quality checks
                import pandas as pd

                # Handle timezone-aware comparisons
                now = pd.Timestamp.now(tz='UTC')
                last_bar_time = data.index[-1]
                if last_bar_time.tz is None:
                    last_bar_time = last_bar_time.tz_localize('UTC')

                days_since_last = (now - last_bar_time).days

                checks = {
                    'data_loaded': len(data) > 0,
                    'recent_data': days_since_last < 30,
                    'no_nans': not data[['open', 'high', 'low', 'close']].isna().any().any(),
                    'volume_present': 'volume' in data.columns,
                    'price_variance': data['close'].std() > 0.01,
                    'volume_variance': data['volume'].std() > 0 if 'volume' in data.columns else True,
                    'min_bars': len(data) >= 1000
                }

                # Log issues
                failed_checks = [k for k, v in checks.items() if not v]
                if failed_checks:
                    results['issues'].extend([f"{asset}: {check}" for check in failed_checks])
                    results['status'] = 'warn' if results['status'] == 'pass' else 'fail'

                results['assets_checked'][asset] = {
                    'bars': len(data),
                    'date_range': f"{data.index[0]} to {data.index[-1]}",
                    'checks_passed': len(checks) - len(failed_checks),
                    'total_checks': len(checks)
                }

                # Timeout check
                if time.time() - start_time > timeout:
                    results['issues'].append(f"Timeout exceeded ({timeout}s)")
                    results['status'] = 'fail'
                    break

            except RealDataRequiredError as e:
                results['issues'].append(f"{asset}: Missing data file")
                results['status'] = 'fail'
            except Exception as e:
                results['issues'].append(f"{asset}: {str(e)}")
                results['status'] = 'fail'

        duration = time.time() - start_time
        print(f"   ✅ Preflight completed in {duration:.1f}s")

        if results['status'] == 'pass':
            print(f"   🎯 All {len(assets)} assets passed preflight")
        else:
            print(f"   ⚠️  Issues found: {len(results['issues'])}")
            for issue in results['issues'][:3]:  # Show first 3
                print(f"      - {issue}")

        return results

    def tier1_smoke_slice(self, config, asset='ETH_4H', start_date='2025-07-01',
                          end_date='2025-09-01', timeout=600):
        """
        Tier 1: Smoke slice (≤60-90 days)
        - One asset × 4H + 1D HTF
        - Validates plumbing, health bands, trade plumbing
        """
        print(f"\n🧪 TIER 1: SMOKE SLICE TEST")
        print("-" * 40)
        print(f"   Asset: {asset}")
        print(f"   Period: {start_date} to {end_date}")

        start_time = time.time()

        try:
            # Load config
            if isinstance(config, str):
                with open(config, 'r') as f:
                    config_dict = json.load(f)
            else:
                config_dict = config

            # Load data
            primary_data = load_tv(asset)
            htf_data = load_tv(asset.replace('4H', '1D'))

            # Slice to test period
            start_dt = pd.to_datetime(start_date)
            end_dt = pd.to_datetime(end_date)

            test_data = primary_data[(primary_data.index >= start_dt) &
                                   (primary_data.index <= end_dt)]

            if len(test_data) < 50:
                return {
                    'status': 'fail',
                    'error': f'Insufficient data: {len(test_data)} bars',
                    'duration': time.time() - start_time
                }

            # Run smoke test
            from scripts.smoke_backtest import SmokeBacktest
            smoke = SmokeBacktest(config_dict)
            results = smoke.run(test_data, htf_data, timeout=timeout)

            # Health band checks
            health_check = self._check_health_bands(results, config_dict)

            results.update({
                'health_bands': health_check,
                'duration': time.time() - start_time,
                'bars_tested': len(test_data)
            })

            print(f"   ✅ Smoke test completed in {results['duration']:.1f}s")
            print(f"   📊 Return: {results.get('total_return', 0):+.2f}%")
            print(f"   🎯 Trades: {results.get('total_trades', 0)}")

            return results

        except Exception as e:
            return {
                'status': 'fail',
                'error': str(e),
                'duration': time.time() - start_time
            }

    def tier2_walk_forward(self, config, windows, timeout=1200):
        """
        Tier 2: Short walk-forward (2-3× 60-90d windows)
        - Confirms same config works on adjacent periods
        """
        print(f"\n🚶 TIER 2: WALK-FORWARD VALIDATION")
        print("-" * 40)

        results = []

        for i, (start, end) in enumerate(windows):
            print(f"   Window {i+1}: {start} to {end}")

            window_result = self.tier1_smoke_slice(
                config, start_date=start, end_date=end, timeout=timeout//len(windows)
            )

            window_result['window'] = i+1
            window_result['period'] = f"{start} to {end}"
            results.append(window_result)

            # Early abort if window fails
            if window_result['status'] == 'fail':
                print(f"   ❌ Window {i+1} failed: {window_result.get('error', 'Unknown')}")
                break

        # Aggregate results
        passed_windows = len([r for r in results if r['status'] == 'pass'])

        aggregate = {
            'status': 'pass' if passed_windows == len(windows) else 'fail',
            'windows_passed': passed_windows,
            'total_windows': len(windows),
            'individual_results': results
        }

        if aggregate['status'] == 'pass':
            returns = [r.get('total_return', 0) for r in results if 'total_return' in r]
            aggregate['avg_return'] = np.mean(returns) if returns else 0
            aggregate['return_consistency'] = np.std(returns) if len(returns) > 1 else 0

        print(f"   📊 Walk-forward: {passed_windows}/{len(windows)} windows passed")

        return aggregate

    def tier3_full_backtest(self, config, assets=['ETH_4H'], months=18, timeout=1800):
        """
        Tier 3: Full backtest (12-24m)
        - Only for top 1-3 configs that survived Tiers 1-2
        """
        print(f"\n🏁 TIER 3: FULL BACKTEST")
        print("-" * 40)
        print(f"   Assets: {assets}")
        print(f"   Duration: {months} months")

        start_time = time.time()

        try:
            # Calculate date range
            end_date = datetime.now()
            start_date = end_date - timedelta(days=months * 30)

            results = {}

            for asset in assets:
                print(f"   Testing {asset}...")

                asset_result = self._run_full_asset_backtest(
                    config, asset, start_date, end_date, timeout//len(assets)
                )

                results[asset] = asset_result

                # Early abort on failure
                if asset_result['status'] == 'fail':
                    break

            # Aggregate multi-asset results
            passed_assets = len([r for r in results.values() if r['status'] == 'pass'])

            aggregate = {
                'status': 'pass' if passed_assets == len(assets) else 'fail',
                'assets_passed': passed_assets,
                'total_assets': len(assets),
                'duration': time.time() - start_time,
                'individual_results': results
            }

            if aggregate['status'] == 'pass':
                returns = [r.get('total_return', 0) for r in results.values() if 'total_return' in r]
                aggregate['portfolio_return'] = np.mean(returns) if returns else 0

            print(f"   📊 Full backtest: {passed_assets}/{len(assets)} assets passed")

            return aggregate

        except Exception as e:
            return {
                'status': 'fail',
                'error': str(e),
                'duration': time.time() - start_time
            }

    def _check_health_bands(self, results, config):
        """Check health bands against thresholds"""
        health_bands = {
            'macro_veto_rate': results.get('macro_veto_rate', 0),
            'smc_2hit_rate': results.get('smc_2hit_rate', 0),
            'hob_relevance': results.get('hob_relevance', 0),
            'delta_breaches': results.get('delta_breaches', 0)
        }

        # Expected ranges from calibration
        thresholds = {
            'macro_veto_rate': (0.05, 0.15),  # 5-15%
            'smc_2hit_rate': (0.30, 1.0),     # ≥30%
            'hob_relevance': (0.0, 0.30),     # ≤30%
            'delta_breaches': (0, 0)          # No breaches
        }

        checks = {}
        for metric, value in health_bands.items():
            min_val, max_val = thresholds[metric]
            checks[metric] = {
                'value': value,
                'threshold': thresholds[metric],
                'pass': min_val <= value <= max_val
            }

        return checks

    def _run_full_asset_backtest(self, config, asset, start_date, end_date, timeout):
        """Run full backtest for single asset"""
        try:
            # This would integrate with existing backtest infrastructure
            from run_performance_analysis import run_performance_analysis

            results = run_performance_analysis(
                config=config,
                asset=asset,
                start_date=start_date,
                end_date=end_date,
                timeout=timeout
            )

            return results

        except Exception as e:
            return {
                'status': 'fail',
                'error': str(e)
            }

    def budgeted_search(self, search_space, max_configs=40, keep_ratio=0.25):
        """
        Budgeted search using successive halving
        - Evaluate many configs on smoke slice
        - Keep top N% for walk-forward
        - Only finalists get full backtest
        """
        print(f"\n🔬 BUDGETED CONFIG SEARCH")
        print("=" * 50)
        print(f"   Initial configs: {max_configs}")
        print(f"   Keep ratio: {keep_ratio}")

        # Generate candidate configs
        candidates = self._sample_configs(search_space, max_configs)
        print(f"   📊 Generated {len(candidates)} candidate configs")

        # Tier 1: Smoke slice on all candidates
        print(f"\n   🧪 Tier 1: Smoke slice evaluation...")
        smoke_results = []

        for i, config in enumerate(candidates):
            print(f"      Config {i+1}/{len(candidates)}")
            result = self.tier1_smoke_slice(config)
            result['config_id'] = i
            smoke_results.append(result)

        # Keep top performers
        valid_results = [r for r in smoke_results if r['status'] == 'pass']
        if not valid_results:
            return {'status': 'fail', 'error': 'No configs passed smoke test'}

        # Sort by return and keep top %
        valid_results.sort(key=lambda x: x.get('total_return', 0), reverse=True)
        keep_count = max(1, int(len(valid_results) * keep_ratio))
        top_configs = valid_results[:keep_count]

        print(f"   📈 Top {keep_count} configs selected for walk-forward")

        # Tier 2: Walk-forward on survivors
        print(f"\n   🚶 Tier 2: Walk-forward validation...")
        walk_forward_windows = [
            ('2025-05-01', '2025-06-15'),
            ('2025-07-01', '2025-08-15'),
            ('2025-08-15', '2025-09-30')
        ]

        finalists = []
        for result in top_configs:
            config = candidates[result['config_id']]
            wf_result = self.tier2_walk_forward(config, walk_forward_windows)

            if wf_result['status'] == 'pass':
                finalists.append({
                    'config': config,
                    'config_id': result['config_id'],
                    'smoke_return': result.get('total_return', 0),
                    'wf_avg_return': wf_result.get('avg_return', 0),
                    'wf_consistency': wf_result.get('return_consistency', 0)
                })

        print(f"   🏆 {len(finalists)} finalists ready for full backtest")

        return {
            'status': 'pass',
            'candidates_tested': len(candidates),
            'smoke_survivors': len(top_configs),
            'finalists': finalists,
            'smoke_results': smoke_results
        }

    def _sample_configs(self, search_space, count):
        """Sample configurations from search space"""
        configs = []

        # Load base config
        with open(self.config_base, 'r') as f:
            base_config = json.load(f)

        for i in range(count):
            config = base_config.copy()

            # Sample from search space
            if 'confidence_threshold' in search_space:
                min_conf, max_conf = search_space['confidence_threshold']
                config['fusion']['calibration_thresholds']['confidence'] = np.random.uniform(min_conf, max_conf)

            if 'strength_threshold' in search_space:
                min_str, max_str = search_space['strength_threshold']
                config['fusion']['calibration_thresholds']['strength'] = np.random.uniform(min_str, max_str)

            # Add other parameter sampling as needed
            configs.append(config)

        return configs

def main():
    """Example usage of tiered testing framework"""
    tester = TieredTester()

    # Tier 0: Quick preflight
    preflight = tester.tier0_preflight(['ETH_4H', 'ETH_1D'])
    if preflight['status'] == 'fail':
        print("❌ Preflight failed, aborting")
        return

    # Example search space
    search_space = {
        'confidence_threshold': (0.25, 0.35),
        'strength_threshold': (0.35, 0.45)
    }

    # Run budgeted search
    search_results = tester.budgeted_search(search_space, max_configs=20)

    if search_results['status'] == 'pass' and search_results['finalists']:
        print(f"\n🏁 Running full backtest on {len(search_results['finalists'])} finalists...")

        for finalist in search_results['finalists'][:3]:  # Top 3 only
            full_result = tester.tier3_full_backtest(
                finalist['config'],
                assets=['ETH_4H'],
                months=12
            )

            print(f"   Finalist {finalist['config_id']}: {full_result['status']}")

if __name__ == "__main__":
    main()