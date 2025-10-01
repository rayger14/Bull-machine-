#!/usr/bin/env python3
"""
Config Sweep with Early Stopping for Bull Machine v1.7
Implements budgeted search with successive halving
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import pandas as pd
import numpy as np
import json
import time
import multiprocessing as mp
from pathlib import Path
import argparse
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

from scripts.tiered_testing import TieredTester

class ConfigSweep:
    def __init__(self, base_config='configs/v170/assets/ETH_v17_tuned.json',
                 results_dir='results/config_sweep'):
        """Initialize config sweep framework"""
        self.base_config = base_config
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)

        self.tester = TieredTester(base_config)

        print("🔬 BULL MACHINE v1.7 CONFIG SWEEP")
        print("=" * 50)

    def run_sweep(self, search_space, max_configs=40, keep_ratio=0.25,
                  parallel=True, max_workers=None):
        """
        Run complete config sweep with successive halving
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M")
        sweep_id = f"sweep_{timestamp}"

        print(f"🚀 Starting sweep: {sweep_id}")
        print(f"   Max configs: {max_configs}")
        print(f"   Keep ratio: {keep_ratio}")
        print(f"   Parallel: {parallel}")

        # Generate candidate configs
        candidates = self._generate_candidate_configs(search_space, max_configs)
        print(f"   ✅ Generated {len(candidates)} candidate configs")

        # Save candidates
        candidates_file = self.results_dir / f"{sweep_id}_candidates.json"
        with open(candidates_file, 'w') as f:
            json.dump(candidates, f, indent=2, default=str)

        # Phase 1: Smoke slice evaluation
        print(f"\n📊 PHASE 1: SMOKE SLICE EVALUATION")
        print("-" * 40)

        if parallel and max_workers != 1:
            smoke_results = self._run_smoke_parallel(candidates, max_workers)
        else:
            smoke_results = self._run_smoke_sequential(candidates)

        # Filter and rank results
        valid_results = [r for r in smoke_results if r['status'] == 'pass']
        if not valid_results:
            return self._save_sweep_results(sweep_id, {
                'status': 'fail',
                'error': 'No configs passed smoke test',
                'candidates': len(candidates),
                'smoke_survivors': 0
            })

        # Rank by composite score
        for result in valid_results:
            result['composite_score'] = self._calculate_composite_score(result)

        valid_results.sort(key=lambda x: x['composite_score'], reverse=True)

        # Select survivors
        keep_count = max(1, int(len(valid_results) * keep_ratio))
        survivors = valid_results[:keep_count]

        print(f"   📈 Smoke phase: {len(valid_results)}/{len(candidates)} passed")
        print(f"   🏆 Selected {keep_count} survivors for walk-forward")

        # Phase 2: Walk-forward validation
        print(f"\n🚶 PHASE 2: WALK-FORWARD VALIDATION")
        print("-" * 40)

        walk_forward_windows = [
            ('2025-05-01', '2025-06-15'),
            ('2025-07-01', '2025-08-15'),
            ('2025-08-15', '2025-09-30')
        ]

        finalists = []
        for i, result in enumerate(survivors):
            print(f"   Testing survivor {i+1}/{len(survivors)}")

            config = candidates[result['config_id']]
            wf_result = self.tester.tier2_walk_forward(config, walk_forward_windows, timeout=600)

            if wf_result['status'] == 'pass':
                finalist = {
                    'config_id': result['config_id'],
                    'config': config,
                    'smoke_score': result['composite_score'],
                    'smoke_return': result.get('total_return', 0),
                    'wf_avg_return': wf_result.get('avg_return', 0),
                    'wf_consistency': wf_result.get('return_consistency', 0),
                    'wf_windows_passed': wf_result.get('windows_passed', 0)
                }
                finalists.append(finalist)

        print(f"   🎯 Walk-forward: {len(finalists)}/{len(survivors)} advanced")

        # Save comprehensive results
        sweep_results = {
            'sweep_id': sweep_id,
            'status': 'pass' if finalists else 'partial',
            'timestamp': timestamp,
            'search_space': search_space,
            'candidates_generated': len(candidates),
            'smoke_survivors': len(survivors),
            'finalists': len(finalists),
            'smoke_results': smoke_results,
            'walk_forward_results': finalists,
            'top_configs': finalists[:3] if finalists else []
        }

        return self._save_sweep_results(sweep_id, sweep_results)

    def _generate_candidate_configs(self, search_space, count):
        """Generate candidate configurations using sampling strategies"""
        with open(self.base_config, 'r') as f:
            base_config = json.load(f)

        candidates = []

        # Include base config as candidate 0
        candidates.append(base_config)

        # Generate variations
        for i in range(1, count):
            config = self._deep_copy_config(base_config)

            # Sample confidence threshold
            if 'confidence_threshold' in search_space:
                min_val, max_val = search_space['confidence_threshold']
                config['fusion']['calibration_thresholds']['confidence'] = \
                    np.random.uniform(min_val, max_val)

            # Sample strength threshold
            if 'strength_threshold' in search_space:
                min_val, max_val = search_space['strength_threshold']
                config['fusion']['calibration_thresholds']['strength'] = \
                    np.random.uniform(min_val, max_val)

            # Sample SMC parameters
            if 'smc_params' in search_space:
                smc_space = search_space['smc_params']
                if 'ob_threshold' in smc_space:
                    min_val, max_val = smc_space['ob_threshold']
                    config['domains']['smc']['order_blocks']['threshold'] = \
                        np.random.uniform(min_val, max_val)

            # Sample momentum parameters
            if 'momentum_params' in search_space:
                mom_space = search_space['momentum_params']
                if 'rsi_period' in mom_space:
                    min_val, max_val = mom_space['rsi_period']
                    config['domains']['momentum']['rsi']['period'] = \
                        int(np.random.uniform(min_val, max_val))

            # Sample Wyckoff parameters
            if 'wyckoff_params' in search_space:
                wyc_space = search_space['wyckoff_params']
                if 'hps_threshold' in wyc_space:
                    min_val, max_val = wyc_space['hps_threshold']
                    if 'hps_scoring' in config['domains']['wyckoff']:
                        config['domains']['wyckoff']['hps_scoring']['threshold'] = \
                            np.random.uniform(min_val, max_val)

            # Add config metadata
            config['_meta'] = {
                'config_id': i,
                'generation_method': 'random_sampling',
                'timestamp': datetime.now().isoformat()
            }

            candidates.append(config)

        return candidates

    def _run_smoke_sequential(self, candidates):
        """Run smoke tests sequentially"""
        results = []

        for i, config in enumerate(candidates):
            print(f"      Config {i+1}/{len(candidates)}", end=" ")

            try:
                result = self.tester.tier1_smoke_slice(
                    config,
                    start_date='2025-07-01',
                    end_date='2025-09-01',
                    timeout=300
                )
                result['config_id'] = i

                status_icon = "✅" if result['status'] == 'pass' else "❌"
                return_str = f"{result.get('total_return', 0):+.1f}%" if result['status'] == 'pass' else "FAIL"

                print(f"{status_icon} {return_str}")

            except Exception as e:
                result = {
                    'status': 'fail',
                    'error': str(e),
                    'config_id': i
                }
                print(f"❌ ERROR")

            results.append(result)

        return results

    def _run_smoke_parallel(self, candidates, max_workers=None):
        """Run smoke tests in parallel"""
        if max_workers is None:
            max_workers = min(8, mp.cpu_count())

        print(f"      Using {max_workers} parallel workers")

        # Prepare work items
        work_items = []
        for i, config in enumerate(candidates):
            work_items.append((i, config))

        # Run in parallel
        with mp.Pool(max_workers) as pool:
            results = pool.map(self._smoke_worker, work_items)

        return results

    def _smoke_worker(self, work_item):
        """Worker function for parallel smoke tests"""
        config_id, config = work_item

        try:
            # Create new tester instance for this worker
            tester = TieredTester(self.base_config)

            result = tester.tier1_smoke_slice(
                config,
                start_date='2025-07-01',
                end_date='2025-09-01',
                timeout=300
            )
            result['config_id'] = config_id

        except Exception as e:
            result = {
                'status': 'fail',
                'error': str(e),
                'config_id': config_id
            }

        return result

    def _calculate_composite_score(self, result):
        """Calculate composite score for ranking configs"""
        if result['status'] != 'pass':
            return -999

        # Base score from return
        base_score = result.get('total_return', 0)

        # Bonus for trades (activity)
        trade_bonus = min(5, result.get('total_trades', 0)) * 0.1

        # Penalty for poor win rate
        win_rate = result.get('win_rate', 0)
        win_penalty = max(0, (30 - win_rate) * 0.05)

        # Health band bonus/penalties
        health_bonus = 0

        # Good macro veto rate (5-15%)
        macro_rate = result.get('macro_veto_rate', 0)
        if 0.05 <= macro_rate <= 0.15:
            health_bonus += 0.5

        # Good SMC hit rate (≥30%)
        smc_rate = result.get('smc_2hit_rate', 0)
        if smc_rate >= 0.30:
            health_bonus += 0.3

        # No delta breaches
        if result.get('delta_breaches', 0) == 0:
            health_bonus += 0.2

        composite = base_score + trade_bonus - win_penalty + health_bonus

        return composite

    def _deep_copy_config(self, config):
        """Deep copy configuration"""
        import copy
        return copy.deepcopy(config)

    def _save_sweep_results(self, sweep_id, results):
        """Save sweep results to files"""
        # Main results file
        results_file = self.results_dir / f"{sweep_id}_results.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)

        # Summary file
        summary_file = self.results_dir / f"{sweep_id}_summary.txt"
        with open(summary_file, 'w') as f:
            f.write(f"Bull Machine v1.7 Config Sweep: {sweep_id}\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Status: {results['status']}\n")
            f.write(f"Candidates: {results['candidates_generated']}\n")
            f.write(f"Smoke survivors: {results['smoke_survivors']}\n")
            f.write(f"Finalists: {results['finalists']}\n\n")

            if results.get('top_configs'):
                f.write("TOP CONFIGURATIONS:\n")
                f.write("-" * 20 + "\n")
                for i, config in enumerate(results['top_configs'], 1):
                    f.write(f"{i}. Config {config['config_id']}:\n")
                    f.write(f"   Smoke Return: {config['smoke_return']:+.2f}%\n")
                    f.write(f"   WF Avg Return: {config['wf_avg_return']:+.2f}%\n")
                    f.write(f"   WF Consistency: {config['wf_consistency']:.3f}\n\n")

        print(f"\n💾 Results saved:")
        print(f"   📊 {results_file}")
        print(f"   📋 {summary_file}")

        return results

def main():
    """CLI for config sweep"""
    parser = argparse.ArgumentParser(description='Bull Machine v1.7 Config Sweep')
    parser.add_argument('--configs', type=int, default=20,
                       help='Number of configs to test (default: 20)')
    parser.add_argument('--keep-ratio', type=float, default=0.25,
                       help='Ratio of configs to keep (default: 0.25)')
    parser.add_argument('--parallel', action='store_true',
                       help='Run tests in parallel')
    parser.add_argument('--workers', type=int, default=None,
                       help='Number of parallel workers')

    args = parser.parse_args()

    # Define search space
    search_space = {
        'confidence_threshold': (0.25, 0.35),
        'strength_threshold': (0.35, 0.45),
        'smc_params': {
            'ob_threshold': (0.3, 0.7)
        },
        'momentum_params': {
            'rsi_period': (12, 16)
        },
        'wyckoff_params': {
            'hps_threshold': (0.4, 0.8)
        }
    }

    # Run sweep
    sweeper = ConfigSweep()
    results = sweeper.run_sweep(
        search_space=search_space,
        max_configs=args.configs,
        keep_ratio=args.keep_ratio,
        parallel=args.parallel,
        max_workers=args.workers
    )

    if results['status'] == 'pass' and results['finalists'] > 0:
        print(f"\n🏆 SUCCESS: {results['finalists']} finalists ready for full backtest")
        print(f"   Top config: {results['top_configs'][0]['config_id']} "
              f"({results['top_configs'][0]['smoke_return']:+.2f}%)")
    else:
        print(f"\n⚠️  PARTIAL: {results['smoke_survivors']} survivors, "
              f"{results['finalists']} finalists")

if __name__ == "__main__":
    main()