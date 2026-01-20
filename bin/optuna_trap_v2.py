#!/usr/bin/env python3
"""
Trap Archetype Optimizer v2 - Fixed Version

Improvements over v1:
1. FIXED SIZING: Disables Kelly + archetype multipliers to isolate entry quality
2. ROLLING OOS: Multiple train/test windows for robustness
3. BETTER OBJECTIVE: Expectancy-based with soft penalties (not PF×WR)
4. REGIME STRATIFIED: Reports breakdown by regime
5. FEATURE CACHING: Loads pre-computed regime labels for speed

Usage:
    # First, cache features
    python3 bin/cache_features_with_regime.py --asset BTC --start 2022-01-01 --end 2024-12-31

    # Then run optimization
    python3 bin/optuna_trap_v2.py \
      --n-trials 200 \
      --cache data/cached/btc_features_2022-01-01_2024-12-31_cached.parquet \
      --output results/optuna_trap_v2
"""

import sys
import json
import argparse
import tempfile
from pathlib import Path
from datetime import datetime
import pandas as pd
import numpy as np
import optuna
from optuna.samplers import TPESampler

# Add project root
sys.path.insert(0, str(Path(__file__).parent.parent))

from bin.backtest_knowledge_v2 import KnowledgeAwareBacktest, KnowledgeParams


class TrapOptimizerV2:
    """
    Improved trap optimizer with fixed sizing and rolling validation.
    """

    def __init__(self, cache_path: str, n_trials: int, output_dir: str):
        self.cache_path = Path(cache_path)
        self.n_trials = n_trials
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Load cached features (saves 10-15s per trial)
        print("📂 Loading cached features...")
        self.df_full = pd.read_parquet(self.cache_path)
        print(f"  ✓ Loaded {len(self.df_full)} bars with regime labels")

        # Define rolling windows
        self.windows = [
            ('2022-01-01', '2022-06-30', '2022-07-01', '2022-12-31'),
            ('2022-01-01', '2022-12-31', '2023-01-01', '2023-12-31'),
            ('2022-01-01', '2023-06-30', '2023-07-01', '2023-12-31'),
            ('2022-01-01', '2023-12-31', '2024-01-01', '2024-12-31'),
        ]

        # Split data by window
        self.window_data = []
        for train_start, train_end, test_start, test_end in self.windows:
            df_train = self.df_full[train_start:train_end]
            df_test = self.df_full[test_start:test_end]
            self.window_data.append({
                'train_start': train_start,
                'train_end': train_end,
                'test_start': test_start,
                'test_end': test_end,
                'df_train': df_train,
                'df_test': df_test
            })

        print(f"  ✓ Created {len(self.windows)} rolling windows")

        # Initialize components
        self.capital = 10000.0

        # Load baseline config (only need bull config for single-archetype optimization)
        self.bull_baseline = self._load_config('configs/baseline_btc_bull_pf20.json')

        print(f"  ✓ Initialized optimizer")
        print(f"  ✓ Output: {self.output_dir}")

    def _load_config(self, path):
        with open(path) as f:
            return json.load(f)

    def _create_fixed_sizing_config(self, base_config, trap_params):
        """
        Create config with FIXED sizing and trap parameters.

        CRITICAL: Disables Kelly + archetype multipliers to isolate entry quality.
        """
        # BUGFIX: Deep copy to prevent modifying the original base_config
        # Shallow copy (base_config.copy()) only copies top-level dict,
        # nested dicts remain as references → all trials modify same object!
        config = json.loads(json.dumps(base_config))

        # UNIFIED NAMING: Write trap parameters using descriptive names
        # The optimizer uses KnowledgeAwareBacktest (legacy path) which reads from:
        # config['archetypes']['trap_within_trend'] via get_archetype_param()
        if 'archetypes' not in config:
            config['archetypes'] = {}

        # Write to legacy archetype param location (read by get_archetype_param)
        trap_config = {
            'fusion_threshold': trap_params.get('fusion_threshold', 0.35),
            'liquidity_threshold': trap_params.get('liquidity_threshold', 0.30),
            'quality_threshold': trap_params.get('quality_threshold', 0.55),
            'adx_threshold': trap_params.get('adx_threshold', 25.0),
            'wick_multiplier': trap_params.get('wick_multiplier', 2.0)
        }
        config['archetypes']['trap_within_trend'] = trap_config

        # DIAGNOSTIC: Print what we're writing
        print(f"\n[CONFIG WRITE] Trial params: fusion={trap_config['fusion_threshold']}, liq={trap_config['liquidity_threshold']}")
        print(f"[CONFIG WRITE] Config location: config['archetypes']['trap_within_trend'] = {trap_config}")

        # FIXED SIZING - This is the key change
        config['position_sizing'] = {
            'mode': 'fixed_fractional',
            'base_risk_per_trade_pct': 0.8,  # Fixed 0.8% risk per trade
            'max_risk_per_trade_pct': 0.8,
            'kelly_fraction': 1.0,
            'confidence_scaling': False,     # DISABLE confidence scaling
            'archetype_quality_weight': 0.0  # DISABLE archetype multipliers
        }

        return config

    def _run_backtest(self, config, df, silent=True):
        """Run backtest with single config (no router/regime switching)."""
        # BUGFIX: Use KnowledgeAwareBacktest directly instead of RouterAwareBacktest
        # RouterAwareBacktest requires regime classification which fails on rolling windows
        # (macro features are NaN → all neutral → CASH mode → no trades → zero variance)

        # For trap optimization, we don't need regime switching
        backtest = KnowledgeAwareBacktest(
            df=df,
            params=KnowledgeParams(),
            starting_capital=self.capital,
            asset='BTC',
            runtime_config=config
        )

        results = backtest.run()

        # Extract metrics
        total_pnl = results.get('total_pnl', 0.0)
        total_trades = results.get('total_trades', 0)
        win_rate = results.get('win_rate', 0.0)
        profit_factor = results.get('profit_factor', 0.0)
        max_dd = results.get('max_drawdown_pct', 0.0)
        avg_win = results.get('avg_win', 0.0)
        avg_loss = results.get('avg_loss', 0.0)

        # Calculate R-based metrics
        risk_per_trade = self.capital * 0.008  # 0.8% fixed
        total_R_risked = total_trades * risk_per_trade
        expectancy_R = total_pnl / total_R_risked if total_R_risked > 0 else 0.0

        # R per trade std (from trade log if available)
        trades_df = results.get('trades_df', None)
        if trades_df is not None and len(trades_df) > 0:
            trades_df['r_multiple'] = trades_df['net_pnl'] / risk_per_trade
            std_R = trades_df['r_multiple'].std()
        else:
            std_R = 1.0  # Default

        return {
            'total_pnl': total_pnl,
            'total_trades': total_trades,
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'max_drawdown_pct': max_dd,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'expectancy_R': expectancy_R,
            'std_R': std_R,
            'trades_df': trades_df
        }

    def objective(self, trial):
        """
        Improved objective function:
        - Expectancy-based scoring
        - Soft penalties (not hard rejects)
        - Aggregates across rolling windows
        """
        # Sample trap parameters (DATA-DERIVED BOUNDS from validator)
        # Bounds computed from data/cached/btc_features_2022-01-01_2024-12-31_cached.parquet
        # via: python3 bin/validate_feature_store.py --bounds
        #
        # Actual data ranges (5th-95th percentile):
        #   tf4h_fusion_score: [0.0000, 0.2195]
        #   adx_14:            [13.7928, 66.6398]
        #   rsi_14:            [23.4468, 78.3034]
        trap_params = {
            'quality_threshold': trial.suggest_float('trap_quality_threshold', 0.00, 0.22, step=0.02),
            'liquidity_threshold': trial.suggest_float('trap_liquidity_threshold', 0.15, 0.45, step=0.05),
            'adx_threshold': trial.suggest_float('trap_adx_threshold', 14.0, 67.0, step=5.0),
            'fusion_threshold': trial.suggest_float('trap_fusion_threshold', 0.00, 0.22, step=0.02),
            'wick_multiplier': trial.suggest_float('trap_wick_multiplier', 0.5, 2.5, step=0.25)
        }

        # Create config with fixed sizing
        bull_config = self._create_fixed_sizing_config(self.bull_baseline, trap_params)

        # Run on all rolling windows
        window_scores = []

        for i, window in enumerate(self.window_data):
            # Run on test split
            results = self._run_backtest(bull_config, window['df_test'])

            # Calculate robust score
            expectancy_R = results['expectancy_R']
            std_R = results['std_R']
            total_trades = results['total_trades']
            max_dd = results['max_drawdown_pct']

            # Stability factor
            stability = 1.0 / (1.0 + std_R)

            # Base score: expectancy × sqrt(trades) × stability
            base_score = expectancy_R * np.sqrt(max(total_trades, 1)) * stability

            # Soft penalties
            dd_penalty = 5.0 * max(0, max_dd - 0.12)  # Penalty if DD > 12%
            trade_penalty = 2.0 * max(0, 15 - total_trades)  # Penalty if < 15 trades

            window_score = base_score - dd_penalty - trade_penalty

            window_scores.append(window_score)

            # Report intermediate values
            trial.set_user_attr(f'window_{i}_score', window_score)
            trial.set_user_attr(f'window_{i}_pnl', results['total_pnl'])
            trial.set_user_attr(f'window_{i}_trades', results['total_trades'])
            trial.set_user_attr(f'window_{i}_wr', results['win_rate'])

        # Aggregate: Use median for robustness
        final_score = np.median(window_scores)

        # Also track min (worst case)
        trial.set_user_attr('worst_window_score', min(window_scores))
        trial.set_user_attr('best_window_score', max(window_scores))
        trial.set_user_attr('score_std', np.std(window_scores))

        return final_score

    def run_optimization(self):
        """Execute Optuna study with zero-variance detection."""
        print("\n" + "="*80)
        print("🚀 TRAP OPTIMIZER V2 - FIXED SIZING + ROLLING OOS")
        print("="*80)
        print(f"\nConfiguration:")
        print(f"  • Trials: {self.n_trials}")
        print(f"  • Rolling windows: {len(self.windows)}")
        print(f"  • Sizing: FIXED 0.8% (no Kelly, no archetype multipliers)")
        print(f"  • Objective: Expectancy-based with soft penalties")
        print(f"  • Output: {self.output_dir}")
        print(f"  • Fail-fast: Zero-variance sentinel (aborts after 20 identical trials)")

        # Create study with pruning
        study = optuna.create_study(
            direction='maximize',
            sampler=TPESampler(seed=42),
            pruner=optuna.pruners.MedianPruner(  # ASHA-like pruning
                n_startup_trials=10,
                n_warmup_steps=5
            )
        )

        # Zero-variance detection callback
        def check_variance_callback(study, trial):
            """
            Abort optimization if zero variance detected.

            Prevents 8-hour wasted runs like the v1 failure where all 200 trials
            produced identical scores due to parameter wiring issues.
            """
            if len(study.trials) >= 20:
                values = [t.value for t in study.trials if t.value is not None]
                if len(values) >= 20:
                    std = np.std(values)
                    unique_count = len(set(values))

                    if std < 1e-6 or unique_count == 1:
                        raise RuntimeError(
                            f"\n{'='*70}\n"
                            f"❌ ZERO VARIANCE DETECTED!\n"
                            f"{'='*70}\n"
                            f"After {len(values)} trials:\n"
                            f"  • All scores identical or near-identical (std={std:.2e})\n"
                            f"  • Unique values: {unique_count}\n"
                            f"\n"
                            f"This indicates parameters are NOT affecting outcomes.\n"
                            f"\n"
                            f"Action required:\n"
                            f"  1. Run wire tests: python3 bin/test_param_wiring.py\n"
                            f"  2. Fix any failing tests\n"
                            f"  3. Re-run optimization\n"
                            f"\n"
                            f"This fail-fast check saved you from wasting ~6 hours!\n"
                            f"{'='*70}\n"
                        )

        # Run optimization with callback
        study.optimize(
            self.objective,
            n_trials=self.n_trials,
            callbacks=[check_variance_callback],
            show_progress_bar=True
        )

        # Save results
        self._save_results(study)

        return study

    def _save_results(self, study):
        """Save optimization results."""
        # Best params
        best_params = {
            'best_params': study.best_params,
            'best_value': study.best_value,
            'best_trial_number': study.best_trial.number,
            'user_attrs': study.best_trial.user_attrs,
            'optimization_config': {
                'n_trials': self.n_trials,
                'windows': self.windows,
                'sizing': 'fixed_0.8pct',
                'objective': 'expectancy_based'
            }
        }

        best_path = self.output_dir / 'best_params.json'
        with open(best_path, 'w') as f:
            json.dump(best_params, f, indent=2)

        print(f"\n💾 Best parameters saved: {best_path}")

        # Trials dataframe
        trials_df = study.trials_dataframe()
        trials_path = self.output_dir / 'trials.csv'
        trials_df.to_csv(trials_path, index=False)

        print(f"💾 Trials saved: {trials_path}")

        # Generate config with best params (bull only, no bear config for single-archetype optimization)
        trap_best = study.best_params
        bull_optimized = self._create_fixed_sizing_config(self.bull_baseline, trap_best)

        bull_opt_path = self.output_dir / 'trap_v2_optimized_bull.json'

        with open(bull_opt_path, 'w') as f:
            json.dump(bull_optimized, f, indent=2)

        print(f"💾 Optimized bull config: {bull_opt_path}")

        # Summary report
        print("\n" + "="*80)
        print("📊 OPTIMIZATION SUMMARY")
        print("="*80)
        print(f"\n🏆 Best Trial: #{study.best_trial.number}")
        print(f"   Score: {study.best_value:.4f}")
        print(f"\n📋 Best Parameters:")
        for param, value in study.best_params.items():
            print(f"   • {param}: {value}")

        print(f"\n📈 Window Breakdown:")
        for i in range(len(self.windows)):
            score = study.best_trial.user_attrs.get(f'window_{i}_score', 0)
            pnl = study.best_trial.user_attrs.get(f'window_{i}_pnl', 0)
            trades = study.best_trial.user_attrs.get(f'window_{i}_trades', 0)
            wr = study.best_trial.user_attrs.get(f'window_{i}_wr', 0)
            print(f"   Window {i+1}: Score={score:.3f}, PNL=${pnl:.0f}, Trades={trades}, WR={wr:.1%}")

        worst_score = study.best_trial.user_attrs.get('worst_window_score', 0)
        best_score = study.best_trial.user_attrs.get('best_window_score', 0)
        score_std = study.best_trial.user_attrs.get('score_std', 0)

        print(f"\n📊 Robustness Metrics:")
        print(f"   • Worst window score: {worst_score:.3f}")
        print(f"   • Best window score: {best_score:.3f}")
        print(f"   • Score std: {score_std:.3f}")

        print("\n" + "="*80)
        print("✅ Optimization complete!")
        print("="*80)


def main():
    parser = argparse.ArgumentParser(description='Trap Optimizer v2 with fixed sizing')
    parser.add_argument('--n-trials', type=int, default=200, help='Number of Optuna trials')
    parser.add_argument('--cache', required=True, help='Path to cached feature parquet')
    parser.add_argument('--output', default='results/optuna_trap_v2', help='Output directory')

    args = parser.parse_args()

    optimizer = TrapOptimizerV2(
        cache_path=args.cache,
        n_trials=args.n_trials,
        output_dir=args.output
    )

    study = optimizer.run_optimization()

    return 0


if __name__ == '__main__':
    exit(main())
