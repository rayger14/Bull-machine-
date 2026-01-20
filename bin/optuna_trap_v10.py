#!/usr/bin/env python3
"""
Optuna optimization for trap_within_trend archetype parameters.

Problem: 104 trades (83% of all trades), NET LOSS -$352.95, 46% WR
Target: WR 55%+, PF 1.5+, avg loss < $50
Expected gain: +$400-600/year
"""

import sys
import json
import argparse
from pathlib import Path
from datetime import datetime
import pandas as pd
import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import optuna
from optuna.visualization import plot_optimization_history, plot_param_importances

# Import backtest engine
from bin.backtest_router_v10_full import RouterAwareBacktest
from bin.backtest_knowledge_v2 import KnowledgeParams
from engine.router_v10 import RouterV10
from engine.regime_detector import RegimeDetector
from engine.event_calendar import EventCalendar


class TrapOptunaOptimizer:
    """Optuna optimizer for trap-within-trend archetype."""

    def __init__(self, train_start, train_end, val_start, val_end, bull_config_path, bear_config_path, capital=10000):
        self.train_start = train_start
        self.train_end = train_end
        self.val_start = val_start
        self.val_end = val_end
        self.bull_config_path = bull_config_path
        self.bear_config_path = bear_config_path
        self.capital = capital

        # Load baseline configs
        with open(bull_config_path) as f:
            self.bull_config_base = json.load(f)
        with open(bear_config_path) as f:
            self.bear_config_base = json.load(f)

        # Load feature stores
        print(f"Loading training data ({train_start} to {train_end})...")
        self.df_train = self._load_feature_store(train_start, train_end)
        print(f"  Loaded {len(self.df_train)} bars")

        print(f"Loading validation data ({val_start} to {val_end})...")
        self.df_val = self._load_feature_store(val_start, val_end)
        print(f"  Loaded {len(self.df_val)} bars")

        # Initialize router components (shared)
        self.regime_detector = RegimeDetector(model_path='models/regime_gmm_v3.1_fixed.pkl')
        self.event_calendar = EventCalendar()

    def _load_feature_store(self, start, end):
        """Load feature store for given date range."""
        # Determine which parquet file to load
        if start >= '2024-01-01':
            path = 'data/features_mtf/BTC_1H_2024-01-01_to_2024-12-31.parquet'
        else:
            path = 'data/features_mtf/BTC_1H_2022-01-01_to_2023-12-31.parquet'

        df = pd.read_parquet(path)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.set_index('timestamp')

        # Filter to date range
        df = df[(df.index >= start) & (df.index <= end)]

        return df

    def _create_config_with_params(self, trial):
        """Create config with Optuna-sampled trap parameters."""
        # Sample hyperparameters
        trap_quality = trial.suggest_float('trap_quality_threshold', 0.45, 0.65, step=0.05)
        trap_confirmation = trial.suggest_int('trap_confirmation_bars', 2, 5)
        trap_volume_ratio = trial.suggest_float('trap_volume_ratio', 1.5, 2.5, step=0.1)
        trap_stop_mult = trial.suggest_float('trap_stop_multiplier', 0.8, 1.5, step=0.1)

        # Create modified configs
        bull_config = self.bull_config_base.copy()
        bear_config = self.bear_config_base.copy()

        # Apply trap parameters to both configs
        for config in [bull_config, bear_config]:
            if 'archetypes' not in config:
                config['archetypes'] = {}
            if 'trap_within_trend' not in config['archetypes']:
                config['archetypes']['trap_within_trend'] = {}

            config['archetypes']['trap_within_trend'].update({
                'quality_threshold': trap_quality,
                'confirmation_bars': trap_confirmation,
                'volume_ratio': trap_volume_ratio,
                'stop_multiplier': trap_stop_mult
            })

        return bull_config, bear_config

    def _run_backtest(self, bull_config, bear_config, df):
        """Run backtest with given configs and data."""
        # Save configs to temp files for router
        import tempfile
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(bull_config, f)
            bull_config_path = f.name
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(bear_config, f)
            bear_config_path = f.name

        # Create router with config paths
        router = RouterV10(
            bull_config_path=bull_config_path,
            bear_config_path=bear_config_path
        )

        # Create empty params object (configs are passed separately)
        params = KnowledgeParams()

        # Run backtest
        backtest = RouterAwareBacktest(
            df=df,
            params=params,
            bull_config=bull_config,
            bear_config=bear_config,
            router=router,
            regime_detector=self.regime_detector,
            event_calendar=self.event_calendar,
            starting_capital=self.capital,
            asset='BTC'
        )

        results = backtest.run()

        # Cleanup temp files
        import os
        try:
            os.unlink(bull_config_path)
            os.unlink(bear_config_path)
        except:
            pass

        return results

    def objective(self, trial):
        """Optuna objective function."""
        # Sample parameters
        bull_config, bear_config = self._create_config_with_params(trial)

        # Run training backtest
        results_train = self._run_backtest(bull_config, bear_config, self.df_train)

        # Extract metrics
        pf_train = results_train['profit_factor']
        wr_train = results_train['win_rate']
        dd_train = results_train['max_drawdown']
        total_trades = results_train['total_trades']

        # Constraint: Max drawdown must be < 10%
        if dd_train > 0.10:
            return -1000.0  # Heavily penalize

        # Constraint: Must have reasonable number of trades (not too few)
        if total_trades < 20:
            return -500.0  # Penalize too few trades

        # Objective: Maximize (PF × WR)
        # This balances profitability and consistency
        objective_value = pf_train * wr_train

        # Store additional metrics for analysis
        trial.set_user_attr('train_pf', pf_train)
        trial.set_user_attr('train_wr', wr_train)
        trial.set_user_attr('train_dd', dd_train)
        trial.set_user_attr('train_trades', total_trades)
        trial.set_user_attr('train_pnl', results_train['total_pnl'])

        # Run validation backtest (for monitoring, not optimization)
        try:
            results_val = self._run_backtest(bull_config, bear_config, self.df_val)
            trial.set_user_attr('val_pf', results_val['profit_factor'])
            trial.set_user_attr('val_wr', results_val['win_rate'])
            trial.set_user_attr('val_dd', results_val['max_drawdown'])
            trial.set_user_attr('val_trades', results_val['total_trades'])
            trial.set_user_attr('val_pnl', results_val['total_pnl'])
        except Exception as e:
            print(f"Validation failed: {e}")
            trial.set_user_attr('val_pf', 0.0)
            trial.set_user_attr('val_wr', 0.0)

        return objective_value

    def run_optimization(self, n_trials=200, output_dir='results/optuna_trap_v10'):
        """Run Optuna optimization."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        print("\n" + "="*80)
        print("TRAP-WITHIN-TREND OPTUNA OPTIMIZATION")
        print("="*80)
        print(f"Training period: {self.train_start} to {self.train_end}")
        print(f"Validation period: {self.val_start} to {self.val_end}")
        print(f"Number of trials: {n_trials}")
        print(f"Objective: Maximize (PF × WR), constrain DD < 10%")
        print("="*80 + "\n")

        # Create study
        study = optuna.create_study(
            direction='maximize',
            study_name='trap_within_trend_optimization',
            sampler=optuna.samplers.TPESampler(seed=42)
        )

        # Run optimization
        study.optimize(self.objective, n_trials=n_trials, show_progress_bar=True)

        # Print results
        print("\n" + "="*80)
        print("OPTIMIZATION COMPLETE")
        print("="*80)

        print("\n📊 Best Trial:")
        best_trial = study.best_trial
        print(f"  Objective Value (PF × WR): {best_trial.value:.4f}")
        print(f"\n  Best Parameters:")
        for key, value in best_trial.params.items():
            print(f"    {key}: {value}")

        print(f"\n  Training Metrics:")
        print(f"    PF: {best_trial.user_attrs['train_pf']:.2f}")
        print(f"    WR: {best_trial.user_attrs['train_wr']*100:.1f}%")
        print(f"    DD: {best_trial.user_attrs['train_dd']*100:.2f}%")
        print(f"    Trades: {best_trial.user_attrs['train_trades']}")
        print(f"    PNL: ${best_trial.user_attrs['train_pnl']:.2f}")

        if 'val_pf' in best_trial.user_attrs:
            print(f"\n  Validation Metrics:")
            print(f"    PF: {best_trial.user_attrs['val_pf']:.2f}")
            print(f"    WR: {best_trial.user_attrs['val_wr']*100:.1f}%")
            print(f"    DD: {best_trial.user_attrs['val_dd']*100:.2f}%")
            print(f"    Trades: {best_trial.user_attrs['val_trades']}")
            print(f"    PNL: ${best_trial.user_attrs['val_pnl']:.2f}")

        # Save results
        results = {
            'best_params': best_trial.params,
            'best_value': best_trial.value,
            'train_metrics': {
                'profit_factor': best_trial.user_attrs['train_pf'],
                'win_rate': best_trial.user_attrs['train_wr'],
                'max_drawdown': best_trial.user_attrs['train_dd'],
                'total_trades': best_trial.user_attrs['train_trades'],
                'total_pnl': best_trial.user_attrs['train_pnl']
            },
            'val_metrics': {
                'profit_factor': best_trial.user_attrs.get('val_pf', 0.0),
                'win_rate': best_trial.user_attrs.get('val_wr', 0.0),
                'max_drawdown': best_trial.user_attrs.get('val_dd', 0.0),
                'total_trades': best_trial.user_attrs.get('val_trades', 0),
                'total_pnl': best_trial.user_attrs.get('val_pnl', 0.0)
            },
            'n_trials': n_trials,
            'timestamp': datetime.now().isoformat()
        }

        with open(output_path / 'best_params.json', 'w') as f:
            json.dump(results, f, indent=2)

        print(f"\n✅ Results saved to: {output_path / 'best_params.json'}")

        # Create optimized config
        bull_config, bear_config = self._create_config_with_params(best_trial)

        with open(output_path / 'trap_optimized_bull.json', 'w') as f:
            json.dump(bull_config, f, indent=2)

        with open(output_path / 'trap_optimized_bear.json', 'w') as f:
            json.dump(bear_config, f, indent=2)

        print(f"✅ Optimized configs saved to: {output_path}")

        # Export trial data for analysis
        trials_df = study.trials_dataframe()
        trials_df.to_csv(output_path / 'trials.csv', index=False)
        print(f"✅ Trial data saved to: {output_path / 'trials.csv'}")

        # Plot optimization history
        try:
            import matplotlib.pyplot as plt
            fig = plot_optimization_history(study)
            fig.write_html(str(output_path / 'optimization_history.html'))
            print(f"✅ Optimization history plot: {output_path / 'optimization_history.html'}")

            fig = plot_param_importances(study)
            fig.write_html(str(output_path / 'param_importances.html'))
            print(f"✅ Parameter importance plot: {output_path / 'param_importances.html'}")
        except Exception as e:
            print(f"⚠️  Could not generate plots: {e}")

        print("="*80 + "\n")

        return study


def main():
    parser = argparse.ArgumentParser(description='Optuna optimization for trap-within-trend archetype')
    parser.add_argument('--train-start', default='2022-01-01', help='Training start date')
    parser.add_argument('--train-end', default='2023-12-31', help='Training end date')
    parser.add_argument('--val-start', default='2024-01-01', help='Validation start date')
    parser.add_argument('--val-end', default='2024-12-31', help='Validation end date')
    parser.add_argument('--bull-config', default='configs/v10_bases/btc_bull_v10_baseline.json', help='Bull config path')
    parser.add_argument('--bear-config', default='configs/v10_bases/btc_bear_v10_baseline.json', help='Bear config path')
    parser.add_argument('--n-trials', type=int, default=200, help='Number of Optuna trials')
    parser.add_argument('--capital', type=float, default=10000, help='Starting capital')
    parser.add_argument('--output', default='results/optuna_trap_v10', help='Output directory')

    args = parser.parse_args()

    # Create optimizer
    optimizer = TrapOptunaOptimizer(
        train_start=args.train_start,
        train_end=args.train_end,
        val_start=args.val_start,
        val_end=args.val_end,
        bull_config_path=args.bull_config,
        bear_config_path=args.bear_config,
        capital=args.capital
    )

    # Run optimization
    study = optimizer.run_optimization(n_trials=args.n_trials, output_dir=args.output)

    print("\n🎯 Next Steps:")
    print("1. Review best parameters in results/optuna_trap_v10/best_params.json")
    print("2. Analyze trial data in results/optuna_trap_v10/trials.csv")
    print("3. Validate on full 2022-2024 period with optimized configs")
    print("4. If validation passes, commit optimized configs to repo")


if __name__ == '__main__':
    main()
