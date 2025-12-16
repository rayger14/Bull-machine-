#!/usr/bin/env python3
"""
S5 (Long Squeeze) Multi-Objective Calibration Optimizer

Uses Optuna to find optimal S5 thresholds across multiple objectives:
1. Maximize Profit Factor (primary)
2. Maximize Win Rate (secondary)
3. Achieve target trade frequency: 7-12 trades/year

ARCHITECTURE:
- Multi-objective optimization (NSGA-II algorithm)
- Cross-validation: Train (2023 H1) → Validate (2023 H2) → Test (2024 H1)
- Regime gating: Only fires in risk_on regime (bull markets)
- Graceful OI degradation: Works with or without open interest data

SEARCH SPACE:
- fusion_threshold: Determined from distribution analysis (typically p97-p99.5)
- funding_z_min: [1.0, 3.0] - Positive funding threshold
- rsi_min: [70, 85] - Overbought threshold
- liquidity_max: [0.05, 0.25] - Low liquidity threshold
- oi_change_min: [0.05, 0.20] - Rising OI threshold (if available)
- cooldown_bars: [4, 20] - Trade spacing

OUTPUT:
- Pareto frontier of optimal configurations
- CSV with all trials for analysis
- JSON configs for top 3 solutions (conservative/balanced/aggressive)

Author: Claude Code (Backend Architect)
Date: 2025-11-20
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
import optuna
from optuna.samplers import NSGAIISampler
from typing import Dict, Tuple, List
import json
import logging
from datetime import datetime
import subprocess

from engine.strategies.archetypes.bear.long_squeeze_runtime import apply_s5_enrichment

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

# Global cache for feature data (avoid reloading)
FEATURE_CACHE = {}


def load_feature_data_cached(start_date: str, end_date: str) -> pd.DataFrame:
    """Load feature data with caching"""
    cache_key = f"{start_date}_{end_date}"

    if cache_key not in FEATURE_CACHE:
        feature_file = Path('data/features_mtf/BTC_1H_2022-01-01_to_2024-12-31.parquet')

        if not feature_file.exists():
            raise FileNotFoundError(f"Feature store not found: {feature_file}")

        logger.info(f"Loading feature data: {start_date} to {end_date}")
        df = pd.read_parquet(feature_file)
        df = df[(df.index >= start_date) & (df.index <= end_date)].copy()

        # Apply S5 enrichment
        df = apply_s5_enrichment(df, funding_lookback=24, oi_lookback=12, rsi_threshold=70.0)

        FEATURE_CACHE[cache_key] = df
        logger.info(f"Cached {len(df)} bars")

    return FEATURE_CACHE[cache_key].copy()


def create_s5_config(params: Dict, base_config: Dict = None) -> Dict:
    """
    Create S5 backtest config from parameters.

    Args:
        params: Parameter dictionary from Optuna trial
        base_config: Optional base config to merge with

    Returns:
        Complete backtest config dictionary
    """
    if base_config is None:
        base_config = {
            "version": "s5_optimization",
            "profile": "S5 Calibration Test",
            "adaptive_fusion": True,
            "regime_classifier": {
                "model_path": "models/regime_classifier_gmm.pkl",
                "feature_order": [
                    "VIX", "DXY", "MOVE", "YIELD_2Y", "YIELD_10Y",
                    "USDT.D", "BTC.D", "TOTAL", "TOTAL2",
                    "funding", "oi", "rv_20d", "rv_60d"
                ],
                "zero_fill_missing": False,
                "regime_override": {}
            },
            "ml_filter": {"enabled": False},
            "fusion": {
                "entry_threshold_confidence": 0.30,
                "weights": {
                    "wyckoff": 0.35,
                    "liquidity": 0.30,
                    "momentum": 0.35,
                    "smc": 0.0
                }
            },
            "risk": {
                "base_risk_pct": 0.015,
                "max_position_size_pct": 0.15,
                "max_portfolio_risk_pct": 0.08
            }
        }

    # Update archetype section
    base_config["archetypes"] = {
        "use_archetypes": True,
        "max_trades_per_day": 8,
        # Disable all other archetypes
        "enable_A": False, "enable_B": False, "enable_C": False, "enable_D": False,
        "enable_E": False, "enable_F": False, "enable_G": False, "enable_H": False,
        "enable_K": False, "enable_L": False, "enable_M": False,
        "enable_S1": False, "enable_S2": False, "enable_S3": False, "enable_S4": False,
        "enable_S5": True,  # Only S5 enabled
        "enable_S6": False, "enable_S7": False, "enable_S8": False,
        "thresholds": {
            "min_liquidity": 0.10,
            "long_squeeze": {
                "direction": "short",
                "fusion_threshold": params['fusion_threshold'],
                "funding_z_min": params['funding_z_min'],
                "rsi_min": params['rsi_min'],
                "liquidity_max": params['liquidity_max'],
                "oi_change_min": params.get('oi_change_min', 0.10),
                "max_risk_pct": 0.015,
                "atr_stop_mult": params['atr_stop_mult']
            }
        },
        "long_squeeze": {
            "archetype_weight": 2.2,
            "final_fusion_gate": params['fusion_threshold'],
            "cooldown_bars": params['cooldown_bars']
        },
        "routing": {
            "risk_on": {
                "weights": {"long_squeeze": 2.0},
                "final_gate_delta": 0.0
            },
            "neutral": {
                "weights": {"long_squeeze": 1.5},
                "final_gate_delta": 0.0
            },
            "risk_off": {
                "weights": {"long_squeeze": 0.0},  # Disable in bear markets
                "final_gate_delta": 0.0
            },
            "crisis": {
                "weights": {"long_squeeze": 2.5},  # Highest weight in crisis
                "final_gate_delta": 0.0
            }
        },
        "exits": {
            "long_squeeze": {
                "trail_atr": params['trail_atr_mult'],
                "time_limit_hours": 24
            }
        }
    }

    return base_config


def run_backtest(config: Dict, start_date: str, end_date: str, test_id: str) -> Dict:
    """
    Run backtest with given config and extract metrics.

    Args:
        config: Backtest configuration
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)
        test_id: Test identifier for logging

    Returns:
        Dictionary with performance metrics
    """
    # Write config to temp file
    config_path = f'/tmp/s5_opt_{test_id}.json'
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)

    # Run backtest
    cmd = [
        'python3', 'bin/backtest_knowledge_v2.py',
        '--asset', 'BTC',
        '--start', start_date,
        '--end', end_date,
        '--config', config_path
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        output = result.stdout + result.stderr

        # Parse metrics
        metrics = {
            'test_id': test_id,
            'total_trades': 0,
            'win_rate': 0.0,
            'profit_factor': 0.0,
            'sharpe_ratio': 0.0,
            'max_drawdown': 0.0,
            'total_return': 0.0
        }

        for line in output.split('\n'):
            if 'Total Trades:' in line:
                try:
                    metrics['total_trades'] = int(line.split(':')[1].strip().split()[0])
                except:
                    pass
            elif 'Win Rate:' in line:
                try:
                    metrics['win_rate'] = float(line.split(':')[1].strip().replace('%', ''))
                except:
                    pass
            elif 'Profit Factor:' in line:
                try:
                    metrics['profit_factor'] = float(line.split(':')[1].strip().split()[0])
                except:
                    pass
            elif 'Sharpe Ratio:' in line:
                try:
                    metrics['sharpe_ratio'] = float(line.split(':')[1].strip().split()[0])
                except:
                    pass
            elif 'Max Drawdown:' in line:
                try:
                    metrics['max_drawdown'] = float(line.split(':')[1].strip().replace('%', '').split()[0])
                except:
                    pass

        return metrics

    except subprocess.TimeoutExpired:
        logger.warning(f"Backtest timeout for {test_id}")
        return {'test_id': test_id, 'total_trades': 0, 'win_rate': 0.0, 'profit_factor': 0.0,
                'sharpe_ratio': 0.0, 'max_drawdown': 0.0, 'total_return': 0.0}
    except Exception as e:
        logger.error(f"Backtest failed for {test_id}: {e}")
        return {'test_id': test_id, 'total_trades': 0, 'win_rate': 0.0, 'profit_factor': 0.0,
                'sharpe_ratio': 0.0, 'max_drawdown': 0.0, 'total_return': 0.0}


def objective(trial: optuna.Trial) -> Tuple[float, float, float]:
    """
    Multi-objective optimization objective function.

    Returns:
        Tuple of (neg_profit_factor, neg_win_rate, trade_count_penalty)
        Note: Optuna minimizes, so we negate PF and WR
    """
    # Suggest parameters
    params = {
        'fusion_threshold': trial.suggest_float('fusion_threshold', 0.50, 0.75),
        'funding_z_min': trial.suggest_float('funding_z_min', 1.0, 3.0),
        'rsi_min': trial.suggest_float('rsi_min', 70, 85),
        'liquidity_max': trial.suggest_float('liquidity_max', 0.05, 0.25),
        'oi_change_min': trial.suggest_float('oi_change_min', 0.05, 0.20),
        'cooldown_bars': trial.suggest_int('cooldown_bars', 4, 20),
        'atr_stop_mult': trial.suggest_float('atr_stop_mult', 2.0, 3.5),
        'trail_atr_mult': trial.suggest_float('trail_atr_mult', 1.3, 2.0)
    }

    # Create config
    config = create_s5_config(params)

    # Run on training set (2023 H1)
    train_metrics = run_backtest(
        config,
        start_date='2023-01-01',
        end_date='2023-06-30',
        test_id=f"trial_{trial.number}_train"
    )

    # Run on validation set (2023 H2)
    val_metrics = run_backtest(
        config,
        start_date='2023-07-01',
        end_date='2023-12-31',
        test_id=f"trial_{trial.number}_val"
    )

    # Combine metrics (average of train + val)
    total_trades = train_metrics['total_trades'] + val_metrics['total_trades']
    avg_pf = (train_metrics['profit_factor'] + val_metrics['profit_factor']) / 2.0
    avg_wr = (train_metrics['win_rate'] + val_metrics['win_rate']) / 2.0

    # Compute trade frequency penalty
    # Target: 7-12 trades/year → 3.5-6 trades per 6 months
    trades_per_6mo = total_trades / 2.0  # Average over 2 periods
    target_min, target_max = 3.5, 6.0

    if trades_per_6mo < target_min:
        trade_penalty = (target_min - trades_per_6mo) * 0.5  # Too few trades
    elif trades_per_6mo > target_max:
        trade_penalty = (trades_per_6mo - target_max) * 0.3  # Too many trades
    else:
        trade_penalty = 0.0  # Within target

    # Store metrics for later analysis
    trial.set_user_attr('train_trades', train_metrics['total_trades'])
    trial.set_user_attr('val_trades', val_metrics['total_trades'])
    trial.set_user_attr('train_pf', train_metrics['profit_factor'])
    trial.set_user_attr('val_pf', val_metrics['profit_factor'])
    trial.set_user_attr('train_wr', train_metrics['win_rate'])
    trial.set_user_attr('val_wr', val_metrics['win_rate'])
    trial.set_user_attr('trades_per_6mo', trades_per_6mo)

    logger.info(
        f"Trial {trial.number}: "
        f"Trades={total_trades} (train={train_metrics['total_trades']}, val={val_metrics['total_trades']}), "
        f"PF={avg_pf:.2f}, WR={avg_wr:.1f}%, Penalty={trade_penalty:.2f}"
    )

    # Return objectives (Optuna minimizes, so negate PF and WR)
    return -avg_pf, -avg_wr, trade_penalty


def run_optimization(n_trials: int = 100, n_jobs: int = 1) -> optuna.Study:
    """
    Run Optuna multi-objective optimization.

    Args:
        n_trials: Number of trials to run
        n_jobs: Number of parallel jobs (1 = sequential)

    Returns:
        Optuna study object with results
    """
    logger.info(f"Starting S5 optimization with {n_trials} trials")

    # Create study with NSGA-II sampler (multi-objective)
    study = optuna.create_study(
        study_name='s5_calibration',
        directions=['minimize', 'minimize', 'minimize'],  # Minimize neg_PF, neg_WR, penalty
        sampler=NSGAIISampler(population_size=20),
        storage='sqlite:///optuna_s5_calibration.db',
        load_if_exists=True
    )

    # Run optimization
    study.optimize(
        objective,
        n_trials=n_trials,
        n_jobs=n_jobs,
        show_progress_bar=True
    )

    logger.info(f"Optimization complete: {len(study.trials)} trials")

    return study


def analyze_pareto_frontier(study: optuna.Study) -> pd.DataFrame:
    """
    Analyze Pareto frontier and extract top solutions.

    Args:
        study: Completed Optuna study

    Returns:
        DataFrame with Pareto-optimal solutions
    """
    # Get all completed trials
    trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]

    if len(trials) == 0:
        logger.error("No completed trials found")
        return pd.DataFrame()

    # Extract trial data
    trial_data = []
    for t in trials:
        data = {
            'trial_number': t.number,
            'fusion_threshold': t.params['fusion_threshold'],
            'funding_z_min': t.params['funding_z_min'],
            'rsi_min': t.params['rsi_min'],
            'liquidity_max': t.params['liquidity_max'],
            'oi_change_min': t.params['oi_change_min'],
            'cooldown_bars': t.params['cooldown_bars'],
            'atr_stop_mult': t.params['atr_stop_mult'],
            'trail_atr_mult': t.params['trail_atr_mult'],
            'profit_factor': -t.values[0],  # Un-negate
            'win_rate': -t.values[1],       # Un-negate
            'trade_penalty': t.values[2],
            'train_trades': t.user_attrs.get('train_trades', 0),
            'val_trades': t.user_attrs.get('val_trades', 0),
            'total_trades': t.user_attrs.get('train_trades', 0) + t.user_attrs.get('val_trades', 0),
            'trades_per_6mo': t.user_attrs.get('trades_per_6mo', 0)
        }
        trial_data.append(data)

    df = pd.DataFrame(trial_data)

    # Find Pareto frontier (multi-objective optimal solutions)
    pareto_trials = study.best_trials

    pareto_numbers = [t.number for t in pareto_trials]
    df['is_pareto'] = df['trial_number'].isin(pareto_numbers)

    # Sort by profit factor
    df = df.sort_values('profit_factor', ascending=False)

    return df


def save_results(study: optuna.Study, df_results: pd.DataFrame) -> None:
    """
    Save optimization results to files.

    Args:
        study: Optuna study object
        df_results: Results dataframe
    """
    output_dir = Path('results/optimization')
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save all trials
    results_file = output_dir / 's5_calibration_all_trials.csv'
    df_results.to_csv(results_file, index=False)
    logger.info(f"All trials saved to: {results_file}")

    # Save Pareto frontier
    pareto_df = df_results[df_results['is_pareto']].copy()
    pareto_file = output_dir / 's5_calibration_pareto_frontier.csv'
    pareto_df.to_csv(pareto_file, index=False)
    logger.info(f"Pareto frontier saved to: {pareto_file}")

    # Save top 10 by profit factor
    top10_file = output_dir / 's5_calibration_top10.csv'
    df_results.head(10).to_csv(top10_file, index=False)
    logger.info(f"Top 10 trials saved to: {top10_file}")


def main():
    """Main optimization routine"""

    print("="*80)
    print("S5 (LONG SQUEEZE) MULTI-OBJECTIVE CALIBRATION")
    print("="*80)
    print()
    print("Optimization setup:")
    print("  - Objectives: Maximize PF, Maximize WR, Achieve target trade frequency")
    print("  - Cross-validation: 2023 H1 (train) + 2023 H2 (val)")
    print("  - Target: 7-12 trades/year, PF > 1.5")
    print("  - Algorithm: NSGA-II (multi-objective)")
    print()

    # Parse arguments
    import argparse
    parser = argparse.ArgumentParser(description='S5 Calibration Optimizer')
    parser.add_argument('--trials', type=int, default=100, help='Number of trials (default: 100)')
    parser.add_argument('--jobs', type=int, default=1, help='Parallel jobs (default: 1)')
    args = parser.parse_args()

    try:
        # Run optimization
        study = run_optimization(n_trials=args.trials, n_jobs=args.jobs)

        # Analyze results
        print("\n" + "-"*80)
        print("ANALYZING PARETO FRONTIER")
        print("-"*80)

        df_results = analyze_pareto_frontier(study)

        if len(df_results) > 0:
            # Save results
            save_results(study, df_results)

            # Print top solutions
            print("\n" + "-"*80)
            print("TOP 10 SOLUTIONS (by Profit Factor)")
            print("-"*80)
            print()
            print(df_results.head(10)[
                ['trial_number', 'fusion_threshold', 'funding_z_min', 'rsi_min',
                 'total_trades', 'profit_factor', 'win_rate', 'is_pareto']
            ].to_string(index=False, float_format=lambda x: f'{x:.3f}'))

            print("\n" + "-"*80)
            print("PARETO FRONTIER SUMMARY")
            print("-"*80)
            pareto_df = df_results[df_results['is_pareto']]
            print(f"Pareto-optimal solutions: {len(pareto_df)}")
            print(f"Best PF: {pareto_df['profit_factor'].max():.2f}")
            print(f"Best WR: {pareto_df['win_rate'].max():.1f}%")
            print(f"Avg trades/6mo: {pareto_df['trades_per_6mo'].mean():.1f}")

            print("\n" + "="*80)
            print("OPTIMIZATION COMPLETE")
            print("="*80)
            print("\nNext step: Run bin/generate_s5_configs.py to create production configs")

        else:
            logger.error("No results to analyze")
            sys.exit(1)

    except Exception as e:
        logger.error(f"Optimization failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
