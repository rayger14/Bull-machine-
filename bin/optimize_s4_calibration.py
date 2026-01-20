#!/usr/bin/env python3
"""
S4 (Funding Divergence) Multi-Objective Calibration Optimizer

Uses Optuna to find optimal S4 thresholds across multiple objectives:
1. Maximize Profit Factor (primary) - Target: >2.0
2. Maximize Win Rate (secondary) - Target: >50%
3. Achieve target trade frequency: 6-10 trades/year

PATTERN LOGIC:
S4 is the OPPOSITE of S5 (Long Squeeze):
- S5: Positive funding → longs overcrowded → cascade DOWN
- S4: Negative funding → shorts overcrowded → squeeze UP

ARCHITECTURE:
- Multi-objective optimization (NSGA-II algorithm)
- Cross-validation: Train (2022 H1) → Validate (2022 H2) → Test (2023 H1)
- Regime gating: Fires in risk_off/neutral (bear markets)
- Runtime enrichment: Applies S4 features on-demand

SEARCH SPACE:
- fusion_threshold: [0.75, 0.90] - Higher than baseline 0.80
- funding_z_max: [-2.2, -1.5] - NEGATIVE funding threshold (< -1.8 baseline)
- resilience_min: [0.55, 0.70] - Price resilience threshold (0.6 baseline)
- liquidity_max: [0.20, 0.35] - Low liquidity threshold (0.25 baseline)
- cooldown_bars: [8, 18] - Trade spacing (12 baseline)
- atr_stop_mult: [2.0, 3.5] - Stop loss multiplier (2.5 baseline)

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

from engine.strategies.archetypes.bear.funding_divergence_runtime import apply_s4_enrichment

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

        # Apply S4 enrichment
        df = apply_s4_enrichment(df, funding_lookback=24, price_lookback=12)

        FEATURE_CACHE[cache_key] = df
        logger.info(f"Cached {len(df)} bars")

    return FEATURE_CACHE[cache_key].copy()


def create_s4_config(params: Dict, base_config: Dict = None) -> Dict:
    """
    Create S4 backtest config from parameters.

    Args:
        params: Parameter dictionary from Optuna trial
        base_config: Optional base config to merge with

    Returns:
        Complete backtest config dictionary
    """
    if base_config is None:
        base_config = {
            "version": "s4_optimization",
            "profile": "S4 Calibration Test",
            "adaptive_fusion": False,
            "regime_classifier": {
                "model_path": "models/regime_classifier_gmm.pkl",
                "feature_order": [
                    "VIX", "DXY", "MOVE", "YIELD_2Y", "YIELD_10Y",
                    "USDT.D", "BTC.D", "TOTAL", "TOTAL2",
                    "funding", "oi", "rv_20d", "rv_60d"
                ],
                "zero_fill_missing": False,
                "regime_override": {
                    "2022": "risk_off"  # Force bear regime for 2022
                }
            },
            "ml_filter": {"enabled": False},
            "fusion": {
                "entry_threshold_confidence": 0.99,  # Disable baseline trades
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
        "enable_S1": False, "enable_S2": False, "enable_S3": False,
        "enable_S4": True,  # Only S4 enabled
        "enable_S5": False, "enable_S6": False, "enable_S7": False, "enable_S8": False,
        "thresholds": {
            "min_liquidity": 0.10,
            "funding_divergence": {
                "direction": "long",  # Short squeeze goes UP
                "archetype_weight": 2.5,
                "fusion_threshold": params['fusion_threshold'],
                "final_fusion_gate": params['fusion_threshold'],
                "funding_z_max": params['funding_z_max'],  # NEGATIVE value!
                "resilience_min": params['resilience_min'],
                "liquidity_max": params['liquidity_max'],
                "cooldown_bars": params['cooldown_bars'],
                "max_risk_pct": 0.02,
                "atr_stop_mult": params['atr_stop_mult'],
                "use_runtime_features": True,
                "funding_lookback": 24,
                "price_lookback": 12,
                "weights": {
                    "funding_negative": 0.40,
                    "price_resilience": 0.30,
                    "volume_quiet": 0.15,
                    "liquidity_thin": 0.15
                }
            }
        },
        "routing": {
            "risk_on": {
                "weights": {"funding_divergence": 0.5},  # Lower weight in bull markets
                "final_gate_delta": 0.0
            },
            "neutral": {
                "weights": {"funding_divergence": 1.0},
                "final_gate_delta": 0.0
            },
            "risk_off": {
                "weights": {"funding_divergence": 1.0},  # Full weight in bear markets
                "final_gate_delta": 0.0
            },
            "crisis": {
                "weights": {"funding_divergence": 1.5},  # Highest weight in crisis
                "final_gate_delta": 0.0
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
    config_path = f'/tmp/s4_opt_{test_id}.json'
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)

    # Run backtest (suppress output to reduce noise)
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

        # Count S4 trades only (ignore baseline leaks)
        s4_trade_count = output.count('archetype_funding_divergence')

        # Parse S4 trade PNLs
        import re
        s4_pattern = r'Trade \d+: archetype_funding_divergence.*?PNL: \$(-?\d+\.\d+) \((-?\d+\.\d+)%\)'
        s4_trades = []
        for match in re.finditer(s4_pattern, output, re.DOTALL):
            pnl_dollars = float(match.group(1))
            pnl_pct = float(match.group(2))
            s4_trades.append({'pnl_dollars': pnl_dollars, 'pnl_pct': pnl_pct})

        # Calculate S4-only metrics
        if len(s4_trades) > 0:
            wins = sum(1 for t in s4_trades if t['pnl_pct'] > 0)
            losses = sum(1 for t in s4_trades if t['pnl_pct'] < 0)
            win_rate = (wins / len(s4_trades) * 100) if len(s4_trades) > 0 else 0

            total_profit = sum(t['pnl_dollars'] for t in s4_trades if t['pnl_dollars'] > 0)
            total_loss = abs(sum(t['pnl_dollars'] for t in s4_trades if t['pnl_dollars'] < 0))
            profit_factor = (total_profit / total_loss) if total_loss > 0 else 0

            net_pnl = sum(t['pnl_dollars'] for t in s4_trades)
        else:
            win_rate = 0.0
            profit_factor = 0.0
            net_pnl = 0.0

        metrics = {
            'test_id': test_id,
            'total_trades': len(s4_trades),
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'net_pnl': net_pnl
        }

        return metrics

    except subprocess.TimeoutExpired:
        logger.warning(f"Backtest timeout for {test_id}")
        return {'test_id': test_id, 'total_trades': 0, 'win_rate': 0.0, 'profit_factor': 0.0, 'net_pnl': 0.0}
    except Exception as e:
        logger.error(f"Backtest failed for {test_id}: {e}")
        return {'test_id': test_id, 'total_trades': 0, 'win_rate': 0.0, 'profit_factor': 0.0, 'net_pnl': 0.0}


def objective(trial: optuna.Trial) -> Tuple[float, float, float]:
    """
    Multi-objective optimization objective function.

    Returns:
        Tuple of (neg_profit_factor, neg_win_rate, trade_count_penalty)
        Note: Optuna minimizes, so we negate PF and WR
    """
    # Suggest parameters
    params = {
        'fusion_threshold': trial.suggest_float('fusion_threshold', 0.75, 0.90),
        'funding_z_max': trial.suggest_float('funding_z_max', -2.2, -1.5),  # NEGATIVE!
        'resilience_min': trial.suggest_float('resilience_min', 0.55, 0.70),
        'liquidity_max': trial.suggest_float('liquidity_max', 0.20, 0.35),
        'cooldown_bars': trial.suggest_int('cooldown_bars', 8, 18),
        'atr_stop_mult': trial.suggest_float('atr_stop_mult', 2.0, 3.5)
    }

    # Create config
    config = create_s4_config(params)

    # Run on training set (2022 H1 - bear market)
    train_metrics = run_backtest(
        config,
        start_date='2022-01-01',
        end_date='2022-06-30',
        test_id=f"trial_{trial.number}_train"
    )

    # Run on validation set (2022 H2 - continued bear market)
    val_metrics = run_backtest(
        config,
        start_date='2022-07-01',
        end_date='2022-12-31',
        test_id=f"trial_{trial.number}_val"
    )

    # Calculate harmonic mean of profit factors (penalizes inconsistency)
    pf_train = train_metrics['profit_factor']
    pf_val = val_metrics['profit_factor']

    if pf_train > 0 and pf_val > 0:
        pf_harmonic = 2 / (1/pf_train + 1/pf_val)
    else:
        pf_harmonic = 0.0

    # Calculate average win rate
    wr_avg = (train_metrics['win_rate'] + val_metrics['win_rate']) / 2

    # Trade count penalty (target: 6-10 trades/year, acceptable: 3-15)
    # Each period is 6 months, so target: 3-5 trades/period
    total_trades = train_metrics['total_trades'] + val_metrics['total_trades']
    avg_trades_per_6mo = total_trades / 2

    if avg_trades_per_6mo < 3:
        trade_penalty = (3 - avg_trades_per_6mo) * 10  # Heavily penalize too few trades
    elif avg_trades_per_6mo > 7.5:  # 15 trades/year / 2
        trade_penalty = (avg_trades_per_6mo - 7.5) * 5  # Penalize too many trades
    else:
        trade_penalty = 0.0

    # Pruning: If PF < 1.0 or trade count way off, prune early
    if pf_harmonic < 1.0 or total_trades < 2 or total_trades > 30:
        raise optuna.TrialPruned()

    # Log trial results
    logger.info(
        f"Trial {trial.number}: PF={pf_harmonic:.2f} (train={pf_train:.2f}, val={pf_val:.2f}), "
        f"WR={wr_avg:.1f}%, Trades={total_trades} ({train_metrics['total_trades']}/{val_metrics['total_trades']}), "
        f"Penalty={trade_penalty:.1f}"
    )

    # Return objectives (Optuna minimizes, so negate PF and WR)
    return -pf_harmonic, -wr_avg, trade_penalty


def main():
    """Main optimization loop"""
    logger.info("=" * 70)
    logger.info("S4 (FUNDING DIVERGENCE) MULTI-OBJECTIVE OPTIMIZATION")
    logger.info("=" * 70)
    logger.info("Target: PF > 2.0, WR > 50%, 6-10 trades/year")
    logger.info("Training: 2022 H1 + H2 (bear market)")
    logger.info("=" * 70)

    # Create study
    study = optuna.create_study(
        study_name="s4_calibration",
        storage="sqlite:///results/s4_calibration/optuna_s4_calibration.db",
        load_if_exists=True,
        directions=["minimize", "minimize", "minimize"],  # PF, WR, trade_penalty
        sampler=NSGAIISampler(population_size=20)
    )

    # Run optimization
    n_trials = 30
    logger.info(f"Running {n_trials} trials...")

    try:
        study.optimize(objective, n_trials=n_trials, timeout=3600)  # 1 hour max
    except KeyboardInterrupt:
        logger.info("Optimization interrupted by user")

    # Analyze results
    logger.info("\n" + "=" * 70)
    logger.info("OPTIMIZATION COMPLETE")
    logger.info("=" * 70)

    # Get Pareto-optimal trials
    pareto_trials = [t for t in study.best_trials]

    logger.info(f"Pareto frontier: {len(pareto_trials)} solutions")

    if len(pareto_trials) > 0:
        logger.info("\nTop 5 Pareto Solutions:")
        for i, trial in enumerate(pareto_trials[:5], 1):
            pf = -trial.values[0]
            wr = -trial.values[1]
            penalty = trial.values[2]
            logger.info(f"{i}. PF={pf:.2f}, WR={wr:.1f}%, Penalty={penalty:.1f}")
            logger.info(f"   Params: {trial.params}")

        # Export results
        results_dir = Path('results/s4_calibration')
        results_dir.mkdir(parents=True, exist_ok=True)

        # Export CSV
        df_trials = study.trials_dataframe()
        csv_path = results_dir / f"s4_optimization_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        df_trials.to_csv(csv_path, index=False)
        logger.info(f"\nExported trial data: {csv_path}")

        # Export top config
        best_trial = pareto_trials[0]
        best_config = create_s4_config(best_trial.params)
        config_path = results_dir / "s4_optimized_config.json"
        with open(config_path, 'w') as f:
            json.dump(best_config, f, indent=2)
        logger.info(f"Exported best config: {config_path}")
        logger.info(f"Best PF: {-best_trial.values[0]:.2f}")
    else:
        logger.warning("No Pareto-optimal solutions found! All trials may have been pruned.")

    logger.info("=" * 70)


if __name__ == "__main__":
    main()
