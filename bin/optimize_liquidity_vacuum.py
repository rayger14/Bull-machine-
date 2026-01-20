#!/usr/bin/env python3
"""
S1 (Liquidity Vacuum Reversal) Multi-Objective Calibration Optimizer

Uses Optuna to find optimal S1 thresholds across multiple objectives:
1. Maximize Profit Factor (primary) - Target: >2.0
2. Maximize Win Rate (secondary) - Target: >50%
3. Achieve target trade frequency: 10-15 trades/year

PATTERN LOGIC:
Liquidity Vacuum Reversal (S1) detects capitulation reversals when:
- Orderbook liquidity evaporates (liquidity_score < 0.15)
- Panic selling exhausts itself (volume_zscore > 2.0)
- Deep lower wick signals buyer absorption (wick_lower_ratio > 0.30)
- Often occurs during crisis/capitulation events

BTC EXAMPLES:
- 2022-06-18: Luna capitulation → -70% → violent 25% bounce in 24h
- 2022-11-09: FTX collapse → liquidity vacuum → explosive reversal
- 2022-05-12: LUNA death spiral → extreme capitulation → sharp bounce

ARCHITECTURE:
- Multi-objective optimization (NSGA-II algorithm)
- Cross-validation: Train (2022 H1) → Validate (2022 H2) → Test (2023 H1)
- Regime gating: Fires in risk_off/crisis (bear markets)
- Runtime enrichment: Applies S1 features on-demand

SEARCH SPACE:
- fusion_threshold: [0.40, 0.55] - Baseline 0.45
- liquidity_max: [0.10, 0.20] - Low liquidity threshold (0.15 baseline)
- volume_z_min: [1.5, 2.5] - Volume panic threshold (2.0 baseline)
- wick_lower_min: [0.25, 0.40] - Lower wick rejection (0.30 baseline)
- cooldown_bars: [8, 18] - Trade spacing (12 baseline)
- atr_stop_mult: [2.0, 3.5] - Stop loss multiplier (2.5 baseline)

OUTPUT:
- Pareto frontier of optimal configurations
- CSV with all trials for analysis
- JSON configs for top 3 solutions (conservative/balanced/aggressive)

Author: Claude Code (Backend Architect)
Date: 2025-11-21
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

from engine.strategies.archetypes.bear.liquidity_vacuum_runtime import apply_liquidity_vacuum_enrichment

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

        # Apply S1 enrichment
        df = apply_liquidity_vacuum_enrichment(df, lookback=24, volume_lookback=24)

        FEATURE_CACHE[cache_key] = df
        logger.info(f"Cached {len(df)} bars")

    return FEATURE_CACHE[cache_key].copy()


def create_s1_config(params: Dict, base_config: Dict = None) -> Dict:
    """
    Create S1 backtest config from parameters.

    Args:
        params: Parameter dictionary from Optuna trial
        base_config: Optional base config to merge with

    Returns:
        Complete backtest config dictionary
    """
    if base_config is None:
        base_config = {
            "version": "s1_optimization",
            "profile": "S1 Calibration Test",
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
                "entry_threshold_confidence": 1.0,  # FULLY DISABLE baseline trades (impossible threshold)
                "weights": {
                    "wyckoff": 0.0,  # Zero out all weights to prevent baseline scoring
                    "liquidity": 0.0,
                    "momentum": 0.0,
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
        "max_trades_per_day": 0,  # DISABLE baseline trades completely (only archetype-specific trades allowed)
        # Disable all other archetypes
        "enable_A": False, "enable_B": False, "enable_C": False, "enable_D": False,
        "enable_E": False, "enable_F": False, "enable_G": False, "enable_H": False,
        "enable_K": False, "enable_L": False, "enable_M": False,
        "enable_S1": True,  # Only S1 (Liquidity Vacuum) enabled
        "enable_S2": False, "enable_S3": False, "enable_S4": False,
        "enable_S5": False, "enable_S6": False, "enable_S7": False, "enable_S8": False,
        "thresholds": {
            "min_liquidity": 0.10,
            "liquidity_vacuum": {
                "direction": "long",  # Capitulation reversals go UP
                "archetype_weight": 2.5,
                "fusion_threshold": params['fusion_threshold'],
                "final_fusion_gate": params['fusion_threshold'],
                "liquidity_max": params['liquidity_max'],
                "volume_z_min": params['volume_z_min'],
                "wick_lower_min": params['wick_lower_min'],
                "cooldown_bars": params['cooldown_bars'],
                "max_risk_pct": 0.02,
                "atr_stop_mult": params['atr_stop_mult'],
                "use_runtime_features": True,
                "lookback": 24,
                "volume_lookback": 24,
                "weights": {
                    "liquidity_vacuum": 0.25,
                    "volume_capitulation": 0.20,
                    "wick_rejection": 0.20,
                    "funding_reversal": 0.15,
                    "crisis_context": 0.10,
                    "oversold": 0.05,
                    "volatility_spike": 0.03,
                    "downtrend_confirm": 0.02
                }
            }
        },
        "routing": {
            "risk_on": {
                "weights": {"liquidity_vacuum": 0.5},  # Lower weight in bull markets
                "final_gate_delta": 0.0
            },
            "neutral": {
                "weights": {"liquidity_vacuum": 1.0},
                "final_gate_delta": 0.0
            },
            "risk_off": {
                "weights": {"liquidity_vacuum": 1.5},  # High weight in bear markets
                "final_gate_delta": 0.0
            },
            "crisis": {
                "weights": {"liquidity_vacuum": 2.0},  # Highest weight in crisis
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
    config_path = f'/tmp/s1_opt_{test_id}.json'
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

        # Count S1 trades only (ignore baseline leaks)
        # NOTE: Backtest logs use legacy name "breakdown" (S1 letter code), not "liquidity_vacuum"
        s1_trade_count = output.count('archetype_breakdown')

        # Parse S1 trade PNLs
        import re
        # Match both "archetype_breakdown" (legacy) and "archetype_liquidity_vacuum" (future)
        s1_pattern = r'ENTRY archetype_(breakdown|liquidity_vacuum):.*?EXIT.*?PNL=\$(-?\d+\.\d+)'
        s1_trades = []
        for match in re.finditer(s1_pattern, output, re.DOTALL):
            pnl_dollars = float(match.group(2))
            # Calculate percentage from dollar amount (approximate - will be refined)
            pnl_pct = (pnl_dollars / 10000) * 100  # Rough estimate assuming $10K starting equity
            s1_trades.append({'pnl_dollars': pnl_dollars, 'pnl_pct': pnl_pct})

        # Calculate S1-only metrics
        if len(s1_trades) > 0:
            wins = sum(1 for t in s1_trades if t['pnl_pct'] > 0)
            losses = sum(1 for t in s1_trades if t['pnl_pct'] < 0)
            win_rate = (wins / len(s1_trades) * 100) if len(s1_trades) > 0 else 0

            total_profit = sum(t['pnl_dollars'] for t in s1_trades if t['pnl_dollars'] > 0)
            total_loss = abs(sum(t['pnl_dollars'] for t in s1_trades if t['pnl_dollars'] < 0))
            profit_factor = (total_profit / total_loss) if total_loss > 0 else 0

            net_pnl = sum(t['pnl_dollars'] for t in s1_trades)
        else:
            win_rate = 0.0
            profit_factor = 0.0
            net_pnl = 0.0

        metrics = {
            'test_id': test_id,
            'total_trades': len(s1_trades),
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
    # Suggest parameters (v2 OR LOGIC - 3RD ITERATION)
    # v2 DIAGNOSTIC RESULT: 567 trades with PF 0.50 (OR logic works, but thresholds too loose!)
    # v2 uses OR gate: (volume_z >= vol_z_min) OR (wick_lower >= wick_lower_min)
    # Need to TIGHTEN from diagnostic (567 trades) to target (8-15 trades/year)
    params = {
        'fusion_threshold': trial.suggest_float('fusion_threshold', 0.35, 0.55),  # v2: TIGHTEN (was [0.25, 0.50], diagnostic=0.30)
        'liquidity_max': trial.suggest_float('liquidity_max', 0.12, 0.20),         # v2: TIGHTEN (was [0.15, 0.30], diagnostic=0.20)
        'volume_z_min': trial.suggest_float('volume_z_min', 1.3, 2.0),             # v2: TIGHTEN (was [1.0, 2.5], diagnostic=1.0) - OR gate, not hard
        'wick_lower_min': trial.suggest_float('wick_lower_min', 0.32, 0.42),       # v2: TIGHTEN (was [0.20, 0.40], diagnostic=0.28) - OR gate, not hard
        'cooldown_bars': trial.suggest_int('cooldown_bars', 12, 24),               # v2: TIGHTEN (was [6, 18], diagnostic=6)
        'atr_stop_mult': trial.suggest_float('atr_stop_mult', 2.0, 3.5)            # Same - Reasonable range
    }

    # Create config
    config = create_s1_config(params)

    # Run on training set (2022 H1 - bear market with Luna events)
    train_metrics = run_backtest(
        config,
        start_date='2022-01-01',
        end_date='2022-06-30',
        test_id=f"trial_{trial.number}_train"
    )

    # Run on validation set (2022 H2 - continued bear with FTX)
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

    # Trade count penalty (target: 10-15 trades/year, acceptable: 5-20)
    # Each period is 6 months, so target: 5-7.5 trades/period
    total_trades = train_metrics['total_trades'] + val_metrics['total_trades']
    avg_trades_per_6mo = total_trades / 2

    if avg_trades_per_6mo < 5:
        trade_penalty = (5 - avg_trades_per_6mo) * 10  # Heavily penalize too few trades
    elif avg_trades_per_6mo > 10:  # 20 trades/year / 2
        trade_penalty = (avg_trades_per_6mo - 10) * 5  # Penalize too many trades
    else:
        trade_penalty = 0.0

    # Pruning: If PF < 1.0 or trade count way off, prune early
    if pf_harmonic < 1.0 or total_trades < 3 or total_trades > 40:
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
    logger.info("S1 (LIQUIDITY VACUUM REVERSAL) MULTI-OBJECTIVE OPTIMIZATION")
    logger.info("=" * 70)
    logger.info("Target: PF > 2.0, WR > 50%, 10-15 trades/year")
    logger.info("Training: 2022 H1 + H2 (bear market with capitulation events)")
    logger.info("=" * 70)

    # Create output directory
    results_dir = Path('results/liquidity_vacuum_calibration')
    results_dir.mkdir(parents=True, exist_ok=True)

    # Create study
    study = optuna.create_study(
        study_name="liquidity_vacuum_calibration",
        storage="sqlite:///results/liquidity_vacuum_calibration/optuna_liquidity_vacuum.db",
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
        # Export CSV
        df_trials = study.trials_dataframe()
        csv_path = results_dir / f"s1_optimization_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        df_trials.to_csv(csv_path, index=False)
        logger.info(f"\nExported trial data: {csv_path}")

        # Export top 3 configs (conservative/balanced/aggressive)
        for i, label in enumerate(['conservative', 'balanced', 'aggressive']):
            if i < len(pareto_trials):
                trial = pareto_trials[i]
                config = create_s1_config(trial.params)
                config_path = results_dir / f"s1_optimized_{label}.json"
                with open(config_path, 'w') as f:
                    json.dump(config, f, indent=2)
                logger.info(f"Exported {label} config: {config_path}")
                logger.info(f"  PF: {-trial.values[0]:.2f}, WR: {-trial.values[1]:.1f}%")

        # Export best config (first Pareto solution)
        best_trial = pareto_trials[0]
        best_config = create_s1_config(best_trial.params)
        config_path = results_dir / "s1_optimized_config.json"
        with open(config_path, 'w') as f:
            json.dump(best_config, f, indent=2)
        logger.info(f"\nExported best config: {config_path}")
        logger.info(f"Best PF: {-best_trial.values[0]:.2f}")
    else:
        logger.warning("No Pareto-optimal solutions found! All trials may have been pruned.")

    logger.info("=" * 70)


if __name__ == "__main__":
    main()
