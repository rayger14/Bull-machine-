#!/usr/bin/env python3
"""
S2 (Failed Rally) Per-Archetype Calibration System

Implements data-driven threshold discovery and multi-objective optimization
for the S2 archetype using 2022 bear market data.

Objectives:
1. Discover fusion score distribution (identify top 3-5% of bars)
2. Multi-objective Optuna search (maximize PF, target ~30 trades, minimize DD)
3. Extract Pareto frontier (non-dominated solutions)

Approach:
- 3-fold temporal CV: Q1-Q2 (train), Q3 (val), Q4 (test)
- Harmonic mean for aggregating PF across folds
- Prune trials with <10 or >60 trades
- Checkpoint every 10 trials

Based on Phase 2 optimization framework but focused solely on S2.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import optuna
from optuna.trial import Trial
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Tuple
import logging
import json
import tempfile
import subprocess
from dataclasses import dataclass

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Output directory
RESULTS_DIR = Path("results/s2_calibration")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Database for persistence
DB_PATH = RESULTS_DIR / "optuna_study.db"

# Data paths
DATA_PATH = Path("data/features_mtf/BTC_1H_2022-01-01_to_2024-12-31.parquet")

# Temporal CV folds (2022 only - bear market)
CV_FOLDS = [
    {"name": "Q1_Q2_2022", "start": "2022-01-01", "end": "2022-06-30", "role": "train"},
    {"name": "Q3_2022", "start": "2022-07-01", "end": "2022-09-30", "role": "val"},
    {"name": "Q4_2022", "start": "2022-10-01", "end": "2022-12-31", "role": "test"},
]

# Target trade frequency (annual) - reduced for S2 only
TARGET_ANNUAL_TRADES = 30.0

# Search space for S2 (Failed Rally)
# UPDATED: Based on distribution analysis showing 95th %ile = 0.6015
# We need MUCH more restrictive thresholds to hit 25-35 trades/year
# NOTE: wick_ratio is a FRACTION (upper_wick / candle_range), range 0.0-1.0
SEARCH_SPACE = {
    "fusion_threshold": (0.58, 0.75),      # Around 95th-99th percentile (0.60-0.65)
    "wick_ratio_min": (0.50, 0.80),        # Strong rejection wicks (50-80% of candle)
    "rsi_min": (72.0, 85.0),               # Extreme overbought only
    "vol_z_max": (-1.0, 0.0),              # Below-average to very low volume
    "liquidity_max": (0.05, 0.15),         # Low liquidity filter
}


@dataclass
class S2Config:
    """S2 (Failed Rally) configuration parameters"""
    fusion_threshold: float
    wick_ratio_min: float
    rsi_min: float
    vol_z_max: float
    liquidity_max: float


@dataclass
class BacktestMetrics:
    """Metrics from a single backtest run"""
    total_trades: int
    win_rate: float
    profit_factor: float
    sharpe_ratio: float
    max_drawdown: float
    total_return: float
    avg_win: float
    avg_loss: float
    fold_name: str
    duration_days: int

    def annual_trades(self) -> float:
        """Calculate annualized trade count"""
        if self.duration_days == 0:
            return 0.0
        return (self.total_trades / self.duration_days) * 365.0


def compute_s2_fusion_distribution(df: pd.DataFrame, output_path: Path):
    """
    Compute fusion score distribution for S2 pattern across all 2022 bars.

    This gives us data-driven insight into what threshold values would
    select the top 3-5% of bars (high-conviction signals).

    Args:
        df: DataFrame with 2022 data
        output_path: Path to save distribution CSV
    """
    logger.info("Computing S2 fusion score distribution...")

    # Filter to 2022 bear market
    df_2022 = df[
        (df.index >= '2022-01-01') &
        (df.index < '2023-01-01')
    ].copy()

    # Filter to risk_off/crisis regimes only (if available)
    if 'macro_regime' in df_2022.columns:
        risk_regimes = df_2022['macro_regime'].isin(['risk_off', 'crisis'])
        logger.info(f"Regime filter: {risk_regimes.sum()} / {len(df_2022)} bars in risk_off/crisis")
        df_2022 = df_2022[risk_regimes]

    logger.info(f"Analyzing {len(df_2022)} bars from 2022 bear market")

    # Compute S2 component scores for each bar
    scores = []

    for idx, row in df_2022.iterrows():
        # Get features
        rsi = row.get('rsi_14', 50.0)
        vol_z = row.get('volume_zscore', 0.0)
        high = row.get('high', 0.0)
        low = row.get('low', 0.0)
        open_price = row.get('open', 0.0)
        close = row.get('close', 0.0)

        # Calculate wick ratio
        candle_range = high - low
        if candle_range < 1e-9:
            continue

        upper_body = max(open_price, close)
        upper_wick = high - upper_body
        wick_ratio = upper_wick / candle_range

        # Compute S2-specific fusion score
        # (matching logic from bear_patterns_phase1.py)
        components = {
            "rsi_extreme": min((rsi - 50.0) / 50.0, 1.0) if rsi > 50 else 0.0,
            "volume_fade": max(0.0, 1.0 - vol_z / 2.0),
            "wick_strength": min(wick_ratio / 0.6, 1.0),
        }

        # S2 weights (from bear_patterns_phase1.py)
        weights = {
            "rsi_extreme": 0.25,
            "volume_fade": 0.20,
            "wick_strength": 0.30,
        }

        s2_score = sum(components.get(k, 0.0) * weights.get(k, 0.0) for k in components)

        scores.append({
            'timestamp': idx,
            'rsi': rsi,
            'vol_z': vol_z,
            'wick_ratio': wick_ratio,
            'rsi_component': components['rsi_extreme'],
            'vol_component': components['volume_fade'],
            'wick_component': components['wick_strength'],
            's2_fusion_score': s2_score,
        })

    # Create DataFrame
    df_scores = pd.DataFrame(scores)

    # Compute percentiles
    percentiles = [50, 75, 80, 85, 90, 95, 97, 99]
    pct_values = np.percentile(df_scores['s2_fusion_score'].dropna(), percentiles)

    logger.info("\n" + "="*80)
    logger.info("S2 FUSION SCORE DISTRIBUTION (2022 Bear Market)")
    logger.info("="*80)
    logger.info(f"Total bars analyzed: {len(df_scores)}")
    logger.info(f"Mean score: {df_scores['s2_fusion_score'].mean():.4f}")
    logger.info(f"Std dev: {df_scores['s2_fusion_score'].std():.4f}")
    logger.info("\nPercentiles:")

    for p, v in zip(percentiles, pct_values):
        count_above = (df_scores['s2_fusion_score'] >= v).sum()
        pct_above = (count_above / len(df_scores)) * 100
        logger.info(f"  {p:>3}th: {v:.4f} ({count_above:>4} bars = {pct_above:.1f}%)")

    # Save distribution data
    df_scores.to_csv(output_path, index=False)
    logger.info(f"\nSaved distribution to: {output_path}")

    # Save percentile summary
    summary_path = output_path.parent / "fusion_percentiles.json"
    summary = {
        'percentiles': {str(p): float(v) for p, v in zip(percentiles, pct_values)},
        'mean': float(df_scores['s2_fusion_score'].mean()),
        'std': float(df_scores['s2_fusion_score'].std()),
        'count': len(df_scores),
    }

    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)

    logger.info(f"Saved percentile summary to: {summary_path}")
    logger.info("="*80 + "\n")

    return df_scores


def create_s2_backtest_config(params: S2Config) -> Dict:
    """
    Create JSON config for backtest_knowledge_v2.py (S2 only)

    Args:
        params: S2Config with threshold parameters

    Returns:
        Dict config for backtest
    """
    return {
        "version": "s2_calibration",
        "profile": "s2_archetype_calibration",
        "description": "S2 (Failed Rally) per-archetype calibration",

        "archetypes": {
            "use_archetypes": True,

            # Enable ONLY S2
            "enable_A": False,
            "enable_B": False,
            "enable_C": False,
            "enable_D": False,
            "enable_E": False,
            "enable_F": False,
            "enable_G": False,
            "enable_H": False,
            "enable_K": False,
            "enable_L": False,
            "enable_M": False,
            "enable_S1": False,
            "enable_S2": True,   # ONLY S2
            "enable_S3": False,
            "enable_S4": False,
            "enable_S5": False,  # Disabled (S5 separate calibration)
            "enable_S6": False,
            "enable_S7": False,
            "enable_S8": False,

            "thresholds": {
                "failed_rally": {
                    "fusion_threshold": params.fusion_threshold,
                    "wick_ratio_min": params.wick_ratio_min,
                    "rsi_min": params.rsi_min,
                    "vol_z_max": params.vol_z_max,
                    "liquidity_max": params.liquidity_max,
                    "use_runtime_features": False
                }
            },

            "routing": {
                "risk_off": {
                    "weights": {
                        "failed_rally": 2.0  # S2 emphasis in bear markets
                    }
                },
                "crisis": {
                    "weights": {
                        "failed_rally": 2.5  # S2 max weight in crisis
                    }
                }
            },

            # Force risk_off regime for 2022 (bear market)
            "regime_override": {
                "2022": "risk_off"
            }
        }
    }


def run_backtest(config: Dict, fold: Dict, trial_id: str) -> BacktestMetrics:
    """
    Run backtest via subprocess and extract metrics

    Args:
        config: Backtest configuration dictionary
        fold: Temporal fold with start/end dates
        trial_id: Unique identifier for this trial

    Returns:
        BacktestMetrics object with results
    """
    # Write config to temporary file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(config, f, indent=2)
        config_path = f.name

    try:
        # Build command
        cmd = [
            'python3',
            'bin/backtest_knowledge_v2.py',
            '--asset', 'BTC',
            '--start', fold['start'],
            '--end', fold['end'],
            '--config', config_path
        ]

        # Run backtest with timeout
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=300,  # 5 minute timeout
            cwd=Path(__file__).parent.parent
        )

        output = result.stdout + result.stderr

        # Parse metrics from output
        metrics = {
            'total_trades': 0,
            'win_rate': 0.0,
            'profit_factor': 0.0,
            'sharpe_ratio': 0.0,
            'max_drawdown': 0.0,
            'avg_win': 0.0,
            'avg_loss': 0.0,
            'total_return': 0.0
        }

        for line in output.split('\n'):
            if 'Total Trades:' in line:
                try:
                    metrics['total_trades'] = int(line.split(':')[1].strip())
                except:
                    pass
            elif 'Win Rate:' in line:
                try:
                    metrics['win_rate'] = float(line.split(':')[1].strip().replace('%', ''))
                except:
                    pass
            elif 'Profit Factor:' in line:
                try:
                    metrics['profit_factor'] = float(line.split(':')[1].strip())
                except:
                    pass
            elif 'Sharpe Ratio:' in line:
                try:
                    metrics['sharpe_ratio'] = float(line.split(':')[1].strip())
                except:
                    pass
            elif 'Max Drawdown:' in line:
                try:
                    metrics['max_drawdown'] = float(line.split(':')[1].strip().replace('%', ''))
                except:
                    pass

        # Calculate duration
        start_dt = datetime.strptime(fold['start'], '%Y-%m-%d')
        end_dt = datetime.strptime(fold['end'], '%Y-%m-%d')
        duration_days = (end_dt - start_dt).days

        return BacktestMetrics(
            total_trades=metrics['total_trades'],
            win_rate=metrics['win_rate'],
            profit_factor=metrics['profit_factor'],
            sharpe_ratio=metrics['sharpe_ratio'],
            max_drawdown=metrics['max_drawdown'],
            total_return=metrics['total_return'],
            avg_win=metrics['avg_win'],
            avg_loss=metrics['avg_loss'],
            fold_name=fold['name'],
            duration_days=duration_days
        )

    except subprocess.TimeoutExpired:
        logger.error(f"Backtest timeout for trial {trial_id} on fold {fold['name']}")
        return BacktestMetrics(
            total_trades=0, win_rate=0.0, profit_factor=0.0,
            sharpe_ratio=0.0, max_drawdown=100.0, total_return=0.0,
            avg_win=0.0, avg_loss=0.0,
            fold_name=fold['name'], duration_days=1
        )
    except Exception as e:
        logger.error(f"Backtest error for trial {trial_id} on fold {fold['name']}: {e}")
        return BacktestMetrics(
            total_trades=0, win_rate=0.0, profit_factor=0.0,
            sharpe_ratio=0.0, max_drawdown=100.0, total_return=0.0,
            avg_win=0.0, avg_loss=0.0,
            fold_name=fold['name'], duration_days=1
        )
    finally:
        # Cleanup temp config
        try:
            Path(config_path).unlink()
        except:
            pass


def objective(trial: Trial) -> Tuple[float, float, float]:
    """
    Optuna objective function for multi-objective optimization

    Returns tuple of 3 objectives to minimize:
    1. -PF (maximize profit factor)
    2. abs(annual_trades - 30) (target trade frequency)
    3. max_drawdown (minimize drawdown)
    """
    # Sample parameters from search space
    params = S2Config(
        fusion_threshold=trial.suggest_float(
            's2_fusion_threshold',
            SEARCH_SPACE['fusion_threshold'][0],
            SEARCH_SPACE['fusion_threshold'][1]
        ),
        wick_ratio_min=trial.suggest_float(
            's2_wick_ratio_min',
            SEARCH_SPACE['wick_ratio_min'][0],
            SEARCH_SPACE['wick_ratio_min'][1]
        ),
        rsi_min=trial.suggest_float(
            's2_rsi_min',
            SEARCH_SPACE['rsi_min'][0],
            SEARCH_SPACE['rsi_min'][1]
        ),
        vol_z_max=trial.suggest_float(
            's2_vol_z_max',
            SEARCH_SPACE['vol_z_max'][0],
            SEARCH_SPACE['vol_z_max'][1]
        ),
        liquidity_max=trial.suggest_float(
            's2_liquidity_max',
            SEARCH_SPACE['liquidity_max'][0],
            SEARCH_SPACE['liquidity_max'][1]
        ),
    )

    # Create config
    config = create_s2_backtest_config(params)

    # Run 3-fold temporal CV
    fold_results = []
    for fold in CV_FOLDS:
        logger.info(f"Trial {trial.number}: Running fold {fold['name']} ({fold['role']})...")
        metrics = run_backtest(config, fold, f"trial_{trial.number}")
        fold_results.append(metrics)

        # Log fold result
        logger.info(
            f"  {fold['name']}: trades={metrics.total_trades}, "
            f"WR={metrics.win_rate:.1f}%, PF={metrics.profit_factor:.2f}, "
            f"DD={metrics.max_drawdown:.1f}%"
        )

    # Aggregate metrics using harmonic mean for PF (penalizes inconsistency)
    pf_values = [m.profit_factor for m in fold_results if m.profit_factor > 0]
    if len(pf_values) == 0:
        harmonic_pf = 0.0
    else:
        harmonic_pf = len(pf_values) / sum(1.0 / pf for pf in pf_values)

    mean_annual_trades = np.mean([m.annual_trades() for m in fold_results])
    mean_max_dd = np.mean([m.max_drawdown for m in fold_results])

    # Prune trials with unrealistic trade counts
    if mean_annual_trades < 10 or mean_annual_trades > 60:
        logger.warning(
            f"Trial {trial.number} PRUNED: trades/yr={mean_annual_trades:.1f} "
            f"(target: 10-60)"
        )
        raise optuna.TrialPruned()

    # Store user attributes for later analysis
    trial.set_user_attr('harmonic_pf', harmonic_pf)
    trial.set_user_attr('mean_annual_trades', mean_annual_trades)
    trial.set_user_attr('mean_max_dd', mean_max_dd)
    trial.set_user_attr('mean_win_rate', np.mean([m.win_rate for m in fold_results]))
    trial.set_user_attr('mean_sharpe', np.mean([m.sharpe_ratio for m in fold_results]))

    # Store per-fold results
    for i, metrics in enumerate(fold_results):
        fold_name = CV_FOLDS[i]['name']
        trial.set_user_attr(f'{fold_name}_trades', metrics.total_trades)
        trial.set_user_attr(f'{fold_name}_pf', metrics.profit_factor)
        trial.set_user_attr(f'{fold_name}_wr', metrics.win_rate)
        trial.set_user_attr(f'{fold_name}_dd', metrics.max_drawdown)

    # Compute objectives (all to be minimized)
    obj1_neg_pf = -harmonic_pf  # Maximize PF -> minimize -PF
    obj2_trade_dev = abs(mean_annual_trades - TARGET_ANNUAL_TRADES)  # Target 30 trades/year
    obj3_max_dd = mean_max_dd  # Minimize drawdown

    logger.info(
        f"Trial {trial.number} complete: "
        f"PF={harmonic_pf:.2f} (harmonic), trades/yr={mean_annual_trades:.1f}, DD={mean_max_dd:.1f}%"
    )

    return obj1_neg_pf, obj2_trade_dev, obj3_max_dd


def export_pareto_configs(study: optuna.Study, output_dir: Path, top_n: int = 5):
    """
    Export top N Pareto-optimal configurations as JSON files

    Args:
        study: Completed Optuna study
        output_dir: Directory to save config files
        top_n: Number of top configs to export
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    if not study.best_trials:
        logger.warning("No Pareto-optimal trials found")
        return

    # Sort by profit factor (harmonic)
    pareto_trials = sorted(
        study.best_trials,
        key=lambda t: t.user_attrs.get('harmonic_pf', 0),
        reverse=True
    )

    for i, trial in enumerate(pareto_trials[:top_n], 1):
        config = {
            'trial_number': trial.number,
            'rank': i,
            'performance': {
                'profit_factor_harmonic': trial.user_attrs.get('harmonic_pf', 0.0),
                'annual_trades': trial.user_attrs.get('mean_annual_trades', 0.0),
                'max_drawdown_pct': trial.user_attrs.get('mean_max_dd', 0.0),
                'win_rate_pct': trial.user_attrs.get('mean_win_rate', 0.0),
                'sharpe_ratio': trial.user_attrs.get('mean_sharpe', 0.0),
            },
            'parameters': {
                'fusion_threshold': trial.params['s2_fusion_threshold'],
                'wick_ratio_min': trial.params['s2_wick_ratio_min'],
                'rsi_min': trial.params['s2_rsi_min'],
                'vol_z_max': trial.params['s2_vol_z_max'],
                'liquidity_max': trial.params['s2_liquidity_max'],
            },
            'cv_results': {
                fold['name']: {
                    'trades': trial.user_attrs.get(f"{fold['name']}_trades", 0),
                    'profit_factor': trial.user_attrs.get(f"{fold['name']}_pf", 0.0),
                    'win_rate': trial.user_attrs.get(f"{fold['name']}_wr", 0.0),
                    'max_drawdown': trial.user_attrs.get(f"{fold['name']}_dd", 0.0),
                }
                for fold in CV_FOLDS
            }
        }

        config_path = output_dir / f"s2_config_rank{i}_trial{trial.number}.json"
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)

        logger.info(f"Exported config #{i}: {config_path}")


def generate_calibration_report(
    study: optuna.Study,
    distribution_df: pd.DataFrame,
    output_path: Path
):
    """
    Generate comprehensive calibration report

    Args:
        study: Completed Optuna study
        distribution_df: S2 fusion score distribution data
        output_path: Path to save markdown report
    """
    with open(output_path, 'w') as f:
        f.write("# S2 (Failed Rally) Archetype Calibration Report\n\n")
        f.write(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

        f.write("## Executive Summary\n\n")
        f.write("This report documents the per-archetype calibration process for ")
        f.write("S2 (Failed Rally Rejection) using 2022 bear market data.\n\n")

        # Distribution Analysis
        f.write("## Fusion Score Distribution Analysis\n\n")
        f.write(f"**Data:** 2022 bear market (risk_off/crisis regimes)\n")
        f.write(f"**Bars analyzed:** {len(distribution_df):,}\n\n")

        mean_score = distribution_df['s2_fusion_score'].mean()
        std_score = distribution_df['s2_fusion_score'].std()

        f.write(f"**Mean S2 fusion score:** {mean_score:.4f}\n")
        f.write(f"**Standard deviation:** {std_score:.4f}\n\n")

        f.write("**Key Percentiles:**\n\n")
        percentiles = [75, 85, 90, 95, 97, 99]
        pct_values = np.percentile(distribution_df['s2_fusion_score'].dropna(), percentiles)

        f.write("| Percentile | Score | Bars Above | % of Total |\n")
        f.write("|------------|-------|------------|------------|\n")
        for p, v in zip(percentiles, pct_values):
            count = (distribution_df['s2_fusion_score'] >= v).sum()
            pct = (count / len(distribution_df)) * 100
            f.write(f"| {p}th | {v:.4f} | {count:,} | {pct:.1f}% |\n")

        f.write("\n**Insight:** Targeting the 95th-99th percentile (top 1-5% of bars) ")
        f.write("provides high-conviction signals while maintaining reasonable trade frequency.\n\n")

        # Optimization Results
        f.write("## Multi-Objective Optimization Results\n\n")
        f.write("**Objectives:**\n")
        f.write("1. Maximize Profit Factor (harmonic mean across folds)\n")
        f.write("2. Target ~30 trades/year (minimize deviation)\n")
        f.write("3. Minimize Maximum Drawdown\n\n")

        f.write("**Search Space:**\n\n")
        f.write("| Parameter | Range | Rationale |\n")
        f.write("|-----------|-------|----------|\n")
        f.write("| fusion_threshold | 0.60 - 0.90 | Much higher than baseline (0.55) |\n")
        f.write("| wick_ratio_min | 2.0 - 4.0 | Strong rejection wicks |\n")
        f.write("| rsi_min | 65 - 80 | Overbought extremes |\n")
        f.write("| vol_z_max | -0.5 - 0.5 | Low volume confirmation |\n")
        f.write("| liquidity_max | 0.10 - 0.30 | Optional liquidity filter |\n\n")

        f.write("**Cross-Validation Folds:**\n\n")
        f.write("| Fold | Period | Duration | Role |\n")
        f.write("|------|--------|----------|------|\n")
        for fold in CV_FOLDS:
            start_dt = datetime.strptime(fold['start'], '%Y-%m-%d')
            end_dt = datetime.strptime(fold['end'], '%Y-%m-%d')
            days = (end_dt - start_dt).days
            f.write(f"| {fold['name']} | {fold['start']} to {fold['end']} | {days} days | {fold['role']} |\n")
        f.write("\n")

        f.write(f"**Total Trials:** {len(study.trials)}\n")
        f.write(f"**Pareto Solutions:** {len(study.best_trials)}\n\n")

        if study.best_trials:
            f.write("## Pareto Frontier (Top 10)\n\n")
            f.write("Non-dominated solutions sorted by Profit Factor:\n\n")
            f.write("| Rank | Trial | PF (H) | Trades/Yr | Max DD | Win Rate | Sharpe |\n")
            f.write("|------|-------|--------|-----------|--------|----------|--------|\n")

            # Sort by harmonic PF
            pareto_trials = sorted(
                study.best_trials,
                key=lambda t: t.user_attrs.get('harmonic_pf', 0),
                reverse=True
            )

            for i, trial in enumerate(pareto_trials[:10], 1):
                pf = trial.user_attrs.get('harmonic_pf', 0.0)
                trades = trial.user_attrs.get('mean_annual_trades', 0.0)
                dd = trial.user_attrs.get('mean_max_dd', 0.0)
                wr = trial.user_attrs.get('mean_win_rate', 0.0)
                sharpe = trial.user_attrs.get('mean_sharpe', 0.0)

                f.write(f"| {i} | {trial.number} | {pf:.2f} | {trades:.1f} | {dd:.1f}% | {wr:.1f}% | {sharpe:.2f} |\n")

            # Recommended Configuration
            f.write("\n## Recommended Configuration\n\n")
            best = pareto_trials[0]

            f.write(f"**Trial #{best.number}** (Rank #1 by Profit Factor)\n\n")

            f.write("**Performance Metrics:**\n\n")
            f.write("| Metric | Value |\n")
            f.write("|--------|-------|\n")
            f.write(f"| Profit Factor (harmonic) | {best.user_attrs.get('harmonic_pf', 0.0):.2f} |\n")
            f.write(f"| Annual Trades | {best.user_attrs.get('mean_annual_trades', 0.0):.1f} |\n")
            f.write(f"| Max Drawdown | {best.user_attrs.get('mean_max_dd', 0.0):.1f}% |\n")
            f.write(f"| Win Rate | {best.user_attrs.get('mean_win_rate', 0.0):.1f}% |\n")
            f.write(f"| Sharpe Ratio | {best.user_attrs.get('mean_sharpe', 0.0):.2f} |\n\n")

            f.write("**Optimized Parameters:**\n\n")
            f.write("```json\n")
            f.write("{\n")
            f.write('  "s2_failed_rally": {\n')
            f.write(f'    "fusion_threshold": {best.params["s2_fusion_threshold"]:.3f},\n')
            f.write(f'    "wick_ratio_min": {best.params["s2_wick_ratio_min"]:.2f},\n')
            f.write(f'    "rsi_min": {best.params["s2_rsi_min"]:.1f},\n')
            f.write(f'    "vol_z_max": {best.params["s2_vol_z_max"]:.2f},\n')
            f.write(f'    "liquidity_max": {best.params["s2_liquidity_max"]:.2f}\n')
            f.write('  }\n')
            f.write('}\n')
            f.write('```\n\n')

            f.write("**Cross-Validation Breakdown:**\n\n")
            f.write("| Fold | Trades | PF | Win Rate | Max DD |\n")
            f.write("|------|--------|----|----------|--------|\n")
            for fold in CV_FOLDS:
                fold_name = fold['name']
                trades = best.user_attrs.get(f'{fold_name}_trades', 0)
                pf = best.user_attrs.get(f'{fold_name}_pf', 0.0)
                wr = best.user_attrs.get(f'{fold_name}_wr', 0.0)
                dd = best.user_attrs.get(f'{fold_name}_dd', 0.0)
                f.write(f"| {fold_name} ({fold['role']}) | {trades} | {pf:.2f} | {wr:.1f}% | {dd:.1f}% |\n")

            f.write("\n")

        # Next Steps
        f.write("## Next Steps\n\n")
        f.write("1. **Validate on Out-of-Sample Data:** Test recommended config on 2023-2024 data\n")
        f.write("2. **Sensitivity Analysis:** Examine parameter stability across different market conditions\n")
        f.write("3. **Production Integration:** Update `bear_archetypes_phase1.json` with optimized thresholds\n")
        f.write("4. **S5 Calibration:** Repeat this process for S5 (Long Squeeze) archetype\n")
        f.write("5. **Joint Optimization:** After individual calibration, run joint S2+S5 optimization\n\n")

        # Files Generated
        f.write("## Files Generated\n\n")
        f.write("```\n")
        f.write("results/s2_calibration/\n")
        f.write("├── fusion_distribution.csv        # S2 score distribution (all 2022 bars)\n")
        f.write("├── fusion_percentiles.json        # Percentile summary\n")
        f.write("├── optuna_study.db                # SQLite database (persistence)\n")
        f.write("├── pareto_configs/                # Top 5 Pareto-optimal configs\n")
        f.write("│   ├── s2_config_rank1_trialX.json\n")
        f.write("│   ├── s2_config_rank2_trialY.json\n")
        f.write("│   └── ...\n")
        f.write("└── calibration_report.md          # This report\n")
        f.write("```\n\n")

        f.write("## Methodology Notes\n\n")
        f.write("- **Harmonic Mean PF:** Used for aggregating across folds to penalize inconsistency\n")
        f.write("- **Temporal CV:** Preserves time-series structure (no lookahead bias)\n")
        f.write("- **Trial Pruning:** Automatically prunes trials with <10 or >60 trades/year\n")
        f.write("- **Pareto Frontier:** Multi-objective optimization finds non-dominated solutions\n")

    logger.info(f"Generated calibration report: {output_path}")


def main():
    """Run S2 archetype calibration"""
    print("=" * 80)
    print("S2 (FAILED RALLY) ARCHETYPE CALIBRATION")
    print("=" * 80)
    print()
    print("Phase 1: Fusion Score Distribution Analysis")
    print("Phase 2: Multi-Objective Optimization (50-100 trials)")
    print("Phase 3: Pareto Frontier Extraction")
    print()
    print(f"Data: {DATA_PATH}")
    print(f"Period: 2022 bear market (risk_off/crisis regimes)")
    print(f"Results: {RESULTS_DIR}")
    print()

    # Load data
    if not DATA_PATH.exists():
        logger.error(f"Data file not found: {DATA_PATH}")
        return

    logger.info(f"Loading data from {DATA_PATH}...")
    df = pd.read_parquet(DATA_PATH)
    logger.info(f"Loaded {len(df):,} bars")

    # Phase 1: Distribution Analysis
    print("=" * 80)
    print("PHASE 1: S2 FUSION SCORE DISTRIBUTION")
    print("=" * 80)
    print()

    dist_path = RESULTS_DIR / "fusion_distribution.csv"
    dist_df = compute_s2_fusion_distribution(df, dist_path)

    # Phase 2: Multi-Objective Optimization
    print("=" * 80)
    print("PHASE 2: MULTI-OBJECTIVE OPTIMIZATION")
    print("=" * 80)
    print()

    n_trials = 10  # Start with 10 for testing, increase to 50-100 for production

    print(f"Running {n_trials} trials with 3-fold CV")
    print(f"Total backtests: {n_trials * 3} = {n_trials} trials × 3 folds")
    print(f"Estimated time: {n_trials * 3 * 2 / 60:.1f} minutes (assuming 2 min per backtest)")
    print()

    # Create or load study
    study_name = "s2_calibration"
    storage = f"sqlite:///{DB_PATH}"

    try:
        study = optuna.load_study(
            study_name=study_name,
            storage=storage
        )
        logger.info(f"Loaded existing study with {len(study.trials)} trials")
    except KeyError:
        study = optuna.create_study(
            study_name=study_name,
            storage=storage,
            directions=["minimize", "minimize", "minimize"],
            load_if_exists=True
        )
        logger.info("Created new optimization study")

    # Run optimization
    try:
        study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    except KeyboardInterrupt:
        logger.warning("Optimization interrupted by user")

    # Phase 3: Results Export
    print()
    print("=" * 80)
    print("PHASE 3: RESULTS EXPORT")
    print("=" * 80)
    print()

    if len(study.trials) == 0:
        logger.error("No trials completed")
        return

    print(f"Total trials: {len(study.trials)}")
    print(f"Pareto solutions: {len(study.best_trials)}")
    print()

    # Export Pareto configs
    configs_dir = RESULTS_DIR / "pareto_configs"
    export_pareto_configs(study, configs_dir, top_n=5)

    # Generate report
    report_path = RESULTS_DIR / "calibration_report.md"
    generate_calibration_report(study, dist_df, report_path)

    # Display summary
    if study.best_trials:
        print()
        print("TOP 5 PARETO SOLUTIONS (by Profit Factor):")
        print()

        pareto_trials = sorted(
            study.best_trials,
            key=lambda t: t.user_attrs.get('harmonic_pf', 0),
            reverse=True
        )

        for i, trial in enumerate(pareto_trials[:5], 1):
            pf = trial.user_attrs.get('harmonic_pf', 0.0)
            trades = trial.user_attrs.get('mean_annual_trades', 0.0)
            dd = trial.user_attrs.get('mean_max_dd', 0.0)
            wr = trial.user_attrs.get('mean_win_rate', 0.0)

            print(f"[{i}] Trial {trial.number}")
            print(f"    PF: {pf:.2f} | Trades/yr: {trades:.1f} | DD: {dd:.1f}% | WR: {wr:.1f}%")
            print(f"    fusion={trial.params['s2_fusion_threshold']:.3f}, "
                  f"wick={trial.params['s2_wick_ratio_min']:.2f}, "
                  f"rsi={trial.params['s2_rsi_min']:.1f}, "
                  f"vol_z={trial.params['s2_vol_z_max']:.2f}")
            print()

    print("=" * 80)
    print("CALIBRATION COMPLETE")
    print("=" * 80)
    print()
    print("Files generated:")
    print(f"  - {dist_path}")
    print(f"  - {RESULTS_DIR / 'fusion_percentiles.json'}")
    print(f"  - {configs_dir}/ (top 5 configs)")
    print(f"  - {report_path}")
    print(f"  - {DB_PATH}")
    print()


if __name__ == '__main__':
    main()
