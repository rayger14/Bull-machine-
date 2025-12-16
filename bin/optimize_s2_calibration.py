#!/usr/bin/env python3
"""
S2 (Failed Rally) Multi-Objective Optuna Calibrator

Data-driven optimization using empirical distribution analysis to set search ranges.

Multi-Objective Function:
  1. Maximize Profit Factor (harmonic mean across folds)
  2. Target ~10 trades/year (minimize deviation)
  3. Minimize Maximum Drawdown

Features:
- 3-fold temporal CV: 2022 H1 (train), H2 (validate), 2023 H1 (test)
- Regime gating: Only backtest in risk_off/crisis regimes
- Pareto frontier extraction
- SQLite persistence for resume capability
- Automatic pruning of unrealistic configs

Usage:
    # Run distribution analysis first
    python3 bin/analyze_s2_distribution.py

    # Then run optimization
    python3 bin/optimize_s2_calibration.py --trials 50 --timeout 7200

    # Resume existing study
    python3 bin/optimize_s2_calibration.py --trials 50 --resume
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import optuna
from optuna.trial import Trial
import pandas as pd
import numpy as np
import json
import logging
import subprocess
import tempfile
import argparse
from datetime import datetime
from typing import Dict, Tuple, List
from dataclasses import dataclass

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s'
)
logger = logging.getLogger(__name__)

# Output directory
RESULTS_DIR = Path("results/s2_calibration")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Database for Optuna persistence
DB_PATH = RESULTS_DIR / "optuna_s2_calibration.db"
STUDY_NAME = "s2_failed_rally_calibration_v1"

# Temporal CV folds (2022 bear + 2023 H1 test)
CV_FOLDS = [
    {"name": "2022_H1", "start": "2022-01-01", "end": "2022-06-30", "role": "train"},
    {"name": "2022_H2", "start": "2022-07-01", "end": "2022-12-31", "role": "validate"},
    {"name": "2023_H1", "start": "2023-01-01", "end": "2023-06-30", "role": "test"},
]

# Target metrics
TARGET_ANNUAL_TRADES = 10.0
MIN_PROFIT_FACTOR = 1.3
MAX_DRAWDOWN_PCT = 15.0


@dataclass
class S2Parameters:
    """S2 archetype parameters"""
    fusion_threshold: float
    wick_ratio_min: float
    rsi_min: float
    volume_z_max: float
    liquidity_max: float
    cooldown_bars: int


@dataclass
class FoldMetrics:
    """Metrics from a single backtest fold"""
    fold_name: str
    total_trades: int
    win_rate: float
    profit_factor: float
    sharpe_ratio: float
    max_drawdown: float
    total_return: float
    duration_days: int

    def annual_trades(self) -> float:
        """Calculate annualized trade count"""
        if self.duration_days == 0:
            return 0.0
        return (self.total_trades / self.duration_days) * 365.0


def load_search_ranges() -> Dict:
    """
    Load data-driven search ranges from distribution analysis.

    Falls back to conservative defaults if analysis not run yet.
    """
    percentiles_path = RESULTS_DIR / "fusion_percentiles_2022.json"

    if percentiles_path.exists():
        logger.info(f"Loading search ranges from: {percentiles_path}")
        with open(percentiles_path, 'r') as f:
            data = json.load(f)
            return data['recommended_search_ranges']
    else:
        logger.warning("Distribution analysis not found, using conservative defaults")
        logger.warning("Run bin/analyze_s2_distribution.py first for better ranges")

        return {
            'fusion_threshold': [0.65, 0.85],
            'wick_ratio_min': [0.46, 0.57],  # FIXED: ratio is [0,1], not [2,4]
            'rsi_min': [75.0, 85.0],
            'volume_z_max': [-2.0, 0.0],
            'liquidity_max': [0.05, 0.25],
            'cooldown_bars': [4, 20],
        }


def create_s2_backtest_config(params: S2Parameters) -> Dict:
    """
    Create backtest configuration for S2-only testing.

    Args:
        params: S2 parameters

    Returns:
        Config dict for backtest_knowledge_v2.py
    """
    return {
        "version": "s2_calibration_v1",
        "profile": "s2_optuna_calibration",
        "description": f"S2 calibration: fusion={params.fusion_threshold:.3f}, wick={params.wick_ratio_min:.2f}",

        "adaptive_fusion": True,

        "regime_classifier": {
            "model_path": "models/regime_classifier_gmm.pkl",
            "feature_order": [
                "VIX", "DXY", "MOVE", "YIELD_2Y", "YIELD_10Y",
                "USDT.D", "BTC.D", "TOTAL", "TOTAL2",
                "funding", "oi", "rv_20d", "rv_60d"
            ],
            "zero_fill_missing": False,
            "regime_override": {
                "2022": "risk_off",  # Force bear market classification
                "2023-01": "neutral",
                "2023-02": "neutral",
                "2023-03": "neutral",
                "2023-04": "risk_on",
                "2023-05": "risk_on",
                "2023-06": "risk_on",
            }
        },

        "ml_filter": {
            "enabled": False
        },

        "fusion": {
            "entry_threshold_confidence": 0.99,  # DISABLE baseline trades, S2-only calibration
            "weights": {
                "wyckoff": 0.35,
                "liquidity": 0.30,
                "momentum": 0.35,
                "smc": 0.0
            }
        },

        "archetypes": {
            "use_archetypes": True,
            "max_trades_per_day": 8,

            # ONLY S2 enabled
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
            "enable_S2": True,  # <-- ONLY S2
            "enable_S3": False,
            "enable_S4": False,
            "enable_S5": False,
            "enable_S6": False,
            "enable_S7": False,
            "enable_S8": False,

            "thresholds": {
                "min_liquidity": params.liquidity_max,
                "failed_rally": {
                    "direction": "short",
                    "archetype_weight": 2.0,
                    "fusion_threshold": params.fusion_threshold,
                    "final_fusion_gate": params.fusion_threshold,
                    "cooldown_bars": params.cooldown_bars,
                    "max_risk_pct": 0.015,
                    "atr_stop_mult": 2.0,
                    "wick_ratio_min": params.wick_ratio_min,
                    "rsi_min": params.rsi_min,
                    "volume_z_max": params.volume_z_max,
                    "require_rsi_divergence": False,
                    "use_runtime_features": True,  # Enable S2 runtime enrichment

                    "weights": {
                        "ob_retest": 0.25,
                        "wick_rejection": 0.25,
                        "rsi_signal": 0.20,
                        "volume_fade": 0.15,
                        "tf4h_confirm": 0.15
                    }
                }
            },

            "routing": {
                "risk_on": {
                    "weights": {"failed_rally": 0.0},  # Disable in bull markets
                    "final_gate_delta": 0.0
                },
                "neutral": {
                    "weights": {"failed_rally": 0.5},  # Reduced in neutral
                    "final_gate_delta": 0.0
                },
                "risk_off": {
                    "weights": {"failed_rally": 2.0},  # Full weight in bear
                    "final_gate_delta": 0.02
                },
                "crisis": {
                    "weights": {"failed_rally": 2.5},  # Max weight in crisis
                    "final_gate_delta": 0.04
                }
            },

            "exits": {
                "failed_rally": {
                    "enable_trail": True,
                    "trail_atr_mult": 1.5,
                    "time_limit_hours": 48
                }
            }
        },

        "context": {
            "crisis_fuse": {"enabled": False}
        },

        "risk": {
            "base_risk_pct": 0.015,
            "max_position_size_pct": 0.15,
            "max_portfolio_risk_pct": 0.08
        }
    }


def run_backtest(config: Dict, fold: Dict, trial_num: int) -> FoldMetrics:
    """
    Run backtest subprocess and extract metrics.

    Args:
        config: Backtest configuration
        fold: Temporal fold definition
        trial_num: Trial number for logging

    Returns:
        FoldMetrics with results
    """
    # Write config to temp file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(config, f, indent=2)
        config_path = f.name

    try:
        # Run backtest
        cmd = [
            'python3',
            'bin/backtest_knowledge_v2.py',
            '--asset', 'BTC',
            '--start', fold['start'],
            '--end', fold['end'],
            '--config', config_path
        ]

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=300,  # 5 min timeout
            cwd=Path(__file__).parent.parent
        )

        output = result.stdout + result.stderr

        # Parse metrics
        metrics = {
            'total_trades': 0,
            'win_rate': 0.0,
            'profit_factor': 0.0,
            'sharpe_ratio': 0.0,
            'max_drawdown': 0.0,
            'total_return': 0.0
        }

        for line in output.split('\n'):
            try:
                if 'Total Trades:' in line:
                    metrics['total_trades'] = int(line.split(':')[1].strip())
                elif 'Win Rate:' in line:
                    metrics['win_rate'] = float(line.split(':')[1].strip().replace('%', ''))
                elif 'Profit Factor:' in line:
                    metrics['profit_factor'] = float(line.split(':')[1].strip())
                elif 'Sharpe Ratio:' in line:
                    metrics['sharpe_ratio'] = float(line.split(':')[1].strip())
                elif 'Max Drawdown:' in line:
                    metrics['max_drawdown'] = float(line.split(':')[1].strip().replace('%', ''))
                elif 'Total Return:' in line:
                    val = line.split(':')[1].strip().replace('%', '').replace('$', '').replace(',', '')
                    metrics['total_return'] = float(val)
            except (IndexError, ValueError):
                continue

        # Calculate duration
        start_dt = datetime.strptime(fold['start'], '%Y-%m-%d')
        end_dt = datetime.strptime(fold['end'], '%Y-%m-%d')
        duration_days = (end_dt - start_dt).days

        return FoldMetrics(
            fold_name=fold['name'],
            total_trades=metrics['total_trades'],
            win_rate=metrics['win_rate'],
            profit_factor=metrics['profit_factor'],
            sharpe_ratio=metrics['sharpe_ratio'],
            max_drawdown=metrics['max_drawdown'],
            total_return=metrics['total_return'],
            duration_days=duration_days
        )

    except subprocess.TimeoutExpired:
        logger.error(f"Backtest timeout for trial {trial_num}, fold {fold['name']}")
        return FoldMetrics(
            fold_name=fold['name'],
            total_trades=0,
            win_rate=0.0,
            profit_factor=0.0,
            sharpe_ratio=0.0,
            max_drawdown=100.0,
            total_return=0.0,
            duration_days=1
        )

    except Exception as e:
        logger.error(f"Backtest error: {e}")
        return FoldMetrics(
            fold_name=fold['name'],
            total_trades=0,
            win_rate=0.0,
            profit_factor=0.0,
            sharpe_ratio=0.0,
            max_drawdown=100.0,
            total_return=0.0,
            duration_days=1
        )

    finally:
        # Cleanup temp file
        try:
            Path(config_path).unlink()
        except:
            pass


def objective(trial: Trial) -> Tuple[float, float, float]:
    """
    Optuna objective function for multi-objective optimization.

    Returns:
        Tuple of 3 objectives to MINIMIZE:
        1. -PF (maximize profit factor)
        2. abs(annual_trades - target) (target trade frequency)
        3. max_drawdown (minimize drawdown)
    """
    # Load search ranges
    search_ranges = load_search_ranges()

    # Sample parameters
    params = S2Parameters(
        fusion_threshold=trial.suggest_float(
            'fusion_threshold',
            search_ranges['fusion_threshold'][0],
            search_ranges['fusion_threshold'][1]
        ),
        wick_ratio_min=trial.suggest_float(
            'wick_ratio_min',
            search_ranges['wick_ratio_min'][0],
            search_ranges['wick_ratio_min'][1]
        ),
        rsi_min=trial.suggest_float(
            'rsi_min',
            search_ranges['rsi_min'][0],
            search_ranges['rsi_min'][1]
        ),
        volume_z_max=trial.suggest_float(
            'volume_z_max',
            search_ranges['volume_z_max'][0],
            search_ranges['volume_z_max'][1]
        ),
        liquidity_max=trial.suggest_float(
            'liquidity_max',
            search_ranges['liquidity_max'][0],
            search_ranges['liquidity_max'][1]
        ),
        cooldown_bars=trial.suggest_int(
            'cooldown_bars',
            int(search_ranges['cooldown_bars'][0]),
            int(search_ranges['cooldown_bars'][1])
        ),
    )

    # Create config
    config = create_s2_backtest_config(params)

    # Run 3-fold CV
    fold_results: List[FoldMetrics] = []

    for fold in CV_FOLDS:
        logger.info(f"Trial {trial.number}: Running {fold['name']} ({fold['role']})...")

        metrics = run_backtest(config, fold, trial.number)
        fold_results.append(metrics)

        logger.info(
            f"  {fold['name']}: trades={metrics.total_trades}, "
            f"WR={metrics.win_rate:.1f}%, PF={metrics.profit_factor:.2f}, "
            f"DD={metrics.max_drawdown:.1f}%"
        )

    # Compute aggregated metrics
    # 1. Harmonic mean PF (penalizes inconsistency)
    pf_values = [m.profit_factor for m in fold_results if m.profit_factor > 0]
    if len(pf_values) == 0:
        harmonic_pf = 0.0
    else:
        harmonic_pf = len(pf_values) / sum(1.0 / pf for pf in pf_values)

    # 2. Mean annual trades
    mean_annual_trades = np.mean([m.annual_trades() for m in fold_results])

    # 3. Mean max drawdown
    mean_max_dd = np.mean([m.max_drawdown for m in fold_results])

    # 4. Other metrics
    mean_win_rate = np.mean([m.win_rate for m in fold_results])
    mean_sharpe = np.mean([m.sharpe_ratio for m in fold_results])

    # Prune trials with unrealistic metrics
    if mean_annual_trades < 3 or mean_annual_trades > 30:
        logger.warning(
            f"Trial {trial.number} PRUNED: trades/yr={mean_annual_trades:.1f} "
            f"(target: 3-30)"
        )
        raise optuna.TrialPruned()

    if harmonic_pf < 0.8:
        logger.warning(
            f"Trial {trial.number} PRUNED: PF={harmonic_pf:.2f} (min: 0.8)"
        )
        raise optuna.TrialPruned()

    # Store user attributes for analysis
    trial.set_user_attr('harmonic_pf', harmonic_pf)
    trial.set_user_attr('mean_annual_trades', mean_annual_trades)
    trial.set_user_attr('mean_max_dd', mean_max_dd)
    trial.set_user_attr('mean_win_rate', mean_win_rate)
    trial.set_user_attr('mean_sharpe', mean_sharpe)

    # Store per-fold results
    for metrics in fold_results:
        trial.set_user_attr(f'{metrics.fold_name}_trades', metrics.total_trades)
        trial.set_user_attr(f'{metrics.fold_name}_pf', metrics.profit_factor)
        trial.set_user_attr(f'{metrics.fold_name}_wr', metrics.win_rate)
        trial.set_user_attr(f'{metrics.fold_name}_dd', metrics.max_drawdown)

    # Compute objectives (all to MINIMIZE)
    obj1_neg_pf = -harmonic_pf
    obj2_trade_deviation = abs(mean_annual_trades - TARGET_ANNUAL_TRADES)
    obj3_max_dd = mean_max_dd

    logger.info(
        f"Trial {trial.number} complete: "
        f"PF={harmonic_pf:.2f}, trades/yr={mean_annual_trades:.1f}, DD={mean_max_dd:.1f}%"
    )

    return obj1_neg_pf, obj2_trade_deviation, obj3_max_dd


def export_pareto_frontier(study: optuna.Study) -> pd.DataFrame:
    """
    Extract and export Pareto frontier to CSV.

    Args:
        study: Completed Optuna study

    Returns:
        DataFrame with Pareto solutions
    """
    if not study.best_trials:
        logger.warning("No Pareto solutions found")
        return pd.DataFrame()

    # Extract Pareto trials
    pareto_data = []

    for trial in study.best_trials:
        data = {
            'trial_number': trial.number,
            'harmonic_pf': trial.user_attrs.get('harmonic_pf', 0.0),
            'annual_trades': trial.user_attrs.get('mean_annual_trades', 0.0),
            'max_drawdown': trial.user_attrs.get('mean_max_dd', 0.0),
            'win_rate': trial.user_attrs.get('mean_win_rate', 0.0),
            'sharpe_ratio': trial.user_attrs.get('mean_sharpe', 0.0),
            'fusion_threshold': trial.params['fusion_threshold'],
            'wick_ratio_min': trial.params['wick_ratio_min'],
            'rsi_min': trial.params['rsi_min'],
            'volume_z_max': trial.params['volume_z_max'],
            'liquidity_max': trial.params['liquidity_max'],
            'cooldown_bars': trial.params['cooldown_bars'],
        }

        # Add per-fold metrics
        for fold in CV_FOLDS:
            fold_name = fold['name']
            data[f'{fold_name}_trades'] = trial.user_attrs.get(f'{fold_name}_trades', 0)
            data[f'{fold_name}_pf'] = trial.user_attrs.get(f'{fold_name}_pf', 0.0)

        pareto_data.append(data)

    df = pd.DataFrame(pareto_data)

    # Sort by PF descending
    df = df.sort_values('harmonic_pf', ascending=False)

    # Save to CSV
    output_path = RESULTS_DIR / "pareto_frontier_top10.csv"
    df.head(10).to_csv(output_path, index=False)
    logger.info(f"Exported Pareto frontier: {output_path}")

    return df


def main():
    """Run S2 calibration optimization"""
    parser = argparse.ArgumentParser(description='S2 Optuna Calibration')
    parser.add_argument('--trials', type=int, default=50, help='Number of trials')
    parser.add_argument('--timeout', type=int, default=7200, help='Timeout in seconds (default: 2 hours)')
    parser.add_argument('--resume', action='store_true', help='Resume existing study')
    args = parser.parse_args()

    print("="*80)
    print("S2 (FAILED RALLY) MULTI-OBJECTIVE CALIBRATION")
    print("="*80)
    print()
    print(f"Trials: {args.trials}")
    print(f"Timeout: {args.timeout}s ({args.timeout/3600:.1f} hours)")
    print(f"Database: {DB_PATH}")
    print()
    print("Objectives:")
    print("  1. Maximize Profit Factor (harmonic mean)")
    print(f"  2. Target {TARGET_ANNUAL_TRADES} trades/year")
    print(f"  3. Minimize Drawdown")
    print()
    print("CV Folds:")
    for fold in CV_FOLDS:
        print(f"  - {fold['name']}: {fold['start']} to {fold['end']} ({fold['role']})")
    print()

    # Load or create study
    storage = f"sqlite:///{DB_PATH}"

    try:
        study = optuna.load_study(
            study_name=STUDY_NAME,
            storage=storage
        )
        logger.info(f"Loaded existing study with {len(study.trials)} trials")

        if not args.resume:
            logger.warning("Use --resume to continue existing study or delete DB to start fresh")
            return 1

    except KeyError:
        study = optuna.create_study(
            study_name=STUDY_NAME,
            storage=storage,
            directions=["minimize", "minimize", "minimize"],  # 3 objectives
            load_if_exists=True
        )
        logger.info("Created new optimization study")

    # Run optimization
    print("\nStarting optimization...")
    print("Press Ctrl+C to interrupt and save progress")
    print()

    try:
        study.optimize(
            objective,
            n_trials=args.trials,
            timeout=args.timeout,
            show_progress_bar=True
        )
    except KeyboardInterrupt:
        logger.warning("\nOptimization interrupted by user")

    # Export results
    print("\n" + "="*80)
    print("OPTIMIZATION COMPLETE")
    print("="*80)
    print()
    print(f"Total trials: {len(study.trials)}")
    print(f"Pareto solutions: {len(study.best_trials)}")
    print()

    if len(study.trials) == 0:
        logger.error("No trials completed")
        return 1

    # Export Pareto frontier
    df_pareto = export_pareto_frontier(study)

    if len(df_pareto) > 0:
        print("Top 10 Pareto Solutions (by Profit Factor):")
        print()
        print(df_pareto.head(10)[
            ['trial_number', 'harmonic_pf', 'annual_trades', 'max_drawdown', 'win_rate']
        ].to_string(index=False))
        print()

        # Recommended config
        best = df_pareto.iloc[0]
        print("Recommended Configuration (Rank #1):")
        print(f"  Trial: {best['trial_number']}")
        print(f"  PF: {best['harmonic_pf']:.2f}")
        print(f"  Annual Trades: {best['annual_trades']:.1f}")
        print(f"  Max DD: {best['max_drawdown']:.1f}%")
        print(f"  Win Rate: {best['win_rate']:.1f}%")
        print()
        print("  Parameters:")
        print(f"    fusion_threshold: {best['fusion_threshold']:.3f}")
        print(f"    wick_ratio_min: {best['wick_ratio_min']:.2f}")
        print(f"    rsi_min: {best['rsi_min']:.1f}")
        print(f"    volume_z_max: {best['volume_z_max']:.2f}")
        print(f"    liquidity_max: {best['liquidity_max']:.2f}")
        print(f"    cooldown_bars: {best['cooldown_bars']}")
        print()

    print(f"Database saved: {DB_PATH}")
    print(f"Pareto frontier: {RESULTS_DIR / 'pareto_frontier_top10.csv'}")
    print()
    print("Next step: python3 bin/generate_s2_configs.py")

    return 0


if __name__ == '__main__':
    sys.exit(main())
