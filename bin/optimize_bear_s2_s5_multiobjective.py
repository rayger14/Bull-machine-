#!/usr/bin/env python3
"""
Phase 2: Multi-Objective Optimization for S2 (Failed Rally) and S5 (Long Squeeze)

Optimizes 3 objectives using Optuna:
1. Maximize Profit Factor (minimize -PF)
2. Minimize deviation from target trade frequency (32.5 trades/year)
3. Minimize Maximum Drawdown

Uses 3-fold temporal cross-validation:
- H1 2022 (Jan-Jun)
- H2 2022 (Jul-Dec)
- H1 2023 (Jan-Jun)

Based on Phase 1 results:
- S2 search space: fusion [0.40-0.60], wick_ratio [1.8-3.0], rsi [65-78]
- S5 search space: fusion [0.38-0.58], funding_z [1.0-2.2], rsi [65-78], liquidity [0.15-0.32]

Results stored in SQLite database for persistence.
Pareto frontier exported at completion.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import optuna
from optuna.trial import Trial
import subprocess
import json
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Tuple
import sqlite3
import tempfile
import logging
from dataclasses import dataclass, asdict

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Output directory
RESULTS_DIR = Path("results/phase2_optimization")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Database for persistence
DB_PATH = RESULTS_DIR / "optimization_study.db"

# Temporal CV folds
CV_FOLDS = [
    {"name": "H1_2022", "start": "2022-01-01", "end": "2022-06-30"},
    {"name": "H2_2022", "start": "2022-07-01", "end": "2022-12-31"},
    {"name": "H1_2023", "start": "2023-01-01", "end": "2023-06-30"},
]

# Target trade frequency (annual)
TARGET_ANNUAL_TRADES = 32.5
TARGET_DAILY_TRADES = TARGET_ANNUAL_TRADES / 365.0


@dataclass
class OptimizationConfig:
    """Configuration for a single optimization trial"""
    # S2 (Failed Rally) parameters
    s2_fusion_threshold: float
    s2_wick_ratio_min: float
    s2_rsi_min: float

    # S5 (Long Squeeze) parameters
    s5_fusion_threshold: float
    s5_funding_z_min: float
    s5_rsi_min: float
    s5_liquidity_max: float


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

    # Additional metadata
    fold_name: str
    duration_days: int

    def annual_trades(self) -> float:
        """Calculate annualized trade count"""
        if self.duration_days == 0:
            return 0.0
        return (self.total_trades / self.duration_days) * 365.0


def create_backtest_config(params: OptimizationConfig) -> Dict:
    """
    Create JSON config for backtest_knowledge_v2.py

    Based on bear_market_2022_test.json structure with optimized parameters.
    """
    return {
        "version": "phase2_optimization",
        "profile": "bear_market_optimization",
        "description": "Phase 2 multi-objective optimization",

        "archetypes": {
            "use_archetypes": True,

            # Enable only S2 and S5
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
            "enable_S2": True,
            "enable_S3": False,
            "enable_S4": False,
            "enable_S5": True,
            "enable_S6": False,
            "enable_S7": False,
            "enable_S8": False,

            "thresholds": {
                "failed_rally": {
                    "fusion_threshold": params.s2_fusion_threshold,
                    "wick_ratio_min": params.s2_wick_ratio_min,
                    "rsi_min": params.s2_rsi_min,
                    "vol_z_max": 0.5,
                    "use_runtime_features": False
                },
                "long_squeeze": {
                    "fusion_threshold": params.s5_fusion_threshold,
                    "funding_z_min": params.s5_funding_z_min,
                    "rsi_min": params.s5_rsi_min,
                    "liquidity_max": params.s5_liquidity_max
                }
            },

            "routing": {
                "neutral": {
                    "weights": {
                        "failed_rally": 1.5,
                        "long_squeeze": 1.5
                    }
                },
                "risk_off": {
                    "weights": {
                        "failed_rally": 2.0,
                        "long_squeeze": 2.2
                    }
                },
                "crisis": {
                    "weights": {
                        "failed_rally": 2.5,
                        "long_squeeze": 2.8
                    }
                }
            },

            # Force risk_off regime for 2022
            "regime_override": {
                "2022": "risk_off",
                "2023": "risk_off"
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
            elif 'Avg Win:' in line:
                try:
                    # Handle format like "Avg Win: $123.45" or "Avg Win: 123.45"
                    val = line.split(':')[1].strip().replace('$', '')
                    metrics['avg_win'] = float(val)
                except:
                    pass
            elif 'Avg Loss:' in line:
                try:
                    val = line.split(':')[1].strip().replace('$', '')
                    metrics['avg_loss'] = float(val)
                except:
                    pass
            elif 'Total Return:' in line:
                try:
                    val = line.split(':')[1].strip().replace('%', '').replace('$', '')
                    metrics['total_return'] = float(val)
                except:
                    pass

        # Calculate duration
        from datetime import datetime
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
            fold_name=fold['name'], duration_days=0
        )
    except Exception as e:
        logger.error(f"Backtest error for trial {trial_id} on fold {fold['name']}: {e}")
        return BacktestMetrics(
            total_trades=0, win_rate=0.0, profit_factor=0.0,
            sharpe_ratio=0.0, max_drawdown=100.0, total_return=0.0,
            avg_win=0.0, avg_loss=0.0,
            fold_name=fold['name'], duration_days=0
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
    2. abs(annual_trades - 32.5) (target trade frequency)
    3. max_drawdown (minimize drawdown)
    """
    # Sample parameters from search space
    params = OptimizationConfig(
        # S2 parameters
        s2_fusion_threshold=trial.suggest_float('s2_fusion_threshold', 0.40, 0.60),
        s2_wick_ratio_min=trial.suggest_float('s2_wick_ratio_min', 1.8, 3.0),
        s2_rsi_min=trial.suggest_float('s2_rsi_min', 65.0, 78.0),

        # S5 parameters
        s5_fusion_threshold=trial.suggest_float('s5_fusion_threshold', 0.38, 0.58),
        s5_funding_z_min=trial.suggest_float('s5_funding_z_min', 1.0, 2.2),
        s5_rsi_min=trial.suggest_float('s5_rsi_min', 65.0, 78.0),
        s5_liquidity_max=trial.suggest_float('s5_liquidity_max', 0.15, 0.32),
    )

    # Create config
    config = create_backtest_config(params)

    # Run 3-fold temporal CV
    fold_results = []
    for fold in CV_FOLDS:
        logger.info(f"Trial {trial.number}: Running fold {fold['name']}...")
        metrics = run_backtest(config, fold, f"trial_{trial.number}")
        fold_results.append(metrics)

        # Log fold result
        logger.info(
            f"  {fold['name']}: trades={metrics.total_trades}, "
            f"WR={metrics.win_rate:.1f}%, PF={metrics.profit_factor:.2f}, "
            f"DD={metrics.max_drawdown:.1f}%"
        )

    # Aggregate metrics across folds (mean)
    mean_pf = np.mean([m.profit_factor for m in fold_results])
    mean_annual_trades = np.mean([m.annual_trades() for m in fold_results])
    mean_max_dd = np.mean([m.max_drawdown for m in fold_results])

    # Store user attributes for later analysis
    trial.set_user_attr('mean_pf', mean_pf)
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
    obj1_neg_pf = -mean_pf  # Maximize PF -> minimize -PF
    obj2_trade_dev = abs(mean_annual_trades - TARGET_ANNUAL_TRADES)  # Target 32.5 trades/year
    obj3_max_dd = mean_max_dd  # Minimize drawdown

    logger.info(
        f"Trial {trial.number} complete: "
        f"PF={mean_pf:.2f}, trades/yr={mean_annual_trades:.1f}, DD={mean_max_dd:.1f}%"
    )

    return obj1_neg_pf, obj2_trade_dev, obj3_max_dd


def export_pareto_frontier(study: optuna.Study, output_path: Path):
    """
    Export Pareto frontier trials to CSV

    Args:
        study: Completed Optuna study
        output_path: Path to save CSV file
    """
    # Get all trials
    trials = study.get_trials(deepcopy=False, states=[optuna.trial.TrialState.COMPLETE])

    if not trials:
        logger.warning("No completed trials to export")
        return

    # Extract trial data
    records = []
    for trial in trials:
        record = {
            'trial_number': trial.number,
            'obj1_neg_pf': trial.values[0],
            'obj2_trade_dev': trial.values[1],
            'obj3_max_dd': trial.values[2],

            # Derived metrics
            'mean_pf': trial.user_attrs.get('mean_pf', 0.0),
            'mean_annual_trades': trial.user_attrs.get('mean_annual_trades', 0.0),
            'mean_max_dd': trial.user_attrs.get('mean_max_dd', 0.0),
            'mean_win_rate': trial.user_attrs.get('mean_win_rate', 0.0),
            'mean_sharpe': trial.user_attrs.get('mean_sharpe', 0.0),

            # Parameters
            's2_fusion_threshold': trial.params['s2_fusion_threshold'],
            's2_wick_ratio_min': trial.params['s2_wick_ratio_min'],
            's2_rsi_min': trial.params['s2_rsi_min'],
            's5_fusion_threshold': trial.params['s5_fusion_threshold'],
            's5_funding_z_min': trial.params['s5_funding_z_min'],
            's5_rsi_min': trial.params['s5_rsi_min'],
            's5_liquidity_max': trial.params['s5_liquidity_max'],

            # Per-fold results
            'H1_2022_trades': trial.user_attrs.get('H1_2022_trades', 0),
            'H1_2022_pf': trial.user_attrs.get('H1_2022_pf', 0.0),
            'H2_2022_trades': trial.user_attrs.get('H2_2022_trades', 0),
            'H2_2022_pf': trial.user_attrs.get('H2_2022_pf', 0.0),
            'H1_2023_trades': trial.user_attrs.get('H1_2023_trades', 0),
            'H1_2023_pf': trial.user_attrs.get('H1_2023_pf', 0.0),
        }
        records.append(record)

    # Create DataFrame
    df = pd.DataFrame(records)

    # Sort by PF (best first)
    df = df.sort_values('mean_pf', ascending=False)

    # Save to CSV
    df.to_csv(output_path, index=False)
    logger.info(f"Exported {len(df)} trials to {output_path}")

    # Also save Pareto-optimal solutions only
    pareto_trials = [t for t in trials if len(study.best_trials) > 0 and t.number in [bt.number for bt in study.best_trials]]
    if pareto_trials:
        pareto_records = [r for r in records if r['trial_number'] in [t.number for t in pareto_trials]]
        pareto_df = pd.DataFrame(pareto_records)
        pareto_path = output_path.parent / f"{output_path.stem}_pareto{output_path.suffix}"
        pareto_df.to_csv(pareto_path, index=False)
        logger.info(f"Exported {len(pareto_df)} Pareto-optimal trials to {pareto_path}")


def generate_report(study: optuna.Study, output_path: Path):
    """
    Generate optimization report

    Args:
        study: Completed Optuna study
        output_path: Path to save markdown report
    """
    with open(output_path, 'w') as f:
        f.write("# Phase 2: Multi-Objective Optimization Report\n\n")
        f.write(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

        f.write("## Optimization Setup\n\n")
        f.write("**Objectives (all minimized):**\n")
        f.write("1. Maximize Profit Factor (minimize -PF)\n")
        f.write("2. Minimize trade frequency deviation from 32.5 trades/year\n")
        f.write("3. Minimize Maximum Drawdown\n\n")

        f.write("**Search Space:**\n")
        f.write("- S2 fusion_threshold: [0.40, 0.60]\n")
        f.write("- S2 wick_ratio_min: [1.8, 3.0]\n")
        f.write("- S2 rsi_min: [65, 78]\n")
        f.write("- S5 fusion_threshold: [0.38, 0.58]\n")
        f.write("- S5 funding_z_min: [1.0, 2.2]\n")
        f.write("- S5 rsi_min: [65, 78]\n")
        f.write("- S5 liquidity_max: [0.15, 0.32]\n\n")

        f.write("**Cross-Validation Folds:**\n")
        for fold in CV_FOLDS:
            f.write(f"- {fold['name']}: {fold['start']} to {fold['end']}\n")
        f.write("\n")

        f.write("## Results\n\n")
        f.write(f"**Total Trials:** {len(study.trials)}\n")
        f.write(f"**Pareto Solutions:** {len(study.best_trials)}\n\n")

        if study.best_trials:
            f.write("### Pareto Frontier\n\n")
            f.write("| Trial | PF | Trades/Year | Max DD | Win Rate | Sharpe |\n")
            f.write("|-------|-------|-------------|---------|----------|--------|\n")

            # Sort by PF
            pareto_trials = sorted(study.best_trials,
                                  key=lambda t: t.user_attrs.get('mean_pf', 0),
                                  reverse=True)

            for trial in pareto_trials[:10]:  # Top 10
                pf = trial.user_attrs.get('mean_pf', 0.0)
                trades = trial.user_attrs.get('mean_annual_trades', 0.0)
                dd = trial.user_attrs.get('mean_max_dd', 0.0)
                wr = trial.user_attrs.get('mean_win_rate', 0.0)
                sharpe = trial.user_attrs.get('mean_sharpe', 0.0)

                f.write(f"| {trial.number} | {pf:.2f} | {trades:.1f} | {dd:.1f}% | {wr:.1f}% | {sharpe:.2f} |\n")

            f.write("\n### Recommended Configuration\n\n")
            # Choose trial with best PF among Pareto solutions
            best = pareto_trials[0]

            f.write("**Performance:**\n")
            f.write(f"- Profit Factor: {best.user_attrs.get('mean_pf', 0.0):.2f}\n")
            f.write(f"- Annual Trades: {best.user_attrs.get('mean_annual_trades', 0.0):.1f}\n")
            f.write(f"- Max Drawdown: {best.user_attrs.get('mean_max_dd', 0.0):.1f}%\n")
            f.write(f"- Win Rate: {best.user_attrs.get('mean_win_rate', 0.0):.1f}%\n")
            f.write(f"- Sharpe Ratio: {best.user_attrs.get('mean_sharpe', 0.0):.2f}\n\n")

            f.write("**Parameters:**\n")
            f.write("```json\n")
            f.write("{\n")
            f.write('  "s2_failed_rally": {\n')
            f.write(f'    "fusion_threshold": {best.params["s2_fusion_threshold"]:.3f},\n')
            f.write(f'    "wick_ratio_min": {best.params["s2_wick_ratio_min"]:.2f},\n')
            f.write(f'    "rsi_min": {best.params["s2_rsi_min"]:.1f}\n')
            f.write('  },\n')
            f.write('  "s5_long_squeeze": {\n')
            f.write(f'    "fusion_threshold": {best.params["s5_fusion_threshold"]:.3f},\n')
            f.write(f'    "funding_z_min": {best.params["s5_funding_z_min"]:.2f},\n')
            f.write(f'    "rsi_min": {best.params["s5_rsi_min"]:.1f},\n')
            f.write(f'    "liquidity_max": {best.params["s5_liquidity_max"]:.2f}\n')
            f.write('  }\n')
            f.write('}\n')
            f.write('```\n\n')

            f.write("**Cross-Validation Results:**\n\n")
            f.write("| Fold | Trades | PF | Win Rate | Max DD |\n")
            f.write("|------|--------|----|----------|--------|\n")
            for fold in CV_FOLDS:
                fold_name = fold['name']
                trades = best.user_attrs.get(f'{fold_name}_trades', 0)
                pf = best.user_attrs.get(f'{fold_name}_pf', 0.0)
                wr = best.user_attrs.get(f'{fold_name}_wr', 0.0)
                dd = best.user_attrs.get(f'{fold_name}_dd', 0.0)
                f.write(f"| {fold_name} | {trades} | {pf:.2f} | {wr:.1f}% | {dd:.1f}% |\n")

            f.write("\n")

        f.write("## Next Steps\n\n")
        f.write("1. Review Pareto frontier to select optimal tradeoff\n")
        f.write("2. Validate selected configuration on out-of-sample data (H2 2023)\n")
        f.write("3. Update production config files\n")
        f.write("4. Run final backtest with selected parameters\n")

    logger.info(f"Generated report: {output_path}")


def main():
    """Run Phase 2 multi-objective optimization"""
    print("=" * 80)
    print("PHASE 2: MULTI-OBJECTIVE OPTIMIZATION (S2 + S5)")
    print("=" * 80)
    print()
    print("Objectives:")
    print("  1. Maximize Profit Factor")
    print("  2. Target 32.5 trades/year")
    print("  3. Minimize Max Drawdown")
    print()
    print("Search Space:")
    print("  S2: fusion[0.40-0.60], wick[1.8-3.0], rsi[65-78]")
    print("  S5: fusion[0.38-0.58], funding_z[1.0-2.2], rsi[65-78], liq[0.15-0.32]")
    print()
    print("Cross-Validation:")
    for fold in CV_FOLDS:
        print(f"  {fold['name']}: {fold['start']} to {fold['end']}")
    print()
    print(f"Results will be saved to: {RESULTS_DIR}")
    print()

    # Create or load study
    study_name = "phase2_s2_s5_multiobjective"
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
            directions=["minimize", "minimize", "minimize"],  # All objectives minimized
            load_if_exists=True
        )
        logger.info("Created new optimization study")

    # Get number of trials from user or use default
    n_trials = 50  # Default: 50 trials (each runs 3 folds, so 150 backtests total)

    print(f"Running {n_trials} trials (each trial = 3 CV folds = {n_trials * 3} backtests)")
    print(f"Estimated time: {n_trials * 3 * 2 / 60:.1f} minutes (assuming 2 min per backtest)")
    print()

    # Run optimization
    try:
        study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    except KeyboardInterrupt:
        logger.warning("Optimization interrupted by user")

    # Export results
    print()
    print("=" * 80)
    print("OPTIMIZATION COMPLETE")
    print("=" * 80)
    print()

    if len(study.trials) == 0:
        logger.error("No trials completed")
        return

    print(f"Total trials: {len(study.trials)}")
    print(f"Pareto solutions: {len(study.best_trials)}")
    print()

    # Export all results
    all_results_path = RESULTS_DIR / "all_trials.csv"
    export_pareto_frontier(study, all_results_path)

    # Generate report
    report_path = RESULTS_DIR / "optimization_report.md"
    generate_report(study, report_path)

    # Display best solutions
    if study.best_trials:
        print("TOP 5 PARETO SOLUTIONS (by Profit Factor):")
        print()

        pareto_trials = sorted(study.best_trials,
                              key=lambda t: t.user_attrs.get('mean_pf', 0),
                              reverse=True)

        for i, trial in enumerate(pareto_trials[:5], 1):
            pf = trial.user_attrs.get('mean_pf', 0.0)
            trades = trial.user_attrs.get('mean_annual_trades', 0.0)
            dd = trial.user_attrs.get('mean_max_dd', 0.0)
            wr = trial.user_attrs.get('mean_win_rate', 0.0)

            print(f"[{i}] Trial {trial.number}")
            print(f"    PF: {pf:.2f} | Trades/yr: {trades:.1f} | DD: {dd:.1f}% | WR: {wr:.1f}%")
            print(f"    S2: fusion={trial.params['s2_fusion_threshold']:.3f}, "
                  f"wick={trial.params['s2_wick_ratio_min']:.2f}, "
                  f"rsi={trial.params['s2_rsi_min']:.1f}")
            print(f"    S5: fusion={trial.params['s5_fusion_threshold']:.3f}, "
                  f"funding_z={trial.params['s5_funding_z_min']:.2f}, "
                  f"rsi={trial.params['s5_rsi_min']:.1f}, "
                  f"liq={trial.params['s5_liquidity_max']:.2f}")
            print()

    print("Files generated:")
    print(f"  - {all_results_path}")
    print(f"  - {all_results_path.parent / (all_results_path.stem + '_pareto.csv')}")
    print(f"  - {report_path}")
    print(f"  - {DB_PATH} (SQLite database)")
    print()


if __name__ == '__main__':
    main()
