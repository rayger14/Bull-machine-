#!/usr/bin/env python3
"""
Bear Frontier v10 - Map PF/DD Pareto Curve (2022-2023)

Purpose:
  Discover defensive performance ceiling for bear/sideways markets
  Map complete PF vs DD tradeoff for capital preservation
  Use multi-fold CV + harmonic mean for robustness

Search Space:
  - Fusion weights (Dirichlet-sampled): wyckoff/liquidity/momentum/temporal
  - Archetype thresholds: B/C/H/K/L fusion thresholds
  - Exits: trail_atr, max_bars, range/trend stop factors
  - Sizing: size_min, size_max
  - Global gates: final_fusion_floor, min_liquidity

Objective:
  Maximize PNL + PF_bonus - DD_penalty (defensive weights)
  Record full Pareto frontier for later analysis

Usage:
  python3 bin/optuna_bear_frontier_v10.py --trials 240 --output reports/bear_frontier_v10
"""

import argparse
import json
import optuna
import subprocess
import tempfile
import re
import numpy as np
from pathlib import Path
from datetime import datetime

# ─────────────────────────────────────────────────────────────────────────────
# Bear Folds (2022-2023 defensive periods)
# ─────────────────────────────────────────────────────────────────────────────
FOLDS = {
    'H1_2022': ('2022-01-01', '2022-06-30'),  # Luna crash
    'H2_2022': ('2022-07-01', '2022-12-31'),  # FTX collapse
    'H1_2023': ('2023-01-01', '2023-06-30'),  # Banking crisis
}

# ─────────────────────────────────────────────────────────────────────────────
# Helper: Run backtest and extract metrics
# ─────────────────────────────────────────────────────────────────────────────
def run_backtest(config_path: str, start_date: str, end_date: str, asset: str = "BTC"):
    """Run backtest and extract metrics."""
    cmd = [
        "python3", "bin/backtest_knowledge_v2.py",
        "--asset", asset,
        "--start", start_date,
        "--end", end_date,
        "--config", config_path
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300, check=False)
        output = result.stdout + result.stderr

        metrics = {
            'pnl': 0.0,
            'trades': 0,
            'roi': 0.0,
            'win_rate': 0.0,
            'drawdown': 0.0,
            'profit_factor': 0.0,
        }

        # Parse metrics
        pnl_match = re.search(r'Total PNL:\s+\$?([-\d,\.]+)', output)
        if pnl_match:
            metrics['pnl'] = float(pnl_match.group(1).replace(',', ''))

        trades_match = re.search(r'Total Trades:\s+(\d+)', output)
        if trades_match:
            metrics['trades'] = int(trades_match.group(1))

        roi_match = re.search(r'ROI:\s+([-\d\.]+)%', output)
        if roi_match:
            metrics['roi'] = float(roi_match.group(1))

        wr_match = re.search(r'Win Rate:\s+([\d\.]+)%', output)
        if wr_match:
            metrics['win_rate'] = float(wr_match.group(1))

        dd_match = re.search(r'Max Drawdown:\s+([\d\.]+)%', output)
        if dd_match:
            metrics['drawdown'] = float(dd_match.group(1))

        pf_match = re.search(r'Profit Factor:\s+([\d\.]+)', output)
        if pf_match:
            metrics['profit_factor'] = float(pf_match.group(1))

        return metrics

    except Exception as e:
        print(f"Backtest failed: {e}")
        return None


# ─────────────────────────────────────────────────────────────────────────────
# Scoring Function (Defensive - emphasis on capital preservation)
# ─────────────────────────────────────────────────────────────────────────────
def compute_fold_score(metrics):
    """
    Score = PNL + PF_bonus - DD_penalty (defensive weights)

    Bear market scoring:
    - Higher DD penalty (preserve capital)
    - Lower PF bonus (focus on consistency)
    - Require more trades for confidence
    """
    if metrics is None or metrics['trades'] == 0:
        return -9999.0

    pnl = metrics['pnl']
    pf = metrics['profit_factor']
    dd = metrics['drawdown']
    trades = metrics['trades']

    score = pnl

    # PF bonus (more conservative cap for bear)
    pf_capped = min(pf, 4.0)
    pf_bonus = pf_capped * 100
    score += pf_bonus

    # DD penalty (much steeper for bear - capital preservation)
    if dd > 6.0:
        dd_penalty = ((dd - 6.0) ** 2.2) * 200
        score -= dd_penalty
    elif dd > 3.0:
        dd_penalty = ((dd - 3.0) ** 1.5) * 80
        score -= dd_penalty

    # Undertrade penalty (need more data in bear)
    if trades < 12:
        undertrade_penalty = (12 - trades) ** 2 * 60
        score -= undertrade_penalty

    return score


def harmonic_mean(scores):
    """Harmonic mean - punishes imbalanced performance."""
    if len(scores) == 0:
        return -9999.0

    min_score = min(scores)
    if min_score <= 0:
        shift = abs(min_score) + 1
        scores_shifted = [s + shift for s in scores]
    else:
        scores_shifted = scores
        shift = 0

    h_mean = len(scores_shifted) / sum(1.0 / s for s in scores_shifted)
    return h_mean - shift


# ─────────────────────────────────────────────────────────────────────────────
# Objective Function
# ─────────────────────────────────────────────────────────────────────────────
def objective(trial: optuna.Trial, base_config: dict, asset: str):
    """
    Optuna objective - maps PF/DD space for bear markets.
    """
    cfg = json.loads(json.dumps(base_config))  # Deep copy

    # ─── Fusion Weights (Conservative for bear) ───
    alpha = [1.0, 1.5, 0.8, 0.5]  # Bias toward liquidity
    weights_raw = [trial.suggest_float(f'w_{i}', 0.01, 5.0) for i in range(4)]
    weights_sum = sum(weights_raw)
    w_wyckoff, w_liquidity, w_momentum, w_temporal = [w / weights_sum for w in weights_raw]

    cfg['fusion']['weights'] = {
        'wyckoff': round(w_wyckoff, 3),
        'liquidity': round(w_liquidity, 3),
        'momentum': round(w_momentum, 3),
        'smc': round(w_temporal, 3)
    }

    # ─── Global Gates (More selective for bear) ───
    final_fusion_floor = trial.suggest_float('final_fusion_floor', 0.38, 0.56, step=0.02)
    min_liquidity = trial.suggest_float('min_liquidity', 0.20, 0.32, step=0.02)

    cfg['fusion']['entry_threshold_confidence'] = final_fusion_floor
    cfg['archetypes']['thresholds']['min_liquidity'] = min_liquidity

    # ─── Archetype Thresholds (Higher for bear) ───
    B_fusion = trial.suggest_float('B_fusion', 0.32, 0.44, step=0.02)
    C_fusion = trial.suggest_float('C_fusion', 0.46, 0.60, step=0.02)
    H_fusion = trial.suggest_float('H_fusion', 0.52, 0.66, step=0.02)
    K_fusion = trial.suggest_float('K_fusion', 0.40, 0.52, step=0.02)
    L_fusion = trial.suggest_float('L_fusion', 0.32, 0.44, step=0.02)

    cfg['archetypes']['thresholds']['B']['fusion'] = B_fusion
    cfg['archetypes']['thresholds']['C']['fusion'] = C_fusion
    cfg['archetypes']['thresholds']['H']['fusion'] = H_fusion
    cfg['archetypes']['thresholds']['K']['fusion'] = K_fusion
    cfg['archetypes']['thresholds']['L']['fusion'] = L_fusion

    # ─── Exits (Wider stops for bear volatility) ───
    trail_atr = trial.suggest_float('trail_atr', 1.20, 2.00, step=0.10)
    max_bars = trial.suggest_int('max_bars', 54, 84, step=6)
    range_stop_factor = trial.suggest_float('range_stop_factor', 0.65, 0.85, step=0.05)
    trend_stop_factor = trial.suggest_float('trend_stop_factor', 1.10, 1.40, step=0.05)

    cfg['pnl_tracker']['exits']['trail_atr_mult'] = trail_atr
    cfg['pnl_tracker']['exits']['max_bars_in_trade'] = max_bars
    cfg['pnl_tracker']['exits']['range_stop_factor'] = range_stop_factor
    cfg['pnl_tracker']['exits']['trend_stop_factor'] = trend_stop_factor

    # ─── Dynamic Sizing (Conservative for bear) ───
    size_min = trial.suggest_float('size_min', 0.40, 0.70, step=0.05)
    size_max = trial.suggest_float('size_max', 0.70, 1.00, step=0.05)

    cfg['decision_gates']['size_min'] = size_min
    cfg['decision_gates']['size_max'] = size_max

    # ─── Run Folds ───
    fold_scores = []
    fold_metrics_log = []

    for fold_name, (start, end) in FOLDS.items():
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as tmp:
            json.dump(cfg, tmp, indent=2)
            tmp_path = tmp.name

        try:
            metrics = run_backtest(tmp_path, start, end, asset)
            if metrics:
                score = compute_fold_score(metrics)
                fold_scores.append(score)
                fold_metrics_log.append({
                    'fold': fold_name,
                    'trades': metrics['trades'],
                    'pnl': metrics['pnl'],
                    'pf': metrics['profit_factor'],
                    'dd': metrics['drawdown']
                })
        finally:
            Path(tmp_path).unlink(missing_ok=True)

    if not fold_scores or all(s < -9000 for s in fold_scores):
        print(f"Trial #{trial.number}: All folds failed")
        return -9999.0

    h_mean = harmonic_mean(fold_scores)

    # Log summary
    print(f"Trial #{trial.number}: H-Mean={h_mean:.1f} | Folds: {fold_metrics_log}")

    return h_mean


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Bear Frontier v10 Optimizer")
    parser.add_argument('--trials', type=int, default=240, help='Number of trials')
    parser.add_argument('--asset', default='BTC', help='Asset symbol')
    parser.add_argument('--output', required=True, help='Output directory')
    parser.add_argument('--base-config', default='configs/v10_bases/btc_bear_v10_base.json',
                        help='Base config template')
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*80}")
    print(f"Bear Frontier v10 - Pareto Mapping (2022-2023)")
    print(f"{'='*80}")
    print(f"Asset: {args.asset}")
    print(f"Trials: {args.trials}")
    print(f"Folds: {list(FOLDS.keys())}")
    print(f"Output: {output_dir}")
    print(f"Base: {args.base_config}")
    print(f"{'='*80}\n")

    # Load base config
    with open(args.base_config) as f:
        base_config = json.load(f)

    # Create Optuna study
    study = optuna.create_study(
        direction='maximize',
        study_name='bear_frontier_v10',
        sampler=optuna.samplers.TPESampler(seed=42, multivariate=True)
    )

    # Run optimization
    study.optimize(
        lambda trial: objective(trial, base_config, args.asset),
        n_trials=args.trials,
        show_progress_bar=True
    )

    # Save results
    df_trials = study.trials_dataframe()
    trials_path = output_dir / 'trials.csv'
    df_trials.to_csv(trials_path, index=False)

    best_trial = study.best_trial
    summary = {
        'study_name': study.study_name,
        'n_trials': len(study.trials),
        'best_trial': best_trial.number,
        'best_value': best_trial.value,
        'best_params': best_trial.params,
        'timestamp': datetime.utcnow().isoformat()
    }

    summary_path = output_dir / 'summary.json'
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\n{'='*80}")
    print(f"Bear Frontier Mapping Complete")
    print(f"{'='*80}")
    print(f"Best Trial: #{best_trial.number}")
    print(f"Best H-Mean Score: {best_trial.value:.2f}")
    print(f"Trials saved: {trials_path}")
    print(f"Summary saved: {summary_path}")
    print(f"{'='*80}\n")

    return 0


if __name__ == '__main__':
    exit(main())
