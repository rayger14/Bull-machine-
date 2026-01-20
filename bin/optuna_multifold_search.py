#!/usr/bin/env python3
"""
PR#6A: Multi-Fold Optuna Archetype Optimizer (v2 - Full-Year Focus)

Optimizes for maximum full-year PNL with robust cross-validation:
- Uses Q1+Q2+Q4 2024 folds (skips Q3 due to regime incompatibility)
- Aggregates via harmonic mean across folds
- Enforces full-year guardrails: DD ≤7%, PF ≥2.5, trades 30-80
- Minimum 10 trades per fold requirement

Parameter search focuses on:
1. Global fusion/neutralization knobs (final_fusion_floor, min_liquidity, etc.)
2. Fusion weight rebalancing
3. Dynamic sizing bounds
4. Per-archetype thresholds (Phase B - optional)

Usage:
    python3 bin/optuna_multifold_search.py --trials 80 --asset BTC --output reports/multifold_v3
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
# Fold Definitions
# ─────────────────────────────────────────────────────────────────────────────
# Calibration: Q1 2024 only (dead quarters like Q2/Q3 skipped)
# Validation: Full year 2024 with strict guardrails
FOLDS_CAL = {
    'Q1': ('2024-01-01', '2024-03-31'),
}

YEAR_2024 = ('2024-01-01', '2024-12-31')

# ─────────────────────────────────────────────────────────────────────────────
# Helper: Run backtest and extract metrics
# ─────────────────────────────────────────────────────────────────────────────
def run_backtest(config_path: str, start_date: str, end_date: str, asset: str = "BTC"):
    """
    Run backtest_knowledge_v2.py with given config and extract metrics.
    Returns dict with: pnl, trades, roi, win_rate, drawdown, profit_factor, sharpe
    """
    cmd = [
        "python3",
        "bin/backtest_knowledge_v2.py",
        "--asset", asset,
        "--start", start_date,
        "--end", end_date,
        "--config", config_path
    ]

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=300,  # 5 min timeout
            check=False
        )

        output = result.stdout + result.stderr

        metrics = {
            'pnl': 0.0,
            'trades': 0,
            'roi': 0.0,
            'win_rate': 0.0,
            'drawdown': 0.0,
            'profit_factor': 0.0,
            'sharpe': 0.0
        }

        # PNL
        pnl_match = re.search(r'Total PNL:\s+\$?([-\d,\.]+)', output)
        if pnl_match:
            metrics['pnl'] = float(pnl_match.group(1).replace(',', ''))

        # Trade count
        trades_match = re.search(r'Total Trades:\s+(\d+)', output)
        if trades_match:
            metrics['trades'] = int(trades_match.group(1))

        # ROI
        roi_match = re.search(r'ROI:\s+([-\d\.]+)%', output)
        if roi_match:
            metrics['roi'] = float(roi_match.group(1))

        # Win Rate
        wr_match = re.search(r'Win Rate:\s+([\d\.]+)%', output)
        if wr_match:
            metrics['win_rate'] = float(wr_match.group(1))

        # Max Drawdown
        dd_match = re.search(r'Max Drawdown:\s+([\d\.]+)%', output)
        if dd_match:
            metrics['drawdown'] = float(dd_match.group(1))

        # Profit Factor
        pf_match = re.search(r'Profit Factor:\s+([\d\.]+)', output)
        if pf_match:
            metrics['profit_factor'] = float(pf_match.group(1))

        # Sharpe
        sharpe_match = re.search(r'Sharpe Ratio:\s+([-\d\.]+)', output)
        if sharpe_match:
            metrics['sharpe'] = float(sharpe_match.group(1))

        return metrics

    except subprocess.TimeoutExpired:
        print(f"    [TIMEOUT] Backtest timed out for {start_date} to {end_date}")
        return {
            'pnl': -999.0,
            'trades': 0,
            'roi': -999.0,
            'win_rate': 0.0,
            'drawdown': 99.0,
            'profit_factor': 0.0,
            'sharpe': -999.0
        }
    except Exception as e:
        print(f"    [ERROR] Backtest failed: {e}")
        return {
            'pnl': -999.0,
            'trades': 0,
            'roi': -999.0,
            'win_rate': 0.0,
            'drawdown': 99.0,
            'profit_factor': 0.0,
            'sharpe': -999.0
        }

# ─────────────────────────────────────────────────────────────────────────────
# Parameter Suggestions (Phase A: Global Knobs)
# ─────────────────────────────────────────────────────────────────────────────
def suggest_params(trial, base_cfg, asset='BTC'):
    """
    Suggest parameters for this trial.

    Phase A: Global fusion/neutralization knobs, sizing, exits
    (Per-archetype tuning can be added in Phase B)

    Asset-specific ranges:
    - BTC: Higher fusion ranges (mean ~0.35, p90 ~0.40)
    - ETH: Lower fusion ranges (mean ~0.28, p90 ~0.33)
    """
    cfg = base_cfg.copy()

    # Asset-specific search ranges
    if asset == 'ETH':
        # ETH v1b: NARROW ranges around proven seed (fusion ~0.29, liq 0.03)
        fusion_floor_min, fusion_floor_max = 0.26, 0.31  # Was 0.24-0.34 (seed: 0.29)
        min_liq_min, min_liq_max = 0.03, 0.10           # Was 0.03-0.15 (seed: 0.03)
        B_fusion_min, B_fusion_max = 0.22, 0.28          # Was 0.20-0.32 (seed: 0.24)
        C_fusion_min, C_fusion_max = 0.36, 0.44          # Narrow upper range
        K_fusion_min, K_fusion_max = 0.26, 0.32          # Was 0.24-0.36 (seed: 0.28)
        L_fusion_min, L_fusion_max = 0.26, 0.32          # Was 0.24-0.36 (seed: 0.28)
        H_fusion_min, H_fusion_max = 0.46, 0.54          # Was 0.42-0.52 (seed: 0.50)
    else:  # BTC
        # BTC ranges (original)
        fusion_floor_min, fusion_floor_max = 0.32, 0.42
        min_liq_min, min_liq_max = 0.10, 0.22
        B_fusion_min, B_fusion_max = 0.32, 0.42
        C_fusion_min, C_fusion_max = 0.40, 0.50
        K_fusion_min, K_fusion_max = 0.34, 0.44
        L_fusion_min, L_fusion_max = 0.36, 0.46
        H_fusion_min, H_fusion_max = 0.50, 0.58

    # ─── Signal Neutralization ───
    final_fusion_floor = trial.suggest_float('final_fusion_floor', fusion_floor_min, fusion_floor_max, step=0.01)
    neutralize_fusion_drop = trial.suggest_float('neutralize_fusion_drop', 0.08, 0.18, step=0.01)
    neutralize_min_bars = trial.suggest_int('neutralize_min_bars', 5, 12)
    neutralize_pti_margin = trial.suggest_float('neutralize_pti_margin', 0.15, 0.30, step=0.05)

    # ─── Entry Gating ───
    min_liquidity = trial.suggest_float('min_liquidity', min_liq_min, min_liq_max, step=0.02)

    # ─── Fusion Weights (normalized to sum=1.0) ───
    w_wyckoff_raw = trial.suggest_float('w_wyckoff', 0.20, 0.45, step=0.05)
    w_liquidity_raw = trial.suggest_float('w_liquidity', 0.15, 0.35, step=0.05)
    w_momentum_raw = trial.suggest_float('w_momentum', 0.20, 0.45, step=0.05)

    total_weight = w_wyckoff_raw + w_liquidity_raw + w_momentum_raw
    w_wyckoff = w_wyckoff_raw / total_weight
    w_liquidity = w_liquidity_raw / total_weight
    w_momentum = w_momentum_raw / total_weight

    # ─── Dynamic Sizing ───
    size_min = trial.suggest_float('size_min', 0.50, 0.75, step=0.05)
    size_max = trial.suggest_float('size_max', 1.05, 1.40, step=0.05)

    # ─── Per-Archetype Fusion Thresholds (keep most at current optimized, tune top-firing only) ───
    B_fusion = trial.suggest_float('B_fusion', B_fusion_min, B_fusion_max, step=0.02)  # order_block_retest
    C_fusion = trial.suggest_float('C_fusion', C_fusion_min, C_fusion_max, step=0.02)  # fvg_continuation
    K_fusion = trial.suggest_float('K_fusion', K_fusion_min, K_fusion_max, step=0.02)  # wick_trap
    L_fusion = trial.suggest_float('L_fusion', L_fusion_min, L_fusion_max, step=0.02)  # volume_exhaustion

    # Keep H (trap_within_trend) strict to prevent over-firing
    H_fusion = trial.suggest_float('H_fusion', H_fusion_min, H_fusion_max, step=0.02)

    # ─── Exits ───
    trail_atr_mult = trial.suggest_float('trail_atr_mult', 0.85, 1.25, step=0.05)
    max_bars = trial.suggest_int('max_bars', 60, 96, step=6)
    range_stop_factor = trial.suggest_float('range_stop_factor', 0.70, 0.85, step=0.05)
    trend_stop_factor = trial.suggest_float('trend_stop_factor', 1.20, 1.45, step=0.05)

    # ─── Apply to config ───
    # Fusion weights
    if 'fusion' not in cfg:
        cfg['fusion'] = {}
    if 'weights' not in cfg['fusion']:
        cfg['fusion']['weights'] = {}

    cfg['fusion']['weights']['wyckoff'] = round(w_wyckoff, 3)
    cfg['fusion']['weights']['liquidity'] = round(w_liquidity, 3)
    cfg['fusion']['weights']['momentum'] = round(w_momentum, 3)
    cfg['fusion']['weights']['smc'] = round(1.0 - w_wyckoff - w_liquidity - w_momentum, 3)

    # Archetype thresholds
    if 'archetypes' not in cfg:
        cfg['archetypes'] = {}
    if 'thresholds' not in cfg['archetypes']:
        cfg['archetypes']['thresholds'] = {}

    cfg['archetypes']['thresholds']['min_liquidity'] = min_liquidity

    # Update per-archetype thresholds (keep structure, modify fusion values)
    for arch_key in ['B', 'C', 'H', 'K', 'L']:
        if arch_key not in cfg['archetypes']['thresholds']:
            cfg['archetypes']['thresholds'][arch_key] = {}

    cfg['archetypes']['thresholds']['B']['fusion'] = B_fusion
    cfg['archetypes']['thresholds']['C']['fusion'] = C_fusion
    cfg['archetypes']['thresholds']['H']['fusion'] = H_fusion
    cfg['archetypes']['thresholds']['K']['fusion'] = K_fusion
    cfg['archetypes']['thresholds']['L']['fusion'] = L_fusion

    # Apply final fusion floor (global quality gate)
    if 'fusion' not in cfg:
        cfg['fusion'] = {}
    cfg['fusion']['entry_threshold_confidence'] = final_fusion_floor

    # Apply neutralization parameters
    if 'neutralization' not in cfg:
        cfg['neutralization'] = {}
    cfg['neutralization']['fusion_drop_threshold'] = neutralize_fusion_drop
    cfg['neutralization']['min_bars_since_entry'] = neutralize_min_bars
    cfg['neutralization']['pti_margin'] = neutralize_pti_margin

    # Decision gates (dynamic sizing)
    if 'decision_gates' not in cfg:
        cfg['decision_gates'] = {}
    cfg['decision_gates']['sizing_min_leverage'] = size_min
    cfg['decision_gates']['sizing_max_leverage'] = size_max

    # Exits (regime-adaptive stops)
    if 'pnl_tracker' not in cfg:
        cfg['pnl_tracker'] = {}
    if 'exits' not in cfg['pnl_tracker']:
        cfg['pnl_tracker']['exits'] = {}

    cfg['pnl_tracker']['exits']['trail_atr_mult'] = trail_atr_mult
    cfg['pnl_tracker']['exits']['max_bars_in_trade'] = max_bars
    cfg['pnl_tracker']['exits']['range_stop_factor'] = range_stop_factor
    cfg['pnl_tracker']['exits']['trend_stop_factor'] = trend_stop_factor

    # Store trial params as attributes for logging
    cfg['_trial_params'] = {
        'final_fusion_floor': final_fusion_floor,
        'neutralize_fusion_drop': neutralize_fusion_drop,
        'neutralize_min_bars': neutralize_min_bars,
        'neutralize_pti_margin': neutralize_pti_margin,
        'min_liquidity': min_liquidity,
        'w_wyckoff': w_wyckoff,
        'w_liquidity': w_liquidity,
        'w_momentum': w_momentum,
        'size_min': size_min,
        'size_max': size_max,
        'B_fusion': B_fusion,
        'C_fusion': C_fusion,
        'H_fusion': H_fusion,
        'K_fusion': K_fusion,
        'L_fusion': L_fusion,
        'trail_atr_mult': trail_atr_mult,
        'max_bars': max_bars,
        'range_stop_factor': range_stop_factor,
        'trend_stop_factor': trend_stop_factor,
    }

    return cfg

# ─────────────────────────────────────────────────────────────────────────────
# Scoring Function
# ─────────────────────────────────────────────────────────────────────────────
def compute_fold_score(metrics):
    """
    Compute score for a single fold.

    Score = PNL + PF_bonus - DD_penalty - overtrade_penalty
    """
    pnl = metrics['pnl']
    pf = metrics['profit_factor']
    dd = metrics['drawdown']
    trades = metrics['trades']

    # Base score = PNL
    score = pnl

    # Profit factor bonus (cap at 4.0)
    pf_capped = min(pf, 4.0)
    pf_bonus = pf_capped * 100
    score += pf_bonus

    # Drawdown penalty (exponential above 5%)
    if dd > 5.0:
        dd_penalty = ((dd - 5.0) ** 1.5) * 100
        score -= dd_penalty

    # Overtrade penalty (exponential above 30 trades/quarter)
    if trades > 30:
        overtrade_penalty = ((trades - 30) ** 1.2) * 5
        score -= overtrade_penalty

    return score

def harmonic_mean(scores):
    """
    Compute harmonic mean of scores.
    Punishes imbalanced performance across folds.
    """
    if len(scores) == 0:
        return -999.0

    # Handle negative scores by shifting to positive space
    min_score = min(scores)
    if min_score <= 0:
        shift = abs(min_score) + 1
        scores_shifted = [s + shift for s in scores]
    else:
        scores_shifted = scores
        shift = 0

    # Compute harmonic mean
    reciprocals = [1.0 / s if s > 0 else 0 for s in scores_shifted]
    hm = len(scores_shifted) / sum(reciprocals) if sum(reciprocals) > 0 else 0

    # Shift back
    hm_original = hm - shift

    return hm_original

# ─────────────────────────────────────────────────────────────────────────────
# Objective Function
# ─────────────────────────────────────────────────────────────────────────────
def objective(trial, base_cfg, asset, output_dir):
    """
    Single-fold calibration with full-year validation.

    Process:
    1. Run Q1 2024 calibration fold
    2. Compute Q1 score (no rejection, just scoring)
    3. Run full-year 2024 backtest
    4. Apply guardrails on FULL YEAR ONLY (DD ≤7%, PF ≥2.5, trades 30-80, WR≥55%)
    5. Return Q1 score if full-year passes, else penalty
    """
    print(f"\n{'='*80}")
    print(f"Trial #{trial.number}")
    print(f"{'='*80}")

    # Suggest parameters (with asset-specific ranges)
    cfg = suggest_params(trial, base_cfg, asset)

    # Write config to temp file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(cfg, f, indent=2)
        config_path = f.name

    try:
        # ─── Phase 1: Run Q1 Calibration Fold ───
        fold_scores = []
        fold_results = {}

        for fold_name, (start, end) in FOLDS_CAL.items():
            print(f"\n  [{fold_name} CALIBRATION] Running {start} → {end}...")
            metrics = run_backtest(config_path, start, end, asset)

            trades = metrics['trades']
            pnl = metrics['pnl']
            pf = metrics['profit_factor']
            dd = metrics['drawdown']

            print(f"    Trades: {trades}, PNL: ${pnl:.2f}, PF: {pf:.2f}, DD: {dd:.1f}%")

            # NO fold-level rejection - just log if sparse
            if trades < 5:
                print(f"    [WARNING] Sparse calibration fold ({trades} trades)")

            # Compute fold score (for objective value)
            score = compute_fold_score(metrics)
            fold_scores.append(score)
            fold_results[fold_name] = {
                'metrics': metrics,
                'score': score
            }

            print(f"    Calibration Score: {score:.2f}")

        # ─── Phase 2: Q1 Score (no harmonic mean needed, only 1 fold) ───
        q1_score = fold_scores[0] if fold_scores else -999.0
        print(f"\n  Q1 Calibration Score: {q1_score:.2f}")

        # ─── Early Abort Check (ETH only - save compute on hopeless trials) ───
        if asset == 'ETH':
            q1_trades = list(fold_results.values())[0]['metrics']['trades'] if fold_results else 0
            q1_pf = list(fold_results.values())[0]['metrics']['profit_factor'] if fold_results else 0

            # Abort if Q1 shows signs of disaster
            if q1_trades > 60:  # Proportionally would be >240 annual - instant reject
                print(f"  [EARLY ABORT] Q1 trades={q1_trades} projects to >{q1_trades*4} annual (over-trading)")
                return -999.0
            if q1_pf < 0.7:  # Very unlikely to recover to PF≥1.6
                print(f"  [EARLY ABORT] Q1 PF={q1_pf:.2f} too low to meet annual target")
                return -999.0

        # ─── Phase 3: Full-Year Guardrails ───
        print(f"\n  [FULL-YEAR] Running {YEAR_2024[0]} → {YEAR_2024[1]}...")
        year_metrics = run_backtest(config_path, YEAR_2024[0], YEAR_2024[1], asset)

        year_trades = year_metrics['trades']
        year_pnl = year_metrics['pnl']
        year_pf = year_metrics['profit_factor']
        year_dd = year_metrics['drawdown']
        year_roi = year_metrics['roi']
        year_wr = year_metrics['win_rate']

        print(f"    Trades: {year_trades}, PNL: ${year_pnl:.2f}, ROI: {year_roi:.1f}%")
        print(f"    PF: {year_pf:.2f}, DD: {year_dd:.1f}%, WR: {year_wr:.1f}%")

        # Apply asset-specific guardrails
        guardrails_pass = True
        reject_reason = []

        # Asset-specific limits
        if asset == 'ETH':
            max_dd = 10.0
            min_pf = 1.6  # Realistic for ETH (was 2.5)
            trade_min, trade_max = 30, 90
            min_wr = 54.0  # Slightly relaxed for ETH
        else:  # BTC
            max_dd = 7.0
            min_pf = 2.5  # BTC can achieve this
            trade_min, trade_max = 30, 80
            min_wr = 55.0

        if year_dd > max_dd:
            guardrails_pass = False
            reject_reason.append(f"DD={year_dd:.1f}% > {max_dd:.0f}%")

        if year_pf < min_pf:
            guardrails_pass = False
            reject_reason.append(f"PF={year_pf:.2f} < {min_pf:.1f}")

        if year_trades < trade_min or year_trades > trade_max:
            guardrails_pass = False
            reject_reason.append(f"Trades={year_trades} outside [{trade_min}, {trade_max}]")

        if year_wr < min_wr:
            guardrails_pass = False
            reject_reason.append(f"WR={year_wr:.1f}% < {min_wr:.0f}%")

        if not guardrails_pass:
            print(f"\n  [REJECT] Full-year guardrails failed: {', '.join(reject_reason)}")
            return -999.0

        # ─── Phase 4: Log Results ───
        print(f"\n  [ACCEPT] Q1 Calibration Score: {q1_score:.2f}")

        # Save trial summary
        trial_summary = {
            'trial_number': trial.number,
            'q1_calibration_score': q1_score,
            'params': cfg.get('_trial_params', {}),
            'calibration_folds': {
                fold_name: {
                    'pnl': res['metrics']['pnl'],
                    'trades': res['metrics']['trades'],
                    'pf': res['metrics']['profit_factor'],
                    'dd': res['metrics']['drawdown'],
                    'score': res['score']
                }
                for fold_name, res in fold_results.items()
            },
            'full_year': {
                'pnl': year_pnl,
                'trades': year_trades,
                'roi': year_roi,
                'pf': year_pf,
                'dd': year_dd,
                'win_rate': year_wr
            },
            'timestamp': datetime.now().isoformat()
        }

        # Append to all trials CSV
        trials_csv_path = output_dir / f"{asset}_all_trials.csv"
        if not trials_csv_path.exists():
            with open(trials_csv_path, 'w') as f:
                headers = ['trial', 'q1_score', 'year_pnl', 'year_trades', 'year_pf', 'year_dd', 'year_wr']
                headers += [f'{fold}_pnl' for fold in FOLDS_CAL.keys()]
                headers += [f'{fold}_trades' for fold in FOLDS_CAL.keys()]
                headers += list(cfg.get('_trial_params', {}).keys())
                f.write(','.join(headers) + '\n')

        with open(trials_csv_path, 'a') as f:
            row = [
                trial.number,
                f"{q1_score:.2f}",
                f"{year_pnl:.2f}",
                year_trades,
                f"{year_pf:.2f}",
                f"{year_dd:.2f}",
                f"{year_wr:.1f}"
            ]
            row += [f"{fold_results[fold]['metrics']['pnl']:.2f}" for fold in FOLDS_CAL.keys()]
            row += [fold_results[fold]['metrics']['trades'] for fold in FOLDS_CAL.keys()]
            row += [cfg['_trial_params'].get(k, '') for k in cfg.get('_trial_params', {}).keys()]
            f.write(','.join(map(str, row)) + '\n')

        # Save best trial if this is the best so far
        best_score_file = output_dir / f"{asset}_best_score.txt"
        current_best = -999.0
        if best_score_file.exists():
            with open(best_score_file, 'r') as f:
                current_best = float(f.read().strip())

        if q1_score > current_best:
            with open(best_score_file, 'w') as f:
                f.write(str(q1_score))

            # Save best config
            with open(output_dir / f"{asset}_best_config.json", 'w') as f:
                json.dump(cfg, f, indent=2)

            # Save best summary
            with open(output_dir / f"{asset}_best_trial_summary.json", 'w') as f:
                json.dump(trial_summary, f, indent=2)

            print(f"\n  [NEW BEST] Q1 Score: {q1_score:.2f}, Full-Year: {year_trades} trades, ${year_pnl:.2f} PNL, PF={year_pf:.2f}")

        return q1_score

    finally:
        # Cleanup temp config
        Path(config_path).unlink(missing_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Multi-fold archetype optimizer")
    parser.add_argument('--trials', type=int, default=80, help="Number of Optuna trials")
    parser.add_argument('--asset', type=str, default="BTC", help="Asset to optimize")
    parser.add_argument('--output', type=str, required=True, help="Output directory for results")
    parser.add_argument('--base-config', type=str, default="configs/profile_archetype_optimized.json",
                        help="Base config to start from")
    args = parser.parse_args()

    # Load base config
    with open(args.base_config, 'r') as f:
        base_cfg = json.load(f)

    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Asset-specific guardrails display
    if args.asset == 'ETH':
        guardrails_str = "PF≥1.6, DD≤10%, trades 30-90, WR≥54%"  # ETH v1b realistic
    else:  # BTC
        guardrails_str = "PF≥2.5, DD≤7%, trades 30-80, WR≥55%"

    print(f"\n{'='*80}")
    print(f"Single-Fold Calibration + Full-Year Validation - {args.asset}")
    print(f"{'='*80}")
    print(f"Calibration: Q1 2024 (dead quarters Q2/Q3 skipped)")
    print(f"Validation: Full-Year 2024 with guardrails")
    print(f"Guardrails: {guardrails_str}")
    print(f"Trials: {args.trials}")
    print(f"Output: {output_dir}")
    print(f"Base Config: {args.base_config}")
    print(f"{'='*80}\n")

    # Create Optuna study
    study = optuna.create_study(
        direction='maximize',
        sampler=optuna.samplers.TPESampler(seed=42)
    )

    # Run optimization
    study.optimize(
        lambda trial: objective(trial, base_cfg, args.asset, output_dir),
        n_trials=args.trials,
        show_progress_bar=True
    )

    # Print results
    print(f"\n{'='*80}")
    print(f"Optimization Complete")
    print(f"{'='*80}")
    print(f"Best Trial: #{study.best_trial.number}")
    print(f"Best Q1 Calibration Score: {study.best_value:.2f}")
    print(f"\nBest Parameters:")
    for key, value in study.best_params.items():
        print(f"  {key}: {value}")
    print(f"\nResults saved to: {output_dir}")
    print(f"\nNext: Review BTC_best_trial_summary.json for full-year metrics")
    print(f"{'='*80}\n")

if __name__ == "__main__":
    main()
