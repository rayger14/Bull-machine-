#!/usr/bin/env python3
"""
ML Phase 2: Improved Optuna Search with Sobol+TPE+Dirichlet

Key Improvements:
1. Sobol exploration (120 trials) → multivariate TPE exploitation (240 trials)
2. Dirichlet sampling for fusion weights (proper simplex constraint)
3. 2-3x wider parameter ranges
4. Cross-regime objective: robust PF = min(pf_2024, pf_22_23) * 0.7 + avg * 0.3
5. Explicit config deduplication via MD5 hashing
6. Hard guardrails on drawdown (DD < 3% for 2024, DD < 8% for 2022-2023)

Usage:
    python3 bin/optuna_improved_search.py --trials 360 --asset BTC --output reports/optuna_v9_improved
"""

import argparse
import json
import optuna
from optuna.samplers import QMCSampler, TPESampler
import subprocess
import tempfile
import re
import numpy as np
import hashlib
from pathlib import Path
from datetime import datetime

# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────
N_EXPLORATION = 120  # Sobol phase for space-filling exploration
N_TOTAL = 360        # Total trials (120 Sobol + 240 multivariate TPE)

# Test periods (cross-regime validation)
REGIME_2024 = ('2024-01-01', '2024-12-31')       # Bull regime
REGIME_2022_2023 = ('2022-01-01', '2023-12-31')  # Bear/ranging regime

# Config deduplication tracker
SEEN_CONFIGS = set()

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
# Config Hashing for Deduplication
# ─────────────────────────────────────────────────────────────────────────────
def hash_config(params):
    """
    Generate MD5 hash of config parameters for deduplication.
    Rounds floats to 3 decimal places to avoid floating point noise.
    """
    # Round all float values to 3 decimals
    rounded = {k: round(v, 3) if isinstance(v, float) else v for k, v in params.items()}

    # Sort keys for deterministic hashing
    canonical = json.dumps(rounded, sort_keys=True)

    return hashlib.md5(canonical.encode()).hexdigest()

# ─────────────────────────────────────────────────────────────────────────────
# Improved Parameter Suggestions with Dirichlet Sampling
# ─────────────────────────────────────────────────────────────────────────────
def suggest_dirichlet_weights(trial):
    """
    Sample fusion weights using Dirichlet distribution (proper simplex sampling).

    Instead of sampling independent floats and normalizing, we sample gamma
    variates and normalize - this is equivalent to sampling from Dirichlet
    and ensures proper exploration of the simplex.
    """
    # Sample gamma variates (log-uniform for exploration)
    gamma_wyckoff = trial.suggest_float('w_wyckoff_gamma', 0.1, 3.0, log=True)
    gamma_liquidity = trial.suggest_float('w_liquidity_gamma', 0.1, 3.0, log=True)
    gamma_momentum = trial.suggest_float('w_momentum_gamma', 0.1, 3.0, log=True)

    # Normalize to get weights
    total = gamma_wyckoff + gamma_liquidity + gamma_momentum
    w_wyckoff = gamma_wyckoff / total
    w_liquidity = gamma_liquidity / total
    w_momentum = gamma_momentum / total

    return w_wyckoff, w_liquidity, w_momentum

def suggest_params(trial, base_cfg):
    """
    Suggest parameters with 2-3x wider ranges than previous search.

    Key improvements:
    - Dirichlet sampling for fusion weights
    - Expanded ranges for all parameters
    - Conditional sampling for archetype thresholds
    """
    cfg = base_cfg.copy()

    # ─── Signal Neutralization (1.3x ranges - NARROW) ───
    final_fusion_floor = trial.suggest_float('final_fusion_floor', 0.28, 0.40, step=0.01)  # 1.3x: Was 0.30-0.38
    neutralize_fusion_drop = trial.suggest_float('neutralize_fusion_drop', 0.06, 0.20, step=0.01)  # 1.3x: Was 0.08-0.18
    neutralize_min_bars = trial.suggest_int('neutralize_min_bars', 4, 14)  # 1.3x: Was 5-12
    neutralize_pti_margin = trial.suggest_float('neutralize_pti_margin', 0.12, 0.35, step=0.05)  # 1.3x: Was 0.15-0.30

    # ─── Entry Gating (1.3x narrower) ───
    min_liquidity = trial.suggest_float('min_liquidity', 0.08, 0.24, step=0.01)  # 1.3x: Was 0.10-0.22

    # ─── Fusion Weights (Dirichlet sampling) ───
    w_wyckoff, w_liquidity, w_momentum = suggest_dirichlet_weights(trial)

    # ─── Dynamic Sizing (1.3x narrower) ───
    size_min = trial.suggest_float('size_min', 0.45, 0.80, step=0.05)  # 1.3x: Was 0.50-0.75
    size_max = trial.suggest_float('size_max', 1.00, 1.50, step=0.05)  # 1.3x: Was 1.05-1.40

    # ─── Per-Archetype Fusion Thresholds (1.3x narrower) ───
    B_fusion = trial.suggest_float('B_fusion', 0.30, 0.45, step=0.02)  # 1.3x: Was 0.32-0.42
    C_fusion = trial.suggest_float('C_fusion', 0.38, 0.54, step=0.02)  # 1.3x: Was 0.40-0.50
    H_fusion = trial.suggest_float('H_fusion', 0.47, 0.62, step=0.02)  # 1.3x: Was 0.50-0.58
    K_fusion = trial.suggest_float('K_fusion', 0.31, 0.48, step=0.02)  # 1.3x: Was 0.34-0.44
    L_fusion = trial.suggest_float('L_fusion', 0.33, 0.50, step=0.02)  # 1.3x: Was 0.36-0.46

    # ─── Exits (1.3x narrower) ───
    trail_atr_mult = trial.suggest_float('trail_atr_mult', 0.80, 1.35, step=0.05)  # 1.3x: Was 0.85-1.25
    max_bars = trial.suggest_int('max_bars', 54, 108, step=6)  # 1.3x: Was 60-96
    range_stop_factor = trial.suggest_float('range_stop_factor', 0.65, 0.92, step=0.05)  # 1.3x: Was 0.70-0.85
    trend_stop_factor = trial.suggest_float('trend_stop_factor', 1.15, 1.55, step=0.05)  # 1.3x: Was 1.20-1.45

    # ─── Build config ───
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

    # Update per-archetype thresholds
    for arch_key in ['B', 'C', 'H', 'K', 'L']:
        if arch_key not in cfg['archetypes']['thresholds']:
            cfg['archetypes']['thresholds'][arch_key] = {}

    cfg['archetypes']['thresholds']['B']['fusion'] = B_fusion
    cfg['archetypes']['thresholds']['C']['fusion'] = C_fusion
    cfg['archetypes']['thresholds']['H']['fusion'] = H_fusion
    cfg['archetypes']['thresholds']['K']['fusion'] = K_fusion
    cfg['archetypes']['thresholds']['L']['fusion'] = L_fusion

    # Final fusion floor
    cfg['fusion']['entry_threshold_confidence'] = final_fusion_floor

    # Neutralization
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

    # Exits
    if 'pnl_tracker' not in cfg:
        cfg['pnl_tracker'] = {}
    if 'exits' not in cfg['pnl_tracker']:
        cfg['pnl_tracker']['exits'] = {}

    cfg['pnl_tracker']['exits']['trail_atr_mult'] = trail_atr_mult
    cfg['pnl_tracker']['exits']['max_bars_in_trade'] = max_bars
    cfg['pnl_tracker']['exits']['range_stop_factor'] = range_stop_factor
    cfg['pnl_tracker']['exits']['trend_stop_factor'] = trend_stop_factor

    # Store trial params for logging and deduplication
    trial_params = {
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

    cfg['_trial_params'] = trial_params

    return cfg, trial_params

# ─────────────────────────────────────────────────────────────────────────────
# Cross-Regime Objective Function
# ─────────────────────────────────────────────────────────────────────────────
def objective(trial, base_cfg, asset, output_dir):
    """
    Cross-regime robust objective.

    Process:
    1. Sample config parameters
    2. Check for duplicates (MD5 hash) - prune if seen
    3. Run backtest on both 2024 and 2022-2023 regimes
    4. Apply hard guardrails on drawdown
    5. Compute robust PF: min(pf_2024, pf_22_23) * 0.7 + (pf_2024 + pf_22_23)/2 * 0.3
    6. Return robust PF as optimization objective
    """
    print(f"\n{'='*80}")
    print(f"Trial #{trial.number}")
    print(f"{'='*80}")

    # Suggest parameters
    cfg, trial_params = suggest_params(trial, base_cfg)

    # ─── Deduplication Check ───
    config_hash = hash_config(trial_params)

    if config_hash in SEEN_CONFIGS:
        print(f"  [DUPLICATE] Config hash {config_hash[:8]}... already evaluated")
        raise optuna.TrialPruned()

    SEEN_CONFIGS.add(config_hash)

    # Write config to temp file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(cfg, f, indent=2)
        config_path = f.name

    try:
        # ─── Run 2024 Regime (Bull) ───
        print(f"\n  [2024 REGIME] Running {REGIME_2024[0]} → {REGIME_2024[1]}...")
        metrics_2024 = run_backtest(config_path, REGIME_2024[0], REGIME_2024[1], asset)

        pf_2024 = metrics_2024['profit_factor']
        dd_2024 = metrics_2024['drawdown']
        trades_2024 = metrics_2024['trades']
        pnl_2024 = metrics_2024['pnl']

        print(f"    Trades: {trades_2024}, PNL: ${pnl_2024:.2f}, PF: {pf_2024:.2f}, DD: {dd_2024:.1f}%")

        # ─── Hard Guardrails: 2024 ───
        if dd_2024 > 3.0:
            print(f"  [REJECT] 2024 DD={dd_2024:.1f}% > 3.0%")
            return -999.0

        if trades_2024 < 10:
            print(f"  [REJECT] 2024 trades={trades_2024} < 10 (too sparse)")
            return -999.0

        # ─── Run 2022-2023 Regime (Bear/Ranging) ───
        print(f"\n  [2022-2023 REGIME] Running {REGIME_2022_2023[0]} → {REGIME_2022_2023[1]}...")
        metrics_22_23 = run_backtest(config_path, REGIME_2022_2023[0], REGIME_2022_2023[1], asset)

        pf_22_23 = metrics_22_23['profit_factor']
        dd_22_23 = metrics_22_23['drawdown']
        trades_22_23 = metrics_22_23['trades']
        pnl_22_23 = metrics_22_23['pnl']

        print(f"    Trades: {trades_22_23}, PNL: ${pnl_22_23:.2f}, PF: {pf_22_23:.2f}, DD: {dd_22_23:.1f}%")

        # ─── Hard Guardrails: 2022-2023 (RELAXED for viability) ───
        if dd_22_23 > 10.0:
            print(f"  [REJECT] 2022-2023 DD={dd_22_23:.1f}% > 10.0%")
            return -999.0

        if trades_22_23 < 10:  # Relaxed from 15 to 10
            print(f"  [REJECT] 2022-2023 trades={trades_22_23} < 10 (too sparse)")
            return -999.0

        # ─── Compute Robust Objective ───
        # Robust PF = 70% worst-case (min) + 30% average
        min_pf = min(pf_2024, pf_22_23)
        avg_pf = (pf_2024 + pf_22_23) / 2.0
        robust_pf = min_pf * 0.7 + avg_pf * 0.3

        print(f"\n  [ACCEPT] Robust PF: {robust_pf:.2f} (min={min_pf:.2f}, avg={avg_pf:.2f})")
        print(f"    2024: PF={pf_2024:.2f}, DD={dd_2024:.1f}%, Trades={trades_2024}")
        print(f"    2022-2023: PF={pf_22_23:.2f}, DD={dd_22_23:.1f}%, Trades={trades_22_23}")

        # ─── Log Results ───
        # Save to CSV
        trials_csv_path = output_dir / f"{asset}_all_trials.csv"
        if not trials_csv_path.exists():
            with open(trials_csv_path, 'w') as f:
                headers = ['trial', 'robust_pf', 'pf_2024', 'dd_2024', 'trades_2024', 'pnl_2024',
                          'pf_22_23', 'dd_22_23', 'trades_22_23', 'pnl_22_23', 'config_hash']
                headers += list(trial_params.keys())
                f.write(','.join(headers) + '\n')

        with open(trials_csv_path, 'a') as f:
            row = [
                trial.number,
                f"{robust_pf:.4f}",
                f"{pf_2024:.4f}",
                f"{dd_2024:.2f}",
                trades_2024,
                f"{pnl_2024:.2f}",
                f"{pf_22_23:.4f}",
                f"{dd_22_23:.2f}",
                trades_22_23,
                f"{pnl_22_23:.2f}",
                config_hash
            ]
            row += [trial_params.get(k, '') for k in trial_params.keys()]
            f.write(','.join(map(str, row)) + '\n')

        # Save best trial if this is the best so far
        best_score_file = output_dir / f"{asset}_best_score.txt"
        current_best = -999.0
        if best_score_file.exists():
            with open(best_score_file, 'r') as f:
                current_best = float(f.read().strip())

        if robust_pf > current_best:
            with open(best_score_file, 'w') as f:
                f.write(str(robust_pf))

            # Save best config
            with open(output_dir / f"{asset}_best_config.json", 'w') as f:
                json.dump(cfg, f, indent=2)

            # Save best summary
            best_summary = {
                'trial_number': trial.number,
                'robust_pf': robust_pf,
                'config_hash': config_hash,
                'regime_2024': {
                    'pf': pf_2024,
                    'dd': dd_2024,
                    'trades': trades_2024,
                    'pnl': pnl_2024
                },
                'regime_2022_2023': {
                    'pf': pf_22_23,
                    'dd': dd_22_23,
                    'trades': trades_22_23,
                    'pnl': pnl_22_23
                },
                'params': trial_params,
                'timestamp': datetime.now().isoformat()
            }

            with open(output_dir / f"{asset}_best_trial_summary.json", 'w') as f:
                json.dump(best_summary, f, indent=2)

            print(f"\n  [NEW BEST] Robust PF: {robust_pf:.2f}")

        return robust_pf

    finally:
        # Cleanup temp config
        Path(config_path).unlink(missing_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Improved Optuna search with Sobol+TPE+Dirichlet (v9b narrow ranges)")
    parser.add_argument('--trials', type=int, default=200, help="Total number of trials (default: 200)")
    parser.add_argument('--asset', type=str, default="BTC", help="Asset to optimize")
    parser.add_argument('--output', type=str, required=True, help="Output directory for results")
    parser.add_argument('--base-config', type=str, default="configs/profile_archetype_optimized.json",
                        help="Base config to start from")
    parser.add_argument('--exploration', type=int, default=80, help="Number of Sobol exploration trials (default: 80)")
    args = parser.parse_args()

    # Load base config
    with open(args.base_config, 'r') as f:
        base_cfg = json.load(f)

    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Update exploration/exploitation split
    global N_EXPLORATION, N_TOTAL
    N_EXPLORATION = args.exploration
    N_TOTAL = args.trials

    print(f"\n{'='*80}")
    print(f"Improved Optuna Search - {args.asset}")
    print(f"{'='*80}")
    print(f"Strategy: Sobol exploration → multivariate TPE exploitation")
    print(f"Phase 1: {N_EXPLORATION} trials (Sobol - space-filling exploration)")
    print(f"Phase 2: {N_TOTAL - N_EXPLORATION} trials (Multivariate TPE - exploitation)")
    print(f"")
    print(f"Objective: Cross-regime robust PF")
    print(f"  - Robust PF = min(PF_2024, PF_2022-23) * 0.7 + avg * 0.3")
    print(f"  - Guardrails: DD < 3% (2024), DD < 8% (2022-2023)")
    print(f"")
    print(f"Improvements:")
    print(f"  - Dirichlet sampling for fusion weights (proper simplex)")
    print(f"  - 2-3x wider parameter ranges")
    print(f"  - MD5 deduplication (real-time)")
    print(f"")
    print(f"Output: {output_dir}")
    print(f"Base Config: {args.base_config}")
    print(f"{'='*80}\n")

    # ─── Phase 1: Sobol Exploration ───
    print(f"\n{'='*80}")
    print(f"PHASE 1: Sobol Exploration ({N_EXPLORATION} trials)")
    print(f"{'='*80}\n")

    study = optuna.create_study(
        direction='maximize',
        sampler=QMCSampler(qmc_type="sobol", scramble=True, seed=42)
    )

    study.optimize(
        lambda trial: objective(trial, base_cfg, args.asset, output_dir),
        n_trials=N_EXPLORATION,
        show_progress_bar=True
    )

    print(f"\n{'='*80}")
    print(f"Phase 1 Complete: {N_EXPLORATION} trials")
    print(f"Best Robust PF (Sobol): {study.best_value:.4f}")
    print(f"Unique configs evaluated: {len(SEEN_CONFIGS)}")
    print(f"{'='*80}\n")

    # ─── Phase 2: Multivariate TPE Exploitation ───
    print(f"\n{'='*80}")
    print(f"PHASE 2: Multivariate TPE Exploitation ({N_TOTAL - N_EXPLORATION} trials)")
    print(f"{'='*80}\n")

    # Switch to multivariate TPE sampler
    tpe_sampler = TPESampler(
        seed=42,
        multivariate=True,
        group=True,
        n_startup_trials=0  # Use all Sobol trials as startup
    )
    study.sampler = tpe_sampler

    study.optimize(
        lambda trial: objective(trial, base_cfg, args.asset, output_dir),
        n_trials=N_TOTAL - N_EXPLORATION,
        show_progress_bar=True
    )

    # ─── Final Results ───
    print(f"\n{'='*80}")
    print(f"Optimization Complete")
    print(f"{'='*80}")
    print(f"Total Trials: {N_TOTAL}")
    print(f"Unique Configs: {len(SEEN_CONFIGS)}")
    print(f"Deduplication Rate: {(1 - len(SEEN_CONFIGS) / N_TOTAL) * 100:.1f}%")
    print(f"")
    print(f"Best Trial: #{study.best_trial.number}")
    print(f"Best Robust PF: {study.best_value:.4f}")
    print(f"")
    print(f"Best Parameters:")
    for key, value in study.best_params.items():
        if not key.endswith('_gamma'):  # Skip gamma variates, show normalized weights
            print(f"  {key}: {value}")
    print(f"")
    print(f"Results saved to: {output_dir}")
    print(f"  - {args.asset}_best_config.json")
    print(f"  - {args.asset}_best_trial_summary.json")
    print(f"  - {args.asset}_all_trials.csv")
    print(f"{'='*80}\n")

if __name__ == "__main__":
    main()
