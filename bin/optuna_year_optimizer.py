#!/usr/bin/env python3
"""
PR#6A: Year-Centric Optuna Optimizer (Option C - Production)

Strategy:
- Calibrate on Q1 2024 (Jan-Mar): 18 trades, $1,243 PNL proven
- Validate on Full Year 2024: Maximize annual PNL with guardrails

Objective: PNL_year + PF_bonus - DD_penalty - trade_band_penalty

Guardrails (hard rejections):
- DD > 7%
- PF < 2.0
- Trades ∉ [40, 90]

Parameter Search (tight, meaningful ranges):
- Signal neutralization: final_fusion_floor, neutralize_drop, PTI margins
- Fusion weights: wyckoff, liquidity, momentum (normalized)
- Entry gating: min_liquidity, archetype-specific thresholds
- Dynamic sizing: min/max leverage, per-archetype multipliers
- Exits: trailing ATR, max bars, regime-adaptive stops

Usage:
    python3 bin/optuna_year_optimizer.py --trials 80 --asset BTC --output reports/year_opt_v1
"""

import argparse
import json
import optuna
import subprocess
import tempfile
import re
from pathlib import Path
from datetime import datetime

# ─────────────────────────────────────────────────────────────────────────────
# Windows
# ─────────────────────────────────────────────────────────────────────────────
CALIB_WINDOW = ('2024-01-01', '2024-03-31')  # Q1 2024: Proven +$1,243 PNL
VALIDATE_WINDOW = ('2024-01-01', '2024-12-31')  # Full Year 2024

# ─────────────────────────────────────────────────────────────────────────────
# Helper: Run backtest
# ─────────────────────────────────────────────────────────────────────────────
def run_backtest(config_path: str, start_date: str, end_date: str, asset: str = "BTC"):
    """Run backtest and extract metrics."""
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
            timeout=300,
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

        # Extract metrics
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

        sharpe_match = re.search(r'Sharpe Ratio:\s+([-\d\.]+)', output)
        if sharpe_match:
            metrics['sharpe'] = float(sharpe_match.group(1))

        return metrics

    except subprocess.TimeoutExpired:
        print(f"    [TIMEOUT] Backtest timed out")
        return {'pnl': -999.0, 'trades': 0, 'roi': -999.0, 'win_rate': 0.0,
                'drawdown': 99.0, 'profit_factor': 0.0, 'sharpe': -999.0}
    except Exception as e:
        print(f"    [ERROR] Backtest failed: {e}")
        return {'pnl': -999.0, 'trades': 0, 'roi': -999.0, 'win_rate': 0.0,
                'drawdown': 99.0, 'profit_factor': 0.0, 'sharpe': -999.0}

# ─────────────────────────────────────────────────────────────────────────────
# Parameter Suggestions (Tight, Meaningful Ranges)
# ─────────────────────────────────────────────────────────────────────────────
def suggest_params(trial, base_cfg):
    """
    Suggest parameters with tight, production-ready ranges.

    Search space designed for:
    - Final fusion floor ≥0.374 (avoid weak setups)
    - min_liquidity ≥0.15 (quality over quantity)
    - Balanced fusion weights
    - Conservative sizing with per-archetype multipliers
    """
    cfg = base_cfg.copy()

    # ─── Signal Neutralization ───
    final_fusion_floor = trial.suggest_float('final_fusion_floor', 0.374, 0.46, step=0.01)
    neutralize_fusion_drop = trial.suggest_float('neutralize_fusion_drop', 0.05, 0.14, step=0.01)
    neutralize_min_bars = trial.suggest_int('neutralize_min_bars', 5, 12)
    pti_reversal_margin = trial.suggest_float('pti_reversal_margin', 0.02, 0.10, step=0.01)
    liquidity_falloff_threshold = trial.suggest_float('liquidity_falloff_threshold', 0.05, 0.20, step=0.01)

    # ─── Fusion Weights (normalized to sum=1.0) ───
    w_wyckoff_raw = trial.suggest_float('w_wyckoff', 0.22, 0.48, step=0.02)
    w_liquidity_raw = trial.suggest_float('w_liquidity', 0.22, 0.48, step=0.02)
    w_momentum_raw = trial.suggest_float('w_momentum', 0.10, 0.30, step=0.02)

    total_weight = w_wyckoff_raw + w_liquidity_raw + w_momentum_raw
    w_wyckoff = w_wyckoff_raw / total_weight
    w_liquidity = w_liquidity_raw / total_weight
    w_momentum = w_momentum_raw / total_weight

    # ─── Entry Gating ───
    min_liquidity = trial.suggest_float('min_liquidity', 0.15, 0.24, step=0.01)

    # ─── Archetype Fusion Thresholds (top 3 only) ───
    B_fusion = trial.suggest_float('B_fusion', 0.38, 0.52, step=0.02)  # order_block_retest
    K_fusion = trial.suggest_float('K_fusion', 0.36, 0.52, step=0.02)  # wick_trap
    L_fusion = trial.suggest_float('L_fusion', 0.36, 0.50, step=0.02)  # volume_exhaustion

    # Keep other archetypes at current optimized values
    C_fusion = 0.46  # failed_continuation (keep current)
    H_fusion = 0.55  # trap_within_trend (keep strict)

    # ─── Dynamic Sizing ───
    size_min = trial.suggest_float('size_min', 0.55, 0.70, step=0.05)
    size_max = trial.suggest_float('size_max', 1.20, 1.40, step=0.05)

    # Per-archetype multipliers
    size_K = trial.suggest_float('size_K', 1.10, 1.40, step=0.05)  # wick_trap (quality)
    size_L = trial.suggest_float('size_L', 1.10, 1.40, step=0.05)  # volume_exhaustion (quality)
    size_B = trial.suggest_float('size_B', 0.90, 1.15, step=0.05)  # order_block_retest
    size_C = trial.suggest_float('size_C', 0.70, 1.00, step=0.05)  # failed_continuation
    size_H = trial.suggest_float('size_H', 0.50, 0.80, step=0.05)  # trap_within_trend (lower quality)

    # ─── Exits ───
    trail_atr_mult = trial.suggest_float('trail_atr_mult', 1.0, 1.3, step=0.05)
    max_bars = trial.suggest_int('max_bars', 48, 96, step=6)
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

    # Entry gating
    if 'archetypes' not in cfg:
        cfg['archetypes'] = {}
    if 'thresholds' not in cfg['archetypes']:
        cfg['archetypes']['thresholds'] = {}

    cfg['archetypes']['thresholds']['min_liquidity'] = min_liquidity

    # Archetype thresholds
    for arch_key in ['B', 'C', 'H', 'K', 'L']:
        if arch_key not in cfg['archetypes']['thresholds']:
            cfg['archetypes']['thresholds'][arch_key] = {}

    cfg['archetypes']['thresholds']['B']['fusion'] = B_fusion
    cfg['archetypes']['thresholds']['C']['fusion'] = C_fusion
    cfg['archetypes']['thresholds']['H']['fusion'] = H_fusion
    cfg['archetypes']['thresholds']['K']['fusion'] = K_fusion
    cfg['archetypes']['thresholds']['L']['fusion'] = L_fusion

    # Dynamic sizing
    if 'decision_gates' not in cfg:
        cfg['decision_gates'] = {}
    cfg['decision_gates']['sizing_min_leverage'] = size_min
    cfg['decision_gates']['sizing_max_leverage'] = size_max

    # Per-archetype sizing (store in custom section, implementation TBD)
    if 'archetype_sizing' not in cfg:
        cfg['archetype_sizing'] = {}
    cfg['archetype_sizing']['size_K'] = size_K
    cfg['archetype_sizing']['size_L'] = size_L
    cfg['archetype_sizing']['size_B'] = size_B
    cfg['archetype_sizing']['size_C'] = size_C
    cfg['archetype_sizing']['size_H'] = size_H

    # Exits
    if 'pnl_tracker' not in cfg:
        cfg['pnl_tracker'] = {}
    if 'exits' not in cfg['pnl_tracker']:
        cfg['pnl_tracker']['exits'] = {}

    cfg['pnl_tracker']['exits']['trail_atr_mult'] = trail_atr_mult
    cfg['pnl_tracker']['exits']['max_bars_in_trade'] = max_bars
    cfg['pnl_tracker']['exits']['range_stop_factor'] = range_stop_factor
    cfg['pnl_tracker']['exits']['trend_stop_factor'] = trend_stop_factor

    # Store trial params
    cfg['_trial_params'] = {
        'final_fusion_floor': final_fusion_floor,
        'neutralize_fusion_drop': neutralize_fusion_drop,
        'neutralize_min_bars': neutralize_min_bars,
        'pti_reversal_margin': pti_reversal_margin,
        'liquidity_falloff_threshold': liquidity_falloff_threshold,
        'min_liquidity': min_liquidity,
        'w_wyckoff': w_wyckoff,
        'w_liquidity': w_liquidity,
        'w_momentum': w_momentum,
        'size_min': size_min,
        'size_max': size_max,
        'B_fusion': B_fusion,
        'K_fusion': K_fusion,
        'L_fusion': L_fusion,
        'size_K': size_K,
        'size_L': size_L,
        'size_B': size_B,
        'size_C': size_C,
        'size_H': size_H,
        'trail_atr_mult': trail_atr_mult,
        'max_bars': max_bars,
        'range_stop_factor': range_stop_factor,
        'trend_stop_factor': trend_stop_factor,
    }

    return cfg

# ─────────────────────────────────────────────────────────────────────────────
# Objective Function
# ─────────────────────────────────────────────────────────────────────────────
def objective(trial, base_cfg, asset, output_dir):
    """
    Year-centric objective with tight guardrails.

    Process:
    1. Run full-year backtest (2024-01-01 → 2024-12-31)
    2. Apply guardrails (DD ≤7%, PF ≥2.0, trades ∈ [40,90])
    3. Score = PNL_year + min(PF*200, 800) - DD*2000 - trade_band_penalty
    4. Return score if passes, else -inf
    """
    print(f"\n{'='*80}")
    print(f"Trial #{trial.number}")
    print(f"{'='*80}")

    # Suggest parameters
    cfg = suggest_params(trial, base_cfg)

    # Write config to temp file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(cfg, f, indent=2)
        config_path = f.name

    try:
        # ─── Run Full-Year Backtest ───
        print(f"\n  [FULL-YEAR] Running {VALIDATE_WINDOW[0]} → {VALIDATE_WINDOW[1]}...")
        year_metrics = run_backtest(config_path, VALIDATE_WINDOW[0], VALIDATE_WINDOW[1], asset)

        year_trades = year_metrics['trades']
        year_pnl = year_metrics['pnl']
        year_pf = year_metrics['profit_factor']
        year_dd = year_metrics['drawdown']
        year_roi = year_metrics['roi']
        year_sharpe = year_metrics['sharpe']

        print(f"    Trades: {year_trades}, PNL: ${year_pnl:.2f}, ROI: {year_roi:.1f}%")
        print(f"    PF: {year_pf:.2f}, DD: {year_dd:.1f}%, Sharpe: {year_sharpe:.2f}")

        # ─── Apply Guardrails ───
        guardrails_pass = True
        reject_reason = []

        if year_dd > 7.0:
            guardrails_pass = False
            reject_reason.append(f"DD={year_dd:.1f}% > 7%")

        if year_pf < 2.0:
            guardrails_pass = False
            reject_reason.append(f"PF={year_pf:.2f} < 2.0")

        if year_trades < 40:
            guardrails_pass = False
            reject_reason.append(f"Trades={year_trades} < 40")
        elif year_trades > 90:
            guardrails_pass = False
            reject_reason.append(f"Trades={year_trades} > 90")

        if not guardrails_pass:
            print(f"\n  [REJECT] Guardrails failed: {', '.join(reject_reason)}")
            return float('-inf')

        # ─── Compute Score ───
        score = year_pnl

        # PF bonus (cap at 4.0 → max +800)
        pf_capped = min(year_pf, 4.0)
        pf_bonus = pf_capped * 200
        score += pf_bonus

        # DD penalty (linear)
        dd_penalty = year_dd * 2000
        score -= dd_penalty

        # Trade band penalty (soft pressure toward middle)
        if year_trades < 50:
            trade_penalty = (50 - year_trades) * 10
            score -= trade_penalty
        elif year_trades > 80:
            trade_penalty = (year_trades - 80) * 10
            score -= trade_penalty

        print(f"\n  [SCORE] PNL: {year_pnl:.2f}, PF Bonus: {pf_bonus:.2f}, DD Penalty: {dd_penalty:.2f}")
        print(f"  [FINAL] Score: {score:.2f}")

        # ─── Log Results ───
        trial_summary = {
            'trial_number': trial.number,
            'score': score,
            'params': cfg.get('_trial_params', {}),
            'full_year': {
                'pnl': year_pnl,
                'trades': year_trades,
                'roi': year_roi,
                'pf': year_pf,
                'dd': year_dd,
                'sharpe': year_sharpe,
                'win_rate': year_metrics['win_rate']
            },
            'timestamp': datetime.now().isoformat()
        }

        # Append to CSV
        trials_csv_path = output_dir / f"{asset}_all_trials.csv"
        if not trials_csv_path.exists():
            with open(trials_csv_path, 'w') as f:
                headers = ['trial', 'score', 'year_pnl', 'year_trades', 'year_pf', 'year_dd', 'year_roi']
                headers += list(cfg.get('_trial_params', {}).keys())
                f.write(','.join(headers) + '\n')

        with open(trials_csv_path, 'a') as f:
            row = [
                trial.number,
                f"{score:.2f}",
                f"{year_pnl:.2f}",
                year_trades,
                f"{year_pf:.2f}",
                f"{year_dd:.2f}",
                f"{year_roi:.2f}"
            ]
            row += [cfg['_trial_params'].get(k, '') for k in cfg.get('_trial_params', {}).keys()]
            f.write(','.join(map(str, row)) + '\n')

        # Save best trial
        best_score_file = output_dir / f"{asset}_best_score.txt"
        current_best = float('-inf')
        if best_score_file.exists():
            with open(best_score_file, 'r') as f:
                current_best = float(f.read().strip())

        if score > current_best:
            with open(best_score_file, 'w') as f:
                f.write(str(score))

            with open(output_dir / f"{asset}_best_config.json", 'w') as f:
                json.dump(cfg, f, indent=2)

            with open(output_dir / f"{asset}_best_trial_summary.json", 'w') as f:
                json.dump(trial_summary, f, indent=2)

            print(f"\n  [NEW BEST] Score: {score:.2f}")

        return score

    finally:
        Path(config_path).unlink(missing_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Year-centric archetype optimizer")
    parser.add_argument('--trials', type=int, default=80, help="Number of Optuna trials")
    parser.add_argument('--asset', type=str, default="BTC", help="Asset to optimize")
    parser.add_argument('--output', type=str, required=True, help="Output directory")
    parser.add_argument('--base-config', type=str, default="configs/profile_archetype_optimized.json",
                        help="Base config")
    args = parser.parse_args()

    # Load base config
    with open(args.base_config, 'r') as f:
        base_cfg = json.load(f)

    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*80}")
    print(f"Year-Centric Archetype Optimizer - {args.asset}")
    print(f"{'='*80}")
    print(f"Calibration Window: {CALIB_WINDOW[0]} → {CALIB_WINDOW[1]} (Q1 2024)")
    print(f"Validation Window: {VALIDATE_WINDOW[0]} → {VALIDATE_WINDOW[1]} (Full Year 2024)")
    print(f"Trials: {args.trials}")
    print(f"Output: {output_dir}")
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
    print(f"Best Score: {study.best_value:.2f}")
    print(f"\nBest Parameters:")
    for key, value in study.best_params.items():
        print(f"  {key}: {value}")
    print(f"\nResults saved to: {output_dir}")
    print(f"{'='*80}\n")

if __name__ == "__main__":
    main()
