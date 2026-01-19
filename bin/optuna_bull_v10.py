#!/usr/bin/env python3
"""
PR#6A Step 2: Bull Market Optimizer (v10)

Strategy:
- Train on 2024-01-01 to 2024-12-31 (bull market conditions)
- Start from PF-20 baseline config (configs/baseline_btc_bull_pf20.json)
- Tight search ranges (±15% around baseline values)
- Bull-focused guardrails: PF ≥ 15, DD ≤ 3%, trades 15-50, win_rate ≥ 0.55

Objective: PNL + PF_bonus - DD_penalty - trade_band_penalty

Guardrails (hard rejections):
- PF < 15.0 (must maintain exceptional bull performance)
- DD > 3.0% (tight risk control)
- Trades ∉ [15, 50] (quality over quantity)
- Win Rate < 55% (maintain edge consistency)

Parameter Search (tight ranges around PF-20 values):
- Fusion weights: wyckoff (0.38-0.50), liquidity (0.17-0.27), momentum (0.28-0.45)
- min_liquidity: 0.16-0.22 (baseline: 0.185)
- Archetype thresholds: ±0.05 around baseline values
- Sizing: size_min (0.59-0.79), size_max (1.01-1.37)
- Exits: trail_atr (1.0-1.35), max_bars (73-98)

Usage:
    python3 bin/optuna_bull_v10.py --trials 60 --output reports/bull_v10
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
# Training Window (Bull Market)
# ─────────────────────────────────────────────────────────────────────────────
TRAIN_WINDOW = ('2024-01-01', '2024-12-31')  # Full 2024 bull market

# ─────────────────────────────────────────────────────────────────────────────
# PF-20 Baseline Values (for tight search ranges)
# ─────────────────────────────────────────────────────────────────────────────
BASELINE = {
    'w_wyckoff': 0.44265,
    'w_liquidity': 0.22676,
    'w_momentum': 0.33059,
    'min_liquidity': 0.18470,
    'B_fusion': 0.35913,  # order_block_retest
    'C_fusion': 0.49429,  # failed_continuation
    'H_fusion': 0.54431,  # trap_within_trend
    'K_fusion': 0.43519,  # wick_trap
    'L_fusion': 0.34945,  # volume_exhaustion
    'size_min': 0.69456,
    'size_max': 1.18779,
    'trail_atr_mult': 1.13320,
    'max_bars': 86,
    'range_stop_factor': 0.72142,
    'trend_stop_factor': 1.24451
}

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
# Parameter Suggestions (Tight Ranges ±15% around PF-20 baseline)
# ─────────────────────────────────────────────────────────────────────────────
def suggest_params(trial, base_cfg):
    """
    Suggest parameters with tight ranges around PF-20 baseline.

    Search Strategy:
    - Stay close to proven PF-20 values
    - Allow ±15% exploration for bull market fine-tuning
    - Maintain config structure compatibility
    """
    cfg = base_cfg.copy()

    # ─── Fusion Weights (±15% around baseline, normalized) ───
    w_wyckoff_raw = trial.suggest_float('w_wyckoff',
        BASELINE['w_wyckoff'] * 0.85, BASELINE['w_wyckoff'] * 1.15, step=0.02)
    w_liquidity_raw = trial.suggest_float('w_liquidity',
        BASELINE['w_liquidity'] * 0.85, BASELINE['w_liquidity'] * 1.15, step=0.02)
    w_momentum_raw = trial.suggest_float('w_momentum',
        BASELINE['w_momentum'] * 0.85, BASELINE['w_momentum'] * 1.15, step=0.02)

    total_weight = w_wyckoff_raw + w_liquidity_raw + w_momentum_raw
    w_wyckoff = w_wyckoff_raw / total_weight
    w_liquidity = w_liquidity_raw / total_weight
    w_momentum = w_momentum_raw / total_weight

    # ─── Entry Gating (±15%) ───
    min_liquidity = trial.suggest_float('min_liquidity',
        BASELINE['min_liquidity'] * 0.85, BASELINE['min_liquidity'] * 1.15, step=0.01)

    # ─── Archetype Fusion Thresholds (±15%) ───
    B_fusion = trial.suggest_float('B_fusion',
        BASELINE['B_fusion'] * 0.85, BASELINE['B_fusion'] * 1.15, step=0.02)
    C_fusion = trial.suggest_float('C_fusion',
        BASELINE['C_fusion'] * 0.85, BASELINE['C_fusion'] * 1.15, step=0.02)
    H_fusion = trial.suggest_float('H_fusion',
        BASELINE['H_fusion'] * 0.85, BASELINE['H_fusion'] * 1.15, step=0.02)
    K_fusion = trial.suggest_float('K_fusion',
        BASELINE['K_fusion'] * 0.85, BASELINE['K_fusion'] * 1.15, step=0.02)
    L_fusion = trial.suggest_float('L_fusion',
        BASELINE['L_fusion'] * 0.85, BASELINE['L_fusion'] * 1.15, step=0.02)

    # ─── Dynamic Sizing (±15%) ───
    size_min = trial.suggest_float('size_min',
        BASELINE['size_min'] * 0.85, BASELINE['size_min'] * 1.15, step=0.05)
    size_max = trial.suggest_float('size_max',
        BASELINE['size_max'] * 0.85, BASELINE['size_max'] * 1.15, step=0.05)

    # ─── Exits (±15%) ───
    trail_atr_mult = trial.suggest_float('trail_atr_mult',
        BASELINE['trail_atr_mult'] * 0.85, BASELINE['trail_atr_mult'] * 1.15, step=0.05)
    max_bars = trial.suggest_int('max_bars',
        int(BASELINE['max_bars'] * 0.85), int(BASELINE['max_bars'] * 1.15), step=2)
    range_stop_factor = trial.suggest_float('range_stop_factor',
        BASELINE['range_stop_factor'] * 0.85, BASELINE['range_stop_factor'] * 1.15, step=0.05)
    trend_stop_factor = trial.suggest_float('trend_stop_factor',
        BASELINE['trend_stop_factor'] * 0.85, BASELINE['trend_stop_factor'] * 1.15, step=0.05)

    # ─── Apply to config ───
    # Fusion weights
    if 'fusion' not in cfg:
        cfg['fusion'] = {}
    if 'weights' not in cfg['fusion']:
        cfg['fusion']['weights'] = {}

    cfg['fusion']['weights']['wyckoff'] = round(w_wyckoff, 3)
    cfg['fusion']['weights']['liquidity'] = round(w_liquidity, 3)
    cfg['fusion']['weights']['momentum'] = round(w_momentum, 3)
    cfg['fusion']['weights']['smc'] = 0.0  # Keep SMC disabled

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
    cfg['decision_gates']['size_min'] = size_min
    cfg['decision_gates']['size_max'] = size_max

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
# Objective Function
# ─────────────────────────────────────────────────────────────────────────────
def objective(trial, base_cfg, asset, output_dir):
    """
    Bull market objective with strict guardrails.

    Process:
    1. Run 2024 backtest
    2. Apply bull guardrails (PF ≥15, DD ≤3%, trades ∈ [15,50], WR ≥55%)
    3. Score = PNL + PF_bonus - DD_penalty - trade_band_penalty
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
        # ─── Run 2024 Backtest ───
        print(f"\n  [BULL 2024] Running {TRAIN_WINDOW[0]} → {TRAIN_WINDOW[1]}...")
        metrics = run_backtest(config_path, TRAIN_WINDOW[0], TRAIN_WINDOW[1], asset)

        trades = metrics['trades']
        pnl = metrics['pnl']
        pf = metrics['profit_factor']
        dd = metrics['drawdown']
        roi = metrics['roi']
        wr = metrics['win_rate']
        sharpe = metrics['sharpe']

        print(f"    Trades: {trades}, PNL: ${pnl:.2f}, ROI: {roi:.1f}%")
        print(f"    PF: {pf:.2f}, DD: {dd:.1f}%, WR: {wr:.1f}%, Sharpe: {sharpe:.2f}")

        # ─── Apply Bull Guardrails ───
        guardrails_pass = True
        reject_reason = []

        if pf < 15.0:
            guardrails_pass = False
            reject_reason.append(f"PF={pf:.2f} < 15.0")

        if dd > 3.0:
            guardrails_pass = False
            reject_reason.append(f"DD={dd:.1f}% > 3%")

        if trades < 15:
            guardrails_pass = False
            reject_reason.append(f"Trades={trades} < 15")
        elif trades > 50:
            guardrails_pass = False
            reject_reason.append(f"Trades={trades} > 50")

        if wr < 55.0:
            guardrails_pass = False
            reject_reason.append(f"WR={wr:.1f}% < 55%")

        if not guardrails_pass:
            print(f"\n  [REJECT] Bull guardrails failed: {', '.join(reject_reason)}")
            return float('-inf')

        # ─── Compute Score ───
        score = pnl

        # PF bonus (scale: PF 15-25 → +3000-5000 bonus)
        pf_capped = min(pf, 25.0)
        pf_bonus = pf_capped * 200
        score += pf_bonus

        # DD penalty (aggressive for bull: 1% DD = -3000 penalty)
        dd_penalty = dd * 3000
        score -= dd_penalty

        # Trade band penalty (pressure toward 25-35 range)
        if trades < 25:
            trade_penalty = (25 - trades) * 20
            score -= trade_penalty
        elif trades > 35:
            trade_penalty = (trades - 35) * 20
            score -= trade_penalty

        print(f"\n  [SCORE] PNL: {pnl:.2f}, PF Bonus: {pf_bonus:.2f}, DD Penalty: {dd_penalty:.2f}")
        print(f"  [FINAL] Score: {score:.2f}")

        # ─── Log Results ───
        trial_summary = {
            'trial_number': trial.number,
            'score': score,
            'params': cfg.get('_trial_params', {}),
            'metrics': {
                'pnl': pnl,
                'trades': trades,
                'roi': roi,
                'pf': pf,
                'dd': dd,
                'win_rate': wr,
                'sharpe': sharpe
            },
            'timestamp': datetime.now().isoformat()
        }

        # Append to CSV
        trials_csv_path = output_dir / f"{asset}_bull_v10_trials.csv"
        if not trials_csv_path.exists():
            with open(trials_csv_path, 'w') as f:
                headers = ['trial', 'score', 'pnl', 'trades', 'pf', 'dd', 'wr', 'roi']
                headers += list(cfg.get('_trial_params', {}).keys())
                f.write(','.join(headers) + '\n')

        with open(trials_csv_path, 'a') as f:
            row = [
                trial.number,
                f"{score:.2f}",
                f"{pnl:.2f}",
                trades,
                f"{pf:.2f}",
                f"{dd:.2f}",
                f"{wr:.1f}",
                f"{roi:.2f}"
            ]
            row += [cfg['_trial_params'].get(k, '') for k in cfg.get('_trial_params', {}).keys()]
            f.write(','.join(map(str, row)) + '\n')

        # Save best trial
        best_score_file = output_dir / f"{asset}_bull_v10_best_score.txt"
        current_best = float('-inf')
        if best_score_file.exists():
            with open(best_score_file, 'r') as f:
                current_best = float(f.read().strip())

        if score > current_best:
            with open(best_score_file, 'w') as f:
                f.write(str(score))

            with open(output_dir / f"{asset}_bull_v10.json", 'w') as f:
                # Add metadata
                cfg['version'] = '2.1.0-bull-v10'
                cfg['profile'] = 'bull_v10_optimized'
                cfg['description'] = f'Bull market config optimized on 2024. PF: {pf:.2f}, DD: {dd:.1f}%, WR: {wr:.1f}%'
                json.dump(cfg, f, indent=2)

            with open(output_dir / f"{asset}_bull_v10_summary.json", 'w') as f:
                json.dump(trial_summary, f, indent=2)

            print(f"\n  [NEW BEST] Score: {score:.2f} | PF: {pf:.2f}, DD: {dd:.1f}%, Trades: {trades}")

        return score

    finally:
        Path(config_path).unlink(missing_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Bull market optimizer (v10)")
    parser.add_argument('--trials', type=int, default=60, help="Number of Optuna trials")
    parser.add_argument('--asset', type=str, default="BTC", help="Asset to optimize")
    parser.add_argument('--output', type=str, required=True, help="Output directory")
    parser.add_argument('--base-config', type=str, default="configs/baseline_btc_bull_pf20.json",
                        help="Base config (PF-20 baseline)")
    args = parser.parse_args()

    # Load base config
    base_config_path = Path(args.base_config)
    if not base_config_path.exists():
        print(f"❌ Base config not found: {base_config_path}")
        print(f"   Expected PF-20 baseline at: configs/baseline_btc_bull_pf20.json")
        return 1

    with open(base_config_path, 'r') as f:
        base_cfg = json.load(f)

    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*80}")
    print(f"Bull Market Optimizer v10 - {args.asset}")
    print(f"{'='*80}")
    print(f"Training Window: {TRAIN_WINDOW[0]} → {TRAIN_WINDOW[1]} (2024 Bull)")
    print(f"Base Config: {args.base_config}")
    print(f"Trials: {args.trials}")
    print(f"Output: {output_dir}")
    print(f"\nBull Guardrails:")
    print(f"  - PF ≥ 15.0 (maintain exceptional performance)")
    print(f"  - DD ≤ 3.0% (tight risk control)")
    print(f"  - Trades ∈ [15, 50] (quality over quantity)")
    print(f"  - Win Rate ≥ 55% (maintain edge)")
    print(f"\nSearch Strategy:")
    print(f"  - Tight ranges (±15% around PF-20 baseline)")
    print(f"  - Focus on bull market fine-tuning")
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
    if study.best_trial:
        print(f"Best Trial: #{study.best_trial.number}")
        print(f"Best Score: {study.best_value:.2f}")
        print(f"\nBest Parameters:")
        for key, value in study.best_params.items():
            if key in BASELINE:
                baseline_val = BASELINE[key]
                delta_pct = ((value - baseline_val) / baseline_val) * 100
                print(f"  {key}: {value:.4f} (baseline: {baseline_val:.4f}, Δ{delta_pct:+.1f}%)")
            else:
                print(f"  {key}: {value:.4f}")
        print(f"\nResults saved to: {output_dir}")
        print(f"  - Config: {output_dir}/{args.asset}_bull_v10.json")
        print(f"  - Summary: {output_dir}/{args.asset}_bull_v10_summary.json")
        print(f"  - Trials: {output_dir}/{args.asset}_bull_v10_trials.csv")
    else:
        print(f"⚠️  No trials passed bull guardrails!")
        print(f"   Consider relaxing constraints or widening search ranges.")
    print(f"{'='*80}\n")

    return 0

if __name__ == "__main__":
    exit(main())
