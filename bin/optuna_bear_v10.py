#!/usr/bin/env python3
"""
Optuna optimization for bear market config (2022 capital preservation).

Problem: 2022 lost -$965 with 25% WR (32 trades)
Target: Reduce loss to -$400 max, improve WR to 35%+
Expected gain: +$565 total improvement

Strategy:
- Train on 2022 H1 (Jan-Jun), validate on 2022 H2 (Jul-Dec)
- Focus on capital preservation in crisis conditions
- Optimize risk parameters and quality gates for bear config
- Integrate with RouterV10 for regime-aware config switching
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
# Training Window (Bear Market)
# ─────────────────────────────────────────────────────────────────────────────
TRAIN_WINDOW = ('2022-01-01', '2023-12-31')  # Luna crash, FTX collapse, bear grind

# ─────────────────────────────────────────────────────────────────────────────
# Defensive Baseline Values (from baseline_btc_bear_defensive.json)
# ─────────────────────────────────────────────────────────────────────────────
BASELINE = {
    # Fusion weights
    'w_wyckoff': 0.25,
    'w_liquidity': 0.30,
    'w_momentum': 0.25,
    'w_smc': 0.20,

    # Position sizing
    'base_risk_pct': 0.8,
    'max_risk_pct': 1.5,
    'kelly_fraction': 0.20,

    # Stops
    'trailing_atr_mult': 2.0,
    'initial_stop_atr_mult': 2.5,
    'breakeven_atr': 1.5,
    'profit_lock_atr': 0.8,

    # Targets
    'primary_rr': 2.5,
    'secondary_rr': 4.0,
    'partial_exit_pct': 50,

    # Quality gates
    'min_structural_quality': 0.65,
    'min_timing_quality': 0.60,
    'min_context_quality': 0.55,

    # Archetype fusion
    'quality_boost_threshold': 0.75,
    'quality_boost_multiplier': 1.20,
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
# Parameter Suggestions (Conservative Ranges ±15% around defensive baseline)
# ─────────────────────────────────────────────────────────────────────────────
def suggest_params(trial, base_cfg):
    """
    Suggest parameters with conservative ranges for bear market optimization.

    Search Strategy:
    - Preserve defensive positioning (lower risk, wider stops)
    - Fine-tune for capital preservation
    - Allow ±15% exploration around baseline
    """
    cfg = base_cfg.copy()

    # ─── Fusion Weights (±15% around baseline, normalized) ───
    w_wyckoff_raw = trial.suggest_float('w_wyckoff',
        BASELINE['w_wyckoff'] * 0.85, BASELINE['w_wyckoff'] * 1.15, step=0.02)
    w_liquidity_raw = trial.suggest_float('w_liquidity',
        BASELINE['w_liquidity'] * 0.85, BASELINE['w_liquidity'] * 1.15, step=0.02)
    w_momentum_raw = trial.suggest_float('w_momentum',
        BASELINE['w_momentum'] * 0.85, BASELINE['w_momentum'] * 1.15, step=0.02)
    w_smc_raw = trial.suggest_float('w_smc',
        BASELINE['w_smc'] * 0.85, BASELINE['w_smc'] * 1.15, step=0.02)

    total_weight = w_wyckoff_raw + w_liquidity_raw + w_momentum_raw + w_smc_raw
    w_wyckoff = w_wyckoff_raw / total_weight
    w_liquidity = w_liquidity_raw / total_weight
    w_momentum = w_momentum_raw / total_weight
    w_smc = w_smc_raw / total_weight

    # ─── Position Sizing (±15%) ───
    base_risk_pct = trial.suggest_float('base_risk_pct',
        BASELINE['base_risk_pct'] * 0.85, BASELINE['base_risk_pct'] * 1.15, step=0.05)
    max_risk_pct = trial.suggest_float('max_risk_pct',
        BASELINE['max_risk_pct'] * 0.85, BASELINE['max_risk_pct'] * 1.15, step=0.05)
    kelly_fraction = trial.suggest_float('kelly_fraction',
        BASELINE['kelly_fraction'] * 0.85, BASELINE['kelly_fraction'] * 1.15, step=0.01)

    # ─── Stops (±15%) ───
    trailing_atr_mult = trial.suggest_float('trailing_atr_mult',
        BASELINE['trailing_atr_mult'] * 0.85, BASELINE['trailing_atr_mult'] * 1.15, step=0.05)
    initial_stop_atr_mult = trial.suggest_float('initial_stop_atr_mult',
        BASELINE['initial_stop_atr_mult'] * 0.85, BASELINE['initial_stop_atr_mult'] * 1.15, step=0.05)
    breakeven_atr = trial.suggest_float('breakeven_atr',
        BASELINE['breakeven_atr'] * 0.85, BASELINE['breakeven_atr'] * 1.15, step=0.05)
    profit_lock_atr = trial.suggest_float('profit_lock_atr',
        BASELINE['profit_lock_atr'] * 0.85, BASELINE['profit_lock_atr'] * 1.15, step=0.05)

    # ─── Targets (±15%) ───
    primary_rr = trial.suggest_float('primary_rr',
        BASELINE['primary_rr'] * 0.85, BASELINE['primary_rr'] * 1.15, step=0.1)
    secondary_rr = trial.suggest_float('secondary_rr',
        BASELINE['secondary_rr'] * 0.85, BASELINE['secondary_rr'] * 1.15, step=0.1)
    partial_exit_pct = trial.suggest_int('partial_exit_pct',
        int(BASELINE['partial_exit_pct'] * 0.85), int(BASELINE['partial_exit_pct'] * 1.15), step=5)

    # ─── Quality Gates (±15%) ───
    min_structural_quality = trial.suggest_float('min_structural_quality',
        BASELINE['min_structural_quality'] * 0.85, BASELINE['min_structural_quality'] * 1.15, step=0.02)
    min_timing_quality = trial.suggest_float('min_timing_quality',
        BASELINE['min_timing_quality'] * 0.85, BASELINE['min_timing_quality'] * 1.15, step=0.02)
    min_context_quality = trial.suggest_float('min_context_quality',
        BASELINE['min_context_quality'] * 0.85, BASELINE['min_context_quality'] * 1.15, step=0.02)

    # ─── Archetype Fusion (±15%) ───
    quality_boost_threshold = trial.suggest_float('quality_boost_threshold',
        BASELINE['quality_boost_threshold'] * 0.85, BASELINE['quality_boost_threshold'] * 1.15, step=0.02)
    quality_boost_multiplier = trial.suggest_float('quality_boost_multiplier',
        BASELINE['quality_boost_multiplier'] * 0.85, BASELINE['quality_boost_multiplier'] * 1.15, step=0.02)

    # ─── Apply to config ───
    # Fusion weights
    if 'fusion' not in cfg:
        cfg['fusion'] = {}
    if 'weights' not in cfg['fusion']:
        cfg['fusion']['weights'] = {}

    cfg['fusion']['weights']['wyckoff'] = round(w_wyckoff, 3)
    cfg['fusion']['weights']['liquidity'] = round(w_liquidity, 3)
    cfg['fusion']['weights']['momentum'] = round(w_momentum, 3)
    cfg['fusion']['weights']['smc'] = round(w_smc, 3)

    # Position sizing
    if 'position_sizing' not in cfg:
        cfg['position_sizing'] = {}

    cfg['position_sizing']['base_risk_per_trade_pct'] = base_risk_pct
    cfg['position_sizing']['max_risk_per_trade_pct'] = max_risk_pct
    cfg['position_sizing']['kelly_fraction'] = kelly_fraction

    # Stops
    if 'stops' not in cfg:
        cfg['stops'] = {}

    cfg['stops']['trailing_atr_multiplier'] = trailing_atr_mult
    cfg['stops']['initial_stop_atr_multiplier'] = initial_stop_atr_mult
    cfg['stops']['breakeven_profit_threshold_atr'] = breakeven_atr
    cfg['stops']['profit_lock_interval_atr'] = profit_lock_atr

    # Targets
    if 'targets' not in cfg:
        cfg['targets'] = {}

    cfg['targets']['primary_target_rr'] = primary_rr
    cfg['targets']['secondary_target_rr'] = secondary_rr
    cfg['targets']['partial_exit_at_primary_pct'] = partial_exit_pct

    # Quality gates
    if 'archetype_gates' not in cfg:
        cfg['archetype_gates'] = {}

    cfg['archetype_gates']['min_structural_quality'] = min_structural_quality
    cfg['archetype_gates']['min_timing_quality'] = min_timing_quality
    cfg['archetype_gates']['min_context_quality'] = min_context_quality

    # Archetype fusion
    if 'archetype_fusion' not in cfg:
        cfg['archetype_fusion'] = {}

    cfg['archetype_fusion']['quality_boost_threshold'] = quality_boost_threshold
    cfg['archetype_fusion']['quality_boost_multiplier'] = quality_boost_multiplier

    # Store trial params
    cfg['_trial_params'] = {
        'w_wyckoff': w_wyckoff,
        'w_liquidity': w_liquidity,
        'w_momentum': w_momentum,
        'w_smc': w_smc,
        'base_risk_pct': base_risk_pct,
        'max_risk_pct': max_risk_pct,
        'kelly_fraction': kelly_fraction,
        'trailing_atr_mult': trailing_atr_mult,
        'initial_stop_atr_mult': initial_stop_atr_mult,
        'breakeven_atr': breakeven_atr,
        'profit_lock_atr': profit_lock_atr,
        'primary_rr': primary_rr,
        'secondary_rr': secondary_rr,
        'partial_exit_pct': partial_exit_pct,
        'min_structural_quality': min_structural_quality,
        'min_timing_quality': min_timing_quality,
        'min_context_quality': min_context_quality,
        'quality_boost_threshold': quality_boost_threshold,
        'quality_boost_multiplier': quality_boost_multiplier,
    }

    return cfg

# ─────────────────────────────────────────────────────────────────────────────
# Objective Function
# ─────────────────────────────────────────────────────────────────────────────
def objective(trial, base_cfg, asset, output_dir):
    """
    Bear market objective with defensive guardrails.

    Process:
    1. Run 2022-2023 backtest (bear market)
    2. Apply bear guardrails (PF ≥1.8, DD ≤8%, trades ≥40, WR ≥50%)
    3. Score = PNL + PF_bonus - DD_penalty + trade_bonus
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
        # ─── Run 2022-2023 Backtest ───
        print(f"\n  [BEAR 2022-2023] Running {TRAIN_WINDOW[0]} → {TRAIN_WINDOW[1]}...")
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

        # ─── Apply Bear Guardrails ───
        guardrails_pass = True
        reject_reason = []

        if pf < 1.8:
            guardrails_pass = False
            reject_reason.append(f"PF={pf:.2f} < 1.8")

        if dd > 8.0:
            guardrails_pass = False
            reject_reason.append(f"DD={dd:.1f}% > 8%")

        if trades < 40:
            guardrails_pass = False
            reject_reason.append(f"Trades={trades} < 40")

        if wr < 50.0:
            guardrails_pass = False
            reject_reason.append(f"WR={wr:.1f}% < 50%")

        if not guardrails_pass:
            print(f"\n  [REJECT] Bear guardrails failed: {', '.join(reject_reason)}")
            return float('-inf')

        # ─── Compute Score ───
        score = pnl

        # PF bonus (conservative: PF 1.8-3.5 → +600-1200 bonus)
        pf_capped = min(pf, 3.5)
        pf_bonus = (pf_capped - 1.0) * 400  # Scale: 1.8→320, 3.5→1000
        score += pf_bonus

        # DD penalty (defensive: 1% DD = -400 penalty)
        dd_penalty = dd * 400
        score -= dd_penalty

        # Trade bonus (reward 60-80 range for bear market sampling)
        if trades >= 60 and trades <= 80:
            trade_bonus = 300  # Bonus for good sample size
            score += trade_bonus
        elif trades < 60:
            trade_penalty = (60 - trades) * 10  # Mild penalty below 60
            score -= trade_penalty
        # No penalty above 80 (more trades OK in bear)

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
        trials_csv_path = output_dir / f"{asset}_bear_v10_trials.csv"
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
        best_score_file = output_dir / f"{asset}_bear_v10_best_score.txt"
        current_best = float('-inf')
        if best_score_file.exists():
            with open(best_score_file, 'r') as f:
                current_best = float(f.read().strip())

        if score > current_best:
            with open(best_score_file, 'w') as f:
                f.write(str(score))

            with open(output_dir / f"{asset}_bear_v10.json", 'w') as f:
                # Add metadata
                cfg['version'] = 'bear_v10_optimized'
                cfg['profile'] = 'bear_defensive'
                cfg['description'] = f'Bear market config optimized on 2022-2023. PF: {pf:.2f}, DD: {dd:.1f}%, WR: {wr:.1f}%'
                json.dump(cfg, f, indent=2)

            with open(output_dir / f"{asset}_bear_v10_summary.json", 'w') as f:
                json.dump(trial_summary, f, indent=2)

            print(f"\n  [NEW BEST] Score: {score:.2f} | PF: {pf:.2f}, DD: {dd:.1f}%, Trades: {trades}")

        return score

    finally:
        Path(config_path).unlink(missing_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Bear market optimizer (v10)")
    parser.add_argument('--trials', type=int, default=60, help="Number of Optuna trials")
    parser.add_argument('--asset', type=str, default="BTC", help="Asset to optimize")
    parser.add_argument('--output', type=str, required=True, help="Output directory")
    parser.add_argument('--base-config', type=str, default="configs/baseline_btc_bear_defensive.json",
                        help="Base config (bear defensive baseline)")
    args = parser.parse_args()

    # Load base config
    base_config_path = Path(args.base_config)
    if not base_config_path.exists():
        print(f"❌ Base config not found: {base_config_path}")
        print(f"   Expected bear defensive baseline at: configs/baseline_btc_bear_defensive.json")
        return 1

    with open(base_config_path, 'r') as f:
        base_cfg = json.load(f)

    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*80}")
    print(f"Bear Market Optimizer v10 - {args.asset}")
    print(f"{'='*80}")
    print(f"Training Window: {TRAIN_WINDOW[0]} → {TRAIN_WINDOW[1]} (2022-2023 Bear)")
    print(f"Base Config: {args.base_config}")
    print(f"Trials: {args.trials}")
    print(f"Output: {output_dir}")
    print(f"\nBear Guardrails (Capital Preservation Focus):")
    print(f"  - PF ≥ 1.8 (maintain edge in adverse conditions)")
    print(f"  - DD ≤ 8.0% (defensive risk control)")
    print(f"  - Trades ≥ 40 (sufficient sample size)")
    print(f"  - Win Rate ≥ 50% (better than coin flip)")
    print(f"\nSearch Strategy:")
    print(f"  - Conservative ranges (±15% around defensive baseline)")
    print(f"  - Focus on capital preservation and risk-adjusted returns")
    print(f"  - Quality gates to filter low-probability setups")
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
        print(f"  - Config: {output_dir}/{args.asset}_bear_v10.json")
        print(f"  - Summary: {output_dir}/{args.asset}_bear_v10_summary.json")
        print(f"  - Trials: {output_dir}/{args.asset}_bear_v10_trials.csv")
    else:
        print(f"⚠️  No trials passed bear guardrails!")
        print(f"   Consider relaxing constraints or widening search ranges.")
    print(f"{'='*80}\n")

    return 0

if __name__ == "__main__":
    exit(main())
