#!/usr/bin/env python3
"""
PR#6A: Optuna Archetype Threshold & Signal Neutralization Optimizer

Comprehensive parameter search targeting:
1. Signal neutralization aggressiveness (fusion floor, drop tolerance, PTI margins)
2. Fusion weight rebalancing (wyckoff, liquidity, momentum, SMC)
3. Entry gating thresholds (min_liquidity, archetype-specific fusion mins)
4. Dynamic sizing bounds and per-archetype multipliers
5. Optional exit parameter tuning

Walk-forward validation:
- Calibrate on Q2 2024 (Apr-Jun)
- Validate on Q3 2024 (Jul-Sep)
- Report Q4 2024 (not in objective)

Objective: Harmonic mean of Q2/Q3 scores with DD penalties, overtrading penalties, PF bonus
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
# Helper: Run backtest and extract metrics
# ─────────────────────────────────────────────────────────────────────────────
def run_backtest(config_path: str, start_date: str, end_date: str, asset: str = "BTC"):
    """
    Run backtest_knowledge_v2.py with given config and extract metrics.
    Returns dict with: pnl, trades, roi, win_rate, drawdown, profit_factor
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

        # Extract metrics using regex
        metrics = {
            'pnl': 0.0,
            'trades': 0,
            'roi': 0.0,
            'win_rate': 0.0,
            'drawdown': 0.0,
            'profit_factor': 0.0,
            'sharpe': 0.0,
            'avg_pnl_per_trade': 0.0
        }

        # PNL patterns
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

        # Drawdown (extract max DD)
        dd_match = re.search(r'Max Drawdown:\s+([\d\.]+)%', output)
        if dd_match:
            metrics['drawdown'] = float(dd_match.group(1))

        # Profit Factor
        pf_match = re.search(r'Profit Factor:\s+([\d\.]+)', output)
        if pf_match:
            metrics['profit_factor'] = float(pf_match.group(1))

        # Sharpe
        sharpe_match = re.search(r'Sharpe:\s+([-\d\.]+)', output)
        if sharpe_match:
            metrics['sharpe'] = float(sharpe_match.group(1))

        # Average PNL per trade
        if metrics['trades'] > 0:
            metrics['avg_pnl_per_trade'] = metrics['pnl'] / metrics['trades']

        return metrics

    except subprocess.TimeoutExpired:
        print(f"⚠️  Backtest timeout for {start_date} to {end_date}")
        return None
    except Exception as e:
        print(f"⚠️  Backtest error: {e}")
        return None


# ─────────────────────────────────────────────────────────────────────────────
# Parameter Suggestion
# ─────────────────────────────────────────────────────────────────────────────
def suggest_params(trial, base_cfg):
    """
    Suggest parameters for this trial and return modified config dict.
    """
    cfg = json.loads(json.dumps(base_cfg))  # Deep copy

    # ═════════════════════════════════════════════════════════════════════════
    # 1) Signal Neutralization Aggressiveness
    # ═════════════════════════════════════════════════════════════════════════
    if "decision_gates" not in cfg:
        cfg["decision_gates"] = {}

    # Final fusion floor (minimum fusion before neutralization)
    cfg["decision_gates"]["final_fusion_floor"] = trial.suggest_float(
        "final_fusion_floor", 0.32, 0.45, step=0.01
    )

    # Fusion drop tolerance (how much drop triggers neutralization)
    cfg["decision_gates"]["neutralize_fusion_drop"] = trial.suggest_float(
        "neutralize_fusion_drop", 0.05, 0.18, step=0.01
    )

    # Minimum bars before neutralization can trigger
    cfg["decision_gates"]["neutralize_min_bars"] = trial.suggest_int(
        "neutralize_min_bars", 2, 8
    )

    # PTI reversal margin (how much PTI flip triggers exit)
    cfg["decision_gates"]["pti_reversal_margin"] = trial.suggest_float(
        "pti_reversal_margin", 0.10, 0.30, step=0.02
    )

    # Liquidity falloff threshold (exit if liquidity drops below this)
    cfg["decision_gates"]["liquidity_falloff_threshold"] = trial.suggest_float(
        "liquidity_falloff_threshold", 0.20, 0.35, step=0.02
    )

    # ═════════════════════════════════════════════════════════════════════════
    # 2) Fusion Weight Rebalancing
    # ═════════════════════════════════════════════════════════════════════════
    if "fusion" not in cfg:
        cfg["fusion"] = {}
    if "weights" not in cfg["fusion"]:
        cfg["fusion"]["weights"] = {}

    # Suggest weights (must sum to 1.0)
    w_wyckoff = trial.suggest_float("w_wyckoff", 0.15, 0.55, step=0.05)
    w_liquidity = trial.suggest_float("w_liquidity", 0.15, 0.55, step=0.05)
    w_momentum = trial.suggest_float("w_momentum", 0.05, 0.45, step=0.05)

    # SMC gets remainder
    w_sum = w_wyckoff + w_liquidity + w_momentum
    w_smc = max(0.05, 1.0 - w_sum)

    # Normalize if needed
    total = w_wyckoff + w_liquidity + w_momentum + w_smc
    cfg["fusion"]["weights"]["wyckoff"] = w_wyckoff / total
    cfg["fusion"]["weights"]["liquidity"] = w_liquidity / total
    cfg["fusion"]["weights"]["momentum"] = w_momentum / total
    cfg["fusion"]["weights"]["smc"] = w_smc / total

    # ═════════════════════════════════════════════════════════════════════════
    # 3) Entry Gating Thresholds
    # ═════════════════════════════════════════════════════════════════════════
    if "archetypes" not in cfg:
        cfg["archetypes"] = {"thresholds": {}}
    if "thresholds" not in cfg["archetypes"]:
        cfg["archetypes"]["thresholds"] = {}

    # Global minimum liquidity
    cfg["archetypes"]["thresholds"]["min_liquidity"] = trial.suggest_float(
        "min_liquidity", 0.10, 0.28, step=0.02
    )

    # Gate5 liquidity threshold (second-stage gating)
    cfg["decision_gates"]["gate5_liquidity_threshold"] = trial.suggest_float(
        "gate5_liquidity", 0.25, 0.40, step=0.02
    )

    # Archetype-specific fusion minimums
    # B: order_block_retest (high quality, can lower)
    if "B" not in cfg["archetypes"]["thresholds"]:
        cfg["archetypes"]["thresholds"]["B"] = {}
    cfg["archetypes"]["thresholds"]["B"]["fusion"] = trial.suggest_float(
        "B_fusion", 0.30, 0.42, step=0.02
    )

    # K: wick_trap (needs tightening)
    if "K" not in cfg["archetypes"]["thresholds"]:
        cfg["archetypes"]["thresholds"]["K"] = {}
    cfg["archetypes"]["thresholds"]["K"]["fusion"] = trial.suggest_float(
        "K_fusion", 0.32, 0.44, step=0.02
    )

    # L: volume_exhaustion (needs tightening)
    if "L" not in cfg["archetypes"]["thresholds"]:
        cfg["archetypes"]["thresholds"]["L"] = {}
    cfg["archetypes"]["thresholds"]["L"]["fusion"] = trial.suggest_float(
        "L_fusion", 0.34, 0.46, step=0.02
    )

    # C: failed_continuation (already tuned, narrow range)
    if "C" not in cfg["archetypes"]["thresholds"]:
        cfg["archetypes"]["thresholds"]["C"] = {}
    cfg["archetypes"]["thresholds"]["C"]["fusion"] = trial.suggest_float(
        "C_fusion", 0.42, 0.50, step=0.02
    )

    # H: trap_within_trend (strict to prevent dominance)
    if "H" not in cfg["archetypes"]["thresholds"]:
        cfg["archetypes"]["thresholds"]["H"] = {}
    cfg["archetypes"]["thresholds"]["H"]["fusion"] = trial.suggest_float(
        "H_fusion", 0.50, 0.60, step=0.02
    )

    # ═════════════════════════════════════════════════════════════════════════
    # 4) Dynamic Sizing Bounds
    # ═════════════════════════════════════════════════════════════════════════
    cfg["decision_gates"]["sizing_min_leverage"] = trial.suggest_float(
        "size_min", 0.5, 0.8, step=0.05
    )
    cfg["decision_gates"]["sizing_max_leverage"] = trial.suggest_float(
        "size_max", 1.0, 1.5, step=0.05
    )

    # Archetype-specific sizing multipliers
    if "archetype_sizing" not in cfg:
        cfg["archetype_sizing"] = {}

    cfg["archetype_sizing"]["order_block_retest"] = trial.suggest_float(
        "size_B", 1.1, 1.4, step=0.05
    )
    cfg["archetype_sizing"]["wick_trap"] = trial.suggest_float(
        "size_K", 0.9, 1.2, step=0.05
    )
    cfg["archetype_sizing"]["volume_exhaustion"] = trial.suggest_float(
        "size_L", 0.9, 1.2, step=0.05
    )
    cfg["archetype_sizing"]["failed_continuation"] = trial.suggest_float(
        "size_C", 1.0, 1.3, step=0.05
    )
    cfg["archetype_sizing"]["trap_within_trend"] = trial.suggest_float(
        "size_H", 0.6, 0.9, step=0.05
    )

    # ═════════════════════════════════════════════════════════════════════════
    # 5) Optional: Exit Parameters
    # ═════════════════════════════════════════════════════════════════════════
    # Base trailing ATR multiplier
    if "pnl_tracker" not in cfg:
        cfg["pnl_tracker"] = {"exits": {}}
    if "exits" not in cfg["pnl_tracker"]:
        cfg["pnl_tracker"]["exits"] = {}

    cfg["pnl_tracker"]["exits"]["trail_atr_mult"] = trial.suggest_float(
        "trail_atr_mult", 0.8, 1.3, step=0.1
    )

    # Max bars in trade (prevent runaway holds)
    cfg["pnl_tracker"]["exits"]["max_bars_in_trade"] = trial.suggest_int(
        "max_bars", 48, 120, step=12
    )

    # Regime adaptive factors
    cfg["pnl_tracker"]["exits"]["range_stop_factor"] = trial.suggest_float(
        "range_stop_factor", 0.6, 0.9, step=0.05
    )
    cfg["pnl_tracker"]["exits"]["trend_stop_factor"] = trial.suggest_float(
        "trend_stop_factor", 1.1, 1.4, step=0.05
    )

    return cfg


# ─────────────────────────────────────────────────────────────────────────────
# Objective Function
# ─────────────────────────────────────────────────────────────────────────────
def objective(trial, base_cfg, asset):
    """
    Walk-forward objective:
    1. Calibrate on Q2 2024 (Apr-Jun)
    2. Validate on Q3 2024 (Jul-Sep)
    3. Compute harmonic mean with penalties
    """
    # Generate trial config
    trial_cfg = suggest_params(trial, base_cfg)

    # Write temporary config
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(trial_cfg, f, indent=2)
        cfg_path = f.name

    try:
        # ─────────────────────────────────────────────────────────────────────
        # Q2 2024: Calibration
        # ─────────────────────────────────────────────────────────────────────
        print(f"\n[Trial {trial.number}] Running Q2 2024 (calibration)...")
        m2 = run_backtest(cfg_path, "2024-04-01", "2024-06-30", asset)

        if m2 is None or m2['trades'] == 0:
            print(f"⚠️  Q2 failed or zero trades")
            return -9999.0

        # ─────────────────────────────────────────────────────────────────────
        # Q3 2024: Validation
        # ─────────────────────────────────────────────────────────────────────
        print(f"[Trial {trial.number}] Running Q3 2024 (validation)...")
        m3 = run_backtest(cfg_path, "2024-07-01", "2024-09-30", asset)

        if m3 is None or m3['trades'] == 0:
            print(f"⚠️  Q3 failed or zero trades")
            return -9999.0

        # ─────────────────────────────────────────────────────────────────────
        # Compute Scores
        # ─────────────────────────────────────────────────────────────────────
        def compute_score(m):
            """
            Score = PNL + PF_bonus - DD_penalty - overtrade_penalty
            """
            pnl = m['pnl']
            dd = m['drawdown'] / 100.0  # Convert % to decimal
            trades = m['trades']
            pf = m['profit_factor']

            # Drawdown penalty: Penalize DD > 8%
            dd_pen = max(0.0, dd - 0.08) * 2000.0

            # Overtrading penalty: Penalize > 80 trades per quarter
            tr_pen = max(0, trades - 80) * 10.0

            # Profit factor bonus: Reward PF up to 4.0
            pf_bonus = min(pf, 4.0) * 150.0

            score = pnl + pf_bonus - dd_pen - tr_pen
            return score

        s2 = compute_score(m2)
        s3 = compute_score(m3)

        # ─────────────────────────────────────────────────────────────────────
        # Harmonic Mean (penalizes imbalance)
        # ─────────────────────────────────────────────────────────────────────
        if s2 <= 0 or s3 <= 0:
            hm = min(s2, s3)  # If either negative, use worse one
        else:
            hm = 2.0 * s2 * s3 / (s2 + s3 + 1e-9)

        # ─────────────────────────────────────────────────────────────────────
        # Logging
        # ─────────────────────────────────────────────────────────────────────
        print(f"\n{'='*80}")
        print(f"Trial {trial.number} Results:")
        print(f"{'='*80}")
        print(f"Q2 2024: {m2['trades']} trades, ${m2['pnl']:.2f} PNL, "
              f"{m2['roi']:.1f}% ROI, {m2['drawdown']:.2f}% DD, PF={m2['profit_factor']:.2f}")
        print(f"Q3 2024: {m3['trades']} trades, ${m3['pnl']:.2f} PNL, "
              f"{m3['roi']:.1f}% ROI, {m3['drawdown']:.2f}% DD, PF={m3['profit_factor']:.2f}")
        print(f"Scores:  Q2={s2:.1f}, Q3={s3:.1f}, Harmonic Mean={hm:.1f}")
        print(f"{'='*80}\n")

        # Store metrics for later analysis
        trial.set_user_attr('q2_pnl', m2['pnl'])
        trial.set_user_attr('q2_trades', m2['trades'])
        trial.set_user_attr('q2_roi', m2['roi'])
        trial.set_user_attr('q2_dd', m2['drawdown'])
        trial.set_user_attr('q2_pf', m2['profit_factor'])

        trial.set_user_attr('q3_pnl', m3['pnl'])
        trial.set_user_attr('q3_trades', m3['trades'])
        trial.set_user_attr('q3_roi', m3['roi'])
        trial.set_user_attr('q3_dd', m3['drawdown'])
        trial.set_user_attr('q3_pf', m3['profit_factor'])

        trial.set_user_attr('harmonic_mean', hm)

        return hm

    finally:
        # Clean up temp config
        Path(cfg_path).unlink(missing_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# Main Entry Point
# ─────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="PR#6A Optuna Archetype Optimizer"
    )
    parser.add_argument(
        "--trials",
        type=int,
        default=80,
        help="Number of Optuna trials (default: 80)"
    )
    parser.add_argument(
        "--asset",
        type=str,
        default="BTC",
        help="Asset to optimize (default: BTC)"
    )
    parser.add_argument(
        "--base-config",
        type=str,
        default="configs/profile_archetype_optimized.json",
        help="Base config to start from"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="reports/archetype_optimization_v2",
        help="Output directory for results"
    )

    args = parser.parse_args()

    # ─────────────────────────────────────────────────────────────────────────
    # Load Base Config
    # ─────────────────────────────────────────────────────────────────────────
    print(f"\n{'='*80}")
    print(f"PR#6A Optuna Archetype Optimizer")
    print(f"{'='*80}")
    print(f"Asset:        {args.asset}")
    print(f"Base Config:  {args.base_config}")
    print(f"Trials:       {args.trials}")
    print(f"Output:       {args.output}")
    print(f"{'='*80}\n")

    with open(args.base_config, 'r') as f:
        base_cfg = json.load(f)

    # ─────────────────────────────────────────────────────────────────────────
    # Create Optuna Study
    # ─────────────────────────────────────────────────────────────────────────
    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=42),
        study_name=f"pr6a_archetype_{args.asset}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )

    # ─────────────────────────────────────────────────────────────────────────
    # Run Optimization
    # ─────────────────────────────────────────────────────────────────────────
    study.optimize(
        lambda trial: objective(trial, base_cfg, args.asset),
        n_trials=args.trials,
        show_progress_bar=True
    )

    # ─────────────────────────────────────────────────────────────────────────
    # Save Results
    # ─────────────────────────────────────────────────────────────────────────
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Best trial summary
    best = study.best_trial

    summary = {
        'best_trial_number': best.number,
        'best_harmonic_mean': best.value,
        'best_params': best.params,
        'q2_metrics': {
            'pnl': best.user_attrs.get('q2_pnl', 0),
            'trades': best.user_attrs.get('q2_trades', 0),
            'roi': best.user_attrs.get('q2_roi', 0),
            'drawdown': best.user_attrs.get('q2_dd', 0),
            'profit_factor': best.user_attrs.get('q2_pf', 0)
        },
        'q3_metrics': {
            'pnl': best.user_attrs.get('q3_pnl', 0),
            'trades': best.user_attrs.get('q3_trades', 0),
            'roi': best.user_attrs.get('q3_roi', 0),
            'drawdown': best.user_attrs.get('q3_dd', 0),
            'profit_factor': best.user_attrs.get('q3_pf', 0)
        },
        'timestamp': datetime.now().isoformat()
    }

    summary_path = output_dir / f"{args.asset}_best_trial_summary.json"
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)

    # Generate best config
    best_cfg = suggest_params(best, base_cfg)
    best_cfg['version'] = f"2.0.0-pr6a-optuna-v2-{datetime.now().strftime('%Y%m%d')}"
    best_cfg['profile'] = f"archetype_optuna_v2_{args.asset.lower()}"
    best_cfg['description'] = (
        f"Optuna-optimized PR#6A config (Trial #{best.number}). "
        f"Q2+Q3 harmonic mean: {best.value:.1f}. "
        f"Q2: {summary['q2_metrics']['trades']} trades, ${summary['q2_metrics']['pnl']:.0f} PNL. "
        f"Q3: {summary['q3_metrics']['trades']} trades, ${summary['q3_metrics']['pnl']:.0f} PNL."
    )

    best_cfg_path = output_dir / f"{args.asset}_best_config.json"
    with open(best_cfg_path, 'w') as f:
        json.dump(best_cfg, f, indent=2)

    # All trials CSV
    trials_data = []
    for t in study.trials:
        if t.state == optuna.trial.TrialState.COMPLETE:
            trials_data.append({
                'trial': t.number,
                'harmonic_mean': t.value,
                'q2_pnl': t.user_attrs.get('q2_pnl', 0),
                'q2_trades': t.user_attrs.get('q2_trades', 0),
                'q3_pnl': t.user_attrs.get('q3_pnl', 0),
                'q3_trades': t.user_attrs.get('q3_trades', 0),
                **t.params
            })

    import pandas as pd
    df = pd.DataFrame(trials_data)
    csv_path = output_dir / f"{args.asset}_all_trials.csv"
    df.to_csv(csv_path, index=False)

    # ─────────────────────────────────────────────────────────────────────────
    # Print Summary
    # ─────────────────────────────────────────────────────────────────────────
    print(f"\n{'='*80}")
    print(f"Optimization Complete!")
    print(f"{'='*80}")
    print(f"Best Trial:   #{best.number}")
    print(f"Harmonic Mean: {best.value:.1f}")
    print(f"\nQ2 2024 (Calibration):")
    print(f"  Trades: {summary['q2_metrics']['trades']}")
    print(f"  PNL:    ${summary['q2_metrics']['pnl']:.2f}")
    print(f"  ROI:    {summary['q2_metrics']['roi']:.1f}%")
    print(f"  DD:     {summary['q2_metrics']['drawdown']:.2f}%")
    print(f"  PF:     {summary['q2_metrics']['profit_factor']:.2f}")
    print(f"\nQ3 2024 (Validation):")
    print(f"  Trades: {summary['q3_metrics']['trades']}")
    print(f"  PNL:    ${summary['q3_metrics']['pnl']:.2f}")
    print(f"  ROI:    {summary['q3_metrics']['roi']:.1f}%")
    print(f"  DD:     {summary['q3_metrics']['drawdown']:.2f}%")
    print(f"  PF:     {summary['q3_metrics']['profit_factor']:.2f}")
    print(f"\nFiles saved:")
    print(f"  Summary:    {summary_path}")
    print(f"  Best Config: {best_cfg_path}")
    print(f"  All Trials:  {csv_path}")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()
