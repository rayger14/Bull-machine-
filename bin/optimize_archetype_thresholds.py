#!/usr/bin/env python3
"""
Optimize archetype thresholds to balance trade quality vs quantity.

Goal: Find optimal settings that maximize PNL while maintaining reasonable trade frequency.
"""

import json
import subprocess
import sys
from pathlib import Path
import optuna
from optuna.samplers import TPESampler

# Base configuration
BASE_CONFIG = "configs/profile_experimental.json"
OUTPUT_DIR = Path("reports/archetype_optimization")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def load_base_config():
    """Load the base experimental config."""
    with open(BASE_CONFIG) as f:
        return json.load(f)

def run_backtest(config_path: str) -> tuple[int, float, float]:
    """
    Run backtest and extract key metrics.
    Returns: (num_trades, final_equity, pnl_per_trade)
    """
    cmd = [
        "python3", "bin/backtest_knowledge_v2.py",
        "--asset", "BTC",
        "--start", "2024-01-01",
        "--end", "2024-12-31",
        "--config", config_path
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    
    # Parse output for metrics
    num_trades = 0
    final_equity = 10000.0
    
    for line in result.stdout.split('\n'):
        if 'equity=' in line:
            # Extract final equity from last equity line
            parts = line.split('equity=')
            if len(parts) > 1:
                equity_str = parts[1].split()[0].replace(',', '')
                try:
                    final_equity = float(equity_str)
                except:
                    pass
        elif line.startswith('Trade '):
            num_trades += 1
    
    pnl = final_equity - 10000.0
    pnl_per_trade = pnl / max(num_trades, 1)
    
    return num_trades, final_equity, pnl_per_trade

def objective(trial):
    """
    Optuna objective function.
    
    Optimize for: PNL with penalty for too many/few trades
    """
    
    # Load base config
    config = load_base_config()
    
    # Parameters to optimize
    min_liquidity = trial.suggest_float("min_liquidity", 0.05, 0.30, step=0.05)
    
    # trap_within_trend thresholds (H archetype - currently 81% of trades)
    h_fusion = trial.suggest_float("H_fusion", 0.35, 0.55, step=0.05)
    h_adx = trial.suggest_int("H_adx", 20, 35, step=5)
    
    # failed_continuation thresholds (C archetype - 16% of trades)
    c_fusion = trial.suggest_float("C_fusion", 0.38, 0.52, step=0.02)
    c_disp_atr = trial.suggest_float("C_disp_atr", 0.80, 1.50, step=0.10)
    
    # wick_trap thresholds (currently 1.6%)
    wick_fusion = trial.suggest_float("wick_fusion", 0.36, 0.50, step=0.02)
    
    # Update config
    config["archetypes"]["thresholds"]["min_liquidity"] = min_liquidity
    config["archetypes"]["thresholds"]["H"]["fusion"] = h_fusion
    config["archetypes"]["thresholds"]["H"]["adx"] = h_adx
    config["archetypes"]["thresholds"]["C"]["fusion"] = c_fusion
    config["archetypes"]["thresholds"]["C"]["disp_atr"] = c_disp_atr
    
    # Assuming wick_trap maps to one of the existing archetypes
    # (adjust based on actual archetype mapping)
    
    # Save trial config
    trial_config_path = OUTPUT_DIR / f"trial_{trial.number:04d}_config.json"
    with open(trial_config_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    # Run backtest
    try:
        num_trades, final_equity, pnl_per_trade = run_backtest(str(trial_config_path))
        pnl = final_equity - 10000.0
        
        # Log metrics
        trial.set_user_attr("num_trades", num_trades)
        trial.set_user_attr("final_equity", final_equity)
        trial.set_user_attr("pnl", pnl)
        trial.set_user_attr("pnl_per_trade", pnl_per_trade)
        
        # Objective: Maximize PNL with trade count constraints
        # Penalize if trades < 30 or > 70
        trade_penalty = 0
        if num_trades < 30:
            trade_penalty = (30 - num_trades) * 20  # Penalty for too few trades
        elif num_trades > 70:
            trade_penalty = (num_trades - 70) * 10  # Penalty for too many trades
        
        score = pnl - trade_penalty
        
        print(f"Trial {trial.number}: trades={num_trades}, PNL=${pnl:.2f}, score={score:.2f}")
        
        return score
        
    except Exception as e:
        print(f"Trial {trial.number} failed: {e}")
        return -10000  # Large penalty for failed trials

def main():
    """Run optimization."""
    print("=" * 80)
    print("Archetype Threshold Optimization")
    print("=" * 80)
    print(f"Target: Maximize PNL with 30-70 trades")
    print(f"Baseline: 36 trades @ $2042 PNL (legacy)")
    print(f"Current:  92 trades @ $1478 PNL (untuned archetypes)")
    print("=" * 80)
    
    # Create study
    study = optuna.create_study(
        direction="maximize",
        sampler=TPESampler(seed=42)
    )
    
    # Run optimization
    n_trials = int(sys.argv[1]) if len(sys.argv) > 1 else 50
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    
    # Report results
    print("\n" + "=" * 80)
    print("Optimization Complete")
    print("=" * 80)
    
    best_trial = study.best_trial
    print(f"\nBest Trial: #{best_trial.number}")
    print(f"  Score: {best_trial.value:.2f}")
    print(f"  Trades: {best_trial.user_attrs['num_trades']}")
    print(f"  Final Equity: ${best_trial.user_attrs['final_equity']:.2f}")
    print(f"  PNL: ${best_trial.user_attrs['pnl']:.2f}")
    print(f"  PNL/Trade: ${best_trial.user_attrs['pnl_per_trade']:.2f}")
    
    print("\nBest Parameters:")
    for key, value in best_trial.params.items():
        print(f"  {key}: {value}")
    
    # Save best config
    best_config = load_base_config()
    best_config["archetypes"]["thresholds"]["min_liquidity"] = best_trial.params["min_liquidity"]
    best_config["archetypes"]["thresholds"]["H"]["fusion"] = best_trial.params["H_fusion"]
    best_config["archetypes"]["thresholds"]["H"]["adx"] = best_trial.params["H_adx"]
    best_config["archetypes"]["thresholds"]["C"]["fusion"] = best_trial.params["C_fusion"]
    best_config["archetypes"]["thresholds"]["C"]["disp_atr"] = best_trial.params["C_disp_atr"]
    
    best_config_path = OUTPUT_DIR / "best_archetype_config.json"
    with open(best_config_path, 'w') as f:
        json.dump(best_config, f, indent=2)
    
    print(f"\nBest config saved to: {best_config_path}")
    
    # Save study results
    study_df = study.trials_dataframe()
    study_df.to_csv(OUTPUT_DIR / "optimization_results.csv", index=False)
    print(f"Full results saved to: {OUTPUT_DIR / 'optimization_results.csv'}")

if __name__ == "__main__":
    main()
