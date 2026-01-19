#!/usr/bin/env python3
"""
SPY-Specific Strict Optimizer with Minimum Trade Requirements

Differences from optimize_v2_cached.py:
1. MINIMUM 20 TRADES REQUIREMENT (reject configs with < 20 trades)
2. REQUIRE POSITIVE PNL (reject losing configs)
3. Equity-tuned parameter ranges for low-volatility assets
4. Modified objective: (PNL / MaxDD) * sqrt(trades) if PNL > 0, else 0

This prevents overfitting to lucky 2-trade samples.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
import json
import optuna
from optuna.samplers import TPESampler
from typing import Dict

# Import the backtest_knowledge_v2 module for proper backtesting
from bin.backtest_knowledge_v2 import KnowledgeParams, KnowledgeAwareBacktest


def load_feature_store(asset: str, start_date: str, end_date: str) -> pd.DataFrame:
    """Load pre-built MTF feature store."""
    feature_dir = Path('data/features_mtf')
    pattern = f"{asset}_1H_*.parquet"
    files = list(feature_dir.glob(pattern))

    if not files:
        raise FileNotFoundError(f"No feature store found for {asset} in {feature_dir}")

    feature_path = sorted(files)[-1]
    print(f"Loading feature store: {feature_path}")

    df = pd.read_parquet(feature_path)

    # Filter to date range
    start_ts = pd.Timestamp(start_date, tz='UTC')
    end_ts = pd.Timestamp(end_date, tz='UTC')

    df = df[(df.index >= start_ts) & (df.index <= end_ts)].copy()

    print(f"Loaded {len(df)} bars from {df.index[0]} to {df.index[-1]}")

    return df


def objective(trial: optuna.Trial, df: pd.DataFrame) -> float:
    """
    Optuna objective function with STRICT constraints for SPY.

    Constraints:
    - Minimum 20 trades (statistical significance)
    - Positive PNL (no losing configs)
    - Equity-tuned parameter ranges

    Returns:
        Score (to maximize): (PNL / MaxDD) * sqrt(trades) if valid, else 0.0
    """
    # Sample parameters from EQUITY-TUNED search space
    # SPY is low volatility, so we need:
    # - Lower thresholds (more selective entry)
    # - Higher liquidity weight (institutional flow matters)
    # - Lower momentum weight (mean-reverting)
    # - Tighter stops (less room for drawdown)

    params = KnowledgeParams(
        # Weights (must sum to <= 1.0)
        wyckoff_weight=trial.suggest_float('wyckoff_weight', 0.20, 0.35),
        liquidity_weight=trial.suggest_float('liquidity_weight', 0.30, 0.50),  # Higher for SPY
        momentum_weight=trial.suggest_float('momentum_weight', 0.05, 0.15),   # Lower for SPY
        macro_weight=trial.suggest_float('macro_weight', 0.10, 0.25),
        pti_weight=trial.suggest_float('pti_weight', 0.05, 0.20),

        # Thresholds (SPY-specific: tighter for selectivity)
        tier1_threshold=trial.suggest_float('tier1_threshold', 0.65, 0.85),
        tier2_threshold=trial.suggest_float('tier2_threshold', 0.50, 0.70),
        tier3_threshold=trial.suggest_float('tier3_threshold', 0.35, 0.55),

        # Confirmations (stricter for SPY)
        require_m1m2_confirmation=trial.suggest_categorical('require_m1m2_confirmation', [True, False]),
        require_macro_alignment=trial.suggest_categorical('require_macro_alignment', [True, False]),

        # Risk management (tighter for low-vol equity)
        atr_stop_mult=trial.suggest_float('atr_stop_mult', 1.2, 2.5),  # Tighter stops
        trailing_atr_mult=trial.suggest_float('trailing_atr_mult', 1.0, 2.0),
        max_hold_bars=trial.suggest_int('max_hold_bars', 12, 72),  # 12h to 72h (3 days)
        max_risk_pct=trial.suggest_float('max_risk_pct', 0.01, 0.03),  # Lower risk per trade
        volatility_scaling=trial.suggest_categorical('volatility_scaling', [True, False]),

        # Smart exits (always enabled)
        use_smart_exits=True,
        breakeven_after_tp1=True
    )

    # Run backtest using actual backtest_knowledge_v2 engine
    try:
        backtest = KnowledgeAwareBacktest(df, params, starting_capital=10000.0)
        results = backtest.run()
    except Exception as e:
        print(f"   Backtest failed: {e}")
        return 0.0

    # Extract metrics
    total_pnl = results['total_pnl']
    total_trades = results['total_trades']
    profit_factor = results['profit_factor']
    max_drawdown = results['max_drawdown']

    # Log trial metrics
    trial.set_user_attr('total_pnl', total_pnl)
    trial.set_user_attr('total_trades', total_trades)
    trial.set_user_attr('profit_factor', profit_factor)
    trial.set_user_attr('sharpe_ratio', results['sharpe_ratio'])
    trial.set_user_attr('max_drawdown', max_drawdown)
    trial.set_user_attr('win_rate', results['win_rate'])

    # STRICT CONSTRAINTS
    if total_trades < 20:
        # Reject: Not enough trades for statistical significance
        print(f"   REJECTED: Only {total_trades} trades (need ≥20)")
        return 0.0

    if total_pnl <= 0:
        # Reject: Losing config
        print(f"   REJECTED: Negative PNL ${total_pnl:.2f}")
        return 0.0

    if max_drawdown >= 0.10:
        # Reject: Excessive drawdown (>10%)
        print(f"   REJECTED: MaxDD {max_drawdown*100:.1f}% (limit 10%)")
        return 0.0

    # OBJECTIVE: Risk-adjusted return with trade count bonus
    # Score = (PNL / MaxDD) * sqrt(trades)
    # This rewards:
    # - High PNL
    # - Low MaxDD
    # - More trades (statistical robustness)

    if max_drawdown > 0:
        risk_adj_return = total_pnl / max_drawdown
    else:
        risk_adj_return = total_pnl * 1000  # No drawdown = very good

    score = risk_adj_return * np.sqrt(total_trades)

    print(f"   ✅ VALID: {total_trades} trades, PNL=${total_pnl:.2f}, DD={max_drawdown*100:.1f}%, Score={score:.2f}")

    return score


def optimize_spy_strict(start_date: str, end_date: str, n_trials: int = 300) -> Dict:
    """
    Run strict Bayesian optimization for SPY.

    Args:
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)
        n_trials: Number of optimization trials (default 300 for thorough search)

    Returns:
        Dict with top 10 configs and metadata
    """
    print("=" * 80)
    print(f"SPY STRICT OPTIMIZER (Min 20 Trades, Positive PNL Required)")
    print(f"Date Range: {start_date} → {end_date}")
    print(f"Trials: {n_trials}")
    print("=" * 80)

    # Load feature store
    df = load_feature_store('SPY', start_date, end_date)

    # Create Optuna study
    sampler = TPESampler(seed=42)
    study = optuna.create_study(
        direction='maximize',
        sampler=sampler,
        study_name="SPY_strict_optimization"
    )

    # Run optimization
    print(f"\nRunning {n_trials} trials with strict constraints...")
    print("Constraints:")
    print("  - Minimum 20 trades")
    print("  - Positive PNL required")
    print("  - Maximum 10% drawdown")
    print("  - Equity-tuned parameter ranges")
    print()

    study.optimize(lambda trial: objective(trial, df), n_trials=n_trials, show_progress_bar=True)

    # Extract top 10 VALID configs (score > 0)
    valid_trials = [t for t in study.trials if t.value > 0]
    top_trials = sorted(valid_trials, key=lambda t: t.value, reverse=True)[:10]

    print(f"\n{'=' * 80}")
    print(f"OPTIMIZATION COMPLETE")
    print(f"{'=' * 80}")
    print(f"Total trials: {len(study.trials)}")
    print(f"Valid configs (passed constraints): {len(valid_trials)}")
    print(f"Rejected configs: {len(study.trials) - len(valid_trials)}")

    if not top_trials:
        print("\n⚠️  NO VALID CONFIGS FOUND!")
        print("Recommendation:")
        print("  - Loosen constraints (reduce min trades to 15?)")
        print("  - Expand parameter ranges")
        print("  - Check if SPY data has sufficient signals")
        return {
            'asset': 'SPY',
            'start_date': start_date,
            'end_date': end_date,
            'n_trials': n_trials,
            'valid_configs': 0,
            'top_10_configs': []
        }

    # Build results
    results = []
    for rank, trial in enumerate(top_trials, 1):
        config = {
            'rank': rank,
            'score': trial.value,
            'params': trial.params,
            'metrics': {
                'total_pnl': trial.user_attrs['total_pnl'],
                'total_trades': trial.user_attrs['total_trades'],
                'profit_factor': trial.user_attrs['profit_factor'],
                'sharpe_ratio': trial.user_attrs['sharpe_ratio'],
                'max_drawdown': trial.user_attrs['max_drawdown'],
                'win_rate': trial.user_attrs['win_rate']
            }
        }
        results.append(config)

    # Display top 10
    print(f"\nTOP 10 CONFIGS:")
    print(f"{'Rank':<6} {'Score':>10} {'PNL':>12} {'Trades':>8} {'PF':>8} {'WinRate':>8} {'MaxDD':>8}")
    print("-" * 80)

    for cfg in results:
        m = cfg['metrics']
        print(f"#{cfg['rank']:<5} {cfg['score']:>10.2f} ${m['total_pnl']:>10,.2f} "
              f"{m['total_trades']:>8} {m['profit_factor']:>8.2f} "
              f"{m['win_rate']*100:>7.1f}% {m['max_drawdown']*100:>7.2f}%")

    output = {
        'asset': 'SPY',
        'start_date': start_date,
        'end_date': end_date,
        'n_trials': n_trials,
        'valid_configs': len(valid_trials),
        'top_10_configs': results
    }

    # Save results
    output_dir = Path('reports/optuna_results')
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / 'SPY_knowledge_v3_strict_best_configs.json'

    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)

    print(f"\n✅ Results saved to: {output_path}")
    print("=" * 80)

    return output


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='SPY Strict Optimizer')
    parser.add_argument('--start', type=str, default='2024-01-01',
                       help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end', type=str, default='2024-12-31',
                       help='End date (YYYY-MM-DD)')
    parser.add_argument('--trials', type=int, default=300,
                       help='Number of optimization trials')

    args = parser.parse_args()

    optimize_spy_strict(args.start, args.end, args.trials)
