#!/usr/bin/env python3
"""
Knowledge-Aware Optimizer v3.0 - Full 69-Feature Engine

Uses Bayesian optimization (Optuna) to find optimal parameters for the
knowledge-aware backtest engine that leverages ALL 69 MTF features.

This replaces the simplified optimizer (v2) which only used ~10 features.

Search Space:
- Domain weights: Wyckoff, Liquidity, Momentum, Macro, PTI
- Entry thresholds: Tier 1, Tier 2, Tier 3
- Entry modifiers: M1/M2 confirmation, macro alignment, FRVP zone
- Exit parameters: ATR multipliers, max hold, partial exits
- Position sizing: Risk %, volatility scaling

Output: Top 10 configs → reports/optuna_results/
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
from datetime import datetime
import argparse
import json
import optuna
from optuna.samplers import TPESampler
from typing import Dict, List
import logging

# Import knowledge-aware backtest
from backtest_knowledge_v2 import KnowledgeParams, KnowledgeAwareBacktest

logging.basicConfig(level=logging.WARNING)  # Reduce backtest verbosity
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


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
    Optuna objective function for knowledge-aware backtest.

    Samples parameters from expanded search space and runs full backtest.
    """
    # Sample domain weights (must sum to ≤ 1.0)
    wyckoff_weight = trial.suggest_float('wyckoff_weight', 0.20, 0.40)
    liquidity_weight = trial.suggest_float('liquidity_weight', 0.20, 0.40)
    momentum_weight = trial.suggest_float('momentum_weight', 0.10, 0.25)
    macro_weight = trial.suggest_float('macro_weight', 0.05, 0.20)
    pti_weight = trial.suggest_float('pti_weight', 0.05, 0.20)

    # Check weight sum
    total_weight = wyckoff_weight + liquidity_weight + momentum_weight + macro_weight
    if total_weight > 0.95:  # Leave room for residual FRVP weight
        return -1e6  # Prune invalid trial

    # Sample entry thresholds
    tier1_threshold = trial.suggest_float('tier1_threshold', 0.40, 0.60)
    tier2_threshold = trial.suggest_float('tier2_threshold', 0.30, 0.50)
    tier3_threshold = trial.suggest_float('tier3_threshold', 0.20, 0.40)

    # Ensure tier ordering (tier1 > tier2 > tier3)
    if not (tier1_threshold > tier2_threshold > tier3_threshold):
        return -1e6  # Prune invalid trial

    # Sample entry modifiers
    require_m1m2 = trial.suggest_categorical('require_m1m2_confirmation', [True, False])
    require_macro = trial.suggest_categorical('require_macro_alignment', [True, False])

    # Sample exit parameters
    atr_stop_mult = trial.suggest_float('atr_stop_mult', 1.5, 3.5)
    trailing_atr_mult = trial.suggest_float('trailing_atr_mult', 1.5, 3.0)
    max_hold_bars = trial.suggest_int('max_hold_bars', 72, 336, step=24)  # 3-14 days in 24h increments

    # Sample position sizing
    max_risk_pct = trial.suggest_float('max_risk_pct', 0.01, 0.03)
    volatility_scaling = trial.suggest_categorical('volatility_scaling', [True, False])

    # Build params
    params = KnowledgeParams(
        wyckoff_weight=wyckoff_weight,
        liquidity_weight=liquidity_weight,
        momentum_weight=momentum_weight,
        macro_weight=macro_weight,
        pti_weight=pti_weight,
        tier1_threshold=tier1_threshold,
        tier2_threshold=tier2_threshold,
        tier3_threshold=tier3_threshold,
        require_m1m2_confirmation=require_m1m2,
        require_macro_alignment=require_macro,
        atr_stop_mult=atr_stop_mult,
        trailing_atr_mult=trailing_atr_mult,
        max_hold_bars=max_hold_bars,
        max_risk_pct=max_risk_pct,
        volatility_scaling=volatility_scaling,
        use_smart_exits=True,  # Always use smart exits
        breakeven_after_tp1=True  # Always move to breakeven after TP1
    )

    # Run backtest
    try:
        backtest = KnowledgeAwareBacktest(df, params, starting_capital=10000.0)
        results = backtest.run()
    except Exception as e:
        logger.warning(f"Trial {trial.number} failed: {e}")
        return -1e6  # Prune failed trial

    # Log metrics to trial
    trial.set_user_attr('total_pnl', results['total_pnl'])
    trial.set_user_attr('total_trades', results['total_trades'])
    trial.set_user_attr('profit_factor', results['profit_factor'])
    trial.set_user_attr('sharpe_ratio', results['sharpe_ratio'])
    trial.set_user_attr('max_drawdown', results['max_drawdown'])
    trial.set_user_attr('win_rate', results['win_rate'])

    # Objective: Maximize profit factor × sqrt(trade_count) / (1 + max_drawdown)
    # This balances profitability, trade frequency, and risk
    if results['total_trades'] == 0:
        return -1e6  # Prune configs that don't trade

    # Penalize low trade counts (need statistical significance)
    trade_penalty = np.sqrt(results['total_trades']) / np.sqrt(10)  # Normalize to 10 trades
    trade_penalty = min(trade_penalty, 1.0)  # Cap at 1.0

    # Penalize high drawdowns
    dd_penalty = 1.0 / (1.0 + results['max_drawdown'])

    # Reward high profit factor
    pf = max(results['profit_factor'], 0.1)  # Floor at 0.1

    # Composite score
    score = pf * trade_penalty * dd_penalty

    # Bonus for high Sharpe
    if results['sharpe_ratio'] > 1.0:
        score *= (1.0 + results['sharpe_ratio'] / 10.0)

    return score


def optimize_asset(asset: str, start_date: str, end_date: str, n_trials: int = 200) -> List[Dict]:
    """
    Run Bayesian optimization for knowledge-aware backtest.

    Args:
        asset: Asset symbol
        start_date: Start date
        end_date: End date
        n_trials: Number of optimization trials

    Returns:
        List of top 10 configs with metrics
    """
    print("=" * 80)
    print(f"Knowledge-Aware Optimization - {asset} ({start_date} → {end_date})")
    print("=" * 80)

    # Load feature store
    df = load_feature_store(asset, start_date, end_date)

    # Create Optuna study
    sampler = TPESampler(seed=42)
    study = optuna.create_study(
        direction='maximize',
        sampler=sampler,
        study_name=f"{asset}_knowledge_optimization"
    )

    # Run optimization
    print(f"\nRunning {n_trials} trials with full 69-feature knowledge engine...")
    print("This will take longer than v2 optimizer due to comprehensive feature usage.\n")

    study.optimize(
        lambda trial: objective(trial, df),
        n_trials=n_trials,
        show_progress_bar=True,
        n_jobs=1  # Sequential (knowledge backtest is already fast)
    )

    # Extract top 10 configs
    top_trials = sorted(study.trials, key=lambda t: t.value if t.value is not None else -1e9, reverse=True)[:10]

    results = []
    for i, trial in enumerate(top_trials):
        if trial.value is None or trial.value < 0:
            continue  # Skip failed trials

        config = {
            'rank': i + 1,
            'score': trial.value,
            'params': trial.params,
            'metrics': {
                'total_pnl': trial.user_attrs.get('total_pnl', 0.0),
                'total_trades': trial.user_attrs.get('total_trades', 0),
                'profit_factor': trial.user_attrs.get('profit_factor', 0.0),
                'sharpe_ratio': trial.user_attrs.get('sharpe_ratio', 0.0),
                'max_drawdown': trial.user_attrs.get('max_drawdown', 0.0),
                'win_rate': trial.user_attrs.get('win_rate', 0.0)
            }
        }
        results.append(config)

    # Print summary
    print("\n" + "=" * 80)
    print(f"Top 3 Knowledge-Aware Configurations for {asset}")
    print("=" * 80)

    for config in results[:3]:
        print(f"\nRank {config['rank']}: Score = {config['score']:.3f}")
        print(f"  Domain Weights:")
        print(f"    Wyckoff: {config['params']['wyckoff_weight']:.3f}")
        print(f"    Liquidity: {config['params']['liquidity_weight']:.3f}")
        print(f"    Momentum: {config['params']['momentum_weight']:.3f}")
        print(f"    Macro: {config['params']['macro_weight']:.3f}")
        print(f"    PTI: {config['params']['pti_weight']:.3f}")
        print(f"  Entry Thresholds:")
        print(f"    Tier 1: {config['params']['tier1_threshold']:.3f}")
        print(f"    Tier 2: {config['params']['tier2_threshold']:.3f}")
        print(f"    Tier 3: {config['params']['tier3_threshold']:.3f}")
        print(f"  Entry Modifiers:")
        print(f"    Require M1/M2: {config['params']['require_m1m2_confirmation']}")
        print(f"    Require Macro Alignment: {config['params']['require_macro_alignment']}")
        print(f"  Exit Parameters:")
        print(f"    ATR Stop Mult: {config['params']['atr_stop_mult']:.2f}")
        print(f"    Trailing ATR Mult: {config['params']['trailing_atr_mult']:.2f}")
        print(f"    Max Hold Bars: {config['params']['max_hold_bars']}")
        print(f"  Position Sizing:")
        print(f"    Max Risk %: {config['params']['max_risk_pct']:.3f}")
        print(f"    Volatility Scaling: {config['params']['volatility_scaling']}")
        print(f"  Metrics:")
        print(f"    PNL: ${config['metrics']['total_pnl']:.2f}")
        print(f"    Trades: {config['metrics']['total_trades']}")
        print(f"    Win Rate: {config['metrics']['win_rate']:.1%}")
        print(f"    Profit Factor: {config['metrics']['profit_factor']:.2f}")
        print(f"    Sharpe: {config['metrics']['sharpe_ratio']:.2f}")
        print(f"    Max DD: {config['metrics']['max_drawdown']:.1%}")

    return results


def main():
    parser = argparse.ArgumentParser(
        description='Knowledge-aware Bayesian optimization (full 69-feature engine)'
    )
    parser.add_argument('--asset', required=True, help='Asset to optimize (BTC, ETH, SPY, TSLA)')
    parser.add_argument('--start', default='2024-01-01', help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end', default='2024-12-31', help='End date (YYYY-MM-DD)')
    parser.add_argument('--trials', type=int, default=200, help='Number of optimization trials')

    args = parser.parse_args()

    # Run optimization
    results = optimize_asset(args.asset, args.start, args.end, args.trials)

    if not results:
        print("\n⚠️ No valid configurations found. Try adjusting search space or increasing trials.")
        return

    # Save results
    output_dir = Path('reports/optuna_results')
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / f'{args.asset}_knowledge_v3_best_configs.json'

    with open(output_path, 'w') as f:
        json.dump({
            'asset': args.asset,
            'period': f"{args.start} to {args.end}",
            'n_trials': args.trials,
            'timestamp': datetime.now().isoformat(),
            'optimizer_version': 'v3_full_knowledge',
            'features_used': 'ALL 69 MTF features',
            'top_10_configs': results
        }, f, indent=2)

    print(f"\n✅ Results saved to: {output_path}")
    print(f"\nComparison with v2 simplified baseline:")
    print(f"  v2 (10 features): Check reports/optuna_results/{args.asset}_best_configs.json")
    print(f"  v3 (69 features): {output_path}")


if __name__ == '__main__':
    main()
