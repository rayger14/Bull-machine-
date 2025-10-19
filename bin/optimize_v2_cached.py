#!/usr/bin/env python3
"""
Bayesian Optimizer v2.0 - Cached Feature Store Search

Uses Optuna to find optimal fusion weights, thresholds, and exit parameters
by loading pre-built feature stores and running vectorized backtests.

Objective: Maximize Profit_Factor × sqrt(Trade_Count)

Search Space:
- wyckoff_weight: [0.25, 0.45]
- liquidity_weight (HOB): [0.25, 0.45]
- momentum_weight: [0.1, 0.25]
- threshold: [0.55, 0.75]
- fakeout_penalty: [0.05, 0.25]
- exit_aggressiveness: [0.4, 0.8]

Output: Top 10 configs per asset → reports/optuna_results/
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
from typing import Dict, List, Tuple

# Placeholder for fast backtest (will implement in Phase 3)
# For now, we'll use a simplified scoring function


def load_feature_store(asset: str, start_date: str, end_date: str) -> pd.DataFrame:
    """
    Load pre-built MTF feature store.

    Args:
        asset: Asset symbol (BTC, ETH, SPY, TSLA)
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)

    Returns:
        DataFrame with 69 MTF features at 1H resolution
    """
    # Find the feature store file
    feature_dir = Path('data/features_mtf')

    # Look for files matching the asset
    pattern = f"{asset}_1H_*.parquet"
    files = list(feature_dir.glob(pattern))

    if not files:
        raise FileNotFoundError(f"No feature store found for {asset} in {feature_dir}")

    # Load the most recent file
    feature_path = sorted(files)[-1]
    print(f"Loading feature store: {feature_path}")

    df = pd.read_parquet(feature_path)

    # Filter to date range
    start_ts = pd.Timestamp(start_date, tz='UTC')
    end_ts = pd.Timestamp(end_date, tz='UTC')

    df = df[(df.index >= start_ts) & (df.index <= end_ts)].copy()

    print(f"Loaded {len(df)} bars from {df.index[0]} to {df.index[-1]}")

    return df


def compute_fusion_score(row: pd.Series, params: Dict) -> float:
    """
    Compute fusion score using weighted domain scores from MTF features.

    Args:
        row: Feature row with MTF domain scores
        params: Parameter dict with weights

    Returns:
        Fusion score [0.0, 1.0]
    """
    # Extract domain scores from MTF schema (normalize to [0, 1])
    # Wyckoff: tf1d_wyckoff_score is already [0, 1]
    wyckoff = row.get('tf1d_wyckoff_score', 0.5)

    # SMC (Structure): Use 4H structure alignment + squiggle confidence
    smc_structure = row.get('tf4h_structure_alignment', 0.5)
    smc_confidence = row.get('tf4h_squiggle_confidence', 0.5)
    smc = (smc_structure + smc_confidence) / 2.0

    # HOB (Hidden Order Blocks / Liquidity): Use BOMS strength + FVG presence
    hob_boms = row.get('tf1d_boms_strength', 0.5)
    hob_fvg = 1.0 if row.get('tf4h_fvg_present', False) else 0.0
    hob = (hob_boms + hob_fvg) / 2.0

    # Momentum: Use ADX + RSI positioning
    adx = row.get('adx_14', 20.0) / 100.0  # Normalize ADX to [0, 1]
    rsi = row.get('rsi_14', 50.0)
    rsi_momentum = abs(rsi - 50.0) / 50.0  # Distance from 50 = momentum strength
    momentum = (adx + rsi_momentum) / 2.0

    # Weighted combination
    fusion = (
        params['wyckoff_weight'] * wyckoff +
        params['liquidity_weight'] * hob +  # HOB represents liquidity
        params['momentum_weight'] * momentum +
        (1.0 - params['wyckoff_weight'] - params['liquidity_weight'] - params['momentum_weight']) * smc
    )

    # Apply fakeout penalty if detected
    if row.get('tf1h_fakeout_detected', False):
        fusion -= params['fakeout_penalty']

    # Macro governor veto
    if row.get('mtf_governor_veto', False):
        fusion *= 0.5  # Halve score if governor vetoes

    return np.clip(fusion, 0.0, 1.0)


def simulate_backtest(df: pd.DataFrame, params: Dict) -> Dict:
    """
    Simplified backtest simulation using cached features.

    In Phase 3, this will be replaced with fast_backtest_v2.py (vectorized).
    For now, we'll use a heuristic scoring function.

    Args:
        df: Feature store DataFrame
        params: Parameter dict

    Returns:
        Dict with metrics (PNL, trades, profit_factor, sharpe, max_dd)
    """
    # Compute fusion scores
    df['fusion_score'] = df.apply(lambda row: compute_fusion_score(row, params), axis=1)

    # Fill NaN values (early bars without ADX/indicators) with 0.0
    df['fusion_score'] = df['fusion_score'].fillna(0.0)

    # DEBUG: Log fusion score distribution
    if len(df) > 0:
        print(f"   Fusion scores: min={df['fusion_score'].min():.4f}, max={df['fusion_score'].max():.4f}, mean={df['fusion_score'].mean():.4f}")

    # Generate signals (threshold-based)
    df['signal'] = 0
    df.loc[df['fusion_score'] > params['threshold'], 'signal'] = 1  # Long
    # Disable shorts for now (fusion scores don't go high enough for inverted logic)
    # df.loc[df['fusion_score'] < (1.0 - params['threshold']), 'signal'] = -1  # Short

    # DEBUG: Log signal distribution
    long_signals = (df['signal'] == 1).sum()
    print(f"   Long signals: {long_signals} bars (threshold={params['threshold']:.4f})")

    # Track positions
    position = 0
    entry_price = 0.0
    trades = []
    equity = 10000.0
    peak = equity
    max_dd = 0.0

    entries_attempted = 0
    signals_seen = 0

    for idx, row in df.iterrows():
        current_price = row['close']
        signal = row['signal']

        if signal != 0:
            signals_seen += 1

        # Entry logic
        if position == 0 and signal != 0:
            entries_attempted += 1
            position = signal
            entry_price = current_price

        # Exit logic (simplified)
        elif position != 0:
            # Exit on opposite signal, signal returns to 0, MTF conflict, or at end of period
            should_exit = (
                signal == -position or
                signal == 0 or  # Exit when signal neutralizes
                row.get('mtf_conflict_score', 0.0) > 0.7 or
                (params['exit_aggressiveness'] > 0.6 and row.get('tf1h_pti_reversal_likely', False))
            )

            if should_exit:
                # Close trade
                pnl_pct = (current_price - entry_price) / entry_price * position
                pnl_dollars = equity * pnl_pct * 0.95  # 95% allocation

                # Apply costs (3bp total: 2bp slippage + 1bp fees)
                pnl_dollars -= abs(pnl_dollars) * 0.0003

                equity += pnl_dollars

                trades.append({
                    'entry': entry_price,
                    'exit': current_price,
                    'pnl': pnl_dollars,
                    'position': position
                })

                # Update drawdown
                if equity > peak:
                    peak = equity
                dd = (peak - equity) / peak
                if dd > max_dd:
                    max_dd = dd

                position = 0

    # DEBUG: Log entry attempts
    print(f"   Signals seen: {signals_seen}, Entries attempted: {entries_attempted}, Trades closed: {len(trades)}")

    # Calculate metrics
    if not trades:
        return {
            'total_pnl': 0.0,
            'total_trades': 0,
            'profit_factor': 0.0,
            'sharpe_ratio': 0.0,
            'max_drawdown': 0.0,
            'final_equity': equity,
            'score': 0.0
        }

    total_pnl = sum(t['pnl'] for t in trades)
    winning_trades = [t for t in trades if t['pnl'] > 0]
    losing_trades = [t for t in trades if t['pnl'] < 0]

    gross_profit = sum(t['pnl'] for t in winning_trades) if winning_trades else 0.0
    gross_loss = abs(sum(t['pnl'] for t in losing_trades)) if losing_trades else 1.0

    profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0.0

    # Sharpe (simplified: returns std)
    trade_returns = [t['pnl'] / 10000 for t in trades]
    sharpe = np.mean(trade_returns) / np.std(trade_returns) if len(trade_returns) > 1 else 0.0
    sharpe = sharpe * np.sqrt(252 / len(trades)) if len(trades) > 0 else 0.0  # Annualized

    # Objective: PF × sqrt(Trade_Count)
    score = profit_factor * np.sqrt(len(trades))

    return {
        'total_pnl': total_pnl,
        'total_trades': len(trades),
        'profit_factor': profit_factor,
        'sharpe_ratio': sharpe,
        'max_drawdown': max_dd,
        'final_equity': equity,
        'score': score
    }


def objective(trial: optuna.Trial, df: pd.DataFrame) -> float:
    """
    Optuna objective function.

    Args:
        trial: Optuna trial object
        df: Feature store DataFrame

    Returns:
        Objective score (to maximize)
    """
    # Sample parameters from search space
    # ADJUSTED: Threshold range lowered from [0.55, 0.75] → [0.20, 0.50]
    # to match actual fusion score distribution (max observed ~0.31)
    params = {
        'wyckoff_weight': trial.suggest_float('wyckoff_weight', 0.25, 0.45),
        'liquidity_weight': trial.suggest_float('liquidity_weight', 0.25, 0.45),
        'momentum_weight': trial.suggest_float('momentum_weight', 0.1, 0.25),
        'threshold': trial.suggest_float('threshold', 0.20, 0.50),
        'fakeout_penalty': trial.suggest_float('fakeout_penalty', 0.05, 0.25),
        'exit_aggressiveness': trial.suggest_float('exit_aggressiveness', 0.4, 0.8),
    }

    # Normalize weights to sum to 1.0
    total_weight = params['wyckoff_weight'] + params['liquidity_weight'] + params['momentum_weight']
    smc_weight = max(0.0, 1.0 - total_weight)

    if total_weight > 1.0:
        # Prune invalid trials
        return 0.0

    # Run backtest
    results = simulate_backtest(df, params)

    # Log metrics
    trial.set_user_attr('total_pnl', results['total_pnl'])
    trial.set_user_attr('total_trades', results['total_trades'])
    trial.set_user_attr('profit_factor', results['profit_factor'])
    trial.set_user_attr('sharpe_ratio', results['sharpe_ratio'])
    trial.set_user_attr('max_drawdown', results['max_drawdown'])

    return results['score']


def optimize_asset(asset: str, start_date: str, end_date: str, n_trials: int = 200) -> List[Dict]:
    """
    Run Bayesian optimization for a single asset.

    Args:
        asset: Asset symbol
        start_date: Start date
        end_date: End date
        n_trials: Number of optimization trials

    Returns:
        List of top 10 configs with metrics
    """
    print("=" * 80)
    print(f"Optimizing {asset} ({start_date} → {end_date})")
    print("=" * 80)

    # Load feature store
    df = load_feature_store(asset, start_date, end_date)

    # Create Optuna study
    sampler = TPESampler(seed=42)
    study = optuna.create_study(
        direction='maximize',
        sampler=sampler,
        study_name=f"{asset}_optimization"
    )

    # Run optimization
    print(f"\nRunning {n_trials} trials...")
    study.optimize(lambda trial: objective(trial, df), n_trials=n_trials, show_progress_bar=True)

    # Extract top 10 configs
    top_trials = sorted(study.trials, key=lambda t: t.value, reverse=True)[:10]

    results = []
    for i, trial in enumerate(top_trials):
        config = {
            'rank': i + 1,
            'score': trial.value,
            'params': trial.params,
            'metrics': {
                'total_pnl': trial.user_attrs.get('total_pnl', 0.0),
                'total_trades': trial.user_attrs.get('total_trades', 0),
                'profit_factor': trial.user_attrs.get('profit_factor', 0.0),
                'sharpe_ratio': trial.user_attrs.get('sharpe_ratio', 0.0),
                'max_drawdown': trial.user_attrs.get('max_drawdown', 0.0)
            }
        }
        results.append(config)

    # Print summary
    print("\n" + "=" * 80)
    print(f"Top 3 Configurations for {asset}")
    print("=" * 80)
    for config in results[:3]:
        print(f"\nRank {config['rank']}: Score = {config['score']:.3f}")
        print(f"  Params: {json.dumps(config['params'], indent=4)}")
        print(f"  PNL: ${config['metrics']['total_pnl']:.2f}")
        print(f"  Trades: {config['metrics']['total_trades']}")
        print(f"  Profit Factor: {config['metrics']['profit_factor']:.2f}")
        print(f"  Sharpe: {config['metrics']['sharpe_ratio']:.2f}")
        print(f"  Max DD: {config['metrics']['max_drawdown']:.1%}")

    return results


def main():
    parser = argparse.ArgumentParser(
        description='Bayesian optimization using cached feature stores'
    )
    parser.add_argument('--asset', required=True, help='Asset to optimize (BTC, ETH, SPY, TSLA)')
    parser.add_argument('--start', default='2024-01-01', help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end', default='2025-10-17', help='End date (YYYY-MM-DD)')
    parser.add_argument('--trials', type=int, default=200, help='Number of optimization trials')

    args = parser.parse_args()

    # Run optimization
    results = optimize_asset(args.asset, args.start, args.end, args.trials)

    # Save results
    output_dir = Path('reports/optuna_results')
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / f'{args.asset}_best_configs.json'

    with open(output_path, 'w') as f:
        json.dump({
            'asset': args.asset,
            'period': f"{args.start} to {args.end}",
            'n_trials': args.trials,
            'timestamp': datetime.now().isoformat(),
            'top_10_configs': results
        }, f, indent=2)

    print(f"\n✅ Results saved to: {output_path}")


if __name__ == '__main__':
    main()
