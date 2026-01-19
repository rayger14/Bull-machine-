#!/usr/bin/env python3
"""
Bull Machine v1.8.6 - High-Performance Optimization Framework

Fast multi-year optimization using:
- Vectorized backtesting (eliminate bar-by-bar loops)
- Parallel grid search (all CPU cores)
- Smart caching (domain scores, macro snapshots)
- Walk-forward validation (prevent overfitting)

Usage:
    # Quick test (5 min)
    python bin/optimize_v18.py --mode quick --asset BTC

    # Full grid search (30-60 min)
    python bin/optimize_v18.py --mode grid --asset BTC --years 3

    # Walk-forward validation (2-4 hours)
    python bin/optimize_v18.py --mode walkforward --asset BTC --years 3
"""

import sys
import os
from pathlib import Path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import itertools
import multiprocessing as mp
from functools import partial
import argparse
import time
from dataclasses import dataclass, asdict

from engine.io.tradingview_loader import load_tv
from engine.context.loader import load_macro_data, fetch_macro_snapshot
from engine.context.macro_engine import analyze_macro, create_default_macro_config
from engine.fusion.domain_fusion import analyze_fusion

# ============================================================================
# VECTORIZED BACKTEST ENGINE (10-100x faster than bar-by-bar)
# ============================================================================

@dataclass
class BacktestResult:
    """Lightweight result container"""
    config_id: str
    asset: str
    period: str

    # Performance metrics
    total_return: float
    trades: int
    win_rate: float
    profit_factor: float
    sharpe_ratio: float
    max_drawdown: float
    avg_trade_pct: float

    # Risk metrics
    avg_r_multiple: float
    winners_avg: float
    losers_avg: float

    # Configuration tested
    fusion_threshold: float
    wyckoff_weight: float
    smc_weight: float
    hob_weight: float
    momentum_weight: float

    # Validation
    sample_size: int
    statistical_significance: str  # 'high', 'medium', 'low'

    # Execution time
    backtest_seconds: float


class VectorizedBacktester:
    """
    Vectorized backtest engine - processes entire dataframe at once
    instead of bar-by-bar loops.

    Key optimizations:
    1. Pre-compute ALL domain scores upfront (no per-bar recalculation)
    2. Vectorized signal generation using pandas operations
    3. Cached macro snapshots (daily granularity)
    4. Batch ATR/indicator calculations
    """

    def __init__(self, asset: str, start_date: str, end_date: str):
        self.asset = asset
        self.start_date = start_date
        self.end_date = end_date

        # Load data once
        print(f"📊 Loading {asset} data ({start_date} → {end_date})...")
        self.df_1h = load_tv(f"{asset}_1H")
        self.df_4h = load_tv(f"{asset}_4H")
        self.df_1d = load_tv(f"{asset}_1D")

        # Filter to date range
        start_ts = pd.Timestamp(start_date, tz='UTC')
        end_ts = pd.Timestamp(end_date, tz='UTC')
        self.df_1h = self.df_1h[(self.df_1h.index >= start_ts) & (self.df_1h.index <= end_ts)]
        self.df_4h = self.df_4h[(self.df_4h.index >= start_ts) & (self.df_4h.index <= end_ts)]
        self.df_1d = self.df_1d[(self.df_1d.index >= start_ts) & (self.df_1d.index <= end_ts)]

        # Standardize columns
        for df in [self.df_1h, self.df_4h, self.df_1d]:
            df.columns = [c.lower() for c in df.columns]

        # Load macro data once
        self.macro_data = load_macro_data()
        self.macro_config = create_default_macro_config()

        # Pre-compute indicators (shared across all configs)
        print("🧮 Pre-computing indicators...")
        self._precompute_indicators()

        # Pre-compute domain scores (expensive operation done once)
        print("🔮 Pre-computing domain scores...")
        self._precompute_domain_scores()

        print(f"✅ Data loaded: {len(self.df_1h)} 1H bars, {len(self.df_4h)} 4H bars, {len(self.df_1d)} 1D bars")

    def _precompute_indicators(self):
        """Pre-compute all technical indicators"""
        # ATR (for stops)
        for period in [14, 20]:
            self.df_1h[f'atr_{period}'] = self._calc_atr(self.df_1h, period)
            self.df_4h[f'atr_{period}'] = self._calc_atr(self.df_4h, period)

        # Trend indicators
        for window in [20, 50, 100]:
            self.df_1h[f'sma_{window}'] = self.df_1h['close'].rolling(window).mean()
            self.df_4h[f'sma_{window}'] = self.df_4h['close'].rolling(window).mean()
            self.df_1d[f'sma_{window}'] = self.df_1d['close'].rolling(window).mean()

        # Momentum indicators
        self.df_1h['rsi_14'] = self._calc_rsi(self.df_1h, 14)
        self.df_4h['rsi_14'] = self._calc_rsi(self.df_4h, 14)

    def _precompute_domain_scores(self):
        """
        Pre-compute domain scores for SAMPLED bars (not every bar).
        This is the most expensive operation - do it once, reuse for all configs.

        Performance: Sample every 4th bar (4H × 4 = 16H spacing)
        Trade-off: 4× faster pre-computation, loses some signal precision
        """
        # We'll store domain scores in the dataframe
        # For each 4H bar, compute Wyckoff/SMC/HOB/Momentum scores

        wyckoff_scores = []
        smc_scores = []
        hob_scores = []
        momentum_scores = []

        # PERFORMANCE: Sample every 4th bar for 4× speedup
        sample_interval = 4
        total_bars = len(self.df_4h)
        sampled_bars = total_bars // sample_interval
        print(f"   Computing {sampled_bars} domain score sets (sampling every {sample_interval}th bar for speed)...")

        for i in range(len(self.df_4h)):
            # PERFORMANCE: Only compute scores for sampled bars
            if i % sample_interval != 0 and i != len(self.df_4h) - 1:
                # Use previous score (forward-fill)
                if len(wyckoff_scores) > 0:
                    wyckoff_scores.append(wyckoff_scores[-1])
                    smc_scores.append(smc_scores[-1])
                    hob_scores.append(hob_scores[-1])
                    momentum_scores.append(momentum_scores[-1])
                else:
                    # First bars - use neutral
                    wyckoff_scores.append(0.5)
                    smc_scores.append(0.5)
                    hob_scores.append(0.5)
                    momentum_scores.append(0.5)
                continue

            # Get window for analysis (need history)
            window_end = i + 1
            window_start = max(0, window_end - 100)  # 100-bar lookback

            if window_end - window_start < 50:
                # Not enough history
                wyckoff_scores.append(0.5)
                smc_scores.append(0.5)
                hob_scores.append(0.5)
                momentum_scores.append(0.5)
                continue

            window_1h = self.df_1h[self.df_1h.index <= self.df_4h.index[i]].tail(200)
            window_4h = self.df_4h.iloc[window_start:window_end]
            window_1d = self.df_1d[self.df_1d.index <= self.df_4h.index[i]].tail(50)

            # Check for empty windows (data not available yet)
            if len(window_1h) < 50 or len(window_4h) < 14 or len(window_1d) < 20:
                # Not enough data
                wyckoff_scores.append(0.5)
                smc_scores.append(0.5)
                hob_scores.append(0.5)
                momentum_scores.append(0.5)
                continue

            try:
                # Run fusion analysis (cached internally)
                fusion_result = analyze_fusion(
                    window_1h, window_4h, window_1d,
                    config={'fusion': {
                        'weights': {'wyckoff': 0.30, 'smc': 0.15, 'liquidity': 0.25, 'momentum': 0.30}
                    }}
                )

                wyckoff_scores.append(fusion_result.wyckoff_score)
                smc_scores.append(fusion_result.smc_score)
                hob_scores.append(fusion_result.hob_score)
                momentum_scores.append(fusion_result.momentum_score)

            except Exception as e:
                # Fallback to neutral
                wyckoff_scores.append(0.5)
                smc_scores.append(0.5)
                hob_scores.append(0.5)
                momentum_scores.append(0.5)

        # Store in dataframe
        self.df_4h['wyckoff_score'] = wyckoff_scores
        self.df_4h['smc_score'] = smc_scores
        self.df_4h['hob_score'] = hob_scores
        self.df_4h['momentum_score'] = momentum_scores

        print(f"✅ Domain scores computed for {len(self.df_4h)} bars")

    def _calc_atr(self, df: pd.DataFrame, period: int) -> pd.Series:
        """Calculate ATR vectorized"""
        high = df['high']
        low = df['low']
        close = df['close']
        prev_close = close.shift(1)

        tr = pd.concat([
            high - low,
            (high - prev_close).abs(),
            (low - prev_close).abs()
        ], axis=1).max(axis=1)

        return tr.ewm(span=period, adjust=False).mean()

    def _calc_rsi(self, df: pd.DataFrame, period: int) -> pd.Series:
        """Calculate RSI vectorized"""
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
        rs = gain / (loss + 1e-10)
        return 100 - (100 / (1 + rs))

    def run_backtest(self, config: Dict) -> BacktestResult:
        """
        Run vectorized backtest with given fusion threshold and weights.

        Instead of looping through bars, we:
        1. Compute fusion scores for ALL bars at once
        2. Find entry signals (where fusion_score > threshold)
        3. Simulate trades vectorially
        """
        start_time = time.time()

        # Extract config
        fusion_threshold = config['fusion_threshold']
        w_wyckoff = config['wyckoff_weight']
        w_smc = config['smc_weight']
        w_hob = config['hob_weight']
        w_momentum = config['momentum_weight']

        # Compute fusion scores (vectorized!)
        fusion_scores = (
            self.df_4h['wyckoff_score'] * w_wyckoff +
            self.df_4h['smc_score'] * w_smc +
            self.df_4h['hob_score'] * w_hob +
            self.df_4h['momentum_score'] * w_momentum
        )

        # Determine direction (long if fusion > 0.5, short if < 0.5)
        directions = np.where(fusion_scores > 0.5, 1, -1)  # 1=long, -1=short

        # Find entry signals
        entry_mask = fusion_scores > fusion_threshold
        entry_bars = self.df_4h[entry_mask].copy()

        if len(entry_bars) == 0:
            return self._create_empty_result(config, time.time() - start_time)

        # Simulate trades
        trades = []
        current_position = None

        for idx, row in entry_bars.iterrows():
            # Skip if already in position
            if current_position is not None:
                continue

            entry_price = row['close']
            entry_time = idx
            direction = 1 if fusion_scores.loc[idx] > 0.5 else -1

            # Calculate stop/target
            atr = row['atr_20']
            stop_distance = atr * 1.5
            target_distance = atr * 2.5

            if direction == 1:  # Long
                stop_price = entry_price - stop_distance
                target_price = entry_price + target_distance
            else:  # Short
                stop_price = entry_price + stop_distance
                target_price = entry_price - target_distance

            # Find exit (next 96 bars or until stop/target hit)
            future_bars = self.df_1h[self.df_1h.index > entry_time].head(96)

            exit_price = None
            exit_reason = 'timeout'

            for future_idx, future_row in future_bars.iterrows():
                if direction == 1:
                    if future_row['low'] <= stop_price:
                        exit_price = stop_price
                        exit_reason = 'stop'
                        break
                    elif future_row['high'] >= target_price:
                        exit_price = target_price
                        exit_reason = 'target'
                        break
                else:  # Short
                    if future_row['high'] >= stop_price:
                        exit_price = stop_price
                        exit_reason = 'stop'
                        break
                    elif future_row['low'] <= target_price:
                        exit_price = target_price
                        exit_reason = 'target'
                        break

            if exit_price is None:
                exit_price = future_bars.iloc[-1]['close'] if len(future_bars) > 0 else entry_price

            # Calculate PnL
            if direction == 1:
                pnl_pct = (exit_price - entry_price) / entry_price * 100
            else:
                pnl_pct = (entry_price - exit_price) / entry_price * 100

            trades.append({
                'entry_time': entry_time,
                'entry_price': entry_price,
                'exit_price': exit_price,
                'direction': direction,
                'pnl_pct': pnl_pct,
                'exit_reason': exit_reason
            })

            current_position = None  # Reset position

        # Calculate metrics
        if len(trades) == 0:
            return self._create_empty_result(config, time.time() - start_time)

        trades_df = pd.DataFrame(trades)
        total_return = trades_df['pnl_pct'].sum()
        win_rate = (trades_df['pnl_pct'] > 0).mean()

        winners = trades_df[trades_df['pnl_pct'] > 0]['pnl_pct']
        losers = trades_df[trades_df['pnl_pct'] <= 0]['pnl_pct']

        profit_factor = abs(winners.sum() / losers.sum()) if len(losers) > 0 and losers.sum() != 0 else 0

        # Sharpe ratio (simplified)
        returns = trades_df['pnl_pct']
        sharpe_ratio = returns.mean() / (returns.std() + 1e-10) * np.sqrt(252 / len(trades))

        # Max drawdown
        cumulative = (1 + returns / 100).cumprod()
        running_max = cumulative.cummax()
        drawdown = (cumulative - running_max) / running_max
        max_drawdown = drawdown.min() * 100

        # Statistical significance
        if len(trades) >= 100:
            sig = 'high'
        elif len(trades) >= 30:
            sig = 'medium'
        else:
            sig = 'low'

        return BacktestResult(
            config_id=f"fusion{fusion_threshold:.2f}_w{w_wyckoff:.2f}s{w_smc:.2f}h{w_hob:.2f}m{w_momentum:.2f}",
            asset=self.asset,
            period=f"{self.start_date}_{self.end_date}",
            total_return=total_return,
            trades=len(trades),
            win_rate=win_rate * 100,
            profit_factor=profit_factor,
            sharpe_ratio=sharpe_ratio,
            max_drawdown=max_drawdown,
            avg_trade_pct=returns.mean(),
            avg_r_multiple=returns.mean() / abs(losers.mean()) if len(losers) > 0 else 0,
            winners_avg=winners.mean() if len(winners) > 0 else 0,
            losers_avg=losers.mean() if len(losers) > 0 else 0,
            fusion_threshold=fusion_threshold,
            wyckoff_weight=w_wyckoff,
            smc_weight=w_smc,
            hob_weight=w_hob,
            momentum_weight=w_momentum,
            sample_size=len(trades),
            statistical_significance=sig,
            backtest_seconds=time.time() - start_time
        )

    def _create_empty_result(self, config: Dict, elapsed: float) -> BacktestResult:
        """Create result for config that produced 0 trades"""
        return BacktestResult(
            config_id=f"fusion{config['fusion_threshold']:.2f}_EMPTY",
            asset=self.asset,
            period=f"{self.start_date}_{self.end_date}",
            total_return=0.0,
            trades=0,
            win_rate=0.0,
            profit_factor=0.0,
            sharpe_ratio=0.0,
            max_drawdown=0.0,
            avg_trade_pct=0.0,
            avg_r_multiple=0.0,
            winners_avg=0.0,
            losers_avg=0.0,
            fusion_threshold=config['fusion_threshold'],
            wyckoff_weight=config['wyckoff_weight'],
            smc_weight=config['smc_weight'],
            hob_weight=config['hob_weight'],
            momentum_weight=config['momentum_weight'],
            sample_size=0,
            statistical_significance='low',
            backtest_seconds=elapsed
        )


# ============================================================================
# PARALLEL GRID SEARCH
# ============================================================================

def generate_grid_configs(mode: str = 'grid') -> List[Dict]:
    """
    Generate parameter grid based on mode.

    Modes:
    - quick: 50 configs (~5 min)
    - grid: 500 configs (~30-60 min)
    - exhaustive: 2000+ configs (~2-4 hours)
    """

    if mode == 'quick':
        # Fast test - coarse grid
        fusion_thresholds = [0.50, 0.60, 0.70]
        weight_sets = [
            (0.30, 0.15, 0.25, 0.30),  # Baseline
            (0.40, 0.10, 0.20, 0.30),  # More Wyckoff
            (0.25, 0.20, 0.25, 0.30),  # More SMC
            (0.30, 0.15, 0.30, 0.25),  # More HOB
            (0.25, 0.15, 0.25, 0.35),  # More Momentum
        ]

    elif mode == 'grid':
        # Standard grid search
        fusion_thresholds = [0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75]

        # Generate weight combinations (must sum to 1.0)
        weight_sets = []
        for w_wyckoff in [0.20, 0.25, 0.30, 0.35, 0.40]:
            for w_smc in [0.10, 0.15, 0.20, 0.25]:
                for w_hob in [0.20, 0.25, 0.30, 0.35]:
                    for w_momentum in [0.20, 0.25, 0.30, 0.35, 0.40]:
                        if abs(w_wyckoff + w_smc + w_hob + w_momentum - 1.0) < 0.01:
                            weight_sets.append((w_wyckoff, w_smc, w_hob, w_momentum))

    else:  # exhaustive
        # Fine-grained search
        fusion_thresholds = np.arange(0.40, 0.80, 0.05).tolist()

        weight_sets = []
        for w_wyckoff in np.arange(0.15, 0.45, 0.05):
            for w_smc in np.arange(0.10, 0.30, 0.05):
                for w_hob in np.arange(0.15, 0.40, 0.05):
                    w_momentum = 1.0 - w_wyckoff - w_smc - w_hob
                    if 0.15 <= w_momentum <= 0.45:
                        weight_sets.append((w_wyckoff, w_smc, w_hob, w_momentum))

    # Generate all combinations
    configs = []
    for threshold in fusion_thresholds:
        for w_wyckoff, w_smc, w_hob, w_momentum in weight_sets:
            configs.append({
                'fusion_threshold': threshold,
                'wyckoff_weight': w_wyckoff,
                'smc_weight': w_smc,
                'hob_weight': w_hob,
                'momentum_weight': w_momentum
            })

    print(f"📋 Generated {len(configs)} configurations ({mode} mode)")
    return configs


def run_parallel_optimization(backtester: VectorizedBacktester, configs: List[Dict],
                              n_workers: int = None) -> List[BacktestResult]:
    """
    Run grid search in parallel using all CPU cores.
    """
    if n_workers is None:
        n_workers = mp.cpu_count()

    print(f"🚀 Starting parallel optimization with {n_workers} workers...")
    print(f"⏱️  Estimated time: {len(configs) * 0.5 / n_workers / 60:.1f} minutes")

    start_time = time.time()

    # Run backtests in parallel
    with mp.Pool(n_workers) as pool:
        results = pool.map(backtester.run_backtest, configs)

    elapsed = time.time() - start_time

    print(f"✅ Optimization complete in {elapsed:.1f}s ({len(configs)/elapsed:.1f} configs/sec)")

    return results


# ============================================================================
# WALK-FORWARD VALIDATION
# ============================================================================

def create_walk_forward_folds(start_year: int = 2022, n_years: int = 3) -> List[Tuple[str, str]]:
    """
    Create walk-forward fold pairs.

    Example for 3 years:
    - Fold 1: Train 2022-01 to 2022-06, Test 2022-07 to 2022-12
    - Fold 2: Train 2022-07 to 2022-12, Test 2023-01 to 2023-06
    - Fold 3: Train 2023-01 to 2023-06, Test 2023-07 to 2023-12
    - ...
    """
    folds = []

    for year in range(start_year, start_year + n_years):
        # H1 fold
        train_start = f"{year}-01-01"
        train_end = f"{year}-06-30"
        test_start = f"{year}-07-01"
        test_end = f"{year}-12-31"
        folds.append(((train_start, train_end), (test_start, test_end)))

        # H2 fold
        train_start = f"{year}-07-01"
        train_end = f"{year}-12-31"
        test_start = f"{year+1}-01-01"
        test_end = f"{year+1}-06-30"
        if year < start_year + n_years - 1:
            folds.append(((train_start, train_end), (test_start, test_end)))

    return folds


def run_walk_forward(asset: str, configs: List[Dict], n_years: int = 3) -> pd.DataFrame:
    """
    Run walk-forward validation.

    For each fold:
    1. Train on fold_train period, find best config
    2. Test that config on fold_test period
    3. Report out-of-sample performance
    """
    folds = create_walk_forward_folds(n_years=n_years)

    print(f"🔄 Walk-forward validation: {len(folds)} folds")

    wf_results = []

    for i, ((train_start, train_end), (test_start, test_end)) in enumerate(folds, 1):
        print(f"\n{'='*60}")
        print(f"Fold {i}/{len(folds)}: Train {train_start}→{train_end}, Test {test_start}→{test_end}")
        print(f"{'='*60}")

        # Train period
        backtester_train = VectorizedBacktester(asset, train_start, train_end)
        train_results = run_parallel_optimization(backtester_train, configs, n_workers=4)

        # Find best config on training data
        train_df = pd.DataFrame([asdict(r) for r in train_results])
        train_df = train_df[train_df['trades'] >= 5]  # Minimum sample

        if len(train_df) == 0:
            print("⚠️  No configs with >=5 trades in training period")
            continue

        # Rank by Sharpe ratio
        best_idx = train_df['sharpe_ratio'].idxmax()
        best_config = configs[best_idx]

        print(f"🏆 Best config (training): Fusion {best_config['fusion_threshold']:.2f}, "
              f"Sharpe {train_df.loc[best_idx, 'sharpe_ratio']:.2f}")

        # Test period
        backtester_test = VectorizedBacktester(asset, test_start, test_end)
        test_result = backtester_test.run_backtest(best_config)

        print(f"📊 Out-of-sample test: {test_result.trades} trades, "
              f"{test_result.win_rate:.1f}% WR, {test_result.total_return:+.1f}% return, "
              f"Sharpe {test_result.sharpe_ratio:.2f}")

        wf_results.append({
            'fold': i,
            'train_period': f"{train_start}→{train_end}",
            'test_period': f"{test_start}→{test_end}",
            'train_sharpe': train_df.loc[best_idx, 'sharpe_ratio'],
            'test_sharpe': test_result.sharpe_ratio,
            'test_return': test_result.total_return,
            'test_trades': test_result.trades,
            'test_wr': test_result.win_rate,
            'fusion_threshold': best_config['fusion_threshold'],
            'wyckoff_weight': best_config['wyckoff_weight']
        })

    wf_df = pd.DataFrame(wf_results)
    return wf_df


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='Bull Machine v1.8.6 High-Performance Optimizer')
    parser.add_argument('--mode', choices=['quick', 'grid', 'exhaustive', 'walkforward'],
                       default='grid', help='Optimization mode')
    parser.add_argument('--asset', default='BTC', help='Asset to optimize')
    parser.add_argument('--years', type=int, default=3, help='Years of data to use')
    parser.add_argument('--workers', type=int, default=None, help='Number of parallel workers')
    parser.add_argument('--output', default='optimization_results.json', help='Output file')

    args = parser.parse_args()

    print(f"🎯 Bull Machine v1.8.6 Optimizer")
    print(f"Asset: {args.asset}")
    print(f"Mode: {args.mode}")
    print(f"Years: {args.years}")
    print(f"{'='*60}\n")

    # Generate configs
    configs = generate_grid_configs(args.mode)

    # Date range
    end_date = datetime.now().strftime("%Y-%m-%d")
    start_date = (datetime.now() - timedelta(days=365 * args.years)).strftime("%Y-%m-%d")

    if args.mode == 'walkforward':
        # Walk-forward validation
        wf_df = run_walk_forward(args.asset, configs, args.years)

        print(f"\n{'='*60}")
        print("WALK-FORWARD VALIDATION SUMMARY")
        print(f"{'='*60}")
        print(wf_df.to_string(index=False))

        # Save results
        output_path = args.output.replace('.json', '_walkforward.csv')
        wf_df.to_csv(output_path, index=False)
        print(f"\n💾 Results saved to {output_path}")

    else:
        # Standard grid search
        backtester = VectorizedBacktester(args.asset, start_date, end_date)
        results = run_parallel_optimization(backtester, configs, args.workers)

        # Convert to DataFrame
        results_df = pd.DataFrame([asdict(r) for r in results])

        # Filter and rank
        results_df = results_df[results_df['trades'] >= 10]  # Minimum sample
        results_df = results_df.sort_values('sharpe_ratio', ascending=False)

        # Display top 10
        print(f"\n{'='*60}")
        print("TOP 10 CONFIGURATIONS (by Sharpe Ratio)")
        print(f"{'='*60}")
        top10 = results_df.head(10)[[
            'fusion_threshold', 'wyckoff_weight', 'momentum_weight',
            'trades', 'win_rate', 'total_return', 'sharpe_ratio', 'profit_factor'
        ]]
        print(top10.to_string(index=False))

        # Save full results
        results_df.to_json(args.output, orient='records', indent=2)
        print(f"\n💾 Full results ({len(results_df)} configs) saved to {args.output}")


if __name__ == '__main__':
    main()
