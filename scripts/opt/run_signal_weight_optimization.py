#!/usr/bin/env python3
"""
Bull Machine v1.6.2 - Signal Weight Optimization
Address low trade frequency and improve R:R through signal weight rebalancing

Current Issues from Extended Backtest:
- ETH: Only 0.2 trades/month (need 1.0+)
- BTC: 0.3 trades/month + negative returns
- Entry thresholds too conservative

Optimization Strategy:
1. Lower entry thresholds (0.28 → 0.20-0.25)
2. Rebalance signal weights (boost volume/momentum vs trend)
3. Add regime-specific adjustments
4. Test multiple weight configurations
"""

import sys
import os
import json
import warnings
from datetime import datetime
import pandas as pd
import numpy as np

sys.path.append('.')
warnings.filterwarnings('ignore')


def load_extended_data(asset, data_paths):
    """Load extended dataset for optimization"""
    print(f"\n=== Loading {asset} for Signal Optimization ===")

    data = {}
    for timeframe, filepath in data_paths.items():
        if os.path.exists(filepath):
            df = pd.read_csv(filepath)

            if 'time' in df.columns:
                df['timestamp'] = pd.to_datetime(df['time'], unit='s')
            elif 'Date' in df.columns:
                df['timestamp'] = pd.to_datetime(df['Date'])

            df.columns = df.columns.str.lower()
            if 'buy+sell v' in df.columns:
                df['volume'] = df['buy+sell v']

            df = df.set_index('timestamp').sort_index()
            start_date = '2022-01-01'
            end_date = '2025-09-01'

            if len(df) > 0:
                actual_start = max(df.index[0], pd.Timestamp(start_date))
                actual_end = min(df.index[-1], pd.Timestamp(end_date))
                df = df[actual_start:actual_end]

            df = add_technical_indicators(df, timeframe)
            data[timeframe] = df
            print(f"  └─ {len(df)} bars from {df.index[0].strftime('%Y-%m-%d')} to {df.index[-1].strftime('%Y-%m-%d')}")

    return data


def add_technical_indicators(df, timeframe):
    """Enhanced technical indicators for optimization"""
    # Core indicators
    df['sma_20'] = df['close'].rolling(20).mean()
    df['sma_50'] = df['close'].rolling(50).mean()
    df['sma_200'] = df['close'].rolling(200).mean()
    df['vol_sma'] = df['volume'].rolling(20).mean()
    df['vol_ratio'] = df['volume'] / df['vol_sma']

    # Enhanced momentum suite
    df['rsi'] = calculate_rsi(df['close'], 14)
    df['rsi_oversold'] = df['rsi'] < 35
    df['rsi_overbought'] = df['rsi'] > 65
    df['momentum_5'] = df['close'].pct_change(5)
    df['momentum_20'] = df['close'].pct_change(20)

    # Volatility and range analysis
    high_low = df['high'] - df['low']
    high_close = np.abs(df['high'] - df['close'].shift())
    low_close = np.abs(df['low'] - df['close'].shift())
    tr = np.maximum(high_low, np.maximum(high_close, low_close))
    df['atr'] = tr.rolling(14).mean()
    df['atr_pct'] = df['atr'] / df['close']

    # Range position and structure
    df['range_position'] = (df['close'] - df['low'].rolling(10).min()) / (df['high'].rolling(10).max() - df['low'].rolling(10).min())
    df['range_position'] = df['range_position'].fillna(0.5)

    # Higher highs/lows detection
    df['hh'] = df['high'] > df['high'].shift(1)
    df['ll'] = df['low'] < df['low'].shift(1)
    df['structure_break'] = df['close'] > df['high'].rolling(20).max().shift(1)

    # Volume patterns
    df['volume_spike'] = df['vol_ratio'] > 1.5
    df['volume_accumulation'] = (df['vol_ratio'] > 1.1) & (df['vol_ratio'] <= 1.5)

    return df


def calculate_rsi(prices, window=14):
    """Calculate RSI"""
    delta = prices.diff()
    gain = delta.where(delta > 0, 0).rolling(window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi


def calculate_optimized_ensemble_score(df, i, weight_config):
    """Optimized ensemble scoring with configurable weights"""
    if i < 50:
        return 0, {}

    current_bar = df.iloc[i]
    lookback = df.iloc[max(0, i-50):i+1]

    # Initialize components with configurable weights
    weights = weight_config['weights']
    components = {}

    # 1. Trend Alignment Component
    sma_20 = current_bar['sma_20']
    sma_50 = current_bar['sma_50']
    sma_200 = current_bar['sma_200']
    close = current_bar['close']

    trend_score = 0
    if not pd.isna(sma_20) and not pd.isna(sma_50) and not pd.isna(sma_200):
        if close > sma_20 > sma_50 > sma_200:
            trend_score = 1.0  # Perfect alignment
        elif close > sma_20 > sma_50:
            trend_score = 0.75
        elif close > sma_20:
            trend_score = 0.5
        elif close > sma_50:
            trend_score = 0.25

    components['trend_alignment'] = trend_score * weights['trend']

    # 2. Volume Expansion (Boosted Weight)
    vol_ratio = current_bar['vol_ratio'] if not pd.isna(current_bar['vol_ratio']) else 1.0
    volume_score = 0

    if vol_ratio > 2.5:
        volume_score = 1.0
    elif vol_ratio > 2.0:
        volume_score = 0.8
    elif vol_ratio > 1.5:
        volume_score = 0.6
    elif vol_ratio > 1.2:
        volume_score = 0.4
    elif vol_ratio > 1.0:
        volume_score = 0.2

    components['volume_expansion'] = volume_score * weights['volume']

    # 3. Momentum Confluence (Enhanced)
    rsi = current_bar['rsi'] if not pd.isna(current_bar['rsi']) else 50
    momentum_5 = current_bar['momentum_5'] if not pd.isna(current_bar['momentum_5']) else 0
    momentum_20 = current_bar['momentum_20'] if not pd.isna(current_bar['momentum_20']) else 0

    momentum_score = 0

    # Oversold bounce setup
    if 25 <= rsi <= 40:
        momentum_score += 0.4
        if momentum_5 > 0.02:
            momentum_score += 0.3
        if momentum_20 > 0.05:
            momentum_score += 0.3

    # Building momentum
    elif 40 <= rsi <= 60:
        if momentum_5 > 0.015:
            momentum_score += 0.5
        if momentum_20 > 0.03:
            momentum_score += 0.3
        if momentum_5 > momentum_20:  # Accelerating
            momentum_score += 0.2

    components['momentum_confluence'] = momentum_score * weights['momentum']

    # 4. Structure Breakout (Simplified but Effective)
    structure_break = current_bar['structure_break'] if not pd.isna(current_bar['structure_break']) else False
    range_pos = current_bar['range_position'] if not pd.isna(current_bar['range_position']) else 0.5

    structure_score = 0
    if structure_break:
        structure_score = 1.0
    elif range_pos > 0.8:
        structure_score = 0.6
    elif range_pos > 0.7:
        structure_score = 0.4
    elif range_pos < 0.3:  # Oversold setup
        structure_score = 0.5

    components['structure_break'] = structure_score * weights['structure']

    # 5. Wyckoff-Style Accumulation
    vol_spike = current_bar['volume_spike'] if not pd.isna(current_bar['volume_spike']) else False
    vol_accum = current_bar['volume_accumulation'] if not pd.isna(current_bar['volume_accumulation']) else False

    wyckoff_score = 0
    # Spring setup: Low range + volume spike
    if range_pos < 0.4 and vol_spike:
        wyckoff_score = 1.0
    # Accumulation: Mid range + steady volume
    elif 0.4 <= range_pos <= 0.6 and vol_accum:
        wyckoff_score = 0.6
    # Test of supply: High range + volume
    elif range_pos > 0.7 and (vol_spike or vol_accum):
        wyckoff_score = 0.4

    components['wyckoff_accumulation'] = wyckoff_score * weights['wyckoff']

    # 6. Volatility Contraction (Mean Reversion Setup)
    atr_pct = current_bar['atr_pct'] if not pd.isna(current_bar['atr_pct']) else 0.05
    atr_avg = lookback['atr_pct'].mean()

    volatility_score = 0
    if atr_pct < atr_avg * 0.6:  # Very low volatility
        volatility_score = 1.0
    elif atr_pct < atr_avg * 0.8:
        volatility_score = 0.6
    elif atr_pct < atr_avg * 1.0:
        volatility_score = 0.3

    components['volatility_contraction'] = volatility_score * weights['volatility']

    # 7. Market Regime Filter
    regime_multiplier = 1.0

    # Penalize extreme conditions
    if atr_pct > 0.12:  # Very high volatility
        regime_multiplier *= 0.6
    if rsi > 75:  # Extreme overbought
        regime_multiplier *= 0.7

    # Boost favorable conditions
    if 30 <= rsi <= 45:  # Sweet spot for longs
        regime_multiplier *= 1.2

    # Calculate final score
    base_score = sum(components.values())
    final_score = base_score * regime_multiplier

    components['regime_multiplier'] = regime_multiplier
    components['final_score'] = final_score

    return final_score, components


def run_weight_optimization_backtest(asset, data, weight_configs):
    """Test multiple weight configurations"""
    print(f"\n=== Running {asset} Weight Optimization ===")

    df = data.get('1D')
    if df is None or len(df) < 200:
        print(f"Insufficient {asset} data")
        return {}

    print(f"Testing {len(weight_configs)} weight configurations on {len(df)} bars")

    results = {}

    for config_name, config in weight_configs.items():
        print(f"\n--- Testing {config_name} ---")

        # Backtest parameters
        entry_threshold = config['entry_threshold']
        cooldown_days = config['cooldown_days']
        risk_pct = config['risk_pct']
        initial_capital = 10000

        # Run backtest
        trades = []
        capital = initial_capital
        last_trade_idx = -999

        for i in range(50, len(df) - 1):
            if i - last_trade_idx < cooldown_days:
                continue

            # Calculate optimized score
            score, components = calculate_optimized_ensemble_score(df, i, config)

            if score >= entry_threshold:
                current_bar = df.iloc[i]
                entry_price = current_bar['close']
                entry_time = current_bar.name

                # Position sizing
                position_size = capital * risk_pct / entry_price

                # Dynamic exit based on signal strength
                if score > 0.6:
                    exit_days = 8
                elif score > 0.4:
                    exit_days = 6
                else:
                    exit_days = 4

                exit_idx = min(i + exit_days, len(df) - 1)
                exit_bar = df.iloc[exit_idx]
                exit_price = exit_bar['close']

                # Calculate PnL
                fee_bps = 5
                slip_bps = 2
                entry_cost = entry_price * (1 + (fee_bps + slip_bps) / 10000)
                exit_proceeds = exit_price * (1 - (fee_bps + slip_bps) / 10000)
                pnl = position_size * (exit_proceeds - entry_cost)
                capital += pnl

                # Enhanced tagging
                tags = []
                if components.get('volume_expansion', 0) > 0.05:
                    tags.append('volume_signal')
                if components.get('momentum_confluence', 0) > 0.05:
                    tags.append('momentum_signal')
                if components.get('structure_break', 0) > 0.05:
                    tags.append('structure_break')
                if components.get('wyckoff_accumulation', 0) > 0.05:
                    tags.append('wyckoff_setup')
                if score > 0.5:
                    tags.append('high_confidence')

                trade = {
                    'entry_time': entry_time,
                    'exit_time': df.index[exit_idx],
                    'entry_price': entry_price,
                    'exit_price': exit_price,
                    'pnl': pnl,
                    'score': score,
                    'components': components,
                    'tags': tags,
                    'config': config_name,
                    'side': 'long'
                }

                trades.append(trade)
                last_trade_idx = i

        # Calculate metrics
        metrics = calculate_comprehensive_metrics(trades, initial_capital)
        metrics['config_name'] = config_name
        metrics['final_capital'] = capital

        results[config_name] = {
            'trades': trades,
            'metrics': metrics,
            'config': config
        }

        print(f"  {config_name}: {metrics['total_trades']} trades, "
              f"{metrics['win_rate']:.1f}% WR, {metrics['total_pnl_pct']:.2f}% PnL, "
              f"PF: {metrics['profit_factor']:.2f}, {metrics['trades_per_month']:.1f}/mo")

    return {
        'asset': asset,
        'optimization_results': results
    }


def calculate_comprehensive_metrics(trades, initial_capital=10000):
    """Calculate trading metrics"""
    if not trades:
        return {
            'total_trades': 0, 'win_rate': 0, 'total_pnl_pct': 0,
            'max_drawdown_pct': 0, 'profit_factor': 0, 'sharpe_ratio': 0,
            'avg_trade': 0, 'trades_per_month': 0, 'gross_profit': 0, 'gross_loss': 0
        }

    total_trades = len(trades)
    winning_trades = len([t for t in trades if t['pnl'] > 0])
    win_rate = (winning_trades / total_trades) * 100

    pnl_series = [t['pnl'] for t in trades]
    total_pnl = sum(pnl_series)
    total_pnl_pct = (total_pnl / initial_capital) * 100
    avg_trade = total_pnl / total_trades

    # Drawdown
    cumulative_pnl = np.cumsum([0] + pnl_series)
    running_max = np.maximum.accumulate(cumulative_pnl)
    drawdowns = (cumulative_pnl - running_max)
    max_drawdown = abs(min(drawdowns))
    max_drawdown_pct = (max_drawdown / initial_capital) * 100

    # Profit factor
    gross_profit = sum([p for p in pnl_series if p > 0])
    gross_loss = abs(sum([p for p in pnl_series if p < 0]))
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')

    # Sharpe
    returns = np.array(pnl_series) / initial_capital
    sharpe_ratio = np.mean(returns) / np.std(returns) * np.sqrt(252) if np.std(returns) > 0 else 0

    # Frequency
    if len(trades) > 1:
        date_range_days = (trades[-1]['entry_time'] - trades[0]['entry_time']).days
        months = date_range_days / 30.44
        trades_per_month = total_trades / months if months > 0 else 0
    else:
        trades_per_month = 0

    return {
        'total_trades': total_trades, 'win_rate': win_rate, 'total_pnl_pct': total_pnl_pct,
        'max_drawdown_pct': max_drawdown_pct, 'profit_factor': profit_factor, 'sharpe_ratio': sharpe_ratio,
        'avg_trade': avg_trade, 'trades_per_month': trades_per_month,
        'gross_profit': gross_profit, 'gross_loss': gross_loss
    }


def run_signal_weight_optimization():
    """Run comprehensive signal weight optimization"""
    print("="*80)
    print("BULL MACHINE v1.6.2 - SIGNAL WEIGHT OPTIMIZATION")
    print("Addressing Low Trade Frequency + Boosting Performance")
    print("="*80)

    # Data paths
    data_paths = {
        'ETH': {'1D': '/Users/raymondghandchi/Downloads/Chart logs 2/COINBASE_ETHUSD, 1D_64942.csv'},
        'BTC': {'1D': '/Users/raymondghandchi/Downloads/Chart logs 2/COINBASE_BTCUSD, 1D_85c84.csv'}
    }

    # Weight configurations to test
    weight_configs = {
        'Conservative': {
            'entry_threshold': 0.28,
            'cooldown_days': 14,
            'risk_pct': 0.02,
            'weights': {
                'trend': 0.25,
                'volume': 0.15,
                'momentum': 0.20,
                'structure': 0.15,
                'wyckoff': 0.15,
                'volatility': 0.10
            }
        },
        'Volume_Focused': {
            'entry_threshold': 0.22,
            'cooldown_days': 10,
            'risk_pct': 0.025,
            'weights': {
                'trend': 0.15,
                'volume': 0.30,     # Boosted
                'momentum': 0.25,   # Boosted
                'structure': 0.15,
                'wyckoff': 0.10,
                'volatility': 0.05
            }
        },
        'Momentum_Breakout': {
            'entry_threshold': 0.20,
            'cooldown_days': 8,
            'risk_pct': 0.03,
            'weights': {
                'trend': 0.10,
                'volume': 0.20,
                'momentum': 0.35,   # Highest weight
                'structure': 0.20,  # Boosted for breakouts
                'wyckoff': 0.10,
                'volatility': 0.05
            }
        },
        'Wyckoff_Spring': {
            'entry_threshold': 0.25,
            'cooldown_days': 12,
            'risk_pct': 0.025,
            'weights': {
                'trend': 0.15,
                'volume': 0.20,
                'momentum': 0.15,
                'structure': 0.15,
                'wyckoff': 0.30,    # Highest weight
                'volatility': 0.05
            }
        },
        'Aggressive_Frequency': {
            'entry_threshold': 0.18,
            'cooldown_days': 6,
            'risk_pct': 0.02,
            'weights': {
                'trend': 0.10,
                'volume': 0.25,
                'momentum': 0.30,
                'structure': 0.20,
                'wyckoff': 0.10,
                'volatility': 0.05
            }
        }
    }

    results = {}

    # Run optimization for each asset
    for asset, paths in data_paths.items():
        try:
            data = load_extended_data(asset, paths)

            if data.get('1D') is not None and len(data['1D']) > 200:
                result = run_weight_optimization_backtest(asset, data, weight_configs)
                results[asset] = result
            else:
                print(f"\nSkipping {asset} - insufficient data")

        except Exception as e:
            print(f"\nERROR optimizing {asset}: {e}")
            import traceback
            traceback.print_exc()

    # Comprehensive analysis
    print("\n" + "="*80)
    print("SIGNAL WEIGHT OPTIMIZATION RESULTS")
    print("="*80)

    # Results table
    print(f"{'Asset':<6} {'Config':<18} {'Trades':<8} {'Win %':<8} {'PnL %':<10} {'Max DD':<8} {'PF':<6} {'TR/Mo':<6} {'Score':<6}")
    print("-" * 80)

    best_configs = {}

    for asset, result in results.items():
        if result and 'optimization_results' in result:
            best_config = None
            best_score = 0

            for config_name, config_result in result['optimization_results'].items():
                metrics = config_result['metrics']

                # Optimization score: (PnL% * PF * TradesToMo) / (MaxDD% + 1)
                # Favors: High returns, good PF, adequate frequency, low drawdown
                score = (metrics['total_pnl_pct'] * metrics['profit_factor'] * metrics['trades_per_month']) / (metrics['max_drawdown_pct'] + 1)

                if score > best_score:
                    best_score = score
                    best_config = config_name

                print(f"{asset:<6} {config_name:<18} {metrics['total_trades']:<8} "
                      f"{metrics['win_rate']:<8.1f} {metrics['total_pnl_pct']:<10.2f} "
                      f"{metrics['max_drawdown_pct']:<8.2f} {metrics['profit_factor']:<6.2f} "
                      f"{metrics['trades_per_month']:<6.1f} {score:<6.1f}")

            best_configs[asset] = best_config

    # Best configuration deep dive
    print("\n" + "="*80)
    print("OPTIMAL CONFIGURATION ANALYSIS")
    print("="*80)

    combined_performance = {'total_trades': 0, 'total_pnl': 0, 'total_capital': 20000}

    for asset, best_config in best_configs.items():
        if asset in results and best_config:
            best_result = results[asset]['optimization_results'][best_config]
            metrics = best_result['metrics']
            trades = best_result['trades']
            config = best_result['config']

            print(f"\n{asset} Optimal: {best_config}")
            print(f"  Performance Metrics:")
            print(f"    Entry Threshold: {config['entry_threshold']}")
            print(f"    Total Trades: {metrics['total_trades']} ({metrics['trades_per_month']:.1f}/month)")
            print(f"    Win Rate: {metrics['win_rate']:.1f}%")
            print(f"    Total Return: {metrics['total_pnl_pct']:.2f}%")
            print(f"    Max Drawdown: {metrics['max_drawdown_pct']:.2f}%")
            print(f"    Profit Factor: {metrics['profit_factor']:.2f}")
            print(f"    Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")

            # Signal breakdown
            volume_trades = [t for t in trades if 'volume_signal' in t.get('tags', [])]
            momentum_trades = [t for t in trades if 'momentum_signal' in t.get('tags', [])]
            structure_trades = [t for t in trades if 'structure_break' in t.get('tags', [])]
            wyckoff_trades = [t for t in trades if 'wyckoff_setup' in t.get('tags', [])]

            print(f"  Signal Distribution:")
            print(f"    Volume signals: {len(volume_trades)} ({len(volume_trades)/len(trades)*100:.1f}%)")
            print(f"    Momentum signals: {len(momentum_trades)} ({len(momentum_trades)/len(trades)*100:.1f}%)")
            print(f"    Structure breaks: {len(structure_trades)} ({len(structure_trades)/len(trades)*100:.1f}%)")
            print(f"    Wyckoff setups: {len(wyckoff_trades)} ({len(wyckoff_trades)/len(trades)*100:.1f}%)")

            # Weight analysis
            weights = config['weights']
            print(f"  Optimal Weights:")
            for component, weight in sorted(weights.items(), key=lambda x: x[1], reverse=True):
                print(f"    {component}: {weight:.2f}")

            # Add to combined performance
            combined_performance['total_trades'] += metrics['total_trades']
            combined_performance['total_pnl'] += metrics['final_capital'] - 10000

    # Combined portfolio assessment
    combined_pnl_pct = (combined_performance['total_pnl'] / combined_performance['total_capital']) * 100

    print(f"\n  Combined Portfolio:")
    print(f"    Total Trades: {combined_performance['total_trades']}")
    print(f"    Combined Return: {combined_pnl_pct:.2f}%")

    # Final RC-ready assessment
    print("\n" + "="*80)
    print("FINAL RC-READY ASSESSMENT (Optimized)")
    print("="*80)

    rc_criteria = {
        'ETH': {'min_pnl': 8, 'min_pf': 1.5, 'max_dd': 15, 'min_freq': 0.8},
        'BTC': {'min_pnl': 12, 'min_pf': 1.5, 'max_dd': 20, 'min_freq': 0.8}
    }

    overall_ready = True
    ready_count = 0

    for asset, best_config in best_configs.items():
        if asset in results and best_config and asset in rc_criteria:
            best_result = results[asset]['optimization_results'][best_config]
            metrics = best_result['metrics']
            criteria = rc_criteria[asset]

            pnl_ok = metrics['total_pnl_pct'] >= criteria['min_pnl']
            pf_ok = metrics['profit_factor'] >= criteria['min_pf']
            dd_ok = metrics['max_drawdown_pct'] <= criteria['max_dd']
            freq_ok = metrics['trades_per_month'] >= criteria['min_freq']

            asset_ready = pnl_ok and pf_ok and dd_ok and freq_ok
            if asset_ready:
                ready_count += 1

            status = "✅" if asset_ready else "❌"
            print(f"{asset} RC-Ready ({best_config}): {status}")
            print(f"  PnL: {metrics['total_pnl_pct']:.1f}% (need ≥{criteria['min_pnl']}%) {'✅' if pnl_ok else '❌'}")
            print(f"  PF: {metrics['profit_factor']:.2f} (need ≥{criteria['min_pf']}) {'✅' if pf_ok else '❌'}")
            print(f"  DD: {metrics['max_drawdown_pct']:.1f}% (need ≤{criteria['max_dd']}%) {'✅' if dd_ok else '❌'}")
            print(f"  Freq: {metrics['trades_per_month']:.1f}/mo (need ≥{criteria['min_freq']}) {'✅' if freq_ok else '❌'}")

            if not asset_ready:
                overall_ready = False

    # Final verdict
    if ready_count >= 1:
        status_msg = f"✅ PARTIAL RC-READY ({ready_count}/2 assets optimized)"
    else:
        status_msg = "❌ NEEDS FURTHER OPTIMIZATION"

    print(f"\nFinal RC-Ready Status: {status_msg}")
    print(f"Combined Portfolio Return: {combined_pnl_pct:.2f}%")

    # Save results
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    results_file = f"reports/v162_signal_optimization_{timestamp}.json"

    os.makedirs('reports', exist_ok=True)
    with open(results_file, 'w') as f:
        serializable_results = {}
        for asset, result in results.items():
            if result:
                serializable_result = dict(result)
                if 'optimization_results' in serializable_result:
                    for config_name, config_result in serializable_result['optimization_results'].items():
                        if 'trades' in config_result:
                            for trade in config_result['trades']:
                                if 'entry_time' in trade:
                                    trade['entry_time'] = trade['entry_time'].isoformat()
                                if 'exit_time' in trade:
                                    trade['exit_time'] = trade['exit_time'].isoformat()
                serializable_results[asset] = serializable_result

        json.dump(serializable_results, f, indent=2, default=str)

    print(f"\nOptimization results saved to: {results_file}")
    return results


if __name__ == '__main__':
    results = run_signal_weight_optimization()