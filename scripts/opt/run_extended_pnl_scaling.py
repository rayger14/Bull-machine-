#!/usr/bin/env python3
"""
Bull Machine v1.6.2 - Extended Dataset + PnL Scaling Experiment
Full market cycle validation (2022-2025) + Multiple risk levels

Tests:
1. Extended 36-month backtest on 1D data (full market cycles)
2. PnL scaling experiment: 2%, 3%, 5% risk levels
3. Multi-regime validation (bear, bull, sideways markets)
"""

import sys
import os
import json
import warnings
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

sys.path.append('.')
warnings.filterwarnings('ignore')


def load_extended_data(asset, data_paths):
    """Load extended dataset for full market cycle analysis"""
    print(f"\n=== Loading {asset} Extended Dataset ===")

    data = {}
    for timeframe, filepath in data_paths.items():
        if os.path.exists(filepath):
            print(f"Loading {timeframe}: {filepath}")
            df = pd.read_csv(filepath)

            # Handle Chart Logs 2 format
            if 'time' in df.columns:
                df['timestamp'] = pd.to_datetime(df['time'], unit='s')
            elif 'Date' in df.columns:
                df['timestamp'] = pd.to_datetime(df['Date'])

            # Standardize columns
            df.columns = df.columns.str.lower()
            if 'buy+sell v' in df.columns:
                df['volume'] = df['buy+sell v']

            # Extended date range for full market cycle (2022-2025)
            df = df.set_index('timestamp').sort_index()
            start_date = '2022-01-01'
            end_date = '2025-09-01'

            # Filter to available date range
            if len(df) > 0:
                actual_start = max(df.index[0], pd.Timestamp(start_date))
                actual_end = min(df.index[-1], pd.Timestamp(end_date))
                df = df[actual_start:actual_end]

            # Add technical indicators
            df = add_technical_indicators(df, timeframe)

            data[timeframe] = df
            print(f"  └─ {len(df)} bars from {df.index[0].strftime('%Y-%m-%d')} to {df.index[-1].strftime('%Y-%m-%d')}")
        else:
            print(f"WARNING: {filepath} not found")

    return data


def add_technical_indicators(df, timeframe):
    """Add technical indicators for enhanced ensemble scoring"""
    # SMAs for trend detection
    df['sma_20'] = df['close'].rolling(20).mean()
    df['sma_50'] = df['close'].rolling(50).mean()
    df['sma_200'] = df['close'].rolling(200).mean()

    # Volume indicators
    df['vol_sma'] = df['volume'].rolling(20).mean()
    df['vol_ratio'] = df['volume'] / df['vol_sma']

    # ATR for volatility
    high_low = df['high'] - df['low']
    high_close = np.abs(df['high'] - df['close'].shift())
    low_close = np.abs(df['low'] - df['close'].shift())
    tr = np.maximum(high_low, np.maximum(high_close, low_close))
    df['atr'] = tr.rolling(14).mean()

    # Enhanced momentum indicators
    df['rsi'] = calculate_rsi(df['close'], 14)
    df['momentum'] = df['close'].pct_change(5)
    df['range_position'] = (df['close'] - df['low'].rolling(10).min()) / (df['high'].rolling(10).max() - df['low'].rolling(10).min())
    df['range_position'] = df['range_position'].fillna(0.5)

    # Market regime detection
    df['volatility_regime'] = df['atr'] / df['close']
    df['trend_strength'] = (df['sma_20'] - df['sma_50']) / df['sma_50']

    return df


def calculate_rsi(prices, window=14):
    """Calculate RSI indicator"""
    delta = prices.diff()
    gain = delta.where(delta > 0, 0).rolling(window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi


def calculate_enhanced_ensemble_score(df, i, config):
    """Enhanced ensemble scoring for 1D data with all confluence layers"""
    if i < 50:
        return 0, {}

    current_bar = df.iloc[i]
    lookback_window = df.iloc[max(0, i-50):i+1]

    # Initialize components
    components = {
        'trend_alignment': 0,
        'wyckoff_accumulation': 0,
        'volume_expansion': 0,
        'momentum_confluence': 0,
        'structure_break': 0,
        'volatility_contraction': 0,
        'fibonacci_zone': 0,
        'regime_filter': 1.0
    }

    # 1. Trend Alignment (25% weight)
    sma_20 = current_bar['sma_20']
    sma_50 = current_bar['sma_50']
    sma_200 = current_bar['sma_200']
    close = current_bar['close']

    if not pd.isna(sma_20) and not pd.isna(sma_50) and not pd.isna(sma_200):
        if close > sma_20 > sma_50 > sma_200:
            components['trend_alignment'] = 0.25
        elif close > sma_20 > sma_50:
            components['trend_alignment'] = 0.18
        elif close > sma_20:
            components['trend_alignment'] = 0.12

    # 2. Wyckoff Accumulation/Spring (20% weight)
    vol_ratio = current_bar['vol_ratio'] if not pd.isna(current_bar['vol_ratio']) else 1.0
    range_pos = current_bar['range_position'] if not pd.isna(current_bar['range_position']) else 0.5

    # Spring setup: Low range position + volume spike
    if range_pos < 0.4 and vol_ratio > 1.3:
        components['wyckoff_accumulation'] = 0.20
    # Accumulation: Mid range + steady volume
    elif 0.4 <= range_pos <= 0.6 and vol_ratio > 1.1:
        components['wyckoff_accumulation'] = 0.15

    # 3. Volume Expansion (15% weight)
    if vol_ratio > 2.0:
        components['volume_expansion'] = 0.15
    elif vol_ratio > 1.5:
        components['volume_expansion'] = 0.10
    elif vol_ratio > 1.2:
        components['volume_expansion'] = 0.05

    # 4. Momentum Confluence (15% weight)
    rsi = current_bar['rsi'] if not pd.isna(current_bar['rsi']) else 50
    momentum = current_bar['momentum'] if not pd.isna(current_bar['momentum']) else 0

    # Oversold bounce potential
    if 25 <= rsi <= 40 and momentum > 0.01:
        components['momentum_confluence'] = 0.15
    # Building momentum
    elif 40 <= rsi <= 60 and momentum > 0.015:
        components['momentum_confluence'] = 0.12

    # 5. Structure Break (10% weight)
    recent_high = lookback_window['high'].rolling(20).max().iloc[-1]
    recent_low = lookback_window['low'].rolling(20).min().iloc[-1]

    if close > recent_high * 1.02:  # Break above resistance
        components['structure_break'] = 0.10
    elif close > recent_high * 1.005:  # Testing resistance
        components['structure_break'] = 0.05

    # 6. Volatility Contraction (10% weight)
    atr_current = current_bar['atr'] if not pd.isna(current_bar['atr']) else 0
    atr_avg = lookback_window['atr'].mean()

    if atr_current < atr_avg * 0.7:  # Low volatility setup
        components['volatility_contraction'] = 0.10
    elif atr_current < atr_avg * 0.85:
        components['volatility_contraction'] = 0.05

    # 7. Fibonacci Zone (5% weight)
    # Check if near key Fib retracement levels
    if 0.35 <= range_pos <= 0.40 or 0.60 <= range_pos <= 0.65:  # 38.2% or 61.8%
        components['fibonacci_zone'] = 0.05
    elif 0.48 <= range_pos <= 0.52:  # 50%
        components['fibonacci_zone'] = 0.03

    # 8. Market Regime Filter
    volatility_regime = current_bar['volatility_regime'] if not pd.isna(current_bar['volatility_regime']) else 0.05
    trend_strength = current_bar['trend_strength'] if not pd.isna(current_bar['trend_strength']) else 0

    # Penalize high volatility or strong downtrends
    if volatility_regime > 0.08:  # High volatility
        components['regime_filter'] *= 0.7
    if trend_strength < -0.05:  # Strong downtrend
        components['regime_filter'] *= 0.5

    # Calculate final score
    base_score = sum([v for k, v in components.items() if k != 'regime_filter'])
    final_score = base_score * components['regime_filter']

    return final_score, components


def run_pnl_scaling_backtest(asset, data, config, risk_levels=[0.02, 0.03, 0.05]):
    """Run backtest with multiple risk levels for PnL scaling experiment"""
    print(f"\n=== Running {asset} PnL Scaling Experiment ===")

    # Use 1D data for extended timeframe
    df = data.get('1D')
    if df is None or len(df) < 200:
        print(f"Insufficient {asset} 1D data for backtest")
        return {}

    print(f"Extended backtest: {len(df)} daily bars from {df.index[0].strftime('%Y-%m-%d')} to {df.index[-1].strftime('%Y-%m-%d')}")

    results = {}

    for risk_pct in risk_levels:
        print(f"\n--- Testing {risk_pct*100}% Risk Level ---")

        # Backtest parameters
        entry_threshold = config.get('entry_threshold', 0.28)  # Slightly lower for 1D
        cooldown_days = config.get('cooldown_bars', 14)  # 2 weeks for 1D
        initial_capital = 10000

        # Run backtest
        trades = []
        capital = initial_capital
        last_trade_idx = -999

        for i in range(50, len(df) - 1):
            if i - last_trade_idx < cooldown_days:
                continue

            # Calculate ensemble score
            score, components = calculate_enhanced_ensemble_score(df, i, config)

            if score >= entry_threshold:
                current_bar = df.iloc[i]
                entry_price = current_bar['close']
                entry_time = current_bar.name

                # Position sizing
                position_size = capital * risk_pct / entry_price

                # Exit logic based on signal strength
                if score > 0.45:
                    exit_days = 8
                elif score > 0.35:
                    exit_days = 6
                else:
                    exit_days = 4

                exit_idx = min(i + exit_days, len(df) - 1)
                exit_bar = df.iloc[exit_idx]
                exit_price = exit_bar['close']

                # Calculate PnL with fees/slippage
                fee_bps = 5
                slip_bps = 2

                entry_cost = entry_price * (1 + (fee_bps + slip_bps) / 10000)
                exit_proceeds = exit_price * (1 - (fee_bps + slip_bps) / 10000)

                pnl = position_size * (exit_proceeds - entry_cost)
                capital += pnl

                # Enhanced tagging
                tags = []
                if components.get('trend_alignment', 0) > 0.15:
                    tags.append('trend_aligned')
                if components.get('wyckoff_accumulation', 0) > 0.12:
                    tags.append('wyckoff_setup')
                if components.get('volume_expansion', 0) > 0.08:
                    tags.append('volume_expansion')
                if score > 0.40:
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
                    'risk_pct': risk_pct,
                    'side': 'long',
                    'exit_days': exit_days
                }

                trades.append(trade)
                last_trade_idx = i

        # Calculate metrics
        metrics = calculate_comprehensive_metrics(trades, initial_capital)

        # Add risk-adjusted metrics
        metrics['risk_level'] = risk_pct
        metrics['capital_efficiency'] = metrics['total_pnl_pct'] / (risk_pct * 100) if risk_pct > 0 else 0
        metrics['return_per_trade'] = metrics['total_pnl_pct'] / max(metrics['total_trades'], 1)

        results[f"{risk_pct:.1%}"] = {
            'trades': trades,
            'metrics': metrics,
            'final_capital': capital
        }

        print(f"  Risk {risk_pct:.1%}: {metrics['total_trades']} trades, "
              f"{metrics['win_rate']:.1f}% WR, {metrics['total_pnl_pct']:.2f}% PnL, "
              f"PF: {metrics['profit_factor']:.2f}, DD: {metrics['max_drawdown_pct']:.2f}%")

    return {
        'asset': asset,
        'extended_results': results,
        'config': config
    }


def calculate_comprehensive_metrics(trades, initial_capital=10000):
    """Calculate comprehensive trading metrics"""
    if not trades:
        return {
            'total_trades': 0, 'win_rate': 0, 'total_pnl_pct': 0,
            'max_drawdown_pct': 0, 'profit_factor': 0, 'sharpe_ratio': 0,
            'avg_trade': 0, 'trades_per_month': 0, 'gross_profit': 0, 'gross_loss': 0
        }

    # Basic metrics
    total_trades = len(trades)
    winning_trades = len([t for t in trades if t['pnl'] > 0])
    win_rate = (winning_trades / total_trades) * 100

    # PnL analysis
    pnl_series = [t['pnl'] for t in trades]
    total_pnl = sum(pnl_series)
    total_pnl_pct = (total_pnl / initial_capital) * 100
    avg_trade = total_pnl / total_trades

    # Drawdown calculation
    cumulative_pnl = np.cumsum([0] + pnl_series)
    running_max = np.maximum.accumulate(cumulative_pnl)
    drawdowns = (cumulative_pnl - running_max)
    max_drawdown = abs(min(drawdowns))
    max_drawdown_pct = (max_drawdown / initial_capital) * 100

    # Profit factor
    gross_profit = sum([p for p in pnl_series if p > 0])
    gross_loss = abs(sum([p for p in pnl_series if p < 0]))
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')

    # Sharpe ratio
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


def load_asset_config(asset):
    """Load asset-specific configuration"""
    config_path = f"configs/v160/assets/{asset}.json"
    try:
        with open(config_path, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        return {
            'entry_threshold': 0.28,
            'cooldown_bars': 14,
            'risk': {'risk_pct': 0.02},
            'features': {'bojan': True, 'po3': True, 'wyckoff_m1m2': True}
        }


def run_extended_pnl_experiment():
    """Run extended dataset + PnL scaling experiment"""
    print("="*80)
    print("BULL MACHINE v1.6.2 - EXTENDED DATASET + PNL SCALING EXPERIMENT")
    print("Full Market Cycle (2022-2025) | Multiple Risk Levels | 1D Enhanced Ensemble")
    print("="*80)

    # Data paths (using 1D for extended timeframe)
    data_paths = {
        'ETH': {
            '1D': '/Users/raymondghandchi/Downloads/Chart logs 2/COINBASE_ETHUSD, 1D_64942.csv'
        },
        'BTC': {
            '1D': '/Users/raymondghandchi/Downloads/Chart logs 2/COINBASE_BTCUSD, 1D_85c84.csv'
        }
    }

    # Risk levels for scaling experiment
    risk_levels = [0.015, 0.02, 0.03, 0.05]

    results = {}

    # Run extended backtests for each asset
    for asset, paths in data_paths.items():
        try:
            # Load extended dataset
            data = load_extended_data(asset, paths)

            if data.get('1D') is not None and len(data['1D']) > 200:
                # Load config
                config = load_asset_config(asset)

                # Run PnL scaling experiment
                result = run_pnl_scaling_backtest(asset, data, config, risk_levels)
                results[asset] = result
            else:
                print(f"\nSkipping {asset} - insufficient extended data")

        except Exception as e:
            print(f"\nERROR running {asset} extended backtest: {e}")
            import traceback
            traceback.print_exc()
            continue

    # Comprehensive analysis
    print("\n" + "="*80)
    print("EXTENDED DATASET + PNL SCALING RESULTS")
    print("="*80)

    # Results table
    print(f"{'Asset':<6} {'Risk':<6} {'Period':<12} {'Trades':<8} {'Win %':<8} {'PnL %':<10} {'Max DD':<10} {'PF':<8} {'TR/Mo':<6}")
    print("-" * 80)

    best_configs = {}

    for asset, result in results.items():
        if result and 'extended_results' in result:
            best_config = None
            best_score = 0

            for risk_level, risk_result in result['extended_results'].items():
                metrics = risk_result['metrics']

                # Calculate date range
                if risk_result['trades']:
                    start_date = risk_result['trades'][0]['entry_time'].strftime('%Y-%m')
                    end_date = risk_result['trades'][-1]['entry_time'].strftime('%Y-%m')
                    period = f"{start_date}..{end_date}"
                else:
                    period = "No trades"

                # Scoring for best config (PnL * PF / Max DD)
                score = (metrics['total_pnl_pct'] * metrics['profit_factor']) / max(metrics['max_drawdown_pct'], 0.1)
                if score > best_score:
                    best_score = score
                    best_config = risk_level

                print(f"{asset:<6} {risk_level:<6} {period:<12} {metrics['total_trades']:<8} "
                      f"{metrics['win_rate']:<8.1f} {metrics['total_pnl_pct']:<10.2f} "
                      f"{metrics['max_drawdown_pct']:<10.2f} {metrics['profit_factor']:<8.2f} "
                      f"{metrics['trades_per_month']:<6.1f}")

            best_configs[asset] = best_config

    # Best configuration analysis
    print("\n" + "="*80)
    print("OPTIMAL RISK LEVEL ANALYSIS")
    print("="*80)

    for asset, best_risk in best_configs.items():
        if asset in results and best_risk:
            best_result = results[asset]['extended_results'][best_risk]
            metrics = best_result['metrics']
            trades = best_result['trades']

            print(f"\n{asset} Optimal Configuration: {best_risk} Risk")
            print(f"  Extended Period Performance:")
            print(f"    Total Trades: {metrics['total_trades']}")
            print(f"    Trade Frequency: {metrics['trades_per_month']:.1f}/month")
            print(f"    Win Rate: {metrics['win_rate']:.1f}%")
            print(f"    Total Return: {metrics['total_pnl_pct']:.2f}%")
            print(f"    Max Drawdown: {metrics['max_drawdown_pct']:.2f}%")
            print(f"    Profit Factor: {metrics['profit_factor']:.2f}")
            print(f"    Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
            print(f"    Capital Efficiency: {metrics['capital_efficiency']:.2f}")

            # Signal analysis
            trend_trades = [t for t in trades if 'trend_aligned' in t.get('tags', [])]
            wyckoff_trades = [t for t in trades if 'wyckoff_setup' in t.get('tags', [])]
            volume_trades = [t for t in trades if 'volume_expansion' in t.get('tags', [])]
            high_conf_trades = [t for t in trades if 'high_confidence' in t.get('tags', [])]

            print(f"    Signal Breakdown:")
            print(f"      Trend-aligned: {len(trend_trades)} ({len(trend_trades)/len(trades)*100:.1f}%)")
            print(f"      Wyckoff setups: {len(wyckoff_trades)} ({len(wyckoff_trades)/len(trades)*100:.1f}%)")
            print(f"      Volume expansion: {len(volume_trades)} ({len(volume_trades)/len(trades)*100:.1f}%)")
            print(f"      High confidence: {len(high_conf_trades)} ({len(high_conf_trades)/len(trades)*100:.1f}%)")

    # RC-ready assessment with optimal configs
    print("\n" + "="*80)
    print("RC-READY ASSESSMENT (Extended + Optimized)")
    print("="*80)

    rc_criteria = {
        'ETH': {'min_pnl': 15, 'min_pf': 1.8, 'max_dd': 12, 'min_trades_month': 1.0},
        'BTC': {'min_pnl': 25, 'min_pf': 1.8, 'max_dd': 18, 'min_trades_month': 1.0}
    }

    overall_ready = True
    for asset, best_risk in best_configs.items():
        if asset in results and best_risk and asset in rc_criteria:
            best_result = results[asset]['extended_results'][best_risk]
            metrics = best_result['metrics']
            criteria = rc_criteria[asset]

            pnl_ok = metrics['total_pnl_pct'] >= criteria['min_pnl']
            pf_ok = metrics['profit_factor'] >= criteria['min_pf']
            dd_ok = metrics['max_drawdown_pct'] <= criteria['max_dd']
            freq_ok = metrics['trades_per_month'] >= criteria['min_trades_month']

            status = "✅" if (pnl_ok and pf_ok and dd_ok and freq_ok) else "❌"
            print(f"{asset} RC-Ready (@ {best_risk}): {status}")
            print(f"  PnL: {metrics['total_pnl_pct']:.1f}% (need ≥{criteria['min_pnl']}%) {'✅' if pnl_ok else '❌'}")
            print(f"  PF: {metrics['profit_factor']:.2f} (need ≥{criteria['min_pf']}) {'✅' if pf_ok else '❌'}")
            print(f"  DD: {metrics['max_drawdown_pct']:.1f}% (need ≤{criteria['max_dd']}%) {'✅' if dd_ok else '❌'}")
            print(f"  Freq: {metrics['trades_per_month']:.1f}/mo (need ≥{criteria['min_trades_month']}) {'✅' if freq_ok else '❌'}")

            if not (pnl_ok and pf_ok and dd_ok and freq_ok):
                overall_ready = False

    print(f"\nOverall RC-Ready Status: {'✅ READY FOR PRODUCTION' if overall_ready else '❌ NEEDS OPTIMIZATION'}")

    # Save results
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    results_file = f"reports/v162_extended_pnl_scaling_{timestamp}.json"

    os.makedirs('reports', exist_ok=True)
    with open(results_file, 'w') as f:
        # Convert datetime objects for JSON serialization
        serializable_results = {}
        for asset, result in results.items():
            if result:
                serializable_result = dict(result)
                if 'extended_results' in serializable_result:
                    for risk_level, risk_result in serializable_result['extended_results'].items():
                        if 'trades' in risk_result:
                            for trade in risk_result['trades']:
                                if 'entry_time' in trade:
                                    trade['entry_time'] = trade['entry_time'].isoformat()
                                if 'exit_time' in trade:
                                    trade['exit_time'] = trade['exit_time'].isoformat()
                serializable_results[asset] = serializable_result

        json.dump(serializable_results, f, indent=2, default=str)

    print(f"\nExtended results saved to: {results_file}")
    return results


if __name__ == '__main__':
    results = run_extended_pnl_experiment()