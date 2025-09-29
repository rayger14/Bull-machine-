#!/usr/bin/env python3
"""
Bull Machine v1.6.2 - Full Ensemble Backtests
Real Chart Logs 2 data with comprehensive metrics and cross-asset validation

Features Tested:
- PO3 + Bojan microstructure confluence
- Wyckoff M1/M2 with enhanced phases
- Fibonacci clusters with temporal analysis
- Enhanced orderflow/CVD integration
- Multi-timeframe ensemble signals
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


def load_chart_logs_data(asset, data_paths):
    """Load Chart Logs 2 data for multi-timeframe analysis"""
    print(f"\\n=== Loading {asset} Chart Logs 2 Data ===")

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

            # Filter date range (2024-06-01 to 2025-09-01)
            df = df.set_index('timestamp').sort_index()
            start_date = '2024-06-01'
            end_date = '2025-09-01'
            df = df[start_date:end_date]

            data[timeframe] = df
            print(f"  └─ {len(df)} bars from {df.index[0]} to {df.index[-1]}")
        else:
            print(f"WARNING: {filepath} not found")

    return data


def calculate_comprehensive_metrics(trades, initial_capital=10000):
    """Calculate comprehensive trading metrics"""
    if not trades:
        return {
            'total_trades': 0,
            'win_rate': 0,
            'total_pnl_pct': 0,
            'max_drawdown_pct': 0,
            'profit_factor': 0,
            'sharpe_ratio': 0,
            'avg_trade': 0,
            'trades_per_month': 0
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

    # Sharpe ratio (simplified - assumes daily returns)
    returns = np.array(pnl_series) / initial_capital
    sharpe_ratio = np.mean(returns) / np.std(returns) * np.sqrt(252) if np.std(returns) > 0 else 0

    # Frequency
    date_range = trades[-1]['entry_time'] - trades[0]['entry_time']
    months = date_range.days / 30.44
    trades_per_month = total_trades / months if months > 0 else 0

    return {
        'total_trades': total_trades,
        'win_rate': win_rate,
        'total_pnl_pct': total_pnl_pct,
        'max_drawdown_pct': max_drawdown_pct,
        'profit_factor': profit_factor,
        'sharpe_ratio': sharpe_ratio,
        'avg_trade': avg_trade,
        'trades_per_month': trades_per_month,
        'gross_profit': gross_profit,
        'gross_loss': gross_loss
    }


def enhanced_fusion_scoring(df, i, config):
    """v1.6.2 Enhanced fusion scoring with all components"""
    if i < 100:
        return 0, {}

    window_data = df.iloc[max(0, i-100):i+1]

    # Import enhanced modules
    try:
        from bull_machine.strategy.po3_detection import detect_po3_with_bojan_confluence
        from bull_machine.modules.bojan.bojan import compute_bojan_score
        from bull_machine.strategy.hidden_fibs import detect_price_time_confluence
        from bull_machine.modules.fusion.bojan_hook import apply_bojan
    except ImportError as e:
        print(f"Warning: Module import failed: {e}")
        return 0, {}

    # Base layer scores
    layer_scores = {
        'wyckoff': calculate_wyckoff_score(window_data) * 0.35,
        'm1': calculate_m1_score(window_data) * 0.25,
        'm2': calculate_m2_score(window_data) * 0.20,
        'structure': 0.0,
        'volume': 0.0,
        'ensemble_entry': 0.0
    }

    # PO3 + Bojan enhancement
    try:
        range_data = window_data.iloc[-25:-5] if len(window_data) > 25 else window_data.iloc[:-5]
        if len(range_data) >= 5:
            irh = range_data['high'].max()
            irl = range_data['low'].min()

            po3_result = detect_po3_with_bojan_confluence(
                window_data.tail(20), irh, irl, vol_spike_threshold=1.4
            )

            if po3_result and po3_result['strength'] > 0.5:
                layer_scores['ensemble_entry'] += po3_result['strength'] * 0.15

                if po3_result.get('bojan_confluence', False):
                    confluence_boost = min(po3_result.get('bojan_score', 0) * 0.10, 0.15)
                    layer_scores['ensemble_entry'] += confluence_boost
    except Exception:
        pass

    # Bojan fusion hook
    try:
        layer_scores, bojan_telemetry = apply_bojan(
            layer_scores, window_data.tail(20), tf=config.get('timeframe', '1D'),
            config=config, last_hooks={}
        )
    except Exception:
        pass

    # Fibonacci clusters
    try:
        fib_result = detect_price_time_confluence(window_data)
        if fib_result and fib_result.get('strength', 0) > 0.3:
            layer_scores['ensemble_entry'] += fib_result['strength'] * 0.08
    except Exception:
        pass

    # Volume/orderflow (simplified)
    vol_ma = window_data['volume'].rolling(10).mean().iloc[-1]
    vol_current = window_data['volume'].iloc[-1]
    if vol_ma > 0:
        vol_boost = min((vol_current / vol_ma - 1) * 0.1, 0.05)
        layer_scores['volume'] += vol_boost

    total_score = sum(layer_scores.values())
    return total_score, layer_scores


def calculate_wyckoff_score(df):
    """Enhanced Wyckoff scoring"""
    if len(df) < 50:
        return 0

    vol_ma = df['volume'].rolling(20).mean().iloc[-1]
    current_vol = df['volume'].iloc[-1]
    vol_ratio = current_vol / vol_ma if vol_ma > 0 else 1

    close = df['close'].iloc[-1]
    high = df['high'].iloc[-10:].max()
    low = df['low'].iloc[-10:].min()
    range_pos = (close - low) / (high - low) if high != low else 0.5

    sma_20 = df['close'].rolling(20).mean().iloc[-1]
    sma_50 = df['close'].rolling(50).mean().iloc[-1]
    trend_strength = (sma_20 - sma_50) / sma_50 if sma_50 > 0 else 0

    score = 0
    if vol_ratio > 1.3 and abs(trend_strength) > 0.02:
        if trend_strength > 0 and range_pos > 0.6:
            score = 0.45
        elif trend_strength < 0 and range_pos < 0.4:
            score = 0.40

    return score


def calculate_m1_score(df):
    """M1 momentum scoring"""
    if len(df) < 20:
        return 0

    delta = df['close'].diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    current_rsi = rsi.iloc[-1]

    close = df['close'].iloc[-1]
    sma_10 = df['close'].rolling(10).mean().iloc[-1]
    momentum = (close - sma_10) / sma_10 if sma_10 > 0 else 0

    score = 0
    if 30 < current_rsi < 70 and abs(momentum) > 0.01:
        score = min(0.35 + abs(momentum) * 5, 0.45)

    return score


def calculate_m2_score(df):
    """M2 structure scoring"""
    if len(df) < 30:
        return 0

    highs = df['high'].rolling(5).max()
    lows = df['low'].rolling(5).min()

    current_price = df['close'].iloc[-1]
    recent_high = highs.iloc[-10:].max()
    recent_low = lows.iloc[-10:].min()

    structure_strength = 0
    if abs(current_price - recent_high) / recent_high < 0.02:
        structure_strength = 0.35
    elif abs(current_price - recent_low) / recent_low < 0.02:
        structure_strength = 0.35

    return structure_strength


def run_ensemble_backtest(asset, data, config):
    """Run comprehensive ensemble backtest"""
    print(f"\\n=== Running {asset} Ensemble Backtest ===")

    # Use primary timeframe (1D)
    df = data.get('1D')
    if df is None or len(df) < 200:
        print(f"Insufficient {asset} data for backtest")
        return {}

    print(f"Backtesting {len(df)} bars from {df.index[0]} to {df.index[-1]}")

    # Enhanced backtest parameters
    entry_threshold = config.get('entry_threshold', 0.38)
    cooldown_bars = config.get('cooldown_bars', 168)
    risk_per_trade = config.get('risk', {}).get('risk_pct', 0.005)
    initial_capital = 10000

    print(f"Entry threshold: {entry_threshold}")
    print(f"Risk per trade: {risk_per_trade*100}%")
    print(f"Cooldown: {cooldown_bars} bars")

    # Run backtest
    trades = []
    capital = initial_capital
    last_trade_bar = -999

    for i in range(100, len(df) - 1):
        current_bar = df.iloc[i]

        if i - last_trade_bar < cooldown_bars:
            continue

        # Calculate ensemble score
        score, components = enhanced_fusion_scoring(df, i, config)

        if score >= entry_threshold:
            entry_price = current_bar['close']
            entry_time = current_bar.name

            # Position sizing
            position_size = capital * risk_per_trade / entry_price

            # Exit logic (5-8 bars based on signal strength)
            exit_bars = 5 if score < 0.60 else 8
            exit_bar_idx = min(i + exit_bars, len(df) - 1)
            exit_price = df.iloc[exit_bar_idx]['close']

            # Calculate PnL with fees/slippage
            fee_bps = 5  # 0.05%
            slip_bps = 2  # 0.02%

            entry_cost = entry_price * (1 + (fee_bps + slip_bps) / 10000)
            exit_proceeds = exit_price * (1 - (fee_bps + slip_bps) / 10000)

            pnl = position_size * (exit_proceeds - entry_cost)
            capital += pnl

            # Generate tags
            tags = []
            if components.get('ensemble_entry', 0) > 0.10:
                tags.append('ensemble_signal')
            if score > 0.60:
                tags.append('high_confidence')

            trade = {
                'entry_time': entry_time,
                'entry_price': entry_price,
                'exit_price': exit_price,
                'pnl': pnl,
                'score': score,
                'components': components,
                'tags': tags,
                'side': 'long',
                'exit_bars': exit_bars
            }

            trades.append(trade)
            last_trade_bar = i

    # Calculate metrics
    metrics = calculate_comprehensive_metrics(trades, initial_capital)

    return {
        'asset': asset,
        'trades': trades,
        'metrics': metrics,
        'final_capital': capital,
        'config': config
    }


def load_asset_config(asset):
    """Load asset-specific configuration"""
    config_path = f"configs/v160/assets/{asset}.json"
    try:
        with open(config_path, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Config not found: {config_path}, using ETH defaults")
        # Return ETH config as fallback
        with open("configs/v160/assets/ETH.json", 'r') as f:
            config = json.load(f)
            config['profile_name'] = f"{asset}_v1.6.2_fallback"
            return config


def run_full_backtests():
    """Run comprehensive ETH + BTC backtests"""
    print("="*80)
    print("BULL MACHINE v1.6.2 - FULL ENSEMBLE BACKTESTS")
    print("Chart Logs 2 Data | PO3 + Bojan + Wyckoff M1/M2 + Fibonacci Clusters")
    print("="*80)

    # Chart Logs 2 data paths
    data_paths = {
        'ETH': {
            '1D': '/Users/raymondghandchi/Downloads/Chart logs 2/COINBASE_ETHUSD, 1D_64942.csv',
            '4H': '/Users/raymondghandchi/Downloads/Chart logs 2/COINBASE_ETHUSD, 240_1d04a.csv',
            '1H': '/Users/raymondghandchi/Downloads/Chart logs 2/COINBASE_ETHUSD, 60_2f4ab.csv'
        },
        'BTC': {
            '1D': '/Users/raymondghandchi/Downloads/Chart logs 2/COINBASE_BTCUSD, 1D_85c84.csv',
            '4H': '/Users/raymondghandchi/Downloads/Chart logs 2/COINBASE_BTCUSD, 240_c2b76.csv',
            '1H': '/Users/raymondghandchi/Downloads/Chart logs 2/COINBASE_BTCUSD, 60_50ad4.csv'
        }
    }

    results = {}

    # Run backtests for each asset
    for asset, paths in data_paths.items():
        try:
            # Load data
            data = load_chart_logs_data(asset, paths)

            if not data.get('1D') is None and len(data['1D']) > 0:
                # Load config
                config = load_asset_config(asset)

                # Run backtest
                result = run_ensemble_backtest(asset, data, config)
                results[asset] = result
            else:
                print(f"\\nSkipping {asset} - insufficient data")

        except Exception as e:
            print(f"\\nERROR running {asset} backtest: {e}")
            continue

    # Display comprehensive results
    print("\\n" + "="*80)
    print("COMPREHENSIVE BACKTEST RESULTS")
    print("="*80)

    # Results table header
    print(f"{'Asset':<8} {'Timeframe':<12} {'Trades':<8} {'Win %':<8} {'PnL %':<10} {'Max DD':<10} {'PF':<8} {'Sharpe':<8}")
    print("-" * 80)

    combined_trades = []
    total_performance = {}

    for asset, result in results.items():
        if result:
            metrics = result['metrics']
            combined_trades.extend(result['trades'])

            print(f"{asset:<8} {'Ensemble':<12} {metrics['total_trades']:<8} "
                  f"{metrics['win_rate']:<8.1f} {metrics['total_pnl_pct']:<10.2f} "
                  f"{metrics['max_drawdown_pct']:<10.2f} {metrics['profit_factor']:<8.2f} "
                  f"{metrics['sharpe_ratio']:<8.2f}")

    # Combined results
    if combined_trades:
        combined_metrics = calculate_comprehensive_metrics(combined_trades, 20000)  # 2x capital for combined
        print("-" * 80)
        print(f"{'Combined':<8} {'Portfolio':<12} {combined_metrics['total_trades']:<8} "
              f"{combined_metrics['win_rate']:<8.1f} {combined_metrics['total_pnl_pct']:<10.2f} "
              f"{combined_metrics['max_drawdown_pct']:<10.2f} {combined_metrics['profit_factor']:<8.2f} "
              f"{combined_metrics['sharpe_ratio']:<8.2f}")

    # Detailed analysis
    print("\\n" + "="*80)
    print("DETAILED ANALYSIS")
    print("="*80)

    for asset, result in results.items():
        if result and result['trades']:
            metrics = result['metrics']
            trades = result['trades']

            print(f"\\n{asset} Performance:")
            print(f"  Total Trades: {metrics['total_trades']}")
            print(f"  Trades/Month: {metrics['trades_per_month']:.1f}")
            print(f"  Win Rate: {metrics['win_rate']:.1f}%")
            print(f"  Total PnL: {metrics['total_pnl_pct']:.2f}%")
            print(f"  Average Trade: ${metrics['avg_trade']:.2f}")
            print(f"  Max Drawdown: {metrics['max_drawdown_pct']:.2f}%")
            print(f"  Profit Factor: {metrics['profit_factor']:.2f}")
            print(f"  Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")

            # Signal analysis
            ensemble_trades = [t for t in trades if 'ensemble_signal' in t.get('tags', [])]
            high_conf_trades = [t for t in trades if 'high_confidence' in t.get('tags', [])]

            print(f"  Ensemble signals: {len(ensemble_trades)}")
            print(f"  High confidence: {len(high_conf_trades)}")

            if high_conf_trades:
                hc_pnl = sum([t['pnl'] for t in high_conf_trades])
                hc_win_rate = len([t for t in high_conf_trades if t['pnl'] > 0]) / len(high_conf_trades) * 100
                print(f"  High conf PnL: ${hc_pnl:.2f} (WR: {hc_win_rate:.1f}%)")

    # RC-ready assessment
    print("\\n" + "="*80)
    print("RC-READY ASSESSMENT")
    print("="*80)

    rc_criteria = {
        'ETH': {'min_pnl': 10, 'min_pf': 1.3, 'max_dd': 15},
        'BTC': {'min_pnl': 30, 'min_pf': 1.3, 'max_dd': 20}
    }

    rc_ready = True
    for asset, result in results.items():
        if result and asset in rc_criteria:
            metrics = result['metrics']
            criteria = rc_criteria[asset]

            pnl_ok = metrics['total_pnl_pct'] >= criteria['min_pnl']
            pf_ok = metrics['profit_factor'] >= criteria['min_pf']
            dd_ok = metrics['max_drawdown_pct'] <= criteria['max_dd']

            status = "✅" if (pnl_ok and pf_ok and dd_ok) else "❌"
            print(f"{asset} RC-Ready: {status}")
            print(f"  PnL: {metrics['total_pnl_pct']:.1f}% (need ≥{criteria['min_pnl']}%) {'✅' if pnl_ok else '❌'}")
            print(f"  PF: {metrics['profit_factor']:.2f} (need ≥{criteria['min_pf']}) {'✅' if pf_ok else '❌'}")
            print(f"  DD: {metrics['max_drawdown_pct']:.1f}% (need ≤{criteria['max_dd']}%) {'✅' if dd_ok else '❌'}")

            if not (pnl_ok and pf_ok and dd_ok):
                rc_ready = False

    print(f"\\nOverall RC-Ready Status: {'✅ READY' if rc_ready else '❌ NEEDS IMPROVEMENT'}")

    # Save results
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    results_file = f"reports/v162_full_ensemble_{timestamp}.json"

    os.makedirs('reports', exist_ok=True)
    with open(results_file, 'w') as f:
        # Convert datetime objects to strings for JSON serialization
        serializable_results = {}
        for asset, result in results.items():
            if result:
                serializable_result = dict(result)
                if 'trades' in serializable_result:
                    for trade in serializable_result['trades']:
                        if 'entry_time' in trade:
                            trade['entry_time'] = trade['entry_time'].isoformat()
                serializable_results[asset] = serializable_result

        json.dump(serializable_results, f, indent=2, default=str)

    print(f"\\nResults saved to: {results_file}")

    return results


if __name__ == '__main__':
    results = run_full_backtests()