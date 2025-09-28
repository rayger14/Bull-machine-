#!/usr/bin/env python3
"""
SPY Multi-Timeframe Backtest with Enhanced Orderflow System
Tests the newly implemented CVD and liquidity sweep logic across different timeframes
"""

import sys
import os
import warnings
from datetime import datetime, timedelta
from typing import Dict, List, Any, Tuple
import pandas as pd
import numpy as np
import yfinance as yf

# Add bull_machine to path
sys.path.append('.')

from bull_machine.backtest.datafeed import DataFeed
from bull_machine.backtest.strategy_adapter_v14_enhanced import create_enhanced_adapter
from bull_machine.tools.backtest import simulate_trade
from bull_machine.core.config_loader import load_config
from bull_machine.modules.orderflow.lca import orderflow_lca, analyze_market_structure

warnings.filterwarnings('ignore')

class SPYBacktestSuite:
    """Comprehensive SPY backtesting suite with orderflow enhancements"""

    def __init__(self, start_date: str = "2023-01-01", end_date: str = "2024-12-01"):
        self.start_date = start_date
        self.end_date = end_date
        self.timeframes = ["1m", "5m", "15m", "1H", "4H", "1D"]
        self.results = {}
        self.spy_data = None

    def fetch_spy_data(self) -> Dict[str, pd.DataFrame]:
        """Fetch SPY data from Yahoo Finance for multiple timeframes"""
        print(f"📊 Fetching SPY data from {self.start_date} to {self.end_date}...")

        spy = yf.Ticker("SPY")

        # Fetch different intervals directly (focus on accessible timeframes)
        intervals = {
            "1H": "1h",
            "1D": "1d"
        }

        data_dict = {}

        for tf, yf_interval in intervals.items():
            try:
                print(f"  Fetching {tf} data...")
                data = spy.history(start=self.start_date, end=self.end_date, interval=yf_interval)

                if data.empty:
                    print(f"  ⚠️ No data available for {tf}")
                    continue

                # Normalize column names
                data.columns = [col.lower() for col in data.columns]
                data = data[['open', 'high', 'low', 'close', 'volume']].copy()

                # Remove timezone for compatibility
                if hasattr(data.index, 'tz_localize'):
                    data.index = data.index.tz_localize(None)

                data_dict[tf] = data
                print(f"  ✅ Fetched {len(data)} {tf} bars")

            except Exception as e:
                print(f"  ❌ Error fetching {tf} data: {e}")
                continue

        if not data_dict:
            raise ValueError("Failed to fetch any SPY data")

        # Use daily data as base for resampling if needed
        self.spy_data = data_dict.get("1D")
        print(f"📅 Date range: {list(data_dict.values())[0].index[0]} to {list(data_dict.values())[0].index[-1]}")

        return data_dict

    def resample_to_timeframe(self, data: pd.DataFrame, tf: str) -> pd.DataFrame:
        """Resample data to specific timeframe"""
        tf_map = {
            "1m": "1min",
            "5m": "5min",
            "15m": "15min",
            "30m": "30min",
            "1H": "1h",
            "4H": "4h",
            "1D": "1D"
        }

        freq = tf_map.get(tf, "1h")
        resampled = data.resample(freq).agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum'
        }).dropna()

        return resampled

    def generate_enhanced_signals(self, data: pd.DataFrame, tf: str, config: Dict) -> List[Dict]:
        """Generate trading signals using enhanced orderflow system"""
        signals = []
        lookback = min(250, len(data) // 4)  # Adaptive lookback

        print(f"  🔍 Analyzing {len(data)} bars with {lookback} bar lookback...")

        for i in range(lookback, len(data) - 20):  # Leave room for trade execution
            window = data.iloc[max(0, i-lookback):i+1].copy()

            if len(window) < 50:  # Need sufficient data
                continue

            # Get orderflow analysis
            try:
                orderflow_score = orderflow_lca(window, config)
                market_structure = analyze_market_structure(window, config)

                # Enhanced signal logic with CVD and liquidity sweeps
                current_close = window['close'].iloc[-1]
                current_high = window['high'].iloc[-1]
                current_low = window['low'].iloc[-1]

                # Basic trend filter
                sma_20 = window['close'].rolling(20).mean().iloc[-1]
                sma_50 = window['close'].rolling(50).mean().iloc[-1] if len(window) >= 50 else sma_20

                trend_bullish = current_close > sma_20 > sma_50
                trend_bearish = current_close < sma_20 < sma_50

                # Signal generation with orderflow confirmation
                bos_detected = market_structure['bos_analysis']['detected']
                bos_direction = market_structure['bos_analysis']['direction']
                intent_conviction = market_structure['intent_analysis']['conviction']
                structure_health = market_structure['structure_health']

                # Long signal criteria
                if (orderflow_score > 0.7 and
                    bos_detected and bos_direction == 'bullish' and
                    intent_conviction in ['high', 'very_high'] and
                    trend_bullish and
                    structure_health == 'strong'):

                    # Calculate stop and targets
                    atr = window['high'].subtract(window['low']).rolling(14).mean().iloc[-1]
                    stop_price = current_low - (atr * 1.5)
                    target_1 = current_close + (atr * 2.0)
                    target_2 = current_close + (atr * 4.0)

                    signal = {
                        'timestamp': window.index[-1],
                        'bar_index': i,
                        'side': 'long',
                        'entry_price': current_close,
                        'stop_price': stop_price,
                        'targets': [target_1, target_2],
                        'confidence': orderflow_score,
                        'orderflow_data': {
                            'bos_strength': market_structure['bos_analysis']['strength'],
                            'intent_conviction': intent_conviction,
                            'structure_health': structure_health,
                            'cvd_delta': market_structure['intent_analysis'].get('cvd_delta', 0),
                            'liquidity_pump': market_structure['intent_analysis'].get('liquidity_pump', False)
                        },
                        'timeframe': tf
                    }
                    signals.append(signal)

                # Short signal criteria
                elif (orderflow_score < 0.3 and
                      bos_detected and bos_direction == 'bearish' and
                      intent_conviction in ['high', 'very_high'] and
                      trend_bearish):

                    # Calculate stop and targets for short
                    atr = window['high'].subtract(window['low']).rolling(14).mean().iloc[-1]
                    stop_price = current_high + (atr * 1.5)
                    target_1 = current_close - (atr * 2.0)
                    target_2 = current_close - (atr * 4.0)

                    signal = {
                        'timestamp': window.index[-1],
                        'bar_index': i,
                        'side': 'short',
                        'entry_price': current_close,
                        'stop_price': stop_price,
                        'targets': [target_1, target_2],
                        'confidence': 1.0 - orderflow_score,  # Invert for short
                        'orderflow_data': {
                            'bos_strength': market_structure['bos_analysis']['strength'],
                            'intent_conviction': intent_conviction,
                            'structure_health': structure_health,
                            'cvd_delta': market_structure['intent_analysis'].get('cvd_delta', 0),
                            'liquidity_pump': market_structure['intent_analysis'].get('liquidity_pump', False)
                        },
                        'timeframe': tf
                    }
                    signals.append(signal)

            except Exception as e:
                # Skip problematic bars
                continue

        print(f"  📈 Generated {len(signals)} signals for {tf}")
        return signals

    def simulate_trades(self, signals: List[Dict], data: pd.DataFrame, tf: str) -> List[Dict]:
        """Simulate trade execution and outcomes"""
        trades = []

        print(f"  🎯 Simulating {len(signals)} trades for {tf}...")

        for signal in signals:
            bar_idx = signal['bar_index']

            # Get future bars for trade simulation (next 20 bars or until end)
            future_start = bar_idx + 1
            future_end = min(future_start + 20, len(data))
            future_bars = data.iloc[future_start:future_end]

            if len(future_bars) == 0:
                continue

            # Create risk plan for simulation
            risk_plan = {
                'entry': signal['entry_price'],
                'stop': signal['stop_price'],
                'tp_levels': signal['targets']
            }

            # Simulate the trade (create proper bar objects)
            try:
                # Convert DataFrame rows to objects with attributes
                class Bar:
                    def __init__(self, row):
                        self.open = row['open']
                        self.high = row['high']
                        self.low = row['low']
                        self.close = row['close']
                        self.volume = row['volume']

                entry_bar = Bar(data.iloc[bar_idx])
                future_bar_objects = [Bar(future_bars.iloc[i]) for i in range(len(future_bars))]

                trade_result = simulate_trade(
                    signal=signal,
                    risk_plan=risk_plan,
                    entry_bar=entry_bar,
                    future_bars=future_bar_objects,
                    bar_idx=bar_idx
                )

                if trade_result:
                    # Enhance with orderflow data
                    trade_result.update({
                        'timestamp': signal['timestamp'],
                        'timeframe': tf,
                        'confidence': signal['confidence'],
                        'orderflow_data': signal['orderflow_data']
                    })
                    trades.append(trade_result)

            except Exception as e:
                print(f"    ⚠️ Trade simulation error: {e}")
                continue

        print(f"  ✅ Simulated {len(trades)} completed trades for {tf}")
        return trades

    def calculate_performance_metrics(self, trades: List[Dict]) -> Dict[str, float]:
        """Calculate comprehensive performance metrics"""
        if not trades:
            return {
                'total_trades': 0,
                'total_pnl': 0.0,
                'win_rate': 0.0,
                'avg_r': 0.0,
                'profit_factor': 0.0,
                'max_consecutive_losses': 0,
                'sharpe_ratio': 0.0
            }

        # Basic metrics
        total_trades = len(trades)
        total_pnl = sum(trade['pnl'] for trade in trades)
        total_r = sum(trade['r'] for trade in trades)
        avg_r = total_r / total_trades if total_trades > 0 else 0

        # Win/loss analysis
        winning_trades = [t for t in trades if t['pnl'] > 0]
        losing_trades = [t for t in trades if t['pnl'] <= 0]

        win_rate = len(winning_trades) / total_trades if total_trades > 0 else 0

        # Profit factor
        gross_profit = sum(t['pnl'] for t in winning_trades) if winning_trades else 0
        gross_loss = abs(sum(t['pnl'] for t in losing_trades)) if losing_trades else 1
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')

        # Consecutive losses
        consecutive_losses = 0
        max_consecutive_losses = 0
        for trade in trades:
            if trade['pnl'] <= 0:
                consecutive_losses += 1
                max_consecutive_losses = max(max_consecutive_losses, consecutive_losses)
            else:
                consecutive_losses = 0

        # Sharpe ratio (simplified)
        if len(trades) > 1:
            returns = [t['r'] for t in trades]
            sharpe_ratio = np.mean(returns) / np.std(returns) if np.std(returns) > 0 else 0
        else:
            sharpe_ratio = 0

        return {
            'total_trades': total_trades,
            'total_pnl': total_pnl,
            'total_r': total_r,
            'avg_r': avg_r,
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'max_consecutive_losses': max_consecutive_losses,
            'sharpe_ratio': sharpe_ratio,
            'gross_profit': gross_profit,
            'gross_loss': gross_loss
        }

    def run_backtest_for_timeframe(self, tf: str, data_dict: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """Run complete backtest for a specific timeframe"""
        print(f"\n🚀 Running backtest for {tf} timeframe...")

        # Use pre-fetched data for this timeframe
        if tf in data_dict:
            tf_data = data_dict[tf]
            print(f"  📊 Using {len(tf_data)} {tf} bars")
        else:
            print(f"  ❌ No data available for {tf}")
            return {
                'timeframe': tf,
                'signals': [],
                'trades': [],
                'metrics': self.calculate_performance_metrics([])
            }

        # Load config with orderflow features enabled
        config = load_config()
        config['features']['orderflow_lca'] = True
        config['features']['negative_vip'] = True
        config['features']['mtf_dl2'] = True
        config['features']['six_candle_leg'] = True

        # Generate signals
        signals = self.generate_enhanced_signals(tf_data, tf, config)

        if not signals:
            print(f"  ❌ No signals generated for {tf}")
            return {
                'timeframe': tf,
                'signals': [],
                'trades': [],
                'metrics': self.calculate_performance_metrics([])
            }

        # Simulate trades
        trades = self.simulate_trades(signals, tf_data, tf)

        # Calculate metrics
        metrics = self.calculate_performance_metrics(trades)

        print(f"  📈 Results for {tf}:")
        print(f"    Signals: {len(signals)}")
        print(f"    Executed Trades: {metrics['total_trades']}")
        print(f"    Win Rate: {metrics['win_rate']:.1%}")
        print(f"    Avg R: {metrics['avg_r']:.2f}")
        print(f"    Total PnL: ${metrics['total_pnl']:.2f}")
        print(f"    Profit Factor: {metrics['profit_factor']:.2f}")

        return {
            'timeframe': tf,
            'signals': signals,
            'trades': trades,
            'metrics': metrics,
            'config': config
        }

    def run_full_suite(self) -> Dict[str, Any]:
        """Run backtest across all timeframes"""
        print("🎯 Starting SPY Multi-Timeframe Backtest Suite...")
        print("=" * 60)

        # Fetch data for all timeframes
        data_dict = self.fetch_spy_data()

        # Run backtests for each available timeframe
        all_results = {}

        for tf in data_dict.keys():
            try:
                result = self.run_backtest_for_timeframe(tf, data_dict)
                all_results[tf] = result
                self.results[tf] = result
            except Exception as e:
                print(f"❌ Error testing {tf}: {e}")
                all_results[tf] = {
                    'timeframe': tf,
                    'error': str(e),
                    'signals': [],
                    'trades': [],
                    'metrics': self.calculate_performance_metrics([])
                }

        return all_results

    def generate_summary_report(self) -> str:
        """Generate comprehensive summary report"""
        if not self.results:
            return "No backtest results available"

        report = []
        report.append("=" * 80)
        report.append("📊 SPY ENHANCED ORDERFLOW BACKTEST RESULTS")
        report.append("=" * 80)
        report.append(f"Test Period: {self.start_date} to {self.end_date}")
        report.append(f"Enhanced Features: CVD, Liquidity Sweeps, BOS Detection")
        report.append("")

        # Summary table
        report.append("TIMEFRAME PERFORMANCE SUMMARY:")
        report.append("-" * 80)
        header = f"{'TF':<6}{'Signals':<10}{'Trades':<8}{'Win%':<8}{'Avg R':<8}{'Total R':<10}{'PnL $':<12}{'PF':<8}"
        report.append(header)
        report.append("-" * 80)

        total_trades_all = 0
        total_pnl_all = 0
        total_r_all = 0

        for tf, result in self.results.items():
            if 'error' in result:
                continue

            metrics = result['metrics']
            signals_count = len(result['signals'])

            total_trades_all += metrics['total_trades']
            total_pnl_all += metrics['total_pnl']
            total_r_all += metrics.get('total_r', 0)

            row = f"{tf:<6}{signals_count:<10}{metrics['total_trades']:<8}{metrics['win_rate']:<8.1%}{metrics['avg_r']:<8.2f}{metrics.get('total_r', 0):<10.2f}${metrics['total_pnl']:<11.2f}{metrics['profit_factor']:<8.2f}"
            report.append(row)

        report.append("-" * 80)
        avg_r_all = total_r_all / total_trades_all if total_trades_all > 0 else 0
        report.append(f"{'TOTAL':<6}{'':<10}{total_trades_all:<8}{'':<8}{avg_r_all:<8.2f}{total_r_all:<10.2f}${total_pnl_all:<11.2f}{'':<8}")
        report.append("")

        # Detailed analysis
        report.append("DETAILED ANALYSIS:")
        report.append("-" * 40)

        for tf, result in self.results.items():
            if 'error' in result:
                report.append(f"\n{tf} - ERROR: {result['error']}")
                continue

            metrics = result['metrics']
            report.append(f"\n{tf} TIMEFRAME:")
            report.append(f"  Total Signals Generated: {len(result['signals'])}")
            report.append(f"  Trades Executed: {metrics['total_trades']}")
            report.append(f"  Win Rate: {metrics['win_rate']:.1%}")
            report.append(f"  Average R: {metrics['avg_r']:.2f}")
            report.append(f"  Total R: {metrics.get('total_r', 0):.2f}")
            report.append(f"  Total PnL: ${metrics['total_pnl']:.2f}")
            report.append(f"  Profit Factor: {metrics['profit_factor']:.2f}")
            report.append(f"  Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
            report.append(f"  Max Consecutive Losses: {metrics['max_consecutive_losses']}")

            # Orderflow insights
            if result['trades']:
                cvd_positive = sum(1 for t in result['trades']
                                 if t.get('orderflow_data', {}).get('cvd_delta', 0) > 0)
                liquidity_pumps = sum(1 for t in result['trades']
                                    if t.get('orderflow_data', {}).get('liquidity_pump', False))

                report.append(f"  CVD Positive Trades: {cvd_positive}/{len(result['trades'])}")
                report.append(f"  Liquidity Pump Trades: {liquidity_pumps}/{len(result['trades'])}")

        report.append("\n" + "=" * 80)
        report.append("🎯 ENHANCED ORDERFLOW SYSTEM VALIDATION COMPLETE")
        report.append("=" * 80)

        return "\n".join(report)

def main():
    """Main execution function"""
    print("🚀 SPY Enhanced Orderflow Backtest Suite")
    print("Testing CVD, Liquidity Sweeps, and BOS Detection")
    print("-" * 50)

    # Create and run backtest suite
    suite = SPYBacktestSuite(
        start_date="2023-01-01",  # Longer period for more meaningful results
        end_date="2024-09-01"
    )

    # Run full suite
    results = suite.run_full_suite()

    # Generate and display report
    report = suite.generate_summary_report()
    print("\n" + report)

    # Save detailed results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_file = f"spy_orderflow_backtest_{timestamp}.txt"

    with open(report_file, 'w') as f:
        f.write(report)
        f.write("\n\nDETAILED TRADE DATA:\n")
        f.write("=" * 50 + "\n")

        for tf, result in results.items():
            if 'trades' in result and result['trades']:
                f.write(f"\n{tf} TRADES:\n")
                for i, trade in enumerate(result['trades'], 1):
                    f.write(f"{i:3d}. {trade['timestamp']} | {trade['side']:<5} | "
                           f"R: {trade['r']:6.2f} | PnL: ${trade['pnl']:8.2f} | "
                           f"Reason: {trade['exit_reason']}\n")

    print(f"\n📄 Detailed report saved to: {report_file}")

    return results

if __name__ == "__main__":
    results = main()