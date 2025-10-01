#!/usr/bin/env python3
"""
Bull Machine v1.7 Performance Analysis
Analyze trade signals and calculate comprehensive performance metrics
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

import pandas as pd
import numpy as np
import json
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

from engine.io.tradingview_loader import load_tv
from engine.smc.smc_engine import SMCEngine
from engine.wyckoff.wyckoff_engine import WyckoffEngine
from engine.liquidity.hob import HOBDetector
from engine.momentum.momentum_engine import MomentumEngine

class PerformanceAnalyzer:
    """Analyze Bull Machine performance with detailed metrics"""

    def __init__(self, config_path: str):
        with open(config_path, 'r') as f:
            self.config = json.load(f)

        # Initialize engines
        self.smc_engine = SMCEngine(self.config['domains']['smc'])
        self.wyckoff_engine = WyckoffEngine(self.config['domains']['wyckoff'])
        self.hob_engine = HOBDetector(self.config['domains']['liquidity']['hob_detection'])
        self.momentum_engine = MomentumEngine(self.config['domains']['momentum'])

        self.trades = []
        self.signals = []

    def run_backtest_with_trades(self, df: pd.DataFrame) -> None:
        """Run backtest and track actual trade performance"""

        print("📊 RUNNING PERFORMANCE BACKTEST")
        print("=" * 50)

        # Get thresholds
        cal_mode = self.config['fusion'].get('calibration_mode', False)
        if cal_mode:
            cal_thresholds = self.config['fusion']['calibration_thresholds']
            min_confidence = cal_thresholds['confidence']
            min_strength = cal_thresholds['strength']
        else:
            min_confidence = self.config['fusion']['entry_threshold_confidence']
            min_strength = self.config['fusion']['entry_threshold_strength']

        print(f"🎮 Mode: {'CALIBRATION' if cal_mode else 'PRODUCTION'}")
        print(f"📏 Thresholds: confidence ≥ {min_confidence}, strength ≥ {min_strength}")

        # Portfolio tracking
        portfolio = {
            'capital': 100000.0,  # Start with $100k
            'position': 0.0,
            'entry_price': 0.0,
            'entry_timestamp': None,
            'trades_count': 0
        }

        risk_pct = self.config['risk_management']['risk_pct']  # 7.5%
        print(f"💰 Risk per trade: {risk_pct*100:.1f}%")

        # Process data
        for i in range(50, len(df), 5):  # Every 5th bar for performance
            window_data = df.iloc[:i+1]
            recent_data = window_data.tail(100)
            current_bar = window_data.iloc[-1]

            try:
                # Generate domain signals
                domain_signals = self._get_domain_signals(recent_data)

                # Simple fusion
                active_signals = [s for s in domain_signals.values() if s is not None]
                if not active_signals:
                    continue

                # Get directions and confidences
                directions = []
                confidences = []

                for signal in active_signals:
                    if hasattr(signal, 'direction') and hasattr(signal, 'confidence'):
                        directions.append(signal.direction)
                        confidences.append(signal.confidence)

                if not directions or not confidences:
                    continue

                # Vote on direction
                long_votes = directions.count('long')
                short_votes = directions.count('short')

                if long_votes > short_votes:
                    fusion_direction = 'long'
                    fusion_strength = long_votes / len(directions)
                elif short_votes > long_votes:
                    fusion_direction = 'short'
                    fusion_strength = short_votes / len(directions)
                else:
                    continue  # Skip neutral

                avg_confidence = np.mean(confidences)

                # Check entry criteria
                if avg_confidence >= min_confidence and fusion_strength >= min_strength:

                    # Close existing position if opposite direction
                    if portfolio['position'] != 0:
                        if ((portfolio['position'] > 0 and fusion_direction == 'short') or
                            (portfolio['position'] < 0 and fusion_direction == 'long')):
                            self._close_position(portfolio, current_bar)

                    # Open new position if no position
                    if portfolio['position'] == 0:
                        self._open_position(portfolio, current_bar, fusion_direction,
                                          avg_confidence, fusion_strength, len(active_signals), risk_pct)

            except Exception as e:
                continue

        # Close final position if any
        if portfolio['position'] != 0:
            final_bar = df.iloc[-1]
            self._close_position(portfolio, final_bar)

        print(f"\n✅ Backtest complete. Generated {len(self.trades)} trades")

    def _get_domain_signals(self, data: pd.DataFrame) -> dict:
        """Get signals from all domain engines"""
        signals = {}

        try:
            signals['smc'] = self.smc_engine.analyze(data)
        except:
            signals['smc'] = None

        try:
            signals['momentum'] = self.momentum_engine.analyze(data)
        except:
            signals['momentum'] = None

        try:
            signals['wyckoff'] = self.wyckoff_engine.analyze(data, usdt_stagnation=0.5)
        except:
            signals['wyckoff'] = None

        try:
            signals['hob'] = self.hob_engine.detect_hob(data)
        except:
            signals['hob'] = None

        return signals

    def _open_position(self, portfolio: dict, current_bar: pd.Series, direction: str,
                      confidence: float, strength: float, active_engines: int, risk_pct: float):
        """Open a new position"""

        current_price = current_bar['close']

        # Calculate position size
        position_value = portfolio['capital'] * risk_pct

        if direction == 'long':
            portfolio['position'] = position_value / current_price
        else:  # short
            portfolio['position'] = -position_value / current_price

        portfolio['entry_price'] = current_price
        portfolio['entry_timestamp'] = current_bar.name
        portfolio['trades_count'] += 1

        # Create trade record
        trade = {
            'trade_id': portfolio['trades_count'],
            'entry_timestamp': current_bar.name,
            'entry_price': current_price,
            'direction': direction,
            'size': abs(portfolio['position']),
            'confidence': confidence,
            'strength': strength,
            'active_engines': active_engines,
            'position_value': position_value,
            'capital_at_entry': portfolio['capital']
        }

        self.trades.append(trade)
        print(f"🔄 Trade #{portfolio['trades_count']}: {direction.upper()} @ ${current_price:.2f} (conf: {confidence:.3f})")

    def _close_position(self, portfolio: dict, current_bar: pd.Series):
        """Close existing position"""

        if portfolio['position'] == 0 or not self.trades:
            return

        current_price = current_bar['close']

        # Calculate PnL
        if portfolio['position'] > 0:  # Long position
            pnl = portfolio['position'] * (current_price - portfolio['entry_price'])
        else:  # Short position
            pnl = abs(portfolio['position']) * (portfolio['entry_price'] - current_price)

        # Update portfolio
        portfolio['capital'] += pnl

        # Update trade record
        trade = self.trades[-1]
        trade.update({
            'exit_timestamp': current_bar.name,
            'exit_price': current_price,
            'pnl': pnl,
            'return_pct': (pnl / trade['position_value']) * 100,
            'hold_time_hours': (current_bar.name - portfolio['entry_timestamp']).total_seconds() / 3600,
            'capital_at_exit': portfolio['capital']
        })

        print(f"✅ Close: ${current_price:.2f} | PnL: ${pnl:+,.2f} ({trade['return_pct']:+.1f}%)")

        # Reset position
        portfolio['position'] = 0
        portfolio['entry_price'] = 0
        portfolio['entry_timestamp'] = None

    def calculate_performance_metrics(self) -> dict:
        """Calculate comprehensive performance metrics"""

        if not self.trades:
            return {'error': 'No trades to analyze'}

        completed_trades = [t for t in self.trades if 'exit_price' in t]

        if not completed_trades:
            return {'error': 'No completed trades'}

        print(f"\n📈 CALCULATING PERFORMANCE METRICS")
        print("=" * 50)

        # Basic metrics
        total_trades = len(completed_trades)
        wins = len([t for t in completed_trades if t['pnl'] > 0])
        losses = len([t for t in completed_trades if t['pnl'] < 0])
        win_rate = (wins / total_trades) * 100

        # PnL metrics
        total_pnl = sum(t['pnl'] for t in completed_trades)
        returns = [t['return_pct'] for t in completed_trades]

        avg_return = np.mean(returns)
        avg_win = np.mean([t['return_pct'] for t in completed_trades if t['pnl'] > 0]) if wins > 0 else 0
        avg_loss = np.mean([t['return_pct'] for t in completed_trades if t['pnl'] < 0]) if losses > 0 else 0

        # Risk metrics
        volatility = np.std(returns)
        sharpe_ratio = (avg_return / volatility) if volatility > 0 else 0

        # Drawdown calculation
        running_capital = 100000  # Initial capital
        peak_capital = 100000
        max_dd = 0
        drawdowns = []

        for trade in completed_trades:
            running_capital += trade['pnl']
            if running_capital > peak_capital:
                peak_capital = running_capital

            dd = (peak_capital - running_capital) / peak_capital
            drawdowns.append(dd)
            max_dd = max(max_dd, dd)

        # Profit factor
        gross_profit = sum(t['pnl'] for t in completed_trades if t['pnl'] > 0)
        gross_loss = abs(sum(t['pnl'] for t in completed_trades if t['pnl'] < 0))
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')

        # Time metrics
        hold_times = [t['hold_time_hours'] for t in completed_trades if 'hold_time_hours' in t]
        avg_hold_time = np.mean(hold_times) if hold_times else 0

        # Engine analysis
        engine_counts = {}
        for trade in completed_trades:
            count = trade.get('active_engines', 0)
            engine_counts[count] = engine_counts.get(count, 0) + 1

        # Confidence analysis
        high_conf_trades = [t for t in completed_trades if t['confidence'] >= 0.4]
        high_conf_win_rate = (len([t for t in high_conf_trades if t['pnl'] > 0]) / len(high_conf_trades) * 100) if high_conf_trades else 0

        return {
            'total_trades': total_trades,
            'wins': wins,
            'losses': losses,
            'win_rate': win_rate,
            'total_pnl': total_pnl,
            'total_return_pct': (total_pnl / 100000) * 100,
            'avg_return': avg_return,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'volatility': volatility,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_dd * 100,
            'profit_factor': profit_factor,
            'avg_hold_time_hours': avg_hold_time,
            'engine_distribution': engine_counts,
            'high_confidence_win_rate': high_conf_win_rate,
            'gross_profit': gross_profit,
            'gross_loss': gross_loss
        }

    def print_detailed_report(self, metrics: dict):
        """Print comprehensive performance report"""

        print("\n" + "="*80)
        print("🎯 BULL MACHINE v1.7 PERFORMANCE REPORT")
        print("="*80)

        if 'error' in metrics:
            print(f"❌ {metrics['error']}")
            return

        # Overview
        print(f"\n📊 TRADING OVERVIEW")
        print("-" * 40)
        print(f"Total Trades: {metrics['total_trades']}")
        print(f"Wins: {metrics['wins']} | Losses: {metrics['losses']}")
        print(f"Win Rate: {metrics['win_rate']:.1f}%")
        print(f"Average Hold Time: {metrics['avg_hold_time_hours']:.1f} hours")

        # Returns
        print(f"\n💰 RETURN ANALYSIS")
        print("-" * 40)
        print(f"Total PnL: ${metrics['total_pnl']:,.2f}")
        print(f"Total Return: {metrics['total_return_pct']:+.1f}%")
        print(f"Average Return per Trade: {metrics['avg_return']:+.2f}%")
        print(f"Average Win: {metrics['avg_win']:+.2f}%")
        print(f"Average Loss: {metrics['avg_loss']:+.2f}%")

        # Risk
        print(f"\n📈 RISK METRICS")
        print("-" * 40)
        print(f"Volatility: {metrics['volatility']:.2f}%")
        print(f"Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
        print(f"Max Drawdown: {metrics['max_drawdown']:.1f}%")
        print(f"Profit Factor: {metrics['profit_factor']:.2f}")

        # Quality Analysis
        print(f"\n🔬 SIGNAL QUALITY")
        print("-" * 40)
        print(f"High Confidence Win Rate: {metrics['high_confidence_win_rate']:.1f}%")
        print(f"Gross Profit: ${metrics['gross_profit']:,.2f}")
        print(f"Gross Loss: ${metrics['gross_loss']:,.2f}")

        # Engine Distribution
        print(f"\n⚙️ ENGINE CONFLUENCE")
        print("-" * 40)
        for engines, count in sorted(metrics['engine_distribution'].items()):
            pct = (count / metrics['total_trades']) * 100
            print(f"{engines} engines: {count} trades ({pct:.1f}%)")

        # Recent Trades
        print(f"\n📋 RECENT TRADES (Last 5)")
        print("-" * 40)
        recent_trades = self.trades[-5:] if len(self.trades) >= 5 else self.trades

        for i, trade in enumerate(recent_trades, 1):
            if 'exit_price' in trade:
                direction_emoji = "📈" if trade['direction'] == 'long' else "📉"
                pnl_emoji = "✅" if trade['pnl'] > 0 else "❌"
                print(f"{i}. {direction_emoji} {trade['direction'].upper()} @ ${trade['entry_price']:.2f} → "
                      f"${trade['exit_price']:.2f} | {pnl_emoji} {trade['return_pct']:+.1f}% | "
                      f"Conf: {trade['confidence']:.2f}")
            else:
                print(f"{i}. 🔄 {trade['direction'].upper()} @ ${trade['entry_price']:.2f} | OPEN")

        # Regime Analysis (if enough data)
        if len(self.trades) >= 10:
            self._analyze_market_regimes()

    def _analyze_market_regimes(self):
        """Analyze performance by market conditions"""
        print(f"\n🌊 REGIME ANALYSIS")
        print("-" * 40)

        # Simple regime classification based on price trends
        completed_trades = [t for t in self.trades if 'exit_price' in t]

        if len(completed_trades) < 5:
            print("Insufficient data for regime analysis")
            return

        # Classify trades by market direction during the trade
        bullish_trades = []
        bearish_trades = []
        sideways_trades = []

        for trade in completed_trades:
            price_change = ((trade['exit_price'] - trade['entry_price']) / trade['entry_price']) * 100

            if price_change > 2:
                bullish_trades.append(trade)
            elif price_change < -2:
                bearish_trades.append(trade)
            else:
                sideways_trades.append(trade)

        # Calculate regime performance
        for regime_name, regime_trades in [("Bullish", bullish_trades), ("Bearish", bearish_trades), ("Sideways", sideways_trades)]:
            if regime_trades:
                regime_wins = len([t for t in regime_trades if t['pnl'] > 0])
                regime_win_rate = (regime_wins / len(regime_trades)) * 100
                regime_avg_return = np.mean([t['return_pct'] for t in regime_trades])

                print(f"{regime_name}: {len(regime_trades)} trades, {regime_win_rate:.1f}% win rate, {regime_avg_return:+.1f}% avg return")

def main():
    """Run performance analysis"""

    print("🚀 BULL MACHINE v1.7 PERFORMANCE ANALYSIS")
    print("="*60)

    try:
        # Load data
        df = load_tv('ETH_4H')
        test_data = df.tail(300)  # Last 300 bars for comprehensive test

        print(f"📊 Analyzing {len(test_data)} bars ({test_data.index[0]} to {test_data.index[-1]})")

        # Initialize analyzer
        analyzer = PerformanceAnalyzer('configs/v170/assets/ETH_v17_tuned.json')

        # Run backtest
        analyzer.run_backtest_with_trades(test_data)

        # Calculate metrics
        metrics = analyzer.calculate_performance_metrics()

        # Print report
        analyzer.print_detailed_report(metrics)

        print(f"\n🎉 ANALYSIS COMPLETE")
        print("="*60)

    except Exception as e:
        print(f"❌ Analysis failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()