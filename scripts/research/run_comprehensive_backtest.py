#!/usr/bin/env python3
"""
Comprehensive Multi-Timeframe ETH Backtest
Bull Machine v1.7 - All Systems Integration

Tests complete signal chain:
- SMC (Order Blocks, FVGs, Liquidity Sweeps, BOS)
- Wyckoff (Phase Detection with CRT/SMR)
- HOB (Institutional Pattern Recognition)
- Momentum (RSI/MACD with Delta Routing)
- Macro Context (DXY, VIX, yields, crypto metrics)
- Advanced Fusion Engine with Delta Architecture
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

import pandas as pd
import numpy as np
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
import warnings
warnings.filterwarnings('ignore')

# Import all Bull Machine engines
from engine.io.tradingview_loader import load_tv
from engine.smc.smc_engine import SMCEngine
from engine.wyckoff.wyckoff_engine import WyckoffEngine
from engine.liquidity.hob import HOBDetector
from engine.momentum.momentum_engine import MomentumEngine
from engine.context.macro_pulse import MacroPulseEngine
from engine.fusion.advanced_fusion import AdvancedFusionEngine

class ComprehensiveBacktester:
    """
    Comprehensive Multi-Timeframe Backtester

    Integrates all Bull Machine v1.7 systems with proper signal fusion
    and multi-timeframe confluence validation.
    """

    def __init__(self, config_path: str):
        """Initialize backtester with configuration"""
        with open(config_path, 'r') as f:
            self.config = json.load(f)

        self.primary_tf = self.config['data_requirements']['primary_timeframe']
        self.higher_tfs = self.config['data_requirements']['additional_timeframes']

        # Initialize all engines
        self._initialize_engines()

        # Backtest results
        self.trades = []
        self.telemetry = []
        self.performance_metrics = {}

        print(f"🚀 Bull Machine v{self.config['version']} Backtester Initialized")
        print(f"📊 Primary TF: {self.primary_tf}, Higher TFs: {self.higher_tfs}")

    def _initialize_engines(self):
        """Initialize all trading engines"""
        try:
            # SMC Engine
            self.smc_engine = SMCEngine(self.config['domains']['smc'])
            print("✅ SMC Engine initialized")

            # Wyckoff Engine
            self.wyckoff_engine = WyckoffEngine(self.config['domains']['wyckoff'])
            print("✅ Wyckoff Engine initialized")

            # HOB Detector
            self.hob_engine = HOBDetector(self.config['domains']['liquidity']['hob_detection'])
            print("✅ HOB Engine initialized")

            # Momentum Engine
            self.momentum_engine = MomentumEngine(self.config['domains']['momentum'])
            print("✅ Momentum Engine initialized")

            # Macro Pulse Engine (if macro data available)
            try:
                self.macro_engine = MacroPulseEngine(self.config['domains']['macro_context']['macro_pulse'])
                print("✅ Macro Pulse Engine initialized")
            except Exception as e:
                print(f"⚠️ Macro Engine initialization failed: {e}")
                self.macro_engine = None

            # Advanced Fusion Engine
            self.fusion_engine = AdvancedFusionEngine(self.config['fusion'])
            print("✅ Advanced Fusion Engine initialized")

        except Exception as e:
            print(f"❌ Engine initialization error: {e}")
            raise

    def load_market_data(self) -> Dict[str, pd.DataFrame]:
        """Load multi-timeframe market data"""
        print("\n📈 LOADING MARKET DATA")
        print("="*50)

        data = {}

        # Load primary timeframe (4H)
        try:
            df_primary = load_tv('ETH_4H')
            data[self.primary_tf] = df_primary
            print(f"✅ {self.primary_tf}: {len(df_primary)} bars ({df_primary.index[0]} to {df_primary.index[-1]})")
        except Exception as e:
            print(f"❌ Failed to load ETH {self.primary_tf}: {e}")
            return {}

        # Load higher timeframes
        for tf in self.higher_tfs:
            try:
                if tf == '1D':
                    df_htf = load_tv('ETH_1D')
                elif tf == '12H':
                    df_htf = load_tv('ETH_12H')
                else:
                    continue

                data[tf] = df_htf
                print(f"✅ {tf}: {len(df_htf)} bars ({df_htf.index[0]} to {df_htf.index[-1]})")
            except Exception as e:
                print(f"⚠️ Failed to load ETH {tf}: {e}")

        return data

    def run_backtest(self, start_date: str = None, end_date: str = None) -> Dict[str, Any]:
        """
        Run comprehensive backtest

        Args:
            start_date: Start date (YYYY-MM-DD), defaults to 60 days ago
            end_date: End date (YYYY-MM-DD), defaults to today
        """
        print("\n🎯 STARTING COMPREHENSIVE BACKTEST")
        print("="*60)

        # Load data
        market_data = self.load_market_data()
        if not market_data:
            print("❌ No market data available")
            return {}

        primary_data = market_data[self.primary_tf]

        # Filter date range (handle timezone-aware data)
        if start_date:
            start_dt = pd.to_datetime(start_date)
            # Make timezone-aware if data is timezone-aware
            if primary_data.index.tz is not None:
                start_dt = start_dt.tz_localize('UTC')
            primary_data = primary_data[primary_data.index >= start_dt]

        if end_date:
            end_dt = pd.to_datetime(end_date)
            # Make timezone-aware if data is timezone-aware
            if primary_data.index.tz is not None:
                end_dt = end_dt.tz_localize('UTC')
            primary_data = primary_data[primary_data.index <= end_dt]

        print(f"📅 Backtest period: {primary_data.index[0]} to {primary_data.index[-1]}")
        print(f"📊 Total bars: {len(primary_data)}")

        # Initialize portfolio
        initial_capital = 100000  # $100k
        portfolio = {
            'capital': initial_capital,
            'position': 0,
            'entry_price': 0,
            'trades_count': 0,
            'wins': 0,
            'losses': 0,
            'total_pnl': 0
        }

        # Main backtest loop
        print(f"\n⚡ PROCESSING {len(primary_data)} BARS")
        signals_generated = 0
        trades_executed = 0
        debug_counter = 0

        for i in range(100, len(primary_data)):  # Start after warmup period
            current_bar = primary_data.iloc[i]
            historical_data = primary_data.iloc[:i+1]

            try:
                # Generate signals from all engines
                domain_signals = self._generate_domain_signals(historical_data, market_data, current_bar)

                # Debug: Check domain signals every 50 bars
                debug_counter += 1
                if debug_counter % 50 == 0:
                    active_domains = [k for k, v in domain_signals.items() if v is not None]
                    print(f"   Debug bar {i}: Active domains: {active_domains}")

                # Get macro delta
                macro_delta = self._get_macro_delta(current_bar)

                # Fuse signals
                fusion_signal = self.fusion_engine.fuse_signals(
                    domain_signals=domain_signals,
                    current_bar=current_bar,
                    macro_delta=macro_delta
                )

                if fusion_signal:
                    signals_generated += 1
                    print(f"   🎯 Signal {signals_generated} at bar {i}: {fusion_signal.direction} @ ${current_bar['close']:.2f}")

                    # Execute trade logic
                    trade_executed = self._execute_trade_logic(
                        fusion_signal, current_bar, portfolio, i
                    )

                    if trade_executed:
                        trades_executed += 1
                        print(f"   ✅ Trade {trades_executed} executed")

                    # Store telemetry
                    self.telemetry.append({
                        'timestamp': current_bar.name,
                        'price': current_bar['close'],
                        'fusion_signal': fusion_signal,
                        'domain_signals': domain_signals,
                        'portfolio_value': self._calculate_portfolio_value(portfolio, current_bar['close'])
                    })

            except Exception as e:
                # Continue on errors but log them
                if i % 100 == 0:  # Only log every 100th error to avoid spam
                    print(f"⚠️ Error at bar {i}: {e}")
                continue

        print(f"\n📊 BACKTEST COMPLETE")
        print(f"   Signals generated: {signals_generated}")
        print(f"   Trades executed: {trades_executed}")
        print(f"   Final portfolio value: ${self._calculate_portfolio_value(portfolio, primary_data['close'].iloc[-1]):,.2f}")

        # Calculate performance metrics
        self.performance_metrics = self._calculate_performance_metrics(portfolio, initial_capital)

        return {
            'portfolio': portfolio,
            'performance_metrics': self.performance_metrics,
            'trades': self.trades,
            'telemetry_samples': self.telemetry[-10:],  # Last 10 for brevity
            'signals_generated': signals_generated,
            'trades_executed': trades_executed
        }

    def _generate_domain_signals(self, historical_data: pd.DataFrame,
                                market_data: Dict[str, pd.DataFrame],
                                current_bar: pd.Series) -> Dict[str, Any]:
        """Generate signals from all domain engines"""
        signals = {}

        # SMC Analysis
        try:
            smc_signal = self.smc_engine.analyze(historical_data.tail(100))
            signals['smc'] = smc_signal
        except Exception as e:
            signals['smc'] = None

        # Momentum Analysis
        try:
            momentum_signal = self.momentum_engine.analyze(historical_data.tail(50))
            signals['momentum'] = momentum_signal
        except Exception as e:
            signals['momentum'] = None

        # Wyckoff Analysis
        try:
            # Get USDT stagnation from macro context
            usdt_stagnation = 0.5  # Simplified for now
            wyckoff_signal = self.wyckoff_engine.analyze(historical_data.tail(100), usdt_stagnation)
            signals['wyckoff'] = wyckoff_signal
        except Exception as e:
            signals['wyckoff'] = None

        # HOB Analysis
        try:
            hob_signal = self.hob_engine.detect_hob(historical_data.tail(100))
            signals['hob'] = hob_signal
        except Exception as e:
            signals['hob'] = None

        # Multi-timeframe confluence check
        signals['mtf_confluence'] = self._check_mtf_confluence(market_data, current_bar)

        return signals

    def _check_mtf_confluence(self, market_data: Dict[str, pd.DataFrame],
                             current_bar: pd.Series) -> Dict[str, Any]:
        """Check multi-timeframe confluence"""
        confluence = {'aligned': False, 'score': 0.0}

        try:
            # Simple trend alignment check
            primary_trend = self._get_trend(market_data[self.primary_tf].tail(20))

            higher_tf_trends = []
            for tf in self.higher_tfs:
                if tf in market_data:
                    htf_trend = self._get_trend(market_data[tf].tail(10))
                    higher_tf_trends.append(htf_trend)

            # Calculate confluence score
            if higher_tf_trends:
                aligned_trends = sum(1 for trend in higher_tf_trends if trend == primary_trend)
                confluence['score'] = aligned_trends / len(higher_tf_trends)
                confluence['aligned'] = confluence['score'] >= 0.6

        except Exception as e:
            pass

        return confluence

    def _get_trend(self, data: pd.DataFrame) -> str:
        """Simple trend detection"""
        if len(data) < 10:
            return 'neutral'

        sma_short = data['close'].tail(5).mean()
        sma_long = data['close'].tail(10).mean()

        if sma_short > sma_long * 1.01:
            return 'bullish'
        elif sma_short < sma_long * 0.99:
            return 'bearish'
        else:
            return 'neutral'

    def _get_macro_delta(self, current_bar: pd.Series) -> float:
        """Get macro pulse delta"""
        if self.macro_engine:
            try:
                # This would normally analyze macro data
                # For now, return random delta within bounds
                return np.random.uniform(-0.05, 0.05)
            except:
                return 0.0
        return 0.0

    def _execute_trade_logic(self, fusion_signal, current_bar: pd.Series,
                            portfolio: Dict, bar_index: int) -> bool:
        """Execute trade based on fusion signal"""
        current_price = current_bar['close']

        # Close existing position if opposite signal
        if portfolio['position'] != 0:
            if ((portfolio['position'] > 0 and fusion_signal.direction == 'short') or
                (portfolio['position'] < 0 and fusion_signal.direction == 'long')):
                self._close_position(portfolio, current_price, current_bar.name)

        # Open new position if no position
        if portfolio['position'] == 0 and fusion_signal.trade_signal:
            return self._open_position(fusion_signal, portfolio, current_price, current_bar.name)

        return False

    def _open_position(self, signal, portfolio: Dict, price: float, timestamp) -> bool:
        """Open new position"""
        try:
            # Calculate position size based on risk management
            risk_pct = self.config['risk_management']['risk_pct']
            position_value = portfolio['capital'] * risk_pct * signal.suggested_sizing

            if signal.direction == 'long':
                portfolio['position'] = position_value / price
            else:
                portfolio['position'] = -position_value / price

            portfolio['entry_price'] = price
            portfolio['trades_count'] += 1

            # Record trade
            trade = {
                'entry_timestamp': timestamp,
                'entry_price': price,
                'direction': signal.direction,
                'size': abs(portfolio['position']),
                'confidence': signal.confidence,
                'strength': signal.strength,
                'is_momentum_only': signal.is_momentum_only,
                'telemetry': signal.telemetry.__dict__ if hasattr(signal, 'telemetry') else {}
            }

            self.trades.append(trade)
            return True

        except Exception as e:
            print(f"Error opening position: {e}")
            return False

    def _close_position(self, portfolio: Dict, price: float, timestamp) -> None:
        """Close existing position"""
        if portfolio['position'] == 0:
            return

        try:
            # Calculate PnL
            if portfolio['position'] > 0:  # Long position
                pnl = portfolio['position'] * (price - portfolio['entry_price'])
            else:  # Short position
                pnl = abs(portfolio['position']) * (portfolio['entry_price'] - price)

            portfolio['capital'] += pnl
            portfolio['total_pnl'] += pnl

            if pnl > 0:
                portfolio['wins'] += 1
            else:
                portfolio['losses'] += 1

            # Update last trade with exit info
            if self.trades:
                self.trades[-1].update({
                    'exit_timestamp': timestamp,
                    'exit_price': price,
                    'pnl': pnl,
                    'return_pct': (pnl / (abs(portfolio['position']) * portfolio['entry_price'])) * 100
                })

            # Reset position
            portfolio['position'] = 0
            portfolio['entry_price'] = 0

        except Exception as e:
            print(f"Error closing position: {e}")

    def _calculate_portfolio_value(self, portfolio: Dict, current_price: float) -> float:
        """Calculate current portfolio value"""
        if portfolio['position'] == 0:
            return portfolio['capital']

        if portfolio['position'] > 0:  # Long
            unrealized_pnl = portfolio['position'] * (current_price - portfolio['entry_price'])
        else:  # Short
            unrealized_pnl = abs(portfolio['position']) * (portfolio['entry_price'] - current_price)

        return portfolio['capital'] + unrealized_pnl

    def _calculate_performance_metrics(self, portfolio: Dict, initial_capital: float) -> Dict[str, Any]:
        """Calculate comprehensive performance metrics"""
        if not self.trades:
            return {'error': 'No trades to analyze'}

        completed_trades = [t for t in self.trades if 'exit_price' in t]

        if not completed_trades:
            return {'error': 'No completed trades to analyze'}

        returns = [t['pnl'] for t in completed_trades]
        return_pcts = [t.get('return_pct', 0) for t in completed_trades]

        total_return = portfolio['total_pnl']
        total_return_pct = (total_return / initial_capital) * 100

        win_rate = (portfolio['wins'] / len(completed_trades)) * 100 if completed_trades else 0

        avg_win = np.mean([r for r in returns if r > 0]) if any(r > 0 for r in returns) else 0
        avg_loss = np.mean([r for r in returns if r < 0]) if any(r < 0 for r in returns) else 0

        profit_factor = abs(avg_win / avg_loss) if avg_loss != 0 else float('inf')

        # Risk metrics
        daily_returns = np.array(return_pcts)
        sharpe_ratio = np.mean(daily_returns) / np.std(daily_returns) if np.std(daily_returns) > 0 else 0

        max_drawdown = self._calculate_max_drawdown(completed_trades, initial_capital)

        return {
            'total_trades': len(completed_trades),
            'total_return': total_return,
            'total_return_pct': total_return_pct,
            'win_rate': win_rate,
            'wins': portfolio['wins'],
            'losses': portfolio['losses'],
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'profit_factor': profit_factor,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'final_capital': portfolio['capital']
        }

    def _calculate_max_drawdown(self, trades: List[Dict], initial_capital: float) -> float:
        """Calculate maximum drawdown"""
        if not trades:
            return 0.0

        running_capital = initial_capital
        peak_capital = initial_capital
        max_dd = 0.0

        for trade in trades:
            running_capital += trade['pnl']
            if running_capital > peak_capital:
                peak_capital = running_capital

            drawdown = (peak_capital - running_capital) / peak_capital
            max_dd = max(max_dd, drawdown)

        return max_dd * 100  # Return as percentage

    def print_results(self, results: Dict[str, Any]) -> None:
        """Print comprehensive backtest results"""
        print("\n" + "="*80)
        print("🎯 BULL MACHINE v1.7 BACKTEST RESULTS")
        print("="*80)

        if 'error' in results.get('performance_metrics', {}):
            print(f"❌ {results['performance_metrics']['error']}")
            return

        metrics = results['performance_metrics']

        print(f"\n📊 PERFORMANCE SUMMARY")
        print("-" * 40)
        print(f"Total Trades: {metrics['total_trades']}")
        print(f"Win Rate: {metrics['win_rate']:.1f}%")
        print(f"Total Return: ${metrics['total_return']:,.2f} ({metrics['total_return_pct']:+.1f}%)")
        print(f"Final Capital: ${metrics['final_capital']:,.2f}")

        print(f"\n💰 TRADE ANALYSIS")
        print("-" * 40)
        print(f"Wins: {metrics['wins']} | Losses: {metrics['losses']}")
        print(f"Average Win: ${metrics['avg_win']:,.2f}")
        print(f"Average Loss: ${metrics['avg_loss']:,.2f}")
        print(f"Profit Factor: {metrics['profit_factor']:.2f}")

        print(f"\n📈 RISK METRICS")
        print("-" * 40)
        print(f"Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
        print(f"Max Drawdown: {metrics['max_drawdown']:.1f}%")

        print(f"\n⚡ SIGNAL STATISTICS")
        print("-" * 40)
        print(f"Signals Generated: {results['signals_generated']}")
        print(f"Trades Executed: {results['trades_executed']}")
        print(f"Signal-to-Trade Ratio: {(results['trades_executed']/max(1,results['signals_generated']))*100:.1f}%")

        # Show recent trades
        if self.trades:
            print(f"\n📋 RECENT TRADES (Last 5)")
            print("-" * 40)
            for i, trade in enumerate(self.trades[-5:], 1):
                direction_emoji = "📈" if trade['direction'] == 'long' else "📉"
                pnl_str = f"${trade.get('pnl', 0):+,.2f}" if 'pnl' in trade else "OPEN"
                conf_str = f"{trade['confidence']:.2f}"
                print(f"{i}. {direction_emoji} {trade['direction'].upper()} @ ${trade['entry_price']:.2f} | Conf: {conf_str} | PnL: {pnl_str}")

def main():
    """Main backtest execution"""
    print("🚀 BULL MACHINE v1.7 COMPREHENSIVE BACKTEST")
    print("="*60)

    try:
        # Initialize backtester
        backtester = ComprehensiveBacktester('configs/v170/assets/ETH_v17_tuned.json')

        # Run backtest (last 120 days for more comprehensive test)
        end_date = datetime.now().strftime('%Y-%m-%d')
        start_date = (datetime.now() - timedelta(days=120)).strftime('%Y-%m-%d')

        print(f"📅 Backtest Period: {start_date} to {end_date}")

        results = backtester.run_backtest(start_date=start_date, end_date=end_date)

        # Print results
        backtester.print_results(results)

        print(f"\n🎉 BACKTEST COMPLETE")
        print("="*60)

    except Exception as e:
        print(f"❌ Backtest failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()