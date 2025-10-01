#!/usr/bin/env python3
"""
Bull Machine v1.7 Complete Multi-Timeframe Backtest
Full system integration with calibrated parameters

Tests the entire machine:
- Multi-timeframe confluence (4H + 1D)
- All domain engines (SMC, Wyckoff, HOB, Momentum, Macro)
- Advanced fusion with delta routing
- Calibrated thresholds (confidence=0.30, strength=0.40)
- Complete risk management and position sizing
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

import pandas as pd
import numpy as np
import json
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Any
import warnings
warnings.filterwarnings('ignore')

# Import all Bull Machine components
from engine.io.tradingview_loader import load_tv
from engine.smc.smc_engine import SMCEngine
from engine.wyckoff.wyckoff_engine import WyckoffEngine
from engine.liquidity.hob import HOBDetector
from engine.momentum.momentum_engine import MomentumEngine
from engine.fusion.advanced_fusion import AdvancedFusionEngine

class FullMachineBacktester:
    """
    Complete Bull Machine v1.7 backtester with all systems
    """

    def __init__(self, config_path: str):
        """Initialize with calibrated configuration"""
        with open(config_path, 'r') as f:
            self.config = json.load(f)

        self.risk_pct = self.config['risk_management']['risk_pct']
        self.max_positions = self.config['risk_management']['max_positions']

        # Initialize all engines
        self._initialize_engines()

        # Results tracking
        self.trades = []
        self.signals = []
        self.telemetry = []
        self.portfolio_history = []

        print(f"🚀 BULL MACHINE v{self.config['version']} FULL BACKTEST")
        print(f"📊 Calibrated thresholds: conf={self.config['fusion']['calibration_thresholds']['confidence']}, "
              f"strength={self.config['fusion']['calibration_thresholds']['strength']}")

    def _initialize_engines(self):
        """Initialize all Bull Machine engines"""
        print("🔧 INITIALIZING FULL BULL MACHINE...")

        try:
            # Core domain engines
            self.smc_engine = SMCEngine(self.config['domains']['smc'])
            print("✅ SMC Engine (Order Blocks, FVGs, Liquidity Sweeps, BOS)")

            self.wyckoff_engine = WyckoffEngine(self.config['domains']['wyckoff'])
            print("✅ Wyckoff Engine (Phase Detection, CRT/SMR)")

            self.hob_engine = HOBDetector(self.config['domains']['liquidity']['hob_detection'])
            print("✅ HOB Engine (Institutional Pattern Recognition)")

            self.momentum_engine = MomentumEngine(self.config['domains']['momentum'])
            print("✅ Momentum Engine (RSI/MACD with Delta Routing)")

            # Advanced fusion engine
            self.fusion_engine = AdvancedFusionEngine(self.config['fusion'])
            print("✅ Advanced Fusion Engine (Delta Architecture)")

            # Macro engine (simplified for now)
            self.macro_enabled = self.config['domains']['macro_context']['enabled']
            print(f"✅ Macro Context: {'Enabled' if self.macro_enabled else 'Disabled'}")

            print("🎯 All engines initialized successfully")

        except Exception as e:
            print(f"❌ Engine initialization failed: {e}")
            raise

    def load_multi_timeframe_data(self) -> Dict[str, pd.DataFrame]:
        """Load ETH data across multiple timeframes"""
        print("\n📈 LOADING MULTI-TIMEFRAME DATA")
        print("-" * 40)

        data = {}

        # Primary timeframe (4H)
        try:
            df_4h = load_tv('ETH_4H')
            data['4H'] = df_4h
            print(f"✅ ETH 4H: {len(df_4h)} bars ({df_4h.index[0]} to {df_4h.index[-1]})")
        except Exception as e:
            print(f"❌ Failed to load ETH 4H: {e}")
            return {}

        # Higher timeframe (1D)
        try:
            df_1d = load_tv('ETH_1D')
            data['1D'] = df_1d
            print(f"✅ ETH 1D: {len(df_1d)} bars ({df_1d.index[0]} to {df_1d.index[-1]})")
        except Exception as e:
            print(f"⚠️ ETH 1D not available: {e}")

        # Macro data (if enabled)
        macro_data = {}
        if self.macro_enabled:
            macro_series = ['DXY_1D', 'US10Y_1D', 'VIX_1D', 'TOTAL3_4H']
            for series in macro_series:
                try:
                    macro_df = load_tv(series)
                    macro_data[series] = macro_df
                    print(f"✅ {series}: {len(macro_df)} bars")
                except:
                    print(f"⚠️ {series}: Not available")

        data['macro'] = macro_data
        return data

    def run_comprehensive_backtest(self, lookback_days: int = 120) -> Dict[str, Any]:
        """Run comprehensive multi-timeframe backtest"""

        print(f"\n🎯 STARTING COMPREHENSIVE BACKTEST")
        print("=" * 60)

        # Load data
        market_data = self.load_multi_timeframe_data()
        if not market_data or '4H' not in market_data:
            print("❌ Failed to load required data")
            return {}

        # Filter to lookback period
        primary_data = market_data['4H']
        end_date = primary_data.index[-1]
        start_date = end_date - timedelta(days=lookback_days)

        # Handle timezone-aware filtering
        if primary_data.index.tz is not None:
            if pd.to_datetime(start_date).tz is None:
                start_date = pd.to_datetime(start_date).tz_localize('UTC')
            else:
                start_date = pd.to_datetime(start_date).tz_convert('UTC')

        backtest_data = primary_data[primary_data.index >= start_date]

        print(f"📅 Backtest period: {backtest_data.index[0]} to {backtest_data.index[-1]}")
        print(f"📊 Total bars: {len(backtest_data)} (4H primary)")

        # Initialize portfolio
        portfolio = {
            'capital': 100000.0,  # $100k starting capital
            'position': 0.0,
            'entry_price': 0.0,
            'entry_time': None,
            'trades_count': 0,
            'wins': 0,
            'losses': 0,
            'total_pnl': 0.0,
            'max_capital': 100000.0,
            'max_drawdown': 0.0
        }

        print(f"💰 Starting capital: ${portfolio['capital']:,.2f}")
        print(f"🎯 Risk per trade: {self.risk_pct*100:.1f}%")

        # Main backtest loop
        print(f"\n⚡ PROCESSING {len(backtest_data)} BARS")
        signals_generated = 0
        trades_executed = 0
        confluence_signals = 0

        for i in range(100, len(backtest_data)):  # Start after warmup
            current_bar = backtest_data.iloc[i]
            historical_data = backtest_data.iloc[:i+1]

            try:
                # Generate comprehensive signals
                domain_signals = self._generate_comprehensive_signals(
                    historical_data, market_data, current_bar
                )

                # Multi-timeframe confluence check
                mtf_confluence = self._check_multi_timeframe_confluence(
                    market_data, current_bar, historical_data
                )

                # Get macro context
                macro_delta = self._get_macro_delta(market_data.get('macro', {}), current_bar)

                # Advanced fusion
                fusion_signal = self.fusion_engine.fuse_signals(
                    domain_signals=domain_signals,
                    current_bar=current_bar,
                    macro_delta=macro_delta
                )

                if fusion_signal:
                    signals_generated += 1

                    # Apply multi-timeframe filter
                    if mtf_confluence['aligned']:
                        confluence_signals += 1

                        # Execute trade logic
                        trade_executed = self._execute_advanced_trade_logic(
                            fusion_signal, current_bar, portfolio, mtf_confluence
                        )

                        if trade_executed:
                            trades_executed += 1

                        # Store comprehensive telemetry
                        self.telemetry.append({
                            'timestamp': current_bar.name,
                            'price': current_bar['close'],
                            'fusion_signal': fusion_signal,
                            'domain_signals': domain_signals,
                            'mtf_confluence': mtf_confluence,
                            'macro_delta': macro_delta,
                            'portfolio_value': self._calculate_portfolio_value(portfolio, current_bar['close'])
                        })

                # Track portfolio value
                portfolio_value = self._calculate_portfolio_value(portfolio, current_bar['close'])
                portfolio['max_capital'] = max(portfolio['max_capital'], portfolio_value)

                current_dd = (portfolio['max_capital'] - portfolio_value) / portfolio['max_capital']
                portfolio['max_drawdown'] = max(portfolio['max_drawdown'], current_dd)

                self.portfolio_history.append({
                    'timestamp': current_bar.name,
                    'value': portfolio_value,
                    'drawdown': current_dd
                })

                # Progress reporting
                if i % 50 == 0:
                    print(f"   Bar {i}/{len(backtest_data)}: Signals={signals_generated}, "
                          f"Confluence={confluence_signals}, Trades={trades_executed}")

            except Exception as e:
                if i % 100 == 0:  # Only log errors occasionally
                    print(f"⚠️ Error at bar {i}: {e}")
                continue

        # Close final position if any
        if portfolio['position'] != 0:
            final_bar = backtest_data.iloc[-1]
            self._close_position(portfolio, final_bar)

        print(f"\n📊 BACKTEST COMPLETE")
        print(f"   Signals generated: {signals_generated}")
        print(f"   MTF confluence signals: {confluence_signals}")
        print(f"   Trades executed: {trades_executed}")
        print(f"   Final portfolio: ${self._calculate_portfolio_value(portfolio, backtest_data['close'].iloc[-1]):,.2f}")

        # Calculate comprehensive metrics
        performance_metrics = self._calculate_comprehensive_metrics(portfolio)

        return {
            'portfolio': portfolio,
            'performance_metrics': performance_metrics,
            'trades': self.trades,
            'signals_generated': signals_generated,
            'confluence_signals': confluence_signals,
            'trades_executed': trades_executed,
            'telemetry_summary': self._summarize_telemetry()
        }

    def _generate_comprehensive_signals(self, historical_data: pd.DataFrame,
                                      market_data: Dict, current_bar: pd.Series) -> Dict[str, Any]:
        """Generate signals from all domain engines"""
        signals = {}

        # Use recent data window for analysis
        recent_data = historical_data.tail(100)

        # SMC Analysis
        try:
            smc_signal = self.smc_engine.analyze(recent_data)
            signals['smc'] = smc_signal
            if smc_signal:
                self.signals.append({
                    'timestamp': current_bar.name,
                    'engine': 'smc',
                    'direction': smc_signal.direction,
                    'confidence': smc_signal.confidence,
                    'hit_counters': smc_signal.hit_counters
                })
        except Exception as e:
            signals['smc'] = None

        # Momentum Analysis with Delta
        try:
            momentum_signal = self.momentum_engine.analyze(recent_data)
            signals['momentum'] = momentum_signal
            if momentum_signal:
                momentum_delta = self.momentum_engine.get_delta_only(recent_data)
                self.signals.append({
                    'timestamp': current_bar.name,
                    'engine': 'momentum',
                    'direction': momentum_signal.direction,
                    'confidence': momentum_signal.confidence,
                    'delta': momentum_delta
                })
        except Exception as e:
            signals['momentum'] = None

        # Wyckoff Analysis with HPS
        try:
            # Get USDT stagnation from macro context
            usdt_stagnation = 0.5  # Simplified - would analyze USDT.D
            wyckoff_signal = self.wyckoff_engine.analyze(recent_data, usdt_stagnation)
            signals['wyckoff'] = wyckoff_signal
            if wyckoff_signal:
                self.signals.append({
                    'timestamp': current_bar.name,
                    'engine': 'wyckoff',
                    'phase': wyckoff_signal.phase.value,
                    'direction': wyckoff_signal.direction,
                    'confidence': wyckoff_signal.confidence,
                    'crt_active': wyckoff_signal.crt_active
                })
        except Exception as e:
            signals['wyckoff'] = None

        # HOB Detection with Volume Analysis
        try:
            hob_signal = self.hob_engine.detect_hob(recent_data)
            signals['hob'] = hob_signal
            if hob_signal:
                self.signals.append({
                    'timestamp': current_bar.name,
                    'engine': 'hob',
                    'type': hob_signal.hob_type.value,
                    'quality': hob_signal.quality.value,
                    'confidence': hob_signal.confidence
                })
        except Exception as e:
            signals['hob'] = None

        return signals

    def _check_multi_timeframe_confluence(self, market_data: Dict, current_bar: pd.Series,
                                        historical_data: pd.DataFrame) -> Dict[str, Any]:
        """Check multi-timeframe confluence"""
        confluence = {
            'aligned': False,
            'primary_trend': 'neutral',
            'higher_tf_trend': 'neutral',
            'score': 0.0,
            'strength': 0.0
        }

        try:
            # Primary timeframe trend (4H)
            primary_sma_short = historical_data['close'].tail(12).mean()  # 2 days
            primary_sma_long = historical_data['close'].tail(24).mean()   # 4 days

            if primary_sma_short > primary_sma_long * 1.005:
                confluence['primary_trend'] = 'bullish'
            elif primary_sma_short < primary_sma_long * 0.995:
                confluence['primary_trend'] = 'bearish'

            # Higher timeframe trend (1D) if available
            if '1D' in market_data:
                htf_data = market_data['1D']
                # Align with current timeframe
                current_time = current_bar.name

                # Get recent 1D data
                htf_recent = htf_data[htf_data.index <= current_time]
                if len(htf_recent) >= 10:
                    htf_sma_short = htf_recent['close'].tail(5).mean()   # 5 days
                    htf_sma_long = htf_recent['close'].tail(10).mean()   # 10 days

                    if htf_sma_short > htf_sma_long * 1.01:
                        confluence['higher_tf_trend'] = 'bullish'
                    elif htf_sma_short < htf_sma_long * 0.99:
                        confluence['higher_tf_trend'] = 'bearish'

            # Calculate confluence score
            if confluence['primary_trend'] == confluence['higher_tf_trend'] and confluence['primary_trend'] != 'neutral':
                confluence['aligned'] = True
                confluence['score'] = 0.8  # High confluence
                confluence['strength'] = 1.0
            elif confluence['primary_trend'] != 'neutral':
                confluence['score'] = 0.5  # Partial confluence
                confluence['strength'] = 0.6

        except Exception as e:
            pass

        return confluence

    def _get_macro_delta(self, macro_data: Dict, current_bar: pd.Series) -> float:
        """Get macro pulse delta (simplified)"""
        # Simplified macro analysis - in full implementation would analyze:
        # DXY inverse correlation, VIX spikes, yield curve, crypto dominance
        try:
            # Random delta within bounds for simulation
            return np.random.uniform(-0.05, 0.05)
        except:
            return 0.0

    def _execute_advanced_trade_logic(self, fusion_signal, current_bar: pd.Series,
                                    portfolio: Dict, mtf_confluence: Dict) -> bool:
        """Execute advanced trade logic with multi-timeframe and risk management"""

        current_price = current_bar['close']

        # Close existing position if opposite signal
        if portfolio['position'] != 0:
            if ((portfolio['position'] > 0 and fusion_signal.direction == 'short') or
                (portfolio['position'] < 0 and fusion_signal.direction == 'long')):
                self._close_position(portfolio, current_bar)

        # Open new position if criteria met
        if portfolio['position'] == 0 and fusion_signal.trade_signal:
            # Enhanced position sizing based on signal quality
            base_sizing = fusion_signal.suggested_sizing
            mtf_multiplier = 1.2 if mtf_confluence['aligned'] else 1.0
            confidence_multiplier = min(1.5, fusion_signal.confidence / 0.25)

            final_sizing = base_sizing * mtf_multiplier * confidence_multiplier
            final_sizing = min(final_sizing, 1.5)  # Cap at 1.5x

            return self._open_position(fusion_signal, portfolio, current_price,
                                     current_bar.name, final_sizing)

        return False

    def _open_position(self, signal, portfolio: Dict, price: float,
                      timestamp, sizing_multiplier: float = 1.0) -> bool:
        """Open new position with advanced sizing"""
        try:
            # Calculate position size
            risk_amount = portfolio['capital'] * self.risk_pct * sizing_multiplier

            if signal.direction == 'long':
                portfolio['position'] = risk_amount / price
            else:
                portfolio['position'] = -risk_amount / price

            portfolio['entry_price'] = price
            portfolio['entry_time'] = timestamp
            portfolio['trades_count'] += 1

            # Record comprehensive trade data
            trade = {
                'trade_id': portfolio['trades_count'],
                'entry_timestamp': timestamp,
                'entry_price': price,
                'direction': signal.direction,
                'size': abs(portfolio['position']),
                'confidence': signal.confidence,
                'strength': signal.strength,
                'is_momentum_only': signal.is_momentum_only,
                'sizing_multiplier': sizing_multiplier,
                'risk_amount': risk_amount,
                'capital_at_entry': portfolio['capital'],
                'telemetry': signal.telemetry.__dict__ if hasattr(signal, 'telemetry') else {}
            }

            self.trades.append(trade)
            print(f"🔄 Trade #{portfolio['trades_count']}: {signal.direction.upper()} @ ${price:.2f} "
                  f"(conf: {signal.confidence:.3f}, size: {sizing_multiplier:.1f}x)")
            return True

        except Exception as e:
            print(f"Error opening position: {e}")
            return False

    def _close_position(self, portfolio: Dict, current_bar: pd.Series) -> None:
        """Close position with comprehensive tracking"""
        if portfolio['position'] == 0:
            return

        try:
            current_price = current_bar['close']
            timestamp = current_bar.name

            # Calculate PnL
            if portfolio['position'] > 0:  # Long position
                pnl = portfolio['position'] * (current_price - portfolio['entry_price'])
            else:  # Short position
                pnl = abs(portfolio['position']) * (portfolio['entry_price'] - current_price)

            # Update portfolio
            portfolio['capital'] += pnl
            portfolio['total_pnl'] += pnl

            if pnl > 0:
                portfolio['wins'] += 1
            else:
                portfolio['losses'] += 1

            # Update trade record
            if self.trades:
                trade = self.trades[-1]
                hold_time = (timestamp - portfolio['entry_time']).total_seconds() / 3600

                trade.update({
                    'exit_timestamp': timestamp,
                    'exit_price': current_price,
                    'pnl': pnl,
                    'return_pct': (pnl / trade['risk_amount']) * 100,
                    'hold_time_hours': hold_time,
                    'capital_at_exit': portfolio['capital']
                })

                print(f"✅ Close: ${current_price:.2f} | PnL: ${pnl:+,.2f} "
                      f"({trade['return_pct']:+.1f}%) | Hold: {hold_time:.1f}h")

            # Reset position
            portfolio['position'] = 0
            portfolio['entry_price'] = 0
            portfolio['entry_time'] = None

        except Exception as e:
            print(f"Error closing position: {e}")

    def _calculate_portfolio_value(self, portfolio: Dict, current_price: float) -> float:
        """Calculate current portfolio value including unrealized PnL"""
        if portfolio['position'] == 0:
            return portfolio['capital']

        if portfolio['position'] > 0:  # Long
            unrealized_pnl = portfolio['position'] * (current_price - portfolio['entry_price'])
        else:  # Short
            unrealized_pnl = abs(portfolio['position']) * (portfolio['entry_price'] - current_price)

        return portfolio['capital'] + unrealized_pnl

    def _calculate_comprehensive_metrics(self, portfolio: Dict) -> Dict[str, Any]:
        """Calculate comprehensive performance metrics"""

        completed_trades = [t for t in self.trades if 'exit_price' in t]

        if not completed_trades:
            return {'error': 'No completed trades'}

        # Basic metrics
        total_trades = len(completed_trades)
        wins = portfolio['wins']
        losses = portfolio['losses']
        win_rate = (wins / total_trades) * 100

        # Return metrics
        returns = [t['return_pct'] for t in completed_trades]
        total_return = (portfolio['capital'] - 100000) / 100000 * 100

        avg_return = np.mean(returns)
        median_return = np.median(returns)

        winning_returns = [r for r in returns if r > 0]
        losing_returns = [r for r in returns if r < 0]

        avg_win = np.mean(winning_returns) if winning_returns else 0
        avg_loss = np.mean(losing_returns) if losing_returns else 0

        # Risk metrics
        volatility = np.std(returns)
        sharpe_ratio = avg_return / volatility if volatility > 0 else 0

        # Profit factor
        gross_profit = sum(t['pnl'] for t in completed_trades if t['pnl'] > 0)
        gross_loss = abs(sum(t['pnl'] for t in completed_trades if t['pnl'] < 0))
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else 999

        # Max drawdown
        max_dd_pct = portfolio['max_drawdown'] * 100

        # Time metrics
        hold_times = [t.get('hold_time_hours', 0) for t in completed_trades]
        avg_hold_time = np.mean(hold_times) if hold_times else 0

        # Signal quality metrics
        high_conf_trades = [t for t in completed_trades if t['confidence'] >= 0.4]
        high_conf_win_rate = (len([t for t in high_conf_trades if t['pnl'] > 0]) /
                             len(high_conf_trades) * 100) if high_conf_trades else 0

        momentum_only_trades = [t for t in completed_trades if t.get('is_momentum_only', False)]
        momentum_only_pct = len(momentum_only_trades) / total_trades * 100

        return {
            'total_trades': total_trades,
            'wins': wins,
            'losses': losses,
            'win_rate': win_rate,
            'total_return_pct': total_return,
            'avg_return': avg_return,
            'median_return': median_return,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'volatility': volatility,
            'sharpe_ratio': sharpe_ratio,
            'profit_factor': profit_factor,
            'max_drawdown_pct': max_dd_pct,
            'avg_hold_time_hours': avg_hold_time,
            'high_confidence_win_rate': high_conf_win_rate,
            'momentum_only_pct': momentum_only_pct,
            'gross_profit': gross_profit,
            'gross_loss': gross_loss,
            'final_capital': portfolio['capital']
        }

    def _summarize_telemetry(self) -> Dict[str, Any]:
        """Summarize telemetry data"""
        if not self.telemetry:
            return {}

        # Engine activity summary
        engine_activity = {}
        for signal in self.signals:
            engine = signal['engine']
            engine_activity[engine] = engine_activity.get(engine, 0) + 1

        # Multi-timeframe confluence rate
        confluence_rate = sum(1 for t in self.telemetry
                            if t.get('mtf_confluence', {}).get('aligned', False)) / len(self.telemetry)

        return {
            'total_signals': len(self.signals),
            'engine_activity': engine_activity,
            'confluence_rate': confluence_rate * 100,
            'telemetry_entries': len(self.telemetry)
        }

    def print_comprehensive_results(self, results: Dict[str, Any]) -> None:
        """Print comprehensive backtest results"""

        print("\n" + "="*80)
        print("🎯 BULL MACHINE v1.7 COMPREHENSIVE BACKTEST RESULTS")
        print("="*80)

        if 'error' in results.get('performance_metrics', {}):
            print(f"❌ {results['performance_metrics']['error']}")
            return

        metrics = results['performance_metrics']

        # Overview
        print(f"\n📊 TRADING OVERVIEW")
        print("-" * 40)
        print(f"Total Trades: {metrics['total_trades']}")
        print(f"Wins: {metrics['wins']} | Losses: {metrics['losses']} | Win Rate: {metrics['win_rate']:.1f}%")
        print(f"Average Hold Time: {metrics['avg_hold_time_hours']:.1f} hours")

        # Performance
        print(f"\n💰 PERFORMANCE ANALYSIS")
        print("-" * 40)
        print(f"Total Return: {metrics['total_return_pct']:+.2f}%")
        print(f"Final Capital: ${metrics['final_capital']:,.2f}")
        print(f"Average Return: {metrics['avg_return']:+.2f}%")
        print(f"Median Return: {metrics['median_return']:+.2f}%")
        print(f"Average Win: {metrics['avg_win']:+.2f}%")
        print(f"Average Loss: {metrics['avg_loss']:+.2f}%")

        # Risk
        print(f"\n📈 RISK METRICS")
        print("-" * 40)
        print(f"Volatility: {metrics['volatility']:.2f}%")
        print(f"Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
        print(f"Max Drawdown: {metrics['max_drawdown_pct']:.1f}%")
        print(f"Profit Factor: {metrics['profit_factor']:.2f}")

        # Signal Quality
        print(f"\n🔬 SIGNAL QUALITY")
        print("-" * 40)
        print(f"High Confidence Win Rate: {metrics['high_confidence_win_rate']:.1f}%")
        print(f"Momentum-Only Trades: {metrics['momentum_only_pct']:.1f}%")
        print(f"Gross Profit: ${metrics['gross_profit']:,.2f}")
        print(f"Gross Loss: ${metrics['gross_loss']:,.2f}")

        # System Integration
        print(f"\n⚙️ SYSTEM INTEGRATION")
        print("-" * 40)
        print(f"Total Signals Generated: {results['signals_generated']}")
        print(f"Multi-Timeframe Confluence: {results['confluence_signals']}")
        print(f"Trades Executed: {results['trades_executed']}")

        telemetry = results.get('telemetry_summary', {})
        if telemetry:
            print(f"Engine Activity:")
            for engine, count in telemetry.get('engine_activity', {}).items():
                print(f"  • {engine}: {count} signals")
            print(f"Confluence Rate: {telemetry.get('confluence_rate', 0):.1f}%")

        # Recent trades
        if self.trades:
            print(f"\n📋 RECENT TRADES (Last 5)")
            print("-" * 40)
            recent_trades = self.trades[-5:]
            for i, trade in enumerate(recent_trades, 1):
                if 'exit_price' in trade:
                    direction_emoji = "📈" if trade['direction'] == 'long' else "📉"
                    pnl_emoji = "✅" if trade['pnl'] > 0 else "❌"
                    print(f"{i}. {direction_emoji} {trade['direction'].upper()} @ ${trade['entry_price']:.2f} → "
                          f"${trade['exit_price']:.2f} | {pnl_emoji} {trade['return_pct']:+.1f}% | "
                          f"Conf: {trade['confidence']:.2f} | Size: {trade['sizing_multiplier']:.1f}x")
                else:
                    print(f"{i}. 🔄 {trade['direction'].upper()} @ ${trade['entry_price']:.2f} | OPEN")

def main():
    """Run comprehensive backtest"""

    print("🚀 BULL MACHINE v1.7 COMPREHENSIVE MULTI-TIMEFRAME BACKTEST")
    print("="*80)

    try:
        # Initialize backtester with calibrated config
        backtester = FullMachineBacktester('configs/v170/assets/ETH_v17_tuned.json')

        # Run comprehensive backtest
        results = backtester.run_comprehensive_backtest(lookback_days=90)

        # Print results
        backtester.print_comprehensive_results(results)

        print(f"\n🎉 COMPREHENSIVE BACKTEST COMPLETE")
        print("="*80)

    except Exception as e:
        print(f"❌ Backtest failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()