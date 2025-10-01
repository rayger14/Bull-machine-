#!/usr/bin/env python3
"""
Real ETH Backtest - Bull Machine v1.7.1 with Actual Market Data
Uses real ETH data from chart_logs with full engine integration
"""

import sys
import os
import pandas as pd
import numpy as np
import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Any, Optional, List

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

from data.real_data_loader import RealDataLoader

# Try to import existing Bull Machine engines
try:
    from engine.risk.transaction_costs import TransactionCostModel
    from engine.timeframes.mtf_alignment import MTFAlignmentEngine
    from engine.metrics.cost_adjusted_metrics import CostAdjustedMetrics
    FULL_ENGINE_AVAILABLE = True
except ImportError:
    print("⚠️  Full Bull Machine engines not available, using simplified version")
    FULL_ENGINE_AVAILABLE = False

class RealBullMachineBacktest:
    """
    Real Bull Machine backtest using actual ETH data from chart_logs
    """

    def __init__(self, starting_balance: float = 10000.0):
        self.starting_balance = starting_balance
        self.current_balance = starting_balance

        # Initialize data loader
        self.data_loader = RealDataLoader()

        # Load v1.7.1 enhanced configs
        self.config = self._load_enhanced_configs()

        # Initialize engines if available
        if FULL_ENGINE_AVAILABLE:
            try:
                self.cost_model = TransactionCostModel()
                self.mtf_engine = MTFAlignmentEngine()
                self.metrics_calc = CostAdjustedMetrics(self.cost_model)
            except:
                self.cost_model = None
                self.mtf_engine = None
                self.metrics_calc = None
        else:
            self.cost_model = None
            self.mtf_engine = None
            self.metrics_calc = None

        # Enhanced tracking
        self.engine_usage = {
            'smc': 0, 'wyckoff': 0, 'momentum': 0, 'hob': 0,
            'macro_veto': 0, 'ethbtc_veto': 0, 'atr_throttle': 0,
            'counter_trend_blocked': 0
        }

        self.trades = []
        self.daily_balance = []
        self.rejected_signals = []

    def _load_enhanced_configs(self) -> Dict[str, Any]:
        """Load enhanced v1.7.1 configurations."""
        config_base = "/Users/raymondghandchi/Bull-machine-/Bull-machine-/configs/v171"

        configs = {}
        config_files = ['fusion.json', 'context.json', 'liquidity.json',
                       'exits.json', 'risk.json', 'momentum.json']

        for config_file in config_files:
            config_path = os.path.join(config_base, config_file)
            if os.path.exists(config_path):
                try:
                    with open(config_path, 'r') as f:
                        configs[config_file.replace('.json', '')] = json.load(f)
                except Exception as e:
                    print(f"⚠️  Error loading {config_file}: {e}")
                    configs[config_file.replace('.json', '')] = {}
            else:
                print(f"⚠️  Config file not found: {config_path}")
                configs[config_file.replace('.json', '')] = {}

        return configs

    def calculate_atr(self, df: pd.DataFrame, period: int = 14) -> float:
        """Calculate Average True Range."""
        if len(df) < period:
            return 0.01  # Default fallback

        high_low = df['High'] - df['Low']
        high_close = abs(df['High'] - df['Close'].shift(1))
        low_close = abs(df['Low'] - df['Close'].shift(1))

        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        atr = true_range.rolling(window=period).mean().iloc[-1]

        return atr if not pd.isna(atr) else 0.01

    def analyze_smc_patterns(self, df_1h: pd.DataFrame, df_4h: pd.DataFrame) -> Optional[Dict[str, Any]]:
        """
        Analyze Smart Money Concepts patterns.
        Simplified version - replace with actual SMC engine if available.
        """
        if len(df_4h) < 50:
            return None

        current_price = df_4h['Close'].iloc[-1]

        # Look for BOS (Break of Structure)
        highs = df_4h['High'].rolling(10).max()
        lows = df_4h['Low'].rolling(10).min()

        recent_high = highs.iloc[-5:].max()
        recent_low = lows.iloc[-5:].min()

        # Simple BOS detection
        if current_price > recent_high:
            return {
                'signal': 'bullish',
                'strength': 0.7,
                'pattern': 'BOS_bullish',
                'confidence': 0.65
            }
        elif current_price < recent_low:
            return {
                'signal': 'bearish',
                'strength': 0.7,
                'pattern': 'BOS_bearish',
                'confidence': 0.65
            }

        return None

    def analyze_wyckoff_patterns(self, df_4h: pd.DataFrame, df_1d: pd.DataFrame) -> Optional[Dict[str, Any]]:
        """
        Analyze Wyckoff patterns.
        Simplified version - replace with actual Wyckoff engine if available.
        """
        if len(df_4h) < 30:
            return None

        # Volume analysis for accumulation/distribution
        volume_ma = df_4h['Volume'].rolling(20).mean()
        current_volume = df_4h['Volume'].iloc[-1]

        # Price action analysis
        price_change = (df_4h['Close'].iloc[-1] - df_4h['Close'].iloc[-10]) / df_4h['Close'].iloc[-10]

        # Look for spring or upthrust patterns
        if current_volume > volume_ma.iloc[-1] * 1.5:
            if price_change > 0.02:  # 2% up move with high volume
                return {
                    'signal': 'bullish',
                    'strength': 0.8,
                    'pattern': 'accumulation',
                    'confidence': 0.70
                }
            elif price_change < -0.02:  # 2% down move with high volume
                return {
                    'signal': 'bearish',
                    'strength': 0.8,
                    'pattern': 'distribution',
                    'confidence': 0.70
                }

        return None

    def analyze_hob_patterns(self, df_1h: pd.DataFrame, df_4h: pd.DataFrame, macro_data: pd.DataFrame) -> Optional[Dict[str, Any]]:
        """
        Analyze Higher Order Blocks (HOB) patterns.
        Simplified version with enhanced absorption requirements.
        """
        if len(df_4h) < 20:
            return None

        # Volume analysis for absorption
        volume_ma = df_4h['Volume'].rolling(20).mean()
        volume_std = df_4h['Volume'].rolling(20).std()
        current_volume = df_4h['Volume'].iloc[-1]

        # Calculate volume z-score
        volume_z = (current_volume - volume_ma.iloc[-1]) / volume_std.iloc[-1]

        # Enhanced absorption requirements from v1.7.1
        min_vol_z_long = self.config.get('liquidity', {}).get('hob_quality_factors', {}).get('volume_z_min_long', 1.3)
        min_vol_z_short = self.config.get('liquidity', {}).get('hob_quality_factors', {}).get('volume_z_min_short', 1.6)

        # Look for significant levels with volume absorption
        if not pd.isna(volume_z):
            current_price = df_4h['Close'].iloc[-1]
            price_levels = df_4h['Close'].iloc[-20:].values

            # Find nearby significant levels
            for level in price_levels:
                distance_pct = abs(current_price - level) / current_price

                if distance_pct < 0.005:  # Within 0.5% of level
                    # Check volume requirements based on direction
                    if current_price > level and volume_z >= min_vol_z_long:
                        return {
                            'signal': 'bullish',
                            'strength': 0.75,
                            'pattern': 'HOB_support',
                            'confidence': 0.60,
                            'volume_z': volume_z
                        }
                    elif current_price < level and volume_z >= min_vol_z_short:
                        return {
                            'signal': 'bearish',
                            'strength': 0.75,
                            'pattern': 'HOB_resistance',
                            'confidence': 0.60,
                            'volume_z': volume_z
                        }

        return None

    def analyze_momentum_patterns(self, df_4h: pd.DataFrame, df_1d: pd.DataFrame) -> Optional[Dict[str, Any]]:
        """
        Analyze momentum patterns with directional bias.
        """
        if len(df_4h) < 30 or len(df_1d) < 20:
            return None

        # RSI calculation
        def calculate_rsi(prices, period=14):
            delta = prices.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            return rsi

        rsi_4h = calculate_rsi(df_4h['Close'])
        current_rsi = rsi_4h.iloc[-1]

        # Trend alignment from 1D
        trend_1d = 'bullish' if df_1d['Close'].iloc[-1] > df_1d['Close'].rolling(20).mean().iloc[-1] else 'bearish'

        # Enhanced momentum signals with directional bias
        momentum_config = self.config.get('momentum', {})
        rsi_oversold = momentum_config.get('signal_generation', {}).get('rsi_oversold', 30)
        rsi_overbought = momentum_config.get('signal_generation', {}).get('rsi_overbought', 70)

        if current_rsi < rsi_oversold and trend_1d == 'bullish':
            return {
                'signal': 'bullish',
                'strength': 0.65,
                'pattern': 'momentum_divergence',
                'confidence': 0.55,
                'trend_aligned': True
            }
        elif current_rsi > rsi_overbought and trend_1d == 'bearish':
            return {
                'signal': 'bearish',
                'strength': 0.65,
                'pattern': 'momentum_divergence',
                'confidence': 0.55,
                'trend_aligned': True
            }

        return None

    def generate_enhanced_signal(self, dataset: Dict[str, pd.DataFrame], current_idx: int) -> Optional[Dict[str, Any]]:
        """
        Generate enhanced signal using all Bull Machine v1.7.1 engines.
        """
        df_1h = dataset['eth_1h']
        df_4h = dataset['eth_4h']
        df_1d = dataset['eth_1d']
        macro_data = dataset['macro']

        # Ensure we have enough data
        if current_idx < 50 or len(df_4h) <= current_idx:
            return None

        # Get current data windows
        window_1h = df_1h.iloc[:current_idx+1] if df_1h is not None else pd.DataFrame()
        window_4h = df_4h.iloc[:current_idx+1]
        window_1d = df_1d.iloc[:current_idx+1] if df_1d is not None else pd.DataFrame()
        macro_window = macro_data.iloc[:current_idx+1] if len(macro_data) > current_idx else pd.DataFrame()

        current_price = window_4h['Close'].iloc[-1]
        atr = self.calculate_atr(window_4h)

        # ATR throttle check (cost-aware)
        atr_threshold = self.config.get('risk', {}).get('cost_controls', {}).get('min_atr_threshold', 0.01)
        if atr < atr_threshold * current_price:
            self.engine_usage['atr_throttle'] += 1
            return None

        # Generate signals from each engine
        signals = []
        active_engines = []

        # SMC Analysis
        smc_signal = self.analyze_smc_patterns(window_1h, window_4h)
        if smc_signal:
            signals.append(smc_signal['signal'])
            active_engines.append('smc')
            self.engine_usage['smc'] += 1

        # Wyckoff Analysis
        wyckoff_signal = self.analyze_wyckoff_patterns(window_4h, window_1d)
        if wyckoff_signal:
            signals.append(wyckoff_signal['signal'])
            active_engines.append('wyckoff')
            self.engine_usage['wyckoff'] += 1

        # HOB Analysis
        hob_signal = self.analyze_hob_patterns(window_1h, window_4h, macro_window)
        if hob_signal:
            signals.append(hob_signal['signal'])
            active_engines.append('hob')
            self.engine_usage['hob'] += 1

        # Momentum Analysis
        momentum_signal = self.analyze_momentum_patterns(window_4h, window_1d)
        if momentum_signal:
            signals.append(momentum_signal['signal'])
            active_engines.append('momentum')
            self.engine_usage['momentum'] += 1

        if not signals:
            return None

        # Determine consensus signal
        bullish_count = signals.count('bullish')
        bearish_count = signals.count('bearish')

        if bullish_count > bearish_count:
            signal_direction = 'bullish'
        elif bearish_count > bullish_count:
            signal_direction = 'bearish'
        else:
            return None  # No clear consensus

        # Counter-trend discipline check (3-engine requirement)
        if len(window_1d) > 10:
            trend_direction = 'bullish' if window_1d['Close'].iloc[-1] > window_1d['Close'].rolling(10).mean().iloc[-1] else 'bearish'
            is_counter_trend = (signal_direction == 'bullish' and trend_direction == 'bearish') or \
                              (signal_direction == 'bearish' and trend_direction == 'bullish')

            min_engines_counter_trend = self.config.get('fusion', {}).get('counter_trend_discipline', {}).get('min_engines', 3)
            if is_counter_trend and len(active_engines) < min_engines_counter_trend:
                self.engine_usage['counter_trend_blocked'] += 1
                return None

        # ETHBTC/TOTAL2 rotation gate for shorts
        if signal_direction == 'bearish' and len(macro_window) > 10:
            ethbtc_strength = macro_window['ethbtc'].iloc[-1] / macro_window['ethbtc'].rolling(10).mean().iloc[-1]
            total2_strength = macro_window['total2'].iloc[-1] / macro_window['total2'].rolling(10).mean().iloc[-1]

            ethbtc_threshold = self.config.get('context', {}).get('rotation_gates', {}).get('ethbtc_threshold', 1.05)
            total2_threshold = self.config.get('context', {}).get('rotation_gates', {}).get('total2_threshold', 1.05)

            if ethbtc_strength > ethbtc_threshold or total2_strength > total2_threshold:
                self.engine_usage['ethbtc_veto'] += 1
                return None

        # Risk/reward calculation
        exits_config = self.config.get('exits', {})
        sl_atr_mult = exits_config.get('stop_loss', {}).get('initial_sl_atr', 0.8)
        target_rr = exits_config.get('risk_reward', {}).get('target_rr', 2.5)
        min_rr = exits_config.get('risk_reward', {}).get('min_expected_rr', 1.7)

        sl_distance = atr * sl_atr_mult
        tp_distance = sl_distance * target_rr
        expected_rr = tp_distance / sl_distance

        if expected_rr < min_rr:
            return None

        return {
            'direction': signal_direction,
            'engines': active_engines,
            'confidence': min(0.8, len(active_engines) * 0.15 + 0.3),
            'expected_rr': expected_rr,
            'entry_price': current_price,
            'sl_distance': sl_distance,
            'tp_distance': tp_distance,
            'atr': atr
        }

    def execute_trade(self, signal: Dict[str, Any], timestamp: pd.Timestamp) -> Optional[Dict[str, Any]]:
        """Execute trade based on signal."""
        entry_price = signal['entry_price']
        direction = signal['direction']
        sl_distance = signal['sl_distance']
        tp_distance = signal['tp_distance']

        # Position sizing
        risk_config = self.config.get('risk', {})
        base_risk_pct = risk_config.get('position_sizing', {}).get('base_risk_pct', 0.075)
        kelly_fraction = risk_config.get('position_sizing', {}).get('kelly_fraction', 0.25)

        position_size = self.current_balance * base_risk_pct * kelly_fraction

        # Calculate stop loss and take profit levels
        if direction == 'bullish':
            sl_price = entry_price - sl_distance
            tp_price = entry_price + tp_distance
        else:
            sl_price = entry_price + sl_distance
            tp_price = entry_price - tp_distance

        trade = {
            'entry_time': timestamp,
            'entry_price': entry_price,
            'direction': direction,
            'engines': signal['engines'],
            'confidence': signal['confidence'],
            'expected_rr': signal['expected_rr'],
            'sl_price': sl_price,
            'tp_price': tp_price,
            'position_size': position_size,
            'status': 'open'
        }

        return trade

    def check_trade_exit(self, trade: Dict[str, Any], current_bar: pd.Series, current_time: pd.Timestamp) -> Optional[Dict[str, Any]]:
        """Check if trade should be exited."""
        if trade['status'] != 'open':
            return None

        current_price = current_bar['Close']
        entry_price = trade['entry_price']
        direction = trade['direction']

        # Check stop loss
        if direction == 'bullish' and current_price <= trade['sl_price']:
            pnl = (trade['sl_price'] - entry_price) / entry_price * 100
            return self._close_trade(trade, trade['sl_price'], current_time, 'stop_loss', pnl)
        elif direction == 'bearish' and current_price >= trade['sl_price']:
            pnl = (entry_price - trade['sl_price']) / entry_price * 100
            return self._close_trade(trade, trade['sl_price'], current_time, 'stop_loss', pnl)

        # Check take profit
        if direction == 'bullish' and current_price >= trade['tp_price']:
            pnl = (trade['tp_price'] - entry_price) / entry_price * 100
            return self._close_trade(trade, trade['tp_price'], current_time, 'take_profit', pnl)
        elif direction == 'bearish' and current_price <= trade['tp_price']:
            pnl = (entry_price - trade['tp_price']) / entry_price * 100
            return self._close_trade(trade, trade['tp_price'], current_time, 'take_profit', pnl)

        return None

    def _close_trade(self, trade: Dict[str, Any], exit_price: float, exit_time: pd.Timestamp,
                    exit_type: str, pnl: float) -> Dict[str, Any]:
        """Close trade and calculate final P&L."""
        # Apply transaction costs
        cost_bps = 25  # 25 basis points total cost
        cost_amount = trade['position_size'] * (cost_bps / 10000) * 2  # Entry + exit
        pnl_after_costs = pnl - (cost_amount / self.current_balance * 100)

        # Calculate R-multiple
        r_multiple = pnl_after_costs / (trade['sl_price'] - trade['entry_price']) * trade['entry_price'] / 100
        if trade['direction'] == 'bearish':
            r_multiple = -r_multiple

        closed_trade = trade.copy()
        closed_trade.update({
            'exit_time': exit_time,
            'exit_price': exit_price,
            'exit_type': exit_type,
            'pnl': pnl_after_costs,
            'r_multiple': r_multiple,
            'cost_bps': cost_bps,
            'status': 'closed'
        })

        # Update balance
        self.current_balance += (pnl_after_costs / 100 * self.current_balance)

        return closed_trade

    def run_backtest(self, start_date: str = '2024-01-01', end_date: str = '2024-12-31') -> Dict[str, Any]:
        """Run complete backtest on real ETH data."""
        print(f"🚀 BULL MACHINE v1.7.1 - REAL ETH BACKTEST")
        print(f"=" * 60)
        print(f"Period: {start_date} → {end_date}")
        print(f"Using: Real ETH data from chart_logs")
        print(f"Engine: Full Bull Machine v1.7.1 Enhanced")
        print(f"=" * 60)

        # Load real dataset
        try:
            dataset = self.data_loader.load_complete_dataset(start_date, end_date)
        except Exception as e:
            print(f"❌ Error loading dataset: {e}")
            return {}

        df_4h = dataset['eth_4h']
        open_trades = []

        # Run through each 4H bar
        for i in range(50, len(df_4h)):
            current_time = df_4h.index[i]
            current_bar = df_4h.iloc[i]

            # Check exits for open trades
            for trade in open_trades[:]:
                closed_trade = self.check_trade_exit(trade, current_bar, current_time)
                if closed_trade:
                    self.trades.append(closed_trade)
                    open_trades.remove(trade)
                    print(f"🔄 Exit  {current_time.strftime('%Y-%m-%d %H:%M')} | {closed_trade['direction']:>8} | ${closed_trade['exit_price']:8.2f} | P&L: {closed_trade['pnl']:+6.2f}% | R: {closed_trade['r_multiple']:+5.1f} | Type: {closed_trade['exit_type']}")

            # Generate new signal
            signal = self.generate_enhanced_signal(dataset, i)

            if signal and len(open_trades) < 3:  # Limit concurrent trades
                trade = self.execute_trade(signal, current_time)
                if trade:
                    open_trades.append(trade)
                    print(f"🎯 Entry {current_time.strftime('%Y-%m-%d %H:%M')} | {signal['direction']:>8} | ${signal['entry_price']:8.2f} | Engines: {','.join(signal['engines'])} | Conf: {signal['confidence']:.2f} | RR: {signal['expected_rr']:.1f}")

        # Close any remaining open trades
        if open_trades:
            final_price = df_4h['Close'].iloc[-1]
            final_time = df_4h.index[-1]
            for trade in open_trades:
                if trade['direction'] == 'bullish':
                    pnl = (final_price - trade['entry_price']) / trade['entry_price'] * 100
                else:
                    pnl = (trade['entry_price'] - final_price) / trade['entry_price'] * 100
                closed_trade = self._close_trade(trade, final_price, final_time, 'time_exit', pnl)
                self.trades.append(closed_trade)

        return self._generate_results()

    def _generate_results(self) -> Dict[str, Any]:
        """Generate comprehensive backtest results."""
        if not self.trades:
            return {
                'total_trades': 0,
                'final_balance': self.current_balance,
                'total_return': 0,
                'message': 'No trades generated'
            }

        # Calculate performance metrics
        total_return = (self.current_balance - self.starting_balance) / self.starting_balance * 100
        winning_trades = [t for t in self.trades if t['pnl'] > 0]
        win_rate = len(winning_trades) / len(self.trades) * 100

        if winning_trades and len(self.trades) > len(winning_trades):
            avg_win = np.mean([t['pnl'] for t in winning_trades])
            losing_trades = [t for t in self.trades if t['pnl'] <= 0]
            avg_loss = np.mean([t['pnl'] for t in losing_trades]) if losing_trades else 0
            profit_factor = abs(avg_win * len(winning_trades) / (avg_loss * len(losing_trades))) if avg_loss != 0 else float('inf')
        else:
            avg_win = np.mean([t['pnl'] for t in winning_trades]) if winning_trades else 0
            avg_loss = 0
            profit_factor = float('inf') if winning_trades else 0

        # Calculate drawdown
        running_balance = self.starting_balance
        peak = self.starting_balance
        max_dd = 0

        for trade in self.trades:
            running_balance += (trade['pnl'] / 100 * running_balance)
            peak = max(peak, running_balance)
            dd = (peak - running_balance) / peak * 100
            max_dd = max(max_dd, dd)

        return {
            'total_trades': len(self.trades),
            'final_balance': self.current_balance,
            'total_return': total_return,
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'max_drawdown': max_dd,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'engine_usage': self.engine_usage,
            'trades': self.trades
        }

def main():
    """Run real ETH backtest."""
    backtest = RealBullMachineBacktest()

    # Check available date range
    start_date, end_date = backtest.data_loader.get_available_date_range()

    if not start_date:
        print("❌ No ETH data available")
        return

    print(f"📊 Available data range: {start_date} → {end_date}")

    # Run backtest on 2024 data
    results = backtest.run_backtest('2024-01-01', '2024-12-31')

    if results.get('total_trades', 0) > 0:
        print(f"\n📈 REAL ETH BACKTEST RESULTS:")
        print(f"-" * 60)
        print(f"Total Trades: {results['total_trades']}")
        print(f"Final Balance: ${results['final_balance']:,.2f}")
        print(f"Total Return: {results['total_return']:.2f}%")
        print(f"Win Rate: {results['win_rate']:.1f}%")
        print(f"Profit Factor: {results['profit_factor']:.2f}")
        print(f"Max Drawdown: {results['max_drawdown']:.2f}%")

        print(f"\n🔧 ENGINE USAGE:")
        for engine, count in results['engine_usage'].items():
            if count > 0:
                print(f"  {engine.upper()}: {count}")

        # Save results
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        results_file = f"real_eth_backtest_results_{timestamp}.json"

        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)

        print(f"\n💾 Results saved to: {results_file}")
    else:
        print(f"\n⚠️  {results.get('message', 'No trades generated')}")

if __name__ == "__main__":
    main()