#!/usr/bin/env python3
"""
SOL Full Backtest with Bull Machine v1.7.1
Using real SOL data from chart_logs
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import os
import sys

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.real_data_loader import RealDataLoader

class SOLBullMachineBacktest:
    """Complete Bull Machine v1.7.1 backtest for SOL."""

    def __init__(self):
        self.trades = []
        self.balance = 10000
        self.position = None
        self.engine_usage = {
            'smc': 0,
            'wyckoff': 0,
            'hob': 0,
            'momentum': 0,
            'macro': 0
        }

        # Load enhanced v1.7.1 configs
        self.config = self.load_v171_configs()

        # Initialize data loader
        self.data_loader = RealDataLoader()

    def load_v171_configs(self):
        """Load v1.7.1 enhanced configurations."""
        config = {}
        config_files = ['fusion', 'context', 'liquidity', 'risk', 'momentum']

        for file in config_files:
            try:
                with open(f'configs/v171/{file}.json', 'r') as f:
                    config[file] = json.load(f)
            except:
                # Use defaults if config not found
                config[file] = {}

        return config

    def load_sol_data(self, symbol: str, timeframe: str, start_date: str = None, end_date: str = None):
        """Load SOL data for specified timeframe."""
        cache_key = f"{symbol}_{timeframe}"

        df = self.data_loader.load_raw_data(symbol, timeframe)
        if df is None:
            return None

        # Filter by date range if provided
        if start_date:
            start_dt = pd.to_datetime(start_date)
            df = df[df.index >= start_dt]

        if end_date:
            end_dt = pd.to_datetime(end_date)
            df = df[df.index <= end_dt]

        return df

    def analyze_smc_patterns(self, data_12h, data_1d):
        """Analyze Smart Money Concepts patterns for SOL."""
        if len(data_12h) < 20 or len(data_1d) < 5:
            return None

        # Order blocks detection
        recent_high = data_12h['High'].rolling(10).max().iloc[-1]
        recent_low = data_12h['Low'].rolling(10).min().iloc[-1]
        current = data_12h['Close'].iloc[-1]

        # Breaker blocks
        if current > recent_high * 1.02:
            signal = 1
        elif current < recent_low * 0.98:
            signal = -1
        else:
            signal = 0

        # Fair value gaps
        if len(data_12h) >= 3:
            gap_up = (data_12h['Low'].iloc[-1] > data_12h['High'].iloc[-3])
            gap_down = (data_12h['High'].iloc[-1] < data_12h['Low'].iloc[-3])

            if gap_up:
                signal = max(signal, 1)
            elif gap_down:
                signal = min(signal, -1)

        if signal != 0:
            self.engine_usage['smc'] += 1
            return {'signal': signal, 'confidence': 0.7}

        return None

    def analyze_wyckoff_patterns(self, data_1d, data_1w):
        """Analyze Wyckoff patterns for SOL."""
        if len(data_1d) < 20 or data_1w is None or len(data_1w) < 4:
            return None

        # Volume analysis
        avg_volume = data_1d['Volume'].rolling(20).mean().iloc[-1]
        current_volume = data_1d['Volume'].iloc[-1]

        # Price action
        price_change = (data_1d['Close'].iloc[-1] / data_1d['Close'].iloc[-5] - 1)

        # Spring pattern
        if current_volume > avg_volume * 1.5 and price_change < -0.05:
            recent_low = data_1d['Low'].rolling(20).min().iloc[-1]
            if data_1d['Low'].iloc[-1] < recent_low * 1.01:
                self.engine_usage['wyckoff'] += 1
                return {'signal': 1, 'confidence': 0.75}

        # UTAD pattern
        if current_volume > avg_volume * 1.5 and price_change > 0.05:
            recent_high = data_1d['High'].rolling(20).max().iloc[-1]
            if data_1d['High'].iloc[-1] > recent_high * 0.99:
                self.engine_usage['wyckoff'] += 1
                return {'signal': -1, 'confidence': 0.75}

        return None

    def analyze_hob_patterns(self, data_12h, data_1d, macro_data):
        """Analyze High Odds Breakout patterns for SOL."""
        if len(data_12h) < 10 or len(data_1d) < 5:
            return None

        # Volume surge detection
        avg_volume = data_12h['Volume'].rolling(10).mean()
        std_volume = data_12h['Volume'].rolling(10).std()
        volume_z = (data_12h['Volume'].iloc[-1] - avg_volume.iloc[-1]) / (std_volume.iloc[-1] + 1e-8)

        # Enhanced requirements from v1.7.1
        min_z_long = self.config.get('liquidity', {}).get('hob_quality_factors', {}).get('volume_z_min_long', 1.3)
        min_z_short = self.config.get('liquidity', {}).get('hob_quality_factors', {}).get('volume_z_min_short', 1.6)

        # Breakout detection
        resistance = data_1d['High'].rolling(5).max().iloc[-1]
        support = data_1d['Low'].rolling(5).min().iloc[-1]
        current = data_12h['Close'].iloc[-1]

        if current > resistance * 0.995 and volume_z > min_z_long:
            self.engine_usage['hob'] += 1
            return {'signal': 1, 'confidence': 0.8}
        elif current < support * 1.005 and volume_z > min_z_short:
            self.engine_usage['hob'] += 1
            return {'signal': -1, 'confidence': 0.8}

        return None

    def analyze_momentum_patterns(self, data_1d, data_1w):
        """Analyze momentum patterns for SOL."""
        if len(data_1d) < 20 or data_1w is None or len(data_1w) < 4:
            return None

        # RSI calculation
        delta = data_1d['Close'].diff()
        gain = delta.where(delta > 0, 0).rolling(14).mean()
        loss = -delta.where(delta < 0, 0).rolling(14).mean()
        rs = gain / (loss + 1e-8)
        rsi = 100 - (100 / (1 + rs))

        # MACD
        ema12 = data_1d['Close'].ewm(span=12).mean()
        ema26 = data_1d['Close'].ewm(span=26).mean()
        macd = ema12 - ema26
        signal_line = macd.ewm(span=9).mean()

        # Generate signals
        if rsi.iloc[-1] < 30 and macd.iloc[-1] > signal_line.iloc[-1]:
            self.engine_usage['momentum'] += 1
            return {'signal': 1, 'confidence': 0.65}
        elif rsi.iloc[-1] > 70 and macd.iloc[-1] < signal_line.iloc[-1]:
            self.engine_usage['momentum'] += 1
            return {'signal': -1, 'confidence': 0.65}

        return None

    def analyze_macro_context(self, data_1d, macro_data):
        """Analyze macro context for SOL trading."""
        if len(data_1d) < 20 or macro_data is None or len(macro_data) < 5:
            return None

        # SOL trend
        sma_20 = data_1d['Close'].rolling(20).mean().iloc[-1]
        sma_50 = data_1d['Close'].rolling(50).mean().iloc[-1] if len(data_1d) >= 50 else sma_20
        current = data_1d['Close'].iloc[-1]

        # Trend alignment
        if current > sma_20 and sma_20 > sma_50:
            trend = 1
        elif current < sma_20 and sma_20 < sma_50:
            trend = -1
        else:
            trend = 0

        if trend != 0:
            # Check volatility
            if 'vix' in macro_data.columns:
                vix = macro_data['vix'].iloc[-1] if not macro_data['vix'].empty else 20
                if vix > 30:  # High volatility environment
                    confidence = 0.5
                else:
                    confidence = 0.7
            else:
                confidence = 0.6

            self.engine_usage['macro'] += 1
            return {'signal': trend, 'confidence': confidence}

        return None

    def enhanced_signal_generation(self, dataset, current_idx):
        """Generate enhanced multi-engine signals with v1.7.1 improvements."""
        # Extract data windows
        data_12h = dataset['sol_12h'][:current_idx] if dataset.get('sol_12h') is not None else None
        data_1d = dataset['sol_1d'][:current_idx] if dataset.get('sol_1d') is not None else None
        data_1w = dataset['sol_1w'][:current_idx] if dataset.get('sol_1w') is not None else None
        macro = dataset['macro'][:current_idx] if 'macro' in dataset else None

        if data_1d is None or len(data_1d) < 20:
            return None

        # Get signals from all engines
        signals = []

        # Core engines
        smc_signal = self.analyze_smc_patterns(data_12h, data_1d) if data_12h is not None else None
        wyckoff_signal = self.analyze_wyckoff_patterns(data_1d, data_1w)
        hob_signal = self.analyze_hob_patterns(data_12h, data_1d, macro) if data_12h is not None else None
        momentum_signal = self.analyze_momentum_patterns(data_1d, data_1w)
        macro_signal = self.analyze_macro_context(data_1d, macro)

        # Collect active signals
        for sig in [smc_signal, wyckoff_signal, hob_signal, momentum_signal, macro_signal]:
            if sig:
                signals.append(sig)

        if len(signals) == 0:
            return None

        # v1.7.1 Counter-trend discipline
        min_engines = self.config.get('fusion', {}).get('counter_trend_discipline', {}).get('min_engines', 3)

        # Calculate consensus
        long_signals = [s for s in signals if s['signal'] > 0]
        short_signals = [s for s in signals if s['signal'] < 0]

        # Require multiple engines for signals
        if len(long_signals) >= 2:
            avg_confidence = np.mean([s['confidence'] for s in long_signals])

            # Check for counter-trend - would need price history
            is_counter_trend = False  # Simplified for SOL
            if is_counter_trend and len(long_signals) < min_engines:
                return None

            return {
                'signal': 1,
                'confidence': avg_confidence,
                'engines': len(long_signals),
                'timestamp': data_1d.index[-1]
            }

        elif len(short_signals) >= 2:
            avg_confidence = np.mean([s['confidence'] for s in short_signals])

            # v1.7.1: No ETHBTC/TOTAL2 gates for SOL, but could add SOLBTC gates

            # Check for counter-trend
            is_counter_trend = False  # Simplified
            if is_counter_trend and len(short_signals) < min_engines:
                return None

            return {
                'signal': -1,
                'confidence': avg_confidence,
                'engines': len(short_signals),
                'timestamp': data_1d.index[-1]
            }

        return None

    def execute_trade(self, signal, price):
        """Execute trade with v1.7.1 risk management."""
        if self.position is not None:
            # Close existing position
            exit_price = price
            pnl_pct = (exit_price / self.position['entry_price'] - 1) * self.position['side'] * 100

            # Apply transaction costs
            pnl_pct -= 0.3  # 0.15% each way

            self.balance *= (1 + pnl_pct / 100)

            self.trades.append({
                'entry_time': self.position['entry_time'],
                'exit_time': signal['timestamp'],
                'side': 'long' if self.position['side'] == 1 else 'short',
                'entry_price': self.position['entry_price'],
                'exit_price': exit_price,
                'pnl_pct': pnl_pct,
                'balance': self.balance,
                'engines_used': self.position['engines']
            })

            self.position = None

        # Open new position
        if signal['signal'] != 0:
            self.position = {
                'entry_time': signal['timestamp'],
                'entry_price': price,
                'side': signal['signal'],
                'engines': signal['engines']
            }

    def run_backtest(self, start_date: str, end_date: str):
        """Run full backtest on SOL data."""
        print(f"🚀 Running SOL Bull Machine v1.7.1 Backtest")
        print(f"Period: {start_date} → {end_date}")

        # Load SOL data (12H and 1D available, 1W for longer timeframe)
        dataset = {}

        # Load 12H data (720 minutes)
        sol_12h = self.load_sol_data('COINBASE_SOLUSD', '12H', start_date, end_date)
        if sol_12h is not None:
            dataset['sol_12h'] = sol_12h
            print(f"✅ Loaded SOL 12H: {len(sol_12h)} bars")

        # Load daily data
        sol_1d = self.load_sol_data('COINBASE_SOLUSD', '1D', start_date, end_date)
        if sol_1d is None:
            print("❌ No SOL daily data available")
            return None
        dataset['sol_1d'] = sol_1d
        print(f"✅ Loaded SOL 1D: {len(sol_1d)} bars")

        # Load weekly data
        sol_1w = self.load_sol_data('COINBASE_SOLUSD', '1W', start_date, end_date)
        if sol_1w is not None:
            dataset['sol_1w'] = sol_1w
            print(f"✅ Loaded SOL 1W: {len(sol_1w)} bars")

        # Generate synthetic macro data aligned with SOL
        macro = self.data_loader.generate_macro_data(start_date, end_date, '1D')
        dataset['macro'] = macro

        # Run through each day
        total_bars = len(dataset['sol_1d'])

        for i in range(20, total_bars):  # Start after warmup
            signal = self.enhanced_signal_generation(dataset, i)

            if signal:
                current_price = dataset['sol_1d']['Close'].iloc[i]
                self.execute_trade(signal, current_price)

            # Progress indicator
            if i % 50 == 0:
                print(f"Progress: {i}/{total_bars} ({i/total_bars*100:.1f}%)")

        # Close final position if any
        if self.position is not None:
            final_signal = {
                'signal': 0,
                'timestamp': dataset['sol_1d'].index[-1]
            }
            final_price = dataset['sol_1d']['Close'].iloc[-1]
            self.execute_trade(final_signal, final_price)

        return self.generate_results(start_date, end_date, dataset['sol_1d'])

    def generate_results(self, start_date, end_date, price_data):
        """Generate comprehensive results."""
        if len(self.trades) == 0:
            return {
                'period': {
                    'start': start_date,
                    'end': end_date,
                    'days': (pd.to_datetime(end_date) - pd.to_datetime(start_date)).days
                },
                'performance': {
                    'total_trades': 0,
                    'final_balance': self.balance,
                    'total_return': 0
                }
            }

        # Calculate metrics
        winning_trades = [t for t in self.trades if t['pnl_pct'] > 0]
        losing_trades = [t for t in self.trades if t['pnl_pct'] <= 0]

        days = (pd.to_datetime(end_date) - pd.to_datetime(start_date)).days

        # Calculate max drawdown
        balances = [10000]
        for trade in self.trades:
            balances.append(trade['balance'])

        peak = balances[0]
        max_dd = 0
        for bal in balances:
            if bal > peak:
                peak = bal
            dd = (peak - bal) / peak * 100
            if dd > max_dd:
                max_dd = dd

        results = {
            'period': {
                'start': start_date,
                'end': end_date,
                'days': days,
                'years': days / 365.25
            },
            'performance': {
                'total_trades': len(self.trades),
                'winning_trades': len(winning_trades),
                'losing_trades': len(losing_trades),
                'win_rate': len(winning_trades) / len(self.trades) * 100 if self.trades else 0,
                'avg_win': np.mean([t['pnl_pct'] for t in winning_trades]) if winning_trades else 0,
                'avg_loss': np.mean([t['pnl_pct'] for t in losing_trades]) if losing_trades else 0,
                'final_balance': self.balance,
                'total_return': (self.balance / 10000 - 1) * 100,
                'max_drawdown': max_dd,
                'profit_factor': abs(sum([t['pnl_pct'] for t in winning_trades]) /
                                    sum([t['pnl_pct'] for t in losing_trades])) if losing_trades else 0
            },
            'engine_usage': self.engine_usage,
            'trades': self.trades
        }

        return results

def main():
    """Run SOL backtest with Bull Machine v1.7.1."""
    backtest = SOLBullMachineBacktest()

    # Get available date range for SOL
    sol_1d = backtest.load_sol_data('COINBASE_SOLUSD', '1D')

    if sol_1d is None or len(sol_1d) == 0:
        print("❌ No SOL data available")
        return

    # Use full available range
    start_date = sol_1d.index[0].strftime('%Y-%m-%d')
    end_date = sol_1d.index[-1].strftime('%Y-%m-%d')

    print(f"\n📊 Available SOL data range: {start_date} → {end_date}")
    print(f"Total days: {(pd.to_datetime(end_date) - pd.to_datetime(start_date)).days}")

    # Run backtest
    results = backtest.run_backtest(start_date, end_date)

    if results:
        # Save results
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f'sol_backtest_results_{timestamp}.json'

        with open(filename, 'w') as f:
            # Convert datetime objects to strings for JSON
            results_json = results.copy()
            for trade in results_json['trades']:
                trade['entry_time'] = trade['entry_time'].strftime('%Y-%m-%d %H:%M:%S')
                trade['exit_time'] = trade['exit_time'].strftime('%Y-%m-%d %H:%M:%S')

            json.dump(results_json, f, indent=2)

        # Display results
        print("\n" + "="*80)
        print("📈 SOL BULL MACHINE v1.7.1 BACKTEST RESULTS")
        print("="*80)

        perf = results['performance']
        print(f"\n💰 PERFORMANCE:")
        print(f"   Starting Balance: $10,000")
        print(f"   Final Balance: ${perf['final_balance']:,.2f}")
        print(f"   Total Return: {perf['total_return']:+.2f}%")
        print(f"   Max Drawdown: {perf['max_drawdown']:.2f}%")

        print(f"\n📊 TRADING STATS:")
        print(f"   Total Trades: {perf['total_trades']}")
        print(f"   Win Rate: {perf['win_rate']:.1f}%")
        print(f"   Avg Win: {perf['avg_win']:+.2f}%")
        print(f"   Avg Loss: {perf['avg_loss']:.2f}%")
        print(f"   Profit Factor: {perf['profit_factor']:.2f}")

        print(f"\n🔧 ENGINE USAGE:")
        for engine, count in results['engine_usage'].items():
            if count > 0:
                print(f"   {engine.upper()}: {count} signals")

        print(f"\n✅ Results saved to: {filename}")

if __name__ == "__main__":
    main()