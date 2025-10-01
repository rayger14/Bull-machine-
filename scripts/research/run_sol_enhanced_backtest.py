#!/usr/bin/env python3
"""
Enhanced SOL Backtest with Bull Machine v1.7.1
Using real SOL data with 1H/4H/1D timeframes (matching ETH setup)
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

class EnhancedSOLBullMachine:
    """Enhanced Bull Machine v1.7.1 backtest for SOL with multi-timeframe data."""

    def __init__(self):
        self.trades = []
        self.balance = 10000
        self.position = None
        self.engine_usage = {
            'smc': 0,
            'wyckoff': 0,
            'hob': 0,
            'momentum': 0,
            'macro': 0,
            'counter_trend_blocked': 0,
            'solbtc_veto': 0
        }

        # Load enhanced v1.7.1 configs
        self.config = self.load_v171_configs()

        # Initialize data loader
        self.data_loader = RealDataLoader()

    def load_v171_configs(self):
        """Load v1.7.1 enhanced configurations."""
        config = {}
        config_files = ['fusion', 'context', 'liquidity', 'risk', 'momentum', 'exits']

        for file in config_files:
            try:
                with open(f'configs/v171/{file}.json', 'r') as f:
                    config[file] = json.load(f)
            except:
                config[file] = {}

        return config

    def analyze_smc_patterns(self, window_1h, window_4h):
        """Analyze Smart Money Concepts patterns for SOL."""
        if window_1h is None or len(window_1h) < 20 or len(window_4h) < 10:
            return None

        # Order blocks on 1H
        recent_high = window_1h['High'].rolling(20).max().iloc[-1]
        recent_low = window_1h['Low'].rolling(20).min().iloc[-1]
        current = window_1h['Close'].iloc[-1]

        # Breaker blocks
        signal = 0
        confidence = 0.6

        # Strong breakout above resistance
        if current > recent_high * 1.015:
            signal = 1
            # Check 4H confirmation
            if window_4h['Close'].iloc[-1] > window_4h['Close'].iloc[-2]:
                confidence = 0.75

        # Strong breakdown below support
        elif current < recent_low * 0.985:
            signal = -1
            # Check 4H confirmation
            if window_4h['Close'].iloc[-1] < window_4h['Close'].iloc[-2]:
                confidence = 0.75

        # Fair value gaps on 1H
        if len(window_1h) >= 3:
            gap_up = (window_1h['Low'].iloc[-1] > window_1h['High'].iloc[-3])
            gap_down = (window_1h['High'].iloc[-1] < window_1h['Low'].iloc[-3])

            if gap_up and signal >= 0:
                signal = 1
                confidence = max(confidence, 0.7)
            elif gap_down and signal <= 0:
                signal = -1
                confidence = max(confidence, 0.7)

        if signal != 0:
            self.engine_usage['smc'] += 1
            return {'signal': signal, 'confidence': confidence}

        return None

    def analyze_wyckoff_patterns(self, window_4h, window_1d):
        """Analyze Wyckoff patterns for SOL."""
        if len(window_4h) < 20 or len(window_1d) < 10:
            return None

        # Volume analysis on 4H
        avg_volume = window_4h['Volume'].rolling(20).mean().iloc[-1]
        current_volume = window_4h['Volume'].iloc[-1]

        # Price action
        price_change_4h = (window_4h['Close'].iloc[-1] / window_4h['Close'].iloc[-5] - 1)

        # Spring pattern
        if current_volume > avg_volume * 1.5 and price_change_4h < -0.03:
            recent_low = window_4h['Low'].rolling(20).min().iloc[-1]
            if window_4h['Low'].iloc[-1] < recent_low * 1.01:
                # Check daily support
                daily_low = window_1d['Low'].rolling(5).min().iloc[-1]
                if window_4h['Low'].iloc[-1] > daily_low * 0.98:
                    self.engine_usage['wyckoff'] += 1
                    return {'signal': 1, 'confidence': 0.75}

        # UTAD pattern
        if current_volume > avg_volume * 1.5 and price_change_4h > 0.03:
            recent_high = window_4h['High'].rolling(20).max().iloc[-1]
            if window_4h['High'].iloc[-1] > recent_high * 0.99:
                # Check daily resistance
                daily_high = window_1d['High'].rolling(5).max().iloc[-1]
                if window_4h['High'].iloc[-1] < daily_high * 1.02:
                    self.engine_usage['wyckoff'] += 1
                    return {'signal': -1, 'confidence': 0.75}

        return None

    def analyze_hob_patterns(self, window_1h, window_4h, macro_window):
        """Analyze High Odds Breakout patterns for SOL."""
        if window_1h is None or len(window_1h) < 20 or len(window_4h) < 10:
            return None

        # Volume surge detection on 1H
        avg_volume = window_1h['Volume'].rolling(20).mean()
        std_volume = window_1h['Volume'].rolling(20).std()
        volume_z = (window_1h['Volume'].iloc[-1] - avg_volume.iloc[-1]) / (std_volume.iloc[-1] + 1e-8)

        # Enhanced requirements from v1.7.1
        min_z_long = self.config.get('liquidity', {}).get('hob_quality_factors', {}).get('volume_z_min_long', 1.3)
        min_z_short = self.config.get('liquidity', {}).get('hob_quality_factors', {}).get('volume_z_min_short', 1.6)

        # Breakout detection on 4H
        resistance = window_4h['High'].rolling(10).max().iloc[-1]
        support = window_4h['Low'].rolling(10).min().iloc[-1]
        current = window_1h['Close'].iloc[-1]

        if current > resistance * 0.995 and volume_z > min_z_long:
            # Additional momentum check
            momentum = (window_1h['Close'].iloc[-1] / window_1h['Close'].iloc[-5] - 1)
            if momentum > 0.02:  # 2% move in 5 hours
                self.engine_usage['hob'] += 1
                return {'signal': 1, 'confidence': 0.8}

        elif current < support * 1.005 and volume_z > min_z_short:
            # Additional momentum check
            momentum = (window_1h['Close'].iloc[-1] / window_1h['Close'].iloc[-5] - 1)
            if momentum < -0.02:  # -2% move in 5 hours
                self.engine_usage['hob'] += 1
                return {'signal': -1, 'confidence': 0.8}

        return None

    def analyze_momentum_patterns(self, window_4h, window_1d):
        """Analyze momentum patterns for SOL."""
        if len(window_4h) < 26 or len(window_1d) < 14:
            return None

        # RSI on 4H
        delta = window_4h['Close'].diff()
        gain = delta.where(delta > 0, 0).rolling(14).mean()
        loss = -delta.where(delta < 0, 0).rolling(14).mean()
        rs = gain / (loss + 1e-8)
        rsi = 100 - (100 / (1 + rs))

        # MACD on 4H
        ema12 = window_4h['Close'].ewm(span=12).mean()
        ema26 = window_4h['Close'].ewm(span=26).mean()
        macd = ema12 - ema26
        signal_line = macd.ewm(span=9).mean()

        # Daily trend
        daily_sma = window_1d['Close'].rolling(20).mean().iloc[-1] if len(window_1d) >= 20 else window_1d['Close'].mean()
        daily_trend = 1 if window_1d['Close'].iloc[-1] > daily_sma else -1

        # Generate signals
        if rsi.iloc[-1] < 35 and macd.iloc[-1] > signal_line.iloc[-1] and daily_trend > 0:
            self.engine_usage['momentum'] += 1
            return {'signal': 1, 'confidence': 0.65}
        elif rsi.iloc[-1] > 65 and macd.iloc[-1] < signal_line.iloc[-1] and daily_trend < 0:
            self.engine_usage['momentum'] += 1
            return {'signal': -1, 'confidence': 0.65}

        return None

    def analyze_macro_context(self, window_1d, macro_window):
        """Analyze macro context for SOL trading."""
        if len(window_1d) < 20 or macro_window is None or len(macro_window) < 5:
            return None

        # SOL trend analysis
        sma_20 = window_1d['Close'].rolling(20).mean().iloc[-1]
        sma_50 = window_1d['Close'].rolling(50).mean().iloc[-1] if len(window_1d) >= 50 else sma_20
        current = window_1d['Close'].iloc[-1]

        # Trend alignment
        if current > sma_20 * 1.02 and sma_20 > sma_50:
            trend = 1
        elif current < sma_20 * 0.98 and sma_20 < sma_50:
            trend = -1
        else:
            return None

        # Check volatility environment
        if 'vix' in macro_window.columns:
            vix = macro_window['vix'].iloc[-1] if not macro_window['vix'].empty else 20
            if vix > 35:  # Very high volatility - reduce confidence
                confidence = 0.4
            elif vix > 25:  # High volatility
                confidence = 0.5
            else:  # Normal volatility
                confidence = 0.65
        else:
            confidence = 0.55

        self.engine_usage['macro'] += 1
        return {'signal': trend, 'confidence': confidence}

    def enhanced_signal_generation(self, dataset, current_idx):
        """Generate enhanced multi-engine signals with v1.7.1 improvements."""
        # Extract data windows
        window_1h = dataset['sol_1h'][:current_idx] if 'sol_1h' in dataset and dataset['sol_1h'] is not None else None
        window_4h = dataset['sol_4h'][:current_idx] if 'sol_4h' in dataset else pd.DataFrame()
        window_1d = dataset['sol_1d'][:current_idx] if 'sol_1d' in dataset else pd.DataFrame()
        macro_window = dataset['macro'][:current_idx] if 'macro' in dataset else None

        if len(window_4h) < 20 or len(window_1d) < 10:
            return None

        # Get signals from all engines
        signals = []

        # Core engines
        smc_signal = self.analyze_smc_patterns(window_1h, window_4h)
        wyckoff_signal = self.analyze_wyckoff_patterns(window_4h, window_1d)
        hob_signal = self.analyze_hob_patterns(window_1h, window_4h, macro_window)
        momentum_signal = self.analyze_momentum_patterns(window_4h, window_1d)
        macro_signal = self.analyze_macro_context(window_1d, macro_window)

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

        # Check if counter-trend (simplified)
        recent_trend = (window_4h['Close'].iloc[-1] / window_4h['Close'].iloc[-10] - 1) if len(window_4h) >= 10 else 0

        # Process long signals
        if len(long_signals) >= 2:
            avg_confidence = np.mean([s['confidence'] for s in long_signals])

            # Counter-trend check
            is_counter_trend = recent_trend < -0.05  # Down 5% = counter-trend long
            if is_counter_trend and len(long_signals) < min_engines:
                self.engine_usage['counter_trend_blocked'] += 1
                return None

            return {
                'signal': 1,
                'confidence': avg_confidence,
                'engines': len(long_signals),
                'timestamp': window_4h.index[-1],
                'price': window_4h['Close'].iloc[-1]
            }

        # Process short signals
        elif len(short_signals) >= 2:
            avg_confidence = np.mean([s['confidence'] for s in short_signals])

            # Counter-trend check
            is_counter_trend = recent_trend > 0.05  # Up 5% = counter-trend short
            if is_counter_trend and len(short_signals) < min_engines:
                self.engine_usage['counter_trend_blocked'] += 1
                return None

            # Could add SOLBTC check here if data available

            return {
                'signal': -1,
                'confidence': avg_confidence,
                'engines': len(short_signals),
                'timestamp': window_4h.index[-1],
                'price': window_4h['Close'].iloc[-1]
            }

        return None

    def execute_trade(self, signal, price):
        """Execute trade with v1.7.1 risk management."""
        if self.position is not None:
            # Close existing position
            exit_price = price

            # Calculate raw P&L
            if self.position['side'] == 1:  # Long
                pnl_pct = (exit_price / self.position['entry_price'] - 1) * 100
            else:  # Short
                pnl_pct = (1 - exit_price / self.position['entry_price']) * 100

            # Apply transaction costs
            pnl_pct -= 0.3  # 0.15% each way

            # Apply asymmetric R/R management from v1.7.1
            stop_loss_pct = 3.0  # 3% stop
            take_profit_pct = 7.5  # 7.5% target (2.5:1 R/R)

            # Determine exit type
            if pnl_pct <= -stop_loss_pct:
                pnl_pct = -stop_loss_pct
                exit_type = 'stop_loss'
            elif pnl_pct >= take_profit_pct:
                pnl_pct = take_profit_pct
                exit_type = 'take_profit'
            else:
                exit_type = 'signal_reversal'

            self.balance *= (1 + pnl_pct / 100)

            self.trades.append({
                'entry_time': self.position['entry_time'],
                'exit_time': signal['timestamp'] if signal else self.position['entry_time'],
                'side': 'long' if self.position['side'] == 1 else 'short',
                'entry_price': self.position['entry_price'],
                'exit_price': exit_price,
                'pnl_pct': pnl_pct,
                'balance': self.balance,
                'engines_used': self.position['engines'],
                'exit_type': exit_type
            })

            self.position = None

        # Open new position
        if signal and signal['signal'] != 0:
            self.position = {
                'entry_time': signal['timestamp'],
                'entry_price': price,
                'side': signal['signal'],
                'engines': signal['engines']
            }

    def run_backtest(self, start_date: str = None, end_date: str = None):
        """Run enhanced backtest on SOL data with 1H/4H/1D timeframes."""
        print(f"🚀 Running Enhanced SOL Bull Machine v1.7.1 Backtest")
        print("=" * 70)
        print("Engine: Complete Bull Machine v1.7.1 Enhanced")
        print("Data: Real SOL with 1H/4H/1D timeframes")
        print("Enhancements: All v1.7.1 improvements")
        print("=" * 70)

        # Load SOL data
        dataset = {}

        # Load 1H data
        sol_1h = self.data_loader.load_raw_data('COINBASE_SOLUSD', '1H')
        if sol_1h is not None:
            dataset['sol_1h'] = sol_1h
            print(f"✅ Loaded SOL 1H: {len(sol_1h)} bars")
        else:
            print("⚠️  No SOL 1H data available")

        # Load 4H data
        sol_4h = self.data_loader.load_raw_data('COINBASE_SOLUSD', '4H')
        if sol_4h is None:
            print("❌ No SOL 4H data available - required for backtesting")
            return None
        dataset['sol_4h'] = sol_4h
        print(f"✅ Loaded SOL 4H: {len(sol_4h)} bars")

        # Load 1D data
        sol_1d = self.data_loader.load_raw_data('COINBASE_SOLUSD', '1D')
        if sol_1d is None:
            print("❌ No SOL 1D data available - required for backtesting")
            return None
        dataset['sol_1d'] = sol_1d
        print(f"✅ Loaded SOL 1D: {len(sol_1d)} bars")

        # Find overlapping period
        if sol_1h is not None:
            start_overlap = max(sol_1h.index[0], sol_4h.index[0], sol_1d.index[0])
            end_overlap = min(sol_1h.index[-1], sol_4h.index[-1], sol_1d.index[-1])
        else:
            start_overlap = max(sol_4h.index[0], sol_1d.index[0])
            end_overlap = min(sol_4h.index[-1], sol_1d.index[-1])

        print(f"\n📊 Overlapping data period: {start_overlap} → {end_overlap}")
        print(f"Duration: {(end_overlap - start_overlap).days} days")

        # Filter to overlapping period
        if sol_1h is not None:
            dataset['sol_1h'] = sol_1h[(sol_1h.index >= start_overlap) & (sol_1h.index <= end_overlap)]
        dataset['sol_4h'] = sol_4h[(sol_4h.index >= start_overlap) & (sol_4h.index <= end_overlap)]
        dataset['sol_1d'] = sol_1d[(sol_1d.index >= start_overlap) & (sol_1d.index <= end_overlap)]

        # Generate synthetic macro data
        macro = self.data_loader.generate_macro_data(
            start_overlap.strftime('%Y-%m-%d'),
            end_overlap.strftime('%Y-%m-%d'),
            '4H'
        )
        dataset['macro'] = macro

        print(f"\n✅ Dataset ready for backtesting:")
        print(f"   SOL 1H: {len(dataset['sol_1h']) if 'sol_1h' in dataset and dataset['sol_1h'] is not None else 0} bars")
        print(f"   SOL 4H: {len(dataset['sol_4h'])} bars")
        print(f"   SOL 1D: {len(dataset['sol_1d'])} bars")
        print(f"   Macro: {len(dataset['macro'])} bars")

        # Run through each 4H bar
        total_bars = len(dataset['sol_4h'])

        for i in range(26, total_bars):  # Start after warmup for indicators
            signal = self.enhanced_signal_generation(dataset, i)

            if signal:
                self.execute_trade(signal, signal['price'])

                # Display trade entry
                if self.position:
                    side = 'bullish' if signal['signal'] == 1 else 'bearish'
                    engines_str = f"Engines: {signal['engines']}"
                    print(f"🎯 Entry {signal['timestamp'].strftime('%Y-%m-%d %H:%M')} | {side:>8} | "
                          f"${signal['price']:8.2f} | {engines_str} | Conf: {signal['confidence']:.2f}")

            # Progress indicator
            if i % 100 == 0:
                print(f"Progress: {i}/{total_bars} ({i/total_bars*100:.1f}%)")

        # Close final position if any
        if self.position is not None:
            final_price = dataset['sol_4h']['Close'].iloc[-1]
            self.execute_trade(None, final_price)

        return self.generate_results(
            start_overlap.strftime('%Y-%m-%d'),
            end_overlap.strftime('%Y-%m-%d'),
            dataset['sol_4h']
        )

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
                    'total_return': (self.balance / 10000 - 1) * 100
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
    """Run enhanced SOL backtest with Bull Machine v1.7.1."""
    backtest = EnhancedSOLBullMachine()

    # Run backtest
    results = backtest.run_backtest()

    if results:
        # Save results
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f'sol_enhanced_results_{timestamp}.json'

        with open(filename, 'w') as f:
            # Convert datetime objects to strings for JSON
            results_json = results.copy()
            for trade in results_json['trades']:
                trade['entry_time'] = trade['entry_time'].strftime('%Y-%m-%d %H:%M:%S')
                trade['exit_time'] = trade['exit_time'].strftime('%Y-%m-%d %H:%M:%S')

            json.dump(results_json, f, indent=2)

        # Display results
        print("\n" + "="*80)
        print("📈 ENHANCED SOL BULL MACHINE v1.7.1 BACKTEST RESULTS")
        print("="*80)

        perf = results['performance']
        period = results['period']

        print(f"\n📅 PERIOD:")
        print(f"   Start: {period['start']}")
        print(f"   End: {period['end']}")
        print(f"   Duration: {period['days']} days ({period['years']:.2f} years)")

        print(f"\n💰 PERFORMANCE:")
        print(f"   Starting Balance: $10,000")
        print(f"   Final Balance: ${perf['final_balance']:,.2f}")
        print(f"   Total Return: {perf['total_return']:+.2f}%")
        print(f"   Max Drawdown: {perf['max_drawdown']:.2f}%")

        print(f"\n📊 TRADING STATS:")
        print(f"   Total Trades: {perf['total_trades']}")
        print(f"   Win Rate: {perf['win_rate']:.1f}%")
        print(f"   Winning Trades: {perf['winning_trades']}")
        print(f"   Losing Trades: {perf['losing_trades']}")
        print(f"   Avg Win: {perf['avg_win']:+.2f}%")
        print(f"   Avg Loss: {perf['avg_loss']:.2f}%")
        print(f"   Profit Factor: {perf['profit_factor']:.2f}")

        print(f"\n🔧 ENGINE USAGE:")
        for engine, count in results['engine_usage'].items():
            if count > 0:
                print(f"   {engine.upper()}: {count}")

        # Institutional assessment
        print(f"\n🎯 INSTITUTIONAL ASSESSMENT:")
        checks = {
            'Win Rate': perf['win_rate'] > 50,
            'Profit Factor': perf['profit_factor'] > 1.5,
            'Max Drawdown': perf['max_drawdown'] < 35,
            'Positive Return': perf['total_return'] > 0
        }

        for metric, passed in checks.items():
            status = "✅ PASS" if passed else "❌ FAIL"
            print(f"   {metric}: {status}")

        health_score = sum(checks.values()) / len(checks) * 100
        print(f"\n🏥 OVERALL HEALTH SCORE: {health_score:.0f}%")

        # Final verdict
        print(f"\n🚀 VERDICT:")
        print("=" * 80)
        if health_score >= 80 and perf['total_return'] > 50:
            print("✅ EXCEPTIONAL PERFORMANCE - Production Ready!")
        elif health_score >= 60:
            print("✅ SOLID PERFORMANCE - Consider deployment")
        else:
            print("⚠️  Performance needs improvement for SOL")

        print(f"\n💾 Results saved to: {filename}")

if __name__ == "__main__":
    main()