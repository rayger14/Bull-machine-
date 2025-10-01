#!/usr/bin/env python3
"""
SPY Analysis with Complete Bull Machine v1.7
Multi-timeframe analysis with trade setup recommendations
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

# Import Bull Machine components
from engine.smc.smc_engine import SMCEngine
from engine.wyckoff.wyckoff_engine import WyckoffEngine
from engine.liquidity.hob import HOBDetector
from engine.momentum.momentum_engine import MomentumEngine

class SPYBullMachineAnalyzer:
    """Complete Bull Machine analysis for SPY with MTF confluence"""

    def __init__(self, config_path: str):
        with open(config_path, 'r') as f:
            self.config = json.load(f)

        # Initialize engines
        self.smc_engine = SMCEngine(self.config['domains']['smc'])
        self.momentum_engine = MomentumEngine(self.config['domains']['momentum'])
        self.wyckoff_engine = WyckoffEngine(self.config['domains']['wyckoff'])
        self.hob_engine = HOBDetector(self.config['domains']['liquidity']['hob_detection'])

        # Get calibrated thresholds
        self.conf_threshold = self.config['fusion']['calibration_thresholds']['confidence']
        self.strength_threshold = self.config['fusion']['calibration_thresholds']['strength']

        print(f"🚀 BULL MACHINE v{self.config['version']} SPY ANALYSIS")
        print(f"📊 Calibrated: conf={self.conf_threshold}, strength={self.strength_threshold}")

    def load_spy_data(self):
        """Load SPY data from provided charts"""
        print(f"\n📊 LOADING SPY MULTI-TIMEFRAME DATA")
        print("-" * 50)

        data = {}

        # Load 1H data
        try:
            spy_1h = pd.read_csv('/Users/raymondghandchi/Downloads/SPY/BATS_SPY, 60_f3fa4.csv')
            spy_1h['time'] = pd.to_datetime(spy_1h['time'], unit='s', utc=True)
            spy_1h = spy_1h.set_index('time').sort_index()
            spy_1h.columns = spy_1h.columns.str.lower()

            # Fix volume column for SPY data structure
            if 'total buy volume' in spy_1h.columns and 'total sell volume' in spy_1h.columns:
                spy_1h['volume'] = spy_1h['total buy volume'] + spy_1h['total sell volume']
            elif 'buy+sell v' in spy_1h.columns:
                spy_1h['volume'] = spy_1h['buy+sell v']

            data['1H'] = spy_1h
            print(f"✅ SPY 1H: {len(spy_1h)} bars ({spy_1h.index[0]} to {spy_1h.index[-1]})")
        except Exception as e:
            print(f"⚠️ SPY 1H failed: {e}")
            data['1H'] = None

        # Load 4H data
        try:
            spy_4h = pd.read_csv('/Users/raymondghandchi/Downloads/SPY/BATS_SPY, 240_46931.csv')
            spy_4h['time'] = pd.to_datetime(spy_4h['time'], unit='s', utc=True)
            spy_4h = spy_4h.set_index('time').sort_index()
            spy_4h.columns = spy_4h.columns.str.lower()

            # Fix volume column for SPY data structure
            if 'total buy volume' in spy_4h.columns and 'total sell volume' in spy_4h.columns:
                spy_4h['volume'] = spy_4h['total buy volume'] + spy_4h['total sell volume']
            elif 'buy+sell v' in spy_4h.columns:
                spy_4h['volume'] = spy_4h['buy+sell v']

            data['4H'] = spy_4h
            print(f"✅ SPY 4H: {len(spy_4h)} bars ({spy_4h.index[0]} to {spy_4h.index[-1]})")
        except Exception as e:
            print(f"❌ SPY 4H failed: {e}")
            return None

        # Load 1D data
        try:
            spy_1d = pd.read_csv('/Users/raymondghandchi/Downloads/SPY/BATS_SPY, 1D_b1032.csv')
            spy_1d['time'] = pd.to_datetime(spy_1d['time'], unit='s', utc=True)
            spy_1d = spy_1d.set_index('time').sort_index()
            spy_1d.columns = spy_1d.columns.str.lower()

            # Fix volume column for SPY data structure
            if 'total buy volume' in spy_1d.columns and 'total sell volume' in spy_1d.columns:
                spy_1d['volume'] = spy_1d['total buy volume'] + spy_1d['total sell volume']
            elif 'buy+sell v' in spy_1d.columns:
                spy_1d['volume'] = spy_1d['buy+sell v']

            data['1D'] = spy_1d
            print(f"✅ SPY 1D: {len(spy_1d)} bars ({spy_1d.index[0]} to {spy_1d.index[-1]})")
        except Exception as e:
            print(f"⚠️ SPY 1D failed: {e}")
            data['1D'] = None

        # Normalize data for engine compatibility
        for timeframe, df in data.items():
            if df is not None:
                # Ensure core OHLCV columns exist and are properly formatted
                required_cols = ['open', 'high', 'low', 'close', 'volume']
                for col in required_cols:
                    if col not in df.columns:
                        if col == 'volume':
                            # Default volume if not available
                            df['volume'] = 1000000
                        else:
                            print(f"⚠️ Missing {col} column in {timeframe} data")

                # Keep only core OHLCV columns for engine compatibility
                df = df[['open', 'high', 'low', 'close', 'volume']].copy()
                data[timeframe] = df

        return data

    def run_spy_backtest(self, data, lookback_bars=200):
        """Run comprehensive SPY backtest"""
        print(f"\n🎯 RUNNING SPY BACKTEST ({lookback_bars} bars)")
        print("=" * 60)

        if '4H' not in data:
            print("❌ No 4H data available")
            return None

        # Use 4H as primary timeframe
        primary_data = data['4H'].tail(lookback_bars)
        data_1h = data.get('1H')
        data_1d = data.get('1D')

        print(f"📅 Backtest period: {primary_data.index[0]} to {primary_data.index[-1]}")
        print(f"📊 Bars: {len(primary_data)} (4H primary)")

        # Initialize portfolio
        portfolio = {
            'capital': 100000.0,
            'position': 0.0,
            'entry_price': 0.0,
            'trades_count': 0,
            'wins': 0,
            'losses': 0
        }

        trades = []
        signals = []

        # Process backtest
        for i in range(50, len(primary_data)):
            current_bar = primary_data.iloc[i]
            historical_data = primary_data.iloc[:i+1]
            recent_data = historical_data.tail(60)

            try:
                # Generate domain signals
                domain_signals = self._generate_domain_signals(recent_data)

                # MTF confluence
                mtf_confluence = self._analyze_mtf_confluence(data_1h, historical_data, data_1d, current_bar)

                # Signal fusion
                fusion_result = self._fuse_signals(domain_signals, current_bar)

                if fusion_result and mtf_confluence['aligned']:
                    signals.append({
                        'timestamp': current_bar.name,
                        'price': current_bar['close'],
                        'direction': fusion_result['direction'],
                        'confidence': fusion_result['confidence'],
                        'strength': fusion_result['strength'],
                        'mtf_confluence': mtf_confluence
                    })

                    # Execute trade
                    trade_executed = self._execute_spy_trade(
                        portfolio, current_bar, fusion_result, mtf_confluence, trades
                    )

            except Exception as e:
                continue

        # Close final position
        if portfolio['position'] != 0:
            self._close_position(portfolio, primary_data.iloc[-1], trades)

        return self._calculate_spy_results(portfolio, trades, signals)

    def _generate_domain_signals(self, data):
        """Generate signals from all engines"""
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
            signals['wyckoff'] = self.wyckoff_engine.analyze(data, usdt_stagnation=0.3)  # Lower for SPY
        except:
            signals['wyckoff'] = None

        try:
            signals['hob'] = self.hob_engine.detect_hob(data)
        except:
            signals['hob'] = None

        return signals

    def _analyze_mtf_confluence(self, data_1h, data_4h, data_1d, current_bar):
        """Analyze multi-timeframe confluence for SPY"""
        confluence = {
            'aligned': False,
            'trends': {},
            'score': 0.0,
            'level': 'none'
        }

        current_time = current_bar.name

        try:
            # 1H trend (if available)
            if data_1h is not None:
                aligned_1h = data_1h[data_1h.index <= current_time]
                if len(aligned_1h) >= 24:
                    sma_1h_fast = aligned_1h['close'].tail(12).mean()  # 12H
                    sma_1h_slow = aligned_1h['close'].tail(24).mean()  # 24H

                    if sma_1h_fast > sma_1h_slow * 1.002:
                        confluence['trends']['1H'] = 'bullish'
                    elif sma_1h_fast < sma_1h_slow * 0.998:
                        confluence['trends']['1H'] = 'bearish'
                    else:
                        confluence['trends']['1H'] = 'neutral'

            # 4H trend
            if len(data_4h) >= 12:
                sma_4h_fast = data_4h['close'].tail(6).mean()   # 24H
                sma_4h_slow = data_4h['close'].tail(12).mean()  # 48H

                if sma_4h_fast > sma_4h_slow * 1.003:
                    confluence['trends']['4H'] = 'bullish'
                elif sma_4h_fast < sma_4h_slow * 0.997:
                    confluence['trends']['4H'] = 'bearish'
                else:
                    confluence['trends']['4H'] = 'neutral'

            # 1D trend (if available)
            if data_1d is not None:
                aligned_1d = data_1d[data_1d.index <= current_time]
                if len(aligned_1d) >= 10:
                    sma_1d_fast = aligned_1d['close'].tail(5).mean()
                    sma_1d_slow = aligned_1d['close'].tail(10).mean()

                    if sma_1d_fast > sma_1d_slow * 1.005:
                        confluence['trends']['1D'] = 'bullish'
                    elif sma_1d_fast < sma_1d_slow * 0.995:
                        confluence['trends']['1D'] = 'bearish'
                    else:
                        confluence['trends']['1D'] = 'neutral'

            # Calculate alignment
            trends = [t for t in confluence['trends'].values() if t != 'neutral']
            if len(trends) >= 2:
                bullish_count = trends.count('bullish')
                bearish_count = trends.count('bearish')

                if bullish_count == len(trends):
                    confluence['level'] = 'full_bullish'
                    confluence['score'] = 1.0
                    confluence['aligned'] = True
                elif bearish_count == len(trends):
                    confluence['level'] = 'full_bearish'
                    confluence['score'] = 1.0
                    confluence['aligned'] = True
                elif bullish_count > bearish_count:
                    confluence['score'] = bullish_count / len(trends)
                    confluence['aligned'] = confluence['score'] >= 0.67
                    confluence['level'] = 'partial_bullish'
                elif bearish_count > bullish_count:
                    confluence['score'] = bearish_count / len(trends)
                    confluence['aligned'] = confluence['score'] >= 0.67
                    confluence['level'] = 'partial_bearish'

        except Exception as e:
            pass

        return confluence

    def _fuse_signals(self, domain_signals, current_bar):
        """Fuse domain signals using Bull Machine logic"""
        active_signals = [s for s in domain_signals.values() if s is not None]

        if len(active_signals) < 1:
            return None

        directions = []
        confidences = []

        for signal in active_signals:
            if hasattr(signal, 'direction') and hasattr(signal, 'confidence'):
                directions.append(signal.direction)
                confidences.append(signal.confidence)

        if not directions or not confidences:
            return None

        # Direction voting
        long_votes = directions.count('long')
        short_votes = directions.count('short')

        if long_votes > short_votes:
            fusion_direction = 'long'
            fusion_strength = long_votes / len(directions)
        elif short_votes > long_votes:
            fusion_direction = 'short'
            fusion_strength = short_votes / len(directions)
        else:
            return None

        avg_confidence = np.mean(confidences)

        # Check thresholds
        if avg_confidence >= self.conf_threshold and fusion_strength >= self.strength_threshold:
            return {
                'direction': fusion_direction,
                'confidence': avg_confidence,
                'strength': fusion_strength,
                'engines': len(active_signals)
            }

        return None

    def _execute_spy_trade(self, portfolio, current_bar, fusion_result, mtf_confluence, trades):
        """Execute SPY trade with proper sizing"""
        current_price = current_bar['close']

        # Close opposite position
        if portfolio['position'] != 0:
            if ((portfolio['position'] > 0 and fusion_result['direction'] == 'short') or
                (portfolio['position'] < 0 and fusion_result['direction'] == 'long')):
                self._close_position(portfolio, current_bar, trades)

        # Open new position
        if portfolio['position'] == 0:
            # SPY-specific sizing (more conservative)
            base_risk = 0.05  # 5% base risk for SPY
            confidence_mult = min(1.2, fusion_result['confidence'] / 0.3)
            mtf_mult = 1.0 + (mtf_confluence['score'] * 0.3)

            final_sizing = base_risk * confidence_mult * mtf_mult
            final_sizing = min(final_sizing, 0.08)  # Cap at 8% for SPY

            position_value = portfolio['capital'] * final_sizing

            if fusion_result['direction'] == 'long':
                portfolio['position'] = position_value / current_price
            else:
                portfolio['position'] = -position_value / current_price

            portfolio['entry_price'] = current_price
            portfolio['trades_count'] += 1

            trade = {
                'trade_id': portfolio['trades_count'],
                'entry_timestamp': current_bar.name,
                'entry_price': current_price,
                'direction': fusion_result['direction'],
                'confidence': fusion_result['confidence'],
                'engines': fusion_result['engines'],
                'mtf_confluence': mtf_confluence,
                'sizing': final_sizing / base_risk,
                'capital_at_entry': portfolio['capital']
            }

            trades.append(trade)
            return True

        return False

    def _close_position(self, portfolio, current_bar, trades):
        """Close current position"""
        if portfolio['position'] == 0 or not trades:
            return

        current_price = current_bar['close']

        # Calculate PnL
        if portfolio['position'] > 0:
            pnl = portfolio['position'] * (current_price - portfolio['entry_price'])
        else:
            pnl = abs(portfolio['position']) * (portfolio['entry_price'] - current_price)

        portfolio['capital'] += pnl

        if pnl > 0:
            portfolio['wins'] += 1
        else:
            portfolio['losses'] += 1

        # Update trade
        trade = trades[-1]
        trade.update({
            'exit_timestamp': current_bar.name,
            'exit_price': current_price,
            'pnl': pnl,
            'return_pct': (pnl / (abs(portfolio['position']) * portfolio['entry_price'])) * 100
        })

        portfolio['position'] = 0
        portfolio['entry_price'] = 0

    def _calculate_spy_results(self, portfolio, trades, signals):
        """Calculate SPY backtest results"""
        completed_trades = [t for t in trades if 'exit_price' in t]

        results = {
            'portfolio': portfolio,
            'total_signals': len(signals),
            'total_trades': len(completed_trades),
            'trades': completed_trades
        }

        if completed_trades:
            returns = [t['return_pct'] for t in completed_trades]
            total_return = (portfolio['capital'] - 100000) / 100000 * 100

            results.update({
                'total_return_pct': total_return,
                'win_rate': portfolio['wins'] / len(completed_trades) * 100,
                'avg_return': np.mean(returns),
                'best_trade': max(returns),
                'worst_trade': min(returns),
                'final_capital': portfolio['capital']
            })

        return results

    def analyze_current_setup(self, data):
        """Analyze current SPY setup for this week"""
        print(f"\n🔍 CURRENT SPY SETUP ANALYSIS")
        print("=" * 50)

        if '4H' not in data:
            print("❌ No 4H data for analysis")
            return None

        # Get latest data
        latest_4h = data['4H'].tail(100)
        latest_1h = data.get('1H')
        latest_1d = data.get('1D')

        current_bar = latest_4h.iloc[-1]
        current_price = current_bar['close']

        print(f"📅 Analysis Date: {current_bar.name}")
        print(f"💰 Current SPY Price: ${current_price:.2f}")

        # Generate current signals
        domain_signals = self._generate_domain_signals(latest_4h.tail(60))

        # MTF analysis
        mtf_confluence = self._analyze_mtf_confluence(latest_1h, latest_4h, latest_1d, current_bar)

        # Signal fusion
        fusion_result = self._fuse_signals(domain_signals, current_bar)

        # Display analysis
        print(f"\n🤖 DOMAIN ANALYSIS")
        print("-" * 30)

        for engine, signal in domain_signals.items():
            if signal:
                direction = getattr(signal, 'direction', 'N/A')
                confidence = getattr(signal, 'confidence', 0)
                print(f"   {engine.upper()}: {direction} (conf: {confidence:.3f})")
            else:
                print(f"   {engine.upper()}: No signal")

        print(f"\n🔄 MULTI-TIMEFRAME CONFLUENCE")
        print("-" * 30)
        print(f"   Trends: {mtf_confluence['trends']}")
        print(f"   Level: {mtf_confluence['level']}")
        print(f"   Score: {mtf_confluence['score']:.2f}")
        print(f"   Aligned: {'✅ YES' if mtf_confluence['aligned'] else '❌ NO'}")

        print(f"\n⚡ FUSION RESULT")
        print("-" * 30)
        if fusion_result:
            print(f"   Direction: {fusion_result['direction'].upper()}")
            print(f"   Confidence: {fusion_result['confidence']:.3f} (≥{self.conf_threshold})")
            print(f"   Strength: {fusion_result['strength']:.3f} (≥{self.strength_threshold})")
            print(f"   Active Engines: {fusion_result['engines']}")
        else:
            print("   No fusion signal generated")

        # Trade recommendation
        print(f"\n🎯 TRADE RECOMMENDATION")
        print("=" * 30)

        if fusion_result and mtf_confluence['aligned']:
            print(f"✅ TRADE SETUP ACTIVE")
            print(f"   Direction: {fusion_result['direction'].upper()}")
            print(f"   Entry: ${current_price:.2f}")
            print(f"   Confidence: {fusion_result['confidence']:.3f}")
            print(f"   MTF Support: {mtf_confluence['level']}")

            # Calculate position sizing
            base_risk = 0.05
            confidence_mult = min(1.2, fusion_result['confidence'] / 0.3)
            mtf_mult = 1.0 + (mtf_confluence['score'] * 0.3)
            final_sizing = base_risk * confidence_mult * mtf_mult
            final_sizing = min(final_sizing, 0.08)

            print(f"   Suggested Size: {final_sizing*100:.1f}% of capital")

            # Risk levels
            if fusion_result['direction'] == 'long':
                recent_low = latest_4h['low'].tail(20).min()
                stop_loss = recent_low * 0.995
                take_profit = current_price * 1.03
            else:
                recent_high = latest_4h['high'].tail(20).max()
                stop_loss = recent_high * 1.005
                take_profit = current_price * 0.97

            print(f"   Stop Loss: ${stop_loss:.2f}")
            print(f"   Take Profit: ${take_profit:.2f}")

        else:
            print(f"❌ NO TRADE SETUP")
            reasons = []
            if not fusion_result:
                reasons.append("Insufficient signal strength")
            if not mtf_confluence['aligned']:
                reasons.append("MTF not aligned")
            print(f"   Reasons: {', '.join(reasons)}")

        return {
            'current_price': current_price,
            'domain_signals': domain_signals,
            'mtf_confluence': mtf_confluence,
            'fusion_result': fusion_result,
            'trade_setup': fusion_result and mtf_confluence['aligned']
        }

def main():
    """Run SPY analysis with Bull Machine"""

    try:
        analyzer = SPYBullMachineAnalyzer('configs/v170/assets/ETH_v17_tuned.json')

        # Load SPY data
        spy_data = analyzer.load_spy_data()
        if not spy_data:
            print("❌ Failed to load SPY data")
            return

        # Run backtest
        backtest_results = analyzer.run_spy_backtest(spy_data, lookback_bars=300)

        if backtest_results:
            print(f"\n📈 SPY BACKTEST RESULTS")
            print("=" * 40)
            print(f"Total Return: {backtest_results.get('total_return_pct', 0):+.2f}%")
            print(f"Total P&L: ${(backtest_results.get('final_capital', 100000) - 100000):+,.2f}")
            print(f"Win Rate: {backtest_results.get('win_rate', 0):.1f}%")
            print(f"Total Trades: {backtest_results.get('total_trades', 0)}")
            print(f"Total Signals: {backtest_results.get('total_signals', 0)}")
            print(f"Signal-to-Trade Ratio: {(backtest_results.get('total_trades', 0) / max(1, backtest_results.get('total_signals', 1))) * 100:.1f}%")
            print(f"Final Capital: ${backtest_results.get('final_capital', 100000):,.2f}")

            # Trade-by-trade breakdown
            trades = backtest_results.get('trades', [])
            if trades:
                print(f"\n💵 TRADE-BY-TRADE BREAKDOWN")
                print("-" * 50)
                for i, trade in enumerate(trades, 1):
                    entry_date = trade['entry_timestamp'].strftime('%m/%d')
                    exit_date = trade['exit_timestamp'].strftime('%m/%d')
                    pnl = trade.get('pnl', 0)
                    return_pct = trade.get('return_pct', 0)

                    print(f"Trade {i}: {trade['direction'].upper()} | "
                          f"{entry_date}-{exit_date} | "
                          f"${trade['entry_price']:.2f} → ${trade['exit_price']:.2f} | "
                          f"P&L: ${pnl:+,.2f} ({return_pct:+.1f}%)")

                # Additional metrics
                if 'avg_return' in backtest_results:
                    print(f"\n📊 PERFORMANCE METRICS")
                    print("-" * 30)
                    print(f"Average Trade: {backtest_results['avg_return']:+.1f}%")
                    print(f"Best Trade: {backtest_results['best_trade']:+.1f}%")
                    print(f"Worst Trade: {backtest_results['worst_trade']:+.1f}%")

                    # Calculate profit factor
                    profitable_trades = [t for t in trades if t.get('pnl', 0) > 0]
                    losing_trades = [t for t in trades if t.get('pnl', 0) <= 0]

                    if profitable_trades and losing_trades:
                        gross_profit = sum(t.get('pnl', 0) for t in profitable_trades)
                        gross_loss = abs(sum(t.get('pnl', 0) for t in losing_trades))
                        profit_factor = gross_profit / gross_loss if gross_loss > 0 else 999

                        print(f"Gross Profit: ${gross_profit:+,.2f}")
                        print(f"Gross Loss: ${gross_loss:+,.2f}")
                        print(f"Profit Factor: {profit_factor:.2f}")

        # Analyze current setup
        current_analysis = analyzer.analyze_current_setup(spy_data)

        print(f"\n🎉 SPY ANALYSIS COMPLETE")

    except Exception as e:
        print(f"❌ Analysis failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()