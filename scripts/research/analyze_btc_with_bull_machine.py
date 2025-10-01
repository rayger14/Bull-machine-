#!/usr/bin/env python3
"""
BTC Analysis with Bull Machine v1.7
Complete multi-timeframe analysis with immediate setup identification
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

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

class BTCBullMachineAnalyzer:
    def __init__(self):
        # Load ETH config (works for BTC too with adaptive parameters)
        with open('configs/v170/assets/ETH_v17_tuned.json', 'r') as f:
            self.config = json.load(f)

        # Initialize engines
        self.smc_engine = SMCEngine(self.config['domains']['smc'])
        self.momentum_engine = MomentumEngine(self.config['domains']['momentum'])
        self.wyckoff_engine = WyckoffEngine(self.config['domains']['wyckoff'])
        self.hob_engine = HOBDetector(self.config['domains']['liquidity']['hob_detection'])

        # Get calibrated thresholds
        self.conf_threshold = self.config['fusion']['calibration_thresholds']['confidence']
        self.strength_threshold = self.config['fusion']['calibration_thresholds']['strength']

        print(f"🚀 BULL MACHINE v{self.config['version']} BTC ANALYSIS")
        print(f"📊 Calibrated: conf={self.conf_threshold}, strength={self.strength_threshold}")

    def load_btc_data(self):
        """Load BTC data from provided charts"""
        print(f"\n📊 LOADING BTC MULTI-TIMEFRAME DATA")
        print("-" * 50)

        data = {}

        # Load 1H data
        try:
            btc_1h = pd.read_csv('data/btc/COINBASE_BTCUSD, 60_9db64.csv')
            btc_1h['time'] = pd.to_datetime(btc_1h['time'], unit='s', utc=True)
            btc_1h = btc_1h.set_index('time').sort_index()
            btc_1h.columns = btc_1h.columns.str.lower()

            # Fix volume column for BTC data structure
            if 'total buy volume' in btc_1h.columns and 'total sell volume' in btc_1h.columns:
                btc_1h['volume'] = btc_1h['total buy volume'] + btc_1h['total sell volume']
            elif 'buy+sell v' in btc_1h.columns:
                btc_1h['volume'] = btc_1h['buy+sell v']

            data['1H'] = btc_1h
            print(f"✅ BTC 1H: {len(btc_1h)} bars ({btc_1h.index[0]} to {btc_1h.index[-1]})")
        except Exception as e:
            print(f"⚠️ BTC 1H failed: {e}")
            data['1H'] = None

        # Load 4H data
        try:
            btc_4h = pd.read_csv('data/btc/COINBASE_BTCUSD, 240_3e00d.csv')
            btc_4h['time'] = pd.to_datetime(btc_4h['time'], unit='s', utc=True)
            btc_4h = btc_4h.set_index('time').sort_index()
            btc_4h.columns = btc_4h.columns.str.lower()

            # Fix volume column for BTC data structure
            if 'total buy volume' in btc_4h.columns and 'total sell volume' in btc_4h.columns:
                btc_4h['volume'] = btc_4h['total buy volume'] + btc_4h['total sell volume']
            elif 'buy+sell v' in btc_4h.columns:
                btc_4h['volume'] = btc_4h['buy+sell v']

            data['4H'] = btc_4h
            print(f"✅ BTC 4H: {len(btc_4h)} bars ({btc_4h.index[0]} to {btc_4h.index[-1]})")
        except Exception as e:
            print(f"❌ BTC 4H failed: {e}")
            return None

        # Load 1D data
        try:
            btc_1d = pd.read_csv('data/btc/COINBASE_BTCUSD, 1D_96a85.csv')
            btc_1d['time'] = pd.to_datetime(btc_1d['time'], unit='s', utc=True)
            btc_1d = btc_1d.set_index('time').sort_index()
            btc_1d.columns = btc_1d.columns.str.lower()

            # Fix volume column for BTC data structure
            if 'total buy volume' in btc_1d.columns and 'total sell volume' in btc_1d.columns:
                btc_1d['volume'] = btc_1d['total buy volume'] + btc_1d['total sell volume']
            elif 'buy+sell v' in btc_1d.columns:
                btc_1d['volume'] = btc_1d['buy+sell v']

            data['1D'] = btc_1d
            print(f"✅ BTC 1D: {len(btc_1d)} bars ({btc_1d.index[0]} to {btc_1d.index[-1]})")
        except Exception as e:
            print(f"⚠️ BTC 1D failed: {e}")
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

    def run_btc_backtest(self, data, lookback_bars=200):
        """Run comprehensive BTC backtest"""
        print(f"\n🎯 RUNNING BTC BACKTEST ({lookback_bars} bars)")
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
            try:
                current_bar = primary_data.iloc[i]
                window_4h = primary_data.iloc[:i+1]
                recent_4h = window_4h.tail(60)

                # Generate domain signals
                domain_signals = self._generate_domain_signals(recent_4h)

                # MTF confluence
                mtf_confluence = self._analyze_mtf_confluence(data_1h, recent_4h, data_1d, current_bar)

                # Signal fusion
                fusion_result = self._fuse_signals(domain_signals, current_bar)

                if fusion_result:
                    signals.append({
                        'timestamp': current_bar.name,
                        'price': current_bar['close'],
                        'direction': fusion_result['direction'],
                        'confidence': fusion_result['confidence'],
                        'strength': fusion_result['strength'],
                        'mtf_aligned': mtf_confluence['aligned']
                    })

                    # Execute trade if MTF aligned
                    if mtf_confluence['aligned']:
                        # Close opposite position
                        if portfolio['position'] != 0:
                            if ((portfolio['position'] > 0 and fusion_result['direction'] == 'short') or
                                (portfolio['position'] < 0 and fusion_result['direction'] == 'long')):
                                self._close_position(portfolio, current_bar, trades)

                        # Open new position
                        if portfolio['position'] == 0:
                            self._open_position(portfolio, current_bar, fusion_result, trades)

            except Exception as e:
                continue

        # Close final position
        if portfolio['position'] != 0:
            self._close_position(portfolio, primary_data.iloc[-1], trades)

        return self._calculate_btc_results(portfolio, trades, signals)

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
            signals['wyckoff'] = self.wyckoff_engine.analyze(data, usdt_stagnation=0.5)  # BTC volatility
        except:
            signals['wyckoff'] = None

        try:
            signals['hob'] = self.hob_engine.detect_hob(data)
        except:
            signals['hob'] = None

        return signals

    def _analyze_mtf_confluence(self, data_1h, data_4h, data_1d, current_bar):
        """Analyze multi-timeframe confluence for BTC"""
        confluence = {
            'aligned': False,
            'trends': {},
            'score': 0.0,
            'level': 'none'
        }

        try:
            current_time = current_bar.name

            # 1H trend (if available)
            if data_1h is not None:
                aligned_1h = data_1h[data_1h.index <= current_time]
                if len(aligned_1h) >= 12:
                    sma_1h_fast = aligned_1h['close'].tail(6).mean()
                    sma_1h_slow = aligned_1h['close'].tail(12).mean()

                    if sma_1h_fast > sma_1h_slow * 1.002:
                        confluence['trends']['1H'] = 'bullish'
                    elif sma_1h_fast < sma_1h_slow * 0.998:
                        confluence['trends']['1H'] = 'bearish'
                    else:
                        confluence['trends']['1H'] = 'neutral'

            # 4H trend
            if len(data_4h) >= 24:
                sma_4h_fast = data_4h['close'].tail(12).mean()
                sma_4h_slow = data_4h['close'].tail(24).mean()

                if sma_4h_fast > sma_4h_slow * 1.005:
                    confluence['trends']['4H'] = 'bullish'
                elif sma_4h_fast < sma_4h_slow * 0.995:
                    confluence['trends']['4H'] = 'bearish'
                else:
                    confluence['trends']['4H'] = 'neutral'

            # 1D trend (if available)
            if data_1d is not None:
                aligned_1d = data_1d[data_1d.index <= current_time]
                if len(aligned_1d) >= 10:
                    sma_1d_fast = aligned_1d['close'].tail(5).mean()
                    sma_1d_slow = aligned_1d['close'].tail(10).mean()

                    if sma_1d_fast > sma_1d_slow * 1.01:
                        confluence['trends']['1D'] = 'bullish'
                    elif sma_1d_fast < sma_1d_slow * 0.99:
                        confluence['trends']['1D'] = 'bearish'
                    else:
                        confluence['trends']['1D'] = 'neutral'

            # Check alignment
            trends = list(confluence['trends'].values())
            if len(trends) >= 2:
                # Count trend alignment
                bullish_count = trends.count('bullish')
                bearish_count = trends.count('bearish')
                neutral_count = trends.count('neutral')

                total_trends = len(trends)

                if bullish_count >= total_trends * 0.6:  # 60% bullish alignment
                    confluence['aligned'] = True
                    confluence['level'] = 'bullish'
                    confluence['score'] = bullish_count / total_trends
                elif bearish_count >= total_trends * 0.6:  # 60% bearish alignment
                    confluence['aligned'] = True
                    confluence['level'] = 'bearish'
                    confluence['score'] = bearish_count / total_trends

        except Exception as e:
            pass

        return confluence

    def _fuse_signals(self, domain_signals, current_bar):
        """Fuse domain signals with calibrated thresholds"""
        active_signals = [s for s in domain_signals.values() if s is not None]

        if len(active_signals) < 1:
            return None

        # Collect directions and confidences
        directions = []
        confidences = []

        for signal in active_signals:
            if hasattr(signal, 'direction') and hasattr(signal, 'confidence'):
                directions.append(signal.direction)
                confidences.append(signal.confidence)

        if not directions or not confidences:
            return None

        # Direction consensus
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

        # Apply calibrated entry criteria
        if (avg_confidence >= self.conf_threshold and
            fusion_strength >= self.strength_threshold):

            return {
                'direction': fusion_direction,
                'confidence': avg_confidence,
                'strength': fusion_strength,
                'engines': len(active_signals)
            }

        return None

    def _open_position(self, portfolio, current_bar, fusion_result, trades):
        """Open new position"""
        try:
            current_price = current_bar['close']
            risk_pct = 0.08  # 8% risk per trade for crypto

            # Enhanced sizing based on signal quality
            base_sizing = 1.0
            confidence_multiplier = min(1.5, fusion_result['confidence'] / 0.25)
            engine_multiplier = min(1.3, 1.0 + (fusion_result['engines'] - 1) * 0.1)

            final_sizing = base_sizing * confidence_multiplier * engine_multiplier
            final_sizing = min(final_sizing, 1.8)  # Cap at 1.8x for crypto

            position_value = portfolio['capital'] * risk_pct * final_sizing

            if fusion_result['direction'] == 'long':
                portfolio['position'] = position_value / current_price
            else:
                portfolio['position'] = -position_value / current_price

            portfolio['entry_price'] = current_price
            portfolio['trades_count'] += 1

            trade = {
                'entry_timestamp': current_bar.name,
                'entry_price': current_price,
                'direction': fusion_result['direction'],
                'confidence': fusion_result['confidence'],
                'strength': fusion_result['strength'],
                'engines': fusion_result['engines'],
                'sizing': final_sizing,
                'capital_at_entry': portfolio['capital']
            }

            trades.append(trade)

        except Exception as e:
            print(f"   ❌ Trade open error: {e}")

    def _close_position(self, portfolio, current_bar, trades):
        """Close current position"""
        try:
            if portfolio['position'] == 0 or not trades:
                return

            current_price = current_bar['close']

            # Calculate PnL
            if portfolio['position'] > 0:  # Long
                pnl = portfolio['position'] * (current_price - portfolio['entry_price'])
            else:  # Short
                pnl = abs(portfolio['position']) * (portfolio['entry_price'] - current_price)

            portfolio['capital'] += pnl

            # Update win/loss tracking
            if pnl > 0:
                portfolio['wins'] += 1
            else:
                portfolio['losses'] += 1

            # Update trade record
            trade = trades[-1]
            trade.update({
                'exit_timestamp': current_bar.name,
                'exit_price': current_price,
                'pnl': pnl,
                'return_pct': (pnl / (abs(portfolio['position']) * portfolio['entry_price'])) * 100
            })

            # Reset position
            portfolio['position'] = 0
            portfolio['entry_price'] = 0

        except Exception as e:
            print(f"   ❌ Trade close error: {e}")

    def _calculate_btc_results(self, portfolio, trades, signals):
        """Calculate BTC backtest results"""
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
        """Analyze current BTC setup for immediate opportunities"""
        print(f"\n🔍 CURRENT BTC SETUP ANALYSIS")
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
        print(f"💰 Current BTC Price: ${current_price:,.2f}")

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

        # Weekly outlook analysis
        print(f"\n📈 IMMEDIATE SETUP ANALYSIS")
        print("-" * 40)

        # Price action analysis
        recent_bars = latest_4h.tail(24)  # Last 4 days
        price_momentum = (current_price - recent_bars['close'].iloc[0]) / recent_bars['close'].iloc[0] * 100
        volatility = recent_bars['close'].std() / recent_bars['close'].mean() * 100

        print(f"🎯 MARKET STRUCTURE:")
        print(f"   • 4-day momentum: {price_momentum:+.1f}%")
        print(f"   • Volatility (CV): {volatility:.1f}%")

        # Support/Resistance levels
        recent_highs = recent_bars['high'].rolling(5).max().iloc[-1]
        recent_lows = recent_bars['low'].rolling(5).min().iloc[-1]

        print(f"📊 KEY LEVELS:")
        print(f"   • Resistance: ${recent_highs:,.2f} (+{((recent_highs - current_price) / current_price * 100):+.1f}%)")
        print(f"   • Support: ${recent_lows:,.2f} ({((recent_lows - current_price) / current_price * 100):+.1f}%)")

        # Engine consensus
        active_engines = sum(1 for signal in domain_signals.values() if signal is not None)
        bullish_engines = sum(1 for signal in domain_signals.values()
                              if signal and getattr(signal, 'direction', None) == 'long')
        bearish_engines = sum(1 for signal in domain_signals.values()
                              if signal and getattr(signal, 'direction', None) == 'short')

        print(f"🤖 ENGINE CONSENSUS:")
        print(f"   • Active: {active_engines}/4 engines")
        print(f"   • Bullish: {bullish_engines} | Bearish: {bearish_engines}")

        if bullish_engines > bearish_engines:
            bias = "BULLISH"
        elif bearish_engines > bullish_engines:
            bias = "BEARISH"
        else:
            bias = "NEUTRAL"

        print(f"   • Current Bias: {bias}")

        # Trade recommendation
        print(f"\n🎯 IMMEDIATE TRADE RECOMMENDATION")
        print("=" * 40)

        if fusion_result and mtf_confluence['aligned']:
            print(f"✅ **IMMEDIATE {fusion_result['direction'].upper()} SETUP CONFIRMED**")
            print(f"   🎯 Entry Zone: ${current_price:,.2f} ± 1%")
            print(f"   📊 Confidence: {fusion_result['confidence']:.1%}")
            print(f"   ⏱️  Time Horizon: 1-3 days")
            print(f"   🔥 Signal Strength: {fusion_result['strength']:.1%}")

            if fusion_result['direction'] == 'long':
                target1 = current_price * 1.03
                target2 = current_price * 1.06
                stop = current_price * 0.96
            else:
                target1 = current_price * 0.97
                target2 = current_price * 0.94
                stop = current_price * 1.04

            print(f"   🎯 Target 1: ${target1:,.2f}")
            print(f"   🎯 Target 2: ${target2:,.2f}")
            print(f"   🛡️  Stop Loss: ${stop:,.2f}")
            print(f"   📈 Risk/Reward: 1:1.5")

            print(f"\n💡 EXECUTION PLAN:")
            print(f"   • Entry: Market order or limit at ${current_price:,.2f}")
            print(f"   • Size: 2-5% of portfolio (crypto volatility)")
            print(f"   • Management: Trail stop after Target 1")

        else:
            reasons = []
            if not fusion_result:
                reasons.append("Insufficient signal strength")
            if not mtf_confluence['aligned']:
                reasons.append("MTF not aligned")

            print(f"❌ NO IMMEDIATE SETUP")
            print(f"   Reasons: {', '.join(reasons)}")
            print(f"   Current Outlook: {bias}")

            if bias == "BULLISH":
                print(f"   Watch For: Break above ${recent_highs:,.2f}")
                print(f"   Strategy: Wait for pullback to ${recent_lows:,.2f} support")
            elif bias == "BEARISH":
                print(f"   Watch For: Break below ${recent_lows:,.2f}")
                print(f"   Strategy: Wait for bounce to ${recent_highs:,.2f} resistance")
            else:
                print(f"   Range: ${recent_lows:,.2f} - ${recent_highs:,.2f}")
                print(f"   Strategy: Wait for range break in either direction")

        return {
            'current_price': current_price,
            'domain_signals': domain_signals,
            'mtf_confluence': mtf_confluence,
            'fusion_result': fusion_result,
            'immediate_setup': fusion_result is not None and mtf_confluence['aligned'],
            'bias': bias,
            'support': recent_lows,
            'resistance': recent_highs,
            'momentum_4d': price_momentum
        }

def main():
    """Main analysis function"""
    analyzer = BTCBullMachineAnalyzer()

    # Load BTC data
    btc_data = analyzer.load_btc_data()
    if not btc_data:
        print("❌ Failed to load BTC data")
        return

    # Run backtest
    backtest_results = analyzer.run_btc_backtest(btc_data, lookback_bars=500)  # Test on 2-3 months for better sample size

    if backtest_results:
        print(f"\n📈 BTC BACKTEST RESULTS")
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
                      f"${trade['entry_price']:,.2f} → ${trade['exit_price']:,.2f} | "
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
    current_analysis = analyzer.analyze_current_setup(btc_data)

    print(f"\n🎉 BTC ANALYSIS COMPLETE")

if __name__ == "__main__":
    main()