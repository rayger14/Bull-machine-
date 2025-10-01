#!/usr/bin/env python3
"""
SOL Analysis with Bull Machine v1.7
Complete multi-timeframe analysis with immediate setup identification and outlook
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

class SOLBullMachineAnalyzer:
    def __init__(self):
        # Load calibrated config
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

        print(f"🚀 BULL MACHINE v{self.config['version']} SOL ANALYSIS")
        print(f"📊 Calibrated: conf={self.conf_threshold}, strength={self.strength_threshold}")

    def load_sol_data(self):
        """Load SOL data from provided charts"""
        print(f"\n📊 LOADING SOL MULTI-TIMEFRAME DATA")
        print("-" * 50)

        data = {}

        # Load 1H data
        try:
            sol_1h = pd.read_csv('/Users/raymondghandchi/Downloads/COINBASE_SOLUSD, 60_59909.csv')
            sol_1h['time'] = pd.to_datetime(sol_1h['time'], unit='s', utc=True)
            sol_1h = sol_1h.set_index('time').sort_index()
            sol_1h.columns = sol_1h.columns.str.lower()

            # Fix volume column for SOL data structure
            if 'total buy volume' in sol_1h.columns and 'total sell volume' in sol_1h.columns:
                sol_1h['volume'] = sol_1h['total buy volume'] + sol_1h['total sell volume']
            elif 'buy+sell v' in sol_1h.columns:
                sol_1h['volume'] = sol_1h['buy+sell v']

            data['1H'] = sol_1h
            print(f"✅ SOL 1H: {len(sol_1h)} bars ({sol_1h.index[0]} to {sol_1h.index[-1]})")
        except Exception as e:
            print(f"⚠️ SOL 1H failed: {e}")
            data['1H'] = None

        # Load 4H data
        try:
            sol_4h = pd.read_csv('/Users/raymondghandchi/Downloads/COINBASE_SOLUSD, 240_5b58e.csv')
            sol_4h['time'] = pd.to_datetime(sol_4h['time'], unit='s', utc=True)
            sol_4h = sol_4h.set_index('time').sort_index()
            sol_4h.columns = sol_4h.columns.str.lower()

            # Fix volume column for SOL data structure
            if 'total buy volume' in sol_4h.columns and 'total sell volume' in sol_4h.columns:
                sol_4h['volume'] = sol_4h['total buy volume'] + sol_4h['total sell volume']
            elif 'buy+sell v' in sol_4h.columns:
                sol_4h['volume'] = sol_4h['buy+sell v']

            data['4H'] = sol_4h
            print(f"✅ SOL 4H: {len(sol_4h)} bars ({sol_4h.index[0]} to {sol_4h.index[-1]})")
        except Exception as e:
            print(f"❌ SOL 4H failed: {e}")
            return None

        # Load 1D data
        try:
            sol_1d = pd.read_csv('/Users/raymondghandchi/Downloads/COINBASE_SOLUSD, 1D_dbe35.csv')
            sol_1d['time'] = pd.to_datetime(sol_1d['time'], unit='s', utc=True)
            sol_1d = sol_1d.set_index('time').sort_index()
            sol_1d.columns = sol_1d.columns.str.lower()

            # Fix volume column for SOL data structure
            if 'total buy volume' in sol_1d.columns and 'total sell volume' in sol_1d.columns:
                sol_1d['volume'] = sol_1d['total buy volume'] + sol_1d['total sell volume']
            elif 'buy+sell v' in sol_1d.columns:
                sol_1d['volume'] = sol_1d['buy+sell v']

            data['1D'] = sol_1d
            print(f"✅ SOL 1D: {len(sol_1d)} bars ({sol_1d.index[0]} to {sol_1d.index[-1]})")
        except Exception as e:
            print(f"⚠️ SOL 1D failed: {e}")
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

    def run_sol_backtest(self, data, lookback_bars=250):
        """Run comprehensive SOL backtest"""
        print(f"\n🎯 RUNNING SOL BACKTEST ({lookback_bars} bars)")
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

        return self._calculate_sol_results(portfolio, trades, signals)

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
            signals['wyckoff'] = self.wyckoff_engine.analyze(data, usdt_stagnation=0.4)  # SOL volatility
        except:
            signals['wyckoff'] = None

        try:
            signals['hob'] = self.hob_engine.detect_hob(data)
        except:
            signals['hob'] = None

        return signals

    def _analyze_mtf_confluence(self, data_1h, data_4h, data_1d, current_bar):
        """Analyze multi-timeframe confluence for SOL"""
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

                    if sma_1h_fast > sma_1h_slow * 1.003:  # SOL more sensitive
                        confluence['trends']['1H'] = 'bullish'
                    elif sma_1h_fast < sma_1h_slow * 0.997:
                        confluence['trends']['1H'] = 'bearish'
                    else:
                        confluence['trends']['1H'] = 'neutral'

            # 4H trend
            if len(data_4h) >= 24:
                sma_4h_fast = data_4h['close'].tail(12).mean()
                sma_4h_slow = data_4h['close'].tail(24).mean()

                if sma_4h_fast > sma_4h_slow * 1.007:  # SOL volatility adjustment
                    confluence['trends']['4H'] = 'bullish'
                elif sma_4h_fast < sma_4h_slow * 0.993:
                    confluence['trends']['4H'] = 'bearish'
                else:
                    confluence['trends']['4H'] = 'neutral'

            # 1D trend (if available)
            if data_1d is not None:
                aligned_1d = data_1d[data_1d.index <= current_time]
                if len(aligned_1d) >= 10:
                    sma_1d_fast = aligned_1d['close'].tail(5).mean()
                    sma_1d_slow = aligned_1d['close'].tail(10).mean()

                    if sma_1d_fast > sma_1d_slow * 1.015:  # Higher threshold for 1D
                        confluence['trends']['1D'] = 'bullish'
                    elif sma_1d_fast < sma_1d_slow * 0.985:
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
            risk_pct = 0.10  # 10% risk per trade for SOL (high volatility)

            # Enhanced sizing based on signal quality
            base_sizing = 1.0
            confidence_multiplier = min(1.6, fusion_result['confidence'] / 0.25)
            engine_multiplier = min(1.4, 1.0 + (fusion_result['engines'] - 1) * 0.1)

            final_sizing = base_sizing * confidence_multiplier * engine_multiplier
            final_sizing = min(final_sizing, 2.0)  # Cap at 2.0x for SOL

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

    def _calculate_sol_results(self, portfolio, trades, signals):
        """Calculate SOL backtest results"""
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
        """Analyze current SOL setup for immediate opportunities"""
        print(f"\n🔍 CURRENT SOL SETUP ANALYSIS")
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
        print(f"💰 Current SOL Price: ${current_price:.2f}")

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

        # Comprehensive SOL outlook analysis
        return self._generate_sol_outlook(data, current_price, domain_signals, mtf_confluence, fusion_result)

    def _generate_sol_outlook(self, data, current_price, domain_signals, mtf_confluence, fusion_result):
        """Generate comprehensive SOL outlook"""

        print(f"\n📈 COMPREHENSIVE SOL OUTLOOK")
        print("=" * 50)

        latest_4h = data['4H'].tail(100)
        latest_1h = data.get('1H')
        latest_1d = data.get('1D')

        # === IMMEDIATE SETUP ANALYSIS ===
        print(f"\n🎯 IMMEDIATE SETUP ANALYSIS")
        print("-" * 40)

        # Price action analysis
        recent_bars = latest_4h.tail(24)  # Last 4 days
        price_momentum_4d = (current_price - recent_bars['close'].iloc[0]) / recent_bars['close'].iloc[0] * 100
        price_momentum_1d = (current_price - recent_bars['close'].iloc[-6]) / recent_bars['close'].iloc[-6] * 100
        volatility = recent_bars['close'].std() / recent_bars['close'].mean() * 100

        print(f"🎯 PRICE MOMENTUM:")
        print(f"   • 4-day momentum: {price_momentum_4d:+.1f}%")
        print(f"   • 1-day momentum: {price_momentum_1d:+.1f}%")
        print(f"   • Volatility (CV): {volatility:.1f}%")

        # Support/Resistance levels
        recent_highs = recent_bars['high'].rolling(7).max().iloc[-1]
        recent_lows = recent_bars['low'].rolling(7).min().iloc[-1]

        print(f"📊 KEY LEVELS:")
        print(f"   • Resistance: ${recent_highs:.2f} (+{((recent_highs - current_price) / current_price * 100):+.1f}%)")
        print(f"   • Support: ${recent_lows:.2f} ({((recent_lows - current_price) / current_price * 100):+.1f}%)")

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
            immediate_bias = "BULLISH"
        elif bearish_engines > bullish_engines:
            immediate_bias = "BEARISH"
        else:
            immediate_bias = "NEUTRAL"

        print(f"   • Immediate Bias: {immediate_bias}")

        # === LONGER-TERM OUTLOOK ===
        print(f"\n🔮 LONGER-TERM OUTLOOK (1-2 Weeks)")
        print("-" * 40)

        # Weekly momentum if 1D data available
        if latest_1d is not None:
            weekly_bars = latest_1d.tail(14)  # Last 2 weeks
            weekly_momentum = (current_price - weekly_bars['close'].iloc[0]) / weekly_bars['close'].iloc[0] * 100

            print(f"📈 WEEKLY STRUCTURE:")
            print(f"   • 2-week momentum: {weekly_momentum:+.1f}%")

            # Weekly trend analysis
            weekly_sma_fast = weekly_bars['close'].tail(5).mean()
            weekly_sma_slow = weekly_bars['close'].tail(10).mean()

            if weekly_sma_fast > weekly_sma_slow * 1.02:
                weekly_trend = "BULLISH"
            elif weekly_sma_fast < weekly_sma_slow * 0.98:
                weekly_trend = "BEARISH"
            else:
                weekly_trend = "NEUTRAL"

            print(f"   • Weekly trend: {weekly_trend}")

            # Higher TF levels
            weekly_high = weekly_bars['high'].max()
            weekly_low = weekly_bars['low'].min()

            print(f"📊 WEEKLY LEVELS:")
            print(f"   • Weekly High: ${weekly_high:.2f} (+{((weekly_high - current_price) / current_price * 100):+.1f}%)")
            print(f"   • Weekly Low: ${weekly_low:.2f} ({((weekly_low - current_price) / current_price * 100):+.1f}%)")

        # === TRADE RECOMMENDATION ===
        print(f"\n🎯 TRADE RECOMMENDATION")
        print("=" * 40)

        if fusion_result and mtf_confluence['aligned']:
            print(f"✅ **IMMEDIATE {fusion_result['direction'].upper()} SETUP CONFIRMED**")
            print(f"   🎯 Entry Zone: ${current_price:.2f} ± 2%")
            print(f"   📊 Confidence: {fusion_result['confidence']:.1%}")
            print(f"   ⏱️  Time Horizon: 1-4 days")
            print(f"   🔥 Signal Strength: {fusion_result['strength']:.1%}")
            print(f"   🚀 SOL Volatility: HIGH - Larger moves expected")

            if fusion_result['direction'] == 'long':
                target1 = current_price * 1.05
                target2 = current_price * 1.10
                target3 = current_price * 1.15
                stop = current_price * 0.94
            else:
                target1 = current_price * 0.95
                target2 = current_price * 0.90
                target3 = current_price * 0.85
                stop = current_price * 1.06

            print(f"   🎯 Target 1: ${target1:.2f} (5%)")
            print(f"   🎯 Target 2: ${target2:.2f} (10%)")
            print(f"   🎯 Target 3: ${target3:.2f} (15%)")
            print(f"   🛡️  Stop Loss: ${stop:.2f}")
            print(f"   📈 Risk/Reward: 1:2.5")

            print(f"\n💡 EXECUTION STRATEGY:")
            print(f"   • Entry: Scale in over 2-4 hours")
            print(f"   • Size: 3-8% of portfolio (SOL volatility)")
            print(f"   • Management: Take 1/3 at Target 1, trail remainder")
            print(f"   • SOL Note: Expect 15-25% moves in crypto")

        else:
            reasons = []
            if not fusion_result:
                reasons.append("Insufficient signal strength")
            if not mtf_confluence['aligned']:
                reasons.append("MTF not aligned")

            print(f"❌ NO IMMEDIATE SETUP")
            print(f"   Reasons: {', '.join(reasons)}")
            print(f"   Current Bias: {immediate_bias}")

            if immediate_bias == "BULLISH":
                print(f"   📈 Strategy: Wait for pullback to ${recent_lows:.2f} support")
                print(f"   🎯 Watch For: Break above ${recent_highs:.2f} with volume")
                print(f"   ⚠️  Risk: Crypto momentum can reverse quickly")
            elif immediate_bias == "BEARISH":
                print(f"   📉 Strategy: Wait for bounce to ${recent_highs:.2f} resistance")
                print(f"   🎯 Watch For: Break below ${recent_lows:.2f} with volume")
                print(f"   ⚠️  Risk: SOL can bounce hard from oversold")
            else:
                print(f"   📊 Range: ${recent_lows:.2f} - ${recent_highs:.2f}")
                print(f"   📈 Strategy: Wait for range break + signal confluence")
                print(f"   🎯 Target: 10-20% move after breakout")

        # === MARKET CONTEXT ===
        print(f"\n🌍 SOL MARKET CONTEXT")
        print("-" * 30)
        print(f"💎 Solana Fundamentals:")
        print(f"   • Ecosystem: DeFi, NFTs, Gaming expansion")
        print(f"   • Technical: High throughput, low fees")
        print(f"   • Volatility: Higher than BTC/ETH - bigger moves")
        print(f"   • Correlation: Strong with broader crypto")

        return {
            'current_price': current_price,
            'domain_signals': domain_signals,
            'mtf_confluence': mtf_confluence,
            'fusion_result': fusion_result,
            'immediate_setup': fusion_result is not None and mtf_confluence['aligned'],
            'immediate_bias': immediate_bias,
            'support': recent_lows,
            'resistance': recent_highs,
            'momentum_4d': price_momentum_4d,
            'momentum_1d': price_momentum_1d,
            'volatility': volatility
        }

def main():
    """Main analysis function"""
    analyzer = SOLBullMachineAnalyzer()

    # Load SOL data
    sol_data = analyzer.load_sol_data()
    if not sol_data:
        print("❌ Failed to load SOL data")
        return

    # Run backtest
    backtest_results = analyzer.run_sol_backtest(sol_data, lookback_bars=250)

    if backtest_results:
        print(f"\n📈 SOL BACKTEST RESULTS")
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

    # Analyze current setup with comprehensive outlook
    current_analysis = analyzer.analyze_current_setup(sol_data)

    print(f"\n🎉 SOL ANALYSIS COMPLETE")

if __name__ == "__main__":
    main()