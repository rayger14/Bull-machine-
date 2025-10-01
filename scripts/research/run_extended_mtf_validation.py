#!/usr/bin/env python3
"""
Bull Machine v1.7 Extended Multi-Timeframe Validation
Complete 300-bar validation with proper MTF hierarchy: 1H → 4H → 1D
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
from engine.io.tradingview_loader import load_tv
from engine.smc.smc_engine import SMCEngine
from engine.wyckoff.wyckoff_engine import WyckoffEngine
from engine.liquidity.hob import HOBDetector
from engine.momentum.momentum_engine import MomentumEngine

class ExtendedMTFValidator:
    """Extended Multi-Timeframe Validator with complete hierarchy"""

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

        # Results tracking
        self.trades = []
        self.signals = []
        self.mtf_analysis = []

        print(f"🚀 BULL MACHINE v{self.config['version']} EXTENDED MTF VALIDATION")
        print(f"📊 Calibrated: conf={self.conf_threshold}, strength={self.strength_threshold}")

    def load_complete_mtf_data(self):
        """Load complete multi-timeframe dataset"""
        print(f"\n📈 LOADING COMPLETE MULTI-TIMEFRAME DATA")
        print("-" * 50)

        data = {}

        # Primary timeframe (4H)
        try:
            eth_4h = load_tv('ETH_4H')
            data['4H'] = eth_4h
            print(f"✅ ETH 4H: {len(eth_4h)} bars ({eth_4h.index[0]} to {eth_4h.index[-1]})")
        except Exception as e:
            print(f"❌ ETH 4H failed: {e}")
            return None

        # Lower timeframe (1H)
        try:
            eth_1h = load_tv('ETH_1H')
            data['1H'] = eth_1h
            print(f"✅ ETH 1H: {len(eth_1h)} bars ({eth_1h.index[0]} to {eth_1h.index[-1]})")
        except Exception as e:
            print(f"⚠️ ETH 1H not available: {e}")
            data['1H'] = None

        # Higher timeframe (1D)
        try:
            eth_1d = load_tv('ETH_1D')
            data['1D'] = eth_1d
            print(f"✅ ETH 1D: {len(eth_1d)} bars ({eth_1d.index[0]} to {eth_1d.index[-1]})")
        except Exception as e:
            print(f"⚠️ ETH 1D not available: {e}")
            data['1D'] = None

        # Use last 300 bars from 4H for extended validation
        if data['4H'] is not None:
            validation_4h = data['4H'].tail(300)
            data['4H_validation'] = validation_4h

            # Align other timeframes
            start_time = validation_4h.index[0]
            end_time = validation_4h.index[-1]

            if data['1H'] is not None:
                data['1H_validation'] = data['1H'][(data['1H'].index >= start_time) & (data['1H'].index <= end_time)]

            if data['1D'] is not None:
                data['1D_validation'] = data['1D'][(data['1D'].index >= start_time) & (data['1D'].index <= end_time)]

            print(f"\n🔍 VALIDATION DATASET (300 bars 4H):")
            print(f"   Period: {start_time} to {end_time}")
            print(f"   4H bars: {len(validation_4h)}")
            if data['1H'] is not None:
                print(f"   1H bars: {len(data.get('1H_validation', []))}")
            if data['1D'] is not None:
                print(f"   1D bars: {len(data.get('1D_validation', []))}")

        return data

    def analyze_mtf_confluence(self, data_1h, data_4h, data_1d, current_4h_bar):
        """Analyze complete multi-timeframe confluence"""

        confluence = {
            'timestamp': current_4h_bar.name,
            'price': current_4h_bar['close'],
            'trends': {},
            'alignment_score': 0.0,
            'confluence_level': 'none',
            'aligned': False
        }

        current_time = current_4h_bar.name

        try:
            # 1H trend analysis (if available)
            if data_1h is not None and len(data_1h) > 20:
                aligned_1h = data_1h[data_1h.index <= current_time]
                if len(aligned_1h) >= 20:
                    sma_1h_fast = aligned_1h['close'].tail(12).mean()  # 12H
                    sma_1h_slow = aligned_1h['close'].tail(24).mean()  # 24H (1 day)

                    if sma_1h_fast > sma_1h_slow * 1.003:
                        confluence['trends']['1H'] = 'bullish'
                    elif sma_1h_fast < sma_1h_slow * 0.997:
                        confluence['trends']['1H'] = 'bearish'
                    else:
                        confluence['trends']['1H'] = 'neutral'

            # 4H trend analysis (primary)
            if len(data_4h) >= 24:
                sma_4h_fast = data_4h['close'].tail(6).mean()   # 24H
                sma_4h_slow = data_4h['close'].tail(12).mean()  # 48H

                if sma_4h_fast > sma_4h_slow * 1.005:
                    confluence['trends']['4H'] = 'bullish'
                elif sma_4h_fast < sma_4h_slow * 0.995:
                    confluence['trends']['4H'] = 'bearish'
                else:
                    confluence['trends']['4H'] = 'neutral'

            # 1D trend analysis (if available)
            if data_1d is not None and len(data_1d) > 10:
                aligned_1d = data_1d[data_1d.index <= current_time]
                if len(aligned_1d) >= 10:
                    sma_1d_fast = aligned_1d['close'].tail(5).mean()   # 5 days
                    sma_1d_slow = aligned_1d['close'].tail(10).mean()  # 10 days

                    if sma_1d_fast > sma_1d_slow * 1.01:
                        confluence['trends']['1D'] = 'bullish'
                    elif sma_1d_fast < sma_1d_slow * 0.99:
                        confluence['trends']['1D'] = 'bearish'
                    else:
                        confluence['trends']['1D'] = 'neutral'

            # Calculate alignment score
            trends = [t for t in confluence['trends'].values() if t != 'neutral']

            if len(trends) >= 2:
                # Check for alignment
                bullish_trends = [t for t in trends if t == 'bullish']
                bearish_trends = [t for t in trends if t == 'bearish']

                if len(bullish_trends) == len(trends):
                    confluence['confluence_level'] = 'full_bullish'
                    confluence['alignment_score'] = 1.0
                    confluence['aligned'] = True
                elif len(bearish_trends) == len(trends):
                    confluence['confluence_level'] = 'full_bearish'
                    confluence['alignment_score'] = 1.0
                    confluence['aligned'] = True
                elif len(bullish_trends) > len(bearish_trends):
                    confluence['confluence_level'] = 'partial_bullish'
                    confluence['alignment_score'] = len(bullish_trends) / len(trends)
                    confluence['aligned'] = confluence['alignment_score'] >= 0.67
                elif len(bearish_trends) > len(bullish_trends):
                    confluence['confluence_level'] = 'partial_bearish'
                    confluence['alignment_score'] = len(bearish_trends) / len(trends)
                    confluence['aligned'] = confluence['alignment_score'] >= 0.67

        except Exception as e:
            print(f"MTF analysis error: {e}")

        return confluence

    def run_extended_validation(self):
        """Run extended 300-bar validation with complete MTF"""

        print(f"\n🎯 STARTING EXTENDED 300-BAR VALIDATION")
        print("=" * 60)

        # Load data
        mtf_data = self.load_complete_mtf_data()
        if not mtf_data or '4H_validation' not in mtf_data:
            print("❌ Failed to load validation data")
            return None

        validation_4h = mtf_data['4H_validation']
        validation_1h = mtf_data.get('1H_validation')
        validation_1d = mtf_data.get('1D_validation')

        # Initialize portfolio
        portfolio = {
            'capital': 100000.0,
            'position': 0.0,
            'entry_price': 0.0,
            'entry_time': None,
            'trades_count': 0,
            'wins': 0,
            'losses': 0,
            'total_pnl': 0.0
        }

        print(f"💰 Starting capital: ${portfolio['capital']:,.2f}")
        print(f"🎯 Risk per trade: 7.5%")

        # Process validation data
        signals_generated = 0
        mtf_aligned_signals = 0
        trades_executed = 0

        print(f"\n⚡ PROCESSING {len(validation_4h)} BARS WITH COMPLETE MTF")

        for i in range(50, len(validation_4h)):  # Start after warmup
            current_bar = validation_4h.iloc[i]
            historical_4h = validation_4h.iloc[:i+1]
            recent_4h = historical_4h.tail(60)

            try:
                # Generate domain signals
                domain_signals = self._generate_domain_signals(recent_4h)

                # Multi-timeframe confluence analysis
                mtf_confluence = self.analyze_mtf_confluence(
                    validation_1h, historical_4h, validation_1d, current_bar
                )
                self.mtf_analysis.append(mtf_confluence)

                # Signal fusion
                active_signals = [s for s in domain_signals.values() if s is not None]

                if len(active_signals) >= 1:
                    # Collect signals
                    directions = []
                    confidences = []
                    engine_details = []

                    for engine, signal in domain_signals.items():
                        if signal and hasattr(signal, 'direction') and hasattr(signal, 'confidence'):
                            directions.append(signal.direction)
                            confidences.append(signal.confidence)
                            engine_details.append(f"{engine}({signal.direction[:1].upper()}:{signal.confidence:.2f})")

                    if directions and confidences:
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
                            continue

                        avg_confidence = np.mean(confidences)

                        # Check entry criteria
                        if avg_confidence >= self.conf_threshold and fusion_strength >= self.strength_threshold:
                            signals_generated += 1

                            # Record signal
                            signal_record = {
                                'timestamp': current_bar.name,
                                'price': current_bar['close'],
                                'direction': fusion_direction,
                                'confidence': avg_confidence,
                                'strength': fusion_strength,
                                'engines': len(active_signals),
                                'engine_details': ' + '.join(engine_details),
                                'mtf_confluence': mtf_confluence,
                                'aligned': mtf_confluence['aligned']
                            }
                            self.signals.append(signal_record)

                            # MTF filter
                            if mtf_confluence['aligned']:
                                mtf_aligned_signals += 1

                                # Execute trade
                                trade_executed = self._execute_trade(
                                    portfolio, current_bar, fusion_direction,
                                    avg_confidence, mtf_confluence
                                )

                                if trade_executed:
                                    trades_executed += 1

                                # Print signal details
                                if signals_generated <= 10:  # Limit output
                                    print(f"\n🎯 SIGNAL #{signals_generated} at {current_bar.name}")
                                    print(f"   Direction: {fusion_direction.upper()}")
                                    print(f"   Confidence: {avg_confidence:.3f}")
                                    print(f"   MTF: {mtf_confluence['confluence_level']} (score: {mtf_confluence['alignment_score']:.2f})")
                                    print(f"   Trends: {mtf_confluence['trends']}")
                                    print(f"   Engines: {' + '.join(engine_details)}")

            except Exception as e:
                continue

        # Close final position
        if portfolio['position'] != 0:
            self._close_position(portfolio, validation_4h.iloc[-1])

        # Calculate results
        results = self._calculate_extended_results(
            portfolio, signals_generated, mtf_aligned_signals, trades_executed
        )

        return results

    def _generate_domain_signals(self, recent_data):
        """Generate signals from all domain engines"""
        signals = {}

        try:
            signals['smc'] = self.smc_engine.analyze(recent_data)
        except:
            signals['smc'] = None

        try:
            signals['momentum'] = self.momentum_engine.analyze(recent_data)
        except:
            signals['momentum'] = None

        try:
            signals['wyckoff'] = self.wyckoff_engine.analyze(recent_data, usdt_stagnation=0.5)
        except:
            signals['wyckoff'] = None

        try:
            signals['hob'] = self.hob_engine.detect_hob(recent_data)
        except:
            signals['hob'] = None

        return signals

    def _execute_trade(self, portfolio, current_bar, direction, confidence, mtf_confluence):
        """Execute trade with MTF-enhanced sizing"""

        # Close opposite position
        if portfolio['position'] != 0:
            if ((portfolio['position'] > 0 and direction == 'short') or
                (portfolio['position'] < 0 and direction == 'long')):
                self._close_position(portfolio, current_bar)

        # Open new position
        if portfolio['position'] == 0:
            current_price = current_bar['close']

            # Enhanced sizing based on MTF strength
            base_risk = 0.075  # 7.5%
            mtf_multiplier = 1.0 + (mtf_confluence['alignment_score'] * 0.5)  # Up to 1.5x
            confidence_multiplier = min(1.3, confidence / 0.25)

            final_sizing = base_risk * mtf_multiplier * confidence_multiplier
            final_sizing = min(final_sizing, 0.15)  # Cap at 15%

            position_value = portfolio['capital'] * final_sizing

            if direction == 'long':
                portfolio['position'] = position_value / current_price
            else:
                portfolio['position'] = -position_value / current_price

            portfolio['entry_price'] = current_price
            portfolio['entry_time'] = current_bar.name
            portfolio['trades_count'] += 1

            trade = {
                'trade_id': portfolio['trades_count'],
                'entry_timestamp': current_bar.name,
                'entry_price': current_price,
                'direction': direction,
                'confidence': confidence,
                'mtf_confluence': mtf_confluence,
                'sizing_multiplier': final_sizing / base_risk,
                'capital_at_entry': portfolio['capital']
            }

            self.trades.append(trade)
            return True

        return False

    def _close_position(self, portfolio, current_bar):
        """Close current position"""
        if portfolio['position'] == 0 or not self.trades:
            return

        current_price = current_bar['close']

        # Calculate PnL
        if portfolio['position'] > 0:
            pnl = portfolio['position'] * (current_price - portfolio['entry_price'])
        else:
            pnl = abs(portfolio['position']) * (portfolio['entry_price'] - current_price)

        portfolio['capital'] += pnl
        portfolio['total_pnl'] += pnl

        if pnl > 0:
            portfolio['wins'] += 1
        else:
            portfolio['losses'] += 1

        # Update trade record
        trade = self.trades[-1]
        trade.update({
            'exit_timestamp': current_bar.name,
            'exit_price': current_price,
            'pnl': pnl,
            'return_pct': (pnl / (abs(portfolio['position']) * portfolio['entry_price'])) * 100
        })

        # Reset position
        portfolio['position'] = 0
        portfolio['entry_price'] = 0
        portfolio['entry_time'] = None

    def _calculate_extended_results(self, portfolio, signals_generated, mtf_aligned, trades_executed):
        """Calculate comprehensive extended validation results"""

        completed_trades = [t for t in self.trades if 'exit_price' in t]

        results = {
            'signals_generated': signals_generated,
            'mtf_aligned_signals': mtf_aligned,
            'trades_executed': trades_executed,
            'portfolio': portfolio
        }

        if completed_trades:
            returns = [t['return_pct'] for t in completed_trades]
            wins = [t for t in completed_trades if t['pnl'] > 0]

            results.update({
                'total_trades': len(completed_trades),
                'win_rate': len(wins) / len(completed_trades) * 100,
                'total_return': (portfolio['capital'] - 100000) / 100000 * 100,
                'avg_return': np.mean(returns),
                'best_trade': max(returns) if returns else 0,
                'worst_trade': min(returns) if returns else 0,
                'mtf_filter_rate': mtf_aligned / max(1, signals_generated) * 100
            })

        # MTF analysis summary
        if self.mtf_analysis:
            alignment_scores = [m['alignment_score'] for m in self.mtf_analysis if m['alignment_score'] > 0]
            confluence_levels = {}
            for m in self.mtf_analysis:
                level = m['confluence_level']
                confluence_levels[level] = confluence_levels.get(level, 0) + 1

            results['mtf_stats'] = {
                'avg_alignment_score': np.mean(alignment_scores) if alignment_scores else 0,
                'confluence_distribution': confluence_levels,
                'total_mtf_analysis': len(self.mtf_analysis)
            }

        return results

    def print_extended_results(self, results):
        """Print comprehensive extended validation results"""

        print(f"\n" + "="*80)
        print("🎯 BULL MACHINE v1.7 EXTENDED MTF VALIDATION RESULTS")
        print("="*80)

        # Signal generation
        print(f"\n📊 SIGNAL GENERATION (300 bars)")
        print("-" * 40)
        print(f"Total signals: {results['signals_generated']}")
        print(f"MTF aligned signals: {results['mtf_aligned_signals']}")
        print(f"Trades executed: {results['trades_executed']}")
        print(f"MTF filter rate: {results.get('mtf_filter_rate', 0):.1f}%")

        # Performance
        if 'total_trades' in results:
            print(f"\n💰 TRADING PERFORMANCE")
            print("-" * 40)
            print(f"Total trades: {results['total_trades']}")
            print(f"Win rate: {results['win_rate']:.1f}%")
            print(f"Total return: {results['total_return']:+.2f}%")
            print(f"Average return: {results['avg_return']:+.2f}%")
            print(f"Best trade: {results['best_trade']:+.1f}%")
            print(f"Worst trade: {results['worst_trade']:+.1f}%")
            print(f"Final capital: ${results['portfolio']['capital']:,.2f}")

        # MTF Analysis
        if 'mtf_stats' in results:
            mtf_stats = results['mtf_stats']
            print(f"\n🔄 MULTI-TIMEFRAME ANALYSIS")
            print("-" * 40)
            print(f"Average alignment score: {mtf_stats['avg_alignment_score']:.2f}")
            print(f"Total MTF analyses: {mtf_stats['total_mtf_analysis']}")

            print(f"\nConfluence distribution:")
            for level, count in mtf_stats['confluence_distribution'].items():
                pct = count / mtf_stats['total_mtf_analysis'] * 100
                print(f"  • {level}: {count} ({pct:.1f}%)")

        # Recent trades
        if self.trades:
            print(f"\n📋 RECENT TRADES (Last 5)")
            print("-" * 40)
            recent_trades = [t for t in self.trades if 'exit_price' in t][-5:]
            for i, trade in enumerate(recent_trades, 1):
                mtf_level = trade['mtf_confluence']['confluence_level']
                print(f"{i}. {trade['direction'].upper()} @ ${trade['entry_price']:.2f} → "
                      f"${trade['exit_price']:.2f} | {trade['return_pct']:+.1f}% | "
                      f"MTF: {mtf_level} | Size: {trade['sizing_multiplier']:.1f}x")

def main():
    """Run extended MTF validation"""

    try:
        validator = ExtendedMTFValidator('configs/v170/assets/ETH_v17_tuned.json')
        results = validator.run_extended_validation()

        if results:
            validator.print_extended_results(results)
            print(f"\n🎉 EXTENDED VALIDATION COMPLETE")
        else:
            print(f"❌ Validation failed")

    except Exception as e:
        print(f"❌ Validation error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()