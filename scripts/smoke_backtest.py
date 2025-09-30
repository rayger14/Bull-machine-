#!/usr/bin/env python3
"""
Smoke Backtest for Bull Machine v1.7
Fast validation engine for Tier 1 testing
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import pandas as pd
import numpy as np
import time
import signal
from datetime import datetime

from engine.smc.smc_engine import SMCEngine
from engine.wyckoff.wyckoff_engine import WyckoffEngine
from engine.liquidity.hob import HOBDetector
from engine.momentum.momentum_engine import MomentumEngine

class TimeoutError(Exception):
    pass

def timeout_handler(signum, frame):
    raise TimeoutError("Smoke test timed out")

class SmokeBacktest:
    def __init__(self, config):
        """Initialize smoke backtest with config"""
        self.config = config
        self.health_metrics = {
            'macro_veto_count': 0,
            'smc_2hit_count': 0,
            'hob_signals': 0,
            'delta_breaches': 0,
            'total_bars': 0,
            'total_signals': 0
        }

        # Initialize engines
        try:
            self.smc_engine = SMCEngine(config['domains']['smc'])
            self.momentum_engine = MomentumEngine(config['domains']['momentum'])
            self.wyckoff_engine = WyckoffEngine(config['domains']['wyckoff'])
            self.hob_engine = HOBDetector(config['domains']['liquidity']['hob_detection'])
        except Exception as e:
            raise RuntimeError(f"Engine initialization failed: {e}")

        # Get thresholds
        self.conf_threshold = config['fusion']['calibration_thresholds']['confidence']
        self.strength_threshold = config['fusion']['calibration_thresholds']['strength']

    def run(self, primary_data, htf_data=None, timeout=600):
        """Run smoke test with timeout protection"""

        # Set up timeout
        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(timeout)

        try:
            result = self._execute_smoke_test(primary_data, htf_data)
            signal.alarm(0)  # Cancel timeout
            return result

        except TimeoutError:
            return {
                'status': 'fail',
                'error': f'Timeout after {timeout}s',
                'partial_results': self._get_partial_results()
            }
        except Exception as e:
            signal.alarm(0)
            return {
                'status': 'fail',
                'error': str(e)
            }

    def _execute_smoke_test(self, primary_data, htf_data):
        """Execute the actual smoke test"""
        start_time = time.time()

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

        # Process data with early stopping checks
        for i in range(50, min(len(primary_data), 300)):  # Cap at 300 bars for smoke test
            try:
                current_bar = primary_data.iloc[i]
                window_data = primary_data.iloc[:i+1]
                recent_data = window_data.tail(60)

                self.health_metrics['total_bars'] += 1

                # Early abort conditions
                abort_reason = self._should_abort()
                if abort_reason:
                    return {
                        'status': 'fail',
                        'error': f'Early abort: {abort_reason}',
                        'bars_processed': i,
                        'health_metrics': self.health_metrics
                    }

                # Generate domain signals
                domain_signals = self._generate_domain_signals(recent_data)

                # Check for HOB signals (health metric)
                if domain_signals.get('hob'):
                    self.health_metrics['hob_signals'] += 1

                # MTF confluence check
                mtf_aligned = self._check_mtf_confluence(htf_data, current_bar, recent_data)

                # Signal fusion
                fusion_result = self._fuse_signals(domain_signals)

                if fusion_result:
                    self.health_metrics['total_signals'] += 1
                    signals.append({
                        'timestamp': current_bar.name,
                        'direction': fusion_result['direction'],
                        'confidence': fusion_result['confidence'],
                        'strength': fusion_result['strength']
                    })

                    # Execute trade if MTF aligned
                    if mtf_aligned:
                        # Close opposite position
                        if portfolio['position'] != 0:
                            if ((portfolio['position'] > 0 and fusion_result['direction'] == 'short') or
                                (portfolio['position'] < 0 and fusion_result['direction'] == 'long')):
                                self._close_position(portfolio, current_bar, trades)

                        # Open new position
                        if portfolio['position'] == 0:
                            self._open_position(portfolio, current_bar, fusion_result, trades)
                    else:
                        self.health_metrics['macro_veto_count'] += 1

                # Check SMC 2-hit rate
                if domain_signals.get('smc') and hasattr(domain_signals['smc'], 'confidence'):
                    if domain_signals['smc'].confidence >= 0.5:  # High confidence SMC
                        self.health_metrics['smc_2hit_count'] += 1

            except Exception as e:
                continue  # Skip problematic bars

        # Close final position
        if portfolio['position'] != 0:
            self._close_position(portfolio, primary_data.iloc[-1], trades)

        # Calculate results
        return self._calculate_smoke_results(portfolio, trades, signals, time.time() - start_time)

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
            signals['wyckoff'] = self.wyckoff_engine.analyze(data, usdt_stagnation=0.5)
        except:
            signals['wyckoff'] = None

        try:
            signals['hob'] = self.hob_engine.detect_hob(data)
        except:
            signals['hob'] = None

        return signals

    def _check_mtf_confluence(self, htf_data, current_bar, primary_data):
        """Simple MTF confluence check"""
        if htf_data is None:
            return True  # Default to aligned if no HTF data

        try:
            current_time = current_bar.name
            aligned_htf = htf_data[htf_data.index <= current_time]

            if len(aligned_htf) < 10 or len(primary_data) < 24:
                return False

            # Primary timeframe trend
            sma_fast = primary_data['close'].tail(12).mean()
            sma_slow = primary_data['close'].tail(24).mean()

            # HTF trend
            htf_fast = aligned_htf['close'].tail(5).mean()
            htf_slow = aligned_htf['close'].tail(10).mean()

            # Check alignment
            primary_bullish = sma_fast > sma_slow * 1.005
            primary_bearish = sma_fast < sma_slow * 0.995

            htf_bullish = htf_fast > htf_slow * 1.01
            htf_bearish = htf_fast < htf_slow * 0.99

            return (primary_bullish and htf_bullish) or (primary_bearish and htf_bearish)

        except:
            return False

    def _fuse_signals(self, domain_signals):
        """Fuse domain signals"""
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

        # Apply thresholds
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
            risk_pct = 0.075  # 7.5% risk

            position_value = portfolio['capital'] * risk_pct

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
                'confidence': fusion_result['confidence']
            }

            trades.append(trade)

        except Exception:
            pass

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

            # Update win/loss
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

        except Exception:
            pass

    def _should_abort(self):
        """Check early abort conditions"""
        # No entries after significant bars
        if self.health_metrics['total_bars'] > 100 and self.health_metrics['total_signals'] == 0:
            return "no_signals_generated"

        # Excessive macro vetoing
        if self.health_metrics['total_bars'] > 50:
            macro_veto_rate = self.health_metrics['macro_veto_count'] / max(1, self.health_metrics['total_signals'])
            if macro_veto_rate > 0.25:
                return "excessive_macro_veto"

        # Delta cap breaches
        if self.health_metrics['delta_breaches'] > 0:
            return "delta_cap_breach"

        return None

    def _calculate_smoke_results(self, portfolio, trades, signals, duration):
        """Calculate smoke test results"""
        completed_trades = [t for t in trades if 'exit_price' in t]

        total_return = (portfolio['capital'] - 100000) / 100000 * 100

        # Calculate health metrics
        macro_veto_rate = (self.health_metrics['macro_veto_count'] /
                          max(1, self.health_metrics['total_signals']))

        smc_2hit_rate = (self.health_metrics['smc_2hit_count'] /
                        max(1, self.health_metrics['total_bars']))

        hob_relevance = (self.health_metrics['hob_signals'] /
                        max(1, self.health_metrics['total_signals']))

        results = {
            'status': 'pass',
            'total_return': total_return,
            'total_trades': len(completed_trades),
            'total_signals': len(signals),
            'win_rate': (portfolio['wins'] / max(1, len(completed_trades))) * 100,
            'final_capital': portfolio['capital'],
            'duration': duration,
            'bars_processed': self.health_metrics['total_bars'],

            # Health band metrics
            'macro_veto_rate': macro_veto_rate,
            'smc_2hit_rate': smc_2hit_rate,
            'hob_relevance': hob_relevance,
            'delta_breaches': self.health_metrics['delta_breaches'],

            # Trade details
            'trades': completed_trades,
            'signals': signals
        }

        return results

    def _get_partial_results(self):
        """Get partial results for timeout cases"""
        return {
            'bars_processed': self.health_metrics['total_bars'],
            'signals_generated': self.health_metrics['total_signals'],
            'health_metrics': self.health_metrics
        }

def main():
    """Test smoke backtest"""
    import json
    from engine.io.tradingview_loader import load_tv

    # Load config
    with open('../configs/v170/assets/ETH_v17_tuned.json', 'r') as f:
        config = json.load(f)

    # Load test data
    eth_4h = load_tv('ETH_4H').tail(200)  # Last 200 bars
    eth_1d = load_tv('ETH_1D').tail(100)  # Last 100 bars

    # Run smoke test
    smoke = SmokeBacktest(config)
    results = smoke.run(eth_4h, eth_1d, timeout=300)

    print("🧪 SMOKE TEST RESULTS")
    print("=" * 30)
    print(f"Status: {results['status']}")
    print(f"Duration: {results.get('duration', 0):.1f}s")
    print(f"Return: {results.get('total_return', 0):+.2f}%")
    print(f"Trades: {results.get('total_trades', 0)}")
    print(f"Macro Veto: {results.get('macro_veto_rate', 0):.1%}")

if __name__ == "__main__":
    main()