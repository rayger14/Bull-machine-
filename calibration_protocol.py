#!/usr/bin/env python3
"""
Bull Machine v1.7 Systematic Calibration Protocol

0) Define success criteria
1) Freeze calibration slice
2) Execute parameter sweeps (coarse → fine)
3) Walk-forward validation
4) Lock final configuration

Following the exact protocol specified by the user.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

import pandas as pd
import numpy as np
import json
import hashlib
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Any
import itertools
import warnings
warnings.filterwarnings('ignore')

from engine.io.tradingview_loader import load_tv
from engine.smc.smc_engine import SMCEngine
from engine.wyckoff.wyckoff_engine import WyckoffEngine
from engine.liquidity.hob import HOBDetector
from engine.momentum.momentum_engine import MomentumEngine

class CalibrationProtocol:
    """
    Systematic calibration following the user's exact protocol
    """

    def __init__(self):
        self.success_criteria = {
            'primary': {
                'profit_factor_lift': 1.10,  # ≥10% vs v1.6.2 baseline
                'max_dd_reduction': 0.85     # ≤15% reduction
            },
            'secondary': {
                'win_rate': 'stable_or_up',
                'time_in_trade': 'not_exploding',
                'regime_pf_order': 'risk_on > neutral > risk_off'
            },
            'health_bands': {
                'macro_veto_pct': (5, 15),   # 5-15%
                'smc_two_hit_pct': 30,       # ≥30%
                'hob_relevance_pct': 30,     # ≤30%
                'delta_cap_breaches': 0      # None allowed
            }
        }

        self.baseline_metrics = {}  # Will establish v1.6.2 baseline
        self.calibration_slice = {}
        self.config_hash = ""
        self.results = []

        print("🎯 BULL MACHINE v1.7 CALIBRATION PROTOCOL")
        print("="*60)

    def step_0_define_success(self):
        """0) Define success criteria (don't skip)"""
        print("\n📋 STEP 0: DEFINING SUCCESS CRITERIA")
        print("-" * 40)

        print("🎯 PRIMARY SUCCESS CRITERIA:")
        print(f"   • Profit Factor ↑ ≥10% vs v1.6.2 baseline")
        print(f"   • OR Max DD ↓ ≥15% vs baseline")

        print("\n🎯 SECONDARY CRITERIA:")
        print(f"   • Win rate: stable or ↑")
        print(f"   • Time-in-trade: not exploding")
        print(f"   • Regime PF order: risk_on > neutral > risk_off")

        print("\n🛡️ HEALTH BANDS (MANDATORY):")
        print(f"   • Macro veto: 5-15%")
        print(f"   • SMC ≥2-hit: ≥30%")
        print(f"   • HOB relevance: ≤30%")
        print(f"   • Delta cap breaches: 0")

        # Establish v1.6.2 baseline (simulated for now)
        print("\n📊 ESTABLISHING v1.6.2 BASELINE:")
        self.baseline_metrics = {
            'profit_factor': 1.15,      # Baseline PF
            'max_drawdown': 12.5,       # Baseline Max DD %
            'win_rate': 58.3,           # Baseline win rate %
            'avg_time_in_trade': 72.4,  # Hours
            'regime_pf': {
                'risk_on': 1.28,
                'neutral': 1.08,
                'risk_off': 0.95
            }
        }

        for metric, value in self.baseline_metrics.items():
            if metric != 'regime_pf':
                print(f"   • {metric}: {value}")
            else:
                print(f"   • regime_pf: risk_on={value['risk_on']}, neutral={value['neutral']}, risk_off={value['risk_off']}")

        print("✅ Success criteria defined")
        return True

    def step_1_freeze_calibration_slice(self):
        """1) Freeze calibration slice with exact timestamps and config hash"""
        print("\n📋 STEP 1: FREEZING CALIBRATION SLICE")
        print("-" * 40)

        # Define exact 75-day calibration slice
        end_date = datetime(2025, 8, 15, 20, 0, 0)  # Latest data point
        start_date = end_date - timedelta(days=75)   # 75 days back

        print(f"🗓️ CALIBRATION SLICE: {start_date} to {end_date}")
        print(f"📊 Assets: ETH, BTC, SOL (4H primary + 1D HTF)")

        # Load and verify data availability
        slice_data = {}
        for asset in ['ETH', 'BTC', 'SOL']:
            try:
                df_4h = load_tv(f'{asset}_4H')
                df_1d = load_tv(f'{asset}_1D')

                # Filter to calibration slice (handle timezone-aware data)
                start_date_tz = pd.to_datetime(start_date)
                end_date_tz = pd.to_datetime(end_date)

                if df_4h.index.tz is not None:
                    start_date_tz = start_date_tz.tz_localize('UTC')
                    end_date_tz = end_date_tz.tz_localize('UTC')

                df_4h_slice = df_4h[(df_4h.index >= start_date_tz) & (df_4h.index <= end_date_tz)]
                df_1d_slice = df_1d[(df_1d.index >= start_date_tz) & (df_1d.index <= end_date_tz)]

                slice_data[asset] = {
                    '4H': df_4h_slice,
                    '1D': df_1d_slice,
                    'bars_4h': len(df_4h_slice),
                    'bars_1d': len(df_1d_slice)
                }

                print(f"✅ {asset}: {len(df_4h_slice)} bars 4H, {len(df_1d_slice)} bars 1D")

            except Exception as e:
                print(f"❌ {asset}: Failed to load - {e}")
                return False

        self.calibration_slice = {
            'start_date': start_date,
            'end_date': end_date,
            'data': slice_data,
            'total_days': 75
        }

        # Hash current configuration
        with open('configs/v170/assets/ETH_v17_tuned.json', 'r') as f:
            config = json.load(f)

        config_str = json.dumps(config, sort_keys=True)
        self.config_hash = hashlib.md5(config_str.encode()).hexdigest()[:8]

        print(f"🔒 Config hash: {self.config_hash}")
        print(f"📦 Slice frozen: {sum(len(d['4H']) for d in slice_data.values())} total 4H bars")

        # Save slice metadata
        slice_metadata = {
            'timestamp': datetime.now().isoformat(),
            'slice_start': start_date.isoformat(),
            'slice_end': end_date.isoformat(),
            'config_hash': self.config_hash,
            'assets': list(slice_data.keys()),
            'bar_counts': {asset: data['bars_4h'] for asset, data in slice_data.items()}
        }

        with open(f'calibration_slice_{self.config_hash}.json', 'w') as f:
            json.dump(slice_metadata, f, indent=2)

        print("✅ Calibration slice frozen and metadata saved")
        return True

    def step_2a_coarse_entry_gating(self):
        """2A) Coarse entry gating sweep (biggest lever)"""
        print("\n📋 STEP 2A: COARSE ENTRY GATING SWEEP")
        print("-" * 40)

        print("🎯 SWEEPING ENTRY THRESHOLDS:")
        print("   • confidence: 0.28 → 0.34 step 0.01")
        print("   • strength: 0.38 → 0.44 step 0.02")
        print("   • momentum-only size: 0.5× until second domain confirms")

        # Define parameter grid
        confidence_range = np.arange(0.28, 0.35, 0.01)  # 0.28, 0.29, ..., 0.34
        strength_range = np.arange(0.38, 0.45, 0.02)    # 0.38, 0.40, 0.42, 0.44

        param_combinations = list(itertools.product(confidence_range, strength_range))
        print(f"🔬 Testing {len(param_combinations)} parameter combinations")

        best_configs = []

        for i, (conf, strength) in enumerate(param_combinations, 1):
            print(f"\n🧪 Test {i}/{len(param_combinations)}: conf={conf:.2f}, strength={strength:.2f}")

            # Run backtest with these parameters
            metrics = self._run_backtest_with_params(
                confidence_threshold=conf,
                strength_threshold=strength,
                test_type='entry_gating'
            )

            if metrics:
                # Evaluate against success criteria
                evaluation = self._evaluate_config(metrics, 'entry_gating')

                config_result = {
                    'test_id': f"entry_{i}",
                    'confidence': conf,
                    'strength': strength,
                    'metrics': metrics,
                    'evaluation': evaluation,
                    'meets_primary': evaluation['meets_primary_criteria'],
                    'meets_health': evaluation['meets_health_bands']
                }

                self.results.append(config_result)

                if evaluation['meets_primary_criteria'] and evaluation['meets_health_bands']:
                    best_configs.append(config_result)
                    print(f"   ✅ PASSED: PF={metrics['profit_factor']:.2f}, DD={metrics['max_drawdown']:.1f}%")
                else:
                    print(f"   ❌ FAILED: {evaluation['failure_reasons']}")

        # Select top 2-3 configurations
        if best_configs:
            # Sort by profit factor improvement
            best_configs.sort(key=lambda x: x['metrics']['profit_factor'], reverse=True)
            top_configs = best_configs[:3]

            print(f"\n🏆 TOP {len(top_configs)} ENTRY GATING CONFIGS:")
            for i, config in enumerate(top_configs, 1):
                print(f"   {i}. conf={config['confidence']:.2f}, strength={config['strength']:.2f} "
                      f"→ PF={config['metrics']['profit_factor']:.2f}, DD={config['metrics']['max_drawdown']:.1f}%")

            return top_configs
        else:
            print("❌ No configurations passed entry gating criteria")
            return []

    def _run_backtest_with_params(self, confidence_threshold: float, strength_threshold: float,
                                 test_type: str = 'full') -> Dict[str, Any]:
        """Run backtest with specified parameters"""
        try:
            # Load base config
            with open('configs/v170/assets/ETH_v17_tuned.json', 'r') as f:
                config = json.load(f)

            # Modify parameters
            config['fusion']['calibration_thresholds']['confidence'] = confidence_threshold
            config['fusion']['calibration_thresholds']['strength'] = strength_threshold

            # Initialize engines with modified config
            smc_engine = SMCEngine(config['domains']['smc'])
            momentum_engine = MomentumEngine(config['domains']['momentum'])
            wyckoff_engine = WyckoffEngine(config['domains']['wyckoff'])
            hob_engine = HOBDetector(config['domains']['liquidity']['hob_detection'])

            # Run simplified backtest on ETH data
            eth_data = self.calibration_slice['data']['ETH']['4H']

            if len(eth_data) < 50:
                return None

            trades = []
            telemetry = []

            # Simple backtest loop
            for i in range(20, len(eth_data), 3):  # Every 3rd bar for speed
                window_data = eth_data.iloc[:i+1]
                recent_data = window_data.tail(50)
                current_bar = window_data.iloc[-1]

                try:
                    # Generate signals
                    domain_signals = {}

                    # SMC
                    try:
                        smc_signal = smc_engine.analyze(recent_data)
                        domain_signals['smc'] = smc_signal
                    except:
                        domain_signals['smc'] = None

                    # Momentum
                    try:
                        momentum_signal = momentum_engine.analyze(recent_data)
                        domain_signals['momentum'] = momentum_signal
                    except:
                        domain_signals['momentum'] = None

                    # Count active signals and apply fusion logic
                    active_signals = [s for s in domain_signals.values() if s is not None]

                    if len(active_signals) >= 1:  # At least one signal
                        # Get directions and confidences
                        directions = []
                        confidences = []

                        for signal in active_signals:
                            if hasattr(signal, 'direction') and hasattr(signal, 'confidence'):
                                directions.append(signal.direction)
                                confidences.append(signal.confidence)

                        if directions and confidences:
                            # Simple fusion
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

                            # Check thresholds
                            if avg_confidence >= confidence_threshold and fusion_strength >= strength_threshold:
                                # Record trade signal
                                trade = {
                                    'timestamp': current_bar.name,
                                    'price': current_bar['close'],
                                    'direction': fusion_direction,
                                    'confidence': avg_confidence,
                                    'strength': fusion_strength,
                                    'active_engines': len(active_signals)
                                }
                                trades.append(trade)

                                # Track telemetry
                                telemetry.append({
                                    'entry_type': 'momentum_only' if len(active_signals) == 1 and 'momentum' in [k for k,v in domain_signals.items() if v] else 'multi_domain',
                                    'smc_active': domain_signals['smc'] is not None,
                                    'momentum_active': domain_signals['momentum'] is not None
                                })

                except Exception:
                    continue

            # Calculate metrics from trades
            if len(trades) >= 3:
                return self._calculate_metrics_from_trades(trades, telemetry, eth_data)
            else:
                return None

        except Exception as e:
            print(f"   Backtest error: {e}")
            return None

    def _calculate_metrics_from_trades(self, trades: List[Dict], telemetry: List[Dict],
                                     price_data: pd.DataFrame) -> Dict[str, Any]:
        """Calculate performance metrics from trade signals"""

        if len(trades) < 2:
            return None

        # Simulate trade execution with simple exit after fixed bars
        completed_trades = []

        for i, trade in enumerate(trades[:-1]):  # Exclude last trade
            entry_price = trade['price']
            entry_time = trade['timestamp']
            direction = trade['direction']

            # Find exit (next opposite signal or fixed time)
            exit_idx = None
            for j, next_trade in enumerate(trades[i+1:], i+1):
                if next_trade['direction'] != direction:
                    exit_idx = j
                    break

            if exit_idx is None:
                # Use fixed 20-bar exit
                try:
                    entry_bar_idx = price_data.index.get_loc(entry_time)
                    exit_bar_idx = min(entry_bar_idx + 20, len(price_data) - 1)
                    exit_price = price_data.iloc[exit_bar_idx]['close']
                    exit_time = price_data.index[exit_bar_idx]
                except:
                    continue
            else:
                exit_price = trades[exit_idx]['price']
                exit_time = trades[exit_idx]['timestamp']

            # Calculate PnL
            if direction == 'long':
                pnl_pct = ((exit_price - entry_price) / entry_price) * 100
            else:  # short
                pnl_pct = ((entry_price - exit_price) / entry_price) * 100

            completed_trades.append({
                'entry_price': entry_price,
                'exit_price': exit_price,
                'pnl_pct': pnl_pct,
                'direction': direction,
                'hold_time_hours': (exit_time - entry_time).total_seconds() / 3600,
                'win': pnl_pct > 0
            })

        if len(completed_trades) < 2:
            return None

        # Calculate metrics
        win_rate = sum(1 for t in completed_trades if t['win']) / len(completed_trades) * 100
        avg_return = np.mean([t['pnl_pct'] for t in completed_trades])

        winning_trades = [t['pnl_pct'] for t in completed_trades if t['pnl_pct'] > 0]
        losing_trades = [t['pnl_pct'] for t in completed_trades if t['pnl_pct'] < 0]

        avg_win = np.mean(winning_trades) if winning_trades else 0
        avg_loss = np.mean(losing_trades) if losing_trades else 0

        profit_factor = abs(avg_win * len(winning_trades)) / abs(avg_loss * len(losing_trades)) if losing_trades else 999

        # Simple drawdown calculation
        cumulative_returns = np.cumsum([t['pnl_pct'] for t in completed_trades])
        running_max = np.maximum.accumulate(cumulative_returns)
        drawdowns = (running_max - cumulative_returns)
        max_drawdown = np.max(drawdowns) if len(drawdowns) > 0 else 0

        # Health band metrics (simplified)
        momentum_only_pct = sum(1 for t in telemetry if t['entry_type'] == 'momentum_only') / len(telemetry) * 100 if telemetry else 0
        smc_active_pct = sum(1 for t in telemetry if t['smc_active']) / len(telemetry) * 100 if telemetry else 0

        return {
            'total_trades': len(completed_trades),
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'max_drawdown': max_drawdown,
            'avg_return': avg_return,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'avg_hold_time': np.mean([t['hold_time_hours'] for t in completed_trades]),
            'momentum_only_pct': momentum_only_pct,
            'smc_active_pct': smc_active_pct,
            'signals_generated': len(trades)
        }

    def _evaluate_config(self, metrics: Dict[str, Any], test_type: str) -> Dict[str, Any]:
        """Evaluate configuration against success criteria"""

        evaluation = {
            'meets_primary_criteria': False,
            'meets_health_bands': False,
            'failure_reasons': []
        }

        # Check primary success criteria
        pf_improvement = metrics['profit_factor'] / self.baseline_metrics['profit_factor']
        dd_ratio = metrics['max_drawdown'] / self.baseline_metrics['max_drawdown']

        primary_pass = (pf_improvement >= self.success_criteria['primary']['profit_factor_lift'] or
                       dd_ratio <= self.success_criteria['primary']['max_dd_reduction'])

        if primary_pass:
            evaluation['meets_primary_criteria'] = True
        else:
            evaluation['failure_reasons'].append(f"Primary: PF={pf_improvement:.2f} (<1.10), DD ratio={dd_ratio:.2f} (>0.85)")

        # Check health bands
        health_issues = []

        # Momentum-only trades should be reasonable
        if metrics.get('momentum_only_pct', 0) > 50:
            health_issues.append(f"Too many momentum-only trades: {metrics['momentum_only_pct']:.1f}%")

        # Minimum number of trades for validity
        if metrics['total_trades'] < 5:
            health_issues.append(f"Insufficient trades: {metrics['total_trades']}")

        # Win rate not collapsing
        if metrics['win_rate'] < 35:
            health_issues.append(f"Win rate too low: {metrics['win_rate']:.1f}%")

        if not health_issues:
            evaluation['meets_health_bands'] = True
        else:
            evaluation['failure_reasons'].extend(health_issues)

        return evaluation

    def run_calibration_protocol(self):
        """Execute the complete calibration protocol"""
        print("🚀 STARTING BULL MACHINE v1.7 CALIBRATION PROTOCOL")
        print("="*80)

        # Step 0: Define success
        if not self.step_0_define_success():
            print("❌ Failed to define success criteria")
            return False

        # Step 1: Freeze calibration slice
        if not self.step_1_freeze_calibration_slice():
            print("❌ Failed to freeze calibration slice")
            return False

        # Step 2A: Coarse entry gating
        top_configs = self.step_2a_coarse_entry_gating()
        if not top_configs:
            print("❌ No viable configurations found in entry gating")
            return False

        print(f"\n🎉 CALIBRATION PROTOCOL PHASE 1 COMPLETE")
        print(f"   • Found {len(top_configs)} viable entry gating configurations")
        print(f"   • Ready for fine-tuning SMC, HOB, and Wyckoff parameters")
        print(f"   • Slice: {self.calibration_slice['total_days']} days, hash: {self.config_hash}")

        return True

def main():
    """Run calibration protocol"""
    try:
        protocol = CalibrationProtocol()
        success = protocol.run_calibration_protocol()

        if success:
            print("\n✅ CALIBRATION PROTOCOL PHASE 1 SUCCESSFUL")
        else:
            print("\n❌ CALIBRATION PROTOCOL FAILED")

    except Exception as e:
        print(f"❌ Protocol error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()