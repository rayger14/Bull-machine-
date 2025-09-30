#!/usr/bin/env python3
"""
Bull Machine v1.7 Baseline Analysis
Establish realistic baseline and adjust calibration approach
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

from engine.io.tradingview_loader import load_tv
from engine.smc.smc_engine import SMCEngine
from engine.momentum.momentum_engine import MomentumEngine
from engine.wyckoff.wyckoff_engine import WyckoffEngine
from engine.liquidity.hob import HOBDetector

def analyze_current_system_baseline():
    """Analyze current v1.7 system to establish realistic baseline"""

    print("📊 BULL MACHINE v1.7 BASELINE ANALYSIS")
    print("="*60)

    # Load data
    print("📈 Loading calibration data (300 bars)...")
    try:
        eth_data = load_tv('ETH_4H')
        calibration_data = eth_data.tail(300)
        print(f"✅ {len(calibration_data)} bars from {calibration_data.index[0]} to {calibration_data.index[-1]}")
    except Exception as e:
        print(f"❌ Failed to load data: {e}")
        return

    # Load current config
    with open('configs/v170/assets/ETH_v17_tuned.json', 'r') as f:
        config = json.load(f)

    current_conf = config['fusion']['calibration_thresholds']['confidence']
    current_strength = config['fusion']['calibration_thresholds']['strength']

    print(f"🔧 Current thresholds: confidence={current_conf}, strength={current_strength}")

    # Initialize engines
    print("\n🤖 Initializing engines...")
    try:
        smc_engine = SMCEngine(config['domains']['smc'])
        momentum_engine = MomentumEngine(config['domains']['momentum'])
        wyckoff_engine = WyckoffEngine(config['domains']['wyckoff'])
        hob_engine = HOBDetector(config['domains']['liquidity']['hob_detection'])
        print("✅ All engines initialized")
    except Exception as e:
        print(f"❌ Engine initialization failed: {e}")
        return

    # Analyze signal generation across different thresholds
    print(f"\n🔍 SIGNAL GENERATION ANALYSIS")
    print("-" * 40)

    threshold_tests = [
        (0.25, 0.35),  # Very relaxed
        (0.28, 0.38),  # Relaxed
        (0.30, 0.40),  # Current calibration
        (0.32, 0.42),  # Slightly strict
        (0.35, 0.45),  # Production level
    ]

    baseline_results = []

    for conf_thresh, strength_thresh in threshold_tests:
        print(f"\n🧪 Testing: confidence={conf_thresh}, strength={strength_thresh}")

        # Run comprehensive signal analysis
        signals_analysis = analyze_signal_generation(
            data=calibration_data,
            smc_engine=smc_engine,
            momentum_engine=momentum_engine,
            wyckoff_engine=wyckoff_engine,
            hob_engine=hob_engine,
            confidence_threshold=conf_thresh,
            strength_threshold=strength_thresh
        )

        if signals_analysis:
            baseline_results.append({
                'confidence': conf_thresh,
                'strength': strength_thresh,
                **signals_analysis
            })

            print(f"   📊 Signals: {signals_analysis['signals_generated']}")
            print(f"   💰 Trades: {signals_analysis['trades_executed']}")
            if signals_analysis['trades_executed'] > 0:
                print(f"   📈 PF: {signals_analysis['profit_factor']:.2f}, "
                      f"DD: {signals_analysis['max_drawdown']:.1f}%, "
                      f"WR: {signals_analysis['win_rate']:.1f}%")

    # Establish realistic baseline
    print(f"\n📋 BASELINE ESTABLISHMENT")
    print("=" * 40)

    # Find best performing threshold combination
    valid_results = [r for r in baseline_results if r['trades_executed'] >= 5]

    if valid_results:
        # Sort by profit factor
        best_result = max(valid_results, key=lambda x: x['profit_factor'])

        print(f"🎯 RECOMMENDED v1.7 BASELINE:")
        print(f"   • Confidence threshold: {best_result['confidence']}")
        print(f"   • Strength threshold: {best_result['strength']}")
        print(f"   • Profit Factor: {best_result['profit_factor']:.2f}")
        print(f"   • Max Drawdown: {best_result['max_drawdown']:.1f}%")
        print(f"   • Win Rate: {best_result['win_rate']:.1f}%")
        print(f"   • Trades: {best_result['trades_executed']}")

        # Calculate success targets
        target_pf = best_result['profit_factor'] * 1.10
        target_dd = best_result['max_drawdown'] * 0.85

        print(f"\n🎯 SUCCESS TARGETS FOR CALIBRATION:")
        print(f"   • Target PF: ≥ {target_pf:.2f} (+10%)")
        print(f"   • Target DD: ≤ {target_dd:.1f}% (-15%)")

        # Update config with best baseline
        config['fusion']['calibration_thresholds']['confidence'] = best_result['confidence']
        config['fusion']['calibration_thresholds']['strength'] = best_result['strength']

        # Save updated config
        with open('configs/v170/assets/ETH_v17_baseline.json', 'w') as f:
            json.dump(config, f, indent=2)

        print(f"💾 Saved baseline config: ETH_v17_baseline.json")

        # Now run focused calibration around this baseline
        run_focused_calibration_around_baseline(
            baseline_conf=best_result['confidence'],
            baseline_strength=best_result['strength'],
            target_pf=target_pf,
            target_dd=target_dd,
            calibration_data=calibration_data,
            engines=(smc_engine, momentum_engine, wyckoff_engine, hob_engine)
        )

    else:
        print("❌ No valid baseline configurations found")
        print("   System may need fundamental adjustments")

def analyze_signal_generation(data, smc_engine, momentum_engine, wyckoff_engine, hob_engine,
                            confidence_threshold, strength_threshold):
    """Comprehensive signal generation analysis"""

    signals = []
    trades = []
    telemetry = []

    # Process data
    for i in range(30, len(data), 2):  # Every 2nd bar for speed
        window_data = data.iloc[:i+1]
        recent_data = window_data.tail(60)
        current_bar = window_data.iloc[-1]

        try:
            # Generate signals from all engines
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

            # Wyckoff
            try:
                wyckoff_signal = wyckoff_engine.analyze(recent_data, usdt_stagnation=0.5)
                domain_signals['wyckoff'] = wyckoff_signal
            except:
                domain_signals['wyckoff'] = None

            # HOB
            try:
                hob_signal = hob_engine.detect_hob(recent_data)
                domain_signals['hob'] = hob_signal
            except:
                domain_signals['hob'] = None

            # Count active signals
            active_signals = [s for s in domain_signals.values() if s is not None]
            signals.append({
                'timestamp': current_bar.name,
                'price': current_bar['close'],
                'active_engines': len(active_signals),
                'smc_active': domain_signals['smc'] is not None,
                'momentum_active': domain_signals['momentum'] is not None,
                'wyckoff_active': domain_signals['wyckoff'] is not None,
                'hob_active': domain_signals['hob'] is not None
            })

            # Apply fusion logic
            if len(active_signals) >= 1:
                directions = []
                confidences = []

                for signal in active_signals:
                    if hasattr(signal, 'direction') and hasattr(signal, 'confidence'):
                        directions.append(signal.direction)
                        confidences.append(signal.confidence)

                if directions and confidences:
                    # Vote on direction
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

                    # Check entry thresholds
                    if avg_confidence >= confidence_threshold and fusion_strength >= strength_threshold:
                        trade = {
                            'timestamp': current_bar.name,
                            'price': current_bar['close'],
                            'direction': fusion_direction,
                            'confidence': avg_confidence,
                            'strength': fusion_strength,
                            'active_engines': len(active_signals)
                        }
                        trades.append(trade)

                        # Track quality metrics
                        telemetry.append({
                            'momentum_only': len(active_signals) == 1 and domain_signals['momentum'] is not None,
                            'multi_engine': len(active_signals) >= 2,
                            'smc_confluence': domain_signals['smc'] is not None and len(active_signals) >= 2
                        })

        except Exception:
            continue

    # Calculate performance metrics
    if len(trades) < 2:
        return None

    # Simple trade execution simulation
    completed_trades = []

    for i, trade in enumerate(trades[:-1]):
        entry_price = trade['price']
        direction = trade['direction']

        # Find exit
        exit_price = None
        for j in range(i + 1, len(trades)):
            if trades[j]['direction'] != direction:
                exit_price = trades[j]['price']
                break

        if exit_price is None:
            # Fixed exit after some time
            try:
                entry_idx = data.index.get_loc(trade['timestamp'])
                exit_idx = min(entry_idx + 15, len(data) - 1)
                exit_price = data.iloc[exit_idx]['close']
            except:
                continue

        # Calculate PnL
        if direction == 'long':
            pnl_pct = ((exit_price - entry_price) / entry_price) * 100
        else:
            pnl_pct = ((entry_price - exit_price) / entry_price) * 100

        completed_trades.append({
            'pnl_pct': pnl_pct,
            'win': pnl_pct > 0
        })

    if len(completed_trades) < 2:
        return None

    # Calculate metrics
    returns = [t['pnl_pct'] for t in completed_trades]
    wins = [t for t in completed_trades if t['win']]
    losses = [t for t in completed_trades if not t['win']]

    win_rate = len(wins) / len(completed_trades) * 100
    avg_win = np.mean([t['pnl_pct'] for t in wins]) if wins else 0
    avg_loss = np.mean([t['pnl_pct'] for t in losses]) if losses else 0

    profit_factor = (abs(avg_win) * len(wins)) / (abs(avg_loss) * len(losses)) if losses else 999

    # Max drawdown
    cumulative = np.cumsum(returns)
    running_max = np.maximum.accumulate(cumulative)
    drawdowns = running_max - cumulative
    max_drawdown = np.max(drawdowns) if len(drawdowns) > 0 else 0

    # Quality metrics
    momentum_only_pct = sum(1 for t in telemetry if t['momentum_only']) / len(telemetry) * 100 if telemetry else 0
    multi_engine_pct = sum(1 for t in telemetry if t['multi_engine']) / len(telemetry) * 100 if telemetry else 0

    return {
        'signals_generated': len(signals),
        'trades_executed': len(completed_trades),
        'win_rate': win_rate,
        'profit_factor': profit_factor,
        'max_drawdown': max_drawdown,
        'avg_return': np.mean(returns),
        'momentum_only_pct': momentum_only_pct,
        'multi_engine_pct': multi_engine_pct
    }

def run_focused_calibration_around_baseline(baseline_conf, baseline_strength, target_pf, target_dd,
                                          calibration_data, engines):
    """Run focused calibration around established baseline"""

    print(f"\n🎯 FOCUSED CALIBRATION AROUND BASELINE")
    print("-" * 40)

    smc_engine, momentum_engine, wyckoff_engine, hob_engine = engines

    # Define small parameter grid around baseline
    conf_range = [baseline_conf - 0.02, baseline_conf, baseline_conf + 0.02]
    strength_range = [baseline_strength - 0.02, baseline_strength, baseline_strength + 0.02]

    print(f"🔬 Testing {len(conf_range) * len(strength_range)} combinations around baseline")
    print(f"   Confidence: {conf_range}")
    print(f"   Strength: {strength_range}")

    best_configs = []

    for conf in conf_range:
        for strength in strength_range:
            if conf <= 0 or strength <= 0:
                continue

            print(f"\n🧪 Testing: conf={conf:.2f}, strength={strength:.2f}")

            metrics = analyze_signal_generation(
                data=calibration_data,
                smc_engine=smc_engine,
                momentum_engine=momentum_engine,
                wyckoff_engine=wyckoff_engine,
                hob_engine=hob_engine,
                confidence_threshold=conf,
                strength_threshold=strength
            )

            if metrics and metrics['trades_executed'] >= 3:
                meets_targets = (
                    metrics['profit_factor'] >= target_pf or
                    metrics['max_drawdown'] <= target_dd
                )

                if meets_targets:
                    best_configs.append({
                        'confidence': conf,
                        'strength': strength,
                        'profit_factor': metrics['profit_factor'],
                        'max_drawdown': metrics['max_drawdown'],
                        'win_rate': metrics['win_rate'],
                        'trades': metrics['trades_executed']
                    })
                    print(f"   ✅ MEETS TARGETS: PF={metrics['profit_factor']:.2f}, DD={metrics['max_drawdown']:.1f}%")
                else:
                    print(f"   ❌ Below targets: PF={metrics['profit_factor']:.2f}, DD={metrics['max_drawdown']:.1f}%")

    if best_configs:
        best = max(best_configs, key=lambda x: x['profit_factor'])
        print(f"\n🏆 CALIBRATED CONFIGURATION:")
        print(f"   • Confidence: {best['confidence']:.2f}")
        print(f"   • Strength: {best['strength']:.2f}")
        print(f"   • Profit Factor: {best['profit_factor']:.2f} (target: {target_pf:.2f})")
        print(f"   • Max Drawdown: {best['max_drawdown']:.1f}% (target: {target_dd:.1f}%)")
        print(f"   • Win Rate: {best['win_rate']:.1f}%")

        return best
    else:
        print(f"\n⚠️ No configurations met improvement targets")
        print(f"   Baseline appears optimal for current market conditions")
        return None

if __name__ == "__main__":
    analyze_current_system_baseline()