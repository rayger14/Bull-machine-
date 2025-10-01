#!/usr/bin/env python3
"""
Bull Machine v1.7 Entry Gating Calibration
Focus on the biggest lever: entry thresholds
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

import pandas as pd
import numpy as np
import json
import hashlib
from datetime import datetime, timedelta
import itertools
import warnings
warnings.filterwarnings('ignore')

from engine.io.tradingview_loader import load_tv
from engine.smc.smc_engine import SMCEngine
from engine.momentum.momentum_engine import MomentumEngine

def run_entry_gating_sweep():
    """Execute focused entry gating parameter sweep"""

    print("🎯 BULL MACHINE v1.7 ENTRY GATING CALIBRATION")
    print("="*60)

    # Define success criteria
    baseline_pf = 1.15
    baseline_dd = 12.5

    print("🎯 SUCCESS CRITERIA:")
    print(f"   • Profit Factor ≥ {baseline_pf * 1.10:.2f} (10% improvement)")
    print(f"   • OR Max DD ≤ {baseline_dd * 0.85:.1f}% (15% reduction)")

    # Load calibration data (75 days)
    print("\n📊 LOADING CALIBRATION DATA")
    try:
        eth_data = load_tv('ETH_4H')

        # Use recent 300 bars (roughly 75 days at 4H)
        calibration_data = eth_data.tail(300)

        print(f"✅ Loaded {len(calibration_data)} bars ({calibration_data.index[0]} to {calibration_data.index[-1]})")

        # Hash config for tracking
        with open('configs/v170/assets/ETH_v17_tuned.json', 'r') as f:
            config = json.load(f)
        config_hash = hashlib.md5(json.dumps(config, sort_keys=True).encode()).hexdigest()[:8]
        print(f"🔒 Config hash: {config_hash}")

    except Exception as e:
        print(f"❌ Failed to load data: {e}")
        return

    # Initialize engines
    print("\n🔧 INITIALIZING ENGINES")
    try:
        smc_engine = SMCEngine(config['domains']['smc'])
        momentum_engine = MomentumEngine(config['domains']['momentum'])
        print("✅ Engines initialized")
    except Exception as e:
        print(f"❌ Engine initialization failed: {e}")
        return

    # Define parameter sweep
    confidence_range = [0.28, 0.30, 0.32, 0.34]  # Focused range
    strength_range = [0.38, 0.40, 0.42, 0.44]    # Focused range

    combinations = list(itertools.product(confidence_range, strength_range))
    print(f"\n🔬 TESTING {len(combinations)} PARAMETER COMBINATIONS")
    print("   Sweeping: confidence × strength")

    results = []
    best_configs = []

    for i, (conf, strength) in enumerate(combinations, 1):
        print(f"\n🧪 Test {i}/{len(combinations)}: confidence={conf:.2f}, strength={strength:.2f}")

        try:
            # Run backtest with these parameters
            metrics = run_focused_backtest(
                data=calibration_data,
                smc_engine=smc_engine,
                momentum_engine=momentum_engine,
                confidence_threshold=conf,
                strength_threshold=strength
            )

            if metrics and metrics['total_trades'] >= 3:
                # Evaluate against criteria
                pf_improvement = metrics['profit_factor'] / baseline_pf
                dd_ratio = metrics['max_drawdown'] / baseline_dd

                meets_primary = (pf_improvement >= 1.10 or dd_ratio <= 0.85)
                meets_health = (
                    metrics['win_rate'] >= 35 and  # Minimum win rate
                    metrics['momentum_only_pct'] <= 60 and  # Not too momentum-heavy
                    metrics['total_trades'] >= 5  # Sufficient trades
                )

                result = {
                    'confidence': conf,
                    'strength': strength,
                    'profit_factor': metrics['profit_factor'],
                    'max_drawdown': metrics['max_drawdown'],
                    'win_rate': metrics['win_rate'],
                    'total_trades': metrics['total_trades'],
                    'pf_improvement': pf_improvement,
                    'dd_ratio': dd_ratio,
                    'meets_primary': meets_primary,
                    'meets_health': meets_health,
                    'momentum_only_pct': metrics['momentum_only_pct']
                }

                results.append(result)

                if meets_primary and meets_health:
                    best_configs.append(result)
                    print(f"   ✅ PASSED: PF={metrics['profit_factor']:.2f} (+{(pf_improvement-1)*100:.1f}%), "
                          f"DD={metrics['max_drawdown']:.1f}% ({dd_ratio:.2f}x), WR={metrics['win_rate']:.1f}%")
                else:
                    reasons = []
                    if not meets_primary:
                        reasons.append(f"Primary: PF={pf_improvement:.2f}, DD={dd_ratio:.2f}")
                    if not meets_health:
                        reasons.append(f"Health: WR={metrics['win_rate']:.1f}%, Mom={metrics['momentum_only_pct']:.1f}%")
                    print(f"   ❌ FAILED: {'; '.join(reasons)}")
            else:
                print(f"   ⚠️ INSUFFICIENT DATA: {metrics['total_trades'] if metrics else 0} trades")

        except Exception as e:
            print(f"   💥 ERROR: {e}")
            continue

    # Report results
    print(f"\n" + "="*60)
    print("🏆 ENTRY GATING CALIBRATION RESULTS")
    print("="*60)

    print(f"📊 TESTED: {len(results)} valid configurations")
    print(f"✅ PASSED: {len(best_configs)} configurations")

    if best_configs:
        # Sort by profit factor improvement
        best_configs.sort(key=lambda x: x['profit_factor'], reverse=True)

        print(f"\n🥇 TOP CONFIGURATIONS:")
        for i, config in enumerate(best_configs[:3], 1):
            print(f"   {i}. conf={config['confidence']:.2f}, strength={config['strength']:.2f}")
            print(f"      → PF={config['profit_factor']:.2f} (+{(config['pf_improvement']-1)*100:.1f}%), "
                  f"DD={config['max_drawdown']:.1f}% ({config['dd_ratio']:.2f}x)")
            print(f"      → WR={config['win_rate']:.1f}%, Trades={config['total_trades']}, "
                  f"Momentum-only={config['momentum_only_pct']:.1f}%")

        # Recommend best configuration
        best = best_configs[0]
        print(f"\n🎯 RECOMMENDED CONFIGURATION:")
        print(f"   • entry_threshold_confidence: {best['confidence']:.2f}")
        print(f"   • entry_threshold_strength: {best['strength']:.2f}")
        print(f"   • Expected improvement: PF +{(best['pf_improvement']-1)*100:.1f}%, DD {best['dd_ratio']:.2f}x")

        # Update config file
        config['fusion']['calibration_thresholds']['confidence'] = best['confidence']
        config['fusion']['calibration_thresholds']['strength'] = best['strength']

        output_file = f"configs/v170/assets/ETH_v17_calibrated_{config_hash}.json"
        with open(output_file, 'w') as f:
            json.dump(config, f, indent=2)

        print(f"💾 Saved calibrated config: {output_file}")

    else:
        print(f"\n❌ NO VIABLE CONFIGURATIONS FOUND")
        print(f"   Consider:")
        print(f"   • Lowering success criteria")
        print(f"   • Expanding parameter ranges")
        print(f"   • Checking baseline assumptions")

def run_focused_backtest(data: pd.DataFrame, smc_engine, momentum_engine,
                        confidence_threshold: float, strength_threshold: float) -> dict:
    """Run focused backtest with given parameters"""

    trades = []
    telemetry = []

    # Process every 3rd bar for speed
    for i in range(20, len(data), 3):
        window_data = data.iloc[:i+1]
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

            # Apply fusion logic
            active_signals = [s for s in domain_signals.values() if s is not None]

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

                        # Track telemetry
                        is_momentum_only = (len(active_signals) == 1 and
                                          domain_signals['momentum'] is not None and
                                          domain_signals['smc'] is None)

                        telemetry.append({
                            'momentum_only': is_momentum_only,
                            'smc_active': domain_signals['smc'] is not None
                        })

        except Exception:
            continue

    # Calculate metrics from trades
    if len(trades) < 2:
        return {'total_trades': 0}

    # Simulate simple trade execution
    completed_trades = []

    for i, trade in enumerate(trades[:-1]):
        entry_price = trade['price']
        direction = trade['direction']

        # Simple exit: next opposite signal or +20 bars
        exit_price = None
        for j in range(i + 1, len(trades)):
            if trades[j]['direction'] != direction:
                exit_price = trades[j]['price']
                break

        if exit_price is None:
            # Use fixed exit after some bars
            try:
                entry_idx = data.index.get_loc(trade['timestamp'])
                exit_idx = min(entry_idx + 20, len(data) - 1)
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
        return {'total_trades': len(completed_trades)}

    # Calculate performance metrics
    returns = [t['pnl_pct'] for t in completed_trades]
    wins = [t for t in completed_trades if t['win']]
    losses = [t for t in completed_trades if not t['win']]

    win_rate = len(wins) / len(completed_trades) * 100

    avg_win = np.mean([t['pnl_pct'] for t in wins]) if wins else 0
    avg_loss = np.mean([t['pnl_pct'] for t in losses]) if losses else 0

    profit_factor = (abs(avg_win) * len(wins)) / (abs(avg_loss) * len(losses)) if losses else 999

    # Simple max drawdown
    cumulative = np.cumsum(returns)
    running_max = np.maximum.accumulate(cumulative)
    drawdowns = running_max - cumulative
    max_drawdown = np.max(drawdowns) if len(drawdowns) > 0 else 0

    # Health metrics
    momentum_only_pct = sum(1 for t in telemetry if t['momentum_only']) / len(telemetry) * 100 if telemetry else 0

    return {
        'total_trades': len(completed_trades),
        'win_rate': win_rate,
        'profit_factor': profit_factor,
        'max_drawdown': max_drawdown,
        'avg_return': np.mean(returns),
        'momentum_only_pct': momentum_only_pct,
        'signals_generated': len(trades)
    }

if __name__ == "__main__":
    run_entry_gating_sweep()