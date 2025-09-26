#!/usr/bin/env python3
"""
Bull Machine v1.6.0 - ETH Enhanced Ensemble Backtest
🚀 Integrates M1/M2 Wyckoff + Hidden Fibonacci zones for RC target achievement
Target: WR ≥50%, PnL ≥10% (1D)/≥30% (4H), PF ≥1.3, Sharpe ≥2.2
"""

import pandas as pd
import numpy as np
import json
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from bull_machine.modules.fusion.v160_enhanced import CoreTraderV160
from bull_machine.backtest.ensemble_mode import EnsembleAligner
from bull_machine.strategy.atr_exits import check_exit
from bull_machine.strategy.wyckoff_m1m2 import compute_m1m2_scores
from bull_machine.strategy.hidden_fibs import compute_hidden_fib_scores

def load_v160_config() -> Dict:
    """Load v1.6.0 configuration with M1/M2 and Fibonacci enhancements."""
    config = {
        "entry_threshold": 0.44,
        "quality_floors": {
            "wyckoff": 0.25,
            "m1": 0.25,
            "m2": 0.25,
            "liquidity": 0.25,
            "structure": 0.25,
            "momentum": 0.27,
            "volume": 0.25,
            "context": 0.25,
            "mtf": 0.27,
            "fib_retracement": 0.25,
            "fib_extension": 0.25
        },
        "features": {
            "mtf_dl2": True,
            "six_candle_leg": True,
            "wyckoff_phase": True,
            "wyckoff_m1m2": True,
            "liquidity_sweep": True,
            "order_blocks": True,
            "wick_magnet": True,
            "hidden_fibs": True,
            "atr_exits": True,
            "atr_sizing": True,
            "regime_filter": True,
            "ensemble_htf_bias": True,
            "use_asset_profiles": True
        },
        "cooldown_bars": 168,
        "risk": {
            "risk_pct": 0.005,
            "atr_window": 14,
            "sl_atr": 1.8,
            "tp_atr": 3.0,
            "trail_atr": 1.2,
            "profit_ladders": [
                {"ratio": 1.5, "percent": 0.25},
                {"ratio": 2.5, "percent": 0.50},
                {"ratio": 4.0, "percent": 0.25}
            ]
        },
        "regime": {
            "vol_ratio_min": 1.0,
            "atr_pct_max": 0.08
        },
        "ensemble": {
            "enabled": True,
            "min_consensus": 2,
            "consensus_penalty": 0.02,
            "rolling_k": 4,
            "rolling_n": 5,
            "lead_lag_window": 3,
            "dynamic_thresholds": True
        },
        "version": "1.6.0"
    }
    return config

def load_and_align_eth_data() -> pd.DataFrame:
    """Load ETH data from multiple timeframes and align for v1.6.0 testing."""
    print("📊 Loading ETH Multi-Timeframe Data for v1.6.0...")

    # ETH data files
    data_files = {
        "1D": "/Users/raymondghandchi/Desktop/Chart Logs/COINBASE_ETHUSD, 1D_fa116.csv",
        "4H": "/Users/raymondghandchi/Desktop/Chart Logs/COINBASE_ETHUSD, 240_ab8a9.csv",
        "1H": "/Users/raymondghandchi/Downloads/Chart logs 2/COINBASE_ETHUSD, 60_2f4ab.csv"
    }

    timeframes = {}

    for tf, file_path in data_files.items():
        print(f"Loading {tf}...")

        if not Path(file_path).exists():
            print(f"❌ {tf} file not found: {file_path}")
            continue

        df = pd.read_csv(file_path)

        # Handle timestamp
        if 'time' in df.columns:
            df['timestamp'] = pd.to_datetime(df['time'], unit='s')
        elif 'timestamp' not in df.columns:
            df['timestamp'] = pd.date_range(start='2020-01-01', periods=len(df), freq='h')

        # Handle volume
        if 'volume' not in df.columns:
            if 'BUY+SELL V' in df.columns:
                df['volume'] = df['BUY+SELL V']
            else:
                df['volume'] = df['close'] * 100

        # Clean data
        df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']].dropna()
        df = df.sort_values('timestamp').reset_index(drop=True)

        timeframes[tf] = df
        print(f"✅ {tf}: {len(df)} bars ({df['timestamp'].min()} to {df['timestamp'].max()})")

    if len(timeframes) < 3:
        raise ValueError("Need all three timeframes for ensemble trading")

    # Find overlapping period
    start_times = [df['timestamp'].min() for df in timeframes.values()]
    end_times = [df['timestamp'].max() for df in timeframes.values()]

    overlap_start = max(start_times)
    overlap_end = min(end_times)

    print(f"\n📅 Overlap Period: {overlap_start} to {overlap_end}")

    # Use 1H as base for alignment
    base_df = timeframes['1H'].copy()
    base_df = base_df[
        (base_df['timestamp'] >= overlap_start) &
        (base_df['timestamp'] <= overlap_end)
    ].reset_index(drop=True)

    return base_df, timeframes

def create_timeframe_windows(base_df: pd.DataFrame, timeframes: Dict) -> Dict:
    """Create aligned timeframe windows for ensemble scoring."""
    windows = {'1H': [], '4H': [], '1D': []}

    for i in range(50, len(base_df)):  # Start after sufficient history
        current_time = base_df.iloc[i]['timestamp']

        # 1H window (last 50 bars)
        windows['1H'].append(base_df.iloc[i-49:i+1].copy())

        # 4H window (aligned to 4H boundaries)
        tf_4h = timeframes['4H']
        aligned_4h = tf_4h[tf_4h['timestamp'] <= current_time].tail(20)
        windows['4H'].append(aligned_4h)

        # 1D window (aligned to 1D boundaries)
        tf_1d = timeframes['1D']
        aligned_1d = tf_1d[tf_1d['timestamp'] <= current_time].tail(20)
        windows['1D'].append(aligned_1d)

    return windows

def enhanced_ensemble_entry(engine, windows, config, aligner, bar_idx):
    """Enhanced v1.6.0 ensemble entry with M1/M2 and Fibonacci signals."""
    dfs = {
        '1H': windows['1H'][bar_idx],
        '4H': windows['4H'][bar_idx],
        '1D': windows['1D'][bar_idx]
    }

    # Compute base scores for each timeframe
    tf_scores = {}
    for tf, df in dfs.items():
        if len(df) < 20:
            tf_scores[tf] = {k: 0.0 for k in config['quality_floors'].keys()}
            continue

        # Set timeframe context
        engine.set_current_timeframe(tf)

        # Get enhanced scores (includes M1/M2 and Fibs)
        scores = engine.compute_base_scores(df)

        # Ensure all expected keys exist
        for key in config['quality_floors'].keys():
            if key not in scores:
                scores[key] = 0.0

        tf_scores[tf] = scores

    # Update ensemble alignment
    aligner.update(tf_scores)

    # Check ensemble firing with enhanced scoring
    fire, ensemble_score = aligner.fire(tf_scores, dfs)

    return fire, ensemble_score, tf_scores

def run_v160_backtest():
    """Run v1.6.0 enhanced backtest with M1/M2 Wyckoff and Hidden Fibonacci signals."""
    print("🚀 Starting Bull Machine v1.6.0 Enhanced Backtest")
    print("📈 Features: M1/M2 Wyckoff + Hidden Fibonacci + Volatility-Weighted Scoring")

    # Load configuration and data
    config = load_v160_config()
    base_df, timeframes = load_and_align_eth_data()

    # Initialize v1.6.0 components
    engine = CoreTraderV160()
    aligner = EnsembleAligner(config)

    # Create aligned windows
    print("🔄 Creating aligned timeframe windows...")
    windows = create_timeframe_windows(base_df, timeframes)

    # Backtest variables
    initial_balance = 10000.0
    balance = initial_balance
    positions = []
    trades = []
    equity_curve = [initial_balance]

    signals_generated = 0
    last_trade_bar = -999

    print(f"\n🎯 Backtesting {len(windows['1H'])} bars with v1.6.0 enhancements...")

    for bar_idx in range(len(windows['1H'])):
        current_bar = bar_idx + 50  # Adjust for history offset
        current_df = windows['1H'][bar_idx]

        if len(current_df) < 20:
            equity_curve.append(balance)
            continue

        current_price = current_df['close'].iloc[-1]
        current_timestamp = current_df['timestamp'].iloc[-1]

        # Process exits for existing positions
        for position in positions[:]:
            if position['closed_percent'] >= 1.0:
                positions.remove(position)
                continue

            # Use v1.6.0 true R-based exits
            exit_signal = check_exit(current_df, position, '1H', config)

            if exit_signal['close_position']:
                # Calculate PnL
                entry_price = position['entry_price']
                exit_price = exit_signal['exit_price']
                quantity = position['quantity'] * exit_signal['closed_percent']

                if position['side'] == 'LONG':
                    pnl = (exit_price - entry_price) * quantity
                else:
                    pnl = (entry_price - exit_price) * quantity

                balance += pnl

                # Record trade
                trades.append({
                    'entry_timestamp': position['entry_timestamp'],
                    'entry_bar': position['entry_bar'],
                    'side': position['side'],
                    'entry_price': entry_price,
                    'quantity': quantity,
                    'stop_loss': position['stop_loss'],
                    'ensemble_score': position.get('ensemble_score', 0.0),
                    'tf_scores': position.get('tf_scores', {}),
                    'exit_timestamp': current_timestamp,
                    'exit_bar': current_bar,
                    'exit_price': exit_price,
                    'exit_reason': exit_signal['reason'],
                    'pnl_pct': (pnl / (entry_price * quantity)) * 100,
                    'pnl_dollar': pnl,
                    'bars_held': current_bar - position['entry_bar'],
                    'closed_percent': exit_signal['closed_percent'],
                    # v1.6.0 specific fields
                    'm1_score': position.get('m1_score', 0.0),
                    'm2_score': position.get('m2_score', 0.0),
                    'fib_retracement': position.get('fib_retracement', 0.0),
                    'fib_extension': position.get('fib_extension', 0.0)
                })

                # Update position or remove if fully closed
                if exit_signal['closed_percent'] < 1.0:
                    position['quantity'] *= (1 - exit_signal['closed_percent'])
                    position['closed_percent'] = exit_signal['closed_percent']
                else:
                    positions.remove(position)

        # Check for new entries (if no positions)
        if not positions:
            try:
                # Enhanced ensemble entry check
                fire, ensemble_score, tf_scores = enhanced_ensemble_entry(
                    engine, windows, config, aligner, bar_idx
                )

                if fire:
                    signals_generated += 1

                    # Determine trade side using v1.6.0 enhanced logic
                    engine.set_current_timeframe('1H')
                    enhanced_scores = engine.compute_base_scores(current_df)
                    side = engine._determine_trade_side_enhanced(enhanced_scores, config)

                    # Calculate position size
                    risk_pct = config['risk']['risk_pct']
                    quantity = balance * risk_pct

                    # Calculate stop loss
                    atr = current_df['high'].rolling(14).max().iloc[-1] - current_df['low'].rolling(14).min().iloc[-1]
                    sl_atr = config['risk']['sl_atr']

                    if side == 'LONG':
                        stop_loss = current_price - sl_atr * atr
                    else:
                        stop_loss = current_price + sl_atr * atr

                    # Create position
                    position = {
                        'entry_timestamp': current_timestamp,
                        'entry_bar': current_bar,
                        'side': side,
                        'entry_price': current_price,
                        'quantity': quantity / current_price,
                        'stop_loss': stop_loss,
                        'ensemble_score': ensemble_score,
                        'tf_scores': tf_scores,
                        'closed_percent': 0.0,
                        'size': 1.0,
                        'ladder_taken': [False, False, False],
                        # v1.6.0 enhanced fields
                        'm1_score': enhanced_scores.get('m1', 0.0),
                        'm2_score': enhanced_scores.get('m2', 0.0),
                        'fib_retracement': enhanced_scores.get('fib_retracement', 0.0),
                        'fib_extension': enhanced_scores.get('fib_extension', 0.0),
                        'version': '1.6.0'
                    }

                    positions.append(position)
                    last_trade_bar = current_bar

                    print(f"📈 Entry #{signals_generated}: {side} @ ${current_price:.2f} | Score: {ensemble_score:.3f} | M1: {enhanced_scores.get('m1', 0.0):.2f} | M2: {enhanced_scores.get('m2', 0.0):.2f} | Fib: {enhanced_scores.get('fib_retracement', 0.0):.2f}")

            except Exception as e:
                print(f"⚠️ Entry check error at bar {current_bar}: {e}")

        equity_curve.append(balance)

        # Progress update
        if bar_idx % 100 == 0:
            print(f"📊 Progress: {bar_idx}/{len(windows['1H'])} bars | Balance: ${balance:.2f} | Signals: {signals_generated}")

    # Calculate final metrics
    total_bars = len(windows['1H'])
    final_balance = balance
    total_return_pct = ((final_balance - initial_balance) / initial_balance) * 100

    if trades:
        winning_trades = len([t for t in trades if t['pnl_dollar'] > 0])
        win_rate = (winning_trades / len(trades)) * 100
        trades_per_month = len(trades) / (total_bars / (24 * 30))  # Assuming hourly data

        # Calculate drawdown
        peak = initial_balance
        max_dd = 0
        for equity in equity_curve:
            if equity > peak:
                peak = equity
            dd = ((peak - equity) / peak) * 100
            if dd > max_dd:
                max_dd = dd

        # Calculate profit metrics
        profits = [t['pnl_dollar'] for t in trades if t['pnl_dollar'] > 0]
        losses = [abs(t['pnl_dollar']) for t in trades if t['pnl_dollar'] < 0]

        avg_win = np.mean(profits) if profits else 0
        avg_loss = np.mean(losses) if losses else 1
        profit_factor = sum(profits) / sum(losses) if losses else float('inf')

        # Calculate Sharpe ratio (simplified)
        returns = pd.Series(equity_curve).pct_change().dropna()
        sharpe_ratio = returns.mean() / returns.std() * np.sqrt(252) if returns.std() > 0 else 0
    else:
        win_rate = 0
        trades_per_month = 0
        max_dd = 0
        avg_win = 0
        avg_loss = 0
        profit_factor = 0
        sharpe_ratio = 0

    # Results
    results = {
        "version": "1.6.0",
        "total_bars": total_bars,
        "signals_generated": signals_generated,
        "total_trades": len(trades),
        "winning_trades": len([t for t in trades if t['pnl_dollar'] > 0]),
        "win_rate": win_rate,
        "trades_per_month": trades_per_month,
        "initial_balance": initial_balance,
        "final_balance": final_balance,
        "total_return_pct": total_return_pct,
        "max_drawdown": max_dd,
        "avg_win": avg_win,
        "avg_loss": avg_loss,
        "profit_factor": profit_factor,
        "sharpe_ratio": sharpe_ratio,
        "trades": trades[:10],  # First 10 trades for analysis
        "equity_curve": equity_curve[-100:],  # Last 100 points
        "config": config
    }

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"eth_ensemble_backtest_v160_{timestamp}.json"

    with open(filename, 'w') as f:
        json.dump(results, f, indent=2, default=str)

    # Print summary
    print(f"\n{'='*60}")
    print(f"🚀 BULL MACHINE v1.6.0 BACKTEST RESULTS")
    print(f"{'='*60}")
    print(f"📊 Total Bars: {total_bars}")
    print(f"📈 Signals Generated: {signals_generated}")
    print(f"💰 Total Trades: {len(trades)}")
    print(f"🎯 Win Rate: {win_rate:.2f}% (RC target: ≥50%)")
    print(f"📅 Frequency: {trades_per_month:.2f} trades/month (RC target: 2-4)")
    print(f"💵 Total Return: {total_return_pct:.2f}% (RC target: ≥10%)")
    print(f"📉 Max Drawdown: {max_dd:.2f}% (RC target: ≤9.2%)")
    print(f"📊 Profit Factor: {profit_factor:.2f} (RC target: ≥1.3)")
    print(f"⚡ Sharpe Ratio: {sharpe_ratio:.2f} (RC target: ≥2.2)")
    print(f"💾 Results saved: {filename}")

    # RC Target Assessment
    rc_targets = {
        "Win Rate": win_rate >= 50.0,
        "Frequency": 2.0 <= trades_per_month <= 4.0,
        "Total Return": total_return_pct >= 10.0,
        "Max Drawdown": max_dd <= 9.2,
        "Profit Factor": profit_factor >= 1.3,
        "Sharpe Ratio": sharpe_ratio >= 2.2
    }

    targets_met = sum(rc_targets.values())
    print(f"\n🎯 RC TARGETS: {targets_met}/6 MET")
    for target, met in rc_targets.items():
        status = "✅" if met else "❌"
        print(f"   {status} {target}")

    if targets_met >= 4:
        print(f"\n🚀 RC PROMOTION READY: {targets_met}/6 targets achieved!")
    else:
        print(f"\n🔧 OPTIMIZATION NEEDED: Only {targets_met}/6 targets met")

    return results

if __name__ == "__main__":
    try:
        results = run_v160_backtest()
        print("✅ v1.6.0 backtest completed successfully!")
    except Exception as e:
        print(f"❌ Backtest failed: {e}")
        import traceback
        traceback.print_exc()