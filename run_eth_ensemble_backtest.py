#!/usr/bin/env python3
"""
Bull Machine v1.5.1 - ETH Time-Tolerant Ensemble Backtest
✅ Implements realistic MTF alignment with lead-lag tolerance
Target: 2-4 trades/month, ≥50% WR (1D), ≥45% WR (4H)
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

from bull_machine.modules.fusion.v151_core_trader import CoreTraderV151
from bull_machine.backtest.ensemble_mode import EnsembleAligner
from bull_machine.strategy.atr_exits import enhanced_exit_check

def load_and_align_eth_data() -> pd.DataFrame:
    """Load ETH data from multiple timeframes and align."""

    print("📊 Loading ETH Multi-Timeframe Data...")

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

        # Rename columns with TF suffix
        df_renamed = df.rename(columns={
            'open': f'open_{tf}',
            'high': f'high_{tf}',
            'low': f'low_{tf}',
            'close': f'close_{tf}',
            'volume': f'volume_{tf}'
        })

        timeframes[tf] = df_renamed
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

    # Align 4H data
    base_df['timestamp_4H'] = base_df['timestamp'].dt.floor('4h')
    df_4h_aligned = timeframes['4H'].copy()
    df_4h_aligned['timestamp_4H'] = df_4h_aligned['timestamp'].dt.floor('4h')

    base_df = base_df.merge(
        df_4h_aligned[['timestamp_4H', 'open_4H', 'high_4H', 'low_4H', 'close_4H', 'volume_4H']],
        on='timestamp_4H',
        how='left'
    )

    # Align 1D data
    base_df['timestamp_1D'] = base_df['timestamp'].dt.floor('1D')
    df_1d_aligned = timeframes['1D'].copy()
    df_1d_aligned['timestamp_1D'] = df_1d_aligned['timestamp'].dt.floor('1D')

    base_df = base_df.merge(
        df_1d_aligned[['timestamp_1D', 'open_1D', 'high_1D', 'low_1D', 'close_1D', 'volume_1D']],
        on='timestamp_1D',
        how='left'
    )

    # Forward fill missing data
    base_df = base_df.ffill()
    base_df = base_df.dropna()

    print(f"✅ Aligned Dataset: {len(base_df)} 1H bars")
    print(f"💰 ETH Price Range: ${base_df['close_1H'].min():.2f} - ${base_df['close_1H'].max():.2f}")

    return base_df

def get_eth_ensemble_config() -> Dict:
    """Get ETH ensemble configuration with time-tolerant settings."""

    config = {
        "entry_threshold": 0.44,
        "quality_floors": {
            "wyckoff": 0.25,
            "liquidity": 0.25,
            "structure": 0.25,
            "momentum": 0.27,
            "volume": 0.25,
            "context": 0.25,
            "mtf": 0.27
        },
        "features": {
            "mtf_dl2": True,
            "six_candle_leg": True,
            "orderflow_lca": False,  # Disabled for ensemble
            "negative_vip": False,   # Disabled for ensemble
            "wyckoff_phase": True,
            "liquidity_sweep": False,
            "order_blocks": False,
            "wick_magnet": True,
            "live_data": False,
            "use_asset_profiles": True,
            "atr_sizing": True,
            "atr_exits": True
        },
        "risk": {
            "risk_pct": 0.005,  # 0.5% risk per trade
            "atr_window": 14,
            "sl_atr": 1.8,
            "tp_atr": 3.0,
            "trail_atr": 1.2,
            "profit_ladder": [
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
            "min_consensus": 2,        # 2 of 3 TFs must agree (proper ensemble)
            "consensus_penalty": 0.02, # -2% score when only 2/3 agree
            "rolling_k": 4,           # Need 4 of last 5 bars (higher selectivity)
            "rolling_n": 5,
            "lead_lag_window": 3,     # HTF can confirm within 3 bars
            "dynamic_thresholds": True # Enable volatility-based thresholds
        },
        "cooldown_bars": 168,  # 168H (7 day) cooldown between trades
        "timeframe": "ensemble"
    }

    return config

def generate_ensemble_signal(df_aligned: pd.DataFrame, i: int, traders: Dict, aligner: EnsembleAligner) -> Optional[Dict]:
    """Generate time-tolerant ensemble signal."""

    if i < 200:  # Need sufficient history
        return None

    history_length = min(200, i)

    # Extract timeframe data
    tf_data = {}

    # 1H data (most granular)
    tf_data['1H'] = pd.DataFrame({
        'timestamp': df_aligned['timestamp'].iloc[i-history_length:i+1],
        'open': df_aligned['open_1H'].iloc[i-history_length:i+1],
        'high': df_aligned['high_1H'].iloc[i-history_length:i+1],
        'low': df_aligned['low_1H'].iloc[i-history_length:i+1],
        'close': df_aligned['close_1H'].iloc[i-history_length:i+1],
        'volume': df_aligned['volume_1H'].iloc[i-history_length:i+1]
    }).reset_index(drop=True)

    # 4H data
    tf_data['4H'] = pd.DataFrame({
        'timestamp': df_aligned['timestamp'].iloc[i-history_length:i+1:4],
        'open': df_aligned['open_4H'].iloc[i-history_length:i+1:4],
        'high': df_aligned['high_4H'].iloc[i-history_length:i+1:4],
        'low': df_aligned['low_4H'].iloc[i-history_length:i+1:4],
        'close': df_aligned['close_4H'].iloc[i-history_length:i+1:4],
        'volume': df_aligned['volume_4H'].iloc[i-history_length:i+1:4]
    }).reset_index(drop=True)

    # 1D data
    tf_data['1D'] = pd.DataFrame({
        'timestamp': df_aligned['timestamp'].iloc[i-history_length:i+1:24],
        'open': df_aligned['open_1D'].iloc[i-history_length:i+1:24],
        'high': df_aligned['high_1D'].iloc[i-history_length:i+1:24],
        'low': df_aligned['low_1D'].iloc[i-history_length:i+1:24],
        'close': df_aligned['close_1D'].iloc[i-history_length:i+1:24],
        'volume': df_aligned['volume_1D'].iloc[i-history_length:i+1:24]
    }).reset_index(drop=True)

    # Generate scores for each timeframe
    tf_scores = {}

    try:
        for tf in ['1H', '4H', '1D']:
            if tf in traders and len(tf_data[tf]) > 50:
                scores = traders[tf].compute_base_scores(tf_data[tf])
                tf_scores[tf] = scores
    except Exception as e:
        print(f"⚠️  Error generating TF scores: {e}")
        return None

    if len(tf_scores) < 2:
        return None

    # Update ensemble aligner
    aligner.update(tf_scores)

    # Check if ensemble should fire with dynamic threshold
    fire, ensemble_score = aligner.fire(tf_scores, tf_data['1H'])

    if not fire:
        return None

    # Use 1H data for precise entry execution
    entry_price = df_aligned.iloc[i]['close_1H']

    # Determine side based on momentum
    momentum_scores = [tf_scores[tf].get('momentum', 0.5) for tf in tf_scores.keys()]
    avg_momentum = np.mean(momentum_scores)
    side = "LONG" if avg_momentum >= 0.5 else "SHORT"

    # Calculate position size using ATR
    current_atr = tf_data['1H']['high'].tail(14).mean() - tf_data['1H']['low'].tail(14).mean()
    if current_atr <= 0:
        current_atr = entry_price * 0.02  # 2% fallback

    risk_amount = 10000 * 0.005  # 0.5% of 10k balance
    stop_distance = current_atr * 1.8  # 1.8 ATR stop

    if side == "long":
        stop_loss = entry_price - stop_distance
    else:
        stop_loss = entry_price + stop_distance

    quantity = risk_amount / abs(entry_price - stop_loss)

    return {
        "side": side,
        "entry_price": entry_price,
        "quantity": quantity,
        "stop_loss": stop_loss,
        "ensemble_score": ensemble_score,
        "tf_scores": tf_scores,
        "participating_tfs": list(tf_scores.keys()),
        "consensus_strength": len(tf_scores)
    }

def run_eth_ensemble_backtest(df_aligned: pd.DataFrame) -> Dict:
    """Run ETH ensemble backtest with time-tolerant logic."""

    print(f"\n🚀 Running ETH Ensemble Backtest...")

    config = get_eth_ensemble_config()

    # Initialize traders for each timeframe
    traders = {}
    for tf in ['1H', '4H', '1D']:
        tf_config = config.copy()
        tf_config['timeframe'] = tf
        traders[tf] = CoreTraderV151(tf_config)

    # Initialize ensemble aligner
    aligner = EnsembleAligner(config)

    # Trading state
    balance = 10000.0
    active_position = None
    trades = []
    signals_generated = 0
    last_trade_bar = -999
    equity_curve = [balance]
    peak_balance = balance

    # Process each aligned bar
    for i in range(200, len(df_aligned)):
        current_bar = df_aligned.iloc[i]

        # Check exits first
        if active_position:
            # Use 1H data for exits
            exit_history = pd.DataFrame({
                'timestamp': df_aligned['timestamp'].iloc[max(0,i-100):i+1],
                'open': df_aligned['open_1H'].iloc[max(0,i-100):i+1],
                'high': df_aligned['high_1H'].iloc[max(0,i-100):i+1],
                'low': df_aligned['low_1H'].iloc[max(0,i-100):i+1],
                'close': df_aligned['close_1H'].iloc[max(0,i-100):i+1],
                'volume': df_aligned['volume_1H'].iloc[max(0,i-100):i+1]
            })

            # Determine dominant timeframe for exit strategy
            dominant_tf = '1H'  # Default to 1H since we're using 1H bars
            if 'participating_tfs' in active_position:
                if '1D' in active_position['participating_tfs']:
                    dominant_tf = '1D'  # Use 1D exit strategy if 1D participated
                elif '4H' in active_position['participating_tfs']:
                    dominant_tf = '4H'  # Use 4H exit strategy if 4H participated

            exit_result = enhanced_exit_check(exit_history, active_position, config, tf=dominant_tf)

            if exit_result.get("close_position", False):
                closed_percent = exit_result.get("closed_percent", 1.0)
                exit_price = exit_result.get("exit_price", current_bar["close_1H"])
                entry_price = active_position["entry_price"]

                if active_position["side"] == "LONG":
                    pnl_pct = (exit_price - entry_price) / entry_price * 100
                else:
                    pnl_pct = (entry_price - exit_price) / entry_price * 100

                # Calculate dollar P&L for the portion being closed
                position_value = active_position["quantity"] * entry_price
                pnl_dollar = position_value * (pnl_pct / 100) * closed_percent
                balance += pnl_dollar

                # Record trade (partial or full)
                trade_record = {
                    **active_position,
                    "exit_timestamp": current_bar["timestamp"],
                    "exit_bar": i,
                    "exit_price": exit_price,
                    "exit_reason": exit_result.get("reason", "unknown"),
                    "pnl_pct": pnl_pct,
                    "pnl_dollar": pnl_dollar,
                    "bars_held": i - active_position["entry_bar"],
                    "closed_percent": closed_percent
                }
                trades.append(trade_record)

                print(f"EXIT: {active_position['side'].upper()} @ ${exit_price:.2f} "
                      f"({exit_result.get('reason', 'unknown')}) | "
                      f"PnL: {pnl_pct:+.2f}% (${pnl_dollar:+.2f}) | "
                      f"TFs: {active_position.get('participating_tfs', [])}")

                # Handle partial vs full exit
                if closed_percent >= 1.0 or exit_result.get("reason") in ["stop_loss", "wick_magnet", "trailing_stop"]:
                    # Full exit
                    active_position = None
                    last_trade_bar = i
                else:
                    # Partial exit - reduce position size
                    active_position["size"] *= (1.0 - closed_percent)
                    active_position["quantity"] *= (1.0 - closed_percent)

        # Generate ensemble signals
        if not active_position:
            # Check cooldown
            if i - last_trade_bar < config["cooldown_bars"]:
                continue

            ensemble_signal = generate_ensemble_signal(df_aligned, i, traders, aligner)

            if ensemble_signal:
                signals_generated += 1

                # Execute trade
                active_position = {
                    "entry_timestamp": current_bar["timestamp"],
                    "entry_bar": i,
                    **ensemble_signal
                }
                active_position["initial_stop_loss"] = ensemble_signal["stop_loss"]
                active_position["size"] = 1.0  # Full position size for tracking partial exits

                print(f"ENSEMBLE ENTRY: {ensemble_signal['side'].upper()} @ ${ensemble_signal['entry_price']:.2f} | "
                      f"Qty: {ensemble_signal['quantity']:.4f} | "
                      f"Score: {ensemble_signal['ensemble_score']:.3f} | "
                      f"TFs: {ensemble_signal['participating_tfs']} | "
                      f"Consensus: {ensemble_signal['consensus_strength']}/3")

                last_trade_bar = i

        # Update equity curve
        current_equity = balance
        if active_position:
            current_price = current_bar["close_1H"]
            entry_price = active_position["entry_price"]

            if active_position["side"] == "LONG":
                unrealized_pnl = (current_price - entry_price) / entry_price
            else:
                unrealized_pnl = (entry_price - current_price) / entry_price

            position_value = active_position["quantity"] * entry_price
            current_equity += position_value * unrealized_pnl

        equity_curve.append(current_equity)
        if current_equity > peak_balance:
            peak_balance = current_equity

    # Calculate final metrics
    total_trades = len(trades)
    winning_trades = len([t for t in trades if t["pnl_pct"] > 0])
    win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0

    total_return_pct = (balance - 10000.0) / 10000.0 * 100

    # Max drawdown
    max_drawdown = 0
    for equity in equity_curve:
        if peak_balance > 0:
            drawdown = (peak_balance - equity) / peak_balance * 100
            max_drawdown = max(max_drawdown, drawdown)

    # Win/loss metrics
    wins = [t["pnl_pct"] for t in trades if t["pnl_pct"] > 0]
    losses = [t["pnl_pct"] for t in trades if t["pnl_pct"] <= 0]

    avg_win = np.mean(wins) if wins else 0
    avg_loss = np.mean(losses) if losses else 0

    profit_factor = 0
    if avg_loss < 0 and len(losses) > 0:
        total_wins = sum(wins) if wins else 0
        total_losses = abs(sum(losses))
        profit_factor = total_wins / total_losses if total_losses > 0 else float('inf')

    # Trading frequency
    days_traded = (df_aligned.iloc[-1]["timestamp"] - df_aligned.iloc[0]["timestamp"]).days
    trades_per_month = (total_trades / days_traded * 30) if days_traded > 0 else 0

    # Sharpe ratio
    returns = np.diff(equity_curve) / np.array(equity_curve[:-1])
    returns = returns[~np.isnan(returns)]
    sharpe_ratio = 0
    if len(returns) > 0 and np.std(returns) > 0:
        sharpe_ratio = (np.mean(returns) / np.std(returns)) * np.sqrt(252 * 24)  # Hourly data

    return {
        "total_bars": len(df_aligned),
        "signals_generated": signals_generated,
        "total_trades": total_trades,
        "winning_trades": winning_trades,
        "win_rate": win_rate,
        "trades_per_month": trades_per_month,
        "initial_balance": 10000.0,
        "final_balance": balance,
        "total_return_pct": total_return_pct,
        "max_drawdown": max_drawdown,
        "avg_win": avg_win,
        "avg_loss": avg_loss,
        "profit_factor": profit_factor,
        "sharpe_ratio": sharpe_ratio,
        "trades": trades,
        "equity_curve": equity_curve
    }

def main():
    """Run ETH time-tolerant ensemble backtest."""
    print("="*70)
    print("🚀 Bull Machine v1.5.1 - ETH Time-Tolerant Ensemble Backtest")
    print("📊 Testing realistic MTF alignment with lead-lag tolerance")
    print("🎯 Target: 2-4 trades/month, ≥50% WR, ≥10% return")
    print("="*70)

    try:
        # Load and align ETH data
        df_aligned = load_and_align_eth_data()

        # Run ensemble backtest
        results = run_eth_ensemble_backtest(df_aligned)

        # Display results
        print(f"\n{'='*60}")
        print(f"📈 ETH ENSEMBLE BACKTEST RESULTS")
        print(f"{'='*60}")

        print(f"  📅 Period: {df_aligned['timestamp'].min().strftime('%Y-%m-%d')} to {df_aligned['timestamp'].max().strftime('%Y-%m-%d')}")
        print(f"  📊 Total 1H Bars: {results['total_bars']:,}")
        print(f"  🎯 Ensemble Signals: {results['signals_generated']}")
        print(f"  💼 Trades Executed: {results['total_trades']}")
        print(f"  🏆 Win Rate: {results['win_rate']:.1f}% ({results['winning_trades']}/{results['total_trades']})")
        print(f"  📅 Trading Frequency: {results['trades_per_month']:.2f} trades/month")
        print(f"  💰 Starting Balance: ${results['initial_balance']:,.2f}")
        print(f"  💰 Final Balance: ${results['final_balance']:,.2f}")
        print(f"  📈 Total Return: {results['total_return_pct']:+.2f}%")
        print(f"  📉 Max Drawdown: {results['max_drawdown']:.2f}%")
        print(f"  ⬆️  Avg Win: {results['avg_win']:+.2f}%")
        print(f"  ⬇️  Avg Loss: {results['avg_loss']:+.2f}%")
        print(f"  ⚖️  Profit Factor: {results['profit_factor']:.2f}")
        print(f"  📊 Sharpe Ratio: {results['sharpe_ratio']:.2f}")

        # Check RC targets
        print(f"\n{'='*60}")
        print(f"🎯 RC PROMOTION TARGET ANALYSIS")
        print(f"{'='*60}")

        targets_met = 0
        total_targets = 6

        # Trading frequency (2-4 trades/month)
        freq_met = 2.0 <= results['trades_per_month'] <= 4.0
        print(f"  ✅ Trading Frequency: {results['trades_per_month']:.2f}/month {'✅' if freq_met else '❌'} (Target: 2-4)")
        if freq_met: targets_met += 1

        # Win rate (≥50%)
        wr_met = results['win_rate'] >= 50.0
        print(f"  ✅ Win Rate: {results['win_rate']:.1f}% {'✅' if wr_met else '❌'} (Target: ≥50%)")
        if wr_met: targets_met += 1

        # Total return (≥10%)
        return_met = results['total_return_pct'] >= 10.0
        print(f"  ✅ Total Return: {results['total_return_pct']:+.2f}% {'✅' if return_met else '❌'} (Target: ≥10%)")
        if return_met: targets_met += 1

        # Max drawdown (≤9.2%)
        dd_met = results['max_drawdown'] <= 9.2
        print(f"  ✅ Max Drawdown: {results['max_drawdown']:.2f}% {'✅' if dd_met else '❌'} (Target: ≤9.2%)")
        if dd_met: targets_met += 1

        # Profit factor (≥1.3)
        pf_met = results['profit_factor'] >= 1.3
        print(f"  ✅ Profit Factor: {results['profit_factor']:.2f} {'✅' if pf_met else '❌'} (Target: ≥1.3)")
        if pf_met: targets_met += 1

        # Sharpe ratio (≥2.2)
        sharpe_met = results['sharpe_ratio'] >= 2.2
        print(f"  ✅ Sharpe Ratio: {results['sharpe_ratio']:.2f} {'✅' if sharpe_met else '❌'} (Target: ≥2.2)")
        if sharpe_met: targets_met += 1

        print(f"\n🏆 RC TARGETS MET: {targets_met}/{total_targets}")

        if targets_met >= 5:
            print("🎉 READY FOR RC PROMOTION!")
        elif targets_met >= 3:
            print("🔧 NEEDS MINOR TUNING")
        else:
            print("⚠️  NEEDS SIGNIFICANT OPTIMIZATION")

        # Show trade details
        if results['trades']:
            print(f"\n📋 ENSEMBLE TRADE HISTORY:")
            for i, trade in enumerate(results['trades'], 1):
                side = trade['side'].upper()
                entry = trade['entry_price']
                exit_p = trade.get('exit_price', 'Active')
                pnl = trade.get('pnl_pct', 0)
                reason = trade.get('exit_reason', 'active')
                tfs = trade.get('participating_tfs', [])
                timestamp = trade['entry_timestamp'].strftime('%Y-%m-%d %H:%M')

                print(f"  #{i:2d} {timestamp} | {side:5s} | "
                      f"${entry:7.2f} → ${exit_p:7.2f} | "
                      f"PnL: {pnl:+6.2f}% | TFs: {str(tfs):15s} | {reason}")

        # Save results
        output_file = f"eth_ensemble_backtest_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

        # Convert for JSON
        json_results = {}
        for key, value in results.items():
            if key == 'trades':
                json_results[key] = []
                for trade in value:
                    trade_record = {}
                    for k, v in trade.items():
                        if hasattr(v, 'strftime'):
                            trade_record[k] = v.isoformat()
                        elif isinstance(v, (np.integer, np.floating)):
                            trade_record[k] = float(v)
                        else:
                            trade_record[k] = v
                    json_results[key].append(trade_record)
            else:
                if isinstance(value, (np.integer, np.floating)):
                    json_results[key] = float(value)
                elif isinstance(value, list):
                    json_results[key] = [float(x) if isinstance(x, (np.integer, np.floating)) else x for x in value]
                else:
                    json_results[key] = value

        with open(output_file, 'w') as f:
            json.dump(json_results, f, indent=2, default=str)

        print(f"\n💾 Detailed results saved to: {output_file}")

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

    print(f"\n{'='*70}")
    print(f"✅ ETH Ensemble Backtest Complete!")
    print(f"{'='*70}")

if __name__ == "__main__":
    main()