#!/usr/bin/env python3
"""
Bull Machine v1.5.1 - DIAGNOSTIC Individual Timeframe Backtest
⚠️  FOR DEBUGGING ONLY - NOT PRODUCTION TESTING
Use run_btc_ensemble_backtest.py for actual Bull Machine functionality testing
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
from bull_machine.strategy.atr_exits import enhanced_exit_check

# BTC data configuration
BTC_DATA_FILES = {
    "1D": "/Users/raymondghandchi/Downloads/Chart logs 2/COINBASE_BTCUSD, 1D_85c84.csv",
    "1H": "/Users/raymondghandchi/Downloads/Chart logs 2/COINBASE_BTCUSD, 60_50ad4.csv",
    "4H": "/Users/raymondghandchi/Downloads/Chart logs 2/COINBASE_BTCUSD, 240_c2b76.csv"
}

def load_btc_data(file_path: str) -> pd.DataFrame:
    """Load and prepare BTC data from CSV."""
    try:
        print(f"Loading: {Path(file_path).name}")
        df = pd.read_csv(file_path)

        # Check required columns
        required_cols = ['open', 'high', 'low', 'close']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            print(f"❌ Missing columns: {missing_cols}")
            return pd.DataFrame()

        # Handle timestamp
        if 'time' in df.columns:
            df['timestamp'] = pd.to_datetime(df['time'], unit='s')
        elif 'timestamp' not in df.columns:
            df['timestamp'] = pd.date_range(start='2020-01-01', periods=len(df), freq='H')

        # Handle volume
        if 'volume' not in df.columns:
            if 'BUY+SELL V' in df.columns:
                df['volume'] = df['BUY+SELL V']
            else:
                df['volume'] = df['close'] * 100  # Synthetic volume based on price

        # Clean and sort data
        df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']].copy()
        df = df.dropna().sort_values('timestamp').reset_index(drop=True)

        print(f"✅ Loaded {len(df)} bars")
        print(f"📅 Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
        print(f"💰 Price range: ${df['close'].min():.2f} - ${df['close'].max():.2f}")

        return df

    except Exception as e:
        print(f"❌ Error loading {file_path}: {e}")
        return pd.DataFrame()

def get_timeframe_config(timeframe: str) -> Dict:
    """Get optimized config for each timeframe."""

    base_config = {
        "initial_balance": 10000.0,
        "risk": {
            "risk_pct": 0.01,  # 1% risk per trade
            "atr_window": 14,
            "sl_atr": 2.0,
            "tp_atr": 4.0,
            "trail_atr": 1.5
        },
        "quality_floors": {
            'wyckoff': 0.25,
            'liquidity': 0.23,
            'structure': 0.25,
            'momentum': 0.25,
            'volume': 0.23,
            'context': 0.25,
            'mtf': 0.25
        },
        "features": {
            "mtf_dl2": True,
            "six_candle_leg": True,
            "orderflow_lca": True,
            "negative_vip": True,
            "live_data": False,
            "atr_sizing": True,
            "atr_exits": True,
            "ensemble_htf_bias": False  # Disable strict ensemble requirements initially
        },
        "exit_strategy": {
            "profit_ladder": {
                "enabled": True,
                "levels": [
                    {"ratio": 1.5, "percent": 0.25},  # 25% at 1.5R
                    {"ratio": 2.5, "percent": 0.50},  # 50% at 2.5R
                    {"ratio": 4.0, "percent": 0.25}   # 25% at 4R+
                ]
            },
            "trailing_stop": {
                "enabled": True,
                "trigger_ratio": 2.0,  # Start trailing at 2R
                "trail_distance": 1.0   # Trail 1 ATR behind
            }
        }
    }

    # Timeframe-specific adjustments - Very permissive for signal generation
    if timeframe == "1D":
        base_config.update({
            "timeframe": "1D",
            "entry_threshold": 0.25,  # Very low for initial testing
            "cooldown_bars": 2,
            "mtf_ensemble_threshold": 0.30
        })
    elif timeframe == "4H":
        base_config.update({
            "timeframe": "4H",
            "entry_threshold": 0.25,
            "cooldown_bars": 6,
            "mtf_ensemble_threshold": 0.30
        })
        # Disable some features for 4H
        base_config["features"]["orderflow_lca"] = False
        base_config["features"]["negative_vip"] = False
    elif timeframe == "1H":
        base_config.update({
            "timeframe": "1H",
            "entry_threshold": 0.25,
            "cooldown_bars": 12,
            "mtf_ensemble_threshold": 0.30
        })
        # More permissive for 1H
        base_config["quality_floors"]["momentum"] = 0.20
        base_config["features"]["orderflow_lca"] = False
        base_config["features"]["negative_vip"] = False

    return base_config

def run_comprehensive_backtest(df: pd.DataFrame, config: Dict) -> Dict:
    """Run comprehensive backtest with v1.5.1 Core Trader."""

    print(f"\n🚀 Running {config['timeframe']} backtest...")
    print(f"Entry threshold: {config['entry_threshold']}")
    print(f"Cooldown bars: {config['cooldown_bars']}")

    # Initialize Core Trader
    trader = CoreTraderV151(config)

    # Trading state
    balance = config["initial_balance"]
    equity_curve = [balance]
    peak_balance = balance
    active_position = None
    trades = []
    signals_generated = 0
    last_trade_bar = -999

    # Process each bar (need sufficient history)
    min_history = 150
    for i in range(min_history, len(df)):
        current_bar = df.iloc[i]
        history = df.iloc[max(0, i-200):i+1]

        # Check exit first if we have a position
        if active_position:
            exit_result = enhanced_exit_check(history, active_position, config)

            if exit_result.get("close_position", False):
                # Calculate P&L
                entry_price = active_position["entry_price"]
                exit_price = current_bar["close"]

                if active_position["side"] == "long":
                    pnl_pct = (exit_price - entry_price) / entry_price * 100
                else:  # short
                    pnl_pct = (entry_price - exit_price) / entry_price * 100

                # Calculate dollar P&L
                position_value = active_position["position_size"] * entry_price
                pnl_dollar = position_value * (pnl_pct / 100)
                balance += pnl_dollar

                # Record trade
                trade_record = {
                    **active_position,
                    "exit_timestamp": current_bar["timestamp"],
                    "exit_bar": i,
                    "exit_price": exit_price,
                    "exit_reason": exit_result.get("exit_reason", "unknown"),
                    "pnl_pct": pnl_pct,
                    "pnl_dollar": pnl_dollar,
                    "bars_held": i - active_position["entry_bar"]
                }
                trades.append(trade_record)

                print(f"EXIT: {active_position['side'].upper()} @ ${exit_price:.2f} "
                      f"({exit_result.get('exit_reason', 'unknown')}) | PnL: {pnl_pct:+.2f}% (${pnl_dollar:+.2f})")

                active_position = None
                last_trade_bar = i

        # Generate entry signals if no active position
        if not active_position:
            # Check entry using Core Trader
            trade_plan = trader.check_entry(history, last_trade_bar, config, balance)

            if trade_plan:
                signals_generated += 1

                # Extract trade plan details
                entry_price = trade_plan["entry_price"]
                side = trade_plan["side"]
                quantity = trade_plan["quantity"]  # Use 'quantity' not 'position_size'
                stop_loss = trade_plan.get("stop_loss")

                # Convert quantity to position size (shares)
                if quantity and entry_price:
                    position_size = quantity / entry_price
                else:
                    position_size = balance * 0.01 / entry_price  # 1% fallback

                active_position = {
                    "entry_timestamp": current_bar["timestamp"],
                    "entry_bar": i,
                    "entry_price": entry_price,
                    "side": side,
                    "position_size": position_size,
                    "quantity": quantity,
                    "stop_loss": stop_loss,
                    "initial_stop_loss": stop_loss,  # For R calculation
                    "fusion_score": trade_plan.get("weighted_score", 0),
                    "layer_scores": trade_plan.get("layer_scores", {}),
                    "confidence": "high"
                }

                print(f"ENTRY: {side.upper()} @ ${entry_price:.2f} "
                      f"| Qty: ${quantity:.2f} | Size: {position_size:.4f} | Stop: ${stop_loss:.2f} "
                      f"| Score: {trade_plan.get('weighted_score', 0):.3f}")

                last_trade_bar = i

        # Update equity curve
        current_equity = balance
        if active_position:
            # Add unrealized P&L
            current_price = current_bar["close"]
            entry_price = active_position["entry_price"]

            if active_position["side"] == "long":
                unrealized_pnl = (current_price - entry_price) / entry_price
            else:
                unrealized_pnl = (entry_price - current_price) / entry_price

            position_value = active_position["position_size"] * entry_price
            current_equity += position_value * unrealized_pnl

        equity_curve.append(current_equity)

        # Track peak for drawdown
        if current_equity > peak_balance:
            peak_balance = current_equity

    # Calculate final metrics
    total_trades = len(trades)
    winning_trades = len([t for t in trades if t["pnl_pct"] > 0])
    losing_trades = total_trades - winning_trades

    win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0

    total_return_pct = (balance - config["initial_balance"]) / config["initial_balance"] * 100

    # Calculate max drawdown
    max_drawdown = 0
    for equity in equity_curve:
        if peak_balance > 0:
            drawdown = (peak_balance - equity) / peak_balance * 100
            max_drawdown = max(max_drawdown, drawdown)

    # Average win/loss
    wins = [t["pnl_pct"] for t in trades if t["pnl_pct"] > 0]
    losses = [t["pnl_pct"] for t in trades if t["pnl_pct"] <= 0]

    avg_win = np.mean(wins) if wins else 0
    avg_loss = np.mean(losses) if losses else 0

    profit_factor = 0
    if avg_loss < 0:
        total_wins = sum(wins)
        total_losses = abs(sum(losses))
        profit_factor = total_wins / total_losses if total_losses > 0 else float('inf')

    # Calculate Sharpe ratio
    returns = np.diff(equity_curve) / np.array(equity_curve[:-1])
    returns = returns[~np.isnan(returns)]  # Remove NaN values
    sharpe_ratio = 0
    if len(returns) > 0 and np.std(returns) > 0:
        sharpe_ratio = (np.mean(returns) / np.std(returns)) * np.sqrt(252)  # Annualized

    # Trading frequency (trades per month)
    days_traded = (df.iloc[-1]["timestamp"] - df.iloc[0]["timestamp"]).days
    trades_per_month = (total_trades / days_traded * 30) if days_traded > 0 else 0

    return {
        "timeframe": config["timeframe"],
        "total_bars": len(df),
        "signals_generated": signals_generated,
        "total_trades": total_trades,
        "winning_trades": winning_trades,
        "losing_trades": losing_trades,
        "win_rate": win_rate,
        "trades_per_month": trades_per_month,
        "initial_balance": config["initial_balance"],
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
    """Run comprehensive BTC backtests."""
    print("=" * 70)
    print("🚀 Bull Machine v1.5.1 - BTC Comprehensive Backtest")
    print("=" * 70)

    all_results = {}

    # Process each timeframe
    for timeframe, file_path in BTC_DATA_FILES.items():
        print(f"\n{'='*50}")
        print(f"📊 Processing BTC {timeframe}")
        print(f"{'='*50}")

        # Check if file exists
        if not Path(file_path).exists():
            print(f"⚠️  File not found: {file_path}")
            continue

        # Load data
        df = load_btc_data(file_path)
        if df.empty:
            print(f"❌ Failed to load data for {timeframe}")
            continue

        # Get configuration
        config = get_timeframe_config(timeframe)

        # Run backtest
        results = run_comprehensive_backtest(df, config)
        all_results[timeframe] = results

        # Display results
        print(f"\n📈 {timeframe} Results:")
        print(f"  Data Period: {df['timestamp'].min().strftime('%Y-%m-%d')} to {df['timestamp'].max().strftime('%Y-%m-%d')}")
        print(f"  Total Bars: {results['total_bars']:,}")
        print(f"  Signals Generated: {results['signals_generated']}")
        print(f"  Trades Taken: {results['total_trades']}")
        print(f"  Win Rate: {results['win_rate']:.1f}% ({results['winning_trades']}/{results['total_trades']})")
        print(f"  Trades/Month: {results['trades_per_month']:.2f}")
        print(f"  Total Return: {results['total_return_pct']:+.2f}%")
        print(f"  Final Balance: ${results['final_balance']:,.2f}")
        print(f"  Max Drawdown: {results['max_drawdown']:.2f}%")
        print(f"  Avg Win: {results['avg_win']:+.2f}%")
        print(f"  Avg Loss: {results['avg_loss']:+.2f}%")
        print(f"  Profit Factor: {results['profit_factor']:.2f}")
        print(f"  Sharpe Ratio: {results['sharpe_ratio']:.2f}")

        # Show recent trades
        if results['trades']:
            print(f"\n  Recent Trades:")
            for trade in results['trades'][-5:]:
                side = trade['side'].upper()
                entry = trade['entry_price']
                exit_p = trade.get('exit_price', 'Active')
                pnl = trade.get('pnl_pct', 0)
                reason = trade.get('exit_reason', 'active')
                timestamp = trade['entry_timestamp'].strftime('%Y-%m-%d')

                print(f"    {timestamp} | {side} | Entry: ${entry:.2f} | "
                      f"Exit: ${exit_p:.2f} | PnL: {pnl:+.2f}% | {reason}")

    # Summary across all timeframes
    if all_results:
        print(f"\n{'='*70}")
        print(f"📊 COMPREHENSIVE BTC BACKTEST SUMMARY")
        print(f"{'='*70}")

        total_trades = sum(r['total_trades'] for r in all_results.values())
        total_wins = sum(r['winning_trades'] for r in all_results.values())
        avg_return = np.mean([r['total_return_pct'] for r in all_results.values()])

        print(f"\n🎯 Overall Performance:")
        print(f"  Total Trades Across All TFs: {total_trades}")
        print(f"  Total Wins: {total_wins}")
        print(f"  Combined Win Rate: {(total_wins/total_trades*100) if total_trades > 0 else 0:.1f}%")
        print(f"  Average Return Per TF: {avg_return:+.2f}%")

        # Best performing timeframe
        best_tf = max(all_results.items(), key=lambda x: x[1]['total_return_pct'])
        worst_tf = min(all_results.items(), key=lambda x: x[1]['total_return_pct'])

        print(f"\n🏆 Best Performing: {best_tf[0]}")
        print(f"  Return: {best_tf[1]['total_return_pct']:+.2f}%")
        print(f"  Win Rate: {best_tf[1]['win_rate']:.1f}%")
        print(f"  Trades: {best_tf[1]['total_trades']}")

        print(f"\n📉 Worst Performing: {worst_tf[0]}")
        print(f"  Return: {worst_tf[1]['total_return_pct']:+.2f}%")
        print(f"  Win Rate: {worst_tf[1]['win_rate']:.1f}%")
        print(f"  Trades: {worst_tf[1]['total_trades']}")

        # Save results
        output_file = f"btc_comprehensive_backtest_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

        # Convert numpy types for JSON serialization
        def convert_types(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, list) and len(obj) > 0 and hasattr(obj[0], '__len__'):
                # Handle nested arrays/lists
                return [convert_types(item) for item in obj]
            elif hasattr(obj, 'isna') and callable(obj.isna):
                # Handle pandas objects with isna method
                try:
                    if obj.isna():
                        return None
                except (ValueError, TypeError):
                    pass
            return obj

        json_results = {}
        for tf, results in all_results.items():
            json_results[tf] = {}
            for key, value in results.items():
                if key == 'trades':
                    # Convert trade records
                    json_results[tf][key] = []
                    for trade in value:
                        trade_record = {}
                        for k, v in trade.items():
                            if hasattr(v, 'strftime'):  # datetime
                                trade_record[k] = v.isoformat()
                            else:
                                trade_record[k] = convert_types(v)
                        json_results[tf][key].append(trade_record)
                else:
                    json_results[tf][key] = convert_types(value)

        with open(output_file, 'w') as f:
            json.dump(json_results, f, indent=2, default=str)

        print(f"\n💾 Detailed results saved to: {output_file}")

    print(f"\n{'='*70}")
    print(f"✅ BTC Comprehensive Backtest Complete!")
    print(f"{'='*70}")

if __name__ == "__main__":
    main()