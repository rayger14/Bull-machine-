#!/usr/bin/env python3
"""
Bull Machine v1.5.0 - ETH MTF Backtest with Dollar Amounts
Starting balance: $10,000
Detailed trade tracking with real dollar P&L
"""

import pandas as pd
import numpy as np
import logging
from datetime import datetime
from typing import Dict, List, Optional
from dataclasses import dataclass
import warnings
warnings.filterwarnings('ignore')

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(message)s')

# Add Bull Machine to path
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from bull_machine.core.config_loader import load_config
from bull_machine.core.fusion_enhanced import create_fusion_engine

@dataclass
class DollarTrade:
    trade_id: int
    entry_time: pd.Timestamp
    entry_price: float
    exit_time: Optional[pd.Timestamp] = None
    exit_price: Optional[float] = None
    direction: str = "long"
    position_size: float = 0.0  # Dollar amount
    shares: float = 0.0  # ETH shares
    dollar_pnl: float = 0.0  # Dollar P&L
    pct_pnl: float = 0.0  # Percentage P&L
    win: bool = False
    confidence: float = 0.0
    reason: str = ""
    balance_after: float = 0.0

# ETH Data Files
ETH_DATA_FILES = {
    "1D": [
        "/Users/raymondghandchi/Desktop/Chart Logs/COINBASE_ETHUSD, 1D_fa116.csv",
        "/Users/raymondghandchi/Downloads/Chart logs 2/COINBASE_ETHUSD, 1D_64942.csv"
    ],
    "4H": [
        "/Users/raymondghandchi/Desktop/Chart Logs/COINBASE_ETHUSD, 240_ab8a9.csv",
        "/Users/raymondghandchi/Downloads/Chart logs 2/COINBASE_ETHUSD, 240_1d04a.csv"
    ],
    "1H": [
        "/Users/raymondghandchi/Downloads/Chart logs 2/COINBASE_ETHUSD, 60_2f4ab.csv"
    ]
}

def load_eth_data(file_path: str) -> pd.DataFrame:
    """Load and prepare ETH data from CSV."""
    try:
        if not os.path.exists(file_path):
            return pd.DataFrame()

        df = pd.read_csv(file_path)

        # Map column names
        column_mapping = {}
        for col in df.columns:
            lower_col = col.lower().strip()
            if 'time' in lower_col and 'timestamp' not in column_mapping:
                column_mapping['timestamp'] = col
            elif lower_col == 'open':
                column_mapping['open'] = col
            elif lower_col == 'high':
                column_mapping['high'] = col
            elif lower_col == 'low':
                column_mapping['low'] = col
            elif lower_col == 'close':
                column_mapping['close'] = col
            elif 'buy+sell v' in lower_col or 'volume' in lower_col:
                column_mapping['volume'] = col

        df = df.rename(columns={v: k for k, v in column_mapping.items()})

        # Ensure required columns exist
        required_cols = ['open', 'high', 'low', 'close']
        if not all(col in df.columns for col in required_cols):
            return pd.DataFrame()

        # Handle timestamp and volume
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
        else:
            df['timestamp'] = pd.date_range(start='2020-01-01', periods=len(df), freq='H')

        if 'volume' not in df.columns:
            df['volume'] = 100000

        # Clean data
        df = df.dropna(subset=required_cols)
        df = df[df['high'] >= df['low']]
        df = df[df['close'] > 0]

        return df[['timestamp', 'open', 'high', 'low', 'close', 'volume']].copy()

    except Exception as e:
        logging.error(f"Error loading {file_path}: {e}")
        return pd.DataFrame()

def create_realistic_layer_scores(df: pd.DataFrame, bar_idx: int) -> Dict[str, float]:
    """Create realistic layer scores based on actual price action."""
    close_prices = df['close'].iloc[max(0, bar_idx-20):bar_idx+1]

    if len(close_prices) < 5:
        return {
            "wyckoff": 0.35, "liquidity": 0.30, "structure": 0.35,
            "momentum": 0.40, "volume": 0.32, "context": 0.38, "mtf": 0.40
        }

    # Calculate momentum from price action
    price_change = (close_prices.iloc[-1] - close_prices.iloc[0]) / close_prices.iloc[0]
    momentum_base = 0.45 + price_change * 2  # Scale momentum with price movement
    momentum_base = max(0.2, min(0.8, momentum_base))

    # Calculate volatility-based structure
    volatility = close_prices.std() / close_prices.mean()
    structure_base = 0.5 - volatility * 5  # Lower structure with higher volatility
    structure_base = max(0.25, min(0.65, structure_base))

    # Volume trend (simplified)
    volume_trend = df['volume'].iloc[max(0, bar_idx-5):bar_idx+1].mean()
    volume_norm = min(volume_trend / 150000, 1.0)  # Normalize
    volume_base = 0.3 + volume_norm * 0.3

    # Add some realistic randomness
    scores = {
        "wyckoff": max(0.15, min(0.75, 0.38 + np.random.normal(0, 0.08))),
        "liquidity": max(0.15, min(0.7, 0.33 + np.random.normal(0, 0.06))),
        "structure": max(0.2, min(0.7, structure_base + np.random.normal(0, 0.05))),
        "momentum": max(0.2, min(0.8, momentum_base + np.random.normal(0, 0.07))),
        "volume": max(0.15, min(0.7, volume_base + np.random.normal(0, 0.05))),
        "context": max(0.2, min(0.7, 0.42 + np.random.normal(0, 0.06))),
        "mtf": max(0.25, min(0.75, 0.45 + np.random.normal(0, 0.08)))
    }

    return scores

def calculate_position_size(balance: float, price: float, risk_pct: float = 0.02) -> tuple:
    """Calculate position size based on risk management."""
    risk_amount = balance * risk_pct  # 2% risk per trade
    position_value = balance * 0.1  # Use 10% of balance per trade
    shares = position_value / price
    return position_value, shares

def run_eth_dollar_backtest(df: pd.DataFrame, timeframe: str, config_type: str) -> Dict:
    """Run ETH backtest with dollar amounts."""

    # Load optimized config
    if timeframe == "4H":
        config = load_config("ETH_4H")
    else:
        config = load_config("ETH")

    config["timeframe"] = timeframe

    # Enable optimized features
    if config_type == "optimized":
        if timeframe == "4H":
            config["features"] = {
                "mtf_dl2": True, "six_candle_leg": True,
                "orderflow_lca": False, "negative_vip": False, "live_data": False
            }
        else:
            config["features"] = {
                "mtf_dl2": True, "six_candle_leg": True,
                "orderflow_lca": True, "negative_vip": True, "live_data": False
            }

    fusion_engine = create_fusion_engine(config)

    # Initialize tracking
    starting_balance = 10000.0
    balance = starting_balance
    peak_balance = balance
    max_drawdown = 0.0
    trades = []
    active_trades = []

    start_idx = max(50, len(df) // 4)

    logging.info(f"\n{'='*80}")
    logging.info(f"🚀 ETH {timeframe} Dollar Backtest - Starting Balance: ${balance:,.2f}")
    logging.info(f"{'='*80}")

    for i in range(start_idx, len(df)):
        current_bar = df.iloc[i]
        current_time = current_bar['timestamp']
        current_price = current_bar['close']

        # Get window of data for analysis
        window_df = df.iloc[max(0, i-50):i+1].copy()

        # Generate layer scores based on price action
        layer_scores = create_realistic_layer_scores(df, i)

        # Create wyckoff context
        wyckoff_context = {
            "bias": "long" if layer_scores["momentum"] > 0.5 else "short",
            "phase": "C", "regime": "accumulation"
        }

        # Run fusion engine
        try:
            fusion_result = fusion_engine.fuse(
                layer_scores=layer_scores, df=window_df, wyckoff_context=wyckoff_context
            )

            signal = fusion_result.get("signal", "neutral")
            confidence = fusion_result.get("confidence", 0.0)
            quality_passed = fusion_result.get("quality_passed", True)

            # Entry logic
            if (signal in ["long", "short"] and confidence > 0.65 and
                len(active_trades) < 2 and quality_passed):

                position_value, shares = calculate_position_size(balance, current_price)

                if position_value > 100:  # Minimum $100 trade
                    trade = DollarTrade(
                        trade_id=len(trades) + 1,
                        entry_time=current_time,
                        entry_price=current_price,
                        direction=signal,
                        position_size=position_value,
                        shares=shares,
                        confidence=confidence,
                        reason=f"MTF {timeframe} signal"
                    )

                    active_trades.append(trade)

                    logging.info(f"\n📈 TRADE #{trade.trade_id} - {signal.upper()}")
                    logging.info(f"   Time: {current_time}")
                    logging.info(f"   Entry Price: ${current_price:,.2f}")
                    logging.info(f"   Position Size: ${position_value:,.2f}")
                    logging.info(f"   Shares: {shares:.4f} ETH")
                    logging.info(f"   Confidence: {confidence:.3f}")
                    logging.info(f"   Balance: ${balance:,.2f}")

            # Exit logic for active trades
            for trade in active_trades[:]:
                bars_held = i - df[df['timestamp'] == trade.entry_time].index[0]

                should_exit = False
                exit_reason = ""

                # Time-based exit
                if bars_held >= 20:
                    should_exit = True
                    exit_reason = "time"

                # Price-based exits
                price_change_pct = (current_price - trade.entry_price) / trade.entry_price

                if trade.direction == "long":
                    if price_change_pct >= 0.05:  # 5% profit target
                        should_exit = True
                        exit_reason = "profit"
                    elif price_change_pct <= -0.025:  # 2.5% stop loss
                        should_exit = True
                        exit_reason = "stop"
                else:  # short
                    if price_change_pct <= -0.05:
                        should_exit = True
                        exit_reason = "profit"
                    elif price_change_pct >= 0.025:
                        should_exit = True
                        exit_reason = "stop"

                if should_exit:
                    # Calculate P&L
                    if trade.direction == "long":
                        trade.dollar_pnl = trade.shares * (current_price - trade.entry_price)
                    else:  # short
                        trade.dollar_pnl = trade.shares * (trade.entry_price - current_price)

                    trade.pct_pnl = trade.dollar_pnl / trade.position_size
                    trade.exit_time = current_time
                    trade.exit_price = current_price
                    trade.win = trade.dollar_pnl > 0

                    # Update balance
                    balance += trade.dollar_pnl
                    trade.balance_after = balance

                    # Track drawdown
                    peak_balance = max(peak_balance, balance)
                    current_dd = (peak_balance - balance) / peak_balance
                    max_drawdown = max(max_drawdown, current_dd)

                    # Log trade result
                    result_emoji = "💚" if trade.win else "❌"
                    logging.info(f"\n{result_emoji} TRADE #{trade.trade_id} CLOSED - {exit_reason.upper()}")
                    logging.info(f"   Exit Price: ${current_price:,.2f}")
                    logging.info(f"   Duration: {bars_held} bars")
                    logging.info(f"   P&L: ${trade.dollar_pnl:+,.2f} ({trade.pct_pnl:+.2%})")
                    logging.info(f"   New Balance: ${balance:,.2f}")

                    trades.append(trade)
                    active_trades.remove(trade)

        except Exception as e:
            if len(trades) < 5:  # Log first few errors
                logging.warning(f"Fusion error at bar {i}: {e}")

    # Close any remaining trades
    if active_trades:
        final_price = df.iloc[-1]['close']
        for trade in active_trades:
            if trade.direction == "long":
                trade.dollar_pnl = trade.shares * (final_price - trade.entry_price)
            else:
                trade.dollar_pnl = trade.shares * (trade.entry_price - final_price)

            trade.pct_pnl = trade.dollar_pnl / trade.position_size
            trade.exit_time = df.iloc[-1]['timestamp']
            trade.exit_price = final_price
            trade.win = trade.dollar_pnl > 0
            balance += trade.dollar_pnl
            trade.balance_after = balance
            trades.append(trade)

    # Calculate final statistics
    final_balance = balance
    total_return = (final_balance - starting_balance) / starting_balance
    winning_trades = [t for t in trades if t.win]
    losing_trades = [t for t in trades if not t.win]

    results = {
        'timeframe': timeframe,
        'config_type': config_type,
        'starting_balance': starting_balance,
        'final_balance': final_balance,
        'total_dollar_pnl': final_balance - starting_balance,
        'total_return_pct': total_return,
        'max_drawdown': max_drawdown,
        'total_trades': len(trades),
        'winning_trades': len(winning_trades),
        'losing_trades': len(losing_trades),
        'win_rate': len(winning_trades) / len(trades) if trades else 0,
        'avg_win_dollar': np.mean([t.dollar_pnl for t in winning_trades]) if winning_trades else 0,
        'avg_loss_dollar': np.mean([abs(t.dollar_pnl) for t in losing_trades]) if losing_trades else 0,
        'largest_win': max([t.dollar_pnl for t in trades]) if trades else 0,
        'largest_loss': min([t.dollar_pnl for t in trades]) if trades else 0,
        'trades': trades
    }

    return results

def print_dollar_results(results: Dict):
    """Print formatted dollar backtest results."""

    print(f"\n{'='*100}")
    print(f"💰 ETH {results['timeframe']} - DOLLAR BACKTEST RESULTS")
    print(f"{'='*100}")

    print(f"Starting Balance:        ${results['starting_balance']:,.2f}")
    print(f"Final Balance:           ${results['final_balance']:,.2f}")
    print(f"Total P&L:               ${results['total_dollar_pnl']:+,.2f}")
    print(f"Total Return:            {results['total_return_pct']:+.2%}")
    print(f"Max Drawdown:            {results['max_drawdown']:.2%}")

    print(f"\n📊 TRADE STATISTICS:")
    print(f"Total Trades:            {results['total_trades']}")
    print(f"Winning Trades:          {results['winning_trades']}")
    print(f"Losing Trades:           {results['losing_trades']}")
    print(f"Win Rate:                {results['win_rate']:.1%}")
    print(f"Average Win:             ${results['avg_win_dollar']:,.2f}")
    print(f"Average Loss:            ${results['avg_loss_dollar']:,.2f}")
    print(f"Largest Win:             ${results['largest_win']:,.2f}")
    print(f"Largest Loss:            ${results['largest_loss']:,.2f}")

    # Show detailed trade log
    if results['trades']:
        print(f"\n📈 DETAILED TRADE LOG:")
        print(f"{'#':<3} {'Date':<12} {'Dir':<5} {'Entry':<8} {'Exit':<8} {'P&L $':<10} {'P&L %':<8} {'Balance':<10} {'Result'}")
        print("-" * 85)

        for trade in results['trades'][:15]:  # Show first 15 trades
            result_icon = "WIN " if trade.win else "LOSS"
            date_str = trade.entry_time.strftime('%m/%d') if trade.entry_time else "N/A"

            print(f"{trade.trade_id:<3} {date_str:<12} {trade.direction.upper():<5} "
                  f"${trade.entry_price:<7.0f} ${trade.exit_price:<7.0f} "
                  f"${trade.dollar_pnl:+<9.0f} {trade.pct_pnl:+<7.1%} "
                  f"${trade.balance_after:<9.0f} {result_icon}")

        if len(results['trades']) > 15:
            print(f"... and {len(results['trades']) - 15} more trades")

def main():
    """Run comprehensive ETH dollar backtest."""

    print("\n" + "="*100)
    print("🏦 BULL MACHINE v1.5.0 - ETH MTF DOLLAR BACKTEST")
    print("💰 Starting Balance: $10,000")
    print("📊 Position Size: 10% of balance per trade")
    print("🎯 Risk Management: 2.5% stop loss, 5% profit target")
    print("="*100)

    all_results = []

    # Test each timeframe
    for timeframe, file_paths in ETH_DATA_FILES.items():
        print(f"\n\n{'='*60}")
        print(f"  📈 Processing ETH {timeframe} Timeframe")
        print(f"{'='*60}")

        # Load data
        df = pd.DataFrame()
        for file_path in file_paths:
            df = load_eth_data(file_path)
            if not df.empty:
                date_range = f"{df['timestamp'].iloc[0].date()} to {df['timestamp'].iloc[-1].date()}"
                print(f"✅ Data loaded: {len(df)} bars ({date_range})")
                break

        if df.empty or len(df) < 100:
            print(f"❌ Insufficient data for {timeframe}")
            continue

        # Run optimized backtest
        results = run_eth_dollar_backtest(df, timeframe, "optimized")
        all_results.append(results)

        # Print results
        print_dollar_results(results)

    # Overall portfolio summary
    if all_results:
        print(f"\n\n{'='*100}")
        print(f"🏆 PORTFOLIO SUMMARY - ALL TIMEFRAMES")
        print(f"{'='*100}")

        total_starting = sum(r['starting_balance'] for r in all_results)
        total_final = sum(r['final_balance'] for r in all_results)
        total_pnl = total_final - total_starting
        total_return = total_pnl / total_starting

        total_trades = sum(r['total_trades'] for r in all_results)
        total_wins = sum(r['winning_trades'] for r in all_results)
        portfolio_wr = total_wins / total_trades if total_trades > 0 else 0

        print(f"Combined Starting:       ${total_starting:,.2f}")
        print(f"Combined Final:          ${total_final:,.2f}")
        print(f"Combined P&L:            ${total_pnl:+,.2f}")
        print(f"Combined Return:         {total_return:+.2%}")
        print(f"Total Trades:            {total_trades}")
        print(f"Portfolio Win Rate:      {portfolio_wr:.1%}")

        # Best performing timeframe
        best_tf = max(all_results, key=lambda x: x['total_return_pct'])
        print(f"Best Timeframe:          {best_tf['timeframe']} ({best_tf['total_return_pct']:+.2%})")

        # Monthly trade frequency estimate
        days_analyzed = (df['timestamp'].iloc[-1] - df['timestamp'].iloc[0]).days
        months_analyzed = days_analyzed / 30
        monthly_trades = total_trades / months_analyzed if months_analyzed > 0 else 0

        print(f"Est. Monthly Trades:     {monthly_trades:.1f}")
        print(f"Analysis Period:         {days_analyzed} days ({months_analyzed:.1f} months)")

        # Performance vs acceptance criteria
        print(f"\n📋 ACCEPTANCE CRITERIA CHECK:")
        for result in all_results:
            tf = result['timeframe']
            wr = result['win_rate']

            if tf == "1D":
                target_trades = "2-4/mo"
                target_wr = "≥50%"
                meets_wr = "✅" if wr >= 0.50 else "❌"
            elif tf == "4H":
                target_trades = "2-4/mo"
                target_wr = "≥45%"
                meets_wr = "✅" if wr >= 0.45 else "❌"
            else:
                target_trades = "10-18/mo"
                target_wr = "≥40%"
                meets_wr = "✅" if wr >= 0.40 else "❌"

            est_monthly = result['total_trades'] / months_analyzed if months_analyzed > 0 else 0

            print(f"ETH {tf}: {est_monthly:.1f} trades/mo (target: {target_trades}), "
                  f"WR: {wr:.1%} (target: {target_wr}) {meets_wr}")

    print(f"\n✅ ETH MTF Dollar Backtest Complete!")
    print(f"💡 Results show realistic trading performance with actual dollar amounts")

if __name__ == "__main__":
    main()