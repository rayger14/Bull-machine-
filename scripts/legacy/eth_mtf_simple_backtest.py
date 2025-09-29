#!/usr/bin/env python3
"""
Bull Machine v1.5.0 - Simplified ETH MTF Backtest
Starting balance: $10,000 - Focus on core MTF functionality
"""

import pandas as pd
import numpy as np
import logging
from datetime import datetime
from typing import Dict, List, Optional
from dataclasses import dataclass

logging.basicConfig(level=logging.INFO, format='%(message)s')

@dataclass
class Trade:
    id: int
    entry_time: str
    entry_price: float
    exit_time: Optional[str] = None
    exit_price: Optional[float] = None
    direction: str = "long"
    position_dollars: float = 0.0
    shares: float = 0.0
    dollar_pnl: float = 0.0
    pct_return: float = 0.0
    win: bool = False
    confidence: float = 0.0
    duration_bars: int = 0
    balance_after: float = 0.0

# ETH Data Files
ETH_DATA_FILES = {
    "1D": "/Users/raymondghandchi/Desktop/Chart Logs/COINBASE_ETHUSD, 1D_fa116.csv",
    "4H": "/Users/raymondghandchi/Desktop/Chart Logs/COINBASE_ETHUSD, 240_ab8a9.csv",
    "1H": "/Users/raymondghandchi/Downloads/Chart logs 2/COINBASE_ETHUSD, 60_2f4ab.csv"
}

def load_eth_data(file_path: str) -> pd.DataFrame:
    """Load ETH data from CSV."""
    try:
        if not file_path or not pd.api.types.is_file_like(open(file_path)):
            return pd.DataFrame()
    except:
        return pd.DataFrame()

    try:
        df = pd.read_csv(file_path)

        # Map columns
        column_mapping = {}
        for col in df.columns:
            lower_col = col.lower().strip()
            if 'time' in lower_col:
                column_mapping['timestamp'] = col
            elif lower_col == 'open':
                column_mapping['open'] = col
            elif lower_col == 'high':
                column_mapping['high'] = col
            elif lower_col == 'low':
                column_mapping['low'] = col
            elif lower_col == 'close':
                column_mapping['close'] = col
            elif 'volume' in lower_col:
                column_mapping['volume'] = col

        df = df.rename(columns={v: k for k, v in column_mapping.items()})

        if not all(col in df.columns for col in ['open', 'high', 'low', 'close']):
            return pd.DataFrame()

        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
        else:
            df['timestamp'] = pd.date_range(start='2020-01-01', periods=len(df), freq='D')

        if 'volume' not in df.columns:
            df['volume'] = 100000

        df = df.dropna(subset=['open', 'high', 'low', 'close'])
        df = df[df['close'] > 0]

        return df[['timestamp', 'open', 'high', 'low', 'close', 'volume']].reset_index(drop=True)

    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return pd.DataFrame()

def calculate_mtf_signals(df: pd.DataFrame, timeframe: str) -> pd.DataFrame:
    """Calculate MTF trading signals based on multiple factors."""

    # Calculate technical indicators
    df['sma_20'] = df['close'].rolling(20).mean()
    df['sma_50'] = df['close'].rolling(50).mean()
    df['rsi'] = calculate_rsi(df['close'], 14)
    df['bb_upper'], df['bb_lower'] = calculate_bollinger_bands(df['close'], 20)

    # Price momentum
    df['momentum'] = df['close'] / df['close'].shift(10) - 1
    df['volatility'] = df['close'].rolling(20).std()

    # Volume analysis
    df['volume_ma'] = df['volume'].rolling(20).mean()
    df['volume_ratio'] = df['volume'] / df['volume_ma']

    # MTF Signal Generation
    signals = []
    confidence_scores = []

    for i in range(len(df)):
        if i < 50:  # Need enough data
            signals.append('neutral')
            confidence_scores.append(0.0)
            continue

        signal_score = 0.0
        confidence = 0.0

        # Trend following signals
        if df['close'].iloc[i] > df['sma_20'].iloc[i] > df['sma_50'].iloc[i]:
            signal_score += 0.3  # Uptrend
        elif df['close'].iloc[i] < df['sma_20'].iloc[i] < df['sma_50'].iloc[i]:
            signal_score -= 0.3  # Downtrend

        # Momentum signals
        if df['momentum'].iloc[i] > 0.02:  # Strong momentum up
            signal_score += 0.2
        elif df['momentum'].iloc[i] < -0.02:  # Strong momentum down
            signal_score -= 0.2

        # RSI signals
        rsi_val = df['rsi'].iloc[i]
        if 30 < rsi_val < 70:  # Not overbought/oversold
            if rsi_val > 50:
                signal_score += 0.1
            else:
                signal_score -= 0.1

        # Volume confirmation
        if df['volume_ratio'].iloc[i] > 1.2:  # High volume
            signal_score *= 1.2  # Amplify signal

        # Volatility filter (avoid high volatility periods)
        vol_percentile = df['volatility'].iloc[max(0, i-100):i].quantile(0.8)
        if df['volatility'].iloc[i] < vol_percentile:
            confidence += 0.3

        # Timeframe-specific adjustments
        if timeframe == "1D":
            signal_score *= 1.1  # Daily signals are more reliable
            confidence += 0.1
        elif timeframe == "4H":
            signal_score *= 0.9  # 4H needs more confirmation
        elif timeframe == "1H":
            signal_score *= 0.8  # Hourly is noisier
            confidence -= 0.1

        # Final signal determination
        confidence = max(0.0, min(1.0, abs(signal_score) + confidence))

        if signal_score > 0.3:
            signals.append('long')
        elif signal_score < -0.3:
            signals.append('short')
        else:
            signals.append('neutral')

        confidence_scores.append(confidence)

    df['signal'] = signals
    df['confidence'] = confidence_scores

    return df

def calculate_rsi(prices: pd.Series, period: int = 14) -> pd.Series:
    """Calculate RSI indicator."""
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def calculate_bollinger_bands(prices: pd.Series, period: int = 20, std_dev: float = 2) -> tuple:
    """Calculate Bollinger Bands."""
    sma = prices.rolling(period).mean()
    std = prices.rolling(period).std()
    upper_band = sma + (std * std_dev)
    lower_band = sma - (std * std_dev)
    return upper_band, lower_band

def run_eth_backtest(df: pd.DataFrame, timeframe: str) -> Dict:
    """Run ETH backtest with detailed tracking."""

    # Generate signals
    df = calculate_mtf_signals(df, timeframe)

    # Initialize portfolio
    starting_balance = 10000.0
    balance = starting_balance
    position_size_pct = 0.10  # Use 10% of balance per trade
    max_risk_pct = 0.025  # 2.5% stop loss
    take_profit_pct = 0.05  # 5% profit target

    trades = []
    active_trades = []
    trade_counter = 0
    peak_balance = balance
    max_drawdown = 0.0

    print(f"\n{'='*80}")
    print(f"🚀 ETH {timeframe} MTF BACKTEST - Starting Balance: ${balance:,.2f}")
    print(f"{'='*80}")
    print(f"📊 Strategy: MTF Confluence with {position_size_pct:.0%} position sizing")
    print(f"🎯 Risk: {max_risk_pct:.1%} stop loss, {take_profit_pct:.1%} profit target")
    print(f"📈 Data: {len(df)} bars from {df['timestamp'].iloc[0].date()} to {df['timestamp'].iloc[-1].date()}")

    # Start from index 60 to have enough indicator data
    for i in range(60, len(df)):
        current_bar = df.iloc[i]
        current_time = current_bar['timestamp']
        current_price = current_bar['close']
        signal = current_bar['signal']
        confidence = current_bar['confidence']

        # Exit logic for active trades
        for trade in active_trades[:]:
            entry_idx = df[df['timestamp'] == pd.to_datetime(trade.entry_time)].index[0]
            bars_held = i - entry_idx

            price_change = (current_price - trade.entry_price) / trade.entry_price

            should_exit = False
            exit_reason = ""

            # Time-based exit
            max_hold_periods = {"1D": 30, "4H": 20, "1H": 15}
            if bars_held >= max_hold_periods.get(timeframe, 20):
                should_exit = True
                exit_reason = "time"

            # Profit/loss exits
            if trade.direction == "long":
                if price_change >= take_profit_pct:
                    should_exit = True
                    exit_reason = "profit"
                elif price_change <= -max_risk_pct:
                    should_exit = True
                    exit_reason = "stop"
            else:  # short
                if price_change <= -take_profit_pct:
                    should_exit = True
                    exit_reason = "profit"
                elif price_change >= max_risk_pct:
                    should_exit = True
                    exit_reason = "stop"

            # Signal reversal exit
            if (trade.direction == "long" and signal == "short") or \
               (trade.direction == "short" and signal == "long"):
                if confidence > 0.7 and bars_held >= 3:
                    should_exit = True
                    exit_reason = "reversal"

            if should_exit:
                # Calculate P&L
                if trade.direction == "long":
                    trade.dollar_pnl = trade.shares * (current_price - trade.entry_price)
                else:
                    trade.dollar_pnl = trade.shares * (trade.entry_price - current_price)

                trade.pct_return = trade.dollar_pnl / trade.position_dollars
                trade.exit_time = current_time.strftime('%Y-%m-%d %H:%M')
                trade.exit_price = current_price
                trade.win = trade.dollar_pnl > 0
                trade.duration_bars = bars_held

                # Update balance
                balance += trade.dollar_pnl
                trade.balance_after = balance

                # Track drawdown
                peak_balance = max(peak_balance, balance)
                drawdown = (peak_balance - balance) / peak_balance
                max_drawdown = max(max_drawdown, drawdown)

                # Log trade
                result_emoji = "💚" if trade.win else "❌"
                print(f"{result_emoji} Trade #{trade.id}: {trade.direction.upper()} "
                      f"${trade.entry_price:.0f}→${current_price:.0f} "
                      f"({bars_held}bars) P&L: ${trade.dollar_pnl:+.0f} ({trade.pct_return:+.1%}) "
                      f"Balance: ${balance:.0f}")

                trades.append(trade)
                active_trades.remove(trade)

        # Entry logic
        min_confidence = {"1D": 0.65, "4H": 0.70, "1H": 0.75}

        if (signal in ['long', 'short'] and
            confidence >= min_confidence.get(timeframe, 0.7) and
            len(active_trades) < 2):  # Max 2 concurrent trades

            position_dollars = balance * position_size_pct

            if position_dollars >= 100:  # Minimum $100 trade
                trade_counter += 1
                shares = position_dollars / current_price

                trade = Trade(
                    id=trade_counter,
                    entry_time=current_time.strftime('%Y-%m-%d %H:%M'),
                    entry_price=current_price,
                    direction=signal,
                    position_dollars=position_dollars,
                    shares=shares,
                    confidence=confidence
                )

                active_trades.append(trade)

                print(f"📈 Trade #{trade.id}: {signal.upper()} @ ${current_price:.0f} "
                      f"Size: ${position_dollars:.0f} ({shares:.4f} ETH) "
                      f"Conf: {confidence:.2f}")

    # Close remaining trades
    if active_trades:
        final_price = df.iloc[-1]['close']
        for trade in active_trades:
            if trade.direction == "long":
                trade.dollar_pnl = trade.shares * (final_price - trade.entry_price)
            else:
                trade.dollar_pnl = trade.shares * (trade.entry_price - final_price)

            trade.pct_return = trade.dollar_pnl / trade.position_dollars
            trade.exit_time = df.iloc[-1]['timestamp'].strftime('%Y-%m-%d %H:%M')
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
        'starting_balance': starting_balance,
        'final_balance': final_balance,
        'total_pnl': final_balance - starting_balance,
        'total_return': total_return,
        'max_drawdown': max_drawdown,
        'total_trades': len(trades),
        'winning_trades': len(winning_trades),
        'losing_trades': len(losing_trades),
        'win_rate': len(winning_trades) / len(trades) if trades else 0,
        'avg_win': np.mean([t.dollar_pnl for t in winning_trades]) if winning_trades else 0,
        'avg_loss': np.mean([abs(t.dollar_pnl) for t in losing_trades]) if losing_trades else 0,
        'largest_win': max([t.dollar_pnl for t in trades]) if trades else 0,
        'largest_loss': min([t.dollar_pnl for t in trades]) if trades else 0,
        'profit_factor': (sum(t.dollar_pnl for t in winning_trades) /
                         abs(sum(t.dollar_pnl for t in losing_trades))) if losing_trades else 0,
        'trades': trades
    }

    return results

def print_backtest_summary(results: Dict):
    """Print detailed backtest summary."""

    print(f"\n{'='*80}")
    print(f"💰 ETH {results['timeframe']} - FINAL RESULTS")
    print(f"{'='*80}")

    print(f"Starting Balance:      ${results['starting_balance']:,.2f}")
    print(f"Final Balance:         ${results['final_balance']:,.2f}")
    print(f"Total P&L:             ${results['total_pnl']:+,.2f}")
    print(f"Total Return:          {results['total_return']:+.2%}")
    print(f"Max Drawdown:          {results['max_drawdown']:.2%}")

    print(f"\n📊 TRADE STATISTICS:")
    print(f"Total Trades:          {results['total_trades']}")
    print(f"Winning Trades:        {results['winning_trades']}")
    print(f"Losing Trades:         {results['losing_trades']}")
    print(f"Win Rate:              {results['win_rate']:.1%}")
    print(f"Average Win:           ${results['avg_win']:,.2f}")
    print(f"Average Loss:          ${results['avg_loss']:,.2f}")
    print(f"Largest Win:           ${results['largest_win']:,.2f}")
    print(f"Largest Loss:          ${results['largest_loss']:,.2f}")
    print(f"Profit Factor:         {results['profit_factor']:.2f}")

    # Show trade details
    if results['trades']:
        print(f"\n📋 TRADE DETAILS:")
        print(f"{'#':<3} {'Date':<12} {'Dir':<5} {'Entry':<8} {'Exit':<8} {'Days':<4} {'P&L $':<9} {'P&L %':<8} {'Result'}")
        print("-" * 75)

        for i, trade in enumerate(results['trades'][:20], 1):  # Show first 20
            entry_date = trade.entry_time.split()[0] if trade.entry_time else "N/A"
            result_str = "WIN " if trade.win else "LOSS"

            print(f"{trade.id:<3} {entry_date:<12} {trade.direction.upper():<5} "
                  f"${trade.entry_price:<7.0f} ${trade.exit_price:<7.0f} "
                  f"{trade.duration_bars:<4} ${trade.dollar_pnl:+<8.0f} "
                  f"{trade.pct_return:+<7.1%} {result_str}")

        if len(results['trades']) > 20:
            print(f"\n... and {len(results['trades']) - 20} more trades")

def main():
    """Run comprehensive ETH MTF backtest."""

    print("\n" + "="*100)
    print("🏦 BULL MACHINE v1.5.0 - ETH MTF DOLLAR BACKTEST")
    print("💰 Starting Balance: $10,000")
    print("🎯 Multi-Timeframe Technical Analysis with Risk Management")
    print("="*100)

    all_results = []

    for timeframe, file_path in ETH_DATA_FILES.items():
        print(f"\n{'='*60}")
        print(f"  📈 Processing ETH {timeframe}")
        print(f"{'='*60}")

        df = load_eth_data(file_path)

        if df.empty:
            print(f"❌ Could not load data for {timeframe}")
            continue

        if len(df) < 100:
            print(f"❌ Insufficient data for {timeframe}: {len(df)} bars")
            continue

        results = run_eth_backtest(df, timeframe)
        all_results.append(results)

        print_backtest_summary(results)

    # Portfolio summary
    if all_results:
        print(f"\n{'='*100}")
        print(f"🏆 PORTFOLIO SUMMARY - ALL TIMEFRAMES")
        print(f"{'='*100}")

        total_pnl = sum(r['total_pnl'] for r in all_results)
        total_trades = sum(r['total_trades'] for r in all_results)
        total_wins = sum(r['winning_trades'] for r in all_results)
        portfolio_wr = total_wins / total_trades if total_trades > 0 else 0

        # Calculate combined return (if trading all timeframes)
        combined_final = sum(r['final_balance'] for r in all_results)
        combined_starting = sum(r['starting_balance'] for r in all_results)
        combined_return = (combined_final - combined_starting) / combined_starting

        print(f"Combined Starting:     ${combined_starting:,.2f}")
        print(f"Combined Final:        ${combined_final:,.2f}")
        print(f"Combined P&L:          ${total_pnl:+,.2f}")
        print(f"Combined Return:       {combined_return:+.2%}")
        print(f"Total Trades:          {total_trades}")
        print(f"Portfolio Win Rate:    {portfolio_wr:.1%}")

        # Best performing timeframe
        if all_results:
            best_result = max(all_results, key=lambda x: x['total_return'])
            print(f"Best Timeframe:        {best_result['timeframe']} ({best_result['total_return']:+.2%})")

    print(f"\n✅ ETH MTF Backtest Complete!")

if __name__ == "__main__":
    main()