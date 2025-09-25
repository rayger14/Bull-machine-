#!/usr/bin/env python3
"""
Simplified Bull Machine v1.4.2 vs v1.5.0 Standardized Comparison
Direct implementation without complex imports
"""

import pandas as pd
import numpy as np
from datetime import datetime
import json
import csv


class ETHComparisonBacktest:
    """Simplified backtest for direct comparison."""

    def __init__(self):
        self.initial_capital = 10000.0
        self.fee_rate = 0.001  # 0.1%
        self.slippage_rate = 0.0005  # 0.05%

    def create_mock_eth_data(self, days=600):
        """Create realistic ETH-like data for testing."""
        np.random.seed(42)  # Reproducible results

        dates = pd.date_range(start='2023-01-01', periods=days, freq='D')

        # Start around $1500 ETH price
        base_price = 1500
        prices = [base_price]

        # Generate realistic price movements
        for i in range(1, days):
            # Random walk with some trend and volatility clustering
            daily_return = np.random.normal(0.0005, 0.035)  # ~0.05% daily return, 3.5% volatility

            # Add some momentum and mean reversion
            if i > 10:
                momentum = np.mean([prices[j] - prices[j-1] for j in range(i-5, i)]) * 0.1
                daily_return += momentum / prices[i-1]

            new_price = prices[i-1] * (1 + daily_return)
            prices.append(max(new_price, 100))  # Floor at $100

        # Create OHLC from close prices
        opens = [prices[0]] + prices[:-1]  # Open = prev close

        highs = []
        lows = []
        volumes = []

        for i, (open_price, close_price) in enumerate(zip(opens, prices)):
            daily_volatility = np.random.uniform(0.01, 0.04)  # 1-4% intraday range

            high = max(open_price, close_price) * (1 + daily_volatility/2)
            low = min(open_price, close_price) * (1 - daily_volatility/2)

            highs.append(high)
            lows.append(low)

            # Volume correlated with price movement
            price_change = abs(close_price - open_price) / open_price
            base_volume = 50000 + np.random.exponential(100000)
            volume = base_volume * (1 + price_change * 5)
            volumes.append(volume)

        df = pd.DataFrame({
            'date': dates,
            'open': opens,
            'high': highs,
            'low': lows,
            'close': prices,
            'volume': volumes
        })

        return df

    def calculate_indicators(self, df):
        """Calculate technical indicators."""
        df = df.copy()

        # SMAs
        df['sma_20'] = df['close'].rolling(20).mean()
        df['sma_50'] = df['close'].rolling(50).mean()

        # RSI
        delta = df['close'].diff()
        gain = delta.where(delta > 0, 0).rolling(14).mean()
        loss = (-delta).where(delta < 0, 0).rolling(14).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))

        # Bollinger Bands
        df['bb_middle'] = df['close'].rolling(20).mean()
        bb_std = df['close'].rolling(20).std()
        df['bb_upper'] = df['bb_middle'] + (bb_std * 2)
        df['bb_lower'] = df['bb_middle'] - (bb_std * 2)

        # Volume
        df['volume_sma'] = df['volume'].rolling(20).mean()
        df['volume_ratio'] = df['volume'] / df['volume_sma']

        return df

    def generate_v142_signals(self, df):
        """v1.4.2 baseline signals - stricter, fewer trades."""
        signals = []
        entry_threshold = 0.45  # Original threshold

        for i in range(len(df)):
            if i < 50:
                signals.append(False)
                continue

            score = 0

            # Trend alignment (30% weight)
            if df.iloc[i]['close'] > df.iloc[i]['sma_20']:
                score += 0.15
            if df.iloc[i]['sma_20'] > df.iloc[i]['sma_50']:
                score += 0.15

            # RSI not extreme (25% weight)
            rsi = df.iloc[i]['rsi']
            if 30 <= rsi <= 70:
                score += 0.25

            # Bollinger position (20% weight)
            close = df.iloc[i]['close']
            if df.iloc[i]['bb_lower'] <= close <= df.iloc[i]['bb_upper']:
                score += 0.20

            # Volume confirmation (10% weight)
            if df.iloc[i]['volume_ratio'] > 1.2:
                score += 0.10

            # Structure (15% weight) - more conservative
            if i >= 5:
                recent_high = df.iloc[i-4:i+1]['high'].max()
                if df.iloc[i]['high'] >= recent_high * 0.995:  # Near recent high
                    score += 0.15

            signals.append(score >= entry_threshold)

        return pd.Series(signals)

    def generate_v150_signals(self, df):
        """v1.5.0 enhanced signals with frequency controls."""
        signals = []
        entry_threshold = 0.46  # Balanced for 2-4 trades/month
        cooldown_bars = 7
        last_signal_idx = -cooldown_bars

        for i in range(len(df)):
            if i < 50:
                signals.append(False)
                continue

            # Start with v1.4.2 base score
            score = 0

            # Enhanced base scoring for higher quality
            # Multi-timeframe trend (stronger requirement)
            if df.iloc[i]['close'] > df.iloc[i]['sma_20'] and df.iloc[i]['sma_20'] > df.iloc[i]['sma_50']:
                score += 0.25  # Both conditions must be met
            elif df.iloc[i]['close'] > df.iloc[i]['sma_20']:
                score += 0.10  # Partial credit

            # RSI in sweet spot (improved range)
            rsi = df.iloc[i]['rsi']
            if 45 <= rsi <= 65:  # Optimal entry zone
                score += 0.25
            elif 35 <= rsi <= 75:  # Acceptable range
                score += 0.15

            # Bollinger position (middle bias)
            bb_pos = (df.iloc[i]['close'] - df.iloc[i]['bb_lower']) / (df.iloc[i]['bb_upper'] - df.iloc[i]['bb_lower'])
            if 0.4 <= bb_pos <= 0.7:  # Sweet spot
                score += 0.20
            elif 0.2 <= bb_pos <= 0.8:
                score += 0.10

            # Volume confirmation (higher requirement)
            if df.iloc[i]['volume_ratio'] > 1.5:  # Stronger volume
                score += 0.15
            elif df.iloc[i]['volume_ratio'] > 1.2:
                score += 0.08

            # Price structure (more selective)
            if i >= 5:
                recent_closes = df.iloc[i-4:i+1]['close']
                if recent_closes.iloc[-1] == recent_closes.max() and recent_closes.iloc[-1] > recent_closes.iloc[0]:
                    score += 0.15  # New high with uptrend

            # v1.5.0 enhancements (reduced impact)
            if i >= 20:
                # MTF DL2 filter - penalize extreme moves
                recent_prices = df.iloc[i-19:i+1]['close']
                price_std = recent_prices.std()
                if price_std > 0:
                    z_score = abs((df.iloc[i]['close'] - recent_prices.mean()) / price_std)
                    if z_score > 2.5:  # Extreme condition
                        score *= 0.9  # 10% penalty

                # 6-candle structure - reward alternating patterns
                if i >= 6:
                    closes = df.iloc[i-5:i+1]['close'].values
                    directions = [1 if closes[j] > closes[j-1] else -1 for j in range(1, len(closes))]
                    changes = sum(1 for j in range(1, len(directions)) if directions[j] != directions[j-1])
                    if changes >= 2:
                        score += 0.03  # Small bonus for good structure

                # Orderflow LCA - breakout confirmation
                if i >= 10:
                    recent_highs = df.iloc[i-9:i]['high']
                    resistance = recent_highs.quantile(0.75)
                    if df.iloc[i]['high'] > resistance and df.iloc[i]['volume_ratio'] > 1.5:
                        score += 0.04  # Breakout bonus

                # Negative VIP - reduce overconfidence on extreme moves
                if i >= 5:
                    price_change = (df.iloc[i]['close'] - df.iloc[i-4]['close']) / df.iloc[i-4]['close']
                    if abs(price_change) > 0.15 and df.iloc[i]['volume_ratio'] > 2.0:
                        score *= 0.95  # Slight penalty for potential reversal setup

            # Check signal with threshold
            signal = score >= entry_threshold

            # Apply cooldown
            if signal and (i - last_signal_idx) < cooldown_bars:
                signal = False
            elif signal:
                last_signal_idx = i

            signals.append(signal)

        return pd.Series(signals)

    def backtest(self, df, signals, version):
        """Run backtest with standardized parameters."""
        capital = self.initial_capital
        position = 0
        trades = []
        equity_curve = [self.initial_capital]

        trade_amount = 1000  # Fixed $1000 per trade

        in_position = False
        entry_price = 0
        entry_date = None

        for i in range(len(df)):
            current_price = df.iloc[i]['close']
            current_date = df.iloc[i]['date']
            signal = signals.iloc[i]

            # Calculate portfolio value
            portfolio_value = capital + (position * current_price)
            equity_curve.append(portfolio_value)

            if signal and not in_position and capital >= trade_amount:
                # Enter position
                effective_price = current_price * (1 + self.slippage_rate)
                fees = trade_amount * self.fee_rate
                net_amount = trade_amount - fees

                position = net_amount / effective_price
                capital -= trade_amount

                in_position = True
                entry_price = effective_price
                entry_date = current_date

            elif in_position and (not signal or i == len(df) - 1):  # Exit on signal end or last day
                # Exit position
                effective_price = current_price * (1 - self.slippage_rate)
                gross_proceeds = position * effective_price
                fees = gross_proceeds * self.fee_rate
                net_proceeds = gross_proceeds - fees

                capital += net_proceeds
                pnl = net_proceeds - trade_amount
                return_pct = (pnl / trade_amount) * 100

                trades.append({
                    'version': version,
                    'entry_date': entry_date,
                    'exit_date': current_date,
                    'entry_price': entry_price,
                    'exit_price': effective_price,
                    'position_size': position,
                    'trade_amount': trade_amount,
                    'net_proceeds': net_proceeds,
                    'pnl': pnl,
                    'return_pct': return_pct
                })

                position = 0
                in_position = False

        # Calculate metrics
        final_value = capital + (position * df.iloc[-1]['close'])
        total_return = ((final_value - self.initial_capital) / self.initial_capital) * 100

        winning_trades = [t for t in trades if t['pnl'] > 0]
        win_rate = len(winning_trades) / len(trades) * 100 if trades else 0

        # Max drawdown
        peak = max(equity_curve)
        max_dd = 0
        for value in equity_curve:
            if value > peak:
                peak = value
            dd = ((peak - value) / peak) * 100
            max_dd = max(max_dd, dd)

        # Trades per month
        days_in_backtest = (df.iloc[-1]['date'] - df.iloc[0]['date']).days
        months = days_in_backtest / 30.44
        trades_per_month = len(trades) / months if months > 0 else 0

        return {
            'version': version,
            'final_value': final_value,
            'total_return_pct': total_return,
            'total_trades': len(trades),
            'win_rate': win_rate,
            'trades_per_month': trades_per_month,
            'max_drawdown': max_dd,
            'avg_return_per_trade': np.mean([t['return_pct'] for t in trades]) if trades else 0,
            'trades': trades
        }

    def run_comparison(self):
        """Run the full comparison."""
        print("Creating ETH test data...")
        df = self.create_mock_eth_data(600)  # ~20 months

        print("Calculating technical indicators...")
        df = self.calculate_indicators(df)

        print("Generating v1.4.2 signals...")
        signals_v142 = self.generate_v142_signals(df)

        print("Generating v1.5.0 signals...")
        signals_v150 = self.generate_v150_signals(df)

        print("Running v1.4.2 backtest...")
        results_v142 = self.backtest(df, signals_v142, 'v1.4.2')

        print("Running v1.5.0 backtest...")
        results_v150 = self.backtest(df, signals_v150, 'v1.5.0')

        return {
            'test_period': {
                'start_date': str(df['date'].min().date()),
                'end_date': str(df['date'].max().date()),
                'total_days': len(df)
            },
            'v142': results_v142,
            'v150': results_v150
        }


def main():
    backtest = ETHComparisonBacktest()
    results = backtest.run_comparison()

    print("\n" + "="*70)
    print("BULL MACHINE v1.4.2 vs v1.5.0 STANDARDIZED COMPARISON")
    print("="*70)

    print(f"Test Period: {results['test_period']['start_date']} to {results['test_period']['end_date']}")
    print(f"Total Days: {results['test_period']['total_days']}")
    print(f"Parameters: $10,000 initial capital, 0.1% fees, 0.05% slippage, $1,000 per trade")

    v142 = results['v142']
    v150 = results['v150']

    print(f"\n{'Metric':<25} {'v1.4.2':<15} {'v1.5.0':<15} {'Difference':<15}")
    print("-" * 70)
    print(f"{'Total Return':<25} {v142['total_return_pct']:>8.2f}%    {v150['total_return_pct']:>8.2f}%    {v150['total_return_pct']-v142['total_return_pct']:>+8.2f}%")
    print(f"{'Final Value':<25} ${v142['final_value']:>8,.0f}    ${v150['final_value']:>8,.0f}    ${v150['final_value']-v142['final_value']:>+8,.0f}")
    print(f"{'Total Trades':<25} {v142['total_trades']:>11}    {v150['total_trades']:>11}    {v150['total_trades']-v142['total_trades']:>+11}")
    print(f"{'Win Rate':<25} {v142['win_rate']:>8.1f}%    {v150['win_rate']:>8.1f}%    {v150['win_rate']-v142['win_rate']:>+8.1f}%")
    print(f"{'Trades/Month':<25} {v142['trades_per_month']:>11.1f}    {v150['trades_per_month']:>11.1f}    {v150['trades_per_month']-v142['trades_per_month']:>+11.1f}")
    print(f"{'Max Drawdown':<25} {v142['max_drawdown']:>8.2f}%    {v150['max_drawdown']:>8.2f}%    {v150['max_drawdown']-v142['max_drawdown']:>+8.2f}%")
    print(f"{'Avg Return/Trade':<25} {v142['avg_return_per_trade']:>8.2f}%    {v150['avg_return_per_trade']:>8.2f}%    {v150['avg_return_per_trade']-v142['avg_return_per_trade']:>+8.2f}%")

    # Target assessment
    print(f"\nETH-1D v1.5.0 Target Assessment:")
    print(f"  Trades/Month: {v150['trades_per_month']:.1f} (Target: 2-4) {'✓' if 2 <= v150['trades_per_month'] <= 4 else '✗'}")
    print(f"  Win Rate: {v150['win_rate']:.1f}% (Target: ≥50%) {'✓' if v150['win_rate'] >= 50 else '✗'}")
    print(f"  Max Drawdown: {v150['max_drawdown']:.1f}% (Target: ≤20%) {'✓' if v150['max_drawdown'] <= 20 else '✗'}")

    # Save CSV comparison
    with open('eth_comparison_results.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Metric', 'v1.4.2', 'v1.5.0', 'Difference', 'Unit'])
        writer.writerow(['Total Return', f"{v142['total_return_pct']:.2f}", f"{v150['total_return_pct']:.2f}", f"{v150['total_return_pct']-v142['total_return_pct']:+.2f}", '%'])
        writer.writerow(['Final Value', f"{v142['final_value']:.2f}", f"{v150['final_value']:.2f}", f"{v150['final_value']-v142['final_value']:+.2f}", '$'])
        writer.writerow(['Total Trades', v142['total_trades'], v150['total_trades'], v150['total_trades']-v142['total_trades'], 'count'])
        writer.writerow(['Win Rate', f"{v142['win_rate']:.1f}", f"{v150['win_rate']:.1f}", f"{v150['win_rate']-v142['win_rate']:+.1f}", '%'])
        writer.writerow(['Trades/Month', f"{v142['trades_per_month']:.1f}", f"{v150['trades_per_month']:.1f}", f"{v150['trades_per_month']-v142['trades_per_month']:+.1f}", 'count'])
        writer.writerow(['Max Drawdown', f"{v142['max_drawdown']:.2f}", f"{v150['max_drawdown']:.2f}", f"{v150['max_drawdown']-v142['max_drawdown']:+.2f}", '%'])

    # Save detailed trades
    all_trades = v142['trades'] + v150['trades']
    if all_trades:
        with open('eth_detailed_trades.csv', 'w', newline='') as f:
            fieldnames = ['version', 'entry_date', 'exit_date', 'entry_price', 'exit_price', 'pnl', 'return_pct']
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for trade in all_trades:
                writer.writerow({k: trade[k] for k in fieldnames})

    print(f"\n✓ Results saved to eth_comparison_results.csv")
    print(f"✓ Detailed trades saved to eth_detailed_trades.csv")

    # Summary verdict
    target_met = (2 <= v150['trades_per_month'] <= 4 and
                  v150['win_rate'] >= 50 and
                  v150['max_drawdown'] <= 20)

    print(f"\n{'🎯 ETH-1D TARGETS: ' + ('ACHIEVED' if target_met else 'NOT MET'):>50}")


if __name__ == "__main__":
    main()