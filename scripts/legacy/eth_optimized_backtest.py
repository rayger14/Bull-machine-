#!/usr/bin/env python3
"""
ETH v1.5.0 Optimized Backtest with Enhanced Signal Quality
Focus on higher win rate through better signal filtering
"""

import pandas as pd
import numpy as np
import csv
from datetime import datetime


class ETHOptimizedBacktest:
    """Optimized ETH backtest focusing on signal quality."""

    def __init__(self):
        self.initial_capital = 10000.0
        self.fee_rate = 0.001
        self.slippage_rate = 0.0005

    def create_eth_data(self, days=600):
        """Create realistic ETH test data."""
        np.random.seed(42)
        dates = pd.date_range(start='2023-01-01', periods=days, freq='D')

        # Enhanced price generation for more realistic patterns
        base_price = 1500
        prices = [base_price]

        for i in range(1, days):
            # Base random walk
            daily_return = np.random.normal(0.0008, 0.035)  # Slight positive bias

            # Add market cycles (bull/bear phases)
            cycle_position = i / days
            if 0.1 < cycle_position < 0.4:  # Bull phase
                daily_return += 0.002
            elif 0.6 < cycle_position < 0.8:  # Bear phase
                daily_return -= 0.001

            # Momentum and mean reversion
            if i > 20:
                recent_trend = (prices[i-1] - prices[i-20]) / prices[i-20]
                daily_return += recent_trend * 0.1  # Momentum

                # Mean reversion to long-term average
                long_term_avg = np.mean(prices[max(0, i-60):i])
                if prices[i-1] > long_term_avg * 1.2:  # Overextended
                    daily_return -= 0.005
                elif prices[i-1] < long_term_avg * 0.8:  # Oversold
                    daily_return += 0.005

            new_price = prices[i-1] * (1 + daily_return)
            prices.append(max(new_price, 100))

        # Generate OHLC
        opens = [prices[0]] + prices[:-1]
        highs = []
        lows = []
        volumes = []

        for i, (open_price, close_price) in enumerate(zip(opens, prices)):
            daily_vol = np.random.uniform(0.015, 0.045)
            high = max(open_price, close_price) * (1 + daily_vol/2)
            low = min(open_price, close_price) * (1 - daily_vol/2)

            highs.append(high)
            lows.append(low)

            # Volume with realistic patterns
            price_change = abs(close_price - open_price) / open_price
            base_volume = 60000 + np.random.exponential(80000)
            volume = base_volume * (1 + price_change * 3)

            # Higher volume on breakouts and breakdowns
            if i >= 20:
                sma_20 = np.mean(prices[i-19:i+1])
                if close_price > sma_20 * 1.05 or close_price < sma_20 * 0.95:
                    volume *= 1.5

            volumes.append(volume)

        return pd.DataFrame({
            'date': dates,
            'open': opens,
            'high': highs,
            'low': lows,
            'close': prices,
            'volume': volumes
        })

    def calculate_indicators(self, df):
        """Calculate comprehensive technical indicators."""
        df = df.copy()

        # Multi-timeframe moving averages
        for period in [10, 20, 50, 100]:
            df[f'sma_{period}'] = df['close'].rolling(period).mean()
            df[f'ema_{period}'] = df['close'].ewm(span=period).mean()

        # RSI with multiple periods
        for period in [14, 21]:
            delta = df['close'].diff()
            gain = delta.where(delta > 0, 0).rolling(period).mean()
            loss = (-delta).where(delta < 0, 0).rolling(period).mean()
            rs = gain / loss
            df[f'rsi_{period}'] = 100 - (100 / (1 + rs))

        # Bollinger Bands
        df['bb_middle'] = df['close'].rolling(20).mean()
        bb_std = df['close'].rolling(20).std()
        df['bb_upper'] = df['bb_middle'] + (bb_std * 2)
        df['bb_lower'] = df['bb_middle'] - (bb_std * 2)
        df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])

        # MACD
        ema12 = df['close'].ewm(span=12).mean()
        ema26 = df['close'].ewm(span=26).mean()
        df['macd'] = ema12 - ema26
        df['macd_signal'] = df['macd'].ewm(span=9).mean()
        df['macd_histogram'] = df['macd'] - df['macd_signal']

        # Volume indicators
        df['volume_sma_20'] = df['volume'].rolling(20).mean()
        df['volume_ratio'] = df['volume'] / df['volume_sma_20']
        df['volume_ema'] = df['volume'].ewm(span=10).mean()

        # Price momentum
        for period in [3, 7, 14]:
            df[f'momentum_{period}'] = (df['close'] - df['close'].shift(period)) / df['close'].shift(period) * 100

        # Volatility
        df['atr'] = df[['high', 'low', 'close']].apply(
            lambda x: max(x['high'] - x['low'],
                         abs(x['high'] - x.name > 0 and df.iloc[x.name-1]['close'] or x['close']),
                         abs(x['low'] - x.name > 0 and df.iloc[x.name-1]['close'] or x['close'])),
            axis=1
        ).rolling(14).mean()

        return df

    def calculate_signal_quality_v150(self, df):
        """Enhanced v1.5.0 signal calculation with focus on quality."""
        signals = []
        quality_scores = []

        # Stricter parameters for higher quality
        entry_threshold = 0.52  # Even higher threshold
        cooldown_bars = 10      # Longer cooldown
        last_signal_idx = -cooldown_bars

        for i in range(len(df)):
            if i < 100:  # Need more data for quality indicators
                signals.append(False)
                quality_scores.append(0)
                continue

            score = 0
            quality_factors = 0

            # 1. Multi-timeframe trend alignment (25% weight)
            trend_score = 0
            if df.iloc[i]['close'] > df.iloc[i]['sma_20']:
                trend_score += 0.33
            if df.iloc[i]['sma_20'] > df.iloc[i]['sma_50']:
                trend_score += 0.33
            if df.iloc[i]['sma_50'] > df.iloc[i]['sma_100']:
                trend_score += 0.34

            score += trend_score * 0.25
            if trend_score > 0.8:
                quality_factors += 1

            # 2. RSI confluence (15% weight)
            rsi_14 = df.iloc[i]['rsi_14']
            rsi_21 = df.iloc[i]['rsi_21']
            rsi_score = 0

            # Optimal RSI range for long entries
            if 40 <= rsi_14 <= 65 and 40 <= rsi_21 <= 65:
                rsi_score = 1.0
                quality_factors += 1
            elif 30 <= rsi_14 <= 75 and 30 <= rsi_21 <= 75:
                rsi_score = 0.6

            score += rsi_score * 0.15

            # 3. MACD momentum (15% weight)
            macd_score = 0
            if (df.iloc[i]['macd'] > df.iloc[i]['macd_signal'] and
                df.iloc[i]['macd_histogram'] > 0):
                macd_score = 1.0
                quality_factors += 1
            elif df.iloc[i]['macd'] > df.iloc[i]['macd_signal']:
                macd_score = 0.6

            score += macd_score * 0.15

            # 4. Volume confirmation (20% weight)
            volume_score = 0
            vol_ratio = df.iloc[i]['volume_ratio']

            if vol_ratio > 1.5:  # Strong volume
                volume_score = 1.0
                quality_factors += 1
            elif vol_ratio > 1.2:
                volume_score = 0.7
            elif vol_ratio > 1.0:
                volume_score = 0.4

            score += volume_score * 0.20

            # 5. Bollinger Band position (10% weight)
            bb_pos = df.iloc[i]['bb_position']
            bb_score = 0

            # Sweet spot: not oversold, not overbought
            if 0.3 <= bb_pos <= 0.7:
                bb_score = 1.0
                quality_factors += 1
            elif 0.2 <= bb_pos <= 0.8:
                bb_score = 0.6

            score += bb_score * 0.10

            # 6. Price momentum quality (15% weight)
            momentum_score = 0
            mom_3 = df.iloc[i]['momentum_3']
            mom_7 = df.iloc[i]['momentum_7']

            # Positive but not excessive momentum
            if 0.5 <= mom_3 <= 8 and 1 <= mom_7 <= 15:
                momentum_score = 1.0
                quality_factors += 1
            elif mom_3 > 0 and mom_7 > 0:
                momentum_score = 0.6

            score += momentum_score * 0.15

            # v1.5.0 Alpha enhancements (reduced impact for stability)
            if i >= 50:
                # MTF DL2 - penalize extreme volatility
                recent_closes = df.iloc[i-19:i+1]['close']
                volatility = recent_closes.std() / recent_closes.mean()
                if volatility > 0.08:  # High volatility
                    score *= 0.92

                # 6-candle structure - reward consistent patterns
                if i >= 10:
                    closes = df.iloc[i-9:i+1]['close'].values
                    trends = [1 if closes[j] > closes[j-1] else -1 for j in range(1, len(closes))]
                    trend_consistency = len([t for t in trends if t == trends[-1]]) / len(trends)

                    if trend_consistency >= 0.6:  # Good trend consistency
                        score += 0.02
                        if trend_consistency >= 0.8:
                            quality_factors += 1

                # Orderflow LCA - breakout quality
                if i >= 20:
                    recent_highs = df.iloc[i-19:i]['high']
                    resistance = recent_highs.quantile(0.8)

                    if (df.iloc[i]['high'] > resistance and
                        df.iloc[i]['volume_ratio'] > 1.8 and
                        df.iloc[i]['close'] > df.iloc[i]['open']):  # Bullish breakout
                        score += 0.04
                        quality_factors += 1

                # Negative VIP - avoid potential reversals
                if i >= 7:
                    price_change_5d = (df.iloc[i]['close'] - df.iloc[i-5]['close']) / df.iloc[i-5]['close']
                    volume_surge = df.iloc[i]['volume_ratio'] > 2.5

                    if abs(price_change_5d) > 0.20 and volume_surge:  # Potential exhaustion
                        score *= 0.90

            # Quality gate: require minimum number of quality factors
            quality_requirement = 3  # Need at least 3 quality factors
            if quality_factors < quality_requirement:
                score *= 0.7  # Significant penalty

            # Final signal determination
            signal = score >= entry_threshold

            # Apply cooldown
            if signal and (i - last_signal_idx) < cooldown_bars:
                signal = False
            elif signal:
                last_signal_idx = i

            signals.append(signal)
            quality_scores.append(score)

        return pd.Series(signals), pd.Series(quality_scores)

    def backtest(self, df, signals, version):
        """Run optimized backtest."""
        capital = self.initial_capital
        position = 0
        trades = []
        equity_curve = [self.initial_capital]

        # Dynamic position sizing based on volatility
        base_trade_amount = 1000

        in_position = False
        entry_price = 0
        entry_date = None
        entry_atr = 0

        for i in range(len(df)):
            current_price = df.iloc[i]['close']
            current_date = df.iloc[i]['date']
            signal = signals.iloc[i]
            current_atr = df.iloc[i]['atr'] if 'atr' in df.columns else current_price * 0.02

            # Portfolio value
            portfolio_value = capital + (position * current_price)
            equity_curve.append(portfolio_value)

            # Dynamic trade sizing based on volatility
            volatility_factor = min(current_atr / (current_price * 0.02), 2.0)
            trade_amount = base_trade_amount / volatility_factor  # Smaller size in high volatility

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
                entry_atr = current_atr

            elif in_position:
                # Dynamic exit conditions
                exit_signal = False

                # Technical exit: signal ends
                if not signal:
                    exit_signal = True

                # Risk management exits
                price_change = (current_price - entry_price) / entry_price

                # Stop loss: 2x ATR
                if price_change < -(2 * entry_atr / entry_price):
                    exit_signal = True

                # Take profit: 4x ATR or 15% gain
                if price_change > max(4 * entry_atr / entry_price, 0.15):
                    exit_signal = True

                # Time-based exit: max 20 days
                days_held = (current_date - entry_date).days
                if days_held > 20:
                    exit_signal = True

                # Final day exit
                if i == len(df) - 1:
                    exit_signal = True

                if exit_signal:
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
                        'days_held': days_held,
                        'trade_amount': trade_amount,
                        'pnl': pnl,
                        'return_pct': return_pct
                    })

                    position = 0
                    in_position = False

        # Calculate final metrics
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

        # Profit factor
        gross_wins = sum(t['pnl'] for t in trades if t['pnl'] > 0)
        gross_losses = abs(sum(t['pnl'] for t in trades if t['pnl'] <= 0))
        profit_factor = gross_wins / gross_losses if gross_losses > 0 else float('inf')

        return {
            'version': version,
            'final_value': final_value,
            'total_return_pct': total_return,
            'total_trades': len(trades),
            'win_rate': win_rate,
            'trades_per_month': trades_per_month,
            'max_drawdown': max_dd,
            'avg_return_per_trade': np.mean([t['return_pct'] for t in trades]) if trades else 0,
            'profit_factor': profit_factor,
            'avg_days_held': np.mean([t['days_held'] for t in trades]) if trades else 0,
            'trades': trades
        }

    def run_optimized_test(self):
        """Run the optimized ETH backtest."""
        print("Creating optimized ETH test data...")
        df = self.create_eth_data(600)

        print("Calculating comprehensive indicators...")
        df = self.calculate_indicators(df)

        print("Generating optimized v1.5.0 signals...")
        signals_v150, quality_scores = self.calculate_signal_quality_v150(df)

        print("Running optimized v1.5.0 backtest...")
        results = self.backtest(df, signals_v150, 'v1.5.0-Optimized')

        return {
            'test_period': {
                'start_date': str(df['date'].min().date()),
                'end_date': str(df['date'].max().date()),
                'total_days': len(df)
            },
            'results': results,
            'signal_data': {
                'signals': signals_v150,
                'quality_scores': quality_scores,
                'avg_quality': quality_scores.mean()
            }
        }


def main():
    backtest = ETHOptimizedBacktest()
    results = backtest.run_optimized_test()

    print("\n" + "="*70)
    print("BULL MACHINE v1.5.0 OPTIMIZED ETH BACKTEST")
    print("="*70)

    print(f"Test Period: {results['test_period']['start_date']} to {results['test_period']['end_date']}")
    print(f"Total Days: {results['test_period']['total_days']}")

    r = results['results']

    print(f"\n{'OPTIMIZED RESULTS':<30} {'Value':<15}")
    print("-" * 45)
    print(f"{'Total Return':<30} {r['total_return_pct']:>8.2f}%")
    print(f"{'Final Value':<30} ${r['final_value']:>8,.0f}")
    print(f"{'Total Trades':<30} {r['total_trades']:>11}")
    print(f"{'Win Rate':<30} {r['win_rate']:>8.1f}%")
    print(f"{'Trades/Month':<30} {r['trades_per_month']:>11.1f}")
    print(f"{'Max Drawdown':<30} {r['max_drawdown']:>8.2f}%")
    print(f"{'Avg Return/Trade':<30} {r['avg_return_per_trade']:>8.2f}%")
    print(f"{'Profit Factor':<30} {r['profit_factor']:>11.2f}")
    print(f"{'Avg Days Held':<30} {r['avg_days_held']:>11.1f}")

    # Target assessment
    print(f"\n🎯 ETH-1D TARGET ASSESSMENT:")
    trade_freq_ok = 2 <= r['trades_per_month'] <= 4
    win_rate_ok = r['win_rate'] >= 50
    drawdown_ok = r['max_drawdown'] <= 20

    print(f"  Trades/Month: {r['trades_per_month']:.1f} (Target: 2-4) {'✓' if trade_freq_ok else '✗'}")
    print(f"  Win Rate: {r['win_rate']:.1f}% (Target: ≥50%) {'✓' if win_rate_ok else '✗'}")
    print(f"  Max Drawdown: {r['max_drawdown']:.1f}% (Target: ≤20%) {'✓' if drawdown_ok else '✗'}")

    all_targets_met = trade_freq_ok and win_rate_ok and drawdown_ok
    print(f"\n{'🎯 ALL TARGETS: ' + ('ACHIEVED ✓' if all_targets_met else 'NOT MET ✗'):>50}")

    # Signal quality metrics
    sig_data = results['signal_data']
    print(f"\nSignal Quality:")
    print(f"  Average Quality Score: {sig_data['avg_quality']:.3f}")
    print(f"  Signals Generated: {sig_data['signals'].sum()}")

    # Save results
    with open('eth_optimized_results.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Metric', 'Value', 'Unit', 'Target', 'Status'])
        writer.writerow(['Total Return', f"{r['total_return_pct']:.2f}", '%', 'Positive', '✓' if r['total_return_pct'] > 0 else '✗'])
        writer.writerow(['Trades/Month', f"{r['trades_per_month']:.1f}", 'count', '2-4', '✓' if trade_freq_ok else '✗'])
        writer.writerow(['Win Rate', f"{r['win_rate']:.1f}", '%', '≥50%', '✓' if win_rate_ok else '✗'])
        writer.writerow(['Max Drawdown', f"{r['max_drawdown']:.2f}", '%', '≤20%', '✓' if drawdown_ok else '✗'])
        writer.writerow(['Profit Factor', f"{r['profit_factor']:.2f}", 'ratio', '>1.5', '✓' if r['profit_factor'] > 1.5 else '✗'])

    print(f"\n✓ Results saved to eth_optimized_results.csv")

    if all_targets_met:
        print("\n🚀 v1.5.0 READY FOR RC PROMOTION!")
    else:
        print("\n⚠️  Further optimization needed before RC promotion.")


if __name__ == "__main__":
    main()