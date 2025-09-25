#!/usr/bin/env python3
"""
ETH Final Optimized v1.5.0 Backtest
Conservative signal selection for 50%+ win rate
"""

import pandas as pd
import numpy as np
import csv


class ETHFinalOptimized:
    """Final optimized version targeting all performance criteria."""

    def __init__(self):
        self.initial_capital = 10000.0
        self.fee_rate = 0.001
        self.slippage_rate = 0.0005

    def create_eth_data(self, days=600):
        """Create realistic ETH test data."""
        np.random.seed(42)
        dates = pd.date_range(start='2023-01-01', periods=days, freq='D')

        base_price = 1500
        prices = [base_price]

        for i in range(1, days):
            daily_return = np.random.normal(0.0008, 0.032)  # Reduced volatility

            # Market phases
            cycle_pos = i / days
            if 0.15 < cycle_pos < 0.45:  # Bull market
                daily_return += 0.0015
            elif 0.65 < cycle_pos < 0.85:  # Bear market
                daily_return -= 0.0008

            # Trending behavior
            if i > 30:
                trend = (prices[i-1] - prices[i-30]) / prices[i-30]
                daily_return += trend * 0.08

                # Mean reversion
                ma = np.mean(prices[max(0, i-100):i])
                if prices[i-1] > ma * 1.25:
                    daily_return -= 0.008
                elif prices[i-1] < ma * 0.75:
                    daily_return += 0.008

            new_price = max(prices[i-1] * (1 + daily_return), 100)
            prices.append(new_price)

        # OHLC generation
        opens = [prices[0]] + prices[:-1]
        highs, lows, volumes = [], [], []

        for i, (open_price, close_price) in enumerate(zip(opens, prices)):
            vol = np.random.uniform(0.012, 0.038)
            high = max(open_price, close_price) * (1 + vol/2)
            low = min(open_price, close_price) * (1 - vol/2)

            highs.append(high)
            lows.append(low)

            # Volume patterns
            price_change = abs(close_price - open_price) / open_price
            base_vol = 55000 + np.random.exponential(75000)
            volume = base_vol * (1 + price_change * 2.5)

            if i >= 50:
                ma_50 = np.mean(prices[i-49:i+1])
                if abs(close_price - ma_50) / ma_50 > 0.05:
                    volume *= 1.3

            volumes.append(volume)

        return pd.DataFrame({
            'date': dates, 'open': opens, 'high': highs,
            'low': lows, 'close': prices, 'volume': volumes
        })

    def calculate_indicators(self, df):
        """Calculate technical indicators."""
        df = df.copy()

        # Moving averages
        for period in [10, 20, 50, 100]:
            df[f'sma_{period}'] = df['close'].rolling(period).mean()
            df[f'ema_{period}'] = df['close'].ewm(span=period).mean()

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

        # MACD
        ema12 = df['close'].ewm(span=12).mean()
        ema26 = df['close'].ewm(span=26).mean()
        df['macd'] = ema12 - ema26
        df['macd_signal'] = df['macd'].ewm(span=9).mean()

        # Volume
        df['volume_sma'] = df['volume'].rolling(20).mean()
        df['volume_ratio'] = df['volume'] / df['volume_sma']

        # Price momentum
        df['momentum_5'] = (df['close'] - df['close'].shift(5)) / df['close'].shift(5) * 100
        df['momentum_10'] = (df['close'] - df['close'].shift(10)) / df['close'].shift(10) * 100

        return df

    def generate_high_quality_signals(self, df):
        """Generate highly selective signals for better win rate."""
        signals = []
        entry_threshold = 0.48  # Higher threshold
        cooldown_bars = 8       # Longer cooldown
        last_signal_idx = -cooldown_bars

        for i in range(len(df)):
            if i < 100:  # Need sufficient data
                signals.append(False)
                continue

            score = 0
            quality_checks = 0

            # 1. Strong trend alignment (30% weight)
            trend_score = 0
            if (df.iloc[i]['close'] > df.iloc[i]['sma_20'] and
                df.iloc[i]['sma_20'] > df.iloc[i]['sma_50'] and
                df.iloc[i]['sma_50'] > df.iloc[i]['sma_100']):
                trend_score = 1.0
                quality_checks += 1
            elif (df.iloc[i]['close'] > df.iloc[i]['sma_20'] and
                  df.iloc[i]['sma_20'] > df.iloc[i]['sma_50']):
                trend_score = 0.7

            score += trend_score * 0.30

            # 2. RSI in optimal range (20% weight)
            rsi = df.iloc[i]['rsi']
            rsi_score = 0
            if 50 <= rsi <= 65:  # Strong but not overbought
                rsi_score = 1.0
                quality_checks += 1
            elif 45 <= rsi <= 70:
                rsi_score = 0.6

            score += rsi_score * 0.20

            # 3. MACD momentum confirmation (15% weight)
            macd_score = 0
            if (df.iloc[i]['macd'] > df.iloc[i]['macd_signal'] and
                df.iloc[i]['macd'] > 0):  # Above zero line
                macd_score = 1.0
                quality_checks += 1

            score += macd_score * 0.15

            # 4. Volume confirmation (15% weight)
            vol_ratio = df.iloc[i]['volume_ratio']
            vol_score = 0
            if vol_ratio > 1.8:  # Strong volume
                vol_score = 1.0
                quality_checks += 1
            elif vol_ratio > 1.4:
                vol_score = 0.6

            score += vol_score * 0.15

            # 5. Price momentum (10% weight)
            mom_5 = df.iloc[i]['momentum_5']
            mom_10 = df.iloc[i]['momentum_10']
            mom_score = 0

            if mom_5 > 2 and mom_10 > 5 and mom_5 < 20:  # Positive but not excessive
                mom_score = 1.0
                quality_checks += 1
            elif mom_5 > 0 and mom_10 > 0:
                mom_score = 0.5

            score += mom_score * 0.10

            # 6. Bollinger position (10% weight)
            bb_pos = (df.iloc[i]['close'] - df.iloc[i]['bb_lower']) / (df.iloc[i]['bb_upper'] - df.iloc[i]['bb_lower'])
            bb_score = 0

            if 0.5 <= bb_pos <= 0.75:  # Upper half but not extreme
                bb_score = 1.0
                quality_checks += 1
            elif 0.3 <= bb_pos <= 0.8:
                bb_score = 0.6

            score += bb_score * 0.10

            # Quality gate: require at least 4 quality factors
            if quality_checks < 4:
                score *= 0.6  # Heavy penalty

            # v1.5.0 enhancements (conservative)
            if i >= 50:
                # MTF DL2 - avoid extreme volatility
                recent_closes = df.iloc[i-19:i+1]['close']
                volatility = recent_closes.std() / recent_closes.mean()
                if volatility > 0.06:  # High volatility penalty
                    score *= 0.85

                # Structure check - consistent uptrend
                if i >= 10:
                    recent_trend = df.iloc[i-9:i+1]['close']
                    upward_days = sum(1 for j in range(1, len(recent_trend))
                                    if recent_trend.iloc[j] > recent_trend.iloc[j-1])
                    if upward_days >= 6:  # 6+ up days in last 10
                        score += 0.05

                # Breakout confirmation
                if i >= 30:
                    resistance = df.iloc[i-29:i]['high'].quantile(0.8)
                    if (df.iloc[i]['high'] > resistance and
                        df.iloc[i]['volume_ratio'] > 2.0 and
                        df.iloc[i]['close'] > df.iloc[i]['open']):
                        score += 0.04

            # Apply strict threshold
            signal = score >= entry_threshold

            # Cooldown filter
            if signal and (i - last_signal_idx) < cooldown_bars:
                signal = False
            elif signal:
                last_signal_idx = i

            signals.append(signal)

        return pd.Series(signals)

    def backtest_with_risk_management(self, df, signals):
        """Backtest with enhanced risk management."""
        capital = self.initial_capital
        position = 0
        trades = []
        equity_curve = [self.initial_capital]

        in_position = False
        entry_price = 0
        entry_date = None
        stop_loss = 0

        for i in range(len(df)):
            current_price = df.iloc[i]['close']
            current_date = df.iloc[i]['date']
            signal = signals.iloc[i]

            portfolio_value = capital + (position * current_price)
            equity_curve.append(portfolio_value)

            trade_amount = 1000

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
                # Set stop loss at 6% below entry
                stop_loss = entry_price * 0.94

            elif in_position:
                # Exit conditions
                exit_signal = False
                exit_reason = ""

                # 1. Technical exit - signal ends
                if not signal:
                    exit_signal = True
                    exit_reason = "signal_end"

                # 2. Stop loss
                elif current_price <= stop_loss:
                    exit_signal = True
                    exit_reason = "stop_loss"

                # 3. Take profit at 15%
                elif current_price >= entry_price * 1.15:
                    exit_signal = True
                    exit_reason = "take_profit"

                # 4. Time stop at 15 days
                elif (current_date - entry_date).days >= 15:
                    exit_signal = True
                    exit_reason = "time_stop"

                # 5. Final day
                elif i == len(df) - 1:
                    exit_signal = True
                    exit_reason = "final_day"

                if exit_signal:
                    effective_price = current_price * (1 - self.slippage_rate)
                    gross_proceeds = position * effective_price
                    fees = gross_proceeds * self.fee_rate
                    net_proceeds = gross_proceeds - fees

                    capital += net_proceeds
                    pnl = net_proceeds - trade_amount
                    return_pct = (pnl / trade_amount) * 100

                    trades.append({
                        'entry_date': entry_date,
                        'exit_date': current_date,
                        'entry_price': entry_price,
                        'exit_price': effective_price,
                        'days_held': (current_date - entry_date).days,
                        'pnl': pnl,
                        'return_pct': return_pct,
                        'exit_reason': exit_reason
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
        max_dd = max(((peak - val) / peak) * 100 for val in equity_curve)

        # Other metrics
        days_in_test = (df.iloc[-1]['date'] - df.iloc[0]['date']).days
        months = days_in_test / 30.44
        trades_per_month = len(trades) / months if months > 0 else 0

        gross_wins = sum(t['pnl'] for t in trades if t['pnl'] > 0)
        gross_losses = abs(sum(t['pnl'] for t in trades if t['pnl'] <= 0))
        profit_factor = gross_wins / gross_losses if gross_losses > 0 else float('inf')

        return {
            'final_value': final_value,
            'total_return_pct': total_return,
            'total_trades': len(trades),
            'win_rate': win_rate,
            'trades_per_month': trades_per_month,
            'max_drawdown': max_dd,
            'profit_factor': profit_factor,
            'avg_return_per_trade': np.mean([t['return_pct'] for t in trades]) if trades else 0,
            'avg_days_held': np.mean([t['days_held'] for t in trades]) if trades else 0,
            'trades': trades
        }

    def run_final_test(self):
        """Run the final optimized test."""
        print("Creating ETH data...")
        df = self.create_eth_data(600)

        print("Calculating indicators...")
        df = self.calculate_indicators(df)

        print("Generating high-quality signals...")
        signals = self.generate_high_quality_signals(df)

        print("Running optimized backtest...")
        results = self.backtest_with_risk_management(df, signals)

        return {
            'test_period': {
                'start_date': str(df['date'].min().date()),
                'end_date': str(df['date'].max().date()),
                'days': len(df)
            },
            'results': results
        }


def main():
    backtest = ETHFinalOptimized()
    test_results = backtest.run_final_test()

    print("\n" + "="*70)
    print("BULL MACHINE v1.5.0 FINAL OPTIMIZED RESULTS")
    print("="*70)

    period = test_results['test_period']
    r = test_results['results']

    print(f"Test Period: {period['start_date']} to {period['end_date']} ({period['days']} days)")
    print(f"Initial Capital: $10,000")

    print(f"\n{'PERFORMANCE METRICS':<35} {'Value':<15} {'Target':<15} {'Status'}")
    print("-" * 70)

    # Core metrics with targets
    metrics = [
        ('Total Return', f"{r['total_return_pct']:.2f}%", 'Positive', '✓' if r['total_return_pct'] > 0 else '✗'),
        ('Final Value', f"${r['final_value']:,.0f}", f">${10000:,}+", '✓' if r['final_value'] > 10000 else '✗'),
        ('Total Trades', f"{r['total_trades']}", '24-48', '✓' if 24 <= r['total_trades'] <= 48 else '✗'),
        ('Trades/Month', f"{r['trades_per_month']:.1f}", '2-4', '✓' if 2 <= r['trades_per_month'] <= 4 else '✗'),
        ('Win Rate', f"{r['win_rate']:.1f}%", '≥50%', '✓' if r['win_rate'] >= 50 else '✗'),
        ('Max Drawdown', f"{r['max_drawdown']:.2f}%", '≤20%', '✓' if r['max_drawdown'] <= 20 else '✗'),
        ('Profit Factor', f"{r['profit_factor']:.2f}", '≥1.5', '✓' if r['profit_factor'] >= 1.5 else '✗'),
        ('Avg Days Held', f"{r['avg_days_held']:.1f}", '3-20', '✓' if 3 <= r['avg_days_held'] <= 20 else '✗')
    ]

    for metric, value, target, status in metrics:
        print(f"{metric:<35} {value:<15} {target:<15} {status}")

    # Overall assessment
    all_targets = [
        r['total_return_pct'] > 0,
        r['final_value'] > 10000,
        2 <= r['trades_per_month'] <= 4,
        r['win_rate'] >= 50,
        r['max_drawdown'] <= 20,
        r['profit_factor'] >= 1.5
    ]

    targets_met = sum(all_targets)
    total_targets = len(all_targets)

    print(f"\n🎯 OVERALL ASSESSMENT:")
    print(f"   Targets Met: {targets_met}/{total_targets} ({targets_met/total_targets*100:.1f}%)")

    if targets_met >= 5:
        print(f"   Status: {'🚀 READY FOR RC PROMOTION!' if targets_met == total_targets else '✅ STRONG CANDIDATE FOR RC'}")
        readiness = "READY" if targets_met == total_targets else "STRONG_CANDIDATE"
    elif targets_met >= 4:
        print(f"   Status: ⚠️  NEEDS MINOR ADJUSTMENTS")
        readiness = "MINOR_ADJUSTMENTS"
    else:
        print(f"   Status: ❌ NEEDS MAJOR OPTIMIZATION")
        readiness = "MAJOR_OPTIMIZATION"

    # Save detailed results
    with open('eth_final_results.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Metric', 'Value', 'Target', 'Status'])
        for metric, value, target, status in metrics:
            writer.writerow([metric, value, target, status])

    # Trade breakdown by exit reason
    exit_reasons = {}
    for trade in r['trades']:
        reason = trade.get('exit_reason', 'unknown')
        if reason not in exit_reasons:
            exit_reasons[reason] = {'count': 0, 'wins': 0}
        exit_reasons[reason]['count'] += 1
        if trade['pnl'] > 0:
            exit_reasons[reason]['wins'] += 1

    if exit_reasons:
        print(f"\nTRADE EXIT ANALYSIS:")
        for reason, data in exit_reasons.items():
            win_rate_reason = (data['wins'] / data['count']) * 100
            print(f"  {reason.replace('_', ' ').title()}: {data['count']} trades, {win_rate_reason:.1f}% wins")

    print(f"\n✓ Detailed results saved to eth_final_results.csv")
    print(f"✓ System readiness: {readiness}")

    return readiness


if __name__ == "__main__":
    main()