#!/usr/bin/env python3
"""
Bull Machine v1.4.2 vs v1.5.0 Standardized Comparison
Same dates, identical trade-sizing, fees, and slippage
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import json
from pathlib import Path

# Import v1.4.2 baseline (frozen)
from bull_machine.core.fusion_engine_v141 import FusionEngineV141
# Import v1.5.0 enhanced
from bull_machine.core.fusion_enhanced import FusionEngineV150
from bull_machine.core.config_loader import load_config


class StandardizedBacktest:
    """Standardized backtest with identical parameters for fair comparison."""

    def __init__(self, initial_capital: float = 10000.0):
        self.initial_capital = initial_capital
        self.fee_rate = 0.001  # 0.1% per trade
        self.slippage_rate = 0.0005  # 0.05% slippage

    def load_eth_data(self) -> pd.DataFrame:
        """Load ETH data with standardized columns."""
        try:
            # Try multiple possible filenames
            possible_files = [
                "data/ETH_1D.csv",
                "data/ETH-USD_1D.csv",
                "data/ETHUSD_1D.csv",
                "data/eth_daily.csv"
            ]

            df = None
            for filepath in possible_files:
                if Path(filepath).exists():
                    df = pd.read_csv(filepath)
                    print(f"Loaded data from {filepath}")
                    break

            if df is None:
                raise FileNotFoundError("No ETH data file found")

            # Standardize column names
            column_map = {
                'Date': 'date', 'date': 'date', 'timestamp': 'date',
                'Open': 'open', 'open': 'open',
                'High': 'high', 'high': 'high',
                'Low': 'low', 'low': 'low',
                'Close': 'close', 'close': 'close',
                'Volume': 'volume', 'volume': 'volume'
            }

            # Remove spaces and map columns
            df.columns = df.columns.str.strip()
            df = df.rename(columns=column_map)

            # Convert date column
            df['date'] = pd.to_datetime(df['date'])
            df = df.sort_values('date').reset_index(drop=True)

            # Ensure numeric columns
            numeric_cols = ['open', 'high', 'low', 'close', 'volume']
            for col in numeric_cols:
                df[col] = pd.to_numeric(df[col], errors='coerce')

            return df.dropna()

        except Exception as e:
            print(f"Error loading ETH data: {e}")
            return pd.DataFrame()

    def calculate_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate standard technical indicators."""
        df = df.copy()

        # Simple Moving Averages
        df['sma_20'] = df['close'].rolling(20).mean()
        df['sma_50'] = df['close'].rolling(50).mean()

        # RSI
        delta = df['close'].diff()
        gain = delta.where(delta > 0, 0).rolling(14).mean()
        loss = (-delta).where(delta < 0, 0).rolling(14).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))

        # Bollinger Bands
        bb_period = 20
        df['bb_middle'] = df['close'].rolling(bb_period).mean()
        bb_std = df['close'].rolling(bb_period).std()
        df['bb_upper'] = df['bb_middle'] + (bb_std * 2)
        df['bb_lower'] = df['bb_middle'] - (bb_std * 2)

        # Volume indicators
        df['volume_sma'] = df['volume'].rolling(20).mean()
        df['volume_ratio'] = df['volume'] / df['volume_sma']

        return df

    def generate_signals_v142(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate v1.4.2 baseline signals."""
        df = df.copy()

        # Load v1.4.2 config (frozen baseline)
        config_v142 = {
            "entry_threshold": 0.45,
            "quality_floors": {
                "wyckoff": 0.37, "liquidity": 0.32, "structure": 0.35,
                "momentum": 0.40, "volume": 0.30, "context": 0.35, "mtf": 0.40
            },
            "features": {  # All features OFF for baseline
                "mtf_dl2": False, "six_candle_leg": False,
                "orderflow_lca": False, "negative_vip": False, "live_data": False
            }
        }

        # Simple confluence model for v1.4.2
        signal_strength = []

        for i in range(len(df)):
            if i < 50:  # Need enough data for indicators
                signal_strength.append(0)
                continue

            strength = 0

            # Trend analysis
            if df.iloc[i]['close'] > df.iloc[i]['sma_20']:
                strength += 0.15
            if df.iloc[i]['sma_20'] > df.iloc[i]['sma_50']:
                strength += 0.15

            # RSI momentum
            rsi = df.iloc[i]['rsi']
            if 30 <= rsi <= 70:  # Not oversold/overbought
                strength += 0.20

            # Bollinger position
            close = df.iloc[i]['close']
            bb_lower = df.iloc[i]['bb_lower']
            bb_upper = df.iloc[i]['bb_upper']
            if bb_lower <= close <= bb_upper:
                strength += 0.15

            # Volume confirmation
            if df.iloc[i]['volume_ratio'] > 1.2:
                strength += 0.10

            # Structure (simple price action)
            if i >= 3:
                recent_high = df.iloc[i-3:i+1]['high'].max()
                if df.iloc[i]['close'] == recent_high:
                    strength += 0.25

            signal_strength.append(min(strength, 1.0))

        df['signal_strength_v142'] = signal_strength
        df['signal_v142'] = df['signal_strength_v142'] >= config_v142["entry_threshold"]

        return df

    def generate_signals_v150(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate v1.5.0 enhanced signals with frequency controls."""
        df = df.copy()

        # Load v1.5.0 ETH config
        config_v150 = load_config("ETH", "v150")

        # Start with v1.4.2 baseline signals
        df = self.generate_signals_v142(df)

        # Apply v1.5.0 enhancements
        enhanced_signals = []
        cooldown_bars = config_v150.get("trade_frequency", {}).get("cooldown_bars", 6)
        last_signal_index = -cooldown_bars  # Allow first signal

        for i in range(len(df)):
            base_signal = df.iloc[i]['signal_v142']
            enhanced_strength = df.iloc[i]['signal_strength_v142']

            # Apply v1.5.0 alpha enhancements (reduced impact)
            if i >= 20:  # Need data for patterns

                # MTF DL2 filter (extreme condition check)
                recent_prices = df.iloc[i-19:i+1]['close'].values
                price_std = np.std(recent_prices)
                price_mean = np.mean(recent_prices)
                if price_std > 0:
                    z_score = abs((df.iloc[i]['close'] - price_mean) / price_std)
                    if z_score > 2.5:  # Extreme condition, reduce signal
                        enhanced_strength *= 0.9

                # 6-candle leg structure (alternating pattern)
                if i >= 6:
                    closes = df.iloc[i-5:i+1]['close'].values
                    directions = [1 if closes[j] > closes[j-1] else -1 for j in range(1, len(closes))]
                    changes = sum(1 for j in range(1, len(directions)) if directions[j] != directions[j-1])
                    if changes >= 2:  # Good structure
                        enhanced_strength += 0.03  # Reduced from 0.05

                # Orderflow LCA (simplified breakout detection)
                if i >= 10:
                    recent_highs = df.iloc[i-9:i+1]['high'].values
                    recent_volumes = df.iloc[i-9:i+1]['volume'].values
                    current_high = df.iloc[i]['high']
                    prev_resistance = np.percentile(recent_highs[:-1], 75)

                    if current_high > prev_resistance and df.iloc[i]['volume_ratio'] > 1.5:
                        enhanced_strength += 0.04  # Reduced from 0.08

                # Negative VIP (reversal awareness - reduce overconfidence)
                if i >= 5:
                    price_change = (df.iloc[i]['close'] - df.iloc[i-4]['close']) / df.iloc[i-4]['close']
                    volume_surge = df.iloc[i]['volume_ratio'] > 2.0

                    if abs(price_change) > 0.15 and volume_surge:  # Potential reversal setup
                        enhanced_strength *= 0.95  # Slight reduction

            # Apply entry threshold
            signal = enhanced_strength >= config_v150["entry_threshold"]

            # Apply cooldown mechanism
            if signal and (i - last_signal_index) < cooldown_bars:
                signal = False  # Too soon after last signal

            if signal:
                last_signal_index = i

            enhanced_signals.append(signal)

        df['signal_v150'] = enhanced_signals
        return df

    def apply_cooldown_filter(self, signals: pd.Series, cooldown_bars: int = 6) -> pd.Series:
        """Apply cooldown period between signals."""
        filtered_signals = signals.copy()
        last_signal_idx = -cooldown_bars

        for i, signal in enumerate(signals):
            if signal and (i - last_signal_idx) < cooldown_bars:
                filtered_signals.iloc[i] = False
            elif signal:
                last_signal_idx = i

        return filtered_signals

    def backtest_strategy(self, df: pd.DataFrame, signal_column: str, version: str) -> Dict:
        """Run backtest with identical parameters."""

        signals = df[signal_column]
        prices = df['close']

        # Trading variables
        capital = self.initial_capital
        position = 0  # ETH held
        trades = []
        equity_curve = []

        # Trade sizing: fixed dollar amount per trade
        trade_amount = 1000  # $1000 per trade for consistency

        for i, (signal, price) in enumerate(zip(signals, prices)):
            date = df.iloc[i]['date']

            # Calculate current portfolio value
            portfolio_value = capital + (position * price)
            equity_curve.append(portfolio_value)

            if signal and position == 0:  # Enter position
                # Calculate position size with fees and slippage
                effective_price = price * (1 + self.slippage_rate)  # Slippage
                fees = trade_amount * self.fee_rate
                net_amount = trade_amount - fees

                if capital >= trade_amount:
                    position = net_amount / effective_price
                    capital -= trade_amount

                    trades.append({
                        'entry_date': date,
                        'entry_price': effective_price,
                        'trade_amount': trade_amount,
                        'position_size': position,
                        'type': 'BUY'
                    })

            elif not signal and position > 0:  # Exit position (simplified: exit when signal ends)
                # Calculate exit with fees and slippage
                effective_price = price * (1 - self.slippage_rate)  # Slippage
                gross_proceeds = position * effective_price
                fees = gross_proceeds * self.fee_rate
                net_proceeds = gross_proceeds - fees

                capital += net_proceeds

                # Update last trade with exit info
                if trades:
                    last_trade = trades[-1]
                    pnl = net_proceeds - last_trade['trade_amount']
                    trades[-1].update({
                        'exit_date': date,
                        'exit_price': effective_price,
                        'gross_proceeds': gross_proceeds,
                        'net_proceeds': net_proceeds,
                        'pnl': pnl,
                        'return_pct': (pnl / last_trade['trade_amount']) * 100
                    })

                position = 0

        # Final portfolio value
        final_value = capital + (position * prices.iloc[-1])
        total_return = ((final_value - self.initial_capital) / self.initial_capital) * 100

        # Calculate metrics
        completed_trades = [t for t in trades if 'exit_date' in t]
        winning_trades = [t for t in completed_trades if t['pnl'] > 0]

        metrics = {
            'version': version,
            'total_trades': len(completed_trades),
            'winning_trades': len(winning_trades),
            'win_rate': len(winning_trades) / len(completed_trades) * 100 if completed_trades else 0,
            'total_return_pct': total_return,
            'final_value': final_value,
            'trades_per_month': len(completed_trades) / (len(df) / 30.44) if len(df) > 30 else 0,
            'avg_return_per_trade': np.mean([t['return_pct'] for t in completed_trades]) if completed_trades else 0,
            'max_drawdown': self.calculate_max_drawdown(equity_curve),
            'trades': completed_trades
        }

        return metrics

    def calculate_max_drawdown(self, equity_curve: List[float]) -> float:
        """Calculate maximum drawdown percentage."""
        if not equity_curve:
            return 0

        peak = equity_curve[0]
        max_dd = 0

        for value in equity_curve:
            if value > peak:
                peak = value
            drawdown = ((peak - value) / peak) * 100
            max_dd = max(max_dd, drawdown)

        return max_dd

    def run_comparison(self) -> Dict:
        """Run standardized comparison between v1.4.2 and v1.5.0."""

        print("Loading ETH data...")
        df = self.load_eth_data()

        if df.empty:
            return {"error": "No data loaded"}

        print(f"Loaded {len(df)} rows of ETH data from {df['date'].min()} to {df['date'].max()}")

        # Calculate indicators
        print("Calculating technical indicators...")
        df = self.calculate_technical_indicators(df)

        # Generate signals
        print("Generating v1.4.2 baseline signals...")
        df = self.generate_signals_v142(df)

        print("Generating v1.5.0 enhanced signals...")
        df = self.generate_signals_v150(df)

        # Run backtests
        print("Running v1.4.2 backtest...")
        results_v142 = self.backtest_strategy(df, 'signal_v142', 'v1.4.2')

        print("Running v1.5.0 backtest...")
        results_v150 = self.backtest_strategy(df, 'signal_v150', 'v1.5.0')

        # Create comparison
        comparison = {
            'test_period': {
                'start_date': str(df['date'].min()),
                'end_date': str(df['date'].max()),
                'total_days': len(df)
            },
            'parameters': {
                'initial_capital': self.initial_capital,
                'fee_rate': self.fee_rate,
                'slippage_rate': self.slippage_rate,
                'trade_amount': 1000
            },
            'v142_results': results_v142,
            'v150_results': results_v150
        }

        return comparison


def main():
    """Run the standardized comparison."""
    backtest = StandardizedBacktest(initial_capital=10000.0)
    results = backtest.run_comparison()

    if 'error' in results:
        print(f"Error: {results['error']}")
        return

    # Print comparison results
    print("\n" + "="*60)
    print("BULL MACHINE v1.4.2 vs v1.5.0 STANDARDIZED COMPARISON")
    print("="*60)

    print(f"\nTest Period: {results['test_period']['start_date']} to {results['test_period']['end_date']}")
    print(f"Total Days: {results['test_period']['total_days']}")

    print(f"\nParameters:")
    print(f"  Initial Capital: ${results['parameters']['initial_capital']:,.2f}")
    print(f"  Fee Rate: {results['parameters']['fee_rate']:.1%}")
    print(f"  Slippage Rate: {results['parameters']['slippage_rate']:.2%}")

    print(f"\n{'Metric':<25} {'v1.4.2':<15} {'v1.5.0':<15} {'Difference':<15}")
    print("-" * 70)

    v142 = results['v142_results']
    v150 = results['v150_results']

    print(f"{'Total Return':<25} {v142['total_return_pct']:>8.2f}%    {v150['total_return_pct']:>8.2f}%    {v150['total_return_pct']-v142['total_return_pct']:>+8.2f}%")
    print(f"{'Final Value':<25} ${v142['final_value']:>8,.0f}    ${v150['final_value']:>8,.0f}    ${v150['final_value']-v142['final_value']:>+8,.0f}")
    print(f"{'Total Trades':<25} {v142['total_trades']:>11}    {v150['total_trades']:>11}    {v150['total_trades']-v142['total_trades']:>+11}")
    print(f"{'Win Rate':<25} {v142['win_rate']:>8.1f}%    {v150['win_rate']:>8.1f}%    {v150['win_rate']-v142['win_rate']:>+8.1f}%")
    print(f"{'Trades/Month':<25} {v142['trades_per_month']:>11.1f}    {v150['trades_per_month']:>11.1f}    {v150['trades_per_month']-v142['trades_per_month']:>+11.1f}")
    print(f"{'Avg Return/Trade':<25} {v142['avg_return_per_trade']:>8.2f}%    {v150['avg_return_per_trade']:>8.2f}%    {v150['avg_return_per_trade']-v142['avg_return_per_trade']:>+8.2f}%")
    print(f"{'Max Drawdown':<25} {v142['max_drawdown']:>8.2f}%    {v150['max_drawdown']:>8.2f}%    {v150['max_drawdown']-v142['max_drawdown']:>+8.2f}%")

    # ETH-1D Target Assessment
    print(f"\nETH-1D Target Assessment (v1.5.0):")
    print(f"  Trades/Month: {v150['trades_per_month']:.1f} (Target: 2-4) {'✓' if 2 <= v150['trades_per_month'] <= 4 else '✗'}")
    print(f"  Win Rate: {v150['win_rate']:.1f}% (Target: ≥50%) {'✓' if v150['win_rate'] >= 50 else '✗'}")
    print(f"  Max Drawdown: {v150['max_drawdown']:.1f}% (Target: ≤20%) {'✓' if v150['max_drawdown'] <= 20 else '✗'}")

    # Save results to CSV
    import csv

    # Save side-by-side comparison
    with open('comparison_results.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Metric', 'v1.4.2', 'v1.5.0', 'Difference'])
        writer.writerow(['Total Return (%)', f"{v142['total_return_pct']:.2f}", f"{v150['total_return_pct']:.2f}", f"{v150['total_return_pct']-v142['total_return_pct']:+.2f}"])
        writer.writerow(['Final Value ($)', f"{v142['final_value']:.2f}", f"{v150['final_value']:.2f}", f"{v150['final_value']-v142['final_value']:+.2f}"])
        writer.writerow(['Total Trades', v142['total_trades'], v150['total_trades'], v150['total_trades']-v142['total_trades']])
        writer.writerow(['Win Rate (%)', f"{v142['win_rate']:.1f}", f"{v150['win_rate']:.1f}", f"{v150['win_rate']-v142['win_rate']:+.1f}"])
        writer.writerow(['Trades/Month', f"{v142['trades_per_month']:.1f}", f"{v150['trades_per_month']:.1f}", f"{v150['trades_per_month']-v142['trades_per_month']:+.1f}"])
        writer.writerow(['Max Drawdown (%)', f"{v142['max_drawdown']:.2f}", f"{v150['max_drawdown']:.2f}", f"{v150['max_drawdown']-v142['max_drawdown']:+.2f}"])

    print(f"\n✓ Results saved to comparison_results.csv")

    # Save detailed trades
    all_trades = []
    for trade in v142['trades']:
        all_trades.append({**trade, 'version': 'v1.4.2'})
    for trade in v150['trades']:
        all_trades.append({**trade, 'version': 'v1.5.0'})

    if all_trades:
        with open('detailed_trades.csv', 'w', newline='') as f:
            if all_trades:
                fieldnames = list(all_trades[0].keys())
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(all_trades)

        print(f"✓ Detailed trades saved to detailed_trades.csv")


if __name__ == "__main__":
    main()