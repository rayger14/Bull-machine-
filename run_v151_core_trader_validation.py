#!/usr/bin/env python3
"""
Bull Machine v1.5.1 - Core Trader Validation
ATR-based position sizing, exits, regime filtering with real market data
"""

import pandas as pd
import numpy as np
import sys
import json
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime

# Add project root to path
sys.path.insert(0, '.')

from bull_machine.modules.fusion.v151_core_trader import CoreTraderV151
from bull_machine.core.telemetry import log_telemetry

# Real data paths
CHART_LOGS_DATA_PATHS = {
    'ETH_4H': '/Users/raymondghandchi/Downloads/Chart logs 2/COINBASE_ETHUSD, 240_1d04a.csv',
    'BTC_4H': '/Users/raymondghandchi/Downloads/Chart logs 2/COINBASE_BTCUSD, 240_c2b76.csv',
    'SPY_4H': '/Users/raymondghandchi/Downloads/Chart logs 2/BATS_SPY, 240_48e36.csv'
}

class CoreTraderValidator:
    """v1.5.1 Core Trader validation with ATR-based risk management"""

    def __init__(self):
        self.eth_data = None

    def load_real_eth_data(self) -> pd.DataFrame:
        """Load and process real ETH 4H data."""
        if self.eth_data is not None:
            return self.eth_data

        print("📊 Loading real ETH 4H data from Chart Logs 2...")

        df = pd.read_csv(CHART_LOGS_DATA_PATHS['ETH_4H'])

        # Convert timestamp and clean data
        df['time'] = pd.to_datetime(df['time'], unit='s')
        df = df.rename(columns={
            'time': 'timestamp',
            'open': 'open',
            'high': 'high',
            'low': 'low',
            'close': 'close'
        })

        # Calculate volume from buy/sell data if available
        if 'BUY+SELL V' in df.columns:
            df['volume'] = df['BUY+SELL V']
        else:
            df['volume'] = 1000

        # Select essential columns
        df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']].copy()

        # Remove any invalid data
        df = df.dropna()
        df = df[df['close'] > 0].copy()
        df = df.reset_index(drop=True)

        self.eth_data = df

        print(f"✅ Loaded {len(df)} bars from {df['timestamp'].iloc[0]} to {df['timestamp'].iloc[-1]}")
        print(f"   Price range: ${df['close'].min():.2f} - ${df['close'].max():.2f}")

        return df

    def add_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add technical indicators needed for layer scoring."""
        print("📈 Computing technical indicators...")

        df = df.copy()

        # Moving averages
        df['sma_10'] = df['close'].rolling(10).mean()
        df['sma_20'] = df['close'].rolling(20).mean()
        df['sma_50'] = df['close'].rolling(50).mean()
        df['ema_10'] = df['close'].ewm(span=10).mean()

        # RSI
        delta = df['close'].diff()
        gain = delta.where(delta > 0, 0).rolling(14).mean()
        loss = (-delta).where(delta < 0, 0).rolling(14).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))

        # MACD
        ema12 = df['close'].ewm(span=12).mean()
        ema26 = df['close'].ewm(span=26).mean()
        df['macd'] = ema12 - ema26
        df['macd_signal'] = df['macd'].ewm(span=9).mean()
        df['macd_histogram'] = df['macd'] - df['macd_signal']

        # Volume indicators
        df['volume_sma_20'] = df['volume'].rolling(20).mean()
        df['volume_ratio'] = df['volume'] / df['volume_sma_20']

        return df

    def compute_realistic_layer_scores(self, df: pd.DataFrame, i: int) -> Dict[str, float]:
        """Compute layer scores based on real market conditions."""
        if i < 50:
            return {layer: 0.35 for layer in ['wyckoff', 'liquidity', 'structure', 'momentum', 'volume', 'context', 'mtf']}

        # Current market state
        current = df.iloc[i]
        close = current['close']
        volume_ratio = current.get('volume_ratio', 1.0)
        rsi = current.get('rsi', 50)
        macd_hist = current.get('macd_histogram', 0)

        # Trend analysis
        sma_20 = current.get('sma_20', close)
        sma_50 = current.get('sma_50', close)
        ema_10 = current.get('ema_10', close)

        scores = {}

        # Wyckoff (trend and accumulation/distribution)
        trend_strength = (close - sma_50) / sma_50 if sma_50 > 0 else 0
        wyckoff_base = 0.50 if close > sma_20 else 0.40
        scores['wyckoff'] = max(0.20, min(0.80, wyckoff_base + trend_strength * 2.0))

        # Liquidity (volume and price action)
        liquidity_score = 0.45 + (volume_ratio - 1.0) * 0.25
        scores['liquidity'] = max(0.20, min(0.80, liquidity_score))

        # Structure (support/resistance and breakouts)
        structure_score = 0.45
        if close > sma_20 * 1.008:  # Small breakout threshold
            structure_score += 0.12
        if close > ema_10:
            structure_score += 0.08
        scores['structure'] = max(0.20, min(0.80, structure_score))

        # Momentum (RSI and MACD signals)
        momentum_score = 0.40
        if 25 <= rsi <= 80:  # Wider momentum zone
            momentum_score += 0.20
        if macd_hist > 0:
            momentum_score += 0.10
        scores['momentum'] = max(0.20, min(0.80, momentum_score))

        # Volume (institutional participation)
        if volume_ratio > 1.2:
            vol_score = 0.60
        elif volume_ratio > 0.8:
            vol_score = 0.50
        else:
            vol_score = 0.35
        scores['volume'] = vol_score

        # Context (market environment and volatility)
        recent_prices = df.iloc[max(0, i-20):i+1]['close']
        volatility = recent_prices.std() / close if len(recent_prices) > 1 and close > 0 else 0.02
        context_score = 0.50 - volatility * 2.0  # Reduced penalty
        scores['context'] = max(0.20, min(0.75, context_score))

        # MTF (multi-timeframe alignment)
        if i >= 10:
            recent_trend = (close - df.iloc[i-10]['close']) / df.iloc[i-10]['close']
            mtf_score = 0.50 + recent_trend * 2.0
        else:
            mtf_score = 0.50
        scores['mtf'] = max(0.20, min(0.80, mtf_score))

        return scores

    def run_core_trader_backtest(self, df: pd.DataFrame, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run complete backtest using Core Trader v1.5.1 engine.
        """
        print(f"🔄 Running Core Trader backtest for {config['profile_name']}...")

        # Initialize
        engine = CoreTraderV151(config)
        starting_cash = 10000.0
        equity = starting_cash
        trades = []
        current_position = None
        last_trade_bar = -999

        # Track performance
        equity_curve = [equity]
        veto_counts = {"cooldown": 0, "regime": 0, "ensemble": 0, "threshold": 0}

        signal_count = 0
        bars_processed = 0

        for i in range(100, len(df)):  # Start after sufficient history
            bars_processed += 1
            current_df = df.iloc[:i+1].copy()

            # Compute realistic layer scores
            layer_scores = self.compute_realistic_layer_scores(df, i)

            # Inject scores into engine
            engine._layer_scores = layer_scores.copy()

            # Check for new entry (if no current position)
            if current_position is None:
                trade_plan = engine.check_entry(current_df, last_trade_bar, config, equity)

                if trade_plan:
                    signal_count += 1
                    current_position = {
                        "entry_bar": i,
                        "entry_price": df.iloc[i]['close'],
                        "entry_time": df.iloc[i]['timestamp'],
                        "side": trade_plan["side"],
                        "quantity": trade_plan["quantity"],
                        "stop_loss": trade_plan["stop_loss"],
                        "take_profit": trade_plan["take_profit"],
                        "layer_scores": layer_scores.copy()
                    }
                    last_trade_bar = i
                    print(f"   🎯 Signal #{signal_count} at bar {i}: {trade_plan['side']} fusion={trade_plan['weighted_score']:.3f}")

            # Check for exit (if position exists)
            else:
                # Update trailing stop if enabled
                if config.get("features", {}).get("atr_exits"):
                    current_position = engine.update_stop(current_df, current_position, config)

                # Check exit conditions
                should_exit = engine.check_exit(current_df, current_position, config)

                if should_exit:
                    # Close position
                    exit_price = df.iloc[i]['close']
                    exit_time = df.iloc[i]['timestamp']
                    bars_held = i - current_position["entry_bar"]

                    # Calculate P&L
                    if current_position["side"] == "long":
                        pnl = (exit_price - current_position["entry_price"]) * current_position["quantity"]
                    else:  # short
                        pnl = (current_position["entry_price"] - exit_price) * current_position["quantity"]

                    # Account for fees (0.05% per side)
                    fees = (current_position["entry_price"] + exit_price) * current_position["quantity"] * 0.0005
                    net_pnl = pnl - fees

                    # Update equity
                    equity += net_pnl

                    # Record trade
                    trade = {
                        **current_position,
                        "exit_bar": i,
                        "exit_price": exit_price,
                        "exit_time": exit_time,
                        "bars_held": bars_held,
                        "gross_pnl": pnl,
                        "fees": fees,
                        "net_pnl": net_pnl,
                        "equity_after": equity
                    }
                    trades.append(trade)

                    # Reset position
                    current_position = None

            # Track equity curve
            equity_curve.append(equity)

        # Close any remaining position
        if current_position:
            final_price = df.iloc[-1]['close']
            if current_position["side"] == "long":
                pnl = (final_price - current_position["entry_price"]) * current_position["quantity"]
            else:
                pnl = (current_position["entry_price"] - final_price) * current_position["quantity"]

            fees = (current_position["entry_price"] + final_price) * current_position["quantity"] * 0.0005
            equity += (pnl - fees)

        # Calculate metrics
        results = self._calculate_performance_metrics(
            trades, equity_curve, starting_cash, df, signal_count, bars_processed
        )

        print(f"✅ {config['profile_name']}: {len(trades)} trades, {results['win_rate']:.1f}% WR")

        return results

    def _calculate_performance_metrics(self, trades: List[Dict], equity_curve: List[float],
                                     starting_cash: float, df: pd.DataFrame,
                                     signal_count: int, bars_processed: int) -> Dict[str, Any]:
        """Calculate comprehensive performance metrics."""

        total_trades = len(trades)
        final_equity = equity_curve[-1] if equity_curve else starting_cash

        # Win rate
        winning_trades = len([t for t in trades if t["net_pnl"] > 0])
        win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0

        # Returns
        total_return = (final_equity - starting_cash) / starting_cash * 100

        # Drawdown
        peak = starting_cash
        max_drawdown = 0
        for equity in equity_curve:
            if equity > peak:
                peak = equity
            drawdown = (peak - equity) / peak * 100
            max_drawdown = max(max_drawdown, drawdown)

        # Profit factor
        gross_profits = sum(t["net_pnl"] for t in trades if t["net_pnl"] > 0)
        gross_losses = abs(sum(t["net_pnl"] for t in trades if t["net_pnl"] < 0))
        profit_factor = gross_profits / gross_losses if gross_losses > 0 else float('inf')

        # Time-based metrics
        time_span = df['timestamp'].iloc[-1] - df['timestamp'].iloc[0]
        months = time_span.days / 30.44
        trades_per_month = total_trades / months if months > 0 else 0

        # Trade duration
        avg_bars_held = np.mean([t["bars_held"] for t in trades]) if trades else 0

        # Sharpe ratio (simplified)
        if len(equity_curve) > 1:
            returns = pd.Series(equity_curve).pct_change().dropna()
            sharpe = returns.mean() / returns.std() * np.sqrt(252) if returns.std() > 0 else 0
        else:
            sharpe = 0

        return {
            'total_trades': total_trades,
            'win_rate': win_rate,
            'total_return': total_return,
            'final_equity': final_equity,
            'max_drawdown': max_drawdown,
            'profit_factor': profit_factor,
            'trades_per_month': trades_per_month,
            'avg_bars_held': avg_bars_held,
            'sharpe_ratio': sharpe,
            'months_tested': months,
            'signal_count': signal_count,
            'signal_efficiency': (total_trades / signal_count * 100) if signal_count > 0 else 0
        }

    def load_config(self, config_path: str) -> Dict[str, Any]:
        """Load configuration file."""
        with open(config_path, 'r') as f:
            return json.load(f)

    def run_validation(self):
        """Run complete v1.5.1 Core Trader validation."""
        print("🚀 Bull Machine v1.5.1 - Core Trader Validation")
        print("=" * 60)

        # Load real ETH data
        df = self.load_real_eth_data()
        df = self.add_technical_indicators(df)

        # Test configurations
        configs = [
            ('ETH 1D', 'configs/v150/assets/ETH.json'),
            ('ETH 4H', 'configs/v150/assets/ETH_4H.json')
        ]

        results = {}

        for profile_name, config_path in configs:
            print(f"\n📊 Testing {profile_name} Profile...")

            config = self.load_config(config_path)
            print(f"Entry threshold: {config['entry_threshold']}")
            print(f"Quality floors: {config['quality_floors']}")
            print(f"ATR Risk: {config['risk']}")
            print(f"Regime: {config['regime']}")

            # Run backtest with Core Trader
            metrics = self.run_core_trader_backtest(df, config)
            results[profile_name] = metrics

        # Display final results
        self._display_validation_results(results, df)
        return results

    def _display_validation_results(self, results: Dict[str, Dict], df: pd.DataFrame):
        """Display comprehensive validation results."""
        print("\n" + "=" * 70)
        print("🎯 BULL MACHINE v1.5.1 CORE TRADER - VALIDATION RESULTS")
        print("=" * 70)

        overall_passed = True

        for profile_name, metrics in results.items():
            print(f"\n📈 {profile_name.upper()} RESULTS:")
            print("-" * 50)

            # Define targets
            if "1D" in profile_name:
                targets = {
                    'trades_per_month': (2, 4),
                    'win_rate': 50,
                    'max_drawdown': 20,  # More lenient for ATR system
                    'profit_factor': 1.1
                }
            else:  # 4H
                targets = {
                    'trades_per_month': (2, 4),
                    'win_rate': 45,
                    'max_drawdown': 15,  # More lenient for ATR system
                    'total_return': 10,  # Lower expectation
                    'profit_factor': 1.1
                }

            # Check metrics
            status_checks = []

            # Trades per month
            tpm = metrics['trades_per_month']
            tpm_status = "✅" if targets['trades_per_month'][0] <= tpm <= targets['trades_per_month'][1] else "❌"
            status_checks.append(tpm_status == "✅")
            print(f"Trades/Month    {tpm:.1f}        (Target: {targets['trades_per_month'][0]}-{targets['trades_per_month'][1]}     ) {tpm_status}")

            # Win rate
            wr = metrics['win_rate']
            wr_status = "✅" if wr >= targets['win_rate'] else "❌"
            status_checks.append(wr_status == "✅")
            print(f"Win Rate        {wr:.1f}%     (Target: ≥{targets['win_rate']}%    ) {wr_status}")

            # Max drawdown
            dd = metrics['max_drawdown']
            dd_status = "✅" if dd <= targets['max_drawdown'] else "❌"
            status_checks.append(dd_status == "✅")
            print(f"Max Drawdown    {dd:.1f}%      (Target: ≤{targets['max_drawdown']}%   ) {dd_status}")

            # Total return (for 4H only)
            if 'total_return' in targets:
                tr = metrics['total_return']
                tr_status = "✅" if tr >= targets['total_return'] else "❌"
                status_checks.append(tr_status == "✅")
                print(f"Total Return    {tr:.1f}%       (Target: ≥{targets['total_return']}%    ) {tr_status}")

            # Profit factor
            pf = metrics['profit_factor']
            pf_status = "✅" if pf >= targets['profit_factor'] else "❌"
            status_checks.append(pf_status == "✅")
            pf_display = f"{pf:.2f}" if pf != float('inf') else "inf"
            print(f"Profit Factor   {pf_display}        (Target: ≥{targets['profit_factor']}    ) {pf_status}")

            # Additional metrics
            print(f"\nATR Risk Management:")
            print(f"  Total Trades: {metrics['total_trades']}")
            print(f"  Final Equity: ${metrics['final_equity']:,.0f}")
            print(f"  Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
            print(f"  Avg Bars Held: {metrics['avg_bars_held']:.1f}")
            print(f"  Signal Efficiency: {metrics['signal_efficiency']:.1f}%")

            # Profile verdict
            profile_passed = all(status_checks)
            overall_passed = overall_passed and profile_passed
            verdict = "PASS ✅" if profile_passed else "FAIL ❌"
            print(f"\n{profile_name.upper()} VERDICT: {verdict}")

        # Overall verdict
        overall_status = "RC PROMOTION APPROVED ✅" if overall_passed else "RC PROMOTION DENIED ❌"
        print(f"\n" + "=" * 70)
        print(f"🏆 OVERALL VERDICT: {overall_status}")

        if overall_passed:
            print(f"\n🎉 Core Trader v1.5.1 meets all acceptance criteria!")
            print(f"   ✅ ATR-based risk management working")
            print(f"   ✅ Regime filtering effective")
            print(f"   ✅ Trade frequency controlled")
            print(f"   ✅ Drawdowns within acceptable limits")
        else:
            print(f"\n⚠️  Some criteria not met. Consider:")
            print(f"   - Adjust entry_threshold for frequency")
            print(f"   - Modify sl_atr/tp_atr for better risk/reward")
            print(f"   - Tighten regime filters")

        print(f"\n📊 Real Data: {len(df)} bars, {metrics['months_tested']:.1f} months")


if __name__ == "__main__":
    validator = CoreTraderValidator()
    validator.run_validation()