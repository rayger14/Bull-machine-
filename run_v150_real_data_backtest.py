#!/usr/bin/env python3
"""
Bull Machine v1.5.0 - Real Market Data Validation
Uses actual ETH 4H data from Chart Logs 2 for accurate validation
"""

import pandas as pd
import numpy as np
import sys
import json
from pathlib import Path
from typing import Dict, Any, Tuple
from datetime import datetime

# Add project root to path
sys.path.insert(0, '.')

from bull_machine.modules.fusion.v150_enhanced import FusionEngineV150
from bull_machine.core.telemetry import log_telemetry

# Real data paths for permanent reference
CHART_LOGS_DATA_PATHS = {
    'ETH_4H': '/Users/raymondghandchi/Downloads/Chart logs 2/COINBASE_ETHUSD, 240_1d04a.csv',
    'BTC_4H': '/Users/raymondghandchi/Downloads/Chart logs 2/COINBASE_BTCUSD, 240_c2b76.csv',
    'SPY_4H': '/Users/raymondghandchi/Downloads/Chart logs 2/BATS_SPY, 240_48e36.csv'
}

class RealDataV150Validator:
    """v1.5.0 validation using real market data from Chart Logs 2"""

    def load_real_eth_data(self) -> pd.DataFrame:
        """Load and process real ETH 4H data."""
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
            df['volume'] = 1000  # Fallback if volume data missing

        # Select essential columns
        df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']].copy()

        # Remove any invalid data
        df = df.dropna()
        df = df[df['close'] > 0].copy()

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

        # ATR
        df['tr'] = np.maximum(
            df['high'] - df['low'],
            np.maximum(
                abs(df['high'] - df['close'].shift(1)),
                abs(df['low'] - df['close'].shift(1))
            )
        )
        df['atr'] = df['tr'].rolling(14).mean()

        return df

    def compute_realistic_layer_scores(self, df: pd.DataFrame, i: int) -> Dict[str, float]:
        """Compute layer scores based on real market conditions."""
        if i < 50:  # Need sufficient history
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
        scores['wyckoff'] = max(0.25, min(0.80, wyckoff_base + trend_strength * 2.0))

        # Liquidity (volume and price action)
        liquidity_score = 0.45 + (volume_ratio - 1.0) * 0.25
        scores['liquidity'] = max(0.25, min(0.80, liquidity_score))

        # Structure (support/resistance and breakouts)
        structure_score = 0.45
        if close > sma_20 * 1.008:  # Small breakout threshold (real market)
            structure_score += 0.12
        if close > ema_10:
            structure_score += 0.08
        scores['structure'] = max(0.25, min(0.80, structure_score))

        # Momentum (RSI and MACD signals)
        momentum_score = 0.40
        if 30 <= rsi <= 75:  # Wider momentum zone for real data
            momentum_score += 0.20
        if macd_hist > 0:
            momentum_score += 0.10
        scores['momentum'] = max(0.25, min(0.80, momentum_score))

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
        context_score = 0.50 - volatility * 2.5  # Reduced volatility penalty for real data
        scores['context'] = max(0.25, min(0.75, context_score))

        # MTF (multi-timeframe alignment)
        if i >= 10:
            recent_trend = (close - df.iloc[i-10]['close']) / df.iloc[i-10]['close']
            mtf_score = 0.50 + recent_trend * 2.5
        else:
            mtf_score = 0.50
        scores['mtf'] = max(0.25, min(0.80, mtf_score))

        return scores

    def generate_v150_signals(self, df: pd.DataFrame, config: Dict[str, Any]) -> pd.Series:
        """Generate v1.5.0 signals using enhanced fusion engine with real data."""
        print(f"🔄 Generating v1.5.0 signals for {config['profile_name']}...")

        engine = FusionEngineV150(config)
        signals = []
        last_trade_bar = -999  # No previous trades

        # Layer weights for realistic fusion scoring
        layer_weights = {
            'wyckoff': 0.20,      # Primary trend analysis
            'liquidity': 0.15,    # Volume confirmation
            'structure': 0.18,    # Support/resistance
            'momentum': 0.15,     # RSI/MACD signals
            'volume': 0.12,       # Volume patterns
            'context': 0.10,      # Market environment
            'mtf': 0.10           # Multi-timeframe
        }

        signal_count = 0
        for i in range(len(df)):
            if i < 100:  # Need sufficient history
                signals.append(False)
                continue

            # Get data up to current bar
            current_df = df.iloc[:i+1].copy()

            # Compute realistic layer scores
            layer_scores = self.compute_realistic_layer_scores(df, i)

            # Inject scores into engine for v1.5.0 alpha processing
            engine._layer_scores = layer_scores.copy()

            # Check if entry is allowed (includes cooldown and quality floors)
            entry_allowed = engine.check_entry(current_df, last_trade_bar, config)

            if not entry_allowed:
                signals.append(False)
                continue

            # Check confluence vetoes with v1.5.0 alphas
            veto = engine.check_confluence_vetoes(current_df, layer_scores, config)

            if veto:
                signals.append(False)
                continue

            # Calculate fusion score with proper weights
            weighted_score = sum(
                layer_scores.get(layer, 0) * layer_weights.get(layer, 0.1)
                for layer in layer_weights.keys()
            )

            # Entry threshold check
            signal = weighted_score >= config.get('entry_threshold', 0.35)

            if signal:
                last_trade_bar = i
                signal_count += 1
                print(f"   🎯 Signal #{signal_count} at bar {i}: fusion={weighted_score:.3f}")

            signals.append(signal)

        print(f"✅ Generated {signal_count} signals")
        return pd.Series(signals, index=df.index)

    def simulate_backtest(self, df: pd.DataFrame, signals: pd.Series, config: Dict[str, Any]) -> Dict[str, float]:
        """Simulate backtest with proper position sizing and exits."""
        starting_cash = 10000
        cash = starting_cash
        position = 0
        position_value = 0
        equity_curve = []
        trades = []

        fee_rate = 0.0005  # 0.05% fees
        slippage_rate = 0.0002  # 0.02% slippage

        for i, (signal, row) in enumerate(zip(signals, df.itertuples())):
            current_price = row.close
            current_equity = cash + position * current_price
            equity_curve.append(current_equity)

            # Entry logic
            if signal and position == 0:
                # Risk 2% of equity per trade
                risk_amount = current_equity * 0.02
                atr = getattr(row, 'atr', current_price * 0.02)  # 2% default ATR
                stop_distance = atr * 2  # 2x ATR stop

                if stop_distance > 0:
                    position_size = risk_amount / stop_distance
                    entry_price = current_price * (1 + slippage_rate)  # Slippage
                    total_cost = position_size * entry_price * (1 + fee_rate)  # Fees

                    if total_cost <= cash:
                        position = position_size
                        cash -= total_cost
                        position_value = position_size * entry_price

                        trades.append({
                            'entry_bar': i,
                            'entry_price': entry_price,
                            'entry_time': row.timestamp,
                            'position_size': position_size,
                            'stop_price': entry_price - stop_distance
                        })

            # Exit logic (basic profit taking after 5 bars or stop loss)
            elif position > 0 and len(trades) > 0:
                trade = trades[-1]
                bars_held = i - trade['entry_bar']

                # Take profit after 5 bars or stop loss
                if bars_held >= 5 or current_price <= trade['stop_price']:
                    exit_price = current_price * (1 - slippage_rate)
                    total_proceeds = position * exit_price * (1 - fee_rate)

                    pnl = total_proceeds - position_value
                    trades[-1]['exit_bar'] = i
                    trades[-1]['exit_price'] = exit_price
                    trades[-1]['exit_time'] = row.timestamp
                    trades[-1]['pnl'] = pnl
                    trades[-1]['bars_held'] = bars_held

                    cash += total_proceeds
                    position = 0
                    position_value = 0

        # Close any remaining position
        if position > 0:
            final_price = df.iloc[-1]['close']
            final_proceeds = position * final_price * (1 - fee_rate - slippage_rate)
            cash += final_proceeds
            if len(trades) > 0:
                trades[-1]['exit_bar'] = len(df) - 1
                trades[-1]['exit_price'] = final_price
                trades[-1]['pnl'] = final_proceeds - position_value

        # Calculate metrics
        final_value = cash
        total_trades = len([t for t in trades if 'pnl' in t])
        winning_trades = len([t for t in trades if t.get('pnl', 0) > 0])
        win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0

        # Drawdown calculation
        peak = equity_curve[0]
        max_drawdown = 0
        for equity in equity_curve:
            if equity > peak:
                peak = equity
            drawdown = (peak - equity) / peak * 100
            max_drawdown = max(max_drawdown, drawdown)

        # Time period for trades/month
        time_span = df['timestamp'].iloc[-1] - df['timestamp'].iloc[0]
        months = time_span.days / 30.44
        trades_per_month = total_trades / months if months > 0 else 0

        return {
            'total_trades': total_trades,
            'win_rate': win_rate,
            'final_value': final_value,
            'total_return': (final_value - starting_cash) / starting_cash * 100,
            'max_drawdown': max_drawdown,
            'trades_per_month': trades_per_month,
            'profit_factor': sum(t.get('pnl', 0) for t in trades if t.get('pnl', 0) > 0) / abs(sum(t.get('pnl', 0) for t in trades if t.get('pnl', 0) < 0)) if any(t.get('pnl', 0) < 0 for t in trades) else float('inf'),
            'avg_bars_held': np.mean([t.get('bars_held', 0) for t in trades if 'bars_held' in t]) if total_trades > 0 else 0,
            'months_tested': months
        }

    def load_config(self, config_path: str) -> Dict[str, Any]:
        """Load v1.5.0 configuration."""
        with open(config_path, 'r') as f:
            return json.load(f)

    def run_validation(self):
        """Run complete v1.5.0 validation with real market data."""
        print("🚀 Bull Machine v1.5.0 - Real Market Data Validation")
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
            print(f"Cooldown bars: {config['cooldown_bars']}")

            # Generate signals
            signals = self.generate_v150_signals(df, config)

            # Run backtest
            metrics = self.simulate_backtest(df, signals, config)
            results[profile_name] = metrics

            print(f"✅ {profile_name}: {metrics['total_trades']} trades, {metrics['win_rate']:.1f}% WR")

        # Display results
        self.display_final_results(results, df)
        return results

    def display_final_results(self, results: Dict[str, Dict], df: pd.DataFrame):
        """Display comprehensive validation results."""
        print("\n" + "=" * 60)
        print("🎯 BULL MACHINE v1.5.0 - REAL DATA VALIDATION RESULTS")
        print("=" * 60)

        for profile_name, metrics in results.items():
            print(f"\n📈 {profile_name.upper()} RESULTS:")
            print("-" * 40)

            # Define targets based on profile
            if "1D" in profile_name:
                targets = {
                    'trades_per_month': (2, 4),
                    'win_rate': 50,
                    'max_drawdown': 9.2,
                    'profit_factor': 1.3
                }
            else:  # 4H
                targets = {
                    'trades_per_month': (2, 4),
                    'win_rate': 45,
                    'max_drawdown': 20,
                    'total_return': 30,
                    'profit_factor': 1.3
                }

            # Check each metric
            status_checks = []

            tpm = metrics['trades_per_month']
            tpm_status = "✅" if targets['trades_per_month'][0] <= tpm <= targets['trades_per_month'][1] else "❌"
            status_checks.append(tpm_status == "✅")
            print(f"Trades/Month    {tpm:.1f}        (Target: {targets['trades_per_month'][0]}-{targets['trades_per_month'][1]}     ) {tpm_status}")

            wr = metrics['win_rate']
            wr_status = "✅" if wr >= targets['win_rate'] else "❌"
            status_checks.append(wr_status == "✅")
            print(f"Win Rate        {wr:.1f}%     (Target: ≥{targets['win_rate']}%    ) {wr_status}")

            dd = metrics['max_drawdown']
            dd_status = "✅" if dd <= targets['max_drawdown'] else "❌"
            status_checks.append(dd_status == "✅")
            print(f"Max Drawdown    {dd:.1f}%      (Target: ≤{targets['max_drawdown']}%   ) {dd_status}")

            if 'total_return' in targets:
                tr = metrics['total_return']
                tr_status = "✅" if tr >= targets['total_return'] else "❌"
                status_checks.append(tr_status == "✅")
                print(f"Total Return    {tr:.1f}%       (Target: ≥{targets['total_return']}%    ) {tr_status}")

            pf = metrics['profit_factor']
            pf_status = "✅" if pf >= targets['profit_factor'] else "❌"
            status_checks.append(pf_status == "✅")
            pf_display = f"{pf:.1f}" if pf != float('inf') else "inf"
            print(f"Profit Factor   {pf_display}        (Target: ≥{targets['profit_factor']}    ) {pf_status}")

            print(f"\nAdditional Metrics:")
            print(f"  Total Trades: {metrics['total_trades']}")
            print(f"  Final Value: ${metrics['final_value']:,.0f}")
            print(f"  Avg Bars Held: {metrics['avg_bars_held']:.1f}")
            print(f"  Period Tested: {metrics['months_tested']:.1f} months")

            # Final verdict
            passed = all(status_checks)
            verdict = "PASS ✅" if passed else "FAIL ❌"
            print(f"\n{profile_name.upper()} VERDICT: {verdict}")

        # Overall verdict
        all_passed = all(
            all([
                targets['trades_per_month'][0] <= metrics['trades_per_month'] <= targets['trades_per_month'][1],
                metrics['win_rate'] >= (50 if "1D" in profile else 45),
                metrics['max_drawdown'] <= (9.2 if "1D" in profile else 20),
                metrics.get('total_return', 100) >= (0 if "1D" in profile else 30),
                metrics['profit_factor'] >= 1.3
            ])
            for profile, metrics in results.items()
        )

        overall_status = "RC PROMOTION APPROVED ✅" if all_passed else "RC PROMOTION DENIED ❌"
        print(f"\n" + "=" * 60)
        print(f"🏆 OVERALL VERDICT: {overall_status}")

        if not all_passed:
            print(f"\n⚠️  Some criteria not met. Real market data shows:")
            print(f"   - More conservative signal generation needed")
            print(f"   - Consider parameter adjustments for live trading")

        print(f"\n📊 Real Market Data: {len(df)} bars from {df['timestamp'].iloc[0].date()} to {df['timestamp'].iloc[-1].date()}")


if __name__ == "__main__":
    validator = RealDataV150Validator()
    validator.run_validation()