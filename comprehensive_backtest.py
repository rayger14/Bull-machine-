#!/usr/bin/env python3
"""
Bull Machine v1.4.1 Comprehensive Backtest
BTC/ETH validation with enhanced edge case logic
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path
import sys
import logging
from typing import Dict, List

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from bull_machine.scoring.fusion import FusionEngineV141
from bull_machine.modules.wyckoff.state_machine import WyckoffStateMachine
from bull_machine.modules.liquidity.imbalance import calculate_liquidity_score
from bull_machine.modules.bojan.candle_logic import wick_magnets
from bull_machine.modules.risk.dynamic_risk import calculate_dynamic_position_size, calculate_volatility_context
from bull_machine.strategy.exits.advanced_evaluator import AdvancedExitEvaluator


def create_realistic_data(symbol: str, bars: int = 500) -> pd.DataFrame:
    """Create realistic crypto-like price data for backtesting."""
    np.random.seed(42 if symbol == 'BTC' else 123)

    # Base prices
    if symbol == 'BTC':
        base_price = 60000
        volatility = 0.03
        trend = 0.0002
    else:  # ETH
        base_price = 3000
        volatility = 0.04
        trend = 0.0003

    # Generate trending random walk
    returns = np.random.normal(trend, volatility, bars)

    # Add some regime changes
    regime_changes = np.random.choice([0, 1], bars, p=[0.98, 0.02])
    volatility_multiplier = np.where(regime_changes, 3, 1)  # 3x vol during regime changes
    returns *= volatility_multiplier

    # Generate price levels
    prices = base_price * np.exp(np.cumsum(returns))

    # Create OHLC data
    dates = pd.date_range('2024-01-01', periods=bars, freq='h')

    df = pd.DataFrame(index=dates)
    df['close'] = prices
    df['open'] = df['close'].shift(1).fillna(prices[0])

    # Add realistic spreads and volume
    spreads = np.abs(np.random.normal(0, 0.005, bars)) * prices
    df['high'] = df[['open', 'close']].max(axis=1) + spreads
    df['low'] = df[['open', 'close']].min(axis=1) - spreads

    # Volume with patterns
    base_volume = 1000 if symbol == 'BTC' else 5000
    volume_trend = 1 + 0.3 * np.sin(np.arange(bars) * 0.02)
    volume_noise = 1 + np.random.normal(0, 0.3, bars)
    df['volume'] = base_volume * volume_trend * np.abs(volume_noise)

    return df


class ComprehensiveBacktester:
    """Enhanced backtester with all v1.4.1 components."""

    def __init__(self, config: Dict):
        self.config = config
        self.fusion_engine = FusionEngineV141(config)
        self.wyckoff_machine = WyckoffStateMachine()
        self.exit_evaluator = AdvancedExitEvaluator(config.get('exits_config_path'))

        self.trades = []
        self.current_position = None
        self.balance = config['backtest']['initial_balance']
        self.peak_balance = self.balance

    def compute_layer_scores(self, df: pd.DataFrame, idx: int) -> Dict:
        """Compute all layer scores with enhanced logic."""
        current_data = df.iloc[:idx+1]

        if len(current_data) < 20:
            return {layer: 0.5 for layer in self.fusion_engine.weights.keys()}

        scores = {}

        # 1. Wyckoff with enhanced state machine
        wyckoff_result = self.wyckoff_machine.analyze_wyckoff_state(current_data)
        scores['wyckoff'] = max(0.1, min(0.9, wyckoff_result['confidence'] *
                                        (1 if wyckoff_result['bias'] != 'neutral' else 0.5)))

        # 2. Liquidity with clustering
        liquidity_result = calculate_liquidity_score(current_data)
        scores['liquidity'] = liquidity_result['score']

        # 3. Structure (simplified)
        recent = current_data.tail(20)
        structure_score = 0.5
        if len(recent) > 5:
            # Basic trend analysis
            if recent['close'].iloc[-1] > recent['close'].iloc[0]:
                structure_score = 0.65  # Uptrend
            elif recent['close'].iloc[-1] < recent['close'].iloc[0]:
                structure_score = 0.35  # Downtrend
            else:
                structure_score = 0.5   # Sideways

        scores['structure'] = structure_score

        # 4. Momentum (RSI-like)
        if len(current_data) >= 14:
            returns = current_data['close'].pct_change().tail(14)
            gains = returns.where(returns > 0, 0).sum()
            losses = -returns.where(returns < 0, 0).sum()
            rs = gains / (losses + 1e-10)
            rsi = 100 - (100 / (1 + rs))

            # Convert RSI to 0-1 score
            if rsi > 70:
                scores['momentum'] = 0.8  # Overbought but strong
            elif rsi < 30:
                scores['momentum'] = 0.2  # Oversold but weak
            else:
                scores['momentum'] = 0.5 + (rsi - 50) / 100
        else:
            scores['momentum'] = 0.5

        # 5. Volume
        vol_sma = current_data['volume'].tail(10).mean()
        current_vol = current_data.iloc[-1]['volume']
        vol_ratio = current_vol / vol_sma if vol_sma > 0 else 1.0
        scores['volume'] = min(0.9, max(0.1, 0.3 + vol_ratio * 0.4))

        # 6. Context (macro/sentiment proxy)
        volatility = current_data['close'].pct_change().tail(20).std()
        if volatility > 0.05:  # High vol = stress
            scores['context'] = 0.3
        elif volatility < 0.02:  # Low vol = complacency
            scores['context'] = 0.7
        else:
            scores['context'] = 0.5

        # 7. MTF (simplified)
        if len(current_data) > 50:
            ltf_trend = current_data['close'].tail(10).mean()
            mtf_trend = current_data['close'].tail(50).mean()

            if abs(ltf_trend - mtf_trend) / mtf_trend < 0.02:  # Aligned
                scores['mtf'] = 0.75
            else:
                scores['mtf'] = 0.45  # Misaligned
        else:
            scores['mtf'] = 0.5

        # 8. Bojan (if enabled)
        if self.config.get('features', {}).get('bojan'):
            current_price = current_data.iloc[-1]['close']
            bojan_score = wick_magnets(current_data, current_price * 1.02)  # Capped at 0.6
            scores['bojan'] = bojan_score

        return scores, wyckoff_result, liquidity_result

    def run_backtest(self, df: pd.DataFrame, symbol: str) -> Dict:
        """Run comprehensive backtest with enhanced logic."""
        print(f"\n🚀 Running Enhanced Backtest for {symbol}")
        print(f"   Data: {len(df)} bars, {df.index[0]} to {df.index[-1]}")

        self.trades = []
        self.current_position = None
        self.balance = self.config['backtest']['initial_balance']
        self.peak_balance = self.balance

        for i in range(50, len(df)):  # Start after warmup period
            current_bar = df.iloc[i]

            # Compute layer scores
            scores, wyckoff_context, liquidity_data = self.compute_layer_scores(df, i)

            # Fuse scores with enhanced logic
            fusion_result = self.fusion_engine.fuse_scores(
                scores,
                self.config.get('quality_floors'),
                wyckoff_context
            )

            # Position management
            if self.current_position is None:
                # Entry logic
                if self.fusion_engine.should_enter(fusion_result):
                    self.enter_position(df, i, fusion_result, scores, liquidity_data, symbol)
            else:
                # Exit logic
                self.manage_position(df, i, scores, wyckoff_context, liquidity_data)

            # Update balance tracking
            if self.current_position:
                current_pnl = self.calculate_unrealized_pnl(current_bar['close'])
                current_balance = self.balance + current_pnl
                self.peak_balance = max(self.peak_balance, current_balance)

        # Close any remaining position
        if self.current_position:
            self.close_position(df, len(df)-1, "end_of_data")

        # Calculate performance metrics
        performance = self.calculate_performance()

        print(f"   Results: {len(self.trades)} trades, {performance['win_rate']:.1f}% win rate, "
              f"{performance['total_pnl_pct']:.1f}% PnL")

        return {
            'symbol': symbol,
            'performance': performance,
            'trades': self.trades,
            'config_used': self.config
        }

    def enter_position(self, df: pd.DataFrame, idx: int, fusion_result: Dict,
                      scores: Dict, liquidity_data: Dict, symbol: str):
        """Enter position with dynamic risk sizing."""

        current_bar = df.iloc[idx]
        entry_price = current_bar['close']

        # Determine bias from Wyckoff
        wyckoff_score = scores.get('wyckoff', 0.5)
        bias = 'long' if wyckoff_score > 0.5 else 'short'

        # Dynamic position sizing
        base_risk = self.config['risk_management']['max_risk_per_trade']
        risk_data = calculate_dynamic_position_size(
            base_risk, df.iloc[:idx+1], liquidity_data
        )

        # Calculate position size
        risk_amount = self.balance * risk_data['adjusted_risk_pct']

        # Stop loss (2% for longs, 2% for shorts)
        if bias == 'long':
            stop_loss = entry_price * 0.98
        else:
            stop_loss = entry_price * 1.02

        position_size = risk_amount / abs(entry_price - stop_loss)

        # Create position
        self.current_position = {
            'symbol': symbol,
            'side': bias,
            'entry_price': entry_price,
            'entry_time': df.index[idx],
            'entry_idx': idx,
            'size': position_size,
            'stop_loss': stop_loss,
            'entry_scores': scores.copy(),
            'fusion_score': fusion_result['weighted_score'],
            'risk_data': risk_data
        }

        logging.info(f"ENTER {bias.upper()}: {symbol} @ {entry_price:.2f}, "
                    f"size={position_size:.4f}, SL={stop_loss:.2f}")

    def manage_position(self, df: pd.DataFrame, idx: int, scores: Dict,
                       wyckoff_context: Dict, liquidity_data: Dict):
        """Manage existing position with advanced exits."""

        current_bar = df.iloc[idx]
        bars_held = idx - self.current_position['entry_idx']

        # Create trade plan for exit evaluator
        trade_plan = {
            'bias': self.current_position['side'],
            'entry_price': self.current_position['entry_price'],
            'sl': self.current_position['stop_loss'],
            'tp': self.current_position['entry_price'] * (1.06 if self.current_position['side'] == 'long' else 0.94)
        }

        # Check exits
        updated_plan = self.exit_evaluator.evaluate_exits(
            df.iloc[self.current_position['entry_idx']:idx+1],
            trade_plan, scores, bars_held, {'wyckoff_context': wyckoff_context}
        )

        # Process exit signals
        exits = updated_plan.get('exits', {})

        if exits.get('full_exit'):
            self.close_position(df, idx, "advanced_exit_full")
        elif exits.get('partial_exit_pct'):
            partial_pct = exits['partial_exit_pct']
            self.partial_close_position(df, idx, partial_pct, "advanced_exit_partial")

        # Check stop loss
        elif self.check_stop_loss(current_bar):
            self.close_position(df, idx, "stop_loss")

        # Check basic profit target (3R)
        elif self.check_profit_target(current_bar, 3.0):
            self.close_position(df, idx, "profit_target")

    def close_position(self, df: pd.DataFrame, idx: int, reason: str):
        """Close entire position."""
        if not self.current_position:
            return

        current_bar = df.iloc[idx]
        exit_price = current_bar['close']

        # Calculate PnL
        if self.current_position['side'] == 'long':
            pnl_pct = (exit_price - self.current_position['entry_price']) / self.current_position['entry_price']
        else:
            pnl_pct = (self.current_position['entry_price'] - exit_price) / self.current_position['entry_price']

        pnl_dollars = pnl_pct * self.balance * self.current_position['risk_data']['adjusted_risk_pct']

        # Record trade
        trade = {
            'symbol': self.current_position['symbol'],
            'side': self.current_position['side'],
            'entry_price': self.current_position['entry_price'],
            'exit_price': exit_price,
            'entry_time': self.current_position['entry_time'],
            'exit_time': df.index[idx],
            'bars_held': idx - self.current_position['entry_idx'],
            'pnl_pct': pnl_pct,
            'pnl_dollars': pnl_dollars,
            'exit_reason': reason,
            'fusion_score': self.current_position['fusion_score'],
            'entry_scores': self.current_position['entry_scores'],
            'risk_multiplier': self.current_position['risk_data']['risk_multiplier']
        }

        self.trades.append(trade)
        self.balance += pnl_dollars
        self.current_position = None

        logging.info(f"CLOSE: {trade['side'].upper()} @ {exit_price:.2f}, "
                    f"PnL: {pnl_pct:.2%} ({pnl_dollars:.2f}), {reason}")

    def partial_close_position(self, df: pd.DataFrame, idx: int, close_pct: float, reason: str):
        """Partially close position."""
        # Simplified: treat as full close for now
        self.close_position(df, idx, f"{reason}_partial_{close_pct:.0%}")

    def check_stop_loss(self, current_bar: pd.Series) -> bool:
        """Check if stop loss hit."""
        if not self.current_position:
            return False

        current_price = current_bar['close']
        stop_loss = self.current_position['stop_loss']

        if self.current_position['side'] == 'long':
            return current_price <= stop_loss
        else:
            return current_price >= stop_loss

    def check_profit_target(self, current_bar: pd.Series, r_multiple: float) -> bool:
        """Check if profit target hit."""
        if not self.current_position:
            return False

        current_price = current_bar['close']
        entry_price = self.current_position['entry_price']
        stop_loss = self.current_position['stop_loss']

        risk_per_share = abs(entry_price - stop_loss)

        if self.current_position['side'] == 'long':
            target_price = entry_price + (risk_per_share * r_multiple)
            return current_price >= target_price
        else:
            target_price = entry_price - (risk_per_share * r_multiple)
            return current_price <= target_price

    def calculate_unrealized_pnl(self, current_price: float) -> float:
        """Calculate unrealized PnL."""
        if not self.current_position:
            return 0

        entry_price = self.current_position['entry_price']

        if self.current_position['side'] == 'long':
            pnl_pct = (current_price - entry_price) / entry_price
        else:
            pnl_pct = (entry_price - current_price) / entry_price

        return pnl_pct * self.balance * self.current_position['risk_data']['adjusted_risk_pct']

    def calculate_performance(self) -> Dict:
        """Calculate comprehensive performance metrics."""
        if not self.trades:
            return {
                'total_trades': 0,
                'win_rate': 0,
                'total_pnl_pct': 0,
                'total_pnl_dollars': 0,
                'avg_pnl_pct': 0,
                'max_drawdown': 0,
                'sharpe_ratio': 0,
                'profit_factor': 0
            }

        # Basic metrics
        total_trades = len(self.trades)
        wins = [t for t in self.trades if t['pnl_pct'] > 0]
        win_rate = len(wins) / total_trades

        total_pnl_dollars = sum(t['pnl_dollars'] for t in self.trades)
        total_pnl_pct = (self.balance - self.config['backtest']['initial_balance']) / self.config['backtest']['initial_balance']

        avg_pnl_pct = np.mean([t['pnl_pct'] for t in self.trades])

        # Max drawdown
        max_drawdown = (self.peak_balance - self.balance) / self.peak_balance if self.peak_balance > 0 else 0

        # Sharpe ratio (simplified)
        returns = [t['pnl_pct'] for t in self.trades]
        if len(returns) > 1:
            sharpe_ratio = np.mean(returns) / np.std(returns) * np.sqrt(250)
        else:
            sharpe_ratio = 0

        # Profit factor
        gross_profit = sum(t['pnl_dollars'] for t in self.trades if t['pnl_dollars'] > 0)
        gross_loss = abs(sum(t['pnl_dollars'] for t in self.trades if t['pnl_dollars'] < 0))
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')

        return {
            'total_trades': total_trades,
            'win_rate': win_rate * 100,
            'total_pnl_pct': total_pnl_pct * 100,
            'total_pnl_dollars': total_pnl_dollars,
            'avg_pnl_pct': avg_pnl_pct * 100,
            'max_drawdown': max_drawdown * 100,
            'sharpe_ratio': sharpe_ratio,
            'profit_factor': profit_factor,
            'final_balance': self.balance
        }


def main():
    """Run comprehensive BTC/ETH backtests."""

    # Load enhanced config
    config = {
        'features': {
            'wyckoff': True,
            'liquidity': True,
            'structure': True,
            'momentum': True,
            'volume': True,
            'context': True,
            'mtf': True,
            'bojan': True,
            'dynamic_risk': True
        },
        'weights': {
            'wyckoff': 0.30,
            'liquidity': 0.25,
            'structure': 0.15,
            'momentum': 0.15,
            'volume': 0.15,
            'context': 0.05,
            'mtf': 0.10
        },
        'signals': {
            'enter_threshold': 0.72,
            'aggregate_floor': 0.35,
            'context_floor': 0.30,
            'mtf_threshold': 0.60
        },
        'quality_floors': {
            'wyckoff': 0.40,
            'liquidity': 0.35,
            'structure': 0.30,
            'momentum': 0.30,
            'volume': 0.30,
            'context': 0.25,
            'mtf': 0.45
        },
        'risk_management': {
            'max_risk_per_trade': 0.01,  # 1%
            'max_positions': 1,
            'drawdown_limit': 0.20
        },
        'backtest': {
            'initial_balance': 10000,
            'commission': 0.0002,
            'slippage': 0.0003
        },
        'exits_config_path': 'configs/v141/exits_config.json'
    }

    logging.basicConfig(level=logging.INFO)

    # Run backtests
    results = {}

    symbols = ['BTC', 'ETH']
    for symbol in symbols:
        print(f"\n{'='*50}")
        print(f"BULL MACHINE v1.4.1 - {symbol} BACKTEST")
        print(f"{'='*50}")

        # Create realistic data
        df = create_realistic_data(symbol, 800)

        # Run backtest
        backtester = ComprehensiveBacktester(config)
        symbol_results = backtester.run_backtest(df, symbol)
        results[symbol] = symbol_results

    # Generate summary report
    print(f"\n{'='*60}")
    print("COMPREHENSIVE RESULTS SUMMARY")
    print(f"{'='*60}")

    for symbol, result in results.items():
        perf = result['performance']
        print(f"\n🪙 {symbol} Results:")
        print(f"   Trades: {perf['total_trades']}")
        print(f"   Win Rate: {perf['win_rate']:.1f}%")
        print(f"   Total PnL: {perf['total_pnl_pct']:.1f}% (${perf['total_pnl_dollars']:.2f})")
        print(f"   Avg Trade: {perf['avg_pnl_pct']:.2f}%")
        print(f"   Max DD: {perf['max_drawdown']:.1f}%")
        print(f"   Sharpe: {perf['sharpe_ratio']:.2f}")
        print(f"   Profit Factor: {perf['profit_factor']:.2f}")
        print(f"   Final Balance: ${perf['final_balance']:.2f}")

    # Combined metrics
    total_trades = sum(r['performance']['total_trades'] for r in results.values())
    avg_win_rate = np.mean([r['performance']['win_rate'] for r in results.values()])
    total_pnl = sum(r['performance']['total_pnl_dollars'] for r in results.values())

    print(f"\n🏆 Combined Results:")
    print(f"   Total Trades: {total_trades}")
    print(f"   Average Win Rate: {avg_win_rate:.1f}%")
    print(f"   Combined PnL: ${total_pnl:.2f}")

    # Save results
    output_dir = Path('reports/v141_comprehensive')
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(output_dir / 'backtest_results.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\n📊 Detailed results saved to {output_dir}/backtest_results.json")

    return results


if __name__ == "__main__":
    results = main()