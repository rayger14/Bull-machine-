"""
P&L Tracking Module for v1.8 Hybrid Runner

Simplified backtest P&L calculator for evaluating signal performance.
Tracks hypothetical trades with structure-based stops and targets.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from datetime import datetime


class Position:
    """Represents a single open position."""
    
    def __init__(self, signal: Dict, entry_price: float, df_1h: pd.DataFrame, config: Dict):
        self.entry_time = signal['timestamp'] if isinstance(signal['timestamp'], datetime) else datetime.fromisoformat(signal['timestamp'])
        self.entry_price = entry_price
        self.side = signal['side']
        self.confidence = signal['confidence']

        # Calculate stop loss (structure-based)
        self.stop_loss = self._calculate_structure_stop(df_1h, config)

        # Calculate take profit (risk multiple - 2:1 for crypto)
        risk = abs(entry_price - self.stop_loss)
        tp_r = config.get('exits', {}).get('tp1_r', 2.0)  # 2:1 R:R for crypto trends
        if self.side == 'long':
            self.take_profit = entry_price + (risk * tp_r)
        else:
            self.take_profit = entry_price - (risk * tp_r)

        # Position sizing
        self.risk_pct = config.get('risk', {}).get('base_risk_pct', 0.075)

        # Tracking
        self.exit_time = None
        self.exit_price = None
        self.exit_reason = None  # 'stop', 'target', 'signal_flip'
        self.pnl = 0.0
        self.pnl_pct = 0.0
        self.r_multiple = 0.0
        self.entry_bar_processed = False  # Flag to skip exit check on entry bar
    
    def _calculate_structure_stop(self, df_1h: pd.DataFrame, config: Dict) -> float:
        """Calculate structure-based stop using swing highs/lows + ATR."""
        lookback = 20
        atr_k = config.get('exits', {}).get('atr_k', 2.0)  # Wider stops for crypto (2x vs 1x)

        # Calculate ATR using True Range + EMA (proper method)
        high = df_1h['High']
        low = df_1h['Low']
        close = df_1h['Close']

        # True Range: max of 3 calculations
        tr = pd.concat([
            high - low,
            (high - close.shift()).abs(),
            (low - close.shift()).abs()
        ], axis=1).max(axis=1)

        # Use EMA for smoother response to volatility changes (better for crypto)
        atr = tr.ewm(span=14, adjust=False).mean().iloc[-1]

        if self.side == 'long':
            swing_low = df_1h['Low'].tail(lookback).min()
            stop = swing_low - (atr * atr_k)
        else:  # short
            swing_high = df_1h['High'].tail(lookback).max()
            stop = swing_high + (atr * atr_k)

        return stop
    
    def check_exit(self, high: float, low: float, current_time: datetime) -> Optional[str]:
        """Check if position hit stop or target."""
        # Skip exit check on the entry bar
        if not self.entry_bar_processed:
            self.entry_bar_processed = True
            return None

        if self.side == 'long':
            if low <= self.stop_loss:
                self.close(self.stop_loss, current_time, 'stop')
                return 'stop'
            if high >= self.take_profit:
                self.close(self.take_profit, current_time, 'target')
                return 'target'
        else:  # short
            if high >= self.stop_loss:
                self.close(self.stop_loss, current_time, 'stop')
                return 'stop'
            if low <= self.take_profit:
                self.close(self.take_profit, current_time, 'target')
                return 'target'

        return None
    
    def close(self, exit_price: float, exit_time: datetime, exit_reason: str):
        """Close position and calculate P&L."""
        self.exit_price = exit_price
        self.exit_time = exit_time
        self.exit_reason = exit_reason
        
        # Calculate P&L percentage
        if self.side == 'long':
            self.pnl_pct = (exit_price - self.entry_price) / self.entry_price * 100
        else:
            self.pnl_pct = (self.entry_price - exit_price) / self.entry_price * 100
        
        # Calculate R multiple
        risk = abs(self.entry_price - self.stop_loss)
        actual_move = abs(exit_price - self.entry_price)
        if self.pnl_pct > 0:
            self.r_multiple = actual_move / risk
        else:
            self.r_multiple = -actual_move / risk


class Portfolio:
    """Manages portfolio and tracks performance."""
    
    def __init__(self, starting_balance: float = 10000):
        self.starting_balance = starting_balance
        self.balance = starting_balance
        self.open_positions: List[Position] = []
        self.closed_trades: List[Position] = []
        self.peak_equity = starting_balance
        self.max_drawdown = 0.0
    
    def open_position(self, signal: Dict, current_price: float, df_1h: pd.DataFrame, config: Dict):
        """Open new position from signal."""
        position = Position(signal, current_price, df_1h, config)
        self.open_positions.append(position)
        # print(f"   📈 Opened {position.side} @ ${current_price:.2f} | SL: ${position.stop_loss:.2f} | TP: ${position.take_profit:.2f}")
    
    def update_positions(self, current_time: datetime, high: float, low: float, close: float):
        """Check stops/targets on all open positions."""
        closed_this_bar = []

        for position in self.open_positions:
            exit_reason = position.check_exit(high, low, current_time)

            if exit_reason:
                self._close_position(position)
                closed_this_bar.append(position)
                # print(f"   ❌ Closed {position.side} @ ${position.exit_price:.2f} ({exit_reason}) | P&L: ${position.pnl:+.2f}")

        # Remove closed positions
        for position in closed_this_bar:
            self.open_positions.remove(position)
    
    def _close_position(self, position: Position):
        """Process closed position and update balance."""
        # Calculate dollar P&L using risk-based position sizing
        # Risk per trade: 2% of equity (standard crypto practice)
        # Leverage: 5x (reduces margin, not risk)
        risk_per_trade = 0.02  # 2% risk
        leverage = 5.0

        # Calculate position size based on stop distance
        stop_distance_pct = abs(position.entry_price - position.stop_loss) / position.entry_price

        # Notional size based on risk
        risk_dollars = self.balance * risk_per_trade
        notional_size = risk_dollars / stop_distance_pct if stop_distance_pct > 0 else 0

        # Apply leverage to reduce margin (not increase risk)
        margin_used = notional_size / leverage

        # Calculate actual dollar P&L
        dollar_pnl = notional_size * (position.pnl_pct / 100)

        position.pnl = dollar_pnl
        position.notional_size = notional_size
        position.margin_used = margin_used
        self.balance += dollar_pnl
        self.closed_trades.append(position)

        # Debug: print P&L calculation details
        # print(f"      Debug: entry=${position.entry_price:.2f}, exit=${position.exit_price:.2f}, pnl_pct={position.pnl_pct:.4f}%, position_size=${position_size_dollars:.2f}, dollar_pnl=${dollar_pnl:.4f}")
        
        # Update peak and drawdown
        if self.balance > self.peak_equity:
            self.peak_equity = self.balance
        
        current_dd = (self.peak_equity - self.balance) / self.peak_equity * 100
        if current_dd > self.max_drawdown:
            self.max_drawdown = current_dd
    
    def calculate_metrics(self) -> Dict:
        """Calculate final performance metrics."""
        if not self.closed_trades:
            return {
                'total_trades': 0,
                'profit_factor': 0.0,
                'win_rate': 0.0,
                'total_return': 0.0
            }
        
        # Use small threshold for breakeven classification (< $0.10)
        wins = [t for t in self.closed_trades if t.pnl > 0.10]
        losses = [t for t in self.closed_trades if t.pnl < -0.10]
        breakeven = [t for t in self.closed_trades if -0.10 <= t.pnl <= 0.10]
        
        gross_profit = sum(t.pnl for t in wins) if wins else 0
        gross_loss = abs(sum(t.pnl for t in losses)) if losses else 0
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else (float('inf') if gross_profit > 0 else 0)
        
        win_rate = len(wins) / len(self.closed_trades) * 100
        total_return = (self.balance - self.starting_balance) / self.starting_balance * 100
        
        return {
            'total_trades': len(self.closed_trades),
            'wins': len(wins),
            'losses': len(losses),
            'breakeven': len(breakeven),
            'profit_factor': profit_factor,
            'win_rate': win_rate,
            'max_drawdown': self.max_drawdown,
            'total_return': total_return,
            'ending_balance': self.balance,
            'r_multiples': [t.r_multiple for t in self.closed_trades]
        }
    
    def print_summary(self):
        """Print performance summary."""
        metrics = self.calculate_metrics()
        
        print("\n" + "=" * 70)
        print("📊 PERFORMANCE SUMMARY")
        print("=" * 70)
        
        print(f"\n💰 Account:")
        print(f"  Starting: ${self.starting_balance:,.2f}")
        print(f"  Ending:   ${metrics['ending_balance']:,.2f}")
        print(f"  Return:   {metrics['total_return']:+.2f}%")
        print(f"  Max DD:   {metrics['max_drawdown']:.2f}%")
        
        if metrics['total_trades'] > 0:
            print(f"\n📈 Trades:")
            print(f"  Total:    {metrics['total_trades']}")
            print(f"  Wins:     {metrics['wins']} ({metrics['win_rate']:.1f}%)")
            print(f"  Losses:   {metrics['losses']}")
            print(f"  Breakeven: {metrics.get('breakeven', 0)}")

            pf_str = f"{metrics['profit_factor']:.2f}" if metrics['profit_factor'] != float('inf') else "∞"
            print(f"\n💵 Performance:")
            print(f"  PF:       {pf_str}")

            r_mults = metrics['r_multiples']
            if len(r_mults) > 0:
                print(f"  Avg R:    {np.mean(r_mults):+.2f}R")
        
        print("=" * 70)
