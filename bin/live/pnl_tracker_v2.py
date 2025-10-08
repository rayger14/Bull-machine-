"""
P&L Tracking Module v2 - Risk-Based Sizing with Leverage

Implements proper crypto trading position sizing:
- 2% risk per trade (fixed dollar risk)
- 5x leverage (reduces margin, not risk)
- ATR-based stops (2x multiplier for crypto volatility)
- 2:1 R:R targets (capture crypto trends)
- Fees + slippage modeling (realistic fills)
"""

import json
import pandas as pd
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional
from datetime import datetime
from pathlib import Path


def atr(df: pd.DataFrame, period: int = 14, method: str = "ema") -> float:
    """Calculate ATR using True Range + EMA smoothing."""
    high = df['High'] if 'High' in df.columns else df['high']
    low = df['Low'] if 'Low' in df.columns else df['low']
    close = df['Close'] if 'Close' in df.columns else df['close']

    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs()
    ], axis=1).max(axis=1)

    if method == "ema":
        return tr.ewm(span=period, adjust=False).mean().iloc[-1]
    return tr.rolling(period).mean().iloc[-1]


@dataclass
class Position:
    """Represents a single leveraged position."""
    asset: str
    side: str
    entry_price: float
    qty: float
    stop_price: float
    tp_price: float
    leverage: float
    entry_time: datetime
    fees_bps: float = 10.0
    slippage_bps: float = 5.0

    # Tracking fields
    exit_price: Optional[float] = None
    exit_time: Optional[datetime] = None
    exit_reason: Optional[str] = None
    pnl: float = 0.0
    pnl_pct: float = 0.0
    r_multiple: float = 0.0
    notional_size: float = field(init=False)
    margin_used: float = field(init=False)

    def __post_init__(self):
        self.notional_size = self.entry_price * self.qty
        self.margin_used = self.notional_size / self.leverage

    def exit_pnl(self, exit_price: float) -> float:
        """Calculate P&L including fees and slippage."""
        sign = 1 if self.side == "long" else -1
        gross_pnl = sign * (exit_price - self.entry_price) * self.qty

        # Calculate costs (fees + slippage on entry and exit)
        total_cost = self.notional_size * (self.fees_bps + self.slippage_bps) / 10000
        exit_notional = exit_price * self.qty
        total_cost += exit_notional * (self.fees_bps + self.slippage_bps) / 10000

        return gross_pnl - total_cost

    def close(self, exit_price: float, exit_time: datetime, exit_reason: str):
        """Close position and calculate final P&L."""
        self.exit_price = exit_price
        self.exit_time = exit_time
        self.exit_reason = exit_reason

        # Calculate dollar P&L
        self.pnl = self.exit_pnl(exit_price)

        # Calculate percentage P&L
        self.pnl_pct = (self.pnl / self.margin_used) * 100

        # Calculate R-multiple
        risk_distance = abs(self.entry_price - self.stop_price)
        actual_distance = abs(exit_price - self.entry_price)
        if self.pnl > 0:
            self.r_multiple = actual_distance / risk_distance if risk_distance > 0 else 0
        else:
            self.r_multiple = -actual_distance / risk_distance if risk_distance > 0 else 0


class Portfolio:
    """Manages portfolio with leverage and risk-based sizing."""

    def __init__(self, initial_balance: float, config: dict):
        self.initial_balance = initial_balance
        self.balance = initial_balance
        self.equity_peak = initial_balance
        self.max_dd = 0.0
        self.positions: Dict[str, Position] = {}
        self.closed_trades: List[Position] = []

        # Load config with validation
        self.cfg = config.get("pnl_tracker", {})
        self.risk_pct = max(0.001, min(self.cfg.get("risk_per_trade", 0.02), 0.1))
        self.atr_period = max(5, int(self.cfg.get("atr_period", 20)))
        self.stop_mult = max(0.5, self.cfg.get("stop_buffer_multiplier", 2.0))
        self.r_mult_target = max(1.0, self.cfg.get("r_multiple_target", 2.0))
        self.leverage = max(1.0, min(self.cfg.get("leverage", 5.0), 20.0))
        self.max_margin_util = max(0.1, min(self.cfg.get("max_margin_util", 0.5), 1.0))
        self.fees_bps = self.cfg.get("fees_bps", 10.0)
        self.slippage_bps = self.cfg.get("slippage_bps", 5.0)
        self.funding_rate_bps = self.cfg.get("funding_rate_bps", 0.0)

        # Ensure results directory exists
        Path('results').mkdir(exist_ok=True)

    def _compute_qty_and_levels(self, side: str, entry: float, df_1h: pd.DataFrame) -> tuple:
        """Calculate position size and stop/target levels."""
        # Calculate ATR
        atr_value = atr(df_1h, period=self.atr_period, method="ema")

        # Calculate stop distance
        stop_distance = self.stop_mult * atr_value
        stop_price = entry - stop_distance if side == "long" else entry + stop_distance

        # Calculate stop as percentage of entry
        stop_pct = abs(entry - stop_price) / entry

        # Risk-based position sizing
        risk_dollars = self.balance * self.risk_pct
        notional = risk_dollars / stop_pct

        # Apply leverage to determine margin requirement
        margin_needed = notional / self.leverage
        max_margin = self.balance * self.max_margin_util

        # Scale down if margin exceeds limit
        if margin_needed > max_margin:
            scale = max_margin / margin_needed
            notional *= scale

        # Calculate quantity
        qty = notional / entry

        # Calculate take profit
        tp_price = entry + (self.r_mult_target * stop_distance) if side == "long" else entry - (self.r_mult_target * stop_distance)

        return qty, stop_price, tp_price

    def open_position(self, asset: str, side: str, entry_price: float, df_1h: pd.DataFrame, timestamp: datetime) -> bool:
        """Open new position with risk-based sizing."""
        if asset in self.positions:
            return False

        qty, stop_price, tp_price = self._compute_qty_and_levels(side, entry_price, df_1h)

        position = Position(
            asset=asset,
            side=side,
            entry_price=entry_price,
            qty=qty,
            stop_price=stop_price,
            tp_price=tp_price,
            leverage=self.leverage,
            entry_time=timestamp,
            fees_bps=self.fees_bps,
            slippage_bps=self.slippage_bps
        )

        self.positions[asset] = position

        # Log position details
        with open('results/position_log.jsonl', 'a') as f:
            f.write(json.dumps({
                "asset": asset,
                "side": side,
                "entry_price": entry_price,
                "qty": qty,
                "notional": position.notional_size,
                "margin": position.margin_used,
                "stop_price": stop_price,
                "tp_price": tp_price,
                "leverage": self.leverage,
                "timestamp": str(timestamp)
            }) + '\n')

        return True

    def update_positions(self, asset: str, current_time: datetime, high: float, low: float, close: float):
        """Check if position hit stop or target."""
        if asset not in self.positions:
            return

        position = self.positions[asset]

        # Check for stop loss hit
        hit_sl = low <= position.stop_price if position.side == "long" else high >= position.stop_price

        # Check for take profit hit
        hit_tp = high >= position.tp_price if position.side == "long" else low <= position.tp_price

        if hit_sl:
            exit_price = position.stop_price
            position.close(exit_price, current_time, "SL")
            self._finalize_position(position)
            del self.positions[asset]
        elif hit_tp:
            exit_price = position.tp_price
            position.close(exit_price, current_time, "TP")
            self._finalize_position(position)
            del self.positions[asset]

    def force_close_position(self, asset: str, exit_price: float, exit_time: datetime):
        """Force close position at market (e.g., end of backtest)."""
        if asset not in self.positions:
            return

        position = self.positions[asset]
        position.close(exit_price, exit_time, "end_of_backtest")
        self._finalize_position(position)
        del self.positions[asset]

    def _finalize_position(self, position: Position):
        """Update balance and tracking after position close."""
        self.balance += position.pnl
        self.closed_trades.append(position)

        # Update peak equity and drawdown
        self.equity_peak = max(self.equity_peak, self.balance)
        current_dd = (self.equity_peak - self.balance) / self.equity_peak
        self.max_dd = max(self.max_dd, current_dd)

    def calculate_metrics(self) -> Dict:
        """Calculate final performance metrics."""
        if not self.closed_trades:
            return {
                'total_trades': 0,
                'profit_factor': 0.0,
                'win_rate': 0.0,
                'total_return': 0.0
            }

        wins = [t for t in self.closed_trades if t.pnl > 0.10]
        losses = [t for t in self.closed_trades if t.pnl < -0.10]
        breakeven = [t for t in self.closed_trades if -0.10 <= t.pnl <= 0.10]

        gross_profit = sum(t.pnl for t in wins) if wins else 0
        gross_loss = abs(sum(t.pnl for t in losses)) if losses else 0
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else (float('inf') if gross_profit > 0 else 0)

        win_rate = len(wins) / len(self.closed_trades) * 100
        total_return = (self.balance - self.initial_balance) / self.initial_balance * 100

        return {
            'total_trades': len(self.closed_trades),
            'wins': len(wins),
            'losses': len(losses),
            'breakeven': len(breakeven),
            'profit_factor': profit_factor,
            'win_rate': win_rate,
            'max_drawdown': self.max_dd * 100,
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
        print(f"  Starting: ${self.initial_balance:,.2f}")
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
