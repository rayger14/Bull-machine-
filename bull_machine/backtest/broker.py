
from dataclasses import dataclass
from typing import Dict, Any, Optional

@dataclass
class TPLevel:
    price: float
    size_pct: int      # percentage of position
    r_multiple: float  # R multiple (1R, 2R, 3R)
    filled: bool = False

@dataclass
class Position:
    side: str          # 'long'|'short'
    size: float
    entry: float
    stop: Optional[float] = None
    tp_levels: Optional[list[TPLevel]] = None
    be_moved: bool = False      # breakeven moved flag
    trail_active: bool = False  # trailing stop active

class PaperBroker:
    def __init__(self, fee_bps=2, slippage_bps=3, spread_bps=1, partial_fill=True):
        self.fee_bps = fee_bps
        self.slippage_bps = slippage_bps
        self.spread_bps = spread_bps
        self.partial_fill = partial_fill
        self.positions: Dict[str, Position] = {}
        self.realized_pnl = 0.0

    def _apply_costs(self, price: float, side: str) -> float:
        adj = (self.slippage_bps + self.spread_bps) * 1e-4 * price
        return price + adj if side == 'long' else price - adj

    def submit(self, ts, symbol, side, size, price_hint=None, order_type='market', risk_plan=None) -> Dict[str, Any]:
        px = self._apply_costs(price_hint or 0.0, side)
        fee = (self.fee_bps * 1e-4) * px * abs(size)

        # Close existing position if reversing
        prev = self.positions.get(symbol)
        if prev and ((prev.side=='long' and side=='short') or (prev.side=='short' and side=='long')):
            pnl = (px - prev.entry) * prev.size if prev.side=='long' else (prev.entry - px) * prev.size
            self.realized_pnl += pnl - fee
            self.positions.pop(symbol, None)

        # Create TP levels from risk plan
        tp_levels = None
        if risk_plan and risk_plan.get('tp_levels'):
            tp_levels = [
                TPLevel(
                    price=tp['price'],
                    size_pct=tp.get('pct', 33),
                    r_multiple=tp.get('r', 1.0)
                ) for tp in risk_plan['tp_levels']
            ]

        # Create new position with risk management
        self.positions[symbol] = Position(
            side=side,
            size=size,
            entry=px,
            stop=risk_plan.get('stop') if risk_plan else None,
            tp_levels=tp_levels
        )

        return {"ts": ts, "symbol": symbol, "side": side, "price": px, "size_filled": size, "fee": fee, "slippage": 0.0}

    def close(self, ts, symbol, price=None) -> Dict[str, Any]:
        """Manually close position"""
        pos = self.positions.get(symbol)
        if not pos:
            return {"closed": False}

        # Use provided price or position entry as fallback
        close_price = price or pos.entry
        fill_price = self._apply_costs(close_price, 'short' if pos.side == 'long' else 'long')
        fee = (self.fee_bps * 1e-4) * fill_price * abs(pos.size)

        pnl = ((fill_price - pos.entry) * pos.size if pos.side == 'long'
               else (pos.entry - fill_price) * pos.size)
        self.realized_pnl += pnl - fee

        self.positions.pop(symbol, None)
        return {
            "ts": ts, "symbol": symbol, "side": "manual_exit",
            "price": fill_price, "size_filled": pos.size,
            "fee": fee, "pnl": pnl, "reason": "manual_close"
        }

    def mark(self, ts, symbol, price):
        """Mark position to market and check for stop/TP fills"""
        pos = self.positions.get(symbol)
        if not pos:
            return None

        fills = []

        # Check stop loss
        if pos.stop and self._should_stop_fill(pos, price):
            fill = self._execute_stop(ts, symbol, pos, price)
            fills.append(fill)
            self.positions.pop(symbol, None)  # Position closed
            return fills

        # Check TP levels
        if pos.tp_levels:
            for tp in pos.tp_levels:
                if not tp.filled and self._should_tp_fill(pos, tp, price):
                    fill = self._execute_tp(ts, symbol, pos, tp, price)
                    fills.append(fill)

                    # Move to breakeven after first TP
                    if tp.r_multiple >= 1.0 and not pos.be_moved:
                        pos.stop = pos.entry
                        pos.be_moved = True

        return fills if fills else None

    def _should_stop_fill(self, pos: Position, price: float) -> bool:
        """Check if stop should be triggered"""
        if pos.side == 'long':
            return price <= pos.stop
        else:
            return price >= pos.stop

    def _should_tp_fill(self, pos: Position, tp: TPLevel, price: float) -> bool:
        """Check if TP level should be triggered"""
        if pos.side == 'long':
            return price >= tp.price
        else:
            return price <= tp.price

    def _execute_stop(self, ts, symbol, pos: Position, price: float) -> Dict[str, Any]:
        """Execute stop loss fill"""
        fill_price = self._apply_costs(price, 'short' if pos.side == 'long' else 'long')
        fee = (self.fee_bps * 1e-4) * fill_price * pos.size

        pnl = ((fill_price - pos.entry) * pos.size if pos.side == 'long'
               else (pos.entry - fill_price) * pos.size)
        self.realized_pnl += pnl - fee

        return {
            "ts": ts, "symbol": symbol, "side": "stop",
            "price": fill_price, "size_filled": pos.size,
            "fee": fee, "pnl": pnl, "reason": "stop_loss"
        }

    def _execute_tp(self, ts, symbol, pos: Position, tp: TPLevel, price: float) -> Dict[str, Any]:
        """Execute take profit partial fill"""
        fill_size = pos.size * (tp.size_pct / 100.0)
        fill_price = self._apply_costs(price, 'short' if pos.side == 'long' else 'long')
        fee = (self.fee_bps * 1e-4) * fill_price * fill_size

        pnl = ((fill_price - pos.entry) * fill_size if pos.side == 'long'
               else (pos.entry - fill_price) * fill_size)
        self.realized_pnl += pnl - fee

        # Mark TP as filled and reduce position size
        tp.filled = True
        pos.size -= fill_size

        return {
            "ts": ts, "symbol": symbol, "side": f"tp{tp.r_multiple:.0f}",
            "price": fill_price, "size_filled": fill_size,
            "fee": fee, "pnl": pnl, "reason": f"take_profit_{tp.r_multiple:.0f}R"
        }
