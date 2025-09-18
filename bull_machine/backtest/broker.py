
from dataclasses import dataclass
from typing import Dict, Any

@dataclass
class Position:
    side: str          # 'long'|'short'
    size: float
    entry: float
    stop: float | None = None
    tp: list[float] | None = None

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

    def submit(self, ts, symbol, side, size, price_hint=None, order_type='market') -> Dict[str, Any]:
        px = self._apply_costs(price_hint or 0.0, side)
        fee = (self.fee_bps * 1e-4) * px * abs(size)
        prev = self.positions.get(symbol)
        if prev and ((prev.side=='long' and side=='short') or (prev.side=='short' and side=='long')):
            pnl = (px - prev.entry) * prev.size if prev.side=='long' else (prev.entry - px) * prev.size
            self.realized_pnl += pnl - fee
            self.positions.pop(symbol, None)
        self.positions[symbol] = Position(side=side, size=size, entry=px)
        return {"ts": ts, "symbol": symbol, "side": side, "price": px, "size_filled": size, "fee": fee, "slippage": 0.0}

    def close(self, ts, symbol) -> Dict[str, Any]:
        pos = self.positions.get(symbol)
        if not pos: return {"closed": False}
        px = pos.entry  # placeholder; engine should pass last close for now
        fee = (self.fee_bps * 1e-4) * px * abs(pos.size)
        self.positions.pop(symbol, None)
        return {"ts": ts, "symbol": "exit", "side": "exit", "price": px, "size_filled": pos.size, "fee": fee}

    def mark(self, ts, symbol, price):
        return None
