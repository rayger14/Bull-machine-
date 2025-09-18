
from dataclasses import dataclass
from typing import Dict

@dataclass
class BookPosition:
    side: str
    size: float
    entry: float

class Portfolio:
    def __init__(self, starting_cash: float, exposure_cap_pct: float = 0.5):
        self.cash = starting_cash
        self.realized = 0.0
        self.unrealized = 0.0
        self.positions: Dict[str, BookPosition] = {}
        self.high_water = starting_cash
        self.exposure_cap_pct = exposure_cap_pct

    def can_add(self, side: str, risk_amount: float, equity: float) -> bool:
        """Check if position can be added without violating exposure limits"""

        # Calculate current exposure by side
        long_exposure = 0.0
        short_exposure = 0.0

        for pos in self.positions.values():
            exposure = abs(pos.size * pos.entry)  # Notional value
            if pos.side == 'long':
                long_exposure += exposure
            else:
                short_exposure += exposure

        # Calculate new exposure after proposed trade
        trade_exposure = risk_amount  # Simplified: use risk amount as proxy

        if side == 'long':
            new_long_exposure = long_exposure + trade_exposure
            net_exposure = (new_long_exposure - short_exposure) / equity
        else:
            new_short_exposure = short_exposure + trade_exposure
            net_exposure = (long_exposure - new_short_exposure) / equity

        # Enforce 50% net exposure cap (configurable)
        return abs(net_exposure) <= self.exposure_cap_pct

    def on_fill(self, symbol: str, side: str, price: float, size: float, fee: float):
        if side in ('long','short'):
            self.positions[symbol] = BookPosition(side, size, price)
            self.cash -= fee
        elif side == 'exit':
            pos = self.positions.pop(symbol, None)
            if pos:
                pnl = (price - pos.entry) * pos.size if pos.side=='long' else (pos.entry - price) * pos.size
                self.realized += pnl - fee
                self.cash += pnl - fee

    def mark(self, symbol: str, price: float):
        pos = self.positions.get(symbol)
        if not pos:
            self.unrealized = 0.0
            return
        self.unrealized = (price - pos.entry) * pos.size if pos.side=='long' else (pos.entry - price) * pos.size

    def equity(self) -> float:
        eq = self.cash + self.realized + self.unrealized
        if eq > self.high_water:
            self.high_water = eq
        return eq

    def drawdown(self) -> float:
        return self.high_water - self.equity()
