
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
        # TODO: implement 50/50 side exposure cap
        return True

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
