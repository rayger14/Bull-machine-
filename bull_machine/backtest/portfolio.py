
from dataclasses import dataclass
from typing import Dict, Optional, Any
import logging

@dataclass
class BookPosition:
    side: str
    size: float
    entry: float

class Portfolio:
    def __init__(self, starting_cash: float, exposure_cap_pct: float = 0.5, max_positions: int = 8):
        self.cash = starting_cash
        self.realized = 0.0
        self.unrealized = 0.0
        self.positions: Dict[str, BookPosition] = {}
        self.high_water = starting_cash
        self.exposure_cap_pct = exposure_cap_pct
        self.max_positions = max_positions

    def compute_position_size(self, risk_plan: Dict[str, Any], balance: float,
                            base_risk_pct: float = 0.01) -> Dict[str, Any]:
        """
        Compute proper position size based on actual risk (entry to stop distance)
        instead of notional value.
        """
        try:
            entry = risk_plan.get("entry")
            stop = risk_plan.get("stop")

            if entry is None or stop is None:
                logging.warning("Missing entry or stop in risk_plan, using fallback sizing")
                return {
                    "position_size": balance * base_risk_pct / 100,  # Tiny fallback
                    "risk_amount": balance * base_risk_pct,
                    "risk_per_unit": 100.0,
                    "effective_risk_pct": base_risk_pct
                }

            # Calculate per-unit risk (distance from entry to stop)
            risk_per_unit = abs(entry - stop)

            if risk_per_unit <= 0:
                logging.warning(f"Invalid risk_per_unit: {risk_per_unit}, using fallback")
                return {
                    "position_size": balance * base_risk_pct / 100,
                    "risk_amount": balance * base_risk_pct,
                    "risk_per_unit": 100.0,
                    "effective_risk_pct": base_risk_pct
                }

            # Calculate position size based on risk tolerance
            target_risk_amount = balance * base_risk_pct
            position_size = target_risk_amount / risk_per_unit

            return {
                "position_size": position_size,
                "risk_amount": target_risk_amount,  # This is the actual $ at risk
                "risk_per_unit": risk_per_unit,
                "effective_risk_pct": target_risk_amount / balance
            }

        except Exception as e:
            logging.error(f"Error in compute_position_size: {e}")
            return {
                "position_size": balance * base_risk_pct / 100,
                "risk_amount": balance * base_risk_pct,
                "risk_per_unit": 100.0,
                "effective_risk_pct": base_risk_pct
            }

    def scale_to_exposure(self, base_size: float, max_exposure_pct: float = 0.8) -> float:
        """
        Scale position size based on current portfolio exposure.
        Reduces size when approaching exposure limits.
        """
        current_exposure = 0.0
        equity = self.equity()

        # Calculate current exposure as sum of position notional values
        for symbol, pos in self.positions.items():
            notional = pos.size * pos.entry
            current_exposure += notional

        exposure_ratio = current_exposure / equity if equity > 0 else 0

        if exposure_ratio >= max_exposure_pct:
            # Already at limit, minimal size only
            return base_size * 0.1
        elif exposure_ratio >= max_exposure_pct * 0.8:
            # Approaching limit, scale down
            remaining_capacity = (max_exposure_pct - exposure_ratio) / (max_exposure_pct * 0.2)
            return base_size * max(0.3, remaining_capacity)
        else:
            # Plenty of room, use full size
            return base_size

    def can_add(self, side: str, risk_plan: Optional[Dict[str, Any]], equity: float,
               base_risk_pct: float = 0.01) -> tuple[bool, str]:
        """
        Enhanced position validation using proper risk calculation.
        Returns (can_add, reason) tuple.
        """
        # Check position count limits
        current_positions = len(self.positions)
        if current_positions >= self.max_positions:
            return False, f"Max positions reached: {current_positions}/{self.max_positions}"

        # If no risk plan provided, use simple check
        if risk_plan is None:
            logging.warning("No risk_plan provided, using conservative defaults")
            simple_risk = equity * base_risk_pct
            if simple_risk / equity > 0.05:  # 5% max for fallback
                return False, f"Fallback risk too high: {simple_risk/equity:.1%}"
            return True, "Approved with fallback risk"

        # Calculate proper position sizing
        sizing_result = self.compute_position_size(risk_plan, equity, base_risk_pct)
        actual_risk_amount = sizing_result["risk_amount"]

        # Risk checks
        max_risk_per_trade = 0.05  # 5% max risk per trade (increased from 2%)
        risk_pct = actual_risk_amount / equity

        if risk_pct > max_risk_per_trade:
            return False, f"Risk too high: {risk_pct:.1%} > {max_risk_per_trade:.1%}"

        # Position size sanity check
        position_size = sizing_result["position_size"]
        if position_size <= 0:
            return False, f"Invalid position size: {position_size}"

        # All checks passed
        return True, f"Approved: {risk_pct:.2%} risk, {position_size:.4f} size"

    def can_add_legacy(self, side: str, risk_amount: float, equity: float) -> bool:
        """Legacy method for backward compatibility"""
        result, _ = self.can_add(side, None, equity)
        return result

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
