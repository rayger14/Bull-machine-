
from dataclasses import dataclass
from typing import Dict, Any, Optional, List
import pandas as pd
import logging

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

    # Position aging fields (immutable after first entry)
    opened_at_ts: Optional[int] = None      # FIRST fill timestamp (epoch seconds) - never changes
    opened_at_idx: Optional[int] = None     # FIRST fill bar index - never changes
    timeframe: str = "1H"                   # trading timeframe
    bars_held: int = 0                      # current age in bars (updated per bar)
    last_update_idx: int = 0                # bookkeeping for last update

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

        # Auto-compute risk management if not provided
        stop_price = None
        tp_levels = None

        if risk_plan:
            stop_price = risk_plan.get('stop')
            if risk_plan.get('tp_levels'):
                tp_levels = [
                    TPLevel(
                        price=tp['price'],
                        size_pct=tp.get('pct', 33),
                        r_multiple=tp.get('r', 1.0)
                    ) for tp in risk_plan['tp_levels']
                ]

        # Auto-compute if missing (Option A - guarantee every entry has exits)
        if stop_price is None:
            # Default: 2% stop loss
            stop_price = px * 0.98 if side == 'long' else px * 1.02

        if tp_levels is None:
            # Default TP ladder: 1R/2R/3R with 40/30/30 split
            risk_per_unit = abs(px - stop_price)
            tp_levels = [
                TPLevel(
                    price=px + risk_per_unit if side == 'long' else px - risk_per_unit,
                    size_pct=40,
                    r_multiple=1.0
                ),
                TPLevel(
                    price=px + 2*risk_per_unit if side == 'long' else px - 2*risk_per_unit,
                    size_pct=30,
                    r_multiple=2.0
                ),
                TPLevel(
                    price=px + 3*risk_per_unit if side == 'long' else px - 3*risk_per_unit,
                    size_pct=30,
                    r_multiple=3.0
                )
            ]

        # Handle position creation/scaling with aging support
        existing_pos = self.positions.get(symbol)

        if existing_pos is None:
            # NEW position - set immutable open markers
            self.positions[symbol] = Position(
                side=side,
                size=size,
                entry=px,
                stop=stop_price,
                tp_levels=tp_levels,
                opened_at_ts=ts,
                opened_at_idx=getattr(self, '_current_bar_idx', 0),  # Will be set by engine
                timeframe=getattr(self, '_current_timeframe', '1H'),
                bars_held=0,
                last_update_idx=getattr(self, '_current_bar_idx', 0)
            )
        else:
            # SCALE-IN to existing position - preserve opened_at_* fields
            if existing_pos.side == side:
                # Same side scale-in: weighted average price, sum size
                new_size = existing_pos.size + size
                if new_size > 0:
                    existing_pos.entry = (existing_pos.entry * existing_pos.size + px * size) / new_size
                existing_pos.size = new_size
                existing_pos.stop = stop_price  # Update stop to new level
                existing_pos.tp_levels = tp_levels  # Update TP levels
                existing_pos.last_update_idx = getattr(self, '_current_bar_idx', 0)
                # KEEP opened_at_ts, opened_at_idx unchanged!
            else:
                # Different side - this should have been handled by reversal logic above
                logging.warning(f"Unexpected different side entry for {symbol}: {existing_pos.side} -> {side}")
                # Create new position anyway
                self.positions[symbol] = Position(
                    side=side,
                    size=size,
                    entry=px,
                    stop=stop_price,
                    tp_levels=tp_levels,
                    opened_at_ts=ts,
                    opened_at_idx=getattr(self, '_current_bar_idx', 0),
                    timeframe=getattr(self, '_current_timeframe', '1H'),
                    bars_held=0,
                    last_update_idx=getattr(self, '_current_bar_idx', 0)
                )

        import logging
        import json

        # Basic entry log
        logging.info(f"BROKER_ENTER_OK symbol={symbol} side={side} size={size:.4f} price={px:.4f}")

        # Comprehensive entry log with structured data
        pos = self.positions[symbol]  # Position was just created/updated
        position_type = "NEW" if existing_pos is None else "SCALE_IN"
        entry_log = {
            "event": "ENTRY",
            "timestamp": ts.isoformat() if hasattr(ts, 'isoformat') else str(ts),
            "symbol": symbol,
            "side": side,
            "size": size,
            "price": px,
            "fee": fee,
            "position_type": position_type,
            "position_size": pos.size,
            "position_entry": pos.entry,
            "opened_at_ts": pos.opened_at_ts.isoformat() if hasattr(pos.opened_at_ts, 'isoformat') else str(pos.opened_at_ts) if pos.opened_at_ts is not None else None,
            "opened_at_idx": pos.opened_at_idx,
            "bars_held": pos.bars_held,
            "timeframe": pos.timeframe
        }
        logging.info(f"ENTRY_DETAILED: {json.dumps(entry_log)}")

        return {"ts": ts, "symbol": symbol, "side": side, "price": px, "size_filled": size, "fee": fee, "slippage": 0.0}

    def update_position_aging(self, current_bar_idx: int, timeframe: str = "1H"):
        """Update bars_held for all positions based on current bar index."""
        self._current_bar_idx = current_bar_idx
        self._current_timeframe = timeframe

        for symbol, pos in self.positions.items():
            if pos.opened_at_idx is not None:
                pos.bars_held = max(0, current_bar_idx - pos.opened_at_idx)
                pos.last_update_idx = current_bar_idx
                logging.debug(f"[AGING] {symbol}: bars_held={pos.bars_held} "
                             f"(current_idx={current_bar_idx} - opened_at={pos.opened_at_idx})")

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

    def mark(self, ts, symbol, price, exit_signal=None):
        """Mark position to market and check for stop/TP fills or exit signals"""
        pos = self.positions.get(symbol)
        if not pos:
            return None

        fills = []

        # Handle exit signal if provided (highest priority)
        if exit_signal:
            exit_fill = self._process_exit_signal(ts, symbol, pos, price, exit_signal)
            if exit_fill:
                fills.extend(exit_fill if isinstance(exit_fill, list) else [exit_fill])
                # Check if position was completely closed
                if symbol not in self.positions:
                    return fills

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

            # Check if position should be completely closed (all TPs hit or size < threshold)
            if pos.size <= 0.001 or all(tp.filled for tp in pos.tp_levels):
                if pos.size > 0.001:
                    # Close remaining position at market
                    final_fill = self._close_remaining_position(ts, symbol, pos, price)
                    fills.append(final_fill)
                self.positions.pop(symbol, None)

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

    def _close_remaining_position(self, ts, symbol, pos: Position, price: float) -> Dict[str, Any]:
        """Close any remaining position size at market"""
        fill_price = self._apply_costs(price, 'short' if pos.side == 'long' else 'long')
        fee = (self.fee_bps * 1e-4) * fill_price * pos.size

        pnl = ((fill_price - pos.entry) * pos.size if pos.side == 'long'
               else (pos.entry - fill_price) * pos.size)
        self.realized_pnl += pnl - fee

        return {
            "ts": ts, "symbol": symbol, "side": "close_remaining",
            "price": fill_price, "size_filled": pos.size,
            "fee": fee, "pnl": pnl, "reason": "auto_close_remaining"
        }

    def _process_exit_signal(self, ts, symbol, pos: Position, price: float, exit_signal) -> Optional[List[Dict[str, Any]]]:
        """
        Process an exit signal and execute the appropriate action.

        Args:
            ts: Timestamp
            symbol: Trading symbol
            pos: Current position
            price: Current market price
            exit_signal: ExitSignal object

        Returns:
            List of fill dictionaries if action taken, None otherwise
        """
        try:
            from bull_machine.strategy.exits.types import ExitAction

            action = exit_signal.action
            fills = []

            if action == ExitAction.FULL_EXIT:
                # Close entire position at market
                fill = self._execute_exit_signal_close(ts, symbol, pos, price, exit_signal, 1.0)
                fills.append(fill)
                self.positions.pop(symbol, None)

            elif action == ExitAction.PARTIAL_EXIT:
                # Close partial position
                exit_pct = getattr(exit_signal, 'exit_percentage', 0.5)
                fill = self._execute_exit_signal_close(ts, symbol, pos, price, exit_signal, exit_pct)
                fills.append(fill)

                # Reduce position size
                pos.size *= (1.0 - exit_pct)

                # If remaining size is tiny, close completely
                if pos.size <= 0.001:
                    remaining_fill = self._close_remaining_position(ts, symbol, pos, price)
                    fills.append(remaining_fill)
                    self.positions.pop(symbol, None)

            elif action == ExitAction.TIGHTEN_STOP:
                # Update stop loss to new tighter level
                new_stop = getattr(exit_signal, 'new_stop_price', None)
                if new_stop and self._is_valid_stop_update(pos, new_stop):
                    old_stop = pos.stop
                    pos.stop = new_stop
                    logging.info(f"Tightened stop for {symbol}: {old_stop:.2f} -> {new_stop:.2f}")

                    # Return info about stop update (not a trade fill)
                    fills.append({
                        "ts": ts, "symbol": symbol, "side": "stop_update",
                        "old_stop": old_stop, "new_stop": new_stop,
                        "reason": f"exit_signal_{exit_signal.exit_type.value}"
                    })

            elif action == ExitAction.FLIP_POSITION:
                # Close current position and open reverse position
                close_fill = self._execute_exit_signal_close(ts, symbol, pos, price, exit_signal, 1.0)
                fills.append(close_fill)

                # Open reverse position
                flip_side = getattr(exit_signal, 'flip_bias', 'short' if pos.side == 'long' else 'long')
                flip_fill = self.submit(ts, symbol, flip_side, pos.size, price)
                fills.append(flip_fill)

            return fills if fills else None

        except Exception as e:
            logging.error(f"Error processing exit signal for {symbol}: {e}")
            return None

    def _execute_exit_signal_close(self, ts, symbol, pos: Position, price: float,
                                  exit_signal, exit_percentage: float) -> Dict[str, Any]:
        """Execute position close due to exit signal."""
        close_size = pos.size * exit_percentage
        fill_price = self._apply_costs(price, 'short' if pos.side == 'long' else 'long')
        fee = (self.fee_bps * 1e-4) * fill_price * close_size

        pnl = ((fill_price - pos.entry) * close_size if pos.side == 'long'
               else (pos.entry - fill_price) * close_size)
        self.realized_pnl += pnl - fee

        # Calculate R multiple and other metrics
        r_multiple = None
        if pos.stop:
            risk_per_unit = abs(pos.entry - pos.stop)
            if risk_per_unit > 0:
                r_multiple = pnl / (risk_per_unit * close_size)

        # Comprehensive exit log with structured data
        import logging
        import json

        exit_log = {
            "event": "EXIT",
            "timestamp": str(ts),
            "symbol": symbol,
            "side": pos.side,
            "exit_type": exit_signal.exit_type.value,
            "exit_size": close_size,
            "exit_percentage": exit_percentage,
            "exit_price": fill_price,
            "entry_price": pos.entry,
            "pnl": pnl,
            "r_multiple": r_multiple,
            "fee": fee,
            "confidence": exit_signal.confidence,
            "urgency": exit_signal.urgency,
            "position_duration_bars": pos.bars_held,
            "timeframe": pos.timeframe,
            "opened_at_ts": pos.opened_at_ts.isoformat() if hasattr(pos.opened_at_ts, 'isoformat') else str(pos.opened_at_ts) if pos.opened_at_ts is not None else None,
            "opened_at_idx": pos.opened_at_idx,
            "realized_pnl_total": self.realized_pnl
        }
        logging.info(f"EXIT_DETAILED: {json.dumps(exit_log)}")

        return {
            "ts": ts, "symbol": symbol,
            "side": f"exit_{exit_signal.exit_type.value}",
            "price": fill_price, "size_filled": close_size,
            "fee": fee, "pnl": pnl,
            "reason": f"exit_signal_{exit_signal.exit_type.value}",
            "confidence": exit_signal.confidence,
            "urgency": exit_signal.urgency,
            "exit_percentage": exit_percentage
        }

    def _is_valid_stop_update(self, pos: Position, new_stop: float) -> bool:
        """Check if stop update is valid (tighter than current)."""
        if pos.side == 'long':
            # For long positions, new stop should be higher (tighter)
            return new_stop > pos.stop
        else:
            # For short positions, new stop should be lower (tighter)
            return new_stop < pos.stop

    def get_position_data(self, symbol: str) -> Optional[Dict[str, Any]]:
        """
        Get position data formatted for exit signal evaluation.

        Args:
            symbol: Trading symbol

        Returns:
            Position data dict or None if no position
        """
        pos = self.positions.get(symbol)
        if not pos:
            return None

        # Calculate current PnL percentage (approximate)
        current_pnl_pct = 0.0  # Would need current price to calculate accurately

        return {
            'symbol': symbol,
            'bias': pos.side,
            'size': pos.size,
            'entry_time': pos.opened_at_ts.isoformat() if hasattr(pos.opened_at_ts, 'isoformat') else str(pos.opened_at_ts) if pos.opened_at_ts is not None else None,  # Use actual opened timestamp
            'entry_price': pos.entry,
            'stop_price': pos.stop,
            'pnl_pct': current_pnl_pct,
            'be_moved': pos.be_moved,
            'trail_active': pos.trail_active,
            # Include new aging fields
            'bars_held': pos.bars_held,
            'timeframe': pos.timeframe,
            'opened_at_ts': pos.opened_at_ts,
            'opened_at_idx': pos.opened_at_idx,
            'last_update_idx': pos.last_update_idx
        }
