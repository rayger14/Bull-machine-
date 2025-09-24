# NOTE: This is a focused patch for two areas:
#  - Correct R calculation on stop/trailing stop exits
#  - Respect Signal TTL when slicing future bars
from typing import Any, Dict


def calculate_single_r(entry_price, stop_price, exit_price, side):
    risk_per_unit = abs(entry_price - stop_price)
    if risk_per_unit == 0:
        return 0.0
    if side == "long":
        return (exit_price - entry_price) / risk_per_unit
    else:
        return (entry_price - exit_price) / risk_per_unit


def calculate_weighted_r(exits, initial_size):
    if initial_size <= 0:
        return 0.0
    total = 0.0
    for e in exits:
        total += e["r"] * (e["size"] / initial_size)
    return total


def simulate_trade(signal, risk_plan, entry_bar, future_bars, bar_idx) -> Dict[str, Any]:
    entry_price = risk_plan.entry if hasattr(risk_plan, "entry") else risk_plan["entry"]
    stop_price = risk_plan.stop if hasattr(risk_plan, "stop") else risk_plan["stop"]
    side = signal.side if hasattr(signal, "side") else signal["side"]
    tp_levels = risk_plan.tp_levels if hasattr(risk_plan, "tp_levels") else risk_plan["tp_levels"]

    # Respect signal TTL
    ttl = getattr(signal, "ttl_bars", 20) if hasattr(signal, "ttl_bars") else 20
    future_bars = future_bars[:ttl]  # Limit to TTL window

    initial_size = 1.0
    remaining_size = 1.0
    total_pnl = 0.0
    exits = []
    trailing_stop = None

    for i, bar in enumerate(future_bars):
        current_stop = trailing_stop if trailing_stop is not None else stop_price

        # STOP LOGIC (fixed R accounting)
        if side == "long" and bar.low <= current_stop:
            exit_pnl = (current_stop - entry_price) * remaining_size
            total_pnl += exit_pnl
            stop_r = calculate_single_r(entry_price, stop_price, current_stop, side)
            total_r = calculate_weighted_r(
                exits + [{"r": stop_r, "size": remaining_size}], initial_size
            )
            return {
                "entry_bar_idx": bar_idx,
                "exit_bar_idx": bar_idx + i + 1,
                "side": side,
                "entry_price": entry_price,
                "final_exit_price": current_stop,
                "stop_price": stop_price,
                "initial_size": initial_size,
                "r": total_r,
                "pnl": total_pnl,
                "exit_reason": "stop_loss" if not exits else "trailing_stop",
                "bars_held": i + 1,
                "exits": exits,
            }
        elif side == "short" and bar.high >= current_stop:
            exit_pnl = (entry_price - current_stop) * remaining_size
            total_pnl += exit_pnl
            stop_r = calculate_single_r(entry_price, stop_price, current_stop, side)
            total_r = calculate_weighted_r(
                exits + [{"r": stop_r, "size": remaining_size}], initial_size
            )
            return {
                "entry_bar_idx": bar_idx,
                "exit_bar_idx": bar_idx + i + 1,
                "side": side,
                "entry_price": entry_price,
                "final_exit_price": current_stop,
                "stop_price": stop_price,
                "initial_size": initial_size,
                "r": total_r,
                "pnl": total_pnl,
                "exit_reason": "stop_loss" if not exits else "trailing_stop",
                "bars_held": i + 1,
                "exits": exits,
            }

        # TP handling would be here in user's full implementation (left as-is)

    # TTL timeout
    if future_bars:
        final_price = future_bars[-1].close
        exit_pnl = (
            (final_price - entry_price) * remaining_size
            if side == "long"
            else (entry_price - final_price) * remaining_size
        )
        total_pnl += exit_pnl
        timeout_r = calculate_single_r(entry_price, stop_price, final_price, side)
        total_r = calculate_weighted_r(
            exits + [{"r": timeout_r, "size": remaining_size}], initial_size
        )
        return {
            "entry_bar_idx": bar_idx,
            "exit_bar_idx": bar_idx + len(future_bars),
            "side": side,
            "entry_price": entry_price,
            "final_exit_price": final_price,
            "stop_price": stop_price,
            "initial_size": initial_size,
            "r": total_r,
            "pnl": total_pnl,
            "exit_reason": "ttl_timeout",
            "bars_held": len(future_bars),
            "exits": exits,
        }
    return {}


# TTL respect (usage example in your run loop):
# ttl = getattr(signal, 'ttl_bars', 20) if signal else 20
# future_bars = full_series.bars[i+1 : min(i+1+ttl, len(full_series.bars))]
