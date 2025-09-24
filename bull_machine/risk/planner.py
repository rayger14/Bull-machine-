import logging
from typing import Dict, List

from ..core.types import RiskPlan, Series, Signal
from ..core.utils import calculate_atr, find_swing_high_low


def plan(series: Series, signal: Signal, cfg: dict, account_balance: float) -> RiskPlan:
    try:
        risk_cfg = cfg.get("risk", {})
        current_bar = series.bars[-1]
        entry_price = current_bar.close
        stop_price = _calculate_stop_loss(series, signal.side, risk_cfg)
        position_size = _calculate_position_size(entry_price, stop_price, account_balance, risk_cfg)
        tp_levels = _calculate_tp_ladder(entry_price, stop_price, signal.side, risk_cfg)
        rules = _build_execution_rules(risk_cfg)
        return RiskPlan(
            entry=entry_price, stop=stop_price, size=position_size, tp_levels=tp_levels, rules=rules
        )
    except Exception as e:
        logging.error(f"Risk planning error: {e}")
        return _get_default_risk_plan(series, signal)


def _calculate_stop_loss(series: Series, side: str, risk_cfg: Dict) -> float:
    stop_cfg = risk_cfg.get("stop", {})
    method = stop_cfg.get("method", "swing_with_atr_guardrail")
    atr_mult = stop_cfg.get("atr_mult", 2.0)
    current_price = series.bars[-1].close
    if method == "swing_with_atr_guardrail":
        swing_high, swing_low = find_swing_high_low(series)
        if side == "long":
            swing_stop = swing_low * 0.999
        else:
            swing_stop = swing_high * 1.001
        atr = calculate_atr(series)
        if side == "long":
            atr_stop = current_price - (atr * atr_mult)
            if (
                swing_stop < current_price
                and (current_price - swing_stop) / max(1e-9, current_price) < 0.05
            ):
                return swing_stop
            else:
                return atr_stop
        else:
            atr_stop = current_price + (atr * atr_mult)
            if (
                swing_stop > current_price
                and (swing_stop - current_price) / max(1e-9, current_price) < 0.05
            ):
                return swing_stop
            else:
                return atr_stop
    else:
        atr = calculate_atr(series)
        return (
            current_price - (atr * atr_mult) if side == "long" else current_price + (atr * atr_mult)
        )


def _calculate_position_size(entry: float, stop: float, balance: float, risk_cfg: Dict) -> float:
    risk_percent = risk_cfg.get("account_risk_percent", 1.0) / 100
    max_risk_percent = risk_cfg.get("max_risk_percent", 2.0) / 100
    max_risk_absolute = risk_cfg.get("max_risk_per_trade", 200.0)
    risk_per_unit = abs(entry - stop)
    if risk_per_unit == 0:
        return 0.0
    risk_amount = min(balance * risk_percent, balance * max_risk_percent, max_risk_absolute)
    return round(risk_amount / risk_per_unit, 6)


def _calculate_tp_ladder(entry: float, stop: float, side: str, risk_cfg: Dict) -> List[Dict]:
    tp_cfg = risk_cfg.get("tp_ladder", {})
    risk_distance = abs(entry - stop)
    tp_defs = [
        ("tp1", tp_cfg.get("tp1", {"r": 1.0, "pct": 33, "action": "move_stop_to_breakeven"})),
        ("tp2", tp_cfg.get("tp2", {"r": 2.0, "pct": 33, "action": "trail_remainder"})),
        ("tp3", tp_cfg.get("tp3", {"r": 3.0, "pct": 34, "action": "liquidate_or_hard_trail"})),
    ]
    tps = []
    for name, cfg_tp in tp_defs:
        r = cfg_tp["r"]
        price = entry + risk_distance * r if side == "long" else entry - risk_distance * r
        tps.append(
            {"name": name, "r": r, "price": price, "pct": cfg_tp["pct"], "action": cfg_tp["action"]}
        )
    return tps


def _build_execution_rules(risk_cfg: Dict) -> Dict:
    return {
        "be_at": "tp1",
        "trail_at": "tp2",
        "trail_mode": risk_cfg.get("trail_mode", "swing"),
        "ttl_bars": risk_cfg.get("ttl_bars", 18),
    }


def _get_default_risk_plan(series: Series, signal: Signal) -> RiskPlan:
    px = series.bars[-1].close
    stop = px * (0.98 if signal.side == "long" else 1.02)
    return RiskPlan(
        entry=px,
        stop=stop,
        size=0.0,
        tp_levels=[],
        rules={"be_at": "tp1", "trail_at": "tp2", "trail_mode": "swing"},
    )
