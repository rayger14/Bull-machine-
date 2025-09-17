
from typing import Dict, List, Optional, Tuple
from ...core.types import Series, Signal, RiskPlan, WyckoffResult
from ...core.utils import calculate_atr, find_swing_high_low

class AdvancedRiskManager:
    """v1.2.1 Risk Manager with core patches applied"""
    def __init__(self, config: dict):
        self.config = config
        self.risk_cfg = config.get('risk', {})
        self.stop_cfg = self.risk_cfg.get('stop', {})
        self.stop_method = self.stop_cfg.get('method', 'swing_with_atr_guardrail')
        self.atr_mult = self.stop_cfg.get('atr_mult', 3.0)
        self.swing_buffer = self.stop_cfg.get('swing_buffer', 0.001)
        self.account_risk_percent = self.risk_cfg.get('account_risk_percent', 1.0)
        self.max_risk_percent = self.risk_cfg.get('max_risk_percent', 1.25)
        self.max_risk_per_trade = self.risk_cfg.get('max_risk_per_trade', 150.0)

    # ---- Stops ----
    def _atr_based_stop(self, series: Series, side: str, entry_price: float, profile: str) -> Tuple[float, str]:
        atr = calculate_atr(series)
        atr_pct = atr / max(entry_price, 1e-12)
        base_mult = {'scalp': 1.25, 'swing': 3.0, 'trend': 3.5, 'range': 2.0}.get(profile, self.atr_mult)
        if self.stop_cfg.get('volatility_scaling', False):
            if atr_pct > 0.02:
                mult = base_mult * 1.2
            elif atr_pct < 0.008:
                mult = max(base_mult * 0.9, 1.0)
            else:
                mult = base_mult
        else:
            mult = base_mult
        stop = entry_price - atr*mult if side == 'long' else entry_price + atr*mult
        return stop, 'atr_scaled'

    def _swing_stop_with_atr(self, series: Series, side: str, entry_price: float, profile: str) -> Tuple[float, str]:
        swing_high, swing_low = find_swing_high_low(series, lookback=20)
        atr_stop, _ = self._atr_based_stop(series, side, entry_price, profile)
        if side == 'long':
            swing_stop = swing_low * (1 - self.swing_buffer)
            return (swing_stop, 'swing') if (swing_stop > entry_price*0.95 and swing_stop > atr_stop) else (atr_stop, 'atr')
        else:
            swing_stop = swing_high * (1 + self.swing_buffer)
            return (swing_stop, 'swing') if (swing_stop < entry_price*1.05 and swing_stop < atr_stop) else (atr_stop, 'atr')

    def _calculate_stop_loss(self, series: Series, side: str, entry_price: float, 
                             liquidity_result: Dict, profile: str) -> Tuple[float, str]:
        if self.stop_method == 'swing_with_atr_guardrail':
            return self._swing_stop_with_atr(series, side, entry_price, profile)
        elif self.stop_method == 'liquidity_based':
            stop = self._liquidity_based_stop(series, side, liquidity_result, entry_price)
            return stop if stop[0] != 0 else self._atr_based_stop(series, side, entry_price, profile)
        return self._atr_based_stop(series, side, entry_price, profile)

    # ---- Position sizing ----
    def _calculate_position_size(self, series: Series, entry: float, stop: float, 
                                 balance: float, profile: str) -> Tuple[float, float, float]:
        risk_mult = {'scalp': 0.5, 'swing': 1.0, 'trend': 1.2, 'range': 0.8}.get(profile, 1.0)
        if self.stop_cfg.get('volatility_scaling', False):
            atr = calculate_atr(series)
            atr_pct = atr / max(entry, 1e-12)
            target = self.stop_cfg.get('target_volatility', 0.012)
            vol_scale = max(min(target / max(atr_pct, 1e-6), 1.25), 0.6)
            risk_pct = self.account_risk_percent * risk_mult * vol_scale
        else:
            risk_pct = self.account_risk_percent * risk_mult
        risk_pct = min(risk_pct, self.max_risk_percent)
        risk_amt = min(balance * (risk_pct/100), self.max_risk_per_trade)
        risk_per_unit = max(abs(entry - stop), 1e-12)
        size = risk_amt / risk_per_unit
        return size, risk_amt, risk_pct

    def _determine_profile(self, wyckoff_result, liquidity_result) -> str:
        """Determine trading profile based on market conditions"""
        return 'swing'  # Simplified for now

    def _calculate_entry(self, series: Series, signal: Signal, liquidity_result: Dict) -> float:
        """Calculate entry price"""
        return series.bars[-1].close  # Use last close as entry

    def _liquidity_based_stop(self, series: Series, side: str, liquidity_result: Dict, entry_price: float) -> Tuple[float, str]:
        """Calculate stop based on liquidity levels"""
        # Simplified - fallback to ATR
        return 0.0, 'fallback'

    def _calculate_tp_ladder(self, entry: float, stop: float, side: str, profile: str) -> List[Dict]:
        """Calculate take profit levels"""
        risk = abs(entry - stop)
        if side == 'long':
            return [
                {'name': 'tp1', 'price': entry + risk * 1.0, 'pct': 25, 'action': 'move_stop_to_breakeven'},
                {'name': 'tp2', 'price': entry + risk * 2.5, 'pct': 35, 'action': 'trail_remainder'},
                {'name': 'tp3', 'price': entry + risk * 4.5, 'pct': 40, 'action': 'liquidate_or_hard_trail'}
            ]
        else:
            return [
                {'name': 'tp1', 'price': entry - risk * 1.0, 'pct': 25, 'action': 'move_stop_to_breakeven'},
                {'name': 'tp2', 'price': entry - risk * 2.5, 'pct': 35, 'action': 'trail_remainder'},
                {'name': 'tp3', 'price': entry - risk * 4.5, 'pct': 40, 'action': 'liquidate_or_hard_trail'}
            ]

    def _calculate_expected_r(self, tps: List[Dict], profile: str) -> float:
        """Calculate expected R value"""
        return 2.5  # Simplified

    # ---- Public ----
    def plan_trade(self, series: Series, signal: Signal, account_balance: float, liquidity_result: Dict = None, wyckoff_result = None) -> RiskPlan:
        profile = self._determine_profile(wyckoff_result, liquidity_result)
        entry = self._calculate_entry(series, signal, liquidity_result)
        stop, stop_type = self._calculate_stop_loss(series, signal.side, entry, liquidity_result, profile)
        size, risk_amt, risk_pct = self._calculate_position_size(series, entry, stop, account_balance, profile)
        tps = self._calculate_tp_ladder(entry, stop, signal.side, profile)
        expected_r = self._calculate_expected_r(tps, profile)
        return RiskPlan(entry=entry, stop=stop, size=size, tp_levels=tps, stop_type=stop_type,
                        risk_amount=risk_amt, risk_percent=risk_pct, profile=profile, expected_r=expected_r)
