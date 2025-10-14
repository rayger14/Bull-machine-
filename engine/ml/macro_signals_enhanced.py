"""
Enhanced Macro Signals Engine

Implements missing macro signal logic:
1. Funding Rate Trap Detection (>0.01 with OI >0.015)
2. DXY + VIX Synergy Veto (DXY >105 AND VIX >30)
3. Yield Curve Inversion (2Y > 10Y = recession veto)
4. TOTAL2 Divergence (alt season boost)
5. BTC.D + Oil altseason signal
6. VIX + 2Y yield regime shift detector

Returns bounded deltas compatible with regime policy format
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MacroSignalsEnhanced:
    """
    Enhanced macro signals with trap detection and regime awareness
    """

    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize enhanced macro signals

        Args:
            config: Configuration dict with thresholds
        """
        self.config = config or {
            # Funding trap thresholds
            'funding_trap_threshold': 0.01,
            'oi_trap_threshold': 0.015,
            # DXY + VIX double trap
            'dxy_trap_threshold': 105.0,
            'vix_trap_threshold': 30.0,
            # Yield curve
            'yield_inversion_recession_veto': True,
            # TOTAL2 divergence
            'total2_alt_threshold': 0.405,  # TOTAL2/TOTAL ratio
            # BTC.D thresholds
            'btc_d_alt_threshold': 55.0,  # Below this = altseason
            # VIX + yield regime shift
            'vix_yield_regime_threshold': (30.0, 4.5)  # (VIX, 2Y yield)
        }

    def analyze_macro_conditions(self, macro_snapshot: Dict) -> Dict:
        """
        Analyze all macro conditions and return signal adjustments

        Args:
            macro_snapshot: Dict with macro features:
                {VIX, DXY, MOVE, YIELD_2Y, YIELD_10Y, funding,
                 oi, TOTAL, TOTAL2, USDT.D, BTC.D, rv_20d, rv_60d}

        Returns:
            Dict with adjustments:
            {
                "enter_threshold_delta": float,
                "risk_multiplier": float,
                "weight_nudges": dict,
                "suppress": bool,
                "notes": list,
                "signals": dict (for logging)
            }
        """
        signals = {}
        notes = []
        adjustments = {
            'enter_threshold_delta': 0.0,
            'risk_multiplier': 1.0,
            'weight_nudges': {},
            'suppress': False
        }

        # 1. Funding Rate Trap Detection
        funding_signal = self._check_funding_trap(macro_snapshot)
        signals['funding_trap'] = funding_signal
        if funding_signal['active']:
            adjustments['suppress'] = True
            notes.append(f"FUNDING TRAP: {funding_signal['reason']}")
            logger.warning(f"⚠️  Funding trap detected: {funding_signal['reason']}")

        # 2. DXY + VIX Synergy Veto (Double Trap)
        dxy_vix_signal = self._check_dxy_vix_synergy(macro_snapshot)
        signals['dxy_vix_synergy'] = dxy_vix_signal
        if dxy_vix_signal['active']:
            adjustments['suppress'] = True
            adjustments['risk_multiplier'] = min(adjustments['risk_multiplier'], 0.5)
            notes.append(f"DXY+VIX DOUBLE TRAP: {dxy_vix_signal['reason']}")
            logger.warning(f"⚠️  DXY+VIX trap: {dxy_vix_signal['reason']}")

        # 3. Yield Curve Inversion (Recession Veto)
        yield_signal = self._check_yield_curve_inversion(macro_snapshot)
        signals['yield_inversion'] = yield_signal
        if yield_signal['active']:
            adjustments['enter_threshold_delta'] += 0.05  # Raise threshold
            adjustments['risk_multiplier'] *= 0.8
            notes.append(f"YIELD INVERSION: {yield_signal['reason']}")
            logger.info(f"📉 Yield curve inverted: {yield_signal['reason']}")

        # 4. TOTAL2 Divergence (Altseason Boost)
        total2_signal = self._check_total2_divergence(macro_snapshot)
        signals['total2_divergence'] = total2_signal
        if total2_signal['active']:
            adjustments['enter_threshold_delta'] -= 0.05  # Lower threshold (easier entry)
            adjustments['weight_nudges']['momentum'] = 0.05
            notes.append(f"ALTSEASON BOOST: {total2_signal['reason']}")
            logger.info(f"🚀 Alt season detected: {total2_signal['reason']}")

        # 5. BTC.D + Oil Altseason Signal
        btc_d_oil_signal = self._check_btc_d_oil_signal(macro_snapshot)
        signals['btc_d_oil'] = btc_d_oil_signal
        if btc_d_oil_signal['active']:
            adjustments['enter_threshold_delta'] -= 0.03
            notes.append(f"BTC.D+OIL SIGNAL: {btc_d_oil_signal['reason']}")

        # 6. VIX + 2Y Yield Regime Shift
        vix_yield_signal = self._check_vix_yield_regime_shift(macro_snapshot)
        signals['vix_yield_shift'] = vix_yield_signal
        if vix_yield_signal['active']:
            adjustments['suppress'] = True
            notes.append(f"REGIME SHIFT: {vix_yield_signal['reason']}")
            logger.warning(f"⚠️  Regime shift: {vix_yield_signal['reason']}")

        # Apply caps
        adjustments['enter_threshold_delta'] = np.clip(
            adjustments['enter_threshold_delta'], -0.10, +0.10
        )
        adjustments['risk_multiplier'] = np.clip(
            adjustments['risk_multiplier'], 0.0, 1.5
        )

        return {
            **adjustments,
            'notes': notes,
            'signals': signals
        }

    def _check_funding_trap(self, macro: Dict) -> Dict:
        """
        Check for funding rate trap

        Trap conditions:
        - Funding rate >0.01 (1% annualized = extremely elevated)
        - Open interest >0.015 (1.5% of market cap = overleveraged)

        Returns:
            {active: bool, reason: str, severity: float}
        """
        funding = macro.get('funding', 0.0)
        oi = macro.get('oi', 0.0)

        funding_elevated = funding > self.config['funding_trap_threshold']
        oi_elevated = oi > self.config['oi_trap_threshold']

        active = funding_elevated and oi_elevated

        severity = 0.0
        if active:
            severity = min(1.0, funding / 0.02)  # Scale 0-1 based on 2% max

        reason = ""
        if active:
            reason = f"funding={funding:.4f}, oi={oi:.4f} (overleveraged)"

        return {
            'active': active,
            'reason': reason,
            'severity': severity,
            'funding': funding,
            'oi': oi
        }

    def _check_dxy_vix_synergy(self, macro: Dict) -> Dict:
        """
        Check for DXY + VIX double trap

        Trap conditions:
        - DXY >105 (strong dollar = risk-off)
        - VIX >30 (elevated fear)

        Returns:
            {active: bool, reason: str, severity: float}
        """
        dxy = macro.get('DXY', 100.0)
        vix = macro.get('VIX', 20.0)

        dxy_elevated = dxy > self.config['dxy_trap_threshold']
        vix_elevated = vix > self.config['vix_trap_threshold']

        active = dxy_elevated and vix_elevated

        severity = 0.0
        if active:
            dxy_severity = (dxy - 105) / 10.0  # Scale by 10 DXY points
            vix_severity = (vix - 30) / 20.0   # Scale by 20 VIX points
            severity = min(1.0, (dxy_severity + vix_severity) / 2)

        reason = ""
        if active:
            reason = f"DXY={dxy:.1f}, VIX={vix:.1f} (double trap)"

        return {
            'active': active,
            'reason': reason,
            'severity': severity,
            'dxy': dxy,
            'vix': vix
        }

    def _check_yield_curve_inversion(self, macro: Dict) -> Dict:
        """
        Check for yield curve inversion (2Y > 10Y)

        Inversion signals recession risk

        Returns:
            {active: bool, reason: str, spread: float}
        """
        yield_2y = macro.get('YIELD_2Y', 4.0)
        yield_10y = macro.get('YIELD_10Y', 4.0)

        spread = yield_10y - yield_2y  # Normal: positive
        inverted = spread < 0.0

        active = inverted and self.config['yield_inversion_recession_veto']

        reason = ""
        if active:
            reason = f"2Y={yield_2y:.2f}% > 10Y={yield_10y:.2f}% (spread={spread:.2f}%)"

        return {
            'active': active,
            'reason': reason,
            'spread': spread,
            'yield_2y': yield_2y,
            'yield_10y': yield_10y
        }

    def _check_total2_divergence(self, macro: Dict) -> Dict:
        """
        Check for TOTAL2 divergence (altseason signal)

        Signal conditions:
        - TOTAL2/TOTAL >0.405 (altcoin dominance rising)
        - BTC.D declining

        Returns:
            {active: bool, reason: str, total2_ratio: float}
        """
        total = macro.get('TOTAL', np.nan)
        total2 = macro.get('TOTAL2', np.nan)
        btc_d = macro.get('BTC.D', 55.0)

        if pd.isna(total) or pd.isna(total2) or total == 0:
            return {
                'active': False,
                'reason': 'TOTAL/TOTAL2 data unavailable',
                'total2_ratio': np.nan
            }

        total2_ratio = total2 / total
        ratio_elevated = total2_ratio > self.config['total2_alt_threshold']
        btc_d_declining = btc_d < self.config['btc_d_alt_threshold']

        active = ratio_elevated and btc_d_declining

        reason = ""
        if active:
            reason = f"TOTAL2/TOTAL={total2_ratio:.3f}, BTC.D={btc_d:.1f}% (altseason)"

        return {
            'active': active,
            'reason': reason,
            'total2_ratio': total2_ratio,
            'btc_d': btc_d
        }

    def _check_btc_d_oil_signal(self, macro: Dict) -> Dict:
        """
        Check for BTC.D drop + Oil up signal

        Signal from Moneytaur: BTC.D ↓ with Oil ↑ = alt season

        Returns:
            {active: bool, reason: str}
        """
        btc_d = macro.get('BTC.D', 55.0)

        # Note: Oil not in current macro snapshot, using placeholder
        # In production, add WTI/Brent to macro features
        oil_rising = False  # Placeholder

        btc_d_declining = btc_d < self.config['btc_d_alt_threshold']

        active = btc_d_declining  # and oil_rising (when available)

        reason = ""
        if active:
            reason = f"BTC.D={btc_d:.1f}% declining (alt momentum)"

        return {
            'active': active,
            'reason': reason,
            'btc_d': btc_d
        }

    def _check_vix_yield_regime_shift(self, macro: Dict) -> Dict:
        """
        Check for VIX + 2Y yield regime shift

        Signal from ZeroIKA: VIX >30 with rising 2Y yields = regime shift

        Returns:
            {active: bool, reason: str}
        """
        vix = macro.get('VIX', 20.0)
        yield_2y = macro.get('YIELD_2Y', 4.0)

        vix_threshold, yield_threshold = self.config['vix_yield_regime_threshold']

        vix_elevated = vix > vix_threshold
        yield_elevated = yield_2y > yield_threshold

        active = vix_elevated and yield_elevated

        reason = ""
        if active:
            reason = f"VIX={vix:.1f}, 2Y={yield_2y:.2f}% (regime shift)"

        return {
            'active': active,
            'reason': reason,
            'vix': vix,
            'yield_2y': yield_2y
        }

    def get_greenlight_boost(self, macro: Dict) -> Dict:
        """
        Get positive signals that boost entry (greenlights)

        Opposite of vetoes - conditions that favor trading

        Returns:
            {threshold_delta, risk_mult, notes}
        """
        greenlights = {
            'threshold_delta': 0.0,
            'risk_mult': 1.0,
            'notes': []
        }

        # Low VIX + Low DXY = risk-on environment
        vix = macro.get('VIX', 20.0)
        dxy = macro.get('DXY', 100.0)

        if vix < 15 and dxy < 98:
            greenlights['threshold_delta'] -= 0.10  # Lower threshold
            greenlights['risk_mult'] = 1.15
            greenlights['notes'].append("LOW VIX + LOW DXY = RISK-ON")

        # Low funding + Normal OI = healthy market
        funding = macro.get('funding', 0.0)
        if abs(funding) < 0.005:
            greenlights['threshold_delta'] -= 0.05
            greenlights['notes'].append("NEUTRAL FUNDING = HEALTHY")

        # Yield curve normal + steepening
        yield_2y = macro.get('YIELD_2Y', 4.0)
        yield_10y = macro.get('YIELD_10Y', 4.0)
        spread = yield_10y - yield_2y

        if spread > 0.5:  # Healthy positive spread
            greenlights['risk_mult'] *= 1.05
            greenlights['notes'].append(f"YIELD CURVE NORMAL (spread={spread:.2f}%)")

        return greenlights


if __name__ == "__main__":
    # Test enhanced macro signals
    logger.info("=" * 70)
    logger.info("Testing Enhanced Macro Signals")
    logger.info("=" * 70)

    engine = MacroSignalsEnhanced()

    # Test case 1: Funding trap
    logger.info("\n1. Funding Trap Scenario:")
    test_macro_trap = {
        'VIX': 25.0,
        'DXY': 103.0,
        'funding': 0.015,  # 1.5% elevated
        'oi': 0.020,       # 2% elevated
        'YIELD_2Y': 4.3,
        'YIELD_10Y': 4.2,  # Slight inversion
        'BTC.D': 54.0,
        'TOTAL': 100.0,
        'TOTAL2': 42.0
    }

    result_trap = engine.analyze_macro_conditions(test_macro_trap)
    logger.info(f"   Suppress: {result_trap['suppress']}")
    logger.info(f"   Notes: {result_trap['notes']}")

    # Test case 2: Greenlight scenario
    logger.info("\n2. Greenlight Scenario:")
    test_macro_green = {
        'VIX': 12.0,      # Low fear
        'DXY': 96.0,      # Weak dollar
        'funding': 0.003, # Neutral
        'oi': 0.012,
        'YIELD_2Y': 4.0,
        'YIELD_10Y': 4.8, # Normal curve
        'BTC.D': 56.0,
        'TOTAL': 100.0,
        'TOTAL2': 40.0
    }

    result_green = engine.analyze_macro_conditions(test_macro_green)
    greenlight = engine.get_greenlight_boost(test_macro_green)

    logger.info(f"   Threshold Delta: {result_green['enter_threshold_delta']:+.3f}")
    logger.info(f"   Risk Multiplier: {result_green['risk_multiplier']:.2f}x")
    logger.info(f"   Greenlight Boost: {greenlight['threshold_delta']:+.3f}")
    logger.info(f"   Notes: {greenlight['notes']}")

    logger.info("\n✅ Enhanced macro signals tested")
