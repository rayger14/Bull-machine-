"""
Extended Macro Engine for Bull Machine v1.7.3
Implements traders' macro logic: Wyckoff Insider, Moneytaur, ZeroIKA
"""
import numpy as np
from typing import Dict, List, Optional
import logging

logger = logging.getLogger(__name__)

def analyze_macro(macro_snapshot: Dict[str, Dict], config: Dict) -> Dict:
    """
    Analyze macro indicators for vetoes and signals based on traders' logic.

    Traders' Logic:
    - Wyckoff Insider: VIX/MOVE (crisis), Gold/DXY (fiat), USDT.D/USDC.D/BTC.D (dominance)
    - Moneytaur: DXY/Oil (rotations), funding/OI (leverage), yields (curve)
    - ZeroIKA: USDT.D/USDC.D (coil), DXY/EUR/USD (USD), OI (crisis blocks)

    Args:
        macro_snapshot: Dict of macro values with staleness info
        config: Configuration dict with thresholds

    Returns:
        {
            'veto_strength': float,     # 0.0-1.0 veto strength
            'signals': Dict,            # Individual signal flags
            'health': Dict,             # Health metrics
            'regime': str,              # Risk On/Off/Neutral
            'notes': List[str]          # Human-readable analysis
        }
    """
    veto_strength = 0.0
    signals = {}
    notes = []
    regime = "neutral"

    # Helper to safely get values
    def get_value(symbol: str, default: float = 0.0) -> Optional[float]:
        data = macro_snapshot.get(symbol, {})
        return data.get('value', default) if not data.get('stale', True) else None

    # 1. WYCKOFF INSIDER LOGIC
    # ======================

    # VIX: Risk regime (>20 = risk-off veto, Wyckoff: systemic pivots)
    vix = get_value('VIX', 20.0)
    if vix is not None and vix > config.get('vix_regime_switch_threshold', 20.0):
        veto_strength += 0.4
        signals['vix_risk_off'] = True
        notes.append(f"VIX spike {vix:.1f} - risk off regime")
        regime = "risk_off"

    # MOVE: Bond volatility (>80 = credit stress, Wyckoff: crisis detection)
    move = get_value('MOVE', 80.0)
    if move is not None and move > config.get('move_threshold', 80.0):
        veto_strength += 0.3
        signals['move_credit_stress'] = True
        notes.append(f"MOVE spike {move:.1f} - credit stress")

    # Gold: Flight-to-safety (cycle highs = systemic pivot)
    gold = get_value('GOLD', 2500.0)
    if gold is not None and gold > config.get('gold_cycle_high', 2600.0):
        veto_strength += 0.2
        signals['gold_safe_haven'] = True
        notes.append(f"Gold at cycle high {gold:.0f} - safe haven flow")

    # USDT.D + USDC.D: Stablecoin dominance (coil = alt bleed, Wyckoff logic)
    usdt_d = get_value('USDT.D', 0.05)
    usdc_d = get_value('USDC.D', 0.03)
    if usdt_d is not None and usdc_d is not None:
        total_stable_dominance = usdt_d + usdc_d
        if total_stable_dominance > config.get('stablecoin_dominance_threshold', 0.08):
            veto_strength += 0.3
            signals['stablecoin_bleed'] = True
            notes.append(f"Stablecoin dominance {total_stable_dominance:.3f} - alt bleed")

    # BTC.D: Bitcoin dominance (wedge = suppress alts, Wyckoff: phase analysis)
    btc_d = get_value('BTC.D', 0.55)
    if btc_d is not None and btc_d > config.get('btc_dominance_threshold', 0.60):
        veto_strength += 0.2
        signals['btc_dominance'] = True
        notes.append(f"BTC dominance {btc_d:.3f} - alt suppression")

    # 2. MONEYTAUR LOGIC
    # ==================

    # DXY: USD strength (breakout = crypto suppression, liquidity drain)
    dxy = get_value('DXY', 100.0)
    if dxy is not None and dxy > config.get('dxy_breakout_threshold', 105.0):
        veto_strength += 0.3
        signals['dxy_breakout'] = True
        notes.append(f"DXY breakout {dxy:.1f} - liquidity drain")

    # WTI: Oil/energy (stagflation combo with DXY)
    wti = get_value('WTI', 70.0)
    if wti is not None:
        if wti < config.get('wti_low_threshold', 65.0):
            veto_strength -= 0.1  # Clear inflation veto
            signals['wti_inflation_relief'] = True
            notes.append(f"Oil relief {wti:.1f} - inflation pressure off")
        elif dxy is not None and wti > 80.0 and dxy > 104.0:
            veto_strength += 0.3
            signals['stagflation_combo'] = True
            notes.append(f"DXY+Oil stagflation combo - risk off")

    # US10Y/US2Y: Yield curve (inversion = risk-off, Moneytaur: rotations)
    us10y = get_value('US10Y', 4.0)
    us2y = get_value('US2Y', 4.2)
    if us10y is not None and us2y is not None:
        yield_spread = us10y - us2y
        if yield_spread < -config.get('yield_inversion_threshold', 0.2):
            veto_strength += 0.2
            signals['yield_inversion'] = True
            notes.append(f"Yield inversion {yield_spread:.2f} - recession risk")

    # Funding Rates: Leverage stress (Moneytaur: supply/demand)
    funding = get_value('FUNDING', 0.01)
    if funding is not None and funding > config.get('funding_max', 0.01):
        veto_strength += 0.3
        signals['funding_stress'] = True
        notes.append(f"Funding stress {funding:.3f} - leverage unwinding")

    # 3. ZEROIKA LOGIC
    # ================

    # EUR/USD: Forex rotation (USD weakness = crypto bullish)
    eurusd = get_value('EURUSD', 1.1)
    if eurusd is not None and eurusd > config.get('eurusd_bullish_threshold', 1.12):
        veto_strength -= 0.1  # Clear USD suppression
        signals['eurusd_bullish'] = True
        notes.append(f"EUR/USD strength {eurusd:.3f} - USD weakness")

    # Open Interest: Leverage stress (ZeroIKA: OI spikes >2% = crisis block)
    oi_premium = get_value('OI', 0.015)
    if oi_premium is not None and oi_premium > config.get('oi_spot_max', 0.015):
        veto_strength += 0.3
        signals['oi_stress'] = True
        notes.append(f"OI premium {oi_premium:.3f} - leverage crisis")

    # TOTAL3: Alt market cap (lag = alt weakness, ZeroIKA: structural shifts)
    total3 = get_value('TOTAL3', 1.0)
    total2 = get_value('TOTAL2', 1.0)
    if total3 is not None and total2 is not None:
        if total3 / total2 < config.get('total3_lag_threshold', 0.9):
            veto_strength += 0.2
            signals['alt_bleed'] = True
            notes.append(f"TOTAL3 lagging - alt sector bleed")

    # 4. REGIME DETECTION
    # ===================
    if veto_strength > 0.7:
        regime = "risk_off"
    elif veto_strength < 0.2 and signals.get('eurusd_bullish') and not signals.get('vix_risk_off'):
        regime = "risk_on"
    else:
        regime = "neutral"

    # 5. CAP VETO STRENGTH
    # ====================
    veto_strength = min(max(veto_strength, 0.0), 1.0)

    # 6. HEALTH METRICS
    # =================
    health = {
        'macro_veto_rate': veto_strength * 100,  # Convert to percentage
        'active_signals': len([k for k, v in signals.items() if v]),
        'regime': regime,
        'data_freshness': sum(1 for v in macro_snapshot.values()
                             if not v.get('stale', True) and v.get('value') is not None)
    }

    return {
        'veto_strength': veto_strength,
        'signals': signals,
        'health': health,
        'regime': regime,
        'notes': notes,
        'macro_delta': _calculate_macro_delta(signals, config)
    }

def _calculate_macro_delta(signals: Dict, config: Dict) -> float:
    """
    Calculate macro delta for fusion engine (clamped ±0.10).

    Positive delta = bullish macro winds
    Negative delta = bearish macro headwinds
    """
    delta = 0.0

    # Bearish signals (negative delta)
    if signals.get('vix_risk_off'):
        delta -= 0.04
    if signals.get('dxy_breakout'):
        delta -= 0.03
    if signals.get('move_credit_stress'):
        delta -= 0.03
    if signals.get('stablecoin_bleed'):
        delta -= 0.02
    if signals.get('funding_stress'):
        delta -= 0.03
    if signals.get('oi_stress'):
        delta -= 0.03
    if signals.get('yield_inversion'):
        delta -= 0.02

    # Bullish signals (positive delta)
    if signals.get('wti_inflation_relief'):
        delta += 0.02
    if signals.get('eurusd_bullish'):
        delta += 0.02
    if signals.get('gold_safe_haven'):  # Contrarian: late-cycle gold can precede risk-on
        delta += 0.01

    # Clamp to ±0.10 (macro delta cap from spec)
    return max(min(delta, 0.10), -0.10)

def create_default_macro_config() -> Dict:
    """
    Create default macro configuration with traders' thresholds.
    """
    return {
        # Wyckoff Insider thresholds
        'vix_regime_switch_threshold': 20.0,
        'move_threshold': 80.0,
        'gold_cycle_high': 2600.0,
        'stablecoin_dominance_threshold': 0.08,
        'btc_dominance_threshold': 0.60,

        # Moneytaur thresholds
        'dxy_breakout_threshold': 105.0,
        'wti_low_threshold': 65.0,
        'yield_inversion_threshold': 0.2,
        'funding_max': 0.01,

        # ZeroIKA thresholds
        'eurusd_bullish_threshold': 1.12,
        'oi_spot_max': 0.015,
        'total3_lag_threshold': 0.9,

        # Fusion integration
        'macro_weight': 0.3,
        'macro_veto_threshold': 0.70
    }