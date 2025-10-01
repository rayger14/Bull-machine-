"""
Macro Pulse Engine - Intermarket Relationships and Veto/Boost Logic

Implements comprehensive macro analysis including DXY/Oil/Gold/Bonds relationships,
VIX/MOVE spikes, carry trade unwinding, and credit stress detection.

"Macro is the weather. Wyckoff is the map. Liquidity is the terrain."
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum
import logging

from .macro_pulse_calibration import calibrate_macro_thresholds

logger = logging.getLogger(__name__)

class MacroRegime(Enum):
    """Macro market regimes"""
    RISK_ON = "risk_on"
    RISK_OFF = "risk_off"
    STAGFLATION = "stagflation"
    DEFLATION = "deflation"
    NEUTRAL = "neutral"

class VetoStrength(Enum):
    """Veto strength levels"""
    NONE = 0
    LIGHT = 1
    MODERATE = 2
    HARD = 3

@dataclass
class MacroSignal:
    """Individual macro signal"""
    name: str
    value: float  # 0-1 strength
    direction: str  # 'bullish', 'bearish', 'neutral'
    confidence: float
    timeframe: str
    metadata: Dict[str, Any]

@dataclass
class MacroPulse:
    """Complete macro context assessment"""
    regime: MacroRegime
    veto_strength: float  # 0-1
    fire_rate_stats: Dict[str, Any]  # Veto engagement tracking
    boost_strength: float  # 0-1
    suppression_flag: bool
    risk_bias: str  # 'risk_on', 'risk_off', 'neutral'
    macro_delta: float  # -0.1 to +0.1
    active_signals: List[MacroSignal]
    notes: List[str]
    plus_ones: List[str]

def rolling_breakout(df: pd.DataFrame, length: int = 20, col: str = "close") -> float:
    """Return 0..1 breakout strength above recent range."""
    if len(df) < length + 1:
        return 0.0

    win = df[col].tail(length)
    last = df[col].iloc[-1]
    hi, lo = win.max(), win.min()

    if hi == lo:
        return 0.0

    if last > hi:  # upside breakout
        return float(np.clip((last - hi) / max(1e-9, hi - lo), 0.0, 1.0))

    return 0.0

def dxy_breakout_strength(dxy_1d: pd.DataFrame, length: int) -> float:
    """DXY breakout = hard veto on longs (liquidity drain)"""
    return rolling_breakout(dxy_1d, length)

def oil_dxy_stagflation(oil_1d: pd.DataFrame, dxy_1d: pd.DataFrame) -> bool:
    """Oil↑ + DXY↑: stagflation veto - 'poison' combination"""
    if len(oil_1d) < 30 or len(dxy_1d) < 30:
        return False

    oil_up = oil_1d['close'].pct_change(10).iloc[-1] > 0.03  # 3% over 10 days
    dxy_up = dxy_1d['close'].pct_change(10).iloc[-1] > 0.01  # 1% over 10 days

    return bool(oil_up and dxy_up)

def gold_flight_to_safety(gold_1d: pd.DataFrame, risk_1d: pd.DataFrame) -> float:
    """Gold↑ while risk flat/down => hedging; 0..1 caution"""
    if len(gold_1d) < 15 or len(risk_1d) < 15:
        return 0.0

    g = gold_1d['close'].pct_change(10).iloc[-1]
    r = risk_1d['close'].pct_change(10).iloc[-1]

    # Gold rising while risk assets declining = flight to safety
    return float(np.clip(max(0.0, g - max(0.0, r)), 0.0, 0.6))

def yields_spike(us2y_1d: pd.DataFrame, us10y_1d: pd.DataFrame, std_k: float) -> float:
    """Yield spike if daily change exceeds k*std; return 0..1"""
    def spike_score(s):
        if len(s) < 30:
            return 0.0
        ch = s['close'].pct_change().tail(30)
        if ch.std() == 0 or np.isnan(ch.std()):
            return 0.0
        z = abs(ch.iloc[-1]) / ch.std()
        return float(np.clip(z / std_k, 0.0, 1.0))

    return max(spike_score(us2y_1d), spike_score(us10y_1d))

def vix_move_spike(vix_1d: pd.DataFrame, move_1d: pd.DataFrame, vix_th: float, move_th: float) -> float:
    """VIX (equity) / MOVE (bond) volatility spikes = risk-off"""
    vix_val = vix_1d['close'].iloc[-1] if len(vix_1d) else 0.0
    move_val = move_1d['close'].iloc[-1] if len(move_1d) else 0.0

    s1 = float(np.clip((vix_val - vix_th) / max(1.0, vix_th), 0.0, 1.0))
    s2 = float(np.clip((move_val - move_th) / max(1.0, move_th), 0.0, 1.0))

    return max(s1, s2)

def usd_jpy_carry_break(usd_jpy_1d: pd.DataFrame, level: float) -> bool:
    """Carry unwind proxy: sharp break below level ⇒ risk-off veto"""
    if len(usd_jpy_1d) < 5:
        return False

    last = usd_jpy_1d['close'].iloc[-1]
    prev = usd_jpy_1d['close'].iloc[-2]

    # Sharp decline below critical level
    return bool(last < level and (prev - last) / max(prev, 1e-9) > 0.01)

def hyg_credit_stress(hyg_1d: pd.DataFrame, length: int) -> float:
    """Junk credit dropping: veto score 0..1"""
    if len(hyg_1d) < length + 1:
        return 0.0

    win = hyg_1d['close'].tail(length)
    drop = (win.iloc[-1] - win.iloc[0]) / max(win.iloc[0], 1e-9)

    # >3% drop in window = high stress
    return float(np.clip(-drop / 0.03, 0.0, 1.0))

def ethd_gate(ethd_1d: pd.DataFrame) -> str:
    """ETH dominance trend gate for alt rotation"""
    if len(ethd_1d) < 30:
        return 'neutral'

    m = ethd_1d['close'].rolling(30).mean()
    return 'up' if ethd_1d['close'].iloc[-1] > m.iloc[-1] else 'down'

def dxy_time_shift_veto(dxy_1d: pd.DataFrame, btc_1d: pd.DataFrame, shifts: List[int]) -> float:
    """
    DXY accumulation → BTC top after ~200 days lag analysis
    Cross-correlation: if past DXY accumulation precedes BTC tops, return 0..1 veto
    """
    if len(dxy_1d) < 260 or len(btc_1d) < 260:
        return 0.0

    veto = 0.0
    dxy_ret = dxy_1d['close'].pct_change().fillna(0.0)
    btc_ret = btc_1d['close'].pct_change().fillna(0.0)

    for s in shifts:
        if len(dxy_ret) < 60 + s or len(btc_ret) < 60 + s:
            continue

        # Compare DXY last 60d strength to BTC weakness s days later
        dxy_win = dxy_ret.tail(60).mean()

        # Look at BTC performance around the shift period
        btc_ahead_start = max(0, len(btc_ret) - s - 60)
        btc_ahead_end = max(0, len(btc_ret) - s)

        if btc_ahead_end > btc_ahead_start:
            btc_ahead = btc_ret.iloc[btc_ahead_start:btc_ahead_end].mean()

            # If DXY was accumulating and BTC declined after lag
            if dxy_win > 0 and btc_ahead < 0:
                correlation_strength = min(1.0, dxy_win / 0.02)  # 2% DXY move = max
                veto = max(veto, correlation_strength)

    return float(veto)

def usdt_sfp_wolfe(usdt_4h: pd.DataFrame, lookback: int) -> float:
    """
    USDT.D SFP/Wolfe proxy: false breakout and quick re-entry
    Return caution (veto) score when topping SFP occurs
    """
    if len(usdt_4h) < lookback + 3:
        return 0.0

    window = usdt_4h['high'].tail(lookback)
    prev_hi = window[:-3].max()

    # Spike above then close back inside = SFP
    spiked = usdt_4h['high'].iloc[-2] > prev_hi * 1.002  # 0.2% breakout
    failed = usdt_4h['close'].iloc[-1] < prev_hi

    return float(0.7 if spiked and failed else 0.0)

def total3_vs_total(total3_4h: pd.DataFrame, total_4h: pd.DataFrame) -> float:
    """TOTAL3 vs TOTAL divergence for alt rotation detection"""
    if len(total3_4h) < 20 or len(total_4h) < 20:
        return 0.0

    total3_change = total3_4h['close'].pct_change(10).iloc[-1]
    total_change = total_4h['close'].pct_change(10).iloc[-1]

    # TOTAL3 outperforming = alt strength
    divergence = total3_change - total_change
    return float(np.clip(divergence / 0.05, -1.0, 1.0))  # ±5% = max

def ethbtc_trend_gate(ethbtc_1d: pd.DataFrame, ma_length: int) -> str:
    """ETH/BTC trend for alt season detection"""
    if len(ethbtc_1d) < ma_length:
        return 'neutral'

    ma = ethbtc_1d['close'].rolling(ma_length).mean()
    current = ethbtc_1d['close'].iloc[-1]
    trend_ma = ma.iloc[-1]

    if current > trend_ma * 1.02:  # 2% above MA
        return 'up'
    elif current < trend_ma * 0.98:  # 2% below MA
        return 'down'
    else:
        return 'neutral'

class MacroPulseEngine:
    """
    Complete Macro Pulse Engine implementing intermarket relationships.

    Integrates DXY/Oil/Gold/Bonds/VIX analysis with crypto-specific signals
    for institutional-grade macro context awareness.
    """

    def __init__(self, config: Dict[str, Any], adaptive_thresholds: Optional[Dict[str, Any]] = None):
        self.config = config
        self.series_config = config.get('series', {})
        self.adaptive_thresholds = adaptive_thresholds or {}

        # Thresholds (adaptive overrides static config)
        self.veto_threshold = config.get('veto_threshold', 0.7)
        self.boost_cap = config.get('boost_cap', 0.1)

        # Fire rate monitoring
        self.fire_rate_config = config.get('fire_rate_monitoring', {})
        self.veto_history = []  # Track veto events
        self.target_veto_rate = self.fire_rate_config.get('target_veto_rate', 0.1)
        self.max_veto_rate = self.fire_rate_config.get('max_veto_rate', 0.15)
        self.min_veto_rate = self.fire_rate_config.get('min_veto_rate', 0.05)

        # DXY parameters
        self.dxy_breakout_len = config.get('dxy_breakout_len', 20)
        self.dxy_time_shifts = config.get('dxy_time_shift_days', [120, 200, 240])

        # Volatility parameters (adaptive thresholds override config)
        self.vix_spike = self.adaptive_thresholds.get('vix_spike', config.get('vix_spike', 24.0))
        self.move_spike = self.adaptive_thresholds.get('move_spike', config.get('move_spike', 130.0))
        self.yields_spike_std = self.adaptive_thresholds.get('yields_spike_std', config.get('yields_spike_std', 2.0))

        # Credit parameters (adaptive)
        self.usd_jpy_level = self.adaptive_thresholds.get('usd_jpy_break_level', config.get('usd_jpy_break_level', 145.0))
        self.hyg_break_len = self.adaptive_thresholds.get('hyg_break_len', config.get('hyg_break_len', 15))

        # Crypto parameters
        self.usdt_sfp_lookback = config.get('usdt_sfp_lookback', 50)
        self.ethbtc_ma_length = config.get('ethbtc_ma_length', 50)

        # Weights
        self.weights = config.get('weights', {
            'boost_macro': 0.05,
            'boost_context_max': 0.10
        })

    def analyze_macro_pulse(self, series_data: Dict[str, pd.DataFrame]) -> MacroPulse:
        """
        Analyze complete macro pulse including all intermarket relationships.

        Args:
            series_data: Dict of market data by symbol

        Returns:
            MacroPulse with veto/boost analysis
        """
        try:
            # Extract series using config mapping
            def get_series(key: str) -> Optional[pd.DataFrame]:
                symbol = self.series_config.get(key)
                return series_data.get(symbol) if symbol else None

            # Initialize signals list
            signals = []
            notes = []
            plus_ones = []

            # 1. DXY Analysis (The Dollar is the heartbeat of global liquidity)
            dxy_signals = self._analyze_dxy_complex(get_series('DXY'), series_data, signals, notes)

            # 2. Oil/DXY Stagflation Check (Oil and DXY rising together is poison)
            stagflation_veto = self._check_stagflation(get_series('WTI'), get_series('DXY'), signals, notes)

            # 3. Flight to Safety Analysis (Gold + risk asset divergence)
            safety_signals = self._analyze_flight_to_safety(
                get_series('GOLD'), get_series('TOTAL'), signals, notes
            )

            # 4. Yield Curve and Bond Stress
            bond_signals = self._analyze_bond_stress(
                get_series('US2Y'), get_series('US10Y'), signals, notes
            )

            # 5. Volatility Regime (VIX/MOVE spikes)
            vol_signals = self._analyze_volatility_regime(
                get_series('VIX'), get_series('MOVE'), signals, notes
            )

            # 6. Systemic Risk (USDJPY carry, HYG credit)
            systemic_signals = self._analyze_systemic_risk(
                get_series('USDJPY'), get_series('HYG'), signals, notes
            )

            # 7. Crypto Rotation Analysis
            rotation_signals = self._analyze_crypto_rotation(
                get_series('USDT'), get_series('TOTAL3'), get_series('TOTAL'),
                get_series('ETHBTC'), get_series('ETHD'), signals, notes, plus_ones
            )

            # 8. Calculate aggregate veto/boost
            veto_strength, boost_strength, regime = self._calculate_macro_aggregates(signals)

            # 9. Determine risk bias and macro delta
            risk_bias, macro_delta = self._determine_risk_bias(signals, boost_strength, regime)

            # 10. Apply suppression logic
            suppression_flag = veto_strength >= self.veto_threshold

            return MacroPulse(
                regime=regime,
                veto_strength=veto_strength,
                boost_strength=boost_strength,
                suppression_flag=suppression_flag,
                risk_bias=risk_bias,
                macro_delta=macro_delta,
                active_signals=signals,
                notes=notes,
                plus_ones=plus_ones
            )

        except Exception as e:
            logger.error(f"Error in macro pulse analysis: {e}")
            return self._default_macro_pulse()

    def _analyze_dxy_complex(self, dxy_1d: Optional[pd.DataFrame],
                           all_series: Dict[str, pd.DataFrame],
                           signals: List[MacroSignal], notes: List[str]) -> List[MacroSignal]:
        """Comprehensive DXY analysis including breakouts and time-shift correlations"""
        dxy_signals = []

        if dxy_1d is None or len(dxy_1d) < self.dxy_breakout_len:
            return dxy_signals

        # DXY breakout strength
        breakout_strength = dxy_breakout_strength(dxy_1d, self.dxy_breakout_len)
        if breakout_strength > 0.3:
            dxy_signals.append(MacroSignal(
                name="DXY_BREAKOUT",
                value=breakout_strength,
                direction="bearish",
                confidence=0.9,
                timeframe="1D",
                metadata={"threshold": 0.3, "liquidity_drain": True}
            ))
            notes.append(f"DXY breakout {breakout_strength:.2f} (liquidity drain)")

        # DXY time-shift correlation with BTC
        btc_series = all_series.get('BTCUSD_1D') or all_series.get('BTC_1D')
        if btc_series is not None:
            lag_veto = dxy_time_shift_veto(dxy_1d, btc_series, self.dxy_time_shifts)
            if lag_veto > 0.4:
                dxy_signals.append(MacroSignal(
                    name="DXY_LAG_VETO",
                    value=lag_veto,
                    direction="bearish",
                    confidence=0.7,
                    timeframe="1D",
                    metadata={"lag_days": self.dxy_time_shifts, "correlation_based": True}
                ))
                notes.append(f"DXY lag correlation veto {lag_veto:.2f}")

        # DXY breakdown (bullish for risk)
        dxy_breakdown = self._check_dxy_breakdown(dxy_1d)
        if dxy_breakdown > 0.3:
            dxy_signals.append(MacroSignal(
                name="DXY_BREAKDOWN",
                value=dxy_breakdown,
                direction="bullish",
                confidence=0.8,
                timeframe="1D",
                metadata={"liquidity_flow": True}
            ))

        signals.extend(dxy_signals)
        return dxy_signals

    def _check_stagflation(self, oil_1d: Optional[pd.DataFrame], dxy_1d: Optional[pd.DataFrame],
                          signals: List[MacroSignal], notes: List[str]) -> bool:
        """Check for Oil↑ + DXY↑ stagflation scenario"""
        if oil_1d is None or dxy_1d is None:
            return False

        stagflation = oil_dxy_stagflation(oil_1d, dxy_1d)
        if stagflation:
            signals.append(MacroSignal(
                name="STAGFLATION_VETO",
                value=0.85,
                direction="bearish",
                confidence=0.95,
                timeframe="1D",
                metadata={"oil_up": True, "dxy_up": True, "poison_combo": True}
            ))
            notes.append("Oil↑ + DXY↑ (stagflation poison)")

        return stagflation

    def _analyze_flight_to_safety(self, gold_1d: Optional[pd.DataFrame], risk_proxy: Optional[pd.DataFrame],
                                 signals: List[MacroSignal], notes: List[str]) -> List[MacroSignal]:
        """Analyze gold flight-to-safety dynamics"""
        safety_signals = []

        if gold_1d is None or risk_proxy is None:
            return safety_signals

        safety_score = gold_flight_to_safety(gold_1d, risk_proxy)
        if safety_score > 0.3:
            signals.append(MacroSignal(
                name="FLIGHT_TO_SAFETY",
                value=safety_score,
                direction="bearish",
                confidence=0.7,
                timeframe="1D",
                metadata={"gold_outperforming": True, "hedging_mode": True}
            ))
            notes.append(f"Gold flight-to-safety {safety_score:.2f}")

        return safety_signals

    def _analyze_bond_stress(self, us2y: Optional[pd.DataFrame], us10y: Optional[pd.DataFrame],
                           signals: List[MacroSignal], notes: List[str]) -> List[MacroSignal]:
        """Analyze yield curve and bond market stress"""
        bond_signals = []

        if us2y is None or us10y is None:
            return bond_signals

        # Yield spikes
        spike_strength = yields_spike(us2y, us10y, self.yields_spike_std)
        if spike_strength > 0.4:
            signals.append(MacroSignal(
                name="YIELD_SPIKE",
                value=spike_strength,
                direction="bearish",
                confidence=0.85,
                timeframe="1D",
                metadata={"fed_tightening": True, "liquidity_drain": True}
            ))
            notes.append(f"Yield spike {spike_strength:.2f}")

        # Yield curve inversion check
        if len(us2y) > 0 and len(us10y) > 0:
            spread = us10y['close'].iloc[-1] - us2y['close'].iloc[-1]
            if spread < -0.001:  # Inverted by >10bp
                signals.append(MacroSignal(
                    name="YIELD_INVERSION",
                    value=abs(spread) * 100,  # Convert to basis points
                    direction="bearish",
                    confidence=0.8,
                    timeframe="1D",
                    metadata={"recession_risk": True, "inversion_bp": spread * 10000}
                ))
                notes.append(f"Yield curve inverted {spread*10000:.0f}bp")

        return bond_signals

    def _analyze_volatility_regime(self, vix: Optional[pd.DataFrame], move: Optional[pd.DataFrame],
                                 signals: List[MacroSignal], notes: List[str]) -> List[MacroSignal]:
        """Analyze VIX/MOVE volatility spikes"""
        vol_signals = []

        if vix is None and move is None:
            return vol_signals

        spike_strength = vix_move_spike(vix, move, self.vix_spike, self.move_spike)
        if spike_strength > 0.5:
            signals.append(MacroSignal(
                name="VOLATILITY_SPIKE",
                value=spike_strength,
                direction="bearish",
                confidence=0.9,
                timeframe="1D",
                metadata={"risk_off": True, "capital_protection": True}
            ))
            notes.append(f"VIX/MOVE spike {spike_strength:.2f}")

        return vol_signals

    def _analyze_systemic_risk(self, usdjpy: Optional[pd.DataFrame], hyg: Optional[pd.DataFrame],
                             signals: List[MacroSignal], notes: List[str]) -> List[MacroSignal]:
        """Analyze systemic risk via carry trades and credit"""
        systemic_signals = []

        # USDJPY carry unwind
        if usdjpy is not None:
            carry_break = usd_jpy_carry_break(usdjpy, self.usd_jpy_level)
            if carry_break:
                signals.append(MacroSignal(
                    name="CARRY_UNWIND",
                    value=0.8,
                    direction="bearish",
                    confidence=0.85,
                    timeframe="1D",
                    metadata={"yen_carry": True, "systemic_risk": True}
                ))
                notes.append("USDJPY carry unwind")

        # HYG credit stress
        if hyg is not None:
            credit_stress = hyg_credit_stress(hyg, self.hyg_break_len)
            if credit_stress > 0.5:
                signals.append(MacroSignal(
                    name="CREDIT_STRESS",
                    value=credit_stress,
                    direction="bearish",
                    confidence=0.8,
                    timeframe="1D",
                    metadata={"junk_bonds": True, "credit_contagion": True}
                ))
                notes.append(f"HYG credit stress {credit_stress:.2f}")

        return systemic_signals

    def _analyze_crypto_rotation(self, usdt: Optional[pd.DataFrame], total3: Optional[pd.DataFrame],
                               total: Optional[pd.DataFrame], ethbtc: Optional[pd.DataFrame],
                               ethd: Optional[pd.DataFrame], signals: List[MacroSignal],
                               notes: List[str], plus_ones: List[str]) -> List[MacroSignal]:
        """Analyze crypto-specific rotation signals"""
        rotation_signals = []

        # USDT.D SFP/Wolfe analysis
        if usdt is not None:
            sfp_score = usdt_sfp_wolfe(usdt, self.usdt_sfp_lookback)
            if sfp_score > 0.5:
                signals.append(MacroSignal(
                    name="USDT_SFP",
                    value=sfp_score,
                    direction="bearish",
                    confidence=0.7,
                    timeframe="4H",
                    metadata={"false_breakout": True, "trap_setup": True}
                ))
                notes.append("USDT.D SFP trap")

        # TOTAL3 vs TOTAL divergence
        if total3 is not None and total is not None:
            alt_impulse = total3_vs_total(total3, total)
            if abs(alt_impulse) > 0.25:
                direction = "bullish" if alt_impulse > 0 else "bearish"
                signals.append(MacroSignal(
                    name="TOTAL3_DIVERGENCE",
                    value=abs(alt_impulse),
                    direction=direction,
                    confidence=0.75,
                    timeframe="4H",
                    metadata={"alt_rotation": alt_impulse > 0, "alt_bleed": alt_impulse < 0}
                ))
                if alt_impulse > 0:
                    plus_ones.append("TOTAL3 alt leadership")

        # ETH/BTC trend gate
        if ethbtc is not None:
            eth_trend = ethbtc_trend_gate(ethbtc, self.ethbtc_ma_length)
            if eth_trend != 'neutral':
                signals.append(MacroSignal(
                    name="ETHBTC_TREND",
                    value=0.6,
                    direction="bullish" if eth_trend == 'up' else "bearish",
                    confidence=0.7,
                    timeframe="1D",
                    metadata={"eth_trend": eth_trend, "alt_season": eth_trend == 'up'}
                ))
                if eth_trend == 'up':
                    plus_ones.append("ETH/BTC trend gate up")

        # ETH dominance gate
        if ethd is not None:
            ethd_trend = ethd_gate(ethd)
            if ethd_trend != 'neutral':
                signals.append(MacroSignal(
                    name="ETH_DOMINANCE",
                    value=0.5,
                    direction="bullish" if ethd_trend == 'up' else "bearish",
                    confidence=0.6,
                    timeframe="1D",
                    metadata={"ethd_trend": ethd_trend}
                ))

        return rotation_signals

    def _calculate_macro_aggregates(self, signals: List[MacroSignal]) -> Tuple[float, float, MacroRegime]:
        """Calculate aggregate veto/boost strength and determine regime"""
        veto_scores = []
        boost_scores = []

        # Categorize signals
        for signal in signals:
            if signal.direction == "bearish":
                veto_scores.append(signal.value * signal.confidence)
            elif signal.direction == "bullish":
                boost_scores.append(signal.value * signal.confidence)

        # Calculate aggregates
        veto_strength = max(veto_scores) if veto_scores else 0.0
        boost_strength = sum(boost_scores) if boost_scores else 0.0
        boost_strength = min(boost_strength, self.boost_cap)

        # Determine regime
        regime = self._classify_macro_regime(signals, veto_strength, boost_strength)

        return veto_strength, boost_strength, regime

    def _classify_macro_regime(self, signals: List[MacroSignal], veto: float, boost: float) -> MacroRegime:
        """Classify current macro regime"""
        signal_names = [s.name for s in signals]

        # Stagflation check
        if "STAGFLATION_VETO" in signal_names:
            return MacroRegime.STAGFLATION

        # Risk-off conditions
        risk_off_signals = ["DXY_BREAKOUT", "YIELD_SPIKE", "VOLATILITY_SPIKE", "CARRY_UNWIND", "CREDIT_STRESS"]
        if any(name in signal_names for name in risk_off_signals) and veto > 0.6:
            return MacroRegime.RISK_OFF

        # Risk-on conditions
        risk_on_signals = ["DXY_BREAKDOWN", "TOTAL3_DIVERGENCE", "ETHBTC_TREND"]
        if any(name in signal_names for name in risk_on_signals) and boost > 0.05:
            return MacroRegime.RISK_ON

        return MacroRegime.NEUTRAL

    def _determine_risk_bias(self, signals: List[MacroSignal], boost: float, regime: MacroRegime) -> Tuple[str, float]:
        """Determine risk bias and macro delta"""
        signal_names = [s.name for s in signals]

        # Risk bias
        if regime == MacroRegime.RISK_ON:
            risk_bias = "risk_on"
        elif regime in [MacroRegime.RISK_OFF, MacroRegime.STAGFLATION]:
            risk_bias = "risk_off"
        else:
            risk_bias = "neutral"

        # Macro delta calculation
        macro_delta = 0.0
        if risk_bias == "risk_on":
            macro_delta = min(boost, self.weights['boost_context_max'])
        elif risk_bias == "risk_off":
            macro_delta = -min(boost, self.weights['boost_context_max'])

        # Additional boost factors
        if "DXY_BREAKDOWN" in signal_names:
            macro_delta += self.weights['boost_macro']
        if "TOTAL3_DIVERGENCE" in signal_names and risk_bias == "risk_on":
            macro_delta += self.weights['boost_macro']

        # Bound the delta
        macro_delta = np.clip(macro_delta, -self.weights['boost_context_max'], self.weights['boost_context_max'])

        return risk_bias, float(macro_delta)

    def _check_dxy_breakdown(self, dxy_1d: pd.DataFrame) -> float:
        """Check for DXY breakdown (bullish for risk assets)"""
        if len(dxy_1d) < 20:
            return 0.0

        # Look for sustained downtrend
        recent_change = dxy_1d['close'].pct_change(10).iloc[-1]
        short_change = dxy_1d['close'].pct_change(5).iloc[-1]

        if recent_change < -0.02 and short_change < -0.01:
            return min(1.0, abs(recent_change) / 0.05)

        return 0.0

    def _default_macro_pulse(self) -> MacroPulse:
        """Return default neutral macro pulse"""
        return MacroPulse(
            regime=MacroRegime.NEUTRAL,
            veto_strength=0.0,
            boost_strength=0.0,
            suppression_flag=False,
            risk_bias="neutral",
            macro_delta=0.0,
            active_signals=[],
            notes=[],
            plus_ones=[]
        )