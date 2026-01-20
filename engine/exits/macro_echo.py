"""
Macro Echo Rules

Correlation-based macro regime detection from @TheAstronomer's framework.
When traditional finance assets (DXY, Oil, Yields) move in specific patterns,
crypto typically "echoes" with predictable responses.

Key Correlations:
- DXY UP + Yields UP = Risk-off (crypto DOWN)
- DXY DOWN + Oil UP = Risk-on (crypto UP)
- Yields spike (>10% weekly) = Flight to safety (crypto DOWN)
- VIX > 30 = Fear regime (crypto volatile DOWN)

Author: Bull Machine v2.0
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional
from dataclasses import dataclass


@dataclass
class MacroEchoSignal:
    """
    Macro echo correlation signal.

    Attributes:
        regime: 'risk_on' | 'risk_off' | 'neutral' | 'crisis'
        dxy_trend: 'up' | 'down' | 'flat'
        yields_trend: 'up' | 'down' | 'flat'
        oil_trend: 'up' | 'down' | 'flat'
        vix_level: 'low' | 'medium' | 'high' | 'extreme'
        correlation_score: -1 to 1 (negative = bearish for crypto)
        exit_recommended: True if macro suggests exit
    """
    regime: str
    dxy_trend: str
    yields_trend: str
    oil_trend: str
    vix_level: str
    correlation_score: float
    exit_recommended: bool

    def to_dict(self) -> Dict:
        """Convert to dictionary for feature store."""
        return {
            'macro_regime': self.regime,
            'macro_dxy_trend': self.dxy_trend,
            'macro_yields_trend': self.yields_trend,
            'macro_oil_trend': self.oil_trend,
            'macro_vix_level': self.vix_level,
            'macro_correlation_score': self.correlation_score,
            'macro_exit_recommended': self.exit_recommended,
        }


def calculate_trend(series: pd.Series, lookback: int = 7) -> str:
    """
    Calculate trend direction for macro series.

    Args:
        series: Price/value series
        lookback: Bars to analyze

    Returns:
        'up' | 'down' | 'flat'
    """
    if len(series) < lookback:
        return 'flat'

    recent = series.tail(lookback)

    # FIX: Handle NaN values - default to 'flat' when data missing
    if recent.isna().all():
        return 'flat'  # All data is NaN

    start_val = recent.iloc[0]
    end_val = recent.iloc[-1]

    # FIX: Check if start_val or end_val is NaN
    if pd.isna(start_val) or pd.isna(end_val):
        return 'flat'

    change_pct = (end_val - start_val) / start_val if start_val > 0 else 0

    if change_pct > 0.02:  # >2% increase
        return 'up'
    elif change_pct < -0.02:  # >2% decrease
        return 'down'
    else:
        return 'flat'


def classify_vix_level(vix_value: float) -> str:
    """
    Classify VIX fear level.

    Args:
        vix_value: VIX index value

    Returns:
        'low' | 'medium' | 'high' | 'extreme'
    """
    # FIX: Handle NaN values - default to 'medium' when VIX unavailable
    if pd.isna(vix_value):
        return 'medium'  # Neutral when data missing

    if vix_value < 15:
        return 'low'  # Complacency
    elif vix_value < 25:
        return 'medium'  # Normal
    elif vix_value < 35:
        return 'high'  # Elevated fear
    else:
        return 'extreme'  # Panic


def detect_macro_regime(dxy_trend: str, yields_trend: str,
                        oil_trend: str, vix_level: str) -> str:
    """
    Detect overall macro regime.

    Args:
        dxy_trend: Dollar trend
        yields_trend: Treasury yields trend
        oil_trend: Oil price trend
        vix_level: VIX fear level

    Returns:
        'risk_on' | 'risk_off' | 'neutral' | 'crisis'

    Logic:
        - Crisis: VIX extreme OR (DXY up + Yields spiking)
        - Risk-off: DXY up + Yields up (dollar strength = crypto weakness)
        - Risk-on: DXY down + Oil up (weak dollar = crypto strength)
        - Neutral: Mixed signals
    """
    # Crisis detection
    if vix_level == 'extreme':
        return 'crisis'

    if dxy_trend == 'up' and yields_trend == 'up' and vix_level == 'high':
        return 'crisis'

    # Risk-off: Dollar strength
    if dxy_trend == 'up' and yields_trend == 'up':
        return 'risk_off'

    if dxy_trend == 'up' and vix_level in ['high', 'extreme']:
        return 'risk_off'

    # Risk-on: Dollar weakness + commodities strong
    if dxy_trend == 'down' and oil_trend == 'up':
        return 'risk_on'

    if dxy_trend == 'down' and yields_trend == 'down' and vix_level == 'low':
        return 'risk_on'

    # Neutral
    return 'neutral'


def calculate_correlation_score(dxy_trend: str, yields_trend: str,
                                 oil_trend: str, vix_level: str) -> float:
    """
    Calculate overall correlation score for crypto.

    Args:
        dxy_trend: Dollar trend
        yields_trend: Yields trend
        oil_trend: Oil trend
        vix_level: VIX level

    Returns:
        Score from -1 (very bearish) to +1 (very bullish)

    Scoring:
        - DXY up: -0.30
        - DXY down: +0.30
        - Yields up: -0.20
        - Yields down: +0.20
        - Oil up: +0.25
        - Oil down: -0.25
        - VIX extreme: -0.25
        - VIX low: +0.25
    """
    score = 0.0

    # DXY impact (30% weight)
    if dxy_trend == 'down':
        score += 0.30
    elif dxy_trend == 'up':
        score -= 0.30

    # Yields impact (20% weight)
    if yields_trend == 'down':
        score += 0.20
    elif yields_trend == 'up':
        score -= 0.20

    # Oil impact (25% weight)
    if oil_trend == 'up':
        score += 0.25
    elif oil_trend == 'down':
        score -= 0.25

    # VIX impact (25% weight)
    if vix_level == 'low':
        score += 0.25
    elif vix_level == 'high':
        score -= 0.15
    elif vix_level == 'extreme':
        score -= 0.25

    return float(np.clip(score, -1.0, 1.0))


def analyze_macro_echo(macro_data: Dict, lookback: int = 7,
                       config: Optional[Dict] = None) -> MacroEchoSignal:
    """
    Analyze macro echo correlations.

    Args:
        macro_data: Dictionary with macro series:
            {
                'DXY': pd.Series,
                'YIELDS_10Y': pd.Series,
                'OIL': pd.Series,
                'VIX': pd.Series
            }
        lookback: Days to analyze for trends
        config: Optional configuration

    Returns:
        MacroEchoSignal with regime and exit recommendation

    Example:
        >>> macro_signal = analyze_macro_echo({
        ...     'DXY': dxy_series,
        ...     'YIELDS_10Y': yields_series,
        ...     'OIL': oil_series,
        ...     'VIX': vix_series
        ... })
        >>> if macro_signal.exit_recommended:
        ...     print(f"Macro regime {macro_signal.regime} suggests exit")
    """
    config = config or {}

    # Extract series
    dxy_series = macro_data.get('DXY', pd.Series([100.0]))
    yields_series = macro_data.get('YIELDS_10Y', pd.Series([4.0]))
    oil_series = macro_data.get('OIL', pd.Series([75.0]))
    vix_series = macro_data.get('VIX', pd.Series([18.0]))

    # Calculate trends
    dxy_trend = calculate_trend(dxy_series, lookback=lookback)
    yields_trend = calculate_trend(yields_series, lookback=lookback)
    oil_trend = calculate_trend(oil_series, lookback=lookback)

    # Classify VIX
    current_vix = vix_series.iloc[-1] if len(vix_series) > 0 else 18.0
    vix_level = classify_vix_level(current_vix)

    # Detect regime
    regime = detect_macro_regime(dxy_trend, yields_trend, oil_trend, vix_level)

    # Calculate correlation score
    correlation_score = calculate_correlation_score(
        dxy_trend, yields_trend, oil_trend, vix_level
    )

    # Exit recommendation
    exit_threshold = config.get('macro_exit_threshold', -0.5)
    exit_recommended = (
        regime in ['risk_off', 'crisis'] or
        correlation_score < exit_threshold
    )

    return MacroEchoSignal(
        regime=regime,
        dxy_trend=dxy_trend,
        yields_trend=yields_trend,
        oil_trend=oil_trend,
        vix_level=vix_level,
        correlation_score=correlation_score,
        exit_recommended=exit_recommended
    )


def apply_macro_echo_adjustment(fusion_score: float, macro_signal: MacroEchoSignal,
                                 config: Optional[Dict] = None) -> tuple:
    """
    Apply macro echo fusion adjustment.

    Args:
        fusion_score: Current fusion score
        macro_signal: MacroEchoSignal from analyze_macro_echo()
        config: Optional config

    Returns:
        (adjusted_score: float, adjustment: float, reasons: list)

    Logic:
        - Risk-on regime: +0.05
        - Risk-off regime: -0.10
        - Crisis regime: -0.20 (strong warning)
        - Correlation score: +/-0.10 scaled adjustment
    """
    config = config or {}
    adjustment = 0.0
    reasons = []

    # Regime adjustment
    if macro_signal.regime == 'crisis':
        adjustment -= 0.20
        reasons.append(f"Macro crisis regime (VIX={macro_signal.vix_level})")
    elif macro_signal.regime == 'risk_off':
        adjustment -= 0.10
        reasons.append(f"Macro risk-off (DXY={macro_signal.dxy_trend})")
    elif macro_signal.regime == 'risk_on':
        adjustment += 0.05
        reasons.append(f"Macro risk-on (DXY={macro_signal.dxy_trend}, Oil={macro_signal.oil_trend})")

    # Correlation score adjustment (scaled to +/-0.10)
    corr_adjustment = macro_signal.correlation_score * 0.10
    adjustment += corr_adjustment

    if abs(corr_adjustment) > 0.05:
        reasons.append(f"Macro correlation score: {macro_signal.correlation_score:.2f}")

    # Apply adjustment
    adjusted_score = max(0.0, min(fusion_score + adjustment, 1.0))

    return adjusted_score, adjustment, reasons
