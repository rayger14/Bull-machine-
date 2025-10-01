"""
Wyckoff Phase Detection Engine

Implements safer Wyckoff phase detection with volume guards and USDT stagnation integration.
Rejects fake SC/AR if relative volume is too low vs rolling mean.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum
import logging

logger = logging.getLogger(__name__)

class WyckoffPhase(Enum):
    """Wyckoff market phases"""
    ACCUMULATION = "accumulation"    # Phase A-E accumulation
    DISTRIBUTION = "distribution"    # Phase A-E distribution
    MARKUP = "markup"               # Bullish trend
    MARKDOWN = "markdown"           # Bearish trend
    REACCUMULATION = "reaccumulation"  # Phase B in uptrend
    REDISTRIBUTION = "redistribution"  # Phase B in downtrend
    SPRING = "spring"               # Final shakeout before markup
    UPTHRUST = "upthrust"          # Final trap before markdown
    NEUTRAL = "neutral"            # No clear phase

@dataclass
class WyckoffSignal:
    """Wyckoff phase detection signal"""
    timestamp: pd.Timestamp
    phase: WyckoffPhase
    confidence: float  # 0-1
    direction: str     # 'long', 'short', 'neutral'
    strength: float    # 0-1

    # Phase-specific data
    volume_quality: float  # Volume confirmation
    price_structure: Dict[str, Any]  # Support/resistance levels
    crt_active: bool   # Composite Reaccumulation Time

    metadata: Dict[str, Any]

def detect_wyckoff_phase(df: pd.DataFrame, cfg: Dict, usdt_stag_strength: float) -> Dict:
    """
    Detect Wyckoff phase with volume guards and safer logic.

    Args:
        df: OHLCV price data
        cfg: Configuration parameters
        usdt_stag_strength: USDT stagnation strength from macro pulse (0-1)

    Returns:
        Dict with phase, confidence, and additional metadata
    """
    try:
        if len(df) < 50:
            return {"phase": None, "confidence": 0.0, "reason": "insufficient_data"}

        # Basic phase detection (simplified implementation)
        phase, conf = _basic_phase_logic(df, cfg)

        # Reject fake SC/AR if relative volume is too low vs its own rolling
        v = df.get("volume")
        if phase in ("SC", "AR") and v is not None and len(v) > 50:
            rel = v.iloc[-1] / max(1e-9, v.rolling(50).mean().iloc[-1])
            if rel < cfg.get("sc_ar_vol_min", 0.8):  # 0.8x mean
                return {"phase": None, "confidence": 0.0, "reason": "low_vol_trap"}

        # CRT in SMR (B) — add a strong "time-in-range" + coil check
        crt_active, hps_score = crt_smr_check(df, cfg, usdt_stag_strength)

        if phase == "B" and crt_active:
            # Add ~3% confidence boost when Phase B/C context is present
            hps_confidence_boost = min(0.03, hps_score * 0.06)
            return {
                "phase": "B",
                "confidence": max(conf, 0.9) + hps_confidence_boost,
                "crt_active": True,
                "hps_score": hps_score
            }

        return {
            "phase": phase,
            "confidence": conf,
            "crt_active": False,
            "hps_score": hps_score if phase in ["B", "C"] else 0.0
        }

    except Exception as e:
        logger.error(f"Error in Wyckoff phase detection: {e}")
        return {"phase": None, "confidence": 0.0, "reason": "error"}

def crt_smr_check(df: pd.DataFrame, cfg: Dict, usdt_stag_strength: float) -> Tuple[bool, float]:
    """
    Composite Re-accumulation Time (CRT) check in Smart Money Range (SMR).

    Low realized volatility + USDT.D coil → composite re-accumulation time

    Returns:
        Tuple of (crt_active, hps_score)
    """
    try:
        # Check for low realized volatility
        vol_std = df["close"].pct_change().rolling(24).std().iloc[-1]
        if vol_std is None or np.isnan(vol_std):
            return False, 0.0

        vol_threshold = cfg.get("crt_vol_std_max", 0.005)  # 0.5% daily volatility
        usdt_threshold = cfg.get("crt_usdt_stag_min", 0.7)
        hps_floor = cfg.get("hps_floor", 0.5)

        # Calculate HPS (High Probability Setup) score
        vol_component = max(0, 1 - (vol_std / vol_threshold)) if vol_threshold > 0 else 0
        usdt_component = min(1.0, usdt_stag_strength)
        hps_score = (vol_component + usdt_component) / 2

        # Only activate CRT if both conditions met and HPS above floor
        crt_active = (vol_std < vol_threshold and
                     usdt_stag_strength >= usdt_threshold and
                     hps_score >= hps_floor)

        return crt_active, hps_score

    except Exception as e:
        logger.error(f"Error in CRT SMR check: {e}")
        return False, 0.0

def _basic_phase_logic(df: pd.DataFrame, cfg: Dict) -> Tuple[Optional[str], float]:
    """
    Basic Wyckoff phase detection logic.
    This is a simplified implementation - real Wyckoff analysis is much more complex.
    """
    try:
        # Calculate basic metrics
        recent_data = df.tail(20)
        price_range = recent_data['high'].max() - recent_data['low'].min()
        current_price = df['close'].iloc[-1]

        # Volume analysis
        avg_volume = df['volume'].rolling(20).mean().iloc[-1]
        recent_volume = df['volume'].iloc[-1]

        # Simplified phase detection based on price action and volume
        vol_ratio = recent_volume / max(1e-9, avg_volume)

        # Price position in recent range
        range_position = (current_price - recent_data['low'].min()) / max(1e-9, price_range)

        # Very basic phase classification
        if vol_ratio > 1.5 and range_position < 0.3:
            return "accumulation", 0.6
        elif vol_ratio > 1.5 and range_position > 0.7:
            return "distribution", 0.6
        elif range_position < 0.2 and vol_ratio < 0.8:
            return "spring", 0.5
        elif range_position > 0.8 and vol_ratio < 0.8:
            return "upthrust", 0.5
        elif 0.4 <= range_position <= 0.6:
            return "B", 0.4  # Consolidation phase
        else:
            return None, 0.0

    except Exception as e:
        logger.error(f"Error in basic phase logic: {e}")
        return None, 0.0

class WyckoffEngine:
    """
    Wyckoff Phase Detection Engine

    Detects market phases according to Wyckoff methodology with enhanced
    volume validation and macro context integration.
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.volume_threshold = config.get('sc_ar_vol_min', 0.8)
        self.crt_vol_threshold = config.get('crt_vol_std_max', 0.005)
        self.crt_usdt_threshold = config.get('crt_usdt_stag_min', 0.7)

    def analyze(self, data: pd.DataFrame, usdt_stagnation: float = 0.0) -> Optional[WyckoffSignal]:
        """
        Analyze price data for Wyckoff phases.

        Args:
            data: OHLCV price data
            usdt_stagnation: USDT stagnation strength from macro context

        Returns:
            WyckoffSignal if phase detected, None otherwise
        """
        try:
            result = detect_wyckoff_phase(data, self.config, usdt_stagnation)

            if result["phase"] is None:
                return None

            # Convert to enum
            phase_map = {
                "accumulation": WyckoffPhase.ACCUMULATION,
                "distribution": WyckoffPhase.DISTRIBUTION,
                "B": WyckoffPhase.REACCUMULATION,
                "spring": WyckoffPhase.SPRING,
                "upthrust": WyckoffPhase.UPTHRUST
            }

            phase = phase_map.get(result["phase"], WyckoffPhase.NEUTRAL)

            # Determine direction
            if phase in [WyckoffPhase.ACCUMULATION, WyckoffPhase.SPRING, WyckoffPhase.REACCUMULATION]:
                direction = 'long'
                strength = result["confidence"] * 0.8
            elif phase in [WyckoffPhase.DISTRIBUTION, WyckoffPhase.UPTHRUST, WyckoffPhase.REDISTRIBUTION]:
                direction = 'short'
                strength = result["confidence"] * 0.8
            else:
                direction = 'neutral'
                strength = 0.0

            return WyckoffSignal(
                timestamp=data.index[-1],
                phase=phase,
                confidence=result["confidence"],
                direction=direction,
                strength=strength,
                volume_quality=self._calculate_volume_quality(data),
                price_structure=self._analyze_price_structure(data),
                crt_active=result.get("crt_active", False),
                metadata={
                    "reason": result.get("reason", ""),
                    "usdt_stagnation": usdt_stagnation,
                    "volume_threshold": self.volume_threshold
                }
            )

        except Exception as e:
            logger.error(f"Error in Wyckoff analysis: {e}")
            return None

    def _calculate_volume_quality(self, data: pd.DataFrame) -> float:
        """Calculate volume quality score"""
        try:
            if 'volume' not in data.columns or len(data) < 20:
                return 0.0

            recent_vol = data['volume'].iloc[-5:].mean()
            avg_vol = data['volume'].rolling(20).mean().iloc[-1]

            return min(1.0, recent_vol / max(1e-9, avg_vol))

        except Exception as e:
            logger.error(f"Error calculating volume quality: {e}")
            return 0.0

    def _analyze_price_structure(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze price structure for support/resistance levels"""
        try:
            recent_data = data.tail(20)
            return {
                "support": recent_data['low'].min(),
                "resistance": recent_data['high'].max(),
                "midpoint": (recent_data['low'].min() + recent_data['high'].max()) / 2,
                "range_size": recent_data['high'].max() - recent_data['low'].min()
            }
        except Exception as e:
            logger.error(f"Error analyzing price structure: {e}")
            return {}