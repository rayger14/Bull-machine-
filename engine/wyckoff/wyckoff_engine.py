"""
Wyckoff Phase Detection Engine

Implements safer Wyckoff phase detection with volume guards and USDT stagnation integration.
Rejects fake SC/AR if relative volume is too low vs rolling mean.

Enhanced with institutional-grade event detection (18 Wyckoff events) and PTI integration.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum
import logging

logger = logging.getLogger(__name__)

# Import Wyckoff event detection system
try:
    from engine.wyckoff.events import (
        detect_all_wyckoff_events,
        integrate_wyckoff_with_pti
    )
    WYCKOFF_EVENTS_AVAILABLE = True
except ImportError:
    logger.warning("Wyckoff events module not available. Using basic phase detection only.")
    WYCKOFF_EVENTS_AVAILABLE = False

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

        # Enhanced phase classification with trend detection

        # Add trend analysis (SMA crossovers for markup/markdown)
        if len(df) >= 50:
            sma_20 = df['close'].rolling(20).mean().iloc[-1]
            sma_50 = df['close'].rolling(50).mean().iloc[-1]

            # Strong uptrend = markup
            if sma_20 > sma_50 * 1.02 and current_price > sma_20:
                return "markup", 0.7
            # Strong downtrend = markdown
            elif sma_20 < sma_50 * 0.98 and current_price < sma_20:
                return "markdown", 0.7

        # High volume phases
        if vol_ratio > 1.5 and range_position < 0.3:
            return "accumulation", 0.6
        elif vol_ratio > 1.5 and range_position > 0.7:
            return "distribution", 0.6

        # Springs and upthrusts
        elif range_position < 0.2 and vol_ratio < 0.8:
            return "spring", 0.5
        elif range_position > 0.8 and vol_ratio < 0.8:
            return "upthrust", 0.5

        # Consolidation/reaccumulation
        elif 0.4 <= range_position <= 0.6:
            return "B", 0.4

        # Weaker accumulation/distribution signals
        elif range_position < 0.4 and vol_ratio > 1.0:
            return "accumulation", 0.4
        elif range_position > 0.6 and vol_ratio > 1.0:
            return "distribution", 0.4

        # Default: transition (but with low confidence, not None)
        else:
            return "transition", 0.3

    except Exception as e:
        logger.error(f"Error in basic phase logic: {e}")
        return None, 0.0

class WyckoffEngine:
    """
    Wyckoff Phase Detection Engine

    Detects market phases according to Wyckoff methodology with enhanced
    volume validation and macro context integration.

    Enhanced Features:
        - Basic phase detection (accumulation/distribution/markup/markdown)
        - Institutional-grade event detection (18 Wyckoff events)
        - PTI integration for psychological trap detection
        - Sequence tracking (position within Wyckoff cycle)
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.volume_threshold = config.get('sc_ar_vol_min', 0.8)
        self.crt_vol_threshold = config.get('crt_vol_std_max', 0.005)
        self.crt_usdt_threshold = config.get('crt_usdt_stag_min', 0.7)

        # Wyckoff events configuration
        self.wyckoff_events_config = config.get('wyckoff_events', {})
        self.enable_events = self.wyckoff_events_config.get('enabled', False)
        self.enable_pti_integration = self.wyckoff_events_config.get('pti_integration', True)

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

    def detect_wyckoff_events(
        self,
        data: pd.DataFrame,
        pti_scores: Optional[pd.Series] = None
    ) -> pd.DataFrame:
        """
        Detect all Wyckoff events and add to dataframe.

        This is the main entry point for institutional-grade Wyckoff event detection.
        It wraps the events.py module and integrates with PTI if available.

        Args:
            data: OHLCV dataframe with technical indicators
            pti_scores: Optional PTI scores for integration

        Returns:
            DataFrame with Wyckoff event columns added

        New Columns Added:
            - wyckoff_sc, wyckoff_sc_confidence (Selling Climax)
            - wyckoff_bc, wyckoff_bc_confidence (Buying Climax)
            - wyckoff_ar, wyckoff_ar_confidence (Automatic Rally)
            - wyckoff_as, wyckoff_as_confidence (Automatic Reaction)
            - wyckoff_st, wyckoff_st_confidence (Secondary Test)
            - wyckoff_sos, wyckoff_sos_confidence (Sign of Strength)
            - wyckoff_sow, wyckoff_sow_confidence (Sign of Weakness)
            - wyckoff_lps, wyckoff_lps_confidence (Last Point of Support)
            - wyckoff_lpsy, wyckoff_lpsy_confidence (Last Point of Supply)
            - wyckoff_spring_a, wyckoff_spring_a_confidence (Spring Type A)
            - wyckoff_spring_b, wyckoff_spring_b_confidence (Spring Type B)
            - wyckoff_ut, wyckoff_ut_confidence (Upthrust)
            - wyckoff_utad, wyckoff_utad_confidence (UTAD)
            - wyckoff_phase_abc (Phase: A/B/C/D/E/neutral)
            - wyckoff_sequence_position (Position: 1-10)
            - wyckoff_pti_confluence (PTI + Wyckoff trap confluence)
            - wyckoff_pti_score (Composite trap score)

        Example:
            >>> engine = WyckoffEngine(config)
            >>> df = engine.detect_wyckoff_events(df_1h)
            >>> df[df['wyckoff_sc']].iloc[-5:]  # Last 5 Selling Climax events
        """
        if not self.enable_events:
            logger.info("Wyckoff events disabled in config")
            return data

        if not WYCKOFF_EVENTS_AVAILABLE:
            logger.error("Wyckoff events module not available")
            return data

        try:
            logger.info(f"Detecting Wyckoff events on {len(data)} bars")

            # Detect all events
            data = detect_all_wyckoff_events(data, self.wyckoff_events_config)

            # Integrate with PTI if enabled
            if self.enable_pti_integration and pti_scores is not None:
                data = integrate_wyckoff_with_pti(data, pti_scores)
            elif self.enable_pti_integration and 'pti_score' in data.columns:
                data = integrate_wyckoff_with_pti(data)

            logger.info("Wyckoff event detection complete")
            return data

        except Exception as e:
            logger.error(f"Error in Wyckoff event detection: {e}")
            import traceback
            traceback.print_exc()
            return data

    def get_wyckoff_sequence_context(self, data: pd.DataFrame, current_idx: int) -> Dict[str, Any]:
        """
        Get context about current position in Wyckoff sequence.

        This helps understand where we are in the accumulation/distribution cycle
        and what events to expect next.

        Args:
            data: DataFrame with Wyckoff event columns
            current_idx: Current bar index

        Returns:
            Dict with:
                - current_phase: Current Wyckoff phase (A/B/C/D/E)
                - sequence_position: Position in sequence (1-10)
                - recent_events: List of recent events (last 20 bars)
                - next_expected: List of likely next events
                - cycle_progress: Estimated progress through cycle (0-1)

        Example:
            >>> context = engine.get_wyckoff_sequence_context(df, -1)
            >>> print(f"Phase: {context['current_phase']}")
            >>> print(f"Recent events: {context['recent_events']}")
        """
        if current_idx < 0:
            current_idx = len(data) + current_idx

        if 'wyckoff_phase_abc' not in data.columns:
            return {
                'current_phase': 'neutral',
                'sequence_position': 0,
                'recent_events': [],
                'next_expected': [],
                'cycle_progress': 0.0
            }

        # Get current phase and position
        current_phase = data['wyckoff_phase_abc'].iloc[current_idx]
        sequence_pos = int(data['wyckoff_sequence_position'].iloc[current_idx])

        # Find recent events (last 20 bars)
        lookback = min(20, current_idx + 1)
        recent_data = data.iloc[current_idx - lookback + 1:current_idx + 1]

        recent_events = []
        event_cols = [
            ('SC', 'wyckoff_sc'),
            ('BC', 'wyckoff_bc'),
            ('AR', 'wyckoff_ar'),
            ('AS', 'wyckoff_as'),
            ('ST', 'wyckoff_st'),
            ('SOS', 'wyckoff_sos'),
            ('SOW', 'wyckoff_sow'),
            ('Spring_A', 'wyckoff_spring_a'),
            ('Spring_B', 'wyckoff_spring_b'),
            ('UT', 'wyckoff_ut'),
            ('UTAD', 'wyckoff_utad'),
            ('LPS', 'wyckoff_lps'),
            ('LPSY', 'wyckoff_lpsy'),
        ]

        for event_name, event_col in event_cols:
            if event_col in recent_data.columns:
                event_indices = recent_data[recent_data[event_col]].index
                for idx in event_indices:
                    bars_ago = current_idx - data.index.get_loc(idx)
                    recent_events.append({
                        'event': event_name,
                        'bars_ago': bars_ago,
                        'confidence': recent_data.loc[idx, f"{event_col}_confidence"]
                    })

        # Determine next expected events based on phase
        next_expected = self._get_expected_next_events(current_phase, recent_events)

        # Estimate cycle progress
        cycle_progress = sequence_pos / 10.0

        return {
            'current_phase': current_phase,
            'sequence_position': sequence_pos,
            'recent_events': sorted(recent_events, key=lambda x: x['bars_ago']),
            'next_expected': next_expected,
            'cycle_progress': cycle_progress
        }

    def _get_expected_next_events(self, current_phase: str, recent_events: List[Dict]) -> List[str]:
        """
        Determine expected next events based on current phase and recent history.

        Args:
            current_phase: Current Wyckoff phase
            recent_events: List of recent event dicts

        Returns:
            List of expected event names
        """
        # Phase A → expect AR after SC, or ST after AR
        if current_phase == 'A':
            has_sc = any(e['event'] == 'SC' for e in recent_events)
            has_ar = any(e['event'] == 'AR' for e in recent_events)

            if has_sc and not has_ar:
                return ['AR']
            elif has_ar:
                return ['ST', 'SOS']
            else:
                return ['SC', 'BC']

        # Phase B → expect SOS/SOW, then consolidation
        elif current_phase == 'B':
            has_sos = any(e['event'] == 'SOS' for e in recent_events)
            has_sow = any(e['event'] == 'SOW' for e in recent_events)

            if not (has_sos or has_sow):
                return ['SOS', 'SOW']
            else:
                return ['Spring_A', 'Spring_B', 'UT']

        # Phase C → expect Springs/UTs before LPS/LPSY
        elif current_phase == 'C':
            return ['LPS', 'LPSY']

        # Phase D → expect trend beginning
        elif current_phase == 'D':
            return ['Markup', 'Markdown']

        # Phase E → trend continuation
        elif current_phase == 'E':
            return ['Continuation', 'Reversal_Warning']

        return []

    def detect_phase_for_feature_store(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Detect Wyckoff phase for each bar and add columns to dataframe.

        Called during feature engineering to pre-compute phases.

        Args:
            df: OHLCV dataframe

        Returns:
            df with added columns:
                - wyckoff_phase_abc
                - wyckoff_phase_confidence
        """
        try:
            logger.info(f"Computing Wyckoff phases for {len(df)} bars")

            # Initialize output columns with defaults
            df['wyckoff_phase_abc'] = 'neutral'
            df['wyckoff_phase_confidence'] = 0.0

            # Minimum history required for phase detection
            min_history = 50

            if len(df) < min_history:
                logger.warning(f"Insufficient data for Wyckoff phase detection: {len(df)} bars < {min_history}")
                return df

            # Process each bar with sufficient history
            phase_values = []
            confidence_values = []

            for i in range(len(df)):
                # First 50 bars remain neutral
                if i < min_history:
                    phase_values.append('neutral')
                    confidence_values.append(0.0)
                    continue

                # Get historical data up to current bar
                hist_data = df.iloc[:i+1].copy()

                try:
                    # Call analyze() with current historical data
                    # Note: usdt_stagnation defaults to 0.0 since we don't have it in feature store
                    signal = self.analyze(hist_data, usdt_stagnation=0.0)

                    if signal is not None:
                        # Map WyckoffPhase enum to string for ABC phases
                        phase_str = self._map_phase_to_abc(signal.phase)
                        phase_values.append(phase_str)
                        confidence_values.append(signal.confidence)
                    else:
                        # No signal detected, keep neutral
                        phase_values.append('neutral')
                        confidence_values.append(0.0)

                except Exception as e:
                    logger.debug(f"Error analyzing bar {i}: {e}")
                    phase_values.append('neutral')
                    confidence_values.append(0.0)

            # Assign computed values to dataframe
            df['wyckoff_phase_abc'] = phase_values
            df['wyckoff_phase_confidence'] = confidence_values

            # Log summary statistics
            phase_counts = df['wyckoff_phase_abc'].value_counts()
            logger.info(f"Wyckoff phase detection complete. Distribution: {phase_counts.to_dict()}")

            return df

        except Exception as e:
            logger.error(f"Error in detect_phase_for_feature_store: {e}")
            import traceback
            traceback.print_exc()

            # Return dataframe with neutral defaults if error occurs
            if 'wyckoff_phase_abc' not in df.columns:
                df['wyckoff_phase_abc'] = 'neutral'
            if 'wyckoff_phase_confidence' not in df.columns:
                df['wyckoff_phase_confidence'] = 0.0

            return df

    def _map_phase_to_abc(self, phase: WyckoffPhase) -> str:
        """
        Map WyckoffPhase enum to ABC phase string.

        Args:
            phase: WyckoffPhase enum value

        Returns:
            ABC phase string ('A', 'B', 'C', 'D', 'E', 'neutral', etc.)
        """
        # Map major phases to ABC equivalents following Wyckoff methodology:
        # Phase A: Preliminary support/resistance (accumulation/distribution begins)
        # Phase B: Building cause (reaccumulation/redistribution, consolidation)
        # Phase C: Testing phase (springs/upthrusts - final shakeouts)
        # Phase D: Trend beginning (not currently detected by basic phase logic)
        # Phase E: Trend continuation (markup/markdown - sustained trends)
        # neutral: No clear phase detected

        phase_mapping = {
            WyckoffPhase.ACCUMULATION: 'A',
            WyckoffPhase.DISTRIBUTION: 'A',
            WyckoffPhase.REACCUMULATION: 'B',
            WyckoffPhase.REDISTRIBUTION: 'B',
            WyckoffPhase.SPRING: 'C',
            WyckoffPhase.UPTHRUST: 'C',
            WyckoffPhase.MARKUP: 'E',      # Markup is sustained uptrend = Phase E
            WyckoffPhase.MARKDOWN: 'E',    # Markdown is sustained downtrend = Phase E
            WyckoffPhase.NEUTRAL: 'neutral'
        }

        return phase_mapping.get(phase, 'neutral')