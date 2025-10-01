"""
Macro Context Engine - SMT Signal Generation

Implements Smart Money Theory divergence detection across USDT.D, BTC.D, and TOTAL3
for macro market context and high-probability setup identification.
"""

import pandas as pd
import numpy as np
from enum import Enum
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)

class SMTSignalType(Enum):
    """SMT signal types for macro context analysis"""
    USDT_STAGNATION = "usdt_stagnation"
    BTC_WEDGE_BREAK = "btc_wedge_break"
    TOTAL3_DIVERGENCE = "total3_divergence"
    REGIME_SHIFT = "regime_shift"

class HPS_Score(Enum):
    """High-Probability Setup scoring (0-2)"""
    LOW = 0      # Single SMT signal
    MEDIUM = 1   # Two SMT signals aligned
    HIGH = 2     # All three SMT signals confirmed

@dataclass
class SMTSignal:
    """SMT signal data structure"""
    signal_type: SMTSignalType
    timestamp: pd.Timestamp
    confidence: float
    strength: float
    hps_score: HPS_Score
    suppression_active: bool
    metadata: Dict[str, Any]

class MacroContextEngine:
    """
    Macro Context Engine for SMT analysis and signal generation.

    Analyzes USDT.D, BTC.D, and TOTAL3 for:
    - USDT.D stagnation periods (36h+ range)
    - BTC.D wedge formations and breaks
    - TOTAL3 divergence from BTC price action
    - Regime shifts and suppression flags
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.smt_config = config.get('smt', {})

        # USDT.D Configuration
        self.usdt_stagnation_threshold = self.smt_config.get('usdt_stagnation_hours', 36)
        self.usdt_range_threshold = self.smt_config.get('usdt_range_pct', 0.002)  # 0.2%

        # BTC.D Configuration
        self.btc_wedge_min_touches = self.smt_config.get('btc_wedge_touches', 4)
        self.btc_wedge_angle_max = self.smt_config.get('btc_wedge_angle', 15)
        self.btc_break_volume_mult = self.smt_config.get('btc_break_volume', 1.5)

        # TOTAL3 Configuration
        self.total3_divergence_lookback = self.smt_config.get('total3_lookback', 168)  # 7 days
        self.total3_correlation_threshold = self.smt_config.get('total3_correlation', 0.7)

        # Signal Management
        self.min_hps_score = self.smt_config.get('min_hps_score', 1)
        self.suppression_cooldown = self.smt_config.get('suppression_cooldown', 24)

        # State tracking
        self.last_signals = {}
        self.suppression_state = {}

    def analyze_macro_context(self, data: Dict[str, pd.DataFrame]) -> List[SMTSignal]:
        """
        Analyze macro context and generate SMT signals.

        Args:
            data: Dict containing 'USDT.D', 'BTC.D', 'TOTAL3' DataFrames

        Returns:
            List of SMT signals with HPS scoring
        """
        signals = []

        try:
            # Extract macro data
            usdt_data = data.get('USDT.D')
            btc_dom_data = data.get('BTC.D')
            total3_data = data.get('TOTAL3')

            if not all([usdt_data is not None, btc_dom_data is not None, total3_data is not None]):
                logger.warning("Missing macro data for SMT analysis")
                return signals

            current_time = usdt_data.index[-1]

            # 1. USDT.D Stagnation Analysis
            usdt_signal = self._analyze_usdt_stagnation(usdt_data, current_time)
            if usdt_signal:
                signals.append(usdt_signal)

            # 2. BTC.D Wedge Break Analysis
            btc_signal = self._analyze_btc_wedge(btc_dom_data, current_time)
            if btc_signal:
                signals.append(btc_signal)

            # 3. TOTAL3 Divergence Analysis
            total3_signal = self._analyze_total3_divergence(total3_data, data.get('BTC'), current_time)
            if total3_signal:
                signals.append(total3_signal)

            # 4. Calculate HPS scores and apply suppression
            signals = self._calculate_hps_scores(signals, current_time)
            signals = self._apply_suppression_filters(signals, current_time)

            return signals

        except Exception as e:
            logger.error(f"Error in macro context analysis: {e}")
            return []

    def _analyze_usdt_stagnation(self, usdt_data: pd.DataFrame, current_time: pd.Timestamp) -> Optional[SMTSignal]:
        """Detect USDT.D stagnation periods (36h+ tight range)"""
        try:
            # Look back specified hours
            lookback_hours = self.usdt_stagnation_threshold
            lookback_time = current_time - pd.Timedelta(hours=lookback_hours)

            recent_data = usdt_data[usdt_data.index >= lookback_time]
            if len(recent_data) < 12:  # Need minimum data
                return None

            # Calculate range metrics
            high = recent_data['high'].max()
            low = recent_data['low'].min()
            range_pct = (high - low) / low

            # Check for stagnation
            if range_pct <= self.usdt_range_threshold:
                # Calculate signal strength based on duration and tightness
                duration_factor = len(recent_data) / 36  # Longer = stronger
                tightness_factor = 1 - (range_pct / self.usdt_range_threshold)
                strength = min(0.95, 0.3 + 0.4 * duration_factor + 0.3 * tightness_factor)

                return SMTSignal(
                    signal_type=SMTSignalType.USDT_STAGNATION,
                    timestamp=current_time,
                    confidence=0.8,
                    strength=strength,
                    hps_score=HPS_Score.LOW,  # Will be updated later
                    suppression_active=False,
                    metadata={
                        'range_pct': range_pct,
                        'duration_hours': len(recent_data),
                        'high': high,
                        'low': low
                    }
                )

            return None

        except Exception as e:
            logger.error(f"Error in USDT stagnation analysis: {e}")
            return None

    def _analyze_btc_wedge(self, btc_dom_data: pd.DataFrame, current_time: pd.Timestamp) -> Optional[SMTSignal]:
        """Detect BTC.D wedge formations and breakouts"""
        try:
            # Need sufficient data for wedge analysis
            if len(btc_dom_data) < 100:
                return None

            recent_data = btc_dom_data.tail(100)

            # Simplified wedge detection - look for converging trend lines
            highs = recent_data['high'].rolling(5).max()
            lows = recent_data['low'].rolling(5).min()

            # Calculate trend line slopes (simplified)
            high_slope = self._calculate_trend_slope(highs.dropna())
            low_slope = self._calculate_trend_slope(lows.dropna())

            # Check for wedge pattern (converging lines)
            if high_slope is not None and low_slope is not None:
                convergence = abs(high_slope - low_slope)

                # Check for breakout
                latest_close = recent_data['close'].iloc[-1]
                recent_high = highs.tail(20).max()
                recent_low = lows.tail(20).min()

                # Breakout detection
                if latest_close > recent_high * 1.002:  # Upward break
                    direction = "bullish"
                    strength = min(0.9, 0.5 + convergence * 100)
                elif latest_close < recent_low * 0.998:  # Downward break
                    direction = "bearish"
                    strength = min(0.9, 0.5 + convergence * 100)
                else:
                    return None

                return SMTSignal(
                    signal_type=SMTSignalType.BTC_WEDGE_BREAK,
                    timestamp=current_time,
                    confidence=0.75,
                    strength=strength,
                    hps_score=HPS_Score.LOW,
                    suppression_active=False,
                    metadata={
                        'direction': direction,
                        'convergence': convergence,
                        'breakout_level': latest_close,
                        'high_slope': high_slope,
                        'low_slope': low_slope
                    }
                )

            return None

        except Exception as e:
            logger.error(f"Error in BTC wedge analysis: {e}")
            return None

    def _analyze_total3_divergence(self, total3_data: pd.DataFrame, btc_data: pd.DataFrame,
                                   current_time: pd.Timestamp) -> Optional[SMTSignal]:
        """Detect TOTAL3 divergence from BTC price action"""
        try:
            if btc_data is None or len(total3_data) < self.total3_divergence_lookback:
                return None

            # Get recent data for correlation analysis
            lookback = self.total3_divergence_lookback
            total3_recent = total3_data.tail(lookback)['close']
            btc_recent = btc_data.tail(lookback)['close'] if len(btc_data) >= lookback else None

            if btc_recent is None:
                return None

            # Calculate correlation
            correlation = total3_recent.corr(btc_recent)

            # Detect divergence (low or negative correlation)
            if correlation < self.total3_correlation_threshold:
                # Calculate recent performance
                total3_change = (total3_recent.iloc[-1] / total3_recent.iloc[-24] - 1) if len(total3_recent) >= 24 else 0
                btc_change = (btc_recent.iloc[-1] / btc_recent.iloc[-24] - 1) if len(btc_recent) >= 24 else 0

                divergence_magnitude = abs(total3_change - btc_change)
                strength = min(0.9, 0.4 + divergence_magnitude * 2)

                return SMTSignal(
                    signal_type=SMTSignalType.TOTAL3_DIVERGENCE,
                    timestamp=current_time,
                    confidence=0.7,
                    strength=strength,
                    hps_score=HPS_Score.LOW,
                    suppression_active=False,
                    metadata={
                        'correlation': correlation,
                        'total3_change_24h': total3_change,
                        'btc_change_24h': btc_change,
                        'divergence_magnitude': divergence_magnitude
                    }
                )

            return None

        except Exception as e:
            logger.error(f"Error in TOTAL3 divergence analysis: {e}")
            return None

    def _calculate_trend_slope(self, series: pd.Series) -> Optional[float]:
        """Calculate trend line slope using linear regression"""
        try:
            if len(series) < 10:
                return None

            x = np.arange(len(series))
            y = series.values

            # Simple linear regression
            n = len(x)
            sum_x = np.sum(x)
            sum_y = np.sum(y)
            sum_xy = np.sum(x * y)
            sum_x2 = np.sum(x * x)

            slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x * sum_x)
            return slope

        except Exception:
            return None

    def _calculate_hps_scores(self, signals: List[SMTSignal], current_time: pd.Timestamp) -> List[SMTSignal]:
        """Calculate High-Probability Setup scores based on signal confluence"""
        if not signals:
            return signals

        # Count active signals by type
        signal_types = set(signal.signal_type for signal in signals)

        # Update HPS scores based on confluence
        for signal in signals:
            if len(signal_types) == 1:
                signal.hps_score = HPS_Score.LOW
            elif len(signal_types) == 2:
                signal.hps_score = HPS_Score.MEDIUM
            else:  # 3+ signals
                signal.hps_score = HPS_Score.HIGH

        return signals

    def _apply_suppression_filters(self, signals: List[SMTSignal], current_time: pd.Timestamp) -> List[SMTSignal]:
        """Apply suppression flags and cooldown periods"""
        filtered_signals = []

        for signal in signals:
            # Check suppression cooldown
            last_signal_time = self.last_signals.get(signal.signal_type)
            if last_signal_time:
                hours_since_last = (current_time - last_signal_time).total_seconds() / 3600
                if hours_since_last < self.suppression_cooldown:
                    signal.suppression_active = True

            # Apply HPS score filter
            if signal.hps_score.value >= self.min_hps_score and not signal.suppression_active:
                filtered_signals.append(signal)
                self.last_signals[signal.signal_type] = current_time

        return filtered_signals

    def get_macro_regime(self, signals: List[SMTSignal]) -> str:
        """Determine current macro regime based on SMT signals"""
        if not signals:
            return "neutral"

        # High HPS score signals indicate regime shift potential
        high_hps_signals = [s for s in signals if s.hps_score == HPS_Score.HIGH]
        if high_hps_signals:
            return "regime_shift"

        medium_hps_signals = [s for s in signals if s.hps_score == HPS_Score.MEDIUM]
        if medium_hps_signals:
            return "elevated"

        return "normal"