"""
Time Price Integration (TPI) - Bounded Temporal Analysis

Implements conservative TPI analysis with strict caps and validation
to avoid over-optimization while providing useful temporal context.
"""

import pandas as pd
import numpy as np
import math
from enum import Enum
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)

class TPIType(Enum):
    """TPI signal types"""
    TIME_CONFLUENCE = "time_confluence"
    PRICE_TIME_BALANCE = "price_time_balance"
    CYCLE_COMPLETION = "cycle_completion"
    TEMPORAL_SUPPORT = "temporal_support"
    TEMPORAL_RESISTANCE = "temporal_resistance"

@dataclass
class TPISignal:
    """TPI temporal signal"""
    tpi_type: TPIType
    timestamp: pd.Timestamp
    price_level: Optional[float]
    time_projection: Optional[pd.Timestamp]
    confidence: float
    strength: float
    cycle_data: Dict[str, Any]
    metadata: Dict[str, Any]

class TemporalEngine:
    """
    Conservative Temporal Analysis Engine with bounded TPI implementation.

    Provides minimal temporal analysis focused on:
    - Basic cycle detection (bounded to major periods)
    - Time-price confluence (with strict validation)
    - Conservative projections (max 30 days)
    - Cycle completion signals (major cycles only)
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.temporal_config = config.get('temporal', {})

        # Conservative bounds and limits
        self.max_projection_days = self.temporal_config.get('max_projection_days', 30)
        self.min_cycle_bars = self.temporal_config.get('min_cycle_bars', 24)  # 1 day minimum
        self.max_cycle_bars = self.temporal_config.get('max_cycle_bars', 720)  # 30 days maximum
        self.min_confidence = self.temporal_config.get('min_confidence', 0.6)

        # Major cycle periods (Fibonacci-based, conservative)
        self.major_cycles = self.temporal_config.get('major_cycles', [
            21, 34, 55, 89, 144, 233, 377  # Fibonacci sequence
        ])

        # Price-time ratios (simple golden ratio derivatives)
        self.pt_ratios = self.temporal_config.get('price_time_ratios', [
            0.618, 1.0, 1.618, 2.618
        ])

        # State tracking
        self.detected_cycles = {}
        self.historical_patterns = []

    def analyze_temporal_patterns(self, data: pd.DataFrame, current_price: float) -> List[TPISignal]:
        """
        Analyze temporal patterns with conservative bounds.

        Args:
            data: OHLCV data for analysis
            current_price: Current market price

        Returns:
            List of TPI signals with confidence scoring
        """
        try:
            if len(data) < self.min_cycle_bars * 2:
                logger.warning("Insufficient data for temporal analysis")
                return []

            signals = []
            current_time = data.index[-1]

            # 1. Basic cycle detection (major cycles only)
            cycle_signals = self._detect_major_cycles(data, current_time)
            signals.extend(cycle_signals)

            # 2. Time-price confluence analysis
            confluence_signals = self._analyze_time_price_confluence(data, current_price, current_time)
            signals.extend(confluence_signals)

            # 3. Cycle completion analysis
            completion_signals = self._analyze_cycle_completions(data, current_time)
            signals.extend(completion_signals)

            # 4. Filter and validate signals
            validated_signals = self._validate_temporal_signals(signals)

            return validated_signals

        except Exception as e:
            logger.error(f"Error in temporal analysis: {e}")
            return []

    def _detect_major_cycles(self, data: pd.DataFrame, current_time: pd.Timestamp) -> List[TPISignal]:
        """Detect major cycle patterns (conservative approach)"""
        try:
            signals = []

            for cycle_period in self.major_cycles:
                if cycle_period > len(data):
                    continue

                # Simple cycle detection using highs/lows
                cycle_analysis = self._analyze_cycle_period(data, cycle_period, current_time)
                if cycle_analysis['detected']:
                    signals.append(TPISignal(
                        tpi_type=TPIType.CYCLE_COMPLETION,
                        timestamp=current_time,
                        price_level=cycle_analysis.get('projected_price'),
                        time_projection=cycle_analysis.get('projected_time'),
                        confidence=cycle_analysis['confidence'],
                        strength=cycle_analysis['strength'],
                        cycle_data={
                            'period': cycle_period,
                            'phase': cycle_analysis['phase'],
                            'completion_pct': cycle_analysis['completion_pct']
                        },
                        metadata=cycle_analysis
                    ))

            return signals

        except Exception as e:
            logger.error(f"Error detecting major cycles: {e}")
            return []

    def _analyze_cycle_period(self, data: pd.DataFrame, period: int,
                             current_time: pd.Timestamp) -> Dict[str, Any]:
        """Analyze specific cycle period"""
        try:
            if period > len(data):
                return {'detected': False}

            # Get recent data for cycle analysis
            cycle_data = data.tail(period * 2)  # Look at 2 full cycles

            # Find major highs and lows
            highs = cycle_data['high'].rolling(window=period//4, center=True).max()
            lows = cycle_data['low'].rolling(window=period//4, center=True).min()

            # Identify cycle peaks and troughs
            peaks = cycle_data[cycle_data['high'] == highs]
            troughs = cycle_data[cycle_data['low'] == lows]

            if len(peaks) < 2 or len(troughs) < 2:
                return {'detected': False}

            # Calculate cycle metrics
            last_peak = peaks.index[-1] if len(peaks) > 0 else None
            last_trough = troughs.index[-1] if len(troughs) > 0 else None

            if last_peak is None and last_trough is None:
                return {'detected': False}

            # Determine cycle phase
            current_phase = self._determine_cycle_phase(cycle_data, period, last_peak, last_trough)

            # Calculate completion percentage
            completion_pct = self._calculate_cycle_completion(cycle_data, period, current_phase)

            # Project next significant time
            projected_time = self._project_cycle_time(current_time, period, completion_pct)

            # Simple confidence based on regularity
            confidence = self._calculate_cycle_confidence(cycle_data, period)

            return {
                'detected': confidence >= self.min_confidence,
                'confidence': confidence,
                'strength': min(1.0, confidence * 1.2),
                'phase': current_phase,
                'completion_pct': completion_pct,
                'projected_time': projected_time,
                'period': period,
                'last_peak': last_peak,
                'last_trough': last_trough
            }

        except Exception as e:
            logger.error(f"Error analyzing cycle period {period}: {e}")
            return {'detected': False}

    def _analyze_time_price_confluence(self, data: pd.DataFrame, current_price: float,
                                     current_time: pd.Timestamp) -> List[TPISignal]:
        """Analyze time-price confluence using conservative ratios"""
        try:
            signals = []

            # Look for significant moves in recent history
            lookback_bars = min(self.max_cycle_bars, len(data))
            recent_data = data.tail(lookback_bars)

            if len(recent_data) < self.min_cycle_bars:
                return signals

            # Find significant highs and lows
            significant_levels = self._identify_significant_levels(recent_data)

            for level_data in significant_levels:
                level_price = level_data['price']
                level_time = level_data['timestamp']

                # Calculate time and price distances
                time_distance = (current_time - level_time).total_seconds() / 3600  # hours
                price_distance = abs(current_price - level_price)

                # Check for time-price ratio confluence
                for ratio in self.pt_ratios:
                    confluence = self._check_tp_confluence(
                        time_distance, price_distance, level_price, ratio
                    )

                    if confluence['detected']:
                        signals.append(TPISignal(
                            tpi_type=TPIType.TIME_CONFLUENCE,
                            timestamp=current_time,
                            price_level=confluence.get('target_price'),
                            time_projection=confluence.get('target_time'),
                            confidence=confluence['confidence'],
                            strength=confluence['strength'],
                            cycle_data={
                                'ratio': ratio,
                                'reference_level': level_price,
                                'reference_time': level_time
                            },
                            metadata=confluence
                        ))

            return signals

        except Exception as e:
            logger.error(f"Error in time-price confluence analysis: {e}")
            return []

    def _analyze_cycle_completions(self, data: pd.DataFrame,
                                  current_time: pd.Timestamp) -> List[TPISignal]:
        """Analyze potential cycle completions"""
        try:
            signals = []

            # Check each major cycle for completion signals
            for cycle_period in self.major_cycles:
                if cycle_period > len(data):
                    continue

                completion_analysis = self._check_cycle_completion(data, cycle_period, current_time)
                if completion_analysis['completion_detected']:
                    signals.append(TPISignal(
                        tpi_type=TPIType.CYCLE_COMPLETION,
                        timestamp=current_time,
                        price_level=completion_analysis.get('completion_price'),
                        time_projection=completion_analysis.get('next_cycle_start'),
                        confidence=completion_analysis['confidence'],
                        strength=completion_analysis['strength'],
                        cycle_data={
                            'period': cycle_period,
                            'completion_type': completion_analysis['completion_type']
                        },
                        metadata=completion_analysis
                    ))

            return signals

        except Exception as e:
            logger.error(f"Error analyzing cycle completions: {e}")
            return []

    def _determine_cycle_phase(self, data: pd.DataFrame, period: int,
                              last_peak: Optional[pd.Timestamp],
                              last_trough: Optional[pd.Timestamp]) -> str:
        """Determine current phase of cycle"""
        try:
            current_time = data.index[-1]

            if last_peak is None and last_trough is None:
                return 'unknown'

            # Simple phase determination
            if last_peak is None:
                return 'early_cycle'
            elif last_trough is None:
                return 'mid_cycle'
            elif last_peak > last_trough:
                return 'declining_phase'
            else:
                return 'ascending_phase'

        except Exception:
            return 'unknown'

    def _calculate_cycle_completion(self, data: pd.DataFrame, period: int, phase: str) -> float:
        """Calculate how complete the current cycle is"""
        try:
            # Simplified completion calculation
            cycle_data = data.tail(period)
            if len(cycle_data) < period:
                return len(cycle_data) / period

            # Based on phase, estimate completion
            phase_completions = {
                'early_cycle': 0.25,
                'ascending_phase': 0.5,
                'mid_cycle': 0.6,
                'declining_phase': 0.8,
                'unknown': 0.5
            }

            return phase_completions.get(phase, 0.5)

        except Exception:
            return 0.5

    def _project_cycle_time(self, current_time: pd.Timestamp, period: int,
                           completion_pct: float) -> pd.Timestamp:
        """Project next significant cycle time"""
        try:
            remaining_pct = 1.0 - completion_pct
            remaining_hours = remaining_pct * period

            # Cap projections to maximum allowed days
            max_hours = self.max_projection_days * 24
            projected_hours = min(remaining_hours, max_hours)

            return current_time + pd.Timedelta(hours=projected_hours)

        except Exception:
            return current_time + pd.Timedelta(days=self.max_projection_days)

    def _calculate_cycle_confidence(self, data: pd.DataFrame, period: int) -> float:
        """Calculate confidence in cycle detection"""
        try:
            if len(data) < period * 2:
                return 0.0

            # Simple regularity check
            cycle_data = data.tail(period * 2)

            # Check for consistent patterns (simplified)
            price_volatility = cycle_data['close'].pct_change().std()
            volume_consistency = 1.0 - (cycle_data['volume'].std() / cycle_data['volume'].mean())

            # Combine factors
            confidence = (
                (1.0 - min(1.0, price_volatility * 10)) * 0.6 +
                volume_consistency * 0.4
            )

            return max(0.1, min(0.9, confidence))

        except Exception:
            return 0.0

    def _identify_significant_levels(self, data: pd.DataFrame) -> List[Dict[str, Any]]:
        """Identify significant price levels for confluence analysis"""
        try:
            levels = []

            # Simple peak/trough identification
            window = max(5, len(data) // 20)  # Adaptive window

            # Find peaks
            for i in range(window, len(data) - window):
                current_high = data['high'].iloc[i]
                is_peak = True

                # Check if it's a local maximum
                for j in range(i - window, i + window + 1):
                    if j != i and data['high'].iloc[j] >= current_high:
                        is_peak = False
                        break

                if is_peak:
                    levels.append({
                        'price': current_high,
                        'timestamp': data.index[i],
                        'type': 'peak',
                        'volume': data['volume'].iloc[i]
                    })

            # Find troughs
            for i in range(window, len(data) - window):
                current_low = data['low'].iloc[i]
                is_trough = True

                # Check if it's a local minimum
                for j in range(i - window, i + window + 1):
                    if j != i and data['low'].iloc[j] <= current_low:
                        is_trough = False
                        break

                if is_trough:
                    levels.append({
                        'price': current_low,
                        'timestamp': data.index[i],
                        'type': 'trough',
                        'volume': data['volume'].iloc[i]
                    })

            # Sort by significance (volume-weighted)
            levels.sort(key=lambda x: x['volume'], reverse=True)
            return levels[:10]  # Return top 10 most significant

        except Exception as e:
            logger.error(f"Error identifying significant levels: {e}")
            return []

    def _check_tp_confluence(self, time_distance: float, price_distance: float,
                           reference_price: float, ratio: float) -> Dict[str, Any]:
        """Check for time-price confluence at specific ratio"""
        try:
            # Convert time distance to price units (simplified)
            # This is a very basic implementation
            price_time_unit = reference_price * 0.001  # 0.1% per hour as base unit

            expected_price_distance = time_distance * price_time_unit * ratio
            actual_ratio = price_distance / (time_distance * price_time_unit) if time_distance > 0 else 0

            # Check if actual ratio is close to expected ratio
            ratio_difference = abs(actual_ratio - ratio) / ratio if ratio > 0 else 1.0

            if ratio_difference <= 0.2:  # 20% tolerance
                confidence = 1.0 - ratio_difference
                return {
                    'detected': True,
                    'confidence': confidence,
                    'strength': confidence * 0.8,  # Conservative strength
                    'actual_ratio': actual_ratio,
                    'expected_ratio': ratio,
                    'ratio_difference': ratio_difference
                }

            return {'detected': False}

        except Exception as e:
            logger.error(f"Error checking TP confluence: {e}")
            return {'detected': False}

    def _check_cycle_completion(self, data: pd.DataFrame, period: int,
                               current_time: pd.Timestamp) -> Dict[str, Any]:
        """Check for cycle completion signals"""
        try:
            if period > len(data):
                return {'completion_detected': False}

            cycle_data = data.tail(period)

            # Simple completion check based on price action
            start_price = cycle_data['close'].iloc[0]
            current_price = cycle_data['close'].iloc[-1]
            price_change = (current_price - start_price) / start_price

            # Check if we're near cycle start price (potential completion)
            if abs(price_change) < 0.05:  # Within 5% of start
                confidence = 1.0 - (abs(price_change) / 0.05)
                return {
                    'completion_detected': True,
                    'confidence': confidence,
                    'strength': confidence * 0.7,
                    'completion_type': 'price_return',
                    'price_change': price_change
                }

            return {'completion_detected': False}

        except Exception as e:
            logger.error(f"Error checking cycle completion: {e}")
            return {'completion_detected': False}

    def _validate_temporal_signals(self, signals: List[TPISignal]) -> List[TPISignal]:
        """Validate and filter temporal signals"""
        try:
            validated = []

            for signal in signals:
                # Apply minimum confidence filter
                if signal.confidence < self.min_confidence:
                    continue

                # Check projection bounds
                if signal.time_projection:
                    hours_ahead = (signal.time_projection - signal.timestamp).total_seconds() / 3600
                    max_hours = self.max_projection_days * 24
                    if hours_ahead > max_hours:
                        continue

                # Additional validation could go here
                validated.append(signal)

            # Remove duplicates and rank by confidence
            validated.sort(key=lambda x: x.confidence, reverse=True)

            # Limit number of signals to prevent over-fitting
            return validated[:5]  # Max 5 temporal signals

        except Exception as e:
            logger.error(f"Error validating temporal signals: {e}")
            return signals

    def get_temporal_context(self, current_price: float, current_time: pd.Timestamp) -> Dict[str, Any]:
        """Get current temporal context summary"""
        try:
            context = {
                'active_cycles': [],
                'projected_times': [],
                'confluence_levels': [],
                'overall_temporal_bias': 'neutral'
            }

            # This would be populated based on current analysis
            # For now, return basic structure
            return context

        except Exception as e:
            logger.error(f"Error getting temporal context: {e}")
            return {'overall_temporal_bias': 'neutral'}