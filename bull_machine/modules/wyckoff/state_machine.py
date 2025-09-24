"""
Enhanced Wyckoff State Machine
Edge cases: Phase C trap scoring, spring/UTAD reclaim speed, dynamic range re-anchoring
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional, Tuple, List
import logging


class WyckoffStateMachine:
    """
    Enhanced Wyckoff state machine with trap detection and dynamic range management.
    """

    def __init__(self, config: Dict = None):
        self.config = config or {}
        self.current_state = {
            'phase': 'unknown',
            'bias': 'neutral',
            'confidence': 0.0,
            'range_low': None,
            'range_high': None,
            'last_significant_move': None
        }

    def analyze_wyckoff_state(self, df: pd.DataFrame) -> Dict:
        """
        Main analysis function that combines all Wyckoff logic.
        """
        if len(df) < 50:
            return {
                'bias': 'neutral',
                'confidence': 0.0,
                'phase': 'insufficient_data',
                'trap_score': 0.0,
                'reclaim_speed': 0.0,
                'range_valid': False
            }

        # Build/update range
        range_info = self.build_dynamic_range(df)

        # Detect current phase
        phase_analysis = self.detect_phase(df, range_info)

        # Calculate trap scoring for Phase C
        trap_score = self.calculate_trap_score(df, phase_analysis, range_info)

        # Analyze spring/UTAD reclaim speed
        reclaim_analysis = self.analyze_reclaim_speed(df, range_info)

        # Determine final bias and confidence
        bias, confidence = self.determine_bias_confidence(
            phase_analysis, trap_score, reclaim_analysis, range_info
        )

        return {
            'bias': bias,
            'confidence': confidence,
            'phase': phase_analysis['phase'],
            'trap_score': trap_score,
            'reclaim_speed': reclaim_analysis['speed_score'],
            'range_valid': range_info['valid'],
            'range_low': range_info['low'],
            'range_high': range_info['high'],
            'volume_context': phase_analysis.get('volume_context', {}),
            'structure_breaks': phase_analysis.get('structure_breaks', [])
        }

    def build_dynamic_range(self, df: pd.DataFrame) -> Dict:
        """
        Build dynamic range with re-anchoring for volatile HTF swings.
        """
        lookback = min(100, len(df))
        recent = df.tail(lookback)

        # Calculate volatility for dynamic range adjustment
        returns = recent['close'].pct_change().dropna()
        volatility = returns.std()

        # Adaptive range based on volatility
        if volatility > 0.03:  # High volatility
            range_period = min(30, lookback // 2)  # Shorter range
        else:  # Normal volatility
            range_period = min(60, lookback)  # Standard range

        range_data = recent.tail(range_period)

        # Find significant highs and lows
        range_high = range_data['high'].max()
        range_low = range_data['low'].min()
        range_size = range_high - range_low

        # Check for range invalidation (new significant highs/lows)
        current_price = df.iloc[-1]['close']
        recent_high = recent.tail(10)['high'].max()
        recent_low = recent.tail(10)['low'].min()

        # Re-anchor if we've broken significantly out of range
        needs_reanchor = False
        if current_price > range_high * 1.05:  # 5% above range high
            needs_reanchor = True
            # New accumulation range
            new_lookback = min(30, len(recent))
            range_data = recent.tail(new_lookback)
            range_high = range_data['high'].max()
            range_low = range_data['low'].min()
            range_size = range_high - range_low
            logging.info(f"Range re-anchored (upward): {range_low:.2f} - {range_high:.2f}")

        elif current_price < range_low * 0.95:  # 5% below range low
            needs_reanchor = True
            # New distribution range
            new_lookback = min(30, len(recent))
            range_data = recent.tail(new_lookback)
            range_high = range_data['high'].max()
            range_low = range_data['low'].min()
            range_size = range_high - range_low
            logging.info(f"Range re-anchored (downward): {range_low:.2f} - {range_high:.2f}")

        # Validate range quality
        range_valid = range_size > 0 and range_size > (range_high * 0.02)  # At least 2% range

        return {
            'high': range_high,
            'low': range_low,
            'size': range_size,
            'valid': range_valid,
            'reanchored': needs_reanchor,
            'lookback_period': range_period,
            'volatility': volatility
        }

    def detect_phase(self, df: pd.DataFrame, range_info: Dict) -> Dict:
        """
        Detect current Wyckoff phase with enhanced pattern recognition.
        """
        if not range_info['valid']:
            return {'phase': 'ranging', 'confidence': 0.3, 'volume_context': {}}

        current = df.iloc[-1]
        recent = df.tail(20)

        # Calculate position in range
        range_position = (current['close'] - range_info['low']) / range_info['size']

        # Volume analysis
        vol_sma = recent['volume'].mean()
        current_vol = current['volume']
        vol_expansion = current_vol > vol_sma * 1.2

        # Structural analysis
        swing_highs = self.find_swing_points(recent, 'high')
        swing_lows = self.find_swing_points(recent, 'low')

        volume_context = {
            'expansion': vol_expansion,
            'relative_volume': current_vol / vol_sma if vol_sma > 0 else 1.0,
            'avg_volume': vol_sma
        }

        # Phase determination logic
        if range_position < 0.3:
            # Lower portion - potential accumulation
            if vol_expansion and len(swing_lows) >= 2:
                # Multiple lows with volume = accumulation
                phase = 'accumulation_C' if range_position < 0.2 else 'accumulation_B'
                confidence = 0.7 + (0.1 if vol_expansion else 0)
            else:
                phase = 'accumulation_A'
                confidence = 0.5
        elif range_position > 0.7:
            # Upper portion - potential distribution
            if vol_expansion and len(swing_highs) >= 2:
                # Multiple highs with volume = distribution
                phase = 'distribution_C' if range_position > 0.8 else 'distribution_B'
                confidence = 0.7 + (0.1 if vol_expansion else 0)
            else:
                phase = 'distribution_A'
                confidence = 0.5
        else:
            # Middle range - transitional
            if vol_expansion:
                phase = 'transition'
                confidence = 0.6
            else:
                phase = 'ranging'
                confidence = 0.4

        return {
            'phase': phase,
            'confidence': confidence,
            'range_position': range_position,
            'volume_context': volume_context,
            'swing_highs': swing_highs,
            'swing_lows': swing_lows
        }

    def calculate_trap_score(self, df: pd.DataFrame, phase_analysis: Dict,
                           range_info: Dict) -> float:
        """
        Calculate Phase C trap scoring: shallow pullback + low volume = high trap risk.
        """
        if 'C' not in phase_analysis['phase']:
            return 0.0  # No trap risk outside Phase C

        recent = df.tail(20)
        current = df.iloc[-1]

        # Check for shallow pullback
        if phase_analysis['phase'] == 'accumulation_C':
            # For longs: look for shallow pullback from high
            recent_high = recent['high'].max()
            pullback_depth = (recent_high - current['close']) / recent_high
            shallow_pullback = pullback_depth < 0.382  # Less than 38.2% Fib

        elif phase_analysis['phase'] == 'distribution_C':
            # For shorts: look for shallow bounce from low
            recent_low = recent['low'].min()
            bounce_height = (current['close'] - recent_low) / recent_low
            shallow_pullback = bounce_height < 0.382

        else:
            return 0.0

        # Check for low volume (trap characteristic)
        vol_context = phase_analysis.get('volume_context', {})
        low_volume = vol_context.get('relative_volume', 1.0) < 0.8

        # Calculate trap score
        if shallow_pullback and low_volume:
            trap_score = 0.4  # High trap risk
        elif shallow_pullback or low_volume:
            trap_score = 0.2  # Moderate trap risk
        else:
            trap_score = 0.0

        # Additional penalty for multiple failed attempts
        if len(phase_analysis.get('swing_lows', [])) > 3:  # Accumulation
            trap_score += 0.1
        elif len(phase_analysis.get('swing_highs', [])) > 3:  # Distribution
            trap_score += 0.1

        return min(trap_score, 0.5)  # Cap at 0.5

    def analyze_reclaim_speed(self, df: pd.DataFrame, range_info: Dict) -> Dict:
        """
        Analyze spring/UTAD reclaim speed: require ≤5 bars for valid pattern.
        """
        if len(df) < 10:
            return {'speed_score': 0.0, 'pattern_type': None, 'bars_to_reclaim': None}

        recent = df.tail(20)

        # Look for spring pattern (sweep below low then reclaim)
        spring_analysis = self.detect_spring_reclaim(recent, range_info)

        # Look for UTAD pattern (sweep above high then reject)
        utad_analysis = self.detect_utad_reclaim(recent, range_info)

        # Determine best pattern and speed score
        if spring_analysis['detected'] and utad_analysis['detected']:
            # Both detected - use the faster one
            if spring_analysis['bars_to_reclaim'] <= utad_analysis['bars_to_reclaim']:
                return {
                    'speed_score': spring_analysis['speed_score'],
                    'pattern_type': 'spring',
                    'bars_to_reclaim': spring_analysis['bars_to_reclaim']
                }
            else:
                return {
                    'speed_score': utad_analysis['speed_score'],
                    'pattern_type': 'utad',
                    'bars_to_reclaim': utad_analysis['bars_to_reclaim']
                }
        elif spring_analysis['detected']:
            return {
                'speed_score': spring_analysis['speed_score'],
                'pattern_type': 'spring',
                'bars_to_reclaim': spring_analysis['bars_to_reclaim']
            }
        elif utad_analysis['detected']:
            return {
                'speed_score': utad_analysis['speed_score'],
                'pattern_type': 'utad',
                'bars_to_reclaim': utad_analysis['bars_to_reclaim']
            }
        else:
            return {'speed_score': 0.0, 'pattern_type': None, 'bars_to_reclaim': None}

    def detect_spring_reclaim(self, df: pd.DataFrame, range_info: Dict) -> Dict:
        """Detect spring pattern with reclaim speed analysis."""
        if not range_info['valid']:
            return {'detected': False, 'speed_score': 0.0, 'bars_to_reclaim': None}

        range_low = range_info['low']

        # Look for sweep below range low
        sweep_bars = []
        for i, (idx, row) in enumerate(df.iterrows()):
            if row['low'] < range_low * 0.995:  # 0.5% below range low
                sweep_bars.append(i)

        if not sweep_bars:
            return {'detected': False, 'speed_score': 0.0, 'bars_to_reclaim': None}

        # Find the last sweep
        last_sweep_idx = sweep_bars[-1]

        # Look for reclaim above range low
        reclaim_idx = None
        for i in range(last_sweep_idx + 1, len(df)):
            if df.iloc[i]['close'] > range_low:
                reclaim_idx = i
                break

        if reclaim_idx is None:
            return {'detected': False, 'speed_score': 0.0, 'bars_to_reclaim': None}

        # Calculate speed
        bars_to_reclaim = reclaim_idx - last_sweep_idx

        # Speed scoring
        if bars_to_reclaim <= 2:
            speed_score = 0.8  # Excellent speed
        elif bars_to_reclaim <= 5:
            speed_score = 0.6  # Good speed
        elif bars_to_reclaim <= 10:
            speed_score = 0.3  # Slow but valid
        else:
            speed_score = 0.1  # Too slow

        return {
            'detected': True,
            'speed_score': speed_score,
            'bars_to_reclaim': bars_to_reclaim
        }

    def detect_utad_reclaim(self, df: pd.DataFrame, range_info: Dict) -> Dict:
        """Detect UTAD pattern with rejection speed analysis."""
        if not range_info['valid']:
            return {'detected': False, 'speed_score': 0.0, 'bars_to_reclaim': None}

        range_high = range_info['high']

        # Look for sweep above range high
        sweep_bars = []
        for i, (idx, row) in enumerate(df.iterrows()):
            if row['high'] > range_high * 1.005:  # 0.5% above range high
                sweep_bars.append(i)

        if not sweep_bars:
            return {'detected': False, 'speed_score': 0.0, 'bars_to_reclaim': None}

        # Find the last sweep
        last_sweep_idx = sweep_bars[-1]

        # Look for rejection below range high
        reject_idx = None
        for i in range(last_sweep_idx + 1, len(df)):
            if df.iloc[i]['close'] < range_high:
                reject_idx = i
                break

        if reject_idx is None:
            return {'detected': False, 'speed_score': 0.0, 'bars_to_reclaim': None}

        # Calculate speed
        bars_to_reject = reject_idx - last_sweep_idx

        # Speed scoring (similar to spring but for rejection)
        if bars_to_reject <= 2:
            speed_score = 0.8  # Excellent rejection speed
        elif bars_to_reject <= 5:
            speed_score = 0.6  # Good rejection speed
        elif bars_to_reject <= 10:
            speed_score = 0.3  # Slow but valid
        else:
            speed_score = 0.1  # Too slow

        return {
            'detected': True,
            'speed_score': speed_score,
            'bars_to_reclaim': bars_to_reject
        }

    def find_swing_points(self, df: pd.DataFrame, price_type: str) -> List[Dict]:
        """Find swing highs or lows in the data."""
        if len(df) < 5:
            return []

        swings = []
        prices = df[price_type].values

        for i in range(2, len(prices) - 2):
            if price_type == 'high':
                # Swing high: higher than 2 bars on each side
                if prices[i] > prices[i-2] and prices[i] > prices[i-1] and \
                   prices[i] > prices[i+1] and prices[i] > prices[i+2]:
                    swings.append({
                        'index': i,
                        'price': prices[i],
                        'type': 'swing_high'
                    })
            else:  # low
                # Swing low: lower than 2 bars on each side
                if prices[i] < prices[i-2] and prices[i] < prices[i-1] and \
                   prices[i] < prices[i+1] and prices[i] < prices[i+2]:
                    swings.append({
                        'index': i,
                        'price': prices[i],
                        'type': 'swing_low'
                    })

        return swings

    def determine_bias_confidence(self, phase_analysis: Dict, trap_score: float,
                                reclaim_analysis: Dict, range_info: Dict) -> Tuple[str, float]:
        """
        Determine final bias and confidence with trap penalties and reclaim boosts.
        """
        base_confidence = phase_analysis['confidence']
        phase = phase_analysis['phase']

        # Determine base bias from phase
        if 'accumulation' in phase:
            bias = 'long'
        elif 'distribution' in phase:
            bias = 'short'
        else:
            bias = 'neutral'
            return bias, base_confidence

        # Apply trap penalty
        confidence_after_trap = base_confidence - trap_score

        # Apply reclaim speed boost
        speed_bonus = reclaim_analysis['speed_score'] * 0.15  # Max 0.12 boost
        final_confidence = confidence_after_trap + speed_bonus

        # Ensure confidence stays in valid range
        final_confidence = max(0.1, min(0.9, final_confidence))

        # Override bias to neutral if confidence too low
        if final_confidence < 0.4:
            bias = 'neutral'

        return bias, final_confidence