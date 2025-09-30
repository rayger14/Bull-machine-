"""
Smart Money Concepts (SMC) Engine

Unified engine that coordinates all SMC modules for institutional trading analysis.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import logging

from .order_blocks import OrderBlockDetector, OrderBlock
from .fvg import FVGDetector, FairValueGap
from .liquidity_sweeps import LiquiditySweepDetector, LiquiditySweep
from .bos import BOSDetector, BreakOfStructure, TrendState

logger = logging.getLogger(__name__)

@dataclass
class SMCSignal:
    """Unified SMC signal output"""
    timestamp: pd.Timestamp
    direction: str  # 'long', 'short', 'neutral'
    strength: float  # 0-1 overall signal strength
    confidence: float  # 0-1 overall confidence

    # Component signals
    order_blocks: List[OrderBlock]
    fair_value_gaps: List[FairValueGap]
    liquidity_sweeps: List[LiquiditySweep]
    structure_breaks: List[BreakOfStructure]

    # Analysis
    trend_state: TrendState
    confluence_score: float  # How many components align
    institutional_bias: str  # 'bullish', 'bearish', 'neutral'
    entry_zones: List[Tuple[float, float]]  # Potential entry price ranges

    # Hit Counters for per-trade logging
    hit_counters: Dict[str, Any]  # OB hits, FVG hits, Sweep hits, BOS hits
    confluence_rate: float  # % of components that triggered

    metadata: Dict[str, Any]

class SMCEngine:
    """
    Smart Money Concepts Engine

    Integrates Order Blocks, FVGs, Liquidity Sweeps, and BOS analysis
    to provide institutional-grade market structure insights.
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config

        # Initialize component detectors
        self.ob_detector = OrderBlockDetector(config.get('order_blocks', {}))
        self.fvg_detector = FVGDetector(config.get('fvg', {}))
        self.sweep_detector = LiquiditySweepDetector(config.get('liquidity_sweeps', {}))
        self.bos_detector = BOSDetector(config.get('bos', {}))

        # SMC-specific settings
        self.min_confluence = config.get('min_confluence', 2)
        self.proximity_pct = config.get('proximity_pct', 0.02)  # 2% for confluence

    def analyze(self, data: pd.DataFrame) -> SMCSignal:
        """Main analysis method - wrapper for analyze_smc"""
        return self.analyze_smc(data)

    def analyze_smc(self, data: pd.DataFrame) -> SMCSignal:
        """
        Perform comprehensive SMC analysis.

        Args:
            data: OHLCV DataFrame

        Returns:
            Unified SMC signal
        """
        try:
            if len(data) < 50:
                return self._empty_signal(data.index[-1] if not data.empty else pd.Timestamp.now())

            current_price = data['close'].iloc[-1]
            current_time = data.index[-1]

            # Run all component analyses
            order_blocks = self.ob_detector.detect_order_blocks(data)
            fvgs = self.fvg_detector.detect_fvgs(data)
            sweeps = self.sweep_detector.detect_sweeps(data)
            bos_events = self.bos_detector.detect_bos(data)

            # Filter for nearby/relevant signals
            nearby_obs = self.ob_detector.get_nearest_blocks(data, current_price, self.proximity_pct)
            nearby_fvgs = self.fvg_detector.get_nearest_fvgs(data, current_price, self.proximity_pct)
            recent_sweeps = self.sweep_detector.get_recent_sweeps(data, 20)
            recent_bos = bos_events[-5:] if bos_events else []  # Last 5 BOS events

            # Analyze confluence and generate signal
            signal = self._generate_unified_signal(
                current_time, current_price,
                nearby_obs, nearby_fvgs, recent_sweeps, recent_bos, data
            )

            return signal

        except Exception as e:
            logger.error(f"Error in SMC analysis: {e}")
            return self._empty_signal(data.index[-1] if not data.empty else pd.Timestamp.now())

    def _generate_unified_signal(self, timestamp: pd.Timestamp, current_price: float,
                                order_blocks: List[OrderBlock], fvgs: List[FairValueGap],
                                sweeps: List[LiquiditySweep], bos_events: List[BreakOfStructure],
                                data: pd.DataFrame) -> SMCSignal:
        """Generate unified signal from all SMC components"""
        try:
            # Count bullish vs bearish signals
            bullish_signals = 0
            bearish_signals = 0
            total_strength = 0
            total_confidence = 0
            signal_count = 0

            # Initialize hit counters
            hit_counters = {
                'ob_hits': 0,
                'fvg_hits': 0,
                'sweep_hits': 0,
                'bos_hits': 0,
                'total_components': 4,
                'active_components': []
            }

            # Analyze Order Blocks
            for ob in order_blocks:
                if ob.ob_type.value == 'bullish' and current_price >= ob.low * 0.995:
                    bullish_signals += 1
                    total_strength += ob.strength
                    total_confidence += ob.confidence
                    signal_count += 1
                    hit_counters['ob_hits'] += 1
                    hit_counters['active_components'].append('OB_bullish')
                elif ob.ob_type.value == 'bearish' and current_price <= ob.high * 1.005:
                    bearish_signals += 1
                    total_strength += ob.strength
                    total_confidence += ob.confidence
                    signal_count += 1
                    hit_counters['ob_hits'] += 1
                    hit_counters['active_components'].append('OB_bearish')

            # Analyze FVGs
            for fvg in fvgs:
                if fvg.fvg_type.value == 'bullish' and current_price >= fvg.low:
                    bullish_signals += 1
                    total_strength += fvg.strength
                    total_confidence += fvg.confidence
                    signal_count += 1
                    hit_counters['fvg_hits'] += 1
                    hit_counters['active_components'].append('FVG_bullish')
                elif fvg.fvg_type.value == 'bearish' and current_price <= fvg.high:
                    bearish_signals += 1
                    total_strength += fvg.strength
                    total_confidence += fvg.confidence
                    signal_count += 1
                    hit_counters['fvg_hits'] += 1
                    hit_counters['active_components'].append('FVG_bearish')

            # Analyze Liquidity Sweeps
            for sweep in sweeps:
                if sweep.reversal_confirmation:
                    hit_counters['sweep_hits'] += 1
                    if sweep.sweep_type.value == 'sell_side':  # Bullish reversal after sell sweep
                        bullish_signals += 1
                        hit_counters['active_components'].append('Sweep_bullish')
                    else:  # Bearish reversal after buy sweep
                        bearish_signals += 1
                        hit_counters['active_components'].append('Sweep_bearish')
                    total_strength += sweep.strength
                    total_confidence += sweep.confidence
                    signal_count += 1

            # Analyze BOS
            trend_state = TrendState.SIDEWAYS
            if bos_events:
                latest_bos = bos_events[-1]
                trend_state = latest_bos.new_trend
                hit_counters['bos_hits'] += 1

                if latest_bos.bos_type.value == 'bullish':
                    bullish_signals += 1
                    hit_counters['active_components'].append('BOS_bullish')
                else:
                    bearish_signals += 1
                    hit_counters['active_components'].append('BOS_bearish')
                total_strength += latest_bos.strength
                total_confidence += latest_bos.confidence
                signal_count += 1

            # Determine overall direction
            if bullish_signals > bearish_signals and bullish_signals >= self.min_confluence:
                direction = 'long'
                institutional_bias = 'bullish'
            elif bearish_signals > bullish_signals and bearish_signals >= self.min_confluence:
                direction = 'short'
                institutional_bias = 'bearish'
            else:
                direction = 'neutral'
                institutional_bias = 'neutral'

            # Calculate overall metrics
            confluence_score = max(bullish_signals, bearish_signals) / 4.0  # Max 4 components
            confluence_rate = len(hit_counters['active_components']) / hit_counters['total_components']

            avg_strength = total_strength / signal_count if signal_count > 0 else 0
            avg_confidence = total_confidence / signal_count if signal_count > 0 else 0

            # Generate entry zones based on nearby levels
            entry_zones = self._identify_entry_zones(order_blocks, fvgs, current_price)

            return SMCSignal(
                timestamp=timestamp,
                direction=direction,
                strength=min(1.0, avg_strength),
                confidence=min(1.0, avg_confidence),
                order_blocks=order_blocks,
                fair_value_gaps=fvgs,
                liquidity_sweeps=sweeps,
                structure_breaks=bos_events,
                trend_state=trend_state,
                confluence_score=min(1.0, confluence_score),
                institutional_bias=institutional_bias,
                entry_zones=entry_zones,
                hit_counters=hit_counters,
                confluence_rate=confluence_rate,
                metadata={
                    'bullish_signals': bullish_signals,
                    'bearish_signals': bearish_signals,
                    'signal_count': signal_count,
                    'current_price': current_price
                }
            )

        except Exception as e:
            logger.error(f"Error generating unified signal: {e}")
            return self._empty_signal(timestamp)

    def _identify_entry_zones(self, order_blocks: List[OrderBlock],
                            fvgs: List[FairValueGap], current_price: float) -> List[Tuple[float, float]]:
        """Identify potential entry zones from SMC levels"""
        try:
            zones = []

            # Add order block zones
            for ob in order_blocks:
                zones.append((ob.low, ob.high))

            # Add FVG zones
            for fvg in fvgs:
                zones.append((fvg.low, fvg.high))

            # Sort by distance from current price
            zones.sort(key=lambda x: abs(current_price - (x[0] + x[1]) / 2))

            # Return top 3 closest zones
            return zones[:3]

        except Exception as e:
            logger.error(f"Error identifying entry zones: {e}")
            return []

    def _empty_signal(self, timestamp: pd.Timestamp) -> SMCSignal:
        """Return empty/neutral signal"""
        return SMCSignal(
            timestamp=timestamp,
            direction='neutral',
            strength=0.0,
            confidence=0.0,
            order_blocks=[],
            fair_value_gaps=[],
            liquidity_sweeps=[],
            structure_breaks=[],
            trend_state=TrendState.SIDEWAYS,
            confluence_score=0.0,
            institutional_bias='neutral',
            entry_zones=[],
            hit_counters={'ob_hits': 0, 'fvg_hits': 0, 'sweep_hits': 0, 'bos_hits': 0, 'total_components': 4, 'active_components': []},
            confluence_rate=0.0,
            metadata={}
        )