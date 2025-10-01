"""
Advanced Fusion Engine with Delta Routing

Implements proper signal fusion with delta channels to prevent double counting:
- Base weights: Wyckoff + Liquidity + SMC + Temporal
- Delta channels: Momentum + Macro + HOB Volume boosts
- Telemetry: Full waterfall tracking for transparency
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum
import logging

logger = logging.getLogger(__name__)

@dataclass
class FusionTelemetry:
    """Detailed fusion telemetry for transparency"""
    timestamp: pd.Timestamp

    # Base domain scores
    wyckoff_score: float
    liquidity_score: float
    smc_score: float
    temporal_score: float
    base_weighted_score: float

    # Delta contributions
    momentum_delta: float
    macro_delta: float
    hob_volume_delta: float
    wyckoff_hps_delta: float
    total_deltas: float

    # Final fusion result
    fusion_score: float
    fusion_direction: str
    fusion_confidence: float

    # Entry decision
    confidence_threshold: float
    strength_threshold: float
    meets_entry_criteria: bool

    # Signal breakdown
    signal_breakdown: Dict[str, Any]

    metadata: Dict[str, Any]

@dataclass
class AdvancedFusionSignal:
    """Advanced fusion signal with full telemetry"""
    timestamp: pd.Timestamp
    direction: str  # 'long', 'short', 'neutral'
    strength: float  # 0-1
    confidence: float  # 0-1

    # Entry decision
    trade_signal: bool
    entry_price: float
    stop_loss: Optional[float]
    take_profit: Optional[float]

    # Risk management
    is_momentum_only: bool  # If driven only by momentum
    suggested_sizing: float  # Position sizing multiplier

    # Full telemetry
    telemetry: FusionTelemetry

    metadata: Dict[str, Any]

class AdvancedFusionEngine:
    """
    Advanced Fusion Engine with Delta Routing

    Routes signals through proper channels to prevent double counting
    and provides full transparency via telemetry waterfall.
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.domain_weights = config['domain_weights']
        self.calibration_mode = config.get('calibration_mode', False)

        # Threshold management
        if self.calibration_mode:
            cal_thresholds = config.get('calibration_thresholds', {})
            self.confidence_threshold = cal_thresholds.get('confidence', 0.32)
            self.strength_threshold = cal_thresholds.get('strength', 0.40)
        else:
            self.confidence_threshold = config.get('entry_threshold_confidence', 0.35)
            self.strength_threshold = config.get('entry_threshold_strength', 0.40)

        logger.info(f"Fusion Engine: {'Calibration' if self.calibration_mode else 'Production'} mode")
        logger.info(f"Thresholds: confidence >= {self.confidence_threshold}, strength >= {self.strength_threshold}")

    def fuse_signals(self, domain_signals: Dict[str, Any], current_bar: pd.Series,
                    macro_delta: float = 0.0) -> Optional[AdvancedFusionSignal]:
        """
        Advanced signal fusion with delta routing.

        Args:
            domain_signals: Dict of domain signals by domain name
            current_bar: Current OHLCV bar data
            macro_delta: Macro pulse delta (-0.10 to +0.10)

        Returns:
            AdvancedFusionSignal with full telemetry, or None if no signal
        """
        try:
            timestamp = current_bar.name if hasattr(current_bar, 'name') else pd.Timestamp.now()
            current_price = float(current_bar['close'])

            # 1. Calculate base domain scores
            base_scores = self._calculate_base_scores(domain_signals)

            # 2. Calculate delta contributions
            deltas = self._calculate_deltas(domain_signals, current_bar, macro_delta)

            # 3. Combine base + deltas
            fusion_result = self._combine_base_and_deltas(base_scores, deltas)

            # 4. Entry decision logic
            entry_decision = self._evaluate_entry_criteria(fusion_result, base_scores, deltas)

            # 5. Risk management adjustments
            risk_adjustments = self._calculate_risk_adjustments(base_scores, deltas)

            # 6. Create telemetry
            telemetry = FusionTelemetry(
                timestamp=timestamp,
                wyckoff_score=base_scores.get('wyckoff', 0.0),
                liquidity_score=base_scores.get('liquidity', 0.0),
                smc_score=base_scores.get('smc', 0.0),
                temporal_score=base_scores.get('temporal', 0.0),
                base_weighted_score=fusion_result['base_score'],
                momentum_delta=deltas.get('momentum', 0.0),
                macro_delta=deltas.get('macro', 0.0),
                hob_volume_delta=deltas.get('hob_volume', 0.0),
                wyckoff_hps_delta=deltas.get('wyckoff_hps', 0.0),
                total_deltas=sum(deltas.values()),
                fusion_score=fusion_result['final_score'],
                fusion_direction=fusion_result['direction'],
                fusion_confidence=fusion_result['confidence'],
                confidence_threshold=self.confidence_threshold,
                strength_threshold=self.strength_threshold,
                meets_entry_criteria=entry_decision['meets_criteria'],
                signal_breakdown=self._create_signal_breakdown(domain_signals),
                metadata={'calibration_mode': self.calibration_mode}
            )

            if not entry_decision['meets_criteria']:
                return None

            return AdvancedFusionSignal(
                timestamp=timestamp,
                direction=fusion_result['direction'],
                strength=fusion_result['strength'],
                confidence=fusion_result['confidence'],
                trade_signal=True,
                entry_price=current_price,
                stop_loss=entry_decision.get('stop_loss'),
                take_profit=entry_decision.get('take_profit'),
                is_momentum_only=risk_adjustments['is_momentum_only'],
                suggested_sizing=risk_adjustments['sizing_multiplier'],
                telemetry=telemetry,
                metadata={
                    'fusion_engine_version': 'advanced_v1.0',
                    'calibration_mode': self.calibration_mode
                }
            )

        except Exception as e:
            logger.error(f"Error in advanced fusion: {e}")
            return None

    def _calculate_base_scores(self, domain_signals: Dict[str, Any]) -> Dict[str, float]:
        """Calculate base domain scores using configured weights"""
        base_scores = {}

        # Only include non-delta domains in base score
        base_domains = ['wyckoff', 'liquidity', 'smc', 'temporal', 'macro_context']

        for domain in base_domains:
            if domain in domain_signals and domain in self.domain_weights:
                signal = domain_signals[domain]
                if hasattr(signal, 'confidence'):
                    base_scores[domain] = signal.confidence * self.domain_weights[domain]
                else:
                    base_scores[domain] = 0.0
            else:
                base_scores[domain] = 0.0

        return base_scores

    def _calculate_deltas(self, domain_signals: Dict[str, Any], current_bar: pd.Series,
                         macro_delta: float) -> Dict[str, float]:
        """Calculate all delta contributions"""
        deltas = {}

        # Momentum delta (±0.06 cap)
        if 'momentum' in domain_signals:
            momentum_signal = domain_signals['momentum']
            if hasattr(momentum_signal, 'momentum_delta'):
                deltas['momentum'] = np.clip(momentum_signal.momentum_delta, -0.06, 0.06)
            else:
                deltas['momentum'] = 0.0
        else:
            deltas['momentum'] = 0.0

        # Macro delta (±0.10 cap)
        deltas['macro'] = np.clip(macro_delta, -0.10, 0.10)

        # HOB volume delta (only on entry bar with relevant HOB)
        if 'hob' in domain_signals:
            hob_signal = domain_signals['hob']
            # This would normally call hob.calculate_hob_volume_delta()
            # For now, simplified
            deltas['hob_volume'] = 0.0
        else:
            deltas['hob_volume'] = 0.0

        # Wyckoff HPS delta (Phase B/C with HPS >= 0.5)
        if 'wyckoff' in domain_signals:
            wyckoff_signal = domain_signals['wyckoff']
            if hasattr(wyckoff_signal, 'metadata') and 'hps_score' in wyckoff_signal.metadata:
                hps_score = wyckoff_signal.metadata['hps_score']
                if hps_score >= 0.5 and wyckoff_signal.phase.value in ['reaccumulation', 'redistribution']:
                    deltas['wyckoff_hps'] = min(0.03, hps_score * 0.06)
                else:
                    deltas['wyckoff_hps'] = 0.0
            else:
                deltas['wyckoff_hps'] = 0.0
        else:
            deltas['wyckoff_hps'] = 0.0

        return deltas

    def _combine_base_and_deltas(self, base_scores: Dict[str, float],
                                deltas: Dict[str, float]) -> Dict[str, Any]:
        """Combine base scores with delta contributions"""

        # Calculate weighted base score
        base_score = sum(base_scores.values())

        # Apply deltas
        total_deltas = sum(deltas.values())
        final_score = base_score + total_deltas

        # Determine direction based on final score
        if final_score > 0.02:
            direction = 'long'
            strength = min(1.0, abs(final_score))
        elif final_score < -0.02:
            direction = 'short'
            strength = min(1.0, abs(final_score))
        else:
            direction = 'neutral'
            strength = 0.0

        # Calculate confidence (average of participating domains)
        participating_domains = [score for score in base_scores.values() if score > 0]
        confidence = np.mean(participating_domains) if participating_domains else 0.0

        return {
            'base_score': base_score,
            'total_deltas': total_deltas,
            'final_score': final_score,
            'direction': direction,
            'strength': strength,
            'confidence': confidence
        }

    def _evaluate_entry_criteria(self, fusion_result: Dict[str, Any],
                                base_scores: Dict[str, float],
                                deltas: Dict[str, float]) -> Dict[str, Any]:
        """Evaluate entry criteria with guardrails"""

        confidence = fusion_result['confidence']
        strength = fusion_result['strength']

        # Basic threshold check
        meets_basic_criteria = (confidence >= self.confidence_threshold and
                              strength >= self.strength_threshold)

        if not meets_basic_criteria:
            return {
                'meets_criteria': False,
                'reason': f'Below thresholds: conf={confidence:.3f}<{self.confidence_threshold}, str={strength:.3f}<{self.strength_threshold}'
            }

        return {
            'meets_criteria': True,
            'confidence': confidence,
            'strength': strength,
            'stop_loss': None,  # Would calculate based on ATR/levels
            'take_profit': None  # Would calculate based on R:R ratio
        }

    def _calculate_risk_adjustments(self, base_scores: Dict[str, float],
                                   deltas: Dict[str, float]) -> Dict[str, Any]:
        """Calculate risk management adjustments"""

        # Check if momentum-only signal
        non_momentum_score = sum(score for domain, score in base_scores.items()
                               if domain != 'momentum' and score > 0)

        is_momentum_only = (non_momentum_score == 0 and deltas.get('momentum', 0) != 0)

        # Adjust sizing for momentum-only signals
        if is_momentum_only:
            sizing_multiplier = 0.5  # Half size for momentum-only
        else:
            sizing_multiplier = 1.0

        return {
            'is_momentum_only': is_momentum_only,
            'sizing_multiplier': sizing_multiplier,
            'non_momentum_score': non_momentum_score
        }

    def _create_signal_breakdown(self, domain_signals: Dict[str, Any]) -> Dict[str, Any]:
        """Create detailed signal breakdown for telemetry"""
        breakdown = {}

        for domain, signal in domain_signals.items():
            if signal:
                breakdown[domain] = {
                    'active': True,
                    'direction': getattr(signal, 'direction', 'neutral'),
                    'confidence': getattr(signal, 'confidence', 0.0),
                    'strength': getattr(signal, 'strength', 0.0)
                }
            else:
                breakdown[domain] = {'active': False}

        return breakdown