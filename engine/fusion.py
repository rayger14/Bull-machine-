"""
Enhanced Fusion Engine - v1.7 Domain Aggregation with Macro Pulse Integration

Combines signals from all 5 domains (Wyckoff, Liquidity, Momentum, Temporal, Macro Context)
with intelligent veto logic, macro pulse integration, and explainable deltas.

"Macro is the weather. Wyckoff is the map. Liquidity is the terrain."
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass
from enum import Enum
import logging

# Import domain-specific signals
from engine.context.signals import SMTSignal, HPS_Score
from engine.context.macro_pulse import MacroPulseEngine, MacroPulse, MacroRegime
from engine.liquidity.hob import HOBSignal
from engine.liquidity.bojan_rules import LiquidityReaction
from engine.temporal.tpi import TPISignal
from engine.smc.smc_engine import SMCEngine, SMCSignal

logger = logging.getLogger(__name__)

class VetoType(Enum):
    """Types of veto conditions"""
    MACRO_VETO = "macro_veto"
    LIQUIDITY_VETO = "liquidity_veto"
    TEMPORAL_VETO = "temporal_veto"
    VOLUME_VETO = "volume_veto"
    VOLATILITY_VETO = "volatility_veto"

class SignalStrength(Enum):
    """Signal strength classifications"""
    WEAK = "weak"
    MODERATE = "moderate"
    STRONG = "strong"
    INSTITUTIONAL = "institutional"

@dataclass
class VetoCondition:
    """Veto condition data structure"""
    veto_type: VetoType
    reason: str
    severity: float  # 0-1, how strong the veto is
    affected_domains: List[str]
    metadata: Dict[str, Any]

@dataclass
class DomainSignal:
    """Standardized domain signal"""
    domain: str
    signal_type: str
    direction: str  # 'long', 'short', 'neutral'
    strength: float  # 0-1
    confidence: float  # 0-1
    timestamp: pd.Timestamp
    entry_price: Optional[float]
    stop_loss: Optional[float]
    targets: List[float]
    metadata: Dict[str, Any]

@dataclass
class FusionSignal:
    """Final aggregated signal from fusion engine with macro pulse integration"""
    direction: str
    strength: float
    confidence: float
    quality_score: float
    timestamp: pd.Timestamp
    entry_price: float
    stop_loss: float
    targets: List[float]
    contributing_domains: List[str]
    domain_weights: Dict[str, float]
    veto_conditions: List[VetoCondition]
    macro_pulse: Optional[MacroPulse]  # Integrated macro context
    macro_delta: float  # -0.1 to +0.1 adjustment from macro
    risk_bias: str  # 'risk_on', 'risk_off', 'neutral'
    explainable_factors: Dict[str, Any]  # Human-readable explanations
    metadata: Dict[str, Any]

class FusionEngine:
    """
    Enhanced Fusion Engine for v1.7 with veto logic and domain aggregation.

    Combines signals from:
    - Wyckoff (existing v1.6.2 logic)
    - Liquidity (Bojan HOB reactions)
    - Momentum (existing indicators)
    - Temporal (TPI analysis)
    - Macro Context (SMT signals)

    Features:
    - Intelligent veto conditions
    - Domain-specific weightings
    - Quality scoring
    - Risk-adjusted aggregation
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.fusion_config = config.get('fusion', {})

        # Initialize Macro Pulse Engine
        macro_config = config.get('context', {})
        self.macro_pulse_engine = MacroPulseEngine(macro_config) if macro_config else None

        # Initialize SMC Engine
        smc_config = config.get('domains', {}).get('smc', {})
        self.smc_engine = SMCEngine(smc_config) if smc_config else None

        # Domain weights (configurable) - Updated for v1.7 with SMC
        self.domain_weights = self.fusion_config.get('domain_weights', {
            'wyckoff': 0.20,
            'liquidity': 0.20,
            'smc': 0.25,
            'momentum': 0.20,
            'temporal': 0.15,
            'macro_context': 0.15
        })

        # Minimum requirements
        self.min_domains = self.fusion_config.get('min_domains', 3)
        self.min_confidence = self.fusion_config.get('min_confidence', 0.65)
        self.min_strength = self.fusion_config.get('min_strength', 0.6)

        # Veto thresholds
        self.veto_thresholds = self.fusion_config.get('veto_thresholds', {
            'macro_regime_conflict': 0.8,
            'low_volume_threshold': 0.3,
            'high_volatility_threshold': 3.0,
            'liquidity_grab_protection': 0.7
        })

        # Quality scoring weights
        self.quality_weights = self.fusion_config.get('quality_weights', {
            'domain_confluence': 0.3,
            'signal_strength': 0.25,
            'confidence_consistency': 0.2,
            'institutional_quality': 0.15,
            'temporal_alignment': 0.1
        })

    def aggregate_signals(self, domain_signals: Dict[str, Any],
                         market_data: Dict[str, pd.DataFrame]) -> Optional[FusionSignal]:
        """
        Aggregate signals from all domains with veto logic.

        Args:
            domain_signals: Dict containing signals from each domain
            market_data: Current market data for context

        Returns:
            FusionSignal if valid aggregation possible, None otherwise
        """
        try:
            # 1. Analyze Macro Pulse (overrides all other signals if hard veto)
            macro_pulse = None
            if self.macro_pulse_engine:
                macro_pulse = self.macro_pulse_engine.analyze_macro_pulse(market_data)

                # Hard macro veto overrides everything
                if macro_pulse.suppression_flag:
                    logger.info(f"Hard macro veto: {macro_pulse.regime} - {'; '.join(macro_pulse.notes)}")
                    return None

            # 2. Standardize domain signals
            standardized_signals = self._standardize_domain_signals(domain_signals, market_data)

            if len(standardized_signals) < self.min_domains:
                logger.info(f"Insufficient domains: {len(standardized_signals)} < {self.min_domains}")
                return None

            # 3. Check for additional veto conditions
            veto_conditions = self._check_veto_conditions(standardized_signals, market_data)
            blocking_vetos = [v for v in veto_conditions if v.severity >= 0.8]

            if blocking_vetos:
                logger.info(f"Signal blocked by {len(blocking_vetos)} veto conditions")
                return None

            # 4. Calculate domain consensus
            consensus = self._calculate_domain_consensus(standardized_signals)

            if not consensus['valid']:
                logger.info("No valid domain consensus")
                return None

            # 4. Aggregate signal parameters
            aggregated = self._aggregate_signal_parameters(standardized_signals, consensus)

            # 5. Apply macro delta adjustments
            final_strength, final_confidence = self._apply_macro_adjustments(
                aggregated['strength'], aggregated['confidence'], macro_pulse, consensus['direction']
            )

            # 6. Calculate quality score (including macro factors)
            quality_score = self._calculate_quality_score(standardized_signals, consensus, veto_conditions, macro_pulse)

            # 7. Apply final filters (after macro adjustments)
            if final_confidence < self.min_confidence or final_strength < self.min_strength:
                logger.info(f"Signal below thresholds: conf={final_confidence:.3f}, str={final_strength:.3f}")
                return None

            # 8. Build explainable factors
            explainable_factors = self._build_explainable_factors(
                standardized_signals, consensus, macro_pulse, quality_score
            )

            # 9. Build final fusion signal with macro integration
            return FusionSignal(
                direction=consensus['direction'],
                strength=final_strength,
                confidence=final_confidence,
                quality_score=quality_score,
                timestamp=aggregated['timestamp'],
                entry_price=aggregated['entry_price'],
                stop_loss=aggregated['stop_loss'],
                targets=aggregated['targets'],
                contributing_domains=list(standardized_signals.keys()),
                domain_weights=self._calculate_effective_weights(standardized_signals),
                veto_conditions=veto_conditions,
                macro_pulse=macro_pulse,
                macro_delta=macro_pulse.macro_delta if macro_pulse else 0.0,
                risk_bias=macro_pulse.risk_bias if macro_pulse else 'neutral',
                explainable_factors=explainable_factors,
                metadata={
                    'consensus_data': consensus,
                    'aggregation_data': aggregated,
                    'domain_count': len(standardized_signals),
                    'veto_count': len(veto_conditions),
                    'macro_regime': macro_pulse.regime.value if macro_pulse else 'neutral'
                }
            )

        except Exception as e:
            logger.error(f"Error in signal aggregation: {e}")
            return None

    def _standardize_domain_signals(self, domain_signals: Dict[str, Any],
                                   market_data: Dict[str, pd.DataFrame]) -> Dict[str, DomainSignal]:
        """Standardize signals from different domains into common format"""
        try:
            standardized = {}
            current_time = pd.Timestamp.now()
            current_price = self._get_current_price(market_data)

            # 1. Wyckoff signals (from existing v1.6.2 logic)
            if 'wyckoff' in domain_signals and domain_signals['wyckoff']:
                wyckoff_data = domain_signals['wyckoff']
                standardized['wyckoff'] = DomainSignal(
                    domain='wyckoff',
                    signal_type=wyckoff_data.get('signal_type', 'accumulation'),
                    direction=wyckoff_data.get('direction', 'long'),
                    strength=wyckoff_data.get('strength', 0.5),
                    confidence=wyckoff_data.get('confidence', 0.5),
                    timestamp=current_time,
                    entry_price=current_price,
                    stop_loss=wyckoff_data.get('stop_loss'),
                    targets=wyckoff_data.get('targets', []),
                    metadata=wyckoff_data
                )

            # 2. Liquidity signals (HOB/Bojan reactions)
            if 'liquidity' in domain_signals:
                liquidity_signal = domain_signals['liquidity']
                if isinstance(liquidity_signal, (HOBSignal, LiquidityReaction)):
                    direction = 'long' if liquidity_signal.hob_type.name.startswith('BULLISH') else 'short'
                    if hasattr(liquidity_signal, 'direction'):
                        direction = liquidity_signal.direction

                    standardized['liquidity'] = DomainSignal(
                        domain='liquidity',
                        signal_type='hob_reaction',
                        direction=direction,
                        strength=liquidity_signal.strength,
                        confidence=liquidity_signal.confidence,
                        timestamp=liquidity_signal.timestamp,
                        entry_price=getattr(liquidity_signal, 'entry_price', current_price),
                        stop_loss=getattr(liquidity_signal, 'stop_loss', None),
                        targets=self._extract_targets(liquidity_signal),
                        metadata={'hob_quality': getattr(liquidity_signal, 'quality', None)}
                    )

            # 3. Momentum signals (existing indicators)
            if 'momentum' in domain_signals and domain_signals['momentum']:
                momentum_data = domain_signals['momentum']
                standardized['momentum'] = DomainSignal(
                    domain='momentum',
                    signal_type=momentum_data.get('signal_type', 'momentum_shift'),
                    direction=momentum_data.get('direction', 'neutral'),
                    strength=momentum_data.get('strength', 0.5),
                    confidence=momentum_data.get('confidence', 0.5),
                    timestamp=current_time,
                    entry_price=current_price,
                    stop_loss=momentum_data.get('stop_loss'),
                    targets=momentum_data.get('targets', []),
                    metadata=momentum_data
                )

            # 4. Temporal signals (TPI)
            if 'temporal' in domain_signals:
                temporal_signal = domain_signals['temporal']
                if isinstance(temporal_signal, TPISignal):
                    # Derive direction from TPI signal type
                    direction = self._derive_temporal_direction(temporal_signal)

                    standardized['temporal'] = DomainSignal(
                        domain='temporal',
                        signal_type=temporal_signal.tpi_type.value,
                        direction=direction,
                        strength=temporal_signal.strength,
                        confidence=temporal_signal.confidence,
                        timestamp=temporal_signal.timestamp,
                        entry_price=current_price,
                        stop_loss=None,  # Temporal doesn't provide stops
                        targets=[temporal_signal.price_level] if temporal_signal.price_level else [],
                        metadata={'cycle_data': temporal_signal.cycle_data}
                    )

            # 5. SMC signals (Smart Money Concepts)
            if 'smc' in domain_signals or self.smc_engine:
                # If SMC engine exists, run analysis on current market data
                if self.smc_engine and '1H' in market_data:
                    smc_signal = self.smc_engine.analyze_smc(market_data['1H'])
                elif 'smc' in domain_signals:
                    smc_signal = domain_signals['smc']
                else:
                    smc_signal = None

                if smc_signal and smc_signal.direction != 'neutral':
                    standardized['smc'] = DomainSignal(
                        domain='smc',
                        signal_type='institutional_structure',
                        direction=smc_signal.direction,
                        strength=smc_signal.strength,
                        confidence=smc_signal.confidence,
                        timestamp=smc_signal.timestamp,
                        entry_price=current_price,
                        stop_loss=None,  # SMC provides zones, not specific stops
                        targets=self._extract_smc_targets(smc_signal),
                        metadata={
                            'confluence_score': smc_signal.confluence_score,
                            'institutional_bias': smc_signal.institutional_bias,
                            'order_blocks_count': len(smc_signal.order_blocks),
                            'fvg_count': len(smc_signal.fair_value_gaps),
                            'liquidity_sweeps_count': len(smc_signal.liquidity_sweeps),
                            'trend_state': smc_signal.trend_state.value,
                            'entry_zones': smc_signal.entry_zones
                        }
                    )

            # 6. Macro Context signals (SMT)
            if 'macro_context' in domain_signals:
                smt_signals = domain_signals['macro_context']
                if isinstance(smt_signals, list) and smt_signals:
                    # Aggregate multiple SMT signals
                    best_signal = max(smt_signals, key=lambda x: x.strength * x.confidence)
                    direction = self._derive_smt_direction(best_signal, smt_signals)

                    standardized['macro_context'] = DomainSignal(
                        domain='macro_context',
                        signal_type='smt_confluence',
                        direction=direction,
                        strength=best_signal.strength,
                        confidence=best_signal.confidence,
                        timestamp=best_signal.timestamp,
                        entry_price=current_price,
                        stop_loss=None,  # Macro doesn't provide stops
                        targets=[],
                        metadata={
                            'hps_score': best_signal.hps_score.value,
                            'signal_count': len(smt_signals),
                            'suppression_active': any(s.suppression_active for s in smt_signals)
                        }
                    )

            return standardized

        except Exception as e:
            logger.error(f"Error standardizing domain signals: {e}")
            return {}

    def _check_veto_conditions(self, signals: Dict[str, DomainSignal],
                              market_data: Dict[str, pd.DataFrame]) -> List[VetoCondition]:
        """Check for conditions that should veto the signal"""
        try:
            vetos = []

            # 1. Macro regime conflict veto
            macro_veto = self._check_macro_regime_veto(signals, market_data)
            if macro_veto:
                vetos.append(macro_veto)

            # 2. Low volume veto
            volume_veto = self._check_volume_veto(market_data)
            if volume_veto:
                vetos.append(volume_veto)

            # 3. High volatility veto
            volatility_veto = self._check_volatility_veto(market_data)
            if volatility_veto:
                vetos.append(volatility_veto)

            # 4. Liquidity grab protection
            liquidity_veto = self._check_liquidity_grab_veto(signals, market_data)
            if liquidity_veto:
                vetos.append(liquidity_veto)

            return vetos

        except Exception as e:
            logger.error(f"Error checking veto conditions: {e}")
            return []

    def _calculate_domain_consensus(self, signals: Dict[str, DomainSignal]) -> Dict[str, Any]:
        """Calculate consensus direction and strength across domains"""
        try:
            if not signals:
                return {'valid': False}

            # Count directional bias
            long_weight = 0.0
            short_weight = 0.0
            neutral_weight = 0.0

            for domain, signal in signals.items():
                domain_weight = self.domain_weights.get(domain, 0.2)
                weighted_strength = signal.strength * signal.confidence * domain_weight

                if signal.direction == 'long':
                    long_weight += weighted_strength
                elif signal.direction == 'short':
                    short_weight += weighted_strength
                else:
                    neutral_weight += weighted_strength

            total_weight = long_weight + short_weight + neutral_weight

            if total_weight == 0:
                return {'valid': False}

            # Determine consensus
            long_pct = long_weight / total_weight
            short_pct = short_weight / total_weight
            neutral_pct = neutral_weight / total_weight

            # Require clear directional bias
            if max(long_pct, short_pct) < 0.6:  # 60% minimum consensus
                return {'valid': False}

            direction = 'long' if long_pct > short_pct else 'short'
            consensus_strength = max(long_pct, short_pct)

            return {
                'valid': True,
                'direction': direction,
                'consensus_strength': consensus_strength,
                'long_weight': long_pct,
                'short_weight': short_pct,
                'neutral_weight': neutral_pct,
                'participating_domains': len(signals)
            }

        except Exception as e:
            logger.error(f"Error calculating domain consensus: {e}")
            return {'valid': False}

    def _aggregate_signal_parameters(self, signals: Dict[str, DomainSignal],
                                   consensus: Dict[str, Any]) -> Dict[str, Any]:
        """Aggregate signal parameters using weighted averaging"""
        try:
            current_time = pd.Timestamp.now()
            current_price = self._get_current_price_from_signals(signals)

            # Weighted aggregation
            total_weight = 0.0
            weighted_strength = 0.0
            weighted_confidence = 0.0
            all_targets = []
            stop_losses = []

            for domain, signal in signals.items():
                if signal.direction != consensus['direction']:
                    continue  # Skip opposing signals

                domain_weight = self.domain_weights.get(domain, 0.2)
                signal_weight = domain_weight * signal.confidence

                weighted_strength += signal.strength * signal_weight
                weighted_confidence += signal.confidence * signal_weight
                total_weight += signal_weight

                # Collect targets and stops
                if signal.targets:
                    all_targets.extend(signal.targets)
                if signal.stop_loss:
                    stop_losses.append(signal.stop_loss)

            if total_weight == 0:
                return {'valid': False}

            # Calculate aggregated values
            final_strength = weighted_strength / total_weight
            final_confidence = weighted_confidence / total_weight

            # Calculate stop loss (most conservative)
            if stop_losses:
                if consensus['direction'] == 'long':
                    final_stop = max(stop_losses)  # Tightest stop for longs
                else:
                    final_stop = min(stop_losses)  # Tightest stop for shorts
            else:
                # Default stop based on ATR or percentage
                final_stop = self._calculate_default_stop(current_price, consensus['direction'])

            # Process targets
            final_targets = self._process_targets(all_targets, current_price, consensus['direction'])

            return {
                'valid': True,
                'strength': final_strength,
                'confidence': final_confidence,
                'timestamp': current_time,
                'entry_price': current_price,
                'stop_loss': final_stop,
                'targets': final_targets,
                'total_weight': total_weight
            }

        except Exception as e:
            logger.error(f"Error aggregating signal parameters: {e}")
            return {'valid': False}

    def _calculate_quality_score(self, signals: Dict[str, DomainSignal],
                                consensus: Dict[str, Any], vetos: List[VetoCondition]) -> float:
        """Calculate overall signal quality score"""
        try:
            quality_components = {}

            # 1. Domain confluence (more domains = higher quality)
            domain_count = len(signals)
            max_domains = len(self.domain_weights)
            quality_components['domain_confluence'] = min(1.0, domain_count / max_domains)

            # 2. Signal strength (weighted average)
            total_strength = sum(s.strength for s in signals.values())
            quality_components['signal_strength'] = total_strength / len(signals) if signals else 0

            # 3. Confidence consistency (lower std = higher quality)
            confidences = [s.confidence for s in signals.values()]
            confidence_std = np.std(confidences) if len(confidences) > 1 else 0
            quality_components['confidence_consistency'] = max(0, 1.0 - confidence_std)

            # 4. Institutional quality (presence of high-quality signals)
            institutional_signals = [
                s for s in signals.values()
                if (s.domain == 'liquidity' and
                    s.metadata.get('hob_quality') == 'institutional') or
                (s.domain == 'macro_context' and
                    s.metadata.get('hps_score', 0) >= 2)
            ]
            quality_components['institutional_quality'] = min(1.0, len(institutional_signals) / 2)

            # 5. Temporal alignment (if temporal signal present and aligned)
            temporal_alignment = 0.5  # Default neutral
            if 'temporal' in signals:
                temporal_signal = signals['temporal']
                if temporal_signal.direction == consensus['direction']:
                    temporal_alignment = temporal_signal.confidence
            quality_components['temporal_alignment'] = temporal_alignment

            # Calculate weighted quality score
            total_quality = 0.0
            for component, weight in self.quality_weights.items():
                total_quality += quality_components.get(component, 0) * weight

            # Apply veto penalty
            veto_penalty = sum(v.severity for v in vetos) * 0.1
            final_quality = max(0.0, total_quality - veto_penalty)

            return min(1.0, final_quality)

        except Exception as e:
            logger.error(f"Error calculating quality score: {e}")
            return 0.0

    # Helper methods for veto conditions
    def _check_macro_regime_veto(self, signals: Dict[str, DomainSignal],
                                market_data: Dict[str, pd.DataFrame]) -> Optional[VetoCondition]:
        """Check for macro regime conflicts"""
        try:
            if 'macro_context' in signals:
                macro_signal = signals['macro_context']
                if macro_signal.metadata.get('suppression_active'):
                    return VetoCondition(
                        veto_type=VetoType.MACRO_VETO,
                        reason="SMT suppression active",
                        severity=0.9,
                        affected_domains=['all'],
                        metadata={'suppression_details': macro_signal.metadata}
                    )
            return None
        except Exception:
            return None

    def _check_volume_veto(self, market_data: Dict[str, pd.DataFrame]) -> Optional[VetoCondition]:
        """Check for insufficient volume"""
        try:
            if '1H' not in market_data:
                return None

            df = market_data['1H']
            recent_volume = df['volume'].tail(5).mean()
            avg_volume = df['volume'].tail(50).mean()

            volume_ratio = recent_volume / avg_volume if avg_volume > 0 else 1.0

            if volume_ratio < self.veto_thresholds['low_volume_threshold']:
                return VetoCondition(
                    veto_type=VetoType.VOLUME_VETO,
                    reason=f"Low volume: {volume_ratio:.2f}x average",
                    severity=0.7,
                    affected_domains=['liquidity', 'momentum'],
                    metadata={'volume_ratio': volume_ratio}
                )
            return None
        except Exception:
            return None

    def _check_volatility_veto(self, market_data: Dict[str, pd.DataFrame]) -> Optional[VetoCondition]:
        """Check for excessive volatility"""
        try:
            if '1H' not in market_data:
                return None

            df = market_data['1H']
            returns = df['close'].pct_change().tail(24)  # Last 24 hours
            volatility = returns.std() * np.sqrt(24)  # Annualized

            if volatility > self.veto_thresholds['high_volatility_threshold']:
                return VetoCondition(
                    veto_type=VetoType.VOLATILITY_VETO,
                    reason=f"High volatility: {volatility:.1%}",
                    severity=0.6,
                    affected_domains=['temporal', 'wyckoff'],
                    metadata={'volatility': volatility}
                )
            return None
        except Exception:
            return None

    def _check_liquidity_grab_veto(self, signals: Dict[str, DomainSignal],
                                  market_data: Dict[str, pd.DataFrame]) -> Optional[VetoCondition]:
        """Check for potential liquidity grab scenarios"""
        # This would implement liquidity grab detection
        # For now, return None (placeholder)
        return None

    # Helper methods
    def _get_current_price(self, market_data: Dict[str, pd.DataFrame]) -> float:
        """Get current price from market data"""
        try:
            for tf in ['1H', '4H', '1D']:
                if tf in market_data and len(market_data[tf]) > 0:
                    return market_data[tf]['close'].iloc[-1]
            return 50000.0  # Default fallback
        except Exception:
            return 50000.0

    def _get_current_price_from_signals(self, signals: Dict[str, DomainSignal]) -> float:
        """Get current price from signal entry prices"""
        try:
            entry_prices = [s.entry_price for s in signals.values() if s.entry_price]
            return np.mean(entry_prices) if entry_prices else 50000.0
        except Exception:
            return 50000.0

    def _extract_targets(self, signal) -> List[float]:
        """Extract target prices from various signal types"""
        try:
            if hasattr(signal, 'targets') and signal.targets:
                return signal.targets
            elif hasattr(signal, 'target_1') and signal.target_1:
                targets = [signal.target_1]
                if hasattr(signal, 'target_2') and signal.target_2:
                    targets.append(signal.target_2)
                return targets
            elif hasattr(signal, 'exit_levels') and signal.exit_levels:
                return [v for k, v in signal.exit_levels.items() if k.startswith('target')]
            return []
        except Exception:
            return []

    def _extract_smc_targets(self, smc_signal: SMCSignal) -> List[float]:
        """Extract target prices from SMC entry zones"""
        try:
            targets = []

            # Use entry zones as targets
            for zone_low, zone_high in smc_signal.entry_zones:
                zone_center = (zone_low + zone_high) / 2
                targets.append(zone_center)

            # Also consider order block levels
            for ob in smc_signal.order_blocks:
                ob_center = (ob.low + ob.high) / 2
                targets.append(ob_center)

            # Sort and return unique targets (up to 3)
            unique_targets = sorted(list(set(targets)))
            return unique_targets[:3]

        except Exception:
            return []

    def _derive_temporal_direction(self, tpi_signal: TPISignal) -> str:
        """Derive trading direction from TPI signal"""
        try:
            # Simple mapping - could be more sophisticated
            if tpi_signal.tpi_type.name in ['TEMPORAL_SUPPORT', 'CYCLE_COMPLETION']:
                return 'long'
            elif tpi_signal.tpi_type.name in ['TEMPORAL_RESISTANCE']:
                return 'short'
            else:
                return 'neutral'
        except Exception:
            return 'neutral'

    def _derive_smt_direction(self, best_signal: SMTSignal, all_signals: List[SMTSignal]) -> str:
        """Derive trading direction from SMT signals"""
        try:
            # Count bullish vs bearish SMT signals
            bullish_count = sum(1 for s in all_signals
                               if s.signal_type.name in ['USDT_STAGNATION', 'TOTAL3_DIVERGENCE'])
            bearish_count = sum(1 for s in all_signals
                               if s.signal_type.name in ['BTC_WEDGE_BREAK'])

            if bullish_count > bearish_count:
                return 'long'
            elif bearish_count > bullish_count:
                return 'short'
            else:
                return 'neutral'
        except Exception:
            return 'neutral'

    def _calculate_default_stop(self, entry_price: float, direction: str) -> float:
        """Calculate default stop loss"""
        try:
            # Simple percentage-based stop
            stop_pct = 0.02  # 2%
            if direction == 'long':
                return entry_price * (1 - stop_pct)
            else:
                return entry_price * (1 + stop_pct)
        except Exception:
            return entry_price

    def _process_targets(self, all_targets: List[float], entry_price: float, direction: str) -> List[float]:
        """Process and rank targets"""
        try:
            if not all_targets:
                # Default targets based on risk-reward
                if direction == 'long':
                    return [entry_price * 1.02, entry_price * 1.04, entry_price * 1.06]
                else:
                    return [entry_price * 0.98, entry_price * 0.96, entry_price * 0.94]

            # Filter and sort targets
            valid_targets = []
            for target in all_targets:
                if direction == 'long' and target > entry_price:
                    valid_targets.append(target)
                elif direction == 'short' and target < entry_price:
                    valid_targets.append(target)

            # Sort by distance from entry
            if direction == 'long':
                valid_targets.sort()
            else:
                valid_targets.sort(reverse=True)

            return valid_targets[:3]  # Max 3 targets

        except Exception:
            return []

    def _calculate_effective_weights(self, signals: Dict[str, DomainSignal]) -> Dict[str, float]:
        """Calculate effective weights based on signal quality"""
        try:
            effective_weights = {}
            total_quality = sum(s.strength * s.confidence for s in signals.values())

            for domain, signal in signals.items():
                base_weight = self.domain_weights.get(domain, 0.2)
                quality_factor = (signal.strength * signal.confidence) / total_quality if total_quality > 0 else 1.0
                effective_weights[domain] = base_weight * quality_factor

            # Normalize to sum to 1.0
            weight_sum = sum(effective_weights.values())
            if weight_sum > 0:
                effective_weights = {k: v/weight_sum for k, v in effective_weights.items()}

            return effective_weights

        except Exception:
            return self.domain_weights

    def _apply_macro_adjustments(self, strength: float, confidence: float,
                                macro_pulse: Optional[MacroPulse], direction: str) -> Tuple[float, float]:
        """Apply macro delta adjustments to signal strength and confidence"""
        try:
            if not macro_pulse:
                return strength, confidence

            adjusted_strength = strength
            adjusted_confidence = confidence

            # Apply macro delta (bounded ±0.1)
            macro_adjustment = macro_pulse.macro_delta

            # Direction alignment check
            if (direction == 'long' and macro_pulse.risk_bias == 'risk_on') or \
               (direction == 'short' and macro_pulse.risk_bias == 'risk_off'):
                # Aligned with macro - apply positive adjustment
                adjusted_strength = min(1.0, strength + abs(macro_adjustment))
                adjusted_confidence = min(1.0, confidence + abs(macro_adjustment) * 0.5)
            elif (direction == 'long' and macro_pulse.risk_bias == 'risk_off') or \
                 (direction == 'short' and macro_pulse.risk_bias == 'risk_on'):
                # Counter to macro - apply negative adjustment
                adjusted_strength = max(0.1, strength - abs(macro_adjustment))
                adjusted_confidence = max(0.1, confidence - abs(macro_adjustment) * 0.5)

            # Additional boost for specific macro signals
            if macro_pulse.plus_ones:
                boost_factor = len(macro_pulse.plus_ones) * 0.02  # 2% per plus_one
                adjusted_strength = min(1.0, adjusted_strength + boost_factor)

            return adjusted_strength, adjusted_confidence

        except Exception as e:
            logger.error(f"Error applying macro adjustments: {e}")
            return strength, confidence

    def _calculate_quality_score(self, signals: Dict[str, DomainSignal],
                                consensus: Dict[str, Any], vetos: List[VetoCondition],
                                macro_pulse: Optional[MacroPulse] = None) -> float:
        """Calculate overall signal quality score including macro factors"""
        try:
            quality_components = {}

            # 1. Domain confluence (more domains = higher quality)
            domain_count = len(signals)
            max_domains = len(self.domain_weights)
            quality_components['domain_confluence'] = min(1.0, domain_count / max_domains)

            # 2. Signal strength (weighted average)
            total_strength = sum(s.strength for s in signals.values())
            quality_components['signal_strength'] = total_strength / len(signals) if signals else 0

            # 3. Confidence consistency (lower std = higher quality)
            confidences = [s.confidence for s in signals.values()]
            confidence_std = np.std(confidences) if len(confidences) > 1 else 0
            quality_components['confidence_consistency'] = max(0, 1.0 - confidence_std)

            # 4. Institutional quality (presence of high-quality signals)
            institutional_signals = [
                s for s in signals.values()
                if (s.domain == 'liquidity' and
                    s.metadata.get('hob_quality') == 'institutional') or
                (s.domain == 'macro_context' and
                    s.metadata.get('hps_score', 0) >= 2)
            ]
            quality_components['institutional_quality'] = min(1.0, len(institutional_signals) / 2)

            # 5. Temporal alignment (if temporal signal present and aligned)
            temporal_alignment = 0.5  # Default neutral
            if 'temporal' in signals:
                temporal_signal = signals['temporal']
                if temporal_signal.direction == consensus['direction']:
                    temporal_alignment = temporal_signal.confidence
            quality_components['temporal_alignment'] = temporal_alignment

            # 6. Macro alignment bonus (new)
            macro_alignment = 0.5  # Default neutral
            if macro_pulse:
                # Bonus for risk-on alignment with long signals
                if ((consensus['direction'] == 'long' and macro_pulse.risk_bias == 'risk_on') or
                    (consensus['direction'] == 'short' and macro_pulse.risk_bias == 'risk_off')):
                    macro_alignment = 0.8 + macro_pulse.boost_strength * 0.2
                # Penalty for risk-off misalignment
                elif macro_pulse.veto_strength > 0.3:
                    macro_alignment = max(0.2, 0.5 - macro_pulse.veto_strength * 0.3)

            # Update weights to include macro alignment
            enhanced_weights = {
                'domain_confluence': 0.25,
                'signal_strength': 0.20,
                'confidence_consistency': 0.15,
                'institutional_quality': 0.15,
                'temporal_alignment': 0.10,
                'macro_alignment': 0.15
            }

            # Calculate weighted quality score
            total_quality = 0.0
            quality_components['macro_alignment'] = macro_alignment
            for component, weight in enhanced_weights.items():
                total_quality += quality_components.get(component, 0) * weight

            # Apply veto penalty
            veto_penalty = sum(v.severity for v in vetos) * 0.1
            final_quality = max(0.0, total_quality - veto_penalty)

            return min(1.0, final_quality)

        except Exception as e:
            logger.error(f"Error calculating quality score: {e}")
            return 0.0

    def _build_explainable_factors(self, signals: Dict[str, DomainSignal],
                                  consensus: Dict[str, Any], macro_pulse: Optional[MacroPulse],
                                  quality_score: float) -> Dict[str, Any]:
        """Build human-readable explanation of signal factors"""
        try:
            factors = {
                'signal_summary': f"{consensus['direction'].upper()} signal from {len(signals)} domains",
                'domain_breakdown': {},
                'macro_context': {},
                'quality_factors': {},
                'risk_considerations': []
            }

            # Domain breakdown
            for domain, signal in signals.items():
                factors['domain_breakdown'][domain] = {
                    'strength': f"{signal.strength:.2f}",
                    'confidence': f"{signal.confidence:.2f}",
                    'signal_type': signal.signal_type,
                    'contribution': f"{self.domain_weights.get(domain, 0.2):.1%}"
                }

            # Macro context
            if macro_pulse:
                factors['macro_context'] = {
                    'regime': macro_pulse.regime.value,
                    'risk_bias': macro_pulse.risk_bias,
                    'macro_delta': f"{macro_pulse.macro_delta:+.3f}",
                    'active_signals': len(macro_pulse.active_signals),
                    'notes': macro_pulse.notes[:3],  # Top 3 notes
                    'plus_ones': macro_pulse.plus_ones
                }

                # Risk considerations from macro
                if macro_pulse.veto_strength > 0.5:
                    factors['risk_considerations'].append(f"Elevated macro risk: {macro_pulse.veto_strength:.2f}")
                if macro_pulse.regime in [MacroRegime.STAGFLATION, MacroRegime.RISK_OFF]:
                    factors['risk_considerations'].append(f"Challenging macro regime: {macro_pulse.regime.value}")

            # Quality factors
            factors['quality_factors'] = {
                'overall_score': f"{quality_score:.2f}",
                'domain_confluence': f"{len(signals)}/{len(self.domain_weights)} domains active",
                'consensus_strength': f"{consensus.get('consensus_strength', 0):.2f}",
                'institutional_grade': any(
                    s.metadata.get('hob_quality') == 'institutional' for s in signals.values()
                )
            }

            # Additional risk considerations
            if consensus.get('consensus_strength', 0) < 0.7:
                factors['risk_considerations'].append("Moderate domain consensus")
            if quality_score < 0.7:
                factors['risk_considerations'].append("Below optimal quality threshold")

            return factors

        except Exception as e:
            logger.error(f"Error building explainable factors: {e}")
            return {'error': 'Failed to generate explanation'}