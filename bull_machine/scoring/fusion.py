"""
7-Layer Confluence Fusion Engine
Combines all layer scores with v1.4.1 weights and guardrails
"""

import logging
from typing import Dict

import pandas as pd


class FusionEngineV141:
    """
    Enhanced fusion engine for v1.4.1 with proper weights and guardrails.
    """

    def __init__(self, config: Dict):
        """Initialize with configuration."""
        self.config = config
        self.weights = config.get('weights', self._default_weights())
        self.features = config.get('features', {})
        self.thresholds = config.get('signals', {})

        # Validate weights sum to 1.0
        weight_sum = sum(self.weights.values())
        if abs(weight_sum - 1.0) > 0.01:
            logging.warning(f"Weights sum to {weight_sum:.3f}, normalizing...")
            self.weights = {k: v / weight_sum for k, v in self.weights.items()}

        logging.info(f"Fusion engine v1.4.1 initialized with weights: {self.weights}")

    def _default_weights(self) -> Dict[str, float]:
        """Default v1.4.1 layer weights."""
        return {
            'wyckoff': 0.30,
            'liquidity': 0.25,
            'structure': 0.15,
            'momentum': 0.15,
            'volume': 0.15,
            'context': 0.05,
            'mtf': 0.10
        }

    def regime_filter(self, df, wyckoff_score: float, wyckoff_context: Dict = None) -> bool:
        """Check for regime filter - suppress low-vol A/C phases, allow high-vol opportunities."""

        if not wyckoff_context:
            return True  # Pass if no context available

        phase = wyckoff_context.get('phase', 'unknown')

        if phase in ('A', 'C', 'accumulation_C', 'distribution_C'):
            # Check volume context
            if len(df) >= 20:
                vol_ratio = df['volume'].iloc[-1] / df['volume'].rolling(20).mean().iloc[-1]

                # Allow high-vol A/C entries (breakouts, volatile reversals)
                if vol_ratio >= 1.2 or wyckoff_score >= 0.85:
                    return True  # Override veto for high-vol or strong Wyckoff

                # Suppress low-volume A/C phases with weak Wyckoff scores
                if vol_ratio < 1.2 and wyckoff_score < 0.85:
                    logging.info(f"Regime veto: low_vol_{phase}, vol_ratio={vol_ratio:.2f}, wyckoff={wyckoff_score:.2f}")
                    # Log for telemetry
                    self._log_telemetry('regime_veto', {'phase': phase, 'vol_ratio': vol_ratio, 'wyckoff': wyckoff_score})
                    return False

        return True

    def _log_telemetry(self, event: str, data: Dict) -> None:
        """Log telemetry data."""
        try:
            import json
            from pathlib import Path

            telemetry_dir = Path("reports/telemetry")
            telemetry_dir.mkdir(parents=True, exist_ok=True)

            telemetry_file = telemetry_dir / "layer_masks.json"

            # Read existing telemetry
            if telemetry_file.exists():
                with open(telemetry_file, 'r') as f:
                    telemetry = json.load(f)
            else:
                telemetry = {}

            # Add new event
            telemetry[event] = telemetry.get(event, 0) + 1
            telemetry[f'{event}_latest'] = data

            # Write back
            with open(telemetry_file, 'w') as f:
                json.dump(telemetry, f, indent=2)

        except Exception as e:
            logging.debug(f"Telemetry logging failed: {e}")

    def fuse_scores(self, layer_scores: Dict[str, float],
                   quality_floors: Dict[str, float] = None,
                   wyckoff_context: Dict = None,
                   df = None) -> Dict:
        """
        Fuse individual layer scores into final confluence score with enhanced logic.

        Args:
            layer_scores: Dict of layer name -> score (0-1)
            quality_floors: Optional per-layer minimum quality thresholds
            wyckoff_context: Additional Wyckoff context (trap_score, reclaim_speed)

        Returns:
            Dict with fusion results and metadata
        """

        if not layer_scores:
            return {
                'aggregate': 0.0,
                'weighted_score': 0.0,
                'layer_contributions': {},
                'global_veto': True,
                'reason': 'no_layer_scores'
            }

        # Apply regime filter first
        if df is not None and layer_scores.get('wyckoff'):
            if not self.regime_filter(df, layer_scores['wyckoff'], wyckoff_context):
                return {
                    'aggregate': 0.0,
                    'weighted_score': 0.0,
                    'layer_contributions': {},
                    'global_veto': True,
                    'reason': 'regime_filter_veto'
                }

        # Apply quality floors if provided
        filtered_scores = {}
        for layer, score in layer_scores.items():
            floor = quality_floors.get(layer, 0.0) if quality_floors else 0.0
            if score >= floor:
                filtered_scores[layer] = score
            else:
                logging.debug(f"Layer {layer} score {score:.3f} below floor {floor:.3f}")

        # Check for missing critical layers
        required_layers = ['wyckoff', 'liquidity', 'structure']
        missing_critical = [l for l in required_layers if l not in filtered_scores]

        if missing_critical:
            return {
                'aggregate': 0.0,
                'weighted_score': 0.0,
                'layer_contributions': {},
                'global_veto': True,
                'reason': f'missing_critical_layers: {missing_critical}'
            }

        # Apply Wyckoff enhancements if context provided
        enhanced_scores = filtered_scores.copy()
        trap_penalty = 0.0
        reclaim_bonus = 0.0

        if wyckoff_context and 'wyckoff' in enhanced_scores:
            # Apply trap penalty
            trap_score = wyckoff_context.get('trap_score', 0.0)
            trap_penalty = trap_score
            enhanced_scores['wyckoff'] = max(0.1, enhanced_scores['wyckoff'] - trap_penalty)

            # Apply reclaim speed bonus
            reclaim_speed = wyckoff_context.get('reclaim_speed', 0.0)
            reclaim_bonus = reclaim_speed * 0.15  # Max 0.15 boost
            enhanced_scores['wyckoff'] = min(0.9, enhanced_scores['wyckoff'] + reclaim_bonus)

            logging.debug(f"Wyckoff enhancements: trap_penalty={trap_penalty:.3f}, "
                         f"reclaim_bonus={reclaim_bonus:.3f}")

        # Calculate weighted contributions
        contributions = {}
        total_weight = 0

        for layer, weight in self.weights.items():
            if layer in enhanced_scores:
                # Apply Bojan capping for v1.4.1
                if layer == 'bojan':
                    score = min(0.6, enhanced_scores[layer])  # Cap Bojan influence
                else:
                    score = enhanced_scores[layer]

                contributions[layer] = score * weight
                total_weight += weight

        # Normalize by actual total weight (in case some layers missing)
        if total_weight > 0:
            weighted_score = sum(contributions.values()) / total_weight
        else:
            weighted_score = 0.0

        # Calculate simple aggregate (unweighted mean of enhanced scores)
        aggregate = sum(enhanced_scores.values()) / len(enhanced_scores) if enhanced_scores else 0.0

        # Global veto checks
        global_veto = self._check_global_veto(aggregate, enhanced_scores)

        # MTF gate (if enabled)
        mtf_gate = self._check_mtf_gate(enhanced_scores)

        result = {
            'aggregate': aggregate,
            'weighted_score': weighted_score,
            'layer_contributions': contributions,
            'global_veto': global_veto,
            'mtf_gate': mtf_gate,
            'total_weight': total_weight,
            'layers_active': list(enhanced_scores.keys()),
            'reason': 'success' if not global_veto else 'global_veto',
            'wyckoff_adjustments': {
                'trap_penalty': trap_penalty,
                'reclaim_bonus': reclaim_bonus,
                'enhanced_wyckoff_score': enhanced_scores.get('wyckoff', 0)
            }
        }

        logging.debug(f"Fusion result: agg={aggregate:.3f}, weighted={weighted_score:.3f}, "
                     f"veto={global_veto}, layers={len(filtered_scores)}")

        return result

    def _check_global_veto(self, aggregate: float, layer_scores: Dict) -> bool:
        """Check for global veto conditions."""

        # Aggregate too low
        min_aggregate = self.thresholds.get('aggregate_floor', 0.35)
        if aggregate < min_aggregate:
            return True

        # Context stress (macro events)
        context_floor = self.thresholds.get('context_floor', 0.30)
        if layer_scores.get('context', 1.0) < context_floor:
            return True

        # Critical layer failure
        if layer_scores.get('wyckoff', 0) < 0.25:
            return True

        return False

    def _check_mtf_gate(self, layer_scores: Dict) -> bool:
        """Check MTF alignment gate."""

        if not self.features.get('mtf_gate_enabled', True):
            return True  # Pass if gate disabled

        mtf_score = layer_scores.get('mtf', 1.0)
        mtf_threshold = self.thresholds.get('mtf_threshold', 0.60)

        return mtf_score >= mtf_threshold

    def should_enter(self, fusion_result: Dict) -> bool:
        """
        Determine if conditions are met for trade entry.
        """

        if fusion_result['global_veto']:
            return False

        if not fusion_result['mtf_gate']:
            return False

        # Use weighted score for entry decisions
        enter_threshold = self.thresholds.get('enter_threshold', 0.72)
        return fusion_result['weighted_score'] >= enter_threshold

    def run_ablation(self, df: pd.DataFrame, layer_configs: Dict) -> Dict:
        """
        Run ablation study to measure layer contributions.
        """

        sets = [
            ['wyckoff'],
            ['wyckoff', 'liquidity'],
            ['wyckoff', 'liquidity', 'structure'],
            ['wyckoff', 'liquidity', 'structure', 'momentum'],
            ['wyckoff', 'liquidity', 'structure', 'momentum', 'volume'],
            ['wyckoff', 'liquidity', 'structure', 'momentum', 'volume', 'context'],
            ['wyckoff', 'liquidity', 'structure', 'momentum', 'volume', 'context', 'mtf']
        ]

        if self.features.get('bojan', False):
            # Add Bojan to full set
            sets.append(['wyckoff', 'liquidity', 'structure', 'momentum', 'volume', 'context', 'mtf', 'bojan'])

        results = {}

        for layer_set in sets:
            # Mock layer scores for ablation
            mock_scores = {}
            for layer in layer_set:
                if layer == 'wyckoff':
                    mock_scores[layer] = 0.75
                elif layer == 'liquidity':
                    mock_scores[layer] = 0.70
                elif layer == 'bojan':
                    mock_scores[layer] = 0.80  # Will be capped to 0.6
                else:
                    mock_scores[layer] = 0.65

            # Fuse scores
            fusion = self.fuse_scores(mock_scores)

            # Simulate basic performance metrics
            entry_signals = 1 if self.should_enter(fusion) else 0
            estimated_sharpe = fusion['weighted_score'] * 2.0 - 0.5  # Rough estimate

            results['+'.join(layer_set)] = {
                'weighted_score': fusion['weighted_score'],
                'aggregate': fusion['aggregate'],
                'global_veto': fusion['global_veto'],
                'entry_signals': entry_signals,
                'estimated_sharpe': estimated_sharpe,
                'layer_count': len(layer_set)
            }

        return results
