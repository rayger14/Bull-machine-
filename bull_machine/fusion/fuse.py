"""Bull Machine v1.3 - Fusion Engine with MTF Gating"""

import logging
from typing import Dict, Optional

from bull_machine.core.types import Signal, SyncReport


class FusionEngineV1_3:
    """Fusion engine that integrates v1.2.1 modules with v1.3 MTF sync."""

    def __init__(self, config: dict):
        self.config = config
        self.fusion_config = config.get('fusion', {})
        self.weights = self.fusion_config.get('weights', {
            'wyckoff': 0.60,
            'liquidity': 0.40
        })
        self.enter_threshold = self.fusion_config.get('enter_threshold', 0.35)

    def fuse_with_mtf(self, modules: dict, sync_report: Optional[SyncReport]) -> Optional[Signal]:
        """
        Fuse module signals with MTF sync gating.

        Args:
            modules: Dict with keys:
                - 'wyckoff': WyckoffResult
                - 'liquidity': LiquidityResult
                - Optional: 'structure', 'momentum', 'volume', 'context'
            sync_report: MTF sync report (if MTF enabled)

        Returns:
            Signal if conditions met, None otherwise
        """
        # Extract module data
        wyckoff = modules.get('wyckoff')
        liquidity = modules.get('liquidity')

        if not wyckoff or not liquidity:
            logging.warning("Missing required modules for fusion")
            return None

        # Calculate base scores
        wyckoff_score = self._calculate_wyckoff_score(wyckoff)
        liquidity_score = liquidity.overall_score if hasattr(liquidity, 'overall_score') else liquidity.score

        # Add optional v1.2.1 modules if present
        scores = {
            'wyckoff': wyckoff_score,
            'liquidity': liquidity_score
        }

        # Check for v1.2.1 modules
        if 'structure' in modules:
            scores['structure'] = modules['structure'].get('score', 0.0)
        if 'momentum' in modules:
            scores['momentum'] = modules['momentum'].get('score', 0.0)
        if 'volume' in modules:
            scores['volume'] = modules['volume'].get('score', 0.0)
        if 'context' in modules:
            scores['context'] = modules['context'].get('score', 0.0)

        # Calculate weighted fusion score
        fusion_score = self._calculate_fusion_score(scores)

        # Apply MTF sync adjustments
        effective_threshold = self.enter_threshold

        if sync_report:
            # Apply MTF gating
            if sync_report.decision == 'veto':
                logging.info(f"MTF VETO: {', '.join(sync_report.notes)}")
                return None
            elif sync_report.decision == 'raise':
                effective_threshold += sync_report.threshold_bump
                logging.info(f"MTF RAISE: Threshold {self.enter_threshold:.2f} -> {effective_threshold:.2f}")

            # Log alignment
            logging.info(f"MTF Alignment: {sync_report.alignment_score:.1%}")

        # Check threshold
        if fusion_score < effective_threshold:
            logging.info(f"Below threshold: {fusion_score:.3f} < {effective_threshold:.3f}")
            return None

        # Determine side (never emit 'neutral')
        side = self._determine_side(wyckoff, liquidity)
        if side == 'neutral':
            logging.info("Cannot determine clear side (neutral)")
            return None

        # Build signal
        confidence = fusion_score
        reasons = self._build_reasons(wyckoff, liquidity, sync_report)

        signal = Signal(
            ts=0,  # Will be set by caller
            side=side,
            confidence=confidence,
            reasons=reasons,
            ttl_bars=self._calculate_ttl(wyckoff, confidence)
        )

        logging.info(f"Signal generated: {side.upper()} @ {confidence:.3f}")
        return signal

    def _calculate_wyckoff_score(self, wyckoff) -> float:
        """Calculate Wyckoff score from result."""
        if hasattr(wyckoff, 'confidence'):
            return wyckoff.confidence

        # Fallback: average phase and trend confidence
        phase_conf = getattr(wyckoff, 'phase_confidence', 0.5)
        trend_conf = getattr(wyckoff, 'trend_confidence', 0.5)
        return (phase_conf + trend_conf) / 2

    def _calculate_fusion_score(self, scores: Dict[str, float]) -> float:
        """Calculate weighted fusion score."""
        total_score = 0.0
        total_weight = 0.0

        for module, score in scores.items():
            weight = self.weights.get(module, 0.0)
            total_score += score * weight
            total_weight += weight

        if total_weight > 0:
            return total_score / total_weight
        return 0.0

    def _determine_side(self, wyckoff, liquidity) -> str:
        """
        Determine trade side, never returning 'neutral'.

        Priority:
        1. Wyckoff bias if not neutral
        2. Liquidity pressure if not neutral
        3. Fallback based on scores
        """
        # First try Wyckoff
        if hasattr(wyckoff, 'bias') and wyckoff.bias != 'neutral':
            return 'long' if wyckoff.bias in ['long', 'bullish'] else 'short'

        # Then liquidity
        if hasattr(liquidity, 'pressure'):
            if liquidity.pressure == 'bullish':
                return 'long'
            elif liquidity.pressure == 'bearish':
                return 'short'

        # Fallback: use score differentials
        if hasattr(liquidity, 'bullish_score') and hasattr(liquidity, 'bearish_score'):
            if liquidity.bullish_score > liquidity.bearish_score:
                return 'long'
            elif liquidity.bearish_score > liquidity.bullish_score:
                return 'short'

        # Last resort: slight bias based on overall scores
        # This ensures we never return neutral
        wyckoff_score = self._calculate_wyckoff_score(wyckoff)
        if wyckoff_score > 0.5:
            return 'long'
        else:
            return 'short'

    def _build_reasons(self, wyckoff, liquidity, sync_report) -> list:
        """Build signal reasons list."""
        reasons = []

        # Wyckoff reason
        if hasattr(wyckoff, 'phase') and hasattr(wyckoff, 'bias'):
            reasons.append(f"Wyckoff {wyckoff.phase} {wyckoff.bias}")

        # Liquidity reason
        if hasattr(liquidity, 'pressure'):
            reasons.append(f"Liquidity {liquidity.pressure}")

        # MTF reason
        if sync_report and sync_report.alignment_score > 0.7:
            reasons.append(f"MTF aligned ({sync_report.alignment_score:.0%})")

        # Fusion score
        reasons.append("Fusion score met")

        return reasons[:3]  # Limit to 3 reasons

    def _calculate_ttl(self, wyckoff, confidence) -> int:
        """Calculate signal TTL based on Wyckoff phase and confidence."""
        base_ttl = 20

        # Adjust for Wyckoff phase
        if hasattr(wyckoff, 'phase'):
            if wyckoff.phase in ['C', 'D']:
                base_ttl = 25
            elif wyckoff.phase in ['E']:
                base_ttl = 30
            elif wyckoff.phase in ['A', 'B']:
                base_ttl = 15

        # Adjust for confidence
        if confidence > 0.7:
            base_ttl = int(base_ttl * 1.2)
        elif confidence < 0.4:
            base_ttl = int(base_ttl * 0.8)

        return max(10, min(40, base_ttl))
