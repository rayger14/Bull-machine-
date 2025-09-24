"""Enhanced Bull Machine v1.4 Fusion Engine with Quality Gates"""

import logging
from typing import Any, Dict, List, Optional

from bull_machine.core.types import Signal


class EnhancedFusionEngineV1_4:
    """
    Enhanced 6-layer fusion engine with:
    - Quality gates (mask weak layers per-bar)
    - Alignment multipliers (boost confluence)
    - Veto/penalty matrix (hard stops + soft penalties)
    - Higher threshold with margin requirement
    """

    def __init__(self, config: Dict):
        self.config = config
        self.mode = config.get("mode", {})
        self.weights = config.get(
            "weights",
            {
                "wyckoff": 0.30,
                "liquidity": 0.25,
                "structure": 0.20,
                "momentum": 0.10,
                "volume": 0.10,
                "context": 0.05,
            },
        )
        self.quality_floors = config.get(
            "quality_floors",
            {
                "wyckoff": 0.55,
                "liquidity": 0.50,
                "structure": 0.50,
                "momentum": 0.45,
                "volume": 0.45,
                "context": 0.40,
            },
        )
        self.align_boosts = config.get(
            "align_boosts", {"triad": 1.15, "momentum": 1.10, "volume": 1.05}
        )
        self.penalties = config.get(
            "penalties", {"edge_no_reclaim": 0.03, "weak_break": 0.03, "frvp_chop": 0.02}
        )

        self.enter_threshold = self.mode.get("enter_threshold", 0.42)
        self.exit_threshold = self.mode.get("exit_threshold", 0.40)
        self.min_score_delta = self.mode.get("min_score_delta", 0.01)
        self.strict_mtf_relaxed = self.mode.get("strict_mtf_relaxed", False)

        # INSTRUMENTATION: Quality floor counters
        self.quality_stats = {
            "total_layers_checked": 0,
            "layers_kept": 0,
            "layers_masked": 0,
            "quality_floor_applications": 0,
        }

    def fuse_with_mtf(
        self, modules: Dict[str, Any], sync_report: Optional[Any] = None
    ) -> Optional[Signal]:
        """
        Enhanced fusion with quality gates, alignment boosts, and penalties.
        """

        # AGGRESSIVE DEBUG LOGGING
        logging.info(f"🔍 FUSION_ENTRY: modules={list(modules.keys())}")

        # Extract modules
        wyckoff = modules.get("wyckoff")
        liquidity = modules.get("liquidity")
        structure = modules.get("structure")
        momentum = modules.get("momentum")
        volume = modules.get("volume")
        context = modules.get("context")

        if not wyckoff or not liquidity:
            logging.warning("Missing core modules (wyckoff/liquidity)")
            return None

        # Build layer data with quality scores
        layers = self._build_layer_data(modules)

        # LOG RAW LAYER DATA
        for layer in layers:
            logging.info(
                f"🎯 RAW_LAYER: {layer['name']} quality={layer['quality']:.3f} score={layer['score']:.3f}"
            )

        # Hard vetoes (immediate rejection)
        veto_reason = self._check_hard_vetoes(sync_report, wyckoff, liquidity)
        if veto_reason:
            logging.info(f"HARD VETO: {veto_reason}")
            return None

        # Quality gates - mask weak layers for this bar
        logging.info(f"🚪 APPLYING_QUALITY_GATES: floors={self.quality_floors}")
        active_layers = self._apply_quality_gates(layers)
        logging.info(f"🚪 POST_GATES: {len(active_layers)}/{len(layers)} layers active")

        if len(active_layers) < 3:  # Need minimum confluence
            logging.info(f"Insufficient quality layers: {len(active_layers)}/6 active")
            return None

        # Single-layer safeguard: if only one triad layer active, RAISE instead of ALLOW
        triad_active = [
            l for l in active_layers if l["name"] in ["wyckoff", "liquidity", "structure"]
        ]
        if len(triad_active) == 1 and len(active_layers) < 4:
            logging.info(f"Single-opinion risk: only {triad_active[0]['name']} in triad active")
            # Could implement RAISE behavior here, for now we'll allow but note the risk

        # Check triad alignment (wyckoff + structure + liquidity)
        triad_aligned, consensus_side = self._check_triad_alignment(wyckoff, structure, liquidity)
        if not triad_aligned:
            logging.info("Triad not aligned - insufficient consensus")
            return None

        # Apply alignment boosts
        self._apply_alignment_boosts(active_layers, modules, triad_aligned)

        # Calculate penalties (capped to avoid excessive reduction)
        penalty = self._calculate_penalties(modules, wyckoff)
        penalty = min(penalty, 0.06)  # Cap total penalties at 6%

        # Compute masked weighted average
        fused_score = self._compute_weighted_score(active_layers)

        # Apply penalties
        fused_adj = max(0.0, fused_score - penalty)

        # Threshold with margin check (hysteresis for entry)
        min_required = self.enter_threshold + self.min_score_delta
        if fused_adj < min_required:
            logging.info(f"Below entry threshold: {fused_adj:.3f} < {min_required:.3f}")
            return None

        # Build signal
        confidence = fused_adj
        reasons = self._build_reasons(active_layers, triad_aligned, penalty)

        signal = Signal(
            ts=0,  # Set by caller
            side=consensus_side,
            confidence=confidence,
            reasons=reasons,
            ttl_bars=self._calculate_ttl(wyckoff, confidence),
        )

        logging.info(
            f"Enhanced signal: {consensus_side.upper()} @ {confidence:.3f} ({len(active_layers)}/6 layers)"
        )
        return signal

    def _build_layer_data(self, modules: Dict[str, Any]) -> List[Dict]:
        """Build layer data with scores and quality."""
        layers = []

        # Wyckoff
        wy = modules.get("wyckoff")
        if wy:
            layers.append(
                {
                    "name": "wyckoff",
                    "score": self._get_wyckoff_score(wy),
                    "quality": getattr(wy, "quality", getattr(wy, "confidence", 0.5)),
                    "side": getattr(wy, "bias", "neutral"),
                }
            )

        # Liquidity
        liq = modules.get("liquidity")
        if liq:
            layers.append(
                {
                    "name": "liquidity",
                    "score": getattr(liq, "score", 0.0),
                    "quality": getattr(liq, "quality", 0.5),
                    "side": self._liquidity_to_side(getattr(liq, "pressure", "neutral")),
                }
            )

        # Structure
        struct = modules.get("structure")
        if struct:
            layers.append(
                {
                    "name": "structure",
                    "score": struct.get("score", 0.0) if isinstance(struct, dict) else 0.0,
                    "quality": struct.get("quality", 0.5) if isinstance(struct, dict) else 0.5,
                    "side": struct.get("bias", "neutral")
                    if isinstance(struct, dict)
                    else "neutral",
                }
            )

        # Momentum
        mom = modules.get("momentum")
        if mom:
            layers.append(
                {
                    "name": "momentum",
                    "score": mom.get("score", 0.0) if isinstance(mom, dict) else 0.0,
                    "quality": mom.get("quality", 0.5) if isinstance(mom, dict) else 0.5,
                    "side": mom.get("direction", "neutral") if isinstance(mom, dict) else "neutral",
                }
            )

        # Volume
        vol = modules.get("volume")
        if vol:
            layers.append(
                {
                    "name": "volume",
                    "score": vol.get("score", 0.0) if isinstance(vol, dict) else 0.0,
                    "quality": vol.get("quality", 0.5) if isinstance(vol, dict) else 0.5,
                    "side": vol.get("bias", "neutral") if isinstance(vol, dict) else "neutral",
                }
            )

        # Context
        ctx = modules.get("context")
        if ctx:
            layers.append(
                {
                    "name": "context",
                    "score": ctx.get("score", 0.0) if isinstance(ctx, dict) else 0.0,
                    "quality": ctx.get("quality", 0.5) if isinstance(ctx, dict) else 0.5,
                    "side": ctx.get("bias", "neutral") if isinstance(ctx, dict) else "neutral",
                }
            )

        return layers

    def _check_hard_vetoes(self, sync_report, wyckoff, liquidity) -> Optional[str]:
        """Check for hard veto conditions with relaxed MTF logic."""

        # EQ magnet veto (but can be relaxed for strong displacement + volume)
        if sync_report and getattr(sync_report, "eq_magnet", False):
            # Check for breakout exception (placeholder logic)
            # In real implementation, this would check momentum displacement + volume confirmation
            return "EQ magnet active"

        # Severe MTF desync
        if sync_report and getattr(sync_report, "desync", False):
            return "Severe HTF↔LTF desync"

        # Relaxed MTF sync when strict_mtf_relaxed is enabled
        if self.strict_mtf_relaxed and sync_report:
            htf_bias = (
                getattr(sync_report.htf, "bias", "neutral")
                if hasattr(sync_report, "htf")
                else "neutral"
            )
            mtf_bias = (
                getattr(sync_report.mtf, "bias", "neutral")
                if hasattr(sync_report, "mtf")
                else "neutral"
            )
            ltf_bias = getattr(wyckoff, "bias", "neutral")

            # Allow when HTF == LTF and MTF ∈ {same, neutral}
            if htf_bias == ltf_bias and htf_bias != "neutral":
                if mtf_bias in [htf_bias, "neutral"]:
                    return None  # Allow with relaxed MTF

            # Allow mild conflicts if triad quality is high (will check in caller)
            return None

        # Original strict MTF logic
        if sync_report:
            htf_bias = (
                getattr(sync_report.htf, "bias", "neutral")
                if hasattr(sync_report, "htf")
                else "neutral"
            )
            ltf_bias = getattr(wyckoff, "bias", "neutral")

            if htf_bias != "neutral" and ltf_bias != "neutral" and htf_bias != ltf_bias:
                return f"LTF {ltf_bias} against HTF {htf_bias}"

        return None

    def _apply_quality_gates(self, layers: List[Dict]) -> List[Dict]:
        """Apply dynamic quality gates based on triad strength."""
        active = []

        # Calculate triad quality for dynamic floors
        triad_layers = [l for l in layers if l["name"] in ["wyckoff", "liquidity", "structure"]]
        triad_quality = (
            sum(l["quality"] for l in triad_layers) / len(triad_layers) if triad_layers else 0.5
        )

        # Log quality floors being used (only first time)
        if not hasattr(self, "_floors_logged"):
            logging.info(f"🎯 QUALITY_FLOORS: {self.quality_floors}")
            self._floors_logged = True

        kept_count = 0
        masked_count = 0

        # INSTRUMENTATION: Track this quality floor application
        self.quality_stats["quality_floor_applications"] += 1

        for layer in layers:
            name = layer["name"]
            quality = layer["quality"]

            # INSTRUMENTATION: Count each layer checked
            self.quality_stats["total_layers_checked"] += 1

            # Apply dynamic quality floors for supporting layers
            if name in ["momentum", "volume", "context"]:
                base_floor = self.quality_floors.get(name, 0.4)
                # Lower floor when triad is strong
                min_quality = max(0.35, base_floor - 0.10 * max(0.0, triad_quality - 0.60))
            else:
                # Fixed floors for triad layers
                min_quality = self.quality_floors.get(name, 0.4)

            logging.info(f"🚪 GATE_CHECK: {name} quality={quality:.3f} vs floor={min_quality:.3f}")

            if quality >= min_quality:
                active.append(layer)
                kept_count += 1
                self.quality_stats["layers_kept"] += 1  # INSTRUMENTATION
                logging.info(f"✅ KEPT: {name}")
            else:
                masked_count += 1
                self.quality_stats["layers_masked"] += 1  # INSTRUMENTATION
                logging.info(f"❌ MASKED: {name} (quality {quality:.3f} < floor {min_quality:.3f})")

        # Log summary of filtering
        if masked_count > 0:
            logging.info(
                f"Quality gates: kept {kept_count}/{len(layers)} layers, masked {masked_count}"
            )
        else:
            logging.debug(f"Quality gates: kept {kept_count}/{len(layers)} layers")

        # INSTRUMENTATION: Log stats every 100 applications
        if self.quality_stats["quality_floor_applications"] % 100 == 0:
            stats = self.quality_stats
            mask_rate = stats["layers_masked"] / max(1, stats["total_layers_checked"]) * 100
            logging.info(
                f"📊 QUALITY_STATS: {stats['quality_floor_applications']} applications, "
                f"{stats['layers_masked']}/{stats['total_layers_checked']} masked ({mask_rate:.1f}%)"
            )

        return active

    def _check_triad_alignment(self, wyckoff, structure, liquidity) -> tuple[bool, str]:
        """Check if wyckoff + structure + liquidity agree on direction."""

        # Get biases
        wy_bias = getattr(wyckoff, "bias", "neutral")
        struct_bias = structure.get("bias", "neutral") if isinstance(structure, dict) else "neutral"
        liq_pressure = getattr(liquidity, "pressure", "neutral")
        liq_bias = self._liquidity_to_side(liq_pressure)

        # Count votes
        votes = {"long": 0, "short": 0, "neutral": 0}
        for bias in [wy_bias, struct_bias, liq_bias]:
            if bias in votes:
                votes[bias] += 1

        # Require at least 2/3 agreement on non-neutral direction
        max_votes = max(votes["long"], votes["short"])
        if max_votes >= 2:
            consensus = "long" if votes["long"] >= 2 else "short"
            return True, consensus

        return False, "neutral"

    def _apply_alignment_boosts(
        self, active_layers: List[Dict], modules: Dict, triad_aligned: bool
    ):
        """Apply alignment multipliers to boost confluence."""

        if not triad_aligned:
            return

        # Boost triad members
        for layer in active_layers:
            if layer["name"] in ["wyckoff", "structure", "liquidity"]:
                layer["score"] *= self.align_boosts["triad"]

        # Check momentum alignment
        momentum = modules.get("momentum", {})
        if isinstance(momentum, dict):
            mom_direction = momentum.get("direction", "neutral")
            triad_side = self._get_triad_consensus_side(modules)

            if mom_direction == triad_side:
                for layer in active_layers:
                    if layer["name"] == "momentum":
                        layer["score"] *= self.align_boosts["momentum"]

        # Check volume confirmation
        volume = modules.get("volume", {})
        if isinstance(volume, dict) and volume.get("confirms", False):
            for layer in active_layers:
                if layer["name"] == "volume":
                    layer["score"] *= self.align_boosts["volume"]

    def _calculate_penalties(self, modules: Dict, wyckoff) -> float:
        """Calculate soft penalties."""
        penalty = 0.0

        # Edge without reclaim penalty
        if self._is_edge_no_reclaim(modules):
            penalty += self.penalties["edge_no_reclaim"]

        # Weak break penalty
        if self._is_weak_break(modules):
            penalty += self.penalties["weak_break"]

        # FRVP chop penalty
        if self._is_frvp_chop(modules):
            penalty += self.penalties["frvp_chop"]

        return penalty

    def _compute_weighted_score(self, active_layers: List[Dict]) -> float:
        """Compute masked weighted average and renormalize."""
        numerator = 0.0
        denominator = 0.0

        for layer in active_layers:
            name = layer["name"]
            score = layer["score"]
            weight = self.weights.get(name, 0.0)

            numerator += weight * score
            denominator += weight

        return numerator / max(denominator, 1e-9)

    # Helper methods
    def _get_wyckoff_score(self, wyckoff) -> float:
        if hasattr(wyckoff, "confidence"):
            return wyckoff.confidence
        phase_conf = getattr(wyckoff, "phase_confidence", 0.5)
        trend_conf = getattr(wyckoff, "trend_confidence", 0.5)
        return (phase_conf + trend_conf) / 2

    def _liquidity_to_side(self, pressure: str) -> str:
        if pressure == "bullish":
            return "long"
        elif pressure == "bearish":
            return "short"
        return "neutral"

    def _get_triad_consensus_side(self, modules: Dict) -> str:
        wyckoff = modules.get("wyckoff")
        structure = modules.get("structure", {})
        liquidity = modules.get("liquidity")

        wy_bias = getattr(wyckoff, "bias", "neutral")
        struct_bias = structure.get("bias", "neutral") if isinstance(structure, dict) else "neutral"
        liq_bias = self._liquidity_to_side(getattr(liquidity, "pressure", "neutral"))

        votes = [wy_bias, struct_bias, liq_bias]
        return (
            max(set(votes), key=votes.count)
            if votes.count(max(set(votes), key=votes.count)) >= 2
            else "neutral"
        )

    def _is_edge_no_reclaim(self, modules: Dict) -> bool:
        # Simplified check - could be enhanced
        structure = modules.get("structure", {})
        return (
            isinstance(structure, dict)
            and structure.get("edge_entry", False)
            and not structure.get("reclaimed", False)
        )

    def _is_weak_break(self, modules: Dict) -> bool:
        # Simplified check
        momentum = modules.get("momentum", {})
        return isinstance(momentum, dict) and momentum.get("break_strength", 1.0) < 0.5

    def _is_frvp_chop(self, modules: Dict) -> bool:
        # Simplified check
        volume = modules.get("volume", {})
        return isinstance(volume, dict) and volume.get("frvp_chop", False)

    def _build_reasons(
        self, active_layers: List[Dict], triad_aligned: bool, penalty: float
    ) -> List[str]:
        """Build signal reasons."""
        reasons = []

        if triad_aligned:
            reasons.append("Triad aligned")

        reasons.append(f"{len(active_layers)}/6 layers active")

        if penalty > 0:
            reasons.append(f"Penalty: -{penalty:.2f}")

        # Add top scoring layers
        sorted_layers = sorted(active_layers, key=lambda x: x["score"], reverse=True)
        for layer in sorted_layers[:2]:
            reasons.append(f"{layer['name']}: {layer['score']:.2f}")

        return reasons[:3]

    def _calculate_ttl(self, wyckoff, confidence: float) -> int:
        """Calculate signal TTL."""
        base_ttl = 20

        # Adjust for confidence
        if confidence > 0.6:
            base_ttl = int(base_ttl * 1.3)
        elif confidence < 0.5:
            base_ttl = int(base_ttl * 0.8)

        # Adjust for Wyckoff phase
        if hasattr(wyckoff, "phase"):
            if wyckoff.phase in ["C", "D"]:
                base_ttl = int(base_ttl * 1.2)
            elif wyckoff.phase in ["A", "B"]:
                base_ttl = int(base_ttl * 0.9)

        return max(10, min(40, base_ttl))
