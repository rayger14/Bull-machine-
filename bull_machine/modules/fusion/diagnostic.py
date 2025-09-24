"""Diagnostic Fusion Engine - Minimal Gates for Testing"""

import logging
from typing import Any, Dict, Optional

from bull_machine.core.types import Signal


class DiagnosticFusionEngine:
    """
    Minimal fusion engine for diagnostics - removes most gates.
    """

    def __init__(self, config: Dict):
        self.config = config
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
        self.enter_threshold = config.get("mode", {}).get("enter_threshold", 0.20)

    def fuse_with_mtf(self, modules: Dict[str, Any], sync_report: Optional[Any] = None) -> Optional[Signal]:
        """
        Minimal fusion - almost no gates for diagnostic purposes.
        """

        # Extract modules
        wyckoff = modules.get("wyckoff")
        liquidity = modules.get("liquidity")

        if not wyckoff or not liquidity:
            logging.warning("Missing core modules (wyckoff/liquidity)")
            return None

        # Build layer data
        layers = []

        # Wyckoff
        if wyckoff:
            score = self._get_wyckoff_score(wyckoff)
            layers.append({"name": "wyckoff", "score": score, "side": getattr(wyckoff, "bias", "neutral")})

        # Liquidity
        if liquidity:
            layers.append(
                {
                    "name": "liquidity",
                    "score": getattr(liquidity, "score", 0.0),
                    "side": self._liquidity_to_side(getattr(liquidity, "pressure", "neutral")),
                }
            )

        # Structure
        structure = modules.get("structure")
        if structure:
            layers.append(
                {
                    "name": "structure",
                    "score": structure.get("score", 0.0) if isinstance(structure, dict) else 0.0,
                    "side": structure.get("bias", "neutral") if isinstance(structure, dict) else "neutral",
                }
            )

        # Others
        for name in ["momentum", "volume", "context"]:
            module = modules.get(name)
            if module:
                layers.append(
                    {
                        "name": name,
                        "score": module.get("score", 0.0) if isinstance(module, dict) else 0.0,
                        "side": module.get("bias", "neutral") if isinstance(module, dict) else "neutral",
                    }
                )

        if len(layers) < 2:
            logging.info(f"Insufficient layers: {len(layers)}")
            return None

        # Apply layer caps to normalize flat layers
        layer_caps = self.config.get("fusion", {}).get("layer_caps", {})
        for layer in layers:
            cap = layer_caps.get(layer["name"])
            if cap and layer["score"] > cap:
                logging.info(f"LAYER_CAP: {layer['name']} capped from {layer['score']:.3f} to {cap:.3f}")
                layer["score"] = cap

        # Apply variance guards (would need historical data - simplified for now)
        variance_guards = self.config.get("fusion", {}).get("min_variance_guard", {})
        for layer in layers:
            guard_threshold = variance_guards.get(layer["name"])
            if guard_threshold:
                # Simple heuristic: if score is too close to common values, down-weight
                common_values = [0.1, 0.2, 0.3]
                if any(abs(layer["score"] - cv) < 0.05 for cv in common_values):
                    original_weight = self.weights.get(layer["name"], 0.0)
                    self.weights[layer["name"]] = original_weight * 0.5
                    logging.info(f"VARIANCE_GUARD: {layer['name']} down-weighted due to low variance")

        # Weighted fusion with enhanced diagnostics
        total_score = 0.0
        total_weight = 0.0

        # Log detailed per-layer scores
        layer_details = {}
        for layer in layers:
            weight = self.weights.get(layer["name"], 0.0)
            weighted_score = layer["score"] * weight
            total_score += weighted_score
            total_weight += weight
            layer_details[layer["name"]] = {
                "raw_score": layer["score"],
                "weight": weight,
                "weighted": weighted_score,
                "side": layer["side"],
            }

        if total_weight == 0:
            logging.info("Zero total weight")
            return None

        fused_score = total_score / total_weight

        # Detailed fusion diagnostics
        logging.info(f"FUSION_DIAG: layers={len(layers)} fused={fused_score:.3f} threshold={self.enter_threshold:.3f}")
        for name, details in layer_details.items():
            logging.info(
                f"  {name}: raw={details['raw_score']:.3f} weight={details['weight']:.2f} weighted={details['weighted']:.3f} side={details['side']}"
            )

        # Quality floors check
        quality_floors = self.config.get("fusion", {}).get("quality_floors", {})
        if quality_floors:
            floors_passed = 0
            floors_failed = []
            for name, details in layer_details.items():
                floor = quality_floors.get(name, 0.0)
                if details["raw_score"] >= floor:
                    floors_passed += 1
                else:
                    floors_failed.append(f"{name}({details['raw_score']:.3f}<{floor:.3f})")

            logging.info(f"QUALITY_FLOORS: {floors_passed}/{len(layer_details)} passed, failed: {floors_failed}")

            # Enhanced triad rule with override for exceptional confluence
            triad_config = self.config.get("fusion", {}).get("triad", {})
            if triad_config.get("require", False):
                triad_members = triad_config.get("members", ["wyckoff", "structure", "liquidity"])
                min_pass = triad_config.get("min_pass", 2)

                triad_passed = 0
                triad_scores = []
                for member in triad_members:
                    if member in layer_details:
                        floor = quality_floors.get(member, 0.0)
                        score = layer_details[member]["raw_score"]
                        triad_scores.append(score)
                        if score >= floor:
                            triad_passed += 1

                logging.info(f"TRIAD_RULE: {triad_passed}/{len(triad_members)} core layers passed (need {min_pass})")

                # Check for triad override (exceptional confluence)
                override_config = triad_config.get("override", {})
                if override_config.get("enabled", False) and len(triad_scores) >= 2:
                    # Sort scores and check if top 2 exceed threshold
                    sorted_scores = sorted(triad_scores, reverse=True)
                    top_two_sum = sorted_scores[0] + sorted_scores[1]
                    override_threshold = override_config.get("if_sum_of_two_cores_ge", 1.10)

                    if top_two_sum >= override_threshold:
                        logging.info(
                            f"TRIAD_OVERRIDE: Exceptional confluence {top_two_sum:.3f} >= {override_threshold:.3f} - allowing signal"
                        )
                    elif triad_passed < min_pass:
                        logging.info("Triad rule veto - insufficient core layers")
                        return None
                elif triad_passed < min_pass:
                    logging.info("Triad rule veto - insufficient core layers")
                    return None

        # Hysteresis threshold check
        hysteresis_config = self.config.get("fusion", {}).get("hysteresis", {})
        if hysteresis_config:
            enter_threshold = hysteresis_config.get("enter", self.enter_threshold)
            hold_threshold = hysteresis_config.get("hold", self.enter_threshold * 0.9)

            # For now, use enter threshold (would need state tracking for hold)
            threshold_to_use = enter_threshold
            logging.info(f"HYSTERESIS: Using enter threshold {threshold_to_use:.3f} (hold: {hold_threshold:.3f})")
        else:
            threshold_to_use = self.enter_threshold

        if fused_score < threshold_to_use:
            logging.info(f"Below threshold: {fused_score:.3f} < {threshold_to_use:.3f}")
            return None

        # Determine side (majority vote)
        sides = [l["side"] for l in layers if l["side"] != "neutral"]
        if not sides:
            logging.info("No clear side")
            return None

        # Count votes
        long_votes = sides.count("long")
        short_votes = sides.count("short")

        if long_votes > short_votes:
            consensus_side = "long"
        elif short_votes > long_votes:
            consensus_side = "short"
        else:
            logging.info("Side tie")
            return None

        # Build signal
        confidence = fused_score
        reasons = [f"{len(layers)} layers", f"score: {fused_score:.3f}"]

        signal = Signal(ts=0, side=consensus_side, confidence=confidence, reasons=reasons, ttl_bars=20)

        logging.info(f"Diagnostic signal: {consensus_side.upper()} @ {confidence:.3f}")
        return signal

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
