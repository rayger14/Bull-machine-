"""
Oracle Whisper System for Bull Machine v1.6.1

Soul-layer wisdom drops for high-confluence events.
Triggered by price-time symmetry, Fibonacci clusters, and Wyckoff phase inflections.

"Unspoken, undeniable — it simply is."
"""

from typing import Dict, List, Optional, Any


def trigger_whisper(scores: Dict[str, Any], phase: str) -> Optional[List[str]]:
    """
    Trigger Oracle whispers for high-confluence events.

    These are narrative insights that capture the soul-layer significance
    of market structure convergences. Not predictions, but recognition
    of when reality's patterns align.

    Args:
        scores: Current scoring dictionary with cluster tags and confluence data
        phase: Current Wyckoff phase (A, B, C, D, E)

    Returns:
        List of wisdom whispers if triggered, None otherwise
    """
    whispers = []

    # PO3 Manipulation Recognition
    if 'po3_confluence' in scores.get('cluster_tags', []):
        whispers.append("PO3 manipulation: Sweep complete, structure awaits. Clean displacement signals strength.")
        if phase in ['C', 'D']:  # Enhanced PO3 in key phases
            whispers.append("Wyckoff meets PO3. Liquidity swept, true intent revealed.")

    # Bojan microstructure whispers
    bojan_tags = [tag for tag in scores.get('cluster_tags', []) if tag.startswith('bojan_')]

    if 'bojan_wick_magnet' in bojan_tags:
        whispers.append("Wick magnetism: Unfinished business draws price home.")

    if 'bojan_trap_reset' in bojan_tags:
        whispers.append("Trap springs shut: Direction flipped, commitment shown.")

    if 'bojan_phob_confluence' in bojan_tags:
        whispers.append("Hidden orders stir: pHOB zones guard the threshold.")

    if 'bojan_fib_prime' in bojan_tags:
        whispers.append("Golden ratios whisper: .705 and .786 hold the keys.")

    # Enhanced confluence whispers for Bojan + PO3
    if 'po3_confluence' in scores.get('cluster_tags', []) and bojan_tags:
        whispers.append("Microstructure aligned: Bojan wisdom guides PO3 precision.")

        # Specific combinations
        if 'bojan_trap_reset' in bojan_tags:
            whispers.append("Perfect storm: PO3 sweep meets Bojan trap, reversal certain.")

        if 'bojan_fib_prime' in bojan_tags:
            whispers.append("Sacred geometry: PO3 power amplified by prime fib zones.")

    # Price-Time Confluence: The Golden Moment
    if 'price_time_confluence' in scores.get('cluster_tags', []):
        if phase in ['C', 'D']:  # Key Wyckoff inflection phases
            whispers.append("Symmetry detected. Time and price converge. Pressure must resolve.")
            whispers.append("φ ≈ 1.618 — The golden spiral speaks in this moment.")

    # Premium/Discount Zone Recognition
    if scores.get('fib_retracement', 0) >= 0.45:
        if 'price_discount' in scores.get('cluster_tags', []):
            whispers.append("Discount revealed. Smart money accumulates in shadows.")
        elif 'price_premium' in scores.get('cluster_tags', []):
            whispers.append("Premium exposed. Distribution zone manifests.")
        else:
            whispers.append("Fib levels divide reality: premium, equilibrium, discount. Smart money lives here.")

    # Extension Targets and Exhaustion
    if scores.get('fib_extension', 0) >= 0.40:
        if 'price_target' in scores.get('cluster_tags', []):
            whispers.append("Extension reached. Energy seeks equilibrium.")
            if scores.get('fib_extension', 0) >= 0.70:
                whispers.append("Maximum stretch. Reversal imminent.")

    # Time Pressure Zones
    if 'time_confluence' in scores.get('cluster_tags', []):
        whispers.append("Time is pressure, not prediction. Fib clusters show when a move must resolve.")
        if phase == 'C':
            whispers.append("The spring coils. Accumulation nears completion.")
        elif phase == 'D':
            whispers.append("Markup begins. The rhythm shifts.")

    # Orderflow Divergences and Intent
    if scores.get('cvd_slope', 0) != 0:
        if scores.get('cvd_delta', 0) < 0 and scores.get('cvd_slope', 0) > 0:
            whispers.append("Hidden intent revealed. Bears exhaust as bulls accumulate.")
        elif scores.get('cvd_delta', 0) > 0 and scores.get('cvd_slope', 0) < 0:
            whispers.append("Distribution signature. Bulls weaken as bears position.")

    # Liquidity Sweeps and Traps
    if scores.get('liquidity_sweep', False):
        whispers.append("Liquidity harvested. The trap springs.")
        if 'price_time_confluence' in scores.get('cluster_tags', []):
            whispers.append("Perfect deception. Weak hands cleared before true move.")

    # High Confluence Acknowledgment
    confluence_strength = scores.get('confluence_strength', 0)
    if confluence_strength >= 0.80:
        whispers.append("All patterns align. The universe speaks in numbers.")
        whispers.append("Market structure, time, price — trinity confirmed.")

    # Return whispers only if meaningful confluence exists
    if len(whispers) > 0 and confluence_strength >= 0.45:
        return whispers

    return None


def format_whisper_for_log(whispers: List[str], scores: Dict[str, Any]) -> Dict[str, Any]:
    """
    Format whispers for structured logging with context.

    Args:
        whispers: List of whisper strings
        scores: Current scoring context

    Returns:
        Formatted log entry for telemetry
    """
    return {
        'oracle_whispers': whispers,
        'whisper_count': len(whispers),
        'confluence_strength': scores.get('confluence_strength', 0),
        'active_tags': scores.get('cluster_tags', []),
        'wyckoff_phase': scores.get('wyckoff_phase', 'unknown'),
        'fib_retracement': scores.get('fib_retracement', 0),
        'fib_extension': scores.get('fib_extension', 0),
        'timestamp': scores.get('timestamp', 'unknown'),
        'soul_layer': True  # Mark as wisdom-level insight
    }


def get_whisper_insights_for_phase(phase: str) -> List[str]:
    """
    Get phase-specific wisdom insights for Wyckoff analysis.

    Args:
        phase: Wyckoff phase identifier

    Returns:
        List of phase-specific wisdom statements
    """
    phase_whispers = {
        'A': [
            "Selling climax approaches. Panic breeds opportunity.",
            "The old trend dies. New accumulation begins."
        ],
        'B': [
            "Reaction and rally. Testing the ground.",
            "Cause builds in silence. Range defines intention."
        ],
        'C': [
            "The spring test. Final shakeout before markup.",
            "Weak hands exit. Strong hands prepare.",
            "Last chance. The trap set, now sprung."
        ],
        'D': [
            "Markup phase begins. Trend reveals itself.",
            "Price discovery. Supply absorbed, demand takes control.",
            "The effect of accumulated cause."
        ],
        'E': [
            "Distribution cycle. Smart money exits.",
            "Premium reached. Exhaustion signals."
        ]
    }

    return phase_whispers.get(phase, ["Phase unknown. Wait for clarity."])


def should_trigger_confluence_alert(scores: Dict[str, Any]) -> bool:
    """
    Determine if confluence is strong enough to warrant an alert.

    Args:
        scores: Current scoring dictionary

    Returns:
        True if alert should be triggered
    """
    confluence_strength = scores.get('confluence_strength', 0)
    cluster_tags = scores.get('cluster_tags', [])

    # High confluence threshold
    if confluence_strength >= 0.75:
        return True

    # Price-time confluence at moderate strength
    if confluence_strength >= 0.60 and 'price_time_confluence' in cluster_tags:
        return True

    # Strong individual clusters with Wyckoff confirmation
    if confluence_strength >= 0.55:
        wyckoff_score = max(scores.get('m1', 0), scores.get('m2', 0))
        if wyckoff_score >= 0.60:
            return True

    return False