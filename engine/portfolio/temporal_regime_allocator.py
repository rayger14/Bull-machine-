"""
TemporalRegimeAllocator - Regime allocator with temporal awareness

Extends RegimeWeightAllocator with time pressure and phase rhythm intelligence
to answer: "Given current regime + temporal state, how much capital does each
archetype deserve?"

Philosophy Integration:
- Moneytaur: "Time decays liquidity traps - fresh setups have edge, stale setups fade"
- Wyckoff: "Phase rhythm creates allocation windows - accumulation builds 34-55 bars"

Temporal Factors:
1. Temporal Confluence (0-1): Time pressure from multiple timeframe alignment
2. Fib Time Clustering: Fibonacci time alignment (13/21/34/55/89/144 bar cycles)
3. Phase Timing: Wyckoff event freshness (bars_since_spring/utad/sc)

Author: System Architect Agent
Date: 2026-01-12
Spec: TEMPORAL_REGIME_ALLOCATOR_SPEC.md
"""

import logging
from pathlib import Path
from typing import Dict, Optional, Tuple

import yaml

from engine.portfolio.regime_allocator import RegimeWeightAllocator


logger = logging.getLogger(__name__)


class TemporalRegimeAllocator(RegimeWeightAllocator):
    """
    Temporal-aware regime allocator that adjusts capital allocation based on:
    1. Base regime edge (from historical performance)
    2. Temporal confluence (time pressure from multiple timeframes)
    3. Phase timing (Wyckoff event freshness)

    Formula:
        final_weight = base_weight * temporal_boost * phase_boost

    Where:
        - base_weight: sqrt-weighted edge from historical Sharpe (parent class)
        - temporal_boost: 1.00-1.15x based on temporal_confluence
        - phase_boost: 0.80-1.20x based on Wyckoff event freshness
    """

    # Temporal confluence thresholds (time pressure)
    TEMPORAL_HIGH = 0.80    # High time pressure → 1.15x boost
    TEMPORAL_MED = 0.60     # Medium time pressure → 1.05x boost

    # Phase timing windows (bars since Wyckoff events)
    # Based on Moneytaur wisdom: fresh setups = edge, stale setups = fade
    PHASE_TIMING = {
        # Spring archetype: Fresh spring (13-34 bars) = boost
        'spring': {
            'event_key': 'bars_since_spring',
            'perfect_min': 13,      # Perfect timing start
            'perfect_max': 34,      # Perfect timing end
            'stale_threshold': 89,  # Temporal decay threshold
            'perfect_boost': 1.20,  # 20% boost in perfect window
            'stale_penalty': 0.85,  # 15% penalty when stale
        },
        # Liquidity vacuum: Recent SC (21-55 bars) = boost
        'liquidity_vacuum': {
            'event_key': 'bars_since_sc',
            'perfect_min': 21,
            'perfect_max': 55,
            'stale_threshold': 144,
            'perfect_boost': 1.15,
            'stale_penalty': 0.90,
        },
        # Wick trap: Fresh UTAD (13-34 bars) = boost
        'wick_trap': {
            'event_key': 'bars_since_utad',
            'perfect_min': 13,
            'perfect_max': 34,
            'stale_threshold': 89,
            'perfect_boost': 1.10,
            'stale_penalty': 0.80,
        },
        # Wick trap moneytaur variant
        'wick_trap_moneytaur': {
            'event_key': 'bars_since_utad',
            'perfect_min': 13,
            'perfect_max': 34,
            'stale_threshold': 89,
            'perfect_boost': 1.10,
            'stale_penalty': 0.80,
        },
        # Order block retest: Fresh retest opportunity
        'order_block_retest': {
            'event_key': 'bars_since_lps',  # Last Point of Support
            'perfect_min': 8,
            'perfect_max': 21,
            'stale_threshold': 55,
            'perfect_boost': 1.08,
            'stale_penalty': 0.90,
        },
        # Trap within trend: Needs fresh setup
        'trap_within_trend': {
            'event_key': 'bars_since_spring',
            'perfect_min': 13,
            'perfect_max': 34,
            'stale_threshold': 89,
            'perfect_boost': 1.12,
            'stale_penalty': 0.85,
        },
        # Funding divergence: Fresh funding extreme
        'funding_divergence': {
            'event_key': 'bars_since_funding_extreme',
            'perfect_min': 8,
            'perfect_max': 34,
            'stale_threshold': 89,
            'perfect_boost': 1.10,
            'stale_penalty': 0.85,
        },
        # Long squeeze: Immediate opportunity window
        'long_squeeze': {
            'event_key': 'bars_since_funding_extreme',
            'perfect_min': 5,
            'perfect_max': 21,
            'stale_threshold': 55,
            'perfect_boost': 1.15,
            'stale_penalty': 0.75,  # Severe decay
        },
    }

    def __init__(
        self,
        edge_table_path: str,
        config_path: Optional[str] = None,
        config_override: Optional[Dict] = None,
        enable_temporal: bool = True,
        gates_path: Optional[str] = None
    ):
        """
        Initialize temporal-aware allocator.

        Args:
            edge_table_path: Path to archetype_regime_edge_table.csv
            config_path: Path to regime_allocator_config.json (optional)
            config_override: Direct config dict (overrides config_path)
            enable_temporal: If False, acts as vanilla RegimeWeightAllocator
            gates_path: Path to archetype_regime_gates.yaml (optional)
        """
        super().__init__(edge_table_path, config_path, config_override)

        self.enable_temporal = enable_temporal

        # Load archetype-regime gates
        self.gates = self._load_gates(gates_path)

        if enable_temporal:
            logger.info(
                "[TemporalAllocator] Temporal awareness ENABLED - "
                "will apply temporal confluence and phase timing boosts"
            )
        else:
            logger.info(
                "[TemporalAllocator] Temporal awareness DISABLED - "
                "acting as vanilla RegimeWeightAllocator"
            )

        if self.gates:
            logger.info(
                "[TemporalAllocator] Archetype-regime gates LOADED - "
                f"{len(self.gates.get('gates', {}))} archetypes configured"
            )

    def _load_gates(self, gates_path: Optional[str]) -> Dict:
        """Load archetype-regime gates from YAML."""
        if gates_path is None:
            # Try default path
            gates_path = Path(__file__).parent.parent.parent / 'configs' / 'archetype_regime_gates.yaml'

        gates_path = Path(gates_path)

        if not gates_path.exists():
            logger.warning(f"[TemporalAllocator] Gates file not found: {gates_path}")
            return {}

        try:
            with open(gates_path, 'r') as f:
                gates_config = yaml.safe_load(f)
            logger.info(f"[TemporalAllocator] Loaded gates from {gates_path}")
            return gates_config
        except Exception as e:
            logger.error(f"[TemporalAllocator] Failed to load gates: {e}")
            return {}

    def _check_archetype_regime_gate(
        self,
        archetype: str,
        regime: str
    ) -> Tuple[bool, str, float]:
        """
        Check archetype-regime gate.

        Returns:
            Tuple of (enabled, reason, max_allocation)
        """
        if not self.gates or 'gates' not in self.gates:
            return True, "No gates configured", 1.0

        gates = self.gates['gates']
        global_config = self.gates.get('global', {})

        # Check if archetype has gates defined
        if archetype not in gates:
            # Use default
            default_enabled = global_config.get('default_enabled', False)
            default_max = global_config.get('default_max_allocation', 0.10)
            return default_enabled, "Using default gate (archetype not configured)", default_max

        archetype_gates = gates[archetype]

        # Check if regime has gate defined for this archetype
        if regime not in archetype_gates:
            # Use default
            default_enabled = global_config.get('default_enabled', False)
            default_max = global_config.get('default_max_allocation', 0.10)
            return default_enabled, "Using default gate (regime not configured)", default_max

        regime_gate = archetype_gates[regime]
        enabled = regime_gate.get('enabled', False)
        max_alloc = regime_gate.get('max_allocation', 0.0)
        reason = regime_gate.get('reason', 'No reason provided')

        return enabled, reason, max_alloc

    def get_weight_with_temporal(
        self,
        archetype: str,
        regime: str,
        temporal_state: Dict
    ) -> Tuple[float, Dict]:
        """
        Get allocation weight with temporal awareness.

        This is the main entry point for temporal-aware allocation.

        Args:
            archetype: Archetype name (e.g., 'liquidity_vacuum')
            regime: Regime label (crisis/risk_off/neutral/risk_on)
            temporal_state: Dictionary with temporal features:
                - temporal_confluence: float ∈ [0, 1] (time pressure)
                - fib_time_cluster: bool (Fibonacci alignment)
                - bars_since_spring: int (freshness of spring event)
                - bars_since_utad: int (freshness of UTAD)
                - bars_since_sc: int (freshness of Selling Climax)
                - bars_since_lps: int (freshness of Last Point of Support)
                - bars_since_funding_extreme: int (freshness of funding extreme)

        Returns:
            Tuple of (weight, metadata_dict)
            - weight: Final allocation weight ∈ [min_weight, 1.0]
            - metadata: Dict with decomposition of weight calculation
        """
        # CRITICAL: Check archetype-regime gate FIRST
        gate_enabled, gate_reason, gate_max_alloc = self._check_archetype_regime_gate(archetype, regime)

        if not gate_enabled:
            logger.debug(
                f"[TemporalAllocator] GATE REJECTION: {archetype} in {regime} - {gate_reason}"
            )
            return 0.0, {
                'base_weight': 0.0,
                'temporal_boost': 0.0,
                'phase_boost': 0.0,
                'final_weight': 0.0,
                'gate_enabled': False,
                'gate_reason': gate_reason,
                'temporal_enabled': self.enable_temporal
            }

        # Get base weight from parent class (sqrt-weighted edge)
        base_weight = self.get_sqrt_weight(archetype, regime)

        # If temporal disabled, return base weight
        if not self.enable_temporal:
            return base_weight, {
                'base_weight': base_weight,
                'temporal_boost': 1.0,
                'phase_boost': 1.0,
                'final_weight': base_weight,
                'temporal_enabled': False
            }

        # Factor 1: Temporal Boost (time pressure from confluence)
        temporal_boost = self._get_temporal_boost(temporal_state)

        # Factor 2: Phase Boost (Wyckoff event freshness)
        phase_boost = self._get_phase_boost(archetype, temporal_state)

        # Combine factors
        weight = base_weight * temporal_boost * phase_boost

        # Get edge for guardrail checks
        edge_metrics = self.get_edge_metrics(archetype, regime)
        edge = edge_metrics.get('edge_shrunk', 0.0) if edge_metrics['has_data'] else 0.0

        # Apply regime-specific guardrails
        weight = self._apply_guardrails(weight, archetype, regime, edge)

        # Apply gate max allocation cap
        if gate_max_alloc < weight:
            logger.debug(
                f"[TemporalAllocator] Gate cap applied: {archetype} in {regime} "
                f"{weight:.3f} → {gate_max_alloc:.3f}"
            )
            weight = gate_max_alloc

        # Build metadata for transparency
        metadata = {
            'base_weight': base_weight,
            'temporal_boost': temporal_boost,
            'phase_boost': phase_boost,
            'combined_weight': base_weight * temporal_boost * phase_boost,
            'final_weight': weight,
            'was_capped': weight != (base_weight * temporal_boost * phase_boost),
            'edge_shrunk': edge,
            'temporal_enabled': True,
            'gate_enabled': gate_enabled,
            'gate_max_alloc': gate_max_alloc,
            'temporal_confluence': temporal_state.get('temporal_confluence', 0.5),
            'fib_time_cluster': temporal_state.get('fib_time_cluster', False),
        }

        # Log allocation decision (debug level)
        logger.debug(
            f"[TemporalAllocator] {archetype} in {regime}: "
            f"base={base_weight:.3f}, temporal={temporal_boost:.2f}x, "
            f"phase={phase_boost:.2f}x → final={weight:.3f}"
        )

        return weight, metadata

    def _get_temporal_boost(self, temporal_state: Dict) -> float:
        """
        Calculate temporal boost from time pressure.

        High confluence (>0.80) means multiple timeframes agree → boost allocation
        Medium confluence (>0.60) means moderate alignment → small boost
        Low confluence means timeframes disagree → neutral (1.0x)

        Args:
            temporal_state: Dict with 'temporal_confluence' key

        Returns:
            Temporal boost factor ∈ [1.00, 1.15]
        """
        confluence = temporal_state.get('temporal_confluence', 0.5)
        fib_cluster = temporal_state.get('fib_time_cluster', False)

        # Base boost from confluence
        if confluence > self.TEMPORAL_HIGH:
            boost = 1.15  # Strong time pressure
        elif confluence > self.TEMPORAL_MED:
            boost = 1.05  # Moderate time pressure
        else:
            boost = 1.0   # No time pressure

        # Additional 2% boost if Fibonacci time cluster present
        if fib_cluster and confluence > self.TEMPORAL_MED:
            boost *= 1.02
            logger.debug(
                f"[TemporalAllocator] Fib time cluster detected with "
                f"confluence={confluence:.2f} → +2% boost"
            )

        return boost

    def _get_phase_boost(self, archetype: str, temporal_state: Dict) -> float:
        """
        Calculate phase timing boost from Wyckoff event freshness.

        Fresh setups (13-34 bars after event) = edge boost
        Stale setups (>89 bars) = temporal decay penalty

        Implements Moneytaur wisdom: "Time decays liquidity traps"

        Args:
            archetype: Archetype name
            temporal_state: Dict with 'bars_since_*' keys

        Returns:
            Phase boost factor ∈ [0.75, 1.20]
        """
        # Check if archetype has phase timing defined
        if archetype not in self.PHASE_TIMING:
            return 1.0  # Neutral for archetypes without timing rules

        timing = self.PHASE_TIMING[archetype]
        event_key = timing['event_key']

        # Get bars since event (default to 999 = very stale)
        bars_since = temporal_state.get(event_key, 999)

        # Perfect timing window (e.g., 13-34 bars for spring)
        if timing['perfect_min'] <= bars_since <= timing['perfect_max']:
            boost = timing['perfect_boost']
            logger.debug(
                f"[TemporalAllocator] {archetype}: {event_key}={bars_since} bars "
                f"in PERFECT window [{timing['perfect_min']}-{timing['perfect_max']}] "
                f"→ {boost:.2f}x boost"
            )
            return boost

        # Stale setup (>89/144 bars)
        if bars_since > timing['stale_threshold']:
            penalty = timing['stale_penalty']
            logger.debug(
                f"[TemporalAllocator] {archetype}: {event_key}={bars_since} bars "
                f"STALE (>{timing['stale_threshold']}) → {penalty:.2f}x penalty"
            )
            return penalty

        # Between perfect and stale → neutral
        return 1.0

    def _apply_guardrails(
        self,
        weight: float,
        archetype: str,
        regime: str,
        edge: float
    ) -> float:
        """
        Apply regime-specific guardrails to prevent runaway allocation.

        Guardrails:
        1. Crisis regime with negative edge → cap at 20% (per soft gating spec)
        2. Always maintain minimum weight floor (for exploration)

        Args:
            weight: Combined weight before guardrails
            archetype: Archetype name
            regime: Regime label
            edge: Edge metric (Sharpe-like)

        Returns:
            Capped/floored weight
        """
        # Guardrail 1: Crisis cap for negative edge
        if regime == 'crisis' and edge < 0:
            weight = min(weight, self.neg_edge_cap)
            logger.debug(
                f"[TemporalAllocator] Crisis cap applied: {archetype} "
                f"edge={edge:.3f} → capped at {self.neg_edge_cap:.1%}"
            )

        # Guardrail 2: Minimum weight floor
        weight = max(weight, self.min_weight)

        return weight

    def get_allocation_summary_with_temporal(
        self,
        regime: str,
        temporal_state: Dict,
        show_effective: bool = True
    ) -> str:
        """
        Get formatted summary with temporal factors included.

        Args:
            regime: Regime label
            temporal_state: Temporal state dict
            show_effective: Show effective allocation with regime budget

        Returns:
            Formatted summary string
        """
        lines = [f"\nTemporal-Aware Allocation - {regime.upper()} Regime"]
        lines.append("=" * 60)

        # Show temporal state
        confluence = temporal_state.get('temporal_confluence', 0.5)
        fib_cluster = temporal_state.get('fib_time_cluster', False)

        lines.append("Temporal State:")
        lines.append(f"  temporal_confluence: {confluence:.2f} "
                    f"({'HIGH' if confluence > 0.8 else 'MED' if confluence > 0.6 else 'LOW'} pressure)")
        lines.append(f"  fib_time_cluster: {fib_cluster}")

        # Show relevant bars_since values
        for key in ['bars_since_spring', 'bars_since_utad', 'bars_since_sc',
                    'bars_since_lps', 'bars_since_funding_extreme']:
            if key in temporal_state:
                lines.append(f"  {key}: {temporal_state[key]} bars")

        lines.append("")

        # Get archetype weights with temporal factors
        weights = self.get_regime_distribution(regime)

        lines.append("Archetype Weights (with temporal factors):")
        lines.append("-" * 60)

        # Show each archetype with decomposition
        for archetype in sorted(weights.keys()):
            weight, metadata = self.get_weight_with_temporal(
                archetype, regime, temporal_state
            )

            lines.append(f"\nArchetype: {archetype}")
            lines.append(f"  Base weight (edge): {metadata['base_weight']:.3f}")
            lines.append(f"  Temporal boost: {metadata['temporal_boost']:.2f}x")
            lines.append(f"  Phase boost: {metadata['phase_boost']:.2f}x")
            lines.append(f"  Final weight: {metadata['final_weight']:.3f}")

            if metadata['was_capped']:
                lines.append("  → Capped by guardrails")

        # Show cash bucket
        cash_bucket = self.get_cash_bucket_weight(regime)
        if cash_bucket > 0:
            lines.append(f"\nCash Bucket: {cash_bucket:.1%}")

        # Show effective allocation with regime budget
        if show_effective:
            regime_budget = self.get_regime_budget(regime)
            lines.append(f"\nEffective Portfolio Allocation ({regime_budget:.0%} regime budget):")
            lines.append("-" * 60)

            for archetype in sorted(weights.keys()):
                weight, _ = self.get_weight_with_temporal(
                    archetype, regime, temporal_state
                )
                effective_pct = weight * regime_budget
                lines.append(f"  {archetype}: {effective_pct:.1%}")

            if cash_bucket > 0:
                cash_effective = cash_bucket * regime_budget
                lines.append(f"  CASH (regime): {cash_effective:.1%}")

            remaining = 1.0 - regime_budget
            if remaining > 0:
                lines.append(f"  CASH (reserve): {remaining:.1%}")

        return '\n'.join(lines)
