"""
Feature Flags - PR#6A Archetype Refactor + Bull/Bear Split

Controls progressive migration from legacy letter-code system
to canonical slug-based archetype registry.

CRITICAL ARCHITECTURE FIX:
Global feature flags (EVALUATE_ALL_ARCHETYPES, SOFT_LIQUIDITY_FILTER) broke
the gold standard when enabled for bear archetypes. This module now provides
SPLIT flags to allow independent behavior for bull vs bear archetypes.

Bull Archetypes (A-M): Require legacy priority dispatch + hard liquidity filter
Bear Archetypes (S1-S8): Require evaluate-all dispatch + soft liquidity filter

See: GOLD_STANDARD_DISCREPANCY_INVESTIGATION.md for background
"""

# PHASE 0: Feature flags for safe migration
USE_CANONICAL_ARCHETYPE_REGISTRY = True  # Enable new registry.py lookup
ROUTER_SOFT_FILTERS = True  # Turn hard vetoes into soft weights
PARAM_ECHO_ENABLED = True  # Log actual params used per run
GATE_TRACING_ENABLED = True  # Track per-gate pass rates

# PHASE 1: Warnings for legacy usage
WARN_ON_LETTER_ALIASES = True  # Log when letter codes are used
STRICT_SLUG_VALIDATION = False  # Raise on unknown keys (vs. fallback)

# =============================================================================
# PHASE 3: Split Dispatch Behavior (Bull vs Bear)
# =============================================================================

# Bull Archetypes (A, B, C, D, E, F, G, H, K, L, M) - Preserve gold standard
# Gold Standard: 17 trades, PF 6.17 (BTC 2024-01-01 to 2024-09-30)
BULL_EVALUATE_ALL = False        # Legacy priority dispatch (A→H→B→K→L→C→...)
BULL_SOFT_LIQUIDITY = False      # Hard filter at min_liquidity threshold (0.30)

# Bear Archetypes (S1, S2, S3, S4, S5, S6, S7, S8) - Enable flexibility
# Bear patterns need different behavior due to inverted liquidity logic
BEAR_EVALUATE_ALL = True         # Score all, pick best (prevents archetype starvation)
BEAR_SOFT_LIQUIDITY = True       # Soft penalty (0.7x) instead of hard reject

# Backward compatibility (DEPRECATED - will be removed after full migration)
# These default to bull behavior to preserve existing code paths
EVALUATE_ALL_ARCHETYPES = BULL_EVALUATE_ALL  # Default: False (legacy priority)
SOFT_LIQUIDITY_FILTER = BULL_SOFT_LIQUIDITY  # Default: False (hard filter)

LEGACY_PRIORITY_ORDER = True  # Use A-M priority or score-based selection - ENABLED for baseline

# =============================================================================
# PHASE 4: Split Filter Softening (Bull vs Bear)
# =============================================================================

# Bull Archetypes - Hard filters (gold standard behavior)
BULL_SOFT_REGIME = False         # Hard veto on crisis/risk_off regimes
BULL_SOFT_SESSION = False        # Hard veto on low-volume sessions

# Bear Archetypes - Soft filters (allow marginal signals to compete)
BEAR_SOFT_REGIME = False         # 20% penalty during macro stress (disabled for now)
BEAR_SOFT_SESSION = False        # 15% penalty during Asian session (disabled for now)

# Backward compatibility (DEPRECATED)
SOFT_REGIME_FILTER = BULL_SOFT_REGIME    # Default: False (hard filter)
SOFT_SESSION_FILTER = BULL_SOFT_SESSION  # Default: False (hard filter)

# Observability output paths
ARTIFACTS_BASE = "artifacts"  # Base path for run artifacts
PARAM_ECHO_FILE = "params_used.json"  # Per-run param snapshot
GATE_STATS_DIR = "gate_stats"  # Per-archetype gate pass rates
