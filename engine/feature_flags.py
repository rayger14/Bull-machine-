"""
Feature Flags - PR#6A Archetype Refactor

Controls progressive migration from legacy letter-code system
to canonical slug-based archetype registry.
"""

# PHASE 0: Feature flags for safe migration
USE_CANONICAL_ARCHETYPE_REGISTRY = True  # Enable new registry.py lookup
ROUTER_SOFT_FILTERS = True  # Turn hard vetoes into soft weights
PARAM_ECHO_ENABLED = True  # Log actual params used per run
GATE_TRACING_ENABLED = True  # Track per-gate pass rates

# PHASE 1: Warnings for legacy usage
WARN_ON_LETTER_ALIASES = True  # Log when letter codes are used
STRICT_SLUG_VALIDATION = False  # Raise on unknown keys (vs. fallback)

# PHASE 3: Dispatch behavior
EVALUATE_ALL_ARCHETYPES = False  # Score all enabled, pick best (no early return) - DISABLED for baseline
LEGACY_PRIORITY_ORDER = True  # Use A-M priority or score-based selection - ENABLED for baseline

# PHASE 4: Filter softening - DISABLED for baseline validation
SOFT_LIQUIDITY_FILTER = False  # Apply weight penalty vs hard reject
SOFT_REGIME_FILTER = False  # Apply weight penalty vs hard reject
SOFT_SESSION_FILTER = False  # Apply weight penalty vs hard reject

# Observability output paths
ARTIFACTS_BASE = "artifacts"  # Base path for run artifacts
PARAM_ECHO_FILE = "params_used.json"  # Per-run param snapshot
GATE_STATS_DIR = "gate_stats"  # Per-archetype gate pass rates
