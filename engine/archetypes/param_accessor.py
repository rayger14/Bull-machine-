"""
Single source of truth for archetype parameters - PR#6A Refactor.

This module provides a unified interface for reading archetype parameters,
eliminating the disconnect between config['archetypes'][...] and self.thresh_*.

MIGRATION STRATEGY:
- Legacy: get_archetype_param(config, arch, key, default) - simple dict lookup
- New (PR#6A): get_param(ctx, slug, key, default) - canonical with fallbacks
- The new function is migration-safe: checks canonical → legacy → default
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


# ============================================================================
# PR#6A: Canonical Parameter Access with Migration Safety
# ============================================================================

def get_param(ctx: Any, slug: str, key: str, default: Any) -> Any:
    """
    **PR#6A Single Parameter Source** - Migration-safe canonical accessor.

    This is the ONLY function archetypes should use to read parameters.
    Replaces fragmented `self.thresh_X[...]` hardcoded dict access.

    Args:
        ctx: Context object with .config attribute
        slug: Canonical archetype slug (e.g., 'trap_within_trend', 'wick_trap_moneytaur')
              Can also accept letter aliases ('H', 'K') which are resolved
        key: Parameter key (e.g., 'fusion_threshold', 'adx_threshold')
        default: Default value if not found

    Returns:
        Parameter value from first available source:
        1. Canonical: config['archetypes'][slug][key]
        2. Legacy ThresholdPolicy: config['archetypes']['thresholds'][key]
        3. Legacy letter code: config['archetypes']['thresholds'][letter][key]
        4. Default value

    Migration Strategy:
        - Old code: self.thresh_H.get('fusion', 0.36)  # hardcoded, disconnected
        - New code: get_param(ctx, 'trap_within_trend', 'fusion_threshold', 0.36)
        - Works with old configs (thresholds.H) AND new configs (trap_within_trend)

    Example:
        # Old way (BAD - causes zero variance bug):
        adx_threshold = self.thresh_K.get('adx', 25.0)  # reads from thresholds.K
        # Optimizer writes to config['archetypes']['trap_within_trend'] → MISMATCH!

        # New way (GOOD - single source):
        adx_threshold = get_param(ctx, 'wick_trap_moneytaur', 'adx_threshold', 25.0)
        # Resolves 'wick_trap_moneytaur' → checks canonical location → falls back to legacy

    Observability:
        Integrates with ParamEcho to log actual values read for audit trail.
    """
    from engine.archetypes.registry import resolve_archetype_key
    from engine import feature_flags as features

    config = ctx.config if hasattr(ctx, 'config') else ctx

    # Resolve slug (handles aliases like 'H' → 'trap_within_trend')
    try:
        canonical_slug = resolve_archetype_key(slug, warn_on_alias=features.WARN_ON_LETTER_ALIASES)
    except KeyError:
        canonical_slug = slug  # Use as-is if not in registry

    # MIGRATION-SAFE FALLBACK CHAIN
    # 1. Try canonical location (NEW): config['archetypes'][slug][key]
    value = config.get('archetypes', {}).get(canonical_slug, {}).get(key)
    if value is not None:
        _log_param_source(canonical_slug, key, value, f"config['archetypes']['{canonical_slug}']")
        return value

    # 2. Try legacy letter code thresholds (LEGACY): config['archetypes']['thresholds'][letter][key]
    # This handles old configs that still use letter codes
    from engine.archetypes.registry import ARCHETYPES
    if canonical_slug in ARCHETYPES:
        for alias in ARCHETYPES[canonical_slug]['aliases']:
            if len(alias) == 1 and alias.isalpha():  # Letter code
                letter_val = config.get('archetypes', {}).get('thresholds', {}).get(alias, {}).get(key)
                if letter_val is not None:
                    logger.debug(f"[LEGACY FALLBACK] {canonical_slug}.{key} = {letter_val} from thresholds.{alias}")
                    return letter_val

    # 3. Try original slug as-is (handles raw letter codes passed directly)
    if slug != canonical_slug:
        direct_val = config.get('archetypes', {}).get('thresholds', {}).get(slug, {}).get(key)
        if direct_val is not None:
            logger.debug(f"[DIRECT FALLBACK] {slug}.{key} = {direct_val} from thresholds.{slug}")
            return direct_val

    # 4. Default
    _log_param_source(canonical_slug, key, default, "default")
    return default


def _log_param_source(slug: str, key: str, value: Any, source: str):
    """Helper to log parameter source for debugging."""
    from engine import feature_flags as features
    if features.PARAM_ECHO_ENABLED:
        logger.debug(f"[PARAM] {slug}.{key} = {value} (source: {source})")


# ============================================================================
# Legacy Accessor (kept for backward compatibility)
# ============================================================================

def get_archetype_param(config: Dict, archetype: str, key: str, default: Any) -> Any:
    """
    Read archetype parameter from config with single source of truth.

    This accessor eliminates the historical disconnect where:
    - Optimizers wrote to: config['archetypes'][archetype][key]
    - Archetype logic read from: self.thresh_X[key] (disconnected!)

    Args:
        config: Full config dict
        archetype: Archetype name (e.g., 'trap_within_trend', 'order_block_retest')
        key: Parameter key (e.g., 'quality_threshold', 'boms_strength_min')
        default: Default value if not found

    Returns:
        Parameter value from config or default

    Usage:
        quality_th = get_archetype_param(config, 'trap_within_trend', 'quality_threshold', 0.55)

    Wire Test:
        Changing config['archetypes']['trap_within_trend']['quality_threshold']
        will now ACTUALLY affect the archetype's behavior.
    """
    return config.get('archetypes', {}).get(archetype, {}).get(key, default)


def log_params_used(config: Dict, archetype: str, output_path: str) -> None:
    """
    Log all parameters actually used by an archetype.

    Creates a params_used.json file for audit trail. This proves that
    parameters are being read and allows validation that optimizer-set
    values are actually being used.

    Args:
        config: Full config dict
        archetype: Archetype name
        output_path: Path to save params_used JSON

    Creates:
        JSON file with structure:
        {
          "archetype": "trap_within_trend",
          "params": {"quality_threshold": 0.55, ...},
          "timestamp": "2025-11-06T..."
        }

    Usage:
        log_params_used(config, 'trap_within_trend', 'results/run_001/params_trap.json')
    """
    arch_config = config.get('archetypes', {}).get(archetype, {})

    params_snapshot = {
        'archetype': archetype,
        'params': arch_config,
        'timestamp': datetime.now().isoformat(),
        'config_path': config.get('_config_path', 'unknown')  # If available
    }

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(params_snapshot, f, indent=2)

    # Also print to stdout for immediate visibility
    params_str = json.dumps(arch_config, indent=2)
    print(f"[PARAMS] {archetype}: {params_str}")


def validate_param_schema(config: Dict, archetype: str, schema: Dict) -> bool:
    """
    Validate that archetype parameters match expected schema.

    Args:
        config: Full config dict
        archetype: Archetype name
        schema: Expected schema dict with keys and types
            Example: {'quality_threshold': float, 'confirmation_bars': int}

    Returns:
        True if valid, False otherwise (logs errors)

    Usage:
        schema = {'quality_threshold': float, 'adx_threshold': float}
        if not validate_param_schema(config, 'trap_within_trend', schema):
            raise ValueError("Invalid trap parameters")
    """
    arch_config = config.get('archetypes', {}).get(archetype, {})

    valid = True
    for key, expected_type in schema.items():
        if key in arch_config:
            value = arch_config[key]
            if not isinstance(value, expected_type):
                print(f"❌ Invalid param type: {archetype}.{key} = {value} (expected {expected_type.__name__})")
                valid = False
        # Missing params are OK (will use defaults), so no check needed

    return valid


# Archetype parameter schemas (for validation)
ARCHETYPE_SCHEMAS = {
    'trap_within_trend': {
        'quality_threshold': (float, int),
        'liquidity_threshold': (float, int),
        'adx_threshold': (float, int),
        'fusion_threshold': (float, int),
        'wick_multiplier': (float, int),
        'confirmation_bars': int,
        'volume_ratio': (float, int),
        'stop_multiplier': (float, int)
    },
    'order_block_retest': {
        'boms_strength_min': (float, int),
        'wyckoff_min': (float, int),
        'ob_quality_threshold': (float, int),
        'disp_z_min': (float, int),
        'stop_multiplier': (float, int)
    }
}


def validate_archetype_config(config: Dict, archetype: str) -> bool:
    """
    Validate archetype config against known schema.

    Args:
        config: Full config dict
        archetype: Archetype name

    Returns:
        True if valid or no schema defined, False if validation fails
    """
    if archetype not in ARCHETYPE_SCHEMAS:
        return True  # No schema = no validation needed

    schema = ARCHETYPE_SCHEMAS[archetype]
    arch_config = config.get('archetypes', {}).get(archetype, {})

    valid = True
    for key, value in arch_config.items():
        if key not in schema:
            print(f"⚠️  Unknown param: {archetype}.{key} (not in schema)")
            continue

        expected_types = schema[key]
        if not isinstance(expected_types, tuple):
            expected_types = (expected_types,)

        if not isinstance(value, expected_types):
            print(f"❌ Invalid type: {archetype}.{key} = {value} (expected {expected_types})")
            valid = False

    return valid
