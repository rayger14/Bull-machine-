"""
Canonical Archetype Registry - PR#6A

Single source of truth for archetype identity, aliases, and metadata.
Replaces fragmented letter-code system with stable slug-based naming.

DESIGN:
- Each archetype has ONE canonical slug (snake_case, stable forever)
- Aliases provide backward compatibility (letters, shortcodes)
- Class name for future class-based dispatch
- Priority for ordering when multiple archetypes match

MIGRATION STRATEGY:
- Phase 1: Add this registry, resolve letters → slugs at runtime
- Phase 2: Update configs to use slugs (keep aliases working)
- Phase 3: Deprecate letter aliases (warn but don't break)
- Phase 4: Remove letter support entirely (next major version)
"""

import logging
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

# Canonical Archetype Registry
# Format: slug -> {display, class, aliases, priority, description}
ARCHETYPES = {
    # --- CORE TRAP/REVERSAL PATTERNS ---
    "wyckoff_spring_utad": {
        "display": "Spring / UTAD",
        "class": "WyckoffSpringUtad",
        "aliases": ["A", "spring", "utad", "trap_reversal"],
        "priority": 1,
        "description": "PTI-based spring/UTAD detection with displacement confirmation"
    },

    "trap_within_trend": {
        "display": "Trap Within Trend",
        "class": "TrapWithinTrend",
        "aliases": ["H", "trap", "trap_legacy", "htf_trap"],
        "priority": 5,
        "description": "HTF trend + liquidity drop + wick against trend"
    },

    # --- ORDER BLOCK / RETEST PATTERNS ---
    "order_block_retest": {
        "display": "Order Block Retest",
        "class": "OrderBlockRetest",
        "aliases": ["B", "OB", "ob_retest"],
        "priority": 2,
        "description": "BOMS strength + Wyckoff + near BOS zone"
    },

    # --- BOS / STRUCTURE PATTERNS ---
    "bos_choch_reversal": {
        "display": "BOS/CHOCH Reversal",
        "class": "BosChochReversal",
        "aliases": ["C", "bos", "choch", "fvg_continuation"],
        "priority": 3,
        "description": "Displacement + momentum + recent BOS"
    },

    # DEPRECATED: Ghost archetype - no implementation exists
    # "fvg_reclaim": {
    #     "display": "FVG Reclaim",
    #     "class": "FvgReclaim",
    #     "aliases": ["P", "fvg"],
    #     "priority": 13,
    #     "description": "Fair value gap reclaim with volume confirmation (experimental)"
    # },

    # --- LIQUIDITY PATTERNS ---
    "liquidity_sweep_reclaim": {
        "display": "Liquidity Sweep & Reclaim",
        "class": "LiquiditySweepReclaim",
        "aliases": ["G", "sweep", "liquidity_sweep"],
        "priority": 7,
        "description": "BOMS strength + rising liquidity from oversold"
    },

    # DEPRECATED: Ghost archetype - no implementation exists
    # "liquidity_cascade": {
    #     "display": "Liquidity Cascade",
    #     "class": "LiquidityCascade",
    #     "aliases": ["Q", "cascade"],
    #     "priority": 14,
    #     "description": "Multi-level liquidity run with acceleration (experimental)"
    # },

    # --- WICK / REJECTION PATTERNS ---
    "wick_trap_moneytaur": {
        "display": "Wick Trap (Moneytaur)",
        "class": "WickTrapMoneytaur",
        "aliases": ["K", "wick_trap", "moneytaur"],
        "priority": 4,
        "description": "Wick anomaly + ADX > 25 + BOS context"
    },

    # --- CONTINUATION / MOMENTUM PATTERNS ---
    "momentum_continuation": {
        "display": "Momentum Continuation",
        "class": "MomentumContinuation",
        "aliases": ["H_legacy"],  # Note: H now maps to trap_within_trend, keep this for old configs
        "priority": 5,
        "description": "Strong momentum continuation (legacy archetype)"
    },

    # --- FAILED PATTERNS ---
    "failed_continuation": {
        "display": "Failed Continuation",
        "class": "FailedContinuation",
        "aliases": ["D", "failed_fvg"],
        "priority": 8,
        "description": "FVG present + weak RSI + falling ADX"
    },

    "fakeout_real_move": {
        "display": "Fakeout → Real Move",
        "class": "FakeoutRealMove",
        "aliases": ["L", "frm", "false_break", "volume_exhaustion", "retest_cluster"],
        "priority": 6,
        "description": "Fakeout followed by genuine structural move"
    },

    # --- COMPRESSION / EXPANSION PATTERNS ---
    "liquidity_compression": {
        "display": "Liquidity Compression",
        "class": "LiquidityCompression",
        "aliases": ["E", "compression"],
        "priority": 10,
        "description": "Low ATR + narrow range + stable book depth"
    },

    "expansion_exhaustion": {
        "display": "Expansion Exhaustion",
        "class": "ExpansionExhaustion",
        "aliases": ["F", "exhaustion_reversal"],
        "priority": 9,
        "description": "Extreme RSI + high ATR + volume spike"
    },

    "range_expansion_compression_flip": {
        "display": "Range Exp/Compression Flip",
        "class": "RangeExpCompFlip",
        "aliases": ["G_alt", "range_flip"],
        "priority": 7,
        "description": "Volatility regime change with structure break"
    },

    # --- BOMS / PHASE PATTERNS ---
    "boms_phase_shift": {
        "display": "BOMS Phase Shift",
        "class": "BomsPhaseShift",
        "aliases": ["F_alt", "boms", "reaccumulation"],
        "priority": 9,
        "description": "BOMS phase transition with book depth shift"
    },

    "ratio_coil_break": {
        "display": "Ratio Coil Break (Wyckoff Insider)",
        "class": "RatioCoilBreak",
        "aliases": ["M", "coil", "wyckoff_insider", "confluence_breakout"],
        "priority": 11,
        "description": "Low ATR + near POC + BOMS strength"
    },

    # --- EXPERIMENTAL ---
    # DEPRECATED: Ghost archetype - no implementation exists
    # "htf_trap_reversal": {
    #     "display": "HTF Trap Reversal",
    #     "class": "HtfTrapReversal",
    #     "aliases": ["N"],
    #     "priority": 12,
    #     "description": "Multi-timeframe trap with HTF confirmation (experimental)"
    # },
}


def resolve_archetype_key(key: str, warn_on_alias: bool = True) -> str:
    """
    Resolve any archetype key (slug, letter, alias) to canonical slug.

    Args:
        key: User-provided key (e.g., "H", "trap", "trap_within_trend")
        warn_on_alias: Log warning when letter aliases are used

    Returns:
        Canonical slug

    Raises:
        KeyError: If key is unknown

    Examples:
        >>> resolve_archetype_key("H")
        'trap_within_trend'  # + logs warning about letter alias

        >>> resolve_archetype_key("trap")
        'trap_within_trend'  # + logs warning about alias

        >>> resolve_archetype_key("trap_within_trend")
        'trap_within_trend'  # no warning (canonical form)
    """
    k = key.lower().strip()

    # Fast path: already canonical
    if k in ARCHETYPES:
        return k

    # Resolve alias
    for slug, meta in ARCHETYPES.items():
        if k in (alias.lower() for alias in meta["aliases"]):
            # Warn on legacy letter usage
            if warn_on_alias and len(k) == 1 and k.isalpha():
                logger.warning(
                    f"Legacy letter alias '{key}' → '{slug}'. "
                    f"Update configs to use canonical slug: '{slug}'"
                )
            elif warn_on_alias:
                logger.info(f"Alias '{key}' → '{slug}'")

            return slug

    # Unknown key
    raise KeyError(
        f"Unknown archetype key: '{key}'. "
        f"Valid slugs: {list(ARCHETYPES.keys())}"
    )


def get_archetype_meta(slug: str) -> Dict:
    """
    Get metadata for a canonical archetype slug.

    Args:
        slug: Canonical archetype slug

    Returns:
        Metadata dict with {display, class, aliases, priority, description}

    Raises:
        KeyError: If slug is not in registry
    """
    if slug not in ARCHETYPES:
        raise KeyError(f"Unknown archetype slug: '{slug}'")

    return ARCHETYPES[slug]


def get_all_slugs() -> List[str]:
    """Get list of all canonical archetype slugs."""
    return list(ARCHETYPES.keys())


def get_priority_order() -> List[str]:
    """
    Get archetypes in priority order (1 = highest priority).

    Used for legacy dispatch where first match wins.
    In new dispatch, all enabled archetypes are evaluated and best is chosen.
    """
    return sorted(ARCHETYPES.keys(), key=lambda s: ARCHETYPES[s]["priority"])


def validate_config_keys(config: Dict, strict: bool = False) -> Dict[str, str]:
    """
    Validate and resolve all archetype keys in a config.

    Args:
        config: Config dict with 'archetypes' section
        strict: Raise on unknown keys (vs. skip with warning)

    Returns:
        Dict mapping original_key -> resolved_slug

    Example:
        config = {"archetypes": {"H": {...}, "trap": {...}, "unknown": {...}}}
        validate_config_keys(config, strict=False)
        # Returns: {"H": "trap_within_trend", "trap": "trap_within_trend"}
        # Logs warning about "unknown" (skipped)
    """
    resolved = {}
    archetypes = config.get("archetypes", {})

    for key in archetypes.keys():
        # Skip metadata keys (not archetype configs)
        if key in ("use_archetypes", "thresholds", "exits"):
            continue

        # Try to resolve
        try:
            slug = resolve_archetype_key(key, warn_on_alias=True)
            resolved[key] = slug
        except KeyError as e:
            if strict:
                raise
            else:
                logger.warning(f"Skipping unknown archetype key in config: {key}")

    return resolved
