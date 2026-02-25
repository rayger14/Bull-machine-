"""
Parameter Alias Registry - Quick Fix for Name Mismatch

Maps canonical (long) parameter names to their legacy (short) aliases.
This allows both old configs (fusion, pti) and new code (fusion_threshold,
pti_score_threshold) to work together during migration.

PHASE 1 QUICK FIX: Minimal implementation to unblock optimization.
PHASE 2 TODO: Expand to full registry pattern with type validation.
"""

# Canonical parameter names and their accepted aliases
ALIASES = {
    "spring": {
        "fusion_threshold": ["fusion", "fusion_th"],
        "pti_score_threshold": ["pti", "pti_score"],
        "disp_atr_multiplier": ["disp_atr"],
    },
    "order_block_retest": {
        "fusion_threshold": ["fusion", "fusion_th"],
        "boms_strength_min": ["boms_strength", "boms"],
        "wyckoff_min": ["wyckoff", "wyckoff_score"],
    },
    "wick_trap": {
        "fusion_threshold": ["fusion", "fusion_th"],
        "disp_atr_multiplier": ["disp_atr"],
        "momentum_min": ["momentum", "mom"],
        "tf4h_fusion_min": ["tf4h_fusion", "tf4h"],
    },
    "wick_trap_moneytaur": {
        "fusion_threshold": ["fusion", "fusion_th"],
        "adx_threshold": ["adx", "adx_th"],
        "liquidity_threshold": ["liquidity", "liq"],
    },
    "trap_within_trend": {
        "fusion_threshold": ["fusion", "fusion_th"],
        "adx_threshold": ["adx", "adx_th"],
        "liquidity_threshold": ["liquidity", "liq"],
        "quality_threshold": ["quality", "q"],
        "wick_multiplier": ["wick_mult", "wick"],
    },
    "momentum_continuation": {
        "fusion_threshold": ["fusion", "fusion_th"],
        "adx_threshold": ["adx", "adx_th"],
        "liquidity_threshold": ["liquidity", "liq"],
    },
    "retest_cluster": {
        "fusion_threshold": ["fusion", "fusion_th"],
        "vol_z_min": ["vol_z"],
        "rsi_min": ["rsi"],
    },
    "exhaustion_reversal": {
        "fusion_threshold": ["fusion", "fusion_th"],
        "rsi_min": ["rsi_ext", "rsi"],
        "atr_percentile_min": ["atr_pctile", "atr_percentile"],
        "vol_z_min": ["vol_z"],
    },
    "failed_continuation": {
        "fusion_threshold": ["fusion", "fusion_th"],
        "rsi_max": ["rsi_max_thresh", "rsi"],
    },
    "liquidity_sweep": {
        "fusion_threshold": ["fusion", "fusion_th"],
        "boms_strength_min": ["boms_strength", "boms"],
        "liquidity_min": ["liq", "liquidity"],
    },
    "volume_exhaustion": {
        "fusion_threshold": ["fusion", "fusion_th"],
        "atr_percentile_max": ["atr_pctile"],
        "vol_z_min": ["vol_z_min_thresh"],
        "vol_z_max": ["vol_z_max_thresh"],
        "vol_cluster_min": ["vol_cluster"],
    },
    "confluence_breakout": {
        "fusion_threshold": ["fusion", "fusion_th"],
        "atr_percentile_max": ["atr_pctile"],
        "poc_dist_max": ["poc_dist"],
        "boms_strength_min": ["boms_strength", "boms"],
    },
}

# Legacy letter code to canonical slug mapping
LETTER_MAP = {
    "A": "spring",
    "B": "order_block_retest",
    "C": "wick_trap",
    "D": "failed_continuation",
    "E": "volume_exhaustion",
    "F": "exhaustion_reversal",
    "G": "liquidity_sweep",
    "H": "momentum_continuation",
    "K": "trap_within_trend",
    "L": "retest_cluster",
    "M": "confluence_breakout",
}

# Archetype slug aliases - maps display names used in code to canonical threshold names
# This fixes naming mismatches where _pattern_X uses one name but ARCHETYPE_NAMES uses another
ARCHETYPE_SLUG_ALIASES = {
    # M archetype: code uses "coil_break", canonical is "confluence_breakout"
    "coil_break": "confluence_breakout",
    "ratio_coil_break": "confluence_breakout",

    # G archetype: code uses "re_accumulate", canonical is "liquidity_sweep"
    "re_accumulate": "liquidity_sweep",
    "re_accumulation": "liquidity_sweep",

    # Add any other known mismatches here
}


def resolve_canonical(slug: str, key: str) -> str:
    """
    Resolve any parameter alias to its canonical name.

    Args:
        slug: Archetype slug (e.g., 'trap_within_trend')
        key: Parameter key (short or long form)

    Returns:
        Canonical parameter name

    Examples:
        resolve_canonical('spring', 'fusion') -> 'fusion_threshold'
        resolve_canonical('spring', 'pti') -> 'pti_score_threshold'
        resolve_canonical('spring', 'fusion_threshold') -> 'fusion_threshold'
    """
    archetype_aliases = ALIASES.get(slug, {})

    # Already canonical?
    if key in archetype_aliases:
        return key

    # Search aliases
    for canon, alias_list in archetype_aliases.items():
        if key in alias_list:
            return canon

    # Unknown - return as-is (fallback)
    return key


def get_all_aliases(slug: str, canonical_key: str) -> list:
    """
    Get all known aliases for a canonical parameter.

    Args:
        slug: Archetype slug
        canonical_key: Canonical parameter name

    Returns:
        List of alias strings (including canonical)
    """
    archetype_aliases = ALIASES.get(slug, {})
    aliases = archetype_aliases.get(canonical_key, [])
    return [canonical_key] + aliases
