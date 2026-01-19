#!/usr/bin/env python3
"""
Archetype Specification - Single Source of Truth
==================================================

Canonical representation of each archetype's contract:
- Direction (long/short/both)
- Allowed regimes
- Setup type
- Version tracking

NO OTHER FILE may define direction or regime permissions.
All trading and backtests MUST read from ArchetypeSpec.

This prevents direction inversions and regime mismatches.

Author: Claude Code
Date: 2026-01-12
"""

import yaml
from pathlib import Path
from typing import List, Dict, Optional, Literal
from dataclasses import dataclass
import hashlib
import logging

logger = logging.getLogger(__name__)


@dataclass
class ArchetypeSpec:
    """Canonical archetype specification - the single source of truth."""

    id: str
    name: str
    slug: str
    direction: Literal['long', 'short', 'both']
    setup_type: Literal['trend', 'countertrend', 'mean_reversion', 'breakout', 'squeeze']
    allowed_regimes: List[str]
    maturity: Literal['production', 'beta', 'experimental', 'deprecated']
    class_path: str
    regime_tags: List[str]
    description: str
    version: Optional[str] = None

    def is_allowed_in_regime(self, regime: str) -> bool:
        """Check if archetype is permitted in this regime."""
        return regime in self.allowed_regimes

    def validate_trade_direction(self, trade_direction: str) -> bool:
        """Validate that a trade direction matches this archetype's intent."""
        if self.direction == 'both':
            return trade_direction in ['long', 'short']
        return trade_direction == self.direction

    def __str__(self) -> str:
        return f"{self.slug} ({self.direction}, regimes={self.allowed_regimes})"


class ArchetypeRegistry:
    """
    Registry of all archetype specifications.

    Loads from archetype_registry.yaml and provides:
    - Canonical direction lookup
    - Regime validation
    - Spec manifest for audit
    """

    def __init__(self, registry_path: str = "archetype_registry.yaml"):
        """Load archetype registry and create canonical specs."""
        self.registry_path = Path(registry_path)
        self.specs: Dict[str, ArchetypeSpec] = {}
        self._load_registry()

        # Runtime tracking
        self.spec_mismatch_count = 0
        self.direction_mismatch_count = 0
        self.regime_block_count = 0

        logger.info(f"[ArchetypeRegistry] Loaded {len(self.specs)} archetype specs")

    def _load_registry(self):
        """Load archetype specifications from YAML."""
        if not self.registry_path.exists():
            raise FileNotFoundError(f"Archetype registry not found: {self.registry_path}")

        with open(self.registry_path, 'r') as f:
            data = yaml.safe_load(f)

        # Parse active archetypes
        for item in data.get('archetypes', []):
            # Skip deprecated
            if item.get('maturity') == 'deprecated':
                continue

            # Extract allowed regimes from regime_tags
            regime_tags = item.get('regime_tags', [])
            allowed_regimes = self._parse_allowed_regimes(regime_tags)

            # Determine setup type from description or tags
            setup_type = self._infer_setup_type(item)

            spec = ArchetypeSpec(
                id=item['id'],
                name=item['name'],
                slug=item['slug'],
                direction=item.get('direction', 'long'),  # Default to long if not specified
                setup_type=setup_type,
                allowed_regimes=allowed_regimes,
                maturity=item.get('maturity', 'experimental'),
                class_path=item.get('class', ''),
                regime_tags=regime_tags,
                description=item.get('description', ''),
                version=item.get('version')
            )

            self.specs[spec.slug] = spec

    def _parse_allowed_regimes(self, regime_tags: List[str]) -> List[str]:
        """
        Parse regime tags into allowed regimes.

        Handles both explicit regimes and tags like 'bear_market', 'crisis_periods'.
        """
        allowed = set()

        for tag in regime_tags:
            if tag in ['crisis', 'risk_off', 'neutral', 'risk_on']:
                allowed.add(tag)
            elif tag == 'bear_market':
                allowed.update(['crisis', 'risk_off'])
            elif tag == 'bull_market':
                allowed.update(['risk_on', 'neutral'])
            elif tag == 'crisis_periods':
                allowed.add('crisis')

        # Default: allow all if no specific regimes specified
        if not allowed:
            allowed = {'crisis', 'risk_off', 'neutral', 'risk_on'}

        return sorted(list(allowed))

    def _infer_setup_type(self, item: Dict) -> str:
        """Infer setup type from archetype metadata."""
        desc = item.get('description', '').lower()
        name = item.get('name', '').lower()

        if 'counter' in desc or 'counter-trend' in desc or 'squeeze' in name:
            return 'countertrend'
        elif 'mean reversion' in desc or 'reversion' in desc:
            return 'mean_reversion'
        elif 'break' in desc or 'breakout' in name:
            return 'breakout'
        elif 'trend' in desc:
            return 'trend'
        elif 'squeeze' in desc:
            return 'squeeze'
        else:
            return 'mean_reversion'  # Conservative default

    def get_spec(self, slug: str) -> Optional[ArchetypeSpec]:
        """Get archetype specification by slug."""
        return self.specs.get(slug)

    def get_direction(self, slug: str) -> str:
        """Get canonical direction for an archetype."""
        spec = self.get_spec(slug)
        if not spec:
            logger.warning(f"[ArchetypeRegistry] Unknown archetype: {slug}, defaulting to 'long'")
            return 'long'
        return spec.direction

    def is_allowed_in_regime(self, slug: str, regime: str) -> bool:
        """Check if archetype is allowed in this regime."""
        spec = self.get_spec(slug)
        if not spec:
            logger.warning(f"[ArchetypeRegistry] Unknown archetype: {slug}, allowing by default")
            return True
        return spec.is_allowed_in_regime(regime)

    def validate_trade(
        self,
        slug: str,
        direction: str,
        regime: str,
        raise_on_mismatch: bool = False
    ) -> tuple[bool, str]:
        """
        Validate that a trade matches the archetype's canonical spec.

        Returns:
            (is_valid, reason)
        """
        spec = self.get_spec(slug)

        if not spec:
            self.spec_mismatch_count += 1
            msg = f"Unknown archetype: {slug}"
            if raise_on_mismatch:
                raise ValueError(msg)
            return False, msg

        # Check direction
        if not spec.validate_trade_direction(direction):
            self.direction_mismatch_count += 1
            msg = f"Direction mismatch: {slug} spec={spec.direction}, trade={direction}"
            if raise_on_mismatch:
                raise ValueError(msg)
            return False, msg

        # Check regime
        if not spec.is_allowed_in_regime(regime):
            self.regime_block_count += 1
            msg = f"Regime not allowed: {slug} in {regime}, allowed={spec.allowed_regimes}"
            logger.debug(msg)
            return False, msg

        return True, "OK"

    def get_manifest(self) -> Dict:
        """
        Get specification manifest for audit.

        Returns summary of all specs + file hash.
        """
        # Calculate registry file hash
        with open(self.registry_path, 'rb') as f:
            registry_hash = hashlib.md5(f.read()).hexdigest()[:8]

        specs_summary = {}
        for slug, spec in self.specs.items():
            specs_summary[slug] = {
                'direction': spec.direction,
                'setup_type': spec.setup_type,
                'allowed_regimes': spec.allowed_regimes,
                'maturity': spec.maturity
            }

        return {
            'registry_path': str(self.registry_path),
            'registry_hash': registry_hash,
            'total_specs': len(self.specs),
            'specs': specs_summary,
            'counters': {
                'spec_mismatch': self.spec_mismatch_count,
                'direction_mismatch': self.direction_mismatch_count,
                'regime_block': self.regime_block_count
            }
        }

    def print_manifest(self):
        """Print specification manifest for audit trail."""
        manifest = self.get_manifest()

        print("="*80)
        print("ARCHETYPE SPECIFICATION MANIFEST")
        print("="*80)
        print()
        print(f"Registry File: {manifest['registry_path']}")
        print(f"Registry Hash: {manifest['registry_hash']}")
        print(f"Total Specs:   {manifest['total_specs']}")
        print()
        print("Archetype Specifications:")
        print("-" * 80)

        for slug, spec in manifest['specs'].items():
            print(f"  {slug:25s} | {spec['direction']:6s} | {spec['setup_type']:15s} | {', '.join(spec['allowed_regimes'])}")

        print()
        print("Runtime Counters:")
        print(f"  Spec mismatches:      {manifest['counters']['spec_mismatch']}")
        print(f"  Direction mismatches: {manifest['counters']['direction_mismatch']}")
        print(f"  Regime blocks:        {manifest['counters']['regime_block']}")
        print()

        # Validate counters
        if manifest['counters']['spec_mismatch'] > 0:
            print("⚠️  WARNING: Spec mismatches detected - unknown archetypes referenced")
        if manifest['counters']['direction_mismatch'] > 0:
            print("❌ CRITICAL: Direction mismatches detected - trades violating spec direction")

        print("="*80)

    def get_all_slugs(self) -> List[str]:
        """Get list of all archetype slugs."""
        return list(self.specs.keys())


# Global registry instance (lazy loaded)
_registry: Optional[ArchetypeRegistry] = None


def get_registry(registry_path: str = "archetype_registry.yaml") -> ArchetypeRegistry:
    """Get or create global archetype registry."""
    global _registry
    if _registry is None:
        _registry = ArchetypeRegistry(registry_path)
    return _registry
