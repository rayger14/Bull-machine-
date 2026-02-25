"""
Archetype Registry Manager - Centralized archetype loading and validation

Loads archetype_registry.yaml and provides filtered access to archetypes.
Validates all archetypes implement BaseArchetype contract.

Author: System Architect (Claude Code)
Date: 2025-12-12
Version: 2.0
"""

import yaml
from pathlib import Path
from typing import List, Dict, Optional
from importlib import import_module
import logging

from engine.archetypes.base_archetype import BaseArchetype

logger = logging.getLogger(__name__)


class ArchetypeRegistry:
    """
    Centralized archetype registry.

    Responsibilities:
    - Load archetype_registry.yaml
    - Validate class paths and import archetype classes
    - Verify all archetypes implement BaseArchetype
    - Provide filtered access (by maturity, regime, direction)
    - Report status of registered archetypes

    Usage:
        # Initialize registry
        registry = ArchetypeRegistry()

        # Get production archetypes
        prod_archetypes = registry.get_archetypes(maturity=['production'])

        # Get enabled archetypes from config
        enabled = registry.get_archetypes(enabled_only=True, config=config)

        # Get bear market archetypes
        bear = registry.get_archetypes(regime_tags=['risk_off'])

        # Status report
        registry.log_status_report()
    """

    def __init__(self, registry_path: Optional[Path] = None):
        """
        Initialize archetype registry.

        Args:
            registry_path: Path to archetype_registry.yaml
                          (defaults to project_root/archetype_registry.yaml)
        """
        if registry_path is None:
            # Default to project root
            project_root = Path(__file__).parent.parent.parent
            registry_path = project_root / "archetype_registry.yaml"

        self.registry_path = Path(registry_path)
        self.registry_data = None
        self.archetypes = {}
        self.deprecated = {}
        self.loaded_classes = {}  # Cache of imported archetype classes

        self._load_registry()
        self._validate_registry()

    def _load_registry(self):
        """Load and parse registry YAML"""
        if not self.registry_path.exists():
            raise FileNotFoundError(
                f"Archetype registry not found: {self.registry_path}\n"
                f"Expected location: archetype_registry.yaml in project root"
            )

        with open(self.registry_path, 'r') as f:
            self.registry_data = yaml.safe_load(f)

        logger.info(f"Loaded archetype registry from {self.registry_path}")
        logger.info(f"Registry version: {self.registry_data.get('version')}")

        # Load archetypes
        for arch_meta in self.registry_data.get('archetypes', []):
            self.archetypes[arch_meta['id']] = arch_meta

        # Load deprecated archetypes
        for arch_meta in self.registry_data.get('deprecated', []):
            self.deprecated[arch_meta['id']] = arch_meta

        logger.info(f"Loaded {len(self.archetypes)} active archetypes")
        logger.info(f"Loaded {len(self.deprecated)} deprecated archetypes")

    def _validate_registry(self):
        """Validate archetype classes implement BaseArchetype"""
        logger.info("Validating archetype implementations...")

        for arch_id, arch_meta in self.archetypes.items():
            # Skip stub archetypes (they don't have implementations yet)
            if arch_meta.get('maturity') == 'stub':
                logger.info(f"  ⊘ {arch_id:5s} STUB (no implementation expected)")
                continue

            # Try to import class
            class_path = arch_meta.get('class')
            if not class_path:
                logger.warning(f"  ✗ {arch_id:5s} No class path specified")
                continue

            try:
                module_path, class_name = class_path.rsplit('.', 1)
                module = import_module(module_path)
                archetype_class = getattr(module, class_name)

                # Verify implements BaseArchetype
                if not issubclass(archetype_class, BaseArchetype):
                    logger.error(
                        f"  ✗ {arch_id:5s} {class_name} does NOT implement BaseArchetype"
                    )
                    continue

                # Verify required class attributes set
                if archetype_class.ARCHETYPE_ID is None:
                    logger.warning(f"  ⚠ {arch_id:5s} ARCHETYPE_ID not set in class")

                # Cache the class
                self.loaded_classes[arch_id] = archetype_class

                logger.info(f"  ✓ {arch_id:5s} {class_name} implements BaseArchetype")

            except ImportError as e:
                logger.error(f"  ✗ {arch_id:5s} Failed to import: {e}")
            except AttributeError as e:
                logger.error(f"  ✗ {arch_id:5s} Class not found: {e}")
            except Exception as e:
                logger.error(f"  ✗ {arch_id:5s} Validation error: {e}")

    def get_archetypes(
        self,
        maturity: Optional[List[str]] = None,
        regime_tags: Optional[List[str]] = None,
        direction: Optional[str] = None,
        enabled_only: bool = False,
        config: Optional[Dict] = None
    ) -> List[Dict]:
        """
        Get filtered list of archetypes.

        Args:
            maturity: Filter by maturity levels (e.g., ['production', 'calibrated'])
            regime_tags: Filter by regime tags (e.g., ['risk_off', 'crisis'])
            direction: Filter by direction ('long', 'short')
            enabled_only: Only return archetypes enabled in config
            config: Optional config dict for enable flag checking

        Returns:
            List of archetype metadata dicts

        Examples:
            # Get only production archetypes
            prod = registry.get_archetypes(maturity=['production'])

            # Get bear market archetypes
            bear = registry.get_archetypes(regime_tags=['risk_off'])

            # Get enabled archetypes from config
            enabled = registry.get_archetypes(enabled_only=True, config=config)
        """
        results = []

        for arch_id, arch_meta in self.archetypes.items():
            # Filter by maturity
            if maturity and arch_meta.get('maturity') not in maturity:
                continue

            # Filter by regime tags (any overlap)
            if regime_tags:
                arch_regimes = arch_meta.get('regime_tags', [])
                if not any(tag in arch_regimes for tag in regime_tags):
                    continue

            # Filter by direction
            if direction and arch_meta.get('direction') != direction:
                continue

            # Filter by enable flag
            if enabled_only and config:
                enable_flag = arch_meta.get('enable_flag')
                if enable_flag:
                    # Check config['archetypes'][enable_flag]
                    archetypes_config = config.get('archetypes', {})
                    if not archetypes_config.get(enable_flag, False):
                        continue

            results.append(arch_meta)

        return results

    def get_archetype(self, archetype_id: str) -> Optional[Dict]:
        """Get single archetype by ID"""
        return self.archetypes.get(archetype_id)

    def get_archetype_class(self, archetype_id: str):
        """
        Get archetype class for instantiation.

        Args:
            archetype_id: Archetype ID (e.g., 'S1')

        Returns:
            Archetype class (not instance)

        Raises:
            KeyError: If archetype not found or not loaded
        """
        if archetype_id not in self.loaded_classes:
            raise KeyError(
                f"Archetype {archetype_id} not found in loaded classes. "
                f"Available: {list(self.loaded_classes.keys())}"
            )
        return self.loaded_classes[archetype_id]

    def instantiate_archetype(self, archetype_id: str) -> BaseArchetype:
        """
        Instantiate archetype class.

        Args:
            archetype_id: Archetype ID (e.g., 'S1')

        Returns:
            Archetype instance

        Raises:
            KeyError: If archetype not found
        """
        archetype_class = self.get_archetype_class(archetype_id)
        return archetype_class()

    def instantiate_all(
        self,
        maturity: Optional[List[str]] = None,
        enabled_only: bool = False,
        config: Optional[Dict] = None
    ) -> Dict[str, BaseArchetype]:
        """
        Instantiate all archetypes matching filter criteria.

        Args:
            maturity: Filter by maturity levels
            enabled_only: Only instantiate enabled archetypes
            config: Config for enable flag checking

        Returns:
            Dict mapping archetype_id -> archetype_instance

        Example:
            # Instantiate all production archetypes
            instances = registry.instantiate_all(maturity=['production'])

            # Instantiate all enabled archetypes
            instances = registry.instantiate_all(enabled_only=True, config=config)
        """
        archetypes = self.get_archetypes(
            maturity=maturity,
            enabled_only=enabled_only,
            config=config
        )

        instances = {}
        for arch_meta in archetypes:
            arch_id = arch_meta['id']

            # Skip stub archetypes (no implementation)
            if arch_meta.get('maturity') == 'stub':
                logger.debug(f"Skipping stub archetype {arch_id}")
                continue

            # Skip if class not loaded
            if arch_id not in self.loaded_classes:
                logger.warning(
                    f"Skipping {arch_id}: class not loaded (validation failed?)"
                )
                continue

            try:
                instances[arch_id] = self.instantiate_archetype(arch_id)
            except Exception as e:
                logger.error(f"Failed to instantiate {arch_id}: {e}")

        return instances

    def is_deprecated(self, archetype_id: str) -> bool:
        """Check if archetype is deprecated"""
        return archetype_id in self.deprecated

    def get_status_report(self) -> Dict:
        """
        Generate status report of all archetypes.

        Returns:
            Dict with maturity breakdown and status
        """
        by_maturity = {
            'production': [],
            'calibrated': [],
            'development': [],
            'stub': []
        }

        for arch_id, arch_meta in self.archetypes.items():
            maturity = arch_meta.get('maturity', 'stub')
            by_maturity[maturity].append(arch_id)

        return {
            'total_archetypes': len(self.archetypes),
            'by_maturity': {
                k: {'count': len(v), 'ids': v}
                for k, v in by_maturity.items()
            },
            'deprecated_count': len(self.deprecated),
            'deprecated_ids': list(self.deprecated.keys()),
            'loaded_classes_count': len(self.loaded_classes)
        }

    def log_status_report(self):
        """Log status report to console"""
        report = self.get_status_report()

        logger.info("=" * 80)
        logger.info("ARCHETYPE REGISTRY STATUS")
        logger.info("=" * 80)
        logger.info(f"Total archetypes: {report['total_archetypes']}")
        logger.info(f"Deprecated: {report['deprecated_count']}")
        logger.info(f"Loaded classes: {report['loaded_classes_count']}")
        logger.info("")

        for maturity, data in report['by_maturity'].items():
            count = data['count']
            ids = ', '.join(data['ids']) if data['ids'] else 'none'
            logger.info(f"  {maturity.upper():15s}: {count:2d}  [{ids}]")

        if report['deprecated_ids']:
            logger.info(f"\nDeprecated: {', '.join(report['deprecated_ids'])}")

        logger.info("=" * 80)


# ============================================================================
# Convenience Functions (Singleton Pattern)
# ============================================================================

_global_registry = None


def get_registry(registry_path: Optional[Path] = None) -> ArchetypeRegistry:
    """
    Get global archetype registry instance (singleton pattern).

    Args:
        registry_path: Optional path to registry YAML (only used on first call)

    Returns:
        ArchetypeRegistry instance

    Example:
        from engine.archetypes.registry_manager import get_registry

        registry = get_registry()
        prod_archetypes = registry.get_archetypes(maturity=['production'])
    """
    global _global_registry

    if _global_registry is None:
        _global_registry = ArchetypeRegistry(registry_path)

    return _global_registry


def reset_registry():
    """
    Reset global registry (useful for testing).

    Example:
        from engine.archetypes.registry_manager import reset_registry

        reset_registry()
        registry = get_registry(custom_path)
    """
    global _global_registry
    _global_registry = None


# ============================================================================
# CLI Tool for Registry Inspection
# ============================================================================

if __name__ == '__main__':
    """
    CLI tool for inspecting archetype registry.

    Usage:
        python -m engine.archetypes.registry_manager
    """
    import sys

    logging.basicConfig(
        level=logging.INFO,
        format='%(levelname)s: %(message)s'
    )

    print("=" * 80)
    print("ARCHETYPE REGISTRY INSPECTOR")
    print("=" * 80)
    print()

    try:
        registry = get_registry()
        registry.log_status_report()

        print("\n" + "=" * 80)
        print("ARCHETYPE DETAILS")
        print("=" * 80)

        for arch_id, arch_meta in registry.archetypes.items():
            print(f"\n{arch_id}: {arch_meta['name']}")
            print(f"  Slug: {arch_meta['slug']}")
            print(f"  Maturity: {arch_meta['maturity']}")
            print(f"  Direction: {arch_meta['direction']}")
            print(f"  Regime Tags: {', '.join(arch_meta['regime_tags'])}")
            print(f"  Requires Engines: {', '.join(arch_meta['requires_engines'])}")

            requires_features = arch_meta.get('requires_features', {})
            critical = requires_features.get('critical', [])
            recommended = requires_features.get('recommended', [])
            optional = requires_features.get('optional', [])

            print("  Features:")
            print(f"    Critical: {len(critical)} ({', '.join(critical[:3])}...)")
            print(f"    Recommended: {len(recommended)} ({', '.join(recommended[:3])}...)")
            print(f"    Optional: {len(optional)}")

            if arch_meta.get('maturity') == 'stub':
                impl_status = arch_meta.get('implementation_status', {})
                blockers = impl_status.get('blockers', [])
                if blockers:
                    print(f"  Blockers: {len(blockers)}")
                    for blocker in blockers:
                        print(f"    - {blocker}")

        print("\n" + "=" * 80)

    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
