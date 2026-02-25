#!/usr/bin/env python3
"""
Archetype Factory - Production Integration Layer

Loads archetype implementations dynamically from the registry and provides
a unified interface for archetype evaluation during backtesting.

CRITICAL INTEGRATION COMPONENT:
This replaces placeholder logic with real archetype implementations from:
- engine/strategies/archetypes/bull/*.py
- engine/strategies/archetypes/bear/*_runtime.py

REGISTRY-DRIVEN ARCHITECTURE:
All archetypes are defined in archetype_registry.yaml with:
- Archetype ID (A, B, C, K, H, S1, S4, S5, etc.)
- Direction (long/short)
- Class path
- Required features
- Enable flags

USAGE:
```python
factory = ArchetypeFactory(config)
result = factory.evaluate_archetype('order_block_retest', bar, regime)
# Returns: (confidence, direction, metadata)
```

Author: Claude Code (Backend Architect)
Date: 2026-01-08
"""

import logging
import yaml
from pathlib import Path
from typing import Dict, Optional, Tuple, Any
import pandas as pd

logger = logging.getLogger(__name__)


class ArchetypeFactory:
    """
    Factory for loading and evaluating archetype implementations.

    Maps archetype slugs → production implementations with optimized configs.
    """

    def __init__(self, config: Dict[str, Any], archetype_config_overrides: Optional[Dict[str, Dict[str, Any]]] = None):
        """
        Initialize archetype factory.

        Args:
            config: System configuration with enable flags and thresholds
            archetype_config_overrides: Optional dict of archetype-specific config overrides
                Format: {archetype_slug: {param_name: value, ...}}
                Example: {'liquidity_vacuum': {'fusion_threshold': 0.44, 'liquidity_weight': 0.32}}
                Used for walk-forward validation to apply optimized parameters per window
        """
        self.config = config
        self.archetype_config_overrides = archetype_config_overrides or {}
        self.registry = self._load_registry()
        self.instances: Dict[str, Any] = {}
        self._initialize_archetypes()

        logger.info(f"[ArchetypeFactory] Initialized with {len(self.instances)} active archetypes")
        for slug in sorted(self.instances.keys()):
            logger.info(f"  - {slug}")

        if self.archetype_config_overrides:
            logger.info(f"[ArchetypeFactory] Applied config overrides for {len(self.archetype_config_overrides)} archetypes")
            for slug in sorted(self.archetype_config_overrides.keys()):
                logger.info(f"  - {slug}: {len(self.archetype_config_overrides[slug])} params")

    def _load_registry(self) -> Dict[str, Any]:
        """Load archetype registry from YAML."""
        registry_path = Path(__file__).parent.parent.parent / 'archetype_registry.yaml'

        if not registry_path.exists():
            logger.error(f"[ArchetypeFactory] Registry not found: {registry_path}")
            return {'archetypes': []}

        with open(registry_path, 'r') as f:
            registry_data = yaml.safe_load(f)

        logger.info(f"[ArchetypeFactory] Loaded registry from {registry_path}")
        return registry_data

    def _initialize_archetypes(self):
        """
        Initialize archetype instances based on registry and config.

        Only loads archetypes that are:
        1. Enabled in config
        2. Have production/calibrated maturity (skip stubs)
        3. Have valid class paths
        """
        for archetype in self.registry.get('archetypes', []):
            archetype_id = archetype['id']
            slug = archetype['slug']
            maturity = archetype.get('maturity', 'stub')
            enable_flag = archetype.get('enable_flag', f'enable_{archetype_id}')

            # Check if enabled
            if not self.config.get(enable_flag, False):
                logger.debug(f"[ArchetypeFactory] Skipping {slug} - disabled by config")
                continue

            # Skip stub archetypes (not yet implemented)
            if maturity == 'stub':
                logger.warning(f"[ArchetypeFactory] Skipping {slug} - stub implementation (not production-ready)")
                continue

            # Load archetype instance
            try:
                instance = self._load_archetype_instance(archetype)
                if instance:
                    self.instances[slug] = {
                        'instance': instance,
                        'id': archetype_id,
                        'direction': archetype['direction'],
                        'maturity': maturity,
                        'requires_features': archetype.get('requires_features', {})
                    }
                    logger.info(f"[ArchetypeFactory] Loaded {slug} ({archetype_id}) - {maturity}")
            except Exception as e:
                logger.error(f"[ArchetypeFactory] Failed to load {slug}: {e}")

    def _load_archetype_instance(self, archetype: Dict[str, Any]) -> Optional[Any]:
        """
        Dynamically load archetype class and instantiate with config.

        Args:
            archetype: Archetype definition from registry

        Returns:
            Archetype instance or None if load failed
        """
        class_path = archetype.get('class')
        if not class_path:
            logger.error(f"[ArchetypeFactory] No class path for {archetype['slug']}")
            return None

        # Parse class path: "engine.strategies.archetypes.bull.OrderBlockRetestArchetype"
        try:
            module_path, class_name = class_path.rsplit('.', 1)
            module = __import__(module_path, fromlist=[class_name])
            archetype_class = getattr(module, class_name)

            # Create instance with optimized config
            archetype_config = self._get_archetype_config(archetype)
            instance = archetype_class(config=archetype_config)

            return instance

        except Exception as e:
            logger.error(f"[ArchetypeFactory] Failed to load {class_path}: {e}")
            return None

    def _get_archetype_config(self, archetype: Dict[str, Any]) -> Dict[str, Any]:
        """
        Build archetype-specific config with optimized thresholds.

        Merges:
        1. Global config
        2. Archetype-specific default thresholds
        3. Walk-forward parameter overrides (if provided)

        Priority: walk-forward overrides > default thresholds > global config
        """
        archetype_id = archetype['id']
        slug = archetype['slug']

        # Start with base config
        config = {
            'thresholds': {}
        }

        # Apply archetype-specific optimized configs
        # These are from recent optimization runs documented in:
        # - ARCHETYPE_A_OPTIMIZATION.md (min_fusion_score=0.45)
        # - ARCHETYPE_K_REFACTOR_REPORT.md (production config)
        # - ARCHETYPE_B_REFACTOR_REPORT.md (production config)

        if slug == 'wyckoff_spring_utad':  # Archetype A
            config['thresholds'] = {
                'min_fusion_score': 0.45,  # Optimized from 0.35
                'min_wick_ratio': 0.35,     # Optimized from 0.30
                'rsi_threshold': 75,        # Tightened from 80
                'pti_weight': 0.35,
                'wyckoff_weight': 0.30,
                'smc_weight': 0.25,
                'volume_weight': 0.10
            }

        elif slug == 'wick_trap_moneytaur':  # Archetype K
            config['thresholds'] = {
                'min_wick_lower_ratio': 0.40,
                'min_adx': 25,
                'min_fusion_score': 0.40,
                'smc_weight': 0.40,
                'price_action_weight': 0.30,
                'momentum_weight': 0.20,
                'liquidity_weight': 0.10
            }

        elif slug == 'order_block_retest':  # Archetype B
            config['thresholds'] = {
                'max_distance_from_ob': 0.05,
                'min_bounce_body': 0.30,
                'min_fusion_score': 0.35,
                'smc_weight': 0.35,
                'price_action_weight': 0.25,
                'wyckoff_weight': 0.20,
                'volume_weight': 0.15,
                'regime_weight': 0.05
            }

        elif slug == 'bos_choch_reversal':  # Archetype C
            config['thresholds'] = {
                'min_displacement': 0.02,
                'min_momentum_score': 0.50,
                'min_fusion_score': 0.40,
                'smc_weight': 0.50,
                'momentum_weight': 0.30,
                'volume_weight': 0.20
            }

        elif slug == 'trap_within_trend':  # Archetype H
            config['thresholds'] = {
                'min_trend_strength': 0.60,
                'min_wick_rejection': 0.30,
                'min_fusion_score': 0.40,
                'trend_weight': 0.35,
                'smc_weight': 0.30,
                'liquidity_weight': 0.25,
                'volume_weight': 0.10
            }

        # Bear archetypes (S1, S4, S5) use default configs from their classes
        # as they already have optimized thresholds

        # Apply walk-forward parameter overrides (highest priority)
        if slug in self.archetype_config_overrides:
            overrides = self.archetype_config_overrides[slug]
            logger.debug(f"[ArchetypeFactory] Applying {len(overrides)} overrides to {slug}")

            # Merge overrides into thresholds
            config['thresholds'].update(overrides)

            # Also add overrides at top level for archetypes that read params directly
            config.update(overrides)

        return config

    def evaluate_archetype(
        self,
        archetype_slug: str,
        bar: pd.Series,
        regime_label: str = 'neutral'
    ) -> Tuple[float, str, Dict[str, Any]]:
        """
        Evaluate specific archetype on current bar.

        Args:
            archetype_slug: Archetype identifier (e.g., 'order_block_retest')
            bar: Current bar data with OHLCV + features
            regime_label: Current regime classification

        Returns:
            Tuple of (confidence, direction, metadata)
            Returns (0.0, 'hold', {}) if no signal
        """
        # Check if archetype is loaded
        if archetype_slug not in self.instances:
            logger.debug(f"[ArchetypeFactory] Archetype {archetype_slug} not loaded")
            return 0.0, 'hold', {'reason': 'not_loaded'}

        archetype_data = self.instances[archetype_slug]
        instance = archetype_data['instance']
        archetype_id = archetype_data['id']
        expected_direction = archetype_data['direction']

        try:
            # Call archetype's detect() method
            # All archetype implementations follow this interface:
            # detect(row: pd.Series, regime_label: str) -> Tuple[Optional[str], float, Dict]

            result = instance.detect(bar, regime_label)

            if result is None or len(result) != 3:
                logger.warning(f"[ArchetypeFactory] Invalid result from {archetype_slug}: {result}")
                return 0.0, 'hold', {'reason': 'invalid_result'}

            archetype_name, confidence, metadata = result

            # No signal detected
            if archetype_name is None or confidence == 0.0:
                return 0.0, 'hold', metadata

            # Signal detected - return with expected direction from registry
            return confidence, expected_direction, metadata

        except Exception as e:
            logger.error(f"[ArchetypeFactory] Error evaluating {archetype_slug}: {e}", exc_info=True)
            return 0.0, 'hold', {'reason': 'evaluation_error', 'error': str(e)}

    def get_active_archetypes(self) -> list[str]:
        """
        Get list of active archetype slugs.

        Returns:
            List of archetype slugs that are loaded and ready
        """
        return list(self.instances.keys())

    def get_archetype_info(self, archetype_slug: str) -> Optional[Dict[str, Any]]:
        """
        Get metadata about specific archetype.

        Args:
            archetype_slug: Archetype identifier

        Returns:
            Dict with archetype metadata or None if not found
        """
        return self.instances.get(archetype_slug)
