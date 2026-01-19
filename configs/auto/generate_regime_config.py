#!/usr/bin/env python3
"""
Generate regime-specific configs from base config and extracted thresholds.

This script creates 4 config files (one per regime) with:
- Regime-specific thresholds from hmm_thresholds.json
- Base structure from mvp_bull_market_v1.json
- Regime override to force the specific regime

Usage:
    python configs/auto/generate_regime_config.py
"""

import sys
from pathlib import Path
import json
from typing import Dict, Any

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def load_json(path: Path) -> Dict[str, Any]:
    """Load JSON file with error handling."""
    try:
        with open(path, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading {path}: {e}")
        raise


def save_json(data: Dict[str, Any], path: Path) -> None:
    """Save JSON file with pretty formatting."""
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)
        print(f"Saved: {path}")
    except Exception as e:
        print(f"Error saving {path}: {e}")
        raise


def apply_regime_thresholds(
    base_config: Dict[str, Any],
    regime_id: int,
    regime_data: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Apply regime-specific thresholds to base config.

    Args:
        base_config: Base configuration dictionary
        regime_id: Regime ID (0-3)
        regime_data: Threshold data for this regime

    Returns:
        Modified config dictionary
    """
    # Deep copy to avoid modifying original
    import copy
    config = copy.deepcopy(base_config)

    regime_name = regime_data['regime_name']
    sample_size = regime_data['sample_size']

    print(f"\nGenerating config for Regime {regime_id} ({regime_name})")
    print(f"  Sample size: {sample_size}")

    # Update metadata
    config['version'] = f"auto_regime_{regime_id}_{regime_name}"
    config['profile'] = f"regime_{regime_id}_optimized"
    config['description'] = (
        f"Auto-generated config for {regime_name} regime (ID: {regime_id}). "
        f"Thresholds extracted from {sample_size} winning trades."
    )

    # Force regime override
    if 'regime_classifier' not in config:
        config['regime_classifier'] = {}

    config['regime_classifier']['regime_override'] = {
        regime_id: regime_name,
        "_comment": f"Force all periods to {regime_name} for regime-specific backtesting"
    }

    # Update archetype thresholds
    if 'archetypes' not in config:
        config['archetypes'] = {}

    if 'thresholds' not in config['archetypes']:
        config['archetypes']['thresholds'] = {}

    # Apply global minimum liquidity
    config['archetypes']['thresholds']['min_liquidity'] = regime_data['min_liquidity']

    # Apply fusion thresholds to each archetype pattern
    archetype_patterns = [
        'trap_within_trend',
        'order_block_retest',
        'bos_choch_reversal',
        'wick_trap_moneytaur',
        'failed_rally',
        'long_squeeze'
    ]

    for pattern in archetype_patterns:
        if pattern not in config['archetypes']['thresholds']:
            config['archetypes']['thresholds'][pattern] = {}

        # Set fusion threshold
        config['archetypes']['thresholds'][pattern]['fusion_threshold'] = regime_data['fusion_threshold']

        # Add volume z-score requirement if pattern exists in main archetype config
        if pattern in config['archetypes']:
            if 'volume_z_min' not in config['archetypes']['thresholds'][pattern]:
                config['archetypes']['thresholds'][pattern]['volume_z_min'] = regime_data['volume_z_min']

        # Add funding z-score for short patterns
        if pattern in ['failed_rally', 'long_squeeze']:
            config['archetypes']['thresholds'][pattern]['funding_z_min'] = regime_data['funding_z_min']

    # Add extraction metadata
    config['_threshold_metadata'] = {
        'regime_id': regime_id,
        'regime_name': regime_name,
        'sample_size': sample_size,
        'min_liquidity': regime_data['min_liquidity'],
        'fusion_threshold': regime_data['fusion_threshold'],
        'volume_z_min': regime_data['volume_z_min'],
        'funding_z_min': regime_data['funding_z_min']
    }

    return config


def main():
    """Main execution function."""
    print("=" * 80)
    print("REGIME-SPECIFIC CONFIG GENERATION")
    print("=" * 80)

    # Paths
    base_config_path = PROJECT_ROOT / "configs/mvp/mvp_bull_market_v1.json"
    thresholds_path = PROJECT_ROOT / "configs/auto/hmm_thresholds.json"
    output_dir = PROJECT_ROOT / "configs/auto"

    # Verify files exist
    if not base_config_path.exists():
        print(f"ERROR: Base config not found: {base_config_path}")
        sys.exit(1)

    if not thresholds_path.exists():
        print(f"ERROR: Thresholds file not found: {thresholds_path}")
        print("Run bin/extract_thresholds.py first!")
        sys.exit(1)

    # Load data
    print("\n1. Loading base config and thresholds...")
    base_config = load_json(base_config_path)
    thresholds_data = load_json(thresholds_path)

    print(f"   Base config: {base_config.get('version', 'unknown')}")
    print(f"   Thresholds version: {thresholds_data.get('version', 'unknown')}")
    print(f"   Total trades analyzed: {thresholds_data.get('total_trades', 0)}")
    print(f"   Total winners: {thresholds_data.get('total_winners', 0)}")

    # Generate configs for each regime
    print("\n2. Generating regime-specific configs...")

    regimes = thresholds_data.get('regimes', {})

    if not regimes:
        print("ERROR: No regimes found in thresholds file")
        sys.exit(1)

    generated_files = []

    for regime_id_str, regime_data in regimes.items():
        regime_id = int(regime_id_str)
        regime_name = regime_data['regime_name']

        # Generate config
        regime_config = apply_regime_thresholds(base_config, regime_id, regime_data)

        # Save to file
        output_path = output_dir / f"config_regime_{regime_id}.json"
        save_json(regime_config, output_path)

        generated_files.append({
            'regime_id': regime_id,
            'regime_name': regime_name,
            'path': str(output_path),
            'sample_size': regime_data['sample_size'],
            'fusion_threshold': regime_data['fusion_threshold']
        })

    # Create index file
    print("\n3. Creating config index...")
    index_data = {
        'version': '1.0',
        'description': 'Auto-generated regime-specific configurations',
        'base_config': str(base_config_path.relative_to(PROJECT_ROOT)),
        'thresholds_source': str(thresholds_path.relative_to(PROJECT_ROOT)),
        'generation_date': thresholds_data.get('extraction_date', 'unknown'),
        'configs': generated_files
    }

    index_path = output_dir / "config_index.json"
    save_json(index_data, index_path)

    print("\n" + "=" * 80)
    print("GENERATION COMPLETE")
    print("=" * 80)
    print(f"\nGenerated {len(generated_files)} config files:")
    for cfg in generated_files:
        print(f"  Regime {cfg['regime_id']} ({cfg['regime_name']}): {cfg['path']}")
        print(f"    Sample: {cfg['sample_size']} trades, Fusion: {cfg['fusion_threshold']:.3f}")

    print(f"\nIndex file: {index_path}")

    print("\nNext steps:")
    print("  1. Review generated configs in configs/auto/")
    print("  2. Run backtests with each regime config")
    print("  3. Compare performance across regimes")

    return 0


if __name__ == "__main__":
    sys.exit(main())
