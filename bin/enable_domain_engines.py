#!/usr/bin/env python3
"""
Enable Domain Engines - Turn ON Wyckoff, SMC, Temporal, HOB, Fusion, Macro

Modifies archetype configs to activate all 6 domain engines.
Previously archetypes were running with only 20% of their brain active.

This script enables:
1. Wyckoff - Structural event detection (SC, BC, Spring, LPS, etc.)
2. SMC - Smart Money Concepts (Order Blocks, FVG, BOS, CHOCH)
3. Temporal - Time-based confluence (Fibonacci time, Gann windows)
4. HOB - Higher Order Beliefs (meta-patterns)
5. Fusion - Multi-domain signal synthesis
6. Macro - Macro regime context (BTC.D, USDT.D, VIX, DXY)

Usage:
    python bin/enable_domain_engines.py --s1
    python bin/enable_domain_engines.py --s4 --s5
    python bin/enable_domain_engines.py --all
"""

import json
import sys
from pathlib import Path
from datetime import datetime


def enable_all_engines(config_path: str, backup: bool = True, dry_run: bool = False):
    """
    Enable all 6 domain engines in config.

    Args:
        config_path: Path to config JSON file
        backup: Create .backup file before modifying
        dry_run: Print changes without writing
    """
    config_path = Path(config_path)

    if not config_path.exists():
        print(f"❌ Config not found: {config_path}")
        return False

    # Backup original
    if backup and not dry_run:
        backup_path = config_path.with_suffix(f'.json.backup.{datetime.now().strftime("%Y%m%d_%H%M%S")}')
        import shutil
        shutil.copy(config_path, backup_path)
        print(f"📦 Backup created: {backup_path}")

    # Load config
    with open(config_path) as f:
        cfg = json.load(f)

    # Track changes
    changes = []

    # Ensure feature_flags section exists
    if 'feature_flags' not in cfg:
        cfg['feature_flags'] = {}
        changes.append("+ Created 'feature_flags' section")

    # Enable all 6 domain engines
    engine_flags = {
        'enable_wyckoff': True,
        'enable_smc': True,
        'enable_temporal': True,
        'enable_hob': True,
        'enable_fusion': True,
        'enable_macro': True,
    }

    for flag, value in engine_flags.items():
        old_value = cfg['feature_flags'].get(flag, False)
        if old_value != value:
            cfg['feature_flags'][flag] = value
            changes.append(f"+ {flag}: {old_value} → {value}")

    # Enable confluence layers (used by archetypes)
    confluence_flags = {
        'use_temporal_confluence': True,
        'use_fusion_layer': True,
        'use_macro_regime': True,
    }

    for flag, value in confluence_flags.items():
        old_value = cfg['feature_flags'].get(flag, False)
        if old_value != value:
            cfg['feature_flags'][flag] = value
            changes.append(f"+ {flag}: {old_value} → {value}")

    # Add metadata
    if '_domain_engine_metadata' not in cfg:
        cfg['_domain_engine_metadata'] = {}

    cfg['_domain_engine_metadata'].update({
        'enabled_date': datetime.now().isoformat(),
        'enabled_engines': list(engine_flags.keys()),
        'tool': 'bin/enable_domain_engines.py',
        'reason': 'Fix archetype feature coverage from 20% to 100%'
    })
    changes.append("+ Added domain engine metadata")

    # Print changes
    print(f"\n{'='*60}")
    print(f"Config: {config_path}")
    print(f"{'='*60}")

    if not changes:
        print("✓ All domain engines already enabled")
        return True

    print(f"\nChanges ({len(changes)}):")
    for change in changes:
        print(f"  {change}")

    if dry_run:
        print("\n⚠️  DRY RUN - No changes written")
        print("\nResulting feature_flags:")
        print(json.dumps(cfg['feature_flags'], indent=2))
        return True

    # Save
    with open(config_path, 'w') as f:
        json.dump(cfg, f, indent=2)

    print(f"\n✓ Enabled all 6 domain engines: {config_path}")
    print(f"\nActive Engines:")
    print(f"  ✓ Wyckoff    - Structural events (SC, BC, Spring, LPS)")
    print(f"  ✓ SMC        - Order Blocks, FVG, BOS, CHOCH")
    print(f"  ✓ Temporal   - Fibonacci time, Gann windows")
    print(f"  ✓ HOB        - Higher-order belief patterns")
    print(f"  ✓ Fusion     - Multi-domain signal synthesis")
    print(f"  ✓ Macro      - Regime context (BTC.D, USDT.D, VIX, DXY)")

    return True


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Enable all 6 domain engines in archetype configs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Enable for S1 only
  python bin/enable_domain_engines.py --s1

  # Enable for S4 and S5
  python bin/enable_domain_engines.py --s4 --s5

  # Enable for all archetypes
  python bin/enable_domain_engines.py --all

  # Dry run (preview changes)
  python bin/enable_domain_engines.py --s1 --dry-run

  # Enable for custom config
  python bin/enable_domain_engines.py --config configs/my_custom.json
        """
    )

    parser.add_argument('--s1', action='store_true', help='Enable for S1 (Liquidity Vacuum)')
    parser.add_argument('--s4', action='store_true', help='Enable for S4 (Funding Divergence)')
    parser.add_argument('--s5', action='store_true', help='Enable for S5 (Long Squeeze)')
    parser.add_argument('--all', action='store_true', help='Enable for all archetypes (S1, S4, S5)')
    parser.add_argument('--config', type=str, help='Custom config path')
    parser.add_argument('--dry-run', action='store_true', help='Preview changes without writing')
    parser.add_argument('--no-backup', action='store_true', help='Skip creating backup file')

    args = parser.parse_args()

    # Collect configs to process
    configs = []

    if args.config:
        configs.append(args.config)
    else:
        if args.s1 or args.all:
            configs.append('configs/s1_v2_production.json')
        if args.s4 or args.all:
            configs.append('configs/s4_optimized_oos_test.json')
        if args.s5 or args.all:
            configs.append('configs/system_s5_production.json')

    if not configs:
        parser.print_help()
        print("\n❌ Error: Specify --s1, --s4, --s5, --all, or --config")
        sys.exit(1)

    # Process each config
    success_count = 0
    for config in configs:
        success = enable_all_engines(
            config,
            backup=not args.no_backup,
            dry_run=args.dry_run
        )
        if success:
            success_count += 1
        print()  # Blank line between configs

    # Summary
    print(f"{'='*60}")
    print(f"Summary: {success_count}/{len(configs)} configs processed successfully")
    print(f"{'='*60}")

    if args.dry_run:
        print("\n⚠️  DRY RUN MODE - Re-run without --dry-run to apply changes")

    sys.exit(0 if success_count == len(configs) else 1)


if __name__ == "__main__":
    main()
