#!/usr/bin/env python3
"""
Config Migration Script - Letter Codes → Canonical Slugs

Migrates archetype configs from legacy letter code format (A, B, C, ...)
to canonical slug format (spring, order_block_retest, wick_trap, ...).

Usage:
    python3 bin/migrate_archetype_configs.py <config_file> [--dry-run] [--backup]
    python3 bin/migrate_archetype_configs.py configs/ --recursive [--dry-run] [--backup]

Examples:
    # Migrate single file with backup
    python3 bin/migrate_archetype_configs.py configs/profile_experimental.json --backup

    # Dry run on all configs in directory
    python3 bin/migrate_archetype_configs.py configs/ --recursive --dry-run

    # Migrate all configs with backup
    python3 bin/migrate_archetype_configs.py configs/ --recursive --backup
"""

import json
import argparse
import sys
from pathlib import Path
from typing import Dict, Any, List, Tuple
from copy import deepcopy


# Letter code → canonical slug mapping (from threshold_policy.py)
LETTER_TO_SLUG = {
    'A': 'spring',
    'B': 'order_block_retest',
    'C': 'wick_trap',
    'D': 'failed_continuation',
    'E': 'volume_exhaustion',
    'F': 'exhaustion_reversal',
    'G': 'liquidity_sweep',
    'H': 'momentum_continuation',
    'K': 'trap_within_trend',
    'L': 'retest_cluster',
    'M': 'confluence_breakout',
}


def migrate_config(config: Dict[str, Any]) -> Tuple[Dict[str, Any], List[str]]:
    """
    Migrate config from letter codes to canonical slugs.

    Args:
        config: Original config dict

    Returns:
        (migrated_config, changes_made)

    Migration process:
        1. Copy thresholds from config['archetypes']['thresholds'][LETTER]
           → config['archetypes'][slug]
        2. Migrate enable flags: enable_A → enable_spring (kept for backward compat)
        3. Keep thresholds/ for backward compatibility (marked with comment)
    """
    migrated = deepcopy(config)
    changes = []

    if 'archetypes' not in migrated:
        return migrated, changes

    archetypes = migrated['archetypes']

    # Step 1: Migrate threshold parameters
    if 'thresholds' in archetypes:
        thresholds = archetypes['thresholds']

        for letter, slug in LETTER_TO_SLUG.items():
            if letter in thresholds:
                # Copy letter code thresholds to canonical location
                letter_params = thresholds[letter]

                if slug not in archetypes:
                    archetypes[slug] = {}

                # Merge (canonical params take precedence if already exist)
                for key, value in letter_params.items():
                    if key not in archetypes[slug]:
                        archetypes[slug][key] = value
                        changes.append(f"Migrated {letter}.{key} → {slug}.{key} = {value}")

                # Add migration marker
                if changes:
                    archetypes[slug]['_migrated_from'] = letter

    # Step 2: Add canonical enable flags (alongside letter flags for backward compat)
    for letter, slug in LETTER_TO_SLUG.items():
        old_key = f'enable_{letter}'
        new_key = f'enable_{slug}'

        if old_key in archetypes:
            if new_key not in archetypes:
                archetypes[new_key] = archetypes[old_key]
                changes.append(f"Added {new_key} = {archetypes[old_key]} (from {old_key})")

    # Step 3: Add migration metadata
    if changes:
        migrated['_migration_info'] = {
            'version': '1.0',
            'script': 'migrate_archetype_configs.py',
            'description': 'Migrated from letter codes to canonical slugs',
            'backward_compatible': True,
            'changes_count': len(changes)
        }

    return migrated, changes


def migrate_file(
    file_path: Path,
    dry_run: bool = False,
    backup: bool = False
) -> bool:
    """
    Migrate a single config file.

    Returns:
        True if changes were made, False otherwise
    """
    print(f"\n{'[DRY RUN] ' if dry_run else ''}Processing: {file_path}")

    try:
        with open(file_path, 'r') as f:
            original = json.load(f)
    except Exception as e:
        print(f"  ❌ Error reading file: {e}")
        return False

    migrated, changes = migrate_config(original)

    if not changes:
        print(f"  ℹ️  No migration needed (already using canonical slugs or no archetype config)")
        return False

    print(f"  ✅ Found {len(changes)} changes:")
    for change in changes[:5]:  # Show first 5
        print(f"     • {change}")
    if len(changes) > 5:
        print(f"     ... and {len(changes) - 5} more")

    if dry_run:
        print(f"  ℹ️  DRY RUN - No files modified")
        return True

    # Create backup if requested
    if backup:
        backup_path = file_path.with_suffix('.json.bak')
        with open(backup_path, 'w') as f:
            json.dump(original, f, indent=2)
        print(f"  📦 Backup created: {backup_path}")

    # Write migrated config
    with open(file_path, 'w') as f:
        json.dump(migrated, f, indent=2)

    print(f"  ✅ Migration complete!")
    return True


def migrate_directory(
    directory: Path,
    recursive: bool = False,
    dry_run: bool = False,
    backup: bool = False
) -> Tuple[int, int]:
    """
    Migrate all JSON files in directory.

    Returns:
        (files_processed, files_modified)
    """
    pattern = '**/*.json' if recursive else '*.json'
    json_files = list(directory.glob(pattern))

    if not json_files:
        print(f"No JSON files found in {directory}")
        return 0, 0

    print(f"Found {len(json_files)} JSON files")

    files_modified = 0
    for file_path in json_files:
        if migrate_file(file_path, dry_run=dry_run, backup=backup):
            files_modified += 1

    return len(json_files), files_modified


def main():
    parser = argparse.ArgumentParser(
        description='Migrate archetype configs from letter codes to canonical slugs',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    parser.add_argument(
        'path',
        type=Path,
        help='Config file or directory to migrate'
    )

    parser.add_argument(
        '--recursive', '-r',
        action='store_true',
        help='Recursively process all JSON files in directory'
    )

    parser.add_argument(
        '--dry-run', '-n',
        action='store_true',
        help='Show what would be changed without modifying files'
    )

    parser.add_argument(
        '--backup', '-b',
        action='store_true',
        help='Create .bak backup before modifying files'
    )

    args = parser.parse_args()

    if not args.path.exists():
        print(f"❌ Error: {args.path} does not exist")
        sys.exit(1)

    print("=" * 70)
    print("Archetype Config Migration: Letter Codes → Canonical Slugs")
    print("=" * 70)

    if args.path.is_file():
        # Migrate single file
        if migrate_file(args.path, dry_run=args.dry_run, backup=args.backup):
            print("\n✅ Migration successful!")
        else:
            print("\nℹ️  No changes needed")
    else:
        # Migrate directory
        processed, modified = migrate_directory(
            args.path,
            recursive=args.recursive,
            dry_run=args.dry_run,
            backup=args.backup
        )

        print("\n" + "=" * 70)
        print(f"Summary: {modified}/{processed} files modified")
        print("=" * 70)

        if args.dry_run:
            print("\nℹ️  DRY RUN - No files were modified")
            print("   Run without --dry-run to apply changes")

    sys.exit(0)


if __name__ == '__main__':
    main()
