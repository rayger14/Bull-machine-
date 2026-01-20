#!/usr/bin/env python3
"""
STEP 7: Apply OPTIMIZED CALIBRATIONS

Loads best Optuna trial parameters and applies them to archetype configs.

This ensures we're testing with optimized thresholds, not vanilla defaults.

Usage:
    python bin/apply_optimized_calibrations.py --s1 --s4 --s5
    python bin/apply_optimized_calibrations.py --archetype s4 --trial 42
"""

import argparse
import json
import sqlite3
from pathlib import Path
import sys
from typing import Dict, Optional

sys.path.insert(0, str(Path(__file__).parent.parent))


ARCHETYPE_CONFIGS = {
    's1': 'configs/s1_v2_production.json',
    's4': 'configs/s4_optimized_oos_test.json',
    's5': 'configs/s5_production.json'
}

OPTUNA_DATABASES = {
    's1': 'optuna_s1_liquidity_vacuum.db',
    's4': 'optuna_s4_funding_divergence.db',
    's5': 'optuna_s5_failed_rally.db'
}


def get_best_trial_params(db_path: Path, study_name: str = None) -> Optional[Dict]:
    """
    Load best trial parameters from Optuna database.

    Returns:
        {
            'trial_id': int,
            'value': float,  # objective value (PF or Sharpe)
            'params': {...}  # trial parameters
        }
    """
    if not db_path.exists():
        print(f"Warning: Optuna database not found: {db_path}")
        return None

    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        # Get best trial
        query = """
            SELECT trial_id, value, params_json
            FROM trials
            WHERE state = 'COMPLETE'
            ORDER BY value DESC
            LIMIT 1
        """

        cursor.execute(query)
        row = cursor.fetchone()

        conn.close()

        if not row:
            return None

        trial_id, value, params_json = row
        params = json.loads(params_json) if params_json else {}

        return {
            'trial_id': trial_id,
            'value': value,
            'params': params
        }

    except Exception as e:
        print(f"Error loading from {db_path}: {e}")
        return None


def apply_params_to_config(
    config_path: Path,
    params: Dict,
    trial_id: int
) -> bool:
    """
    Apply optimized parameters to config file.

    Returns:
        True if successful
    """
    if not config_path.exists():
        print(f"Error: Config not found: {config_path}")
        return False

    try:
        # Load config
        with open(config_path) as f:
            config = json.load(f)

        # Update parameters
        if 'threshold_policy' not in config:
            config['threshold_policy'] = {}

        # Apply all params
        for param_name, param_value in params.items():
            # Store at top level and in threshold_policy
            config[param_name] = param_value

            if param_name.endswith('_threshold') or param_name.endswith('_min'):
                config['threshold_policy'][param_name] = param_value

        # Mark as optimized
        config['optimized'] = True
        config['optuna_trial_id'] = trial_id

        # Save updated config
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)

        print(f"✓ Applied trial {trial_id} params to {config_path.name}")
        return True

    except Exception as e:
        print(f"Error updating config: {e}")
        return False


def verify_optimized_flag(config_path: Path) -> bool:
    """Check if config has optimized flag set."""
    if not config_path.exists():
        return False

    try:
        with open(config_path) as f:
            config = json.load(f)

        return config.get('optimized', False)

    except:
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Apply optimized calibrations to archetype configs"
    )
    parser.add_argument('--s1', action='store_true', help='Apply to S1')
    parser.add_argument('--s4', action='store_true', help='Apply to S4')
    parser.add_argument('--s5', action='store_true', help='Apply to S5')
    parser.add_argument(
        '--archetype',
        type=str,
        help='Apply to specific archetype (s1/s4/s5)'
    )
    parser.add_argument(
        '--trial',
        type=int,
        help='Use specific trial ID (default: best trial)'
    )

    args = parser.parse_args()

    # Determine which archetypes to update
    archetypes = []

    if args.archetype:
        archetypes.append(args.archetype)
    else:
        if args.s1:
            archetypes.append('s1')
        if args.s4:
            archetypes.append('s4')
        if args.s5:
            archetypes.append('s5')

    if not archetypes:
        print("No archetypes specified. Use --s1, --s4, --s5, or --archetype")
        return 1

    print("\n" + "="*60)
    print("APPLY OPTIMIZED CALIBRATIONS")
    print("="*60)

    success_count = 0
    fail_count = 0

    for archetype in archetypes:
        print(f"\n{archetype.upper()}:")

        config_path = Path(ARCHETYPE_CONFIGS.get(archetype, f'configs/{archetype}.json'))
        db_path = Path(OPTUNA_DATABASES.get(archetype, f'optuna_{archetype}.db'))

        # Load best trial
        best_trial = get_best_trial_params(db_path)

        if not best_trial:
            print(f"  ✗ No optimized parameters found")

            # Check if already optimized
            if verify_optimized_flag(config_path):
                print(f"  ℹ Config already has 'optimized: true' flag")
                success_count += 1
            else:
                fail_count += 1

            continue

        print(f"  Best trial: {best_trial['trial_id']}")
        print(f"  Objective value: {best_trial['value']:.3f}")
        print(f"  Parameters: {len(best_trial['params'])}")

        # Apply parameters
        if apply_params_to_config(config_path, best_trial['params'], best_trial['trial_id']):
            success_count += 1
        else:
            fail_count += 1

    # Summary
    print("\n" + "="*60)

    if fail_count == 0:
        print("\033[0;32m✓ PASS\033[0m: All configs updated with optimized calibrations")
        print(f"\nUpdated: {success_count}/{len(archetypes)}")
        return 0
    else:
        print("\033[0;31m✗ FAIL\033[0m: Some configs could not be updated")
        print(f"\nUpdated: {success_count}/{len(archetypes)}")
        print(f"Failed: {fail_count}/{len(archetypes)}")
        return 1


if __name__ == '__main__':
    exit(main())
