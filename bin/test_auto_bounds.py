#!/usr/bin/env python3
"""
Test auto-bounds feature - demonstrates data-derived parameter bounds.

This shows how auto-bounds prevents the critical issue where optimization
ranges don't match actual data, causing zero variance.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from engine.optimization.auto_bounds import (
    compute_bounds_from_file,
    TRAP_PARAM_MAP
)


def main():
    print("=" * 70)
    print("AUTO-BOUNDS TEST - Data-Derived Parameter Ranges")
    print("=" * 70)
    print()
    print("This feature prevents wasted optimization runs by ensuring")
    print("parameter ranges match actual data ranges.")
    print()

    cache_file = 'data/cached/btc_features_2022-01-01_2024-12-31_cached.parquet'

    # Compute bounds using predefined trap parameter mapping
    bounds = compute_bounds_from_file(
        cache_file,
        TRAP_PARAM_MAP,
        print_report=True
    )

    print()
    print("✅ SUCCESS: Auto-bounds computed successfully")
    print()
    print("Integration example:")
    print("-" * 70)
    print("""
# Before optimization, compute bounds once:
from engine.optimization.auto_bounds import compute_bounds_from_file, TRAP_PARAM_MAP

bounds = compute_bounds_from_file(cache_path, TRAP_PARAM_MAP)

# In your objective function:
def objective(trial):
    trap_params = {
        'quality_threshold': trial.suggest_float(
            'trap_quality_threshold',
            *bounds['quality_threshold'],  # Use computed bounds!
            step=0.02
        ),
        'adx_threshold': trial.suggest_float(
            'trap_adx_threshold',
            *bounds['adx_threshold'],  # Use computed bounds!
            step=5.0
        ),
    }
    """)
    print("-" * 70)

    return 0


if __name__ == '__main__':
    sys.exit(main())
