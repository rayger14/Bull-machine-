#!/usr/bin/env python3
"""
Quick Archetype Optimization Test

Fast smoke test with minimal trials (5 per group) to validate setup.
Useful for testing changes before full optimization run.

Usage:
    python bin/quick_archetype_test.py
"""

import subprocess
import sys
from pathlib import Path


def main():
    """Run quick test with minimal trials."""
    print("=" * 80)
    print("QUICK ARCHETYPE OPTIMIZATION TEST")
    print("=" * 80)
    print("\nRunning 5 trials per group (4 groups = 20 total trials)")
    print("Estimated time: ~20-30 minutes\n")

    # Run with minimal trials
    cmd = [
        "python3",
        "bin/optuna_parallel_archetypes.py",
        "--trials", "5",
        "--base-config", "configs/profile_production.json",
        "--storage", "optuna_test.db",
        "--output", "configs/test_optimized_archetypes.json"
    ]

    try:
        result = subprocess.run(cmd, check=True)

        if result.returncode == 0:
            print("\n" + "=" * 80)
            print("✅ TEST PASSED - Parallel optimization working correctly!")
            print("=" * 80)
            print("\nNext steps:")
            print("1. Review results: python bin/analyze_archetype_optimization.py --storage optuna_test.db")
            print("2. Run full optimization: python bin/optuna_parallel_archetypes.py --trials 100")
            return 0
        else:
            print("\n❌ TEST FAILED")
            return 1

    except subprocess.CalledProcessError as e:
        print(f"\n❌ TEST FAILED: {e}")
        return 1
    except KeyboardInterrupt:
        print("\n⚠️  Test interrupted by user")
        return 130


if __name__ == '__main__':
    exit(main())
