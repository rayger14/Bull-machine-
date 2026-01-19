#!/usr/bin/env python3
"""
Multi-Regime Smoke Test Runner

Runs smoke tests across multiple market regimes to validate archetype performance.
"""

import sys
import os
import shutil
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Test periods to evaluate
TEST_REGIMES = [
    {
        'name': 'Q1_2023_Bull_Recovery',
        'start': '2023-01-01',
        'end': '2023-04-01',
        'description': 'Bull market recovery period (post-FTX)',
        'expected_regime': 'bullish'
    },
    {
        'name': '2022_Crisis',
        'start': '2022-06-01',
        'end': '2022-12-31',
        'description': 'Crisis period (Terra Luna, FTX collapse)',
        'expected_regime': 'crisis/bearish'
    },
    {
        'name': '2023H2_Mixed',
        'start': '2023-08-01',
        'end': '2023-12-31',
        'description': 'Mixed/chop period (consolidation and ETF speculation)',
        'expected_regime': 'mixed/choppy'
    },
]

def run_smoke_test_for_regime(regime_config):
    """Run smoke test for a specific regime by modifying constants."""
    print("=" * 80)
    print(f"TESTING REGIME: {regime_config['name']}")
    print("=" * 80)
    print(f"Period: {regime_config['start']} to {regime_config['end']}")
    print(f"Description: {regime_config['description']}")
    print(f"Expected Regime: {regime_config['expected_regime']}")
    print()

    # Import and modify the smoke test module
    import bin.smoke_test_all_archetypes as smoke_test

    # Temporarily override the constants
    original_start = smoke_test.TEST_PERIOD_START
    original_end = smoke_test.TEST_PERIOD_END

    try:
        smoke_test.TEST_PERIOD_START = regime_config['start']
        smoke_test.TEST_PERIOD_END = regime_config['end']

        # Run the smoke test
        exit_code = smoke_test.main()

        # Rename output files to include regime name
        regime_name = regime_config['name']
        if Path('SMOKE_TEST_REPORT.md').exists():
            shutil.copy('SMOKE_TEST_REPORT.md', f'SMOKE_TEST_REPORT_{regime_name}.md')
        if Path('smoke_test_results.json').exists():
            shutil.copy('smoke_test_results.json', f'smoke_test_results_{regime_name}.json')
        if Path('smoke_test_issues.txt').exists():
            shutil.copy('smoke_test_issues.txt', f'smoke_test_issues_{regime_name}.txt')

        print(f"\n✅ {regime_name} test complete - results saved to *_{regime_name}.* files\n")

        return exit_code

    finally:
        # Restore original constants
        smoke_test.TEST_PERIOD_START = original_start
        smoke_test.TEST_PERIOD_END = original_end


def main():
    """Run smoke tests across all regimes."""
    print("=" * 80)
    print("MULTI-REGIME SMOKE TEST SUITE")
    print("=" * 80)
    print(f"\nTesting {len(TEST_REGIMES)} market regimes:")
    for regime in TEST_REGIMES:
        print(f"  - {regime['name']}: {regime['start']} to {regime['end']}")
    print()

    results = {}
    for regime in TEST_REGIMES:
        exit_code = run_smoke_test_for_regime(regime)
        results[regime['name']] = {
            'exit_code': exit_code,
            'status': 'PASS' if exit_code == 0 else 'WARNINGS'
        }

    # Summary
    print("\n" + "=" * 80)
    print("MULTI-REGIME TEST SUMMARY")
    print("=" * 80)
    print()
    for regime_name, result in results.items():
        status_icon = "✅" if result['status'] == 'PASS' else "⚠️"
        print(f"{status_icon} {regime_name}: {result['status']}")
    print()
    print("=" * 80)
    print("MULTI-REGIME SMOKE TEST COMPLETE")
    print("=" * 80)
    print()
    print("Output files generated:")
    print("  - SMOKE_TEST_REPORT_<regime>.md - Detailed reports")
    print("  - smoke_test_results_<regime>.json - Raw data")
    print("  - smoke_test_issues_<regime>.txt - Issues found")
    print()

    return 0


if __name__ == '__main__':
    sys.exit(main())
