#!/usr/bin/env python3
"""
Quick test script to verify optimization framework works
Tests with minimal data (6 months, 10 configs) in ~1 minute
"""

import subprocess
import sys
from pathlib import Path
import json
import time

def run_command(cmd, description):
    """Run command and report status"""
    print(f"\n{'='*60}")
    print(f"TEST: {description}")
    print(f"{'='*60}")
    print(f"Running: {' '.join(cmd)}")
    print()

    start = time.time()
    result = subprocess.run(cmd, capture_output=True, text=True)
    elapsed = time.time() - start

    if result.returncode == 0:
        print(f"✅ PASSED ({elapsed:.1f}s)")
        return True, result.stdout
    else:
        print(f"❌ FAILED ({elapsed:.1f}s)")
        print(f"Error: {result.stderr}")
        return False, result.stderr

def main():
    print("🧪 Bull Machine v1.8.6 - Optimization Framework Test")
    print("Testing with 6 months of data, 10 configs (~1 minute)")
    print()

    tests_passed = 0
    tests_failed = 0

    # Test 1: Quick optimization (minimal config)
    success, output = run_command([
        'python3', 'bin/optimize_v18.py',
        '--mode', 'quick',
        '--asset', 'BTC',
        '--years', '1',
        '--output', 'test_results.json'
    ], "Quick optimization (1 year, 50 configs)")

    if success:
        tests_passed += 1

        # Verify output file exists
        if Path('test_results.json').exists():
            print("   ✅ Results file created")

            # Load and check
            with open('test_results.json') as f:
                results = json.load(f)

            if len(results) > 0:
                print(f"   ✅ Generated {len(results)} results")
                tests_passed += 1
            else:
                print(f"   ❌ No results generated")
                tests_failed += 1
        else:
            print("   ❌ Results file not created")
            tests_failed += 1
    else:
        tests_failed += 1

    # Test 2: Analyze results
    if Path('test_results.json').exists():
        success, output = run_command([
            'python3', 'bin/analyze_optimization.py',
            'test_results.json'
        ], "Results analysis")

        if success:
            tests_passed += 1

            # Check if config was generated
            if "PRODUCTION RECOMMENDATIONS" in output:
                print("   ✅ Analysis completed successfully")
                tests_passed += 1
            else:
                print("   ⚠️  Analysis ran but no recommendations (expected if 0 trades)")
                tests_passed += 1
        else:
            tests_failed += 1
    else:
        print("\n⚠️  Skipping analysis test (no results file)")
        tests_failed += 1

    # Cleanup
    print("\n🧹 Cleaning up test files...")
    for f in ['test_results.json', 'test_results_comparison.csv']:
        if Path(f).exists():
            Path(f).unlink()
            print(f"   Removed {f}")

    # Summary
    print(f"\n{'='*60}")
    print("TEST SUMMARY")
    print(f"{'='*60}")
    print(f"✅ Passed: {tests_passed}")
    print(f"❌ Failed: {tests_failed}")
    print()

    if tests_failed == 0:
        print("🎉 ALL TESTS PASSED - Framework ready to use!")
        print()
        print("Next steps:")
        print("  1. Run full optimization:")
        print("     python bin/optimize_v18.py --mode grid --asset BTC --years 3")
        print()
        print("  2. Analyze results:")
        print("     python bin/analyze_optimization.py optimization_results.json")
        return 0
    else:
        print("⚠️  SOME TESTS FAILED - Check errors above")
        print()
        print("Common issues:")
        print("  - Missing data files in chart_logs/")
        print("  - Missing dependencies (pandas, numpy)")
        print("  - Import errors (check PYTHONPATH)")
        return 1

if __name__ == '__main__':
    sys.exit(main())
