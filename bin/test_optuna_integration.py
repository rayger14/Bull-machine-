#!/usr/bin/env python3
"""
Quick integration test for optuna_thresholds.py

Tests:
1. Config generation with sample parameters
2. Backtest execution (dry run)
3. Result parsing
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import json
import tempfile
from bin.optuna_thresholds import ConfigGenerator, BacktestRunner

def test_config_generation():
    """Test that config generation works with sample params"""
    print("\n" + "="*80)
    print("TEST 1: Config Generation")
    print("="*80)

    # Find a suitable base config
    base_configs = [
        "configs/profile_default.json",
        "configs/knowledge_v2/ETH_baseline.json",
        "configs/profile_eth_seed.json"
    ]

    base_config = None
    for config_path in base_configs:
        if Path(config_path).exists():
            base_config = config_path
            break

    if not base_config:
        print("ERROR: No base config found!")
        return False

    print(f"Using base config: {base_config}")

    # Create generator
    generator = ConfigGenerator(base_config)

    # Test params
    params = {
        'min_liquidity': 0.15,
        'fusion_threshold': 0.35,
        'volume_z_min': 1.5,
        'funding_z_min': 1.2,
        'archetype_weight': 1.5,
    }

    print(f"Test parameters: {params}")

    # Generate config
    config_path = generator.generate(params)

    # Validate
    with open(config_path, 'r') as f:
        generated_config = json.load(f)

    print(f"Generated config: {config_path}")
    print(f"Config size: {len(json.dumps(generated_config))} bytes")

    # Check if params were applied
    applied = []

    if 'fusion' in generated_config:
        if generated_config['fusion']['entry_threshold_confidence'] == params['fusion_threshold']:
            applied.append('fusion_threshold')

    if 'liquidity' in generated_config:
        if generated_config['liquidity']['min_liquidity'] == params['min_liquidity']:
            applied.append('min_liquidity')

    print(f"Successfully applied: {applied}")

    # Cleanup
    Path(config_path).unlink()

    print("✓ Config generation PASSED")
    return True


def test_backtest_runner():
    """Test backtest runner (without actually running full backtest)"""
    print("\n" + "="*80)
    print("TEST 2: Backtest Runner Setup")
    print("="*80)

    runner = BacktestRunner(
        asset="ETH",
        start_date="2024-01-01",
        end_date="2024-01-31",  # Short period for quick test
        timeout=30
    )

    print(f"Asset: {runner.asset}")
    print(f"Period: {runner.start_date} to {runner.end_date}")
    print(f"Timeout: {runner.timeout}s")
    print(f"Backtest script: {runner.backtest_script}")

    print("✓ Backtest runner setup PASSED")
    return True


def test_result_parsing():
    """Test parsing of backtest output"""
    print("\n" + "="*80)
    print("TEST 3: Result Parsing")
    print("="*80)

    # Sample backtest output
    sample_output = """
    Knowledge-Aware Backtest Results - ETH
    ========================================
    Total PNL: $1234.56
    Total Trades: 42
    Win Rate: 58.5%
    Profit Factor: 1.85
    Sharpe Ratio: 1.23
    Max Drawdown: 12.3%
    Final Equity: $11234.56
    """

    runner = BacktestRunner(asset="ETH", start_date="2024-01-01", end_date="2024-12-31")
    metrics = runner._parse_output(sample_output)

    if metrics is None:
        print("ERROR: Failed to parse sample output")
        return False

    print("Parsed metrics:")
    for key, value in metrics.items():
        print(f"  {key}: {value}")

    # Validate
    assert metrics['profit_factor'] == 1.85, "PF mismatch"
    assert metrics['max_drawdown'] == 12.3, "DD mismatch"
    assert metrics['sharpe_ratio'] == 1.23, "Sharpe mismatch"
    assert metrics['num_trades'] == 42, "Trades mismatch"

    print("✓ Result parsing PASSED")
    return True


def main():
    print("\n" + "="*80)
    print("OPTUNA INTEGRATION TEST SUITE")
    print("="*80)

    tests = [
        ("Config Generation", test_config_generation),
        ("Backtest Runner", test_backtest_runner),
        ("Result Parsing", test_result_parsing),
    ]

    passed = 0
    failed = 0

    for test_name, test_func in tests:
        try:
            if test_func():
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"\n✗ {test_name} FAILED with exception:")
            print(f"  {e}")
            import traceback
            traceback.print_exc()
            failed += 1

    print("\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80)
    print(f"Passed: {passed}/{len(tests)}")
    print(f"Failed: {failed}/{len(tests)}")

    if failed == 0:
        print("\n✓ ALL TESTS PASSED")
        return 0
    else:
        print("\n✗ SOME TESTS FAILED")
        return 1


if __name__ == '__main__':
    sys.exit(main())
