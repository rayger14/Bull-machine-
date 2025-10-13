"""
Determinism Test for v1.8 Hybrid Runner

Validates that the hybrid runner produces identical results when run
twice with the same inputs and configuration.

This ensures:
- Reproducible backtest results
- No random/non-deterministic behavior
- Config changes are trackable
"""

import pytest
import json
import os
import sys
from datetime import datetime
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def run_hybrid_test(asset='BTC', start='2025-09-01', end='2025-09-15'):
    """
    Run hybrid runner and return signals.

    Uses a short 2-week period for fast testing from historical data.
    Actual data range: 2025-08-06 to 2025-10-01
    """
    from bin.live.hybrid_runner import HybridRunner

    config_path = 'configs/v18/BTC_conservative.json'

    runner = HybridRunner(
        asset=asset,
        config_path=config_path,
        start_date=start,
        end_date=end
    )

    signals = runner.run()
    return signals


def test_determinism_signal_count():
    """Test that signal count is identical across runs."""
    print("\n🔍 Testing signal count determinism...")

    # Run twice
    signals1 = run_hybrid_test()
    signals2 = run_hybrid_test()

    # Compare counts
    assert len(signals1) == len(signals2), \
        f"Signal count mismatch: {len(signals1)} vs {len(signals2)}"

    print(f"✅ Signal count determinism: {len(signals1)} signals (both runs)")


def test_determinism_signal_content():
    """Test that signal content is identical across runs."""
    print("\n🔍 Testing signal content determinism...")

    # Run twice
    signals1 = run_hybrid_test()
    signals2 = run_hybrid_test()

    # Compare each signal
    for i, (s1, s2) in enumerate(zip(signals1, signals2)):
        # Timestamp
        assert s1['timestamp'] == s2['timestamp'], \
            f"Signal {i}: Timestamp mismatch"

        # Side
        assert s1['side'] == s2['side'], \
            f"Signal {i}: Side mismatch ({s1['side']} vs {s2['side']})"

        # Confidence (allow tiny float precision difference)
        assert abs(s1['confidence'] - s2['confidence']) < 1e-6, \
            f"Signal {i}: Confidence mismatch ({s1['confidence']:.6f} vs {s2['confidence']:.6f})"

        # Action
        assert s1['action'] == s2['action'], \
            f"Signal {i}: Action mismatch"

        # Macro veto status
        assert s1.get('macro_vetoed', False) == s2.get('macro_vetoed', False), \
            f"Signal {i}: Macro veto mismatch"

    print(f"✅ Signal content determinism: All {len(signals1)} signals identical")


def test_determinism_fusion_validation():
    """Test that fusion validation scores are identical across runs."""
    print("\n🔍 Testing fusion validation determinism...")

    # Clean up old logs
    fusion_log = 'results/fusion_validation.jsonl'
    if os.path.exists(fusion_log):
        os.remove(fusion_log)

    # Run once
    _ = run_hybrid_test()

    # Load fusion validation log (run 1)
    with open(fusion_log, 'r') as f:
        fusion1 = [json.loads(line) for line in f]

    # Clean up
    os.remove(fusion_log)

    # Run again
    _ = run_hybrid_test()

    # Load fusion validation log (run 2)
    with open(fusion_log, 'r') as f:
        fusion2 = [json.loads(line) for line in f]

    # Compare
    assert len(fusion1) == len(fusion2), \
        f"Fusion validation count mismatch: {len(fusion1)} vs {len(fusion2)}"

    for i, (f1, f2) in enumerate(zip(fusion1, fusion2)):
        # Timestamp
        assert f1['timestamp'] == f2['timestamp'], \
            f"Fusion {i}: Timestamp mismatch"

        # Scores (allow tiny float precision difference)
        for key in ['fusion_score', 'wyckoff', 'hob', 'momentum', 'smc']:
            assert abs(f1[key] - f2[key]) < 1e-6, \
                f"Fusion {i}: {key} mismatch ({f1[key]:.6f} vs {f2[key]:.6f})"

    print(f"✅ Fusion validation determinism: All {len(fusion1)} validations identical")


def test_determinism_config_hash():
    """Test that config hash tracking works."""
    print("\n🔍 Testing config hash determinism...")

    # Run and check config hash in signals
    signals = run_hybrid_test()

    if not signals:
        pytest.skip("No signals generated in test period")

    # All signals should have same config hash
    config_hashes = [s.get('config_hash') for s in signals]
    unique_hashes = set(config_hashes)

    assert len(unique_hashes) == 1, \
        f"Multiple config hashes found: {unique_hashes}"

    print(f"✅ Config hash determinism: {unique_hashes.pop()}")


def test_determinism_empty_period():
    """Test determinism when no signals generated."""
    print("\n🔍 Testing determinism with empty period...")

    # Use a very short period that likely has no signals
    signals1 = run_hybrid_test(start='2025-09-01', end='2025-09-02')
    signals2 = run_hybrid_test(start='2025-09-01', end='2025-09-02')

    assert len(signals1) == len(signals2), \
        f"Empty period signal count mismatch: {len(signals1)} vs {len(signals2)}"

    print(f"✅ Empty period determinism: {len(signals1)} signals (both runs)")


if __name__ == '__main__':
    """Run determinism tests."""
    print("=" * 60)
    print("Bull Machine v1.8 Determinism Test Suite")
    print("=" * 60)

    # Run all tests
    test_determinism_signal_count()
    test_determinism_signal_content()
    test_determinism_fusion_validation()
    test_determinism_config_hash()
    test_determinism_empty_period()

    print("\n" + "=" * 60)
    print("✅ ALL DETERMINISM TESTS PASSED")
    print("=" * 60)
